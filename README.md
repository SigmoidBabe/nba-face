# NBA face CUDA runtime

Standalone external runtime for `face.recognition`, registered as `nba.face`.
The package imports no host application modules. The host supplies camera frames
and owns ingestion, CUDA IPC, session scheduling and output delivery.

## Installation and configuration

Install into the host environment with CUDA 12.x, CuPy 13.x and TensorRT
8.6–10.x matching your serialized engines. The CuPy dependency matches the host's
`requirements-gpu.txt` (`cupy-cuda12x`). GPU kernels are compiled lazily with NVRTC
on first use; the CUDA runtime/compiler libraries must be available:

```bash
python -m pip install -e ./packages/nba-face
export NBA_FACE_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DATABASE'
CV_DEMO_RUNTIME_REGISTRY=packages/nba-face/registry.yaml python -m app.cv_demo.main
```

Set engine paths and GPU device in `registry.yaml` first. The GPU must match the
host ingestion GPU. Installation does not build or download engines. For an
existing registry, add this runtime's YAML list item. Send `session.example.json`
through the host's `RuntimeControl.StartRuntime` API. This package implements
recognition sessions and the `face.registration` gRPC API with `register_face`
and `delete_face` methods.

For deployment from a separate Git repository, publish this directory as that
repository's root and replace `package.entry_point` with your host's
`package.repository` and pinned `package.revision` configuration.

## Face registration API

Reinstall this package after updating it so the host discovers its
`cv_demo.grpc_apis` entry point. Enable `grpc_apis: [face.registration]` in the
runtime registry, as shown in the supplied `registry.yaml`. API names must be
unique; enable only one package providing `face.registration`.

Call `cv_demo.api.v1.PackageAPI/Invoke` with:

```json
{
  "api": "face.registration",
  "method": "register_face",
  "payload": {
    "employee_id": 42,
    "image_base64": "<base64-encoded JPEG or PNG bytes>",
    "crop_type": "loose"
  }
}
```

Send raw base64 without a data URL prefix. `employee_id` must be a signed 64-bit
integer or an integer string; arbitrary employee names are unsupported by the
PostgreSQL `BIGINT` schema. Send large identifiers as strings to avoid protobuf
Struct number precision loss. The image must contain exactly one face passing
the configured detector filters. `crop_type` is optional, ignores case and
surrounding whitespace, and accepts `full`, `tight`, or `loose` (default).
`full` embeds and saves the full image, `tight` uses the detected box, and
`loose` expands the box by 25% on each side, clipped to the image bounds.

The API decodes the uploaded image on the CPU, uploads it for the existing
TensorRT detector and ArcFace models, normalizes a 512D embedding, and inserts
an identity in the configured PostgreSQL `public.identity` table. The stored
pose/type is `front`; no pose classification is performed. Every call inserts
a new identity, allowing multiple images per employee. Models are loaded lazily
and independently from runtime inference. Calls share a lock to protect the API's
models and database connection.

Example response (the package object is returned directly):

```json
{
  "identity_id": 123,
  "employee_id": 42,
  "image_url": "file:///var/lib/cv-demo/nba-face/images/550e8400-e29b-41d4-a716-446655440000.jpg"
}
```

Set `config.registration_image_dir` to a writable persistent directory (default:
`face-registration` relative to the server working directory). A UUID-named JPEG
is saved there, and its absolute `file://` URI is stored in `image_url`. This is
a host file reference, not an HTTP download URL. A failed database insert
attempts to remove the newly saved image. No S3 upload is performed.

Soft-delete an individual registration by its positive PostgreSQL identity ID:

```json
{
  "api": "face.registration",
  "method": "delete_face",
  "payload": {"identity_id": 123}
}
```

The response is `{"identity_id":123,"deleted":true}` when an active row was
deleted, or `{"identity_id":123,"deleted":false}` if it was absent or already
deleted. This sets `deleted_at` and `updated_at`; it preserves the row and JPEG.
Subsequent recognition queries exclude that identity. Other registrations for
the same employee remain active. Deletion does not load GPU models.

Invalid arguments raise `ValueError` (`INVALID_ARGUMENT`), unsupported methods
raise `NotImplementedError` (`UNIMPLEMENTED`), and database, GPU or storage
failures propagate as `INTERNAL` through the host. See the shared
[Invoke request](../../docs/grpc_documentation/package_api_invoke_request.md) and
[Invoke response](../../docs/grpc_documentation/package_api_invoke_response.md)
documentation.

## GPU pipeline and engine contracts

Input is a read-only HWC BGR uint8 CuPy array or another provider of the host's
CUDA array interface. CPU frames are rejected. CuPy arrays are used directly;
other providers are wrapped with `cupy.asarray` and must retain their original
device pointer, following CuPy's
[CUDA array interoperability](https://docs.cupy.dev/en/v13.5.1/user_guide/interoperability.html).
Strided frames are supported without a contiguous image copy. Fused CUDA kernels
read the source frame to perform color conversion, bilinear resizing,
letterboxing, cropping and normalization. NMS and embedding normalization also
run on the GPU. TensorRT binds CuPy device pointers on the current CuPy stream,
following NVIDIA's
[Python tensor-address API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html).
There is no PyCUDA context, OpenCV preprocessing, NumPy frame conversion or CPU
inference fallback. Producers must finish writing the frame before handing it
to the runtime, as required by the host CUDA IPC protocol.

Each engine must have one LINEAR device input and one LINEAR device output with
FP16 or FP32 I/O. Profile 0 is used. Static batches are padded with the last item;
dynamic batches are padded to profile minimum and split at profile maximum.
Padded results are discarded. TensorRT execution is serialized and its stream
is synchronized before buffers or execution context bindings can be reused.

* Detector: NCHW RGB input `[B,3,H,W]`, normalized to `[0,1]`, with letterboxing
  using padding value 114. `detector_size` is `[H,W]`. Output is raw detection-only
  YOLOv8 `[B,4+C,N]` (`detector_layout: BCN`) or `[B,N,4+C]` (`BNC`), containing
  pixel `cx,cy,w,h` followed by `C` class scores. Export without built-in NMS.
  Class names are read from the `.txt` file beside the detector engine, using
  the same filename stem and line-order class IDs as `packages/person-inference`.
  For `/opt/models/face-yolov8.engine`, supply `/opt/models/face-yolov8.txt`.
  The decoder picks each anchor's highest-scoring class, then keeps only the
  class named `face` (override with `face_label` in registry config). A box whose
  best class is another label is excluded even if its face score is high.
  Class selection and confidence filtering stay on the GPU. YOLOv5 objectness outputs,
  SCRFD/RetinaFace outputs, landmark outputs and end-to-end NMS engines require a
  different decoder and are rejected. Candidates are capped at 3000 before GPU
  NMS and `max_faces` afterward, per image.
* ArcFace: NCHW RGB `[B,3,112,112]`, normalized as `pixel / 127.5 - 1`, with a
  512D embedding per item. Crops expand detected boxes by `crop_margin` on each
  side (default 20%), clipped to the frame. GPU ROI sampling uses bilinear
  interpolation. Embeddings are L2-normalized on the GPU; nonfinite and zero
  embeddings are omitted. No landmark alignment or pose classification model is
  used. `pose` in output is the matched stored identity's `type`.

For a detector trained with class IDs `0=person`, `1=face`, `2=helmet`, its `.txt`
file must contain:

```text
person
face
helmet
```

Class order must match training/export order. Blank lines within the class list,
duplicate names, a missing face label, and output channel counts that disagree
with the label file are rejected. Trailing empty lines are ignored. The default
`face_label: face` resolves class ID 1 in this example; no numeric class ID needs
to be configured. Single-class models also require a `.txt` file containing
their class name.

The explicit device-to-host download is a compact array of final embeddings and
box/confidence/frame-index metadata. GPU selection/compaction can also synchronize
small counts with the host. PostgreSQL is a CPU/network service and
needs those vectors serialized for SQL; frame pixels, face crops and model
input tensors never leave GPU memory. This pipeline therefore avoids GPU-to-CPU
image copies; it does not claim zero host transfers for database search.

Existing registered embeddings must come from the same ArcFace model and
compatible preprocessing. Otherwise re-enroll the gallery. Thresholds must be
calibrated against your deployed engines and enrollment images.

## Batch APIs and sessions

`FaceRuntime.infer_batch(images)` accepts an ordered list of CUDA HWC images and
returns one candidate list per image, after database search. Detection batches
up to configured `batch_size`; ArcFace batches face crops across those frames
according to its engine profile. Differently sized source frames are supported.

`FaceRuntime.process_batch(frames, session_ids_by_frame)` accepts host frame
dictionaries and one corresponding eligible session ID list per frame. It returns
an ordered list of `{session_id: {snapshot, events}}` results. Frames without
eligible sessions skip inference. Callers should pass frames chronologically
within each session. Calls and session lifecycle changes are protected by a lock.

The existing host invokes `process_frame(frame, session_ids)`, which delegates to
`process_batch`. It shares detection and search among eligible sessions on that
frame and batches all detected faces for ArcFace. Cross-camera frame batching
requires a host that invokes `process_batch`; the package does not buffer frames
past the host call or change host scheduling.

Sessions support full-frame recognition with per-session `similarity_threshold`
overriding the runtime default (0.6). Higher cosine similarity is better. Matches
below threshold have null `identity_id`, `employee_id`, `image_url`, and `pose`;
their nearest similarity is retained. Empty galleries return unmatched faces.
Periodic mode emits snapshots at `output.interval` seconds, including empty
scenes. Event mode emits when a known identity appears or reappears after absence.
Snapshots are returned every processed frame. Zones are rejected.

Snapshots contain `faces_count` and `faces`, each with `box` (source-frame xyxy),
`confidence`, `face_area`, `face_area_ratio`, `estimated_distance_cm`, `identity_id`,
`employee_id`, `image_url`, `similarity`, and `pose`. The distance estimate uses
the legacy worker's area heuristic (60 cm at a frame-area ratio of 0.065).
Events contain metadata only; image uploads, liveness and tracking are not
implemented.

## PostgreSQL face search

Uses the existing schema from `internal/repo/FaceRecognitionDB.py` and the worker
under `internal/ai_runtime/face_recognition_worker`. The database must already
have pgvector and `public.identity` with `id`, `embedding vector(512)`,
`employee_id`, `image_url`, `type`, nullable `deleted_at`, and `updated_at`.
No tables or indexes are created by this package. Registration inserts records and deletion updates
their soft-delete timestamps through the API described above.

Each inference chunk sends embeddings in one parameterized query using
`unnest(... WITH ORDINALITY)` and a lateral nearest-neighbor lookup per vector:

```sql
SELECT i.id AS identity_id, i.employee_id, i.image_url, i.type AS pose,
       1 - (i.embedding <=> query_embedding::vector) AS similarity
FROM public.identity i
WHERE i.deleted_at IS NULL AND i.embedding IS NOT NULL
ORDER BY i.embedding <=> query_embedding::vector
LIMIT 1;
```

This preserves cosine similarity and soft-deletion behavior from the legacy
search. It can use the existing HNSW `vector_cosine_ops` index. Newly committed
registrations are visible on subsequent queries without a gallery reload.

Supply a DSN through `db.dsn_env` (default `NBA_FACE_DATABASE_URL`), or configure
legacy `db.name`, `username`, `password`, `host`, and `port` fields. Search uses a
lazy persistent autocommit connection, a 10-second connection timeout and
`db.statement_timeout_ms` (default 5000). Database failures propagate to the
host's runtime failure handling, rather than reporting everyone as unknown.
Closed or broken connections are reopened on the next invocation. Runtime close
releases the connection and model contexts.
