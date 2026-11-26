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
CV_DEMO_RUNTIME_REGISTRY=packages/nba-face/registry.yaml python -m app.cv_demo.main
```

Set `config.model.face_detection.engine`, `config.model.face_extraction.engine`,
`config.registration_image_dir`, and `config.db` in `registry.yaml`; these settings
are required and have no defaults. All other config settings are optional and
use the defaults shown in the registry comments when omitted. The GPU defaults
to device 0; override `config.device` to match the host ingestion GPU.
Installation does not build or download engines. For an
existing registry, add this runtime's YAML list item. Send `session.example.json`
through the host's `RuntimeControl.StartRuntime` API. This package implements
recognition sessions and the `face.registration` gRPC API with `register_face`
and `delete_face` methods.

For deployment from a separate Git repository, publish this directory as that
repository's root and replace `package.entry_point` with your host's
`package.repository` and pinned `package.revision` configuration.

## Core event images and recordings

The supplied registry enables `recording: true` and `image_save: true` on the
`face.recognition` capability. These independent YAML boolean switches belong
alongside `id` and `output_modes`, outside package `config`. Set either to
`false` to disable that output and restart the host after editing the registry.
Session requests and package results cannot override the loaded switches.

The runtime returns events on their trigger frame, with one `{snapshot, events}`
result per submitted session. Its `faces[].box` values are source-image xyxy
coordinates; the core uses these snapshots to draw yellow face boxes and
identity labels on saved full-frame JPEGs and video. Snapshots from all sessions
on each camera frame contribute overlays, including sessions emitting no event.
The runtime does not retain borrowed CUDA frames or buffer evidence; GPU work
finishes before the host can reuse frame storage. Enrollment images saved by
`face.registration` remain separate durable identity data.

Every emitted event requests the enabled media outputs, including periodic
events and empty periodic snapshots. Choose `output.interval` with this storage
cost in mind. Continuous snapshots still return on every processed frame, even
when no event is due. Do not add media paths, image arrays, video bytes, or host
envelope fields to event `data`.

Configure storage and limits in the host application configuration, for example:

```yaml
event_jpeg_quality: 50
event_images:
  directory: recordings/images
event_recording:
  enabled: true
  directory: recordings/events
  pre_frames: 5
  post_frames: 5
  max_clip_frames: 150
  max_clip_bytes: 134217728
  max_events_per_clip: 256
  queue_size: 2
  idle_timeout_seconds: 5
  fallback_fps: 25
```

`event_recording.enabled` is the global video off switch. Recording uses shared
GPU storage per camera and requires working CUDA/CuPy plus OpenCV `mp4v` encoding.
The default clip includes five ingested frames before and after the trigger;
these are ingestion frames, before inference FPS filtering. Provision GPU memory
for camera resolutions and concurrent clips, and manage disk retention separately.

Metadata snapshots are published immediately through `MetadataService.Subscribe`.
`RuntimeControl.StartRuntime` inference events arrive after enabled media work
finishes, retaining their trigger timestamps and event IDs. The host adds
`image_path` and `video_path` to its envelope, outside `event_payload`. Paths are
absolute server-side file references; remote clients need shared storage access.
Events sharing a video remain distinct: identify events by `event_id`.

Treat events as valid when a media path is null. Inspect `image_error` and
`video_error` for failures, and `video_truncation_reason`, `video_pre_frames`,
and `video_post_frames` for partial clips. Image-only events do not wait for
video. No protobuf regeneration is required. See the
[runtime update instructions](docs/runtime_package_update_instruction.md) for
the complete envelope and recording resource behavior.

## Model updates and worker lifecycle

For OTA, send `.tar.gz` releases containing `manifest.json` and a self-contained
ONNX model to the host's `ModelService.UpdateModelPath`. The host stops the
managed GPU pipeline, builds TensorRT engines, updates registry paths, and
creates fresh runtime and API instances. This package loads the generated paths
from `config.model.face_detection.engine` and
`config.model.face_extraction.engine`. Detector class labels are read from the
`classes` array in `config.model.face_detection.manifest`, which OTA sets to the
staged manifest path. If omitted, the path defaults to `manifest.json` beside
the detector engine. See [model OTA](docs/model-ota.md).

Registration API instances run independently in the supervised GPU worker.
Closing an instance releases its model contexts and database client socket. A
separate local database owner keeps the collection open while clients use it.
Keep the Zvec collection and `registration_image_dir` outside model release
directories so they survive replacement. Session appearance state and periodic
timers restart with new instances; the host restores enabled sessions from its
database. Clients must keep runtime streams open through the nonterminal
`PAUSED` state and handle startup states when inference resumes.

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
Zvec `INT64` field. Send large identifiers as strings to avoid protobuf
Struct number precision loss. The image must contain exactly one face passing
the configured detector filters. `crop_type` is optional, ignores case and
surrounding whitespace, and accepts `full`, `tight`, or `loose` (default).
`full` embeds and saves the full image, `tight` uses the detected box, and
`loose` expands the box by 25% on each side, clipped to the image bounds.

The API decodes the uploaded image on the CPU, uploads it for the existing
TensorRT detector and ArcFace models, normalizes a 512D embedding, and inserts
an identity in the configured Zvec collection. The stored
pose/type is `front`; no pose classification is performed. Every call inserts
a new identity, allowing multiple images per employee. Models are loaded lazily
and independently from runtime inference. Calls share a lock to protect the API's
models and database operations.

Example response (the package object is returned directly):

```json
{
  "identity_id": 123,
  "employee_id": 42,
  "image_url": "/var/lib/cv-demo/nba-face/images/550e8400-e29b-41d4-a716-446655440000.jpg"
}
```

Set `config.registration_image_dir` to a writable persistent directory. This
setting is required and has no default. A UUID-named JPEG
is saved there, and its path is stored in `image_url` without a `file://` prefix.
Relative directories are resolved from the current working directory and remain
relative in `image_url`. This is a host file reference, not an HTTP download URL. A failed database insert
attempts to remove the newly saved image. No S3 upload is performed.

Soft-delete an individual registration by its positive identity ID:

```json
{
  "api": "face.registration",
  "method": "delete_face",
  "payload": {"identity_id": 123}
}
```

The response is `{"identity_id":123,"deleted":true}` when an active document was
deleted, or `{"identity_id":123,"deleted":false}` if it was absent or already
deleted. This sets `deleted_at` and `updated_at`; it preserves the document and JPEG.
Subsequent recognition queries exclude that identity. Other registrations for
the same employee remain active. Deletion does not load GPU models.

Invalid arguments raise `ValueError` (`INVALID_ARGUMENT`), unsupported methods
raise `NotImplementedError` (`UNIMPLEMENTED`), and database, GPU or storage
failures propagate as `INTERNAL` through the host. See the shared
[Invoke request](../../docs/grpc_documentation/package_api_invoke_request.md) and
[Invoke response](../../docs/grpc_documentation/package_api_invoke_response.md)
documentation.

## TOPS calculation API

Enable `face.performance` in `grpc_apis` and reinstall the package so the host
discovers its entry point. Set both per-model operation counts in `registry.yaml`
to match your deployed models and input sizes. The supplied zero values must be
replaced before calling this API. For example (illustrative counts only):

```yaml
config:
  model:
    face_detection:
      engine: /opt/models/face-yolov8.engine
      operations_per_inference: 600000000
    face_extraction:
      engine: /opt/models/arcface.engine
      operations_per_inference: 400000000
```

`model.face_detection` supplies the detector count for one frame;
`model.face_extraction` supplies the ArcFace count for one face. Use the
same counting convention for both models, such as two operations per
multiply-add. The API assumes **one face per frame** and includes both models.
It performs arithmetic without loading engines or running inference.

Call `cv_demo.api.v1.PackageAPI/Invoke` with:

```json
{
  "api": "face.performance",
  "method": "calculate_tops",
  "payload": {
    "cv_capability": "face.recognition",
    "fps": 30,
    "batchsize": 8
  }
}
```

`cv_capability` is required and must be `face.recognition`, the runtime's
supported capability. This includes face detection and face extraction.
Missing or unsupported capabilities raise `ValueError` (`INVALID_ARGUMENT`).

`fps` means **inference batch executions per second**, and `batchsize` is the
number of frames in each batch. If your measured FPS already counts individual
frames, divide it by the batch size before supplying `fps`.

```text
operations_per_inference = detector_operations_per_inference + arcface_operations_per_inference
tops = operations_per_inference * fps * batchsize / 1e12
```

With the illustrative counts above, the response is:

```json
{
  "cv_capability": "face.recognition",
  "fps": 30.0,
  "batchsize": 8,
  "faces_per_frame": 1,
  "detector_operations_per_inference": 600000000.0,
  "arcface_operations_per_inference": 400000000.0,
  "operations_per_inference": 1000000000.0,
  "detector_tops": 0.144,
  "arcface_tops": 0.096,
  "tops": 0.24
}
```

This is estimated model workload throughput. Preprocessing, NMS, database search,
and engine batch padding are excluded. The request accepts finite positive JSON
numbers; `batchsize` must also be an integer at most `2**53 - 1`. Both configured
operation counts must be finite positive numbers and have no defaults. Missing
or invalid values, or overflowing results, raise `ValueError`
(`INVALID_ARGUMENT`). Unsupported methods raise `NotImplementedError`
(`UNIMPLEMENTED`).

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
  Class names are read in array order from the detector manifest's `classes`
  field. Set `config.model.face_detection.manifest` to its path, or supply
  `manifest.json` beside the detector engine.
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

For a detector trained with class IDs `0=person`, `1=face`, `2=helmet`, its
`manifest.json` must include:

```json
{
  "model_type": "detection",
  "classes": ["person", "face", "helmet"]
}
```

Class order must match training/export order. A missing or empty class array,
non-string or blank names, duplicate names, a missing face label, and output
channel counts that disagree with the manifest are rejected. Leading and trailing
whitespace in names is stripped. The default
`face_label: face` resolves class ID 1 in this example; no numeric class ID needs
to be configured. Single-class detectors also require a manifest containing
their class name, such as `"classes": ["face"]`. Extraction models may use the
empty class array shown in [the example manifest](docs/manifest.json).

The explicit device-to-host download is a compact array of final embeddings and
box/confidence/frame-index metadata. GPU selection/compaction can also synchronize
small counts with the host. Zvec searches embeddings on the CPU; frame pixels,
face crops and model input tensors never leave GPU memory. This pipeline therefore avoids GPU-to-CPU
image copies; it does not claim zero host transfers for database search.

Existing registered embeddings must come from the same ArcFace model and
compatible preprocessing. Otherwise re-enroll the gallery. Thresholds must be
calibrated against your deployed engines and enrollment images.

## Build an ArcFace TensorRT engine with batching

Export ArcFace to ONNX with a dynamic batch dimension: input `[B,3,112,112]`
and output `[B,512]`. A fixed-batch ONNX model must be re-exported with dynamic
batch support; `trtexec` shape flags alone do not make it dynamic.

For TensorRT 8.6–10.x, build an engine with batch sizes 1–32, optimized for 8:

```bash
trtexec \
  --onnx=models/face_extraction/w600k_r50.onnx \
  --saveEngine=models/face_extraction/face_extraction.trt \
  --minShapes=input.1:1x3x112x112 \
  --optShapes=input.1:8x3x112x112 \
  --maxShapes=input:32x3x112x112 \
  --skipInference
```

Replace `input` in all three shape arguments with the ONNX model's actual input
tensor name. Optionally add `--fp16` to enable FP16 computation. Build on the
deployment GPU with the TensorRT version used by the runtime, and set
`config.model.face_extraction.engine` to the saved engine path. `--skipInference` builds and
saves the engine without running inference or benchmarks. See NVIDIA's
[`trtexec` documentation](https://docs.nvidia.com/tao/tao-toolkit/6.25.07/text/trtexec_integration/index.html)
for the shape and build flags.

The runtime uses profile 0 and splits face crops into batches of at most 32 for
this engine. Increase `--maxShapes` for a larger supported face batch.
`config.batch_size` controls detector frame batches; ArcFace batches the face
crops from those frames according to its own engine profile.

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

## Zvec face search

Face identities are stored in a persistent local Zvec collection. Configure a
writable collection path in `registry.yaml`:

```yaml
    db:
      path: /var/lib/cv-demo/nba-face/identity.zvec
```

`config.db.path` is required. The runtime creates the collection and indexes on
first use; use a path that does not already contain other files. Relative paths
resolve from the host working directory. Keep the directory on persistent local
storage, outside model releases, and make its parent writable for the adjacent
`.lock` file. Runtime and registration instances must use the same path.

The schema stores a 512D FP32 embedding with a cosine HNSW index, employee ID
(`INT64`), image path, pose/type, and millisecond update and deletion timestamps.
Active documents have `deleted_at = 0`; recognition filters out deleted documents.
Registration creates a random positive numeric identity ID within `2**53 - 1`
so it survives protobuf Struct transport exactly. Multiple registrations for
one employee remain supported.

Each search batch queries the nearest active
identity for each embedding in input order. Similarity is cosine similarity;
higher is better and existing session thresholds still apply. An empty active
gallery returns null matches. Writes are flushed before returning. Collection
access goes through one automatically started local owner process per resolved
database path and OS user. It keeps the Zvec handle open and serves all runtime
and registration instances over persistent Unix sockets. Operations are
serialized around that shared handle; a completed registration or deletion is
visible to the next search from any worker without reopening the collection.
The owner holds the adjacent `.lock` file for its lifetime, preventing a second
cooperating owner from opening the collection. Use the same local filesystem
path, OS user, and temporary directory for all workers. Socket files live in a
private `nba-face-db-<uid>` directory under the system temporary directory. No
network port or separately configured service is required.

The owner closes the collection and exits after 30 seconds with no connected
clients. Model replacement can reuse it while other clients remain connected.
If the owner fails, the current operation reports an error; requests are never
automatically replayed because a lost registration reply may already have
committed. A subsequent operation reconnects and restarts the owner if needed.
The registration image is retained when the write outcome is unknown, so a
committed identity cannot lose its image because its reply was interrupted.
Startup failures identify a log file alongside the socket. Restart all existing
workers when adopting this implementation so they use the same ownership model.
This is a local service, not a distributed database.

Database failures propagate to the host rather than reporting unknown faces.
The implementation follows the [Zvec Python collection API](https://zvec.org/api-reference/python/collection/).
Reinstall the package to install the Zvec dependency, update the registry, and
restart the host. PostgreSQL connection settings are no longer used. Existing
PostgreSQL identities are not migrated automatically; re-enroll the gallery or
import compatible embeddings separately before switching production traffic.
