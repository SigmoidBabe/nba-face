# Face registration API

The `face.registration` API enrolls faces and soft-deletes individual
registrations through the host's `cv_demo.api.v1.PackageAPI/Invoke` endpoint.
Recognition sessions use the registered embeddings for PostgreSQL cosine search.

## Setup

Install the package into the host environment and enable its API in the runtime
registry. See [registry.yaml](../registry.yaml) for the complete example:

```yaml
- runtime_id: nba.face
  package:
    entry_point: nba.face
  grpc_apis:
    - face.registration
  config:
    model:
      face_detection:
        engine: /opt/models/face-yolov8.engine
      face_extraction:
        engine: /opt/models/arcface.engine
    registration_image_dir: /var/lib/cv-demo/nba-face/images
    db:
      url: 'postgresql://USER:PASSWORD@HOST:5432/DATABASE'
  capabilities:
    - id: face.recognition
      output_modes: [periodic, event]
```

Both model engine paths, `registration_image_dir`, and `db` are required and have
no defaults. Operation counts are only required for the performance API. The
detector also requires a class-name `.txt` file beside its engine,
with the same filename stem. The default face class name is `face`. See the
[README](../README.md#gpu-pipeline-and-engine-contracts) for engine requirements.
The image directory must be writable and persistent.

The database must already provide pgvector and `public.identity` with `id`,
`employee_id` (`BIGINT`), `embedding` (`vector(512)`), `image_url`, `type`,
`deleted_at`, and `updated_at`. The package does not create tables or indexes.
Only one provider of `face.registration` should be enabled in the host. Reinstall
the package after updates so the host discovers its API entry point.

## Register a face

Send this Invoke request, replacing the image placeholder with raw base64:

```json
{
  "api": "face.registration",
  "method": "register_face",
  "payload": {
    "employee_id": "42",
    "image_base64": "<base64-encoded JPEG or PNG bytes>",
    "crop_type": "loose"
  }
}
```

| Payload field | Required | Meaning |
| --- | --- | --- |
| `employee_id` | Yes | Signed 64-bit integer or integer string identifying the employee. |
| `image_base64` | Yes | Nonempty base64 string containing a decodable image, without a data URL prefix or embedded whitespace. |
| `crop_type` | No | `full`, `tight`, or `loose`; defaults to `loose`. Case and surrounding whitespace are ignored. |

Use strings for large identifiers to avoid number precision loss through
protobuf Struct. Employee names, booleans, fractional numbers, and values
outside the signed 64-bit range are rejected. Employee IDs need not be positive.

The image must contain exactly one face surviving the configured detection
filters and NMS. This requirement also applies to `full`. The selected crop is
both embedded by ArcFace and saved as a JPEG:

| Crop type | Region |
| --- | --- |
| `full` | Entire uploaded image. |
| `tight` | Detected face box, rounded outward to integer pixel bounds. |
| `loose` | Face box expanded by 25% of its width and height on each side, rounded outward and clipped to the image. |

Registration's loose crop uses a fixed 25% margin, independent of recognition's
`config.crop_margin` (default 20%). No landmark alignment or pose classification
is performed. The API normalizes a finite, nonzero 512-dimensional embedding
and inserts a row with `type = 'front'`.

Example package response:

```json
{
  "identity_id": 123,
  "employee_id": 42,
  "image_url": "/var/lib/cv-demo/nba-face/images/550e8400-e29b-41d4-a716-446655440000.jpg"
}
```

Each call inserts a new identity, including repeated calls for the same employee
or image. The API does not deduplicate registrations or replace an employee's
previous images. Use the returned `identity_id` to delete a particular entry.

`image_url` is the host filesystem path to a UUID-named JPEG, without a `file://`
prefix. It is not an HTTP download URL, and no S3 upload occurs. A relative
`registration_image_dir` is resolved against the host's working directory and
remains relative in the returned path. If saving or inserting fails after file
creation, the API attempts to remove that file.

Images are decoded on the CPU and uploaded for GPU detection and embedding.
The API loads its models lazily, separately from recognition runtime models.
Registration and deletion calls on the same API instance share a lock.
Newly committed identities are eligible for subsequent recognition queries
without a gallery reload. Enrollment and recognition must use compatible
ArcFace models and preprocessing.

## Delete a registration

```json
{
  "api": "face.registration",
  "method": "delete_face",
  "payload": {
    "identity_id": "123"
  }
}
```

`identity_id` must be a positive signed 64-bit integer or integer string. An
active registration returns:

```json
{"identity_id": 123, "deleted": true}
```

An absent or already deleted registration returns the same ID with
`"deleted": false`. Deletion sets `deleted_at` and `updated_at`, preserving the
row and JPEG. Subsequent recognition queries exclude the row; other entries for
the same employee remain active. Deletion does not load GPU models.

## Errors

The package raises exceptions for the host to translate into gRPC statuses:

| Status | Cause |
| --- | --- |
| `INVALID_ARGUMENT` (`ValueError`) | Invalid payload or IDs, unsupported crop type, invalid base64, undecodable image, no face or multiple retained faces, or an invalid embedding. |
| `UNIMPLEMENTED` (`NotImplementedError`) | A method other than `register_face` or `delete_face`. |
| `INTERNAL` | Database, model, GPU, or filesystem failures propagated by the host. |

Response examples above show the package result object. Transport wrapping and
delivery are owned by the host. For recognition output and appearance behavior,
see [Session events](session_event.md).
