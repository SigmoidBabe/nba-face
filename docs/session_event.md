# Recognition sessions and events

The `nba.face` runtime implements full-frame `face.recognition` sessions. The
host owns camera ingestion, scheduling, and event delivery; the package returns
a snapshot and an optional event for each eligible session on a processed frame.

## Start a session

Submit a session through the host's `RuntimeControl.StartRuntime` API. This
example uses appearance events:

```json
{
  "session_id": "nba-face-camera-1",
  "status": true,
  "source": {
    "camera": {
      "camera_id": "camera-1",
      "stream_address": "rtsp://localhost:8554/example"
    }
  },
  "runtime": {"id": "nba.face", "capability": "face.recognition"},
  "inference": {
    "inference_primitive_usecase": "face recognition",
    "config": {"fps": 2, "similarity_threshold": 0.6}
  },
  "output": {"mode": "event"}
}
```

For periodic delivery, use `"output": {"mode": "periodic", "interval": 2}`,
as in [session.example.json](../session.example.json). The host adapts this
request into the runtime session dictionary: `session_id`, `capability`,
`config`, `output`, and optional `zones`. The runtime reads the threshold from
that dictionary's `config.similarity_threshold`; the public request places it
under `inference.config`.

| Runtime session setting | Default | Behavior |
| --- | --- | --- |
| `capability` | Required | Must be `face.recognition`. |
| `config.similarity_threshold` | Runtime threshold, initially `0.6` | Finite number in `[0, 1]`; booleans are rejected. |
| `output.mode` | `periodic` | Either `periodic` or `event`. |
| `output.interval` | `0` | Finite, non-negative number of seconds; booleans are rejected. Used only in periodic mode. |
| `zones` | Omitted | Nonempty zones are rejected; recognition uses the full frame. |

Camera source settings and `fps` are handled by the host. This package does not
open streams or schedule inference itself.

## Recognition and snapshots

For each detected face with a valid embedding, the runtime searches active,
non-null PostgreSQL embeddings for the nearest cosine match. Higher similarity
is better. Each session applies its own threshold to the shared search result.
A similarity equal to the threshold is accepted.

If similarity is absent, nonfinite, or below the threshold, `identity_id`,
`employee_id`, `image_url`, and `pose` become `null`. The similarity itself is
retained, including a below-threshold score. An empty gallery produces unmatched
faces with null match fields. Database failures propagate as errors rather than
being treated as an empty gallery.

Every processed session returns a snapshot, even when no event is due:

```json
{
  "snapshot": {
    "faces_count": 1,
    "faces": [
      {
        "box": [100, 80, 220, 240],
        "confidence": 0.95,
        "face_area": 19200.0,
        "face_area_ratio": 0.0625,
        "estimated_distance_cm": 61.188234,
        "identity_id": 123,
        "employee_id": 42,
        "image_url": "/var/lib/cv-demo/nba-face/images/example.jpg",
        "similarity": 0.82,
        "pose": "front"
      }
    ]
  },
  "events": []
}
```

| Snapshot field | Meaning |
| --- | --- |
| `faces_count` | Number of face entries, including unmatched faces. Detections with invalid embeddings are omitted. |
| `faces[].box` | Integer `[x1, y1, x2, y2]` coordinates in the source frame. |
| `confidence` | Detector confidence, independent of recognition similarity. |
| `face_area` | Detected box area in square pixels, computed before box coordinates are converted to integers. |
| `face_area_ratio` | Detected box area divided by source-frame area. |
| `estimated_distance_cm` | Area heuristic: `60 * sqrt(0.065 / face_area_ratio)`, or `null` for a nonpositive ratio. This is not a calibrated depth measurement. |
| `identity_id` | Matched registration row ID, or `null`. |
| `employee_id` | Matched employee ID, or `null`. Multiple registrations can share an employee. |
| `image_url` | Matched enrollment image reference, or `null`; package registrations store a filesystem path. |
| `similarity` | Nearest cosine similarity, or `null` when no match exists. |
| `pose` | Matched row's stored `type`, or `null`. Package registration writes `front`; inference does not estimate pose. |

An empty scene returns `{"faces_count": 0, "faces": []}` as its snapshot.

## Periodic mode

The first processed frame emits an event, including an empty scene. Later
events emit on processed frames when at least `output.interval` seconds have
elapsed since the previous emission. An interval of zero emits on every
processed frame.

Timing uses the process's monotonic clock, not camera timestamps. There is no
background timer: if no frame is processed, no event is emitted. Processing a
batch quickly does not replay periodic events according to frame capture times.

## Event mode

The session compares the set of recognized `identity_id` values in the current
processed frame with the set from the preceding processed frame. It emits one
event when at least one identity is newly present. The first frame emits only
if it contains a recognized identity.

| Recognized IDs by processed frame | Emits? | Reason |
| --- | --- | --- |
| `[123]` | Yes | First appearance. |
| `[123]` | No | Same identity remains visible. |
| `[]` | No | Disappearance does not emit. |
| `[123]` | Yes | Reappearance after absence. |
| `[123, 456]` | Yes | Identity `456` is newly present. |

There is no absence timeout, tracking, or debounce. A single processed frame
without an identity resets its presence, including when its score falls below
threshold. Frames skipped by the host do not update presence state.
`output.interval` is validated but does not throttle event mode.

Presence is keyed by registration ID, not employee ID. Switching between two
registered images of the same employee can therefore emit another event.
Multiple faces matched to the same identity count as one presence. Unknown
faces alone never trigger appearance events.

## Event payload and runtime integration

When emission is due, `events` contains exactly one object whose `data` is the
entire current snapshot:

```json
{
  "data": {
    "faces_count": 0,
    "faces": []
  }
}
```

This empty example can be emitted in periodic mode. In event mode, the data
includes all current faces, including unmatched and previously present faces,
alongside the newly appearing identity. The package adds no event type,
timestamp, camera ID, or session ID inside this object. Host transport metadata
and envelopes are outside this package's result contract. Events contain no
frame pixels or newly captured image uploads.

`FaceRuntime.process_frame(frame, session_ids)` returns a mapping from each
eligible session ID to its `{snapshot, events}` result. The frame's `image`
must be an HWC BGR `uint8` CUDA array on the configured device. CPU frames are
rejected. `process_batch(frames, session_ids_by_frame)` returns one such mapping
per input frame in order. Inference and search are shared across sessions on a
frame; thresholds, timers, and presence sets are maintained per session.

Pass frames chronologically within each session. Frames with no eligible
sessions skip inference and do not advance session state. Unknown session IDs
raise `KeyError`. Creating a session with an existing ID replaces its state;
closing it removes its state. Recreating it starts with no previous identities
and no prior emission time.

For enrollment and soft deletion, see [Face registration](registration.md).
