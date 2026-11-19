# Face recognition event payload

For the `face.recognition` capability, use the event envelope below. Only
`event_payload` differs from the supplied template; all other example values
are preserved.

```json
{
  "event_payload": {
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
        "similarity": 0.82,
        "pose": "front"
      }
    ]
  },
  "event_id": "7c110961-131d-4ea1-9107-21987aa78530",
  "frame_timestamp": "2026-11-13T08:27:19.503Z",
  "frame_timestamp_ns": "1794558439503367555",
  "timestamp": "2026-11-13T08:27:19.503Z",
  "frame_sequence": 1,
  "session_id": "11",
  "type": "inference_event",
  "camera_id": "CAM-002"
}
```

The host supplies the envelope metadata and uses the runtime event's `data`
snapshot as `event_payload`.

| Payload field | Meaning |
| --- | --- |
| `faces_count` | Number of entries in `faces`, including unmatched faces. Detections with invalid embeddings are omitted. |
| `faces` | All current face results, including previously present and unmatched faces. |
| `faces[].box` | Integer source-frame coordinates `[x1, y1, x2, y2]`. |
| `faces[].confidence` | Face detector confidence, independent of recognition similarity. |
| `faces[].face_area` | Box area in square pixels, computed before coordinates are converted to integers. |
| `faces[].face_area_ratio` | Box area divided by source-frame area. |
| `faces[].estimated_distance_cm` | Area-based estimate: `60 * sqrt(0.065 / face_area_ratio)`, or `null` for a nonpositive ratio. This is not a calibrated depth measurement. |
| `faces[].identity_id` | Matched registration row ID, or `null`. |
| `faces[].employee_id` | Matched employee ID, or `null`. |
| `faces[].image_url` | Matched enrollment image reference, or `null`. Package registrations store a filesystem path. |
| `faces[].similarity` | Nearest cosine similarity, including scores below the session threshold, or `null` when no match exists. |
| `faces[].pose` | Matched registration's stored `type`, or `null`. Package registration stores `front`; inference does not estimate pose. |

When similarity is absent, nonfinite, or below the session threshold,
`identity_id`, `employee_id`, `image_url`, and `pose` are `null`.
An empty scene has `"event_payload": {"faces_count": 0, "faces": []}` and can
emit in periodic mode. Event mode emits only when a recognized identity appears
or reappears after absence. Face recognition uses the full frame and does not
include people counts or zones.

See [Recognition sessions and events](session_event.md) for session configuration
and emission behavior.
