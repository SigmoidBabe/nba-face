"""Persistent Zvec face storage with cosine search and soft deletion."""
from contextlib import contextmanager
import math
from pathlib import Path
import secrets
import time

import zvec


class _FaceStore:
    def __init__(self, config):
        path = config.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Configure config.db.path with a persistent Zvec collection directory")
        self.path = Path(path).expanduser().resolve()
        self.collection = None

    @staticmethod
    def _schema():
        return zvec.CollectionSchema(
            name="identity",
            fields=[
                zvec.FieldSchema("employee_id", zvec.DataType.INT64),
                zvec.FieldSchema("image_url", zvec.DataType.STRING),
                zvec.FieldSchema("type", zvec.DataType.STRING),
                zvec.FieldSchema("deleted_at", zvec.DataType.INT64,
                                 index_param=zvec.InvertIndexParam()),
                zvec.FieldSchema("updated_at", zvec.DataType.INT64),
            ],
            vectors=[zvec.VectorSchema(
                "embedding", zvec.DataType.VECTOR_FP32, dimension=512,
                index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
            )],
        )

    @contextmanager
    def _collection(self):
        # Only the database service owns this store. Its operation lock protects
        # the persistent handle, including the complete search/write operation.
        if self.collection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.collection = (zvec.open(str(self.path)) if self.path.exists() else
                               zvec.create_and_open(str(self.path), self._schema()))
        yield self.collection

    @staticmethod
    def _vector(embedding):
        vector = [float(value) for value in embedding]
        if (len(vector) != 512 or not all(math.isfinite(v) for v in vector)
                or not any(vector)):
            raise ValueError("Zvec identity storage requires finite, nonzero 512D embeddings")
        # Cosine is invariant to scale; normalize before FP32 storage to avoid
        # overflow or underflow for otherwise valid input vectors.
        scale = max(abs(v) for v in vector)
        vector = [v / scale for v in vector]
        norm = math.hypot(*vector)
        return [v / norm for v in vector]

    @staticmethod
    def _check(status):
        if not status.ok():
            raise RuntimeError(f"Zvec write failed: {status.message()}")

    def register(self, employee_id, embedding, image_url):
        vector = self._vector(embedding)
        with self._collection() as collection:
            # Keep numeric API IDs exactly representable through protobuf Struct.
            identity_id = secrets.randbelow(2**53 - 1) + 1
            while collection.fetch(str(identity_id)):
                identity_id = secrets.randbelow(2**53 - 1) + 1
            self._check(collection.insert(zvec.Doc(
                id=str(identity_id), vectors={"embedding": vector},
                fields={"employee_id": employee_id, "image_url": image_url,
                        "type": "front", "deleted_at": 0,
                        "updated_at": time.time_ns() // 1_000_000},
            )))
            collection.flush()
            return {"identity_id": identity_id, "employee_id": employee_id,
                    "image_url": image_url}

    def delete(self, identity_id):
        with self._collection() as collection:
            doc = collection.fetch(str(identity_id)).get(str(identity_id))
            deleted = doc is not None and doc.fields["deleted_at"] == 0
            if deleted:
                now = time.time_ns() // 1_000_000
                self._check(collection.update(zvec.Doc(
                    id=str(identity_id), fields={"deleted_at": now, "updated_at": now},
                )))
                collection.flush()
            return {"identity_id": identity_id, "deleted": deleted}

    def search_batch(self, embeddings):
        """Preserve input order, with a null match for an empty active gallery."""
        if not embeddings:
            return []
        vectors = [self._vector(embedding) for embedding in embeddings]
        rows = []
        with self._collection() as collection:
            for index, vector in enumerate(vectors, 1):
                matches = collection.query(
                    zvec.Query("embedding", vector=vector), topk=1,
                    filter="deleted_at = 0", include_vector=True,
                    output_fields=["employee_id", "image_url", "type"],
                )
                row = dict(ordinality=index, identity_id=None, employee_id=None,
                           image_url=None, pose=None, similarity=None)
                if matches:
                    match = matches[0]
                    stored = self._vector(match.vectors["embedding"])
                    similarity = sum(a * b for a, b in zip(vector, stored))
                    row.update(identity_id=int(match.id),
                               employee_id=match.fields["employee_id"],
                               image_url=match.fields["image_url"], pose=match.fields["type"],
                               similarity=max(-1.0, min(1.0, similarity)))
                rows.append(row)
        return rows

    def close(self):
        if self.collection is not None:
            self.collection.close()
            self.collection = None


class FaceSearch:
    """Persistent client of the single local owner of a Zvec collection."""

    def __init__(self, config):
        from ._database_service import DatabaseClient

        # Keep path validation eager, but start the service only on first use.
        self.path = _FaceStore(config).path
        self._client = DatabaseClient(self.path, config.get("timeout_seconds", 5))

    def register(self, employee_id, embedding, image_url):
        return self._client.call("register", [employee_id, _FaceStore._vector(embedding), image_url])

    def delete(self, identity_id):
        return self._client.call("delete", [identity_id])

    def search_batch(self, embeddings):
        if not embeddings:
            return []
        return self._client.call("search_batch", [[_FaceStore._vector(v) for v in embeddings]])

    def close(self):
        """Release this client's socket; other clients keep the owner alive."""
        self._client.close()
