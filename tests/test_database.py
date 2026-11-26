"""Real Zvec integration: PYTHONPATH=src python -m unittest discover -s tests -v."""
from concurrent.futures import ThreadPoolExecutor
import json
import importlib.util
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from nba_face._database_service import DatabaseClient, DatabaseOutcomeUnknownError, _endpoint
from nba_face.database import FaceSearch


VECTOR = [1.0] + [0.0] * 511


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "faces.zvec"
        self.clients = []
        self.owners = set()

    def client(self):
        client = FaceSearch({"path": str(self.path)})
        self.clients.append(client)
        return client

    def owner(self, client):
        credentials = client._client._socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        pid = struct.unpack("3i", credentials)[0]
        self.owners.add(pid)
        return pid

    def tearDown(self):
        for client in self.clients:
            client.close()
        for pid in self.owners:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.directory.cleanup()

    def test_missing_database_and_persistent_shared_owner(self):
        first, second = self.client(), self.client()
        self.assertFalse(self.path.exists())
        self.assertEqual(first.search_batch([]), [])
        self.assertFalse(self.path.exists())
        self.assertIsNone(first.search_batch([VECTOR])[0]["identity_id"])
        pid = self.owner(first)
        registered = second.register(42, VECTOR, "face.jpg")
        self.assertEqual(self.owner(second), pid)
        for _ in range(10):
            match = first.search_batch([VECTOR])[0]
            self.assertEqual(match["identity_id"], registered["identity_id"])
            self.assertAlmostEqual(match["similarity"], 1)
            self.assertEqual(self.owner(first), pid)
        second.close()
        self.assertEqual(first.search_batch([VECTOR])[0]["employee_id"], 42)

    def test_registration_and_deletion_from_another_process(self):
        reader = self.client()
        reader.search_batch([VECTOR])
        self.owner(reader)
        code = """
import json, sys
from nba_face.database import FaceSearch
s = FaceSearch({'path': sys.argv[1]})
if len(sys.argv) == 2:
    print(json.dumps(s.register(99, [1.] + [0.] * 511, 'cross-process.jpg')))
else:
    print(json.dumps(s.delete(int(sys.argv[2]))))
s.close()
"""
        registered = json.loads(subprocess.check_output(
            [sys.executable, "-c", code, str(self.path)], text=True, timeout=10))
        self.assertEqual(reader.search_batch([VECTOR])[0]["identity_id"], registered["identity_id"])
        deleted = json.loads(subprocess.check_output(
            [sys.executable, "-c", code, str(self.path), str(registered["identity_id"])],
            text=True, timeout=10))
        self.assertTrue(deleted["deleted"])
        self.assertIsNone(reader.search_batch([VECTOR])[0]["identity_id"])

    def test_concurrent_startup_searches_and_writes(self):
        clients = [self.client() for _ in range(8)]

        def work(item):
            index, client = item
            registered = client.register(index, VECTOR, f"{index}.jpg")
            for _ in range(5):
                self.assertIsNotNone(client.search_batch([VECTOR])[0]["identity_id"])
            return registered["identity_id"]

        with ThreadPoolExecutor(max_workers=8) as executor:
            ids = list(executor.map(work, enumerate(clients)))
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(len({self.owner(client) for client in clients}), 1)

    def test_validation_and_error_does_not_break_connection(self):
        client = self.client()
        for vector in ([1.0], [0.0] * 512, [float("nan")] * 512):
            with self.assertRaises(ValueError):
                client.search_batch([vector])
        self.assertFalse(self.path.exists())
        client.search_batch([VECTOR])
        self.owner(client)
        with self.assertRaises(ValueError):
            client._client.call("unsupported", [])
        self.assertIsNone(client.search_batch([VECTOR])[0]["identity_id"])

    def test_owner_failure_is_reported_then_reconnects_with_durable_data(self):
        client = self.client()
        registered = client.register(42, VECTOR, "durable.jpg")
        pid = self.owner(client)
        os.kill(pid, signal.SIGKILL)
        with self.assertRaisesRegex(RuntimeError, "outcome may be unknown"):
            client.search_batch([VECTOR])
        self.assertEqual(client.search_batch([VECTOR])[0]["identity_id"], registered["identity_id"])
        self.assertNotEqual(self.owner(client), pid)
        self.owners.discard(pid)

    def test_lost_registration_reply_is_not_replayed(self):
        writer, reader = self.client(), self.client()
        writer.search_batch([VECTOR])
        self.owner(writer)
        # Submit the real insert, then simulate losing its reply. The client
        # must report an uncertain outcome rather than retrying the insert.
        with patch("nba_face._database_service._receive", side_effect=EOFError):
            with self.assertRaises(DatabaseOutcomeUnknownError):
                writer.register(77, VECTOR, "keep-this-image.jpg")
        deadline = time.monotonic() + 5
        while True:
            match = reader.search_batch([VECTOR])[0]
            if match["identity_id"] is not None:
                break
            if time.monotonic() >= deadline:
                self.fail("Submitted registration did not commit")
            time.sleep(0.01)
        self.assertEqual(match["employee_id"], 77)
        self.assertEqual(match["image_url"], "keep-this-image.jpg")
        # If the client replayed the insert, deleting this identity would
        # leave a duplicate active registration behind.
        writer.delete(match["identity_id"])
        self.assertIsNone(reader.search_batch([VECTOR])[0]["identity_id"])

    def test_invalid_collection_reports_startup_failure(self):
        self.path.mkdir()
        (self.path / "unrelated.txt").write_text("not a collection")
        with self.assertRaisesRegex(RuntimeError, "failed to start; see"):
            self.client().search_batch([VECTOR])

    def test_idle_owner_closes_collection_and_releases_locks(self):
        code = """
import sys
from nba_face import _database_service as service
service.IDLE_SECONDS = 0.1
service.serve(sys.argv[1])
"""
        process = subprocess.Popen([sys.executable, "-c", code, str(self.path)])
        try:
            endpoint = _endpoint(self.path)
            deadline = time.monotonic() + 10
            while not endpoint.exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    self.fail("Owner failed to start")
                time.sleep(0.01)
            client = self.client()
            client.register(42, VECTOR, "idle.jpg")
            client.close()
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertFalse(endpoint.exists())
            self.assertEqual(client.search_batch([VECTOR])[0]["employee_id"], 42)
            self.owner(client)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)


class FailureCleanupTests(unittest.TestCase):
    def test_broken_stream_cleanup_still_closes_socket(self):
        client = DatabaseClient(Path("/unused"))
        stream, connection = Mock(), Mock()
        stream.close.side_effect = BrokenPipeError("already disconnected")
        client._stream, client._socket = stream, connection
        client.close()
        stream.close.assert_called_once()
        connection.close.assert_called_once()
        self.assertIsNone(client._stream)
        self.assertIsNone(client._socket)

    def test_uncertain_write_preserves_image_and_known_failure_removes_it(self):
        # Load just the API with GPU/image dependencies stubbed. The real
        # registration filesystem/error path is what this test exercises.
        source = Path(__file__).parents[1] / "src/nba_face/grpc.py"
        spec = importlib.util.spec_from_file_location("nba_face._cleanup_test", source)
        module = importlib.util.module_from_spec(spec)
        cv2 = Mock()
        cv2.imencode.return_value = (True, Mock(tobytes=lambda: b"jpeg"))
        with patch.dict(sys.modules, {"cv2": cv2, "nba_face.backend": Mock()}):
            spec.loader.exec_module(module)
        cv2.imdecode.return_value = module.np.zeros((1, 1, 3), dtype=module.np.uint8)
        for error, retained in ((DatabaseOutcomeUnknownError("reply lost"), True),
                                (RuntimeError("insert rejected"), False)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                api = module.FaceRegistrationAPI({
                    "db": {"path": directory + "/database"},
                    "registration_image_dir": directory + "/images",
                })
                api.backend = Mock()
                api.backend.registration_embedding.return_value = (VECTOR, (0, 0, 1, 1))
                api.database = Mock()
                api.database.register.side_effect = error
                with self.assertRaises(type(error)):
                    api.invoke("register_face", {"employee_id": 1, "image_base64": "eA=="})
                self.assertEqual(bool(list((Path(directory) / "images").glob("*.jpg"))), retained)
                api.close()


if __name__ == "__main__":
    unittest.main()
