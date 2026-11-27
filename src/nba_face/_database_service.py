"""Private local database owner. Never forks a GPU worker or unpickles IPC data."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time


IDLE_SECONDS = 30
STARTUP_SECONDS = 60


class DatabaseOutcomeUnknownError(RuntimeError):
    """A disconnected request may already have committed; never replay it."""


def _endpoint(path):
    # Short Unix socket names, shared by all processes for this user/path.
    directory = Path(tempfile.gettempdir()) / f"nba-face-db-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"Database socket directory must be private to this user: {directory}")
    name = hashlib.sha256(os.fsencode(path)).hexdigest()[:32]
    return directory / f"{name}.sock"


def _send(stream, message):
    stream.write(json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n")
    stream.flush()


def _receive(stream):
    line = stream.readline()
    if not line:
        raise EOFError("Database service disconnected")
    return json.loads(line)


class DatabaseClient:
    def __init__(self, path, timeout_seconds=5):
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("config.db.timeout_seconds must be a finite positive number")
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._socket = self._stream = None

    def _disconnect(self, *, shutdown=True):
        stream, connection = self._stream, self._socket
        self._stream = self._socket = None
        if connection is not None and shutdown:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for resource in (stream, connection):
            if resource is not None:
                try:
                    resource.close()
                except OSError:
                    # Closing a broken buffered stream can flush and fail too.
                    # Preserve the original uncertain-outcome exception.
                    pass

    def _connect(self, endpoint):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_seconds)
        try:
            connection.connect(str(endpoint))
        except OSError:
            connection.close()
            raise
        self._socket = connection
        self._stream = connection.makefile("rwb")

    def _ensure_connected(self):
        if self._stream is not None:
            return
        endpoint = _endpoint(self.path)
        deadline = time.monotonic() + STARTUP_SECONDS
        # Prevent multiple workers starting owners at the same time.
        with endpoint.with_suffix(".start.lock").open("a+b") as startup_lock:
            while True:
                try:
                    fcntl.flock(startup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Database startup lock timed out: {endpoint}")
                    time.sleep(0.02)
            try:
                self._connect(endpoint)
                return
            except (FileNotFoundError, ConnectionRefusedError):
                pass
            log_path = endpoint.with_suffix(".log")
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "nba_face._database_service", str(self.path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True, close_fds=True,
                )
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Database service failed to start; see {log_path}")
                try:
                    self._connect(endpoint)
                    # Reap our child without coupling its lifetime to the worker.
                    threading.Thread(target=process.wait, daemon=True).start()
                    return
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        process.terminate()
                        process.wait()
                        raise RuntimeError(f"Database service startup timed out; see {log_path}")
                    time.sleep(0.02)

    def _after_fork(self):
        if self._pid != os.getpid():
            # Do not reuse a parent's stream or a possibly locked thread mutex.
            self._lock = threading.Lock()
            # shutdown() would affect the parent's copy of this same socket.
            self._disconnect(shutdown=False)
            self._pid = os.getpid()

    def call(self, method, args):
        self._after_fork()
        if not self._lock.acquire(timeout=self.timeout_seconds):
            raise TimeoutError(f"Database client busy before {method}; request was not sent")
        try:
            self._ensure_connected()
            try:
                _send(self._stream, {"method": method, "args": args})
                response = _receive(self._stream)
            except (OSError, EOFError, ValueError) as exc:
                self._disconnect()
                # A write might already have committed. Retrying could duplicate
                # registration, so leave retry decisions to the caller.
                raise DatabaseOutcomeUnknownError(
                    f"Database service {method} timed out or connection lost "
                    f"(socket timeout {self.timeout_seconds:g}s); operation outcome may be unknown"
                ) from exc
            if "error" in response:
                error = response["error"]
                exception = ValueError if error["type"] == "ValueError" else RuntimeError
                raise exception(error["message"])
            return response["result"]
        finally:
            self._lock.release()

    def close(self):
        self._after_fork()
        with self._lock:
            self._disconnect()


def serve(path):
    from .database import _FaceStore

    path = Path(path).expanduser().resolve()
    endpoint = _endpoint(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Retain the original interprocess lock for the full ownership lifetime.
    with (endpoint.with_suffix(".start.lock").open("a+b") as startup_lock,
          path.with_name(path.name + ".lock").open("a+b") as owner_lock):
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = _FaceStore({"path": str(path)})
        operation_lock = threading.Lock()
        state_lock = threading.Lock()
        clients = 0
        idle_since = time.monotonic()

        def handle(connection):
            nonlocal clients, idle_since
            try:
                with connection, connection.makefile("rwb") as stream:
                    while True:
                        try:
                            request = _receive(stream)
                        except EOFError:
                            return
                        try:
                            method = request["method"]
                            if method not in {"register", "delete", "search_batch"}:
                                raise ValueError("Unknown database operation")
                            with operation_lock:
                                result = getattr(store, method)(*request["args"])
                            response = {"result": result}
                        except Exception as exc:
                            response = {"error": {"type": type(exc).__name__, "message": str(exc)}}
                        _send(stream, response)
            except (OSError, ValueError):
                pass  # Client exited, including a worker killed during a request.
            finally:
                with state_lock:
                    clients -= 1
                    if not clients:
                        idle_since = time.monotonic()

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Initialize before advertising readiness, while errors can still
            # be reported as startup failures. Preserve the on-disk schema.
            with store._collection():
                pass
            endpoint.unlink(missing_ok=True)
            listener.bind(str(endpoint))
            os.chmod(endpoint, 0o600)
            listener.listen(128)
            listener.settimeout(1)
            while True:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    with state_lock:
                        idle = clients == 0 and time.monotonic() - idle_since >= IDLE_SECONDS
                    if idle:
                        try:
                            fcntl.flock(startup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        # Coordinate shutdown with connecting clients. A socket
                        # may be queued for accept but not counted as active yet.
                        if not select.select([listener], [], [], 0)[0]:
                            break
                        fcntl.flock(startup_lock, fcntl.LOCK_UN)
                    continue
                with state_lock:
                    clients += 1
                threading.Thread(target=handle, args=(connection,), daemon=True).start()
        finally:
            listener.close()
            endpoint.unlink(missing_ok=True)
            store.close()


if __name__ == "__main__":
    serve(sys.argv[1])
