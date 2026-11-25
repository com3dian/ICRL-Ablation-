"""Append-only local experiment records for live viewing and audit."""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fcntl

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


def write_json_atomic(path: Path, value: dict, *, durable: bool = True, indent: int | None = None):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.stem}-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=indent)
            stream.write("\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class EventLog:
    def __init__(self, task: str, model_alias: str, feedback: str, config: dict):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = OUTPUT_ROOT / task / model_alias / feedback
        polarity = config.get("reward_polarity", "normal")
        if polarity != "normal":
            self.path /= polarity
        self.path /= f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        write_json_atomic(self.path / "config.json", config, durable=task == "sciworld", indent=2)
        self._stream = (self.path / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._durable = task == "sciworld"

    @classmethod
    def reopen(cls, path: Path):
        """Append to an unfinished run without replacing its config or events."""
        path = path.resolve()
        if not (path / "config.json").is_file() or not (path / "events.jsonl").is_file():
            raise FileNotFoundError(f"Run has no config.json and events.jsonl: {path}")
        if (path / "summary.json").exists():
            raise ValueError(f"Run is already complete: {path}")
        log = cls.__new__(cls)
        log.path = path
        log._durable = json.loads((path / "config.json").read_text(encoding="utf-8")).get("task") == "sciworld"
        log._stream = (path / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        try:
            fcntl.flock(log._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            log._stream.close()
            raise
        try:
            if log._durable:
                log._recover_tail()
        except BaseException:
            log._stream.close()
            raise
        return log

    def _recover_tail(self):
        """Recover only an unterminated final record, keeping its original bytes."""
        event_path = self.path / "events.jsonl"
        with event_path.open("rb+") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            if not size:
                return
            stream.seek(size - 1)
            if stream.read(1) == b"\n":
                return
            position = size
            tail = b""
            while position:
                start = max(0, position - 8192)
                stream.seek(start)
                tail = stream.read(position - start) + tail
                boundary = tail.rfind(b"\n")
                if boundary >= 0:
                    tail = tail[boundary + 1:]
                    break
                position = start
            try:
                json.loads(tail)
            except (ValueError, UnicodeError):
                archive = self.path / f"events-truncated-{uuid.uuid4().hex}.bin"
                with archive.open("xb") as fragment:
                    fragment.write(tail)
                    fragment.flush()
                    os.fsync(fragment.fileno())
                stream.truncate(size - len(tail))
                stream.flush()
                os.fsync(stream.fileno())
                self.emit("log_tail_recovered", discarded_bytes=len(tail), archive=archive.name)
            else:
                self._stream.write("\n")
                self._stream.flush()
                os.fsync(self._stream.fileno())

    def emit(self, event: str, **fields):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
        if self._durable:
            os.fsync(self._stream.fileno())

    def finish(self, summary: dict):
        write_json_atomic(self.path / "summary.json", summary, durable=self._durable, indent=2)
        self.emit("run_end", **summary)

    def close(self):
        self._stream.close()
