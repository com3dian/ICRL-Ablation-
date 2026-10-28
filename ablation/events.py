"""Append-only local experiment records for live viewing and audit."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fcntl

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"


class EventLog:
    def __init__(self, task: str, model_alias: str, feedback: str, config: dict):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = OUTPUT_ROOT / task / model_alias / feedback
        if config.get("reward_polarity", "normal") == "flipped":
            self.path /= "flipped"
        self.path /= f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        self._stream = (self.path / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

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
        log._stream = (path / "events.jsonl").open("a", encoding="utf-8", buffering=1)
        try:
            fcntl.flock(log._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            log._stream.close()
            raise
        return log

    def emit(self, event: str, **fields):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()

    def finish(self, summary: dict):
        (self.path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.emit("run_end", **summary)

    def close(self):
        self._stream.close()
