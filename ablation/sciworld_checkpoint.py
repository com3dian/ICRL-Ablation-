"""Atomic ScienceWorld snapshots; events remain the audit log."""

import hashlib
import json
import warnings
from pathlib import Path

from .events import write_json_atomic


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _config_digest(settings):
    return _digest({key: value for key, value in settings.items() if not key.startswith("_")})


def compact_attempt(attempt):
    return {
        "steps": [{key: step[key] for key in ("step", "action", "observation", "reward", "shown_reward")}
                  for step in attempt["steps"]],
        "generations": {},
        "end": attempt["end"],
    }


def load_checkpoint(event_path, settings):
    """Use a snapshot only when its checksum, configuration and log prefix match."""
    event_path = Path(event_path)
    path = event_path.with_name("checkpoint.json")
    if not path.exists():
        return {}, set(), None, 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("snapshot must be a JSON object")
        checksum = payload.pop("checksum")
        if (checksum != _digest(payload) or payload["version"] != 1
                or payload["config_digest"] != _config_digest(settings)):
            raise ValueError("snapshot checksum, version or configuration differs")
        offset = payload["event_offset"]
        if type(offset) is not int or not 0 <= offset <= event_path.stat().st_size:
            raise ValueError("snapshot event offset is invalid")
        digest = hashlib.sha256()
        with event_path.open("rb") as stream:
            remaining = offset
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError("snapshot extends beyond the event log")
                digest.update(block)
                remaining -= len(block)
            if offset:
                stream.seek(offset - 1)
                if stream.read(1) != b"\n":
                    raise ValueError("snapshot offset is not a record boundary")
        if digest.hexdigest() != payload["event_digest"]:
            raise ValueError("snapshot event log prefix differs")
        attempts = {(item["round"], item["sample_id"]): item["attempt"] for item in payload["attempts"]}
        for attempt in attempts.values():
            attempt["generations"] = {int(step): event for step, event in attempt["generations"].items()}
        return attempts, set(payload["rounds"]), payload["manifest"], offset
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.warn(f"Ignoring ScienceWorld checkpoint; rebuilding from events.jsonl: {exc}", RuntimeWarning)
        return {}, set(), None, 0


class Checkpointer:
    def __init__(self, run_dir, settings):
        self.path = Path(run_dir) / "checkpoint.json"
        self.event_path = Path(run_dir) / "events.jsonl"
        self.config_digest = _config_digest(settings)
        self.digest = hashlib.sha256()
        self.offset = 0

    def save(self, attempts, rounds, manifest):
        # Hash each byte once per process, including the original log on resume.
        with self.event_path.open("rb") as stream:
            stream.seek(self.offset)
            while block := stream.read(1024 * 1024):
                self.digest.update(block)
            self.offset = stream.tell()
        payload = {
            "version": 1, "config_digest": self.config_digest,
            "event_offset": self.offset, "event_digest": self.digest.hexdigest(),
            "attempts": [{"round": key[0], "sample_id": key[1],
                          "attempt": compact_attempt(attempt) if attempt.get("end") else attempt}
                         for key, attempt in sorted(attempts.items())],
            "rounds": sorted(rounds), "manifest": manifest,
        }
        payload["checksum"] = _digest(payload)
        write_json_atomic(self.path, payload)
