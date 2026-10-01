"""Durable batch checkpoint journal (dossier 05 recovery invariants).

Layout under the user-named checkpoint directory
(``ExecutionOptions.checkpoint``):

    journal.json                     one atomically-replaced JSON document
    captures/row<i>__<column>.npy    composite fold inputs (raw numpy bytes)

Durability discipline mirrors the cache store: every append is
write-to-tmp -> flush -> fsync -> os.replace -> directory fsync, so a
crashed append leaves the previous journal intact.  Content rules enforce
the dossier's "never load untrusted pickle to recover a job": the journal
is JSON only and capture payloads are raw ``.npy`` bytes whose digests are
recorded in the journal and verified on every read.  A journal that cannot
be parsed is quarantined (renamed aside, never deleted) and resume then
re-runs the whole batch rather than mixing generations.

Identity (code/model/profile/input digests) is computed by the runtime and
passed in; this module only stores and compares it.  Timings and worker
scheduling data are deliberately absent — they are not identity
(INTERFACES.md) and must never make a resume reject a valid batch.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from .report import BatchCheckpointError

__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "JournalRow",
    "CheckpointJournal",
    "sha256_file",
    "sha256_bytes",
]

JOURNAL_SCHEMA_VERSION = 1

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _canonical(value: Any) -> Any:
    """JSON-safe projection of a Settings value.

    Frozen dataclasses (ExecutionOptions/CacheOptions) become dicts;
    tuples become lists; ndarrays become an exact self-describing marker
    (dtype + shape + raw C-order bytes) — ``knn_weights`` is an ndarray
    by Settings contract, and the journal must represent it losslessly.
    Runtime restores the marker via ``runtime._decode_journal_value``.
    Anything else non-scalar is a coding error — fail closed rather
    than silently stringifying a numerical field.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {k: _canonical(v) for k, v in vars(value).items()}
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {"__ndarray__": {
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }}
    raise BatchCheckpointError(
        f"settings value {type(value).__name__!r} has no canonical "
        f"checkpoint representation; refusing to serialize it lossily")


class JournalRow(dict):
    """One row's committed record (plain dict for JSON roundtripping)."""


class CheckpointJournal:
    """Atomically-appended batch journal with identity + artifact checks."""

    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)
        self.journal_path = os.path.join(self.directory, "journal.json")
        self.captures_dir = os.path.join(self.directory, "captures")
        self._cache: Optional[dict] = None

    # -- loading ----------------------------------------------------------

    def load(self) -> Optional[dict]:
        """Return the current journal document, or None when absent.

        An unreadable/corrupt journal is quarantined (renamed aside with
        the reason) and None is returned — the caller starts a fresh
        generation rather than resuming a mixed state.
        """
        if self._cache is not None:
            return self._cache
        if not os.path.exists(self.journal_path):
            return None
        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            if not isinstance(doc, dict):
                raise ValueError("journal is not a JSON object")
            if doc.get("schema_version") != JOURNAL_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version "
                    f"{doc.get('schema_version')!r}")
            if not isinstance(doc.get("identity"), str):
                raise ValueError("journal has no identity digest")
            if not isinstance(doc.get("rows"), dict):
                raise ValueError("journal has no rows map")
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self.quarantine(f"unreadable journal: {exc}")
            return None
        self._cache = doc
        return doc

    def quarantine(self, reason: str) -> str:
        """Move the current journal aside (never delete) and return the path."""
        self._cache = None
        if not os.path.exists(self.journal_path):
            return ""
        dest = self.journal_path + ".quarantined"
        n = 0
        while os.path.exists(dest):
            n += 1
            dest = self.journal_path + f".quarantined-{n}"
        os.replace(self.journal_path, dest)
        with open(dest + ".reason", "w", encoding="utf-8") as f:
            f.write(reason + "\n")
            f.flush()
            os.fsync(f.fileno())
        return dest

    # -- writing ----------------------------------------------------------

    def create(self, identity: str, identity_detail: dict) -> dict:
        """Start a fresh generation (rejects an existing live journal)."""
        if os.path.exists(self.journal_path):
            raise BatchCheckpointError(
                f"checkpoint directory {self.directory!r} already holds a "
                f"journal; resume must go through load_resume() so changed "
                f"identity is rejected instead of overwritten")
        doc = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "identity": identity,
            "identity_detail": _canonical(identity_detail),
            "generation": 1,
            "rows": {},
            "composite": {"phase": "staged"},
        }
        self._write(doc)
        return doc

    def record_row(self, row_index: int, record: JournalRow) -> None:
        doc = self._require()
        rows = dict(doc["rows"])
        rows[str(row_index)] = _canonical(dict(record))
        self._write({**doc, "rows": rows})

    def record_composite_finalized(self) -> None:
        doc = self._require()
        self._write({**doc,
                     "composite": {"phase": "finalized"},
                     "generation": doc["generation"] + 1})

    def _require(self) -> dict:
        doc = self.load()
        if doc is None:
            raise BatchCheckpointError(
                "checkpoint journal missing while recording progress")
        return doc

    def _write(self, doc: dict) -> None:
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.captures_dir, exist_ok=True)
        self._cache = doc
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".journal-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.journal_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._fsync_dir(self.directory)

    @staticmethod
    def _fsync_dir(path: str) -> None:
        try:
            dfd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)

    # -- capture payloads (raw .npy only; digests always verified) --------

    def capture_path(self, row_index: int, column: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "._-") else "_"
                       for c in column)
        return os.path.join(
            self.captures_dir, f"row{row_index}__{safe}.npy")

    def write_capture(self, row_index: int, column: str, npy_bytes: bytes) -> dict:
        path = self.capture_path(row_index, column)
        os.makedirs(self.captures_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.captures_dir, prefix=".cap-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(npy_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self._fsync_dir(self.captures_dir)
        return {"path": os.path.basename(path), "sha256": sha256_bytes(npy_bytes),
                "bytes": len(npy_bytes)}

    def read_capture(self, descriptor: dict) -> bytes:
        path = os.path.join(self.captures_dir, descriptor["path"])
        if not os.path.exists(path):
            raise BatchCheckpointError(
                f"capture file {descriptor['path']!r} recorded in the "
                f"journal is missing; generation cannot be trusted")
        with open(path, "rb") as f:
            data = f.read()
        if sha256_bytes(data) != descriptor["sha256"] \
                or len(data) != descriptor["bytes"]:
            raise BatchCheckpointError(
                f"capture file {descriptor['path']!r} fails its recorded "
                f"digest; quarantining generation")
        return data

    # -- resume verification ----------------------------------------------

    def verify_identity(self, identity: str) -> None:
        doc = self.load()
        if doc is None:
            return
        if doc["identity"] != identity:
            raise BatchCheckpointError(
                "checkpoint identity mismatch: the recorded code/model/"
                "profile/input digests do not match this call — dossier 05 "
                "rejects changed identity rather than mixing generations "
                f"(recorded {doc['identity'][:12]}…, requested "
                f"{identity[:12]}…)")

    def verify_row_artifacts(self, row_index: int,
                             expected_public: Optional[str] = None) -> bool:
        """True when the row's committed record exists and every recorded
        artifact verifies against its digest (size + sha256)."""
        doc = self.load()
        if doc is None:
            return False
        record = doc["rows"].get(str(row_index))
        if not isinstance(record, dict):
            return False
        for relpath, meta in (record.get("files") or {}).items():
            path = relpath if os.path.isabs(relpath) \
                else os.path.join(expected_public or "", relpath)
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) != meta["bytes"]:
                return False
            if sha256_file(path) != meta["sha256"]:
                return False
        return True

    def committed_row_indexes(self) -> set:
        doc = self.load()
        if doc is None:
            return set()
        return {int(k) for k, v in doc["rows"].items()
                if isinstance(v, dict) and v.get("phase") in
                ("COMMITTED", "SKIPPED")}

    def verify_row_capture(self, row_index: int) -> bool:
        """True when the row's recorded capture payload (if any) exists and
        verifies against its recorded digest (size + sha256).  Rows whose
        capture resumables are missing/tampered must be re-run, not
        folded from an unverified payload at fold time."""
        doc = self.load()
        if doc is None:
            return False
        record = doc["rows"].get(str(row_index))
        if not isinstance(record, dict):
            return False
        cap = record.get("capture")
        if not cap:
            return True
        path = os.path.join(self.captures_dir, cap.get("path", ""))
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) != cap.get("bytes"):
            return False
        return sha256_file(path) == cap.get("sha256")

    def composite_finalized(self) -> bool:
        doc = self.load()
        return bool(doc and doc.get("composite", {}).get("phase") == "finalized")
