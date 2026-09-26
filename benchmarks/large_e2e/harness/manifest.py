"""Workload manifest loading, strict validation and input rehashing.

A manifest is the frozen scientific workload description.  Structural
validation happens coordinator-side before any process is spawned; Settings
key validation happens against the arm identity's recorded Settings field
list (identity.settings_fields), so no engine import is needed here and the
check cannot drift from the actual arm package.

Known schema (schema_version 1):

{
  "schema_version": 1,
  "workload_id": "nonempty string",
  "analysis": "accessibility" | "flow" | "odm",
  "input_files": [{"path": str, "sha256": "<64 hex>", "bytes": int>0}, ...],
  "settings": { ...Settings overrides (no output_folder)... },
  "required_output_patterns": ["*.feather", ...],     # >= 1 entry
  "odm": {"format": str, "speed": float},             # only when analysis=odm
  "notes": "optional string"
}

Relative input paths resolve against the manifest's own directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harness.spec import ValidationError

MANIFEST_SCHEMA_VERSION = 1
ALLOWED_TOP_KEYS = {
    "schema_version", "workload_id", "analysis", "input_files",
    "settings", "required_output_patterns", "odm", "notes",
}
REQUIRED_SETTINGS_KEYS = ("data_folder",)
FORBIDDEN_SETTINGS_KEYS = ("output_folder",)


class Manifest:
    def __init__(self, data: dict, path: Path) -> None:
        self.data = data
        self.path = path
        self.workload_id: str = data["workload_id"]
        self.analysis: str = data["analysis"]
        self.settings: dict = data["settings"]
        self.required_output_patterns: list[str] = data["required_output_patterns"]
        self.odm_options: dict = data.get("odm") or {}
        self.notes: str = data.get("notes", "")
        self.input_files: list[InputFile] = [
            InputFile(entry, path) for entry in data["input_files"]
        ]
        self.manifest_sha256 = sha256_file(path)

    def input_dirs(self) -> list[Path]:
        seen: dict[str, Path] = {}
        for entry in self.input_files:
            seen[str(entry.resolved.parent)] = entry.resolved.parent
        return list(seen.values())

    def total_input_bytes(self) -> int:
        return sum(e.expected_bytes for e in self.input_files)

    def describe(self) -> dict:
        return {
            "workload_id": self.workload_id,
            "analysis": self.analysis,
            "manifest_path": str(self.path),
            "manifest_sha256": self.manifest_sha256,
            "settings": self.settings,
            "required_output_patterns": self.required_output_patterns,
            "odm_options": self.odm_options,
            "input_files": [e.describe() for e in self.input_files],
            "total_input_bytes": self.total_input_bytes(),
            "notes": self.notes,
        }

    # ------------------------------------------------------------------
    def rehash_inputs(self) -> dict:
        """Rehash every input file against the frozen manifest (behavior 8).

        Filenames alone are insufficient: bytes are hashed and compared.
        Returns a report; raises ValidationError on the first mismatch.
        """
        report = {"files": [], "all_match": True}
        for entry in self.input_files:
            actual_bytes, actual_sha = hash_file(entry.resolved)
            ok = (actual_bytes == entry.expected_bytes
                  and actual_sha == entry.expected_sha256)
            report["files"].append({
                "path": str(entry.resolved),
                "expected_bytes": entry.expected_bytes,
                "expected_sha256": entry.expected_sha256,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_sha,
                "match": ok,
            })
            if not ok:
                report["all_match"] = False
        if not report["all_match"]:
            bad = [f["path"] for f in report["files"] if not f["match"]]
            raise ValidationError(
                "input rehash mismatch for mutated inputs "
                f"(cached filename with changed bytes): {bad}")
        return report


class InputFile:
    def __init__(self, entry: dict, manifest_path: Path) -> None:
        if not isinstance(entry, dict):
            raise ValidationError(f"manifest input_files entries must be objects: {entry!r}")
        missing = {"path", "sha256", "bytes"} - set(entry)
        if missing:
            raise ValidationError(f"manifest input entry missing keys {sorted(missing)}: {entry!r}")
        raw_path = Path(str(entry["path"])).expanduser()
        if not raw_path.is_absolute():
            raw_path = manifest_path.parent / raw_path
        self.resolved = raw_path.resolve()
        self.expected_sha256 = str(entry["sha256"]).lower()
        if (len(self.expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.expected_sha256)):
            raise ValidationError(f"input sha256 is not 64 hex chars: {entry['sha256']!r}")
        self.expected_bytes = entry["bytes"]
        if not isinstance(self.expected_bytes, int) or self.expected_bytes <= 0:
            raise ValidationError(f"input bytes must be a positive integer: {entry['bytes']!r}")

    def describe(self) -> dict:
        return {
            "path": str(self.resolved),
            "sha256": self.expected_sha256,
            "bytes": self.expected_bytes,
        }


# ----------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file (hash_file returns (bytes, hex))."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_file(path: Path) -> tuple[int, str]:
    """(size_bytes, sha256_hex) of one file; missing files hash as (0, '')."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0, ""
    return size, sha256_file(path)


def load_manifest(path: Path, settings_fields: set[str] | None = None) -> Manifest:
    """Load + strictly validate a manifest; raises ValidationError.

    settings_fields: the arm identity's recorded Settings field names; when
    given, every manifest settings key must be one of them (unknown keys are
    errors, never silently ignored).
    """
    if not path.exists() or not path.is_file():
        raise ValidationError(f"manifest does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"manifest is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise ValidationError("manifest must be a JSON object")

    unknown = set(data) - ALLOWED_TOP_KEYS
    if unknown:
        raise ValidationError(f"manifest has unknown top-level keys: {sorted(unknown)}")
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValidationError(
            f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION}, "
            f"got {data.get('schema_version')!r}")
    if not isinstance(data.get("workload_id"), str) or not data["workload_id"].strip():
        raise ValidationError("manifest workload_id must be a nonempty string")
    if "analysis" not in data:
        raise ValidationError("manifest is missing 'analysis'")

    # Explicit analysis validation BEFORE any work; no default to accessibility.
    from harness.dispatch import validate_analysis, SUPPORTED_ANALYSES
    try:
        validate_analysis(data["analysis"])
    except Exception as exc:
        raise ValidationError(str(exc))
    if data["analysis"] != data["analysis"].strip().lower():
        raise ValidationError(
            f"manifest analysis must be one of {list(SUPPORTED_ANALYSES)} (lowercase)")

    if not isinstance(data.get("input_files"), list) or not data["input_files"]:
        raise ValidationError("manifest input_files must be a nonempty list")
    if not isinstance(data.get("settings"), dict) or not data["settings"]:
        raise ValidationError("manifest settings must be a nonempty object")
    for key in FORBIDDEN_SETTINGS_KEYS:
        if key in data["settings"]:
            raise ValidationError(
                f"manifest settings may not set {key!r}: the harness owns "
                "per-job output folders")
    for key in REQUIRED_SETTINGS_KEYS:
        value = data["settings"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"manifest settings.{key} must be a nonempty string")
    if settings_fields is not None:
        unknown_keys = set(data["settings"]) - set(settings_fields)
        if unknown_keys:
            raise ValidationError(
                f"manifest settings keys not in arm Settings fields: "
                f"{sorted(unknown_keys)}")
    data_folder = Path(data["settings"]["data_folder"]).expanduser().resolve()
    if not data_folder.is_dir():
        raise ValidationError(f"manifest settings.data_folder is not a directory: {data_folder}")

    patterns = data.get("required_output_patterns")
    if not isinstance(patterns, list) or not patterns or \
            not all(isinstance(p, str) and p.strip() for p in patterns):
        raise ValidationError(
            "manifest required_output_patterns must be a nonempty list of globs")

    odm = data.get("odm")
    if data["analysis"] == "odm":
        if odm is not None and not isinstance(odm, dict):
            raise ValidationError("manifest odm must be an object")
        if odm:
            if "format" in odm and not isinstance(odm["format"], str):
                raise ValidationError("manifest odm.format must be a string")
            if "speed" in odm and not isinstance(odm["speed"], (int, float)):
                raise ValidationError("manifest odm.speed must be a number")
    elif odm is not None:
        raise ValidationError("manifest odm is only allowed when analysis == 'odm'")

    manifest = Manifest(data, path)
    # Every declared input must currently exist (existence is pre-work).
    for entry in manifest.input_files:
        if not entry.resolved.is_file():
            raise ValidationError(f"manifest input file missing: {entry.resolved}")
    return manifest
