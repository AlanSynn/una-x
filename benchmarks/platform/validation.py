"""Output obligation validation and bitwise comparison (HARNESS.md
"Validation isolation"; controls 4 and the deterministic-binding leg).

Arrays are compared as dtype+shape+bytes; geometries as WKB bytes+CRS
handled by the obligation schema.  Empty comparison sets are failures, not
passes.  Baseline determinism is established before any export judgment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .errors import (BitwiseMismatch, EmptyComparisonError,
                     OutputValidationError)
from .manifest import sha256_file


def validate_output_obligations(out_dir: str | Path, obligations: dict) -> dict:
    """Every declared output must exist, be non-empty, and satisfy the
    declared structural expectations.  Returns the output manifest."""
    out_dir = Path(out_dir)
    manifest = {}
    for name, spec in (obligations.get("files") or {}).items():
        p = out_dir / name
        if not p.is_file():
            raise OutputValidationError(
                f"declared output {name} missing in {out_dir} (control 4)")
        size = p.stat().st_size
        if size <= 0:
            raise OutputValidationError(f"declared output {name} is empty")
        if "min_bytes" in spec and size < spec["min_bytes"]:
            raise OutputValidationError(
                f"{name} {size}B < min_bytes {spec['min_bytes']}")
        fmt = spec.get("format")
        if fmt == "geojson":
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise OutputValidationError(f"{name}: invalid geojson ({exc})")
            feats = doc.get("features", [])
            if "expect_features" in spec and len(feats) != spec["expect_features"]:
                raise OutputValidationError(
                    f"{name}: {len(feats)} features != declared "
                    f"{spec['expect_features']}")
            manifest[name] = {"bytes": size, "sha256": sha256_file(p),
                              "features": len(feats)}
        elif fmt == "feather":
            import pandas as pd
            try:
                df = pd.read_feather(p)
            except Exception as exc:
                raise OutputValidationError(f"{name}: unreadable feather ({exc})")
            if "expect_rows" in spec and len(df) != spec["expect_rows"]:
                raise OutputValidationError(
                    f"{name}: {len(df)} rows != declared "
                    f"{spec['expect_rows']}")
            if "expect_columns" in spec and \
                    list(df.columns) != list(spec["expect_columns"]):
                raise OutputValidationError(
                    f"{name}: columns {list(df.columns)} != declared "
                    f"{spec['expect_columns']}")
            manifest[name] = {"bytes": size, "sha256": sha256_file(p),
                              "rows": len(df)}
        else:
            manifest[name] = {"bytes": size, "sha256": sha256_file(p)}
    undeclared = sorted(
        p.name for p in out_dir.iterdir()
        if p.is_file() and p.name not in (obligations.get("files") or {})
        and p.suffix in (".geojson", ".feather", ".csv"))
    return {"validated": manifest, "undeclared": undeclared}


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_files_bitwise(path_a: str | Path, path_b: str | Path,
                          label: str = "") -> None:
    a, b = Path(path_a), Path(path_b)
    if not a.is_file() or not b.is_file():
        raise BitwiseMismatch(f"{label}: missing file(s) {a} / {b}")
    if sha256_file(a) != sha256_file(b):
        raise BitwiseMismatch(
            f"{label}: {a.name} bytes differ (sha256 "
            f"{sha256_file(a)[:12]} != {sha256_file(b)[:12]})")


def compare_dirs_bitwise(dir_a: str | Path, dir_b: str | Path,
                         label: str = "") -> None:
    """All result files in both dirs must be byte-identical, and the set of
    result files must be identical."""
    a, b = Path(dir_a), Path(dir_b)
    files_a = sorted(p.name for p in a.iterdir() if p.is_file())
    files_b = sorted(p.name for p in b.iterdir() if p.is_file())
    if not files_a:
        raise EmptyComparisonError(
            f"{label}: first arm produced no files; empty comparison sets "
            "cannot pass (control 4)")
    if not files_b:
        raise EmptyComparisonError(
            f"{label}: second arm produced no files; empty comparison sets "
            "cannot pass (control 4)")
    if files_a != files_b:
        raise BitwiseMismatch(
            f"{label}: artifact sets differ {files_a} vs {files_b} "
            "(missing export)")
    for name in files_a:
        compare_files_bitwise(a / name, b / name, label)


def compare_frame_bytes_equal(df_a, df_b, label: str) -> None:
    """dtype+shape+bytes equality for two pandas/geometry frames, with
    first-divergence reporting (mirrors the historical comparator)."""
    import numpy as np
    if list(df_a.columns) != list(df_b.columns):
        raise BitwiseMismatch(f"{label}: column sets/order differ")
    if len(df_a) != len(df_b):
        raise BitwiseMismatch(f"{label}: row counts differ "
                              f"{len(df_a)} vs {len(df_b)}")
    for col in df_a.columns:
        ca, cb = df_a[col], df_b[col]
        if str(ca.dtype) != str(cb.dtype):
            raise BitwiseMismatch(
                f"{label}.{col}: dtype {ca.dtype} != {cb.dtype}")
        va, vb = np.asarray(ca), np.asarray(cb)
        if va.shape != vb.shape:
            raise BitwiseMismatch(f"{label}.{col}: shape differs")
        try:
            ba = np.ascontiguousarray(va).tobytes()
            bb = np.ascontiguousarray(vb).tobytes()
        except (TypeError, ValueError):
            ba = None
        if ba is not None:
            if ba != bb:
                flat_a, flat_b = np.ravel(va), np.ravel(vb)
                idx = next((i for i, (x, y) in enumerate(zip(flat_a, flat_b))
                            if x != y), 0)
                raise BitwiseMismatch(
                    f"{label}.{col}: bytes differ at flat index {idx} "
                    f"({flat_a[idx]!r} vs {flat_b[idx]!r})")
        else:
            for i, (x, y) in enumerate(zip(np.ravel(va), np.ravel(vb))):
                if x != y:
                    raise BitwiseMismatch(
                        f"{label}.{col}: value differs at index {i} "
                        f"({x!r} vs {y!r})")
