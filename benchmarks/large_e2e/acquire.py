#!/usr/bin/env python3
"""H01 observed-input acquisition for the UNA large-e2e campaign.

Implements the "Input acquisition" section of
campaigns/una_large_e2e/WORKLOADS.md against the exact public catalog in
campaigns/una_large_e2e/inputs_catalog.json (upstream ref
c15ebda6981397f46eed5c2d55229f71e57d44fb).

Subcommands
-----------
download : acquire catalog files over normal TLS into the local data dir.
           Verifies byte size and Git blob SHA (sha1("blob <len>\\0" + bytes)),
           records SHA-256, then atomically renames into place. Idempotent:
           existing files are rehashed, never overwritten on mismatch.
           Bounded retries, disk preflight before each attempt.
inspect  : record CRS / geometry types / validity / Z availability / columns /
           weight-column findings per file into evidence H01 acquisition.json.
           Observation only - nothing is cleaned, reprojected or simplified.
select   : pinned-RNG origin selection lists (seed 20260925,
           numpy.random.default_rng) written under tests/large_e2e/inputs/.
freeze   : render workload manifests + evidence H01 workloads.json from the
           recorded facts (hashes, counts, Settings.ToDict captures).

Downloaded bulk data lives OUTSIDE the repository by policy (default
<repo>/../campaign_data/inputs). Nothing here republishes data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "campaigns" / "una_large_e2e" / "inputs_catalog.json"
DEFAULT_EVIDENCE = REPO_ROOT / "campaigns" / "una_large_e2e" / "evidence" / "H01"
DEFAULT_TESTS_INPUTS = REPO_ROOT / "tests" / "large_e2e" / "inputs"
# Bulk data must live outside the repo checkout (WORKLOADS.md / H01 policy).
DEFAULT_DATA_DIR = REPO_ROOT.parent / "campaign_data" / "inputs"
DEFAULT_SRC = REPO_ROOT / "src"

SEED = 20260925
# selection-file stem -> requested k (None = all origin rows).
# Each derived fixture gets its own selection manifest, even when the
# requested set is "all rows" for more than one workload.
ORIGIN_VARIANTS = {
    "O2": 16,
    "O3_FLOW_sel256": 256,
    "O3_FLOW_sel1024": 1024,
    "O3_FLOW_sel_all": None,
    "O3_ACCESS_sel_all": None,
    "O3_HOLDOUT_sel_all": None,
}
HTTP_TIMEOUT_S = 180
READ_CHUNK = 1 << 20
MAX_RETRIES = 3
RETRY_BACKOFF_S = (2.0, 5.0, 10.0)
USER_AGENT = "una-x-campaign-H01-acquire/1.0 (local benchmark fixture acquisition)"
DISK_HEADROOM_BYTES = 64 << 20  # 64 MiB slack on top of 2x expected file size


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def git_blob_sha(data: bytes) -> str:
    """Git blob object name of a byte string (git hash-object equivalent)."""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(READ_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_catalog(path: Path) -> dict:
    cat = json.loads(path.read_text(encoding="utf-8"))
    for key in ("repository", "ref", "classification", "files"):
        if key not in cat:
            raise SystemExit(f"catalog missing required key: {key}")
    for entry in cat["files"]:
        for key in ("id", "path", "git_blob_sha", "bytes", "url"):
            if key not in entry:
                raise SystemExit(f"catalog entry missing required key: {key} ({entry})")
    return cat


def load_acquisition(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "task": "H01",
        "campaign": "una_large_e2e",
        "catalog": None,
        "tool": {},
        "data_dir": None,
        "files": {},
    }


def write_acquisition(path: Path, acq: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(acq), encoding="utf-8")


def tool_facts(script_path: Path) -> dict:
    facts = {
        "script": str(script_path.relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(script_path),
        "python": sys.version.split()[0],
    }
    try:
        import numpy
        facts["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import geopandas
        import pyogrio
        facts["geopandas"] = geopandas.__version__
        facts["pyogrio"] = pyogrio.__version__
        facts["gdal"] = pyogrio.__gdal_version_string__
    except ImportError:
        pass
    return facts


def verify_local(path: Path, entry: dict) -> dict:
    """Rehash an existing file against the catalog entry (idempotent)."""
    size = path.stat().st_size
    digest = sha256_file(path)
    blob = None
    if size == entry["bytes"]:
        blob = git_blob_sha(path.read_bytes())  # exact-size guard keeps this bounded
    return {
        "local_path": str(path),
        "bytes_on_disk": size,
        "sha256": digest,
        "blob_sha_computed": blob,
        "blob_sha_expected": entry["git_blob_sha"],
        "bytes_expected": entry["bytes"],
        "size_match": size == entry["bytes"],
        "blob_sha_match": blob == entry["git_blob_sha"] if blob else False,
    }


def disk_preflight(dest_dir: Path, needed_bytes: int) -> None:
    free = shutil.disk_usage(dest_dir).free
    if free < needed_bytes:
        raise SystemExit(
            f"disk preflight failed: {free} bytes free in {dest_dir}, "
            f"need {needed_bytes}"
        )


# ────────────────────────────────────────────────────────────────────────
# download
# ────────────────────────────────────────────────────────────────────────

def download_one(entry: dict, dest_dir: Path) -> tuple[dict, int]:
    """Acquire one catalog file. Returns (record, bytes_downloaded_this_run)."""
    final = dest_dir / Path(entry["path"]).name
    rec: dict = {
        "id": entry["id"],
        "repo_path": entry["path"],
        "url": entry["url"],
        "default_required": bool(entry.get("default_required", False)),
        "attribution": {
            "repository": entry.get("repository"),
            "ref": entry.get("ref"),
            "upstream_path": entry["path"],
            "note": entry.get("redistribution") or entry.get("attribution"),
        },
    }
    if final.exists():
        facts = verify_local(final, entry)
        rec["existing_file"] = facts
        rec["bytes_on_disk"] = facts["bytes_on_disk"]
        rec["sha256"] = facts["sha256"]
        if facts["blob_sha_match"] and facts["size_match"]:
            rec["status"] = "already_present_verified"
            return rec, 0
        rec["status"] = "blocked_existing_mismatch"
        rec["error"] = (
            f"existing file {final} does not match catalog "
            f"(size_match={facts['size_match']}, "
            f"blob_sha_match={facts['blob_sha_match']}); policy forbids overwrite"
        )
        return rec, 0

    errors = []
    limit = entry["bytes"]
    for attempt in range(1, MAX_RETRIES + 1):
        # temp + final + slack, checked before every attempt
        disk_preflight(dest_dir, 2 * limit + DISK_HEADROOM_BYTES)
        tmp = dest_dir / f".{final.name}.tmp-{os.getpid()}"
        started = time.monotonic()
        try:
            req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
            n = 0
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                with open(tmp, "wb") as out:
                    while True:
                        chunk = resp.read(READ_CHUNK)
                        if not chunk:
                            break
                        n += len(chunk)
                        if n > limit:
                            raise IOError(
                                f"byte-size limit exceeded: >{limit} catalog bytes"
                            )
                        out.write(chunk)
            if n != limit:
                raise IOError(f"byte size mismatch: got {n}, catalog says {limit}")
            blob = git_blob_sha(tmp.read_bytes())
            if blob != entry["git_blob_sha"]:
                tmp.unlink(missing_ok=True)
                raise IOError(
                    f"git blob SHA mismatch: got {blob}, "
                    f"catalog says {entry['git_blob_sha']}"
                )
            digest = sha256_file(tmp)
            os.replace(tmp, final)  # atomic publish after verification
            rec["status"] = "downloaded_verified"
            rec["bytes_downloaded"] = n
            rec["bytes_on_disk"] = final.stat().st_size
            rec["sha256"] = digest
            rec["blob_sha_computed"] = blob
            rec["acquired_utc"] = utc_now()
            rec["attempts"] = errors + [{
                "attempt": attempt, "ok": True,
                "elapsed_s": round(time.monotonic() - started, 3),
            }]
            return rec, n
        except Exception as exc:  # noqa: BLE001 - recorded verbatim, retried bounded
            tmp.unlink(missing_ok=True)
            errors.append({
                "attempt": attempt,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - started, 3),
            })
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)])
    rec["status"] = "blocked_download_failed"
    rec["error"] = f"all {MAX_RETRIES} attempts failed"
    rec["attempts"] = errors
    return rec, 0


def cmd_download(args) -> int:
    catalog = load_catalog(args.catalog)
    dest_dir = args.dest.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = [e for e in catalog["files"] if e.get("url")]
    if args.ids:
        want = set(args.ids)
        entries = [e for e in entries if e["id"] in want]
        missing = want - {e["id"] for e in entries}
        if missing:
            raise SystemExit(f"unknown catalog ids: {sorted(missing)}")

    acq_path = args.evidence / "acquisition.json"
    acq = load_acquisition(acq_path)
    acq["catalog"] = {
        "path": str(args.catalog),
        "repository": catalog["repository"],
        "ref": catalog["ref"],
        "classification": catalog["classification"],
        "redistribution": catalog.get("redistribution"),
        "metadata_source_tree": catalog.get("metadata_source_tree"),
    }
    acq["tool"] = {**acq.get("tool", {}), **tool_facts(Path(__file__).resolve())}
    acq["data_dir"] = str(dest_dir)

    total_downloaded = 0
    hard_failures = []
    for entry in entries:
        rec, n = download_one(entry, dest_dir)
        total_downloaded += n
        prev = acq["files"].get(entry["id"], {})
        merged = {k: v for k, v in prev.items() if k != "download"}
        merged["download"] = rec
        acq["files"][entry["id"]] = merged
        print(f"[{rec['status']}] {entry['id']}: "
              f"{rec.get('bytes_downloaded', rec.get('bytes_on_disk', 0))} bytes "
              f"sha256={rec.get('sha256', '-')}")
        if rec["status"].startswith("blocked"):
            if entry.get("default_required"):
                hard_failures.append(entry["id"])
            else:
                print(f"  WARNING: optional file {entry['id']} blocked: "
                      f"{rec.get('error')}")

    acq["totals"] = {
        "bytes_downloaded_this_run": total_downloaded,
        "bytes_on_disk": sum(
            f.get("download", {}).get("bytes_on_disk", 0)
            for f in acq["files"].values()
        ),
    }
    write_acquisition(acq_path, acq)
    print(f"evidence: {acq_path}")
    if hard_failures:
        print(f"BLOCKED required files: {hard_failures}", file=sys.stderr)
        return 2
    return 0


# ────────────────────────────────────────────────────────────────────────
# inspect
# ────────────────────────────────────────────────────────────────────────

def _pyogrio_info(path: Path) -> dict:
    import pyogrio
    raw = pyogrio.read_info(str(path))
    keep = {}
    for key in ("driver", "layer_name", "crs", "fields", "dtypes",
                "geometry_type", "features"):
        if key in raw:
            val = raw[key]
            keep[key] = val.tolist() if hasattr(val, "tolist") else val
    return keep


def _series_stats(values) -> dict:
    import numpy as np
    try:
        arr = np.asarray(values, dtype="float64")
    except (TypeError, ValueError) as exc:
        return {"error": f"not numeric-castable: {type(exc).__name__}: {exc}",
                "n": int(len(values))}
    finite = np.isfinite(arr)
    return {
        "n": int(arr.size),
        "n_null": int(np.count_nonzero(np.isnan(arr))),
        "n_nonfinite": int(np.count_nonzero(~finite)),
        "n_nonpositive": int(np.count_nonzero(finite & (arr <= 0.0))),
        "n_zero": int(np.count_nonzero(arr == 0.0)),
        "min": float(arr[finite].min()) if finite.any() else None,
        "max": float(arr[finite].max()) if finite.any() else None,
        "sum": float(arr[finite].sum()) if finite.any() else None,
    }


def _geometry_facts(gdf) -> dict:
    import numpy as np
    import shapely
    geoms = gdf.geometry.array
    missing = np.asarray(shapely.is_missing(geoms), dtype=bool)
    valid = np.asarray(shapely.is_valid(geoms), dtype=bool)
    empty = np.asarray(shapely.is_empty(geoms), dtype=bool)
    has_z = np.asarray(shapely.has_z(geoms), dtype=bool)
    present = ~missing
    types = gdf.geometry.geom_type.value_counts(dropna=False)
    crs = gdf.crs
    return {
        "n_rows": int(len(gdf)),
        "geometry_types": {str(k): int(v) for k, v in types.items()},
        "n_missing_geometry": int(missing.sum()),
        "n_invalid_geometry": int(np.count_nonzero(present & ~valid)),
        "n_empty_geometry": int(np.count_nonzero(present & empty)),
        "n_with_z": int(np.count_nonzero(present & has_z)),
        "n_without_z": int(np.count_nonzero(present & ~has_z)),
        "crs": None if crs is None else crs.to_string(),
        "crs_epsg": None if crs is None else crs.to_epsg(),
    }


def _column_facts(gdf) -> dict:
    cols = [c for c in gdf.columns if c != "geometry"]
    return {
        "columns": cols,
        "dtypes": {c: str(gdf[c].dtype) for c in cols},
        "null_or_nan_counts": {
            c: int(gdf[c].isna().sum()) for c in cols
        },
    }


def _role_facts(role: str, gdf) -> dict:
    """Role-specific weight/cost/ID inspection. Observation only."""
    facts: dict = {}
    if role == "origins":
        facts["count_column_literal_present"] = "Count" in gdf.columns
        if "Count" in gdf.columns:
            facts["count_column_stats"] = _series_stats(gdf["Count"].to_numpy())
        # Baseline semantics (Topology.BuildAccessPoints): cost_attribute=="Count"
        # is the unit-weights sentinel; the attribute column is NOT required.
        facts["baseline_semantics"] = (
            "Settings.origin_weight_column='Count' is baseline's unit-weights "
            "sentinel (Topology.BuildAccessPoints builds np.ones); a literal "
            "'Count' attribute is not required by the engine"
        )
    elif role == "destinations":
        col = "weekly_departures"
        facts["weekly_departures_present"] = col in gdf.columns
        if col in gdf.columns:
            stats = _series_stats(gdf[col].to_numpy())
            facts["weekly_departures_stats"] = stats
            if stats.get("n_null") or stats.get("n_nonfinite") or stats.get("n_nonpositive"):
                facts["weekly_departures_quality"] = (
                    "WARNING: null/nonfinite/nonpositive values present "
                    "(recorded, not cleaned)"
                )
            else:
                facts["weekly_departures_quality"] = "all values finite and positive"
        else:
            facts["error"] = (
                "required destination weight column 'weekly_departures' is "
                "missing from the observed file; per WORKLOADS.md this is an "
                "error, not a guessed substitution"
            )
    elif role in ("network", "network_3d"):
        import numpy as np
        import shapely
        facts["geometric_length_stats"] = _series_stats(gdf.geometry.length.to_numpy())
        facts["node_id_columns_present"] = {
            "_node_start_id": "_node_start_id" in gdf.columns,
            "_node_end_id": "_node_end_id" in gdf.columns,
        }
        # Z observation: planar length ignores Z, so record the actual third
        # coordinates (all-Z files may still carry a constant, e.g. 0).
        if bool(np.asarray(shapely.has_z(gdf.geometry.array), dtype=bool).any()):
            zc = shapely.get_coordinates(gdf.geometry.array, include_z=True)[:, 2]
            facts["z_value_stats"] = _series_stats(zc)
            facts["z_value_stats"]["n_distinct"] = int(np.unique(zc).size)
        facts["baseline_semantics"] = (
            "network_weight_column='Geometric' uses geometry.length at load; "
            "zero-length segments are recorded here but NOT cleaned"
        )
        if role == "network_3d":
            facts["z_note"] = (
                "n_with_z counts features whose coordinates carry a third "
                "dimension; the elevation engine consumes per-node Z "
                "(cost += coefficient * max(0, z_end - z_start)), so a "
                "constant-Z file would make elevation=True a no-op"
            )
    elif role == "observers":
        facts["baseline_semantics"] = (
            "observers are optional passive flow counters (flow only); "
            "observer_points_snap_to='edge' default"
        )
    return facts


def cmd_inspect(args) -> int:
    import geopandas as gpd

    catalog = load_catalog(args.catalog)
    acq_path = args.evidence / "acquisition.json"
    acq = load_acquisition(acq_path)
    acq["tool"] = {**acq.get("tool", {}), **tool_facts(Path(__file__).resolve())}

    ids = args.ids or [e["id"] for e in catalog["files"]]
    failures = []
    for fid in ids:
        entry = next(e for e in catalog["files"] if e["id"] == fid)
        local = args.dest / Path(entry["path"]).name
        if not local.exists():
            print(f"[missing] {fid}: {local}", file=sys.stderr)
            failures.append(fid)
            continue
        schema: dict = {"inspected_utc": utc_now(), "file": verify_local(local, entry)}
        schema["pyogrio_read_info"] = _pyogrio_info(local)
        # One full read per invocation (run the CLI once per big file to keep
        # peak memory bounded); nothing is modified.
        gdf = gpd.read_file(local)
        schema["geometry"] = _geometry_facts(gdf)
        schema["columns"] = _column_facts(gdf)
        schema["role_facts"] = _role_facts(fid, gdf)
        if schema["geometry"]["n_rows"] != schema["pyogrio_read_info"].get("features"):
            schema["error"] = (
                f"row-count disagreement: full read {schema['geometry']['n_rows']} "
                f"vs pyogrio info {schema['pyogrio_read_info'].get('features')}"
            )
            failures.append(fid)
        if "error" in schema.get("role_facts", {}):
            failures.append(fid)
        acq["files"].setdefault(fid, {})["schema"] = schema
        print(f"[inspected] {fid}: rows={schema['geometry']['n_rows']} "
              f"crs={schema['geometry']['crs']} "
              f"geom={schema['geometry']['geometry_types']} "
              f"z={schema['geometry']['n_with_z']}")
        del gdf

    write_acquisition(acq_path, acq)
    print(f"evidence: {acq_path}")
    return 1 if failures else 0


# ────────────────────────────────────────────────────────────────────────
# select
# ────────────────────────────────────────────────────────────────────────

def _jsonable(v):
    import numpy as np
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        if f != f:
            return "NaN"  # observed non-finite data, JSON-strict serialization
        if f in (float("inf"), float("-inf")):
            return "Infinity" if f > 0 else "-Infinity"
        return f
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v in (float("inf"), float("-inf")):
            return "Infinity" if v > 0 else "-Infinity"
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _origins_entry(args) -> dict:
    acq = load_acquisition(args.evidence / "acquisition.json")
    entry = acq["files"].get("origins")
    if not entry or "schema" not in entry:
        raise SystemExit("run `inspect --ids origins` before `select`")
    # Independent re-derivation; disagreement is fatal (no silent trust).
    import pyogrio
    local = Path(entry["schema"]["file"]["local_path"])
    n_manifest = entry["schema"]["geometry"]["n_rows"]
    n_now = pyogrio.read_info(str(local))["features"]
    if n_now != n_manifest:
        raise SystemExit(
            f"origins feature count changed between inspect ({n_manifest}) "
            f"and select ({n_now}); refusing to select"
        )
    return entry


def cmd_select(args) -> int:
    import numpy as np

    entry = _origins_entry(args)
    local = Path(entry["schema"]["file"]["local_path"])
    n = entry["schema"]["geometry"]["n_rows"]
    src_sha = entry["schema"]["file"]["sha256"]
    out_dir = args.tests_inputs
    out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": 1,
        "task": "H01",
        "seed": SEED,
        "numpy_version": np.__version__,
        "procedure": (
            "Per variant, with a FRESH generator: rng = numpy.random.default_rng"
            "(20260925); draw = rng.choice(n_origins, size=k, replace=False); "
            "ordered = numpy.sort(draw). The ordered list is the derived fixture "
            "(original row order of the origins layer); the raw draw order is "
            "kept only for reproduction audit. Variants are independent "
            "fresh-seed draws so each list is reproducible on its own."
        ),
        "source": {"repo_path": "docs/Boston/Cambridge_building_centroids.geojson",
                   "sha256": src_sha, "feature_count": n},
        "variants": {},
    }

    # Only the small O2 selection carries source attribute rows; the big
    # lists stay index-only (no bulk data in the repo).
    attr_rows = None
    if "O2" in ORIGIN_VARIANTS:
        import geopandas as gpd
        gdf = gpd.read_file(local)
        cols = [c for c in gdf.columns if c != "geometry"]

        def attr_rows_for(indices):
            return [{"row_index": int(i),
                     **{c: _jsonable(gdf.at[int(i), c]) for c in cols}}
                    for i in indices]
    else:
        def attr_rows_for(indices):
            return None

    for stem, k_req in ORIGIN_VARIANTS.items():
        k = n if k_req is None else k_req
        if k > n:
            raise SystemExit(f"variant {stem}: k={k} > n={n}")
        rng = np.random.default_rng(SEED)
        draw = rng.choice(n, size=k, replace=False)
        ordered = np.sort(draw)
        payload = {
            "schema_version": 1,
            "selection_of": "origins",
            "workload_variant": stem,
            "source_repo_path": "docs/Boston/Cambridge_building_centroids.geojson",
            "source_sha256": src_sha,
            "source_feature_count": n,
            "seed": SEED,
            "numpy_version": np.__version__,
            "rng_calls": [
                f"rng = numpy.random.default_rng({SEED})",
                f"draw = rng.choice({n}, size={k}, replace=False)",
                "ordered = numpy.sort(draw)",
            ],
            "requested_k": k_req if k_req is not None else "all",
            "selected_count": int(k),
            "ordered_indices": [int(i) for i in ordered],
            "raw_draw_order": [int(i) for i in draw],
        }
        rows = attr_rows_for(ordered) if stem == "O2" else None
        if rows is not None:
            payload["selected_row_attributes"] = rows
            payload["attributes_note"] = (
                "verbatim source attribute values of the selected rows "
                "(no ID rewriting); non-finite floats are serialized as the "
                "strings 'NaN'/'Infinity'/'-Infinity' to keep the JSON strict"
            )
        path = out_dir / f"{stem}.origin_indices.json"
        path.write_text(pretty_json(payload), encoding="utf-8")
        digest = sha256_file(path)
        index["variants"][stem] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": digest,
            "requested_k": k_req if k_req is not None else "all",
            "selected_count": int(k),
            "first_ordered": int(ordered[0]),
            "last_ordered": int(ordered[-1]),
        }
        print(f"[selected] {stem}: k={k} -> {path.name} sha256={digest}")
        del payload

    (args.evidence / "selection.json").write_text(
        pretty_json(index), encoding="utf-8")
    print(f"evidence: {args.evidence / 'selection.json'}")
    return 0


# ────────────────────────────────────────────────────────────────────────
# freeze
# ────────────────────────────────────────────────────────────────────────

BASE_OVERRIDES = {
    "network_weight_column": "Geometric",
    "origin_weight_column": "Count",
    "destination_weight_column": "weekly_departures",
    "search_radius": 500,
    "turns": False,
    "elevation": False,
    "calculate_reach": True,
    "calculate_exponential_gravity": True,
    "calculate_logistic_gravity": True,
    "calculate_knn_access": True,
    "output_csv": True,
    "output_geojson": True,
    "output_feather": True,
}

FLOW_OVERRIDES = {
    "flow_engine": "aggregate_flow",
    "flow_decay": True,
    "flow_decay_method": "gravity_cap",
    "flow_gravity_cap": "p95",
}

READINESS_BLOCKERS_COMMON = [
    "V/E counts pending a baseline Topology load pass (H05); a null required "
    "field means not ready for qualification",
    "baseline wheel not built yet (W00); timed runs require a clean installed "
    "environment",
    "numerical stripe/thread profile null until frozen (H05/S01)",
]


def _settings_dict(overrides: dict, data_dir: str, files: dict) -> tuple[dict, dict, dict]:
    """Build a baseline Settings with overrides.

    Returns (ToDict-capture-before-Validation, validation record, provenance).
    """
    sys.path.insert(0, str(DEFAULT_SRC))
    from urban_network_analysis.Settings import Settings  # baseline source tree

    provenance = {
        "settings_module_loaded_from":
            sys.modules["urban_network_analysis.Settings"].__file__,
    }
    s = Settings()
    applied = dict(overrides)
    applied["data_folder"] = data_dir
    applied["network_file"] = files["network"]
    applied["origins_file"] = files["origins"]
    applied["destinations_file"] = files["destinations"]
    for key, val in applied.items():
        setattr(s, key, val)
    captured = s.ToDict(compact=False)  # BEFORE Validation (it mutates output_folder)
    validation = {"ran": True}
    try:
        s.Validation()
        validation["passed"] = True
        validation["side_effects_note"] = (
            "Settings.Validation() filled output_folder from "
            "data_folder/Results because it was left null; the recorded "
            "settings_before dict is the pre-Validation capture and keeps "
            "output_folder null (the harness assigns a unique per-job root)"
        )
    except Exception as exc:  # noqa: BLE001 - recorded verbatim
        validation["passed"] = False
        validation["error"] = f"{type(exc).__name__}: {exc}"
    return captured, validation, provenance


def _settings_hash(d: dict) -> str:
    return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _input_block(entry: dict, acq_file: dict) -> dict:
    dl = acq_file["download"]
    schema = acq_file.get("schema", {})
    geom = schema.get("geometry", {})
    if "existing_file" in dl:
        blob_match = dl["existing_file"]["blob_sha_match"]
    else:
        blob_match = dl.get("blob_sha_computed") == entry["git_blob_sha"]
    return {
        "id": entry["id"],
        "url": dl["url"],
        "repo_path": dl["repo_path"],
        "git_blob_sha_expected": entry["git_blob_sha"],
        "git_blob_sha_match": blob_match,
        "sha256": schema.get("file", {}).get("sha256", dl.get("sha256")),
        "bytes": schema.get("file", {}).get("bytes_on_disk", dl.get("bytes_on_disk")),
        "local_path": schema.get("file", {}).get("local_path"),
        "crs": geom.get("crs"),
        "geometry_types": geom.get("geometry_types"),
        "feature_count": geom.get("n_rows"),
        "n_with_z": geom.get("n_with_z"),
        "n_invalid_geometry": geom.get("n_invalid_geometry"),
    }


def cmd_freeze(args) -> int:
    catalog = load_catalog(args.catalog)
    acq = load_acquisition(args.evidence / "acquisition.json")
    data_dir = str(args.dest.resolve())

    cat_by_id = {e["id"]: e for e in catalog["files"]}
    for fid in ("network", "origins", "destinations"):
        rec = acq["files"].get(fid, {})
        if "schema" not in rec:
            raise SystemExit(f"missing schema facts for {fid}; run inspect first")
        if not rec["schema"].get("file", {}).get("sha256"):
            raise SystemExit(f"missing hash facts for {fid}")

    def input_block(fid):
        return _input_block(cat_by_id[fid], acq["files"][fid])

    counts = {}
    for fid in ("network", "origins", "destinations"):
        g = acq["files"][fid]["schema"]["geometry"]
        counts[fid] = {"feature_count": g["n_rows"], "crs": g["crs"]}
    dest_facts = acq["files"]["destinations"]["schema"]["role_facts"]
    orig_facts = acq["files"]["origins"]["schema"]["role_facts"]
    net_facts = acq["files"]["network"]["schema"]["role_facts"]

    head = _git_head()
    selection = json.loads(
        (args.evidence / "selection.json").read_text(encoding="utf-8"))

    def sel_variant(stem: str, embed_indices: bool) -> dict:
        v = selection["variants"][stem]
        payload = json.loads((REPO_ROOT / v["path"]).read_text(encoding="utf-8"))
        return {
            "selection_manifest": v["path"],
            "selection_manifest_sha256": v["sha256"],
            "seed": SEED,
            "numpy_version": selection["numpy_version"],
            "requested_k": payload["requested_k"],
            "selected_count": payload["selected_count"],
            "ordered_indices": payload["ordered_indices"] if embed_indices else None,
            "ordered_indices_note": (
                "embedded above" if embed_indices
                else "persisted in the selection manifest (see selection_manifest)"
            ),
        }

    ve_note = (
        "V/E are actual node/arc counts after a baseline Topology load pass "
        "(BuildTopology with network_precision=3 may merge nodes / drop "
        "degenerate arcs); the GeoJSON feature count is recorded separately "
        "as E_geojson_features. pending_load_pass."
    )

    attribution = {
        "repository": catalog["repository"],
        "ref": catalog["ref"],
        "classification": catalog["classification"],
        "license_note": (
            "Upstream code is MIT-licensed; dataset-specific redistribution "
            "terms are not settled by the code license. Data stays local, is "
            "not republished, and attribution follows the catalog: "
            + str(catalog.get("redistribution"))
        ),
    }

    output_policy = {
        "output_root": (
            "unique campaign-owned per-job directory under the run's --out "
            "root; never inside the inputs tree; no two jobs share a root"
        ),
        "output_wStamp": True,
        "output_wStamp_note": (
            "captured unchanged (baseline default) per WORKLOADS timestamp "
            "policy; applied equally in every arm and declared here"
        ),
        "formats_enabled": ["geojson", "feather", "csv"],
        "output_copy_source_data": False,
        "durability": (
            "baseline synchronous writers only; no fsync, atomic publication "
            "or crash-recovery claims beyond baseline"
        ),
        "checkpoint": (
            "none - baseline has no checkpoint/resume; a failed job is re-run "
            "from scratch in a fresh output root"
        ),
        "verification": (
            "outputs re-hashed and compared before a job is counted "
            "successful; verification cost accounted separately from "
            "application time (BENCHMARKS.md)"
        ),
    }

    environment = {
        "reference": "campaigns/una_large_e2e/evidence/H00/resources.json",
        "python": "3.11.16",
        "numpy": "2.4.6",
        "numba": "0.67.0",
        "geopandas": "1.1.4",
        "pyogrio": "0.13.0",
        "shapely": "2.1.2",
        "source_commit_at_freeze": head,
        "source_tree": "this checkout (branch perf/una-large-e2e), unmodified src/",
        "wheel": None,
        "wheel_status": "pending_W00 - baseline wheel not built yet",
        "note": (
            "timed runs happen only in clean installed (noneditable) "
            "environments per BENCHMARKS.md; this manifest freezes data and "
            "settings, not the eventual wheel identity"
        ),
    }

    stripe_profile = None  # frozen later at H05/S01; never claimed as default

    common_meta = {
        "schema_version": 1,
        "task": "H01",
        "campaign": "una_large_e2e",
        "dataset_class": "observed_tutorial_proxy",
        "label_synthetic": False,
        "source": attribution,
        "crs": counts["network"]["crs"],
        "inputs": {
            "network": input_block("network"),
            "origins": input_block("origins"),
            "destinations": input_block("destinations"),
        },
        "data_finding_notes": {
            "origin_weight_column": orig_facts,
            "destination_weight_column": dest_facts,
            "network_cost": net_facts,
            "policy": (
                "missing required weight columns are errors, not guessed "
                "substitutions; nothing in the data was cleaned, reprojected "
                "or simplified"
            ),
        },
        "stripe_thread_profile": stripe_profile,
        "output_policy": output_policy,
        "environment": environment,
        "preprocessing": {
            "script": "benchmarks/large_e2e/acquire.py",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "data_modifications": (
                "none - files are byte-identical to the catalog blobs "
                "(git blob SHA + SHA-256 verified); selection only reads row "
                "indices, never rewrites IDs or geometry"
            ),
        },
    }

    file_names = {
        "network": Path(cat_by_id["network"]["path"]).name,
        "origins": Path(cat_by_id["origins"]["path"]).name,
        "destinations": Path(cat_by_id["destinations"]["path"]).name,
    }
    acc_outputs = [
        "<job_output_root>/accessibility_<YYYY-MM-DD_HHMM>/Results.feather",
        "<job_output_root>/accessibility_<YYYY-MM-DD_HHMM>/Results.geojson",
        "<job_output_root>/accessibility_<YYYY-MM-DD_HHMM>/Results.csv",
    ]

    blocked_common = list(READINESS_BLOCKERS_COMMON)
    dest_column_ok = bool(dest_facts.get("weekly_departures_present", False))
    if not dest_column_ok:
        blocked_common.insert(0, (
            "destination weight column 'weekly_departures' missing from "
            "MA_bus_stops.geojson - observed workloads are blocked (missing "
            "columns are errors, not substitutions)"))

    manifests = {}

    # ---- O2 ---------------------------------------------------------------
    o2_sel = sel_variant("O2", embed_indices=True)
    overrides = dict(BASE_OVERRIDES)
    settings_before, validation, provenance = _settings_dict(
        overrides, data_dir, file_names)
    manifests["O2"] = {
        **common_meta,
        "workload_id": "O2",
        "role": (
            "observed L2 fixture: full observed network + destinations with 16 "
            "deterministically selected origin rows; tests loading/snapping/"
            "geometry/chronology cheaply, NOT a full-scale performance claim"
        ),
        "origin_selection": o2_sel,
        "counts": {
            "V": None, "E": None,
            "E_geojson_features": counts["network"]["feature_count"],
            "O": o2_sel["selected_count"],
            "D": counts["destinations"]["feature_count"],
            "note": ve_note,
        },
        "settings_overrides": overrides,
        "settings_before": settings_before,
        "settings_provenance": provenance,
        "settings_validation": validation,
        "settings_after": None,
        "settings_after_note": "no flow cap in this workload",
        "expected_engine": "UNA.RunAccessibility",
        "required_outputs": acc_outputs,
        "target_status": "observed L2 fixture; not a performance-scale claim",
        "blocked": list(blocked_common),
        "ready_for_qualification": False,
    }

    # ---- O3_ACCESS ----------------------------------------------------------
    overrides = dict(BASE_OVERRIDES)
    settings_before, validation, provenance = _settings_dict(
        overrides, data_dir, file_names)
    manifests["O3_ACCESS"] = {
        **common_meta,
        "workload_id": "O3_ACCESS",
        "role": (
            "full observed network, all original Cambridge origins and MA bus "
            "stops, radius 500, all four metrics, no turns/elevation, "
            "geometric network cost; default primary large observed "
            "accessibility proxy"
        ),
        "origin_selection": sel_variant("O3_ACCESS_sel_all", embed_indices=False),
        "counts": {
            "V": None, "E": None,
            "E_geojson_features": counts["network"]["feature_count"],
            "O": counts["origins"]["feature_count"],
            "D": counts["destinations"]["feature_count"],
            "note": ve_note,
        },
        "settings_overrides": overrides,
        "settings_before": settings_before,
        "settings_provenance": provenance,
        "settings_validation": validation,
        "settings_after": None,
        "settings_after_note": "no flow cap in this workload",
        "expected_engine": "UNA.RunAccessibility",
        "required_outputs": acc_outputs,
        "target_status": (
            "observed tutorial proxy; NOT the user's U4 production target and "
            "never reported as an L4 success"
        ),
        "blocked": list(blocked_common),
        "ready_for_qualification": False,
    }

    # ---- O3_FLOW ------------------------------------------------------------
    flow_overrides = {**BASE_OVERRIDES, **FLOW_OVERRIDES}
    settings_before, validation, provenance = _settings_dict(
        flow_overrides, data_dir, file_names)
    variants = [
        {**sel_variant(stem, embed_indices=False), "variant_id": f"sel{k}"}
        for stem, k in (("O3_FLOW_sel256", 256),
                        ("O3_FLOW_sel1024", 1024),
                        ("O3_FLOW_sel_all", "all"))
    ]
    manifests["O3_FLOW"] = {
        **common_meta,
        "workload_id": "O3_FLOW",
        "role": (
            "same full network/destination layer with a preregistered origin "
            "selection from {256, 1024, all}; radius 500; AggregateFlow; "
            "automatic gravity cap p95. The actual variant is chosen at H05 "
            "from baseline-only resource/time pilots (largest selection that "
            "fits); a subset is never described as full-origin flow. "
            "K-alternatives stays a protected small test, not replaced by "
            "aggregate flow."
        ),
        "origin_selection": {
            "choice_policy": (
                "H05 picks the largest variant that fits baseline-only "
                "resource/time pilots before candidate timings"
            ),
            "chosen_variant": None,
            "variants": variants,
        },
        "counts": {
            "V": None, "E": None,
            "E_geojson_features": counts["network"]["feature_count"],
            "O": None,
            "O_note": "per-variant 256 / 1024 / all(origins); fixed at H05",
            "D": counts["destinations"]["feature_count"],
            "note": ve_note,
        },
        "settings_overrides": flow_overrides,
        "settings_before": settings_before,
        "settings_provenance": provenance,
        "settings_validation": validation,
        "settings_after": None,
        "settings_after_note": (
            "pending first baseline RunFlow pass: flow_gravity_cap 'p95' "
            "resolves to a numeric value and is written back to Settings "
            "(before/after capture required by WORKLOADS); the resolved "
            "number will be pinned in this manifest at H05"
        ),
        "expected_engine": (
            "UNA.RunFlow with flow_engine='aggregate_flow' "
            "(Engines.AggregateFlow.AggregateFlow)"
        ),
        "required_outputs": [
            "<job_output_root>/flow_<YYYY-MM-DD_HHMM>/<output_file_name>.feather",
            "<job_output_root>/flow_<YYYY-MM-DD_HHMM>/<output_file_name>.geojson",
            "<job_output_root>/flow_<YYYY-MM-DD_HHMM>/<output_file_name>.csv",
            ("companion outputs only if the frozen settings enable them "
             "(flow_compute_node_flow=False, no observers/obstacles: none "
             "expected)"),
        ],
        "target_status": (
            "observed tutorial proxy; NOT the user's U4 production target and "
            "never reported as an L4 success"
        ),
        "blocked": list(blocked_common),
        "ready_for_qualification": False,
    }

    # ---- O3_HOLDOUT -----------------------------------------------------------
    r1200_overrides = {**BASE_OVERRIDES, "search_radius": 1200}
    settings_before, validation, provenance = _settings_dict(
        r1200_overrides, data_dir, file_names)
    net3d = acq["files"].get("network_3d", {})
    net3d_schema = net3d.get("schema")
    variants = [{
        "variant_id": "holdout_r1200_2d",
        "network_input": "network",
        "settings_overrides": {"search_radius": 1200},
        "status": "defined",
    }]
    blocked_holdout = list(blocked_common)
    if net3d_schema:
        g3 = net3d_schema["geometry"]
        z_ok = g3["n_rows"] > 0 and g3["n_with_z"] == g3["n_rows"]
        crs_ok = g3["crs"] == counts["network"]["crs"]
        geom_ok = set(g3["geometry_types"]) <= {"LineString"}
        if "existing_file" in net3d.get("download", {}):
            blob_ok = net3d["download"]["existing_file"]["blob_sha_match"]
        else:
            blob_ok = net3d.get("download", {}).get("blob_sha_computed") \
                == cat_by_id["network_3d"]["git_blob_sha"]
        valid3d = bool(z_ok and crs_ok and geom_ok and blob_ok)
        net2d_schema = acq["files"].get("network", {}).get("schema", {})
        z2 = net2d_schema.get("role_facts", {}).get("z_value_stats")
        z3 = net3d_schema.get("role_facts", {}).get("z_value_stats")
        observed_z = {
            "network_3d_z_value_stats": z3,
            "aggregate_stats_equal_to_plain_network": (
                None if not (z2 and z3)
                else all(z2.get(k) == z3.get(k)
                         for k in ("n", "n_null", "n_nonfinite", "min",
                                   "max", "sum", "n_distinct"))
            ),
            "note": (
                "aggregate third-coordinate statistics of the catalog 3D "
                "network equal those of the plain network (both files carry "
                "Z; the files differ in attribute columns only). Vertex-level "
                "coordinate identity is NOT verified here; planar length "
                "statistics are also identical (shapely length ignores Z)."
            ),
        }
        variants.append({
            "variant_id": "holdout_r1200_3d_elevation",
            "network_input": "network_3d",
            "settings_overrides": {"search_radius": 1200, "elevation": True},
            "network_3d_input": _input_block(cat_by_id["network_3d"], net3d),
            "validity_check": {
                "all_features_carry_z": z_ok,
                "crs_matches_2d_network": crs_ok,
                "all_linestring": geom_ok,
                "blob_verified": blob_ok,
                "valid": valid3d,
            },
            "observed_z_findings": observed_z,
            "status": "valid" if valid3d else "invalid_catalog_3d_network",
        })
        if not valid3d:
            blocked_holdout.append(
                "O3_HOLDOUT 3D variant blocked: catalog 3D network failed the "
                "validity check recorded in variants")
    else:
        variants.append({
            "variant_id": "holdout_r1200_3d_elevation",
            "network_input": "network_3d",
            "status": "blocked_network_3d_not_acquired",
        })
        blocked_holdout.append(
            "O3_HOLDOUT 3D variant blocked: catalog 3D network not acquired")
    manifests["O3_HOLDOUT"] = {
        **common_meta,
        "workload_id": "O3_HOLDOUT",
        "role": (
            "radius 1200 on the same observed network, plus a 3D/elevation "
            "profile using the catalog 3D network when valid; checks settings "
            "generalization, NOT independent network morphology. Cross-"
            "morphology generalization remains unavailable without a "
            "user-supplied second observed morphology."
        ),
        "variants": variants,
        "origin_selection": sel_variant("O3_HOLDOUT_sel_all", embed_indices=False),
        "counts": {
            "V": None, "E": None,
            "E_geojson_features": counts["network"]["feature_count"],
            "O": counts["origins"]["feature_count"],
            "D": counts["destinations"]["feature_count"],
            "note": ve_note,
        },
        "settings_overrides": r1200_overrides,
        "settings_before": settings_before,
        "settings_provenance": provenance,
        "settings_validation": validation,
        "settings_after": None,
        "settings_after_note": "no flow cap in this workload",
        "expected_engine": "UNA.RunAccessibility",
        "required_outputs": acc_outputs,
        "target_status": (
            "observed tutorial proxy holdout; checks settings generalization, "
            "not morphology generalization"
        ),
        "blocked": blocked_holdout,
        "ready_for_qualification": False,
    }

    # ---- write manifests + index ----------------------------------------------
    args.tests_inputs.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "task": "H01",
        "generated_utc": utc_now(),
        "workload_classes": {},
        "selection_files": selection["variants"],
        "settings_hashes": {},
        "unavailable": [{
            "class": "U4",
            "definition": (
                "user's actual production dataset/settings/outputs/job "
                "volume/throughput target"
            ),
            "status": (
                "unavailable - not supplied; O3 is never substituted for it "
                "and never reported as an L4 success"
            ),
        }],
        "blocked": [],
        "notes": [
            "all observed workloads are dataset class observed_tutorial_proxy "
            "(upstream Boston tutorial data), not user production data",
            "V/E counts pending a baseline load pass (recorded per manifest)",
            "stripe/thread profile null until frozen (H05/S01)",
        ],
    }
    for wid, man in manifests.items():
        path = args.tests_inputs / f"{wid}.manifest.json"
        path.write_text(pretty_json(man), encoding="utf-8")
        digest = sha256_file(path)
        index["workload_classes"][wid] = {
            "manifest": str(path.relative_to(REPO_ROOT)),
            "manifest_sha256": digest,
            "dataset_class": man["dataset_class"],
            "expected_engine": man["expected_engine"],
            "ready_for_qualification": man["ready_for_qualification"],
        }
        index["settings_hashes"][wid] = _settings_hash(man["settings_before"])
        for reason in man["blocked"]:
            index["blocked"].append({"workload": wid, "reason": reason})
        print(f"[manifest] {wid}: {path} sha256={digest}")

    (args.evidence / "workloads.json").write_text(
        pretty_json(index), encoding="utf-8")
    print(f"evidence: {args.evidence / 'workloads.json'}")
    return 0


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
        sp.add_argument("--dest", type=Path, default=DEFAULT_DATA_DIR,
                        help="local bulk-data dir (outside the repo)")
        sp.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
        sp.add_argument("--tests-inputs", type=Path, default=DEFAULT_TESTS_INPUTS)

    sp = sub.add_parser("download", help="acquire + verify catalog files")
    common(sp)
    sp.add_argument("--ids", nargs="*", help="subset of catalog ids")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("inspect", help="record schema findings per file")
    common(sp)
    sp.add_argument("--ids", nargs="*", help="subset of catalog ids")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("select", help="pinned-RNG origin selection lists")
    common(sp)
    sp.set_defaults(func=cmd_select)

    sp = sub.add_parser("freeze", help="render workload manifests + index")
    common(sp)
    sp.set_defaults(func=cmd_freeze)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
