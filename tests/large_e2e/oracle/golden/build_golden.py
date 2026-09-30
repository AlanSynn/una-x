"""Build the golden corpus for H03 from the immutable B0 tree.

Runs tests/large_e2e/oracle/runner_arm.py in a FRESH subprocess bound to
the B0 import root (never candidate code), writes:

  golden/golden_b0.npz    — every corpus input and compiled output array
  golden/golden_b0.json   — profile, refusal probes, checks, determinism
  golden/golden_b0.hashes.json — sha256 of the two assets above plus the
                            per-key digests (content-addressed identity)

Golden outputs derive ONLY from this script's B0 subprocess. No golden
value is produced by trace/replica code or loaded by filename-only
reuse (the hashes file is re-verified by the tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from _legacy_paths import LEGACY_WS  # portable-path fix (HARNESS 2026-09-30)
ORACLE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent
B0_ROOT = str(LEGACY_WS) + "/wt-b0/src"
B0_COMMIT = "361928e4ba38f34622cafe065b0025244db61368"
CAMPAIGN_PYTHON = (str(LEGACY_WS) + "/venvs/campaign/"
                   "bin/python")
CACHE_ROOT = str(LEGACY_WS) + "/campaign_data/nbc_oracle"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key_digests(npz_path):
    import numpy as np
    out = {}
    with np.load(npz_path) as z:
        for key in z.files:
            a = np.ascontiguousarray(z[key])
            h = hashlib.sha256()
            h.update(a.dtype.str.encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
            out[key] = h.hexdigest()
    return out


def main():
    npz_path = GOLDEN_DIR / "golden_b0.npz"
    json_path = GOLDEN_DIR / "golden_b0.json"
    env = dict(os.environ)
    env["UNA_ORACLE_ARM_A"] = B0_ROOT
    env["PYTHONPATH"] = B0_ROOT
    env["NUMBA_CACHE_DIR"] = CACHE_ROOT
    env.pop("L1_REUSE_DIR", None)
    proc = subprocess.run(
        [CAMPAIGN_PYTHON, str(ORACLE_DIR / "runner_arm.py"),
         "--out-npz", str(npz_path), "--out-json", str(json_path),
         "--runs", "2"],
        cwd="/tmp", env=env, capture_output=True, text=True, timeout=3600,
    )
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise SystemExit(f"golden build failed rc={proc.returncode}")

    hashes = {
        "golden_b0.npz": sha256_file(npz_path),
        "golden_b0.json": sha256_file(json_path),
        "key_digests": key_digests(npz_path),
        "generator": str(Path(__file__).resolve()),
        "fixture_seed": json.load(open(json_path))["fixture_seed"],
        "import_root": B0_ROOT,
        "b0_commit": B0_COMMIT,
        "built_by_env": {
            "python": sys.version,
            "NUMBA_CACHE_DIR": CACHE_ROOT,
        },
    }
    with open(GOLDEN_DIR / "golden_b0.hashes.json", "w") as fh:
        json.dump(hashes, fh, indent=2, sort_keys=True)
    print(f"golden built: {npz_path.name} sha256={hashes['golden_b0.npz'][:16]}... "
          f"keys={len(hashes['key_digests'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
