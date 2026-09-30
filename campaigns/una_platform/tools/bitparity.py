"""Strict typed-array byte comparison; no pickle, tolerance or implicit casts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load_arrays(path: Path) -> dict:
    path = Path(path)
    if path.suffix == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    raise ValueError("Only .npy/.npz with allow_pickle=False are accepted")


def compare_arrays(reference: dict, candidate: dict) -> dict:
    if not reference or not candidate:
        raise ValueError("Empty comparison set is not parity evidence")
    diffs = []
    compared_bytes = 0
    if set(reference) != set(candidate):
        diffs.append({"reason": "array_set", "reference_only": sorted(set(reference)-set(candidate)),
                      "candidate_only": sorted(set(candidate)-set(reference))})
    for key in sorted(set(reference) & set(candidate)):
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        if a.dtype.hasobject or b.dtype.hasobject:
            raise ValueError("Object arrays are not accepted; compare typed schemas explicitly")
        if a.dtype != b.dtype or a.dtype.str != b.dtype.str or a.dtype.descr != b.dtype.descr:
            diffs.append({"array": key, "reason": "dtype", "reference": str(a.dtype), "candidate": str(b.dtype)})
            continue
        if a.shape != b.shape:
            diffs.append({"array": key, "reason": "shape", "reference": list(a.shape), "candidate": list(b.shape)})
            continue
        aa, bb = a.tobytes(order="C"), b.tobytes(order="C")
        compared_bytes += len(aa)
        if aa != bb:
            first = next(i for i, (x, y) in enumerate(zip(aa, bb)) if x != y)
            diffs.append({"array": key, "reason": "bytes", "first_byte": first,
                          "reference_byte": aa[first], "candidate_byte": bb[first]})
    return {"schema_version": 1, "match": not diffs, "compared_arrays": len(set(reference)&set(candidate)),
            "compared_bytes": compared_bytes, "mismatches": diffs,
            "note": "Array equality only. Provenance, state chronology, artifacts and backend engagement are separate gates."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    a = p.parse_args()
    result = compare_arrays(load_arrays(a.reference), load_arrays(a.candidate))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["match"] else 1)

if __name__ == "__main__":
    main()
