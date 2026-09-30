"""Read-only raw Git-blob verification against campaign source pins."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def audit(root: Path, pins: dict) -> dict:
    root = Path(root).resolve()
    if not pins:
        raise ValueError("No pinned source files")
    rows = []
    for rel, expected in sorted(pins.items()):
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source pin escapes root")
        actual = git_blob(path.read_bytes()) if path.is_file() else None
        rows.append({"path": rel, "expected": expected, "actual": actual, "match": actual == expected})
    return {"match": all(r["match"] for r in rows), "files": rows,
            "note": "Content identity, not application execution or scientific qualification."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--reference", choices=["una", "madina"], default="una")
    p.add_argument("--pins", type=Path, default=Path(__file__).resolve().parents[1] / "source_pins.json")
    a = p.parse_args()
    pins = json.loads(a.pins.read_text())[a.reference]["files"]
    report = audit(a.root, pins)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["match"] else 1)

if __name__ == "__main__":
    main()
