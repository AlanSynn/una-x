"""Campaign harness CLI: `python -m benchmarks.platform run --manifest M ...`

Executes one real job through the installed public API and writes the
session record to a campaign-owned out-of-tree directory.  Build/lint/parity
checks stay local; this runner never writes inside the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .errors import HarnessViolation
from .guards import OutputDirGuard
from .identity import resolve_identity, assert_installed, source_tree_sha256
from .manifest import load_manifest, resource_policy_of
from .records import validate_record, write_session
from .runner import HARNESS_VERSION, new_run_id, run_one_real_job
from .validation import validate_output_obligations


def cmd_run(args: argparse.Namespace) -> int:
    # Children run with cwd set to their work dir; every path handed to
    # them (interpreter, out-dir) must be absolute before spawning.  The
    # interpreter is made absolute WITHOUT resolving symlinks: venv
    # prefix detection depends on invoking through the venv's own path.
    args.python = os.path.abspath(args.python)
    args.out_dir = str(Path(args.out_dir).resolve())
    repo = str(Path(args.repo).resolve())
    manifest = load_manifest(args.manifest, base_dir=str(Path(args.manifest)
                                                          .parent))
    policy = resource_policy_of(manifest)

    expected = manifest.get("expected_identity") or {}
    info = resolve_identity(args.python, expected.get(
        "package", "urban_network_analysis"))
    assert_installed(info, expected.get("package",
                                        "urban_network_analysis"),
                     expected_wheel_tree_sha256=expected.get(
                         "wheel_tree_sha256"),
                     forbidden_roots=[str(Path(repo) / "src")])

    run_id = new_run_id("h")
    t0 = time.perf_counter()
    record = run_one_real_job(
        manifest, repo=repo, python_exe=args.python, out_dir=args.out_dir,
        run_id=run_id, timeout_s=args.timeout_s,
        outdir_guard=OutputDirGuard(args.out_dir, run_id, repo))
    obligations = manifest.get("output_obligations") or {}
    if obligations:
        t_hash0 = time.perf_counter()
        record["output_manifest"] = validate_output_obligations(
            Path(args.out_dir) / "leg_timed", obligations)
        record["timings"]["post_hash_s"] = time.perf_counter() - t_hash0
    record["reference_source_sha256"] = source_tree_sha256(
        str(Path(repo) / "src"))
    validate_record(record)
    session = write_session(Path(args.out_dir), run_id, [record],
                            commands=[" ".join(sys.argv)])
    print(json.dumps({
        "run_id": run_id,
        "status": "validated",
        "harness_version": HARNESS_VERSION,
        "session": str(session),
        "wall_s": time.perf_counter() - t0,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="benchmarks.platform")
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="execute one frozen real job")
    run_p.add_argument("--manifest", required=True)
    run_p.add_argument("--python", required=True,
                       help="interpreter holding the installed package")
    run_p.add_argument("--repo", required=True)
    run_p.add_argument("--out-dir", required=True,
                       help="campaign-owned output directory (outside repo)")
    run_p.add_argument("--timeout-s", type=float, default=600.0)
    run_p.set_defaults(fn=cmd_run)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except HarnessViolation as exc:
        print(f"HARNESS VIOLATION [{exc.check_id}]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
