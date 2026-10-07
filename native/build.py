"""Build the N00 native trace producer extension (campaign una_10x).

AOT C++17 + pybind11, no unsafe FP flags anywhere: the exact compile
flags are recorded in the build manifest.  Run with the DEDICATED venv
(venv-n00) so pybind11's headers and the target interpreter match:

    /Users/alansynn/orca/workspaces/una-x/venv-n00/bin/python native/build.py

Outputs into native/_build/ (gitignored):
    una_n00_trace_producer<EXT_SUFFIX>   the extension module
    build_manifest.json                  sources, flags, digests
    vectorization_report.txt             -Rpass[-missed]=loop-vectorize
"""
import hashlib
import json
import pybind11
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

NATIVE_DIR = Path(__file__).resolve().parent
BUILD_DIR = NATIVE_DIR / "_build"

SOURCES = ["trace_producer.cpp", "bindings.cpp"]

# The flags ARE the record: no -ffast-math (or any unsafe FP flag);
# -ffp-contract=off forbids FMA fusion; -O3 is the optimization level
# the C arm is measured under.  Every flag below appears verbatim in
# build_manifest.json.
BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-Wall",
    "-Wextra",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ext_path() -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return BUILD_DIR / ("una_n00_trace_producer" + suffix)


def is_stale() -> bool:
    out = ext_path()
    manifest = BUILD_DIR / "build_manifest.json"
    if not out.exists() or not manifest.exists():
        return True
    try:
        man = json.loads(manifest.read_text())
    except (ValueError, OSError):
        return True
    if man.get("output_sha256") != sha256(out):
        return True
    for name in SOURCES:
        if man.get("sources", {}).get(name) != \
                sha256(NATIVE_DIR / name):
            return True
    return False


def build(report: bool = True) -> Path:
    BUILD_DIR.mkdir(exist_ok=True)
    cc = shutil.which("clang++") or shutil.which("c++")
    if cc is None:
        raise RuntimeError("no C++ compiler found (need clang++ >= 19 "
                           "or any C++17 toolchain)")
    out = ext_path()
    flags = list(BASE_FLAGS)
    flags += [
        "-I" + sysconfig.get_paths()["include"],
        "-I" + pybind11.get_include(),
        "-undefined", "dynamic_lookup",
        '-DUNA_N00_BUILD_FLAGS="%s"' % " ".join(BASE_FLAGS),
        '-DUNA_N00_BUILD_CC="%s"' % cc,
    ]
    cmd = [cc] + flags + [str(NATIVE_DIR / s) for s in SOURCES] + \
        ["-o", str(out)]

    report_path = BUILD_DIR / "vectorization_report.txt"
    run_cmd = list(cmd)
    if report:
        run_cmd += [
            "-Rpass=loop-vectorize",
            "-Rpass-missed=loop-vectorize",
            "-Rpass-analysis=loop-vectorize",
        ]
    proc = subprocess.run(run_cmd, capture_output=True, text=True)
    if report and (proc.stderr or proc.stdout):
        report_path.write_text(
            "# clang loop-vectorize report (N00 native producer)\n"
            "# cmd: %s\n\n%s%s"
            % (" ".join(cmd), proc.stdout, proc.stderr))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError("native producer build failed: rc=%d"
                           % proc.returncode)

    manifest = {
        "schema": "una-10x-n00-build-manifest-1",
        "compiler": cc,
        "compile_flags": BASE_FLAGS,
        "report_flags": ["-Rpass=loop-vectorize",
                         "-Rpass-missed=loop-vectorize",
                         "-Rpass-analysis=loop-vectorize"],
        "defines": {
            "UNA_N00_BUILD_FLAGS": " ".join(BASE_FLAGS),
            "UNA_N00_BUILD_CC": cc,
        },
        "pybind11_version": pybind11.__version__,
        "python": sys.version.split()[0],
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "sources": {name: sha256(NATIVE_DIR / name)
                    for name in SOURCES},
        "output": out.name,
        "output_sha256": sha256(out),
    }
    (BUILD_DIR / "build_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True))
    return out


def ensure_built() -> Path:
    if is_stale():
        return build()
    return ext_path()


if __name__ == "__main__":
    p = build()
    print(p)
    print(json.dumps(json.loads(
        (BUILD_DIR / "build_manifest.json").read_text()), indent=1))
