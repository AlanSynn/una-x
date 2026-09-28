"""Arm identity JSON: schema, guards, and a generator (HARNESS.md behavior 2).

An identity file is the per-arm proof of WHAT code the workers will import.
Two arms may share a package version string; the version alone is never
identity — module hashes are.

Schema (schema_version 1):
{
  "schema_version": 1,
  "arm": "label",
  "identity_kind": "installed_wheel" | "diagnostic_source",
  "interpreter": "/abs/python",
  "interpreter_version": "3.x.y",
  "site_packages": ["/abs/.../site-packages", ...],   # installed kind only
  "package_root": "/abs/.../urban_network_analysis",
  "package_version": "x.y.z",
  "module_hashes": {"relative/path.py": "sha256", ...},
  "package_tree_sha256": "sha256-of-canonical-module_hashes",
  "settings_fields": ["field", ...],
  "created_utc": "...",
  "notes": "..."
  "identity_sha256": "sha256 of canonical JSON without this field"
}

Worker-side guards (all enforced in-process by the worker before the engine
is used):
  * PYTHONPATH must be absent (installed mode).
  * The resolved package path must be inside a declared site-packages dir,
    checked with Path.resolve()/is_relative_to + os.path.commonpath — never
    a string prefix (so a sibling 'site-packages-fake' directory is refused).
  * The resolved package path must not be under the harness repository root
    (repository import shadowing), and no sys.path entry may sit under it.
  * Declared site-packages must not contain editable-install markers:
    .pth files with import lines or repository paths, __editable__* finders,
    or a direct_url.json marking the distribution editable.
  * An installed distribution metadata directory (dist-info) must exist.
  * Every module hash must match the identity; any changed file on disk
    invalidates the warm identity.

Run as a script with --make to generate an identity under the CURRENT
interpreter (this is the only place the harness imports the engine package;
run.py itself never does).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from harness.manifest import sha256_file
from harness.spec import ValidationError

IDENTITY_SCHEMA_VERSION = 1
IDENTITY_KINDS = ("installed_wheel", "diagnostic_source")
PACKAGE_IMPORT_NAME = "urban_network_analysis"
# PEP 503 normalized distribution name -> dist-info directory prefix.
DIST_INFO_RE = re.compile(r"^urban_network_analysis-.+\.dist-info$", re.IGNORECASE)


class Identity:
    def __init__(self, data: dict, path: Path) -> None:
        self.data = data
        self.path = path
        self.kind: str = data["identity_kind"]
        self.arm: str = data["arm"]
        self.interpreter: str = data["interpreter"]
        self.interpreter_version: str = data["interpreter_version"]
        self.site_packages: list[Path] = [Path(p) for p in data.get("site_packages", [])]
        self.package_root = Path(data["package_root"])
        self.package_version: str = data["package_version"]
        self.module_hashes: dict[str, str] = data["module_hashes"]
        self.package_tree_sha256: str = data["package_tree_sha256"]
        self.settings_fields: set[str] = set(data["settings_fields"])
        self.identity_sha256: str = data["identity_sha256"]

    def describe(self) -> dict:
        out = dict(self.data)
        out.pop("module_hashes", None)
        out["module_hash_count"] = len(self.module_hashes)
        out["identity_file_sha256"] = sha256_file(self.path)
        return out

    @property
    def fingerprint(self) -> str:
        """Stable short fingerprint for embedding in per-job records."""
        return self.package_tree_sha256[:16]


# ----------------------------------------------------------------------
# Canonicalization / integrity
# ----------------------------------------------------------------------
def canonical_identity_bytes(data: dict) -> bytes:
    payload = {k: v for k, v in data.items() if k != "identity_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_identity_sha256(data: dict) -> str:
    return hashlib.sha256(canonical_identity_bytes(data)).hexdigest()


def load_identity(path: Path) -> Identity:
    if not path.exists() or not path.is_file():
        raise ValidationError(f"identity file does not exist: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"identity file is not valid JSON: {exc}")
    required = {
        "schema_version", "arm", "identity_kind", "interpreter",
        "interpreter_version", "package_root", "package_version",
        "module_hashes", "package_tree_sha256", "settings_fields",
        "identity_sha256",
    }
    missing = required - set(data)
    if missing:
        raise ValidationError(f"identity missing required fields: {sorted(missing)}")
    if data["schema_version"] != IDENTITY_SCHEMA_VERSION:
        raise ValidationError(
            f"identity schema_version must be {IDENTITY_SCHEMA_VERSION}, "
            f"got {data['schema_version']!r}")
    if data["identity_kind"] not in IDENTITY_KINDS:
        raise ValidationError(
            f"identity_kind must be one of {list(IDENTITY_KINDS)}, "
            f"got {data['identity_kind']!r}")
    expected = compute_identity_sha256(data)
    if expected != data["identity_sha256"]:
        raise ValidationError(
            "identity integrity check failed: identity_sha256 does not match "
            "file contents (tampered or hand-edited identity)")
    if data["identity_kind"] == "installed_wheel":
        sp = data.get("site_packages")
        if not isinstance(sp, list) or not sp:
            raise ValidationError("installed_wheel identity requires site_packages list")
        for entry in sp:
            if not Path(entry).is_dir():
                raise ValidationError(
                    f"identity site_packages entry does not exist: {entry}")
    return Identity(data, path)


# ----------------------------------------------------------------------
# Guard primitives (unit-testable; exercised by the worker in-process)
# ----------------------------------------------------------------------
def guard_no_pythonpath(environ: dict) -> None:
    """Installed workers refuse a PYTHONPATH (import shadowing vector)."""
    if environ.get("PYTHONPATH"):
        raise ValidationError(
            f"installed worker refuses PYTHONPATH={environ['PYTHONPATH']!r}")


def check_under_site_packages(package_root: Path,
                              site_packages: list[Path]) -> Path:
    """Resolution-based containment; refuses the sibling-prefix trick.

    Uses Path.resolve() + is_relative_to AND os.path.commonpath, the two
    sanctioned mechanisms, so '/x/site-packages-fake' is never accepted as
    'under /x/site-packages' by string coincidence.
    """
    resolved = package_root.resolve()
    for sp in site_packages:
        sp_resolved = sp.resolve()
        try:
            resolved.relative_to(sp_resolved)
        except ValueError:
            continue
        # Second, independent mechanism (os.path.commonpath).
        common = os.path.commonpath([str(resolved), str(sp_resolved)])
        if Path(common).resolve() != sp_resolved:
            raise ValidationError(
                f"commonpath disagreement for {resolved} vs {sp_resolved}")
        return sp_resolved
    raise ValidationError(
        f"resolved package path {resolved} is not inside any declared "
        f"site-packages directory "
        f"({[str(p) for p in site_packages]}); string-prefix similarity to "
        f"'site-packages' is not acceptance")


def guard_repository_shadow(package_root: Path, repo_root: Path,
                            sys_path_entries: list[str]) -> None:
    """Refuse repository shadowing: package or sys.path entries under repo."""
    resolved = package_root.resolve()
    repo = repo_root.resolve()
    try:
        resolved.relative_to(repo)
    except ValueError:
        pass
    else:
        raise ValidationError(
            f"import resolved inside the harness repository ({repo}): "
            f"{resolved} — repository import shadowing is forbidden")
    for entry in sys_path_entries:
        try:
            Path(entry).resolve().relative_to(repo)
        except (ValueError, OSError):
            continue
        raise ValidationError(
            f"sys.path entry {entry!r} is inside the harness repository "
            f"({repo}); source shadowing is forbidden in installed mode")


# H04 W00 stop adjudication (2026-09-28): the stock setuptools
# distutils-precedence shim ships in every venv with setuptools intact and
# is not an editable/source shadowing vector.  The exemption in
# scan_pth_files is name+content pinned to this exact byte sequence; the
# same file name with any other content, or any other .pth content,
# refuses exactly as before (fail-closed — a setuptools update
# re-triggers refusal and re-adjudication).
BENIGN_DISTUTILS_PTH_SHA256 = (
    "2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224")


def scan_pth_files(site_packages: list[Path]) -> list[str]:
    """Find .pth editable/source hooks in declared site-packages.

    A .pth line that starts with 'import ' is an executable hook (editable
    finders); a plain path line is added to sys.path.  Any .pth at all whose
    lines point at or import a source tree is a shadowing vector; in
    installed qualification mode ALL non-comment .pth content is refused so
    the arm cannot quietly extend sys.path.  Sole exempt file: the stock
    setuptools distutils-precedence shim, pinned by name AND sha256
    (BENIGN_DISTUTILS_PTH_SHA256); the exemption prints to stdout.
    """
    findings: list[str] = []
    for sp in site_packages:
        sp = sp.resolve()
        if not sp.is_dir():
            continue
        for pth in sorted(sp.glob("*.pth")):
            if (pth.name == "distutils-precedence.pth"
                    and sha256_file(pth) == BENIGN_DISTUTILS_PTH_SHA256):
                print(f"[identity] exempted {pth}: stock setuptools "
                      "distutils-precedence shim, sha256 match "
                      "(h04 W00 stop adjudication)")
                continue
            try:
                lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                findings.append(f"{pth}: unreadable ({exc})")
                continue
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                findings.append(f"{pth}: {stripped}")
    return findings


def scan_editable_install(site_packages: list[Path]) -> list[str]:
    """Find editable-install markers for the package in site-packages."""
    findings: list[str] = []
    for sp in site_packages:
        sp = sp.resolve()
        if not sp.is_dir():
            continue
        for dist in sorted(sp.glob("*.dist-info")):
            if not DIST_INFO_RE.match(dist.name):
                continue
            direct_url = dist / "direct_url.json"
            if direct_url.exists():
                try:
                    info = json.loads(direct_url.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    findings.append(f"{direct_url}: unparseable direct_url.json")
                    continue
                if info.get("dir_info", {}).get("editable"):
                    findings.append(
                        f"{direct_url}: editable install of {info.get('url')!r}")
        for finder in sorted(sp.glob("__editable__*")):
            findings.append(f"{finder}: editable finder artifact")
    return findings


def require_dist_info(site_packages: list[Path]) -> str:
    """An installed run must carry real distribution metadata."""
    for sp in site_packages:
        sp = sp.resolve()
        if not sp.is_dir():
            continue
        for dist in sorted(sp.glob("*.dist-info")):
            if DIST_INFO_RE.match(dist.name):
                return str(dist)
    raise ValidationError(
        "no urban_network_analysis*.dist-info found in declared site-packages "
        f"{[str(p) for p in site_packages]}; refusing installed qualification "
        "without installed distribution metadata")


def hash_package_tree(package_root: Path) -> dict[str, str]:
    """sha256 every SOURCE file under the package, keyed by relative path.

    Derived interpreter artifacts (__pycache__/, *.pyc) are excluded: they
    change between runs and are not source identity.  Numba's cache lives
    outside the tree (NUMBA_CACHE_DIR), so the tree is stable.
    """
    hashes: dict[str, str] = {}
    root = package_root.resolve()
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
            continue
        key = file_path.relative_to(root).as_posix()
        hashes[key] = sha256_file(file_path)
    if not hashes:
        raise ValidationError(f"package tree is empty: {root}")
    return hashes


def tree_digest(module_hashes: dict[str, str]) -> str:
    canonical = json.dumps(module_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_package_tree(package_root: Path, expected: dict[str, str]) -> dict:
    """Compare on-disk package files against identity hashes.

    Any changed, missing or unexpected file invalidates the warm identity
    ('changed cache/module source hash: invalidates warm identity').
    """
    actual = hash_package_tree(package_root)
    changed = sorted(k for k in actual if k in expected and actual[k] != expected[k])
    missing = sorted(k for k in expected if k not in actual)
    extra = sorted(k for k in actual if k not in expected)
    if changed or missing or extra:
        raise ValidationError(
            "installed package tree does not match identity "
            f"(changed={changed[:8]}, missing={missing[:8]}, extra={extra[:8]})")
    return {"verified_files": len(actual), "tree_sha256": tree_digest(actual)}


def verify_imported_modules(module_objects: dict[str, object],
                            identity: Identity) -> dict:
    """Hash every imported urban_network_analysis module file."""
    checked: dict[str, str] = {}
    verified_names: set[str] = set()
    for name, module in sorted(module_objects.items()):
        if name != PACKAGE_IMPORT_NAME and not name.startswith(PACKAGE_IMPORT_NAME + "."):
            continue
        file_path = getattr(module, "__file__", None)
        if not file_path:
            continue  # namespace/extension shims without a file
        resolved = Path(file_path).resolve()
        root = identity.package_root.resolve()
        try:
            key = resolved.relative_to(root).as_posix()
        except ValueError:
            raise ValidationError(
                f"imported module {name} resolved outside the identity "
                f"package root: {resolved}")
        expected = identity.module_hashes.get(key)
        if expected is None:
            raise ValidationError(f"imported module {key} absent from identity hashes")
        actual = sha256_file(resolved)
        if actual != expected:
            raise ValidationError(
                f"module hash mismatch for {key}: identity={expected} actual={actual}")
        checked[key] = actual
        verified_names.add(name)
    if PACKAGE_IMPORT_NAME not in verified_names:
        raise ValidationError("top-level package module was not hash-verified")
    return {"verified_modules": len(checked)}


# ----------------------------------------------------------------------
# Generator (run under the ARM interpreter)
# ----------------------------------------------------------------------
def make_identity(arm: str, kind: str, out_path: Path,
                  site_packages: list[Path] | None = None,
                  expected_root: Path | None = None,
                  notes: str = "") -> Identity:
    """Import the package under the CURRENT interpreter and freeze identity.

    kind=installed_wheel requires the import to resolve under one of the
    given site-packages dirs; kind=diagnostic_source requires it to resolve
    under expected_root (a source tree) and records the diagnostic kind.
    """
    import importlib
    import datetime as _dt

    if kind not in IDENTITY_KINDS:
        raise ValidationError(f"identity kind must be one of {IDENTITY_KINDS}")

    package = importlib.import_module(PACKAGE_IMPORT_NAME)
    package_root = Path(package.__file__).resolve().parent
    package_version = getattr(package, "__version__", None)
    if package_version is None:
        raise ValidationError(
            f"{PACKAGE_IMPORT_NAME} has no __version__; cannot build identity")

    if kind == "installed_wheel":
        if not site_packages:
            raise ValidationError("installed_wheel identity requires --site-packages")
        check_under_site_packages(package_root, site_packages)
        editable = scan_editable_install(site_packages)
        if editable:
            raise ValidationError(f"refusing identity: editable markers {editable}")
        pth = scan_pth_files(site_packages)
        if pth:
            raise ValidationError(f"refusing identity: .pth hooks present {pth}")
    else:
        if expected_root is None:
            raise ValidationError("diagnostic_source identity requires --source-root")
        try:
            package_root.relative_to(expected_root.resolve())
        except ValueError:
            raise ValidationError(
                f"package root {package_root} is not under diagnostic source "
                f"root {expected_root}")

    settings_fields: list[str] = []
    settings_mod = importlib.import_module(f"{PACKAGE_IMPORT_NAME}.Settings")
    settings_cls = getattr(settings_mod, "Settings", None)
    if settings_cls is not None:
        import dataclasses
        settings_fields = [f.name for f in dataclasses.fields(settings_cls)]

    module_hashes = hash_package_tree(package_root)
    data = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "arm": arm,
        "identity_kind": kind,
        "interpreter": os.path.realpath(sys.executable),
        "interpreter_version": ".".join(str(p) for p in sys.version_info[:3]),
        "site_packages": [str(p) for p in (site_packages or [])],
        "package_root": str(package_root),
        "package_version": str(package_version),
        "module_hashes": module_hashes,
        "package_tree_sha256": tree_digest(module_hashes),
        "settings_fields": settings_fields,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    data["identity_sha256"] = compute_identity_sha256(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return Identity(data, out_path)


def main(argv: list[str]) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m harness.identity",
        description="Generate an arm identity JSON under the CURRENT "
                    "interpreter (run with the arm's venv python).")
    parser.add_argument("--make", action="store_true", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--kind", required=True, choices=list(IDENTITY_KINDS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--site-packages", type=Path, action="append", default=[])
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    try:
        identity = make_identity(
            arm=args.arm, kind=args.kind, out_path=args.out,
            site_packages=args.site_packages or None,
            expected_root=args.source_root, notes=args.notes)
    except ValidationError as exc:
        print(f"identity generation refused: {exc}", file=sys.stderr)
        return 2
    print(f"identity written: {identity.path} "
          f"(tree {identity.package_tree_sha256[:16]}, "
          f"{len(identity.module_hashes)} files)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
