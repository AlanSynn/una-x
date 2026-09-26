"""Stub engine + harness self-test kit for the large-e2e harness tests.

The stub package mirrors the real urban_network_analysis public surface
minimally (UNA/Settings/Topology + Engines.Base export boundary) so the
REAL harness code — worker guards, pool, queues, sampler, verifier — runs
against a tiny, fast, dependency-free engine.  Expected-error fixtures live
here, never in production data.  Nothing in this module is used by run.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CAMPAIGN_PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_PY = REPO_ROOT / "benchmarks" / "large_e2e" / "run.py"
HARNESS_DIR = REPO_ROOT / "benchmarks" / "large_e2e"

STUB_VERSION = "0.0.0-harness-stub"

# ----------------------------------------------------------------------
# Stub package source (written into a fake venv's site-packages)
# ----------------------------------------------------------------------
STUB_INIT = f'''"""Harness self-test stub of urban_network_analysis (NOT production)."""
__version__ = "{STUB_VERSION}"

from .Settings import Settings
from .Topology import Topology
from .UNA import UNA
'''

STUB_SETTINGS = '''"""Stub Settings: dataclass with the fields stub manifests may set."""
from dataclasses import dataclass, fields, asdict


@dataclass
class Settings:
    data_folder: str = ""
    network_file: str = "network.feather"
    origins_file: str = "origins.feather"
    destinations_file: str = "destinations.feather"
    search_radius: int = 1500
    output_folder: object = None
    output_wStamp: bool = False
    output_file_name: str = "Results"
    progressbar: bool = False
    stub_origin_count: int = 4
    stub_edge_count: int = 8
    stub_od_pairs: int = 4
    stub_delay_s: float = 0.0

    def Validation(self):
        return True

    def ToDict(self, compact: bool = True) -> dict:
        return asdict(self)
'''

STUB_TOPOLOGY = '''"""Stub Topology with the num_threads attribute the harness observes."""


class _QuietLogger:
    def log(self, *args, **kwargs):
        pass


class Topology:
    def __init__(self, verbosity: int = 1, num_threads: int = None):
        self.num_threads = num_threads if num_threads is not None else 1
        self.num_clusters = None
        self.logger = _QuietLogger()
'''

STUB_ENGINES_INIT = '"""Stub Engines package."""\n'

STUB_BASE = '''"""Stub Engines.Base: the export boundary the writer adapter wraps."""
import os


class Base:
    def __init__(self, topology):
        self.topology = topology
        self.settings = None
        self.reach = None
        self.gravity_exponential = None
        self.gravity_logistic = None
        self.knn_access = None
        self.edge_flow = None
        self.edge_flow_AB = None
        self.edge_flow_BA = None
        self.node_flow = None
        self.od_matrix = None

    def _write(self, file_name: str, payload: bytes) -> None:
        folder = self.settings.output_folder
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, file_name), "wb") as handle:
            handle.write(payload)

    def ExportAccessibilityResults(self, settings, folder_prefix="",
                                   file_name="Results"):
        self.settings = settings
        self._write(file_name + ".feather",
                    b"STUB-ACC-FEATHER n=" + str(len(self.reach)).encode())
        self._write(file_name + ".geojson",
                    b"STUB-ACC-GEOJSON n=" + str(len(self.reach)).encode())

    def ExportFlowResult(self, settings, folder_prefix="", file_name="flow"):
        self.settings = settings
        self._write(file_name + ".feather",
                    b"STUB-FLOW-FEATHER n=" + str(len(self.edge_flow)).encode())
        self._write(file_name + ".geojson",
                    b"STUB-FLOW-GEOJSON n=" + str(len(self.edge_flow)).encode())

    def ExportODM(self, settings, folder_prefix="ODM_", file_name="ODM",
                  format="Sqlite", speed=5.0):
        self.settings = settings
        self._write(file_name + ".sqlite",
                    b"STUB-ODM pairs=" + str(len(self.od_matrix)).encode())
'''

STUB_UNA = '''"""Stub UNA: the three public methods, fresh engines, real exports."""
import time

from .Engines.Base import Base
from .Settings import Settings
from .Topology import Topology


class _StubAccessibility(Base):
    def Centrality(self, settings):
        self.settings = settings
        time.sleep(float(getattr(settings, "stub_delay_s", 0.0)))
        n = int(getattr(settings, "stub_origin_count", 4))
        self.reach = [float(i) for i in range(n)]
        self.gravity_exponential = list(self.reach)
        self.gravity_logistic = list(self.reach)
        self.knn_access = list(self.reach)


class _StubFlow(Base):
    def Centrality(self, settings):
        self.settings = settings
        time.sleep(float(getattr(settings, "stub_delay_s", 0.0)))
        n = int(getattr(settings, "stub_edge_count", 8))
        self.edge_flow = [float(i) for i in range(n)]
        self.edge_flow_AB = list(self.edge_flow)
        self.edge_flow_BA = list(self.edge_flow)
        self.node_flow = None


class _StubODM(Base):
    def OD_Matrix(self, search_radius: float = 0.0):
        pairs = int(getattr(self.settings, "stub_od_pairs", 4))
        self.od_matrix = [[float(i) for i in range(pairs)] for _ in range(pairs)]


class UNA:
    def __init__(self, verbosity: int = 1):
        self.topology = Topology()
        self.settings = Settings()
        self.accessibility = None
        self.flow = None
        self.resolved_gravity_cap = None

    def RunAccessibility(self):
        engine = _StubAccessibility(self.topology)
        engine.Centrality(self.settings)
        engine.ExportAccessibilityResults(
            self.settings, folder_prefix="accessibility_",
            file_name=self.settings.output_file_name)
        self.accessibility = engine

    def RunFlow(self):
        engine = _StubFlow(self.topology)
        engine.Centrality(self.settings)
        engine.ExportFlowResult(
            self.settings, folder_prefix="flow_",
            file_name=self.settings.output_file_name)
        self.flow = engine

    def RunODM(self, format: str = "Sqlite", speed: float = 5.0,
               file_name: str = None):
        engine = _StubODM(self.topology)
        engine.settings = self.settings
        engine.OD_Matrix(search_radius=self.settings.search_radius)
        engine.ExportODM(
            self.settings, folder_prefix="ODM_",
            file_name=file_name or self.settings.output_file_name,
            format=format, speed=speed)
        return engine
'''

STUB_PACKAGE_FILES = {
    "__init__.py": STUB_INIT,
    "Settings.py": STUB_SETTINGS,
    "Topology.py": STUB_TOPOLOGY,
    "UNA.py": STUB_UNA,
    "Engines/__init__.py": STUB_ENGINES_INIT,
    "Engines/Base.py": STUB_BASE,
}

DIST_INFO_FILES = {
    "METADATA": "Name: urban-network-analysis\nVersion: %s\n" % STUB_VERSION,
    "WHEEL": "Wheel-Version: 1.0\n",
    # direct_url.json WITHOUT dir_info.editable: a normal (non-editable)
    # local-wheel install record.
    "direct_url.json": json.dumps(
        {"url": "file:///stubs/urban_network_analysis-%s-py3-none-any.whl"
                % STUB_VERSION, "dir_info": {}}),
    "RECORD": "",  # minimal; identity module hashes are the real proof
}


def build_stub_venv(venv_root: Path) -> dict:
    """Create <venv_root>/lib/pythonX/site-packages with the stub package.

    Returns {"venv_root", "site_packages", "package_root", "dist_info"}.
    """
    site_packages = venv_root / "lib" / "python3.11" / "site-packages"
    package_root = site_packages / "urban_network_analysis"
    for relpath, content in STUB_PACKAGE_FILES.items():
        target = package_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    dist_info = site_packages / f"urban_network_analysis-{STUB_VERSION}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    for name, content in DIST_INFO_FILES.items():
        (dist_info / name).write_text(content, encoding="utf-8")
    return {
        "venv_root": venv_root,
        "site_packages": site_packages,
        "package_root": package_root,
        "dist_info": dist_info,
    }


# ----------------------------------------------------------------------
# Identity generation in a clean subprocess (no pytest-process pollution)
# ----------------------------------------------------------------------
def generate_identity(python: Path, arm: str, kind: str, out_path: Path,
                      site_packages: list[Path] | None = None,
                      source_root: Path | None = None,
                      extra_sys_path: list[Path] | None = None) -> dict:
    """Run harness.identity --make in a subprocess and return the JSON.

    The declared site-packages (and diagnostic source root) are put on the
    subprocess's sys.path: under a REAL arm interpreter those locations are
    already active, so this only emulates the arm's own resolution for the
    stub layout without changing what the guards then verify.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    sys_path = [str(p) for p in (site_packages or [])]
    if source_root is not None:
        sys_path.append(str(source_root))
    sys_path.extend(str(p) for p in (extra_sys_path or []))
    code = (
        "import sys, json, pathlib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from harness.identity import make_identity\n"
        "args = json.loads(sys.argv[2])\n"
        "for entry in args.get('sys_path', []):\n"
        "    sys.path.append(entry)\n"
        "make_identity(arm=args['arm'], kind=args['kind'],\n"
        " out_path=pathlib.Path(args['out']),\n"
        " site_packages=[pathlib.Path(p) for p in args['sp']] or None,\n"
        " expected_root=pathlib.Path(args['src']) if args['src'] else None)\n"
        "print('IDENTITY_OK')\n"
    )
    driver_args = json.dumps({
        "arm": arm,
        "kind": kind,
        "out": str(out_path),
        "sp": [str(p) for p in (site_packages or [])],
        "src": str(source_root) if source_root else None,
        "sys_path": sys_path,
    })
    completed = subprocess.run(
        [str(python), "-c", code, str(HARNESS_DIR), driver_args],
        capture_output=True, text=True, env=env, cwd=str(out_path.parent),
        timeout=300,
    )
    if completed.returncode != 0 or "IDENTITY_OK" not in completed.stdout:
        raise RuntimeError(
            f"identity generation failed:\n{completed.stdout}\n{completed.stderr}")
    with open(out_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def recompute_identity_sha(data: dict) -> dict:
    """Re-sign an identity after deliberate test edits (integrity checks)."""
    import hashlib
    sys.path.insert(0, str(HARNESS_DIR))
    from harness.identity import compute_identity_sha256
    data = dict(data)
    data.pop("identity_sha256", None)
    data["identity_sha256"] = compute_identity_sha256(data)
    return data


# ----------------------------------------------------------------------
# Real tiny GIS fixture (for diagnostic-source smoke runs against src/**)
# ----------------------------------------------------------------------
def make_real_tiny_fixture(data_dir: Path) -> list[dict]:
    """Real geopandas feather inputs for the REAL engine (diagnostic mode).

    A 3x2 node street grid in metres (EPSG:32633), 2 origins, 4 destinations —
    small enough that the real accessibility analysis completes in seconds
    even under NUMBA_DISABLE_JIT pure-Python execution.  Only the diagnostic
    mode may combine this fixture with JIT disabled; installed mode refuses
    NUMBA_DISABLE_JIT outright.
    """
    import hashlib

    import geopandas as gpd
    from shapely.geometry import LineString, Point

    data_dir.mkdir(parents=True, exist_ok=True)
    crs = "EPSG:32633"
    xs = (0, 100, 200)
    ys = (0, 100)
    lines = []
    for y in ys:
        for x in xs[:-1]:
            lines.append(LineString([(x, y), (x + 100, y)]))
    for x in xs:
        for y in ys[:-1]:
            lines.append(LineString([(x, y), (x, y + 100)]))
    network = gpd.GeoDataFrame(
        {"Geometric": [float(line.length) for line in lines]},
        geometry=lines, crs=crs)
    origins = gpd.GeoDataFrame(
        {"Count": [1.0, 1.0]},
        geometry=[Point(0, 0), Point(100, 100)], crs=crs)
    destinations = gpd.GeoDataFrame(
        {"Count": [1.0, 1.0, 1.0, 1.0]},
        geometry=[Point(200, 0), Point(200, 100),
                  Point(0, 100), Point(100, 0)], crs=crs)

    files = []
    for name, gdf in (("network", network), ("origins", origins),
                      ("destinations", destinations)):
        path = data_dir / f"{name}.feather"
        gdf.to_feather(path)
        files.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
    return files


# ----------------------------------------------------------------------
# Manifest builder
# ----------------------------------------------------------------------
def make_input_files(data_dir: Path, analysis: str = "accessibility") -> list[dict]:
    """Tiny deterministic input files (stub engine never reads them)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("network", "origins", "destinations"):
        path = data_dir / f"{name}.bin"
        path.write_bytes(f"stub-input-{name}-{analysis}\n".encode())
        import hashlib
        files.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
    return files


def make_manifest(path: Path, data_dir: Path, analysis: str = "accessibility",
                  settings_overrides: dict | None = None,
                  required_patterns: list[str] | None = None,
                  odm_options: dict | None = None,
                  input_files: list[dict] | None = None,
                  workload_id: str = "stub_tiny") -> Path:
    if input_files is None:
        input_files = make_input_files(data_dir, analysis)
    patterns = required_patterns
    if patterns is None:
        patterns = (["*.sqlite"] if analysis == "odm"
                    else ["*.feather", "*.geojson"])
    settings = {"data_folder": str(data_dir)}
    settings.update(settings_overrides or {})
    manifest = {
        "schema_version": 1,
        "workload_id": workload_id,
        "analysis": analysis,
        "input_files": input_files,
        "settings": settings,
        "required_output_patterns": patterns,
    }
    if odm_options is not None:
        manifest["odm"] = odm_options
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path


# ----------------------------------------------------------------------
# Harness subprocess runner
# ----------------------------------------------------------------------
def run_harness(out_dir: Path, manifest: Path, identity: Path,
                mode: str = "batch", jobs: int = 1, workers: int = 1,
                numba_threads: int = 1, flow_stripes: str | int = "default",
                queue_depth: int = 2, writer_limit: int | None = None,
                cpu_budget: int = 4, memory_budget_mib: int | None = None,
                timeout_s: float = 120.0, cache_root: Path | None = None,
                diagnostic_source_root: Path | None = None,
                test_fault: str | None = None,
                extra_env: dict | None = None,
                python: Path | None = None,
                timeout: float = 300.0) -> subprocess.CompletedProcess:
    """Invoke run.py exactly as the campaign would (absolute script path)."""
    if writer_limit is None:
        writer_limit = workers  # L=W default policy
    if memory_budget_mib is None:
        # Tiny self-test runs need only a small budget; keep the declared
        # value comfortably under the 80%-of-available admission cap even
        # while the laptop is under memory pressure (H02 never launches
        # large jobs).
        import psutil
        available_mib = psutil.virtual_memory().available // (1024 * 1024)
        memory_budget_mib = max(128, int(available_mib * 0.25))
    if cache_root is None:
        cache_root = out_dir.parent / (out_dir.name + "_cache")
    cmd = [
        str(python or CAMPAIGN_PYTHON), str(RUN_PY),
        "--manifest", str(manifest),
        "--arm", "stub-arm",
        "--identity", str(identity),
        "--mode", mode,
        "--jobs", str(jobs),
        "--workers", str(workers),
        "--numba-threads", str(numba_threads),
        "--flow-stripes", str(flow_stripes),
        "--queue-depth", str(queue_depth),
        "--writer-limit", str(writer_limit),
        "--cpu-budget", str(cpu_budget),
        "--memory-budget-mib", str(memory_budget_mib),
        "--timeout-s", str(timeout_s),
        "--cache-root", str(cache_root),
        "--out", str(out_dir),
    ]
    if diagnostic_source_root is not None:
        cmd += ["--diagnostic-source-root", str(diagnostic_source_root)]
    if test_fault is not None:
        cmd += ["--test-fault", test_fault]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(out_dir.parent), timeout=timeout)


def read_session(out_dir: Path) -> dict:
    with open(out_dir / "session.json", "r", encoding="utf-8") as handle:
        return json.load(handle)
