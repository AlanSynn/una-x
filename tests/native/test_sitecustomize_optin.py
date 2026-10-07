"""N00 sitecustomize opt-in: strict inertness, both directions.

The spawn-worker mechanism (native/sitecustomize.py) is exercised here
in FRESH interpreters — subprocesses, because ``site`` imports
sitecustomize only at startup.  Each case runs a probe script with
PYTHONPATH=src:native (the leg frame) and one controlled variable,
UNA_NATIVE_TRACE, asserting the team-lead ruling's condition 1:

  unset   hook stays None, engagement counters stay zero
  "1"     hook installed (== native_build_trace), a REAL dispatch
          through the installed hook moves the counters, exactly one
          atexit engagement report appears on stderr
  "0"     matched module load: module imported, hook NEVER installed,
          counters zero, and NO engagement report (the control arm's
          zero-report law)
  broken  UNA_NATIVE_TRACE=1 with a shadowing trace_build_native that
          raises on import: the interpreter must still start and run
          the probe (sitecustomize must never raise out of site
          initialization — a raising shim would wedge spawn workers
          exactly like the pyarrow gap did)

Frame facts this file relies on (verified live, 2026-10-07): venv-n00's
interpreter has a framework-level sitecustomize.py (homebrew python
3.11) that native/sitecustomize.py SHADOWS in every leg process
(PYTHONPATH precedes site-packages); the legs ran green under that
shadowing, so the tests reproduce the leg frame exactly.

Run per-directory in the dedicated venv (never combined):

    venv-n00/bin/python -m pytest tests/native/test_sitecustomize_optin.py -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import NATIVE_DIR, REPO, REPO_SRC, ring_case

# Adjudication marker lines (parsed from the probe's stdout / stderr).
PROBE_MARK = "PROBE "
ENGAGE_MARK = "[n00-native-engagement] "

PROBE_SRC = """\
import json, sys
import numpy as np
from urban_network_analysis.batch import cohort
try:
    import trace_build_native as tbn
    tbn_ok = True
except Exception:
    tbn = None
    tbn_ok = False
out = {"tbn_ok": tbn_ok,
       "hook_installed": cohort._NATIVE_BUILD_HOOK is not None,
       "hook_is_twin": tbn_ok and
       cohort._NATIVE_BUILD_HOOK is tbn.native_build_trace}
if tbn_ok:
    out["before"] = tbn.engagement()
hook = cohort._NATIVE_BUILD_HOOK
if hook is not None:
    z = np.load(sys.argv[1])
    cutoff = z["cutoff"]
    rec = hook(z["o"], z["ow"], z["ptr"], z["vec"], z["wts"],
               z["flag"], z["d"], z["dw"], z["dwt"],
               cutoff.item() if cutoff.shape == () else cutoff)
    out["complete"] = bool(rec.complete)
if tbn_ok:
    out["after"] = tbn.engagement()
print("PROBE " + json.dumps(out))
"""

# Tripwire: the marker the parent parses must be the one the probe
# prints (never a divergent retyped literal); PROBE_MARK carries its
# own trailing space, so the template adds none.
assert ('print("%s" + json.dumps(out))' % PROBE_MARK) in PROBE_SRC


@pytest.fixture(scope="module")
def case_npz(tmp_path_factory):
    """The probe dispatch case, built by the suite's single-source
    ring builder and handed to the child as an .npz (never retyped
    inside the probe)."""
    case = ring_case(seed=23, origins=8, d_count=20, v=60, extra=40)
    path = tmp_path_factory.mktemp("probe") / "case.npz"
    np.savez(path, o=case["o"], ow=case["ow"], ptr=case["ptr"],
             vec=case["vec"], wts=case["wts"], flag=case["flag"],
             d=case["d"], dw=case["dw"], dwt=case["dwt"],
             cutoff=case["cutoff"])
    return path


def _run_probe(tmp_path, case_npz, env_value, shadow_dir=None):
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE_SRC)
    parts = ([str(shadow_dir)] if shadow_dir else []) \
        + [str(REPO_SRC), str(NATIVE_DIR)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.pop("UNA_NATIVE_TRACE", None)
    if env_value is not None:
        env["UNA_NATIVE_TRACE"] = env_value
    # NUMBA_CACHE_DIR / PYTHONDONTWRITEBYTECODE arrive via conftest's
    # os.environ edits (owner scratch, no pyc writes) - inherited.
    proc = subprocess.run([sys.executable, str(probe), str(case_npz)],
                          env=env, capture_output=True, text=True,
                          timeout=900)
    lines = [ln for ln in proc.stdout.splitlines()
             if ln.startswith(PROBE_MARK)]
    reports = [ln for ln in proc.stderr.splitlines()
               if ln.startswith(ENGAGE_MARK)]
    return {"proc": proc, "probe": json.loads(lines[-1][len(PROBE_MARK):])
            if lines else None, "reports": reports}


def _assert_inert(res):
    assert res["proc"].returncode == 0, res["proc"].stderr[-2000:]
    assert res["probe"] is not None, "probe never printed"
    assert res["probe"]["hook_installed"] is False
    assert res["probe"]["tbn_ok"] is True
    assert res["probe"]["before"]["dispatches"] == 0
    assert res["probe"]["after"]["dispatches"] == 0
    assert res["reports"] == [], \
        "inert interpreter must carry no engagement evidence lines"


def test_env_unset_inert(tmp_path, case_npz):
    """Env unset: one env read, hook never installed, zero counters -
    the default numba route is untouched by the shim's mere presence
    on PYTHONPATH."""
    res = _run_probe(tmp_path, case_npz, env_value=None)
    _assert_inert(res)


def test_env_zero_matched_load_inert(tmp_path, case_npz):
    """UNA_NATIVE_TRACE=0 (the numba control arm): the module LOADS
    (matched wiring) but nothing installs, counters stay zero, and no
    engagement report exists to trip the control's zero-report law."""
    res = _run_probe(tmp_path, case_npz, env_value="0")
    _assert_inert(res)


def test_env_one_installs_and_dispatch_moves_counters(tmp_path, case_npz):
    """UNA_NATIVE_TRACE=1: hook installed (the native twin), a real
    dispatch through the INSTALLED hook moves dispatches/origins by the
    record's own numbers, and exactly one per-worker engagement report
    lands on stderr at exit."""
    res = _run_probe(tmp_path, case_npz, env_value="1")
    assert res["proc"].returncode == 0, res["proc"].stderr[-2000:]
    probe = res["probe"]
    assert probe is not None, "probe never printed"
    assert probe["hook_installed"] is True
    assert probe["hook_is_twin"] is True
    assert probe["complete"] is True
    assert probe["after"]["dispatches"] == probe["before"]["dispatches"] + 1
    assert probe["after"]["origins"] == probe["before"]["origins"] + 8
    assert len(res["reports"]) == 1, res["reports"]
    report = json.loads(res["reports"][0][len(ENGAGE_MARK):])
    assert report["dispatches"] >= 1, report
    assert report["origins"] >= 8, report
    assert report["errors"] == 0, report


def test_broken_import_never_raises(tmp_path, case_npz):
    """The never-raise law: with UNA_NATIVE_TRACE=1 and a shadowing
    trace_build_native that raises on import, the interpreter must
    still start and run the probe (hook uninstalled, no report, no
    site-error noise) - the exact spawn-worker wedge shape ruled out
    by condition 1."""
    shadow = tmp_path / "shadow_native"
    shadow.mkdir()
    (shadow / "trace_build_native.py").write_text(
        "raise ImportError('n00 test shadow: import must not propagate')\n")
    res = _run_probe(tmp_path, case_npz, env_value="1", shadow_dir=shadow)
    assert res["proc"].returncode == 0, res["proc"].stderr[-2000:]
    probe = res["probe"]
    assert probe is not None, \
        "interpreter failed to start under a broken shim import"
    assert probe["tbn_ok"] is False
    assert probe["hook_installed"] is False
    assert res["reports"] == []
    stderr = res["proc"].stderr
    assert "Error in sitecustomize" not in stderr
    assert "Traceback" not in stderr, stderr[-2000:]
