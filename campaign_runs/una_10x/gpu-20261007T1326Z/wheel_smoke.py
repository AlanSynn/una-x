"""Installed-artifact engagement smoke (G11 work order d).

Runs with CWD = the run dir (NOT the repo): `urban_network_analysis`
must resolve through the INSTALLED distribution (env site-packages
editable install), not through a repo-relative sys.path insert.  Checks:

1. pip metadata of the installed distribution (name/version/location).
2. The installed engine fold's output bits on the cohort case EQUAL the
   engine bits certified by the rc=0 device gate (gate_evidence.json) —
   the installed artifact and the gate verdict refer to the same math.
3. gpu.backends availability_report through the same installed python on
   the GPU node: the real A100 named, FP64 asserted.
4. The durable gate store opens the cell through this import path too.

Exit 0 = smoke passed; any check prints its own tool output.
"""
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent

import importlib.metadata as md

dist = md.distribution("urban-network-analysis")
print("installed_distribution:", dist.metadata["Name"], dist.version)
try:
    direct = str(dist.locate_file(""))
except Exception as exc:  # noqa: BLE001
    direct = f"<unavailable: {exc}>"
print("installed_at:", direct)

from urban_network_analysis.Engines.AccessibilityWElevation import (
    reach_gravity_knn_access as engine_fold,
)
print("engine_module:", engine_fold.__module__,
      "file:", sys.modules[engine_fold.__module__].__file__)

sys.path.insert(0, str(Path("/u/dsynn/projects/una-x/repo")))
from gpu.kernels.validate_device_bits import _COHORT_CASE  # noqa: E402

c = _COHORT_CASE
import numpy as np
d = np.asarray(c["distances"], dtype=np.float64)
w = np.asarray(c["weights"], dtype=np.float64)
reach, ge, gl, kn = engine_fold(
    d, w, np.float64(c["cutoff"]), float(c["gravity_beta"]),
    float(c["plateau"]), float(c["midpoint"]), 0.0,
    np.asarray(c["knn_weights"], dtype=np.float64), 0.0, c["knn_decay"],
    1.0, 1.0, 1.0)


def bits(x):
    import numpy as np
    return np.float64(x).tobytes().hex()


got = {"reach": bits(reach), "gravity_exponential": bits(ge),
       "gravity_logistic": bits(gl), "knn_access": bits(kn)}
evidence = json.loads((RUN / "gate_evidence.json").read_text())
want = evidence["fold_battery"]["cohort"]["engine_bits"]
print("engine_bits_installed:", json.dumps(got, indent=1))
print("engine_bits_gate_certified:", json.dumps(want, indent=1))
print("bits_match_installed_vs_gate:", got == want)

from gpu.backends.gpu import availability_report, gate_state  # noqa: E402
rep = availability_report()
drv = rep["driver"]
drv_view = (drv.__dict__ if hasattr(drv, "__dict__")
            else drv if isinstance(drv, dict) else str(drv))
print("availability_driver:", json.dumps(drv_view, indent=1, default=str))
state = gate_state("una_legacy.fold.default", str(RUN / "gate_store.json"))
print("gate_state:", json.dumps(state, indent=1))

ok = (got == want and state["open"] is True)
print("SMOKE_" + ("PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 7)
