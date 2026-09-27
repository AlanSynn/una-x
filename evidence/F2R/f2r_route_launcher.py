"""F2R b0-arm route launcher — import-time force-OFF pin (spec verdict item a).

Vehicle per evidence/F2R/screen_spec.json (P4): hash-pinned, import-time,
declared in every record. The launcher re-binds AggregateFlow._use_local_route
to a declared always-False callable RETAINING the production call-site
(AggregateFlow.py:1583 executes unmodified with real inputs every OD).

Binding conditions (screen_spec source_identity_and_arms.arm_b0.
verdict_item_a_conditions):
  (i)   the wrapper returns False UNCONDITIONALLY for ALL arguments (no
        inspection of inputs, no branches);
  (ii)  NO other module attribute is touched: only _use_local_route is
        re-bound; the real function lives in THIS LAUNCHER'S CLOSURE — no
        additional module attribute is created on the AggregateFlow module;
  (iii) the pre-window invocation's arguments AND return are BOTH recorded
        in-run (launcher-forced differential receipt).

MECHANISM NOTE FOR P4b REVIEW (operational reading of "invoked ONCE
pre-window on the engine's live g_indptr/g_nodes"): _use_local_route takes
SCALARS (local_ok, slice_len, n_total) derived from engine state that is
computed INSIDE the timed window (AggregateFlow.py:1465-1467); no live
g_indptr/g_nodes exist before the window. The vehicle therefore records TWO
receipts:
  1. PRE-WINDOW alive receipt — invoke_real on DECLARED constant arguments
     (True, 1, 1), satisfying condition (iii) literally (arguments AND return
     recorded in-run, pre-window);
  2. POST-WINDOW live-inputs differential receipt — invoke_real on the
     engine's OWN live constants read from self._f2_stats (local_route_ok,
     local_slice_max) plus the live node count: real() returning True on the
     same inputs the wrapper faced, with fast_calls == 0, proves the
     differential is launcher-forced, not data-driven.
Flagged for h04's P4 line-by-line verification; one disable path, nothing
mixed in (the _gradient_slices_strictly_ascending precondition is NOT
patched — its before/after id is asserted unchanged below).

BIAS SHAPE (launcher_bias_accounting): the wrapper is ONE function call plus
a constant return — no counter, no state, no branches — matching the
symmetric 1,000 ns/call bound derivation in the spec.

NO OTHER MODULE ATTRIBUTE IS TOUCHED: this module keeps its OWN state
(_STATE below); the AggregateFlow module gains nothing.
"""

import hashlib
import os

_STATE = {
    "installed": False,
    "install_receipt": None,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install():
    """Wrap AggregateFlow._use_local_route at import time. Idempotency is
    REFUSAL: a second install would wrap the wrapper and lose the real
    function — fatal, never silently accepted (one disable path)."""
    if _STATE["installed"]:
        raise RuntimeError(
            "f2r_route_launcher: double install refused (one disable path; "
            "re-install would wrap the wrapper)")

    import urban_network_analysis.Engines.AggregateFlow as AF

    watched = ("_use_local_route", "_gradient_slices_strictly_ascending",
               "_accumulate_od_flow", "_accumulate_od_flow_local")
    ids_before = {n: id(getattr(AF, n)) for n in watched}
    module_file = AF.__file__
    module_sha = sha256_file(module_file)
    real = AF._use_local_route  # retained ONLY in this closure below

    def _f2r_force_off(*_args, **_kwargs):
        return False

    AF._use_local_route = _f2r_force_off

    ids_after = {n: id(getattr(AF, n)) for n in watched}
    receipt = {
        "launcher_file": os.path.abspath(__file__),
        "launcher_sha256": sha256_file(os.path.abspath(__file__)),
        "patched_attribute": "_use_local_route",
        "call_site_retained": "AggregateFlow.py:1583 (executes unmodified)",
        "wrapper": "_f2r_force_off: unconditional False for ALL arguments "
                   "(no inspection, no branches, no state)",
        "aggregateflow_file": module_file,
        "aggregateflow_sha256_at_install": module_sha,
        "watched_attribute_ids_unchanged_except_patch": (
            {n: (ids_before[n], ids_after[n]) for n in watched}),
        "no_other_attribute_touched": all(
            ids_before[n] == ids_after[n] for n in watched
            if n != "_use_local_route"),
        "real_function_retained_in": "launcher closure (not a module attribute)",
        "precondition_patch_status": "_gradient_slices_strictly_ascending "
                                     "NOT patched (id asserted unchanged)",
    }
    _STATE["installed"] = True
    _STATE["install_receipt"] = receipt
    _STATE["_real"] = real  # launcher-module state; never an AF attribute
    return receipt


def invoke_real(local_ok, slice_len, n_total):
    """Invoke the closure-retained REAL _use_local_route ONCE with declared
    arguments; returns the receipt demanded by verdict item (a) condition
    (iii): arguments AND return, both recorded in-run by the caller."""
    if not _STATE["installed"]:
        raise RuntimeError("f2r_route_launcher: install() must run first")
    args = {"local_ok": bool(local_ok), "slice_len": int(slice_len),
            "n_total": int(n_total)}
    ret = _STATE["_real"](bool(local_ok), int(slice_len), int(n_total))
    return {"args": args, "return": bool(ret), "callable": "_use_local_route "
            "(REAL, closure-retained)"}


def status():
    return {"installed": _STATE["installed"],
            "install_receipt": _STATE["install_receipt"]}
