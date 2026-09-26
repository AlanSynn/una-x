"""F1 observed-boundary replay (proof 10): T6 — all 32 selected
captured O2 _accumulate_od_flow calls replayed through BOTH namespaces
with fresh zero buffers, byte-compared to each other and to the stored
fixture outputs; trip-volume records; budget_override negative proving
the comparator discriminates."""
from __future__ import annotations

import numpy as np

import observed_replay
from comparator import OracleMismatch, assert_array_bytes_equal
from harness_f1 import b0ns, cns, record_note


RECS = None


def _recs():
    global RECS
    if RECS is None:
        sidecar, _ = observed_replay.load_observed()
        RECS = observed_replay.selected_od_records(sidecar)
    return RECS


def _npz():
    return observed_replay.load_observed()[1]


def test_t6_replay_all_selected_records():
    """Every selected record: candidate replay outputs byte-equal the
    stored fixture outputs AND the B0 replay of the same record;
    delivered is bit-equal across arms."""
    b0 = b0ns()
    c = cns()
    npz = _npz()
    assert len(_recs()) == 32
    for i, rec in enumerate(_recs()):
        out_b = observed_replay.replay_od_flow(b0, _sidecar(), npz, rec)
        out_c = observed_replay.replay_od_flow(c, _sidecar(), npz, rec)
        assert_array_bytes_equal(out_c[0], out_b[0],
                                 f"t6/rec{i}/out_AB")
        assert_array_bytes_equal(out_c[1], out_b[1],
                                 f"t6/rec{i}/out_BA")
        assert_array_bytes_equal(out_c[2], out_b[2],
                                 f"t6/rec{i}/out_node_flow")
        assert float(out_c[3]).hex() == float(out_b[3]).hex(), \
            f"t6/rec{i}/delivered not bit-equal"
        # against the STORED fixture outputs (zero-buffer delta form)
        stored_ab = np.asarray(npz[rec["outputs"]["out_AB"]])
        stored_ba = np.asarray(npz[rec["outputs"]["out_BA"]])
        stored_node = np.asarray(npz[rec["outputs"]["out_node_flow"]])
        assert_array_bytes_equal(out_c[0], stored_ab,
                                 f"t6/rec{i}/cand_vs_stored_AB")
        assert_array_bytes_equal(out_c[1], stored_ba,
                                 f"t6/rec{i}/cand_vs_stored_BA")
        assert_array_bytes_equal(out_c[2], stored_node,
                                 f"t6/rec{i}/cand_vs_stored_node")
        stored_delivered = rec.get("delivered")
        if stored_delivered is not None:
            assert float(out_c[3]).hex() == float(stored_delivered).hex(), \
                f"t6/rec{i}/cand delivered vs stored"


_SIDECAR = None


def _sidecar():
    global _SIDECAR
    if _SIDECAR is None:
        _SIDECAR = observed_replay.load_observed()[0]
    return _SIDECAR


def test_t6_trip_volumes():
    """Captured _compute_trip_volumes calls replay byte-equal across
    arms (Python-side numerical helper compiled in both trees)."""
    b0 = b0ns()
    c = cns()
    npz = _npz()
    trip_key = next((k for k in _sidecar()["calls"]
                     if "trip" in k.lower()), None)
    if trip_key is None:
        record_note("t6_trip_volumes", {"status": "no trip records"})
        return
    recs = _sidecar()["calls"][trip_key]
    for i, rec in enumerate(recs):
        out_b = np.asarray(observed_replay.replay_trip_volumes(b0, npz, rec))
        out_c = np.asarray(observed_replay.replay_trip_volumes(c, npz, rec))
        assert_array_bytes_equal(out_c, out_b, f"t6/trip{i}")


def test_t6_budget_override_negative():
    """A mutated budget must be caught: replaying at least one captured
    call with budget_override != captured budget diverges from the
    stored outputs (comparator discriminates)."""
    b0 = b0ns()
    c = cns()
    npz = _npz()
    fired = None
    for i, rec in enumerate(_recs()):
        budget = float(rec["scalars"]["budget"])
        out_p = observed_replay.replay_od_flow(c, _sidecar(), npz, rec)
        try:
            out_m = observed_replay.replay_od_flow(
                c, _sidecar(), npz, rec, budget_override=budget * 0.75)
        except OracleMismatch:
            fired = {"record": i, "kind": "OracleMismatch-at-call"}
            break
        differs = (out_m[0].tobytes() != out_p[0].tobytes()
                   or out_m[1].tobytes() != out_p[1].tobytes()
                   or out_m[2].tobytes() != out_p[2].tobytes()
                   or float(out_m[3]).hex() != float(out_p[3]).hex())
        if differs:
            idx = _first_byte_diff(out_m[0], out_p[0])
            fired = {"record": i, "kind": "byte-divergence",
                     "first_diff_byte": idx}
            break
    assert fired is not None, ("budget_override negative did NOT fire — "
                               "comparator sensitivity failure")
    # cross-arm sanity on the same record: unperturbed arms stay equal
    rec = _recs()[fired["record"]]
    out_b = observed_replay.replay_od_flow(b0, _sidecar(), npz, rec)
    out_c = observed_replay.replay_od_flow(c, _sidecar(), npz, rec)
    assert_array_bytes_equal(out_c[0], out_b[0], "t6_negative/rec/out_AB")
    record_note("t6_budget_override", fired)


def _first_byte_diff(a, b):
    ab = np.asarray(a).tobytes()
    bb = np.asarray(b).tobytes()
    return next(i for i, (x, y) in enumerate(zip(ab, bb)) if x != y)
