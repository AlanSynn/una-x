"""H00 negative tests: window admissibility (benchmarks/una_10x/window).

Every mutant is a recorded failure class from the platform harness review
(wrong dispatch, thin records, inflated N, queue/memory ledger breaches,
shadow identity, unhashed outputs, fault-status laundering).  None may
validate as a una_10x window.
"""
from __future__ import annotations

import hashlib

import pytest

from benchmarks.una_10x import WindowInvalid, validate_window
from benchmarks.una_10x.timing import PHASE_IN, PublicCallTimer
from benchmarks.una_10x.window import make_window_record

_DIGEST = hashlib.sha256(b"cohort-fixture").hexdigest()
_OUT_SHA = hashlib.sha256(b"composite-bytes").hexdigest()


def _clean_timer_snapshot():
    t = PublicCallTimer()
    t.start_primary()
    with t.span("compute"):
        pass
    t.end_primary()
    t.start_post_hash()
    t.end_post_hash()
    return t.snapshot()


def _legal(**over):
    record = {
        "run_id": "run-test",
        "harness_version": "una-10x-h00-h1",
        "campaign": "una_10x",
        "window_class": "FIRST_COHORT",
        "cohort_digest": _DIGEST,
        "timer_snapshot": _clean_timer_snapshot(),
        "dispatch_expected": "UNA.RunBatch",
        "dispatch_called": ["UNA.RunBatch"],
        "counts": {"submitted": 3, "completed": 3, "failed": 0,
                   "cancelled": 0},
        "queue": {"max_workers": 2, "queue_depth": 4, "peak_inflight": 2,
                  "bounded": True},
        "resource_policy": {"memory_limit_bytes": None},
        "identity": {
            "module_file": "/opt/venv/site-packages/urban_network_analysis"
                           "/UNA.py",
            "installed_tree_sha256": _DIGEST,
            "distribution_version": "1.0",
        },
        "memory": {"peak_is_simultaneous_tree_sum": True,
                   "peak_simultaneous_rss_bytes": 1000,
                   "per_sample": [[0, 900.0]], "samples_taken": 2},
        "outputs": [{"path": "composite.csv", "bytes": 16,
                     "sha256": _OUT_SHA}],
        "fault_status": "none",
        "export_synced": True,
    }
    record.update(over)
    return record


def test_legal_window_validates():
    validate_window(_legal())


def test_make_window_record_refuses_to_produce_invalid_windows():
    bad = _legal()
    bad["counts"]["failed"] = 1
    with pytest.raises(WindowInvalid):
        make_window_record(**{k: bad[k] for k in (
            "run_id", "window_class", "cohort_digest", "timer_snapshot",
            "dispatch_expected", "dispatch_called", "counts", "queue",
            "resource_policy", "identity", "memory", "outputs")},
            fault_status="none", export_synced=True)


@pytest.mark.parametrize("field", [
    "timer_snapshot", "counts", "queue", "identity", "memory", "outputs",
    "cohort_digest", "fault_status",
])
def test_thin_records_never_validate(field):
    record = _legal()
    del record[field]
    with pytest.raises(WindowInvalid, match="missing required fields"):
        validate_window(record)


def test_inflated_completed_count_is_invalid():
    record = _legal()
    record["counts"]["completed"] = 5   # submitted 3
    with pytest.raises(WindowInvalid, match="counts do not close"):
        validate_window(record)


@pytest.mark.parametrize("key", ["failed", "cancelled"])
def test_failed_or_cancelled_windows_are_never_valid(key):
    record = _legal()
    record["counts"][key] = 1
    record["counts"]["completed"] = 2
    with pytest.raises(WindowInvalid, match="never a valid measurement"):
        validate_window(record)


def test_zero_validated_jobs_is_invalid():
    record = _legal()
    record["counts"] = {"submitted": 0, "completed": 0, "failed": 0,
                        "cancelled": 0}
    with pytest.raises(WindowInvalid, match="zero jobs"):
        validate_window(record)


def test_queue_bound_breach_is_invalid():
    record = _legal()
    record["queue"]["peak_inflight"] = 3   # min(2, 4) = 2
    with pytest.raises(WindowInvalid, match="declared bound"):
        validate_window(record)


def test_unbounded_submission_is_invalid():
    record = _legal()
    record["queue"]["bounded"] = False
    with pytest.raises(WindowInvalid, match="unbounded"):
        validate_window(record)


def test_per_worker_memory_maxima_are_inadmissible():
    record = _legal()
    record["memory"]["peak_is_simultaneous_tree_sum"] = False
    with pytest.raises(WindowInvalid, match="whole-tree"):
        validate_window(record)
    record = _legal()
    record["memory"]["samples_taken"] = 0
    with pytest.raises(WindowInvalid, match="no samples"):
        validate_window(record)


def test_memory_limit_breach_is_invalid():
    record = _legal(resource_policy={"memory_limit_bytes": 500})
    with pytest.raises(WindowInvalid, match="exceeds"):
        validate_window(record)


def test_source_shadow_identity_is_invalid():
    record = _legal()
    record["identity"]["module_file"] = "/repo/src/urban_network_analysis" \
                                        "/UNA.py"
    with pytest.raises(WindowInvalid, match="source tree"):
        validate_window(record, source_root="/repo/src")


def test_non_installed_identity_is_invalid():
    record = _legal()
    record["identity"]["module_file"] = "/repo/wheels/unpack/UNA.py"
    with pytest.raises(WindowInvalid, match="site-packages"):
        validate_window(record)
    record = _legal()
    record["identity"]["installed_tree_sha256"] = None
    with pytest.raises(WindowInvalid, match="identity hashes incomplete"):
        validate_window(record)


def test_wrong_analysis_success_is_invalid():
    record = _legal(dispatch_expected="UNA.RunAccessibility",
                    dispatch_called=["UNA.RunODM"])
    with pytest.raises(WindowInvalid, match="wrong-analysis"):
        validate_window(record)


def test_extra_public_entry_is_invalid():
    record = _legal(dispatch_called=["UNA.RunBatch", "UNA.RunFlow"])
    with pytest.raises(WindowInvalid, match="diverge"):
        validate_window(record)


def test_bogus_window_class_and_short_digest_are_invalid():
    with pytest.raises(WindowInvalid, match="cache regime"):
        validate_window(_legal(window_class="WARM"))
    with pytest.raises(WindowInvalid, match="cohort_digest"):
        validate_window(_legal(cohort_digest="abc"))


def test_empty_or_unhashed_outputs_are_invalid():
    with pytest.raises(WindowInvalid, match="no output artifacts"):
        validate_window(_legal(outputs=[]))
    with pytest.raises(WindowInvalid, match="is empty"):
        validate_window(_legal(outputs=[{"path": "a.csv", "bytes": 0,
                                         "sha256": _OUT_SHA}]))
    with pytest.raises(WindowInvalid, match="sha256"):
        validate_window(_legal(outputs=[{"path": "a.csv", "bytes": 4,
                                         "sha256": "deadbeef"}]))


def test_unsynced_export_and_fault_status_are_invalid():
    with pytest.raises(WindowInvalid, match="fsync"):
        validate_window(_legal(export_synced=False))
    with pytest.raises(WindowInvalid, match="never valid"):
        validate_window(_legal(fault_status="deadline_cancel"))


def test_tampered_primary_wall_is_invalid():
    record = _legal()
    record["timer_snapshot"]["primary_s"] = 0.0
    with pytest.raises(WindowInvalid, match="not positive"):
        validate_window(record)
    record = _legal()
    record["timer_snapshot"]["post_hash_s"] = None
    with pytest.raises(WindowInvalid, match="post-hash wall"):
        validate_window(record)


def test_scrubbed_violations_are_recomputed_from_spans():
    # launder a leaking span: scrub the stored violation list; the
    # recompute inside validate_window must still reject the record
    record = _legal()
    snap = record["timer_snapshot"]
    ps, pe = snap["primary_start"], snap["primary_end"]
    snap["spans"].append({"name": "escape", "phase": PHASE_IN, "depth": 0,
                          "tags": {}, "start": ps + 1, "end": pe + 1})
    snap["violations"] = []
    with pytest.raises(WindowInvalid, match="escapes"):
        validate_window(record)
