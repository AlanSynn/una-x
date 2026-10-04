"""C12 replay planner laws (campaign una_10x, dossier 05).

plan_replay never multiplies arms: each output of each job either
reuses one canonical source or folds exactly once; four unsourced
outputs collapse to ONE combined fold; a changed destination-weight
vector blocks every output; allow_split=False selects the safe
trace-only continuation (one combined fold per distinct MetricID).
"""
import pytest

from conftest import metric_job, mk_case
from urban_network_analysis.Engines import _metric_reuse_scratch as mrs

ALL_OUTS = ("reach", "gravity_exponential", "gravity_logistic",
            "knn_access")


def jobs_two(**over):
    base = mk_case(n=16, seed=19)
    second = mk_case(n=16, seed=19)
    for k, v in over.items():
        second[k] = v
    return [metric_job(base), metric_job(second)]


def test_identical_metric_bundle():
    jobs = jobs_two()
    assert jobs[0] is not jobs[1]
    plans = mrs.plan_replay(jobs, allow_split=True)
    assert plans[0]["mode"] == "full_fold"
    assert plans[1]["mode"] == "bundle"
    assert plans[1]["source"] == 0
    assert set(plans[1]["outputs"]) == set(ALL_OUTS)
    assert all(v == 0 for v in plans[1]["outputs"].values())
    assert plans[1]["split_folds"] == []
    assert plans[0]["metric_id"] == plans[1]["metric_id"]


def test_beta_change_split_reach_and_gl_reuse():
    """gravity_beta is read by ge and (exponential) knn only: reach and
    gl reuse; ge and knn fold."""
    jobs = jobs_two(gravity_beta=2.5)
    plans = mrs.plan_replay(jobs, allow_split=True)
    p = plans[1]
    assert p["mode"] == "split"
    assert p["outputs"]["reach"] == 0
    assert p["outputs"]["gravity_logistic"] == 0
    assert set(p["outputs"]) == {"reach", "gravity_logistic"}
    assert sorted(p["split_folds"]) == sorted(
        ["_gravity_exponential_only", "_knn_access_only"])
    assert len(p["split_folds"]) + len(p["outputs"]) == 4


def test_growth_change_split_gl_only_folds():
    """gravity_growth_rate is read by gl only (knn here is
    exponential, whose branch reads beta/plateau, not growth)."""
    jobs = jobs_two(gravity_growth_rate=1.5)
    plans = mrs.plan_replay(jobs, allow_split=True)
    p = plans[1]
    assert p["mode"] == "split"
    assert set(p["outputs"]) == {"reach", "gravity_exponential",
                                 "knn_access"}
    assert p["split_folds"] == ["_gravity_logistic_only"]
    assert len(p["split_folds"]) + len(p["outputs"]) == 4


def test_weights_change_blocks_everything_collapses_to_full_fold():
    w = [0.7, 1.1, 2.3] + list(mk_case(n=16, seed=19)["d_weights"][3:])
    jobs = jobs_two(d_weights=w)
    plans = mrs.plan_replay(jobs, allow_split=True)
    p = plans[1]
    assert p["mode"] == "full_fold"
    assert p["split_folds"] == []
    assert p["outputs"] == {}


def test_cutoff_change_blocks_everything():
    jobs = jobs_two(cutoff=5.0)
    plans = mrs.plan_replay(jobs, allow_split=True)
    assert plans[1]["mode"] == "full_fold"


def test_branch_change_reuses_all_scalar_outputs():
    """A decay-branch change only blocks knn_access; reach/ge/gl reuse
    the previous canonical job."""
    jobs = jobs_two(knn_decay="linear")
    plans = mrs.plan_replay(jobs, allow_split=True)
    p = plans[1]
    assert p["mode"] == "split"
    assert set(p["outputs"]) == {"reach", "gravity_exponential",
                                 "gravity_logistic"}
    assert p["split_folds"] == ["_knn_access_only"]
    assert len(p["split_folds"]) + len(p["outputs"]) == 4


def test_other_branch_knn_ignores_beta():
    """On a shared 'other' decay branch, knn reads neither beta nor
    plateau: a beta change leaves knn reusable against the previous
    same-branch canonical job while ge folds."""
    a, b = jobs_two()
    lin = metric_job(mk_case(n=16, seed=19, knn_decay="linear"))
    lin_beta = metric_job(mk_case(n=16, seed=19, knn_decay="linear",
                                  gravity_beta=2.5))
    plans = mrs.plan_replay([a, lin, lin_beta], allow_split=True)
    p = plans[2]
    assert p["mode"] == "split"
    assert p["outputs"]["reach"] == 0
    assert p["outputs"]["gravity_logistic"] == 0
    assert p["outputs"]["knn_access"] == 1
    assert p["split_folds"] == ["_gravity_exponential_only"]
    assert len(p["split_folds"]) + len(p["outputs"]) == 4


def test_allow_split_false_is_trace_only_continuation():
    jobs = jobs_two(gravity_beta=2.5)
    plans = mrs.plan_replay(jobs, allow_split=False)
    assert plans[0]["mode"] == "full_fold"
    assert plans[1]["mode"] == "full_fold"
    assert all(p["split_folds"] == [] and p["outputs"] == {}
               for p in plans)


def test_plans_never_share_mutable_structure():
    jobs = jobs_two()
    jobs += [metric_job(mk_case(n=16, seed=19, gravity_beta=3.0))]
    plans = mrs.plan_replay(jobs, allow_split=True)
    assert plans[0]["outputs"] is not plans[1]["outputs"]
    assert plans[0]["outputs"] is not plans[2]["outputs"]
    assert plans[0]["split_folds"] is not plans[1]["split_folds"]


def test_job_order_independence_of_partition():
    """The bundle PARTITION is order-independent: the same distinct
    metric inputs group together regardless of arrival order (which
    job is canonical is a first-arrival rule, not part of the
    partition)."""
    a, b = jobs_two()
    c = metric_job(mk_case(n=16, seed=21))
    fwd = mrs.plan_replay([a, b, c], allow_split=True)
    rev = mrs.plan_replay([c, b, a], allow_split=True)
    groups_fwd = {}
    groups_rev = {}
    for i, p in enumerate(fwd):
        groups_fwd.setdefault(p["metric_id"], set()).add(i)
    for i, p in enumerate(rev):
        groups_rev.setdefault(p["metric_id"], set()).add(i)
    # same distinct inputs group the same way: equal group-size
    # multiset (raw indices are arrival-order-relative)
    assert sorted(map(len, groups_fwd.values())) == \
        sorted(map(len, groups_rev.values()))
    assert len(groups_fwd) == len(groups_rev) == 2
    # within a group, exactly one canonical plan folds and the rest
    # bundle
    for groups in (groups_fwd, groups_rev):
        for members in groups.values():
            plans = [fwd[m] if groups is groups_fwd else rev[m]
                     for m in members]
            modes = [p["mode"] for p in plans]
            assert modes.count("bundle") == len(members) - 1


def test_work_counters_never_multiply():
    """Arm accounting: bundle amortizes to one fold total; the split
    plan folds each unsourced output exactly once; no-reuse is one
    combined fold per distinct metric input."""
    a, b = jobs_two()
    one = mrs.plan_replay([a], allow_split=True)
    both = mrs.plan_replay([a, b], allow_split=True)
    # one distinct job: a single full fold serves all four outputs
    assert len(one[0]["split_folds"]) == 0
    # identical second job adds ZERO folds
    assert both[1]["mode"] == "bundle"
    # a beta-changed third job folds exactly its two unsourced outputs
    c = metric_job(mk_case(n=16, seed=19, gravity_beta=2.5))
    three = mrs.plan_replay([a, b, c], allow_split=True)
    assert len(three[2]["split_folds"]) == 2
    assert len(three[2]["outputs"]) == 2
    # allow_split=False would run TWO combined folds for the same set
    safe = mrs.plan_replay([a, b, c], allow_split=False)
    assert sum(1 for p in safe if p["mode"] == "full_fold") == 2
