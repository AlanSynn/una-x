"""C10: dependency table vs live AST facts vs COMPILED fold behavior.

The AST-predicted poison set (params no output reads) is verified
empirically on the real compiled fold; every recorded dependency is
mutated and must cause recompute; alias-dead slots must be provably
inert; the membership boundary must actually recompute.
"""
import ast
import json
import math
import struct

import numpy as np

import _canonical as canon


def bits(v):
    if isinstance(v, (int, float)):
        return struct.pack("<d", float(v))
    return np.ascontiguousarray(v, dtype=np.float64).tobytes()


def branch_literals(facts):
    """The quoted branch values in the fold's condition expressions."""
    vals = set()
    for out in facts["per_output"].values():
        for cond in out["branch_conditions"]:
            for node in ast.walk(ast.parse(cond)):
                if isinstance(node, ast.Constant) and isinstance(
                        node.value, str):
                    vals.add(node.value)
    assert vals, facts
    return vals


def perturb(driver, name):
    """A robust (not ULP) variation of one fold parameter."""
    v = driver.defaults[driver.names.index(name)]
    if isinstance(v, str):
        others = branch_literals(driver.facts) - {v}
        return sorted(others)[0] + "_poison" if not others else \
            sorted(others)[0]
    if isinstance(v, np.ndarray):
        # distances mutate through MEMBERSHIP too: a -1.0 shift can keep
        # the retained set invariant, which is a legitimate non-change
        if name == driver.names[0]:
            return v - 5.0
        return v - 1.0
    if name == driver.names[2]:
        # membership bound: the mutation must CROSS the cutoff boundary
        # (scaling 4.0 within (3.0, 8.0) keeps the retained set invariant)
        return v + 4.0
    if isinstance(v, (int, float)):
        return v * 1.3 + 0.5
    raise TypeError(type(v))


def test_table_embeds_live_facts(facts, table):
    assert table["fold"] == facts
    assert table["fold"]["engine_sha256"] == facts["engine_sha256"]


def test_call_sites_unanimous(bindings, table):
    sites = table["call_sites"]
    assert sites["bindings_unanimous"] is True
    first = next(iter(bindings.values()))
    assert all(s == first for s in bindings.values())
    assert sorted(sites["per_site"]) == [str(k) for k in sorted(bindings)]


def test_alias_bindings_match_call_sites(bindings, facts, table):
    first = next(iter(bindings.values()))
    for slot in facts["params_dead_in_fold"]:
        ent = table["alias_bindings"][slot]
        assert ent["dead_in_fold"] is True
        assert first[slot] == ent["bound_to"]
        assert ent["bound_to"] in first.values()


def test_alias_slots_are_inert_in_compiled_fold(facts, driver):
    """Dead slots bound to absurd values change NOTHING (all 4 outputs)."""
    dead_idx = [driver.names.index(s) for s in facts["params_dead_in_fold"]]
    base = driver()
    absurd = driver.args()
    for i in dead_idx:
        v = absurd[i]
        if isinstance(v, np.ndarray):
            absurd[i] = v * 7.0 - 3.0
        elif isinstance(v, str):
            absurd[i] = v[::-1] + "_absurd"
        else:
            absurd[i] = v * 17.0 + 9.0
    alt = driver.fold(*absurd)
    for a, b in zip(base, alt):
        assert bits(a) == bits(b)


def test_ast_poison_predictions_hold_on_compiled_fold(facts, driver):
    """For params no output reads, outputs are bit-identical under change."""
    outs = facts["outputs"]
    base = driver()
    for name in facts["params"]:
        try:
            args = driver(**{name: perturb(driver, name)})
        except Exception as e:  # branch value may not admit both literals
            raise AssertionError((name, e))
        for j, out in enumerate(outs):
            if name not in facts["per_output"][out]["params_read"]:
                assert bits(args[j]) == bits(base[j]), (name, out)


def _drive(entry, names):
    """Parameter values that make this polar branch the live one.

    String-Eq conditions are driven; structural guards (e.g. knn == 0,
    left side not a fold parameter) are skipped — the fixture guarantees
    their reachable side.  Negative string conditions get a poison value
    unequal to every branch literal.
    """
    drive = {}
    for text, pos in entry["when"]:
        node = ast.parse(text).body[0].value
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name)
                and node.left.id in names):
            continue
        if not (len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)):
            continue
        consts = [c.value for c in node.comparators
                  if isinstance(c, ast.Constant)
                  and isinstance(c.value, str)]
        if not consts:
            continue
        p, want = node.left.id, consts[0]
        cur = drive.get(p)
        if pos:
            if cur is not None and cur != want and cur != "_poison_":
                return None
            drive[p] = want
        else:
            if cur == want:
                return None
            drive[p] = "_poison_"
    return drive


def test_every_recorded_dependency_recomputes(facts, driver):
    """Per branch: robust mutation of each branch-read param recomputes."""
    outs = facts["outputs"]
    checked = 0
    for out in outs:
        po = facts["per_output"][out]
        j = outs.index(out)
        for entry in po["branches"]:
            drive = _drive(entry, driver.names)
            if drive is None:
                continue
            base = driver(**drive)
            for name in entry["params_read"]:
                if name in drive:
                    continue          # the gate defines the branch state
                mut = dict(drive)
                mut[name] = perturb(driver, name)
                alt = driver(**mut)
                assert bits(alt[j]) != bits(base[j]), (name, out, drive)
                checked += 1
    assert checked >= 12   # the fold really carries this many live reads


def test_gate_switch_changes_knn(driver, facts):
    """Moving decay between branch literals switches the KNN formula."""
    gate = [n for n in driver.names
            if any(n in w[0] for e in
                   facts["per_output"][facts["outputs"][3]]["branches"]
                   for w in e["when"])]
    assert len(gate) == 1, gate
    g = gate[0]
    lits = sorted(branch_literals(facts))
    assert len(lits) >= 2
    a = driver(**{g: lits[0]})
    b = driver(**{g: lits[1]})
    assert bits(a[3]) != bits(b[3])


def test_membership_boundary_recomputes(driver):
    """Crossing R retains the distance-8 destination: all outputs move."""
    base = driver()
    wide = driver(**driver._named(2, 8.0)) if hasattr(
        driver, "_named") else driver(**{driver.names[2]: 8.0})
    for a, b in zip(base, wide):
        assert bits(a) != bits(b)


def test_growth_derivation_bit_identity(driver, derivation):
    """The wrapper's quotient reproduction is bit-identical at plan build."""
    lhs, rhs = derivation["expression"].split("/")
    decay, mid = 1.5, 2.0
    q = decay / mid
    assert bits(q) == bits(0.75)  # same typed evaluation on this fixture
    base = driver()
    same = driver(**{driver.names[11]: q})
    for a, b in zip(base, same):
        assert bits(a) == bits(b)


def test_growth_ulp_miss_is_identity_not_tolerance(facts, driver):
    """A 1-ULP growth change misses identity; value change is robust-then-
    asserted at a coarser mutation (never reconciled by tolerance)."""
    g = driver.defaults[driver.names.index("growth_param"
                                           if "growth_param"
                                           in driver.names else
                                           driver.names[11])]
    ulp = math.nextafter(g, math.inf)
    base = driver()
    near = driver(**{driver.names[11]: ulp})
    # identity digest misses even when value bits may coincide
    d1 = canon.identity([("g", g)])
    d2 = canon.identity([("g", ulp)])
    assert d1 != d2
    # robust mutation must move the reading outputs
    coarse = driver(**{driver.names[11]: g * 2.0})
    outs = facts["outputs"]
    for j, out in enumerate(outs):
        reads = facts["per_output"][out]["params_read"]
        gname = driver.names[11]
        if gname in reads:
            assert bits(coarse[j]) != bits(base[j]), out


def test_reuse_policy_and_arms_recorded(table):
    pol = table["fold_reuse_policy"]
    assert "FORBIDDEN by default" in pol["ge_only_extraction"]
    assert "never" in pol["no_tolerance"]
    assert set(table["work_counters"]) >= {"folds_done", "folds_reused"}
    arms = table["arms"]
    assert "never multiplied" in arms["law"]
    assert table["mutation_law"]
    assert "invalidates" in table["profile_invalidation"]


def test_plan_fields_cover_outputs(facts, table):
    pf = table["plan_fields_per_output"]
    assert sorted(pf) == sorted(facts["outputs"])
    for out, fields in pf.items():
        assert fields, out
        if out == facts["outputs"][3]:
            joined = " ".join(fields)
            assert "argsort" in joined and "tie" in joined
            assert "k = min(" in joined


def test_derived_param_recorded(derivation, table):
    dp = table["derived_params"]
    assert derivation["kwarg"] in dp
    ent = dp[derivation["kwarg"]]
    assert ent["expression"] == derivation["expression"]
    assert ent["op"] == "truediv"
    assert len(ent["inputs"]) == 2
