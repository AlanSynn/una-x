"""C10: identity algebra laws driven by the loaded contract.

IDs are built ONLY from contract-listed fields (unknown overrides are
dropped — provenance exclusion by construction).  Every sensitivity
sweep iterates the contract's own field lists; no field name is retyped.
"""
import hashlib
import math
import struct

import pytest

import _canonical as canon


def autoval(name):
    """Deterministic schema-typed value for a field name."""
    h = hashlib.sha256(name.encode()).digest()
    if name.endswith(("_bits", "_ordered", "_ids", "_weights")):
        return [h[0] / 7.0, h[1] / 7.0]
    if name.endswith(("_policy", "_semantics", "_plan", "_env",
                      "_identity", "_version", "_dtype")):
        return h[:8].hex()
    if name.endswith("_id"):
        return h[:16].hex()
    if name.endswith("_value"):
        return 4.0
    return struct.unpack("<d", h[:8])[0]


class Builder:
    """Nested ID construction per the contract's id_fields."""

    def __init__(self, contract):
        self.types = contract["id_fields"]
        self.refs = {}
        for fields in self.types.values():
            for f in fields:
                if f.endswith("_id") and f not in self.refs:
                    hits = [t for t in self.types
                            if t.lower().startswith(f[:-3])]
                    if len(hits) == 1:
                        self.refs[f] = hits[0]

    def build(self, ptype, overrides=None):
        ov = overrides or {}
        pairs = []
        for f in self.types[ptype]:
            if isinstance(ov.get(ptype), dict) and f in ov[ptype]:
                pairs.append((f, ov[ptype][f]))
            elif f in self.refs:
                pairs.append((f, self.build(self.refs[f], ov)))
            else:
                pairs.append((f, autoval(f)))
        return canon.identity(pairs)


@pytest.fixture(scope="module")
def b(contract):
    return Builder(contract)


def tkey(contract, prefix):
    """The contract's exact type-name spelling for a lowercase prefix."""
    hits = [t for t in contract["id_fields"]
            if t.lower().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def test_schema_versions_agree(contract, table):
    assert contract["schema_version"] == table["schema_version"]
    assert "una-10x" in contract["schema_version"]


def test_child_reference_fields_resolve(b, contract):
    # every type except the standalone ArtifactID is embedded somewhere
    roots = [t for t in b.types if t.lower().startswith("artifact")]
    assert len(roots) == 1
    embedded = set(b.refs.values())
    # exactly one further type is terminal (MetricID: nothing embeds it)
    terminal = set(b.types) - embedded - {roots[0]}
    assert len(terminal) == 1, (embedded, b.refs)
    assert terminal.pop().lower().startswith("metric")
    for t, fields in contract["id_fields"].items():
        for f in fields:
            if f in b.refs:
                assert b.refs[f] != t


def test_exhaustive_field_sensitivity(b, contract):
    """Every listed field, in every id type, flips the id when perturbed."""
    flips = 0
    for t, fields in contract["id_fields"].items():
        base = b.build(t)
        for f in fields:
            v = autoval(f)
            if isinstance(v, float):
                v2 = math.nextafter(v, math.inf)
            elif isinstance(v, list):
                v2 = [x + 1.0 for x in v]
            else:
                v2 = v + "!"
            assert base != b.build(t, {t: {f: v2}}), (t, f)
            flips += 1
    assert flips >= 30  # the spec carries a real field lattice


def test_parent_flip_propagates_to_children(b, contract):
    gk = tkey(contract, "graph")
    qk = tkey(contract, "query")
    tk = tkey(contract, "trace")
    mk = tkey(contract, "metric")
    victim = contract["id_fields"][gk][0]
    v = autoval(victim)
    v2 = v + 1.0 if isinstance(v, float) else v + "!"
    ov = {gk: {victim: v2}}
    assert b.build(qk, ov) != b.build(qk)
    assert b.build(tk, ov) != b.build(tk)
    assert b.build(mk, ov) != b.build(mk)


def test_cutoff_typed_exactness(b, contract):
    qk = tkey(contract, "query")
    qf = contract["id_fields"][qk]
    cutoffs = [f for f in qf
               if "cutoff" in f and "dtype" not in f]
    assert len(cutoffs) == 1, cutoffs  # the value field, not the dtype one
    c = cutoffs[0]
    ov = lambda val: {qk: {c: val}}
    ids = {b.build(qk, ov(4.0)),
           b.build(qk, ov(math.nextafter(4.0, math.inf))),
           b.build(qk, ov(4))}
    assert len(ids) == 3  # exact AND typed


def test_trace_never_shared_across_R(b, contract):
    """Filtering a larger-R trace to a smaller R is not an identity hit."""
    qk = tkey(contract, "query")
    tk = tkey(contract, "trace")
    qf = contract["id_fields"][qk]
    c = [f for f in qf if "cutoff" in f][0]
    ov = lambda val: {qk: {c: val}}
    small = b.build(tk, ov(4.0))
    large = b.build(tk, ov(12.0))
    assert small != large


def test_dest_weight_order_is_metric_identity(b, contract):
    mk = tkey(contract, "metric")
    mf = contract["id_fields"][mk]
    wf = [f for f in mf if "dest_weight" in f]
    assert len(wf) == 1, wf
    w = wf[0]
    a = b.build(mk, {mk: {w: [1.5, 0.5]}})
    c = b.build(mk, {mk: {w: [0.5, 1.5]}})
    assert a != c  # same multiset, different ORIGINAL order


def test_distinct_cohort_rows_distinct_metric_ids(b, contract):
    """The cohort's distinct-beta rows must have distinct identities."""
    mk = tkey(contract, "metric")
    mf = contract["id_fields"][mk]
    pf = [f for f in mf if "params" in f]
    assert len(pf) == 1, pf
    ids = set()
    for beta in (0.001, 0.004):   # frozen cohort linspace endpoints
        ids.add(b.build(mk, {mk: {pf[0]: [beta]}}))
    assert len(ids) == 2


def test_provenance_never_enters_identity(b, contract):
    """Unknown (provenance-like) overrides are dropped by construction."""
    gk = tkey(contract, "graph")
    base = b.build(gk)
    with_extra = b.build(gk, {gk: {
        "source_path": "/somewhere/else", "acquired_at": "2026-10-03"}})
    assert base == with_extra


def test_handle_law_in_contract(contract):
    text = " ".join(contract["laws"])
    assert "handle is not a durable cache identity" in text
    assert "pointer address is not a hash" in text
