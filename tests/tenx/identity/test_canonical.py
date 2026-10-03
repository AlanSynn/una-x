"""C10: canonical frame laws (tests/tenx/identity).

The encoder's laws are the identity spec's foundation: bit identity for
floats, order sensitivity for fields and lists, type distinctness.
"""
import math
import struct

import pytest

import _canonical as canon


def test_ulp_flip_changes_identity():
    v = 0.75
    up = math.nextafter(v, math.inf)
    assert v != up and struct.pack("<d", v) != struct.pack("<d", up)
    a = canon.identity([("x", v)])
    b = canon.identity([("x", up)])
    assert a != b


def test_signed_zero_is_visible():
    assert canon.identity([("x", 0.0)]) != canon.identity([("x", -0.0)])


def test_int_vs_float_typed_distinct():
    assert canon.identity([("r", 4)]) != canon.identity([("r", 4.0)])


def test_field_reorder_is_an_identity_change():
    a = canon.identity([("a", 1), ("b", 2)])
    b = canon.identity([("b", 2), ("a", 1)])
    assert a != b


def test_str_vs_bytes_distinct():
    assert canon.identity([("x", "4")]) != canon.identity([("x", b"4")])


def test_bool_vs_int_distinct():
    assert canon.identity([("x", True)]) != canon.identity([("x", 1)])


def test_list_order_and_length():
    a = canon.identity([("w", [1.0, 2.0, 3.0])])
    b = canon.identity([("w", [3.0, 2.0, 1.0])])
    c = canon.identity([("w", [1.0, 2.0])])
    d = canon.identity([("w", [1.0, 2.0, 3.0])])
    assert a != b and a != c and a == d


def test_parent_embedding_propagates():
    parent = canon.identity([("g", 1)])
    child1 = canon.identity([("parent", parent), ("q", "seed")])
    parent2 = canon.identity([("g", 2)])
    child2 = canon.identity([("parent", parent2), ("q", "seed")])
    assert child1 != child2


def test_multiset_same_order_differs_original_order():
    # destination weight bits in ORIGINAL order: same multiset, reordered
    a = canon.identity([("w", [1.5, 0.5]), ("d", [1.0, 2.0])])
    b = canon.identity([("w", [0.5, 1.5]), ("d", [1.0, 2.0])])
    assert a != b


def test_exact_cutoff_typed_law():
    r4 = 4.0
    r4n = math.nextafter(r4, math.inf)
    ids = {canon.identity([("R", r4), ("t", "f")]),
           canon.identity([("R", r4n), ("t", "f")]),
           canon.identity([("R", 4), ("t", "i")])}
    assert len(ids) == 3


def test_nan_payload_distinct_and_value_semantics():
    # NaN carries payload: distinct NaN bit patterns are distinct identities
    a = canon.identity([("x", float("nan"))])
    b = canon.identity([("x", struct.unpack("<d", struct.pack("<Q", 0x7FF8_0000_0000_0001))[0])])
    assert a != b


def test_framed_bytes_roundtrip_exactness():
    v = 0.1
    back = struct.unpack("<d", canon.frame(v)[1:])[0]
    assert back == v and canon.bits(v) == canon.frame(v)[1:]


@pytest.mark.parametrize("bad", [{1: 2}, None])
def test_unframed_types_refused(bad):
    with pytest.raises(TypeError):
        canon.frame(bad)
