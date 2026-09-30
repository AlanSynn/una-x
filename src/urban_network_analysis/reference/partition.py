"""Frozen canonical logical partition for corrected_v1 (NUMERICS.md).

Logical stripe count/membership and the slot fold are part of the
profile, independent of the physical worker count:

- stripe count L = min(32, number_of_origins);
- origin i belongs to stripe (i mod L);
- each stripe processes its origins in increasing origin order;
- the final fold combines stripe partials in increasing stripe ID.

The scalar reference in this package implements the same partition even
with one physical worker, so a one-worker corrected run and an
H-worker corrected run perform the identical logical operation sequence
(and identical floating-point fold).  Chunk-total folds are never
claimed equal to serial-origin folds merely because both are
deterministic; the partition IS the contract, and
PARTITION_VERSION versions it.
"""
from __future__ import annotations

# PARTITION_VERSION lives in the package __init__ (cache-key surface);
# this module implements the partition it versions.

MAX_STRIPES = 32


def canonical_stripe_count(number_of_origins: int) -> int:
    """L = min(32, number_of_origins); 0 origins -> 0 stripes."""
    if number_of_origins < 0:
        raise ValueError("number_of_origins must be nonnegative")
    return min(MAX_STRIPES, number_of_origins)


def stripe_of_origin(origin_index: int, stripe_count: int) -> int:
    """Origin i -> i mod L (requires 0 <= i and 1 <= L)."""
    if stripe_count < 1:
        raise ValueError("stripe_count must be >= 1")
    if origin_index < 0:
        raise ValueError("origin_index must be nonnegative")
    return origin_index % stripe_count


def canonical_stripes(number_of_origins: int) -> list[list[int]]:
    """Partition origins 0..n-1 into canonical stripes.

    Returns a list of L lists; each list holds its stripe's origin
    indices in increasing origin order.  Stripe k holds origins
    k, k+L, k+2L, ... (origin i -> i mod L), so each stripe is
    increasing by construction.
    """
    n = number_of_origins
    L = canonical_stripe_count(n)
    stripes: list[list[int]] = [[] for _ in range(L)]
    for i in range(n):                     # increasing origin order
        stripes[i % L].append(i)           # origin i -> i mod L
    return stripes


def stripe_fold(partials):
    """Final fold: combine per-stripe partials in increasing stripe ID.

    ``partials`` is a sequence of L scalar float64 values (one per
    stripe, in stripe-ID order).  The fold is an ordered left-to-right
    binary64 accumulation starting from 0.0 -- one RN add per stripe,
    never a tree or pairwise reduction.
    """
    acc = 0.0
    for p in partials:                     # increasing stripe ID
        acc = acc + float(p)               # explicit binary64 RN add
    return acc


def aggregate_by_canonical_partition(values, combine) -> float:
    """Reference aggregation over per-origin values.

    Splits origins 0..n-1 into canonical stripes, folds each stripe's
    values in increasing origin order with ``combine(acc, value)``
    (starting from 0.0), then applies the stripe-ID fold.  One physical
    worker performs exactly this sequence.
    """
    values = list(values)
    n = len(values)
    total = 0.0
    for stripe in canonical_stripes(n):    # increasing stripe ID
        stripe_acc = 0.0
        for i in stripe:                   # increasing origin order
            stripe_acc = combine(stripe_acc, float(values[i]))
        total = total + stripe_acc         # explicit binary64 RN add
    return total


__all__ = [
    "MAX_STRIPES",
    "canonical_stripe_count",
    "stripe_of_origin",
    "canonical_stripes",
    "stripe_fold",
    "aggregate_by_canonical_partition",
]
