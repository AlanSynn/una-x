"""Typed validation errors for the corrected_v1 reference profile.

corrected_v1 refuses invalid cost domains before any search runs
(dossier 02 #4): invalid inputs receive explicit typed errors, never
unbounded queues, silent row drops, or sentinel-labeled results.  These
errors are part of the corrected profile's public contract; legacy
profiles intentionally keep their pinned behavior and never raise them.
"""
from __future__ import annotations


class CorrectedValidationError(ValueError):
    """Base class for corrected_v1 input-domain rejections."""


class NonFiniteCostError(CorrectedValidationError):
    """A cost, weight, split, or radius is NaN or infinite."""


class NegativeCostError(CorrectedValidationError):
    """A cost, weight, split, or radius is negative.

    corrected_v1 admits finite nonnegative graphs only (NUMERICS.md
    validation fixtures); negative cycles are outside the declared
    domain rather than silently accepted.
    """


class MalformedGraphError(CorrectedValidationError):
    """The CSR structure is not a well-formed adjacency structure
    (pointer bounds, offset monotonicity, neighbor range)."""


class OutOfRangeTerminalError(CorrectedValidationError):
    """A terminal (origin/destination endpoint) index is outside the
    network-node domain."""


class ParameterDomainError(CorrectedValidationError):
    """A model parameter is outside its declared domain (e.g. a decay
    coefficient that is negative or non-finite).  Legacy profiles accept
    such parameters silently and can emit non-finite accessibility;
    corrected_v1 refuses them (BUG-TEA-NEGATIVE-DECAY-OVERFLOW)."""


__all__ = [
    "CorrectedValidationError",
    "NonFiniteCostError",
    "NegativeCostError",
    "MalformedGraphError",
    "OutOfRangeTerminalError",
    "ParameterDomainError",
]
