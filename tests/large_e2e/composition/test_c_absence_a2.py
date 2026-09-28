"""I00 composition: A2 absence pin (task A2R outcome: NOT_ADMITTED,
never integrated).

The composed tree must carry NO trace of the A2 local-destination-
enumeration track — asserted in BOTH frames (h04 I00 plan item R6:
source-byte scans alone prove the filter, not the runtime):

* source-byte frame: no `_a2_` token in ANY .py file of the candidate
  package tree (both arms scanned: A2 must be absent from B0 too — it
  was never integrated on either side);
* imported-namespace frame: no `_a2` attribute on any imported
  alias-namespace module (post-import dir()/sys.modules walk over every
  `una_composition_cand.*` module in sys.modules);
* behavioral: the route A2 would have replaced (the destination adjust
  path on guard-refused inputs) runs the unchanged original dispatch
  and stays byte-identical to B0 (the a1_refusal cell drives the
  original loop end-to-end through both driver surfaces).
"""
from __future__ import annotations

import os
import sys

import pytest

import fixtures_composition
from harness_composition import (assert_integrated_bytes, assert_od_bytes,
                                 b0ns, cns, record)


def _package_py_files(pkg_dir):
    out = []
    for dirpath, _dirnames, filenames in os.walk(pkg_dir):
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def test_a2_token_absent_from_both_trees():
    """Source-byte frame: no `_a2_` token in any .py file of EITHER
    import root."""
    for tag, ns in (("b0", b0ns()), ("cand", cns())):
        pkg_dir = os.path.join(ns.root, "urban_network_analysis")
        files = _package_py_files(pkg_dir)
        hits = []
        for path in files:
            with open(path, "rb") as fh:
                if b"_a2_" in fh.read():
                    hits.append(os.path.relpath(path, ns.root))
        assert hits == [], f"{tag}: _a2_ token found in {hits}"
        record("a2_token_absent", arm=tag, files_scanned=len(files))


def test_a2_attribute_absent_from_imported_namespace():
    """Imported-namespace frame: no `_a2` attribute on ANY module under
    the candidate alias in sys.modules (walks everything the composed
    import actually pulled in, not a hand-listed subset)."""
    alias = cns().alias
    mods = [m for name, m in sys.modules.items()
            if (name == alias or name.startswith(alias + "."))
            and hasattr(m, "__dict__")]
    assert len(mods) >= 8, \
        f"alias namespace unexpectedly small: {len(mods)} modules"
    hits = [m.__name__ for m in mods
            for attr in vars(m) if attr.startswith("_a2")]
    assert hits == [], f"candidate modules expose _a2 attributes: {hits}"
    record("a2_namespace_absent", modules_scanned=len(mods))


@pytest.mark.parametrize("seed", fixtures_composition.SEEDS)
def test_a2_domain_route_byte_equal(seed):
    """The dispatch A2 would have replaced (guard-refused destinations)
    runs the original loop in the composed tree: byte-identical to B0
    through both driver surfaces and both engine families."""
    arrays = fixtures_composition.a1_refusal(seed=seed)
    assert_od_bytes(arrays, f"absence_a2/a1_refusal/{seed}")
    assert_integrated_bytes(arrays, f"absence_a2/a1_refusal/{seed}")
