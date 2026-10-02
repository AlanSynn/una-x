"""Publication-only settings must never force a numerical miss.

Output folder, file name, timestamp stamping, format flags and the CSV
delimiter are unrepresentable in the stage keys by construction: changing
them reuses every stored stage (5 hits, 0 misses) and the new publication
settings are still honored in the artifacts.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.cache.invalidation.conftest import (
    make_settings, run_access, hash_out, stats_snapshot, stats_delta)

pytestmark = pytest.mark.cache_graph


def _reuse_all(tmp_path, workload, cache_dir, **publication_over):
    s1 = make_settings(workload, tmp_path / "out1", cache_mode="disk",
                       cache_dir=cache_dir)
    run_access(s1)
    s2 = make_settings(workload, tmp_path / "out2", cache_mode="disk",
                       cache_dir=cache_dir, **publication_over)
    before = stats_snapshot(s2)
    run_access(s2)
    hits, misses = stats_delta(s2, before)
    return (hits, misses), s2


@pytest.mark.cache_graph
def test_new_output_folder_reuses_all_stages(tmp_path, workload, cache_dir):
    (hits, misses), _ = _reuse_all(tmp_path, workload, cache_dir,
                                   output_folder=str(tmp_path / "elsewhere"))
    assert (hits, misses) == (5, 0)


@pytest.mark.cache_graph
def test_output_file_name_reuses_all_stages(tmp_path, workload, cache_dir):
    (hits, misses), s2 = _reuse_all(tmp_path, workload, cache_dir,
                                    output_file_name="Published")
    assert (hits, misses) == (5, 0)
    published = [p for p, _ in hash_out(s2)]
    assert published and all(Path(p).name.startswith("Published")
                             for p in published)


@pytest.mark.cache_graph
def test_output_wstamp_reuses_all_stages(tmp_path, workload, cache_dir):
    (hits, misses), s2 = _reuse_all(tmp_path, workload, cache_dir,
                                    output_wStamp=True)
    assert (hits, misses) == (5, 0)
    assert len(list(Path(s2.output_folder).iterdir())) >= 1


@pytest.mark.cache_graph
def test_csv_delimiter_reuses_all_stages_and_reformats(tmp_path, workload,
                                                       cache_dir):
    (hits, misses), s2 = _reuse_all(tmp_path, workload, cache_dir,
                                    csv_delimiter=";")
    assert (hits, misses) == (5, 0)
    csv_files = [Path(s2.output_folder) / p for p, _ in hash_out(s2)
                 if p.endswith(".csv")]
    assert csv_files, "no csv published"
    with open(csv_files[0], newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    assert len(rows) > 1 and len(rows[0]) > 1


@pytest.mark.cache_graph
def test_dropped_format_flag_reuses_all_stages(tmp_path, workload, cache_dir):
    (hits, misses), s2 = _reuse_all(tmp_path, workload, cache_dir,
                                    output_csv=False)
    assert (hits, misses) == (5, 0)
    published = [p for p, _ in hash_out(s2)]
    assert not any(p.endswith(".csv") for p in published)
    assert any(p.endswith(".feather") for p in published)
