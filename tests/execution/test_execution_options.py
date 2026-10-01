"""Contract §6 public schema: ExecutionOptions / CacheOptions exactly.

Frozen, keyword-only, validated at construction; the 13 contract fields and
6 cache fields match the frozen schema verbatim; the serialized
representation is exactly the four dotted keys.
"""
from __future__ import annotations

import dataclasses

import pytest

pytestmark = pytest.mark.execution

from urban_network_analysis.Execution import (
    CacheOptions,
    ExecutionOptions,
    settings_serialization_fields,
)


def test_execution_options_has_exactly_the_contract_fields():
    names = [f.name for f in dataclasses.fields(ExecutionOptions)]
    assert names == [
        "semantic_profile", "backend", "device", "cpu_budget",
        "threads_per_worker", "logical_reduction_plan",
        "memory_limit_bytes", "workspace_limit_bytes",
        "queue_depth", "writer_concurrency", "timeout_s",
        "cache", "checkpoint",
    ]


def test_cache_options_has_exactly_the_contract_fields():
    names = [f.name for f in dataclasses.fields(CacheOptions)]
    assert names == [
        "mode", "directory", "max_memory_bytes", "max_disk_bytes",
        "verification", "schema_version",
    ]


def test_defaults_match_the_frozen_schema():
    o = ExecutionOptions()
    assert o.semantic_profile == "una_legacy"
    assert o.backend == "auto"
    assert o.device is None
    assert o.cpu_budget is None
    assert o.threads_per_worker == 1
    assert o.logical_reduction_plan == "canonical"
    assert o.memory_limit_bytes is None
    assert o.workspace_limit_bytes is None
    assert o.queue_depth == 64
    assert o.writer_concurrency == 1
    assert o.timeout_s is None
    assert o.cache == CacheOptions()
    assert o.checkpoint is None
    c = CacheOptions()
    assert c.mode == "off"
    assert c.directory == ".una-cache"
    assert c.max_memory_bytes is None
    assert c.max_disk_bytes is None
    assert c.verification == "on_read"
    assert c.schema_version == 1


def test_kw_only_is_enforced():
    with pytest.raises(TypeError):
        ExecutionOptions("una_legacy")
    with pytest.raises(TypeError):
        CacheOptions("off")


def test_frozen_is_enforced():
    o = ExecutionOptions(backend="reference")
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.backend = "gpu"
    c = CacheOptions(mode="memory")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.mode = "off"


def test_cached_default_is_shared_and_frozen_safe():
    # a mutable default would be a classic dataclass bug; here the default is
    # an immutable CacheOptions instance, which is safe to share.
    a, b = ExecutionOptions(), ExecutionOptions()
    assert a.cache is b.cache is ExecutionOptions.__dataclass_fields__[
        "cache"].default


@pytest.mark.parametrize("bad", [
    {"backend": "tpu"},
    {"backend": ""},
    {"backend": "Auto"},           # exact literals only
    {"semantic_profile": ""},
    {"semantic_profile": "   "},
    {"semantic_profile": 3},
    {"device": 7},
    {"cpu_budget": -1},
    {"cpu_budget": 0},
    {"cpu_budget": 1.5},           # integer-typed budget
    {"cpu_budget": True},          # bool is an int subclass — rejected
    {"threads_per_worker": 0},
    {"threads_per_worker": True},
    {"threads_per_worker": 2.0},
    {"logical_reduction_plan": ""},
    {"memory_limit_bytes": -5},
    {"memory_limit_bytes": 0},
    {"workspace_limit_bytes": -1},
    {"queue_depth": 0},
    {"queue_depth": 6.4},
    {"writer_concurrency": 0},
    {"writer_concurrency": True},
    {"timeout_s": -0.5},
    {"timeout_s": 0.0},            # must be > 0, not >= 0
    {"timeout_s": "30"},
    {"cache": {"mode": "disk"}},   # raw dict, not CacheOptions
    {"checkpoint": "  "},
])
def test_invalid_execution_options_raise_value_error(bad):
    with pytest.raises(ValueError):
        ExecutionOptions(**bad)


@pytest.mark.parametrize("bad", [
    {"mode": "lru"},
    {"mode": ""},
    {"directory": ""},
    {"directory": 5},
    {"max_memory_bytes": -1},
    {"max_disk_bytes": 0},
    {"verification": "sometimes"},
    {"schema_version": 0},
    {"schema_version": -1},
    {"schema_version": 1.0},
])
def test_invalid_cache_options_raise_value_error(bad):
    with pytest.raises(ValueError):
        CacheOptions(**bad)


def test_valid_expert_options_are_accepted_verbatim():
    o = ExecutionOptions(
        semantic_profile="corrected_v1", backend="reference", device="cpu:0",
        cpu_budget=4, threads_per_worker=2, logical_reduction_plan="canonical",
        memory_limit_bytes=1 << 30, workspace_limit_bytes=1 << 28,
        queue_depth=8, writer_concurrency=2, timeout_s=120.0,
        cache=CacheOptions(mode="disk", directory="/tmp/x",
                           max_memory_bytes=1 << 26, max_disk_bytes=1 << 34,
                           verification="always", schema_version=1),
        checkpoint="runs/ck",
    )
    assert o.backend == "reference"
    assert o.cache.verification == "always"


def test_serialized_representation_is_exactly_four_dotted_keys():
    assert settings_serialization_fields == (
        "execution.semantic_profile",
        "execution.backend",
        "execution.cache.mode",
        "execution.cache.directory",
    )
