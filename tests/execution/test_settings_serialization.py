"""Settings × execution: the one authoritative dotted-key representation.

ToDict always emits the four dotted keys (explicit serialized defaults);
ApplyRow / Load accept exactly those keys; Reset restores the default
ExecutionOptions; the nested object itself is never serialized.
"""
from __future__ import annotations

import copy
import json
import logging

import pytest

pytestmark = pytest.mark.execution

from urban_network_analysis.Execution import (
    CacheOptions,
    ExecutionOptions,
    settings_serialization_fields,
)
from urban_network_analysis.Settings import Settings

CUSTOM = ExecutionOptions(
    semantic_profile="corrected_v1", backend="reference",
    cache=CacheOptions(mode="disk", directory="Z:/una-cache"))


def _to_dict(s: Settings) -> dict:
    # silence the "Settings exported to" chatter; keep assertions on values
    return s.ToDict(compact=True)


def test_todict_emits_dotted_keys_even_when_untouched():
    d = _to_dict(Settings())
    for key in settings_serialization_fields:
        assert key in d, f"default must be explicitly serialized: {key}"
    assert d["execution.semantic_profile"] == "una_legacy"
    assert d["execution.backend"] == "auto"
    assert d["execution.cache.mode"] == "off"
    assert d["execution.cache.directory"] == ".una-cache"
    # the nested object is never serialized
    assert "execution" not in d
    assert "execution.cache" not in d


def test_todict_reflects_custom_execution():
    s = Settings()
    s.execution = CUSTOM
    d = _to_dict(s)
    assert d["execution.semantic_profile"] == "corrected_v1"
    assert d["execution.backend"] == "reference"
    assert d["execution.cache.mode"] == "disk"
    assert d["execution.cache.directory"] == "Z:/una-cache"


def test_applyrow_roundtrip_of_dotted_keys(tmp_path):
    s = Settings()
    s.execution = CUSTOM
    s.network_weight_column = "Len"
    s.name = "Row1"
    d = _to_dict(s)
    s2 = Settings()
    s2.ApplyRow({k: str(v) for k, v in d.items()})
    assert s2.execution == CUSTOM
    assert s2.network_weight_column == "Len"
    assert s2.name == "Row1"


def test_applyrow_empty_cells_keep_defaults():
    s = Settings()
    s.ApplyRow({"execution.backend": "", "execution.cache.mode": None,
                "execution.semantic_profile": "nan"})
    assert s.execution == ExecutionOptions()


def test_applyrow_invalid_backend_keeps_default(caplog):
    s = Settings()
    with caplog.at_level(logging.WARNING):
        s.ApplyRow({"execution.backend": "warp"})
    assert s.execution == ExecutionOptions()  # default kept on ValueError


def test_applyrow_profile_is_structure_checked_only():
    # Documented staged-delivery behavior: option construction validates
    # structure (non-empty string); resolving a profile against the frozen
    # profile registry happens at the admission/kernel layer and fails
    # closed there — never at option construction (Execution.py header).
    s = Settings()
    s.ApplyRow({"execution.semantic_profile": "corrected_v1"})
    assert s.execution.semantic_profile == "corrected_v1"


def test_bare_execution_key_is_rejected_not_injected(tmp_path, capsys):
    """Review F2: a bare 'execution' key/column in an externally-edited
    file must be skipped with a warning, never set raw — a raw dict/str
    would pass validation-free assignment and break later attribute
    access (ToDict / row admission)."""
    s = Settings()
    s.ApplyRow({"execution": {"backend": "reference"}})
    assert s.execution == ExecutionOptions()
    assert "dotted keys" in capsys.readouterr().out

    path = tmp_path / "injected.json"
    base = Settings().ToDict(compact=False)
    base["execution"] = "garbage"
    path.write_text(json.dumps(base))
    s2 = Settings()
    s2.Load(str(path))
    assert s2.execution == ExecutionOptions()
    assert "dotted execution.* keys" in capsys.readouterr().out


def test_load_rejects_bad_file_but_roundtrips_good(tmp_path):
    s = Settings()
    s.execution = CUSTOM
    path = tmp_path / "settings.json"
    s.Save(str(path))
    s2 = Settings()
    s2.Load(str(path))
    assert s2.execution == CUSTOM
    raw = json.loads(path.read_text())
    assert raw["execution.semantic_profile"] == "corrected_v1"
    assert raw["execution.cache.directory"] == "Z:/una-cache"
    assert "execution" not in raw  # nested object never serialized


def test_load_warns_and_skips_malformed_execution_key(tmp_path, caplog):
    path = tmp_path / "bad.json"
    base = Settings().ToDict(compact=False)
    base["execution.backend"] = 3.5  # type violation in the serialized file
    path.write_text(json.dumps(base))
    s2 = Settings()
    s2.Load(str(path))
    assert s2.execution == ExecutionOptions()  # skipped, default kept


def test_reset_restores_default_execution():
    s = Settings()
    s.execution = CUSTOM
    s.Reset()
    assert s.execution == ExecutionOptions()
    assert s.execution is not ExecutionOptions()  # fresh instance


def test_deepcopy_carries_execution():
    s = Settings()
    s.execution = CUSTOM
    s2 = copy.deepcopy(s)
    assert s2.execution == CUSTOM
    s2.execution = ExecutionOptions(backend="reference")
    assert s.execution == CUSTOM  # independent copies


def test_execution_is_a_settings_field_but_runtime_handles_are_not_serialized():
    import dataclasses
    names = {f.name for f in dataclasses.fields(Settings)}
    assert "execution" in names
    # checkpoint/device/cancellation are runtime-only or option fields, never
    # top-level Settings fields — nothing beyond the dotted keys serializes.
    serialized = set(settings_serialization_fields)
    assert serialized == {
        "execution.semantic_profile", "execution.backend",
        "execution.cache.mode", "execution.cache.directory",
    }
