"""In-process public-dispatch trace (H00: true API dispatch law).

Wraps the four public entry methods on the *imported* UNA class and
records every call as "UNA.<name>".  Engagement legs only - the timed,
uninstrumented run must never be traced (platform child marks carry the
same evidence across process boundaries; this probe covers in-process
cohort windows).

The probe restores the original attributes on exit even when the leg
raises, and it refuses to wrap anything that is not one of the legal
public entries (helper functions are not dispatch evidence).
"""
from __future__ import annotations

import functools
import importlib

from .window import LEGAL_PUBLIC_ENTRIES

_UNA_MODULE = "urban_network_analysis.UNA"
_UNA_CLASS = "UNA"


class PublicDispatchProbe:
    """Context manager recording true public-API calls on class UNA."""

    def __init__(self, una_module: str = _UNA_MODULE,
                 una_class: str = _UNA_CLASS):
        self._module_name = una_module
        self._class_name = una_class
        self.called: list[str] = []
        self._wrapped: dict[str, object] = {}
        self._cls = None

    def __enter__(self):
        mod = importlib.import_module(self._module_name)
        cls = getattr(mod, self._class_name, None)
        if cls is None:
            raise AssertionError(
                f"{self._module_name}.{self._class_name} not importable; "
                "cannot probe public dispatch")
        self._cls = cls
        for entry in LEGAL_PUBLIC_ENTRIES:
            name = entry.rsplit(".", 1)[-1]
            original = getattr(cls, name, None)
            if original is None or not callable(original):
                raise AssertionError(
                    f"public entry {entry} missing on {cls.__module__}."
                    f"{cls.__qualname__}; dispatch law cannot be checked")
            probe = self

            @functools.wraps(original)
            def recorder(*args, __orig=original, __name=name, **kwargs):
                probe.called.append(f"UNA.{__name}")
                return __orig(*args, **kwargs)

            setattr(cls, name, recorder)
            self._wrapped[name] = original
        return self

    def __exit__(self, *exc) -> bool:
        if self._cls is not None:
            for name, original in self._wrapped.items():
                setattr(self._cls, name, original)
        self._wrapped.clear()
        self._cls = None
        return False
