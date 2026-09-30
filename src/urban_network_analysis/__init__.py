"""UNA — Urban Network Analysis.

Accessibility and pedestrian-flow metrics on street networks.

Typical use::

    import urban_network_analysis as una

    project = una.UNA()
    project.settings.data_folder  = "Boston"
    project.settings.network_file = "network.geojson"
    ...
    project.RunFlow()

The single source of truth for the package version is ``__version__``
below; pyproject.toml reads it at build time (hatchling dynamic
version), so a release is cut by editing this one line.
"""

__version__ = "2.6.0"


def _commit_date() -> str:
    """Date of the last public commit, baked in at install time.

    ``pip install git+https://...`` runs the hatch_build.py build hook,
    which writes ``_build_info.py`` with the date of the repo's most
    recent commit. Editable installs / clones don't rebuild, so for
    those we ask git directly. Falls back to "unknown".
    """
    try:
        from ._build_info import __commit_date__
        if __commit_date__ != "unknown":
            return __commit_date__
    except ImportError:
        pass
    try:
        import os
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%cs"],
            cwd=here, text=True, stderr=subprocess.DEVNULL,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def about() -> None:
    """Print version, last-commit date, and authorship information."""
    print(f"UNA — Urban Network Analysis, version {__version__} "
          f"(last public commit: {_commit_date()})")
    print("The urban_network_analysis Python package is developed by the "
          "MIT City Form Lab. Website: http://cityform.mit.edu/")


# UNA/Settings/Topology are resolved lazily (PEP 562) so that importing the
# package — in particular the compat.madina facade and the drop-in `madina`
# shim distribution built on it — does not pull the analysis stack (numba,
# sklearn) into environments that only need topology/compat functionality.
# Attribute access is unchanged: `from urban_network_analysis import UNA`
# and `urban_network_analysis.UNA` keep working.
_LAZY_EXPORTS = {"UNA": ".UNA", "Settings": ".Settings", "Topology": ".Topology"}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS))


__all__ = ["UNA", "Settings", "Topology", "__version__", "about"]
