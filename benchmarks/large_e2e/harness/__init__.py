"""UNA large-e2e campaign benchmark harness (task H02).

This package implements the frozen CLI contract in
campaigns/una_large_e2e/HARNESS.md.  It is harness code only: it never
modifies the production package, never adds a hidden source arm, and runs
identically for the B0 baseline arm and every candidate arm.

Design rules enforced across the package:
  * One shared analysis dispatcher (harness.dispatch) for single and batch
    modes and for every arm.
  * Installed-arm identity guards (harness.identity) reject PYTHONPATH,
    editable/.pth and repository shadowing using path-resolution checks,
    never string prefixes.
  * Every absolute path enters through CLI arguments; nothing is inferred
    from the laptop layout.
  * Diagnostic source runs use the distinctly named --diagnostic-source-root
    mode and structurally cannot emit installed-qualified records.
"""
