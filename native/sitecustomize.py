"""N00 (una_10x) spawn-worker opt-in for the native trace producer.

``site`` auto-imports this module in every interpreter whose startup
sys.path includes ``native/`` (it rides PYTHONPATH).  It exists because
parallel rows build traces inside SPAWNED row workers, which re-import
the engine fresh: the parent's process-local ``install()`` reaches
nothing, and the engagement counters are per-process.  Workers opt in
through the SAME install() contract, strictly gated by UNA_NATIVE_TRACE:

  absent/unset  inert - one env read, no imports (default numba route
                byte-identical and default; non-leg processes unaffected)
  "1"           import trace_build_native and install() the cohort hook
                (native arm legs)
  "0"           import trace_build_native WITHOUT installing - matched
                wiring: the numba control carries the same module-load
                cost in its workers, with the hook never installed

Each installing worker reports its FINAL engagement counters to stderr
at exit (``[n00-native-engagement] {...}``); the leg driver parses those
lines into the leg JSON, so the dispatch counters are recorded live per
leg in the evidence (team-lead ruling, N00 Phase 3).
"""
import os

_STATE = os.environ.get("UNA_NATIVE_TRACE")
if _STATE is not None:
    import trace_build_native as _tbn

    if _STATE == "1":
        _tbn.install()

        import atexit

        @atexit.register
        def _report_engagement():
            # Installed workers only: the numba control (state "0") must
            # carry NO engagement evidence lines at all.  A failure to
            # report must never mask the worker's real exit status.
            try:
                import json
                import sys
                print("[n00-native-engagement] "
                      + json.dumps(_tbn.engagement()),
                      file=sys.stderr, flush=True)
            except Exception:
                pass
