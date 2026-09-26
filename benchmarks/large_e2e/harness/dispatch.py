"""THE one shared analysis dispatcher (HARNESS.md mandatory behavior 1).

Both single and batch modes, and every arm, execute the manifest's analysis
through execute_analysis() below.  Unsupported analyses raise
UnsupportedAnalysisError — they never fall back to accessibility.

The dispatcher calls ONLY the real public methods RunAccessibility,
RunFlow and RunODM, exactly once per job, with no extra constructors and no
pre-resolved gravity cap (RunFlow's own original behavior is measured).

Tests mock a UNA-like object and assert exact per-method call counts,
including zero calls to the wrong methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SUPPORTED_ANALYSES = ("accessibility", "flow", "odm")

ANALYSIS_TO_METHOD = {
    "accessibility": "RunAccessibility",
    "flow": "RunFlow",
    "odm": "RunODM",
}

ANALYSIS_TO_ENGINE_ATTR = {
    "accessibility": "accessibility",
    "flow": "flow",
    "odm": "od",  # RunODM keeps its engine local; una.od is NOT part of the API
}

ODM_DEFAULT_FORMAT = "Sqlite"
ODM_DEFAULT_SPEED = 5.0


class UnsupportedAnalysisError(Exception):
    """Raised for any manifest analysis outside SUPPORTED_ANALYSES."""


@dataclass
class DispatchOutcome:
    analysis: str
    method_name: str
    engine_attr: str
    signature: dict = field(default_factory=dict)


def validate_analysis(analysis: object) -> str:
    """Explicit, default-free analysis validation."""
    if not isinstance(analysis, str) or analysis.strip().lower() not in SUPPORTED_ANALYSES:
        raise UnsupportedAnalysisError(
            f"unsupported analysis {analysis!r}; supported: "
            f"{list(SUPPORTED_ANALYSES)} (no default is ever applied)")
    return analysis.strip().lower()


def extract_signature(una_obj: object, analysis: str) -> dict:
    """Engine/result signature appropriate to the requested analysis.

    Duck-typed so it works against the real engines and against the tiny
    stub engine used by harness self-tests.  For accessibility the four
    metric arrays are inspected; for flow the edge/node flow arrays; for
    ODM the engine is local to RunODM so the signature records that
    explicitly instead of inventing an attribute.
    """
    analysis = validate_analysis(analysis)
    engine_attr = ANALYSIS_TO_ENGINE_ATTR[analysis]
    engine = getattr(una_obj, engine_attr, None)

    def _arr_state(value: object) -> str:
        if value is None:
            return "absent"
        try:
            length = len(value)
        except TypeError:
            return "present"
        return "empty" if length == 0 else f"len={length}"

    if analysis == "accessibility":
        attrs = {
            name: _arr_state(getattr(engine, name, None))
            for name in ("reach", "gravity_exponential", "gravity_logistic", "knn_access")
        }
    elif analysis == "flow":
        attrs = {
            name: _arr_state(getattr(engine, name, None))
            for name in ("edge_flow", "edge_flow_AB", "edge_flow_BA", "node_flow")
        }
    else:  # odm
        attrs = {"note": "RunODM keeps its engine local to the public method; "
                         "the emitted ODM files are the result signature"}
    return {
        "engine_attr": engine_attr,
        "engine_present": engine is not None,
        "engine_type": type(engine).__name__ if engine is not None else None,
        "result_attrs": attrs,
    }


def execute_analysis(una_obj: object, analysis: str,
                     odm_options: dict | None = None) -> DispatchOutcome:
    """Dispatch ONE public analysis call on a fresh UNA instance.

    No extra constructor, no extra exporter call, no cap pre-resolution:
    the timed public method is invoked exactly as a user would call it.
    """
    analysis = validate_analysis(analysis)
    method_name = ANALYSIS_TO_METHOD[analysis]

    if analysis == "odm":
        options = dict(odm_options or {})
        fmt = options.get("format", ODM_DEFAULT_FORMAT)
        speed = float(options.get("speed", ODM_DEFAULT_SPEED))
        file_name = options.get("file_name")  # None -> production default
        method = getattr(una_obj, method_name)
        if file_name is None:
            method(format=fmt, speed=speed)
        else:
            method(format=fmt, speed=speed, file_name=file_name)
    else:
        getattr(una_obj, method_name)()

    return DispatchOutcome(
        analysis=analysis,
        method_name=method_name,
        engine_attr=ANALYSIS_TO_ENGINE_ATTR[analysis],
        signature=extract_signature(una_obj, analysis),
    )
