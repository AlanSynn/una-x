"""Dossier-04 batch planner: row DAG, dependency/admission analysis, and the
deterministic ordered state-transition specification for RunBatch.

This module is PURE PLANNING.  It never runs an analysis, never touches the
network stack, never writes a file, and never mutates the caller's Settings
objects.  It inspects the batch's rows in caller order and produces an
immutable `BatchPlan` that a coordinator (the batch runtime, dossier 04 step
5-7) executes.  Everything here is decidable without running a row:

* naming decisions (output-folder default resolution, the "Results" -> name
  substitution) are captured at their original logical boundaries — the same
  decisions the serial loop makes, recorded instead of applied;
* per-row validation outcomes (missing required fields -> skip) are held as
  ORDERED outcomes: a problem on row 8 never preempts row 2, because the
  planner inspects rows without running them and records what the serial
  loop would have done;
* dependency hazards between rows are derived from canonicalized input and
  output artifacts (symlinks resolved, case-insensitive comparison — the
  conservative direction: a different spelling of the same file is NOT
  treated as independent);
* each row is admitted to worker execution only when it is PROVEN
  independent of every other row; anything unproven is reported for serial
  execution with a reason.  Independent rows are admitted even when OTHER
  rows in the same batch are serialized — a batch is never rejected
  wholesale.

Import hygiene: this module imports only the stdlib, `Settings` (a leaf
module: stdlib + numpy) and nothing from the analysis stack (UNA, Topology,
Engines) — planning a batch of 10,000 rows must not load geopandas.

Boundary of the DAG: it covers artifact hazards (input/output files,
output-conflict) and object-identity hazards (aliased Settings, non-base
types).  Row effects on the UNA instance's carried state (topology,
engines, readiness flags) are deliberately NOT row-to-row edges here —
the runtime's fresh-instance-per-worker rule owns them (dossier 04 step
5: each worker owns fresh mutable UNA/Settings/Topology), and the
coordinator reproduces carried state at commit time per ROW_TRANSITIONS.

The observable-prefix invariant the runtime must satisfy (dossier 04, formal
statement): let S_i be the public state and published artifacts after the
serial execution of rows [0, i).  After the coordinator has committed i rows
of a plan, public state and artifacts must equal S_i.  Speculative rows may
exist only in private storage.  `verify_prefix` below is the executable form
used by tests and by the runtime's self-checks; `ROW_TRANSITIONS` and
`BATCH_TRANSITIONS` name the ordered state transitions the serial loop
performs, which the parallel coordinator must reproduce at commit time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..Settings import Settings

__all__ = [
    "BATCH_TRANSITIONS",
    "ROW_TRANSITIONS",
    "BatchPlan",
    "Hazard",
    "InvariantViolation",
    "RowDescriptor",
    "RowPlan",
    "RowValidation",
    "plan_batch",
    "verify_prefix",
]

# ----------------------------------------------------------------------------
# The ordered state-transition specification (dossier 04 planner algorithm,
# step 1 and the serial loop in UNA.RunBatch it must reproduce).  These are
# the names of the transitions, in exact order; the parallel coordinator
# applies them to the public instance at COMMIT time in commit_order.
# ----------------------------------------------------------------------------

#: Per-row transitions, exactly as the serial loop performs them.  The
#: first two run for EVERY row — skipped ones included: the serial loop
#: binds the settings and resolves the output folder BEFORE the skip gate,
#: so a skipped row still mutates the instance's observable state there
#: (the runtime must apply them for skipped rows too).  A row that fails
#: validation stops at `validate_or_skip` (its later transitions never
#: run, in serial or in parallel).
ROW_TRANSITIONS = (
    "bind_settings",          # self.settings = row settings (every row)
    "resolve_output_folder",  # row value > script value > data_folder/Results
                              #   (every row — before the skip gate)
    "validate_or_skip",       # missing required fields -> skipped outcome
    "substitute_output_name", # output_file_name == "Results" -> settings.name
    "run_analysis",           # RunAccessibility / RunFlow
    "capture_composite",      # composite capture, in caller row order
    "record_outcome",         # RowOutcome appended
)

#: Batch-level transitions, in order.  `admit_execution` and the pairing
#: load happen BEFORE any row (call-level admission, contract §6).
BATCH_TRANSITIONS = (
    "admit_execution",     # ExecutionOptions admission before any scientific work
    "load_pairing",        # pairing file -> self.projects (or use existing)
    "init_compositor",     # batch composite state reset
    "run_rows",            # the rows, committed in caller order
    "finalize_composite",  # composite fold + write AFTER the last commit
    "assemble_report",     # batch_report
)


class InvariantViolation(AssertionError):
    """The observable-prefix invariant failed; `row` is the earliest
    committed row whose observable state differs from serial."""

    def __init__(self, row: int, detail: str):
        self.row = row
        super().__init__(
            f"observable-prefix invariant violated at committed row {row}: "
            f"{detail}")


# ----------------------------------------------------------------------------
# Frozen plan records
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RowDescriptor:
    """Immutable per-row snapshot of everything dependency analysis needs.

    Built WITHOUT mutating the live Settings object: `output_folder` and
    `output_stem` are the DECIDED values (the serial loop writes them onto
    the settings object; the planner records the decision so the runtime can
    apply it at the same transition boundary)."""

    index: int                  # caller order — the commit order's basis
    name: Any                   # settings.name (may be any settings value)
    semantic_profile: Any       # settings.execution.semantic_profile
    output_folder: str          # decided value ("" when no folder at all)
    output_folder_defaulted: bool   # True when the row value was absent
    output_stem: Any            # decided output_file_name ("Results" -> name)
    output_name_substituted: bool   # True when the substitution decision fired
    input_paths: tuple          # canonical absolute input file paths
    reads_timestamped_output: bool  # settings.output_wStamp
    exact_settings_type: bool   # type(settings) is Settings
    settings_type_name: str     # qualified type name (for admission reasons)
    aliased_with: tuple         # indices of rows sharing this settings object


@dataclass(frozen=True)
class RowValidation:
    """Ordered per-row validation outcome (dossier 04 step 3: validation
    errors are held, never raised across rows).  An "error" row is one the
    serial loop would fail AT — its position orders the failure; the plan
    never commits past the earliest one."""

    status: str                 # "ok" | "skipped" | "error"
    missing: tuple              # missing required field names (skips)
    detail: str


@dataclass(frozen=True)
class Hazard:
    """A dependency between two rows that forbids overlapping them."""

    with_row: int               # the other row's caller index
    kind: str                   # "output_conflict" | "read_write" | "aliased_settings"
    detail: str


@dataclass(frozen=True)
class RowPlan:
    row: RowDescriptor
    validation: RowValidation
    hazards: tuple              # all hazards against earlier AND later rows
    admission: str              # "worker" | "serial" | "skipped" | "error"
    admission_reasons: tuple    # why not a worker (empty when "worker")


@dataclass(frozen=True)
class BatchPlan:
    """The planned batch: rows in caller order, the commit/fold orders, and
    the transition specification the coordinator must reproduce."""

    analysis: str
    rows: tuple                 # RowPlan per row, caller order (skips included)
    commit_order: tuple         # caller indices of rows that will execute
    fold_order: tuple           # composite capture/fold order == commit order
    notes: tuple

    @property
    def worker_admissible(self) -> tuple:
        """Rows admitted to worker execution, in caller order."""
        return tuple(r.row.index for r in self.rows if r.admission == "worker")

    @property
    def serialized(self) -> tuple:
        """Rows that must execute on the serial route, with reasons."""
        return tuple((r.row.index, r.admission_reasons)
                     for r in self.rows if r.admission == "serial")

    @property
    def skipped(self) -> tuple:
        return tuple(r.row.index for r in self.rows if r.admission == "skipped")


# ----------------------------------------------------------------------------
# Canonicalization (symlinks + case policy)
# ----------------------------------------------------------------------------

def _canonical(path: str) -> str:
    """Canonical form of a path for dependency comparison: absolute,
    normpath'd, symlinks resolved where possible (realpath on a not-yet-
    existing path resolves the existing prefix)."""
    return os.path.realpath(os.path.normpath(os.path.abspath(path)))


def _same_path(a: str, b: str) -> bool:
    """Case-insensitive canonical comparison — the conservative direction.
    On a case-sensitive filesystem two spellings may be genuinely different
    files; treating them as the same file can only over-serialize (never
    corrupt).  Dossier 04: 'a different spelling is not independent'."""
    return a.lower() == b.lower()


def _under(path: str, directory: str) -> bool:
    d = directory.rstrip(os.sep) + os.sep
    return path.lower().startswith(d.lower())


def _stems_overlap(a: Any, b: Any) -> bool:
    """True when files written under stem `a` could collide with files
    written under stem `b`.  Every output basename a row produces is
    `<stem>` or `<stem>_<suffix>` (Engines/Base.py), so a collision is
    possible exactly when one stem is a prefix of the other."""
    sa, sb = str(a).lower(), str(b).lower()
    return sa.startswith(sb) or sb.startswith(sa)


# ----------------------------------------------------------------------------
# Row snapshotting — the serial loop's naming decisions, captured not applied
# ----------------------------------------------------------------------------

_INPUT_FIELDS = ("network_file", "origins_file", "destinations_file",
                 "obstacle_points_file", "observer_points_file")
_REQUIRED_FIELDS = ("network_file", "origins_file", "destinations_file")


def _soft(settings: Any, attr: str, default: Any = None) -> Any:
    """getattr that never raises: a property raising on access is recorded
    by validation as an ordered error (the serial loop fails at that row);
    the DESCRIPTOR just degrades — planning must not raise where the serial
    loop would not have raised yet."""
    try:
        return getattr(settings, attr, default)
    except Exception:
        return default


def _as_text(value: Any) -> str:
    """The serial loop's `or ''` idiom, made plan-time-safe: falsy values
    (None, '', 0, False) are absent; truthy non-strings degrade to '' for
    descriptor purposes — validation has already recorded them as the
    ordered errors the serial loop hits at that row."""
    return value if isinstance(value, str) else ''


def _resolved_input(settings: Any, field_name: str) -> str | None:
    """Canonical absolute path of an input file, mirroring how the serial
    code resolves layer paths (data_folder-relative; os.path.join passes
    absolute values through unchanged)."""
    raw = _as_text(_soft(settings, field_name)).strip()
    if not raw:
        return None
    data_folder = _as_text(_soft(settings, "data_folder"))
    return _canonical(os.path.join(data_folder, raw))


def _validate_row(settings: Any, script_output_folder: str | None) -> RowValidation:
    """Ordered per-row validation with the serial loop's exact semantics:
    falsy values are missing (the serial `or ''` idiom); truthy non-strings
    and unreadable attributes are ERRORS the serial loop hits at this
    row's position — held on the plan, never raised across rows (dossier
    04 step 3).  Detector order matches the serial transition order:
    resolve_output_folder runs before the skip gate, loaders after it."""

    def _err(detail: str) -> RowValidation:
        return RowValidation(status="error", missing=(), detail=detail)

    # -- transition 2 (runs for EVERY row, skipped ones included) ---------
    row_value = _as_text(_soft(settings, "output_folder"))
    if not row_value.strip():
        if not (script_output_folder or '').strip():
            data_folder = _soft(settings, "data_folder")
            if data_folder is not None and not isinstance(data_folder, str):
                return _err(
                    f"data_folder is {type(data_folder).__name__}, not a "
                    f"string; the serial loop fails at this row's "
                    f"output-folder default resolution")

    # -- transition 3: the serial `or ''` idiom, fields in order ----------
    missing = []
    for f in _REQUIRED_FIELDS:
        try:
            raw = getattr(settings, f, None)
        except Exception as exc:
            return _err(f"required field '{f}' is unreadable ({exc}); the "
                        f"serial loop fails at this row")
        if not raw:                     # None/''/0/False: serially missing
            missing.append(f)
        elif not isinstance(raw, str):
            # serial: (value or '').strip() -> AttributeError, in-row
            return _err(f"required field '{f}' is {type(raw).__name__}, "
                        f"not a string; the serial loop fails at this row")
        elif not raw.strip():
            missing.append(f)
    if missing:
        return RowValidation(status="skipped", missing=tuple(missing),
                             detail=f"missing required fields: {list(missing)}")

    # -- transitions 5+ (loaders join data_folder directly) ---------------
    data_folder = _soft(settings, "data_folder")
    if data_folder is not None and not isinstance(data_folder, str):
        return _err(f"data_folder is {type(data_folder).__name__}, not a "
                    f"string; the serial loop fails when this row's "
                    f"loaders resolve its input files")

    return RowValidation(status="ok", missing=(), detail="")


def _decide_output_folder(settings: Any, script_output_folder: str | None) -> tuple:
    """The serial loop's decision (UNA.RunBatch): row value > script value >
    <data_folder>/Results.  Returns (decided_value, was_defaulted).  Never
    raises: values the serial loop would fail on are recorded as ordered
    errors by _validate_row; the descriptor degrades to '' — the runtime's
    serial execution still reproduces the serial failure exactly, because
    it runs the serial code path."""
    row_value = _as_text(_soft(settings, "output_folder"))
    if row_value.strip():
        return row_value, False
    base = (script_output_folder or '').strip()
    if base:
        return base, True
    data_folder = _as_text(_soft(settings, "data_folder"))
    return os.path.join(data_folder, "Results"), True


def _snapshot_row(index: int, settings: Any, script_output_folder: str | None,
                  alias_map: dict, will_run: bool) -> RowDescriptor:
    folder, defaulted = _decide_output_folder(settings, script_output_folder)
    stem = _soft(settings, "output_file_name")
    substituted = False
    if will_run and stem == "Results":
        # the serial loop substitutes settings.name at this boundary — but
        # only for rows that pass the skip gate (skipped rows stop at
        # validate_or_skip and keep their raw naming)
        stem = _soft(settings, "name")
        substituted = True
    if will_run:
        inputs = tuple(p for p in (_resolved_input(settings, f)
                                   for f in _INPUT_FIELDS) if p is not None)
    else:
        inputs = ()     # no transitions past the gate: the row reads nothing
    wstamp = bool(_soft(settings, "output_wStamp", False))
    return RowDescriptor(
        index=index,
        name=_soft(settings, "name"),
        semantic_profile=getattr(getattr(settings, "execution", None),
                                 "semantic_profile", None),
        output_folder=folder,
        output_folder_defaulted=defaulted,
        output_stem=stem,
        output_name_substituted=substituted,
        input_paths=inputs,
        reads_timestamped_output=wstamp,
        exact_settings_type=(type(settings) is Settings),
        settings_type_name=f"{type(settings).__module__}."
                           f"{type(settings).__qualname__}",
        aliased_with=tuple(sorted(j for j in alias_map.get(index, ())
                                  if j != index)),
    )


# ----------------------------------------------------------------------------
# The planner
# ----------------------------------------------------------------------------

def plan_batch(projects: Sequence[Any], *,
               script_output_folder: str | None = None,
               analysis: str = "") -> BatchPlan:
    """Plan a batch in caller order without running anything.

    `projects` is the batch's settings list (self.projects — the pairing
    rows or saved project).  `script_output_folder` is the instance-level
    fallback the serial loop uses (UNA.RunBatch reads it from self.settings
    before the loop).  Call-level admission (execution options, parallel
    flag) is the caller's contract-§6 duty and happens before planning;
    this function only plans rows.

    Raises only for whole-call preconditions the serial call also raises
    for (an empty batch) — never for a per-row problem: per-row validation
    outcomes are recorded, in order, on the plan.
    """
    if not projects:
        raise RuntimeError(
            "No settings to run — provide a pairing_file or call "
            "SaveSettingsToProject() first to populate the batch list.")

    # ---- alias detection BEFORE any snapshotting (dossier 04 step 1) ----
    alias_map: dict[int, set] = {}
    seen: dict[int, set] = {}     # id(settings) -> {row indices}
    for i, s in enumerate(projects):
        seen.setdefault(id(s), set()).add(i)
    for members in seen.values():
        if len(members) > 1:
            for i in members:
                alias_map[i] = set(members)

    # ---- ordered per-row validation FIRST (dossier 04 step 3) -----------
    validations = [_validate_row(s, script_output_folder) for s in projects]
    descriptors = [_snapshot_row(i, s, script_output_folder, alias_map,
                                 v.status == "ok")
                   for i, (s, v) in enumerate(zip(projects, validations))]

    # ---- dependency hazards (dossier 04 step 2) -------------------------
    # An executing row's writes land under its output folder with basenames
    # derived from its stem (<stem>, <stem>_<suffix>; a timestamped
    # subfolder only adds a directory level, never changes basenames).
    # Any input file of another row landing in that (folder, stem-prefix)
    # space is a read-write hazard; overlapping stems in one folder are a
    # write-write hazard; the same Settings object twice is an object
    # hazard.  Skipped rows execute no transitions and are hazard sources
    # for nothing — but they keep their recorded outcome and order.
    hazards: list[tuple] = [[] for _ in descriptors]
    for i, di in enumerate(descriptors):
        if validations[i].status != "ok":
            continue
        for j in range(i + 1, len(descriptors)):
            if validations[j].status != "ok":
                continue
            dj = descriptors[j]
            if di.aliased_with and j in di.aliased_with:
                h = Hazard(with_row=j, kind="aliased_settings",
                           detail="rows share one Settings object; the "
                                  "serial loop's second row observes the "
                                  "first row's mutations")
                hazards[i].append(h)
                hazards[j].append(Hazard(with_row=i, kind=h.kind,
                                         detail=h.detail))
                continue
            fi = _canonical(di.output_folder)
            fj = _canonical(dj.output_folder)
            if _same_path(fi, fj) and _stems_overlap(di.output_stem,
                                                     dj.output_stem):
                h = Hazard(with_row=j, kind="output_conflict",
                           detail=f"both rows write stem "
                                  f"'{di.output_stem}' / '{dj.output_stem}' "
                                  f"under the same output folder")
                hazards[i].append(h)
                hazards[j].append(Hazard(with_row=i, kind=h.kind,
                                         detail=h.detail))
                continue
            for reader, writer, da, db, fb in (
                    (i, j, di, dj, fj), (j, i, dj, di, fi)):
                hit = None
                for p in da.input_paths:
                    if str(fb) and _under(p, fb) and \
                            os.path.basename(p).lower().startswith(
                                str(db.output_stem).lower()):
                        hit = p
                        break
                if hit is not None:
                    h = Hazard(with_row=writer, kind="read_write",
                               detail=f"row {reader} reads {hit}, which row "
                                      f"{writer} writes (stem "
                                      f"'{db.output_stem}' under its output "
                                      f"folder)")
                    hazards[i].append(Hazard(with_row=j, kind=h.kind,
                                             detail=h.detail))
                    hazards[j].append(Hazard(with_row=i, kind=h.kind,
                                             detail=h.detail))
                    break

    # ---- admission (dossier 04 step 4) ----------------------------------
    # A stem containing a path separator writes OUTSIDE the modeled folder
    # ("../x" escapes it, "sub/x" descends beneath it): the write-set model
    # cannot contain such a row, so no other row is provably independent of
    # it and every executing row runs serial, with the reason (dossier 04
    # step 4: unsupported behavior executes serially, never ignored).
    def _stem_escapes(stem: Any) -> bool:
        s = str(stem)
        return os.sep in s or bool(os.altsep and os.altsep in s)

    escaping = sorted(d.index for d, v in zip(descriptors, validations)
                      if v.status == "ok" and _stem_escapes(d.output_stem))

    notes = []
    plans = []
    for d, v, hz in zip(descriptors, validations, hazards):
        if v.status == "skipped":
            plans.append(RowPlan(row=d, validation=v,
                                 hazards=tuple(hz), admission="skipped",
                                 admission_reasons=(v.detail,)))
            continue
        if v.status == "error":
            plans.append(RowPlan(row=d, validation=v,
                                 hazards=tuple(hz), admission="error",
                                 admission_reasons=(v.detail,)))
            continue
        reasons = []
        if not d.exact_settings_type:
            reasons.append(
                f"row settings are not the base Settings type "
                f"({d.settings_type_name}); unshared behavior/state cannot "
                f"be proven row-local, so the row runs on the serial route")
        if hz:
            for h in hz:
                reasons.append(f"{h.kind} with row {h.with_row}: {h.detail}")
        for k in escaping:
            reasons.append(
                f"row {k}'s output stem contains a path separator; its "
                f"writes are not contained by any modeled output folder, "
                f"so no row is provably independent of it")
        plans.append(RowPlan(row=d, validation=v, hazards=tuple(hz),
                             admission="worker" if not reasons else "serial",
                             admission_reasons=tuple(reasons)))

    if any(p.row.reads_timestamped_output for p in plans
           if p.admission == "worker"):
        notes.append(
            "worker rows write into a timestamped subfolder "
            "(settings.output_wStamp); timestamps are resolved at run time "
            "under the runtime's clock discipline, never predicted by the "
            "plan — file-level independence holds because output basenames "
            "derive from the row stem, not the clock")
    if any(p.admission == "serial" for p in plans):
        notes.append("at least one row is not proven independent and runs "
                     "on the serial route; proven-independent rows remain "
                     "admitted to workers")
    if any(_soft(s, "batch_composite_output") for s in projects):
        notes.append(
            "batch composite artifacts are written by the coordinator "
            "AFTER the last commit (the 'finalize_composite' batch "
            "transition), outside the per-row write sets; the capture/fold "
            "order is plan.fold_order")

    # The serial loop aborts at the earliest row-position failure: nothing
    # at or after the first "error" row commits (dossier 04 failure
    # ordering — finish the valid serial prefix, expose the earliest
    # ordered failure).
    commit = []
    for p in plans:
        if p.validation.status == "error":
            break
        if p.validation.status == "ok":
            commit.append(p.row.index)
    commit_order = tuple(commit)
    return BatchPlan(analysis=analysis, rows=tuple(plans),
                     commit_order=commit_order, fold_order=commit_order,
                     notes=tuple(notes))


# ----------------------------------------------------------------------------
# The observable-prefix invariant, executable form
# ----------------------------------------------------------------------------

def verify_prefix(committed: Sequence[Any], serial: Sequence[Any]) -> None:
    """Assert the formal dossier-04 invariant: after every committed row i,
    the coordinator's observable state equals the serial reference S_i.

    `committed[i]` / `serial[i]` are opaque observable-state snapshots
    after i+1 committed rows (anything equality- comparable — typically a
    frozen mapping of the public fields and published artifacts).  Raises
    `InvariantViolation` at the earliest differing row; returns None when
    every prefix matches.  Call this after EVERY commit, not just at batch
    completion (dossier 04: 'Check after EVERY committed row').
    """
    n = max(len(committed), len(serial))
    for i in range(n):
        if i >= len(committed) or i >= len(serial):
            raise InvariantViolation(
                row=i,
                detail=f"prefix length mismatch: coordinator has "
                       f"{len(committed)} committed rows, serial reference "
                       f"{len(serial)}")
        if committed[i] != serial[i]:
            raise InvariantViolation(
                row=i,
                detail=f"committed state {committed[i]!r} != serial state "
                       f"{serial[i]!r}")
