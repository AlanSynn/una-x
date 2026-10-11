# NATIVE_CORE backend (dossier 08)

Ahead-of-time compiled C++17 core for the scope-access kernel family,
bitwise-contracted to the characterized references
(`src/urban_network_analysis/kernels/SPEC.md`, frozen by CPU_ALGORITHMS).
This file records the port decisions and the recorded admissions; the
code is the contract's enforcement, the tests under
`tests/native_core/` are its proof.

## What is native

`src/` compiles to `_native_core.<EXT_SUFFIX>` (pybind11), built by
`build.sh` with `-O2 -std=c++17 -shared -fPIC -fvisibility=hidden
-ffp-contract=off -fno-fast-math` (baseline ISA; `-march=native` and
ISA selection are refused — the characterized laws are reproduced in
source, so the compiler must not canonicalize or contract anything).

- Two-phase scope search (SPEC.md section 3), una_legacy route
  semantics: fast comparison laws spelled as IEEE complements
  (`fast_lt(a,b) = !(b<=a)`, `fast_le(a,b) = !(b<a)`), the NaN-first
  heap tuple law, verbatim CPython heapq push/pop/siftdown
  transcription, the engine's discarded initial pop, unconditional
  start-then-end seed overwrites, bit-exact `1.0 + cutoff` sentinel,
  duplicate-arc row-order snapshot/assignment, flag AND degree>1 push
  gate.  A3 (domain V) and A1 (domain V + d_count) label domains.
- corrected_v1 scope search (`reference/search.py::corrected_scope_labels`):
  +inf labels, minimum seed combine (end replaces on strict improvement),
  closed-radius heap seeding of final seed labels, stale-entry skips,
  no push gate, IEEE `candidate <= cutoff && candidate < label`.
- Destination adjust (SPEC.md section 4 / corrected declared order):
  builtin-min bits `(b < a) ? b : a` (first operand wins on equality
  AND incomparability; the same expression serves both profiles — on
  the corrected validated domain no NaN can occur).
- Runtime ISA detection (`isa.cpp`) reports features and the x86-64
  baseline level.  v1 dispatches the baseline ONLY and acts on nothing;
  ISA variants are NATIVE_SIMD scope (recorded decision, dossier 08
  allows the deferral with reason).

## Region admissions (kept on the reference — recorded, not hidden)

- **Fold (reach/gravity/knn), both profiles.** The una_legacy fold's
  bits include LLVM fastmath reassociation of the compiled
  transcription (1-ulp class, characterized in `kernels/reductions.py`)
  that no declared C++ operation sequence reproduces.  The corrected
  fold retains platform-libm transcendentals via numpy (NUMERICS.md:
  "start by retaining trusted CPU math-sensitive coefficients"; a path
  that cannot match a profile's arithmetic stays outside its admitted
  domain).  `backends/native/_route.py` composes the native search+adjust
  with the SAME canonical fold plans the pinned references use —
  identical plan calls on identical adjust values, so whole-driver
  parity holds by composition and is machine-checked.
- **Portable fallback routes.** The compact route (admission-mirror
  refusal) and every policy-refused input stay on
  `kernels.scope_access.integrated_scope_access_oracle` (legacy) or the
  reference composition (corrected) — the exact pinned artifacts.

## Policy vs enforcement

`_route.py` owns POLICY: the admission mirrors (`scope_route_admits`,
`tail_admits`), layout/dtype canonicality, and the fallback decision
(every delegation records `fallback_reason`).  The core owns
ENFORCEMENT: it re-validates everything it reads (CSR structure,
neighbor range, cost domain, terminal domains, destination read bound)
and reports any disagreement with the policy layer as a loud status
(`STATUS_ERR_ABI`) / `NativeBackendMismatch` — never a silent
re-route mid-kernel (SPEC.md section 7).

Known refusals that delegate (recorded in the capability entry):
non-contiguous admitted arrays (legacy), A1 destination terminals
outside `[0, V+d)` (Python negative-index semantics the pointer-reading
core cannot reproduce), `adjacynct_vector_network_node` length != E
(the guard omits this condition; the portable route's behavior is the
pinned one).  The historical misspelled input name
`adjacynct_vector_network_node` is part of the frozen contract and is
NOT renamed.

## Binding ABI law

Fixed widths only (`int64_t`/`double`/`uint8_t` bools); dtype-agnostic
`py::array` views (unsupported dtype/layout is REFUSED, never cast or
reordered — including byte-order and alignment checks); all validation
before `py::gil_scoped_release`; owner arrays stay alive via the
binding's arguments across the released region; workers touch only raw
pointers and per-worker scratch; structured status codes translated
after GIL reacquire; cancellation (`CancelFlag`, atomic) polled at
origin boundaries; allocation sizes overflow-checked
(`check_domain_size`), `std::bad_alloc` -> `STATUS_ERR_MEMORY`, thread
work partitioned origin `i -> i mod L` with per-worker buffers and
origin-partitioned outputs (worker count cannot move a bit); a cancelled
or failed run publishes nothing (the route assembles public
`KernelOutputs` only on complete runs; incomplete carries
`next_logical_state = origin-<first incomplete>`).

`cutoff` crosses as one int->float64 widening — the ONE charged
conversion, the same one the compiled kernel's float64 parameter
performs.

## Build

`./build.sh` (env: `PYTHON`, `CXX`, `ASAN=1`).  Non-sanitized binary
name is the loader's target; the ASAN build writes the same name —
build it into a scratch copy when a permanent install must stay
un-instrumented.
