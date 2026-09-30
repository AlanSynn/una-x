# Authority and scope

## Binding objective
At least the functionality and API behavior actually supported by pinned Madina must be
available through UNA-X, while existing UNA APIs remain callable. Deliver native CPU,
GPU, public parallel RunBatch, corrected scientific/numerical behavior, caches and robust
failure handling. Maximize validated completed work per end-to-end second within a fixed
CPU/RAM/device/I/O budget. Do not exchange accuracy or scientific work for speed.

## Interpretation decisions
1. Bitparty means bitwise parity, not merely statistically similar results.
2. Bug fixes intentionally alter identified wrong results. Record the delta and test against
   a corrected independent reference; historical bitwise reproduction is a separate profile.
3. “API parity” includes signatures, defaults, inputs, return types, layer/object mutation,
   IDs/indexes, geometry, diagnostics, errors and exports. A similarly named metric is not parity.
4. GPU and native are required implementations. Qualification and automatic use are separate.
   Lack of hardware is a recorded blocker, not a license to implement only detection/stubs.
5. Old campaign restrictions on caches, GPU/native, public batch, and bug fixes no longer apply.
6. Existing UNA default calls remain una_legacy for this campaign. Madina compatibility calls
   default to madina_legacy. corrected_v1 is publicly selectable and fully implemented.
   A release-wide default migration is outside this packet publication; document an explicit
   migration path rather than making a hidden default change. Unsafe nontermination is not
   a useful legacy contract: validated rejection/cancellation is a declared safety correction.

## Authorized changes
Add public keyword-only execution controls, profile/version metadata, compatibility APIs,
optional dependency distributions, native/GPU code, build/test CI, caches, checkpointing,
input diagnostics, validated bug fixes and exact algorithmic/layout changes. Maintain the
current src baseline as an independent oracle. Do not rewrite it to generate new goldens.

## Non-negotiable exclusions
No undocumented approximations, mixed precision, altered detour/radius/domain, truncation of
route sets, silent dropped origins/components, invented city incidents, test-only execution
presented as installed support, or unsupported accelerator use. Do not rebuild the entire
GIS stack to chase a backend name. Do not claim a global optimum from finite experiments.

## Delegation
Workers may choose local names, buffer layouts and bounded algorithms inside the dossier
proofs. The integrator alone edits shared API/default/schema/dispatch files. A worker must
not mark a task complete because a code path exists: produce its required evidence.
Escalate only a material semantic ambiguity with competing scientifically valid definitions,
unavailable external resources, or a needed expansion of publication/resource authority.
Unrelated tasks continue when one task is blocked.

## Priority and termination
Correct runnable references and failure reproduction precede speed. Implement the mandatory
feature floor, then profile and iterate over the measured frontier. A 10% gain is not an
automatic stop. At most three algorithm/schedule variants per measured bottleneck per round,
then reprofile. Stop a round when no admitted candidate improves the frontier and the maximum
remaining removable service cannot repay its validation/maintenance cost. Record rejected
experiments. DECISION.md defines truthful partial and completed outcomes.
