# Pinned source and evidence audit

Read/verified authoring main: 395cdc5f683894b6f2ba460f7dbcefee99981ba3.
Runtime src tree: f1850b77555349666e4a3e16fc090c5cab088008.
Existing tests tree: 407cdce3521342d3ea4476cca07ccbd36e4c2c65.
Existing benchmarks tree: f721ae48cd6a612f43512ca8c1538c003349c9c5.
Retained platform tree: c1e4f15813ed66bc6e6053e257afd8a666e512ed.
These are Git content identities, not benchmark results. `source_pins.json` is authoritative.

## Source facts at this pin
- AccessibilityWElevation integrated_scope_access uses A3/A1 two-phase searches and then
  adjusts all destination distances per origin. A3 still initializes V float64 labels per origin.
- The complete accessibility result key includes beta, plateau, logistic, KNN and radius;
  a changed metric parameter misses the combined producer. Full-result cache reuse already
  exists. Never compare a new cache against that feature deliberately disabled for exact repeats.
- cache/stages.py encodes arrays to .npy bytes and decodes to fresh arrays. Mutable public
  ownership is protected, but a memory hit is not a proven zero-copy internal representation.
- Public batch/worker.py stages outputs and sends deferred project state. Integrate with that
  contract; do not replace it with an unrelated external toy pool and call public RunBatch fast.
- AggregateFlow already contains local overlap scratch and budgeted gradient chunks. F2/F3
  should not be rediscovered as new work. The proposed flow work targets the dense-output
  search producer and invariant ordered route computation, not the already-local consumer.
- kernels/compacted.py is a characterized comparison arm, not an optimized native production
  route. Its speed-neutral result is preserved. It still visits the global destination domain.
- CPU_ALGORITHMS records legacy compiler-sensitive comparison/reduction behavior. A new
  standalone metric expression is not automatically bit-equal to the combined compiled fold.

## Prior component evidence, not current application evidence
The complete 28-file investigation is retained under prior/una_10x_investigation with its
original manifest. It reports 48.014x for one large-local prepared-array case and 15.777x for
one 32-job distinct-beta component cohort. These are derived ratios from its raw measurements.
It ALSO reports 0.360x tiny and 0.838x search-dominated ratios. No geometry loading, exports,
public state/RunBatch, installed wheel, cold compilation or phase-peak memory was qualified.
Its baseline was a transcription because container DNS prevented checkout retrieval.

Do not copy its constants into generic code: beta=.03, plateau=0, midpoint=10, growth=.1,
logistic KNN and [1,1,.5] are fixture choices. Its logical lanes are prange-owned workspaces,
not guaranteed hardware-thread identities. Its permissive Python assertions are not a public
fallback implementation. Its ge_only result is a local hypothesis about compiler parity.

## New work required before conclusions
S01 freezes a clean checkout/environment. S02 executes the pinned-engine bridge and expands
it to adversarial compiler-sensitive states. H01 runs actual public interfaces. W00 measures
observed networks. I00 installs wheels away from all checkout paths. Q00 uses those installed
implementations. Historical component data cannot fill these gates.

## Evidence quality warning carried forward
Platform and predecessor receipts sometimes point at untracked/deleted raw trees. Retain the
receipts unchanged, inventory which bytes are actually present, and label missing data. A hash
cannot reconstruct deleted measurements. This campaign must keep decisive raw records and
counterexamples available in its final bundle/repository evidence, not only their digests.
