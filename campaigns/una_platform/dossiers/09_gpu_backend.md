# D09: Actual exact GPU backend and residency region

## Required scope
Implement at least one substantive GPU execution region and validate it on actual hardware.
Device discovery, empty kernels, CPU calls wrapped in a GPU class, or simulator tests alone do
not count. A missing suitable device records blocked qualification while CPU/API work continues.
No paid cloud provisioning or remote credential changes without explicit authorization.

Primary route: CUDA C++ with explicit binary64 operations on an available capable NVIDIA device.
A portable OpenCL/SYCL or Metal route may be chosen when the actual target hardware warrants it,
but admission must verify required dtype/math support. Never assume an Apple GPU supports the
required FP64 arithmetic or silently convert to FP32. Capability matrix records supported
profiles/stages/types and refusals. One measured functioning device path is required; blanket
all-vendor support is not claimed. Hardware missing is not grounds to replace implementation
with stubs, and untested hardware cannot be declared supported.

## First residency candidate
Choose from the measured dominant region: bounded independent-origin searches with subsequent
ordered destination evaluation, or ordered per-OD/stripe flow processing using already prepared
sparse gradients. Upload immutable CSR/source descriptors once for a batch of independent jobs
when identities match. Keep scratch/results resident across adjacent stages when exactness and
ownership permit. Host prepares numerically sensitive scalar/array coefficients via the trusted
implementation and transfers exact bytes until device math is separately qualified.

## Concrete initial schedule
- One logical owner per search or reduction. Retain original heap comparator, adjacency visit
  order, row snapshot and update order for legacy; corrected rule for corrected_v1.
- Tile B independent owners. Each owns bounded queue/label/touched storage. Derive B from actual
  static+state+workspace+runtime+queue device bytes with safety margin, not just element count.
- Sequential recurrence within owner; parallelize owners. No FP atomics into shared edge flow,
  unordered warp sums or cross-origin tree reduction. Fixed logical stripe outputs fold in the
  reference order. A slow exact first kernel is preferable to a falsely compatible fast kernel.
- Queue/scratch overflow sets a per-owner status before public/device accumulation is committed.
  Retry with a larger admitted arena or rerun the entire uncommitted owner on the reference,
  explicitly reported. Never combine partial GPU contributions with a full CPU retry.
- Download only required results/checkpoint state. Synchronize and check errors before success.

## Numerical gate
Explicit RN primitives, no uncontrolled contraction, no flush-to-zero change, no approximate
reciprocal/transcendental substitutions. Basic IEEE primitives do not guarantee whole-expression
parity if compiler or operation order differs. Test boundaries, ties, subnormals, signed zero,
NaN/Inf refusal and NaN payload copying where observed. Use bitwise reference comparison and
compiled kernel inspection. If exact sensitive math remains CPU, charge that path and transfers.
An exact copying/reindexing-only kernel does not satisfy substantive GPU support.

## Transfer/concurrency gate
Record host preparation, layout packing, H2D/D2H bytes/time, allocations, cold JIT/build, launch,
warm compute, synchronization, materialization, remaining CPU and whole public-call wall.
Device buffers have explicit owner/generation; double buffering may overlap only immutable
input fill/transfer with independent device compute. Cancellation drains owned streams and
prevents publication. Device loss quarantines context and yields explicit failure/retry policy.

## Acceptance
Against strongest same-policy CPU, test small and large cells, corrected and admitted legacy
profiles, real cities, batch concurrency, device capacity, cold and warm states. Forced GPU
request must fail if unavailable instead of claiming fallback is GPU. Auto uses only qualified
crossover domains. A real correct GPU backend slower in one domain stays explicit, not automatic.
