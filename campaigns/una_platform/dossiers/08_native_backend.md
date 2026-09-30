# D08: Actual ahead-of-time native CPU backend

## Required implementation
A C++17 native core with a narrow pybind11 binding is the nominated first backend. A different
native toolchain requires a recorded cost/portability reason, not preference. It must perform
substantive graph/search or ordered flow/accessibility computation, not copy an input array and
call the reference. Keep a portable trusted CPU implementation. Optional native wheels provide
ordinary users binaries; source users may choose to build, but CPU-only installation stays usable.

## Internal ABI contract
Typed buffers carry element type, shape, strides/contiguity, byte order, alignment, mutability,
owner lifetime and semantic profile. Validate before releasing the GIL. Use fixed int32/int64
widths, not platform C long. Binding must not silently cast/reorder unsupported inputs. Either
explicitly charge an admitted exact conversion or refuse into a documented selector fallback.
Check products/offset sums for overflow before allocation. No out-of-bounds or undefined signed
integer overflow. Retain Python owner references while native threads read arrays.

## Algorithm schedule
Port the selected CPU algorithm, not an unrelated shortest-path library. Legacy heap tuples,
row snapshot updates, tie ordering, sentinels, duplicate edges and final reductions must match
the characterized compiler reference. corrected_v1 follows its explicit RN/state rules. Vectorize
independent searches/entities, never an ordered inner reduction. For divergence, record actual
lane efficiency or mark unavailable. Avoid full BxV workspaces unless charged and beneficial.

## Floating point
No global -ffast-math. Disable uncontrolled contraction/reassociation in corrected_v1; explicit
fused operations only where reference specifies them. Retain CPU trusted math-library values
for sensitive coefficients initially. Legacy Numba fastmath may already reorder operations;
turning fastmath off does not prove legacy parity. Reproduce the relevant operation sequence,
use a narrower admitted region, or keep that region on the reference. Never fix a bit mismatch
with tolerance. Preserve signed zero/subnormal/nonfinite rules on admitted values.

## Binding, parallelism and errors
Release GIL only around pure native work. No Python callback/object access inside released
regions. Shared input immutable, per-logical-stripe scratch/output private, coordinator folds
in specified order. Translate structured error codes after reacquiring GIL. Cancellation checks
at bounded safe boundaries never leave partial public writes. Detect CPU ISA at runtime; build
baseline wheels without -march=native, then dispatch validated ISA-specific variants explicitly.

## Gates
ASan/UBSan in developer CI, bounds/fuzz tests for invalid strides/shapes/indices and allocator
failure, bitwise reference tests, owner-lifetime/GC tests, cancellation/race tests, fixed-stripe
cross-thread exactness, cold import and package failure fallback. Engagement test temporarily
replaces native kernel with a fail sentinel and must fail when native is requested. Record
binary hash, compiler flags, CPU ISA, disassembly, build time and complete end-to-end timings.
A slower but correct substantive native route may remain explicit; auto requires measured win.
