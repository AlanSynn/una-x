// Runtime ISA detection for the NATIVE_CORE baseline binary.
//
// Dossier 08: build baseline (no -march=native); ISA-specific variants
// may be dispatched only after validation.  NATIVE_CORE v1 ships the
// validated baseline ONLY — the detection result is recorded (receipt,
// capability registry entry) and variants are deferred to NATIVE_SIMD.
// No ISA-conditional code path exists in this build.
#ifndef UNA_NATIVE_ISA_HPP
#define UNA_NATIVE_ISA_HPP

#include <string>
#include <utility>
#include <vector>

namespace una_native {

// Feature name -> present, probed at runtime (never compile-time).
std::vector<std::pair<std::string, bool>> isa_features();

// The HOST's baseline level: "x86-64-v1".."x86-64-v4" on x86-64,
// "unknown" elsewhere (what this CPU supports, probed at runtime).
// Reported, never acted on, in v1.
std::string isa_baseline_level();

// What THIS BINARY requires: build.sh compiles the generic x86-64-v1
// baseline and refuses -march=... outright.  Capability claims consume
// this constant (a host that reports a higher level must not inflate
// the backend's stated requirement).  NATIVE_SIMD revisits it together
// with its own dispatch validation.
#define ISA_COMPILED_BASELINE "x86-64-v1"

// Deterministic build identity: compiler version + the exact flag set
// from build.sh (no timestamps, so the string is stable per build).
std::string build_fingerprint();

}  // namespace una_native

#endif  // UNA_NATIVE_ISA_HPP
