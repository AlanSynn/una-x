#include "isa.hpp"

#if defined(__x86_64__) || defined(__i386__)
#define UNA_NATIVE_X86 1
#endif

namespace una_native {

std::vector<std::pair<std::string, bool>> isa_features() {
    std::vector<std::pair<std::string, bool>> out;
#ifdef UNA_NATIVE_X86
    out.emplace_back("sse2", static_cast<bool>(__builtin_cpu_supports("sse2")));
    out.emplace_back("sse3", static_cast<bool>(__builtin_cpu_supports("sse3")));
    out.emplace_back("ssse3", static_cast<bool>(__builtin_cpu_supports("ssse3")));
    out.emplace_back("sse4.1", static_cast<bool>(__builtin_cpu_supports("sse4.1")));
    out.emplace_back("sse4.2", static_cast<bool>(__builtin_cpu_supports("sse4.2")));
    out.emplace_back("popcnt", static_cast<bool>(__builtin_cpu_supports("popcnt")));
    out.emplace_back("avx", static_cast<bool>(__builtin_cpu_supports("avx")));
    out.emplace_back("avx2", static_cast<bool>(__builtin_cpu_supports("avx2")));
    out.emplace_back("fma", static_cast<bool>(__builtin_cpu_supports("fma")));
    out.emplace_back("bmi2", static_cast<bool>(__builtin_cpu_supports("bmi2")));
    out.emplace_back("avx512f", static_cast<bool>(__builtin_cpu_supports("avx512f")));
#else
    out.emplace_back("baseline", true);
#endif
    return out;
}

std::string isa_baseline_level() {
#ifdef UNA_NATIVE_X86
    const bool v4 = __builtin_cpu_supports("avx512f");
    const bool v3 = __builtin_cpu_supports("avx2") &&
                    __builtin_cpu_supports("bmi2") &&
                    __builtin_cpu_supports("fma");
    const bool v2 = __builtin_cpu_supports("sse4.2") &&
                    __builtin_cpu_supports("popcnt");
    if (v4) return "x86-64-v4";
    if (v3) return "x86-64-v3";
    if (v2) return "x86-64-v2";
    return "x86-64-v1";
#else
    return "unknown";
#endif
}

std::string build_fingerprint() {
    // Must stay in sync with build.sh (the receipt cross-checks both).
    return "g++ " __VERSION__ " | -O2 -std=c++17 -shared -fPIC "
           "-fvisibility=hidden -ffp-contract=off -fno-fast-math | "
           "baseline-dispatch (ISA variants deferred to NATIVE_SIMD)";
}

}  // namespace una_native
