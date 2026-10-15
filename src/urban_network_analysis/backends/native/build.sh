#!/usr/bin/env bash
# Build the NATIVE_CORE baseline binary (dossier 08).
#
# Contract (kernels/SPEC.md + scope_core.hpp header comment):
#   - strict IEEE: -ffp-contract=off -fno-fast-math.  The characterized
#     fast-math comparison laws are reproduced IN SOURCE (scope_core.hpp
#     fast_lt/fast_le); the compiler must not canonicalize or contract
#     anything, or no bitwise pin can hold.
#   - baseline ISA: NO -march=native and no ISA-selection flags.  The
#     binary dispatches the baseline everywhere; ISA variants are
#     NATIVE_SIMD scope, after their own validation.
#   - hidden visibility except the pybind module export.
#
# Env:
#   PYTHON   interpreter to build for (default: `python` on PATH)
#   CXX      compiler (default: `g++`; needs C++17)
#   ASAN=1   AddressSanitizer + UBSan build (same ABI, suffix unchanged;
#            run tests with LD_PRELOAD=$(gcc -print-file-name=libasan.so)
#            and UBSAN_OPTIONS=halt_on_error=1)
#
# The output lands next to this script (the package the loader imports
# from), named _native_core$(EXT_SUFFIX) for the invoking interpreter.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${HERE}/src"
PY="${PYTHON:-python}"
CXX="${CXX:-g++}"

EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"
case "${EXTRA_CXXFLAGS}" in
    *-ffast-math*|*-fassociative-math*|*-march=*|*-ffp-contract=fast*)
        echo "refusing: EXTRA_CXXFLAGS would break the bitwise contract" >&2
        exit 2;;
esac

PY_INCLUDE="$("${PY}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
EXT_SUFFIX="$("${PY}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PYBIND11_INCLUDE="$("${PY}" -c 'import pybind11; print(pybind11.get_include())')"

CXXFLAGS=(
    -O2 -std=c++17 -shared -fPIC -fvisibility=hidden
    -ffp-contract=off -fno-fast-math
    -fvisibility-inlines-hidden
    -Wall -Wextra
    -I"${SRC_DIR}" -I"${PYBIND11_INCLUDE}" -I"${PY_INCLUDE}"
)

LDLIBS=()
if [[ "${ASAN:-0}" == "1" ]]; then
    CXXFLAGS+=(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    LDLIBS+=(-fsanitize=address,undefined)
fi

OUT="${HERE}/_native_core${EXT_SUFFIX}"

# shellcheck disable=SC2086
"${CXX}" "${CXXFLAGS[@]}" ${EXTRA_CXXFLAGS} \
    "${SRC_DIR}/native_core.cpp" \
    "${SRC_DIR}/scope_core.cpp" \
    "${SRC_DIR}/isa.cpp" \
    ${LDLIBS[@]+"${LDLIBS[@]}"} \
    -o "${OUT}"

echo "built ${OUT}"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${OUT}"
fi
