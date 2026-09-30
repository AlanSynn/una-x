# D11: Installed packaging, compatibility shim and required CI

Keep urban-network-analysis distribution and urban_network_analysis imports. Add native/GPU
extensions via reviewed optional build artifacts/extras; the default CPU-only install must
work without compiler or GPU runtime. Declare required runtime dependencies accurately, including
Arrow usage or an explicit reader fallback. Optional visualization imports are lazy, but actual
Madina create_map functionality and pydeck return must work when its documented extra is installed.

A separate drop-in shim distribution may own `madina` only in a clean environment without the
upstream distribution. Document conflict detection and installation migration. Do not silently
overwrite another package's files, alias arbitrary modules globally, or depend on upstream Madina
as the implementation. Retain license headers for any ported code and third-party library licenses.

Native build: nominated C++17+pybind11, checked headers, fixed-width ABI, appropriate platform tags,
ISA baseline plus capability dispatch, no host-specific universal wheel. GPU artifact records
architecture/runtime compatibility and actual binary/kernel identity. No invisible runtime
binary download. Installed wheel tests run outside all source trees with PYTHONPATH removed;
assert every module/binary location and hash, including the compatibility shim.

CI introduction is required in this campaign because no existing workflow previously qualified
the complete repository. Fast default matrix: supported Python versions and current pinned GIS
deps on Linux/macOS/Windows CPU; pure Python, native build and artifact tests; targeted modern/
legacy dependency bridge tests. Device jobs run on explicitly authorized existing hardware,
not simulated as success. Sanitizers and stress/recovery suites are bounded scheduled/manual
jobs near integration. Do not spend an unbounded hosted matrix after every experiment.

Test-frame hardening: remove developer absolute paths, make output roots explicit and temporary,
forbid test writes into committed historical evidence. Tests cannot load candidate source from
a different primary worktree. Convert stale source-shape assertions to behavior/identity tests
where justified, never delete a failing correctness assertion to get green.

Acceptance: wheel and sdist builds, isolated install smoke, full public API parity on installed
package, source install CPU fallback, substantive native/GPU engagement, missing dependency
errors, read-only baseline, no evidence rewrite, no source shadowing, exact installed parity,
full required CI with skips/unavailable clearly reported. No release/publishing authorization.
