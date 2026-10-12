# COMPAT_D3 receipt — D3 env-probe collision on the compat surface (cd3-20261012T0030Z)

Branch `perf/compat-d3-lazy` @ worktree `/storage/scratch1/1/dsynn6/una-x-compatfix-20261012`,
base `de2872f` (perf/compat-una-reexport tip). Interpreter: alan
(`/storage/home/hcoda1/1/dsynn6/micromamba/envs/alan/bin/python`), `PYTHONHASHSEED=0`,
per-directory islands. `.refs` temporarily symlinked from the main checkout
(`/storage/scratch1/1/dsynn6/una-x/.refs`) and removed after; worktree re-verified clean
of tracked changes. Raw evidence (untracked): `evidence/tasks/COMPAT_D3/cd3-20261012T0030Z/`
(`sha256_manifest.txt` inside).

## Goal

Resolve the D3 collision introduced by the R1 eager reexport: `import
urban_network_analysis.compat.madina.zonal` executed the parent `__init__`'s
`from .una import *`, pulling `una.betweenness` whose module-top
`os.environ['USE_PYGEOS'] = '0'` (betweenness.py:105) broke
`tests/platform_geometry/test_errors_and_unsupported.py::test_facade_import_writes_no_environment_variables`
(bisected single-cause to de2872f). Preserve BOTH the 61/62 top-level parity surface
(delta exactly {pdk}) and the D3 platform_geometry contract. Option (a) shipped:
lazy materialization, PEP 562. Option (b) (amending the D3 test) was NOT needed —
the D3 test file is untouched.

## Single cause, mechanically confirmed

`-X importtime` trace at de2872f (`importtime_de2872f_zonal.txt`): the leaf zonal import
executes `urban_network_analysis.compat.madina` (parent `__init__`) → `.una` →
`una.betweenness`; the zonal package's own imports pull no una module. Baseline on this
worktree at de2872f bytes: platform_geometry **36 passed / 1 failed** (exactly the D3
test), zonal island 26 passed.

## Change (option a)

`src/urban_network_analysis/compat/madina/__init__.py` — module-level
`from .una import *` replaced by `_load_una_star()` + PEP 562 `__getattr__`/`__dir__`:

- First una-surface access (`m.una`, any of the 61 reexported names, `from ... import
  <name>`, `import ...compat.madina.una`, or `dir(m)`) imports `.una` exactly once,
  star-copies its public namespace (una names win the zonal star, upstream `__init__`
  order), re-applies the I6 zonal package restore last, serves the name; unknown names
  raise AttributeError as before.
- The pinned una-chain env write (betweenness.py:105, workflows.py:112 — upstream
  parity) fires at that first load, at its upstream site, instead of at compat-package
  import time. This is the one deliberate behavioral delta vs de2872f, documented in the
  module docstring.
- No `__all__` defined (upstream defines none); star-import of compat.madina
  materializes via the `__all__`-lookup → `__getattr__` route, then lands the full dict.
- After the first una load the namespace is byte-identical to the eager de2872f layout.

## Bytes changed (sha256 from tool output)

| file | sha256 |
|---|---|
| src/urban_network_analysis/compat/madina/__init__.py | 224efbfcd0d3863f6019ed9ae8910f2191decf7e240c4ee9f8de29a78a095d67 |
| tests/madina_api/zonal/test_zonal_facade_env.py | f54b08213e612049408fd30e1efdcd1c782dad7c9620f3f9d0ab888bc25fff70 |
| campaigns/una_platform/evidence/api/api_matrix_errata_20261012_D3LAZY.json (new) | (see git; frozen R1 errata 92ebf2ed… and api_matrix.json 875d80e9… untouched, hashes re-verified pre-commit) |

`git diff de2872f..HEAD --stat`: exactly these + the receipt. No D3 test edit, no
frozen-file edit.

## Verification counts

| suite | de2872f baseline (this worktree, .refs on) | after fix |
|---|---|---|
| tests/platform_geometry | 36 passed / 1 failed (the D3 test) | **37 passed / 0 failed** (121.14 s) |
| tests/madina_api/zonal | 26 passed | **28 passed / 0 skipped** (70.94 s; 26 + 2 net new) |
| tests/madina_api/paths | (R1: 19) | 19 passed (61.20 s) |
| tests/madina_api/access | (R1: 24) | 24 passed (94.13 s) |
| tests/madina_api/flow | (R1: 27) | 27 passed (221.54 s) |
| tests/madina_api/workflows | (R1: 35) | 35 passed (117.11 s) |

Machine-check digests (facade.json / reference.json, preserved in the evidence dir):
upstream top 62 vs facade top 61; only-ref `{pdk}`, only-facade `[]`; `una_public`
36/36 equal; identities pinned (m.una = una package both arms; betweenness/paths =
modules both arms; m.zonal package restore = ledgered I6).

Fresh-interpreter probes (`probe_lazy_surface.out`, `probe_import_orders.out`,
`probe_importtrace_postfix.out`, `star_facade.txt` vs `star_upstream.txt`):

- bare `import ...compat.madina` and `import ...compat.madina.zonal`: `USE_PYGEOS`
  unset, no `compat.madina.una` module in sys.modules (same probe script printed
  env='0'/betweenness-loaded at de2872f bytes — before/after pair on identical code).
- after `dir(m)` / `m.una` / star-import: env == '0' (upstream value), dir surface ==
  star-import surface == 61 names, in BOTH import orders (compat-first, una-first);
  `m.una is sys.modules[...]`; zonal package restore holds.
- facade star-import surface == upstream star-import surface minus exactly `{pdk}`
  (62 → 61; `diff` shows only `pdk` removed).

## Test-file changes (compat suite; nothing weakened)

- `test_top_level_import_carries_pinned_una_env_write` →
  `test_top_level_env_write_fires_at_first_una_load`: same pinned write, trigger
  re-pinned to the lazy contract (bare import clean; una-surface access writes '0').
  This is the only existing-test amendment, and it pins the fact R1's test could no
  longer assert (R1's assertion is falsified by design under the lazy trigger).
- Added: `test_d3_lazy_facade_and_zonal_imports_leave_env_pristine`,
  `test_star_import_materializes_the_full_surface`.
- D3 AST pin (`test_d3_zonal_port_has_no_env_write`) and the platform_geometry D3
  test: untouched, green.

## Residual risks / notes for the reviewer

- Pre-load `vars(m)` (direct `__dict__` inspection without attribute access) shows the
  eager-zonal-only dict; PEP 562 is attribute-mediated, so `dir()`, getattr, from-import,
  star-import, and `import` all materialize. Any consumer reading `vars(module)` directly
  pre-load would see a partial dict — inherent to option (a), documented in the
  docstring; no such consumer exists in the repo.
- `dir(m)` triggers the una load (side effect: the pinned env write). Necessary for
  live full-surface reflection without a hardcoded name list; the only env probes in
  the repo are import-shaped, not dir-shaped (both D3 probes verified green).
- Upstream parity statement holds only post-first-load (byte-identical dict); pre-load,
  a consumer touching no una name sees the eager-zonal surface — the documented delta.
- Not run: whole-repo regression (namespace-only change; islands above cover the compat
  surface + the D3 contract; load discipline).
- Upstream itself has no lazy equivalent (`import madina` writes USE_PYGEOS eagerly at
  zonal.py:3); the lazy trigger is a facade delta ledgered in
  `api_matrix_errata_20261012_D3LAZY.json`, superseding the R1 companion's
  `import_side_effect` field.

## Commits

1. `404fba7` — implementation + compat tests + companion errata (evidence-first message).
2. (this commit) — tracked receipt.

Raw logs: `evidence/tasks/COMPAT_D3/cd3-20261012T0030Z/` (untracked, 20 files +
sha256_manifest.txt).
