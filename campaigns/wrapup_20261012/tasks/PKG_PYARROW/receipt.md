# PKG_PYARROW receipt — wheel METADATA pyarrow under-declaration (F1 fix)

- Task: wrapup 2026-10-12, finding F1 (install-wheel leg iwheel-20261011T1756Z).
- Branch: `perf/pkg-pyarrow-dep` @ **8ef3588** (parent c1f8614).
- Run stamp: pkw-20261012T0426Z. Raw evidence (untracked):
  `/storage/scratch1/1/dsynn6/una-x/evidence/tasks/PKG_PYARROW/pkw-20261012T0426Z/`.
- Worktree: `/storage/scratch1/1/dsynn6/una-x-pkgarrow-20261012` (HEAD==8ef3588 at close).
- Interpreters: alan env python 3.11.4 for build + venv creation; venvs isolated
  (fresh `pip list` showed only pip/setuptools before install); PYTHONHASHSEED=0.

## Change (1 file, +4 lines)

`pyproject.toml` `project.dependencies`: added plain unconstrained `pyarrow`
with a comment stating why (pyogrio `use_arrow=True` needs it; pyogrio only
optionally depends on it).

Constraint choice: repo convention in `pyproject.toml` is unconstrained plain
names for every dependency (numpy, scipy, pandas, geopandas, shapely, ...);
`pyogrio` itself arrives unconstrained via geopandas. pyarrow is imported
directly by shipped code (`cache/identity.py` `geometry_env()` and the
`Topology._get_gdf` read path `gpd.read_file(engine='pyogrio', use_arrow=True)`,
`Topology.py:1498`), so a direct declaration is required, and convention says
no pin. No extra pins added.

`setup/una.yml` already declared `pyarrow` (line 9) — no change needed. No
requirements.txt/lockfiles exist in the repo. Historical
`campaigns/history/una_cpu_2026_09/packet/evidence/installed/requires_dist.txt`
is an immutable record, left untouched.

## Negative control (pre-fix, commit c1f8614 bytes)

Pre-fix wheel: the iwheel leg's wheel, sha256
`6f35647cb2cb1a9983fbb7d992da735e36dbbb0346d174338c967458c40654d9`
(recomputed this leg; matches its recorded `wheel_sha256.txt`, built from the
iwheel worktree at c1f8614, `logs/build_wheel_worktree.log`).

- METADATA: 9 `Requires-Dist` entries, `pyarrow` nowhere in the file
  (`metadata/PREFIX_METADATA.txt`).
- Fresh venv `venv-prefix`, resolution-only `pip install <wheel>`: install
  exit=0 with NO pyarrow collected (`logs/install_wheel_prefix.log`, 0
  "Collecting pyarrow" lines) — identical to the iwheel F1 record.
- Public-API smoke on the same fixture: FAILS exactly as found —
  `RunAccessibility` -> `Topology.py:1498 _get_gdf` ->
  `pyogrio/raw.py:266 read_arrow` -> `RuntimeError: pyarrow required to read
  using 'read_arrow'` (`logs/compute_prefix_negative.log`, exit=1).

## Fix verification (post-fix, commit 8ef3588 bytes)

Wheel rebuilt with the iwheel-established path (`pip wheel --no-deps` from the
worktree, alan python; `logs/build_wheel_postfix.log`):
`dist_postfix/urban_network_analysis-2.6.0-py3-none-any.whl` sha256
`c3f106d9682071d9f0caf656fa642fa40353b0db185b1f258d9feeb7d648c417`
(pip's own log line agrees).

- METADATA diff vs pre-fix is the single line `+Requires-Dist: pyarrow`
  (`metadata/METADATA.diff`; both sides extracted from the wheel zips).
- Fresh venv `venv-postfix`, resolution-only install: resolver collects
  `pyarrow (from urban-network-analysis==2.6.0)` and installs
  `pyarrow-26.0.0` (`logs/install_wheel_postfix.log`). `pip check` clean.
- Census: pyarrow 26.0.0 importable, `importlib.metadata.requires` reports
  `pyarrow`, module census **60/60 OK**, CENSUS_PASS
  (`logs/census_postfix.log`).
- Smoke: RunAccessibility end-to-end on the iwheel fixture
  (network/origins/destinations.geojson, sha256-identical copies), neutral
  cwd, COMPUTE_PASS (`logs/compute_postfix.log`).

## Output cross-check vs iwheel leg (informational)

`Results.geojson` bitwise identical to the iwheel warm leg
(`e4c7238c5e6609df4de1946331486533558eb36eefb8f57dbd48d5ec8f86b82b`).
`Results.feather` bytes differ (34f9bc98... vs 81f6d9f4...): feather
containers embed writer metadata and this leg resolved pyarrow 26.0.0 where
the iwheel leg had 25.0.1. Code bytes are unchanged by this fix; this is
container-metadata drift across a pyarrow minor, not a result change. The
published .geojson values match exactly.

## Scope and residuals

- Source change: packaging metadata only. No src/, tests/, or yml changes.
- Resolved pyarrow 26.0.0 here vs 25.0.1 verified in dev — unconstrained
  declaration resolves to latest and the full public-API smoke passes on it;
  if the project ever wants reproducible fresh installs, that is a separate
  lockfile decision, deliberately not taken here.
- Version string remains 2.6.0 (no bump requested).
- Raw dir also holds both venvs (`dist_venvs/`, ~GBs) for re-verification;
  ephemeral, safe to delete after close.
- No pushes, no merges to main, no releases.
