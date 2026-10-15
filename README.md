# UNA — Urban Network Analysis

UNA is a free, open-source Python package for computing **accessibility**
and **pedestrian-flow** metrics on spatial networks, developed by
Andres Sevtsuk and Raul Kalvo at the
[MIT City Form Lab](https://cityform.mit.edu). It loads a spatial network,
weighted origin and destination point layers, and optionally obstacle or observed point layers, and runs analyses to answer two families of questions:

- **Accessibility** — what can each origin reach on the network within a
  given walking range? Metrics: Reach (cumulative opportunities),
  exponential and logistic gravity, and K-nearest-neighbor access.
- **Flow** — how many trips does each street segment carry between
  origins and destinations? Two route-choice engines: `aggregate_flow`
  (the default — spreads each OD pair's trips over the full envelope of
  viable routes; scales to regional and state-wide models) and
  `k_alternatives` (enumerates discrete alternative paths; supports
  assigned OD routing and route-geometry export. Suitable for
  neighborhood scale analyses). Both engines support turn-aware and
  elevation-aware routing.

Impedance can go beyond geometric length: custom "perceived length"
edge costs, uphill elevation penalties, turn penalties, and obstacle
penalties. Hot loops are Numba-compiled and searches are radius-bounded,
which keeps large-scale runs feasible. Results export to GeoJSON,
Feather, and CSV for mapping in QGIS or any GIS.

**Documentation** and **tutorials** for the UNA Python package are found at: https://urban-network-analysis.readthedocs.io/en/latest/index.html

![Estimated foot-traffic volumes in Bangor, ME](docs/images/maine.jpg)

*Estimated foot-traffic volumes in Bangor, ME. From the [Pedestrian Volume Model for Maine Towns](https://cityform.mit.edu/projects/a-pedestrian-volume-model-for-maine-towns) project.*

## Installation

Two steps — a conda environment for the geospatial dependencies, then
the package itself. Requires Python ≥ 3.11.

```bash
# 1. environment (una.yml is in this repo's setup/ folder)
conda env create -f setup/una.yml
conda activate una

# 2. the package
pip install git+https://github.com/City-Form-Lab/urban_network_analysis.git
```

To modify the code, clone and install in editable mode instead:

```bash
git clone https://github.com/City-Form-Lab/urban_network_analysis.git
pip install -e ./urban_network_analysis
```

## Updating

To update an existing install to the latest version on GitHub:

```bash
conda activate una
pip install --upgrade git+https://github.com/City-Form-Lab/urban_network_analysis.git
```

(Editable installs from a clone update with `git pull` instead.) Verify:

```python
import urban_network_analysis
urban_network_analysis.about()
# UNA — Urban Network Analysis, version 2.6.0 (last public commit: 2026-09-13)
# The urban_network_analysis Python package is developed by the MIT City Form Lab. Website: http://cityform.mit.edu/
```

(`urban_network_analysis.__version__` still returns the bare version
string, e.g. `"2.6.0"`.)

## Quick start

```python
import urban_network_analysis as una

project = una.UNA()

project.settings.data_folder       = "Boston"     # this repo's docs/Boston has tutorial data
project.settings.network_file      = "20260703_PercLenNetwork_InnerCore.geojson"
project.settings.origins_file      = "Cambridge_building_centroids.geojson"
project.settings.destinations_file = "MA_bus_stops.geojson"

project.settings.search_radius             = 500
project.settings.destination_weight_column = "weekly_departures"

project.RunAccessibility()
```

Ready-to-edit driver scripts are in [`examples/`](examples/) —
`UNA_Workspace.py` for single analyses and `UNA_Batch.py` for running
many analyses from a pairings CSV.

## Documentation and tutorials

**Documentation** and **tutorials** for the UNA Python package are found at: https://urban-network-analysis.readthedocs.io/en/latest/index.html



## Performance and verification status (2026-10-12)

Measured claims from the una_10x / una_platform verification campaigns. Every
number below comes from a preregistered, paired, bitwise-checked measurement
campaign; the campaign records (receipts and raw evidence) were archived
off-repository on 2026-10-15 to Dropbox `Alan Synn/Research/una-x/` (per-file
sha256 manifests included) and remain recoverable from this repository's git
history.

**Throughput (CPU, public `RunBatch`).** On the frozen observed-network cohort
of 32 distinct parameter jobs (Boston network, workload sha256
`235405de67cf22dc…a60a`), through the installed public `RunBatch` API at a
**cold** call boundary (initial trace/JIT cost included, timer brackets only
the public call), the optimized engine is **2.38x faster than the baseline
`395cdc5` build** (median paired ratio **2.3798**, one-sided 95% bootstrap
lower bound **2.2345**; 7 paired windows, seed 20261003, 20000 resamples;
installed wheels, equal worker resources, cache off). The comparator
`395cdc5` is itself an already-optimized build — this is the margin over the
strongest current implementation, not over a slow original. Outputs were
bitwise identical between the arms in every window (64 files per window,
per-file sha256), and protected cells showed no degradation (median 0.85,
95% upper 0.93). This is an honest measured result below the program's
stretch target of 10x; the gate was evaluated at full strength and reported
as `target_unmet` rather than lowered. Single-job `RunAccessibility`
throughput was not separately claimed (batch scope only). A further,
independently gated **1.11x** lazy-import improvement (median 1.10654, 95% LB
1.07908, byte-identical outputs) is merged here together with independently
reviewed batch checkpoint/resume correctness fixes; on the combined tree a
confirmatory paired gate (G4RC, `g4rc-20261012T0512Z`) measured **3.08x**
median (one-sided 95% lower bound **3.03x**, 7/7 windows positive, export
outputs byte-identical in every window) over the pre-merge tree, recorded as
merge-decision evidence — a different comparator and timing boundary than
the headline above, which remains measured against `395cdc5` on installed
wheels at a cold call boundary.

**GPU.** Substantive GPU execution is implemented and qualified on a real
NVIDIA device — recorded as a Tesla V100-PCIE-16GB (sm_70, partition
`gpu-v100`; an earlier revision of this paragraph said "A100", which no
tracked evidence supports — see `gpu_qualification.json` and `PLAN_GPU.md`
§2, archived 2026-10-15 (git history / Dropbox `Alan Synn/Research/una-x/`)):
the device kernel gate passed (rc=0) with the
association-free device stages bit-exact against host references, and a
declared-order fold cell is durably recorded. The parity domain of the GPU
fold is the certified battery row set only — general rows diverge by ±1 ulp
because device transcendental functions (libdevice `exp`) differ from host
NumPy by 1 ulp on specific arguments, and the substitution domain against
the delivered numba fastmath fold is empty at cohort-representative scale
(0/4 outputs bit-equal). The GPU route therefore remains
battery-domain, **CPU is the default**, and **no GPU-vs-baseline end-to-end
speedup multiple is claimed** — none was measured through the public API,
and promoting the GPU cell on parity grounds was refused.

**Native backend.** An ahead-of-time C++17 native core
(`urban_network_analysis.backends.native`) implements the characterized
scope search and destination adjustment for both numerical profiles,
verified bitwise against the compiled engines (legacy) and the independent
reference (corrected_v1), with ASan/UBSan-clean builds, fail-sentinel
engagement tests, and a portable pure-Python fallback that is
byte-identical. It ships as an explicit route, not a silent default.

**Madina API parity.** `urban_network_analysis.compat.madina` mirrors the
pinned upstream City-Form-Lab/madina (`8b5c3bd3`): the zonal and una
namespaces (tools, paths, betweenness, workflows) are delivered with
adversarially reviewed suites (131 tests green), and top-level namespace
parity is machine-checked live — 61 of 62 upstream public names, with the
single missing name (`pdk`) recorded in the attribution ledger; the `m.una`
attribute chain and all submodule identities are pinned. A drop-in
installable `madina` shim distribution is specified but not yet packaged
(open packaging item). Numerical profiles `una_legacy`, `madina_legacy`,
and `corrected_v1` are retained as distinct, separately tested references.

## Versioning

The package version is defined once, in
`src/urban_network_analysis/__init__.py` (`__version__`), and read by
the build system at install time. Check yours with:

```python
import urban_network_analysis
print(urban_network_analysis.__version__)
```

## Citing UNA

If you use UNA in academic work, please cite:

- Sevtsuk, A., & Mekonnen, M. (2012). Urban network analysis: A new
  toolbox for ArcGIS. *Revue Internationale de Géomatique*, 22(2),
  287–305. https://doi.org/10.3166/RIG.22.287-305
- Sevtsuk, A. (2021). Estimating pedestrian flows on street networks:
  Revisiting the betweenness index. *Journal of the American Planning
  Association*, 87(4). https://doi.org/10.1080/01944363.2020.1864758
- Sevtsuk, A., & Kalvo, R. (2024). Modeling pedestrian activity in cities 
  with urban network analysis. Environment and Planning B: Urban Analytics 
  and City Science, 52(2). https://doi.org/10.1177/23998083241261766
- Sevtsuk, A., & Alhassan, A. (2025). Madina Python package: Scalable
  urban network analysis for modeling pedestrian and bicycle trips in
  cities. *Journal of Transport Geography*, 123, 104130.
  https://doi.org/10.1016/j.jtrangeo.2025.104130

## License

MIT — see [LICENSE](LICENSE).
