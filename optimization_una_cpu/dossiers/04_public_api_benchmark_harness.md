# Dossier 04: public API benchmark harness

Build a harness outside production runtime that accepts source/wheel arm, workload manifest, worker count, threads per worker, queue depth, writer limit, cache/JIT state, and output root.

Boundaries: import/startup, input/topology, engine construction/CSR, centrality/search+metrics where separable without changing code semantics, export, full public call, full job, batch throughput. Final claims use full public/job/batch boundary.

Use unique output directories, monotonic clock, process-tree memory sampling, exact exit status, and output hashes/signatures after the timed boundary. Do not monkeypatch production functions in final timing.

Installed mode must launch from outside checkout and assert `urban_network_analysis.__file__` belongs to installed environment. Record effective Numba/native threads and dependency versions.