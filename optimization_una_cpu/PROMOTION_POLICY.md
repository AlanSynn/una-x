# Promotion policy

Pre-result gates:

1. Contract gate: no known semantic/API/artifact violation in admitted domain; unsupported cases use original path.
2. Numerical gate: adversarial L0/L1 and genuine L2 compare baseline/candidate at required exactness, including state where relevant.
3. Installed-path gate: built wheel installs cleanly, import resolves outside checkout, optimized path executes for admitted workload, fallback executes for unsupported workload.
4. Complete-region performance gate: B beats strongest tested A under equal CPU/RAM and same required outputs. Kernel-only speed is insufficient.
5. Memory gate: process-tree peak remains within discovered safety budget and no unbounded queue/materialization is introduced.
6. Cold-start gate: no unacceptable first-use regression for expected execution profile; otherwise scope the optimization to warm/resident use and retain fallback/default policy accordingly.
7. Maintainability/DX gate: no new mandatory compiler, runtime download, backend environment variable, or vendor dependency.
8. Genuine-pipeline gate: speedup persists through real GIS public API, not just synthetic arrays.
9. Review gate: independent reviewer checks immutable patch, proof, tests, raw evidence, failure paths, and claim boundary.
10. Target gate: production-wide objective can be claimed only when actual workload and throughput target are supplied and tested.

Do not loosen gates after failure. If measurements are noisy, one preregistered repeat block is allowed. Slower/inconclusive candidates are rejected or labeled inconclusive, not promoted through selective samples.