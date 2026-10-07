"""GPU region surface (G10) for the accessibility fold family.

Buildable-without-hardware implementation of the region specified by
campaign_runs/una_10x/tenx-20261005T2331Z/g00/gpu_region.json:

  - gpu/backends/gpu.py   capability/registry surface (dossier-10 ABI shapes)
  - gpu/runtime/          fail-closed availability, buffer ownership, region
  - gpu/kernels/          device source (K0/K1), host twins, exact K2 tie-law
                          emulation, K3 association plans, gate harness

HARDWARE HONESTY: no CUDA device is required to exist for this package to
import and be tested.  Every device-execution path fails CLOSED with a typed
error on a host without CUDA-class hardware + FP64 + a passed per-cell
bit-validation gate.  There is no host-twin fallback in production paths: the
twins are test/spec artifacts only.  A CPU-only test can never mark hardware
validation passed, and nothing in this package qualifies a backend route.
"""
