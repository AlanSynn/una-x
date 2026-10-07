"""Kernels for the GPU fold region.

- ``fold_kernels.cu``     device source: K0 (order-preserving compaction) and
  K1 (per-element coefficients), double precision only, no reductions.
- ``host_twins.py``       host reference twins + the exact K2 tie-law
  transcription + K3 association-tree plans.
- ``validate_device_bits.py``  the named per-cell bit-validation gate harness
  (unrun on this host; fails closed without a device).
"""
