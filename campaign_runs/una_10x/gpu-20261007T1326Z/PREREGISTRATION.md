# G11-Delta crossover preregistration (filed BEFORE any battery run)

Run dir: /u/dsynn/projects/una-x/runs/gpu-20261007T1326Z
Filed: 2026-10-07T13:26Z (UTC), before gate/battery execution.
Author: gpu-engineer agent (G11-Delta, task #37).

## Crossover arm choice (per work order, stated before the battery)

CPU arm: **Delta CPU arm, strongest matched configuration w4t1 q64 cache-off**
(4 workers x 1 thread, queue depth 64, engine cache off), executed on a Delta
compute node of the same node class as the GPU job's host node.

Why not the local-Mac Q10 walls (36.21-40.90 s): those figures come from a
different ISA and a different host. Using them needs a cross-ISA equivalence
argument, and matched-budget law forbids borrowing an arm the qualification
host does not own. The Delta CPU arm shares the engine source, the inputs, the
env python, and the node class with the GPU arm; the comparison is then a
device-boundary comparison on one host class, which is the claim G11 makes.

## Admission rule (frozen before execution)

- FULL-service crossover: all required outputs counted on both arms
  (preparation, first trace, integrity, durability, cleanup — same accounting
  frame as the campaign's PUBLIC 10x boundary).
- Per-cell admission into CapabilityRegistryEntry ONLY on a median wall win
  for the GPU arm vs the CPU arm, measured on the same node class, matched
  resource budgets (same core count billed class, same memory cap, same I/O
  root).
- The gate default stays CPU regardless of the crossover outcome: device
  dispatch requires the rc=0 bit-validation gate on the real device, and an
  admitted cell changes the DEFAULT only through an explicit, separately
  recorded decision.

## Gates this preregistration does not touch

- The rc=0 device bit-validation gate (una_legacy.fold.default) is a PARITY
  verdict, not a performance verdict; its outcome never depends on the
  crossover.
- tenx_status remains target_unmet; no 10x claim arises from this run dir.
  A future 10x claim needs its own preregistered gate.
