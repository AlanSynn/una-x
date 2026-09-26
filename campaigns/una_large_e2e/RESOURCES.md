# Machine admission and safety

All limits below are policy defaults, not measured resources. H00 records actual OS/ISA, Python/native packages, logical/physical CPUs, affinity/cgroup quota when supported, RAM/available RAM/swap, disk free, threadpoolctl output, Numba threading layer after initialization, Arrow thread settings and relevant environment. No accelerator work.

Default CPU slots C: effective quota/affinity rounded down to at least one, reserving one slot when more than one is available unless the user supplies a different budget. Record heterogeneity. On Apple silicon, limiting worker count is not proof of P-core-only affinity; report execution-slot budget and OS scheduling, not 'E-cores excluded'. Do not invent power/thermal controls or privileged pinning.

Default campaign memory ceiling: min(0.65*physical_RAM, 0.80*available_RAM_at_admission). Keep at least max(1 GiB, 0.10*physical_RAM) available as an external-pressure stop threshold. Also reserve disk for inputs, both arms' outputs, scratch, wheels and logs before a run. These are conservative policies; record measured peak separately. Sampling cannot guarantee a hard memory cap. Use OS limits where safe/supported and a campaign-owned watchdog; do not claim hard enforcement otherwise.

Actual pool inventory is mandatory. Accessibility has Numba H; flow has topology.num_threads, potentially separate turn/cluster executors, BLAS/OpenMP/Arrow pools and Numba phases. W*H_numba<=C is insufficient for flow. For distinct sequential phases within a job, charge maximum concurrently runnable threads; across W jobs phases can overlap, so use a conservative sum of each job's worst admitted concurrency. Writers are bounded separately. Record idle threads separately from active compute but do not assume idle behavior without observation.

Freeze logical flow stripes K before timing because K affects numerical sum order. Default-profile tests retain baseline K. If default K does not fit the budget, report default-profile qualification unavailable or use a separately labelled explicit thread profile equally in both arms. Never lower K silently. Physical execution can be bounded below K only by an independently reviewed fixed-whole-stripe schedule; no per-origin redistribution.

Before spawning W>1, measure one complete job including import/JIT/outputs. Estimate W from simultaneous live memory, not a single worker's minimum. Start queue capacity at W; tune only bounded alternatives. W can be lower at larger scales. Output directories and mutable UNA/Settings/Topology instances are per job. Spawn processes; never fork an already initialized threaded/JIT runtime.

One global heavy-execution lease. Initially one compiler or heavy test process. During performance confirmation no other campaign build/test workload. Monitor unrelated user load; record predeclared pressure/load rejection rules rather than excluding slow samples post hoc.

All nonfinite/hazard probes run in isolated children with timeout and process-tree memory watchdog. On timeout/OOM terminate only owned children, collect partial logs, record reason and unsuccessful count. Do not automatically retry the identical oversized configuration. Do not reduce the frozen network, OD set, radius or outputs to turn a failed target run into a pass.

A job timer ends at public return; a cold timer ends after child exit; a batch timer ends after the last validated result is collected. Durable-flush claims require separate evidence. Mac/Linux process-tree monitoring is the initial supported harness domain; unsupported monitoring platforms remain unqualified rather than silently reporting parent RSS.
