# UNA CPU throughput: laptop execution packet

**Start with [START_HERE.txt](START_HERE.txt).** Open Claude Code in the extracted packet directory and paste that file. The agent should execute the campaign, not return a second plan.

The intended repository is `AlanSynn/una-x`; the proposed local integration branch is `perf/una-cpu-throughput`; the audited base is `c15ebda6981397f46eed5c2d55229f71e57d44fb`. The laptop checkout, hardware and production workload have not been observed. GPU implementation and evaluation are outside this campaign.

## What is ready, and what is not

This is an executable engineering handoff, not an assertion that the candidate is release-ready. It includes the original candidate, original evidence, a dependency DAG, proofs and counterexamples, executable inventory/validation tools, bounded experiments, acceptance rules, and a merge-manifest template.

The prior optimization replaces repeated per-node edge scans with stable ordered CSR construction. The prior synthetic component measurements are preserved, not upgraded to genuine application measurements. Complete-checkout oracle verification, real public-API execution, Feather, installed-wheel verification, laptop resource tuning and independent review remain laptop tasks.

The first implementation decision is **verify or reject the supplied candidate**, not select a new library. At most one additional numerical work-reduction hypothesis is authorized, subject to coverage and proof gates. There is no obligation to implement it.

## Reading map

| Reader | Entry points |
|---|---|
| Coordinator | [Execution prompt](CLAUDE_CODE_EXECUTION_PROMPT.md), [authority](DESIGN_AUTHORITY.md), [plan](PLAN.md), [DAG](TASKS_CLAUDE.yaml) |
| Source/proof worker | [source audit](SOURCE_AUDIT.md), [semantic contract](SEMANTIC_CONTRACT.md), [numerical contract](NUMERICAL_CONTRACT.md), assigned dossier |
| Performance owner | [benchmark protocol](BENCHMARK_PROTOCOL.md), [resource policy](RESOURCE_POLICY.md), [workloads](WORKLOADS.md), [performance model](PERFORMANCE_MODEL.md) |
| Reviewer/integrator | [promotion policy](PROMOTION_POLICY.md), [merge scope](MERGE_SCOPE.md), [decision contract](DECISION_CONTRACT.md) |
| Human reviewing completion | generated evidence/final decision and merge manifest, plus [packet checks](packet_checks.json) |

## Packet integrity

From this directory, using Python 3.11 or newer:

```sh
python tools/validate_packet.py
python tools/verify_prior.py
```

These commands do not install anything, run UNA, contact a provider, or modify a repository. Validation outputs describe the packet only. The prior archive and extracted files are retained unchanged and retain their MIT license. `reference_inputs/AUTHORING_GUIDE.md` is the user-provided authoring source.

The packet uses JSON syntax for `TASKS_CLAUDE.yaml`, which is also valid YAML. This keeps its validator free of a YAML dependency. No special agent framework or model alias is required.