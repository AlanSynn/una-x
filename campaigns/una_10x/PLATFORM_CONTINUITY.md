# Relationship to the unfinished platform campaign

`campaigns/una_platform/` is RETAINED INCOMPLETE, not historical success. The active /goal now
prioritizes 10x execution. Existing completed platform tasks and open branch work remain assets.
Do not edit their receipts to fit this campaign, do not restart tasks from their old 16bf404 base,
and do not let two agents own the same batch/cache/engine files.

At S01 construct `platform_obligations.json` from the live task DAG, receipts, source, tests and
review findings. Each obligation records source SHA, implementation status, evidence status,
remaining work, owner, new task mapping and explicit acceptance gate. A receipt alone does not
prove current installed engagement. Native/GPU next-task messages are not implementation.

Retained obligations and integration mapping:
- Madina API/functionality, shim and workflows: X00; parity matrix against pinned Madina.
- Scientific fixes/corrected reference: S02 and X00; preserve una_legacy, madina_legacy and
  corrected_v1 as distinct references. Never use a model change to claim a speedup.
- Existing public RunBatch plan/runtime/checkpoint: C30 integrates reuse; H01 tests actual
  execution, ordered commits, cancellation and failure-prefix behavior.
- Existing content-addressed store: C10/C20 add finer numerical identities and resident leases;
  retain corruption, single-flight, byte quotas, safe decoding and atomic publication.
- Existing native/backend contracts: N00/N10 specialize the sparse/reuse region; integrate
  reviewed ongoing work first. Do not port the slow portable characterization oracle and use
  it as a deliberately weak comparator.
- GPU support: G00/G10 are real implementation/qualification tasks inherited from the program.
  Hardware absence blocks G10, not independent CPU work. A fallback-only GPU route is not
  substantive support. CPU 10x may qualify while platform status remains incomplete.
- City failures, diagnostic-mode behavior, dependency compatibility and test-frame hardening:
  H01/X00. Preserve incident identity and actual reproduction status.
- Packaging, CI, installed-path and final review: I00/V00/R00/Z00. Unrun CI is unavailable.

No new scientific default is introduced by this packet. When retained prose and executable
references disagree, capture both and issue a versioned contract ruling before modifying
production behavior. Example: kernels/SPEC.md's equal-duplicate paragraph is not a proof of
first-write-only behavior; two equal improvements can BOTH pass the pre-row snapshot. The
actual compiled engine and minimal adversarial test settle that legacy question. Do not erase
the second heap push based on prose. Numerical authority is a profile-specific tested reference,
not a majority vote among descriptions.

Publish two final dispositions: `tenx_status` and `platform_status`. Only completion of all
mandatory inherited capabilities AND their required hardware/API/scientific gates permits
platform_status=qualified. A 10x-only result never marks those missing tasks done.
