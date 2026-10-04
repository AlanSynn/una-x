# Mandatory implementation test and fault matrix

Each row produces expected/observed status and source/fixture/environment digests. Packet model
unit tests below do NOT satisfy these actual-runtime tests. Run L0/L1 first; L2 installed public
calls next; medium selection only after correctness. No production-size rerun after every edit.

| ID | Case | Required observable |
|---|---|---|
| PP01 | Original path vs snapshot reader for every admitted format | Same values/schema/order/UIDs/CRS/dtypes/geometry and required warnings/errors. |
| PP02 | Two rows reuse same numerical prep but different output paths/beta | One admitted prep, distinct computed results, both full artifact sets. |
| PP03 | Bytes change with same path, length and mtime | Revalidation detects or domain refuses; never stale hit. |
| PP04 | Reader, weight/UID column, default, precision, CRS, profile changed independently | Exactly correct dependency invalidation; no cross-profile hits. |
| PP05 | Mutable source, sidecars, custom reader/dataframe, unsupported geometry/dtype | Existing fallback/error, no hidden source snapshot policy. |
| PP06 | Corrupt manifest/WKB/offset/type/budget and owner death | No partial success; bounded cleanup/wakeup, row-ordered failure. |
| PP07 | User mutates returned arrays/GeoSeries/last topology after return | Other rows, cache and future runs unchanged. |
| PP08 | Mixed valid/invalid/serial/turn/elevation/obstacle rows | Original flag resets, settings identity, validation and failure prefix. |
| PP09 | Small budget, many distinct roots, eviction with live leases | Budget never exceeded by cap floor; no use-after-release or registry leak. |
| RP01 | Python replay vs compiled outer vs original engine | Bit-exact each output and required dtypes, including empty origins/destinations. |
| RP02 | Fold lengths at vector boundaries, ties, +/-0, nextafter cutoffs | Same compiled specialization/reduction order and first divergence retained. |
| RP03 | Wrong offsets/IDs/type/layout/digest at late origin | No unsafe indexing or partially published prior output; exact facade refusal. |
| RP04 | Different raw weight/KNN scalar types with same float64 conversion | Keys remain distinct unless reviewed typed equivalence proves reuse. |
| MR01 | Each metric dependency varied once, every decay branch | Only identical dependency closures reuse; independent outputs remain mutable. |
| MR02 | Split fold differs one ULP on any admitted signature | Partial route rejected/refused; full fold continues, no tolerance. |
| MR03 | Required warning/error only in nominally unused metric | No skipped observable; narrow admission or keep calculation. |
| RX01 | Reverse completion, delayed cursor, all ready credits reserved | No deadlock/unbounded map; cursor order unchanged. |
| RX02 | Full pipe, oversized last-state result, worker death during put | Retain every completely delivered result; bounded cancellation, no hung receiver. |
| RX03 | Receiver exception, corrupt staged bytes, file fsync/rename/journal failure | Correct error class and committed prefix; no later public artifacts. |
| RX04 | Cancel after first file of row, then resume | No complete journal for partial row; verify/regenerate exactly required outputs. |
| RX05 | Two batches/same filenames/same root | No cross-run ownership, cleanup or cache-generation corruption. |
| WS01 | Cold FIRST_COHORT and RESIDENT_DISTINCT | First construction/setup included, identities/regimes distinguished. |
| WS02 | Baseline cache-off and strongest permitted input cache | Both comparisons retained, fixed-policy claim not mislabeled strongest-policy. |
| WS03 | Installed wheel from outside both checkouts in every worker | Module/binary hashes and engagement prove actual implementation used. |
| WS04 | Instrumented vs uninstrumented paired controls | Sampler overhead quantified, no causal overstatement of co-tenant effects. |
| WS05 | All expected artifact bytes/state with frozen timestamp policy | 64 files for OBS_COHORT, byte comparisons retained; no basename collision hiding missing files. |

Compare artifacts by original per-row relative-path mapping and the EXISTING declared clock/
naming policy. Do not invent a new canonicalization because faster runs cross fewer minutes.
Preserve output schemas/metadata and expose any filename-policy issue independently. Bare hashes
without raw availability are not a new revalidation. Keep actual data counters distinct from
telemetry counters, and integrity failures distinct from benign pre-execution capability refusals.
