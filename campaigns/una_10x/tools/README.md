# Packet tools

All tools are stdlib-only and do not run application benchmarks automatically.
- validate_packet.py: offline required-file/DAG/prototype integrity checks.
- audit_source.py --repo PATH --out NEW: read-only Git source/provenance audit; refuses overwrite.
- gate_10x.py --samples JSON --out NEW: strict paired installed-public arithmetic gate with frozen
  bootstrap. Refuses synthetic/component/unmatched/failed/empty samples; no filtering of failures.
- check_completion.py --decision JSON --evidence-root PATH: conservative claim/capability/custody
  record validation. It cannot independently attest executions or replace an independent review.

Do not manufacture true fields to satisfy tools. Each has to point at actual execution evidence.
The tests intentionally use synthetic records to test validators; they are NOT benchmark evidence.
