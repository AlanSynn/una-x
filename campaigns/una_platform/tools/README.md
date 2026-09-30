# Packet tooling

These tools support handoff validation, not complete runtime certification.

- validate_packet.py: stdlib DAG/reference/manifest/requirement checks.
- inventory_api.py ROOT: read-only AST census; manual semantic expansion remains required.
- source_audit.py ROOT --reference una|madina: raw Git-blob comparison to pinned files.
- bitparity.py REFERENCE.npz CANDIDATE.npz: exact typed array bytes, no tolerance/pickle. Requires NumPy.
- check_completion.py RECORD.json: fail-closed structural/evidence-hash completion audit.

Run from any working directory. Scripts derive packet paths from __file__, not personal checkout
paths. Outputs go to stdout; redirect to a new campaign-owned evidence path. Unit tests use only
temporary directories and never modify source or committed historical evidence.
