# Independent review report — MADINA_ZONAL mdz-20260930T2124Z

Reviewer: mdz-reviewer2 agent (fresh context, read-only, adversarial).
A first reviewer session (mdz-reviewer) inspected the tree (custody md5
snapshot /tmp/mdz_review_snapshot.txt, still matching at handoff) but
never delivered a report; mdz-reviewer2 is its independent replacement,
not its echo. Report delivered verbatim below.

---

MADINA_ZONAL — Independent Adversarial Review

## Verdict: CHANGES_REQUIRED

The parity discipline is genuinely sound (same-interpreter two-arm design, live-verified reflection ledger, effective uuid4 control, and all 23 retained digests plus the saved HTML reproduce bit-identically on an independent rerun). But two MAJOR test-honesty/evidence-fidelity defects block approval; both are fixable with test/evidence text edits, no src changes.

## Findings

1. MAJOR — insert_multinode_edge "add/remove roundtrip restores the digest" is false; the add_node_to_graph/remove_node_to_graph graph mutation is effectively unpinned.
File: tests/madina_api/zonal/_zonal_scenario.py:259-267. The comment says "insert then remove on a copy restores the digest", but only G2.number_of_nodes() is recorded; "equals_before" compares node counts only. Probe (same seed/steps in the reference venv): after_add == before (full graph_digest) = False; after_remove == before = False. Edge sets differ: before has 10 edges incl. 2|10; after remove 11 incl. self-loop 0|0 (weight 3e9421f5f40d8376), corrupted 0|10 (4028fffff5ef0506 vs original 4029000000000000), and 2|10 gone. Node count coincidentally restores (9->9->9). Root causes: street_id = sorted(n.street_node_ids)[0] is a street node whose nearest_edge_id/edge_start_node/edge_end_node are the zero-filled placeholders from insert_node's pre-fill (chain-rebuild then corrupts the graph), and G2.graph["added_nodes"] IS n.d_graph.graph["added_nodes"] (networkx shallow copy). Consequence: receipt.json verification #3 ("add/remove roundtrip restores graph"), receipt handoff ("add/remove roundtrip restoring the graph"), and evidence.json scenario_matrix overstate what is pinned; a facade bug in either primitive preserving node count but corrupting weights/edges would pass. Fix: record graph_digest(G2) after add and after remove (making the actual post-roundtrip state bitwise-pinned across arms), reword comment/receipt/evidence to what actually holds, and/or roundtrip on a real inserted node id.

2. MAJOR — evidence.json m4_delivery contradicts the retained artifact. evidence.json line 39: "asserted from the digest: per-edge counts {0: 2, 3: 1}". Retained artifacts/facade/insert_multinode_edge.json records inserted_per_edge = {"0": 4, "3": 2} (two insert_node calls over the same 3-point layer: 2 origins + 2 destinations on edge 0, 1+1 on edge 3). Evidence text is wrong by 2x and misdescribes the fixture. Fix the text (or record per-label counts in the digest and cite those).

3. MINOR — vacuous digest field duplicate_label_error: "KeyError" (_zonal_scenario.py:330). The preceding z.load_layer("ldup", ...) loads a NEW label and raises nothing (probe: "ldup load: NO ERROR raised"); the field is a hardcoded string that can never fail. Real duplicate-label coverage exists in turn_params (load_dup_label with the actual message), so parity is not weakened, but the field misleads consumers. Fix: record the real try/except (e.g. reload "l1") or delete the field. Same family: test_facade_importable_and_versioned asserts only stdout.startswith("ok ") — the printed len(DEFAULT_COLORS) is unchecked decoration.

4. MINOR — dossier-named Zonal items absent from this task's tests and unmentioned in its evidence. Dossier 01 line 29 lists clear_nodes under minimum supported Zonal behavior and lines 57-58 require "clear/reinsert"; no test under tests/madina_api/ calls clear_nodes (grep: zero hits), and neither evidence.json limits nor the receipt mentions it. Coverage does exist in TOPOLOGY's tests/platform_geometry/_scenario.py:419 (scenario_clear_reinsert, both arms), as do file-path load_layer (scenario_layer_order), tolerance>0 (scenario_tolerance line 356), prepare_geometry Polygon/Z/MLS (scenario_prep / scenario_mls_pure), and redundant keep/discard/split (lines 654-656) — but a reader of MADINA_ZONAL evidence alone cannot tell. Genuinely uncovered anywhere in-repo and unlisted: Layers.__setitem__ int path (str path covered at _scenario.py:414), Network.set_node_value, update_light_graph remove_nodes path, create_deckGL_map radius/width/opacity/text branches, insert_node weight_attribute on a destination. Fix: add limits/cross-reference lines mapping dossier items to the owning suite.

5. NOTE — reflection surface excludes dunders by construction (_zonal_reflection.py:39 startswith("_") filter), so Layers.__getitem__/__setitem__/__contains__/__iter__/__str__ and Zonal.__getitem__ signatures are never reflected; behavior mostly scenario-covered except __setitem__ int path (see 4).

6. NOTE — digest machinery verified honest; two theoretical gaps. Probes: -0.0 vs 0.0 distinct (8000.../0000...), NaN canonical 7ff8..., object columns via injective str() (Python float repr round-trip unique), bool/float32/categorical branches correct. graph_digest's key+"#2" branch (_zonal_scenario.py:95-96) is unreachable (nx.Graph cannot hold parallel edges) — harmless dead code. gdf_digest omits index name and active geometry-column name (not exploitable here; both arms share the code path). WKB hash truncated to 96 bits — adequate at this fixture scale.

7. NOTE — sabotage drop_text_layer is a misnomer: the default deck has no text layer; the mutation decrements default_deck.n_layers 2->1. The create_deckGL_map "text" branch is unexercised (see 4). Rename or target a real text layer.

8. NOTE — sabotages are digest-level mutations of the reference arm, not code mutants of the facade. Verified each mutates exactly one recorded value (single-field diffs vs honest artifacts for flip_turn_threshold, hide_set_style_error, swap_default_color) and all 7 are comparator-selected. The chain "comparator sees field => broken facade would be caught" is sound, but receipt.json's "intentionally broken candidates" overstates; TOPOLOGY's suite does the stronger end-to-end form (monkeypatched facade mutants, _scenario.py:571-597).

9. NOTE — whole_repo_regression_mdz.log is a 6-line tail (summary + exit 0, no pytest header/collected count); the first-launch collection error described in the receipt is prose-only with no retained evidence.

## Verified-good (checked, not taken on faith)
- Reproducibility: fresh isolated rerun (UNA_LARGE_E2E_ARTIFACTS=/tmp/mdz2_review/artifacts, pytest tests/madina_api/zonal/ -q) -> 21 passed; all 23 retained digests byte-identical (cmp) to artifacts/{facade,reference,reflection}; map_pts.html sha256 identical across arms and runs.
- Reflection ledger live-verified: ref-only names exactly {os, pdk}; facade-only exactly the 14 ledgered names; SIGNATURE_ALIASES key/value each match exactly one real signature; class surfaces and constants equal after aliasing; identity quirk exact (upstream sys.modules['madina.zonal'] is the package with __path__, attribute madina.zonal is the inner zonal.py; compat restores the package).
- Facade cleanliness: git diff HEAD -- src/ empty; no "import madina" in the facade; venv has no installed madina/una dist; reference arm loads from .refs/madina_ref/src; facade-vs-reference source diff is only D3/D4 + docstrings/blanks.
- uuid4 control: pydeck 0.9.3 bindings/layer.py:78 = "self.id = id or str(uuid.uuid4())" with module-level "import uuid" — the module-attribute patch controls emitted ids; controlled ids present in the retained deck JSON (d6b4dc99-..., d3ef2c42-...); UUID(int=...) version/variant bits irrelevant (opaque string id).
- Failures are real: 14/14 turn_params validation failures with distinct upstream messages (none NO_ERROR); quirks artifact shows the four pinned failures verbatim; color_gdf partial-dict KeyError confirmed (utils.py:101).
- Env pins: D3 no-env-mutation, D4 actionable ImportError, namespace-identity restore execute real subprocess probes in the pydeck-free alan interpreter; venv is Python 3.11.4 as claimed.
- No repository writes by the suite under normal runs (artifacts to UNA_LARGE_E2E_ARTIFACTS).

## Files inspected
tests/madina_api/zonal/{_zonal_scenario.py,_zonal_reflection.py,conftest.py,test_zonal_surface_parity.py,test_zonal_reflection.py,test_zonal_facade_env.py}; src/urban_network_analysis/compat/madina/{__init__.py,zonal/*.py}; .refs/madina_ref/src/madina/{__init__.py,zonal/*.py}; campaigns/una_platform/evidence/tasks/MADINA_ZONAL/mdz-20260930T2124Z/{evidence.json,receipt.json,logs/*,artifacts/**}; campaigns/una_platform/dossiers/01_api_and_reference.md; campaigns/una_platform/TASKS_CLAUDE.yaml; tests/large_e2e/_legacy_paths.py; tests/platform_geometry/{_scenario.py,test_zonal_parity.py}; venv pydeck/bindings/{layer.py,widget.py}.

## Commands run
Reflection probes in venv (_zonal_reflection.py --arm {reference,facade} to /tmp/mdz2_review); surface/alias/namespace comparison via alan python; retained-artifact inspections (uuid ids, per-edge counts, graph sizes, quirks, sabotage single-field diffs); roundtrip and digest edge-case probes (/tmp/mdz2_review/probe/probe_roundtrip.py, probe2.py, probe3.py); facade-vs-reference normalized diffs; git status / git diff HEAD --stat -- src/; suite rerun UNA_LARGE_E2E_ARTIFACTS=/tmp/mdz2_review/artifacts alan-python -m pytest tests/madina_api/zonal/ -q (21 passed, 151.69s) followed by cmp of all 23 digests and sha256 of both map_pts.html files against retained evidence. Whole-repo tests not rerun per coordination constraint. Read-only except /tmp/mdz2_review scratch.
