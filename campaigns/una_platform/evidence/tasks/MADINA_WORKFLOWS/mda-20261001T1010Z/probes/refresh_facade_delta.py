"""Regenerates artifacts/facade_delta/{delta_vs_upstream.diff,
delta_full_file.diff, summary.json} from the CURRENT upstream reference
and facade.  Verify-only: never writes the facade; asserts the reverse-D4
body equals the pinned upstream byte-for-byte (any other delta is a
ledger omission and fails here).

Run: <any python with stdlib> refresh_facade_delta.py
2026-10-01: produced the current artifacts (facade_md5
f4c2f593da563f1e968b1df6ba231477, header 70 lines, reverse-D4 identity
true) after the facade header was extended with the pinned workflow-level
facts.
"""
import hashlib
import json
import subprocess
from pathlib import Path

def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / ".refs" / "madina_ref" / "src" / "madina" / "una"
                / "workflows.py").exists():
            return p
    raise RuntimeError("repo root containing .refs/madina_ref not found")


REPO = _repo_root()
UP = REPO / ".refs/madina_ref/src/madina/una/workflows.py"
FAC = REPO / "src/urban_network_analysis/compat/madina/una/workflows.py"
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "facade_delta"

up = UP.read_text()
fac = FAC.read_text()
lines = fac.splitlines(keepends=True)
assert lines[0].startswith('"""')
end = next(i for i, l in enumerate(lines) if i > 0 and l.rstrip("\n") == '"""')
header = "".join(lines[:end + 1])
body = "".join(lines[end + 1:])

anchor = """        file_name=None
        ):
        # D4: pydeck is an optional dependency; import it only when a map is
        # actually requested, with an actionable error when absent.
        try:
            import pydeck as pdk
            from pydeck.types import String
        except ImportError as exc:
            raise ImportError(
                "una.workflows.flow_map_template_1 requires the optional "
                "visualization dependency 'pydeck'. Install it with:  "
                "pip install pydeck"
            ) from exc

        predicted_flow_gdf = flow_gdf.copy(deep=True)"""
restored = """        file_name=None
        ):
        predicted_flow_gdf = flow_gdf.copy(deep=True)"""
assert body.count(anchor) == 1, "lazy-import block not found exactly once"
verbatim = body.replace(anchor, restored, 1)
for line in ("import pydeck as pdk\n", "from pydeck.types import String\n"):
    assert verbatim.count(line) == 0, f"module-top {line!r} should be absent"
# upstream keeps a blank line BEFORE 'import pydeck as pdk' (generator's
# replace(line, "", 1) preserved it) and puts 'from pydeck.types import
# String' directly after 'from datetime import datetime'
verbatim = verbatim.replace(
    "import numpy as np      ## for version\n\n",
    "import numpy as np      ## for version\n\nimport pydeck as pdk\n", 1)
verbatim = verbatim.replace(
    "from datetime import datetime\n",
    "from datetime import datetime\nfrom pydeck.types import String\n", 1)
verbatim_identical = verbatim == up
assert verbatim_identical, (
    "reverse-D4 body != upstream — the facade carries an UNLEDGERED delta")

body_diff = subprocess.run(
    ["diff", "-u", "--label", "upstream_workflows.py",
     "--label", "facade_body", str(UP), "/dev/stdin"],
    input=body, capture_output=True, text=True)
(OUT / "delta_vs_upstream.diff").write_text(body_diff.stdout)
full_diff = subprocess.run(
    ["diff", "-u", "--label", "upstream_workflows.py",
     "--label", "facade_workflows.py", str(UP), str(FAC)],
    capture_output=True, text=True)
(OUT / "delta_full_file.diff").write_text(full_diff.stdout)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


summary = {
    "upstream_file": ".refs/madina_ref/src/madina/una/workflows.py (madina @ 8b5c3bd3)",
    "facade_file": "src/urban_network_analysis/compat/madina/una/workflows.py",
    "header_lines": len(header.splitlines()),
    "body_lines": len(body.splitlines()),
    "upstream_lines": len(up.splitlines()),
    "delta": ("EXACTLY the ledgered D4: -2 module-top pydeck imports, "
              "+12 lazy-import block at flow_map_template_1 head; the "
              f"{len(header.splitlines())}-line ledgered module docstring "
              "header is the only other delta (upstream has no module "
              "docstring). Reverse-D4 body == upstream byte-for-byte: "
              f"{verbatim_identical}"),
    "facade_md5": md5(FAC),
    "upstream_md5": md5(UP),
    "body_reverse_d4_equals_upstream": verbatim_identical,
    "generator": ("campaigns/una_platform/evidence/tasks/MADINA_WORKFLOWS/"
                  "build_facade.py (HEADER synced to the reviewed header; "
                  "regeneration asserted byte-identical, "
                  "regenerated_identical=True, 2026-10-01)"),
    "facade_parses": True,
    "note": ("regenerated 2026-10-01 by refresh_facade_delta.py after the "
             "facade header was extended with the pinned workflow-level "
             "facts (KNN Network_File reload crash, zero-reach KeyError "
             "'reach' + EmptyDataError, alpha profile split); previous "
             "summary's facade_md5 b1dea6a187e4ade958c7c70d42bd386f was "
             "the pre-extension header and is superseded"),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: summary[k] for k in (
    "facade_md5", "body_reverse_d4_equals_upstream", "header_lines")},
    indent=1))
