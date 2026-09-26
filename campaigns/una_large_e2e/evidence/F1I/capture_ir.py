"""F1I-N1 binding packet artifact: in-module IR side-by-side capture.

Loads BOTH trees IN-MODULE (B0 via the oracle b0_import singleton;
candidate via the una_f1_cand alias, real file paths preserved),
compiles the production _accumulate_od_flow of each at the OBSERVED
specialization (the exact signature recorded by the F1 suite's T7
census: int32 CSR/edge ids/preds, int8 direction, float64
distances/weights/outputs, int64 scalars), dumps full LLVM IR per arm
into this directory, and writes the fast-flag FP census with the
proof-5b discipline:

  * every FP instruction (fadd/fsub/fmul/fdiv/fcmp/fmuladd, and
    explicit llvm.fma.* intrinsics) with its fast-math flags, IN ORDER;
  * counts AND normalized site sequence must be EQUAL across arms —
    the ONLY permitted delta anywhere in the IR is the GIL
    save/restore pair around the compiled body (PyEval_SaveThread /
    PyEval_RestoreThread, expected 0 in B0 and 2/2 in the candidate);
  * any other delta = REJECTION per proof.md 5b / obligation 9.2.

The script manages its own FRESH on-disk cache root per run: with a
warm numba cache, the cache-hit load path retains no in-memory LLVM
module and inspect_llvm returns an empty stub — every run must be a
cold in-memory compile for the IR dump to be real.

Run: python capture_ir.py
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

# must precede the first numba import in this process
_FRESH_CACHE = tempfile.mkdtemp(prefix="f1i_ir_cache_")
os.environ["NUMBA_CACHE_DIR"] = _FRESH_CACHE

OUT = os.path.dirname(os.path.abspath(__file__))

ORACLE = ("/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
          "tests/large_e2e/oracle")
sys.path.insert(0, ORACLE)

import numba as nb  # noqa: E402

from support import b0 as _b0  # noqa: E402
sys.path.insert(0, "/Users/alansynn/orca/workspaces/una-x/wt-large-e2e/"
                   "tests/large_e2e/F1")
import cand_load  # noqa: E402

SIG = (nb.types.int32[:],         # indptr
       nb.types.int32[:],         # indices
       nb.types.float64[:],       # weights
       nb.types.int32[:],         # edge_id_of_arc
       nb.types.int8[:],          # dir_of_arc
       nb.types.float64[:],       # d_o
       nb.types.float64[:],       # d_d (dense gradient view)
       nb.types.int32[:],         # pred_o
       nb.types.int32[:],         # pred_d
       nb.types.int64,            # origin_virtual_node
       nb.types.int64,            # dest_virtual_node
       nb.types.int64,            # o_edge_id
       nb.types.int64,            # d_edge_id
       nb.types.float64,          # d_shortest
       nb.types.float64,          # budget
       nb.types.int64,            # decay_curve_id
       nb.types.float64,          # decay_beta
       nb.types.float64,          # decay_midpoint
       nb.types.float64,          # trip_volume
       nb.types.int64,            # n_net
       nb.types.float64[:],       # out_AB
       nb.types.float64[:],       # out_BA
       nb.types.float64[:])       # out_node_flow

FP_SITE_RE = re.compile(
    r"=\s*(?P<op>fmuladd|fadd|fmul|fsub|fdiv|fcmp\w*|"
    r"call\b[^;]*?llvm\.fma\w*)\s+"
    r"(?P<flags>(?:fast|nnan|ninf|nsz|arcp|contract|reassoc|afn)\s*)*"
    r"(?P<rest>.*)$")


def census_sites(ir):
    """Ordered (opcode, flags, normalized_rest) triplets for every FP
    instruction site."""
    sites = []
    for line in ir.splitlines():
        m = FP_SITE_RE.search(line)
        if not m:
            continue
        flags = " ".join((m.group("flags") or "").split())
        rest = re.sub(r"%\d+", "%", m.group("rest"))
        rest = re.sub(r"\s+", " ", rest).strip()
        sites.append([m.group("op"), flags, rest])
    return sites


def count_calls(ir, needle):
    return sum(1 for line in ir.splitlines() if needle in line)


# ---- full-IR residual classifier --------------------------------------
# Beyond the FP census: a line-pairwise classification of the ENTIRE
# normalized-IR diff. Every non-GIL diff line must pair with an
# opposite-side line that is identical under renumber-insensitive
# placeholders (SSA temp numbers, block-label numbering, module-name
# Itanium length prefix, numba ABI/source-hash mangling, string-const
# lengths). Float constants are NOT placeholdered — the FP census above
# already pins them at every FP site. The only unpaired lines permitted
# are the GIL declare/call insertions themselves.

def _placeholder(line):
    line = line.replace("urban_network_analysis", "MOD")
    line = line.replace("una_f1_cand", "MOD")
    line = re.sub(r"\d+MOD", "MOD", line)                      # length prefix
    line = re.sub(r"B\d+v\d+[A-Za-z0-9_]*", "MANGLED", line)   # ABI mangling
    line = re.sub(r"\[\d+ x i8\]", "[N x i8]", line)
    line = re.sub(r"\bB\d+\b", "%L", line)                     # block labels
    line = re.sub(r"%\.?\d+", "%T", line)                      # SSA temps
    return line


def classify_residual(ir_b, ir_c):
    diff = list(difflib.unified_diff(
        ir_b.splitlines(), ir_c.splitlines(), lineterm="", n=1))
    state = {"m": [], "p": []}
    gil, pairs, bad = [], [], []

    def flush():
        m, p = state["m"], state["p"]
        state["m"], state["p"] = [], []
        if not m and not p:
            return
        core_m = [l for l in m if "PyEval_" not in l and l[1:].strip()]
        core_p = [l for l in p if "PyEval_" not in l and l[1:].strip()]
        gil.extend(l for l in m + p if "PyEval_" in l)
        if len(core_m) != len(core_p):
            bad.append({"unbalanced": (len(core_m), len(core_p))})
            return
        for lm, lp in zip(core_m, core_p):
            pairs.append((lm, lp))
            if _placeholder(lm[1:]) != _placeholder(lp[1:]):
                bad.append({"b0": lm, "cand": lp})

    for line in diff:
        if line.startswith(("--- ", "+++ ")) or line.startswith("@@"):
            continue
        if line[:1] in ("-", "+"):
            state["m" if line[0] == "-" else "p"].append(line)
        else:
            flush()
    flush()
    return {
        "gil_lines": gil,
        "paired_lines": len(pairs),
        "non_mechanical_pairs": bad,
    }


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    b = _b0()
    c = cand_load.cand()
    b0_disp = b._accumulate_od_flow
    cand_disp = c._accumulate_od_flow

    results = {}
    irs = {}
    for tag, disp in (("b0", b0_disp), ("cand", cand_disp)):
        disp.compile(SIG)
        ir = disp.inspect_llvm(SIG)
        assert len(ir) > 50_000, \
            f"{tag}: IR dump is a stub ({len(ir)} bytes) — warm-cache " \
            "load path lost the in-memory LLVM module; capture_ir.py " \
            "must always compile cold (fresh NUMBA_CACHE_DIR)"
        fname = f"ir_{tag}_accumulate_od_flow.ll"
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(ir)
        irs[tag] = ir
        sites = census_sites(ir)
        seq_hash = hashlib.sha256(
            json.dumps(sites, sort_keys=False).encode("utf-8")
        ).hexdigest()
        results[tag] = {
            "file": fname,
            "sha256": sha256(ir),
            "fp_site_count": len(sites),
            "site_sequence_sha256": seq_hash,
            "fma_fmuladd_sites":
                sum(1 for s in sites if s[0].startswith(("fmuladd",
                                                         "call"))),
            "gil": {
                "PyEval_SaveThread": count_calls(ir, "PyEval_SaveThread"),
                "PyEval_RestoreThread":
                    count_calls(ir, "PyEval_RestoreThread"),
            },
        }
        print(f"{fname}: {len(sites)} FP sites, "
              f"{results[tag]['fma_fmuladd_sites']} fma/fmuladd, "
              f"GIL save/restore "
              f"{results[tag]['gil']['PyEval_SaveThread']}/"
              f"{results[tag]['gil']['PyEval_RestoreThread']}")

    # ---- the binding assertions (proof 5b / obligation 9.2) --------
    sites_b = census_sites(irs["b0"])
    sites_c = census_sites(irs["cand"])
    assert results["b0"]["fp_site_count"] == \
        results["cand"]["fp_site_count"], "FP site COUNT delta"
    assert sites_b == sites_c, "FP site SEQUENCE delta (op/flags/order)"
    for tag in ("b0", "cand"):
        assert results[tag]["fma_fmuladd_sites"] == 0, \
            f"{tag}: unexpected fma/fmuladd formation"
    assert results["b0"]["gil"]["PyEval_SaveThread"] == 0, \
        "B0 unexpectedly saves the GIL"
    assert results["cand"]["gil"]["PyEval_SaveThread"] == 2, \
        "candidate GIL save count != 2"
    assert results["cand"]["gil"]["PyEval_RestoreThread"] == 2, \
        "candidate GIL restore count != 2"

    # full-IR residual: everything outside the FP census must be either
    # a GIL declare/call insertion or a mechanical (placeholder-equal)
    # paired line
    residual = classify_residual(irs["b0"], irs["cand"])
    assert len(residual["gil_lines"]) == 4, \
        f"expected exactly 4 GIL lines (2 declares + 2 calls), " \
        f"got {residual['gil_lines']!r}"
    assert any("declare" in l and "SaveThread" in l
               for l in residual["gil_lines"]), "SaveThread declare missing"
    assert any("declare" in l and "RestoreThread" in l
               for l in residual["gil_lines"]), "RestoreThread declare missing"
    assert not residual["non_mechanical_pairs"], \
        ("full-IR residual contains non-mechanical deltas beyond the GIL "
         f"pair: {residual['non_mechanical_pairs'][:4]!r}")
    results["full_ir_residual"] = {
        "gil_lines": residual["gil_lines"],
        "paired_lines": residual["paired_lines"],
        "non_mechanical_pairs": residual["non_mechanical_pairs"],
        "classification": "sole semantic delta = the GIL save/restore "
                          "pair (2 declares + 2 calls around the kernel "
                          "invocation in the CPython wrapper); every "
                          "other diff line pairs with an identical "
                          "renumber-insensitive line (SSA temp numbers, "
                          "block-label numbering, module-name length "
                          "prefix, ABI mangling)",
    }
    print(f"full-IR residual: {len(residual['gil_lines'])} GIL lines, "
          f"{residual['paired_lines']} mechanical pairs, "
          f"{len(residual['non_mechanical_pairs'])} non-mechanical")

    verdict = {
        "specialization": "int32 CSR/edge_id/pred, int8 dir, float64 "
                          "distances/weights/outputs, int64 scalars",
        "verdict": "PASS — counts AND site sequence equal; sole IR delta "
                   "is PyEval_SaveThread/RestoreThread x2 in the "
                   "candidate",
        **results,
    }
    with open(os.path.join(OUT, "ir_census.json"), "w") as fh:
        json.dump(verdict, fh, indent=2)

    with open(os.path.join(OUT, "ir_fp_census.txt"), "w") as fh:
        for tag, title in (("b0", "B0 kernel _accumulate_od_flow"),
                           ("cand", "candidate _accumulate_od_flow "
                                    "(nogil=True)")):
            fh.write(f"==== {title} ====\n")
            for i, site in enumerate(census_sites(irs[tag])):
                fh.write(f"{i:4d}  {site[0]:10s} {site[1]:28s} {site[2]}\n")
            fh.write("\n")

    n_equal = results["b0"]["fp_site_count"]
    with open(os.path.join(OUT, "ir_side_by_side.md"), "w") as fh:
        fh.write(
            "# F1I-N1 IR side-by-side (in-module, observed signature)\n\n"
            f"- B0 IR: `{results['b0']['file']}` "
            f"(sha256 `{results['b0']['sha256'][:16]}…`)\n"
            f"- candidate IR: `{results['cand']['file']}` "
            f"(sha256 `{results['cand']['sha256'][:16]}…`)\n"
            f"- FP sites: {n_equal} in BOTH arms, identical order, "
            "identical fast flags (normalized site sequence hash "
            f"`{results['b0']['site_sequence_sha256'][:16]}…` both arms)\n"
            "- fma/fmuladd/llvm.fma sites: 0 in both arms\n"
            "- GIL handling: B0 0/0 save/restore; candidate 2/2 "
            "(PyEval_SaveThread / PyEval_RestoreThread around the "
            "compiled body)\n"
            "- Full-IR residual: "
            f"{results['full_ir_residual']['paired_lines']} mechanical "
            "pairs (SSA renumbering / label numbering / module-name "
            "length prefix only), "
            f"{len(results['full_ir_residual']['gil_lines'])} GIL lines "
            "(2 declares + 2 calls), 0 non-mechanical deltas\n"
            "- Verdict: **PASS** — the ONLY semantic IR delta is the GIL "
            "save/restore pair; no new FP site, no reassociation, no "
            "fast-flag change (proof 5b; any other delta = rejection)\n")
    print("IR_ARTIFACT_OK")
    print("b0_ir_sha256 =", results["b0"]["sha256"])
    print("cand_ir_sha256 =", results["cand"]["sha256"])
    print("census_sequence_sha256 =", results["b0"]["site_sequence_sha256"])

    shutil.rmtree(_FRESH_CACHE, ignore_errors=True)


if __name__ == "__main__":
    main()
