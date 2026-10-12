"""Verify the pin hash in sparse_proof.md against git rev-parse."""
import re
import subprocess

PIN = subprocess.run(["git", "rev-parse", "395cdc5"],
                     capture_output=True, text=True).stdout.strip()
print("true pin  :", PIN)
proof = open("campaigns/una_10x/evidence/proofs/S10_sparse_projection/"
             "sparse_proof.md").read()
found = re.findall(r"395cdc5[0-9a-f]+", proof)
print("in proof  :", found)
ok = True
for h in found:
    good = (h == PIN)
    ok = ok and good
    print("match:", good, "| len", len(h), "vs", len(PIN))
assert ok and found, "PIN MISMATCH in proof"
print("pin bytes verified")
