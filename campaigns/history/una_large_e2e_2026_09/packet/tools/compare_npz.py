#!/usr/bin/env python3
"""Exact deterministic small-fixture NPZ comparison; no pickle or tolerances."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

def compare(a: Path,b: Path) -> dict:
    differences=[]
    with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
        if not left.files or not right.files:
            differences.append({'kind':'empty_archive_not_evidence'})
        if set(left.files)!=set(right.files):
            differences.append({'kind':'keys','left':sorted(left.files),'right':sorted(right.files)})
        for key in sorted(set(left.files)&set(right.files)):
            x,y=left[key],right[key]
            if x.dtype!=y.dtype or x.shape!=y.shape:
                differences.append({'key':key,'kind':'dtype_or_shape','left':[str(x.dtype),list(x.shape)],'right':[str(y.dtype),list(y.shape)]}); continue
            xb=np.ascontiguousarray(x).reshape(-1).view(np.uint8)
            yb=np.ascontiguousarray(y).reshape(-1).view(np.uint8)
            for start in range(0,xb.size,1024*1024):
                mismatch=np.flatnonzero(xb[start:start+1024*1024]!=yb[start:start+1024*1024])
                if mismatch.size:
                    differences.append({'key':key,'kind':'bits','first_byte_offset':start+int(mismatch[0])}); break
    return {'ok':not differences,'differences':differences,'profile':'dtype_shape_exact_bits_in_C_element_order'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('baseline',type=Path); ap.add_argument('candidate',type=Path)
    args=ap.parse_args()
    result=compare(args.baseline,args.candidate)
    print(json.dumps(result,indent=2)); return 0 if result['ok'] else 1
if __name__=='__main__': sys.exit(main())
