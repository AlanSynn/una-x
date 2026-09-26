#!/usr/bin/env python3
"""Read-only Git object bridge for the baseline and preserved archive."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

def git(repo: Path, *args: str) -> str:
    p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=False)
    if p.returncode: raise ValueError(p.stderr.strip() or 'git command failed')
    return p.stdout.strip()

def verify(repo: Path, cfg: dict, ref: str='HEAD') -> dict:
    checks=[]
    targets=[('baseline_src',cfg['baseline_commit']+':src',cfg['baseline_src_tree']),
             ('requested_src',ref+':src',cfg['baseline_src_tree']),
             ('historical_archive','HEAD:'+cfg['archive_path'],cfg['archive_tree'])]
    targets += [('blob:'+p,ref+':'+p,sha) for p,sha in cfg.get('source_blobs',{}).items()]
    for name,obj,want in targets:
        try: got=git(repo,'rev-parse','--verify',obj); error=None
        except ValueError as e: got=None; error=str(e)
        checks.append({'check':name,'object':obj,'expected':want,'actual':got,'pass':got==want,'error':error})
    try: head=git(repo,'rev-parse','HEAD'); status=git(repo,'status','--porcelain=v1')
    except ValueError as e: head=None; status=str(e)
    # Object equality does not cover uncommitted runtime edits.
    try:
        runtime_dirty=bool(git(repo,'status','--porcelain=v1','--','src'))
        checks.append({'check':'runtime_worktree_clean','pass':not runtime_dirty})
    except ValueError as e:
        checks.append({'check':'runtime_worktree_clean','pass':False,'error':str(e)})
    return {'ok':all(c['pass'] for c in checks),'head':head,'worktree_status':status,'checks':checks,
            'scope':'initial B0 source/archive bridge, not a candidate correctness test',
            'note':'A selected runtime change is expected to differ from B0; record/review its new identity rather than editing this baseline manifest.'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--ref',default='HEAD')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    cfg=json.loads((Path(__file__).resolve().parents[1]/'campaign.json').read_text())
    result=verify(args.repo.resolve(),cfg,args.ref)
    body=json.dumps(result,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(body)
    print(body,end=''); return 0 if result['ok'] else 1
if __name__=='__main__': sys.exit(main())
