#!/usr/bin/env python3
"""Validate this packet's static contract. Does not qualify UNA performance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

REQUIRED = {
    'README.md', 'START_HERE.txt', 'EXECUTION.md', 'AUTHORITY.md',
    'SOURCE_AUDIT.md', 'CONTRACT.md', 'MODEL.md', 'OPERATIONS.md',
    'WORKLOADS.md', 'HARNESS.md', 'VALIDATION.md', 'BENCHMARKS.md',
    'RESOURCES.md', 'DECISION.md', 'campaign.json', 'inputs_catalog.json',
    'TASKS_CLAUDE.yaml', 'tools/verify_source.py', 'tools/check_evidence.py',
    'tools/compare_npz.py', 'tests/test_packet.py',
}
FIELDS = {'id', 'title', 'role', 'depends_on', 'base_commit', 'owned_files',
          'read_context', 'objective', 'constraints', 'proof_obligations',
          'validation_tier', 'resource_budget', 'outputs', 'reviewer',
          'completion_condition', 'required', 'review_task', 'experiment'}
SHA1 = re.compile(r'^[0-9a-f]{40}$')

def safe_path(root: Path, name: str) -> Path:
    p = (root / name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes root: {name}')
    return p

def overlap(a: str, b: str) -> bool:
    # Conservative ownership test: wildcard suffixes claim their directory.
    def prefix(s):
        first = min([s.find(c) for c in '*?[' if c in s] or [len(s)])
        head = s[:first]
        if first < len(s) and not head.endswith('/'):
            head = head.rsplit('/', 1)[0] if '/' in head else ''
        return head.rstrip('/')
    x, y = prefix(a), prefix(b)
    return not x or not y or x == y or x.startswith(y + '/') or y.startswith(x + '/')

def validate(root: Path, verify_hashes: bool = True) -> dict:
    root = root.resolve()
    errors, checks = [], []
    def check(name, ok, detail=''):
        checks.append({'check': name, 'pass': bool(ok), 'detail': detail})
        if not ok:
            errors.append(name + (': ' + detail if detail else ''))
    for f in sorted(REQUIRED):
        check('file:' + f, safe_path(root, f).is_file())
    try:
        cfg = json.loads((root / 'campaign.json').read_text())
        dag = json.loads((root / 'TASKS_CLAUDE.yaml').read_text())
        cat = json.loads((root / 'inputs_catalog.json').read_text())
    except (OSError, ValueError) as e:
        return {'ok': False, 'errors': errors + [str(e)], 'checks': checks}
    check('baseline_sha', isinstance(cfg.get('baseline_commit'), str) and bool(SHA1.fullmatch(cfg['baseline_commit'])))
    check('src_tree_sha', isinstance(cfg.get('baseline_src_tree'), str) and bool(SHA1.fullmatch(cfg['baseline_src_tree'])))
    check('archive_tree_sha', isinstance(cfg.get('archive_tree'), str) and bool(SHA1.fullmatch(cfg['archive_tree'])))
    check('no_auto_publish', cfg.get('auto_publish') is False)
    check('cpu_only', cfg.get('gpu_enabled') is False)
    check('exact_portfolio', set(cfg.get('portfolio', [])) == {'A1','A2','A3','F1','F2','F3'})
    check('branch', cfg.get('integration_branch') == 'perf/una-large-e2e')
    check('dag_identity', all(dag.get(k) == cfg.get(k) for k in ('campaign_id','repository','integration_branch','baseline_commit')))
    check('start_size', 2000 <= len((root / 'START_HERE.txt').read_bytes()) <= 5000)
    for f in cat.get('files', []):
        check('input:' + str(f.get('id')), bool(SHA1.fullmatch(f.get('git_blob_sha',''))) and
              isinstance(f.get('bytes'), int) and f['bytes'] > 0 and
              ('/' + cat.get('ref','?') + '/') in f.get('url','') and
              f.get('url','').startswith('https://raw.githubusercontent.com/'))
    seq = dag.get('tasks', [])
    if not isinstance(seq, list):
        return {'ok': False, 'errors': errors + ['tasks is not a list'], 'checks': checks}
    ids = [t.get('id') for t in seq if isinstance(t, dict)]
    check('unique_task_ids', len(seq) == len(ids) == len(set(ids)))
    tasks = {t['id']:t for t in seq if isinstance(t, dict) and isinstance(t.get('id'), str)}
    for ident,t in tasks.items():
        check(ident + ':fields', FIELDS <= set(t), str(sorted(FIELDS - set(t))))
        check(ident + ':dependencies', all(d in tasks for d in t.get('depends_on', [])))
        check(ident + ':proof', bool(t.get('proof_obligations')))
        check(ident + ':outputs', bool(t.get('outputs')))
        check(ident + ':reviewer', bool(t.get('reviewer')))
        check(ident + ':base', t.get('base_commit',{}).get('baseline') == cfg.get('baseline_commit') and
              t.get('base_commit',{}).get('resolve_to_full_sha_before_start') is True)
        for p in t.get('read_context', []):
            try: exists = safe_path(root, p).is_file()
            except ValueError: exists = False
            check(ident + ':context:' + p, exists)
    visiting, done, ancestors = set(), set(), {}
    def visit(ident):
        if ident in visiting: raise ValueError('dependency cycle at ' + ident)
        if ident in done: return ancestors[ident]
        visiting.add(ident); acc = set()
        for d in tasks[ident].get('depends_on', []):
            if d not in tasks: continue
            acc.add(d); acc.update(visit(d))
        visiting.remove(ident); done.add(ident); ancestors[ident] = acc
        return acc
    try:
        for ident in tasks: visit(ident)
        check('acyclic', True)
        conflicts=[]
        for i,a in enumerate(tasks):
            for b in list(tasks)[i+1:]:
                if a in ancestors[b] or b in ancestors[a]: continue
                for x in tasks[a].get('owned_files', []):
                    for y in tasks[b].get('owned_files', []):
                        if overlap(x,y): conflicts.append([a,b,x,y])
        check('no_concurrent_scope_conflicts', not conflicts, str(conflicts))
    except ValueError as e:
        check('acyclic', False, str(e))
    repo = root.parents[1]
    try:
        reg = json.loads((repo/'campaigns/REGISTRY.json').read_text())
        check('registry_active', reg.get('active') == 'campaigns/una_large_e2e')
        for p in ('CLAUDE.md','.claude/commands/goal.md'):
            text = (repo/p).read_text()
            check('entrypoint:' + p, 'campaigns/una_large_e2e' in text and 'perf/una-large-e2e' in text)
    except (OSError,ValueError) as e:
        check('routing',False,str(e))
    if verify_hashes:
        try:
            manifest = json.loads((root/'packet_files.json').read_text())
            for name,want in manifest['sha256'].items():
                p=safe_path(root,name)
                check('hash:' + name, p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == want)
        except (OSError,ValueError,KeyError) as e:
            check('file_manifest',False,str(e))
    return {'ok': not errors, 'task_count': len(tasks), 'checks': checks,
            'errors': errors, 'scope': 'packet consistency only; no application qualification'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--packet',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    result=validate(args.packet)
    body=json.dumps(result,indent=2)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(body)
    print(body,end='')
    return 0 if result['ok'] else 1
if __name__=='__main__': sys.exit(main())
