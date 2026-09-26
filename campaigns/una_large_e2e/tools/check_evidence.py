#!/usr/bin/env python3
"""Check a task report's identities and artifact integrity, not its science."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

SHA1=re.compile(r'^[0-9a-f]{40}$')
SHA256=re.compile(r'^[0-9a-f]{64}$')
TERMINAL={'complete','rejected','not_admitted','blocked'}

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def check(record: dict, task: dict, workspace: Path) -> dict:
    errors=[]; workspace=workspace.resolve()
    if record.get('task_id')!=task['id']: errors.append('task id mismatch')
    if record.get('status') not in TERMINAL: errors.append('status is not terminal')
    if not SHA1.fullmatch(record.get('source_commit') or ''): errors.append('full source commit required')
    if not record.get('actor'): errors.append('actor required')
    if record.get('status')!='complete' and not record.get('reason'): errors.append('negative state requires reason')
    commands=record.get('commands',[])
    if record.get('status')=='complete' and not commands: errors.append('complete task requires commands')
    for c in commands:
        if not isinstance(c.get('argv'),list) or not c['argv']: errors.append('command argv required')
        actual,expected=c.get('returncode'),c.get('expected_returncode',0)
        if type(actual) is not int or type(expected) is not int or actual!=expected:
            errors.append('unexpected or missing command exit status')
    artifacts=record.get('artifacts',[])
    if not artifacts: errors.append('at least one supporting artifact required')
    seen=set()
    for a in artifacts:
        name=a.get('path',''); p=(workspace/name).resolve()
        if not name or Path(name).is_absolute() or not p.is_relative_to(workspace):
            errors.append('artifact path escapes workspace'); continue
        if name in seen: errors.append('duplicate artifact: '+name)
        seen.add(name)
        if not SHA256.fullmatch(a.get('sha256') or ''):
            errors.append('artifact SHA256 required: '+name); continue
        if not p.is_file() or digest(p)!=a['sha256']: errors.append('artifact missing or changed: '+name)
    if task.get('review_task') and record.get('status')=='complete':
        author=record.get('implementation_actor')
        if not author or author==record.get('actor'): errors.append('independent implementation actor required')
        if record.get('review_subject_commit')!=record.get('source_commit'): errors.append('review source mismatch')
        if record.get('independent_review_available') is not True: errors.append('independent review unavailable')
    return {'ok':not errors,'errors':errors,'scope':'record integrity only; does not prove commands ran or review was independent'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('record',type=Path); ap.add_argument('--workspace',type=Path,required=True)
    args=ap.parse_args()
    dag=json.loads((Path(__file__).resolve().parents[1]/'TASKS_CLAUDE.yaml').read_text())
    rec=json.loads(args.record.read_text()); tasks={t['id']:t for t in dag['tasks']}
    if rec.get('task_id') not in tasks:
        result={'ok':False,'errors':['unknown task id']}
    else: result=check(rec,tasks[rec['task_id']],args.workspace)
    print(json.dumps(result,indent=2)); return 0 if result['ok'] else 1
if __name__=='__main__': sys.exit(main())
