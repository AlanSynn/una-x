"""Offline packet DAG, source-pin, required-file and historical-byte validator."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

REQUIRED = ['README.md', 'START_HERE.txt', 'AUTHORITY.md', 'PLATFORM_CONTINUITY.md',
            'SOURCE_AUDIT.md', 'CONTRACT.md', 'MODEL.md', 'EXECUTION.md', 'RESOURCES.md',
            'WORKLOADS.md', 'BENCHMARKS.md', 'VALIDATION.md', 'DECISION.md',
            'TASKS_CLAUDE.yaml', 'campaign.json', 'source_pins.json', 'inputs_catalog.json',
            'tools/audit_source.py', 'tools/gate_10x.py', 'tools/check_completion.py',
            'tests/test_tools.py', 'prior/una_10x_investigation/manifest.json']


def validate(root: Path) -> dict:
    checks = []
    def check(name: str, ok: bool) -> None:
        checks.append({'check': name, 'pass': bool(ok)})
    for rel in REQUIRED:
        check(f'file:{rel}', (root / rel).is_file())
    if not all(c['pass'] for c in checks):
        return {'ok': False, 'checks': checks}
    campaign = json.loads((root / 'campaign.json').read_text())
    check('target remains 10', campaign.get('target_speedup') == 10.0)
    check('auto publication forbidden', campaign.get('auto_publish') is False)
    check('platform retained incomplete', campaign.get('platform_status') == 'retained_incomplete')
    check('numerical bitwise', campaign.get('numerical_requirement') == 'bitwise_per_profile')
    tasks = json.loads((root / 'TASKS_CLAUDE.yaml').read_text())['tasks']
    ids = [t['id'] for t in tasks]
    check('task IDs unique', len(ids) == len(set(ids)))
    by_id = {t['id']: t for t in tasks}
    for task in tasks:
        name = task['id']
        check(f'{name}:deps exist', all(d in by_id for d in task['depends_on']))
        check(f'{name}:required subset', set(task['required_pass']) <= set(task['depends_on']))
        for field in ('owner', 'reads', 'writes', 'actions', 'deliverables', 'completion'):
            check(f'{name}:{field}', bool(task.get(field)))
        for rel in task['reads']:
            # The retained platform packet is validated as an explicit cross-campaign dependency.
            if not rel.startswith('../'):
                check(f'{name}:read:{rel}', (root / rel).is_file())
    visiting, done = set(), set()
    def walk(name: str) -> None:
        if name in visiting:
            raise ValueError('cycle')
        if name in done:
            return
        visiting.add(name)
        for dep in by_id[name]['depends_on']:
            walk(dep)
        visiting.remove(name)
        done.add(name)
    try:
        for name in ids:
            walk(name)
        check('DAG acyclic', True)
    except (KeyError, ValueError):
        check('DAG acyclic', False)
    for required_id in ('S02', 'W00', 'S11', 'C11', 'C12', 'C30', 'N00', 'G10', 'X00', 'Q10', 'R00'):
        check(f'mandatory task:{required_id}', required_id in by_id)
    prior = root / 'prior/una_10x_investigation'
    manifest = json.loads((prior / 'manifest.json').read_text())
    actual = {str(p.relative_to(prior)) for p in prior.rglob('*') if p.is_file()}
    check('prior exact inventory', actual == set(manifest) | {'manifest.json'})
    for rel, expected in manifest.items():
        path = prior / rel
        check(f'prior hash:{rel}', path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    for directory in ('tools', 'tests'):
        for path in (root / directory).glob('*.py'):
            try:
                ast.parse(path.read_text())
                check(f'python syntax:{path.name}', True)
            except SyntaxError:
                check(f'python syntax:{path.name}', False)
    authored = root / 'packet_files.json'
    if authored.is_file():
        inventory = json.loads(authored.read_text())
        for rel, expected in inventory['sha256'].items():
            path = root / rel
            check(f'packet hash:{rel}', path.is_file() and
                  hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    return {'ok': all(c['pass'] for c in checks), 'check_count': len(checks),
            'task_count': len(tasks), 'dossier_count': len(list((root / 'dossiers').glob('*.md'))),
            'scope': 'packet consistency only, not application correctness/performance',
            'checks': checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ns = ap.parse_args()
    result = validate(ns.root)
    print(json.dumps(result, indent=2))
    if not result['ok']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
