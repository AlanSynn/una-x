"""Offline post-P00 amendment validator. Not an implementation acceptance gate."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

REQUIRED_TASKS = ('P01M','P01F','P02D','P02I','P02R','P03C','P03M','P03R',
                  'P04I','P04R','P05S','P05R')


def validate(root: Path) -> dict:
    checks = []
    def check(name, ok):
        checks.append({'check': name, 'pass': bool(ok)})
    try:
        cfg = json.loads((root/'rollout.json').read_text())
        sources = json.loads((root/'sources.json').read_text())
        tasks = json.loads((root.parent/'TASKS_CLAUDE.yaml').read_text())['tasks']
        by_id = {t['id']:t for t in tasks}
        check('IDs unique', len(by_id) == len(tasks))
        for ident in REQUIRED_TASKS:
            check('task:'+ident, ident in by_id)
        check('I00 waits for post-P00 review', 'P05R' in by_id['I00']['depends_on'])
        check('I00 requires real review gate', 'P05R' in by_id['I00']['required_pass'])
        for task in tasks:
            check(task['id']+':valid dependencies', all(x in by_id for x in task['depends_on']))
            check(task['id']+':required subset', set(task['required_pass']) <= set(task['depends_on']))
            for f in task['reads']:
                if not f.startswith('../'):
                    check(task['id']+':read:'+f, (root.parent/f).is_file())
        active, done = set(), set()
        def walk(i):
            if i in active: raise ValueError('dependency cycle')
            if i in done: return
            active.add(i)
            for d in by_id[i]['depends_on']: walk(d)
            active.remove(i); done.add(i)
        for i in by_id: walk(i)
        check('DAG acyclic', True)
        check('correct branch', cfg['branch'] == 'perf/una-10x')
        check('10x threshold unchanged', cfg['target_speedup'] == 10)
        check('future publication forbidden', cfg['auto_publish'] is False)
        check('performance not preclaimed', cfg['application_10x'] == 'not_measured_by_this_amendment')
        check('primary stays frozen', cfg['modify_primary'] is False)
        check('original budget not reset', cfg['reset_heavy_budget'] is False)
        check('runtime not shipped in amendment', cfg['runtime_changes_in_publication'] is False)
        check('correct checkpoint', sources['checkpoint_commit'] == '6130769a5751bf7446734dc39b9f8e3763461cfe')
        for p in list((root/'tools').glob('*.py'))+list((root/'tests').glob('*.py')):
            ast.parse(p.read_text()); check('syntax:'+p.name, True)
        manifest = json.loads((root/'manifest.json').read_text())
        repo = root.parents[2]
        for f, h in manifest['files_sha256'].items():
            p = repo/f
            check('authored:'+f, p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h)
        for f,h in manifest['protected_sha256'].items():
            p = root.parent/f
            check('frozen:'+f, p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h)
        for f in cfg['required_docs']:
            check('design:'+f,(root/f).is_file())
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as exc:
        check('validation exception:'+str(exc), False)
    return {'ok': all(x['pass'] for x in checks), 'check_count': len(checks),
            'scope':'packet structure/design utilities only; application qualification unavailable',
            'checks':checks}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ns=ap.parse_args()
    result=validate(ns.root.resolve())
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['ok'] else 2)


if __name__=='__main__': main()
