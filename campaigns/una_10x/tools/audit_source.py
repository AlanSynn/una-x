"""Read-only Git provenance audit. Writes only a NEW explicitly requested report."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                          text=True, timeout=30, check=False)
    if proc.returncode:
        raise RuntimeError(f'git {args!r}: {proc.stderr.strip()}')
    return proc.stdout.strip()


def audit(repo: Path, pins: dict) -> dict:
    repo = repo.resolve(strict=True)
    head = run_git(repo, 'rev-parse', 'HEAD')
    status = run_git(repo, 'status', '--porcelain=v1', '--untracked-files=normal')
    checks = []
    for name, expected in pins['trees'].items():
        actual = run_git(repo, 'rev-parse', f'HEAD:{name}')
        checks.append({'path': name, 'expected_git_tree': expected,
                       'actual_git_tree': actual, 'match': actual == expected})
    for rel, expected in pins['files'].items():
        path = repo / rel
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'missing/nonregular source: {rel}')
        data = path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        checks.append({'path': rel, 'expected_git_blob': expected,
                       'actual_git_blob': actual, 'match': actual == expected})
    return {'head': head, 'pinned_source_sha': pins['source_sha'],
            'source_matches': all(c['match'] for c in checks),
            'clean': not status, 'status_porcelain': status, 'checks': checks,
            'python': platform.python_version(),
            'note': 'Identity only, not numerical/API/performance qualification. '
                    'Dirty or drifted source requires explicit reconciliation.'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ns = ap.parse_args()
    pins = json.loads((Path(__file__).resolve().parents[1] / 'source_pins.json').read_text())
    result = audit(ns.repo, pins)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    with ns.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(ns.out)
    if not result['source_matches'] or not result['clean']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
