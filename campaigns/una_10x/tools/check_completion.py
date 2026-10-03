"""Conservative completion-record shape/custody checks. Not independent review."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

PASS_STATES = {'observed_cohort_10x', 'observed_single_10x', 'observed_both_10x', 'actual_target_10x'}
CHECKS = ('source_bridge', 'installed', 'bitparity', 'state_artifact', 'protected_cases',
          'resource_bounds', 'independent_review')
CAPABILITIES = ('madina_api', 'scientific_fixes', 'native', 'gpu', 'public_batch',
                'cache', 'recovery', 'installed_validation')


def check_record(data: dict, root: Path) -> list[str]:
    errors = []
    if data.get('tenx_status') not in PASS_STATES | {'improvement_only', 'target_unmet', 'blocked_external'}:
        errors.append('unknown tenx_status')
    if data.get('tenx_status') in PASS_STATES:
        if not data.get('claim_scope'):
            errors.append('explicit claim scope required')
        checks = data.get('required_checks', {})
        for name in CHECKS:
            record = checks.get(name, {})
            if record.get('status') != 'pass' or not record.get('evidence'):
                errors.append(f'missing actual gate: {name}')
        gate = data.get('gate_record', {})
        if gate.get('passes_10x_arithmetic') is not True or gate.get('pair_count', 0) < 7:
            errors.append('paired 10x arithmetic has not passed')
        for field in ('median_ratio', 'one_sided_95_lower'):
            value = gate.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 10:
                errors.append(f'10x metric absent/below threshold: {field}')
        if data.get('tenx_status') == 'observed_both_10x':
            additional = data.get('secondary_gate_record', {})
            if (additional.get('passes_10x_arithmetic') is not True
                    or additional.get('pair_count', 0) < 7
                    or not additional.get('samples_sha256')
                    or additional.get('primary_sha256') == gate.get('primary_sha256')):
                errors.append('both scopes require independent single and cohort gate records')
        if not gate.get('samples_sha256'):
            errors.append('raw sample digest missing')
        if data.get('tenx_status') == 'actual_target_10x' and gate.get('measurement_class') != 'actual_target':
            errors.append('proxy is not actual target')
    records = data.get('evidence_files', [])
    if data.get('tenx_status') in PASS_STATES and not records:
        errors.append('retained raw/evidence files required, not hashes alone')
    root = root.resolve()
    for record in records:
        rel = record.get('path', '')
        path = root / rel
        if not rel or Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink():
            errors.append(f'unsafe evidence path: {rel}')
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file() or hashlib.sha256(resolved.read_bytes()).hexdigest() != record.get('sha256'):
                errors.append(f'missing/corrupt evidence: {rel}')
        except (OSError, ValueError):
            errors.append(f'unavailable evidence: {rel}')
    if data.get('platform_status') == 'qualified':
        caps = data.get('platform_capabilities', {})
        for name in CAPABILITIES:
            item = caps.get(name, {})
            if item.get('status') != 'qualified' or not item.get('evidence'):
                errors.append(f'platform capability incomplete: {name}')
        for name in ('native', 'gpu'):
            item = caps.get(name, {})
            if item.get('substantive_installed_calls', 0) < 1 or item.get('simulated') is not False:
                errors.append(f'{name} substantive execution unproven')
        if caps.get('gpu', {}).get('real_hardware') is not True:
            errors.append('GPU real hardware not qualified')
    if data.get('merge_disposition') == 'merge_ready' and data.get('ci_status') != 'passed':
        errors.append('merge_ready cannot use unavailable CI')
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--decision', type=Path, required=True)
    ap.add_argument('--evidence-root', type=Path, required=True)
    ns = ap.parse_args()
    errors = check_record(json.loads(ns.decision.read_text()), ns.evidence_root)
    print(json.dumps({'ok': not errors, 'errors': errors,
                      'scope': 'record/custody validation; independent review still required'}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
