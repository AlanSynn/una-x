"""Strict paired public-API 10x arithmetic gate; NOT a correctness certificate.

Input: pair_samples.json with pairs [{id, A: record, B: record}]. Records must
refer to actual installed/public runs and immutable manifests. Boolean attestations
must be independently audited; this tool cannot observe a remote execution itself.
Bootstrap convention is frozen in BENCHMARKS.md. No invalid samples are dropped.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

TARGET = 10.0
MIN_PAIRS = 7
RESAMPLES = 20000
SEED = 20261003
MATCHED = ('public_api', 'numerical_profile', 'workload_sha256', 'resource_sha256',
           'cache_policy_sha256', 'output_policy_sha256', 'environment_sha256',
           'harness_sha256', 'boundary', 'cache_regime')
DIGESTS = ('wheel_sha256', 'workload_sha256', 'resource_sha256', 'cache_policy_sha256',
           'output_policy_sha256', 'environment_sha256', 'harness_sha256')


def is_digest(value: object, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{' + str(length) + '}', value) is not None


def positive_number(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError('empty values or invalid quantile')
    ordered = sorted(values)
    p = (len(ordered) - 1) * q
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)


def validate_record(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError('record must be object')
    if record.get('status') != 'valid':
        raise ValueError('invalid/failed window cannot enter qualification')
    for key in ('installed', 'bits_verified', 'state_verified', 'artifacts_verified',
                'resource_compliant', 'initial_trace_cost_included'):
        if record.get(key) is not True:
            raise ValueError(f'{key} must have actual true evidence')
    if record.get('boundary') != 'public_call_complete':
        raise ValueError('component/nonpublic boundary refused')
    if record.get('public_api') not in ('UNA.RunBatch', 'UNA.RunAccessibility', 'UNA.RunFlow'):
        raise ValueError('unrecognized public API')
    if record.get('numerical_profile') not in ('una_legacy', 'madina_legacy', 'corrected_v1'):
        raise ValueError('unknown numerical profile')
    if not is_digest(record.get('source_sha'), 40):
        raise ValueError('full source SHA required')
    if any(not is_digest(record.get(key)) for key in DIGESTS):
        raise ValueError('immutable manifest and wheel hashes required')
    if not positive_number(record.get('wall_s')):
        raise ValueError('positive finite wall_s required')
    jobs = record.get('jobs_requested')
    if type(jobs) is not int or jobs < 1 or type(record.get('jobs_validated')) is not int:
        raise ValueError('positive integer job counts required')
    if record.get('jobs_validated') != jobs or type(record.get('jobs_failed')) is not int or record.get('jobs_failed') != 0:
        raise ValueError('only fully completed validated windows qualify')
    if type(record.get('substantive_calls')) is not int or record['substantive_calls'] < 1:
        raise ValueError('work engagement cannot be empty')
    if record.get('cache_regime') not in ('FIRST_COHORT', 'COLD', 'NO_REUSE'):
        raise ValueError('resident/exact-repeat-only wins are not this primary 10x gate')
    if record['public_api'] == 'UNA.RunBatch' and record.get('distinct_jobs') is not True:
        raise ValueError('cohort primary requires distinct requested jobs')


def evaluate(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError('input must be object')
    if data.get('measurement_class') not in ('observed_proxy', 'actual_target'):
        raise ValueError('synthetic/component evidence cannot satisfy application 10x')
    if not is_digest(data.get('primary_sha256')) or not is_digest(data.get('freeze_sha256')):
        raise ValueError('preregistration and freeze digests required')
    pairs = data.get('pairs')
    if not isinstance(pairs, list) or len(pairs) < MIN_PAIRS:
        raise ValueError(f'at least {MIN_PAIRS} independent paired windows required')
    identifiers = set()
    logs = []
    cells = set()
    sources = {'A': set(), 'B': set()}
    wheels = {'A': set(), 'B': set()}
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get('id'), str) or not pair['id']:
            raise ValueError('unique nonempty pair IDs required')
        if pair['id'] in identifiers:
            raise ValueError('duplicate pair ID')
        identifiers.add(pair['id'])
        for arm in ('A', 'B'):
            validate_record(pair.get(arm))
            sources[arm].add(pair[arm]['source_sha'])
            wheels[arm].add(pair[arm]['wheel_sha256'])
        a, b = pair['A'], pair['B']
        if any(a[k] != b[k] for k in MATCHED):
            raise ValueError('unmatched profile/workload/resource/cache/output/environment/harness/boundary')
        if a['jobs_requested'] != b['jobs_requested']:
            raise ValueError('unmatched job count')
        cells.add(tuple(a[k] for k in MATCHED) + (a['jobs_requested'],))
        logs.append(math.log(a['wall_s']) - math.log(b['wall_s']))
    if len(cells) != 1 or any(len(s) != 1 for s in sources.values()) or any(len(s) != 1 for s in wheels.values()):
        raise ValueError('cannot pool unrelated cells or changing candidate/source wheels')
    rng = random.Random(SEED)
    n = len(logs)
    bootstrap = [statistics.median([logs[rng.randrange(n)] for _ in range(n)])
                 for _ in range(RESAMPLES)]
    median_log = statistics.median(logs)
    lower_log = percentile(bootstrap, 0.05)
    median = math.exp(median_log)
    lower = math.exp(lower_log)
    return {'schema_version': 1, 'evidence_class': 'derived_from_measured',
            'method': 'exp(median paired log ratios); linear percentile bootstrap',
            'pair_count': n, 'pair_ratios': [math.exp(x) for x in logs],
            'target': TARGET, 'median_ratio': median, 'one_sided_95_lower': lower,
            'bootstrap_seed': SEED, 'bootstrap_resamples': RESAMPLES,
            'passes_10x_arithmetic': median >= TARGET and lower >= TARGET,
            'measurement_class': data['measurement_class'],
            'primary_sha256': data['primary_sha256'], 'freeze_sha256': data['freeze_sha256'],
            'source_A': next(iter(sources['A'])), 'source_B': next(iter(sources['B'])),
            'note': 'Independent source/bit/state/artifact/resource/review and protected-case '
                    'gates remain required; attestations are not verified by this calculator.'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ns = ap.parse_args()
    blob = ns.samples.read_bytes()
    result = evaluate(json.loads(blob))
    result['samples_sha256'] = hashlib.sha256(blob).hexdigest()
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    with ns.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
    if not result['passes_10x_arithmetic']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
