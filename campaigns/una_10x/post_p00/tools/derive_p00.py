"""Recompute only supported arithmetic from local P00 JSON. No benchmark is run."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def positive(x, name):
    if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(x)


def derive(model: dict, profile: dict | None = None) -> dict:
    a = positive(model['observed']['baseline_arm']['par_w4_s'], 'baseline wall')
    b = positive(model['observed']['candidate']['w4_c1']['wall_s'], 'candidate wall')
    result = {'scope': 'arithmetic on recorded samples, not a new speed measurement',
              'evidence_class': 'derived_from_measured_values',
              'recorded_A_wall_s': a, 'recorded_B_wall_s': b, 'A_over_B': a/b,
              'historical_sample_budget_10x_s': a/10,
              'planning_headroom_12x_s': a/12,
              'fraction_of_B_to_remove_for_10x': 1-a/(10*b),
              'new_application_measurement': 'unavailable',
              'scenarios': []}
    for front in (4, 6, 8):
        result['scenarios'].append({'front_s': front, 'remaining_budget_s': a/10-front,
                                   'evidence_class': 'hypothetical_sensitivity_analysis'})
    if profile is not None:
        funcs = profile['profile']['top_functions_by_tottime']
        self_time = {r['func']: float(r['tottime_s']) for r in funcs}
        keys = ['raw.py:read_arrow', '~:<built-in method builtins.next>', 'io.py:from_wkb']
        group = sum(self_time[k] for k in keys)
        total = positive(profile['profile']['buckets']['other']['cumtime_max_s'], 'profile total')
        replay = self_time['AccessibilityWElevation.py:integrated_scope_access']
        result['profile_arithmetic'] = {
            'grouped_decode_self_s': group, 'profile_denominator_s': total,
            'grouped_decode_share': group/total, 'integrated_self_s': replay,
            'integrated_share': replay/total,
            'qualification': 'diagnostic; verify next caller attribution; not parallel-span shares'}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p00-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ns = ap.parse_args()
    paths = [ns.p00_dir/'stage_model.json', ns.p00_dir/'runs/prof_cand_serial.json']
    data = [json.loads(p.read_text()) for p in paths]
    result = derive(*data)
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if ns.out:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        with ns.out.open('x', encoding='utf-8') as f:  # never overwrite historic evidence
            f.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
