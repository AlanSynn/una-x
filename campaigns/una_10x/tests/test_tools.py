"""Packet utility self-tests. Synthetic records here are NOT performance evidence."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import audit_source
import check_completion
import gate_10x
import validate_packet


def samples(ratio=12.0):
    record = {'status': 'valid', 'installed': True, 'bits_verified': True,
              'state_verified': True, 'artifacts_verified': True,
              'resource_compliant': True, 'initial_trace_cost_included': True,
              'boundary': 'public_call_complete', 'public_api': 'UNA.RunBatch',
              'numerical_profile': 'una_legacy', 'source_sha': 'a'*40,
              'wall_s': 120.0, 'jobs_requested': 32, 'jobs_validated': 32,
              'jobs_failed': 0, 'substantive_calls': 32,
              'cache_regime': 'FIRST_COHORT', 'distinct_jobs': True}
    record.update({key: 'a'*64 for key in gate_10x.DIGESTS})
    pairs = []
    for i in range(7):
        a, b = copy.deepcopy(record), copy.deepcopy(record)
        b['source_sha'] = 'b'*40
        b['wheel_sha256'] = 'b'*64
        b['wall_s'] = 120.0 / ratio
        pairs.append({'id': f'p{i}', 'A': a, 'B': b})
    return {'schema_version': 1, 'measurement_class': 'observed_proxy',
            'primary_sha256': 'c'*64, 'freeze_sha256': 'd'*64, 'pairs': pairs}


class GateTests(unittest.TestCase):
    def reject_change(self, key, value):
        data = samples()
        data['pairs'][0]['B'][key] = value
        with self.assertRaises(ValueError): gate_10x.evaluate(data)

    def test_over_target(self):
        out = gate_10x.evaluate(samples())
        self.assertTrue(out['passes_10x_arithmetic'])
        self.assertAlmostEqual(out['median_ratio'], 12.0)

    def test_below_target(self):
        self.assertFalse(gate_10x.evaluate(samples(9.5))['passes_10x_arithmetic'])

    def test_lower_bound_not_median_alone(self):
        data = samples()
        for i, ratio in enumerate([8., 8., 9., 12., 13., 14., 14.]):
            data['pairs'][i]['B']['wall_s'] = 120. / ratio
        out = gate_10x.evaluate(data)
        self.assertGreater(out['median_ratio'], 10.)
        self.assertFalse(out['passes_10x_arithmetic'])

    def test_seed_reproducible(self):
        self.assertEqual(gate_10x.evaluate(samples()), gate_10x.evaluate(samples()))

    def test_too_few_pairs(self):
        d = samples(); d['pairs'].pop()
        with self.assertRaises(ValueError): gate_10x.evaluate(d)

    def test_duplicate_pairs(self):
        d = samples(); d['pairs'][1]['id'] = 'p0'
        with self.assertRaises(ValueError): gate_10x.evaluate(d)

    def test_synthetic_refused(self):
        d = samples(); d['measurement_class'] = 'synthetic'
        with self.assertRaises(ValueError): gate_10x.evaluate(d)

    def test_unfrozen_refused(self):
        d = samples(); d['freeze_sha256'] = None
        with self.assertRaises(ValueError): gate_10x.evaluate(d)

    def test_kernel_boundary(self): self.reject_change('boundary', 'kernel')
    def test_wrong_api(self): self.reject_change('public_api', 'external_pool')
    def test_no_installed(self): self.reject_change('installed', False)
    def test_no_bits(self): self.reject_change('bits_verified', False)
    def test_no_state(self): self.reject_change('state_verified', False)
    def test_no_outputs(self): self.reject_change('artifacts_verified', False)
    def test_no_resource(self): self.reject_change('resource_compliant', False)
    def test_untimed_trace(self): self.reject_change('initial_trace_cost_included', False)
    def test_failed_jobs(self): self.reject_change('jobs_failed', 1)
    def test_partial_jobs(self): self.reject_change('jobs_validated', 31)
    def test_unknown_source(self): self.reject_change('source_sha', 'main')
    def test_empty_engagement(self): self.reject_change('substantive_calls', 0)
    def test_warm_identical_only(self): self.reject_change('cache_regime', 'REPEAT_IDENTICAL')
    def test_fake_distinct(self): self.reject_change('distinct_jobs', False)
    def test_nan_time(self): self.reject_change('wall_s', float('nan'))
    def test_zero_time(self): self.reject_change('wall_s', 0.)
    def test_bool_time(self): self.reject_change('wall_s', True)
    def test_mixed_resources(self): self.reject_change('resource_sha256', 'b'*64)
    def test_mixed_profiles(self): self.reject_change('numerical_profile', 'corrected_v1')
    def test_mixed_harness(self): self.reject_change('harness_sha256', 'b'*64)
    def test_mutating_wheel(self): self.reject_change('wheel_sha256', 'f'*64)

    def test_quantile(self):
        self.assertEqual(gate_10x.percentile([1., 3.], .5), 2.)
        with self.assertRaises(ValueError): gate_10x.percentile([], .05)

    def test_no_cli_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); (p/'samples.json').write_text(json.dumps(samples()))
            (p/'out.json').write_text('retained')
            r = subprocess.run([sys.executable, str(ROOT/'tools/gate_10x.py'),
                                '--samples', str(p/'samples.json'), '--out', str(p/'out.json')],
                               capture_output=True, timeout=15)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual((p/'out.json').read_text(), 'retained')


class PacketTests(unittest.TestCase):
    def test_current_packet(self):
        result = validate_packet.validate(ROOT)
        self.assertTrue(result['ok'], [x for x in result['checks'] if not x['pass']])

    def test_target_mutation_refused(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'packet'; shutil.copytree(ROOT,r)
            p=r/'campaign.json'; d=json.loads(p.read_text()); d['target_speedup']=1.1
            p.write_text(json.dumps(d))
            self.assertFalse(validate_packet.validate(r)['ok'])

    def test_cycle_refused(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td)/'packet'; shutil.copytree(ROOT,r)
            p=r/'TASKS_CLAUDE.yaml'; d=json.loads(p.read_text())
            d['tasks'][0]['depends_on']=['Z00']; p.write_text(json.dumps(d))
            self.assertFalse(validate_packet.validate(r)['ok'])

    def test_prior_manifest(self):
        prior = ROOT/'prior/una_10x_investigation'
        manifest = json.loads((prior/'manifest.json').read_text())
        self.assertEqual(len(manifest), 27)
        for rel, want in manifest.items():
            self.assertEqual(hashlib.sha256((prior/rel).read_bytes()).hexdigest(), want)

    def test_required_platform_tasks(self):
        tasks = json.loads((ROOT/'TASKS_CLAUDE.yaml').read_text())['tasks']
        self.assertTrue({'G10', 'N00', 'C30', 'X00'} <= {t['id'] for t in tasks})

    def test_historical_claim_boundary(self):
        readme = (ROOT/'prior/una_10x_investigation/README.md').read_text()
        self.assertIn('NOT the\ninstalled current UNA package', readme)


class CompletionTests(unittest.TestCase):
    def test_missing_gates_refused(self):
        with tempfile.TemporaryDirectory() as td:
            errors = check_completion.check_record({'tenx_status': 'observed_cohort_10x'}, Path(td))
            self.assertGreater(len(errors), 5)

    def test_true_flag_without_ratio_refused(self):
        d={'tenx_status':'observed_cohort_10x','gate_record':
           {'passes_10x_arithmetic':True,'pair_count':7,'samples_sha256':'x'}}
        errors=check_completion.check_record(d,ROOT)
        self.assertIn('10x metric absent/below threshold: median_ratio',errors)

    def test_both_scopes_require_two_gates(self):
        d={'tenx_status':'observed_both_10x','gate_record':{}}
        self.assertIn('both scopes require independent single and cohort gate records',
                      check_completion.check_record(d,ROOT))

    def test_target_unmet_honest(self):
        self.assertEqual(check_completion.check_record({'tenx_status': 'target_unmet',
                          'platform_status': 'incomplete'}, ROOT), [])

    def test_platform_fake_gpu(self):
        data = {'tenx_status':'target_unmet','platform_status':'qualified','platform_capabilities':{}}
        for name in check_completion.CAPABILITIES:
            data['platform_capabilities'][name]={'status':'qualified','evidence':['receipt'],
                                                'substantive_installed_calls':1,'simulated':False}
        data['platform_capabilities']['gpu']['real_hardware'] = False
        self.assertIn('GPU real hardware not qualified', check_completion.check_record(data, ROOT))

    def test_proxy_not_production(self):
        d={'tenx_status':'actual_target_10x','gate_record':{'measurement_class':'observed_proxy'}}
        self.assertIn('proxy is not actual target', check_completion.check_record(d, ROOT))

    def test_raw_hash_custody(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root/'raw').write_bytes(b'raw')
            d={'tenx_status':'target_unmet','evidence_files':[{'path':'raw','sha256':'x'}]}
            self.assertTrue(check_completion.check_record(d,root))
            d['evidence_files'][0]['sha256']=hashlib.sha256(b'raw').hexdigest()
            self.assertEqual(check_completion.check_record(d,root),[])

    def test_path_traversal(self):
        d={'tenx_status':'target_unmet','evidence_files':[{'path':'../secret','sha256':'x'}]}
        self.assertTrue(check_completion.check_record(d,ROOT))

    def test_no_merge_ready_without_ci(self):
        d={'tenx_status':'target_unmet','merge_disposition':'merge_ready','ci_status':'unavailable'}
        self.assertTrue(check_completion.check_record(d,ROOT))


class GitAuditTests(unittest.TestCase):
    def test_audit_real_git_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'config','user.name','Packet Test'],check=True)
            subprocess.run(['git','-C',str(root),'config','user.email','packet@example.invalid'],check=True)
            (root/'src').mkdir(); (root/'src/a.py').write_bytes(b'value = 1\n')
            subprocess.run(['git','-C',str(root),'add','src'],check=True)
            subprocess.run(['git','-C',str(root),'commit','-qm','fixture'],check=True)
            pins={'source_sha':audit_source.run_git(root,'rev-parse','HEAD'),
                  'trees':{'src':audit_source.run_git(root,'rev-parse','HEAD:src')},
                  'files':{'src/a.py':audit_source.run_git(root,'rev-parse','HEAD:src/a.py')}}
            good=audit_source.audit(root,pins)
            self.assertTrue(good['source_matches']); self.assertTrue(good['clean'])
            (root/'src/a.py').write_text('value = 2\n')
            bad=audit_source.audit(root,pins)
            self.assertFalse(bad['source_matches']); self.assertFalse(bad['clean'])


if __name__=='__main__': unittest.main()
