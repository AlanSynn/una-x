from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
bp, inv, audit, comp, val = [load(n) for n in ['bitparity', 'inventory_api', 'source_audit', 'check_completion', 'validate_packet']]

class BitTests(unittest.TestCase):
    def test_identical(self):
        a={'x':np.array([1., -0., np.inf, -np.inf, np.nan])}
        self.assertTrue(bp.compare_arrays(a,a)['match'])
    def test_signed_zero(self):
        self.assertFalse(bp.compare_arrays({'x':np.array([0.])},{'x':np.array([-0.])})['match'])
    def test_nan_payload(self):
        a=np.array([0x7ff8000000000001],dtype=np.uint64).view(np.float64)
        b=np.array([0x7ff8000000000002],dtype=np.uint64).view(np.float64)
        self.assertFalse(bp.compare_arrays({'x':a},{'x':b})['match'])
    def test_one_ulp(self):
        self.assertFalse(bp.compare_arrays({'x':np.array([1.])},{'x':np.array([np.nextafter(1.,2.)])})['match'])
    def test_dtype(self):
        self.assertFalse(bp.compare_arrays({'x':np.array([1.],dtype='f4')},{'x':np.array([1.],dtype='f8')})['match'])
    def test_endian(self):
        self.assertFalse(bp.compare_arrays({'x':np.array([1.],dtype='>f8')},{'x':np.array([1.],dtype='<f8')})['match'])
    def test_shape(self):
        self.assertFalse(bp.compare_arrays({'x':np.zeros((1,2))},{'x':np.zeros((2,1))})['match'])
    def test_missing_field(self):
        self.assertFalse(bp.compare_arrays({'x':np.zeros(1)},{'y':np.zeros(1)})['match'])
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): bp.compare_arrays({}, {})
    def test_object_rejected(self):
        with self.assertRaises(ValueError): bp.compare_arrays({'x':np.array([{}])},{'x':np.array([{}])})
    def test_npz(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.npz'; np.savez(p,x=np.arange(5))
            self.assertTrue(bp.compare_arrays(bp.load_arrays(p),{'x':np.arange(5)})['match'])
    def test_no_pickle(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.npy'; np.save(p,np.array([{}]))
            with self.assertRaises(ValueError): bp.load_arrays(p)

class InventoryTests(unittest.TestCase):
    def test_signature_no_execution(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'a.py').write_text("raise RuntimeError('do not run')\ndef f(x,/,*,y=3): pass\nclass Z:\n def __getitem__(self,k): pass\n def _private(self): pass\n")
            r=inv.inventory(Path(t)); syms={v['symbol']:v for v in r['records']}
            self.assertIn('a.f',syms); self.assertIn('y=3',syms['a.f']['signature'])
            self.assertIn('a.Z.__getitem__',syms); self.assertNotIn('a.Z._private',syms)
            self.assertFalse(r['complete_semantic_census'])
    def test_empty_source(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError): inv.inventory(Path(t))
    def test_exports(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'__init__.py').write_text("from .x import f\n__all__=['f']\n")
            r=inv.inventory(Path(t)); self.assertEqual(len(r['records']),2)

class AuditTests(unittest.TestCase):
    def test_git_blob_known(self):
        self.assertEqual(audit.git_blob(b''),'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
    def test_match_and_drift(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'x').write_bytes(b'a\r\nb\n'); h=audit.git_blob((p/'x').read_bytes())
            self.assertTrue(audit.audit(p,{'x':h})['match'])
            (p/'x').write_bytes(b'a\nb\n'); self.assertFalse(audit.audit(p,{'x':h})['match'])
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError): audit.audit(Path(t),{'../x':'0'*40})
    def test_missing(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(audit.audit(Path(t),{'x':'0'*40})['match'])
    def test_empty_pins(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError): audit.audit(Path(t),{})

class CompletionTests(unittest.TestCase):
    def test_partial_allowed(self):
        self.assertEqual(comp.check({'state':'partial'},{'requirements':{}},ROOT),[])
    def test_false_qualified_rejected(self):
        r=json.loads((ROOT/'templates/completion.json').read_text());r['state']='qualified'
        self.assertTrue(comp.check(r,json.loads((ROOT/'campaign.json').read_text()),ROOT))
    def test_fake_gpu_rejected(self):
        r={'state':'qualified','source_sha':'a'*40,'api_census_complete':True,'features':{},'review':'passed','ci':'passed'}
        errors=comp.check(r,{'requirements':{}},ROOT)
        self.assertTrue(any('GPU qualification' in x for x in errors))
    def test_vacuous_parity_rejected(self):
        r={'state':'qualified','source_sha':'a'*40,'api_census_complete':True,'features':{},'review':'passed','ci':'passed'}
        self.assertTrue(any('non-vacuous' in x for x in comp.check(r,{'requirements':{}},ROOT)))
    def test_unknown_state(self):
        self.assertTrue(comp.check({'state':'fastest'},{'requirements':{}},ROOT))
    def test_honest_complete_fixture_and_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=root/'proof.json';p.write_text('{"test":"fixture"}')
            entry={'path':'proof.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            feat={'status':'passed','tests':1,'evidence':[entry],'substantive_kernel_calls':2,'fallback_only':False,'real_hardware':True,'simulator':False}
            r={'state':'qualified','source_sha':'a'*40,'api_census_complete':True,'features':{'native_backend':copy.deepcopy(feat),'gpu_backend':copy.deepcopy(feat),'bitwise_parity':{'compared_arrays':1,'compared_bytes':8}},'review':'passed','ci':'passed'}
            c={'requirements':{'native_backend':[],'gpu_backend':[]}}
            self.assertEqual(comp.check(r,c,root),[])
            p.write_text('corrupt');self.assertTrue(comp.check(r,c,root))

class PacketTests(unittest.TestCase):
    def test_current_packet(self):
        self.assertTrue(val.validate(ROOT)['passed'])
    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertFalse(val.validate(Path(t))['passed'])
    def test_digest_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'packet';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            with (root/'AUTHORITY.md').open('a') as f:f.write('tamper')
            self.assertFalse(val.validate(root)['passed'])
    def test_cycle(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'packet';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            p=root/'TASKS_CLAUDE.yaml';d=json.loads(p.read_text());d['tasks'][0]['depends_on']=['CLOSE'];p.write_text(json.dumps(d))
            r=val.validate(root);self.assertFalse(next(c['passed'] for c in r['checks'] if c['name']=='acyclic_dag'))
    def test_missing_capability(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'packet';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
            p=root/'campaign.json';d=json.loads(p.read_text());del d['requirements']['gpu_backend'];p.write_text(json.dumps(d))
            r=val.validate(root);self.assertFalse(next(c['passed'] for c in r['checks'] if c['name']=='required_capabilities'))

if __name__ == '__main__':
    unittest.main()
