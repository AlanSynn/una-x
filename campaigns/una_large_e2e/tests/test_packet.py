"""Packet-tool tests only. They do not execute or qualify UNA kernels."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PACKET=Path(__file__).resolve().parents[1]
REPO=PACKET.parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name,PACKET/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
vp=load('validate_packet'); ev=load('check_evidence'); vs=load('verify_source')

class PacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.repo=Path(self.tmp.name)/'repo'
        self.root=self.repo/'campaigns/una_large_e2e'
        shutil.copytree(PACKET,self.root,ignore=shutil.ignore_patterns('__pycache__'))
        for f in ('CLAUDE.md','.claude/commands/goal.md','campaigns/REGISTRY.json'):
            p=self.repo/f; p.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(REPO/f,p)
    def tearDown(self): self.tmp.cleanup()
    def edit(self,file,fn):
        p=self.root/file; value=json.loads(p.read_text()); fn(value); p.write_text(json.dumps(value))
    def validate(self): return vp.validate(self.root,verify_hashes=False)
    def test_packet_passes(self): self.assertTrue(vp.validate(self.root)['ok'])
    def test_unknown_dependency_fails(self):
        self.edit('TASKS_CLAUDE.yaml',lambda d:d['tasks'][0]['depends_on'].append('MISSING'))
        self.assertFalse(self.validate()['ok'])
    def test_cycle_fails(self):
        self.edit('TASKS_CLAUDE.yaml',lambda d:d['tasks'][0]['depends_on'].append('Z00'))
        self.assertFalse(self.validate()['ok'])
    def test_duplicate_id_fails(self):
        self.edit('TASKS_CLAUDE.yaml',lambda d:d['tasks'].append(d['tasks'][0]))
        self.assertFalse(self.validate()['ok'])
    def test_missing_context_fails(self):
        (self.root/'CONTRACT.md').unlink(); self.assertFalse(self.validate()['ok'])
    def test_escape_context_fails(self):
        self.edit('TASKS_CLAUDE.yaml',lambda d:d['tasks'][0]['read_context'].append('../../../../etc/passwd'))
        self.assertFalse(self.validate()['ok'])
    def test_parallel_ownership_fails(self):
        def mutate(d):
            for t in d['tasks']:
                if t['id'] in ('H01','H02'): t['owned_files'].append('collision/shared.py')
        self.edit('TASKS_CLAUDE.yaml',mutate); self.assertFalse(self.validate()['ok'])
    def test_wildcard_file_ownership_fails(self):
        def mutate(d):
            for t in d['tasks']:
                if t['id']=='H01': t['owned_files'].append('collision/_helper_*.py')
                if t['id']=='H02': t['owned_files'].append('collision/_helper_one.py')
        self.edit('TASKS_CLAUDE.yaml',mutate); self.assertFalse(self.validate()['ok'])
    def test_autopublish_fails(self):
        self.edit('campaign.json',lambda d:d.update(auto_publish=True)); self.assertFalse(self.validate()['ok'])
    def test_hash_mutation_fails(self):
        with (self.root/'CONTRACT.md').open('a') as f:f.write('\nchanged\n')
        self.assertFalse(vp.validate(self.root)['ok'])
    def test_wrong_active_route_fails(self):
        p=self.repo/'campaigns/REGISTRY.json'; d=json.loads(p.read_text()); d['active']='optimization_una_cpu'; p.write_text(json.dumps(d))
        self.assertFalse(self.validate()['ok'])

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.f=self.root/'result.txt'; self.f.write_bytes(b'actual test output\n')
        self.task={'id':'X','review_task':False}
        self.record={'task_id':'X','status':'complete','source_commit':'a'*40,'actor':'worker',
                     'commands':[{'argv':['python','tests.py'],'returncode':0}],
                     'artifacts':[{'path':'result.txt','sha256':hashlib.sha256(self.f.read_bytes()).hexdigest()}]}
    def tearDown(self): self.tmp.cleanup()
    def test_good_record(self): self.assertTrue(ev.check(self.record,self.task,self.root)['ok'])
    def test_changed_artifact(self):
        self.f.write_bytes(b'corrupt'); self.assertFalse(ev.check(self.record,self.task,self.root)['ok'])
    def test_short_sha(self):
        self.record['source_commit']='abc1234'; self.assertFalse(ev.check(self.record,self.task,self.root)['ok'])
    def test_failed_command(self):
        self.record['commands'][0]['returncode']=1; self.assertFalse(ev.check(self.record,self.task,self.root)['ok'])
    def test_declared_negative_test(self):
        self.record['commands'][0].update(returncode=1,expected_returncode=1)
        self.assertTrue(ev.check(self.record,self.task,self.root)['ok'])
    def test_escape_artifact(self):
        self.record['artifacts'][0]['path']='../outside'; self.assertFalse(ev.check(self.record,self.task,self.root)['ok'])
    def test_negative_status_without_reason(self):
        self.record['status']='blocked'; self.assertFalse(ev.check(self.record,self.task,self.root)['ok'])
    def test_self_review_fails(self):
        self.task['review_task']=True
        self.record.update(implementation_actor='worker',review_subject_commit='a'*40,independent_review_available=True)
        self.assertFalse(ev.check(self.record,self.task,self.root)['ok'])
    def test_review_identity_passes(self):
        self.task['review_task']=True
        self.record.update(implementation_actor='different-worker',review_subject_commit='a'*40,independent_review_available=True)
        self.assertTrue(ev.check(self.record,self.task,self.root)['ok'])

class SourceBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.cmd('init','-q'); (self.root/'src').mkdir(); (self.root/'src/a.py').write_text('x=1\n')
        self.archive='campaigns/history/old/packet'; p=self.root/self.archive; p.mkdir(parents=True); (p/'evidence.bin').write_bytes(b'\x00\xffpreserved')
        self.cmd('add','.'); self.commit()
        head=self.cmd('rev-parse','HEAD')
        self.cfg={'baseline_commit':head,'baseline_src_tree':self.cmd('rev-parse','HEAD:src'),
                  'archive_path':self.archive,'archive_tree':self.cmd('rev-parse','HEAD:'+self.archive),
                  'source_blobs':{'src/a.py':self.cmd('rev-parse','HEAD:src/a.py')}}
    def tearDown(self): self.tmp.cleanup()
    def cmd(self,*args): return vs.git(self.root,*args)
    def commit(self): self.cmd('-c','user.name=Packet Test','-c','user.email=test@example.invalid','commit','-qm','test')
    def test_bridge(self): self.assertTrue(vs.verify(self.root,self.cfg)['ok'])
    def test_dirty_runtime_fails(self):
        (self.root/'src/a.py').write_text('x=2\n'); self.assertFalse(vs.verify(self.root,self.cfg)['ok'])
    def test_dirty_docs_preserved(self):
        (self.root/'notes.txt').write_text('user work'); self.assertTrue(vs.verify(self.root,self.cfg)['ok'])
    def test_archive_mutation_fails(self):
        (self.root/self.archive/'evidence.bin').write_bytes(b'changed'); self.cmd('add','.'); self.commit()
        self.assertFalse(vs.verify(self.root,self.cfg)['ok'])

try:
    import numpy as np
    cmp=load('compare_npz')
except ImportError:
    np=None

@unittest.skipIf(np is None,'NumPy unavailable: NPZ helper tests not run')
class NpzTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def pair(self,a,b):
        np.savez(self.root/'a.npz',data=a); np.savez(self.root/'b.npz',data=b)
        return cmp.compare(self.root/'a.npz',self.root/'b.npz')['ok']
    def test_empty_archive_fails(self):
        np.savez(self.root/'a.npz'); np.savez(self.root/'b.npz')
        self.assertFalse(cmp.compare(self.root/'a.npz',self.root/'b.npz')['ok'])
    def test_equal(self): self.assertTrue(self.pair(np.array([1.,-0.,np.inf]),np.array([1.,-0.,np.inf])))
    def test_signed_zero(self): self.assertFalse(self.pair(np.array([0.]),np.array([-0.])))
    def test_dtype(self): self.assertFalse(self.pair(np.array([1],dtype='i4'),np.array([1],dtype='i8')))
    def test_nan_payload(self):
        a=np.array([0x7ff8000000000001],dtype='u8').view('f8'); b=np.array([0x7ff8000000000002],dtype='u8').view('f8')
        self.assertFalse(self.pair(a,b))
    def test_shape(self): self.assertFalse(self.pair(np.array([[1.]]),np.array([1.])) )

if __name__=='__main__': unittest.main()
