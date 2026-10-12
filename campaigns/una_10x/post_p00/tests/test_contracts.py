"""Tests of proposed contract models only. No UNA runtime or benchmark is exercised."""
import importlib.util
import math
from pathlib import Path
import struct
import sys
import unittest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from contracts import (ArrayBinding, CommitModel, ReadyCredits, admit_memory,
                       digest, interval_union_ns, metric_key, typed_frame, validate_event)
from derive_p00 import derive

D='a'*64

def array(dtype='<f8', itemsize=8, writable=True, strides=None):
    return ArrayBinding(dtype,(2,),(itemsize,) if strides is None else strides,itemsize,
                        2*itemsize,writable,'aligned',D)

def key(**changes):
    kw=dict(metric='ge',profile='una_legacy',trace_id=D,producer='b'*64,
            arrays={'weights':array()},scalars={'beta':0.1},
            implementation={'fold':'full','llvm':'pinned'},output_cast='float64')
    kw.update(changes); return metric_key(**kw)

class TypedIdentityTests(unittest.TestCase):
    def test_signed_zero(self): self.assertNotEqual(typed_frame(0.0),typed_frame(-0.0))
    def test_int_vs_float(self): self.assertNotEqual(typed_frame(1),typed_frame(1.0))
    def test_bool_vs_int(self): self.assertNotEqual(typed_frame(True),typed_frame(1))
    def test_tuple_vs_list(self): self.assertNotEqual(typed_frame((1,)),typed_frame([1]))
    def test_dict_order(self): self.assertEqual(typed_frame({'a':1,'b':2}),typed_frame({'b':2,'a':1}))
    def test_nested_type(self): self.assertNotEqual(typed_frame({'x':1}),typed_frame({'x':'1'}))
    def test_unknown_type(self):
        with self.assertRaises(TypeError): typed_frame(object())
    def test_nonstring_key(self):
        with self.assertRaises(TypeError): typed_frame({1:'a'})
    def test_nan_payload(self):
        x=struct.unpack('<d',struct.pack('<Q',0x7ff8000000000001))[0]
        y=struct.unpack('<d',struct.pack('<Q',0x7ff8000000000002))[0]
        self.assertTrue(math.isnan(x));self.assertNotEqual(typed_frame(x),typed_frame(y))
    def test_dtype_identity(self): self.assertNotEqual(key(),key(arrays={'weights':array('<f4',4)}))
    def test_stride_identity(self): self.assertNotEqual(key(),key(arrays={'weights':array(strides=(16,))}))
    def test_writable_identity(self): self.assertNotEqual(key(),key(arrays={'weights':array(writable=False)}))
    def test_profile_identity(self): self.assertNotEqual(key(),key(profile='corrected_v1'))
    def test_scalar_identity(self): self.assertNotEqual(key(scalars={'beta':0}),key(scalars={'beta':0.0}))
    def test_producer_identity(self): self.assertNotEqual(key(),key(producer='c'*64))
    def test_incomplete_closure(self):
        with self.assertRaises(ValueError): key(implementation={})
    def test_array_length_mismatch(self):
        with self.assertRaises(ValueError): ArrayBinding('<f8',(2,),(8,),8,8,True,'aligned',D).frame()
    def test_bad_digest(self):
        with self.assertRaises(ValueError): digest('a'*63)

class MemoryAndCreditsTests(unittest.TestCase):
    def test_budget_exact(self): self.assertTrue(admit_memory(100,10,2,10,10,10,10,10,20)['admitted'])
    def test_no_floor_overcommit(self): self.assertFalse(admit_memory(99,10,2,10,10,10,10,10,20)['admitted'])
    def test_negative_budget(self):
        with self.assertRaises(ValueError): admit_memory(-1,0,1,0,0,0,0,0,0)
    def test_bool_not_bytecount(self):
        with self.assertRaises(ValueError): admit_memory(True,0,1,0,0,0,0,0,0)
    def test_zero_workers(self):
        with self.assertRaises(ValueError): admit_memory(100,0,0,0,0,0,0,0,0)
    def test_reserve_before_receive(self):
        q=ReadyCredits(10)
        with self.assertRaises(ValueError): q.receive(0,1)
    def test_reverse_completion_fits(self):
        q=ReadyCredits(40)
        for i in range(4): self.assertTrue(q.reserve(i,10))
        self.assertFalse(q.reserve(4,1))
        for i in reversed(range(4)): q.receive(i,10)
        for i in range(4): q.release_committed(i)
        self.assertEqual(q.used,0)
    def test_oversized(self):
        q=ReadyCredits(20);q.reserve(0,10)
        with self.assertRaises(ValueError): q.receive(0,11)
    def test_duplicate(self):
        q=ReadyCredits(20);q.reserve(0,10)
        with self.assertRaises(ValueError):q.reserve(0,10)
    def test_release_before_receive(self):
        q=ReadyCredits(20);q.reserve(0,10)
        with self.assertRaises(ValueError):q.release_committed(0)

class PublicationTests(unittest.TestCase):
    def test_later_row_cannot_publish(self):
        c=CommitModel([2,2]);c.verify(1);c.fsync(1,0)
        with self.assertRaises(ValueError): c.publish(1,0)
    def test_missing_durability(self):
        c=CommitModel([1]);c.verify(0)
        with self.assertRaises(ValueError): c.publish(0,0)
    def test_partial_row_not_complete(self):
        c=CommitModel([2]);c.verify(0);c.fsync(0,0);c.publish(0,0)
        with self.assertRaises(ValueError):c.commit(0,dir_synced=True,state_applied=True,journaled=True)
        self.assertEqual(c.cancel(),())
    def test_directory_sync_required(self):
        c=CommitModel([1]);c.verify(0);c.fsync(0,0);c.publish(0,0)
        with self.assertRaises(ValueError):c.commit(0,dir_synced=False,state_applied=True,journaled=True)
    def test_full_prefix(self):
        c=CommitModel([2]*32)
        for row in range(32):
            c.verify(row)
            for f in range(2): c.fsync(row,f);c.publish(row,f)
            c.commit(row,dir_synced=True,state_applied=True,journaled=True)
        self.assertEqual(len(c.published),64);self.assertEqual(c.cancel(),tuple(range(32)))
    def test_cancel_stops_publication(self):
        c=CommitModel([1]);c.cancel()
        with self.assertRaises(ValueError):c.verify(0)

class TelemetryTests(unittest.TestCase):
    def event(self):
        return dict(run_id='r',row_id=0,pid=1,tid=1,phase='decode',start_mono_ns=10,
                    end_mono_ns=20,status='complete',process_cpu_start_ns=None,process_cpu_end_ns=None)
    def test_nullable_cpu(self):validate_event(self.event())
    def test_half_available_cpu_rejected(self):
        e=self.event();e['process_cpu_start_ns']=1
        with self.assertRaises(ValueError): validate_event(e)
    def test_reverse_wall(self):
        e=self.event();e['end_mono_ns']=0
        with self.assertRaises(ValueError): validate_event(e)
    def test_nested_union_not_sum(self): self.assertEqual(interval_union_ns([(0,10),(2,8),(9,20)]),20)
    def test_empty_union(self):self.assertEqual(interval_union_ns([]),0)
    def test_disjoint_union(self):self.assertEqual(interval_union_ns([(0,10),(20,30)]),20)
    def test_reverse_interval(self):
        with self.assertRaises(ValueError):interval_union_ns([(4,2)])

class DerivedArithmeticTests(unittest.TestCase):
    def model(self):
        return {'observed':{'baseline_arm':{'par_w4_s':92.1141},
                            'candidate':{'w4_c1':{'wall_s':100.9639}}}}
    def test_budget_arithmetic(self):
        d=derive(self.model());self.assertAlmostEqual(d['historical_sample_budget_10x_s'],9.21141)
        self.assertEqual(d['new_application_measurement'],'unavailable')
    def test_nonfinite_wall_refused(self):
        m=self.model();m['observed']['candidate']['w4_c1']['wall_s']=math.nan
        with self.assertRaises(ValueError):derive(m)
    def test_scenarios_labeled(self):
        d=derive(self.model());self.assertTrue(all(s['evidence_class']=='hypothetical_sensitivity_analysis' for s in d['scenarios']))

if __name__=='__main__':unittest.main()
