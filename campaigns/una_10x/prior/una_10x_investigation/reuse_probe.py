"""Prepared-array cohort experiment with DISTINCT gravity-beta values.

One bounded tape build; every requested metric job is still computed. No GIS,
public RunBatch, persistent-cache I/O, exports, or native/GPU qualification.
This is not a production cache and has no persisted invalidation mechanism.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
import numba as nb
from numba.typed import List
import probe

@nb.njit(parallel=True,cache=True,nogil=True,fastmath=True)
def tape_block(ptr,nbr,cost,flag,ot,ow,dt,dw,weights,cutoff):
    O=ot.shape[0];D=dt.shape[0]
    values=np.empty((O,D),np.float64)
    wv=np.empty((O,D),np.float64)
    ids=np.empty((O,D),np.int64)
    counts=np.empty(O,np.int64)
    deg=np.max(ptr[1:]-ptr[:-1])
    for o in nb.prange(O):
        eo=np.empty(deg,np.int64);ew=np.empty(deg,np.float64)
        labels,_=probe.source_a3(ot[o],ow[o],ptr,nbr,cost,flag,cutoff,eo,ew)
        ds=probe.adjust(labels,dt,dw)
        kept=np.where(ds<=cutoff)[0]
        counts[o]=len(kept)
        for i in range(len(kept)):
            d=kept[i]
            values[o,i]=ds[d]
            wv[o,i]=weights[d]
            ids[o,i]=d
    return values,wv,ids,counts

def build_tape(a,cutoff,block=16):
    ds=List.empty_list(nb.float64[::1]);ws=List.empty_list(nb.float64[::1]);ids=List.empty_list(nb.int64[::1])
    for lo in range(0,len(a[4]),block):
        p,n,c,f,ot,ow,dt,dw,w=a
        vals,weights,dids,ct=tape_block(p,n,c,f,ot[lo:lo+block],ow[lo:lo+block],dt,dw,w,cutoff)
        for j,nk in enumerate(ct):
            ds.append(vals[j,:nk].copy());ws.append(weights[j,:nk].copy());ids.append(dids[j,:nk].copy())
    return ds,ws,ids

@nb.njit(parallel=True,cache=True,nogil=True,fastmath=True)
def replay(ds,ws,cutoff,beta):
    O=len(ds)
    r=np.empty(O,np.int64);ge=np.empty(O);gl=np.empty(O);kn=np.empty(O)
    kw=np.array([1.,1.,0.5])
    for o in nb.prange(O):
        r[o],ge[o],gl[o],kn[o]=probe.fold(ds[o],ws[o],cutoff,beta,0.,10.,0.,kw,beta,'logistic',10.,0.1,0.1)
    return r,ge,gl,kn

def run(K, out):
    nb.set_num_threads(4)
    a=probe.make_fixture(92,2560,512)
    cutoff=20.
    betas=np.linspace(0.005,0.08,K)
    tiny=probe.make_fixture(4,11,16)
    tape=build_tape(tiny,cutoff);replay(tape[0],tape[1],cutoff,float(betas[0]))
    probe.baseline(*tiny,cutoff,float(betas[0]))
    expected=[probe.baseline(*a,cutoff,float(b)) for b in betas]
    rows=[]
    for rep in range(5):
        for arm in (('repeat','reuse') if rep%2==0 else ('reuse','repeat')):
            t=time.perf_counter_ns()
            if arm=='repeat':
                results=[probe.baseline(*a,cutoff,float(b)) for b in betas]
                bt=None; tape_bytes=None
            else:
                t0=time.perf_counter_ns(); tape=build_tape(a,cutoff);bt=time.perf_counter_ns()-t0
                results=[replay(tape[0],tape[1],cutoff,float(b)) for b in betas]
                # Bookkeeping bytes counted after timer below.
            ns=time.perf_counter_ns()-t
            if arm=='reuse': tape_bytes=sum(x.nbytes for group in tape for x in group)
            assert all(probe.identical(x,y) for x,y in zip(expected,results))
            # At least one result differs between requested jobs: not replaying
            # the same final answer and counting it as parameter-sweep work.
            assert expected[0][1].tobytes()!=expected[-1][1].tobytes()
            rows.append({'rep':rep,'arm':arm,'wall_ns':ns,'build_tape_ns':bt,'tape_bytes':tape_bytes,'all_result_bits_equal':True})
    med={v:float(np.median([x['wall_ns'] for x in rows if x['arm']==v]))/1e9 for v in ['repeat','reuse']}
    d={'classification':'measured_source_derived_cohort_prototype_not_UNA_end_to_end','reference_sha':probe.BASE_SHA,'boundary':'K prepared-array accessibility metric jobs with distinct beta values; one tape build, all replays and returned arrays included; no GIS, file cache, public batch, exports or JIT',
       'K':K,'betas':betas.tolist(),'nodes':92*92,'destinations':2560,'origins':512,'threads':4,'radius':cutoff,'input_sha256':probe.fingerprint(a),'raw':rows,'derived_medians_s':med,'derived_speedup':med['repeat']/med['reuse'],'file_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(out).write_text(json.dumps(d,indent=2));print(json.dumps({'K':K,'derived_medians_s':med,'derived_speedup':d['derived_speedup']}),flush=True)

if __name__=='__main__':
    root=Path(__file__).resolve().parent/'raw'
    run(8,root/'reuse_K8.json')
    run(32,root/'reuse_K32.json')
