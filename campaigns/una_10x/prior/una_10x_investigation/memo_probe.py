"""Cohort reuse plus memoization of metric fields whose full inputs are unchanged.

The only varied parameter is exponential gravity beta. In the fixed logistic
KNN configuration, reach, logistic gravity and KNN are identical across jobs.
They are computed once and returned as FRESH arrays per job. Exponential gravity
uses the original typed expression, with local bitwise checks. This is NOT a
validated portable legacy compiler contract or a patch for the whole project.
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
import numba as nb
import probe,reuse_probe

@nb.njit(cache=True,nogil=True,fastmath=True)
def ge_only(distance,weight,cutoff,beta):
    # Retain original selection and expression; never reassociate in Python.
    keep=np.where(distance<=cutoff)[0]
    n_distances=distance[keep]
    n_weights=weight[keep]
    return (n_weights / np.exp(beta * np.maximum(0, n_distances - 0.))).sum()

@nb.njit(parallel=True,cache=True,nogil=True,fastmath=True)
def replay_ge(ds,ws,cutoff,beta):
    result=np.empty(len(ds),np.float64)
    for oi in nb.prange(len(ds)):
        o=np.int64(oi)
        result[o]=ge_only(ds[o],ws[o],cutoff,beta)
    return result


def tests():
    rng=np.random.default_rng(777)
    cases=0
    for n in [0,1,2,3,4,7,8,9,15,16,17,31,32,33,64,65,257]:
        for k in range(10):
            d=rng.random(n)*20
            w=rng.random(n)*2
            beta=float(rng.random()*0.5)
            expected=probe.fold(d,w,12.,beta,0.,10.,0.,np.array([1.,1.,0.5]),beta,'logistic',10.,0.1,0.1)[1]
            got=ge_only(d,w,12.,beta)
            assert np.float64(expected).tobytes()==np.float64(got).tobytes(),(n,k)
            cases+=1
    return cases


def run(K,out):
    nb.set_num_threads(4)
    a=probe.make_fixture(92,2560,512)
    cutoff=20.
    betas=np.linspace(0.005,0.08,K)
    tiny=probe.make_fixture(4,11,16)
    tape=reuse_probe.build_tape(tiny,cutoff)
    reuse_probe.replay(tape[0],tape[1],cutoff,float(betas[0]));replay_ge(tape[0],tape[1],cutoff,float(betas[0]))
    probe.baseline(*tiny,cutoff,float(betas[0]))
    expected=[probe.baseline(*a,cutoff,float(b)) for b in betas]
    rows=[]
    for rep in range(5):
        for arm in (('repeat','reuse_memo') if rep%2==0 else ('reuse_memo','repeat')):
            t=time.perf_counter_ns()
            if arm=='repeat':
                results=[probe.baseline(*a,cutoff,float(b)) for b in betas]
                bt=None
            else:
                t0=time.perf_counter_ns();tape=reuse_probe.build_tape(a,cutoff);bt=time.perf_counter_ns()-t0
                first=reuse_probe.replay(tape[0],tape[1],cutoff,float(betas[0]))
                results=[first]
                for b in betas[1:]:
                    # No aliasing of mutable public output arrays across jobs.
                    results.append((first[0].copy(),replay_ge(tape[0],tape[1],cutoff,float(b)),first[2].copy(),first[3].copy()))
            ns=time.perf_counter_ns()-t
            assert all(probe.identical(x,y) for x,y in zip(expected,results))
            assert expected[0][1].tobytes()!=expected[-1][1].tobytes()
            for i in range(1,K):
                assert not np.shares_memory(results[0][0],results[i][0])
            rows.append({'rep':rep,'arm':arm,'wall_ns':ns,'build_tape_ns':bt,'all_result_bits_equal':True})
    med={v:float(np.median([x['wall_ns'] for x in rows if x['arm']==v]))/1e9 for v in ['repeat','reuse_memo']}
    d={'classification':'measured_source_derived_cohort_prototype_not_UNA_end_to_end','reference_sha':probe.BASE_SHA,'boundary':'K prepared-array metric jobs with distinct beta values; tape construction, first full metrics, ge-only replays, fresh result arrays included. GIS, cache store, batch API, exports and JIT excluded.',
       'K':K,'betas':betas.tolist(),'nodes':92*92,'destinations':2560,'origins':512,'threads':4,'radius':cutoff,'input_sha256':probe.fingerprint(a),'raw':rows,'derived_medians_s':med,'derived_speedup':med['repeat']/med['reuse_memo'],'file_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(out).write_text(json.dumps(d,indent=2));print(json.dumps({'K':K,'derived_medians_s':med,'derived_speedup':d['derived_speedup']}),flush=True)

if __name__=='__main__':
    print(json.dumps({'metric_expression_bit_cases_passed':tests()}),flush=True)
    root=Path(__file__).resolve().parent/'raw'
    run(8,root/'memo_K8.json')
    run(32,root/'memo_K32.json')
