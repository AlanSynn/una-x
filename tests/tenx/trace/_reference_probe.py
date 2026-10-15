"""Research prototype, NOT a UNA package patch or an application benchmark.

Source-derived finite-domain A3 search + unchanged metric expressions, compared
with epoch labels and a reverse terminal->destination index. Source reference:
AlanSynn/una-x @ 395cdc5f683894b6f2ba460f7dbcefee99981ba3,
Engines/_large_access_scratch.py and Engines/AccessibilityWElevation.py.

Only the admitted float64/int64, finite, nonnegative prepared-array domain is
exercised. No claim of legacy NaN behavior or other profile/platform coverage.
The timer includes candidate index construction/workspaces, all origin searches,
and four result arrays. It excludes GIS/graph construction, exports and JIT.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, time
from pathlib import Path
from heapq import heappush, heappop
import numpy as np
import numba as nb
import psutil

JIT = dict(cache=True, nogil=True, fastmath=True)
BASE_SHA = '395cdc5f683894b6f2ba460f7dbcefee99981ba3'

@nb.njit(**JIT)
def fold(d_distance, d_weights, cutoff, gravity_beta, gravity_plateau,
         gravity_logistic_midpoint, knn_gravity_plateau, knn_weights,
         knn_gravity_beta, knn_decay, knn_gravity_logistic_midpoint,
         gravity_growth_rate, knn_gravity_growth_rate):
    # Same executable expressions as reach_gravity_knn_access in the source.
    n_reach_filter = np.where(d_distance <= cutoff)[0]
    n_distances = d_distance[n_reach_filter]
    n_weights = d_weights[n_reach_filter]
    reach = n_weights.sum()
    gravity_exponential = (n_weights / np.exp(gravity_beta * np.maximum(0, n_distances - gravity_plateau))).sum()
    gravity_logistic = (n_weights * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances - gravity_plateau - gravity_logistic_midpoint))))).sum()
    knn = min(n_distances.shape[0], knn_weights.shape[0])
    if knn == 0:
        knn_access = 0.0
    else:
        idx = np.argsort(n_distances)
        n_distances_sorted = n_distances[idx]
        n_weights_sorted = n_weights[idx]
        n_distances_cropped = n_distances_sorted[:knn]
        n_weights_cropped = n_weights_sorted[:knn]
        knn_coefficients_cropped = knn_weights[:knn]
        if knn_decay == 'exponential':
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * np.exp(-gravity_beta * np.maximum(0, n_distances_cropped-gravity_plateau))).sum()
        elif knn_decay == 'logistic':
            knn_access = ((knn_coefficients_cropped * n_weights_cropped) * (1 - 1 / (1 + np.exp(-gravity_growth_rate * (n_distances_cropped - gravity_plateau - gravity_logistic_midpoint))))).sum()
        else:
            knn_access = (knn_coefficients_cropped * n_weights_cropped).sum()
    return reach, gravity_exponential, gravity_logistic, knn_access

@nb.njit(**JIT)
def source_a3(ot, ow, ptr, nbr, cost, flag, cutoff, eo, ew):
    # Same statements / prologue / queue operations as _a3_scope_search_tailless.
    os_, oe = ot[0], ot[1]
    ws, we = ow[0], ow[1]
    labels = np.ones(ptr.shape[0]-1, dtype=ow.dtype) + cutoff
    pred = np.empty(0, dtype=ot.dtype)
    labels[os_] = ws
    labels[oe] = we
    q = [(ws, os_)]
    weight, node = heappop(q)
    if we < cutoff:
        heappush(q, (we, oe))
    if ws < cutoff:
        heappush(q, (ws, os_))
    while q:
        weight, node = heappop(q)
        a, b = ptr[node], ptr[node+1]
        ne = 0
        for i in range(b-a):
            w = cost[a+i] + weight
            n = nbr[a+i]
            if w <= cutoff and w < labels[n]:
                eo[ne] = a+i
                ew[ne] = w
                ne += 1
        for j in range(ne):
            off, w = eo[j], ew[j]
            n = nbr[off]
            labels[n] = w
            if flag[off]:
                if ptr[n+1]-ptr[n] > 1:
                    heappush(q, (w, n))
    return labels, pred

@nb.njit(**JIT)
def adjust(labels, dt, dw):
    out = np.empty(dt.shape[0], dtype=labels.dtype)
    for i in range(dt.shape[0]):
        a = labels[dt[i, 0]] + dw[i, 0]
        b = labels[dt[i, 1]] + dw[i, 1]
        out[i] = min(a, b)
    return out

@nb.njit(**JIT)
def make_reverse_index(V, dt):
    p = np.zeros(V+1, dtype=np.int64)
    for d in range(dt.shape[0]):
        p[dt[d, 0]+1] += 1
        p[dt[d, 1]+1] += 1
    for n in range(V):
        p[n+1] += p[n]
    fill = p[:-1].copy()
    ds = np.empty(2*dt.shape[0], dtype=np.int64)
    for d in range(dt.shape[0]):
        for k in range(2):
            n = dt[d, k]
            ds[fill[n]] = d
            fill[n] += 1
    return p, ds

@nb.njit(**JIT)
def epoch_search(ot, ow, ptr, nbr, cost, flag, cutoff, eo, ew,
                 labels, stamp, epoch, touched):
    V = ptr.shape[0]-1
    sentinel = 1.0 + cutoff
    os_, oe = ot[0], ot[1]
    ws, we = ow[0], ow[1]
    nt = 0
    if stamp[os_] != epoch:
        touched[nt] = os_
        nt += 1
    stamp[os_] = epoch
    labels[os_] = ws
    if stamp[oe] != epoch:
        touched[nt] = oe
        nt += 1
    stamp[oe] = epoch
    labels[oe] = we
    q = [(ws, os_)]
    weight, node = heappop(q)
    if we < cutoff:
        heappush(q, (we, oe))
    if ws < cutoff:
        heappush(q, (ws, os_))
    while q:
        weight, node = heappop(q)
        a, b = ptr[node], ptr[node+1]
        ne = 0
        for i in range(b-a):
            w = cost[a+i] + weight
            n = nbr[a+i]
            prior = labels[n] if stamp[n] == epoch else sentinel
            if w <= cutoff and w < prior:
                eo[ne] = a+i
                ew[ne] = w
                ne += 1
        for j in range(ne):
            off, w = eo[j], ew[j]
            n = nbr[off]
            if stamp[n] != epoch:
                touched[nt] = n
                nt += 1
                stamp[n] = epoch
            labels[n] = w
            if flag[off]:
                if ptr[n+1]-ptr[n] > 1:
                    heappush(q, (w, n))
    return nt

@nb.njit(**JIT)
def local_candidates(touched, nt, ip, ids, marks, epoch, scratch):
    k = 0
    for i in range(nt):
        n = touched[i]
        for j in range(ip[n], ip[n+1]):
            d = ids[j]
            if marks[d] != epoch:
                marks[d] = epoch
                scratch[k] = d
                k += 1
    # Critical: original destination order, NOT search discovery order.
    return np.sort(scratch[:k])

@nb.njit(**JIT)
def collect_local(cands, labels, stamp, epoch, cutoff, dt, dw, weight):
    ds = np.empty(cands.shape[0], dtype=np.float64)
    ws = np.empty(cands.shape[0], dtype=np.float64)
    sentinel = 1.0 + cutoff
    for i in range(cands.shape[0]):
        d = cands[i]
        a, b = dt[d, 0], dt[d, 1]
        la = labels[a] if stamp[a] == epoch else sentinel
        lb = labels[b] if stamp[b] == epoch else sentinel
        ds[i] = min(la + dw[d, 0], lb + dw[d, 1])
        ws[i] = weight[d]
    return ds, ws

@nb.njit(parallel=True, **JIT)
def baseline(ptr, nbr, cost, flag, ot, ow, dt, dw, weights, cutoff, beta):
    O = ot.shape[0]
    r = np.empty(O, dtype=ot.dtype)
    ge = np.empty(O, dtype=cost.dtype)
    gl = np.empty(O, dtype=cost.dtype)
    kn = np.empty(O, dtype=cost.dtype)
    deg = np.max(ptr[1:]-ptr[:-1]) if ptr.shape[0]>1 else 0
    kws = np.array([1.0, 1.0, 0.5])
    for o in nb.prange(O):
        eo = np.empty(deg, np.int64)
        ew = np.empty(deg, np.float64)
        labels, _ = source_a3(ot[o], ow[o], ptr, nbr, cost, flag, cutoff, eo, ew)
        ds = adjust(labels, dt, dw)
        r[o], ge[o], gl[o], kn[o] = fold(ds, weights, cutoff, beta, 0., 10., 0., kws, beta, 'logistic', 10., 0.1, 0.1)
    return r, ge, gl, kn

@nb.njit(parallel=True, **JIT)
def local_driver(ptr, nbr, cost, flag, ot, ow, dt, dw, weights, cutoff, beta, ip, ids, H):
    V = ptr.shape[0]-1
    D, O = dt.shape[0], ot.shape[0]
    r = np.empty(O, dtype=ot.dtype)
    ge = np.empty(O, dtype=cost.dtype)
    gl = np.empty(O, dtype=cost.dtype)
    kn = np.empty(O, dtype=cost.dtype)
    deg = np.max(ptr[1:]-ptr[:-1]) if V else 0
    H = min(H, max(1, O))
    kws = np.array([1.0, 1.0, 0.5])
    # Separate ownership per physical lane. Epochs are limited to O+1;
    # this prototype starts fresh on EVERY call, so no cross-call rollover.
    for lane in nb.prange(H):
        labels = np.empty(V, np.float64)
        stamp = np.zeros(V, np.int64)
        touched = np.empty(V, np.int64)
        marks = np.zeros(D, np.int64)
        cbuf = np.empty(D, np.int64)
        eo = np.empty(deg, np.int64)
        ew = np.empty(deg, np.float64)
        for o in range(lane, O, H):
            epoch = o+1
            nt = epoch_search(ot[o], ow[o], ptr, nbr, cost, flag, cutoff,
                              eo, ew, labels, stamp, epoch, touched)
            cands = local_candidates(touched, nt, ip, ids, marks, epoch, cbuf)
            ds, ws = collect_local(cands, labels, stamp, epoch, cutoff, dt, dw, weights)
            r[o], ge[o], gl[o], kn[o] = fold(ds, ws, cutoff, beta, 0., 10., 0., kws, beta, 'logistic', 10., 0.1, 0.1)
    return r, ge, gl, kn


def admitted(a, cutoff):
    p,n,w,f,ot,ow,dt,dw,dweights = a
    if not (isinstance(cutoff,float) and np.isfinite(cutoff) and cutoff>=0 and np.float64(cutoff)+1>cutoff):
        return False
    if any(type(x) is not np.ndarray for x in a): return False
    if p.ndim!=1 or n.ndim!=1 or w.ndim!=1 or f.ndim!=1 or dweights.ndim!=1: return False
    if p.size<1 or p[0]!=0 or p[-1]!=len(n) or np.any(np.diff(p)<0): return False
    V = len(p)-1
    if len(w)!=len(n) or len(f)!=len(n): return False
    if p.dtype!=np.int64 or n.dtype!=np.int64 or ot.dtype!=np.int64 or dt.dtype!=np.int64: return False
    if any(x.dtype!=np.float64 for x in (w,ow,dw,dweights)) or f.dtype!=np.bool_: return False
    if ot.ndim!=2 or ot.shape[1]!=2 or ow.shape!=ot.shape or dt.ndim!=2 or dt.shape[1]!=2 or dw.shape!=dt.shape: return False
    if len(dweights)!=len(dt): return False
    if any(x.size and (x.min()<0 or x.max()>=V) for x in (n,ot,dt)): return False
    if any(not np.isfinite(x).all() or (x<0).any() for x in (w,ow,dw,dweights)): return False
    if cutoff > np.finfo(np.float64).max/4: return False
    return True


def local(a, cutoff, beta=0.03):
    if not admitted(a, cutoff):
        raise ValueError('Research prototype fast domain refused; not a public dispatcher')
    ip,ids=make_reverse_index(len(a[0])-1,a[6])
    return local_driver(*a, cutoff, beta, ip, ids, nb.get_num_threads())


def make_fixture(side, dests, origins, seed=721):
    rng=np.random.default_rng(seed)
    V=side*side
    grid=np.arange(V,dtype=np.int64).reshape(side,side)
    st=np.concatenate((grid[:,:-1].ravel(),grid[:-1,:].ravel()))
    en=np.concatenate((grid[:,1:].ravel(),grid[1:,:].ravel()))
    # Original forward incidences, then reverse, stable grouping.
    sources=np.concatenate((st,en))
    order=np.argsort(sources,kind='stable')
    p=np.concatenate((np.zeros(1,np.int64),np.cumsum(np.bincount(sources,minlength=V),dtype=np.int64)))
    n=np.concatenate((en,st))[order]
    ec=0.8+0.4*rng.random(len(st))
    w=np.concatenate((ec,ec))[order]
    f=np.ones(len(n),dtype=np.bool_)
    def points(N):
        es=rng.integers(len(st),size=N)
        frac=0.15+0.7*rng.random(N)
        ts=np.array([st[es],en[es]],dtype=np.int64).T
        tw=np.array([ec[es]*frac,ec[es]*(1-frac)],dtype=np.float64).T
        return ts,tw
    ot,ow=points(origins);dt,dw=points(dests)
    return p,n,w,f,ot,ow,dt,dw,0.25+2*rng.random(dests)


def identical(a,b):
    return len(a)==len(b) and all(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes() for x,y in zip(a,b))


def tests():
    nb.set_num_threads(1)
    count=0
    for side in (2,3,5,8):
        for D in (0,1,7,31):
            a=make_fixture(side,D,11,seed=side*D+77)
            for r in (0.,0.4,2.,float(np.nextafter(2.,np.inf)),100.):
                aa=baseline(*a,r,0.03);bb=local(a,r)
                assert identical(aa,bb),(side,D,r)
                count+=1
    # Crafted multiedges, loops, isolated nodes, signed-zero costs/seeds,
    # duplicate origins and endpoints: randomized finite-domain batteries.
    for seed in range(60):
        rng=np.random.default_rng(seed)
        V=12;E=34;D=27;O=9
        st=rng.integers(V,size=E,dtype=np.int64); en=rng.integers(V,size=E,dtype=np.int64)
        src=np.r_[st,en]; order=np.argsort(src,kind='stable')
        p=np.r_[np.int64(0),np.cumsum(np.bincount(src,minlength=V),dtype=np.int64)]
        n=np.r_[en,st][order]
        costs=rng.choice(np.array([0.,-0.,0.5,1.,2.,4.]),size=2*E)[order]
        flag=rng.random(2*E)>0.1
        ot=rng.integers(V,size=(O,2),dtype=np.int64);ow=rng.choice(np.array([0.,-0.,0.25,1.,2.]),size=(O,2))
        dt=rng.integers(V,size=(D,2),dtype=np.int64);dw=rng.choice(np.array([0.,-0.,0.5,1.]),size=(D,2))
        ww=rng.random(D)
        a=(p,n,costs,flag,ot,ow,dt,dw,ww)
        r=2.5
        assert identical(baseline(*a,r,0.03),local(a,r)),seed
        # Whole label states on each origin, not only final aggregate values.
        labels=np.empty(V);stamp=np.zeros(V,np.int64);touch=np.empty(V,np.int64)
        deg=int(np.diff(p).max());eo=np.empty(deg,np.int64);ew=np.empty(deg)
        for o in range(O):
            ref=source_a3(ot[o],ow[o],p,n,costs,flag,r,eo,ew)[0]
            epoch_search(ot[o],ow[o],p,n,costs,flag,r,eo,ew,labels,stamp,o+1,touch)
            view=np.where(stamp==o+1,labels,r+1)
            assert ref.tobytes()==view.tobytes(),('labels',seed,o)
        count+=1
    a=make_fixture(3,3,3)
    assert not admitted(a,float(2**54))
    a_bad=list(a);a_bad[7]=a[7].copy();a_bad[7][0,0]=-1
    assert not admitted(tuple(a_bad),2.)
    print(json.dumps({'component_cases_passed':count,'full_label_cases_passed':540,'guard_checks':2,'application_tests':'not_run'}))


def fingerprint(a):
    h=hashlib.sha256()
    for x in a:
        h.update(x.dtype.str.encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def benchmark(side,D,O,r,H,reps,out):
    nb.set_num_threads(H)
    a=make_fixture(side,D,O)
    assert admitted(a,r)
    # Compile/specialize on small inputs with identical layouts, OUTSIDE timer.
    tiny=make_fixture(4,11,8)
    t=time.perf_counter();baseline(*tiny,r,0.03);local(tiny,r)
    compile_warmup=time.perf_counter()-t
    expected=baseline(*a,r,0.03)
    got=local(a,r)
    assert identical(expected,got)
    rows=[]
    for rep in range(reps):
        order=('baseline','local') if rep%2==0 else ('local','baseline')
        for arm in order:
            t=time.perf_counter_ns()
            result=baseline(*a,r,0.03) if arm=='baseline' else local(a,r)
            ns=time.perf_counter_ns()-t
            assert identical(expected,result),(rep,arm)
            rows.append({'rep':rep,'arm':arm,'wall_ns':ns,'output_bytes_equal':True})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    data={'reference_source_sha':BASE_SHA,'boundary':'prepared CSR and terminal arrays -> four materialized metric arrays; candidate guard/index/all workspaces included; GIS, graph building, exports, process startup and JIT excluded',
          'classification':'measured_source_derived_component_prototype_not_UNA_end_to_end',
          'side':side,'nodes':side*side,'directed_arcs':len(a[1]),'destinations':D,'origins':O,'radius_in_synthetic_cost_units':r,'numba_threads':H,'numba_effective_threads':nb.get_num_threads(),
          'input_sha256':fingerprint(a),'input_bytes':sum(x.nbytes for x in a),'output_bytes':sum(x.nbytes for x in expected),
          'compile_warmup_s':compile_warmup,'records':rows,'platform':platform.platform(),'numpy':np.__version__,'numba':nb.__version__,
          'cpu_affinity':psutil.Process().cpu_affinity(),'process_rss_after_bytes':psutil.Process().memory_info().rss,'memory_peak':'not_sampled','source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(data,indent=2))
    ms={arm:float(np.median([x['wall_ns'] for x in rows if x['arm']==arm]))/1e6 for arm in ('baseline','local')}
    print(json.dumps({'case':out.name,'median_ms_derived':ms,'component_speedup_derived':ms['baseline']/ms['local'],'equal_bits':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tests',action='store_true')
    p.add_argument('--side',type=int,default=128);p.add_argument('--destinations',type=int,default=16384);p.add_argument('--origins',type=int,default=256);p.add_argument('--radius',type=float,default=4.);p.add_argument('--threads',type=int,default=1);p.add_argument('--reps',type=int,default=5);p.add_argument('--out',default='measurements.json')
    x=p.parse_args()
    if x.tests: tests()
    else: benchmark(x.side,x.destinations,x.origins,x.radius,x.threads,x.reps,x.out)
