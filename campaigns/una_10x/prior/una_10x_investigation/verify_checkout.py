"""Run the research candidates against actual pinned UNA-X kernels.

NOT RUN in the authoring container: GitHub connector reads were available but
container network/DNS could not obtain an executable current checkout.
Run in your current campaign dependency environment. No files in the target
repository are written by this script (Numba cache is redirected).
"""
from __future__ import annotations
import argparse,hashlib,importlib,json,os,sys,tempfile
from pathlib import Path

PIN='395cdc5f683894b6f2ba460f7dbcefee99981ba3'
BLOBS={
 'Engines/AccessibilityWElevation.py':'0ad34b98917e2f738d7a9365d236f7fa82c944c7',
 'Engines/_large_access_scratch.py':'63905385e767db3ba0212181c7b60a7fa050da15',
}

def git_blob(b:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    x=ap.parse_args();src=(x.repo/'src').resolve();pkg=src/'urban_network_analysis'
    for name,want in BLOBS.items():
        p=pkg/name
        if not p.is_file() or git_blob(p.read_bytes())!=want:
            raise SystemExit(f'Pinned-source mismatch for {p}; reconcile drift, do not overwrite it')
    if os.environ.get('NUMBA_DISABLE_JIT')=='1': raise SystemExit('This verification requires JIT enabled')
    os.environ.setdefault('NUMBA_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    import numpy as np
    import numba as nb
    sys.path.insert(0,str(src))
    with tempfile.TemporaryDirectory(prefix='una-probe-nbc-') as cache:
        os.environ['NUMBA_CACHE_DIR']=cache
        # Numba was imported above; set the explicit cache directory as well.
        nb.config.CACHE_DIR=cache
        mod=importlib.import_module('urban_network_analysis.Engines.AccessibilityWElevation')
        actual=Path(mod.__file__).resolve()
        if actual!=pkg/'Engines/AccessibilityWElevation.py': raise SystemExit(f'Wrong import: {actual}')
        import probe,reuse_probe,memo_probe
        records=[]
        for H in (1,4):
            nb.set_num_threads(H)
            for side,D,O,r in [(10,26,14,4.),(32,4096,64,4.),(92,2560,64,20.)]:
                a=probe.make_fixture(side,D,O)
                p,n,c,f,ot,ow,dt,dw,w=a
                ref=mod.integrated_scope_access(ot,ow,p,n,c,f,dt,dw,w,0.03,0.,10.,0.1,'logistic',np.array([1.,1.,0.5]),r)
                assert probe.identical(ref,probe.baseline(*a,r,0.03)),('transcription',H,side)
                assert probe.identical(ref,probe.local(a,r)),('sparse',H,side)
                tape=reuse_probe.build_tape(a,r)
                for beta in (0.005,0.03,0.08):
                    expected=mod.integrated_scope_access(ot,ow,p,n,c,f,dt,dw,w,beta,0.,10.,0.1,'logistic',np.array([1.,1.,0.5]),r)
                    got=reuse_probe.replay(tape[0],tape[1],r,beta)
                    assert probe.identical(expected,got),('tape',H,side,beta)
                    assert expected[1].tobytes()==memo_probe.replay_ge(tape[0],tape[1],r,beta).tobytes(),('metric_memo',H,side,beta)
                records.append({'H':H,'side':side,'D':D,'O':O,'radius':r,'actual_kernel_bits_equal':True})
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(json.dumps({'pin':PIN,'source_files':BLOBS,'records':records,'status':'pass','scope':'prepared-array kernel tests, not public API/export/installed-wheel qualification'},indent=2))
    print(x.out)

if __name__=='__main__': main()
