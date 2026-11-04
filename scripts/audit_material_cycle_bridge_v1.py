from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    out=Path('results/material_cycle_bridge_v1');cfg=read('configs/material_cycle_bridge_v1.json');count=0
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h;count+=1
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h;count+=1
    old_count=0;new_count=0
    for pth in sorted(out.glob('pair_*/predictions.json')):
        p=read(pth);old=read(Path(cfg['pool_root'])/pth.parent.name/'predictions.json');n=len(old['pool']);assert p['old_pool_size']==n and len(p['rows'])==n+2;count+=1;old_count+=n;new_count+=2
        for i,r in enumerate(p['rows']):
            T=np.array(r['T']);assert np.isfinite(T).all() and np.allclose(T[3],[0,0,0,1]) and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) and np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-6);count+=1
            assert np.isfinite(list(r['scores'].values())).all();count+=1
            if i<n:assert r['T']==old['pool'][i]['T_current_from_history'] and r['base']==old['pool'][i]['scores']['old_nn'];count+=1
        for m,i in p['choices'].items():assert i==min(range(len(p['rows'])),key=lambda j:p['rows'][j]['scores'][m]);count+=1
    assert old_count==3941 and new_count==24;count+=1
    # A fully consistent cycle can be physically wrong on a symmetric surface.
    x=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0.]])
    U=[np.eye(4)[np.roll(np.arange(4),k)] for k in [0,1,2]]
    ab=U[0]@U[1].T;bc=U[1]@U[2].T;ac=U[0]@U[2].T
    assert np.array_equal(ab@bc,ac);assert np.max(np.linalg.norm(ac@x-x,axis=1))>1;count+=2
    save(out/'audit.json',dict(checks=count,passed=True,old_candidates=old_count,shared_new_path_candidates=new_count,cycle_counterexample='Perfect permutation closure does not imply correct material endpoints on a symmetric object. Algebraic unit example only, not a real experiment.'))
    scripts=[Path(__file__),Path('scripts/diagnose_material_sampling_bias_v1.py')]+list(Path('scripts').glob('*material_cycle_bridge_v1.py'))
    save(out/'final_evidence_index.json',dict(files={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='final_evidence_index.json'},analysis_sources={str(p):sha(p) for p in scripts},report_sha=sha('research/material_cycle_bridge_v1_2026-10-24.md')))
    print('AUDIT_PASS',count,'original/new',old_count,new_count)


if __name__=='__main__':main()
