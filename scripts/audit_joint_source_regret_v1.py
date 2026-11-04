import argparse
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.joint_source_regret_v1 import selectors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/joint_source_balanced_v2');args=ap.parse_args();out=Path(args.output);checks=0;states=0;conditional=0;updates=0
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h;checks+=1
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h;checks+=1
    pool=Path('results/material_cycle_bridge_v1')
    for episode,pth in enumerate(sorted(out.glob('episode_*/predictions.json'))):
        p=read(pth);rr=p['rows'];states+=len(rr);conditional+=len(p['conditional_indices']);updates+=len(p['joint_optimizer_indices'])
        old=[read(pool/f'pair_{episode*3+j:02d}'/'predictions.json') for j in range(3)]
        for i in p['conditional_indices']:
            r=rr[i];v=r['view'];T=np.array(old[v-1]['rows'][r['old_pose_index']]['T']);assert np.allclose(np.linalg.inv(np.array(r['Q'][v])),T,atol=1e-10,rtol=0);checks+=1
        for r in rr:
            Q=np.array(r['Q']);assert np.array_equal(Q[0],np.eye(4));assert np.isfinite(Q).all();checks+=1
            for q in Q:
                assert np.allclose(q[3],[0,0,0,1]) and np.allclose(q[:3,:3].T@q[:3,:3],np.eye(3),atol=1e-6) and np.isclose(np.linalg.det(q[:3,:3]),1,atol=1e-6);checks+=1
            l=r['loss'];assert np.isclose(np.mean(l['deletion_losses']),l['joint_ray'],atol=1e-12);checks+=1
        choices,d=selectors(rr);assert all(p['choices'][m]==i for m,i in choices.items());assert np.allclose(d['regret'],p['robust']['regret']) and min(d['regret'])>=-1e-12;checks+=2
        if 'base_row_count' in p:
            original=read(Path('results/joint_source_regret_v1')/pth.parent.name/'predictions.json');assert p['rows'][:p['base_row_count']]==original['rows'];checks+=1
    assert conditional==3965;checks+=1
    save(out/'audit.json',dict(passed=True,checks=checks,old_hypotheses_retained=conditional,joint_states=states,joint_iterates=updates,scope='Finite conditional pool, not Cartesian completeness or continuous certification.'))
    scripts=list(Path('scripts').glob('*joint_source*py'))
    save(out/'final_evidence_index.json',dict(files={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='final_evidence_index.json'},analysis_sources={str(p):sha(p) for p in scripts},report_sha=sha('research/joint_source_regret_v1_2026-10-24.md')))
    print('AUDIT_PASS',checks,'states',states,'conditional',conditional,'joint_iterates',updates)


if __name__=='__main__':main()
