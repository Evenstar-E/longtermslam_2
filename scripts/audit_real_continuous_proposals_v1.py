from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    root=Path('results/real_continuous_proposals_v1');manifest=read(root/'manifest.json');checks=0;total=0;failures=0
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in manifest[field].items():assert sha(name)==h;checks+=1
    for folder in sorted(root.glob('pair_*')):
        p=read(folder/'predictions.json');old=read(Path(manifest['config']['pool_root'])/folder.name/'predictions.json');n=p['old_pool_size']
        assert p['pool'][:n]==old['pool'];checks+=1;total+=len(p['pool'])-n
        for e in p['pool'][n:]:
            T=np.array(e['T_current_from_history']);assert np.isfinite(T).all();checks+=1
            assert np.allclose(T[3],[0,0,0,1],atol=1e-8) and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6);checks+=1
        failures+=sum(t['error'] is not None for t in p['traces'])
        for group in ['old','point_to_plane','colored_icp','all']:
            eligible=[i for i,e in enumerate(p['pool']) if i<n or group=='all' or e['source']==group]
            for score in ['old_nn','ordinary_overlap_rgb']:
                assert p['choices'][group+'__'+score]==min(eligible,key=lambda i:p['pool'][i]['scores'][score]);checks+=1
        for m,i in p['endpoints'].items():
            method,seed=m.split('__');trace=[t for t in p['traces'] if t['method']==method and t['seed']==seed]
            good=[t['output_index'] for t in trace if t['error'] is None]
            assert i==(good[-1] if good else None);checks+=1
            # This batch has no partial-success endpoints: failed paths failed
            # at scale zero. Thus every accepted endpoint completed all scales.
            assert (i is None) or (len(trace)==3 and all(t['error'] is None for t in trace));checks+=1
    scores=read(root/'scores.json');assert total==132 and failures==4;checks+=1
    assert len(scores['rows'])==12*12;checks+=1
    result=dict(status='PASS',checks=checks,new_poses=total,old_poses=3941,expanded_poses=3941+total,solver_path_failures=failures,
        analysis_sources={str(p):sha(p) for p in [Path(__file__),Path('scripts/score_real_continuous_proposals_v1.py'),Path('scripts/diagnose_real_continuous_proposals_v1.py')]})
    save(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
