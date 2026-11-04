from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import catalog
from inspection_decisions.continuous_material_pose_mainline_v1 import select


def main():
    root=Path('results/continuous_material_motion_envelope_v1');base=Path('results/continuous_material_pose_mainline_v1');cfg=read(base/'config.json');m=read(root/'manifest.json');checks=0;allowed_new=0
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in m[field].items():assert sha(name)==h;checks+=1
    cat=catalog();assert m['translation_limit_m']==np.linalg.norm(cat[:,:3,3],axis=1).max();checks+=1
    assert m['rotation_limit_rad']==np.linalg.norm(Rotation.from_matrix(cat[:,:3,:3]).as_rotvec(),axis=1).max();checks+=1
    for run in read(root/'runs.json'):
        tag=run['scene'];a=np.load(root/tag/'allowed.npz');pred=read(root/tag/'predictions.json')
        with (base/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        slots=inp['T0'][:,:3,3];filtered=[];maps=[]
        for i,bb in enumerate(candidates):
            ff=[];mm=[]
            for t,branches in enumerate(bb):
                ok=a[f'{i}_{t}'];assert len(ok)==len(branches) and ok[:54].all();checks+=1;allowed_new+=int(ok[54:].sum())
                for k,b in enumerate(branches):
                    expected=np.linalg.norm(Rotation.from_matrix(b['T'][:3,:3]).as_rotvec())<=m['rotation_limit_rad']+1e-8 and np.linalg.norm(b['T'][:3,3]-slots[b['slot']])<=m['translation_limit_m']+1e-8
                    assert bool(ok[k])==expected;checks+=1
                ix=np.flatnonzero(ok);mm.append(ix);ff.append([branches[j] for j in ix])
            maps.append(mm);filtered.append(ff)
        for score in cfg['scores']:
            ans=select(filtered,score,True);expected=[[i,int(maps[i][0][k]),int(maps[i][1][l])] for i,k,l in ans['selection']]
            assert pred['envelope_'+score]['selection']==expected;checks+=1
        for method,result in pred.items():
            assert result['status']==0 and len(result['selection'])==24;checks+=1
            for i,k,l in result['selection']:
                assert a[f'{i}_0'][k] and a[f'{i}_1'][l];checks+=1
    assert allowed_new==4723;checks+=1
    result={'status':'PASS','checks':checks,'allowed_continuous':allowed_new,'all_original_candidates_retained':20736,
        'analysis_sources':{str(p):sha(p) for p in [Path(__file__),Path('scripts/score_continuous_material_motion_envelope_v1.py'),Path('scripts/diagnose_continuous_material_pose_mainline_v1.py')]}}
    save(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
