"""Post-seal partition of identity failures at the actually retained poses.

Truth only indexes columns for diagnosis. No oracle pose, reranking or new
prediction is produced. Distinguishes missing geometric support from competition.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_rescan_reversible_surface_v2 import read,save,sha,load
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.reversible_surface_v2 import costs


def main():
    root=Path('results/rescan_reversible_surface_v2');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json')
    assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h,p
    scored=read(root/'scores.json');rows=[]
    for scene in cfg['scenes']:
        H,O,raw,initial,_=load(scene,cfg);ids=[h['uid'] for h in H]
        truth=read_ply(Path('data/rescan_third_capture_v1/evaluation/raw')/scene/'gt_segmentation'/f'{scene}_2.ply')['instance_idx']
        for method in ['hard_forward','joint_unrooted','rooted_forward','rooted_joint']:
            dest=root/scene/method;state=np.load(dest/'state.npz');z=np.load(dest/'source_for2.npz');inc=[]
            for i in range(len(H)):
                x=z[f'xyz_{i}'];inc.append({'xyz':x,'rgb':z[f'rgb_{i}'],'source_voxel':z[f'source_voxel_{i}'], 'tree':cKDTree(x) if len(x) else None})
            prediction=np.load(dest/'prediction2.npz')['identity'];truecost=np.full(len(truth),np.nan)
            for start in range(0,len(truth),25000):
                sl=slice(start,start+25000)
                obs={'xyz':raw[1]['xyz'][sl],'rgb':raw[1]['rgb'][sl].astype(float)/255,'class':raw[1]['predicted_class'][sl]}
                cc,a,_,_=costs(obs,state['poses'][1],H,inc,cfg)
                label=np.full(len(a),-1,int);valid=a>=0;label[valid]=np.array(ids)[a[valid]]
                assert np.array_equal(label,prediction[sl])
                for i,uid in enumerate(ids):
                    mask=truth[sl]==uid;view=truecost[sl];view[mask]=cc[mask,i]
            for r in scored['rows']:
                if r['scene']!=scene or r['period']!=2 or r['method']!=method or not r['strict_persistent']:continue
                uid=r['truth_id'];mask=truth==uid;supported=truecost[mask]<cfg['null_cost'];correct=prediction[mask]==uid
                assert not np.any(correct&~supported)
                rows.append({'scene':scene,'method':method,'truth_id':uid,'points':int(mask.sum()),
                    'correct_fraction':float(np.mean(correct)),
                    'true_pose_supported_but_other_wins_fraction':float(np.mean(supported&~correct)),
                    'true_pose_unsupported_fraction':float(np.mean(~supported)),
                    'true_identity_admissible_fraction':float(np.mean(supported)),
                    'mean_true_identity_cost':float(np.mean(truecost[mask]))})
            print('DIAGNOSED',scene,method,flush=True)
    save(root/'failure_partition.json',{'rows':rows,'source_sha':sha(__file__),
        'scope':'At one retained pose per identity, current scoring model. Unsupported does NOT prove no correct pose or missing physical observability. Supported competing IDs may be dataset-equivalent. No new predictions or model tuning.'})


if __name__=='__main__':main()
