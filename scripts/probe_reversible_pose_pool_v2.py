"""Evaluator-only retained-pose versus cached-pose-pool probe on declared cases.

Uses truth masks ONLY after inference seal; cannot be reported as achieved gain.
No proposals are generated and no prediction is changed.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_rescan_reversible_surface_v2 import read,save,sha,load,database
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.reversible_surface_v2 import costs


def proposals(path):
    data=path.read_bytes();n=int(np.frombuffer(data[:4],'<i4')[0]);counts=np.frombuffer(data[4:4+4*n],'<i4');off=4+4*n;pool=[]
    for count in counts:
        v=np.frombuffer(data[off:off+int(count)*68],'<f4').reshape(-1,17).copy();off+=int(count)*68;pool.append(v)
    assert off==len(data)
    return pool


def main():
    root=Path('results/rescan_reversible_surface_v2');cfg=read(root/'config.json');seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    scene='scene_a';H,O,raw,initial,_=load(scene,cfg);uid_index={h['uid']:i for i,h in enumerate(H)}
    native=Path('results/rescan_third_period_progress_v1')/scene/'initial_archive'
    objs,_=database(native/'proposals.rsdb');pool=proposals(native/'proposals/proposals.bin');pool={o['uid']:v for o,v in zip(objs,pool)}
    truth=read_ply(Path('data/rescan_third_capture_v1/evaluation/raw')/scene/'gt_segmentation'/f'{scene}_2.ply')['instance_idx']
    rows=[]
    for method in ['hard_forward','joint_unrooted','rooted_forward','rooted_joint']:
        dest=root/scene/method;state=np.load(dest/'state.npz');z=np.load(dest/'source_for2.npz')
        for uid in [9,17,24]:
            i=uid_index[uid];mask=truth==uid
            obs={'xyz':raw[1]['xyz'][mask],'rgb':raw[1]['rgb'][mask].astype(float)/255,'class':raw[1]['predicted_class'][mask]}
            x=z[f'xyz_{i}'];inc={'xyz':x,'rgb':z[f'rgb_{i}'],'source_voxel':z[f'source_voxel_{i}'],'tree':cKDTree(x) if len(x) else None}
            candidates=[('retained',state['poses'][1,i]),('native_initial',initial[1][i])]
            candidates += [(f'cached_{k}',v[:16].reshape(4,4,order='F').astype(float)) for k,v in enumerate(pool[uid])]
            vals=[]
            for name,T in candidates:
                cc,_,_,_=costs(obs,[T],[H[i]],[inc],cfg)
                vals.append({'candidate':name,'mean_cost':float(np.mean(cc)),'admissible_fraction':float(np.mean(cc<cfg['null_cost']))})
            rows.append({'scene':scene,'method':method,'truth_id':uid,'cached_candidates':len(pool[uid]),
                         'retained':vals[0],'best_mean_cost':min(vals,key=lambda r:r['mean_cost']),
                         'maximum_admissible_fraction':max(r['admissible_fraction'] for r in vals)})
            print('POOL PROBE',method,uid,rows[-1]['retained']['admissible_fraction'],rows[-1]['maximum_admissible_fraction'],flush=True)
    save(root/'pose_pool_probe.json',{'rows':rows,'source_sha':sha(__file__),
        'proposal_sha':sha(native/'proposals/proposals.bin'),
        'scope':'Post-seal evaluator-only analysis of three cases declared from previous turn. GT isolates target points and selects best diagnostic cost. This is NOT an algorithm result, not equal-budget reranking, not an accuracy upper bound or displacement validation.'})


if __name__=='__main__':main()
