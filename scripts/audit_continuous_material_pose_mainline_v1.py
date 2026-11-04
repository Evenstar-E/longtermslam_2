"""Complete-pool coverage, unchanged baselines, selections and coordinate maps."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import transform
from inspection_decisions.continuous_material_pose_mainline_v1 import select


def main():
    root=Path('results/continuous_material_pose_mainline_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);checks=0;attempts=success=0;failures={};fallback=0
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;checks+=1
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h;checks+=1
    rows=read(root/'scores.json');lookup={(r['scene'],r['asset'],r['method']):r for r in rows}
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        logs=read(dest/'optimization_logs.json');pred=read(dest/'predictions.json');slots=inp['T0'][:,:3,3]
        expected={(i,t,k) for i in range(24) for t in range(2) for k in range(54)}
        assert len(logs)==len(expected)==2592 and {(l['asset'],l['epoch'],l['parent']) for l in logs}==expected;checks+=1;attempts+=len(logs)
        for i,bb in enumerate(candidates):
            for t,branches in enumerate(bb):
                old=inp['branches'][i][t];assert len(old)==54;checks+=1
                for k,b in enumerate(old):
                    actual=branches[k];assert actual['source']=='catalogue' and actual['parent']==k and actual['slot']==b['slot'] and actual['code']==b['code'] and np.array_equal(actual['T'],b['T']);checks+=1
                    assert abs(actual['scores']['nn']-b['cost'])<1e-10;checks+=1
                for b in branches:
                    T=b['T'];assert np.isfinite(T).all() and abs(np.linalg.det(T[:3,:3])-1)<1e-6;checks+=1
                    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6);checks+=1
                    assert all(np.isfinite(v) for v in b['scores'].values());checks+=1
                    assert -.00000001<=b['scores']['nn']<=1.00000001 and -.00000001<=b['scores']['plane']<=1.00000001 and -1.00000001<=b['scores']['visibility']<=1.00000001;checks+=1
                    for direction in b['support'].values():
                        assert direction['positive']+direction['absence']+direction['unknown']==direction['total'];checks+=1
        for log in logs:
            i,t,k=log['asset'],log['epoch'],log['parent'];index=log['endpoint_index'];assert log['complete']==(index is not None);checks+=1
            if index is not None:
                b=candidates[i][t][index];assert b['source']=='point_to_plane' and b['parent']==k and b['slot']==log['slot'];checks+=1
                assert len(log['trace'])==3 and all(v['error'] is None for v in log['trace']);checks+=1;success+=1
                local=np.array(log['trace'][-1]['T_local']);expected_T=local.copy();expected_T[:3,3]+=slots[b['slot']]
                assert np.array_equal(expected_T,b['T']);checks+=1
                anchor=inp['anchors'][i][None];assert np.allclose(transform(anchor,local)+slots[b['slot']],transform(anchor,b['T']));checks+=1
            else:
                assert log['trace'][-1]['error'] is not None;checks+=1
                error=log['trace'][-1]['error'];failures[error]=failures.get(error,0)+1
        for expanded,prefix in [(False,'catalogue'),(True,'expanded')]:
            for score in cfg['scores']:
                answer=select(candidates,score,expanded);actual=pred[prefix+'_'+score]
                assert actual==answer;checks+=1
                for t in [0,1]:
                    selected=[candidates[i][t][pair[t]]['slot'] for i,*pair in actual['selection']];assert len(set(selected))==24;checks+=1
        for i,k,l in pred['fixed_nn_ids_refined']['selection']:
            original=next(v for v in pred['catalogue_nn']['selection'] if v[0]==i)
            for t,j in enumerate([k,l]):
                assert candidates[i][t][j]['slot']==candidates[i][t][original[t+1]]['slot'];checks+=1
            assert lookup[(tag,i,'catalogue_nn')]['both_ids']==lookup[(tag,i,'fixed_nn_ids_refined')]['both_ids'];checks+=1
        fallback+=pred['fixed_nn_ids_refined']['fallback_epochs']
    assert attempts==20736 and len(rows)==192*10;checks+=1
    refinement=read(root/'true_pose_refinement.json');assert len(refinement)==384;checks+=1
    for oracle in read(root/'oracle.json'):assert oracle['good_correct_id_pairs']>=1;checks+=1
    result={'status':'PASS','checks':checks,'attempted_paths':attempts,'successful_endpoints':success,'failed_paths':attempts-success,'failure_reasons':failures,'fixed_nn_selected_epoch_fallbacks':fallback,
        'evaluation_and_analysis_sources':{str(p):sha(p) for p in [Path(__file__),Path('scripts/score_continuous_material_pose_mainline_v1.py'),Path('scripts/check_continuous_material_pose_mainline_v1.py')]}}
    save(root/'audit.json',result);print(result,flush=True)


if __name__=='__main__':main()
