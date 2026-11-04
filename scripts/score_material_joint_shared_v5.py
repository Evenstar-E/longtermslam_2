"""Audit common search graph/cache, then score private material anchors."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from scripts.score_material_continuous_v4 import oracle_fast
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.score_material_joint_descriptor_comparison_v1 import aggregate
from inspection_decisions.material_shared_correspondence_factor_v5 import material_energies
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    config=json.loads(Path('configs/material_joint_shared_v5.json').read_text(encoding='utf-8'));root=Path(config['output']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h;checks+=1
    assert sha(proto['feature_checkpoint'])==proto['feature_checkpoint_sha'];checks+=1
    parent=Path(proto['v4']['comparison_root']);assert sha(parent/'prediction_seal.json')==proto['parent_prediction_seal_sha'];checks+=1
    data=Path(proto['base_config']['dataset']);assert sha(data/'release_manifest.json')==proto['dataset_release_sha'];checks+=1
    release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    rows=[];factor=[]
    for name in proto['scenes']:
        truth=json.loads((data/name/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(data/name);split=json.loads((data/name/'public_metadata.json').read_text(encoding='utf-8'))['split']
        for v in proto['base_config']['variants']:
            assert sha(root/name/f'{v}_union_pools.npz')==sha(parent/name/f'{v}_union_pools.npz');checks+=1
            with np.load(root/name/f'{v}_union_pools.npz') as f:pool=dict(f)
            oracle=oracle_fast(truth,pool,anchors)
            with np.load(root/name/f'{v}_search_graph.npz') as f:graph=dict(f)
            with np.load(root/name/f'{v}_joint_cache.npz') as f:cache=dict(f)
            lookup={tuple(k):i for i,k in enumerate(cache['triples'])}
            for s,c in zip(graph['states'],graph['costs']):
                assert all(np.array_equal(np.sort(s[:,e]),np.arange(24)) for e in [0,1]);checks+=2
                rebuild=sum((cache['costs'][lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(4))
                assert np.max(abs(rebuild-c))<1e-10;checks+=4
            assert np.array_equal(graph['best'],graph['costs'].argmin(0));checks+=1
            assert np.max(np.diff(graph['paths'],axis=0))<=1e-10;checks+=1
            for h,(i,j,k) in enumerate(cache['triples']):
                for col in range(4):
                    for pe,a,b,T in [(0,i,j,cache['transforms'][h,col,0]),(1,i,k,cache['transforms'][h,col,1])]:
                        pp=pool['transforms'][pe,a,b,:pool['counts'][pe,a,b]]
                        assert np.min(np.max(abs(pp-T),axis=(1,2)))<1e-10;checks+=1
            with np.load(root/name/f'{v}_selected_material.npz') as f:material=dict(f)
            for m in proto['methods']:
                col=0 if m.startswith('ordinary') else ['ordinary','fusion','independent','shared'].index(m)
                with np.load(root/name/f'{v}_{m}.npz') as f:pred=dict(f)
                state=int(pred['search_state']);assert np.array_equal(pred['target_slots'],graph['states'][state]);checks+=1
                expected=int(graph['costs'][:int(graph['initial_count']),0].argmin()) if m=='ordinary_frozen_ids' else int(graph['best'][col]);assert state==expected;checks+=1
                for i,(j,k) in enumerate(pred['target_slots']):
                    h=lookup[i,int(j),int(k)];assert abs(pred['objectives'][i]-cache['costs'][h,col])<1e-10;checks+=1
                    assert np.max(abs(pred['transforms'][i]-cache['transforms'][h,col]))<1e-10;checks+=1
                    if m in ['fusion','independent','shared']:
                        a,b=material[f'{m}_{i}_a'],material[f'{m}_{i}_b'];alpha=material[f'{m}_{i}_alpha'];energies=material_energies(a[None],b[None],np.ones(8)/8,alpha)
                        za=a.mean(1);zb=b.mean(1);z=(a*b).mean(1);psi=-np.log(z/(za*zb));ind=float(energies[1][0,0]);shared=float(energies[2][0,0])
                        assert abs(shared-ind-float((alpha*psi).sum()))<1e-10;checks+=1
                        factor.append(dict(scene=name,variant=v,method=m,asset=i,independent=ind,shared=shared,shared_increment=shared-ind,weighted_increment=config['material_weight']*(shared-ind),
                            mean_abs_node_variation=float(np.mean(np.std(a,axis=1)/za)),node_variation_bound=float(np.sum(alpha*(a.std(1)/za)*(b.std(1)/zb)))))
                rr=task_rows(name,split,v,m,pred,truth,anchors,proto['legacy_config'])
                for i,r in enumerate(rr):r.update(oracle_pool_worst_error_m=oracle[i],oracle_pool_qualified=int(oracle[i]<=.05),detail=int(i in truth['detail_assets']))
                rows+=rr
            print('V5 AUDITED/SCORED',name,v,flush=True)
    write_csv(root/'task_scores.csv',rows);write_csv(root/'factor_diagnostics.csv',factor)
    scores=aggregate(rows,[],proto['methods'],proto['base_config']['variants'],proto['controls']);save(root/'scores.json',scores)
    save(root/'scoring_audit.json',dict(checks=checks,source_sha=sha(__file__),all_predictions_sealed_before_truth=True,identical_complete_parent_pose_pools=True,
        common_search_states_verified=True,selected_cache_minima_verified=True,task_rows=len(rows),
        hashes={p:sha(root/p) for p in ['scores.json','task_scores.csv','factor_diagnostics.csv']}))
    print('V5 SCORE COMPLETE',checks,len(rows),flush=True)


if __name__=='__main__':main()
