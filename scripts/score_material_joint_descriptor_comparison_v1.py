"""Same fixed material endpoints, private scoring only after prediction seal."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_observation
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,nearest_targets,pair_stats,write_csv
from scripts.run_textured_material_comparison_v1 import save,sha


def aggregate(rows,pairs,methods,variants,controls=None):
    result=[];paired=[]
    for split in ['validation','test','all']:
        for v in variants:
            for m in methods:
                rr=[r for r in rows if r['variant']==v and r['method']==m and (split=='all' or r['split']==split)]
                if not rr:continue
                scenes={r['scene'] for r in rr};good=[r for r in rr if r['both_ids']]
                counts={}
                for condition in ['visible','outside','occluded','repaint_visible','unchanged_visible']:
                    qq=[r for r in pairs if r['scene'] in scenes and r['variant']==v and r['method']==m and r['condition']==condition]
                    cc={k:sum(r[k] for r in qq) for k in ['total','accepted','correct','false_accepted']}
                    cc['recall']=cc['correct']/cc['total'] if cc['total'] else None
                    cc['false_rate']=cc['false_accepted']/cc['total'] if cc['total'] else None
                    counts[condition]=cc
                acc=sum(counts[c]['accepted'] for c in ['visible','outside','occluded']);correct=sum(counts[c]['correct'] for c in ['visible','outside','occluded'])
                result.append(dict(split=split,variant=v,method=m,tasks=len(rr),scenes=len(scenes),
                    both_ids=sum(r['both_ids'] for r in rr),qualified=sum(r['qualified'] for r in rr),
                    oracle_pool_qualified=sum(r['oracle_pool_qualified'] for r in rr),
                    correct_id_error_12_median_m=float(np.median([r['error_12_m'] for r in good])) if good else None,
                    accepted_precision=correct/acc if acc else None,correspondence=counts))
            for control in (controls or [dict(baseline='learned_fusion',candidate=m) for m in ['learned_visibility','learned_material']]):
                m=control['candidate']
                base={(r['scene'],r['asset']):r for r in rows if r['method']==control['baseline'] and r['variant']==v and (split=='all' or r['split']==split)}
                cand={(r['scene'],r['asset']):r for r in rows if r['method']==m and r['variant']==v and (split=='all' or r['split']==split)}
                common=[k for k in base if base[k]['both_ids'] and cand[k]['both_ids']]
                paired.append(dict(split=split,variant=v,baseline=control['baseline'],method=m,common_correct_ids=len(common),
                    gained=sum(cand[k]['qualified'] and not base[k]['qualified'] for k in base),
                    lost=sum(base[k]['qualified'] and not cand[k]['qualified'] for k in base),
                    baseline_error_12_median_m=float(np.median([base[k]['error_12_m'] for k in common])) if common else None,
                    candidate_error_12_median_m=float(np.median([cand[k]['error_12_m'] for k in common])) if common else None))
    return dict(summary=result,paired_comparisons=paired)


def pool_oracle(scene,pool,anchors,cfg):
    poses=np.array(scene['poses']);perm=np.array(scene['permutations']);values=[]
    for i in range(24):
        q=anchors[i];x=apply(poses[0,i],q);y1=apply(poses[1,i],q);y2=apply(poses[2,i],q)
        p1=pool['transforms'][0,i,perm[1,i],:pool['counts'][0,i,perm[1,i]]]
        p2=pool['transforms'][1,i,perm[2,i],:pool['counts'][1,i,perm[2,i]]]
        best=np.inf
        for T1 in p1:
            for T2 in p2:
                z1,z2=apply(T1,x),apply(T2,x)
                worst=max(np.quantile(np.linalg.norm(z1-y1,axis=1),.95),np.quantile(np.linalg.norm(z2-y2,axis=1),.95),
                          np.quantile(np.linalg.norm((z2-z1)-(y2-y1),axis=1),.95))
                best=min(best,worst)
        values.append(best)
    return values


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/material_joint_descriptor_comparison_v1');args=ap.parse_args()
    root=Path(args.root);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));cfg=proto['base_config']
    seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];checks=1
    for p,h in seal['sources'].items():assert sha(p)==h;checks+=1
    for p,h in seal['files'].items():assert sha(root/p)==h;checks+=1
    data=Path(proto['config']['dataset']);assert sha(data/'release_manifest.json')==proto['dataset_release_sha'];checks+=1
    release=json.loads((data/'release_manifest.json').read_text(encoding='utf-8'))
    for p,h in release['files'].items():assert sha(data/p)==h;checks+=1
    rows=[];allpairs=[]
    for name in proto['scenes']:
        dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(dest)
        records=json.loads((dest/'frames.json').read_text(encoding='utf-8'));pairs=json.loads((dest/'supervision/pairs_v3.json').read_text(encoding='utf-8'))
        split=json.loads((dest/'public_metadata.json').read_text(encoding='utf-8'))['split'];cache={}
        for r in records:
            obs=load_observation(dest/r['observation'],'flat')
            with np.load(dest/r['labels']) as f:gt={k:f[k] for k in ['instance','material_xyz']}
            cache[r['index']]=dict(obs=obs,truth=gt,trees=nearest_targets(obs))
        for v in proto['config']['variants']:
            oracles={}
            for m in proto['methods']:
                feature=proto.get('method_features',{}).get(m,'full' if m.startswith('learned_') else 'superpoint' if m=='superpoint_lightglue' else m)
                if feature not in oracles:
                    with np.load(root/name/f'{v}_{feature}_pools.npz') as f:oracles[feature]=pool_oracle(truth,dict(f),anchors,cfg)
                with np.load(root/name/f'{v}_{m}.npz') as f:pred=dict(f)
                assert pred['transforms'].shape==(24,2,4,4) and np.isfinite(pred['transforms']).all();checks+=1
                for e in [0,1]:assert np.array_equal(np.sort(pred['target_slots'][:,e]),np.arange(24));checks+=1
                R=pred['transforms'][:,:,:3,:3];assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
                rr=task_rows(name,split,v,m,pred,truth,anchors,cfg)
                for i,r in enumerate(rr):
                    r['oracle_pool_worst_error_m']=oracles[feature][i]
                    r['oracle_pool_qualified']=int(oracles[feature][i]<=cfg['joint_success_m'])
                    r['detail']=int(i in truth['detail_assets'])
                rows+=rr
                alias='persistent_visible' if m in proto.get('visibility_methods',['learned_visibility','learned_material']) else m
                pp=pair_stats(dest,v,alias,pred,records,pairs,cfg,cache)
                for p in pp:p['method']=m
                allpairs+=pp
        print('LEARNED SCORED',name,flush=True)
    write_csv(root/'task_scores.csv',rows);write_csv(root/'correspondence_scores.csv',allpairs)
    save(root/'scores.json',aggregate(rows,allpairs,proto['methods'],proto['config']['variants'],proto.get('paired_controls')))
    save(root/'scoring_audit.json',dict(checks=checks,all_predictions_sealed_before_truth=True,
        scorer_sha=sha(__file__),prediction_seal_sha=sha(root/'prediction_seal.json'),rows=len(rows),
        pair_condition_rows=len(allpairs),hashes={p:sha(root/p) for p in ['task_scores.csv','correspondence_scores.csv','scores.json']}))
    print('LEARNED SCORING COMPLETE',checks,flush=True)


if __name__=='__main__':main()
