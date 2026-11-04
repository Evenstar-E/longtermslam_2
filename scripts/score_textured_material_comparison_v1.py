"""Private labels are first read here, after the complete prediction seal exists."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.textured_material_data_io_v1 import load_observation
from inspection_decisions.textured_material_matching_v1 import METHODS, apply, world_points, ray_state
from scripts.run_textured_material_comparison_v1 import sha, save


def check_inputs(root, protocol):
    checks=0
    seal=json.loads((root/'prediction_seal.json').read_text())
    assert seal['complete'] and seal['scenes']==protocol['scenes'];checks+=1
    for p,h in seal['sources'].items():
        assert sha(p)==h,p;checks+=1
    for p,h in seal['files'].items():
        assert sha(root/p)==h,p;checks+=1
    data=Path(protocol['config']['dataset'])
    assert sha(data/'release_manifest.json')==seal['dataset_release_sha'];checks+=1
    release=json.loads((data/'release_manifest.json').read_text())
    for p,h in release['files'].items():
        assert sha(data/p)==h,p;checks+=1
    # A real modality isolation check, not a self-reported label boundary.
    for scene in seal['scenes']:
        with np.load(root/scene/'flat_geometry.npz') as f:
            slots=f['target_slots'];T=f['transforms']
        for v in protocol['config']['variants'][1:]:
            with np.load(root/scene/f'{v}_geometry.npz') as f:
                assert np.array_equal(f['target_slots'],slots) and np.array_equal(f['transforms'],T);checks+=1
        for v in protocol['config']['variants']:
            for m in METHODS:
                with np.load(root/scene/f'{v}_{m}.npz') as f:
                    assert f['transforms'].shape==(24,2,4,4);checks+=1
                    assert np.isfinite(f['transforms']).all();checks+=1
                    for e in range(2):
                        assert np.array_equal(np.sort(f['target_slots'][:,e]),np.arange(24));checks+=1
                    R=f['transforms'][:,:,:3,:3]
                    assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<1e-8;checks+=1
    return checks


def material_anchors(scene_dir):
    buckets=[[] for _ in range(24)]
    records=json.loads((scene_dir/'frames.json').read_text())
    for r in records:
        if r['epoch']!=0:continue
        with np.load(scene_dir/r['labels']) as f:
            ids=f['instance'].ravel();q=f['material_xyz'].reshape(-1,3)
            for i in range(24):
                v=q[ids==i]
                if len(v):buckets[i].append(v[np.linspace(0,len(v)-1,min(64,len(v)),dtype=int)])
    out=[]
    for b in buckets:
        q=np.concatenate(b)
        out.append(q[np.linspace(0,len(q)-1,min(64,len(q)),dtype=int)])
    return out


def task_rows(scene_name, split, variant, method, pred, truth, anchors, cfg):
    poses=np.array(truth['poses']);perm=np.array(truth['permutations'])
    rows=[]
    for i in range(24):
        q=anchors[i];x=apply(poses[0,i],q)
        estimated=[x,apply(pred['transforms'][i,0],x),apply(pred['transforms'][i,1],x)]
        real=[apply(poses[e,i],q) for e in range(3)]
        errs=[np.quantile(np.linalg.norm((estimated[b]-estimated[a])-(real[b]-real[a]),axis=1),.95)
              for a,b in [(0,1),(0,2),(1,2)]]
        idok=pred['target_slots'][i]==perm[1:,i]
        rows.append(dict(scene=scene_name,split=split,variant=variant,method=method,asset=i,
             id1=int(idok[0]),id2=int(idok[1]),both_ids=int(idok.all()),
             error_h1_m=float(errs[0]),error_h2_m=float(errs[1]),error_12_m=float(errs[2]),
             worst_error_m=float(max(errs)),qualified=int(idok.all() and max(errs)<=cfg['joint_success_m']),
             changed=int(i in truth['changed_assets']),triplets=int(pred['triplet_counts'][i])))
    return rows


def nearest_targets(obs):
    out={}
    for slot in np.unique(obs['anonymous_slot']):
        if slot<0:continue
        yy,xx=np.nonzero((obs['anonymous_slot']==slot)&obs['depth_valid'])
        if len(xx):
            uv=np.c_[xx,yy]
            out[int(slot)]=(cKDTree(world_points(obs,uv)),uv)
    return out


def pair_stats(scene_dir, variant, method, pred, records, pairs, cfg, cache):
    results=[]
    # Rows indexed by historical reference; inverse maps a current spatial slot.
    mappings=np.c_[np.arange(24),pred['target_slots']]
    inv=np.argsort(mappings,axis=0)
    GG=np.tile(np.eye(4),(24,3,1,1));GG[:,1:]=pred['transforms']
    for pi,pair in enumerate(pairs):
        ra,rb=records[pair['source']],records[pair['target']]
        ea,eb=ra['epoch'],rb['epoch']
        source,target=cache[ra['index']]['obs'],cache[rb['index']]['obs']
        trees=cache[rb['index']]['trees']
        with np.load(scene_dir/pair['labels']) as f:
            labels={k:f[k] for k in f.files}
        uv=labels['source_uv'];x=world_points(source,uv)
        source_slots=source['anonymous_slot'][uv[:,1],uv[:,0]]
        accepted=np.zeros(len(x),bool);correct=accepted.copy()
        material_error=np.full(len(x),np.inf)
        for slot in np.unique(source_slots):
            if slot<0:continue
            ii=np.flatnonzero(source_slots==slot)
            i=int(inv[int(slot),ea]);dest=int(mappings[i,eb])
            if dest not in trees:continue
            T=GG[i,eb]@np.linalg.inv(GG[i,ea])
            xp=apply(T,x[ii]);dist,j=trees[dest][0].query(xp)
            good=dist<=cfg['accept_distance_m']
            if method in ('fusion_visibility','persistent_visible'):
                state,_=ray_state(xp,[target],dest,cfg)
                good&=state==1
            xy=trees[dest][1][j];tx,ty=xy.T
            tgt=cache[rb['index']]['truth']
            material_error[ii]=np.linalg.norm(tgt['material_xyz'][ty,tx]-labels['material_xyz'][ii],axis=1)
            accepted[ii]=good
            correct[ii]=good & (labels['status'][ii]==0) & (tgt['instance'][ty,tx]==labels['instance'][ii]) & (material_error[ii]<=cfg['correspondence_tolerance_m'])
        status=labels['status'];changed=labels['appearance_changed_in_changed_variant'] & (variant=='changed')
        groups=dict(visible=status==0,occluded=status==2,outside=status==1,
                    repaint_visible=(status==0)&changed,unchanged_visible=(status==0)&~changed)
        for name,mask in groups.items():
            results.append(dict(scene=scene_dir.name,variant=variant,method=method,pair=pi,condition=name,
                total=int(mask.sum()),accepted=int((accepted&mask).sum()),correct=int((correct&mask).sum()),
                false_accepted=int((accepted&~correct&mask).sum())))
    return results


def summarize(rows, pairs, cfg):
    summaries=[]
    for split in ['train','validation','test','all']:
        for v in cfg['variants']:
            for m in METHODS:
                rr=[r for r in rows if r['variant']==v and r['method']==m and (split=='all' or r['split']==split)]
                if not rr:continue
                scenes={r['scene'] for r in rr};ids=[r for r in rr if r['both_ids']]
                pr=[r for r in pairs if r['scene'] in scenes and r['variant']==v and r['method']==m]
                counts={}
                for condition in ['visible','occluded','outside','repaint_visible','unchanged_visible']:
                    qq=[r for r in pr if r['condition']==condition]
                    cc={k:sum(r[k] for r in qq) for k in ['total','accepted','correct','false_accepted']}
                    cc['correct_recall']=cc['correct']/cc['total'] if cc['total'] else None
                    cc['false_acceptance_rate']=cc['false_accepted']/cc['total'] if cc['total'] else None
                    counts[condition]=cc
                vv=list(counts[c] for c in ['visible','occluded','outside'])
                acc=sum(c['accepted'] for c in vv);correct=sum(c['correct'] for c in vv)
                summaries.append(dict(split=split,variant=v,method=m,scenes=len(scenes),tasks=len(rr),
                    both_ids=sum(r['both_ids'] for r in rr),qualified=sum(r['qualified'] for r in rr),
                    correct_id_error_12_median_m=float(np.median([r['error_12_m'] for r in ids])) if ids else None,
                    correct_id_worst_error_median_m=float(np.median([r['worst_error_m'] for r in ids])) if ids else None,
                    accepted_precision=correct/acc if acc else None,accepted=acc,correct=correct,correspondence=counts))
    comparisons=[]
    for split in ['train','validation','test','all']:
        for v in cfg['variants']:
            base={(r['scene'],r['asset']):r for r in rows if r['method']=='fusion' and r['variant']==v and (split=='all' or r['split']==split)}
            for m in ['fusion_visibility','fusion_persistence','persistent_visible']:
                cand={(r['scene'],r['asset']):r for r in rows if r['method']==m and r['variant']==v and (split=='all' or r['split']==split)}
                keys=base.keys()&cand.keys();common=[k for k in keys if base[k]['both_ids'] and cand[k]['both_ids']]
                comparisons.append(dict(split=split,variant=v,method=m,common_correct_ids=len(common),
                    gained=sum(cand[k]['qualified'] and not base[k]['qualified'] for k in keys),
                    lost=sum(base[k]['qualified'] and not cand[k]['qualified'] for k in keys),
                    common_error_12_baseline_median_m=float(np.median([base[k]['error_12_m'] for k in common])) if common else None,
                    common_error_12_candidate_median_m=float(np.median([cand[k]['error_12_m'] for k in common])) if common else None,
                    common_worst_better=sum(cand[k]['worst_error_m']<base[k]['worst_error_m']-1e-9 for k in common),
                    common_worst_worse=sum(cand[k]['worst_error_m']>base[k]['worst_error_m']+1e-9 for k in common)))
    return dict(summary=summaries,paired_comparisons=comparisons)


def write_csv(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/textured_material_comparison_v1');args=ap.parse_args()
    root=Path(args.root);protocol=json.loads((root/'frozen_protocol.json').read_text());cfg=protocol['config']
    checks=check_inputs(root,protocol);rows=[];pairs_out=[]
    for scene_name in protocol['scenes']:
        scene_dir=Path(cfg['dataset'])/scene_name
        truth=json.loads((scene_dir/'supervision/scene.json').read_text())
        split=json.loads((scene_dir/'public_metadata.json').read_text())['split']
        records=json.loads((scene_dir/'frames.json').read_text());anchors=material_anchors(scene_dir)
        pairs=json.loads((scene_dir/'supervision/pairs_v3.json').read_text())
        # Geometry/depth and labels share all four variants; RGB not needed by scorer.
        cache={}
        for r in records:
            obs=load_observation(scene_dir/r['observation'],'flat')
            with np.load(scene_dir/r['labels']) as f:gt={k:f[k] for k in ['instance','material_xyz']}
            cache[r['index']]=dict(obs=obs,truth=gt,trees=nearest_targets(obs))
        for v in cfg['variants']:
            for m in METHODS:
                with np.load(root/scene_name/f'{v}_{m}.npz') as f:pred=dict(f)
                rows+=task_rows(scene_name,split,v,m,pred,truth,anchors,cfg)
                pairs_out+=pair_stats(scene_dir,v,m,pred,records,pairs,cfg,cache)
        print('SCORED',scene_name,flush=True)
    write_csv(root/'task_scores.csv',rows);write_csv(root/'correspondence_scores.csv',pairs_out)
    agg=summarize(rows,pairs_out,cfg);save(root/'scores.json',agg)
    save(root/'scoring_audit.json',dict(checks=checks,all_predictions_sealed_before_scoring=True,
          dataset_files_unchanged=True,geometry_invariant_across_rgb=True,
          scorer_sha=sha(__file__),prediction_seal_sha=sha(root/'prediction_seal.json'),
          task_rows=len(rows),pair_condition_rows=len(pairs_out),
          hashes={p:sha(root/p) for p in ['task_scores.csv','correspondence_scores.csv','scores.json']}))
    print('SCORING COMPLETE',checks,'integrity checks',flush=True)


if __name__=='__main__':main()
