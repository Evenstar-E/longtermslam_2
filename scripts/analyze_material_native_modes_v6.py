"""Post-seal mechanism diagnosis. No prediction or inference parameter updates."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import csv,json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges
from scripts.run_textured_material_comparison_v1 import save,sha
from scripts.score_textured_material_comparison_v1 import write_csv


def quantile(a):return np.quantile(a,[.5,.95,1]).tolist() if len(a) else None


def main():
    cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));root=Path(cfg['output']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));base=proto['base_config'];data=Path(base['dataset'])
    for folder in [root,Path('results/material_mode_intersection_v6')]:assert json.loads((folder/'prediction_seal.json').read_text(encoding='utf-8'))['complete']
    records=[];merge_dist=[];crops=[]
    for name in proto['scenes']:
        truth=json.loads((data/name/'supervision/scene.json').read_text(encoding='utf-8'));perm=np.array(truth['permutations']);fr=json.loads((data/name/'frames.json').read_text(encoding='utf-8'));fr={r['index']:r for r in fr};labels={}
        for v in base['variants']:
            ids=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=base['seed']+e)[1] for e in range(3)]
            with np.load(Path(cfg['native_root'])/name/f'{v}_native_features.npz') as f:ff=dict(f)
            members={};counts=np.zeros((3,24),int)
            for prefix in [k[:-6] for k in ff if k.endswith('_frame')]:
                e,i,k=map(int,prefix.split('_'));fi=ids[e][int(ff[prefix+'_frame'])]
                if fi not in labels:
                    with np.load(data/name/fr[fi]['labels']) as f:labels[fi]={'local':f['material_xyz'],'owner':f['instance']}
                p=ff[prefix+'_pixel'];q=ff[prefix+'_quality'];n=ff[prefix+'_node'];local=labels[fi]['local'][p[:,1],p[:,0]];owner=labels[fi]['owner'][p[:,1],p[:,0]]
                assert (owner==int(np.flatnonzero(perm[e]==i)[0])).all()
                crop=ff[prefix+'_crop'];crops.append(dict(variant=v,detail=int(int(np.flatnonzero(perm[e]==i)[0]) in truth['detail_assets']),long_side_px=int(max(crop[2]-crop[0],crop[3]-crop[1])),keypoints=len(p)))
                for node,point,quality in zip(n,local,q):members.setdefault((e,i,int(node)),[]).append((float(quality),point))
                counts[e,i]+=len(p)
            material={key:max(mm,key=lambda x:x[0])[1] for key,mm in members.items()}
            for mm in members.values():
                if len(mm)==2:merge_dist.append(float(np.linalg.norm(mm[0][1]-mm[1][1])))
            with np.load(Path(cfg['native_root'])/name/f'{v}_native_edges.npz') as f:edges=unpack_edges(dict(f),cfg)
            with np.load(root/name/f'{v}_material_modes.npz') as f:modes=dict(f)
            mi={tuple(key):h for h,key in enumerate(modes['keys'])}
            for i in range(24):
                slots=[i,int(perm[1,i]),int(perm[2,i])];pairdist=[];ne=[]
                for pe,(e,f) in enumerate([(0,1),(0,2),(1,2)]):
                    a,b=slots[e],slots[f];r=edges[pe][a*24+b]
                    d=[float(np.linalg.norm(material[e,a,int(s)]-material[f,b,int(t)])) for s,t in zip(r['source'],r['target'])];pairdist+=d;ne.append(len(d))
                h=mi.get(tuple(slots));td=[]
                if h is not None:
                    for node in modes[f'{h}_node']:
                        x=np.array([material[e,slots[e],int(node[e])] for e in range(3)]);td.append(float(max(np.linalg.norm(x[a]-x[b]) for a,b in [(0,1),(0,2),(1,2)])))
                records.append(dict(scene=name,variant=v,asset=i,detail=int(i in truth['detail_assets']),native_points_min_epoch=int(min(counts[e,slots[e]] for e in range(3))),
                    edge_h1=ne[0],edge_h2=ne[1],edge_12=ne[2],true_pair_edges=len(pairdist),good_pair_edges_15mm=sum(x<=.015 for x in pairdist),
                    pair_material_error_median_m=float(np.median(pairdist)) if pairdist else None,tracks=len(td),good_tracks_15mm=sum(x<=.015 for x in td),
                    track_material_error_median_m=float(np.median(td)) if td else None))
            print('V6 POSTSEAL MATERIAL DIAGNOSIS',name,v,flush=True)
    write_csv(root/'native_material_diagnosis.csv',records)
    with (root/'task_scores.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    summary=[];parts=[];per_scene=[];paired=[]
    ms=proto['methods']+['shared_intersection']
    for v in base['variants']:
        for detail in [0,1]:
            r=[a for a in records if a['variant']==v and a['detail']==detail];cc=[a for a in crops if a['variant']==v and a['detail']==detail]
            summary.append(dict(variant=v,detail=detail,assets=len(r),assets_three_tracks=sum(x['tracks']>=3 for x in r),assets_three_good_tracks=sum(x['good_tracks_15mm']>=3 for x in r),
                true_identity_edge_count=sum(x['true_pair_edges'] for x in r),good_material_edge_count=sum(x['good_pair_edges_15mm'] for x in r),
                true_identity_track_count=sum(x['tracks'] for x in r),good_material_track_count=sum(x['good_tracks_15mm'] for x in r),
                native_points_min_epoch_quantiles=quantile([x['native_points_min_epoch'] for x in r]),original_crop_long_side_quantiles=quantile([x['long_side_px'] for x in cc])))
        for m in ms:
            a=[x for x in rr if x['variant']==v and x['method']==m]
            p=dict(variant=v,method=m,qualified=sum(int(x['qualified']) for x in a),no_good_pose=sum(not int(x['oracle_pool_qualified']) for x in a),good_pose_wrong_id=sum(int(x['oracle_pool_qualified']) and not int(x['both_ids']) for x in a),good_pose_correct_id_bad_selection=sum(int(x['oracle_pool_qualified']) and int(x['both_ids']) and not int(x['qualified']) for x in a))
            assert sum(p[k] for k in ['qualified','no_good_pose','good_pose_wrong_id','good_pose_correct_id_bad_selection'])==96;parts.append(p)
            for name in proto['scenes']:per_scene.append(dict(scene=name,variant=v,method=m,qualified=sum(int(x['qualified']) for x in a if x['scene']==name),both_ids=sum(int(x['both_ids']) for x in a if x['scene']==name)))
        for a,b in [('old_pool','native_graph'),('native_graph','joint_all'),('joint_all','shared_modes'),('cycle_independent','shared_intersection')]:
            left={(x['scene'],x['asset']):x for x in rr if x['variant']==v and x['method']==a};right={(x['scene'],x['asset']):x for x in rr if x['variant']==v and x['method']==b}
            common=[k for k in left if int(left[k]['both_ids']) and int(right[k]['both_ids'])];g=[k for k in left if not int(left[k]['qualified']) and int(right[k]['qualified'])];l=[k for k in left if int(left[k]['qualified']) and not int(right[k]['qualified'])]
            paired.append(dict(variant=v,baseline=a,method=b,gained=g,lost=l,common_correct_ids=len(common),baseline_error_median_m=float(np.median([float(left[k]['worst_error_m']) for k in common])) if common else None,candidate_error_median_m=float(np.median([float(right[k]['worst_error_m']) for k in common])) if common else None))
    save(root/'mechanism_analysis.json',dict(source_sha=sha(__file__),postseal_diagnostic_only=True,no_predictions_changed=True,thresholds='15mm local material coordinate distance is diagnosis only, not method tuning',summary=summary,same_epoch_merge_material_distance_quantiles=quantile(merge_dist),failure_partition=parts,per_scene=per_scene,paired=paired))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
