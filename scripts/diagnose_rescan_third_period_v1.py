"""Post-seal archive provenance and failure localization, never inference input."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.run_rescan_third_period_v1 import win, save, sha
from scripts.prepare_real_temporal_v1 import read_ply
from scripts.prepare_rescan_three_period_v1 import verify_runs,run_root


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def xyz(p):return np.column_stack([p[f] for f in ['x','y','z']]).astype(float)
def objects(db):
    lines=db.read_text(encoding='utf-8').splitlines()
    folder=win(next(x.split(maxsplit=1)[1] for x in lines if x.startswith('model_folder ')))
    return [{'uid':int(t[2]),'path':folder/t[1]} for t in map(str.split,lines) if t and t[0]=='object']


def proposals(path):
    data=path.read_bytes();n=int(np.frombuffer(data[:4],'<i4')[0]);counts=np.frombuffer(data[4:4+4*n],'<i4')
    offset=4+4*n;out=[]
    for count in counts:
        v=np.frombuffer(data[offset:offset+int(count)*68],'<f4').reshape(-1,17).copy()
        offset+=int(count)*68;out.append(v)
        if len(v):assert np.allclose(v[:,:16].reshape(-1,4,4)[:,:,3], [0,0,0,1],atol=1e-5)
    assert offset==len(data)
    return out


def subsample(x):
    _,idx=np.unique(np.floor(x/.05).astype(np.int64),axis=0,return_index=True)
    idx=np.sort(idx)
    if len(idx)>1600:idx=idx[np.linspace(0,len(idx)-1,1600).astype(int)]
    return x[idx]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--progress',action='store_true');args=ap.parse_args()
    base=verify_runs(progress=args.progress)
    if args.progress:base=Path('results/rescan_third_period_progress_v1')
    cfg=read('configs/rescan_third_period_v1.json');score=read(base/'scores.json');memory=[];failures=[]
    for scene in cfg['scenes']:
        prior=Path(cfg['prior_roots'][scene]);initial={o['uid']:o for o in objects(prior/'history.rsdb')}
        inherited=objects(prior/'prediction.rsdb')
        oldpred=read_ply(prior/'predictions/prediction.ply')
        oldtruth=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply')
        assert np.array_equal(xyz(oldpred),xyz(oldtruth))
        oldeq={}
        oldmeta=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        for line in (oldmeta.read_text(encoding='utf-8').splitlines() if oldmeta.exists() else []):
            if line.strip():
                left,right=line.split('|');oldeq[int(left)]=[int(v) for v in right.split()]
        for obj in inherited:
            uid=obj['uid']
            if uid not in initial:continue
            x,y=xyz(read_ply(initial[uid]['path'])),xyz(read_ply(obj['path']))
            mask=oldpred['instance_idx']==uid
            ids,n=np.unique(oldtruth['instance_idx'][mask],return_counts=True)
            d=cKDTree(x).query(y)[0]
            memory.append({'scene':scene,'archive_id':uid,'initial_points':len(x),'updated_points':len(y),
                'updated_points_outside_initial_1mm_fraction':float(np.mean(d>.001)),
                'period1_predicted_support_points':int(mask.sum()),
                'period1_support_strict_same_id_fraction':None if not mask.any() else float(np.mean(oldtruth['instance_idx'][mask]==uid)),
                'period1_support_allowed_identity_fraction':None if not mask.any() else float(sum(int(count) for true_id,count in zip(ids,n) if uid in oldeq.get(int(true_id),[int(true_id)]))/mask.sum()),
                'period1_support_truth_distribution':{str(int(i)):int(c) for i,c in zip(ids,n)},
                'scope':'All-point strict identity composition, not equivalence-adjusted contamination rate. Updated distance includes correct newly visible surfaces, wrong IDs, alignment and sensor noise.'})
        truth=read_ply(Path('data/rescan_third_capture_v1/evaluation/raw')/scene/'gt_segmentation'/f'{scene}_2.ply')
        ids=sorted({r['truth_id'] for r in score['rows'] if r['scene']==scene and r['strict_initial_persistent'] and not r['strict_correct']})
        for mode in cfg['modes']+['inherited_protected']:
            root=run_root(scene,mode,args.progress);objs=objects(root/'proposals.rsdb');pool=proposals(root/'proposals/proposals.bin')
            assert len(objs)==len(pool)
            for uid in ids:
                oi=next((i for i,o in enumerate(objs) if o['uid']==uid),None)
                if oi is None:
                    failures.append({'scene':scene,'mode':mode,'truth_id':uid,'candidate_count':0,'missing_template':True});continue
                x=subsample(xyz(read_ply(objs[oi]['path'])));y=subsample(xyz(truth[truth['instance_idx']==uid]));tx,ty=cKDTree(x),cKDTree(y)
                best=None
                for k,v in enumerate(pool[oi]):
                    T=v[:16].reshape(4,4,order='F').astype(float);R,t=T[:3,:3],T[:3,3]
                    a=ty.query(x@R.T+t)[0];b=tx.query((y-t)@R)[0];tau=.075
                    cost=float((np.mean(np.minimum(a/tau,1)**2)+np.mean(np.minimum(b/tau,1)**2))/2)
                    record={'proposal_index':k,'cost':cost,'model_to_target_support':float(np.mean(a<tau)),
                            'target_to_model_support':float(np.mean(b<tau)),'author_score':float(v[16])}
                    if best is None or cost<best['cost']:best=record
                failures.append({'scene':scene,'mode':mode,'truth_id':uid,'candidate_count':len(pool[oi]),'best_diagnostic_surface_fit':best})
                print('DIAG',scene,mode,uid,len(pool[oi]),None if best is None else round(best['cost'],4),flush=True)
    save(base/'archive_provenance_diagnosis.json',{'rows':memory,'source_sha':sha(__file__),
        'code_mechanism':'database_update.cpp extracts by predicted ID, aligns, merges, overwrites all instance IDs; rs_pointcloud_merge concatenates then shuffles. No probabilistic provenance retained.'})
    save(base/'proposal_failure_diagnosis.json',{'rows':failures,'scope':'After predictions and evaluation. All strict-persistent IDs failing at least one arm. 5cm voxel/max1600points,75mm symmetric truncated NN. GT target isolating only; not original layout energy or independent motion accuracy.','source_sha':sha(__file__)})


if __name__=='__main__':main()
