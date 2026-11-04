"""Run point-level reversible surface experiment; never read current truth."""
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.reversible_surface_v2 import voxelize, run, make_surface, costs, transform


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def xyz(p):return np.column_stack([p[k] for k in ['x','y','z']]).astype(float)
def rgb(p):return np.column_stack([p[k] for k in ['red','green','blue']])


def database(path):
    rows=[r.split() for r in path.read_text(encoding='utf-8').splitlines() if r.strip()]
    folder=Path(next(r[1] for r in rows if r[0]=='model_folder').replace('/mnt/c/','C:/'))
    objs=[{'uid':int(r[2]),'class':int(r[3]),'path':folder/r[1]} for r in rows if r[0]=='object']
    poses={(int(r[1]),int(r[2])):np.array(r[5:],float).reshape(4,4) for r in rows if r[0]=='pose'}
    return objs,poses


def load(scene,cfg):
    old=read('configs/rescan_third_period_v1.json');prior=Path(old['prior_roots'][scene])
    third=Path('results/rescan_third_period_progress_v1' if scene=='scene_a' else 'results/rescan_third_period_v1_completed')/scene/'initial_archive'
    paths=[prior/'history.rsdb',prior/'prediction.rsdb',third/'prediction.rsdb']
    obj,hpose=database(paths[0]);_,p1=database(paths[1]);_,p2=database(paths[2])
    H=[];initial=[[],[]]
    for o in obj:
        p=read_ply(o['path']);paths.append(o['path'])
        if len(p)<6:continue
        z=voxelize(xyz(p),rgb(p),np.full(len(p),o['class']),cfg['voxel_m'])
        anchor=xyz(p)[np.argmin(np.linalg.norm(xyz(p)-xyz(p).mean(0),axis=1))]
        H.append({'uid':o['uid'],'class':o['class'],'xyz':z['xyz'],'rgb':z['rgb'],
                  'tree':cKDTree(z['xyz']),'anchor':anchor,'raw_points':len(p)})
        for t,pp in enumerate([p1,p2]):initial[t].append(pp.get((o['uid'],1),hpose.get((o['uid'],0),np.eye(4))))
    root=Path('data/rescan_three_period_residual_v1');meta=read(root/'public.json');paths.append(root/'public.json')
    O=[];raw=[]
    for t in ['1','2']:
        r=next(r for r in meta['records'] if r['scene']==scene and r['period']==t)
        p=root/r['path'];assert sha(p)==r['sha256'];paths.append(p)
        with np.load(p) as z:
            rr={k:z[k].copy() for k in ['xyz','rgb','predicted_class','proposal_index']}
        source=Path(r['source']);assert sha(source)==r['source_sha'];paths.append(source)
        native=read_ply(source);assert np.array_equal(xyz(native),rr['xyz'])
        rr['native_uid']=native['instance_idx'].copy()
        raw.append(rr);O.append(voxelize(rr['xyz'],rr['rgb'],rr['predicted_class'],cfg['voxel_m']))
    return H,O,raw,initial,paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/rescan_reversible_surface_v2');args=ap.parse_args()
    cp=Path('configs/rescan_reversible_surface_v2.json');cfg=read(cp);out=Path(args.output);out.mkdir(exist_ok=False)
    sources=[cp,Path(__file__),Path('inspection_decisions/reversible_surface_v2.py'),Path('scripts/prepare_real_temporal_v1.py')]
    manifest={'sources':{str(p):sha(p) for p in sources},'inputs':{},'current_GT_read':False,
              'poses':'Same fixed native layout initializations for all arms; no new proposal search.',
              'initial_ownership':'All arms share source predicted native UID labels, NOT truth; preserves newly visible and mistaken surface increments.',
              'prediction':'Full-point costs evaluated against voxel observation surfaces; all identities retained.'}
    save(out/'config.json',cfg);save(out/'manifest.json',manifest);start=time.perf_counter();rows=[]
    for scene in cfg['scenes']:
        H,O,raw,initial,paths=load(scene,cfg);manifest['inputs'].update({str(p):sha(p) for p in paths});save(out/'manifest.json',manifest)
        sceneout=out/scene;sceneout.mkdir()
        uid_to_index={h['uid']:i for i,h in enumerate(H)}
        native_labels=[np.array([uid_to_index.get(int(u),-1) for u in r['native_uid'][o['raw_index']]],int) for r,o in zip(raw,O)]
        save(sceneout/'assets.json',[{k:h[k] for k in ['uid','class','raw_points']} for h in H])
        np.savez_compressed(sceneout/'observations.npz',**{f'{k}_{t+1}':o[k] for t,o in enumerate(O) for k in ['xyz','rgb','raw_index','inverse','mass']},
                            initial_poses=np.array(initial),anchors=np.array([h['anchor'] for h in H]))
        print('LOADED',scene,'raw',[len(r['xyz']) for r in raw],'voxels',[len(o['xyz']) for o in O],flush=True)
        for method in cfg['methods']:
            dest=sceneout/method;dest.mkdir();began=time.perf_counter()
            def monitor(row):
                save(out/'monitor.json',{'scene':scene,'method':method,'iteration':row['iteration'],'elapsed_s':time.perf_counter()-start})
                print('ITER',scene,method,row['iteration'],row['changed_voxels'],row['accepted_pose_updates'],flush=True)
            poses,labels,trace,initlabel=run(O,H,initial,cfg,method,monitor,native_labels)
            np.savez_compressed(dest/'state.npz',poses=np.array(poses),labels1=labels[0],labels2=labels[1],
                                initial_labels1=initlabel[0],initial_labels2=initlabel[1],
                                anchors_world=np.array([[transform(h['anchor'][None],T)[0] for h,T in zip(H,pp)] for pp in poses]))
            save(dest/'trace.json',trace)
            for t in [0,1]:
                other=1-t;inc=None;stat=[]
                if method!='frozen' and (t==1 or method in ['joint_unrooted','rooted_joint']):
                    inc,stat=make_surface(O[other],labels[other],poses[other],H,cfg,method.startswith('rooted'))
                pred=[];matched=[];conf=[]
                for startidx in range(0,len(raw[t]['xyz']),25000):
                    sl=slice(startidx,startidx+25000)
                    batch={'xyz':raw[t]['xyz'][sl], 'rgb':raw[t]['rgb'][sl].astype(float)/255,'class':raw[t]['predicted_class'][sl]}
                    cc,a,_,src=costs(batch,poses[t],H,inc,cfg,True)
                    ids=np.array([h['uid'] for h in H]);p=np.full(len(a),-1,int);ok=a>=0;p[ok]=ids[a[ok]]
                    source=np.full(len(a),-1,int)
                    for i in range(len(H)):
                        use=a==i;source[use]=src[i][use]
                    pred.append(p);matched.append(source);conf.append(np.min(cc,axis=1))
                np.savez_compressed(dest/f'prediction{t+1}.npz',identity=np.concatenate(pred),source_voxel=np.concatenate(matched),cost=np.concatenate(conf))
                if inc is not None:
                    payload={f'{key}_{i}':inc[i][key] for i in range(len(H)) for key in ['xyz','rgb','source_voxel']}
                    np.savez_compressed(dest/f'source_for{t+1}.npz',**payload)
                save(dest/f'surface_stats_for{t+1}.json',stat)
            row={'scene':scene,'method':method,'elapsed_s':time.perf_counter()-began,
                 'source_reassigned_voxels':int(np.sum(labels[0]!=initlabel[0]))}
            rows.append(row);save(out/'runs.json',rows);print('DONE',row,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
         'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
         'scope':'Development only. No full Bayesian posterior; no complete pose search; no independent displacement truth.'})
    print('SEALED',len(rows),flush=True)


if __name__=='__main__':main()
