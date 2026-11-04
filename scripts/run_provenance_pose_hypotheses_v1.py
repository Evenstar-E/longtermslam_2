"""Joint finite-pool development experiment; no current truth access."""
import argparse,time
from pathlib import Path
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha,load,database,xyz,rgb
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem,rigid_projection
from inspection_decisions.reversible_surface_v2 import costs,transform


def read_pool(path):
    data=path.read_bytes();n=int(np.frombuffer(data[:4],'<i4')[0]);count=np.frombuffer(data[4:4+4*n],'<i4');off=4+4*n;out=[]
    for c in count:
        rows=np.frombuffer(data[off:off+int(c)*68],'<f4').reshape(-1,17).copy();off+=int(c)*68
        T=np.array([r[:16].reshape(4,4,order='F') for r in rows],float).reshape(-1,4,4)
        assert np.isfinite(T).all()
        if len(T):
            assert np.allclose(T[:,3], [0,0,0,1],atol=1e-5)
            assert np.all(np.linalg.det(T[:,:3,:3])>.99)
        out.append(T)
    assert off==len(data)
    return out


def pools_for(scene,H,initial):
    old=read('configs/rescan_third_period_v1.json')
    roots=[Path(old['prior_roots'][scene]),Path('results/rescan_third_period_progress_v1' if scene=='scene_a' else 'results/rescan_third_period_v1_completed')/scene/'initial_archive']
    reference,_=database(roots[0]/'history.rsdb');reference={v['uid']:v for v in reference}
    pools=[[],[]];meta=[];paths=[]
    prev=Path('results/rescan_reversible_surface_v2');seal=read(prev/'prediction_seal.json');assert seal['complete']
    previous=[]
    for method in read(prev/'config.json')['methods']:
        p=prev/scene/method/'state.npz';assert sha(p)==seal['sha256'][str(p.relative_to(prev))]
        previous.append((method,np.load(p)['poses']));paths.append(p)
    for t,root in enumerate(roots):
        db=root/'proposals.rsdb';binary=root/'proposals/proposals.bin';paths.extend([db,binary])
        objs,_=database(db);native=read_pool(binary);assert len(objs)==len(native);native={o['uid']:(o,T) for o,T in zip(objs,native)}
        for i,h in enumerate(H):
            obj,T=native[h['uid']];paths.append(obj['path'])
            a,b=read_ply(reference[h['uid']]['path']),read_ply(obj['path'])
            assert np.array_equal(xyz(a),xyz(b)) and np.array_equal(rgb(a),rgb(b)),('canonical model changed',scene,t,h['uid'])
            original=[initial[t][i]]+list(T)+[p[t,i] for _,p in previous]
            projected=np.array([rigid_projection(v) for v in original]);pools[t].append(projected)
            meta.append({'period':t+1,'asset_index':i,'uid':h['uid'],'native_count':len(T),'total_count':len(projected),
                         'order':['selected_native']+[f'native_{k}' for k in range(len(T))]+[f'v2_{m}' for m,_ in previous],
                         'max_rotation_projection_change':float(max(np.max(np.abs(a[:3,:3]-b[:3,:3])) for a,b in zip(original,projected))),
                         'all_native_retained':True,'deduplicated':False})
    return pools,meta,paths


def export(dest,problem,raw,initial_labels):
    p=problem;H=p.H;O=p.O;poses=np.array(p.pose);ids=np.array([h['uid'] for h in H])
    np.savez_compressed(dest/'state.npz',poses=poses,k=p.k,labels1=p.z[0],labels2=p.z[1],
        initial_labels1=initial_labels[0],initial_labels2=initial_labels[1],
        anchors_world=np.array([[transform(h['anchor'][None],T)[0] for h,T in zip(H,pp)] for pp in poses]))
    save(dest/'trace.json',p.trace)
    for t in [0,1]:
        inc=[p.surface(1-t,i) for i in range(len(H))] if t==1 or p.bidir else None
        primary=p.z[t][O[t]['inverse']];labels=np.full(len(primary),-1,int);valid=primary>=0;labels[valid]=ids[primary[valid]]
        sources=[];values=[];readout=[];readout_cost=[]
        for start in range(0,len(primary),25000):
            sl=slice(start,start+25000);r=raw[t]
            batch={'xyz':r['xyz'][sl],'rgb':r['rgb'][sl].astype(float)/255,'class':r['predicted_class'][sl]}
            cc,a,_,src=costs(batch,poses[t],H,inc,p.cfg,True)
            chosen=primary[sl];s=np.full(len(a),-1,int);v=np.full(len(a),p.null,float)
            for i in range(len(H)):
                use=chosen==i;s[use]=src[i][use];v[use]=cc[use,i]
            pp=np.full(len(a),-1,int);ok=a>=0;pp[ok]=ids[a[ok]]
            sources.append(s);values.append(v);readout.append(pp);readout_cost.append(np.min(cc,axis=1))
        np.savez_compressed(dest/f'prediction{t+1}.npz',identity=labels,source_voxel=np.concatenate(sources),cost=np.concatenate(values),
                            readout_identity=np.concatenate(readout),readout_cost=np.concatenate(readout_cost))
        if inc is not None:
            np.savez_compressed(dest/f'source_for{t+1}.npz',**{f'{key}_{i}':inc[i][key] for i in range(len(H)) for key in ['xyz','rgb','source_voxel']})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/provenance_pose_hypotheses_v1');args=ap.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=False);cp=Path('configs/provenance_pose_hypotheses_v1.json');cfg=read(cp)
    files=[cp,Path(__file__),Path('inspection_decisions/provenance_pose_hypotheses_v1.py'),Path('inspection_decisions/reversible_surface_v2.py'),
           Path('scripts/run_rescan_reversible_surface_v2.py'),Path('scripts/prepare_real_temporal_v1.py')]
    manifest={'sources':{str(p):sha(p) for p in files},'inputs':{},'current_GT_read':False,
              'primary_readout':'Lift optimized voxel ownership to every original point; does not erase coupling by independent nearest-neighbor reassignment.',
              'secondary_readout':'All original points independently rescored under final surfaces; diagnostic only, reported separately.'}
    save(out/'config.json',cfg);save(out/'manifest.json',manifest);start=time.perf_counter();runs=[]
    for scene in cfg['scenes']:
        H,O,raw,initial,paths=load(scene,cfg);pools,meta,poolpaths=pools_for(scene,H,initial);paths+=poolpaths
        manifest['inputs'].update({str(p):sha(p) for p in paths});save(out/'manifest.json',manifest)
        root=out/scene;root.mkdir();save(root/'pool_metadata.json',meta)
        np.savez_compressed(root/'pool.npz',**{f'T_{t+1}_{i}':p for t,pp in enumerate(pools) for i,p in enumerate(pp)})
        ids={h['uid']:i for i,h in enumerate(H)}
        native=[np.array([ids.get(int(u),-1) for u in r['native_uid'][o['raw_index']]],int) for r,o in zip(raw,O)]
        save(root/'assets.json',[{k:h[k] for k in ['uid','class','raw_points']} for h in H])
        np.savez_compressed(root/'observations.npz',**{f'{key}_{t+1}':o[key] for t,o in enumerate(O) for key in ['xyz','rgb','raw_index','inverse','mass']},
                            initial_poses=np.array([[p[0] for p in pp] for pp in pools]),anchors=np.array([h['anchor'] for h in H]))
        print('POOL',scene,[(t+1,sum(len(p) for p in pp)) for t,pp in enumerate(pools)],flush=True)
        for method in cfg['methods']:
            dest=root/method;dest.mkdir();began=time.perf_counter();p=Problem(O,H,pools,native,cfg,method)
            def monitor(row,problem):
                small={k:v for k,v in row.items() if k!='candidates'}
                save(out/'monitor.json',{'scene':scene,'method':method,'latest':small,'elapsed_s':time.perf_counter()-start,
                    'column_calls':problem.column_calls,'point_queries':problem.point_queries})
                if 'sweep_complete' in row or row.get('asset_index',1)%5==0:print('PROGRESS',scene,method,small,flush=True)
            p.solve(monitor);cache=p.verify_cache();export(dest,p,raw,native)
            record={'scene':scene,'method':method,'elapsed_s':time.perf_counter()-began,'initial_energy':p.initial_energy,'final_energy':p.energy(),
                    'source_changed_voxels':int(np.sum(p.z[0]!=native[0])),'pose_branches_changed':int(np.sum(p.k!=0)),
                    'column_calls':p.column_calls,'point_queries':p.point_queries,'cache_rebuild_max_error':cache}
            save(dest/'run.json',record);runs.append(record);save(out/'runs.json',runs);print('DONE',record,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
        'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'scope':cfg['scope']})
    print('SEALED',len(runs),flush=True)


if __name__=='__main__':main()
