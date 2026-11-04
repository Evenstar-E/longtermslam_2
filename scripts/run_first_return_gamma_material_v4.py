"""All pixels, all candidates, no evaluation truth. Fixed scene shards."""
from pathlib import Path
import pickle,time,sys,os,shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import linear_sum_assignment
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.first_return_gamma_material_v1 import frame_rays,stats,unary,pair_costs,scalar_cost
from inspection_decisions.first_return_gamma_fast_v1 import pair_costs
from inspection_decisions.sparse_material_evidence_v1 import reduce_slots,assign
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def load_frames(base,tag,centers,cfg):
    all_frames=[];color=[];tokens=[]
    for t in range(3):
        rays=[];rgb=[[] for _ in range(24)]
        for v,path in enumerate(sorted((base/tag).glob(f'frame_{t}_*.npz'))):
            with np.load(path) as z:f={k:z[k].copy() for k in ['rgb','depth','C']}
            lab,_=labels(f,centers,np.arange(24)%4,cfg);r=frame_rays(f,cfg,lab);rays.append(r);tokens.append({'epoch':t,'view':v,'unique_pixels':r['pixel_count'],'log_depth_jacobian_sum':float(r['depth_jacobian_log'].sum())})
            for j in range(24):rgb[j].append(f['rgb'][lab==j].astype(float))
        all_frames.append(rays);color.append([np.concatenate(x) for x in rgb])
    return all_frames,color,tokens


def key(b):return (int(b['slot']),np.asarray(b['T'],float).tobytes())


def refine(H,frames,slots,x,centers,limits,cfg,method):
    x=np.array(x,float);Q=[stats(frames[t],world_pose(x[6*t:6*t+6],centers[slots[t]]),slots[t],cfg)[0] for t in range(2)];cost=scalar_cost(H,*Q,cfg,method);start=cost;trace=[cost];evaluations=1
    for _ in range(cfg['continuous_sweeps']):
        for t in range(2):
            for k in range(6):
                best=None
                for sign in [-1,1]:
                    p=x.copy();p[6*t+k]+=sign*(cfg['rotation_step_rad'] if k<3 else cfg['translation_step_m'])
                    if np.linalg.norm(p[6*t:6*t+3])>limits[0]+1e-12 or np.linalg.norm(p[6*t+3:6*t+6])>limits[1]+1e-12:continue
                    q=stats(frames[t],world_pose(p[6*t:6*t+6],centers[slots[t]]),slots[t],cfg)[0];trial=Q.copy();trial[t]=q;v=scalar_cost(H,*trial,cfg,method);evaluations+=1
                    if v<cost-1e-9 and (best is None or v<best[0]):best=(v,p,q)
                if best is not None:cost,x,Q[t]=best;trace.append(cost)
    return x,{'initial_cost':start,'final_cost':cost,'cost_trace':trace,'evaluations':evaluations,'accepted_steps':len(trace)-1,'solver':'one_coordinate_sweep_actual_grid_events','global_optimum':False}


def main():
    cp=Path('configs/first_return_gamma_material_v4.json');options=read(cp);parent=Path(options['parent_root']);cfg={**read(parent/'config.json'),**options};shard=int(sys.argv[1]);assert 0<=shard<cfg['execution_shards'];out=Path(cfg['output_root'])/f'shard_{shard}';out.mkdir(parents=True,exist_ok=False);base=Path(cfg['base_root']);limits=read(parent/'manifest.json')['motion_limits']
    for name,h in read(parent/'prediction_seal.json')['sha256'].items():assert sha(parent/name)==h
    sources=[cp,Path(__file__),Path('inspection_decisions/first_return_gamma_material_v1.py'),Path('inspection_decisions/first_return_gamma_fast_v1.py'),Path('inspection_decisions/sparse_material_evidence_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py'),Path('inspection_decisions/material_supported_continuous_alignment_v1.py'),Path('scripts/check_first_return_gamma_material_v1.py'),Path('research/first_return_gamma_execution_v4_2026-10-25.md')];inputs=[parent/'prediction_seal.json',parent/'manifest.json',Path('research/first_return_gamma_checks_v1.json')]+list(parent.glob('*/candidates.pkl'))+list(parent.glob('*/predictions.json'))+list(base.glob('*/inference.pkl'))+list(base.glob('*/frame_*.npz'))
    cache_manifest=read('research/first_return_gamma_cached_geometry_v2.json')
    for p,h in cache_manifest['sha256'].items():assert sha(p)==h
    inputs+=[Path('research/first_return_gamma_cached_geometry_v2.json'),Path('research/first_return_gamma_fast_checks_v1.json')]+[Path(p) for p in cache_manifest['sha256']]
    save(out/'config.json',cfg);save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'motion_limits':limits,'depth_noise_integrated':False,'all_raw_pixels':True,'same_pool_not_equal_FLOPs':True});runs=[];begin=time.perf_counter()
    for ordinal,run in enumerate(read(parent/'runs.json')):
        if ordinal%cfg['execution_shards']!=shard:continue
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (parent/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];frames,colors,tokens=load_frames(base,tag,centers,cfg);maximum=0.
        for ff in frames:
            for f in ff:
                ok=f['owner']>=0;p=f['origin']+f['direction'][ok]*f['distance'][ok,None]-centers[f['owner'][ok]];maximum=max(maximum,float(np.linalg.norm(p,axis=1).max(initial=0.)))
        cfg=dict(cfg);cfg['hazard_extent_m']=float(np.ceil((maximum+limits[1]+1e-9)/cfg['hazard_voxel_m'])*cfg['hazard_voxel_m']);save(dest/'scene_config.json',cfg);save(dest/'workspace_certificate.json',{'observed_max_radius_m':maximum,'translation_bound_m':limits[1],'extent_m':cfg['hazard_extent_m'],'bound':'norm(R_inverse*(point-center-u)) <= norm(point-center)+norm(u)','GT_read':False});print('WORKSPACE',tag,cfg['hazard_extent_m'],flush=True);save(dest/'raw_ray_inventory.json',tokens);H=np.array([stats(frames[0],inp['T0'][i],i,cfg)[0] for i in range(24)]);np.savez_compressed(dest/'history_stats.npz',H=H);Q=[];maps=[];geo=[]
        cached=Path('results/first_return_gamma_material_v2')/f'shard_{shard}'/tag
        if (cached/'current_stats_1.npz').exists():
            assert read(cached/'workspace_certificate.json')==read(dest/'workspace_certificate.json')
            for t in range(2):
                for name in [f'current_stats_{t}.npz',f'current_geometry_{t}.json']:
                    assert str(cached/name) in cache_manifest['sha256'];shutil.copyfile(cached/name,dest/name)
                with np.load(dest/f'current_stats_{t}.npz') as z:values=z['Q'];mapping=z['mapping']
                Q.append(values);maps.append(mapping);geo.append(len(values));print('REUSED_EXACT_RAW_STATS',tag,t,len(values),flush=True)
        else:
            for t in range(2):
                cache={};values=[];mapping=np.zeros((24,288),int);descriptors=[]
                for i in range(24):
                    assert len(pools[i][t])==288
                    for j,b in enumerate(pools[i][t]):
                        ky=key(b)
                        if ky not in cache:
                            cache[ky]=len(values);q,meta=stats(frames[t+1],np.array(b['T']),b['slot'],cfg);values.append(q);descriptors.append({'slot':b['slot'],'T':b['T'],'rays':meta})
                        mapping[i,j]=cache[ky]
                values=np.array(values);Q.append(values);maps.append(mapping);geo.append(len(values));np.savez_compressed(dest/f'current_stats_{t}.npz',Q=values,mapping=mapping);save(dest/f'current_geometry_{t}.json',descriptors);print('RAW_STATS',tag,t,values.shape,'s',round(time.perf_counter()-clock,1),flush=True)
        C=np.zeros((len(cfg['methods']),24,24,24));choices=np.zeros(C.shape+(2,),np.int16);unaries={m:np.zeros((2,24,24)) for m in cfg['methods']};appearance=np.zeros((24,2,288))
        def cost_asset(i):
            A,B=[Q[t][maps[t][i]] for t in range(2)];mu=colors[0][i].mean(0)
            rgb=np.array([[np.sum((colors[t+1][b['slot']]-mu)**2)/(2*cfg['appearance_sigma_rgb']**2) for b in pools[i][t]] for t in range(2)]);appearance[i]=rgb
            for mi,m in enumerate(cfg['methods']):
                cost=pair_costs(H[i],A,B,cfg,m)+rgb[0,:,None]+rgb[1,None,:];C[mi,i],choices[mi,i]=reduce_slots(cost,pools[i])
                for t,E in enumerate([A,B]):
                    u=unary(H[i],E,cfg,m)+rgb[t]
                    for s in range(24):unaries[m][t,i,s]=min(u[k] for k,b in enumerate(pools[i][t]) if b['slot']==s)
            print('COST_ASSET',tag,i,'s',round(time.perf_counter()-clock,1),flush=True)
        with ThreadPoolExecutor(max_workers=cfg['asset_cost_workers']) as executor:list(executor.map(cost_asset,range(24)))
        np.savez_compressed(dest/'costs.npz',C=C,choices=choices,appearance=appearance);old=read(parent/tag/'predictions.json');starts=[]
        for family in ['catalogue_plane','material_shared']:
            rr=[r for r in old if r['family']==family];starts.append(tuple(np.array([r['original']['slots'][t] for r in rr]) for t in range(2)))
            for m in ['typed_ray','line_joint']:starts.append(tuple(np.array([r['methods'][m]['stages']['final']['slots'][t] for r in rr]) for t in range(2)))
        for m in cfg['methods']:starts.append(tuple(linear_sum_assignment(unaries[m][t])[1] for t in range(2)))
        save(dest/'assignment_starts.json',starts);assignment={}
        for mi,m in enumerate(cfg['methods']):
            a,b,log=assign(C[mi],starts,cfg['assignment_rounds']);assignment[m]={'slots':np.array([a,b]),'log':log};print('ASSIGNED',tag,m,log['gap'],flush=True)
        save(dest/'assignments.json',assignment);rows=[]
        for i in range(24):
            row={'asset':i,'methods':{}}
            for mi,m in enumerate(cfg['methods']):
                slots=assignment[m]['slots'][:,i];j,k=slots;aa,bb=choices[mi,i,j,k];candidates=[pools[i][0][aa],pools[i][1][bb]];x=np.concatenate([pose_vector(candidates[t]['T'],centers[slots[t]]) for t in range(2)]);p,log=refine(H[i],frames[1:],slots,x,centers,limits,cfg,m);row['methods'][m]={'candidate_indices':[int(aa),int(bb)],'initial_parameters':x,'final_parameters':p,'log':log,'stages':{'candidate':{'slots':slots,'T1':candidates[0]['T'],'T2':candidates[1]['T']},'final':{'slots':slots,'T1':world_pose(p[:6],centers[j]),'T2':world_pose(p[6:],centers[k])}}}
            rows.append(row)
            if i%6==5:print('REFINE',tag,i+1,'s',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',rows);r={**run,'elapsed_s':time.perf_counter()-clock,'unique_current_geometries':geo,'continuous_fits':96};runs.append(r);save(out/'runs.json',runs);print('DONE',tag,r['elapsed_s'],flush=True)
    # Inference sources/inputs remain identical until sealing.
    manifest=read(out/'manifest.json')
    for field in ['sources','inputs']:
        for p,h in manifest[field].items():assert sha(p)==h
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'subset_shard':shard,'elapsed_s':time.perf_counter()-begin,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',shard,flush=True)


if __name__=='__main__':main()
