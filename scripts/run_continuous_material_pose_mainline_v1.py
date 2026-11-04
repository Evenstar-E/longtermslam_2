"""No crop; all 54 seeds per asset/epoch, full batch seal before truth reads."""
from pathlib import Path
import pickle,time
import numpy as np
import open3d as o3d
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import inverse
from inspection_decisions.continuous_material_pose_mainline_v1 import observations,make_cloud,PairModel,refine,select


def main():
    cp=Path('configs/continuous_material_pose_mainline_v1.json');options=read(cp);base=Path(options['base_root']);prior=Path(options['plane_root']);cfg={**read(base/'config.json'),**options}
    for old in [base,prior]:
        for name,h in read(old/'prediction_seal.json')['sha256'].items():assert sha(old/name)==h
    out=Path(cfg['output_root']);out.mkdir(exist_ok=False);save(out/'config.json',cfg)
    sources=[cp,Path(__file__),Path('inspection_decisions/continuous_material_pose_mainline_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py'),Path('scripts/completion_catalog_scene_v1.py'),Path('research/continuous_material_pose_mainline_contract_v1_2026-10-24.md')]
    inputs=[base/'config.json',base/'runs.json']+list(base.glob('*/inference.pkl'))+list(base.glob('*/frame_*.npz'))+list(prior.glob('*/predictions.json'))
    save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'base_seal':sha(base/'prediction_seal.json'),'prior_seal':sha(prior/'prediction_seal.json'),'GT_read':False,'open3d_version':o3d.__version__,'familiar_DEV':True})
    runs=[];start=time.perf_counter()
    for run in read(base/'runs.json'):
        tag=run['scene'];src=base/tag;dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        n=len(inp['T0']);slots=inp['T0'][:,:3,3];groups=np.arange(n)%4;frames=[];points=[]
        for t in [0,1,2]:
            raw=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:raw.append({k:z[k].copy() for k in ['rgb','depth','C']})
            ff,pp=observations(raw,slots,groups,cfg);frames.append(ff);points.append(pp)
        history=[inverse(p,T0) for p,T0 in zip(points[0],inp['T0'])]
        historical_full=[make_cloud(p,0.,cfg) for p in history]
        historical_pyramid=[[make_cloud(p,v,cfg) for v in cfg['voxel_scales_m']] for p in history]
        current_full=[[make_cloud(p-slots[j],0.,cfg) for j,p in enumerate(pp)] for pp in points[1:]]
        current_pyramid=[[[make_cloud(p-slots[j],v,cfg) for v in cfg['voxel_scales_m']] for j,p in enumerate(pp)] for pp in points[1:]]
        candidates=[];logs=[];max_gap=0.;attempts=success=0
        for i,bb in enumerate(inp['branches']):
            per_asset=[]
            for t,branches in enumerate(bb):
                originals=[];endpoints=[]
                models={j:PairModel(history[i],inp['T0'][i],frames[0],points[t+1][j],frames[t+1],i,j,slots[j],cfg,historical_full[i],current_full[t][j]) for j in sorted({b['slot'] for b in branches})}
                for k,b in enumerate(branches):
                    model=models[b['slot']];s,d=model.score(b['T']);gap=abs(s['nn']-b['cost']);max_gap=max(gap,max_gap);assert gap<1e-10,(tag,i,t,k,gap)
                    originals.append({'slot':b['slot'],'code':b['code'],'T':b['T'],'source':'catalogue','parent':k,'scores':s,'support':d})
                for k,b in enumerate(branches):
                    attempts+=1;T,trace=refine(b['T'],historical_pyramid[i],current_pyramid[t][b['slot']],slots[b['slot']],cfg)
                    new_index=None
                    if T is not None:
                        s,d=models[b['slot']].score(T);new_index=len(originals)+len(endpoints);success+=1
                        endpoints.append({'slot':b['slot'],'initial_code':b['code'],'T':T,'source':'point_to_plane','parent':k,'scores':s,'support':d})
                    logs.append({'asset':i,'epoch':t,'parent':k,'slot':b['slot'],'endpoint_index':new_index,'complete':T is not None,'trace':trace})
                per_asset.append(originals+endpoints)
            candidates.append(per_asset)
            if i%6==5:print('MAINLINE',tag,'assets',i+1,'paths',attempts,'successful',success,'seconds',round(time.perf_counter()-clock,2),flush=True)
        predictions={}
        for expanded,name in [(False,'catalogue'),(True,'expanded')]:
            for score in cfg['scores']:predictions[name+'_'+score]=select(candidates,score,expanded)
        endpoint_map={(v['asset'],v['epoch'],v['parent']):v['endpoint_index'] for v in logs}
        direct=[];fallback=0
        for i,k,l in predictions['catalogue_nn']['selection']:
            selected=[]
            for t,j in enumerate([k,l]):
                endpoint=endpoint_map[(i,t,j)];selected.append(endpoint if endpoint is not None else j);fallback+=endpoint is None
            direct.append([i,*selected])
        predictions['fixed_nn_ids_refined']={'status':0,'selection':direct,'fallback_epochs':int(fallback),'objective':None}
        old=read(prior/tag/'predictions.json');predictions['sealed_independent_planes']=old['independent_period_planes']
        assert predictions['catalogue_nn']['selection']==[list(v) for v in old['ordinary_nn']['selection']]
        with (dest/'candidates.pkl').open('wb') as f:pickle.dump(candidates,f)
        save(dest/'optimization_logs.json',logs);save(dest/'predictions.json',predictions)
        item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(attempts=attempts,successful=success,failed=attempts-success,old_candidates=attempts,new_candidates=success,max_old_nn_gap=max_gap,elapsed_s=time.perf_counter()-clock,history_point_counts=list(map(len,history)),current_point_counts=[[len(p) for p in pp] for pp in points[1:]])
        runs.append(item);save(out/'runs.json',runs);print('DONE',tag,{k:item[k] for k in ['attempts','successful','failed','elapsed_s','max_old_nn_gap']},flush=True)
    save(out/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,'GT_read':False,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    print('SEALED',flush=True)


if __name__=='__main__':main()
