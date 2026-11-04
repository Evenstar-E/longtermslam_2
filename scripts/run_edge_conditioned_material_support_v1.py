"""Complete DEV batch, conditional edge model; no evaluator imports or truth."""
from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.continuous_material_pose_mainline_v1 import observations,make_cloud,PairModel,select
from inspection_decisions.material_domain_observation_v1 import reduced_global
from inspection_decisions.material_supported_continuous_alignment_v1 import historical_offsets,pair_costs,pose_vector,world_pose,material_residual,projected_gn
from inspection_decisions.edge_conditioned_material_support_v1 import extract_edges,edge_residual,edge_pair_costs
from scripts.completion_catalog_scene_v1 import inverse
from scripts.check_edge_conditioned_material_support_v1 import checks


def main():
    preflight=checks();cp=Path('configs/edge_conditioned_material_support_v1.json');options=read(cp);cache=Path(options['cached_model_root']);base=Path(options['base_root']);cfg={**read(cache/'config.json'),**options};out=Path(cfg['output_root'])
    for root in [base,cache]:
        for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    out.mkdir(exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',preflight)
    sources=[cp,Path(__file__),Path('inspection_decisions/edge_conditioned_material_support_v1.py'),Path('scripts/check_edge_conditioned_material_support_v1.py'),Path('research/edge_conditioned_material_support_contract_v1_2026-10-24.md'),Path('inspection_decisions/material_supported_continuous_alignment_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py'),Path('inspection_decisions/continuous_material_pose_mainline_v1.py'),Path('inspection_decisions/material_domain_observation_v1.py'),Path('scripts/completion_catalog_scene_v1.py')]
    inputs=[base/'runs.json',base/'config.json',cache/'prediction_seal.json',cache/'manifest.json',cache/'config.json']+list(base.glob('*/frame_*.npz'))+list(base.glob('*/inference.pkl'))+list(cache.glob('*/candidates.pkl'))+list(cache.glob('*/features.pkl'))+list(cache.glob('*/bounded_icp_logs.json'))
    limits=read(cache/'manifest.json')['motion_limits'];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'familiar_DEV':True,'motion_limits':limits,'cached_bounded_ICP_reexecuted':False,'edge_intervals_are_conditional_not_calibrated':True})
    runs=[];start=time.perf_counter()
    for run in read(base/'runs.json'):
        tag=run['scene'];src=base/tag;dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (cache/tag/'features.pkl').open('rb') as f:features=pickle.load(f)
        with (cache/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;frames=[];points=[];edges=[];logs=[]
        for t in range(3):
            raw=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:raw.append({k:z[k].copy() for k in ['rgb','depth','C']})
            ff,pp=observations(raw,slots,groups,cfg);frames.append(ff);points.append(pp);per_slot=[[] for _ in range(24)]
            for v,frame in enumerate(ff):
                frame['edge_view']=v;lab,world=labels(frame,slots,groups,cfg)
                for j in range(24):
                    extracted,log=extract_edges(frame,lab,world,j,slots[j],features[t][j],cfg,cfg['seed']+1000*t+100*v+j);per_slot[j].extend(extracted);logs.append({'epoch':t,'view':v,'slot':j,**log})
            edges.append(per_slot)
        with (dest/'edges.pkl').open('wb') as f:pickle.dump(edges,f)
        save(dest/'edge_logs.json',logs);print('EDGES',tag,'segments',sum(l['segments'] for l in logs),'seconds',round(time.perf_counter()-clock,2),flush=True)
        history=[inverse(p,T0) for p,T0 in zip(points[0],inp['T0'])];hc=[make_cloud(p,0.,cfg) for p in history];cc=[[make_cloud(p-slots[j],0.,cfg) for j,p in enumerate(pp)] for pp in points[1:]]
        models=[[{j:PairModel(history[i],inp['T0'][i],frames[0],points[t+1][j],frames[t+1],i,j,slots[j],cfg,hc[i],cc[t][j]) for j in sorted({b['slot'] for b in candidates[i][t]})} for t in range(2)] for i in range(24)]
        reference=[historical_offsets(ff) for ff in features[0]];predictions={};joint_logs=[];round_logs=[]
        bounded=[[[b for b in bb if b['source'] in ['catalogue','bounded_icp']] for bb in asset] for asset in candidates]
        for score in ['nn','plane']:predictions['bounded_icp_'+score]=select(bounded,score,True)
        def costs_for_pool(pool):
            result=[]
            for i,bb in enumerate(pool):
                p=pair_costs(bb,features[1:],slots,reference[i],cfg);e=edge_pair_costs(bb,edges[1:],slots,edges[0][i],cfg);result.append((p,e))
            return result
        for round_index in range(cfg['joint_rounds']):
            snapshot=[[list(bb[0]),list(bb[1])] for bb in candidates];costs=costs_for_pool(snapshot)
            for shared,name in [(False,'independent'),(True,'shared')]:
                selection=reduced_global(snapshot,[p[1]+e[int(shared)] for p,e in costs],cfg['global_time_limit_s']);round_logs.append({'round':round_index,'model':name,**selection})
                if selection['status']!=0 or not selection['selection']:continue
                for i,k,l in selection['selection']:
                    chosen=[snapshot[i][0][k],snapshot[i][1][l]];centers=[slots[b['slot']] for b in chosen];ff=[features[t+1][b['slot']] for t,b in enumerate(chosen)];ee=[edges[t+1][b['slot']] for t,b in enumerate(chosen)];x=np.concatenate([pose_vector(b['T'],c) for b,c in zip(chosen,centers)])
                    def fun(xx):
                        rp,Jp=material_residual(xx,ff,reference[i],cfg,True);re,Je=edge_residual(xx,ee,edges[0][i],cfg,shared);return np.r_[rp,re],np.r_[Jp,Je]
                    fitted,log=projected_gn(x,fun,cfg['joint_iterations'],limits,cfg);added=[]
                    if log['accepted_steps']:
                        for t,(b,c) in enumerate(zip(chosen,centers)):
                            T=world_pose(fitted[6*t:6*t+6],c);s,d=models[i][t][b['slot']].score(T);index=len(candidates[i][t]);candidates[i][t].append({'slot':b['slot'],'T':T,'source':'edge_'+name,'parent':(k,l)[t],'scores':s,'support':d});added.append(index)
                    joint_logs.append({'round':round_index,'model':name,'asset':i,'initial_pair':[k,l],'added':added,**log})
            print('ROUND',tag,round_index,'seconds',round(time.perf_counter()-clock,2),flush=True)
        costs=costs_for_pool(candidates)
        for shared,name in [(False,'independent'),(True,'shared')]:
            predictions['material_'+name]=reduced_global(candidates,[p[int(shared)] for p,e in costs],cfg['global_time_limit_s'])
            predictions['edge_'+name]=reduced_global(candidates,[p[1]+e[int(shared)] for p,e in costs],cfg['global_time_limit_s'])
            originals=[[bb[0][:54],bb[1][:54]] for bb in candidates]
            predictions['catalogue_material_'+name]=reduced_global(originals,[p[int(shared)][:54,:54] for p,e in costs],cfg['global_time_limit_s'])
            predictions['catalogue_edge_'+name]=reduced_global(originals,[(p[1]+e[int(shared)])[:54,:54] for p,e in costs],cfg['global_time_limit_s'])
        for score in ['nn','plane']:predictions['common_'+score]=select(candidates,score,True)
        with (dest/'candidates.pkl').open('wb') as f:pickle.dump(candidates,f)
        with (dest/'features.pkl').open('wb') as f:pickle.dump(features,f)
        save(dest/'predictions.json',predictions);save(dest/'joint_logs.json',joint_logs);save(dest/'round_logs.json',round_logs);save(dest/'bounded_icp_logs.json',read(cache/tag/'bounded_icp_logs.json'))
        np.savez_compressed(dest/'costs.npz',**{f'{name}_{i}':arr for i,(p,e) in enumerate(costs) for name,arr in [('plane_independent',p[0]),('plane_shared',p[1]),('edge_independent',e[0]),('edge_shared',e[1])]})
        item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(elapsed_s=time.perf_counter()-clock,segments=sum(l['segments'] for l in logs),joint_paths=len(joint_logs),accepted_joint_steps=sum(len(l['accepted_steps']) for l in joint_logs),final_candidates=sum(len(bb) for asset in candidates for bb in asset),global_status={m:r['status'] for m,r in predictions.items()})
        runs.append(item);save(out/'runs.json',runs);print('DONE',tag,item,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',flush=True)


if __name__=='__main__':main()
