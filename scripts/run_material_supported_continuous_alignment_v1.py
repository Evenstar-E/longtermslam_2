"""Complete 192-task joint-material experiment, no evaluator access."""
from pathlib import Path
import pickle,time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import inverse,catalog
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.continuous_material_pose_mainline_v1 import observations,make_cloud,PairModel,select
from inspection_decisions.material_domain_observation_v1 import reduced_global
from inspection_decisions.material_supported_continuous_alignment_v1 import (
    extract_planes,historical_offsets,bounded_icp,pair_costs,pose_vector,world_pose,material_residual,projected_gn,AXES)
from scripts.check_material_supported_continuous_alignment_v1 import checks


def main():
    preflight=checks();cp=Path('configs/material_supported_continuous_alignment_v1.json');options=read(cp);base=Path(options['base_root']);baseline=Path(options['baseline_root']);cfg={**read(base/'config.json'),**options}
    for old in [base,baseline]:
        for name,h in read(old/'prediction_seal.json')['sha256'].items():assert sha(old/name)==h
    out=Path(cfg['output_root']);out.mkdir(exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',preflight)
    sources=[cp,Path(__file__),Path('inspection_decisions/material_supported_continuous_alignment_v1.py'),Path('scripts/check_material_supported_continuous_alignment_v1.py'),Path('inspection_decisions/continuous_material_pose_mainline_v1.py'),Path('inspection_decisions/material_domain_observation_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py'),Path('scripts/completion_catalog_scene_v1.py'),Path('research/material_supported_continuous_alignment_contract_v1_2026-10-24.md')]
    inputs=[base/'runs.json',base/'config.json']+list(base.glob('*/inference.pkl'))+list(base.glob('*/frame_*.npz'))+list(baseline.glob('*/candidates.pkl'))
    cat=catalog();limits=(float(np.linalg.norm(Rotation.from_matrix(cat[:,:3,:3]).as_rotvec(),axis=1).max()),float(np.linalg.norm(cat[:,:3,3],axis=1).max()))
    save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'motion_limits':list(limits),'familiar_DEV':True,'Manhattan_dominant_face_prior':True,'score_is_calibrated_likelihood':False})
    runs=[];start=time.perf_counter()
    for run in read(base/'runs.json'):
        tag=run['scene'];src=base/tag;dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (baseline/tag/'candidates.pkl').open('rb') as f:old=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;frames=[];points=[];features=[];feature_logs=[]
        for t in [0,1,2]:
            raw=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:raw.append({k:z[k].copy() for k in ['rgb','depth','C']})
            ff,pp=observations(raw,slots,groups,cfg);frames.append(ff);points.append(pp);per_slot=[[] for _ in range(24)]
            for v,frame in enumerate(ff):
                lab,world=labels(frame,slots,groups,cfg)
                for slot in range(24):
                    extracted,log=extract_planes(world[lab==slot],frame['C'][:3,3],slots[slot],v,cfg,cfg['seed']+t*1000+v*100+slot)
                    per_slot[slot].extend(extracted);feature_logs.append({'epoch':t,'view':v,'slot':slot,**log})
            features.append(per_slot)
        history=[inverse(p,T0) for p,T0 in zip(points[0],inp['T0'])];historical_full=[make_cloud(p,0.,cfg) for p in history]
        hp=[[make_cloud(p,v,cfg) for v in cfg['voxel_scales_m']] for p in history]
        cf=[[make_cloud(p-slots[j],0.,cfg) for j,p in enumerate(pp)] for pp in points[1:]]
        cpyr=[[[make_cloud(p-slots[j],v,cfg) for v in cfg['voxel_scales_m']] for j,p in enumerate(pp)] for pp in points[1:]]
        candidates=[];icp_logs=[];models=[];bounded_counts=[]
        for i in range(24):
            per_asset=[];per_models=[];lengths=[]
            for t in [0,1]:
                originals=old[i][t][:54];bb=[dict(b) for b in originals];model={j:PairModel(history[i],inp['T0'][i],frames[0],points[t+1][j],frames[t+1],i,j,slots[j],cfg,historical_full[i],cf[t][j]) for j in sorted({b['slot'] for b in originals})}
                for k,b in enumerate(originals):
                    T,log=bounded_icp(b['T'],hp[i],cpyr[t][b['slot']],slots[b['slot']],limits,cfg);index=None
                    if T is not None:
                        s,d=model[b['slot']].score(T);index=len(bb);bb.append({'slot':b['slot'],'T':T,'source':'bounded_icp','parent':k,'scores':s,'support':d})
                    icp_logs.append({'asset':i,'epoch':t,'parent':k,'endpoint':index,**log})
                per_asset.append(bb);per_models.append(model);lengths.append(len(bb))
            candidates.append(per_asset);models.append(per_models);bounded_counts.append(lengths)
            if i%4==3:print('BOUNDED',tag,i+1,'seconds',round(time.perf_counter()-clock,2),flush=True)
        predictions={}
        for score in ['nn','plane']:predictions['bounded_icp_'+score]=select(candidates,score,True)
        reference=[historical_offsets(ff) for ff in features[0]];joint_logs=[];round_logs=[]
        for round_index in range(cfg['joint_rounds']):
            snapshot=[[list(bb[0]),list(bb[1])] for bb in candidates]
            costs=[pair_costs(bb,features[1:],slots,reference[i],cfg) for i,bb in enumerate(candidates)]
            for shared,name in [(False,'independent'),(True,'shared')]:
                result=reduced_global(snapshot,[c[int(shared)] for c in costs],cfg['global_time_limit_s']);round_logs.append({'round':round_index,'model':name,**result})
                if result['status']!=0 or not result['selection']:continue
                for i,k,l in result['selection']:
                    chosen=[candidates[i][0][k],candidates[i][1][l]];centers=[slots[b['slot']] for b in chosen];ff=[features[t+1][b['slot']] for t,b in enumerate(chosen)]
                    x=np.concatenate([pose_vector(b['T'],c) for b,c in zip(chosen,centers)])
                    fun=lambda xx:material_residual(xx,ff,reference[i],cfg,shared)
                    fitted,log=projected_gn(x,fun,cfg['joint_iterations'],limits,cfg);added=[]
                    if log['accepted_steps']:
                        for t,(b,c) in enumerate(zip(chosen,centers)):
                            T=world_pose(fitted[6*t:6*t+6],c);s,d=models[i][t][b['slot']].score(T);index=len(candidates[i][t]);candidates[i][t].append({'slot':b['slot'],'T':T,'source':'material_'+name,'parent':(k,l)[t],'scores':s,'support':d});added.append(index)
                    joint_logs.append({'round':round_index,'model':name,'asset':i,'initial_pair':[k,l],'added':added,**log})
            print('JOINT_ROUND',tag,round_index,'seconds',round(time.perf_counter()-clock,2),flush=True)
        costs=[pair_costs(bb,features[1:],slots,reference[i],cfg) for i,bb in enumerate(candidates)]
        for shared,name in [(False,'independent'),(True,'shared')]:
            predictions['material_'+name]=reduced_global(candidates,[c[int(shared)] for c in costs],cfg['global_time_limit_s'])
            originals=[[bb[0][:54],bb[1][:54]] for bb in candidates]
            predictions['catalogue_material_'+name]=reduced_global(originals,[c[int(shared)][:54,:54] for c in costs],cfg['global_time_limit_s'])
        for score in ['nn','plane']:predictions['common_'+score]=select(candidates,score,True)
        supports=[]
        for name in ['material_independent','material_shared']:
            selection=predictions[name]['selection']
            if not selection:continue
            for i,k,l in selection:
                chosen=[candidates[i][0][k],candidates[i][1][l]];present=[set(f['axis'] for f in features[t+1][b['slot']]) for t,b in enumerate(chosen)]
                born=sorted((present[0]&present[1])-set(reference[i]));supports.append({'method':name,'asset':i,'selected':[k,l],'current_born_shared_axes':born,'joint_minus_independent':float(costs[i][1][k,l]-costs[i][0][k,l]),'historical_axes':sorted(reference[i])})
        with (dest/'candidates.pkl').open('wb') as f:pickle.dump(candidates,f)
        with (dest/'features.pkl').open('wb') as f:pickle.dump(features,f)
        save(dest/'feature_logs.json',feature_logs);save(dest/'bounded_icp_logs.json',icp_logs);save(dest/'joint_logs.json',joint_logs);save(dest/'round_logs.json',round_logs);save(dest/'predictions.json',predictions);save(dest/'support_details.json',supports)
        np.savez_compressed(dest/'material_costs.npz',**{f'{name}_{i}':c[k] for i,c in enumerate(costs) for k,name in enumerate(['independent','shared'])})
        item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(elapsed_s=time.perf_counter()-clock,icp_attempts=len(icp_logs),icp_failures=sum(l['endpoint'] is None for l in icp_logs),accepted_icp_steps=sum(len(v['accepted_steps']) for l in icp_logs for v in l['levels']),joint_paths=len(joint_logs),accepted_joint_steps=sum(len(l['accepted_steps']) for l in joint_logs),bounded_pool_lengths=bounded_counts,final_candidates=sum(len(bb) for asset in candidates for bb in asset),global_status={m:r['status'] for m,r in predictions.items()})
        runs.append(item);save(out/'runs.json',runs);print('DONE',tag,item['global_status'],'seconds',round(item['elapsed_s'],2),flush=True)
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    print('SEALED',flush=True)


if __name__=='__main__':main()
