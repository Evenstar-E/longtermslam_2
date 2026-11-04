"""All 24 assets, all 288 candidates, shared material evidence then full SE3."""
from pathlib import Path
import pickle,time,sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.run_typed_material_incidence_graph_v2 import mean_colors
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from inspection_decisions.sparse_material_evidence_v1 import material_states,PeriodModel,log_weights,pair_costs,reduce_slots,assign,refine,DT,marginal_cost


def main():
    torch.set_num_threads(1);cp=Path('configs/sparse_material_evidence_v1.json');options=read(cp);parent=Path(options['parent_root']);cfg={**read(parent/'config.json'),**options};cfg['evidence_temperature']=cfg['normal_residual_scale_m']**2+cfg['depth_noise_bound_m']**2;shard=int(sys.argv[1]);assert 0<=shard<cfg['execution_shards'];out=Path(cfg['output_root'])/f'shard_{shard}';assert not out.exists()
    for name,h in read(parent/'prediction_seal.json')['sha256'].items():assert sha(parent/name)==h
    for field in ['sources','inputs']:
        for name,h in read(parent/'manifest.json')[field].items():assert sha(name)==h
    out.mkdir(parents=True);save(out/'config.json',cfg);source=[cp,Path(__file__),Path('inspection_decisions/sparse_material_evidence_v1.py'),Path('scripts/check_sparse_material_evidence_v1.py'),Path('research/sparse_material_evidence_execution_contract_v1_2026-10-25.md')];inputs=[parent/'prediction_seal.json',parent/'manifest.json',Path('research/sparse_material_evidence_checks_v1.json')]+list(parent.glob('*/predictions.json'))+list(parent.glob('*/candidates.pkl'))+list(Path(cfg['observation_root']).glob('*/observations.pkl'))+list(Path(cfg['base_root']).glob('*/inference.pkl'))+list(Path(cfg['base_root']).glob('*/frame_*.npz'));limits=read(parent/'manifest.json')['motion_limits'];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in source},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'H_only_dictionary':True,'finite_states_not_complete_Bayes':True,'motion_limits':limits,'primary_stage':'final','same_candidates_not_equal_FLOPs':True});runs=[];begin=time.perf_counter();methods=cfg['methods']
    for ordinal,run in enumerate(read(parent/'runs.json')):
        if ordinal%cfg['execution_shards']!=shard:continue
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (parent/tag/'candidates.pkl').open('rb') as f:pools=pickle.load(f)
        with (Path(cfg['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (Path(cfg['base_root'])/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];colors=mean_colors(Path(cfg['base_root']),tag,data);old=read(parent/tag/'predictions.json');C=np.zeros((len(methods),24,24,24));choices=np.zeros(C.shape+(2,),np.int16);references=[];independent={mode:np.full((2,24,24),np.inf) for mode in ['ray','line']}
        for i in range(24):
            graph,states=material_states(data[0][i],cfg,run['seed']*100+i);D=torch.tensor(states,dtype=DT);H=PeriodModel(data[0][i],graph,cfg)
            with torch.inference_mode():hh=H.evaluate(torch.zeros(6,dtype=DT),D,('ray','line'))
            W={mode:log_weights(hh[mode][0].numpy(),cfg['evidence_temperature']) for mode in ['ray','line']};E={mode:np.zeros((2,288,cfg['material_states'])) for mode in ['ray','line']};U=np.zeros((2,288,cfg['material_states'],len(graph['faces'])),bool);rgb=np.zeros((2,288))
            for t in range(2):
                models=[PeriodModel(d,graph,cfg) for d in data[t+1]];assert len(pools[i][t])==288
                with torch.inference_mode():
                    for a,b in enumerate(pools[i][t]):
                        slot=b['slot'];x=torch.tensor(pose_vector(b['T'],centers[slot]),dtype=DT);vv=models[slot].evaluate(x,D,('ray','line'),True)
                        for mode in ['ray','line']:E[mode][t,a]=vv[mode][0].numpy()
                        U[t,a]=vv['ray'][1].numpy();rgb[t,a]=cfg['support_match_cap_m']**2/3*float(np.mean(((colors[0,i]-colors[t+1,slot])/255)**2))
                for mode in ['ray','line']:
                    unary=marginal_cost(E[mode][t],W[mode],cfg['evidence_temperature'])+rgb[t]
                    for slot in range(24):independent[mode][t,i,slot]=min(unary[a] for a,b in enumerate(pools[i][t]) if b['slot']==slot)
            for mi,method in enumerate(methods):
                mode='line' if method=='joint_line' else 'ray';cost=pair_costs(E[mode][0],E[mode][1],W[mode],cfg['evidence_temperature'],method,U[0],U[1])+rgb[0,:,None]+rgb[1,None,:];C[mi,i],choices[mi,i]=reduce_slots(cost,pools[i])
            np.savez_compressed(dest/f'asset_{i:02d}_evidence.npz',ray=E['ray'],line=E['line'],used=U,rgb=rgb,logw_ray=W['ray'],logw_line=W['line']);references.append({'asset':i,'graph':graph,'states':states,'H_ray_energy':hh['ray'][0].numpy(),'H_line_energy':hh['line'][0].numpy(),'logw_ray':W['ray'],'logw_line':W['line'],'H_effective_states_ray':float(1/np.sum(np.exp(W['ray'])**2)),'H_effective_states_line':float(1/np.sum(np.exp(W['line'])**2))})
            if i%4==3:print('EVIDENCE',tag,i+1,'seconds',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'references.json',references);np.savez_compressed(dest/'joint_costs.npz',cost=C,indices=choices,independent_ray=independent['ray'],independent_line=independent['line']);starts=[]
        for family in ['catalogue_plane','material_shared']:
            rr=[r for r in old if r['family']==family];starts.append(tuple(np.array([r['original']['slots'][t] for r in rr]) for t in range(2)))
            for m in ['typed_ray','line_joint']:starts.append(tuple(np.array([r['methods'][m]['stages']['final']['slots'][t] for r in rr]) for t in range(2)))
        for mode in ['ray','line']:starts.append(tuple(linear_sum_assignment(independent[mode][t])[1] for t in range(2)))
        save(dest/'assignment_starts.json',starts);assignments={}
        for mi,method in enumerate(methods):
            a,b,log=assign(C[mi],starts,cfg['assignment_rounds']);assignments[method]={'slots':np.array([a,b]),'log':log};print('ASSIGNED',tag,method,'gap',log['gap'],flush=True)
        save(dest/'assignments.json',assignments);rows=[]
        for i,reference in enumerate(references):
            row={'asset':i,'methods':{}}
            for mi,method in enumerate(methods):
                slots=assignments[method]['slots'][:,i];j,k=slots;aa,bb=choices[mi,i,j,k];b1,b2=pools[i][0][aa],pools[i][1][bb];x=np.r_[pose_vector(b1['T'],centers[j]),pose_vector(b2['T'],centers[k])];mode='line' if method=='joint_line' else 'ray';p,log=refine([data[1][j],data[2][k]],reference['graph'],reference['states'],reference['logw_'+mode],x,limits,cfg,method);row['methods'][method]={'slots':slots,'candidate_indices':[int(aa),int(bb)],'initial_parameters':x,'final_parameters':p,'log':log,'stages':{'candidate':{'slots':slots,'T1':b1['T'],'T2':b2['T']},'final':{'slots':slots,'T1':world_pose(p[:6],centers[j]),'T2':world_pose(p[6:],centers[k])}}}
            rows.append(row)
            if i%6==5:print('REFINED',tag,i+1,'seconds',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',rows);item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(elapsed_s=time.perf_counter()-clock,assets=24,methods=len(methods),candidate_geometries=24*2*288,material_state_factor_evaluations=24*2*288*cfg['material_states']*2,continuous_fits=24*len(methods));runs.append(item);save(out/'runs.json',runs);print('DONE',tag,item,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'subset_shard':shard,'GT_read':False,'elapsed_s':time.perf_counter()-begin,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SHARD_SEALED',shard,flush=True)


if __name__=='__main__':main()
