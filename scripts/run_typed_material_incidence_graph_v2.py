"""Full pool -> global identity assignment -> continuous material re-estimation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pickle,time,sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save
from scripts.completion_catalog_scene_v1 import catalog
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from inspection_decisions.typed_material_incidence_graph_v1 import (
    JointProblem,build_typed_graph,period_arrays,period_geometry,graph_arrays,period_cost)
from inspection_decisions.typed_material_incidence_fast_v2 import period_arrays,period_geometry,graph_arrays,period_cost
from inspection_decisions.uncertain_face_incidence_schur_v2 import fit
from scripts.check_typed_material_incidence_graph_v1 import checks


def prediction(p,slots,centers):return {'T1':world_pose(p[:6],centers[slots[0]]),'T2':world_pose(p[6:12],centers[slots[1]]),'slots':list(map(int,slots))}


def mean_colors(base,tag,data):
    result=[]
    for t in range(3):
        images=[]
        for path in sorted((base/tag).glob(f'frame_{t}_*.npz')):
            with np.load(path) as z:images.append(z['rgb'].copy())
        result.append([np.mean([images[p['view']][p['pixel'][1],p['pixel'][0]] for p in d['pixels']],axis=0) for d in data[t]])
    return np.array(result)


def pool_for(i,t,cached,parents,cases,centers):
    pool=[{'slot':int(b['slot']),'T':np.array(b['T']),'source':'retained_'+b['source'],'cached_index':k,'catalogue':k<54} for k,b in enumerate(cached[i][t][:108])];assert len(pool)==108
    def key(slot,T):return (slot,tuple(np.round(T,12).ravel()))
    known={key(b['slot'],b['T']) for b in pool if b['catalogue']}
    for slot in range(24):
        for k,C in enumerate(catalog()):
            T=C.copy();T[:3,3]+=centers[slot]
            if key(slot,T) not in known:pool.append({'slot':slot,'T':T,'source':'all_slots_catalogue','catalogue':True,'catalogue_index':k});known.add(key(slot,T))
    for family in ['catalogue_plane','material_shared']:
        old=parents[(i,family)];slot=old['selected_slots'][t]
        for method,pred in old['predictions'].items():pool.append({'slot':slot,'T':np.array(pred['T'+str(t+1)]),'source':'parent_'+family+'_'+method,'catalogue':False})
        case=cases[(i,family)];slot=case['initial_slots'][t]
        for method,pred in case['conditional_predictions'].items():pool.append({'slot':slot,'T':np.array(pred['T'+str(t+1)]),'source':'new_'+family+'_'+method,'catalogue':False})
    return pool


def main():
    cp=Path('configs/typed_material_incidence_graph_v2.json');options=read(cp);shard=int(sys.argv[1]);assert 0<=shard<options['execution_shards'];parent=Path(options['parent_root']);obsroot=Path(options['observation_root']);cachedroot=Path(options['candidate_root']);cfg={**read(obsroot/'config.json'),**options,'execution_shard':shard};out=Path(cfg['output_root'])/('shard_'+str(shard));base=Path(cfg['base_root']);preflight=checks()
    for root in [parent,obsroot,cachedroot]:
        for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    out.mkdir(parents=True,exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',preflight);sources=[cp,Path(__file__),Path('inspection_decisions/typed_material_incidence_graph_v1.py'),Path('inspection_decisions/typed_material_incidence_fast_v2.py'),Path('scripts/check_typed_material_incidence_fast_v2.py'),Path('inspection_decisions/uncertain_face_incidence_schur_v2.py'),Path('scripts/check_typed_material_incidence_graph_v1.py'),Path('research/typed_material_incidence_execution_contract_v1_2026-10-25.md'),Path('research/typed_material_execution_backend_v2_2026-10-25.md')];inputs=[Path('research/typed_material_fast_equivalence_v2.json'),parent/'prediction_seal.json',obsroot/'prediction_seal.json',cachedroot/'prediction_seal.json',cachedroot/'manifest.json']+list(parent.glob('*/predictions.json'))+list(obsroot.glob('*/observations.pkl'))+list(cachedroot.glob('*/candidates.pkl'))+list(base.glob('*/inference.pkl'))+list(base.glob('*/frame_*.npz'));limits=read(cachedroot/'manifest.json')['motion_limits'];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'familiar_DEV':True,'motion_limits':limits,'IDs_reselected':True,'no_new_top_k_or_semantic_gate':True,'all24_slots_catalogue':True,'not_equal_FLOPs':True,'execution_shard':shard});runs=[];start=time.perf_counter();families=['catalogue_plane','material_shared'];methods=cfg['methods']
    for run in [r for j,r in enumerate(read(parent/'runs.json')) if j%cfg['execution_shards']==shard]:
        tag=run['scene'];dest=out/tag;dest.mkdir();clock=time.perf_counter()
        with (obsroot/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
        with (cachedroot/tag/'candidates.pkl').open('rb') as f:cached=pickle.load(f)
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        centers=inp['T0'][:,:3,3];parents={(r['asset'],r['family']):r for r in read(parent/tag/'predictions.json')};arrays=[[period_arrays(d) for d in tt] for tt in data];colors=mean_colors(base,tag,data);cases={};unary=np.full((2,len(methods),2,24,24),np.inf);indices=np.full(unary.shape,-1,int);catunary=unary.copy();catindices=indices.copy();all_pools=[]
        with ThreadPoolExecutor(max_workers=cfg['score_workers']) as executor:
            for i in range(24):
                scoring=[]
                for fi,family in enumerate(families):
                    old=parents[(i,family)];slots=old['selected_slots'];track=[data[0][i],data[1][slots[0]],data[2][slots[1]]];initial=np.array(old['initial_parameters']);clipped,typed=build_typed_graph([(np.zeros(6),track[0]),(initial[:6],track[1])],cfg);case={'asset':i,'family':family,'initial_slots':slots,'initial_parameters':initial,'graphs':{'clipped':clipped,'typed':typed},'initial_fits':{},'conditional_predictions':{},'original':{'T1':old['predictions']['original']['T1'],'T2':old['predictions']['original']['T2'],'slots':slots},'methods':{}}
                    for mi,method in enumerate(methods):
                        graph=clipped if method=='clipped_ray' else typed;problem=JointProblem(track,graph,initial,limits,cfg,method);p,log=fit(problem,problem.start(),cfg['initial_iterations']);case['initial_fits'][method]={'parameters':p,'log':log};case['conditional_predictions'][method]=prediction(p,slots,centers);scoring.append((fi,mi,graph_arrays(graph,p[12:]),method=='line_joint'))
                    cases[(i,family)]=case
                pools=[pool_for(i,t,cached,parents,cases,centers) for t in [0,1]];all_pools.append(pools)
                for t,pool in enumerate(pools):
                    def score_candidate(b):
                        slot=b['slot'];geo=period_geometry(arrays[t+1][slot],pose_vector(b['T'],centers[slot]),cfg);appearance=cfg['support_match_cap_m']**2/3*float(np.mean(((colors[0,i]-colors[t+1,slot])/255)**2));return [(fi,mi,period_cost(geo,g,cfg,line)+appearance) for fi,mi,g,line in scoring]
                    for k,values in enumerate(executor.map(score_candidate,pool)):
                        slot=pool[k]['slot']
                        for fi,mi,value in values:
                            if value<unary[fi,mi,t,i,slot]-1e-12:unary[fi,mi,t,i,slot]=value;indices[fi,mi,t,i,slot]=k
                            if pool[k]['catalogue'] and value<catunary[fi,mi,t,i,slot]-1e-12:catunary[fi,mi,t,i,slot]=value;catindices[fi,mi,t,i,slot]=k
                if i%4==3:print('SCORED_POOLS',tag,i+1,'seconds',round(time.perf_counter()-clock,1),flush=True)
        with (dest/'candidates.pkl').open('wb') as f:pickle.dump(all_pools,f)
        np.savez_compressed(dest/'global_costs.npz',unary=unary,indices=indices,catalogue_unary=catunary,catalogue_indices=catindices)
        assignments={}
        for fi,family in enumerate(families):
            for mi,method in enumerate(methods):
                choices={}
                for name,costs,ids in [('pool',unary,indices),('catalogue',catunary,catindices)]:
                    chosen=[]
                    for t in [0,1]:
                        row,col=linear_sum_assignment(costs[fi,mi,t]);assert np.array_equal(row,np.arange(24));assert np.isfinite(costs[fi,mi,t,row,col]).all();chosen.append([(int(s),int(ids[fi,mi,t,i,s])) for i,s in enumerate(col)])
                    choices[name]=chosen
                assignments[(family,method)]=choices
        for i in range(24):
            for family in families:
                case=cases[(i,family)];oldslots=case['initial_slots'];graphs=case['graphs']
                for method in methods:
                    selected={};stage={}
                    for name in ['pool','catalogue']:
                        picked=[assignments[(family,method)][name][t][i] for t in [0,1]];b=[all_pools[i][t][k] for t,(slot,k) in enumerate(picked)];stage[name]={'T1':b[0]['T'],'T2':b[1]['T'],'slots':[slot for slot,k in picked]};selected[name]={'indices':[k for slot,k in picked],'slots':stage[name]['slots']}
                    slots=stage['pool']['slots'];track=[data[0][i],data[1][slots[0]],data[2][slots[1]]];initial=np.r_[pose_vector(stage['pool']['T1'],centers[slots[0]]),pose_vector(stage['pool']['T2'],centers[slots[1]])];graph=graphs['clipped'] if method=='clipped_ray' else graphs['typed'];problem=JointProblem(track,graph,initial,limits,cfg,method);p=problem.start();p[12:]=case['initial_fits'][method]['parameters'][12:];p,log=fit(problem,p,cfg['reassigned_iterations']);stage['conditional']=case['conditional_predictions'][method];stage['final']=prediction(p,slots,centers);case['methods'][method]={'stages':stage,'selection':selected,'final_parameters':p,'log':log,'initial_parameters':initial}
            if i%6==5:print('REFINED',tag,i+1,'seconds',round(time.perf_counter()-clock,1),flush=True)
        save(dest/'predictions.json',[cases[(i,f)] for f in families for i in range(24)]);item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(elapsed_s=time.perf_counter()-clock,initializations=48,initial_fits=48*len(methods),reassigned_fits=48*len(methods),candidate_evaluations=sum(len(p) for pools in all_pools for p in pools)*len(methods)*2,pool_sizes=sorted({len(p) for pp in all_pools for p in pp}));runs.append(item);save(out/'runs.json',runs);print('DONE',tag,item,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'subset_shard':shard,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SHARD_SEALED',shard,flush=True)


if __name__=='__main__':main()
