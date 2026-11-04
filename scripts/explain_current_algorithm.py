"""Numerical trace of a saved actual v4 case. Truth only annotates explanation."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from scripts.joint_estimation_v4 import solve
from scripts.joint_estimation import transform,fit_rigid,assign

def main():
    base=Path('results/joint_proposals_holdout_v1');sid=250;path=base/'scenes'/f'{sid:04d}'/'measurements.npz';a=np.load(path);obs={k:a[k] for k in ['history_positions','history_descriptors','current_positions','current_descriptors','initial_T']}
    cfg=json.loads(Path('results/joint_estimation_v1/frozen_config.json').read_text())['algorithm'];sampler=json.loads(Path('configs/joint_proposals_v4.json').read_text());seed=21011001+sid+400000
    p=solve(obs,cfg,sampler,seed,'geometry',True);ref=obs['history_positions'].mean(axis=0);feature=obs['history_descriptors'].mean(axis=0);current=obs['current_positions'];appearance=((obs['current_descriptors'][:,None]-feature[None])**2).sum(axis=2)/cfg['descriptor_scale']**2
    top=np.argsort(appearance,axis=1,kind='stable')[:,:4];rng=np.random.default_rng(seed);triples=[rng.choice(len(current),3,replace=False) for _ in range(24)];traces=[];row=np.array([0,0,1]);col=np.array([1,2,2])
    for index,ids in enumerate(triples):
        if np.linalg.norm(np.cross(current[ids[1]]-current[ids[0]],current[ids[2]]-current[ids[0]]))<=cfg['geometry_scale_m']**2:continue
        combos=np.array(list(itertools.product(*(top[i] for i in ids))));unique=(combos[:,0]!=combos[:,1])&(combos[:,0]!=combos[:,2])&(combos[:,1]!=combos[:,2]);combos=combos[unique];points=ref[combos];area=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1);combos=combos[area>cfg['geometry_scale_m']**2];points=ref[combos]
        if not len(combos):continue
        source=np.linalg.norm(current[ids[row]]-current[ids[col]],axis=1);dist=np.linalg.norm(points[:,row]-points[:,col],axis=2);edge=((dist-source)/cfg['geometry_scale_m'])**2;cost=edge.sum(axis=1)+cfg['descriptor_weight']*appearance[ids[None,:],combos].sum(axis=1);chosen=int(np.argmin(cost));mapping=combos[chosen]
        initial=fit_rigid(current[ids],ref[mapping]);np.testing.assert_allclose(initial,p['initial_poses'][index+1],atol=1e-9)
        traces.append(dict(proposal_index=index+1,current_indices=ids.tolist(),top4=top[ids].tolist(),hard_mapping=assign(appearance)[ids].tolist(),historical_mapping=mapping.tolist(),current_edges_m=source.tolist(),historical_edges_m=dist[chosen].tolist(),cost=float(cost[chosen]),evaluator_current_ids=a['truth_assignment'][ids].tolist(),source_contains_changed=bool(a['truth_changed'][ids].any())))
    chosen=next((r for r in traces if r['proposal_index']==p['selected_index']),traces[0])
    after=transform(current,p['T']);before=transform(current,obs['initial_T']);r=p['residual_m'];examples=[]
    for changed in [False,True]:
        candidates=np.flatnonzero((p['assignment']==a['truth_assignment'])&(a['truth_changed']==changed))
        if len(candidates):
            i=int(candidates[0]);j=int(p['assignment'][i]);res=float(r[i]);weight=1/(1+(res/cfg['switch_scale_m'])**4)
            examples.append(dict(current_index=i,historical_id=j,reference_position=ref[j].tolist(),position_before=before[i].tolist(),position_after=after[i].tolist(),residual_m=res,pose_fit_weight=weight,reported_changed=bool(p['changed'][i]),evaluator_changed=bool(a['truth_changed'][i])))
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scene_id=sid,selected_proposal=p['selected_index'],triangle=chosen,examples=examples,scope='Actual frozen v4 run replay; evaluator IDs only label the explanation, not algorithm inputs')
    out=Path(f'results/current_algorithm_trace_v4_{sid:04d}');out.mkdir(exist_ok=False);(out/'trace.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
