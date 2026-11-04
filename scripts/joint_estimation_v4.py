"""SE3-invariant correspondence proposals; inference never sees evaluator truth."""
import itertools,time
import numpy as np
from scripts.joint_estimation import assign,fit_rigid,transform

def solve(obs,cfg,sampler,seed,method='geometry',diagnostics=False):
    begin=time.perf_counter();ref=obs['history_positions'].mean(axis=0);features=obs['history_descriptors'].mean(axis=0);current=obs['current_positions'];desc=obs['current_descriptors']
    appearance=np.sum((desc[:,None]-features[None])**2,axis=2)/cfg['descriptor_scale']**2;A0=assign(appearance)
    rng=np.random.default_rng(seed);triples=[rng.choice(len(current),3,replace=False) for _ in range(cfg['proposal_count'])]
    # Separate random stream avoids changing sampled source triples across ablations.
    choice_rng=np.random.default_rng(seed+99091);k=min(sampler['top_k'],len(ref));top=np.argsort(appearance,axis=1,kind='stable')[:,:k]
    starts=[obs['initial_T']];enumerated=0;fallbacks=0;rows=np.array([0,0,1]);cols=np.array([1,2,2]);eps=cfg['geometry_scale_m']**2
    for index,ids in enumerate(triples):
        hard=method=='hard' or (method=='hybrid' and index<sampler['hybrid_hard_proposals'])
        mapping=A0[ids]
        if not hard:
            combos=np.array(list(itertools.product(*(top[i] for i in ids))))
            unique=(combos[:,0]!=combos[:,1])&(combos[:,0]!=combos[:,2])&(combos[:,1]!=combos[:,2])
            combos=combos[unique];p=ref[combos];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
            combos=combos[area>eps];p=ref[combos];enumerated+=len(combos)
            source_area=np.linalg.norm(np.cross(current[ids[1]]-current[ids[0]],current[ids[2]]-current[ids[0]]))
            if len(combos) and source_area>eps:
                if method=='topk_random':mapping=combos[choice_rng.integers(len(combos))]
                else:
                    target_lengths=np.linalg.norm(p[:,rows]-p[:,cols],axis=2);source_lengths=np.linalg.norm(current[ids[rows]]-current[ids[cols]],axis=1)
                    invariant=np.sum(((target_lengths-source_lengths)/cfg['geometry_scale_m'])**2,axis=1)
                    descriptor=appearance[ids[None,:],combos].sum(axis=1)
                    mapping=combos[np.argmin(invariant+cfg['descriptor_weight']*descriptor)]
            else:fallbacks+=1
        starts.append(fit_rigid(current[ids],ref[mapping]))
    proposal_seconds=time.perf_counter()-begin;solutions=[]
    def cost(T):return cfg['descriptor_weight']*appearance+(np.linalg.norm(transform(current,T)[:,None]-ref[None],axis=2)/cfg['geometry_scale_m'])**2
    for start in starts:
        T=start.copy()
        for _ in range(cfg['iterations']):
            A=assign(cost(T));r=np.linalg.norm(transform(current,T)-ref[A],axis=1);T=fit_rigid(current,ref[A],1/(1+(r/cfg['switch_scale_m'])**4))
        A=assign(cost(T));r=np.linalg.norm(transform(current,T)-ref[A],axis=1)
        score=float((cfg['descriptor_weight']*appearance[np.arange(len(A)),A]+np.minimum((r/cfg['geometry_scale_m'])**2,cfg['truncation_cost'])).sum())
        solutions.append(dict(T=T,assignment=A,residual_m=r,score=score))
    selected=min(range(len(solutions)),key=lambda i:solutions[i]['score']);s=solutions[selected]
    pred=dict(T=s['T'],assignment=s['assignment'],changed=s['residual_m']>cfg['change_threshold_m'],residual_m=s['residual_m'],hypotheses=len(starts),proposal_seconds=proposal_seconds,refinement_seconds=time.perf_counter()-begin-proposal_seconds,enumerated_triples=enumerated,fallbacks=fallbacks)
    if diagnostics:pred.update(initial_poses=starts,solutions=solutions,selected_index=selected)
    return pred

def coverage(pred,truth,evaluation):
    """Evaluation only: classify search stages separately from final ranking."""
    from scipy.spatial.transform import Rotation
    def good(T):
        angle=float(np.rad2deg(Rotation.from_matrix(T[:3,:3]@truth['T'][:3,:3].T).magnitude()));distance=float(np.linalg.norm(T[:3,3]-truth['T'][:3,3]))
        return distance<=evaluation['pose_translation_tolerance_m'] and angle<=evaluation['pose_rotation_tolerance_deg']
    initials=[good(T) for T in pred['initial_poses']];finals=[good(s['T']) for s in pred['solutions']];chosen=finals[pred['selected_index']]
    return dict(initial_coverage=any(initials),final_coverage=any(finals),initial_good_count=sum(initials),final_good_count=sum(finals),selection_miss=bool(any(finals) and not chosen),lost_coverage=bool(any(initials) and not any(finals)))
