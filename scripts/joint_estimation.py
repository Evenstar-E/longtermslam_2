"""Experimental joint SE3 / assignment / changed-anchor switching prototype.

Inference accepts observations only. No true pose, IDs or change labels.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


def fit_rigid(source,target,weights=None):
    weights=np.ones(len(source)) if weights is None else np.asarray(weights)
    if weights.sum()<1e-8:return np.eye(4)
    weights=weights/weights.sum();s=weights@source;t=weights@target
    H=((source-s)*weights[:,None]).T@(target-t)
    u,_,vt=np.linalg.svd(H);R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t-R@s
    return T


def transform(points,T):return points@T[:3,:3].T+T[:3,3]


def assign(cost):
    rows,cols=linear_sum_assignment(cost)
    assert np.array_equal(rows,np.arange(cost.shape[0]))
    return cols


def infer(obs,cfg,method,seed):
    memory=obs['history_descriptors'];positions=obs['history_positions']
    if method=='joint_no_history':memory=memory[:1];positions=positions[:1]
    ref=positions.mean(axis=0);features=memory.mean(axis=0);current=obs['current_positions'];desc=obs['current_descriptors']
    feature_cost=np.sum((desc[:,None]-features[None])**2,axis=2)/cfg['descriptor_scale']**2
    appearance_assignment=assign(feature_cost)
    initial=np.array(obs['initial_T']);rng=np.random.default_rng(seed)
    proposals=[initial]
    # Identical proposal budget for robust sequential and multi-start joint methods.
    for _ in range(cfg['proposal_count']):
        ids=rng.choice(len(current),3,replace=False)
        proposals.append(fit_rigid(current[ids],ref[appearance_assignment[ids]]))
    def residual_matrix(T):return np.linalg.norm(transform(current,T)[:,None]-ref[None],axis=2)
    def joint_cost(T,gate=True):
        r=residual_matrix(T);g=(r/cfg['geometry_scale_m'])**2
        if gate:g=np.minimum(g,cfg['truncation_cost'])
        return cfg['descriptor_weight']*feature_cost+g
    if method=='sequential':
        A=appearance_assignment;T=fit_rigid(current,ref[A]);hypotheses=1
    elif method=='robust_sequential':
        A=appearance_assignment
        T=min(proposals,key=lambda t:np.minimum((np.linalg.norm(transform(current,t)-ref[A],axis=1)/cfg['geometry_scale_m'])**2,cfg['truncation_cost']).sum())
        for _ in range(cfg['iterations']):
            r=np.linalg.norm(transform(current,T)-ref[A],axis=1);weights=1/(1+(r/cfg['switch_scale_m'])**4)
            T=fit_rigid(current,ref[A],weights)
        hypotheses=len(proposals)
    else:
        starts=proposals[:1] if method=='joint_single' else proposals
        gate=method!='joint_no_change_gate';solutions=[]
        for T in starts:
            T=T.copy()
            for _ in range(cfg['iterations']):
                A=assign(joint_cost(T,gate));r=np.linalg.norm(transform(current,T)-ref[A],axis=1)
                weights=1/(1+(r/cfg['switch_scale_m'])**4) if gate else np.ones(len(r))
                T=fit_rigid(current,ref[A],weights)
            A=assign(joint_cost(T,gate));score=float(joint_cost(T,gate)[np.arange(len(A)),A].sum())
            solutions.append((score,T,A))
        _,T,A=min(solutions,key=lambda r:r[0]);hypotheses=len(solutions)
    residual=np.linalg.norm(transform(current,T)-ref[A],axis=1)
    return dict(T=T,assignment=A,changed=residual>cfg['change_threshold_m'],residual_m=residual,hypotheses=hypotheses)


def simulate(cfg,condition,seed):
    rng=np.random.default_rng(seed);n=cfg['objects'];d=cfg['descriptor_dim'];h=cfg['history_views']
    # Repeated but not perfectly symmetric support layout, randomized per scene.
    ref=np.array([[1.25*(i%6),1.4*(i//6),.25*((i//3)%2)] for i in range(n)],float)
    ref+=rng.normal(0,.08,(n,3))
    repeated=condition['repeated_appearance'];groups=rng.normal(0,.8,(4,d))
    latent=groups[np.arange(n)%4]+rng.normal(0,.06 if repeated else .65,(n,d))
    hist_pos=ref[None]+rng.normal(0,cfg['position_noise_m'],(h,n,3))
    hist_desc=latent[None]+rng.normal(0,cfg['descriptor_noise'],(h,n,d))
    changed=np.zeros(n,bool);chosen=rng.choice(n,int(round(n*condition['change_fraction'])),replace=False);changed[chosen]=True
    target=ref.copy();directions=rng.normal(size=(len(chosen),3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    target[chosen]+=directions*rng.uniform(*cfg['change_displacement_m'],size=(len(chosen),1))
    true_T=np.eye(4);true_T[:3,:3]=Rotation.from_euler('xyz',rng.uniform(-.3,.3,3)).as_matrix();true_T[:3,3]=rng.uniform(-2,2,3)
    visible=rng.choice(n,n-int(round(n*condition['missing_fraction'])),replace=False);rng.shuffle(visible)
    current=transform(target[visible],np.linalg.inv(true_T))+rng.normal(0,cfg['position_noise_m'],(len(visible),3))
    current_desc=latent[visible]+rng.normal(0,cfg['descriptor_noise'],(len(visible),d))
    prefix='pose_'+condition['pose_error'];error=np.eye(4);error[:3,:3]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(cfg[prefix+'_rotation_sigma_deg']),3)).as_matrix();error[:3,3]=rng.normal(0,cfg[prefix+'_translation_sigma_m'],3)
    obs=dict(history_positions=hist_pos,history_descriptors=hist_desc,current_positions=current,current_descriptors=current_desc,initial_T=error@true_T)
    truth=dict(T=true_T,assignment=visible,changed=changed[visible])
    return obs,truth


def evaluate(pred,truth,ev):
    R=pred['T'][:3,:3]@truth['T'][:3,:3].T
    angle=float(np.rad2deg(Rotation.from_matrix(R).magnitude()));translation=float(np.linalg.norm(pred['T'][:3,3]-truth['T'][:3,3]))
    a=pred['assignment']==truth['assignment'];pc=pred['changed'];tc=truth['changed']
    # Wrong-identity positive reports count as both false report and missed correct report.
    tp=int((pc&tc&a).sum());fp=int((pc&(~tc|~a)).sum());fn=int((tc&(~pc|~a)).sum());tn=int((~pc&~tc&a).sum())
    return dict(translation_error_m=translation,rotation_error_deg=angle,pose_success=bool(translation<=ev['pose_translation_tolerance_m'] and angle<=ev['pose_rotation_tolerance_deg']),
       assignment_accuracy=float(a.mean()),tp=tp,fp=fp,fn=fn,tn=tn,visible=len(a),changed_visible=int(tc.sum()),
       false_change_on_unchanged=int((pc&~tc).sum()),unchanged_visible=int((~tc).sum()),
       f1=2*tp/max(2*tp+fp+fn,1))
