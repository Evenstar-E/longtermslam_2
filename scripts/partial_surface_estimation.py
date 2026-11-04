"""Surface-based candidate: component-group stable/change switching and translations.

Established robust estimation ingredients, not a claim of literature originality.
Inference functions accept observations and calibration only; oracle use in runner.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scripts.joint_estimation import transform,assign
from scripts.partial_surface_scene import intersect


def signed_box_distance(p,box):
    mid=box.mean(axis=0);half=(box[1]-box[0])/2;q=np.abs(p-mid)-half
    return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)


def perturb(T,v):
    E=np.eye(4);E[:3,:3]=Rotation.from_rotvec(v[3:]).as_matrix();E[:3,3]=v[:3]
    return E@T


def correspondence(obs,T):
    current=np.array([transform(q['points'],T).mean(axis=0) for q in obs['query']])
    cost=np.sum((current[:,None]-obs['bounds'].mean(axis=1)[None])**2,axis=2)
    features=np.array([q['feature'] for q in obs['query']]);cost+=4*np.sum((features[:,None]-obs['reference_features'][None])**2,axis=2)
    return assign(cost)


def evidence(obs,T,A,cfg):
    """One-sided current surface SDF plus visibility-qualified reference rays.

    Reference rays behind measured foreground are occluded, never 'disappeared'.
    Surface change test uses only historically supported faces. It is not a
    calibrated significance test or an absent-object detector.
    """
    scores=[];known=[];support_fractions=[];occluded_counts=[]
    wr=obs['rays']@T[:3,:3].T
    for q,j in zip(obs['query'],A):
        p=transform(q['points'],T);box=obs['bounds'][j]
        # Closest face of reconstructed reference; no true current face labels.
        distances=np.abs(p[:,:,None]-box.T[None]).reshape(len(p),6)
        closest=distances.argmin(axis=1);supported=obs['face_support'][j,closest]
        fraction=float(supported.mean());support_fractions.append(fraction)
        current_score=float(np.median(np.abs(signed_box_distance(p[supported],box)))) if supported.any() else 0.
        # Explicit ray visibility diagnostic using the entire observed depth map.
        d,face=intersect(T[:3,3],wr,*box);measured=obs['depth'].ravel()
        visible=np.isfinite(d)&(measured>0)&obs['face_support'][j,face]
        member=np.zeros(len(d),bool);member[q.get('all_pixels',q.get('pixels',[]))]=True
        occluded=visible&(measured<d-cfg['change_threshold_m'])&~member
        testable=visible&~occluded
        # The upper quartile captures a vacated edge under lateral motion;
        # zero/invalid returns and unsupported historical faces never count.
        ray_score=float(np.quantile(np.minimum(np.abs(measured[testable]-d[testable]),1.),.75)) if testable.sum()>=8 else 0.
        scores.append(max(current_score,ray_score))
        known.append(bool(fraction>=cfg['minimum_supported_fraction'] and supported.sum()>=8 and testable.sum()>=8))
        occluded_counts.append(int(occluded.sum()))
    return np.array(scores),np.array(known),np.array(support_fractions),occluded_counts


def infer(obs,cfg,method,pose_override=None):
    begin=time.perf_counter();T=obs['initial_T'].copy();n=len(obs['query']);weights=np.ones(n);delta=np.zeros((n,3))
    if pose_override is not None:T=pose_override.copy()
    iterations=cfg['outer_iterations'] if method in ('surface_robust','surface_joint') else 0
    for _ in range(iterations):
        A=correspondence(obs,T)
        if method=='surface_joint':
            r=np.array([np.median(np.abs(signed_box_distance(transform(q['points'],T),obs['bounds'][j]))) for q,j in zip(obs['query'],A)])
            # Group switching prevents a moved component becoming many pose anchors.
            weights=1/(1+(r/cfg['group_scale_m'])**4)
        def residual(v):
            trial=perturb(T,v)
            return np.concatenate([np.sqrt(weights[i])*signed_box_distance(transform(q['points'],trial),obs['bounds'][j])/np.sqrt(len(q['points'])) for i,(q,j) in enumerate(zip(obs['query'],A))])
        # Scale adjusted for per-object balancing; same point loss for both methods.
        fit=least_squares(residual,np.zeros(6),loss='soft_l1',f_scale=cfg['surface_scale_m']/np.sqrt(cfg['points_per_object']),max_nfev=cfg['max_nfev'])
        T=perturb(T,fit.x)
    A=correspondence(obs,T);score,known,fraction,occluded=evidence(obs,T,A,cfg)
    if method=='surface_joint':
        for i,(q,j) in enumerate(zip(obs['query'],A)):
            if score[i]<=cfg['change_threshold_m'] or not known[i]:continue
            p=transform(q['points'],T);box=obs['bounds'][j]
            # Minimal regularised local translation; pose stays fixed to stable support.
            fit=least_squares(lambda d:np.r_[signed_box_distance(p-d,box),.05*d],np.zeros(3),bounds=(-.7,.7),loss='soft_l1',f_scale=cfg['surface_scale_m'],max_nfev=cfg['max_nfev'])
            delta[i]=fit.x
    return dict(T=T,assignment=A,changed=(score>cfg['change_threshold_m'])&known,known=known,residual_m=score,support_fraction=fraction,occluded_pixels=occluded,delta=delta,stable_weights=weights,seconds=time.perf_counter()-begin)
