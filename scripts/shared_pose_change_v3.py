"""Shared camera/local-change optimization with matched direct-gating controls."""
import time
import numpy as np
from scipy.optimize import least_squares
from scripts.partial_surface_estimation import perturb,evidence,signed_box_distance
from scripts.joint_estimation import transform
from scripts.surface_motion_v2 import context,visibility,residual


def robust_energy(r,scale):
    return float(np.sum(2*scale**2*(np.sqrt(1+(r/scale)**2)-1)))


def choose_changed(e0,efree,cfg):
    gain=np.asarray(e0)-np.asarray(efree)-cfg['motion_penalty_m2']
    changed=gain>0
    # At least this many unchanged assets anchor world/camera gauge.
    force=np.argsort(gain,kind='stable')[:min(cfg['minimum_stable_objects'],len(gain))]
    changed[force]=False
    return changed,gain


def starts(obs,T,cfg,seed):
    rng=np.random.default_rng(seed+2203);out=[T.copy(),obs['initial_T'].copy()]
    for _ in range(cfg['starts']-2):
        v=np.r_[rng.normal(0,cfg['random_start_translation_m'],3),rng.normal(0,np.deg2rad(cfg['random_start_rotation_deg']),3)]
        out.append(perturb(T,v))
    return out


def solve(obs,base,bcfg,mcfg,cfg,method,seed):
    begin=time.perf_counter();A=np.array(base['prediction_assignment']);n=len(A);initial=np.array(base['T'])
    ctx=[context(obs,i,int(j),mcfg) for i,j in enumerate(A)];scale=.03/np.sqrt(mcfg['current_max_points']);bound=mcfg['translation_bound_m'];penalty=cfg['motion_penalty_m2']
    def select(T,delta):return [visibility(c,obs,T,d,mcfg,True) for c,d in zip(ctx,delta)]
    def data(i,T,d,sel):return residual(ctx[i],obs,T,d,int(A[i]),sel,mcfg,'visible_bidirectional')
    def local_fit(i,T,d,sel):
        fit=least_squares(lambda x:np.r_[data(i,T,x,sel),mcfg['regularization']*x],d,bounds=(-bound,bound),loss='soft_l1',f_scale=scale,max_nfev=cfg['max_nfev'])
        return fit.x
    if method=='frozen_v1':
        T=initial;pc=np.array(base['prediction_changed']);known=np.array(base['known']);delta=np.zeros((n,3));sel=select(T,delta)
        for i in np.flatnonzero(pc&known):delta[i]=local_fit(int(i),T,delta[i],sel[i])
        return dict(T=T,assignment=A,changed=pc,known=known,delta=delta,seconds=time.perf_counter()-begin,selected_start=0,scores=[],stable_count=int((~pc).sum()),hypothesis_changed=pc)
    solutions=[]
    for start in starts(obs,initial,cfg,seed):
        T=start.copy();delta=np.zeros((n,3));pc=np.zeros(n,bool)
        for _ in range(cfg['outer_rounds']):
            sel=select(T,delta)
            if method=='joint_profile':
                free=np.array([local_fit(i,T,delta[i],sel[i]) for i in range(n)])
                e0=[robust_energy(data(i,T,np.zeros(3),sel[i]),scale) for i in range(n)]
                efree=[robust_energy(np.r_[data(i,T,free[i],sel[i]),mcfg['regularization']*free[i]],scale) for i in range(n)]
                pc,gain=choose_changed(e0,efree,cfg);pc&=np.linalg.norm(free,axis=1)>cfg['minimum_motion_m'];delta=np.where(pc[:,None],free,0)
            else:
                score,known,_,_=evidence(obs,T,A,bcfg);pc=(score>bcfg['change_threshold_m'])&known
                # Same stable-count prior, using observed ray score for fallback.
                if (~pc).sum()<cfg['minimum_stable_objects']:pc[np.argsort(score,kind='stable')[:cfg['minimum_stable_objects']]]=False
                delta[:]=0
            sel=select(T,delta)
            def pose_residual(v):
                trial=perturb(T,v);parts=[]
                for i in range(n):
                    if method!='joint_profile' and pc[i]:continue
                    if method=='ray_gate_surface':parts.append(signed_box_distance(transform(ctx[i]['camera_points'][ctx[i]['take']],trial),obs['bounds'][A[i]])/np.sqrt(len(ctx[i]['take'])))
                    else:parts.append(data(i,trial,delta[i],sel[i]))
                return np.concatenate(parts)
            fit=least_squares(pose_residual,np.zeros(6),loss='soft_l1',f_scale=scale,max_nfev=cfg['max_nfev']);T=perturb(T,fit.x)
        score,known,_,_=evidence(obs,T,A,bcfg);sel=select(T,delta)
        free=np.array([local_fit(i,T,delta[i],sel[i]) for i in range(n)])
        e0=np.array([robust_energy(data(i,T,np.zeros(3),sel[i]),scale) for i in range(n)])
        efree=np.array([robust_energy(np.r_[data(i,T,free[i],sel[i]),mcfg['regularization']*free[i]],scale) for i in range(n)])
        if method=='joint_profile':
            pc,gain=choose_changed(e0,efree,cfg);pc&=np.linalg.norm(free,axis=1)>cfg['minimum_motion_m'];pc&=known
        else:pc=(score>bcfg['change_threshold_m'])&known
        delta=np.where(pc[:,None],free,0)
        total=float(np.where(pc,efree+penalty,e0).sum())
        solutions.append(dict(T=T,assignment=A,changed=pc,known=known,delta=delta,score=total,stable_count=int((~pc).sum()),hypothesis_changed=pc.copy()))
    index=int(np.argmin([s['score'] for s in solutions]));best=solutions[index]
    best.update(seconds=time.perf_counter()-begin,selected_start=index,scores=[s['score'] for s in solutions])
    return best
