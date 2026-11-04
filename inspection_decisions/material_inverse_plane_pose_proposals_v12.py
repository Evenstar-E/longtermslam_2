"""Measured inverse-plane continuous proposals and shared-source likelihood probe.

No catalogue geometry, material labels or reference poses are read here.
"""
import numpy as np
from inspection_decisions.material_inverse_plane_control_v11 import (
    project, TARGET_OFFSETS, TARGET_PINV)
from inspection_decisions.material_oriented_surface_control_v10 import supports
from inspection_decisions.material_surface_patch_registration_v9 import (
    inverse, retract6)
from inspection_decisions.material_correspondence_mode_graph_v6 import skew


def project_derivatives(world, frame, slot):
    """Piecewise exact depth/color derivatives, fixed nearest-pixel plane cell."""
    r=project(world,frame,slot);x=np.asarray(world).reshape(-1,3)
    C,K=frame['C'],frame['K'];p=(x-C[:3,3])@C[:3,:3];z=np.maximum(p[:,2],1e-8)
    u=p[:,0]/z*K[0,0]+K[0,2];v=p[:,1]/z*K[1,1]+K[1,2]
    H,W=frame['depth'].shape;cx=np.clip(np.rint(u).astype(int),1,W-2);cy=np.clip(np.rint(v).astype(int),1,H-2)
    X=cx[:,None]+TARGET_OFFSETS[:,0];Y=cy[:,None]+TARGET_OFFSETS[:,1]
    coeff=(1/np.maximum(frame['depth'][Y,X],1e-8))@TARGET_PINV.T
    q=coeff[:,0]*(u-cx)+coeff[:,1]*(v-cy)+coeff[:,2];D=1/np.maximum(q,1e-8)
    J=np.zeros((len(x),2,3));J[:,0,0]=K[0,0]/z;J[:,0,2]=-K[0,0]*p[:,0]/z**2
    J[:,1,1]=K[1,1]/z;J[:,1,2]=-K[1,1]*p[:,1]/z**2;J=J@C[:3,:3].T
    # gap = D(u,v)-z. Oriented normal and gap Jacobian have different scales.
    gapJ=-D[:,None]**2*np.einsum('ni,nij->nj',coeff[:,:2],J)-C[:3,2]
    xx=np.clip(np.floor(u).astype(int),0,W-2);yy=np.clip(np.floor(v).astype(int),0,H-2)
    du=u-xx;dv=v-yy;X4=np.stack([xx,xx+1,xx,xx+1],1);Y4=np.stack([yy,yy,yy+1,yy+1],1)
    colors=frame['rgb'][Y4,X4]/255.
    gx=(1-dv[:,None])*(colors[:,1]-colors[:,0])+dv[:,None]*(colors[:,3]-colors[:,2])
    gy=(1-du[:,None])*(colors[:,2]-colors[:,0])+du[:,None]*(colors[:,3]-colors[:,1])
    rgbJ=np.einsum('nci,nij->ncj',np.stack([gx,gy],2),J)
    r.update(gapJ=gapJ.reshape(*world.shape[:-1],3),rgbJ=rgbJ.reshape(*world.shape[:-1],3,3))
    return r


def continuous_fit(scorer,T,cfg,column):
    """Ordinary symmetric RGBD GN directions with exact chosen-score backtracking.

Normals only gate measured material correspondences. Occlusion/unknown cells
produce no fictitious depth or texture residual. Patch means use ONLY the
currently selected valid pixels; the old all-five mean defect is not inherited.
"""
    T=T.copy();current=float(scorer.score(T[None])[0,column]);path=[current]
    for _ in range(cfg['continuous_steps']):
        AA=[];rr=[]
        for direction,(a,frames,slot) in enumerate([(scorer.a,scorer.fb,scorer.ib),(scorer.b,scorer.fa,scorer.ia)]):
            x=a['patch_xyz'].reshape(-1,3);color=a['patch_rgb'].reshape(-1,3)
            U=inverse(T) if direction else T;world=x@U[:3,:3].T+U[:3,3]
            J=np.concatenate([-skew(world),np.broadcast_to(np.eye(3),(len(world),3,3))],2)
            if direction:J=-np.einsum('ij,njk->nik',T[:3,:3].T,np.concatenate([-skew(x),np.broadcast_to(np.eye(3),(len(x),3,3))],2))
            normal=np.repeat(a['patch_normal']@U[:3,:3].T,5,0);normal_valid=np.repeat(a['patch_normal_valid'],5)
            choices=[project_derivatives(world,f,slot) for f in frames]
            if not choices:continue
            admissible=[]
            for r in choices:
                comp=(np.sum(normal*r['normal'],1)>=cfg['normal_cosine_min'])|~normal_valid|~r['normal_valid']
                admissible.append(r['surface']&comp&(abs(r['gap'])<.05))
            pick=np.array([np.where(ok,abs(r['gap']),np.inf) for ok,r in zip(admissible,choices)]).argmin(0)
            for fi,r in enumerate(choices):
                take=admissible[fi]&(pick==fi);ii=np.flatnonzero(take)
                if not len(ii):continue
                residual=r['gap'][ii];jac=np.einsum('ni,nij->nj',r['gapJ'][ii],J[ii]);weight=(1+(residual/.02)**2)**-.25
                AA.append(jac*weight[:,None]/.02);rr.append(residual*weight/.02)
                residual=(r['rgb']-color).reshape(-1,5,3);jac=np.einsum('nci,nij->ncj',r['rgbJ'],J).reshape(-1,5,3,6)
                if column==3:
                    mask=take.reshape(-1,5);den=np.maximum(mask.sum(1),1)
                    residual-=((residual*mask[...,None]).sum(1)/den[:,None])[:,None]
                    jac-=((jac*mask[...,None,None]).sum(1)/den[:,None,None])[:,None]
                residual=residual.reshape(-1,3)[ii];jac=jac.reshape(-1,3,6)[ii]
                weight=(1+np.mean(residual**2,1)/cfg['color_scale']**2)**-.25
                AA.append((jac*weight[:,None,None]/cfg['color_scale']).reshape(-1,6));rr.append((residual*weight[:,None]/cfg['color_scale']).ravel())
        if not AA:break
        A=np.concatenate(AA);b=np.concatenate(rr);d=np.linalg.lstsq(np.r_[A,.01*np.eye(6)],np.r_[-b,np.zeros(6)],rcond=None)[0]
        scale=min(1,.04/max(np.linalg.norm(d[:3]),1e-12),.02/max(np.linalg.norm(d[3:]),1e-12));d*=scale
        trial=np.array([retract6(T,alpha*d) for alpha in [1,.5,.25,.125]])
        values=scorer.score(trial)[:,column];h=int(values.argmin())
        if values[h]>=current-1e-12:break
        T=trial[h];current=float(values[h]);path.append(current)
    return T,np.array(path)


def source_support(a,frames,slot,T,cfg):
    """Per H patch support in each target: common measured source index retained."""
    world=np.einsum('bij,npj->bnpi',T[:,:3,:3],a['patch_xyz'])+T[:,None,None,:3,3]
    normal=np.einsum('bij,nj->bni',T[:,:3,:3],a['patch_normal'])
    best=np.zeros((len(T),len(a['patch_xyz'])))
    for f in frames:
        r=project(world,f,slot);comp=(np.sum(normal[:,:,None]*r['normal'],axis=-1)>=cfg['normal_cosine_min'])|~a['patch_normal_valid'][None,:,None]|~r['normal_valid']
        hit=r['surface']&comp&(abs(r['gap'])<=cfg['ray_tolerance_m'])
        value=supports(a['patch_rgb'][None],r['rgb'],hit,r['gap'],cfg)[...,0]
        best=np.maximum(best,value)
    return best


def persistence_factors(s1,s2,cfg):
    """Marginalize separate vs common latent Bernoulli persistence analytically.

These are bounded pseudo-likelihoods of measured supports, not calibrated
probabilities. Identical fixed domain/null/prior for both controls. Missing
support never counts as matching. No learned persistence claims.
"""
    eta=cfg['material_null_likelihood'];pi=cfg['shared_prior']
    l1=eta+(1-eta)*s1;l2=eta+(1-eta)*s2
    independent=(pi*l1+(1-pi)*eta)*(pi*l2+(1-pi)*eta)
    shared=pi*l1*l2+(1-pi)*eta**2
    return -np.stack([np.log(independent),np.log(shared)],-1).mean(-2)/(-2*np.log(eta))


class TripleScore:
    def __init__(self,scorers,cfg):self.scorers=scorers;self.cfg=cfg
    def score(self,T):
        result=[]
        for begin in range(0,len(T),32):
            U=T[begin:begin+32];a,b,c=self.scorers
            pair=a.score(U[:,0])[:,2:]+b.score(U[:,1])[:,2:]+.5*c.score(U[:,1]@inverse(U[:,0]))[:,2:]
            s1=source_support(a.a,a.fb,a.ib,U[:,0],self.cfg);s2=source_support(b.a,b.fb,b.ib,U[:,1],self.cfg)
            factors=persistence_factors(s1,s2,self.cfg) if s1.shape[1] else np.ones((len(U),2))
            result.append(np.c_[pair,pair[:,1,None]+self.cfg['material_factor_weight']*factors])
        return np.concatenate(result)


def joint_fit(scorer,T,cfg,column):
    """Equal-budget ordinary/shared 12D local descent on the actual objective.

Both latent models generate candidates for the common final union. Numerical
central differences are deliberately explicit; no surrogate convergence claim.
"""
    T=T.copy();value=float(scorer.score(T[None])[0,column]);path=[value]
    unit=np.tile([.005,.005,.005,.002,.002,.002],2)
    for _ in range(cfg['shared_steps']):
        trial=[]
        for k in range(12):
            for sign in [1,-1]:
                U=T.copy();d=np.zeros(6);d[k%6]=sign*unit[k];U[k//6]=retract6(U[k//6],d);trial.append(U)
        f=scorer.score(np.array(trial))[:,column].reshape(12,2);gradient=(f[:,0]-f[:,1])/2
        norm=np.linalg.norm(gradient)
        if norm<1e-10:break
        d=-gradient/norm*unit;trial=[]
        for alpha in [2,1,.5,.25]:trial.append(np.array([retract6(T[e],alpha*d[e*6:e*6+6]) for e in range(2)]))
        vals=scorer.score(np.array(trial))[:,column];h=int(vals.argmin())
        if vals[h]>=value-1e-12:break
        T=trial[h];value=float(vals[h]);path.append(value)
    return T,np.array(path)


def unique_poses(poses):
    out=[]
    for T in poses:
        if not any(np.max(abs(T-U))<1e-9 for U in out):out.append(np.array(T))
    return np.array(out)


def propagation(A,B,C):
    """C maps C1->C2. Each row is a coupled H1,H2 mode, never fake endpoints."""
    return unique_poses([np.array([a,c@a]) for a in A for c in C]+[np.array([inverse(c)@b,b]) for b in B for c in C])
