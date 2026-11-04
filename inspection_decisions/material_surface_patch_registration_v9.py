"""Actual RGB-D surface patches and censor-aware continuous projective matching.

FPFH/FGR, ICP, Colored ICP, projective RGB-D alignment and NCC are established
tools. This module is a task-model experiment, not a novelty declaration.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from inspection_decisions.textured_material_matching_v1 import world_points,apply
from inspection_decisions.material_correspondence_inference_v3 import inverse
from inspection_decisions.material_correspondence_mode_graph_v6 import skew

def sample_surfaces(frames,cfg):
    """Voxel representatives remain real pixels. Patches keep their source view."""
    buckets=[[] for _ in range(24)]
    for fi,f in enumerate(frames):
        yy,xx=np.nonzero((f['anonymous_slot']>=0)&f['depth_valid']);xyz=world_points(f,np.c_[xx,yy]);ids=f['anonymous_slot'][yy,xx]
        for slot in np.unique(ids):
            take=ids==slot;buckets[int(slot)].append((xyz[take],np.c_[xx[take],yy[take]],np.full(int(take.sum()),fi)))
    out=[]
    for slot,parts in enumerate(buckets):
        xyz=np.concatenate([p[0] for p in parts]);uv=np.concatenate([p[1] for p in parts]);frame=np.concatenate([p[2] for p in parts]);_,ix=np.unique(np.floor(xyz/cfg['surface_voxel_m']).astype(int),axis=0,return_index=True);ix=ix[np.argsort(ix)]
        if len(ix)>cfg['surface_points']:
            rng=np.random.default_rng(cfg['seed']+slot);ix=np.sort(rng.choice(ix,cfg['surface_points'],replace=False))
        xyz,uv,frame=xyz[ix],uv[ix],frame[ix];color=np.array([frames[int(fi)]['rgb'][p[1],p[0]] for p,fi in zip(uv,frame)],float)/255
        candidates=[]
        for h,(p,fi) in enumerate(zip(uv,frame)):
            f=frames[int(fi)];pix=p+np.array(cfg['patch_offsets']);H,W=f['depth'].shape
            if (pix<0).any() or (pix[:,0]>=W).any() or (pix[:,1]>=H).any():continue
            x,y=pix.T
            if not (f['depth_valid'][y,x]&(f['anonymous_slot'][y,x]==slot)).all():continue
            z=f['depth'][y,x]
            if z.max()-z.min()>.025:continue
            candidates.append(h)
        candidates=np.array(candidates,int)
        if len(candidates)>cfg['patches']:candidates=candidates[np.linspace(0,len(candidates)-1,cfg['patches'],dtype=int)]
        px=[];pc=[];pu=[];pf=[]
        for h in candidates:
            pix=uv[h]+np.array(cfg['patch_offsets']);f=frames[int(frame[h])];px.append(world_points(f,pix));pc.append(f['rgb'][pix[:,1],pix[:,0]]/255.);pu.append(pix);pf.append(frame[h])
        out.append(dict(xyz=xyz,rgb=color,uv=uv,frame=frame,patch_xyz=np.array(px).reshape(-1,5,3),patch_rgb=np.array(pc).reshape(-1,5,3),patch_uv=np.array(pu).reshape(-1,5,2),patch_frame=np.array(pf,int)))
    return out

def make_cloud(record):
    import open3d as o3d
    p=o3d.geometry.PointCloud();p.points=o3d.utility.Vector3dVector(record['xyz']);p.colors=o3d.utility.Vector3dVector(record['rgb']);return p

def geometry_features(record,cfg):
    import open3d as o3d
    p=make_cloud(record);p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['normal_radius_m'],max_nn=24));norm=np.asarray(p.normals).copy()
    # Observed source camera orients every normal, rather than catalogue axes.
    view=record['camera']-record['xyz'];norm[np.sum(norm*view,axis=1)<0]*=-1;p.normals=o3d.utility.Vector3dVector(norm)
    f=o3d.pipelines.registration.compute_fpfh_feature(p,o3d.geometry.KDTreeSearchParamHybrid(radius=cfg['fpfh_radius_m'],max_nn=64));return norm,np.asarray(f.data).copy()

def global_pose(a,b,cfg,seed):
    import open3d as o3d
    o3d.utility.random.seed(int(seed));pa,pb=make_cloud(a),make_cloud(b);fa,fb=o3d.pipelines.registration.Feature(),o3d.pipelines.registration.Feature();fa.data=a['fpfh'];fb.data=b['fpfh'];pa.normals=o3d.utility.Vector3dVector(a['normal']);pb.normals=o3d.utility.Vector3dVector(b['normal'])
    option=o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=.06,iteration_number=cfg['fgr_iterations'],maximum_tuple_count=500)
    return np.array(o3d.pipelines.registration.registration_fgr_based_on_feature_matching(pa,pb,fa,fb,option).transformation)

def icp_fit(a,b,T,cfg,colored=False):
    import open3d as o3d
    pa,pb=make_cloud(a),make_cloud(b);pa.normals=o3d.utility.Vector3dVector(a['normal']);pb.normals=o3d.utility.Vector3dVector(b['normal']);r=o3d.pipelines.registration
    criteria=r.ICPConvergenceCriteria(max_iteration=cfg['fit_steps'])
    if colored:
        try:return np.array(r.registration_colored_icp(pa,pb,.08,T,r.TransformationEstimationForColoredICP(),criteria).transformation),False
        except RuntimeError as ex:
            if 'No correspondences found' not in str(ex):raise
            return T.copy(),True
    return np.array(r.registration_icp(pa,pb,.08,T,r.TransformationEstimationPointToPlane(),criteria).transformation),False

def bilinear_projection(world,frame,slot,derivatives=False):
    """Four valid same-slot pixels define the continuous target surface/color."""
    shape=world.shape[:-1];x=world.reshape(-1,3);C,K=frame['C'],frame['K'];p=(x-C[:3,3])@C[:3,:3];z=p[:,2];u=p[:,0]/np.maximum(z,1e-8)*K[0,0]+K[0,2];v=p[:,1]/np.maximum(z,1e-8)*K[1,1]+K[1,2];H,W=frame['depth'].shape;inside=(z>0)&(u>=0)&(u<W-1)&(v>=0)&(v<H-1);xx=np.clip(np.floor(u).astype(int),0,W-2);yy=np.clip(np.floor(v).astype(int),0,H-2);du=u-xx;dv=v-yy
    weights=np.stack([(1-du)*(1-dv),du*(1-dv),(1-du)*dv,du*dv],1);X=np.stack([xx,xx+1,xx,xx+1],1);Y=np.stack([yy,yy,yy+1,yy+1],1);depth=frame['depth'][Y,X];valid=inside&frame['depth_valid'][Y,X].all(1)&(np.ptp(depth,axis=1)<.025);owned=(frame['anonymous_slot'][Y,X]==slot).all(1);surface=valid&owned;D=(weights*depth).sum(1);colors=frame['rgb'][Y,X]/255.;rgb=(weights[:,:,None]*colors).sum(1)
    result=dict(surface=surface.reshape(shape),valid=valid.reshape(shape),gap=(D-z).reshape(shape),rgb=rgb.reshape(*shape,3))
    if not derivatives:return result
    gx=(1-dv[:,None])*(colors[:,1]-colors[:,0])+dv[:,None]*(colors[:,3]-colors[:,2]);gy=(1-du[:,None])*(colors[:,2]-colors[:,0])+du[:,None]*(colors[:,3]-colors[:,1]);dx=(1-dv)*(depth[:,1]-depth[:,0])+dv*(depth[:,3]-depth[:,2]);dy=(1-du)*(depth[:,2]-depth[:,0])+du*(depth[:,3]-depth[:,1]);J=np.zeros((len(p),2,3));zz=np.maximum(z,1e-8);J[:,0,0]=K[0,0]/zz;J[:,0,2]=-K[0,0]*p[:,0]/zz**2;J[:,1,1]=K[1,1]/zz;J[:,1,2]=-K[1,1]*p[:,1]/zz**2;J=J@C[:3,:3].T
    normal=C[:3,2]-dx[:,None]*J[:,0]-dy[:,None]*J[:,1];rgbJ=np.einsum('nci,nij->ncj',np.stack([gx,gy],2),J)
    result.update(normal=normal.reshape(*shape,3),rgbJ=rgbJ.reshape(*shape,3,3));return result

def projection_terms(world,color,frames,slot,cfg):
    """Returns per-patch independent and spatial-pattern supports, plus free gap."""
    shape=world.shape[:-2];M=world.shape[-2];best_ind=np.zeros(shape);best_patch=np.zeros(shape);best_free=np.full(world.shape[:-1],np.inf);any_hit=np.zeros(world.shape[:-1],bool)
    for f in frames:
        r=bilinear_projection(world,f,slot);hit=r['surface']&(abs(r['gap'])<=cfg['ray_tolerance_m']);any_hit|=hit;gap=r['gap'];free=r['valid']&(gap>cfg['ray_tolerance_m']);best_free=np.minimum(best_free,np.where(free,gap-cfg['ray_tolerance_m'],np.inf));geometry=np.exp(-.5*(gap/cfg['ray_tolerance_m'])**2)*hit
        distance=np.mean((r['rgb']-color)**2,axis=-1);ind=(geometry*np.exp(-.5*distance/cfg['color_scale']**2)).mean(-1);best_ind=np.maximum(best_ind,ind)
        count=hit.sum(-1);den=np.maximum(count,1)[...,None,None];a=color;ma=(a*hit[...,None]).sum(-2,keepdims=True)/den;mb=(r['rgb']*hit[...,None]).sum(-2,keepdims=True)/den;ac=(a-ma)*hit[...,None];bc=(r['rgb']-mb)*hit[...,None];sa=(ac*ac).sum((-2,-1));sb=(bc*bc).sum((-2,-1));dot=(ac*bc).sum((-2,-1));corr=np.maximum(0.,dot/np.sqrt(np.maximum(sa*sb,1e-12)));flat=(sa<1e-5)&(sb<1e-5);corr=np.where(flat,np.exp(-.5*((ma-mb)**2).mean((-2,-1))/cfg['color_scale']**2),corr);support=geometry.mean(-1)*corr*(count>=3);best_patch=np.maximum(best_patch,support)
    best_free=np.where(np.isfinite(best_free)&~any_hit,best_free,0.);free=np.log1p((best_free/cfg['geometry_scale_m'])**2).mean((-1,-2));return best_ind.mean(-1),best_patch.mean(-1),free

class SurfaceScore:
    def __init__(self,a,b,fa,fb,ia,ib,cfg):
        self.a=a;self.b=b;self.fa=fa;self.fb=fb;self.ia=ia;self.ib=ib;self.cfg=cfg;self.ta=cKDTree(a['xyz']);self.tb=cKDTree(b['xyz'])
    def score(self,poses):
        result=[];a,b=self.a,self.b;cfg=self.cfg
        for start in range(0,len(poses),64):
            T=np.asarray(poses[start:start+64]);U=inverse(T);x=np.einsum('bij,nj->bni',T[:,:3,:3],a['xyz'])+T[:,None,:3,3];y=np.einsum('bij,nj->bni',U[:,:3,:3],b['xyz'])+U[:,None,:3,3];da,ia=self.tb.query(x);db,ib=self.ta.query(y);g=.5*(np.minimum(da/cfg['geometry_scale_m'],1).mean(1)+np.minimum(db/cfg['geometry_scale_m'],1).mean(1));ca=np.minimum(np.sqrt(np.mean((a['rgb'][None]-b['rgb'][ia])**2,axis=2))/cfg['color_scale'],1).mean(1);cb=np.minimum(np.sqrt(np.mean((b['rgb'][None]-a['rgb'][ib])**2,axis=2))/cfg['color_scale'],1).mean(1);fusion=g+.5*(ca+cb)
            if len(a['patch_xyz']) and len(b['patch_xyz']):
                pa=np.einsum('bij,npj->bnpi',T[:,:3,:3],a['patch_xyz'])+T[:,None,None,:3,3];pb=np.einsum('bij,npj->bnpi',U[:,:3,:3],b['patch_xyz'])+U[:,None,None,:3,3];aa=projection_terms(pa,a['patch_rgb'][None],self.fb,self.ib,cfg);bb=projection_terms(pb,b['patch_rgb'][None],self.fa,self.ia,cfg);free=.5*(aa[2]+bb[2]);ind=1-.5*(aa[0]+bb[0])+cfg['free_weight']*free;patch=1-.5*(aa[1]+bb[1])+cfg['free_weight']*free
            else:ind=patch=np.ones(len(T))
            result.append(np.c_[g,fusion,ind,patch])
        return np.concatenate(result)

def retract6(T,d):
    S=np.eye(4);S[:3,:3]=Rotation.from_rotvec(d[:3]).as_matrix();S[:3,3]=d[3:];return S@T

def projective_fit(scorer,T,cfg):
    """Normal-depth and locally normalized texture directions, actual-score descent."""
    T=T.copy();current=float(scorer.score(T[None])[0,3]);path=[current]
    for _ in range(cfg['fit_steps']):
        AA=[];rr=[]
        # Forward and inverse observations; both constrain the same six DoF.
        for direction,(a,frames,slot) in enumerate([(scorer.a,scorer.fb,scorer.ib),(scorer.b,scorer.fa,scorer.ia)]):
            x=a['patch_xyz'].reshape(-1,3);c=a['patch_rgb'].reshape(-1,3);U=inverse(T) if direction else T;z=apply(U,x);J=np.concatenate([-skew(z),np.broadcast_to(np.eye(3),(len(z),3,3))],axis=2)
            if direction:J=-np.einsum('ij,njk->nik',T[:3,:3].T,np.concatenate([-skew(x),np.broadcast_to(np.eye(3),(len(x),3,3))],2))
            choices=[bilinear_projection(z,f,slot,True) for f in frames]
            if not choices:continue
            # One best measured view per actual source sample; no duplicated factors.
            score=np.array([np.where(r['surface'],abs(r['gap']),np.inf) for r in choices]);pick=score.argmin(0)
            for fi,r in enumerate(choices):
                take=(pick==fi)&r['surface']&(abs(r['gap'])<.05);ii=np.flatnonzero(take)
                if not len(ii):continue
                D=-np.einsum('ni,nij->nj',r['normal'][ii],J[ii]);res=r['gap'][ii];w=1/np.sqrt(1+(res/.02)**2);AA.append(D*w[:,None]/.02);rr.append(res*w/.02)
                # Center within each five-pixel patch removes local color offset.
                color=(r['rgb']-c).reshape(-1,5,3);color-=color.mean(1,keepdims=True);cj=np.einsum('nci,nij->ncj',r['rgbJ'],J).reshape(-1,5,3,6);cj-=cj.mean(1,keepdims=True);res=color.reshape(-1,3)[ii];jac=cj.reshape(-1,3,6)[ii];w=1/np.sqrt(1+np.mean(res**2,axis=1)/cfg['color_scale']**2);AA.append((jac*w[:,None,None]/cfg['color_scale']).reshape(-1,6));rr.append((res*w[:,None]/cfg['color_scale']).ravel())
        if not AA:break
        A=np.concatenate(AA);b=np.concatenate(rr);d=np.linalg.lstsq(np.r_[A,np.eye(6)*.01],np.r_[-b,np.zeros(6)],rcond=None)[0];d*=min(1,.04/max(np.linalg.norm(d),1e-12));accepted=False
        for alpha in [1,.5,.25,.125]:
            U=retract6(T,d*alpha);value=float(scorer.score(U[None])[0,3])
            if value<current-1e-12:T=U;current=value;accepted=True;break
        path.append(current)
        if not accepted:break
    return T,np.array(path)
