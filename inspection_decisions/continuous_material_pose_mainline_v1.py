"""Full-pool standard continuous geometric baselines for the actual main task.

No ground-truth identities, complete shapes or evaluation poses are accepted.
All distances use observed points; raster visibility is operational, not a
physical material certificate. These are ordinary controls, not a novel model.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scripts.completion_catalog_scene_v1 import inverse,transform
from inspection_decisions.first_return_material_surface_v1 import labels,project


def make_cloud(points,voxel,cfg):
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(points)
    if voxel:cloud=cloud.voxel_down_sample(voxel)
    if len(cloud.points):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(cfg['normal_min_radius_m'],voxel*cfg['normal_radius_voxel_multiplier']),max_nn=cfg['normal_max_neighbors']))
    return cloud


def observations(frames,slots,groups,cfg):
    data=[];clouds=[[] for _ in slots]
    for frame in frames:
        lab,world=labels(frame,slots,groups,cfg);data.append({**frame,'labels':lab})
        for i in range(len(slots)):clouds[i].append(world[lab==i])
    return data,[np.concatenate(p) for p in clouds]


def visibility(query,frames,slot,cfg):
    positive=absent=unknown=0
    for f in frames:
        camera,uv=project(query,f['C'],cfg)
        safe=np.where(np.isfinite(uv),uv,-1e8);pixel=np.rint(safe).astype(np.int64)
        u,v=pixel[:,0],pixel[:,1]
        inside=(camera[:,2]>0)&(u>=0)&(u<cfg['width'])&(v>=0)&(v<cfg['height'])
        depth=np.zeros(len(query));lab=np.full(len(query),-1,int)
        depth[inside]=f['depth'][v[inside],u[inside]];lab[inside]=f['labels'][v[inside],u[inside]]
        known=inside&np.isfinite(depth)&(depth>0);near=known&(np.abs(camera[:,2]-depth)<=cfg['depth_tolerance_m'])
        match=near&(lab==slot);free=(known&(camera[:,2]<depth-cfg['depth_tolerance_m']))|(near&(lab!=slot))
        positive+=int(match.sum());absent+=int(free.sum());unknown+=int((~(match|free)).sum())
    total=len(query)*len(frames)
    return {'positive':positive,'absence':absent,'unknown':unknown,'total':total,'score':float((absent-positive)/total) if total else 0.}


class PairModel:
    def __init__(self,historical,T0,history_frames,current_world,current_frames,asset,slot,slot_center,cfg,hcloud=None,ccloud=None):
        self.h=np.asarray(historical);self.T0=T0;self.hf=history_frames;self.cw=np.asarray(current_world);self.cf=current_frames
        self.asset=asset;self.slot=slot;self.center=np.asarray(slot_center);self.c=self.cw-self.center;self.cfg=cfg
        self.ht=cKDTree(self.h);self.ct=cKDTree(self.c)
        self.hn=np.asarray((hcloud if hcloud is not None else make_cloud(self.h,0.,cfg)).normals)
        self.cn=np.asarray((ccloud if ccloud is not None else make_cloud(self.c,0.,cfg)).normals)

    def score(self,T):
        q=inverse(self.cw,T);forward=transform(self.h,T)-self.center;scale=self.cfg['distance_scale_m']
        if not len(self.c) or not len(self.h):
            d1=d2=np.array([np.inf]);e1=e2=np.ones(1)
        else:
            d1,i1=self.ht.query(q,workers=1);d2,i2=self.ct.query(forward,workers=1)
            delta1=q-self.h[i1];delta2=forward-self.c[i2]
            e1=np.where(d1<=scale,np.minimum(np.abs(np.sum(delta1*self.hn[i1],1))/scale,1)**2,1.)
            e2=np.where(d2<=scale,np.minimum(np.abs(np.sum(delta2*self.cn[i2],1))/scale,1)**2,1.)
        one=np.minimum(d1/scale,1)**2;two=np.minimum(d2/scale,1)**2
        hs=visibility(transform(q,self.T0),self.hf,self.asset,self.cfg)
        cs=visibility(transform(self.h,T),self.cf,self.slot,self.cfg)
        scores={'nn':float(one.mean()),'symmetric_nn':float((one.mean()+two.mean())/2),
                'plane':float((e1.mean()+e2.mean())/2),'visibility':float((hs['score']+cs['score'])/2)}
        return scores,{'current_to_history':hs,'history_to_current':cs}


def refine(T,source_pyramid,target_pyramid,center,cfg):
    local=np.asarray(T).copy();local[:3,3]-=center;trace=[];success=True
    for level,(voxel,iterations) in enumerate(zip(cfg['voxel_scales_m'],cfg['max_iterations'])):
        maximum=max(voxel*cfg['correspondence_distance_voxel_multiplier'],cfg['minimum_correspondence_distance_m'])
        error=None;stat={}
        try:
            if min(len(source_pyramid[level].points),len(target_pyramid[level].points))<3:raise ValueError('fewer than three observed points')
            ans=o3d.pipelines.registration.registration_icp(source_pyramid[level],target_pyramid[level],maximum,local,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,relative_rmse=1e-6,max_iteration=iterations))
            proposal=np.asarray(ans.transformation)
            stat={'fitness':float(ans.fitness),'inlier_rmse':float(ans.inlier_rmse),'correspondences':len(ans.correspondence_set),'T_local':proposal.tolist()}
            if not np.isfinite(proposal).all() or len(ans.correspondence_set)<3:raise ValueError('nonfinite pose or fewer than three correspondences')
            if not np.allclose(proposal[:3,:3].T@proposal[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(proposal[:3,:3])<0:raise ValueError('nonrigid numerical output')
            local=proposal.copy()
        except (RuntimeError,ValueError) as exc:error=str(exc);success=False
        trace.append({'level':level,'voxel_m':voxel,'max_correspondence_m':maximum,'max_iterations':iterations,'error':error,**stat})
        if not success:break
    if not success:return None,trace
    world=local.copy();world[:3,3]+=center
    return world,trace


def select(candidates,score,expanded=True):
    n=len(candidates);selected=np.zeros((n,2),int);objective=0.
    for t in [0,1]:
        costs=np.full((n,n),1e6);indices=np.full((n,n),-1,int)
        for i,bb in enumerate(candidates):
            for k,b in enumerate(bb[t]):
                if not expanded and b['source']!='catalogue':continue
                if b['scores'][score]<costs[i,b['slot']]:costs[i,b['slot']]=b['scores'][score];indices[i,b['slot']]=k
        row,col=linear_sum_assignment(costs)
        if (indices[row,col]<0).any():return {'status':2,'selection':None,'objective':None}
        selected[row,t]=indices[row,col];objective+=float(costs[row,col].sum())
    return {'status':0,'selection':[[i,int(k),int(l)] for i,(k,l) in enumerate(selected)],'objective':objective}
