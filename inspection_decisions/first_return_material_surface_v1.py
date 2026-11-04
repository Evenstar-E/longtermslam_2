"""Observed continuous patches and shared historical material-ray bounds.

Standard mesh raycasting/visibility controls, not a new reconstruction theory.
No full shapes, current identities, or evaluation truth are accepted here.
"""
import numpy as np
from scipy.ndimage import minimum_filter
import open3d as o3d
from scripts.completion_catalog_scene_v1 import inverse, transform, PALETTE


def labels(frame, slots, groups, cfg):
    d=frame['depth']; h,w=d.shape; yy,xx=np.mgrid[:h,:w]; f=cfg['focal_px']
    color=np.linalg.norm(frame['rgb'].astype(float)[:,:,None,:]-PALETTE[None,None,:,:],axis=-1)
    group=color.argmin(-1); z=d.ravel()
    world=transform(np.c_[((xx.ravel()-(w-1)/2)/f)*z,((yy.ravel()-(h-1)/2)/f)*z,z],frame['C'])
    dist=np.linalg.norm(world[:,None,:]-slots[None,:,:],axis=-1)
    dist+=np.where(group.ravel()[:,None]==groups[None,:],0,100)
    slot=dist.argmin(1); valid=(color.min(-1).ravel()<8)&(dist[np.arange(len(z)),slot]<.9)
    slot[~valid]=-1
    return slot.reshape(h,w),world.reshape(h,w,3)


class Patch:
    def __init__(self, vertices, triangles):
        self.vertices=np.asarray(vertices,float).reshape(-1,3)
        self.triangles=np.asarray(triangles,np.uint32).reshape(-1,3)
        self.scene=None
        if len(self.triangles):
            self.scene=o3d.t.geometry.RaycastingScene(nthreads=1)
            self.scene.add_triangles(o3d.core.Tensor(self.vertices.astype(np.float32)),
                                     o3d.core.Tensor(self.triangles))


def build_patches(frames, slots, groups, cfg):
    n=len(slots); clouds=[[] for _ in range(n)]; faces=[[] for _ in range(n)]; offsets=np.zeros(n,int)
    for frame in frames:
        lab,world=labels(frame,slots,groups,cfg); h,w=lab.shape
        for slot in range(n):
            valid=lab==slot; index=np.full((h,w),-1,int); index[valid]=np.arange(valid.sum())
            xyz=world[valid]; clouds[slot].append(xyz)
            for ay,ax,by,bx,cy,cx in [(0,0,0,1,1,1),(0,0,1,1,1,0)]:
                aa=index[ay:ay+h-1,ax:ax+w-1];bb=index[by:by+h-1,bx:bx+w-1];cc=index[cy:cy+h-1,cx:cx+w-1]
                ids=np.stack([aa,bb,cc],-1); ok=(ids>=0).all(-1)
                tri=ids[ok]
                if len(tri):
                    depths=frame['depth'][valid][tri]
                    tri=tri[np.ptp(depths,axis=1)<=cfg['surface_depth_edge_m']]
                faces[slot].append(tri+offsets[slot])
            offsets[slot]+=len(xyz)
    return [Patch(np.concatenate(clouds[i]),np.concatenate(faces[i])) for i in range(n)]


def project(x, C, cfg):
    p=inverse(x,C); f=cfg['focal_px'];h,w=cfg['height'],cfg['width']
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=p[:,:2]/p[:,2,None]*f+np.array([(w-1)/2,(h-1)/2])
    return p,uv


def point_violation(source, M, targets, cfg):
    world=transform(source.vertices,M); maximum=0.;count=0
    for frame in targets:
        if 'point_min' not in frame:frame['point_min']=minimum_filter(frame['depth'],size=3,mode='constant',cval=0.)
        p,uv=project(world,frame['C'],cfg); pix=np.rint(uv).astype(int)
        inside=(p[:,2]>0)&(pix[:,0]>=0)&(pix[:,0]<cfg['width'])&(pix[:,1]>=0)&(pix[:,1]<cfg['height'])
        pix=pix[inside];z=p[inside,2]
        delta=frame['point_min'][pix[:,1],pix[:,0]]-z
        count+=int(np.sum(delta>cfg['depth_tolerance_m']))
        maximum=max(maximum,float(np.max(delta,initial=0.)))
    return count,maximum


def surface_violation(source, M, targets, cfg):
    if source.scene is None:return 0,0.
    world=transform(source.vertices,M); maximum=0.;count=0
    for frame in targets:
        p,uv=project(world,frame['C'],cfg)
        if not len(p) or (p[:,2]<=0).any():
            # This experiment's meshes are always in front of all cameras.
            if len(p):raise ValueError('Mesh crosses camera plane; clipping not implemented')
            continue
        low=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0])
        high=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[cfg['width']-1,cfg['height']-1])
        if (low>high).any():continue
        yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        rc=np.c_[(xx.ravel()-(cfg['width']-1)/2)/cfg['focal_px'],
                 (yy.ravel()-(cfg['height']-1)/2)/cfg['focal_px'],np.ones(xx.size)]
        # Cast actual target pixel rays in SOURCE WORLD coordinates. Camera-Z
        # parameter is preserved by rigid transforms; directions not normalized.
        origin=inverse(frame['C'][:3,3][None],M)[0]
        direction=rc@frame['C'][:3,:3].T@M[:3,:3]
        rays=np.c_[np.broadcast_to(origin,direction.shape),direction].astype(np.float32)
        z=source.scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
        delta=frame['depth'][yy.ravel(),xx.ravel()]-z
        count+=int(np.sum(delta>cfg['depth_tolerance_m']))
        maximum=max(maximum,float(np.max(delta,initial=0.)))
    return count,maximum


def bidirectional(source, target, Ts, Tt, source_frames, target_frames, cfg, surface=False):
    M=Tt@np.linalg.inv(Ts);fun=surface_violation if surface else point_violation
    a=fun(source,M,target_frames,cfg);b=fun(target,np.linalg.inv(M),source_frames,cfg)
    return {'ok':a[0]+b[0]==0,'violations':a[0]+b[0],'max_front_gap_m':max(a[1],b[1])}


def historical_ray(anchor,T0,frames,cfg):
    for v,frame in enumerate(frames):
        p,uv=project(transform(anchor[None],T0),frame['C'],cfg);u,y=np.rint(uv[0]).astype(int)
        if 0<=u<cfg['width'] and 0<=y<cfg['height']:
            depth=float(frame['depth'][y,u]);rc=np.array([(u-(cfg['width']-1)/2)/cfg['focal_px'],(y-(cfg['height']-1)/2)/cfg['focal_px'],1.])
            o=inverse(frame['C'][:3,3][None],T0)[0];r=rc@frame['C'][:3,:3].T@T0[:3,:3]
            if np.linalg.norm(o+depth*r-anchor)<1e-8:
                return {'view':v,'pixel':[int(u),int(y)],'origin':o.tolist(),'ray':r.tolist(),'depth':depth}
    raise ValueError('Historical anchor source pixel not recovered')


def material_envelope(branches, anchor, ray, feasible, cfg):
    lows=[];highs=[];r=np.array(ray['ray']);eps=cfg['depth_noise_bound_m']
    for k,l in np.argwhere(feasible):
        A,B=branches[0][k]['T'],branches[1][l]['T'];D=B[:3,:3]-A[:3,:3]
        d=D@anchor+B[:3,3]-A[:3,3];extra=eps*np.abs(D@r)
        lows.append(d-extra);highs.append(d+extra)
    if not lows:return {'empty':True,'report':False,'lower':None,'upper':None,'radius_m':None}
    lo=np.min(lows,axis=0);hi=np.max(highs,axis=0);radius=float(np.linalg.norm(hi-lo)/2)
    return {'empty':False,'report':radius<=.05,'lower':lo.tolist(),'upper':hi.tolist(),'radius_m':radius}
