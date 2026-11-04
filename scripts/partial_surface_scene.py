"""CPU depth renderer and observation-only reconstruction of nonplanar cuboids.

Renderer truth remains separate from the inference dictionary. The shared box
geometry is an intentionally favourable synthetic model, not a learned mapper.
"""
import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from scripts.joint_estimation import transform

PALETTE = np.array([[210,65,55],[55,190,80],[65,90,215],[205,175,45]], float)


def camera(position, target):
    z = np.asarray(target)-position; z = z/np.linalg.norm(z)
    x = np.cross([0,1,0], z); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    T = np.eye(4); T[:3,:3] = np.column_stack([x,y,z]); T[:3,3] = position
    return T


def rays(cfg):
    h,w = cfg['image_height'],cfg['image_width']; yy,xx = np.mgrid[:h,:w]
    return np.stack([(xx-(w-1)/2)/cfg['focal_px'],(yy-(h-1)/2)/cfg['focal_px'],np.ones_like(xx)],axis=-1).reshape(-1,3)


def intersect(origin, directions, lo, hi):
    """Ray slab intersection, depth parameter for unnormalised camera rays."""
    safe = np.where(np.abs(directions)>1e-10,directions,np.where(directions>=0,1e-10,-1e-10))
    a=(lo-origin)/safe; b=(hi-origin)/safe
    lower=np.minimum(a,b); upper=np.maximum(a,b)
    near=lower.max(axis=1); far=upper.min(axis=1)
    axis=lower.argmax(axis=1)
    face=2*axis+(directions[np.arange(len(directions)),axis]<0)
    depth=np.where((far>=near)&(near>0),near,np.inf)
    return depth,face


def render(boxes,T,groups,cfg,rng,occluder=False):
    cr=rays(cfg); wr=cr@T[:3,:3].T
    # A fixed enclosing room supplies valid far returns, rather than treating
    # missing depth as observed free space.
    wall=np.where(wr>0,30.,-30.)
    depth=np.min((wall-T[:3,3])/np.where(np.abs(wr)>1e-10,wr,1e-10),axis=1)
    identity=np.full(len(cr),-1)
    shapes=list(boxes)
    if occluder: shapes.append(np.array([[-.55,-1.8,2.7],[.10,1.8,3.0]]))
    for j,box in enumerate(shapes):
        d,_=intersect(T[:3,3],wr,*box); front=d<depth
        depth[front]=d[front];identity[front]=j if j<len(boxes) else -2
    rgb=np.full((len(cr),3),25.,float)
    for j,g in enumerate(groups):rgb[identity==j]=PALETTE[g]
    rgb[identity==-2]=110
    rgb=np.clip(rgb+rng.normal(0,2,rgb.shape),0,255).astype(np.uint8)
    valid=np.isfinite(depth);depth[valid]+=rng.normal(0,cfg['depth_noise_m'],valid.sum());depth[~valid]=0
    h,w=cfg['image_height'],cfg['image_width']
    return rgb.reshape(h,w,3),depth.reshape(h,w),identity.reshape(h,w)


def detect(rgb,depth,cfg):
    """Connected colour components; no renderer instance masks or IDs."""
    colour=rgb.astype(float);dist=np.linalg.norm(colour[:,:,None]-PALETTE,axis=3)
    group=dist.argmin(axis=2);best=dist.min(axis=2);out=[];cr=rays(cfg)
    for g in range(len(PALETTE)):
        labels,n=label((group==g)&(best<30)&(depth>0))
        for k in range(1,n+1):
            pix=np.flatnonzero(labels.ravel()==k)
            if len(pix)<cfg['minimum_points']:continue
            points=cr[pix]*depth.ravel()[pix,None]
            out.append(dict(pixels=pix,points=points,feature=np.eye(4)[g],group=g))
    # A foreground strip can split one instance; merge nearby fragments using
    # observed metric proximity, never renderer labels. Repeated palette objects
    # are separated by >1 m in this procedural generator.
    merged=[]
    for d in out:
        match=next((m for m in merged if m['group']==d['group'] and np.linalg.norm(m['points'].mean(0)-d['points'].mean(0))<.8),None)
        if match is None:merged.append(d)
        else:
            match['pixels']=np.r_[match['pixels'],d['pixels']]
            match['points']=np.concatenate([match['points'],d['points']])
    return merged


def make_scene(cfg,condition,seed):
    rng=np.random.default_rng(seed);n=cfg['objects']
    centres=np.array([[(j%4-1.5)*1.55,(j//4-.5)*1.5,5.5+.35*((j%3)-1)] for j in range(n)])
    centres+=rng.normal(0,.07,centres.shape);sizes=rng.uniform(.65,1.0,(n,3))
    boxes=np.stack([centres-sizes/2,centres+sizes/2],axis=1);groups=np.arange(n)%4
    target=np.array([0.,0.,5.5]);positions=[[-3,0,0],[3,0,0],[-6,1,5.5],[6,-1,5.5],[-2,0,11],[2,0,11]]
    clouds=[[] for _ in range(n)];centroid_views=[[] for _ in range(n)];history=[]
    for pos in positions:
        T=camera(np.array(pos,float),target);rgb,dep,gt=render(boxes,T,groups,cfg,rng)
        detections=detect(rgb,dep,cfg);history.append((rgb,dep,T))
        # Privileged known historical tracks author reference associations only.
        for d in detections:
            ids=gt.ravel()[d['pixels']];valid=ids>=0
            if not valid.any():continue
            owner=int(np.bincount(ids[valid],minlength=n).argmax())
            world=transform(d['points'],T);clouds[owner].append(world);centroid_views[owner].append(world.mean(axis=0))
    bounds=[];support=[];reference_centroids=[];historical_counts=[]
    for j in range(n):
        p=np.concatenate(clouds[j]);lo=np.quantile(p,.002,axis=0);hi=np.quantile(p,.998,axis=0)
        bounds.append([lo,hi]);historical_counts.append(len(p));reference_centroids.append(np.mean(centroid_views[j],axis=0))
        # Only faces actually supported by reference measurements are testable.
        face=np.argmin(np.abs(p[:,:,None]-np.stack([lo,hi],axis=1)[None]).reshape(len(p),6),axis=1)
        support.append(np.bincount(face,minlength=6)>=cfg['minimum_points'])
    angle=1 if condition['wide_view'] else .12
    pos=np.array([rng.choice([-1,1])*angle*3.6,rng.uniform(-.5,.5),rng.uniform(-.2,.2)])
    T=camera(pos,target);current_boxes=boxes.copy();changed=np.zeros(n,bool);delta=np.zeros((n,3))
    selected=rng.choice(n,condition['changed_objects'],replace=False);changed[selected]=True
    directions=rng.normal(size=(len(selected),3));directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-9)
    delta[selected]=directions*rng.uniform(.28,.45,(len(selected),1));current_boxes+=delta[:,None,:]
    rgb,dep,gt=render(current_boxes,T,groups,cfg,rng,condition['occluder']);detections=detect(rgb,dep,cfg);rng.shuffle(detections)
    query=[];ids=[]
    for d in detections:
        labels=gt.ravel()[d['pixels']];ids.append(int(np.bincount(labels[labels>=0],minlength=n).argmax()))
        take=np.linspace(0,len(d['points'])-1,min(cfg['points_per_object'],len(d['points'])),dtype=int)
        query.append(dict(points=d['points'][take],feature=d['feature'],pixels=d['pixels'][take],all_pixels=d['pixels']))
    E=np.eye(4);E[:3,:3]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(cfg['pose_rotation_sigma_deg']),3)).as_matrix();E[:3,3]=rng.normal(0,cfg['pose_translation_sigma_m'],3)
    obs=dict(bounds=np.array(bounds),face_support=np.array(support),reference_centroids=np.array(reference_centroids),reference_features=np.eye(4)[groups],query=query,initial_T=E@T,depth=dep,rgb=rgb,rays=rays(cfg),historical_counts=historical_counts)
    truth=dict(T=T,assignment=np.array(ids),changed=changed[np.array(ids)],delta=delta[np.array(ids)],boxes=boxes,current_boxes=current_boxes)
    visuals=dict(history=history,current=(rgb,dep,gt))
    return obs,truth,visuals
