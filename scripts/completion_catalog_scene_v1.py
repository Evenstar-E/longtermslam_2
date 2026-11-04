"""Calibrated 24-asset voxel-aligned rigid geometry, anonymous current slots.

Finite motion catalog is declared before scene sampling. Full shapes/physical
identities are generation/evaluation only. Inference gets partial history.
"""
import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from visual_reinspection.renderer import look_at
from inspection_decisions.completion_compatible_anchor_v1 import bitset

PALETTE=np.array([[205,65,55],[55,190,80],[65,90,215],[205,175,45]],float)


def transform(x,T):return x@T[:3,:3].T+T[:3,3]
def inverse(x,T):return (x-T[:3,3])@T[:3,:3]


def catalog():
    translations=[[0,0,0],[.16,0,0],[-.16,0,0],[0,.16,0],[0,-.16,0],[0,0,.16],[0,0,-.16],[.08,.08,.08],[-.08,.08,-.08]]
    angles=[[0,0,0],[0,12,0],[0,-12,0],[12,0,0],[-12,0,0],[0,0,12],[0,0,-12],[8,8,8],[-8,8,-8]]
    out=[]
    for t,a in zip(translations,angles):
        T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',a,degrees=True).as_matrix();T[:3,3]=t;out.append(T)
    return np.array(out)


def ray_box(origin,direction,lo,hi):
    safe=np.where(np.abs(direction)>1e-12,direction,np.where(direction>=0,1e-12,-1e-12))
    a=(lo-origin)/safe;b=(hi-origin)/safe
    near=np.minimum(a,b).max(-1);far=np.maximum(a,b).min(-1)
    return np.where((far>=np.maximum(near,0))&(near>0),near,np.inf)


def shapes(n):
    result=[]
    for i in range(n):
        half=np.array([.24+.08*(i%3),.24+.08*((i//3)%2),.24])
        body=np.array([-half,half])
        x=.16 if i%2 else -.16;y=.16 if (i//2)%2 else -.16
        tab=np.array([[x-.08,y-.08,-.40],[x+.08,y+.08,-.24]])
        x2=.16 if (i//4)%2 else -.16;y2=.16 if (i//8)%2 else -.16
        back=np.array([[x2-.08,y2-.08,.24],[x2+.08,y2+.08,.40]])
        result.append([body,tab,back])
    return result


def grid(cfg):
    h=cfg['voxel_m'];n=int(round(2*cfg['extent_m']/h));q=(np.arange(n)+.5)*h-cfg['extent_m']
    centers=np.stack(np.meshgrid(q,q,q,indexing='ij'),-1).reshape(-1,3)
    corners=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]])*h/2
    return centers,corners,n


def render(geometry,poses,groups,C,cfg,rng):
    h,w=cfg['height'],cfg['width'];f=cfg['focal_px'];yy,xx=np.mgrid[:h,:w]
    rays=np.stack([(xx-(w-1)/2)/f,(yy-(h-1)/2)/f,np.ones_like(xx)],-1).reshape(-1,3)
    wr=rays@C[:3,:3].T
    depth=np.full(len(wr),30.);owner=np.full(len(wr),-1,int)
    for i,(boxes,T) in enumerate(zip(geometry,poses)):
        o=inverse(C[:3,3][None],T)[0];r=wr@T[:3,:3]
        for lo,hi in boxes:
            d=ray_box(o,r,lo,hi);take=d<depth;depth[take]=d[take];owner[take]=i
    # Two static bars cause actual ray occlusion; no mask is given to inference.
    for lo,hi in [(np.array([-.13,-3,-1.1]),np.array([.13,3,-.9])),
                  (np.array([-5,-.08,.9]),np.array([5,.08,1.1]))]:
        d=ray_box(C[:3,3],wr,lo,hi);take=d<depth;depth[take]=d[take];owner[take]=-2
    rgb=np.full((len(wr),3),80.,float)
    for i,g in enumerate(groups):rgb[owner==i]=PALETTE[g]
    depth+=rng.uniform(-cfg['depth_noise_bound_m'],cfg['depth_noise_bound_m'],len(depth))
    rgb=np.clip(rgb+rng.uniform(-1,1,rgb.shape),0,255).astype(np.uint8)
    return {'depth':depth.reshape(h,w),'rgb':rgb.reshape(h,w,3),'C':C},owner.reshape(h,w)


def observed_slots(frame,slots,groups,cfg):
    d=frame['depth'];rgb=frame['rgb'].astype(float);f=cfg['focal_px'];h,w=d.shape
    color=np.linalg.norm(rgb[:,:,None,:]-PALETTE[None,None,:,:],axis=-1)
    group=color.argmin(-1);valid=color.min(-1)<8
    yy,xx=np.nonzero(valid);z=d[yy,xx]
    p=np.c_[(xx-(w-1)/2)*z/f,(yy-(h-1)/2)*z/f,z]
    world=transform(p,frame['C']);distance=np.linalg.norm(world[:,None,:]-slots[None,:,:],axis=-1)
    distance[:, :] += np.where(group[yy,xx,None]==groups[None,:],0,100)
    assignment=distance.argmin(-1)
    result=[]
    for j in range(len(slots)):
        use=np.flatnonzero((assignment==j)&(distance[np.arange(len(assignment)),assignment]<.9))
        points=world[use]
        if len(use)>cfg['rays_per_slot_frame']:
            chosen=use[np.linspace(0,len(use)-1,cfg['rays_per_slot_frame'],dtype=int)]
        else:chosen=use
        result.append({'points':points,'pixels':np.c_[yy[chosen],xx[chosen]],'depth':z[chosen]})
    return result


def free_cells(frame,T,centers,corners,cfg):
    world=transform(centers,T);C=frame['C'];cam=inverse(world,C)
    rc=C[:3,:3].T@T[:3,:3]
    corner_cam=cam[:,None,:]+corners[None,:,:]@rc.T
    f=cfg['focal_px'];h,w=frame['depth'].shape
    uv=corner_cam[:,:,:2]/corner_cam[:,:,2:]*f+np.array([(w-1)/2,(h-1)/2])
    center_uv=cam[:,:2]/cam[:,2:]*f+np.array([(w-1)/2,(h-1)/2]);ij=np.rint(center_uv).astype(int)
    radius=cfg['free_window_radius_px']
    inside=(corner_cam[:,:,2].min(1)>0)&(uv[:,:,0].min(1)>=0)&(uv[:,:,0].max(1)<=w-1)&(uv[:,:,1].min(1)>=0)&(uv[:,:,1].max(1)<=h-1)
    inside&=(np.abs(uv-ij[:,None,:]).max((1,2))<=radius)
    u=np.clip(ij[:,0],0,w-1);v=np.clip(ij[:,1],0,h-1)
    minimum=frame['min_depth'][v,u]
    # A measured central ray must actually intersect the voxel. Sound for the
    # declared full-cell occupancy model, not arbitrary subpixel thin surfaces.
    dc=np.c_[(u-(w-1)/2)/f,(v-(h-1)/2)/f,np.ones(len(u))]
    local_direction=dc@C[:3,:3].T@T[:3,:3]
    local_origin=inverse(C[:3,3][None],T)[0]
    entry=ray_box(local_origin,local_direction,centers-cfg['voxel_m']/2,centers+cfg['voxel_m']/2)
    free=inside&np.isfinite(entry)&(corner_cam[:,:,2].max(1)<minimum-cfg['depth_tolerance_m'])&(minimum>0)
    return bitset(np.flatnonzero(free))


def band_cells(frame,T,pixel,depth,cfg,n):
    f=cfg['focal_px'];h,w=frame['depth'].shape;y,x=pixel
    r=np.array([(x-(w-1)/2)/f,(y-(h-1)/2)/f,1.])@frame['C'][:3,:3].T
    o=inverse(frame['C'][:3,3][None],T)[0];r=r@T[:3,:3]
    eps=cfg['depth_tolerance_m'];a=o+r*(depth-eps);b=o+r*(depth+eps)
    low=np.maximum(0,np.floor((np.minimum(a,b)+cfg['extent_m'])/cfg['voxel_m']).astype(int))
    high=np.minimum(n-1,np.floor((np.maximum(a,b)+cfg['extent_m'])/cfg['voxel_m']).astype(int))
    if np.any(low>high):return 0
    indices=np.stack(np.meshgrid(*(np.arange(l,u+1) for l,u in zip(low,high)),indexing='ij'),-1).reshape(-1,3)
    lo=indices*cfg['voxel_m']-cfg['extent_m'];hi=lo+cfg['voxel_m'];delta=b-a
    safe=np.where(np.abs(delta)>1e-15,delta,1e-15)
    t0=(lo-a)/safe;t1=(hi-a)/safe
    enter=np.maximum(np.minimum(t0,t1).max(1),0);leave=np.minimum(np.maximum(t0,t1).min(1),1)
    hit=leave>=enter-1e-10
    return bitset((indices[hit,0]*n+indices[hit,1])*n+indices[hit,2])


def prepare(seed,history_views,cfg):
    rng=np.random.default_rng(seed);n=cfg['objects'];groups=np.arange(n)%4
    slots=np.array([[(i%6-2.5)*1.65,(i//6-1.5)*1.5,.08*((i%3)-1)] for i in range(n)])
    # Layout jitter is catalog metadata, known to every method.
    slots+=rng.uniform(-.04,.04,slots.shape)
    geometry=shapes(n);cat=catalog();T0=np.tile(np.eye(4),(n,1,1));T0[:,:3,3]=slots
    permutations=[];indices=[];true_poses=[]
    for t in [0,1]:
        owner=np.arange(n)
        for g in range(4):
            members=np.flatnonzero(groups==g)
            swap=rng.choice(members,2,replace=False);owner[swap]=owner[swap[::-1]]
        code=rng.integers(0,len(cat),n);code[rng.random(n)<.35]=0
        tt=np.array([T0[owner[i]]@cat[code[i]] for i in range(n)])
        permutations.append(owner);indices.append(code);true_poses.append(tt)
    positions=[[[0,0,-11],[-7,3,-9],[7,-3,-9]],[[1,2,-11],[9,1,-7]],[[0,1,11],[-9,-1,7]]]
    frames=[];anonymous=[]
    for t,pp in enumerate(positions):
        ff=[];aa=[]
        for view,pos in enumerate(pp[:history_views] if t==0 else pp):
            C=look_at(np.array(pos,float),np.zeros(3));frame,_=render(geometry,T0 if t==0 else true_poses[t-1],groups,C,cfg,np.random.default_rng(seed+10000*(t+1)+view))
            frame['min_depth']=minimum_filter(frame['depth'],size=2*cfg['free_window_radius_px']+1,mode='constant',cval=0.)
            ff.append(frame);aa.append(observed_slots(frame,slots,groups,cfg))
        frames.append(ff);anonymous.append(aa)
    centers,corners,ngrid=grid(cfg);history=[];all_branches=[];anchors=[];truth_shape=[];geometry_cache={}
    for i in range(n):
        cloud=np.concatenate([inverse(a[i]['points'],T0[i]) for a in anonymous[0]])
        if not len(cloud):raise ValueError(('missing historical anchor',seed,history_views,i))
        anchor=cloud[np.argmin(np.linalg.norm(cloud-cloud.mean(0),axis=1))];anchors.append(anchor)
        histfree=0;histbands=[]
        for frame,a in zip(frames[0],anonymous[0]):
            histfree|=free_cells(frame,T0[i],centers,corners,cfg)
            histbands.extend(band_cells(frame,T0[i],p,d,cfg,ngrid) for p,d in zip(a[i]['pixels'],a[i]['depth']))
        history.append({'free':histfree,'bands':histbands})
        tree=cKDTree(cloud);per_asset=[]
        for t in [0,1]:
            branches=[]
            for slot in np.flatnonzero(groups==groups[i]):
                points=np.concatenate([a[slot]['points'] for a in anonymous[t+1]])
                for code,increment in enumerate(cat):
                    T=T0[slot]@increment;key=(t,int(slot),code)
                    if key not in geometry_cache:
                        free=0;bands=[]
                        for frame,a in zip(frames[t+1],anonymous[t+1]):
                            free|=free_cells(frame,T,centers,corners,cfg)
                            bands.extend(band_cells(frame,T,p,d,cfg,ngrid) for p,d in zip(a[slot]['pixels'],a[slot]['depth']))
                        geometry_cache[key]=(free,bands)
                    free,bands=geometry_cache[key]
                    cost=float(np.mean(np.minimum(tree.query(inverse(points,T),workers=1)[0]/.08,1)**2)) if len(points) else 1.
                    branches.append({'free':free,'bands':bands,'slot':int(slot),'code':code,'T':T,'cost':cost})
            per_asset.append(branches)
        all_branches.append(per_asset)
        occupied=np.zeros(len(centers),bool)
        for lo,hi in geometry[i]:occupied|=((centers>=lo-1e-8)&(centers<=hi+1e-8)).all(1)
        truth_shape.append(bitset(np.flatnonzero(occupied)))
        if i%6==0:print('PREPARE',seed,history_views,i,'cached',len(geometry_cache),flush=True)
    truth={'slot1':np.array(permutations[0]),'slot2':np.array(permutations[1]),'code1':indices[0],'code2':indices[1],
           'poses':np.array(true_poses),'anchor_displacement':np.array([transform(a[None],true_poses[1][i])[0]-transform(a[None],true_poses[0][i])[0] for i,a in enumerate(anchors)]),
           'occupancy':truth_shape}
    return history,all_branches,np.array(anchors),frames,truth
