"""Actual-observation projective geometry; known patches only, translation only.

This is a two-stage measurement-backend experiment, not full joint SE(3)/identity
inference or calibrated probabilistic visibility estimation.
"""
import copy
import numpy as np
from scipy.ndimage import label,distance_transform_edt
from comparative_evidence.scene import PALETTE
from comparative_evidence.estimation_v1 import world_components,reference_map
from inspection_decisions.correspondence_joint_v1 import prepare as old_prepare
from inspection_decisions.correspondence_motion_objective_v1 import solve as motion_solve


def capture(history,current,sensor):
    def encode(o):
        return dict(token=o['token'],epoch=o['epoch'],view_id=o['view_id'],nuisance_group=o['nuisance_group'],
            nominal_T=o['nominal_T'].tolist(),rgb=o['rgb'].tolist(),depth=o['depth'].tolist(),
            components=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()} for c in o['components']])
    return dict(sensor=copy.deepcopy(sensor),history=[encode(o) for o in history],current=[encode(o) for o in current])


def decode(o):
    r=copy.deepcopy(o)
    for k,dtype in [('nominal_T',float),('rgb',np.uint8),('depth',np.float32)]:r[k]=np.array(r[k],dtype=dtype)
    for c in r['components']:
        for k,dtype in [('points_camera',float),('normals_camera',float),('normal_valid',bool),('pixels',int)]:c[k]=np.array(c[k],dtype=dtype)
    return r


def component_mask(o,c):
    rgb=o['rgb'].astype(float);dist=np.linalg.norm(rgb-PALETTE[c['group']],axis=2)
    lab,_=label((dist<30)&(o['depth']>0));ids=np.unique(lab.ravel()[c['pixels']]);ids=ids[ids>0]
    return np.isin(lab,ids)


def triangles_from_mask(depth,T,mask,sensor,cfg):
    h,w=depth.shape;yy,xx=np.mgrid[:h,:w];z=depth
    pc=np.stack([(xx-(w-1)/2)*z/sensor['focal_px'],(yy-(h-1)/2)*z/sensor['focal_px'],z],axis=-1)
    world=pc@T[:3,:3].T+T[:3,3];pieces=[]
    for positions in [[(0,0),(0,1),(1,0)],[(1,1),(1,0),(0,1)]]:
        pp=np.stack([world[dy:dy+h-1,dx:dx+w-1] for dy,dx in positions],axis=2)
        zz=np.stack([z[dy:dy+h-1,dx:dx+w-1] for dy,dx in positions],axis=2)
        keep=np.logical_and.reduce([mask[dy:dy+h-1,dx:dx+w-1] for dy,dx in positions])
        keep&=(np.max(zz,axis=2)-np.min(zz,axis=2)<cfg['mesh_depth_jump_m'])
        pieces.append(pp[keep])
    return np.concatenate(pieces) if pieces else np.empty((0,3,3))


def rasterize(triangles,T,shift,sensor):
    h,w=sensor['image_height'],sensor['image_width'];image=np.full(h*w,np.inf)
    triangles=np.asarray(triangles).reshape(-1,3,3)
    if not len(triangles):return image.reshape(h,w)
    world=triangles+shift;camera=(world-T[:3,3])@T[:3,:3]
    normal=np.cross(world[:,1]-world[:,0],world[:,2]-world[:,0])
    keep=(camera[:,:,2].min(axis=1)>.5)&(np.einsum('ij,ij->i',normal,world.mean(axis=1)-T[:3,3])>0)
    camera=camera[keep]
    if not len(camera):return image.reshape(h,w)
    uv=camera[:,:,:2]/camera[:,:,2,None]*sensor['focal_px']+[(w-1)/2,(h-1)/2]
    low=np.maximum(np.floor(uv.min(axis=1)).astype(int),[0,0]);high=np.minimum(np.ceil(uv.max(axis=1)).astype(int),[w-1,h-1])
    keep=np.all(high>=low,axis=1);uv=uv[keep];low=low[keep];high=high[keep];camera=camera[keep]
    if not len(uv):return image.reshape(h,w)
    size=(high-low).max(axis=0)+1;oy,ox=np.mgrid[:size[1],:size[0]]
    x=low[:,0,None]+ox.ravel();y=low[:,1,None]+oy.ravel()
    x0,x1,x2=[uv[:,i,0,None] for i in range(3)];y0,y1,y2=[uv[:,i,1,None] for i in range(3)]
    den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);safe=np.where(abs(den)>1e-12,den,1.)
    a=((y1-y2)*(x-x2)+(x2-x1)*(y-y2))/safe;b=((y2-y0)*(x-x2)+(x0-x2)*(y-y2))/safe;c=1-a-b
    valid=(abs(den)>1e-12)&(a>=-1e-8)&(b>=-1e-8)&(c>=-1e-8)&(x<=high[:,0,None])&(y<=high[:,1,None])
    invz=a/camera[:,0,2,None]+b/camera[:,1,2,None]+c/camera[:,2,2,None]
    depth=1/np.maximum(invz,1e-12);np.minimum.at(image,(y*w+x)[valid],depth[valid])
    return image.reshape(h,w)


def make_view(o,c):
    mask=component_mask(o,c)
    return dict(T=o['nominal_T'],depth=o['depth'],mask=mask,distance=distance_transform_edt(~mask),token=o['token'])


def frame_loss(triangles,view,d,sensor,cfg,censored=True,boundary=True):
    predicted=rasterize(triangles,view['T'],d,sensor);known=np.isfinite(predicted)
    overlap=known&view['mask'];foreground=known&~view['mask']&(view['depth']+cfg['occlusion_margin_m']<predicted)
    excluded=foreground if censored else np.zeros_like(known)
    residual=np.zeros_like(predicted);residual[overlap]=np.log1p(((predicted[overlap]-view['depth'][overlap])/cfg['sigma'])**2)
    edge=known&~view['mask']&~excluded
    if boundary:residual[edge]=np.log1p((view['distance'][edge]/cfg['boundary_sigma_px'])**2)
    score=float(residual[known].sum()/max(int(known.sum()),1))
    return score,dict(known_pixels=int(known.sum()),comparable_pixels=int(overlap.sum()),censored_pixels=int(excluded.sum()),boundary_pixels=int(edge.sum()))


def optimize(triangles,views,start,sensor,cfg,censored=True,boundary=True):
    start=np.array(start);cache={}
    def objective(d):
        key=tuple(np.round(d,9))
        if key not in cache:
            loss=0.;stats=[]
            for view in views:
                v,s=frame_loss(triangles,view,d,sensor,cfg,censored,boundary);loss+=v;stats.append(s)
            comparable=sum(s['comparable_pixels'] for s in stats)
            if comparable<cfg['minimum_comparable_pixels']:loss+=20+cfg['minimum_comparable_pixels']-comparable
            loss/=max(len(views),1);loss+=cfg['initial_prior_weight']*np.sum((d-start)**2)/cfg['local_search_bound_m']**2
            cache[key]=(float(loss),stats)
        return cache[key]
    d=start.copy();initial=objective(d)[0];trace=[initial]
    for step in cfg['search_steps_m']:
        for _ in range(cfg['sweeps_per_step']):
            for axis in range(3):
                candidates=[d.copy()]
                for sign in [-1,1]:
                    p=d.copy();p[axis]+=sign*step
                    if np.max(abs(p-start))<=cfg['local_search_bound_m']+1e-9:candidates.append(p)
                d=min(candidates,key=lambda p:objective(p)[0]);trace.append(objective(d)[0])
    loss,stats=objective(d);assert max(np.diff(trace),default=0)<1e-9
    return dict(d=d.tolist(),loss=loss,initial_loss=initial,trace=trace,stats=stats,evaluations=len(cache))


def prepare(raw,cfg):
    history=[decode(o) for o in raw['history']];current=[decode(o) for o in raw['current']];sensor=raw['sensor']
    legacy=old_prepare(history,current);initial=motion_solve(legacy,'shared_motion',cfg);refs=reference_map(history);meshes=[[] for r in refs]
    # Historical tracking uses the same observed nearest-centre rule as the frozen reference map.
    for frame_index,o in enumerate(history):
        for i,c in enumerate(world_components(o)):
            if frame_index==0:j=i
            else:
                choices=[(np.linalg.norm(c['centre']-r['centre']),j) for j,r in enumerate(refs) if c['group']==r['group']]
                if not choices or min(choices)[0]>1.25:continue
                j=min(choices)[1]
            triangles=triangles_from_mask(o['depth'],o['nominal_T'],component_mask(o,c),sensor,cfg)
            if len(triangles):meshes[j].append(triangles)
    meshes=[np.concatenate(p) if p else np.empty((0,3,3)) for p in meshes]
    views=[[make_view(o,c) for c in world_components(o)] for o in current]
    observations=[[] for r in refs]
    for f,pairs in enumerate(initial['best']['matches']):
        for i,j in pairs:observations[j].append((f,i))
    names=['joint_censored','independent_censored','joint_uncensored','joint_depth_only'];tables={k:copy.deepcopy(legacy) for k in names};diagnostics=[]
    for k in names:
        for f in tables[k]['frames']:f['valid']=np.zeros_like(np.array(f['valid'],bool)).tolist()
    for j,obs in enumerate(observations):
        if not obs or not len(meshes[j]):continue
        start=np.median([legacy['frames'][f]['D'][i][j] for f,i in obs],axis=0)
        use_views=[views[f][i] for f,i in obs]
        fits={}
        for method in names:
            if method=='independent_censored':fits[method]=[optimize(meshes[j],[v],start,sensor,cfg) for v in use_views]
            else:
                fit=optimize(meshes[j],use_views,start,sensor,cfg,method!='joint_uncensored',method!='joint_depth_only')
                fits[method]=[fit]*len(obs)
            for (f,i),fit in zip(obs,fits[method]):
                table=tables[method]['frames'][f];table['D'][i][j]=fit['d']
                table['valid'][i][j]=sum(s['comparable_pixels'] for s in fit['stats'])>=cfg['minimum_comparable_pixels']
        diagnostics.append(dict(reference=j,observations=obs,start=start.tolist(),triangles=len(meshes[j]),fits=fits))
    # Convert no array or image truth into the final estimator tables. S retains the common legacy shape score.
    return dict(legacy=legacy,initial=initial,tables=tables,diagnostics=diagnostics,
        mesh_triangle_counts=[len(m) for m in meshes],limitations=['initial identity fixed by common legacy stage','local translation search, not full SE3 or joint A/T/delta objective','energies and covariance not calibrated'])
