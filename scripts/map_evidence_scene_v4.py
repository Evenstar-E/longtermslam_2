"""RGB-D-supported extrema with common per-frame calibration nuisances."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.partial_surface_scene import make_scene,render,detect
from scripts.joint_estimation import transform,assign
from scripts.fragment_adapter_v3 import merge_excess_fragments


def nominal_pose(T,bounds,rng):
    u=rng.uniform(-1,1,7);out=T.copy()
    out[:3,:3]=Rotation.from_rotvec(u[3:6]*bounds[1]).as_matrix()@T[:3,:3]
    out[:3,3]+=u[:3]*bounds[0]
    return out,u


def summaries(rgb,depth,T,pixels,rays,bounds):
    h,w=depth.shape;pix=np.asarray(pixels);grid=rays.reshape(h,w,3)*depth[:,:,None]
    p=transform(grid.reshape(-1,3)[pix],T);mask=np.zeros(h*w,bool);mask[pix]=True;mask=mask.reshape(h,w)
    y,x=np.unravel_index(pix,(h,w));inside=(x>0)&(x<w-1)&(y>0)&(y<h-1)
    valid=np.flatnonzero(inside)
    normal_points=[];normals=[]
    if len(valid):
        yy,xx=y[valid],x[valid];same=mask[yy,xx-1]&mask[yy,xx+1]&mask[yy-1,xx]&mask[yy+1,xx];valid=valid[same];yy,xx=y[valid],x[valid]
        a=grid[yy,xx+1]-grid[yy,xx-1];b=grid[yy+1,xx]-grid[yy-1,xx];normal=np.cross(a,b);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-9);normal=normal@T[:3,:3].T
        toward=T[:3,3]-p[valid];normal*=np.where(np.sum(normal*toward,axis=1)>0,1,-1)[:,None]
        normals=normal;normal_points=p[valid]
    endpoints=[]
    for axis in range(3):
        lo,hi=np.quantile(p[:,axis],[.01,.99]);extent=hi-lo
        for side,value in enumerate((lo,hi)):
            supported=False;source='censored'
            if len(normals):
                selected=(normals[:,axis]>(.94)) if side else (normals[:,axis]<(-.94))
                if selected.sum()>=10:
                    point=np.median(normal_points[selected],axis=0);value=point[axis];supported=True;source='observed_plane'
            if not supported:
                selected=np.abs(p[:,axis]-value)<max(.02,.04*extent)
                edge=np.flatnonzero(selected&inside);free=[]
                for k in edge:
                    yy,xx=y[k],x[k];near=[(yy,xx-1),(yy,xx+1),(yy-1,xx),(yy+1,xx)]
                    free.append(any(not mask[a,b] and depth[a,b]>depth[yy,xx]+.15 for a,b in near))
                if extent>.25 and len(free)>=5 and np.mean(free)>.1:supported=True;source='visible_boundary'
                point=p[np.argmin(np.abs(p[:,axis]-value))].copy();point[axis]=value
            v=point-T[:3,3];skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            # Nominal-minus-true first-order coordinate error; bounded nuisance
            # shared by all assets in this frame. Depth bias follows camera ray.
            camera=transform(point[None],np.linalg.inv(T))[0];direction=T[:3,:3]@(camera/max(camera[2],1e-9))
            jac=np.r_[np.eye(3)[axis]*bounds[0],-skew[axis]*bounds[1],direction[axis]*bounds[2]]
            endpoints.append(dict(axis=axis,side=side,value=float(value),point=point.tolist(),jacobian=jac.tolist(),tight=bool(supported),source=source))
    return endpoints,p


def generate(cfg,basecfg,condition,seed):
    rng=np.random.default_rng(seed+744);oldcond=dict(wide_view=cfg['query_wide_view'],changed_objects=condition['changed_objects'],occluder=cfg['query_occluder']);original,truth,visuals=make_scene(basecfg,oldcond,seed);rays=original['rays'];groups=np.arange(basecfg['objects'])%4
    hb=np.array([cfg['historical_translation_bound_m'],cfg['historical_rotation_bound_rad'],cfg['historical_depth_bias_bound_m']])*(1 if condition['map_noise'] else 0)
    qb=np.array([cfg['query_translation_bound_m'],cfg['query_rotation_bound_rad'],cfg['query_depth_bias_bound_m']])
    histories=[];clouds=[[] for _ in range(basecfg['objects'])];realized=[];images=[]
    indices=[0] if condition['history_views']==1 else [0,1,4]
    for frame,index in enumerate(indices):
        trueT=visuals['history'][index][2];rgb,depth,labels=render(truth['boxes'],trueT,groups,basecfg,rng)
        T,u=nominal_pose(trueT,hb,rng);depth=depth+u[6]*hb[2];realized.append(u.tolist());images.append(rgb)
        for d in detect(rgb,depth,basecfg):
            # Historical track IDs privileged as in all prior branches.
            owner=int(np.bincount(labels.ravel()[d['pixels']],minlength=basecfg['objects']).argmax())
            endpoints,p=summaries(rgb,depth,T,d['pixels'],rays,hb);clouds[owner].append(p)
            histories.append(dict(asset=owner,frame=frame,endpoints=endpoints))
    bounds=[]
    for j,parts in enumerate(clouds):
        if not parts:raise RuntimeError(f'No history for asset {j}; explicit missing-reference adapter required')
        p=np.concatenate(parts);bounds.append(np.quantile(p,[.01,.99],axis=0))
    obs=dict(original);obs['bounds']=np.array(bounds)
    # Query depth has independently declared current calibration uncertainty.
    T,u=nominal_pose(truth['T'],qb,rng);obs['initial_T']=T;obs['depth']=original['depth']+u[6]*qb[2]
    obs,merged=merge_excess_fragments(obs,basecfg['points_per_object'])
    assignment_truth=[]
    for members in merged:
        labels=truth['assignment'][members]
        if len(set(labels.tolist()))!=1:raise RuntimeError('Ambiguous cross-asset merge; do not silently exclude')
        assignment_truth.append(int(labels[0]))
    features=np.array([q['feature'] for q in obs['query']]);current_centres=np.array([transform(q['points'],T).mean(0) for q in obs['query']]);ref_centres=obs['bounds'].mean(1)
    costs=np.sum((current_centres[:,None]-ref_centres[None])**2,axis=2)+4*np.sum((features[:,None]-obs['reference_features'][None])**2,axis=2);A=assign(costs)
    current=[]
    for i,(q,j) in enumerate(zip(obs['query'],A)):
        end,_=summaries(obs['rgb'],obs['depth'],T,q['all_pixels'],rays,qb);current.append(dict(asset=int(j),frame=len(indices),query=i,endpoints=end))
    oldids=truth['assignment'].tolist();selected=[oldids.index(j) for j in assignment_truth]
    inp=dict(assets=basecfg['objects'],frames=len(indices)+1,history=histories,current=current,assignment=A.tolist(),reference_bounds=obs['bounds'].tolist(),condition=condition)
    gt=dict(assignment=assignment_truth,changed=truth['changed'][selected].tolist(),delta=truth['delta'][selected].tolist(),realized_history_nuisance=realized,realized_query_nuisance=u.tolist(),reference_boxes=truth['boxes'].tolist(),all_delta=(truth['current_boxes'].mean(1)-truth['boxes'].mean(1)).tolist())
    return inp,gt,obs['rgb'],images
