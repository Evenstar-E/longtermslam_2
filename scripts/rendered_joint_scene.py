"""CPU pinhole planar textured component scenes and RGB-D-only measurements."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from visual_reinspection.renderer import look_at

def render(centers,textures,T,K,cfg,rng,occluder=False):
    h,w=cfg['height'],cfg['width'];depth=np.full((h,w),12.,np.float32);rgb=np.full((h,w,3),55.,float);ids=np.full((h,w),-1,np.int16)
    half_side=cfg.get('panel_side_m',.4)/2
    planes=[(c,np.array([half_side,half_side]),textures[i],i) for i,c in enumerate(centers)]
    if occluder:planes.append((np.array([centers[:,0].mean()+.25,centers[:,1].mean(),2.]),np.array([1.05,2.6]),None,-2))
    for center,half,texture,i in planes:
        corners=np.array([[center[0]+sx*half[0],center[1]+sy*half[1],center[2]] for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
        p=(corners-T[:3,3])@T[:3,:3];uv=p@K.T;uv=uv[:,:2]/uv[:,2,None]
        x0,y0=np.maximum(np.floor(uv.min(axis=0)).astype(int)-1,0);x1,y1=np.minimum(np.ceil(uv.max(axis=0)).astype(int)+2,[w,h])
        if x1<=x0 or y1<=y0:continue
        y,x=np.mgrid[y0:y1,x0:x1];rays=np.stack([(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],np.ones_like(x)],-1)@T[:3,:3].T
        distance=(center[2]-T[2,3])/rays[...,2];world=T[:3,3]+distance[...,None]*rays
        inside=(np.abs(world[...,0]-center[0])<=half[0])&(np.abs(world[...,1]-center[1])<=half[1])&(distance>0)&(distance<depth[y0:y1,x0:x1])
        if texture is None:col=np.full((*inside.shape,3),95.)
        else:
            u=np.clip(((world[...,0]-center[0])/(2*half[0])+.5)*4,0,3).astype(int);v=np.clip((.5-(world[...,1]-center[1])/(2*half[1]))*2,0,1).astype(int)
            col=texture[v,u]
        depth[y0:y1,x0:x1][inside]=distance[inside];rgb[y0:y1,x0:x1][inside]=col[inside];ids[y0:y1,x0:x1][inside]=i
    illumination=rng.uniform(.85,1.12);rgb=np.clip(rgb*illumination+rng.normal(0,cfg.get('rgb_sigma',2.),rgb.shape),0,255).astype(np.uint8)
    depth[ids!=-1]+=rng.normal(0,cfg.get('depth_sigma_m',.015),int((ids!=-1).sum())).astype(np.float32)
    return dict(rgb=rgb,depth=depth,ids=ids,T=T)

def extract(rgb,depth,K,cfg):
    """No mesh/instance map/true pose input. Current detections are anonymous."""
    a=rgb.astype(float)/255;v=a.max(axis=2);s=(v-a.min(axis=2))/np.maximum(v,1e-9)
    mask=((s>cfg['mask_saturation'])&(v>cfg['mask_min_value'])).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8);detections=[]
    for k in range(1,n):
        x,y,w,h,area=stats[k]
        if area<cfg['minimum_component_pixels']:continue
        yy,xx=np.where(labels==k);z=depth[yy,xx];valid=np.isfinite(z)&(z>0);yy,xx,z=yy[valid],xx[valid],z[valid]
        if len(z)<cfg['minimum_component_pixels']:continue
        points=np.column_stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z])
        patch=rgb[y:y+h,x:x+w].astype(float).max(axis=2)/255
        feature=cv2.resize(patch,(4,2),interpolation=cv2.INTER_AREA).ravel()
        feature=feature/(feature.mean()+1e-9)*.5
        detections.append(dict(position=points.mean(axis=0),descriptor=feature,rectangle=[int(x),int(y),int(w),int(h)],pixels=np.column_stack([yy,xx])))
    return detections

def label_detection(detection,ids):
    """Evaluator / historical annotation authoring only."""
    y,x=detection['pixels'].T;values,counts=np.unique(ids[y,x],return_counts=True);idx=np.argmax(counts)
    return int(values[idx]) if counts[idx]/len(y)>=.8 and values[idx]>=0 else -1

def make_scene(cfg,algorithm_cfg,condition,seed):
    rng=np.random.default_rng(seed);n=cfg['render']['objects'];h=cfg['render']['history_views'];cam=cfg['camera'];render_cfg={**cam,**cfg['render']};K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    centers=np.array([[1.15*(i%6)+.25*(i//6%2),1.1*(i//6),rng.uniform(-.08,.08)] for i in range(n)])
    centers[:,:2]+=rng.normal(0,.12,(n,2));angle=rng.uniform(-.18,.18);centers[:,:2]=centers[:,:2]@np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).T
    # Tiled appearance becomes pixels, then an independent detector/descriptor observes them.
    groups=rng.uniform(.45,.95,(4,2,4));intensities=np.clip(groups[np.arange(n)%4]+rng.normal(0,.015 if condition['repeated_appearance'] else .20,(n,2,4)),.25,1.)
    palette=np.array([[230,65,55],[60,220,75],[55,85,230],[220,180,45]])
    textures=intensities[...,None]*palette[np.arange(n)%4,None,None,:]
    target=centers.copy();changed=np.zeros(n,bool);selected=rng.choice(n,int(round(n*condition['change_fraction'])),replace=False);changed[selected]=True
    direction=rng.normal(size=(len(selected),3));direction/=np.linalg.norm(direction,axis=1)[:,None];target[selected]+=direction*rng.uniform(.45,.9,(len(selected),1))
    focus=np.r_[centers[:,:2].mean(axis=0),0.];frames=[];history_positions=[];history_descriptors=[];history_complete=True
    for view in range(h):
        eye=focus+np.r_[rng.uniform(-.6,.6,2),rng.uniform(7.7,8.5)];T=look_at(eye,focus+rng.uniform(-.15,.15,3));frame=render(centers,textures,T,K,render_cfg,rng)
        detections=extract(frame['rgb'],frame['depth'],K,cfg['render']);pos=np.full((n,3),np.nan);feat=np.full((n,8),np.nan)
        for d in detections:
            j=label_detection(d,frame['ids'])
            if j>=0:pos[j]=d['position']@T[:3,:3].T+T[:3,3];feat[j]=d['descriptor']
        history_complete&=bool(np.isfinite(pos).all());history_positions.append(pos);history_descriptors.append(feat);frames.append(frame)
    eye=focus+np.r_[rng.uniform(-.8,.8,2),rng.uniform(7.2,9.)];T=look_at(eye,focus+rng.uniform(-.3,.3,3));frame=render(target,textures,T,K,render_cfg,rng,condition['occluder']);frames.append(frame)
    ds=extract(frame['rgb'],frame['depth'],K,cfg['render']);current_ids=np.array([label_detection(d,frame['ids']) for d in ds]);actual_changed=np.array([changed[i] if i>=0 else False for i in current_ids])
    s=algorithm_cfg['simulation'];prefix='pose_'+condition['pose_error'];error=np.eye(4);error[:3,:3]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(s[prefix+'_rotation_sigma_deg']),3)).as_matrix();error[:3,3]=rng.normal(0,s[prefix+'_translation_sigma_m'],3)
    obs=dict(history_positions=np.array(history_positions),history_descriptors=np.array(history_descriptors),current_positions=np.array([d['position'] for d in ds]).reshape(-1,3),current_descriptors=np.array([d['descriptor'] for d in ds]).reshape(-1,8),initial_T=error@T)
    truth=dict(T=T,assignment=current_ids,changed=actual_changed)
    visible=[i for i in range(n) if (frame['ids']==i).sum()>=cfg['render']['minimum_component_pixels']]
    info=dict(history_complete=history_complete,detections=len(ds),visible_targets=len(visible),detected_unique_visible=len(set(current_ids.tolist())&set(visible)),ambiguous_detections=int((current_ids<0).sum()),rectangles=[d['rectangle'] for d in ds],physical_changed_ids=np.flatnonzero(changed).tolist(),visible_ids=visible)
    return obs,truth,frames,info,K
