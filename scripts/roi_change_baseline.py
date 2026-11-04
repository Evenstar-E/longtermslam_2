"""Ordinary point reprojection with explicit coverage and occlusion rejection; no truth inputs."""
import numpy as np
import cv2

def anchor_component(depth,classes,rectangle,label):
    x0,y0,x1,y1=map(int,rectangle);mask=np.zeros(depth.shape,np.uint8)
    mask[y0:y1+1,x0:x1+1]=((classes[y0:y1+1,x0:x1+1]==label)&np.isfinite(depth[y0:y1+1,x0:x1+1])&(depth[y0:y1+1,x0:x1+1]>0)).astype(np.uint8)
    n,ids,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    if n<2:return mask.astype(bool)
    largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    return ids==largest

def compare(anchor_depth,query_depth,anchor_mask,K,T_query_anchor,query_classes,label,cfg,method='occlusion_aware'):
    y,x=np.where(anchor_mask&np.isfinite(anchor_depth)&(anchor_depth>0));count=len(x)
    if count<cfg['minimum_projected_pixels']:return {'result':'insufficient','reason':'anchor_points','anchor_pixels':count}
    z=anchor_depth[y,x].astype(float);p=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])
    p=p@T_query_anchor[:3,:3].T+T_query_anchor[:3,3]
    positive=p[:,2]>cfg['minimum_depth_m'];p=p[positive]
    uv=p@K.T;uv=uv[:,:2]/uv[:,2,None]
    h,w=query_depth.shape;inside=(uv[:,0]>=-.5)&(uv[:,0]<w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<h-.5)
    xy=np.floor(uv[inside]+.5).astype(int);zz=p[inside,2]
    if len(xy):
        pixel=xy[:,1]*w+xy[:,0];order=np.lexsort((zz,pixel));pix=pixel[order];keep=np.r_[True,pix[1:]!=pix[:-1]];chosen=order[keep];xy=xy[chosen];zz=zz[chosen]
    n=len(zz);meta={'anchor_pixels':count,'projected_pixels':n,'projected_fraction':n/count}
    if n<cfg['minimum_projected_pixels'] or n/count<cfg['minimum_projected_fraction']:return {**meta,'result':'insufficient','reason':'projection_coverage'}
    d=query_depth[xy[:,1],xy[:,0]];valid=np.isfinite(d)&(d>0);tol=cfg['depth_tolerance_m']
    support=valid&(np.abs(d-zz)<=tol);far=valid&(d>zz+tol);near=valid&(d<zz-tol)
    same=support&(query_classes[xy[:,1],xy[:,0]]==label)
    meta.update(valid_pixels=int(valid.sum()),geometry_support=int(support.sum()),class_support=int(same.sum()),farther=int(far.sum()),nearer=int(near.sum()),invalid=int((~valid).sum()))
    if valid.sum()/n<cfg['minimum_valid_fraction']:return {**meta,'result':'insufficient','reason':'invalid_depth'}
    gate=cfg['decision_fraction']
    if method=='naive':
        if (far.sum()+near.sum())/n>=gate:return {**meta,'result':'unsupported','reason':'absolute_difference'}
    elif method=='occlusion_aware':
        if far.sum()/n>=gate:return {**meta,'result':'unsupported','reason':'farther_depth'}
        if near.sum()/n>=gate:return {**meta,'result':'insufficient','reason':'occluded'}
    else:raise ValueError(method)
    if same.sum()/n>=gate:return {**meta,'result':'consistent','reason':'class_and_depth_support'}
    if support.sum()/n>=gate:return {**meta,'result':'insufficient','reason':'class_mismatch'}
    return {**meta,'result':'insufficient','reason':'mixed_evidence'}

def interval_present(first,last,t):
    appeared=[a for a in first if a<=t];disappeared=[a for a in last if a<=t]
    return bool(appeared and (not disappeared or max(appeared)>max(disappeared)))

def query_map(nodes,anchor_points_world,semantic_label,anchor_stamp,query_stamp,cfg):
    centroid=anchor_points_world.mean(axis=0)
    candidates=[(float(np.linalg.norm(np.array(n['world_surface_centroid'])-centroid)),n) for n in nodes if n['semantic_label']==semantic_label and not n['dynamic_trajectory']]
    if not candidates:return {'result':'insufficient','reason':'no_class_candidate'}
    distance,node=min(candidates,key=lambda r:(r[0],int(r[1]['id_decimal'])))
    meta={'id_decimal':node['id_decimal'],'symbol':node['symbol'],'centroid_distance_m':distance}
    if distance>cfg['maximum_centroid_distance_m']:return {**meta,'result':'insufficient','reason':'association_distance'}
    if cfg['anchor_presence_required'] and not interval_present(node['first_observed_ns'],node['last_observed_ns'],anchor_stamp):return {**meta,'result':'insufficient','reason':'anchor_interval_inconsistent'}
    present=interval_present(node['first_observed_ns'],node['last_observed_ns'],query_stamp)
    return {**meta,'result':'consistent' if present else 'unsupported','reason':'map_presence_interval'}
