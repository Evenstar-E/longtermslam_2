"""Algebraically identical vectorization and fixed-R gauge translation cache."""
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.sparse_ray_material_incidence_v1 import polygon_distances


def all_regions(x,rows,cfg):
    if not rows:return [],{}
    R=Rotation.from_rotvec(x[:3]).as_matrix();origin=(np.array([r['origin'] for r in rows])-x[3:])@R;ri=np.array([r['ray_inside'] for r in rows])@R;ro=np.array([r['ray_outside'] for r in rows])@R;z=np.array([r['depth'] for r in rows]);zo=np.array([r['outside_depth'] for r in rows]);eps=cfg['depth_noise_bound_m'];tol=cfg['denominator_tolerance'];out=[];status={}
    def add(reason,count):
        if count:status[reason]=status.get(reason,0)+int(count)
    for a in range(3):
        A=ri[:,a];B=ro[:,a];stable=(np.abs(A)>tol)&(np.abs(B)>tol);add('grazing or singular candidate source plane',2*(~stable).sum());ratio=np.divide(A,B,out=np.zeros(len(rows)),where=stable);forward=stable&(ratio>0);add('intersection behind camera',2*(stable&~forward).sum());high_return=np.divide(zo-eps,ratio,out=np.full(len(rows),-np.inf),where=forward);low=z-eps;high=np.minimum(z+eps,high_return);valid=forward&(high>=low)&(low>0);add('neighbor return censors candidate source face',2*(forward&~valid).sum())
        for k in range(3):
            if a==k:continue
            bi=ri[:,k];bout=ratio*ro[:,k];step=bout-bi;mask=valid&(np.abs(step)>tol);add('no extent in candidate exit direction',(valid&~mask).sum());indices=np.flatnonzero(mask);key=tuple(sorted((a,k)))
            if not len(indices):continue
            lower=np.where(step>0,bi,bout);upper=np.where(step>0,bout,bi);depth=np.stack([low,low,high,high],axis=1);slope=np.stack([lower,upper,upper,lower],axis=1);vertices=np.stack([origin[:,a,None]+A[:,None]*depth,origin[:,k,None]+slope*depth],axis=-1)
            if a>k:vertices=vertices[:,:,::-1]
            for j in indices:
                row=rows[j];out.append({'key':key,'source_axis':a,'exit_axis':k,'exit_sign':float(np.sign(step[j])),'vertices':vertices[j].copy(),'source_depth_interval':[float(low[j]),float(high[j])],'first_return_truncates_source_interval':bool(high_return[j]<z[j]+eps),'ray_id':row['ray_id'],'epoch':row['epoch'],'view':row['view']})
    out.sort(key=lambda r:(r['ray_id'],r['source_axis'],r['exit_axis']));return out,status


def translate_regions(base,epsilon):
    result=[]
    for obj in base:
        row=dict(obj);v=obj['vertices'].copy()
        if 2 in obj['key']:v[:,obj['key'].index(2)]-=epsilon
        row['vertices']=v;result.append(row)
    return result


def nearest_support_costs(current,reference,ray_count,cfg):
    """Compute exact unordered distances once, reuse for ordered ablation."""
    cap=cfg['support_match_cap_m'];best=np.full(ray_count,cap);ordered=np.full(ray_count,cap);available=np.zeros(ray_count,bool);total_pairs=0;ordered_pairs=0
    for key in set(tuple(r['key']) for r in current):
        C=[r for r in current if tuple(r['key'])==key];H=[r for r in reference if tuple(r['key'])==key];ids=np.array([r['ray_id'] for r in C]);available[ids]=True
        if not H:continue
        P=np.array([r['vertices'] for r in C]);Q=np.array([r['vertices'] for r in H]);lower=np.maximum(P.min(1)[:,None,:]-Q.max(1)[None,:,:],Q.min(1)[None,:,:]-P.max(1)[:,None,:]);bound=np.linalg.norm(np.maximum(lower,0),axis=-1);ii,jj=np.where(bound<cap);m=np.full(len(C),cap);mo=np.full(len(C),cap)
        if len(ii):
            distance=polygon_distances(P[ii],Q[jj]);np.minimum.at(m,ii,distance);same=np.array([r['source_axis'] for r in C])[ii]==np.array([r['source_axis'] for r in H])[jj];np.minimum.at(mo,ii[same],distance[same]);total_pairs+=len(ii);ordered_pairs+=int(same.sum())
        np.minimum.at(best,ids,m);np.minimum.at(ordered,ids,mo)
    def output(values,pairs):return {'mean_capped_squared_cost':float(np.mean((values/cap)**2)) if ray_count else 0.,'per_ray_distance_m':values,'hypothesis_available':available.copy(),'exact_region_pairs':pairs,'conditional_recurrence_prior':True,'free_birth_control_cost':0.}
    return output(best,total_pairs),output(ordered,ordered_pairs)
