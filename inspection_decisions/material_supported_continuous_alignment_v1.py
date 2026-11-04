"""Observed signed dominant-face supports, profiled current-born plane offsets.

Explicit small-motion Manhattan hypothesis. Plane BA, projection and variable
elimination are standard. No continuous surface domain or novelty certificate.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

AXES=np.r_[np.eye(3),-np.eye(3)]


def skew(x):
    a,b,c=x;return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])


def right_jacobian(w):
    angle=np.linalg.norm(w);W=skew(w)
    if angle<1e-7:return np.eye(3)-W/2+W@W/6
    return np.eye(3)-(1-np.cos(angle))/angle**2*W+(angle-np.sin(angle))/angle**3*(W@W)


def pose_vector(T,center):
    return np.r_[Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3]-center]


def world_pose(x,center):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix();T[:3,3]=x[3:]+center;return T


def project_parameters(x,rotation_limit,translation_limit):
    result=np.asarray(x,float).copy()
    for start in range(0,len(x),6):
        for offset,radius in [(0,rotation_limit),(3,translation_limit)]:
            v=result[start+offset:start+offset+3];norm=np.linalg.norm(v)
            if norm>radius:result[start+offset:start+offset+3]=v*(radius/norm)
    return result


def projected_gn(x,fun,iterations,limits,cfg):
    x=project_parameters(x,*limits);r,J=fun(x);cost=float(r@r);trace=[]
    for iteration in range(iterations):
        delta=np.linalg.lstsq(J,-r,rcond=cfg['gn_rcond'])[0]
        if not np.isfinite(delta).all():break
        accepted=False
        for back in range(cfg['backtracking_steps']):
            trial=project_parameters(x+delta*(.5**back),*limits);rr,JJ=fun(trial);value=float(rr@rr)
            if value<cost-cfg['descent_tolerance']:
                trace.append({'iteration':iteration,'before':cost,'after':value,'backtracks':back,'parameters':trial.tolist()});x,r,J,cost=trial,rr,JJ,value;accepted=True;break
        if not accepted:break
    return x,{'initial_or_final_empty':len(r)==0,'final_cost':cost,'accepted_steps':trace,'rank':int(np.linalg.matrix_rank(J,tol=1e-7)) if J.size else 0}


def extract_planes(points,camera,center,view,cfg,seed):
    """RANSAC/TLS observed samples only; largest patch per signed axis/view."""
    p=np.asarray(points)-center;camera=np.asarray(camera)-center;remaining=np.arange(len(p));best={};all_records=[]
    o3d.utility.random.seed(int(seed))
    for attempt in range(cfg['planes_per_view_max']):
        if len(remaining)<cfg['plane_min_points']:break
        cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(p[remaining])
        _,inliers=cloud.segment_plane(cfg['plane_fit_threshold_m'],3,cfg['plane_ransac_iterations'])
        if len(inliers)<cfg['plane_min_points']:break
        selected=remaining[np.array(inliers)];xyz=p[selected];centroid=xyz.mean(0);_,_,V=np.linalg.svd(xyz-centroid,full_matrices=False);normal=V[-1]
        if normal@(camera-centroid)<0:normal=-normal
        k=int(np.argmax(AXES@normal));aligned=float(AXES[k]@normal)
        record={'axis':k,'normal':normal,'centroid':centroid,'count':len(selected),'view':view,'axis_cosine':aligned,'rms_m':float(np.sqrt(np.mean(((xyz-centroid)@normal)**2))),'measured_indices':selected}
        all_records.append(record)
        if aligned>=cfg['plane_axis_cosine_min'] and (k not in best or len(selected)>best[k]['count']):best[k]=record
        mask=np.ones(len(remaining),bool);mask[np.array(inliers)]=False;remaining=remaining[mask]
    return list(best.values()),{'raw_points':len(p),'extracted':len(all_records),'retained':len(best),'remaining':len(remaining)}


def historical_offsets(features):
    return {k:float(np.mean([AXES[k]@f['centroid'] for f in features if f['axis']==k])) for k in sorted({f['axis'] for f in features})}


def material_residual(parameters,features,reference,cfg,shared=True):
    rows=[];jac=[];offset_rows=[];offset_jac=[];keys=[]
    for t,ff in enumerate(features):
        x=parameters[6*t:6*t+6];R=Rotation.from_rotvec(x[:3]).as_matrix();Jr=right_jacobian(x[:3])
        for f in ff:
            k=f['axis'];n=AXES[k];rotated=R@n;normal_error=(rotated-f['normal'])/cfg['sigma_normal']
            normal_J=np.zeros((3,12));normal_J[:,6*t:6*t+3]=-R@skew(n)@Jr/cfg['sigma_normal']
            rows.extend(normal_error);jac.extend(normal_J)
            v=f['centroid']-x[3:];q=rotated@v;Jq=np.zeros(12);Jq[6*t:6*t+3]=np.cross(n,v@R)@Jr;Jq[6*t+3:6*t+6]=-rotated
            offset_rows.append(q-reference.get(k,0.));offset_jac.append(Jq);keys.append(('fixed',k) if k in reference else ('born',k) if shared else ('born',t,k))
    values=np.array(offset_rows);matrix=np.array(offset_jac).reshape(-1,12)
    for key in set(keys):
        if key[0]=='fixed':continue
        ix=[i for i,k in enumerate(keys) if k==key];values[ix]-=values[ix].mean();matrix[ix]-=matrix[ix].mean(0)
    rows.extend(values/cfg['sigma_offset_m']);jac.extend(matrix/cfg['sigma_offset_m'])
    return np.array(rows),np.array(jac).reshape(-1,12)


def material_statistics(T,features,center,reference,cfg):
    """Sufficient statistics for complete finite-pair scoring, not a posterior."""
    R=T[:3,:3];u=T[:3,3]-center;normal=0.;count=np.zeros(6);total=np.zeros(6);square=np.zeros(6);fixed=0.
    for f in features:
        k=f['axis'];n=AXES[k];q=n@R.T@(f['centroid']-u);normal+=float(np.sum((R@n-f['normal'])**2))/cfg['sigma_normal']**2
        if k in reference:fixed+=(q-reference[k])**2/cfg['sigma_offset_m']**2
        else:count[k]+=1;total[k]+=q;square[k]+=q*q
    centered=np.maximum(square-np.divide(total*total,count,out=np.zeros(6),where=count>0),0.)
    return {'unary':float(normal+fixed+centered.sum()/cfg['sigma_offset_m']**2),'count':count,'sum':total,'square':square}


def pair_costs(candidates,features,centers,reference,cfg):
    stats=[[material_statistics(b['T'],features[t][b['slot']],centers[b['slot']],reference,cfg) for b in bb] for t,bb in enumerate(candidates)]
    independent=np.array([s['unary'] for s in stats[0]])[:,None]+np.array([s['unary'] for s in stats[1]])[None,:]
    extra=np.zeros_like(independent)
    for k in range(6):
        n1=np.array([s['count'][k] for s in stats[0]])[:,None];n2=np.array([s['count'][k] for s in stats[1]])[None,:]
        a=np.array([s['sum'][k] for s in stats[0]])[:,None];b=np.array([s['sum'][k] for s in stats[1]])[None,:]
        m1=np.divide(a,n1,out=np.zeros_like(a),where=n1>0);m2=np.divide(b,n2,out=np.zeros_like(b),where=n2>0)
        weight=np.divide(n1*n2,n1+n2,out=np.zeros_like(extra),where=n1+n2>0)
        extra+=weight*(m1-m2)**2/cfg['sigma_offset_m']**2
    return independent,independent+extra


def icp_residual(x,source,target,normal,tree,maximum):
    R=Rotation.from_rotvec(x[:3]).as_matrix();query=source@R.T+x[3:];distance,index=tree.query(query,workers=1);use=distance<=maximum
    delta=query-target[index];value=np.sum(delta*normal[index],axis=1)/maximum
    active=use&(np.abs(value)<1);r=np.where(use,np.clip(value,-1,1),1.)/np.sqrt(max(1,len(source)))
    J=np.zeros((len(source),6));normals=normal[index[active]];J[active,:3]=np.cross(source[active],normals@R)@right_jacobian(x[:3])/maximum
    J[active,3:]=normals/maximum;J/=np.sqrt(max(1,len(source)))
    return r,J


def bounded_icp(T,source_pyramid,target_pyramid,center,limits,cfg):
    x=pose_vector(T,center);log=[]
    for level,(h,c,v,it) in enumerate(zip(source_pyramid,target_pyramid,cfg['voxel_scales_m'],cfg['bounded_icp_iterations'])):
        source=np.asarray(h.points);target=np.asarray(c.points);normal=np.asarray(c.normals)
        if min(len(source),len(target))<3:return None,{'failure':'fewer than three samples','levels':log}
        maximum=max(v*cfg['correspondence_distance_voxel_multiplier'],cfg['minimum_correspondence_distance_m']);tree=cKDTree(target)
        fun=lambda xx:icp_residual(xx,source,target,normal,tree,maximum)
        x,trace=projected_gn(x,fun,it,limits,cfg);log.append(trace)
    return world_pose(x,center),{'failure':None,'levels':log}
