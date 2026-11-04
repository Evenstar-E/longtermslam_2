"""Sparse conditional ray wedges, face-role-invariant correlated edge support.

Axis pair is a category; each polygon is an alternative instance observation.
Known-track pose/geometry optimisation is conditional, not identity recovery.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.material_supported_continuous_alignment_v1 import right_jacobian


class InvalidIncidence(ValueError):
    pass


def raw_pairs(frame,lab,slot,center,epoch,view,cfg):
    """Keep ALL adjacent non-own returns, including potentially censored ones."""
    yy,xx=np.where(lab==slot);p=np.c_[xx,yy];h,w=lab.shape;rows=[]
    def direction(pixel):
        return np.c_[(pixel[:,0]-(w-1)/2)/cfg['focal_px'],(pixel[:,1]-(h-1)/2)/cfg['focal_px'],np.ones(len(pixel))]@frame['C'][:3,:3].T
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        q=p+[dx,dy];inside=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h);pi=p[inside];po=q[inside];ok=lab[po[:,1],po[:,0]]!=slot;pi=pi[ok];po=po[ok];ri=direction(pi);ro=direction(po)
        for a,b,r,s in zip(pi,po,ri,ro):
            z=float(frame['depth'][a[1],a[0]]);zo=float(frame['depth'][b[1],b[0]])
            if not np.isfinite([z,zo]).all() or min(z,zo)<=cfg['depth_noise_bound_m']:continue
            rows.append({'epoch':epoch,'view':view,'inside_pixel':a,'outside_pixel':b,'origin':frame['C'][:3,3]-center,'ray_inside':r,'ray_outside':s,'depth':z,'outside_depth':zo,'ray_id':len(rows)})
    return rows


def coordinate_jets(x,obs,axis):
    """o_axis, r_in_axis, r_out_axis and exact absolute-rotvec derivatives."""
    R=Rotation.from_rotvec(x[:3]).as_matrix();Jr=right_jacobian(x[:3]);n=np.eye(3)[axis];rotated=R@n;values=[];jac=[]
    for v,translation in [(obs['origin']-x[3:],True),(obs['ray_inside'],False),(obs['ray_outside'],False)]:
        J=np.zeros(6);J[:3]=np.cross(n,v@R)@Jr
        if translation:J[3:]=-rotated
        values.append(float(rotated@v));jac.append(J)
    return np.array(values),np.array(jac)


def incidence_data(x,obs,a,k,cfg):
    p,Jp=coordinate_jets(x,obs,a);q,Jq=coordinate_jets(x,obs,k);ao,A,B=p;bo,bi,br=q;tol=cfg['denominator_tolerance']
    if abs(A)<=tol or abs(B)<=tol:raise InvalidIncidence('grazing or singular candidate source plane')
    ratio=A/B;Jratio=(B*Jp[1]-A*Jp[2])/B**2
    if ratio<=0:raise InvalidIncidence('intersection behind camera')
    low=obs['depth']-cfg['depth_noise_bound_m'];upper_source=obs['depth']+cfg['depth_noise_bound_m'];upper_return=(obs['outside_depth']-cfg['depth_noise_bound_m'])/ratio
    high=min(upper_source,upper_return)
    if high<low or low<=0:raise InvalidIncidence('neighbor return censors candidate source face')
    Jhigh=np.zeros(6) if upper_source<=upper_return else -(obs['outside_depth']-cfg['depth_noise_bound_m'])*Jratio/ratio**2
    bout=ratio*br;Jbout=Jratio*br+ratio*Jq[2];step=bout-bi
    if abs(step)<=tol:raise InvalidIncidence('no extent in candidate exit direction')
    return {'a':a,'k':k,'key':tuple(sorted((a,k))),'ao':ao,'A':A,'bo':bo,'bi':bi,'bout':bout,'sign':float(np.sign(step)),
            'Jao':Jp[0],'JA':Jp[1],'Jbo':Jq[0],'Jbi':Jq[1],'Jbout':Jbout,'low':low,'high':high,'Jhigh':Jhigh,'ratio':ratio,'first_return_truncates_source_interval':upper_return<upper_source}


def region(x,obs,a,k,cfg,jacobian=False):
    d=incidence_data(x,obs,a,k,cfg);vertices=[];jac=[]
    # Trapezoid in (source-plane offset, exit-plane offset), preserving z.
    co=[('bi','Jbi'),('bout','Jbout')] if d['sign']>0 else [('bout','Jbout'),('bi','Jbi')]
    for z,Jz,(field,jfield) in [(d['low'],np.zeros(6),co[0]),(d['low'],np.zeros(6),co[1]),(d['high'],d['Jhigh'],co[1]),(d['high'],d['Jhigh'],co[0])]:
        v=np.array([d['ao']+d['A']*z,d['bo']+d[field]*z]);J=np.array([d['Jao']+z*d['JA']+d['A']*Jz,d['Jbo']+z*d[jfield]+d[field]*Jz])
        if a>k:v=v[::-1];J=J[::-1]
        vertices.append(v);jac.append(J)
    result={'key':d['key'],'source_axis':a,'exit_axis':k,'exit_sign':d['sign'],'vertices':np.array(vertices),'source_depth_interval':[d['low'],d['high']],
            'first_return_truncates_source_interval':d['first_return_truncates_source_interval'],'ray_id':obs['ray_id'],'epoch':obs['epoch'],'view':obs['view']}
    if jacobian:result['jacobian']=np.array(jac)
    return result


def hypotheses(x,obs,cfg):
    result=[];status={}
    for a in range(3):
        for k in range(3):
            if a==k:continue
            try:result.append(region(x,obs,a,k,cfg))
            except InvalidIncidence as error:status[(a,k)]=str(error)
    return result,status


def profiled_factor(x,obs,a,k,support,cfg):
    """Residual/J for pose6 and shared edge coordinates2; profile latent depth."""
    d=incidence_data(x,obs,a,k,cfg);key=d['key'];ia=key.index(a);ik=key.index(k);da=support[ia];bk=support[ik];s=d['sign']
    alpha=np.array([d['A'],s*d['bi'],-s*d['bout']]);beta=np.array([d['ao']-da,s*(d['bo']-bk),s*(bk-d['bo'])])
    JA=np.zeros((3,8));JB=np.zeros((3,8));JA[:,:6]=[d['JA'],s*d['Jbi'],-s*d['Jbout']];JB[:,:6]=[d['Jao'],s*d['Jbo'],-s*d['Jbo']];JB[0,6+ia]=-1;JB[1,6+ik]=-s;JB[2,6+ik]=s
    low=d['low'];high=d['high'];Jhigh=np.r_[d['Jhigh'],0.,0.];candidates=[(low,np.zeros(8)),(high,Jhigh)]
    for b1 in [False,True]:
        for b2 in [False,True]:
            mask=np.array([True,b1,b2]);aa=alpha[mask];bb=beta[mask];den=aa@aa
            if den<1e-24:continue
            num=aa@bb;z=-num/den;dz=-((np.sum(bb[:,None]*JA[mask]+aa[:,None]*JB[mask],axis=0))*den-num*2*np.sum(aa[:,None]*JA[mask],axis=0))/den**2
            if z<low:z=low;dz=np.zeros(8)
            elif z>high:z=high;dz=Jhigh
            candidates.append((z,dz))
    values=[]
    for z,dz in candidates:
        r=alpha*z+beta;r[1:]=np.maximum(r[1:],0);values.append(float(r@r))
    index=int(np.argmin(values));z,dz=candidates[index];raw=alpha*z+beta;active=np.r_[True,raw[1:]>0];r=np.where(active,raw,0);J=(JA*z+JB+alpha[:,None]*dz)*active[:,None];scale=cfg['sigma_sparse_m']
    return r/scale,J/scale,{'source_depth_profiled':float(z),'source_depth_interval':[low,high],'conditional_axis_pair':key,'first_return_truncates_source_interval':d['first_return_truncates_source_interval']}


def joint_residual(parameters,records,cfg):
    keys={tuple(sorted((r['a'],r['k']))) for r in records}
    if len(keys)>1:raise ValueError('one conditional track must share one unordered coordinate pair')
    rows=[];jac=[]
    for record in records:
        t=record['epoch'];x=np.zeros(6) if t==0 else parameters[6*(t-1):6*t];r,J,_=profiled_factor(x,record['observation'],record['a'],record['k'],parameters[12:],cfg);M=np.zeros((3,14))
        if t>0:M[:,6*(t-1):6*t]=J[:,:6]
        M[:,12:]=J[:,6:];rows.extend(r);jac.extend(M)
    return np.array(rows),np.array(jac).reshape(-1,14)


def fit_joint_track(initial,records,limits,cfg):
    """Conditional chosen-track solver. Never deletes invalid trial factors."""
    def project(p):
        p=p.copy()
        for start in [0,6]:
            for off,radius in [(0,limits[0]),(3,limits[1])]:
                v=p[start+off:start+off+3];norm=np.linalg.norm(v)
                if norm>radius:p[start+off:start+off+3]=v*radius/norm
        return p
    x=project(np.asarray(initial,float));r,J=joint_residual(x,records,cfg);cost=float(r@r);trace=[];invalid=0
    for iteration in range(cfg['gn_iterations']):
        delta=np.linalg.lstsq(J,-r,rcond=1e-8)[0];accepted=False
        for back in range(cfg['backtracking_steps']):
            candidate=project(x+delta*(.5**back))
            try:rr,JJ=joint_residual(candidate,records,cfg)
            except InvalidIncidence:invalid+=1;continue
            value=float(rr@rr)
            if value<cost-1e-10:trace.append({'iteration':iteration,'before':cost,'after':value,'parameters':candidate.copy()});x,r,J,cost=candidate,rr,JJ,value;accepted=True;break
        if not accepted:break
    return x,{'final_cost':cost,'rank':int(np.linalg.matrix_rank(J,tol=1e-7)),'accepted_steps':trace,'invalid_trials_rejected':invalid,'physical_track_identity_assumed_not_inferred':True}


def polygon_distances(P,Q):
    """Exact Euclidean separation of corresponding convex quadrilateral pairs."""
    P=np.asarray(P);Q=np.asarray(Q)
    if len(P)==0:return np.empty(0)
    edgesP=np.roll(P,-1,axis=1)-P;edgesQ=np.roll(Q,-1,axis=1)-Q;edges=np.concatenate([edgesP,edgesQ],axis=1);normal=np.stack([-edges[:,:,1],edges[:,:,0]],axis=-1)
    # Tangents and coordinate axes cover degenerate segments/points as well.
    axes=np.concatenate([normal,edges,np.broadcast_to(np.eye(2),(len(P),2,2))],axis=1)
    p=P[:,None,:,:]*axes[:,:,None,:];q=Q[:,None,:,:]*axes[:,:,None,:];p=p.sum(-1);q=q.sum(-1);separate=((p.max(-1)<q.min(-1)-1e-12)|(q.max(-1)<p.min(-1)-1e-12)).any(-1)
    def vertex_to_edge(A,B):
        E=np.roll(B,-1,axis=1)-B;v=A[:,:,None,:]-B[:,None,:,:];den=(E*E).sum(-1);t=np.divide((v*E[:,None,:,:]).sum(-1),den[:,None,:],out=np.zeros(v.shape[:-1]),where=den[:,None,:]>1e-28);t=np.clip(t,0,1);return np.linalg.norm(v-t[:,:,:,None]*E[:,None,:,:],axis=-1).min((1,2))
    distance=np.minimum(vertex_to_edge(P,Q),vertex_to_edge(Q,P));return np.where(separate,distance,0.)


def nearest_support_cost(current,reference,ray_count,cfg,ordered=False):
    """Ordinary capped nearest-region recurrence CONTROL, not measured ID.

    A current ray can choose any source-axis/exit-axis hypothesis. Unknown
    birth is a constant cap here; allowing cost-free birth makes it all zero.
    """
    cap=cfg['support_match_cap_m'];best=np.full(ray_count,cap);available=np.zeros(ray_count,bool);pairs=0
    keys=set(tuple(r['key']) for r in current)
    for key in keys:
        for source in ([0,1,2] if ordered else [None]):
            C=[r for r in current if tuple(r['key'])==key and (source is None or r['source_axis']==source)];H=[r for r in reference if tuple(r['key'])==key and (source is None or r['source_axis']==source)]
            if not C:continue
            for c in C:available[c['ray_id']]=True
            if not H:continue
            P=np.array([r['vertices'] for r in C]);Q=np.array([r['vertices'] for r in H]);lo=np.maximum(P.min(1)[:,None,:]-Q.max(1)[None,:,:],Q.min(1)[None,:,:]-P.max(1)[:,None,:]);bound=np.linalg.norm(np.maximum(lo,0),axis=-1);ii,jj=np.where(bound<cap);minimum=np.full(len(C),cap)
            if len(ii):
                distances=polygon_distances(P[ii],Q[jj]);np.minimum.at(minimum,ii,distances);pairs+=len(ii)
            for c,distance in zip(C,minimum):best[c['ray_id']]=min(best[c['ray_id']],distance)
    return {'mean_capped_squared_cost':float(np.mean((best/cap)**2)) if ray_count else 0.,'per_ray_distance_m':best,'hypothesis_available':available,'exact_region_pairs':pairs,'conditional_recurrence_prior':True,'free_birth_control_cost':0.}
