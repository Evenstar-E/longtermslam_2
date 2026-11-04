"""DEV prototype: measured reference face/edge dictionary and shared-pixel assignment.

Offsets are fitted once from H+C1 at the given poses, not joint SE3 estimates.
Finite depth-grid minimization is exact on the declared grid, not continuous MAP.
No CAD, truth labels, or true face-count input. Dictionary induction is a prior.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions


def arrays(x, rows):
    R=Rotation.from_rotvec(x[:3]).as_matrix()
    return ((np.array([r['origin'] for r in rows])-x[3:])@R,
            np.array([r['ray_inside'] for r in rows])@R,
            np.array([r['ray_outside'] for r in rows])@R)


def pixel_groups(rows):
    tokens=[(r['epoch'],r['view'],*r['inside_pixel']) for r in rows]
    lookup={}; groups=[]
    for token in tokens:
        if token not in lookup:lookup[token]=len(lookup)
        groups.append(lookup[token])
    return np.array(groups,int),len(lookup)


def induce_graph(references,cfg):
    """Density modes per axis, then only locally witnessed incident face pairs.

    Multiple parallel planes and disjoint collinear segments are retained.
    Min witnesses / bandwidth / span-gap are explicit dictionary priors.
    """
    points=[];regions=[];seen=set()
    for x,rows in references:
        if not rows:continue
        o,ri,_=arrays(x,rows);P=o+ri*np.array([r['depth'] for r in rows])[:,None]
        rr,_=all_regions(x,rows,cfg)
        for j,r in enumerate(rows):
            token=(r['epoch'],r['view'],*r['inside_pixel'])
            if token not in seen:seen.add(token);points.append(P[j])
        for r in rr:
            copy=dict(r);copy['token']=(rows[r['ray_id']]['epoch'],rows[r['ray_id']]['view'],*rows[r['ray_id']]['inside_pixel']);copy['point']=P[r['ray_id']];regions.append(copy)
    if not points:return {'faces':[],'edges':[],'unique_reference_pixels':0}
    P=np.array(points);faces=[];radius=cfg['face_mode_radius_m'];minimum=cfg['minimum_face_pixels']
    for a in range(3):
        values=P[:,a];remaining=np.ones(len(P),bool)
        while remaining.sum()>=minimum:
            v=np.sort(values[remaining]);left=np.searchsorted(v,v-radius);right=np.searchsorted(v,v+radius,side='right');j=int(np.argmax(right-left));gain=int(right[j]-left[j])
            if gain<minimum:break
            d=float(np.median(v[left[j]:right[j]]));mask=remaining&(np.abs(values-d)<=radius)
            if mask.sum()<minimum:remaining[np.argmin(np.where(remaining,abs(values-d),np.inf))]=False;continue
            faces.append({'axis':a,'offset':d,'support_pixels':int(mask.sum())});remaining[mask]=False
    edges=[];tol=cfg['graph_witness_tolerance_m'];by_key={}
    for key in [(0,1),(0,2),(1,2)]:
        rr=[r for r in regions if r['key']==key]
        if rr:by_key[key]=(rr,np.array([r['vertices'] for r in rr]))
    for f,F in enumerate(faces):
        for g in range(f+1,len(faces)):
            G=faces[g];a=F['axis'];k=G['axis']
            if a==k:continue
            key=tuple(sorted((a,k)));support=np.array([F['offset'],G['offset']]) if a<k else np.array([G['offset'],F['offset']]);witness={}
            if key not in by_key:continue
            rr,V=by_key[key];E=np.roll(V,-1,axis=1)-V;delta=support-V;cross=E[:,:,0]*delta[:,:,1]-E[:,:,1]*delta[:,:,0]
            area=abs(np.sum(V[:,:,0]*np.roll(V[:,:,1],-1,axis=1)-V[:,:,1]*np.roll(V[:,:,0],-1,axis=1),axis=1));inside=(area>1e-18)&(np.all(cross>=-1e-12,axis=1)|np.all(cross<=1e-12,axis=1))
            den=(E*E).sum(2);t=np.clip(np.divide((delta*E).sum(2),den,out=np.zeros_like(den),where=den>1e-28),0,1);distance=np.where(inside,0.,np.linalg.norm(delta-t[:,:,None]*E,axis=2).min(1))
            for j in np.flatnonzero(distance<=tol):
                r=rr[j];witness[r['token']]=float(r['point'][3-a-k])
            if len(witness)<cfg['minimum_edge_pixels']:continue
            values=np.sort(list(witness.values()));splits=np.flatnonzero(np.diff(values)>cfg['edge_span_gap_m'])+1
            for part in np.split(values,splits):
                if len(part)<cfg['minimum_edge_pixels']:continue
                edges.append({'faces':[f,g],'span_axis':3-a-k,'span':[float(part.min()),float(part.max())],'support_pixels':len(part)})
    return {'faces':faces,'edges':edges,'unique_reference_pixels':len(P),'dictionary_is_measured_prior_not_complete_shape':True}


def score_graph(x,rows,graph,cfg):
    """Minimize assignments and one depth-grid variable per actual source pixel.

    Each ray includes its source-plane residual, so both objectives have equal
    ray weighting. Shared version requires the SAME source face and depth for
    every neighbor of a source pixel; independent is its explicit relaxation.
    Shared graph offsets and observed local spans are used in both controls.
    Missing supports have per-ray capped cost, NOT inferred instance births.
    """
    if not rows:return {'independent':0.,'shared':0.,'shared_no_domain':0.,'pixels':0,'contradictory_pixels':0,'unmatched_rays':0}
    o,ri,ro=arrays(x,rows);ids,n=pixel_groups(rows);N=len(rows);count=np.bincount(ids,minlength=n);depth=np.array([r['depth'] for r in rows]);outside=np.array([r['outside_depth'] for r in rows]);eps=cfg['depth_noise_bound_m'];cap=cfg['support_match_cap_m'];cap2=cap*cap
    first=np.full(n,N,int);np.minimum.at(first,ids,np.arange(N))
    assert np.allclose(depth,depth[first[ids]],rtol=0,atol=1e-12) and np.allclose(o,o[first[ids]],rtol=0,atol=1e-12) and np.allclose(ri,ri[first[ids]],rtol=0,atol=1e-12),'same source pixel must have one measurement/camera/ray'
    grid=np.linspace(-eps,eps,cfg['shared_depth_grid_points']);base=depth[:,None]+grid[None];best_ind=np.full(N,cap2);best_shared=np.full(n,cap2)*count;best_no_domain=best_shared.copy();has=np.zeros(N,bool)
    for f,F in enumerate(graph['faces']):
        a=F['axis'];df=F['offset'];edges=[e for e in graph['edges'] if f in e['faces']]
        if not edges:continue
        A=ri[:,a];B=ro[:,a];stable=(abs(A)>cfg['denominator_tolerance'])&(abs(B)>cfg['denominator_tolerance']);za=np.divide(df-o[:,a],A,out=depth.copy(),where=stable);z=np.c_[base,np.clip(za,depth-eps,depth+eps)]
        lam=np.divide(df-o[:,a],B,out=np.full(N,-1.),where=stable);forward=np.divide(A,B,out=np.zeros(N),where=stable)>0;valid=stable&forward&(lam>0)&(lam<=outside-eps)
        source=(o[:,a,None]+A[:,None]*z-df)**2;best=np.full(z.shape,cap2);no_domain=best.copy()
        for edge in edges:
            g=edge['faces'][1] if edge['faces'][0]==f else edge['faces'][0];G=graph['faces'][g];k=G['axis'];dg=G['offset'];b_out=o[:,k]+ro[:,k]*lam;ratio=np.divide(A,B,out=np.zeros(N),where=stable);s=np.sign(ratio*ro[:,k]-ri[:,k]);edge_valid=valid&(s!=0)
            inside=o[:,k,None]+ri[:,k,None]*z
            residual=source+np.maximum(s[:,None]*(inside-dg),0)**2+np.maximum(s*(dg-b_out),0)[:,None]**2
            residual=np.where(edge_valid[:,None],np.minimum(residual,cap2),cap2);no_domain=np.minimum(no_domain,residual)
            h=edge['span_axis'];along=o[:,h,None]+ri[:,h,None]*z;lo,hi=edge['span'];margin=cfg['edge_span_margin_m'];span=np.maximum(lo-margin-along,0)**2+np.maximum(along-hi-margin,0)**2
            residual=np.minimum(residual+span,cap2);best=np.minimum(best,residual)
        has|=np.any(best<cap2,axis=1);best_ind=np.minimum(best_ind,best.min(1))
        # For each face, candidate depths coincide for all rays of a pixel:
        # measurement, camera, ray and projected face depth must be identical.
        total=np.zeros((n,z.shape[1]));total_no_domain=np.zeros_like(total);np.add.at(total,ids,best);np.add.at(total_no_domain,ids,no_domain)
        best_shared=np.minimum(best_shared,total.min(1));best_no_domain=np.minimum(best_no_domain,total_no_domain.min(1))
    independent_per_pixel=np.bincount(ids,weights=best_ind,minlength=n)
    assert np.all(best_shared>=independent_per_pixel-1e-12)
    return {'independent':float(best_ind.mean()/cap2),'shared':float(best_shared.sum()/N/cap2),'shared_no_domain':float(best_no_domain.sum()/N/cap2),'pixels':n,'contradictory_pixels':int(np.sum(best_shared>independent_per_pixel+1e-12)),'unmatched_rays':int((~has).sum()),'depth_grid_max_spacing_m':float(2*eps/(cfg['shared_depth_grid_points']-1))}
