"""Conditional material-domain predicates and ordinary real-image support gates.

No graph prior may overwrite a hard observation. No continuous-domain solver,
calibrated real sensor certificate, posterior or original algorithm is claimed.
"""
import numpy as np
from inspection_decisions.ray_conditioned_plane_polytope_v1 import intersection_bounds
from inspection_decisions.ray_material_evidence_v1 import ray_states


def conditional_domain_observation(patch, origin, directions, depths, tolerance):
    """Retain q(c), not a fabricated label at q(mean(c)).

    Vertex events are CONDITIONAL samples of a rule in c. Robust interval flags
    use all-P extrema under the existing stable-denominator assumptions.
    A near target return is compatibility, not proof of this material identity.
    """
    o=np.asarray(origin,float);r=np.asarray(directions,float).reshape(-1,3)
    d=np.broadcast_to(np.asarray(depths,float),(len(r),))
    normals=np.asarray(patch['normals']);den=normals@r.T;num=1-normals@o
    with np.errstate(divide='ignore',invalid='ignore'):
        z=num[:,None]/den;p=o+z[:,:,None]*r[None]
        uv=p[:,:,:2]/p[:,:,2,None]*patch['focal']+patch['principal']
    bounds=intersection_bounds(patch,o,r)
    finite=np.isfinite(uv).all((0,2))
    # Stable denominators and source Z>0 make each projected coordinate a
    # linear-fractional function over P: vertex extrema bound its full locus.
    span=np.ptp(uv,axis=0)
    fixed=bounds['valid']&finite&(span.max(1)<=1e-10)
    event=np.zeros(z.shape,np.int8)
    event[z<d[None]-tolerance]=-1  # absent at q(c), conditional on this plane
    valid=bounds['valid']&np.isfinite(d)&(d>0)
    robust_free=valid&(bounds['upper']<d-tolerance)
    fixed_label=np.full(len(r),-1,np.int8)  # -1 UNKNOWN, 0 ABSENT
    fixed_label[fixed&robust_free]=0
    return dict(conditional_source_uv=uv,conditional_depth=z,
                conditional_absence=event==-1,robust_free=robust_free,
                fixed_coordinate=fixed,fixed_label=fixed_label,
                coordinate_span_px=span,valid=valid)


def hard_labels_feasible(coordinates, labels):
    """Only explicitly supplied point labels; unknown (-1) adds no fact."""
    known={}
    for coordinate,label in zip(coordinates,labels):
        if label==-1:continue
        key=tuple(np.asarray(coordinate,float))
        if key in known and known[key]!=int(label):return False
        known[key]=int(label)
    return True


def positive_support(points, target, tolerance, interval):
    """Operational raster support, not a material-identity certificate.

    The four-neighbor rule never fills an unobserved surface: it only restricts
    which transformed measured points may count as positive image evidence.
    It is still a raster compatibility heuristic and not an exact ray model.
    """
    state,_,_,_=ray_states(points,target.depth,target.mask,target.K,tolerance,interval)
    q=points@target.K.T
    uv=np.divide(q[:,:2],q[:,2,None],out=np.full((len(q),2),-1e8),where=np.abs(q[:,2,None])>1e-12)
    low=np.floor(uv).astype(np.int64);ok=points[:,2]>0
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        u=low[:,0]+dx;v=low[:,1]+dy
        inside=(u>=0)&(v>=0)&(u<target.depth.shape[1])&(v<target.depth.shape[0])
        depth=np.zeros(len(points));mask=np.zeros(len(points),bool)
        depth[inside]=target.depth[v[inside],u[inside]];mask[inside]=target.mask[v[inside],u[inside]]
        ok&=inside&mask&(depth>=interval[0])&(depth<=interval[1])&(np.abs(points[:,2]-depth)<=tolerance)
    stable=ok&(state==1)
    return dict(points=int(len(points)),raw_positive=int((state==1).sum()),stable_positive=int(stable.sum()),
                measured_absence=int((state==2).sum()),unknown=int((state==0).sum()))


def pair_support(history,current,T,cfg):
    R=T[:3,:3];t=T[:3,3]
    a=positive_support((current.all_xyz-t)@R,history,cfg['depth_tolerance_m'],cfg['depth_valid_interval_m'])
    b=positive_support(history.all_xyz@R.T+t,current,cfg['depth_tolerance_m'],cfg['depth_valid_interval_m'])
    return {'current_to_history':a,'history_to_current':b}
