"""Bounded-depth local planar patches: interval geometry, not a novelty claim.

Each observed pixel supplies TWO inverse-depth halfspaces for one shared
plane. Fractional-linear extrema are taken over its vertices only when the
denominator keeps a strict sign on the whole polytope. No scalar sigma proxy.
"""
from itertools import combinations
import numpy as np

TRIPLES=np.array(list(combinations(range(18),3)))


def fit_patch(depth3, center_uv, focal, principal, epsilon):
    z=np.asarray(depth3,float).reshape(-1)
    assert len(z)==9 and epsilon>0 and np.all(z>epsilon)
    yy,xx=np.mgrid[-1:2,-1:2];X=np.c_[xx.ravel(),yy.ravel(),np.ones(9)]
    # inverse depth = a*pixel_dx + b*pixel_dy + c. This basis is much better
    # conditioned than global camera-normal coordinates at a tiny image patch.
    A=np.r_[X,-X];b=np.r_[1/(z-epsilon),-1/(z+epsilon)]
    mat=A[TRIPLES];rhs=b[TRIPLES];nonsingular=np.abs(np.linalg.det(mat))>1e-12
    if not nonsingular.any():return None
    v=np.linalg.solve(mat[nonsingular],rhs[nonsingular,:,None])[...,0]
    v=v[np.all(v@A.T<=b[None,:]+1e-12,axis=1)]
    if not len(v):return None
    v=np.unique(np.round(v,14),axis=0)
    center=np.array(center_uv,float);principal=np.array(principal,float)
    normals=np.c_[v[:,0]*focal,v[:,1]*focal,v[:,2]-v[:,0]*(center[0]-principal[0])-v[:,1]*(center[1]-principal[1])]
    return {'coefficients':v,'normals':normals,'A':A,'b':b,'center_uv':center,'focal':focal,'principal':principal,'epsilon':epsilon}


def intersection_bounds(patch, origin, directions):
    """All plausible planes intersect a target ray inside the observed patch.

Origin/directions expressed in SOURCE camera coordinates; target camera-Z
parameter is preserved. 'valid' includes robust support-footprint membership.
This is CONDITIONAL on a planar continuous patch being physically present.
"""
    normals=patch['normals'];o=np.asarray(origin);r=np.asarray(directions)
    den=normals@r.T;num=1-normals@o
    same_sign=(den.min(0)>1e-12)|(den.max(0)<-1e-12)
    with np.errstate(divide='ignore',invalid='ignore'):z=num[:,None]/den
    finite=np.isfinite(z).all(0)&(z.min(0)>0)
    low=z.min(0);high=z.max(0)
    # Projection along a line is fractional-linear. If camera-Z stays positive,
    # footprint extrema also occur at the two depth-interval endpoints.
    points=np.stack([o[None,:]+low[:,None]*r,o[None,:]+high[:,None]*r])
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=points[...,:2]/points[...,2,None]*patch['focal']+patch['principal']
    support=(points[...,2]>0).all(0)&(np.abs(uv-patch['center_uv'])<=1+1e-10).all((0,2))
    return {'valid':same_sign&finite&support,'lower':low,'upper':high,
            'denominator_sign_stable':same_sign,'inside_source_patch':support}
