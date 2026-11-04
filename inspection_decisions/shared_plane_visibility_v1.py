"""Shared bounded-error plane visibility factor, prototype not full pipeline.

Missing/uncertain footprint rays supply NO inequality. One parameter vector
is shared across rays and periods; solving each period separately is weaker.
LP and quantifier ordering are standard; no originality claim is implied.
"""
import numpy as np
from scipy.optimize import linprog
from inspection_decisions.ray_conditioned_plane_polytope_v1 import intersection_bounds


def visibility_halfspaces(patch, origin, directions, measured_depth, epsilon_target):
    bounds=intersection_bounds(patch,origin,directions);valid=bounds['valid']
    r=np.asarray(directions)[valid];d=np.asarray(measured_depth)[valid]-epsilon_target;o=np.asarray(origin)
    uv=patch['center_uv'];pr=patch['principal'];f=patch['focal']
    L=np.array([[f,0,0],[0,f,0],[-(uv[0]-pr[0]),-(uv[1]-pr[1]),1.]])
    # z=(1-c^T L^T o)/(c^T L^T r). Denominator sign was checked
    # over the ENTIRE original feasible polytope, not at a fitted normal.
    den=patch['normals']@r.T
    sign=np.where(den.min(0)>0,1.,-1.)
    A=((o[None,:]+d[:,None]*r)@L)*sign[:,None]
    b=sign
    return {'A':A,'b':b,'used_rays':int(valid.sum()),'skipped_rays':int((~valid).sum())}


def feasible_shared_plane(patch, factors):
    A=np.concatenate([patch['A']]+[v['A'] for v in factors]);b=np.concatenate([patch['b']]+[v['b'] for v in factors])
    ans=linprog(np.zeros(3),A_ub=A,b_ub=b,bounds=[(None,None)]*3,method='highs')
    return {'feasible':True if ans.status==0 else False if ans.status==2 else None,'status':int(ans.status),
            'constraints':len(b),'parameters':ans.x.tolist() if ans.x is not None else None}
