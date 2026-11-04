"""Shared occupancy elimination and task-directed finite-pool support queries.

Known Boolean elimination/search tools. Exact for positive hit clauses and
forbidden cells, a conservative discrete observation relaxation, not a new
general shape/pose certification theorem.
"""
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def bitset(indices):
    out = 0
    for v in indices: out |= 1 << int(v)
    return out


def cells(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length()-1
        mask ^= low


def conflict(groups):
    forbidden = 0
    for g in groups: forbidden |= g['free']
    for period, g in enumerate(groups):
        for ray, band in enumerate(g['bands']):
            if not (band & ~forbidden):
                return {'period': period, 'ray': ray, 'band': band, 'forbidden': forbidden}
    return None


def pairwise_ok(groups):
    return all(conflict([groups[a], groups[b]]) is None
               for a,b in [(0,1),(0,2),(1,2)])


def explicit_shape_feasible(groups, time_limit=5.):
    """Independent explicit binary occupancy feasibility control, no elimination."""
    forbidden = 0
    bands = []
    for g in groups:
        forbidden |= g['free']; bands.extend(g['bands'])
    universe = sorted(set(v for b in bands for v in cells(b)))
    if not universe:
        return {'feasible': not bands, 'status': 0 if not bands else 2, 'variables': 0, 'elapsed_s': 0.}
    index = {v:i for i,v in enumerate(universe)}
    rr,cc = [],[]
    for j,b in enumerate(bands):
        for v in cells(b): rr.append(j); cc.append(index[v])
    A=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(bands),len(universe))).tocsc()
    upper=np.array([0. if (forbidden>>v)&1 else 1. for v in universe])
    start=time.perf_counter()
    result=milp(np.zeros(len(universe)),integrality=np.ones(len(universe)),
                bounds=Bounds(np.zeros(len(universe)),upper),
                constraints=LinearConstraint(A,np.ones(len(bands)),np.full(len(bands),np.inf)),
                options={'time_limit':time_limit})
    feasible=True if result.status==0 else False if result.status==2 else None
    return {'feasible':feasible,'status':int(result.status),'variables':len(universe),
            'elapsed_s':time.perf_counter()-start}


def pair_displacements(branch1,branch2,anchor):
    a=np.array([b['T'][:3,:3]@anchor+b['T'][:3,3] for b in branch1])
    b=np.array([b['T'][:3,:3]@anchor+b['T'][:3,3] for b in branch2])
    return b[None,:,:]-a[:,None,:]


def enumerate_pairs(history,branches):
    n,m=map(len,branches);pair=np.zeros((n,m),bool);full=pair.copy();example=None
    for k,a in enumerate(branches[0]):
        for l,b in enumerate(branches[1]):
            groups=[history,a,b]
            pair[k,l]=pairwise_ok(groups)
            if pair[k,l]:
                fail=conflict(groups);full[k,l]=fail is None
                if fail is not None and example is None:
                    example={'k1':k,'k2':l,'period':fail['period'],'ray':fail['ray'],
                             'band_cells':list(cells(fail['band'])),
                             'free_coverage_by_period':[list(cells(fail['band']&g['free'])) for g in groups]}
    return pair,full,example


def envelope(displacement, feasible):
    x=displacement[feasible]
    if not len(x): return {'empty':True,'lower':None,'upper':None,'report':False,'radius_m':None}
    lo,hi=x.min(0),x.max(0);radius=float(np.linalg.norm(hi-lo)/2)
    return {'empty':False,'lower':lo.tolist(),'upper':hi.tolist(),'report':radius<=.05,'radius_m':radius}


def lazy_envelope(history,branches,displacement):
    """Exact LOCAL axis extrema; cache is fresh, independent of exhaustive masks."""
    n,m=displacement.shape[:2];values=displacement.reshape(-1,3);cache={};bounds=[]
    start=time.perf_counter()
    for axis in range(3):
        ends=[]
        for sign in [1,-1]:
            found=None
            for j in np.argsort(sign*values[:,axis],kind='stable'):
                j=int(j)
                if j not in cache:
                    k,l=divmod(j,m)
                    cache[j]=conflict([history,branches[0][k],branches[1][l]]) is None
                if cache[j]: found=float(values[j,axis]);break
            ends.append(found)
        bounds.append(ends)
    result={'empty':bounds[0][0] is None,'lower':[b[0] for b in bounds],
            'upper':[b[1] for b in bounds],'checks':len(cache),'full_pool_size':n*m,
            'elapsed_s':time.perf_counter()-start}
    return result


def global_selection(all_branches, feasible_masks, time_limit=20.):
    """Global one asset per observed slot, two epochs, finite pose pairs."""
    n=len(all_branches);variables=[];cost=[];rr=[];cc=[]
    for i,(branches,mask) in enumerate(zip(all_branches,feasible_masks)):
        for k,l in zip(*np.nonzero(mask)):
            a,b=branches[0][k],branches[1][l];j=len(variables)
            variables.append((i,int(k),int(l)));cost.append(a['cost']+b['cost'])
            rr.extend([i,n+a['slot'],2*n+b['slot']]);cc.extend([j]*3)
    if not variables:return {'status':2,'selection':None,'elapsed_s':0.,'variables':0}
    A=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(3*n,len(variables))).tocsc()
    start=time.perf_counter()
    result=milp(np.array(cost),integrality=np.ones(len(cost)),bounds=Bounds(0,1),
                constraints=LinearConstraint(A,np.ones(3*n),np.ones(3*n)),
                options={'time_limit':time_limit,'mip_rel_gap':0.})
    selected=None
    if result.x is not None:
        chosen=np.flatnonzero(result.x>.5)
        assert len(chosen)==n
        selected=sorted([variables[j] for j in chosen])
        assert len({i for i,_,_ in selected})==n
        for t in [0,1]:
            assert len({all_branches[i][t][(k,l)[t]]['slot'] for i,k,l in selected})==n
    return {'status':int(result.status),'selection':selected,'elapsed_s':time.perf_counter()-start,
            'variables':len(cost),'objective':float(result.fun) if result.fun is not None else None,
            'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None}
