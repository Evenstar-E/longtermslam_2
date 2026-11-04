"""Finite persistent edge *sets*, not one identity per direction category.

Interval stabbing is a standard combinatorial primitive. Minimum cardinality
is an explicit shape-complexity assumption, never measured physical identity.
"""
from functools import lru_cache
import numpy as np
from inspection_decisions.edge_conditioned_material_support_v1 import endpoints


def segment_intervals(x,features,cfg,epoch):
    rows=[]
    for j,f in enumerate(features):
        obj=endpoints(x,[f],cfg)[tuple(f['key'])];l,h=obj[:2]
        low=float(np.max(l));high=float(np.min(h))
        rows.append({'key':tuple(f['key']),'epoch':epoch,'view':f['view'],'segment':j,'lower':low,'upper':high,
                     'internally_compatible':low<=high,'minimum_extra_halfwidth_m':max(0.,(low-high)/2),'witnesses':len(l)})
    return rows


def minimum_support_cover(intervals):
    """Exact 1D finite-interval stabbing, including a disjoint packing witness.

    Input inconsistencies are reported, never dropped/widened to save a pose.
    Duplicate segments do not affect minimum support cardinality.
    """
    if any(r['lower']>r['upper'] for r in intervals):return {'feasible':False,'minimum_supports':None,'points':[],'assignment':[],'packing':[]}
    order=sorted(range(len(intervals)),key=lambda i:intervals[i]['upper']);uncovered=set(order);points=[];assignment=[None]*len(intervals);packing=[]
    for i in order:
        if i not in uncovered:continue
        point=intervals[i]['upper'];index=len(points);points.append(point);packing.append(i)
        for j in list(uncovered):
            if intervals[j]['lower']<=point<=intervals[j]['upper']:assignment[j]=index;uncovered.remove(j)
    return {'feasible':True,'minimum_supports':len(points),'points':points,'assignment':assignment,'packing':packing}


def optimal_cover_relations(intervals,maximum_exact_segments=12):
    """All minimum-cardinality covers; no arbitrary greedy match is a fact.

    Exact subset partition DP when small enough. All features are preserved;
    larger cases return an explicit unresolved relation, not a fake answer.
    """
    cover=minimum_support_cover(intervals);N=len(intervals)
    if not cover['feasible'] or N>maximum_exact_segments:return {**cover,'relations_resolved':False,'segments':N,'possible_same':None,'forced_same':None,'optimal_partition_count':None}
    lower=np.array([r['lower'] for r in intervals]);upper=np.array([r['upper'] for r in intervals]);size=1<<N;valid=np.zeros(size,bool);valid[0]=True
    low=np.full(size,-np.inf);high=np.full(size,np.inf)
    for mask in range(1,size):
        bit=mask&-mask;i=bit.bit_length()-1;rest=mask^bit;low[mask]=max(low[rest],lower[i]);high[mask]=min(high[rest],upper[i]);valid[mask]=low[mask]<=high[mask]
    @lru_cache(None)
    def solve(mask):
        if not mask:return 0,0,0,1
        first=mask&-mask;rest=mask^first;subset=rest;best=N+1;possible=0;forced=None;count=0
        while True:
            group=subset|first
            if valid[group]:
                k,p,f,c=solve(mask^group);k+=1
                if k<=best:
                    bits=[i for i in range(N) if group>>i&1];same=sum(1<<(i*N+j) for i in bits for j in bits)
                    p|=same;f|=same
                    if k<best:best=k;possible=p;forced=f;count=c
                    else:possible|=p;forced&=f;count+=c
            if not subset:break
            subset=(subset-1)&rest
        return best,possible,forced,count
    k,p,f,count=solve(size-1);assert k==cover['minimum_supports']
    decode=lambda value:[[bool(value>>(i*N+j)&1) for j in range(N)] for i in range(N)]
    return {**cover,'relations_resolved':True,'segments':N,'possible_same':decode(p),'forced_same':decode(f),'optimal_partition_count':count}


def unrestricted_support_feasibility(intervals):
    """Unknown finite cardinality alone gives no cross-period restriction."""
    valid=all(r['lower']<=r['upper'] for r in intervals)
    return {'feasible':valid,'constructive_supports':[(r['lower']+r['upper'])/2 for r in intervals] if valid else None,
            'uses_unknown_cardinality':True,'physical_correspondence_claimed':False}
