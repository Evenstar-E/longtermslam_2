"""Mechanism and independent solver equivalence checks, not performance evidence."""
import itertools
import numpy as np
from inspection_decisions.completion_compatible_anchor_v1 import (
    bitset,conflict,pairwise_ok,explicit_shape_feasible,enumerate_pairs,pair_displacements,envelope,lazy_envelope)


def main():
    rng=np.random.default_rng(20261024)
    count=0
    for trial in range(120):
        groups=[{'free':int(rng.integers(0,256)),
                 'bands':[int(rng.integers(0,256)) for _ in range(int(rng.integers(0,4)))]} for _ in range(3)]
        brute=any(all(not(s&g['free']) and all(s&b for b in g['bands']) for g in groups) for s in range(256))
        assert (conflict(groups) is None)==brute
        if trial<20:assert explicit_shape_feasible(groups)['feasible']==brute
        count+=1
    groups=[{'free':0,'bands':[bitset([0,1])]}, {'free':bitset([0]),'bands':[]},{'free':bitset([1]),'bands':[]}]
    assert pairwise_ok(groups) and conflict(groups) is not None
    def branch(free,x):
        T=np.eye(4);T[0,3]=x
        return {'free':free,'bands':[],'T':T}
    branches=[[branch(1,0),branch(0,1)],[branch(2,0),branch(0,2)]]
    h=groups[0];p,f,e=enumerate_pairs(h,branches)
    assert e is not None and p.all() and not f[0,0] and f.sum()==3
    d=pair_displacements(*branches,np.zeros(3));a=envelope(d,f);b=lazy_envelope(h,branches,d)
    assert np.allclose(a['lower'],b['lower']) and np.allclose(a['upper'],b['upper'])
    print({'random_exhaustive_equivalence':count,'explicit_MILP_equivalence':20,'higher_order_mechanism':True,'lazy_extrema':True})


if __name__=='__main__':main()
