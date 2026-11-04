"""Exhaustive finite assignment check of derived bounds, not task validation."""
import itertools
import numpy as np
from scipy.special import logsumexp
from inspection_decisions.shared_contamination_transport_v1 import shared_bounds

def main():
    rng=np.random.default_rng(20261024);count=0
    for n in [2,3]:
        assignments=[]
        for cols in itertools.product(range(-1,n),repeat=n):
            chosen=[c for c in cols if c>=0]
            if len(set(chosen))<len(chosen):continue
            assignments.append((np.array([i for i,c in enumerate(cols) if c>=0],int),np.array(chosen,int)))
        for _ in range(20):
            w=rng.normal(size=(3,n,n));r=rng.normal(size=3);p=np.ones(3)/3;lb,ub=shared_bounds(w,r,p)
            exact=max(logsumexp(np.log(p)+np.array([a[ij].sum() for a in w])+r)/n for ij in assignments)
            assert lb<=exact+1e-12 and exact<=ub+1e-12;count+=1
    # Identical regimes must reduce to ordinary maximum matching.
    w=np.repeat(np.eye(3)[None],3,axis=0);lb,ub=shared_bounds(w,np.zeros(3),np.ones(3)/3);assert abs(lb-1)<1e-12 and abs(ub-1)<1e-12;count+=1
    print('PASS',count,'finite mathematical checks')

if __name__=='__main__':main()
