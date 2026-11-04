"""Exact cardinality model counting for shared positive occupancy constraints.

Uses known DPLL/component caching/Beta-Bernoulli tools, not new inference theory.
Event probabilities are NOT normalized raw RGB-D likelihoods.
"""
import math,itertools
from collections import Counter
import numpy as np
from scipy.special import betaln,logsumexp
from inspection_decisions.completion_compatible_anchor_v1 import cells


def normalize(clauses):
    """Remove duplicates and compact names, preserving active variable count."""
    clauses=tuple(sorted(set(int(c) for c in clauses)))
    if not clauses:return ()
    if 0 in clauses:return (0,)
    union=0
    for c in clauses:union|=c
    labels={v:i for i,v in enumerate(cells(union))}
    return tuple(sorted(sum(1<<labels[v] for v in cells(c)) for c in clauses))


def variable_count(clauses):
    union=0
    for c in clauses:union|=c
    return union.bit_count()


def binomial(n):return np.array([math.comb(n,j) for j in range(n+1)],float)


class CounterCompiler:
    def __init__(self):self.cache={};self.calls=0;self.hits=0;self.splits=0

    def count(self,clauses):
        key=normalize(clauses);self.calls+=1
        if key in self.cache:self.hits+=1;return self.cache[key]
        if not key:result=np.ones(1)
        elif key==(0,):result=np.zeros(1)
        else:
            n=variable_count(key);units=0
            for c in key:
                if c.bit_count()==1:units|=c
            if units:
                left=tuple(c for c in key if not c&units)
                free=n-units.bit_count()-variable_count(left)
                reduced=np.convolve(self.count(left),binomial(free))
                result=np.r_[np.zeros(units.bit_count()),reduced]
            else:
                # Independent incidence components factor BEFORE rho integration.
                remaining=list(key);components=[]
                while remaining:
                    component=[remaining.pop()];seen=component[0]
                    while True:
                        attached=[c for c in remaining if c&seen]
                        if not attached:break
                        for c in attached:remaining.remove(c);component.append(c);seen|=c
                    components.append(component)
                if len(components)>1:
                    self.splits+=1;result=np.ones(1)
                    for component in components:result=np.convolve(result,self.count(component))
                else:
                    degree=Counter(v for c in key for v in cells(c));v=max(degree,key=lambda x:(degree[x],-x));bit=1<<v
                    zero=tuple(c&~bit for c in key)
                    one=tuple(c for c in key if not c&bit)
                    if 0 in zero:z=np.zeros(n)
                    else:z=np.convolve(self.count(zero),binomial(n-1-variable_count(zero)))
                    o=np.convolve(self.count(one),binomial(n-1-variable_count(one)))
                    result=np.r_[z,0.]+np.r_[0.,o]
            assert len(result)==n+1 and np.all(result>=0)
        self.cache[key]=result
        return result


def event(groups):
    forbidden=0;bands=[]
    for g in groups:forbidden|=g['free'];bands.extend(g['bands'])
    bands=list(set(b&~forbidden for b in bands))
    if 0 in bands:return None
    units=0
    for b in bands:
        if b.bit_count()==1:units|=b
    residual=tuple(b for b in bands if not b&units)
    return {'free_count':forbidden.bit_count(),'unit_count':units.bit_count(),
            'clauses':normalize(residual),'active_count':variable_count(residual)}


def probability(spec,compiler,prior='beta'):
    if spec is None:return -np.inf
    f,r,n=spec['free_count'],spec['unit_count'],spec['active_count']
    counts=compiler.count(spec['clauses']);assert len(counts)==n+1
    j=np.arange(n+1);ok=counts>0
    if prior=='beta':terms=np.log(counts[ok])+betaln(1+r+j[ok],1+f+n-j[ok])
    elif prior=='fixed':terms=np.log(counts[ok])-(f+r+n)*np.log(2.)
    else:raise ValueError(prior)
    return float(logsumexp(terms))


def sis_probability(spec,particles,seed):
    """Rao-Blackwellized conditional SIS for SAME Beta(1,1) constraint event.

    Analytically integrates forced cells. Each remaining clause proposal samples
    its unassigned cells from the conditional Beta-Bernoulli law given a hit.
    No resampling, clipping, or fake zero-likelihood fallback.
    """
    if spec is None:return -np.inf,0.
    f,r,n=spec['free_count'],spec['unit_count'],spec['active_count']
    initial=float(betaln(1+r,1+f));clauses=spec['clauses']
    if not clauses:return initial,float(particles)
    rng=np.random.default_rng(seed);state=np.full((particles,n),-1,np.int8)
    alpha=np.full(particles,1.+r);beta=np.full(particles,1.+f);weight=np.zeros(particles)
    for clause in clauses:
        variables=np.array(list(cells(clause)),int)
        satisfied=(state[:,variables]==1).any(1)
        impossible=~satisfied&np.all(state[:,variables]!=-1,axis=1)
        weight[impossible]=-np.inf
        for unknown_mask in range(1,1<<len(variables)):
            unknown=np.array([(unknown_mask>>j)&1 for j in range(len(variables))],bool)
            take=np.flatnonzero(np.isfinite(weight)&~satisfied&np.all((state[:,variables]==-1)==unknown,axis=1))
            if not len(take):continue
            vv=variables[unknown];m=len(vv)
            patterns=((np.arange(1,1<<m)[:,None]>>np.arange(m))&1).astype(np.int8)
            ones=patterns.sum(1);aa=alpha[take,None];bb=beta[take,None]
            lp=betaln(aa+ones,bb+m-ones)-betaln(aa,bb)
            norm=logsumexp(lp,axis=1);weight[take]+=norm
            cumulative=np.cumsum(np.exp(lp-norm[:,None]),axis=1);cumulative[:,-1]=1.
            choice=(cumulative<rng.random(len(take))[:,None]).sum(1)
            assignment=patterns[choice];state[take[:,None],vv[None,:]]=assignment
            success=assignment.sum(1);alpha[take]+=success;beta[take]+=m-success
        assert ((state[:,variables]==1).any(1)|~np.isfinite(weight)).all()
    if not np.isfinite(weight).any():return -np.inf,0.
    logmean=float(logsumexp(weight)-np.log(particles))
    ess=float(np.exp(2*logsumexp(weight)-logsumexp(2*weight)))
    return initial+logmean,ess


def exhaustive_probability(groups,prior='beta'):
    """Small-problem independent control over all cells, including forbidden."""
    universe=0
    for g in groups:
        universe|=g['free']
        for b in g['bands']:universe|=b
    ids=list(cells(universe));n=len(ids);terms=[]
    for values in itertools.product([0,1],repeat=n):
        assignment=sum(int(v)<<j for j,v in zip(ids,values));j=sum(values)
        if all(not(assignment&g['free']) and all(assignment&b for b in g['bands']) for g in groups):
            terms.append(float(betaln(1+j,1+n-j)) if prior=='beta' else -n*np.log(2.))
    return float(logsumexp(terms))
