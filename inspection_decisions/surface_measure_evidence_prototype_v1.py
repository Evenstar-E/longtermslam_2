"""Small exact positive-kernel Gamma measure factor; NOT a full task solver.

Two independent expansions: event cause labels and Poisson jump partitions.
Inputs are precomputed hit kernels K and all-ray exposure A. No images/GT.
"""
import itertools
import numpy as np
from scipy.special import gammaln,logsumexp


def marked_evidence(K,A,mass,rate=1.):
    K=np.array(K,float);A=np.array(A,float);mass=np.array(mass,float);N,F=K.shape
    with np.errstate(divide='ignore'):logK=np.log(K)
    terms=[];counts=[]
    for labels in itertools.product(range(F),repeat=N):
        n=np.bincount(labels,minlength=F);terms.append(float(logK[np.arange(N),labels].sum()+np.sum(gammaln(mass+n)-gammaln(mass)-n*np.log(rate+A))));counts.append(n)
    normalizer=logsumexp(terms);logZ=-np.sum(mass*np.log1p(A/rate))+normalizer;weights=np.exp(np.array(terms)-normalizer);mean_count=weights@np.array(counts);mean_theta=(mass+mean_count)/(rate+A)
    return float(logZ),mean_theta


def partitions(items):
    if not items:yield [];return
    head,*tail=items
    for rest in partitions(tail):
        yield [[head]]+[b.copy() for b in rest]
        for j in range(len(rest)):
            out=[b.copy() for b in rest];out[j]=[head]+out[j];yield out


def partition_evidence(K,A,mass,rate=1.):
    K=np.array(K,float);A=np.array(A,float);mass=np.array(mass,float)
    with np.errstate(divide='ignore'):logK=np.log(K)
    terms=[]
    for blocks in partitions(list(range(len(K)))):
        value=0.
        for B in blocks:value+=gammaln(len(B))+logsumexp(np.log(mass)+logK[B].sum(0)-len(B)*np.log(rate+A))
        terms.append(value)
    return float(-np.sum(mass*np.log1p(A/rate))+logsumexp(terms))
