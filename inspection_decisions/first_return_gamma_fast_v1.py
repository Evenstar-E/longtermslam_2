"""Same integer-count conjugate factors, cached log-Gamma lookup backend.

No approximation, subsampling, pruning or parameter changes. Independent
equation checks compare against direct scalar and reference pair evaluation.
"""
import numpy as np
from scipy.special import gammaln
from inspection_decisions.first_return_gamma_material_v1 import unary,posterior


def pair_costs(H,A,B,cfg,method):
    u=unary(H,A,cfg,method);v=unary(H,B,cfg,method);cost=u[:,None]+v[None,:]
    if method in ['gamma_independent','gamma_mean']:return cost
    a,b,alpha,beta=posterior(H,cfg)
    for k in range(A.shape[-1]):
        ia=np.flatnonzero((A[:,0,k]+A[:,1,k])>0);ib=np.flatnonzero((B[:,0,k]+B[:,1,k])>0)
        if not len(ia) or not len(ib):continue
        n=A[ia,0,k].astype(np.int64);p=B[ib,0,k].astype(np.int64);nn=n[:,None]+p[None,:]
        if method=='gamma_joint':
            L=A[ia,1,k];M=B[ib,1,k];table=gammaln(a[k]+np.arange(int(n.max()+p.max())+1))
            cross=table[nn]-table[n,None]-table[p][None,:]+table[0]
            cross+=(a[k]+n[:,None])*np.log(b[k]+L[:,None])+(a[k]+p[None,:])*np.log(b[k]+M[None,:])
            cross-=(a[k]+nn)*np.log(b[k]+L[:,None]+M[None,:])+a[k]*np.log(b[k])
        else:
            m=A[ia,2,k].astype(np.int64);q=B[ib,2,k].astype(np.int64);mm=m[:,None]+q[None,:]
            def correction(base,left,right,total):
                tab=gammaln(base+np.arange(int(left.max()+right.max())+1));return tab[total]-tab[left,None]-tab[right][None,:]+tab[0]
            cross=correction(alpha[k],n,p,nn)+correction(beta[k],m,q,mm)-correction(alpha[k]+beta[k],n+m,p+q,nn+mm)
        cost[np.ix_(ia,ib)]-=cross
    return cost
