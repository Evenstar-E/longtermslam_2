"""Keep the feature-cache source frozen; version actual region inference."""
from pathlib import Path


def main():
    source=Path('inspection_decisions/material_surface_distribution_matching_v15.py').read_text(encoding='utf-8')
    start=source.index('def dense_modes(');end=source.index('\ndef compatibility(',start)
    mode='''def dense_modes(a,b,cfg):
    tree=b['_tree'] if '_tree' in b else cKDTree(b['xyz']);b['_tree']=tree;points=[];means=[];cov=[];mass=[];prior=[];sources=[];cosines=[];normalizer=np.zeros(len(a['xyz']))
    for start in range(0,len(a['xyz']),64):
        ix=np.arange(start,min(start+64,len(a['xyz'])));sim=np.asarray(a['desc'][ix],np.float32)@np.asarray(b['desc'],np.float32).T;kernel=np.exp((sim-1)/.12);den=kernel@b['area'];normalizer[ix]=den;work=sim.copy();peaks=[];usable=[]
        for _ in range(cfg['modes_per_source']):
            peak=work.argmax(1);ok=np.isfinite(work[np.arange(len(ix)),peak]);peaks.append(peak);usable.append(ok);d2=np.sum((b['xyz'][None]-b['xyz'][peak,None])**2,-1);work[d2<cfg['mode_separation_m']**2]=-np.inf
        peaks=np.array(peaks).T;usable=np.array(usable).T
        for row,i in enumerate(ix):
            peak=peaks[row,usable[row]];centers=b['xyz'][peak];union=np.unique(np.concatenate([np.asarray(h,int) for h in tree.query_ball_point(centers,cfg['mode_radius_m'])]));distance=np.linalg.norm(b['xyz'][union,None]-centers[None],axis=-1);assign=distance.argmin(1)
            for mode,j in enumerate(peak):
                near=union[(assign==mode)&(distance[:,mode]<=cfg['mode_radius_m'])];w=kernel[row,near]*b['area'][near];total=w.sum();w/=max(total,1e-30);mu=w@b['xyz'][near];delta=b['xyz'][near]-mu;C=np.einsum('n,ni,nj->ij',w,delta,delta)
                points.append(b['xyz'][j]);means.append(mu);cov.append(C);mass.append(total/max(den[row],1e-30));prior.append(b['area'][near].sum());sources.append(i);cosines.append(sim[row,j])
    result=dict(source=np.array(sources,int),point=np.array(points),mean=np.array(means),cov=np.array(cov),mass=np.array(mass),prior=np.array(prior),cosine=np.array(cosines),slot_normalizer=normalizer)
    for key in ['mass','prior']:assert (np.bincount(result['source'],weights=result[key],minlength=len(a['xyz']))<=1+1e-6).all(),key
    return result

'''
    source=source[:start]+mode+source[end:];source=source[:source.index('\nclass LearnedPair:')]
    source+='''
class RegionPair:
    def __init__(self,pair,forward,backward,cfg):
        self.p=pair;self.modes=[forward,backward];self.cfg=cfg;self.precision=[];self.prefactor=[]
        s2=cfg['marginal_sigma_m']**2
        for m in self.modes:
            S=s2*np.eye(3)+m['cov'];self.precision.append(np.linalg.inv(S));self.prefactor.append(s2**1.5/np.sqrt(np.linalg.det(S)))
    def one_mixture(self,x,a,m,precision,prefactor,kind):
        src=m['source'];target=m['point'] if kind==0 else m['mean'];r=x[src]-target;s2=self.cfg['marginal_sigma_m']**2
        if kind==2:K=prefactor*np.exp(-.5*np.einsum('ni,nij,nj->n',r,precision,r))
        else:K=np.exp(-.5*np.sum(r*r,axis=1)/s2)
        n=len(x);post=m['mass'];prior=m['prior'];remaining=np.maximum(0,1-np.bincount(src,weights=post,minlength=n));base=np.maximum(0,1-np.bincount(src,weights=prior,minlength=n));u=self.cfg['outlier_kernel'];num=remaining*u+np.bincount(src,weights=post*K,minlength=n);den=base*u+np.bincount(src,weights=prior*K,minlength=n)
        ratio=num/np.maximum(den,1e-20)*m['slot_normalizer']/a['normalizer'];return -np.log(np.maximum(ratio,1e-12)).mean()
    def values(self,T):
        p=self.p;geo,rgb,_=p.nearest(T);forward=apply(T,p.a['xyz']);backward=apply(inverse(T),p.b['xyz']);fixed,detail=p.projected(T,False);free=fixed-1+.5*(detail[0]['kernel'].mean()+detail[1]['kernel'].mean());fusion=[];mixture=np.zeros(3)
        for h,(a,b,tree,x) in enumerate([(p.a,p.db,p.tb,forward),(p.b,p.da,p.ta,backward)]):
            d,j=tree.query(x,k=4);sim=np.sum(a['desc'][:,None]*b['desc'][j],axis=-1);fusion.append(np.min(np.minimum(d/.06,1)**2+.5*(1-sim),axis=1).mean())
            for kind in range(3):mixture[kind]+=.5*self.one_mixture(x,a,self.modes[h],self.precision[h],self.prefactor[h],kind)
        return np.r_[geo,rgb,np.mean(fusion)+free,geo+free+self.cfg['marginal_weight']*mixture]
'''
    path=Path('inspection_decisions/material_surface_distribution_v15b.py');assert not path.exists();compile(source,str(path),'exec');path.write_text(source,encoding='utf-8');print('V15b disjoint region / moment model generated')


if __name__=='__main__':main()
