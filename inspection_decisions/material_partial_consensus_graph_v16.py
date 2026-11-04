"""Rare/local partial correspondence hypotheses; no private labels/file I/O."""
import numpy as np
from scipy.spatial.transform import Rotation
from inspection_decisions.material_surface_distribution_v15b import compatibility
from inspection_decisions.textured_material_matching_v1 import rigid,apply
from inspection_decisions.material_surface_patch_registration_v9 import inverse

FAMILIES=['global24_tls','local24_tls','local3_tls','local3_partial']


def graph_state(a,m,cfg):
    C,y=compatibility(a,m,cfg,False);gram=C@C;A=gram*C;prior=np.sqrt(np.maximum(m['mass'],1e-12)).astype(np.float32);v=prior.copy()
    for _ in range(8):v=A@v;v/=max(np.linalg.norm(v),1e-12)
    degree=C.sum(1);den=np.maximum(degree**2-np.sum(C*C,axis=1),1e-12);cohesion=A.sum(1)/den
    return dict(C=C,A=A,gram=gram,global_weight=v,cohesion=cohesion,y=y,x=a['xyz'][m['source']])


def select_seeds(g,cfg,local):
    order=np.argsort(-g['cohesion' if local else 'global_weight'],kind='stable');seeds=[];suppressed=np.zeros(len(order),bool)
    for h in order:
        h=int(h)
        if local:
            if suppressed[h]:continue
            norm=np.sqrt(np.maximum(g['gram'][h,h]*np.diag(g['gram']),1e-20));cosine=g['gram'][h]/norm;suppressed|=(cosine>cfg['graph_suppression_cosine'])&(g['C'][h]>cfg['graph_suppression_compatibility']);suppressed[h]=True
        elif seeds and np.min(np.linalg.norm(g['x'][seeds]-g['x'][h],axis=1))<.04:continue
        seeds.append(h)
        if len(seeds)>=cfg['seed_budget']:break
    return np.array(seeds,int)


def neighborhood(g,m,h,cfg,local):
    if local:
        degree=np.maximum(g['A'].sum(1),1e-12);rank=g['A'][h]/np.sqrt(degree[h]*degree)
    else:rank=g['A'][h]*g['global_weight']
    order=np.argsort(-rank,kind='stable');ix=np.r_[h,order[order!=h][:cfg['local_edges']-1]]
    if local:
        B=g['A'][np.ix_(ix,ix)];d=np.maximum(B.sum(1),1e-12);B=B/np.sqrt(d[:,None]*d[None]);w=np.sqrt(np.maximum(m['mass'][ix],1e-12)).astype(np.float32)
        for _ in range(8):w=B@w;w/=max(np.linalg.norm(w),1e-12)
        w*=g['C'][h,ix]
    else:w=g['A'][h,ix]*g['global_weight'][ix]
    return ix,np.asarray(w,float)


def unique_support(ix,weight,m,count):
    chosen=[];seen=set()
    for k in np.argsort(-weight,kind='stable'):
        source=int(m['source'][ix[k]])
        if source in seen:continue
        seen.add(source);chosen.append(int(k))
        if len(chosen)>=count:break
    return np.array(chosen,int)


def triangle_start(g,m,h,ix,cfg):
    x,y=g['x'][ix],g['y'][ix];u=x-x[0];v=y-y[0];area_x=np.linalg.norm(np.cross(u[:,None],u[None]),axis=-1);area_y=np.linalg.norm(np.cross(v[:,None],v[None]),axis=-1);C=g['C'][np.ix_(ix,ix)];score=C[0,:,None]*C[0,None,:]*C*np.minimum(area_x,area_y)
    source=m['source'][ix];valid=(source[:,None]!=source[None])&(source[:,None]!=source[0])&(source[None]!=source[0])&(area_x>1e-6)&(area_y>1e-6);score=np.where(valid,score,-1.)
    k,l=np.unravel_index(score.argmax(),score.shape)
    if score[k,l]<0:return None
    return rigid(x[[0,k,l]],y[[0,k,l]])


def source_minimum(values,source):
    # Stable one nearest correspondence per source; no duplicate source votes.
    order=np.lexsort((np.arange(len(source)),values,source));_,first=np.unique(source[order],return_index=True);return order[first]


def tls_energy(T,x,y,source,cfg):
    r2=np.sum((apply(T,x)-y)**2,axis=1);chosen=source_minimum(r2,source);return float(np.minimum(r2[chosen]/cfg['tls_cutoff_m']**2,1).mean())


def partial_statistics(T,x,y,source,target,prior,cfg,volume):
    r2=np.sum((apply(T,x)-y)**2,axis=1);sigma=cfg['fit_sigma_m'];density=np.exp(-.5*r2/sigma**2)/(2*np.pi*sigma*sigma)**1.5;unique,si=np.unique(source,return_inverse=True);mass=np.bincount(si,weights=prior,minlength=len(unique));null=np.maximum(0,1-mass);weighted=prior*density;den=null/volume+np.bincount(si,weights=weighted,minlength=len(unique));den=np.maximum(den,1e-30);responsibility=weighted/den[si]
    # One feasibility projection: reduces weights, so source-row sums stay<=1.
    # Not an exact optimal-transport solution or an exact EM M-step.
    target_sum=np.bincount(target,weights=responsibility);weight=responsibility/np.maximum(target_sum[target],1.)
    return float(-np.log(den).mean()),weight,responsibility,si


def retract_between(T,U,alpha):
    D=U@inverse(T);S=np.eye(4);S[:3,:3]=Rotation.from_rotvec(alpha*Rotation.from_matrix(D[:3,:3]).as_rotvec()).as_matrix();S[:3,3]=alpha*D[:3,3];return S@T


def refine(T,g,m,ix,cfg,partial,volume):
    x,y=g['x'][ix],g['y'][ix];source=m['source'][ix];_,target=np.unique(y,axis=0,return_inverse=True);prior=m['mass'][ix]
    def objective(Q):return partial_statistics(Q,x,y,source,target,prior,cfg,volume)[0] if partial else tls_energy(Q,x,y,source,cfg)
    value=objective(T);path=[value]
    for _ in range(cfg['fit_steps']):
        if partial:_,weights,_,_=partial_statistics(T,x,y,source,target,prior,cfg,volume)
        else:
            r2=np.sum((apply(T,x)-y)**2,axis=1);chosen=source_minimum(r2,source);weights=np.zeros(len(x));weights[chosen]=(r2[chosen]<cfg['tls_cutoff_m']**2)
        U=rigid(x,y,weights) if np.count_nonzero(weights>1e-10)>=3 else None;accepted=False
        if U is not None:
            for alpha in [1.,.5,.25]:
                Q=retract_between(T,U,alpha);v=objective(Q)
                if v<value-1e-12:T=Q;value=v;accepted=True;break
        path.append(value)
        if not accepted:break
    return T,np.array(path)


def proposals(a,m,target_record,cfg):
    g=graph_state(a,m,cfg);global_seeds=select_seeds(g,cfg,False);local_seeds=select_seeds(g,cfg,True);volume=float(np.prod(np.ptp(target_record['xyz'],axis=0)+cfg['outlier_padding_m']));out={name:[] for name in FAMILIES};trace={};starts=[]
    for local,seeds in [(False,global_seeds),(True,local_seeds)]:
        for n,h in enumerate(seeds):
            ix,w=neighborhood(g,m,int(h),cfg,local);chosen=unique_support(ix,w,m,cfg['initial_support']);T=rigid(g['x'][ix[chosen]],g['y'][ix[chosen]],w[chosen]+1e-12)
            if T is not None:
                name='local24_tls' if local else 'global24_tls';U,path=refine(T,g,m,ix,cfg,False,volume);out[name].append(U);trace[f'{name}_{n}_path']=path
            if local:
                initial=triangle_start(g,m,int(h),ix,cfg)
                if initial is None:continue
                starts.append(initial)
                for partial,name in [(False,'local3_tls'),(True,'local3_partial')]:
                    U,path=refine(initial.copy(),g,m,ix,cfg,partial,volume);out[name].append(U);trace[f'{name}_{n}_initial']=initial;trace[f'{name}_{n}_path']=path
    for name in out:out[name]=np.array(out[name]).reshape(-1,4,4)
    trace['global_seeds']=global_seeds;trace['local_seeds']=local_seeds
    return out,trace
