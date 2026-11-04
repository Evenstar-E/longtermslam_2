"""Common-pool rigid inference with normalized continuous material evidence."""
import numpy as np
import torch
from scipy.special import logsumexp
from inspection_decisions import textured_material_matching_v1 as old


METHODS=['ordinary','ray','hard2','soft2','soft3','soft3_no_coverage','soft3_no_visibility']


def inverse(T):
    T=np.asarray(T);out=np.zeros_like(T);out[...,3,3]=1;out[...,:3,:3]=np.swapaxes(T[...,:3,:3],-1,-2)
    out[...,:3,3]=-np.einsum('...ij,...j->...i',out[...,:3,:3],T[...,:3,3]);return out


def unique_poses(values):
    out=[]
    for T in values:
        if T is not None and np.isfinite(T).all() and not any(np.max(abs(T-U))<1e-7 for U in out):out.append(np.array(T))
    return np.array(out)


def spectral_pose(a,b,q,cfg,match=None):
    """Ordinary second-order compatibility proposal, NOT author SC2-PCR port."""
    if match is None:
        order=np.argsort(-q,axis=1)[:,:cfg['graph_topk']]
        ia=np.repeat(np.arange(len(q)),order.shape[1]);ib=order.ravel();weight=q[ia,ib]
        primary=np.arange(len(q))*order.shape[1]
        extra=np.setdiff1d(np.arange(len(ia)),primary)
        extra=extra[np.argsort(-weight[extra])[:max(0,192-len(primary))]]
        take=np.r_[primary,extra];ia,ib,weight=ia[take],ib[take],weight[take]
    else:
        ia,ib=match['ia'],match['ib'];weight=np.exp(-match['distance']**2/.5)
    if len(ia)<3:return None
    x,y=a['xyz'][ia],b['xyz'][ib]
    dx=np.linalg.norm(x[:,None]-x[None],axis=2);dy=np.linalg.norm(y[:,None]-y[None],axis=2)
    C=np.maximum(0.,1-((dx-dy)/cfg['graph_sigma_m'])**2)
    C[(ia[:,None]==ia[None])|(ib[:,None]==ib[None])]=0
    C2=C*(C@C)/max(len(C),1)
    v=np.maximum(weight,1e-6);v/=max(np.linalg.norm(v),1e-12)
    for _ in range(cfg['graph_iterations']):
        v=C2@v;v/=max(np.linalg.norm(v),1e-12)
    seeds=np.argsort(-v)[:3];best=None
    for seed in seeds:
        w=C2[seed]*v;ix=np.argsort(-w)[:min(24,len(w))];T=old.rigid(x[ix],y[ix],w[ix]+1e-10)
        if T is None:continue
        r=np.linalg.norm(old.apply(T,x)-y,axis=1)
        value=np.sum(weight*np.exp(-.5*(r/.025)**2))
        if best is None or value>best[0]:best=(value,T)
    return None if best is None else best[1]


def ray_batch(world,frames,slot):
    B,N=world.shape[:2];hit=np.zeros((B,N),bool);gap=np.full((B,N),np.inf)
    normal=np.zeros((B,N,3));offset=np.zeros((B,N))
    for f in frames:
        C=f['C'];pc=(world-C[:3,3])@C[:3,:3];z=pc[:,:,2]
        uv=pc[:,:,:2]/np.maximum(z[:,:,None],1e-8)*[f['K'][0,0],f['K'][1,1]]+f['K'][:2,2]
        xy=np.rint(uv).astype(int);h,w=f['depth'].shape
        ok=(z>0)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
        ii,jj=np.nonzero(ok)
        if not len(ii):continue
        x,y=xy[ii,jj].T;valid=f['depth_valid'][y,x];ii,jj,x,y=ii[valid],jj[valid],x[valid],y[valid]
        dz=f['depth'][y,x]-z[ii,jj]
        yes=(abs(dz)<=.015)&(f['anonymous_slot'][y,x]==slot)
        hit[ii[yes],jj[yes]]=True
        update=(dz>.015)&(dz<gap[ii,jj]);bi,ni=ii[update],jj[update]
        gap[bi,ni]=dz[update];normal[bi,ni]=C[:3,2]
        offset[bi,ni]=f['depth'][y[update],x[update]]+C[:3,3]@C[:3,2]-.015
    active=np.isfinite(gap)&~hit
    free=np.where(active,np.minimum(np.maximum(gap-.015,0)/.04,1),0.)
    return dict(hit=hit,free=free,normal=normal,offset=offset,active=active)


def neural_distribution(model,a,b,cfg,mode):
    with torch.inference_mode():
        logq,state=model.distribution(torch.from_numpy(a['desc']).float(),torch.from_numpy(b['desc']).float(),torch.from_numpy(b['area']).float())
        q=logq.exp().numpy();p=state.softmax(1).numpy()
    q=(1-cfg['uniform_floor'])*q+cfg['uniform_floor']*b['area'][None]
    coverage=p[:,2]/np.maximum(p[:,1]+p[:,2],1e-8) if mode=='soft3' else p[:,2]
    return dict(q=q.astype(float),coverage=coverage.astype(float),states=p)


class PairScore:
    def __init__(self,a,b,match,ga,gb,neural,frames_b,slot_b,cfg):
        self.a=a;self.b=b;self.match=match;self.ga=ga;self.gb=gb;self.neural=neural;self.frames=frames_b;self.slot=slot_b;self.cfg=cfg

    def score(self,Ts,methods=METHODS,return_parts=False):
        Ts=np.asarray(Ts);result={m:[] for m in methods};part=[]
        for start in range(0,len(Ts),24):
            T=Ts[start:start+24];B=len(T)
            x=np.einsum('bij,nj->bni',T[:,:3,:3],self.a['points'])+T[:,None,:3,3]
            Ti=inverse(T);y=np.einsum('bij,nj->bni',Ti[:,:3,:3],self.b['points'])+Ti[:,None,:3,3]
            d1=self.b['tree'].query(x.reshape(-1,3))[0].reshape(B,-1);d2=self.a['tree'].query(y.reshape(-1,3))[0].reshape(B,-1)
            g=.5*(np.minimum(d1/.04,1).mean(1)+np.minimum(d2/.04,1).mean(1))
            mm=self.match
            if len(mm['ia']):
                xx=self.a['xyz'][mm['ia']];yy=self.b['xyz'][mm['ib']]
                z=np.einsum('bij,nj->bni',T[:,:3,:3],xx)+T[:,None,:3,3]
                support=np.exp(-.5*np.sum((z-yy[None])**2,axis=2)/.025**2)*np.exp(-.5*(mm['distance']/.5)**2)[None]
                texture=1-np.minimum(1,support.sum(1)/6)
            else:texture=np.ones(B)
            base=g+texture
            if methods==['ordinary']:
                result['ordinary'].append(base)
                if return_parts:part.append(dict(base=base,free=np.zeros(B)))
                continue
            world=np.einsum('bij,nj->bni',T[:,:3,:3],self.ga['xyz'])+T[:,None,:3,3]
            ray=ray_batch(world,self.frames,self.slot);free=np.sum(ray['free']*self.ga['area'][None],axis=1)
            values={'ordinary':base,'ray':base+.25*free};energies={}
            required={'soft3' if m.startswith('soft3') else m for m in methods if m not in ['ordinary','ray']}
            if required:
                aa,bb=self.ga,self.gb
                d2=np.maximum(0.,(world*world).sum(2)[:,:,None]+(bb['xyz']**2).sum(1)[None,None]-2*world@bb['xyz'].T)
                sigma2=np.maximum(self.cfg['sigma_floor_m']**2,aa['sigma'][:,None]**2+bb['sigma'][None]**2)
                lk=-.5*d2/sigma2[None]-1.5*np.log(2*np.pi*sigma2)[None]
                denominator=logsumexp(lk+np.log(bb['area'].clip(1e-30))[None,None],axis=2)
                support=np.exp(-.5*(d2/sigma2[None]).min(2))
            for name,item in self.neural.items():
                if name not in required:continue
                lr=logsumexp(lk+np.log(item['q'].clip(1e-30))[None],axis=2)-denominator
                R=np.exp(lr)
                for method in [name]+(['soft3_no_coverage','soft3_no_visibility'] if name=='soft3' else []):
                    cov=np.ones_like(item['coverage']) if method=='soft3_no_coverage' else item['coverage']
                    visible=1. if method=='soft3_no_visibility' else ray['hit']
                    mix=support*cov[None]*visible
                    nll=-np.sum(np.log(np.maximum(1-mix+mix*R,1e-12))*aa['area'][None],axis=1)
                    values[method]=base+.25*free+self.cfg['evidence_weight']*nll;energies[method]=nll
            for m in methods:result[m].append(values[m])
            if return_parts:part.append(dict(base=base,free=free,**energies))
        out={m:np.concatenate(v) for m,v in result.items()}
        return (out,part) if return_parts else out

    def differentiable(self,T,method,reference):
        """Fixed-neighbor/pixel-sector derivative; actual line search recomputes all."""
        dtype=T.dtype;device=T.device
        def tensor(x):return torch.as_tensor(x,dtype=dtype,device=device)
        pa,pb=tensor(self.a['points']),tensor(self.b['points'])
        x=pa@T[:3,:3].T+T[:3,3];yi=(pb-T[:3,3])@T[:3,:3]
        xx=old.apply(reference,self.a['points']);yy=old.apply(inverse(reference),self.b['points'])
        j=self.b['tree'].query(xx)[1];i=self.a['tree'].query(yy)[1]
        g=.5*(torch.linalg.vector_norm(x-pb[j],dim=1).div(.04).clamp(max=1).mean()+
               torch.linalg.vector_norm(yi-pa[i],dim=1).div(.04).clamp(max=1).mean())
        mm=self.match
        if len(mm['ia']):
            dx=tensor(self.a['xyz'][mm['ia']])@T[:3,:3].T+T[:3,3]-tensor(self.b['xyz'][mm['ib']])
            mass=(torch.exp(-.5*(dx*dx).sum(1)/.025**2)*tensor(np.exp(-.5*(mm['distance']/.5)**2))).sum()
            texture=1-(mass/6).clamp(max=1)
        else:texture=g*0+1
        value=g+texture
        if method=='ordinary':return value
        ga=self.ga;world=tensor(ga['xyz'])@T[:3,:3].T+T[:3,3]
        refworld=old.apply(reference,ga['xyz'])[None];ray=ray_batch(refworld,self.frames,self.slot)
        free=((tensor(ray['offset'][0])-(world*tensor(ray['normal'][0])).sum(1))/.04).clamp(0,1)*tensor(ray['active'][0])
        value=value+.25*(free*tensor(ga['area'])).sum()
        if method=='ray':return value
        name='soft3' if method.startswith('soft3') else method;item=self.neural[name];bb=item['b']
        delta=world[:,None]-tensor(bb['xyz'])[None];d2=(delta*delta).sum(2)
        sigma2=tensor(np.maximum(self.cfg['sigma_floor_m']**2,ga['sigma'][:,None]**2+bb['sigma'][None]**2))
        lk=-.5*d2/sigma2-1.5*torch.log(2*torch.pi*sigma2)
        lr=torch.logsumexp(lk+tensor(np.log(item['q'].clip(1e-30))),1)-torch.logsumexp(lk+tensor(np.log(bb['area'].clip(1e-30)))[None],1)
        support=torch.exp(-.5*(d2/sigma2).min(1).values)
        cov=tensor(np.ones_like(item['coverage']) if method=='soft3_no_coverage' else item['coverage'])
        vis=1. if method=='soft3_no_visibility' else tensor(ray['hit'][0])
        mix=support*cov*vis;ratio=1-mix+mix*lr.exp()
        return value+self.cfg['evidence_weight']*(-torch.log(ratio.clamp_min(1e-12))*tensor(ga['area'])).sum()


def refine_joint(T1,T2,pairs,method,cfg):
    Ts=[T1.copy(),T2.copy()];history=[]
    def value(Ts):
        T12=Ts[1]@inverse(Ts[0])
        return float(pairs[0].score(Ts[0][None],[method])[method][0]+pairs[1].score(Ts[1][None],[method])[method][0]+.5*pairs[2].score(T12[None],[method])[method][0])
    current=value(Ts);history.append(current)
    for _ in range(cfg['continuous_steps']):
        delta=torch.zeros(2,6,dtype=torch.double,requires_grad=True)
        A=torch.zeros(2,4,4,dtype=torch.double)
        for k in range(2):
            v=delta[k];A[k,0,1]=-v[2];A[k,0,2]=v[1];A[k,1,0]=v[2];A[k,1,2]=-v[0];A[k,2,0]=-v[1];A[k,2,1]=v[0];A[k,:3,3]=v[3:]
        TT=torch.matrix_exp(A)@torch.as_tensor(np.array(Ts),dtype=torch.double)
        inv=torch.eye(4,dtype=torch.double);inv[:3,:3]=TT[0,:3,:3].T;inv[:3,3]=-TT[0,:3,:3].T@TT[0,:3,3]
        T12=TT[1]@inv;ref12=Ts[1]@inverse(Ts[0])
        loss=pairs[0].differentiable(TT[0],method,Ts[0])+pairs[1].differentiable(TT[1],method,Ts[1])+.5*pairs[2].differentiable(T12,method,ref12)
        loss.backward();grad=delta.grad.detach().numpy()
        if not np.isfinite(grad).all():raise RuntimeError('Nonfinite joint derivative')
        step=-grad/max(np.linalg.norm(grad),1e-12)*.02;accepted=False
        for alpha in [1.,.5,.25,.125,.0625]:
            dd=torch.as_tensor(step*alpha,dtype=torch.double);AA=torch.zeros(2,4,4,dtype=torch.double)
            for k in range(2):
                v=dd[k];AA[k,0,1]=-v[2];AA[k,0,2]=v[1];AA[k,1,0]=v[2];AA[k,1,2]=-v[0];AA[k,2,0]=-v[1];AA[k,2,1]=v[0];AA[k,:3,3]=v[3:]
            proposal=(torch.matrix_exp(AA)@torch.as_tensor(np.array(Ts),dtype=torch.double)).numpy();new=value(proposal)
            if new<current-1e-10:Ts=list(proposal);current=new;accepted=True;break
        history.append(current)
        if not accepted:break
    return np.array(Ts),np.array(history)
