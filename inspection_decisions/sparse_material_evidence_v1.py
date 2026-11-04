"""Finite HISTORY-ONLY material quadrature and two-period Gibbs evidence.

Fixed reference graph; no integration over graph births or calibrated sensor
posterior claim. Every state evaluates the same actual pixel once per period.
"""
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment,linprog,minimize
from scipy.sparse import coo_matrix
from inspection_decisions.typed_material_incidence_graph_v1 import build_typed_graph

DT=torch.float64


def material_states(history,cfg,seed):
    _,graph=build_typed_graph([(np.zeros(6),history)],cfg);K=cfg['material_states'];F=len(graph['faces']);U=qmc.Sobol(max(F,1),scramble=True,seed=seed).random_base2(int(np.log2(K)));states=np.zeros((K,F))
    for j,f in enumerate(graph['faces']):
        bands=np.array(f['historical_intervals'] or [[f['offset']-cfg['face_mode_radius_m'],f['offset']+cfg['face_mode_radius_m']]],float);length=bands[:,1]-bands[:,0]
        if length.sum()>1e-14:
            pos=U[:,j]*length.sum();ends=np.cumsum(length);ix=np.minimum(np.searchsorted(ends,pos,side='right'),len(ends)-1);states[:,j]=bands[ix,0]+pos-np.r_[0.,ends[:-1]][ix]
        else:states[:,j]=bands[np.minimum((U[:,j]*len(bands)).astype(int),len(bands)-1),0]
    return graph,states


def rotation(w):
    z=w[0]*0;W=torch.stack([z,-w[2],w[1],w[2],z,-w[0],-w[1],w[0],z]).reshape(3,3);theta=torch.linalg.vector_norm(w);return torch.eye(3,dtype=DT)+torch.sinc(theta/torch.pi)*W+.5*torch.sinc(theta/(2*torch.pi))**2*(W@W)


class PeriodModel:
    def __init__(self,data,graph,cfg):
        self.cfg=cfg;self.N=len(data['pixels']);self.M=len(data['boundary']);self.F=len(graph['faces']);pp=data['pixels'];bb=data['boundary'];N=self.N
        def tensor(x,dtype=DT):return torch.tensor(np.asarray(x),dtype=dtype)
        self.O=tensor([p['origin'] for p in pp]).reshape(-1,3);self.ri=tensor([p['ray'] for p in pp]).reshape(-1,3);self.z=tensor([p['depth'] for p in pp]);normal=np.zeros((N,4,3));sigma=np.full((N,4),np.inf)
        for j,p in enumerate(pp):n=len(p['normals']);normal[j,:n]=p['normals'];sigma[j,:n]=p['normal_sigma']
        self.normal=tensor(normal);self.sigma=tensor(sigma);self.missing=tensor((~np.isfinite(sigma))&np.isfinite(sigma).any(1)[:,None],torch.bool);self.pi=tensor([b['pixel_index'] for b in bb],torch.long);self.ro=tensor([b['ray_outside'] for b in bb]).reshape(-1,3);self.outside=tensor([b['outside_depth'] for b in bb]);self.axes=tensor([f['axis'] for f in graph['faces']],torch.long);self.termination=tensor([f.get('birth_role')=='termination_supported_not_source_observed' for f in graph['faces']],torch.bool)
        edges=[(f,g,e['span_axis'],*e['span']) for e in graph['edges'] for f,g in [e['faces'],e['faces'][::-1]]];E=np.array(edges).reshape(-1,5);self.f=tensor(E[:,0],torch.long);self.g=tensor(E[:,1],torch.long);self.h=tensor(E[:,2],torch.long);self.lo=tensor(E[:,3]);self.hi=tensor(E[:,4]);self.D=len(edges)

    def evaluate(self,x,d,modes=('ray',),used=False):
        cfg=self.cfg;K=len(d);cap2=cfg['support_match_cap_m']**2;N=self.N;M=self.M;F=self.F
        if not F:
            v=x.sum()*0+torch.full((K,),(cap2/3 if N else 0)+(cap2/3 if M else 0),dtype=DT);return {mode:(v,torch.zeros((K,F),dtype=torch.bool)) for mode in modes}
        R=rotation(x[:3]);O=(self.O-x[3:])@R;ri=self.ri@R;ro=self.ro@R;Q=O+ri*self.z[:,None];dot=self.normal@R;ncost=torch.clamp(1-dot**2,min=0)*cfg['normal_residual_scale_m']**2/self.sigma[:,:,None]**2;ncost=torch.where(self.missing[:,:,None],torch.inf,ncost);ncost=ncost.min(1).values
        source=torch.clamp((Q[:,self.axes][None]-d[:,None,:])**2+ncost[:,self.axes][None],max=cap2);bc=None;valid=None
        if M and self.D:
            a=self.axes[self.f];k=self.axes[self.g];B=ro[:,a];A=ri[self.pi][:,a];stable=(abs(B)>cfg['denominator_tolerance'])&(abs(A)>cfg['denominator_tolerance']);safe=torch.where(stable,B,torch.ones_like(B));lam=(d[:,self.f][:,None,:]-O[self.pi][:,a][None])/safe[None];ratio=A/safe;step=ratio*ro[:,k]-ri[self.pi][:,k];s=torch.sign(step);valid=stable[None]&(ratio[None]>0)&(lam>0)&(abs(step)[None]>cfg['denominator_tolerance']);outside=O[self.pi][:,k][None]+ro[:,k][None]*lam;inside=Q[self.pi][:,k][None];along=Q[self.pi][:,self.h];margin=cfg['edge_span_margin_m'];common=B[None]**2*torch.relu(lam-self.outside[None,:,None]+cfg['depth_noise_bound_m'])**2+torch.relu(self.lo[None]-margin-along)[None]**2+torch.relu(along-self.hi[None]-margin)[None]**2
        result={}
        for mode in modes:
            best=torch.full((K,M,F),cap2,dtype=DT);vc=None
            if M and self.D:
                v=(torch.relu(s[None]*(inside-d[:,self.g][:,None,:]))**2+torch.relu(s[None]*(d[:,self.g][:,None,:]-outside))**2) if mode=='ray' else ((inside+outside)/2-d[:,self.g][:,None,:])**2
                vc=torch.where(valid,torch.clamp(v+common,max=cap2),cap2);best=best.scatter_reduce(2,self.f[None,None,:].expand(K,M,-1),vc,reduce='amin',include_self=True)
            total=source/(3*max(N,1))
            if M:total=total.scatter_add(1,self.pi[None,:,None].expand(K,M,F),best/(3*M))
            values,face=total.min(2);energy=values.sum(1);mask=torch.zeros((K,F),dtype=torch.bool)
            if used and N:
                active=source.gather(2,face[:,:,None]).squeeze(2)<cap2-1e-14;mask.scatter_reduce_(1,face,active,reduce='amax',include_self=True)
                if M and self.D:
                    sf=face[:,self.pi];allowed=self.f[None,None,:]==sf[:,:,None];vv=torch.where(allowed,vc,torch.inf);bv,choice=vv.min(2);active=bv<cap2-1e-14;mask.scatter_reduce_(1,self.g[choice],active,reduce='amax',include_self=True);mask.scatter_reduce_(1,sf,active,reduce='amax',include_self=True)
            result[mode]=(energy,mask&self.termination[None])
        return result


def log_weights(history_energy,tau):
    a=-np.asarray(history_energy)/tau;return a-logsumexp(a)


def marginal_cost(E,logw,tau):return -tau*logsumexp(logw[None]-np.asarray(E)/tau,axis=1)


def pair_costs(E1,E2,logw,tau,method,used1=None,used2=None):
    if method=='independent_ray':return marginal_cost(E1,logw,tau)[:,None]+marginal_cost(E2,logw,tau)[None]
    if method in ['joint_ray','joint_line']:
        a=E1.min(1);b=E2.min(1);A=np.exp(-(E1-a[:,None])/tau+logw[None]/2);B=np.exp(-(E2-b[:,None])/tau+logw[None]/2);return a[:,None]+b[None]-tau*np.log(np.maximum(A@B.T,np.finfo(float).tiny))
    result=np.full((len(E1),len(E2)),np.inf)
    for l in range(E1.shape[1]):
        value=E1[:,l,None]+E2[None,:,l]-tau*logw[l]
        if method=='profile_mdl_ray':value=value+tau*np.log(2)*np.logical_or(used1[:,None,l,:],used2[None,:,l,:]).sum(2)
        result=np.minimum(result,value)
    return result


def reduce_slots(cost,pools):
    slots=[np.array([b['slot'] for b in p]) for p in pools];C=np.zeros((24,24));indices=np.zeros((24,24,2),int)
    for j in range(24):
        aa=np.flatnonzero(slots[0]==j)
        for k in range(24):
            bb=np.flatnonzero(slots[1]==k);v=cost[np.ix_(aa,bb)];a,b=np.unravel_index(v.argmin(),v.shape);C[j,k]=v[a,b];indices[j,k]=[aa[a],bb[b]]
    return C,indices


def assign(C,starts,max_rounds=24):
    n=len(C);best=None;traces=[]
    for a,b in starts:
        a=np.array(a,int);b=np.array(b,int);assert sorted(a)==sorted(b)==list(range(n));value=float(C[np.arange(n),a,b].sum());trace=[value]
        for _ in range(max_rounds):
            _,aa=linear_sum_assignment(C[np.arange(n),:,b]);_,bb=linear_sum_assignment(C[np.arange(n),aa,:]);new=float(C[np.arange(n),aa,bb].sum());assert new<=value+1e-11
            if new>=value-1e-12:break
            a,b,value=aa,bb,new;trace.append(value)
        if best is None or value<best[0]-1e-12:best=(value,a.copy(),b.copy())
        traces.append(trace)
    # Axial 3-index assignment LP bound; fractional optimum is never rounded silently.
    i,j,k=np.indices(C.shape);col=np.arange(C.size);A=coo_matrix((np.ones(3*C.size),(np.r_[i.ravel(),n+j.ravel(),2*n+k.ravel()],np.tile(col,3))),shape=(3*n,C.size)).tocsr();lp=linprog(C.ravel(),A_eq=A,b_eq=np.ones(3*n),bounds=(0,None),method='highs',options={'time_limit':10})
    lower=float(lp.fun) if lp.success else None;integral=False
    if lp.success and np.max(abs(lp.x-np.round(lp.x)))<1e-7:
        ii,jj,kk=np.unravel_index(np.flatnonzero(lp.x>.5),C.shape);assert np.array_equal(ii,np.arange(n));integral=True;value=float(C[ii,jj,kk].sum())
        if value<best[0]:best=(value,jj,kk)
    return best[1],best[2],{'energy':best[0],'LP_lower_bound':lower,'LP_integral':integral,'gap':best[0]-lower if lower is not None else None,'starts':len(starts),'traces':traces,'global_optimum_certified':bool(lower is not None and best[0]-lower<1e-10)}


def evidence_tensor(E1,E2,logw,tau,method,U1=None,U2=None):
    if method=='independent_ray':return -tau*(torch.logsumexp(logw-E1/tau,0)+torch.logsumexp(logw-E2/tau,0))
    e=E1+E2-tau*logw
    if method=='profile_mdl_ray':e=e+tau*np.log(2)*torch.logical_or(U1,U2).sum(1)
    return -tau*torch.logsumexp(-e/tau,0) if method in ['joint_ray','joint_line'] else e.min()


def refine(data,graph,states,logw,initial,limits,cfg,method):
    mode='line' if method=='joint_line' else 'ray';models=[PeriodModel(d,graph,cfg) for d in data];D=torch.tensor(states,dtype=DT);W=torch.tensor(logw,dtype=DT);tau=cfg['evidence_temperature'];used=method=='profile_mdl_ray';trace=[]
    def fun(x):
        p=torch.tensor(x,dtype=DT,requires_grad=True);a,u=models[0].evaluate(p[:6],D,(mode,),used)[mode];b,v=models[1].evaluate(p[6:],D,(mode,),used)[mode];cost=evidence_tensor(a,b,W,tau,method,u,v)/tau;cost.backward();return float(cost.detach()),p.grad.detach().numpy()
    def constraints(x):return np.array([limit**2-np.dot(x[j+off:j+off+3],x[j+off:j+off+3]) for j in [0,6] for off,limit in [(0,limits[0]),(3,limits[1])]])
    def cj(x):
        J=np.zeros((4,12))
        for row,(j,off) in enumerate([(0,0),(0,3),(6,0),(6,3)]):J[row,j+off:j+off+3]=-2*x[j+off:j+off+3]
        return J
    initial=np.array(initial,float);start,_=fun(initial)
    def callback(x):trace.append(float(fun(x)[0]*tau))
    result=minimize(fun,initial,method='SLSQP',jac=True,constraints={'type':'ineq','fun':constraints,'jac':cj},callback=callback,options={'maxiter':cfg['continuous_iterations'],'ftol':cfg['descent_tolerance']/tau,'disp':False});x=result.x;value,_=fun(x);accepted=bool(np.isfinite(value) and np.isfinite(x).all() and constraints(x).min()>=-1e-10 and value<start)
    return (x if accepted else initial),{'initial_cost':start*tau,'final_cost':value*tau if accepted else start*tau,'accepted':accepted,'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'iterations':int(result.nit),'evaluations':int(result.nfev),'callback_costs':trace,'model':'finite_H_only_Gibbs_not_complete_sensor_Bayes'}
