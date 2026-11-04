"""Deployment-matched, measure-aware material correspondence distributions."""
import copy
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from inspection_decisions.material_joint_descriptor_v1 import MaterialDescriptor,metric_patches,sample_frame
from inspection_decisions.textured_material_matching_v1 import world_points


def area_weights(xyz):
    if len(xyz)<2:return np.ones(len(xyz),np.float32)
    d=cKDTree(xyz).query(xyz,k=min(5,len(xyz)))[0]
    raw=np.maximum(d[:,-1]**2,1e-6)
    raw=np.clip(raw,np.median(raw)*.1,np.median(raw)*10.)
    return (raw/raw.sum()).astype(np.float32)


def observed_groups(frames,cfg):
    """Keep repeatable interest points and add measured support, without labels."""
    sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.012)
    buckets=[[] for _ in range(24)];coverage=[[] for _ in range(24)]
    counts=np.zeros(24,int)
    for frame in frames:
        for i in np.unique(frame['anonymous_slot']):
            if i>=0:counts[int(i)]+=1
    for fi,frame in enumerate(frames):
        slots=frame['anonymous_slot'];mask=((slots>=0)&frame['depth_valid']).astype(np.uint8)*255
        kp,desc=sift.detectAndCompute(cv2.cvtColor(frame['rgb'],cv2.COLOR_RGB2GRAY),mask)
        if desc is not None:
            uv=np.rint(np.array([k.pt for k in kp])).astype(int)
            uv[:,0]=np.clip(uv[:,0],0,slots.shape[1]-1);uv[:,1]=np.clip(uv[:,1],0,slots.shape[0]-1)
            xyz=world_points(frame,uv);ids=slots[uv[:,1],uv[:,0]]
            for k,slot in enumerate(ids):
                if slot>=0 and frame['depth_valid'][uv[k,1],uv[k,0]]:
                    buckets[int(slot)].append((float(kp[k].response),fi,uv[k],xyz[k]))
        for slot,(uv,xyz) in sample_frame(frame,cfg['coverage_features']).items():
            count=max(1,cfg['coverage_features']//max(counts[slot],1))
            ix=np.linspace(0,len(uv)-1,min(count,len(uv)),dtype=int)
            coverage[slot].extend([(0.,fi,uv[k],xyz[k]) for k in ix])
    selected=[]
    for slot in range(24):
        seen=set();out=[]
        for records,cap in [(sorted(buckets[slot],key=lambda r:-r[0]),cfg['sift_features']),
                            (coverage[slot],cfg['coverage_features'])]:
            added=0
            for rec in records:
                key=tuple(np.floor(rec[3]/cfg['voxel_m']).astype(int))
                if key in seen:continue
                seen.add(key);out.append(rec);added+=1
                if added>=cap:break
        selected.append(out)
    result=[dict(uv=[],frame=[],xyz=[],sigma=[],patch=[]) for _ in range(24)]
    for fi,frame in enumerate(frames):
        records=[(slot,k,r) for slot,values in enumerate(selected) for k,r in enumerate(values) if r[1]==fi]
        if not records:continue
        uv=np.array([r[2][2] for r in records]);patch,xyz=metric_patches(frame,uv,cfg)
        px=np.rint(uv).astype(int);z=frame['depth'][px[:,1],px[:,0]]
        sigma=np.sqrt((z/frame['K'][0,0])**2/12+(z/frame['K'][1,1])**2/12+.002**2)
        for j,(slot,k,r) in enumerate(records):
            out=result[slot]
            for name,value in [('uv',uv[j]),('frame',fi),('xyz',xyz[j]),('sigma',sigma[j]),('patch',patch[j])]:out[name].append(value)
    for out in result:
        for key in out:out[key]=np.asarray(out[key])
        out['patch']=out['patch'].astype(np.float32);out['xyz']=out['xyz'].astype(float)
        out['area']=area_weights(out['xyz'])
    return result


class CorrespondenceModel(nn.Module):
    def __init__(self,dim=64,temperature=.12):
        super().__init__();prototype=MaterialDescriptor(dim)
        self.rgb=prototype.rgb;self.geometry=prototype.geometry;self.fusion=prototype.fusion
        self.appearance=copy.deepcopy(prototype.rgb)
        self.state=nn.Sequential(nn.Linear(3*dim+3,64),nn.SiLU(),nn.Linear(64,3))
        self.change=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        self.temperature=temperature

    def encode(self,x):
        return F.normalize(self.fusion(torch.cat([self.rgb(x[:,:6]),self.geometry(x[:,6:])],1)),dim=1)

    def encode_appearance(self,x):
        return self.appearance(x[:,:6])

    def forward(self,x):return self.encode(x)

    def distribution(self,a,b,area):
        mu=area/area.sum();sim=a@b.T/self.temperature
        logq=F.log_softmax(sim+mu.clamp_min(1e-12).log()[None],dim=1);q=logq.exp()
        mean=q@b;average=(q*sim).sum(1)
        kl=(q*(logq-mu.clamp_min(1e-12).log()[None])).sum(1)
        var=(q*(sim-average[:,None])**2).sum(1)
        context=torch.cat([a,mean,a*mean,torch.stack([average,kl,var],1)],1)
        return logq,self.state(context)

    def change_logits(self,a,b):
        return self.change(torch.cat([abs(a-b),a*b],1)).squeeze(1)

    def warm_start(self,state):
        for name in ['rgb','geometry','fusion']:
            getattr(self,name).load_state_dict({k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')})
        self.appearance.load_state_dict(self.rgb.state_dict())


def episode_loss(model,source,target,area,states,teacher,changed,mode,cfg):
    a,b=model.encode(source),model.encode(target);logq,state_logits=model.distribution(a,b,area)
    target_states=states if mode=='soft3' else torch.where(states==1,torch.zeros_like(states),states)
    state_loss=F.cross_entropy(state_logits,target_states)
    covered=states==2
    if covered.any():
        label=teacher[covered]
        if mode=='hard2':label=F.one_hot(label.argmax(1),num_classes=len(target)).float()
        correspondence=-(label*logq[covered]).sum(1).mean()
    else:correspondence=a.sum()*0
    # Separate appearance path: appearance classification never changes the
    # physical encoder, including its RGB branch.
    if covered.any():
        ia=torch.nonzero(covered,as_tuple=True)[0];ib=teacher[covered].argmax(1)
        aa=model.encode_appearance(source[ia]);bb=model.encode_appearance(target[ib])
        c=model.change_logits(aa,bb);label=changed[ia].float()
        # Per-episode balanced risk where both classes exist; priors/decision
        # calibration are explicitly evaluated after training.
        if label.min()!=label.max():weights=torch.where(label>0,.5/label.mean(),.5/(1-label.mean()))
        else:weights=torch.ones_like(label)
        appearance=(F.binary_cross_entropy_with_logits(c,label,reduction='none')*weights).mean()
    else:appearance=a.sum()*0
    loss=correspondence+state_loss+cfg['appearance_weight']*appearance
    return loss,dict(loss=float(loss.detach()),correspondence=float(correspondence.detach()),
        state=float(state_loss.detach()),appearance=float(appearance.detach()),covered=int(covered.sum()))


def conditional_evidence_numpy(Ts,a,b,q,covered,cfg,visible=None):
    """Marginal spatial likelihood ratio to uniform material AREA measure.

All target samples contribute, never only the nearest match. Uniform appearance
information gives exactly neutral evidence for every pose and sampling density.
"""
    Ts=np.asarray(Ts);x=np.einsum('bij,nj->bni',Ts[:,:3,:3],a['xyz'])+Ts[:,None,:3,3]
    d2=np.maximum(0.,(x*x).sum(2)[:,:,None]+(b['xyz']**2).sum(1)[None,None]-2*x@b['xyz'].T)
    sigma2=np.maximum(cfg['sigma_floor_m']**2,a['sigma'][:,None]**2+b['sigma'][None]**2)
    K=np.exp(-.5*d2/sigma2[None])
    num=np.sum(K*q[None],axis=2);den=np.sum(K*b['area'][None,None],axis=2)
    ratio=np.ones_like(den);np.divide(num,den,out=ratio,where=den>1e-12)
    support=K.max(2)*covered[None]
    if visible is not None:support*=visible
    evidence=1-support+support*ratio
    return -np.log(np.maximum(evidence,1e-12)).mean(1)
