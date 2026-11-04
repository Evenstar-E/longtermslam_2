"""Probability, quadrature, gradient-separation and state semantics checks."""
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions.material_correspondence_distribution_v3 import CorrespondenceModel,conditional_evidence_numpy,episode_loss


def main():
    torch.set_num_threads(2);torch.manual_seed(7);cfg=json.loads(Path('configs/material_correspondence_distribution_v3.json').read_text())
    model=CorrespondenceModel();a=torch.nn.functional.normalize(torch.randn(6,64),dim=1);b=torch.nn.functional.normalize(torch.randn(9,64),dim=1);area=torch.rand(9);area/=area.sum()
    logq,s=model.distribution(a,b,area);assert torch.max(abs(logq.exp().sum(1)-1))<1e-6
    duplicated=torch.cat([b,b[:1]]);area2=torch.cat([area.clone(),area[:1]/2]);area2[0]/=2
    logq2,s2=model.distribution(a,duplicated,area2)
    assert torch.max(abs(s-s2))<2e-6
    q=logq.exp().detach().numpy();q2=logq2.exp().detach().numpy()
    assert np.max(abs(q[:,0]-q2[:,0]-q2[:,-1]))<1e-6
    rng=np.random.default_rng(4);x=rng.normal(size=(6,3));y=rng.normal(size=(9,3))
    src=dict(xyz=x,sigma=np.full(6,.015));dst=dict(xyz=y,sigma=np.full(9,.015),area=area.numpy())
    T=np.tile(np.eye(4),(3,1,1));T[1,:3,3]=.1;T[2,:3,3]=-.1
    neutral=conditional_evidence_numpy(T,src,dst,np.tile(area.numpy(),(6,1)),np.ones(6),cfg)
    assert np.max(abs(neutral))<1e-7
    dst2=dict(xyz=np.r_[y,y[:1]],sigma=np.full(10,.015),area=area2.numpy())
    e1=conditional_evidence_numpy(T,src,dst,q,np.ones(6),cfg);e2=conditional_evidence_numpy(T,src,dst2,q2,np.ones(6),cfg)
    assert np.max(abs(e1-e2))<1e-6
    source=torch.randn(6,10,12,12);target=torch.randn(9,10,12,12)
    aa=model.encode_appearance(source);bb=model.encode_appearance(target[:6]);model.change_logits(aa,bb).sum().backward()
    assert all(p.grad is None for p in model.rgb.parameters()) and all(p.grad is None for p in model.geometry.parameters())
    model.zero_grad(set_to_none=True)
    states=torch.tensor([0,1,2,2,1,0]);teacher=torch.zeros(6,9);teacher[2,2]=1;teacher[3,3]=1;change=torch.tensor([0,0,0,1,0,0],dtype=torch.bool)
    loss,stat=episode_loss(model,source,target,area,states,teacher,change,'soft3',cfg);loss.backward()
    assert np.isfinite(list(stat.values())).all()
    print('PASS normalized correspondence, split-area invariance, uniform neutrality, appearance-gradient isolation, 3-state finite loss',flush=True)


if __name__=='__main__':main()
