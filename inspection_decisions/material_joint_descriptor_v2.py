"""Proper factorization of visibility, material assignment and appearance state.

This repairs V1's incompatible interpretation of one matchability logit; the
factorization and balanced classification are standard, not claimed originality.
"""
import numpy as np
import torch
from torch.nn import functional as F
from inspection_decisions.material_joint_descriptor_v1 import MaterialDescriptor


class FactoredMaterialDescriptor(MaterialDescriptor):
    def __init__(self,dim=64,persistence_prior=.5,balanced=True):
        super().__init__(dim);self.persistence_prior=float(persistence_prior);self.balanced=balanced

    def persistence_evidence_logits(self,a,b):
        return super().persistence_logits(a,b)

    def persistence_logits(self,a,b):
        raw=self.persistence_evidence_logits(a,b)
        return raw+np.log(self.persistence_prior/(1-self.persistence_prior)) if self.balanced else raw


def log_joint_distribution(similarity,matchability):
    conditional=F.log_softmax(similarity,dim=1)
    matched=F.logsigmoid(matchability)[:,None]+conditional
    unmatched=F.logsigmoid(-matchability)[:,None]
    return torch.cat([matched,unmatched],dim=1)


def factored_loss(model,source,target,visible,positive,changed,near,temperature=.12,head_weight=.25):
    a,b=model(source),model(target);sim=a@b.T/temperature
    logits=sim.masked_fill(near,-1e4);ii=torch.nonzero(visible,as_tuple=True)[0]
    if len(ii):logits[ii,positive[ii]]=sim[ii,positive[ii]]
    m=model.context_logits(a,b);joint=log_joint_distribution(logits,m)
    y=torch.where(visible,positive,torch.full_like(positive,len(b)))
    physical=F.nll_loss(joint,y)
    visibility=F.binary_cross_entropy_with_logits(m,visible.float())
    corr=F.cross_entropy(logits[ii],positive[ii]) if len(ii) else a.sum()*0
    if len(ii):
        c=model.persistence_evidence_logits(a[ii],b[positive[ii]])
        label=(~changed[ii]).float()
        weights=(label/(2*model.persistence_prior)+(1-label)/(2*(1-model.persistence_prior))) if model.balanced else torch.ones_like(label)
        persist=(F.binary_cross_entropy_with_logits(c,label,reduction='none')*weights).mean()
    else:persist=physical*0
    total=physical+head_weight*persist
    with torch.no_grad():
        accuracy=(sim[ii].argmax(1)==positive[ii]).float().mean() if len(ii) else sim.new_tensor(0.)
    return total,dict(loss=float(total.detach()),physical=float(physical.detach()),
        correspondence=float(corr.detach()),matchability=float(visibility.detach()),
        persistence=float(persist.detach()),retrieval=float(accuracy))
