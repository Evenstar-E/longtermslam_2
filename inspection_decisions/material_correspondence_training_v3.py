"""Class-balanced appearance supervision isolated from physical gradients."""
import torch
from torch.nn import functional as F
from inspection_decisions.material_correspondence_distribution_v3 import episode_loss


def balanced_loss(model,source,target,area,states,teacher,changed,mode,cfg):
    physical,stats=episode_loss(model,source,target,area,states,teacher,changed,mode,{**cfg,'appearance_weight':0.})
    ii=torch.nonzero(states==2,as_tuple=True)[0]
    if len(ii):
        ib=teacher[ii].argmax(1)
        a=model.encode_appearance(source[ii]);b=model.encode_appearance(target[ib]);logit=model.change_logits(a,b)
        label=changed[ii].float();pi=cfg['appearance_positive_prior']
        weight=torch.where(label>0,.5/pi,.5/(1-pi))
        app=(F.binary_cross_entropy_with_logits(logit,label,reduction='none')*weight).mean()
    else:app=physical*0
    loss=physical+cfg['appearance_weight']*app
    stats['loss']=float(loss.detach());stats['appearance']=float(app.detach())
    return loss,stats
