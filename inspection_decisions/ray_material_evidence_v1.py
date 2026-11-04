"""Historical visibility-conditioned shared-material evidence, with exact ordinary control.

No pose annotations, template meshes, or completed surfaces are accepted here.
The independent-ray factorization is a composite likelihood, not a posterior.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.special import betaln
from scipy.spatial import cKDTree

def occupancy_log_bf(occupied_h,free_h,occupied_c,free_c,alpha=1.,beta=1.):
    """Integrated shared occupancy versus two independent latent occupancies."""
    return (betaln(alpha+occupied_h+occupied_c,beta+free_h+free_c)+betaln(alpha,beta)
            -betaln(alpha+occupied_h,beta+free_h)-betaln(alpha+occupied_c,beta+free_c))

def ray_states(points,depth,mask,K,tolerance,valid_interval):
    """0 unknown/occluded, 1 observed target surface, 2 known target absence.

    Behind any valid measured surface stays unknown. Near non-target surface is
    target absence only because this track supplies privileged target masks.
    """
    z=points[:,2];uv=points@K.T
    uv=np.divide(uv[:,:2],uv[:,2,None],out=np.full((len(z),2),-1e8),where=np.abs(uv[:,2,None])>1e-12)
    u=np.rint(uv[:,0]).astype(np.int64);v=np.rint(uv[:,1]).astype(np.int64)
    inside=(z>0)&(u>=0)&(v>=0)&(u<depth.shape[1])&(v<depth.shape[0]);d=np.zeros(len(z));target=np.zeros(len(z),bool)
    d[inside]=depth[v[inside],u[inside]];target[inside]=mask[v[inside],u[inside]]
    known=inside&(d>=valid_interval[0])&(d<=valid_interval[1]);near=known&(np.abs(z-d)<=tolerance)
    surface=near&target;free=(known&(z<d-tolerance))|(near&~target)
    states=np.zeros(len(z),np.int8);states[surface]=1;states[free]=2
    return states,u,v,np.where(known,z-d,np.nan)

class Frame:
    def __init__(self,root,stem,cfg):
        root=Path(root)/'public';self.depth=np.array(Image.open(root/'depth'/(stem+'.png')),dtype=float)/1000
        self.mask=np.array(Image.open(root/'masks'/(stem+'.png')))>0
        if self.mask.ndim==3:self.mask=self.mask.any(2)
        self.image=np.array(Image.open(root/'rgb'/(stem+'.png')).convert('RGB'),dtype=float)/255
        self.K=np.loadtxt(root/'cam_K.txt').reshape(3,3)
        with np.load(root/'clouds'/(stem+'.npz')) as z:self.xyz=z['xyz'].astype(float);self.rgb=z['rgb'].astype(float)/255
        self.all_xyz=self.xyz.copy();self.tree=cKDTree(self.all_xyz)
        if len(self.xyz)>cfg['max_points_per_frame']:
            ix=np.random.default_rng(cfg['seed']).choice(len(self.xyz),cfg['max_points_per_frame'],replace=False);self.xyz=self.xyz[ix];self.rgb=self.rgb[ix]

class RayPair:
    def __init__(self,history,current,anchor,cfg):
        self.h=history;self.c=current;self.anchor=np.asarray(anchor);self.cfg=cfg
        self.alpha,self.beta=cfg['occupancy_beta_prior'];self.bins=cfg['color_bins_per_channel'];self.categories=self.bins**3;self.gamma=cfg['color_dirichlet_per_category'];self.calls=0
        self.surface_bf=float(occupancy_log_bf(1,0,1,0,self.alpha,self.beta));self.free_bf=float(occupancy_log_bf(0,1,1,0,self.alpha,self.beta))
        self.same_color_bf=np.log(self.categories*(self.gamma+1)/(self.categories*self.gamma+1));self.different_color_bf=np.log(self.categories*self.gamma/(self.categories*self.gamma+1))

    def direction(self,query,rgb,target):
        cfg=self.cfg;state,u,v,residual=ray_states(query,target.depth,target.mask,target.K,cfg['depth_tolerance_m'],cfg['depth_valid_interval_m'])
        surf=state==1;free=state==2;near_rgb=np.zeros_like(rgb);near_rgb[surf]=target.image[v[surf],u[surf]]
        color_error=np.minimum(np.mean(np.abs(near_rgb-rgb),axis=1)/.1,1)
        q1=np.minimum((rgb*self.bins).astype(int),self.bins-1);q2=np.minimum((near_rgb*self.bins).astype(int),self.bins-1);same=(q1==q2).all(1)
        bf=np.where(surf,self.surface_bf,np.where(free,self.free_bf,0.))
        # For a single endpoint each, the beta-binomial marginal reduces exactly
        # to weighted counts. The ordinary equivalent below must match this.
        material=np.where(surf,np.where(same,self.same_color_bf,self.different_color_bf),0.)
        valid=surf|free;depth_loss=np.minimum(np.nan_to_num(np.abs(residual)/cfg['depth_tolerance_m'],nan=0),1)**2
        return dict(surface=float(surf.mean()),free=float(free.mean()),unknown=float((state==0).mean()),
                    same_color_surface=float((surf&same).mean()),different_color_surface=float((surf&~same).mean()),
                    censored_depth=float(np.mean(np.where(valid,depth_loss,0))),
                    censored_rgb=float(np.mean(np.where(surf,.5*depth_loss+.5*color_error,np.where(free,1,0)))),
                    ordinary_overlap_rgb=float(np.mean(np.where(surf,color_error-1,np.where(free,1,0)))),
                    occupancy_log_bf=float(bf.mean()),material_log_bf=float((bf+material).mean()))

    def score(self,T):
        self.calls+=1;R=T[:3,:3];t=T[:3,3]
        a=self.direction((self.c.xyz-t)@R,self.c.rgb,self.h);b=self.direction(self.h.xyz@R.T+t,self.h.rgb,self.c);d={k:(a[k]+b[k])/2 for k in a}
        # Identical old one-sided geometry, sampled for this new comparison only.
        distance=self.h.tree.query((self.c.xyz-t)@R,workers=1)[0];nn=.6*np.mean(np.minimum(distance/.075,1)**2);motion=.2*min(np.linalg.norm(R@self.anchor+t-self.anchor),1.)
        equivalent=-(d['surface']*self.surface_bf+d['free']*self.free_bf+d['same_color_surface']*self.same_color_bf+d['different_color_surface']*self.different_color_bf)
        scores=dict(old_nn=float(nn+motion),old_nn_no_motion=float(nn),ordinary_censored_depth=d['censored_depth'],ordinary_censored_rgb=d['censored_rgb'],ordinary_overlap_rgb=d['ordinary_overlap_rgb'],beta_geometry=-d['occupancy_log_bf'],beta_material=-d['material_log_bf'],ordinary_exact_beta_equivalent=float(equivalent))
        return scores,d
