"""Measured-footprint common-material evidence; no private labels/file access."""
import numpy as np
from scipy.special import softmax
from inspection_decisions.material_correspondence_inference_v3 import ray_batch
from inspection_decisions.material_correspondence_distribution_v3 import area_weights
from inspection_decisions.material_joint_descriptor_v1 import farthest_indices


def measured_bank(group,grid,features,cfg):
    """Merge redundant voxel witnesses; representative XYZ is an actual sample."""
    xyz=grid['xyz'][grid['valid']];desc=features[grid['valid']]
    sigma=np.broadcast_to(group['sigma'][:,None],grid['valid'].shape)[grid['valid']]
    key=np.floor(xyz/cfg['bank_voxel_m']).astype(np.int64);_,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True)
    sums=np.zeros((len(first),desc.shape[1]));np.add.at(sums,inv,desc);sums/=np.maximum(np.linalg.norm(sums,axis=1,keepdims=True),1e-12)
    take=farthest_indices(xyz[first],cfg['bank_points']);ix=first[take]
    return dict(xyz=xyz[ix],desc=sums[take],sigma=sigma[ix],area=area_weights(xyz[ix]),raw_witnesses=len(xyz))


def pixel_footprint(group,frames,cfg):
    ii=np.linspace(0,len(group['xyz'])-1,min(cfg['queries'],len(group['xyz'])),dtype=int);xyz=group['xyz'][ii]
    signs=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
    nodes=[]
    for index,point in zip(ii,xyz):
        f=frames[int(group['frame'][index])];z=((point-f['C'][:3,3])@f['C'][:3,:3])[2]
        scale=np.array([z/f['K'][0,0]/np.sqrt(12),z/f['K'][1,1]/np.sqrt(12),cfg['depth_noise_bound_m']/np.sqrt(3)])
        nodes.append(point+(signs*scale)@f['C'][:3,:3].T)
    return dict(indices=ii,xyz=xyz,nodes=np.array(nodes),node_weights=np.full(8,1/8),sigma=group['sigma'][ii],area=area_weights(xyz))


class MaterialView:
    def __init__(self,source,source_desc,bank,frames,slot,cfg):
        self.source=source;self.bank=bank;self.frames=frames;self.slot=slot;self.cfg=cfg
        mu=bank['area'];q=softmax(source_desc@bank['desc'].T/.12+np.log(mu)[None],axis=1)
        self.q=(1-cfg['uniform_floor'])*q+cfg['uniform_floor']*mu[None]

    def likelihood(self,poses,chunk=8):
        a,b=self.source,self.bank;Q,S=a['nodes'].shape[:2];results=[]
        # Source uncertainty is already integrated by its actual pixel/depth
        # cubature. Adding it again here would double-count that uncertainty.
        var=np.broadcast_to(np.maximum(self.cfg['sigma_floor_m']**2,b['sigma'][None]**2),(Q,len(b['xyz'])))
        for start in range(0,len(poses),chunk):
            T=poses[start:start+chunk];x=np.einsum('bij,nsj->bnsi',T[:,:3,:3],a['nodes'])+T[:,None,None,:3,3]
            d2=np.maximum(0,(x*x).sum(3)[:,:,:,None]+(b['xyz']**2).sum(1)[None,None,None]-2*x@b['xyz'].T)
            exponent=-.5*d2/var[None,:,None];lk=exponent-1.5*np.log(2*np.pi*var)[None,:,None]
            kernel=np.exp(lk-lk.max(3,keepdims=True));ratio=(kernel*self.q[None,:,None]).sum(3)/np.maximum((kernel*b['area'][None,None,None]).sum(3),1e-300)
            support=np.exp(exponent.max(3));hit=ray_batch(x.reshape(len(T),Q*S,3),self.frames,self.slot)['hit'].reshape(len(T),Q,S)
            results.append(1+hit*support*(ratio-1))
        out=np.concatenate(results);assert np.isfinite(out).all() and out.min()>=self.cfg['uniform_floor']-1e-10
        return out


def material_energies(a,b,node_weight,source_weight):
    """Return ordinary expected-log, independent-marginal, shared-marginal.

    a[P,Q,S], b[R,Q,S]; every method uses identical observations/quadrature.
    """
    wa=np.asarray(node_weight);alpha=np.asarray(source_weight)
    point_a=-(np.log(a)*wa[None,None]*alpha[None,:,None]).sum((1,2));point_b=-(np.log(b)*wa[None,None]*alpha[None,:,None]).sum((1,2))
    za=a@wa;zb=b@wa;independent_a=-(np.log(za)*alpha[None]).sum(1);independent_b=-(np.log(zb)*alpha[None]).sum(1)
    joint=np.einsum('pqs,rqs,s->prq',a,b,wa,optimize=True)
    shared=-(np.log(joint)*alpha[None,None]).sum(2)
    return point_a[:,None]+point_b[None],independent_a[:,None]+independent_b[None],shared
