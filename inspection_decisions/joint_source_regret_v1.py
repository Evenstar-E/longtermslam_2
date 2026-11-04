"""Joint multi-view ICP proposals and coherent source-deletion regret selection."""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.real_temporal_v1 import moved
from inspection_decisions.ray_material_evidence_v1 import RayPair


def rigid_fit(x,y,w):
    w=w/max(w.sum(),1e-300);a=np.sum(w[:,None]*x,axis=0);b=np.sum(w[:,None]*y,axis=0)
    U,_,Vt=np.linalg.svd((x-a).T@(w[:,None]*(y-b)))
    D=np.eye(3);D[2,2]=np.linalg.det(Vt.T@U.T);R=Vt.T@D@U.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=b-R@a;return T


def joint_icp(frames,start,cfg,use_color=False):
    """Q_v maps view v into historical coordinates, Q_0 fixed to identity.

    Each conditional update reconstructs the shared surface from OTHER views.
    This is ordinary alternating multi-view ICP, not claimed as novel.
    """
    Q=[q.copy() for q in start];clouds=[];colors=[];rows=[]
    for frame in frames:
        ids=np.linspace(0,len(frame.xyz)-1,min(len(frame.xyz),cfg['solver_points_per_view'])).astype(int)
        clouds.append(frame.xyz[ids]);colors.append(frame.rgb[ids])
    for sweep in range(cfg['solver_sweeps']):
        support=[]
        for v in range(1,len(frames)):
            shared=np.concatenate([moved(clouds[j],Q[j]) for j in range(len(frames)) if j!=v])
            rgb=np.concatenate([colors[j] for j in range(len(frames)) if j!=v])
            z=moved(clouds[v],Q[v]);distance,index=cKDTree(shared).query(z)
            good=distance<cfg['correspondence_radius_m'];support.append(int(sum(good)))
            if sum(good)<3:continue
            w=1/(1+(distance[good]/cfg['robust_scale_m'])**2)
            if use_color:w*=np.exp(-np.sum((colors[v][good]-rgb[index[good]])**2,axis=1)/(2*cfg['color_scale']**2))
            if w.sum()<1e-12:continue
            Q[v]=rigid_fit(clouds[v][good],shared[index[good]],w)
        rows.append(dict(Q=[q.tolist() for q in Q],sweep=sweep+1,support=support))
    return rows


class Scores:
    def __init__(self,frames,ray_cfg,cfg):
        self.frames=frames;self.cfg=cfg;self.edges=list(itertools.combinations(range(len(frames)),2));self.pairs={};self.cache={}
        for a,b in self.edges:self.pairs[a,b]=RayPair(frames[a],frames[b],frames[a].xyz.mean(0),ray_cfg)
        self.calls=0

    def edge(self,a,b,T):
        key=(a,b,np.round(T,12).tobytes())
        if key in self.cache:return self.cache[key]
        self.calls+=1;h=self.frames[a];c=self.frames[b];pair=self.pairs[a,b];x=moved(h.xyz,T);y=moved(c.xyz,np.linalg.inv(T))
        d1=c.tree.query(x)[0];d2=h.tree.query(y)[0];scale=self.cfg['evaluation_distance_m']
        nn=float((np.mean(np.minimum(d1/scale,1)**2)+np.mean(np.minimum(d2/scale,1)**2))/2)
        forward=pair.direction(x,h.rgb,c);backward=pair.direction(y,c.rgb,h)
        ray=float((forward['ordinary_overlap_rgb']+backward['ordinary_overlap_rgb'])/2)
        value=dict(nn=nn,ray=ray,surface=float((forward['surface']+backward['surface'])/2),unknown=float((forward['unknown']+backward['unknown'])/2))
        self.cache[key]=value;return value

    def state(self,Q):
        values=[self.edge(a,b,np.linalg.inv(Q[b])@Q[a]) for a,b in self.edges]
        ray=np.array([v['ray'] for v in values]);nn=np.array([v['nn'] for v in values])
        deleted=np.array([np.mean([ray[k] for k,(a,b) in enumerate(self.edges) if d not in (a,b)]) for d in range(len(Q))])
        return dict(joint_nn=float(nn.mean()),joint_ray=float(ray.mean()),deletion_losses=deleted.tolist(),edge_values=values)


def selectors(rows):
    """Finite-pool minimax regret, no calibrated failure probability/certificate."""
    losses=np.array([r['loss']['deletion_losses'] for r in rows]);optimum=losses.min(axis=0)
    regret=(losses-optimum).max(axis=1);worst=losses.max(axis=1)
    scores=dict(joint_nn=np.array([r['loss']['joint_nn'] for r in rows]),joint_ray=np.array([r['loss']['joint_ray'] for r in rows]),worst_source=worst,minimax_source_regret=regret)
    return {m:int(np.argmin(v)) for m,v in scores.items()},dict(deletion_best_losses=optimum.tolist(),regret=regret.tolist(),worst=worst.tolist())
