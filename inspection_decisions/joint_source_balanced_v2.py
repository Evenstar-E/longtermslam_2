"""Stronger ordinary control: each source contributes its own correspondences."""
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.real_temporal_v1 import moved
from inspection_decisions.joint_source_regret_v1 import rigid_fit


def joint_icp_balanced(frames,start,cfg,use_color=False):
    Q=[q.copy() for q in start];clouds=[];colors=[];rows=[]
    for f in frames:
        ids=np.linspace(0,len(f.xyz)-1,min(len(f.xyz),cfg['solver_points_per_view'])).astype(int);clouds.append(f.xyz[ids]);colors.append(f.rgb[ids])
    for sweep in range(cfg['solver_sweeps']):
        support=[]
        for v in range(1,len(frames)):
            x=[];y=[];weights=[];counts=[];z=moved(clouds[v],Q[v])
            for s in range(len(frames)):
                if s==v:continue
                target=moved(clouds[s],Q[s]);distance,index=cKDTree(target).query(z);good=distance<cfg['correspondence_radius_m'];counts.append(int(sum(good)))
                if sum(good)<3:continue
                w=1/(1+(distance[good]/cfg['robust_scale_m'])**2)
                if use_color:w*=np.exp(-np.sum((colors[v][good]-colors[s][index[good]])**2,axis=1)/(2*cfg['color_scale']**2))
                if w.sum()<1e-12:continue
                x.append(clouds[v][good]);y.append(target[index[good]]);weights.append(w/w.sum())
            if x:Q[v]=rigid_fit(np.concatenate(x),np.concatenate(y),np.concatenate(weights))
            support.append(counts)
        rows.append(dict(Q=[q.tolist() for q in Q],sweep=sweep+1,support=support))
    return rows
