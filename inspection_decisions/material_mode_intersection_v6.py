"""Prescore aggregation control: same distance bounds, no product tightening."""
import numpy as np
from inspection_decisions.material_correspondence_mode_graph_v6 import compatible,cliques,rigid,joint_fit,apply

def intersection_modes(track,cfg):
    x,w=track['x'],track['weight']
    if len(x)<3:return []
    C=np.minimum.reduce([compatible(x[:,0],x[:,1],cfg),compatible(x[:,0],x[:,2],cfg),compatible(x[:,1],x[:,2],cfg)])
    out=[]
    for ix in cliques(C,w,cfg['modes']):
        a=rigid(x[ix,0],x[ix,1],w[ix]);b=rigid(x[ix,0],x[ix,2],w[ix])
        if a is None or b is None:continue
        initial=np.array([a,b]);T,path=joint_fit(initial,x[ix],w[ix],cfg)
        if any(max(np.median(np.linalg.norm(apply(T[e],x[:,0])-apply(r['T'][e],x[:,0]),axis=1)) for e in [0,1])<.005 for r in out):continue
        out.append(dict(T=T,initial=initial,path=path,selected=ix))
    return out
