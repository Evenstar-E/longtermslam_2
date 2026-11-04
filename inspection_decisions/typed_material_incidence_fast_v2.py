"""Same exact scalar energy, sorted reductions replacing repeated scatter."""
import numpy as np
from inspection_decisions.typed_material_incidence_graph_v1 import (
    period_arrays as original_arrays,period_geometry as original_geometry,
    graph_arrays as original_graph)


def period_arrays(data):
    out=original_arrays(data);order=np.argsort(out['pi'],kind='stable');unique,starts=np.unique(out['pi'][order],return_index=True);out.update(boundary_order=order,source_pixels=unique,source_starts=starts);return out


def period_geometry(data,x,cfg):
    out=original_geometry(data,x,cfg)
    for key in ['boundary_order','source_pixels','source_starts']:out[key]=data[key]
    return out


def graph_arrays(graph,d):
    out=original_graph(graph,d);order=np.argsort(out['f'],kind='stable')
    for key in ['f','g','h','lo','hi']:out[key]=out[key][order]
    out['source_faces'],out['source_starts']=np.unique(out['f'],return_index=True);return out


def period_cost(geo,graph,cfg,line=False):
    Q=geo['Q'];N=len(Q);pi=geo['pi'];M=len(pi);cap2=cfg['support_match_cap_m']**2;axes=graph['axes'];F=len(axes)
    if not F:return (cap2/3 if N else 0)+(cap2/3 if M else 0)
    d=graph['d'];cost=np.minimum((Q[:,axes]-d[None])**2+geo['normal_cost'][:,axes],cap2)/(3*max(N,1));f=graph['f'];g=graph['g'];D=len(f)
    if M:
        best=np.full((M,F),cap2)
        if D:
            a=axes[f];k=axes[g];h=graph['h'];O=geo['O'][pi];ri=geo['ri'][pi];ro=geo['ro'];B=ro[:,a];A=ri[:,a];stable=(abs(B)>cfg['denominator_tolerance'])&(abs(A)>cfg['denominator_tolerance']);lam=np.divide(d[f][None]-O[:,a],B,out=np.full((M,D),-1.),where=stable);ratio=np.divide(A,B,out=np.zeros((M,D)),where=stable);step=ratio*ro[:,k]-ri[:,k];s=np.sign(step);valid=stable&(ratio>0)&(lam>0)&(abs(step)>cfg['denominator_tolerance']);outside=O[:,k]+ro[:,k]*lam;inside=Q[pi][:,k];v=np.maximum(s*(inside-d[g]),0)**2+np.maximum(s*(d[g]-outside),0)**2
            if line:v=((inside+outside)/2-d[g])**2
            margin=cfg['edge_span_margin_m'];along=Q[pi][:,h];v+=B*B*np.maximum(lam-geo['outside'][:,None]+cfg['depth_noise_bound_m'],0)**2+np.maximum(graph['lo'][None]-margin-along,0)**2+np.maximum(along-graph['hi'][None]-margin,0)**2;v=np.where(valid,np.minimum(v,cap2),cap2);best[:,graph['source_faces']]=np.minimum.reduceat(v,graph['source_starts'],axis=1)
        cost[geo['source_pixels']]+=np.add.reduceat(best[geo['boundary_order']],geo['source_starts'],axis=0)/(3*M)
    return float(cost.min(1).sum())
