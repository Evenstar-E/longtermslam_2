"""Soft domain labels at actual material observations, never a filled square.

Integer Potts min-cut and exact finite-pool lazy optimization are STANDARD.
This is a graph-energy research hypothesis, NOT a normalized sensor posterior.
"""
import time
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from scipy.optimize import milp,Bounds,LinearConstraint
from inspection_decisions.first_return_material_surface_v1 import project
from inspection_decisions.ray_conditioned_plane_polytope_v1 import intersection_bounds
from scripts.completion_catalog_scene_v1 import transform

ANCHORS=np.array([[x,y] for y in [-1.,0.,1.] for x in [-1.,0.,1.]])


def graph_edges(x,cfg):
    if len(x)<2:return []
    distance,index=cKDTree(x).query(x,k=min(len(x),cfg['graph_neighbors']+1))
    if distance.ndim==1:return []
    edges={}
    for i,(dd,jj) in enumerate(zip(distance,index)):
        for d,j in zip(dd[1:],jj[1:]):
            if d<=cfg['graph_radius_source_px']+1e-12 and i!=j:
                key=tuple(sorted([i,int(j)]));edges[key]=int(round(cfg['capacity_scale']/(1+d*d)))
    return [(a,b,w) for (a,b),w in sorted(edges.items())]


def observation_graph(uv,event,cfg):
    uv=np.asarray(uv,float).reshape(-1,2);event=np.asarray(event,int)
    # Input event +1 = observed free; -1 = measured same-slot surface match.
    raw=np.r_[ANCHORS,uv];x,inverse=np.unique(np.round(raw,cfg['coordinate_round_decimals']),axis=0,return_inverse=True)
    unary=np.zeros(len(x),np.int64);np.add.at(unary,inverse[9:],event*cfg['capacity_scale'])
    anchors=np.unique(inverse[:9]);queries=np.unique(inverse[9:])
    return {'x':x,'unary':unary,'anchors':anchors,'queries':queries,'edges':graph_edges(x,cfg),
            'free':int(np.sum(event==1)),'matches':int(np.sum(event==-1))}


def query_graph(patch,T0,T,frames,slot_labels,slot,cfg):
    M=T@np.linalg.inv(T0)@patch['source_camera'];world=transform(patch['corner_points'],M)
    center_n=patch['normals'].mean(0);locations=[];events=[];counts=Counter()
    for frame,lab in zip(frames,slot_labels):
        cam,uv=project(world,frame['C'],cfg)
        if (cam[:,2]<=0).any():raise ValueError('Patch crosses camera plane')
        low=np.maximum(np.floor(uv.min(0)).astype(int)-1,[0,0]);high=np.minimum(np.ceil(uv.max(0)).astype(int)+1,[cfg['width']-1,cfg['height']-1])
        if (low>high).any():continue
        yy,xx=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        rc=np.c_[(xx.ravel()-(cfg['width']-1)/2)/cfg['focal_px'],(yy.ravel()-(cfg['height']-1)/2)/cfg['focal_px'],np.ones(xx.size)]
        relative=np.linalg.inv(M)@frame['C'];o=relative[:3,3];r=rc@relative[:3,:3].T
        bounds=intersection_bounds(patch,o,r);d=frame['depth'][yy.ravel(),xx.ravel()];valid=bounds['valid']
        free=valid&(bounds['upper']<d-cfg['depth_tolerance_m'])
        match=valid&(bounds['lower']>=d-cfg['depth_tolerance_m'])&(bounds['upper']<=d+cfg['depth_tolerance_m'])&(lab[yy.ravel(),xx.ravel()]==slot)
        use=free|match;counts['queried']+=len(d);counts['supported']+=int(valid.sum());counts['free']+=int(free.sum());counts['matches']+=int(match.sum())
        if use.any():
            z=(1-center_n@o)/(r[use]@center_n);p=o+z[:,None]*r[use]
            uv=p[:,:2]/p[:,2,None]*patch['focal']+patch['principal']-patch['center_uv']
            locations.append(uv);events.append(np.where(free[use],1,-1))
    graph=observation_graph(np.concatenate(locations) if locations else np.empty((0,2)),np.concatenate(events) if events else np.empty(0,int),cfg)
    graph['counts']=dict(counts);return graph


def cut_energy(g,spatial=True,return_labels=False):
    n=len(g['x']);source=n;sink=n+1;unary=g['unary'];edges=g['edges'] if spatial else []
    inf=int(np.abs(unary).sum()+sum(w for _,_,w in edges)+1)
    rr=[];cc=[];vv=[];offset=int(np.minimum(unary,0).sum())
    anchors=set(map(int,g['anchors']))
    for i,u in enumerate(unary):
        d0=max(-int(u),0)+(inf if i in anchors else 0);d1=max(int(u),0)
        if d0:rr.append(source);cc.append(i);vv.append(d0)
        if d1:rr.append(i);cc.append(sink);vv.append(d1)
    for a,b,w in edges:rr.extend([a,b]);cc.extend([b,a]);vv.extend([w,w])
    capacities=coo_matrix((np.array(vv,np.int64),(rr,cc)),shape=(n+2,n+2)).tocsr()
    answer=maximum_flow(capacities,source,sink,method='dinic')
    energy=int(answer.flow_value)+offset
    if not return_labels:return energy
    residual=capacities-answer.flow;residual.data=np.where(residual.data>0,1,0);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,source,directed=True,return_predecessors=False)
    label=np.zeros(n,bool);label[reachable[reachable<n]]=True
    assert label[list(anchors)].all()
    direct=int(unary@label)+sum(w for a,b,w in edges if label[a]!=label[b]);assert direct==energy
    return energy,label


def shared_graph(a,b,cfg):
    raw=np.r_[a['x'],b['x']];x,mapping=np.unique(np.round(raw,cfg['coordinate_round_decimals']),axis=0,return_inverse=True)
    ma=mapping[:len(a['x'])];mb=mapping[len(a['x']):];unary=np.zeros(len(x),np.int64)
    np.add.at(unary,ma,a['unary']);np.add.at(unary,mb,b['unary']);edges=Counter()
    for graph,m in [(a,ma),(b,mb)]:
        for i,j,w in graph['edges']:
            if m[i]!=m[j]:edges[tuple(sorted([int(m[i]),int(m[j])]))]+=w
    # Cross-epoch regularization only ADDS nonnegative terms. Keep all original
    # within-epoch edges: independent energies are an admissible lower bound.
    qa=a['queries'];qb=b['queries'];cross={}
    if len(qa) and len(qb):
        for first,second,mf,ms in [(a,b,ma,mb),(b,a,mb,ma)]:
            qf=first['queries'];qs=second['queries'];k=min(len(qs),cfg['graph_neighbors'])
            dd,jj=cKDTree(second['x'][qs]).query(first['x'][qf],k=k)
            if k==1:dd=dd[:,None];jj=jj[:,None]
            for ii,(dist,inds) in enumerate(zip(dd,jj)):
                for d,j in zip(dist,inds):
                    v1=int(mf[qf[ii]]);v2=int(ms[qs[j]])
                    if v1!=v2 and d<=cfg['graph_radius_source_px']+1e-12:
                        cross[tuple(sorted([v1,v2]))]=int(round(cfg['capacity_scale']/(1+d*d)))
    for pair,w in cross.items():edges[pair]+=w
    return {'x':x,'unary':unary,'anchors':np.unique(np.r_[ma[a['anchors']],mb[b['anchors']]]),
            'queries':np.unique(np.r_[ma[qa],mb[qb]]),'edges':[(i,j,w) for (i,j),w in sorted(edges.items())],
            'cross_edges':len(cross),'matches':a['matches']+b['matches'],'free':a['free']+b['free']}


def reduced_global(branches,costs,time_limit=20.):
    """Exact elimination of pose choices conditional on observed slot pairs."""
    n=len(branches);variables=[];values=[];rr=[];cc=[]
    for i,(bb,cost) in enumerate(zip(branches,costs)):
        for s in sorted({b['slot'] for b in bb[0]}):
            ia=np.array([k for k,b in enumerate(bb[0]) if b['slot']==s])
            for t in sorted({b['slot'] for b in bb[1]}):
                ib=np.array([k for k,b in enumerate(bb[1]) if b['slot']==t]);sub=cost[np.ix_(ia,ib)]
                k,l=np.unravel_index(np.argmin(sub),sub.shape);k=int(ia[k]);l=int(ib[l]);j=len(values)
                variables.append((i,k,l));values.append(float(cost[k,l]));rr.extend([i,n+s,2*n+t]);cc.extend([j]*3)
    A=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(3*n,len(values))).tocsc();stamp=time.perf_counter()
    ans=milp(np.array(values),integrality=np.ones(len(values)),bounds=Bounds(0,1),constraints=LinearConstraint(A,np.ones(3*n),np.ones(3*n)),
             options={'time_limit':time_limit,'mip_rel_gap':0.})
    selected=sorted(variables[j] for j in np.flatnonzero(ans.x>.5)) if ans.x is not None else None
    return {'status':int(ans.status),'selection':selected,'objective':float(ans.fun) if ans.fun is not None else None,
            'elapsed_s':time.perf_counter()-stamp,'variables':len(values),'mip_gap':float(ans.mip_gap) if getattr(ans,'mip_gap',None) is not None else None}


def lazy_shared(branches,graphs,independent,cfg):
    lower=[v.copy() for v in independent];known=[np.zeros(v.shape,bool) for v in lower];cache={};log=[];start=time.perf_counter()
    for iteration in range(cfg['lazy_max_rounds']):
        result=reduced_global(branches,lower,cfg['global_time_limit_s'])
        if result['status']!=0 or not result['selection']:return {**result,'certified':False,'reason':'global_solver_unresolved'},lower,known,cache,log
        pending=[(i,k,l) for i,k,l in result['selection'] if not known[i][k,l]]
        if not pending:return {**result,'certified':True,'rounds':iteration+1,'elapsed_s':time.perf_counter()-start},lower,known,cache,log
        total_extra=0
        for i,k,l in pending:
            extra=0;cross=0
            for a,b in zip(graphs[i][0][k],graphs[i][1][l]):
                joined=shared_graph(a,b,cfg);cross+=joined['cross_edges'];ej=cut_energy(joined)
                delta=ej-a['energy']-b['energy'];assert delta>=0,(i,k,l,delta)
                extra+=delta
            increment=extra/(cfg['capacity_scale']*max(1,len(graphs[i][0][k])))
            lower[i][k,l]+=increment;known[i][k,l]=True;cache[(i,int(k),int(l))]={'extra_energy_integer':int(extra),'cross_edges':cross,'increment':increment};total_extra+=extra
        log.append({'round':iteration+1,'lower_bound':result['objective'],'new_exact_pairs':len(pending),'added_energy_integer':int(total_extra),'elapsed_s':time.perf_counter()-start})
        if time.perf_counter()-start>cfg['lazy_scene_limit_s']:
            return {'status':1,'selection':None,'certified':False,'reason':'predeclared_scene_limit','rounds':iteration+1,'elapsed_s':time.perf_counter()-start},lower,known,cache,log
    return {'status':1,'selection':None,'certified':False,'reason':'predeclared_round_limit'},lower,known,cache,log
