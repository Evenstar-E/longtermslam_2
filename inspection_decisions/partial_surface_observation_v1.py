"""Observed partial-surface fits and directional quadratic constraints.

Normal/nearest-surface information is conditional and local, not covariance.
No renderer, evaluator, filesystem, or physical-ID dependency.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from comparative_evidence.estimation_v1 import reference_map,world_components
from inspection_decisions.correspondence_joint_v1 import pair_shift,assign,report
from inspection_decisions.correspondence_motion_objective_v1 import profile_cost,seeds_from


def capture(history,current):
    refs=reference_map(sorted(history,key=lambda o:o['view_id']))
    def cloud(c):
        return dict(group=int(c['group']),points=c['points'].tolist(),normals=c['normals'].tolist(),
                    valid=c.get('valid',c.get('normal_valid')).tolist())
    return dict(references=[dict(ref_id=r['ref_id'],anchor_token=r['anchor_token'],anchor_pixels=r['anchor_pixels'],**cloud(r)) for r in refs],
        frames=[dict(token=o['token'],components=[dict(pixels=c['pixels'].tolist(),**cloud(c)) for c in world_components(o)]) for o in current])


def trimmed_shift(p,q,cfg):
    d=np.median(q,0)-np.median(p,0);tree=cKDTree(p)
    for _ in range(cfg['pair_iterations']):
        distance,idx=tree.query(q-d);k=max(3,int(np.ceil(cfg['trim_fraction']*len(q))))
        use=np.argsort(distance,kind='stable')[:k];d=np.mean(q[use]-p[idx[use]],0)
    # One-sided score: missing historical samples are not a current mismatch.
    score=float(np.mean(np.minimum(tree.query(q-d)[0],.3)))
    return d,score


def plane_shift(p,pn,pv,q,qn,qv,cfg):
    p=p[pv];pn=pn[pv];q=q[qv];qn=qn[qv]
    if min(len(p),len(q))<cfg['minimum_planes']:return None
    compatible=qn@pn.T>cfg['normal_cosine']
    d=np.median(q,0)-np.median(p,0);N=None;z=None
    for _ in range(cfg['pair_iterations']):
        distances=np.sum((q[:,None]-d-p[None])**2,axis=2)
        distances=np.where(compatible,distances,np.inf);idx=np.argmin(distances,axis=1)
        rows=np.flatnonzero(np.isfinite(distances[np.arange(len(q)),idx]))
        if len(rows)<cfg['minimum_planes']:return None
        normal=pn[idx[rows]];rhs=np.einsum('ij,ij->i',normal,q[rows]-p[idx[rows]])
        residual=abs(rhs-normal@d);k=max(cfg['minimum_planes'],int(np.ceil(cfg['trim_fraction']*len(rows))))
        use=np.argsort(residual,kind='stable')[:k];N=normal[use];z=rhs[use]
        # Preserve the initial value only in unobserved directions; do not add information there.
        d+=np.linalg.lstsq(N,z-N@d,rcond=1e-3)[0]
    H=3*(N.T@N)/(len(N)*cfg['sigma']**2)
    # Distance to a locally supported plane; boundary samples outside overlap don't add a reverse term.
    score=float(np.mean(abs(z-N@d)))
    return d,score,H


def prepare(raw,cfg):
    n=len(raw['references']);backends={k:dict(references=[{z:r[z] for z in ['ref_id','anchor_token','anchor_pixels']} for r in raw['references']],frames=[]) for k in ['legacy','trimmed','surface']}
    for f in raw['frames']:
        m=len(f['components']);tables={k:dict(token=f['token'],pixels=[c['pixels'] for c in f['components']],
            D=np.zeros((m,n,3)),S=np.ones((m,n))*1e4,valid=np.zeros((m,n),bool),H=np.zeros((m,n,3,3))) for k in backends}
        for i,c in enumerate(f['components']):
            q=np.array(c['points']);qn=np.array(c['normals']);qv=np.array(c['valid'],bool)
            for j,r in enumerate(raw['references']):
                if c['group']!=r['group']:continue
                # The same deterministic cap is used by all three backends.
                take=np.linspace(0,len(r['points'])-1,min(cfg['reference_point_cap'],len(r['points'])),dtype=int)
                p=np.array(r['points'])[take];pn=np.array(r['normals'])[take];pv=np.array(r['valid'],bool)[take]
                old=pair_shift(p,q);trim=trimmed_shift(p,q,cfg);plane=plane_shift(p,pn,pv,q,qn,qv,cfg)
                for k,fit in [('legacy',old),('trimmed',trim),('surface',plane)]:
                    if fit is None:continue
                    d,s=fit[:2];tables[k]['D'][i,j]=d;tables[k]['S'][i,j]=s;tables[k]['valid'][i,j]=True
                    tables[k]['H'][i,j]=fit[2] if k=='surface' else np.eye(3)/cfg['sigma']**2
        for k,table in tables.items():
            for name in ['D','S','valid','H']:table[name]=table[name].tolist()
            backends[k]['frames'].append(table)
    return backends


def arrays(data):
    n=len(data['references'])
    return [(np.array(f['D']).reshape(-1,n,3),np.array(f['S']).reshape(-1,n),np.array(f['valid'],bool).reshape(-1,n),np.array(f['H']).reshape(-1,n,3,3)) for f in data['frames']]


def pair_cost(D,H,t,delta):
    r=D-t-delta[None];return np.einsum('mni,mnij,mnj->mn',r,H,r)


def directional_profile(D,H,t,cfg,F):
    r=D-t;g=np.einsum('...ij,...j->...i',H,r)
    delta=np.linalg.solve(H+np.eye(3)/(F*cfg['motion_scale_m']**2),g[...,None])[...,0]
    stable=np.einsum('...i,...i->...',r,g)
    return stable-np.maximum(np.einsum('...i,...i->...',g,delta)-cfg['change_penalty']/F,0)


def full_energy(fs,matches,t,delta,cfg):
    value=cfg['change_penalty']*sum(np.dot(d,d)>1e-20 for d in delta)+np.sum(delta**2)/cfg['motion_scale_m']**2
    for (D,S,V,H),pairs in zip(fs,matches):
        value+=12*(len(D)-len(pairs));C=pair_cost(D,H,t,delta)+S/.1
        value+=sum(C[i,j] for i,j in pairs)
    return float(value)


def bounded_quadratic(H,g,t,bound):
    x=np.linalg.lstsq(H,g,rcond=1e-10)[0]
    if np.max(abs(x))<=bound:return x
    result=minimize(lambda z:float(z@H@z-2*g@z),t,jac=lambda z:2*(H@z-g),
        bounds=[(-bound,bound)]*3,method='L-BFGS-B',options=dict(ftol=1e-13,gtol=1e-10,maxiter=100))
    return result.x if result.fun<=float(t@H@t-2*g@t)+1e-8 else t


def fit(fs,seed,cfg):
    n=fs[0][0].shape[1];t=np.array(seed);delta=np.zeros((n,3));F=max(len(fs),1)
    matches=[assign(directional_profile(D,H,t,cfg,F)+S/.1,V) for D,S,V,H in fs]
    trace=[full_energy(fs,matches,t,delta,cfg)]
    for _ in range(cfg['iterations']):
        M=np.zeros((n,3,3));g=np.zeros((n,3))
        for (D,S,V,H),pairs in zip(fs,matches):
            for i,j in pairs:M[j]+=H[i,j];g[j]+=H[i,j]@(D[i,j]-t)
        for j in range(n):
            d=np.linalg.solve(M[j]+np.eye(3)/cfg['motion_scale_m']**2,g[j])
            gain=2*g[j]@d-d@M[j]@d-d@d/cfg['motion_scale_m']**2
            delta[j]=d if gain>cfg['change_penalty'] else 0
        trace.append(full_energy(fs,matches,t,delta,cfg))
        Mt=np.zeros((3,3));gt=np.zeros(3)
        for (D,S,V,H),pairs in zip(fs,matches):
            for i,j in pairs:Mt+=H[i,j];gt+=H[i,j]@(D[i,j]-delta[j])
        if matches:t=bounded_quadratic(Mt,gt,t,cfg['public_session_translation_bound_m'])
        trace.append(full_energy(fs,matches,t,delta,cfg))
        matches=[assign(pair_cost(D,H,t,delta)+S/.1,V) for D,S,V,H in fs]
        trace.append(full_energy(fs,matches,t,delta,cfg))
    assert np.max(np.diff(trace),initial=0)<1e-6
    return dict(translation=t.tolist(),delta=delta.tolist(),matches=matches,score=trace[-1],energy_trace=trace)


def support_information(data,matches):
    n=len(data['references']);M=np.zeros((n,3,3))
    for (D,S,V,H),pairs in zip(arrays(data),matches):
        for i,j in pairs:M[j]+=H[i,j]
    return np.linalg.eigvalsh(M)


def surface_seeds(data,cfg):
    fs=arrays(data);voted=seeds_from([(D,S,V) for D,S,V,H in fs],cfg)
    ds=np.concatenate([D[V] for D,S,V,H in fs]);hs=np.concatenate([H[V] for D,S,V,H in fs])
    middle=np.zeros(3)
    for axis in range(3):
        w=hs[:,axis,axis];order=np.argsort(ds[:,axis],kind='stable')
        if w.sum()>0:middle[axis]=ds[order[np.searchsorted(np.cumsum(w[order]),w.sum()/2)],axis]
    middle=np.clip(middle,-cfg['public_session_translation_bound_m'],cfg['public_session_translation_bound_m'])
    return [voted[0],middle]+voted[1:cfg['max_seeds']]


def solve_directional(data,cfg,seeds=None):
    fs=arrays(data);seeds=surface_seeds(data,cfg) if seeds is None else seeds
    candidates=[fit(fs,s,cfg) for s in seeds];unique={}
    for r in sorted(candidates,key=lambda r:r['score']):
        signature=tuple(tuple(tuple(p) for p in pairs) for pairs in r['matches']);unique.setdefault(signature,r)
    ranked=sorted(unique.values(),key=lambda r:r['score']);best=ranked[0]
    branches=[r for r in ranked if r['score']<=best['score']+cfg['branch_tolerance']]
    return dict(prediction=report(data,best,branches),best=best,branches=branches,distinct_assignment_hypotheses=len(ranked),seed_count=len(seeds))


def apply_surface_gate(result,data,cfg):
    # Also applied to the isotropic same-surface control; no fake precision in missing directions.
    eigen=support_information(data,result['best']['matches']);result['surface_information_eigenvalues']=eigen.tolist()
    for a,e in zip(result['prediction']['assets'],eigen):
        if e[0]<cfg['observability_ratio']*max(e[-1],1e-12):a['label']=-1
    return result
