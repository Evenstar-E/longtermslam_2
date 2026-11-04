"""Observed-only competing assignments, shared translation and sparse changes.

Translation-only prototype. Energies/branch sets are heuristic, not posterior.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from comparative_evidence.estimation_v1 import reference_map,world_components

def pair_shift(p,q):
    d=np.median(q,0)-np.median(p,0);tree=cKDTree(p)
    for _ in range(8):
        dist,idx=tree.query(q-d);w=1/(1+(dist/.12)**2)
        d=np.sum((q-p[idx])*w[:,None],0)/max(w.sum(),1e-9)
    a=cKDTree(p).query(q-d)[0];b=cKDTree(q-d).query(p)[0]
    return d,float(np.mean(np.minimum(a,.3))+np.mean(np.minimum(b,.3)))

def prepare(history,current):
    refs=reference_map(sorted(history,key=lambda o:o['view_id']));n=len(refs);frames=[]
    for obs in current:
        cs=world_components(obs);D=np.zeros((len(cs),n,3));S=np.ones((len(cs),n))*1e4;valid=np.zeros((len(cs),n),bool)
        for i,c in enumerate(cs):
            for j,r in enumerate(refs):
                if c['group']!=r['group']:continue
                d,s=pair_shift(r['points'][::max(1,len(r['points'])//100)],c['points'][::max(1,len(c['points'])//100)])
                D[i,j]=d;S[i,j]=s;valid[i,j]=True
        frames.append(dict(token=obs['token'],pixels=[c['pixels'].tolist() for c in cs],D=D.tolist(),S=S.tolist(),valid=valid.tolist()))
    return dict(references=[dict(ref_id=r['ref_id'],anchor_token=r['anchor_token'],anchor_pixels=r['anchor_pixels']) for r in refs],frames=frames)

def assign(cost,valid):
    m,n=cost.shape;M=np.c_[np.where(valid,cost,1e6),np.full((m,m),12.)]
    i,j=linear_sum_assignment(M);return [(int(x),int(y)) for x,y in zip(i,j) if y<n and valid[x,y]]

def solve(data,method,cfg):
    n=len(data['references']);frames=[(np.array(f['D']).reshape(-1,n,3),np.array(f['S']).reshape(-1,n),np.array(f['valid'],bool).reshape(-1,n)) for f in data['frames']]
    sigma=cfg['sigma'];penalty=cfg['change_penalty'];bound=cfg['public_session_translation_bound_m']
    allD=np.concatenate([D[V] for D,S,V in frames]);grid=np.round(allD/.1).astype(int);cells,count=np.unique(grid,axis=0,return_counts=True)
    seeds=[np.zeros(3)]+[c*.1 for c in cells[np.argsort(-count,kind='stable')[:cfg['max_seeds']]]]
    seeds=[s for s in seeds if np.max(abs(s))<=bound]
    if method in ('nearest','hungarian'):seeds=[np.zeros(3)]
    solutions=[]
    for initial in seeds:
        t=np.array(initial);delta=np.zeros((n,3));matches=[]
        for iteration in range(cfg['iterations']):
            matches=[];offsets=[];per=[[] for _ in range(n)]
            for D,S,V in frames:
                residual=np.linalg.norm(D-t-delta[None],axis=2)/sigma
                if method=='joint_branches':C=np.minimum(residual**2,penalty)+S/.1
                elif method=='robust_multistart':C=np.log1p(residual**2)+S/.1
                else:C=residual**2+S/.1
                if method=='nearest':
                    pairs=[(i,int(np.argmin(np.where(V[i],C[i],1e6)))) for i in range(len(D)) if V[i].any()]
                else:pairs=assign(C,V)
                matches.append(pairs)
                for i,j in pairs:offsets.append(D[i,j]);per[j].append(D[i,j])
            if offsets:
                a=np.array(offsets);dist=np.linalg.norm(a-t,axis=1)
                use=np.argsort(dist)[:max(3,int(np.ceil(.65*len(a))))] if method not in ('nearest','hungarian') else np.arange(len(a))
                t=np.clip(np.median(a[use],axis=0),-bound,bound)
            if method=='joint_branches':
                for j,v in enumerate(per):
                    if v:
                        d=np.median(v,0)-t;delta[j]=d if np.linalg.norm(d)>.12 else 0
            # Ordinary fixed-association controls are not allowed to change assignment after fitting drift.
            if method in ('nearest','hungarian'):break
        score=0.;per=[[] for _ in range(n)]
        for (D,S,V),pairs in zip(frames,matches):
            score+=12*(len(D)-len(pairs))
            for i,j in pairs:
                per[j].append(D[i,j]);r=np.linalg.norm(D[i,j]-t)/sigma
                score+=(min(r*r,penalty) if method=='joint_branches' else np.log1p(r*r))+S[i,j]/.1
        disp=np.array([np.median(v,0)-t if v else np.zeros(3) for v in per])
        solutions.append(dict(translation=t.tolist(),delta=disp.tolist(),matches=matches,score=float(score)))
    unique={}
    for s in sorted(solutions,key=lambda s:s['score']):
        signature=tuple(tuple(tuple(p) for p in pairs) for pairs in s['matches']);unique.setdefault(signature,s)
    ranked=sorted(unique.values(),key=lambda s:s['score']);best=ranked[0]
    branches=[s for s in ranked if s['score']<=best['score']+cfg['branch_tolerance']] if method=='joint_branches' else [best]
    p=report(data,best,branches);return dict(prediction=p,best=best,branches=branches,distinct_assignment_hypotheses=len(ranked),seed_count=len(seeds))

def report(data,best,branches):
    n=len(data['references']);assets=[];assignments=[]
    maps=[{(f,i):j for f,pairs in enumerate(s['matches']) for i,j in pairs} for s in branches]
    for j in range(n):
        support=sum(y==j for y in maps[0].values());d=np.array(best['delta'][j]);ambiguous=False
        for s,m in zip(branches,maps):
            if any(m.get(k)!=j for k,v in maps[0].items() if v==j) or np.linalg.norm(d-np.array(s['delta'][j]))>.1:ambiguous=True
        label=-1 if not support or ambiguous else int(np.max(abs(d))>.12)
        assets.append(dict(ref_id=j,label=label,delta=d.tolist(),covariance=(np.eye(3)*.06**2).tolist(),current_support=support,association_margin_m=0. if ambiguous else 10.))
    for f,pairs in enumerate(best['matches']):
        for i,j in pairs:assignments.append(dict(ref_id=j,token=data['frames'][f]['token'],pixels=data['frames'][f]['pixels'][i]))
    return dict(assets=assets,references=data['references'],current_assignments=assignments,residual_median=None,
        limits=['translation-only prototype, fixed nominal covariance not calibrated','point-cloud translation approximates material displacement under partial views'])
