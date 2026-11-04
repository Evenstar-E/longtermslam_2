"""Observed-only component evidence. No evaluator/filesystem imports."""
import numpy as np
from comparative_evidence.estimation_v1 import world_components,associate
from inspection_decisions.report_risk_stopping_v1 import feature_row

NEW=['component_conflict','component_report','component_unpaired','component_no_modes','component_no_identity','global_augmented']

def capture(fit,h,q,meta):
    refs=fit['references'];n=len(refs);start=fit['delta_start'];end=fit['nuisance_start']
    observations={j:[] for j in range(n)}
    for epoch,frames in [('history',h),('current',q)]:
        for obs in frames:
            for j,c,gap in associate(world_components(obs),refs):
                observations[j].append(dict(epoch=epoch,offset=(c['centre']-refs[j]['centre']).tolist(),
                    ambiguity=float(np.exp(-gap/.15)),source=str(obs['nuisance_group'])))
    return dict(mode_deltas=[m['mean'][start:end].reshape(n,3).tolist() for m in fit['modes']],
        weights=np.asarray(fit['mode_weights']).tolist(),selected=fit['mean'][start:end].reshape(n,3).tolist(),
        sensor_var=[float(sum(w*np.trace(S[start+3*j:start+3*j+3,start+3*j:start+3*j+3]) for w,S in zip(fit['mode_weights'],fit['sensor_covariances']))) for j in range(n)],
        observations={str(j):v for j,v in observations.items()})

def local_rows(state,previous,initial):
    p=state['prediction'];e=state['evidence'];w=np.array(e['weights']);D=np.array(e['mode_deltas']);ids=[];B=[];V=[]
    prev={a['ref_id']:a for a in previous['assets']} if previous else {};first={a['ref_id']:a for a in initial['assets']}
    for a in sorted(p['assets'],key=lambda a:a['ref_id']):
        j=a['ref_id'];ids.append(j);d=np.array(a['delta']);C=np.array(a['covariance']);sd=np.sqrt(max(np.trace(C),0))
        b=[a['decision_risk'],float(a['label']<0),float(a['label']==1),np.linalg.norm(d)/.12,sd/.12,
           np.sqrt(max(np.linalg.eigvalsh(C).max(),0))/.12,np.log1p(a['current_support']),1/(1+max(a['association_margin_m'],0))]
        for reference in (prev,first):
            old=reference.get(j,a);b.extend([np.linalg.norm(d-np.array(old['delta']))/.1,float(a['label']!=old['label']),a['decision_risk']-old['decision_risk']])
        distances=np.linalg.norm(D[:,j]-d,axis=1);take=np.flatnonzero(w>0);change=(np.max(abs(D[:,j]),axis=1)>.12).astype(float);mix=float(w@change)
        maxgap=max((np.linalg.norm(D[x,j]-D[y,j]) for x in take for y in take),default=0.)
        observations=e['observations'][str(j)];current=[o for o in observations if o['epoch']=='current'];offset=np.array([o['offset'] for o in current])
        scatter=float(np.sqrt(np.mean(np.sum((offset-offset.mean(0))**2,axis=1)))) if len(offset) else 0.
        last=float(np.linalg.norm(offset[-1]-offset[:-1].mean(0))) if len(offset)>1 else 0.
        v=[np.sqrt(w@(distances**2))/.1,maxgap/.1,float(w@(distances>.1)),mix*(1-mix),
           float(w@(change!=a['label'])) if a['label']>=0 else 0.,np.linalg.norm(d-np.array(e['selected'][j]))/.1,
           np.sqrt(max(e['sensor_var'][j],0))/.1,max(0.,1-e['sensor_var'][j]/max(np.trace(C),1e-12)),
           float(np.mean([o['ambiguity'] for o in current])) if current else 0.,scatter/.1,last/.1,
           len({o['source'] for o in observations})/max(len(observations),1)]
        B.append(b);V.append(v)
    return ids,np.array(B),np.array(V)

def representation(state,previous,initial,method):
    ids,B,V=local_rows(state,previous,initial);context=feature_row(state,previous,initial,True)
    if method=='global_augmented':
        A=np.c_[B,V];return ids,np.r_[context,A.mean(0),A.max(0),A.std(0)][None]
    if method=='component_unpaired':V=np.broadcast_to(V.mean(0),V.shape).copy()
    if method=='component_no_modes':V=V.copy();V[:,:8]=0
    if method=='component_no_identity':V=V.copy();V[:,8:]=0
    X=np.c_[np.tile(context,(len(ids),1)),B] if method=='component_report' else np.c_[np.tile(context,(len(ids),1)),B,V]
    assert np.isfinite(X).all()
    return ids,X

def fit(X,y,weights,alpha):
    X=np.array(X);y=np.array(y);w=np.array(weights);w=w/w.sum();mean=w@X;scale=np.sqrt(w@((X-mean)**2));scale[scale<1e-9]=1
    A=np.c_[np.ones(len(X)),(X-mean)/scale];reg=np.eye(A.shape[1]);reg[0,0]=0
    beta=np.linalg.solve(A.T@(w[:,None]*A)+alpha*reg,A.T@(w*y))
    return dict(mean=mean.tolist(),scale=scale.tolist(),w=beta.tolist())

def predict(X,model):
    A=np.c_[np.ones(len(X)),(X-np.array(model['mean']))/model['scale']]
    return np.maximum(0,A@model['w'])

def values(state,previous,initial,method,bundle):
    ids,X=representation(state,previous,initial,method);model=bundle['heads'][method]
    factor=1. if method=='global_augmented' else 1/8
    return float(predict(X,model['risk']).sum()*factor),float(predict(X,model['next'][state['stage']]).sum()*factor+.05/8)

def decision(method,state,previous,initial,bundle,setting):
    if state['stage']==5:return True,dict(reason='budget')
    r,c=values(state,previous,initial,method,bundle)
    return bool(r<=setting and r<=c),dict(risk=r,next_cost=c,reason='component_one_step')

def first_stop(states,method,bundle,setting):
    from inspection_decisions.optional_stopping_v1 import first_stop as old_stop
    if method not in NEW:return old_stop(states,method,bundle['legacy'],setting)
    previous=None;initial=None;decisions=[]
    for state in states:
        if initial is None:initial=state['prediction']
        stop,detail=decision(method,state,previous,initial,bundle,setting);decisions.append(dict(stage=state['stage'],stop=stop,**detail))
        if stop:return state['stage'],decisions
        previous=state['prediction']
    raise ValueError('No STOP')
