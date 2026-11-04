"""Causal report-state heads and STOP rules. No evaluator or filesystem access."""
import numpy as np
from inspection_decisions.costs import report_risk


def moments(values):
    a=np.asarray(values,dtype=float)
    return [float(a.mean()),float(a.max()),float(a.std())] if a.size else [0.,0.,0.]


def features(prediction,previous,initial,stage,rank_score,coverage_score,temporal=True):
    assets=sorted(prediction['assets'],key=lambda a:a['ref_id'])
    x=[stage/5.,len(assets)/8.,report_risk(prediction),rank_score,coverage_score/8.]
    columns=[[],[],[],[],[],[],[],[]]
    for a in assets:
        sd=np.sqrt(np.maximum(np.diag(a['covariance']),0))
        vals=[a['decision_risk'],float(a['label']<0),float(a['label']==1),
              np.linalg.norm(a['delta']),np.linalg.norm(sd),
              np.log1p(a['current_support']),1/(1+max(a['association_margin_m'],0)),
              np.max(sd)]
        for c,v in zip(columns,vals):c.append(v)
    for c in columns:x.extend(moments(c))
    if temporal:
        for reference in (previous,initial):
            lookup={a['ref_id']:a for a in reference['assets']} if reference is not None else {}
            deltas=[[] for _ in range(5)];matched=0
            for a in assets:
                if a['ref_id'] not in lookup:continue
                b=lookup[a['ref_id']];matched+=1
                vals=[np.linalg.norm(np.array(a['delta'])-b['delta']),float(a['label']!=b['label']),
                      a['decision_risk']-b['decision_risk'],
                      np.trace(a['covariance'])-np.trace(b['covariance']),
                      np.log1p(a['current_support'])-np.log1p(b['current_support'])]
                for c,v in zip(deltas,vals):c.append(v)
            x.append(matched/max(len(assets),1))
            for c in deltas:x.extend(moments(c))
    x=np.array(x);assert np.isfinite(x).all()
    return x


def fit_head(X,Y,alpha):
    X=np.asarray(X);Y=np.asarray(Y);mean=X.mean(0);scale=X.std(0);scale[scale<1e-9]=1.
    A=np.c_[np.ones(len(X)),(X-mean)/scale];reg=np.eye(A.shape[1]);reg[0,0]=0.
    w=np.linalg.solve(A.T@A/len(A)+alpha*reg,A.T@Y/len(A))
    return dict(mean=mean.tolist(),scale=scale.tolist(),w=w.tolist(),alpha=alpha,
                train_min=X.min(0).tolist(),train_max=X.max(0).tolist())


def predict(x,model):
    z=np.r_[1.,(x-np.array(model['mean']))/model['scale']]
    raw=z@np.array(model['w']);return np.array([max(0.,raw[0]),raw[1]])


def feature_row(state,previous,initial,temporal):
    return features(state['prediction'],previous,initial,state['stage'],
                    state['best_rank_score'],state['best_coverage_score'],temporal)


def should_stop(method,state,previous,initial,models,setting,price=.05):
    stage=state['stage']
    if stage==5:return True,dict(reason='budget')
    if method.startswith('full_'):return False,dict(reason='full_budget')
    if method=='original_rank':return state['best_rank_score']<=0,dict(reason='old_score')
    if method=='fixed_rank':return stage>=setting,dict(reason='fixed_stage')
    if method.startswith('declared_'):
        r=report_risk(state['prediction']);return r<=setting,dict(risk=r,reason='declared_risk')
    temporal=method!='dual_static';model=models['temporal' if temporal else 'static']
    x=feature_row(state,previous,initial,temporal);r,g=predict(x,model)
    remaining=price*(5-stage)/8
    stop=bool(r<=setting and (method=='risk_temporal' or g<=remaining))
    outside=float(((x<np.array(model['train_min'])-1e-9)|(x>np.array(model['train_max'])+1e-9)).mean())
    return stop,dict(risk=float(r),gain=float(g),remaining_price=remaining,
                     feature_outside_train_fraction=outside,reason='learned_head')


def first_stop(states,method,models,setting,price=.05):
    """Lazy consumption is intentional: no future report is accessed after STOP."""
    previous=None;initial=None;decisions=[]
    for state in states:
        if initial is None:initial=state['prediction']
        stop,detail=should_stop(method,state,previous,initial,models,setting,price)
        decisions.append(dict(stage=state['stage'],stop=stop,**detail))
        if stop:return state['stage'],decisions
        previous=state['prediction']
    raise ValueError('Incomplete path without STOP')
