"""Finite-horizon continuation regression; observed-only causal STOP actor."""
import numpy as np
from inspection_decisions.report_risk_stopping_v1 import feature_row,fit_head,predict,first_stop as legacy_first_stop

NEW=['optional_cf_temporal','optional_in_temporal','optional_cf_static',
     'one_step_temporal','full_tail_temporal','optional_plain_temporal']


def scalar_fit(X,y,alpha):
    return fit_head(X,np.c_[y,np.zeros(len(y))],alpha)


def value(X,model):
    X=np.asarray(X)
    if X.ndim==1:return float(predict(X,model)[0])
    return np.array([predict(x,model)[0] for x in X])


def pack(paths,temporal):
    rows=[]
    for p in paths:
        states=p['observed']['states'];previous=None;X=[]
        for state in states:
            X.append(feature_row(state,previous,states[0]['prediction'],temporal));previous=state['prediction']
        rows.append(dict(family=p['scene'],policy=p['policy'],X=np.array(X),
                         L=np.array([m['task_loss']/8 for m in p['labels_data']['metrics']])))
    return rows


def fit_risk(rows,alpha):
    X=np.concatenate([r['X'] for r in rows]);y=np.concatenate([r['L'] for r in rows])
    return scalar_fit(X,y,alpha)


def ordinary_bank(rows,risk,alpha,cost,kind='recursive'):
    """Within-training backward bank; no held-family data is supplied here."""
    bank=[None]*5;targets=[None]*5
    for t in range(4,-1,-1):
        X=np.array([r['X'][t] for r in rows]);nextX=np.array([r['X'][t+1] for r in rows])
        if t==4 or kind=='one_step':y=cost+np.array([r['L'][t+1] for r in rows])
        elif kind=='full_tail':y=(5-t)*cost+np.array([r['L'][5] for r in rows])
        else:y=cost+np.minimum(value(nextX,risk),value(nextX,bank[t+1]))
        bank[t]=scalar_fit(X,y,alpha);targets[t]=y.tolist()
    return bank,targets


def crossfit_bank(rows,risk,alpha,cost,folds=4):
    targets=np.zeros((len(rows),5));teachers=[]
    for fold in range(folds):
        training=[r for r in rows if r['family']%folds!=fold]
        ids=[i for i,r in enumerate(rows) if r['family']%folds==fold]
        teacher_risk=fit_risk(training,alpha)
        teacher_bank,_=ordinary_bank(training,teacher_risk,alpha,cost)
        held=[rows[i] for i in ids]
        assert not {r['family'] for r in training}&{r['family'] for r in held}
        for t in range(5):
            nextX=np.array([r['X'][t+1] for r in held])
            y=cost+np.array([r['L'][5] for r in held]) if t==4 else cost+np.minimum(
                value(nextX,teacher_risk),value(nextX,teacher_bank[t+1]))
            targets[ids,t]=y
        teachers.append(dict(fold=fold,training_families=sorted({r['family'] for r in training}),
                             held_families=sorted({r['family'] for r in held}),risk=teacher_risk,bank=teacher_bank))
    bank=[scalar_fit(np.array([r['X'][t] for r in rows]),targets[:,t],alpha) for t in range(5)]
    return bank,dict(teachers=teachers,targets=targets.tolist(),
                     rows=[dict(family=r['family'],policy=r['policy']) for r in rows])


def train_banks(paths,legacy,alpha=10.,price=.05,folds=4):
    heads={};audit={}
    for temporal in (True,False):
        rows=pack(paths,temporal);risk=legacy['heads']['temporal' if temporal else 'static']
        risk_check=fit_risk(rows,alpha)
        X=np.concatenate([r['X'] for r in rows])
        assert np.max(abs(value(X,risk)-value(X,risk_check)))<1e-12
        bank,detail=crossfit_bank(rows,risk,alpha,price/8,folds)
        name='optional_cf_temporal' if temporal else 'optional_cf_static'
        heads[name]=dict(temporal=temporal,risk=risk,bank=bank);audit[name]=detail
        if not temporal:continue
        for name,kind in [('optional_in_temporal','recursive'),('one_step_temporal','one_step'),('full_tail_temporal','full_tail')]:
            bank,targets=ordinary_bank(rows,risk,alpha,price/8,kind)
            heads[name]=dict(temporal=True,risk=risk,bank=bank);audit[name]=dict(targets_by_stage=targets)
    return heads,audit


def decision(method,state,previous,initial,bundle,setting):
    t=state['stage']
    if t==5:return True,dict(reason='budget')
    key='optional_cf_temporal' if method=='optional_plain_temporal' else method
    head=bundle['heads'][key];x=feature_row(state,previous,initial,head['temporal'])
    risk=value(x,head['risk']);continuation=value(x,head['bank'][t])
    cap=True if method=='optional_plain_temporal' else risk<=setting
    stop=bool(cap and risk<=continuation)
    model=head['bank'][t]
    outside=float(((x<np.array(model['train_min'])-1e-9)|(x>np.array(model['train_max'])+1e-9)).mean())
    return stop,dict(risk=risk,continuation_cost=continuation,advantage_to_stop=continuation-risk,
                     risk_cap_allows=bool(cap),outside_stage_training_fraction=outside,reason='optional_value')


def first_stop(states,method,bundle,setting,price=.05):
    if method not in NEW:return legacy_first_stop(states,method,bundle['legacy']['heads'],setting,price)
    previous=None;initial=None;decisions=[]
    for state in states:
        if initial is None:initial=state['prediction']
        stop,detail=decision(method,state,previous,initial,bundle,setting)
        decisions.append(dict(stage=state['stage'],stop=stop,**detail))
        if stop:return state['stage'],decisions
        previous=state['prediction']
    raise ValueError('No terminal observation')
