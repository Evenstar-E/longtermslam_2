"""Observed component-set representation and regret-weighted linear ranking.

Pure actor/trainer code. No file/evaluator reads; no learned visibility claim.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from inspection_decisions.value_pilot_v1 import encode_options


def available_pairs(graph,pairs):
    legal={(a['epoch'],a['view']) for a in graph['actions'] if a['epoch']!='stop'}
    result=[]
    for pair in pairs:
        pair=tuple(map(tuple,pair))
        if not all(a in legal for a in pair):continue
        if sum(a[0]=='history' for a in pair)>graph['context']['history_remaining']:continue
        if sum(a[0]=='current' for a in pair)>graph['context']['current_remaining']:continue
        result.append(pair)
    return result


def representation(graph,pairs,pooled=False):
    pairs=available_pairs(graph,pairs)
    actions=[(a['epoch'],a['view']) for a in graph['actions'] if a['epoch']!='stop']
    options=[()]+[(a,) for a in actions]+pairs
    nodes={n['id']:n for n in graph['nodes']};assets=[k for k,n in nodes.items() if n['kind']=='asset']
    ai={n:j for j,n in enumerate(assets)};n=max(len(assets),1)
    owner={e['source']:e['target'] for e in graph['edges'] if e['kind']=='patch_asset'}
    actnodes={(a['epoch'],a['view']):a['node'] for a in graph['actions'] if a['epoch']!='stop'}
    # No physical ID is used: aggregation is invariant to simultaneous renaming.
    z=[]
    for k in assets:
        f=np.array(nodes[k]['features']);sd=float(np.linalg.norm(f[3:6]))
        z.append([1.,f[9],float(f[8]<0),float(f[8]>0),sd,1/(1+max(f[7],0)),f[6]])
    z=np.asarray(z).reshape(-1,7)
    c={a:np.zeros(len(assets)) for a in actions}
    reverse={v:k for k,v in actnodes.items()}
    for e in graph['edges']:
        if e['kind']=='action_patch' and e['features'][2]>0:
            a=reverse[e['source']];j=ai[owner[e['target']]]
            c[a][j]=max(c[a][j],e['features'][0]*e['features'][1])
    def features(option):
        if not option:return np.zeros(49)
        values=np.array([c[a] for a in option]);union=1-np.prod(1-values,axis=0)
        maximum=values.max(0);total=values.sum(0)
        h=np.max([c[a] for a in option if a[0]=='history'],axis=0) if any(a[0]=='history' for a in option) else np.zeros(len(assets))
        q=np.max([c[a] for a in option if a[0]=='current'],axis=0) if any(a[0]=='current' for a in option) else np.zeros(len(assets))
        basis=np.stack([union,maximum,total,union-maximum,h*q,total-union],axis=1)
        per_asset=(basis.T@z/n).ravel()
        f=np.array([nodes[actnodes[a]]['features'] for a in option])
        extra=[len(option),sum(a[0]=='current' for a in option),f[:,10].sum()/1e5,
               (f[:,8]+f[:,9]).sum()/10,f[:,11].sum(),f[:,12].mean(),
               len(option)*graph['context']['mean_declared_report_risk']]
        return np.r_[per_asset,extra]
    X=np.array([features(o) for o in options]);lookup={a:features((a,)) for a in actions}
    additive=np.array([sum((lookup[a] for a in o),np.zeros(49)) for o in options])
    coverage=np.array([np.max([c[a] for a in o],axis=0).sum() if o else 0. for o in options])
    result=dict(options=options,set=X,additive=additive,coverage=coverage,counts=np.array(list(map(len,options))))
    if pooled:
        if actions:
            old=encode_options(graph,pairs);assert old['options']==options
            result['pooled']=old['X']-old['X'][[0]] # STOP anchor, same observed representation
        else:result['pooled']=np.zeros((1,427))
    return result


def rank_objective(w,D,sign,weight,cost,alpha):
    margin=sign*(D@w-cost)
    value=float(np.dot(weight,np.logaddexp(0,-margin))+.5*alpha*np.dot(w,w))
    grad=D.T@(-weight*sign*expit(-margin))+alpha*w
    return value,grad


def fit(groups,kind,alpha,price):
    key='set' if kind.startswith('set') else 'additive' if kind.startswith('additive') else 'pooled'
    X=np.concatenate([g[key] for g in groups]);scale=np.sqrt(np.mean(X**2,axis=0));scale[scale<1e-10]=1.
    n=len(groups);matrix=[];targets=[];weights=[];costs=[]
    for g in groups:
        x=g[key]/scale;y=np.asarray(g['gain']);counts=g['counts']
        if kind=='set_mse':
            matrix.append(x);targets.extend(y);weights.extend([1/n/len(y)]*len(y))
        else:
            i,j=np.triu_indices(len(y),1);delta=(y-price*counts)[i]-(y-price*counts)[j]
            use=abs(delta)>1e-12;i=i[use];j=j[use];delta=delta[use]
            if not len(delta):continue
            matrix.append(x[i]-x[j]);targets.extend(np.sign(delta));costs.extend(price*(counts[i]-counts[j]))
            weights.extend(abs(delta)/sum(abs(delta))/n)
    D=np.concatenate(matrix);target=np.asarray(targets);weight=np.asarray(weights)
    if kind=='set_mse':
        w=np.linalg.solve(D.T@(weight[:,None]*D)+alpha*np.eye(D.shape[1]),D.T@(weight*target))
        opt=dict(success=True,iterations=1,objective=float(np.dot(weight,(D@w-target)**2)+alpha*np.dot(w,w)))
    else:
        objective=lambda w:rank_objective(w,D,target,weight,np.asarray(costs),alpha)
        result=minimize(objective,np.zeros(D.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
        if not result.success:raise RuntimeError(str(result.message))
        w=result.x;opt=dict(success=True,iterations=int(result.nit),objective=float(result.fun))
    return dict(kind=kind,key=key,alpha=alpha,price=price,w=w.tolist(),scale=scale.tolist(),
                train_min=X.min(0).tolist(),train_max=X.max(0).tolist(),optimizer=opt)


def decide(rep,model):
    X=rep[model['key']];score=X/np.asarray(model['scale'])@np.asarray(model['w'])-model['price']*rep['counts']
    score[0]=0.;choice=int(np.argmax(score))
    outside=(X<np.array(model['train_min'])-1e-9)|(X>np.array(model['train_max'])+1e-9)
    return choice,dict(scores=score.tolist(),out_of_train_range_fraction=float(outside.mean()),
                       selected_out_of_train_range_fraction=float(outside[choice].mean()))
