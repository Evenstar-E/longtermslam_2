"""Small, deterministic value learners; no evaluator or filesystem in actors.

All direct/structured learners consume the same lossy observed-graph encoding.
This is kernel regression with explicit value decomposition, not a trained GNN.
"""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from inspection_decisions.evidence_graph import NODE_FEATURES, EDGE_FEATURES


def moments(rows, width):
    x = np.asarray(rows, float).reshape(-1, width)
    return np.r_[len(x), x.mean(0) if len(x) else np.zeros(width),
                 x.std(0) if len(x) else np.zeros(width)]


def encode_options(graph, pairs, cross_cov=True):
    """Only graph numeric fields and predeclared action tuples are consumed."""
    nodes = {n['id']: n for n in graph['nodes']}
    edges = []
    for e in graph['edges']:
        v = np.asarray(e['features'], float).copy()
        if not cross_cov and e['kind'] == 'nuisance_coupling':
            v[:18] = 0
        edges.append((e['kind'], e['source'], e['target'], v))
    glob = []
    for kind, names in NODE_FEATURES.items():
        glob.extend(moments([n['features'] for n in nodes.values() if n['kind']==kind], len(names)))
    for kind, names in EDGE_FEATURES.items():
        glob.extend(moments([v for k,s,t,v in edges if k==kind], len(names)))
    glob.extend(graph['context'][k] for k in ('history_remaining','current_remaining',
                 'return_reserve_m','mean_declared_report_risk'))
    assets = {s:t for k,s,t,v in edges if k=='patch_asset'}
    patches = [n for n in nodes if nodes[n]['kind']=='patch']
    pi = {n:i for i,n in enumerate(patches)}
    cov = np.zeros((len(patches),len(patches),19))
    for k,s,t,v in edges:
        if k=='nuisance_coupling': cov[pi[s],pi[t]] = v
    action_nodes = {(a['epoch'],a['view']):a['node'] for a in graph['actions'] if a['epoch']!='stop'}
    local, weights, coverage = {}, {}, {}
    for action, node in action_nodes.items():
        pe = [(t,v) for k,s,t,v in edges if k=='action_patch' and s==node]
        w = np.zeros(len(patches))
        for p,v in pe: w[pi[p]] = v[0]
        wn = w/max(w.sum(),1e-12)
        patch_mean = sum((wn[pi[p]]*np.asarray(nodes[p]['features']) for p in patches), np.zeros(len(NODE_FEATURES['patch'])))
        asset_mean = sum((wn[pi[p]]*np.asarray(nodes[assets[p]]['features']) for p in patches), np.zeros(len(NODE_FEATURES['asset'])))
        nv = [nodes[t]['features'] for k,s,t,v in edges if k=='action_nuisance' and s==node]
        coupling = np.einsum('i,ijc,j->c',wn,cov,wn)
        local[action] = np.r_[nodes[node]['features'],moments([v for p,v in pe],4),
                              patch_mean,asset_mean,moments(nv,16),coupling]
        weights[action] = wn
        # Coverage proxy known at initial time; no true visibility.
        cover = {}
        for p,v in pe:
            if v[2]>0: cover[assets[p]] = max(cover.get(assets[p],0.),v[0]*v[1])
        coverage[action] = cover
    singles = list(action_nodes)
    options = [()] + [(a,) for a in singles] + [tuple(map(tuple,p)) for p in pairs]
    width = len(next(iter(local.values())))
    X, scores = [], []
    pair_edges = {(s,t):v for k,s,t,v in edges if k=='action_pair'}
    for option in options:
        u = local[option[0]] if option else np.zeros(width)
        v = local[option[1]] if len(option)==2 else np.zeros(width)
        relation = np.zeros(23)
        if len(option)==2:
            a,b = option
            relation = np.r_[pair_edges[(action_nodes[a],action_nodes[b])],
                              np.einsum('i,ijc,j->c',weights[a],cov,weights[b])]
        X.append(np.r_[glob,u,v,relation,len(option)])
        cover = {}
        for a in option:
            for j,z in coverage[a].items(): cover[j] = max(cover.get(j,0.),z)
        scores.append(sum(cover.values()))
    return dict(X=np.asarray(X),options=options,coverage=np.asarray(scores),
                risk=float(graph['context']['mean_declared_report_risk']))


def fit_regressor(X, y, alpha, bandwidth, kernel, scaler):
    mean, scale = scaler
    Z = (X-mean)/scale
    # Count truly varying columns using training inputs only.
    effective = max(int(np.count_nonzero(np.std(Z,axis=0)>1e-12)),1)
    if kernel=='linear': K = Z@Z.T/effective
    else: K = np.exp(-cdist(Z,Z,'sqeuclidean')/(2*bandwidth*effective))
    center = float(np.mean(y))
    coef = solve(K+alpha*np.eye(len(K)),y-center,assume_a='pos')
    return dict(mean=mean,scale=scale,Z=Z,coef=coef,center=center,
                effective=effective,bandwidth=bandwidth,kernel=kernel)


def predict_regressor(model, X):
    Z = (X-model['mean'])/model['scale']
    if model['kernel']=='linear': K = Z@model['Z'].T/model['effective']
    else: K = np.exp(-cdist(Z,model['Z'],'sqeuclidean')/(2*model['bandwidth']*model['effective']))
    return K@model['coef']+model['center']


def fit_value_model(name, X, gain, single_indices, pair_indices, pair_members, alpha, bandwidth):
    """X: families x 31 options x features. Targets are train-only realizations."""
    flat = X[:,1:].reshape(-1,X.shape[-1])
    mean = flat.mean(0); scale = flat.std(0); scale[scale<1e-12] = 1.
    scaler = mean,scale
    model = dict(name=name,alpha=alpha,bandwidth=bandwidth,
                 single_indices=single_indices,pair_indices=pair_indices,pair_members=pair_members)
    if name.startswith('flat'):
        model['direct'] = fit_regressor(flat,gain[:,1:].ravel(),alpha,bandwidth,
                                        'linear' if name=='flat_linear' else 'rbf',scaler)
    else:
        model['single'] = fit_regressor(X[:,single_indices].reshape(-1,X.shape[-1]),
                       gain[:,single_indices].ravel(),alpha,bandwidth,'rbf',scaler)
        interaction = gain[:,pair_indices]-gain[:,[a for a,b in pair_members]]-gain[:,[b for a,b in pair_members]]
        if name=='mean_interaction_rbf': model['constant_interaction'] = float(interaction.mean())
        elif name.startswith('structured'):
            model['interaction'] = fit_regressor(X[:,pair_indices].reshape(-1,X.shape[-1]),
                                                interaction.ravel(),alpha,bandwidth,'rbf',scaler)
    return model


def predict_value(model, X):
    shape = X.shape[:2]; result = np.zeros(shape)
    if 'direct' in model:
        result[:,1:] = predict_regressor(model['direct'],X[:,1:].reshape(-1,X.shape[-1])).reshape(shape[0],-1)
    else:
        singles = model['single_indices']; pairs = model['pair_indices']; members = model['pair_members']
        result[:,singles] = predict_regressor(model['single'],X[:,singles].reshape(-1,X.shape[-1])).reshape(shape[0],-1)
        result[:,pairs] = result[:,[a for a,b in members]]+result[:,[b for a,b in members]]
        if 'constant_interaction' in model: result[:,pairs] += model['constant_interaction']
        if 'interaction' in model:
            result[:,pairs] += predict_regressor(model['interaction'],X[:,pairs].reshape(-1,X.shape[-1])).reshape(shape[0],-1)
    return result


def choose(predicted_gain, counts, price):
    # STOP first, deterministic first maximum, no true outcomes in actor.
    return np.argmax(predicted_gain-price*np.asarray(counts)[None,:],axis=1)
