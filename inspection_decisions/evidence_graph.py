"""Observed-evidence graph. Pure input builder: no filesystem/evaluator access.

Edges are geometry/posterior approximations, never observed future visibility
or truth correspondence. IDs are local indices, not semantic asset identities.
"""
import copy
import numpy as np
from scipy.spatial import cKDTree
from comparative_evidence.support_forecast_v5 import prepare_support, visible_points
from comparative_evidence.interface import shortest_path
from inspection_decisions.costs import report_risk

NODE_FEATURES = {
    'asset':['delta_x','delta_y','delta_z','sd_x','sd_y','sd_z','current_support','association_margin','reported_code','declared_risk'],
    'patch':['current_epoch','centroid_x','centroid_y','centroid_z','normal_x','normal_y','normal_z','point_fraction','missing_opposite_support'],
    'nuisance':['frame_type','observed','mean0','mean1','mean2','mean3','mean4','mean5','mean6','var0','var1','var2','var3','var4','var5','var6'],
    'action':['stop','current_epoch','camera_x','camera_y','camera_z','forward_x','forward_y','forward_z','travel_m','return_delta_m','estimated_payload_bytes','acquisition_s','observed_group'],
}
EDGE_FEATURES = {
    'patch_asset':['point_fraction'],
    'patch_nuisance':[f'J{i}' for i in range(21)],
    'cross_epoch_support':['mean_nearest_distance','normal_overlap_kernel'],
    'nuisance_coupling':[f'shared_cov{i}' for i in range(9)]+[f'joint_cov{i}' for i in range(9)]+['shared_block_count'],
    'action_nuisance':['weight','observed'],
    'action_patch':['held_frustum_fraction','missing_support_fraction','opposite_epoch','mean_range_m'],
    'action_pair':['cross_epoch','held_asset_overlap','held_complement_mass','shared_latent_count'],
}


def legal_actions(state):
    result=[]
    if len(state['history_read'])<state['limits']['history']:
        result += [('history',v) for v in range(12) if v not in state['history_read']]
    if len(state['current_observed'])-1<state['limits']['current']:
        result += [('current',v) for v in range(12) if v not in state['current_observed']]
    return result


def nuisance_terms(epoch,view,meta):
    row=meta[epoch][view]
    if meta['error_model']['structure']=='epoch_shared_plus_frame':
        return [(('group',row['nuisance_group']),.7),(('frame',row['token']),.3)]
    return [(('group',row['nuisance_group']),1.)]


def build_graph(fit,history,current,meta,state,prediction):
    history=sorted(history,key=lambda o:o['view_id']);current=sorted(current,key=lambda o:o['view_id'])
    nodes=[];edges=[];audit_patches=[]
    def node(kind,features):
        values=np.asarray(features,float).ravel()
        if len(values)!=len(NODE_FEATURES[kind]) or not np.isfinite(values).all():raise ValueError(kind)
        k=len(nodes);nodes.append(dict(id=k,kind=kind,features=values.tolist()));return k
    def edge(kind,source,target,features):
        values=np.asarray(features,float).ravel()
        if len(values)!=len(EDGE_FEATURES[kind]) or not np.isfinite(values).all():raise ValueError(kind)
        edges.append(dict(kind=kind,source=source,target=target,features=values.tolist()))
    n=len(fit['references']);start=fit['nuisance_start'];C=np.asarray(fit['covariance']);Cu=C[start:,start:]
    asset_nodes=[]
    for j,a in enumerate(prediction['assets']):
        d=fit['mean'][fit['delta_start']+3*j:fit['delta_start']+3*j+3]
        sd=np.sqrt(np.maximum(np.diag(a['covariance']),0))
        asset_nodes.append(node('asset',list(d)+list(sd)+[a['current_support'],a['association_margin_m'],a['label'],.35 if a['label']<0 else a['decision_risk']]))
    nuisance_nodes={};known_keys={k:i for i,k in enumerate(fit['nuisance_keys'])}
    def nuisance(key):
        if key not in nuisance_nodes:
            observed=key in known_keys
            if observed:
                col=start+7*known_keys[key];mean=fit['mean'][col:col+7];var=np.diag(C[col:col+7,col:col+7])
            else:mean=np.zeros(7);var=np.ones(7)/3
            nuisance_nodes[key]=node('nuisance',[int(key[0]=='frame'),int(observed)]+list(mean)+list(var))
        return nuisance_nodes[key]
    for key in fit['nuisance_keys']:nuisance(key)
    cache=prepare_support(fit,history,current,meta);patches=[]
    for epoch in ('history','current'):
        opposite='current' if epoch=='history' else 'history'
        for patch in cache[epoch]:
            j=patch['asset'];centre=patch['points'].mean(0);normal=patch['normals'].mean(0)
            k=node('patch',[int(epoch=='current')]+list(centre)+list(normal)+[len(patch['points'])/80,float(patch['missing_'+opposite].mean())])
            B=np.zeros((3,len(Cu)));active=[]
            for col,weight in patch['active']:
                J=weight*patch['jacobians'].mean(0);local=col-start
                B[:,local:local+7]=J;key=fit['nuisance_keys'][local//7];active.append(local)
                edge('patch_nuisance',k,nuisance(key),J.ravel())
            edge('patch_asset',k,asset_nodes[j],[len(patch['points'])/80])
            patches.append(dict(node=k,epoch=epoch,patch=patch,B=B,active=active))
            audit_patches.append(dict(node=k,token=patch['token'],anonymous_reference=j,epoch=epoch))
    for i,left in enumerate(patches):
        for right in patches[i+1:]:
            Bi,Bj=left['B'],right['B'];shared=set(left['active'])&set(right['active'])
            joint=Bi@Cu@Bj.T;common=np.zeros((3,3))
            for col in shared:common+=Bi[:,col:col+7]@Cu[col:col+7,col:col+7]@Bj[:,col:col+7].T
            for src,dst,A,B in ((left,right,common,joint),(right,left,common.T,joint.T)):
                edge('nuisance_coupling',src['node'],dst['node'],list(A.ravel())+list(B.ravel())+[len(shared)])
            p,q=left['patch'],right['patch']
            if left['epoch']!=right['epoch'] and p['asset']==q['asset']:
                for src,dst in ((left,right),(right,left)):
                    p,q=src['patch'],dst['patch'];distance,ix=cKDTree(q['points']).query(p['points'])
                    agreement=np.maximum(0,np.einsum('ij,ij->i',p['normals'],q['normals'][ix]))**2
                    edge('cross_epoch_support',src['node'],dst['node'],[distance.mean(),np.mean(np.exp(-.5*(distance/.08)**2)*agreement)])
    actions=[];action_records=[]
    held_sizes=[e['payload_bytes'] for e in state['events'] if e.get('payload_bytes',0)>0]
    estimated_bytes=float(np.mean(held_sizes)) if held_sizes else 0.
    current_return=shortest_path(meta['edges'],state['position'],0)[0]
    for epoch,view in legal_actions(state):
        row=meta[epoch][view];T=np.asarray(row['nominal_T']);terms=nuisance_terms(epoch,view,meta)
        travel=shortest_path(meta['edges'],state['position'],view)[0] if epoch=='current' else 0.
        return_delta=shortest_path(meta['edges'],view,0)[0]-current_return if epoch=='current' else 0.
        known_group=int(terms[0][0] in known_keys)
        k=node('action',[0,int(epoch=='current')]+list(T[:3,3])+list(T[:3,2])+[travel,return_delta,estimated_bytes,meta['sensor']['acquisition_s'] if epoch=='current' else 0.,known_group])
        for key,weight in terms:edge('action_nuisance',k,nuisance(key),[weight,int(key in known_keys)])
        footprint=np.zeros(n);complement=np.zeros(n)
        for pp in patches:
            p=pp['patch'];j=p['asset'];world=p['points']+fit['mean'][3*j:3*j+3]
            if epoch=='current':world=world+fit['mean'][fit['delta_start']+3*j:fit['delta_start']+3*j+3]
            valid=visible_points(world,p['normals'],T,meta['sensor']);fraction=float(valid.mean())
            # prepare_support measures the opposite epoch only. A held patch
            # already supports its own epoch; do not invent an absent key/value.
            missing=float(p['missing_'+epoch][valid].mean()) if valid.any() and pp['epoch']!=epoch else 0.
            edge('action_patch',k,pp['node'],[fraction,missing,int(pp['epoch']!=epoch),float(np.linalg.norm(world-T[:3,3],axis=1).mean())])
            if pp['epoch']!=epoch:
                footprint[j]=max(footprint[j],fraction);complement[j]=max(complement[j],fraction*missing)
        actions.append(dict(node=k,epoch=epoch,view=view,footprint=footprint,complement=complement,keys={key for key,weight in terms}))
        action_records.append(dict(node=k,epoch=epoch,view=view))
    for i,left in enumerate(actions):
        for right in actions[i+1:]:
            overlap=float(np.minimum(left['footprint'],right['footprint']).mean())
            # A held-geometry interaction FEATURE, never a measured pair gain.
            complement=float(np.minimum(left['complement'],right['complement']).mean()) if left['epoch']!=right['epoch'] else 0.
            features=[int(left['epoch']!=right['epoch']),overlap,complement,len(left['keys']&right['keys'])]
            edge('action_pair',left['node'],right['node'],features);edge('action_pair',right['node'],left['node'],features)
    stop=node('action',[1,0]+[0.]*11);action_records.append(dict(node=stop,epoch='stop',view=-1))
    return dict(schema='observed_evidence_graph_v1',node_feature_names=NODE_FEATURES,edge_feature_names=EDGE_FEATURES,
        nodes=nodes,edges=edges,actions=action_records,
        context=dict(history_remaining=state['limits']['history']-len(state['history_read']),
            current_remaining=state['limits']['current']-len(state['current_observed'])+1,
            return_reserve_m=current_return,mean_declared_report_risk=report_risk(prediction)),
        audit=dict(source_tokens=sorted({o['token'] for o in history+current}),patch_sources=audit_patches,
            nuisance_keys=[dict(node=k,type=key[0],name=key[1],observed=key in known_keys) for key,k in nuisance_nodes.items()],
            limits=['No filesystem/truth/future-image reads','Known frustum/normal proxy only; future visibility unknown',
                    'Shared/joint blocks reflect approximate source nuisance covariance, not full surface uncertainty',
                    'Unknown nuisance nodes use normalized prior I/3; cost bytes estimated from held events',
                    'Pair features are not observed utility; all models get same graph data']))


def without_nuisance_edges(graph):
    result=copy.deepcopy(graph)
    result['edges']=[e for e in graph['edges'] if e['kind'] not in ('nuisance_coupling','patch_nuisance','action_nuisance')]
    result['audit']['ablation']='removed explicit nuisance edges only; same nodes/features/actions'
    return result
