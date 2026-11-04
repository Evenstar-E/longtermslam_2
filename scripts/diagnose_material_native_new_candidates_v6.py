"""Post-seal counterfactuals for the two newly qualified native-pool cases."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
import torch
from inspection_decisions import textured_material_matching_v1 as old
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.material_correspondence_distribution_v3 import observed_groups
from inspection_decisions.material_correspondence_inference_v3 import PairScore,inverse
from inspection_decisions.material_correspondence_mode_graph_v6 import unpack_edges
from scripts.run_material_native_modes_v6 import NativeScore
from scripts.score_material_native_modes_v6 import physical_error
from scripts.score_textured_material_comparison_v1 import material_anchors
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    torch.set_num_threads(1);cfg=json.loads(Path('configs/material_native_modes_v6.json').read_text(encoding='utf-8'));root=Path(cfg['output']);proto=json.loads((root/'frozen_protocol.json').read_text(encoding='utf-8'));assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];base=proto['base_config'];bcfg=proto['legacy_config'];records=[];name='validation_000';i=2
    truth=json.loads((Path(base['dataset'])/name/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(Path(base['dataset'])/name);poses=np.array(truth['poses']);perm=np.array(truth['permutations']);j,k=map(int,perm[1:,i]);x=old.apply(poses[0,i],anchors[i]);y1=old.apply(poses[1,i],anchors[i]);y2=old.apply(poses[2,i],anchors[i])
    for v in ['rich','changed']:
        frames=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=base['seed']+e)[0] for e in range(3)];lf=[load_scene_epoch(base['dataset'],name,e,v,shuffle_seed=bcfg['seed']+e)[0] for e in range(3)];ep=[old.extract_epoch(f,bcfg,bcfg['seed']+e) for e,f in enumerate(lf)];groups=[observed_groups(f,base) for f in frames]
        with np.load(Path(cfg['native_root'])/name/f'{v}_native_edges.npz') as f:edges=unpack_edges(dict(f),cfg)
        with np.load(root/name/f'{v}_pair_pools.npz') as f:pool=dict(f)
        with np.load(root/name/f'{v}_native_graph.npz') as f:p=dict(f)
        with np.load(root/name/f'{v}_joint_cache.npz') as f:cache=dict(f)
        scorers={}
        def score(pe,a,b,T):
            e,f=[(0,1),(0,2),(1,2)][pe];key=pe,a,b
            if key not in scorers:
                A,B=ep[e][a],ep[f][b];scorers[key]=NativeScore(PairScore(A,B,old.matches(A,B,bcfg),groups[e][a],groups[f][b],{},frames[f],b,base),edges[pe][a*24+b])
            return scorers[key].score(T)
        def objective(a,b,c,diagnose=False):
            n1,n2=map(int,[pool['counts'][0,a,b],pool['counts'][1,a,c]]);A=pool['transforms'][0,a,b,:n1][pool['family_masks'][2,0,a,b,:n1]];B=pool['transforms'][1,a,c,:n2][pool['family_masks'][2,1,a,c,:n2]]
            T1=np.repeat(A,len(B),axis=0);T2=np.tile(B,(len(A),1,1));values=(score(0,a,b,A)[:,None]+score(1,a,c,B)[None]+.5*score(2,b,c,T2@inverse(T1)).reshape(len(A),len(B))).ravel()
            if not diagnose:return float(values.min())
            errors=np.array([physical_error(G1,G2,x,y1,y2) for G1,G2 in zip(T1,T2)]);good=errors<=.05;bestgood=int(np.argmin(np.where(good,values,np.inf)));best=int(values.argmin());assert good.any()
            return dict(best_cost=float(values[best]),best_cost_material_error_m=float(errors[best]),qualified_min_cost=float(values[bestgood]),qualified_error_m=float(errors[bestgood]),qualified_pair_rank=int(1+(values<values[bestgood]).sum()),total_pose_pairs=len(values),qualified_pose_pairs=int(good.sum()))
        state=p['target_slots'].copy();forced=state.copy()
        for e,t in enumerate([j,k]):
            owner=int(np.flatnonzero(forced[:,e]==t)[0]);forced[[i,owner],e]=forced[[owner,i],e]
        affected=np.flatnonzero((state!=forced).any(1));forcedcost=float(p['objectives'].sum())
        for a in affected:forcedcost+=objective(int(a),*map(int,forced[a]))-float(p['objectives'][a])
        records.append(dict(scene=name,variant=v,asset=i,diagnostic_only=True,true_identity=[j,k],selected_identity=state[i].tolist(),true_tuple_visited=bool(((cache['triples']==[i,j,k]).all(1)).any()),
            selected_global_cost=float(p['objectives'].sum()),forced_legal_global_cost=forcedcost,forced_legal_cost_delta=forcedcost-float(p['objectives'].sum()),affected_assets=affected.tolist(),true_tuple_pose_ranking=objective(i,j,k,True)))
    save(root/'new_candidate_counterfactuals.json',dict(source_sha=sha(__file__),postseal_diagnostic_only=True,predictions_unchanged=True,scope='New oracle cases selected after scores, explanatory counterfactual only; not independent confirmation or a method',records=records));print(json.dumps(records),flush=True)


if __name__=='__main__':main()
