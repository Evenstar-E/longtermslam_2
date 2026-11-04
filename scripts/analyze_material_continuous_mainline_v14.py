"""Post-seal distinguish missing candidates, wrong objective, identity search."""
import csv,json
from pathlib import Path
import numpy as np
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,METHODS,load
from scripts.score_textured_material_comparison_v1 import material_anchors,write_csv
from scripts.score_material_native_modes_v6 import physical_error
from inspection_decisions.textured_material_matching_v1 import apply
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    c=configuration();root=Path(c['output']);seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];scores=json.loads((root/'scores.json').read_text(encoding='utf-8'))
    for p,h in seal['sources'].items():assert sha(p)==h
    ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);anchors=material_anchors(scene)
    rows=list(csv.DictReader((root/'task_metrics.csv').open(encoding='utf-8-sig')));lookup={(r['variant'],r['method'],int(r['asset'])):r for r in rows};out=[];fitrows=[]
    for v in c['variants']:
        cache=load(root/f'{v}_cache.npz');payload=load(root/f'{v}_proposals.npz');fits=load(root/f'{v}_fits.npz');ql={tuple(t):q for q,t in enumerate(cache['triples'])}
        for i,a in enumerate(anchors):
            j,k=map(int,perm[1:,i]);q=ql.get((i,j,k));errors=None
            x=apply(poses[0,i],a);y1=apply(poses[1,i],a);y2=apply(poses[2,i],a)
            if q is not None:
                U=payload[f'{q}_transforms'];errors=np.array([physical_error(*T,x,y1,y2) for T in U]);values=payload[f'{q}_values']
                for col in [3,6]:
                    initial=fits[f'{q}_{col}_initial'];end=fits[f'{q}_{col}_T'];path=fits[f'{q}_{col}_path']
                    fitrows.append(dict(variant=v,asset=i,column=col,initial_error_m=physical_error(*initial,x,y1,y2),final_error_m=physical_error(*end,x,y1,y2),energy_decrease=float(path[0]-path[-1]),steps=len(path)-1))
            for col,m in enumerate(METHODS[1:]):
                task=lookup[v,m,i];qualified=int(task['qualified']);both=int(task['both_ids']);best_error=float(errors.min()) if errors is not None else None;pool_qualified=errors is not None and best_error<=.05;local_h=int(values[:,col].argmin()) if errors is not None else None
                if qualified:category='success'
                elif q is None:category='true_identity_not_visited'
                elif not pool_qualified:category='no_qualified_pose_in_true_identity_pool'
                elif both:category='objective_prefers_bad_pose_at_correct_identity'
                else:category='identity_assignment_failed_despite_qualified_pose'
                qual_value=float(values[errors<=.05,col].min()) if pool_qualified else None;best_value=float(values[local_h,col]) if local_h is not None else None
                out.append(dict(variant=v,method=m,asset=i,rear=int(perm[2,i]%3==0),category=category,selected_both_ids=both,selected_error_m=float(task['worst_error_m']),pool_best_error_m=best_error,local_best_error_m=float(errors[local_h]) if errors is not None else None,qualified_pose_best_objective=qual_value,local_best_objective=best_value,qualified_objective_gap=qual_value-best_value if pool_qualified else None))
    write_csv(root/'failure_attribution.csv',out);write_csv(root/'true_identity_fit_effects.csv',fitrows);summary=[]
    for v in c['variants']:
        for m in METHODS[1:]:
            rr=[r for r in out if r['variant']==v and r['method']==m];summary.append(dict(variant=v,method=m,categories={k:sum(r['category']==k for r in rr) for k in sorted(set(r['category'] for r in rr))},rear_categories={k:sum(r['category']==k for r in rr if r['rear']) for k in sorted(set(r['category'] for r in rr if r['rear']))}))
    support=json.loads((root/'continuous_support_summary.json').read_text(encoding='utf-8'));shared=[r for r in scores['comparisons'] if r['candidate']=='material_track_shared'];gain=sum(len(r['gained']) for r in shared);lost=sum(len(r['lost']) for r in shared)
    save(root/'mainline_decision.json',dict(complete=True,primary_rows=scores['primary_rows'],failure_summary=summary,shared_vs_same_information_track=dict(gain=gain,lost=lost,net=gain-lost,by_variant=shared),preserved_source_material_edges=support['all512_rear_near_material_edges'],scope='Categories depend on frozen finite visited identity/pose pool, not global impossibility. Identity failure may include scoring and finite assignment search. True-identity fit effects are GT-filtered diagnostics, not predictions. Four appearances correlated, one seen TRAIN. No author-system benchmark or novelty confirmation.'))
    print(json.dumps(dict(shared_gained=gain,shared_lost=lost,failures=[r for r in summary if r['method'] in ['continuous_pair','material_track_shared']]),indent=2))


if __name__=='__main__':main()
