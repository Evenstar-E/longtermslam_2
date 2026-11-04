"""POSTHOC full-task attribution: new scores on EXACT retained V14 pool."""
import json
from pathlib import Path
import numpy as np
from scripts.run_material_surface_distribution_v15b import configuration,METHODS,load
from scripts.run_material_continuous_surface_correspondence_v14c import setup
from scripts.score_textured_material_comparison_v1 import material_anchors,task_rows,write_csv
from scripts.run_textured_material_comparison_v1 import save


def main():
    c=configuration();root=Path(c['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];ic=json.loads(Path(c['input_config']).read_text(encoding='utf-8'));scene=Path(ic['dataset'])/ic['scene'];truth=json.loads((scene/'supervision/scene.json').read_text(encoding='utf-8'));anchors=material_anchors(scene);rows=[];summary=[]
    for v in c['variants']:
        pool=load(root/f'{v}_proposals.npz');cache=load(root/f'{v}_cache.npz');graph=load(root/f'{v}_search_graph.npz');_,_,_,bc,_,_,_,_,_=setup(c,v);lookup={tuple(t):h for h,t in enumerate(cache['triples'])};costs=[];poses=[]
        for q in range(len(cache['triples'])):
            U=pool[f'{q}_transforms'];V=pool[f'{q}_values'];ix=np.flatnonzero(pool[f'{q}_stage']==0);h=ix[V[ix].argmin(0)];costs.append(V[h,np.arange(6)]);poses.append(U[h])
        costs=np.array(costs);poses=np.array(poses);value=np.array([sum((costs[lookup[i,int(j),int(k)]] for i,(j,k) in enumerate(s)),start=np.zeros(6)) for s in graph['states']]);best=value.argmin(0)
        for col,m in enumerate(METHODS[2:]):
            s=graph['states'][best[col]];ix=[lookup[i,int(j),int(k)] for i,(j,k) in enumerate(s)];pred=dict(target_slots=s,transforms=poses[ix,col],triplet_counts=np.zeros(24,int));rr=task_rows(ic['scene'],'train',v,m,pred,truth,anchors,bc);rows+=rr;summary.append(dict(variant=v,method=m,qualified=sum(r['qualified'] for r in rr),both_ids=sum(r['both_ids'] for r in rr)))
    write_csv(root/'retained_pool_ablation_tasks.csv',rows);save(root/'retained_pool_ablation_summary.json',dict(summary=summary,primary_rows=len(rows),scope='POSTHOC all24x4x6 structural counterfactual, designed after main results. Uses ONLY already frozen scores and exact parent V14coupled candidates, fixed full identity states, no new training/transform/score/parameter. No independent confirmation, no GT used to pick restriction percase. Compares representation/scoring transfer versus actual new pose pool contribution.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
