from pathlib import Path
import pickle,collections
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save

def main():
    root=Path('results/completion_compatible_anchor_v1');rows=[];hist=collections.Counter();examples=[]
    for run in read(root/'runs.json'):
        folder=root/run['scene'];pred=read(folder/'predictions.json');rec=read(folder/'records.json')
        same=pred['pairwise']['selection']==pred['full_shared']['selection']
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        mask=np.load(folder/'feasible_masks.npz');rejects=[]
        for i,(h,bb) in enumerate(zip(p['history'],p['branches'])):
            for g in [h]+bb[0]+bb[1]:
                for b in g['bands']:hist[b.bit_count()]+=1
            for k,l in zip(*np.nonzero(mask[f'pair_{i}']&~mask[f'full_{i}'])):rejects.append((i,int(k),int(l)))
        selected={tuple(x) for x in pred['pairwise']['selection']}
        rows.append({'scene':run['scene'],'full_pairwise_same_selected_branches':same,'joint_only_rejections':len(rejects),
                     'joint_rejections_in_pairwise_selected_solution':sum(x in selected for x in rejects),
                     'local_box_changed_assets':sum(r['pairwise_envelope']!=r['full_envelope'] for r in rec)})
        examples.extend({'scene':run['scene'],**e} for e in read(folder/'joint_contradictions.json'))
    out={'rows':rows,'band_cardinality_histogram_over_model_clauses':dict(sorted(hist.items())),
         'histogram_is_not_independent_observation_count':True,'first_joint_contradiction':examples[0] if examples else None,
         'necessary_condition':'A newly rejected positive clause must contain >=2 cells; singleton clauses cannot create a full-vs-pairwise difference.'}
    save(root/'joint_effect_diagnosis.json',out);print(out)

if __name__=='__main__':main()
