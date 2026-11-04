from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha

def main():
    base=Path('results/completion_compatible_anchor_v1');checks=0
    old=read(base/'prediction_seal.json')
    for p,h in old['sha256'].items():assert sha(base/p)==h;checks+=1
    for name in ['history_conditioned_completion_evidence_v1','history_conditioned_completion_ratio_v2']:
        root=Path('results')/name;seal=read(root/'prediction_seal.json');assert seal['complete']
        for p,h in seal['sha256'].items():assert sha(root/p)==h;checks+=1
        for p,h in read(root/'manifest.json')['sources'].items():assert sha(p)==h;checks+=1
        cfg=read(root/'config.json')
        for run in read(root/'runs.json'):
            scene=run['scene'];pred=read(root/scene/'predictions.json');scores=np.load(root/scene/'scores.npz')
            masks=np.load(base/scene/'feasible_masks.npz')
            with (base/scene/'inference.pkl').open('rb') as f:p=pickle.load(f)
            n=len(p['history'])
            for method in cfg['methods']:
                output=pred[method];assert output['status']==0 and len(output['selection'])==n;checks+=1
                assert len({i for i,_,_ in output['selection']})==n
                for t in [0,1]:assert len({p['branches'][i][t][(k,l)[t]]['slot'] for i,k,l in output['selection']})==n;checks+=1
                for i,k,l in output['selection']:
                    assert masks[f'full_{i}'][k,l] and np.isfinite(scores[f'{method}_{i}'][k,l]);checks+=1
                for i in range(n):
                    array=scores[f'{method}_{i}'];mask=masks[f'full_{i}']
                    assert np.isneginf(array[~mask]).all();checks+=1
                    if name.endswith('v1'):assert np.max(array[mask])<=1e-8;checks+=1
        save(root/'audit.json',{'passed':True,'cumulative_checks':checks,'all_global_solves_optimal':True,
                                'all_predictions_use_same_full_mask':True,'sealed_inputs_sources_unchanged':True})
        analysis=[Path('scripts/score_history_conditioned_completion_evidence_v1.py'),Path(__file__)]
        if name.endswith('v1'):
            analysis += [Path('scripts/benchmark_history_conditioned_completion_evidence_v1.py'),
                         Path('scripts/diagnose_history_conditioned_completion_evidence_v1.py'),Path('scripts/diagnose_completion_cardinality_bias_v1.py')]
        else:analysis.append(Path('scripts/score_history_conditioned_completion_ratio_v2.py'))
        save(root/'analysis_sources.json',{str(p):sha(p) for p in analysis})
        save(root/'evidence_index.json',{'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evidence_index.json'}})
    print({'passed':True,'checks':checks})

if __name__=='__main__':main()
