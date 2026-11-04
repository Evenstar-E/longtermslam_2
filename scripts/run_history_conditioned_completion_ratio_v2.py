from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.run_history_conditioned_completion_evidence_v1 import Engine,solve
from inspection_decisions.history_conditioned_completion_evidence_v1 import event


def main():
    base=Path('results/completion_compatible_anchor_v1');old=read(base/'prediction_seal.json')
    for p,h in old['sha256'].items():assert sha(base/p)==h
    root=Path('results/history_conditioned_completion_ratio_v2');root.mkdir(exist_ok=False)
    methods=[f'{prior}_{score}' for prior in ['beta','fixed'] for score in ['raw_joint_ratio','conditional_joint_ratio','conditional_pair_ratio']]
    cfg={'input_root':str(base),'methods':methods,'priors':['beta','fixed'],'particles':[],
         'primary':'beta_conditional_joint_ratio versus beta_conditional_pair_ratio','GT_previously_scored_DEV':True}
    save(root/'config.json',cfg)
    sources=[Path(__file__),Path('scripts/run_history_conditioned_completion_evidence_v1.py'),
             Path('inspection_decisions/history_conditioned_completion_evidence_v1.py'),
             Path('research/history_conditioned_completion_ratio_contract_v2_2026-10-24.md')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inference_GT_read':False,'base_seal':sha(base/'prediction_seal.json')})
    engine=Engine();start=time.perf_counter();runs=[]
    def logs(groups,prior):
        spec=event(groups)
        if spec is None:return -np.inf,-np.inf
        value=engine.evaluate(spec,prior)
        free=-np.log1p(spec['free_count']) if prior=='beta' else -spec['free_count']*np.log(2.)
        return value,value-free
    for run in read(base/'runs.json'):
        tag=run['scene'];dest=root/tag;dest.mkdir();began=time.perf_counter()
        with (base/tag/'inference.pkl').open('rb') as f:p=pickle.load(f)
        disk=np.load(base/tag/'feasible_masks.npz');masks=[disk[f'full_{i}'] for i in range(len(p['history']))]
        scores={m:[] for m in methods}
        for h,bb,mask in zip(p['history'],p['branches'],masks):
            arrays={m:np.full(mask.shape,-np.inf) for m in methods}
            for prior in ['beta','fixed']:
                eh,ch=logs([h],prior)
                singles=[[logs([b],prior) for b in bt] for bt in bb]
                shared=[[logs([h,b],prior) for b in bt] for bt in bb]
                for k,l in zip(*np.nonzero(mask)):
                    full,conditional=logs([h,bb[0][k],bb[1][l]],prior)
                    current,current_conditional=logs([bb[0][k],bb[1][l]],prior)
                    arrays[prior+'_raw_joint_ratio'][k,l]=full-eh-current
                    arrays[prior+'_conditional_joint_ratio'][k,l]=conditional-ch-current_conditional
                    arrays[prior+'_conditional_pair_ratio'][k,l]=shared[0][k][1]+shared[1][l][1]-2*ch-singles[0][k][1]-singles[1][l][1]
            for m in methods:scores[m].append(arrays[m])
        pred={m:solve(p['branches'],masks,scores[m],20.) for m in methods}
        save(dest/'predictions.json',pred);save(dest/'sampling_errors.json',[])
        np.savez_compressed(dest/'scores.npz',**{f'{m}_{i}':a for m in methods for i,a in enumerate(scores[m])})
        record={'scene':tag,'elapsed_s':time.perf_counter()-began,'global_status':{m:r['status'] for m,r in pred.items()}}
        runs.append(record);save(root/'runs.json',runs);print(record,flush=True)
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
         'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('SEALED RATIOS',flush=True)

if __name__=='__main__':main()
