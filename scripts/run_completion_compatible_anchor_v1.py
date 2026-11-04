"""Build complete inference inputs, solve without evaluation truth, seal first."""
from pathlib import Path
import json,time,pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.completion_catalog_scene_v1 import prepare
from inspection_decisions.completion_compatible_anchor_v1 import (
    conflict,enumerate_pairs,pair_displacements,envelope,lazy_envelope,
    explicit_shape_feasible,global_selection)


def ordinary_assignment(all_branches):
    n=len(all_branches);chosen=np.zeros((n,2),int)
    for t in [0,1]:
        costs=np.full((n,n),1e6);indices=np.full((n,n),-1,int)
        for i,bb in enumerate(all_branches):
            for k,b in enumerate(bb[t]):
                if b['cost']<costs[i,b['slot']]:costs[i,b['slot']]=b['cost'];indices[i,b['slot']]=k
        rr,cc=linear_sum_assignment(costs)
        chosen[rr,t]=indices[rr,cc];assert (chosen[:,t]>=0).all()
    return {'status':0,'selection':[(i,int(k),int(l)) for i,(k,l) in enumerate(chosen)],'elapsed_s':None,
            'objective':float(sum(all_branches[i][0][k]['cost']+all_branches[i][1][l]['cost'] for i,(k,l) in enumerate(chosen)))}


def main():
    root=Path('results/completion_compatible_anchor_v1');root.mkdir(exist_ok=False)
    cfg=read('configs/completion_compatible_anchor_v1.json');save(root/'config.json',cfg)
    sources=[Path(__file__),Path('scripts/completion_catalog_scene_v1.py'),Path('inspection_decisions/completion_compatible_anchor_v1.py'),
             Path('configs/completion_compatible_anchor_v1.json'),Path('research/completion_compatible_anchor_contract_v1_2026-10-24.md')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'GT_used_in_inference':False,
                             'generator_knows_GT':True,'finite_catalog_includes_generated_motion_by_design':True})
    start=time.perf_counter();runs=[]
    for seed in cfg['seeds']:
        for hv in cfg['history_views']:
            tag=f's{seed}_h{hv}';dest=root/tag;dest.mkdir();began=time.perf_counter()
            history,branches,anchors,frames,truth=prepare(seed,hv,cfg)
            # Full shape and physical identities are saved only in evaluation file.
            with (dest/'evaluation_only.pkl').open('wb') as f:pickle.dump(truth,f)
            del truth
            with (dest/'inference.pkl').open('wb') as f:pickle.dump({'history':history,'branches':branches,'anchors':anchors},f)
            for t,ff in enumerate(frames):
                for v,frame in enumerate(ff):np.savez_compressed(dest/f'frame_{t}_{v}.npz',**{k:frame[k] for k in ['rgb','depth','C']})
            pairs=[];full=[];records=[];controls=[];mechanisms=[]
            for i,(h,bb,a) in enumerate(zip(history,branches,anchors)):
                displacement=pair_displacements(*bb,a)
                stamp=time.perf_counter();pm,fm,example=enumerate_pairs(h,bb);elapsed=time.perf_counter()-stamp
                pairs.append(pm);full.append(fm)
                pair_env=envelope(displacement,pm);full_env=envelope(displacement,fm)
                lazy=lazy_envelope(h,bb,displacement)
                assert lazy['empty']==full_env['empty']
                if not lazy['empty']:
                    assert np.allclose(lazy['lower'],full_env['lower'],atol=1e-12)
                    assert np.allclose(lazy['upper'],full_env['upper'],atol=1e-12)
                # Prespecified controls: first branch, middle branch, cost-best pair.
                candidates=[(0,0),(len(bb[0])//2,len(bb[1])//2),
                            (int(np.argmin([b['cost'] for b in bb[0]])),int(np.argmin([b['cost'] for b in bb[1]])))]
                for k,l in candidates:
                    gg=[h,bb[0][k],bb[1][l]];expected=conflict(gg) is None
                    control=explicit_shape_feasible(gg)
                    assert control['feasible'] is None or control['feasible']==expected
                    controls.append({'asset':i,'k1':k,'k2':l,'eliminated_feasible':expected,**control})
                records.append({'asset':i,'pairwise_pairs':int(pm.sum()),'full_pairs':int(fm.sum()),
                                'joint_only_rejections':int((pm&~fm).sum()),'pairwise_envelope':pair_env,
                                'full_envelope':full_env,'lazy':lazy,'enumeration_s':elapsed,
                                'history_hit_rays':len(h['bands'])})
                if example:mechanisms.append({'asset':i,**example})
                if i%6==0:print('ASSET',tag,i,'pair/full',int(pm.sum()),int(fm.sum()),'lazy',lazy['checks'],flush=True)
            predictions={'plain_nn':ordinary_assignment(branches),
                         'pairwise':global_selection(branches,pairs,cfg['global_time_limit_s']),
                         'full_shared':global_selection(branches,full,cfg['global_time_limit_s'])}
            save(dest/'predictions.json',predictions);save(dest/'records.json',records)
            save(dest/'explicit_controls.json',controls);save(dest/'joint_contradictions.json',mechanisms)
            np.savez_compressed(dest/'feasible_masks.npz',**{f'{name}_{i}':mask for name,mm in [('pair',pairs),('full',full)] for i,mask in enumerate(mm)})
            run={'scene':tag,'elapsed_s':time.perf_counter()-began,'assets':len(records),
                 'joint_only_rejections':sum(r['joint_only_rejections'] for r in records),
                 'explicit_controls':len(controls),'unresolved_controls':sum(c['feasible'] is None for c in controls),
                 'global_status':{m:p['status'] for m,p in predictions.items()},
                 'lazy_checks':sum(r['lazy']['checks'] for r in records),
                 'complete_pairs':sum(r['lazy']['full_pool_size'] for r in records)}
            runs.append(run);save(root/'runs.json',runs);print('DONE',run,flush=True)
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
         'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evaluation_only.pkl'}})
    print('SEALED',flush=True)


if __name__=='__main__':main()
