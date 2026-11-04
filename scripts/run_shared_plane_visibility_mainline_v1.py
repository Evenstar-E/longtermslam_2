"""Full finite-pool shared-plane comparison; inference never reads truth files."""
from pathlib import Path
import time,pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.run_completion_compatible_anchor_v1 import ordinary_assignment
from inspection_decisions.completion_compatible_anchor_v1 import global_selection
from inspection_decisions.first_return_material_surface_v1 import material_envelope
from inspection_decisions.shared_plane_visibility_mainline_v1 import history_patches,masks_for_asset,PlaneFeasibility


def numerical_controls(patches,factors,bb,solver):
    records=[]
    if not patches:return records
    pp=sorted(set([0,len(patches)//2]));kk=[]
    for t in [0,1]:
        kk.append(sorted(set([0,len(bb[t])//2,int(np.argmin([b['cost'] for b in bb[t]]))])))
        for k in kk[t]:
            for j in pp:
                ff=[factors[t][k][j]];fast,how=solver.solve(patches[j],ff);explicit,status=solver.solve(patches[j],ff,force_lp=True)
                assert fast==explicit,(t,k,j,fast,explicit)
                records.append({'epoch':t,'branch':k,'patch':j,'fast':fast,'fast_how':how,'explicit':explicit,'explicit_status':status})
    for k,l in [(kk[0][0],kk[1][0]),(len(bb[0])//2,len(bb[1])//2),
                (int(np.argmin([b['cost'] for b in bb[0]])),int(np.argmin([b['cost'] for b in bb[1]])))]:
        for j in pp:
            ff=[factors[0][k][j],factors[1][l][j]];fast,how=solver.solve(patches[j],ff);explicit,status=solver.solve(patches[j],ff,force_lp=True)
            assert fast==explicit,(k,l,j,fast,explicit)
            records.append({'epoch':'joint','branch':[k,l],'patch':j,'fast':fast,'fast_how':how,'explicit':explicit,'explicit_status':status})
    return records


def main():
    cp=Path('configs/shared_plane_visibility_mainline_v1.json');cfg=read(cp);base=Path(cfg['base_root']);base_seal=read(base/'prediction_seal.json')
    for p,h in base_seal['sha256'].items():assert sha(base/p)==h,p
    combined={**read(base/'config.json'),**cfg}
    root=Path('results/shared_plane_visibility_mainline_v1');root.mkdir(exist_ok=False);save(root/'config.json',combined)
    sources=[cp,Path(__file__),Path('inspection_decisions/shared_plane_visibility_mainline_v1.py'),
             Path('inspection_decisions/shared_plane_visibility_v1.py'),Path('inspection_decisions/ray_conditioned_plane_polytope_v1.py'),
             Path('inspection_decisions/first_return_material_surface_v1.py'),Path('inspection_decisions/completion_compatible_anchor_v1.py'),
             Path('scripts/run_completion_compatible_anchor_v1.py'),Path('research/shared_plane_visibility_mainline_contract_v1_2026-10-24.md')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'base_seal_sha256':sha(base/'prediction_seal.json'),
         'base_sealed_inputs':base_seal['sha256'],'uses_evaluation_truth':False,'familiar_DEV_data':True,'source_patches':'history_only'})
    start=time.perf_counter();runs=[]
    for run in read(base/'runs.json'):
        tag=run['scene'];src=base/tag;dest=root/tag;dest.mkdir();stamp=time.perf_counter()
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        frames=[]
        for t in range(3):
            ff=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:ff.append({k:z[k].copy() for k in ['rgb','depth','C']})
            frames.append(ff)
        patches,patch_counts=history_patches(frames[0],inp['T0'][:,:3,3],np.arange(len(inp['T0']))%4,combined)
        with (dest/'history_patches.pkl').open('wb') as f:pickle.dump(patches,f)
        solver=PlaneFeasibility(cfg['lp_tolerance']);records=[];all_masks={m:[] for m in ['nominal_plane','independent_period_planes','shared_period_plane']};controls=[]
        for i,bb in enumerate(inp['branches']):
            began=time.perf_counter();masks,counts,factors,trace,witnesses=masks_for_asset(patches[i],inp['T0'][i],bb,frames[1:],combined,solver)
            ctrl=numerical_controls(patches[i],factors,bb,solver);controls.extend([{'asset':i,**v} for v in ctrl])
            rec={'asset':i,**counts,'elapsed_s':time.perf_counter()-began,'envelopes':{
                m:material_envelope(bb,inp['anchors'][i],inp['historical_rays'][i],mask,combined) for m,mask in masks.items()}}
            for m,mask in masks.items():all_masks[m].append(mask)
            with (dest/f'asset_{i:02d}_factors.pkl').open('wb') as f:pickle.dump({'factors':factors,'unary_trace':trace,'joint_witnesses':witnesses},f)
            records.append(rec)
            if i%3==0:print('PLANE',tag,i,'patches',counts['patches'],'nom/ind/shared',*[counts[k] for k in ['nominal_pairs','independent_pairs','shared_pairs']],
                            'seconds',round(rec['elapsed_s'],3),flush=True)
        predictions={'ordinary_nn':ordinary_assignment(inp['branches'])}
        for m,mm in all_masks.items():
            if all(mask.any() for mask in mm):predictions[m]=global_selection(inp['branches'],mm,cfg['global_time_limit_s'])
            else:predictions[m]={'status':2,'selection':None,'reason':'empty_asset_candidate_set','elapsed_s':0.,'objective':None}
        save(dest/'predictions.json',predictions);save(dest/'records.json',records);save(dest/'numerical_controls.json',controls)
        np.savez_compressed(dest/'feasible_masks.npz',**{f'{m}_{i}':mask for m,mm in all_masks.items() for i,mask in enumerate(mm)})
        result={**{k:run[k] for k in ['scene','geometry','seed','history_views']},'elapsed_s':time.perf_counter()-stamp,'patch_counts':patch_counts,
                'solver_counts':dict(solver.stats),'numerical_controls':len(controls),'global_status':{m:p['status'] for m,p in predictions.items()},
                'joint_only_rejections':sum(r['joint_only_rejections'] for r in records),'active_joint_patch_checks':sum(r['active_joint_patch_checks'] for r in records)}
        runs.append(result);save(root/'runs.json',runs);print('DONE',tag,round(result['elapsed_s'],3),result['global_status'],'extra',result['joint_only_rejections'],flush=True)
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
         'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('SEALED',flush=True)


if __name__=='__main__':main()
