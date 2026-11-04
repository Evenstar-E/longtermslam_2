"""Run paired geometric-model intervention, seal every prediction before scoring."""
import time,pickle
from pathlib import Path
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import prepare
from scripts.run_completion_compatible_anchor_v1 import ordinary_assignment
from inspection_decisions.completion_compatible_anchor_v1 import global_selection
from inspection_decisions.first_return_material_surface_v1 import build_patches,bidirectional,historical_ray,material_envelope


def main():
    cfg=read('configs/first_return_material_surface_v1.json');root=Path('results/first_return_material_surface_v1');root.mkdir(exist_ok=False)
    save(root/'config.json',cfg)
    sources=[Path(__file__),Path('inspection_decisions/first_return_material_surface_v1.py'),Path('scripts/first_return_surface_scene_v1.py'),
             Path('scripts/completion_catalog_scene_v1.py'),Path('inspection_decisions/completion_compatible_anchor_v1.py'),
             Path('scripts/run_completion_compatible_anchor_v1.py'),Path('configs/first_return_material_surface_v1.json'),
             Path('research/first_return_material_surface_contract_v1_2026-10-24.md')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'uses_evaluation_truth_in_inference':False,'known_history_frames_and_slots':True,
                             'familiar_DEV_layouts':True,'full_generated_geometry_only_in_generator_evaluator':True})
    runs=[];start=time.perf_counter()
    for mode in cfg['geometry_modes']:
        for seed in cfg['seeds']:
            for hv in cfg['history_views']:
                tag=f'{mode}_s{seed}_h{hv}';dest=root/tag;dest.mkdir();stamp=time.perf_counter()
                history,branches,anchors,frames,truth=prepare(seed,hv,cfg,mode)
                with (dest/'evaluation_only.pkl').open('wb') as f:pickle.dump(truth,f)
                del truth,history
                T0=np.array([next(b['T'] for b in bb[0] if b['slot']==i and b['code']==0) for i,bb in enumerate(branches)])
                slots=T0[:,:3,3];groups=np.arange(len(slots))%4
                patches=[build_patches(ff,slots,groups,cfg) for ff in frames]
                rays=[historical_ray(a,T0[i],frames[0],cfg) for i,a in enumerate(anchors)]
                with (dest/'inference.pkl').open('wb') as f:pickle.dump({'branches':branches,'anchors':anchors,'historical_rays':rays,'T0':T0},f)
                for t,ff in enumerate(frames):
                    for v,frame in enumerate(ff):np.savez_compressed(dest/f'frame_{t}_{v}.npz',**{k:frame[k] for k in ['rgb','depth','C']})
                records=[];mask_outputs={};predictions={'ordinary_nn':ordinary_assignment(branches)}
                masks={'point_visibility':[],'first_return_surface':[]};cache={};diagnostics=[]
                def check(i,s,t,bs,bt,surface):
                    slot_s=i if s==0 else bs['slot'];slot_t=i if t==0 else bt['slot']
                    Ts=T0[i] if s==0 else bs['T'];Tt=T0[i] if t==0 else bt['T']
                    key=(s,t,slot_s,slot_t,-1 if s==0 else bs['code'],-1 if t==0 else bt['code'],surface)
                    if key not in cache:
                        cache[key]=bidirectional(patches[s][slot_s],patches[t][slot_t],Ts,Tt,frames[s],frames[t],cfg,surface)
                    return cache[key]
                for i,bb in enumerate(branches):
                    rec={'asset':i,'methods':{}}
                    for method,surface in [('point_visibility',False),('first_return_surface',True)]:
                        unary=[]
                        for t in [1,2]:unary.append(np.array([check(i,0,t,None,b,surface)['ok'] for b in bb[t-1]]))
                        mask=unary[0][:,None]&unary[1][None,:];tested=int(mask.sum())
                        for k,l in np.argwhere(mask):
                            if not check(i,1,2,bb[0][k],bb[1][l],surface)['ok']:mask[k,l]=False
                        masks[method].append(mask);mask_outputs[f'{method}_{i}']=mask
                        rec['methods'][method]={'history_survivors':[int(u.sum()) for u in unary],'current_pairs_tested':tested,
                            'retained_pairs':int(mask.sum()),'material_envelope':material_envelope(bb,anchors[i],rays[i],mask,cfg)}
                    records.append(rec)
                    if i%6==0:print('PATCH',tag,i,{m:r['retained_pairs'] for m,r in rec['methods'].items()},flush=True)
                for method in masks:
                    if all(mask.any() for mask in masks[method]):predictions[method]=global_selection(branches,masks[method],cfg['global_time_limit_s'])
                    else:predictions[method]={'status':2,'selection':None,'elapsed_s':0.,'objective':None,'reason':'at_least_one_asset_has_no_feasible_pair'}
                save(dest/'predictions.json',predictions);save(dest/'records.json',records)
                save(dest/'visibility_checks.json',[{'key':list(k),**v} for k,v in cache.items()])
                np.savez_compressed(dest/'feasible_masks.npz',**mask_outputs)
                run={'scene':tag,'geometry':mode,'seed':seed,'history_views':hv,'elapsed_s':time.perf_counter()-stamp,
                     'checks':len(cache),'global_status':{m:p['status'] for m,p in predictions.items()},
                     'vertices':[[len(p.vertices) for p in pp] for pp in patches],'triangles':[[len(p.triangles) for p in pp] for pp in patches]}
                runs.append(run);save(root/'runs.json',runs);print('DONE',tag,run['elapsed_s'],run['global_status'],flush=True)
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
         'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evaluation_only.pkl'}})
    print('SEALED',flush=True)


if __name__=='__main__':main()
