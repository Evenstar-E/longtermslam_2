"""Post-seal candidate recall, explicitly GT-identity conditioned (not task score)."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_data_io_v1 import load_scene_epoch
from inspection_decisions.textured_material_dataset_v1 import inverse
from scripts.evaluate_material_correspondence_repairs_v3 import annotate,visibility
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    v4=json.loads(Path('configs/material_continuous_correspondence_v4.json').read_text(encoding='utf-8'));cfg=json.loads(Path(v4['base_config']).read_text(encoding='utf-8'));root=Path(v4['comparison_root'])
    seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));assert seal['complete'];data=Path(cfg['dataset']);records=[]
    for name in seal['scenes']:
        dest=data/name;truth=json.loads((dest/'supervision/scene.json').read_text(encoding='utf-8'));scene={**truth,'geometry':[np.array(x) for x in truth['geometry']],'poses':np.array(truth['poses']),'occluders':np.array(truth['occluders'])};private={}
        for variant in cfg['variants']:
            frames=[];groups=[]
            for e in range(3):
                f,raw=load_scene_epoch(data,name,e,variant,shuffle_seed=cfg['seed']+e);frames.append(f);groups.append(annotate(f,raw,dest,cfg,private))
            with np.load(root/name/f'{variant}_field_correspondences.npz') as f:correspondences=dict(f)
            for pe,(ea,eb) in enumerate([(0,1),(0,2),(1,2)]):
                for i,a in enumerate(groups[ea]):
                    identity=int(np.bincount(a['identity']).argmax());j=truth['permutations'][eb][identity];b=groups[eb][j]
                    visible=visibility(a,scene,eb,frames[eb]);d2=((a['q'][:,None]-b['q'][None])**2).sum(2)
                    covered=visible&(((a['identity'][:,None]==b['identity'][None])&(d2<=.025**2)).any(1))
                    src=correspondences['source_indices'][pe,i,j];okay=src>=0;src=src[okay];unique=np.unique(src)
                    for method in ['ncc','random','dense','surface']:
                        y=correspondences[method][pe,i,j][okay];q=inverse(y,scene['poses'][eb,identity]);error=np.linalg.norm(q-a['q'][src],axis=1)
                        for query in unique:
                            best=float(error[src==query].min());records.append(dict(scene=name,variant=variant,pair=f'{ea}{eb}',slot=i,query=int(query),method=method,
                                visible=int(visible[query]),original_covered=int(covered[query]),best_of_three_error_m=best,has_25mm_mode=int(best<=.025)))
        print('V4 MODE DIAGNOSTIC',name,flush=True)
    summary=[]
    for variant in cfg['variants']:
        for method in ['ncc','random','dense','surface']:
            for state in ['visible_covered','visible_uncovered','not_visible']:
                rr=[r for r in records if r['variant']==variant and r['method']==method and
                    ((r['visible'] and r['original_covered']) if state=='visible_covered' else (r['visible'] and not r['original_covered']) if state=='visible_uncovered' else not r['visible'])]
                summary.append(dict(variant=variant,method=method,state=state,queries=len(rr),has_25mm_mode=sum(r['has_25mm_mode'] for r in rr),
                    candidate_recall=sum(r['has_25mm_mode'] for r in rr)/len(rr) if rr else None,median_best_error_m=float(np.median([r['best_of_three_error_m'] for r in rr])) if rr else None))
    save(root/'mode_diagnostics.json',dict(source_sha=sha(__file__),GT_identity_conditioned=True,best_of_three_is_oracle_candidate_metric=True,
        selected_task_accuracy=False,summary=summary,records=len(records)))
    # Preserve per-query paired descriptive evidence rather than only averages.
    from scripts.score_textured_material_comparison_v1 import write_csv
    write_csv(root/'mode_diagnostics.csv',records)


if __name__=='__main__':main()
