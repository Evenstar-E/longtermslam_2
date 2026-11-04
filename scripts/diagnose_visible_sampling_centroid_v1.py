"""Post-seal oracle-map diagnostic, not a new inference/benchmark method."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.real_temporal_v1 import moved
from inspection_decisions.visible_sampling_centroid_v1 import Model,fuse_surface


def main():
    cfg=read('configs/visible_sampling_centroid_v1.json');out=Path(cfg['output_root']);pool=Path(cfg['pool_root']);root=Path(cfg['public_root'])
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h
    fc=read('results/ray_material_full_resolution_v1/effective_configuration.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']}
    poses={s:np.loadtxt(root/'evaluation/poses'/(s+'.txt')).reshape(4,4) for s in frames};rows=[]
    for pth in sorted(out.glob('pair_*/predictions.json')):
        p=read(pth);old=read(pool/pth.parent.name/'predictions.json');pair=p['pair'];h=pair['history_stem'];c=pair['current_stem'];gt=poses[c]@np.linalg.inv(poses[h]);sources=[(frames[h],np.eye(4))];source_errors=[]
        for b in old['bridges']:
            s=b['stem'];reference=poses[s]@np.linalg.inv(poses[h]);estimated=np.array(b['T_bridge_from_history']);sources.append((frames[s],reference))
            delta=np.linalg.norm(moved(frames[s].xyz,np.linalg.inv(estimated))-moved(frames[s].xyz,np.linalg.inv(reference)),axis=1)
            source_errors.append(dict(stem=s,mean_reconstruction_difference_mm=float(delta.mean()*1000),p95_reconstruction_difference_mm=float(np.quantile(delta,.95)*1000)))
        refsurface=fuse_surface(sources,frames[h].xyz.mean(0),cfg)
        with np.load(pth.parent/'surface.npz') as z:estimated_surface={k:z[k] for k in z.files}
        models={'estimated_map':Model(estimated_surface,frames[c],cfg),'reference_aligned_map':Model(refsurface,frames[c],cfg)}
        comparison={}
        poses_to_evaluate={'reference_pose':gt,'pair_only':np.array(old['rows'][p['choices']['pair_only']]['T']),'visible_selected':np.array(old['rows'][p['choices']['visible_centroid']]['T'])}
        for label,model in models.items():
            comparison[label]={}
            for name,T in poses_to_evaluate.items():
                score,diag=model.score(T,0.)
                comparison[label][name]=dict(observation_scores=score,diagnosis=diag)
        rows.append(dict(pair=pth.parent.name,source_errors=source_errors,comparison=comparison))
    save(out/'map_error_diagnosis.json',dict(rows=rows,scope='POSTHOC evaluator-only reference alignment, only3fixedqueryposes permap. No oracle reranking, no prediction update. Reference annotations are manual and lack independently verified millimeter error bounds. Estimated-vs-reference surface differences are not ground-truth surface errors.'))
    for r in rows:
        if r['pair'] in ['pair_06','pair_07','pair_08']:
            print(r['pair'],r['source_errors'])
            print([(a,b,round(q['observation_scores']['ordinary_rendered_pixels'],5),q['diagnosis']['surface_pixels'],round(q['diagnosis']['moments']['mean_shift_mm'],3)) for a,v in r['comparison'].items() for b,q in v.items()])


if __name__=='__main__':main()
