"""Full-point approximation check, same fixed pose pool and no evaluation input."""
from pathlib import Path
import time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,RayPair

def main():
    checkpath=Path('configs/ray_material_full_resolution_v1.json');check=read(checkpath);cfg=read(check['base_configuration']);cfg['max_points_per_frame']=check['max_points_per_frame'];cfg['scope']+=' Full-resolution diagnostic, no new pose proposals.'
    origin=Path('results/ray_material_evidence_v1');out=Path(check['output_root']);out.mkdir(exist_ok=False);root=Path(cfg['public_root']);public=read(root/'public.json')
    for name,h in read(origin/'prediction_seal.json')['sha256'].items():assert sha(origin/name)==h
    save(out/'effective_configuration.json',cfg);save(out/'start_manifest.json',dict(sources={str(p):sha(p) for p in [Path(__file__),checkpath,Path(check['base_configuration']),Path('inspection_decisions/ray_material_evidence_v1.py')]},inputs={str(p):sha(p) for p in list(origin.glob('pair_*/predictions.json'))+[root/'public.json']+[p for p in (root/'public').rglob('*') if p.is_file()]},GT_pose_read=False))
    frames={r['stem']:Frame(root,r['stem'],cfg) for r in public['clouds']};start=time.perf_counter()
    for folder in sorted(origin.glob('pair_*')):
        p=read(folder/'predictions.json');pair=p['pair'];model=RayPair(frames[pair['history_stem']],frames[pair['current_stem']],p['anchor'],cfg)
        for entry in p['pool']:entry['scores'],entry['diagnosis']=model.score(np.array(entry['T_current_from_history']))
        p['choices']={m:min(range(len(p['pool'])),key=lambda i:p['pool'][i]['scores'][m]) for m in cfg['methods']};p['original_proposals_only']={m:min(p['initial_indices'],key=lambda i:p['pool'][i]['scores'][m]) for m in cfg['methods']};p['unique_full_score_evaluations']=model.calls
        p['optimizer_outputs_scope']='Original sparse-point optimizer traces only; no full-resolution optimization performed.'
        dest=out/folder.name;dest.mkdir();save(dest/'predictions.json',p);print('FULL_POINT_RESCORING',folder.name,round(time.perf_counter()-start,2),flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},complete=True,GT_pose_read=False,elapsed_s=time.perf_counter()-start))

if __name__=='__main__':main()
