"""Independent pose-reference scoring after ALL predictions have been sealed."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,RayPair

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/ray_material_evidence_v1');ap.add_argument('--config',default='configs/ray_material_evidence_v1.json');args=ap.parse_args()
    out=Path(args.output);cfg=read(args.config);root=Path(cfg['public_root']);seal=read(out/'prediction_seal.json')
    assert seal['complete']
    for name,h in seal['sha256'].items():assert sha(out/name)==h
    public=read(root/'public.json');frames={r['stem']:Frame(root,r['stem'],cfg) for r in public['clouds']};rows=[];diagnoses=[]
    for folder in sorted(out.glob('pair_*')):
        p=read(folder/'predictions.json');pair=p['pair'];a=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);b=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4);GT=b@np.linalg.inv(a);anchor=np.array(p['anchor']);target=GT[:3,:3]@anchor+GT[:3,3]
        model=RayPair(frames[pair['history_stem']],frames[pair['current_stem']],anchor,cfg);gts,gtd=model.score(GT)
        for scope,choices in [('shared_pool',p['choices']),('original_proposals_only',p['original_proposals_only'])]:
            for method,i in choices.items():
                entry=p['pool'][i];T=np.array(entry['T_current_from_history']);error=float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-target));angle=float(np.rad2deg(Rotation.from_matrix(T[:3,:3]@GT[:3,:3].T).magnitude()))
                rows.append(dict(pair=folder.name,scope=scope,method=method,pose_index=i,pose_source=entry['source'],anchor_error_m=error,rotation_error_deg=angle,within_1cm=error<=.01,within_10cm=error<=.1,score=entry['scores'][method],GT_score=gts[method],GT_score_better=gts[method]<entry['scores'][method]-1e-10,diagnosis=entry['diagnosis']))
        diagnoses.append(dict(pair=folder.name,GT_scores=gts,GT_ray_diagnosis=gtd,unique_pose_evaluations=p['unique_full_score_evaluations'],exact_equivalent_max_error=max(abs(e['scores']['beta_material']-e['scores']['ordinary_exact_beta_equivalent']) for e in p['pool']),escape_controls=[dict(index=i,scores=p['pool'][i]['scores'],diagnosis=p['pool'][i]['diagnosis']) for i in p['escape_indices']]))
    summary=[]
    for scope in ['original_proposals_only','shared_pool']:
        for method in cfg['methods']:
            rr=[r for r in rows if r['scope']==scope and r['method']==method];errors=np.array([r['anchor_error_m'] for r in rr]);summary.append(dict(scope=scope,method=method,pairs=len(rr),within_1cm=sum(r['within_1cm'] for r in rr),within_10cm=sum(r['within_10cm'] for r in rr),median_anchor_error_m=float(np.median(errors)),max_anchor_error_m=float(errors.max()),median_rotation_error_deg=float(np.median([r['rotation_error_deg'] for r in rr])),unknown_escape_selections=sum(r['pose_source']=='unknown_escape_control' for r in rr),reference_scores_worse_than_selected=sum(r['GT_score']>r['score']+1e-10 for r in rr)))
    save(out/'scores.json',dict(summary=summary,rows=rows,diagnosis=diagnoses,scope=cfg['scope'],confidence='No calibrated posterior, physical certificate, or independent test claim.'))
    print(summary);print('PAIR08',[r for r in rows if r['pair']=='pair_08' and r['scope']=='shared_pool'])

if __name__=='__main__':main()
