"""Post-seal evaluation only. Reference poses never change predictions/configs."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,RayPair
from inspection_decisions.real_temporal_v1 import moved
from inspection_decisions.material_cycle_bridge_v1 import BridgeModel,partial_map,compose_maps


def main():
    cfg=read('configs/material_cycle_bridge_v1.json');out=Path(cfg['output_root']);root=Path(cfg['public_root']);pool=Path(cfg['pool_root'])
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h
    fc=read(pool/'effective_configuration.json');frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']}
    poses={s:np.loadtxt(root/'evaluation/poses'/(s+'.txt')).reshape(4,4) for s in frames}
    rows=[]
    for pth in sorted(out.glob('pair_*/predictions.json')):
        p=read(pth);hs=p['pair']['history_stem'];cs=p['pair']['current_stem'];gt=poses[cs]@np.linalg.inv(poses[hs]);anchor=np.array(p['anchor'])
        bridges=[(frames[b['stem']],np.array(b['T_bridge_from_history']),np.array(b['T_current_from_bridge'])) for b in p['bridges']]
        model=BridgeModel(frames[hs],frames[cs],bridges,cfg);base=RayPair(frames[hs],frames[cs],anchor,fc).score(gt)[0]['old_nn'];gscore,gdiag=model.score(gt,base)
        bs=[]
        for spec,b in zip(p['bridges'],model.bridges):
            stem=spec['stem'];hsgt=poses[stem]@np.linalg.inv(poses[hs]);scgt=poses[cs]@np.linalg.inv(poses[stem])
            q=b['q'];ids=np.flatnonzero(q>=0);err=np.linalg.norm(moved(model.hp[0][ids],gt)-model.cp[0][q[ids]],axis=1)
            quant_radius=np.sqrt(cfg['sensor_sigma_m']**2+model.hp[2][ids]+model.cp[2][q[ids]])
            refq=compose_maps(partial_map(model.hp,b['patch'],hsgt,cfg),partial_map(b['patch'],model.cp,scgt,cfg))
            rid=np.flatnonzero(refq>=0);referr=np.linalg.norm(moved(model.hp[0][rid],gt)-model.cp[0][refq[rid]],axis=1)
            bs.append(dict(stem=stem,estimated_composed_matches=len(ids),reference_consistent_15mm=int(sum(err<=.015)),reference_consistent_quantization=int(sum(err<=quant_radius)),median_endpoint_error_mm=float(np.median(err)*1000) if len(ids) else None,
                path_anchor_error_mm=float(np.linalg.norm(moved(anchor[None],b['path'])-moved(anchor[None],gt))*1000),
                reference_pose_composed_matches=len(rid),reference_pose_consistent_15mm=int(sum(referr<=.015)),reference_pose_median_endpoint_error_mm=float(np.median(referr)*1000) if len(rid) else None))
        winners={method:dict(score=p['rows'][i]['scores'][method],score_at_reference_pose=gscore[method],reference_pose_better=gscore[method]<p['rows'][i]['scores'][method],diagnosis=p['rows'][i]['diagnosis']) for method,i in p['choices'].items()}
        rows.append(dict(pair=pth.parent.name,bridges=bs,reference_pose_diagnosis=gdiag,winners=winners))
    total=sum(b['estimated_composed_matches'] for r in rows for b in r['bridges']);consistent=sum(b['reference_consistent_15mm'] for r in rows for b in r['bridges'])
    rt=sum(b['reference_pose_composed_matches'] for r in rows for b in r['bridges']);rc=sum(b['reference_pose_consistent_15mm'] for r in rows for b in r['bridges'])
    summary=dict(estimated_track_count=total,estimated_reference_consistent_15mm=consistent,estimated_fraction=consistent/total,reference_pose_track_count=rt,reference_pose_consistent_15mm=rc,reference_pose_fraction=rc/rt,
        reference_pose_worse_than_selected={m:sum(not r['winners'][m]['reference_pose_better'] for r in rows) for m in rows[0]['winners']})
    save(out/'material_witness_diagnosis.json',dict(summary=summary,rows=rows,scope='Posthoc reference-pose endpoint consistency, not independently labeled material identity; correlated witnesses, no binomial significance. No updated predictions.'))
    print(summary)
    print('HARD',[(r['pair'],r['bridges']) for r in rows if r['pair'] in ['pair_06','pair_07','pair_08']])


if __name__=='__main__':main()
