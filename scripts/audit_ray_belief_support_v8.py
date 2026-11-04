"""Posthoc finite-particle coverage diagnostic. No tuning or new policy runs."""
import hashlib,json
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes,mixture_particles
from comparative_evidence.uncertainty_v3 import attach_geometry_information


def main():
    out=Path('results/comparative_ray_branches_v8/belief_support_diagnostic.json')
    if out.exists():raise FileExistsError(out)
    records=[];inputs={}
    for sid in range(48):
        ep=Episode(Path('results/comparative_interface_v1/scenes')/f'{sid:04d}',4,2)
        h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
        fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta);particles=mixture_particles(fit)[0];ep.finish()
        prior=particles[:,fit['delta_start']:fit['nuisance_start']].reshape(32,-1,3)
        ids={r['ref_id']:j for j,r in enumerate(fit['references'])}
        for epoch in ('history','current'):
            for view in range(1,12):
                source=Path('results/comparative_event_branches_v7')/f'shard_{sid%2}'/'branches'/f'{sid:04d}_{epoch}_{view:02d}.json'
                inputs[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
                p=json.loads(source.read_text())['prediction']
                for a in p['assets']:
                    delta=np.array(a['point_estimate_delta']);j=ids[a['ref_id']]
                    distance=float(np.linalg.norm(prior[:,j]-delta,axis=1).min())
                    records.append(dict(scene=sid,epoch=epoch,view=view,ref_id=a['ref_id'],after_label=a['label'],
                        nearest_prior_particle_m=distance,represented_within_task_tolerance=distance<=.10))
        print(f'belief support scene{sid:02d}',flush=True)
    byscene=[np.mean([r['represented_within_task_tolerance'] for r in records if r['scene']==sid]) for sid in range(48)]
    report=dict(scope='Posthoc diagnostic;48initialbeliefs,1056cachedafterfits,8448assetpointestimates. No truth used in distance; not a calibration/causal test.',
        limitation='32point finite prior coverage can fail even for adequate continuous prior; after point is an estimate, not ground truth. Geometry refit is not constrained to reweighting these points. No new solver or parameter selected.',
        asset_estimates=len(records),within_10cm_fraction=float(np.mean([r['represented_within_task_tolerance'] for r in records])),
        nearest_distance_m_quantiles=np.quantile([r['nearest_prior_particle_m'] for r in records],[.5,.9,.95]).tolist(),
        per_scene_fraction=byscene,records=records,inputs_sha256=inputs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v for k,v in report.items() if k not in ('records','inputs_sha256','per_scene_fraction')})


if __name__=='__main__':main()
