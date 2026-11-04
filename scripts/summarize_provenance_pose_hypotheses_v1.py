"""Compact descriptive outcomes from sealed-run evaluation artifacts."""
from pathlib import Path
from scripts.run_provenance_pose_hypotheses_v1 import read,save


def main():
    root=Path('results/provenance_pose_hypotheses_v1');cfg=read(root/'config.json');runs=read(root/'runs.json')
    scores={p:read(root/(p+'scores.json')) for p in ['', 'nn_', 'supported_']}
    sources=read(root/'provenance_diagnosis.json')['source_corrections'];rows=[]
    for method in cfg['methods']:
        item={'method':method}
        for prefix,data in scores.items():
            rr=[r for r in data['summaries'] if r['method']==method and r['period']==2]
            item[prefix+'strict']=sum(r['strict_correct'] for r in rr);item[prefix+'equivalent']=sum(r['equivalent_correct'] for r in rr)
            item[prefix+'strict_total']=sum(r['strict_total'] for r in rr);item[prefix+'equivalent_total']=sum(r['equivalent_total'] for r in rr)
            item[prefix+'strict_set']=[f"{r['scene']}:{r['truth_id']}" for r in data['rows'] if r['method']==method and r['period']==2 and r['strict_persistent'] and r['strict_correct']]
        ss=[r for r in sources if r['method']==method]
        for key in ['equivalent_fixed_voxels','equivalent_damaged_voxels','strict_fixed_voxels','strict_damaged_voxels']:
            item[key]=sum(r[key] for r in ss)
        item['elapsed_s']=sum(r['elapsed_s'] for r in runs if r['method']==method)
        item['point_queries']=sum(r['point_queries'] for r in runs if r['method']==method);rows.append(item)
    gauge=read(root/'anchor_gauge_diagnosis.json')['rows'];both=[r for r in gauge if min(r['assigned_voxels'])>0]
    summary={'rows':rows,'gauge':{'interventions':len(gauge),'assets':len({(r['scene'],r['asset_uid']) for r in gauge}),
       'both_epochs_assigned_assets':len({(r['scene'],r['asset_uid']) for r in both}),
       'max_extension_cost_difference':max(r['extension_cost_max_difference'] for r in gauge),
       'lower_full_energy_interventions':sum(r['full_energy_delta']<-1e-7 for r in both),
       'exactly_unchanged_full_cost_interventions':sum(r['maximum_full_point_cost_difference']==0 for r in both),
       'max_anchor_displacement_change_m':max(r['anchor_displacement_change_m'] for r in both)},
       'scope':'Descriptive development-set counts; no independent trials, motion accuracy or new inference results.'}
    save(root/'summary.json',summary)
    print(__import__('json').dumps(summary,indent=2))


if __name__=='__main__':main()
