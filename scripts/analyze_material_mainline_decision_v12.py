"""Separate candidate, association, objective and mainline-gain evidence."""
import csv,json
from pathlib import Path
import numpy as np
from collections import Counter
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save


def read(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def main():
    cfg=configuration();root=Path(cfg['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete']
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));tasks=read(root/'task_metrics.csv');oracle={(r['variant'],int(r['asset'])):r for r in read(root/'oracle_by_stage.csv')};full={(r['variant'],int(r['asset'])):r for r in read(root/'full_propagation_oracle.csv')};info={(r['variant'],int(r['asset'])):r for r in read(root/'local_information.csv')};mech={(r['variant'],int(r['asset'])):r for r in read(root/'pose_mechanism.csv')};rows=[];summary=[]
    pi=cfg['shared_prior'];eta=cfg['material_null_likelihood'];bound=cfg['material_factor_weight']*np.log((pi+(1-pi)*eta**2)/(pi+(1-pi)*eta)**2)/(-2*np.log(eta))
    bound_rows=[]
    for v in cfg['variants']:
        with np.load(root/f'{v}_shared_material.npz') as f:pred=dict(f)
        with np.load(root/f'{v}_independent_material.npz') as f:ind=dict(f)
        for r in [r for r in tasks if r['variant']==v and r['method']=='independent_material' and int(r['both_ids']) and not int(r['qualified'])]:
            i=int(r['asset']);m=mech.get((v,i))
            if m:
                gap=float(m['true_independent'])-float(ind['objectives'][i]);goodgap=float(m['best_pose_independent'])-float(ind['objectives'][i]) if float(m['best_new_error_m'])<=.05 else None
                bound_rows.append(dict(variant=v,asset=i,rear=int(r['rear']),true_pose_independent_energy_gap=gap,concrete_qualified_pose_energy_gap=goodgap,maximum_shared_rebate=float(bound),true_pose_gap_exceeds_shared_capacity=int(gap>bound),concrete_qualified_gap_exceeds_capacity=int(goodgap is not None and goodgap>bound)))
        for r in [r for r in tasks if r['variant']==v and r['method']=='shared_material']:
            i=int(r['asset']);key=v,i;o=oracle[key];f=full[key];s=info[key];m=mech.get(key)
            if int(r['qualified']):category='complete_task_success'
            elif float(o['continuous_error_m'])>.05:
                if float(f['full_propagation_error_m'])<=.05:category='propagation_budget_omitted_good_mode'
                elif int(s['supported_pixels'])==0:category='no_good_pose_and_no_H_patch_hit_at_true_pose'
                else:category='no_good_pose_despite_some_true_H_patch_support'
            elif not int(r['both_ids']):category='good_pose_exists_but_identity_not_selected'
            else:category='good_pose_exists_but_pose_objective_selects_bad'
            exact_bad_preference=bool(not int(r['qualified']) and int(r['both_ids']) and m and float(m['best_new_error_m'])<=.05 and float(pred['objectives'][i])<=float(m['best_pose_shared'])+1e-9)
            rows.append(dict(variant=v,asset=i,rear=int(r['rear']),category=category,both_ids=int(r['both_ids']),worst_error_m=float(r['worst_error_m']),continuous_oracle_error_m=float(o['continuous_error_m']),all_old_mode_propagation_error_m=float(f['full_propagation_error_m']),true_H_patch_hits=int(s['supported_pixels']),true_local_fusion_rank=int(s['geometry_texture_local_rank']),correct_ID_bad_energy_no_higher_than_concrete_good_pose=int(exact_bad_preference)))
        rr=[r for r in rows if r['variant']==v];ss={r['method']:r for r in scores['summary'] if r['variant']==v};comp=next(r for r in scores['comparisons'] if r['variant']==v and r['baseline']=='independent_material' and r['candidate']=='shared_material')
        summary.append(dict(variant=v,categories=dict(Counter(r['category'] for r in rr)),rear_categories=dict(Counter(r['category'] for r in rr if r['rear'])),actual_common_candidate_sharing_gained=comp['gained'],actual_common_candidate_sharing_lost=comp['lost'],shared_vs_independent_net_qualified=ss['shared_material']['qualified']-ss['independent_material']['qualified'],shared_vs_v11_normal_points_net_qualified=ss['shared_material']['qualified']-ss['retained_points']['qualified']))
    strong=[];previous=read(Path(cfg['parent_root'])/'task_metrics.csv')
    for v in cfg['variants']:
        b={int(r['asset']):r for r in tasks if r['variant']==v and r['method']=='shared_material'}
        for method in ['projective_points','surface_rgb_fusion']:
            a={int(r['asset']):r for r in previous if r['variant']==v and r['method']==method}
            strong.append(dict(variant=v,old_method=method,gained=[i for i in a if int(b[i]['qualified']) and not int(a[i]['qualified'])],lost=[i for i in a if int(a[i]['qualified']) and not int(b[i]['qualified'])],scope='Cross-version task comparison, not pure same-representation sharing effect.'))
    write_csv(root/'failure_decisions.csv',rows);write_csv(root/'shared_rebate_capacity.csv',bound_rows);save(root/'mainline_decision.json',dict(summary=summary,old_strong_comparisons=strong,maximum_shared_rebate=float(bound),correct_ID_bad_pose_comparisons=len(bound_rows),true_pose_gap_exceeds_shared_capacity=sum(r['true_pose_gap_exceeds_shared_capacity'] for r in bound_rows),concrete_qualified_pose_gap_exceeds_capacity=sum(r['concrete_qualified_gap_exceeds_capacity'] for r in bound_rows),capacity_scope='Fixed formula/prior/null/weight, specified same-ID candidate pairs only; not an impossibility proof over the entire qualified pose set.',scope='Classification of this implemented model and candidate budgets on one seen TRAIN. A local support/rank failure is NOT raw-data impossibility. Common-candidate net counts are actual tasks, not objective descent. No posthoc parameter adjustment.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
