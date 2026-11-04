"""Actual mechanism gains and candidate-vs-objective failure decisions."""
import csv,json
from pathlib import Path
from collections import Counter
from scripts.run_material_surface_correspondence_v13b import configuration
from scripts.score_textured_material_comparison_v1 import write_csv
from scripts.run_textured_material_comparison_v1 import save


def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))


def main():
    cfg=configuration();root=Path(cfg['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));tasks=read(root/'task_metrics.csv');oracle={(r['variant'],int(r['asset'])):r for r in read(root/'oracle_by_stage.csv')};nodes={(r['variant'],int(r['asset'])):r for r in read(root/'node_material_coverage.csv') if r['pair']=='1'};old=read(Path(cfg['parent_root'])/'task_metrics.csv');rows=[];summary=[];oldcompare=[]
    for v in cfg['variants']:
        for r in [r for r in tasks if r['variant']==v and r['method']=='shared_phase']:
            i=int(r['asset']);o=oracle[v,i];n=nodes[v,i];error=float(o['continuous_error_m']) if o['continuous_error_m'] else None
            if int(r['qualified']):cause='complete_task_success'
            elif not int(o['correct_identity_searched']):cause='correct_identity_outside_finite_graph'
            elif error is None or error>.05:
                if not int(n['near_material_edges']):cause='no_good_pose_and_no_15mm_H2_node_pair'
                elif not int(n['near_edges_in_descriptor_top2']):cause='no_good_pose_and_true_near_edges_outside_descriptor_top2'
                else:cause='no_good_pose_despite_some_material_proposal_edges'
            elif not int(r['both_ids']):cause='good_pose_exists_but_wrong_identity_selected'
            else:cause='good_pose_exists_but_objective_selects_bad_pose'
            rows.append(dict(variant=v,asset=i,rear=int(r['rear']),category=cause,both_ids=int(r['both_ids']),worst_error_m=float(r['worst_error_m']),oracle_error_m=error,near_material_H2_edges=int(n['near_material_edges']),near_H2_edges_in_descriptor_top2=int(n['near_edges_in_descriptor_top2'])))
        rr=[r for r in rows if r['variant']==v];ss={r['method']:r for r in scores['summary'] if r['variant']==v};comparisons=[r for r in scores['comparisons'] if r['variant']==v]
        summary.append(dict(variant=v,shared_phase_qualified=ss['shared_phase']['qualified'],shared_phase_rear_qualified=ss['shared_phase']['rear_qualified'],phase_vs_normal_net=ss['partial_phase']['qualified']-ss['partial_normals']['qualified'],sharing_phase_net=ss['shared_phase']['qualified']-ss['partial_phase']['qualified'],shared_phase_vs_replay_net=ss['shared_phase']['qualified']-ss['retained_v12']['qualified'],categories=dict(Counter(r['category'] for r in rr)),rear_categories=dict(Counter(r['category'] for r in rr if r['rear'])),controlled_comparisons=comparisons))
        bb={int(r['asset']):r for r in tasks if r['variant']==v and r['method']=='shared_phase'}
        for m in ['projective_points','surface_rgb_fusion']:
            aa={int(r['asset']):r for r in old if r['variant']==v and r['method']==m}
            oldcompare.append(dict(variant=v,old_method=m,gained=[i for i in aa if int(bb[i]['qualified']) and not int(aa[i]['qualified'])],lost=[i for i in aa if int(aa[i]['qualified']) and not int(bb[i]['qualified'])]))
    write_csv(root/'failure_decisions.csv',rows);save(root/'mainline_decision.json',dict(summary=summary,old_strong_comparisons=oldcompare,scope='Actual frozen full tasks, no best-row cherry-picking. Different representation/search scope across versions; only withinV13 point/normal/phase and independent/shared contrasts isolate those mechanisms. One seenTRAIN, no originality or generalization confirmation.'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
