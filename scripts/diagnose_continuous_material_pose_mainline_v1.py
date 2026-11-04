"""Post-seal drift and unknown-region loss controls on ALL bounded NN failures."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.completion_catalog_scene_v1 import inverse
from inspection_decisions.continuous_material_pose_mainline_v1 import observations,project


def state_groups(query,frames,slot,cfg):
    positive=np.zeros(len(query),bool);absent=positive.copy()
    for f in frames:
        camera,uv=project(query,f['C'],cfg);pix=np.rint(np.where(np.isfinite(uv),uv,-1e8)).astype(int);u,v=pix[:,0],pix[:,1]
        inside=(camera[:,2]>0)&(u>=0)&(u<cfg['width'])&(v>=0)&(v<cfg['height']);d=np.zeros(len(query));lab=np.full(len(query),-1)
        d[inside]=f['depth'][v[inside],u[inside]];lab[inside]=f['labels'][v[inside],u[inside]]
        near=inside&(np.abs(camera[:,2]-d)<=cfg['depth_tolerance_m'])
        positive|=near&(lab==slot);absent|=(inside&(camera[:,2]<d-cfg['depth_tolerance_m']))|(near&(lab!=slot))
    return {'compatible_positive':positive,'absence_without_positive':~positive&absent,'all_views_unknown':~positive&~absent}


def main():
    root=Path('results/continuous_material_pose_mainline_v1');envelope=Path('results/continuous_material_motion_envelope_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);limits=read(envelope/'manifest.json')
    for old in [root,envelope]:
        for name,h in read(old/'prediction_seal.json')['sha256'].items():assert sha(old/name)==h
    rows=read(root/'scores.json');seeds=read(root/'true_pose_refinement.json');failures=read(envelope/'ranking_failures.json');loss=[];outside=[];true_endpoints=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (root/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        original=read(root/tag/'predictions.json');allowed=np.load(envelope/tag/'allowed.npz');logs=read(root/tag/'optimization_logs.json')
        for i,k,l in original['expanded_nn']['selection']:
            for t,j in enumerate([k,l]):
                b=candidates[i][t][j];outside.append({'scene':tag,'asset':i,'epoch':t,'source':b['source'],'outside_envelope':not bool(allowed[f'{i}_{t}'][j]),
                    'rotation_deg':float(np.degrees(np.linalg.norm(Rotation.from_matrix(b['T'][:3,:3]).as_rotvec()))),'translation_from_slot_m':float(np.linalg.norm(b['T'][:3,3]-inp['T0'][b['slot'],:3,3]))})
        for record in [r for r in seeds if r['scene']==tag]:
            log=next(l for l in logs if l['asset']==record['asset'] and l['epoch']==record['epoch'] and l['parent']==next(j for j,b in enumerate(inp['branches'][record['asset']][record['epoch']]) if b['slot']==truth[f"slot{record['epoch']+1}"][record['asset']] and b['code']==truth[f"code{record['epoch']+1}"][record['asset']]))
            index=log['endpoint_index'];true_endpoints.append({**record,'history_views':run['history_views'],'allowed_endpoint':bool(index is not None and allowed[f"{record['asset']}_{record['epoch']}"][index])})
        cases=[r for r in failures if r['scene']==tag and r['method']=='envelope_nn']
        if not cases:continue
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;frames=[];points=[]
        for t in [0,1,2]:
            raw=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:raw.append({k:z[k].copy() for k in ['rgb','depth','C']})
            ff,pp=observations(raw,slots,groups,cfg);frames.append(ff);points.append(pp)
        for case in cases:
            i=case['asset'];history=inverse(points[0][i],inp['T0'][i]);tree=cKDTree(history);groupsum={g:0. for g in ['compatible_positive','absence_without_positive','all_views_unknown']};counts={g:0 for g in groupsum};total=0.;true_cost=0.;selected_cost=0.
            for t,j in enumerate(case['selected']):
                b=candidates[i][t][j];current=points[t+1][b['slot']];actual=truth['poses'][t,i];true_query=inverse(current,actual);selected_query=inverse(current,b['T'])
                # World positions under the generating pose, only diagnostic.
                true_history_world=true_query@inp['T0'][i,:3,:3].T+inp['T0'][i,:3,3]
                masks=state_groups(true_history_world,frames[0],i,cfg);true_loss=np.minimum(tree.query(true_query,workers=1)[0]/cfg['distance_scale_m'],1)**2;selected_loss=np.minimum(tree.query(selected_query,workers=1)[0]/cfg['distance_scale_m'],1)**2
                for g,mask in masks.items():groupsum[g]+=float((selected_loss[mask]-true_loss[mask]).sum()/len(current));counts[g]+=int(mask.sum())
                total+=len(current);true_cost+=float(true_loss.mean());selected_cost+=float(selected_loss.mean())
            assert abs(sum(groupsum.values())-(selected_cost-true_cost))<1e-10
            assert abs(selected_cost-case['selected_cost'])<1e-10
            loss.append({'scene':tag,'geometry':run['geometry'],'asset':i,'selected_error_m':case['selected_error_m'],'selected_cost':selected_cost,'generated_true_pose_cost':true_cost,
                'selected_minus_true_cost':selected_cost-true_cost,'group_cost_differences':groupsum,'point_counts':counts,'total_points':total,
                'interpretation':'Posthoc projection groups under generated true poses; not observable material labels available to inference.'})
    seed_summary=[]
    for hv in [1,3]:
        for epoch in [0,1]:
            rr=[r for r in true_endpoints if r['history_views']==hv and r['epoch']==epoch]
            seed_summary.append({'history_views':hv,'epoch':epoch,'n':len(rr),'complete':sum(r['complete'] for r in rr),'drift_gt_5cm':sum(r['pushed_beyond_5cm'] for r in rr),
                'allowed_complete':sum(r['allowed_endpoint'] for r in rr),'allowed_drift_gt_5cm':sum(r['allowed_endpoint'] and r['pushed_beyond_5cm'] for r in rr)})
    summary={'expanded_nn_selected_epoch_poses':len(outside),'expanded_nn_selected_outside_motion_envelope':sum(r['outside_envelope'] for r in outside),
        'bounded_nn_same_id_wrong_material_cases':len(loss),'unknown_group_favors_wrong':sum(r['group_cost_differences']['all_views_unknown']<0 for r in loss),
        'positive_group_favors_generated_true':sum(r['group_cost_differences']['compatible_positive']>0 for r in loss),
        'both_pattern':sum(r['group_cost_differences']['all_views_unknown']<0 and r['group_cost_differences']['compatible_positive']>0 for r in loss)}
    save(root/'mechanism_diagnosis.json',{'postseal_GT_assisted':True,'predictions_changed':False,'summary':summary,'true_seed_summary':seed_summary,'true_seed_endpoint_details':true_endpoints,'expanded_nn_motion_prior_check':outside,'bounded_nn_loss_decomposition':loss,'source_sha256':sha(Path(__file__))})
    print(summary,flush=True);print(seed_summary,flush=True)


if __name__=='__main__':main()
