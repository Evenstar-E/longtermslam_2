"""Postseal physical-edge support coverage; geometry/axes never enter inference."""
from pathlib import Path
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box
from scripts.diagnose_set_valued_material_edges_v1 import domain
from inspection_decisions.sparse_ray_material_incidence_v1 import region,polygon_distances,InvalidIncidence


def actual_edges(obs,x,boxes,cfg):
    R=Rotation.from_rotvec(x[:3]).as_matrix();o=(obs['origin']-x[3:])@R;ri=obs['ray_inside']@R;ro=obs['ray_outside']@R;z=min(float(ray_box(o,ri,lo,hi)) for lo,hi in boxes)
    if not np.isfinite(z) or abs(z-obs['depth'])>cfg['depth_noise_bound_m']+1e-8:return [],'source depth not an own material first hit'
    qi=o+z*ri;edges=[];censored=False
    for a in range(3):
        if abs(ri[a])<1e-10 or abs(ro[a])<1e-10:continue
        na=-np.sign(ri[a])*np.eye(3)[a];d=qi[a];levels=np.unique(np.array(boxes)[:,:,a])
        if np.min(np.abs(levels-d))>1e-7 or not domain(qi,na,boxes):continue
        lam=(d-o[a])/ro[a]
        if not (0<lam<=obs['outside_depth']-cfg['depth_noise_bound_m']+1e-10):censored=True;continue
        qo=o+lam*ro;delta=qo-qi
        for k in range(3):
            if k==a or abs(delta[k])<1e-10:continue
            nk=np.sign(delta[k])*np.eye(3)[k]
            for b in np.unique(np.array(boxes)[:,:,k]):
                alpha=(b-qi[k])/delta[k]
                if not (-1e-7<=alpha<=1+1e-7):continue
                point=qi+alpha*delta
                if not (domain(point-1e-5*nk,na,boxes) and not domain(point+1e-5*nk,na,boxes)):continue
                key=tuple(sorted((a,k)));support=np.array([d,b] if a<k else [b,d]);covered=False;distance=None
                try:
                    obj=region(x,obs,a,k,cfg);distance=float(polygon_distances(obj['vertices'][None],np.tile(support,(4,1))[None])[0]);covered=distance<1e-8
                except InvalidIncidence:pass
                edges.append({'key':key,'source_axis':a,'exit_axis':k,'support_coordinates_m':support.tolist(),'conditional_region_covers_actual_support':covered,'support_distance_m':distance,'actual_source_depth':z,'actual_outside_plane_depth':lam})
    return edges,'censored real source plane' if censored and not edges else 'actual edge' if edges else 'no axis-aligned local boundary witness'


def main():
    root=Path('results/sparse_ray_material_incidence_v1_run4');cfg=read(root/'config.json');base=Path(cfg['base_root']);assert read(root/'prediction_seal.json')['complete']
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    rows=[];tasks=[];reasons={};raw_total=0;source_errors=0;eligible_raw=0;covered_raw=0
    from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (root/tag/'raw_pairs.pkl').open('rb') as f:raw=pickle.load(f)
        slots=inp['T0'][:,:3,3];geometry=geometry_for(run['seed'],24,run['geometry']);scene_rows=[]
        for t in range(3):
            owner=np.arange(24) if t==0 else np.argsort(truth[f'slot{t}'])
            for slot,observations in enumerate(raw[t]):
                i=int(owner[slot]);T=inp['T0'][i] if t==0 else truth['poses'][t-1,i];x=pose_vector(T,slots[slot])
                for obs in observations:
                    raw_total+=1;ee,reason=actual_edges(obs,x,geometry[i],cfg);reasons[reason]=reasons.get(reason,0)+1;source_errors+=reason=='source depth not an own material first hit';eligible_raw+=bool(ee);covered_raw+=any(e['conditional_region_covers_actual_support'] for e in ee)
                    for e in ee:
                        item={'scene':tag,'asset':i,'epoch':t,'slot':slot,'view':obs['view'],'ray_id':obs['ray_id'],'inside_pixel':obs['inside_pixel'].tolist(),'outside_pixel':obs['outside_pixel'].tolist(),**e};rows.append(item);scene_rows.append(item)
        for i in range(24):
            rr=[r for r in scene_rows if r['asset']==i and 2 in r['key']];by_edge={}
            for r in rr:
                token=tuple(r['key'])+tuple(np.round(r['support_coordinates_m'],9));by_edge.setdefault(token,[]).append(r)
            unordered=False;ordered=False;switch_only_edges=0;long_any=False
            for token,ee in by_edge.items():
                early=[e for e in ee if e['epoch']<2];late=[e for e in ee if e['epoch']==2]
                if not early or not late:continue
                unordered=True;same=any(a['source_axis']==b['source_axis'] for a in early for b in late);ordered|=same;switch_only_edges+=not same
                def enough(records):
                    groups={}
                    for e in records:groups.setdefault((e['epoch'],e['view']),[]).append(e)
                    for group in groups.values():
                        pixels=np.unique(np.array([e['inside_pixel']+e['outside_pixel'] for e in group]),axis=0);mid=(pixels[:,:2]+pixels[:,2:])/2
                        if len(mid)>=4 and np.linalg.norm(mid[:,None]-mid[None,:],axis=-1).max()>=3:return True
                    return False
                long_any|=enough(early) and enough(late)
            tasks.append({'scene':tag,'asset':i,'actual_shared_depth_edge_unordered':bool(unordered),'actual_shared_depth_edge_ordered_source_face':bool(ordered),'role_switch_only_depth_edge_count':int(switch_only_edges),'actual_shared_depth_edge_at_least_four_ray_pairs_and_span':bool(long_any)})
        print('PHYSICAL_SPARSE_GATE',tag,flush=True)
    summary={'raw_ray_pairs':raw_total,'actual_edge_eligible_ray_pairs':eligible_raw,'actual_edge_support_instances':len(rows),'actual_eligible_rays_with_true_support_in_conditional_region':covered_raw,'source_depth_inconsistencies':source_errors,'tasks':len(tasks),'tasks_shared_depth_edge_ordered_source_face':sum(t['actual_shared_depth_edge_ordered_source_face'] for t in tasks),'tasks_shared_depth_edge_unordered':sum(t['actual_shared_depth_edge_unordered'] for t in tasks),'tasks_shared_depth_edge_role_switch_only':sum(t['actual_shared_depth_edge_unordered'] and not t['actual_shared_depth_edge_ordered_source_face'] for t in tasks),'tasks_shared_depth_edge_at_least_four_ray_pairs_and_span':sum(t['actual_shared_depth_edge_at_least_four_ray_pairs_and_span'] for t in tasks),'reasons':reasons,'GT_only_after_full_prediction_seal':True,'GT_used_to_change_inference':False}
    save(root/'physical_geometry_gate.json',{'summary':summary,'rows':rows,'tasks':tasks,'source_sha256':sha(Path(__file__))});print(summary,flush=True)


if __name__=='__main__':main()
