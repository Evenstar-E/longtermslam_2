"""Postseal physical finite-edge gate. Truth never changes inference features."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box,inverse
from inspection_decisions.material_supported_continuous_alignment_v1 import AXES,pose_vector
from inspection_decisions.set_valued_material_edges_v1 import segment_intervals


def occupied(p,boxes):
    p=np.asarray(p);result=np.zeros(p.shape[:-1],bool)
    for lo,hi in boxes:result|=((p>=lo)&(p<=hi)).all(-1)
    return result


def domain(p,normal,boxes):
    return occupied(p-1e-5*normal,boxes)&~occupied(p+1e-5*normal,boxes)


def rays(frame,pixels,T,cfg):
    h,w=frame['depth'].shape;p=np.asarray(pixels);r=np.c_[(p[:,0]-(w-1)/2)/cfg['focal_px'],(p[:,1]-(h-1)/2)/cfg['focal_px'],np.ones(len(p))]@frame['C'][:3,:3].T@T[:3,:3]
    o=inverse(frame['C'][:3,3][None],T)[0];return o,r


def inspect(feature,frame,T,center,boxes,cfg):
    a,k=feature['key'];ja,jk=a%3,k%3;na=AXES[a];nk=AXES[k];N=len(feature['inside'])
    measured=inverse(feature['inside']+center,T);levels_a=np.unique(np.array(boxes)[:,:,ja]);d=float(levels_a[np.argmin(np.abs(levels_a-measured[:,ja].mean()))]);plane_error=float(abs(d-measured[:,ja].mean()))
    o,ri=rays(frame,feature['inside_pixels'],T,cfg);_,ro=rays(frame,feature['outside_pixels'],T,cfg)
    zi=(d-o[ja])/ri[:,ja];zo=(d-o[ja])/ro[:,ja];qi=o+zi[:,None]*ri;qo=o+zo[:,None]*ro
    true_hit=np.min(np.array([ray_box(o,ri,lo,hi) for lo,hi in boxes]),axis=0);on_side=np.isfinite(true_hit)&(np.abs(zi-true_hit)<=.008)&domain(qi,na,boxes)
    p=feature['outside_pixels'];actual=frame['depth'][p[:,1],p[:,0]];farther=actual-zo>cfg['depth_noise_bound_m'];delta=qo-qi
    levels_k=np.unique(np.array(boxes)[:,:,jk]);possible=[]
    for level in levels_k:
        alpha=np.divide(level-qi[:,jk],delta[:,jk],out=np.full(N,np.nan),where=np.abs(delta[:,jk])>1e-12);point=qi+alpha[:,None]*delta
        matched=on_side&farther&(alpha>=-1e-7)&(alpha<=1+1e-7)&(delta@nk>0)&domain(point-1e-5*nk,na,boxes)&~domain(point+1e-5*nk,na,boxes)
        possible.append(matched)
    possible=np.array(possible);common=levels_k[possible.all(1)];any_witness=possible.any(0)
    interval=segment_intervals(pose_vector(T,center),[feature],cfg,0)[0];signed=[float(nk[jk]*z) for z in common];covered=[z for z in signed if interval['lower']<=z<=interval['upper']]
    normal_error=float(np.degrees(np.arccos(np.clip((T[:3,:3]@na)@feature['plane_normal'],-1,1))))
    return {'witnesses':N,'true_sideplane_offset':d,'mean_conditional_plane_offset_error_m':plane_error,'normal_error_deg':normal_error,'true_plane_first_hit_witnesses':int(on_side.sum()),'true_plane_neighbor_farther_witnesses':int(farther.sum()),'actual_axial_edge_witnesses':int(any_witness.sum()),'one_physical_edge_all_witnesses':bool(len(common)),'possible_physical_edge_signed_offsets_m':signed,'working_interval_covers_a_physical_edge':bool(covered),'working_interval':interval,'conditional_source_plane_is_not_actual_surface_point':True}


def main():
    root=Path('results/set_valued_material_edges_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);seal=read(root/'prediction_seal.json');assert seal['complete']
    for name,h in seal['sha256'].items():assert sha(root/name)==h
    for group in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[group].items():assert sha(name)==h
    rows=[];truth_hash={}
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag;truth_hash[tag]=sha(src/'evaluation_only.pkl')
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (root/tag/'edges.pkl').open('rb') as f:edges=pickle.load(f)
        geometry=geometry_for(run['seed'],24,run['geometry']);slots=inp['T0'][:,:3,3]
        for t in range(3):
            frames=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:frames.append({k:z[k].copy() for k in ['C','depth']})
            owner=np.arange(24) if t==0 else np.argsort(truth[f'slot{t}'])
            for slot,ff in enumerate(edges[t]):
                i=int(owner[slot]);T=inp['T0'][i] if t==0 else truth['poses'][t-1,i]
                for j,f in enumerate(ff):rows.append({'scene':tag,'geometry':run['geometry'],'history_views':run['history_views'],'epoch':t,'slot':slot,'asset':i,'segment':j,'key':f['key'],'view':f['view'],**inspect(f,frames[f['view']],T,slots[slot],geometry[i],cfg)})
        print('PHYSICAL_GATE',tag,flush=True)
    groups={}
    for r in rows:
        if r['one_physical_edge_all_witnesses']:groups.setdefault((r['scene'],r['asset'],tuple(r['key'])),[]).append(r)
    alias=[]
    for (scene,asset,key),rr in groups.items():
        distinct=[]
        for a in rr:
            for b in rr:
                if a is b:continue
                for x in a['possible_physical_edge_signed_offsets_m']:
                    for y in b['possible_physical_edge_signed_offsets_m']:
                        if abs(x-y)>.02 and abs(a['true_sideplane_offset']-b['true_sideplane_offset'])<.008:distinct.append([a['epoch'],a['view'],a['segment'],x,b['epoch'],b['view'],b['segment'],y])
        if distinct:alias.append({'scene':scene,'asset':asset,'key':key,'distinct_physical_edges_same_sideplane':distinct})
    summary={'observed_segments':len(rows),'one_physical_edge_all_witnesses':sum(r['one_physical_edge_all_witnesses'] for r in rows),'working_interval_covers_a_physical_edge':sum(r['working_interval_covers_a_physical_edge'] for r in rows),'intervals_internally_compatible_at_true_pose':sum(r['working_interval']['internally_compatible'] for r in rows),'normal_error_gt_10deg':sum(r['normal_error_deg']>10 for r in rows),'all_witnesses_true_sideplane_first_hits':sum(r['true_plane_first_hit_witnesses']==r['witnesses'] for r in rows),'all_witnesses_have_some_actual_axial_edge':sum(r['actual_axial_edge_witnesses']==r['witnesses'] for r in rows),'same_sideplane_same_direction_multiple_actual_edge_groups':len(alias),'GT_only_after_full_seal':True,'predictions_or_features_changed':False,'new_task_accuracy_computed':False}
    save(root/'physical_geometry_gate.json',{'summary':summary,'rows':rows,'actual_multiple_edge_groups':alias,'source_sha256':sha(Path(__file__)),'truth_sha256':truth_hash});print(summary,flush=True)


if __name__=='__main__':main()
