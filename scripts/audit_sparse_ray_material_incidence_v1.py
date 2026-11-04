"""Audit complete restricted-refinement pipeline, candidate fairness and raw rays."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions


def main():
    root=Path('results/sparse_ray_material_incidence_v1_run4');cfg=read(root/'config.json');base=Path(cfg['base_root']);manifest=read(root/'manifest.json');count=0;max_move=0.;rays=0;candidate_counts={};unavailable_invariance=True
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;count+=1
    for field in ['sources','inputs']:
        for name,h in manifest[field].items():assert sha(name)==h;count+=1
    assert not any('evaluation_only' in name for name in manifest['inputs']);count+=1
    families={};total_equivalence=0;max_equivalence=0.;reference_counts=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];dest=root/tag
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'raw_pairs.pkl').open('rb') as f:raw=pickle.load(f)
        slots=inp['T0'][:,:3,3]
        for t in range(3):
            frames=[]
            for path in sorted((base/tag).glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                lab,_=labels(frame,slots,np.arange(24)%4,cfg);frames.append((frame,lab))
            for slot,rows in enumerate(raw[t]):
                assert [r['ray_id'] for r in rows]==list(range(len(rows)));count+=len(rows)
                for row in rows:
                    frame,lab=frames[row['view']];p=np.array(row['inside_pixel']);q=np.array(row['outside_pixel']);h,w=lab.shape
                    assert lab[p[1],p[0]]==slot and lab[q[1],q[0]]!=slot and np.abs(p-q).sum()==1;count+=3
                    assert row['depth']==frame['depth'][p[1],p[0]] and row['outside_depth']==frame['depth'][q[1],q[0]];count+=2
                    assert np.max(np.abs(np.array(row['origin'])-(frame['C'][:3,3]-slots[slot])))<1e-12;count+=1
                    for pix,key in [(p,'ray_inside'),(q,'ray_outside')]:
                        r=np.r_[(pix-[((w-1)/2),((h-1)/2)])/cfg['focal_px'],1.]@frame['C'][:3,:3].T;assert np.max(np.abs(r-row[key]))<1e-12;count+=1
                    rays+=1
        pred=read(dest/'predictions.json')
        for family in ['catalogue_plane','material_shared']:
            rr=[r for r in pred if r['family']==family];assert len(rr)==24 and {r['asset'] for r in rr}==set(range(24));count+=2
            for t in [0,1]:assert {r['selected_slots'][t] for r in rr}==set(range(24));count+=1
        for row in pred:
            family=row['family'];families[family]=families.get(family,0)+1;scan=row['scan'];eps=np.array([r['epsilon_m'] for r in scan]);assert 0. in eps and len(np.unique(eps))==len(eps) and 17<=len(eps)<=18;count+=3;candidate_counts[len(eps)]=candidate_counts.get(len(eps),0)+1
            assert row['availability_invariant_on_scan'] and len({s['hypothesis_count'] for s in scan})==1 and len({s['unavailable_rays'] for s in scan})==1;count+=3
            unavailable_invariance&=row['availability_invariant_on_scan'];T1=np.array(row['initial_T1']);T2=np.array(row['initial_T2']);center=slots[row['selected_slots'][1]];v=T2[:3,2]
            H,_=all_regions(np.zeros(6),raw[0][row['asset']],cfg);C1,_=all_regions(pose_vector(T1,slots[row['selected_slots'][0]]),raw[1][row['selected_slots'][0]],cfg);reference_counts.append({'scene':tag,'asset':row['asset'],'family':family,'actual_reference_hypotheses':len(H)+len(C1),'reference_union_cache_groups':row['reference_hypothesis_count'],'stored_field_name_correction':'reference_hypothesis_count in predictions means union-cache token count, not observations or candidate hypotheses'})
            for method,p in row['predictions'].items():
                S=np.array(p['T2']);e=p['epsilon_m'];assert np.max(np.abs(np.array(p['T1'])-T1))<1e-14;assert np.max(np.abs(S[:3,:3]-T2[:3,:3]))<1e-12;count+=2
                err=np.max(np.abs(S[:3,3]-T2[:3,3]-e*v));max_move=max(max_move,float(err));assert err<1e-12 and np.linalg.norm(S[:3,3]-center)<=manifest['motion_limits'][1]+1e-10;count+=2
                assert abs(e-eps[p['scan_index']])<1e-14;count+=1
                if method=='original':assert e==0;count+=1
                else:
                    minimum=min(s[method] for s in scan);eligible=[j for j,s in enumerate(scan) if s[method]<=minimum+cfg['selection_tolerance']];best=min(eligible,key=lambda j:abs(eps[j]));assert p['scan_index']==best;count+=1
        eq=dest/'prior_run_equivalence.json'
        if eq.exists():
            value=read(eq);assert value['all_selected_epsilon_equal'] and value['max_score_error']<1e-10;count+=value['assets']*5;total_equivalence+=value['assets'];max_equivalence=max(max_equivalence,value['max_score_error'])
            old=read(Path('results/sparse_ray_material_incidence_v1_run3')/tag/'predictions.json');assert len(old)==len(pred);count+=1
            for j in range(len(pred)):
                a=pred[j];b=old[j];assert (a['asset'],a['family'])==(b['asset'],b['family']);assert len(a['scan'])==len(b['scan']);assert np.array_equal([s['epsilon_m'] for s in a['scan']],[s['epsilon_m'] for s in b['scan']]);count+=3
    assert families=={'catalogue_plane':192,'material_shared':192};count+=1
    save(root/'reference_count_semantics.json',reference_counts);save(root/'audit.json',{'passed':True,'checks':count,'raw_ray_pairs_checked':rays,'family_asset_counts':families,'candidate_count_distribution':candidate_counts,'max_frozen_pose_translation_error_m':max_move,'availability_invariant_all':bool(unavailable_invariance),'prior_pairwise_run_asset_initializations_checked':total_equivalence,'prior_pairwise_run_max_score_error':max_equivalence,'GT_read':False,'task_metrics_not_implied_by_software_audit':True,'source_sha256':sha(Path(__file__))});print({'checks':count,'rays':rays,'families':families,'candidates':candidate_counts,'max_pose_error':max_move,'equivalence_assets':total_equivalence},flush=True)


if __name__=='__main__':main()
