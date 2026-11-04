"""Reproducibility, exact ray-pair reconstruction and interval-cover witnesses."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.edge_conditioned_material_support_v1 import plane_ray,extract_edges,profile_intervals,endpoints
from inspection_decisions.set_valued_material_edges_v1 import minimum_support_cover
from scripts.check_set_valued_material_edges_v1 import checks
from inspection_decisions.first_return_material_surface_v1 import labels


def negative_counterexample():
    cfg=read('configs/edge_conditioned_material_support_v1.json');cfg['focal_px']=80.;h,w=64,80;yy,xx=np.mgrid[:h,:w];lab=np.full((h,w),-1);lab[12:52,15:65]=0;lab[27:38,34:46]=-1;depth=np.where(lab==0,4.,30.);depth[12:52,65:68]=2.
    frame={'depth':depth,'C':np.eye(4),'edge_view':0};world=np.stack([(xx-(w-1)/2)*depth/80,(yy-(h-1)/2)*depth/80,depth],-1);plane={'axis':5,'view':0,'normal':np.array([0.,0.,-1]),'centroid':world[lab==0].mean(0),'measured_indices':np.arange((lab==0).sum())}
    ff,_=extract_edges(frame,lab,world,0,np.zeros(3),[plane],cfg,21);group=[f for f in ff if tuple(f['key'])==(5,3)];assert len(group)==2;rows=[]
    for f in group:
        l,h,_,_,_=endpoints(np.zeros(6),[f],cfg)[(5,3)];rows.append({'lower':float(max(l)),'upper':float(min(h)),'witnesses':len(l)})
    cost,b=profile_intervals(np.array([r['lower'] for r in rows]),np.array([r['upper'] for r in rows]),np.ones(2));cover=minimum_support_cover(rows);assert cover['minimum_supports']==2 and cost>1.
    return {'GT_used':False,'zero_motion_true_synthetic_pose':True,'same_direction_key':[5,3],'actual_distinct_segment_intervals':rows,'forced_single_offset':float(b),'forced_single_offset_raw_cost':float(cost),'forced_single_offset_normalized_cost':float(cost/cfg['sigma_edge_m']**2),'multi_instance_supports':cover['points'],'multi_instance_intersection_violation':0.,'single_direction_identity_gate':'FAILED'}


def main():
    root=Path('results/set_valued_material_edges_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);n=0;ray_error=0.
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h;n+=1
    for field in ['sources','inputs']:
        for name,h in read(root/'manifest.json')[field].items():assert sha(name)==h;n+=1
    assert len(read(root/'all_gauge_scans.json'))==192;n+=1
    for run in read(root/'runs.json'):
        tag=run['scene']
        with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (root/tag/'edges.pkl').open('rb') as f:edges=pickle.load(f)
        slots=inp['T0'][:,:3,3]
        for t in range(3):
            frames=[]
            for path in sorted((base/tag).glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                lab,world=labels(frame,slots,np.arange(24)%4,cfg);frames.append((frame,lab,world))
            for slot,ff in enumerate(edges[t]):
                for f in ff:
                    frame,lab,world=frames[f['view']];p=f['inside_pixels'];q=f['outside_pixels'];assert np.all(np.abs(p-q).sum(1)==1);n+=len(p)
                    assert np.all(lab[p[:,1],p[:,0]]==slot) and np.all(lab[q[:,1],q[:,0]]!=slot);n+=2*len(p)
                    for pixels,stored in [(p,f['inside']),(q,f['outside'])]:
                        predicted,d,_=plane_ray(frame,pixels,f['plane_normal'],f['plane_centroid'],slots[slot],cfg);err=np.max(np.abs(predicted-stored));ray_error=max(ray_error,float(err));assert err<1e-12;n+=len(pixels)
                        if pixels is q:assert np.all(frame['depth'][q[:,1],q[:,0]]-d>cfg['edge_farther_return_margin_m']);n+=len(pixels)
                    assert np.all((f['outside']-f['inside'])@np.r_[np.eye(3),-np.eye(3)][f['key'][1]]>0);n+=len(p)
        for group in read(root/tag/'minimum_cover_relations.json'):
            rr=group['intervals'];result=minimum_support_cover(rr);assert result['feasible']==group['feasible'];n+=1
            if not result['feasible']:assert group['minimum_supports'] is None;n+=1;continue
            for r,k in zip(rr,group['assignment']):assert r['lower']<=group['points'][k]<=r['upper'];n+=1
            packing=[rr[j] for j in group['packing']];assert len(packing)==group['minimum_supports'];n+=1
            assert all(packing[j]['upper']<packing[j+1]['lower'] for j in range(len(packing)-1));n+=1
            if group['relations_resolved']:
                possible=np.array(group['possible_same']);forced=np.array(group['forced_same']);assert np.all(forced<=possible) and np.array_equal(possible,possible.T) and np.array_equal(forced,forced.T);n+=3
                for a,b in zip(*np.where(possible)):assert max(rr[a]['lower'],rr[b]['lower'])<=min(rr[a]['upper'],rr[b]['upper']);n+=1
    negative=negative_counterexample();save(root/'single_identity_negative_gate.json',negative);unit=checks();save(root/'audit.json',{'checks':n,'additional_unit_checks':unit,'ray_reconstruction_max_error_m':ray_error,'passed':True,'GT_read':False,'raw_edge_validity_not_implied_by_software_audit':True,'source_sha256':sha(Path(__file__))});print({'checks':n,'max_ray_error':ray_error,'negative_gate':negative},flush=True)
    # Partial V1 is explicitly rejected rather than silently sealed or scored.
    partial=Path('results/edge_conditioned_material_support_v1');save(partial/'aborted_status.json',{'complete':False,'status':'rejected_single_direction_physical_identity_model','full_batch_prediction_seal_exists':False,'GT_scored':False,'completed_scenes':[r['scene'] for r in read(partial/'runs.json')],'interrupted_by_owned_exec_session_ctrl_c':True,'counterexample':str(root/'single_identity_negative_gate.json'),'all_old_results_preserved':True})


if __name__=='__main__':main()
