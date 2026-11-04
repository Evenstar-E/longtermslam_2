"""Full 192-task module experiment, all inference before evaluator access."""
from pathlib import Path
import pickle,time
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.run_rigid_identity_bounds_v1 import read,save as save_json,sha
from scripts.completion_catalog_scene_v1 import inverse
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.continuous_material_pose_mainline_v1 import observations,make_cloud,PairModel
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector,world_pose
from inspection_decisions.sparse_ray_material_incidence_v1 import raw_pairs,hypotheses,nearest_support_cost
from scripts.check_sparse_ray_material_incidence_v1 import checks
from scripts.check_sparse_ray_material_incidence_cache_v1 import checks as cache_checks
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions,translate_regions,nearest_support_costs


def json_value(value):
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


def save(path,value):save_json(path,json_value(value))


def main():
    cp=Path('configs/sparse_ray_material_incidence_v2.json');options=read(cp);cache=Path(options['cached_model_root']);base=Path(options['base_root']);cfg={**read(cache/'config.json'),**options};baseline=Path(cfg['baseline_root']);out=Path(cfg['output_root']);preflight=checks();preflight['cache_equivalence']=cache_checks()
    for root in [base,cache,baseline]:
        for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    out.mkdir(exist_ok=False);save(out/'config.json',cfg);save(out/'preflight.json',preflight)
    sources=[cp,Path(__file__),Path('inspection_decisions/sparse_ray_material_incidence_cache_v1.py'),Path('scripts/check_sparse_ray_material_incidence_cache_v1.py'),Path('configs/sparse_ray_material_incidence_v1.json'),Path('research/sparse_ray_material_incidence_cache_contract_v1_2026-10-25.md'),Path('inspection_decisions/sparse_ray_material_incidence_v1.py'),Path('scripts/check_sparse_ray_material_incidence_v1.py'),Path('research/sparse_ray_material_incidence_contract_v1_2026-10-25.md'),Path('research/sparse_ray_material_refinement_amendment_v1_2026-10-25.md'),Path('inspection_decisions/continuous_material_pose_mainline_v1.py'),Path('inspection_decisions/material_supported_continuous_alignment_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py'),Path('scripts/completion_catalog_scene_v1.py')]
    inputs=[base/'runs.json',base/'config.json',cache/'config.json',cache/'manifest.json']+list(base.glob('*/frame_*.npz'))+list(base.glob('*/inference.pkl'))+list(cache.glob('*/candidates.pkl'))+list(cache.glob('*/predictions.json'))+list(baseline.glob('*/predictions.json'))
    limits=read(cache/'manifest.json')['motion_limits'];save(out/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in inputs},'GT_read':False,'familiar_DEV':True,'motion_limits':limits,'restricted_C2_single_direction_refinement':True,'identity_rotation_C1_frozen':True,'same_candidate_count_not_same_FLOPs':True})
    runs=[];start=time.perf_counter()
    for run in read(base/'runs.json'):
        clock=time.perf_counter();tag=run['scene'];src=base/tag;dest=out/tag;dest.mkdir()
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (cache/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;all_raw=[];frames=[];points=[]
        for t in range(3):
            raw=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:raw.append({k:z[k].copy() for k in ['rgb','depth','C']})
            ff,pp=observations(raw,slots,groups,cfg);frames.append(ff);points.append(pp);rows=[[] for _ in range(24)]
            for v,frame in enumerate(ff):
                lab,_=labels(frame,slots,groups,cfg)
                for slot in range(24):
                    rr=raw_pairs(frame,lab,slot,slots[slot],t,v,cfg)
                    for row in rr:row['ray_id']=len(rows[slot]);rows[slot].append(row)
            all_raw.append(rows)
        with (dest/'raw_pairs.pkl').open('wb') as f:pickle.dump(all_raw,f)
        initial_sources={'catalogue_plane':read(baseline/tag/'predictions.json')['catalogue_plane']['selection'],'material_shared':read(cache/tag/'predictions.json')['material_shared']['selection']};results=[]
        for family,selection in initial_sources.items():
            for i,k,l in selection:
                chosen=[candidates[i][0][k],candidates[i][1][l]];centers=[slots[b['slot']] for b in chosen];x1=pose_vector(chosen[0]['T'],centers[0]);x2=pose_vector(chosen[1]['T'],centers[1]);H,_=all_regions(np.zeros(6),all_raw[0][i],cfg);C1,_=all_regions(x1,all_raw[1][chosen[0]['slot']],cfg);reference=H+C1;current_raw=all_raw[2][chosen[1]['slot']]
                R=Rotation.from_rotvec(x2[:3]).as_matrix();v=R[:,2];u=x2[3:];dot=float(u@v);disc=max(0.,dot*dot+limits[1]**2-u@u);ends=[-dot-np.sqrt(disc),-dot+np.sqrt(disc)];epsilon=np.unique(np.r_[0.,np.linspace(*ends,cfg['gauge_scan_points'])]);scan=[]
                history=inverse(points[0][i],inp['T0'][i]);slot=chosen[1]['slot'];model=PairModel(history,inp['T0'][i],frames[0],points[2][slot],frames[2],i,slot,slots[slot],cfg,make_cloud(history,0.,cfg),make_cloud(points[2][slot]-slots[slot],0.,cfg))
                initial_availability=None;availability_invariant=True;zero_regions=None;base_current,base_status=all_regions(x2,current_raw,cfg)
                for eps in epsilon:
                    xx=x2.copy();xx[3:]+=eps*v;T=world_pose(xx,centers[1]);current=translate_regions(base_current,float(eps));status=base_status;un,ordered=nearest_support_costs(current,reference,len(current_raw),cfg);standard,_=model.score(T)
                    if initial_availability is None:initial_availability=un['hypothesis_available']
                    else:availability_invariant&=np.array_equal(initial_availability,un['hypothesis_available'])
                    scan.append({'epsilon_m':float(eps),'nn':float(standard['nn']),'plane':float(standard['plane']),'sparse_unordered':un['mean_capped_squared_cost'],'sparse_ordered':ordered['mean_capped_squared_cost'],'hypothesis_count':len(current),'unavailable_rays':int((~un['hypothesis_available']).sum()),'exact_region_pairs':un['exact_region_pairs'],'hypothesis_status':status})
                    if eps==0:zero_regions=current
                chosen_methods={'original':int(np.flatnonzero(epsilon==0)[0])}
                for method in ['nn','plane','sparse_ordered','sparse_unordered']:
                    minimum=min(row[method] for row in scan);eligible=[j for j,row in enumerate(scan) if row[method]<=minimum+cfg['selection_tolerance']];chosen_methods[method]=min(eligible,key=lambda j:abs(scan[j]['epsilon_m']))
                predictions={}
                for method,index in chosen_methods.items():
                    xx=x2.copy();xx[3:]+=epsilon[index]*v;predictions[method]={'T1':chosen[0]['T'],'T2':world_pose(xx,centers[1]),'epsilon_m':float(epsilon[index]),'scan_index':index}
                results.append({'asset':i,'family':family,'selected_slots':[b['slot'] for b in chosen],'initial_branch_indices':[k,l],'initial_T1':chosen[0]['T'],'initial_T2':chosen[1]['T'],'gauge_epsilon_interval_m':ends,'raw_ray_counts':[len(all_raw[0][i]),len(all_raw[1][chosen[0]['slot']]),len(current_raw)],'reference_hypothesis_count':len(reference),'availability_invariant_on_scan':bool(availability_invariant),'predictions':predictions,'scan':scan})
                if i%6==5:print('PROGRESS',tag,family,i+1,'seconds',round(time.perf_counter()-clock,2),flush=True)
        save(dest/'predictions.json',results);item={k:run[k] for k in ['scene','geometry','seed','history_views']};item.update(raw_ray_pairs=sum(len(rr) for tt in all_raw for rr in tt),asset_initializations=len(results),elapsed_s=time.perf_counter()-clock);runs.append(item);save(out/'runs.json',runs);print('DONE',tag,item,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'GT_read':False,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}});print('SEALED',flush=True)


if __name__=='__main__':main()
