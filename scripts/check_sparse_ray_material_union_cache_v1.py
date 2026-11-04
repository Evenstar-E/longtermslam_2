"""Compare union geometry to pairwise exact reference algorithm, no GT."""
from pathlib import Path
import pickle,time
import numpy as np
import shapely
from scripts.run_rigid_identity_bounds_v1 import read
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions,translate_regions,nearest_support_costs as pairwise
from inspection_decisions.sparse_ray_material_union_cache_v1 import prepare_reference,nearest_support_costs
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector


def checks():
    start=time.perf_counter();options=read('configs/sparse_ray_material_incidence_v1.json');cache=Path(options['cached_model_root']);cfg={**read(cache/'config.json'),**options};base=Path(cfg['base_root']);tag=read(base/'runs.json')[0]['scene']
    # The raw pairs are unaltered outputs from the first fully computed run3
    # condition; their SHA is included in the final runner input manifest.
    raw_path=Path('results/sparse_ray_material_incidence_v1_run3')/tag/'raw_pairs.pkl'
    with raw_path.open('rb') as f:raw=pickle.load(f)
    with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
    with (cache/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
    i,k,l=read(cache/tag/'predictions.json')['material_shared']['selection'][0];chosen=[candidates[i][0][k],candidates[i][1][l]];slots=inp['T0'][:,:3,3];x1=pose_vector(chosen[0]['T'],slots[chosen[0]['slot']]);x2=pose_vector(chosen[1]['T'],slots[chosen[1]['slot']]);H,_=all_regions(np.zeros(6),raw[0][i],cfg);C1,_=all_regions(x1,raw[1][chosen[0]['slot']],cfg);ref=H+C1;base_regions,_=all_regions(x2,raw[2][chosen[1]['slot']],cfg);reference=prepare_reference(ref);count=0;maximum=0.;old_scores=[];new_scores=[]
    for eps in np.linspace(-.08,.08,9):
        current=translate_regions(base_regions,eps);old=pairwise(current,ref,len(raw[2][chosen[1]['slot']]),cfg);new=nearest_support_costs(current,reference,len(raw[2][chosen[1]['slot']]),cfg)
        for a,b in zip(old,new):
            err=float(np.max(np.abs(a['per_ray_distance_m']-b['per_ray_distance_m'])));maximum=max(maximum,err);assert err<1e-10 and np.array_equal(a['hypothesis_available'],b['hypothesis_available']);count+=2
        old_scores.append([r['mean_capped_squared_cost'] for r in old]);new_scores.append([r['mean_capped_squared_cost'] for r in new])
    for j in [0,1]:
        def choose(values):
            v=np.array(values)[:,j];minimum=v.min();return min(np.where(v<=minimum+1e-12)[0],key=lambda index:abs(np.linspace(-.08,.08,9)[index]))
        assert choose(old_scores)==choose(new_scores);count+=1
    # Multiple overlapping polygons and separated degenerate supports.
    for shift in [0.,.1,2.]:
        rr=[{'key':(0,2),'source_axis':0,'vertices':np.array([[0,0],[1,0],[1,0],[0,0]],float),'ray_id':0}, {'key':(0,2),'source_axis':2,'vertices':np.tile([2.,0.],(4,1)),'ray_id':1}];cc=[{**r,'vertices':r['vertices']+[shift,.03]} for r in rr];old=pairwise(cc,rr,2,cfg);new=nearest_support_costs(cc,prepare_reference(rr),2,cfg)
        for a,b in zip(old,new):assert np.max(np.abs(a['per_ray_distance_m']-b['per_ray_distance_m']))<1e-10;count+=1
    return {'checks':count,'max_distance_error_m':maximum,'GT_read':False,'selection_same':True,'shapely_version':shapely.__version__,'GEOS_version':shapely.geos_version_string,'elapsed_s':time.perf_counter()-start}


if __name__=='__main__':print(checks())
