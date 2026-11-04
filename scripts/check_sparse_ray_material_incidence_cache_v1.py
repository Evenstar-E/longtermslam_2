"""Actual RGB-D, no GT: compare cached/vectorized and scalar geometry/costs."""
from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.material_supported_continuous_alignment_v1 import pose_vector
from inspection_decisions.sparse_ray_material_incidence_v1 import raw_pairs,hypotheses,nearest_support_cost
from inspection_decisions.sparse_ray_material_incidence_cache_v1 import all_regions,translate_regions,nearest_support_costs


def scalar(x,rows,cfg):
    rr=[];ss={}
    for row in rows:
        a,b=hypotheses(x,row,cfg);rr.extend(a)
        for status in b.values():ss[status]=ss.get(status,0)+1
    return rr,ss


def checks():
    start=time.perf_counter();options=read('configs/sparse_ray_material_incidence_v1.json');cache=Path(options['cached_model_root']);cfg={**read(cache/'config.json'),**options};base=Path(cfg['base_root']);tag=read(base/'runs.json')[0]['scene']
    with (base/tag/'inference.pkl').open('rb') as f:inp=pickle.load(f)
    with (cache/tag/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
    i,k,l=read(cache/tag/'predictions.json')['material_shared']['selection'][0];chosen=[candidates[i][0][k],candidates[i][1][l]];slots=inp['T0'][:,:3,3];raw=[];counts=0;max_error=0.;max_cost=0.
    for t,slot in [(0,i),(1,chosen[0]['slot']),(2,chosen[1]['slot'])]:
        rr=[]
        for v,path in enumerate(sorted((base/tag).glob(f'frame_{t}_*.npz'))):
            with np.load(path) as z:frame={key:z[key].copy() for key in ['rgb','depth','C']}
            lab,_=labels(frame,slots,np.arange(24)%4,cfg)
            for row in raw_pairs(frame,lab,slot,slots[slot],t,v,cfg):row['ray_id']=len(rr);rr.append(row)
        raw.append(rr)
    x1=pose_vector(chosen[0]['T'],slots[chosen[0]['slot']]);x2=pose_vector(chosen[1]['T'],slots[chosen[1]['slot']]);H,_=all_regions(np.zeros(6),raw[0],cfg);C1,_=all_regions(x1,raw[1],cfg);ref=H+C1;base_regions,status=all_regions(x2,raw[2],cfg)
    for t,x in enumerate([np.zeros(6),x1,x2]):
        fast,fs=all_regions(x,raw[t],cfg);old,os=scalar(x,raw[t],cfg);assert fs==os and len(fast)==len(old);counts+=2
        for a,b in zip(fast,old):
            assert (a['key'],a['source_axis'],a['exit_axis'],a['ray_id'])==(b['key'],b['source_axis'],b['exit_axis'],b['ray_id']);err=float(np.max(np.abs(a['vertices']-b['vertices'])));max_error=max(max_error,err);assert err<1e-11;counts+=2
    from scipy.spatial.transform import Rotation
    v=Rotation.from_rotvec(x2[:3]).as_matrix()[:,2];old_costs=[];new_costs=[]
    for eps in np.linspace(-.08,.08,9):
        xx=x2.copy();xx[3:]+=eps*v;fast=translate_regions(base_regions,eps);direct,ss=scalar(xx,raw[2],cfg);assert ss==status and len(fast)==len(direct);counts+=2
        for a,b in zip(fast,direct):
            err=float(np.max(np.abs(a['vertices']-b['vertices'])));max_error=max(max_error,err);assert err<1e-11;counts+=1
        new=nearest_support_costs(fast,ref,len(raw[2]),cfg);old=[nearest_support_cost(direct,ref,len(raw[2]),cfg,ordered) for ordered in [False,True]]
        for a,b in zip(new,old):
            err=float(np.max(np.abs(a['per_ray_distance_m']-b['per_ray_distance_m'])));max_cost=max(max_cost,err);assert err<1e-10 and np.array_equal(a['hypothesis_available'],b['hypothesis_available']);counts+=2
        old_costs.append([v['mean_capped_squared_cost'] for v in old]);new_costs.append([v['mean_capped_squared_cost'] for v in new])
    for j in [0,1]:
        def choose(cost):
            c=np.array(cost)[:,j];best=c.min();return min(np.where(c<=best+1e-12)[0],key=lambda index:abs(np.linspace(-.08,.08,9)[index]))
        assert choose(old_costs)==choose(new_costs);counts+=1
    return {'checks':counts,'max_region_coordinate_error_m':max_error,'max_nearest_region_distance_error_m':max_cost,'GT_read':False,'selection_same':True,'elapsed_s':time.perf_counter()-start}


if __name__=='__main__':print(checks())
