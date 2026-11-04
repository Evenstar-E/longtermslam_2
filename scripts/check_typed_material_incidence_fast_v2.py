import pickle,time
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save
from inspection_decisions import typed_material_incidence_graph_v1 as old
from inspection_decisions import typed_material_incidence_fast_v2 as fast


def checks():
    options=read('configs/typed_material_incidence_graph_v1.json');cfg={**read(Path(options['observation_root'])/'config.json'),**options};tag='aligned_s270100_h1'
    with (Path(options['observation_root'])/tag/'observations.pkl').open('rb') as f:data=pickle.load(f)
    row=read(Path(options['parent_root'])/tag/'predictions.json')[0];s1,s2=row['selected_slots'];x=np.array(row['initial_parameters']);base,graph=old.build_typed_graph([(np.zeros(6),data[0][0]),(x[:6],data[1][s1])],cfg);maximum=0.;n=0;timings={}
    for G in [base,graph]:
        d=[f['offset'] for f in G['faces']];go=old.graph_arrays(G,d);gf=fast.graph_arrays(G,d);ao=old.period_arrays(data[2][s2]);af=fast.period_arrays(data[2][s2]);rng=np.random.default_rng(20261025)
        for _ in range(80):
            xx=x[6:]+rng.normal(0,.01,6);oo=old.period_geometry(ao,xx,cfg);ff=fast.period_geometry(af,xx,cfg)
            for line in [False,True]:
                a=old.period_cost(oo,go,cfg,line);b=fast.period_cost(ff,gf,cfg,line);maximum=max(maximum,abs(a-b));assert abs(a-b)<1e-12;n+=1
        if G is graph:
            for name,module,geo,gg in [('old',old,oo,go),('fast',fast,ff,gf)]:
                start=time.perf_counter()
                for _ in range(200):module.period_cost(geo,gg,cfg)
                timings[name+'_ms_per_score']=(time.perf_counter()-start)*5
    return {'checks':n,'max_scalar_energy_difference':maximum,'timings':timings,'GT_read':False,'only_reduction_order_changes':True}


if __name__=='__main__':
    result=checks();save('research/typed_material_fast_equivalence_v2.json',result);print(result)
