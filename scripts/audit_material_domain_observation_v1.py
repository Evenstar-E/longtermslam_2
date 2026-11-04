"""Complete pools, graph optimality, lower-bound refinement and input audit."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.material_domain_observation_v1 import cut_energy,shared_graph,reduced_global
from scripts.run_history_conditioned_completion_evidence_v1 import solve as full_global


def main():
    root=Path('results/material_domain_observation_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);planes=Path(cfg['plane_root']);checks=0
    for p,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/p)==h,p;checks+=1
    manifest=read(root/'manifest.json')
    for p,h in manifest['sources'].items():assert sha(p)==h,p;checks+=1
    assert sha(base/'prediction_seal.json')==manifest['base_seal'];assert sha(planes/'prediction_seal.json')==manifest['plane_seal'];checks+=2
    for old in [base,planes]:
        for p,h in read(old/'prediction_seal.json')['sha256'].items():assert sha(old/p)==h,p;checks+=1
    full_control=[]
    for run_index,run in enumerate(read(root/'runs.json')):
        dest=root/run['scene'];src=base/run['scene'];pred=read(dest/'predictions.json');records=read(dest/'records.json');arrays=np.load(dest/'costs.npz')
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'graphs.pkl').open('rb') as f:graphs=pickle.load(f)
        for i,(bb,gg) in enumerate(zip(inp['branches'],graphs)):
            assert [len(x) for x in bb]==[54,54] and [len(x) for x in gg]==[54,54];checks+=1
            c,d,lo,known=[arrays[f'{m}_{i}'] for m in ['coverage','independent','shared_lower','shared_exact']]
            assert c.shape==d.shape==lo.shape==known.shape==(54,54);assert np.isfinite(c).all() and np.isfinite(d).all();checks+=2
            assert np.all(d>=c-1e-10) and np.all(lo>=d-1e-10);assert np.array_equal(lo[~known],d[~known]);checks+=2
            # Actual data graphs: first/middle/cost-best branch, first/middle patch.
            for t in [0,1]:
                indices=sorted(set([0,27,int(np.argmin([b['cost'] for b in bb[t]]))]))
                for k in indices:
                    for j in sorted(set([0,len(gg[t][k])//2])) if gg[t][k] else []:
                        graph=gg[t][k][j];value,label=cut_energy(graph,return_labels=True)
                        assert value==graph['energy'];assert cut_energy(graph,spatial=False)==graph['coverage_energy'];checks+=2
        for method,p in pred.items():
            if p['selection']:
                assert p['status']==0 and len(p['selection'])==24;checks+=1
                for t in [0,1]:assert len({inp['branches'][i][t][(k,l)[t]]['slot'] for i,k,l in p['selection']})==24;checks+=1
                if method=='shared_material_domain':
                    assert p['certified'];checks+=1
                    selected_cost=0.
                    for i,k,l in p['selection']:
                        assert arrays[f'shared_exact_{i}'][k,l];checks+=1
                        total=0
                        for a,b in zip(graphs[i][0][k],graphs[i][1][l]):
                            total+=cut_energy(shared_graph(a,b,cfg))-a['energy']-b['energy']
                        norm=cfg['capacity_scale']*max(1,len(graphs[i][0][k]))
                        expected=arrays[f'independent_{i}'][k,l]+total/norm
                        assert abs(expected-arrays[f'shared_lower_{i}'][k,l])<1e-10;checks+=1;selected_cost+=expected
                    assert abs(selected_cost-p['objective'])<1e-8;checks+=1
            else:assert p['status']!=0;checks+=1
        assert pred['ordinary_nn']['selection']==read(src/'predictions.json')['ordinary_nn']['selection'];checks+=1
        # One full 69,984-variable pose-pair solve independently verifies slot
        # elimination on actual full data, beyond the small enumerated controls.
        if run_index==0:
            costs=[arrays[f'independent_{i}'] for i in range(24)]
            full=full_global(inp['branches'],[np.ones((54,54),bool) for i in range(24)],[-c for c in costs],20.)
            assert full['status']==0 and abs(full['negative_evidence']-pred['independent_domains']['objective'])<1e-7;checks+=1
            full_control.append({'scene':run['scene'],'full_variables':full['variables'],'reduced_variables':pred['independent_domains']['variables'],
                'full_objective':full['negative_evidence'],'reduced_objective':pred['independent_domains']['objective']})
        for before,after in zip(read(dest/'lazy_log.json'),read(dest/'lazy_log.json')[1:]):assert after['lower_bound']>=before['lower_bound']-1e-8;checks+=1
    save(root/'audit.json',{'status':'PASS','checks':checks,'full_pool_control':full_control,'script_sha256':sha(Path(__file__))});print({'status':'PASS','checks':checks})


if __name__=='__main__':main()
