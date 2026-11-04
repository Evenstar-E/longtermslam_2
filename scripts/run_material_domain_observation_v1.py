"""Complete-pool material-domain energy experiment; no evaluation truth read."""
import pickle,time
from pathlib import Path
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.run_completion_compatible_anchor_v1 import ordinary_assignment
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.material_domain_observation_v1 import query_graph,cut_energy,reduced_global,lazy_shared


def main():
    cp=Path('configs/material_domain_observation_v1.json');options=read(cp);base=Path(options['base_root']);planes=Path(options['plane_root']);cfg={**read(base/'config.json'),**options}
    for old in [base,planes]:
        for p,h in read(old/'prediction_seal.json')['sha256'].items():assert sha(old/p)==h,p
    root=Path('results/material_domain_observation_v1');root.mkdir(exist_ok=False);save(root/'config.json',cfg)
    sources=[cp,Path(__file__),Path('inspection_decisions/material_domain_observation_v1.py'),Path('scripts/check_material_domain_observation_v1.py'),
        Path('inspection_decisions/ray_conditioned_plane_polytope_v1.py'),Path('inspection_decisions/first_return_material_surface_v1.py'),
        Path('scripts/completion_catalog_scene_v1.py'),Path('scripts/run_completion_compatible_anchor_v1.py'),Path('research/material_domain_observation_contract_v1_2026-10-24.md')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'base_seal':sha(base/'prediction_seal.json'),'plane_seal':sha(planes/'prediction_seal.json'),
        'reads_evaluation_truth':False,'familiar_DEV_data':True,'full_candidate_pools':True,'score_is_calibrated_likelihood':False})
    start=time.perf_counter();runs=[]
    for run in read(base/'runs.json'):
        tag=run['scene'];src=base/tag;dest=root/tag;dest.mkdir();stamp=time.perf_counter()
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (planes/tag/'history_patches.pkl').open('rb') as f:patches=pickle.load(f)
        frames=[];slot_labels=[];groups=np.arange(len(inp['T0']))%4;slots=inp['T0'][:,:3,3]
        for t in [1,2]:
            ff=[];ll=[]
            for path in sorted(src.glob(f'frame_{t}_*.npz')):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                ff.append(frame);ll.append(labels(frame,slots,groups,cfg)[0])
            frames.append(ff);slot_labels.append(ll)
        all_graphs=[];coverage=[];independent=[];records=[];counts={'queried':0,'supported':0,'matches':0,'free':0};cuts=0
        for i,bb in enumerate(inp['branches']):
            began=time.perf_counter();graphs=[];cov=[];ind=[];rec={'asset':i,'patches':len(patches[i]),'branches':[]}
            for t in [0,1]:
                gg=[];c=[];d=[]
                for k,b in enumerate(bb[t]):
                    gl=[];cc=ee=matches=free=0
                    for p in patches[i]:
                        g=query_graph(p,inp['T0'][i],b['T'],frames[t],slot_labels[t],b['slot'],cfg)
                        g['energy']=cut_energy(g);g['coverage_energy']=cut_energy(g,spatial=False);cuts+=2
                        assert g['energy']>=g['coverage_energy']
                        cc+=g['coverage_energy'];ee+=g['energy'];matches+=g['matches'];free+=g['free'];gl.append(g)
                        for key in counts:counts[key]+=g['counts'].get(key,0)
                    norm=cfg['capacity_scale']*max(1,len(patches[i]));tie=cfg['tie_nn_weight']*b['cost']
                    c.append(cc/norm+tie);d.append(ee/norm+tie);gg.append(gl)
                    rec['branches'].append({'epoch':t,'branch':k,'matches':matches,'free':free,'coverage_energy_integer':cc,'domain_energy_integer':ee})
                graphs.append(gg);cov.append(np.array(c));ind.append(np.array(d))
            all_graphs.append(graphs);coverage.append(cov[0][:,None]+cov[1][None,:]);independent.append(ind[0][:,None]+ind[1][None,:]);records.append(rec)
            print('DOMAIN',tag,i,'patches',len(patches[i]),'seconds',round(time.perf_counter()-began,3),flush=True)
        with (dest/'graphs.pkl').open('wb') as f:pickle.dump(all_graphs,f)
        predictions={'ordinary_nn':ordinary_assignment(inp['branches']),'coverage_only':reduced_global(inp['branches'],coverage,cfg['global_time_limit_s']),
                     'independent_domains':reduced_global(inp['branches'],independent,cfg['global_time_limit_s'])}
        shared,lower,known,cache,log=lazy_shared(inp['branches'],all_graphs,independent,cfg);predictions['shared_material_domain']=shared
        save(dest/'predictions.json',predictions);save(dest/'records.json',records);save(dest/'lazy_log.json',log)
        save(dest/'shared_pair_details.json',[{'asset':i,'k':k,'l':l,**v} for (i,k,l),v in cache.items()])
        np.savez_compressed(dest/'costs.npz',**{f'{m}_{i}':a for m,arrays in [('coverage',coverage),('independent',independent),('shared_lower',lower),('shared_exact',known)] for i,a in enumerate(arrays)})
        result={**{k:run[k] for k in ['scene','geometry','seed','history_views']},'elapsed_s':time.perf_counter()-stamp,'ray_instances':counts,
            'initial_cut_calls':cuts,'exact_shared_pairs':len(cache),'lazy_rounds':shared.get('rounds'),
            'shared_certified':shared.get('certified',False),'statuses':{m:p['status'] for m,p in predictions.items()}}
        runs.append(result);save(root/'runs.json',runs);print('DONE',tag,result,flush=True)
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('SEALED',flush=True)


if __name__=='__main__':main()
