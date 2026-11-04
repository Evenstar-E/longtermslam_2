"""New evidence predictions, no current GT/old scores access; seal before eval."""
from pathlib import Path
import pickle,time,hashlib,json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.history_conditioned_completion_evidence_v1 import CounterCompiler,event,probability,sis_probability


class Engine:
    def __init__(self):
        self.compiler=CounterCompiler();self.exact={};self.sample={};self.sample_hits=0
    def signature(self,spec):return (spec['free_count'],spec['unit_count'],spec['clauses'])
    def evaluate(self,spec,prior='beta',particles=None):
        if spec is None:return -np.inf
        key=self.signature(spec)
        if particles is None:
            full=(prior,key)
            if full not in self.exact:self.exact[full]=probability(spec,self.compiler,prior)
            return self.exact[full]
        full=(particles,key)
        if full in self.sample:self.sample_hits+=1
        else:
            seed=int.from_bytes(hashlib.sha256(repr(full).encode()).digest()[:8],'little')
            self.sample[full]=sis_probability(spec,particles,seed)
        return self.sample[full][0]


def solve(branches,masks,score,time_limit):
    n=len(branches);variables=[];cost=[];rr=[];cc=[];offset=0.
    for i,(bb,mask,values) in enumerate(zip(branches,masks,score)):
        valid=mask&np.isfinite(values)
        if not valid.any():return {'status':2,'selection':None,'reason':'no finite score for an asset','variables':len(variables)}
        baseline=float((-values[valid]).min());offset+=baseline
        for k,l in zip(*np.nonzero(valid)):
            j=len(variables);variables.append((i,int(k),int(l)));cost.append(float(-values[k,l]-baseline))
            rr.extend([i,n+bb[0][k]['slot'],2*n+bb[1][l]['slot']]);cc.extend([j]*3)
    A=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(3*n,len(cost))).tocsc();start=time.perf_counter()
    result=milp(np.array(cost),integrality=np.ones(len(cost)),bounds=Bounds(0,1),
                constraints=LinearConstraint(A,np.ones(3*n),np.ones(3*n)),options={'time_limit':time_limit,'mip_rel_gap':0.})
    selected=None
    if result.x is not None:
        selected=sorted(variables[j] for j in np.flatnonzero(result.x>.5))
        assert len(selected)==n and len({i for i,_,_ in selected})==n
        for t in [0,1]:assert len({branches[i][t][(k,l)[t]]['slot'] for i,k,l in selected})==n
    return {'status':int(result.status),'selection':selected,'elapsed_s':time.perf_counter()-start,'variables':len(variables),
            'negative_evidence':float(result.fun+offset) if result.fun is not None else None,
            'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None}


def main():
    cfg=read('configs/history_conditioned_completion_evidence_v1.json');base=Path(cfg['input_root']);old=read(base/'prediction_seal.json')
    for p,h in old['sha256'].items():assert sha(base/p)==h
    root=Path('results/history_conditioned_completion_evidence_v1');root.mkdir(exist_ok=False);save(root/'config.json',cfg)
    sources=[Path(__file__),Path('inspection_decisions/history_conditioned_completion_evidence_v1.py'),
             Path('scripts/check_history_conditioned_completion_evidence_v1.py'),Path('configs/history_conditioned_completion_evidence_v1.json'),
             Path('research/history_conditioned_completion_evidence_contract_v1_2026-10-24.md')]
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in sources},'base_prediction_seal':sha(base/'prediction_seal.json'),
                             'inference_reads_GT':False,'input_already_DEV':True,'same_full_mask_all_methods':True})
    engine=Engine();start=time.perf_counter();runs=[];reference_cases=[]
    for previous in read(base/'runs.json'):
        tag=previous['scene'];dest=root/tag;dest.mkdir();began=time.perf_counter()
        with (base/tag/'inference.pkl').open('rb') as f:p=pickle.load(f)
        disk=np.load(base/tag/'feasible_masks.npz');masks=[disk[f'full_{i}'] for i in range(len(p['history']))]
        scores={m:[] for m in cfg['methods']};error_rows=[];times={m:0. for m in ['exact','sis64','sis256']}
        for i,(h,bb,mask) in enumerate(zip(p['history'],p['branches'],masks)):
            stamp=time.perf_counter();hspec=event([h]);zh={prior:engine.evaluate(hspec,prior) for prior in cfg['priors']}
            singles={prior:[[engine.evaluate(event([h,b]),prior)-zh[prior] for b in bt] for bt in bb] for prior in cfg['priors']}
            arrays={m:np.full(mask.shape,-np.inf) for m in cfg['methods']}
            for prior in cfg['priors']:
                arrays[prior+'_independent_epoch']=np.array(singles[prior][0])[:,None]+np.array(singles[prior][1])[None,:]
                arrays[prior+'_independent_epoch'][~mask]=-np.inf
            specs={}
            for k,l in zip(*np.nonzero(mask)):
                spec=event([h,bb[0][k],bb[1][l]]);assert spec is not None
                specs[int(k),int(l)]=spec
                for prior in cfg['priors']:arrays[prior+'_shared_exact'][k,l]=engine.evaluate(spec,prior)-zh[prior]
            times['exact']+=time.perf_counter()-stamp
            for particles in cfg['particles']:
                stamp=time.perf_counter();method=f'beta_shared_sis{particles}'
                for (k,l),spec in specs.items():
                    arrays[method][k,l]=engine.evaluate(spec,particles=particles)-zh['beta']
                    error_rows.append({'asset':i,'k1':k,'k2':l,'particles':particles,
                                       'log_error':float(arrays[method][k,l]-arrays['beta_shared_exact'][k,l]) if np.isfinite(arrays[method][k,l]) else None,
                                       'zero_estimate':bool(not np.isfinite(arrays[method][k,l]))})
                times[f'sis{particles}']+=time.perf_counter()-stamp
            for method in cfg['methods']:scores[method].append(arrays[method])
            ordered=list(specs);nn=np.array([bb[0][k]['cost']+bb[1][l]['cost'] for k,l in ordered])
            for index in [0,len(ordered)//2,int(np.argmin(nn))]:
                k,l=ordered[index];reference_cases.append({'scene':tag,'asset':i,'k1':k,'k2':l,'spec':specs[k,l]})
            print('EVIDENCE',tag,i,'pairs',int(mask.sum()),'compiler/cache',engine.compiler.calls,len(engine.compiler.cache),
                  'elapsed',round(time.perf_counter()-start,2),flush=True)
            save(root/'monitor.json',{'scene':tag,'asset':i,'elapsed_s':time.perf_counter()-start})
        predictions={m:solve(p['branches'],masks,scores[m],cfg['global_time_limit_s']) for m in cfg['methods']}
        save(dest/'predictions.json',predictions);save(dest/'sampling_errors.json',error_rows)
        np.savez_compressed(dest/'scores.npz',**{f'{m}_{i}':a for m in cfg['methods'] for i,a in enumerate(scores[m])})
        run={'scene':tag,'elapsed_s':time.perf_counter()-began,'kernel_times_s':times,
             'global_status':{m:v['status'] for m,v in predictions.items()},
             'feasible_pairs':sum(int(m.sum()) for m in masks),'compiler_cache_size':len(engine.compiler.cache),
             'exact_event_cache_size':len(engine.exact),'sampling_cache_size':len(engine.sample),'sampling_cache_hits':engine.sample_hits}
        runs.append(run);save(root/'runs.json',runs);print('DONE',run,flush=True)
    # Reference cases selected from full-feasible pool by fixed no-GT rules.
    with (root/'reference_events.pkl').open('wb') as f:pickle.dump(reference_cases,f)
    save(root/'compiler_stats.json',{'calls':engine.compiler.calls,'hits':engine.compiler.hits,'splits':engine.compiler.splits,
                                    'keys':len(engine.compiler.cache),'exact_events':len(engine.exact),'sampling_events':len(engine.sample)})
    save(root/'prediction_seal.json',{'complete':True,'elapsed_s':time.perf_counter()-start,
         'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('SEALED ALL EVIDENCE',flush=True)


if __name__=='__main__':main()
