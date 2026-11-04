"""Exact same counter, cache reset per event vs reused across events; no GT."""
from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.history_conditioned_completion_evidence_v1 import CounterCompiler,probability

def main():
    root=Path('results/history_conditioned_completion_evidence_v1');seal=read(root/'prediction_seal.json')
    assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    with (root/'reference_events.pkl').open('rb') as f:cases=pickle.load(f)
    rows=[]
    for repeat in range(3):
        outputs={}
        for method in (['reset_per_event','reuse'] if repeat%2==0 else ['reuse','reset_per_event']):
            compiler=CounterCompiler();values=[];calls=hits=0;start=time.perf_counter()
            for case in cases:
                if method=='reset_per_event':compiler=CounterCompiler()
                before_c,before_h=compiler.calls,compiler.hits
                values.append(probability(case['spec'],compiler))
                calls+=compiler.calls-before_c;hits+=compiler.hits-before_h
            elapsed=time.perf_counter()-start;outputs[method]=np.array(values)
            rows.append({'repeat':repeat,'method':method,'events':len(cases),'elapsed_s':elapsed,'calls':calls,'hits':hits})
        assert np.allclose(outputs['reset_per_event'],outputs['reuse'],rtol=0,atol=1e-10)
    save(root/'cache_benchmark.json',{'rows':rows,'results_equal':True,'GT_read':False})
    print(rows)

if __name__=='__main__':main()
