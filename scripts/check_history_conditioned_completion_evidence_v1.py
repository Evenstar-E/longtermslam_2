import numpy as np
from inspection_decisions.history_conditioned_completion_evidence_v1 import (
    CounterCompiler,event,probability,exhaustive_probability,sis_probability)

def main():
    rng=np.random.default_rng(20261024);compiler=CounterCompiler();count=0;errors=[]
    for trial in range(160):
        groups=[{'free':int(rng.integers(0,256)),
                 'bands':[int(rng.integers(1,256)) for _ in range(int(rng.integers(0,5)))]} for _ in range(3)]
        spec=event(groups)
        for prior in ['beta','fixed']:
            a=probability(spec,compiler,prior);b=exhaustive_probability(groups,prior)
            assert (np.isneginf(a) and np.isneginf(b)) or abs(a-b)<1e-10,(groups,a,b)
            count+=1
        if spec is not None and len(spec['clauses']):
            estimate,_=sis_probability(spec,8192,trial)
            errors.append(abs(estimate-probability(spec,compiler)))
    for trial in range(50):
        bands=[]
        for _ in range(8):
            ids=rng.choice(8,int(rng.integers(2,5)),replace=False)
            bands.append(sum(1<<int(j) for j in ids))
        groups=[{'free':0,'bands':bands}];spec=event(groups)
        for prior in ['beta','fixed']:
            assert abs(probability(spec,compiler,prior)-exhaustive_probability(groups,prior))<1e-10
            count+=1
        estimate,_=sis_probability(spec,8192,1000+trial)
        errors.append(abs(estimate-probability(spec,compiler)))
    groups=[{'free':0,'bands':[3,12]}]
    assert abs(probability(event(groups),compiler)-exhaustive_probability(groups))<1e-12
    duplicate=[{'free':0,'bands':[3,12,3,12]}]
    assert probability(event(groups),compiler)==probability(event(duplicate),compiler)
    # Disconnected shape variables remain coupled by the shared rho prior.
    joint=np.exp(probability(event(groups),compiler))
    wrong=np.exp(probability(event([{'free':0,'bands':[3]}]),compiler))**2
    assert abs(joint-wrong)>.001
    assert max(errors,default=0)<.035,errors
    print({'brute_force_prior_checks':count,'SIS8192_cases':len(errors),'max_SIS_logerror':max(errors,default=0),
           'duplicate_invariance':True,'shared_density_integration':True,'compiler_calls':compiler.calls,'cache_hits':compiler.hits})

if __name__=='__main__':main()
