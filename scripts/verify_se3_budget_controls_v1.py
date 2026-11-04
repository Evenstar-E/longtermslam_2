from pathlib import Path
import numpy as np
from comparative_evidence.estimation_v1 import world_components
from comparative_evidence.evaluation_v3 import evaluate
from inspection_decisions.projective_visibility_v1 import decode
from inspection_decisions.se3_visibility_v1 import readout
from inspection_decisions.graph_dataset import digest,canonical_bytes
from scripts.run_se3_visibility_v1 import CFG,cases,scene_root
from scripts.run_evidence_value_pilot_v1 import read,save


def main():
    cfg=read(CFG);source=Path(cfg['output']);out=Path('results/se3_visibility_budget_audit_v1');a=read(out/'analysis.json');contract=read(out/'diagnostic_contract.json');checks=0;extra=0
    def ck(ok):
        nonlocal checks
        assert ok;checks+=1
    for p,sha in contract['sources'].items():ck(digest(p)==sha)
    for sid,case in enumerate(cases(cfg)):
        saved=read(out/f'{sid:04d}_controls.json');options=read(out/f'{sid:04d}_options.json');b=read(source/f'{sid:04d}_backend.json');raw=read(source/f'{sid:04d}_observed.json')['raw'];total=sum(len(world_components(decode(o))) for o in raw['current'])
        scores=dict(ordinary_prior_score=[r['best']['score'] for r in options['options']],ordinary_pixel_score=[sum(v['fit']['loss'] for v in r['projective_fits'])+12*(total-sum(len(p) for p in r['best']['matches'])) for r in options['options']])
        for method,values in scores.items():
            r=saved[method];ck(np.allclose(values,r['scores'],atol=1e-12));ck(r['selected']==min(range(len(values)),key=lambda i:values[i]))
            ck(canonical_bytes(r['result'])==canonical_bytes(options['options'][r['selected']]))
            row=next(x for x in a['rows'] if x['scene']==sid and x['method']==method)
            ck(canonical_bytes(evaluate(scene_root(source,case['family'],case['variant']),r['result']['prediction']))==canonical_bytes(row['metric']))
        for k,branch in enumerate(b['registration']['retained']):
            if k!=0 and k!=b['late_selected']:
                ck(canonical_bytes(readout(raw,np.array(branch['G']),branch['matches'],cfg))==canonical_bytes(options['options'][k]));extra+=1
    save(out/'verification.json',dict(status='PASS',checks=checks,control_endpoint_reevaluations=60,extra_branch_exact_replays=extra,
        scope='Verification of POSTHOC controls; not prospective confirmation',exclusions=[]))
    print('POSTHOC CONTROL VERIFICATION PASS',checks,'endpoints60 extra branch replays',extra)


if __name__=='__main__':main()
