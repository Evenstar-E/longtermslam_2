from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.completion_compatible_anchor_v1 import conflict,pair_displacements,envelope,lazy_envelope

def main():
    root=Path('results/completion_compatible_anchor_v1');seal=read(root/'prediction_seal.json')
    for path,value in seal['sha256'].items():assert sha(root/path)==value
    rows=[]
    for run in read(root/'runs.json'):
        folder=root/run['scene']
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        for repeat in range(3):
            times={};checks=0
            for method in (['exhaustive','lazy'] if repeat%2==0 else ['lazy','exhaustive']):
                start=time.perf_counter();out=[]
                for h,bb,a in zip(p['history'],p['branches'],p['anchors']):
                    d=pair_displacements(*bb,a)
                    if method=='exhaustive':
                        mask=np.array([[conflict([h,x,y]) is None for y in bb[1]] for x in bb[0]])
                        result=envelope(d,mask)
                    else:result=lazy_envelope(h,bb,d);checks+=result['checks']
                    out.append(result)
                times[method]=time.perf_counter()-start
                if method=='exhaustive':reference=out
                else:current=out
            for a,b in zip(reference,current):
                assert a['empty']==b['empty']
                if not a['empty']:
                    assert np.allclose(a['lower'],b['lower'],atol=1e-12) and np.allclose(a['upper'],b['upper'],atol=1e-12)
            rows.append({'scene':run['scene'],'repeat':repeat,**times,'lazy_checks':checks,'full_checks':run['complete_pairs']})
    save(root/'timing_control.json',{'rows':rows,'GT_read':False,'all_bounds_equal':True})
    print(rows,flush=True)

if __name__=='__main__':main()
