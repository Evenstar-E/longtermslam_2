"""Postseal GT-assisted mechanism controls, never new algorithm predictions."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.completion_compatible_anchor_v1 import pair_displacements,envelope

def main():
    root=Path('results/completion_compatible_anchor_v1');seal=read(root/'prediction_seal.json')
    for path,value in seal['sha256'].items():assert sha(root/path)==value
    rows=[]
    for run in read(root/'runs.json'):
        folder=root/run['scene']
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        with (folder/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        masks=np.load(folder/'feasible_masks.npz')
        for i,(h,bb,a) in enumerate(zip(p['history'],p['branches'],p['anchors'])):
            base=masks[f'full_{i}'];S=truth['occupancy'][i]
            known_shape=[np.array([not(b['free']&S) and all(band&S for band in b['bands']) for b in bt]) for bt in bb]
            shape=known_shape[0][:,None]&known_shape[1][None,:]
            known_id=[np.array([b['slot']==truth[f'slot{t+1}'][i] for b in bt]) for t,bt in enumerate(bb)]
            identity=known_id[0][:,None]&known_id[1][None,:]
            d=pair_displacements(*bb,a)
            for name,mask in [('original',base),('oracle_identity',base&identity),('oracle_shape',shape),('oracle_both',shape&identity)]:
                e=envelope(d,mask)
                rows.append({'scene':run['scene'],'asset':i,'condition':name,'pairs':int(mask.sum()),
                             'report':e['report'],'radius_m':e['radius_m'],'empty':e['empty']})
    summary={}
    for name in sorted({r['condition'] for r in rows}):
        rr=[r for r in rows if r['condition']==name]
        summary[name]={'n':len(rr),'reports':sum(r['report'] for r in rr),'empty':sum(r['empty'] for r in rr),
                       'median_radius_m':float(np.median([r['radius_m'] for r in rr if not r['empty']])),
                       'median_pairs':float(np.median([r['pairs'] for r in rr]))}
    save(root/'ambiguity_diagnosis.json',{'GT_assisted':True,'changes_predictions':False,'rows':rows,'summary':summary})
    print(summary)

if __name__=='__main__':main()
