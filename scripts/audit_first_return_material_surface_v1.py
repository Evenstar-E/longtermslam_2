"""Audit frozen inputs, global infeasibility accounting and conditional bounds."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha


def main():
    root=Path('results/first_return_material_surface_v1');checks=0
    for p,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/p)==h,p;checks+=1
    for p,h in read(root/'manifest.json')['sources'].items():assert sha(p)==h,p;checks+=1
    diag=read(root/'true_branch_diagnosis.json');env=read(root/'material_envelopes.json')
    for d,e in zip(diag,env):
        assert (d['scene'],d['asset'],d['method'])==(e['scene'],e['asset'],e['method']);checks+=1
        if d['true_pair_retained']:assert e['contains_truth'];checks+=1
    for run in read(root/'runs.json'):
        dest=root/run['scene'];pred=read(dest/'predictions.json');masks=np.load(dest/'feasible_masks.npz')
        with (dest/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        for i,bb in enumerate(inp['branches']):
            assert [len(x) for x in bb]==[54,54];checks+=1
            for method in ['point_visibility','first_return_surface']:
                mask=masks[f'{method}_{i}'];assert mask.shape==(54,54) and mask.dtype==bool;checks+=1
        for method,p in pred.items():
            if p['selection']:
                assert len(p['selection'])==24
                for t in [0,1]:assert len({inp['branches'][i][t][(k,l)[t]]['slot'] for i,k,l in p['selection']})==24;checks+=1
            else:assert p['status']!=0;checks+=1
        if run['geometry']=='aligned':
            prior=Path('results/completion_compatible_anchor_v1')/f"s{run['seed']}_h{run['history_views']}"
            with (prior/'inference.pkl').open('rb') as f:old=pickle.load(f)
            assert np.array_equal(inp['anchors'],old['anchors']);checks+=1
            for bb,oo in zip(inp['branches'],old['branches']):
                for bbs,oos in zip(bb,oo):
                    for b,o in zip(bbs,oos):assert b['cost']==o['cost'] and np.array_equal(b['T'],o['T']);checks+=1
            for path in dest.glob('frame_*.npz'):
                a=np.load(path);b=np.load(prior/path.name)
                for k in ['rgb','depth','C']:assert np.array_equal(a[k],b[k]);checks+=1
        if run['history_views']==3:
            other=root/f"{run['geometry']}_s{run['seed']}_h1"
            for path in dest.glob('frame_*.npz'):
                if path.name.startswith('frame_0'):continue
                a=np.load(path);b=np.load(other/path.name)
                for k in ['rgb','depth','C']:assert np.array_equal(a[k],b[k]);checks+=1
    save(root/'audit.json',{'status':'PASS','checks':checks,'script_sha256':sha(Path(__file__))})
    print({'status':'PASS','checks':checks})


if __name__=='__main__':main()
