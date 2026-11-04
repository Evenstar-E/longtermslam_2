"""Seals, complete-pool nesting, LP controls, assignments and bound audit."""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha


def main():
    root=Path('results/shared_plane_visibility_mainline_v1');manifest=read(root/'manifest.json');cfg=read(root/'config.json');base=Path(cfg['base_root']);n=0
    for p,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/p)==h,p;n+=1
    for p,h in manifest['sources'].items():assert sha(p)==h,p;n+=1
    assert sha(base/'prediction_seal.json')==manifest['base_seal_sha256'];n+=1
    for p,h in manifest['base_sealed_inputs'].items():assert sha(base/p)==h,p;n+=1
    for run in read(root/'runs.json'):
        dest=root/run['scene'];src=base/run['scene'];pred=read(dest/'predictions.json');masks=np.load(dest/'feasible_masks.npz')
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'history_patches.pkl').open('rb') as f:patches=pickle.load(f)
        assert len(patches)==24;n+=1
        for i,bb in enumerate(inp['branches']):
            assert [len(x) for x in bb]==[54,54];n+=1
            a,b,c=[masks[f'{m}_{i}'] for m in ['nominal_plane','independent_period_planes','shared_period_plane']]
            assert a.shape==b.shape==c.shape==(54,54);assert not np.any(a&~c);assert not np.any(c&~b);n+=3
            for patch in patches[i]:
                assert np.all(patch['coefficients']@patch['A'].T<=patch['b'][None,:]+2e-12);n+=1
        for row in read(dest/'numerical_controls.json'):assert row['fast']==row['explicit'] and row['explicit'] is not None;n+=1
        assert run['solver_counts'].get('lp_status_1',0)==0 and run['solver_counts'].get('lp_status_3',0)==0 and run['solver_counts'].get('lp_status_4',0)==0;n+=1
        for method,p in pred.items():
            if p['selection']:
                assert p['status']==0 and len(p['selection'])==24;n+=1
                for t in [0,1]:assert len({inp['branches'][i][t][(k,l)[t]]['slot'] for i,k,l in p['selection']})==24;n+=1
                for i,k,l in p['selection']:
                    if method!='ordinary_nn':assert masks[f'{method}_{i}'][k,l];n+=1
            else:assert p['status']!=0;n+=1
        assert pred['ordinary_nn']['selection']==read(src/'predictions.json')['ordinary_nn']['selection'];n+=1
    for env in read(root/'material_envelopes.json'):
        if env['true_pair_retained']:assert env['contains_truth'];n+=1
    save(root/'audit.json',{'status':'PASS','checks':n,'script_sha256':sha(Path(__file__))});print({'status':'PASS','checks':n})


if __name__=='__main__':main()
