from pathlib import Path
import pickle
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.completion_compatible_anchor_v1 import conflict,pairwise_ok,cells

def main():
    root=Path('results/completion_compatible_anchor_v1');seal=read(root/'prediction_seal.json');checks=0
    assert seal['complete']
    for path,value in seal['sha256'].items():assert sha(root/path)==value;checks+=1
    for path,value in read(root/'manifest.json')['sources'].items():assert sha(path)==value;checks+=1
    samples=read(root/'observation_soundness.json')
    for r in samples:
        assert not any(r['forbidden_true_occupied']) and not any(r['hit_bands_without_true_occupancy']) and r['true_full_retained'];checks+=1
    for seed in read(root/'config.json')['seeds']:
        for t in [1,2]:
            for v in [0,1]:
                with np.load(root/f's{seed}_h1'/f'frame_{t}_{v}.npz') as a,np.load(root/f's{seed}_h3'/f'frame_{t}_{v}.npz') as b:
                    for key in ['rgb','depth','C']:assert np.array_equal(a[key],b[key]);checks+=1
    for run in read(root/'runs.json'):
        folder=root/run['scene'];records=read(folder/'records.json');masks=np.load(folder/'feasible_masks.npz')
        for i,r in enumerate(records):
            pm,fm=masks[f'pair_{i}'],masks[f'full_{i}']
            assert np.all(~fm|pm) and int((pm&~fm).sum())==r['joint_only_rejections'];checks+=1
        controls=read(folder/'explicit_controls.json')
        for c in controls:assert c['feasible'] is not None and c['feasible']==c['eliminated_feasible'];checks+=1
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        for e in read(folder/'joint_contradictions.json'):
            i=e['asset'];gg=[p['history'][i],p['branches'][i][0][e['k1']],p['branches'][i][1][e['k2']]]
            assert pairwise_ok(gg) and conflict(gg) is not None
            B=set(e['band_cells']);covered=set().union(*(set(x) for x in e['free_coverage_by_period']))
            assert B<=covered and all(not B<=set(x) for x in e['free_coverage_by_period']);checks+=1
        pred=read(folder/'predictions.json')
        for method,x in pred.items():
            assert x['status']==0 and x['selection'] is not None
            if method!='plain_nn':
                maskname='pair' if method=='pairwise' else 'full'
                for i,k,l in x['selection']:assert masks[f'{maskname}_{i}'][k,l];checks+=1
    output={'passed':True,'checks':checks,'prediction_files_unchanged':True,'paired_current_frames_equal':True,
            'full_subset_pairwise':True,'all_true_branches_retained':True,'all_explicit_controls_resolved_equal':True}
    save(root/'audit.json',output)
    save(root/'evidence_index.json',{'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='evidence_index.json'}})
    print(output)

if __name__=='__main__':main()
