"""Verify input/output seals, provenance, shared poses and support constraints."""
from pathlib import Path
import json
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha,load
from inspection_decisions.reversible_surface_v2 import inverse,transform,root_mask


def main():
    root=Path('results/rescan_reversible_surface_v2');cfg=read(root/'config.json');checks=[]
    def check(ok,name):
        if not bool(ok):raise AssertionError(name)
        checks.append(name)
    seal=read(root/'prediction_seal.json');check(seal['complete'],'inference complete')
    for p,h in seal['sha256'].items():check(sha(root/p)==h,'sealed '+p)
    manifest=read(root/'manifest.json')
    for kind in ['sources','inputs']:
        for p,h in manifest[kind].items():check(sha(p)==h,kind+' '+p)
    check(not manifest['current_GT_read'],'no current truth accessed by inference')
    for scene in cfg['scenes']:
        H,O,raw,initial,_=load(scene,cfg)
        for t in [0,1]:
            check(O[t]['mass'].sum()==len(raw[t]['xyz']),f'{scene} period {t} all-point mass')
            check(len(O[t]['inverse'])==len(raw[t]['xyz']),f'{scene} period {t} inverse coverage')
        for method in cfg['methods']:
            dest=root/scene/method;state=np.load(dest/'state.npz');poses=state['poses']
            check(poses.shape==(2,len(H),4,4),f'{scene} {method} one pose per asset/period')
            check(np.allclose(np.linalg.det(poses[:,:,:3,:3]),1,atol=1e-4),f'{scene} {method} proper rotation')
            check(np.allclose(poses[:,:,:3,:3]@poses[:,:,:3,:3].transpose(0,1,3,2),np.eye(3),atol=1e-4),f'{scene} {method} orthogonal rotation')
            if method in ['frozen','hard_forward','rooted_forward']:
                check(np.array_equal(poses[0],np.array(initial[0])),f'{scene} {method} frozen source pose')
                check(np.array_equal(state['labels1'],state['initial_labels1']),f'{scene} {method} frozen source labels')
            for t in [0,1]:
                p=np.load(dest/f'prediction{t+1}.npz');check(len(p['identity'])==len(raw[t]['xyz']),f'{scene} {method} epoch {t} all predictions')
                check(np.isfinite(p['cost']).all(),f'{scene} {method} epoch {t} finite costs')
                for i,h in enumerate(H):
                    check(np.allclose(state['anchors_world'][t,i],transform(h['anchor'][None],poses[t,i])[0]),f'{scene} {method} epoch {t} anchor {i}')
                file=dest/f'source_for{t+1}.npz'
                if not file.exists():
                    check(np.all(p['source_voxel']==-1),f'{scene} {method} epoch {t} historical-only provenance');continue
                inc=np.load(file);other=1-t;seen=[]
                for i,h in enumerate(H):
                    idx=inc[f'source_voxel_{i}'];seen.extend(idx.tolist())
                    check(np.all(state[f'labels{other+1}'][idx]==i),f'{scene} {method} epoch {t} source owner {i}')
                    check(np.allclose(inc[f'xyz_{i}'],inverse(O[other]['xyz'][idx],poses[other,i])),f'{scene} {method} epoch {t} source coordinates {i}')
                    check(np.array_equal(inc[f'rgb_{i}'],O[other]['rgb'][idx]),f'{scene} {method} epoch {t} source color {i}')
                    if method.startswith('rooted'):
                        allidx=np.flatnonzero(state[f'labels{other+1}']==i)
                        keep,_=root_mask(inverse(O[other]['xyz'][allidx],poses[other,i]),O[other]['rgb'][allidx],h,cfg)
                        check(np.array_equal(idx,allidx[keep]),f'{scene} {method} epoch {t} rooted eligibility {i}')
                    active=(p['identity']==h['uid'])&(p['source_voxel']>=0)
                    check(np.all(np.isin(p['source_voxel'][active],idx)),f'{scene} {method} epoch {t} prediction lineage {i}')
                check(len(seen)==len(set(seen)),f'{scene} {method} epoch {t} source observation single owner')
    scores=read(root/'scores.json')
    for s in scores['summaries']:
        rr=[r for r in scores['rows'] if all(r[k]==s[k] for k in ['scene','method','period'])]
        check(sum(r['strict_correct'] for r in rr if r['strict_persistent'])==s['strict_correct'],'score arithmetic '+str(s))
    save(root/'audit.json',{'passed':True,'checks':len(checks),'source_sha':sha(__file__),
        'scope':'Reproducibility/provenance/shared-pose checks, not evidence of novelty or independent ground-truth motion.'})
    print(json.dumps({'passed':True,'checks':len(checks)}))


if __name__=='__main__':main()
