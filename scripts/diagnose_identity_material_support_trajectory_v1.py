from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,sha


def main():
    root=Path('results/identity_material_support_v1');seal=read(root/'prediction_seal.json');assert seal['complete']
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    rows=[]
    for scene in read(root/'config.json')['scenes']:
        a=np.load(root/scene/'tie_preserve/state.npz');b=np.load(root/scene/'material_support/state.npz');m=np.load(root/scene/'material_support/material_state.npz')
        assets=read(root/scene/'assets.json');withdrawn=[]
        for t in [1,2]:
            z=b[f'labels{t}'];mask=~m[f'M{t}']&(z>=0)
            for i,n in zip(*np.unique(z[mask],return_counts=True)):
                withdrawn.append({'period':t,'asset_uid':assets[i]['uid'],'class':assets[i]['class'],'voxels':int(n)})
        ta=read(root/scene/'tie_preserve/trace.json');tb=read(root/scene/'material_support/trace.json')
        rows.append({'scene':scene,'same_poses':bool(np.array_equal(a['poses'],b['poses'])),
            'identity_differences':[int(np.sum(a[f'labels{t}']!=b[f'labels{t}'])) for t in [1,2]],'withdrawn':withdrawn,
            'max_trace_energy_difference':max(abs(x['energy_after']-y['energy_after']) for x,y in zip(ta,tb))})
    save(root/'trajectory_diagnosis.json',{'rows':rows,'source_sha':sha(__file__),'scope':'Prediction-state comparison, no GT or new branch selection.'})
    print(__import__('json').dumps(rows,indent=2))


if __name__=='__main__':main()
