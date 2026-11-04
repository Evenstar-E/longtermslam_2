"""Add predetermined cross-detail pairs to cover outside-view negatives.

No original image, pair or frozen label is changed. Existing pairs saw the
entire selected target; four extra pairs per scene test a different field of view.
"""
from pathlib import Path
import numpy as np
from scripts.build_textured_material_dataset_v1 import read,save,sha
from inspection_decisions.textured_material_dataset_v1 import make_scene,view_specs,correspondence_labels,transform,inverse,changed_mask


def main():
    root=Path('data/textured_material_v1');cfg=read(root/'config.json');paths=[];added=0
    for si,(split,n) in enumerate(cfg['splits'].items()):
        for wi in range(n):
            dest=root/f'{split}_{wi:03d}';scene=make_scene(cfg,si,wi);views=view_specs(scene,cfg)
            records=read(dest/'supervision/pairs_v2.json')
            for group in range(cfg['detail_assets_per_scene']):
                a=7+3*group;b=7+3*((group+1)%cfg['detail_assets_per_scene'])+1
                with np.load(dest/'supervision'/f'frame_{a:03d}.npz') as f:truth=dict(f)
                p=correspondence_labels(scene,views[a],views[b],truth,cfg['correspondence_samples'],scene['seed']+88000+group)
                mat=p['material_xyz'].astype(float);world=np.zeros_like(mat)
                for aid in np.unique(p['instance']):
                    ix=p['instance']==aid;world[ix]=transform(mat[ix],scene['poses'][views[b]['epoch'],aid])
                cam=views[b]['camera'];pc=inverse(world,cam['C']);z=pc[:,2]
                p['target_uv']=pc[:,:2]/np.maximum(z,1e-9)[:,None]*[cam['K'][0,0],cam['K'][1,1]]+cam['K'][:2,2]
                p['expected_target_z']=z;p['appearance_changed_in_changed_variant']=np.zeros(len(z),bool)
                path=dest/'supervision'/f'pair_crossview_{group:03d}.npz'
                if path.exists():raise FileExistsError(path)
                np.savez_compressed(path,**p);paths.append(path);added+=len(z);c=np.bincount(p['status'],minlength=3)
                records.append(dict(source=a,target=b,labels=str(path.relative_to(dest)),total=len(z),
                    visible=int(c[0]),outside=int(c[1]),occluded=int(c[2]),purpose='different field of view; no assumed match'))
            path=dest/'supervision/pairs_v3.json';save(path,records);paths.append(path)
    save(root/'pair_labels_v3_seal.json',dict(complete=True,added_correspondences=added,
        previous_pair_seal_sha256=sha(root/'pair_labels_v2_seal.json'),
        source_sha256=sha(Path(__file__)),sha256={str(p.relative_to(root)):sha(p) for p in paths}))
    print('CROSSVIEW_COVERAGE_ADDED',added,flush=True)


if __name__=='__main__':main()
