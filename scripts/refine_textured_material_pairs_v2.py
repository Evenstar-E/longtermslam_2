"""Preserve all labels, export target subpixels as float64 for grazing rays.

Source images, old labels and original dataset seal remain unchanged.
No examples are removed or selected on model performance.
"""
from pathlib import Path
import numpy as np
from scripts.build_textured_material_dataset_v1 import read,save,sha
from inspection_decisions.textured_material_dataset_v1 import make_scene,view_specs,transform,inverse,changed_mask


def main():
    root=Path('data/textured_material_v1');cfg=read(root/'config.json');paths=[];count=0
    for si,(split,n) in enumerate(cfg['splits'].items()):
        for wi in range(n):
            dest=root/f'{split}_{wi:03d}';scene=make_scene(cfg,si,wi);views=view_specs(scene,cfg);records=[]
            for pi,pr in enumerate(read(dest/'supervision/pairs.json')):
                with np.load(dest/pr['labels']) as f:labels=dict(f)
                mat=labels['material_xyz'].astype(float);ids=labels['instance'];target=views[pr['target']];cam=target['camera']
                world=np.empty_like(mat)
                for aid in np.unique(ids):
                    ix=ids==aid;world[ix]=transform(mat[ix],scene['poses'][target['epoch'],aid])
                pc=inverse(world,cam['C']);z=pc[:,2];safe=np.maximum(z,1e-9)
                labels['target_uv']=pc[:,:2]/safe[:,None]*[cam['K'][0,0],cam['K'][1,1]]+cam['K'][:2,2]
                labels['expected_target_z']=z
                # Additional supervision distinguishes physical material identity from paint persistence.
                with np.load(dest/'supervision'/f'frame_{pr["source"]:03d}.npz') as f:
                    uv=labels['source_uv'];faces=f['face'][uv[:,1],uv[:,0]]
                epoch_change=(views[pr['source']]['epoch']==2)!=(target['epoch']==2)
                labels['appearance_changed_in_changed_variant']=epoch_change & np.isin(ids,scene['changed_assets']) & changed_mask(mat,faces)
                path=dest/'supervision'/f'pair_v2_{pi:03d}.npz'
                if path.exists():raise FileExistsError(path)
                np.savez_compressed(path,**labels);paths.append(path);count+=len(ids)
                records.append({**pr,'labels':str(path.relative_to(dest))})
            p=dest/'supervision/pairs_v2.json';save(p,records);paths.append(p)
    save(root/'pair_labels_v2_seal.json',dict(complete=True,sampled_correspondences=count,
        reason='float32 subpixel rounding amplified at grazing surfaces; same observations and labels, float64 export',
        no_correspondences_removed=True,source_sha256=sha(Path(__file__)),
        sha256={str(p.relative_to(root)):sha(p) for p in paths}))
    print('PAIR_V2_SEALED',count,'samples',flush=True)


if __name__=='__main__':main()
