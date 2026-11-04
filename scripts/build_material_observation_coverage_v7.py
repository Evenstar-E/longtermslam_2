"""Fresh TRAIN-only full24 acquisition; frozen V1 renderer, no overwrites."""
import json,time
from pathlib import Path
import numpy as np
from inspection_decisions.textured_material_dataset_v1 import make_scene,view_specs,camera,render
from scripts.build_textured_material_dataset_v1 import save,sha


def views_for_slots(scene,cfg):
    views=[v for v in view_specs(scene,cfg) if v['kind']=='overview']
    for e in range(3):
        for slot in range(24):
            rng=np.random.default_rng(scene['seed']+880000+e*1000+slot)
            target=scene['slots'][slot];offset=np.array([-.35+.30*e,.26-.10*e,-1.45])
            if e==2 and slot%cfg['rear_view_slot_modulo']==0:offset[2]=1.45
            offset+=rng.uniform(-.10,.10,3)
            views.append(dict(epoch=e,kind='detail',spatial_slot=slot,camera=camera(cfg['detail'],target+offset,target)))
    return views


def main():
    config=Path('configs/textured_material_dataset_v2.json');cfg=json.loads(config.read_text(encoding='utf-8'));root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=False);dest=root/'train_000';(dest/'observations').mkdir(parents=True);(dest/'supervision').mkdir()
    paths=[config,Path(__file__),Path('inspection_decisions/textured_material_dataset_v1.py'),Path('visual_reinspection/renderer.py')];sources={str(p):sha(p) for p in paths};save(root/'config.json',cfg);save(root/'source_manifest.json',sources);start=time.perf_counter()
    scene=make_scene(cfg,cfg['generator_namespace_index'],0);views=views_for_slots(scene,cfg)
    old=json.loads(Path('data/textured_material_v1/dataset_index.json').read_text(encoding='utf-8'))
    for field in ['shape_keys','texture_keys']:
        assert not set(map(int,scene[field]))&{int(k) for r in old for k in r[field]}
    save(dest/'supervision/scene.json',scene);save(dest/'supervision/acquisition.json',views)
    save(dest/'public_metadata.json',dict(split='train',nominal_spatial_centers=scene['slots'],calibration='C camera-to-world, depth camera-z, integer pixel centers',anonymous_slot_method='Nearest nominal center from measured depth within0.70m; no persistent identity',oracle_instance_masks_are_private=True))
    frames=[]
    for fi,view in enumerate(views):
        obs,truth=render(scene,view,cfg,fi);name=f'frame_{fi:03d}.npz';np.savez_compressed(dest/'observations'/name,**obs);np.savez_compressed(dest/'supervision'/name,**truth)
        # No persistent acquisition group, target instance, or camera recipe here.
        frames.append(dict(index=fi,epoch=view['epoch'],kind=view['kind'],observation=str(Path('observations')/name),labels=str(Path('supervision')/name),width=view['camera']['width'],height=view['camera']['height']))
        if (fi+1)%8==0:save(root/'build_progress.json',dict(frames=fi+1,total=len(views),elapsed_s=time.perf_counter()-start));print('V7 RENDER',fi+1,'/',len(views),flush=True)
    save(dest/'frames.json',frames);save(root/'dataset_index.json',[dict(scene='train_000',split='train',frames=len(frames),asset_tasks=24,seed=scene['seed'],shape_keys=scene['shape_keys'],texture_keys=scene['texture_keys'])])
    save(root/'acquisition_budget.json',{c:dict(frames=len(rr),raw_pixels=sum(r['width']*r['height'] for r in rr),frames_per_epoch=[sum(r['epoch']==e for r in rr) for e in range(3)]) for c in cfg['conditions'] for rr in [[r for r in frames if c=='full24' or r['kind']=='overview']]})
    assert all(sha(p)==h for p,h in sources.items())
    save(root/'release_manifest.json',dict(complete=True,TRAIN_only=True,source_hashes=sources,files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},elapsed_s=time.perf_counter()-start))
    print('V7 DATA SEALED',len(frames),round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':main()
