"""Generate split-disjoint textured multiasset data; never overwrite an old root."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from inspection_decisions.textured_material_dataset_v1 import make_scene, view_specs, render, correspondence_labels


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preview(dest, records):
    selected=[r for r in records if r['kind']=='detail' and r['group']==0]
    canvas=Image.new('RGB',(4*256,3*284+36),'#eeeeee');draw=ImageDraw.Draw(canvas)
    names=['flat','rich','repeated','changed'];keys=['rgb_flat','rgb','rgb_repeated','rgb_changed']
    for c,n in enumerate(names):draw.text((c*256+8,8),n,fill='black')
    for row,rec in enumerate(selected):
        with np.load(dest/rec['observation']) as f:
            for col,key in enumerate(keys):
                im=Image.fromarray(f[key]).resize((256,256))
                canvas.paste(im,(col*256,36+row*284))
        draw.text((8,36+row*284+260),f'epoch {rec["epoch"]}: same physical asset, known camera',fill='black')
    canvas.save(dest/'texture_preview.png')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/textured_material_dataset_v1.json');args=ap.parse_args()
    cfg=read(args.config);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),Path('inspection_decisions/textured_material_dataset_v1.py'),Path('visual_reinspection/renderer.py'),Path(args.config)]
    save(root/'config.json',cfg);save(root/'source_manifest.json',{str(p):sha(p) for p in source_paths})
    records=[];start=time.perf_counter()
    for si,(split,count) in enumerate(cfg['splits'].items()):
        for wi in range(count):
            tag=f'{split}_{wi:03d}';dest=root/tag;public=dest/'observations';private=dest/'supervision';public.mkdir(parents=True);private.mkdir()
            scene=make_scene(cfg,si,wi);views=view_specs(scene,cfg);frames=[]
            save(private/'scene.json',scene)
            save(dest/'public_metadata.json',dict(split=split,nominal_spatial_centers=scene['slots'],
                 calibration='C is camera-to-world; depth is camera-z metres; K uses integer pixel centers',
                 anonymous_slot_method='Nearest nominal center from measured depth, radius 0.70m; no persistent ID or RGB',
                 oracle_instance_masks_are_private=True))
            for fi,view in enumerate(views):
                obs,truth=render(scene,view,cfg,fi)
                name=f'frame_{fi:03d}.npz';np.savez_compressed(public/name,**obs);np.savez_compressed(private/name,**truth)
                r=dict(index=fi,epoch=view['epoch'],kind=view['kind'],group=view['group'],
                       observation=str(Path('observations')/name),labels=str(Path('supervision')/name),
                       width=view['camera']['width'],height=view['camera']['height'],
                       foreground_pixels=int((truth['instance']>=0).sum()),
                       changed_pixels=int(truth['changed_material_mask'].sum()))
                frames.append(r)
            # All three period pairs for four detail groups; predefined overview pairs.
            pairs=[]
            for group in range(cfg['detail_assets_per_scene']):
                group_frames=[r['index'] for r in frames if r['kind']=='detail' and r['group']==group]
                pairs += [(group_frames[a],group_frames[b]) for a,b in [(0,1),(0,2),(1,2)]]
            pairs += [(0,1),(0,3),(0,5),(1,4),(2,6),(3,5)]
            pair_records=[]
            for pi,(a,b) in enumerate(pairs):
                with np.load(private/f'frame_{a:03d}.npz') as f:truth={k:f[k] for k in ['instance','material_xyz']}
                labels=correspondence_labels(scene,views[a],views[b],truth,cfg['correspondence_samples'],scene['seed']+pi)
                path=private/f'pair_{pi:03d}.npz';np.savez_compressed(path,**labels)
                counts=np.bincount(labels['status'],minlength=3)
                pair_records.append(dict(source=a,target=b,labels=str(path.relative_to(dest)),
                     total=int(counts.sum()),visible=int(counts[0]),outside=int(counts[1]),occluded=int(counts[2])))
            save(dest/'frames.json',frames);save(private/'pairs.json',pair_records)
            preview(dest,frames)
            rec=dict(scene=tag,split=split,frames=len(frames),asset_tasks=cfg['objects'],pairs=len(pairs),
                     correspondences=sum(r['total'] for r in pair_records),visible_correspondences=sum(r['visible'] for r in pair_records),
                     shape_keys=scene['shape_keys'],texture_keys=scene['texture_keys'],seed=scene['seed'])
            records.append(rec);save(root/'build_progress.json',dict(completed=records,elapsed_s=time.perf_counter()-start))
            print('GENERATED',tag,'frames',len(frames),'visible_pairs',rec['visible_correspondences'],flush=True)
    # Split membership and generation seeds are supervision, not model inputs.
    save(root/'dataset_index.json',records)
    save(root/'dataset_seal.json',dict(complete=True,no_model_predictions=True,
         source_hashes_unchanged=all(sha(p)==h for p,h in read(root/'source_manifest.json').items()),
         elapsed_s=time.perf_counter()-start,
         sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))
    print('DATASET_SEALED',sum(r['frames'] for r in records),'frames',flush=True)


if __name__=='__main__':main()
