"""Point-level score after learned relabeling of actual native masks; GT used only here."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply
from scripts.run_livingscenes_real_v1 import read,save,sha

def main():
    out=Path('results/livingscenes_native_masks_v1');seal=read(out/'prediction_seal.json')
    for name,h in seal['sha256'].items():assert sha(out/name)==h
    cfg=read(out/'config.json');root=Path(cfg['public_root']);public=read(root/'public.json');pred=read(out/'predictions.json')
    identity=read('data/real_cross_session_v1/prepared_v2/evaluation/identity.json');rows=[];caps=[]
    for scene in cfg['spaces']:
        truth=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply')
        rawids=truth['instance_idx'];archive={identity[scene]['0'][h['token']] for h in public[scene]['0'] if h['eligible']}
        masks={}
        for c in public[scene]['1']:
            if c['eligible']:
                with np.load(root/c['path']) as z:masks[c['token']]=z['point_indices']
        eq={};p=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                left,right=line.split('|');eq[int(left)]=[int(v) for v in right.split()]
        objects=[]
        for identifier in np.unique(rawids):
            mask=rawids==identifier;classes,counts=np.unique(truth['class_idx'][mask],return_counts=True);cls=int(classes[np.argmax(counts)])
            if cls in [0,1,2,22] or mask.sum()<128:continue
            identifier=int(identifier);allowed=eq.get(identifier,[identifier]);persistent=bool(set(allowed)&archive)
            objects.append((identifier,mask,allowed,persistent,cls))
            overlap=sorted([int((rawids[ix]==identifier).sum()) for ix in masks.values()],reverse=True)
            strictcap=(overlap[0] if overlap else 0)/mask.sum()
            allowedcap=sum(overlap[:len(set(allowed)&archive)])/mask.sum() if persistent else None
            caps.append(dict(scene=scene,instance=identifier,persistent=persistent,raw_points=int(mask.sum()),strict_one_group_coverage_cap=float(strictcap),
                allowed_identity_one_to_one_cap=None if allowedcap is None else float(allowedcap),top_group_point_counts=overlap[:4],
                interpretation='Evaluator-only upper bound for fixed disjoint native masks with one-to-one archive matching; NOT camera observability bound or simultaneous achievable guarantee.'))
        for method in cfg['methods']:
            labels=np.full(len(truth),-1,dtype=np.int64)
            for r in pred:
                if r['scene']!=scene or r['method']!=method or r['history'] is None:continue
                labels[masks[r['current']]]=identity[scene]['0'][r['history']]
            for identifier,mask,allowed,persistent,cls in objects:
                u,n=np.unique(labels[mask],return_counts=True);k=int(np.argmax(n));dominant=int(u[k]);purity=float(n[k]/mask.sum());fraction=float(np.isin(labels[mask],allowed).mean())
                rows.append(dict(scene=scene,method=method,instance=identifier,cls=cls,persistent=persistent,correct=fraction>=.8,correct_fraction=fraction,
                    dominant_prediction=dominant,purity=purity,status='correct' if fraction>=.8 else 'wrong_archive_identity' if dominant in archive and purity>=.8 else 'mixed_or_unresolved'))
    summary=[]
    for method in cfg['methods']:
        for scene in cfg['spaces']+['all']:
            rr=[r for r in rows if r['method']==method and r['persistent'] and (scene=='all' or r['scene']==scene)]
            summary.append(dict(method=method,scene=scene,persistent=len(rr),correct=sum(r['correct'] for r in rr),
                wrong=sum(r['status']=='wrong_archive_identity' for r in rr),mixed_or_unresolved=sum(r['status']=='mixed_or_unresolved' for r in rr)))
    save(out/'scores.json',{'summary':summary,'rows':rows,'fixed_mask_caps':caps,'motion_accuracy':None,
        'scope':'No current GT masks/classes provided to learned inference; native Rescan masks/classes were anonymized. Historical GT initialization unchanged. Baseline integration, not a new algorithm.'})
    print(json.dumps([r for r in summary if r['scene']=='all'],indent=2))
    print('NATIVE FOUR',json.dumps([r for r in rows if r['scene']=='scene_a' and r['instance'] in [7,12,17,23] and r['method']=='more_classwise_sequential'],indent=2))
    print('CAPS',json.dumps([r for r in caps if r['scene']=='scene_a' and r['instance'] in [7,12,17,23]],indent=2))

if __name__=='__main__':main()
