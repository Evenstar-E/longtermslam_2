"""Evaluate sealed privileged-mask identity predictions under the old common contract."""
import json
from pathlib import Path
from scripts.run_livingscenes_real_v1 import read,save,sha

def main():
    out=Path('results/livingscenes_real_v1');seal=read(out/'prediction_seal.json')
    for name,h in seal['sha256'].items():assert sha(out/name)==h
    cfg=read(out/'config.json');root=Path(cfg['public_root']);truth=read(root/'evaluation/identity.json');public=read(root/'public.json')
    predictions=read(out/'predictions.json');rows=[]
    for p in predictions:
        scene=p['scene'];cid=truth[scene]['1'][p['current']];hid=None if p['history'] is None else truth[scene]['0'][p['history']]
        allowed=[cid];eq=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        if eq.exists():
            for line in eq.read_text(encoding='utf-8').splitlines():
                left,right=line.split('|')
                if int(left)==cid:allowed=[int(v) for v in right.split()]
        hids={truth[scene]['0'][h['token']] for h in public[scene]['0'] if h['eligible']}
        current=next(c for c in public[scene]['1'] if c['token']==p['current'])
        history=None if p['history'] is None else next(h for h in public[scene]['0'] if h['token']==p['history'])
        rows.append(dict(p,current_id=cid,history_id=hid,persistent=bool(set(allowed)&hids),correct=hid in allowed,strict_correct=hid==cid,
            semantic_class=current['semantic_class'],in_declared_prior_classes=current['semantic_class'] in [5,6,7,14,18],
            current_fps_padding=max(0,1024-current['raw_points']),history_fps_padding=None if history is None else max(0,1024-history['raw_points'])))
    summary=[]
    for method in cfg['methods']:
        for scene in cfg['spaces']+['all']:
            rr=[r for r in rows if r['method']==method and (scene=='all' or r['scene']==scene)];pp=[r for r in rr if r['persistent']]
            summary.append(dict(method=method,scene=scene,persistent=len(pp),correct=sum(r['correct'] for r in pp),strict_correct=sum(r['strict_correct'] for r in pp),
                wrong=sum(r['history'] is not None and not r['correct'] for r in pp),unmatched=sum(r['history'] is None for r in pp),
                new_or_missing_reference=len(rr)-len(pp),correct_rejections=sum(not r['persistent'] and r['history'] is None for r in rr)))
    residual=[r for r in rows if r['scene']=='scene_a' and r['current_id'] in [7,12,17,23]]
    save(out/'scores.json',{'summary':summary,'rows':rows,'previous_native_failure_cases':residual,'motion_accuracy':None,
        'scope':cfg['scope'],'supported_class_subgroup_is_posthoc_description':True})
    print(json.dumps([r for r in summary if r['scene']=='all'],indent=2));print('NATIVE FOUR',json.dumps([r for r in residual if r['method']==cfg['primary']],indent=2))

if __name__=='__main__':main()
