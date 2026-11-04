"""Ordinary RGB histogram control, declared AFTER geometry results; no novel claim."""
import json,hashlib
from pathlib import Path
import numpy as np
from inspection_decisions.real_temporal_v1 import assign

def main():
    root=Path('data/real_cross_session_v1/prepared_v2');out=Path('results/real_temporal_color_control_v1');out.mkdir(exist_ok=False)
    public=json.loads((root/'public.json').read_text(encoding='utf-8'))
    cfg=json.loads(Path('configs/real_cross_session_v1.json').read_text(encoding='utf-8'))
    base=Path('results/real_temporal_v1');seal=json.loads((base/'prediction_seal.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((base/'candidate_fits.jsonl').read_bytes()).hexdigest()==seal['files_sha256']['candidate_fits.jsonl']
    fits=[json.loads(s) for s in (base/'candidate_fits.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup={(r['scene'],r['current'],r['history'],r['method']):r for r in fits}
    contract={'status':'POSTHOC ordinary engineering control, not independent validation or new algorithm',
        'histogram':'8x8x8 RGB histogram, unit mass, Hellinger distance',
        'fixed_costs':{'rgb_only':'Hellinger','rgb_spatial':'0.5*Hellinger+0.5*min(centroid_distance_m,1)',
            'rgb_surface':'0.5*Hellinger+0.5*best_existing_surface_cost'},
        'no_parameter_search':True,'null_cost':cfg['null_assignment_cost'],'no_new_geometry_fits':True}
    (out/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    rows=[]
    for scene in cfg['spaces']:
        hs=[x for x in public[scene]['0'] if x['eligible']];cs=[x for x in public[scene]['1'] if x['eligible']]
        hist={}
        for capture,items in [('0',hs),('1',cs)]:
            for x in items:
                with np.load(root/x['path']) as z:h=np.histogramdd(z['rgb'],bins=(8,8,8),range=((0,256),)*3)[0].ravel()
                hist[(capture,x['token'])]=h/h.sum()
        costs={m:np.ones((len(cs),len(hs)))*1e6 for m in contract['fixed_costs']}
        for i,c in enumerate(cs):
            for j,h in enumerate(hs):
                if c['semantic_class']!=h['semantic_class']:continue
                key=(scene,c['token'],h['token'])
                rgb=float(np.linalg.norm(np.sqrt(hist[('0',h['token'])])-np.sqrt(hist[('1',c['token'])]))/np.sqrt(2))
                spatial=min(lookup[(*key,'centroid_hungarian')]['cost'],1.)
                surface=min(lookup[(*key,m)]['cost'] for m in ['centroid_icp','fgr','fgr_gicp','teaser'])
                costs['rgb_only'][i,j]=rgb;costs['rgb_spatial'][i,j]=(rgb+spatial)/2;costs['rgb_surface'][i,j]=(rgb+surface)/2
        for method,cost in costs.items():
            for i,j in assign(cost,cfg).items():
                rows.append(dict(scene=scene,method=method,current=cs[i]['token'],history=None if j is None else hs[j]['token'],
                    cost=None if j is None else float(cost[i,j])))
    pred=out/'predictions.json';pred.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'prediction_seal.json').write_text(json.dumps({'sha256':hashlib.sha256(pred.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    # Scoring only begins after predictions are sealed.
    truth=json.loads((root/'evaluation/identity.json').read_text(encoding='utf-8'));scored=[]
    for r in rows:
        s=r['scene'];cid=truth[s]['1'][r['current']];hid=None if r['history'] is None else truth[s]['0'][r['history']]
        p=Path('data/real_cross_session_v1/rescan/metadata')/s/'gt_segmentation'/f'{s}_1.txt';eq={}
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                left,right=line.split('|');eq[int(left)]=[int(v) for v in right.split()]
        allowed=eq.get(cid,[cid]);hids={truth[s]['0'][h['token']] for h in public[s]['0'] if h['eligible']}
        scored.append(dict(r,persistent=bool(set(allowed)&hids),correct=hid in allowed,strict_correct=hid==cid))
    summary=[]
    for method in contract['fixed_costs']:
        for scene in cfg['spaces']+['all']:
            rr=[r for r in scored if r['method']==method and (scene=='all' or r['scene']==scene)];pp=[r for r in rr if r['persistent']]
            summary.append(dict(method=method,scene=scene,persistent=len(pp),correct=sum(r['correct'] for r in pp),
                strict_correct=sum(r['strict_correct'] for r in pp),unmatched=sum(r['history'] is None for r in pp),
                wrong=sum(r['history'] is not None and not r['correct'] for r in pp),new_or_missing_reference=len(rr)-len(pp),
                correct_rejections=sum(not r['persistent'] and r['history'] is None for r in rr)))
    (out/'scores.json').write_text(json.dumps({'summary':summary,'rows':scored,'contract':contract},indent=2),encoding='utf-8')
    print(json.dumps([r for r in summary if r['scene']=='all'],indent=2))

if __name__=='__main__':main()
