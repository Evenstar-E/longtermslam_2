"""Score sealed real-data predictions; diagnostics keep truth out of estimators."""
import hashlib,json,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from inspection_decisions.real_temporal_v1 import prepare,fit_stats,moved

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=Path('results/real_temporal_v1');root=Path('data/real_cross_session_v1/prepared_v2')
    seal=read(out/'prediction_seal.json')
    for name,h in seal['files_sha256'].items():assert digest(out/name)==h
    truth=read(root/'evaluation/identity.json');public=read(root/'public.json');cfg=read(out/'config.json')
    predictions=[json.loads(s) for s in (out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    candidates=[json.loads(s) for s in (out/'candidate_fits.jsonl').read_text(encoding='utf-8').splitlines()]
    index={(r['scene'],r['current'],r['history'],r['method']):r for r in candidates}
    equivalences={}
    for scene in cfg['spaces']:
        p=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        eq={}
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                left,right=line.split('|');eq[int(left)]=[int(x) for x in right.split()]
        equivalences[scene]=eq
    scores=[]
    for r in predictions:
        scene=r['scene'];cid=truth[scene]['1'][r['current']]
        hid=None if r['history'] is None else truth[scene]['0'][r['history']]
        hids={truth[scene]['0'][h['token']] for h in public[scene]['0'] if h['eligible']}
        allowed=equivalences[scene].get(cid,[cid]);persistent=any(x in hids for x in allowed)
        scores.append(dict(scene=scene,method=r['method'],current=r['current'],current_id=cid,history_id=hid,
            persistent_eligible=persistent,strict_correct=hid==cid,equivalence_correct=hid in allowed,
            allowed_ids=allowed,status='unmatched' if hid is None else 'correct' if hid in allowed else 'wrong_identity',
            correct_rejection=(not persistent and hid is None)))
    summary=[]
    for method in cfg['methods']:
        for scene in cfg['spaces']+['all']:
            rows=[r for r in scores if r['method']==method and (scene=='all' or r['scene']==scene)]
            p=[r for r in rows if r['persistent_eligible']]
            summary.append(dict(scene=scene,method=method,current_eligible=len(rows),persistent_eligible=len(p),
                correct=sum(r['equivalence_correct'] for r in p),strict_correct=sum(r['strict_correct'] for r in p),
                persistent_unmatched=sum(r['history_id'] is None for r in p),
                persistent_wrong=sum(r['history_id'] is not None and not r['equivalence_correct'] for r in p),
                new_or_missing_reference=len(rows)-len(p),correct_rejections=sum(r['correct_rejection'] for r in rows)))
    # Correct physical pair diagnostics use stable IDs, NOT equivalence swaps as metric-motion truth.
    diagnostics=[];allmodes=[]
    for scene in cfg['spaces']:
        for c in public[scene]['1']:
            if not c['eligible']:continue
            cid=truth[scene]['1'][c['token']]
            hh=[h for h in public[scene]['0'] if h['eligible'] and truth[scene]['0'][h['token']]==cid]
            if not hh:continue
            h=hh[0]
            modes=[index[(scene,c['token'],h['token'],m)] for m in cfg['methods'] if (scene,c['token'],h['token'],m) in index and index[(scene,c['token'],h['token'],m)]['T_current_from_history'] is not None]
            fitted=[r for r in modes if r['method']!='centroid_hungarian']
            good=[r for r in fitted if r['stats']['support_a']>=.5 and r['stats']['support_b']>=.5 and r['stats']['cost']<.35]
            spans=[(float(np.linalg.norm(np.array(a['displacement_at_anchor_m'])-b['displacement_at_anchor_m'])),a,b) for a,b in itertools.combinations(good,2)]
            maxspan,aa,bb=max(spans,key=lambda t:t[0]) if spans else (None,None,None)
            with np.load(root/h['path']) as z:a,fa=prepare(z['xyz'],cfg)
            with np.load(root/c['path']) as z:b,fb=prepare(z['xyz'],cfg)
            eig=np.linalg.eigvalsh(np.cov(np.asarray(a.points).T));planarity=float(eig[0]/max(eig[-1],1e-12))
            anchor=np.asarray(a.points).mean(0)
            landmarks=anchor[None]+np.array([[0,0,0],[.25,0,0],[0,.25,0],[0,0,.25]])
            landmark_span=max([float(np.linalg.norm(moved(landmarks,np.array(u['T_current_from_history']))-moved(landmarks,np.array(v['T_current_from_history'])),axis=1).max()) for u,v in itertools.combinations(good,2)] or [0.])
            diagnostics.append(dict(scene=scene,instance=cid,current=c['token'],history=h['token'],
                raw_history_points=h['raw_points'],raw_current_points=c['raw_points'],
                class_idx=c['semantic_class'],good_fitting_modes=len(good),
                anchor_displacement_span_m=maxspan,landmark_span_m=landmark_span,
                spans_10cm=bool(maxspan is not None and maxspan>.1),
                top_span_methods=None if aa is None else [aa['method'],bb['method']],
                observed_shape_eigen_ratio=planarity,modes=[dict(method=r['method'],displacement=r['displacement_at_anchor_m'],stats=r['stats']) for r in modes],
                caveat='Multiple plausible surface fits are a model/solver diagnostic, not an independent physical motion label.'))
            allmodes.extend(modes)
    summarydoc={'summary':summary,'rows':scores,'diagnostics':diagnostics,
        'diagnostic_totals':{'strict_persistent_pairs':len(diagnostics),'multiple_good_modes':sum(r['good_fitting_modes']>=2 for r in diagnostics),
            'anchor_disagreement_over_10cm':sum(r['spans_10cm'] for r in diagnostics),
            'landmark_disagreement_over_10cm':sum(r['good_fitting_modes']>=2 and r['landmark_span_m']>.1 for r in diagnostics)},
        'quantitative_motion_accuracy':None,'independent_spaces':3,'scope':cfg['scope'],'equivalence_rule':'row-level allowed ID accuracy, NOT original pointwise IoU metric',
        'posthoc_diagnostic_thresholds':{'support_each':.5,'cost_max':.35,'span_m':.1},
        'prediction_seal_sha256':digest(out/'prediction_seal.json')}
    save(out/'scores.json',summarydoc)
    print(json.dumps({'pooled':[r for r in summary if r['scene']=='all'],'diagnostic_totals':summarydoc['diagnostic_totals']},indent=2))

if __name__=='__main__':main()
