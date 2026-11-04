"""Recompute fixed-evidence aggregates and report/geometry diagnostics."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    base=Path('results/comparative_reports_v4')
    manifest=json.loads((base/'manifest.json').read_text());checks=0
    for collection in ('sources','inputs'):
        for path,digest in manifest[collection].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest;checks+=1
    prior_manifest=json.loads(Path('results/comparative_active_v3/manifest.json').read_text())
    for path,digest in prior_manifest['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest;checks+=1
    data_manifest=json.loads(Path('results/comparative_interface_v1/manifest.json').read_text())
    for scene in data_manifest['scenes']:
        root=Path('results/comparative_interface_v1/scenes')/f"{scene['scene']:04d}"
        for path,digest in scene['hashes'].items():
            assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest;checks+=1
    rows=[json.loads(x) for x in (base/'metrics.jsonl').read_text().splitlines()]
    assert len(rows)==720==manifest['completed'];assert manifest['replay_checks']==144
    totals=defaultdict(lambda:defaultdict(float))
    for row in rows:
        for key in ('changed','qualified_tp','tp','fp','bad_motion_reports','known','task_loss'):
            totals[row['method']+'/'+row['variant']][key]+=row[key]
        assert row['identity_wrong_references']==0;checks+=1
    old=[json.loads(x) for x in Path('results/comparative_active_v3/metrics.jsonl').read_text().splitlines()]
    old={(r['scene'],r['method'],r['variant']):r for r in old}
    for row in rows:
        if row['variant']!='legacy':continue
        prior=old[row['scene'],row['method'],'risk']
        for key in ('qualified_tp','tp','fp','bad_motion_reports','known','task_loss','ownership_records'):
            assert row[key]==prior[key];checks+=1
    for row in old.values():
        if row['variant'] in ('point','two_sigma'):
            for key in ('changed','qualified_tp','tp','fp','bad_motion_reports','known','task_loss'):
                totals[row['method']+'/'+row['variant']][key]+=row[key]
    categories={};geometry={};stability={};details=[]
    for method in ('candidate','schur','risk'):
        cat=defaultdict(lambda:defaultdict(int));geo=defaultdict(int);stable=defaultdict(int)
        for sid in range(48):
            saved=json.loads((base/'predictions'/f'{sid:04d}_{method}.json').read_text())
            root=Path('results/comparative_interface_v1/scenes')/f'{sid:04d}'
            truth=json.loads((root/'evaluation/truth.json').read_text())
            for a,b in zip(saved['predictions']['centres_qmc512']['assets'],saved['predictions']['centres_qmc2048']['assets']):
                assert a['point_estimate_delta']==b['point_estimate_delta'];checks+=1
                stable['asset_slots']+=1
                stable['label_disagreements']+=a['label']!=b['label']
                stable['report_delta_disagreements_gt_1mm']+=int(a['label']==b['label']==1 and np.linalg.norm(np.array(a['delta'])-b['delta'])>.001)
                stable['max_risk_difference']=max(stable['max_risk_difference'],abs(a['decision_risk']-b['decision_risk']))
            for g in saved['geometry']:
                ref=g['ref_id'];owner=g['owner'];point_good=g['selected_error_m']<=.10
                if not point_good:
                    geo['selected_point_bad']+=1
                    geo['some_existing_mode_good']+=g['best_existing_mode_error_m']<=.10
                    geo['all_existing_modes_bad']+=g['best_existing_mode_error_m']>.10
                    details.append(dict(scene=sid,method=method,**g))
                for variant,p in saved['predictions'].items():
                    a=next(a for a in p['assets'] if a['ref_id']==ref)
                    if a['label']<0:key='unknown_changed'
                    elif a['label']==0:key='wrong_unchanged'
                    else:
                        good=np.linalg.norm(np.array(a['delta'])-truth['delta'][owner])<=.10
                        key=('good_point_' if point_good else 'bad_point_')+('good_report' if good else 'bad_report')
                    cat[variant][key]+=1
            for variant in saved['predictions']:assert sum(cat[variant].values())<=72
        for variant in cat:assert sum(cat[variant].values())==72;checks+=1
        categories[method]=dict(cat);geometry[method]=dict(geo);stability[method]=dict(stable)
    result=dict(scope='48 reused DEV scenes, 144 fixed v3 traces; no new acquisition or independent confirmation',
        totals=dict(totals),categories=categories,geometry=geometry,geometry_errors=details,
        quadrature512_vs2048=stability,checks=checks,replay_checks=manifest['replay_checks'])
    (base/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for method in ('candidate','schur','risk'):
        for name in ('legacy','centres_same32','qmc512','centres_qmc512','centres_qmc2048','point','two_sigma'):
            r=totals[method+'/'+name]
            print(method,name,'qualified',int(r['qualified_tp']),'FP',int(r['fp']),
                  'badmotion',int(r['bad_motion_reports']),'known',int(r['known']),'loss',round(r['task_loss'],3))
    print('geometry',json.dumps(geometry));print('stability',json.dumps(stability));print('checks',checks)


if __name__=='__main__':main()
