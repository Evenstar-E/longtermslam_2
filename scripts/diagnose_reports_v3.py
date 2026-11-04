"""Posthoc point-estimate versus terminal-report errors; evaluator-only."""
import hashlib,json
from pathlib import Path
import numpy as np
from comparative_evidence.evaluation_v1 import score_prediction


def main():
    base=Path('results/comparative_active_v3');counts={};records=[]
    for method in ('candidate','schur','risk'):
        c=dict(good_point_bad_report=0,bad_point_good_report=0,bad_point_bad_report=0,
               good_both=0,unknown_changed=0,wrong_unchanged=0)
        for sid in range(48):
            root=Path('results/comparative_interface_v1/scenes')/f'{sid:04d}'
            truth=json.loads((root/'evaluation'/'truth.json').read_text())
            p=json.loads((base/'predictions'/f'{sid:04d}_{method}.json').read_text())['prediction']
            score=score_prediction(root,p)
            for ownership in score['ownership_records']:
                j=ownership['owner']
                if j<0 or not truth['changed'][j] or not ownership['identity_ok']:continue
                asset=next(a for a in p['assets'] if a['ref_id']==ownership['ref_id'])
                error=float(np.linalg.norm(np.array(asset['point_estimate_delta'])-truth['delta'][j]))
                good=error<=.10
                if asset['label']<0:c['unknown_changed']+=1;continue
                if asset['label']==0:c['wrong_unchanged']+=1;continue
                report_good=ownership['motion_error_m']<=.10
                key='good_both' if good and report_good else ('good_point_bad_report' if good else
                    ('bad_point_good_report' if report_good else 'bad_point_bad_report'))
                c[key]+=1
                if not report_good:records.append(dict(method=method,scene=sid,owner=j,
                    point_error_m=error,report_error_m=ownership['motion_error_m']))
        assert sum(c.values())==72
        counts[method]=c
    result=dict(scope='Posthoc same48 development scenes; no optimization or inference truth access',
        counts=counts,records=records,
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (base/'report_error_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(counts))


if __name__=='__main__':main()
