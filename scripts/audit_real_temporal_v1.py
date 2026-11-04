"""Verify data integrity, anonymization, transform contracts and reported counts."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scripts.score_real_temporal_v1 import read,save,digest
from scripts.prepare_real_temporal_v1 import read_ply

def main():
    out=Path('results/real_temporal_v1');root=Path('data/real_cross_session_v1');checks=[]
    def check(name,value):
        checks.append({'check':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    acq=read(root/'rescan/acquisition.json')
    for r in acq['records']:check('raw '+r['member'],digest(root/'rescan'/r['path'])==r['sha256'])
    prepared=root/'prepared_v2';public=read(prepared/'public.json');ids=read(prepared/'evaluation/identity.json')
    check('no instance keys public JSON','instance' not in (prepared/'public.json').read_text(encoding='utf-8'))
    manifest=read(out/'start_manifest.json')
    for path,h in manifest['input_sha256'].items():check('input '+path,digest(prepared/path)==h)
    for path,h in manifest['inference_sources'].items():check('inference source '+path,digest(path)==h)
    for name,h in read(out/'prediction_seal.json')['files_sha256'].items():check('seal '+name,digest(out/name)==h)
    rows=[json.loads(s) for s in (out/'candidate_fits.jsonl').read_text(encoding='utf-8').splitlines()]
    errors=[]
    for i,r in enumerate(rows):
        T=r['T_current_from_history']
        if T is None:errors.append(r);continue
        T=np.array(T);check('SO3 '+str(i),np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) and abs(np.linalg.det(T[:3,:3])-1)<1e-5)
        c=np.array(r['anchor_history_m']);check('material anchor '+str(i),np.allclose(T[:3,:3]@c+T[:3,3]-c,r['displacement_at_anchor_m']))
    scores=read(out/'scores.json')
    for r in scores['summary']:
        check('denominator '+r['scene']+r['method'],r['correct']+r['persistent_wrong']+r['persistent_unmatched']==r['persistent_eligible'])
    check('all current object predictions',sum(len([c for c in d['1'] if c['eligible']]) for d in public.values())*5==len(scores['rows']))
    # Check no changes to the previous simulation's sealed inference sources.
    old=Path('results/fragment_union_joint_v1/frozen_contract.json')
    for name,h in read(old)['sources'].items():check('previous frozen source '+name,digest(Path(name))==h)
    failures=read(out/'failure_decomposition.json')['rows']
    cases=[r for r in failures if r['method']=='teaser' and r['status']=='wrong_identity' and r['chosen_fit'] and r['chosen_fit']['cost']<.35]
    selected=min(cases,key=lambda r:r['chosen_fit']['cost'])
    scene=selected['scene'];token=selected['current'];target=selected['history_id'];trueid=selected['current_id']
    selectedfits=[r for r in rows if r['method']=='teaser' and r['scene']==scene and r['current']==token and ids[scene]['0'][r['history']] in [trueid,target]]
    save(out/'visible_wrong_identity_case.json',{'case':selected,'fits':selectedfits,'scope':'Actual observed instance, wrong physical ID despite small surface cost; no independent motion error assigned.'})
    check('no claimed motion accuracy',scores['quantitative_motion_accuracy'] is None)
    save(out/'audit.json',{'checks':checks,'checks_passed':len(checks),'fit_failures':len(errors),
        'fit_failure_reasons':errors,'no_asserted_motion_gt':True,'no_dataset_model_training':True})
    print('AUDIT',len(checks),'PASS','fit exceptions/unsupported',len(errors));print(json.dumps(selected,indent=2))

if __name__=='__main__':main()
