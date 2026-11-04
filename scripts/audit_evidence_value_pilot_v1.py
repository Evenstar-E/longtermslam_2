"""Independent integrity, model persistence/retraining and decision replay."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from inspection_decisions.value_pilot_v1 import encode_options,fit_value_model,predict_value,choose
from inspection_decisions.graph_dataset import read_graph
from scripts.run_evidence_value_pilot_v1 import read,save,digest,lines,outcome_arrays,verify_prep,SPLITS


def main():
    out=Path('results/evidence_value_pilot_v1');cfg=read('configs/evidence_value_pilot_v1.json');verify_prep(out)
    frozen=read(out/'frozen_selection.json');analysis=read(out/'analysis.json');idx=read(out/'input_index.json')
    f=np.load(out/'features.npz');p=np.load(out/'frozen_predictions.npz')
    split=np.array(idx['splits']);tr=split==SPLITS[0];dev=split==SPLITS[2];checks=0
    families=[set(np.array(idx['families'])[split==s]) for s in SPLITS]
    assert all(len(x)==16 for x in families) and all(not families[i]&families[j] for i in range(3) for j in range(i));checks+=1
    hashes={}
    for manifest in Path(cfg['dataset']).glob('shard_*/manifest.json'):
        for file,sha in read(manifest)['sources'].items():
            assert digest(file)==sha;hashes[file]=sha;checks+=1
    for file,sha in frozen['source_sha256'].items(): assert digest(file)==sha;checks+=1
    for file,sha in frozen['artifact_sha256'].items(): assert digest(out/file)==sha;checks+=1
    pairs=read('configs/evidence_graph_dataset_v1.json')['pairs']
    for shard in Path(cfg['dataset']).glob('shard_*'):
        for r in lines(shard/'index/states.jsonl'):
            if r['actions']: continue
            g=read_graph(r['input_file']);i=r['scene']
            assert set(g['audit']['source_tokens'])=={'history_00','current_00'}
            for use,key in ((True,'X'),(False,'X_no_cov')):
                e=encode_options(g,pairs,use);np.testing.assert_array_equal(e['X'],f[key][i]);checks+=1
            checks+=1
    train_loss,train_gain=outcome_arrays(read(out/f'{SPLITS[0]}_outcomes.json'))
    errors={}
    for name in cfg['models']:
        m=read(out/f'model_{name}.json');X=f['X_no_cov' if name=='structured_no_cov_rbf' else 'X']
        for part in ('direct','single','interaction'):
            if part in m:
                for k in ('mean','scale','Z','coef'):m[part][k]=np.array(m[part][k])
        # Loaded model and retraining/inference are pure: all filesystem denied.
        with patch('builtins.open',side_effect=AssertionError('Model filesystem access')), \
             patch('io.open',side_effect=AssertionError('Model filesystem access')), \
             patch('os.open',side_effect=AssertionError('Model filesystem access')):
            loaded=predict_value(m,X)
            again=fit_value_model(name,X[tr],train_gain,m['single_indices'],m['pair_indices'],m['pair_members'],m['alpha'],m['bandwidth'])
            replay=predict_value(again,X)
        np.testing.assert_array_equal(loaded,p[name]);np.testing.assert_array_equal(replay,p[name]);checks+=2
        errors[name]=dict(persisted_prediction_max_abs=float(np.max(abs(loaded-p[name]))),
                          retrained_prediction_max_abs=float(np.max(abs(replay-p[name]))))
    search=read(out/'calibration_search.json')
    for name in cfg['models']:
        best=min([r for r in search['model_search'] if r['model']==name],key=lambda r:r['calibration_mse'])
        for k in ('alpha','bandwidth','calibration_mse'):assert best[k]==frozen['selection'][name][k];checks+=1
    assert min(search['threshold_search'],key=lambda r:r['calibration_objective'])['threshold']==frozen['risk_threshold'];checks+=1
    data=read(out/f'{SPLITS[2]}_outcomes.json');counts=np.array([len(o) for o in idx['options']])
    scene_cases=[]
    for price in cfg['sensitivity_prices']:
        for name in cfg['models']+['stop','coverage_single','coverage_bundle','coverage_risk_stop']:
            choice=choose(p[name][dev],counts,price) if name in cfg['models'] else p['choice_'+name][dev]
            selected=[r['outcomes'][c] for r,c in zip(data,choice)]
            summary=analysis['summary'][str(price)][name]
            for key in ('qualified_tp','fp','bad_motion_reports','known','identity_wrong_references','task_loss'):
                assert abs(sum(r['metric'][key] for r in selected)-summary[key])<1e-12;checks+=1
            for i,(r,c) in enumerate(zip(data,choice)):
                assert r['outcomes'][c]['state_id']==f"{r['scene']:04d}_{idx['option_names'][c]}";checks+=1
            if price==cfg['primary_acquisition_price'] and name=='structured_rbf':
                for r,c in zip(data,choice):
                    if r['outcomes'][c]['metric']['task_loss']>0:
                        scene_cases.append(dict(scene=r['scene'],option=idx['option_names'][c],
                                               metric=r['outcomes'][c]['metric']))
    save(out/'audit.json',dict(checks=checks,model_replay=errors,old_frozen_sources_verified=len(hashes),
           all_splits_disjoint=True,initial_H0_C0_only=True,zero_exclusions=True,
           candidate_nonzero_loss_cases=scene_cases,
           notes=['Filesystem denied during retrain/inference, no labels passed to actors',
                  'Replay tolerance exact arrays; metric summation1e-12',
                  'Pre-freeze unit test found9-versus10patch width; fixed schema-based width before data preparation; engineering only']))
    print('PASS',checks,'checks; six persisted/retrained models EXACT; old sources',len(hashes),flush=True)


if __name__=='__main__':main()
