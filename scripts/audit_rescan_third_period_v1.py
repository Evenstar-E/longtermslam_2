"""Input provenance, intervention scope, immutable predictions and result accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_rescan_third_period_v1 import sha,save,win
from scripts.prepare_rescan_three_period_v1 import verify_runs,run_root
from scripts.prepare_real_temporal_v1 import read_ply


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs-only',action='store_true');args=ap.parse_args()
    checks=0;data=Path('data/rescan_three_period_native_v1');base=Path('results/rescan_third_period_progress_v1')
    for p,h in read(data/'seal.json')['sha256'].items():assert sha(data/p)==h;checks+=1
    # Preserve exactly the earlier executed exporter source after adding the repair-root analysis API.
    script=Path('scripts/prepare_rescan_three_period_v1.py').read_text(encoding='utf-8')
    start=script.index('def run_root(');end=script.index('def verify_runs(',start)
    original=script[:start]+script[end:]
    original=original.replace('def verify_runs(modes=None,progress=False):','def verify_runs(modes=None):').replace('root = run_root(scene,mode,progress)','root = base / scene / mode')
    expected=read(data/'contract.json')['source_script_sha']
    assert hashlib.sha256(original.encode('utf-8')).hexdigest()==expected
    snapshot=data/'preparation_source_executed.py'
    if snapshot.exists():assert sha(snapshot)==expected
    else:snapshot.write_text(original,encoding='utf-8',newline='')
    checks+=1
    cfg=read('configs/rescan_third_period_v1.json')
    for scene in cfg['scenes']:
        root=Path('data/rescan_protected_archive_v1')/scene;record=read(root/'intervention.json')
        prior=Path(cfg['prior_roots'][scene]);a=(prior/'prediction.rsdb').read_text(encoding='utf-8').splitlines();b=(root/'prediction.rsdb').read_text(encoding='utf-8').splitlines()
        assert [x for x in a if not x.startswith('model_folder ')]==[x for x in b if not x.startswith('model_folder ')];checks+=1
        assert sha('scripts/run_rescan_protected_archive_v1.py')==record['source_sha'];checks+=1
        for r in record['models']:assert sha(r['source'])==r['sha256']==sha(r['destination']);checks+=1
    public=read(data/'public.json');point_totals={}
    for scene,periods in public.items():
        prediction=Path(read(run_root(scene,'initial_archive')/'native_groups.json')['prediction']);points=read_ply(prediction)
        mapping=read(data/'evaluation/native_token_mapping_capture2.json')[scene]
        seen=[]
        for r in periods['2']:
            z=np.load(data/r['path']);idx=z['point_indices'];seen.extend(idx.tolist())
            assert len(idx)==r['raw_points'] and bool(len(idx)>=128)==r['eligible']
            assert np.array_equal(z['xyz'],np.column_stack([points[f][idx] for f in ['x','y','z']]))
            assert np.all(points['instance_idx'][idx]==mapping[r['token']]);checks+=3
        assert len(seen)==len(set(seen));checks+=1
        point_totals[scene]={t:sum(r['eligible'] for r in rows) for t,rows in periods.items()}
    save(base/'input_audit.json',{'passed':True,'checks':checks,'eligible_groups':point_totals,'executed_preparation_source_sha':expected,'protected_intervention_preserves_other_database_lines':True,'scope':'No capture2 GT read; checks export, point indexing, initial/protected model provenance.'})
    if args.inputs_only:
        print('INPUT_AUDIT_PASS',checks,point_totals);return
    verify_runs(progress=True)
    verify_runs(['initial_archive','inherited_unfused'],progress=False)
    checks+=4
    residual=Path('data/rescan_three_period_residual_v1')
    for p,h in read(residual/'seal.json')['sha256'].items():assert sha(residual/p)==h;checks+=1
    score=read(base/'scores.json');summaries=score['summaries'];assert len(summaries)==6;checks+=1
    for summary in summaries:
        rows=[r for r in score['rows'] if r['scene']==summary['scene'] and r['mode']==summary['mode']]
        for kind in ['strict','equivalent']:
            selected=[r for r in rows if r[kind+'_initial_persistent']]
            assert summary[kind+'_total']==len(selected) and summary[kind+'_correct']==sum(r[kind+'_correct'] for r in selected);checks+=1
    original=Path('third_party/Rescan_memory_port_v1');port=Path('third_party/Rescan_progress_port_v1');changed=[]
    for p in original.rglob('*'):
        if p.is_file() and 'bin' not in p.relative_to(original).parts:
            if sha(p)!=sha(port/p.relative_to(original)):changed.append(str(p.relative_to(original)))
    assert changed==['lib\\rs\\rs_pointcloud.h'];checks+=1
    activations=[]
    for mode in ['initial_archive','inherited_unfused','inherited_protected']:
        root=run_root('scene_a',mode,True)
        for log in root.glob('*.log'):
            lines=[x for x in log.read_text(encoding='utf-8',errors='replace').splitlines() if 'SAMPLER_PROGRESS_FIX' in x]
            activations.append({'mode':mode,'stage':log.stem,'count':len(lines),'first_examples':lines[:5]})
    engine_comparison={}
    for mode in ['initial_archive','inherited_unfused']:
        old_root=run_root('scene_a',mode,False);new_root=run_root('scene_a',mode,True)
        oldpred=read_ply(Path(read(old_root/'native_groups.json')['prediction']))
        newpred=read_ply(Path(read(new_root/'native_groups.json')['prediction']))
        assert np.array_equal(np.column_stack([oldpred[f] for f in ['x','y','z']]),np.column_stack([newpred[f] for f in ['x','y','z']]))
        engine_comparison[mode]={f:float(np.mean(oldpred[f]==newpred[f])) for f in ['instance_idx','class_idx']}
        checks+=1
    save(base/'audit.json',{'passed':True,'checks':checks,'prediction_seals_and_input_hashes_verified':True,
        'sampler_progress_activations':activations,'changed_port_files':changed,
        'original_vs_progress_initial_archive_point_agreement':engine_comparison,
        'scope':'Descriptive two-space experiment; no motion truth, no original full fused sequential reproduction, no novelty certificate.'})
    assert read(base/'archive_provenance_diagnosis.json')['rows'] and read(base/'proposal_failure_diagnosis.json')['rows']
    save(base/'completion_monitor.json',{'pending':[],'evaluation_started':True,'evaluation_complete':True,'diagnosis_complete':True,
        'first_diagnosis_attempt':'Missing optional scene_c capture1 equivalence metadata; failure log preserved.',
        'completed_diagnosis_log':'diagnosis_completed.log','inference_predictions_modified':False})
    report=Path('research/rescan_third_period_results_v1_2026-10-24.md')
    files=[p for p in base.rglob('*') if p.is_file() and p.name!='final_evidence_index.json']
    sources=list(Path('scripts').glob('*rescan*third*py'))+list(Path('scripts').glob('*rescan*progress*py'))+[Path('scripts/run_rescan_protected_archive_v1.py'),report]
    save(base/'final_evidence_index.json',{'files':{str(p):sha(p) for p in files},'sources':{str(p):sha(p) for p in sources},
        'public_data_seals':{str(p):sha(p) for p in [data/'seal.json',residual/'seal.json']},
        'scene_c_seals':{str(run_root('scene_c',m)/'prediction_seal.json'):sha(run_root('scene_c',m)/'prediction_seal.json') for m in ['initial_archive','inherited_unfused','inherited_protected']}})
    print('FULL_AUDIT_PASS',checks)


if __name__=='__main__':main()
