"""Frozen action traces isolate uncertainty and terminal rule from acquisition."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information,perturb_icp_delta_covariance
from comparative_evidence.decisions_v3 import terminal_prediction
from comparative_evidence.evaluation_v3 import evaluate


def main():
    cfg=json.loads(Path('configs/comparative_selection_v3.json').read_text())
    out=Path('results/comparative_fixed_v3')
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'predictions').mkdir()
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    paths=['configs/comparative_selection_v3.json','scripts/run_comparative_fixed_v3.py',
        'comparative_evidence/uncertainty_v3.py','comparative_evidence/decisions_v3.py',
        'comparative_evidence/geometry_modes_v2.py','comparative_evidence/evaluation_v3.py',
        'comparative_evidence/evaluation_v1.py','comparative_evidence/interface.py']
    manifest=dict(sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        completed=0,base_traces=0,exclusions=[]);begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(cfg['scenes']):
            root=Path(cfg['source_data'])/'scenes'/f'{sid:04d}'
            for method in cfg['fixed_trace_methods']:
                source=Path(cfg['source_traces'])/'predictions'/f'{sid:04d}_{method}.json'
                saved=json.loads(source.read_text());budget=cfg['full_budget'] if method=='modes_full' else cfg['budget']
                ep=Episode(root,*budget)
                h=[ep.act('retrieve_history',i) for i in saved['state']['history_read']]
                q=[ep.preview()]+[ep.act('observe_current',i) for i in saved['state']['current_observed'] if i]
                fit=estimate_modes(h,q,ep.metadata())
                # Recomputed point estimates must match frozen v2 exactly.
                for a,b in zip(fit['assets'],saved['prediction']['assets']):
                    np.testing.assert_array_equal(a['delta'],b['delta'])
                gn=attach_geometry_information(fit,h,q,ep.metadata())
                perturb,audit=perturb_icp_delta_covariance(gn,h,q,ep.metadata(),cfg['perturbation_draws'])
                predictions={
                    'inherited_2sigma':terminal_prediction(fit,'two_sigma'),
                    'inherited_risk':terminal_prediction(fit,'risk'),
                    'gn_2sigma':terminal_prediction(gn,'two_sigma'),
                    'gn_risk':terminal_prediction(gn,'risk'),
                    'point':terminal_prediction(gn,'point'),
                    'perturb_2sigma':terminal_prediction(gn,'two_sigma',perturb),
                    'perturb_risk':terminal_prediction(gn,'risk',perturb)}
                # Evaluation happens after all inference; no truth in model args.
                target=out/'predictions'/f'{sid:04d}_{method}.json'
                target.write_text(json.dumps(dict(predictions=predictions,perturbation=audit,
                    uncertainty=gn['uncertainty_audit'],source_prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
                for variant,p in predictions.items():
                    row=dict(scene=sid,method=method,variant=variant,**evaluate(root,p),
                        prediction_file=target.name,prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                    f.write(json.dumps(row)+'\n');manifest['completed']+=1
                f.flush();manifest['base_traces']+=1
            (out/'manifest_checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if (sid+1)%4==0:print(f'fixed {sid+1}/48 {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
