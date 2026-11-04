"""Replay all v3 active observations, isolate terminal approximation errors."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.decisions_v4 import terminal_ablation
from comparative_evidence.evaluation_v3 import evaluate


def main():
    cfgpath=Path('configs/comparative_reports_v4.json');cfg=json.loads(cfgpath.read_text())
    out=Path('results/comparative_reports_v4')
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'predictions').mkdir();(out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    paths=[cfgpath,Path(__file__),Path('comparative_evidence/decisions_v4.py')]
    paths+=list(Path('comparative_evidence').glob('*.py'))
    manifest=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  inputs={},completed=0,replay_checks=0,exclusions=[])
    begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(cfg['scenes']):
            root=Path(cfg['source_data'])/f'{sid:04d}'
            for method in cfg['methods']:
                source=Path(cfg['source_traces'])/f'{sid:04d}_{method}.json'
                manifest['inputs'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
                saved=json.loads(source.read_text());ep=Episode(root,*cfg['budget'])
                h=[ep.act('retrieve_history',0)];q=[ep.preview()]
                for action in saved['prediction']['decisions']:
                    epoch=action['epoch'];i=action['view']
                    obs=ep.act('retrieve_history' if epoch=='history' else 'observe_current',i)
                    (h if epoch=='history' else q).append(obs)
                fit=attach_geometry_information(estimate_modes(h,q,ep.metadata()),h,q,ep.metadata())
                predictions=terminal_ablation(fit)
                assert predictions['legacy']['assets']==saved['prediction']['assets']
                manifest['replay_checks']+=1
                # All inference done. Only this evaluator block accesses truth.
                truth=json.loads((root/'evaluation/truth.json').read_text())
                scored={name:evaluate(root,p) for name,p in predictions.items()}
                geometry=[]
                for row in scored['legacy']['ownership_records']:
                    owner=row['owner'];j=next(i for i,a in enumerate(fit['assets']) if a['ref_id']==row['ref_id'])
                    if owner<0 or not row['identity_ok'] or not truth['changed'][owner]:continue
                    sl=slice(fit['delta_start']+3*j,fit['delta_start']+3*j+3)
                    errors=[float(np.linalg.norm(m['mean'][sl]-truth['delta'][owner])) for m in fit['modes']]
                    geometry.append(dict(owner=owner,ref_id=row['ref_id'],mode_error_m=errors,
                        selected_error_m=float(np.linalg.norm(fit['mean'][sl]-truth['delta'][owner])),
                        best_existing_mode_error_m=min(errors),
                        current_support=fit['assets'][j]['current_support']))
                target=out/'predictions'/source.name
                target.write_text(json.dumps(dict(predictions=predictions,geometry=geometry,
                    mode_audit=fit['mode_audit'],uncertainty_audit=fit['uncertainty_audit'],
                    state=saved['state'],source_sha256=manifest['inputs'][str(source)]),indent=2),encoding='utf-8')
                for name,score in scored.items():
                    f.write(json.dumps(dict(scene=sid,method=method,variant=name,**score))+'\n')
                    manifest['completed']+=1
                f.flush()
            if (sid+1)%4==0:print(f'reports {sid+1}/48 {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
