"""Three matched active policies share geometry-consistent GN and task reports."""
import hashlib,json,time
from pathlib import Path
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v3 import run_selection_v3
from comparative_evidence.evaluation_v3 import evaluate


def main():
    cfg=json.loads(Path('configs/comparative_selection_v3.json').read_text())
    out=Path('results/comparative_active_v3')
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'predictions').mkdir()
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    paths=['configs/comparative_selection_v3.json','scripts/run_comparative_active_v3.py',
        'comparative_evidence/uncertainty_v3.py','comparative_evidence/decisions_v3.py',
        'comparative_evidence/selection_v3.py','comparative_evidence/geometry_modes_v2.py',
        'comparative_evidence/selection_v2.py','comparative_evidence/evaluation_v3.py',
        'comparative_evidence/evaluation_v1.py','comparative_evidence/interface.py','comparative_evidence/policy.py']
    manifest=dict(sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
                  completed=0,exclusions=[]);begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(cfg['scenes']):
            root=Path(cfg['source_data'])/'scenes'/f'{sid:04d}'
            for name,selector in cfg['active_methods']:
                result=execute_policy(root,*cfg['budget'],lambda port:run_selection_v3(port,selector))
                p=result['prediction'];variants={'risk':p,**p['terminal_alternatives']}
                target=out/'predictions'/f'{sid:04d}_{name}.json'
                target.write_text(json.dumps(result,indent=2),encoding='utf-8')
                for variant,prediction in variants.items():
                    row=dict(scene=sid,method=name,variant=variant,**evaluate(root,prediction),
                        cost=result['cost'],timings=result['timings'],
                        prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                    f.write(json.dumps(row)+'\n');manifest['completed']+=1
                f.flush()
            (out/'manifest_checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if (sid+1)%4==0:print(f'active {sid+1}/48 {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
