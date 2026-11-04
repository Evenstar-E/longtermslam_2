"""Pre-frozen developmental factorial: decision objective vs shared geometry."""
import hashlib
import json
import time
from pathlib import Path

from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v2 import run_selection_v2
from comparative_evidence.evaluation_v1 import score_prediction


def main():
    cfg=json.loads(Path('configs/comparative_selection_v2.json').read_text())
    out=Path('results/comparative_selection_v2')
    if out.exists():raise FileExistsError('Preserve old run; use separate version')
    out.mkdir();(out/'predictions').mkdir()
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    sources=['configs/comparative_selection_v2.json','scripts/run_comparative_selection_v2.py',
        'comparative_evidence/estimation_local_v2.py','comparative_evidence/geometry_modes_v2.py',
        'comparative_evidence/selection_v2.py','comparative_evidence/estimation_v1.py',
        'comparative_evidence/selection_v1.py','comparative_evidence/evaluation_v1.py',
        'comparative_evidence/interface.py','comparative_evidence/policy.py']
    manifest=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        input_manifest_sha256=hashlib.sha256((Path(cfg['source_data'])/'manifest.json').read_bytes()).hexdigest(),
        completed=0,exclusions=[],stage='All48 REUSED development scenes; no confirmation')
    begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as file:
        for sid in range(cfg['scenes']):
            root=Path(cfg['source_data'])/'scenes'/f'{sid:04d}'
            for name,method,backend in cfg['experiments']:
                budget=cfg['full_evidence_budget'] if method=='full_evidence' else cfg['budget']
                start=time.perf_counter()
                result=execute_policy(root,*budget,lambda port:run_selection_v2(port,method,backend))
                scores=score_prediction(root,result['prediction'])
                path=out/'predictions'/f'{sid:04d}_{name}.json'
                path.write_text(json.dumps(result,indent=2),encoding='utf-8')
                row=dict(scene=sid,method=name,selector=method,backend=backend,budget=budget,
                    **scores,cost=result['cost'],timings=result['timings'],seconds=time.perf_counter()-start,
                    prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                file.write(json.dumps(row)+'\n');file.flush();manifest['completed']+=1
                (out/'manifest_checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'{sid+1}/48 scenes {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
