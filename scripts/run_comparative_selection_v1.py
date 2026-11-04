"""Frozen-source development comparison, truth read only AFTER each policy."""
import hashlib
import json
import time
from pathlib import Path

from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v1 import run_selection
from comparative_evidence.evaluation_v1 import score_prediction


def main():
    cfg = json.loads(Path('configs/comparative_selection_v1.json').read_text())
    out = Path('results/comparative_selection_v1')
    if out.exists():
        raise FileExistsError('Preserve existing run; use separate version')
    out.mkdir(); (out/'predictions').mkdir()
    (out/'frozen_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    sources = [Path(__file__), Path('configs/comparative_selection_v1.json'),
        *[Path('comparative_evidence')/s for s in ('estimation_v1.py', 'selection_v1.py',
          'evaluation_v1.py', 'interface.py', 'policy.py', 'scene.py')]]
    manifest = dict(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        source_data_manifest_sha256=hashlib.sha256((Path(cfg['source_data'])/'manifest.json').read_bytes()).hexdigest(),
        completed=0, exclusions=[], failures=[])
    start = time.perf_counter()
    with (out/'metrics.jsonl').open('w', encoding='utf-8') as f:
        for sid in range(cfg['scenes']):
            root = Path(cfg['source_data'])/'scenes'/f'{sid:04d}'
            for method in cfg['methods']:
                budget = cfg['full_evidence_budget'] if method == 'full_evidence' else cfg['budget']
                begin = time.perf_counter()
                result = execute_policy(root, *budget, lambda port: run_selection(port, method))
                # Only here may truth/owner files be opened. No truth in policy args.
                scores = score_prediction(root, result['prediction'])
                saved = out/'predictions'/f'{sid:04d}_{method}.json'
                saved.write_text(json.dumps(result, indent=2), encoding='utf-8')
                row = dict(scene=sid, method=method, budget=budget, **scores,
                    cost=result['cost'], timings=result['timings'], seconds=time.perf_counter()-begin,
                    prediction_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
                f.write(json.dumps(row)+'\n'); f.flush(); manifest['completed'] += 1
                (out/'manifest_checkpoint.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(f'{sid+1}/48 scenes, {time.perf_counter()-start:.1f}s', flush=True)
    manifest['seconds'] = time.perf_counter()-start
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
