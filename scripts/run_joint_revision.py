import hashlib,json,time
from pathlib import Path
import numpy as np
from scripts.joint_estimation import evaluate
from scripts.joint_estimation_v2 import infer_decoupled

def main():
    config=Path('configs/joint_estimation_v2.json');revision=json.loads(config.read_text());cfg=json.loads(Path(revision['base_config']).read_text());base=Path('results/joint_estimation_v1');out=Path('results/joint_estimation_v2');out.mkdir(exist_ok=False)
    (out/'frozen_revision.json').write_bytes(config.read_bytes());manifest=json.loads((base/'manifest.json').read_text())
    for p,d in manifest['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d
    seed_rows={r['scene_id']:r for r in [json.loads(l) for l in (base/'metrics.jsonl').read_text().splitlines()]}
    started=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(manifest['scene_count']):
            scene=np.load(base/'scenes'/f'{sid:04d}.npz');obs={k:scene[k] for k in scene.files if not k.startswith('truth_')};truth={k[6:]:scene[k] for k in scene.files if k.startswith('truth_')}
            old=seed_rows[sid];tick=time.perf_counter();pred=infer_decoupled(obs,cfg['algorithm'],old['seed']+400000)
            row=dict(scene_id=sid,seed=old['seed'],condition=old['condition'],method='joint_decoupled',seconds=time.perf_counter()-tick,**evaluate(pred,truth,cfg['evaluation']))
            f.write(json.dumps(row,allow_nan=False)+'\n')
            if (sid+1)%120==0:print(json.dumps(dict(scenes=sid+1,elapsed_s=time.perf_counter()-started)),flush=True)
    paths=[config,Path(revision['base_manifest']),Path('scripts/joint_estimation_v2.py'),Path(__file__),out/'metrics.jsonl']
    (out/'manifest.json').write_text(json.dumps(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},reused_scenes=480,seconds=time.perf_counter()-started),indent=2),encoding='utf-8')

if __name__=='__main__':main()
