"""Generate development data and exercise the action contract; no policy ranking."""
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from comparative_evidence.interface import Episode
from comparative_evidence.scene import generate_scene


def main():
    cfg_path = Path('configs/comparative_evidence_interface_v1.json')
    cfg = json.loads(cfg_path.read_text()); out = Path('results/comparative_interface_v1')
    if out.exists():
        raise FileExistsError('Preserve development data; use an isolated version')
    out.mkdir(); (out/'scenes').mkdir()
    (out/'frozen_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    proto = Path('configs/comparative_evidence_v1_protocol.json')
    source_paths = [cfg_path, proto, Path(__file__), Path('comparative_evidence/scene.py'),
                    Path('comparative_evidence/interface.py'), Path('scripts/partial_surface_scene.py')]
    manifest = dict(stage='Interface engineering validation; not algorithm effectiveness',
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        completed_scenes=0, exclusions=[], scenes=[], episode_records=0)
    factors = cfg['factors']; cells = [dict(zip(factors, v)) for v in itertools.product(*factors.values())]
    start = time.perf_counter(); image_rows = []
    with (out/'episodes.jsonl').open('w', encoding='utf-8') as f:
        for sid in range(cfg['development_scenes']):
            condition = cells[sid % len(cells)]; seed = cfg['seed_start']+sid
            root = out/'scenes'/f'{sid:04d}'
            catalog = generate_scene(root, seed, condition, cfg)
            episodes = []
            for h_budget, c_budget in ((2, 1), (4, 2), (8, 4)):
                ep = Episode(root, h_budget, c_budget)
                # Fixed action trace tests data flow and accounting only. It is
                # NOT called a candidate, a coverage policy or a task baseline.
                for j in range(h_budget):
                    ep.act('retrieve_history', j)
                for j in range(1, c_budget+1):
                    ep.act('observe_current', j*2)
                cost = ep.finish()
                row = dict(scene=sid, seed=seed, condition=condition,
                           trace='fixed_interface_exercise_not_policy', cost=cost, state=ep.state())
                f.write(json.dumps(row)+'\n'); f.flush(); episodes.append(cost)
                manifest['episode_records'] += 1
            hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*') if p.is_file()}
            visible = {}; components = {}
            for token in ('history_00', 'current_00'):
                with np.load(root/'evaluation'/f'{token}_owner.npz') as data:
                    counts = np.bincount(data['owner'][data['owner'] >= 0], minlength=8)
                    visible[token] = counts.tolist()
            # Evaluator-only scene census; not given to the API.
            if sid < 4:
                row_images = []
                for token in ('history_00', 'current_00', 'current_06'):
                    with np.load(root/'payloads'/f'{token}.npz') as data:
                        pic = Image.fromarray(data['rgb']).resize((256, 192))
                    strip = Image.new('RGB', (256, 216), 'white'); strip.paste(pic, (0, 24))
                    ImageDraw.Draw(strip).text((4, 5), f'scene {sid} {token}', fill='black')
                    row_images.append(strip)
                image_rows.append(row_images)
            manifest['scenes'].append(dict(scene=sid, seed=seed, condition=condition,
                payloads=24, hashes=hashes, preview_visible_pixel_counts=visible))
            manifest['completed_scenes'] += 1
            (out/'manifest_checkpoint.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            if (sid+1) % 4 == 0:
                print(f'{sid+1}/48 scenes; {time.perf_counter()-start:.1f}s', flush=True)
    sheet = Image.new('RGB', (768, 216*len(image_rows)), 'white')
    for y, row in enumerate(image_rows):
        for x, pic in enumerate(row):
            sheet.paste(pic, (x*256, y*216))
    sheet.save(out/'scene_contact_sheet.png')
    manifest['seconds'] = time.perf_counter()-start
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(scenes=manifest['completed_scenes'], episodes=manifest['episode_records'],
                          seconds=manifest['seconds'])), flush=True)


if __name__ == '__main__':
    main()
