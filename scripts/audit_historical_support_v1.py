"""Posthoc oracle diagnostic on frozen v4b scenes; NOT a deployable algorithm.

Only faces marked tight by the existing observation extractor are fixed in the
observed-face oracle. Missing support means unrecognized by this extractor,
not proof the raw historical images never contained the surface.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.diagnose_map_evidence_v4b import labels
from scripts.map_evidence_v4b import Model


def supported_faces(records):
    return {(r['asset'], e['axis'], e['side'])
            for r in records for e in r['endpoints'] if e['tight']}


def fix_supported_reference(model, inp, truth):
    """Oracle values, selected ONLY using the frozen input's support marks."""
    boxes = np.asarray(truth['reference_boxes'])
    for asset, axis, side in supported_faces(inp['history']):
        index = 6 * asset + 2 * axis + side
        model.lower[index] = model.upper[index] = boxes[asset, side, axis]


def score(lab, inp, truth):
    lab = np.asarray(lab)
    changed = np.asarray(truth['changed'])
    correct = np.asarray(inp['assignment']) == truth['assignment']
    return dict(visible=len(lab), changed=int(changed.sum()),
                known=int((lab >= 0).sum()),
                tp=int(((lab == 1) & changed & correct).sum()),
                fp=int(((lab == 1) & (~changed | ~correct)).sum()),
                wrong_known=int(((lab >= 0) & (((lab == 1) != changed) | ~correct)).sum()))


def main():
    source = Path('results/map_evidence_v4b')
    out = Path('results/historical_support_audit_v1')
    out.mkdir(exist_ok=True)
    target = out / 'audit.json'
    if target.exists() or (out / 'checkpoint.json').exists():
        raise RuntimeError('Preserve existing diagnostic; use another output version')
    cfg = json.loads((source / 'frozen_config.json').read_text())
    primary = {r['scene']: r for r in map(json.loads, (source / 'metrics.jsonl').read_text().splitlines())
               if r['method'] == 'joint_feasibility' and r['scene'] >= 64}
    oracle = {(r['scene'], r['method']): r['labels'] for r in
              json.loads((source / 'oracle_diagnostics.json').read_text())['records']}
    result = dict(scope='Posthoc reused fresh scenes, oracle-only support diagnosis; not new method validation',
                  limits=['Tight support is extractor-specific, not a raw-image visibility oracle',
                          'True supported face values remain privileged information',
                          'Remaining full-reference gain is not a causal irrecoverable-history percentage',
                          'Exact truth constraints may conflict with the observation model; all scenes retained'],
                  source_sha256={}, records=[], assets=[], summaries={})
    paths = [Path(__file__), source / 'frozen_config.json', source / 'metrics.jsonl',
             source / 'oracle_diagnostics.json', Path('scripts/map_evidence_v4b.py'),
             Path('scripts/diagnose_map_evidence_v4b.py')]
    for sid in range(64, 128):
        ip = source / 'cases' / f'{sid:04d}_input.json'
        tp = source / 'cases' / f'{sid:04d}_truth.json'
        paths.extend([ip, tp])
        inp, truth = json.loads(ip.read_text()), json.loads(tp.read_text())
        history, current = supported_faces(inp['history']), supported_faces(inp['current'])
        for i, asset in enumerate(inp['assignment']):
            shared = history & current
            result['assets'].append(dict(scene=sid, query=i, asset=asset,
                history_faces=sum(j == asset for j, _, _ in history),
                shared_face_axes=len({a for j, a, _ in shared if j == asset}),
                baseline_label=primary[sid]['labels'][i], changed=truth['changed'][i]))
        labs = {'baseline': primary[sid]['labels'],
                **{m: oracle[sid, m] for m in ('oracle_calibration', 'oracle_reference', 'oracle_both')}}
        statuses = {}
        for method in ('oracle_supported_reference', 'oracle_supported_and_calibration'):
            model = Model(inp, cfg)
            fix_supported_reference(model, inp, truth)
            if method.endswith('and_calibration'):
                u = np.r_[np.asarray(truth['realized_history_nuisance']).reshape(-1),
                          truth['realized_query_nuisance']]
                model.lower[model.u0:model.z0] = u
                model.upper[model.u0:model.z0] = u
            statuses[method] = int(model.solve().status)
            labs[method] = labels(model, cfg)
        for method, lab in labs.items():
            result['records'].append(dict(scene=sid, condition=inp['condition'], method=method,
                labels=lab, base_status=statuses.get(method), **score(lab, inp, truth)))
        (out / 'checkpoint.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        if (sid + 1) % 8 == 0:
            print(f'{sid - 63}/64 scenes', flush=True)
    for method in labs:
        records = [r for r in result['records'] if r['method'] == method]
        s = {k: sum(r[k] for r in records) for k in ('visible', 'changed', 'known', 'tp', 'fp', 'wrong_known')}
        s.update(coverage=s['known'] / s['visible'], recall=s['tp'] / s['changed'])
        s['nonoptimal_base_scenes'] = [r['scene'] for r in records if r['base_status'] not in (None, 0)]
        result['summaries'][method] = s
    assets = result['assets']
    result['support_summary'] = {
        'history_face_counts': {str(k): sum(a['history_faces'] == k for a in assets) for k in range(7)},
        'shared_axis_counts': {str(k): sum(a['shared_face_axes'] == k for a in assets) for k in range(4)},
        'unknown_by_shared_axes': {str(k): sum(a['baseline_label'] == -1 and a['shared_face_axes'] == k
                                             for a in assets) for k in range(4)}}
    for p in paths:
        result['source_sha256'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(summaries=result['summaries'], support=result['support_summary']), indent=2))


if __name__ == '__main__':
    main()
