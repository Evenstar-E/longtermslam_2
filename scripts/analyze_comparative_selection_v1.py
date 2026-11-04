"""Paired exploratory development analysis plus ownership/decision audit."""
import hashlib
import json
from pathlib import Path

import numpy as np

from comparative_evidence.evaluation_v1 import score_prediction
from comparative_evidence.policy import execute_policy
from comparative_evidence.selection_v1 import run_selection


def main():
    out = Path('results/comparative_selection_v1')
    cfg = json.loads((out/'frozen_config.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    rows = [json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
    checks = 0
    for p, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, p
        checks += 1
    assert len(rows) == 336 and len({(r['scene'], r['method']) for r in rows}) == 336
    checks += 1
    summaries = {}; diagnostics = {}; point_decision = {}
    for method in cfg['methods']:
        rs = sorted([r for r in rows if r['method'] == method], key=lambda r: r['scene'])
        summary = {k: sum(r[k] for r in rs) for k in ('changed', 'unchanged', 'tp', 'qualified_tp', 'fp',
            'unchanged_false_alarm_assets', 'known', 'wrong_known', 'identity_wrong_references', 'assets')}
        summary['completion'] = summary['qualified_tp']/summary['changed']
        summary['recall'] = summary['tp']/summary['changed']
        summary['decision_coverage'] = summary['known']/summary['assets']
        for k in ('travel_m', 'payload_bytes', 'modeled_acquisition_s'):
            summary['mean_'+k] = float(np.mean([r['cost'][k] for r in rs]))
        summary['median_policy_compute_s'] = float(np.median([r['timings']['policy_wall_excluding_api_s'] for r in rs]))
        diag = dict(missing_reference=0, identity_wrong=0, qualified=0,
                    reported_bad_motion=0, unknown_good_motion=0, unknown_bad_motion=0,
                    unchanged_good_motion=0, unchanged_bad_motion=0)
        point = dict(tp=0, qualified_tp=0, fp=0, changed=summary['changed'])
        owned_errors = []
        for r in rs:
            sid = r['scene']; path = out/'predictions'/f'{sid:04d}_{method}.json'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == r['prediction_sha256']
            saved = json.loads(path.read_text()); prediction = saved['prediction']
            root = Path(cfg['source_data'])/'scenes'/f'{sid:04d}'
            rescored = score_prediction(root, prediction)
            assert all(rescored[k] == r[k] for k in rescored)
            checks += 2
            truth = json.loads((root/'evaluation'/'truth.json').read_text())
            for j, changed in enumerate(truth['changed']):
                if not changed:
                    continue
                owned = [a for a in r['ownership_records'] if a['owner'] == j]
                if len(owned) != 1:
                    diag['missing_reference'] += 1
                    continue
                a = owned[0]
                if not a['identity_ok']:
                    diag['identity_wrong'] += 1
                    continue
                good = a['motion_error_m'] <= .10
                owned_errors.append(a['motion_error_m'])
                if a['label'] == 1:
                    diag['qualified' if good else 'reported_bad_motion'] += 1
                elif a['label'] == -1:
                    diag['unknown_good_motion' if good else 'unknown_bad_motion'] += 1
                else:
                    diag['unchanged_good_motion' if good else 'unchanged_bad_motion'] += 1
            # Same estimates, common direct-threshold decision audit ONLY.
            # Declared posthoc; does not replace frozen primary decisions.
            alt = json.loads(json.dumps(prediction))
            for asset in alt['assets']:
                asset['label'] = int(np.max(np.abs(asset['delta'])) > .12)
            ps = score_prediction(root, alt)
            for k in ('tp', 'qualified_tp', 'fp'):
                point[k] += ps[k]
            budget = cfg['full_evidence_budget'] if method == 'full_evidence' else cfg['budget']
            assert (saved['cost']['history_views'], saved['cost']['additional_current_views']) == tuple(budget)
            assert saved['cost']['returned'] and saved['cost']['closed']
            assert [x['epoch'] for x in prediction['decisions']].count('history') == budget[0]-1
            assert [x['epoch'] for x in prediction['decisions']].count('current') == budget[1]
            events = saved['state']['events']
            for event in events:
                if event['payload_bytes']:
                    token = f"{event['epoch']}_{event['view_id']:02d}"
                    raw = (root/'payloads'/f'{token}.npz').read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == event['payload_sha256']
                    assert len(raw) == event['payload_bytes']
                    checks += 2
            checks += 4
        assert sum(diag.values()) == summary['changed']
        checks += 1
        summary['all_owned_changed_motion_median_m'] = float(np.median(owned_errors))
        summary['all_owned_changed_motion_p90_m'] = float(np.quantile(owned_errors, .9))
        summaries[method] = summary; diagnostics[method] = diag; point_decision[method] = point
    rng = np.random.default_rng(24681); comparisons = {}
    #16cells*3developmentseeds: stratify complete factorial cells, pair scenes.
    ids = np.arange(48).reshape(3, 16).T
    indices = np.stack([np.concatenate([cell[rng.integers(0, 3, 3)] for cell in ids]) for _ in range(3000)])
    candidate = sorted([r for r in rows if r['method'] == 'candidate_separation'], key=lambda r:r['scene'])
    cand_tp = np.array([r['qualified_tp'] for r in candidate]); changed = np.array([r['changed'] for r in candidate])
    for method in ('coverage', 'entropy_mc', 'schur_information', 'task_risk_mc', 'nearest_replay'):
        baseline = sorted([r for r in rows if r['method'] == method], key=lambda r:r['scene'])
        difference = cand_tp-np.array([r['qualified_tp'] for r in baseline])
        draws = difference[indices].sum(axis=1)/changed[indices].sum(axis=1)
        comparisons[method] = dict(completion_difference=float(difference.sum()/changed.sum()),
                                  exploratory_paired_95_ci=np.quantile(draws, [.025, .975]).tolist())
    # Replays prespecified scene1 plus coherent/repeated/limited scene15;
    # not selected for method advantage. Truth accessed only after policy.
    for sid, method in ((1, 'candidate_separation'), (15, 'task_risk_mc')):
        root = Path(cfg['source_data'])/'scenes'/f'{sid:04d}'
        replay = execute_policy(root, *cfg['budget'], lambda port: run_selection(port, method))
        old = json.loads((out/'predictions'/f'{sid:04d}_{method}.json').read_text())
        assert replay['prediction'] == old['prediction']
        checks += 1
    result = dict(stage='Exploratory development; not independent confirmation', checks=checks,
        summaries=summaries, changed_slot_failure_decomposition=diagnostics,
        posthoc_common_direct_threshold_audit=point_decision,
        candidate_minus_controls=comparisons, bootstrap_repetitions=3000,
        multiplicity='five exploratory comparisons; no familywise significance or confirmation claim',
        source_hashes_and_loaded_payloads_preserved=True, exclusions=[])
    (out/'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
