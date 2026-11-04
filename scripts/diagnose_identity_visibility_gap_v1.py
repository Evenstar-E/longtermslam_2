"""POSTHOC: does matched-only readout omit observed free-depth contradictions?

No decisions/reports changed. This is a mechanism audit, not another algorithm
performance result or a tuned combination with the original branch energy.
"""
from pathlib import Path
import numpy as np
from inspection_decisions.projective_visibility_v1 import decode, rasterize
from inspection_decisions.se3_visibility_v1 import meshes_from_history
from scripts.run_profiled_identity_actions_v1 import read, save, digest


def main():
    out = Path('results/profiled_identity_actions_v1_repaired')
    cfg = read(out/'frozen_contract.json')['config']; records = []
    for world in range(cfg['worlds']):
        wd = out/f'{world:04d}'
        for action in range(7):
            prefix = 'initial' if action == 0 else f'endpoint_{action}'
            raw = read(wd/f'{prefix}_observed.json'); result = read(wd/f'{prefix}_backend.json')
            diag = read(wd/('initial_identity_diagnostic.json' if action == 0 else f'endpoint_{action}_identity_diagnostic.json'))
            history = [decode(o) for o in raw['history']]; meshes = meshes_from_history(history, raw['sensor'], cfg)
            current = [decode(o) for o in raw['current']]; branches = []
            for k, branch in enumerate(result['options']):
                G = np.asarray(branch['map_transform_current_from_archive'])
                delta = np.asarray(branch['best']['delta']); total = 0; conflict = 0; omitted_conflict = 0
                for f, o in enumerate(current):
                    T = np.linalg.inv(G)@o['nominal_T']; shape = o['depth'].shape
                    image = np.full(shape, np.inf); owner = np.full(shape, -1, int)
                    for j, mesh in enumerate(meshes):
                        z = rasterize(mesh, T, delta[j], raw['sensor']); front = z < image
                        image[front] = z[front]; owner[front] = j
                    known = np.isfinite(image)
                    # An actually acquired depth beyond an expected historical
                    # patch is a contradiction of this fitted branch. This does
                    # NOT declare unobserved archive geometry to be free space.
                    contradicted = known&(o['depth'] > image+.12)
                    supported = {j for i, j in branch['best']['matches'][f]}
                    total += int(known.sum()); conflict += int(contradicted.sum())
                    omitted_conflict += int((contradicted&~np.isin(owner, list(supported))).sum())
                branches.append(dict(branch=k, ordinary_E=branch['best']['score'],
                    known_pixels=total, contradicted_pixels=conflict,
                    contradiction_fraction=conflict/max(total, 1),
                    contradicted_pixels_on_unmatched_archive=omitted_conflict,
                    physical_identity=diag['retained'][k]))
            selected = result['selected']
            evidence_selected = min(range(len(branches)), key=lambda k: branches[k]['contradiction_fraction'])
            records.append(dict(world=world, action=action, ordinary_selected=selected,
                minimum_contradiction_branch=evidence_selected, branches=branches,
                ordinary_identity_ok=branches[selected]['physical_identity']['all_supported_pairs_correct'],
                minimum_contradiction_identity_ok=branches[evidence_selected]['physical_identity']['all_supported_pairs_correct'],
                correct_branch_retained=diag['correct_branch_retained']))
        print('VISIBILITY GAP world', world, flush=True)
    selected_paths = []
    for world in range(cfg['worlds']):
        decisions = read(out/f'{world:04d}'/'decisions.json')
        a = decisions['profiled_report_risk']
        selected_paths.append(next(r for r in records if r['world'] == world and r['action'] == a))
    def aggregate(rows):
        wrong = [r for r in rows if not r['ordinary_identity_ok']]
        available = [r for r in wrong if r['correct_branch_retained']]
        return dict(endpoints=len(rows), ordinary_selected_identity_ok=sum(r['ordinary_identity_ok'] for r in rows),
            correct_branch_retained=sum(r['correct_branch_retained'] for r in rows),
            wrong_selected_but_correct_retained=len(available),
            minimum_contradiction_identity_ok=sum(r['minimum_contradiction_identity_ok'] for r in rows),
            contradictions_pick_correct_when_ordinary_wrong=sum(r['minimum_contradiction_identity_ok'] for r in available),
            ordinary_selected_omitted_conflict_pixels=sum(r['branches'][r['ordinary_selected']]['contradicted_pixels_on_unmatched_archive'] for r in rows))
    save(out/'visibility_gap_diagnostic.json', dict(status='POSTHOC mechanism diagnostic only; no task reports or algorithm scores modified',
        source_sha256=digest(__file__), all_endpoints=aggregate(records),
        candidate_selected_paths=aggregate(selected_paths), rows=records,
        limits=['Contradiction-only argmin is not proposed or validated as a full estimator',
                'True local movement, patch errors and calibration may also create contradictions',
                'Physical identity audit checks supported assigned pairs, not task completeness or displacement accuracy']))
    print('VISIBILITY GAP', aggregate(records), 'CANDIDATE', aggregate(selected_paths), flush=True)


if __name__ == '__main__': main()
