"""Final integrity supplement; no new experimental samples or algorithm updates."""
import csv
from pathlib import Path
from scripts.run_profiled_identity_actions_v1 import read, save, digest
from tests.test_fragment_union_degeneracy_v1 import counterexample
import numpy as np


def main():
    out = Path('results/fragment_union_joint_v1'); frozen = read(out/'frozen_contract.json'); checks = 0
    for path, sha in frozen['sources'].items():
        assert digest(path) == sha, path; checks += 1
    audit = read(out/'audit.json')
    assert digest('scripts/audit_fragment_union_joint_v1.py') == audit['source_sha256']; checks += 1
    for path, sha in audit['artifacts'].items():
        assert digest(out/path) == sha, path; checks += 1
    prior = Path('results/profiled_identity_actions_v1_repaired')
    old_seal = read(prior/'final_integrity.json')
    for path, sha in old_seal['artifacts'].items():
        assert digest(prior/path) == sha, path; checks += 1
    for path, sha in read('results/se3_visibility_v1/frozen_contract.json')['sources'].items():
        assert digest(path) == sha, path; checks += 1
    checks += 1; assert read(out/'paired_world_analysis.json')['source_sha256'] == digest('scripts/analyze_fragment_union_joint_v1.py')
    for filename, script in [('objective_cross_diagnostic.json', 'scripts/diagnose_fragment_union_objective_v1.py'),
                             ('visibility_diagnostic.json', 'scripts/diagnose_fragment_union_visibility_v1.py')]:
        assert read(out/filename)['source_sha256'] == digest(script); checks += 1
    with (out/'metrics.csv').open(encoding='utf-8', newline='') as file:
        records = list(csv.DictReader(file))
    assert len(records) == 288; checks += 1
    assert read(out/'figure_metadata.json')['source_data_sha256'] == digest(out/'metrics.csv'); checks += 1
    rows = read(out/'rows.json'); summary = read(out/'summary.json')
    assert len(rows) == summary['method_outputs'] == 288 and summary['API_paths'] == 48; checks += 1
    assert all(r['cost']['returned'] for r in rows) and not summary['exclusions']; checks += 1
    acquired = set(); seeds = set()
    for world in range(12):
        seeds.add(read(out/'scenes/fresh'/f'{world:04d}'/'evaluation/truth.json')['seed'])
        for action in [1, 6]:
            for obs in read(out/'fresh'/f'{world:04d}'/f'a{action}'/'observed.json')['history']+read(out/'fresh'/f'{world:04d}'/f'a{action}'/'observed.json')['current']:
                acquired.add((world, obs['token']))
    assert len(acquired) == summary['fresh_acquired_unique_payloads'] == 48; checks += 1
    assert seeds == set(range(22532026, 22532038)); checks += 1
    model = counterexample(); G = np.eye(4); delta = np.zeros((1, 3))
    rejected = model.energy('union_joint', G, delta, [[0, -1]])
    absorbed = model.energy('union_joint', G, delta, [[0, 0]])
    assert rejected == 12. and absorbed == 0.; checks += 1
    save(out/'known_defect.json', dict(status='negative counterexample: defect reproduced, NOT repaired',
        energy_reject_unrelated=rejected, energy_absorb_unrelated=absorbed,
        source_sha256=digest('tests/test_fragment_union_degeneracy_v1.py')))
    reports = ['research/fragment_union_joint_v1_2026-10-23.md', '.agent/HANDOFF.md', '.agent/TODO.md']
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                 if p.is_file() and p.name != 'final_integrity.json'}
    save(out/'final_integrity.json', dict(pass_all=True, checks=checks,
        audit_checks=audit['checks'], tests_before_run=11, posthoc_negative_counterexamples=2,
        frozen_source_count=len(frozen['sources']), prior_results_preserved=True,
        independent_fresh_worlds=12, fresh_stored_payloads=96, fresh_acquired_unique_payloads=48,
        qualified_motion_evaluation_m=.10, novelty_confirmed=False, candidate_promoted=False,
        reports={p:digest(p) for p in reports}, artifacts=artifacts,
        seal_source_sha256=digest(__file__), jobs_running=False, exclusions=[]))
    print('FINAL INTEGRITY PASS', checks, 'sources', len(frozen['sources']), 'prior artifacts preserved', flush=True)


if __name__ == '__main__': main()
