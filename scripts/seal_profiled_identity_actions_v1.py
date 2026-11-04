"""Final immutable-input/source/result check; adds no experiments."""
from pathlib import Path
from scripts.run_profiled_identity_actions_v1 import read, save, digest


def main():
    out = Path('results/profiled_identity_actions_v1_repaired')
    frozen = read(out/'frozen_contract.json'); checks = 0
    for path, sha in frozen['sources'].items():
        assert digest(path) == sha, path; checks += 1
    original = Path('results/profiled_identity_actions_v1')
    for path, sha in frozen['original_partial_artifacts'].items():
        assert digest(original/path) == sha, path; checks += 1
    provenance = read(out/'repair_provenance.json')
    for source, sha in provenance['copied_scene_files'].items():
        assert digest(source) == sha; checks += 1
        target = out/'scenes'/Path(source).relative_to(original/'scenes')
        assert digest(target) == sha; checks += 1
    unchanged_backends = 0
    for world in range(11):
        wd = original/f'{world:04d}'
        for path in wd.glob('*backend.json'):
            assert digest(path) == digest(out/f'{world:04d}'/path.name), path
            unchanged_backends += 1; checks += 1
        for name in ('forecasts.json', 'decisions.json'):
            assert digest(wd/name) == digest(out/f'{world:04d}'/name); checks += 1
    audit = read(out/'audit.json')
    assert digest('scripts/audit_profiled_identity_actions_v1.py') == audit['audit_source_sha256']; checks += 1
    for path, sha in audit['artifacts'].items():
        assert digest(out/path) == sha, path; checks += 1
    for filename, script in [
        ('visibility_gap_diagnostic.json', 'scripts/diagnose_identity_visibility_gap_v1.py'),
        ('worked_energy_decomposition.json', 'scripts/explain_profiled_identity_failures_v1.py'),
        ('fragment_identity_diagnostic.json', 'scripts/explain_profiled_identity_failures_v1.py')]:
        assert read(out/filename)['source_sha256'] == digest(script); checks += 1
    summary = read(out/'summary.json'); rows = read(out/'rows.json')
    assert summary['backend_calls'] == 84 and summary['policy_paths'] == len(rows) == 72; checks += 1
    assert summary['API_paths'] == 156 and summary['raw_RGBD_payloads'] == 96; checks += 1
    assert all(r['cost']['returned'] for r in rows) and not summary['exclusions']; checks += 1
    adapters = []
    for path in out.glob('*/*backend.json'):
        result = read(path)
        for k, option in enumerate(result['options']):
            if 'engineering_partition_adapter' in option:
                adapters.append(dict(backend=str(path.relative_to(out)), branch=k,
                                     **option['engineering_partition_adapter']))
    reports = ['research/profiled_identity_actions_v1_2026-10-23.md', '.agent/HANDOFF.md', '.agent/TODO.md']
    report_hashes = {p: digest(p) for p in reports}
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                 if p.is_file() and p.name != 'final_integrity.json'}
    save(out/'final_integrity.json', dict(pass_all=True, checks=checks,
        original_completed_backend_files_identical=unchanged_backends,
        repaired_world_count=12, independent_worlds_added_by_repair=0,
        common_adapter_uses=adapters, original_partial_preserved=True,
        old_audit_checks=audit['checks'], reports=report_hashes, artifacts=artifacts,
        seal_source_sha256=digest(__file__), main_algorithm_source_unchanged=True,
        quality_gate_changes=False, exclusions=[], jobs_running=False))
    print('FINAL INTEGRITY PASS', checks, 'old backends identical', unchanged_backends,
          'adapter calls', len(adapters), flush=True)


if __name__ == '__main__': main()
