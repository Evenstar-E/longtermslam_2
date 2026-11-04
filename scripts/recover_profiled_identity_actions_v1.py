"""Uniform engineering repair replay on byte-identical original 12 worlds."""
import shutil
from pathlib import Path
import inspection_decisions.profiled_identity_actions_v1 as algorithm
import inspection_decisions.stable_branch_readout_v1 as adapter
import scripts.run_profiled_identity_actions_v1 as runner


def install():
    algorithm.readout = adapter.readout


def main():
    source = Path('results/profiled_identity_actions_v1')
    destination = Path('results/profiled_identity_actions_v1_repaired')
    original = runner.read(source/'frozen_contract.json')
    for path, sha in original['sources'].items():
        assert runner.digest(path) == sha, path
    source_artifacts = {str(p.relative_to(source)): runner.digest(p) for p in source.rglob('*') if p.is_file()}
    old_cfg = runner.merged_config
    def config():
        cfg = old_cfg(); cfg['output'] = str(destination)
        cfg['status'] = 'engineering repair replay on identical observed 12 worlds; not independent confirmation'
        return cfg
    copied = {}
    def copy_scene(root, world, cfg, sensor):
        origin = source/'scenes'/f'{world:04d}'
        shutil.copytree(origin, root)
        for path in origin.rglob('*'):
            if path.is_file():
                target = root/path.relative_to(origin)
                assert runner.digest(path) == runner.digest(target)
                copied[str(path)] = runner.digest(path)
    original_save = runner.save
    def save(path, value):
        if Path(path).name == 'frozen_contract.json':
            value = dict(value)
            value.update(frozen_before_world_generation=False, frozen_before_repaired_replay=True,
                engineering_repair_record='research/profiled_identity_actions_repair_v1_2026-10-23.md',
                original_partial_artifacts=source_artifacts, original_completed_worlds=11,
                new_independent_worlds=0, original_failure='world11 IndexError: frame-dependent component repartition')
        original_save(path, value)
    runner.merged_config = config; runner.generate = copy_scene; runner.save = save
    runner.SOURCES += [__file__, 'inspection_decisions/stable_branch_readout_v1.py',
        'tests/test_stable_branch_readout_v1.py', 'research/profiled_identity_actions_repair_v1_2026-10-23.md']
    install(); runner.run()
    assert all(runner.digest(source/p) == sha for p, sha in source_artifacts.items())
    runner.save(destination/'repair_provenance.json', dict(original_artifacts_unchanged=True,
        copied_scene_files=copied, new_independent_worlds=0, partial_original_retained=True))


if __name__ == '__main__': main()
