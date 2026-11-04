"""Merge all fixed shards and seal eight scenes before this evaluator reads GT."""
from pathlib import Path
import shutil
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    cfg=read('configs/sparse_material_evidence_v1.json');root=Path(cfg['output_root']);assert not (root/'prediction_seal.json').exists();runs=[];sources={};inputs={};mapping={};times=[]
    for j in range(cfg['execution_shards']):
        shard=root/f'shard_{j}';seal=read(shard/'prediction_seal.json');assert seal['complete'] and seal['subset_shard']==j
        for name,h in seal['sha256'].items():assert sha(shard/name)==h
        for field,target in [('sources',sources),('inputs',inputs)]:
            for name,h in read(shard/'manifest.json')[field].items():assert sha(name)==h;assert name not in target or target[name]==h;target[name]=h
        for r in read(shard/'runs.json'):assert r['scene'] not in mapping;mapping[r['scene']]=(shard,r)
        inputs[str(shard/'prediction_seal.json')]=sha(shard/'prediction_seal.json');times.append(seal['elapsed_s'])
    expected=read(Path(cfg['parent_root'])/'runs.json');assert len(mapping)==8 and set(mapping)=={r['scene'] for r in expected}
    for r in expected:
        shard,item=mapping[r['scene']];dest=root/r['scene'];assert dest.resolve().parent==root.resolve() and not dest.exists();shutil.copytree(shard/r['scene'],dest);runs.append(item)
    config=read(root/'shard_0/config.json');save(root/'config.json',config);save(root/'runs.json',runs);sources[str(Path(__file__))]=sha(Path(__file__));save(root/'manifest.json',{'sources':sources,'inputs':inputs,'motion_limits':read(root/'shard_0/manifest.json')['motion_limits'],'GT_read':False,'complete_8scenes':True,'finite_H_only_material_states':True,'not_independent_384_predictions':True,'primary_stage':'final'});save(root/'prediction_seal.json',{'complete':True,'scenes':8,'assets':192,'methods':len(config['methods']),'continuous_fits':sum(r['continuous_fits'] for r in runs),'candidate_geometries':sum(r['candidate_geometries'] for r in runs),'material_state_factor_evaluations':sum(r['material_state_factor_evaluations'] for r in runs),'max_shard_elapsed_s':max(times),'GT_read':False,'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('FULL_EVIDENCE_BATCH_SEALED',flush=True)


if __name__=='__main__':main()
