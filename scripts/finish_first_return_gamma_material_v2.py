"""Seal all scenes, audit without GT, only then evaluate the unchanged task."""
from pathlib import Path
import shutil,os,subprocess,sys
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    root=Path('results/first_return_gamma_material_v2');options=read('configs/first_return_gamma_material_v2.json')
    if not (root/'prediction_seal.json').exists():
        runs={};sources={};inputs={};times=[]
        for j in range(3):
            shard=root/f'shard_{j}';seal=read(shard/'prediction_seal.json');assert seal['complete']
            for p,h in seal['sha256'].items():assert sha(shard/p)==h
            m=read(shard/'manifest.json')
            for f,target in [('sources',sources),('inputs',inputs)]:
                for p,h in m[f].items():assert sha(p)==h;assert p not in target or target[p]==h;target[p]=h
            for r in read(shard/'runs.json'):assert r['scene'] not in runs;runs[r['scene']]=(shard,r)
            inputs[str(shard/'prediction_seal.json')]=sha(shard/'prediction_seal.json');times.append(seal['elapsed_s'])
        expected=read(Path(options['parent_root'])/'runs.json');assert len(runs)==len(expected)==8;ordered=[]
        for e in expected:
            shard,r=runs[e['scene']];dest=root/e['scene'];assert dest.resolve().parent==root.resolve()
            if not dest.exists():shutil.copytree(shard/e['scene'],dest,copy_function=os.link)
            ordered.append(r)
        save(root/'runs.json',ordered);save(root/'config.json',read(root/'shard_0/config.json'));save(root/'manifest.json',{'sources':sources,'inputs':inputs,'motion_limits':read(root/'shard_0/manifest.json')['motion_limits'],'GT_read':False,'hardlinked_immutable_scene_files':True,'depth_noise_integrated':False});save(root/'prediction_seal.json',{'complete':True,'GT_read':False,'scenes':8,'assets':192,'continuous_fits':768,'max_shard_elapsed_s':max(times),'sha256':{str(p.relative_to(root)):sha(p) for p in root.iterdir() if p.is_file()}|{str(p.relative_to(root)):sha(p) for e in expected for p in (root/e['scene']).iterdir() if p.is_file()}});print('ALL_8_SEALED_BEFORE_GT',flush=True)
    for module,marker,key in [('audit_first_return_gamma_material_v2','audit.json','auditor_sha256'),('score_first_return_gamma_material_v2','evaluation_sources.json','script_sha256')]:
        if not (root/marker).exists() or read(root/marker).get(key)!=sha(Path('scripts')/(module+'.py')):subprocess.run([sys.executable,'-m','scripts.'+module],check=True)
        assert read(root/'audit.json')['passed']
    print('FULL_HAZARD_CHAIN_COMPLETE',flush=True)


if __name__=='__main__':main()
