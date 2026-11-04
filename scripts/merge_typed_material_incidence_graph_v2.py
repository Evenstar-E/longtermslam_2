"""Verify all fixed shards, merge ALL eight scenes, then seal before truth."""
from pathlib import Path
import shutil
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    root=Path('results/typed_material_incidence_graph_v2');assert not (root/'prediction_seal.json').exists();cfg=read(root/'shard_0/config.json');cfg.pop('execution_shard');expected=read(Path(cfg['parent_root'])/'runs.json');by_scene={};sources={};inputs={};elapsed=[]
    for j in range(cfg['execution_shards']):
        shard=root/('shard_'+str(j));seal=read(shard/'prediction_seal.json');assert seal['complete'] and seal['subset_shard']==j
        for name,h in seal['sha256'].items():assert sha(shard/name)==h
        manifest=read(shard/'manifest.json')
        for field,target in [('sources',sources),('inputs',inputs)]:
            for name,h in manifest[field].items():assert sha(name)==h;assert name not in target or target[name]==h;target[name]=h
        inputs[str(shard/'prediction_seal.json')]=sha(shard/'prediction_seal.json');elapsed.append(seal['elapsed_s'])
        for item in read(shard/'runs.json'):assert item['scene'] not in by_scene;by_scene[item['scene']]=(shard,item)
    assert set(by_scene)=={r['scene'] for r in expected} and len(by_scene)==8
    runs=[]
    for item in expected:
        tag=item['scene'];shard,run=by_scene[tag];dest=root/tag;assert dest.resolve().parent==root.resolve() and not dest.exists();shutil.copytree(shard/tag,dest);runs.append(run)
    save(root/'config.json',cfg);save(root/'runs.json',runs);sources[str(Path(__file__))]=sha(Path(__file__));sources['scripts/run_typed_material_incidence_graph_v1.py']=sha('scripts/run_typed_material_incidence_graph_v1.py')
    legacy=Path('results/typed_material_incidence_graph_v1');tag='aligned_s270100_h1';equivalence={'old_first_scene_exists':(legacy/tag/'predictions.json').exists(),'GT_read':False}
    if equivalence['old_first_scene_exists']:
        for field in ['sources','inputs']:
            for name,h in read(legacy/'manifest.json')[field].items():assert sha(name)==h
        old=read(legacy/tag/'predictions.json');new=read(root/tag/'predictions.json');assert len(old)==len(new)==48;prediction_max=0.;slot_changes=0;conditional_max=0.
        for a,b in zip(old,new):
            assert (a['asset'],a['family'])==(b['asset'],b['family']) and a['graphs']==b['graphs']
            for method in cfg['methods']:
                conditional_max=max(conditional_max,float(np.max(abs(np.array(a['initial_fits'][method]['parameters'])-np.array(b['initial_fits'][method]['parameters'])))))
                for stage in ['pool','catalogue','final','conditional']:
                    aa=a['methods'][method]['stages'][stage];bb=b['methods'][method]['stages'][stage];slot_changes+=int(aa['slots']!=bb['slots'])
                    for pose in ['T1','T2']:prediction_max=max(prediction_max,float(np.max(abs(np.array(aa[pose])-np.array(bb[pose])))))
        with np.load(legacy/tag/'global_costs.npz') as a,np.load(root/tag/'global_costs.npz') as b:
            score_max=max(float(np.max(abs(a[k]-b[k]))) for k in ['unary','catalogue_unary']);argmin_changes=sum(int(np.sum(a[k]!=b[k])) for k in ['indices','catalogue_indices'])
        assert score_max<1e-12 and conditional_max<1e-12;equivalence.update(initializations=48,max_global_matrix_score_difference=score_max,within_slot_argmin_changes=argmin_changes,stage_identity_changes=slot_changes,max_prediction_matrix_difference=prediction_max,max_conditional_parameter_difference=conditional_max)
        for name in ['predictions.json','global_costs.npz','candidates.pkl']:inputs[str(legacy/tag/name)]=sha(legacy/tag/name)
    save(root/'execution_equivalence.json',equivalence);save(root/'manifest.json',{'sources':sources,'inputs':inputs,'GT_read':False,'familiar_DEV':True,'motion_limits':read(root/'shard_0/manifest.json')['motion_limits'],'IDs_reselected':True,'all24_slots_catalogue':True,'all_fixed_shards_merged':True,'not_equal_FLOPs':True});save(root/'prediction_seal.json',{'complete':True,'GT_read':False,'scenes':8,'conditional_initializations':384,'initial_and_reassigned_fits':3072,'candidate_evaluations':sum(r['candidate_evaluations'] for r in runs),'elapsed_s_shard_max':max(elapsed),'elapsed_s_shard_sum':sum(elapsed),'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('FULL_BATCH_SEALED',equivalence,flush=True)


if __name__=='__main__':main()
