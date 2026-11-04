"""Check frozen inference, actual mask geometry, pose provenance and count contracts."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks=[]
    def check(name,value):
        checks.append({'check':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    port=Path('third_party/LivingScenes_cpu_port_v1');original=Path('third_party/LivingScenes')
    for suffix,cfgname in [('real','livingscenes_real_v1'),('native_masks','livingscenes_native_masks_v1')]:
        out=Path('results')/cfgname;start=read(out/'start_manifest.json');cfg=read(out/'config.json');root=Path(cfg['public_root'])
        for name,h in read(out/'prediction_seal.json')['sha256'].items():check(suffix+' prediction seal '+name,sha(out/name)==h)
        for name,h in start['sources_sha256'].items():check(suffix+' source '+name,sha(name)==h)
        for name,h in start['port_sha256'].items():check(suffix+' port '+name,sha(port/name)==h)
        for name,h in start['public_input_sha256'].items():check(suffix+' input '+name,sha(root/name)==h)
        check(suffix+' frozen config',sha(Path('configs')/(cfgname+'.json'))==start['config_sha256'])
        check(suffix+' checkpoint',sha(cfg['checkpoint'])==cfg['checkpoint_sha256'])
        scores=read(out/'scores.json');check(suffix+' no asserted motion accuracy',scores['motion_accuracy'] is None)
        for row in scores['summary']:
            if suffix=='real':total=row['correct']+row['wrong']+row['unmatched']
            else:total=row['correct']+row['wrong']+row['mixed_or_unresolved']
            check(suffix+' denominator '+row['scene']+row['method'],total==row['persistent'])
        public=read(root/'public.json')
        for scene,periods in public.items():
            if suffix!='native_masks':continue
            truth=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply')
            xyz=np.column_stack([truth[c] for c in ['x','y','z']]);used=[]
            for obj in periods['1']:
                with np.load(root/obj['path']) as z:
                    check(scene+obj['token']+' no persistent ID field',set(z.files)<=set(['xyz','rgb','point_indices']))
                    ix=z['point_indices'];check(scene+obj['token']+' original point indices',np.array_equal(z['xyz'],xyz[ix]));used.extend(ix.tolist())
            check(scene+' disjoint masks',len(used)==len(set(used)))
    for relative in ['ops/points_alignment.py','ops/utils.py','common/workaround/utils.py','structures/utils.py']:
        check('unchanged official PyTorch3D '+relative,sha(port/'pytorch3d'/relative)==sha(Path('third_party/pytorch3d_python_v074/pytorch3d')/relative))
    for relative in ['lib_shape_prior/core/models/sim3sdf_vanilla.py','lib_shape_prior/core/lib/vec_sim3/vec_dgcnn_atten.py','lib_more/utils.py']:
        check('unchanged model source '+relative,sha(port/relative)==sha(original/relative))
    probes=[]
    for token in ['o006','o011','o016','o020']:
        out=Path('results/livingscenes_real_v1/pose_probes')/token;seal=read(out/'seal.json')
        check(token+' pose source sealed',seal['source_sha256']==sha('scripts/run_livingscenes_pose_probe_v1.py'))
        check(token+' pose output sealed',seal['results_sha256']==sha(out/'results.json'))
        check(token+' pose input sealed',seal['input_sha256']==sha(out/'frozen_input.json'))
        rows=read(out/'results.json');check(token+' full 400 steps',len(rows)==2 and rows[1]['n_steps']==400)
        for row in rows:
            T=np.asarray(row['T_current_from_history']);R=T[:3,:3];c=np.array(row['historical_anchor_m'])
            check(token+str(row['optim'])+' proper finite SE3',np.isfinite(T).all() and np.allclose(R.T@R,np.eye(3),atol=1e-4) and abs(np.linalg.det(R)-1)<1e-4)
            check(token+str(row['optim'])+' material anchor formula',np.allclose(R@c+T[:3,3]-c,row['displacement_at_historical_anchor_m']))
            check(token+str(row['optim'])+' no surrogate motion truth',row['motion_error_to_independent_truth'] is None)
        probes.append({'token':token,'input':read(out/'frozen_input.json'),'results':rows})
    previous=read('results/real_temporal_v1/final_evidence_index.json')
    for name,h in previous['sources_sha256'].items():check('previous real source '+name,sha(name)==h)
    for name,h in previous['artifacts_sha256'].items():
        if name.endswith('prepared_v2\\public.json') or name.endswith('prepared_v2/public.json'):
            check('GT-mask public metadata matches previous seal',sha(name)==h)
    native_root=Path('data/livingscenes_native_masks_v1');native_public=read(native_root/'public.json')
    for source in read(native_root/'acquisition.json')['sources']:
        check('native mask acquisition '+source['path'],sha(source['path'])==source['sha256'])
        scene=next(p for p in Path(source['path']).parts if p in native_public)
        cloud=read_ply(source['path']);mapping=read(native_root/'evaluation/native_mask_token_mapping.json')[scene]
        for item in native_public[scene]['1']:
            ix=np.flatnonzero(cloud['instance_idx']==mapping[item['token']]);classes,counts=np.unique(cloud['class_idx'][ix],return_counts=True)
            with np.load(native_root/item['path']) as z:check(scene+item['token']+' native-only export provenance',np.array_equal(ix,z['point_indices']) and item['semantic_class']==int(classes[np.argmax(counts)]) and item['raw_points']==len(ix))
    old=read('results/fragment_union_joint_v1/frozen_contract.json')
    for name,h in old['sources'].items():check('previous frozen '+name,sha(name)==h)
    numerical=read('results/livingscenes_cpu_port_v1/model_numerical_checks/checks.json')
    check('independent KNN formulation agrees',all(v==0 for v in numerical['direct_knn_max_code_difference'].values()))
    check('single-patch rotation invariance',numerical['invariant_cosine']>.99999)
    result={'checks':checks,'passed':len(checks),'probe_results':probes,'scope':'Integration/provenance checks, not independent algorithm confirmation or CUDA equivalence proof.'}
    Path('results/livingscenes_real_v1/audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('AUDIT',len(checks),'PASS')

if __name__=='__main__':main()
