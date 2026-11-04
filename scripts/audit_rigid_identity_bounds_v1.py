"""Audit completed algorithm experiments, continuous partitions and independent GT separation."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    checks=[]
    def check(name,value):
        checks.append({'name':name,'pass':bool(value)})
        if not value:raise AssertionError(name)
    cfg=read('configs/rigid_identity_bounds_v1.json')
    for scene in cfg['spaces']:
        base=Path('results/rigid_identity_cached_v1')/scene
        for directory in [base,Path('results/rigid_identity_cuts_v1_completed')/scene,Path('results/rigid_identity_mass_cuts_v1')/scene]:
            for name,h in read(directory/'prediction_seal.json')['sha256'].items():check('sealed '+str(directory/name),sha(directory/name)==h)
        manifest=read(base/'start_manifest.json')
        for name,h in manifest['sources_sha256'].items():check('source '+name,sha(name)==h)
        for name,h in manifest['inputs_sha256'].items():check('input '+name,sha(name)==h)
        for folder in base.glob('class_*'):
            b=read(folder/'bounded.json');r=read(folder/'results.json');n=len(r['current']);m=len(r['history']);check('all48splits '+str(folder),len(b['leaf_cells'])==m+cfg['max_splits_per_class'])
            rootlo=np.r_[[-np.pi]*3,b['root_position_lo']];roothi=np.r_[[np.pi]*3,b['root_position_hi']];rootvol=np.prod(roothi-rootlo)
            for i in range(m):
                leaves=[v for v in b['leaf_cells'] if v['owner']==i];check('partition volume '+str(folder)+str(i),np.isclose(sum(np.prod(np.array(v['hi'])-v['lo']) for v in leaves),rootvol,rtol=1e-10))
            check('bounds below feasible UB '+str(folder),all(t['lower_bound']<=t['upper_bound']+1e-6 for t in b['trace']))
            for T in np.array(b['best']['poses']):check('proper SE3 '+str(folder),np.isfinite(T).all() and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) and abs(np.linalg.det(T[:3,:3])-1)<1e-4)
            whole=read(Path('results/rigid_identity_cuts_v1_completed')/scene/(folder.name+'.json'));mass=read(Path('results/rigid_identity_mass_cuts_v1')/scene/(folder.name+'.json'))
            check('cut uses same UB '+str(folder),whole['same_feasible_upper_bound']==mass['same_feasible_upper_bound']==b['best']['energy'])
            check('no extra ICP in bound ablation '+str(folder),whole['additional_ICP_calls']==mass['additional_ICP_calls']==0)
            check('cut LB <= UB '+str(folder),max(whole['cut_master']['lower_bound'],mass['cut_master']['lower_bound'])<=b['best']['energy']+1e-6)
            for j,a in enumerate(b['best']['assignment']):
                for name,sets in [('base',b['identity_sets']),('whole',whole['identity_sets']),('mass',mass['identity_sets'])]:check('incumbent identity not excluded '+str(folder)+str(j)+name,a in sets[j]['possible_identities'])
            filled=read(Path('results/rigid_identity_budget_fill_v1')/scene/(folder.name+'.json'))
            check('ordinary control not under budget '+str(folder),filled['total_extra_calls']>=filled['candidate_extra_calls'])
            check('ordinary overshoot bounded '+str(folder),filled['total_extra_calls']-filled['candidate_extra_calls']<=m-1)
    scores=read('results/rigid_identity_cached_v1/scores.json')
    for row in scores['summary']:check('score denominator '+row['scene']+row['method'],row['correct']+row['wrong']+row['mixed_or_unresolved']==row['persistent'])
    motion=Path('results/ycbineoat_motion_v1')
    for name,h in read(motion/'prediction_seal.json')['sha256'].items():check('motion sealed '+name,sha(motion/name)==h)
    for name,h in read(motion/'start_manifest.json')['sources_sha256'].items():check('motion source '+name,sha(name)==h)
    for name,h in read(motion/'start_manifest.json')['public_sha256'].items():check('motion public '+name,sha(name)==h)
    prepared=Path('data/ycbineoat_motion_prepared_v2');acq=read(prepared/'acquisition.json')
    for row in acq['records']:check('motion raw extraction '+row['path'],sha(prepared/row['path'])==row['sha256'])
    check('pose-independent pair selection preserved',read(prepared/'selection.json')['pairs']==read('data/ycbineoat_motion_prepared_v1/selection.json')['pairs'])
    check('gzip CRC validation completed',acq['archive_gzip_crc_read_to_EOF'])
    check('archive SHA verified',sha('data/ycbineoat_motion_v1_parallel/cracker_box_reorient.tar.gz')==acq['archive_sha256'])
    mscore=read(motion/'scores.json');check('12 motion pairs',all(r['pairs']==12 for r in mscore['summary']))
    check('all reference poses within declared domain',all(r['GT_pose_in_declared_domain'] for r in mscore['surrogate_energy_falsification']))
    for name,h in read('results/fragment_union_joint_v1/frozen_contract.json')['sources'].items():check('old frozen source '+name,sha(name)==h)
    for name,h in read('results/livingscenes_real_v1/final_evidence_index.json')['sources_sha256'].items():check('previous learned source '+name,sha(name)==h)
    save('results/rigid_identity_cached_v1/audit.json',{'checks':checks,'passed':len(checks),'claim':'Software, provenance, domain partition and explicit energy checks; not physical-noise or novelty certificate.'})
    print('AUDIT',len(checks),'PASS')

if __name__=='__main__':main()
