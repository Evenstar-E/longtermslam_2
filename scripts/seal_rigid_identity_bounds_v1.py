"""Seal completed algorithm and public real-motion evidence, preserving older experiments."""
from pathlib import Path
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    root=Path('results/rigid_identity_cached_v1');audit=read(root/'audit.json');assert all(r['pass'] for r in audit['checks'])
    sources=[]
    for folder,patterns in [('scripts',['*rigid_identity*v1.py','*rigid_mass*v1.py','*ycbineoat*.py']),('inspection_decisions',['rigid_identity*v1.py']),('configs',['rigid_identity_bounds_v1.json','ycbineoat_motion*.json']),('tests',['test_rigid_identity_bounds_v1.py']),('research',['rigid_identity*2026-10-24.md'])]:
        for pattern in patterns:sources.extend(Path(folder).glob(pattern))
    sources=sorted(set(sources))
    roots=[root,Path('results/rigid_identity_bounds_v1'),Path('results/rigid_identity_cache_check_v1'),Path('results/rigid_identity_cuts_v1'),Path('results/rigid_identity_cuts_v1_completed'),Path('results/rigid_identity_mass_cuts_v1'),Path('results/rigid_identity_budget_fill_v1'),Path('results/ycbineoat_motion_v1'),Path('data/ycbineoat_motion_prepared_v1'),Path('data/ycbineoat_motion_prepared_v2')]
    artifacts={str(p):sha(p) for parent in roots for p in parent.rglob('*') if p.is_file() and p.name!='final_evidence_index.json' and '__pycache__' not in p.parts}
    inventory=read('data/real_cross_session_v1/rescan/inventory.json');entries=inventory['entries']
    availability={'inventory_sha256':sha('data/real_cross_session_v1/rescan/inventory.json'),'archive_url':inventory['url'],'h264_streams':sum(p['name'].endswith('.h264') for p in entries),'depth_streams':sum(p['name'].endswith('.depth') for p in entries),'raw_streams_decoded':False,'camera_intrinsics_and_trajectories_verified':False}
    save(root/'rescan_raw_availability.json',availability);artifacts[str(root/'rescan_raw_availability.json')]=sha(root/'rescan_raw_availability.json')
    acquisition=read('data/ycbineoat_motion_v1_parallel/acquisition.json');archive=Path('data/ycbineoat_motion_v1_parallel/cracker_box_reorient.tar.gz');assert sha(archive)==acquisition['sha256']
    record={'sources_sha256':{str(p):sha(p) for p in sources},'artifacts_sha256':artifacts,'public_motion_archive':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':acquisition['sha256'],'acquisition_sha256':sha('data/ycbineoat_motion_v1_parallel/acquisition.json')},
        'checks_passed':audit['passed'],'new_unit_tests_observed_passed':6,'cache_checks_observed_passed':99,'independent_annotation_control_available':True,'new_algorithm_implemented_and_executed':True,'new_algorithm_task_gain_established':False,
        'scope':'Continuous finite-domain surrogate-energy bounds, not a physical correctness certificate. Rescan34persistent+6new; one real375frame sequence12correlatedpairs/16selectedframes. No new mainline scenes or sample expansion.',
        'superseded_attempts':'Uncached first run stopped for exact caching repair; first cut writer failed JSON serialization; prepared_v1 failed masks-vs-gt_mask path. Preserved, not scored as failure0 or hidden.',
        'raw_multiasset_next':availability}
    save(root/'final_evidence_index.json',record);print('SEALED',len(sources),'sources',len(artifacts),'artifacts')

if __name__=='__main__':main()
