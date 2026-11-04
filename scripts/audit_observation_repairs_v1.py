"""Check artifacts, matched inputs, exact equivalence, and preservation of old work."""
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,ray_states

def main():
    checks=0
    def check(value,message):
        nonlocal checks
        if not value:raise AssertionError(message)
        checks+=1
    legacy=read('results/rigid_identity_cached_v1/final_evidence_index.json')
    for name,h in legacy['sources_sha256'].items():check(sha(name)==h,'Changed legacy source '+name)
    public=read('data/livingscenes_native_masks_v1/public.json')
    candidate=read('results/candidate_complete_v1/scores.json')
    for scene in ['scene_a','scene_c']:
        dest=Path('results/candidate_complete_v1')/scene
        for name,h in read(dest/'seal.json')['sha256'].items():check(sha(dest/name)==h,'Candidate seal '+name)
        manifest=read(dest/'start_manifest.json')
        for field in ['sources','inputs']:
            for name,h in manifest[field].items():check(sha(name)==h,'Candidate manifest '+name)
        pred=read(dest/'predictions.json');proposals=read(dest/'pair_proposals.json');H={h['token']:h for h in public[scene]['0']};C={c['token']:c for c in public[scene]['1']}
        check(len(proposals['poses'])==len(pred['history'])*len(pred['current']),'All HxC pair proposals')
        for method,r in pred['methods'].items():
            check(len(r['assignment'])==len(pred['current']),'Every current group predicted')
            if method.startswith('gated'):
                for j,i in enumerate(r['assignment']):
                    if i>=0:check(H[pred['history'][i]]['semantic_class']==C[pred['current'][j]]['semantic_class'],'Hard gate respected')
    check(len(candidate['summary'])==6,'All six candidate variants')
    check(all(r['persistent']==34 and r['new']==6 for r in candidate['summary']),'No object exclusions')
    for folder in ['results/ray_material_evidence_v1','results/ray_material_full_resolution_v1']:
        dest=Path(folder);manifest=read(dest/'start_manifest.json')
        for field in ['sources','inputs']:
            for name,h in manifest[field].items():
                check(sha(name)==h,'Ray manifest '+name)
                if field=='inputs':check('evaluation' not in Path(name).parts,'No evaluation inputs')
        for name,h in read(dest/'prediction_seal.json')['sha256'].items():check(sha(dest/name)==h,'Ray seal '+name)
        scores=read(dest/'scores.json');check(len(scores['rows'])==12*8*2,'All method/pair/scope scores')
        size=0
        for pth in sorted(dest.glob('pair_*/predictions.json')):
            p=read(pth);size+=len(p['pool']);check(p['GT_pose_read'] is False,'Inference GT access flag')
            check(p['choices']['beta_material']==p['choices']['ordinary_exact_beta_equivalent'],'Same selector as exact ordinary control')
            for entry in p['pool']:check(abs(entry['scores']['beta_material']-entry['scores']['ordinary_exact_beta_equivalent'])<1e-12,'Algebraic equivalence')
        check(size==3941,'Identical 3941-pose pool')
    cfg=read('results/ray_material_full_resolution_v1/effective_configuration.json');root=Path(cfg['public_root']);metadata=read(root/'public.json')
    for row in metadata['clouds']:
        f=Frame(root,row['stem'],cfg);state,*_=ray_states(f.xyz,f.depth,f.mask,f.K,cfg['depth_tolerance_m'],cfg['depth_valid_interval_m'])
        check(np.all(state==1),'Self-projection roundtrip '+row['stem'])
    for scene in ['scene_c_0','scene_c_1']:
        for frame in range(3):
            a=np.array(Image.open(Path('data/rescan_depth_decoded_probe_v1')/f'{scene}_{frame:04d}_reference_mm.png'));b=np.array(Image.open(Path('data/rescan_depth_decoded_probe_v1')/f'reference_c{scene[-1]}_{frame}.pgm'))
            check(np.array_equal(a,b),'Original codec bytewise comparison')
    save('results/ray_material_full_resolution_v1/audit.json',dict(passed=True,checks=checks,scope='Software/provenance/matched-input verification, not algorithm success or sensor metrology validation.',legacy_sources_unchanged=len(legacy['sources_sha256']),all_observation_points_used=True))
    paths=[Path('research/observation_model_repairs_v1_2026-10-24.md'),Path('scripts/probe_rescan_raw_header_v1.py')]+[p for pattern in ['scripts/*candidate_complete_v1.py','scripts/*complete_candidate_v1.py','scripts/*ray_material*v1.py','scripts/*observation_repairs_v1.py','scripts/*rescan*probe_v1.py','configs/candidate_complete_v1.json','configs/ray_material*v1.json','inspection_decisions/ray_material_evidence_v1.py'] for p in Path('.').glob(pattern)]
    artifacts=[p for folder in ['results/candidate_complete_v1','results/ray_material_evidence_v1','results/ray_material_full_resolution_v1','data/rescan_raw_probe_v1','data/rescan_decoder_source_probe_v1','data/rescan_depth_decoded_probe_v1'] for p in Path(folder).rglob('*') if p.is_file() and p.name!='final_evidence_index.json']
    save('results/ray_material_full_resolution_v1/final_evidence_index.json',dict(sources_sha256={str(p):sha(p) for p in paths},artifacts_sha256={str(p):sha(p) for p in artifacts},novelty='Not established; single-ray Beta material score exactly equals ordinary weighted overlap.',next_model='Joint shared-material partial transport with a shared unknown surface; not implemented or confirmed yet.'))
    print('PASS',checks,'checks;',len(artifacts),'artifacts; old sources unchanged',len(legacy['sources_sha256']))

if __name__=='__main__':main()
