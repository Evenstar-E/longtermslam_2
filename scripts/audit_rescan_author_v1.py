"""Audit input sanitization and the exact scope of the baseline's memory port."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    original=Path('third_party/Rescan');port=Path('third_party/Rescan_memory_port_v1');same=[];changed=[]
    for p in original.rglob('*'):
        if not p.is_file() or '.git' in p.relative_to(original).parts or 'bin' in p.relative_to(original).parts:continue
        relative=p.relative_to(original)
        if not (port/relative).exists():raise AssertionError(relative)
        (same if sha(p)==sha(port/relative) else changed).append(str(relative).replace('\\','/'))
    assert set(changed)=={'lib/rs/rs_database.h','apps/seg2rsdb/main.cpp'},changed
    checks=[]
    for root in [Path('results/rescan_author_v1/inference_memory_port'),Path('results/rescan_author_v1/inference_long_scene_a_v2')]:
        for scene in ['scene_a','scene_b','scene_c']:
            current=root/scene/'current.ply'
            if not current.exists():continue
            original_current=Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply'
            a=read_ply(original_current);b=read_ply(current)
            assert np.all(b['instance_idx']==-1) and np.all(b['class_idx']==0)
            assert all(np.array_equal(a[n],b[n]) for n in a.dtype.names if n not in ['class_idx','instance_idx'])
            checks.append({'input':str(current),'persistent_ids_removed':True,'current_semantic_gt_removed':True,'all_other_fields_exact':True,'sha256':sha(current)})
            seal=root/scene/'prediction_seal.json'
            if seal.exists():
                data=json.loads(seal.read_text(encoding='utf-8'))
                for name,h in data['files_sha256'].items():assert sha(root/scene/name)==h,(root,scene,name)
    record={'algorithm_and_dependency_files_identical':len(same),'changed_files':changed,
        'change_scope':'Memory buffers, initialization and ownership only; see memory_only.patch; no scores/proposals/SA/threshold changes',
        'sanitized_inputs':checks,'completed_prediction_seals_verified':True,
        'ASAN_scope':'Ported seg2rsdb on scene_c returned 0 with no ASAN error; not a proof all executable paths are memory safe'}
    Path('results/rescan_author_v1/algorithm_input_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('RESCAN AUDIT',len(same),'unchanged files;',changed,';',len(checks),'sanitized inputs')

if __name__=='__main__':main()
