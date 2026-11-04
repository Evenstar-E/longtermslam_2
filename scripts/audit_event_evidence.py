"""Check exact source arrays, calibration, projection, event windows and immutable input lineage."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np,yaml
from PIL import Image
from scripts.event_geometry import rigid_matrix,world_mesh,projected_surface,depth_evidence
from scripts.build_event_robust_pairs import select
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
base=Path('data/event_evidence_v1');results=Path('results/event_evidence_v1');analysis=results/'analysis_001';pairs_dir=results/'robust_pairs_001'
checks=[]
def add(k,v):checks.append({'check':k,'passed':bool(v)})
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
plan=read(base/'plan.json');acq=read(base/'acquisition.json');cfg=read(base/'frozen_config.json')
add('complete_213_original_frames',acq['complete'] and len(acq['frames'])==213)
add('6_original_static_ids',len(plan['nodes'])==6 and len({r['id'] for r in plan['nodes']})==6)
add('240_requests_retained',len(acq['requests'])==len(plan['requests'])==240)
add('outside_requests_explicit',sum(r.get('reason')=='outside_recording' for r in acq['requests'])==4)
add('frozen_config_bytes',Path('configs/event_evidence_v1.json').read_bytes()==(base/'frozen_config.json').read_bytes())
add('config_hash',sha('configs/event_evidence_v1.json')==plan['protocol_sha256']==acq['protocol_sha256'])
add('plan_hash',sha(base/'plan.json')==acq['input_plan_sha256'])
frames={r['stamp_ns']:r for r in acq['frames']}
for frame in acq['frames']:
    stamp=frame['stamp_ns'];T=np.array(frame['T_world_camera']);R=T[:3,:3]
    add('proper_camera_pose:'+str(stamp),np.allclose(R.T@R,np.eye(3),atol=1e-10) and np.isclose(np.linalg.det(R),1))
    pose=frame['pose_record'];ext=acq['original_extrinsic']
    add('pose_composition:'+str(stamp),np.allclose(T,rigid_matrix(pose['translation'],pose['quaternion_xyzw'])@rigid_matrix(ext['translation'],ext['quaternion_xyzw']),atol=1e-12) and frame['pose_gap_ns']==0)
    for kind,ch in frame['channels'].items():
        add('file_hash:'+str(stamp)+':'+kind,sha(ch['path'])==ch['file_sha256'])
        a=np.load(ch['path']) if kind=='depth' else np.array(Image.open(ch['path']).convert('RGB') if kind=='rgb' else Image.open(ch['path']),dtype=np.uint8 if kind=='rgb' else np.uint16)
        add('exact_source_array:'+str(stamp)+':'+kind,hashlib.sha256(a.tobytes()).hexdigest()==ch['array_sha256'] and list(a.shape)==ch['shape'] and str(a.dtype)==ch['dtype'])
for request in acq['requests']:
    ns=request.get('selected_stamp_ns')
    add('request_contract:'+request['id'],(request.get('reason')=='outside_recording' and not plan['bag_start_ns']<=request['time_ns']<=plan['bag_end_ns']) or (ns in frames and abs(ns-request['time_ns'])<=.03e9))
for n in plan['nodes']:add('mesh_source:'+n['name'],sha(n['mesh_path'])==n['mesh_sha256'])
rows=read(analysis/'frame_evidence.json');summary=read(analysis/'summary.json')
add('1278_unique_frame_instance_records',len(rows)==1278 and len({(r['instance'],r['stamp_ns']) for r in rows})==1278)
for r in rows:
    for tol,e in r['sensitivity'].items():
        add('partition:'+r['instance']+':'+str(r['stamp_ns'])+':'+tol,e['geometry_support']+e['free']+e['occluded']+e['invalid']==e['projected_pixels'] and e['class_support']+e['class_mismatch']==e['geometry_support'])
    add('primary_tolerance:'+r['instance']+':'+str(r['stamp_ns']),r['primary']==r['sensitivity']['0.05'])
lookup={(r['instance'],r['stamp_ns']):r for r in rows}
remap=yaml.safe_load(Path('third_party/khronos_ros1_deps/hydra/config/label_remaps/tesse_cd_apartment.yaml').read_text());lut=np.zeros(65536,np.int32)
for r in remap:lut[r['sub_id']]=r['super_id']
K=np.array(acq['camera']['K']).reshape(3,3)
for frame in [acq['frames'][0],acq['frames'][len(acq['frames'])//2],acq['frames'][-1]]:
    T=np.array(frame['T_world_camera']);inverse=np.linalg.inv(T);depth=np.load(frame['channels']['depth']['path']);labels=lut[np.array(Image.open(frame['channels']['labels']['path']),dtype=np.uint16)]
    for n in plan['nodes']:
        world=world_mesh(np.load(n['mesh_path']),n['bbox'])
        cam=(world-T[:3,3])@T[:3,:3];positive=cam[:,2]>.1
        if positive.any():
            sample=world[positive][:100];direct=((sample-T[:3,3])@T[:3,:3])@K.T;direct=direct[:,:2]/direct[:,2,None]
            cv,_=cv2.projectPoints(sample,cv2.Rodrigues(inverse[:3,:3])[0],inverse[:3,3],K,None)
            add('opencv_projection_parity:'+n['name']+':'+str(frame['stamp_ns']),np.allclose(direct,cv.reshape(-1,2),atol=1e-7))
        xy,z=projected_surface(world,T,K,acq['camera']['width'],acq['camera']['height'])
        e=depth_evidence(xy,z,depth,labels,n['semantic_label'],.05,cfg['diagnostic'])
        add('selected_evidence_replay:'+n['name']+':'+str(frame['stamp_ns']),e==lookup[(n['name'],frame['stamp_ns'])]['primary'])
pairs=read(pairs_dir/'pairs.json');pair_summary=read(pairs_dir/'summary.json')
add('six_posthoc_pairs_disclosed',len(pairs)==6 and all(p['has_pair'] for p in pairs) and pair_summary['pairs_available']==6)
add('robust_rule_hash',sha('configs/event_robust_pairs_v1.json')==pair_summary['config_sha256'])
for pair in pairs:
    name=pair['instance'];pre,post=select(pair['event'],[r for r in rows if r['instance']==name])
    add('pair_selection_replay:'+name,pre==pair['before'] and post==pair['after'])
    add('outside_both_clock_hypotheses:'+name,pre['stamp_ns']<=min(pair['event']['candidate_times_ns'].values())-.15e9 and post['stamp_ns']>=max(pair['event']['candidate_times_ns'].values())+.15e9)
    add('three_tolerances_same_selected_states:'+name,all(x['sensitivity'][t]['state']==x['primary']['state'] for x in [pre,post] for t in x['sensitivity']))
events=read(analysis/'event_checks.json')
add('two_record_end_censors_explicit',sum(e['event']['possible_record_end_censor'] for e in events)==2)
add('local0of6_preserved',summary['local_bracketed_instances']==0)
add('no_epoch_or_identity_truth_invented',not summary['label_epoch_verified'] and not summary['instance_identity_verified'] and not summary['visibility_ground_truth_created'])
native=read(results/'prediction_nodes.json');add('native_id_export_success',read(results/'prediction_export_result.json')['returncode']==0)
add('native_original_ids_unique',len({r['id_decimal'] for r in native})==len(native))
for n in native:add('native_category_bits:'+n['id_decimal'],int(n['id_decimal'])&0x00ffffffffffffff==n['category_index'])
for r in read(analysis/'association_identity_checks.json'):add('no_identity_claim:'+r['instance'],not r['identity_verified'])
for rel,h in read(analysis/'manifest.json')['source_sha256'].items():add('source_manifest:'+rel,sha(rel)==h)
for rel,h in read(analysis/'manifest.json')['artifacts_sha256'].items():add('artifact_manifest:'+rel,sha(analysis/rel)==h)
old=read('data/khronos_official_v1/acquisition.json');add('original_bag_unchanged',sha('data/khronos_official_v1/tesse_cd_apartment.bag')==old['sha256'])
oldmap=read('results/khronos_reproduction_v1/run_001/result.json');add('original_map_unchanged',sha('results/khronos_reproduction_v1/run_001/algorithm/final.4dmap')==oldmap['map_sha256'])
report={'passed':all(r['passed'] for r in checks),'checks':checks,'scope':'Input/array integrity,projection convention,selected replay,event/pair selection and honesty of uncertainty. Not independent accuracy or novelty validation.'}
args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
sources=[Path(__file__),Path('scripts/event_geometry.py'),Path('scripts/analyze_event_evidence.py'),Path('scripts/build_event_robust_pairs.py'),Path('scripts/extract_event_frames_wsl.py'),Path('tools/event_prediction_export/main.cpp'),Path('configs/event_evidence_v1.json'),Path('configs/event_gate_decision_v1.json'),Path('configs/event_robust_pairs_v1.json')]
manifest={'source_sha256':{str(p):sha(p) for p in sources},'artifact_sha256':{str(p.relative_to(results)):sha(p) for p in results.rglob('*') if p.is_file()},'data_acquisition_sha256':sha(base/'acquisition.json')}
args.output.with_name(args.output.stem+'_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'passed':report['passed'],'checks':len(checks),'failed':[r for r in checks if not r['passed']]}))
if not report['passed']:raise RuntimeError('Event evidence audit failed')
