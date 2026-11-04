"""Check saved evidence of the official example without requiring a running ROS master."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
base=Path('results/khronos_reproduction_v1');run=base/'run_001'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
out=args.output
if out.exists():raise FileExistsError(out)
checks=[]
def add(name,value):checks.append({'check':name,'passed':bool(value)})
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
result=read(run/'result.json');content=read(run/'map_content.json');info=read(run/'bag_info.json')
acq=read(Path('data/khronos_official_v1/acquisition.json'));build=read(base/'build_007_result.json')
add('build_returned_zero',build['returncode']==0)
log=(base/'build_007.log').read_bytes()
add('21_required_packages_compiled',b'All 21 packages succeeded!' in log)
add('run_report_complete',result['complete'])
add('clock_reached_recorded_end',result['reached_bag_end'] and abs(result['last_clock_s']-info['end_s'])<.2)
add('dataset_exited_cleanly','[dataset-2] process has finished cleanly' in (run/'launch.log').read_text(errors='replace',encoding='utf-8'))
add('service_and_launch_zero',result['finish_service_returncode']==0 and result['launch_returncode']==0)
add('clean_finish_marker','Experiment Finished Cleanly' in (run/'algorithm/experiment_log.txt').read_text(encoding='utf-8'))
bag=Path('data/khronos_official_v1/tesse_cd_apartment.bag');mapfile=run/'algorithm/final.4dmap'
add('bag_size',bag.stat().st_size==acq['bytes'])
add('bag_hash',sha(bag)==acq['sha256'])
add('map_size',mapfile.stat().st_size==result['map_bytes']>0)
add('map_hash',sha(mapfile)==result['map_sha256'])
add('multiple_map_states',content['time_steps']>1)
add('map_spans_input',content['earliest_ns']/1e9<=info['start_s']+.2 and content['latest_ns']/1e9>=info['end_s']-1)
add('nonempty_scene_graph',content['graph_nodes_without_mesh']>0 and content['graph_edges']>0)
add('nonempty_mesh',content['mesh_vertices']>0 and content['mesh_faces']>0)
add('parameters_saved', (run/'supervisor_rosparams.yaml').stat().st_size>0)
head=subprocess.check_output(['git','-C','third_party/Khronos_ROS1','rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C','third_party/Khronos_ROS1','status','--porcelain'],text=True).strip()
add('original_pinned_root_unmodified',head==build['root_revision'] and not status)
fetch=read(base/'source_fetch.json')
for entry in fetch['records']:
    if entry['status']!='fetched':continue
    name=entry['local-name'];path=Path('third_party/khronos_ros1_deps')/name
    add('source_commit:'+name,subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==entry['version'])
native=read(base/'environment.json');adapt=read(base/'message_dependency_adaptation.json')
add('declared_build_patch_recorded',native['native_package_xml_sha256']==adapt['after_sha256'])
timing={}
for rel in ['active_window/all_timing_raw.csv','frontend_timing_raw.csv','backend_timing_raw.csv']:
    rows=list(csv.DictReader((run/'algorithm/timing'/rel).open()))
    timing[rel]={'records':len(rows),'first_ns':int(rows[0]['timestamp(ns)']),'last_ns':int(rows[-1]['timestamp(ns)'])}
    add('nonempty_processing:'+rel,len(rows)>0 and timing[rel]['last_ns']>=9e10)
add('gt_inputs_disclosed','GT poses' in result['privileged_inputs'])
batch=read(Path('results/real_batch_v2/audit.json'));add('separate_batch_audit_passed',batch['passed'])
record={'passed':all(c['passed'] for c in checks),'checks':checks,'processing_records':timing,
 'scope':'Official old ROS1 synthetic GT-input build/playback/map-load verification, not paper metric reproduction or real inspection efficacy.',
 'limitations':['1745 input RGB-D frames; configured input decimation 0.099s; timing counts are not proof every input image was processed.',
 'TF_REPEATED_DATA warnings retained; upstream ROS parameter shell export failed, separately captured via argv-safe rosparam.',
 'Negative extract_object timing values observed; no latency/performance claim.',
 'Source archives were copied without .git; upstream recorded commit blank; pinned root and dependency commits verified separately.']}
out.write_text(json.dumps(record,indent=2),encoding='utf-8')
artifacts=[];unreadable=[]
for p in base.rglob('*'):
    try:
        if p.is_file():artifacts.append(p)
    except OSError as e:unreadable.append({'path':str(p),'error':str(e)})
manifest={'artifacts_sha256':{str(p.relative_to(base)):sha(p) for p in artifacts},
 'source_sha256':{str(p):sha(p) for p in [Path('scripts/run_khronos_example_wsl.py'),Path('scripts/build_khronos_wsl.py'),Path(__file__),Path('tools/khronos_map_audit/main.cpp'),Path('configs/khronos_headless_apartment_v1.launch')]},
 'unreadable_entries':unreadable,
 'note':'Post-run manifest. Supervisor later gained argv-safe parameter dump for future runs; run_001 parameter capture was separate. WSL latest log symlink may be unreadable on Windows; real log files retained.'}
(base/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'passed':record['passed'],'checks':len(checks),'processing_records':timing,'failed':[c for c in checks if not c['passed']]}))
if not record['passed']:raise RuntimeError('Khronos evidence audit failed')
