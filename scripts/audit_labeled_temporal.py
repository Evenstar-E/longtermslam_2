"""Verify lineage, explicit past queries, baseline isolation, counts and preserved geometry."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from scripts.temporal_metrics import deduplicate
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
base=Path('results/labeled_temporal_v1');history=base/'history_query_002';analysis=args.analysis
out=args.output
if out.exists():raise FileExistsError(out)
checks=[]
def add(name,value):checks.append({'check':name,'passed':bool(value)})
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def csvrows(path):return list(csv.DictReader(path.open(encoding='utf-8')))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
acq=read(Path('data/labeled_temporal_v1/acquisition.json'));add('six_official_truth_files_acquired',acq['complete'] and len(acq['records'])==6)
for r in acq['records']:add('raw_truth_hash:'+r['path'],sha(Path(r['path']))==r['sha256'])
add('acquisition_protocol_unchanged',sha(Path('configs/labeled_temporal_v1.json'))==acq['protocol_sha256'])
add('frozen_config_copy',Path('configs/labeled_temporal_v1.json').read_bytes()==Path('data/labeled_temporal_v1/frozen_config.json').read_bytes())
inventory=read(base/'label_inventory.json')
for scene,total,changed in [('apartment',76,6),('office',205,8)]:
    item=inventory['scenes'][scene]
    add('label_inventory:'+scene,item['node_count']==total and item['csv_changed_instances']==changed and item['changed_instances_found_by_name']==changed)
    for r in item['csv_instance_attributes']:
        attrs=r['dsg'];last=r['disappeared_at_ns'] or 2**64-1
        add('csv_to_dsg:'+scene+':'+r['object_symbol'],r['appeared_at_ns'] in attrs['first_observed_ns'] and last in attrs['last_observed_ns'])
add('native_metric_sources_unmodified',read(base/'native_metric_source_check.json')['passed'])
add('evaluation_package_built',read(base/'build_eval_003_result.json')['returncode']==0)
official=base/'apartment_official_001';add('official_evaluation_complete',read(official/'result.json')['complete'])
raw=csvrows(official/'results/static_objects.csv');unique,duplicates=deduplicate(raw)
add('duplicate_rows_disclosed',len(raw)==10 and len(unique)==4 and duplicates==6)
result=read(history/'result.json');add('history_driver_success',result['returncode']==0)
add('history_driver_source_hash',sha(Path('tools/temporal_query_audit/main.cpp'))==result['driver_sha256'])
add('query_protocol_unchanged',sha(Path('configs/temporal_query_audit_v1.json'))==result['config_sha256'])
schedule=csvrows(history/'schedule.csv')
add('ten_past_pairs',len(schedule)==10)
for r in schedule:add('past_query:'+r['RobotIndex']+':'+r['QueryIndex'],int(r['QueryTime'])<=int(r['RobotTime']))
stamps={r['RobotIndex']:r['RobotTime'] for r in schedule}
by_key={(r['Name'],r['Query']):r for r in unique}
for epoch in ['official_unshifted','static_shift_sensitivity']:
    full=history/epoch/'official_map';ablated=history/epoch/'intervals_disabled'
    for method in ['official_map','intervals_disabled']:
        rows=csvrows(history/epoch/method/'measurements.csv');uu,dd=deduplicate(rows)
        add('complete_unique_branch:'+epoch+':'+method,len(uu)==10 and dd==0)
        for r in rows:
            add('no_future_leak:'+epoch+':'+method+':'+r['Name']+':'+r['Query'],int(r['Query'])<=int(stamps[r['Name']]))
            # Count-level evaluator cross-check independent of report's helper.
            tp=int(r['AppearedTP'])+int(r['DisappearedTP'])
            fp=int(r['AppearedFP'])+int(r['DisappearedFP'])+int(r['AppearedHallucinatedP'])+int(r['DisappearedHallucinatedP'])
            fn=int(r['AppearedFN'])+int(r['DisappearedFN'])+int(r['AppearedMissedP'])+int(r['DisappearedMissedP'])
            got=next(x for x in read(analysis/'summary.json')['historical_queries'][epoch][method] if (x['name'],str(x['query_ns']))==(r['Name'],r['Query']))['change_inclusive']
            add('inclusive_counts:'+epoch+':'+method+':'+r['Name']+':'+r['Query'],(tp,fp,fn)==(got['tp'],got['fp'],got['fn']))
            if epoch=='official_unshifted' and method=='official_map' and r['Query']==stamps[r['Name']]:
                add('official_diagonal_replay:'+r['Name'],r==by_key[(r['Name'],r['Query'])])
    frows=csvrows(full/'vis/objects.csv');arows=csvrows(ablated/'vis/objects.csv')
    key=lambda r:(r['MapName'],r['QueryTime'],r['IsGT'],r['ID'])
    fa={key(r):r for r in frows};aa={key(r):r for r in arows}
    add('same_object_keys:'+epoch,set(fa)==set(aa))
    ignore={'Present','HasAppeared','HasDisappeared'}
    add('same_visualized_geometry_semantics:'+epoch,all({k:v for k,v in r.items() if k not in ignore}=={k:v for k,v in aa[q].items() if k not in ignore} for q,r in fa.items()))
    fassoc={tuple(r.items()) for r in csvrows(full/'vis/associations.csv')};aassoc={tuple(r.items()) for r in csvrows(ablated/'vis/associations.csv')}
    add('same_geometric_associations:'+epoch, fassoc==aassoc)
summary=read(analysis/'summary.json');add('unverified_epoch_disclosed',summary['label_epoch_verified'] is False)
add('no_asset_identity_accuracy_invented',summary['asset_identity_accuracy'] is None)
add('no_visibility_accuracy_invented',summary['visibility_conditioned_accuracy'] is None)
old=read(Path('results/khronos_reproduction_v1/run_001/result.json'))
add('original_map_unchanged',sha(Path('results/khronos_reproduction_v1/run_001/algorithm/final.4dmap'))==old['map_sha256'])
report={'passed':all(r['passed'] for r in checks),'checks':checks,'scope':'Integrity,counts,past-query schedule,official diagonal replay,and same-map ablation isolation. Not proof of real-world efficacy,truth epoch,visibility or novelty.'}
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
sources=[Path('configs/labeled_temporal_v1.json'),Path('configs/temporal_query_audit_v1.json'),Path('tools/temporal_query_audit/main.cpp'),Path('scripts/temporal_metrics.py'),Path('scripts/analyze_labeled_temporal.py'),Path(__file__)]
manifest={'source_sha256':{str(p):sha(p) for p in sources},'artifact_sha256':{str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()},'truth_acquisition_sha256':sha(Path('data/labeled_temporal_v1/acquisition.json'))}
manifest_path=out.with_name(out.stem+'_manifest.json')
if manifest_path.exists():raise FileExistsError(manifest_path)
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'passed':report['passed'],'checks':len(checks),'failed':[r for r in checks if not r['passed']]}))
if not report['passed']:raise RuntimeError('Labeled temporal audit failed')
