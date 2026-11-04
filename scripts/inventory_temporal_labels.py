"""Inventory official labels one node at a time; never supply truth to the mapper."""
import collections,csv,decimal,hashlib,json,time
from pathlib import Path
import ijson
base=Path('data/labeled_temporal_v1');out=Path('results/labeled_temporal_v1/label_inventory.json')
if out.exists():raise FileExistsError(out)
acq=json.loads((base/'acquisition.json').read_text(encoding='utf-8'))
assert acq['complete']
scenes={};start=time.monotonic()
for scene in ['apartment','office']:
    root=base/'ground_truth'/scene
    changes=[]
    with (root/'gt_changes.csv').open() as f:
        for row in csv.DictReader(f):
            changes.append({'object_symbol':row['ObjectSymbol'].strip(),'appeared_at_ns':int(row['AppearedAt']),
                            'disappeared_at_ns':int(row['DisappearedAt'])})
    nodes=[];types=collections.Counter();layers=collections.Counter()
    with (root/'gt_dsg_consolidated.json').open('rb') as f:
        # Keep uint64-max sentinels exact; use_float=True overflows signed parser integers.
        for node in ijson.items(f,'nodes.item',use_float=False):
            attrs=node['attributes'];types[attrs.get('type','unknown')]+=1;layers[str(node.get('layer'))]+=1
            mesh=attrs.get('mesh',{})
            record={'id':node.get('id'),'layer':node.get('layer'),'name':attrs.get('name'),'type':attrs.get('type'),
                'semantic_label':attrs.get('semantic_label'),'first_observed_ns':attrs.get('first_observed_ns'),
                'last_observed_ns':attrs.get('last_observed_ns'),
                'trajectory_stamps_count':len(attrs.get('trajectory_timestamps',[])),
                'mesh_point_count':len(mesh.get('points',[])),
                'position':attrs.get('position'),'bbox':attrs.get('bounding_box')}
            nodes.append(record)
    names={n['name']:n for n in nodes}
    scenes[scene]={'node_count':len(nodes),'types':dict(types),'layers':dict(layers),'nodes':nodes,
       'csv_changed_instances':len(changes),'changes_csv':changes,
       'changed_instances_found_by_name':sum(r['object_symbol'] in names for r in changes),
       'csv_instance_attributes':[{**row,'dsg':names.get(row['object_symbol'])} for row in changes],
       'visibility_labels_available':False,'defect_labels_available':False,
       'time_alignment':'CSV is dataset-relative; do not apply directly to absolute bag timestamps. DSG attributes and official builder need cross-check.'}
    print(scene,'nodes',len(nodes),'csv changes',len(changes),'found',scenes[scene]['changed_instances_found_by_name'],flush=True)
record={'scenes':scenes,'elapsed_s':time.monotonic()-start,'ijson_version':ijson.__version__,
        'protocol_sha256':hashlib.sha256(Path('configs/labeled_temporal_v1.json').read_bytes()).hexdigest(),
        'scope':'Inventory of two simulated layouts only; not task performance or independent real-scene validation.'}
out.write_text(json.dumps(record,indent=2,allow_nan=False,default=lambda x:float(x) if isinstance(x,decimal.Decimal) else str(x)),encoding='utf-8')
