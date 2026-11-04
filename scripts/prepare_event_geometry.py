"""Extract six original-ID meshes and event requests before reading any event image."""
import csv,decimal,hashlib,json
from pathlib import Path
import ijson,numpy as np
base=Path('data/event_evidence_v1');base.mkdir(exist_ok=False)
cfg=Path('configs/event_evidence_v1.json');(base/'frozen_config.json').write_bytes(cfg.read_bytes())
labels=list(csv.DictReader(Path('data/labeled_temporal_v1/ground_truth/apartment/gt_changes.csv').open()))
names={r['ObjectSymbol'].strip() for r in labels};nodes=[]
with Path('data/labeled_temporal_v1/ground_truth/apartment/gt_dsg_consolidated.json').open('rb') as f:
    for n in ijson.items(f,'nodes.item',use_float=False):
        a=n['attributes']
        if a.get('name') not in names or a.get('trajectory_timestamps'):continue
        number=int(a['name'][2:-1]);points=np.asarray(a['mesh']['points'],dtype=np.float64)
        mesh_path=base/f'instance_{number}_local_points.npy';np.save(mesh_path,points)
        nodes.append({'id':n['id'],'name':a['name'],'semantic_label':a['semantic_label'],'bbox':a['bounding_box'],
           'mesh_path':str(mesh_path),'mesh_points':len(points),'mesh_sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest()})
assert len(nodes)==6 and {r['name'] for r in nodes}==names
info=json.loads(Path('results/khronos_reproduction_v1/run_001/bag_info.json').read_text())
t0=round(info['start_s']*1e9);t1=round(info['end_s']*1e9);duration=t1-t0
events=[]
for label in labels:
    for kind,key in [('appearance','AppearedAt'),('disappearance','DisappearedAt')]:
        t=int(label[key])
        if not t:continue
        events.append({'instance':label['ObjectSymbol'].strip(),'kind':kind,'csv_time_ns':t,
          'possible_record_end_censor':abs(t-duration)<10000,
          'candidate_times_ns':{'unshifted':t,'plus_bag_start':t+t0}})
requests=[]
for k,t in enumerate(range(t0,t1+1,500000000)):requests.append({'id':f'grid_{k:03d}','time_ns':t,'kind':'uniform'})
requests.append({'id':'bag_end','time_ns':t1,'kind':'endpoint'})
for i,e in enumerate(events):
    for clock,t in e['candidate_times_ns'].items():
        for offset in [-1.,-.25,.25,1.]:
            target=t+round(offset*1e9)
            requests.append({'id':f'event_{i}_{clock}_{offset:+.2f}','time_ns':target,'kind':'event_neighbor','event_index':i,'clock':clock,'offset_s':offset,
              'within_recording':t0<=target<=t1})
record={'protocol_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'bag_start_ns':t0,'bag_end_ns':t1,
 'nodes':nodes,'events':events,'requests':requests,'scope':'All six static objects; label geometry only; images not yet inspected'}
(base/'plan.json').write_text(json.dumps(record,indent=2,default=lambda v:float(v) if isinstance(v,decimal.Decimal) else str(v)),encoding='utf-8')
print(json.dumps({'objects':len(nodes),'events':len(events),'requests':len(requests),'event_end_censors':sum(e['possible_record_end_censor'] for e in events)}))
