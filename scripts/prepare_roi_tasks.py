"""Freeze development tasks/annotations before any method scores; evaluator truth stays separate."""
import collections,hashlib,json
from functools import lru_cache
from pathlib import Path
import numpy as np,yaml
from PIL import Image
from scripts.event_geometry import world_mesh
from scripts.roi_change_baseline import anchor_component

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
base=Path('data/roi_baseline_v1');base.mkdir(exist_ok=False);(base/'inputs').mkdir();(base/'masks').mkdir()
config=Path('configs/roi_baseline_v1.json');cfg=read(config);(base/'config.json').write_bytes(config.read_bytes())
plan=read('data/event_evidence_v1/plan.json');acq=read('data/event_evidence_v1/acquisition.json')
evidence=read('results/event_evidence_v1/analysis_001/frame_evidence.json');robust=read('results/event_evidence_v1/robust_pairs_001/pairs.json')
frames={r['stamp_ns']:r for r in acq['frames']};K=np.array(acq['camera']['K']).reshape(3,3);h,w=acq['camera']['height'],acq['camera']['width']
lut=np.zeros(65536,np.int32)
for r in yaml.safe_load(Path('third_party/khronos_ros1_deps/hydra/config/label_remaps/tesse_cd_apartment.yaml').read_text()):lut[r['sub_id']]=r['super_id']
@lru_cache(maxsize=16)
def images(stamp):
    f=frames[stamp];return np.load(f['channels']['depth']['path']),lut[np.array(Image.open(f['channels']['labels']['path']),dtype=np.uint16)]

def full_rectangle(node,stamp):
    dims=np.array(node['bbox']['dimensions']);corners=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])*dims
    world=world_mesh(corners,node['bbox']);T=np.array(frames[stamp]['T_world_camera']);p=(world-T[:3,3])@T[:3,:3]
    if np.any(p[:,2]<=.1):return None
    uv=p@K.T;uv=uv[:,:2]/uv[:,2,None];x0,y0=uv.min(axis=0);x1,y1=uv.max(axis=0);q=cfg['anchor_quality'];m=q['image_margin_px']
    if x0<m or y0<m or x1>w-1-m or y1>h-1-m:return None
    if (x1-x0)*(y1-y0)<q['minimum_box_area_px'] or min(x1-x0,y1-y0)<q['minimum_box_side_px']:return None
    return [int(np.floor(x0)),int(np.floor(y0)),int(np.ceil(x1)),int(np.ceil(y1))]

tasks=[];truth=[];annotations=[];Klist=K.tolist()
for index,node in enumerate(plan['nodes']):
    name=node['name'];pair=next(p for p in robust if p['instance']==name);event=pair['event'];lo=min(event['candidate_times_ns'].values());hi=max(event['candidate_times_ns'].values())
    censors=[e for e in plan['events'] if e['instance']==name and e['possible_record_end_censor']]
    censor_limit=min(min(e['candidate_times_ns'].values()) for e in censors) if censors else acq['frames'][-1]['stamp_ns']+1e9
    def consensus_present(stamp):return (stamp<=lo-.15e9 if event['kind']=='disappearance' else hi+.15e9<=stamp<censor_limit-.15e9)
    own=[r for r in evidence if r['instance']==name]
    candidates=sorted([r for r in own if consensus_present(r['stamp_ns']) and r['primary']['state']=='supported_surface'],key=lambda r:(-r['primary']['class_support'],r['stamp_ns']))
    anchor=None;rectangle=None;mask=None
    for r in candidates:
        box=full_rectangle(node,r['stamp_ns'])
        if box is None:continue
        d,s=images(r['stamp_ns']);candidate_mask=anchor_component(d,s,box,node['semantic_label'])
        if candidate_mask.sum()<cfg['anchor_quality']['minimum_component_pixels']:continue
        anchor=r;rectangle=box;mask=candidate_mask;break
    annotation={'instance':name,'id_decimal':str(node['id']),'anchor_found':anchor is not None,
       'annotation_source':'GT-assisted full anchor 3D box projected to rectangle;point mask derived only from observed depth and class,not GT mesh'}
    if anchor:
        stamp=anchor['stamp_ns'];path=base/'masks'/f'anchor_{index}.npy';np.save(path,mask)
        annotation.update(anchor_stamp_ns=stamp,rectangle=rectangle,component_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for condition in cfg['task_slots']:
        tid=f'task_{len(tasks):03d}';selected=None
        if anchor:
            others=[r for r in own if consensus_present(r['stamp_ns']) and abs(r['stamp_ns']-stamp)>=1e9]
            if condition=='changed':selected=pair['after'] if event['kind']=='disappearance' else pair['before']
            elif condition=='unchanged':
                options=[r for r in others if r['primary']['state']=='supported_surface']
                if options:selected=min(options,key=lambda r:(abs(r['stamp_ns']-stamp),r['stamp_ns']))
            elif condition=='occluded':
                options=[r for r in others if r['primary']['state']=='occluded']
                if options:selected=min(options,key=lambda r:(-r['primary']['occluded'],r['stamp_ns']))
            elif condition=='out_of_view':
                options=[r for r in others if r['primary']['projected_pixels']==0]
                if options:selected=min(options,key=lambda r:(abs(r['stamp_ns']-stamp),r['stamp_ns']))
        task={'task_id':tid,'available':selected is not None,'K':Klist,'semantic_label':node['semantic_label']}
        if selected:
            anchor_frame=frames[stamp];query_frame=frames[selected['stamp_ns']]
            task.update(anchor_channels=anchor_frame['channels'],query_channels=query_frame['channels'],anchor_mask_path=str(path),anchor_rectangle=rectangle,
               anchor_stamp_ns=stamp,query_stamp_ns=selected['stamp_ns'],T_world_anchor=anchor_frame['T_world_camera'],T_world_query=query_frame['T_world_camera'])
        else:task['reason']='no_quality_anchor' if anchor is None else 'no_fixed_control_frame'
        expected='unsupported' if condition=='changed' else ('consistent' if condition=='unchanged' else 'insufficient')
        truth.append({'task_id':tid,'instance':name,'id_decimal':str(node['id']),'condition':condition,'expected':expected,'event_kind':event['kind'],
          'annotation_oracle':'Class/depth-driven development selection;not independent confirmation','query_evidence':selected})
        tasks.append(task)
    annotations.append(annotation)
write(base/'inputs/tasks.json',{'tasks':tasks,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
 'contract':'No GT mesh,instance IDs,event direction or expected answers. Anchor may be later;both frames available at retrospective final recording time.'})
write(base/'evaluator_truth.json',{'truth':truth,'annotations':annotations,'scope':'Post-hoc development cases and sensor-quality controls;not unbiased accuracy benchmark'})
write(base/'plan.json',{'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'input_tasks_sha256':hashlib.sha256((base/'inputs/tasks.json').read_bytes()).hexdigest(),
 'truth_sha256':hashlib.sha256((base/'evaluator_truth.json').read_bytes()).hexdigest(),'annotations':annotations,
 'slots':len(tasks),'available':sum(t['available'] for t in tasks),'missing':[{'id':t['task_id'],'reason':t.get('reason')} for t in tasks if not t['available']]})
print(json.dumps({'slots':len(tasks),'available':sum(t['available'] for t in tasks),'anchors':annotations,'missing':[t['task_id'] for t in tasks if not t['available']]},indent=2))
