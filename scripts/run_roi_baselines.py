"""Prediction-only runner: never opens evaluator truth or GT object meshes."""
import hashlib,json,time
from pathlib import Path
import numpy as np,yaml
from PIL import Image
from scripts.roi_change_baseline import compare,query_map

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    started=time.perf_counter();base=Path('data/roi_baseline_v1');out=Path('results/roi_baseline_v1');out.mkdir(exist_ok=False)
    cfg=read(base/'config.json');inputs=base/'inputs/tasks.json';tasks=read(inputs)['tasks']
    mapping=Path('third_party/khronos_ros1_deps/hydra/config/label_remaps/tesse_cd_apartment.yaml')
    lut=np.zeros(65536,np.int32)
    for r in yaml.safe_load(mapping.read_text()):lut[r['sub_id']]=r['super_id']
    nodes_path=Path('results/event_evidence_v1/prediction_nodes.json');nodes=read(nodes_path)
    used={str(p):sha(p) for p in [inputs,base/'config.json',mapping,nodes_path,Path(__file__),Path('scripts/roi_change_baseline.py')]}
    rows=[];sensitivity=[]
    for t in tasks:
        tid=t['task_id']
        if not t['available']:
            for method in ['naive','occlusion_aware','khronos_raw','khronos_guarded']:
                rows.append(dict(task_id=tid,method=method,result='data_unavailable',reason=t['reason']))
            continue
        def load(ch):
            for item in ch.values():
                p=item['path'];used[p]=sha(p);assert used[p]==item['file_sha256']
            return np.load(ch['depth']['path']),lut[np.array(Image.open(ch['labels']['path']),dtype=np.uint16)]
        ad,_=load(t['anchor_channels']);qd,ql=load(t['query_channels']);mask=np.load(t['anchor_mask_path']);used[t['anchor_mask_path']]=sha(t['anchor_mask_path'])
        K=np.array(t['K']);A=np.array(t['T_world_anchor']);Q=np.array(t['T_world_query']);T=np.linalg.inv(Q)@A
        predictions={}
        for method in ['naive','occlusion_aware']:
            predictions[method]=compare(ad,qd,mask,K,T,ql,t['semantic_label'],cfg['comparison'],method)
            for tol in cfg['comparison']['sensitivity_tolerances_m']:
                c={**cfg['comparison'],'depth_tolerance_m':tol}
                sensitivity.append(dict(task_id=tid,method=method,tolerance_m=tol,**compare(ad,qd,mask,K,T,ql,t['semantic_label'],c,method)))
        y,x=np.where(mask);z=ad[y,x];points=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])@A[:3,:3].T+A[:3,3]
        raw=query_map(nodes,points,t['semantic_label'],t['anchor_stamp_ns'],t['query_stamp_ns'],cfg['map_adapter']);predictions['khronos_raw']=raw
        predictions['khronos_guarded']=dict(result='insufficient',reason='shared_evidence_gate',ordinary_reason=predictions['occlusion_aware']['reason']) if predictions['occlusion_aware']['result']=='insufficient' else {**raw,'guard':'passed_shared_evidence_gate'}
        rows.extend(dict(task_id=tid,method=m,**p) for m,p in predictions.items())
    for name,data in [('predictions.jsonl',rows),('sensitivity.jsonl',sensitivity)]:
        (out/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in data),encoding='utf-8')
    manifest=dict(inputs=used,predictions_sha256=sha(out/'predictions.jsonl'),sensitivity_sha256=sha(out/'sensitivity.jsonl'),seconds=time.perf_counter()-started,records=len(rows),sensitivity_records=len(sensitivity),truth_read=False)
    (out/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in manifest.items() if k!='inputs'}))

if __name__=='__main__':main()
