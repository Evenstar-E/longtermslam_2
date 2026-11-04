"""Checkpointed observation-only inference plus separate true-model audit."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from scripts.map_evidence_scene_v4 import generate
from scripts.map_evidence_v4 import infer,Model


def true_model_audit(inp,truth,cfg):
    model=Model(inp,cfg);x=np.zeros(model.size)
    x[:model.d0]=np.array(truth['reference_boxes']).transpose(0,2,1).reshape(-1)
    d=np.array(truth['all_delta']);x[model.d0:model.u0]=d.reshape(-1)
    x[model.u0:model.z0]=np.r_[np.array(truth['realized_history_nuisance']).reshape(-1),truth['realized_query_nuisance']]
    x[model.z0:model.ad0]=np.max(np.abs(d),axis=1)>cfg['motion_axis_threshold_m']
    # Assets absent from current input are not estimated, so no actual delta
    # variables are assigned to them in this conditional audit.
    for j in range(model.n):
        if j not in model.q:x[model.d0+3*j:model.d0+3*j+3]=0;x[model.z0+j]=0
    x[model.ad0:model.au0]=np.abs(x[model.d0:model.u0]);x[model.au0:]=np.abs(x[model.u0:model.z0])
    values=model.constraint.A@x;v=np.maximum(model.constraint.lb-values,values-model.constraint.ub);worst=float(np.maximum(v,0).max())
    return dict(max_constraint_violation_m=worst,true_state_inside=bool(worst<=1e-6),identity_correct=bool(inp['assignment']==truth['assignment']))


def main():
    cfg=json.loads(Path('configs/map_evidence_v4.json').read_text());bcfg=json.loads(Path('configs/partial_surface_v1.json').read_text());out=Path('results/map_evidence_v4');out.mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    if (out/'metrics.jsonl').exists():raise RuntimeError('Preserve existing batch; use new version')
    (out/'frozen_config.json').write_text(json.dumps(cfg,indent=2));(out/'base_config.json').write_text(json.dumps(bcfg,indent=2));sources=['configs/map_evidence_v4.json','scripts/map_evidence_scene_v4.py','scripts/map_evidence_v4.py','scripts/run_map_evidence_v4.py','scripts/partial_surface_scene.py','scripts/joint_estimation.py','scripts/fragment_adapter_v3.py']
    manifest=dict(source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},cases=0,records=0,exclusions=[],input_sha256={},audits={});begin=time.perf_counter()
    factors=cfg['factors'];cells=[dict(zip(factors,v)) for v in itertools.product(*factors.values())]
    (out/'manifest_checkpoint.json').write_text(json.dumps(manifest,indent=2))
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(len(cells)*cfg['seeds_per_cell']):
            cell=sid//cfg['seeds_per_cell'];seed=cfg['seed_start']+sid;inp,truth,rgb,_=generate(cfg,bcfg,cells[cell],seed);source=out/'cases'/f'{sid:04d}_input.json';source.write_text(json.dumps(inp),encoding='utf-8');(out/'cases'/f'{sid:04d}_truth.json').write_text(json.dumps(truth),encoding='utf-8');manifest['input_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            p=infer(inp,cfg,seed)
            # Evaluation and true-set audit happen only after inference.
            manifest['audits'][str(sid)]=true_model_audit(inp,truth,cfg)
            tc=np.array(truth['changed']);correct=np.array(inp['assignment'])==truth['assignment']
            for method,labels in p['labels'].items():
                labels=np.array(labels);pc=labels==1;known=labels>=0;tp=int((pc&tc&correct).sum());fp=int((pc&(~tc|~correct)).sum());fn=int((tc&(~pc|~correct)).sum());wrong=int((known&((pc!=tc)|~correct)).sum())
                row=dict(scene=sid,seed=seed,cell=cell,condition=cells[cell],method=method,labels=labels.tolist(),truth_changed=tc.tolist(),assignment=inp['assignment'],truth_assignment=truth['assignment'],tp=tp,fp=fp,fn=fn,wrong_known=wrong,known=int(known.sum()),visible=len(tc),changed=int(tc.sum()),unchanged=int((~tc).sum()),seconds=p['seconds'],calls=p['calls'],base_status=p['base_status'],timeouts=p['timeouts'],sampling_accepted=p['sampling_accepted'])
                f.write(json.dumps(row)+'\n');manifest['records']+=1
            f.flush();manifest['cases']+=1;(out/'manifest_checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            if (sid+1)%8==0:print(f'{sid+1}/64 cases,{time.perf_counter()-begin:.1f}s',flush=True)
            if sid%8==0:
                from PIL import Image
                Image.fromarray(rgb).save(out/'cases'/f'{sid:04d}_query.png')
    manifest['seconds']=time.perf_counter()-begin;(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Complete',flush=True)


if __name__=='__main__':main()
