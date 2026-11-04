"""Score closed predictions separately; retain imperfect development-control labels."""
import collections,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=Path('results/roi_baseline_v1');base=Path('data/roi_baseline_v1');cfg=read(base/'config.json')
    truth={r['task_id']:r for r in read(base/'evaluator_truth.json')['truth']};tasks={r['task_id']:r for r in read(base/'inputs/tasks.json')['tasks']}
    rows=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()];lookup={(r['task_id'],r['method']):r for r in rows}
    manifest=read(out/'prediction_manifest.json');checks=0
    def check(v):
        nonlocal checks
        assert v;checks+=1
    check(sha(out/'predictions.jsonl')==manifest['predictions_sha256'])
    for p,digest in manifest['inputs'].items():check(sha(p)==digest)
    check((base/'config.json').read_bytes()==Path('configs/roi_baseline_v1.json').read_bytes())
    check(len(rows)==96);check(len(lookup)==96)
    for t in tasks.values():
        check(not any(k in t for k in ['instance','expected','event_kind','query_evidence']))
        for m in ['naive','occlusion_aware','khronos_raw','khronos_guarded']:
            check((lookup[t['task_id'],m]['result']=='data_unavailable')== (not t['available']))
    table=[]
    for method in ['naive','occlusion_aware','khronos_raw','khronos_guarded']:
        for condition in cfg['task_slots']:
            own=[r for r in rows if r['method']==method and truth[r['task_id']]['condition']==condition]
            available=[r for r in own if r['result']!='data_unavailable']
            table.append(dict(method=method,condition=condition,assigned=len(own),available=len(available),
                correct_under_frozen_labels=sum(r['result']==truth[r['task_id']]['expected'] for r in available) if not(method=='khronos_raw' and condition in ['occluded','out_of_view']) else None,
                counts=dict(collections.Counter(r['result'] for r in own))))
    own=[r for r in rows if r['method']=='occlusion_aware'];changed=[r for r in own if truth[r['task_id']]['condition']=='changed'];controls=[r for r in own if truth[r['task_id']]['condition'] in ['occluded','out_of_view'] and r['result']!='data_unavailable']
    criterion=dict(changed_correct=sum(r['result']=='unsupported' for r in changed),changed_assigned=6,changed_available=5,
        false_change_unchanged=sum(r['result']=='unsupported' for r in own if truth[r['task_id']]['condition']=='unchanged'),
        rejection_on_available_controls=sum(r['result']=='insufficient' for r in controls)/len(controls))
    criterion['frozen_pass']=criterion['changed_correct']>=5 and criterion['false_change_unchanged']==0 and criterion['rejection_on_available_controls']>=.8
    sensitivity=[json.loads(l) for l in (out/'sensitivity.jsonl').read_text().splitlines()]
    unstable=[]
    for tid in tasks:
        for m in ['naive','occlusion_aware']:
            labels={r['result'] for r in sensitivity if r['task_id']==tid and r['method']==m}
            if len(labels)>1:unstable.append([tid,m])
    result=dict(table=table,frozen_criterion=criterion,depth_sensitivity_changed_outputs=unstable,
      decision='Stop the current specialized algorithm novelty claim. Baseline failure is dominated by coverage/data/protocol limits; no demonstrated distinctive history or identity mechanism.',
      protocol_limits=['Projected query pixel count divided by anchor pixel count changes with range and viewpoint; not physical surface coverage.','Sparse GT vertex occlusion diagnostic is not majority occlusion of observed anchor ROI.','Retrospective selected development examples, privileged pose/classes/ROI; not independent confirmation.','Khronos raw state query is not capture-quality rejection; guarded version shares ordinary method gate.'])
    (out/'scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # All five available changed cases, with input rectangle and observed component.
    fig,axes=plt.subplots(5,2,figsize=(12,15))
    for i,t in enumerate([t for t in tasks.values() if t['available'] and truth[t['task_id']]['condition']=='changed']):
        for j,ch in enumerate(['anchor_channels','query_channels']):
            ax=axes[i,j];ax.imshow(Image.open(t[ch]['rgb']['path']));ax.axis('off')
            if j==0:
                mask=np.load(t['anchor_mask_path']);overlay=np.zeros((*mask.shape,4));overlay[mask]=[1,0,0,.45];ax.imshow(overlay)
                ax.set_title(f"{truth[t['task_id']]['instance']} anchor: observed class/depth mask")
            else:
                r=lookup[t['task_id'],'occlusion_aware'];ax.set_title(f"query: {r['result']} / {r['reason']}")
    fig.tight_layout();fig.savefig(out/'changed_cases.png',dpi=120);plt.close(fig)
    check(len(unstable)==0)
    (out/'audit.json').write_text(json.dumps(dict(checks_passed=checks,scope='Frozen inputs, hashes, missing slots and score accounting; not efficacy proof',scorer_sha256=sha(__file__)),indent=2),encoding='utf-8')
    print(json.dumps(dict(criterion=criterion,unstable=unstable,checks=checks),indent=2))

if __name__=='__main__':main()
