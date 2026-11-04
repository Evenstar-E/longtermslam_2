"""Score sealed full-scene engineering ablation; labels are evaluator-only."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.prepare_real_temporal_v1 import read_ply

def main():
    root=Path('data/livingscenes_native_masks_v1');public=read(root/'public.json');truth=read('data/real_cross_session_v1/prepared_v2/evaluation/identity.json');rows=[];diagnostics=[]
    for scene in ['scene_a','scene_c']:
        dest=Path('results/candidate_complete_v1')/scene
        for name,h in read(dest/'seal.json')['sha256'].items():assert sha(dest/name)==h
        pred=read(dest/'predictions.json');proposals=read(dest/'pair_proposals.json');cost=np.array(proposals['costs'])
        gt=read_ply(Path('data/real_cross_session_v1/rescan/raw')/scene/'gt_segmentation'/f'{scene}_1.ply');ids=gt['instance_idx'];histids={truth[scene]['0'][t] for t in pred['history']};masks={}
        current={c['token']:c for c in public[scene]['1']};history={h['token']:h for h in public[scene]['0']}
        for token in pred['current']:
            with np.load(root/current[token]['path']) as z:masks[token]=z['point_indices']
        ambiguity={};path=Path('data/real_cross_session_v1/rescan/metadata')/scene/'gt_segmentation'/f'{scene}_1.txt'
        if path.exists():
            for line in path.read_text().splitlines():
                a,b=line.split('|');ambiguity[int(a)]=[int(v) for v in b.split()]
        for method,result in pred['methods'].items():
            labels=np.full(len(gt),-1,int)
            for j,i in enumerate(result['assignment']):
                if i>=0:labels[masks[pred['current'][j]]]=truth[scene]['0'][pred['history'][i]]
            for identity in np.unique(ids):
                index=ids==identity;cls,count=np.unique(gt['class_idx'][index],return_counts=True)
                if int(cls[np.argmax(count)]) in [0,1,2,22] or index.sum()<128:continue
                identity=int(identity);allowed=ambiguity.get(identity,[identity]);persistent=bool(set(allowed)&histids)
                u,n=np.unique(labels[index],return_counts=True);k=np.argmax(n);dominant=int(u[k]);purity=float(n[k]/index.sum());fraction=float(np.isin(labels[index],allowed).mean());null=float((labels[index]<0).mean())
                status='correct' if persistent and fraction>=.8 else 'wrong_archive_identity' if persistent and dominant in histids and purity>=.8 else 'mixed_or_unresolved' if persistent else 'correct_new_rejection' if null>=.8 else 'new_assigned_to_archive'
                rows.append(dict(scene=scene,method=method,identity=identity,persistent=persistent,correct_fraction=fraction,null_fraction=null,dominant_prediction=dominant,purity=purity,status=status))
        for j,token in enumerate(pred['current']):
            u,n=np.unique(ids[masks[token]],return_counts=True);identity=int(u[n.argmax()]);allowed=ambiguity.get(identity,[identity]);truecols=[i for i,h in enumerate(pred['history']) if truth[scene]['0'][h] in allowed]
            gated=[i for i,h in enumerate(pred['history']) if current[token]['semantic_class']==history[h]['semantic_class']]
            order=np.argsort(cost[j]);besttrue=min(truecols,key=lambda i:cost[j,i]) if truecols else None
            diagnostics.append(dict(scene=scene,token=token,dominant_truth_id=identity,dominant_truth_fraction=float(n.max()/n.sum()),true_id_present_complete=bool(truecols),true_id_present_gated=bool(set(truecols)&set(gated)),best_true_cost=None if besttrue is None else float(cost[j,besttrue]),best_true_rank=None if besttrue is None else int(np.flatnonzero(order==besttrue)[0]+1),best_cost=float(cost[j,order[0]]),best_identity=truth[scene]['0'][pred['history'][order[0]]],cost_breakdown_scope='Same existing NN/shape/RGB/motion energy, complete shared ICP proposals, evaluator diagnosis only.'))
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];pp=[r for r in rr if r['persistent']]
        summary.append(dict(method=method,persistent=len(pp),correct=sum(r['status']=='correct' for r in pp),wrong=sum(r['status']=='wrong_archive_identity' for r in pp),mixed_or_unresolved=sum(r['status']=='mixed_or_unresolved' for r in pp),new=len(rr)-len(pp),correct_new_rejection=sum(r['status']=='correct_new_rejection' for r in rr)))
    save('results/candidate_complete_v1/scores.json',dict(summary=summary,rows=rows,candidate_diagnosis=diagnostics,scope='Engineering repair, native masks, two existing development spaces, no motion GT.'))
    print(summary);print('REMOVED_BY_GATE',[r for r in diagnostics if r['true_id_present_complete'] and not r['true_id_present_gated']])

if __name__=='__main__':main()
