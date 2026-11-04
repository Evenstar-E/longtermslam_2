"""Offline GT-conditioned proposal coverage vs final native readout, not a new algorithm."""
import json
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply
from inspection_decisions.real_temporal_v1 import prepare,fit_stats,moved

def database(p):
    objects=[];poses={}
    for line in Path(p).read_text(encoding='utf-8').splitlines():
        v=line.split()
        if not v:continue
        if v[0]=='object':objects.append({'filename':v[1],'uid':int(v[2]),'class':int(v[3])})
        if v[0]=='pose':poses[(int(v[2]),int(v[1]))]={'object_index':int(v[3]),'score':float(v[4]),'T':np.array(v[5:],float).reshape(4,4)}
    return objects,poses

def main():
    base=Path('results/rescan_author_v1/inference_long_scene_a_v2');root=base/'scene_a'
    scores=json.loads((base/'scores.json').read_text(encoding='utf-8'))
    cfg=json.loads(Path('configs/real_cross_session_v1.json').read_text(encoding='utf-8'))
    objects,hposes=database(root/'history.rsdb');finalobjects,fposes=database(root/'prediction.rsdb')
    data=(root/'proposals/proposals.bin').read_bytes();n=int(np.frombuffer(data[:4],'<i4')[0]);counts=np.frombuffer(data[4:4+4*n],'<i4')
    assert n==len(objects)
    offset=4+4*n;proposals=[]
    for count in counts:
        values=np.frombuffer(data[offset:offset+int(count)*68],'<f4').reshape(-1,17).copy();offset+=int(count)*68;proposals.append(values)
    assert offset==len(data)
    raw=Path('data/real_cross_session_v1/rescan/raw/scene_a/gt_segmentation');h=read_ply(raw/'scene_a_0.ply');c=read_ply(raw/'scene_a_1.ply')
    rows=[]
    for record in scores['rows']:
        if not record['persistent'] or record['correct']:continue
        uid=record['instance'];j=next(i for i,x in enumerate(objects) if x['uid']==uid)
        aa=np.column_stack([h[k][h['instance_idx']==uid] for k in ['x','y','z']]);bb=np.column_stack([c[k][c['instance_idx']==uid] for k in ['x','y','z']])
        a,_=prepare(aa,cfg);b,_=prepare(bb,cfg);x=np.asarray(a.points);y=np.asarray(b.points)
        canonical=moved(x,np.linalg.inv(hposes[(0,uid)]['T']));evaluated=[]
        for k,values in enumerate(proposals[j]):
            T=values[:16].reshape(4,4,order='F').astype(float)
            assert np.allclose(T[3],[0,0,0,1],atol=1e-6)
            stats=fit_stats(canonical,y,T,.075)
            evaluated.append(dict(index=k,author_score=float(values[16]),**stats))
        best=min(evaluated,key=lambda r:r['cost']) if evaluated else None
        final=fposes.get((1,uid));finalfit=None
        if final is not None:
            obj=finalobjects[final['object_index']]
            if obj['uid']==uid:finalfit=fit_stats(canonical,y,final['T'],.075)
        rows.append(dict(record,retained_proposals=int(counts[j]),best_correct_identity_proposal_fit=best,
            proposals_with_surface_cost_below_035=sum(r['cost']<.35 for r in evaluated),
            final_true_id_template_fit_to_actual_instance=finalfit,
            final_template_uid=None if final is None else finalobjects[final['object_index']]['uid'],
            caveat='GT isolates the correct target AFTER native inference. Surface-fit candidate coverage is not motion ground truth, global-optimality proof, or original Rescan energy.'))
    out={'rows':rows,'pose_binary_schema':'int32 n_arrays, int32 counts[n], then 16 column-major float32 + float32 author score for each proposal',
        'source':'author save_pose_proposals/read_pose_proposals; matrix convention checked with homogeneous row and fixed historical model frame',
        'status':'POSTHOC diagnosis, no native output changed'}
    (base/'failure_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
