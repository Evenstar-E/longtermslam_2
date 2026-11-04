"""Fixed-measure counterfactual on V7 endpoints; no fitting or threshold search."""
import json
from pathlib import Path
import numpy as np
from inspection_decisions.material_partial_track_transport_v7 import banks_from_features,PairTransport
from inspection_decisions.material_partial_track_fixed_measure_v8 import FixedMeasureTransport
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=json.loads(Path('configs/material_partial_track_fixed_measure_v8.json').read_text(encoding='utf-8'));ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));root=Path('results/material_partial_track_transport_v7');d=json.loads((root/'mechanism_diagnosis.json').read_text(encoding='utf-8'));out=[];checks=0
    for v in cfg['variants']:
        with np.load(Path(ic['native_root'])/'full24'/f'{v}_features.npz') as f:banks=banks_from_features(dict(f))
        with np.load(root/f'{v}_ordinary_union.npz') as f:base=dict(f)
        with np.load(root/f'{v}_shared_transport.npz') as f:ours=dict(f)
        for r in d['loss_cases']:
            if r['variant']!=v:continue
            i=r['asset'];values=[];norms=[]
            for endpoint,pred in [('good_feasible',base),('bad_selected',ours)]:
                slots=[i,*pred['target_slots'][i]];N=[len(banks[e][a]['xyz']) for e,a in enumerate(slots)];Z=N[0]*2+1.5*(N[1]+N[2])+cfg['support_regularizer'];q=r[endpoint];values.append(1-q['shared_gain']/Z+cfg['free_weight']*q['free']);norms.append(Z)
            if r['same_identity']:assert norms[0]==norms[1];checks+=1
            out.append(dict(variant=v,asset=i,same_identity=r['same_identity'],old_bad_minus_good=r['new_energy_change'],fixed_bad_minus_good=values[1]-values[0],good_Z=norms[0],bad_Z=norms[1],fixed_prefers_good=values[1]>values[0]))
    f=dict(C=np.eye(4),K=np.array([[10.,0,5],[0,10,5],[0,0,1.]]),depth=np.ones((11,11)),depth_valid=np.ones((11,11),bool),anonymous_slot=np.zeros((11,11),int));a=dict(node=np.arange(3),xyz=np.array([[0.,0,1],[.1,0,1],[0,.1,1]]));edge=dict(source=np.array([],int),target=np.array([],int),weight=np.array([]));p=FixedMeasureTransport(a,a,edge,[f],[f],0,0,cfg);q=PairTransport(a,a,edge,[f],[f],0,0,cfg)
    for d in [0,.2,-.2,5]:
        T=np.eye(4);T[2,3]=d;x=p.evidence(T);y=q.evidence(T);assert x['potential']==6 and np.array_equal(x['gain'],y['gain']) and x['value']==y['value'] and x['free']==y['free'];checks+=4
    result=dict(passed=True,checks=checks,endpoint_counterfactuals=out,same_id_original_inversions=sum(r['same_identity'] and r['old_bad_minus_good']<0 for r in out),same_id_inversions_repaired=sum(r['same_identity'] and r['old_bad_minus_good']<0 and r['fixed_prefers_good'] for r in out),interpretation='Same endpoints and identities isolate normalization only. Outcome-informed TRAIN diagnosis, not new predictive success.');save(Path('results/material_fixed_measure_check_v8.json'),result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
