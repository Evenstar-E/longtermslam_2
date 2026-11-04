"""Post-seal causal boundary: feasible good endpoints versus selected bad ones."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import json,csv,time
from pathlib import Path
import numpy as np
from inspection_decisions.material_partial_track_transport_v7 import *
from scripts.run_material_partial_track_transport_v7 import setup
from scripts.run_textured_material_comparison_v1 import save

def main():
    cfg=json.loads(Path('configs/material_partial_track_transport_v7.json').read_text(encoding='utf-8'));root=Path(cfg['output']);assert json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'))['complete'];metrics=list(csv.DictReader((root/'task_metrics.csv').open(encoding='utf-8-sig')));ic=json.loads(Path(cfg['input_config']).read_text(encoding='utf-8'));truth=json.loads((Path(ic['dataset'])/ic['scene']/'supervision/scene.json').read_text(encoding='utf-8'));poses=np.array(truth['poses']);perm=np.array(truth['permutations']);out=[];audits=[];t0=time.perf_counter()
    for v in cfg['variants']:
        _,bc,ep,groups,views,banks,edges=setup(cfg,v)
        def ev_endpoint(slots,T):
            slots=list(map(int,slots));ev=[];rr=[];Ts=[T[0],T[1],T[1]@inverse(T[0])]
            for pe,(e,f) in enumerate(PAIRS):
                a,b=slots[e],slots[f];tr=PairTransport(banks[e][a],banks[f][b],edges[pe][a*24+b],views[e][a],views[f][b],a,b,cfg);ev.append(tr.evidence(Ts[pe]))
            values,state=joint_values(ev,cfg,True);selected=selected_edges(ev,state,'shared_transport',[banks[e][a] for e,a in enumerate(slots)]);details=[]
            for pe,((e,f),r) in enumerate(zip(PAIRS,selected)):
                a,b=slots[e],slots[f];ai=int(np.flatnonzero(perm[e]==a)[0]);bi=int(np.flatnonzero(perm[f]==b)[0]);x=apply(inverse(poses[e,ai]),r['x']);y=apply(inverse(poses[f,bi]),r['y']);correct=(np.linalg.norm(x-y,axis=1)<=.015)&(ai==bi);gain=ev[pe]['gain'][r['source'],r['target']];positive=np.maximum(gain,0);details.append(dict(pair=pe,selected=len(x),correct_15mm=int(correct.sum()),incorrect_positive_gain=float(positive[~correct].sum()),all_positive_gain=float(positive.sum()),potential=ev[pe]['potential'],free=ev[pe]['free'],pair_assignment_gain=ev[pe]['value']))
            return dict(energy=float(values[2]),Z=state['Z'],free=state['free'],memory_gain=float(state['memory_path'][-1]),shared_gain=float(state['shared_path'][-1]),three_epoch_tracks=int(((state['shared']>=0).sum(1)==3).sum()),pairs=details)
        with np.load(root/f'{v}_ordinary_union.npz') as f:base=dict(f)
        with np.load(root/f'{v}_shared_transport.npz') as f:ours=dict(f)
        bb={int(r['asset']):r for r in metrics if r['variant']==v and r['method']=='ordinary_union'};oo={int(r['asset']):r for r in metrics if r['variant']==v and r['method']=='shared_transport'}
        for i in range(24):
            if bb[i]['qualified']!='1' or oo[i]['qualified']=='1':continue
            a=ev_endpoint([i,*base['target_slots'][i]],base['transforms'][i]);b=ev_endpoint([i,*ours['target_slots'][i]],ours['transforms'][i]);out.append(dict(variant=v,asset=i,same_identity=bool(np.array_equal(base['target_slots'][i],ours['target_slots'][i])),base_error_m=float(bb[i]['worst_error_m']),selected_error_m=float(oo[i]['worst_error_m']),good_feasible=a,bad_selected=b,new_energy_change=b['energy']-a['energy']))
        # Verify safe bound pruning by evaluating one whole Cartesian triple
        # without pruning, with an independent loop and finite minimum.
        with np.load(root/f'{v}_pools.npz') as f:pools=dict(f)
        with np.load(root/f'{v}_cache.npz') as f:cache=dict(f)
        i=0;j,k=map(int,base['target_slots'][i]);h=np.flatnonzero(np.all(cache['triples']==[i,j,k],axis=1))[0];A=pools['transforms'][0,i,j,:pools['counts'][0,i,j]];B=pools['transforms'][1,i,k,:pools['counts'][1,i,k]];keys=[(0,i,j),(1,i,k),(2,j,k)];tr=[PairTransport(banks[e][a],banks[f][b],edges[pe][a*24+b],views[e][a],views[f][b],a,b,cfg) for (pe,a,b),(e,f) in zip(keys,PAIRS)];ea=[tr[0].evidence(T) for T in A];eb=[tr[1].evidence(T) for T in B];minimum=np.full(3,np.inf)
        for a,T in enumerate(A):
            for b,U in enumerate(B):minimum=np.minimum(minimum,joint_values([ea[a],eb[b],tr[2].evidence(U@inverse(T))],cfg))
        err=float(np.max(abs(minimum-cache['discrete_costs'][h,1:])));assert err<1e-9;audits.append(dict(variant=v,triple=[i,j,k],cartesian_pairs=len(A)*len(B),maximum_error=err))
    result=dict(loss_cases=out,pruning_audit=audits,elapsed_s=time.perf_counter()-t0,interpretation='Post-seal diagnostic only; known-good baseline endpoints are feasible in common pool. Same-ID lower wrong energy proves model ranking conflict at these endpoints, not global unobservability or proof of all causes. No retuning.');save(root/'mechanism_diagnosis.json',result)
    print(json.dumps(dict(losses=len(out),same_id_losses=sum(r['same_identity'] for r in out),same_id_lower_wrong_energy=sum(r['same_identity'] and r['new_energy_change']<0 for r in out),pruning_audit=audits),indent=2))

if __name__=='__main__':main()
