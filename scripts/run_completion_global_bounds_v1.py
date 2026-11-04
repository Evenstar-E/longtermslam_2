from pathlib import Path
import pickle,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.completion_compatible_anchor_v1 import pair_displacements


def components(branches):
    n=len(branches);unseen=set(range(n));out=[]
    slots=[set(b['slot'] for b in bb[0])|set(b['slot'] for b in bb[1]) for bb in branches]
    while unseen:
        group={min(unseen)};unseen-=group
        while True:
            union=set().union(*(slots[i] for i in group));add={i for i in unseen if slots[i]&union}
            if not add:break
            group|=add;unseen-=add
        out.append(sorted(group))
    return out


def system(group,branches,masks,anchors):
    slots=sorted(set(b['slot'] for i in group for bb in branches[i] for b in bb));m=len(group)
    assert len(slots)==m
    slot_index={j:k for k,j in enumerate(slots)};rr=[];cc=[];variables=[];delta=[]
    for row,i in enumerate(group):
        bb=branches[i];d=pair_displacements(*bb,anchors[i])
        for k,l in zip(*np.nonzero(masks[f'{i}'])):
            j=len(variables);variables.append((i,int(k),int(l)));delta.append(d[k,l])
            rr.extend([row,m+slot_index[bb[0][k]['slot']],2*m+slot_index[bb[1][l]['slot']]]);cc.extend([j]*3)
    A=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(3*m,len(variables))).tocsc()
    return variables,np.array(delta),LinearConstraint(A,np.ones(3*m),np.ones(3*m))


def main():
    base=Path('results/completion_compatible_anchor_v1');seal=read(base/'prediction_seal.json')
    for path,v in seal['sha256'].items():assert sha(base/path)==v
    root=Path('results/completion_global_bounds_v1');root.mkdir(exist_ok=False);rows=[];start=time.perf_counter()
    for run in read(base/'runs.json'):
        folder=base/run['scene']
        with (folder/'inference.pkl').open('rb') as f:p=pickle.load(f)
        masks=np.load(folder/'feasible_masks.npz');records=read(folder/'records.json')
        groups=components(p['branches'])
        save(root/(run['scene']+'_components.json'),groups)
        for method in ['pair','full']:
            for group in groups:
                selected={str(i):masks[f'{method}_{i}'] for i in group}
                variables,delta,constraint=system(group,p['branches'],selected,p['anchors']);owners=np.array([v[0] for v in variables])
                for i in group:
                    local=records[i][('pairwise' if method=='pair' else 'full')+'_envelope'];lo=[];hi=[];details=[]
                    for axis in range(3):
                        for sign in [1,-1]:
                            cost=np.where(owners==i,sign*delta[:,axis],0.)
                            began=time.perf_counter()
                            result=milp(cost,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=constraint,
                                        options={'time_limit':3.,'mip_rel_gap':0.})
                            if result.status==0:value=sign*float(result.fun)
                            else:value=local['lower' if sign==1 else 'upper'][axis]
                            (lo if sign==1 else hi).append(value)
                            details.append({'axis':axis,'sign':sign,'status':int(result.status),'elapsed_s':time.perf_counter()-began,'used_local_fallback':result.status!=0})
                    radius=float(np.linalg.norm(np.array(hi)-lo)/2)
                    assert np.all(np.array(lo)>=np.array(local['lower'])-1e-7) and np.all(np.array(hi)<=np.array(local['upper'])+1e-7)
                    rows.append({'scene':run['scene'],'asset':i,'method':method,'lower':lo,'upper':hi,'radius_m':radius,
                                 'report':radius<=.05,'all_extrema_solved':all(d['status']==0 for d in details),'queries':details})
                print('GLOBAL',run['scene'],method,group,'variables',len(variables),flush=True)
    save(root/'predictions.json',rows)
    save(root/'run.json',{'elapsed_s':time.perf_counter()-start,'envelopes':len(rows),'objective_solves':6*len(rows),
                          'reports':sum(r['report'] for r in rows),'inference_GT_read':False,'DEV_previously_scored':True})
    save(root/'manifest.json',{'sources':{str(p):sha(p) for p in [Path(__file__),Path('research/completion_global_bounds_amendment_v1_2026-10-24.md')]},
                               'base_prediction_seal':sha(base/'prediction_seal.json')})
    save(root/'prediction_seal.json',{'complete':True,'sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print('SEALED GLOBAL',flush=True)


if __name__=='__main__':main()
