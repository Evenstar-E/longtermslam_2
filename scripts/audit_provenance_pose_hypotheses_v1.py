"""Post-seal audit of complete pools, branch effects, source lineage and readout."""
from pathlib import Path
import numpy as np
from scripts.run_provenance_pose_hypotheses_v1 import read,save,sha,load,pools_for
from inspection_decisions.provenance_pose_hypotheses_v1 import Problem
from inspection_decisions.reversible_surface_v2 import costs,inverse,transform


def main():
    root=Path('results/provenance_pose_hypotheses_v1');cfg=read(root/'config.json');checks=[];events=[];dense_errors=[]
    def check(value,name):
        if not bool(value):raise AssertionError(name)
        checks.append(name)
    seal=read(root/'prediction_seal.json');check(seal['complete'],'complete inference')
    for p,h in seal['sha256'].items():check(sha(root/p)==h,'sealed '+p)
    manifest=read(root/'manifest.json')
    for group in ['sources','inputs']:
        for p,h in manifest[group].items():check(sha(p)==h,group+' '+p)
    for p,h in read(root/'execution_metadata_inputs.json')['sha256'].items():check(sha(p)==h,'supplemental metadata '+p)
    for scene in cfg['scenes']:
        H,O,raw,initial,_=load(scene,cfg);pools,meta,_=pools_for(scene,H,initial);stored=np.load(root/scene/'pool.npz')
        for row in meta:
            t,i=row['period']-1,row['asset_index']
            check(row['total_count']==row['native_count']+6,f'{scene} full native pool {t} {i}')
            check(np.array_equal(stored[f'T_{t+1}_{i}'],pools[t][i]),f'{scene} pool origins {t} {i}')
        uid={h['uid']:i for i,h in enumerate(H)};ids=np.array([h['uid'] for h in H])
        native=[np.array([uid.get(int(u),-1) for u in r['native_uid'][o['raw_index']]],int) for r,o in zip(raw,O)]
        for method in cfg['methods']:
            path=root/scene/method;state=np.load(path/'state.npz');trace=read(path/'trace.json');run=read(path/'run.json')
            p=Problem(O,H,pools,native,cfg,method);p.pose=[list(state['poses'][0]),list(state['poses'][1])];p.z=[state['labels1'],state['labels2']];p.k=state['k']
            periods=2 if p.bidir else 1;check(len(trace)==cfg['sweeps']*len(H)*periods,f'{scene} {method} full sweeps')
            previous=run['initial_energy'];harmful=0;back_harmful=0;coupled_avoids=0
            for row in trace:
                t,i=row['period']-1,row['asset_index'];cc=row['candidates'];count=1 if method=='selected_forward' else len(pools[t][i])
                check([v['k'] for v in cc]==list(range(count)),f'{scene} {method} all candidates {row["sweep"]} {t} {i}')
                check(abs(previous-row['energy_before'])<1e-7,'energy trace continuity');previous=row['energy_after']
                check(abs(row['energy_after']-row['energy_before']-row['own_delta']-row['outgoing_delta'])<2e-7,'exact outgoing delta')
                metric='joint_delta' if p.coupled else 'own_delta';best=min(v[metric] for v in cc)
                if row['accepted']:
                    selected=cc[row['new_k']]
                    check(selected[metric]<=best+cfg['acceptance_tolerance']+1e-10,'selection minimizes declared block criterion')
                    check(selected[metric]<0,'strictly improving declared criterion')
                    harmful+=row['energy_after']>row['energy_before']+1e-7
                    back_harmful+=row['outgoing_delta']>1e-7
                else:check(best>=-cfg['acceptance_tolerance']-1e-10,'no improving block skipped')
                if p.coupled:
                    check(row['energy_after']<=row['energy_before']+2e-7,'coupled surrogate monotonicity')
                    ownbest=min(cc,key=lambda r:r['own_delta'])
                    coupled_avoids+=ownbest['own_delta']<-cfg['acceptance_tolerance'] and ownbest['joint_delta']>1e-7
            check(abs(previous-run['final_energy'])<1e-7,'trace final energy')
            for t in [0,1]:
                for i in range(len(H)):
                    check(np.array_equal(state['poses'][t,i],pools[t][i][state['k'][t,i]]),'selected pose belongs to common pool')
                inc=[p.surface(1-t,i) for i in range(len(H))] if t==1 or p.bidir else None
                sparse=np.column_stack([p.column(t,i,p.pose[t][i],None if inc is None else inc[i]) for i in range(len(H))])
                dense=costs(O[t],p.pose[t],H,inc,cfg)[0]
                error=float(np.max(np.abs(sparse-np.minimum(dense,cfg['null_cost']))));dense_errors.append(error)
                check(error<1e-6,'real all-voxel dense/sphere equality');p.C[t]=sparse
                pred=np.load(path/f'prediction{t+1}.npz');a=p.z[t][O[t]['inverse']];label=np.full(len(a),-1,int);ok=a>=0;label[ok]=ids[a[ok]]
                check(np.array_equal(label,pred['identity']),'primary readout preserves optimized state on every raw point')
                check(len(pred['readout_identity'])==len(raw[t]['xyz']),'secondary readout retains all raw points')
                check(O[t]['mass'].sum()==len(raw[t]['xyz']),'all-point mass retained')
                for i,h in enumerate(H):
                    check(np.allclose(state['anchors_world'][t,i],transform(h['anchor'][None],p.pose[t][i])[0]),'immutable historical anchor')
                if inc is not None:
                    disk=np.load(path/f'source_for{t+1}.npz');seen=[]
                    for i,v in enumerate(inc):
                        for key in ['xyz','rgb','source_voxel']:check(np.array_equal(v[key],disk[f'{key}_{i}']),'conditional source rebuilt from final labels and pose')
                        seen.extend(v['source_voxel'].tolist())
                    check(len(seen)==len(set(seen)),'source observation has at most one owner')
            check(abs(p.energy()-run['final_energy'])<2e-7,'reconstructed joint energy equals recorded energy')
            if not p.bidir:
                check(np.array_equal(p.z[0],native[0]),'forward source ownership unchanged')
                check(np.all(state['k'][0]==0),'forward source pose unchanged')
            events.append({'scene':scene,'method':method,'accepted_joint_energy_increases':int(harmful),
                'accepted_outgoing_harm':int(back_harmful),'coupled_blocks_where_own_best_would_increase_joint_energy':int(coupled_avoids),
                'initial_energy':run['initial_energy'],'final_energy':run['final_energy'],
                'source_changed_voxels':run['source_changed_voxels'],'pose_branches_changed':run['pose_branches_changed'],
                'point_queries':run['point_queries'],'elapsed_s':run['elapsed_s']})
            print('AUDITED',scene,method,flush=True)
    save(root/'audit.json',{'passed':True,'checks':len(checks),'real_dense_max_error':max(dense_errors),'source_sha':sha(__file__),
        'scope':'Pool coverage, state/source dependency, robust surrogate accounting and raw-point readout. No physical accuracy or novelty certificate.'})
    save(root/'branch_effects.json',{'rows':events,'scope':'Surrogate-energy branch events; identity effect requires separate ground-truth evaluation.'})
    print('PASS',len(checks),flush=True)


if __name__=='__main__':main()
