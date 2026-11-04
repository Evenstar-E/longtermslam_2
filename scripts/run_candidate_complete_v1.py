"""Shared geometric proposals for hard-gate vs complete identity candidates."""
import argparse,time
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.rigid_identity_cached_v1 import CachedJointProblem
from inspection_decisions.rigid_identity_bounds_v1 import pose

class GateVariant(CachedJointProblem):
    def __init__(self,base,penalty):
        super().__init__(base.H,base.C,np.zeros((base.n,base.m)),base.cfg);self.base=base;self.penalty=penalty
    def _values(self,j,i,T):
        full,pair=self.base._values(j,i,T);return full+self.penalty[j,i],pair+self.penalty[j,i]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',required=True);args=ap.parse_args()
    cfgpath=Path('configs/candidate_complete_v1.json');cfg=read(cfgpath);assert args.scene in cfg['spaces'];root=Path(cfg['public_root']);public=read(root/'public.json')[args.scene]
    out=Path('results/candidate_complete_v1')/args.scene;out.mkdir(exist_ok=False,parents=True);basecfg=read(cfg['base_objective']);items={};sources=[Path(__file__),cfgpath,Path(cfg['base_objective']),Path('inspection_decisions/rigid_identity_cached_v1.py'),Path('inspection_decisions/rigid_identity_bounds_v1.py')]
    paths=[root/'public.json']
    for period in ['0','1']:
        items[period]=[]
        for e in public[period]:
            if not e['eligible']:continue
            path=root/e['path'];embedding=Path('results/livingscenes_native_masks_v1/embeddings')/args.scene/period/(e['token']+'.npz');paths.extend([path,embedding])
            with np.load(path) as z:v=dict(e,xyz=z['xyz'],rgb=z['rgb'])
            with np.load(embedding) as z:v['descriptor']=z['z_inv'].reshape(-1)
            items[period].append(v)
    save(out/'start_manifest.json',{'sources':{str(p):sha(p) for p in sources},'inputs':{str(p):sha(p) for p in paths},'GT_read':False,'cfg':cfg})
    H=items['0'];C=items['1'];hh=np.array([v['descriptor'] for v in H]);cc=np.array([v['descriptor'] for v in C]);hh/=np.linalg.norm(hh,axis=1,keepdims=True);cc/=np.linalg.norm(cc,axis=1,keepdims=True)
    base=CachedJointProblem(H,C,np.clip((1-cc@hh.T)/2,0,1),basecfg);pairposes={};paircost=np.empty((base.n,base.m));started=time.perf_counter()
    for i,h in enumerate(H):
        for j,c in enumerate(C):
            T=pose(np.zeros(3),base.anchor[i]+c['xyz'].mean(0)-h['xyz'].mean(0),base.anchor[i]);T=base.fit_pose(i,c['xyz'],T);pairposes[j,i]=T;paircost[j,i]=base.pair_cost(j,i,T)
        save(out/'monitor.json',{'phase':'all_pair_proposals','completed_archive_objects':i+1,'total':len(H),'elapsed_s':time.perf_counter()-started});print('PAIRS',args.scene,i+1,len(H),flush=True)
    save(out/'pair_proposals.json',{'history':[v['token'] for v in H],'current':[v['token'] for v in C],'poses':{f'{j},{i}':T.tolist() for (j,i),T in pairposes.items()},'costs':paircost.tolist()})
    result={}
    for gated in [True,False]:
        name='gated' if gated else 'complete';penalty=cfg['hard_gate_penalty']*np.array([[c['semantic_class']!=h['semantic_class'] for h in H] for c in C],float) if gated else np.zeros_like(paircost)
        p=GateVariant(base,penalty);cost=paircost+penalty;one=p.assign(cost,True);many=p.assign(cost)
        for key,a in [('one_to_one',one),('independent_many',many)]:result[name+'_'+key]={'assignment':a.tolist(),'per_current_pose':[None if i<0 else pairposes[j,i].tolist() for j,i in enumerate(a)]}
        rng=np.random.default_rng(cfg['seed']);best=None;starts=[]
        for restart in range(basecfg['initial_restarts']):
            a=many.copy() if restart==0 else one.copy() if restart==1 else np.array([int(rng.integers(-1,p.m)) for _ in range(p.n)])
            poses=[pose(np.zeros(3),np.clip(p.anchor[i],p.position_lo,p.position_hi),p.anchor[i]) for i in range(p.m)]
            for i in sorted(set(a)-{-1}):
                ix=np.flatnonzero(a==i);j=int(ix[np.argmin(cost[ix,i])]) if restart<2 else int(rng.choice(ix));poses[i]=pairposes[j,i].copy()
            trial=p.refine(a,poses,basecfg['alternating_iterations']);starts.append(trial)
            if best is None or trial['energy']<best['energy']:best=trial
        result[name+'_common_rigid']=dict(best,starts=starts,additional_icp_calls=p.fit_calls)
        print('VARIANT_DONE',args.scene,name,best['energy'],flush=True)
    save(out/'predictions.json',{'history':[v['token'] for v in H],'current':[v['token'] for v in C],'methods':result,'elapsed_s':time.perf_counter()-started,'common_pair_ICP_calls':base.fit_calls,'actual_point_queries':base.point_queries,'scope':cfg['scope']})
    save(out/'seal.json',{'sha256':{p.name:sha(p) for p in out.glob('*.json') if p.name!='seal.json'},'GT_read':False})

if __name__=='__main__':main()
