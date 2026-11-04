"""Real all-object experiment, ID blind, configuration and payload hashes frozen first."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from inspection_decisions.rigid_identity_bounds_v1 import JointProblem,pose

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run_class(p,cfg,rng,out):
    start=time.perf_counter();pair_cost=np.empty((p.n,p.m));pair_pose={}
    base=[]
    for i,h in enumerate(p.H):
        base.append(pose(np.zeros(3),np.clip(p.anchor[i],p.position_lo,p.position_hi),p.anchor[i]))
        for j,c in enumerate(p.C):
            T=pose(np.zeros(3),p.anchor[i]+c['xyz'].mean(0)-h['xyz'].mean(0),p.anchor[i]);T=p.fit_pose(i,c['xyz'],T)
            pair_pose[j,i]=T;pair_cost[j,i]=p.pair_cost(j,i,T)
    # Pairwise controls have own per-pair poses; only common-pose controls below
    # are feasible for the joint objective. Keep this distinction explicit.
    one=p.assign(pair_cost,True);independent=p.assign(pair_cost)
    controls={}
    for name,a in [('pair_one_to_one',one),('pair_many_independent_pose',independent)]:
        controls[name]={'assignment':a.tolist(),'per_current_pose':[None if i<0 else pair_pose[j,i].tolist() for j,i in enumerate(a)],'energy_pairwise':float(sum(p.mass[j]*(cfg['null_cost'] if i<0 else pair_cost[j,i]) for j,i in enumerate(a))),'shared_pose_feasible':name=='pair_one_to_one'}
    best=None;starts=[]
    for restart in range(cfg['initial_restarts']):
        a=independent.copy() if restart==0 else one.copy() if restart==1 else np.array([int(rng.integers(-1,p.m)) for _ in range(p.n)])
        poses=[T.copy() for T in base]
        for i in sorted(set(a)-{-1}):
            members=np.flatnonzero(a==i)
            j=int(members[np.argmin(pair_cost[members,i])]) if restart<2 else int(rng.choice(members))
            poses[i]=pair_pose[j,i].copy()
        solution=p.refine(a,poses,cfg['alternating_iterations']);starts.append(solution)
        if best is None or solution['energy']<best['energy']:best=solution
    controls['ordinary_common_rigid']=best
    save(out/'controls.json',{'controls':controls,'starts':starts,'pair_costs':pair_cost.tolist(),'elapsed_s':time.perf_counter()-start,'fit_calls':p.fit_calls,'point_queries':p.point_queries})
    base_calls=p.fit_calls
    def monitor(record):
        save(out/'monitor.json',record)
        if record['step']%8==0:print('BOUND',out.name,record,flush=True)
    bounded=p.solve_bounds(best,monitor)
    save(out/'bounded.json',bounded)
    # Ordinary restart search with the SAME allowed number of additional ICP
    # proposals as bounded search, but different method-generated groupings.
    # Bound evaluation/MILP overhead is reported separately, not equal FLOPs.
    target_calls=p.fit_calls-base_calls;extra_calls=0;budget_start=p.fit_calls;ordinary_best=best;restart_log=[]
    while extra_calls<target_calls:
        a=np.array([int(rng.integers(-1,p.m)) for _ in range(p.n)]);poses=[T.copy() for T in base]
        for i in sorted(set(a)-{-1}):
            j=int(rng.choice(np.flatnonzero(a==i)));poses[i]=pair_pose[j,i].copy()
        before=p.fit_calls
        # At most one proposed shared ICP per active identity. A final restart
        # may overshoot by <= m-1; record exact calls, no equal-budget claim.
        trial=p.refine(a,poses,1);restart_log.append({'energy':trial['energy'],'fit_calls':p.fit_calls-before})
        if trial['energy']<ordinary_best['energy']:ordinary_best=trial
        extra_calls=p.fit_calls-budget_start
        if p.fit_calls==before:break
    controls['ordinary_extra_restarts']=ordinary_best
    save(out/'results.json',{'history':[h['token'] for h in p.H],'current':[c['token'] for c in p.C],'controls':controls,'candidate':bounded['best'],'identity_sets':bounded['identity_sets'],
        'bounded_extra_icp_calls':target_calls,'ordinary_extra_icp_calls':extra_calls,'ordinary_restart_log':restart_log,'elapsed_s':time.perf_counter()-start,
        'frame_scope':cfg['frame_scope'],'motion_accuracy':None})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);args=parser.parse_args()
    cfgpath=Path('configs/rigid_identity_bounds_v1.json');cfg=read(cfgpath);assert args.scene in cfg['spaces']
    out=Path('results/rigid_identity_bounds_v1')/args.scene;out.mkdir(exist_ok=False,parents=True)
    root=Path(cfg['public_root']);public=read(root/'public.json')[args.scene]
    sources=[Path(__file__),Path('inspection_decisions/rigid_identity_bounds_v1.py'),cfgpath]
    paths=[root/e['path'] for period in public.values() for e in period]
    embedding_root=Path('results/livingscenes_native_masks_v1/embeddings')/args.scene
    paths += list(embedding_root.rglob('*.npz'))+[root/'public.json']
    save(out/'start_manifest.json',{'sources_sha256':{str(p):sha(p) for p in sources},'inputs_sha256':{str(p):sha(p) for p in paths},'uses_evaluation_IDs':False})
    save(out/'config.json',cfg);rng=np.random.default_rng(cfg['seed']);counts=[]
    for cls in sorted({r['semantic_class'] for r in public['1'] if r['eligible']}):
        items={}
        for period in ['0','1']:
            items[period]=[]
            for entry in public[period]:
                if not entry['eligible'] or entry['semantic_class']!=cls:continue
                with np.load(root/entry['path']) as z:d={'token':entry['token'],'xyz':z['xyz'],'rgb':z['rgb']}
                with np.load(embedding_root/period/(entry['token']+'.npz')) as z:d['descriptor']=z['z_inv'].reshape(-1)
                items[period].append(d)
        H=items['0'];C=items['1'];dest=out/f'class_{cls}';dest.mkdir()
        if not H:save(dest/'results.json',{'history':[],'current':[x['token'] for x in C],'all_null':True});continue
        hh=np.array([x['descriptor'] for x in H]);cc=np.array([x['descriptor'] for x in C]);hh/=np.linalg.norm(hh,axis=1,keepdims=True);cc/=np.linalg.norm(cc,axis=1,keepdims=True)
        shape=np.clip((1-cc@hh.T)/2,0,1);problem=JointProblem(H,C,shape,cfg)
        print('CLASS_START',args.scene,cls,'history',len(H),'current',len(C),flush=True)
        run_class(problem,cfg,rng,dest);counts.append({'class':cls,'history':len(H),'current':len(C)})
        print('CLASS_DONE',args.scene,cls,flush=True)
    save(out/'prediction_seal.json',{'complete':True,'counts':counts,'sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*.json') if p.name!='prediction_seal.json'},'uses_current_GT_masks':False,'uses_persistent_IDs':False,'motion_accuracy':None})
    print('SCENE_SEALED',args.scene,flush=True)

if __name__=='__main__':main()
