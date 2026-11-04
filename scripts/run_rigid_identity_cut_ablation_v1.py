"""Post-gap, label-blind bound ablation on identical pose leaves and feasible UB."""
import argparse,time
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.rigid_identity_bounds_v1 import Cell,JointProblem
from inspection_decisions.rigid_identity_cuts_v1 import RigidityCutsProblem

def load_problem(scene,cls,cfg):
    root=Path(cfg['public_root']);public=read(root/'public.json')[scene];items={}
    for period in ['0','1']:
        items[period]=[]
        for e in public[period]:
            if not e['eligible'] or e['semantic_class']!=cls:continue
            with np.load(root/e['path']) as z:v={'token':e['token'],'xyz':z['xyz'],'rgb':z['rgb']}
            with np.load(Path('results/livingscenes_native_masks_v1/embeddings')/scene/period/(e['token']+'.npz')) as z:v['descriptor']=z['z_inv'].reshape(-1)
            items[period].append(v)
    hh=np.array([v['descriptor'] for v in items['0']]);cc=np.array([v['descriptor'] for v in items['1']]);hh/=np.linalg.norm(hh,axis=1,keepdims=True);cc/=np.linalg.norm(cc,axis=1,keepdims=True)
    return items['0'],items['1'],np.clip((1-cc@hh.T)/2,0,1)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);args=parser.parse_args()
    cfg=read('configs/rigid_identity_bounds_v1.json');base=Path('results/rigid_identity_cached_v1')/args.scene
    seal=read(base/'prediction_seal.json')
    for name,h in seal['sha256'].items():assert sha(base/name)==h
    out=Path('results/rigid_identity_cuts_v1_completed')/args.scene;out.mkdir(exist_ok=False,parents=True)
    sources=[Path(__file__),Path('inspection_decisions/rigid_identity_cuts_v1.py'),Path('inspection_decisions/rigid_identity_bounds_v1.py')]
    save(out/'start_manifest.json',{'source_sha256':{str(p):sha(p) for p in sources},'base_prediction_seal_sha256':sha(base/'prediction_seal.json'),'scope':'Posthoc based on loose bounds, before current experiment GT scoring; no score-weight or ICP changes. Same fixed leaves, UB and observed point inputs. Integer assignments enforce rigidity cuts; base master did not need assignment integrality absent cuts.'})
    summaries=[]
    for folder in sorted(base.glob('class_*')):
        cls=int(folder.name.split('_')[1]);saved=read(folder/'bounded.json');H,C,shape=load_problem(args.scene,cls,cfg)
        p=RigidityCutsProblem(H,C,shape,cfg);p.cells={r['key']:Cell(r['owner'],np.array(r['lo']),np.array(r['hi']),r['key']) for r in saved['leaf_cells']}
        start=time.perf_counter();p.build_conflicts();cut=p.master();reference=JointProblem(H,C,shape,cfg)
        reference.cells=p.cells;reference.cell_cost=p.cell_cost # EXACT same distance-bound evaluations.
        raw=reference.master();raw={k:v for k,v in raw.items() if k not in ['assignment','selected']};UB=saved['best']['energy'];threshold=UB+cfg['near_optimal_energy_slack'];sets=[]
        costs=np.array([p.bound_cost(cell) for cell in p.cells.values()]).T;rowmin=np.minimum(costs.min(1),p.mass*cfg['null_cost']);rowminsum=rowmin.sum();cells=list(p.cells.values())
        for j in range(p.n):
            possible=[];forced=[];envelopes=[]
            for i in [-1]+list(range(p.m)):
                solve=p.master((j,i));forced.append({'identity':i,**solve})
                if solve['lower_bound']>threshold+1e-7:continue
                possible.append(i)
                if i<0:continue
                keep=[cell for k,cell in enumerate(cells) if cell.owner==i and costs[j,k]+rowminsum-rowmin[j]<=threshold+1e-7]
                # Independent per-row relaxation gives conservative cell
                # exclusion without pretending to optimize material motion.
                if keep:
                    low=np.min([cell.lo[3:] for cell in keep],axis=0)-p.anchor[i];high=np.max([cell.hi[3:] for cell in keep],axis=0)-p.anchor[i]
                    envelopes.append({'identity':i,'historical_anchor':p.anchor[i].tolist(),'retained_cells':len(keep),'map_anchor_delta_lo_m':low.tolist(),'map_anchor_delta_hi_m':high.tolist(),'enclosing_box_radius_m':float(np.linalg.norm(high-low)/2),'historical_world_material_displacement':None})
            sets.append({'current':C[j]['token'],'possible_identities':possible,'forced_bounds':forced,'conditional_map_anchor_envelopes':envelopes,'absolute_motion_unresolved':True})
        record={'class':cls,'history':[h['token'] for h in H],'current':[c['token'] for c in C],'base_master':raw,'cut_master':cut,'same_feasible_upper_bound':UB,
            'identity_sets':sets,'conflicts':[{'history':i,'current_a':j,'current_b':k,'minimum_point_separation_m':s,'historical_diameter_upper_m':d} for i,j,k,s,d in p.conflicts],
            'elapsed_s':time.perf_counter()-start,'additional_ICP_calls':0,'new_point_estimate':False,'claims':'Energy/domain conditional bounds only; physical truth can be excluded by model mismatch. No independent G or motion GT.'}
        save(out/(folder.name+'.json'),record);summaries.append({'class':cls,'base_lb':raw['lower_bound'],'cut_lb':cut['lower_bound'],'UB':UB,'conflicts':len(p.conflicts),'elapsed_s':record['elapsed_s']})
        save(out/'monitor.json',summaries[-1]);print('CUT_CLASS_DONE',args.scene,summaries[-1],flush=True)
    save(out/'summary.json',summaries);save(out/'prediction_seal.json',{'sha256':{p.name:sha(p) for p in out.glob('*.json') if p.name!='prediction_seal.json'},'complete':True,'GT_read':False})

if __name__=='__main__':main()
