"""GT-assisted noise/sampling intervention, never updates sealed predictions."""
from pathlib import Path
import pickle,time
import numpy as np
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import render
from inspection_decisions.first_return_material_surface_v1 import build_patches,bidirectional


def main():
    root=Path('results/first_return_material_surface_v1');seal=read(root/'prediction_seal.json')
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    cfg=read(root/'config.json');rows=[];stamp=time.perf_counter()
    for run in read(root/'runs.json'):
        dest=root/run['scene']
        with (dest/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (dest/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        geometry=geometry_for(run['seed'],cfg['objects'],run['geometry']);groups=np.arange(cfg['objects'])%4;slots=inp['T0'][:,:3,3]
        measured=[];clean=[];clean_cfg=dict(cfg,depth_noise_bound_m=0.)
        for t in range(3):
            ff=[];cc=[]
            for v,path in enumerate(sorted(dest.glob(f'frame_{t}_*.npz'))):
                with np.load(path) as z:frame={k:z[k] for k in ['rgb','depth','C']}
                ff.append(frame)
                # Counterfactual renderer uses evaluation poses/full shape ONLY here.
                poses=inp['T0'] if t==0 else truth['poses'][t-1]
                cf,_=render(geometry,poses,groups,frame['C'],clean_cfg,np.random.default_rng(run['seed']+10000*(t+1)+v))
                cc.append(cf)
            measured.append(ff);clean.append(cc)
        observed_patches=[build_patches(ff,slots,groups,cfg) for ff in measured]
        clean_patches=[build_patches(ff,slots,groups,cfg) for ff in clean]
        for i,bb in enumerate(inp['branches']):
            correct=[next(b for b in bb[t] if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1]]
            ids=[i,correct[0]['slot'],correct[1]['slot']];Ts=[inp['T0'][i],correct[0]['T'],correct[1]['T']]
            for variant,patches,frames in [('measured_source_clean_target',observed_patches,clean),('clean_source_measured_target',clean_patches,measured),('both_clean',clean_patches,clean)]:
                for surface in [False,True]:
                    trace=[]
                    for s,t in [(0,1),(0,2),(1,2)]:
                        trace.append(bidirectional(patches[s][ids[s]],patches[t][ids[t]],Ts[s],Ts[t],frames[s],frames[t],cfg,surface))
                    rows.append({'scene':run['scene'],'geometry':run['geometry'],'asset':i,'variant':variant,'surface':surface,
                                 'true_pair_retained':all(v['ok'] for v in trace),'checks':trace})
        print('DIAGNOSED',run['scene'],flush=True)
    summary={}
    for mode in cfg['geometry_modes']:
        summary[mode]={}
        for variant in ['measured_source_clean_target','clean_source_measured_target','both_clean']:
            summary[mode][variant]={}
            for surface in [False,True]:
                rr=[r for r in rows if r['geometry']==mode and r['variant']==variant and r['surface']==surface]
                gaps=[c['max_front_gap_m'] for r in rr for c in r['checks'] if not c['ok']]
                summary[mode][variant]['surface' if surface else 'point']={'n':len(rr),'true_pair_retained':sum(r['true_pair_retained'] for r in rr),
                    'conflict_gap_quantiles_m':np.quantile(gaps,[0,.5,.9,1]).tolist() if gaps else []}
    save(root/'noise_sampling_diagnosis.json',{'GT_assisted':True,'updates_predictions':False,'elapsed_s':time.perf_counter()-stamp,'summary':summary,'rows':rows,'script_sha256':sha(Path(__file__))})
    print(summary,flush=True)


if __name__=='__main__':main()
