"""Run ID-blind same-input registration/association; seal predictions before scoring."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import open3d as o3d
from inspection_decisions.real_temporal_v1 import prepare,register,assign

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/real_temporal_v1');args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    root=Path('data/real_cross_session_v1/prepared_v2');cfgpath=Path('configs/real_cross_session_v1.json')
    cfg=json.loads(cfgpath.read_text(encoding='utf-8'));public=json.loads((root/'public.json').read_text(encoding='utf-8'))
    save(out/'config.json',cfg)
    save(out/'start_manifest.json',{'config_sha256':digest(cfgpath),'public_sha256':digest(root/'public.json'),
        'inference_sources':{p:digest(p) for p in ['inspection_decisions/real_temporal_v1.py',__file__]},
        'input_sha256':{str(p.relative_to(root)):digest(p) for p in (root/'public').rglob('*.npz')},
        'scope':cfg['scope'],'open3d':o3d.__version__})
    start=time.perf_counter();count=0
    with (out/'candidate_fits.jsonl').open('x',encoding='utf-8') as stream, (out/'predictions.jsonl').open('x',encoding='utf-8') as predictions:
        for scene in cfg['spaces']:
            current=[x for x in public[scene]['1'] if x['eligible']];history=[x for x in public[scene]['0'] if x['eligible']]
            clouds={}
            for capture,items in [('0',history),('1',current)]:
                for item in items:
                    with np.load(root/item['path']) as data:p,f=prepare(data['xyz'],cfg)
                    clouds[(capture,item['token'])]=(p,f)
            allfits={m:{} for m in cfg['methods']};costs={m:np.ones((len(current),len(history)))*1e6 for m in cfg['methods']}
            for i,c in enumerate(current):
                b,fb=clouds[('1',c['token'])]
                for j,h in enumerate(history):
                    if h['semantic_class']!=c['semantic_class']:continue
                    a,fa=clouds[('0',h['token'])]
                    for method in cfg['methods']:
                        work=out/'native'/scene/f'{c["token"]}_{h["token"]}'
                        try:T,stats=register(a,b,fa,fb,method,cfg,work)
                        except (RuntimeError,ValueError) as e:T=None;stats={'cost':1.,'error':str(e)}
                        cost=float(np.linalg.norm(np.asarray(a.points).mean(0)-np.asarray(b.points).mean(0))) if method=='centroid_hungarian' else stats['cost']
                        # Baseline centroid gate uses 1 metre scale, all others symmetric surface cost.
                        costs[method][i,j]=cost
                        anchor=np.asarray(a.points).mean(0)
                        displacement=None if T is None else (T[:3,:3]@anchor+T[:3,3]-anchor).tolist()
                        row=dict(scene=scene,method=method,current=c['token'],history=h['token'],cost=cost,
                            T_current_from_history=None if T is None else T.tolist(),anchor_history_m=anchor.tolist(),
                            displacement_at_anchor_m=displacement,stats=stats)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();allfits[method][(i,j)]=row;count+=1
                print(scene,c['token'],'candidates',count,flush=True)
                save(out/'monitor.json',{'scene':scene,'current':c['token'],'candidate_fits':count,'elapsed_s':time.perf_counter()-start})
            for method in cfg['methods']:
                for i,j in assign(costs[method],cfg).items():
                    row=dict(scene=scene,method=method,current=current[i]['token'],history=None if j is None else history[j]['token'],
                        fit=None if j is None else allfits[method][(i,j)])
                    predictions.write(json.dumps(row,allow_nan=False)+'\n');predictions.flush()
    save(out/'prediction_seal.json',{'completed':True,'candidate_fits':count,'elapsed_s':time.perf_counter()-start,
        'files_sha256':{p.name:digest(p) for p in [out/'predictions.jsonl',out/'candidate_fits.jsonl',out/'config.json',out/'start_manifest.json']},
        'evaluation_access':'No evaluation directory or persistent ID loaded by this runner or estimator'})
    print('SEALED',count,flush=True)

if __name__=='__main__':main()
