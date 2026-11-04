"""Four frozen development probes using predicted identity; full 400-step author pose refinement."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from inspection_decisions.livingscenes_cpu_runtime_v1 import load_solver

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--token',required=True);args=parser.parse_args()
    base=Path('results/livingscenes_real_v1');cfg=read(base/'config.json');probe=cfg['pose_probes'];token=args.token
    assert token in probe['current_tokens']
    seal=read(base/'prediction_seal.json')
    for name,h in seal['sha256'].items():assert sha(base/name)==h
    out=base/'pose_probes'/token;out.mkdir(exist_ok=False,parents=True)
    predicted=next(r for r in read(base/'predictions.json') if r['scene']==probe['scene'] and r['method']==probe['matched_by'] and r['current']==token)
    save(out/'frozen_input.json',{'prediction':predicted,'probe_contract':probe,'uses_evaluator_ID':False,'decoder_parameters_frozen':'They are not optimization variables in original code; skip their unused parameter gradients without changing gradient with respect to pose.'})
    if predicted['history'] is None:save(out/'results.json',{'unmatched':True});return
    root=Path(cfg['public_root']);public=read(root/'public.json')[probe['scene']]
    H=next(x for x in public['0'] if x['token']==predicted['history']);C=next(x for x in public['1'] if x['token']==token)
    solver=load_solver(Path.cwd(),cfg)
    import torch
    from inspection_decisions.real_temporal_v1 import fit_stats
    with np.load(root/H['path']) as z:a=torch.from_numpy(z['xyz'].astype(np.float32))[None]
    with np.load(root/C['path']) as z:b=torch.from_numpy(z['xyz'].astype(np.float32))[None]
    solver.model.requires_grad_(False)
    rows=[];started=time.perf_counter();counter={'calls':0}
    def hook(module,inputs,output):
        counter['calls']+=1
        if counter['calls']%50==0:
            record={'decoder_forward_calls':counter['calls'],'elapsed_s':time.perf_counter()-started,'current':token}
            save(out/'monitor.json',record);print('POSE_PROGRESS',record,flush=True)
    handle=solver.model.decoder.register_forward_hook(hook)
    for optim in probe['optim']:
        start=time.perf_counter();R,t=solver._solve_pairwise_registration(a,b,optim=optim)
        T=np.eye(4);T[:3,:3]=R.detach().numpy()[0];T[:3,3]=t.detach().numpy()[0,:,0]
        anchor=a.numpy()[0].mean(0)
        stats=fit_stats(a.numpy()[0],b.numpy()[0],T,.075)
        record={'current':token,'history':predicted['history'],'optim':optim,'n_steps':solver.cfg['registration']['n_steps'] if optim else 0,
            'T_current_from_history':T.tolist(),'displacement_at_historical_anchor_m':(T[:3,:3]@anchor+T[:3,3]-anchor).tolist(),
            'historical_anchor_m':anchor.tolist(),'surface_fit':stats,'elapsed_s':time.perf_counter()-start,'motion_error_to_independent_truth':None}
        rows.append(record);save(out/'results.partial.json',rows);print('POSE_DONE',token,optim,record['elapsed_s'],flush=True)
    handle.remove();save(out/'results.json',rows)
    save(out/'seal.json',{'results_sha256':sha(out/'results.json'),'input_sha256':sha(out/'frozen_input.json'),'source_sha256':sha(__file__),'elapsed_s':time.perf_counter()-started,'decoder_forward_calls':counter['calls']})

if __name__=='__main__':main()
