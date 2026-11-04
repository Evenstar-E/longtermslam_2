from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    out=Path('results/shared_contamination_transport_v1');cfg=read('configs/shared_contamination_transport_v1.json');root=Path(cfg['public_root']);pool=Path(cfg['pool_root']);rows=[];gaps=[]
    for name,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/name)==h
    for field in ['sources','inputs']:
        for name,h in read(out/'manifest.json')[field].items():assert sha(name)==h
    for folder in sorted(out.glob('pair_*')):
        p=read(folder/'predictions.json');old=read(pool/folder.name/'predictions.json');pair=p['pair'];a=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);b=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4);GT=b@np.linalg.inv(a);anchor=np.array(old['anchor']);target=GT[:3,:3]@anchor+GT[:3,3]
        for method,i in p['choices'].items():
            T=np.array(old['pool'][i]['T_current_from_history']);error=float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-target));rows.append(dict(pair=folder.name,method=method,accepted=p['accepted'][method],anchor_error_m=error,pose_index=i,score=p['rows'][i]['scores'][method],diagnosis=p['rows'][i]['diagnosis']))
        for r in p['rows']:
            for v in r['diagnosis']['bounds'].values():gaps.append(v['gap'])
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];e=np.array([r['anchor_error_m'] for r in rr]);summary.append(dict(method=method,accepted=sum(r['accepted'] for r in rr),qualified_1cm=sum(r['accepted'] and r['anchor_error_m']<=.01 for r in rr),qualified_10cm=sum(r['accepted'] and r['anchor_error_m']<=.1 for r in rr),median_mm=float(np.median(e)*1000),max_mm=float(e.max()*1000)))
    save(out/'scores.json',dict(summary=summary,rows=rows,maximum_discrete_bound_gap=float(max(gaps)),median_discrete_bound_gap=float(np.median(gaps)),scope=cfg['scope']));print(summary);print('bounds max/median',max(gaps),np.median(gaps));print('HARD',[(r['pair'],r['method'],r['anchor_error_m']) for r in rows if r['pair'] in ['pair_07','pair_08']])

if __name__=='__main__':main()
