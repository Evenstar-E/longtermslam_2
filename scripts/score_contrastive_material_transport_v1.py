from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    out=Path('results/contrastive_material_transport_v1');cfg=read('configs/contrastive_material_transport_v1.json');poolroot=Path(cfg['pool_root']);root=Path(cfg['public_root']);rows=[]
    for name,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/name)==h
    for field in ['sources','inputs']:
        for name,h in read(out/'manifest.json')[field].items():assert sha(name)==h
    for folder in sorted(out.glob('pair_*')):
        p=read(folder/'predictions.json');pool=read(poolroot/folder.name/'predictions.json');pair=p['pair'];h=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);c=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4);GT=c@np.linalg.inv(h);anchor=np.array(pool['anchor']);target=GT[:3,:3]@anchor+GT[:3,3]
        for m,i in p['choices'].items():
            T=np.array(pool['pool'][i]['T_current_from_history']);error=float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-target));rows.append(dict(pair=folder.name,method=m,pose_index=i,anchor_error_m=error,score=p['rows'][i]['scores'][m],diagnosis=p['rows'][i]['diagnosis']))
    summary=[]
    for m in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==m];e=np.array([r['anchor_error_m'] for r in rr]);summary.append(dict(method=m,within_1cm=int((e<=.01).sum()),within_10cm=int((e<=.1).sum()),median_mm=float(np.median(e)*1000),max_mm=float(e.max()*1000)))
    save(out/'scores.json',dict(rows=rows,summary=summary,scope=cfg['scope']));print(summary);print('pair08',[r for r in rows if r['pair']=='pair_08'])

if __name__=='__main__':main()
