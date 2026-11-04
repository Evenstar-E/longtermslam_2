from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    cfg=read('configs/material_cycle_bridge_v1.json');out=Path(cfg['output_root']);root=Path(cfg['public_root']);rows=[]
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h
    for pth in sorted(out.glob('pair_*/predictions.json')):
        p=read(pth);pair=p['pair'];a=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);b=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4);gt=b@np.linalg.inv(a);anchor=np.array(p['anchor']);target=gt[:3,:3]@anchor+gt[:3,3]
        old=read(Path(cfg['pool_root'])/pth.parent.name/'predictions.json')
        assert p['old_pool_size']==len(old['pool'])
        for oldr,newr in zip(old['pool'],p['rows']):assert oldr['T_current_from_history']==newr['T'] and oldr['scores']['old_nn']==newr['base']
        choices=dict(p['choices'],old_pool_pair_only=old['choices']['old_nn'])
        for method,i in choices.items():
            r=p['rows'][i];T=np.array(r['T']);e=float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-target));rot=float(np.degrees(np.arccos(np.clip((np.trace(T[:3,:3]@gt[:3,:3].T)-1)/2,-1,1))))
            rows.append(dict(pair=pth.parent.name,method=method,anchor_error_m=e,rotation_error_deg=rot,pose_index=i,source=r['source'],diagnosis=r['diagnosis'],bridge_support=p['bridge_support']))
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];e=np.array([r['anchor_error_m'] for r in rr]);summary.append(dict(method=method,qualified_1cm=int(sum(e<=.01)),qualified_10cm=int(sum(e<=.1)),median_mm=float(np.median(e)*1000),max_mm=float(max(e)*1000)))
    save(out/'scores.json',dict(summary=summary,rows=rows,scope=cfg['scope']));print(summary)
    print('HARD',[(r['pair'],r['method'],round(r['anchor_error_m']*1000,3)) for r in rows if r['pair'] in ['pair_07','pair_08']])


if __name__=='__main__':main()
