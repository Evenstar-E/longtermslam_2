from pathlib import Path
import argparse
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/joint_source_regret_v1');args=ap.parse_args()
    cfg=read('configs/joint_source_regret_v1.json');out=Path(args.output);root=Path(cfg['public_root']);rows=[]
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h
    for ep,pth in enumerate(sorted(out.glob('episode_*/predictions.json'))):
        p=read(pth);poses=[np.loadtxt(root/'evaluation/poses'/(s+'.txt')).reshape(4,4) for s in p['stems']]
        for method,i in p['choices'].items():
            r=p['rows'][i]
            for v in range(1,4):
                T=np.linalg.inv(np.array(r['Q'][v]));gt=poses[v]@np.linalg.inv(poses[0]);anchor=np.array(p['anchors'][v-1]);error=float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-gt[:3,:3]@anchor-gt[:3,3]))
                rows.append(dict(pair=f'pair_{ep*3+v-1:02d}',episode=ep,method=method,anchor_error_m=error,state_index=i,source=r['source'],loss=r['loss']))
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        e=np.array([r['anchor_error_m'] for r in rows if r['method']==method]);summary.append(dict(method=method,qualified_1cm=int(sum(e<=.01)),qualified_10cm=int(sum(e<=.1)),median_mm=float(np.median(e)*1000),max_mm=float(max(e)*1000)))
    save(out/'scores.json',dict(summary=summary,rows=rows,scope=cfg['scope']));print(summary);print('HARD',[(r['pair'],r['method'],round(r['anchor_error_m']*1000,3),r['source']) for r in rows if r['episode']==2])


if __name__=='__main__':main()
