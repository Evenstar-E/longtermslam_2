from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    cfg=read('configs/visible_sampling_centroid_v1.json');out=Path(cfg['output_root']);root=Path(cfg['public_root']);pool=Path(cfg['pool_root']);rows=[];gap=[]
    for p,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/p)==h
    for field in ['sources','inputs']:
        for p,h in read(out/'manifest.json')[field].items():assert sha(p)==h
    for pth in sorted(out.glob('pair_*/predictions.json')):
        p=read(pth);old=read(pool/pth.parent.name/'predictions.json');pair=p['pair'];a=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);b=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4);gt=b@np.linalg.inv(a);anchor=np.array(old['anchor']);target=gt[:3,:3]@anchor+gt[:3,3]
        assert len(p['rows'])==len(old['rows'])
        for method,i in p['choices'].items():
            T=np.array(old['rows'][i]['T']);err=float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-target));rows.append(dict(pair=pth.parent.name,method=method,pose_index=i,anchor_error_m=err,source=old['rows'][i]['source'],diagnosis=p['rows'][i]['diagnosis']))
        gap.extend(abs(r['scores']['ordinary_rendered_pixels']-r['scores']['visible_centroid_plus_scatter']) for r in p['rows'])
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        e=np.array([r['anchor_error_m'] for r in rows if r['method']==method]);summary.append(dict(method=method,qualified_1cm=int(sum(e<=.01)),qualified_10cm=int(sum(e<=.1)),median_mm=float(np.median(e)*1000),max_mm=float(max(e)*1000)))
    save(out/'scores.json',dict(summary=summary,rows=rows,maximum_pixel_moment_score_difference=max(gap),scope=cfg['scope']));print(summary);print('EQUIVALENCE',max(gap));print('HARD',[(r['pair'],r['method'],round(r['anchor_error_m']*1000,3)) for r in rows if r['pair'] in ['pair_07','pair_08']])


if __name__=='__main__':main()
