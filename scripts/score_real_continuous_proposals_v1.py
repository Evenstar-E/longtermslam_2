from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    cfg=read('configs/real_continuous_proposals_v1.json');out=Path(cfg['output_root']);root=Path(cfg['public_root'])
    for name,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/name)==h
    for field in ['sources','inputs']:
        for name,h in read(out/'manifest.json')[field].items():assert sha(name)==h
    rows=[];oracles=[];failures=[]
    for folder in sorted(out.glob('pair_*')):
        p=read(folder/'predictions.json');pair=p['pair'];h=np.loadtxt(root/'evaluation/poses'/(pair['history_stem']+'.txt')).reshape(4,4);c=np.loadtxt(root/'evaluation/poses'/(pair['current_stem']+'.txt')).reshape(4,4)
        GT=c@np.linalg.inv(h);anchor=np.array(p['anchor']);target=GT[:3,:3]@anchor+GT[:3,3];errors=[]
        for e in p['pool']:
            T=np.array(e['T_current_from_history']);errors.append(float(np.linalg.norm(T[:3,:3]@anchor+T[:3,3]-target)))
        for name,i in {**p['choices'],**{'endpoint__'+m:i for m,i in p['endpoints'].items()}}.items():
            rows.append(dict(pair=folder.name,method=name,accepted=i is not None,pose_index=i,anchor_error_m=errors[i] if i is not None else None,source=p['pool'][i]['source'] if i is not None else None))
        for group in ['old','point_to_plane','colored_icp','all']:
            eligible=[i for i,e in enumerate(p['pool']) if i<p['old_pool_size'] or group=='all' or e['source']==group]
            j=min(eligible,key=lambda i:errors[i]);oracles.append(dict(pair=folder.name,group=group,pose_index=j,anchor_error_m=errors[j],within_1cm_candidates=sum(errors[i]<=.01 for i in eligible)))
        failures.extend(dict(pair=folder.name,**t) for t in p['traces'] if t['error'] is not None)
    summary=[]
    for m in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==m];e=np.array([r['anchor_error_m'] for r in rr if r['accepted']]);summary.append(dict(method=m,total=len(rr),accepted=len(e),within_1cm=int((e<=.01).sum()),within_10cm=int((e<=.1).sum()),median_mm=float(np.median(e)*1000) if len(e) else None,max_mm=float(e.max()*1000) if len(e) else None))
    os=[dict(group=g,pairs=sum(r['group']==g for r in oracles),within_1cm=sum(r['group']==g and r['anchor_error_m']<=.01 for r in oracles)) for g in ['old','point_to_plane','colored_icp','all']]
    save(out/'scores.json',dict(summary=summary,rows=rows,oracles=oracles,oracle_summary=os,failures=failures,GT_read_only_after_seal=True,scope=cfg['scope']));print(summary,flush=True);print('ORACLE',os,'FAILURES',len(failures),flush=True)


if __name__=='__main__':main()
