"""Posthoc selection ablation, GT never changes sealed main predictions."""
from pathlib import Path
import argparse
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.joint_source_regret_v1 import selectors


def errors(Q,p,poses):
    result=[]
    for v in range(1,4):
        T=np.linalg.inv(np.array(Q[v]));gt=poses[v]@np.linalg.inv(poses[0]);a=np.array(p['anchors'][v-1]);result.append(float(np.linalg.norm(T[:3,:3]@a+T[:3,3]-gt[:3,:3]@a-gt[:3,3])*1000))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/joint_source_regret_v1');args=ap.parse_args()
    cfg=read('configs/joint_source_regret_v1.json');out=Path(args.output);root=Path(cfg['public_root']);results=[]
    for path,h in read(out/'prediction_seal.json')['sha256'].items():assert sha(out/path)==h
    for pth in sorted(out.glob('episode_*/predictions.json')):
        p=read(pth);poses=[np.loadtxt(root/'evaluation/poses'/(s+'.txt')).reshape(4,4) for s in p['stems']];rows=p['rows'];initial=np.array(rows[0]['Q']);rr={}
        # This removes optimizer outputs, not individual original hypotheses.
        keep=[i for i,r in enumerate(rows) if 'joint_icp_' not in r['source']];sub=[rows[i] for i in keep];subchoices,_=selectors(sub)
        for m,i in p['choices'].items():
            r=rows[i];Q=np.array(r['Q']);changed=[v for v in range(1,4) if not np.allclose(Q[v],initial[v],atol=1e-8,rtol=0)]
            rr[m]=dict(index=i,source=r['source'],changed_views=changed,errors_mm=errors(Q,p,poses),loss=r['loss'],regret=p['robust']['regret'][i])
            if m in subchoices:
                j=keep[subchoices[m]];rr[m]['without_joint_optimizer']=dict(index=j,source=rows[j]['source'],errors_mm=errors(rows[j]['Q'],p,poses),loss=rows[j]['loss'])
        # Evaluate all saved joint iterate errors for diagnostic pool coverage only.
        candidates=[(i,errors(rows[i]['Q'],p,poses)) for i in p['joint_optimizer_indices']]
        coverage=dict(joint_iterates=len(candidates),at_least_one_all_three_under_10mm=any(max(e)<=10 for i,e in candidates),minimum_max_anchor_error_mm=min(max(e) for i,e in candidates))
        results.append(dict(episode=pth.parent.name,choices=rr,reference_only_pool_coverage=coverage))
    save(out/'diagnosis.json',dict(episodes=results,scope='Posthoc no-joint-proposal ablation and oracle finite-pool coverage. No new fits/no changed predictions/no parameter tuning. Oracle minimum is not an algorithm result.'))
    for r in results:
        print(r['episode'],r['reference_only_pool_coverage'])
        for m,d in r['choices'].items():print(m,d['source'],d['changed_views'],[round(v,3) for v in d['errors_mm']])


if __name__=='__main__':main()
