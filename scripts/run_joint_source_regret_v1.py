from pathlib import Path
import time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.joint_source_regret_v1 import Scores,joint_icp,selectors


def main():
    cp=Path('configs/joint_source_regret_v1.json');cfg=read(cp);root=Path(cfg['public_root']);pool=Path(cfg['pool_root']);second=Path(cfg['second_start_root']);out=Path(cfg['output_root']);out.mkdir(exist_ok=False)
    for folder in [pool,second]:
        for p,h in read(folder/'prediction_seal.json')['sha256'].items():assert sha(folder/p)==h
    fcpath=Path('results/ray_material_full_resolution_v1/effective_configuration.json');fc=read(fcpath)
    sources=[cp,Path(__file__),Path('inspection_decisions/joint_source_regret_v1.py'),Path('inspection_decisions/ray_material_evidence_v1.py'),Path('inspection_decisions/real_temporal_v1.py'),Path('scripts/run_rigid_identity_bounds_v1.py')]
    inputs=[root/'public.json',fcpath]+list(pool.glob('pair_*/predictions.json'))+list(second.glob('pair_*/predictions.json'))+[p for p in (root/'public').rglob('*') if p.is_file()]
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},cfg=cfg,GT_read=False))
    frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']};allpairs=[read(p) for p in sorted(pool.glob('pair_*/predictions.json'))];began=time.perf_counter()
    for episode in range(4):
        start=episode*3;pairs=allpairs[start:start+3];stems=[pairs[0]['pair']['history_stem']]+[p['pair']['current_stem'] for p in pairs];ff=[frames[s] for s in stems]
        initial=[np.eye(4)]+[np.linalg.inv(np.array(p['rows'][p['choices']['pair_only']]['T'])) for p in pairs]
        visible=[np.eye(4)]+[np.linalg.inv(np.array(p['rows'][read(second/f'pair_{start+j:02d}'/'predictions.json')['choices']['visible_centroid']]['T'])) for j,p in enumerate(pairs)]
        rows=[dict(Q=[q.tolist() for q in initial],source='initial_pair_choices')];conditional=[];optimizer=[]
        for view,p in enumerate(pairs,1):
            for k,r in enumerate(p['rows']):
                Q=[q.copy() for q in initial];Q[view]=np.linalg.inv(np.array(r['T']));conditional.append(len(rows));rows.append(dict(Q=[q.tolist() for q in Q],source='old_conditional',view=view,old_pose_index=k))
        for label,state in [('pair_start',initial),('visible_start',visible)]:
            rows.append(dict(Q=[q.tolist() for q in state],source=label))
            for color in [False,True]:
                steps=joint_icp(ff,state,cfg,color)
                for r in steps:optimizer.append(len(rows));rows.append(dict(r,source='joint_icp_'+label+('_color' if color else '_geometry')))
        score=Scores(ff,fc,cfg)
        for k,r in enumerate(rows):
            r['loss']=score.state([np.array(q) for q in r['Q']])
            if (k+1)%150==0:print('SCORE',episode,k+1,len(rows),'edges',score.calls,flush=True)
        choices,robust=selectors(rows);dest=out/f'episode_{episode:02d}';dest.mkdir()
        save(dest/'predictions.json',dict(stems=stems,pairs=[p['pair'] for p in pairs],anchors=[p['anchor'] for p in pairs],rows=rows,choices=dict(initial_pair_choices=0,**choices),conditional_indices=conditional,joint_optimizer_indices=optimizer,robust=robust,unique_pair_score_evaluations=score.calls,scope=cfg['scope'],GT_read=False))
        print('DONE',episode,'seconds',round(time.perf_counter()-began,1),flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))


if __name__=='__main__':main()
