from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame
from inspection_decisions.joint_source_regret_v1 import Scores,selectors
from inspection_decisions.joint_source_balanced_v2 import joint_icp_balanced


def main():
    cp=Path('configs/joint_source_balanced_v2.json');cfg2=read(cp);cfg=read(cfg2['base_config']);base=Path(cfg2['base_results']);out=Path(cfg2['output_root']);out.mkdir(exist_ok=False);root=Path(cfg['public_root'])
    for p,h in read(base/'prediction_seal.json')['sha256'].items():assert sha(base/p)==h
    for field in ['sources','inputs']:
        for p,h in read(base/'manifest.json')[field].items():assert sha(p)==h
    fcpath=Path('results/ray_material_full_resolution_v1/effective_configuration.json');fc=read(fcpath)
    sources=[cp,Path(cfg2['base_config']),Path(__file__),Path('inspection_decisions/joint_source_balanced_v2.py'),Path('inspection_decisions/joint_source_regret_v1.py')]
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(base/'prediction_seal.json'):sha(base/'prediction_seal.json'),str(base/'manifest.json'):sha(base/'manifest.json'),**read(base/'manifest.json')['inputs']},cfg=dict(cfg,output_root=cfg2['output_root'],scope=cfg['scope']+' '+cfg2['scope']),GT_read=False))
    frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']}
    for pth in sorted(base.glob('episode_*/predictions.json')):
        p=read(pth);rows=p['rows'];old_count=len(rows);ff=[frames[s] for s in p['stems']];starts={label:next(r['Q'] for r in rows if r['source']==label) for label in ['pair_start','visible_start']}
        for label,Q in starts.items():
            for color in [False,True]:
                for r in joint_icp_balanced(ff,[np.array(q) for q in Q],cfg,color):p['joint_optimizer_indices'].append(len(rows));rows.append(dict(r,source='balanced_joint_icp_'+label+('_color' if color else '_geometry')))
        scores=Scores(ff,fc,cfg)
        for r in rows[old_count:]:r['loss']=scores.state([np.array(q) for q in r['Q']])
        choices,robust=selectors(rows);p.update(choices=dict(initial_pair_choices=0,**choices),robust=robust,base_row_count=old_count,new_pair_score_evaluations=scores.calls,scope=cfg2['scope'])
        dest=out/pth.parent.name;dest.mkdir();save(dest/'predictions.json',p);print('BALANCED_DONE',pth.parent.name,len(rows),flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))


if __name__=='__main__':main()
