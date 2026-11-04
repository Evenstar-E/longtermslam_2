"""All22 legal initial actions, same held evidence; offline evaluator branches.

Scores computed for ALL actions before reading ANY branch image. Actual branch
images/afterfits never enter a choosing policy or a different action's forecast.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from comparative_evidence.interface import Episode,shortest_path
from comparative_evidence.geometry_modes_v2 import estimate_modes,mixture_particles
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_losses,common_terminal
from comparative_evidence.decisions_v3 import terminal_prediction
from comparative_evidence.event_likelihood_v7 import (held_clouds,feature_forecast,
    acquisition_scores,observation_features,log_likelihood)
from comparative_evidence.support_forecast_v5 import prepare_support,reciprocal_forecast
from comparative_evidence.selection_v6 import uniform_support
from comparative_evidence.selection_v3 import score_v2
from comparative_evidence.evaluation_v3 import evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=2);args=parser.parse_args()
    assert 0<=args.shard<args.shards
    cfgpath=Path('configs/comparative_selection_v7.json');cfg=json.loads(cfgpath.read_text())
    out=Path('results/comparative_event_branches_v7')/f'shard_{args.shard}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'branches').mkdir();(out/'frozen_config.json').write_bytes(cfgpath.read_bytes())
    paths=[cfgpath,Path(__file__)]+list(Path('comparative_evidence').glob('*.py'))
    manifest=dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scenes=0,branches=0,exclusions=[],shard=args.shard,shards=args.shards)
    begin=time.perf_counter()
    with (out/'metrics.jsonl').open('w',encoding='utf-8') as f:
        for sid in range(args.shard,cfg['scenes'],args.shards):
            root=Path(cfg['source_data'])/f'{sid:04d}';ep=Episode(root,*cfg['budget'])
            h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
            fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
            belief=mixture_particles(fit);losses,regret=common_losses(fit,belief)
            records=held_clouds(fit,h,q,meta);cache=prepare_support(fit,h,q,meta);unit=uniform_support(cache)
            initial=common_terminal(fit);initial_point=terminal_prediction(fit,'point')
            options=[(epoch,v) for epoch in ('history','current') for v in range(1,12)]
            forecasts={};frozen_scores={}
            for action in options:
                model=feature_forecast(action,fit,meta,records,belief[0]);constant=model.copy()
                constant['probabilities']=np.broadcast_to(model['probabilities'].mean(0),model['probabilities'].shape).copy()
                event_scores=acquisition_scores(model,losses,regret);constant_scores=acquisition_scores(constant,losses,regret)
                paired,_=reciprocal_forecast(action,fit,meta,cache);unweighted,_=reciprocal_forecast(action,fit,meta,unit)
                old_c=0. if paired is None else score_v2('candidate_regret',*paired,fit,belief,losses,regret)
                old_r=0. if unweighted is None else score_v2('task_risk_compatible_mc',*unweighted,fit,belief,losses,regret)
                _,audit=reciprocal_forecast(action,fit,meta,cache)
                travel=shortest_path(meta['edges'],0,action[1])[0] if action[0]=='current' else 0.
                raw=dict(events_candidate=event_scores['candidate'],events_risk=event_scores['risk'],
                    constant_events_candidate=constant_scores['candidate'],constant_events_risk=constant_scores['risk'],
                    weighted_candidate=old_c,uniform_risk=old_r,support_coverage=audit['missing_support_mass'])
                forecasts[action]=(model,constant)
                frozen_scores[action]=dict(raw=raw,normalized={k:float(v/(1+travel/7)) for k,v in raw.items()},
                    pi=model['probabilities'].mean(0).tolist(),audit=model['audit'])
            ep.finish()
            # ALL model scores now frozen. Branch actions below cannot modify them.
            for action in options:
                branch=Episode(root,*cfg['budget']);bh=[branch.act('retrieve_history',0)];bq=[branch.preview()]
                obs=branch.act('retrieve_history' if action[0]=='history' else 'observe_current',action[1])
                (bh if action[0]=='history' else bq).append(obs)
                mask,z=observation_features(obs,fit)
                after=attach_geometry_information(estimate_modes(bh,bq,meta),bh,bq,meta)
                prediction=common_terminal(after);point=terminal_prediction(after,'point');cost=branch.finish()
                model,constant=forecasts[action];ll=log_likelihood(model,mask,z);lc=log_likelihood(constant,mask,z)
                pi=model['probabilities'].mean(0)
                event_log=np.log(np.where(mask[None],model['probabilities'],1-model['probabilities'])).sum(1)
                event_only_nll=float(-logsumexp(event_log)+np.log(len(ll)))
                target=out/'branches'/f'{sid:04d}_{action[0]}_{action[1]:02d}.json'
                saved=dict(initial=initial,initial_point=initial_point,prediction=prediction,point_prediction=point,
                    cost=cost,state=branch.state(),scores=frozen_scores[action],observed_mask=mask.tolist(),observed_feature=z.tolist(),
                    diagnostics=dict(full_feature_nll=float(-logsumexp(ll)+np.log(len(ll))),
                        constant_event_feature_nll=float(-logsumexp(lc)+np.log(len(ll))),
                        event_only_nll=event_only_nll,event_brier=float(np.mean((pi-mask)**2))))
                target.write_text(json.dumps(saved,indent=2),encoding='utf-8')
                # Evaluation truth only after scores, actions and inference.
                row=dict(scene=sid,epoch=action[0],view=action[1],scores=frozen_scores[action],
                    initial=evaluate(root,initial),initial_point=evaluate(root,initial_point),
                    after=evaluate(root,prediction),after_point=evaluate(root,point),
                    observed_mask=mask.tolist(),diagnostics=saved['diagnostics'],
                    prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                f.write(json.dumps(row)+'\n');manifest['branches']+=1
            f.flush();manifest['scenes']+=1
            (out/'checkpoint.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'branch shard{args.shard} scene{sid:02d} {manifest["branches"]} branches {time.perf_counter()-begin:.1f}s',flush=True)
    manifest['seconds']=time.perf_counter()-begin
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()
