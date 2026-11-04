"""GT-assisted postseal score decomposition, not new method predictions."""
from pathlib import Path
import pickle
import numpy as np
from scipy.special import betaln,logsumexp
from scripts.run_rescan_reversible_surface_v2 import read,save,sha
from inspection_decisions.history_conditioned_completion_evidence_v1 import event,probability,CounterCompiler


def terms(spec,compiler):
    logz=probability(spec,compiler);f=spec['free_count'];r=spec['unit_count'];n=spec['active_count']
    free=float(betaln(1,1+f));forced=float(betaln(1+r,1+f)-free);residual=logz-free-forced
    count=compiler.count(spec['clauses']);j=np.arange(n+1);ok=count>0
    logw=np.log(count[ok])+betaln(1+r+j[ok],1+f+n-j[ok]);w=np.exp(logw-logsumexp(logw))
    means=(1+r+j[ok])/(2+r+f+n)
    mean=float(w@means)
    return {'f':f,'r':r,'n':n,'free_logprob':free,'forced_given_free_logprob':forced,
            'residual_given_forced_logprob':residual,'logZ':logz,'posterior_mean_rho':mean}


def main():
    root=Path('results/history_conditioned_completion_evidence_v1');base=Path(read(root/'config.json')['input_root'])
    seal=read(root/'prediction_seal.json')
    for p,h in seal['sha256'].items():assert sha(root/p)==h
    allscore=read(root/'scores.json');compiler=CounterCompiler();rows=[];scenes=[]
    for run in read(root/'runs.json'):
        tag=run['scene']
        with (base/tag/'inference.pkl').open('rb') as f:p=pickle.load(f)
        with (base/tag/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        pred=read(root/tag/'predictions.json');selected={i:(k,l) for i,k,l in pred['beta_shared_exact']['selection']}
        score=np.load(root/tag/'scores.npz');scene_gap=0.
        for i,(h,bb) in enumerate(zip(p['history'],p['branches'])):
            true=tuple(next(k for k,b in enumerate(bb[t]) if b['slot']==truth[f'slot{t+1}'][i] and b['code']==truth[f'code{t+1}'][i]) for t in [0,1])
            k,l=selected[i];correct=bool(bb[0][k]['slot']==truth['slot1'][i] and bb[1][l]['slot']==truth['slot2'][i])
            hd=terms(event([h]),compiler);st=terms(event([h,bb[0][k],bb[1][l]]),compiler)
            tt=terms(event([h,bb[0][true[0]],bb[1][true[1]]]),compiler)
            gap=st['logZ']-tt['logZ'];scene_gap+=gap
            expected=score[f'beta_shared_exact_{i}'][k,l]-score[f'beta_shared_exact_{i}'][true]
            assert abs(gap-expected)<1e-8
            component={key:st[key]-tt[key] for key in ['free_logprob','forced_given_free_logprob','residual_given_forced_logprob']}
            assert abs(sum(component.values())-gap)<1e-8
            overlap_selected=score[f'beta_shared_exact_{i}'][k,l]-score[f'beta_independent_epoch_{i}'][k,l]
            overlap_true=score[f'beta_shared_exact_{i}'][true]-score[f'beta_independent_epoch_{i}'][true]
            rows.append({'scene':tag,'asset':i,'selected_both_ids_correct':correct,'selected':st,'true':tt,'history':hd,
                         'true_occupied_fraction':truth['occupancy'][i].bit_count()/4096,
                         'selected_minus_true_logscore':gap,'gap_terms':component,
                         'joint_correction_selected':float(overlap_selected),'joint_correction_true':float(overlap_true)})
        assert scene_gap>=-1e-7
        scenes.append({'scene':tag,'selected_global_minus_true_logscore':scene_gap})
    wrong=[r for r in rows if not r['selected_both_ids_correct']]
    wrong_better=[r for r in wrong if r['selected_minus_true_logscore']>1e-8]
    sets={}
    methods=read(root/'config.json')['methods']
    for method in methods:sets[method]={(r['scene'],r['asset']) for r in allscore if r['method']==method and r['joint_physical_5cm']}
    old={(r['scene'],r['asset']) for r in read(base/'scores.json') if r['method']=='full_shared' and r['joint_success_5cm']}
    def compare(a,b):return {'gained':len(a-b),'lost':len(b-a),'net':len(a)-len(b)}
    summary={'primary_vs_independent':compare(sets['beta_shared_exact'],sets['beta_independent_epoch']),
             'primary_vs_old_geometric':compare(sets['beta_shared_exact'],old),
             'wrong_identity_assets':len(wrong),'wrong_identity_with_local_score_above_true':len(wrong_better),
             'wrong_better_with_fewer_forced_occupied':sum(r['selected']['r']<r['true']['r'] for r in wrong_better),
             'wrong_better_positive_forced_term':sum(r['gap_terms']['forced_given_free_logprob']>0 for r in wrong_better),
             'median_history_posterior_rho':float(np.median([r['history']['posterior_mean_rho'] for r in rows])),
             'median_true_occupied_fraction':float(np.median([r['true_occupied_fraction'] for r in rows])),
             'median_selected_minus_true_occupied_count':float(np.median([r['selected']['r']-r['true']['r'] for r in wrong_better]))}
    save(root/'score_decomposition.json',{'GT_assisted':True,'changes_predictions':False,'summary':summary,'global_checks':scenes,'rows':rows})
    print(summary)

if __name__=='__main__':main()
