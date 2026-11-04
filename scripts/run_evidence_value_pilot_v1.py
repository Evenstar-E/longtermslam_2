"""Three explicit stages: prepare separated data, train/freeze, evaluate DEV."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from inspection_decisions.graph_dataset import read_graph
from inspection_decisions.value_pilot_v1 import encode_options, fit_value_model, predict_value, choose

CONFIG = Path('configs/evidence_value_pilot_v1.json')
SOURCES = [CONFIG,Path('inspection_decisions/value_pilot_v1.py'),Path(__file__),
           Path('research/evidence_value_pilot_contract_v1_2026-10-23.md')]
SPLITS = ['pipeline_train','pipeline_calibration','pipeline_development']


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]
def save(p, v):
    def default(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x))
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n',encoding='utf-8')


def option_name(option):
    return '_'.join(('h' if e=='history' else 'c')+f'{v:02d}' for e,v in option) if option else 'initial'


def prepare(cfg,out):
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    dataset = Path(cfg['dataset']); dcfg = read('configs/evidence_graph_dataset_v1.json')
    records = []; labels = {}; source_hashes = {str(p):digest(p) for p in SOURCES}
    for shard in sorted(dataset.glob('shard_*')):
        for name in ('index/states.jsonl','labels/state_outcomes.jsonl'):
            source_hashes[str(shard/name)] = digest(shard/name)
        records += lines(shard/'index/states.jsonl')
        labels.update({r['state_id']:r for r in lines(shard/'labels/state_outcomes.jsonl')})
    initial = sorted((r for r in records if not r['actions']),key=lambda r:r['scene'])
    X=[]; X_no=[]; coverage=[]; risks=[]; split_labels={s:[] for s in SPLITS}; family=[]; splits=[]
    for r in initial:
        assert digest(r['input_file'])==r['input_sha256']
        source_hashes[r['input_file']] = r['input_sha256']
        graph = read_graph(r['input_file'])
        e = encode_options(graph,dcfg['pairs']); no = encode_options(graph,dcfg['pairs'],False)
        X.append(e['X']); X_no.append(no['X']); coverage.append(e['coverage']); risks.append(e['risk'])
        if len(X)==1: options = e['options']
        assert e['options']==options
        row = [labels[f"{r['scene']:04d}_{option_name(o)}"] for o in options]
        split_labels[r['split']].append(dict(scene=r['scene'],family_id=r['family_id'],outcomes=row))
        family.append(r['family_id']);splits.append(r['split'])
    np.savez_compressed(out/'features.npz',X=X,X_no_cov=X_no,coverage=coverage,risk=risks)
    save(out/'input_index.json',dict(families=family,splits=splits,options=options,
                                    option_names=[option_name(o) for o in options]))
    for s in SPLITS: save(out/f'{s}_outcomes.json',split_labels[s])
    save(out/'preparation.json',dict(source_sha256=source_hashes,
        artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()},
        families=len(family),shape=np.asarray(X).shape,
        note='Preparation partitions existing labels only. Train process opens train/calibration files only.'))
    print('Prepared',np.asarray(X).shape,flush=True)


def outcome_arrays(rows):
    loss = np.array([[o['metric']['task_loss'] for o in r['outcomes']] for r in rows])
    return loss,loss[:,[0]]-loss


def verify_prep(out):
    p = read(out/'preparation.json')
    for path,sha in p['source_sha256'].items(): assert digest(path)==sha,(path,'source changed')
    for name,sha in p['artifact_sha256'].items(): assert digest(out/name)==sha,(name,'input changed')


def train(cfg,out):
    if (out/'frozen_selection.json').exists(): raise FileExistsError('Already frozen')
    verify_prep(out)
    inputs = np.load(out/'features.npz'); idx = read(out/'input_index.json'); splits = np.array(idx['splits'])
    tr = splits==SPLITS[0]; ca = splits==SPLITS[1]
    train_loss,train_gain = outcome_arrays(read(out/f'{SPLITS[0]}_outcomes.json'))
    cal_loss,cal_gain = outcome_arrays(read(out/f'{SPLITS[1]}_outcomes.json'))
    counts = np.array([len(o) for o in idx['options']]); singles = np.flatnonzero(counts==1).tolist(); pairs = np.flatnonzero(counts==2).tolist()
    singles_lookup = {tuple(idx['options'][i][0]):i for i in singles}
    members = [(singles_lookup[tuple(idx['options'][i][0])],singles_lookup[tuple(idx['options'][i][1])]) for i in pairs]
    predictions={}; selection={}; search=[]; timings={}
    for name in cfg['models']:
        t0=time.perf_counter(); X=inputs['X_no_cov' if name=='structured_no_cov_rbf' else 'X']
        grid = [(a,1.) for a in cfg['linear_alpha']] if name=='flat_linear' else [(a,b) for a in cfg['rbf_alpha'] for b in cfg['rbf_bandwidth']]
        best=None
        for alpha,bw in grid:
            model=fit_value_model(name,X[tr],train_gain,singles,pairs,members,alpha,bw)
            pred=predict_value(model,X[ca]); mse=float(np.mean((pred[:,1:]-cal_gain[:,1:])**2))
            search.append(dict(model=name,alpha=alpha,bandwidth=bw,calibration_mse=mse))
            if best is None or mse<best[0]: best=(mse,model,alpha,bw)
        mse,model,alpha,bw=best
        predictions[name]=predict_value(model,X)
        save(out/f'model_{name}.json',model)
        selection[name]=dict(alpha=alpha,bandwidth=bw,calibration_mse=mse)
        timings[name]=dict(search_and_all_split_inference_s=time.perf_counter()-t0,candidates=len(grid),
                           solves_per_candidate=2 if name.startswith('structured') else 1)
        print(name,selection[name],flush=True)
    coverage=inputs['coverage']; c_single=np.argmax(np.where(counts[None,:]==1,coverage,-np.inf),axis=1)
    c_bundle=np.argmax(coverage,axis=1)
    # All ties favor STOP/earlier action; risk is an estimate, not truth error rate.
    risks=inputs['risk']; price=cfg['primary_acquisition_price']; threshold_search=[]
    for threshold in cfg['risk_thresholds']:
        chosen=np.where(risks[ca]<=threshold,0,c_bundle[ca])
        objective=float(np.mean(cal_loss[np.arange(sum(ca)),chosen]+price*counts[chosen]))
        threshold_search.append(dict(threshold=threshold,calibration_objective=objective))
    threshold=min(threshold_search,key=lambda r:r['calibration_objective'])['threshold']
    baseline=dict(stop=np.zeros(len(splits),int),coverage_single=c_single,coverage_bundle=c_bundle,
                  coverage_risk_stop=np.where(risks<=threshold,0,c_bundle))
    np.savez_compressed(out/'frozen_predictions.npz',**predictions,**{'choice_'+k:v for k,v in baseline.items()})
    save(out/'calibration_search.json',dict(model_search=search,threshold_search=threshold_search))
    save(out/'frozen_selection.json',dict(selection=selection,risk_threshold=threshold,timings=timings,
        trained_families=np.array(idx['families'])[tr],calibration_families=np.array(idx['families'])[ca],
        dev_labels_opened=False,selection_input='only train/cal outcomes; all graph inputs allowed for prediction',
        artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.name.startswith(('model_','frozen_predictions','calibration_search'))},
        source_sha256={str(p):digest(p) for p in SOURCES}))
    print('Selection/predictions frozen BEFORE development outcome loading.',flush=True)


def summary(rows,chosen,counts,price):
    selected=[r['outcomes'][int(i)] for r,i in zip(rows,chosen)]
    result={k:sum(r['metric'][k] for r in selected) for k in
            ('changed','unchanged','qualified_tp','fp','bad_motion_reports','known','identity_wrong_references')}
    result['task_loss']=sum(r['metric']['task_loss'] for r in selected)
    result['mean_acquisitions']=float(np.mean(counts[chosen]));result['stops']=int(np.sum(chosen==0))
    result['mean_objective']=result['task_loss']/len(rows)+price*result['mean_acquisitions']
    for k in ('total_travel_m','payload_bytes','modeled_motion_sensor_transfer_s'):
        result['mean_'+k]=float(np.mean([r['cost'][k] for r in selected]))
    result['unknown']=8*len(rows)-result['known']
    return result


def evaluate(cfg,out):
    if (out/'analysis.json').exists(): raise FileExistsError('Already evaluated; never silently overwrite')
    verify_prep(out); frozen=read(out/'frozen_selection.json')
    for name,sha in frozen['artifact_sha256'].items(): assert digest(out/name)==sha
    for name,sha in frozen['source_sha256'].items(): assert digest(name)==sha
    idx=read(out/'input_index.json'); mask=np.array(idx['splits'])==SPLITS[2]
    data=read(out/f'{SPLITS[2]}_outcomes.json'); loss,gain=outcome_arrays(data)
    counts=np.array([len(o) for o in idx['options']]); pred=np.load(out/'frozen_predictions.npz')
    summaries={}; choices={}; diagnostics={}; decisions=[]
    for price in cfg['sensitivity_prices']:
        summaries[str(price)]={}; choices[str(price)]={}
        for name in cfg['models']+['stop','coverage_single','coverage_bundle','coverage_risk_stop']:
            choice=choose(pred[name][mask],counts,price) if name in cfg['models'] else pred['choice_'+name][mask]
            choices[str(price)][name]=choice
            summaries[str(price)][name]=summary(data,choice,counts,price)
            for r,c in zip(data,choice): decisions.append(dict(price=price,model=name,scene=r['scene'],
                   family_id=r['family_id'],option=idx['option_names'][c],outcome=r['outcomes'][c]))
    for name in cfg['models']:
        p=pred[name][mask]; rho=[]
        for y,z in zip(gain[:,1:],p[:,1:]):
            if np.ptp(y)>1e-12 and np.ptp(z)>1e-12: rho.append(float(spearmanr(y,z).statistic))
        pair=np.flatnonzero(counts==2); lookup={tuple(idx['options'][i][0]):i for i in np.flatnonzero(counts==1)}
        a=[lookup[tuple(idx['options'][i][0])] for i in pair]; b=[lookup[tuple(idx['options'][i][1])] for i in pair]
        true_interaction=gain[:,pair]-gain[:,a]-gain[:,b]
        pred_interaction=p[:,pair]-p[:,a]-p[:,b]
        diagnostics[name]=dict(mse=float(np.mean((p[:,1:]-gain[:,1:])**2)),
             mean_within_family_spearman=float(np.mean(rho)) if rho else None,rank_defined_families=len(rho),
             interaction_mse=float(np.mean((pred_interaction-true_interaction)**2)))
    price=cfg['primary_acquisition_price']; ch=choices[str(price)]; n=len(data); take=np.arange(n)
    rng=np.random.default_rng(cfg['bootstrap']['seed']); boots=rng.integers(n,size=(cfg['bootstrap']['replicates'],n))
    comparisons={}
    for control in cfg['primary_contrasts']:
        a=ch['structured_rbf']; b=ch[control]
        d=loss[take,a]+price*counts[a]-loss[take,b]-price*counts[b]
        dl=loss[take,a]-loss[take,b]
        comparisons[control]=dict(mean_objective_difference=float(d.mean()),
            objective_ci95=np.quantile(d[boots].mean(1),[.025,.975]),
            mean_loss_difference=float(dl.mean()),loss_ci95=np.quantile(dl[boots].mean(1),[.025,.975]),
            wins=int(np.sum(d< -1e-9)),ties=int(np.sum(np.abs(d)<=1e-9)),losses=int(np.sum(d>1e-9)),
            scene_differences=[dict(scene=r['scene'],objective_difference=x,loss_difference=y) for r,x,y in zip(data,d,dl)])
    oracle=np.argmin(loss+price*counts[None,:],axis=1)
    save(out/'decisions.json',decisions)
    save(out/'analysis.json',dict(scope='offline reuse of actually solved endpoints; zero new backend fits; not adaptive policy',
        families=n,options=len(counts),summary=summaries,diagnostics=diagnostics,contrasts=comparisons,
        hindsight_oracle_diagnostic_only=summary(data,oracle,counts,price),
        no_exclusions=True,selection=frozen['selection'],risk_threshold=frozen['risk_threshold'],
        caveats=['Only16DEVfamilies, one per cell; exploratory unadjusted bootstrap',
                 'Known old data audit; not previously untouched test',
                 'Unknown, wrong association, false alarms and wrong motion retained',
                 'No GNN / adaptive two-step rollout / exact compute-matched comparison'] ))
    print(json.dumps(summaries[str(price)],indent=2),flush=True)
    print('Contrasts',json.dumps(comparisons,default=lambda x:x.tolist())[:1800],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate']);args=parser.parse_args()
    cfg=read(CONFIG);out=Path(cfg['output'])
    {'prepare':prepare,'train':train,'evaluate':evaluate}[args.stage](cfg,out)


if __name__=='__main__':main()
