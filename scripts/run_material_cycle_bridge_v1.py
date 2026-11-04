from pathlib import Path
import time
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from inspection_decisions.ray_material_evidence_v1 import Frame,RayPair
from inspection_decisions.real_temporal_v1 import prepare,register,fit_stats
from inspection_decisions.material_cycle_bridge_v1 import BridgeModel


def main():
    cp=Path('configs/material_cycle_bridge_v1.json');cfg=read(cp)
    root=Path(cfg['public_root']);pool=Path(cfg['pool_root']);out=Path(cfg['output_root']);out.mkdir(exist_ok=False)
    for p,h in read(pool/'prediction_seal.json')['sha256'].items(): assert sha(pool/p)==h
    sources=[cp,Path(__file__),Path('inspection_decisions/material_cycle_bridge_v1.py'),Path('inspection_decisions/shared_contamination_transport_v1.py'),Path('inspection_decisions/contrastive_material_transport_v1.py'),Path('inspection_decisions/ray_material_evidence_v1.py'),Path('inspection_decisions/real_temporal_v1.py'),Path('configs/ycbineoat_motion_execution_v1.json')]
    inputs=[root/'public.json',pool/'effective_configuration.json']+list(pool.glob('pair_*/predictions.json'))+[p for p in (root/'public').rglob('*') if p.is_file()]
    save(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},inputs={str(p):sha(p) for p in inputs},cfg=cfg,GT_read=False))
    fc=read(pool/'effective_configuration.json');gc=read('configs/ycbineoat_motion_execution_v1.json')['geometry']
    frames={r['stem']:Frame(root,r['stem'],fc) for r in read(root/'public.json')['clouds']}
    prepared={s:prepare(f.xyz,gc) for s,f in frames.items()}
    pairs=[read(p) for p in sorted(pool.glob('pair_*/predictions.json'))];edges={};edge_records=[]
    # New edges ONLY between the three current views of each predefined episode.
    for start in range(0,len(pairs),3):
        stems=[p['pair']['current_stem'] for p in pairs[start:start+3]]
        for ia in range(3):
            for ib in range(ia+1,3):
                sa,sb=stems[ia],stems[ib];a,fa=prepared[sa];b,fb=prepared[sb];options=[]
                for method in cfg['bridge_methods']:
                    T,stats=register(a,b,fa,fb,method,gc,out/'unused')
                    cost=fit_stats(frames[sa].xyz,frames[sb].xyz,T,gc['registration_distance_m'])['cost']
                    options.append(dict(method=method,T=T.tolist(),cost=cost,stats=stats))
                chosen=min(options,key=lambda x:x['cost']);T=np.array(chosen['T'])
                edges[sa,sb]=T;edges[sb,sa]=np.linalg.inv(T)
                edge_records.append(dict(a=sa,b=sb,options=options,selected=chosen['method']))
        print('EDGES_DONE',start//3,flush=True)
    save(out/'bridge_edges.json',dict(edges=edge_records,GT_read=False))
    began=time.perf_counter()
    for idx,p in enumerate(pairs):
        pair=p['pair'];h=frames[pair['history_stem']];c=frames[pair['current_stem']];bridges=[];bridge_info=[]
        for j in range((idx//3)*3,(idx//3)*3+3):
            if j==idx: continue
            other=pairs[j];stem=other['pair']['current_stem'];hs=np.array(other['pool'][other['choices']['old_nn']]['T_current_from_history']);sc=edges[stem,pair['current_stem']]
            bridges.append((frames[stem],hs,sc));bridge_info.append(dict(stem=stem,history_pose_index=other['choices']['old_nn'],T_bridge_from_history=hs.tolist(),T_current_from_bridge=sc.tolist()))
        model=BridgeModel(h,c,bridges,cfg);ray=RayPair(h,c,p['anchor'],fc);rows=[]
        for entry in p['pool']:
            rows.append(dict(T=entry['T_current_from_history'],source=entry['source'],base=entry['scores']['old_nn']))
        for k,b in enumerate(model.bridges):
            T=b['path'];base=ray.score(T)[0]['old_nn'];rows.append(dict(T=T.tolist(),source='shared_bridge_path_'+str(k),base=base))
        for r in rows:
            r['scores'],r['diagnosis']=model.score(np.array(r['T']),r['base'])
        choices={method:min(range(len(rows)),key=lambda k:rows[k]['scores'][method]) for method in rows[0]['scores']}
        dest=out/f'pair_{idx:02d}';dest.mkdir();save(dest/'predictions.json',dict(pair=pair,anchor=p['anchor'],old_pool_size=len(p['pool']),choices=choices,bridges=bridge_info,bridge_support=[dict(hs_matches=b['hs_matches'],sc_matches=b['sc_matches'],composed_matches=int((b['q']>=0).sum())) for b in model.bridges],rows=rows,GT_read=False))
        print('PAIR_DONE',idx,'seconds',round(time.perf_counter()-began,2),flush=True)
    save(out/'prediction_seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*.json')},GT_read=False))


if __name__=='__main__': main()
