"""Postseal only: does RAW RGB-D contain shared finite-edge ray witnesses?

Uses true pose/box planes as an oracle to diagnose extraction losses; these
features must NEVER become estimator inputs or measured identity claims.
"""
from pathlib import Path
import pickle
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,save,sha
from scripts.first_return_surface_scene_v1 import geometry_for
from scripts.completion_catalog_scene_v1 import ray_box,inverse
from inspection_decisions.first_return_material_surface_v1 import labels
from inspection_decisions.material_supported_continuous_alignment_v1 import AXES
from scripts.diagnose_set_valued_material_edges_v1 import domain,rays


def main():
    root=Path('results/set_valued_material_edges_v1');cfg=read(root/'config.json');base=Path(cfg['base_root']);cache=Path(cfg['cached_model_root']);assert read(root/'prediction_seal.json')['complete']
    for name,h in read(root/'prediction_seal.json')['sha256'].items():assert sha(root/name)==h
    all_edges=[];tasks=[]
    for run in read(root/'runs.json'):
        tag=run['scene'];src=base/tag;geometry=geometry_for(run['seed'],24,run['geometry'])
        with (src/'inference.pkl').open('rb') as f:inp=pickle.load(f)
        with (src/'evaluation_only.pkl').open('rb') as f:truth=pickle.load(f)
        with (cache/tag/'features.pkl').open('rb') as f:planes=pickle.load(f)
        slots=inp['T0'][:,:3,3];groups=np.arange(24)%4;observed=[]
        for t in range(3):
            owner=np.arange(24) if t==0 else np.argsort(truth[f'slot{t}'])
            for v,path in enumerate(sorted(src.glob(f'frame_{t}_*.npz'))):
                with np.load(path) as z:frame={k:z[k].copy() for k in ['rgb','depth','C']}
                lab,_=labels(frame,slots,groups,cfg);h,w=lab.shape
                for slot in range(24):
                    i=int(owner[slot]);T=inp['T0'][i] if t==0 else truth['poses'][t-1,i];boxes=geometry[i];yy,xx=np.where(lab==slot);pixels=np.c_[xx,yy]
                    if not len(pixels):continue
                    o,ri=rays(frame,pixels,T,cfg);zi=np.min(np.array([ray_box(o,ri,lo,hi) for lo,hi in boxes]),axis=0);qi=o+zi[:,None]*ri
                    retained={f['axis'] for f in planes[t][slot] if f['view']==v};registry={}
                    for a,na in enumerate(AXES):
                        j=a%3;levels=np.unique(np.array(boxes)[:,:,j])
                        for d in levels:
                            select=np.abs(qi[:,j]-d)<1e-7;select&=domain(qi,na,boxes);select&=np.abs(zi-frame['depth'][yy,xx])<cfg['depth_noise_bound_m']+1e-8
                            if not select.any():continue
                            source=pixels[select];qsource=qi[select]
                            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                                other=source+[dx,dy];inside=(other[:,0]>=0)&(other[:,0]<w)&(other[:,1]>=0)&(other[:,1]<h);p=source[inside];q=other[inside];qq=qsource[inside];valid=lab[q[:,1],q[:,0]]!=slot;p=p[valid];q=q[valid];qq=qq[valid]
                                if not len(p):continue
                                _,ro=rays(frame,q,T,cfg);zo=np.divide(d-o[j],ro[:,j],out=np.full(len(q),np.nan),where=np.abs(ro[:,j])>1e-10);qo=o+zo[:,None]*ro
                                far=(zo>0)&(frame['depth'][q[:,1],q[:,0]]-zo>cfg['depth_noise_bound_m']);delta=qo-qq
                                for k,nk in enumerate(AXES):
                                    if k%3==j:continue
                                    for b in np.unique(np.array(boxes)[:,:,k%3]):
                                        alpha=np.divide(b-qq[:,k%3],delta[:,k%3],out=np.full(len(p),np.nan),where=np.abs(delta[:,k%3])>1e-10);point=qq+alpha[:,None]*delta
                                        hit=far&(alpha>=-1e-7)&(alpha<=1+1e-7)&(delta@nk>0)&domain(point-1e-5*nk,na,boxes)&~domain(point+1e-5*nk,na,boxes)
                                        if hit.any():registry.setdefault((a,float(d),k,float(b)),[]).extend(p[hit].tolist())
                    for key,pp in registry.items():
                        pp=np.unique(np.array(pp),axis=0);span=float(np.ptp(pp,axis=0).max());a,d,k,b=key;item={'scene':tag,'asset':i,'slot':slot,'epoch':t,'view':v,'side_axis':a,'side_offset':d,'edge_axis':k,'edge_offset':b,'source_pixel_witnesses':len(pp),'pixel_span':span,'dominant_plane_retained_this_axis':a in retained};observed.append(item);all_edges.append(item)
        for i in range(24):
            early=[e for e in observed if e['asset']==i and e['epoch']<2 and e['edge_axis']%3==2];late=[e for e in observed if e['asset']==i and e['epoch']==2 and e['edge_axis']%3==2];pairs=[]
            for a in early:
                for b in late:
                    if a['side_axis']==b['side_axis'] and a['edge_axis']==b['edge_axis'] and abs(a['side_offset']-b['side_offset'])<1e-8 and abs(a['edge_offset']-b['edge_offset'])<1e-8:pairs.append((a,b))
            long=lambda e:e['source_pixel_witnesses']>=cfg['edge_min_witnesses'] and e['pixel_span']>=cfg['edge_min_span_px']
            tasks.append({'scene':tag,'asset':i,'shared_z_edge_any_ray':bool(pairs),'shared_z_edge_minimum_pixel_support':any(long(a) and long(b) for a,b in pairs),'shared_z_edge_both_planes_retained':any(long(a) and long(b) and a['dominant_plane_retained_this_axis'] and b['dominant_plane_retained_this_axis'] for a,b in pairs),'pairs':[list(p) for p in pairs]})
        print('ORACLE_RAYS',tag,flush=True)
    actual=read(root/'physical_geometry_gate.json')['rows'];good=[r for r in actual if r['working_interval_covers_a_physical_edge']];extracted_shared=[]
    for task in tasks:
        early=[r for r in good if r['scene']==task['scene'] and r['asset']==task['asset'] and r['epoch']<2 and r['key'][1]%3==2];late=[r for r in good if r['scene']==task['scene'] and r['asset']==task['asset'] and r['epoch']==2 and r['key'][1]%3==2]
        pair=any(a['key']==b['key'] and abs(a['true_sideplane_offset']-b['true_sideplane_offset'])<.008 and any(abs(x-y)<.008 for x in a['possible_physical_edge_signed_offsets_m'] for y in b['possible_physical_edge_signed_offsets_m']) for a in early for b in late);extracted_shared.append(pair)
    summary={'asset_conditions':len(tasks),'oracle_actual_ray_edges':len(all_edges),'tasks_shared_depth_edge_at_least_one_actual_ray_each':sum(t['shared_z_edge_any_ray'] for t in tasks),'tasks_shared_depth_edge_minimum_pixel_support_each':sum(t['shared_z_edge_minimum_pixel_support'] for t in tasks),'tasks_shared_depth_edge_minimum_pixel_support_and_both_dominant_planes_retained':sum(t['shared_z_edge_both_planes_retained'] for t in tasks),'tasks_shared_depth_edge_validated_extracted':sum(extracted_shared),'oracle_not_inference':True,'no_raw_RGBD_unobservability_claim':True}
    save(root/'oracle_edge_visibility.json',{'summary':summary,'tasks':tasks,'oracle_edges':all_edges,'source_sha256':sha(Path(__file__)),'GT_after_full_seal':True,'features_changed':False});print(summary,flush=True)


if __name__=='__main__':main()
