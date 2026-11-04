"""Diagnostic evidence at projected original-ID geometry; never convert occlusion into removal."""
import argparse,collections,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,yaml
from PIL import Image
from scripts.event_geometry import world_mesh,projected_surface,depth_evidence

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def event_checks(event,rows,cfg):
    result={'event':event,'clocks':{}}
    lower,upper=cfg['event_checks']['pre_post_window_s']
    for clock,t in event['candidate_times_ns'].items():
        before=[r for r in rows if lower*1e9<=t-r['stamp_ns']<=upper*1e9]
        after=[r for r in rows if lower*1e9<=r['stamp_ns']-t<=upper*1e9]
        pre_state='supported_surface' if event['kind']=='disappearance' else 'free_space_evidence'
        post_state='free_space_evidence' if event['kind']=='disappearance' else 'supported_surface'
        pre=[r['stamp_ns'] for r in before if r['primary']['state']==pre_state]
        post=[r['stamp_ns'] for r in after if r['primary']['state']==post_state]
        result['clocks'][clock]={'bracketed_diagnostic':bool(pre and post),'expected_before_state':pre_state,'expected_after_state':post_state,
            'before_evidence_ns':pre,'after_evidence_ns':post,
            'all_before_states':dict(collections.Counter(r['primary']['state'] for r in before)),
            'all_after_states':dict(collections.Counter(r['primary']['state'] for r in after))}
    a,b=sorted(event['candidate_times_ns'].values())
    between=[r for r in rows if a+.15e9<=r['stamp_ns']<=b-.15e9]
    supported=[r['stamp_ns'] for r in between if r['primary']['state']=='supported_surface']
    free=[r['stamp_ns'] for r in between if r['primary']['state']=='free_space_evidence']
    n=cfg['event_checks']['minimum_disagreement_interval_samples']
    observation=('supported_surface' if len(supported)>=n and not free else
                 'free_space_evidence' if len(free)>=n and not supported else 'conflicting_or_insufficient')
    favors=None
    if observation!='conflicting_or_insufficient':
        earlier=(observation=='free_space_evidence')==(event['kind']=='disappearance')
        favors='unshifted' if earlier else 'plus_bag_start'
    result['disagreement_interval']={'supported_ns':supported,'free_ns':free,'observation':observation,'diagnostically_favors':favors,
       'label_epoch_verified':False,'censor_warning':event['possible_record_end_censor']}
    return result

def run(output):
    output.mkdir(exist_ok=False);(output/'figures').mkdir()
    base=Path('data/event_evidence_v1');plan=read(base/'plan.json');acq=read(base/'acquisition.json');cfg=read(base/'frozen_config.json')
    assert acq['complete']
    assert hashlib.sha256(Path('configs/event_evidence_v1.json').read_bytes()).hexdigest()==plan['protocol_sha256']==acq['protocol_sha256']
    gate=read('configs/event_gate_decision_v1.json');write(output/'gate_config.json',gate)
    camera=acq['camera'];K=np.array(camera['K']).reshape(3,3)
    label_map=yaml.safe_load(Path('third_party/khronos_ros1_deps/hydra/config/label_remaps/tesse_cd_apartment.yaml').read_text())
    lut=np.zeros(65536,np.int32)
    for row in label_map:lut[row['sub_id']]=row['super_id']
    geometry={n['name']:world_mesh(np.load(n['mesh_path']),n['bbox']) for n in plan['nodes']}
    rows=[];lookup={};frames={r['stamp_ns']:r for r in acq['frames']}
    for fi,frame in enumerate(acq['frames']):
        depth=np.load(frame['channels']['depth']['path']);labels=lut[np.array(Image.open(frame['channels']['labels']['path']),dtype=np.uint16)]
        T=np.array(frame['T_world_camera'])
        for node in plan['nodes']:
            xy,z=projected_surface(geometry[node['name']],T,K,camera['width'],camera['height'],cfg['geometry']['minimum_camera_depth_m'])
            evidence={str(t):depth_evidence(xy,z,depth,labels,node['semantic_label'],t,cfg['diagnostic']) for t in cfg['diagnostic']['sensitivity_depth_tolerances_m']}
            item={'instance':node['name'],'id_decimal':str(node['id']),'stamp_ns':frame['stamp_ns'],
                'primary':evidence[str(cfg['diagnostic']['primary_depth_tolerance_m'])],'sensitivity':evidence,
                'projected_box_xyxy':None if not len(xy) else [int(xy[:,0].min()),int(xy[:,1].min()),int(xy[:,0].max()),int(xy[:,1].max())]}
            rows.append(item);lookup[(node['name'],frame['stamp_ns'])]=item
        if fi%50==0:print('Analyzed frame',fi,'/',len(acq['frames']),flush=True)
    write(output/'frame_evidence.json',rows)
    by_instance={n['name']:[r for r in rows if r['instance']==n['name']] for n in plan['nodes']}
    events=[event_checks(e,by_instance[e['instance']],cfg) for e in plan['events']]
    write(output/'event_checks.json',events)
    stats={}
    for n in plan['nodes']:
        selected=by_instance[n['name']];states=collections.Counter(r['primary']['state'] for r in selected)
        stats[n['name']]={'id_decimal':str(n['id']),'semantic_label':n['semantic_label'],'states':dict(states),
            'local_bracketed_non_censor':any(e['event']['instance']==n['name'] and not e['event']['possible_record_end_censor'] and any(v['bracketed_diagnostic'] for v in e['clocks'].values()) for e in events),
            'any_supported_ns':[r['stamp_ns'] for r in selected if r['primary']['state']=='supported_surface'],
            'any_free_ns':[r['stamp_ns'] for r in selected if r['primary']['state']=='free_space_evidence']}
    bracketed=sum(s['local_bracketed_non_censor'] for s in stats.values())
    # Original IDs, geometric alternatives, and display category ambiguity remain explicit.
    prediction_nodes=read('results/event_evidence_v1/prediction_nodes.json')
    old_cases=read('results/labeled_temporal_v1/analysis_003/instance_cases.json')
    inventory=read('results/labeled_temporal_v1/label_inventory.json')['scenes']['apartment']['nodes']
    associations=[]
    for case in old_cases:
        if case['map_index']!='3':continue
        target=case['association'];index=None if target in (None,'Failed') else int(target[2:-1])
        candidates=[] if index is None else [r for r in prediction_nodes if r['category_index']==index]
        n=next(n for n in plan['nodes'] if n['name']==case['truth_instance']);center=np.array(n['bbox']['world_P_center'])
        details=[]
        for pred in candidates:
            pos=np.array(pred['world_surface_centroid'])
            other=sorted([{'id_decimal':str(x['id']),'name':x['name'],'center_distance_m':float(np.linalg.norm(pos-np.array(x['bbox']['world_P_center'])))} for x in inventory if x['semantic_label']==n['semantic_label'] and x['trajectory_stamps_count']==0],key=lambda x:x['center_distance_m'])
            details.append({'prediction':pred,'distance_to_requested_gt_box_center_m':float(np.linalg.norm(pos-center)),
               'nearest_same_class_gt_centers':other[:3],'not_unique_identity_ground_truth':True})
        associations.append({'instance':n['name'],'gt_id_decimal':str(n['id']),'old_display_target':target,'candidate_raw_predictions':details,
            'identity_verified':False,'reason':'Raw IDs disambiguated where unique; geometric class/depth compatibility alone cannot prove persistent asset identity.'})
    write(output/'association_identity_checks.json',associations)
    summary={'frames':len(acq['frames']),'instances':len(plan['nodes']),'frame_instance_records':len(rows),'events':len(events),
       'possible_record_end_censors':sum(e['event']['possible_record_end_censor'] for e in events),'stats':stats,
       'local_bracketed_instances':bracketed,'exploratory_gate_required':gate['exploratory_minimum_locally_bracketed_instances'],
       'decision':'sufficient_for_next_label_review_only' if bracketed>=gate['exploratory_minimum_locally_bracketed_instances'] else 'insufficient_for_task_efficacy_inference_on_this_sample',
       'label_epoch_verified':False,'instance_identity_verified':False,'visibility_ground_truth_created':False,
       'primary_tolerance_m':cfg['diagnostic']['primary_depth_tolerance_m'],
       'clock_evidence':dict(collections.Counter(e['disagreement_interval']['diagnostically_favors'] or 'unresolved' for e in events if not e['event']['possible_record_end_censor'])),
       'scope':'Frozen diagnostic projection evidence in one synthetic development recording; not algorithm/task accuracy or novelty.'}
    write(output/'summary.json',summary)
    request_lookup={r['id']:r for r in acq['requests']}
    def overlay(ax,instance,stamp,title):
        if stamp is None:ax.text(.5,.5,'Outside recording',ha='center',va='center');ax.axis('off');return
        frame=frames[stamp];node=next(n for n in plan['nodes'] if n['name']==instance);T=np.array(frame['T_world_camera'])
        rgb=np.array(Image.open(frame['channels']['rgb']['path']).convert('RGB'));depth=np.load(frame['channels']['depth']['path'])
        labels=lut[np.array(Image.open(frame['channels']['labels']['path']),dtype=np.uint16)]
        xy,z=projected_surface(geometry[instance],T,K,camera['width'],camera['height'])
        ax.imshow(rgb)
        if len(xy):
            d=depth[xy[:,1],xy[:,0]];tol=cfg['diagnostic']['primary_depth_tolerance_m'];valid=np.isfinite(d)&(d>0)
            colors=np.full(len(z),'#999999',dtype=object)
            colors[valid&(d<z-tol)]='#E69F00';colors[valid&(d>z+tol)]='#56B4E9'
            support=valid&(np.abs(d-z)<=tol);colors[support]='#CC79A7';colors[support&(labels[xy[:,1],xy[:,0]]==node['semantic_label'])]='#009E73'
            skip=max(1,len(xy)//1000);ax.scatter(xy[::skip,0],xy[::skip,1],s=2,c=colors[::skip],alpha=.7)
        e=lookup[(instance,stamp)]['primary'];ax.set_title(f'{title}\nt={stamp/1e9:.3f}s; {e["state"]}\nprojected={e["projected_pixels"]}, support={e["class_support"]}, free={e["free"]}',fontsize=8)
        ax.set_xlim(0,camera['width']);ax.set_ylim(camera['height'],0);ax.axis('off')
    for node in plan['nodes']:
        name=node['name'];index=next(i for i,e in enumerate(plan['events']) if e['instance']==name and not e['possible_record_end_censor'])
        primary_event=plan['events'][index];fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
        for i,clock in enumerate(['unshifted','plus_bag_start']):
            for j,offset in enumerate([-.25,.25]):
                req=request_lookup[f'event_{index}_{clock}_{offset:+.2f}']
                overlay(axes[i,j],name,req.get('selected_stamp_ns'),clock+(' before' if offset<0 else ' after'))
        allrows=by_instance[name];best_support=max(allrows,key=lambda r:(r['primary']['class_support'],-r['stamp_ns']))
        best_free=max(allrows,key=lambda r:(r['primary']['free'],-r['stamp_ns']))
        overlay(axes[0,2],name,best_support['stamp_ns'],'Largest support count: illustration only')
        overlay(axes[1,2],name,best_free['stamp_ns'],'Largest free count: illustration only')
        fig.suptitle(f'{name}: {primary_event["kind"]}; green=class/depth support, orange=nearer occluder, blue=farther space\nOriginal RGB with derived GT projection; selected illustrative maxima do not define the sampling or decision',fontsize=10)
        fig.savefig(output/'figures'/f'instance_{name[2:-1]}.png',dpi=160);fig.savefig(output/'figures'/f'instance_{name[2:-1]}.pdf');plt.close(fig)
    fig,axes=plt.subplots(6,1,figsize=(12,13),layout='constrained',sharex=True,sharey=True)
    for ax,node in zip(axes,plan['nodes']):
        selected=by_instance[node['name']];t=np.array([r['stamp_ns']/1e9 for r in selected]);n=np.array([r['primary']['projected_pixels'] for r in selected]);valid=n>=cfg['diagnostic']['minimum_projected_pixels']
        for field,label,color,marker in [('class_support','Class+depth support','#009E73','o'),('free','Farther-depth / free evidence','#0072B2','x')]:
            values=np.array([r['primary'][field] for r in selected]);ax.scatter(t[valid],values[valid]/n[valid],s=9,color=color,marker=marker,label=label)
        ax.scatter(t[~valid],np.full((~valid).sum(),-.04),s=9,marker='|',color='#777777',label='Too few projected pixels')
        event=next(e for e in plan['events'] if e['instance']==node['name'] and not e['possible_record_end_censor'])
        for clock,color,style in [('unshifted','#CC79A7','-'),('plus_bag_start','#444444','--')]:ax.axvline(event['candidate_times_ns'][clock]/1e9,color=color,linestyle=style,label=clock)
        ax.set_ylim(-.08,1.03);ax.set_ylabel(node['name']+'\nfraction');ax.grid(alpha=.15)
    axes[0].legend(fontsize=7,ncol=3,loc='upper left');axes[-1].set_xlabel('Original image/TF stamp [s]')
    fig.suptitle('All frozen sampled frames: diagnostic surface evidence, not instance truth or detection accuracy\nView changes affect scores; no interpolation over unobserved intervals',fontsize=11)
    fig.savefig(output/'figures/timeline.png',dpi=160);fig.savefig(output/'figures/timeline.pdf');plt.close(fig)
    lines=['# 六个静态变化实例的图像／几何证据核验','',f'固定样本：{len(acq["frames"])}帧，6实例，{len(rows)}条逐帧实例记录；3种深度容限分别保留。','',
       '| 实例 | 表面支持帧 | 后方空域帧 | 遮挡帧 | 投影不足帧 | 非截止事件有双侧证据 |','|---|---:|---:|---:|---:|---|']
    for name,s in stats.items():
        st=s['states'];lines.append(f'| {name} | {st.get("supported_surface",0)} | {st.get("free_space_evidence",0)} | {st.get("occluded",0)} | {st.get("insufficient_projection",0)} | {s["local_bracketed_non_censor"]} |')
    lines+=['',f'探索性准入：{bracketed}/6实例有固定窗口双侧诊断证据，预设门槛为4/6；结论：`{summary["decision"]}`。','',
       '计数不是可见性真值或算法准确率。深度／同类语义支持只能说明指定位置有兼容表面；不能单独证明持久实例身份。看不见、被遮挡、投影不足均不算移除。','',
       '两个等于录像时长的消失端点单列为可能的记录截止，不纳入真实事件双侧准入。两种时钟约定都保留，不按有利结果挑选。','',
       '![所有固定帧的证据](figures/timeline.png)','',
       '[完整逐帧记录](frame_evidence.json) · [事件窗口](event_checks.json) · [原始ID／关联核对](association_identity_checks.json) · [汇总](summary.json)','']
    for n in plan['nodes']:lines+=['### '+n['name'],'',f'![{n["name"]}](figures/instance_{n["name"][2:-1]}.png)','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(output/'manifest.json',{'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/event_geometry.py'),Path('configs/event_evidence_v1.json'),Path('configs/event_gate_decision_v1.json')]},
       'acquisition_sha256':hashlib.sha256((base/'acquisition.json').read_bytes()).hexdigest(),'artifacts_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({k:summary[k] for k in ['frames','frame_instance_records','local_bracketed_instances','decision','clock_evidence']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
