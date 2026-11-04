"""Choose auditable observation pairs away from both candidate event times; no method scores used."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scripts.event_geometry import world_mesh,projected_surface

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def select(event,rows):
    low,high=min(event['candidate_times_ns'].values()),max(event['candidate_times_ns'].values())
    before_state='supported_surface' if event['kind']=='disappearance' else 'free_space_evidence'
    after_state='free_space_evidence' if event['kind']=='disappearance' else 'supported_surface'
    before=[r for r in rows if r['stamp_ns']<=low-150000000 and r['primary']['state']==before_state]
    after=[r for r in rows if r['stamp_ns']>=high+150000000 and r['primary']['state']==after_state]
    return (max(before,key=lambda r:r['stamp_ns']) if before else None,min(after,key=lambda r:r['stamp_ns']) if after else None)

def run(out):
    out.mkdir(exist_ok=False)
    plan=read('data/event_evidence_v1/plan.json');acq=read('data/event_evidence_v1/acquisition.json')
    rows=read('results/event_evidence_v1/analysis_001/frame_evidence.json')
    frames={r['stamp_ns']:r for r in acq['frames']};K=np.array(acq['camera']['K']).reshape(3,3)
    pairs=[]
    for node in plan['nodes']:
        e=next(e for e in plan['events'] if e['instance']==node['name'] and not e['possible_record_end_censor'])
        pre,post=select(e,[r for r in rows if r['instance']==node['name']])
        record={'instance':node['name'],'id_decimal':str(node['id']),'event':e,'has_pair':pre is not None and post is not None,
          'before':pre,'after':post,'identity_verified':False,'clock_verified':False}
        for side,item in [('before',pre),('after',post)]:
            if item is not None:record[side+'_channels']=frames[item['stamp_ns']]['channels']
        if pre is not None and post is not None:
            record['observation_interval_s']=(post['stamp_ns']-pre['stamp_ns'])/1e9
            # Same selected raster points/depth count under the fixed tolerance sensitivity grid.
            record['same_states_at_all_tolerances']=all(item['sensitivity'][key]['state']==item['primary']['state'] for item in [pre,post] for key in item['sensitivity'])
        pairs.append(record)
    write(out/'pairs.json',pairs)
    fig,axes=plt.subplots(6,2,figsize=(10,19),layout='constrained')
    for i,(node,pair) in enumerate(zip(plan['nodes'],pairs)):
        world=world_mesh(np.load(node['mesh_path']),node['bbox'])
        for j,side in enumerate(['before','after']):
            ax=axes[i,j];item=pair[side]
            if item is None:ax.text(.5,.5,'Unresolved: no qualifying observation',ha='center');ax.axis('off');continue
            frame=frames[item['stamp_ns']];rgb=np.array(Image.open(frame['channels']['rgb']['path']).convert('RGB'))
            xy,z=projected_surface(world,np.array(frame['T_world_camera']),K,acq['camera']['width'],acq['camera']['height'])
            ax.imshow(rgb)
            color='#009E73' if item['primary']['state']=='supported_surface' else '#0072B2'
            if len(xy):
                # Box only, so raw appearance remains legible. Full point overlays are separate.
                x0,y0=xy.min(axis=0);x1,y1=xy.max(axis=0)
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((x0,y0),max(1,x1-x0),max(1,y1-y0),fill=False,edgecolor=color,linewidth=1.5))
            e=item['primary'];ax.set_title(f'{node["name"]} {side}: t={item["stamp_ns"]/1e9:.3f}s\n{e["state"]}; support={e["class_support"]}, free={e["free"]}, projected={e["projected_pixels"]}',fontsize=9)
            ax.axis('off')
    fig.suptitle('Observation pairs outside both candidate event times: one synthetic development recording\nGreen box: compatible object surface; blue box: expected location with farther-depth evidence\nPost-hoc deterministic data qualification; not algorithm performance or persistent-ID proof',fontsize=10)
    fig.savefig(out/'observation_pairs.png',dpi=170);fig.savefig(out/'observation_pairs.pdf');plt.close(fig)
    report={'pairs_available':sum(p['has_pair'] for p in pairs),'instances':len(pairs),'all_selected_states_tolerance_stable':all(p.get('same_states_at_all_tolerances',False) for p in pairs),
      'config_sha256':hashlib.sha256(Path('configs/event_robust_pairs_v1.json').read_bytes()).hexdigest(),
      'scope':'Exploratory observation-pair qualification; exact event time and unique physical identity unverified',
      'local_gate_interpretation':'Original0/6local brackets limits clock verification only; it cannot reject cross-visit change tasks.'}
    write(out/'summary.json',report)
    lines=['# 不依赖精确事件秒数的跨时段观察对','',
       '保留原始0/6局部窗口结果，用于说明事件时钟仍未核实。跨次变化任务不要求看到事件发生瞬间；这里使用更早／更晚且不跨候选时间歧义的观察。选择规则为事后数据资格检查，不是确认实验。','',
       '| 实例 | 变化 | 历史帧秒 | 当前帧秒 | 两次观察间隔秒 | 各容限状态一致 |','|---|---|---:|---:|---:|---|']
    for p in pairs:
        lines.append(f'| {p["instance"]} | {p["event"]["kind"]} | {p["before"]["stamp_ns"]/1e9 if p["before"] else "未决"} | {p["after"]["stamp_ns"]/1e9 if p["after"] else "未决"} | {p.get("observation_interval_s","未决")} | {p.get("same_states_at_all_tolerances",False)} |')
    lines+=['','![历史／当前观察对](observation_pairs.png)','',
      '同类语义＋深度支持并不证明唯一实例身份；原始图像和深度路径在pairs.json中。没有对照方法在这些选择后的帧上运行，不能把6个可用证据对称为6次算法成功。','',
      '[逐对数据与原始输入](pairs.json) · [规则](../../../configs/event_robust_pairs_v1.json)']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
