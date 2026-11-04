"""Make explicit-time, denominator-aware descriptive artifacts; no performance-based exclusions."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.temporal_metrics import deduplicate,derive

def read(path):return list(csv.DictReader(path.open(encoding='utf-8')))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def percent(value):return '未定义' if value is None else f'{100*value:.1f}%'

def analyze(history,output):
    output.mkdir(exist_ok=False)
    official=Path('results/labeled_temporal_v1/apartment_official_001')
    assert json.loads((official/'result.json').read_text())['complete']
    assert json.loads((history/'result.json').read_text())['returncode']==0
    rows=read(official/'results/static_objects.csv');unique,dupes=deduplicate(rows)
    official_derived=[derive(r) for r in unique]
    write(output/'official_unique_raw.json',unique)
    write(output/'official_derived.json',official_derived)
    branches={}
    for epoch in ['official_unshifted','static_shift_sensitivity']:
        branches[epoch]={}
        for method in ['official_map','intervals_disabled']:
            raw=read(history/epoch/method/'measurements.csv');uu,dd=deduplicate(raw)
            assert len(uu)==10 and dd==0
            branches[epoch][method]=[derive(r) for r in uu]
    background=read(official/'results/background_mesh.csv')
    summary={'official_raw_rows':len(rows),'official_unique_queries':len(unique),'duplicate_rows':dupes,
       'history_query_pairs_per_branch':10,'history_branches':4,'history_total_records':40,
       'independent_scene_count':1,'privileged_inputs':'GT poses and GT semantic classes',
       'official_diagonal':official_derived,'historical_queries':branches,
       'final_background_20cm':{'accuracy':float(background[-1]['Accuracy@0.2']),'completeness':float(background[-1]['Completeness@0.2'])},
       'label_epoch_verified':False,'asset_identity_accuracy':None,'visibility_conditioned_accuracy':None,
       'limitations':['Surface association within2m by semantic class is not exact asset-ID correctness.',
         'Queries and epochs are correlated diagnostics,not independent trials.',
         'Shifted static truth is sensitivity only; both conventions reported,neither selected as a correction.',
         'No current-view visibility labels; missed objects are not proven algorithm failures on observable tasks.',
         'Intervals-disabled is same-map ablation,not an independent mapping baseline.',
         'Official object-state change metrics include dynamic-trajectory GT nodes; denominator is not the six static changed instances.',
         'Official TN counters and NaN segmentation degree are not interpreted as balanced accuracy.',
         'Two layouts have labels,but only apartment is evaluated; office remains reserved.']}
    write(output/'summary.json',summary)
    # Instance examples from the unmodified official diagonal visualization.
    vis=official/'eval_visualization'
    objects,_=deduplicate(read(vis/'objects.csv'),('MapName','QueryTime','IsGT','ID'))
    raw_associations=read(vis/'associations.csv')
    # Upstream visualization converts both numeric dynamic IDs and prefixed static IDs
    # to O(index), so a displayed ID may collide. Retain every distinct target.
    associations,_=deduplicate(raw_associations,tuple(raw_associations[0]))
    lookup={(r['MapName'],r['QueryTime'],r['IsGT'],r['ID']):r for r in objects}
    amap={}
    for r in associations:amap.setdefault((r['MapName'],r['QueryTime'],r['FromGT'],r['FromID']),set()).add(r['ToID'])
    collisions=[{'key':list(k),'targets':sorted(v)} for k,v in amap.items() if len(v)>1]
    write(output/'visualization_id_collisions.json',collisions)
    labels=list(csv.DictReader(Path('data/labeled_temporal_v1/ground_truth/apartment/gt_changes.csv').open()))
    cases=[]
    for r in unique:
        for label in labels:
            symbol=label['ObjectSymbol'].strip();name=r['Name'];query=r['Query']
            targets=sorted(amap.get((name,query,'1',symbol),set()))
            target=targets[0] if len(targets)==1 else None
            cases.append({'map_index':name,'query_ns':int(query),'truth_instance':symbol,'csv_label':label,
              'association':target,'all_visualization_targets':targets,'ambiguous_display_id':len(targets)>1,'association_is_geometric_not_verified_identity':True,
              'truth':lookup.get((name,query,'1',symbol)),
              'prediction':None if target in (None,'Failed') else lookup.get((name,query,'0',target))})
    write(output/'instance_cases.json',cases)
    stamps=[int(s) for s in (official/'results/map_timestamps.txt').read_text().splitlines()]
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e6e6e6')
    for row,epoch in enumerate(branches):
        arrays=[]
        for method in ['official_map','intervals_disabled']:
            array=np.full((4,4),np.nan)
            for cell in branches[epoch][method]:
                value=cell['change_inclusive']['recall']
                if value is not None:array[int(cell['name']),stamps.index(cell['query_ns'])]=value
            arrays.append(array)
        for col,array in enumerate([arrays[0],arrays[1],arrays[0]-arrays[1]]):
            ax=axes[row,col]
            im=ax.imshow(np.ma.masked_invalid(array),cmap=cmap,vmin=0,vmax=1)
            for i in range(4):
                for j in range(4):
                    value=array[i,j]
                    ax.text(j,i,'future' if j>i else ('NA' if not np.isfinite(value) else f'{100*value:.0f}%'),ha='center',va='center',fontsize=8,color='black' if not np.isfinite(value) or value>.6 else 'white')
            ax.set_xticks(range(4),[f'{s/1e9:.2f}' for s in stamps],fontsize=8)
            ax.set_yticks(range(4),[f'{s/1e9:.2f}' for s in stamps],fontsize=8)
            ax.set_xlabel('Queried scene time [s]');ax.set_ylabel('Available robot time [s]')
            ax.set_title(['Official presence intervals','Intervals disabled: same map','Recall difference'][col]+'\n'+('Official GT timestamps' if row==0 else 'Static GT shifted: sensitivity only'),fontsize=10)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Inclusive change recall (or paired difference)',shrink=.7)
    fig.suptitle('Apartment development only; GT poses/semantics; truth epoch unresolved\n10 correlated past-query pairs per branch; grey future cells were never evaluated',fontsize=11)
    fig.savefig(output/'history_matrix.png',dpi=180);fig.savefig(output/'history_matrix.pdf');plt.close(fig)
    final=official_derived[-1];cp=final['change_conditional'];ip=final['change_inclusive'];op=final['object_presence']
    lines=['# 带标签的时序基线与历史查询诊断','',
      '这是一个合成开发场景，具有GT位姿与语义。两个场景的标签已获取，但office仍未评价。没有真实巡检准确率或新算法结论。','',
      f'官方静态表原有{len(rows)}行，实际只有{len(unique)}个不同地图／查询时间；{dupes}行是重复。新增驱动明确执行10个过去查询，每分支不重复，2种时间约定×2种区间设置共40条记录。','',
      '| 最终地图、官方原始时间约定 | 原始计数 TP/FP/FN | Precision | Recall |',
      '|---|---|---:|---:|',
      f"| 对象存在 | {op['tp']}/{op['fp']}/{op['fn']} | {percent(op['precision'])} | {percent(op['recall'])} |",
      f"| 变化，仅匹配对象 | {cp['tp']}/{cp['fp']}/{cp['fn']} | {percent(cp['precision'])} | {percent(cp['recall'])} |",
      f"| 变化，计入未匹配幻检／漏检 | {ip['tp']}/{ip['fp']}/{ip['fn']} | {percent(ip['precision'])} | {percent(ip['recall'])} |",'',
      '这些计数基于官方双向几何关联，并非严格的一一资产身份混淆矩阵。GT的76个节点中有10个带动态轨迹，官方对象变化统计没有先排除它们；20%不是6个静态变化实例的检出率。时间原点未通过逐帧事件核验，所有变化结果暂属诊断。','',
      '![历史查询矩阵](history_matrix.png)','',
      '未来查询单元没有运行；灰色不代表零分。关闭区间保留相同对象几何与轨迹，不是独立静态SLAM对手。两个时间约定都保留，偏移不是已确认修复。','',
      '官方关联可视化存在动态数字ID与静态前缀ID显示碰撞，保留全部候选并标记冲突，不任意挑选一个匹配。这影响可视化身份解读，不能推断内部原始NodeId也相同。详见visualization_id_collisions.json。','',
      '[全部计数](summary.json) · [实例案例](instance_cases.json) · [去重官方原表](official_unique_raw.json)','',
      '后续准入条件：逐帧确认事件时间与真实可见性，核对几何匹配是否对应正确资产，再开展保留office场景评价及真实数据验证。不能从当前数字直接推导新算法需求。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'official_duplicates':dupes,'unique_official_queries':len(unique),'history_records':40,'final':final},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.history,a.output)
