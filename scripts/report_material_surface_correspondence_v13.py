"""Full controlled tasks, correspondence precision, and honest scope report."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_material_surface_correspondence_v13b import configuration,METHODS,ACTUAL_SCOPE
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));coverage=json.loads((root/'chart_coverage_summary.json').read_text(encoding='utf-8'));sampling=json.loads((root/'sampling_loss_summary.json').read_text(encoding='utf-8'));decision=json.loads((root/'mainline_decision.json').read_text(encoding='utf-8'));summary=scores['summary'];lookup={(r['variant'],r['method']):r for r in summary};variants=cfg['variants'];rows=list(csv.DictReader((root/'task_metrics.csv').open(encoding='utf-8-sig')));oracle=list(csv.DictReader((root/'oracle_by_stage.csv').open(encoding='utf-8-sig')))
    names=['V12 replay','Multiscale fusion','Independent points','Shared points','Independent normals','Shared normals','Independent phase','Shared phase'];colors=['#777777','#0072B2','#009E73','#009E73','#E69F00','#E69F00','#D55E00','#CC79A7']
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,4,figsize=(15,4.9),sharex=True,sharey=True,layout='constrained')
        for ax,v in zip(axes,variants):
            nn=[lookup[v,m]['qualified'] for m in METHODS];ax.barh(np.arange(8),nn,color=colors,edgecolor='#333333',linewidth=.6)
            for y,n in enumerate(nn):ax.text(n+.2,y,str(n),va='center')
            ax.set(xlim=(0,24),xticks=[0,8,16,24],yticks=np.arange(8),yticklabels=names,title=v,xlabel='Qualified tasks / 24');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        axes[0].invert_yaxis();fig.suptitle('V13: explicit material correspondences + continuous pose\nOne seen TRAIN; correlated appearances; common finite candidates for new scorers',fontsize=12)
        fig.savefig(root/'task_comparison.png',dpi=180);fig.savefig(root/'task_comparison.pdf');plt.close(fig)
    table='| 方法 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in METHODS:table+='| '+m+' | '+' | '.join(str(lookup[v,m]['qualified']) for v in variants)+' |\n'
    oldscores=json.loads((Path(cfg['parent_root'])/'scores.json').read_text(encoding='utf-8'));oldtable='| 旧V9强控制 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in ['projective_points','surface_rgb_fusion']:oldtable+='| '+m+' | '+' | '.join(str(next(r for r in oldscores['summary'] if r['variant']==v and r['method']==m)['qualified']) for v in variants)+' |\n'
    rear='| 方法 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in METHODS:rear+='| '+m+' | '+' | '.join(str(lookup[v,m]['rear_qualified']) for v in variants)+' |\n'
    precision='| 方法 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in METHODS[2:]:
        precision+='| '+m+' | '+' | '.join((f"{lookup[v,m]['correct_material_links']}/{lookup[v,m]['material_links']} ({lookup[v,m]['link_precision']:.1%})" if lookup[v,m]['link_precision'] is not None else '无链接') for v in variants)+' |\n'
    stage='| 外观 | 保留公共池 | 新全局提案池 | 联合连续池 | 因有限保留丢掉的旧合格资产 |\n|---|---:|---:|---:|---|\n'
    for v in variants:
        oo=[r for r in oracle if r['variant']==v];cc=next(r for r in coverage['summary'] if r['variant']==v);counts=[sum(bool(r[s+'_error_m']) and float(r[s+'_error_m'])<=.05 for r in oo) for s in ['retained','global','continuous']]
        stage+='| '+v+' | '+' | '.join(map(str,counts))+f" | {cc['old_qualified_lost_by_finite_retention']} |\n"
    details='| rich背面资产 | V12实际误差 | 普通融合 | 独立方向 | 共享方向 | 公共池最好误差（仅诊断） |\n|---|---:|---:|---:|---:|---:|\n'
    tasklookup={(r['variant'],r['method'],int(r['asset'])):r for r in rows}
    for i in [int(r['asset']) for r in rows if r['variant']=='rich' and r['method']=='shared_phase' and int(r['rear'])]:
        cells=[]
        for m in ['retained_v12','multiscale_fusion','partial_phase','shared_phase']:
            r=tasklookup['rich',m,i];cells.append(('正确ID' if int(r['both_ids']) else '错ID')+f" / {100*float(r['worst_error_m']):.2f}cm")
        o=next(r for r in oracle if r['variant']=='rich' and int(r['asset'])==i);error=o['continuous_error_m'];details+=f'| {i} | '+' | '.join(cells)+' | '+(f'{100*float(error):.2f}cm' if error else '身份未搜索')+' |\n'
    comp='\n'.join('- '+r['variant']+' / '+r['baseline']+' → '+r['candidate']+f": 新增{r['gained']}，丢失{r['lost']}。" for r in scores['comparisons']);diag={v:json.loads((root/f'{v}_diagnostics.json').read_text(encoding='utf-8')) for v in variants}
    stats={v:{k:d[k] for k in ['seconds','counters','triples','states']} for v,d in diag.items()}
    doc=f'''# V13：局部材料坐标、方向多模态和显式跨期对应的全量结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: VERIFIED（完整执行和目标/来源核对，不是原创/泛化确认）
- Version Label: material_surface_correspondence_v13_run2

## 本轮优先推进了什么

V12简单共享奖励已停止调参。本轮真正实现新全局位姿：64个实际材料中心的物理尺度纹理坐标、观测掩码、16方向分布；普通RANSAC与局部方向生成器；显式可拒绝容量对应及跨期材料簇；3步SE(3)联合更新。10项原型检查与7项周期边界检查通过。全部24×4×8={scores['primary_rows']}任务，预测封存后才读取私有标注。

原尝试遇到0/2π浮点边界索引越界，是工程执行错误，未计0分；失败源/协议/进度保留在旧根。v13b仅修索引，图像chart cache/超参数/公式不变。最终根为`results/material_surface_correspondence_v13_run2`。

## 完整任务成功数（每列24项）

{table}
判据：两个当前身份都正确，H1/H2/C12三段固定历史材料锚点95%误差全部≤5cm。没有把低自身能量、正确ID或多链接当完整成功。

不能只与较弱分支比较，旧强方法参照仍保留：

{oldtable}

跨版本每资产增减：`{json.dumps(decision['old_strong_comparisons'],ensure_ascii=False)}`。这是不同表示/搜索范围的任务参照，不是纯机制消融。

背面8项保留：

{rear}
![全部方法和外观的完整任务结果](../results/material_surface_correspondence_v13_run2/task_comparison.png)

## 成功/失败的实际增减

{comp}

法向对照是标准机制，方向必须胜过该对照；共享必须胜过同方向独立，才能支持对应机制的独立贡献。普通融合也使用全部新候选，没有给新模型独占好初值。

## 实际选中链接是否真是同材料

{precision}
分子为真实物理资产相同且材料坐标差≤15mm的链接；分母为实际选中链接。跨三段累计而非独立样本，样本相关；中心允许15mm近似对应，不声称逐像素精确身份。共享链接数量变少不能自动算更准确，需结合主任务。

## 候选链路与旧池保留范围

{stage}
这些是正确身份下GT离线oracle，**不是算法成功率**。本轮新方法共用有限实际耦合候选；保留所有V12生成候选和8个实际选中端点，V12精确预测回放。但没有为新目标重穷举全部未选旧Cartesian组合。这是预测前写明的预算取舍，不能声称跨V12是纯评分消融。上表与`retained_pool_coverage.csv`公开检查有无丢掉旧好解，不能隐去。

完整执行范围：{ACTUAL_SCOPE}

核对未发现相同已访问正确身份三元组内丢失旧合格候选。flat全因子参考oracle16与公共图14之差，是资产22/23的真身份三元组原本就不在两版相同有限身份图中，不是新有限候选截断造成。V12实际12项成功的端点全部保留，新模型退步不能用删掉这些实际好解来解释。

## rich全部背面构件，不挑最好案例

{details}

## 对应覆盖与方向假设诊断

同一批原始512中心的采样干预（几何只诊断一次，不能当四次独立验证）：

```json
{json.dumps(sampling,ensure_ascii=False,indent=2)}
```

**直接根因：**背面8项原512中心保有33对15mm内的跨期近材料点；5cm邻域平面资格筛选后变成0，删除发生在FPS之前。取消平面筛选但仍仅FPS64时只保留9对、分布7构件，给定真对应仍无完整达标任务；使用全部512及真对应时7/8达标。这些是表示干预与标签辅助拟合，**不是新算法7/8成功**。不能说原始图像没有共同材料，也不能把只取消平面筛选而维持64个离散中心视为已解决。

```json
{json.dumps(coverage,ensure_ascii=False,indent=2)}
```

q-assisted分支使用私有标签筛材料近邻并挑选最有利姿态，只是诊断，未进入任何预测。每边保留32个最有利端点的有界探针不构成全局不可行证明。实际普通/方向生成器oracle也与保留公共池合并，不能当作者完整系统结果。

方向误差与中心偏移的分箱是相关证据，未做平移干预；环形模型只处理旋转而中心独立采样含平移是已记录假设风险，不能仅靠相关性就证明因果。

独立解析反例已核验：同一3cm周期平面条纹、真实旋转0°、中心偏移1.5cm，幅值描述近乎相同，但本版前两个方向峰45°/135°。这证明纯旋转假设存在结构缺陷；它不是实际任务失败次数，也不能证明全部实际错误的原因。实际材料点关系需要Rθ与二维偏移δ联合建模，**本版未实现δ估计**。见模型契约及`results/material_chart_center_shift_counterexample_v13.json`。

## 主线决策的实际证据

```json
{json.dumps(decision['summary'],ensure_ascii=False,indent=2)}
```

主线链路是否推进、完整任务净增、核心机制净增要分别判断。候选新增但实际未选不能算成功；法向贡献不能称方向贡献；普通多尺度表示的贡献不能称共享贡献。只有独立方向胜过法向对照、共享方向再胜过同方向独立，才有对应新机制的收益证据；即使满足，也仍需独立场景和完整作者强基线确认。

## 文献与创新边界

本轮决定：共享方向9/13/10/13低于旧V12的12/16/14/16；独立方向9/14/10/14未超过法向，rich/changed共享再丢资产15，所有背面仍0。**不保留V13为主算法，不扫方向温度/权重/训练轮数。**当前表示先删光背面近材料对，会阻断对共享对应收益的可信检验，修正它应优先于扩大本版实验。

下一拟实现`material_continuous_surface_correspondence_v14.py`（尚未实现）：保留全部512真实中心，不以大邻域纯平面资格硬删中心；目标使用实际测量支持的连续图像/表面坐标，显式估计二维中心偏移δ与SE(3)，法向不可辨时保留不确定性。必要对照为同512输入的普通稠密几何纹理/连续配准、独立连续对应与共享对应；所有新位姿候选共用。先确认33对近材料支持没有在入口消失，再全24×4检验背面与资产8实际选解；保留支持、加入δ本身归必要修复，不能自动算原创。

数学、已有方法和可证伪目标见[模型契约](material_surface_correspondence_model_v13_2026-10-27.md)。[PointMBF](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PointMBF_A_Multi-scale_Bidirectional_Fusion_Network_for_Unsupervised_RGB-D_Point_ICCV_2023_paper.html)已有多尺度几何纹理融合；[RoReg](https://hpwang-whu.github.io/RoReg/)已有有向描述与局部旋转配准；[CoFiNet](https://proceedings.neurips.cc/paper/2021/file/c85b2ea9a678e74fdc8bafe5d0707c31-Paper.pdf)已有粗细对应与传输。不能把局部frame、环相关、Hungarian、GN或系统拼装本身称为新理论。本轮待检验只有实际曝光/方向多解与共享材料对应是否有同表示净增。

本轮没有训练网络，也未完整运行以上作者系统；机制控制不是现代完整强方法排名。所有新方法读取相同新chart，较V12查询像素约12.8倍（仍来自相同已采集RGB-D），跨版本收益可能是表示/信息利用的工程增量。一个已见TRAIN、四相关外观，无独立确认。

## 核对与执行产物

- {scores['checks']}评分/状态/位姿/旧端点核对，{scores['fit_paths']}条真实优化路径，{scores['measured_centers_verified']}实际中心来源；选中目标最大重建误差{scores['max_selected_objective_error']:.3g}。
- 原型10项＋周期修复7项/1001点检查；PNG人工打开检查，PDF导出；最终来源/旧证据完整性另见completion_integrity.json。
- 原评分入口有一处未配对括号，解析时尚未读标注；独立v13c评分只修语法，预测/候选/源封存不变，元数据在scoring_protocol.json。以后冻结前编译所有实际入口，不能只测内核。
- 各外观统计：`{json.dumps(stats,ensure_ascii=False)}`。
- [逐任务](../results/material_surface_correspondence_v13_run2/task_metrics.csv)、[实际材料链接](../results/material_surface_correspondence_v13_run2/material_links.csv)、[候选oracle](../results/material_surface_correspondence_v13_run2/oracle_by_stage.csv)、[节点覆盖](../results/material_surface_correspondence_v13_run2/node_material_coverage.csv)、[方向与中心偏移](../results/material_surface_correspondence_v13_run2/phase_vs_center_offset.csv)。
'''
    report=Path('research/material_surface_correspondence_results_v13_2026-10-27.md');report.write_text(doc,encoding='utf-8');save(root/'figure_provenance.json',dict(source_sha=sha(root/'scores.json'),methods=METHODS,variants=variants,counts={v:[lookup[v,m]['qualified'] for m in METHODS] for v in variants},replication='One seenTRAIN; four correlated appearances; no uncertainty intervals or independent claim.',png_dpi=180));print('V13 REPORT',str(report))


if __name__=='__main__':main()
