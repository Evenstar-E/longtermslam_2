"""Complete counts, honest common-scale plot and evidence-linked research log."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_material_inverse_plane_pose_proposals_v12 import configuration,METHODS
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    cfg=configuration();root=Path(cfg['output']);scores=json.loads((root/'scores.json').read_text(encoding='utf-8'));summary=scores['summary'];lookup={(r['variant'],r['method']):r for r in summary};variants=cfg['variants']
    full=json.loads((root/'full_propagation_summary.json').read_text(encoding='utf-8'))
    decision=json.loads((root/'mainline_decision.json').read_text(encoding='utf-8'))
    local=json.loads((root/'local_information_summary.json').read_text(encoding='utf-8'))
    orows=list(csv.DictReader((root/'oracle_by_stage.csv').open(encoding='utf-8-sig')))
    labels=['Old points','Old patches','Composition points','Composition patches','Continuous points','Continuous patches','Independent material','Shared material']
    colors=['#777777','#777777','#0072B2','#0072B2','#009E73','#009E73','#D55E00','#CC79A7']
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,4,figsize=(15,4.8),sharex=True,sharey=True,layout='constrained')
        for ax,v in zip(axes,variants):
            y=np.arange(8);n=[lookup[v,m]['qualified'] for m in METHODS]
            ax.barh(y,n,color=colors,edgecolor='#222222',linewidth=.6)
            for yi,nn in zip(y,n):ax.text(nn+.25,yi,str(nn),va='center',fontsize=10)
            ax.set(xlim=(0,24),xticks=[0,8,16,24],title=v,xlabel='Qualified tasks / 24',yticks=y,yticklabels=labels)
            ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        axes[0].invert_yaxis();fig.suptitle('V12: identity + all-three material displacement errors <= 5 cm\nOne seen TRAIN scene; correlated appearances; fixed finite identity graph',fontsize=12)
        fig.savefig(root/'task_comparison.png',dpi=180);fig.savefig(root/'task_comparison.pdf');plt.close(fig)
    table='| 方法 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in METHODS:table+='| '+m+' | '+' | '.join(str(lookup[v,m]['qualified']) for v in variants)+' |\n'
    previous=json.loads(Path('results/material_surface_patch_registration_v9_run2/scores.json').read_text(encoding='utf-8'))
    oldtable='| 已保留V9强控制 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in ['surface_geometry','surface_rgb_fusion','projective_points']:
        oldtable+='| '+m+' | '+' | '.join(str(next(r for r in previous['summary'] if r['variant']==v and r['method']==m)['qualified']) for v in variants)+' |\n'
    rear='| 方法 | flat | rich | repeated | changed |\n|---|---:|---:|---:|---:|\n'
    for m in METHODS:rear+='| '+m+' | '+' | '.join(str(lookup[v,m]['rear_qualified']) for v in variants)+' |\n'
    ot='| 外观 | 原池 | 有限预算传播 | 连续共同池 | 全原池传播诊断 | 预算漏掉的合格构件 |\n|---|---:|---:|---:|---:|---|\n'
    for v in variants:
        rr=[r for r in orows if r['variant']==v];counts=[sum(float(r[f'{s}_error_m'])<=.05 for r in rr) for s in ['retained','propagated','continuous']];ff=next(r for r in full['summary'] if r['variant']==v)
        ot+='| '+v+' | '+' | '.join(map(str,counts))+f" | {ff['full_propagation_with_retained_qualified']} | {ff['budget_missed_qualified']} |\n"
    comp='\n'.join('- '+r['variant']+' / '+r['baseline']+' → '+r['candidate']+f": 新增{r['gained']}，丢失{r['lost']}，共同正确身份误差变化中位数{r['paired_error_change_median_m']}m。" for r in scores['comparisons'])
    taskrows=list(csv.DictReader((root/'task_metrics.csv').open(encoding='utf-8-sig')))
    tasklookup={(r['variant'],r['method'],int(r['asset'])):r for r in taskrows}
    rear_assets=[int(r['asset']) for r in taskrows if r['variant']=='rich' and r['method']=='shared_material' and int(r['rear'])]
    reardetail='| rich背面构件 | 旧point | 连续point | 独立材料 | 共享材料 | 共同池oracle（仅诊断） |\n|---|---:|---:|---:|---:|---:|\n'
    for i in rear_assets:
        cells=[]
        for m in ['retained_points','continuous_points','independent_material','shared_material']:
            r=tasklookup['rich',m,i];cells.append(('正确ID' if int(r['both_ids']) else '错ID')+f" / {100*float(r['worst_error_m']):.1f}cm")
        oo=next(r for r in orows if r['variant']=='rich' and int(r['asset'])==i)
        reardetail+=f'| {i} | '+' | '.join(cells)+f" | {100*float(oo['continuous_error_m']):.1f}cm |\n"
    diag={v:json.loads((root/f'{v}_diagnostics.json').read_text(encoding='utf-8')) for v in variants}
    doc=f'''# V12：真实生成连续位姿与普通传播的完整任务结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: VERIFIED（完整运行与目标重算；不是泛化/原创确认）
- Version Label: material_inverse_plane_pose_proposals_v12

## 本轮做了什么

上轮必要观测修复已经完成，因此本轮优先新位姿和完整主线收益。实现普通H1/C12、H2/C12组合；恢复后逆深度场的正反向连续RGB-D配准；独立与共享同源材料伪似然及等预算12维局部下降。每种生成器端点进入同一个最终池，两个材料模型不独占自己的好初值。原有全部Cartesian位姿、实际耦合端点、有限身份状态均保留。

完整主线：24构件×4外观×8方法={scores['primary_rows']}行。无新网络训练、无新增数据或隐藏真位姿生成；运行结束封存后才加载私有标注评分。数学、预算和近似见[模型契约](material_inverse_plane_pose_model_v12_2026-10-27.md)。

## 完整任务成功数（每列24项）

{table}
身份与H1/H2/C12三段固定历史材料锚点95%误差同时≤5cm。不是当前可见点表面距离，也不是单边位姿误差。

还必须与旧强控制比较，不能只胜过V11较弱分支就称收益：

{oldtable}
跨版本表示/查询位置改变，这个表是任务水平参照，不是纯共享因子消融；共享与独立的纯公式差异只看V12同最终候选两行。

旧强方法逐任务增减，防止总数掩盖退步：{json.dumps(decision['old_strong_comparisons'],ensure_ascii=False)}。

背面8项的实际成功数：

{rear}
![完整任务比较](../results/material_inverse_plane_pose_proposals_v12/task_comparison.png)

## 候选是否真的补解

{ot}
所有oracle均在正确身份下用私有标注离线诊断，**不等于推断成功**。全原池传播仅诊断所有既有A/C、B/C组合，没有评分选解，不能冒充新的高预算算法结果。有限预算是每边每种分数前2的并集，不承诺穷举新增传播模式。连续共同池含普通和两材料生成器全部端点，其收益不能单独归于共享模型。

上表“全原池传播诊断”也合并相同旧保留候选，确保和有限预算传播列的oracle可比；原始CSV另存纯传播结果，不能把未合并旧解的纯传播总数与合并池直接比较。

## 同候选实际增减

{comp}

### 保留全部rich背面构件的具体误差

{reardetail}
误差为三段固定历史锚点95%误差的最大值，阈值5cm；错ID即不合格，不能因为其几何误差小而计为成功。没有只挑最好构件。

## 原因判定规则

按已经封存的预测进行实际分类：

{json.dumps(decision['summary'],ensure_ascii=False,indent=2)}

真实姿态附近的局部导数诊断：

{json.dumps(local['summary'],ensure_ascii=False,indent=2)}

这里只测试本轮48块、前向H2与top2视图。局部数值秩不等于全局唯一性；深度噪声会使几何矩阵假性满秩，重涂/照明会让高纹理梯度不符合材料一致性。因此不能据此声称整个原始数据可观测或不可观测。

- oracle新增而实际没有新增：现有候选已经能表达正确解，需优先检查真实量测目标与身份偏好；不能继续用“完全没好位姿”解释。
- 全传播oracle有而有限预算oracle无：固定预算排序/裁剪存在工程瓶颈；先给所有方法同样恢复候选，不能给自己模型独占。
- oracle也没有：本轮传播/局部优化未产生解，不能由此推出原始图像没有信息。
- 共享只降低自己的能量或与独立持平：不形成论文核心独立收益；共享Bernoulli边缘化本身不是创新。

共享公式展开后只多了π(1−π)(1−η)²s1s2的共同支撑奖励。重复纹理两期都给错误表面提供高支撑时，这个因子没有真实性证据；若实测失败，不扫π、λ或轮数来包装创新。

本公式固定权重下共享修正上界B={decision['maximum_shared_rebate']:.6f}。{decision['correct_ID_bad_pose_comparisons']}个实际正确身份错误姿态对照中，真位姿相对错误姿态的独立能量差超过B的有{decision['true_pose_gap_exceeds_shared_capacity']}个；某一实际已生成合格候选差距超过B的有{decision['concrete_qualified_pose_gap_exceeds_capacity']}个。完整表在`shared_rebate_capacity.csv`。这只能证明指定两候选之间本公式无法反转排序，不能证明所有合格位姿都不可能获胜，也不是原始数据的不可辨识性证明。

## 本轮研究决策与下一项具体实现

链路确实前进到新姿态生成与完整任务；flat从上轮有向point9到12，较旧V9强point11是净增1（新增3、丢失2）。其他外观未获成功数净增，重复纹理还低于旧融合16。**共享与独立成功集合全同，不确认共享主创新，停止当前Bernoulli奖励公式的调参路线。**连续优化带来的工程收益保留。

rich/changed的背面失败分成：构件4在当前48块上真姿态无命中；另外6项有一些真实材料支持却没有生成合格姿态；构件8已有4.63cm合格候选，但选中约77.44cm的错误姿态，其独立目标相对好候选优势约0.3026，远超共享修正上界0.0556。全旧模式传播没有新增背面合格解，也没有预算漏掉的合格任务，不再以增加传播种子解释当前失败。

下一模块拟为`inspection_decisions/material_surface_correspondence_v13.py`（**尚未实现**）：显式建立局部材料坐标与多尺度几何/纹理对应变量，利用它们生成全局位姿，并让可见/遮挡/未覆盖状态参与对应代价；不再只给旧表面分数叠加共同命中奖励。相同V11真实RGB-D、实际逆深度支撑与旧候选；必要对照为普通多尺度几何纹理匹配、独立部分对应、共享对应，全部生成器端点共用。优先验收背面6项缺解能否补解、构件8是否真实选中合格解、最后才检验同候选共享净增。若仅普通匹配产生收益归工程；若新模型无净增就不进入更多训练轮数/独立大实验。相关已有方法和强基线需在实现前核验，不能仅凭模块名/新公式宣称新颖。

## 验证、边界与产物

- 24项单元检查，最大导数误差约2.10e-10。
- {scores['checks']}评分/状态/SE3/下降路径核对；{scores['fit_paths_verified']}真实优化路径；选中目标最大重建误差{scores['max_selected_objective_error']:.3g}。
- 每外观一个完整Cartesian无剪枝复核：{json.dumps(scores['brute_force'],ensure_ascii=False)}。
- 各外观生成统计与耗时：{json.dumps(diag,ensure_ascii=False)}。
- 一个已见TRAIN，四外观相关；结果是开发证据，无独立场景确认、无作者完整系统强基线结论、无概率校准结论。
- [逐任务结果](../results/material_inverse_plane_pose_proposals_v12/task_metrics.csv)、[分阶段oracle](../results/material_inverse_plane_pose_proposals_v12/oracle_by_stage.csv)、[所有旧模式传播诊断](../results/material_inverse_plane_pose_proposals_v12/full_propagation_oracle.csv)、[真位姿与最佳候选目标诊断](../results/material_inverse_plane_pose_proposals_v12/pose_mechanism.csv)。

执行命令见模型契约；原V1至V11与旧192唯一证据不覆盖。图采用统一0–24坐标与直接数值，不画N=1场景的虚假置信区间。
'''
    report=Path('research/material_inverse_plane_pose_results_v12_2026-10-27.md');report.write_text(doc,encoding='utf-8')
    save(root/'figure_provenance.json',dict(source=str(root/'scores.json'),source_sha=sha(root/'scores.json'),methods=METHODS,variants=variants,counts={v:[lookup[v,m]['qualified'] for m in METHODS] for v in variants},replication='One seen TRAIN; four correlated appearances. No uncertainty intervals or independent generalization claim.',export_png_dpi=180))
    print('V12 REPORT',str(report))


if __name__=='__main__':main()
