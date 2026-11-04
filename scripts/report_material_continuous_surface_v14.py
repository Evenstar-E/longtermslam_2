"""Full-task result and mathematically bounded novelty assessment."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_material_continuous_surface_correspondence_v14c import configuration,METHODS


def main():
    c=configuration();root=Path(c['output']);s=json.loads((root/'scores.json').read_text(encoding='utf-8'));d=json.loads((root/'mainline_decision.json').read_text(encoding='utf-8'));support=json.loads((root/'continuous_support_summary.json').read_text(encoding='utf-8'));L={(r['variant'],r['method']):r for r in s['summary']};variants=c['variants'];oracle=list(csv.DictReader((root/'oracle_by_stage.csv').open(encoding='utf-8-sig')))
    def table(key):
        t='| 方法 | '+' | '.join(variants)+' |\n|---|'+'---:|'*4+'\n'
        for m in METHODS:t+='| '+m+' | '+' | '.join(str(L[v,m][key]) for v in variants)+' |\n'
        return t
    old=json.loads((Path(c['parent_root'])/'scores.json').read_text(encoding='utf-8'));oldtable='| 此前较强普通控制 | '+' | '.join(variants)+' |\n|---|'+'---:|'*4+'\n'
    for m in ['projective_points','surface_rgb_fusion']:oldtable+='| '+m+' | '+' | '.join(str(next(r for r in old['summary'] if r['variant']==v and r['method']==m)['qualified']) for v in variants)+' |\n'
    labels=['V12 replay','Dense geometry','Dense RGBD fusion','Fixed projection','Continuous pair','Independent group','Independent track','Shared track'];colors=['#777777','#0072B2','#56B4E9','#E69F00','#009E73','#CC79A7','#D55E00','#882255']
    with plt.rc_context({'font.size':10}):
        fig,axes=plt.subplots(1,4,figsize=(15,5),sharex=True,sharey=True,layout='constrained')
        for ax,v in zip(axes,variants):
            n=[L[v,m]['qualified'] for m in METHODS];ax.barh(np.arange(8),n,color=colors,edgecolor='#333333',linewidth=.5)
            for y,x in enumerate(n):ax.text(x+.2,y,str(x),va='center')
            ax.set(xlim=(0,24),xticks=[0,8,16,24],yticks=np.arange(8),yticklabels=labels,title=v,xlabel='Qualified tasks / 24');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        axes[0].invert_yaxis();fig.suptitle('V14c: all measured centers, continuous surface correspondence\nOne seen TRAIN; correlated appearances; common finite candidate graph',fontsize=12);fig.savefig(root/'task_comparison.png',dpi=180);fig.savefig(root/'task_comparison.pdf');plt.close(fig)
    o='| 外观 | 保留旧端点 | 加普通全局提案 | 加连续优化 |\n|---|---:|---:|---:|\n'
    for v in variants:
        rr=[r for r in oracle if r['variant']==v];o+='| '+v+' | '+' | '.join(str(sum(bool(r[k+'_error_m']) and float(r[k+'_error_m'])<=.05 for r in rr)) for k in ['retained','global','continuous'])+' |\n'
    comp='\n'.join('- '+r['variant']+'：'+r['baseline']+' → '+r['candidate']+f"，新增{r['gained']}，丢失{r['lost']}，共同正确身份下误差变化中位数={r['paired_worst_error_change_median_m']}m。" for r in s['comparisons'])
    probe=json.loads((root/'public_solver_probe.json').read_text(encoding='utf-8'));probe_scores=json.loads((root/'solver_probe_scores.json').read_text(encoding='utf-8'));bridge=json.loads((root/'material_bridge_support_summary.json').read_text(encoding='utf-8'));radio=json.loads((root/'material_radiometry_summary.json').read_text(encoding='utf-8'));triples=json.loads((root/'continuous_triple_support_summary.json').read_text(encoding='utf-8'))
    ftable='| 外观/方法 | 成功 | 正确身份缺好位姿 | 有好位姿但评分选坏位姿 | 身份分配失败 | 真身份未访问 |\n|---|---:|---:|---:|---:|---:|\n'
    for r in d['failure_summary']:
        if r['method'] not in ['continuous_pair','material_track_shared']:continue
        cc=r['categories'];ftable+=f"| {r['variant']}/{r['method']} | "+' | '.join(str(cc.get(k,0)) for k in ['success','no_qualified_pose_in_true_identity_pool','objective_prefers_bad_pose_at_correct_identity','identity_assignment_failed_despite_qualified_pose','true_identity_not_visited'])+' |\n'
    text=f'''# V14c：保留全部真实中心、连续材料对应与自由空间证据

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: 完整预测后评分；任务收益与新颖性分开解释
- Version Label: material_continuous_surface_correspondence_v14_run3

## 本轮真实实现

全部512真实源中心不再经过平面筛选或FPS64。目标使用原top2视图全部原生RGB-D像素，建立实际三角形逆深度连续场，无法解析处回退真实像素。连续二维对应偏移与两期SE(3)交替优化。相同候选、相同RGB-D下设置稠密几何、普通融合、固定投影、连续对应、独立材料组、独立轨迹、共享轨迹对照；旧V12预测精确回放。无新训练/下载/采集。

**本轮结论：不将V14共享Gaussian/轨迹作为已成立的论文主创新。**同信息共享新增0、丢失1；固定投影→连续δ没有新增完整成功。保留全512/连续场作为观测基础，不继续扫当前共享系数或把更多迭代当创新。完整任务与错误归因已完成，下一主攻部署一致的全局材料对应，不能继续只重排同一批错误位姿。

原V14性能问题通过位姿评分与KD树缓存解决，等价检查最大差0。随后发现原聚合把“有匹配”和“有自由空间冲突”误作互斥：存在软对应就清零冲突。V14c按原始几何射线逐视图固定平均保留冲突，δ和另一个视角命中均不能取消。7项最终聚合检查通过。原V14/V14b中断结果保留、不评分；只复核复用其stage0/1几何候选，**所有能量与连续优化重新运行**。这是必要正确性修复，不是原创点。

## 完整任务结果

全部24×4×8={s['primary_rows']}项。两期身份都正确且三段固定历史材料锚点95%位移误差均≤5cm才成功。

{table('qualified')}
此前较强普通方法也保留作任务参照（跨版本表示/搜索不同，不能视为单机制消融）：

{oldtable}
背面8项：

{table('rear_qualified')}
![完整任务对比](../results/material_continuous_surface_correspondence_v14_run3/task_comparison.png)

普通稠密控制/谱一致性/SVD/ICP属于已有工具。跨V13同时改变表示与查询预算，不能把跨版本涨分都归给共享。真正共享的贡献只看本轮独立轨迹→共享轨迹。

## 对照反映什么

{comp}

相同信息独立轨迹→共享轨迹共新增{d['shared_vs_same_information_track']['gain']}、丢失{d['shared_vs_same_information_track']['lost']}。四种外观相关，不能作为四个独立场景或统计显著性证据。

## 候选不足与评分偏好分开

{o}
上表是私有标签辅助离线oracle，**不是算法成功率**。有限已访问身份图中没有真三元组，不等于真实问题不可辨识；池内没有好解，也不能证明连续空间不存在好解。

{ftable}
“评分选坏位姿”仅在身份正确且同三元组池中已存在≤5cm候选时判定；身份分配失败可能混合评分和有限分配搜索。逐项证据见failure_attribution.csv，而不是统一归因于初值或数据。

实际rich背面：7项缺合格候选、资产8有4.6266cm候选却选77.5752cm错误位姿；独立连续目标差0.01435、共享目标差0.03132，都更偏好坏解，共享还扩大该差距。repeated资产3在独立轨迹中误差1.280mm，改共享后同身份误差78.781cm，因此丢失的是材料量化成功，而非仅身份标签。

新全局与连续阶段的合格候选覆盖均未超过保留阶段：flat14、rich17、repeated16、changed17。3步局部位姿单步12维增量范数上限0.04，旋转累计至多0.12rad；它无法承担从近180°错朝向重新全局定位的任务。此界只限制该局部阶段，不证明原始数据不可恢复；全局候选生成也已实际测试而未补出新合格资产。

## 输入支持与实际对应

原始512中心中的33对背面近材料支持已复核保留，未在输入层删除。正确位姿/正确身份查询统计：

```json
{json.dumps(support['summary'],ensure_ascii=False,indent=2)}
```

这些是GT条件诊断，不能计为实际匹配成功。实际选中链接精度见active_material_links.csv与scores.json：同物理资产且材料坐标差≤15mm，激活门限与实际位姿求解相同。连续位置的私有标签按最近原生像素核验，是像素尺度近似。

## 求解器、跨期桥接与颜色模型的额外诊断

在完整预测封存前，不读GT，对rich全部24×2个实际输出追加3步原几何拟合，再以完整目标有限差分做一步下降探针；48项中{sum(r['gradient_after_geometry_improves'] for r in probe['summary'].values())}项仍可下降。端点独立保存，**没有插入/替换主768预测**。这是求解近似证据，不能自动算材料位移收益；封存后对全部48项统一评分：

```json
{json.dumps(probe_scores['summary'],ensure_ascii=False,indent=2)}
```

原射线曝光解析world-point梯度原型已实现，100点×3分量数值最大差2.88e−8，**未接入本轮任务求解**，不能列为新任务收益。

桥接支持诊断：

```json
{json.dumps(bridge,ensure_ascii=False,indent=2)}
```

H未采到近邻不等于历史图像完全未见，GT配对拟合不是新算法。此诊断用于判断只以H点分组是否丢掉C1↔C2的材料补充机会，而不是借助GT生成新预测。

**必须使用实际模型的原生稠密目标域复核，不能拿稀疏512↔512三重近邻数推断没有共同材料。**正确位姿下真实连续查询得到：

```json
{json.dumps(triples['summary'],ensure_ascii=False,indent=2)}
```

优化δ后背面8/8均有至少3个正确三期材料链接，合计110个；固定δ也有98个、分布7/8。输入支持确实恢复，尚不能说算法已找到/利用这些正确链接；也不证明几何条件足以稳定量化。稀疏bank桥接表仅是不同采样域的补充诊断，不作为实际观测不足的结论。

实际数据的近同材料RGB诊断：rich共{radio['total_pair_queries']}个≤3mm材料邻近对，其中{radio['paired_triple_points']}个H点在两个当前时期均满足该条件；逐资产/时期RGB向量差中位数的中位为{radio['median_asset_epoch_rgb_l2']:.4f}（RGB归一到0–1）。逐项偏置与6×6协方差见material_radiometry_summary.json。近点纹理变化和像素近似仍在，不能把全部差异唯一归因为照明。

生成器明确包含时期光照、法向相关明暗、每帧曝光/白平衡。因此c_e=s_eρ+ε_e下残差协方差含ρρᵀ且均值随光照变化；当前固定各向同性共享Gaussian遗漏这些结构。这是可检验的模型风险，尚无单独干预证明其贡献多少失败。下一若改光照不变特征/反射率，需要普通仿射校正和同表征独立控制，不能把标准曝光补偿当原创。

## 数学分析与创新性边界

本版在感知中的跨期材料对应/位姿估计模块工作。对于e1=Ψ1−G1x、e2=Ψ2−G2x，轨迹残差严格满足r_track=R21 e1−e2。100组随机刚体核对最大误差2.67e−15；固定姿态下加入该行不增加原6维残差信息秩。它可以改变误差耦合/权重，不能称为第三个独立观测。独立/共享颜色协方差是公共历史颜色误差边缘化的标准统计工具，连续RGB-D、局部δ、谱一致性和SVD也不是新发明。

因此完整链路推进、具体任务净增、论文原创性是三个结论。本轮结果只检验当前构造是否产生净增，不确认全新方法。若仍缺位姿，下一应检验连续求解可达性；若好位姿已存在而目标偏好错误，下一应改观测似然而非继续加候选/迭代。当前位姿方向仅用加权几何Jacobian，完整能量回溯但未纳入全部自由空间/光度/轨迹导数，这是明确的近似求解限制；颜色稳定、无显式重涂null和局部多对一仍是模型限制。

## 可靠性与限制

评分核对{s['checks']}项、优化路径{s['fit_paths_verified']}条、真实源中心{s['source_centers_verified']}个，最大选中目标重算差{s['max_selected_objective_error']:.3g}。预测seal先于私有标注读取；代码/候选/测量证据保留，旧192另做哈希核验。模型公式见material_continuous_surface_model_v14_2026-10-27.md。

单个已见TRAIN，分割/匿名构件mask已给定，四相关外观；尚无独立真实跨期或现代发表作者完整系统比较。SC²只是参考机制，不是作者完整系统的成绩。
'''
    Path('research/material_continuous_surface_results_v14_2026-10-27.md').write_text(text,encoding='utf-8');print('V14c report + figures written')


if __name__=='__main__':main()
