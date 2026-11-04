"""Read sealed DEV evidence; export descriptive figures and a Chinese report."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('results/material_correspondence_comparison_v3')
REP = Path('results/material_correspondence_replacement_v3')
VARIANTS = ['flat', 'rich', 'repeated', 'changed']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(root, method, stage=None):
    with (Path(root)/'task_scores.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['split'] in ['validation','test'] and r['method'] == method and (stage is None or r['stage'] == stage)]
    result = []
    for variant in VARIANTS:
        rr = [r for r in rows if r['variant'] == variant]
        assert len(rr) == 96
        result.append(sum(int(r['qualified']) for r in rr))
    return result


def main():
    main_scores = load(ROOT/'scores.json')
    diag = load(ROOT/'repair_diagnostics.json')
    target = load('results/material_correspondence_training_v3/target_distribution_diagnostic.json')
    sources = [ROOT/'scores.json', REP/'scores.json', ROOT/'task_scores.csv', REP/'task_scores.csv',
               Path('results/textured_material_comparison_v1/task_scores.csv'),
               Path('results/material_joint_descriptor_comparison_v2/task_scores.csv'),
               ROOT/'repair_diagnostics.json', Path('results/material_correspondence_training_v3/target_distribution_diagnostic.json')]
    table = [
        ('Old SIFT fusion', counts('results/textured_material_comparison_v1', 'fusion')),
        ('Previous V2 balanced', counts('results/material_joint_descriptor_comparison_v2', 'balanced_material')),
        ('Shared pool ordinary', counts(ROOT, 'ordinary', 'refined')),
        ('Shared pool ordinary + ray', counts(ROOT, 'ray', 'refined')),
        ('V3 learned additive', counts(ROOT, 'soft3', 'refined')),
        ('V3 learned replacement (posthoc)', counts(REP, 'soft3', 'refined')),
    ]
    # Historical method names are checked against actual CSVs, never filled with guesses.
    app = {r['model']: r for r in diag['appearance_metrics']}
    for r in app.values():
        r['precision'] = r['repaint_detected']/max(1, r['repaint_detected']+r['unchanged_false_positive'])
    out = dict(variants=VARIANTS, tasks_per_variant=96, scenes=4, development_only=True,
               task_comparison=[dict(method=k, qualified=v) for k,v in table], appearance=app,
               states=diag['states'], teacher=target)
    (ROOT/'report_source_data.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = ['#666666', '#CC79A7', '#999999', '#0072B2', '#D55E00', '#009E73']
    fig, ax = plt.subplots(figsize=(12,5.5), layout='constrained')
    x=np.arange(4); width=.125
    for i, ((name, values), color) in enumerate(zip(table, colors)):
        pos=x+(i-2.5)*width
        bars=ax.bar(pos,values,width,label=name,color=color,edgecolor='black',linewidth=.4,
                    hatch='//' if i==5 else None)
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=x,xticklabels=VARIANTS,ylim=(0,20),yticks=np.arange(0,21,4),
           ylabel='Qualified objects / 96',title='Four reused DEV scenes: identity AND fixed-material displacement <= 5 cm')
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9,frameon=False)
    fig.savefig(ROOT/'repair_task_comparison.png',dpi=180);fig.savefig(ROOT/'repair_task_comparison.pdf');plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(11,4.2),layout='constrained')
    names=['before_31','repaired_31','repaired_47','repaired_73']; labels=['Before\nseed 31','Repair\nseed 31','Repair\nseed 47','Repair\nseed 73']
    for ax,key,title,limit in zip(axes,['repaint_recall','unchanged_fpr','precision'],
                                    ['Repaint recall (64 positives)','Unchanged false-positive rate (12,249 negatives)','Precision among predicted repaint'],[60,6,7]):
        values=[100*app[n][key] for n in names]
        bars=ax.bar(np.arange(4),values,color=['#666666','#D55E00','#0072B2','#009E73'],edgecolor='black',linewidth=.4)
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3,fontsize=9)
        ax.set(xticks=np.arange(4),xticklabels=labels,ylim=(0,limit),ylabel='Percent',title=title,axisbelow=True)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('GT-identity + actual covered-pair diagnostic; TRAIN-only thresholds target 5% FPR',fontsize=11)
    fig.savefig(ROOT/'repair_appearance_operating_points.png',dpi=180);fig.savefig(ROOT/'repair_appearance_operating_points.pdf');plt.close(fig)
    rows='\n'.join('| '+name+' | '+' | '.join(map(str,values))+' |' for name,values in table)
    replacement='\n'.join('| '+m+' | '+' | '.join(map(str,counts(REP,m,'refined')))+' |' for m in ['ray','hard2','soft2','soft3'])
    pairs=[r for r in main_scores['refined']['paired_comparisons'] if r['split']=='all' and r['baseline']=='ordinary' and r['method']=='soft3']
    pairtable='\n'.join(f"| {r['variant']} | {r['common_correct_ids']} | {r['baseline_error_12_median_m']*100:.3f} | {r['candidate_error_12_median_m']*100:.3f} |" for r in pairs)
    approws='\n'.join(f"| {n} | {r['repaint_detected']}/64 | {100*r['repaint_recall']:.2f}% | {r['unchanged_false_positive']}/12249 | {100*r['unchanged_fpr']:.2f}% | {100*r['precision']:.2f}% |" for n,r in app.items())
    report=r'''# 材料对应模型V3：修复、重训与完整验证（2026-10-26）

本轮已完成，不是待执行计划。修复训练/部署采样不一致、候选丢失与外观分支稀有正样本训练后，系统成绩恢复；但新学习似然仍没有稳定超过同候选池的强普通评分。主线仍为24构件竞争下的跨期身份与固定历史材料点位移，未缩成单点分类。

## 1. 改了什么，实际训练了什么

- 部署与训练共同使用实测SIFT兴趣点（最多96）＋实测覆盖点（最多32），不足的SIFT配额不补满。不再把GT对应点插入训练目标集合。GT仅用于TRAIN标签、封存后评分及明确标注的条件诊断，不输入推断。
- 将未观测、物理可见但候选未覆盖、可见且覆盖分开；以材料邻域、像素足迹及估计面积测度监督对应分布。独立外观编码器＋change头，与物理描述子/状态头分别裁剪梯度。
- 4608 TRAIN episodes；hard2/soft2/soft3主seed31各固定700步，soft3另47/73各700步。共同warmstart旧contrastive31，不选择最优checkpoint。额外外观3seed各固定300步，24正＋24负平衡批次；24603已覆盖训练对中114个重涂正例（允许重复采样），物理权重哈希完全不变。
- 保留旧六方法每个身份边上的H1、H2、C1→C2精确位姿；合并SIFT/已有官方SP+LG/普通几何相容/学习提案，全部方法共享候选，不按身份top-k剪掉。标准二阶几何相容参考[SC2-PCR官方实现](https://github.com/ZhiChen902/SC2-PCR)，本地控制不是作者完整复现，也不作为原创。
- 从同一归一化对应分布构造连续位姿评分，三期用G12=G2 inv(G1)保持循环；四步联合SE3优化，实际重算可见性和近邻的线搜索保证原目标下降。

## 2. 实际数学模型及边界

记源/目标实测点为x_i、y_j，目标面积权重μ_j，源权重α_i，各自和为1，神经特征z，测量方差σ²。定义

\[
Q_\theta(j\mid i)=\operatorname{softmax}_j(z_i^Tz_j/\tau+\log\mu_j),\qquad
\bar Q_{ij}=0.9Q_{ij}+0.1\mu_j.
\]

3状态概率p0,p1,p2分别对应未观测、可见未覆盖、可见覆盖；条件覆盖概率h_i=p2/(p1+p2)。给定刚体位姿T，

\[
K_{ij}(T)=\mathcal N(Tx_i-y_j;0,\max(0.006^2,\sigma_i^2+\sigma_j^2)I),\quad
R_i(T)=\frac{\sum_j\bar Q_{ij}K_{ij}(T)}{\sum_j\mu_jK_{ij}(T)}.
\]

首回波命中掩码与最大未归一化核支持的乘积记为ν_i(T)，则

\[
E_{mat}(T)=-\sum_i\alpha_i\log[1-\nu_i(T)h_i+\nu_i(T)h_iR_i(T)].
\]

当描述子无信息Q=μ时R=1，材料项严格中性；已给定面积测度的拆分重复不改变概率质量。面积本身由近邻间距估计，不宣称任何重新采样下都精确不变。训练通过软对应交叉熵与状态监督连接该分布，并未端到端对完整24身份/SE3优化求导。

主实验E=E_普通几何与SIFT支持+0.25E_已知自由空间+0.25E_mat。没有假称这些量为已校准独立观测的完整生成似然。ν含启发式首回波门控，局部优化仍可能停在错模态；它不是连续表面的完整重建模型。重涂头独立诊断，没有门控物理匹配。

## 3. 全任务结果

同4个已见DEV场景，每外观96对象；合格要求两期身份正确且三段固定历史材料点p95误差均≤5cm。四外观共享场景/对象，不当384独立样本做显著性推断。主seed31负责全任务，其余seed仅训练与条件头诊断。

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
__TASK_ROWS__

主实验hard2/soft2/soft3和两消融均为1/16/10/16；ray为1/16/12/16。同池真身份位姿oracle为10/16/16/16，只是评分后诊断。rich当前池上限仅16，不能用继续排序期待超过它。四步联合优化降低了各自目标，但相对池选择未增加合格对象数。

保留候选＋普通强评分解释了大部分系统恢复；本轮没有单独去掉学习生成的提案，因此不能断言学习提案没有贡献。已经确认的是：新学习评分在同候选条件下未超过强普通ray。

### 事后结构对照：替换旧纹理支持，避免两套外观证据相加

完全相同已封存候选/模型，不重新训练或扫系数；移除旧SIFT纹理项。该对照在主结果后设计，明确为POSTHOC DEV，不能充当独立确认。

| 方法（去掉旧纹理项） | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
__REPLACEMENT_ROWS__

soft3相对纯几何ray在rich/changed增加3/2，但repeated减少3；并未稳定超越保留旧纹理的强普通ray。因此“只要消除重复计分就能解决”也不成立。不拼接每种条件的赢家充当可部署方法。

### 共同正确身份下的条件精度

主实验ordinary与soft3共同正确身份子集，C1→C2误差中位数（cm）：

| 外观 | 共同正确身份数 | ordinary | soft3 |
|---|---:|---:|---:|
__PAIR_ROWS__

这些局部改善不抹去错身份和总体合格损失；也不能与筛选前全任务平均混用。

## 4. 问题原因：哪些已证实，哪些只是下一假设

1. **训练/部署不一致与候选丢失属于工程问题，已修复。** 现有真实独立候选无GT正点注入；旧六方法全部6912个精确位姿逐项核对保留。仍未获得新整体收益，不能继续把差距统归因于旧候选丢失。
2. **离散覆盖不足是明确的表示问题。** 正确身份条件下30639未观测、30373可见未覆盖、12313可见覆盖。可见42686点中71.15%未被当前目标采样覆盖。三状态只是解释缺失，不生成缺失对应。旧FPS采样oracle89/96不能移用为当前SIFT＋覆盖采样的oracle。
3. **软监督干预不够强。** 训练已覆盖24603查询中，teacher熵中位数0，只有15.75%有多个权重>0.01的目标，平均有效候选1.126。soft2很接近hard2，不能仅凭二者相似判整个连续分布思想无效，也不能宣称软分布已证明优越。
4. **匹配与状态表示仍弱。** 实际独立候选且有覆盖时，soft3首选匹配36.07%、前三63.46%，hard2为35.80%/63.24%。条件覆盖Brier .2589→.1985，但常数先验约.205，因此改善幅度有限；所有true-candidate查询的状态argmax仍落在未观测类，不能声称三状态识别成功。重复纹理多模态及状态类不平衡是研究/训练假设；当前数据不能单独确认各自因果贡献。
5. **优化目标与物理误差仍错位。** 单调下降但合格数不增加，证明修好优化器不等于修好材料对应。局部目标下降、身份变化、姿态模态和材料测量应分别核查，不能仅加迭代。

## 5. 重涂辅助头：真实改善，但仍不可靠

诊断采用GT正确身份＋实际候选已覆盖的材料点对，不是完整任务的重涂检出。前后阈值均仅按TRAIN未变点5%误报率选定，不在DEV调阈值。DEV有64真重涂、12249未变对：

| 模型 | 检出数 | 召回 | 误报数 | 未变误报率 | 阳性精度 |
|---|---:|---:|---:|---:|---:|
__APP_ROWS__

seed31召回4.69%→42.19%，误报率4.66%→4.26%；但522误报对应27真阳性，阳性精度只有4.92%。稀有基率使大部分警报仍错，不能称重涂已解决。另两seed召回35.94%/50%，物理权重逐字节不变；只证明辅助分支训练有改善，不能替代主线位移收益。

## 6. 核验与可复现入口

使用现有CPU环境，没有安装包/改CUDA/删除旧证据。主实验5376任务行＋12320对应条件行，11886项评分核验；结构对照3840任务行＋8800对应条件行，10765项核验。归一、中性、面积拆分、梯度隔离通过；实际NumPy/Torch目标、12独立有限差分、联合下降通过（梯度最大误差主1.053e-6/替换1.073e-6）。预测先封存再读评分GT，训练/模型/数据/旧证据完整性见completion_integrity.json。

按顺序运行：build_material_correspondence_cache_v3 → train_material_correspondence_v3 → check_material_correspondence_v3/check_material_correspondence_inference_v3 → run_material_correspondence_v3 → train_material_appearance_v3 → score_material_correspondence_v3 → evaluate_material_correspondence_repairs_v3。事后对照run_material_correspondence_replacement_v3 → score_material_correspondence_v3 --root results/material_correspondence_replacement_v3。命令均用`.venv-real/Scripts/python.exe -B -m scripts.<入口>`。生成器拒绝覆盖已存在根，不要直接重复运行旧根。

主评分初次源码在新增--root参数前已封存，保留原件scorer_source_at_initial_scoring.py；原scoring_audit对应它，结构对照对应当前脚本，不能把源码SHA差异掩盖掉。

## 7. 主线判断与下一实现边界

- 链路：推进，部署一致采样、可解释缺失状态、同池强控制、统一分布到连续SE3与全24身份已接通。
- 收益：系统从旧弱学习模型恢复，辅助重涂头显著改善；但新学习评分没有稳定全任务净增益，当前主候选仍落后同池强普通对照。
- 创新：尚未确认。概率归一、类平衡、普通谱几何相容和SE3优化均不作原创包装。此轮实证将问题定位到“可见但离散集合缺对应”和重复纹理歧义，而非继续修评分小常数。
- 下一模块应为`material_continuous_correspondence_v4.py`（尚未实现）：让visible-uncovered分支产生受实测深度/首回波约束的连续表面对应与多模态位姿，而非仅降低其权重。共同输入仍为匿名RGB-D及旧完整候选，必要对照为同预算加密采样、普通局部连续配准、现有强ray；分离学习提案与学习评分收益。先在TRAIN冻结结构，再生成新种子confirmation。验收看完整身份＋材料点位移，在repeated不退步并超过同输入强控制；否则只保留工程修复。

局限：12程序生成箱体场景/已知相机和匿名槽位；4个评测场景均已被多轮分析，属于DEV；没有新增独立确认或真实跨期精度。四外观/对象内大量点有关联，未把点数当独立统计样本。
'''
    report=report.replace('__TASK_ROWS__',rows).replace('__REPLACEMENT_ROWS__',replacement).replace('__PAIR_ROWS__',pairtable).replace('__APP_ROWS__',approws)
    reportpath=Path('research/material_correspondence_repairs_v3_2026-10-26.md')
    reportpath.write_text(report,encoding='utf-8')
    manifest=dict(purpose='Internal descriptive research report; publisher requirements not applicable',
                  reused_dev_scenes=4,tasks_per_variant=96,unit='scene/object repeated across appearances',
                  uncertainty='No inferential intervals or independent point-sample claims',
                  transformations='Sum qualified over four scenes; appearance metrics from fixed TRAIN operating points; no exclusions',
                  source_hashes={str(p):sha(p) for p in sources},source_script_sha=sha(__file__),
                  outputs={str(p):sha(p) for p in [reportpath,ROOT/'report_source_data.json',*ROOT.glob('repair_*.png'),*ROOT.glob('repair_*.pdf')]})
    (ROOT/'report_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(table=table,report=str(reportpath)),ensure_ascii=False))


if __name__=='__main__':
    main()
