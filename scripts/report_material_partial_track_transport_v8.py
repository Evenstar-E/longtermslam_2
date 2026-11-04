"""Concise evidence-led report for the implemented core and normalization repair."""
import json
from pathlib import Path

def main():
    roots=[Path('results/material_partial_track_transport_v7'),Path('results/material_partial_track_fixed_measure_v8')];data=[json.loads((r/'scores.json').read_text(encoding='utf-8')) for r in roots];methods=['ordinary_union','visibility_pair','history_union','shared_transport'];names=['普通共同候选','独立可见性容量匹配','只追加历史材料','持续材料回改'];variants=['flat','rich','repeated','changed'];lines=[]
    for stage,d in zip(['原归一化 V7','固定观测测度 V8'],data):
        for m,name in zip(methods,names):
            vals=[next(r['qualified'] for r in d['summary'] if r['variant']==v and r['method']==m) for v in variants];lines.append('|'+stage+'|'+name+'|'+'|'.join(str(x) for x in vals)+'|')
    comparisons=[]
    for r in data[1]['comparisons']:
        if r['candidate']=='shared_transport' and r['baseline'] in ['ordinary_union','visibility_pair','history_union']:comparisons.append(f"|{r['variant']}|{r['baseline']}|{len(r['gained'])} / {len(r['lost'])}|{r['gained']}|{r['lost']}|")
    counters=[]
    for root,d in zip(roots,data):
        seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));dd=[json.loads((root/f'{v}_diagnostics.json').read_text(encoding='utf-8')) for v in variants];counters.append(f"- {root.name}：{seal['elapsed_s']:.2f} 秒，{sum(q['counters']['cartesian_pairs'] for q in dd):,} 个完整位姿对比较，{d['fit_count']} 次连续拟合，其中 {d['fits_objective_improved']} 次内部目标下降；{d['checks']} 项评分/一致性核对，选中目标重建最大误差 {d['max_reconstruction_error']:.3g}。")
    text='''# 持续材料核心已实现：完整收益检验与固定观测测度修复

2026-10-26。本轮遵循“先判断缺陷是否阻断主线，再决定修复优先级”。实际完成两个阶段，每阶段 24 对象 × 4 外观 × 4 方法 = 384 条主要比较记录；每方法 96 项任务。另有离散阶段与旧方法对照，每阶段总计 864 行。仅一个 TRAIN 场景，四外观/对象相关；没有新网络训练，也没有独立泛化确认。原历史 192 项任务的结果未修改，本轮不是把 96 项新 TRAIN 任务称为旧 192 项重跑。

## 1. 实现与取舍

新增材料容量、空匹配、可见性分型、三期部分材料簇以及连续 12DoF 位姿更新。持续版允许 C2 证据回改 H–C1 对应；对照保留独立匹配和只能追加历史材料。源文件是 [核心模型](../inspection_decisions/material_partial_track_transport_v7.py)，完整公式和文献边界见 [模型说明](material_partial_track_transport_model_v7_2026-10-26.md)。这次不再扩准备数据或反复检查现成方法能否运行。

保留全部旧 Base/RANSAC/graph 候选，同时给所有方法相同的标准 PCA24 初值。每个访问身份三元组完整比较两期候选笛卡尔积；合法的独立匹配下界只跳过不可能胜出的共享聚类计算。全部旧身份状态保留，再加入公共状态与邻居；这是有限搜索图，不是身份全局最优。共同连续起点和四类优化端点再次共享，普通方法也能使用新模型端点。同输入/最终候选不等于同 FLOPs。

## 2. 主指标结果

每格分母均为 24。合格要求两期身份都正确，且三个时间段固定历史材料锚点的 P95 位移误差均不超过 5 cm。未删除背面八对象。

|阶段|方法|无纹理|丰富纹理|重复纹理|局部重涂|
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(lines)+'''

修复后的逐对象配对比较，增/失不能只看总数：

|外观|持续材料相对的对照|新增 / 丢失|新增对象|丢失对象|
|---|---|---:|---|---|
'''+ '\n'.join(comparisons)+'''

全部对象明细、正确身份条件下的误差、背面分组、候选 oracle、实际选用端点来源见 [V8 评分](../results/material_partial_track_fixed_measure_v8/scores.json) 和 [逐任务表](../results/material_partial_track_fixed_measure_v8/task_metrics.csv)。两阶段原始结果都保留，不只展示修复后的版本。

![全部方法的严格合格数](../results/material_partial_track_fixed_measure_v8/task_comparison.png)

## 3. 为什么这次应该先修公式

原版相对普通方法在 rich/repeated/changed 分别丢失 4/3/4 项，没有新增。其中 9 项是身份正确却选错位姿：**正确位姿已在公共候选中，但新目标函数更喜欢错误位姿**。同候选的完整枚举核对证明，抽查的 4 个三元组共 5,621 个位姿对，使用/不使用下界剪枝的最小值完全一致。这不是靠再加搜索轮数就能解决的现象。

原式为 $E=1-V/Z(G)+0.25F$，其中 $V$ 是材料匹配总增益，$F$ 是首回波自由空间冲突，$Z(G)$ 随位姿预测出的可见点数变化。rich 对象 0：正确解误差 0.00663 m，$V=69.50,Z=360.5,F=0.1154,E=0.8361$；错误解误差 0.67457 m，$V=44.44,Z=184,F=0.2475,E=0.8204$。错误解支持更少、冲突更多，却因分母变小获胜。9 个同身份新增失败都具有这种原始增益与冲突双重劣势。

因此 V8 只把分母改为与姿态无关的实际观测总量：

$$Z_0=6+\\sum_{e<f}\\omega_{ef}(N_e+N_f),\\qquad E_0=1-V/Z_0+0.25F.$$

相同身份、相同观测下，若 $V_{bad}<V_{good}$ 且 $F_{bad}>F_{good}$，必有 $E_0(bad)>E_0(good)$。该修复不改变匹配代价、unknown 的空对应语义、材料容量或位姿更新。冻结同一批旧端点做反事实比较，9/9 同身份排序反例全部修正，见 [反事实检查](../results/material_fixed_measure_check_v8.json)。这只证明修复机制；整条链路是否受益由上表完整重跑判断。

修复是在看到本 TRAIN 失败后提出，不是预注册的独立确认。跨版本公共候选生成规则相同，但连续端点/新身份状态可因新目标变化，所以跨版本成绩不等于严格固定最终池的单因素消融；同端点反事实专门隔离归一化效应。普通归一化修正是正确性修复，不包装成论文创新。

## 4. 工程与研究边界

已推进的链路：新核心真正在全对象任务里执行，比较了“独立”“历史追加”“后期回改”，而非只验证旧方法。已经查明的建模错误：条件可见点归一化会奖励减少需要解释的观测支持。这个反例比继续扩大搜索预算更值得先处理。

当前表示仍是**稀疏实测点的材料簇＋连续位姿**，不是连续未知表面重建。unknown 没有伪造纹理或伪造材料锚点，缺支持时也没有校准的拒绝机制。原生 SP/LG 权重冻结；没有在此基础上重新训练新的几何纹理骨干。

共享材料约束是否有独立贡献，以同版本的独立容量匹配、历史追加和普通配准为准；修复失分、目标函数下降或选中更多新端点均不足以确认创新。部分匹配同步、多视配准等已有方法，原始出处在模型说明中。作者完整强 3D 配准模型尚未运行，不能声称超过它们。

本轮明确结论：V8 持续材料 3/16/14/16，与独立可见性匹配、历史追加的**合格集合完全相同**。相对同池普通方法，flat 增加对象 2/12/13 三项，另外三外观无新增/丢失；但独立控制已解释这三项，不能算共享创新。另一个旧候选受限 Base 控制在 flat 原本有 2 项（13/19）；新结果对它是新增 2/12、丢失 19，净增 1，而非对所有既有方法都净增 3。两者不是同最终候选池的纯消融。

共同正确身份集合上，持续版相对同池普通版的最坏材料误差配对中位变化为 rich -0.249 mm、repeated -0.722 mm、changed +0.017 mm；相对独立匹配四外观均为 0。不能拿各自不同正确身份子集的中位数声称大幅精度提升。背面八对象在所有方法/外观仍全部失败。rich/changed 共同池 oracle 为 17/24，正面已 16/16 成功，说明背面仍只有一项有合格候选，其余七项不能靠最终评分追回。flat/repeated 的共同池 oracle 为 8/16，也不是实际成功率。

因此：**主线链路已推进，标准观测模型有局部收益，持续共享材料尚无独立创新收益；停止把当前稀疏材料簇的回改迭代当作主创新候选。** 1088 次连续拟合中 982 次内部目标下降，但各方法连续前后的合格集合没有净变化，继续增加步数不是优先项。

## 下一执行优先级

下一新模块拟为 `material_surface_patch_registration_v9.py`，本轮尚未实现。重点从“在稀疏点匹配上再加共享约束”转到“从实际可见表面块生成连续材料对应和新位姿”：联合表面几何、局部纹理与首回波曝光，只使用真实观测，未见表面保持未知。优先针对 rich/changed 背面七个当前缺好候选的对象，同时仍评价全部 24 对象及四外观，不能只报失败子集。

必须提供相同表面块/点数/视图与初值的普通几何纹理配准对照，区分加密观测的工程收益与持续材料约束收益。验收顺序是：公共候选中是否实际新增合格位姿 → 是否变成完整身份与位移成功 → 新模型是否胜过同表示普通控制。只提高 oracle、增加采样、改善内部损失都不足以确认创新。若普通控制解释全部收益，就停止该共享机制的创新主张，而不是再换归一化系数。独立多场景及已发表强 3D 方法比较仍是后续确认要求。

## 5. 验证与复现

703 项容量/空匹配/部分轨迹/单调性/梯度检查通过，12DoF 雅可比最大差分误差 3.90e-10。固定测度另有 25 项公式与算子检查。两阶段先封存全部预测再评分；V8 是明确的 TRAIN 结果驱动修复。

'''+ '\n'.join(counters)+'''

主要命令：

```powershell
.venv-real/Scripts/python.exe -m scripts.check_material_partial_track_transport_v7
.venv-real/Scripts/python.exe -m scripts.run_material_partial_track_transport_v7
.venv-real/Scripts/python.exe -m scripts.score_material_partial_track_transport_v7
.venv-real/Scripts/python.exe -m scripts.diagnose_material_partial_track_transport_v7
.venv-real/Scripts/python.exe -m scripts.check_material_fixed_measure_v8
.venv-real/Scripts/python.exe -m scripts.run_material_partial_track_fixed_measure_v8
.venv-real/Scripts/python.exe -m scripts.score_material_partial_track_fixed_measure_v8
```

已生成的实验目录拒绝覆盖；以上完整重跑需复制配置到新输出目录，不删除唯一证据。原 V1/V4/V5/V6/V7 证据、数据和权重以及旧 192 项评分哈希由完成核验脚本检查。图保留全部零值与全部方法，原始绘图数据一并提供，不用相关对象虚构统计置信区间。
'''
    Path('research/material_partial_track_transport_results_v8_2026-10-26.md').write_text(text,encoding='utf-8');print('REPORT WRITTEN')

if __name__=='__main__':main()
