# V4 实测表面内连续材料对应：完整实验结果（2026-10-26）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-26
- Verification Status: ANALYZED（代码实验完成并封存，未做独立完整复跑）
- Version Label: material_continuous_correspondence_v4

## 执行完成了什么

新建局部Siamese密集特征场，在每个源查询的三个粗锚点周围，从真实RGB-D取得12×12局部网格。连续分支对峰值邻域取概率均值，投影到有三个实际测量顶点支持的局部三角片；缺失/退化时回退实测点。完整实现和数学见`material_continuous_correspondence_protocol_v4_2026-10-26.md`。三角片是分片平面插值假设，**不证明顶点之间的真实材料表面已知**，细薄/转角可能被近似；首回波通过既有ray位姿评分约束，并非新增逐三角形内部首回波检验。局部分类使用固定网格，尚未对重复落在同一深度像素的格点做面积去重，不能继承V3给定面积测度拆分不变性或称完整连续生成模型。

TRAIN缓存4608组、886182局部对、46343有支持正对。原62200个TRAIN“可见但未采中”查询中，14428个（23.20%）在固定三个粗锚点周围找到材料邻域支持；不是完整任务检出率。两个seed31/47各900固定步，训练耗时51.48s，未按DEV选checkpoint。主seed31全任务，47仅TRAIN回放；不声称多seed任务复现。

主推断1764.99s，另有评分前补充的Old-only shared search控制：4个旧DEV×4外观×8方法×24对象=3072行。额外控制不增加新对应/位姿，仅用同一V4完整旧假设搜索，避免把更充分搜索归给网络；它在任何DEV真值评分之前单独封存于results/material_continuous_retention_control_v4。没有新学习评分，全部用同一个冻结几何＋SIFT＋ray评分；保留旧完整池、六方法及V3ray整组假设。每族用“旧池＋自身新增”，另有全并集控制，**不是所有分支同候选池**。ncc/random/dense/surface输入及32查询×3粗锚点×144格点预算相同，普通geo为3旧位姿×6投影迭代；不声称等FLOPs。random仅新增局部网络随机，粗匹配仍是共同V3模型；ncc内部名指3×3向量L2归一相关，输入RGB已按整块归一，不冒称专门调优的标准NCC。

## 全任务效果

每外观96对象。合格=两期身份正确＋固定历史材料点三段p95误差均≤5cm；四外观共享对象/场景，不能当384独立样本。

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| V3 strong ray | 1 | 16 | 12 | 16 |
| Old-only shared search | 1 | 16 | 12 | 16 |
| Normalized patch grid | 1 | 16 | 12 | 16 |
| Random local grid | 1 | 16 | 12 | 16 |
| Learned grid | 2 | 16 | 12 | 15 |
| Learned continuous | 1 | 16 | 12 | 15 |
| Geometric continuous | 2 | 16 | 12 | 15 |
| All proposals + ray | 2 | 16 | 12 | 16 |

同模型连续surface−离散dense的净变化为[-1, 0, 0, 0]；相对旧搜索一致控制retained为[0, 0, 0, -1]；相对强原baseline为[0, 0, 0, -1]；相对普通连续geo为[-1, 0, 0, 0]。这些比较须同时报告，不用单个条件赢家替代整体方法。

## 每族真实身份位姿池的事后上限（oracle）

下表只判断“这一族是否生成过能量化的位姿”，使用GT选择，不是方法成功率；每族自己的池独立计算。oracle函数以已知3cm/4cm正交运动误差得到5cm，并与原评分函数逐项一致（误差0）。

| 位姿池 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| V3 strong ray | 10 | 16 | 16 | 16 |
| Old-only shared search | 10 | 16 | 16 | 16 |
| Normalized patch grid | 11 | 16 | 16 | 16 |
| Random local grid | 10 | 17 | 16 | 17 |
| Learned grid | 11 | 17 | 16 | 17 |
| Learned continuous | 11 | 18 | 16 | 17 |
| Geometric continuous | 13 | 17 | 16 | 17 |
| All proposals + ray | 13 | 18 | 16 | 17 |

## 机制诊断：原采样未覆盖的可见材料点

GT身份条件下，每个查询允许从3个粗锚点细化结果里取最好的一个，误差≤2.5cm即有可用模态；**不是实际选中对应的精度，更不是全局身份正确率**。括号为查询数，多个点与时期对相关。

| 3模态候选来源 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| Normalized patch grid | 1.68% (4284) | 6.19% (3587) | 5.76% (3628) | 5.82% (3610) |
| Random local grid | 1.98% (4284) | 5.83% (3587) | 5.95% (3628) | 5.51% (3610) |
| Learned grid | 2.31% (4284) | 10.01% (3587) | 9.56% (3628) | 9.34% (3610) |
| Learned continuous | 2.40% (4284) | 10.04% (3587) | 9.62% (3628) | 9.39% (3610) |

连续分支相对同网络离散点实际改变了1095808/2654208个输出（41.29%），不是空接线。全身份边上的平均移动量如下；没有用GT筛选这些位移：

| 外观 | 非零变化数/输出数 | 平均连续修正(cm) |
|---|---:|---:|
| flat | 241480/663552 | 0.321 |
| rich | 285639/663552 | 0.336 |
| repeated | 284458/663552 | 0.324 |
| changed | 284231/663552 | 0.335 |

TRAIN固定160组回放，正412/负1280：随机网络局部峰落入teacher邻域40.05%，seed31为48.06%、47为47.09%。这是训练拟合信号，不是泛化证据。null边缘概率正/负均值seed31约0.542/0.423，区分较弱；不能拿“145个联合类别argmax常为null”误判为整个nonnull边缘塌缩。推断按原协议未使用未校准null硬筛点，故本轮不验证无对应判别头带来的收益。

## 逐条件得失与身份变化

| 外观 | 比较 | 新增合格 | 丢失合格 | 共同正确身份数 |
|---|---|---:|---:|---:|
| flat | baseline→ncc | 0 | 0 | 8 |
| flat | baseline→random | 0 | 0 | 8 |
| flat | baseline→dense | 1 | 0 | 9 |
| flat | baseline→surface | 0 | 0 | 9 |
| flat | baseline→geo | 1 | 0 | 10 |
| flat | baseline→union | 1 | 0 | 7 |
| flat | dense→surface | 0 | 1 | 11 |
| flat | baseline→retained | 0 | 0 | 10 |
| flat | retained→ncc | 0 | 0 | 8 |
| flat | retained→random | 0 | 0 | 8 |
| flat | retained→dense | 1 | 0 | 9 |
| flat | retained→surface | 0 | 0 | 9 |
| flat | retained→geo | 1 | 0 | 10 |
| flat | retained→union | 1 | 0 | 7 |
| rich | baseline→ncc | 0 | 0 | 28 |
| rich | baseline→random | 0 | 0 | 29 |
| rich | baseline→dense | 0 | 0 | 28 |
| rich | baseline→surface | 0 | 0 | 27 |
| rich | baseline→geo | 0 | 0 | 29 |
| rich | baseline→union | 0 | 0 | 27 |
| rich | dense→surface | 0 | 0 | 27 |
| rich | baseline→retained | 0 | 0 | 29 |
| rich | retained→ncc | 0 | 0 | 28 |
| rich | retained→random | 0 | 0 | 29 |
| rich | retained→dense | 0 | 0 | 28 |
| rich | retained→surface | 0 | 0 | 27 |
| rich | retained→geo | 0 | 0 | 29 |
| rich | retained→union | 0 | 0 | 27 |
| repeated | baseline→ncc | 0 | 0 | 17 |
| repeated | baseline→random | 0 | 0 | 16 |
| repeated | baseline→dense | 0 | 0 | 17 |
| repeated | baseline→surface | 0 | 0 | 18 |
| repeated | baseline→geo | 0 | 0 | 18 |
| repeated | baseline→union | 0 | 0 | 16 |
| repeated | dense→surface | 0 | 0 | 18 |
| repeated | baseline→retained | 0 | 0 | 18 |
| repeated | retained→ncc | 0 | 0 | 17 |
| repeated | retained→random | 0 | 0 | 16 |
| repeated | retained→dense | 0 | 0 | 17 |
| repeated | retained→surface | 0 | 0 | 18 |
| repeated | retained→geo | 0 | 0 | 18 |
| repeated | retained→union | 0 | 0 | 16 |
| changed | baseline→ncc | 0 | 0 | 28 |
| changed | baseline→random | 0 | 0 | 29 |
| changed | baseline→dense | 0 | 1 | 27 |
| changed | baseline→surface | 0 | 1 | 29 |
| changed | baseline→geo | 0 | 1 | 29 |
| changed | baseline→union | 0 | 0 | 27 |
| changed | dense→surface | 0 | 0 | 28 |
| changed | baseline→retained | 0 | 0 | 30 |
| changed | retained→ncc | 0 | 0 | 28 |
| changed | retained→random | 0 | 0 | 29 |
| changed | retained→dense | 0 | 1 | 27 |
| changed | retained→surface | 0 | 1 | 29 |
| changed | retained→geo | 0 | 1 | 29 |
| changed | retained→union | 0 | 0 | 27 |

## 原因与创新边界

本轮把“对应表示/位姿提案”与“评分”分开，能检查新增支持是否转化成物理量化，而非再调评分权重。连续分支与dense使用完全相同学习特征/局部分数，差异只有观测域内连续坐标；与geo比较普通连续配准能否解释收益。训练正例只有5.23%的局部对，粗锚点邻域对原未覆盖查询只覆盖23.20%左右，网络没有全表面搜索能力；不能把局部补点当成完整未知表面建模。

粗到细密集匹配和连续细化早有先例：[LoFTR](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)、[RoMa](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html)。这里没有运行其作者模型，不能称胜过已发表强系统，也不能把小CNN、soft-argmax、表面投影或标准几何相容包装主创新。是否值得继续须依据全任务、dense与geo三个对照的共同结论；新确认数据与真实跨期验证仍未完成。

## 完成后的判断：暂停局部连续细化作为主创新

1. 额外旧池相同搜索retained与baseline四组成功集合相同，所以没有把本次收益误归给更完整的旧假设搜索。
2. dense为2/16/12/15，surface为1/16/12/15；连续坐标没有优于同网络格点，且丰富纹理原未覆盖查询的3模态召回只从359/3587到360/3587。原有局部表示更细但不能直接声称解决了材料关联。
3. 普通geo为2/16/12/15，全union为2/16/12/16。union新增flat合格对象是test_000/11，与geo同一个，误差5.43→2.36cm；dense的flat新增则是test_000/14，不能混称各方法恢复相同对象。union这一项新增可以由普通geo提供，不是连续学习独占收益。
4. surface的rich真身份池从16到18，多出的test_000/7与test_001/10都是无近景对象；这是真实的新提案信号，但最终仍错身份。不能因它们未选中就说新表示完全无用，也不能把18当实际成功。
5. 结论：链路和问题定位推进，主算法整体收益没有增加；不继续扫局部CNN步数/连续均值窗口，也不把此轮当论文创新确认。4场景都是旧DEV，不能推出所有连续对应方法不可能有效。

### 四个失败的事后可行身份交换诊断

这是GT指导的诊断，不是新算法成绩，也没有改已封存预测。把指定对象改为真实目标槽位，同时交换占用这些槽位的其他对象以保持双射；对受影响对象穷举各自池内两期位姿组合，重算同一完整周期目标。不是全局最优求解证明。

| 场景/外观/对象 | 当前对象能量 | 正确身份最小池能量 | 可行交换前→后总能量 | 正确身份最低能量解误差(cm) |
|---|---:|---:|---|---:|
| test_000/rich/7 | 4.3192 | 4.6529 | 8.6763→9.0159 | 4.04 |
| test_001/rich/10 | 4.3549 | 4.5646 | 8.5246→8.8442 | 5.94 |
| test_000/changed/11 | 3.0252 | 2.8790 | 7.0605→7.0421 | 2.14 |
| test_001/flat/23 | 4.0300 | 4.0386 | 8.3644→8.5436 | 5.92 |

**工程原因的直接证据：**changed/test_000/11存在合法身份交换使受影响总能量7.0605→7.0421，并可得到2.14cm达标位姿；当前“先用独立边最小值定身份、再固定身份选三期姿态”尚未达到这个交换邻域的局部最优。该修复应作为普通联合搜索对照，不称原创。

**模型问题的直接证据：**rich/test_000/7的正确身份最低能量解已达4.04cm，但能量4.6529仍高于错误身份的4.3192；对应可行身份交换总能量也升高0.3396。rich/test_001/10正确身份最优解误差5.94cm，池内另有合格解但能量更高4.6533；错身份只要4.3549。普通目标在这些比较中确实更偏爱错误物理解释，不能只靠增加迭代解决。这不证明全局最优一定错，却直接支持需要改材料关联证据，而不只修求解器。

下一执行边界见`material_joint_identity_material_next_v5_2026-10-26.md`：先补同候选普通完整目标身份/位姿交换，建立工程底线；研究分支改多期共享材料对应证据，和同特征普通融合、独立材料因子及普通联合配准比较。当前V4局部细化不再作为主创新候选。

## 工程问题、核验与产物

首次训练启动在任何训练/输出根创建之前因Windows默认GBK读UTF8封存JSON报错，改显式UTF8后完成；没有把失败算成训练结果。普通连续几何对照是主要计算开销。单独试验batch投影与32个标量投影误差0、速度仅1.27×，**未接入本次已冻结推断**；不能将其当主线算法贡献。

核心检查包括有界三角片投影、离散/连续差异、缺口回退、跨槽位支持、null/有限梯度；评分审计66246项，逐族逐项核对旧六方法和V3ray精确位姿，baseline预测逐字节一致。完整性见completion_integrity.json。原V3/旧192/原数据保持，未安装依赖或改GPU环境。

本报告图为描述性总数与明确标注的oracle，没有独立样本显著性声明或置信区间；代码生成原始表和来源哈希。所有4场景反复使用，属于DEV；程序生成箱体、已知相机/匿名空间槽位、每景4近景等局限不变。结果不代表真实工程部署精度。

## 可复现入口

`.venv-real/Scripts/python.exe -B -m scripts.<模块>`，顺序：check_material_continuous_v4 → build_material_continuous_cache_v4 → train_material_continuous_v4 → run_material_continuous_v4 → score_material_continuous_v4 → diagnose_material_continuous_v4 → report_material_continuous_v4。另diagnose_material_continuous_train_v4是TRAIN回放。生成器拒绝覆盖已有根；不要重跑已封存根或将已有DEV改称新test。
