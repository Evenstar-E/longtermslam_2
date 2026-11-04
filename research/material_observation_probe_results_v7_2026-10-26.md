# V7准备实验：全24构件观测与普通部分配准（2026-10-26）

## Material Passport

- Origin Mode: local simulation and fixed-model experiment
- Verification Status: ANALYZED；数据生成与完整推断单次运行，原始量测/目标另行重建，不是独立复跑
- Scope: ONE fresh TRAIN scene × 2 acquisition conditions × 4 paired appearances × 24 assets × 4 methods = 768 task rows

**本轮已经完成“全24可解析观测＋普通匹配基线”，不是新的论文算法。丰富/重复/重涂从远景0/24提升到16/24、14/24、16/24，证明观测准备能产生实质工程收益。普通部分联合配准没有额外成功；余下失败主要集中在末期转到背面的8个构件。因此下一模型不能仅仅删除三期闭环门槛，必须处理“同一构件、不同可见材料表面”的关联与位移不确定性。**

## 完成的数据与输入边界

独立新根data/textured_material_v2_train_probe，只生成1新TRAIN，未改V1及旧192。scene seed=961126，形状/纹理采用新命名空间，已核对与V1不重叠；仍为相同程序箱体族，不是分布外泛化。纹理函数、RGB/深度噪声和渲染器保持原版，没有按结果挑纹理或挑构件。

7帧overview，另外24空间槽位×3期=72帧256×256实际近景，共79帧。相机按**空间槽位**安排，随机扰动只依赖期和槽位，不沿持续构件身份追踪；采集recipe在private supervision中。模型只读RGB-D、相机标定、公开匿名槽位，不读近景分组/身份/材质坐标。epoch2每第三个空间槽位改为背面拍摄，预先固定，不保证每个材料点三期共视。所有24构件保留全24×24身份竞争，没有将近景成功子集当全部任务。

| 条件 | 原始帧 | 原始像素数 | 构件原图ROI长边中位数 |
|---|---:|---:|---:|
| overview_only | 7 | 2,150,400 | 37.5px |
| full24 | 79 | 6,868,992 | 98.75px |

**采集预算增加约3.19倍原始像素，视图也增加；两条件不是同采集预算算法对照。** 同条件内4方法接收相同数据/评分/身份搜索图。新旧条件用同一组历史材料评价锚点，避免换评价点制造收益。ROI放大不计作新原始像素。

## 实际成功（每列24）

两期身份正确，固定历史材料点H→C1、H→C2、C1→C2三段p95全部≤5cm。

| 采集 | 方法 | flat | rich | repeated | changed |
|---|---|---:|---:|---:|---:|
| overview_only | Base geometry/SIFT pool | 0 | 0 | 0 | 0 |
| overview_only | Native RANSAC | 0 | 0 | 0 | 0 |
| overview_only | Native geometry graph | 0 | 0 | 0 | 0 |
| overview_only | Ordinary partial pair joint | 0 | 0 | 0 | 0 |
| full24 | Base geometry/SIFT pool | 2 | 14 | 12 | 14 |
| full24 | Native RANSAC | 0 | 16 | 14 | 16 |
| full24 | Native geometry graph | 0 | 16 | 14 | 16 |
| full24 | Ordinary partial pair joint | 0 | 16 | 14 | 16 |

Base pool含普通几何初值/ICP及RootSIFT初值/ICP，但也使用共同native评分，不能称“仅几何”。Native RANSAC在full24的rich/repeated/changed相对Base各2增0失，是现成前端的收益；flat反而0增2失。native_graph与RANSAC是不同候选家族，并非相互嵌套，不能预期oracle必单调。partial_joint以native_graph为底座，增加普通多期拟合端点；它与native_graph所有外观成功集合完全一致。

## 真身份候选上限（诊断，非实际成功）

| 采集 | 方法 | flat | rich | repeated | changed |
|---|---|---:|---:|---:|---:|
| overview_only | Base geometry/SIFT pool | 0 | 0 | 0 | 0 |
| overview_only | Native RANSAC | 0 | 0 | 0 | 0 |
| overview_only | Native geometry graph | 0 | 0 | 0 | 0 |
| overview_only | Ordinary partial pair joint | 0 | 0 | 0 | 0 |
| full24 | Base geometry/SIFT pool | 5 | 14 | 14 | 14 |
| full24 | Native RANSAC | 6 | 17 | 16 | 17 |
| full24 | Native geometry graph | 7 | 16 | 16 | 16 |
| full24 | Ordinary partial pair joint | 7 | 16 | 16 | 16 |

Partial joint的oracle只加入实际访问tuple生成的成对拟合，不以GT补生成未访问模式，也不跨mode混搭姿态。full24 rich/changed的native_graph已经取到本族16项上限；额外普通联合优化没有补出缺失背面位姿。RANSAC各多1个潜在达标背面候选，未变成成功。不能将候选上限当物理不可能证书。

## 三层支持：同一历史检测节点上的嵌套核查

物理：实际光线投射确认该历史材料点在C1、C2选择的两个前端视图中均可见。采样：两期实际native点中均有15mm内材料邻点。匹配：上述条件成立且实际保留的H-C1/H-C2边均正确。末列另报告真实三期闭环中材料正确的**不同历史节点**，不把重复匹配当多个独立支持。

| 条件 | 外观 | 历史节点 | 两期均物理可见 | 再满足采样 | 再满足匹配 | 正确闭环的唯一历史节点 |
|---|---|---:|---:|---:|---:|---:|
| overview_only | flat | 2006 | 390 | 11 | 0 | 0 |
| overview_only | rich | 2982 | 403 | 2 | 0 | 0 |
| overview_only | repeated | 2996 | 378 | 8 | 0 | 0 |
| overview_only | changed | 2982 | 403 | 2 | 0 | 0 |
| full24 | flat | 2141 | 892 | 40 | 0 | 0 |
| full24 | rich | 2978 | 1434 | 431 | 195 | 166 |
| full24 | repeated | 2990 | 1423 | 376 | 144 | 117 |
| full24 | changed | 2978 | 1434 | 422 | 191 | 160 |

15mm仅材料标签诊断阈值，不是算法调参或5cm任务门限。full24 rich中，1434个双期可见历史节点只有431个被两期采样覆盖，进一步仅195个被双期正确匹配。三期正确闭环有166个唯一源节点；≥3个正确双期匹配与≥3个正确三期闭环均覆盖15/24构件。**对这个条件，只去掉第三条匹配边并不会自动补出更多有3个正确同源材料支持的构件。** repeated对应15→14，changed15→15。允许不同两期使用不同材料轨迹仍有意义，但不能宣称其必然带来收益。

coverage.csv同时保存固定共同评价锚点的全视图物理可见/采样覆盖，和实际native源点的上述嵌套计数。它们查询点不同，不能把固定锚点sampled_three与实际cycles直接组成同一个漏斗。所有任务、构件和失败都保留，无筛掉背面病例。

## 余下8个构件具体是什么困难

| 外观 | C2采集 | 构件数 | 两期身份正确 | 成功 | 候选上限 |
|---|---|---:|---:|---:|---:|
| rich | 正面近景 | 16 | 16 | 16 | 16 |
| rich | 背面近景 | 8 | 4 | 0 | 1 |
| repeated | 正面近景 | 16 | 14 | 14 | 16 |
| repeated | 背面近景 | 8 | 4 | 0 | 0 |
| changed | 正面近景 | 16 | 16 | 16 | 16 |
| changed | 背面近景 | 8 | 4 | 0 | 1 |

rich/changed正面16个构件全部成功，背面8个全部失败；repeated正面另有2个身份错误，背面仍0成功。这里“正面/背面”来自预设相机方向，源几何、遮挡和观察角仍变化，子组不是随机分配，不能推广为所有正背面任务规律。

背面rich的8项中，只有asset8有1个RANSAC合格候选但错身份；其余7项当前池无好位姿。asset4/14/20的历史native节点在所选C1/C2视图中双期共同可见数为0；其他仍有少量侧面共视但多数未采到正确目标材料点。全可用视图存在少量共视不代表top2前端能够抓到、匹配到。不能把所有背面失败一概称物理不可观测，也不能将“看见同一个构件”当“看见同一块材料”。

## 实现的数学基线与新算法边界

partial_joint使用三个**独立匹配边集合**M01、M02、M12（各可包含不同材料点），直接优化：

\[
E(G_1,G_2)=\sum_{(x,y)\in M_{01}}w\rho(\|G_1x-y\|^2)
+\sum_{(x,y)\in M_{02}}w\rho(\|G_2x-y\|^2)
+\tfrac12\sum_{(x,y)\in M_{12}}w\rho(\|G_2G_1^{-1}x-y\|^2).
\]

这是普通稳健多期配准，不是带材料landmark的BA，更不是新持续材料模型。最多4个同池初值/4步12DoF IRLS，支持缺C12和各边材料集合不同；置信只作拟合权重，不冒充校准概率。实际4444次拟合、62条最终对象预测采用新端点，但没有任务成功增量，不能说未接入，也不能用“有数值活动”冒充创新。

所有方法相同geometry/SIFT/ray＋native支持评分。为固定新数据的CPU预算，每构件从192几何点取32个实际射线查询、目标只用公开可见像素最多的2视图；native仍沿用V6的2视角/64实际像素/2图像对/96对应等预算。此评分观测预算与V6不同，所以跨轮数字不作为纯算法增量。公共身份图两轮各64状态，普通坐标Hungarian初始保持双射；非全局最优。

## 原因分类与主线决策

- **工程输入改进已经有证据：** 同一新场景从仅远景到full24，在固定现成模型下成功数明显提高。不能将收益归给尚未实现的新模型；也不能以这1个TRAIN场景宣称泛化或统计显著。
- **普通联合优化没有新增成功：** full24丰富外观已选中本族全部可达标候选；剩余主要缺准确背面姿态。继续加步数不是优先项。
- **研究缺口更加具体：** 纹理对应应只约束持续可见材料；同构件不同表面需要几何/可见性解释与未知状态，不能强行做同材料正匹配。当前输入下有些姿态仍可能多解，模型应保留不确定性，不能补造不可见纹理。
- **下一项直接做核心模型，暂停继续扩这份准备数据：** 在冻结full24观测与native边上实现部分材料对应的容量/空匹配/未知状态，联合连续SE3；与同容量普通匹配＋本轮普通多期配准、同输入普通可见性融合比较。候选必须真正生成新位姿或减少错误接受，不能只后置加小项。先检查它是否能利用现存侧面/两期支持，而非默认删闭环就会恢复8背面构件。具体实施与已有方法边界见[新契约](material_visibility_partial_tracks_contract_v7b_2026-10-26.md)；旧V7契约作为冻结历史不改写。
- **创新性：尚未确立。** 全24链路推进、工程收益明确、标准前端增益明确；新持续材料模块仍未实现，未获得其独立收益。

## 核验与资源

数据生成39.32s；原生前端253.99s；普通基线92.59s，均阶段墙钟。没有新训练、安装、GPU改动或删除；新数据/结果独立保存。

完整评分13709项，768行；公开原始像素/回投/同期待合并/边/4444拟合端点/所有选中目标重建541044项，最大目标误差3.55e-15。新普通配准12列解析J有限差分最大3.09e-10，实际不同材料集/缺C12时下降检查通过。两个条件的原生量测标签污染检测均为0。网络结果复用冻结缓存，未独立全重跑；没有新增真实跨期数据精度和独立确认集。

## 图与复现

![全方法采集对照，实际成功而非oracle](../results/material_observation_probe_v7/acquisition_task_comparison.png)

![同源节点三层支持和正背面失败](../results/material_observation_probe_v7/coverage_and_view_failures.png)

命令：python -B -m scripts.build_material_observation_coverage_v7 → cache_material_observation_native_v7 → check_material_partial_pair_registration_v7 → run_material_observation_registration_v7 → score_material_observation_probe_v7 → audit_material_observation_probe_v7 → report_material_observation_probe_v7。生成根存在时拒绝覆盖。新配置分别为textured_material_dataset_v2.json、material_observation_probe_v7.json、material_observation_registration_v7.json。
