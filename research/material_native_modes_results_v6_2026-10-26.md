# V6完整结果：原生局部匹配与跨期材料模式（2026-10-26）

## Material Passport

- Origin Mode: local implementation and DEV experiment
- Verification Status: ANALYZED；全量推断单次运行，目标与原始观测另行重建；未独立复跑全部网络
- Scope: 4 repeatedly used procedural DEV scenes × 4 paired appearances × 24 assets × 7 methods = 2688 scored rows

**本轮没有确认主算法创新。强普通前端仅在rich/changed各增加1个达标候选；三种材料模式与普通joint都未提高成功数，共享模式未被最终选中。当前实验的关键缺口是原始观测尺度、前端对应质量、三期闭环覆盖和目标函数的辨别能力，不能继续把“换共享图公式”当创新收益。**

## 做了什么，边界是什么

原生SuperPoint检测＋LightGlue图像匹配已真实执行，使用本地固定作者源码/权重；不是在任意密集点采样描述子冒充原生检测。匿名构件ROI由公开深度/槽位获得，不输入跨期身份。两期普通RANSAC、普通距离兼容图产生新连续SE3并加普通ICP；全部旧V4 union姿态原样保留。三期模型从真实图像匹配闭环产生离散材料模式，模式直接拟合G1/G2，不是后置小评分。

2视角/构件、每view64个原始深度像素、ROI长边256、每身份边96对应、每tuple64轨迹、最多4模式/4步共同IRLS。两视角间仅匹配描述子相似选择的2/4图像对。相同预算上限不等于各方法实际FLOPs或拟合数相同。原生前端760.26s，六方法主推断788.82s，评分前交集补充40.34s；四CPU进程的阶段墙钟时间，不含完整审计。没有新训练。

全方法相同普通geometry/SIFT/ray＋原生唯一节点支持评分，统一有限身份图；old_pool是旧姿态加**本轮新评分**，不是原封不动V5。native_graph之后三期方法只额外开放本族成对模式，未混搭跨模式的G1/G2。共享product的补充intersection在读GT前设计、另根封存；两者同时报告，不按成绩选版本。详细数学见[协议](material_native_modes_protocol_v6_2026-10-26.md)。

## 实际成功数（每列96）

成功要求两期身份正确且固定历史材料锚点H→C1、H→C2、C1→C2三段误差p95均≤5cm。

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| Old pool + native score | 2 | 16 | 10 | 16 |
| Native RANSAC | 1 | 16 | 8 | 16 |
| Native geometry graph | 1 | 16 | 8 | 16 |
| Ordinary joint fit | 1 | 16 | 8 | 16 |
| Cycle + independent modes | 1 | 16 | 8 | 16 |
| Shared product modes | 1 | 16 | 8 | 16 |
| Shared intersection modes | 1 | 16 | 8 | 16 |

V5旧普通joint为2/16/12/16。本轮old_pool为2/16/10/16，说明新原生支持评分自身就使repeated退步2项；新增RANSAC/graph后repeated再降2项。不能把全部退步归给新模式。old_pool→native_graph的flat是1增2失，repeated是0增2失，rich/changed均无成功变化。joint_all/cycle_independent/shared_product/shared_intersection相对native_graph的成功集合完全相同。

## 候选容量诊断（GT已知时的上限，不是方法实际成绩）

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| Old pool + native score | 13 | 18 | 16 | 17 |
| Native RANSAC | 13 | 19 | 16 | 18 |
| Native geometry graph | 13 | 19 | 16 | 18 |
| Ordinary joint fit | 13 | 19 | 16 | 18 |
| Cycle + independent modes | 13 | 19 | 16 | 18 |
| Shared product modes | 13 | 19 | 16 | 18 |
| Shared intersection modes | 13 | 19 | 16 | 18 |

新增rich/changed的同一案例validation_000/asset2得到合格候选；其余大部分池缺口仍在。共享/独立/普通joint额外模式未进一步提高合格上限。该oracle只允许族内unary笛卡尔对，或同一tuple的一个成对模式；禁止跨mode任意混搭。有限图不保证全局身份最优。

## 身份正确数（不能代替位移达标）

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| Old pool + native score | 15 | 34 | 19 | 32 |
| Native RANSAC | 8 | 32 | 12 | 33 |
| Native geometry graph | 10 | 35 | 14 | 32 |
| Ordinary joint fit | 10 | 35 | 14 | 32 |
| Cycle + independent modes | 10 | 35 | 14 | 32 |
| Shared product modes | 10 | 35 | 14 | 32 |
| Shared intersection modes | 10 | 35 | 14 | 32 |

## 机制是否真的被检验

| 外观 | 采集 | 构件数 | ≥3实际闭环 | ≥3材料正确闭环 | 真身份边对应数 | 其中材料正确 |
|---|---|---:|---:|---:|---:|---:|
| flat | 无近景 | 80 | 0 | 0 | 1791 | 59 |
| flat | 有近景 | 16 | 0 | 0 | 291 | 32 |
| rich | 无近景 | 80 | 3 | 0 | 4872 | 501 |
| rich | 有近景 | 16 | 13 | 13 | 1476 | 753 |
| repeated | 无近景 | 80 | 5 | 0 | 4685 | 630 |
| repeated | 有近景 | 16 | 14 | 14 | 1541 | 870 |
| changed | 无近景 | 80 | 2 | 0 | 4904 | 500 |
| changed | 有近景 | 16 | 13 | 12 | 1449 | 730 |

材料正确的定义是三期本地材料坐标最大两两距离≤15mm，仅封存后的诊断标准，不反馈推断或调参。它不是精确同点标签，也不是本轮成功门限。真身份匹配不等于同材料对应，实际闭环同样可能闭合在错误纹理位置。

**每列80个无近景构件的原图ROI长边中位39px，另外16个有近景的中位87.5px。** 在rich/repeated/changed下，无近景构件虽每期至少约128个检测点（两view总数），但没有任何构件获得≥3条上述材料正确三期轨迹；有近景分别13/14/12项。ROI放大至256不会增加独立实测深度，点数达到上限也不代表细节可用。采集分组不是随机干预（近景对象选择亦含构件属性），因此这些是强关联证据，不足以把所有失败因果归给像素数；也不能断言所有原始视角不可观测。当前top2视角、图像对选择、native检测/剪枝和完整三期闭环要求都可能继续丢支持。

全tuple生成拟合数：{'joint_all': 5888, 'cycle_independent': 550, 'shared_modes': 293, 'shared_intersection': 306}。最终2688条预测中，普通joint选中1个自己新拟合模式，独立/共享product/共享intersection均选中0个，回退到普通native_graph候选。因此不能将相同成功数解释成共享模式在充分支持下与普通法等价；它在这个输入和候选/评分组合里基本未起作用。但池oracle也未增加，单纯给共享模式更多权重没有依据。

## 为什么新增好候选仍失败：具体反例

validation_000/asset2真身份为C1槽9、C2槽16；实际选C2槽15。在真身份下2352个位姿对中有13个达标，最便宜达标对材料误差4.47cm，评分5.70159；最便宜对评分5.59662，却有84.06cm材料误差，合格对仅排第9。**这直接表明当前目标会偏爱材料错位；不是只有优化步数不够。**

rich访问过真身份tuple；changed没有访问过，存在有限搜索覆盖缺口。但对完整双射状态执行合法身份交换并重新穷举受影响轨迹后，完整目标分别反而上升0.33024、0.50865。这是一个具体反事实，不是全部错误的证明，也不排除其他全局状态更好；它说明只扩搜索未必按当前目标恢复正确身份。诊断计算发生在封存后，不记作新方法预测或收益。

## native_graph失败分解（四列互斥相加96）

| 外观 | 达标 | 本族池无好位姿 | 有好位姿但错身份 | 有好位姿且对身份但选错 |
|---|---:|---:|---:|---:|
| flat | 1 | 83 | 10 | 2 |
| rich | 16 | 77 | 3 | 0 |
| repeated | 8 | 80 | 8 | 0 |
| changed | 16 | 78 | 2 | 0 |

池无好位姿是当前前端/表示/预算的诊断，不是原始数据物理不可观测证书。

## 工程问题与研究问题

1. **工程输入与前端：** 原图有效像素少、固定top2/两图像对、native学习域差异、低正确匹配率。先修输入与观测覆盖；不能将ROI插值、加近景或原生模型移植包装论文创新。
2. **建模与目标：** 当前软支持按6个节点饱和，误匹配也能提供正支持；geometry/SIFT/ray可能偏向错误表面解释。84cm对优于4.47cm对是直接反例，但具体哪一评分分量致错尚未独立消融。未校准LG置信不是概率。
3. **材料共享表示：** 必须完整三期实际闭环才能进入共享模型，排除了大量两期材料证据。需要允许部分轨迹与不可见状态，而不是强造第三条边。部分轨迹、多图同步、容量约束本身已有研究，仍须同信息普通BA/匹配控制。
4. **对照的细节：** shared/product和intersection所选轨迹无重复节点；cycle_independent的两独立集合取并集仍有3个拟合包含重复观测节点，joint_all有140个。两者兼容矩阵用了相同排斥门限，但不能声称最终支持集合具有严格相同一对一容量。该限制已暴露，冻结结果不重写；下一版必须显式处理并给普通控制同样的容量模型。

## 是否推进主线与决策

- **实验链路推进：是。** 已有全24身份竞争、原生强普通前端、新连续候选、实际三期对应、联合SE3和完整位移评价；没有缩为单物体已知身份。
- **算法收益：未增加。** rich/changed仅候选各增1、实际0；flat/repeated退步；共享没有独立收益。
- **创新证据：未确认。** 不把原生匹配、图兼容、闭环、IRLS和本轮intersection称原创。暂不把V6共享模式作为论文主创新，不在这4旧DEV上继续扫模式/门限/评分权重。
- **下一项：** 全24构件都具有可解析的实际观测、允许两期部分材料轨迹的V7数据与模型契约，见[下一步](material_partial_tracks_next_v7_2026-10-26.md)。先在TRAIN核验材料可见/采样/匹配三层覆盖，然后冻结新seed；与同输入普通pairwise＋多期BA比较。采集补齐的收益记工程，只有同信息下部分持续材料匹配带来的稳定联合位移收益才支持论文贡献。

## 核验与已有方法

全量评分86797项；原始公开RGB-D/像素逆缩放/深度回投/合并节点/真实闭环/拟合端点/选中目标另行1081358项，最大选中目标重建误差7.11e-15；原生scalar/batch支持核对，7037个真实拟合端点损失重建。解析12列J误差5.83e-10。网络输出复用冻结缓存，未独立重跑全部网络。所有7方法预测先于本轮GT封存；4景已反复使用，不能以2688行当独立样本或宣称统计显著。未新增真实数据精度。

[LightGlue ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)作者2D模块实际执行；[FlowWeb CVPR2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Zhou_FlowWeb_Joint_Image_2015_CVPR_paper.html)和[GeoTransformer CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Qin_Geometric_Transformer_for_Fast_and_Robust_Point_Cloud_Registration_CVPR_2022_paper.html)为已有循环匹配和几何对应工作背景，其完整作者3D方法未在本轮运行。

## 图与复现

![全部实际成功与合法候选oracle](../results/material_native_modes_comparison_v6/native_modes_results.png)

![观测覆盖与材料正确覆盖；两组分母不同](../results/material_native_modes_comparison_v6/native_material_support.png)

命令顺序：cache_material_native_v6 → run_material_native_modes_v6 → run_material_mode_intersection_v6 → score_material_native_modes_v6 → audit_material_native_modes_v6 → analyze_material_native_modes_v6 → diagnose_material_native_new_candidates_v6 → report_material_native_modes_v6（均python -B -m scripts.）。生成根存在时拒绝覆盖；不要覆盖封存证据。
