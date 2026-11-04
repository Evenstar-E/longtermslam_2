# RGB-D材料描述子：实际训练与完整任务评测（2026-10-26）

## Material Passport
- 8个TRAIN场景、704缓存案例/33792查询；21040可见、6144画外、6608遮挡、258可见重涂。
- 两版共8次固定500步训练，主种子31预先指定，47/73只作训练/描述子波动检查；没有按heldout结果选checkpoint。
- 端到端覆盖全部4个旧留出场景/96构件、三期、四种外观；这些场景先前已分析，本轮属于DEV，不是独立确认。
- 新网络双CNN分支约数万参数，CPU训练；无GPU环境/现有包修改。原数据、旧192分数保留。
- 官方预训练源码固定commit5a650c444718caffc2669dc0094cde908f42dd73；实测深度均匀采样点的SuperPoint dense描述子＋原LightGlue图像坐标匹配，属于模块强对照，不称作者完整benchmark。

## 完整任务：全部4个DEV场景

联合合格＝两期身份正确，固定历史材料点H→C1/H→C2/C1→C2三段p95误差均≤5cm。每格分母96。

|方法|纯色|丰富纹理|重复纹理|局部重涂|
|---|---:|---:|---:|---:|
|旧SIFT普通融合（同一4场景）|1|14|12|13|
|metric_rootsift|0|8|8|6|
|untrained|0|8|5|9|
|contrastive|0|7|2|7|
|learned_fusion|1|9|3|6|
|learned_material|2|2|2|3|
|superpoint_lightglue|6|6|4|6|
|factorized_fusion|1|7|3|7|
|factorized_material|4|7|3|6|
|balanced_fusion|1|5|2|4|
|balanced_visibility|2|5|4|4|
|balanced_material|2|5|4|4|

## 真身份共同候选池覆盖（仅事后oracle）

|特征|纯色|丰富纹理|重复纹理|重涂|
|---|---:|---:|---:|---:|
|metric_rootsift|4|8|8|7|
|untrained|5|8|10|9|
|contrastive|4|7|6|7|
|learned_fusion|4|11|7|9|
|superpoint_lightglue|7|9|9|9|
|factorized_fusion|6|7|6|7|
|balanced_fusion|5|6|7|6|

## 同特征、同提案的头增量

|阶段|外观|候选/普通融合|共同正确ID|三期合格新增/丢失|共同ID的C1→C2误差 cm：基线/候选|
|---|---|---|---:|---:|---:|
|V1|flat|learned_visibility/learned_fusion|5|0/0|66.55/66.25|
|V1|flat|learned_material/learned_fusion|3|1/0|8.19/8.54|
|V1|rich|learned_visibility/learned_fusion|20|1/1|6.38/6.38|
|V1|rich|learned_material/learned_fusion|7|0/7|7.68/8.00|
|V1|repeated|learned_visibility/learned_fusion|7|0/0|5.97/5.97|
|V1|repeated|learned_material/learned_fusion|3|0/1|2.14/2.05|
|V1|changed|learned_visibility/learned_fusion|18|1/0|7.28/7.28|
|V1|changed|learned_material/learned_fusion|8|0/3|7.28/7.44|
|V2|flat|factorized_visibility/factorized_fusion|7|1/0|61.68/61.68|
|V2|flat|factorized_material/factorized_fusion|6|3/0|35.11/34.01|
|V2|flat|balanced_visibility/balanced_fusion|6|1/0|34.54/33.74|
|V2|flat|balanced_material/balanced_fusion|6|1/0|34.54/33.74|
|V2|rich|factorized_visibility/factorized_fusion|19|0/0|4.06/4.06|
|V2|rich|factorized_material/factorized_fusion|17|0/0|3.92/3.92|
|V2|rich|balanced_visibility/balanced_fusion|12|0/0|3.67/4.15|
|V2|rich|balanced_material/balanced_fusion|10|0/0|3.58/4.06|
|V2|repeated|factorized_visibility/factorized_fusion|7|0/0|5.02/5.02|
|V2|repeated|factorized_material/factorized_fusion|8|0/0|5.13/5.03|
|V2|repeated|balanced_visibility/balanced_fusion|8|2/0|5.11/5.60|
|V2|repeated|balanced_material/balanced_fusion|7|2/0|5.04/5.18|
|V2|changed|factorized_visibility/factorized_fusion|17|0/0|4.13/4.13|
|V2|changed|factorized_material/factorized_fusion|15|0/1|3.92/3.92|
|V2|changed|balanced_visibility/balanced_fusion|15|0/0|5.38/5.38|
|V2|changed|balanced_material/balanced_fusion|14|0/0|5.38/5.38|

## 校准问题及数学修复

V1把同一m同时当dustbin相对logit与物理可见概率，实际Softmax边缘为sigmoid(m+logsumexp(s))，不等于sigmoid(m)。V2明确P(v,j,c)=P(v)P(j|v)P(c|v,j)，通过概率归一、逐行分数平移不变、可见边缘和解析梯度检查。标准概率分解修复不是原创理论。
V1 TRAIN Brier0.613→factorized0.229；balanced种子31/47/73为0.229/0.225/0.220。平衡外观头TRAIN AUROC0.780/0.882/0.887，但先验校正后0.5阈值仍未检出任何258个重涂点。不能只报AUROC或训练loss称鲁棒。

## 条件描述子诊断（与端到端分开）

诊断目标集合人为包含可见GT正点及随机干扰；采样也按标签分层，不是部署搜索。只用于查看表示/头在这些条件下的泛化，不得当作全24身份成功率。

|模型|条件物理点检索正确率|可见性Brier|纹理持续性AUROC|重涂检出/总数|未重涂误报数|
|---|---:|---:|---:|---:|---:|
|untrained|54.98%|0.246|0.531|114/114|10337|
|contrastive_31|61.85%|0.246|0.551|114/114|10337|
|full_31|63.01%|0.610|0.223|0/114|0|
|factorized_31|62.00%|0.228|0.224|0/114|0|
|balanced_31|61.91%|0.228|0.669|0/114|0|
|balanced_47|62.35%|0.225|0.796|0/114|0|
|balanced_73|59.98%|0.219|0.819|0/114|0|

常数可见先验的条件诊断Brier约0.236。V2约0.228比常数仅小幅改善，远不足以宣称高质量可见性估计；V1→V2的大幅差值主要是概率语义修复。未训练外观头把所有点判为重涂，虽然114/114检出，却误报10337个未重涂点。

## 对应位置支持与算法缺口

只在事后给定真身份和真位姿来选最近材料对应，当前相同采样位置可拟合合格89/96项。其中无近景80项可合格73项、有近景16项全部可合格；两组C2真实近邻支持中位分别6/15。这里用到了GT，不能当算法成绩，也不证明观测中的对应可被唯一辨认。

它否定了“当前所有困难任务都只是没有采到可形成准确位姿的材料点”这一解释。真实采样位置支持通常存在，但实际描述子、匹配与RANSAC提案没有把这些链接提取出来。

训练条件诊断人为注入了真实正点，部署时使用独立均匀采样，真点未必恰好出现；条件检索由随机CNN约55.0%升到约62%并不保证实际配准。新的候选池覆盖仍只有个位数至十余项，说明训练检索改善没有充分转化为部署提案。

## 方向级结论与下一步

1. **已完成真正训练与已发表模块对照，当前“小CNN＋两个标量头”不适合作为主创新结论。** 旧同输入SIFT融合rich/repeated/changed为14/12/13；新V1普通学习融合9/3/6，V2 balanced材料5/4/4，未建立稳定系统收益。不能只报rich比随机CNN8→9的局部变化。
2. **可见性概率修复属于模型正确性进展。** 头的独立增量仍弱：balanced材料与同特征visibility在四种外观中合格任务集合完全相同；factorized材料仅flat新增2，changed丢1。泛化重涂114点仍0检出。
3. **不继续在本版本扫轮数/温度/阈值寻找论文收益。** 保留旧SIFT/官方方法的候选，下一模块改为部署一致的材料对应训练：使用实际独立采样目标集合，不注入保证存在的正点，显式区分物理不可见、可见但采样未覆盖、可见已覆盖；用材料坐标邻域/像素足迹分布监督替代孤立精确索引。
4. **用强几何一致性配准拆开描述子与提案问题。** 普通融合与新模型必须共享旧、新候选，完整24身份和固定历史材料锚点保持；普通强对照能够解释的收益归为工程。进一步创新应围绕不确定材料对应与连续位姿/可见支持的联合推断，而不是再加一个头。新模块尚未实现。
5. **仍需独立确认。** 本轮4个场景已见，尚未生成新confirmation场景。本轮也没有新增真实跨期精度结果，不作论文创新或真实泛化确认。

## 限制与判断规则

- 必须区分metric rectification/均匀采样收益、神经网络训练收益、辅助头额外收益。metricRootSIFT、随机CNN、普通contrastive和同特征融合分别提供这些对照。
- 模型只编码局部RGB与几何高度/支撑，不是完整PointMBF/ViGG架构；世界轴切平面不是SE(3)等变网络。
- 几何损失权重在本轮融合/可见性/材料方法均为1，补掉上一轮1→0.5的混淆；共同位姿池数量相同，计算FLOPs不相同。
- 标签中的物理可见性、离散采样是否覆盖真点、错误匹配置信度不同。训练保证注入正点，而真实采样无此保证；不能用校准改善代替候选定位和联合身份结果。
- 概率分解已用于V2训练和头输出，配准端仍采用启发式距离/正支持评分＋软权重，尚未实现同一个归一化概率模型下的连续材料对应与位姿联合推断。因此本轮不能作为完整概率模型优劣的验证。
- 盒体/程序纹理、已知相机、名义槽位、仅4/24近景仍有利或限制。flat/repeated克隆对仍有物理身份置换对称。
- 所有CNN、InfoNCE、概率分解、类平衡、RANSAC/ICP都是已有构造；论文创新必须来自超出这些控制且可复现的联合材料机制增量。

![全部DEV任务与oracle](../results/material_joint_descriptor_comparison_v2/trained_task_comparison.png)

![训练与校准](../results/material_joint_descriptor_comparison_v2/training_and_calibration.png)
