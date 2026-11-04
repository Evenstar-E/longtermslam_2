# V15：全域学习材料区域与全局位姿，完整结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: 实际预测封存后评分与来源核对，不是独立复跑/原创性确认
- Version Label: material_surface_distribution_matching_v15

## 本轮真正推进了什么

**结论：主线链路推进，当前区域协方差构想不通过核心收益验收。**本轮14/16/14/16可由普通同信息学习融合复现；同均值加入协方差新增0/丢失0。新增全局提案没有扩大合格资产覆盖。保留全域学习特征基础，停止继续扫当前区域宽度/协方差系数。

复用已经TRAIN-only训练的V3 soft3_31空间纹理编码器，不重新训练旧CNN。此前交接遗漏V3已有区域分布训练/V4已有局部密集细化，本轮核对后纠正，没有把它们再当新工作。新实现对每外观1,357,700个原生目标像素作全域搜索，从每个512源中心的3个分离实测区域产生全局SE(3)，不局限于3粗锚点附近。

区域互斥、剩余概率保留、后验/先验分别有几何矩。空间协方差进入距离相容图和位姿混合核；普通点相容、普通几何/学习融合、点/均值/协方差密度都共享实际新旧位姿。旧V14的全部耦合候选保留。完整数学/代码边界见material_surface_distribution_model_v15_2026-10-27.md。

## 完整任务结果

24资产×4外观×8方法=768，每列24；两身份正确且三段固定材料锚点p95误差均≤5cm才成功。

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| retained_v14_geometry | 12 | 16 | 14 | 16 |
| retained_v14_continuous | 11 | 16 | 14 | 16 |
| common_geometry | 13 | 16 | 14 | 16 |
| common_rgb_fusion | 10 | 16 | 13 | 16 |
| common_descriptor_fusion | 14 | 16 | 14 | 16 |
| common_point_density | 14 | 16 | 14 | 16 |
| common_mean_density | 14 | 16 | 14 | 16 |
| common_region_density | 14 | 16 | 14 | 16 |


背面8项：

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| retained_v14_geometry | 0 | 0 | 0 | 0 |
| retained_v14_continuous | 0 | 0 | 0 | 0 |
| common_geometry | 0 | 0 | 0 | 0 |
| common_rgb_fusion | 0 | 0 | 0 | 0 |
| common_descriptor_fusion | 0 | 0 | 0 | 0 |
| common_point_density | 0 | 0 | 0 | 0 |
| common_mean_density | 0 | 0 | 0 | 0 |
| common_region_density | 0 | 0 | 0 | 0 |


![全任务结果](../results/material_surface_distribution_matching_v15/task_comparison.png)

不能以主观最好分支拼接赢家作为最终方法；四外观相关、单个已见TRAIN，不当独立样本显著性证据。V14旧原始强参考另有V12 12/16/14/16、V9普通RGB融合9/16/16/16；不同表示/预算版本只作任务参照。

## 是否生成以前没有的正确位姿

| 外观 | 保留V14全池 | 点提案族＋旧池 | 区域提案族＋旧池 | 全并集 |
|---|---:|---:|---:|---:|
| flat | 14 | 14 | 14 | 14 |
| rich | 17 | 17 | 17 | 17 |
| repeated | 16 | 16 | 16 | 16 |
| changed | 17 | 17 | 17 | 17 |


新增合格资产：

```json
{
  "flat": {
    "point": [],
    "region": [],
    "union": []
  },
  "rich": {
    "point": [],
    "region": [],
    "union": []
  },
  "repeated": {
    "point": [],
    "region": [],
    "union": []
  },
  "changed": {
    "point": [],
    "region": [],
    "union": []
  }
}
```

这是正确身份下有限实际候选池的事后oracle，**不是算法成功率**。点/区域族都包含旧池，只对实际保存的新端点取子集；全并集还包括混合与跨期传播。旧有限身份图不扩展，flat22/23真三元组仍可能未访问。

## 后续全量结构归因：提升来自新提案还是新评分

| 仅保留V14候选，用本轮评分 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| common_geometry | 12 | 16 | 14 | 16 |
| common_descriptor_fusion | 13 | 16 | 14 | 16 |
| common_region_density | 14 | 16 | 14 | 16 |


这是主结果后设计的POSTHOC统一反事实：全部24×4×6均只用原V14候选及已冻结本轮分数、同状态图，不改参数/训练/位姿。区域密度在仅旧池时已经14/16/14/16，故本轮这一组完整成功数不需要新增全局提案也能复现。新增端点仍让普通几何flat12→13、学习融合13→14，不能称完全无用；但这不是区域协方差的独占贡献。

## 同信息机制增量

- flat：retained_v14_geometry → common_geometry，新增[21]，丢失[]，共同正确身份下误差变化中位数-0.005378138452424412m。
- flat：common_descriptor_fusion → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- flat：common_point_density → common_mean_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- flat：common_mean_density → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- rich：retained_v14_geometry → common_geometry，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- rich：common_descriptor_fusion → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数-0.0016988355740062753m。
- rich：common_point_density → common_mean_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- rich：common_mean_density → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- repeated：retained_v14_geometry → common_geometry，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- repeated：common_descriptor_fusion → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数-0.0007159604040474746m。
- repeated：common_point_density → common_mean_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- repeated：common_mean_density → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- changed：retained_v14_geometry → common_geometry，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- changed：common_descriptor_fusion → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- changed：common_point_density → common_mean_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。
- changed：common_mean_density → common_region_density，新增[]，丢失[]，共同正确身份下误差变化中位数0.0m。

均值→协方差评分合计净变化0，必须与普通学习融合及旧强控制一起解释。该比较隔离评分协方差；**全局区域生成器同时改变峰值→均值和协方差，未完成单独均值生成器的全链对照**，不能把所有新全局解归给协方差。若普通几何在共同池解释全部提升，就归为候选/表示收益。

## 真实执行的发表方法模块对照

| 发表模块控制 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| PointDSC官方默认10cm | 13 | 16 | 14 | 16 |
| PointDSC固定任务尺度5cm | 0 | 6 | 5 | 5 |

背面8项：

| 发表模块控制 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| PointDSC官方默认10cm | 0 | 0 | 0 | 0 |
| PointDSC固定任务尺度5cm | 0 | 0 | 0 | 0 |


固定官方PointDSC源码/3DMatch权重，作者CPU活动算法没有改写；没有缺失活动权重，作者demo strict=False仅忽略已不用的legacy gamma。使用与主实验相同的全原生V3描述子，按作者demo每源一个NN构造对应；任务头采用双向作者内点比例及相同有限身份状态图。默认10cm与预先固定5cm内点/NMS控制全部报告，5cm方案在读取本轮任何GT分数前生成并封存，没有按分数挑选；sigma_spat和学习权重不变。

这是作者核心配准模块的全24×4真实任务对照；不是完整FCGF前端/3DMatch benchmark，最终候选池与三模态对应预算也不等同于V15。不得拿该表单独宣称超过完整作者强系统或把尺度不适配归为发表方法普遍缺陷。每个方法仍用相同5cm固定材料锚点评估门限。

## 失败类型

| 外观/方法 | 成功 | 缺合格位姿 | 有好位姿但评分选错 | 身份分配失败 | 真身份未访问 |
|---|---:|---:|---:|---:|---:|
| flat/common_descriptor_fusion | 14 | 8 | 0 | 0 | 2 |
| flat/common_mean_density | 14 | 8 | 0 | 0 | 2 |
| flat/common_region_density | 14 | 8 | 0 | 0 | 2 |
| rich/common_descriptor_fusion | 16 | 7 | 1 | 0 | 0 |
| rich/common_mean_density | 16 | 7 | 1 | 0 | 0 |
| rich/common_region_density | 16 | 7 | 1 | 0 | 0 |
| repeated/common_descriptor_fusion | 14 | 8 | 0 | 2 | 0 |
| repeated/common_mean_density | 14 | 8 | 0 | 2 | 0 |
| repeated/common_region_density | 14 | 8 | 0 | 2 | 0 |
| changed/common_descriptor_fusion | 16 | 7 | 1 | 0 | 0 |
| changed/common_mean_density | 16 | 7 | 1 | 0 | 0 |
| changed/common_region_density | 16 | 7 | 1 | 0 | 0 |


缺候选仅针对已访问真身份池，不是原始数据不可辨识证明。池里有合格解但同身份选坏，才称该池内评分偏好错误；身份分配失败混合评分与有限搜索。

## 全域材料检索与候选裁剪

```json
{
  "rear_assets": 8,
  "dense_supported": 379,
  "peak_supported": 56,
  "region_supported": 97,
  "rear_peak_oracle_pair_qualified": 8,
  "point_raw_pair_qualified": 0,
  "point_selected_pair_qualified": 0,
  "region_raw_pair_qualified": 0,
  "region_selected_pair_qualified": 0
}
```

该诊断仅rich，在GT身份下核对H2材料支持、实际全域top3峰和区域并集，以及真正生成的全部raw/预算selected端点。pair5cm不等于完整三段任务合格，GT筛对应SVD从不进入预测。若raw有好解而selected没有，则裁剪值得改；若top3连正确材料都没有，则先改表征/搜索，而不是再迭代同一个局部姿态。

进一步只改变种子保留的诊断：

```json
[
  {
    "regional": 0,
    "original_pair_qualified": 0,
    "allseed_pair_qualified": 2,
    "original_true_peak_seed_total": 1,
    "allseed_qualified_poses": 7
  },
  {
    "regional": 1,
    "original_pair_qualified": 0,
    "allseed_pair_qualified": 2,
    "original_true_peak_seed_total": 1,
    "allseed_qualified_poses": 12
  }
]
```

每个生成器在8个背面任务选的96个谱种子，只有1个是真实近材料峰。枚举全部1536个种子，保持原C²/全局谱权重/每共识24个源唯一对应不变，普通点和区域各有2/8个H2配对出现≤5cm位姿（不是相同资产集合），各仍有6/8失败。该干预由GT选择诊断身份并用GT报告最好解，**不是算法成功/同预算比较**。它说明12谱种子的裁剪漏掉部分解，同时也不能用只扩大种子数量解释其余失败。raw24候选无好解仅排除最终top3位姿保留作为充分解释，**不排除更早top3对应/12谱种子裁剪**。

下一值得实现的是同样12个提案预算下的多局部/可拒绝稀疏共识与普通归一化谱/鲁棒估计控制，检验少量真实材料关系是否被大量几何自洽伪对应压制。该原因尚未唯一归因，不能仅凭谱排名就宣称已证明所有错误来自主特征向量。

## 工程与创新分开

实际float32谱图次正规数处理是运行加速，公开同图位姿差0，不算创新。13区域/6原型正确性检查不是任务收益。评分7177项，来源中心147456个，最大实际选中目标重算误差0。各slot是实测匿名空间候选域，持续实例/材料标签不进入推断。

全域密集搜索、Gaussian混合、面积/矩、谱相容和SVD已有先例；本轮只据同信息对照评价本构造是否值得继续，未证明高原创性。原metricpatch编码非SE3等变、区域非Gaussian的矩近似、有限3模态、宽残余分量、固定历史/局部表面与未知处理均有局限；未完成统一未见表面生成式地图、真实跨期或发表作者完整系统比较。所有旧代码、数据、权重和192唯一证据保留。
