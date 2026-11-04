# V16：局部、可拒绝共识的完整任务结果

## 本轮实际完成

冻结V3学习字段，四外观×24资产×8方法，共768条任务；全部预测封存后才进行材料真值评分。每族每方向最多12种子、3刚体更新，局部三点的TLS与partial逐字节共用初始位姿。保留所有V15联合端点和原有限身份搜索图。无新训练、数据或参数扫描。

## 完整任务

数字为身份均正确且三段固定历史材料锚点误差均≤5cm的资产数；每列24项。

|方法|flat|rich|repeated|changed|
|---|---|---|---|---|
|retained_v15_point|14|16|14|16|
|common_geometry|13|16|14|16|
|common_descriptor_fusion|14|16|14|16|
|common_point_density|14|16|14|16|
|family_global|14|16|14|16|
|family_local|14|16|14|16|
|family_triplet|14|16|14|16|
|family_partial|14|16|14|16|

背面子集每列8项：

|方法|flat|rich|repeated|changed|
|---|---|---|---|---|
|retained_v15_point|0|0|0|0|
|common_geometry|0|0|0|0|
|common_descriptor_fusion|0|0|0|0|
|common_point_density|0|0|0|0|
|family_global|0|0|0|0|
|family_local|0|0|0|0|
|family_triplet|0|0|0|0|
|family_partial|0|0|0|0|

前四行为旧结果重放/共同池头；最后四行为“旧池＋该生成族”的消融，不声称八方法候选池相同。普通TLS本身已可拒绝外点。

## 生成上限与实际选择

真值仅用于事后读取正确身份的池内最低材料锚点误差；候选合格不等于推断成功。

|候选范围|flat|rich|repeated|changed|
|---|---|---|---|---|
|retained|14|17|16|17|
|global24_tls|14|17|16|17|
|local24_tls|14|17|16|17|
|local3_tls|14|17|16|17|
|local3_partial|14|17|16|17|
|union|14|17|16|17|

## 同资产的增减

新增与丢失同时列出；四外观相关，不能把768条方法记录当独立样本量。

|外观|对照→候选|新增资产|丢失资产|共同正确ID误差变化中位数(m)|
|---|---|---|---|---|
|flat|retained_v15_point→common_point_density|[]|[]|-1.1469358864358865e-05|
|flat|common_geometry→common_point_density|[0]|[]|-0.00027280337361600257|
|flat|common_descriptor_fusion→common_point_density|[]|[]|0.0|
|flat|retained_v15_point→family_global|[]|[]|0.0|
|flat|family_global→family_local|[]|[]|0.0|
|flat|family_local→family_triplet|[]|[]|0.0|
|flat|family_triplet→family_partial|[]|[]|0.0|
|rich|retained_v15_point→common_point_density|[]|[]|0.0|
|rich|common_geometry→common_point_density|[]|[]|-1.5293631291498667e-05|
|rich|common_descriptor_fusion→common_point_density|[]|[]|-0.001216455325200138|
|rich|retained_v15_point→family_global|[]|[]|0.0|
|rich|family_global→family_local|[]|[]|0.0|
|rich|family_local→family_triplet|[]|[]|0.0|
|rich|family_triplet→family_partial|[]|[]|0.0|
|repeated|retained_v15_point→common_point_density|[]|[]|0.0|
|repeated|common_geometry→common_point_density|[]|[]|-0.0001512900447214235|
|repeated|common_descriptor_fusion→common_point_density|[]|[]|-4.090905417204857e-05|
|repeated|retained_v15_point→family_global|[]|[]|0.0|
|repeated|family_global→family_local|[]|[]|0.0|
|repeated|family_local→family_triplet|[]|[]|0.0|
|repeated|family_triplet→family_partial|[]|[]|0.0|
|changed|retained_v15_point→common_point_density|[]|[]|0.0|
|changed|common_geometry→common_point_density|[]|[]|-6.391507576641864e-06|
|changed|common_descriptor_fusion→common_point_density|[]|[]|-0.000608609903195746|
|changed|retained_v15_point→family_global|[]|[]|0.0|
|changed|family_global→family_local|[]|[]|0.0|
|changed|family_local→family_triplet|[]|[]|0.0|
|changed|family_triplet→family_partial|[]|[]|0.0|

## 主共同评分的失败归因

|类别|flat|rich|repeated|changed|
|---|---|---|---|---|
|identity_assignment_failed_despite_qualified_pose|0|0|2|0|
|no_qualified_pose_in_true_identity_pool|8|7|8|7|
|objective_prefers_bad_pose_at_correct_identity|0|1|0|1|
|success|14|16|14|16|
|true_identity_not_visited|2|0|0|0|

## rich背面H2配对诊断

下表为GT选择的8个真实身份配对，5cm配对误差，不是完整三段任务，不进入预测。

|生成族|raw合格配对/8|前3位姿合并双向后合格配对/8|
|---|---|---|
|global24_tls|0|0|
|local24_tls|0|0|
|local3_tls|0|0|
|local3_partial|0|0|

原global种子真实材料峰1；local种子真实材料峰0。局部同起点共有96个，初始化/普通TLS/partial达到5cm的种子分别为0/0/0。这不是全seed上限，也不是超预算新方法。

混合目标下降但实际材料误差变大的局部种子：38/96。该数字描述目标与任务不一致，不能解释为实现了更好匹配。

## 解释边界与下一步

局部归一化、三点SVD、TLS和Gaussian外点混合都是已知工具。本轮是分离失败机制的算法实验，不能把实现这些组件称为原创贡献。局部种子与局部权重仍作为一包改变；partial同时改变特征先验、空匹配和目标容量，也尚未拆出三者各自作用。

空匹配只接收局部候选外质量，没有射线可见性或历史未知表面的物理建模。容量缩减不是精确EM/最优传输，接受检验仅保证原混合目标不升。若同起点普通TLS已解释收益，就保留为工程改进；若partial相对TLS没有新增合格或改善材料误差，就不继续扫该混合模型参数。

判断需要同时看：实际任务增减、候选覆盖、raw→selected损失、同起点目标下降与真实误差变化。优先改阻断主线的对应或选择模型，不能用更多迭代替代原因分析。后续明确决策见本轮decision文件。

## 已有发表方法参考

V15官方PointDSC模块默认13/16/14/16；预先固定5cm阈值控制0/6/5/5。保留较强默认结果，不能择弱报告。输入为我们的冻结描述子NN和固定任务身份图，不是作者完整FCGF系统，也不是与V16同候选/同生成预算控制。

[MAC（CVPR 2023）](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_3D_Registration_With_Maximal_Cliques_CVPR_2023_paper.html)已研究局部极大团可能优于最大团；[CPD](https://arxiv.org/abs/0905.2635)已有概率混合配准。没有独立数据及强先例排除，不能确认高创新。

## 正确性与局限

72项合成数学正确性核对通过；本轮完整评分核对147905项，源中心验证147456个，同起点核验28008组、目标不升路径112032条，选中目标函数最大重算差0.0。这些是实现完整性，不是独立重跑确认。

一个已见TRAIN场景、四相关外观；有限身份图仍可能不访问真实身份。无实机或真实跨期确认。旧192项结果和旧源码/权重通过完成审计另行核验。

## 进一步定位：局部池没有形成足够真对应

96个rich背面正向局部池的真实材料源点数分布：{0: 71, 1: 22, 2: 3}；达到3个真源点的池为0。材料真对应标准为1.5cm；这不是一般可观测性或所有近似姿态不可能的证明。它与实际全部raw配对失败共同支持：目前主瓶颈在共识形成之前，不能只改最后top3裁剪或继续迭代。

目标容量缩减在91/96个终点实际触发；故不能用‘容量没起作用’解释零增益。混合模型还存在可解析的软模态平均偏差：无噪声20mm双模态反例中，TLS保持真位姿而partial从真位姿漂移9.847mm；见模型文档。反例不是对真实任务38例漂移的唯一原因认定。

一次诊断CSV输出因前向/背面行列集合不同而失败，失败CSV及源码快照保留；补齐可选列后重跑诊断和报告。冻结推断、主评分及参数未改变。这是导出工程问题，不是算法收益。
