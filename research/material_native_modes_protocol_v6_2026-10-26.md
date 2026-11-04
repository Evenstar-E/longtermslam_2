# V6执行协议：原生匹配、材料模式与连续位姿生成

## Material Passport

- Origin Mode: local code experiment
- Status: prescore protocol; actual completion in results report
- Data: 4 reused DEV scenes, no new independent confirmation

## 研究问题与已有方法边界

V5只给固定姿态池评分，rich78/96、repeat80/96缺达标位姿。V6直接从实际RGB图像匹配、实测RGB-D点生成新的连续SE3；以同数据普通强匹配和普通多期联合配准判断材料模式选择是否有独立增量。

[LightGlue，ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html)是已有学习局部匹配器；本地源码/权重固定于source_manifest的commit。两处SuperPoint utility端口沿用旧本地适配，不重写网络，原生forward产生检测点/描述子，LightGlue原始forward启用作者自适应层/点剪枝。这里是作者2D模块＋本地匿名构件ROI/3D流程，不是作者完整3Dbenchmark复现。

循环一致性已有[FlowWeb，CVPR2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Zhou_FlowWeb_Joint_Image_2015_CVPR_paper.html)等方法；距离/角度不变几何对应已有[GeoTransformer，CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Qin_Geometric_Transformer_for_Fast_and_Robust_Point_Cloud_Registration_CVPR_2022_paper.html)。本轮不运行其作者模型，也不把图、互斥、SVD、IRLS或闭环本身称原创。

## 固定观测预算与限制

每epoch匿名构件选可见像素最多的2个观察，裁剪ROI长边256；原生检测上限512，按实际原始深度像素去重后每view至多64点。OpenCV half-pixel逆缩放与SuperPoint原点约定分别处理；3D只用原始深度像素中心。图像放大不创造新深度样本。全24×24身份对保留，不用真身份或采集group。

每身份对2×2图像对中按描述子相似的最佳双射选2对，覆盖两个视图，降低CPU成本；不是匹配了所有可用视角。额外同epoch原生LG匹配需实测世界距离≤15mm才合并节点，坐标选真实代表点，保留raw_node/raw_xyz；等价关系是有噪声的观测假设，不保证真材料完全相同。缺点是前端选2视角/2图像对和每pair96边可能漏支持，必须报告，不将缺失归为原始数据不可观测。

## 模式定义与新姿态

观测节点是epoch/view/实际像素；跨期边来自真实LightGlue匹配。三节点材料轨迹必须同时具有H↔C1、H↔C2、C1↔C2原始观测边（可经同epoch实测等价节点），不能用位姿矩阵相乘伪造观测闭环。

每边至多96对应，每三元身份(i,j,k)至多64真实轨迹。普通pairwise RANSAC与独立距离相容图各最多4模式、各加ICP；保持全部旧V4union候选。三期共轨迹不足3或退化时不虚构新支持，保留pairwise旧/新候选。

对轨迹r,s，距离d_H,d_1,d_2，原共享图用C01 C02 C12乘积；独立图用C01/C02分别选模式。三期同源/同目标观测及4mm近重复互斥对两者相同。模式产生G1,G2，不是旧姿态后置加分。四步相同12DoF Cauchy IRLS求解

\[
\sum_r w_r\{\rho(\|G_1x_H-x_1\|^2)+\rho(\|G_2x_H-x_2\|^2)+\tfrac12\rho(\|G_2G_1^{-1}x_1-x_2\|^2)\}.
\]

解析J已核对12独立有限差分，max5.83e-10；实际稳健目标回溯下降。坐标和标签都由观测构造，不用GT面/姿态/对应。

## 共同评分与必要对照

六主方法：old_pool（仅旧姿态）、native_ransac、native_graph、joint_all（全部同轨迹稳健多期配准）、cycle_independent（普通循环支持＋独立模式选择）、shared_modes（联合轨迹兼容模式）。后三者相同轨迹、权重、模式/迭代预算、优化器，区别是如何选择匹配模式。

全部方法统一普通目标：旧geometry/SIFT/ray＋原生匹配正支持项。原生支持为confidence×25mm残差Gaussian，按源/目标节点分别最大聚合后取较小总量，以6个有效支持饱和；无匹配项为常数，不虚构消失。该项是普通特征融合控制，不称创新，也不是校准概率。全局身份状态图对全部方法共同，两轮各64状态，并保留旧V4/V5完整身份初值；不是全局最优。姿态对按每族unary掩码全部枚举，新增多期模式保持tuple专属成对，不能任意跨模式混搭。

## 评分前补充：距离兼容交集

在任何GT结果前观察到flat产品共享图几乎无提案（两个场景0/1）。d1−dH、d2−dH、d2−d1相关，乘积C再阈值0.2会比单期门限额外收缩。保留原模型，新增shared_intersection：C=min(C01,C02,C12)=[1−max(e01²,e02²,e12²)/τ²]_+，τ和其余参数不变。它是同几何约束的不同聚合，不是新概率独立性定理。

补充方法与原模型均报告；相同native输入、轨迹、4模式/4步优化、主run已访问身份图，没有额外搜索。全部额外tuple模式先封存再评分，避免按任务收益挑版本。

## 验收

同时报告每方法真实身份合法候选上限与全24身份竞争后的固定材料点三段p95≤5cm；coupled oracle只允许原pairCartesian或同一tuple的成对模式，不能将各模式任意混合抬高上限。额外核查GT真实身份tuple实际轨迹/可用模式覆盖，若某机制没有三期材料观测支持，不用其无收益证明所有跨期共享无效。标准native与joint_all已解释的增益不归给创新。全部4景为DEV，不再按它扫参数；独立新场景/真实链路仍需后续确认。
