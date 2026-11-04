# V4：实测表面内连续对应提案（冻结实验协议）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-26
- Verification Status: UNVERIFIED（本文件为运行前协议，实际完成以结果报告为准）
- Version Label: material_continuous_correspondence_v4

用户已授权继续本地实现/训练/验证。本协议固定预算与比较，不索取重复确认。

## 研究问题与可证伪条件

V3在实际采样中把“物理可见但未采中”分开，却没有生成对应。本轮问：以同样三个粗锚点为入口，在其周围**实际测得的局部表面**内学习细定位，是否能新增可形成准确位姿的材料对应，并在24构件三期身份竞争中带来收益？主指标仍为两期身份正确＋固定历史材料锚点三段p95误差均≤5cm，不用GT身份小实验替代。

## 已实现的模型及准确边界

保留V3源/目标兴趣点集合及固定soft3_31粗描述子。每源最多32查询，在每个目标身份组内保留余弦相似度最高的3个粗锚点，不按真身份取点。目标锚点周围半径6cm、12×12切平面投影采样；坐标取实际有效深度像素反投影，保持匿名构件、深度一致性；同构件网格三角形各边≤4cm且非退化才允许连续内插。未见表面或跨槽位/跨深度边界不连接。

小型Siamese全卷积网络同时读两尺度RGB/高度/支持10通道。源块中心四格特征均值构成查询，目标保留12×12局部特征，得分

\[
s_{ik\ell}=\langle f_\theta(P_i)_{center},f_\theta(P_{j_k})_\ell\rangle/0.12,
\quad P_\theta(\ell,\emptyset\mid i,j_k)=\operatorname{softmax}([s_{ik\ell},s_\emptyset]).
\]

TRAIN标签仅在原V3判定物理可见、当前实际格点有同材料2.5cm内支持时，以9mm Gaussian邻域软分布监督；其他为null。没有把真对应插进锚点或网格。TRAIN批次正负各24，seed31/47各固定900步，不在DEV选checkpoint。推断null不做未校准硬筛选，候选置信取条件空间峰值概率；这是明确限制，不宣称null概率已成功校准。

离散控制取局部峰值实际测点。连续分支先在峰值3×3邻域取条件分布矩

\[
\bar y_{ik}=\frac{\sum_{\ell\in\mathcal N(\ell^*)}e^{s_{ik\ell}}y_{j_k\ell}}{\sum_{\ell\in\mathcal N(\ell^*)}e^{s_{ik\ell}}},\quad
\hat y_{ik}=\operatorname{Proj}_{\mathcal S^{obs}_{j_k,\mathcal N}}\bar y_{ik}.
\]

投影集合仅包含该邻域有实测三顶点支持的三角形；没有有效三角形时退回峰值实测点。三粗锚点各保留一模态，不跨模态求平均；并不保留每块内部的所有峰，复杂歧义仍可能丢失。使用普通二阶距离相容求3个刚体种子，各加普通ICP，新增位姿最多6；完整旧候选、旧六方法和V3强ray精确整组假设都保留。没有新学习评分，所有提案由冻结V3“几何＋SIFT＋ray”评分和同一24身份求解选择，单独测对应/提案贡献。

## 必要对照与预算

1. baseline：逐字节复制已封存V3强ray。
2. ncc：相同块/格点/查询，手工3×3归一块向量，格点匹配。
3. random：相同网络随机初始化，格点匹配。
4. dense：训练网络，相同格点，直接取最大概率点——隔离加密采样/细描述子收益。
5. surface：与dense完全相同特征、分数及输入，只换有界实测表面内的连续均值/投影。
6. geo：从旧评分最好的3个位姿起点，各做6次同表面集合内的普通最近点投影/刚体拟合。
7. union：将各分支新增提案全合并，用完全相同普通评分，不包装学习评分收益。

各方法共享完整旧池，新增部分分族保留以识别提案来源；不是所有方法候选都相同。ncc/random/dense/surface的查询32、粗锚点3、网格144、几何种子3相同；geo的3×6投影与网络运算不等FLOPs，记录实际候选数与运行时间，不声称严格等算力。每族独立池oracle只在封存后诊断，不作为算法分数。

## 新颖性边界

粗到细匹配、稠密局部特征、soft-argmax和普通相容/SVD均已有先例，例如[LoFTR，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)和[RoMa，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html)。本轮不是其作者复现，更不能把小CNN替代称强发表方法比较。待验证的是有观测支持的连续材料对应能否补上多期量化缺口；即使有效，尚需与同信息普通连续配准区分，方可讨论额外方法贡献。

## 评价与停止条件

在4个反复使用的DEV场景完成7方法、全部24对象×4外观；模型/预测封存后才读DEV真值。记录身份、材料误差、每族pool oracle、旧位姿精确保留、各条件得失。事后GT身份下的3模态候选召回只作为机制诊断，不能替代完整任务。若dense解释全部收益，连续机制只算工程细化；若geo解释收益，不称新学习模型优势；若整体未超过baseline，不继续在同DEV扫步数/阈值。结构在TRAIN冻结后，只有出现值得确认的受控收益才另生成独立新种子数据，当前不宣称新confirmation或真实精度。

## 评分前协议补充：旧池相同搜索控制

在任何DEV真值评分之前，通过源代码核查发现：直接复制的V3baseline没有V4末尾检查所有旧方法整组假设的完整搜索，可能混入求解器变化。增加第8个方法`retained`：使用原baseline_family_pools，无任何新匹配/提案，运行与V4完全相同的身份起点、池组合、全部旧整组假设选择。控制单独生成并封存于`results/material_continuous_retention_control_v4`，随后才允许任何评分GT读取。

这不修改主7方法的已冻结模型/预算/推断，评分必须同时报surface相对retained、dense、geo及原baseline，不能将单纯更充分搜索当成学习收益。补充入口run_material_continuous_retention_control_v4；finish及scorer在评分前检查它已完成。该修正依据代码差异，不依据任务结果；不是事后按收益选择对照。
