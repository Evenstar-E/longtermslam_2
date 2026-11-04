# V7 持续材料部分匹配模型：实现与可检验边界

日期：2026-10-26。作用于冻结 full24 的一个 TRAIN 场景，24 个对象、三期、四外观。不是新训练，不是独立确认。本文件在本轮任务评分前建立；结果另存报告。

## 本轮优先级

上轮的关键缺口是背面任务没有好位姿候选，以及仅用独立匹配边不能表达持续材料身份。本轮仅修复会阻断模型比较的候选覆盖：保留全部 Base/RANSAC/graph 姿态，给所有方法加入相同的 24 个标准 PCA 旋转初值。停止继续扩准备数据、扫旧训练轮数和修饰审计数量。主要工作是让新材料模型真正产生连续位姿，再用严格任务收益判断。

## 观测、未知与材料容量

每期匿名槽位中的原生 SP/LG 实测点组成集合 $X_e=\{x_{ei}\}$，同一实际观测节点只保留一个代表。点来自真实渲染 RGB-D 像素，不补造隐藏表面纹理。历史身份为任务查询，当前两期身份在全部 24 个槽位中竞争。

$G_0=I,G_1,G_2\in SE(3)$，相对位姿 $T_{ef}=G_fG_e^{-1}$。将点投影到目标期的公开 RGB-D：同槽位深度一致标记为 hit；位于实测首回波之前标记 free；在回波后、视野外或缺深度标记 unknown。同期间任一视图 hit 优先，free 采用最小正间隙；这是明确的工程聚合近似，不是独立射线完整似然。

令 $u_i(T),v_j(T)\in\{0,1\}$ 为双向 hit，$s_{ij}\in[0,1]$ 为冻结 LG 匹配置信度（无原生边取 0，不叫概率），定义

$$c_{ij}(T)=\frac{\|T x_i-y_j\|^2}{\sigma^2}+1-s_{ij},\qquad
g_{ij}(T)=u_i(T)+v_j(T)-c_{ij}(T),\quad\sigma=0.04\;\mathrm m.$$

独立可见性对照精确求解带空匹配的容量问题：

$$V_{ef}(T)=\max_{P\ge0}\sum_{ij}P_{ij}g_{ij}(T),\quad
P_{ij}\in\{0,1\},\ P\mathbf1\le\mathbf1,\ P^\top\mathbf1\le\mathbf1.$$

空匹配增益为零，因此没有共同观测时不强造同材料对应。实现是带虚节点的 Hungarian；这是标准算法。

持续模型以材料簇 $\mathcal C$ 表示对应，每簇每期最多一个观测，可以只含一期或两期；允许 C1–C2 材料首次出现而 H 缺失。共享增益

$$V_{\rm shared}(G)=\sum_{C\in\mathcal C}\sum_{e<f}\omega_{ef}\!sum_{i,j}\mathbf1[(e,i),(f,j)\in C]\,g^{ef}_{ij}(G_fG_e^{-1}),\quad\omega=(1,1,1/2).$$

历史融合对照先匹配 H–C1，再把 C2 加入材料簇，之后不回改。持续模型从相同初始化出发，依次重新分配 H、C1、C2 的成员，做两轮。每一步固定其余两期，精确解一个容量分配问题；整体多期问题只得到局部解，不宣称全局最优。单调性与独立松弛界为

$$V_{\rm history}\le V_{\rm shared}\le\sum_{e<f}\omega_{ef}V_{ef}.$$

该界仅证明程序与约束一致，不能证明识别和位移更准确。

## 位姿目标和连续更新

三个匹配版本使用同一标度和首回波项：

$$E(G)=1-\frac{V(G)}{Z(G)}+0.25F(G),\qquad
Z=6+\sum_{e<f}\omega_{ef}(\sum_i u_i+\sum_jv_j).$$

$F$ 是三对、双方向所有原生点的 $\log[1+(d_{\rm free}/\sigma)^2]$ 均值，权重总和归一为 2.5。unknown 不产生 free 残差。这是未校准的条件能量，不是完整生成概率；位姿相关的可见支持归一化也可能形成偏好，必须实测检查。

固定当前材料分配和射线片段，联合更新两个位姿的 12 个自由度。对 $z=G_fG_h^{-1}x$，左扰动雅可比为目标块 $[-[z]_\times,I]$、源块 $R_fR_h^\top[[x]_\times,-I]$。使用阻尼最小二乘、步长限制及完整目标回溯；每次回溯重新算曝光与对应，只接受实际下降。最多 4 步，不以损失下降代替位移合格。

目前是**离散实测材料簇＋连续位姿**；没有可学习连续表面、显式未知表面形状或校准的拒绝模块。不能称为已完成“连续未知材料表面联合重建”。

## 比较与计算边界

四方法：ordinary_union（旧几何/SIFT/射线/native 分数与普通对应配准）、visibility_pair（独立容量匹配）、history_union（只追加历史材料）、shared_transport（允许后期证据回改）。所有方法相同公开输入、相同最终候选池和有限身份状态图。

保留全部旧身份状态，加入两类公共初始指派和最多 32 个合法邻居；这不是全部 $24!^2$ 身份全局搜索。每个访问身份三元组完整比较 H–C1 与 H–C2 候选的笛卡尔积。独立容量目标是两个共享目标的严格下界；只有它已经不能改善当前两个最优值时才跳过材料聚类，不裁掉可能取胜的候选。

取四方法离散全局最优状态中的身份三元组，共同的最多四个不同起点，各做四种连续更新；所有端点再次共享评分。记录更新来源，避免把“只有新模型拿到新候选”算贡献。同候选不等于同 FLOPs，额外分配调用与运行时间必须公开。旧普通控制也获得新模型生成的端点，是较强的候选共享检验。

主指标：身份两期均正确、三个时间段固定历史材料锚点的 P95 位移误差均不超过 5 cm。完整保留背面八对象。评分前预测封存，私有位姿只在评分/诊断阶段读取。oracle 仅检查候选上限，不作为实际成功。

## 创新性边界及已核对的相关工作

- [Partial Permutation Synchronization, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Fast_Accurate_and_Memory-Efficient_Partial_Permutation_Synchronization_CVPR_2022_paper.pdf)：部分多对象匹配与一致性已有系统研究。不能将允许缺失的共享对应、材料簇或同步本身叫原创。
- [Wednesday, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Jin_Multiway_Point_Cloud_Mosaicking_with_Diffusion_and_Global_Optimization_CVPR_2024_paper.pdf)：部分重叠点云的多视配准、配对重叠推断与全局位姿优化已有工作。联合多期本身不足以构成贡献。
- [RGB-D low-overlap and photometric inconsistency, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Shou_Unsupervised_RGB-D_Point_Cloud_Registration_for_Scenes_with_Low_Overlap_ICCV_2025_paper.html)：重叠对应筛选与光度适应已有更接近的公开工作，后续强方法比较不能只停留在老 ICP。

以上仅核对公开原始摘要/文本，不冒称完整系统综述或已运行作者模型。当前可检验假设是：在相同实际支持与候选下，**可见性分型的部分材料容量与后期纠错，能否提高跨期固定材料位移量化**。未获对照净增之前，不宣称创新性成立；即使 TRAIN 净增，也仍需独立多场景和已发表强方法。
