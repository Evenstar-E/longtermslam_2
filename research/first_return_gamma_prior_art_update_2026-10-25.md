# 本轮首回波模型的先例边界补充

核查日期2026-10-25。仅采用原作者/机构记录或原论文，不把检索未找到当作首创证据。

- [Lambda-Field: A Continuous Counterpart of the Bayesian Occupancy Grid for Risk Assessment, 2019](https://arxiv.org/abs/1903.02285)：原论文提出基于强度场的连续风险表示与数据估计。因此强度/存活场和以路径积分计算事件概率不能作为本工作的广义创新。其导航风险任务与此处历史材料身份任务不同，不能说原方法已解决此任务；也不能反过来因此宣称本组合首创。
- [Online Inference of Joint Occupancy using Forward Sensor Models and Trajectory Posteriors for Deliberate Robot Navigation, CMU-RI-TR-20-47, 2020](https://publications.ri.cmu.edu/online-inference-of-joint-occupancy-using-forward-sensor-models-and-trajectory-posteriors-for-deliberate-robot-navigation)：原机构摘要明确讨论深度射线的单元依赖、前向传感器模型、MRF、噪声和位姿不确定性及GPU推断。因而“考虑首回波可见性/遮挡”“不立即丢弃原始测量”“联合地图与位姿”均不是可直接声称的新思想。
- [Modern MAP inference methods for accurate and fast occupancy grid mapping on higher order factor graphs](https://researchwith.stevens.edu/en/publications/modern-map-inference-methods-for-accurate-and-fast-occupancy-grid/)：原作者机构记录说明将前向传感器模型与现代图模型MAP推断结合已有先例。不能把换求解器、因子图或Gamma共轭视作核心创新。

本轮实际新代码定位：对象自身材料坐标中的命中/曝光统计，保留未知先验，对跨期身份的共享积分和独立积分作同输入全任务比较。这是需要实验检验的窄候选贡献，不是经本次有限检索证明独创的理论。普通Beta占据控制为自实现控制，**并非上述作者系统的完整复现**。若实验收益由它复现，须归为表示/工程修复。

没有下载或运行上述作者系统；没有宣称完成真实跨期强基线。Gamma查表、资产线程、缓存复制均为工程实现。本轮有限先例核查不构成系统性穷尽文献调研。

## 表面表示的附加边界（同日核查）

- [NeuS, 2021](https://arxiv.org/abs/2106.10689)：原文明确指出常规体渲染可带来表面几何偏差，提出以SDF零水平集及一阶无偏的体渲染形式重建表面。因此“用连续SDF替换自由体密度”“修复表面偏差”本身也不能泛称创新。
- [Volume Rendering of Neural Implicit Surfaces / VolSDF, 2021](https://arxiv.org/abs/2106.12052)：原文采用施加于SDF的Laplace CDF建立密度，并研究不透明度近似误差界。密度与表面之间的几何约束、误差有界射线采样均有先例。

据此提出的工作假设，而非原论文替我们证明的结论：本轮8cm分段常率把薄表面位置压进单元，可能存在体素内位置偏差；若要作为失败原因，需要本任务的固定真身份代价排序或网格相位干预证据。不能直接用NeuS在多视图重建中的结论断言本轮全部误差由体密度造成。尚未执行NeuS/VolSDF作者代码；其任务与本任务也不同。
