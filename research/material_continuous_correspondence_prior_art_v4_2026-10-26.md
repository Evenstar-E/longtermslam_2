# V4已有方法边界与主线贡献位置

本文件用于防止把普通方法组件当成新算法，不是完整系统文献综述，也不据少数论文断言“从未有人做过”。核对时间2026-10-26，来源均为论文出版方原文页。

| 已有工作 | 论文明确采用的机制 | 对当前实现的约束 |
|---|---|---|
| [LoFTR，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html) | 先建立粗像素对应，再细化局部对应 | “粗候选＋局部密集相关＋细定位”不能作新颖性主张 |
| [RoMa，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html) | 鲁棒预训练粗特征、CNN细特征，classification-regression与后续稳健回归 | 局部密集分类再连续坐标细化已有充分先例；当前小CNN不能等同其作者模型或强预训练效果 |
| [CoFiNet，NeurIPS2021](https://papers.nips.cc/paper_files/paper/2021/hash/c85b2ea9a678e74fdc8bafe5d0707c31-Abstract.html) | 无关键点检测的粗到细点云对应 | 从稀疏点转向局部细化也不能单独称3D配准新理论 |
| [PointMBF，ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PointMBF_A_Multi-scale_Bidirectional_Fusion_Network_for_Unsupervised_RGB-D_Point_ICCV_2023_paper.html) | RGB图像与深度生成点云之间多尺度双向特征融合，用于无监督配准 | 当前RGB/高度/支持块10通道CNN不是多层双向pixel-point融合，更不是PointMBF复现；几何纹理联合学习本身早已有 |

## 当前新增模块实际做了什么

位置是**对应生成与几何配准前端**，不是规划或控制。V3固定材料点训练集与多对象任务保留；V4增加局部密集描述子和实测三角片内坐标细化，并用相同普通评分测试这些提案。没有把已发表网络原样拼接当原创；也没有声称有端到端多对象身份/多期位姿训练。

可供研究检验的更窄主张是：在离散目标未采中时，观测域内的连续材料对应是否比相同输入、相同局部样本预算的格点匹配或普通连续几何配准更能量化**同一历史材料点**。这只是待验证的增量，不自动满足论文新颖性。

## 哪些结果不能算突破

- 相对随机网络提升，只证明局部特征学到信息。
- 相对旧稀疏方法提升但与dense相同，主要支持加密采样/细描述子工程价值。
- surface与普通geo相同，不能称提出了更强持续材料模型。
- 真身份下3候选有一个更近，不能替代真实身份选择和位姿成功。
- 给三角片三个实测顶点，不证明片内部不存在物理折角/薄缝；连续插值仍是模型假设。
- 全体旧候选仍在，但新增错误解被普通评分选中，说明候选生成和最终任务收益必须分开。

## 何时才进入下一主模型

先看完整V4：若没有超越dense/geo和强ray的受控收益，就不能靠重命名继续主张连续细化创新。若oracle位姿池增加而实际任务未增加，研究缺口在多期材料关联/身份-位姿联合选择；若oracle也未增，缺口仍是可区分特征、粗候选覆盖或物理观测支持。上述判断须由同一任务的实际分族oracle与得失支持，而不是推测更多训练必然有效。

下一创新候选必须给出多期相同材料点共享变量如何改变后验/可辨识性，并用同输入普通联合配准、强特征对应、无共享模型对照；只增加一个局部CNN或表面投影不够。独立新场景与真实跨期精度仍是确认条件。
