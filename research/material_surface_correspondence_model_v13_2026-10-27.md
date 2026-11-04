# V13：带观测掩码的材料坐标、方向多解与显式跨期对应

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: 原型与10项正确性检查完成；完整效果见本版结果报告。
- Version Label: material_surface_correspondence_v13

## 优先级与可证伪假设

V12的必要观测修复和连续配准已完成。rich/changed背面6项仍缺合格姿态；构件8已有4.63cm候选却选77.44cm错误姿态。共享Bernoulli奖励无同候选成功收益，且指定两候选间0.3026目标差超过0.0556修正能力。因此本轮优先改变**材料对应证据和全局位姿生成**，不继续扫旧共享奖励、普通传播种子、GN步数或扩无关小场景。

假设：历史与当前确实共有的少量表面，如果能在实际物理尺度上读取纹理并保留局部方向多解，就可以直接生成旧局部优化缺失的SE(3)。同一真实材料在多个时期的对应可用共享容量约束排除互不相容的链接。前者可能只是既有方法的工程收益；后者必须超过同表示的独立对应才能支持新的任务机制。

本轮作用模块为感知中的材料对应与跨期位姿估计，不是无人机控制或路径规划。

## 已核验的已有工作边界

| 一手来源 | 已有机制 | 对本项目的限制 |
|---|---|---|
| [PointMBF，ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PointMBF_A_Multi-scale_Bidirectional_Fusion_Network_for_Unsupervised_RGB-D_Point_ICCV_2023_paper.html) | 多尺度双向几何/图像特征融合 | “同时利用空间与纹理”不能直接称原创 |
| [GeoTransformer，CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Qin_Geometric_Transformer_for_Fast_and_Robust_Point_Cloud_Registration_CVPR_2022_paper.html) | 不变几何上下文、对应及局部到全局配准 | 几何一致性/全局生成本身不是创新 |
| [CoFiNet，NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/c85b2ea9a678e74fdc8bafe5d0707c31-Paper.pdf) | 粗到细、密度适应与最优传输对应 | 加入部分对应/传输不能直接称原创 |
| [SpinNet，CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ao_SpinNet_Learning_a_General_Surface_Descriptor_for_3D_Point_Cloud_CVPR_2021_paper.html) | 圆柱空间与旋转不变/等变局部表示 | Fourier/环形旋转表示不是新理论 |
| [RoReg，作者项目页](https://hpwang-whu.github.io/RoReg/) | 有向描述子和局部旋转参与全局配准 | 单对应局部方向生成SE(3)已存在 |

上述作者系统本轮未完整运行，不把机制级手工控制冒充其模型基准。待检验候选仅限**实际曝光掩码＋方向多模态＋跨期共享材料对应与SE(3)**是否有同信息独立净增；本表不是穷尽性新颖性证明。

### 完整强方法的实际接入条件（已读官方仓库，尚未部署）

本机实测Python3.12.14、PyTorch2.6.0+cpu，CUDA不可用。不是笼统说“强方法跑不了”，而是目前没有完成相应环境与模型移植：

- [PointMBF官方仓库](https://github.com/phdymz/PointMBF)列出的测试环境为Python3.8/PyTorch1.7.1/CUDA11.1/RTX3090，提供预训练权重入口；本机CPU路径和权重载入未验证。
- [GeoTransformer官方仓库](https://github.com/qinzheng93/GeoTransformer)列出Ubuntu20.04/GCC9.3/Python3.8/PyTorch1.7.1/CUDA11.1测试环境和公开release权重；给出了自定义点云demo及2.5cm体素尺度说明，本项目小构件需显式核对尺度，不可任意缩放后宣称同任务。
- [RoReg官方仓库](https://github.com/HpWang-whu/RoReg)的FCGF骨干要求Linux/CUDA环境，并依赖MinkowskiEngine；本机Windows CPU不能直接视为其已验证运行环境。官方预训练模型入口已核验，未下载或运行。

本轮不改系统、不安装这些依赖，也不把未运行称为强基线失败。后续完整强方法接入应优先实际模型推断与尺度适配，和手工机制对照分别记录。

## 1. 材料局部坐标与真实图像采样

每个epoch/匿名slot继续使用原V9真实512个测量中心，不读CAD、真位姿或材料标签。在5cm邻域用实际点拟合局部平面，RMSE≤4mm、至少8点；用原观测相机确定法向朝向。对合格中心作最远点采样，保留64个。若完全无平面可拟合，使用原观测法向作明确fallback，不删构件。

中心x_i和右手正交基B_i=[t_i1,t_i2,n_i]定义环形坐标：

\[
 q_i(r,\theta)=x_i+r(t_{i1}\cos\theta+t_{i2}\sin\theta),
 \quad r\in\{0.015,0.03,0.06\}\mathrm m,
 \quad\theta=2\pi h/16.
\]

将q投影回**该中心的实际来源图像**，沿用V11的3×3逆深度场与四邻像素RGB。只有同slot、有效测深、平面点与首回波深度差≤6mm且法向兼容才标记m_i(r,θ)=1。其余颜色不参与描述。q是测量约束下的平面查询位置，不是假称所有平面点都有真实表面；有效mask与RGB/实际中心来源全部保存。

令f_i为每块自身有效RGB去均值后、以**该块全部有效观测能量**归一化的向量，缺测元素为0。这里掩码由既有测量固定，不随候选T缩小归一化域。无纹理时f=0。

普通匹配描述D_ij使用FPFH与环形纹理Fourier幅值的余弦距离平均。两者都是标准描述机制。本轮不训练网络；它是能独立定位失败机制的实现，不是PointMBF/SpinNet的复现。

## 2. 保留纹理方向多解，不平均错误旋转

每对源/目标块计算所有16个循环位移的相关：

\[
 c_{ij}(h)=\langle f_i(r,\theta),f_j(r,\theta+2\pi h/16)\rangle,
 \qquad
 p_{ij}(h)=\operatorname{softmax}_h(c_{ij}/0.15).
\]

\[
 \kappa_{ij}=1-H(p_{ij})/\log16.
\]

无纹理时p均匀、κ=0，不编造确定的纹理方向。重复纹理可保留多个峰；评分时按实际候选旋转投影到环坐标后，对p作圆周线性插值，**不用方向均值**。这是相关分数构造的伪似然与熵可靠度，尚未概率校准，不能称正确后验。

设R对应的局部旋转为B_j^T R B_i，θ_R由其二维切向分量atan2给出。每对候选对应的成本：

\[
 d_{ij}(T)=\frac{\|Rx_i+t-y_j\|^2}{0.03^2}+D_{ij}
 +\lambda_n(1-n_j^TRn_i)
 +\lambda_\theta\kappa_{ij}\frac{-\log p_{ij}(\theta_R)}{\log16}.
\]

点对照去掉后两项；法向对照只保留λ_n项；完整方向模型λ_n=0.5、λ_θ=0.75。法向单独消融是必要的，不能把标准法向检查收益归为纹理方向创新。

## 3. 曝光约束与可拒绝的部分对应

沿用实际首回波hit/free/unknown判定。目标实际表面20mm内才为hit；点在测得表面前方是自由空间冲突；遮挡后、域外或无测量处不虚构表面支持。每源中心和每目标中心至多匹配一次，无正收益时允许不对应。

\[
 g_{ij}(T)=h_i(T)+h'_j(T)-d_{ij}(T),
 \qquad
 V_{ef}(T)=\max_{P\ge0}\sum_{ij}P_{ij}g_{ij}(T),
\]

其中实现使用离散可选匹配P∈{0,1}，各行/列和≤1，带中性dummy节点的Hungarian精确求解该pair可选匹配。匹配评分遍历**全部64×64对**，不沿用生成阶段的top2裁剪。未知观测不被当作命中；但本式整体仍是任务能量，不是经过噪声校准的生成概率。

共享模型用每期最多一个节点的材料簇C替代独立P，联合H/C1/C2链接：

\[
 V_{ind}=\sum_{ef}w_{ef}V_{ef},\qquad
 V_{shared}=\max_C\sum_{ef}w_{ef}\sum_{(i,j)\in C}g^{ef}_{ij},
 \quad w=(1,1,0.5).
\]

实现为历史附着后两轮重访的块坐标容量匹配，**不保证共享全局最大值**。容量投影为各pair可行部分对应，所以V_shared≤V_ind。与V12直接增加共同命中奖励不同，本版共享项约束可行对应集合，不能靠凭空加奖励降低能量；该容量工具继承旧V8机制，本身不声称原创。

\[
 E=1-V/Z+\lambda_F F,\quad
 Z=2N_H+1.5N_1+1.5N_2,\quad\lambda_F=0.25.
\]

Z为固定已观测节点域，不用候选可见数量归一化。自由空间项沿用鲁棒log1p与实际射线曝光，不把对面表面法向不兼容改成自由空间证据。

## 4. 真正生成新全局位姿并联合更新

对每源节点保留描述最接近的2个目标用于**有限提案生成**；每对若有纹理取相关前2峰，否则取8个均匀方向。一个局部材料方向假设即可产生：

\[
 R=B_jR_z(\theta_h)B_i^T,\qquad t=y_j-Rx_i.
\]

普通对照用相同描述、同源/目标边进行192轮三点RANSAC。两种生成器各按同一普通几何描述一致性保留3个位姿，全供全部最终方法共用。因源/目标中心独立采样、法向有噪声，这些是初值而非精确对应保证。方向峰用三点抛物线作有限子bin细化；多峰保留，不提前求平均。

**预先记录的模型风险：**环形循环相关只建模绕中心的旋转；独立采样的两个中心可能相差数毫米乃至数厘米，此时真实关系还含平面内平移，不能严格视为纯SO(2)。最小环半径仅15mm，中心偏差可能与其同量级。事后必须检查方向误差与材料中心偏移、法向一致性的关系，不能把这种模型失配都归为纹理不够、样本少或温度没调好。

该风险已在完整任务仍运行时用解析反例核验，而非任务结果后调参：平面材质ρ(x,y)=0.5+0.4cos(2πx/0.03)，真实旋转0°，两个采样中心平移0.015m。对本版3个半径/16角度，Fourier幅值描述子误差约浮点精度，但真实方向相关为−1，前两个相关峰为45°和135°。源码`check_material_chart_center_shift_v13.py`与JSON保留。这证明纯SO(2)不能处理所有独立中心，不证明数据集全部错误都由此导致。

正确的局部观测关系一般应含平面内偏移：

\[
 f_j(z)\approx f_i(R_{\theta}z+\delta_{ij}),\quad
 \delta_{ij}\in\mathbb R^2,
\]

并需把对应位置y_j+B_j^{tan}δ_ij与SE(3)及实际射线曝光约束关联。**本版尚未估计δ**；不得把这条后续公式写成已实现成果。补平移本身也属已知局部配准工具，核心潜在收益仍须相同连续位置/普通配准控制来检验跨期材料耦合。

H1/H2新初值的组合、普通C12一致性传播和所有旧实际端点构成公共耦合候选池。采用相同普通fusion/phase最优起点并集，分别运行点/方向的独立/共享3步联合SE(3)GN与重匹配；点残差提供方向，完整目标回溯接受。**这不是含可见性跳变和phase项的精确EM或全局收敛证明。**全部优化端点再次共用。

## 5. 实际搜索范围与预算：明确偏离原草案

数据cache的配置曾写“所有旧Cartesian继续穷举”，它只是观测准备时的计划，未执行任何任务推断。本轮实际执行契约在`run_material_surface_correspondence_v13.py::ACTUAL_SCOPE`和`frozen_protocol.json`中明确替代该句：

- 完整保留每三元组V12的**所有实际生成耦合候选**及8个实际选中端点（包括原Cartesian胜者）。
- 保留原V12有限身份状态图；所有V13新评分共用同一有限候选池。
- V12完整搜索得到的旧预测逐字段回放作强任务参照。
- **新目标不穷举未被旧方法选中的全部旧Cartesian组合**，不声称与旧V12是纯评分目标消融，也不声称全局候选完备。
- 事后分别记录保留公共池oracle与旧完整池oracle，检查这一范围变化是否丢失已有好解；不能隐去这项限制。

理由：主线当前是形成新的实际对应/全局位姿，而非再次耗费主要预算穷举旧模式。这个取舍不改变新方法之间的共同候选控制，但限制跨版本归因。

新图像查询：每slot64×48个环点，对比V12的48×5像素块，读数约12.8倍；它们来自同一批已采集RGB-D，不是新增采集。所有V13对照共享，跨V12总收益可能来自表示/像素利用，不能全归新联合机制。匿名slot的可用分组仍是仿真输入假设，没有实机分割系统结论。

## 6. 八个全量方法与验收

1. retained_v12：旧独立材料模型精确回放。
2. multiscale_fusion：相同新描述/候选，普通几何纹理近邻融合。
3. partial_points / shared_points：相同点对应，独立/共享。
4. partial_normals / shared_normals：只加入标准法向项。
5. partial_phase / shared_phase：加入带mask、保留多峰的纹理方向项。

全部24构件×4外观×8方法=768项，保留背面8项与重复纹理22/23。先封存预测再读取私有标注。主指标仍是两个身份正确且H1/H2/C12固定历史材料锚点95%误差均≤5cm。额外评价实际选中中心对应的15mm材料坐标精度，要求物理构件身份也正确；不把更多链接或低自身能量等同正确匹配。

验收顺序：新全局候选是否补6项缺解 → 构件8能否实际选好解 → 方向超过同法向控制、共享超过同方向独立控制 → 才谈独立场景/训练/完整作者强基线。一个已见TRAIN、四相关外观不构成泛化与原创确认。若仅普通表示或匹配有效，按工程收益保留；不靠调方向温度、权重、采样预算在这一个场景宣布创新。

## 命令

```powershell
.\.venv-real\Scripts\python.exe -m scripts.check_material_surface_correspondence_v13
.\.venv-real\Scripts\python.exe -m scripts.prepare_material_surface_charts_v13
.\.venv-real\Scripts\python.exe -m scripts.run_material_surface_correspondence_v13
.\.venv-real\Scripts\python.exe -m scripts.score_material_surface_correspondence_v13
```

已经存在的封存输出不可原地重跑覆盖；新版本改模型要用新文件/输出根目录。

## 执行中的必要工程修复（不构成算法收益）

原全量任务在角度接近0/2π处遇到浮点取模结果舍入为16的数组越界，保存在`results/material_surface_correspondence_v13/failure_note.json`，未计任务0分。原源与协议未改。新`material_surface_correspondence_v13b.py`只修复周期索引，增加7项边界/1001点周期检查；新runner/scorer为v13b，实际输出根为`results/material_surface_correspondence_v13_run2`。原真实图像chart cache、超参数与所有研究公式保持不变。模型有效性必须以修复后完整结果判断。
