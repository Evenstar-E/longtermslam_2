# 有限面片集合关联与共同位姿—局部位移直接推断：执行前契约

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: authorized local algorithm implementation and exploratory experiment
- Origin Date: 2026-10-23
- Verification Status: UNVERIFIED before execution
- Version Label: fragment_union_joint_v1

## 问题与可检验构造

8个重复构件的跨期定量复检。未知地图SE(3)、0或2个真实局部平移、有限历史可见面片、部分视场和真实前景遮挡。目标仍是正确物理身份、正确变化报告且变化位移误差≤10cm。不是只修旧world11一片观测。

旧后端先逐帧求位移argmin再用等方差三维量评分；每帧一对一关联还限制同一构件只能接一片。候选直接保留有限历史三角面片和完整当前掩码：A_fi可以多片指向同一j，U_fj=union_{i:A_fi=j}M_fi。对一个j/f的历史可见投影与U_fj只计算一次删失深度/边界因素，不按碎片反复计算完整模型。共享G和每资产delta_j同时更新，稀疏运动惩罚每资产激活一次。

J_union=sum_fj ell(S_j,G^{-1}T_f,delta_j,U_fj)+4 sum_j I(delta_j!=0)+sum_j||delta_j||²/.6²+12 N_null。ell在已知历史面片投影支持内，重叠像素用Huber深度差/.06，非前景遮挡且不在观测并集的像素用Huber边界距离/1.5px，上限12；无观测并集则可比较历史支持缺失罚12。前景删失只在不属于U_fj且实际深度更近.06m的位置；未知历史面不补齐、不把未观测空间当空。每个资产/帧按实际投影已知像素归一；投影在画面外无数据项，当前赋给它的观测却无投影支持时收取缺失罚，防止把模型挪出图像逃避解释。

这是启发式有限面片直接目标，不称校准似然或全局最优。直接非线性目标做有限局部坐标搜索，A与delta/G交替，每步完整同一目标接受下降；记录轨迹与评价次数。多起点共享原普通注册保留≤3个G，初始delta0/共同旧读出和普通原始点关联候选；不从真值给初值。所有方法最终报告阈值maxabs(delta)>.12与≤10cm评价不变；无当前支持报告UNKNOWN。

## 强对照与消融

legacy_compressed：稳定编号适配器下的原共同E后端。

point_one_to_one：原始有限历史点与当前三维点的直接Huber几何因素，完整G/delta联合更新，每帧一对一；不压缩为逐帧位移。

point_many：相同直接点因素和共同G/delta，允许多片同资产，按碎片计平均几何代价。

point_set_ray：按资产/帧合并原始点后归一的普通直接多片联合配准，加同一已知面片全场景投影的实际深度删失因素。避免把普通方法弱化为无轮廓/无遮挡/无多片的旧方法；不是完整EM-Fusion代码复现。

union_fixed_pose：同候选集合因素/多片关联/位移，固定注册G；隔离仅集合表示与允许G共同优化。

union_joint：完整候选。候选与普通直接方法同输入、初始模式和局部搜索轮数上限，原生每次因素计算成本不同，报告CPU与评价次数，不宣称同FLOP。主要比较质量与获取相同证据下的收益，不把额外视点当算法收益。

## 文献边界

[Co-Fusion](https://arxiv.org/abs/1706.06629)已实现RGBD多物体分割、运动跟踪与重建；[EM-Fusion](https://emfusion.is.tue.mpg.de/)及[原论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Strecke_EM-Fusion_Dynamic_Object-Level_SLAM_With_Probabilistic_Data_Association_ICCV_2019_paper.pdf)已做物体级直接对齐、概率数据关联和遮挡处理。不能将“联合”“多片”“直接对齐”单独宣称首创。候选具体检验不完整跨期历史、相关地图/身份模式、有限面片并集因素与稀疏变化报告的构造；是否超出现有方法尚待完整文献及外部实现比较。即使比本地普通对照好，也仅为增量实验信号，不自动确认创新。

## 数据及判定

开发：完整复用已观察12世界，每世界固定动作1短后退和6右转，各2帧当前观测，24端点。旧失败world11保留，不能把它排除或只对它调碎片阈值。

新世界：12个未观察种子22532026..37，前4静态后8各2移动，程序族仍同重复立面，不称独立真实域。相同固定动作1/6，各24新端点；同一世界两动作相关，32变化报告机会对应16个不同物理移动，分别按动作报告，不能充作32独立真实事件。冻结实现后才能生成/运行；预览/历史/动作RGBD由真实API取得，原始输入和方法输出先保存，再读取owner/truth评价。

本轮专测感知后端，动作固定：不得称新主动策略收益。相同行程/采集/返航费用全部记录。每组所有6方法和失败全保留；10cm门、UNKNOWN和误报惩罚不改。若候选对普通point_set_ray无实际任务质量增益，不调权重/门槛选优、不扩同分布样本宣称创新。若只有相对旧压缩或一对一有收益，则归为共同后端/关联表示改善，不确认集合因素原创。
