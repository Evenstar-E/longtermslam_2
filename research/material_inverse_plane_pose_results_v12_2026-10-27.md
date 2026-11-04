# V12：真实生成连续位姿与普通传播的完整任务结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: VERIFIED（完整运行与目标重算；不是泛化/原创确认）
- Version Label: material_inverse_plane_pose_proposals_v12

## 本轮做了什么

上轮必要观测修复已经完成，因此本轮优先新位姿和完整主线收益。实现普通H1/C12、H2/C12组合；恢复后逆深度场的正反向连续RGB-D配准；独立与共享同源材料伪似然及等预算12维局部下降。每种生成器端点进入同一个最终池，两个材料模型不独占自己的好初值。原有全部Cartesian位姿、实际耦合端点、有限身份状态均保留。

完整主线：24构件×4外观×8方法=768行。无新网络训练、无新增数据或隐藏真位姿生成；运行结束封存后才加载私有标注评分。数学、预算和近似见[模型契约](material_inverse_plane_pose_model_v12_2026-10-27.md)。

## 完整任务成功数（每列24项）

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| retained_points | 9 | 16 | 14 | 16 |
| retained_patch | 7 | 16 | 14 | 16 |
| propagated_points | 8 | 16 | 14 | 16 |
| propagated_patch | 7 | 16 | 14 | 16 |
| continuous_points | 10 | 16 | 14 | 16 |
| continuous_patch | 11 | 16 | 14 | 16 |
| independent_material | 12 | 16 | 14 | 16 |
| shared_material | 12 | 16 | 14 | 16 |

身份与H1/H2/C12三段固定历史材料锚点95%误差同时≤5cm。不是当前可见点表面距离，也不是单边位姿误差。

还必须与旧强控制比较，不能只胜过V11较弱分支就称收益：

| 已保留V9强控制 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| surface_geometry | 9 | 16 | 14 | 16 |
| surface_rgb_fusion | 9 | 16 | 16 | 16 |
| projective_points | 11 | 16 | 14 | 16 |

跨版本表示/查询位置改变，这个表是任务水平参照，不是纯共享因子消融；共享与独立的纯公式差异只看V12同最终候选两行。

旧强方法逐任务增减，防止总数掩盖退步：[{"variant": "flat", "old_method": "projective_points", "gained": [1, 3, 13], "lost": [9, 21], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "flat", "old_method": "surface_rgb_fusion", "gained": [1, 3, 13], "lost": [], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "rich", "old_method": "projective_points", "gained": [], "lost": [], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "rich", "old_method": "surface_rgb_fusion", "gained": [], "lost": [], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "repeated", "old_method": "projective_points", "gained": [], "lost": [], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "repeated", "old_method": "surface_rgb_fusion", "gained": [], "lost": [22, 23], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "changed", "old_method": "projective_points", "gained": [], "lost": [], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}, {"variant": "changed", "old_method": "surface_rgb_fusion", "gained": [], "lost": [], "scope": "Cross-version task comparison, not pure same-representation sharing effect."}]。

背面8项的实际成功数：

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| retained_points | 0 | 0 | 0 | 0 |
| retained_patch | 0 | 0 | 0 | 0 |
| propagated_points | 0 | 0 | 0 | 0 |
| propagated_patch | 0 | 0 | 0 | 0 |
| continuous_points | 0 | 0 | 0 | 0 |
| continuous_patch | 0 | 0 | 0 | 0 |
| independent_material | 0 | 0 | 0 | 0 |
| shared_material | 0 | 0 | 0 | 0 |

![完整任务比较](../results/material_inverse_plane_pose_proposals_v12/task_comparison.png)

## 候选是否真的补解

| 外观 | 原池 | 有限预算传播 | 连续共同池 | 全原池传播诊断 | 预算漏掉的合格构件 |
|---|---:|---:|---:|---:|---|
| flat | 14 | 15 | 16 | 15 | [] |
| rich | 17 | 17 | 17 | 17 | [] |
| repeated | 16 | 16 | 16 | 16 | [] |
| changed | 17 | 17 | 17 | 17 | [] |

所有oracle均在正确身份下用私有标注离线诊断，**不等于推断成功**。全原池传播仅诊断所有既有A/C、B/C组合，没有评分选解，不能冒充新的高预算算法结果。有限预算是每边每种分数前2的并集，不承诺穷举新增传播模式。连续共同池含普通和两材料生成器全部端点，其收益不能单独归于共享模型。

上表“全原池传播诊断”也合并相同旧保留候选，确保和有限预算传播列的oracle可比；原始CSV另存纯传播结果，不能把未合并旧解的纯传播总数与合并池直接比较。

## 同候选实际增减

- flat / retained_points → propagated_points: 新增[3, 13]，丢失[2, 9, 15]，共同正确身份误差变化中位数0.0m。
- flat / retained_patch → propagated_patch: 新增[3]，丢失[9]，共同正确身份误差变化中位数0.0m。
- flat / propagated_points → continuous_points: 新增[1, 2, 16]，丢失[10]，共同正确身份误差变化中位数-0.0035155236258082034m。
- flat / propagated_patch → continuous_patch: 新增[5, 10, 15, 16]，丢失[]，共同正确身份误差变化中位数-0.0030400305553221867m。
- flat / continuous_patch → independent_material: 新增[1]，丢失[]，共同正确身份误差变化中位数0.0m。
- flat / independent_material → shared_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- rich / retained_points → propagated_points: 新增[]，丢失[]，共同正确身份误差变化中位数-0.00014211936157899462m。
- rich / retained_patch → propagated_patch: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- rich / propagated_points → continuous_points: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- rich / propagated_patch → continuous_patch: 新增[]，丢失[]，共同正确身份误差变化中位数-4.172747062008986e-05m。
- rich / continuous_patch → independent_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- rich / independent_material → shared_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- repeated / retained_points → propagated_points: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- repeated / retained_patch → propagated_patch: 新增[]，丢失[]，共同正确身份误差变化中位数-0.00017726421169933455m。
- repeated / propagated_points → continuous_points: 新增[]，丢失[]，共同正确身份误差变化中位数-0.00028701611581232085m。
- repeated / propagated_patch → continuous_patch: 新增[]，丢失[]，共同正确身份误差变化中位数-0.000270729421398595m。
- repeated / continuous_patch → independent_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- repeated / independent_material → shared_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- changed / retained_points → propagated_points: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- changed / retained_patch → propagated_patch: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- changed / propagated_points → continuous_points: 新增[]，丢失[]，共同正确身份误差变化中位数-9.880590891523993e-05m。
- changed / propagated_patch → continuous_patch: 新增[]，丢失[]，共同正确身份误差变化中位数-3.822063174285795e-06m。
- changed / continuous_patch → independent_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。
- changed / independent_material → shared_material: 新增[]，丢失[]，共同正确身份误差变化中位数0.0m。

### 保留全部rich背面构件的具体误差

| rich背面构件 | 旧point | 连续point | 独立材料 | 共享材料 | 共同池oracle（仅诊断） |
|---|---:|---:|---:|---:|---:|
| 4 | 正确ID / 62.0cm | 正确ID / 62.3cm | 正确ID / 62.2cm | 正确ID / 62.2cm | 32.6cm |
| 6 | 错ID / 530.2cm | 错ID / 551.1cm | 错ID / 551.5cm | 错ID / 551.5cm | 24.9cm |
| 7 | 错ID / 545.7cm | 错ID / 545.7cm | 错ID / 545.5cm | 错ID / 545.5cm | 43.6cm |
| 8 | 错ID / 164.9cm | 正确ID / 77.4cm | 正确ID / 77.4cm | 正确ID / 77.4cm | 4.6cm |
| 11 | 错ID / 604.0cm | 正确ID / 76.0cm | 正确ID / 76.0cm | 正确ID / 76.0cm | 39.1cm |
| 14 | 错ID / 615.2cm | 正确ID / 82.7cm | 正确ID / 82.6cm | 正确ID / 82.6cm | 39.3cm |
| 17 | 正确ID / 78.0cm | 正确ID / 78.9cm | 正确ID / 78.7cm | 正确ID / 78.7cm | 27.9cm |
| 20 | 正确ID / 75.7cm | 正确ID / 75.7cm | 正确ID / 75.7cm | 正确ID / 75.7cm | 37.8cm |

误差为三段固定历史锚点95%误差的最大值，阈值5cm；错ID即不合格，不能因为其几何误差小而计为成功。没有只挑最好构件。

## 原因判定规则

按已经封存的预测进行实际分类：

[
  {
    "variant": "flat",
    "categories": {
      "complete_task_success": 12,
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 7,
      "good_pose_exists_but_identity_not_selected": 4
    },
    "rear_categories": {
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 7
    },
    "actual_common_candidate_sharing_gained": [],
    "actual_common_candidate_sharing_lost": [],
    "shared_vs_independent_net_qualified": 0,
    "shared_vs_v11_normal_points_net_qualified": 3
  },
  {
    "variant": "rich",
    "categories": {
      "complete_task_success": 16,
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 6,
      "good_pose_exists_but_pose_objective_selects_bad": 1
    },
    "rear_categories": {
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 6,
      "good_pose_exists_but_pose_objective_selects_bad": 1
    },
    "actual_common_candidate_sharing_gained": [],
    "actual_common_candidate_sharing_lost": [],
    "shared_vs_independent_net_qualified": 0,
    "shared_vs_v11_normal_points_net_qualified": 0
  },
  {
    "variant": "repeated",
    "categories": {
      "complete_task_success": 14,
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 7,
      "good_pose_exists_but_identity_not_selected": 2
    },
    "rear_categories": {
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 7
    },
    "actual_common_candidate_sharing_gained": [],
    "actual_common_candidate_sharing_lost": [],
    "shared_vs_independent_net_qualified": 0,
    "shared_vs_v11_normal_points_net_qualified": 0
  },
  {
    "variant": "changed",
    "categories": {
      "complete_task_success": 16,
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 6,
      "good_pose_exists_but_pose_objective_selects_bad": 1
    },
    "rear_categories": {
      "no_good_pose_and_no_H_patch_hit_at_true_pose": 1,
      "no_good_pose_despite_some_true_H_patch_support": 6,
      "good_pose_exists_but_pose_objective_selects_bad": 1
    },
    "actual_common_candidate_sharing_gained": [],
    "actual_common_candidate_sharing_lost": [],
    "shared_vs_independent_net_qualified": 0,
    "shared_vs_v11_normal_points_net_qualified": 0
  }
]

真实姿态附近的局部导数诊断：

[
  {
    "variant": "flat",
    "rear_with_three_patches_having_any_supported_pixel": 5,
    "rear_with_rank6_geometry": 5,
    "rear_with_rank6_geometry_texture": 5,
    "rear_no_supported_pixel": [
      4
    ]
  },
  {
    "variant": "rich",
    "rear_with_three_patches_having_any_supported_pixel": 5,
    "rear_with_rank6_geometry": 5,
    "rear_with_rank6_geometry_texture": 5,
    "rear_no_supported_pixel": [
      4
    ]
  },
  {
    "variant": "repeated",
    "rear_with_three_patches_having_any_supported_pixel": 5,
    "rear_with_rank6_geometry": 5,
    "rear_with_rank6_geometry_texture": 5,
    "rear_no_supported_pixel": [
      4
    ]
  },
  {
    "variant": "changed",
    "rear_with_three_patches_having_any_supported_pixel": 5,
    "rear_with_rank6_geometry": 5,
    "rear_with_rank6_geometry_texture": 5,
    "rear_no_supported_pixel": [
      4
    ]
  }
]

这里只测试本轮48块、前向H2与top2视图。局部数值秩不等于全局唯一性；深度噪声会使几何矩阵假性满秩，重涂/照明会让高纹理梯度不符合材料一致性。因此不能据此声称整个原始数据可观测或不可观测。

- oracle新增而实际没有新增：现有候选已经能表达正确解，需优先检查真实量测目标与身份偏好；不能继续用“完全没好位姿”解释。
- 全传播oracle有而有限预算oracle无：固定预算排序/裁剪存在工程瓶颈；先给所有方法同样恢复候选，不能给自己模型独占。
- oracle也没有：本轮传播/局部优化未产生解，不能由此推出原始图像没有信息。
- 共享只降低自己的能量或与独立持平：不形成论文核心独立收益；共享Bernoulli边缘化本身不是创新。

共享公式展开后只多了π(1−π)(1−η)²s1s2的共同支撑奖励。重复纹理两期都给错误表面提供高支撑时，这个因子没有真实性证据；若实测失败，不扫π、λ或轮数来包装创新。

本公式固定权重下共享修正上界B=0.055641。19个实际正确身份错误姿态对照中，真位姿相对错误姿态的独立能量差超过B的有19个；某一实际已生成合格候选差距超过B的有2个。完整表在`shared_rebate_capacity.csv`。这只能证明指定两候选之间本公式无法反转排序，不能证明所有合格位姿都不可能获胜，也不是原始数据的不可辨识性证明。

## 本轮研究决策与下一项具体实现

链路确实前进到新姿态生成与完整任务；flat从上轮有向point9到12，较旧V9强point11是净增1（新增3、丢失2）。其他外观未获成功数净增，重复纹理还低于旧融合16。**共享与独立成功集合全同，不确认共享主创新，停止当前Bernoulli奖励公式的调参路线。**连续优化带来的工程收益保留。

rich/changed的背面失败分成：构件4在当前48块上真姿态无命中；另外6项有一些真实材料支持却没有生成合格姿态；构件8已有4.63cm合格候选，但选中约77.44cm的错误姿态，其独立目标相对好候选优势约0.3026，远超共享修正上界0.0556。全旧模式传播没有新增背面合格解，也没有预算漏掉的合格任务，不再以增加传播种子解释当前失败。

下一模块拟为`inspection_decisions/material_surface_correspondence_v13.py`（**尚未实现**）：显式建立局部材料坐标与多尺度几何/纹理对应变量，利用它们生成全局位姿，并让可见/遮挡/未覆盖状态参与对应代价；不再只给旧表面分数叠加共同命中奖励。相同V11真实RGB-D、实际逆深度支撑与旧候选；必要对照为普通多尺度几何纹理匹配、独立部分对应、共享对应，全部生成器端点共用。优先验收背面6项缺解能否补解、构件8是否真实选中合格解、最后才检验同候选共享净增。若仅普通匹配产生收益归工程；若新模型无净增就不进入更多训练轮数/独立大实验。相关已有方法和强基线需在实现前核验，不能仅凭模块名/新公式宣称新颖。

## 验证、边界与产物

- 24项单元检查，最大导数误差约2.10e-10。
- 11618评分/状态/SE3/下降路径核对；6252真实优化路径；选中目标最大重建误差0。
- 每外观一个完整Cartesian无剪枝复核：[{"variant": "flat", "pose_pairs": 1716, "max_error": 0.0}, {"variant": "rich", "pose_pairs": 1935, "max_error": 0.0}, {"variant": "repeated", "pose_pairs": 2025, "max_error": 0.0}, {"variant": "changed", "pose_pairs": 2070, "max_error": 0.0}]。
- 各外观生成统计与耗时：{"flat": {"complete": true, "seconds": 1262.1792953000404, "counters": {"composed_candidates": 6217, "ordinary_fits": 2056, "ordinary_improved": 1658, "independent_fits": 296, "shared_fits": 296, "joint_improved": 206}, "actual_material_cartesian": 398030, "triples": 286, "states": 202, "private_labels_read": false, "same_material_candidates": true, "equal_flops": false}, "rich": {"complete": true, "seconds": 484.405662199948, "counters": {"composed_candidates": 4358, "ordinary_fits": 980, "ordinary_improved": 781, "independent_fits": 143, "shared_fits": 143, "joint_improved": 63}, "actual_material_cartesian": 91809, "triples": 137, "states": 197, "private_labels_read": false, "same_material_candidates": true, "equal_flops": false}, "repeated": {"complete": true, "seconds": 523.0731752999127, "counters": {"composed_candidates": 4994, "ordinary_fits": 1024, "ordinary_improved": 788, "independent_fits": 156, "shared_fits": 156, "joint_improved": 73}, "actual_material_cartesian": 103153, "triples": 147, "states": 200, "private_labels_read": false, "same_material_candidates": true, "equal_flops": false}, "changed": {"complete": true, "seconds": 390.3996361999307, "counters": {"composed_candidates": 3505, "ordinary_fits": 776, "ordinary_improved": 616, "independent_fits": 113, "shared_fits": 113, "joint_improved": 46}, "actual_material_cartesian": 70101, "triples": 111, "states": 196, "private_labels_read": false, "same_material_candidates": true, "equal_flops": false}}。
- 一个已见TRAIN，四外观相关；结果是开发证据，无独立场景确认、无作者完整系统强基线结论、无概率校准结论。
- [逐任务结果](../results/material_inverse_plane_pose_proposals_v12/task_metrics.csv)、[分阶段oracle](../results/material_inverse_plane_pose_proposals_v12/oracle_by_stage.csv)、[所有旧模式传播诊断](../results/material_inverse_plane_pose_proposals_v12/full_propagation_oracle.csv)、[真位姿与最佳候选目标诊断](../results/material_inverse_plane_pose_proposals_v12/pose_mechanism.csv)。

执行命令见模型契约；原V1至V11与旧192唯一证据不覆盖。图采用统一0–24坐标与直接数值，不画N=1场景的虚假置信区间。
