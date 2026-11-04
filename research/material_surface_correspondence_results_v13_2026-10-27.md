# V13：局部材料坐标、方向多模态和显式跨期对应的全量结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-27
- Verification Status: VERIFIED（完整执行和目标/来源核对，不是原创/泛化确认）
- Version Label: material_surface_correspondence_v13_run2

## 本轮优先推进了什么

V12简单共享奖励已停止调参。本轮真正实现新全局位姿：64个实际材料中心的物理尺度纹理坐标、观测掩码、16方向分布；普通RANSAC与局部方向生成器；显式可拒绝容量对应及跨期材料簇；3步SE(3)联合更新。10项原型检查与7项周期边界检查通过。全部24×4×8=768任务，预测封存后才读取私有标注。

原尝试遇到0/2π浮点边界索引越界，是工程执行错误，未计0分；失败源/协议/进度保留在旧根。v13b仅修索引，图像chart cache/超参数/公式不变。最终根为`results/material_surface_correspondence_v13_run2`。

## 完整任务成功数（每列24项）

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| retained_v12 | 12 | 16 | 14 | 16 |
| multiscale_fusion | 7 | 13 | 10 | 13 |
| partial_points | 9 | 14 | 10 | 14 |
| shared_points | 9 | 13 | 10 | 13 |
| partial_normals | 9 | 14 | 10 | 14 |
| shared_normals | 9 | 13 | 10 | 13 |
| partial_phase | 9 | 14 | 10 | 14 |
| shared_phase | 9 | 13 | 10 | 13 |

判据：两个当前身份都正确，H1/H2/C12三段固定历史材料锚点95%误差全部≤5cm。没有把低自身能量、正确ID或多链接当完整成功。

不能只与较弱分支比较，旧强方法参照仍保留：

| 旧V9强控制 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| projective_points | 11 | 16 | 14 | 16 |
| surface_rgb_fusion | 9 | 16 | 16 | 16 |


跨版本每资产增减：`[{"variant": "flat", "old_method": "projective_points", "gained": [1, 3, 13], "lost": [0, 2, 9, 15, 21]}, {"variant": "flat", "old_method": "surface_rgb_fusion", "gained": [1, 3, 13], "lost": [0, 2, 15]}, {"variant": "rich", "old_method": "projective_points", "gained": [], "lost": [2, 13, 15]}, {"variant": "rich", "old_method": "surface_rgb_fusion", "gained": [], "lost": [2, 13, 15]}, {"variant": "repeated", "old_method": "projective_points", "gained": [], "lost": [2, 15, 18, 21]}, {"variant": "repeated", "old_method": "surface_rgb_fusion", "gained": [], "lost": [2, 15, 18, 21, 22, 23]}, {"variant": "changed", "old_method": "projective_points", "gained": [], "lost": [2, 13, 15]}, {"variant": "changed", "old_method": "surface_rgb_fusion", "gained": [], "lost": [2, 13, 15]}]`。这是不同表示/搜索范围的任务参照，不是纯机制消融。

背面8项保留：

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| retained_v12 | 0 | 0 | 0 | 0 |
| multiscale_fusion | 0 | 0 | 0 | 0 |
| partial_points | 0 | 0 | 0 | 0 |
| shared_points | 0 | 0 | 0 | 0 |
| partial_normals | 0 | 0 | 0 | 0 |
| shared_normals | 0 | 0 | 0 | 0 |
| partial_phase | 0 | 0 | 0 | 0 |
| shared_phase | 0 | 0 | 0 | 0 |

![全部方法和外观的完整任务结果](../results/material_surface_correspondence_v13_run2/task_comparison.png)

## 成功/失败的实际增减

- flat / retained_v12 → multiscale_fusion: 新增[]，丢失[0, 1, 2, 13, 15]。
- flat / multiscale_fusion → partial_points: 新增[1, 13]，丢失[]。
- flat / partial_points → partial_normals: 新增[]，丢失[]。
- flat / partial_normals → partial_phase: 新增[]，丢失[]。
- flat / partial_points → shared_points: 新增[]，丢失[]。
- flat / partial_normals → shared_normals: 新增[]，丢失[]。
- flat / partial_phase → shared_phase: 新增[]，丢失[]。
- rich / retained_v12 → multiscale_fusion: 新增[]，丢失[2, 13, 15]。
- rich / multiscale_fusion → partial_points: 新增[15]，丢失[]。
- rich / partial_points → partial_normals: 新增[]，丢失[]。
- rich / partial_normals → partial_phase: 新增[]，丢失[]。
- rich / partial_points → shared_points: 新增[]，丢失[15]。
- rich / partial_normals → shared_normals: 新增[]，丢失[15]。
- rich / partial_phase → shared_phase: 新增[]，丢失[15]。
- repeated / retained_v12 → multiscale_fusion: 新增[22]，丢失[2, 5, 15, 18, 21]。
- repeated / multiscale_fusion → partial_points: 新增[5]，丢失[22]。
- repeated / partial_points → partial_normals: 新增[]，丢失[]。
- repeated / partial_normals → partial_phase: 新增[]，丢失[]。
- repeated / partial_points → shared_points: 新增[]，丢失[]。
- repeated / partial_normals → shared_normals: 新增[]，丢失[]。
- repeated / partial_phase → shared_phase: 新增[]，丢失[]。
- changed / retained_v12 → multiscale_fusion: 新增[]，丢失[2, 13, 15]。
- changed / multiscale_fusion → partial_points: 新增[15]，丢失[]。
- changed / partial_points → partial_normals: 新增[]，丢失[]。
- changed / partial_normals → partial_phase: 新增[]，丢失[]。
- changed / partial_points → shared_points: 新增[]，丢失[15]。
- changed / partial_normals → shared_normals: 新增[]，丢失[15]。
- changed / partial_phase → shared_phase: 新增[]，丢失[15]。

法向对照是标准机制，方向必须胜过该对照；共享必须胜过同方向独立，才能支持对应机制的独立贡献。普通融合也使用全部新候选，没有给新模型独占好初值。

## 实际选中链接是否真是同材料

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
| partial_points | 408/2985 (13.7%) | 527/3019 (17.5%) | 447/3034 (14.7%) | 533/3016 (17.7%) |
| shared_points | 375/2906 (12.9%) | 495/2930 (16.9%) | 446/2953 (15.1%) | 504/2945 (17.1%) |
| partial_normals | 377/2972 (12.7%) | 535/3026 (17.7%) | 447/3018 (14.8%) | 540/3004 (18.0%) |
| shared_normals | 373/2894 (12.9%) | 496/2937 (16.9%) | 447/2936 (15.2%) | 509/2919 (17.4%) |
| partial_phase | 373/2942 (12.7%) | 539/2873 (18.8%) | 475/2924 (16.2%) | 531/2895 (18.3%) |
| shared_phase | 371/2861 (13.0%) | 518/2792 (18.6%) | 453/2836 (16.0%) | 514/2817 (18.2%) |

分子为真实物理资产相同且材料坐标差≤15mm的链接；分母为实际选中链接。跨三段累计而非独立样本，样本相关；中心允许15mm近似对应，不声称逐像素精确身份。共享链接数量变少不能自动算更准确，需结合主任务。

## 候选链路与旧池保留范围

| 外观 | 保留公共池 | 新全局提案池 | 联合连续池 | 因有限保留丢掉的旧合格资产 |
|---|---:|---:|---:|---|
| flat | 14 | 14 | 14 | [] |
| rich | 17 | 17 | 17 | [] |
| repeated | 16 | 16 | 16 | [] |
| changed | 17 | 17 | 17 | [] |

这些是正确身份下GT离线oracle，**不是算法成功率**。本轮新方法共用有限实际耦合候选；保留所有V12生成候选和8个实际选中端点，V12精确预测回放。但没有为新目标重穷举全部未选旧Cartesian组合。这是预测前写明的预算取舍，不能声称跨V12是纯评分消融。上表与`retained_pool_coverage.csv`公开检查有无丢掉旧好解，不能隐去。

完整执行范围：All24x4 on seenTRAIN. Common measured64-node charts and common finite coupled candidate union for all7new scorers. Retain EVERY V12 generated coupled candidate and every actual V12 per-triple selected endpoint, including old Cartesian winners; V12 exact predictions replayed separately. New objectives do NOT exhaustively re-search all unselected V9 Cartesian combinations. This explicitly supersedes the planned exhaustive-Cartesian statement in the chart-cache config, which only prepared observations. Fixed same V12 finite identity graph. More queried texture pixels thanV12, not same-FLOP or pure information-budget comparison. Same-feature points/normal/phase independent/shared controls isolate actual mechanism. No reference poses/IDs/material labels before prediction seal.

核对未发现相同已访问正确身份三元组内丢失旧合格候选。flat全因子参考oracle16与公共图14之差，是资产22/23的真身份三元组原本就不在两版相同有限身份图中，不是新有限候选截断造成。V12实际12项成功的端点全部保留，新模型退步不能用删掉这些实际好解来解释。

## rich全部背面构件，不挑最好案例

| rich背面资产 | V12实际误差 | 普通融合 | 独立方向 | 共享方向 | 公共池最好误差（仅诊断） |
|---|---:|---:|---:|---:|---:|
| 4 | 正确ID / 62.20cm | 正确ID / 39.19cm | 错ID / 524.43cm | 错ID / 524.43cm | 38.78cm |
| 6 | 错ID / 551.54cm | 错ID / 492.71cm | 正确ID / 67.42cm | 正确ID / 67.42cm | 37.43cm |
| 7 | 错ID / 545.47cm | 错ID / 553.18cm | 错ID / 557.42cm | 错ID / 557.42cm | 46.79cm |
| 8 | 正确ID / 77.44cm | 正确ID / 78.94cm | 错ID / 313.24cm | 错ID / 313.24cm | 4.63cm |
| 11 | 正确ID / 76.01cm | 错ID / 184.50cm | 错ID / 332.36cm | 错ID / 331.60cm | 48.06cm |
| 14 | 正确ID / 82.62cm | 正确ID / 87.61cm | 错ID / 201.21cm | 错ID / 201.21cm | 47.23cm |
| 17 | 正确ID / 78.71cm | 错ID / 190.63cm | 错ID / 342.12cm | 错ID / 342.06cm | 52.96cm |
| 20 | 正确ID / 75.73cm | 正确ID / 46.66cm | 正确ID / 76.04cm | 正确ID / 75.99cm | 46.56cm |


## 对应覆盖与方向假设诊断

同一批原始512中心的采样干预（几何只诊断一次，不能当四次独立验证）：

```json
{
  "summary": [
    {
      "family": "all512",
      "rear_H2_near_edges": 33,
      "rear_with_any_H2_near_edge": 8,
      "rear_q_assisted_qualified": 7,
      "rear_mutual_pairs": 29
    },
    {
      "family": "plane_eligible",
      "rear_H2_near_edges": 0,
      "rear_with_any_H2_near_edge": 0,
      "rear_q_assisted_qualified": 0,
      "rear_mutual_pairs": 0
    },
    {
      "family": "plane_FPS64",
      "rear_H2_near_edges": 0,
      "rear_with_any_H2_near_edge": 0,
      "rear_q_assisted_qualified": 0,
      "rear_mutual_pairs": 0
    },
    {
      "family": "unfiltered_FPS64",
      "rear_H2_near_edges": 9,
      "rear_with_any_H2_near_edge": 7,
      "rear_q_assisted_qualified": 0,
      "rear_mutual_pairs": 9
    }
  ],
  "scope": "Geometry once onrich; all4appearances share geometry, not4 independent replications. Labels read only AFTER seal. SAME original512 measured centers, plane eligibility vs actual64FPS vs unfiltered64FPS, all15mm qnear/GT-association fits are diagnostics NOT inference success. No source/prediction edits, no raw-data impossibility claim."
}
```

**直接根因：**背面8项原512中心保有33对15mm内的跨期近材料点；5cm邻域平面资格筛选后变成0，删除发生在FPS之前。取消平面筛选但仍仅FPS64时只保留9对、分布7构件，给定真对应仍无完整达标任务；使用全部512及真对应时7/8达标。这些是表示干预与标签辅助拟合，**不是新算法7/8成功**。不能说原始图像没有共同材料，也不能把只取消平面筛选而维持64个离散中心视为已解决。

```json
{
  "summary": [
    {
      "variant": "flat",
      "old_qualified_lost_by_finite_retention": [],
      "old_qualified_outside_shared_identity_graph": [
        22,
        23
      ],
      "rear_no_15mm_material_pair": [
        4,
        6,
        7,
        8,
        11,
        14,
        17,
        20
      ],
      "rear_near_edges": 0,
      "rear_near_edges_in_descriptor_top2": 0,
      "rear_oracle_qualified": {
        "ordinary": 0,
        "frame": 0,
        "q_assisted_phase_all_edges": 0,
        "q_assisted_phase_top2_edges": 0,
        "q_assisted_rigid": 0
      }
    },
    {
      "variant": "rich",
      "old_qualified_lost_by_finite_retention": [],
      "old_qualified_outside_shared_identity_graph": [],
      "rear_no_15mm_material_pair": [
        4,
        6,
        7,
        8,
        11,
        14,
        17,
        20
      ],
      "rear_near_edges": 0,
      "rear_near_edges_in_descriptor_top2": 0,
      "rear_oracle_qualified": {
        "ordinary": 1,
        "frame": 1,
        "q_assisted_phase_all_edges": 0,
        "q_assisted_phase_top2_edges": 0,
        "q_assisted_rigid": 0
      }
    },
    {
      "variant": "repeated",
      "old_qualified_lost_by_finite_retention": [],
      "old_qualified_outside_shared_identity_graph": [],
      "rear_no_15mm_material_pair": [
        4,
        6,
        7,
        8,
        11,
        14,
        17,
        20
      ],
      "rear_near_edges": 0,
      "rear_near_edges_in_descriptor_top2": 0,
      "rear_oracle_qualified": {
        "ordinary": 0,
        "frame": 0,
        "q_assisted_phase_all_edges": 0,
        "q_assisted_phase_top2_edges": 0,
        "q_assisted_rigid": 0
      }
    },
    {
      "variant": "changed",
      "old_qualified_lost_by_finite_retention": [],
      "old_qualified_outside_shared_identity_graph": [],
      "rear_no_15mm_material_pair": [
        4,
        6,
        7,
        8,
        11,
        14,
        17,
        20
      ],
      "rear_near_edges": 0,
      "rear_near_edges_in_descriptor_top2": 0,
      "rear_oracle_qualified": {
        "ordinary": 1,
        "frame": 1,
        "q_assisted_phase_all_edges": 0,
        "q_assisted_phase_top2_edges": 0,
        "q_assisted_rigid": 0
      }
    }
  ],
  "phase_offset_bins": [
    {
      "offset_min_m": 0,
      "offset_max_m": 0.003,
      "rows": 100,
      "median_top2_phase_error_deg": 5.4855174359076795,
      "median_descriptor_rank": 2.0,
      "scope": "Correlated observation diagnostic, not causal intervention or independent scenes."
    },
    {
      "offset_min_m": 0.003,
      "offset_max_m": 0.01,
      "rows": 696,
      "median_top2_phase_error_deg": 6.959648968719839,
      "median_descriptor_rank": 6.0,
      "scope": "Correlated observation diagnostic, not causal intervention or independent scenes."
    },
    {
      "offset_min_m": 0.01,
      "offset_max_m": 0.02,
      "rows": 2252,
      "median_top2_phase_error_deg": 11.523756461696934,
      "median_descriptor_rank": 14.0,
      "scope": "Correlated observation diagnostic, not causal intervention or independent scenes."
    },
    {
      "offset_min_m": 0.02,
      "offset_max_m": 0.03,
      "rows": 3188,
      "median_top2_phase_error_deg": 24.320338091763826,
      "median_descriptor_rank": 21.0,
      "scope": "Correlated observation diagnostic, not causal intervention or independent scenes."
    }
  ],
  "scope": "Private GT association/oracle diagnostics AFTER seal, not predictions. q-assistedphase uses measured correlations but GT selects near-material links and best poses. 32 endpoints per side bounds the oracle probe. Retained-pool audit explicitly checks changed search scope. No raw-data impossibility claim."
}
```

q-assisted分支使用私有标签筛材料近邻并挑选最有利姿态，只是诊断，未进入任何预测。每边保留32个最有利端点的有界探针不构成全局不可行证明。实际普通/方向生成器oracle也与保留公共池合并，不能当作者完整系统结果。

方向误差与中心偏移的分箱是相关证据，未做平移干预；环形模型只处理旋转而中心独立采样含平移是已记录假设风险，不能仅靠相关性就证明因果。

独立解析反例已核验：同一3cm周期平面条纹、真实旋转0°、中心偏移1.5cm，幅值描述近乎相同，但本版前两个方向峰45°/135°。这证明纯旋转假设存在结构缺陷；它不是实际任务失败次数，也不能证明全部实际错误的原因。实际材料点关系需要Rθ与二维偏移δ联合建模，**本版未实现δ估计**。见模型契约及`results/material_chart_center_shift_counterexample_v13.json`。

## 主线决策的实际证据

```json
[
  {
    "variant": "flat",
    "shared_phase_qualified": 9,
    "shared_phase_rear_qualified": 0,
    "phase_vs_normal_net": 0,
    "sharing_phase_net": 0,
    "shared_phase_vs_replay_net": -3,
    "categories": {
      "good_pose_exists_but_objective_selects_bad_pose": 4,
      "complete_task_success": 9,
      "no_good_pose_and_no_15mm_H2_node_pair": 8,
      "good_pose_exists_but_wrong_identity_selected": 1,
      "correct_identity_outside_finite_graph": 2
    },
    "rear_categories": {
      "no_good_pose_and_no_15mm_H2_node_pair": 8
    },
    "controlled_comparisons": [
      {
        "variant": "flat",
        "baseline": "retained_v12",
        "candidate": "multiscale_fusion",
        "gained": [],
        "lost": [
          0,
          1,
          2,
          13,
          15
        ],
        "common_correct_ids": 18,
        "paired_error_change_median_m": 0.020946219438070422
      },
      {
        "variant": "flat",
        "baseline": "multiscale_fusion",
        "candidate": "partial_points",
        "gained": [
          1,
          13
        ],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "flat",
        "baseline": "partial_points",
        "candidate": "partial_normals",
        "gained": [],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "flat",
        "baseline": "partial_normals",
        "candidate": "partial_phase",
        "gained": [],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "flat",
        "baseline": "partial_points",
        "candidate": "shared_points",
        "gained": [],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "flat",
        "baseline": "partial_normals",
        "candidate": "shared_normals",
        "gained": [],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "flat",
        "baseline": "partial_phase",
        "candidate": "shared_phase",
        "gained": [],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      }
    ]
  },
  {
    "variant": "rich",
    "shared_phase_qualified": 13,
    "shared_phase_rear_qualified": 0,
    "phase_vs_normal_net": 0,
    "sharing_phase_net": -1,
    "shared_phase_vs_replay_net": -3,
    "categories": {
      "complete_task_success": 13,
      "good_pose_exists_but_objective_selects_bad_pose": 3,
      "no_good_pose_and_no_15mm_H2_node_pair": 7,
      "good_pose_exists_but_wrong_identity_selected": 1
    },
    "rear_categories": {
      "no_good_pose_and_no_15mm_H2_node_pair": 7,
      "good_pose_exists_but_wrong_identity_selected": 1
    },
    "controlled_comparisons": [
      {
        "variant": "rich",
        "baseline": "retained_v12",
        "candidate": "multiscale_fusion",
        "gained": [],
        "lost": [
          2,
          13,
          15
        ],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.023416394762831055
      },
      {
        "variant": "rich",
        "baseline": "multiscale_fusion",
        "candidate": "partial_points",
        "gained": [
          15
        ],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": -0.0018154435338593698
      },
      {
        "variant": "rich",
        "baseline": "partial_points",
        "candidate": "partial_normals",
        "gained": [],
        "lost": [],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "rich",
        "baseline": "partial_normals",
        "candidate": "partial_phase",
        "gained": [],
        "lost": [],
        "common_correct_ids": 18,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "rich",
        "baseline": "partial_points",
        "candidate": "shared_points",
        "gained": [],
        "lost": [
          15
        ],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "rich",
        "baseline": "partial_normals",
        "candidate": "shared_normals",
        "gained": [],
        "lost": [
          15
        ],
        "common_correct_ids": 20,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "rich",
        "baseline": "partial_phase",
        "candidate": "shared_phase",
        "gained": [],
        "lost": [
          15
        ],
        "common_correct_ids": 18,
        "paired_error_change_median_m": 0.0
      }
    ]
  },
  {
    "variant": "repeated",
    "shared_phase_qualified": 10,
    "shared_phase_rear_qualified": 0,
    "phase_vs_normal_net": 0,
    "sharing_phase_net": 0,
    "shared_phase_vs_replay_net": -4,
    "categories": {
      "complete_task_success": 10,
      "good_pose_exists_but_objective_selects_bad_pose": 4,
      "no_good_pose_and_no_15mm_H2_node_pair": 8,
      "good_pose_exists_but_wrong_identity_selected": 2
    },
    "rear_categories": {
      "no_good_pose_and_no_15mm_H2_node_pair": 8
    },
    "controlled_comparisons": [
      {
        "variant": "repeated",
        "baseline": "retained_v12",
        "candidate": "multiscale_fusion",
        "gained": [
          22
        ],
        "lost": [
          2,
          5,
          15,
          18,
          21
        ],
        "common_correct_ids": 17,
        "paired_error_change_median_m": 0.03089255090326315
      },
      {
        "variant": "repeated",
        "baseline": "multiscale_fusion",
        "candidate": "partial_points",
        "gained": [
          5
        ],
        "lost": [
          22
        ],
        "common_correct_ids": 20,
        "paired_error_change_median_m": -0.00013868731868377573
      },
      {
        "variant": "repeated",
        "baseline": "partial_points",
        "candidate": "partial_normals",
        "gained": [],
        "lost": [],
        "common_correct_ids": 22,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "repeated",
        "baseline": "partial_normals",
        "candidate": "partial_phase",
        "gained": [],
        "lost": [],
        "common_correct_ids": 22,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "repeated",
        "baseline": "partial_points",
        "candidate": "shared_points",
        "gained": [],
        "lost": [],
        "common_correct_ids": 22,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "repeated",
        "baseline": "partial_normals",
        "candidate": "shared_normals",
        "gained": [],
        "lost": [],
        "common_correct_ids": 22,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "repeated",
        "baseline": "partial_phase",
        "candidate": "shared_phase",
        "gained": [],
        "lost": [],
        "common_correct_ids": 22,
        "paired_error_change_median_m": 0.0
      }
    ]
  },
  {
    "variant": "changed",
    "shared_phase_qualified": 13,
    "shared_phase_rear_qualified": 0,
    "phase_vs_normal_net": 0,
    "sharing_phase_net": -1,
    "shared_phase_vs_replay_net": -3,
    "categories": {
      "complete_task_success": 13,
      "good_pose_exists_but_objective_selects_bad_pose": 4,
      "no_good_pose_and_no_15mm_H2_node_pair": 7
    },
    "rear_categories": {
      "no_good_pose_and_no_15mm_H2_node_pair": 7,
      "good_pose_exists_but_objective_selects_bad_pose": 1
    },
    "controlled_comparisons": [
      {
        "variant": "changed",
        "baseline": "retained_v12",
        "candidate": "multiscale_fusion",
        "gained": [],
        "lost": [
          2,
          13,
          15
        ],
        "common_correct_ids": 19,
        "paired_error_change_median_m": 0.022850700466978978
      },
      {
        "variant": "changed",
        "baseline": "multiscale_fusion",
        "candidate": "partial_points",
        "gained": [
          15
        ],
        "lost": [],
        "common_correct_ids": 17,
        "paired_error_change_median_m": -0.0023934646304460266
      },
      {
        "variant": "changed",
        "baseline": "partial_points",
        "candidate": "partial_normals",
        "gained": [],
        "lost": [],
        "common_correct_ids": 18,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "changed",
        "baseline": "partial_normals",
        "candidate": "partial_phase",
        "gained": [],
        "lost": [],
        "common_correct_ids": 17,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "changed",
        "baseline": "partial_points",
        "candidate": "shared_points",
        "gained": [],
        "lost": [
          15
        ],
        "common_correct_ids": 18,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "changed",
        "baseline": "partial_normals",
        "candidate": "shared_normals",
        "gained": [],
        "lost": [
          15
        ],
        "common_correct_ids": 19,
        "paired_error_change_median_m": 0.0
      },
      {
        "variant": "changed",
        "baseline": "partial_phase",
        "candidate": "shared_phase",
        "gained": [],
        "lost": [
          15
        ],
        "common_correct_ids": 17,
        "paired_error_change_median_m": 0.0
      }
    ]
  }
]
```

主线链路是否推进、完整任务净增、核心机制净增要分别判断。候选新增但实际未选不能算成功；法向贡献不能称方向贡献；普通多尺度表示的贡献不能称共享贡献。只有独立方向胜过法向对照、共享方向再胜过同方向独立，才有对应新机制的收益证据；即使满足，也仍需独立场景和完整作者强基线确认。

## 文献与创新边界

本轮决定：共享方向9/13/10/13低于旧V12的12/16/14/16；独立方向9/14/10/14未超过法向，rich/changed共享再丢资产15，所有背面仍0。**不保留V13为主算法，不扫方向温度/权重/训练轮数。**当前表示先删光背面近材料对，会阻断对共享对应收益的可信检验，修正它应优先于扩大本版实验。

下一拟实现`material_continuous_surface_correspondence_v14.py`（尚未实现）：保留全部512真实中心，不以大邻域纯平面资格硬删中心；目标使用实际测量支持的连续图像/表面坐标，显式估计二维中心偏移δ与SE(3)，法向不可辨时保留不确定性。必要对照为同512输入的普通稠密几何纹理/连续配准、独立连续对应与共享对应；所有新位姿候选共用。先确认33对近材料支持没有在入口消失，再全24×4检验背面与资产8实际选解；保留支持、加入δ本身归必要修复，不能自动算原创。

数学、已有方法和可证伪目标见[模型契约](material_surface_correspondence_model_v13_2026-10-27.md)。[PointMBF](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_PointMBF_A_Multi-scale_Bidirectional_Fusion_Network_for_Unsupervised_RGB-D_Point_ICCV_2023_paper.html)已有多尺度几何纹理融合；[RoReg](https://hpwang-whu.github.io/RoReg/)已有有向描述与局部旋转配准；[CoFiNet](https://proceedings.neurips.cc/paper/2021/file/c85b2ea9a678e74fdc8bafe5d0707c31-Paper.pdf)已有粗细对应与传输。不能把局部frame、环相关、Hungarian、GN或系统拼装本身称为新理论。本轮待检验只有实际曝光/方向多解与共享材料对应是否有同表示净增。

本轮没有训练网络，也未完整运行以上作者系统；机制控制不是现代完整强方法排名。所有新方法读取相同新chart，较V12查询像素约12.8倍（仍来自相同已采集RGB-D），跨版本收益可能是表示/信息利用的工程增量。一个已见TRAIN、四相关外观，无独立确认。

## 核对与执行产物

- 43334评分/状态/位姿/旧端点核对，5004条真实优化路径，18432实际中心来源；选中目标最大重建误差0。
- 原型10项＋周期修复7项/1001点检查；PNG人工打开检查，PDF导出；最终来源/旧证据完整性另见completion_integrity.json。
- 原评分入口有一处未配对括号，解析时尚未读标注；独立v13c评分只修语法，预测/候选/源封存不变，元数据在scoring_protocol.json。以后冻结前编译所有实际入口，不能只测内核。
- 各外观统计：`{"flat": {"seconds": 1291.0899769999087, "counters": {"pair_families": 481, "ordinary_pose_trials": 81718, "frame_pose_trials": 127108, "continuous_fits": 2064, "continuous_improved": 1839, "new_pair_endpoints": 2886}, "triples": 286, "states": 202}, "rich": {"seconds": 808.6229338000994, "counters": {"pair_families": 250, "ordinary_pose_trials": 43079, "frame_pose_trials": 64672, "continuous_fits": 1020, "continuous_improved": 936, "new_pair_endpoints": 1500}, "triples": 137, "states": 197}, "repeated": {"seconds": 859.8080936002079, "counters": {"pair_families": 234, "ordinary_pose_trials": 40515, "frame_pose_trials": 60504, "continuous_fits": 1108, "continuous_improved": 990, "new_pair_endpoints": 1404}, "triples": 147, "states": 200}, "changed": {"seconds": 653.1989013000857, "counters": {"pair_families": 202, "ordinary_pose_trials": 34605, "frame_pose_trials": 52270, "continuous_fits": 812, "continuous_improved": 748, "new_pair_endpoints": 1212}, "triples": 111, "states": 196}}`。
- [逐任务](../results/material_surface_correspondence_v13_run2/task_metrics.csv)、[实际材料链接](../results/material_surface_correspondence_v13_run2/material_links.csv)、[候选oracle](../results/material_surface_correspondence_v13_run2/oracle_by_stage.csv)、[节点覆盖](../results/material_surface_correspondence_v13_run2/node_material_coverage.csv)、[方向与中心偏移](../results/material_surface_correspondence_v13_run2/phase_vs_center_offset.csv)。
