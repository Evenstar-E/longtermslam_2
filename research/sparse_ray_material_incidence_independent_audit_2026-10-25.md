# 稀疏射线因子独立只读审计

沿用 academic-research-suite experiment-agent 的 code_runner 路由，现有 continuous_baseline_audit 子代理只读检查。未改文件、未读evaluation_only.pkl或GT任务/物理诊断、未重复整批推断。

## 数学与独立数值核对

- kappa=A/B、首回波截断上界及Jhigh、无序轴对交换、潜在z的端点/活跃集导数未发现直接代数错误。
- 120个随机profile问题（包括首回波截断），独立SciPy标量优化与端点比较，最大成本差7.67e-13。
- 60组普通凸四边形、退化线和点，独立凸组合SLSQP距离核对，最大距离差2.09e-7m，数值求解容差量级。
- 50组随机混合角色、多假设同ray、退化区域，逐对与并集缓存的无序/有序距离及可用性共200项通过，最大差4.34e-18m。

## 接口与语义

raw_pairs每视图ray_id会重置，runner实际用slot内累计长度重新编号，跨view无碰撞。

条件14参数joint_residual必须共享同一无序轴对；初查指出前提未assert，父任务已在所有完整推断前补入检查并新增单元。即使坐标相同也不保证空间断开的真实边是同一物理实例。已知轨迹成本下降不证明位移正确。

InvalidIncidence只否决一个条件面/终止方向解释能够提供域外见证，不能证明不是边或姿态错误。无解释/无历史参照时的封顶是重复支持先验，不是负观测似然。沿规范平移不改变删失状态，父任务在全部扫描中另外核验。

## 完整批次与工程等价

run4保持身份/R/C1冻结，epsilon0与17点共同候选、相同平局规则；参考按每个凸梯形保留后做真实union_all(grid_size=0)，不跨区域凸包，ordered与unordered组语义保持。manifest记录源/合同、实际preflight读的run3 raw及完成组pred哈希，开跑核验旧三根seal，未见GT输入。

指出两项非算法语义问题：reference_hypothesis_count现为缓存组数，父任务另存实际参考假设数；run3比较用zip未assert扫描长度，父任务最终无GT审计另核每组长度及epsilon序列，不能只靠zip声称完整等价。原推断输出不修改。

审计只支持数学/实现与实验隔离，不支持原创性、全SE3最优或完整系统性能结论。
