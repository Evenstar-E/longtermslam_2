"""Facts from audited pipeline, not a trained-policy performance claim."""
import json
from pathlib import Path


def main():
    base=Path('results/evidence_graph_dataset_v1');a=json.loads((base/'analysis.json').read_text());c=a['counts'];r=a['realized_interactions'];g=a['graph_statistics'];n=a['shared_nuisance_structure']
    identity=json.loads((base/'identity_diagnostic.json').read_text())
    lines=['# 证据图与实际估计器标签采集v1：完成报告','',
        '## Material Passport','',
        '- 完成日期：2026-10-23（契约及采集启动于2026-10-22，跨日完成）。',
        '- Verification Status: VERIFIED（输入/标签、数据来源、费用和完整场景重放），不表示算法创新或收益已经成立。',
        '- 契约：`evidence_graph_contract_v1_2026-10-22.md`；新代码：`inspection_decisions/evidence_graph.py`和`graph_dataset.py`；冻结配置：`configs/evidence_graph_dataset_v1.json`。','',
        '## 已完成什么','',
        f"已实现并实际运行新的证据图与真值分离标签收集器。生成{c['families']}个新程序化场景家族、{c['fresh_payload_files']}帧新RGB-D，执行{c['actual_estimator_fits']}次完整共同估计器调用；保存{c['states']}个当前状态图、{c['transitions']}个转移含STOP、{c['pairs']}组固定动作对标签。API载荷实际读取{c['actual_payload_reads']}次，其中包含各分支重复的H0/C0和相同视图读取，不能算成新独立图像。",'',
        '每个家族包含初始状态、全部22个单步终点和8个预定两步终点；后端仍为冻结的几何多模式/GN/共同报告规则。没有训练新策略，也没有复活旧v8或生成原256确认集。','',
        '## 图和标签如何分开','',
        '构图纯函数只接收已取得RGB-D、当前拟合、名义相机和资源。输入图先写盘，随后评价器才生成真实任务损失/归属评价/实际费用。场景seed、划分和真实评价保留在索引/标签端，不能作为策略数值特征；未来图像、未来真实可见性、真实位移和所有权均不进入图。',
        '',
        '输入包含匿名构件、已见历史/当前表面、标定变量和合法动作；边包含跨期支持、雅可比、真正共用源变量的协方差、完整联合拟合的误差交叉项、候选动作与已见表面的关系。未来未观察误差节点使用明确的I/3先验；未来载荷字节仅用已读帧大小估计。',
        '',
        f"平均每图{g['mean_nodes']:.1f}个节点、{g['mean_edges']:.1f}条有向边；图输入压缩后总计{g['total_compressed_bytes']/1e6:.2f}MB。这不是完整未知形状/遮挡预测，也不是经过校准的物理后验。",'',
        '![实际新场景的聚合证据图](../results/evidence_graph_dataset_v1/graph_example.png)','',
        '## 表示确实表达了什么','',
        '新场景0取H1后，两个历史帧间128条有向片段对的“共用源变量协方差”均为0，但平均联合交叉协方差范数约4.41e-4平方米；新场景4的共享误差条件下，相应平均范数约5.12e-4和5.17e-4平方米。先验独立与后验相关已经分开表示，避免把所有相关性归为同一个概念。',
        '',
        f"全数据审计中，有{n['states_with_cross_frame_pairs']}个状态含同一时期的跨帧片段对：其中{n['coherent_nonzero_states']}个共享误差状态具有非零共用源变量边，{n['independent_zero_states']}个独立帧状态的该项为0。这个结果验证表示/来源关系，不能据此宣称它改善了动作选择。",'',
        '## 场景划分与交互标签','',
        '训练、校准、开发三侧各16个新家族，每侧覆盖全部16个条件单元。同一家族的所有视图、状态及分支始终同侧，seed与旧48开发场景无交集。它们仍来自同一简化生成器；这里的校准划分名称不意味着已经建立风险保证。','',
        '|管线划分|家族|状态|固定两步对|正交互|零交互|负交互|','|---|---:|---:|---:|---:|---:|---:|']
    for name,s in a['splits'].items():lines.append(f"|{name}|{s['families']}|{s['states']}|{s['pairs']}|{s['positive']}|{s['zero']}|{s['negative']}|")
    lines += ['',
        f"交互标签采用L(a)+L(b)-L(ab)-L(initial)，总计正{r['positive']}、零{r['zero']}、负{r['negative']}，范围[{r['range'][0]:.3f},{r['range'][1]:.3f}]。每个分支只有一个观测/噪声实现，不能把它当精确期望、因果互补性或最优Q值；同一家族的8对高度相关，不按384个独立场景推断。未按收益挑选动作对或排除失败场景。",'',
        '![全部两步对标签符号](../results/evidence_graph_dataset_v1/pair_labels.png)','',
        f"全部状态中零实际损失{a['label_census']['zero_loss_states']}个，身份错误参考计数{a['label_census']['identity_wrong_references_across_all_states']}（跨重复状态合计）。它们是标签分布审查，不是某个新策略的成功率；不能挑一条事后最好路径来报算法收益。",'',
        f"事后定位：错误归属涉及{identity['affected_states']}个状态，集中在新场景{identity['affected_families']}两个家族，不能写成46个独立错误场景。它们全部保留。图中的同参考构件/跨期支持关系依赖当前后端的估计对应，不是真实身份保证，故关联错误也可能污染关系边；当前图尚未做多身份假设或纠错。后续各模型共享该后端，不在本轮另开一串后端修复来掩盖问题。",'',
        '## 检查与问题分类','',
        '- 4项测试通过：禁止构图文件读取/污染未读图像索引；共享与独立源边；未来变量先验；收集器输入先持久化而真值只能经分离评价器进入标签。',
        f"- {a['checks']}项完整性/图数值/合法动作/来源/划分/费用/标签与原后端评价核查通过；新场景0全部31个状态、输入图原始摘要、报告和标签完整重放通过。旧v8及停止审计源/输入哈希保持一致，零排除。", 
        '- 开发期单测发现同一时期片段不存在`missing_自身时期`字段：其支持已由该片段提供，修正为该特征0；修正在运行冻结前完成。这是接口语义工程修复，没有当成科研发现。',
        '- 费用包含立即返航，真实资源增量只在标签端；声明式移动/传感/传输时间不等于测得的UAV全任务时间。实际后端调用CPU已单独记录，分支复用的中间输入不重复计拟合。',
        '- 源误差关系来自条件GN/多模式协方差，已见面视锥只是代理。独立帧可存在联合后验交叉项；未知面/遮挡/身份歧义和任意构件SE(3)尚未解决。','',
        '## 主线与论文收益审查','',
        '- 主线推进：新候选的“共享误差证据表示＋实际估计器动作后损失”已从方案变为可加载输入、真实分支和标签，包含单步及预定组合，而不再只使用旧像素似然/解释分离度。',
        '- 工程部分：序列化、分离输入和标签、哈希/重放、图索引和资源核算；这些不作为原创贡献。',
        '- 研究信息：表示能区分共用源误差与联合后验相关；实际组合标签有正负变化。是否可预测、是否比平坦模型/普通两步规划有用，本轮没有回答。',
        '- 论文收益：尚无新算法收益或新颖性确证。不能把1488次估计器调用、非零图边、正交互标签数或新seed当创新证明。',
        '- 下一项：先做同输入的普通平坦价值/风险停止与结构价值模型训练比较，按家族训练/校准/开发隔离；验证动作价值排序与实际报告收益，再决定是否扩数据。16训练家族仅适合有边界的初步比较，不能支撑大网络泛化结论。普通视图图Q与匹配两步规划仍须保留。',
        '- 环境：当前虚拟环境未安装PyTorch；本轮是NumPy/SciPy后端和数据采集，没有GPU训练。可先执行CPU小模型/线性对照；不得把尚未安装的技术栈写成完成。','',
        '## 可复现入口','',
        '- `python -m unittest tests.test_evidence_graph_v1 -v`。',
        '- `python -m scripts.collect_evidence_graph_v1 --shard 0 --shards 2`及shard1，拒绝覆盖；源/契约/配置运行前冻结。',
        '- `python -m scripts.analyze_evidence_graph_v1`；分析包括预定新场景0完整重放。',
        '- `python -m scripts.plot_evidence_graph_v1`；示例图和全部标签符号图为数据审查，非性能图。',
        f"- 两采集分片耗时{a['seconds'][0]:.1f}/{a['seconds'][1]:.1f}秒，曾并发，不作算法速度优越性结论。",'',
        '全部结果：`results/evidence_graph_dataset_v1/`；完整输入/标签契约见前述文件。','']
    Path('research/evidence_graph_dataset_v1_2026-10-23.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
