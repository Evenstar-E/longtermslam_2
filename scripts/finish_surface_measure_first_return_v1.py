"""Merge sealed shards without copying large caches, audit, score, report."""
from pathlib import Path
import argparse,datetime,time
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def report(root):
    summary=read(root/'summary.json');pairs=read(root/'paired_outcomes.json');panel=read(root/'H3_recovery_panel.json');audit=read(root/'audit.json');cfg=read(root/'config.json')
    get=lambda f,m,h:next(r for r in summary if (r['family'],r['method'],r['stage'],r['history_views'])==(f,m,'final',h))
    rows=[]
    for f,m,label in [('surface',m,m) for m in cfg['methods']]+[('matched_voxel','shared','同曝光/同总先验体格共享'),('voxel','gamma_joint','旧体格共享'),('catalogue_plane','typed_ray','旧强 typed')]:
        a,b,c=[get(f,m,h) for h in [None,1,3]];rows.append(f"| {label} | {a['success']}/192 | {b['success']}/96 | {c['success']}/96 | {a['both_ids']}/192 |")
    paired=[]
    for p in pairs:
        if p['history_views'] is not None:continue
        paired.append(f"| {p['control_family']}/{p['control']} | +{p['gained']}/−{p['lost']} | {p['common_correct_ID_tasks']} | +{p['common_ID_quant_gained']}/−{p['common_ID_quant_lost']} |")
    shared=get('surface','surface_shared',None);ind=get('surface','surface_independent',None);H3=get('surface','surface_shared',3);H1=get('surface','surface_shared',1)
    recovered=sum(p['recovered_strong_only'] for p in panel);lostpanel=sum(p['old_strong_only'] for p in panel);newloss=sum(p['new_loss_vs_old'] for p in panel)
    models=audit['models'];empty=sum(p['surfaces']==0 for p in models)
    lines=['# 连续材料表面＋未见支持：全192结果', '', '## Material Passport', '', '- Mode: run; status: complete, all predictions sealed before evaluator access.', '- Scope: eight familiar DEV conditions, 192 asset tasks; not 192 independent scenes, not real-data validation.', '- Primary: final continuous poses; same 288 candidates and 24 identity slots; four same-representation methods.', '', '## 实现了什么', '', '连续 Gaussian 薄表面和非零局部未知体格在同一首回波场内竞争，一个正点只分配一次来源；实现解析射线曝光、全 SE(3) 几何自动微分和实际体格事件重算。H 来源用标准变分更新，当前使用 H 后验提出来源责任的条件变分评分。完成全192×4=768条连续拟合、完整封存、无真值重建审计及统一评分。H字典为经验贝叶斯且固定；没有实现连续隐面的出生与中心/法向联合优化。', '', '## 反映了什么问题', '', '| 方法 | 全部成功 | H1 | H3 | 两期身份正确 |', '|---|---:|---:|---:|---:|']+rows+['', f'H3旧差距为74−52=22个净成功数。本轮 H3 为{H3["success"]}：相对旧共享净变化{H3["success"]-52:+d}，相对旧强typed净变化{H3["success"]-74:+d}。旧强typed成功而旧共享失败的面板有{lostpanel}项，其中本轮追回{recovered}项；同时原旧共享成功的H3任务新增失败{newloss}项。不能把净22当成固定22个失败样本。', '', '| 相对控制 | 全任务新增/损失 | 共同正确身份数 | 交集内量化新增/损失 |','|---|---:|---:|---:|']+paired+['', '## 分析得到了什么原因', '', f'共同输入和表示下，共享相对独立净变化为{shared["success"]-ind["success"]:+d}。这种比较支持或反驳共享非可分评分的增量；新旧表示比较还同时改变了 H 表面假设和背景曝光近似，不能单独归因于连续表示。对逐项错误的因果诊断尚需另外的已封存后干预，不能仅凭总分声称所有错误来自某个机制。', '', f'字典统计：192个资产条件中{empty}个无可提取面元，其余使用实际历史局部法向支持的面元；这些数量不是物体真实面数。全部正点及前景射线保留，背景4×4求积；两个固定探针相对全像素面元曝光误差约2.10%/1.93%，不是全数据误差上界。深度仍用实测中心距离，未积分2mm噪声。', '', '## 是否推进主线', '', '主线链路：是，仍是24资产两期身份与固定材料锚点位移任务，未改成单物体评分演示。', f'算法收益：全192共享{shared["success"]}，独立{ind["success"]}，旧共享93、旧强120；H1={H1["success"]}、H3={H3["success"]}。是否增加必须按上述实际差值解释，不能以352项数值正确性检查替代任务收益。', '创新性：尚不能确认高创新。Gamma积分、来源变分、Gaussian表面和自动微分均属既有工具；本轮是具体统一观测模型的可运行候选。若同表示独立或正点控制复现收益，则不能把结果归为新的共享机制；本轮未做新公开真实数据和作者强方法复现。', '', '## 下一步怎么执行', '', '在已封存预测上分别核对：真身份交集的连续量化、H3旧强面板的追回/新损失、正确身份下的位姿排序。若收益来自背景求积/字典容量变化或固定来源近似而不是共享观测机制，先修正这些混杂；若已证明同表示共享具有稳定量化增量，再冻结到目录外连续运动和公开跨期数据。下一版本不得在本轮分数上扫薄层宽度与先验以制造22项追回。', '', '## 可追溯产物', '', f'- 无真值审计：{audit["checks"]}项，768条初始/最终轨迹从原始RGB-D重建；最大代价误差{audit["max_raw_cost_error"]:.3g}。', '- 运行器另外逐标量核对全部442,368个化简后的身份对因子；审计不独立重新遍历每个被舍弃候选对。', '- 根 prediction_seal.json / manifest.json / audit.json / scores.json / summary.json / paired_outcomes.json / H3_recovery_panel.json / evaluation_sources.json。大载荷只留在各分片，不复制到根目录。', '- 数学契约：surface_measure_first_return_contract_v1_2026-10-25.md。352项积分与梯度核对只验证数值正确性。']
    diagnosis=read(root/'postseal_ranking_diagnostic.json')
    matched=get('matched_voxel','shared',None);matched3=get('matched_voxel','shared',3)
    lines += ['', '## 同输入纯体格对照', '', f'额外192条纯体格拟合在主实验读取真值前独立完成封存和原始数据审计：总成功{matched["success"]}/192，H3={matched3["success"]}/96。新共享相对该对照净变化{shared["success"]-matched["success"]:+d}，H3净变化{H3["success"]-matched3["success"]:+d}。先验总质量一致，面元预留质量在无面元对照中分回局部未知格；暴露输入、候选池、开始集合和6轮优化预算相同。这比直接拿旧93做表示消融更能排除背景求积的影响，但仍不能把表面字典、足迹形状和来源近似各自拆成独立原因。', '对照路径：results/surface_measure_matched_voxel_v1；不复制其大载荷。']
    lines += ['', '## 全192正确身份条件下的排序诊断（不是算法成功率）', '', '| 方法 | 正确身份下候选量化通过 | 错误排序 | 其中严格低于保留真位姿代价 | 实际身份错误 | 实际身份正确但量化失败 |', '|---|---:|---:|---:|---:|---:|']
    for r in diagnosis['summary']:
        if r['history_views'] is None:
            lines.append(f"| {r['method']} | {r['correct_ID_pool_quant_ok']}/192 | {r['correct_ID_pool_wrong']} | {r['strict_wrong_ranking']} | {r['actual_wrong_ID']} | {r['actual_correct_ID_quant_fail']} |")
    lines += ['', '所有真位姿候选均保留。若正确身份下的池内最优仍严格偏好错误位姿，这部分失败不能只用候选缺失或有限身份分配未优化好解释；但此诊断不能单独证明错误由Gaussian足迹、背景求积、先验或来源变分哪一项造成。', '', '历史面元来源责任（无真值统计）：']
    for h in [1,3]:
        mm=[m for m in models if m['history_views']==h];fraction=sum(m['history_surface_hit_fraction'] for m in mm)/len(mm);foot=[m['mean_footprint_m'] for m in mm if m['mean_footprint_m'] is not None]
        lines.append(f'- H{h}：平均历史正回波分配给表面核的比例{fraction:.2%}；平均足迹尺度{sum(foot)/len(foot):.4f}米。来源责任不是几何真实性标签。')
    vb=read('research/surface_measure_variational_bound_checks_v1.json')
    lines += ['', f'另外180项小因子检查验证当前变分项不超过精确来源求和；冻结来源提案的最大gap为{vb["frozen_proposal_ELBO_gap_max"]:.3g}，联合坐标变分最大gap为{vb["joint_coordinate_ELBO_gap_max"]:.3g}。这些是不同小因子上的近似误差，不是全192错误的因果归因；说明不能将原型319项精确公式性质直接宣传为全模型精确证据。']
    lines += ['', '## 本轮明确决策与追加干预', '', '**不将当前径向Gaussian表面版本作为论文主攻模型。**共享比独立有收益，但新增表面相对匹配体格仅净增1、共同身份量化净损3，H3没有追回22项净差距。保留本版与完整链路作为失败对照，旧93和120仍是外部参照。', '', 'H3相对旧强方法的净差距按计数分解为：少15项正确身份，并多7项身份正确后的量化失败；不是逐项因果分配。', '', '固定核、观测、先验、候选位姿，只改变H+C1+C2来源更新：40轮时0/40错误排序翻转；对全部40例做1000轮上限收敛复核，所选较优解最多145轮、最后ELBO改善<1e-7，仍0/40翻转。最小错误代价优势9.14；本次普通来源迭代未收敛不是充分解释，但不声称已排除精确积分/全局变分的所有误差。', '', '单径向表面核的真实192字典对称性分析与两期锚点范围：surface_kernel_stabilizer_diagnostic_v1_2026-10-26.md。完整混合场仍由未知体格打破该对称，H3强误差证书仅2项，不能拿它解释全部22项净差距。', '', '完整决策：[surface_measure_decision_v1_2026-10-26.md](surface_measure_decision_v1_2026-10-26.md)。下一候选数学与实现契约：[continuous_material_support_measure_next_v2_2026-10-26.md](continuous_material_support_measure_next_v2_2026-10-26.md)，明确标注尚未实现。']
    text='\n'.join(lines)+'\n'
    start=text.index('## 下一步怎么执行');end=text.index('## 可追溯产物',start)
    text=text[:start]+'## 下一步怎么执行\n\n下一代码模块为 inspection_decisions/continuous_material_support_measure_v2.py（尚未实现）：用物理面积加权、局部连续支持核保留轮廓、孔洞与断开区域，并保持未见先验非零。来源推断需有可核验的折叠预测/SMC正常数，不能仅换成Gibbs后用平均log似然冒充证据。共同输入包含全部24资产/288候选；同时比较同推断的普通各向异性Gaussian、同局部表示的独立/共享及普通表面控制。单换椭圆核或更多VB轮次不作为论文创新。详见后附决策与v2数学契约；本轮不再扫径向核宽度/先验或扩充同类种子。\n\n'+text[end:]
    target=Path('research/surface_measure_first_return_results_v1_2026-10-25.md');target.write_text(text,encoding='utf-8')
    return target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal-only',action='store_true');args=parser.parse_args();root=Path('results/surface_measure_first_return_v1');shards=[root/f'shard_{i}' for i in range(3)]
    controlroot=Path('results/surface_measure_matched_voxel_v1');announced=False
    while not all((controlroot/f'shard_{i}/prediction_seal.json').exists() for i in range(3)):
        if not announced:print('WAIT_MATCHED_VOXEL_SEALS_BEFORE_ANY_GT',flush=True);announced=True
        time.sleep(5)
    if not (controlroot/'audit.json').exists():
        from scripts.finish_surface_measure_matched_voxel_v1 import main as finish_control
        finish_control()
    assert read(controlroot/'audit.json')['passed']
    if not (root/'prediction_seal.json').exists():
        runs=[];sources={};inputs={}
        for shard in shards:
            seal=read(shard/'prediction_seal.json');assert seal['complete'] and not seal['GT_read']
            for p,h in seal['sha256'].items():assert sha(shard/p)==h
            manifest=read(shard/'manifest.json');sources.update(manifest['sources']);inputs.update(manifest['inputs']);inputs[str(shard/'prediction_seal.json')]=sha(shard/'prediction_seal.json')
            for run in read(shard/'runs.json'):runs.append({**run,'storage':str((shard/run['scene']).relative_to(root))})
        order={run['scene']:i for i,run in enumerate(read('results/first_return_gamma_material_v5/runs.json'))};runs.sort(key=lambda r:order[r['scene']]);assert len(runs)==8 and len({r['scene'] for r in runs})==8
        inputs[str(controlroot/'prediction_seal.json')]=sha(controlroot/'prediction_seal.json')
        inputs[str(controlroot/'audit.json')]=sha(controlroot/'audit.json')
        inputs['research/surface_measure_matched_voxel_control_v1_2026-10-25.md']=sha('research/surface_measure_matched_voxel_control_v1_2026-10-25.md')
        cfg=read(shards[0]/'config.json');save(root/'config.json',cfg);save(root/'runs.json',runs);save(root/'manifest.json',dict(sources=sources,inputs=inputs,motion_limits=read(shards[0]/'manifest.json')['motion_limits'],GT_read=False,large_payloads_not_duplicated=True))
        for field in ['sources','inputs']:
            for p,h in read(root/'manifest.json')[field].items():assert sha(p)==h
        supervision_path=root/'execution_supervision.json'
        if supervision_path.exists():
            supervision=read(supervision_path);supervision.update(status='all_inference_finished_before_GT',inference_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(supervision_path,supervision)
        save(root/'prediction_seal.json',dict(complete=True,GT_read=False,scenes=8,assets=192,continuous_fits=768,sha256={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}));print('ALL192_SEALED_BEFORE_GT',flush=True)
    if args.seal_only:return
    if not (root/'audit.json').exists():
        from scripts.audit_surface_measure_first_return_v1 import main as audit
        audit()
    assert read(root/'audit.json')['passed']
    if not (root/'evaluation_sources.json').exists():
        from scripts.score_surface_measure_first_return_v1 import main as score
        score()
    if not (root/'postseal_ranking_diagnostic.json').exists():
        from scripts.diagnose_surface_measure_first_return_v1 import main as diagnose
        diagnose()
    result=report(root)
    paths=[result,Path(__file__),Path('scripts/audit_surface_measure_first_return_v1.py'),Path('scripts/score_surface_measure_first_return_v1.py'),Path('scripts/diagnose_surface_measure_first_return_v1.py')]
    paths += [root/n for n in ['prediction_seal.json','manifest.json','audit.json','scores.json','summary.json','paired_outcomes.json','H3_recovery_panel.json','evaluation_sources.json']]
    paths += [Path('research/surface_measure_first_return_contract_v1_2026-10-25.md'),Path('research/surface_measure_first_return_checks_v1.json'),Path('research/surface_measure_first_return_actual_probe_v1.json')]
    paths += [root/'postseal_ranking_diagnostic.json',Path('research/surface_measure_variational_bound_checks_v1.json'),Path('scripts/check_surface_measure_variational_bound_v1.py')]
    paths += [controlroot/'prediction_seal.json',controlroot/'manifest.json',controlroot/'audit.json',Path('scripts/run_surface_measure_matched_voxel_v1.py'),Path('scripts/finish_surface_measure_matched_voxel_v1.py'),Path('research/surface_measure_matched_voxel_control_v1_2026-10-25.md')]
    paths += [root/'postseal_source_update_intervention.json',root/'postseal_source_update_intervention_converged_v2.json',Path('scripts/intervene_surface_source_update_v1.py'),Path('scripts/intervene_surface_source_converged_v2.py'),Path('scripts/diagnose_surface_kernel_stabilizer_v1.py'),Path('research/surface_kernel_stabilizer_diagnostic_v1_2026-10-26.json'),Path('research/surface_kernel_stabilizer_diagnostic_v1_2026-10-26.md'),Path('research/surface_measure_decision_v1_2026-10-26.md'),Path('research/continuous_material_support_measure_next_v2_2026-10-26.md')]
    save(root/'evidence_index.json',dict(complete=True,sha256={str(p):sha(p) for p in paths},mutable_handoff_excluded=True));print('COMPLETE',result,flush=True)


if __name__=='__main__':main()
