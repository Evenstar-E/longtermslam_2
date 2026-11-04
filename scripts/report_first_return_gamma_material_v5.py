"""Build the Chinese result report from sealed, audited, scored artifacts."""
from pathlib import Path
import numpy as np
from scripts.run_rigid_identity_bounds_v1 import read,sha
from scripts.run_sparse_ray_material_incidence_v3 import save


def main():
    root=Path('results/first_return_gamma_material_v5');summary=read(root/'summary.json');paired=read(root/'paired_outcomes.json');audit=read(root/'audit.json');seal=read(root/'prediction_seal.json');scores=read(root/'scores.json');mechanism=read(root/'mechanism_summary.json')['rows'];cfg=read(root/'config.json');assert audit['passed'] and seal['assets']==192
    lines=['# 首回波未知材料模型：完整192项结果','', '## Material Passport','', '状态：全部8组预测封存、无GT审计、正式评分和机制诊断完成。熟悉SIM DEV、24资产×8条件，H1/H3及相关几何变体并非192个独立场景。无新实机或真实跨期数据，无已发表作者系统的完整复现。主线仍为多资产物理身份＋同一历史材料锚点位移≤5cm。','', '## 本轮实际实现','', '对象自身材料坐标中使用非负分段体交互率。每条原始像素射线只出现一次；自身回波贡献命中，其他对象/背景只贡献到实际回波为止的曝光长度。H未见单元保留Gamma先验，不从支持字典删除。全量RGB-D沿用匿名槽位前端，任何候选均有有限颜色代价。','', r'$M_H(n,L)=\sum_v[\log\Gamma(a_v^H+n_v)-\log\Gamma(a_v^H)+a_v^H\log b_v^H-(a_v^H+n_v)\log(b_v^H+L_v)]$。共享为 $-M_H(n_1+n_2,L_1+L_2)$，独立为两期边际之和；插件用历史后验均值。','', 'Gamma(.5,.08m)、Beta(1,1)、8cm格距；域从全部匿名实测点＋运动球给出包围证书，各方法共同使用。保留全部288候选，24³身份指派；连续阶段12维SE(3)一轮±坐标搜索、实际重算跨格事件、只接受球内下降。相同输入和候选，不代表相同FLOPs。','', '**本轮为中心深度近似**：2mm传感器噪声没有积分；Gamma消元仅对声明的统计精确。独立体率不是不透明表面的精确物理模型，8cm网格可能损害5cm位移量化；一次坐标扫也不是连续全局最优。','', '## 完整任务结果','', '| 方法 | 候选阶段成功 | 最终成功/192 | 两期身份均正确 | 正确身份中位误差/m |','|---|---:|---:|---:|---:|']
    labels={'gamma_joint':'Gamma共享积分','gamma_independent':'Gamma独立积分','gamma_mean':'Gamma均值插件','beta_joint':'Beta共享占据积分'}
    for m in cfg['methods']:
        r=summary['hazard'][m];v=r['final'];lines.append(f"| {labels[m]} | {r['candidate']['success']} | {v['success']} | {v['both_ids']} | {v['correct_ID_median_error_m']:.5f} |")
    for family,m,label in [('catalogue_plane','previous_typed_ray','此前较强typed系统'),('material_shared','previous_typed_ray','此前另一初始化typed'),('Gibbs','previous_joint_ray','此前有限H字典Gibbs')]:
        v=summary[family][m]['final'];lines.append(f"| {label} | — | {v['success']} | {v['both_ids']} | {v['correct_ID_median_error_m']:.5f} |")
    if (root/'task_outcome_strata.json').exists():
        strata=read(root/'task_outcome_strata.json');moving=[r for r in strata['groups'] if r['family']=='hazard' and r['true_displacement_gt_5cm']];lines+=['','真实材料点位移>5cm的176项：'+ '；'.join(f"{labels[r['method']]} {r['success']}/176" for r in moving)+'。共享相对独立新增15成功、丢失4，净11；其余16项净1。此分层为封存后诊断，不能当作176个独立场景。']
    lines+=['','旧系统与新模型原始像素数量、外观项、观测表示均不同，旧分数用于部署表现比较，不能单独归因于Gamma积分。Gamma三者是同统计积分消融；Beta采用路径单元空闲计数，物理假设也不同。Beta为自实现基线，非作者系统。','', '| 共享Gamma相对控制 | 新增成功 | 新增失败 | 净收益 | 共同正确身份中的量化净收益 |','|---|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['control_family']}/{r['control']} | {r['gained']} | {r['lost']} | {r['gained']-r['lost']} | {r['common_correct_ID_success_gained']-r['common_correct_ID_success_lost']} |")
    lines+=['','当前真实运动仍来自已知9项运动目录；offgrid表示物体几何偏移，并非目录外运动。全部方法看到共同扩展池和连续搜索，仍不能据本实验宣称未知连续运动泛化。','', '曝光路径严格在回波终止，但共享单元交互率的更新仍会影响整格，包括未实际走过的格内部分。这不是精细遮挡边界恢复。计算域也由当前匿名观测确定，因此这里只声称固定共域下的条件代价比较，不能称为无限未知空间的完整校准贝叶斯证据。','', '## 单历史与三历史条件','', '| 方法 | H1成功/96 | H3成功/96 |','|---|---:|---:|']
    for family,m,label in [('hazard',m,labels[m]) for m in cfg['methods']]+[('catalogue_plane','previous_typed_ray','此前较强typed'),('Gibbs','previous_joint_ray','此前Gibbs')]:
        values=[sum(r['success'] for r in scores if r['family']==family and r['method']==m and r['stage']=='final' and r['history_views']==h) for h in [1,3]];lines.append(f'| {label} | {values[0]} | {values[1]} |')
    lines+=['','## 机制诊断与因果边界','', '正共享修正并不必然表示找到同一材料表面。两期均无命中的单元也有精确修正：', '', r'$\Delta_v^{free}=a_v\log\frac{(b_v+L_{1v})(b_v+L_{2v})}{b_v(b_v+L_{1v}+L_{2v})}\ge0$。','', '以下为实际选中候选的全任务分解；真身份/位姿行仅作预测封存后的干预，绝不计入算法成功。各项符号可相反，因此不把比例强行解释为百分比归因。','', '| H视角 | 模式 | 共享修正总量 | 无当前命中单元 | 含当前命中单元 | C2命中且H无正命中比例 | C2命中且H完全未经过比例 |','|---:|---|---:|---:|---:|---:|---:|']
    for r in mechanism:
        if r['method'] not in ['gamma_joint','true_ID_pose_intervention']:continue
        n=r['C2_hits'];lines.append(f"| {r['history_views']} | {r['method']} | {r['cross_log_evidence']:.2f} | {r['no_current_hit_cell_cross']:.2f} | {r['current_hit_cell_cross']:.2f} | {r['C2_hits_H_no_positive']/n:.2%} | {r['C2_hits_H_unvisited']/n:.2%} |")
    lines+=['','固定正确身份后，在同一完整候选池内选取各模型最低代价位姿，检验代价排序。真实位姿在全部池中均保留；若模型严格偏好位移错误解，便不能把这类失败仅归为候选裁剪或求解器没找到更低代价。此为诊断，不能当作算法成功率。','', '| H视角 | 方法 | 正确身份下量化≤5cm/96 | 严格偏好错误量化的任务数 |','|---:|---|---:|---:|']
    for r in read(root/'mechanism_summary.json')['true_ID_pool_ranking']:lines.append(f"| {r['history_views']} | {r['method']} | {r['true_ID_pool_quantification_ok']} | {r['objective_strictly_prefers_wrong_quantification']} |")
    repairs=[]
    for run in read(root/'runs.json'):
        old=read(Path(cfg['geometry_root'])/run['scene']/'assignments.json');new=read(root/run['scene']/'assignments.json')
        for m in cfg['methods']:
            a=np.array(old[m]['slots']);b=np.array(new[m]['slots']);q=new[m]['log']['standard_MIP_repair'];repairs.append({'scene':run['scene'],'method':m,'triggered':q['triggered'],'accepted':q['accepted'],'slot_pairs_changed':int(np.any(a!=b,axis=0).sum()),'old_gap':q['old_gap'],'new_gap':new[m]['log']['gap'],'energy_decrease':old[m]['log']['energy']-new[m]['log']['energy']})
    save(root/'assignment_repair_summary.json',repairs)
    lines+=['','## 关联求解器质量','', '| 方法 | MIP触发/接受次数 | 改变身份对数 | 修复后最大数值间隙 |','|---|---:|---:|---:|']
    for m in cfg['methods']:
        rr=[r for r in repairs if r['method']==m];lines.append(f"| {m} | {sum(r['triggered'] for r in rr)}/{sum(r['accepted'] for r in rr)} | {sum(r['slot_pairs_changed'] for r in rr)} | {max(r['new_gap'] for r in rr):.3e} |")
    lines+=['','v5兼容旧日志字段LP_lower_bound，但它保存了LP与MIP的较强数值下界，并非都来自LP；不宣称连续位姿全局最优。']
    mainv=summary['hazard']['gamma_joint']['final']['success'];ind=summary['hazard']['gamma_independent']['final']['success'];beta=summary['hazard']['beta_joint']['final']['success'];mean=summary['hazard']['gamma_mean']['final']['success']
    lines+=['','## 本轮判断','',f'共享Gamma最终{mainv}，独立{ind}，插件{mean}，普通Beta{beta}，此前较强系统120。主线链路已经推进到未知先验的原始首回波观测及完整身份/连续位姿实验；是否有任务净收益以此表为准，不能用正确性检查或某个成功例子代替。', '', 'Gamma共轭、存活率、体素、Hungarian、函数查表均是既有工具。仅本轮熟悉DEV上的差异不足以确立高创新性；需同时证明同输入强控制无法复现、机制确实修复目标失败，并在目录外运动和真实跨期数据上保持。若插件或Beta同样有效，应归为表示/工程收益。', '', '先例边界见 [本轮先例核查](first_return_gamma_prior_art_update_2026-10-25.md)：Lambda-Field已有强度场，CMU 2020已有前向射线模型与联合占据不确定性。因此不声称“首次考虑首回波/未知空间/联合估计”。','', '## 验证与工程记录','',f"820项射线/公式检查、17,376项加速等价检查；封存后审计{audit['checks']}项，通过。重建全部选中槽位对因子及全部最终位姿目标，最大代价差{audit['max_factor_or_final_cost_error']:.3e}；H和抽取候选的原始统计差{audit['max_raw_statistic_error']:.3e}。v5连续拟合{audit['continuous_fits']}次，{audit['fits_with_descent']}次发生实际下降；处理原始射线{audit['unique_raw_rays_across_conditions']}条（跨相关条件重复采样，非独立样本量）。", '', '审计没有重新证明所有被丢弃候选的最小性；LP为数值界，未宣称全部指派全局最优。全部8组封存且审计通过后才读取本轮GT评分。', '', 'v1被域外命中检查中止；v2保存完整首3场景统计后为运行时间改用同公式加速；v3因Windows硬链接权限在推断前失败；v4改普通复制。失败目录与旧源保留，均未产生被选择性评分的预测；没有修改旧基线或按GT重调当前参数。', '', '## v5关联求解修复', '', '本轮GT评分前发现v4有明显整数解间隙，因此增加标准60秒MILP补求解，四方法同规则；全部768轨迹重新精修，v5全部预测再次封存后才评分。v4维持未评分档案，未在两版本间按GT择优。此项为工程公平性修复，不计创新。', '', '## 下一步实施边界','', '先根据上述共享修正分解、H1/H3与正确身份量化结果，决定改空间表面表示还是噪声处理。噪声版必须为同一原始像素设置一个共享真实距离，不能各对象独立挑深度；单元独立先验的网格尺度也必须明确。推导见 first_return_noise_and_grid_limits_2026-10-25.md，尚未实现。继续固定同输入Beta/插件对照，并增加目录外连续运动与公开真实数据和适用作者基线；不继续在当前DEV上调Gamma温度或用小例子重证问题必要性。','']
    lines+=['','## 明确的方向决定','', '保留已验证有收益的共享因子，当前独立体素表示仍不替换旧120系统。99项失败中90项身份错误、9项量化错误；下一步用连续材料表面＋未见支持的统一首回波因子改表示和关联。具体结论见 [方向决定](first_return_gamma_decision_2026-10-25.md)。','', '下一表面测度因子的来源枚举/分割展开、重复核质量守恒和曝光梯度已完成319项小型正确性检查，**尚未接入全192**。数学、复杂度和先例边界见 [下一模型](continuous_surface_unknown_material_next_v1_2026-10-25.md)；不得把该原型混入当前93/192成绩。','']
    report=Path('research/first_return_gamma_results_2026-10-25.md');report.write_text('\n'.join(lines),encoding='utf-8')
    files=[report,Path(__file__),Path('inspection_decisions/first_return_gamma_material_v1.py'),Path('inspection_decisions/first_return_gamma_fast_v1.py')]+[Path('scripts')/(s+'_first_return_gamma_material_v5.py') for s in ['run','audit','score','finish','diagnose']]+list(Path('research').glob('first_return_gamma_*2026-10-25.md'))+[Path('research/first_return_gamma_checks_v1.json'),Path('research/first_return_gamma_fast_checks_v1.json'),Path('research/first_return_gamma_cached_geometry_v2.json'),Path('research/first_return_noise_and_grid_limits_2026-10-25.md'),Path('research/public_real_data_gate_hope_2026-10-25.md')]+[root/p for p in ['prediction_seal.json','manifest.json','audit.json','summary.json','paired_outcomes.json','mechanism_summary.json','evaluation_sources.json','scores.json']]
    files += [Path('research/continuous_surface_unknown_material_next_v1_2026-10-25.md'),Path('inspection_decisions/surface_measure_evidence_prototype_v1.py'),Path('scripts/check_surface_measure_evidence_prototype_v1.py'),Path('research/surface_measure_evidence_prototype_checks_v1.json'),Path('scripts/analyze_first_return_gamma_task_outcomes_v1.py'),root/'task_outcome_strata.json',root/'assignment_repair_summary.json']
    save(root/'report_sources.json',{'script_sha256':sha(Path(__file__)),'report_sha256':sha(report)});files.append(root/'report_sources.json');save(root/'evidence_index.json',{'complete':True,'mutable_handoff_excluded':True,'sha256':{str(p):sha(p) for p in files}});print('REPORT_COMPLETE',report,flush=True)


if __name__=='__main__':main()
