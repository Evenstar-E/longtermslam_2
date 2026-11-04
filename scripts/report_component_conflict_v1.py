from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_evidence_value_pilot_v1 import read,save
from inspection_decisions.graph_dataset import digest

def main():
    out=Path('results/component_conflict_v1');a=read(out/'analysis.json');cfg=read('configs/component_conflict_v1.json');methods=cfg['methods']
    names=['Component conflict','Component report only','Unpaired mean conflict','No mode conflict','No association/source','Global same-info pooled','Frozen one-step','Ordinary finite horizon','Original rank STOP','Declared risk','Full rank','Full coverage']
    fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True,layout='constrained');y=np.arange(len(methods))
    for ax,key,title in zip(axes,['task_loss','mean_acquisitions','mean_travel'],['Total task loss','Mean acquisitions','Mean return-inclusive travel (m)']):
        vals=[a['summaries']['all'][m][key] for m in methods];ax.barh(y,vals,color=['#0072B2']+['#777777']*11)
        ax.set_xlim(0,max(vals)*1.2+.01);ax.set_title(title)
        for i,v in enumerate(vals):ax.text(v+max(vals)*.02,i,f'{v:.2f}',va='center',fontsize=8)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();fig.suptitle('32 fresh families /48 changes /208 unchanged assets\n64 actual acquisition paths;384 shared-prefix STOP outcomes')
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.svg');plt.close(fig)
    labels=dict(zip(methods,names));rows=a['rows'];by={(r['scene'],r['method']):r for r in rows};records=[]
    for sid in range(32):
        records.append(dict(scene=sid,methods={m:dict(stage=by[sid,m]['stage'],loss=by[sid,m]['metric']['task_loss'],qualified=by[sid,m]['metric']['qualified_tp'],fp=by[sid,m]['metric']['fp'],bad=by[sid,m]['metric']['bad_motion_reports']) for m in methods},candidate_decisions=by[sid,'component_conflict']['decisions']))
    save(out/'family_comparisons.json',records)
    lines=['# 构件级几何／匹配冲突表示实验','', '2026-10-23，完成。模式权重为能量启发值，四个几何解共用硬匹配，不是真实身份后验。','',
    '## 算法与数学模型','',
    '每个H0参考构件保留14维局部报告/时序量和12维冲突量：8维几何模式分歧/传感与校准方差，4维匹配竞争、跨视角质心散布、最新视角偏差和来源多样性。61维全局报告上下文仍提供给所有读出。构件候选87维，报告对照75维，全局同信息均值/最大值/标准差汇总139维。', '',
    '共享岭回归正则固定10，预测当前构件损失和下一次观测后的构件损失。R=sum_j max(0,f(x_j))/8，C=0.05/8+sum_j max(0,g_t(x_j))/8。当前R<=C且R<=校准阈值时停止，预算5强制停止。没有新Bellman结构或变更后端。构件目标按状态等权，缺失所有者的损失等分到报告构件，仅为保持任务损失精确可加的记账约定。', '',
    '对照区分构件级监督与额外信息：component_report同监督去掉冲突；component_unpaired以平均冲突代替局部冲突，同时会丢失分布，不能把差异纯归因为绑定；global_augmented保留相同局部信息的全局统计。模式和匹配/来源消融分别清零相应通道。', '',
    '## 数据与隔离','',
    '原16训练/16校准家族重新执行64条完整路径，384拟合，报告逐状态与旧路径完全一致，仅保存更多已观测信息。标签在整条actor完成后计算。冻结后32新种子22402026–22402057，16原分布/16实际遮挡物变化，各覆盖16因素组合。新增评价64路径384拟合，12规则共享前缀形成384终点，独立评价单位仍为32家族。前几轮新评价均未用于选模型/阈值。', '',
    f"实际验证另用{a['validation_fits']}次拟合，场景{a['physical_scene']}，本轮总新增{a['total_new_fits']}次。4项测试与{a['audit_checks']}项主核查通过，无排除。构件级样本不是独立家族。", '',
    '## 全方法新场景结果','', '| 方法 | 合格/48 | 误报 | 错误位移 | 未知 | 损失 | 平均采集 | 返航路程m |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        v=a['summaries']['all'][m];lines.append(f"| {labels[m]} | {v['qualified_tp']} | {v['fp']} | {v['bad_motion_reports']} | {v['unknown']} | {v['task_loss']:.2f} | {v['mean_acquisitions']:.4f} | {v['mean_travel']:.3f} |")
    lines+=['','![全部比较](../results/component_conflict_v1/comparison.png)','','## 配对区间','', '5000次场景家族配对bootstrap，全32按原分布/遮挡变化分层，探索性未多重校正95%。候选减对照。','', '| 对照 | 平均损失差[区间] | 平均采集差[区间] | 综合目标差[区间] |','|---|---|---|---|']
    for m in methods[1:]:
        d=a['contrasts']['all'][m]
        def fmt(k):v=d[k];return f"{v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
        lines.append(f"| {labels[m]} | {fmt('loss')} | {fmt('acquisitions')} | {fmt('objective')} |")
    lines+=['','## 分条件结果','', '| 条件 | 方法 | 合格/24 | 误报 | 错误位移 | 损失 | 采集 |','|---|---|---:|---:|---:|---:|---:|']
    for cond in ['control','shift']:
        for m,v in a['summaries'][cond].items():lines.append(f"| {cond} | {labels[m]} | {v['qualified_tp']} | {v['fp']} | {v['bad_motion_reports']} | {v['task_loss']:.2f} | {v['mean_acquisitions']:.4f} |")
    lines+=['','## 主线审查','', '本轮实际改变的是估计证据的状态表示及构件级风险读出，主动动作排序和报告后端不变。工程核查通过不能作为论文收益；额外模式信息、多解释、共享回归本身不声明创新。是否有增量必须同时参考报告监督对照、同信息汇总和普通单步强对照。','',
    '简化调色板RGB-D、平移变化、有限家族和启发式模式权重限制了结论范围。全局与构件对照同时改变监督粒度，须结合component_report隔离理解。阈值由相同校准约束独立选择，部分比较存在阈值差异，消融不是纯粹单因素因果证明。','',
    '完整状态、标签、模型、阈值搜索、384策略终点和全部32家族的比较均在results/component_conflict_v1。风险预测非概率，无总体安全或非劣保证。']
    Path('research/component_conflict_v1_2026-10-23.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(out/'figure_manifest.json',dict(script_sha256=digest(__file__),analysis_sha256=digest(out/'analysis.json'),png_sha256=digest(out/'comparison.png'),exclusions=[]))

if __name__=='__main__':main()
