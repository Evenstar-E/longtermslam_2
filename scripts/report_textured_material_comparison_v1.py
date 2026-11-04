"""Create readable report and exported plots from sealed scores, no inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/textured_material_comparison_v1');args=ap.parse_args()
    root=Path(args.root);d=json.loads((root/'scores.json').read_text())
    cfg=json.loads((root/'frozen_protocol.json').read_text())['config']
    methods=cfg['methods'];variants=cfg['variants'];labels=['Geometry','Texture','Fusion','+Visibility','+Persistence','Both']
    table={(r['split'],r['variant'],r['method']):r for r in d['summary']}
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    for ax,v in zip(axs.ravel(),variants):
        rr=[table['test',v,m] for m in methods]
        xx=np.arange(6)
        ax.bar(xx-.17,[r['both_ids'] for r in rr],width=.34,label='Both identities',color=colors[0])
        ax.bar(xx+.17,[r['qualified'] for r in rr],width=.34,label='Identity + all motion errors <=5cm',color=colors[1])
        ax.set_xticks(xx,labels,rotation=20,ha='right');ax.set_ylim(0,50)
        ax.set_ylabel('Objects / 48');ax.set_title(v+' | 2 held-out scenes');ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8);fig.savefig(root/'heldout_task_comparison.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,v in enumerate(variants):
        for k,m in enumerate(['fusion','fusion_visibility','fusion_persistence','persistent_visible']):
            r=table['test',v,m];cc=r['correspondence'];xx=j+(k-1.5)*.18
            axs[0].bar(xx,cc['visible']['correct_recall'] or 0,width=.17,color=colors[k],label=labels[k+2] if j==0 else None)
            hidden=[cc[c] for c in ['occluded','outside']];den=sum(c['total'] for c in hidden)
            axs[1].bar(xx,sum(c['false_accepted'] for c in hidden)/den if den else 0,width=.17,color=colors[k])
    for ax in axs:
        ax.set_xticks(range(4),variants);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    axs[0].set_title('Visible material correspondence recall (2cm)');axs[0].legend(fontsize=8)
    axs[1].set_title('False acceptance on occluded/outside points')
    fig.savefig(root/'heldout_correspondence_comparison.png',dpi=170);plt.close(fig)
    lines=['# 纹理—几何—持续材料匹配首轮结果（2026-10-26）','','## Material Passport',
        '- 类型：实际执行的非学习机制比较；不是预训练网络或训练收益。',
        '- 输入：12个合成场景、24构件/场景、三期、四种同几何外观；全部预测封存后评分。',
        '- 留出：2验证场景、2测试场景。测试48构件相关，仅2个独立场景；不能声称统计确认或真实域泛化。',
        '- RootSIFT、ICP、可见性裁剪、三期一致性均为已有机制；组合收益不自动等于数学创新。','',
        '## 测试集主表','','联合合格要求两期身份均正确，固定历史表面材料点的H→C1/H→C2/C1→C2三段p95误差均≤5cm。条件误差仅对正确身份统计，不能脱离分母比较。','',
        '|外观|方法|两期身份正确/48|联合合格/48|正确身份下C1→C2中位误差 cm|可见材料对应召回|接受精度|',
        '|---|---|---:|---:|---:|---:|---:|']
    def num(v,scale=1):return '—' if v is None else f'{v*scale:.2f}'
    for v in variants:
        for m in methods:
            r=table['test',v,m]
            lines.append(f"|{v}|{m}|{r['both_ids']}|{r['qualified']}|{num(r['correct_id_error_12_median_m'],100)}|{num(r['correspondence']['visible']['correct_recall'],100)}%|{num(r['accepted_precision'],100)}%|")
    lines += ['','## 与普通融合共同正确身份的比较','','|外观|新增机制|共同正确身份数|C1→C2基线/候选中位 cm|联合合格新增/丢失|共同样本最坏误差改善/变差|',
              '|---|---|---:|---:|---:|---:|']
    for r in d['paired_comparisons']:
        if r['split']!='test':continue
        lines.append(f"|{r['variant']}|{r['method']}|{r['common_correct_ids']}|{num(r['common_error_12_baseline_median_m'],100)}/{num(r['common_error_12_candidate_median_m'],100)}|{r['gained']}/{r['lost']}|{r['common_worst_better']}/{r['common_worst_worse']}|")
    if (root/'postseal_failure_diagnostics.csv').exists():
        with (root/'postseal_failure_diagnostics.csv').open(encoding='utf-8-sig') as f:diag=list(csv.DictReader(f))
        lines+=['','## 事后原因诊断','','真身份和真运动只用于以下候选覆盖/材料对应诊断，未回灌模型。','',
            '|外观|真ID共同候选池可合格/48|H→C1/C2真材料内点中位数|共同历史描述子中位数|',
            '|---|---:|---:|---:|']
        for v in variants:
            rr=[r for r in diag if r['split']=='test' and r['variant']==v]
            lines.append(f"|{v}|{sum(int(r['oracle_pool_qualified']) for r in rr)}|{np.median([int(r['true_h1_material_inliers']) for r in rr]):.1f}/{np.median([int(r['true_h2_material_inliers']) for r in rr]):.1f}|{np.median([int(r['shared_historical_descriptor_indices']) for r in rr]):.1f}|")
    lines+=['','## 本轮直接结论','',
        '1. **纹理有实际信息收益，候选尚未全面胜出。** rich联合合格：geometry 0、texture 7、fusion 7、persistent_visible 7 /48；repeated为0/5/6/6，changed为0/5/6/5。不能把普通纹理收益当作新机制收益。',
        '2. **局部量化改善有共同样本证据。** rich与fusion共同正确ID的9项，C1→C2误差中位1.93→0.91cm，最坏材料误差5项改善、0项变差；但两期正确ID总数15→9且联合合格仍7。与texture共同正确的8项为1.11→0.87cm，联合合格集合没有新增。不能用不同正确ID子集的41.77→0.91cm作改善证据。',
        '3. **可见性版本已复现几乎全部遮挡改善，额外持续性增量很小。** rich隐藏点误接受fusion 23.36%、visibility-only 11.99%、both 11.98%；可见材料召回80.74%、81.28%、81.73%。但可见性版本也把几何权重从1改为0.5，尚无同权重无可见性控制，不能把全部改善单独归因首回波建模。只加持续性没有新增合格任务。',
        '4. **重涂时仍有伤害。** changed测试test_000资产22：fusion两期ID正确，最坏误差4.53cm；visibility-only/both把第二期身份配错，约6.15m误差。true-ID候选仍保留可合格解。该例表明可见性组合未保护弱外观下的身份，不能声称已解决外观变化。',
        '5. **固定候选池已经饱和。** rich真ID共同池oracle7/48，候选7/48；repeated为6/6；changed为oracle6、候选5。oracle只是事后诊断，不是方法得分，也不证明原始RGB-D信息不足。不得继续在rich旧池上扫评分权重，期待提高合格数。']
    lines+=['','## 可见重涂的拒绝代价','',
        '测试changed条件有924个可见重涂材料点。身份/材料对应仍存在，不能把外观改变直接标成物理无对应。','',
        '|方法|正确接受/924|错误接受/924|正确召回|','|---|---:|---:|---:|']
    for m in ['texture','fusion','fusion_visibility','fusion_persistence','persistent_visible']:
        cc=table['test','changed',m]['correspondence']['repaint_visible']
        lines.append(f"|{m}|{cc['correct']}|{cc['false_accepted']}|{num(cc['correct_recall'],100)}%|")
    lines+=['','候选重涂错误接受18→0，但正确召回39.07→13.31%。这主要说明变得更保守，不能称作重涂鲁棒性已改善。']
    if (root/'postseal_coverage_analysis.json').exists():
        cv=json.loads((root/'postseal_coverage_analysis.json').read_text())
        lines+=['','## 采样与尺度限制','',
            'rich真ID池的7个可合格任务全部来自有近景的8个构件；无近景40项为0。GT位姿诊断：无近景组历史SIFT在C2仍可见的数量中位3，历史特征在两期都可见的中位0；近景组对应81/73.5。它定位当前稀疏特征/位姿提案的限制，不是全部原始像素不可观测证明。','',
            '像素材料对应并非对象均衡指标：rich测试可见标注11091个来自近景、2619个来自全景，约80.9%为近景。fusion→both在近景召回86.76→88.62%，全景55.21→52.54%；总体召回提高掩盖了全景退化。保留全部48构件主表，不删除40个困难任务。','',
            '下一模块应实现可训练的跨尺度RGB-D材料描述子与无对应分支，让正对应、遮挡/画外、可见重涂分别受监督；普通融合与候选必须共享新特征和提案。预训练局部匹配作为共享强对照，而非只给候选升级输入。三期身份与固定历史材料锚点仍为主任务。',
            '本轮test已被事后分析；后续架构开发视为已见开发集。模型和数据采集协议固定后，再生成独立种子的新场景确认；尺度/视场覆盖应预先分层，不能只保留近景或为候选单独加观测。']
    lines+=['','## 解释边界与下一步','',
        '- 几何结果应跨四种RGB完全一致，评分审计逐数组核验此条件。所有融合对照共享每条身份边的位姿提案；持续性项增加计算，未宣称等FLOPs。',
        '- 可见性组更低错误接受必须与可见召回共同读；仅降低覆盖不能作为位移量化或算法创新收益。对应指标是位姿诱导的材料对应，并非原始SIFT匹配精度。',
        '- flat/repeated中的22、23号是完全相同几何与材料外观的克隆对；严格物理身份存在置换对称。保持严格主表，不把对称性错误包装成可被任意特征学习解决的问题。',
        '- 当前仍使用已知相机与名义空间槽位、程序化盒体和纹理。所有192项旧任务保留；旧120/192不得跨数据直接排名。',
        '- 下一步依据真身份候选覆盖和真实材料内点诊断，接入预训练局部匹配强对照；若提案已充分而排序失败，优先改观测/不匹配模型。训练只能使用train，并保留普通融合及两个消融。','',
        '![测试集身份与位移](../results/textured_material_comparison_v1/heldout_task_comparison.png)','',
        '![材料对应召回与隐藏点误接受](../results/textured_material_comparison_v1/heldout_correspondence_comparison.png)']
    report=Path('research/textured_material_comparison_results_v1_2026-10-26.md')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'report_manifest.json',dict(report=str(report),report_sha=sha(report),
         source_sha=sha(__file__),figures={p.name:sha(p) for p in root.glob('*.png')},
         scores_sha=sha(root/'scores.json')))
    print('REPORT',report,flush=True)


if __name__=='__main__':main()
