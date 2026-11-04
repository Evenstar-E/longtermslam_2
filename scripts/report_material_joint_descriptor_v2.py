"""Report trained models, proper probabilities, and full-scene task endpoints."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_textured_material_comparison_v1 import save,sha


def main():
    roots=[Path('results/material_joint_descriptor_comparison_v1'),Path('results/material_joint_descriptor_comparison_v2')]
    scores=[json.loads((r/'scores.json').read_text(encoding='utf-8')) for r in roots]
    table={}
    for d in scores:
        for r in d['summary']:table[r['split'],r['variant'],r['method']]=r
    methods=['metric_rootsift','untrained','contrastive','learned_fusion','learned_material','superpoint_lightglue',
             'factorized_fusion','factorized_material','balanced_fusion','balanced_visibility','balanced_material']
    labels=['Metric SIFT','Random CNN','Contrastive','V1 fusion','V1 heads','SP+LightGlue','V2 fusion','V2 heads','Balanced fusion','+Visibility','+Heads']
    variants=['flat','rich','repeated','changed']
    old=json.loads(Path('results/textured_material_comparison_v1/scores.json').read_text(encoding='utf-8'))
    oldcounts={v:sum(r['qualified'] for r in old['summary'] if r['split'] in ['validation','test'] and r['variant']==v and r['method']=='fusion') for v in variants}
    fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for ax,v in zip(axs.ravel(),variants):
        rr=[table['all',v,m] for m in methods];x=np.arange(len(rr))
        ax.bar(x-.17,[r['qualified'] for r in rr],width=.34,label='Joint qualified')
        ax.bar(x+.17,[r['oracle_pool_qualified'] for r in rr],width=.34,label='True-ID pose pool oracle')
        ax.axhline(oldcounts[v],color='black',ls='--',lw=1,label='Old SIFT fusion')
        ax.set_xticks(x,labels,rotation=55,ha='right',fontsize=8);ax.set_ylim(0,100);ax.set_ylabel('Objects / 96');ax.set_title(v+' | 4 already-seen development scenes');ax.grid(axis='y',alpha=.2)
    axs[0,0].legend(fontsize=8);fig.savefig(roots[1]/'trained_task_comparison.png',dpi=165);plt.close(fig)
    t1=Path('results/material_joint_descriptor_training_v1');t2=Path('results/material_joint_descriptor_training_v2')
    cal1=json.loads((t1/'training_calibration.json').read_text(encoding='utf-8'))['results'];cal2=json.loads((t2/'training_calibration.json').read_text(encoding='utf-8'))['results']
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for root,name in [(t1,'V1'),(t2,'V2')]:
        logs=json.loads((root/'training_log.json').read_text(encoding='utf-8'))
        for mode in sorted({r['mode'] for r in logs}):
            rr=[r for r in logs if r['seed']==31 and r['mode']==mode]
            axs[0].plot([r['step'] for r in rr],[r['retrieval'] for r in rr],label=name+' '+mode)
    axs[0].set_xlabel('Fixed training step');axs[0].set_ylabel('Augmented training retrieval');axs[0].legend(fontsize=7);axs[0].grid(alpha=.2)
    xx=['V1','Factored','Balanced'];cc=[next(r for r in cal1 if r['mode']=='full' and r['seed']==31),next(r for r in cal2 if r['mode']=='factorized'),next(r for r in cal2 if r['mode']=='balanced' and r['seed']==31)]
    axs[1].bar(xx,[r['matchability_brier'] for r in cc]);axs[1].set_title('TRAIN visibility Brier (lower better)');axs[1].set_ylim(0,1)
    prior=21040/33792
    axs[1].axhline(prior*(1-prior),color='gray',ls='--',label='Constant visible prior')
    axs[1].legend(fontsize=7)
    axs[2].bar(xx,[r['persistence_auc'] for r in cc]);axs[2].axhline(.5,color='gray',ls='--');axs[2].set_title('TRAIN persistence AUROC');axs[2].set_ylim(0,1)
    fig.savefig(roots[1]/'training_and_calibration.png',dpi=165);plt.close(fig)
    lines=['# RGB-D材料描述子：实际训练与完整任务评测（2026-10-26）','','## Material Passport',
        '- 8个TRAIN场景、704缓存案例/33792查询；21040可见、6144画外、6608遮挡、258可见重涂。',
        '- 两版共8次固定500步训练，主种子31预先指定，47/73只作训练/描述子波动检查；没有按heldout结果选checkpoint。',
        '- 端到端覆盖全部4个旧留出场景/96构件、三期、四种外观；这些场景先前已分析，本轮属于DEV，不是独立确认。',
        '- 新网络双CNN分支约数万参数，CPU训练；无GPU环境/现有包修改。原数据、旧192分数保留。',
        '- 官方预训练源码固定commit5a650c444718caffc2669dc0094cde908f42dd73；实测深度均匀采样点的SuperPoint dense描述子＋原LightGlue图像坐标匹配，属于模块强对照，不称作者完整benchmark。','',
        '## 完整任务：全部4个DEV场景','','联合合格＝两期身份正确，固定历史材料点H→C1/H→C2/C1→C2三段p95误差均≤5cm。每格分母96。','',
        '|方法|纯色|丰富纹理|重复纹理|局部重涂|','|---|---:|---:|---:|---:|']
    lines.append('|旧SIFT普通融合（同一4场景）|'+'|'.join(str(oldcounts[v]) for v in variants)+'|')
    for m in methods:lines.append('|'+m+'|'+'|'.join(str(table['all',v,m]['qualified']) for v in variants)+'|')
    lines+=['','## 真身份共同候选池覆盖（仅事后oracle）','','|特征|纯色|丰富纹理|重复纹理|重涂|','|---|---:|---:|---:|---:|']
    for m in ['metric_rootsift','untrained','contrastive','learned_fusion','superpoint_lightglue','factorized_fusion','balanced_fusion']:
        lines.append('|'+m+'|'+'|'.join(str(table['all',v,m]['oracle_pool_qualified']) for v in variants)+'|')
    lines+=['','## 同特征、同提案的头增量','','|阶段|外观|候选/普通融合|共同正确ID|三期合格新增/丢失|共同ID的C1→C2误差 cm：基线/候选|','|---|---|---|---:|---:|---:|']
    def num(v):return '—' if v is None else f'{v*100:.2f}'
    for k,d in enumerate(scores):
        for r in d['paired_comparisons']:
            if r['split']!='all':continue
            lines.append(f"|V{k+1}|{r['variant']}|{r['method']}/{r['baseline']}|{r['common_correct_ids']}|{r['gained']}/{r['lost']}|{num(r['baseline_error_12_median_m'])}/{num(r['candidate_error_12_median_m'])}|")
    lines+=['','## 校准问题及数学修复','',
        'V1把同一m同时当dustbin相对logit与物理可见概率，实际Softmax边缘为sigmoid(m+logsumexp(s))，不等于sigmoid(m)。V2明确P(v,j,c)=P(v)P(j|v)P(c|v,j)，通过概率归一、逐行分数平移不变、可见边缘和解析梯度检查。标准概率分解修复不是原创理论。',
        'V1 TRAIN Brier0.613→factorized0.229；balanced种子31/47/73为0.229/0.225/0.220。平衡外观头TRAIN AUROC0.780/0.882/0.887，但先验校正后0.5阈值仍未检出任何258个重涂点。不能只报AUROC或训练loss称鲁棒。','',
        '## 条件描述子诊断（与端到端分开）','',
        '诊断目标集合人为包含可见GT正点及随机干扰；采样也按标签分层，不是部署搜索。只用于查看表示/头在这些条件下的泛化，不得当作全24身份成功率。']
    if (roots[1]/'heldout_patch_diagnostic.json').exists():
        patch=json.loads((roots[1]/'heldout_patch_diagnostic.json').read_text(encoding='utf-8'))['results']
        lines+=['','|模型|条件物理点检索正确率|可见性Brier|纹理持续性AUROC|重涂检出/总数|未重涂误报数|','|---|---:|---:|---:|---:|---:|']
        for r in patch:
            lines.append(f"|{r['model']}|{r['conditional_physical_retrieval']*100:.2f}%|{r['matchability_brier']:.3f}|{r['persistence_auc']:.3f}|{r['repaint_detected_at_05']}/{r['repaint_total']}|{r['unchanged_false_repaint_at_05']}|")
        lines+=['','常数可见先验的条件诊断Brier约0.236。V2约0.228比常数仅小幅改善，远不足以宣称高质量可见性估计；V1→V2的大幅差值主要是概率语义修复。未训练外观头把所有点判为重涂，虽然114/114检出，却误报10337个未重涂点。']
    if (roots[1]/'postseal_sampling_oracle.json').exists():
        sample=json.loads((roots[1]/'postseal_sampling_oracle.json').read_text(encoding='utf-8'))
        lines+=['','## 对应位置支持与算法缺口','',
            f"只在事后给定真身份和真位姿来选最近材料对应，当前相同采样位置可拟合合格{sample['qualification']}/{sample['tasks']}项。其中无近景80项可合格73项、有近景16项全部可合格；两组C2真实近邻支持中位分别6/15。这里用到了GT，不能当算法成绩，也不证明观测中的对应可被唯一辨认。",'',
            '它否定了“当前所有困难任务都只是没有采到可形成准确位姿的材料点”这一解释。真实采样位置支持通常存在，但实际描述子、匹配与RANSAC提案没有把这些链接提取出来。','',
            '训练条件诊断人为注入了真实正点，部署时使用独立均匀采样，真点未必恰好出现；条件检索由随机CNN约55.0%升到约62%并不保证实际配准。新的候选池覆盖仍只有个位数至十余项，说明训练检索改善没有充分转化为部署提案。']
    lines+=['','## 方向级结论与下一步','',
        '1. **已完成真正训练与已发表模块对照，当前“小CNN＋两个标量头”不适合作为主创新结论。** 旧同输入SIFT融合rich/repeated/changed为14/12/13；新V1普通学习融合9/3/6，V2 balanced材料5/4/4，未建立稳定系统收益。不能只报rich比随机CNN8→9的局部变化。',
        '2. **可见性概率修复属于模型正确性进展。** 头的独立增量仍弱：balanced材料与同特征visibility在四种外观中合格任务集合完全相同；factorized材料仅flat新增2，changed丢1。泛化重涂114点仍0检出。',
        '3. **不继续在本版本扫轮数/温度/阈值寻找论文收益。** 保留旧SIFT/官方方法的候选，下一模块改为部署一致的材料对应训练：使用实际独立采样目标集合，不注入保证存在的正点，显式区分物理不可见、可见但采样未覆盖、可见已覆盖；用材料坐标邻域/像素足迹分布监督替代孤立精确索引。',
        '4. **用强几何一致性配准拆开描述子与提案问题。** 普通融合与新模型必须共享旧、新候选，完整24身份和固定历史材料锚点保持；普通强对照能够解释的收益归为工程。进一步创新应围绕不确定材料对应与连续位姿/可见支持的联合推断，而不是再加一个头。新模块尚未实现。',
        '5. **仍需独立确认。** 本轮4个场景已见，尚未生成新confirmation场景。本轮也没有新增真实跨期精度结果，不作论文创新或真实泛化确认。']
    lines+=['','## 限制与判断规则','',
        '- 必须区分metric rectification/均匀采样收益、神经网络训练收益、辅助头额外收益。metricRootSIFT、随机CNN、普通contrastive和同特征融合分别提供这些对照。',
        '- 模型只编码局部RGB与几何高度/支撑，不是完整PointMBF/ViGG架构；世界轴切平面不是SE(3)等变网络。',
        '- 几何损失权重在本轮融合/可见性/材料方法均为1，补掉上一轮1→0.5的混淆；共同位姿池数量相同，计算FLOPs不相同。',
        '- 标签中的物理可见性、离散采样是否覆盖真点、错误匹配置信度不同。训练保证注入正点，而真实采样无此保证；不能用校准改善代替候选定位和联合身份结果。',
        '- 概率分解已用于V2训练和头输出，配准端仍采用启发式距离/正支持评分＋软权重，尚未实现同一个归一化概率模型下的连续材料对应与位姿联合推断。因此本轮不能作为完整概率模型优劣的验证。',
        '- 盒体/程序纹理、已知相机、名义槽位、仅4/24近景仍有利或限制。flat/repeated克隆对仍有物理身份置换对称。',
        '- 所有CNN、InfoNCE、概率分解、类平衡、RANSAC/ICP都是已有构造；论文创新必须来自超出这些控制且可复现的联合材料机制增量。','',
        '![全部DEV任务与oracle](../results/material_joint_descriptor_comparison_v2/trained_task_comparison.png)','',
        '![训练与校准](../results/material_joint_descriptor_comparison_v2/training_and_calibration.png)']
    report=Path('research/material_joint_descriptor_results_v2_2026-10-26.md');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(roots[1]/'report_manifest.json',dict(source_sha=sha(__file__),report=str(report),report_sha=sha(report),
        scores={str(r):sha(r/'scores.json') for r in roots},figures={p.name:sha(p) for p in roots[1].glob('*.png')}))
    print('LEARNED REPORT',report,flush=True)


if __name__=='__main__':main()
