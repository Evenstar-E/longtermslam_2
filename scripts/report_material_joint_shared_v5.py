"""V5 plots and evidence report; no experimental data changes."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_textured_material_comparison_v1 import save,sha

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    cfg=load('configs/material_joint_shared_v5.json');root=Path(cfg['output']);proto=load(root/'frozen_protocol.json');scores=load(root/'scores.json');vs=proto['base_config']['variants'];ms=proto['methods']
    ss={(x['method'],x['variant']):x for x in scores['summary'] if x['split']=='all'}
    with (root/'task_scores.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    with (root/'factor_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:ff=list(csv.DictReader(f))
    qualified={m:[ss[m,v]['qualified'] for v in vs] for m in ms};ids={m:[ss[m,v]['both_ids'] for v in vs] for m in ms}
    labels=['Ordinary frozen IDs','Ordinary joint IDs/poses','Pointwise fusion','Independent material','Shared material'];colors=['#777777','#009E73','#999999','#0072B2','#D55E00']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    x=np.arange(4);fig,axes=plt.subplots(2,1,figsize=(11.5,8.3),layout='constrained')
    for ax,data,title,limit in zip(axes,[ids,qualified],['Both-period identity correct','Identity AND material displacement p95 <= 5 cm'],[40,20]):
        for k,m in enumerate(ms):
            bars=ax.bar(x+(k-2)*.16,data[m],.16,color=colors[k],hatch='//' if k==4 else 'xx' if k==1 else None,edgecolor='black',linewidth=.4,label=labels[k]);ax.bar_label(bars,padding=2,fontsize=8)
        ax.set(xticks=x,xticklabels=vs,ylim=(0,limit),ylabel='Objects / 96',title=title);ax.grid(axis='y',alpha=.15)
    axes[1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False,fontsize=9)
    fig.suptitle('Four reused DEV scenes; same complete pose pool and finite joint-search graph',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(root/f'joint_shared_task_comparison.{ext}',dpi=180)
    plt.close(fig);fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained');scale=[]
    for v,color,style in zip(vs,['#666666','#0072B2','#D55E00','#009E73'],['-',':','--','-.']):
        z=np.sort([abs(float(a['weighted_increment'])) for a in ff if a['method']=='shared' and a['variant']==v]);scale.append(dict(variant=v,median_abs=float(np.median(z)),max_abs=float(z.max())))
        ax.step(z,np.arange(1,len(z)+1)/len(z),where='post',label=v,color=color,linestyle=style)
    ax.set(xlabel='Absolute weighted shared-minus-independent energy',ylabel='Empirical fraction of 96 objects',xlim=(0,max(a['max_abs'] for a in scale)*1.05),ylim=(0,1.02),title='Actual pixel/depth uncertainty; no enlarged material domain');ax.legend(frameon=False);ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(root/f'shared_factor_scale.{ext}',dpi=180)
    plt.close(fig)
    eq=dict(tracks=0,same_ids=0,same_poses=0);abs_sum=[];query_psi=[];both=[]
    for name in proto['scenes']:
        for v in vs:
            with np.load(root/name/f'{v}_independent.npz') as f:a=dict(f)
            with np.load(root/name/f'{v}_shared.npz') as f:b=dict(f)
            eq['tracks']+=24;eq['same_ids']+=int((a['target_slots']==b['target_slots']).all(1).sum());eq['same_poses']+=int((np.max(abs(a['transforms']-b['transforms']),axis=(1,2,3))<1e-12).sum())
            with np.load(root/name/f'{v}_selected_material.npz') as f:
                for i in range(24):
                    a,b,alpha=f[f'shared_{i}_a'],f[f'shared_{i}_b'],f[f'shared_{i}_alpha'];psi=-np.log((a*b).mean(1)/(a.mean(1)*b.mean(1)));query_psi.extend(abs(psi));abs_sum.append(float(cfg['material_weight']*alpha@abs(psi)))
                    both.extend(((np.max(abs(a-1),axis=1)>1e-6)&(np.max(abs(b-1),axis=1)>1e-6)).tolist())
    parts=[]
    for v in vs:
        rr=[a for a in rows if a['variant']==v and a['method']=='ordinary_joint'];p=dict(variant=v,qualified=sum(int(a['qualified']) for a in rr),no_good_pose=sum(not int(a['oracle_pool_qualified']) for a in rr),good_pose_wrong_id=sum(int(a['oracle_pool_qualified']) and not int(a['both_ids']) for a in rr),good_pose_correct_id_bad_selection=sum(int(a['oracle_pool_qualified']) and int(a['both_ids']) and not int(a['qualified']) for a in rr),correct_id_no_good_pose=sum(int(a['both_ids']) and not int(a['oracle_pool_qualified']) for a in rr));assert sum(p[k] for k in ['qualified','no_good_pose','good_pose_wrong_id','good_pose_correct_id_bad_selection'])==96;parts.append(p)
    save(root/'failure_partition.json',dict(scope='fixed-pool postseal diagnosis, not raw-data impossibility',records=parts));save(root/'shared_independent_equivalence.json',eq)
    data=dict(qualified=qualified,identities=ids,scale=scale,equivalence=eq,partition=parts,median_weighted_sum_abs_psi=float(np.median(abs_sum)),max_weighted_sum_abs_psi=float(max(abs_sum)),query_abs_psi_quantiles=np.quantile(query_psi,[.5,.95,1]).tolist(),fraction_queries_both_nonneutral=float(np.mean(both)));save(root/'report_source_data.json',data)
    table=lambda values:'\n'.join('| '+label+' | '+' | '.join(map(str,values[m]))+' |' for label,m in zip(labels,ms))
    ptable='\n'.join('| '+p['variant']+' | '+' | '.join(str(p[k]) for k in ['qualified','no_good_pose','good_pose_wrong_id','good_pose_correct_id_bad_selection','correct_id_no_good_pose'])+' |' for p in parts)
    stable='\n'.join(f"| {a['variant']} | {a['median_abs']:.8f} | {a['max_abs']:.8f} |" for a in scale)
    audit=load(root/'scoring_audit.json');raw=load(root/'objective_reconstruction_audit.json');reg=load(root/'real_search_regression.json')
    report=f'''# V5完整结果：联合搜索与共享材料位置误差因子（2026-10-26）

## Material Passport

- Origin Mode: local code experiment
- Verification Status: ANALYZED（原始目标已重建核验，完整推断未独立复跑）
- Scope: four reused DEV scenes; no new confirmation or real-data accuracy

**结论：普通联合搜索提高身份正确数，没有增加位移达标任务；共享位置误差因子相对独立融合没有成功增量。停止将这个具体因子作为当前主创新，不继续扫权重或扩大物理误差域。**

## 完成内容与边界

主推断{load(root/'prediction_seal.json')['elapsed_s']:.2f}s，完整4旧DEV×4外观×5方法1920行；固定V4 matcher31，没有新训练或DEV选checkpoint。V4完整union姿态池逐字节保留，被评价身份三元组内的姿态对全部穷举；这不是新的连续SE3优化。身份采用共同8旧初值、两轮各最多128新状态；邻域是旧身份链接与普通边前三目标，单/双期交换保持双射。全部目标同一个有限搜索图，不声称全局最优或检查完所有552单期交换。

源32查询、目标最多256实测voxel代表点。特征在冗余voxel内平均，XYZ取真实测点；fusion/independent/shared用完全相同数据、神经场、面积、姿态、8个像素/深度求积节点。bank比V4局部窗口更广，因此与V4比较含表示和评分变化；只有shared−independent能直接隔离共享项。普通geometry/SIFT/ray不使用新神经项。全部预测先封存再读GT，仍是反复使用DEV，未新增独立确认或真实精度。

## 联合达标（两期身份正确＋固定材料点三段p95≤5cm；每列96）

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
{table(qualified)}

五者成功集合相同，V4 union同为2/16/12/16。共享项确实改变了{eq['tracks']-eq['same_ids']}个对象的身份、{eq['tracks']-eq['same_poses']}个对象的位姿，不能称未接入或全部输出相同；变化没有带来成功增量。

## 身份正确数

| 方法 | flat | rich | repeated | changed |
|---|---:|---:|---:|---:|
{table(ids)}

联合普通搜索净增1/2/2/2正确身份，是工程收益。旧重涂test_000/11已在ordinary_frozen_ids恢复2.14cm，材料融合1.84cm；恢复发生在共享之前，不能归给共享或交换单独作用。正式普通搜索真实数据回归通过：从旧surface初值自动产生旧链接合法交换，{reg['cached_swaps']}个有缓存交换中最优能量下降{reg['best']['delta']:.8f}，原8/11交换下降{reg['archived_negative_swap']['delta']:.8f}；{reg['uncached_swaps']}个未缓存交换未算，回归不冒充新方法成绩。

## 失败分解：普通joint，互斥四类加总96

| 外观 | 达标 | 池内无好位姿 | 有好位姿但错ID | 有好位姿/对ID但选错 | 对ID且无好位姿（子集） |
|---|---:|---:|---:|---:|---:|
{ptable}

rich78/96、repeated80/96在当前池中没有达标真身份位姿；rich正确身份35项中19项也如此。后置评分不能创造缺失位姿。这是**当前池的容量问题，不是原始RGB-D不可观测的证明**。两个V4丰富纹理新增候选病例仍错身份，池oracle18不能当实际成功。

## 数学模型和实测幅度

完整公式/物理域见material_joint_shared_protocol_v5_2026-10-26.md。独立项为−log E[L1]−log E[L2]，共享项为−log E[L1L2]，差值ψ=−log(1+Cov(L1,L2)/(E[L1]E[L2]))。这是标准共同潜变量证据对比，不是原创公式声明。fusion控制为−E[log L1+log L2]。

历史节点为z/f/√12横向及±0.002m/√3光轴深度偏移的相机轴对角局部近似，未保留离轴深度交叉协方差；不是精确像素×深度乘积求积。逐点实际重投影检查节点仍在原像素±0.5及原深度范围，未扩域；源误差已积分，核中不再重复加入。未知支持L=1，Q无信息时L=1；q精确或证据对q常数时ψ=0。光滑小域下ψ≈−∇logL1ᵀΣ∇logL2，故作用随量测协方差缩小。条件材料证据与首回波门控仍不是整张图的正规化生成概率。

| 外观 | 加权共享增量绝对值中位数 | 最大值 |
|---|---:|---:|
{stable}

消除正负抵消后，0.25Σα|ψ|中位数{data['median_weighted_sum_abs_psi']:.8f}、最大{data['max_weighted_sum_abs_psi']:.8f}。单查询|ψ|中位数{data['query_abs_psi_quantiles'][0]:.3g}、95分位{data['query_abs_psi_quantiles'][1]:.5f}、最大{data['query_abs_psi_quantiles'][2]:.5f}，不是每个节点都小；但归一后的对象级共享作用小。所选假设下两期证据均非中性的查询占{100*data['fraction_queries_both_nonneutral']:.2f}%（模型诊断，不是GT可见率）。小项能翻转近并列决策，实际2个身份变化说明不能说它数学上永远不起作用；当前没有任务净收益。

## 原因、主线与下一项

已证实：普通联合搜索改善部分身份；合格数未变；共享相对独立无增量。数学和实测共同支持的解释：本因子只关联像素内位置误差，在某期缺支持或足迹很小时退化，而当前主要缺离散材料对应及准确姿态生成。不能推出所有材料共享无效、数据不可能、加大骨干必然解决。本轮骨干仍是小CNN，bank聚合也可能有训练/部署差异，没有运行新的发表作者模型。

**拒绝共享位置误差因子作为现阶段主创新；保留工程搜索和负结果证据。**下一变量应是哪些观测属于同一材料的离散对应模式，让跨期关联参与生成连续位姿。下一执行契约material_correspondence_modes_next_v6_2026-10-26.md，V6尚未实现；普通强前端与同输入模式约束对照不可省略，当前不宣称创新确认。

## 核验与复现

数学独立积分/Jensen/无信息/缺视图/单点退化、原实际ray值和合法交换检查通过。评分{audit['checks']}项；从原始公共RGB-D及冻结权重重建{raw['checks']}项，材料场误差{raw['max_field_error']:.3g}、目标最大误差{raw['max_objective_error']:.3g}；真实像素/深度域逐点核对。旧192、原数据和模型保留，未安装包或改GPU环境。最终哈希见completion_integrity.json。

入口：.venv-real/Scripts/python.exe -B -m scripts.check_material_joint_shared_v5；run_material_joint_shared_v5；score_material_joint_shared_v5；audit_material_joint_shared_objectives_v5；check_material_joint_real_regression_v5；report_material_joint_shared_v5。生成器拒绝覆盖旧根。多外观/对象/点有关联，图为描述性总数和经验分布，没有独立样本显著性声明。
'''
    rp=Path('research/material_joint_shared_results_v5_2026-10-26.md');rp.write_text(report,encoding='utf-8')
    sources=[root/p for p in ['scores.json','task_scores.csv','factor_diagnostics.csv','scoring_audit.json','objective_reconstruction_audit.json','real_search_regression.json']]
    save(root/'report_manifest.json',dict(source_sha=sha(__file__),purpose='internal descriptive research report',unit='4 reused scenes, correlated objects/appearances/queries',transformations='all included, fixed-pool partition, empirical energy distribution; no inferential intervals',sources={str(p):sha(p) for p in sources},outputs={str(p):sha(p) for p in [rp,root/'report_source_data.json',root/'failure_partition.json',root/'shared_independent_equivalence.json',*root.glob('*.png'),*root.glob('*.pdf')]}))
    print('V5 REPORT COMPLETE',qualified,ids,flush=True)

if __name__=='__main__':main()
