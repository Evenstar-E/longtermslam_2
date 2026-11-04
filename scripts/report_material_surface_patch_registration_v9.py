"""Full comparison figure and evidence tables; no inference from correlated assets."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_material_surface_patch_registration_v9 import METHODS
from scripts.run_textured_material_comparison_v1 import save,sha

def main():
    root=Path('results/material_surface_patch_registration_v9_run2');d=json.loads((root/'scores.json').read_text(encoding='utf-8'));variants=['flat','rich','repeated','changed'];names=['Old native score','Surface geometry','Geometry + RGB','Projective points','Projective patches'];families=['old','fgr','icp','colored','projective','union'];flabel=['Retained old','+ FGR / its ICP','+ ordinary ICP','+ Colored ICP','+ patch update','All proposals'];oracle=list(csv.DictReader((root/'oracle_by_producer.csv').open(encoding='utf-8-sig')));source=[];counts=np.zeros((6,4),int);back=np.zeros_like(counts)
    for f,family in enumerate(families):
        for j,v in enumerate(variants):
            rr=[r for r in oracle if r['variant']==v];counts[f,j]=sum(float(r[f'{family}_error_m'])<=.05 for r in rr);back[f,j]=sum(float(r[f'{family}_error_m'])<=.05 and r['rear']=='1' for r in rr)
    colors=['#555555','#56B4E9','#0072B2','#E69F00','#009E73'];hatches=['','//','..','\\','xx']
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        fig,(a,b)=plt.subplots(1,2,figsize=(13,5.8),gridspec_kw={'width_ratios':[1.35,1]});fig.subplots_adjust(left=.06,right=.98,top=.77,bottom=.2,wspace=.4)
        for col,m in enumerate(METHODS):
            vals=[next(r['qualified'] for r in d['summary'] if r['variant']==v and r['method']==m) for v in variants];bars=a.bar(np.arange(4)+(col-2)*.15,vals,.14,color=colors[col],hatch=hatches[col],edgecolor='black',linewidth=.5,label=names[col])
            for bar,val in zip(bars,vals):a.text(bar.get_x()+bar.get_width()/2,val+.35,str(val),ha='center',fontsize=8)
            source.extend(dict(method=m,variant=v,qualified=int(val),total=24) for v,val in zip(variants,vals))
        a.set(xticks=np.arange(4),xticklabels=['Flat','Rich','Repeated','Repainted'],ylim=(0,24),yticks=np.arange(0,25,4),ylabel='Qualified tasks / 24',title='Actual identity + displacement success');a.grid(axis='y',alpha=.2);a.set_axisbelow(True);im=b.imshow(counts,vmin=0,vmax=24,cmap='Blues',aspect='auto')
        for f in range(6):
            for j in range(4):b.text(j,f,f'{counts[f,j]} ({back[f,j]})',ha='center',va='center',color='white' if counts[f,j]>=16 else 'black')
        b.set(xticks=np.arange(4),xticklabels=['Flat','Rich','Repeat','Repaint'],yticks=np.arange(6),yticklabels=flabel,title='Candidate oracle: all 24 (rear 8)\nKnown IDs; not deployed success');handles,labels=a.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.98));fig.text(.5,.09,'One TRAIN scene; appearances and objects are correlated. No confidence intervals / generalization claim.',ha='center');fig.text(.5,.04,'Same final candidates within V9; Colored ICP is a shared proposal backend, not a standalone published-system benchmark.',ha='center',fontsize=9);fig.savefig(root/'task_and_candidate_comparison.png',dpi=180,facecolor='white');fig.savefig(root/'task_and_candidate_comparison.pdf',facecolor='white');plt.close(fig)
    save(root/'figure_provenance.json',dict(sources={str(root/'scores.json'):sha(root/'scores.json'),str(root/'oracle_by_producer.csv'):sha(root/'oracle_by_producer.csv')},source_table=source,oracle_all24=counts.tolist(),oracle_rear8=back.tolist(),meaning='All five methods and four appearances retained, zeros labeled; oracle separate from actual success. One TRAIN scene; no independent-sample CI. Internal figure, journal requirements unspecified.',dimensions_inches=[13,5.8],dpi=180))
    table=[];cn=['旧native评分','表面几何','普通几何纹理融合','逐点投影','表面块投影']
    for m,label in zip(METHODS,cn):
        vals=[next(r['qualified'] for r in d['summary'] if r['variant']==v and r['method']==m) for v in variants];table.append('|'+label+'|'+'|'.join(str(x) for x in vals)+'|')
    ot=['|'+label+'|'+'|'.join(f'{counts[i,j]}（{back[i,j]}）' for j in range(4))+'|' for i,label in enumerate(flabel)];comp=['|'+r['variant']+'|'+r['baseline']+'|'+str(r['gained'])+'|'+str(r['lost'])+'|' for r in d['comparisons'] if r['candidate']=='material_patch'];diagnostics=[json.loads((root/f'{v}_diagnostics.json').read_text(encoding='utf-8')) for v in variants];seal=json.loads((root/'prediction_seal.json').read_text(encoding='utf-8'));sampled=[]
    for v in variants:
        for radius in ['6mm','15mm']:
            rr=[r for r in oracle if r['variant']==v];key=f'association_{radius}_error_m';good=[r for r in rr if r[key] and float(r[key])<=.05];sampled.append(f"- {v}，标签辅助{radius}近邻对应后再SVD拟合：{len(good)}/24（背面{sum(r['rear']=='1' for r in good)}/8）。这需要真实身份与材料坐标关联，只是可形成位姿的理想对照，不是实际结果或唯一可辨识性证明。")
    text='''# V9 实测表面块主线实验

2026-10-26。一个已见 TRAIN 场景，24对象、三期、四外观；五方法共480条主要记录，另有两旧方法对照，总计672行。没有新网络训练或独立确认。旧192项任务未重跑也未改写。

## 本轮实现

从原公开RGB-D提取每期每对象512实测表面点、48个五像素块；保存全部原始像素/相机/深度来源。官方Open3D FPFH/FGR全局几何、point-to-plane ICP、Colored ICP与新增连续块投影更新共同产生位姿，所有评分使用相同最终候选。旧V8离散池与cache中各评分选中的耦合端点保留；V8未选中的所有拟合端点未逐个重新纳入，因此“retained old”oracle不是V8全端点oracle的完全嵌套复刻。原版强预测端点及全部旧身份状态保留。

五个评分依次是旧native、同表示几何、普通几何颜色融合、同实际像素/曝光的逐点投影、使用块内纹理空间排列的投影。数学与已发表方法边界见 [模型契约](material_surface_patch_model_v9_2026-10-26.md)。Colored ICP真实执行为共享提案后端，五列并不是作者完整系统原始benchmark。

## 实际任务结果

每格分母24；身份两期正确，且三时间段固定历史材料锚点P95位移误差都≤5cm。背面对象保留。

|方法|无纹理|丰富纹理|重复纹理|局部重涂|
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

表面块相对每个对照的逐对象变化：

|外观|对照|新增对象|丢失对象|
|---|---|---|---|
'''+ '\n'.join(comp)+'''

## 准确位姿是否真的产生

下表为已知正确身份的候选oracle，格式“全24（背面8）”；不当作实际成功。各新增单类以相同retained-old＋FGR/其ICP为基础，再加入本类实际拟合端点。Union为最终共同池。

|候选族|无纹理|丰富纹理|重复纹理|局部重涂|
|---|---:|---:|---:|---:|
'''+ '\n'.join(ot)+'''

![实际任务与候选上限](../results/material_surface_patch_registration_v9_run2/task_and_candidate_comparison.png)

## 理想关联与真实位姿目标值

'''+ '\n'.join(sampled)+'''

真实位姿与真实身份的完整目标值只在预测封存后计算，详见scores.json的true_pose_costs。它用于区分“当前候选缺好解”和“目标本身偏好其他解释”，没有作为新推断输入或回改参数。理想关联强不等于普通观测能唯一识别这些关联。

## 运行与核验

'''+ f"- 推断总耗时{seal['elapsed_s']:.2f}秒；普通ICP {d['fit_counts'][0]}次，Colored ICP {d['fit_counts'][1]}次，新投影更新{d['fit_counts'][2]}次。\n- 新投影更新内部目标下降{sum(x['counters']['projective_improved'] for x in diagnostics)}次；这不是任务成功数。Colored ICP {sum(x['counters']['colored_no_correspondences'] for x in diagnostics)}次无可用对应，保持该初值并记录，不删除任务。\n- {d['raw_points_verified']}个表面代表点和{d['raw_patch_samples_verified']}个块像素的RGB/XYZ/匿名槽位来源核对通过；{d['checks']}项评分/候选/状态核对；选中目标重建最大误差{d['max_selected_objective_error']:.3g}。\n- 四个三元组无下界剪枝完整枚举与正式最小值比较最大差异{max(r['max_error'] for r in d['pruning_checks']):.3g}。\n"+'''
- 22项双线性深度/颜色梯度、反向位姿雅可比、遮挡/自由空间与回溯检查通过。梯度最大数值差异约2e-10。
- 首次尝试只有协议、未产生预测；epoch变量被文件句柄覆盖导致停止。失败源快照和说明保留，修复后新run2根完整重跑。该工程错误不作为算法负结果。
- 模型是已知投影、块相关和优化工具的任务原型；没有训练新骨干、没有连续隐藏表面重建、没有校准拒绝模块。局部去均值GN是近似方向，完整NCC能量回溯保证下降，不保证找到最好解。

复现入口：

```powershell
.venv-real/Scripts/python.exe -m scripts.check_material_surface_patch_registration_v9
.venv-real/Scripts/python.exe -m scripts.cache_material_surface_geometry_v9
.venv-real/Scripts/python.exe -m scripts.run_material_surface_patch_registration_v9 --output results/material_surface_patch_registration_v9_run2
.venv-real/Scripts/python.exe -m scripts.score_material_surface_patch_registration_v9
```

已封存目录拒绝覆盖，重跑须使用新的输出目录。采样/候选生成与评分代码、所有原始表格和失败记录保留。后续判断必须分别报告链路、候选收益、实际收益、独立于普通控制的模型收益；不因内部目标下降或oracle提升就确认创新。
'''
    if (root/'pose_mechanism.json').exists():
        text+='''
## 主线收益、失败原因与下一优先级

本轮确实取得工程收益：无纹理的同表示逐点投影达到11/24（上一轮持续版3/24），普通几何纹理融合在重复纹理达到16/24（上一轮14/24）。这两种方法各有取舍，不能按外观挑最好成绩合成一个不存在的11/16/16/16系统。新表面块版本为6/16/14/16，未超过上述对照；无纹理相对同输入逐点投影1增6失，重复纹理相对普通融合0增2失。不能把新公共池的工程收益归给NCC块机制。

材料块在rich和changed两期身份都达到24/24，但所有背面八对象位移仍失败。rich普通几何颜色与逐点投影同样有24正确身份；changed两者为22，块版为24，属于身份子模块局部收益，尚未转成完整任务增益，也不是稳健的新颖性确认。

新候选在无纹理上限从retained-old7提升到14，主要普通ICP/ColoredICP带来；新增块更新族只有8。丰富纹理、重复纹理和重涂的候选上限仍17/16/17，未补出背面缺失的准确位姿。富纹理/重涂背面八项身份已对，却只有一项候选合格，且最终仍未选中。

更关键的是目标偏好：rich和changed八个身份正确的背面错误解，块目标均比真实位姿得分更优。rich对象7旋转误差只有0.0704°，但固定材料位移误差0.468m；它把历史正面贴到了当前反面，目标1.215，真实位姿却1.604。仅增加优化轮数无法消除这种偏好。

使用真实五像素几何的有向法向、以及当前深度场导数检查，rich八项中六项的错误H–C2命中大多/全部法向相反，另两项是约180°翻转且法向相容。说明“同匿名构件的深度命中”不能直接等同于“同一有向材料表面”。法向约束能针对前六项，但不能据此声称解决两个翻转，也不能把标准法向过滤包装成创新。

真实位姿在这些H–C2块上仅0–3/240像素命中；512点表示中6mm标签辅助近邻只有0–1个H–C2点，不足三点拟合。放宽到15mm、利用真实材料坐标关联后七项可拟合达标，但这是带标签的邻域几何oracle，不证明原始图像能自动辨认这些关联，更不证明整个原始数据物理不可观测。

下一步优先做**有向材料表面物理兼容的同输入对照**，验证能否消除上述实际反面错误支持，并把已有合格候选转成成功；普通法向过滤必须是基线。然后才评估是否值得建立“持续有向表面＋新出现/未见表面”的联合模型，专门区分同构件的新表面和持续材料点。保留全部24×4任务，不能用剔除背面或缩成单物体来制造收益。

新GN更新采用近似的局部去均值颜色方向，中心化尚未按每一视图实际hit集合完全一致地构造，可能降低遮挡边界上的优化效率。完整NCC评分和回溯仍正确，当前实验不是最优求解证书。不过八项错误位姿目标已经低于真实解，故本轮优先级是观测语义/有向表面兼容，而不是先打磨这个优化近似。若标准物理兼容已经解释全部收益，就将其归为工程修复，继续要求新联合模型有独立净增。

结论分开记录：主线链路推进；数据表示/普通配准有实际收益；NCC块新机制未获完整任务创新收益。相关数据来自一个已见TRAIN；仍需冻结独立场景、现代已发表强3D完整方法和真实跨期数据，才能讨论论文竞争力。
'''
    Path('research/material_surface_patch_results_v9_2026-10-26.md').write_text(text,encoding='utf-8');print('V9 REPORT/FIGURE WRITTEN')

if __name__=='__main__':main()
