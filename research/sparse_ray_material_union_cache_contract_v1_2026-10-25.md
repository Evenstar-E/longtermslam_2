# 最近参考区域并集的等价工程加速

任一查询区域P对有限参考集的最短距离满足dist(P,union_j Q_j)=min_j dist(P,Q_j)。本轮参考是**不确定性区域**，不是材料表面。分别对无序轴对、同源轴子集做集合并集，保持所有孔洞和不连通部分；不对跨观测集合取凸包、不简化、不设置吸附网格。

使用项目虚拟环境内Shapely2.1.2/GEOS3.13.1 double precision实现。实际RGB-D及退化线/点44项检查，最近距离差最大1.74e-18m，选解相同。保留NumPy逐对版本和前级12414项等价检查。完整run4还逐项核对run3已经完成的各组扫描评分与选解（所有原完成组都核对，不筛选有利例子）。

run3在同样算法下运算较慢，停止于未完成全批状态；其完成组与中间状态保留、未GT评分。run4源与所有输入在首次GT评分前冻结。原模型/先验/参数/候选不变，工程加速不作算法创新或精度收益。

官方接口：[distance](https://shapely.readthedocs.io/en/stable/reference/shapely.distance.html)、[union_all](https://shapely.readthedocs.io/en/stable/reference/shapely.union_all.html)。
