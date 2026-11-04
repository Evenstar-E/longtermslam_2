"""No-GT geometry counterexamples and numerical relaxation assertions."""
import copy
import numpy as np
from inspection_decisions.shared_material_incidence_v1 import score_graph,pixel_groups
from scripts.run_rigid_identity_bounds_v1 import read


def checks():
    cfg={**read('configs/sparse_ray_material_incidence_v3.json'),**read('configs/shared_material_incidence_v1.json')}
    def row(ro,j=0,pixel=(10,10)):
        return {'epoch':2,'view':0,'inside_pixel':np.array(pixel),'outside_pixel':np.array([11+j,10]),'origin':np.zeros(3),'ray_inside':np.array([.1,.2,1.]),'ray_outside':np.array(ro),'depth':1.,'outside_depth':2.,'ray_id':j}
    graph={'faces':[{'axis':2,'offset':1.},{'axis':0,'offset':.105},{'axis':1,'offset':.205}], 'edges':[{'faces':[0,1],'span_axis':1,'span':[.19,.21]},{'faces':[0,2],'span_axis':0,'span':[.09,.11]}]}
    rows=[row([.11,.2,1.]),row([.1,.21,1.],1)];s=score_graph(np.zeros(6),rows,graph,cfg);assert s['shared']<1e-20 and s['pixels']==1
    # Same physical edge, the source face changes from z to x.
    swap=row([.105,.2,1.1]);swap['ray_inside']=np.array([.105,.2,1.]);s=score_graph(np.zeros(6),[swap],graph,cfg);assert s['shared']<1e-20
    # Two neighbors can each fit a different parallel source face exactly,
    # but one actual pixel cannot simultaneously lie on both faces.
    conflicting={'faces':[{'axis':2,'offset':.999},{'axis':0,'offset':.105},{'axis':2,'offset':1.001},{'axis':0,'offset':.095}], 'edges':[{'faces':[0,1],'span_axis':1,'span':[.19,.21]},{'faces':[2,3],'span_axis':1,'span':[.19,.21]}]}
    pair=[row([.11,.2,1.]),row([.09,.2,1.],1)];conf=score_graph(np.zeros(6),pair,conflicting,cfg);assert conf['independent']<1e-20 and conf['shared']>1e-5 and conf['contradictory_pixels']==1
    separate=copy.deepcopy(pair);separate[1]['inside_pixel'][0]+=1;ind=score_graph(np.zeros(6),separate,conflicting,cfg);assert ind['shared']<1e-20
    # Same direction can have separate material edges; no global convex filling.
    holes=copy.deepcopy(graph);holes['faces'].append({'axis':0,'offset':-.105});holes['edges'].append({'faces':[0,3],'span_axis':1,'span':[.19,.21]});left=row([-.11,.2,1.]);left['ray_inside'][0]=-.1;assert score_graph(np.zeros(6),[left],holes,cfg)['shared']<1e-20
    far=copy.deepcopy(graph);far['edges'][0]['span']=[.8,.9];one=score_graph(np.zeros(6),[rows[0]],far,cfg);assert one['shared']>one['shared_no_domain']
    empty=score_graph(np.zeros(6),rows,{'faces':[],'edges':[]},cfg);assert empty['shared']==empty['independent']==1.
    rng=np.random.default_rng(9025);maximum=0.
    for _ in range(100):
        x=np.r_[rng.normal(0,.02,3),rng.normal(0,.01,3)];v=score_graph(x,rows,graph,cfg);assert v['shared']+1e-12>=v['independent'];maximum=max(maximum,v['shared']-v['independent'])
    return {'checks':107,'GT_read':False,'conflict_independent':conf['independent'],'conflict_shared':conf['shared'],'conflict_separate_pixels_shared':ind['shared'],'random_largest_shared_minus_independent':maximum,'continuous_MAP_not_claimed':True}


if __name__=='__main__':print(checks())
