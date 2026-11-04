"""No truth: unknown exit face, role swap, history unions and fast scalar math."""
import numpy as np
from inspection_decisions.typed_material_incidence_graph_v1 import (
    JointProblem,build_typed_graph,level_intervals,strip_projection,period_arrays,
    period_geometry,graph_arrays,period_cost)
from scripts.check_uncertain_face_incidence_graph_v1 import fixture
from scripts.run_rigid_identity_bounds_v1 import read


def counterexample(cfg):
    data=[]
    for epoch in [0,1]:
        pixels=[{'epoch':epoch,'view':0,'pixel':[j,0],'origin':np.zeros(3),'ray':np.array([x,y,1.]),'depth':.5,'normals':np.array([[0.,0.,1.]]),'normal_sigma':np.array([.03])} for j,(x,y) in enumerate([(.1,.2),(.08,.2),(.1,.18),(.08,.18)])]
        b={'epoch':epoch,'view':0,'inside_pixel':[0,0],'outside_pixel':[1,0],'origin':np.zeros(3),'ray_inside':pixels[0]['ray'],'ray_outside':np.array([.12,.2,1.]),'depth':.5,'outside_depth':.54,'pixel_index':0,'ray_id':0};data.append({'pixels':pixels,'boundary':[b]})
    base,graph=build_typed_graph([(np.zeros(6),d) for d in data],cfg);assert len(base['edges'])==0 and graph['termination_nodes']>=1 and len(graph['edges'])>=1;g=graph['normal_source_nodes'];dg=graph['faces'][g]['offset'];p={'epoch':2,'view':0,'pixel':[0,0],'origin':np.zeros(3),'ray':np.array([2*dg,.2,1.]),'depth':.5,'normals':np.array([[1.,0.,0.]]),'normal_sigma':np.array([.03])};b={'epoch':2,'view':0,'inside_pixel':[0,0],'outside_pixel':[0,1],'origin':np.zeros(3),'ray_inside':p['ray'],'ray_outside':p['ray']+np.array([0,0,.02]),'depth':.5,'outside_depth':.54,'pixel_index':0,'ray_id':0};data.append({'pixels':[p],'boundary':[b]});problem=JointProblem(data,graph,np.zeros(12),[.24,.16],cfg,'typed_ray');x=problem.project(problem.start());_,_,diag=problem.evaluate(x);assert diag['pixel_face'][-1]==g;trial=x.copy();trial[12]=.6;assert problem.project(trial)[12]<=.502+1e-10;unconstrained=JointProblem(data,graph,np.zeros(12),[.24,.16],cfg,'ray_no_history');assert unconstrained.project(trial)[12]==.6;return data,graph,{'old_edges':len(base['edges']),'new_edges':len(graph['edges']),'termination_node_can_be_later_source':True,'history_drift_projected':True}


def checks():
    _,_,cfg=fixture();cfg={**cfg,**read('configs/typed_material_incidence_graph_v1.json')};n=0;intervals=[(0.,1.,1.,i) for i in range(4)]+[(2.,3.,1.,i) for i in range(4)];assert level_intervals(intervals,4)==[[0.,1.],[2.,3.]];assert level_intervals([(0.,1.,1.,0)]*8,2)==[];assert strip_projection([[0,0],[0,0],[1,1],[1,1]],0,.4,.6)==[.4,.6];n+=3;data,graph,example=counterexample(cfg);rng=np.random.default_rng(102526);maximum=0.;jacmax=0.
    for method in cfg['methods']:
        p=JointProblem(data,graph,np.zeros(12),[.24,.16],cfg,method)
        for _ in range(12):
            x=p.start();x[:12]=rng.normal(0,.001,12);x[12:]+=rng.normal(0,.0001,p.F);r,J,diag=p.evaluate(x);ga=graph_arrays(graph,x[12:]);fast=0.
            for t,d in enumerate(data):
                xx=np.zeros(6) if t==0 else x[6*(t-1):6*t];fast+=period_cost(period_geometry(period_arrays(d),xx,cfg),ga,cfg,line=method=='line_joint')
            err=abs(float(r@r)-fast);maximum=max(maximum,err);assert err<1e-12;n+=1
        x=p.start()+rng.normal(0,1e-4,p.size);r,J,diag=p.evaluate(x)
        for col in range(p.size):
            hi=x.copy();lo=x.copy();hi[col]+=1e-7;lo[col]-=1e-7;rh,_,dh=p.evaluate(hi,False);rl,_,dl=p.evaluate(lo,False);analytic=float(2*(J[:,col].T@r)[0]);err=abs((rh@rh-rl@rl)/2e-7-analytic);jacmax=max(jacmax,err);assert err<2e-6,(method,col,err);n+=1
    return {'checks':n,'GT_read':False,'counterexample':example,'max_fast_scalar_difference':maximum,'max_objective_gradient_difference':jacmax,'standard_midpoint_line_control_not_author_system':True}


if __name__=='__main__':print(checks())
