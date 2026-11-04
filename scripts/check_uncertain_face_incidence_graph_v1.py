"""No-GT executable geometry / Jacobian / common-depth consistency checks."""
import numpy as np
from inspection_decisions.uncertain_face_incidence_graph_v1 import JointProblem,axis_evidence
from scripts.run_rigid_identity_bounds_v1 import read


def fixture():
    cfg={**read('configs/sparse_ray_material_incidence_v3.json'),**read('configs/uncertain_face_incidence_graph_v1.json')};data=[]
    for t in range(3):
        pixels=[{'epoch':t,'view':0,'pixel':[5+j,5],'origin':np.zeros(3),'ray':np.array([.1+j*.01,.2,1.]),'depth':.5,'normals':np.array([[0.,0.,1.]]),'normal_sigma':np.array([.1])} for j in range(2)]
        boundary=[{'epoch':t,'view':0,'inside_pixel':[5,5],'outside_pixel':[6,5],'origin':np.zeros(3),'ray_inside':pixels[0]['ray'],'ray_outside':np.array([.12,.2,1.]),'depth':.5,'outside_depth':.54,'ray_id':0,'pixel_index':0},{'epoch':t,'view':0,'inside_pixel':[5,5],'outside_pixel':[5,6],'origin':np.zeros(3),'ray_inside':pixels[0]['ray'],'ray_outside':np.array([.1,.22,1.]),'depth':.5,'outside_depth':.54,'ray_id':1,'pixel_index':0}]
        data.append({'pixels':pixels,'boundary':boundary})
    graph={'faces':[{'axis':2,'offset':.503},{'axis':0,'offset':.049},{'axis':1,'offset':.099}], 'edges':[{'faces':[0,1],'span_axis':1,'span':[.05,.15]},{'faces':[0,2],'span_axis':0,'span':[.02,.08]}]};return data,graph,cfg


def checks():
    data,graph,cfg=fixture();rng=np.random.default_rng(70025);initial=rng.normal(0,.001,12);maximum=0.;n=0;fitstats={}
    for method in cfg['methods']:
        p=JointProblem(data,graph,initial,[.23,.16],cfg,method);x=p.start();r,J,_=p.evaluate(x);J=J.toarray();numeric=np.empty_like(J)
        for k in range(p.size):
            hi=x.copy();lo=x.copy();hi[k]+=1e-7;lo[k]-=1e-7;numeric[:,k]=(p.evaluate(hi,False)[0]-p.evaluate(lo,False)[0])/2e-7
        error=float(abs(J-numeric).max());maximum=max(maximum,error);assert error<3e-6,(method,error,np.unravel_index(abs(J-numeric).argmax(),J.shape));n+=p.size
        fitted,log=p.fit();assert log['final_cost']<=log['initial_cost']+1e-12;assert np.all(np.linalg.norm(fitted[:12].reshape(2,6)[:,:3],axis=1)<=.23+1e-10);assert np.all(np.linalg.norm(fitted[:12].reshape(2,6)[:,3:],axis=1)<=.16+1e-10)
        if p.free_depth:assert np.max(abs(fitted[p.depth_start:]-p.z))<=cfg['depth_noise_bound_m']+1e-12
        assert p.pi[0]==p.pi[1] and p.pi[2]==p.pi[3];fitstats[method]={'before':log['initial_cost'],'after':log['final_cost'],'accepted':len(log['accepted_steps'])};n+=5
    evidence=axis_evidence(data[0]['pixels'],np.zeros(6),cfg);assert np.all(evidence[:,2]>.99) and np.all(evidence[:,:2]<1e-10);n+=1
    return {'checks':n,'max_absolute_J_error':maximum,'GT_read':False,'all_shared_source_pixels_use_one_depth_index':True,'fit_objective_checks':fitstats,'not_global_optimum_claim':True}


if __name__=='__main__':print(checks())
