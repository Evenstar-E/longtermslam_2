"""No-GT counterexample: observed source surface, unobserved exit surface."""
import numpy as np
from inspection_decisions.uncertain_face_incidence_graph_v1 import induce_graph
from inspection_decisions.sparse_ray_material_incidence_v1 import region,polygon_distances
from scripts.run_rigid_identity_bounds_v1 import read,save


def main():
    cfg={**read('configs/sparse_ray_material_incidence_v3.json'),**read('configs/uncertain_face_incidence_graph_v1.json')};references=[]
    for epoch in [0,1]:
        pixels=[{'epoch':epoch,'view':0,'pixel':[j,0],'origin':np.zeros(3),'ray':np.array([x,y,1.]),'depth':.5,'normals':np.array([[0.,0.,1.]]),'normal_sigma':np.array([.03])} for j,(x,y) in enumerate([(.1,.2),(.08,.2),(.1,.18),(.08,.18)])]
        b={'epoch':epoch,'view':0,'inside_pixel':[0,0],'outside_pixel':[1,0],'origin':np.zeros(3),'ray_inside':pixels[0]['ray'],'ray_outside':np.array([.12,.2,1.]),'depth':.5,'outside_depth':.54,'pixel_index':0,'ray_id':0};references.append((np.zeros(6),{'pixels':pixels,'boundary':[b]}))
    graph=induce_graph(references,cfg);b=references[0][1]['boundary'][0];reg=region(np.zeros(6),b,2,0,cfg);support=np.array([.054,.5]);distance=float(polygon_distances(reg['vertices'][None],np.tile(support,(4,1))[None])[0]);assert distance<1e-12;assert len(graph['faces'])==1 and graph['faces'][0]['axis']==2 and len(graph['edges'])==0
    result={'GT_read':False,'perfect_front_source_normal':True,'raw_boundary_support_region_distance_m':distance,'source_normal_conditioned_faces':len(graph['faces']),'graph_edges':len(graph['edges']),'failure':'requiring both incident nodes to have observed source-normal support removes a valid boundary when the exit surface is unobserved','not_per_task_causal_diagnosis':True};save('research/uncertain_face_unseen_termination_counterexample_v1.json',result);print(result)


if __name__=='__main__':main()
