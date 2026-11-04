"""Numerical bound/partition/MILP unit checks, NOT research samples."""
import unittest
import numpy as np
from inspection_decisions.rigid_identity_bounds_v1 import Cell,JointProblem,pose,transform,lower_point_distances,anchor_cloud
from scipy.spatial import cKDTree

class RigidBoundsTests(unittest.TestCase):
    def test_continuous_distance_bound_on_nontrivial_poses(self):
        rng=np.random.default_rng(4);h=rng.normal(size=(55,3));q=rng.normal(size=(43,3));a,r=anchor_cloud(h)
        for scale in [.001,.1,1.,3.14]:
            center=rng.normal(size=6);half=np.array([scale]*3+[scale/2]*3);cell=Cell(0,center-half,center+half,0)
            lb=lower_point_distances(q,h,a,r,cell)
            for _ in range(25):
                v=rng.uniform(cell.lo,cell.hi);T=pose(v[:3],v[3:],a);d=cKDTree(transform(h,T)).query(q)[0]
                self.assertTrue(np.all(lb<=d+1e-10))

    def test_split_preserves_every_parent_sample(self):
        rng=np.random.default_rng(3);parent=Cell(0,-np.ones(6),np.ones(6),0);children=parent.split(.4,1)
        for point in rng.uniform(-1,1,(500,6)):
            self.assertTrue(any(np.all(point>=c.lo)&np.all(point<=c.hi) for c in children))
        self.assertEqual(len({c.key for c in children}),2)

    def problem(self):
        cfg={'distance_scale_m':.2,'geometry_weight':.6,'shape_weight':.2,'rgb_weight':.2,'motion_weight':.2,'null_cost':.75,'rgb_bins_per_channel':8,'master_time_limit_s':5.}
        h=[{'xyz':np.array([[0.,0,0],[.2,0,0],[0,.2,0]]),'rgb':np.zeros((3,3),np.uint8)}]
        c=[{'xyz':h[0]['xyz']+np.array([.3,0,0]),'rgb':np.zeros((3,3),np.uint8)},{'xyz':h[0]['xyz']+np.array([.4,0,0]),'rgb':np.zeros((3,3),np.uint8)}]
        return JointProblem(h,c,np.zeros((2,1)),cfg)

    def test_master_is_lower_bound_on_joint_common_transform(self):
        p=self.problem();master=p.master();T=pose(np.zeros(3),p.anchor[0]+[.3,0,0],p.anchor[0]);energy=p.energy(np.array([0,0]),[T])
        self.assertLessEqual(master['lower_bound'],energy+1e-10)
        for force in [(0,0),(1,0),(0,-1)]:
            r=p.master(force);a=np.array([0,0]);a[force[0]]=force[1]
            self.assertLessEqual(r['lower_bound'],p.energy(a,[T])+1e-10)

    def test_geometric_energy_does_not_multiply_on_fragment_split(self):
        p=self.problem();T=pose(np.zeros(3),p.anchor[0]+[.35,0,0],p.anchor[0]);e=p.energy(np.array([0,0]),[T])
        p.C=[{'xyz':np.concatenate([c['xyz'] for c in p.C]),'rgb':np.zeros((6,3),np.uint8)}];p.n=1;p.mass=np.array([1.]);p.appearance=np.zeros((1,1))
        self.assertAlmostEqual(e,p.energy(np.array([0]),[T]),places=12)

    def test_cross_fragment_cut_tightens_without_excluding_actual_poses(self):
        from inspection_decisions.rigid_identity_cuts_v1 import RigidityCutsProblem
        p=self.problem();p.C[1]['xyz']+=np.array([3.,0,0]);p.cfg['motion_weight']=0.
        # Rebuild roots after moving software-unit-test data, not real cases.
        cut=RigidityCutsProblem(p.H,p.C,np.zeros((2,1)),p.cfg);base=JointProblem(p.H,p.C,np.zeros((2,1)),p.cfg)
        self.assertTrue(cut.build_conflicts())
        lb=cut.master()['lower_bound'];self.assertGreater(lb,base.master()['lower_bound']+.1)
        rng=np.random.default_rng(8)
        for _ in range(20):
            r=rng.uniform(-np.pi,np.pi,3);position=rng.uniform(cut.position_lo,cut.position_hi);T=pose(r,position,cut.anchor[0])
            for a in [np.array([0,0]),np.array([0,-1]),np.array([-1,0]),np.array([-1,-1])]:
                self.assertLessEqual(lb,cut.energy(a,[T])+1e-8)

    def test_mass_cut_survives_a_single_nearby_point(self):
        from inspection_decisions.rigid_identity_mass_cuts_v1 import MassRigidityCutsProblem
        from inspection_decisions.rigid_identity_cuts_v1 import RigidityCutsProblem
        p=self.problem();p.cfg['motion_weight']=0.
        p.C[1]['xyz']=np.r_[p.C[1]['xyz']+np.array([3.,0,0]),p.C[0]['xyz'][:1]];p.C[1]['rgb']=np.zeros((4,3),np.uint8)
        whole=RigidityCutsProblem(p.H,p.C,np.zeros((2,1)),p.cfg);mass=MassRigidityCutsProblem(p.H,p.C,np.zeros((2,1)),p.cfg)
        self.assertEqual(len(whole.build_conflicts()),0);self.assertGreater(len(mass.build_conflicts()),0)
        lb=mass.master()['lower_bound'];self.assertGreater(lb,whole.master()['lower_bound']+.05)
        rng=np.random.default_rng(9)
        for _ in range(20):
            T=pose(rng.uniform(-np.pi,np.pi,3),rng.uniform(mass.position_lo,mass.position_hi),mass.anchor[0])
            for a in [np.array([0,0]),np.array([0,-1]),np.array([-1,0]),np.array([-1,-1])]:self.assertLessEqual(lb,mass.energy(a,[T])+1e-8)

if __name__=='__main__':unittest.main()
