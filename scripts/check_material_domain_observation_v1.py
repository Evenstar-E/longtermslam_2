"""Exact small graph enumeration, cross-epoch lower bound and pool reduction."""
import itertools
import numpy as np
from inspection_decisions.material_domain_observation_v1 import observation_graph,shared_graph,cut_energy,reduced_global
from scripts.run_history_conditioned_completion_evidence_v1 import solve


def brute(g):
    free=[i for i in range(len(g['x'])) if i not in set(g['anchors'])];best=np.inf
    for bits in itertools.product([0,1],repeat=len(free)):
        z=np.ones(len(g['x']),int);z[free]=bits
        val=int(g['unary']@z)+sum(w for a,b,w in g['edges'] if z[a]!=z[b]);best=min(best,val)
    return int(best)


def main():
    cfg={'graph_neighbors':3,'graph_radius_source_px':1.,'capacity_scale':1024,'coordinate_round_decimals':12}
    rng=np.random.default_rng(88703);checks=0
    for j in range(80):
        n=int(rng.integers(1,8));uv=rng.uniform(-1,1,(n,2));event=rng.choice([-1,1],n)
        g=observation_graph(uv,event,cfg);energy,z=cut_energy(g,return_labels=True);assert energy==brute(g);checks+=1
        if j<24:
            h=observation_graph(rng.uniform(-1,1,(3,2)),rng.choice([-1,1],3),cfg);joint=shared_graph(g,h,cfg)
            ej,z=cut_energy(joint,return_labels=True);assert ej>=energy+cut_energy(h)
            assert ej==brute(joint);checks+=2
    # Same material coordinate must be one node across epochs.
    a=observation_graph([[.2,.1]],[-1],cfg);b=observation_graph([[.2,.1]],[1],cfg)
    joint=shared_graph(a,b,cfg);assert len(joint['x'])==10;assert cut_energy(joint)>=cut_energy(a)+cut_energy(b);checks+=2
    for j in range(8):
        branches=[[[{'slot':slot,'code':code} for slot in range(4) for code in range(2)] for t in [0,1]] for i in range(4)]
        cost=[rng.normal(size=(8,8)) for i in range(4)];reduced=reduced_global(branches,cost)
        full=solve(branches,[np.ones((8,8),bool) for _ in range(4)],[-v for v in cost],20.)
        assert reduced['status']==full['status']==0
        assert abs(reduced['objective']-full['negative_evidence'])<1e-8;checks+=1
    print({'status':'PASS','checks':checks})


if __name__=='__main__':main()
