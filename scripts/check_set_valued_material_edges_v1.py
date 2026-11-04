"""Tests exposing false merge and free-support escape, plus exact covers."""
import itertools
import numpy as np
from inspection_decisions.set_valued_material_edges_v1 import minimum_support_cover,optimal_cover_relations,unrestricted_support_feasibility


def checks():
    count=0;rng=np.random.default_rng(1918)
    rows=[{'lower':1.219,'upper':1.281},{'lower':-.331,'upper':-.269}]
    result=optimal_cover_relations(rows);assert result['minimum_supports']==2 and not result['possible_same'][0][1];count+=1
    # Genuine recurrence: under a one-support shape model, two overlapping
    # period observations share the same b; moving one beyond overlap breaks it.
    overlap=[{'lower':-.01,'upper':.01},{'lower':0.,'upper':.02}];separated=[overlap[0],{'lower':.03,'upper':.05}]
    assert minimum_support_cover(overlap)['minimum_supports']==1 and minimum_support_cover(separated)['minimum_supports']==2;count+=1
    assert unrestricted_support_feasibility(separated)['feasible'];count+=1
    # Ambiguous correspondence must remain ambiguous across optimum partitions.
    chain=[{'lower':0.,'upper':1.},{'lower':.5,'upper':1.5},{'lower':1.2,'upper':2.}];rr=optimal_cover_relations(chain)
    assert rr['minimum_supports']==2 and rr['possible_same'][0][1] and not rr['forced_same'][0][1] and rr['optimal_partition_count']==2;count+=1
    for _ in range(100):
        N=int(rng.integers(1,7));a=rng.uniform(-1,1,N);b=a+rng.uniform(.01,.7,N);rows=[{'lower':x,'upper':y} for x,y in zip(a,b)];res=optimal_cover_relations(rows)
        # Independent exhaustive subset cover over all actual interval endpoints.
        points=sorted(set(np.r_[a,b]));best=N
        for k in range(N+1):
            if any(all(any(x<=p<=y for p in pp) for x,y in zip(a,b)) for pp in itertools.combinations(points,k)):best=k;break
        assert best==res['minimum_supports'];count+=1
        for p in res['points']:assert any(x<=p<=y for x,y in zip(a,b))
        assert minimum_support_cover(rows+rows)['minimum_supports']==best;count+=1
        packing=[rows[i] for i in res['packing']];assert all(packing[i]['upper']<packing[i+1]['lower'] for i in range(len(packing)-1));count+=1
    assert not minimum_support_cover([{'lower':1.,'upper':0.}])['feasible'];count+=1
    assert not optimal_cover_relations([{'lower':float(i),'upper':float(i)+.1} for i in range(13)])['relations_resolved'];count+=1
    return {'checks':count,'same_direction_different_physical_edges_preserved':True,'duplicate_segmentation_cardinality_invariant':True,'unknown_cardinality_escape_exposed':True,'minimum_cardinality_is_a_prior_not_measured_identity':True}


if __name__=='__main__':print(checks())
