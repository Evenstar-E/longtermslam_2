"""Complete-pose cost inside legal ID swaps; standard engineering control."""
import numpy as np
from inspection_decisions.material_correspondence_inference_v3 import inverse,ray_batch


def ordinary_score(scorer,poses,chunk=256):
    """Exact V3 ray objective, larger batching; no geometric approximation."""
    result=[]
    for start in range(0,len(poses),chunk):
        T=np.asarray(poses[start:start+chunk]);B=len(T)
        x=np.einsum('bij,nj->bni',T[:,:3,:3],scorer.a['points'])+T[:,None,:3,3]
        U=inverse(T);y=np.einsum('bij,nj->bni',U[:,:3,:3],scorer.b['points'])+U[:,None,:3,3]
        d1=scorer.b['tree'].query(x.reshape(-1,3))[0].reshape(B,-1);d2=scorer.a['tree'].query(y.reshape(-1,3))[0].reshape(B,-1)
        g=.5*(np.minimum(d1/.04,1).mean(1)+np.minimum(d2/.04,1).mean(1));mm=scorer.match
        if len(mm['ia']):
            x=np.einsum('bij,nj->bni',T[:,:3,:3],scorer.a['xyz'][mm['ia']])+T[:,None,:3,3]
            r=x-scorer.b['xyz'][mm['ib']][None];w=np.exp(-.5*(mm['distance']/.5)**2)
            texture=1-np.minimum(1,(np.exp(-.5*(r*r).sum(2)/.025**2)*w[None]).sum(1)/6)
        else:texture=np.ones(B)
        world=np.einsum('bij,nj->bni',T[:,:3,:3],scorer.ga['xyz'])+T[:,None,:3,3]
        free=(ray_batch(world,scorer.frames,scorer.slot)['free']*scorer.ga['area'][None]).sum(1)
        result.append(g+texture+.25*free)
    return np.concatenate(result)


def legal_neighbors(slots,allowed,pair_cost,old_allowed):
    """Every output is a full bijection; priority from old predictions, no GT."""
    n=len(slots);actions=set()
    for e in [0,1]:
        inv=np.argsort(slots[:,e])
        for i in range(n):
            for target in allowed[e][i]:
                j=int(inv[target])
                if i!=j:actions.add((e,min(i,j),max(i,j)))
    actions|={(2,i,j) for e,i,j in list(actions)}
    out=[]
    for e,i,j in sorted(actions):
        new=slots.copy()
        for k in ([0,1] if e==2 else [e]):new[[i,j],k]=new[[j,i],k]
        def proxy(s):
            return sum(pair_cost[0][a,s[a,0]]+pair_cost[1][a,s[a,1]]+.5*pair_cost[2][s[a,0],s[a,1]] for a in [i,j])
        priority=not any(new[a,k] in old_allowed[k][a] and new[a,k]!=slots[a,k] for a in [i,j] for k in [0,1])
        out.append((int(priority),float(proxy(new)-proxy(slots)),new))
    return out


def common_search(initials,objective,allowed,pair_cost,old_allowed,rounds=2,per_round=128):
    """All objectives see exactly the same evaluated assignment states.

    Full pose pairs are optimized by callback for every object triple. The
    finite neighborhood graph is shared; this is not a global-optimum claim.
    """
    seen={};states=[];costs=[];paths=[]
    def add(slots):
        key=slots.tobytes()
        if key in seen:return
        assert all(np.array_equal(np.sort(slots[:,e]),np.arange(len(slots))) for e in [0,1])
        val=sum((objective(i,int(j),int(k))[0] for i,(j,k) in enumerate(slots)),start=np.zeros(4))
        seen[key]=len(states);states.append(slots.copy());costs.append(val)
    for x in initials:add(x)
    initial_count=len(states)
    for _ in range(rounds):
        allcost=np.array(costs);best=np.argmin(allcost,axis=0);proposals=[]
        for index in np.unique(best):proposals.extend(legal_neighbors(states[index],allowed,pair_cost,old_allowed))
        proposals.sort(key=lambda x:(x[0],x[1],x[2].tobytes()));added=0
        for _,__,slots in proposals:
            if slots.tobytes() in seen:continue
            add(slots);added+=1
            if added>=per_round:break
        paths.append(np.min(np.array(costs),axis=0).tolist())
        if not added:break
    costs=np.array(costs);return dict(states=np.array(states),costs=costs,initial_count=initial_count,
        best=np.argmin(costs,axis=0),initial_best=np.argmin(costs[:initial_count],axis=0),paths=paths)
