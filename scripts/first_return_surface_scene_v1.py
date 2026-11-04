"""Paired aligned/offgrid continuous-box scenes. No change to sealed old generator."""
from scripts.completion_catalog_scene_v1 import *

def geometry_for(seed,n,mode):
    g=shapes(n)
    if mode=='aligned': return g
    assert mode=='offgrid'
    rng=np.random.default_rng(seed+800000)
    scale=rng.uniform(.94,1.06,(n,3));shift=rng.uniform(-.037,.037,(n,3))
    return [[box*scale[i]+shift[i] for box in boxes] for i,boxes in enumerate(g)]

def prepare(seed,history_views,cfg,mode):
    rng=np.random.default_rng(seed);n=cfg['objects'];groups=np.arange(n)%4
    slots=np.array([[(i%6-2.5)*1.65,(i//6-1.5)*1.5,.08*((i%3)-1)] for i in range(n)])
    # Layout jitter is catalog metadata, known to every method.
    slots+=rng.uniform(-.04,.04,slots.shape)
    geometry=geometry_for(seed,n,mode);cat=catalog();T0=np.tile(np.eye(4),(n,1,1));T0[:,:3,3]=slots
    permutations=[];indices=[];true_poses=[]
    for t in [0,1]:
        owner=np.arange(n)
        for g in range(4):
            members=np.flatnonzero(groups==g)
            swap=rng.choice(members,2,replace=False);owner[swap]=owner[swap[::-1]]
        code=rng.integers(0,len(cat),n);code[rng.random(n)<.35]=0
        tt=np.array([T0[owner[i]]@cat[code[i]] for i in range(n)])
        permutations.append(owner);indices.append(code);true_poses.append(tt)
    positions=[[[0,0,-11],[-7,3,-9],[7,-3,-9]],[[1,2,-11],[9,1,-7]],[[0,1,11],[-9,-1,7]]]
    frames=[];anonymous=[]
    for t,pp in enumerate(positions):
        ff=[];aa=[]
        for view,pos in enumerate(pp[:history_views] if t==0 else pp):
            C=look_at(np.array(pos,float),np.zeros(3));frame,_=render(geometry,T0 if t==0 else true_poses[t-1],groups,C,cfg,np.random.default_rng(seed+10000*(t+1)+view))
            frame['min_depth']=minimum_filter(frame['depth'],size=2*cfg['free_window_radius_px']+1,mode='constant',cval=0.)
            ff.append(frame);aa.append(observed_slots(frame,slots,groups,cfg))
        frames.append(ff);anonymous.append(aa)
    centers,corners,ngrid=grid(cfg);history=[];all_branches=[];anchors=[];truth_shape=[];geometry_cache={}
    for i in range(n):
        cloud=np.concatenate([inverse(a[i]['points'],T0[i]) for a in anonymous[0]])
        if not len(cloud):raise ValueError(('missing historical anchor',seed,history_views,i))
        anchor=cloud[np.argmin(np.linalg.norm(cloud-cloud.mean(0),axis=1))];anchors.append(anchor)
        histfree=0;histbands=[]
        for frame,a in zip(frames[0],anonymous[0]):
            histfree|=free_cells(frame,T0[i],centers,corners,cfg)
            histbands.extend(band_cells(frame,T0[i],p,d,cfg,ngrid) for p,d in zip(a[i]['pixels'],a[i]['depth']))
        history.append({'free':histfree,'bands':histbands})
        tree=cKDTree(cloud);per_asset=[]
        for t in [0,1]:
            branches=[]
            for slot in np.flatnonzero(groups==groups[i]):
                points=np.concatenate([a[slot]['points'] for a in anonymous[t+1]])
                for code,increment in enumerate(cat):
                    T=T0[slot]@increment;key=(t,int(slot),code)
                    if key not in geometry_cache:
                        free=0;bands=[]
                        for frame,a in zip(frames[t+1],anonymous[t+1]):
                            free|=free_cells(frame,T,centers,corners,cfg)
                            bands.extend(band_cells(frame,T,p,d,cfg,ngrid) for p,d in zip(a[slot]['pixels'],a[slot]['depth']))
                        geometry_cache[key]=(free,bands)
                    free,bands=geometry_cache[key]
                    cost=float(np.mean(np.minimum(tree.query(inverse(points,T),workers=1)[0]/.08,1)**2)) if len(points) else 1.
                    branches.append({'free':free,'bands':bands,'slot':int(slot),'code':code,'T':T,'cost':cost})
            per_asset.append(branches)
        all_branches.append(per_asset)
        occupied=np.zeros(len(centers),bool)
        for lo,hi in geometry[i]:occupied|=((centers>=lo-1e-8)&(centers<=hi+1e-8)).all(1)
        truth_shape.append(bitset(np.flatnonzero(occupied)))
        if i%6==0:print('PREPARE',seed,history_views,i,'cached',len(geometry_cache),flush=True)
    truth={'slot1':np.array(permutations[0]),'slot2':np.array(permutations[1]),'code1':indices[0],'code2':indices[1],
           'poses':np.array(true_poses),'anchor_displacement':np.array([transform(a[None],true_poses[1][i])[0]-transform(a[None],true_poses[0][i])[0] for i,a in enumerate(anchors)]),
           'occupancy':truth_shape}
    return history,all_branches,np.array(anchors),frames,truth
