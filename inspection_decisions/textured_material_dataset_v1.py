"""Analytic RGB-D generator with persistent material textures and private labels.

All variants use one geometric first hit. Textures have no explicit ID markings.
No frozen generator is modified. This is a data tool, not a new estimator.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from visual_reinspection.renderer import look_at


VARIANTS = ('flat', 'rich', 'repeated', 'changed')


def transform(x, T):
    return np.asarray(x) @ T[:3, :3].T + T[:3, 3]


def inverse(x, T):
    return (np.asarray(x) - T[:3, 3]) @ T[:3, :3]


def camera(spec, eye, target):
    w, h, f = spec['width'], spec['height'], spec['focal_px']
    K = np.array([[f, 0., (w-1)/2], [0., f, (h-1)/2], [0., 0., 1.]])
    return dict(K=K, C=look_at(eye, target), width=w, height=h)


def make_scene(cfg, split_index, scene_index):
    seed = cfg['seed'] + 100000 * split_index + 1000 * scene_index
    rng = np.random.default_rng(seed)
    n = cfg['objects']
    # All geometry/texture instances occupy disjoint split seed ranges.
    shape_keys = rng.choice(10000, n, replace=False) + 1000000 * (split_index+1)
    texture_keys = rng.choice(10000, n, replace=False) + 4000000 * (split_index+1)
    geometry = []
    for key in shape_keys:
        gr = np.random.default_rng(int(key))
        half = gr.uniform([.20, .20, .19], [.34, .33, .29])
        boxes = [[-half, half]]
        for side in [-1, 1]:
            wh = gr.uniform(.045, .095, 2)
            xy = gr.uniform(-.12, .12, 2)
            z = side * (half[2] + .07)
            boxes.append([np.r_[xy-wh, z-.07], np.r_[xy+wh, z+.07]])
        geometry.append(np.array(boxes))
    # One repeated-shape pair per world: shape must not be an implicit ID barcode.
    geometry[-1] = geometry[-2].copy()
    shape_keys[-1] = shape_keys[-2]
    slots = np.array([[(i%6-2.5)*1.6, (i//6-1.5)*1.45, 0.] for i in range(n)])
    slots += rng.uniform(-.035, .035, slots.shape)
    poses = np.tile(np.eye(4), (3, n, 1, 1))
    poses[0, :, :3, 3] = slots
    permutations = [np.arange(n)]
    for epoch in [1, 2]:
        # destination slot of each persistent physical instance
        perm = rng.permutation(n)
        permutations.append(perm)
        rv = rng.normal(size=(n, 3))
        rv *= rng.uniform(0., .22, (n, 1)) / np.linalg.norm(rv, axis=1)[:, None]
        poses[epoch, :, :3, :3] = Rotation.from_rotvec(rv).as_matrix()
        poses[epoch, :, :3, 3] = slots[perm] + rng.uniform(-.075, .075, (n, 3))
    occluders = np.array([[[-.11, -3., -.85], [.11, 3., -.69]],
                          [[-5., -.07, .7], [5., .07, .86]]])
    changed_assets = rng.choice(n, 6, replace=False)
    # Predetermined coverage of appearance change and the identical-shape pair.
    chosen = list(dict.fromkeys([int(changed_assets[0]), n-2, n-1]))
    chosen += [int(i) for i in rng.permutation(n) if int(i) not in chosen]
    detail_assets = np.array(chosen[:cfg['detail_assets_per_scene']])
    return dict(seed=seed, geometry=geometry, shape_keys=shape_keys,
                texture_keys=texture_keys, slots=slots, poses=poses,
                permutations=np.array(permutations), occluders=occluders,
                changed_assets=changed_assets, detail_assets=detail_assets)


def view_specs(scene, cfg):
    out = []
    eyes = [[[0, 0, -11], [-7, 3, -9], [7, -3, -9]],
            [[1, 2, -11], [9, 1, -7]], [[0, 1, 11], [-9, -1, 7]]]
    for epoch, positions in enumerate(eyes):
        for v, eye in enumerate(positions):
            out.append(dict(epoch=epoch, kind='overview', group=v,
                            camera=camera(cfg['overview'], eye, [0, 0, 0])))
    for group, asset in enumerate(scene['detail_assets']):
        for epoch in range(3):
            # Cross-period overlapping front views; full views also include unseen backs.
            # Aim at the nominal slot, never encode the exact object pose in the camera.
            cr=np.random.default_rng(scene['seed']+77000+group*100+epoch)
            target=scene['slots'][scene['permutations'][epoch,asset]]
            offset=np.array([-.35+.30*epoch,.26-.10*epoch,-1.45])+cr.uniform(-.10,.10,3)
            eye = target+offset
            out.append(dict(epoch=epoch, kind='detail', group=group,
                            camera=camera(cfg['detail'], eye, target)))
    return out


def ray_box(o, d, box):
    parallel = np.abs(d) < 1e-12
    safe = np.where(parallel, 1., d)
    a, b = (box[0]-o)/safe, (box[1]-o)/safe
    low, high = np.minimum(a, b), np.maximum(a, b)
    low = np.where(parallel, -np.inf, low)
    high = np.where(parallel, np.inf, high)
    near, far = low.max(1), high.min(1)
    valid = (~(parallel & ((o < box[0]) | (o > box[1]))).any(1)) & (far >= near) & (near > 0)
    axis = low.argmax(1)
    side = (d[np.arange(len(d)), axis] < 0).astype(np.int8)
    return np.where(valid, near, np.inf), 2*axis+side


def cast(scene, epoch, C, uv, K):
    """Arbitrary subpixel rays; returned depth is camera-z, not Euclidean range."""
    dc = np.c_[(uv[:, 0]-K[0, 2])/K[0, 0], (uv[:, 1]-K[1, 2])/K[1, 1], np.ones(len(uv))]
    dw = dc @ C[:3, :3].T
    origin = C[:3, 3]
    depth = np.full(len(uv), 30.)
    owner = np.full(len(uv), -1, np.int16)
    primitive = np.full(len(uv), -1, np.int8)
    face = np.full(len(uv), -1, np.int8)
    for i, boxes in enumerate(scene['geometry']):
        T = scene['poses'][epoch, i]
        o, d = inverse(origin[None], T)[0], dw @ T[:3, :3]
        # Broad bounding box reduces detailed slab work without dropping any ray.
        bounds = np.array([boxes[:, 0].min(0), boxes[:, 1].max(0)])
        entry, _ = ray_box(o, d, bounds)
        active = np.flatnonzero(entry < depth)
        if not len(active):
            continue
        for j, box in enumerate(boxes):
            z, f = ray_box(o, d[active], box)
            take = z < depth[active]
            ii = active[take]
            depth[ii], owner[ii], primitive[ii], face[ii] = z[take], i, j, f[take]
    for box in scene['occluders']:
        z, _ = ray_box(origin, dw, box)
        take = z < depth
        depth[take], owner[take], primitive[take], face[take] = z[take], -2, -1, -1
    material = np.zeros((len(uv), 3))
    normals = np.zeros((len(uv), 3))
    for i in range(len(scene['geometry'])):
        ix = np.flatnonzero(owner == i)
        if not len(ix):
            continue
        T = scene['poses'][epoch, i]
        material[ix] = inverse(origin + depth[ix, None]*dw[ix], T)
        local_n = np.zeros((len(ix), 3))
        local_n[np.arange(len(ix)), face[ix]//2] = 2*(face[ix]%2)-1
        normals[ix] = local_n @ T[:3, :3].T
    return dict(depth_exact=depth, instance=owner, primitive=primitive,
                face=face, material_xyz=material, world_normal=normals, direction=dw)


@lru_cache(maxsize=2048)
def texture_parameters(seed):
    rng = np.random.default_rng(int(seed))
    return dict(base=rng.uniform([105, 107, 105], [165, 170, 178]),
                frequencies=rng.normal(size=(12, 3))*rng.uniform(8, 85, (12, 1)),
                phase=rng.uniform(0, 2*np.pi, 12),
                spots=rng.uniform(-.33, .33, (14, 3)),
                radius=rng.uniform(.025, .10, 14),
                lines=rng.uniform(-.35, .35, (10, 2)),
                angle=rng.uniform(0, np.pi, 10),
                length=rng.uniform(.025, .18, 10))


def albedo(points, face, seed):
    """Continuous physical-coordinate paint/stain/scratch field; no pixel seeds."""
    p = texture_parameters(int(seed))
    xyz = np.asarray(points)
    wave = np.sin(xyz @ p['frequencies'].T + p['phase'])
    values = np.broadcast_to(p['base'], (len(xyz), 3)).copy()
    values += (wave[:, :5].sum(1)*5 + wave[:, 5:].sum(1)*2)[:, None]
    for center, radius in zip(p['spots'], p['radius']):
        stain = np.exp(-np.sum((xyz-center)**2, axis=1)/(2*radius**2))
        values += stain[:, None] * np.array([27., -33., -52.])
    uv = np.zeros((len(xyz), 2))
    for axis, axes in enumerate([(1, 2), (0, 2), (0, 1)]):
        mask = face//2 == axis
        uv[mask] = xyz[mask][:, axes]
    for center, angle, length in zip(p['lines'], p['angle'], p['length']):
        q = uv-center
        along = q @ np.array([np.cos(angle), np.sin(angle)])
        across = q @ np.array([-np.sin(angle), np.cos(angle)])
        scratch = np.exp(-.5*(across/.0025)**2) * (np.abs(along) < length)
        values -= scratch[:, None]*65
    return np.clip(values, 10., 245.)


def changed_mask(points, face):
    return (points[:, 0] > -.08) & (points[:, 0] < .15) & (points[:, 1] > -.13) & (points[:, 1] < .14) & (face//2 == 2)


def render(scene, view, cfg, frame_index):
    cam = view['camera'];h,w = cam['height'],cam['width']
    yy,xx = np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()]
    truth = cast(scene, view['epoch'], cam['C'], uv, cam['K'])
    owner=truth['instance'];fg=owner>=0
    rng=np.random.default_rng(scene['seed']+999000+frame_index)
    light=np.array([.25+.12*np.sin(view['epoch']), .65, -.85+.5*view['epoch']]);light/=np.linalg.norm(light)
    shade=.58+.42*np.maximum(truth['world_normal']@light,0)
    exposure=rng.uniform(.90,1.10);wb=rng.uniform(.96,1.04,3)
    noise=rng.normal(0,cfg['rgb_noise_std'],(len(owner),3))
    rich=np.full((len(owner),3),85.,float);repeated=rich.copy();flat=rich.copy();change=np.zeros(len(owner),bool)
    for i in range(cfg['objects']):
        ix=np.flatnonzero(owner==i)
        if not len(ix):continue
        p,f=truth['material_xyz'][ix],truth['face'][ix]
        rich[ix]=albedo(p,f,scene['texture_keys'][i])
        # Shared appearance across instances is an intentional hard control.
        repeated[ix]=albedo(p,f,scene['texture_keys'][0])
        flat[ix]=[145,150,155]
        if view['epoch']==2 and i in scene['changed_assets']:
            change[ix]=changed_mask(p,f)
    changed=rich.copy();changed[change]=[172,161,134]
    observations={}
    for name, colors in zip(VARIANTS,[flat,rich,repeated,changed]):
        colors=colors.copy();colors[fg]*=shade[fg,None]*exposure*wb
        colors[fg]+=noise[fg]
        colors[owner==-2]=[60,66,72]
        observations['rgb' if name=='rich' else 'rgb_'+name]=np.clip(colors,0,255).reshape(h,w,3).astype(np.uint8)
    measured=truth['depth_exact']+rng.uniform(-cfg['depth_noise_bound_m'],cfg['depth_noise_bound_m'],len(owner))
    world=cam['C'][:3,3]+truth['direction']*measured[:,None]
    # This frontend receives measured depth and nominal spatial centers only.
    dist=np.linalg.norm(world[:,None,:]-scene['slots'][None,:,:],axis=2)
    slot=dist.argmin(1).astype(np.int16);slot[dist.min(1)>.70]=-1
    observations.update(depth=measured.reshape(h,w).astype(np.float32), C=cam['C'], K=cam['K'], anonymous_slot=slot.reshape(h,w))
    truth.pop('direction')
    truth.update(albedo=rich, changed_material_mask=change)
    truth={k:v.reshape((h,w)+v.shape[1:]).astype(np.float32) if v.dtype.kind=='f' else v.reshape((h,w)+v.shape[1:]) for k,v in truth.items()}
    return observations,truth


def correspondence_labels(scene, source, target, source_truth, samples, seed):
    """Ground-truth labels only: project same physical point and recast target ray.

    status 0=visible, 1=outside, 2=occluded. No nearest surface correspondence.
    """
    owner=source_truth['instance'].ravel();ix=np.flatnonzero(owner>=0)
    rng=np.random.default_rng(seed)
    if len(ix)>samples:ix=np.sort(rng.choice(ix,samples,replace=False))
    material=source_truth['material_xyz'].reshape(-1,3)[ix].astype(float)
    ids=owner[ix];world=np.zeros_like(material)
    for i in np.unique(ids):
        mask=ids==i;world[mask]=transform(material[mask],scene['poses'][target['epoch'],i])
    cam=target['camera'];pc=inverse(world,cam['C']);z=pc[:,2]
    safe=np.maximum(z,1e-9);uv=pc[:,:2]/safe[:,None]*np.array([cam['K'][0,0],cam['K'][1,1]])+cam['K'][:2,2]
    inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<=cam['width']-1)&(uv[:,1]>=0)&(uv[:,1]<=cam['height']-1)
    status=np.ones(len(ix),np.int8);jj=np.flatnonzero(inside)
    if len(jj):
        target_hit=cast(scene,target['epoch'],cam['C'],uv[jj],cam['K'])
        visible=(target_hit['instance']==ids[jj]) & (abs(target_hit['depth_exact']-z[jj])<2e-5)
        status[jj]=np.where(visible,0,2)
    sy,sx=np.divmod(ix,source['camera']['width'])
    return dict(source_uv=np.c_[sx,sy].astype(np.int16),target_uv=uv.astype(np.float32),
                instance=ids,material_xyz=material.astype(np.float32),status=status,
                expected_target_z=z.astype(np.float32))
