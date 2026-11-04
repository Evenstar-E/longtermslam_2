"""CPU pinhole ray caster. Metric world: x right, y up, z toward initial camera."""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw


def intrinsics(cfg):
    c = cfg["camera"]
    return np.array([[c["fx"], 0., c["cx"]], [0., c["fy"], c["cy"]], [0., 0., 1.]])


def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0., 1., 0.])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, f])
    T[:3, 3] = eye
    return T


def transform(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def project(points, K):
    p = np.asarray(points)
    return p[:, :2] / p[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]


def unproject(uv, z, K):
    uv, z = np.asarray(uv), np.asarray(z)
    return np.column_stack([(uv[:, 0]-K[0, 2])*z/K[0, 0],
                            (uv[:, 1]-K[1, 2])*z/K[1, 1], z])


def make_texture(seed, kind, label=""):
    rng = np.random.default_rng(seed)
    n = 512
    bases = {"plate": [112, 132, 143], "wall": [120, 115, 100],
             "beam": [60, 70, 74], "bolt": [183, 183, 170], "occluder": [119, 76, 35]}
    base = np.array(bases[kind])
    noise = rng.normal(0, 3, (n, n, 1))
    a = np.clip(base + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    # Deterministic surface markings, shared across visits. No change label encoded.
    count = 180 if kind == "wall" else 75
    for _ in range(count):
        x, y = rng.integers(8, n-16, 2)
        w, h = rng.integers(2, 13, 2)
        delta = int(rng.choice([-35, -18, 22, 38]))
        col = tuple(np.clip(base+delta, 0, 255).astype(int))
        d.rectangle((int(x), int(y), int(x+w), int(y+h)), fill=col)
    if kind == "plate":
        d.rectangle((7, 7, n-8, n-8), outline=(53, 70, 77), width=5)
        # Legible explicit ID is deliberately available as a cheap interface.
        d.rectangle((170, 15, 340, 65), fill=(211, 206, 175))
        d.text((225, 28), label, fill=(25, 30, 35), font_size=30)
        d.line((15, 420, 120, 410), fill=(180, 112, 56), width=4)
    return np.asarray(im)


@dataclass
class Solid:
    name: str
    kind: str
    center: np.ndarray
    half: np.ndarray
    texture: np.ndarray
    radius: float = 0.
    asset: str | None = None


def scene(cfg, removed=False, occluder="none"):
    solids = []
    def box(name, center, half, kind, tex=None, asset=None):
        solids.append(Solid(name, "box", np.array(center, float), np.array(half, float),
                            make_texture(cfg["seed"]+len(solids), kind) if tex is None else tex,
                            asset=asset))
    box("background", [0, 1.2, -.28], [2.6, 1.3, .06], "wall")
    box("floor", [0, -.16, 1.5], [3, .1, 3], "wall")
    s = cfg["scene"]
    for i, (x, aid) in enumerate(zip(s["asset_centres_x"], s["asset_ids"])):
        tex = make_texture(cfg["seed"]+101, "plate", aid)  # repeated texture, distinct ID
        box(aid, [x, s["plate_center_y"], s["plate_center_z"]],
            s["plate_half_extents"], "plate", tex, aid)
        for side in [-1, 1]:
            box(aid+"_flange_"+str(side), [x+side*.43, 1.2, .06], [.045, .6, .09], "beam", asset=aid)
        for dx in s["bolt_offsets_x"]:
            for dy in s["bolt_offsets_y"]:
                name = f"{aid}_bolt_{dx}_{dy}"
                if removed and aid == cfg["task"]["asset_id"] and dx > 0 and dy > 0:
                    continue
                solids.append(Solid(name, "sphere", np.array([x+dx, 1.2+dy, s["bolt_center_z"]]),
                                    np.zeros(3), make_texture(cfg["seed"]+88, "bolt"),
                                    s["bolt_radius_m"], aid))
    box("top_crossbeam", [0, 1.92, .02], [2.5, .06, .13], "beam")
    if occluder != "none":
        half = [.13, .18, .05] if occluder == "narrow" else [.65, .52, .05]
        box("foreground_obstruction", [.23, 1.47, .7], half, "occluder")
    return solids


def intersect(solid, origin, directions):
    """Return ray parameter (camera z depth) and world normal for nearest positive hit."""
    if solid.kind == "box":
        inv = np.divide(1., directions, out=np.full_like(directions, 1e12),
                        where=np.abs(directions) > 1e-12)
        ta = (solid.center-solid.half-origin)*inv
        tb = (solid.center+solid.half-origin)*inv
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        near, far = lo.max(axis=1), hi.min(axis=1)
        hit = (far >= np.maximum(near, 1e-5))
        t = np.where(near > 1e-5, near, far)
        t = np.where(hit, t, np.inf)
        safe = np.where(np.isfinite(t), t, 0.)
        p = origin + safe[:, None]*directions
        local = (p-solid.center) / solid.half
        axis = np.argmax(np.abs(local), axis=1)
        normal = np.zeros_like(directions)
        normal[np.arange(len(t)), axis] = np.sign(local[np.arange(len(t)), axis])
        return t, normal
    oc = origin-solid.center
    aa = np.sum(directions*directions, axis=1)
    bb = directions @ oc
    cc = oc@oc-solid.radius**2
    disc = bb*bb-aa*cc
    root = np.sqrt(np.maximum(disc, 0))
    near = (-bb-root)/aa
    far = (-bb+root)/aa
    t = np.where(near > 1e-5, near, far)
    t = np.where((disc >= 0) & (t > 1e-5), t, np.inf)
    safe = np.where(np.isfinite(t), t, 0)
    normal = (origin+safe[:, None]*directions-solid.center)/solid.radius
    return t, normal


def render(cfg, T_wc, removed=False, brightness=1., occluder="none", noise_seed=0,
           add_sensor_noise=True):
    K, c = intrinsics(cfg), cfg["camera"]
    yy, xx = np.mgrid[:c["height"], :c["width"]]
    dc = np.stack([(xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(xx)], -1).reshape(-1, 3)
    directions = dc @ T_wc[:3, :3].T
    origin = T_wc[:3, 3]
    n = len(dc)
    depth = np.full(n, np.inf)
    rgb = np.tile([72., 90., 110.], (n, 1))
    ids = np.full(n, -1, np.int16)
    normal_image = np.zeros((n, 3))
    solids = scene(cfg, removed, occluder)
    light = np.array([.3, .6, 1.])
    light /= np.linalg.norm(light)
    for sid, solid in enumerate(solids):
        t, normals = intersect(solid, origin, directions)
        take = t < depth
        idx = np.flatnonzero(take)
        if not len(idx):
            continue
        p = origin + t[idx, None]*directions[idx]
        if solid.kind == "box":
            axis = np.argmax(np.abs(normals[idx]), axis=1)
            uv = np.empty((len(idx), 2))
            for a, axes in enumerate([(2, 1), (0, 2), (0, 1)]):
                mask = axis == a
                uv[mask] = (p[mask][:, axes]-solid.center[list(axes)])/(2*solid.half[list(axes)])+.5
        else:
            no = normals[idx]
            uv = np.column_stack([.5+np.arctan2(no[:, 0], no[:, 2])/(2*np.pi),
                                  .5-np.arcsin(np.clip(no[:, 1], -1, 1))/np.pi])
        tex = solid.texture
        ix = np.clip((uv[:, 0]*(tex.shape[1]-1)).astype(int), 0, tex.shape[1]-1)
        iy = np.clip(((1-uv[:, 1])*(tex.shape[0]-1)).astype(int), 0, tex.shape[0]-1)
        shade = .58+.42*np.maximum(normals[idx]@light, 0)
        rgb[idx] = tex[iy, ix] * shade[:, None]
        depth[idx], ids[idx], normal_image[idx] = t[idx], sid, normals[idx]
    rgb = np.clip(rgb*brightness, 0, 255)
    truth_depth = np.where(np.isfinite(depth), depth, 0).reshape(c["height"], c["width"]).astype(np.float32)
    measured = truth_depth.copy()
    if add_sensor_noise:
        rng = np.random.default_rng(noise_seed)
        measured = np.where(measured > 0, measured+rng.normal(0, c["depth_noise_sigma_m"], measured.shape), 0).astype(np.float32)
        rgb = np.clip(rgb+rng.normal(0, .5, rgb.shape), 0, 255)
    return {"rgb": rgb.reshape(c["height"], c["width"], 3).astype(np.uint8),
            "depth": measured, "truth_depth": truth_depth,
            "object_ids": ids.reshape(c["height"], c["width"]),
            "object_names": [s.name for s in solids],
            "normals": normal_image.reshape(c["height"], c["width"], 3)}


def historical_roi(cfg, T_wc):
    """Authoring helper only. Runtime methods receive the resulting image annotation."""
    b = cfg["task"]["historical_roi_world_box"]
    lo, hi = np.array(b["min"]), np.array(b["max"])
    corners = np.array([[x, y, z] for x in [lo[0], hi[0]]
                        for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
    uv = project(transform(corners, np.linalg.inv(T_wc)), intrinsics(cfg))
    return [int(np.floor(uv[:, 0].min())), int(np.floor(uv[:, 1].min())),
            int(np.ceil(uv[:, 0].max())), int(np.ceil(uv[:, 1].max()))]


def path_clear(cfg, points, removed, occluder, radius):
    """Evaluator collision check for provided straight segments, not an autonomy claim."""
    solids = scene(cfg, removed, occluder)
    for start, end in zip(points[:-1], points[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(np.array(end)-start)/.02))+1)
        samples = np.linspace(start, end, count)
        for solid in solids:
            if solid.kind == "sphere":
                if np.any(np.linalg.norm(samples-solid.center, axis=1) <= radius+solid.radius):
                    return False
            else:
                delta = np.maximum(np.abs(samples-solid.center)-solid.half, 0)
                if np.any(np.linalg.norm(delta, axis=1) <= radius):
                    return False
    return True

