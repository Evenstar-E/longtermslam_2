import time
import numpy as np
from inspection_decisions.material_continuous_correspondence_v4 import closest_on_triangles
from inspection_decisions.material_surface_projection_batch_v4 import closest_batch


def main():
    rng=np.random.default_rng(261029);x=rng.normal(size=(32,3));tri=rng.normal(size=(32,242,3,3));valid=rng.random((32,242))>.5;valid[0]=False;fallback=rng.normal(size=(32,3))
    start=time.perf_counter()
    for _ in range(30):
        expected=np.array([fallback[i] if not valid[i].any() else closest_on_triangles(p,tri[i,valid[i]]) for i,p in enumerate(x)])
    scalar=time.perf_counter()-start;start=time.perf_counter()
    for _ in range(30):actual=closest_batch(x,tri,valid,fallback)
    batch=time.perf_counter()-start;error=float(np.max(abs(actual-expected)));assert error<1e-12
    print('PASS 32 exact scalar/batched projections, including empty support; error',error,'speedup',scalar/batch,flush=True)


if __name__=='__main__':main()
