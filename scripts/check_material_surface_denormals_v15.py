"""Real public graph: float32 subnormal arithmetic runtime/equivalence probe."""
import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import time
from pathlib import Path
import numpy as np
import torch
from scripts.run_material_surface_distribution_v15b import configuration
from scripts.prepare_material_surface_fields_v15 import arrays,load_fields
from inspection_decisions.material_surface_distribution_v15b import dense_modes,global_proposals
from scripts.run_textured_material_comparison_v1 import save


def main():
    torch.set_num_threads(1);c=configuration();_,_,banks,dense=arrays(c,'rich');load_fields(c,'rich',banks,dense);m=dense_modes(banks[0][0],dense[1][0],c);rows=[]
    for regional in [False,True]:
        poses=[];seconds=[]
        for flush in [False,True]:
            torch.set_flush_denormal(flush);start=time.perf_counter();poses.append(global_proposals(banks[0][0],m,c,regional));seconds.append(time.perf_counter()-start)
        assert poses[0].shape==poses[1].shape;error=float(np.max(abs(poses[0]-poses[1])));assert error<1e-8,error;rows.append(dict(regional=regional,normal_seconds=seconds[0],flush_seconds=seconds[1],maximum_pose_difference=error))
    save(Path('results/material_surface_denormal_checks_v15.json'),dict(passed=True,rows=rows,scope='Same actual1536-edge graph, float32 underflow only. No float32 normal-range values or physical parameters intentionally changed. Public no GT; not task gain or universal degeneracy proof.'));print(rows)


if __name__=='__main__':main()
