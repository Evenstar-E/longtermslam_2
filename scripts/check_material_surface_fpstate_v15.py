import os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
import time
from pathlib import Path
import numpy as np
import torch
from scripts.run_material_surface_distribution_v15b import configuration
from scripts.prepare_material_surface_fields_v15 import arrays,load_fields
from inspection_decisions.material_surface_distribution_v15b import dense_modes,global_proposals,RegionPair
from scripts.run_material_continuous_surface_correspondence_v14c import DensePair
from scripts.run_textured_material_comparison_v1 import save


def state():return float((np.array([70000],dtype=np.uint32).view(np.float32)*np.float32(1))[0])
def main():
    c=configuration();torch.set_num_threads(1);torch.set_flush_denormal(True);record=[('initial',state())];_,views,banks,dense=arrays(c,'rich');record.append(('after_setup',state()));load_fields(c,'rich',banks,dense);record.append(('after_load',state()));a={**banks[0][0],'normalizer':banks[0][0]['normalizer_1']};b={**banks[1][0],'normalizer':banks[1][0]['normalizer_0']};m=dense_modes(a,dense[1][0],c);record.append(('after_modes',state()));back=dense_modes(b,dense[0][0],c);base=DensePair(a,b,views[0][0],views[1][0],0,0,dense[0][0],dense[1][0],c);rp=RegionPair(base,m,back,c);record.append(('after_inverse',state()));times=[]
    for h in range(3):
        before=state();start=time.perf_counter();global_proposals(a,m,c,bool(h%2));times.append(time.perf_counter()-start);record.append((f'after_proposal{h}',state()));torch.set_flush_denormal(True)
    save(Path('results/material_surface_fpstate_v15.json'),dict(states=record,seconds=times,private_labels_read=False));print(record,times)


if __name__=='__main__':main()
