"""Algebraic limitation, not a task experiment or an impossibility result."""
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scripts.run_textured_material_comparison_v1 import save


def main():
    rng=np.random.default_rng(14);maximum=0.
    for _ in range(100):
        R1,R2=Rotation.random(2,random_state=rng).as_matrix();t1,t2=rng.normal(size=(2,3));x,y1,y2=rng.normal(size=(3,32,3));R21=R2@R1.T;t21=t2-R21@t1
        e1=y1-(x@R1.T+t1);e2=y2-(x@R2.T+t2)
        actual=y1@R21.T+t21-y2;derived=e1@R21.T-e2
        maximum=max(maximum,float(np.max(abs(actual-derived))))
        A=np.c_[R21,-np.eye(3)];assert np.linalg.matrix_rank(np.r_[np.eye(6),A])==6
    assert maximum<1e-12
    save(Path('results/material_track_information_v14.json'),dict(passed=True,random_cases=100,maximum_identity_error=maximum,independent_residual_rank=6,with_track_rank=6,identity='r_track=R21*e1-e2',scope='At fixed poses, track is a linear combination of two geometry residuals. At e1=0, its first-order Jacobian adds no rank either. It can change residual weighting/correlation, but is not a third independent observation or an original source of information. Does not prove absence of denoising/task benefit.'))
    print('TRACK IDENTITY',maximum,'rank 6 -> 6')


if __name__=='__main__':main()
