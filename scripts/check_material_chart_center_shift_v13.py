"""Assumption counterexample: same planar texture, shifted sampling centers."""
import numpy as np
from pathlib import Path
from inspection_decisions.material_surface_correspondence_v13 import whiten_samples,phase_distribution
from scripts.prepare_material_surface_charts_v13 import configuration
from scripts.run_textured_material_comparison_v1 import save


def main():
    cfg=configuration();r=np.array(cfg['chart_radii_m'])[:,None];theta=2*np.pi*np.arange(cfg['chart_angles'])[None,:]/cfg['chart_angles'];wavelength=.03;offset=.015;k=2*np.pi/wavelength
    rgb=[]
    for center in [0.,offset]:
        intensity=.5+.4*np.cos(k*(center+r*np.cos(theta)));rgb.append(np.repeat(intensity[None,:,:,None],3,-1))
    A,_=whiten_samples(rgb[0],np.ones((1,3,cfg['chart_angles']),bool));B,_=whiten_samples(rgb[1],np.ones((1,3,cfg['chart_angles']),bool))
    a=dict(xyz=np.zeros((1,3)),texture=A);b=dict(xyz=np.zeros((1,3)),texture=B);corr,p,rel=phase_distribution(a,b,cfg);top=np.argsort(-corr[0,0])[:2];angles=360*top/cfg['chart_angles'];errors=np.minimum(angles,360-angles)
    spectrum_error=float(np.max(abs(abs(np.fft.rfft(A,axis=2))-abs(np.fft.rfft(B,axis=2)))))
    assert spectrum_error<1e-12 and corr[0,0,0]<-.99 and errors.min()>20
    save(Path('results/material_chart_center_shift_counterexample_v13.json'),dict(verified=True,true_rotation_deg=0.,center_shift_m=offset,texture_period_m=wavelength,spectrum_max_difference=spectrum_error,true_angle_correlation=float(corr[0,0,0]),top_two_angles_deg=angles.tolist(),minimum_top_two_rotation_error_deg=float(errors.min()),reliability=float(rel[0,0]),meaning='Exact assumption counterexample: independently centered patches of the same material need in-plane translation as well as rotation. Not an actual-task success/failure count, not evidence all dataset cases have this cause.'))
    print('CENTER SHIFT COUNTEREXAMPLE',angles.tolist(),'TRUE0deg, SHIFT',offset)


if __name__=='__main__':main()
