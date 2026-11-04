import numpy as np
from pathlib import Path
from inspection_decisions.material_surface_correspondence_v13b import circular_interpolate
from scripts.run_textured_material_comparison_v1 import save


def main():
    p=np.arange(1,17,dtype=float);p/=p.sum();p=p[None,None];angles=[0.,-1e-16,1e-16,2*np.pi,-2*np.pi,4*np.pi];checks=0
    for angle in angles:
        value=circular_interpolate(p,np.array([[angle]]));assert abs(float(value[0,0])-float(p[0,0,0]))<1e-14;checks+=1
    angles=np.linspace(-7,7,1001).reshape(-1,1);prob=np.broadcast_to(p,(len(angles),1,16));a=circular_interpolate(prob,angles);b=circular_interpolate(prob,angles+2*np.pi);error=float(np.max(abs(a-b)));assert error<1e-13;checks+=1
    save(Path('results/material_surface_correspondence_periodic_checks_v13b.json'),dict(passed=True,checks=checks,periodic_grid=1001,max_periodic_error=error,scope='Index correctness only, no research efficacy.'));print('V13B PERIODIC CHECK',checks,error)


if __name__=='__main__':main()
