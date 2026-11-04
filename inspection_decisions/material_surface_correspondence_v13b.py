"""Periodic-index repair only; measured-chart representation remains frozenV13."""
import numpy as np
from inspection_decisions.material_surface_correspondence_v13 import (
    ChartPair as OriginalChartPair,optional_assignment)


def circular_interpolate(prob,theta):
    L=prob.shape[-1];u=np.mod(theta*L/(2*np.pi),L);floor=np.floor(u);lo=floor.astype(int)%L;frac=u-floor
    a=np.take_along_axis(prob,lo[...,None],axis=-1)[...,0];b=np.take_along_axis(prob,((lo+1)%L)[...,None],axis=-1)[...,0]
    return (1-frac)*a+frac*b


class ChartPair(OriginalChartPair):
    def evidence(self,T,oriented):
        base=super().evidence(T,0)
        if not oriented:return base
        frame=np.einsum('nji,jk,mkl->nmil',self.b['basis'],T[:3,:3],self.a['basis']).transpose(1,0,2,3)
        cost=self.cfg['normal_cost_weight']*(1-np.clip(frame[:,:,2,2],-1,1))
        if oriented==2:
            theta=np.arctan2(frame[:,:,1,0],frame[:,:,0,0]);p=circular_interpolate(self.prob,theta)
            cost+=self.cfg['phase_cost_weight']*self.reliability*(-np.log(np.maximum(p,1e-12))/np.log(self.cfg['chart_angles']))
        result=dict(base);result['gain']=base['gain']-cost;result['matched'],result['value']=optional_assignment(result['gain']);return result
