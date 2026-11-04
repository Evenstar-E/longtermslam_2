"""Numerically identical caching only; no changes to the frozen V14 model."""
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree
from inspection_decisions.material_continuous_surface_correspondence_v14 import (
    DensePair as OriginalPair,ContinuousTriple as OriginalTriple,
    fit_pose,spectral_proposals,PAIRS)


class DensePair(OriginalPair):
    def __init__(self,a,b,fa,fb,ia,ib,dense_a,dense_b,cfg):
        self.a=a;self.b=b;self.fa=fa;self.fb=fb;self.ia=ia;self.ib=ib;self.da=dense_a;self.db=dense_b;self.cfg=cfg
        for record in [dense_a,dense_b]:
            if '_tree' not in record:record['_tree']=cKDTree(record['xyz'])
        self.ta=dense_a['_tree'];self.tb=dense_b['_tree']


class ContinuousTriple(OriginalTriple):
    def __init__(self,pairs,cfg):super().__init__(pairs,cfg);self.values=OrderedDict();self.states=OrderedDict()
    def evaluate(self,T,return_state=False):
        key=np.asarray(T,dtype=np.float64).tobytes()
        if return_state and key in self.states:self.states.move_to_end(key);return self.values[key].copy(),self.states[key]
        if not return_state and key in self.values:self.values.move_to_end(key);return self.values[key].copy()
        value,state=super().evaluate(T,True);self.values[key]=value.copy()
        if return_state:
            self.states[key]=state
            while len(self.states)>2:self.states.popitem(last=False)
        while len(self.values)>512:
            old,_=self.values.popitem(last=False);self.states.pop(old,None)
        return (value,state) if return_state else value
