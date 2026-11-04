"""Fixed acquisition measure: prevent pose-conditioned support-denominator escape.

This is a normalization repair/control, not a novel matching solver. The gain,
unknown semantics, capacity, material partition and SE(3) solver stay frozen.
"""
from inspection_decisions.material_partial_track_transport_v7 import PairTransport

class FixedMeasureTransport(PairTransport):
    def evidence(self,T):
        evidence=super().evidence(T)
        evidence['visible_potential']=evidence['potential']
        evidence['potential']=float(len(self.a['xyz'])+len(self.b['xyz']))
        return evidence
