"""Observed-only full paths or actual early-stop reruns; evaluator is external."""
import time
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_terminal
from inspection_decisions.evidence_graph import build_graph,legal_actions
from inspection_decisions.component_set_v1 import representation,decide
from inspection_decisions.report_risk_stopping_v1 import should_stop


def run_path(root,policy,action_model,pairs,budget,snapshot,stopper=None):
    ep=Episode(root,*budget);h=[ep.act('retrieve_history',0)];q=[ep.preview()]
    meta=ep.metadata();states=[];backend_s=0.;previous=None;initial=None
    while True:
        begin=time.perf_counter()
        fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
        prediction=common_terminal(fit);backend_s+=time.perf_counter()-begin
        graph=build_graph(fit,h,q,meta,ep.state(),prediction)
        rep=representation(graph,pairs);_,detail=decide(rep,action_model)
        if len(rep['options'])>1:
            rank=1+int(np.argmax(detail['scores'][1:]));coverage=1+int(np.argmax(rep['coverage'][1:]))
            choice=rank if policy=='rank' else coverage;option=rep['options'][choice]
            best_rank=float(detail['scores'][rank]);best_coverage=float(rep['coverage'][coverage])
        else:option=();best_rank=0.;best_coverage=0.
        state=dict(stage=len(states),prediction=prediction,best_rank_score=best_rank,
                   best_coverage_score=best_coverage,selected_option=option)
        if initial is None:initial=prediction
        if stopper is not None:
            stop,decision=should_stop(stopper['method'],state,previous,initial,
                                     stopper['models'],stopper['setting'],stopper['price'])
            state['stop_decision']=decision
            if stop:option=()
        snapshot(graph,state);states.append(state)
        if not option:break
        action=option[0];assert action in legal_actions(ep.state())
        obs=ep.act('retrieve_history' if action[0]=='history' else 'observe_current',action[1])
        (h if action[0]=='history' else q).append(obs);previous=prediction
        assert len(states)<=5
    cost=ep.finish()
    return dict(states=states,events=ep.state()['events'],cost=cost,metadata=meta,
                backend_fits=len(states),backend_s=backend_s)
