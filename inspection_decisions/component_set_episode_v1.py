"""Real acquisition/refit/replanning, with evaluation left to the runner."""
import time
import numpy as np
from comparative_evidence.interface import Episode
from comparative_evidence.geometry_modes_v2 import estimate_modes
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from comparative_evidence.selection_v5 import common_terminal
from inspection_decisions.evidence_graph import build_graph,legal_actions
from inspection_decisions.component_set_v1 import representation,decide


def run_episode(root,method,models,pairs,budget,snapshot):
    ep=Episode(root,*budget);h=[ep.act('retrieve_history',0)];q=[ep.preview()];meta=ep.metadata()
    trajectory=[];backend_s=0.;actor_s=0.;fit_count=0;pending=[]
    while True:
        start=time.perf_counter();fit=attach_geometry_information(estimate_modes(h,q,meta),h,q,meta)
        prediction=common_terminal(fit);backend_s+=time.perf_counter()-start;fit_count+=1
        start=time.perf_counter();graph=build_graph(fit,h,q,meta,ep.state(),prediction)
        rep=representation(graph,pairs,pooled=method=='pooled_rank')
        if method=='coverage_commit' and trajectory:
            if pending:
                option=(pending.pop(0),);choice=rep['options'].index(option)
            else:choice=0
            details={'committed_initial_plan':True}
        elif method.startswith('coverage'):
            choice=int(np.argmax(rep['coverage']));details={'coverage_scores':rep['coverage'].tolist()}
            if method=='coverage_commit':pending=list(rep['options'][choice][1:])
        else:choice,details=decide(rep,models[method])
        option=rep['options'][choice];actor_s+=time.perf_counter()-start
        record=dict(stage=len(trajectory),selected_option=option,details=details,
                    history_read=ep.state()['history_read'],current_observed=ep.state()['current_observed'])
        # Persist decision and observed graph BEFORE acquiring another image.
        snapshot(graph,prediction,record);trajectory.append(record)
        if not option:break
        action=option[0];assert action in legal_actions(ep.state())
        obs=ep.act('retrieve_history' if action[0]=='history' else 'observe_current',action[1])
        (h if action[0]=='history' else q).append(obs)
        assert len(trajectory)<=sum(budget)-1
    cost=ep.finish()
    return dict(prediction=prediction,cost=cost,trajectory=trajectory,events=ep.state()['events'],
                timings=dict(backend_s=backend_s,actor_including_graph_s=actor_s),backend_fits=fit_count)
