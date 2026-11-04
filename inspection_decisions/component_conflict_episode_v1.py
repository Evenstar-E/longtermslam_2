"""Instrument frozen backend without changing its estimates or acquisition."""
from unittest.mock import patch
from comparative_evidence.uncertainty_v3 import attach_geometry_information
from inspection_decisions import report_risk_episode_v1 as episode
from inspection_decisions.component_conflict_v1 import capture,decision

def run(root,policy,action_model,pairs,budget,snapshot,stopper=None):
    latest={}
    def attach(fit,h,q,meta):
        result=attach_geometry_information(fit,h,q,meta);latest['evidence']=capture(result,h,q,meta);return result
    def snap(graph,state):state['evidence']=latest['evidence'];snapshot(graph,state)
    def stop(method,state,previous,initial,models,setting,price):
        state['evidence']=latest['evidence'];return decision(method,state,previous,initial,models,setting)
    with patch.object(episode,'attach_geometry_information',attach),patch.object(episode,'should_stop',stop):
        return episode.run_path(root,policy,action_model,pairs,budget,snap,stopper)
