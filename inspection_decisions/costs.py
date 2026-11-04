"""Return-aware accounting and causal stopping BASELINES, not a novel planner."""
from comparative_evidence.interface import shortest_path


def prefix_cost(events, meta, stage, speed_m_s=1., bandwidth_bytes_s=10_000_000., index_bytes=0):
    if type(stage) is not int or stage < 0 or speed_m_s <= 0 or bandwidth_bytes_s <= 0:
        raise ValueError('Invalid stage or cost assumptions')
    acquired=[e for e in events if e['action']!='return_to_preview']
    # Common preview and H0, followed by stage acquired actions.
    if len(acquired)<2+stage:
        raise ValueError('Missing prefix events')
    prefix=acquired[:2+stage]
    if [(e['epoch'],e['view_id']) for e in prefix[:2]] != [('current',0),('history',0)]:
        raise ValueError('Expected common C0,H0')
    position=0
    for e in prefix:
        if e['epoch']=='current':position=e['view_id']
    travelled=sum(e['travel_m'] for e in prefix)
    reserve=shortest_path(meta['edges'],position,0)[0]
    payload=sum(e['payload_bytes'] for e in prefix)
    acquisition=sum(e['modeled_acquisition_s'] for e in prefix)
    seconds=(travelled+reserve)/speed_m_s+acquisition+(payload+index_bytes)/bandwidth_bytes_s
    return dict(history_views=sum(e['epoch']=='history' for e in prefix),
        additional_current_views=sum(e['epoch']=='current' for e in prefix)-1,
        travelled_before_return_m=travelled,return_m=reserve,total_travel_m=travelled+reserve,
        payload_bytes=payload,index_bytes=index_bytes,modeled_acquisition_s=acquisition,
        modeled_motion_sensor_transfer_s=seconds,inference_s=None,
        timing_limits='Assumed speed/bandwidth; inference per prefix unavailable, not zero. Not measured flight/total latency.')


def report_risk(prediction):
    """Mean declared report loss after support gate; NOT an error probability."""
    assets=prediction['assets']
    return sum(.35 if a['label']<0 else a['decision_risk'] for a in assets)/len(assets)


def threshold_stop(predictions, threshold):
    """Consumes observed prefixes lazily; never requests a later prefix on stop."""
    if threshold<0:raise ValueError('Negative risk threshold')
    last=None
    for stage,p in enumerate(predictions):
        last=stage
        if report_risk(p)<=threshold:return stage
    if last is None:raise ValueError('Empty trace')
    return last
