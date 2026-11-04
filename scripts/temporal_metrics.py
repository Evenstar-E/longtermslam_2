"""Transparent postprocessing; undefined denominators remain missing, duplicate trials never multiply n."""
import math

def deduplicate(rows,keys=('Name','Query')):
    unique={};duplicates=0
    for row in rows:
        key=tuple(row[k] for k in keys)
        if key in unique:
            if row!=unique[key]:raise ValueError('Conflicting records for same map/query: '+str(key))
            duplicates+=1
        else:unique[key]=row
    return list(unique.values()),duplicates

def prf(tp,fp,fn):
    if min(tp,fp,fn)<0:raise ValueError('Negative count')
    return {'tp':tp,'fp':fp,'fn':fn,'precision':None if tp+fp==0 else tp/(tp+fp),
       'recall':None if tp+fn==0 else tp/(tp+fn),
       'count_f1':None if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn)}

def derive(row):
    n=lambda k:int(row[k])
    tp=n('AppearedTP')+n('DisappearedTP');fp=n('AppearedFP')+n('DisappearedFP');fn=n('AppearedFN')+n('DisappearedFN')
    hp=n('AppearedHallucinatedP')+n('DisappearedHallucinatedP')
    missed=n('AppearedMissedP')+n('DisappearedMissedP')
    return {'name':row['Name'],'query_ns':int(row['Query']),
        'object_presence':prf(n('NumObjDetected'),n('NumObjHallucinated'),n('NumObjMissed')),
        'change_conditional':prf(tp,fp,fn),'change_inclusive':prf(tp,fp+hp,fn+missed),
        'unassociated_hallucinated_changes':hp,'unassociated_missed_changes':missed}
