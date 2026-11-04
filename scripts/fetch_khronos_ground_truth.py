"""Acquire only six official label/geometry files, with bounded downloads and provenance."""
import concurrent.futures,hashlib,json,threading,time
from pathlib import Path
import gdown
base=Path('data/labeled_temporal_v1');base.mkdir(exist_ok=False)
cfg=Path('configs/labeled_temporal_v1.json')
(base/'frozen_config.json').write_bytes(cfg.read_bytes())
index=json.loads(Path('results/khronos_reproduction_v1/official_drive_index_v2.json').read_text())
selected=[]
for row in index['rows']:
    parts=row['path'].replace('\\','/').split('/')
    if len(parts)==4 and parts[:2]==['tesse_cd','ground_truth'] and parts[2] in ['apartment','office'] and parts[3] in ['gt_changes.csv','gt_dsg_consolidated.json','gt_background.ply']:
        selected.append((row,parts[2],parts[3]))
assert len(selected)==6
started=time.monotonic()
def fetch(item):
    row,scene,name=item;dest=base/'ground_truth'/scene/name;dest.parent.mkdir(parents=True,exist_ok=True)
    cancel=threading.Event();last=[0.]
    def progress(done,total):
        if done>1024**3 or (total is not None and total>1024**3):cancel.set()
        if time.monotonic()-last[0]>25:
            last[0]=time.monotonic();print(scene,name,'MiB',round(done/2**20,2),flush=True)
    record={'id':row['id'],'official_path':row['path'],'path':str(dest),'scene':scene,'complete':False}
    try:
        answer=gdown.download(id=row['id'],output=str(dest),quiet=True,use_cookies=False,timeout=45,retries=0,progress=progress,cancel=cancel)
        if not answer or cancel.is_set():raise RuntimeError('Incomplete or >1GiB file cap')
        h=hashlib.sha256()
        with dest.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        record.update(complete=True,bytes=dest.stat().st_size,sha256=h.hexdigest())
    except Exception as e:record['error']=repr(e)
    (dest.parent/(name+'.acquisition.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(scene,name,'complete',record['complete'],flush=True)
    return record
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,selected))
report={'complete':all(r['complete'] for r in records),'records':records,'elapsed_s':time.monotonic()-started,
 'protocol_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'source':'Official Khronos Drive listing; synthetic TESSE labels',
 'scope':'Two independent simulated layouts; no real-world data or algorithm performance implied'}
(base/'acquisition.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'complete':report['complete'],'files':len(records),'elapsed_s':report['elapsed_s']}),flush=True)
