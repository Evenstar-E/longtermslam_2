"""Acquire official smallest ROS1 Khronos sample with progress, cap and hash."""
import hashlib,json,threading,time
from pathlib import Path
import gdown
out=Path("data/khronos_official_v1");out.mkdir(exist_ok=False)
dest=out/"tesse_cd_apartment.bag"
cancel=threading.Event();last=[0,0]
def progress(done,total):
    if total and total>6*1024**3:cancel.set()
    if done>6*1024**3:cancel.set()
    now=time.monotonic()
    if now-last[0]>20:
        last[:]=[now,done]
        (out/"progress.json").write_text(json.dumps({"downloaded_bytes":done,"total_bytes":total,"running":True}),encoding="utf-8")
        print("download MiB",round(done/2**20,1),"of",None if total is None else round(total/2**20,1),flush=True)
start=time.monotonic()
try:
    result=gdown.download(id="1K26TxTEWraBaUf6AhCApwOk4q3CeaNcW",output=str(dest),quiet=True,
             use_cookies=False,timeout=40,retries=0,progress=progress,cancel=cancel)
    if result is None or cancel.is_set():raise RuntimeError("Download incomplete or cap exceeded")
    h=hashlib.sha256()
    with dest.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    report={"complete":True,"path":str(dest),"bytes":dest.stat().st_size,"sha256":h.hexdigest(),
            "official_file_id":"1K26TxTEWraBaUf6AhCApwOk4q3CeaNcW","elapsed_s":time.monotonic()-start,
            "source":"Khronos official dataset GoogleDrive; simulated TESSE apartment with GT pose/semantic input"}
except Exception as e:report={"complete":False,"error":repr(e),"elapsed_s":time.monotonic()-start}
(out/"acquisition.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report),flush=True)

