"""Inspect bounded HTTP range of official uncompressed tar; never fetch whole archive."""
import json,hashlib,urllib.request,tarfile
from pathlib import Path
base=Path("data/external_feasibility_v1")
info=json.loads((base/"sources/openloris_hf_info.json").read_text(encoding="utf-8"))
url="https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/"+info["sha"]+"/package/office1-1_7-package.tar"
req=urllib.request.Request(url,headers={"User-Agent":"visual-reinspection-research/0.1","Range":"bytes=0-65535"})
with urllib.request.urlopen(req,timeout=40) as r:
    headers=dict(r.headers)
    if r.status!=206: raise RuntimeError("Range not honored: "+str(r.status))
    body=r.read(65537)
    if len(body)!=65536: raise RuntimeError("unexpected range length")
dest=base/"range_probe"
dest.mkdir(exist_ok=False)
(dest/"first_64k.bin").write_bytes(body)
record={"url":url,"status":206,"headers":headers,"bytes":len(body),"sha256":hashlib.sha256(body).hexdigest(),"members":[]}
offset=0
while offset+512<=len(body):
    h=body[offset:offset+512]
    if not h.strip(b"\0"): break
    t=tarfile.TarInfo.frombuf(h,"utf-8","replace")
    row={"offset":offset,"name":t.name,"size":t.size,"type":repr(t.type),"data_offset":offset+512}
    record["members"].append(row)
    print(row,flush=True)
    offset+=512+((t.size+511)//512)*512
(dest/"probe.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print("headers",headers,flush=True)

