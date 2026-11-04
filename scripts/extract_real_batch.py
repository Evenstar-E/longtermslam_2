"""Extract the frozen batch's selected original RGB-D files from official archive."""
import argparse,hashlib,json,shutil,time
from pathlib import Path,PurePosixPath
import py7zr
from scripts.index_openloris import RangeSource,RemoteMember
base=Path("data/external_feasibility_v1")
p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("data/real_batch_v1/raw"));p.add_argument("--reuse",type=Path)
args=p.parse_args();out=args.output;out.mkdir(exist_ok=False)
reuse_records={}
if args.reuse:
    old=json.loads((args.reuse/"acquisition.json").read_text())
    reuse_records={x["path"]:x for x in old["records"]}
plan_path=Path("data/real_batch_v1/plan.json")
plan=json.loads(plan_path.read_text())
info=json.loads((base/"sources/openloris_hf_info.json").read_text())
url="https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/"+info["sha"]+"/package/office1-1_7-package.tar"
members=json.loads((base/"index_v1/outer_index.json").read_text())
source=RangeSource(url,10618337280,base/"range_cache",2048)
records=[];started=time.monotonic()
try:
    for seq in plan["sequences"]:
        name="office1-"+str(seq)
        member=next(m for m in members if m["name"]==name+".7z")
        frames=[f for f in plan["unique_frames"] if f["sequence"]==seq]
        targets=sorted(set([f[k] for f in frames for k in ("rgb","depth")]+[name+"/sensors.yaml",name+"/trans_matrix.yaml"]))
        for target in targets:
            p=PurePosixPath(target)
            if p.is_absolute() or ".." in p.parts:raise ValueError("Unsafe member")
            if not (out/target).resolve().is_relative_to(out.resolve()):raise ValueError("Unsafe output")
        if all(t in reuse_records for t in targets):
            for t in targets:
                prior=args.reuse/t
                if hashlib.sha256(prior.read_bytes()).hexdigest()!=reuse_records[t]["sha256"]:raise RuntimeError("Reuse hash mismatch")
                (out/t).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(prior,out/t)
        else:
            stream=RemoteMember(source,member["offset"],member["size"],block=4*1024*1024)
            with py7zr.SevenZipFile(stream,"r") as z:
                if any(x not in z.getnames() for x in targets):raise ValueError("Missing selected member")
                stream.maximum_read=65536
                z.extract(path=out,targets=targets,recursive=False)
        for target in targets:
            p=out/target;records.append({"path":target,"bytes":p.stat().st_size,
                  "sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        print("SEQUENCE EXTRACTED",seq,"frames",len(frames),"files",len(targets),flush=True)
finally:
    (out/"acquisition.json").write_text(json.dumps({"complete":len(records)==len(plan["unique_frames"])*2+6,
       "selected_file_count":len(records),"records":records,"source_url":url,
       "plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),
       "new_range_bytes":source.transferred,"range_requests":source.records,
       "elapsed_s":time.monotonic()-started,"full_tar_hash_checked":False,
       "integrity":"HTTP Range checked; selected7z CRC and local SHA256; original files unchanged."},indent=2),encoding="utf-8")

