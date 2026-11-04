"""Extract only planned samples and calibration from remote solid 7z streams."""
import argparse,hashlib,json,time
from pathlib import Path,PurePosixPath
import py7zr
from scripts.index_openloris import RangeSource,RemoteMember,jsonwrite

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--budget-mb",type=int,default=384)
    a=p.parse_args()
    base=Path("data/external_feasibility_v1")
    plan=json.loads((base/"sample_plan.json").read_text())["selected"]
    members=json.loads((base/"index_v1/outer_index.json").read_text())
    info=json.loads((base/"sources/openloris_hf_info.json").read_text())
    url="https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/"+info["sha"]+"/package/office1-1_7-package.tar"
    a.output.mkdir(parents=True,exist_ok=False)
    source=RangeSource(url,10618337280,base/"range_cache",a.budget_mb)
    records=[]
    start=time.perf_counter()
    try:
        for role in ("reference","current"):
            row=plan[role]
            seq="office1-"+str(row["sequence"])
            member=next(m for m in members if m["name"]==seq+".7z")
            targets=[row["rgb"],row["depth"],seq+"/trans_matrix.yaml",seq+"/sensors.yaml"]
            for target in targets:
                parts=PurePosixPath(target)
                if parts.is_absolute() or ".." in parts.parts: raise ValueError("unsafe member")
                dest=(a.output/target).resolve()
                if not dest.is_relative_to(a.output.resolve()): raise ValueError("outside output")
                if dest.exists(): raise FileExistsError(dest)
            stream=RemoteMember(source,member["offset"],member["size"],block=4*1024*1024)
            with py7zr.SevenZipFile(stream,"r") as archive:
                if any(t not in archive.getnames() for t in targets): raise ValueError("missing target")
                # BinaryIO permits short reads. Bound decoder read-ahead after header parsing;
                # otherwise py7zr may feed 1MiB per small PNG despite buffered LZMA input.
                stream.maximum_read=65536
                archive.extract(path=a.output,targets=targets,recursive=False)
            for target in targets:
                path=a.output/target
                records.append({"role":role,"archive_member":member["name"],"path":target,
                    "bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            print("extracted",role,seq,flush=True)
    finally:
        jsonwrite(a.output/"acquisition.json",{"source_url":url,
            "hf_revision":info["sha"],"dataset_license":"CC BY-ND 4.0; local research only; no derivative dataset publication",
            "selection_plan_sha256":hashlib.sha256((base/"sample_plan.json").read_bytes()).hexdigest(),
            "elapsed_s":time.perf_counter()-start,"new_bytes_transferred":source.transferred,
            "range_requests":source.records,"files":records,
            "complete":len(records)==8,
            "integrity_scope":"HTTP Content-Range/length checked, per-range and extracted-file SHA256 saved; selected 7z-file CRC checked by py7zr; full 10.6GB tar SHA not verified."})
if __name__=="__main__": main()

