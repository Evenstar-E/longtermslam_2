"""Bounded public-source acquisition for external data/baseline feasibility audit."""
import argparse,hashlib,json,time,urllib.request
from pathlib import Path

SOURCES={
"khronos_repo":"https://api.github.com/repos/MIT-SPARK/Khronos",
"khronos_commits":"https://api.github.com/repos/MIT-SPARK/Khronos/commits?per_page=1",
"khronos_branches":"https://api.github.com/repos/MIT-SPARK/Khronos/branches?per_page=100",
"khronos_tree":"https://api.github.com/repos/MIT-SPARK/Khronos/git/trees/main?recursive=1",
"khronos_readme":"https://raw.githubusercontent.com/MIT-SPARK/Khronos/main/README.md",
"khronos_license":"https://raw.githubusercontent.com/MIT-SPARK/Khronos/main/LICENSE",
"torwic_repo":"https://api.github.com/repos/Viky397/TorWICDataset",
"torwic_tree":"https://api.github.com/repos/Viky397/TorWICDataset/git/trees/main?recursive=1",
"torwic_readme":"https://raw.githubusercontent.com/Viky397/TorWICDataset/main/README.md",
"openloris_download":"https://raw.githubusercontent.com/lifelong-robotic-vision/OpenLORIS-Scene/master/download.md",
"openloris_hf_info":"https://huggingface.co/api/datasets/shixuesong/openloris-scene",
"openloris_hf_tree":"https://huggingface.co/api/datasets/shixuesong/openloris-scene/tree/main?recursive=true&limit=1000",
}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    log=[]
    for key,url in SOURCES.items():
        start=time.perf_counter()
        row={"id":key,"url":url}
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"visual-reinspection-research/0.1"})
            with urllib.request.urlopen(req,timeout=25) as response:
                body=response.read(8*1024*1024+1)
                if len(body)>8*1024*1024: raise ValueError("metadata size cap exceeded")
                row.update(status=response.status,final_url=response.url)
            suffix=".json" if key.endswith(("repo","commits","branches","tree","info")) else ".txt"
            dest=a.output/(key+suffix)
            with dest.open("xb") as f: f.write(body)
            row.update(path=dest.name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
            print(key,"OK",len(body),flush=True)
        except Exception as exc:
            row["error"]=type(exc).__name__+": "+str(exc)
            print(key,row["error"],flush=True)
        row["elapsed_s"]=time.perf_counter()-start
        log.append(row)
    with (a.output/"acquisition.json").open("x",encoding="utf-8") as f:
        json.dump({"purpose":"public primary-source snapshot, not method reproduction",
                   "records":log},f,ensure_ascii=False,indent=2)
if __name__=="__main__": main()

