"""Range-index official OpenLORIS tar/7z archives without full downloads."""
import argparse,hashlib,io,json,tarfile,time,urllib.request
from pathlib import Path
import py7zr


class RangeSource:
    def __init__(self,url,total,cache,budget_mb):
        self.url,self.total,self.cache=url,total,cache
        self.budget=budget_mb*1024*1024
        self.transferred=0
        self.records=[]
        cache.mkdir(parents=True,exist_ok=True)

    def get(self,start,size):
        end=start+size-1
        if start<0 or end>=self.total: raise ValueError("range out of bounds")
        path=self.cache/f"{start}_{size}.bin"
        if path.exists(): return path.read_bytes()
        if self.transferred+size>self.budget: raise RuntimeError("bounded transfer budget exceeded")
        req=urllib.request.Request(self.url+f"?range_start={start}&range_size={size}",
            headers={"Range":f"bytes={start}-{end}","User-Agent":"visual-reinspection-research/0.1"})
        begin=time.perf_counter()
        with urllib.request.urlopen(req,timeout=45) as r:
            if r.status!=206: raise RuntimeError("Server ignored range")
            if r.headers.get("Content-Range")!=f"bytes {start}-{end}/{self.total}":
                raise RuntimeError("Content-Range mismatch: "+str(r.headers.get("Content-Range")))
            data=r.read(size+1)
            if len(data)!=size: raise RuntimeError("Range length mismatch")
        path.write_bytes(data)
        self.transferred+=size
        self.records.append({"offset":start,"bytes":size,"sha256":hashlib.sha256(data).hexdigest(),
                             "elapsed_s":time.perf_counter()-begin})
        print("range",start,size,"transferred MiB",round(self.transferred/2**20,2),flush=True)
        return data


class RemoteMember(io.RawIOBase):
    def __init__(self,source,offset,size,block=262144):
        self.source,self.offset,self.length,self.block=source,offset,size,block
        self.position=0
        self.maximum_read=None
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if position<0: raise ValueError("negative seek")
        self.position=position
        return position
    def read(self,size=-1):
        if size<0: size=self.length-self.position
        size=min(size,max(0,self.length-self.position))
        if self.maximum_read is not None:
            size=min(size,self.maximum_read)
        pieces=[]
        while size:
            block_start=self.position//self.block*self.block
            block_size=min(self.block,self.length-block_start)
            data=self.source.get(self.offset+block_start,block_size)
            local=self.position-block_start
            take=min(size,len(data)-local)
            pieces.append(data[local:local+take])
            self.position+=take
            size-=take
        return b"".join(pieces)


def jsonwrite(path,data):
    with path.open("x",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sequences",nargs="+",default=["office1-1.7z","office1-2.7z"])
    p.add_argument("--budget-mb",type=int,default=32)
    a=p.parse_args()
    base=Path("data/external_feasibility_v1")
    info=json.loads((base/"sources/openloris_hf_info.json").read_text(encoding="utf-8"))
    tree=json.loads((base/"sources/openloris_hf_tree.json").read_text(encoding="utf-8"))
    path="package/office1-1_7-package.tar"
    size=next(x["size"] for x in tree if x["path"]==path)
    url="https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/"+info["sha"]+"/"+path
    a.output.mkdir(parents=True,exist_ok=False)
    source=RangeSource(url,size,base/"range_cache",a.budget_mb)
    members=[]
    offset=0
    while offset+512<=size:
        raw=source.get(offset,512)
        if not raw.strip(b"\0"): break
        entry=tarfile.TarInfo.frombuf(raw,"utf-8","replace")
        if not entry.isfile(): raise RuntimeError("Unsupported tar extension: "+entry.name)
        members.append({"name":entry.name,"offset":offset+512,"size":entry.size})
        offset+=512+((entry.size+511)//512)*512
    jsonwrite(a.output/"outer_index.json",members)
    print("outer members",members,flush=True)
    for member in members:
        if member["name"] not in a.sequences: continue
        stream=RemoteMember(source,member["offset"],member["size"])
        with py7zr.SevenZipFile(stream,"r") as z:
            files=[{"name":f.filename,"uncompressed":f.uncompressed,"compressed":f.compressed,
                    "is_directory":f.is_directory} for f in z.list()]
            index={"tar_member":member,"solid":z._is_solid(),
                   "compression_methods":z._get_method_names(),
                   "blocks":len(z.header.main_streams.unpackinfo.folders),"files":files}
        jsonwrite(a.output/(member["name"]+".json"),index)
        print(member["name"],"solid",index["solid"],"blocks",index["blocks"],"files",len(files),flush=True)
        print("first files",files[:12],flush=True)
    jsonwrite(a.output/"ranges.json",{"url":url,"total":size,"new_bytes_transferred":source.transferred,
                                    "requests":source.records})
if __name__=="__main__": main()

