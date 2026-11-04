"""Selective extraction from the authors' public ZIP, with bounded range requests."""
import argparse, hashlib, io, json, struct, time, zipfile, zlib
from pathlib import Path
import requests

URL='https://rescan.cs.princeton.edu/assets/rescan_dataset.zip'
ROOT=Path('data/real_cross_session_v1/rescan')

class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.s=requests.Session(); self.pos=0; self.transferred=0; self.requests=[]
        r=self.s.get(URL,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'},timeout=60)
        r.raise_for_status()
        if r.status_code!=206: raise RuntimeError('Server must support byte ranges')
        self.size=int(r.headers['Content-Range'].split('/')[-1]); self.transferred+=len(r.content)
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        if self.transferred+n>700*1024**2: raise RuntimeError('700 MiB transfer limit')
        start=self.pos; end=start+n-1
        r=self.s.get(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=120)
        r.raise_for_status()
        expected=f'bytes {start}-{end}/{self.size}'
        if r.status_code!=206 or r.headers.get('Content-Range')!=expected or len(r.content)!=n:
            raise RuntimeError(f'Invalid range response: {r.status_code} {r.headers.get("Content-Range")} expected {expected}')
        self.pos+=n; self.transferred+=n; self.requests.append([start,end,n]); return r.content

def main():
    p=argparse.ArgumentParser();p.add_argument('--extract',action='store_true');p.add_argument('--metadata',action='store_true');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        entries=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc':i.CRC} for i in archive.infolist()]
        (ROOT/'inventory.json').write_text(json.dumps({'url':URL,'archive_bytes':remote.size,'entries':entries},indent=2),encoding='utf-8')
        ply=[i for i in archive.infolist() if '/gt_segmentation/' in i.filename and i.filename.endswith('.ply') and '__MACOSX' not in i.filename]
        print('archive_bytes',remote.size,'entries',len(entries),'GT PLY',len(ply),flush=True)
        for i in ply: print(i.filename,i.file_size,i.compress_size,flush=True)
        if a.metadata:
            for i in archive.infolist():
                if i.filename.endswith('.txt'):
                    dest=ROOT/'metadata'/i.filename;dest.parent.mkdir(parents=True,exist_ok=True)
                    dest.write_bytes(archive.read(i));print(i.filename, dest.read_text(encoding='utf-8')[:800],flush=True)
        if a.extract:
            # Lexicographic selection before inference, first three spaces, first two captures each.
            scenes=sorted({i.filename.split('/gt_segmentation/')[0] for i in ply})[:3]
            chosen=[i for scene in scenes for i in sorted([j for j in ply if j.filename.split('/gt_segmentation/')[0]==scene],key=lambda q:q.filename)[:2]]
            records=[]
            for i in chosen:
                dest=ROOT/'raw'/i.filename
                if dest.exists(): raise FileExistsError(dest)
                data=archive.read(i) # ZipFile validates the selected member's CRC.
                dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
                records.append({'member':i.filename,'path':str(dest.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'zip_crc_verified':True})
                (ROOT/'acquisition.partial.json').write_text(json.dumps({'complete':False,'records':records},indent=2),encoding='utf-8')
                print('saved',i.filename,flush=True)
            (ROOT/'acquisition.json').write_text(json.dumps({'complete':True,'url':URL,'selection':'lexicographic first three spaces, first two captures; before inference','records':records,'transferred_bytes':remote.transferred,'ranges':remote.requests,'full_archive_hash_verified':False},indent=2),encoding='utf-8')
    (ROOT/'range_log.json').write_text(json.dumps({'transferred_bytes':remote.transferred,'ranges':remote.requests},indent=2),encoding='utf-8')

if __name__=='__main__':main()
