"""Verified HTTP byte ranges of the same predefined official public archive."""
import argparse,concurrent.futures,hashlib,json,math,shutil,time,requests
from pathlib import Path

URL='https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/cracker_box_reorient.tar.gz'
SIZE=1477933744
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    root=Path('data/ycbineoat_motion_v1_parallel');root.mkdir(exist_ok=True)
    r=requests.get(URL,headers={'Range':'bytes=0-0'},timeout=(30,60));r.raise_for_status()
    assert r.status_code==206 and r.headers['Content-Range']==f'bytes 0-0/{SIZE}' and len(r.content)==1
    validator=r.headers.get('ETag') or r.headers.get('Last-Modified');save(root/'range_probe.json',{'url':URL,'status':r.status_code,'headers':dict(r.headers),'validator':validator})
    print('RANGE_AVAILABLE',validator,flush=True)
    if args.probe:return
    target=root/'cracker_box_reorient.tar.gz';assert not target.exists()
    prefix=Path('data/ycbineoat_motion_v1/cracker_box_reorient.tar.gz.partial');prefix_size=prefix.stat().st_size
    # Parent stream must be stopped before calling download. Size is checked
    # again after chunk transfer so concurrent mutation cannot go unnoticed.
    parts=root/'parts';parts.mkdir(exist_ok=False);start=time.perf_counter();chunk_size=16*2**20
    spans=[(k,a,min(a+chunk_size,SIZE)-1) for k,a in enumerate(range(prefix_size,SIZE,chunk_size))]
    def fetch(span):
        k,a,b=span;headers={'Range':f'bytes={a}-{b}'}
        if validator:headers['If-Range']=validator
        h=hashlib.sha256();n=0;p=parts/f'{k:04d}.part'
        with requests.get(URL,headers=headers,stream=True,timeout=(30,90)) as response:
            response.raise_for_status();assert response.status_code==206
            assert response.headers['Content-Range']==f'bytes {a}-{b}/{SIZE}'
            if validator:assert (response.headers.get('ETag') or response.headers.get('Last-Modified'))==validator
            with p.open('xb') as f:
                for block in response.iter_content(2**20):f.write(block);h.update(block);n+=len(block)
        assert n==b-a+1
        return {'index':k,'start':a,'end':b,'bytes':n,'sha256':h.hexdigest(),'path':str(p)}
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch,span) for span in spans]):
            records.append(future.result());done=prefix_size+sum(v['bytes'] for v in records)
            save(root/'monitor.json',{'completed_bytes':done,'expected_bytes':SIZE,'completed_parts':len(records),'total_parts':len(spans),'elapsed_s':time.perf_counter()-start})
            print('RANGE_PROGRESS',done,SIZE,len(records),round(time.perf_counter()-start,1),flush=True)
    assert prefix.stat().st_size==prefix_size
    digest=hashlib.sha256();total=0
    with target.open('xb') as f:
        for source in [prefix]+[Path(v['path']) for v in sorted(records,key=lambda x:x['index'])]:
            with source.open('rb') as ff:
                while block:=ff.read(4*2**20):f.write(block);digest.update(block);total+=len(block)
    assert total==SIZE
    save(root/'acquisition.json',{'url':URL,'official_entry':'https://github.com/wenbowen123/iros20-6d-pose-tracking','bytes':total,'sha256':digest.hexdigest(),'validator':validator,'prefix_reused':str(prefix),'prefix_bytes':prefix_size,'ranges':sorted(records,key=lambda x:x['index']),'elapsed_s':time.perf_counter()-start,'selection':'Smallest official archive, chosen before frame/pose inspection','scope':'Real manipulation video, not cross-day multi-asset reinspection; no motion-accuracy result yet'})
    print('ARCHIVE_COMPLETE',total,digest.hexdigest(),flush=True)

if __name__=='__main__':main()
