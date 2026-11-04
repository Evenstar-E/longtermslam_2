"""Download a predefined public calibrated motion sequence; no inference or GT scoring."""
import hashlib,json,time,requests
from pathlib import Path

def main():
    root=Path('data/ycbineoat_motion_v1');root.mkdir(exist_ok=True)
    url='https://archive.cs.rutgers.edu/archive/a/2020/pracsys/Bowen/iros2020/YCBInEOAT/cracker_box_reorient.tar.gz'
    target=root/'cracker_box_reorient.tar.gz';partial=root/'cracker_box_reorient.tar.gz.partial'
    if target.exists():raise FileExistsError('Never overwrite acquired data')
    start=time.perf_counter();digest=hashlib.sha256();total=0;last=0
    with requests.get(url,stream=True,timeout=(30,90)) as r:
        r.raise_for_status();expected=int(r.headers.get('Content-Length',0))
        if expected not in [0,1477933744]:raise ValueError(('Unexpected official archive size',expected))
        with partial.open('xb') as f:
            for block in r.iter_content(2**20):
                f.write(block);digest.update(block);total+=len(block)
                if time.perf_counter()-last>20:
                    record={'downloaded_bytes':total,'expected_bytes':expected,'elapsed_s':time.perf_counter()-start}
                    (root/'download_monitor.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print('DOWNLOAD',record,flush=True);last=time.perf_counter()
    if expected and total!=expected:raise ValueError('Incomplete download')
    partial.rename(target)
    (root/'acquisition.json').write_text(json.dumps({'url':url,'official_entry':'https://github.com/wenbowen123/iros20-6d-pose-tracking','selection':'Smallest official archive by directory size, chosen before downloading or viewing frames/poses. Not performance-selected.','bytes':total,'sha256':digest.hexdigest(),'elapsed_s':time.perf_counter()-start,'scope':'Real robot manipulation video, NOT cross-day multi-asset reinspection. No model run or pose accuracy established.'},indent=2),encoding='utf-8')
    print('DOWNLOADED',total,digest.hexdigest(),flush=True)

if __name__=='__main__':main()
