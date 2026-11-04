"""Author Rescan first temporal update; current IDs/classes scrubbed before inference."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scripts.prepare_real_temporal_v1 import read_ply

def linux(p):
    s=str(Path(p).resolve());return '/mnt/'+s[0].lower()+s[2:].replace('\\','/')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/rescan_author_v1/inference');parser.add_argument('--binaries',default='third_party/Rescan/bin');parser.add_argument('--spaces',nargs='+',default=['scene_a','scene_b','scene_c']);parser.add_argument('--stage-timeout',type=int,default=240);args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=False)
    raw=Path('data/real_cross_session_v1/rescan/raw');records=[]
    bins=Path(args.binaries)
    contract={'method':'ICCV2019 Rescan author implementation, unchanged algorithms',
        'input':'historical full scan+GT initial labels, current full colored scan with ALL instance IDs -1 and classes 0',
        'task':'first temporal update: seg2rsdb -> pose_proposal -> segment_transfer',
        'fusion':'Post-output Poisson reconstruction omitted: cannot affect this first update; no subsequent scan evaluated',
        'comparison':'Different front-end contract from privileged-current-mask module table. No fair end-to-end ranking claimed.',
        'binaries':str(bins),
        'timeout_per_stage_s':args.stage_timeout,'parameters':'author defaults; paths only',
        'source_commit':subprocess.check_output(['git','-C','third_party/Rescan','rev-parse','HEAD'],text=True).strip()}
    (out/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    for scene in args.spaces:
        dest=out/scene;dest.mkdir()
        hist=dest/'history.ply';cur=dest/'current.ply'
        shutil.copyfile(raw/scene/'gt_segmentation'/f'{scene}_0.ply',hist)
        original=raw/scene/'gt_segmentation'/f'{scene}_1.ply';content=original.read_bytes();header,_=content.split(b'end_header\n',1)
        points=read_ply(original).copy();points['instance_idx']=-1;points['class_idx']=0
        cur.write_bytes(header+b'end_header\n'+points.tobytes())
        assert np.all(read_ply(cur)['instance_idx']==-1) and np.all(read_ply(cur)['class_idx']==0)
        cls=dest/'classes.txt';shutil.copyfile('data/real_cross_session_v1/rescan/metadata/nyu40_classes.txt',cls)
        initial=dest/'history.rsdb';proposals=dest/'proposals.rsdb';final=dest/'prediction.rsdb'
        stages=[('seg2rsdb',[hist,cls,initial,'-v']),('pose_proposal',[initial,cur,proposals,'-v']),('segment_transfer',[proposals,'-o',final,'-v'])]
        for name,stage_args in stages:
            command=['wsl','-d','Ubuntu-20.04','--exec','env','OMP_NUM_THREADS=1','timeout','--signal=TERM',str(args.stage_timeout),linux(bins/name)]+[linux(a) if isinstance(a,Path) else a for a in stage_args]
            start=time.perf_counter()
            with (dest/(name+'.log')).open('xb') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.stage_timeout+30)
            record=dict(scene=scene,stage=name,returncode=r.returncode,elapsed_s=time.perf_counter()-start)
            records.append(record);print(record,flush=True)
            (out/'monitor.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            if r.returncode!=0:break
        if final.exists():
            files=[p for p in dest.rglob('*') if p.is_file()]
            (dest/'prediction_seal.json').write_text(json.dumps({'files_sha256':{str(p.relative_to(dest)):sha(p) for p in files}},indent=2),encoding='utf-8')
    (out/'run.json').write_text(json.dumps({'records':records,'contract':contract},indent=2),encoding='utf-8')

if __name__=='__main__':main()
