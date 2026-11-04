"""Export privileged-mask, anonymized public inputs; keep persistent IDs separate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read_ply(path):
    data=Path(path).read_bytes();header,payload=data.split(b'end_header\n',1)
    lines=header.decode('ascii').splitlines()
    if lines[1]!='format binary_little_endian 1.0':raise ValueError('PLY format')
    types={'float':'<f4','int':'<i4','uchar':'u1'}
    fields=[(s.split()[2],types[s.split()[1]]) for s in lines if s.startswith('property ')]
    n=int(next(s.split()[2] for s in lines if s.startswith('element vertex ')))
    dtype=np.dtype(fields)
    values=np.frombuffer(payload,dtype=dtype,count=n)
    if payload[n*dtype.itemsize:].strip():raise ValueError('Unexpected bytes after PLY vertices')
    if len(values)!=n:raise ValueError('PLY vertex count')
    return values

def main():
    cfg=json.loads(Path('configs/real_cross_session_v1.json').read_text(encoding='utf-8'))
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='prepared_v2');args=parser.parse_args()
    root=Path('data/real_cross_session_v1');out=root/args.output;out.mkdir(exist_ok=False)
    acquisition=json.loads((root/'rescan/acquisition.json').read_text(encoding='utf-8'))
    for r in acquisition['records']:
        assert hashlib.sha256((root/'rescan'/r['path']).read_bytes()).hexdigest()==r['sha256']
    truth={};public={};inventory=[]
    for si,scene in enumerate(cfg['spaces']):
        truth[scene]={};public[scene]={}
        for capture in cfg['captures']:
            path=root/'rescan/raw'/scene/'gt_segmentation'/f'{scene}_{capture}.ply'
            p=read_ply(path);ids=sorted(np.unique(p['instance_idx']).tolist())
            rng=np.random.default_rng(cfg['seed']+si*10+capture);rng.shuffle(ids)
            objects=[];mapping={}
            for identifier in ids:
                mask=p['instance_idx']==identifier;class_ids,counts=np.unique(p['class_idx'][mask],return_counts=True)
                cls=int(class_ids[np.argmax(counts)]);count=int(mask.sum())
                if cls in cfg['excluded_classes']:continue
                token=f'o{len(objects):03d}'
                xyz=np.column_stack([p[n][mask] for n in ['x','y','z']]).astype(float)
                color=np.column_stack([p[n][mask] for n in ['red','green','blue']])
                dest=out/'public'/scene/str(capture);dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/(token+'.npz'),xyz=xyz,rgb=color)
                objects.append({'token':token,'semantic_class':cls,'raw_points':count,'eligible':count>=cfg['minimum_raw_points'],
                    'path':str((dest/(token+'.npz')).relative_to(out))})
                mapping[token]=identifier
                inventory.append(dict(scene=scene,capture=capture,token=token,instance=identifier,cls=cls,points=count))
            public[scene][str(capture)]=objects;truth[scene][str(capture)]=mapping
    (out/'public.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    (out/'evaluation').mkdir()
    (out/'evaluation/identity.json').write_text(json.dumps(truth,indent=2),encoding='utf-8')
    (out/'evaluation/inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps([{'scene':s,'captures':{k:len(v) for k,v in d.items()}} for s,d in public.items()]))

if __name__=='__main__':main()
