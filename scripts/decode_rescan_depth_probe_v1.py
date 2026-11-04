"""Bounded reference-codec translation for complete frames in acquired prefixes.

NOT a claim of calibrated Rescan metric depths, intrinsics, or camera poses.
"""
from pathlib import Path
import re,struct
import numpy as np
from PIL import Image
from scripts.run_rigid_identity_bounds_v1 import save,sha

def decode(payload,n):
    bits=np.unpackbits(np.frombuffer(payload,dtype=np.uint8));position=0;output=np.empty(n,np.uint16);index=0;last=0
    def take(k):
        nonlocal position
        if position+k>len(bits):raise ValueError('Truncated compressed payload')
        value=0
        for b in bits[position:position+k]:value=(value<<1)|int(b)
        position+=k;return value
    while index<n:
        prefix=take(2);repeat=1
        if prefix==0:value=last
        elif prefix==2:value=(last-1)&65535
        elif prefix==3:value=(last+1)&65535
        elif take(1)==0:repeat=take(5)+5;value=last
        elif take(1)==0:value=(last+(2 if take(1) else -2))&65535
        else:value=take(11)
        if index+repeat>n:raise ValueError('Decoded run exceeds declared image')
        output[index:index+repeat]=value;index+=repeat;last=value
    if len(bits)-position>7:raise ValueError('Resolution/payload mismatch: more than padding left')
    return output,position

def main():
    source=Path('data/rescan_decoder_source_probe_v1');text=(source/'uplinksimple_shift2depth.h').read_text();table=np.array([int(v) for v in re.findall(r'\d+',text.split('table[] = {')[1].split('};')[0])],np.uint16)
    out=Path('data/rescan_depth_decoded_probe_v1');out.mkdir(exist_ok=False);rows=[]
    for path in sorted(Path('data/rescan_raw_probe_v1').glob('*.prefix.bin')):
        data=path.read_bytes();offset=0
        for frame in range(3):
            size=struct.unpack_from('<I',data,offset)[0];offset+=4
            if size>len(data)-offset:break
            shift,used=decode(data[offset:offset+size],640*480);offset+=size;depth=table[np.minimum(shift,len(table)-1)].copy();depth[depth>=table[-1]]=0;stem=path.name.split('.prefix')[0]+f'_{frame:04d}'
            np.save(out/(stem+'_shift.npy'),shift.reshape(480,640));Image.fromarray(depth.reshape(480,640)).save(out/(stem+'_reference_mm.png'))
            rows.append(dict(stream=path.name,frame=frame,compressed_frame_bytes=size,decoded_pixels=len(shift),consumed_bits=used,padding_bits=size*8-used,positive_reference_depth_fraction=float((depth>0).mean()),reference_depth_range_m=[float(depth[depth>0].min()/1000),float(depth.max()/1000)],input_prefix_sha256=sha(path)))
    save(out/'manifest.json',dict(rows=rows,sources={str(p):sha(p) for p in [Path(__file__)]+list(source.glob('*')) if p.is_file()},scope='First three complete frames each of two stream prefixes. Official ScanNet/Occipital codec; 640x480 consumes complete payloads. Fixed generic SDK shift-to-depth table, NOT verified Rescan device calibration. No intrinsics, camera trajectory, RGB/depth sync or GT frame registration verified. Prefix has no full member CRC.'))
    print(rows)

if __name__=='__main__':main()
