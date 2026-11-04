"""Bounded official raw-stream prefix inspection, without overwriting old data/seals."""
import struct,zipfile,zlib
from pathlib import Path
from scripts.fetch_rescan_subset_v1 import RemoteZip,URL
from scripts.run_rigid_identity_bounds_v1 import save,sha

def main():
    out=Path('data/rescan_raw_probe_v1');out.mkdir(exist_ok=False);remote=RemoteZip();records=[]
    with zipfile.ZipFile(remote) as archive:
        for name in ['scene_c/depth/scene_c_0.depth','scene_c/depth/scene_c_1.depth']:
            info=archive.getinfo(name);remote.seek(info.header_offset);header=remote.read(30);assert header[:4]==b'PK\x03\x04';fields=struct.unpack('<4s5H3I2H',header);n,e=fields[-2:];start=info.header_offset+30+n+e;remote.seek(start);raw=remote.read(min(info.compress_size,2**20));prefix=zlib.decompressobj(-15).decompress(raw,2**20) if info.compress_type==8 else raw
            target=out/Path(name).name;target=target.with_suffix('.prefix.bin');target.write_bytes(prefix)
            records.append({'member':name,'uncompressed_full_bytes':info.file_size,'compressed_full_bytes':info.compress_size,'prefix_bytes':len(prefix),'prefix_sha256':sha(target),'first128_hex':prefix[:128].hex(),'first16_u32_le':list(struct.unpack('<16I',prefix[:64])),'scope':'Prefix only, no full ZIP CRC verified; binary interpretation not established.'})
            print(records[-1],flush=True)
    save(out/'probe.json',{'url':URL,'records':records,'network_bytes':remote.transferred,'ranges':remote.requests,'camera_intrinsics_verified':False,'frame_poses_verified':False})

if __name__=='__main__':main()
