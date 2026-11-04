"""Select frozen temporal positions, safely extract RGBD/masks and separate GT files."""
import hashlib,json,tarfile,shutil
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    cfgpath=Path('configs/ycbineoat_motion_control_v1.json');cfg=read(cfgpath)
    source=Path('data/ycbineoat_motion_v1_parallel');archive=source/'cracker_box_reorient.tar.gz'
    acquisition=read(source/'acquisition.json');assert sha(archive)==acquisition['sha256']
    out=Path('data/ycbineoat_motion_prepared_v1');out.mkdir(exist_ok=False)
    save(out/'preparation_start.json',{'config_sha256':sha(cfgpath),'source_sha256':sha(__file__),'archive_sha256':acquisition['sha256'],'pose_content_read_for_selection':False})
    records=[]
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members if m.isfile()]
        rgb=sorted([n for n in names if '/rgb/' in n and n.endswith('.png')])
        assert rgb
        N=len(rgb);pairs=[];excluded=[]
        for f in cfg['historical_frame_fractions']:
            a=int(np.floor(f*(N-1)))
            for gap in cfg['query_gaps_as_sequence_fraction']:
                b=a+max(1,int(np.floor(gap*N)));record={'history_index':a,'current_index':b,'history_fraction':f,'gap_fraction':gap}
                if b>=N:excluded.append(record);continue
                pairs.append(dict(record,history_stem=PurePosixPath(rgb[a]).stem,current_stem=PurePosixPath(rgb[b]).stem))
        selected=sorted({p[k] for p in pairs for k in ['history_stem','current_stem']});byname={m.name:m for m in members if m.isfile()}
        sequence_root=rgb[0].split('/rgb/')[0]
        wanted={f'{sequence_root}/cam_K.txt':out/'public/cam_K.txt'}
        for stem in selected:
            for folder in ['rgb','depth','masks']:wanted[f'{sequence_root}/{folder}/{stem}.png']=out/'public'/folder/(stem+'.png')
            wanted[f'{sequence_root}/annotated_poses/{stem}.txt']=out/'evaluation/poses'/(stem+'.txt')
        save(out/'selection.json',{'sequence_frames':N,'pairs':pairs,'out_of_range_exclusions':excluded,'selected_stems':selected,'archive_inventory':names,'selection_uses_GT_pose':False})
        for name,dest in wanted.items():
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive member')
            if name not in byname:raise FileNotFoundError(name)
            member=byname[name];assert member.isfile() and not member.issym() and not member.islnk()
            dest=dest.resolve();assert dest.is_relative_to(out.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,dest.open('xb') as target:shutil.copyfileobj(src,target)
            records.append({'member':name,'path':str(dest.relative_to(out.resolve())),'sha256':sha(dest),'bytes':dest.stat().st_size})
    # Reading gzip to EOF validates its own CRC/length. This is not an official
    # publisher checksum, which the download directory does not supply.
    import gzip
    with gzip.open(archive,'rb') as f:
        while f.read(8*2**20):pass
    K=np.loadtxt(out/'public/cam_K.txt').reshape(3,3);clouds=[]
    for stem in selected:
        rgb=np.array(Image.open(out/'public/rgb'/(stem+'.png')))[...,:3]
        depth=np.array(Image.open(out/'public/depth'/(stem+'.png'))).astype(float)/cfg['depth_divisor']
        mask=np.array(Image.open(out/'public/masks'/(stem+'.png')))
        if mask.ndim==3:
            candidates=[mask[...,c] for c in range(mask.shape[-1]) if mask[...,c].sum()>0];mask=candidates[0]
        valid=(mask>0)&(depth>=cfg['depth_valid_interval_m'][0])&(depth<cfg['depth_valid_interval_m'][1]);v,u=np.nonzero(valid);z=depth[valid]
        xyz=np.column_stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]);dest=out/'public/clouds'/(stem+'.npz');dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,xyz=xyz,rgb=rgb[valid]);clouds.append({'stem':stem,'points':len(xyz),'path':str(dest.relative_to(out)),'sha256':sha(dest)})
    save(out/'public.json',{'pairs':pairs,'clouds':clouds,'GT_pose_access_by_cloud_export':False,'provided_masks_privileged':True})
    save(out/'acquisition.json',{'records':records,'archive_gzip_crc_read_to_EOF':True,'archive_sha256':acquisition['sha256'],'independent_GT_pose_files_exported_but_not_read':True})
    print('PREPARED',N,'frames',len(pairs),'pairs',len(selected),'observed clouds, pose files withheld')

if __name__=='__main__':main()
