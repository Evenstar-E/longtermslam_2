"""Official archive uses gt_mask/, not the later reader's masks/ convention.

Keep original failed preparation, preserve pair selection, extract in stream order.
"""
import gzip,tarfile,shutil
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
from scripts.run_rigid_identity_bounds_v1 import read,save,sha

def main():
    cfgpath=Path('configs/ycbineoat_motion_control_v1.json');cfg=read(cfgpath)
    source=Path('data/ycbineoat_motion_v1_parallel');archive=source/'cracker_box_reorient.tar.gz';acq=read(source/'acquisition.json')
    assert sha(archive)==acq['sha256'];out=Path('data/ycbineoat_motion_prepared_v2');out.mkdir(exist_ok=False)
    save(out/'preparation_start.json',{'config_sha256':sha(cfgpath),'source_sha256':sha(__file__),'archive_sha256':acq['sha256'],'pose_content_read_for_selection':False,'previous_failure':'Original archive has gt_mask, not masks; previous prepared_v1 retained, pairs unchanged.'})
    with tarfile.open(archive,'r:gz') as tar:names=[m.name for m in tar.getmembers() if m.isfile()]
    rgb=sorted(n for n in names if '/rgb/' in n and n.endswith('.png'));N=len(rgb);pairs=[];excluded=[]
    for f in cfg['historical_frame_fractions']:
        a=int(np.floor(f*(N-1)))
        for gap in cfg['query_gaps_as_sequence_fraction']:
            b=a+max(1,int(np.floor(gap*N)));item={'history_index':a,'current_index':b,'history_fraction':f,'gap_fraction':gap}
            if b>=N:excluded.append(item);continue
            pairs.append(dict(item,history_stem=PurePosixPath(rgb[a]).stem,current_stem=PurePosixPath(rgb[b]).stem))
    selected=sorted({p[k] for p in pairs for k in ['history_stem','current_stem']});seqroot=rgb[0].split('/rgb/')[0]
    previous=read('data/ycbineoat_motion_prepared_v1/selection.json');assert pairs==previous['pairs']
    save(out/'selection.json',{'sequence_frames':N,'pairs':pairs,'out_of_range_exclusions':excluded,'selected_stems':selected,'archive_inventory':names,'selection_uses_GT_pose':False})
    wanted={seqroot+'/cam_K.txt':out/'public/cam_K.txt'}
    for stem in selected:
        for source_folder,dest_folder in [('rgb','rgb'),('depth','depth'),('gt_mask','masks')]:wanted[f'{seqroot}/{source_folder}/{stem}.png']=out/'public'/dest_folder/(stem+'.png')
        wanted[f'{seqroot}/annotated_poses/{stem}.txt']=out/'evaluation/poses'/(stem+'.txt')
    assert set(wanted)<=set(names);records=[]
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if member.name not in wanted:continue
            path=PurePosixPath(member.name);assert not path.is_absolute() and '..' not in path.parts
            assert member.isfile() and not member.issym() and not member.islnk()
            dest=wanted[member.name].resolve();assert dest.is_relative_to(out.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst)
            records.append({'member':member.name,'path':str(dest.relative_to(out.resolve())),'bytes':dest.stat().st_size,'sha256':sha(dest)})
    assert len(records)==len(wanted)
    with gzip.open(archive,'rb') as f:
        while f.read(8*2**20):pass
    K=np.loadtxt(out/'public/cam_K.txt').reshape(3,3);clouds=[]
    for stem in selected:
        rgb=np.array(Image.open(out/'public/rgb'/(stem+'.png')))[...,:3];depth=np.array(Image.open(out/'public/depth'/(stem+'.png'))).astype(float)/cfg['depth_divisor'];mask=np.array(Image.open(out/'public/masks'/(stem+'.png')))
        if mask.ndim==3:mask=next(mask[...,i] for i in range(min(3,mask.shape[-1])) if mask[...,i].sum()>0)
        valid=(mask>0)&(depth>=cfg['depth_valid_interval_m'][0])&(depth<cfg['depth_valid_interval_m'][1]);v,u=np.nonzero(valid);z=depth[valid]
        xyz=np.column_stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]);dest=out/'public/clouds'/(stem+'.npz');dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,xyz=xyz,rgb=rgb[valid]);clouds.append({'stem':stem,'points':len(xyz),'path':str(dest.relative_to(out)),'sha256':sha(dest),'image_shape':list(depth.shape),'provided_mask_pixels':int((mask>0).sum())})
    save(out/'public.json',{'pairs':pairs,'clouds':clouds,'GT_pose_access_by_cloud_export':False,'provided_masks_privileged':True})
    save(out/'acquisition.json',{'records':records,'archive_gzip_crc_read_to_EOF':True,'archive_sha256':acq['sha256'],'independent_GT_pose_files_exported_but_not_read':True})
    print('PREPARED',N,'frames',len(pairs),'pairs',len(selected),'observed clouds; poses withheld')

if __name__=='__main__':main()
