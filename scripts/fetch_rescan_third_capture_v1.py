"""Prepare unused third captures for next multi-asset experiment; no scoring."""
from pathlib import Path,PurePosixPath
import zipfile
import numpy as np
from scripts.fetch_rescan_subset_v1 import RemoteZip,URL
from scripts.prepare_real_temporal_v1 import read_ply
from scripts.run_rigid_identity_bounds_v1 import read,save,sha


def main():
    out=Path('data/rescan_third_capture_v1')
    if out.exists():
        assert sorted(p.name for p in out.iterdir())==['selection.json'],'Only an empty selected network attempt may resume'
    else:out.mkdir()
    inventory=read('data/real_cross_session_v1/rescan/inventory.json')
    names=[f'{s}/gt_segmentation/{s}_2.ply' for s in ['scene_a','scene_c']]
    records=[next(r for r in inventory['entries'] if r['name']==name) for name in names]
    if (out/'selection.json').exists():
        assert read(out/'selection.json')['members']==records
        save(out/'network_resume.json',dict(reason='Default sandbox socket denied before download; resume same frozen selection',source_sha=sha(__file__)))
    else:save(out/'selection.json',dict(members=records,rule='First unused capture2 in both previously studied native-mask spaces; chosen before reading labels or fitting any new method.',old_inventory_sha=sha('data/real_cross_session_v1/rescan/inventory.json'),source_sha=sha(__file__),scope='Fresh time periods in already-known spaces, NOT independent new spaces. Next mainline multi-asset input preparation only; no motion ground truth/calibration inferred.'))
    remote=RemoteZip();acquired=[];public=[]
    with zipfile.ZipFile(remote) as archive:
        for record in records:
            name=record['name'];info=archive.getinfo(name);assert info.CRC==record['crc'] and info.file_size==record['bytes'];assert '..' not in PurePosixPath(name).parts
            scene=name.split('/')[0];raw=out/'evaluation/raw'/name;raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(archive.read(info))
            content=raw.read_bytes();header,_=content.split(b'end_header\n',1);points=read_ply(raw).copy()
            # Do not use, summarize or expose true per-point labels in public input.
            points['instance_idx']=-1;points['class_idx']=0
            dest=out/'public'/scene/'capture2.ply';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(header+b'end_header\n'+points.tobytes())
            assert np.all(read_ply(dest)['instance_idx']==-1) and np.all(read_ply(dest)['class_idx']==0)
            acquired.append(dict(member=name,path=str(raw.relative_to(out)),bytes=info.file_size,sha256=sha(raw),selected_member_crc_verified=True))
            public.append(dict(scene=scene,capture=2,path=str(dest.relative_to(out)),points=len(points),sha256=sha(dest),current_GT_instance_and_class_removed=True))
            save(out/'acquisition.partial.json',dict(records=acquired,complete=False));print('SAVED',scene,len(points),flush=True)
    save(out/'public.json',dict(clouds=public,scope='Unsegmented current scans. No privileged current object masks; native segmentation/inference still to run.'))
    save(out/'acquisition.json',dict(complete=True,url=URL,records=acquired,transferred_bytes=remote.transferred,ranges=remote.requests,full_archive_hash_verified=False))
    save(out/'seal.json',dict(sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},identity_evaluation_run=False))


if __name__=='__main__':main()
