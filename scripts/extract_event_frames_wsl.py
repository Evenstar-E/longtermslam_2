"""Two-pass offline extraction of frozen sample requests and original camera poses."""
import bisect,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,rosbag
from scripts.event_geometry import rigid_matrix
base=Path('data/event_evidence_v1');plan=json.loads((base/'plan.json').read_text());cfg=json.loads((base/'frozen_config.json').read_text())
dest=base/'frames';dest.mkdir(exist_ok=False)
bag_path=Path('data/khronos_official_v1/tesse_cd_apartment.bag')
rgb_topic='/tesse/left_cam/rgb/image_raw';depth_topic='/tesse/depth_cam/mono/image_raw';label_topic='/tesse/seg_cam/converted/image_raw'
rgb_stamps=[];poses={};camera=None;extrinsic=None;start=time.monotonic();duplicate_tfs=0
def transform_record(t):
    a=t.transform.translation;q=t.transform.rotation
    return {'translation':[a.x,a.y,a.z],'quaternion_xyzw':[q.x,q.y,q.z,q.w]}
with rosbag.Bag(str(bag_path)) as bag:
    for topic,msg,bt in bag.read_messages(topics=[rgb_topic,'/tf','/tf_static','/tesse/left_cam/camera_info']):
        if topic==rgb_topic:
            rgb_stamps.append({'stamp_ns':msg.header.stamp.to_nsec(),'bag_ns':bt.to_nsec()})
            if len(rgb_stamps)%400==0:print('Indexed RGB',len(rgb_stamps),flush=True)
        elif topic=='/tesse/left_cam/camera_info':
            this={'K':list(msg.K),'width':msg.width,'height':msg.height,'D':list(msg.D),'frame':msg.header.frame_id}
            if camera is not None and camera!=this:raise RuntimeError('Camera calibration changes')
            camera=this
        else:
            for t in msg.transforms:
                parent=t.header.frame_id.lstrip('/');child=t.child_frame_id.lstrip('/')
                if topic=='/tf' and parent=='world' and child=='base_link_gt':
                    ns=t.header.stamp.to_nsec();record=transform_record(t)
                    if ns in poses:
                        duplicate_tfs+=1
                        if poses[ns]!=record:raise RuntimeError('Conflicting original TF')
                    poses[ns]=record
                elif topic=='/tf_static' and parent=='base_link_gt' and child=='left_cam':
                    record=transform_record(t)
                    if extrinsic is not None and extrinsic!=record:raise RuntimeError('Conflicting camera extrinsics')
                    extrinsic=record
assert camera and extrinsic and len(poses)>0
assert len(rgb_stamps)==len({r['stamp_ns'] for r in rgb_stamps})
stamps=sorted(r['stamp_ns'] for r in rgb_stamps);pose_stamps=sorted(poses)
def nearest(values,t):
    i=bisect.bisect_left(values,t);options=values[max(0,i-1):min(len(values),i+1)]
    return min(options,key=lambda s:(abs(s-t),s))
selection={};request_rows=[]
for request in plan['requests']:
    r=dict(request)
    if not plan['bag_start_ns']<=r['time_ns']<=plan['bag_end_ns']:
        r.update(selected_stamp_ns=None,reason='outside_recording');request_rows.append(r);continue
    stamp=nearest(stamps,r['time_ns']);gap=abs(stamp-r['time_ns'])
    if gap>cfg['sampling']['nearest_frame_max_gap_s']*1e9:raise RuntimeError('Frame time contract violated')
    r.update(selected_stamp_ns=stamp,selection_gap_ns=gap)
    selection.setdefault(stamp,[]).append(r['id']);request_rows.append(r)
T_base_camera=rigid_matrix(extrinsic['translation'],extrinsic['quaternion_xyzw'])
records={}
for stamp in selection:
    ns=nearest(pose_stamps,stamp)
    if abs(ns-stamp)>cfg['sampling']['pose_max_gap_s']*1e9:raise RuntimeError('Pose time contract violated')
    pose=poses[ns];T=rigid_matrix(pose['translation'],pose['quaternion_xyzw'])@T_base_camera
    records[stamp]={'stamp_ns':stamp,'pose_stamp_ns':ns,'pose_gap_ns':abs(ns-stamp),'pose_record':pose,'T_world_camera':T.tolist(),'requests':selection[stamp],'channels':{}}
print('Selected unique frames',len(selection),'from',len(stamps),flush=True)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def array(msg):
    h,w=msg.height,msg.width
    if (h,w)!=(camera['height'],camera['width']):raise RuntimeError('Dimension mismatch')
    if msg.encoding=='rgb8':return np.frombuffer(msg.data,np.uint8).reshape(h,msg.step)[:,:w*3].reshape(h,w,3).copy()
    dtype={'32FC1':'f4','16UC1':'u2'}[msg.encoding];dt=np.dtype(('>' if msg.is_bigendian else '<')+dtype)
    return np.frombuffer(msg.data,dt).reshape(h,msg.step//dt.itemsize)[:,:w].astype(np.float32 if dtype=='f4' else np.uint16)
finished=0
with rosbag.Bag(str(bag_path)) as bag:
    for topic,msg,bt in bag.read_messages(topics=[rgb_topic,depth_topic,label_topic]):
        stamp=msg.header.stamp.to_nsec()
        if stamp not in selection:continue
        name={rgb_topic:'rgb',depth_topic:'depth',label_topic:'labels'}[topic]
        expected={'rgb':'rgb8','depth':'32FC1','labels':'16UC1'}[name]
        if msg.encoding!=expected:raise RuntimeError('Encoding changed')
        a=array(msg);path=dest/(str(stamp)+'_'+name+('.npy' if name=='depth' else '.png'))
        if path.exists():raise RuntimeError('Duplicate selected image')
        if name=='depth':np.save(path,a)
        elif not cv2.imwrite(str(path),a[:,:,::-1] if name=='rgb' else a):raise RuntimeError('Image save failed')
        records[stamp]['channels'][name]={'path':str(path),'encoding':msg.encoding,'shape':list(a.shape),'dtype':str(a.dtype),'bag_time_ns':bt.to_nsec(),'file_sha256':digest(path),'array_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        if len(records[stamp]['channels'])==3:
            finished+=1
            if finished%25==0:print('Extracted complete frames',finished,'elapsed_s',round(time.monotonic()-start,1),flush=True)
all_records=[records[s] for s in sorted(records)]
report={'complete':all(len(r['channels'])==3 for r in all_records),'frames':all_records,'requests':request_rows,'camera':camera,'original_extrinsic':extrinsic,
 'rgb_frames_in_original_bag':len(stamps),'pose_records':len(poses),'duplicate_tfs':duplicate_tfs,'elapsed_s':time.monotonic()-start,
 'protocol_sha256':hashlib.sha256(Path('configs/event_evidence_v1.json').read_bytes()).hexdigest(),
 'input_plan_sha256':hashlib.sha256((base/'plan.json').read_bytes()).hexdigest(),'source_bag_sha256':json.loads(Path('data/khronos_official_v1/acquisition.json').read_text())['sha256'],
 'environment':{'numpy':np.__version__,'opencv':cv2.__version__},'scope':'Offline original RGB/depth/semantic extraction; no inference supplied with future observations'}
(base/'acquisition.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'complete':report['complete'],'frames':len(all_records),'requests':len(request_rows),'elapsed_s':report['elapsed_s']}),flush=True)
if not report['complete']:raise RuntimeError('Missing selected channels')
