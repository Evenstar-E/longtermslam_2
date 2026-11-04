import json,pathlib,rosbag
p=pathlib.Path('data/khronos_official_v1/tesse_cd_apartment.bag')
out={}
topics=['/tesse/left_cam/camera_info','/tesse/left_cam/rgb/image_raw','/tesse/depth_cam/mono/image_raw','/tesse/seg_cam/converted/image_raw','/tf','/tf_static','/tesse/odom']
with rosbag.Bag(str(p)) as bag:
    for topic in topics:
        first=next(bag.read_messages(topics=[topic]),None)
        if first is None:out[topic]=None;continue
        msg=first.message
        if hasattr(msg,'encoding'):
            out[topic]={'encoding':msg.encoding,'width':msg.width,'height':msg.height,'step':msg.step,'bigendian':msg.is_bigendian,'frame':msg.header.frame_id,'stamp_ns':msg.header.stamp.to_nsec()}
        elif hasattr(msg,'K'):
            out[topic]={'K':list(msg.K),'frame':msg.header.frame_id,'width':msg.width,'height':msg.height,'D':list(msg.D)}
        elif hasattr(msg,'transforms'):
            out[topic]=[{'parent':t.header.frame_id,'child':t.child_frame_id,'stamp_ns':t.header.stamp.to_nsec(),'translation':[t.transform.translation.x,t.transform.translation.y,t.transform.translation.z],'quaternion_xyzw':[t.transform.rotation.x,t.transform.rotation.y,t.transform.rotation.z,t.transform.rotation.w]} for t in msg.transforms]
        else:out[topic]={'type':msg._type,'frame':msg.header.frame_id,'stamp_ns':msg.header.stamp.to_nsec()}
path=pathlib.Path('results/event_evidence_v1');path.mkdir(exist_ok=False)
(path/'bag_input_contract.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
