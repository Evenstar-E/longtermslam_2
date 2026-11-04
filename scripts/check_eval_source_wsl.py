"""Verify metric source is identical to pinned public source, despite native build-only patches."""
import hashlib,json,pathlib,sys
project=pathlib.Path(sys.argv[1]);original=project/'third_party/Khronos_ROS1/khronos_eval';native=pathlib.Path('/home/tong/uav_khronos_ros1_ws_20261021/src/khronos/khronos_eval')
paths=['src/object_evaluator.cpp','src/dynamic_object_evaluator.cpp','src/pipeline_evaluator.cpp','src/mesh_evaluator.cpp','app/exp_pipeline.cpp']
record=[]
for rel in paths:
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    a,b=sha(original/rel),sha(native/rel)
    record.append({'path':rel,'pinned_source_sha256':a,'native_sha256':b,'same':a==b})
out=project/'results/labeled_temporal_v1/native_metric_source_check.json'
with out.open('x') as f:json.dump({'passed':all(r['same'] for r in record),'records':record},f,indent=2)
print('Metric sources identical:',all(r['same'] for r in record))
