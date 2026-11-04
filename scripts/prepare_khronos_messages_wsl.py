"""Copy the pinned message-only package and declare its missing build dependency."""
import hashlib,json,pathlib,shutil,sys
project=pathlib.Path(sys.argv[1]);ws=pathlib.Path('/home/tong/uav_khronos_ros1_ws_20261021')
source=project/'third_party/khronos_ros1_deps/semantic_inference/semantic_inference_msgs'
dest=ws/'src/semantic_inference_msgs'
shutil.copytree(source,dest)
package=ws/'src/khronos/khronos_ros/package.xml'
before=package.read_text()
after=before.replace('  <depend>sensor_msgs</depend>', '  <depend>sensor_msgs</depend>\n  <depend>semantic_inference_msgs</depend>')
assert after!=before
package.write_text(after)
record={'semantic_inference_revision':'0ff8e94bb652b3c99b8e9b347a04afd1caa4fa5a',
 'scope':'Only ROS message definitions copied; no neural inference/model package installed.',
 'patch':'Declare semantic_inference_msgs dependency in native build-copy khronos_ros/package.xml; original pinned source unchanged.',
 'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),
 'copied_files':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}}
(project/'results/khronos_reproduction_v1/message_dependency_adaptation.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
