"""Use already installed system glog/gflags in the evaluation build copy only."""
import hashlib,json,pathlib,sys
project=pathlib.Path(sys.argv[1]);root=pathlib.Path('/home/tong/uav_khronos_ros1_ws_20261021/src/khronos/khronos_eval')
changes={}
for name in ['package.xml','CMakeLists.txt']:
    file=root/name;before=file.read_text()
    if name=='package.xml':
        after=before.replace('  <depend>glog_catkin</depend>\n','').replace('  <depend>gflags_catkin</depend>\n','')
    else:
        after=before.replace('find_package(GTSAM REQUIRED)','find_package(GTSAM REQUIRED)\nfind_package(glog REQUIRED)\nfind_package(gflags REQUIRED)')
        after=after.replace('target_link_libraries(${PROJECT_NAME} gtsam ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})','target_link_libraries(${PROJECT_NAME} gtsam glog::glog ${gflags_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})')
    assert after!=before
    file.write_text(after)
    changes[name]={'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'before':before,'after':after}
record={'scope':'Native evaluation build system only; no metric or algorithm code changed. Use installed system glog/gflags instead of missing catkin wrappers.','changes':changes}
(project/'results/labeled_temporal_v1/eval_system_library_adaptation.json').write_text(json.dumps(record,indent=2))
print('Recorded native-only evaluation library adaptation')
