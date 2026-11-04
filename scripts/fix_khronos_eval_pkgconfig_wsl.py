"""Ubuntu20.04 exposes installed glog via pkg-config, not a CMake glog package."""
import hashlib,json,pathlib,sys
project=pathlib.Path(sys.argv[1]);path=pathlib.Path('/home/tong/uav_khronos_ros1_ws_20261021/src/khronos/khronos_eval/CMakeLists.txt')
before=path.read_text();after=before.replace('find_package(glog REQUIRED)','find_package(PkgConfig REQUIRED)\npkg_check_modules(GLOG REQUIRED IMPORTED_TARGET libglog)').replace('glog::glog','PkgConfig::GLOG')
assert before!=after
path.write_text(after)
(project/'results/labeled_temporal_v1/eval_pkgconfig_adaptation.json').write_text(json.dumps({'before':before,'after':after,'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'scope':'Build system only; metric code unchanged'},indent=2))
print('Using Ubuntu system glog pkg-config')
