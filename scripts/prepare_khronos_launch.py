"""Generate a headless copy of official launch; preserve algorithm settings and provenance."""
import hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
source=Path("third_party/Khronos_ROS1/khronos_ros/launch/uhumans2_khronos.launch")
tree=ET.parse(source);root=tree.getroot()
changes={}
for arg in root.findall("arg"):
    name=arg.get("name")
    defaults={"sim_time_required":"true","dataset":"tesse_cd_apartment","paused":"true","play_rate":"0.5"}
    if name in defaults:
        changes[name]={"before":arg.get("default"),"after":defaults[name]}
        arg.set("default",defaults[name])
root.insert(0,ET.Element("arg",{"name":"start_visualizer","default":"false"}))
dest=Path("configs/khronos_headless_apartment_v1.launch")
if dest.exists():raise FileExistsError(dest)
ET.indent(tree,space="  ")
tree.write(dest,encoding="utf-8",xml_declaration=True)
record={"official_root":"4e484c4879a76a37e4bf6304b5c58132dc11b279",
 "official_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "generated_launch_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
 "changed_launch_defaults":changes,"added_arg":{"start_visualizer":False},
 "algorithm_config_modified":False,
 "privileged_inputs":"Official simulated example uses GT poses and GT semantic labels; this is mapping/change pipeline reproduction,not visual localization or fair real-data benchmark."}
Path("results/khronos_reproduction_v1/launch_adaptation.json").write_text(json.dumps(record,indent=2),encoding="utf-8")

