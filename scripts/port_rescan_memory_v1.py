"""Isolated memory-safety port; no proposal, scoring, optimization, or thresholds changed."""
import difflib,json,shutil,subprocess
from pathlib import Path
from scripts.run_rescan_author_v1 import linux

original=Path('third_party/Rescan');port=Path('third_party/Rescan_memory_port_v1')
shutil.copytree(original,port,ignore=shutil.ignore_patterns('.git','bin'))
changes=[]
def edit(name,old,new,count=1):
    p=port/name;s=p.read_text(encoding='utf-8');assert s.count(old)==count,(name,s.count(old),old)
    p.write_text(s.replace(old,new),encoding='utf-8');changes.append({'path':name,'old':old,'new':new})
edit('lib/rs/rs_database.h','char class_name[64] = {0};','char class_name[512] = {0};')
edit('lib/rs/rs_database.h','  rsdb->model_folder              = NULL;','  rsdb->model_folder              = NULL;\n  rsdb->pointcloud_data_present = 0;')
edit('apps/seg2rsdb/main.cpp','scene.scn_filename           = opts->in_ply_filename;','scene.scn_filename           = msh_strdup(opts->in_ply_filename);')
edit('lib/rs/rs_database.h','scene.scn_filename           = filename;','scene.scn_filename           = msh_strdup(filename);')
# The generic insertion takes a string pointer, whereas the table copies 512 bytes.
edit('lib/rs/rs_database.h','  hashtable_insert(&rsdb->idx_to_class_name, (HASHTABLE_U64)class_idx, class_name );',
     '  char stored_class_name[512] = {0};\n  snprintf(stored_class_name, sizeof(stored_class_name), "%s", class_name);\n  hashtable_insert(&rsdb->idx_to_class_name, (HASHTABLE_U64)class_idx, stored_class_name );',count=2)
out=Path('results/rescan_author_v1/memory_port');out.mkdir(exist_ok=False)
(out/'changes.json').write_text(json.dumps(changes,indent=2),encoding='utf-8')
diff=[]
for name in sorted({r['path'] for r in changes}):
    diff.extend(difflib.unified_diff((original/name).read_text(encoding='utf-8').splitlines(True),(port/name).read_text(encoding='utf-8').splitlines(True),fromfile='original/'+name,tofile='memory_port/'+name))
(out/'memory_only.patch').write_text(''.join(diff),encoding='utf-8')
shell='set -e; for target in seg2rsdb pose_proposal segment_transfer; do cmake -S "third_party/Rescan_memory_port_v1/apps/$target" -B "tools/real_registration_v1/rescan_port_build/$target" -DCMAKE_BUILD_TYPE=Release; cmake --build "tools/real_registration_v1/rescan_port_build/$target" -j 2; done'
with (out/'build.log').open('wb') as f:r=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell],stdout=f,stderr=subprocess.STDOUT,timeout=300)
(out/'build_result.json').write_text(json.dumps({'returncode':r.returncode}),encoding='utf-8');print('port build',r.returncode,flush=True)
