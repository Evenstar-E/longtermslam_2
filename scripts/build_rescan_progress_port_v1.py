"""Isolated total-progress fix for Poisson point subsampling; no objective changes."""
import difflib
import shutil
import subprocess
from pathlib import Path
from scripts.run_rescan_third_period_v1 import save,sha


def main():
    source=Path('third_party/Rescan_memory_port_v1');port=Path('third_party/Rescan_progress_port_v1')
    shutil.copytree(source,port,ignore=shutil.ignore_patterns('.git','bin'))
    file=port/'lib/rs/rs_pointcloud.h';old=file.read_text(encoding='utf-8')
    needle='    n_marked += n_valid_neigh;'
    assert old.count(needle)==1
    replacement='''    // A selected point must remove itself even if the neighbor routine omits it.
    // This only changes the no-self branch, which otherwise can repeat forever.
    if( unmarked[sample_idx] > 0 )
    {
      fprintf(stderr, "SAMPLER_PROGRESS_FIX level=%d sample=%d neighbors=%zu newly_marked=%zu\\n",
              level, sample_idx, n_neigh, n_valid_neigh);
      unmarked[sample_idx] = 0;
      n_valid_neigh++;
    }
    n_marked += n_valid_neigh;'''
    new=old.replace(needle,replacement);file.write_text(new,encoding='utf-8')
    out=Path('results/rescan_progress_port_v1');out.mkdir(exist_ok=False)
    (out/'progress_only.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='memory_port/rs_pointcloud.h',tofile='progress_port/rs_pointcloud.h')),encoding='utf-8')
    changed=[];same=0
    for p in source.rglob('*'):
        if p.is_file() and 'bin' not in p.relative_to(source).parts:
            if sha(p)!=sha(port/p.relative_to(source)):changed.append(str(p.relative_to(source)))
            else:same+=1
    assert changed==['lib\\rs\\rs_pointcloud.h'],changed
    records=[]
    for target in ['pose_proposal','segment_transfer']:
        build=f'tools/real_registration_v1/rescan_progress_build/{target}'
        commands=[['cmake','-S',f'third_party/Rescan_progress_port_v1/apps/{target}','-B',build,'-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',build,'-j','2']]
        with (out/f'{target}_build.log').open('xb') as log:
            for c in commands:
                r=subprocess.run(['wsl','-d','Ubuntu-20.04','--exec']+c,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                records.append({'command':c,'returncode':r.returncode});assert r.returncode==0
    save(out/'build.json',{'records':records,'changed':changed,'unchanged_files':same,'source_sha':sha(__file__),
        'scope':'Guarantees selected point is marked, no geometric thresholds/objective/pose/identity score changes. Patched runs must be labeled; actual activation logged.'})
    print('BUILT',changed,'unchanged',same)


if __name__=='__main__':main()
