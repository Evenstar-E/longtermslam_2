"""Keep failed author outputs; local ASAN build to identify an engineering crash."""
from pathlib import Path
import argparse,subprocess
from scripts.run_rescan_author_v1 import linux

parser=argparse.ArgumentParser();parser.add_argument('--source',default='third_party/Rescan');parser.add_argument('--output',default='results/rescan_author_v1/crash_diagnostic');args=parser.parse_args()
root=Path(args.output);root.mkdir(exist_ok=False)
source_code=Path(args.source)
binary=root/'seg2rsdb_asan'
args=['g++','-std=c++11','-O1','-g','-fsanitize=address','-fno-omit-frame-pointer',
    '-I'+linux(source_code/'lib'),'-I'+linux(source_code/'lib/rs'),
    linux(source_code/'apps/seg2rsdb/main.cpp'),'-o',linux(binary)]
with (root/'build.log').open('wb') as f:subprocess.run(['wsl','-d','Ubuntu-20.04','--exec']+args,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=120)
source=Path('results/rescan_author_v1/inference/scene_c')
cmd=['wsl','-d','Ubuntu-20.04','--exec','env','ASAN_OPTIONS=detect_leaks=0',linux(binary),linux(source/'history.ply'),linux(source/'classes.txt'),linux(root/'history.rsdb'),'-v']
with (root/'run.log').open('wb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=90)
print('returncode',r.returncode)
print((root/'run.log').read_text(encoding='utf-8',errors='replace')[-8000:])
