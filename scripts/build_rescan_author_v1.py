"""Fetch the research-only GCO dependency and build unchanged author targets in WSL."""
import hashlib,io,json,subprocess,zipfile
from pathlib import Path
import requests

def main():
    out=Path('results/rescan_author_v1');out.mkdir(exist_ok=False)
    url='https://vision.cs.uwaterloo.ca/files/gco-v3.0.zip'
    r=requests.get(url,timeout=60);r.raise_for_status();(out/'gco-v3.0.zip').write_bytes(r.content)
    dest=Path('third_party/Rescan/lib/gco');dest.mkdir(exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for item in z.infolist():
            if item.is_dir():continue
            p=dest/Path(item.filename).name;p.write_bytes(z.read(item))
    licensefiles=[p for p in dest.iterdir() if any(s in p.name.lower() for s in ['license','readme'])]
    (out/'dependency_manifest.json').write_text(json.dumps({'url':url,'zip_sha256':hashlib.sha256(r.content).hexdigest(),
        'research_only':True,'license_files':[str(p) for p in licensefiles]},indent=2),encoding='utf-8')
    for p in licensefiles:print(p,p.read_text(encoding='utf-8',errors='replace')[:5000],flush=True)
    shell='set -e; for target in seg2rsdb pose_proposal segment_transfer create_eval_files; do cmake -S "third_party/Rescan/apps/$target" -B "tools/real_registration_v1/rescan_build/$target" -DCMAKE_BUILD_TYPE=Release; cmake --build "tools/real_registration_v1/rescan_build/$target" -j 2; done'
    cmd=['wsl','-d','Ubuntu-20.04','--exec','bash','-c',shell]
    with (out/'build.log').open('xb') as f:run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    (out/'build_result.json').write_text(json.dumps({'returncode':run.returncode},indent=2),encoding='utf-8')
    print('build_returncode',run.returncode,flush=True)

if __name__=='__main__':main()
