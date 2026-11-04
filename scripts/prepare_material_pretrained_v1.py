"""Fetch only official public inference sources/weights, no environment install."""
import hashlib
import json
from pathlib import Path
import requests


def main():
    root=Path('third_party/material_local_matching_v1');root.mkdir(parents=True,exist_ok=True)
    s=requests.Session();s.headers['User-Agent']='material-rgbd-research'
    if (root/'source_manifest.json').exists():commit=json.loads((root/'source_manifest.json').read_text(encoding='utf-8'))['commit']
    else:
        r=s.get('https://api.github.com/repos/cvg/LightGlue/commits/main',timeout=40);r.raise_for_status();commit=r.json()['sha']
    manifest=dict(repo='https://github.com/cvg/LightGlue',commit=commit,files={})
    urls={Path(n).name:f'https://raw.githubusercontent.com/cvg/LightGlue/{commit}/{n}' for n in ['LICENSE','lightglue/superpoint.py','lightglue/lightglue.py','lightglue/utils.py']}
    urls.update({n:'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/'+n for n in ['superpoint_v1.pth','superpoint_lightglue.pth']})
    for name,u in urls.items():
        r=s.get(u,timeout=90);r.raise_for_status();p=root/name
        if p.exists() and p.read_bytes()!=r.content:raise RuntimeError('Refuse overwriting different third-party source: '+str(p))
        if not p.exists():p.write_bytes(r.content)
        manifest['files'][name]=dict(url=u,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content))
        (root/'download_progress.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print('FETCHED',name,len(r.content),flush=True)
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('PRETRAINED SEALED',commit,flush=True)


if __name__=='__main__':main()
