"""Pinned official public source/4.4MB weights only; no package installation."""
from pathlib import Path
import json,hashlib,urllib.request

COMMIT='b009d536ac10b570853833f2178397c154745da9'
FILES=['README.md','demo_registration.py','models/PointDSC.py','models/common.py','utils/SE3.py','snapshot/PointDSC_3DMatch_release/config.json','snapshot/PointDSC_3DMatch_release/models/model_best.pkl']


def main():
    root=Path('third_party/PointDSC_official_v1');root.mkdir(parents=True,exist_ok=False);records=[]
    for name in FILES:
        url=f'https://raw.githubusercontent.com/XuyangBai/PointDSC/{COMMIT}/{name}'
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(8000000)
        assert len(data)<8000000;path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),url=url))
        print('POINTDSC FETCH',name,len(data),flush=True)
    (root/'source_manifest.json').write_text(json.dumps(dict(complete=True,repository='https://github.com/XuyangBai/PointDSC',commit=COMMIT,files=records,weights='Official3DMatch model_best, no training or replacements',scope='Read-only originals preserved. No author benchmark executed; CPU compatibility not yet verified.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
