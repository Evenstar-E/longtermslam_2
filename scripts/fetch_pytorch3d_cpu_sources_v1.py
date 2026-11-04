"""Fetch pinned, official Python ICP source; compiled KNN/FPS are ported separately."""
import hashlib,json
from pathlib import Path
import requests

root=Path('third_party/pytorch3d_python_v074');root.mkdir(exist_ok=False)
records=[]
for name in ['pytorch3d/ops/points_alignment.py','pytorch3d/ops/utils.py',
             'pytorch3d/common/workaround/utils.py','pytorch3d/structures/utils.py',
             'pytorch3d/ops/sample_farthest_points.py','pytorch3d/ops/knn.py','LICENSE']:
    url='https://raw.githubusercontent.com/facebookresearch/pytorch3d/v0.7.4/'+name
    response=requests.get(url,timeout=60)
    record={'url':url,'status':response.status_code,'path':name}
    if response.status_code==200:
        target=root/name;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(response.content)
        record['sha256']=hashlib.sha256(response.content).hexdigest()
    records.append(record);print(record,flush=True)
(root/'acquisition.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
