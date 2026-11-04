"""Read public dataset pages; never submit access forms or credentials."""
import json
from html.parser import HTMLParser
from pathlib import Path
import requests

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            self.links += [v for k,v in attrs if k=='href']

root=Path('data/real_cross_session_v1/access'); root.mkdir(parents=True,exist_ok=True)
for name,url in [('rescan','https://rescan.cs.princeton.edu/'),
                 ('tuw','https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-change-detection-dataset-of-indoor-environments/')]:
    try:
        r=requests.get(url,timeout=45); parser=Links(); parser.feed(r.text)
        (root/(name+'.html')).write_text(r.text,encoding='utf-8')
        record={'url':url,'status':r.status_code,'links':parser.links}
    except requests.RequestException as e:
        record={'url':url,'error':str(e)}
    (root/(name+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))
