"""Fix a scorer parse error only; sealed inference/source evidence untouched."""
from pathlib import Path


def main():
    source=Path('scripts/score_material_surface_correspondence_v13b.py').read_text(encoding='utf-8')
    old="records=json.loads((scene/'frames.json').read_text(encoding='utf-8')))"
    new="records=json.loads((scene/'frames.json').read_text(encoding='utf-8'))"
    assert source.count(old)==1;source=source.replace(old,new);compile(source,'score_material_surface_correspondence_v13c.py','exec')
    p=Path('scripts/score_material_surface_correspondence_v13c.py');assert not p.exists();p.write_text(source,encoding='utf-8');print('SCORER V13C parse fix generated')


if __name__=='__main__':main()
