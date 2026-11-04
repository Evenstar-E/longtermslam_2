"""Record final task port separately from the earlier unsuccessful import attempt."""
import difflib, hashlib, json
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    original=Path('third_party/LivingScenes'); port=Path('third_party/LivingScenes_cpu_port_v1')
    out=Path('results/livingscenes_cpu_port_v1'); changed=[]; same=[]; added=[]; diff=[]
    for p in sorted(port.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts: continue
        rel=p.relative_to(port); source=original/rel
        if not source.exists(): added.append(str(rel)); continue
        if sha(p)==sha(source): same.append(str(rel)); continue
        changed.append(str(rel))
        diff.extend(difflib.unified_diff(source.read_text(encoding='utf-8').splitlines(True),p.read_text(encoding='utf-8').splitlines(True),fromfile='original/'+rel.as_posix(),tofile='cpu_port/'+rel.as_posix()))
    (out/'cpu_port_final.patch').write_text(''.join(diff),encoding='utf-8')
    record={'scope':'CPU task inference: original encoder/decoder weights, matching, pairwise pose optimization and ICP. No mesh output, training or official benchmark reproduction.',
        'earlier_manifest_status':'port_manifest.json/cpu_port.patch precede the eager training-registry import fix; retained as integration history, superseded by this final record.',
        'changed_original_files':changed,'identical_original_files':same,'added_compatibility_files':added,
        'files_sha256':{str(p.relative_to(port)):sha(p) for p in port.rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        'preparation_script_sha256':sha(Path('scripts/prepare_livingscenes_cpu_port_v1.py'))}
    (out/'port_manifest_final.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('FINAL PORT',len(changed),'changed',len(same),'identical',len(added),'added')

if __name__=='__main__': main()
