"""One-time cleanup audit. Plans and verifies; never deletes files.

Run with -B. All payload deletion is performed separately from the saved plan.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/cleanup_2026-10-25'


def native_path(p):
    # CMake/WSL object paths can exceed the legacy Win32 260-character limit.
    raw = str(p.absolute())
    return Path('\\\\?\\' + raw) if os.name == 'nt' and not raw.startswith('\\\\?\\') else p


def digest(p):
    with native_path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(name, value):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')


def inventory():
    files, links, dirs = {}, [], []
    stack = [ROOT]
    while stack:
        parent = stack.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                p = Path(entry.path)
                if p == OUT:
                    continue
                rel = p.relative_to(ROOT).as_posix()
                st = entry.stat(follow_symlinks=False)
                if st.st_file_attributes & 0x400:
                    links.append(rel)
                elif entry.is_dir(follow_symlinks=False):
                    dirs.append(rel)
                    stack.append(p)
                else:
                    files[rel] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                                      inode=st.st_ino, device=st.st_dev, nlink=st.st_nlink)
    return files, links, dirs


def provenance(files):
    selected = []
    for rel in files:
        p = Path(rel)
        if p.parts[0] in ('.cache', '.venv', '.venv-real') or '__pycache__' in p.parts:
            continue
        if p.suffix.lower() not in ('.json', '.jsonl', '.md', '.txt', '.toml', '.yaml', '.yml', '.py', '.ps1'):
            continue
        # Large raw JSON arrays contain observations/GT, not cache dependencies.
        # Always include named provenance documents, regardless of size.
        named_provenance = any(x in p.name.lower() for x in ('manifest', 'seal', 'index', 'contract', 'acquisition', 'evidence'))
        if files[rel]['bytes'] > 16 * 2**20 and not named_provenance:
            continue
        selected.append(rel)
    # Rust regex avoids pathological backtracking on long numeric JSON arrays.
    common = ['rg', '--hidden', '--no-ignore', '--no-filename', '--only-matching', '--max-filesize', '16M']
    for pattern in ['!.cache/**', '!.venv/**', '!.venv-real/**', '!**/__pycache__/**', '!research/cleanup_2026-10-25/**']:
        common += ['--glob', pattern]
    for ext in ('json', 'jsonl', 'md', 'txt', 'toml', 'yaml', 'yml', 'py', 'ps1'):
        common += ['--glob', '*.' + ext]
    def search(pattern):
        run = subprocess.run(common + ['--regexp', pattern, '.'], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if run.returncode not in (0, 1):
            raise RuntimeError(run.stderr.decode('utf-8', errors='replace'))
        return run.stdout.decode('utf-8', errors='replace').splitlines()
    hashes = {s.lower() for s in search(r'\b[[:xdigit:]]{64}\b')}
    strings = search(r'"([^"\r\n]*(?:__pycache__|\.cache)[^"\r\n]*)"')
    refs = set()
    for text in strings:
        text = text.strip('"').replace('\\', '/').lower()
        while '//' in text:
            text = text.replace('//', '/')
        if text.endswith('.pyc') or text.startswith('.cache/') or '/.cache/' in text:
            refs.add(text)
    for rel in selected:
        if files[rel]['bytes'] > 16 * 2**20:
            raw = (ROOT/rel).read_bytes()
            hashes.update(s.decode().lower() for s in re.findall(rb'\b[a-fA-F0-9]{64}\b', raw))
            if b'__pycache__' in raw or b'.cache/uv' in raw:
                raise RuntimeError('Large provenance document contains cache references; manual review required: ' + rel)
    return hashes, refs, len(selected)


def prepare():
    if (OUT / 'plan.json').exists():
        raise FileExistsError('Refuse to overwrite existing cleanup plan')
    started = time.time()
    files, links, dirs = inventory()
    print('INVENTORY', len(files), 'files', sum(x['bytes'] for x in files.values()), 'bytes', flush=True)
    protected_hashes, refs, scanned = provenance(files)
    print('PROVENANCE', scanned, 'text files', len(protected_hashes), 'hash tokens', flush=True)
    candidates, skipped, backups = {}, [], {}
    uv_prefix = '.cache/uv/'
    uv_map = {}
    # Byte-exact offline reconstruction of unpacked wheel files from installed copies.
    for rel in files:
        if not rel.startswith(uv_prefix + 'archive-v0/'):
            continue
        p = ROOT / rel
        sub = Path(*Path(rel).parts[4:])
        if p.name == 'RECORD' and p.parent.name.endswith('.dist-info'):
            backup = OUT / 'uv_wheel_records' / Path(rel).parts[3] / 'RECORD'
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(p.read_bytes())
            backups[rel] = backup.relative_to(ROOT).as_posix()
            uv_map[rel] = backups[rel]
            continue
        copies = [ROOT / env / 'Lib/site-packages' / sub for env in ('.venv', '.venv-real')]
        if '.data' in sub.parts[0] and len(sub.parts) > 2 and sub.parts[1] == 'data':
            copies += [ROOT / env / Path(*sub.parts[2:]) for env in ('.venv', '.venv-real')]
        h = digest(p)
        match = next((q for q in copies if q.is_file() and q.stat().st_size == files[rel]['bytes'] and digest(q) == h), None)
        if match is None:
            raise RuntimeError('Unmatched uv wheel payload: ' + rel)
        uv_map[rel] = match.relative_to(ROOT).as_posix()
    for rel, info in files.items():
        p = ROOT / rel
        category, rebuild = None, None
        if rel.startswith(uv_prefix) and info['bytes'] > 0 and p.name not in ('.lock', '.gitignore', 'CACHEDIR.TAG'):
            category = 'uv_package_cache' if rel in uv_map else 'uv_index_cache'
            rebuild = uv_map.get(rel, 'Regenerated by uv from package index/interpreter; installed environments retained')
        elif '__pycache__' in Path(rel).parts and p.suffix == '.pyc':
            try:
                import importlib.util
                source = Path(importlib.util.source_from_cache(str(p)))
            except ValueError:
                skipped.append(dict(path=rel, reason='No standard source mapping'))
                continue
            source_rel = source.relative_to(ROOT).as_posix()
            if source_rel not in files:
                skipped.append(dict(path=rel, reason='Missing source; preserve possible sourceless code'))
                continue
            category, rebuild = 'python_bytecode', source_rel
        elif rel == '.cache/matplotlib/fontlist-v3.11.0.json':
            category, rebuild = 'matplotlib_font_index', 'matplotlib.font_manager._load_fontmanager(try_read_cache=False)'
        if category is None:
            continue
        h = digest(p)
        norm = rel.lower()
        mentioned = any(norm in ref or (ref.endswith('.pyc') and norm.endswith('/' + ref)) for ref in refs)
        if h in protected_hashes or mentioned:
            skipped.append(dict(path=rel, reason='Referenced by provenance hash/path; retained'))
            continue
        candidates[rel] = dict(info, sha256=h, category=category, rebuild_from=rebuild)
    # Keep the entire package if even a small marker has a provenance hash hit.
    protected_archives = {Path(x['path']).parts[3] for x in skipped
                          if x['reason'].startswith('Referenced') and x['path'].startswith(uv_prefix + 'archive-v0/')}
    for rel in list(candidates):
        if rel.startswith(uv_prefix + 'archive-v0/') and Path(rel).parts[3] in protected_archives:
            candidates.pop(rel)
            skipped.append(dict(path=rel, reason='Keep complete archive containing protected marker'))
    uv_map = {rel: source for rel, source in uv_map.items() if rel in candidates}
    by_identity = collections.defaultdict(list)
    for rel, info in candidates.items():
        by_identity[(info['device'], info['inode'])].append(info)
    reclaim = sum(group[0]['bytes'] for group in by_identity.values() if len(group) == group[0]['nlink'])
    retained = {rel: info for rel, info in files.items() if rel not in candidates}
    def hash_record(item):
        rel, info = item
        return rel, dict(info, sha256=digest(ROOT / rel))
    print('HASHING_RETAINED', len(retained), 'files; full data/results/code/runtime baseline', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        retained = dict(pool.map(hash_record, retained.items()))
    save('retained_before.json', retained)
    summary = collections.defaultdict(lambda: dict(files=0, bytes=0))
    for info in candidates.values():
        summary[info['category']]['files'] += 1
        summary[info['category']]['bytes'] += info['bytes']
    plan = dict(workspace=str(ROOT), created_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                policy='Delete only verified rebuildable unreferenced caches; preserve all data/results/code and reparse points',
                files=candidates, empty_dir_candidates=sorted([d for d in dirs if '__pycache__' in Path(d).parts or d.startswith(uv_prefix)], key=lambda d: -len(Path(d).parts)),
                preserved_reparse_points=links, skipped=skipped, uv_archive_rebuild_map=uv_map,
                summary=dict(summary), logical_bytes=sum(x['bytes'] for x in candidates.values()),
                last_link_payload_bytes_estimate=reclaim, provenance_text_files_scanned=scanned,
                retained_files=len(retained), preparation_seconds=round(time.time()-started, 2))
    save('plan.json', plan)
    print(json.dumps({k: plan[k] for k in ('summary', 'logical_bytes', 'last_link_payload_bytes_estimate', 'retained_files', 'preparation_seconds')}, indent=2), flush=True)


def verify():
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    before = json.loads((OUT/'retained_before.json').read_text(encoding='utf-8'))
    def check(item):
        rel, info = item
        p = native_path(ROOT / rel)
        if not p.is_file(): return (rel, 'missing')
        if p.stat().st_size != info['bytes'] or digest(p) != info['sha256']: return (rel, 'changed')
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        failures = [x for x in pool.map(check, before.items()) if x]
    remaining = [p for p in plan['files'] if native_path(ROOT/p).exists()]
    uv_rebuild_failures = []
    for original, copy in plan['uv_archive_rebuild_map'].items():
        expected = plan['files'][original]['sha256']
        if digest(ROOT/copy) != expected:
            uv_rebuild_failures.append(original)
    current, links, _ = inventory()
    added = sorted(set(current) - set(before) - set(plan['files']))
    result = dict(complete=not failures and not remaining and not uv_rebuild_failures,
                  retained_files_checked=len(before), retained_hash_failures=failures,
                  deletion_targets_remaining=remaining, unexpected_added_files=added,
                  reparse_points_unchanged=sorted(links)==sorted(plan['preserved_reparse_points']),
                  uv_archive_files_offline_rebuildable=len(plan['uv_archive_rebuild_map']), uv_rebuild_failures=uv_rebuild_failures,
                  logical_bytes_removed=sum(x['bytes'] for p,x in plan['files'].items() if p not in remaining),
                  last_link_payload_bytes_estimate=plan['last_link_payload_bytes_estimate'])
    save('verification.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if not result['complete'] or added or not result['reparse_points_unchanged']:
        raise RuntimeError('Cleanup verification requires review')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'verify'])
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else verify()
