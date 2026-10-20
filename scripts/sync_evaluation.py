"""Immutable code releases and verified, non-destructive private-result transfer.

Uses the existing SSH alias. No credentials, models, or baseline WAV collections
are transferred. All remote writes are restricted to the caller's own workspace.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,write_json

REMOTE_PREAMBLE='''import os, pathlib, json, hashlib, base64, io, zipfile, sys
root=pathlib.Path.home()/'speechlab_tts_eval'
if not root.is_dir() or root.stat().st_uid!=os.getuid(): raise ValueError('Not an owned workspace')
def safe(rel):
 p=pathlib.PurePosixPath(rel)
 if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('Unsafe relative path')
 return p
def owned(p):
 if root.resolve() not in p.resolve().parents: raise ValueError('Outside workspace')
 for q in [p,*p.parents]:
  if q==root.parent: break
  if q.exists() and (q.is_symlink() or q.stat().st_uid!=os.getuid()): raise ValueError('Unowned or symlink path')
'''

def ssh(code,payload=b''):
    command='python3 -c '+shlex.quote(REMOTE_PREAMBLE+'\n'+code)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','kedi',command],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')[-4000:])
    return result.stdout

def release_bundle(repo):
    files={}
    patterns=('src/__init__.py','src/evaluation/*.py','src/evaluation/*.html','scripts/*.py','configs/*.json','tests/*.py','docs/EVALUATION*.md','requirements-evaluation.txt')
    for pattern in patterns:
        for path in sorted(repo.glob(pattern)):
            if path.name in ('build_full_handoff.py','EVALUATION_STATUS.md','EVALUATION_IMPLEMENTATION.md','EVALUATION_EXPLORATORY_REVIEW.md'):
                continue  # Local GPT/report tooling and account-specific audit notes.
            if path.is_file():files[path.relative_to(repo).as_posix()]=path.read_bytes()
    hashes={k:digest(v) for k,v in sorted(files.items())}
    release=digest(json.dumps(hashes,sort_keys=True).encode())[:20]
    files['release_manifest.json']=json.dumps({'release_id':release,'sha256':hashes},sort_keys=True).encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    return release,stream.getvalue(),hashes

def unpack_results(data,destination):
    destination=private_directory(destination);destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive paths')
        manifest=json.loads(archive.read('transfer_manifest.json'))
        if set(names)-{'transfer_manifest.json'}!=set(manifest['sha256']):raise ValueError('Inventory mismatch')
        # Validate every path/hash/conflict before writing any content.
        for name in names:
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:raise ValueError('Unsafe archive path')
            target=destination/p
            if destination not in target.resolve().parents or any(q.is_symlink() for q in [target,*target.parents]):raise ValueError('Unsafe destination')
            content=archive.read(name)
            if name!='transfer_manifest.json' and digest(content)!=manifest['sha256'][name]:raise ValueError('Checksum mismatch')
            if target.exists() and target.read_bytes()!=content:raise ValueError('Existing result differs; choose a new destination')
        for name in names:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(archive.read(name))
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['push','run','compare','status','pull','research-push'])
    p.add_argument('--release');p.add_argument('--run-id',default='baseline_full_20261019');p.add_argument('--destination',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--artifact-id',default='natural_reference_v2')
    p.add_argument('--baseline-run-id',default='baseline_full_20261019_v2')
    p.add_argument('--include-reports',action='store_true',help='Also sync report MD/HTML/CSV/PNG and subdirectories; excludes GPT/HANDOFF files and audio/models')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):raise ValueError('Invalid run ID')
    if a.release is not None and not re.fullmatch('[0-9a-f]{20}',a.release):raise ValueError('Invalid release')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in (a.artifact_id,a.baseline_run_id)):raise ValueError('Invalid artifact/run ID')
    if a.action=='push':
        release,data,hashes=release_bundle(Path(__file__).resolve().parents[1])
        code='''z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read()))
m=json.loads(z.read('release_manifest.json')); dest=root/'releases'/m['release_id']; owned(dest)
if set(z.namelist())-{'release_manifest.json'}!=set(m['sha256']): raise ValueError('Inventory mismatch')
for name in z.namelist():
 p=dest/str(safe(name)); owned(p); data=z.read(name)
 if name!='release_manifest.json' and hashlib.sha256(data).hexdigest()!=m['sha256'][name]: raise ValueError('Checksum mismatch')
 if p.exists() and p.read_bytes()!=data: raise ValueError('Immutable release conflict')
for name in z.namelist():
 p=dest/str(safe(name)); p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): p.write_bytes(z.read(name))
print(json.dumps({'release_id':m['release_id'],'files':len(m['sha256']),'remote_path':str(dest)}))
'''
        print(ssh(code,data).decode().strip())
        if a.destination:write_json(private_directory(a.destination)/'release_receipt.json',{'release_id':release,'sha256':hashes})
    elif a.action=='research-push':
        if a.source is None:raise ValueError('--source required')
        suffixes=('.json','.jsonl','.md','.html','.csv','.png') if a.include_reports else ('.json','.jsonl')
        source=private_directory(a.source)
        paths=source.rglob('*') if a.include_reports else source.iterdir()
        files={p.relative_to(source).as_posix():p.read_bytes() for p in paths if p.is_file() and not p.is_symlink() and source in p.resolve().parents and p.suffix in suffixes and not any(part.upper().startswith(('GPT_','HANDOFF')) for part in p.relative_to(source).parts)}
        manifest={'sha256':{k:digest(v) for k,v in files.items()},'artifact_id':a.artifact_id}
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,data in files.items():z.writestr(name,data)
            z.writestr('artifact_manifest.json',json.dumps(manifest,sort_keys=True))
        code='''z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); m=json.loads(z.read('artifact_manifest.json'))
dest=root/'research'/m['artifact_id']; owned(dest)
if set(z.namelist())-{'artifact_manifest.json'}!=set(m['sha256']): raise ValueError('Inventory mismatch')
for name in z.namelist():
 p=dest/str(safe(name)); owned(p); data=z.read(name)
 if name!='artifact_manifest.json' and hashlib.sha256(data).hexdigest()!=m['sha256'][name]: raise ValueError('Checksum mismatch')
 if p.exists() and p.read_bytes()!=data: raise ValueError('Immutable artifact conflict')
for name in z.namelist():
 p=dest/str(safe(name));p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():p.write_bytes(z.read(name))
print(json.dumps({'artifact_path':str(dest),'verified_files':len(m['sha256']),'raw_audio_transferred':0}))
'''
        print(ssh(code,buf.getvalue()).decode().strip())
    elif a.action in ('run','compare'):
        if not a.release:raise ValueError('--release required')
        code=f'''import subprocess
release=root/'releases'/{a.release!r}; owned(release)
manifest=json.loads((release/'release_manifest.json').read_text())
for name,expected in manifest['sha256'].items():
 if hashlib.sha256((release/name).read_bytes()).hexdigest()!=expected: raise ValueError('Changed release')
run=root/'runs'/{a.run_id!r}; owned(run)
if {a.action!r}=='compare' and not (root/'runs'/{a.baseline_run_id!r}/'completion.json').exists(): raise ValueError('Baseline incomplete; comparison not launched')
if run.exists(): raise ValueError('Run already exists; use a new run ID')
run.mkdir()
env=os.environ.copy(); env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(root/'cache'),MPLCONFIGDIR=str(root/'cache/matplotlib'))
cmd=[str(root/'env/bin/python'),str(release/'scripts/full_evaluate.py'),'--baseline','/local/speech/users/sp4050/outputs/baseline','--run',str(run),'--cache',str(root/'cache/features')]
if {a.action!r}=='compare':
 cmd=[str(root/'env/bin/python'),str(release/'scripts/natural_distribution.py'),'compare','--features',str(root/'research'/{a.artifact_id!r}),'--baseline',str(root/'runs'/{a.baseline_run_id!r}),'--output',str(run)]
with (run/'execution.log').open('xb') as log:
 process=subprocess.Popen(cmd,cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'launch.json').write_text(json.dumps({{'pid':process.pid,'release_id':{a.release!r},'command':cmd,'device':'CPU','thread_limit':2}},indent=2))
print(json.dumps({{'pid':process.pid,'run':str(run)}}))
'''
        print(ssh(code).decode().strip())
    elif a.action=='status':
        code=f'''run=root/'runs'/{a.run_id!r}; owned(run)
for name in ('launch.json','completion.json'):
 p=run/name
 if p.exists(): print(name+': '+p.read_text())
p=run/'execution.log'
if p.exists(): print('Log tail:\\n'+'\\n'.join(p.read_text(errors='replace').splitlines()[-15:]))
'''
        print(ssh(code).decode().strip())
    else:
        if a.destination is None:raise ValueError('--destination required')
        code=f'''run=root/'runs'/{a.run_id!r}; owned(run)
if not (run/'completion.json').exists(): raise ValueError('Run incomplete')
files={{}}
for p in sorted(run.rglob('*')):
 if p.is_file():
  owned(p); rel=p.relative_to(run).as_posix()
  if p.suffix=='.wav' and not rel.startswith('report/audio/'): continue
  if p.suffix not in ('.json','.jsonl','.csv','.md','.html','.png','.log','.wav'): continue
  files[rel]=p.read_bytes()
m={{'run_id':{a.run_id!r},'remote_path':str(run),'sha256':{{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}}}}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for name,data in files.items(): z.writestr(name,data)
 z.writestr('transfer_manifest.json',json.dumps(m,sort_keys=True))
sys.stdout.buffer.write(buf.getvalue())
'''
        result=unpack_results(ssh(code),a.destination)
        print(json.dumps({'verified_files':len(result['sha256']),'destination':str(a.destination),'remote_path':result['remote_path']}))

if __name__=='__main__':main()
