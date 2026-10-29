"""Inspect/pull the new owned Kedi run; never transfer models or credentials."""
import argparse
import json
import os
import shutil
from pathlib import Path

from sync_evaluation import ssh, unpack_results
from src.evaluation.common import digest, read_jsonl, private_directory

PROJECT_EXPR="pathlib.Path('/local/speech/users')/__import__('pwd').getpwuid(os.getuid()).pw_name/'tts_localdrive_20261021'"
RUN='runs/v1_supplied_spellings'


def main(a):
    if a.action=='status':
        code=f'''import subprocess
project={PROJECT_EXPR};run=project/{RUN!r}
out={{}}
for name in ['setup_status_v2.json','asr_prefetch_status.json','model_access.json']:
 p=project/name
 if p.exists():out[name]=json.loads(p.read_text())
for name in ['cpu_status.json','pilot_pipeline_status.json','pilot_validation.json','full_pipeline_status.json','report/status.json']:
 p=run/name
 if p.exists():
  state={{k:v for k,v in json.loads(p.read_text()).items() if k!='records'}}
  if state.get('status')=='running':state['live_elapsed_s']=state.get('elapsed_s',0)+__import__('time').time()-p.stat().st_mtime
  out[name]=state
p=run/'generation_attempts.jsonl'
rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
from collections import Counter
out['generation_status_counts']=dict(Counter((r['configuration_id']+':'+r['status']) for r in rows))
out['last_attempts']=[{{k:r.get(k) for k in ['generation_id','configuration_id','status','audio_s','rtf','hit_max','error_type','error']}} for r in rows[-3:]]
print(json.dumps(out))
'''
        print(ssh(code).decode('utf-8'));return
    if a.output is None or a.dataset is None:raise ValueError('--output and --dataset required')
    destination=private_directory(a.output)
    if destination.exists():raise ValueError('Choose a fresh snapshot; prior versions remain untouched')
    code=f'''project={PROJECT_EXPR};run=project/{RUN!r}
if project.is_symlink() or project.stat().st_uid!=os.getuid():raise ValueError('Unowned project')
allowed={{'.json','.jsonl','.csv','.md','.html','.png','.log','.txt','.xml','.praat'}}
files={{}}
for p in run.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(run)
 if p.suffix not in allowed and not ({a.include_audio!r} and p.suffix=='.wav') and rel.as_posix() not in ['pilot_worker.py','full_worker.py','interim_cpu_worker.py','indic_complete_cpu_worker.py']:continue
 if any(part in ['model_cache','evaluation_model_cache','tts_env'] for part in rel.parts):raise ValueError('Forbidden transfer')
 if p.stat().st_uid!=os.getuid():raise ValueError('Unowned result file')
 files[{RUN!r}+'/'+rel.as_posix()]=p.read_bytes()
for name in ['cohort.jsonl','pilot.jsonl','dataset.json','source_json_crosscheck.json']:
 p=project/'datasets/dataset_v2'/name
 if p.exists():
  if p.is_symlink() or p.stat().st_uid!=os.getuid():raise ValueError('Unowned dataset metadata')
  files['datasets/dataset_v2/'+name]=p.read_bytes()
releases=set()
for name,key in [('full_pipeline_status.json','release'),('report/completion.json','source_release'),
 ('validation/full_report_before_diagnostics/completion.json','source_release')]:
 p=run/name
 if p.exists():
  value=json.loads(p.read_text()).get(key)
  if value:releases.add(pathlib.Path(value))
for release in releases:
 if release.parent!=project/'code' or release.is_symlink():raise ValueError('Invalid source release')
 manifest=release/'manifest.json';data=json.loads(manifest.read_text())
 for name,expected in data['sha256'].items():
  rel=pathlib.PurePosixPath(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe source path')
  p=release/name
  if p.is_symlink() or p.stat().st_uid!=os.getuid():raise ValueError('Unowned source file')
  content=p.read_bytes()
  if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Source release changed')
  files[{RUN!r}+'/provenance/source/'+release.name+'/'+name]=content
 files[{RUN!r}+'/provenance/source/'+release.name+'/manifest.json']=manifest.read_bytes()
if sum(len(v) for v in files.values())>1_500_000_000:raise ValueError('Unexpected result-transfer size')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for name,data in files.items():z.writestr(name,data)
 z.writestr('transfer_manifest.json',json.dumps({{'sha256':{{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}}}))
sys.stdout.buffer.write(buf.getvalue())
'''
    result=unpack_results(ssh(code),destination)
    rows=read_jsonl(a.dataset/'cohort.jsonl')
    for row in rows:
        original=a.dataset/row['natural_audio_relative_path']
        if digest(original.read_bytes())!=row['natural_sha256']:raise ValueError('Local natural recording changed')
        target=destination/'datasets/dataset_v2'/row['natural_audio_relative_path'];target.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(original,target)
        except OSError:shutil.copyfile(original,target)
    for name in ['cohort.jsonl','pilot.jsonl','dataset.json']:
        target=destination/'datasets/dataset_v2'/name
        if target.exists():
            if target.read_bytes()!=(a.dataset/name).read_bytes():raise ValueError('Remote/local dataset metadata differ')
        else:shutil.copyfile(a.dataset/name,target)
    print(json.dumps({'snapshot':str(destination),'verified_result_files':len(result['sha256']),
        'natural_references_linked':len(rows),'model_weights_transferred':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','pull']);parser.add_argument('--output',type=Path)
    parser.add_argument('--dataset',type=Path);parser.add_argument('--include-audio',action='store_true')
    main(parser.parse_args())
