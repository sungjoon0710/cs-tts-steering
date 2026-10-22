"""Full automatic analysis, authorized independently of pending human validation.

Writes ONLY to --run, reads audio in place, never launches ASR/TTS/GPU inference.
The original pilot CLI and results remain separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,read_jsonl,utc_now,write_json,write_jsonl
from src.evaluation.records import import_snapshot,select_pilot
from src.evaluation.expanded import VERSION,content_proxies,expanded_audio
from src.evaluation.text import NORMALIZATION_VERSION

MODELS=('indic-speak','orpheus-hi-ft','orpheus-hi-pt')
FILES=('utterances.jsonl',*(m+'/generations.jsonl' for m in MODELS),
       'eval/asr.jsonl','eval/metrics.csv','eval/summary.csv','eval/summary.md','eval.log',*(m+'.log' for m in MODELS))


def source_snapshot(baseline,run):
    destination=run/'source';destination.mkdir(parents=True,exist_ok=True)
    hashes={}
    for rel in FILES:
        source=baseline/rel
        if not source.exists():continue  # model not generated yet
        data=source.read_bytes();target=destination/rel
        if target.exists() and target.read_bytes()!=data:raise ValueError('Baseline metadata changed; use a new run')
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[rel]=digest(data)
    inventory=[]
    for model in MODELS:
        if not (baseline/model).is_dir():continue
        for voice in sorted((baseline/model).iterdir()):
            if not voice.is_dir():continue
            for path in sorted(voice.glob('*.wav')):
                if path.is_symlink():raise ValueError('Audio symlink requires explicit review')
                st=path.stat();r={'relative_path':path.relative_to(baseline).as_posix(),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
                try:
                    with wave.open(str(path),'rb') as w:r.update(status='header_ok',sample_rate=w.getframerate(),channels=w.getnchannels(),sample_width_bytes=w.getsampwidth(),frames=w.getnframes(),duration_s=w.getnframes()/w.getframerate())
                except (OSError,wave.Error,EOFError) as e:r.update(status='unreadable_header',error_type=type(e).__name__)
                inventory.append(r)
    existing=destination/'audio_inventory.jsonl'
    if existing.exists() and read_jsonl(existing)!=inventory:raise ValueError('Source inventory changed; use a new run')
    write_jsonl(existing,inventory);hashes['audio_inventory.jsonl']=digest(existing.read_bytes())
    receipt=destination/'snapshot.json'
    if not receipt.exists():write_json(receipt,{'source':str(baseline),'copied_at_utc':utc_now(),'sha256':hashes,'audio_access':'read in place; no copied WAVs'})
    return destination


def main(baseline,run,config,cache_dir):
    started=time.time();run=private_directory(run);run.mkdir(parents=True,exist_ok=True)
    baseline=Path(baseline).resolve()
    if run==baseline or baseline in run.parents or run in baseline.parents:raise ValueError('Output must be separate from source baseline')
    if not (baseline/'utterances.jsonl').is_file():raise ValueError('No baseline manifest')
    lock=run/'RUNNING'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        source=source_snapshot(baseline,run)
        utterances,attempts,generations,asr=import_snapshot(source)
        by_id={u['utterance_id']:u for u in utterances}
        settings=json.loads(Path(config).read_text(encoding='utf-8'))
        if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=settings:raise ValueError('Configuration changed; use a new run')
        write_json(run/'config.json',settings)
        pilot=select_pilot(utterances)
        legacy=list(csv.DictReader((source/'eval/metrics.csv').open(encoding='utf-8')))
        legacy_index={(r['line'],r['model']+'/'+r['voice']):r for r in legacy}
        if len(legacy_index)!=len(legacy):raise ValueError('Duplicate legacy metric key')
        warnings_ids={r['line'] for r in legacy if r['runaway'].lower()=='true' or r['too_short'].lower()=='true'}
        challenge_ids=warnings_ids-set(pilot['utterance_ids'])
        write_json(run/'cohorts.json',{'baseline_ids':sorted(by_id,key=int),'original_pilot_ids':pilot['utterance_ids'],
            'challenge_view_ids':sorted(challenge_ids,key=int),'all_legacy_warning_ids':sorted(warnings_ids,key=int),
            'relationship':'challenge view overlaps baseline; never add its recordings to baseline denominator'})
        for name,rows in (('utterances',utterances),('attempts',attempts),('generations',generations)):
            write_jsonl(run/'tables'/(name+'.jsonl'),rows)
        packages={}
        for name in ('numpy','scipy','soundfile','praat-parselmouth','rapidfuzz','jiwer','matplotlib','uroman','opensmile','pandas','audinterface','audformat','audresample'):
            try:packages[name]=metadata.version(name)
            except metadata.PackageNotFoundError:packages[name]=None
        cache_dir=Path(cache_dir);cache_dir.mkdir(parents=True,exist_ok=True)
        code_root=Path(__file__).resolve().parents[1]
        extractor_hashes={name:digest((code_root/'src/evaluation'/name).read_bytes()) for name in ('audio.py','expanded.py')}
        rows=[];cache_hits=0
        for index,g in enumerate(generations):
            uid=g['utterance_id'];u=by_id[uid];path=baseline/g['relative_audio_path']
            filehash=digest(path.read_bytes()) if path.is_file() else 'missing'
            signature=digest(json.dumps({'sha256':filehash,'version':VERSION,'extractor_hashes':extractor_hashes,'packages':packages,'egemaps':settings['egemaps']},sort_keys=True).encode())
            feature_file=cache_dir/(signature+'.json')
            if feature_file.exists():audio=json.loads(feature_file.read_text(encoding='utf-8'));cache_hits+=1
            else:
                audio=expanded_audio(path,egemaps=settings['egemaps']);write_json(feature_file,audio)
            if path.exists() and audio.get('audio_sha256')!=filehash:raise ValueError('Audio changed during extraction')
            hyp=asr.get(g['remote_audio_path']);has_audio=audio['status'] not in ('missing_audio','unreadable_audio','empty_audio','nonfinite_audio','unsupported_channels')
            strict,raw,harmonized,proxies=content_proxies(u['text_raw'],hyp if has_audio else None,audio)
            lr=legacy_index.get((uid,g['configuration_id']))
            if lr is not None and (lr['text']!=u['text_raw'] or lr['hyp']!=hyp):raise ValueError('Legacy ASR/reference association mismatch')
            flags={'missing_audio':audio['status']=='missing_audio','unreadable_audio':audio['status'] in ('unreadable_audio','nonfinite_audio'),
                'empty_audio':audio['status']=='empty_audio','fully_silent':audio['status']=='fully_silent',
                'near_silent_warning':audio.get('rms_dbfs') is not None and audio['rms_dbfs']<settings['near_silent_rms_dbfs_warning'],
                'clipping_warning':bool(audio.get('clipped_sample_fraction',0)>0),'token_limit_warning':bool(g['hit_token_limit']),
                'legacy_short_warning':lr['too_short'].lower()=='true' if lr else None,
                'excessive_duration_warning':float(lr['dur_ratio'])>settings['excessive_duration_ratio_warning'] if lr else None,
                'asr_omission_warning':proxies['asr_deletion_fraction'] is not None and proxies['asr_deletion_fraction']>=settings['asr_deletion_fraction_warning'],
                'asr_insertion_warning':proxies['asr_insertion_fraction'] is not None and proxies['asr_insertion_fraction']>=settings['asr_insertion_fraction_warning'],
                'asr_repetition_warning':proxies['asr_repeated_trigram_fraction'] is not None and proxies['asr_repeated_trigram_fraction']>=settings['asr_repeated_trigram_fraction_warning']}
            rows.append({'generation_id':g['generation_id'],'utterance_id':uid,'configuration_id':g['configuration_id'],
                'cohort':'baseline','in_original_pilot':uid in pilot['utterance_ids'],'in_challenge_view':uid in challenge_ids,
                'text_raw':u['text_raw'],'asr_raw':hyp,'script_profile':u['script_profile'],'input_word_count':u['normalized_words'],
                'audio':audio,'automatic_proxy':strict,'raw_script_sensitive':raw,'uroman_sensitivity':harmonized,
                'asr_indicators':proxies,'automatic_warnings':flags,'manual_failure_status':'pending',
                'language':{'status':'pending_verified_or_provisional_annotations'},'poi':{'status':'pending_annotations','pier':None},
                'source_audio_path':str(path),'selected_attempt_id':g['attempt_id'],'attempt_count':g['attempt_count'],
                'successful_retry':g['successful_retry'],'metric_version':VERSION,'human_validation':'pending',
                'asr_checkpoint':'upstream source identifies openai/whisper-large-v3; legacy cache revision unverified','checkpoint_revision':None})
            if (index+1)%25==0:print(f'CPU evaluation {index+1}/{len(generations)}; feature cache hits {cache_hits}',flush=True)
        write_jsonl(run/'tables/metrics.jsonl',rows)
        write_jsonl(run/'views/challenge_metrics.jsonl',[r for r in rows if r['in_challenge_view']])
        write_jsonl(run/'annotations/manual_validation_queue.jsonl',[{'generation_id':r['generation_id'],'audio_sha256':r['audio'].get('audio_sha256'),
            'priority_warning_count':sum(v is True for v in r['automatic_warnings'].values()),'status':'pending'} for r in
            sorted(rows,key=lambda r:sum(v is True for v in r['automatic_warnings'].values()),reverse=True)])
        repo=Path(__file__).resolve().parents[1]
        sources={p.relative_to(repo).as_posix():digest(p.read_bytes()) for p in sorted((repo/'src/evaluation').glob('*')) if p.is_file()}
        for p in sorted((repo/'scripts').glob('*.py')):sources[p.relative_to(repo).as_posix()]=digest(p.read_bytes())
        write_json(run/'provenance.json',{'metric_version':VERSION,'source_hashes':sources,'config_sha256':digest(Path(config).read_bytes()),
            'source_snapshot_sha256':digest((source/'snapshot.json').read_bytes()),'packages':packages,'python':sys.version,
            'platform':platform.platform(),'device':'CPU','hostname':platform.node(),'executed_at_utc':utc_now(),
            'elapsed_s':time.time()-started,'feature_cache_hits':cache_hits,'recordings':len(rows),'new_asr_runs':0,'tts_runs':0,
            'status':'FULL_AUTOMATIC_ANALYSIS; PROVISIONAL_AND_EXPLORATORY; HUMAN_RESULTS_PENDING'})
        from src.evaluation.full_report import full_report
        full_report(run)
        write_json(run/'completion.json',{'status':'complete','recordings':len(rows),'finished_at_utc':utc_now(),'runtime_s':time.time()-started})
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/evaluation_full.json')
    p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();main(a.baseline,a.run,a.config,a.cache)
