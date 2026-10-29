"""Pinned, resumable generation for the independent local-audio-linked experiment."""
import argparse
import gc
import importlib.metadata as metadata
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest, read_jsonl, stable_id, write_json, write_jsonl
from src.tts_settings import token_budget

# Keep the original matched-cohort IDs for resume. Its historical "orpheus-hi"
# identifier means the FT checkpoint, not the newly added pretrained model.
PINS = {'indic-speak': ('bodhan-ai/indic-speak','76e0814f189321efe550011850cd288dae366aad'),
        'orpheus-hi': ('canopylabs/3b-hi-ft-research_release','70c4b32af539cac4a3e2856a6fd4fab6ff8c446b'),
        'codec': ('hubertsiuzdak/snac_24khz','d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec')}
DECODING = {'indic-speak':dict(temperature=.6,top_p=.9,top_k=50,repetition_penalty=1.2),
            'orpheus-hi':dict(temperature=.6,top_p=.9,repetition_penalty=1.1)}


def jobs(cohort):
    result = []
    for row in cohort:
        budget=max(token_budget(row['input_romanized']),token_budget(row['input_devanagari']))
        for model, voice, condition in [('indic-speak','Amit','devanagari'),('indic-speak','Amit','romanized'),
                ('indic-speak','Kavya','devanagari'),('indic-speak','Kavya','romanized'),
                ('orpheus-hi','ऋतिका','devanagari')]:
            text = row['input_' + condition]
            config = f'{model}/{voice}/{condition}'
            gid = stable_id('localdrive_matched_v1', row['utterance_id'], config,
                digest(text.encode()), digest(row['canonical_text'].encode()), PINS[model][1], '1234',
                json.dumps(DECODING[model],sort_keys=True),budget)
            result.append({'generation_id':gid,'utterance_id':row['utterance_id'],
                'configuration_id':config,'model':model,'voice':voice,'script_condition':condition,
                'input_text':text,'canonical_text':row['canonical_text'],
                'canonical_text_sha256':digest(row['canonical_text'].encode()),
                'checkpoint':PINS[model][0],'revision':PINS[model][1],'codec_revision':PINS['codec'][1],
                'seed':1234,'decoding':DECODING[model],'max_new_tokens':budget,'pilot':row['pilot'],
                'relative_audio_path':f'generated/{model}/{voice}/{condition}/{gid}.wav',
                'text_validation':'provisional_not_human_verified','planned_status':'not_attempted'})
    return result


def prefetch(project):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    result = {}; cache = project / 'model_cache'; cache.mkdir(exist_ok=True)
    for key, (repo, revision) in PINS.items():
        try:
            # Test access before starting a large checkpoint transfer. Standard
            # library authentication is used; no credential is read or printed here.
            hf_hub_download(repo,'config.json',revision=revision,cache_dir=cache)
            info = HfApi().model_info(repo,revision=revision,files_metadata=True)
            names = [s.rfilename for s in info.siblings]
            has_safe = any(n.endswith('.safetensors') for n in names)
            selected = [n for n in names if n.endswith(('.json','.txt','.model','.py','.yaml','.yml','.md'))
                or n.endswith('.safetensors') or n.startswith('vocos/') and n.endswith('.pt')
                or (not has_safe or key=='codec') and n.endswith('.bin') and Path(n).name.startswith(('pytorch_model','model'))]
            selected = [n for n in selected if not any(part in n.lower() for part in ['optimizer','scheduler','rng_state','training_args'])]
            sizes = {s.rfilename:s.size for s in info.siblings if s.rfilename in selected}
            write_json(project / f'download_plan_{key}.json',{'model':repo,'revision':revision,'files':sizes,
                'known_bytes':sum(v for v in sizes.values() if v is not None)})
            path = snapshot_download(repo,revision=revision,cache_dir=cache,allow_patterns=selected,max_workers=2)
            # The vendor SNAC loader requests its named repository. Pin that
            # lookup in this experiment's own cache and generate offline.
            if key=='codec':
                ref=Path(path).parent.parent/'refs/main';ref.parent.mkdir(exist_ok=True)
                if ref.exists() and ref.read_text().strip()!=revision:raise ValueError('Codec cache revision conflict')
                ref.write_text(revision)
            result[key] = {'status':'available','path':str(path),'repo':repo,'revision':revision}
        except Exception as exc:
            result[key] = {'status':'blocked','repo':repo,'revision':revision,'error_type':type(exc).__name__,
                'http_status':getattr(getattr(exc,'response',None),'status_code',None)}
        write_json(project / 'model_access.json',result)
    return result


def generate(a):
    import numpy as np
    import soundfile as sf
    import torch
    from src.tts import IndicSpeak, OrpheusHindi, token_budget
    if not a.allow_provisional_text:
        raise ValueError('Text equivalence is not human verified; explicitly authorize provisional inputs before generation')
    if not json.loads((a.run/'experiment.json').read_text()).get('provisional_text_policy_approved'):
        raise ValueError('Experiment text-policy approval is not recorded')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no unapproved CPU generation fallback')
    access = json.loads((a.project / 'model_access.json').read_text())
    selected = [r for r in read_jsonl(a.run / 'planned_generations.jsonl')
        if r['model'] == a.model and (a.scope == 'full' or r['pilot'])]
    if access.get(a.model,{}).get('status') != 'available' or access.get('codec',{}).get('status') != 'available':
        raise RuntimeError('Required pinned model/codec unavailable; inspect model_access.json')
    log = a.run / 'generation_attempts.jsonl'
    previous = read_jsonl(log) if log.exists() else []
    successful = {r['generation_id']:r for r in previous if r['status'] == 'generated'}
    selected = [r for r in selected if r['generation_id'] not in successful]
    if a.limit is not None:selected=selected[:a.limit]
    for gid, r in successful.items():
        path = a.run / r['relative_audio_path']
        if not path.is_file() or digest(path.read_bytes()) != r['audio_sha256']:
            raise ValueError('Existing successful generation changed: ' + gid)
    if not selected:
        print('All requested recordings already generated and hash-verified'); return
    started = time.time(); torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(.9,18*1024**3/torch.cuda.get_device_properties(0).total_memory))
    cls = IndicSpeak if a.model == 'indic-speak' else OrpheusHindi
    kwargs = {'snapshot':access[a.model]['path']}
    if a.model == 'orpheus-hi':
        kwargs['codec_snapshot'] = access['codec']['path']
    model = cls(**kwargs)
    consecutive_errors = 0
    for i, row in enumerate(selected):
        if time.time() - started > a.max_seconds:
            raise TimeoutError('Generation stage time limit reached; existing successes preserved')
        path = a.run / row['relative_audio_path']; path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise ValueError('Unlogged audio exists; refusing to overwrite: ' + str(path))
        attempt = 1 + sum(r['generation_id'] == row['generation_id'] for r in previous)
        record = dict(row,attempt=attempt,started_epoch=time.time(),device=torch.cuda.get_device_name(0))
        try:
            wave, info = model(row['input_text'],row['voice'],row['seed'],max_new_tokens=row['max_new_tokens'])
            if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError('Invalid generated waveform')
            sf.write(path,wave,24000,subtype='PCM_16')
            record.update(info,status='generated',audio_sha256=digest(path.read_bytes()),
                fully_silent=not bool(np.any(wave)),runtime_s=time.time()-record['started_epoch'])
            consecutive_errors = 0
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__,error=str(exc)[:500])
            consecutive_errors += 1
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False) + '\n'); f.flush()
        previous.append(record)
        print(json.dumps({'completed_in_stage':i+1,'stage_total':len(selected),'generation_id':row['generation_id'],
            'configuration_id':row['configuration_id'],'status':record['status'],'elapsed_s':time.time()-started}),flush=True)
        if consecutive_errors >= 3:
            raise RuntimeError('Three consecutive generation failures; stage stopped for inspection')
    del model; gc.collect(); torch.cuda.empty_cache()
    write_json(a.run / f'generation_{a.model}_{a.scope}_completion.json',{'finished_epoch':time.time(),
        'elapsed_s':time.time()-started,'packages':{p:metadata.version(p) for p in ['torch','transformers','snac','soundfile']},
        'source_sha256':digest(Path(__file__).read_bytes()),'human_validation_complete':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['manifest','prefetch','generate'])
    parser.add_argument('--project',type=Path,required=True); parser.add_argument('--run',type=Path)
    parser.add_argument('--dataset',type=Path); parser.add_argument('--model',choices=['indic-speak','orpheus-hi'])
    parser.add_argument('--scope',choices=['pilot','full'],default='pilot')
    parser.add_argument('--max-seconds',type=int,default=1200)
    parser.add_argument('--limit',type=int,help='Bounded initial smoke generation; successes resume in the pilot/full stage')
    parser.add_argument('--allow-provisional-text',action='store_true')
    a = parser.parse_args()
    if a.action == 'prefetch':
        print(json.dumps(prefetch(a.project)))
    elif a.action == 'manifest':
        if a.run is None or a.dataset is None:parser.error('--run and --dataset required')
        a.run.mkdir(parents=True,exist_ok=True)
        path = a.run / 'planned_generations.jsonl'
        if path.exists():raise ValueError('Manifest exists; preserve it')
        write_jsonl(path,jobs(read_jsonl(a.dataset / 'cohort.jsonl')))
        write_json(a.run / 'experiment.json',{'dataset':str(a.dataset),'dataset_sha256':digest((a.dataset/'cohort.jsonl').read_bytes()),
            'seed':1234,'checkpoint_pins':PINS,'prompt_source':'src/tts.py; pinned official Indic-Speak build_prompt or explicit Orpheus voice prefix',
            'generation_authorized':True,'provisional_text_policy_approved':a.allow_provisional_text,
            'text_policy':'keep supplied spellings; provisional study; no silent correction'})
    else:
        if a.run is None or a.model is None:parser.error('--run and --model required')
        generate(a)
