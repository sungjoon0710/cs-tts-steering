"""Stage 1 only: frozen style controls and alpha-zero hook readiness. No fitting."""
import argparse,csv,hashlib,html,importlib.metadata as metadata,json,os,sys,time,warnings
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,read_jsonl,stable_id,write_json
from src.tts_settings import AUDIO_BASE,CODEBOOK_SIZE,token_budget
from scripts.matched_generate import PINS

VENDOR_SHA='9be5d6391bc527a7ac626cb819494f98b486ac6b616e247e9ca74c4cae525ef5'
DECODING=dict(temperature=.6,top_p=.9,top_k=50,repetition_penalty=1.2)

def plan(rows):
    if len(rows)!=10 or len({r['utterance_id'] for r in rows})!=10:raise ValueError('Expected10 unique frozen inputs')
    if Counter(r['split'] for r in rows)!=Counter(construct=6,heldout=4):raise ValueError('Frozen split changed')
    if len({r['source_episode'] for r in rows})!=10:raise ValueError('Source episodes overlap')
    jobs=[]
    for r in rows:
        cap=max(token_budget(r['Romanized_input']),token_budget(r['Devanagari_input']))
        for voice in ['Amit','Kavya']:
            for script,column in [('devanagari','Devanagari_input'),('romanized','Romanized_input')]:
                for style in ['', 'ANGER']:
                    gid=stable_id('style_stage1_v1',r['utterance_id'],voice,script,style,r[column],cap,1234,PINS['indic-speak'][1],json.dumps(DECODING,sort_keys=True))
                    jobs.append({'generation_id':gid,'utterance_id':r['utterance_id'],'split':r['split'],'voice':voice,
                        'script':script,'style':style,'input_text':r[column],'canonical_text':r['Devanagari_input'],
                        'seed':1234,'max_new_tokens':cap,'model_revision':PINS['indic-speak'][1]})
    return jobs

def structure(tokens,eos,cap):
    span=[]
    for t in tokens:
        if not AUDIO_BASE<=t<AUDIO_BASE+7*CODEBOOK_SIZE:break
        span.append(t)
    frames=len(span)//7
    bad=sum(not all(AUDIO_BASE+p*CODEBOOK_SIZE<=span[i*7+p]<AUDIO_BASE+(p+1)*CODEBOOK_SIZE for p in range(7)) for i in range(frames))
    tail=tokens[len(span):];stopped=len(tail)==1 and tail[0] in eos
    return {'audio_tokens':len(span),'complete_frames':frames,'invalid_frames':bad,'incomplete_trailing_tokens':len(span)%7,
        'normal_EOS':stopped,'terminal_tokens':list(tail),'token_cap_reached':len(tokens)>=cap,
        'usable_complete_code_frames':frames-bad}

def prompt_checks(tok,text,voice,style,prompt,build):
    start,end=tok.convert_tokens_to_ids('<|style>'),tok.convert_tokens_to_ids('<style|>')
    speech=tok.convert_tokens_to_ids('<|start_of_speech|>');nl=tok.encode('\n',add_special_tokens=False)
    neutral=build(tok,text,voice,style='')
    if style:
        a=prompt.index(start);b=prompt.index(end)
        ok=prompt.count(start)==prompt.count(end)==1 and prompt[a+1:b]==tok.encode(style,add_special_tokens=False)
        ok=ok and a==neutral.index(tok.convert_tokens_to_ids('<speaker|>'))+1+len(nl)
        ok=ok and prompt[b+1:b+1+len(nl)]==nl and prompt[:a]+prompt[b+1+len(nl):]==neutral
    else:ok=start not in prompt and end not in prompt and prompt==neutral
    return {'style_placement_valid':bool(ok),'speaker_and_text_unchanged':bool(ok),'speech_start_is_last_prompt_token':prompt[-1]==speech}

def csvsave(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def review(run,rows,jobs,records):
    latest={r['generation_id']:r for r in records};sections=[]
    for u in rows:
        cells=[]
        for voice in ['Amit','Kavya']:
            for script in ['devanagari','romanized']:
                audio=[]
                for style in ['','ANGER']:
                    j=next(j for j in jobs if (j['utterance_id'],j['voice'],j['script'],j['style'])==(u['utterance_id'],voice,script,style))
                    r=latest.get(j['generation_id']);cell='Not generated'
                    if r and r['status']=='generated':cell='<audio controls preload="none" src="'+html.escape(r['audio_path'],quote=True)+'"></audio>'
                    elif r:cell=html.escape(r['status'])
                    audio.append('<td>'+cell+'</td>')
                cells.append('<tr><th>'+voice+' / '+script+'</th>'+''.join(audio)+'</tr>')
        sections.append('<section><h2>'+html.escape(u['utterance_id'])+' — '+u['split']+'</h2><p>'+html.escape(u['Romanized_input'])+'</p><p>'+html.escape(u['Devanagari_input'])+'</p><table><tr><th>Condition</th><th>No style tag</th><th>ANGER tag</th></tr>'+''.join(cells)+'</table></section>')
    page='''<!doctype html><meta charset="utf-8"><title>Stage 1 style controls</title><style>body{font:16px/1.5 system-ui;margin:24px}td,th{border:1px solid #ccc;padding:8px}audio{width:280px}table{border-collapse:collapse}section{margin:30px 0}</style><h1>Stage 1: no style versus ANGER</h1><p>Unblinded researcher review. No human listening or style-success labels have been recorded. Supplied transcripts and pronunciations are unverified. These are style-conditioned outputs, not validated emotional speech.</p>'''+''.join(sections)
    (run/'listen.html').write_text(page,encoding='utf-8')

def main(a):
    import numpy as np,soundfile as sf,torch
    from src.tts import IndicSpeak
    from src.representation_probe import ZeroSteeringProbe
    os.umask(0o077);run=a.run.resolve();baseline=a.baseline_project.resolve()
    if run==baseline or baseline in run.parents:raise ValueError('Use an independent output root')
    if run.stat().st_uid!=os.getuid() or run.is_symlink():raise ValueError('Use an owned real output directory')
    source=a.manifest.read_bytes();manifest_sha=digest(source)
    if manifest_sha!=a.manifest_sha256:raise ValueError('Frozen input manifest changed')
    with a.manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    jobs=plan(rows);smoke_id=next(r['utterance_id'] for r in rows if r['split']=='construct')
    cohort={r['utterance_id']:r for r in read_jsonl(baseline/'datasets/dataset_v2/cohort.jsonl')}
    for r in rows:
        old=cohort[r['utterance_id']]
        if (r['Romanized_input'],r['Devanagari_input'])!=(old['input_romanized'],old['input_devanagari']):raise ValueError('Input differs from frozen cohort')
    snap=a.snapshot;codec=a.codec_snapshot
    if snap.name!=PINS['indic-speak'][1] or codec.name!=PINS['codec'][1]:raise ValueError('Checkpoint revision changed')
    if digest((snap/'inference.py').read_bytes())!=VENDOR_SHA:raise ValueError('Vendor inference source changed')
    if (codec.parent.parent/'refs/main').read_text().strip()!=PINS['codec'][1]:raise ValueError('Codec cache reference changed')
    for name in ['tokens','checks','logs','audio']:(run/name).mkdir(exist_ok=True)
    config={'stage':'style_stage1_v1','manifest_SHA256':manifest_sha,'model':PINS['indic-speak'],'codec':PINS['codec'],
        'vendor_inference_SHA256':VENDOR_SHA,'decoding':DECODING,'seed':1234,'smoke_input':smoke_id,'planned':80,
        'nonzero_steering_allowed':False,'vector_fitting_allowed':False,'baseline_project':str(baseline),
        'source_SHA256':{str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p.read_bytes()) for p in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/tts.py',Path(__file__).resolve().parents[1]/'src/representation_probe.py']}}
    if (run/'config.json').exists():
        if not a.resume or json.loads((run/'config.json').read_text())!=config:raise ValueError('Existing run or changed config')
    else:
        write_json(run/'config.json',config);(run/'planned_generations.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs),encoding='utf-8')
    historical=['planned_generations.jsonl','generation_attempts.jsonl','tables/metrics.jsonl','report/status.json']
    baseline_run=baseline/'runs/v1_supplied_spellings';before={n:digest((baseline_run/n).read_bytes()) for n in historical}
    write_json(run/'checks/baseline_hashes_before.json',before)
    records=read_jsonl(run/'generations.jsonl');start=time.monotonic();phase='loading';probe_results=[]
    def status(state,**extra):
        latest={r['generation_id']:r for r in records}
        write_json(run/'logs/status.json',{'status':state,'phase':phase,'generated':sum(r['status']=='generated' for r in latest.values()),
            'attempts':len(records),'planned':80,'elapsed_s':time.monotonic()-start,**extra})
    status('running')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(min(.9,18*1024**3/torch.cuda.get_device_properties(0).total_memory))
        model=IndicSpeak(device='cuda',snapshot=snap)
        if model.decoding!=DECODING:raise ValueError('Decoding settings changed')
        if Path(sys.modules['inference'].__file__).resolve()!=(snap/'inference.py').resolve():raise ValueError('Wrong vendor module imported')
        if len(model.tts.lm.model.layers)!=28 or model.tts.lm.config.hidden_size!=3072:raise ValueError('Unexpected live model geometry')
        write_json(run/'checks/runtime.json',{'python':sys.version,'GPU':torch.cuda.get_device_name(0),'packages':{k:metadata.version(k) for k in ['torch','transformers','snac','soundfile','numpy']},'model_class':type(model.tts.lm).__name__,'layers':28,'hidden_size':3072,'checkpoint_weights_changed':False})
        def call(j,callback):
            return model(j['input_text'],j['voice'],j['seed'],j['max_new_tokens'],style=j['style'],token_callback=callback)
        def generate(j):
            if time.monotonic()-start>a.max_seconds:raise TimeoutError('Stage1 time limit reached')
            latest={r['generation_id']:r for r in records};old=latest.get(j['generation_id'])
            if old and old['status']=='generated':
                if digest((run/old['audio_path']).read_bytes())!=old['audio_SHA256']:raise ValueError('Existing WAV changed')
                return old
            attempt=1+sum(r['generation_id']==j['generation_id'] for r in records)
            token_path=run/'tokens'/f"{j['generation_id']}_attempt{attempt}.json"
            audio=Path('audio')/j['voice']/j['script']/(j['style'] or 'no_style')/(j['utterance_id']+f'_attempt{attempt}.wav')
            if token_path.exists() or (run/audio).exists():raise ValueError('Refuse unlogged overwrite')
            result={**j,'attempt':attempt,'status':'failed','audio_path':audio.as_posix(),'tokens_path':str(token_path.relative_to(run))};captured={}
            def capture(prompt,tokens):
                captured.update(prompt_token_ids=list(prompt),generated_token_ids=list(tokens),
                    prompt_text=model.tts.tok.decode(list(prompt),skip_special_tokens=False),
                    prompt_checks=prompt_checks(model.tts.tok,j['input_text'],j['voice'],j['style'],list(prompt),model._build_prompt),
                    token_structure=structure(tokens,{model.tts._eos,model.tts.tok.eos_token_id},j['max_new_tokens']))
                write_json(token_path,captured)
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');wav,info=call(j,capture)
                if wav.ndim!=1 or not len(wav) or not np.isfinite(wav).all() or not np.any(wav):raise ValueError('Invalid waveform')
                target=run/audio;target.parent.mkdir(parents=True,exist_ok=True);sf.write(target,wav,24000,subtype='PCM_16')
                decoded,sr=sf.read(target,dtype='float64');header=sf.info(target)
                valid=sr==24000 and header.channels==1 and header.subtype=='PCM_16' and len(decoded)>0 and np.isfinite(decoded).all() and np.any(decoded)
                s=captured['token_structure'];prompt_ok=all(captured['prompt_checks'].values())
                result.update(info,status='generated',audio_SHA256=digest(target.read_bytes()),float_wave_SHA256=digest(wav.tobytes()),
                    tokens_SHA256=digest(token_path.read_bytes()),duration_s=len(wav)/24000,waveform_valid=bool(valid),
                    clipped_fraction=float(np.mean(np.abs(decoded)>=32767/32768)),prompt_checks=captured['prompt_checks'],token_structure=s,
                    technical_pass=bool(valid and prompt_ok and s['normal_EOS'] and s['usable_complete_code_frames']>0 and s['invalid_frames']==0 and not s['token_cap_reached']),
                    warnings=[str(w.message) for w in caught]+(['incomplete trailing tokens discarded by baseline decoder'] if s['incomplete_trailing_tokens'] else []))
            except Exception as e:result.update(error_type=type(e).__name__,error=str(e)[:1000],technical_pass=False)
            result.update(elapsed_s=time.monotonic()-t,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved())
            with (run/'generations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
            records.append(result);status('running');print(json.dumps({k:result.get(k) for k in ['utterance_id','voice','script','style','status','technical_pass','elapsed_s']}),flush=True)
            return result
        phase='initial_eight';smoke=[j for j in jobs if j['utterance_id']==smoke_id]
        initial=[generate(j) for j in smoke];review(run,rows,jobs,records)
        if not all(r.get('technical_pass') for r in initial):raise RuntimeError('Initial eight have technical failures/warnings; full stage not launched')
        phase='alpha_zero_readiness';status('running')
        for j,reference in zip(smoke,initial):
            prior=json.loads((run/reference['tokens_path']).read_text());captured={};torch.cuda.reset_peak_memory_stats()
            with ZeroSteeringProbe(model.tts.lm,len(prior['prompt_token_ids'])) as probe:
                wav,_=call(j,lambda p,g:captured.update(prompt=list(p),tokens=list(g)))
            result=probe.result();result.update(generation_id=j['generation_id'],
                prompt_exact=captured['prompt']==prior['prompt_token_ids'],tokens_exact=captured['tokens']==prior['generated_token_ids'],
                float_wave_exact=digest(wav.tobytes())==reference['float_wave_SHA256'],peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
            probe_results.append(result);write_json(run/'checks/alpha_zero.json',probe_results)
            if not all([result['prompt_exact'],result['tokens_exact'],result['float_wave_exact'],result['fresh_cache'],result['hooks_removed'],result['all_outputs_unchanged'],all(result['finite_layers'].values()),result['forward_calls']>1,result['prefill_calls']==1,result['audio_positions_checked']>0,set(result['finite_layers'])=={'8','14','20'}]):raise RuntimeError('Alpha-zero hook equivalence failed')
            print('alpha-zero passed '+j['generation_id'],flush=True)
        # One generation after the last hook removal checks the restored path.
        captured={};wav,_=call(smoke[0],lambda p,g:captured.update(tokens=list(g)))
        ref=json.loads((run/initial[0]['tokens_path']).read_text())
        restored=captured['tokens']==ref['generated_token_ids'] and digest(wav.tobytes())==initial[0]['float_wave_SHA256']
        write_json(run/'checks/after_hook_removal.json',{'tokens_and_float_wave_exact':restored})
        if not restored:raise RuntimeError('Post-removal behavior changed')
        write_json(run/'checks/initial_gate.json',{'passed':True,'initial_recordings':8,'zero_equivalence_cases':8,'post_removal_equivalence':True})
        phase='remaining_controls';status('running')
        for j in jobs:
            generate(j)
            if len(records)>=3 and all(r['status']=='failed' for r in records[-3:]):raise RuntimeError('Three consecutive failures; stopped')
        phase='complete';review(run,rows,jobs,records)
        latest={r['generation_id']:r for r in records};csvsave(run/'generation_summary.csv',[{k:r.get(k) for k in ['generation_id','utterance_id','split','voice','script','style','status','technical_pass','audio_path','tokens_path','duration_s','hit_max','peak_GPU_allocated_bytes','error_type']} for r in latest.values()])
        status('complete',technical_pass=sum(bool(r.get('technical_pass')) for r in latest.values()),vector_fitting_run=False,nonzero_steering_run=False,human_listening_performed=False)
    except Exception as e:
        review(run,rows,jobs,records);status('failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        after={n:digest((baseline_run/n).read_bytes()) for n in historical}
        write_json(run/'checks/baseline_preserved.json',{'unchanged':before==after,'before':before,'after':after})
        if before!=after:raise RuntimeError('Historical baseline metadata changed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline-project',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--codec-snapshot',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=3600);p.add_argument('--resume',action='store_true')
    main(p.parse_args())
