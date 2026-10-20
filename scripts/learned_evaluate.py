"""Owned-output learned diagnostics on existing WAVs; downloads require approval.

No TTS generation. Preserve model revisions, waveform hashes and missing statuses.
NISQA-TTS predictions and speaker embeddings are not human naturalness judgments.
"""
from __future__ import annotations
import argparse,gc,json,os,sys,time,importlib.metadata as metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl

ASR=('openai/whisper-small','973afd24965f72e36ca33b3055d56a652f456b4d')
SPEAKER=('microsoft/wavlm-base-plus-sv','feb593a6c23c1cc3d9510425c29b0a14d2b07b1e')

def load_audio(row):
    import soundfile as sf
    import torch,torchaudio.functional as af
    path=Path(row['source_audio_path'])
    if digest(path.read_bytes())!=row['audio']['audio_sha256']:raise ValueError('Source audio changed')
    y,sr=sf.read(path,dtype='float32')
    if y.ndim!=1 or not len(y):raise ValueError('Expected nonempty mono baseline')
    return af.resample(torch.from_numpy(y),sr,16000).numpy() if sr!=16000 else y

def record(row,**fields):
    return {'generation_id':row['generation_id'],'utterance_id':row['utterance_id'],
        'configuration_id':row['configuration_id'],'audio_sha256':row['audio']['audio_sha256'],
        'in_challenge_view':row['in_challenge_view'],**fields}

def asr_speaker(rows,out,cache,include_speaker=True):
    import numpy as np,torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperProcessor,WhisperForConditionalGeneration,AutoFeatureExtractor,WavLMForXVector
    if not torch.cuda.is_available():raise RuntimeError('Approved GPU job requires an available CUDA device')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(min(.9,8*1024**3/torch.cuda.get_device_properties(0).total_memory))
    device='cuda';paths={}
    for name,revision in ([ASR,SPEAKER] if include_speaker else [ASR]):
        paths[name]=snapshot_download(name,revision=revision,cache_dir=str(cache),token=False,
            allow_patterns=['model.safetensors' if name==ASR[0] else 'pytorch_model.bin','*.json','merges.txt','vocab.json'])
    processor=WhisperProcessor.from_pretrained(paths[ASR[0]],local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(paths[ASR[0]],local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='eager').to(device).eval()
    hypotheses=[]
    for i,row in enumerate(rows):
        try:
            audio=load_audio(row)
            features=processor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True,truncation=False)
            with torch.inference_mode():
                ids=model.generate(features.input_features.to(device,torch.float16),attention_mask=features.attention_mask.to(device),
                    language='hi',task='transcribe',do_sample=False,num_beams=1,max_new_tokens=440,
                    return_timestamps=True,condition_on_prev_tokens=False)
            hypothesis=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
            hypotheses.append(record(row,status='computed_asr_proxy',hypothesis_raw=hypothesis,model=ASR[0],revision=ASR[1]))
        except Exception as e:
            hypotheses.append(record(row,status='extraction_failed',hypothesis_raw=None,error_type=type(e).__name__,error=str(e)[:400]))
            if len(hypotheses)>=3 and all(r['status']=='extraction_failed' for r in hypotheses[-3:]):
                write_jsonl(out/'second_asr.jsonl',hypotheses)
                raise RuntimeError('Three consecutive ASR extraction failures; inspect outputs before continuing') from e
        if (i+1)%25==0:
            write_jsonl(out/'second_asr.jsonl',hypotheses);print(f'Whisper-small {i+1}/{len(rows)}',flush=True)
    write_jsonl(out/'second_asr.jsonl',hypotheses)
    del model,processor;gc.collect();torch.cuda.empty_cache()
    if not include_speaker:
        write_json(out/'decoding.json',{'asr_model':ASR,'language':'hi','task':'transcribe','do_sample':False,
            'num_beams':1,'max_new_tokens':440,'return_timestamps':True,'condition_on_prev_tokens':False,
            'truncation':False,'attention_mask':True,'batch_size':1,'sample_rate':16000,'dtype':'float16','attention':'eager',
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})
        return
    extractor=AutoFeatureExtractor.from_pretrained(paths[SPEAKER[0]],local_files_only=True)
    model=WavLMForXVector.from_pretrained(paths[SPEAKER[0]],local_files_only=True).to(device).eval()
    embeddings=[]
    for i,row in enumerate(rows):
        try:
            audio=load_audio(row)
            inputs=extractor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to(device)
            with torch.inference_mode():
                vector=model(**inputs).embeddings[0]
                vector=torch.nn.functional.normalize(vector,dim=-1).cpu().numpy()
            embeddings.append(record(row,status='computed_embedding_diagnostic',embedding=vector.astype(float).tolist(),model=SPEAKER[0],revision=SPEAKER[1]))
        except Exception as e:embeddings.append(record(row,status='extraction_failed',embedding=None,error_type=type(e).__name__,error=str(e)[:400]))
        if (i+1)%25==0:write_jsonl(out/'speaker_embeddings.jsonl',embeddings);print(f'WavLM embeddings {i+1}/{len(rows)}',flush=True)
    write_jsonl(out/'speaker_embeddings.jsonl',embeddings)
    write_json(out/'decoding.json',{'asr_model':ASR,'speaker_model':SPEAKER,'ASR':{'sample_rate':16000,'language':'hi','task':'transcribe',
        'batch_size':1,'do_sample':False,'num_beams':1,'max_new_tokens':440,'return_timestamps':True,'condition_on_prev_tokens':False,
        'truncation':False,'attention_mask':True,'dtype':'float16','attention':'eager'},'speaker':{'sample_rate':16000,'batch_size':1,'dtype':'float32','attention_mask':True,'L2_normalized':True},
        'comparison_limit':'Legacy uses missing attention masks and potentially truncated long input; exact settings are not identical',
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})

def nisqa(rows,out,source):
    # Official source/checkpoint provided separately and pinned in its manifest.
    import csv
    sys.path.insert(0,str(source))
    from nisqa.NISQA_model import nisqaModel
    with (out/'inputs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['generation_id','wav']);writer.writeheader()
        for r in rows:
            if digest(Path(r['source_audio_path']).read_bytes())!=r['audio']['audio_sha256']:raise ValueError('Source changed')
            writer.writerow({'generation_id':r['generation_id'],'wav':r['source_audio_path']})
    args={'mode':'predict_csv','pretrained_model':str(source/'weights/nisqa_tts.tar'),'data_dir':'','csv_file':str(out/'inputs.csv'),
        'csv_deg':'wav','output_dir':str(out),'tr_num_workers':0,'tr_bs_val':1,'tr_device':'cpu','tr_parallel':False,'ms_channel':None}
    model=nisqaModel(args);model.predict()
    values=model.ds_val.df.to_dict('records');by_path={r['source_audio_path']:r for r in rows}
    result=[]
    for v in values:
        row=by_path[v['wav']];score=float(v['mos_pred'])
        result.append(record(row,status='exploratory_model_prediction_unvalidated_hinglish',nisqa_tts_predicted_mos=score,
            checkpoint_sha256=digest((source/'weights/nisqa_tts.tar').read_bytes())))
    write_jsonl(out/'nisqa_tts.jsonl',result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['asr-speaker','asr','nisqa']);p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--nisqa-source',type=Path)
    a=p.parse_args()
    if a.mode in ('asr','asr-speaker') and a.cache is None:p.error('--cache is required for ASR/speaker evaluation')
    if a.mode=='nisqa' and a.nisqa_source is None:p.error('--nisqa-source is required')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    out=private_directory(a.output)
    if out.exists():raise ValueError('Use a new output directory; prior results preserved')
    out.mkdir(parents=True);start=time.time();rows=read_jsonl(a.baseline_run/'tables/metrics.jsonl')
    if not rows:raise ValueError('No baseline measurements')
    if a.mode in ('asr-speaker','asr'):asr_speaker(rows,out,a.cache,include_speaker=a.mode=='asr-speaker')
    else:nisqa(rows,out,a.nisqa_source)
    versions={}
    for package in ['torch','torchaudio','transformers','numpy','soundfile','librosa','huggingface-hub']:
        try:versions[package]=metadata.version(package)
        except Exception:versions[package]=None
    files={'asr':['second_asr.jsonl'],'asr-speaker':['second_asr.jsonl','speaker_embeddings.jsonl'],'nisqa':['nisqa_tts.jsonl']}[a.mode]
    coverage={name:{'records':len(rs:=read_jsonl(out/name)),'failed':sum(r.get('status')=='extraction_failed' for r in rs)} for name in files}
    complete=all(c['records']==len(rows) and c['failed']==0 for c in coverage.values())
    write_json(out/'completion.json',{'status':'complete' if complete else 'complete_with_failures','coverage':coverage,'mode':a.mode,'recordings':len(rows),'runtime_s':time.time()-start,
        'baseline_metrics_sha256':digest((a.baseline_run/'tables/metrics.jsonl').read_bytes()),'script_sha256':digest(Path(__file__).read_bytes()),'packages':versions,'tts_runs':0})
