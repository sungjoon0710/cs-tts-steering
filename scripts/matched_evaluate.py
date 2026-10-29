"""Reuse existing evaluators for the independent natural-audio/script experiment."""
import argparse
import csv
import html
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest, read_jsonl, stable_id, write_json, write_jsonl
from src.evaluation.expanded import content_proxies, expanded_audio
from src.evaluation.language import language_errors
from src.evaluation.learned import speaker_consistency
from src.evaluation.provenance import paired_cluster_intervals
from src.evaluation.text import edit_score, NORMALIZATION_VERSION, PIER_SOURCE
from lab_rate_evaluate import detector_body, measure


def save_csv(path, rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def natural_input(row, dataset, run, channel='mean'):
    source=dataset/row['natural_audio_relative_path']
    if digest(source.read_bytes())!=row['natural_sha256']:raise ValueError('Natural source changed')
    target=run/'analysis_audio'/channel/(row['utterance_id']+'.wav')
    receipt=target.with_suffix('.json')
    if target.exists():
        old=json.loads(receipt.read_text())
        if old['source_sha256']!=row['natural_sha256'] or old['sha256']!=digest(target.read_bytes()):
            raise ValueError('Derived audio changed')
        return target
    y,sr=sf.read(source,dtype='float64',always_2d=True)
    mono=y.mean(axis=1) if channel=='mean' else y[:,0]
    divisor=math.gcd(sr,24000)
    mono=resample_poly(mono,24000//divisor,sr//divisor,window=('kaiser',5.0)) if sr!=24000 else mono
    target.parent.mkdir(parents=True,exist_ok=True)
    sf.write(target,mono,24000,subtype='PCM_16')
    write_json(receipt,{'source_sha256':row['natural_sha256'],'sha256':digest(target.read_bytes()),
        'source_rate':sr,'source_channels':y.shape[1],'channel_policy':channel,
        'output_rate':24000,'subtype':'PCM_16','resampling':'polyphase Kaiser beta5; no peak normalization'})
    return target


def catalog(dataset, run):
    cohort=read_jsonl(dataset/'cohort.jsonl'); utterances={r['utterance_id']:r for r in cohort}
    attempts=read_jsonl(run/'generation_attempts.jsonl'); last={r['generation_id']:r for r in attempts}
    records=[]
    for row in cohort:
        path=natural_input(row,dataset,run)
        records.append({'generation_id':stable_id('natural-v1',row['utterance_id'],row['natural_sha256']),
            'utterance_id':row['utterance_id'],'configuration_id':'natural/'+row['source_speaker'],
            'kind':'natural','script_condition':'natural_spoken','pilot':row['pilot'],
            'source_audio_path':str(path),'audio_sha256':digest(path.read_bytes()),
            'original_audio_sha256':row['natural_sha256'],'original_audio_path':str(dataset/row['natural_audio_relative_path']),
            'generation_status':'existing_natural','canonical_text':row['canonical_text'],'in_challenge_view':False})
    for row in read_jsonl(run/'planned_generations.jsonl'):
        prior=last.get(row['generation_id']); path=run/row['relative_audio_path']
        status='not_attempted' if prior is None else prior['status']
        if status=='generated':
            if not path.is_file() or digest(path.read_bytes())!=prior['audio_sha256']:
                raise ValueError('Generated source changed or missing')
        records.append({**row,'kind':'synthetic','source_audio_path':str(path),
            'audio_sha256':prior.get('audio_sha256') if prior else None,
            'generation_status':status,'in_challenge_view':False,
            'attempt_count':sum(r['generation_id']==row['generation_id'] for r in attempts),
            'prior_failed_attempts':sum(r['generation_id']==row['generation_id'] and r['status']=='failed' for r in attempts),
            'token_limit_warning':prior.get('hit_max') if prior else None,
            'generation_error_type':prior.get('error_type') if prior else None})
    if len({r['generation_id'] for r in records})!=len(records):raise ValueError('Duplicate recording IDs')
    return records,utterances


def acoustics(a):
    records,utterances=catalog(a.dataset,a.run)
    path=a.run/'tables/metrics.jsonl'; previous={r['generation_id']:r for r in read_jsonl(path)}
    body=detector_body(a.lab_script.read_bytes(),'terminal_v2'); results=[]; coverage=[]
    started=time.time()
    for row in records:
        available=row['generation_status'] in ('existing_natural','generated')
        coverage.append({k:row[k] for k in ['generation_id','utterance_id','configuration_id','kind','pilot','generation_status']})
        if not available:continue
        gid=row['generation_id'];old=previous.get(gid)
        if old and old['audio']['audio_sha256']==row['audio_sha256']:
            results.append(old);continue
        audio=expanded_audio(row['source_audio_path'],egemaps=True)
        if audio.get('audio_sha256')!=row['audio_sha256']:raise ValueError('Audio changed during feature extraction')
        rate={'status':'not_extractable','nuclei_per_s':None}
        if audio['status']=='computed_unvalidated':
            wave,sr=sf.read(row['source_audio_path'],dtype='float64')
            try:rate=measure(wave,sr,body,'terminal_v2')
            except Exception as exc:rate={'status':'extraction_failed','error_type':type(exc).__name__,'nuclei_per_s':None}
        results.append({**row,'audio':audio,'nuclei_rate':rate,'text_raw':row['canonical_text'],
            'reference_status':'metadata_linked; no human transcript verification'})
        if len(results)%10==0:
            write_jsonl(path,results);print(f'Acoustics {len(results)} available recordings',flush=True)
    write_jsonl(path,results);write_jsonl(a.run/'tables/coverage.jsonl',coverage)
    sensitivity=[]
    for row in utterances.values():
        if not row['pilot'] or row['source_channels']==1:continue
        first=natural_input(row,a.dataset,a.run,'first');features=expanded_audio(first,egemaps=False)
        sensitivity.append({'utterance_id':row['utterance_id'],'channel_policy':'first',
            'audio_sha256':digest(first.read_bytes()),'audio':features})
    write_jsonl(a.run/'metrics/channel_sensitivity.jsonl',sensitivity)
    channel_checks=[]
    for row in utterances.values():
        y,sr=sf.read(a.dataset/row['natural_audio_relative_path'],dtype='float64',always_2d=True)
        rms=np.sqrt(np.mean(y*y,axis=0));mean_rms=float(np.sqrt(np.mean(y.mean(axis=1)**2)))
        nonzero=len(rms)>1 and rms[0]>0 and rms[1]>0
        channel_checks.append({'utterance_id':row['utterance_id'],'source_speaker':row['source_speaker'],
            'source_rate':sr,'source_channels':y.shape[1],'duration_s':len(y)/sr,
            'stereo_channels_identical':bool(np.array_equal(y[:,0],y[:,1])) if y.shape[1]==2 else None,
            'channel_correlation':float(np.corrcoef(y[:,0],y[:,1])[0,1]) if nonzero else None,
            'mean_downmix_rms_ratio':mean_rms/float(rms.mean()) if rms.mean()>0 else None,
            'background_audio_review':'requires listening','speaker_overlap_review':'requires listening',
            'transcript_content_review':'not human verified'})
    save_csv(a.run/'report/natural_recording_checks.csv',channel_checks)
    write_json(a.run/'metrics/acoustic_completion.json',{'available_records':len(results),'expected_records':len(records),
        'status_counts':dict(Counter(r['audio']['status'] for r in results)),
        'finite_88_eGeMAPS':sum(len(r['audio'].get('egemaps',{}).get('features',{}))==88 and
            all(v is not None and math.isfinite(v) for v in r['audio']['egemaps']['features'].values()) for r in results),
        'runtime_s':time.time()-started,'source_sha256':digest(Path(__file__).read_bytes()),
        'extractor_sha256':{name:digest((Path(__file__).resolve().parents[1]/'src/evaluation'/name).read_bytes())
            for name in ['audio.py','expanded.py','text.py','language.py']},
        'packages':{p:metadata.version(p) for p in ['numpy','scipy','soundfile','praat-parselmouth','opensmile']},
        'natural_channel_policy':'mean; resample to 24 kHz PCM16; pilot first-channel sensitivity at same 24 kHz',
        'manual_validation_complete':False})


def learned(a):
    import learned_evaluate
    rows=[r for r in read_jsonl(a.run/'tables/metrics.jsonl')
        if r['audio']['status']=='computed_unvalidated' and (a.scope=='full' or r['pilot'])]
    backend=a.action
    filename='nisqa_tts.jsonl' if backend=='nisqa' else 'second_asr.jsonl'
    existing=list((a.run/'metrics').glob(backend+'_*'))
    cached={};speaker_cached={}
    for folder in sorted(existing):
        for row in read_jsonl(folder/filename):
            if row.get('status')!='extraction_failed':cached[row['generation_id']]=row
        if backend=='asr-small':
            for row in read_jsonl(folder/'speaker_embeddings.jsonl'):
                if row.get('embedding') is not None:speaker_cached[row['generation_id']]=row
    for row in rows:
        old=cached.get(row['generation_id'])
        if old and old['audio_sha256']!=row['audio']['audio_sha256']:
            raise ValueError('Cached learned result belongs to changed audio')
        vector=speaker_cached.get(row['generation_id'])
        if vector and vector['audio_sha256']!=row['audio']['audio_sha256']:
            raise ValueError('Cached speaker result belongs to changed audio')
    rows=[r for r in rows if r['generation_id'] not in cached or backend=='asr-small' and r['generation_id'] not in speaker_cached]
    if not rows:print(f'{backend}: all requested waveforms already evaluated');return
    output=a.run/'metrics'/f'{backend}_{a.scope}_batch{len(existing)+1:04}'
    output.mkdir(parents=True)
    start=time.time()
    if backend=='nisqa':
        if a.nisqa_source is None:raise ValueError('--nisqa-source required')
        learned_evaluate.nisqa(rows,output,a.nisqa_source)
    else:
        model=learned_evaluate.ASR
        if backend=='asr-large':
            if a.large_revision is None:raise ValueError('Pinned --large-revision required')
            model=('openai/whisper-large-v3',a.large_revision)
        learned_evaluate.asr_speaker(rows,output,a.cache,include_speaker=backend=='asr-small',
            asr_model=model,local_files_only=a.offline)
    write_json(output/'completion.json',{'recordings':len(rows),'runtime_s':time.time()-start,
        'model':model if backend!='nisqa' else 'NISQA-TTS; checksum in predictions',
        'scope':a.scope,'source_sha256':digest(Path(__file__).read_bytes()),'manual_validation_complete':False})


def report(a):
    cohort=read_jsonl(a.dataset/'cohort.jsonl'); by_id={r['utterance_id']:r for r in cohort}
    records,utterances=catalog(a.dataset,a.run)
    audio={r['generation_id']:r for r in read_jsonl(a.run/'tables/metrics.jsonl')}
    hypotheses={};quality={};vectors={}
    for backend in ['asr-small','asr-large']:
        values={}
        for folder in sorted((a.run/'metrics').glob(backend+'_*')):
            for r in read_jsonl(folder/'second_asr.jsonl'):
                if r.get('hypothesis_raw') is not None:values[r['generation_id']]=r
        hypotheses[backend]=values
    for folder in sorted((a.run/'metrics').glob('nisqa_*')):
        for r in read_jsonl(folder/'nisqa_tts.jsonl'):quality[r['generation_id']]=r
    for folder in sorted((a.run/'metrics').glob('asr-small_*')):
        for r in read_jsonl(folder/'speaker_embeddings.jsonl'):
            if r.get('embedding') is not None:vectors[r['generation_id']]=r
    flat=[];detailed=[]
    for row in records:
        gid=row['generation_id']; feature=audio.get(gid,{}); u=by_id[row['utterance_id']]
        values={'generation_id':gid,'utterance_id':row['utterance_id'],'configuration_id':row['configuration_id'],
            'kind':row['kind'],'script_condition':row['script_condition'],'pilot':row['pilot'],
            'language_condition':u['condition'],'source_speaker':u['source_speaker'],
            'input_word_count':u['token_count'],'english_proportion_provisional':u['english_tokens']/u['token_count'],
            'switch_count_provisional':sum(x!=y for x,y in zip(u['language_labels'],u['language_labels'][1:])),
            'generation_status':row['generation_status'],'audio_status':feature.get('audio',{}).get('status','not_extracted'),
            'duration_s':feature.get('audio',{}).get('duration_s'),'f0_median_hz':feature.get('audio',{}).get('f0_median_hz'),
            'f0_range_st':feature.get('audio',{}).get('f0_range_st'),'pause_proportion':feature.get('audio',{}).get('internal_pause_proportion'),
            'nuclei_per_s':feature.get('nuclei_rate',{}).get('nuclei_per_s'),
            'nisqa_tts_prediction':quality.get(gid,{}).get('nisqa_tts_predicted_mos'),
            'attempt_count':row.get('attempt_count',0),'failed_attempt_count':row.get('prior_failed_attempts',0),
            'token_limit_warning':row.get('token_limit_warning'),
            'generation_error_type':row.get('generation_error_type'),
            'clipped_sample_fraction':feature.get('audio',{}).get('clipped_sample_fraction'),
            'duration_vs_natural_ratio':feature.get('audio',{}).get('duration_s',0)/u['source_duration_s'] if feature else None}
        for name in ['f0_std_st','voiced_frame_fraction','rms_dbfs','intensity_praat_mean_db',
                'intensity_praat_iqr_db','silence_proportion','internal_pause_count','internal_pause_total_s',
                'articulation_time_estimate_s']:
            values[name]=feature.get('audio',{}).get(name)
        values['nuclei_per_detected_sounding_s']=feature.get('nuclei_rate',{}).get('nuclei_per_detected_sounding_s')
        detail={'generation_id':gid,'canonical_text':u['canonical_text'],'reference_status':'provisional; supplied spellings retained'}
        for backend in ['asr-small','asr-large']:
            hypothesis=hypotheses[backend].get(gid,{}).get('hypothesis_raw')
            if hypothesis is None:
                strict=raw=roman={'status':'missing_asr','wer':None,'cer':None};proxies={}
            else:
                strict,raw,roman,proxies=content_proxies(u['canonical_text'],hypothesis,feature.get('audio',{}))
            detail[backend]={'hypothesis_raw':hypothesis,'strict':strict,'raw_script':raw,'uroman':roman,'asr_rates':proxies}
            values[backend+'_wer']=strict.get('wer');values[backend+'_cer']=strict.get('cer')
            values[backend+'_status']=strict['status']
            values[backend+'_raw_wer']=raw.get('wer');values[backend+'_uroman_wer']=roman.get('wer')
            for name in ['asr_speaking_rate_wps','asr_articulation_rate_proxy_wps','asr_deletion_fraction','asr_insertion_fraction','asr_repeated_trigram_fraction']:
                values[backend+'_'+name]=proxies.get(name)
        sh=hypotheses['asr-small'].get(gid,{}).get('hypothesis_raw');lh=hypotheses['asr-large'].get(gid,{}).get('hypothesis_raw')
        values['recognizer_disagreement_wer']=edit_score(lh,sh).get('wer')
        tagged=[{'token':t,'label':l} for t,l in zip(u['canonical_tokens'],u['language_labels'])]
        detail['provisional_POI']=language_errors(u['canonical_text'],lh,tagged)
        detail['provisional_POI_small']=language_errors(u['canonical_text'],sh,tagged)
        for backend,key in [('asr-large','provisional_POI'),('asr-small','provisional_POI_small')]:
            for subset in ['english_poi','hindi_poi','switch_window_poi']:
                values[backend+'_'+subset]=(detail[key].get(subset) or {}).get('pier')
                values[backend+'_'+subset+'_status']=(detail[key].get(subset) or {}).get('status',detail[key]['status'])
        flat.append(values);detailed.append(detail)
    out=a.run/'report';out.mkdir(exist_ok=True)
    save_csv(out/'per_recording.csv',flat);write_jsonl(out/'content_details.jsonl',detailed)
    write_json(out/'metric_definitions.json',{
        'normalization':{'version':NORMALIZATION_VERSION,'method':'NFC, Latin case folding, documented punctuation handling; preserve Devanagari combining marks; no spelling correction'},
        'WER':'(substitutions + deletions + insertions) / canonical reference word count; missing ASR is null, not zero',
        'CER':'Levenshtein distance / canonical non-space Unicode codepoint count; not a phoneme or grapheme error rate',
        'normalization_sensitivity':'raw NFC/script-sensitive; hi-en-nfc-v1; exploratory uroman harmonization, not validated equivalence',
        'recognizer_disagreement_WER':'Whisper-large hypothesis is reference; Whisper-small hypothesis is candidate; neither is ground truth',
        'selected_word_errors':{'alignment':PIER_SOURCE,'labels':'provided, unverified corpus HI/EN positions; not verified embedded PIER','no_selected_words':'not applicable, null'},
        'Praat':'60-600 Hz autocorrelation;10ms frames; voiced F0 median/std/range; intensity-based silent intervals are distinct from unvoiced speech; detailed settings in tables/metrics.jsonl',
        'rate_proxies':'ASR token counts / duration or intensity-derived articulation time; lab terminal_v2 nuclei / total or detected-sounding time; not validated words/syllables or articulation boundaries',
        'eGeMAPS':'openSMILE eGeMAPSv02 Functionals,88 features,16kHz polyphase input; correlated descriptors, not88 independent quality scores',
        'NISQA_TTS':'official nisqa_tts.tar model prediction; not human MOS or calibrated Hindi-English naturalness',
        'WavLM':'within-configuration mean pair cosine and matched same-voice cross-script cosine; content, language and duration can affect embeddings',
        'paired_intervals':'mean B minus A;2000 source-episode cluster bootstrap draws,seed 1234,percentile95% interval; exploratory, no multiplicity adjustment, no decoding-seed variance',
        'natural_reference':'same selected input IDs, different actors/recording conditions; stereo mean downmix and24kHz PCM16 resampling for metrics; source WAVs unchanged',
        'human_validation_complete':False})
    save_csv(out/'transcript_pairs.csv',[{k:u[k] for k in ['utterance_id','source_speaker',
        'romanized_text_original','devanagari_text_original','input_romanized','input_devanagari',
        'canonical_text','condition','pilot','natural_audio_relative_path','natural_sha256',
        'transcript_verification','script_equivalence','text_policy']} for u in cohort])
    summaries=[]
    for config in sorted({r['configuration_id'] for r in flat})+['natural_reference_pooled']:
        rows=[r for r in flat if r['kind']=='natural'] if config=='natural_reference_pooled' else [r for r in flat if r['configuration_id']==config]
        summary={'configuration_id':config,'expected':len(rows),'generated_or_natural':sum(r['generation_status'] in ('generated','existing_natural') for r in rows)}
        for metric in ['duration_s','f0_range_st','nuclei_per_s','nisqa_tts_prediction','asr-small_wer','asr-large_wer',
                'asr-small_uroman_wer','asr-large_uroman_wer',
                'asr-large_english_poi','asr-large_hindi_poi','asr-large_switch_window_poi',
                'asr-small_english_poi','asr-small_hindi_poi','asr-small_switch_window_poi']:
            x=[r[metric] for r in rows if r[metric] is not None and math.isfinite(r[metric])]
            summary[metric+'_n']=len(x);summary[metric+'_median']=float(np.median(x)) if x else None
        summaries.append(summary)
    save_csv(out/'configuration_summary.csv',summaries)
    reliability=[]
    for config in sorted({r['configuration_id'] for r in flat if r['kind']=='synthetic'}):
        rows=[r for r in flat if r['configuration_id']==config]
        reliability.append({'configuration_id':config,'expected':len(rows),
            'attempted_inputs':sum(r['attempt_count']>0 for r in rows),
            'generated':sum(r['generation_status']=='generated' for r in rows),
            'failed_final_status':sum(r['generation_status']=='failed' for r in rows),
            'not_attempted':sum(r['generation_status']=='not_attempted' for r in rows),
            'failed_attempts':sum(r['failed_attempt_count'] for r in rows),
            'token_limit_warnings':sum(bool(r['token_limit_warning']) for r in rows),
            'silent_or_empty_audio':sum(r['audio_status'] in ('fully_silent','empty_audio') for r in rows),
            'manual_truncation_repetition_artifact_review':'not performed'})
    save_csv(out/'generation_reliability.csv',reliability)
    speaker=[]
    for config in sorted({r['configuration_id'] for r in records}):
        rs=[r for r in records if r['configuration_id']==config and vectors.get(r['generation_id'],{}).get('embedding') is not None]
        if len(rs)<3:
            speaker.append({'configuration_id':config,'recordings':len(rs),'mean_pair_cosine':None,'status':'insufficient_embeddings'});continue
        result=speaker_consistency(np.array([vectors[r['generation_id']]['embedding'] for r in rs]),repeats=200)
        speaker.append({'configuration_id':config,'recordings':len(rs),'mean_pair_cosine':result['pairwise_mean_cosine'],
            'status':'within-configuration diagnostic; source characters not verified acoustic identity'})
    save_csv(out/'speaker_consistency.csv',speaker)
    script_speaker=[]
    generation_index={(r['utterance_id'],r['configuration_id']):r for r in records}
    for voice in ['Amit','Kavya']:
        for uid in by_id:
            dev=generation_index[(uid,f'indic-speak/{voice}/devanagari')]
            roman=generation_index[(uid,f'indic-speak/{voice}/romanized')]
            x=vectors.get(dev['generation_id'],{}).get('embedding');y=vectors.get(roman['generation_id'],{}).get('embedding')
            norm=float(np.linalg.norm(x)*np.linalg.norm(y)) if x is not None and y is not None else 0
            script_speaker.append({'utterance_id':uid,'voice':voice,'devanagari_generation_id':dev['generation_id'],
                'romanized_generation_id':roman['generation_id'],
                'cosine':float(np.dot(x,y)/norm) if norm>0 else None,
                'status':'paired same-voice embedding diagnostic; not naturalness or verified identity' if norm>0 else 'missing_or_zero_embeddings'})
    save_csv(out/'script_speaker_pairs.csv',script_speaker)
    sensitivity=[]
    natural_features={r['utterance_id']:r['audio'] for r in audio.values() if r['kind']=='natural'}
    for row in read_jsonl(a.run/'metrics/channel_sensitivity.jsonl'):
        mean=natural_features[row['utterance_id']];first=row['audio'];mh=mean.get('f0_median_hz');fh=first.get('f0_median_hz')
        sensitivity.append({'utterance_id':row['utterance_id'],'first_minus_mean_F0_st':12*math.log2(fh/mh) if mh and fh else None,
            'mean_pause_proportion':mean.get('internal_pause_proportion'),'first_pause_proportion':first.get('internal_pause_proportion'),
            'interpretation':'same 24 kHz processing; channel sensitivity, not a quality difference'})
    save_csv(out/'channel_sensitivity.csv',sensitivity)
    episode_map={r['utterance_id']:r['source_episode_name'].split('-')[0]
        for r in json.loads((a.dataset/'source_json_crosscheck.json').read_text())} if (a.dataset/'source_json_crosscheck.json').exists() else {}
    groups=defaultdict(dict)
    for row in flat:
        config='natural_reference' if row['kind']=='natural' else row['configuration_id']
        groups[config][row['utterance_id']]=row
    configs=['indic-speak/Amit/devanagari','indic-speak/Kavya/devanagari','orpheus-hi/ऋतिका/devanagari']
    contrasts=[(configs[0],configs[1]),(configs[0],configs[2]),(configs[1],configs[2]),
        ('indic-speak/Amit/devanagari','indic-speak/Amit/romanized'),
        ('indic-speak/Kavya/devanagari','indic-speak/Kavya/romanized')]
    contrasts += [('natural_reference',config) for config in groups if config!='natural_reference']
    paired=[]
    for left,right in contrasts:
        for metric in ['asr-large_wer','asr-large_cer','asr-small_wer','asr-small_cer','nisqa_tts_prediction','duration_s','f0_range_st','nuclei_per_s',
                'asr-large_uroman_wer','asr-small_uroman_wer',
                'asr-large_english_poi','asr-large_hindi_poi','asr-large_switch_window_poi',
                'asr-small_english_poi','asr-small_hindi_poi','asr-small_switch_window_poi']:
            ids=[uid for uid in groups[left].keys() & groups[right].keys()
                if groups[left][uid][metric] is not None and groups[right][uid][metric] is not None]
            ids.sort(); differences=[groups[right][uid][metric]-groups[left][uid][metric] for uid in ids]
            clusters=[episode_map.get(uid,uid) for uid in ids]
            result=paired_cluster_intervals(differences,clusters,repeats=2000)[0] if len(ids)>=3 and len(set(clusters))>=3 else None
            paired.append({'configuration_A':left,'configuration_B':right,'metric':metric,'matched_inputs':len(ids),
                'mean_B_minus_A':float(np.mean(differences)) if differences else None,
                'median_B_minus_A':float(np.median(differences)) if differences else None,
                'B_higher':sum(x>0 for x in differences),'B_lower':sum(x<0 for x in differences),'ties':sum(x==0 for x in differences),
                'ci_low':result['bootstrap_95ci'][0] if result else None,'ci_high':result['bootstrap_95ci'][1] if result else None,
                'source_clusters':len(set(clusters)),'cluster_level':'source_JSON_episode' if episode_map else 'utterance',
                'status':'exploratory; supplied spellings and language tags unverified; not a quality ranking' if result else 'insufficient paired coverage'})
    save_csv(out/'paired_comparisons.csv',paired)
    asr_checks=[]
    for config,values in sorted(groups.items()):
        for backend in ['asr-small','asr-large']:
            rs=[r for r in values.values() if r[backend+'_wer'] is not None]
            asr_checks.append({'configuration_id':config,'backend':backend,'recordings':len(rs),
                'WER_over_100_percent':sum(r[backend+'_wer']>1 for r in rs),
                'maximum_WER':max((r[backend+'_wer'] for r in rs),default=None),
                'maximum_repeated_trigram_fraction':max((r[backend+'_asr_repeated_trigram_fraction'] or 0 for r in rs),default=None),
                'interpretation':'Insertion-heavy/repetitive ASR hypotheses require listening; not confirmed audio repetition.'})
    save_csv(out/'asr_diagnostics.csv',asr_checks)
    egemaps=[]
    for gid,r in audio.items():
        features=r['audio'].get('egemaps',{}).get('features')
        if features:egemaps.append({'generation_id':gid,'utterance_id':r['utterance_id'],'configuration_id':r['configuration_id'],**features})
    save_csv(out/'egemaps_features.csv',egemaps)
    plot_available(flat,out)
    cards=[]
    for u in cohort:
        cells=[]
        natural=a.dataset/u['natural_audio_relative_path']
        cells.append('<audio controls preload="none" src="'+quote(os.path.relpath(natural,a.run).replace(os.sep,'/'))+'"></audio>')
        for model,voice,condition in [('indic-speak','Amit','devanagari'),('indic-speak','Amit','romanized'),
                ('indic-speak','Kavya','devanagari'),('indic-speak','Kavya','romanized'),('orpheus-hi','ऋतिका','devanagari')]:
            row=next(r for r in records if r['utterance_id']==u['utterance_id'] and r.get('model')==model and r.get('voice')==voice and r['script_condition']==condition)
            cell='<audio controls preload="none" src="'+quote(row['relative_audio_path'])+'"></audio>' if row['generation_status']=='generated' else html.escape(row['generation_status'])
            if row.get('token_limit_warning'):cell+='<br><b>Token cap reached; needs review</b>'
            cells.append(cell)
        warning=any(r.get('token_limit_warning') or r['generation_status']=='failed' for r in records if r['utterance_id']==u['utterance_id'])
        cards.append('<section data-warning="'+str(warning).lower()+'" data-pilot="'+str(u['pilot']).lower()+'"><h2>'+html.escape(u['utterance_id'])+
            (' — pilot' if u['pilot'] else '')+'</h2><p>Natural source label: '+html.escape(u['source_speaker'])+' (metadata; listening verification incomplete)</p><p>'+html.escape(u['input_romanized'])+'</p><p>'+html.escape(u['input_devanagari'])+
            '</p><table><tr>'+''.join('<th>'+t+'</th>' for t in ['Natural','Amit Deva','Amit Roman','Kavya Deva','Kavya Roman','Ritika Deva'])+
            '</tr><tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr></table></section>')
    page='''<!doctype html><meta charset="utf-8"><title>New matched-input experiment</title>
<style>body{font:16px/1.5 system-ui;margin:25px}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ccc;padding:8px}audio{width:230px}section{padding:12px 0;border-bottom:1px solid #ccc}</style>
<h1>Natural audio and paired TTS inputs</h1><p>Unblinded researcher comparison, not a listening-test instrument. Provisional transcripts and spellings; no human quality judgments collected. Natural and generated voices differ. Natural players use unchanged original stereo audio; metric inputs use documented mono conversion.</p>
<button onclick="document.querySelectorAll('section').forEach(s=>s.hidden=s.dataset.pilot!=='true')">Pilot only</button>
<button onclick="document.querySelectorAll('section').forEach(s=>s.hidden=s.dataset.warning!=='true')">Token-cap / failed-generation warnings</button>
<button onclick="document.querySelectorAll('section').forEach(s=>s.hidden=false)">All 100 inputs</button>'''+''.join(cards)
    (a.run/'index.html').write_text(page,encoding='utf-8')
    write_json(out/'status.json',{'expected_natural':100,'expected_synthetic':500,
        'actual_generated':sum(r['generation_status']=='generated' for r in records),'acoustic_records':len(audio),
        'generation_token_limit_warnings':sum(bool(r['token_limit_warning']) for r in flat if r['kind']=='synthetic'),
        'failed_generation_attempts':sum(r['failed_attempt_count'] for r in flat if r['kind']=='synthetic'),
        'final_failed_generations':sum(r['generation_status']=='failed' for r in flat if r['kind']=='synthetic'),
        'ASR_records':{k:len(v) for k,v in hypotheses.items()},'NISQA_records':len(quality),'speaker_vectors':len(vectors),
        'manual_validation_complete':False,'old_experiment_results_reused':False,'natural_reference_rankings_reused':False})
    status=json.loads((out/'status.json').read_text())
    table='\n'.join('| '+str(r['configuration_id'])+' | '+str(r['generated_or_natural'])+'/'+str(r['expected'])+' | '+
        ('—' if r['asr-large_wer_median'] is None else f"{100*r['asr-large_wer_median']:.1f}% (n={r['asr-large_wer_n']})")+' | '+
        ('—' if r['asr-small_wer_median'] is None else f"{100*r['asr-small_wer_median']:.1f}% (n={r['asr-small_wer_n']})")+' | '+
        ('—' if r['nisqa_tts_prediction_median'] is None else f"{r['nisqa_tts_prediction_median']:.3f}")+' |' for r in summaries if not r['configuration_id'].startswith('natural/'))
    bins=dict(Counter(r['length_bin'] for r in cohort));pilot_bins=dict(Counter(r['length_bin'] for r in cohort if r['pilot']))
    script_observations=[]
    for voice in ['Amit','Kavya']:
        contrasts=[r for r in paired if r['configuration_A']==f'indic-speak/{voice}/devanagari' and r['configuration_B']==f'indic-speak/{voice}/romanized']
        for metric in ['asr-large_wer','asr-large_uroman_wer','asr-small_wer']:
            r=next(x for x in contrasts if x['metric']==metric)
            if r['ci_low'] is not None:
                script_observations.append(f"{voice}, {metric}: mean Romanized minus Devanagari = {100*r['mean_B_minus_A']:+.1f} percentage points (exploratory 95% interval {100*r['ci_low']:+.1f} to {100*r['ci_high']:+.1f}; paired n={r['matched_inputs']}); paired median difference {100*r['median_B_minus_A']:+.1f} points; Romanized higher/lower/tied on {r['B_higher']}/{r['B_lower']}/{r['ties']} inputs.")
    script_observations='\n\n'.join(script_observations) or 'Insufficient paired ASR coverage.'
    summary=f'''# Independent local-audio/script experiment — executed status

Completed synthetic files: **{status['actual_generated']}/500**. Natural recordings: 100 existing source copies with SHA-256 verification. Acoustic records: {status['acoustic_records']}; NISQA records: {status['NISQA_records']}; ASR coverage: {status['ASR_records']}; speaker vectors: {status['speaker_vectors']}.

Failed generation attempts: {status['failed_generation_attempts']}; final failed generations: {status['final_failed_generations']}; token-limit warnings: {status['generation_token_limit_warnings']}. A token-cap hit is an automatic warning, not confirmed audible truncation or repetition. Flagged recordings remain in the denominator and retain their original outputs.

| Configuration | Available / expected | Large-ASR median WER | Small-ASR median WER | NISQA-TTS median prediction |
|---|---:|---:|---:|---:|
{table}

**Observed input-script contrasts.** Both versions use the same canonical reference; these differences describe automatic error proxies, not verified pronunciation errors. Uroman is a sensitivity analysis, not phonetic ground truth.

{script_observations}

**Recognizer outliers.** Whisper-small produces some highly repetitive, insertion-heavy hypotheses. WER can legitimately exceed 100%; these values are preserved, not clipped or removed. The small-ASR mean script contrast can be dominated or reversed by these outliers, so paired medians and higher/lower/tied counts accompany the means. asr_diagnostics.csv documents coverage and values above 100%. The full-range scatter is preserved; script_content_focus.png shows a separately labelled 0–150% view with outside-view counts. Natural-recording ASR errors are also high and need transcript/listening calibration; they do not establish poor naturalness or justify subtracting natural error from synthetic error.

These are automatic proxies and predictions, not human naturalness or verified synthesis errors. The natural pooled row combines six source-character labels; the CSV retains individual-character summaries as overlapping views, not additional recordings. Missing measurements remain missing. Failure and not-attempted statuses remain in per_recording.csv, with attempts/token-limit warnings and the original audio hashes in manifests/logs. Paired comparisons use the same content IDs; source-episode cluster intervals are exploratory, with utterance fallback only if source groups are unavailable. No architecture-only winner is established.

**Dataset and scope.** This is a new 100-input cohort using locally downloaded natural WAVs and their prepared transcript IDs, with 90 provisional code-switched items and 10 provisional Hindi-only controls. These are the source recordings for the new cohort; they do not recover the still-unavailable original recordings for the previous 100 inputs. The previous 300-recording baseline remains separate and unmodified. Full length bins: {bins}; pilot length bins: {pilot_bins}. The pilot is weighted toward longer inputs and supports functional checks, not representative quality conclusions. Source CSV names/IDs and all speaker labels agree; the nine known local/Kedi transcript mismatches were excluded. A source-JSON crosscheck records 93 exact normalized transcript matches and 7 spelling variants in the selected cohort; it is not waveform-content validation.

**Input policy.** The user chose supplied spellings rather than correction. Original Romanized/Devanagari records are retained. The Devanagari condition projects the supplied Hindi words into the Romanized sentence's punctuation/case layout while preserving English words. Both conditions use the same canonical Devanagari content reference and language-tag positions. Source conversion/spelling errors and unverified tags can affect the apparent script difference; this is a provisional input-representation study, not an isolated proof that either script is intrinsically superior. Some ambiguous language labels were conservatively excluded; remaining labels are not human verified.

**Reproduction controls.** Model checkpoints are pinned in experiment.json/model_access.json. Each script pair shares voice, checkpoint, seed 1234, decoding parameters and the SAME per-utterance max_new_tokens; that limit is the maximum of the prior character-based budgets for both inputs. Generation IDs include inputs/settings/pins and successful WAVs are hash-checked before reuse. Independent source releases and package locks record generation/evaluation software. Failed attempts are retained rather than silently regenerated/discarded.

**Natural processing.** Unchanged original recordings remain available in the listening page. Metric inputs use explicit stereo mean downmix and polyphase resampling to 24 kHz PCM16, without peak normalization; ASR/speaker models resample to 16 kHz as documented. Pilot first-channel sensitivity uses the same 24 kHz settings. Text matching does not match actor identity, performance, music, channels or recording conditions. Tell the supervisor about channel handling when interpreting natural-versus-synthetic differences.

**What is still unverified.** Human transcript/pronunciation accuracy, speaker overlap/background contamination, audible segment boundaries, conversational naturalness and switch smoothness have not been validated. ASR agreement does not establish ground truth; natural-ASR error cannot be subtracted from synthetic-ASR error. NISQA is an uncalibrated Hinglish model prediction, and speaker cosine is a within-configuration diagnostic with duration/content effects. Corpus tags support provisional selected-word errors, not human-verified embedded PIER.

**Exclusions.** No prior unmatched-reference W1/TTSDS2 score is imported. Exact text correspondence would not resolve all reference-population or the documented TTSDS2 implementation concerns. Direct waveform/actor-similarity distances are not used as default naturalness measures. Old/new baseline differences cannot be interpreted as model improvement because inputs, script composition and possibly model/software revisions differ.

**Files.** index.html offers original and generated audio side by side. per_recording.csv and configuration_summary.csv give coverage and results; content_details.jsonl preserves hypotheses, normalizations and edit alignments; paired_comparisons.csv gives exploratory paired differences/intervals; egemaps_features.csv contains the 88-feature vectors; channel_sensitivity.csv and speaker_consistency.csv retain their specific limitations. All results stay in this independent experiment.
'''
    (out/'technical_summary.md').write_text(summary,encoding='utf-8')
    write_json(out/'completion.json',{'source_release':str(Path(__file__).resolve().parents[1]),
        'script_sha256':digest(Path(__file__).read_bytes()),
        'cohort_sha256':digest((a.dataset/'cohort.jsonl').read_bytes()),
        'features_sha256':digest((a.run/'tables/metrics.jsonl').read_bytes()),
        'attempts_sha256':digest((a.run/'generation_attempts.jsonl').read_bytes()),
        'source_episode_mapping_sha256':digest((a.dataset/'source_json_crosscheck.json').read_bytes()) if episode_map else None,
        'coverage':status,'manual_validation_complete':False})
    print(json.dumps(json.loads((out/'status.json').read_text())))


def plot_available(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=defaultdict(list)
    for row in rows:
        if row['duration_s'] is not None:
            key='natural_reference' if row['kind']=='natural' else row['configuration_id']
            groups[key].append(row)
    if not groups:return
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for group,values in sorted(groups.items()):
        for ax,metric,label in zip(axes,['duration_s','f0_range_st'],['Duration (seconds)','Robust voiced F0 range (semitones)']):
            x=sorted(r[metric] for r in values if r[metric] is not None)
            if x:ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=f'{group.replace("ऋतिका","Ritika")} (n={len(x)})')
            ax.set(xlabel=label,ylabel='Cumulative fraction')
    axes[0].legend(fontsize=7);fig.suptitle('Available recordings only; descriptive acoustics, different voices/recording conditions')
    fig.tight_layout();fig.savefig(output/'acoustic_distributions.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for i,voice in enumerate(['Amit','Kavya']):
        dev={r['utterance_id']:r for r in rows if r['configuration_id']==f'indic-speak/{voice}/devanagari'}
        roman={r['utterance_id']:r for r in rows if r['configuration_id']==f'indic-speak/{voice}/romanized'}
        for j,backend in enumerate(['asr-large','asr-small']):
            metric=backend+'_wer';ax=axes[i,j]
            matched=[uid for uid in sorted(dev.keys()&roman.keys()) if dev[uid][metric] is not None and roman[uid][metric] is not None]
            for control,label,color in [(False,'Provisional code-switched','#2878a5'),(True,'Provisional Hindi controls','#cc791d')]:
                ids=[uid for uid in matched if ('hindi_only' in dev[uid]['language_condition'].casefold())==control]
                if ids:ax.scatter([dev[uid][metric]*100 for uid in ids],[roman[uid][metric]*100 for uid in ids],s=24,alpha=.65,label=label,color=color)
            bound=max([100]+[max(dev[uid][metric],roman[uid][metric])*100 for uid in matched])*1.03
            ax.plot([0,bound],[0,bound],color='gray',lw=.8,ls='--')
            ax.set(xlim=(0,bound),ylim=(0,bound),xlabel='Devanagari input: proxy WER (%)',ylabel='Romanized input: proxy WER (%)',
                title=f'{voice}; Whisper {backend.removeprefix("asr-")}; matched n={len(matched)}')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same utterance and canonical reference per dot; supplied spellings unverified\nPoints above diagonal have higher Romanized-input ASR error, not proven synthesis error',fontsize=10)
    fig.tight_layout();fig.savefig(output/'script_content_comparison.png',dpi=140)
    for i,voice in enumerate(['Amit','Kavya']):
        dev={r['utterance_id']:r for r in rows if r['configuration_id']==f'indic-speak/{voice}/devanagari'}
        roman={r['utterance_id']:r for r in rows if r['configuration_id']==f'indic-speak/{voice}/romanized'}
        for j,backend in enumerate(['asr-large','asr-small']):
            key=backend+'_wer';ids=[u for u in dev.keys()&roman.keys() if dev[u][key] is not None and roman[u][key] is not None]
            outside=sum(dev[u][key]>1.5 or roman[u][key]>1.5 for u in ids)
            axes[i,j].set(xlim=(0,150),ylim=(0,150))
            axes[i,j].text(.03,.97,f'{outside}/{len(ids)} pairs outside view; all retained in statistics',
                transform=axes[i,j].transAxes,va='top',fontsize=8)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Same-input ASR proxy errors: 0–150% detail view\nFull-range plot and all outliers retained separately; no verified synthesis-error ranking',fontsize=10)
    fig.tight_layout();fig.savefig(output/'script_content_focus.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['acoustics','asr-small','asr-large','nisqa','report'])
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--scope',choices=['pilot','full'],default='pilot');parser.add_argument('--cache',type=Path)
    parser.add_argument('--large-revision');parser.add_argument('--offline',action='store_true')
    parser.add_argument('--nisqa-source',type=Path)
    parser.add_argument('--lab-script',type=Path,default=Path('/proj/speech/tools/speechlab/praat_scripts/speakingRate.praat'))
    a=parser.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if a.action=='acoustics':acoustics(a)
    elif a.action=='report':report(a)
    else:learned(a)
