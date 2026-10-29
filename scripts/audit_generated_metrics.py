"""Reconcile cached generated-only metrics and estimate source-cluster sensitivity.

No speech/model inference, natural-reference scores, or raw transcript export.
"""
import argparse,csv,itertools,json,math,sys
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import importlib.metadata as metadata
import numpy as np
from scipy.stats import spearmanr
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl
from src.evaluation.text import edit_score,normalize
from src.evaluation.expanded import romanize,raw_score
from src.evaluation.language import language_errors,difficulty
from src.evaluation.learned import speaker_consistency
from src.evaluation.provenance import paired_cluster_intervals


def require(ok,message):
    if not ok:raise ValueError(message)


def index(rows,key,label):
    out={r[key]:r for r in rows};require(len(out)==len(rows),'Duplicate '+label)
    return out


def close(a,b):
    return a is None and b is None or isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)


def csvout(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def deleted_positions(score):
    """Deleted reference-token indices, not confirmed acoustic omissions."""
    return {op['reference_index'] for op in score['alignment'] if op['operation']=='delete'}


def cluster_weights(clusters,repeats=3000):
    names,inverse=np.unique(clusters,return_inverse=True);g=len(names)
    return np.random.default_rng(1234).multinomial(g,np.repeat(1/g,g),size=repeats)[:,inverse].astype(float)


def ratio_difference(a_errors,b_errors,denominators,weights):
    den=np.asarray(denominators,float);difference=np.asarray(b_errors,float)-np.asarray(a_errors,float)
    require(np.all(den>0),'Invalid reference denominator')
    draws=(weights@difference)/(weights@den)
    return {'B_minus_A':float(difference.sum()/den.sum()),'bootstrap_95ci':np.quantile(draws,[.025,.975]).tolist()}


def main(a):
    out=private_directory(a.output);require(not out.exists(),'Use a new output directory');out.mkdir(parents=True)
    files={'baseline':a.baseline/'tables/metrics.jsonl','small':a.extension/'asr_remote/second_asr.jsonl',
        'speaker':a.extension/'speaker_remote/speaker_embeddings.jsonl','nisqa':a.extension/'nisqa_remote/nisqa_tts.jsonl',
        'language':a.extension/'language_remote/language_metrics.jsonl','labels':a.extension/'language_remote/input_language.jsonl',
        'legacy_rate':a.extension/'lab_rate_remote/lab_syllable_rates.jsonl',
        'corrected_rate':a.audio_audit/'rate_terminal_v2/lab_syllable_rates.jsonl','sources':a.sources,
        'prior_derived':a.extension/'report/metrics.jsonl','prior_speaker':a.extension/'report/speaker_per_recording.jsonl',
        'rate_cache':a.extension/'ttsds_scalar_remote/scalar_cache.json','rate_inputs':a.extension/'ttsds_remote/inputs.json'}
    loaded={k:read_jsonl(p) for k,p in files.items() if p.suffix=='.jsonl'};rows=loaded['baseline']
    base=index(rows,'generation_id','baseline IDs');require(len(base)==300,'Expected300 recordings')
    ids=set(base);by={};mismatches=[];checks=Counter()
    for name in ['small','speaker','nisqa','language','legacy_rate','corrected_rate']:
        by[name]=index(loaded[name],'generation_id',name)
        require(set(by[name])==ids,'Coverage mismatch: '+name)
        for gid,r in by[name].items():
            require(all(r[k]==base[gid][k] for k in ('utterance_id','configuration_id','in_challenge_view')),'Record alignment: '+name)
            if 'audio_sha256' in r:require(r['audio_sha256']==base[gid]['audio']['audio_sha256'],'Audio hash: '+name)
    derived=index(loaded['prior_derived'],'generation_id','prior derived records')
    prior_speaker=index(loaded['prior_speaker'],'generation_id','prior speaker records')
    require(set(derived)==ids==set(prior_speaker),'Prior derived metric coverage')
    rate_cache=json.loads(files['rate_cache'].read_text(encoding='utf-8'))
    rate_inputs=json.loads(files['rate_inputs'].read_text(encoding='utf-8'))
    require(rate_cache['official_inputs_sha256']==digest(files['rate_inputs'].read_bytes()),'Event-rate source manifest hash')
    rate_generations=index(rate_inputs['generations'],'generation_id','event-rate input records')
    require(set(rate_generations)==ids and all(rate_generations[k]['audio_sha256']==base[k]['audio']['audio_sha256'] for k in ids),'Event-rate generation alignment')
    event_rates={}
    for config in {r['configuration_id'] for r in rows}:
        cache=rate_cache['datasets'][config];require(len(cache['ids'])==100 and len(set(cache['ids']))==100,'Event-rate IDs')
        for source,field in [('Allosaurus SR','phone_events_per_s'),('mHubert-147 Token SR','token_events_per_s')]:
            values=cache['factors'][source]['values'];require(len(values)==100 and np.isfinite(values).all(),'Event-rate values')
            for uid,value in zip(cache['ids'],values):event_rates.setdefault((config,uid),{})[field]=float(value)
    sources=index(loaded['sources'],'utterance_id','sources');labels=index(loaded['labels'],'utterance_id','labels')
    require(len(sources)==100 and set(sources)==set(labels),'Input/source/label coverage')
    groups=defaultdict(dict);flat=[];pooled_nisqa=0
    @lru_cache(maxsize=1500)
    def roman(t):return romanize(t)
    def verify(old,new,fields,scope):
        for k in fields:
            checks[scope]+=1
            if not close(old.get(k),new.get(k)):mismatches.append({'scope':scope,'field':k,'old':old.get(k),'new':new.get(k)})
    for r in rows:
        gid=r['generation_id'];uid=r['utterance_id'];config=r['configuration_id'];audio=r['audio'];ref=r['text_raw']
        require(uid not in groups[config],'Repeated configuration/input')
        require(digest(ref.encode())==sources[uid]['text_sha256']==labels[uid]['text_sha256'],'Input text digest')
        large=edit_score(ref,r['asr_raw']);small=edit_score(ref,by['small'][gid]['hypothesis_raw'])
        fields=['wer','cer','reference_words','reference_characters','substitutions','deletions','insertions','character_errors']
        verify(r['automatic_proxy'],large,fields,'large_content')
        verify(derived[gid]['second_asr_proxy'],small,fields,'small_content')
        cross=edit_score(r['asr_raw'],by['small'][gid]['hypothesis_raw'])
        verify(derived[gid]['cross_recognizer_difference'],cross,['wer','cer'],'recognizer_disagreement')
        verify(r['raw_script_sensitive'],raw_score(ref,r['asr_raw']),['wer','cer'],'raw_script')
        harmonized=edit_score(roman(ref),roman(r['asr_raw']))
        verify(r['uroman_sensitivity'],harmonized,['wer','cer'],'uroman')
        lang=language_errors(ref,r['asr_raw'],labels[uid]['labels'])
        for key in ['english_poi','hindi_poi','switch_window_poi']:
            verify(by['language'][gid][key],lang[key],['pier','poi_words','substitutions','deletions','insertions'],'provisional_poi')
        f=np.array([p['hz'] for p in audio['f0_track'] if p['hz'] is not None],float)
        median=float(np.median(f));st=12*np.log2(f/median)
        verify(audio,{'f0_median_hz':median,'f0_std_st':float(st.std()),'f0_range_st':float(np.quantile(st,.95)-np.quantile(st,.05)),
            'duration_s':audio['frames']/audio['sample_rate']},['duration_s','f0_median_hz','f0_std_st','f0_range_st'],'cached_audio_arithmetic')
        intervals=audio['intensity_intervals'];sounding=[p for p in intervals if p['label']=='sounding']
        require(sounding and all(p['end_s']>=p['start_s'] for p in intervals),'Invalid intensity segmentation')
        for threshold in [.15,.2,.25]:
            lengths=[p['end_s']-p['start_s'] for p in intervals if p['label']=='silent' and p['start_s']>=sounding[0]['start_s'] and p['end_s']<=sounding[-1]['end_s'] and p['end_s']-p['start_s']+1e-9>=threshold]
            verify(audio['pause_sensitivity'][str(threshold)],{'count':len(lengths),'total_s':sum(lengths),'median_s':float(np.median(lengths)) if lengths else None,'p90_s':float(np.quantile(lengths,.9)) if lengths else None},['count','total_s','median_s','p90_s'],'cached_pause_arithmetic')
        features=audio['egemaps']['features'];require(len(features)==88 and all(v is not None and math.isfinite(v) for v in features.values()),'Invalid eGeMAPS vector')
        n=by['nisqa'][gid]['nisqa_tts_predicted_mos'];require(math.isfinite(n),'Invalid NISQA prediction')
        verify(derived[gid],{'nisqa_tts_predicted_mos':n},['nisqa_tts_predicted_mos'],'nisqa_record_join')
        pooled_nisqa+=not 1<=n<=5
        oldrate=by['legacy_rate'][gid];rate=by['corrected_rate'][gid]
        delta=rate['detected_nuclei']-oldrate['detected_nuclei'];require(delta in (0,1),'Unexpected rate change beyond terminal candidate')
        counts=large['substitutions']+large['deletions']+large['insertions']
        s_counts=small['substitutions']+small['deletions']+small['insertions']
        asr_rate=len(large['hypothesis_tokens'])/audio['duration_s']
        articulation_rate=len(large['hypothesis_tokens'])/audio['articulation_time_estimate_s']
        verify(r['asr_indicators'],{'asr_speaking_rate_wps':asr_rate,'asr_articulation_rate_proxy_wps':articulation_rate},
            ['asr_speaking_rate_wps','asr_articulation_rate_proxy_wps'],'asr_rate_arithmetic')
        row={'generation_id':gid,'utterance_id':uid,'configuration_id':config,'in_challenge_view':r['in_challenge_view'],
            'source_episode':sources[uid]['episode_dialogue'].split('-')[0],'source_dialogue':sources[uid]['episode_dialogue'],
            'audio_sha256':audio['audio_sha256'],'reference_words':large['reference_words'],'reference_characters':large['reference_characters'],
            'large_word_errors':counts,'large_character_errors':large['character_errors'],'small_word_errors':s_counts,'small_character_errors':small['character_errors'],
            'wer_large':large['wer'],'cer_large':large['cer'],'wer_small':small['wer'],'cer_small':small['cer'],
            'asr_disagreement_wer':cross['wer'],
            'nisqa_tts_prediction':n,'duration_s':audio['duration_s'],'f0_range_st':audio['f0_range_st'],
            'pause_proportion':audio['internal_pause_proportion'],'rms_dbfs':audio['rms_dbfs'],
            'legacy_nuclei_per_s':oldrate['nuclei_per_s'],'corrected_nuclei_per_s':rate['nuclei_per_s'],
            'legacy_nuclei':oldrate['detected_nuclei'],'corrected_nuclei':rate['detected_nuclei'],
            'rate_delta_nuclei':delta,'label_coverage':difficulty(labels[uid]['labels'])['coverage'],
            'asr_words_per_s':asr_rate,'asr_words_per_estimated_articulation_s':articulation_rate,
            **event_rates[(config,uid)]}
        groups[config][uid]=row;flat.append(row)
    expected=set(sources)
    require(len(groups)==3 and all(set(g)==expected for g in groups.values()),'Not three matched100-input configurations')
    # Recompute speaker summaries from the cached vectors, not cosine pairs as independent observations.
    speaker_groups={};matrices={};summaries=[]
    oldsummary=json.loads((a.extension/'report/summary.json').read_text(encoding='utf-8'))
    oldspeaker={r['configuration_id']:r['speaker'] for r in oldsummary['configurations']}
    ordered=sorted(expected,key=int)
    for config,g in sorted(groups.items()):
        vectors=np.asarray([by['speaker'][g[u]['generation_id']]['embedding'] for u in ordered],float)
        require(vectors.shape==(100,512) and np.isfinite(vectors).all(),'Speaker vector shape')
        require(np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))<1e-5,'Speaker vector norm')
        result=speaker_consistency(vectors,repeats=200)
        verify(oldspeaker[config],result,['pairwise_mean_cosine','pairwise_q05_cosine'],'speaker_aggregation')
        e=vectors/np.linalg.norm(vectors,axis=1)[:,None];matrices[config]=e@e.T-np.eye(100)
        for u,v in zip(ordered,result['loo_centroid_cosine']):
            g[u]['speaker_loo_cosine']=float(v)
            verify(prior_speaker[g[u]['generation_id']],{'loo_centroid_cosine':float(v)},['loo_centroid_cosine'],'speaker_record_join')
        speaker_groups[config]=result
        rs=[g[u] for u in ordered];tot_words=sum(r['reference_words'] for r in rs);tot_chars=sum(r['reference_characters'] for r in rs)
        summaries.append({'configuration_id':config,'recordings':100,'reference_words':tot_words,
            'large_median_wer':float(np.median([r['wer_large'] for r in rs])),
            'large_macro_mean_wer':float(np.mean([r['wer_large'] for r in rs])),
            'large_corpus_wer':sum(r['large_word_errors'] for r in rs)/tot_words,
            'large_corpus_cer':sum(r['large_character_errors'] for r in rs)/tot_chars,
            'small_median_wer':float(np.median([r['wer_small'] for r in rs])),
            'small_corpus_wer':sum(r['small_word_errors'] for r in rs)/tot_words,
            'nisqa_median':float(np.median([r['nisqa_tts_prediction'] for r in rs])),
            'legacy_rate_valid':sum(r['legacy_nuclei_per_s'] is not None for r in rs),
            'corrected_rate_valid':sum(r['corrected_nuclei_per_s'] is not None for r in rs),
            'legacy_rate_median':float(np.median([r['legacy_nuclei_per_s'] for r in rs if r['legacy_nuclei_per_s'] is not None])),
            'corrected_rate_median':float(np.median([r['corrected_nuclei_per_s'] for r in rs])),
            'terminal_nuclei_added':sum(r['rate_delta_nuclei'] for r in rs),'speaker_pairwise_mean':result['pairwise_mean_cosine'],
            'within_config_rho_nisqa_wer':float(spearmanr([r['nisqa_tts_prediction'] for r in rs],[r['wer_large'] for r in rs]).statistic),
            'within_config_rho_speaker_duration':float(spearmanr([r['speaker_loo_cosine'] for r in rs],[r['duration_s'] for r in rs]).statistic)})
    metrics=['wer_large','cer_large','wer_small','nisqa_tts_prediction','duration_s','f0_range_st','pause_proportion','corrected_nuclei_per_s','speaker_loo_cosine']
    paired=[];corpus=[];speaker_differences=[]
    for cluster_key in ['source_episode','source_dialogue']:
        clusters=[next(iter(groups.values()))[u][cluster_key] for u in ordered];weights=cluster_weights(clusters,a.repeats)
        for A,B in itertools.combinations(sorted(groups),2):
            aa=[groups[A][u] for u in ordered];bb=[groups[B][u] for u in ordered]
            diffs=np.asarray([[y[m]-x[m] for m in metrics] for x,y in zip(aa,bb)])
            intervals=paired_cluster_intervals(diffs,clusters,repeats=a.repeats)
            for j,(m,ci) in enumerate(zip(metrics,intervals)):
                values=diffs[:,j];influential=int(np.argmax(np.abs(values-values.mean())))
                paired.append({'configuration_A':A,'configuration_B':B,'metric':m,'cluster_level':cluster_key,
                    'matched_inputs':ci['matched_recordings'],'clusters':ci['source_clusters_sampled'],
                    'mean_B_minus_A':ci['mean_B_minus_A'],'ci_low':ci['bootstrap_95ci'][0],'ci_high':ci['bootstrap_95ci'][1],
                    'median_paired_difference':float(np.median(values)),'B_lower':int((values<0).sum()),'ties':int((values==0).sum()),'B_higher':int((values>0).sum()),
                    'largest_mean_influence_input':ordered[influential],'mean_without_that_input':float(np.delete(values,influential).mean()),
                    'status':'exploratory source-cluster bootstrap; no quality ranking; no multiple-comparison significance claims'})
            for name,den in [('wer','reference_words'),('cer','reference_characters')]:
                for backend in ['large','small']:
                    errors=backend+('_word_errors' if name=='wer' else '_character_errors')
                    r=ratio_difference([x[errors] for x in aa],[x[errors] for x in bb],[x[den] for x in aa],weights)
                    corpus.append({'configuration_A':A,'configuration_B':B,'metric':backend+'_corpus_'+name,
                        'cluster_level':cluster_key,'clusters':len(set(clusters)),'B_minus_A':r['B_minus_A'],
                        'ci_low':r['bootstrap_95ci'][0],'ci_high':r['bootstrap_95ci'][1]})
            difference=matrices[B]-matrices[A];den=weights.sum(axis=1)**2-(weights*weights).sum(axis=1)
            draws=((weights@difference)*weights).sum(axis=1)/den
            speaker_differences.append({'configuration_A':A,'configuration_B':B,'cluster_level':cluster_key,
                'pairwise_mean_B_minus_A':speaker_groups[B]['pairwise_mean_cosine']-speaker_groups[A]['pairwise_mean_cosine'],
                'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),
                'method':'Matched source-cluster resampling of recording vectors; self/same-original-index pairs excluded; not perceptual identity'})
    require(not mismatches,'Cached computation mismatches; inspect '+str(mismatches[:2]))
    csvout(out/'generated_metrics.csv',flat);csvout(out/'configuration_summary.csv',summaries)
    csvout(out/'paired_cluster_comparisons.csv',paired);csvout(out/'paired_corpus_error_comparisons.csv',corpus)
    csvout(out/'paired_speaker_comparisons.csv',speaker_differences)
    write_json(out/'audit_validation.json',{'records':300,'matched_inputs':100,'challenge_view_records':sum(r['in_challenge_view'] for r in rows),
        'source_episodes':len({r['source_episode'] for r in flat}),'source_dialogues':len({r['source_dialogue'] for r in flat}),
        'recomputed_field_checks':dict(checks),'mismatches':mismatches,'eGeMAPS_complete_finite_vectors':300,
        'nisqa_predictions_outside_nominal_1_5':pooled_nisqa,'prediction_policy':'Retained unbounded predictions; not clipped to collected-MOS scale',
        'raw_event_rates':'Two generated-only arrays reused after source hash/ID verification; no natural values, reference distances or TTSDS2 scores used',
        'bootstrap_replicates':a.repeats,'packages':{p:metadata.version(p) for p in ['numpy','scipy','rapidfuzz','uroman']},
        'source_hashes':{k:digest(p.read_bytes()) for k,p in files.items()},'script_sha256':digest(Path(__file__).read_bytes()),
        'original_audio_available':False,'matched_natural_evaluation':'not executed; missing original recordings',
        'unmatched_natural_rankings_reused':False,'manual_transcript_validation':False,'human_quality_ratings':False})
    print(json.dumps({'records':len(flat),'cached_mismatches':len(mismatches),'checks':dict(checks),'outputs':str(out)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','extension','audio-audit','sources','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3000);main(p.parse_args())
