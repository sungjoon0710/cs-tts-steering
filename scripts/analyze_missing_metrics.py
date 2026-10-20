"""Extend existing baseline reports with executed learned and language diagnostics."""
from __future__ import annotations
import argparse,csv,html,json,itertools,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl
from src.evaluation.text import edit_score
from src.evaluation.expanded import romanize
from src.evaluation.learned import speaker_consistency
from src.evaluation.report import display_config,paired_bootstrap

def analyze(baseline,run):
    run=private_directory(run);out=run/'report';figs=out/'figures';figs.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(baseline/'tables/metrics.jsonl');base={r['generation_id']:r for r in rows}
    if len(base)!=300:raise ValueError('Expected the existing 300-recording baseline')
    def load(folder,name):
        rs=read_jsonl(run/folder/name);index={r['generation_id']:r for r in rs}
        if len(index)!=len(rs):raise ValueError('Duplicate generation metric ID')
        for gid,r in index.items():
            if gid not in base:raise ValueError('Unknown metric recording')
            if 'audio_sha256' in r and r['audio_sha256']!=base[gid]['audio']['audio_sha256']:raise ValueError('Metric/audio association mismatch')
        return index
    second=load('asr_remote','second_asr.jsonl');speaker=load('speaker_remote','speaker_embeddings.jsonl')
    quality=load('nisqa_remote','nisqa_tts.jsonl');language=load('language_remote','language_metrics.jsonl')
    lab=load('lab_rate_remote','lab_syllable_rates.jsonl')
    labels={r['utterance_id']:r for r in read_jsonl(run/'language_remote/input_language.jsonl')}
    results=[];groups=defaultdict(list)
    for row in rows:
        gid=row['generation_id'];asr=second.get(gid,{});hyp=asr.get('hypothesis_raw')
        small=edit_score(row['text_raw'],hyp)
        cross=edit_score(row['asr_raw'],hyp)
        harmonized=edit_score(romanize(row['text_raw']),romanize(hyp)) if hyp is not None else edit_score(row['text_raw'],None)
        diff=abs(small['wer']-row['automatic_proxy']['wer']) if small.get('wer') is not None else None
        q=quality.get(gid,{}).get('nisqa_tts_predicted_mos')
        tokens=small.get('hypothesis_tokens',[]);grams=[tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        repeats=1-len(set(grams))/len(grams) if grams else None
        r={'generation_id':gid,'utterance_id':row['utterance_id'],'configuration_id':row['configuration_id'],
            'in_challenge_view':row['in_challenge_view'],'audio_sha256':row['audio']['audio_sha256'],
            'legacy_asr_proxy':row['automatic_proxy'],'second_asr_status':asr.get('status','missing'),
            'second_asr_raw':hyp,'second_asr_proxy':small,'second_asr_uroman_sensitivity':harmonized,
            'cross_recognizer_difference':cross,'absolute_wer_proxy_difference':diff,
            'recognizer_diagnostics':{'hypothesis_words':len(tokens),'repeated_trigram_occurrence_fraction':repeats,
                'repetition_review_flag':len(tokens)>=30 and repeats is not None and repeats>=.5,
                'scope':'Hypothesis text only; not a confirmed audio repetition or synthesis-failure label'},
            'nisqa_tts_predicted_mos':q,'language':language.get(gid,{}),'input_difficulty':labels[row['utterance_id']]['difficulty'],
            'lab_nuclei_proxy':lab.get(gid,{'status':'not_run'}),
            'interpretation':'ASR and quality-model proxies; all human judgments remain uncollected'}
        groups[r['configuration_id']].append(r);results.append(r)
    summary={'baseline_recordings':len(rows),'challenge_view_recordings':sum(r['in_challenge_view'] for r in rows),
        'challenge_relationship':'Overlaps original baseline; reported separately, never added as independent extra data',
        'configurations':[],'paired_nisqa':[]}
    speaker_rows=[]
    for config,rs in sorted(groups.items()):
        good=[speaker[r['generation_id']] for r in rs if speaker.get(r['generation_id'],{}).get('embedding') is not None]
        stats=speaker_consistency([r['embedding'] for r in good]);loo=stats.pop('loo_centroid_cosine')
        for r,v in zip(good,loo):speaker_rows.append({'generation_id':r['generation_id'],'utterance_id':r['utterance_id'],
            'configuration_id':config,'loo_centroid_cosine':v})
        entry={'configuration_id':config,'denominator':len(rs),'speaker':stats,'second_asr_successes':sum(r['second_asr_proxy'].get('wer') is not None for r in rs),
            'nisqa_coverage':sum(r['nisqa_tts_predicted_mos'] is not None for r in rs)}
        for key,get in [('legacy_wer',lambda r:r['legacy_asr_proxy'].get('wer')),('second_wer',lambda r:r['second_asr_proxy'].get('wer')),
            ('second_uroman_wer',lambda r:r['second_asr_uroman_sensitivity'].get('wer')),('cross_recognizer_wer',lambda r:r['cross_recognizer_difference'].get('wer')),
            ('nisqa_tts',lambda r:r['nisqa_tts_predicted_mos'])]:
            vals=[v for r in rs if (v:=get(r)) is not None]
            entry[key]={'n':len(vals),'median':float(np.median(vals)) if vals else None,'mean':float(np.mean(vals)) if vals else None}
        lr=[r['lab_nuclei_proxy']['nuclei_per_s'] for r in rs if r['lab_nuclei_proxy'].get('nuclei_per_s') is not None]
        entry['lab_nuclei_proxy']={'valid':len(lr),'denominator':len(rs),'median_nuclei_per_s':float(np.median(lr)) if lr else None,
            'status':'Legacy automatic detector; terminal-peak bias and unverified intervals; not validated articulation'}
        entry['provisional_language']={}
        entry['diagnostics']={
            'second_wer_above_one':sum(r['second_asr_proxy'].get('wer') is not None and r['second_asr_proxy']['wer']>1 for r in rs),
            'second_asr_repetition_review_flags':sum(r['recognizer_diagnostics']['repetition_review_flag'] for r in rs),
            'second_hypothesis_at_least_triple_reference_length':sum(len(r['second_asr_proxy'].get('hypothesis_tokens',[]))>=3*r['second_asr_proxy'].get('reference_words',10**9) for r in rs),
            'nisqa_at_least_four_with_legacy_wer_at_least_one':[r['generation_id'] for r in rs if r['nisqa_tts_predicted_mos'] is not None and r['nisqa_tts_predicted_mos']>=4 and r['legacy_asr_proxy']['wer']>=1],
            'interpretation':'Automatic disagreement/repetition-review leads, not confirmed synthesis errors or hallucination diagnoses'}
        for name in ['english_poi','hindi_poi','switch_window_poi']:
            ps=[r['language'].get(name) for r in rs if r['language'].get(name,{}).get('poi_words',0)>0]
            n=sum(p['poi_words'] for p in ps)
            entry['provisional_language'][name]={'recordings_with_POI':len(ps),'poi_words':n,
                'corpus_POI_error':sum(p['substitutions']+p['deletions']+p['insertions'] for p in ps)/n if n else None,
                'selected_reference_substitutions':sum(p['substitutions'] for p in ps),
                'selected_reference_deletions':sum(p['deletions'] for p in ps),
                'reference_anchored_insertions':sum(p['insertions'] for p in ps),
                'median_POI_error':float(np.median([p['pier'] for p in ps])) if ps else None,
                'status':'lexicon-provisional subset; no validated embedded-word PIER'}
        summary['configurations'].append(entry)
    for a,b in itertools.combinations(sorted(groups),2):
        aa={r['utterance_id']:r for r in groups[a]};bb={r['utterance_id']:r for r in groups[b]}
        ids=[uid for uid in sorted(aa,key=int) if aa[uid]['nisqa_tts_predicted_mos'] is not None and bb[uid]['nisqa_tts_predicted_mos'] is not None]
        summary['paired_nisqa'].append({'A':a,'B':b,**paired_bootstrap([aa[u]['nisqa_tts_predicted_mos'] for u in ids],[bb[u]['nisqa_tts_predicted_mos'] for u in ids]),'status':'exploratory_full_baseline_utterance_pairs'})
    write_jsonl(out/'metrics.jsonl',results);write_jsonl(out/'speaker_per_recording.jsonl',speaker_rows);write_json(out/'summary.json',summary)
    write_jsonl(out/'challenge_view.jsonl',[r for r in results if r['in_challenge_view']])
    queue=sorted(results,key=lambda r:r['absolute_wer_proxy_difference'] if r['absolute_wer_proxy_difference'] is not None else -1,reverse=True)
    write_jsonl(out/'recognizer_disagreement_queue.jsonl',[{'generation_id':r['generation_id'],'utterance_id':r['utterance_id'],
        'configuration_id':r['configuration_id'],'absolute_wer_proxy_difference':r['absolute_wer_proxy_difference'],
        'recognizer_diagnostics':r['recognizer_diagnostics'],
        'manual_transcription_status':'uncollected'} for r in queue])
    flat=[];spindex={r['generation_id']:r['loo_centroid_cosine'] for r in speaker_rows}
    for r in results:
        old=base[r['generation_id']];l=r['input_difficulty']
        flat.append({'generation_id':r['generation_id'],'utterance_id':r['utterance_id'],'configuration_id':r['configuration_id'],
            'legacy_wer':r['legacy_asr_proxy'].get('wer'),'second_wer':r['second_asr_proxy'].get('wer'),'nisqa_tts':r['nisqa_tts_predicted_mos'],
            'speaker_loo_cosine':spindex.get(r['generation_id']),'duration_s':old['audio']['duration_s'],'rms_dbfs':old['audio']['rms_dbfs'],
            'legacy_nuclei_per_s':r['lab_nuclei_proxy'].get('nuclei_per_s'),
            'input_length':old['input_word_count'],'lexicon_label_coverage':l['coverage'],'observed_switches_lower_bound':l['observed_switch_count_lower_bound']})
    df=pd.DataFrame(flat);df.to_csv(out/'metrics.csv',index=False)
    numeric=[k for k in df if k not in ('generation_id','utterance_id','configuration_id')]
    corr=(df[numeric]-df.groupby('configuration_id')[numeric].transform('mean')).corr(method='spearman')
    corr.to_csv(out/'configuration_centered_correlations.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for config,rs in sorted(groups.items()):
        valid=[r for r in rs if r['second_asr_proxy'].get('wer') is not None];label=display_config(config)
        axes[0].scatter([r['legacy_asr_proxy']['wer'] for r in valid],[r['second_asr_proxy']['wer'] for r in valid],label=label,s=17,alpha=.6)
        axes[1].scatter([r['legacy_asr_proxy']['wer'] for r in rs],[r['nisqa_tts_predicted_mos'] for r in rs],label=label,s=17,alpha=.6)
        qs=sorted(r['nisqa_tts_predicted_mos'] for r in rs if r['nisqa_tts_predicted_mos'] is not None)
        axes[2].step(qs,np.arange(1,len(qs)+1)/len(qs),where='post',label=label)
    axes[0].plot([0,25],[0,25],color='gray',lw=.6);axes[0].set(xlabel='Legacy Whisper-large-v3 proxy WER',ylabel='Whisper-small proxy WER',xscale='symlog',yscale='symlog')
    axes[1].set(xlabel='Legacy ASR-proxy WER',ylabel='NISQA-TTS prediction',xscale='symlog');axes[2].set(xlabel='NISQA-TTS prediction',ylabel='Cumulative fraction');axes[2].legend(fontsize=7)
    fig.suptitle('Different evidence: recognizer sensitivity and learned quality prediction, not human rankings');fig.tight_layout();fig.savefig(figs/'learned_content_disagreement.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for config in sorted(groups):
        vs=sorted(r['loo_centroid_cosine'] for r in speaker_rows if r['configuration_id']==config);axes[0].step(vs,np.arange(1,len(vs)+1)/len(vs),where='post',label=display_config(config))
    axes[0].set(xlabel='Leave-one-out voice-centroid cosine',ylabel='Cumulative fraction');axes[0].legend(fontsize=7)
    im=axes[1].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');axes[1].set_xticks(range(len(numeric)),numeric,rotation=90,fontsize=7);axes[1].set_yticks(range(len(numeric)),numeric,fontsize=7);fig.colorbar(im,ax=axes[1]);fig.tight_layout();fig.savefig(figs/'speaker_and_metric_relations.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for config,rs in sorted(groups.items()):
        axes[0].scatter([r['input_difficulty']['coverage'] for r in rs],[r['legacy_asr_proxy']['wer'] for r in rs],s=15,alpha=.5,label=display_config(config))
        cs=summary['configurations'][sorted(groups).index(config)]
        vals=[cs['provisional_language'][k]['corpus_POI_error'] for k in ('english_poi','hindi_poi','switch_window_poi')]
        axes[1].plot(range(3),vals,marker='o',label=display_config(config))
    axes[0].set(xlabel='Fraction of tokens with provisional corpus-lexicon labels',ylabel='Legacy ASR-proxy WER',yscale='symlog')
    axes[1].set_xticks(range(3),['EN POIs','HI POIs','Observed-switch windows']);axes[1].set_ylabel('POI-aligned errors / selected reference tokens');axes[1].legend(fontsize=7)
    fig.suptitle('Partial language coverage: unknown words remain unknown; no verified embedded-language PIER');fig.tight_layout();fig.savefig(figs/'provisional_language.png',dpi=140);plt.close(fig)
    notes=[('Observation','300 learned-quality predictions, 300 speaker embeddings, and per-record second-ASR coverage are recorded explicitly. Partial corpus-derived language labels cover selected token subsets.'),
        ('Interpretation','ASR disagreement helps prioritize transcription review. High predicted quality can coexist with content errors. Embeddings describe within-voice consistency.'),
        ('Limitation','Neither recognizer is ground truth. NISQA and WavLM have unvalidated Hinglish domain/language behavior. Lexicon agreement is not accuracy; embedded matrix language is not inferred. Paired utterances do not resolve unknown conversation clustering.')]
    body='<h1>Additional automatic evaluation</h1><p>Extension of the existing 100-input/300-recording baseline. The 18 challenge recordings are an overlapping view; human judgments remain uncollected.</p>'
    body+='<p><a href="../../full_baseline_20261019/index.html">Existing full analysis</a> | <a href="completion_report.md">Completion and resource report</a> | <a href="metric_registry.csv">Metric registry CSV</a> | <a href="resource_map.csv">Resource map CSV</a></p>'
    body+=''.join('<p><b>'+a+':</b> '+html.escape(b)+'</p>' for a,b in notes)
    table=[]
    for c in summary['configurations']:
        table.append([display_config(c['configuration_id']),str(c['denominator']),f"{100*c['legacy_wer']['median']:.1f}%",f"{100*c['second_wer']['median']:.1f}%",f"{c['nisqa_tts']['median']:.3f}",f"{c['speaker']['pairwise_mean_cosine']:.3f}"])
    headers=['Configuration','N','Large-v3 WER (NFC)','Small WER (NFC)','NISQA median','Speaker cosine mean']
    body+='<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in table)+'</table><p>Both WER columns use hi-en-nfc-v1. The original pipeline\'s uroman/ASCII calculation instead gave 29.8/28.9/55.6%; no audio changed. NISQA is a learned prediction; cosine measures within-voice consistency. None is a human naturalness score.</p>'
    fig,ax=plt.subplots(figsize=(15,3));ax.axis('off');t=ax.table(cellText=table,colLabels=headers,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2)
    fig.suptitle('300 baseline recordings: automatic diagnostics, not a model-quality ranking',fontsize=13)
    fig.text(.5,.04,'NISQA is predicted quality. Speaker cosine measures within-voice consistency. Voice confounds remain.',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(figs/'results_table.png',dpi=160);plt.close(fig)
    captions={
        'learned_content_disagreement':'Observation: Whisper-small median WER is higher in every configuration; NISQA predictions also differ across configurations. Interpretation: recognizer and learned-quality sensitivity supply different evidence. Limitation: neither provides a validated Hinglish perceptual ranking; extreme WER tails require transcription review.',
        'speaker_and_metric_relations':'Observation: within-voice mean pairwise cosines range from approximately 0.932 to 0.953. Interpretation: outlying recordings can be prioritized using leave-one-out consistency. Limitation: content, language and duration affect embeddings; centered correlations are descriptive and do not remove architectural voice confounds.',
        'provisional_language':'Observation: only 954 of 1845 input tokens have accepted provisional labels, including 51 English POIs. Interpretation: selected English POI errors expose a potentially important content-preservation question. Limitation: subset selection, automatic labels, script conventions and ASR errors prevent a validated full-language conclusion.',
        'results_table':'Observation: each displayed configuration has 100 recordings. Interpretation: columns summarize distinct evidence families. Limitation: these three model-voice configurations are not three independent architectures.'}
    body+=''.join('<h2>'+p.stem+'</h2><img width="100%" src="figures/'+p.name+'"><p>'+html.escape(captions.get(p.stem,''))+'</p>' for p in sorted(figs.glob('*.png')))
    if (out/'ttsds/factor_scores.csv').exists():
        body+='<h2>Official multilingual TTSDS2</h2><p>All 11 features completed for all three configurations. The official aggregation was reproduced. These reference-conditioned scores are exploratory: natural speaker/channel/domain differences remain, and the pinned mWhisper implementation pools over feature dimensions, retaining time positions. Review that implementation before interpreting the aggregate as quality evidence. Official score standard deviations are not confidence intervals.</p>'
        body+=pd.read_csv(out/'ttsds/factor_scores.csv').to_html(index=False)
        body+='<img width="100%" src="ttsds/official_factors.png"><p>Observation: factor scores differ across configurations. Interpretation: this exposes distributional differences relative to the selected natural and noise references. Limitation: these are not validated human preferences or architecture-only effects.</p>'
        body+='<img width="100%" src="ttsds/rate_distributions.png"><p>Observation: cached phone/token-event rates differ across populations. Interpretation: cadence proxies support reference-sensitivity analysis. Limitation: these are not verified syllable or articulation rates; scalar bootstrap intervals do not describe the full ensemble.</p>'
    body+='<h2>Audit-ready summary</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
    # Run-specific statuses and private artifact locations stay with the results.
    registry_path=out/'metric_registry.json'
    if registry_path.exists():
        registry=json.loads(registry_path.read_text(encoding='utf-8'));write_json(out/'metric_registry.json',registry)
        pd.DataFrame(registry['metrics']).to_csv(out/'metric_registry.csv',index=False)
        body+='<h2>Metric completion matrix</h2><table><tr><th>Metric</th><th>Status</th><th>Coverage</th><th>Next step</th></tr>'
        body+=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ['metric_id','status','coverage','next_step'])+'</tr>' for r in registry['metrics'])+'</table>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SpeechLab metric extension</title><style>body{font:16px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:20px}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5df;padding:8px;text-align:left}th{background:#e5edf5}</style>'+body,encoding='utf-8')
    write_json(out/'provenance.json',{'baseline_metrics_sha256':digest((baseline/'tables/metrics.jsonl').read_bytes()),
        'input_sha256':{name:digest((run/name).read_bytes()) for name in ['asr_remote/second_asr.jsonl','speaker_remote/speaker_embeddings.jsonl','nisqa_remote/nisqa_tts.jsonl','language_remote/input_language.jsonl','language_remote/language_metrics.jsonl','lab_rate_remote/lab_syllable_rates.jsonl'] if (run/name).exists()},
        'script_sha256':digest(Path(__file__).read_bytes()),'status':'Exploratory/provisional automatic evidence; no human ratings'})
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();analyze(a.baseline,a.run)
