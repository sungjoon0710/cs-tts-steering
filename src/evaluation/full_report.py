"""Full-baseline reporting with overlapping diagnostic views and explicit limits."""
from __future__ import annotations

import csv
import html
import itertools
import json
import shutil
from collections import Counter,defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from .common import read_jsonl,write_json
from .report import display_config,paired_bootstrap,value

METRICS=['automatic_proxy.wer','automatic_proxy.cer','raw_script_sensitive.wer','uroman_sensitivity.wer',
         'audio.duration_s','audio.f0_median_hz','audio.f0_range_st','audio.f0_std_st',
         'audio.rms_dbfs','audio.internal_pause_proportion','audio.voiced_frame_fraction',
         'audio.lab_voice_quality.jitter_local','audio.lab_voice_quality.shimmer_local','audio.lab_voice_quality.hnr_mean_db',
         'asr_indicators.asr_speaking_rate_wps','asr_indicators.asr_articulation_rate_proxy_wps']


def flatten(obj,prefix=''):
    output={}
    for key,v in obj.items():
        path=prefix+'.'+key if prefix else key
        if isinstance(v,dict):output.update(flatten(v,path))
        elif isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v):output[path]=float(v)
    return output


def safe_stats(values):
    a=np.array([v for v in values if v is not None and np.isfinite(v)],float)
    if not len(a):return {'n':0,'mean':None,'median':None,'q05':None,'q95':None}
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'q05':float(np.quantile(a,.05)),'q95':float(np.quantile(a,.95))}


def full_report(run):
    run=Path(run);rows=read_jsonl(run/'tables/metrics.jsonl');out=run/'report';figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    settings=json.loads((run/'config.json').read_text(encoding='utf-8')) if (run/'config.json').exists() else {}
    if len({r['generation_id'] for r in rows})!=len(rows):raise ValueError('Duplicate generation IDs')
    groups=defaultdict(list)
    for r in rows:groups[r['configuration_id']].append(r)
    summary={'status':'full automatic baseline; all quality interpretations provisional/exploratory',
        'recordings':len(rows),'utterances':len({r['utterance_id'] for r in rows}),
        'challenge_view_recordings':sum(r['in_challenge_view'] for r in rows),
        'cohort_note':'Challenge view is an overlapping subset, not extra data. Baseline denominator includes it once.',
        'human_judgments':'pending; no automatic score establishes naturalness',
        'language_analysis':'pending reliable corpus joins or validated/provisional tagger',
        'configurations':[], 'paired_differences':[]}
    for config,rs in sorted(groups.items()):
        entry={'configuration_id':config,'denominator':len(rs),'audio_statuses':dict(Counter(r['audio']['status'] for r in rs)),
            'automatic_warning_counts':{k:sum(r['automatic_warnings'].get(k) is True for r in rs) for k in rs[0]['automatic_warnings']},
            'warning_interpretation':'Automatic candidates, not confirmed perceptual failures',
            'human_confirmed_failure_rate':None,'human_review_coverage':0,
            'metrics':{k:safe_stats([value(r,k) for r in rs]) for k in METRICS},
            'egemaps_successes':sum(r['audio'].get('egemaps',{}).get('status')=='exploratory' for r in rs)}
        valid=[r['automatic_proxy'] for r in rs if r['automatic_proxy'].get('wer') is not None]
        entry['corpus_proxy_wer']=sum(v['substitutions']+v['deletions']+v['insertions'] for v in valid)/sum(v['reference_words'] for v in valid) if valid else None
        entry['corpus_proxy_cer']=sum(v['character_errors'] for v in valid)/sum(v['reference_characters'] for v in valid) if valid else None
        entry['content_metric_coverage']=len(valid)
        summary['configurations'].append(entry)
    for ca,cb in itertools.combinations(sorted(groups),2):
        aa={r['utterance_id']:r for r in groups[ca]};bb={r['utterance_id']:r for r in groups[cb]}
        for metric in METRICS:
            ids=sorted((uid for uid in aa.keys()&bb.keys() if value(aa[uid],metric) is not None and value(bb[uid],metric) is not None),key=int)
            if not ids:continue
            a=np.array([value(aa[u],metric) for u in ids]);b=np.array([value(bb[u],metric) for u in ids]);d=b-a
            influence=int(np.argmax(np.abs(d-d.mean())))
            summary['paired_differences'].append({'configuration_A':ca,'configuration_B':cb,'metric':metric,
                **paired_bootstrap(a,b,seed=settings.get('seed',1234),repeats=settings.get('bootstrap_repeats',2000)),
                'status':'exploratory baseline; conversation clustering unavailable',
                'largest_mean_influence_utterance_id':ids[influence],
                'mean_without_that_one_item':float(np.delete(d,influence).mean()) if len(d)>1 else None,
                'sensitivity_note':'Leave-one-out influence only; the item remains in primary estimates'})
    write_json(out/'summary.json',summary)
    difficulty=[]
    for config,rs in sorted(groups.items()):
        for kind in ('script_profile','length_bin'):
            bins=defaultdict(list)
            for r in rs:
                label=r['script_profile'] if kind=='script_profile' else ('1-7' if r['input_word_count']<=7 else '8-20' if r['input_word_count']<=20 else '21+')
                bins[label].append(r)
            for label,pop in sorted(bins.items()):
                difficulty.append({'configuration_id':config,'condition_type':kind,'condition':label,'n':len(pop),
                    'proxy_wer':safe_stats([value(r,'automatic_proxy.wer') for r in pop]),
                    'warnings':{k:sum(r['automatic_warnings'].get(k) is True for r in pop) for k in pop[0]['automatic_warnings']},
                    'interpretation':'Script is not language; length uses intended normalized tokens; exploratory'})
    write_json(out/'difficulty_conditions.json',difficulty)
    flat=[]
    for r in rows:
        entry={'generation_id':r['generation_id'],'utterance_id':r['utterance_id'],'configuration_id':r['configuration_id'],
               'in_challenge_view':r['in_challenge_view'],'script_profile':r['script_profile'],'input_word_count':r['input_word_count']}
        for k in METRICS:entry[k]=value(r,k)
        entry.update({'egemaps.'+k:v for k,v in r['audio'].get('egemaps',{}).get('features',{}).items()})
        flat.append(entry)
    df=pd.DataFrame(flat);df.to_csv(out/'metrics.csv',index=False)
    # Rank correlations after removing configuration means; this is exploratory,
    # not a causal separation of speaker and linguistic content.
    acoustic=[k for k in df if k.startswith(('audio.','egemaps.')) and df[k].notna().sum()>=30 and df[k].nunique()>1]
    centered=df[acoustic]-df.groupby('configuration_id')[acoustic].transform('mean')
    corr=centered.corr(method='spearman',min_periods=30);corr.to_csv(out/'acoustic_rank_correlations.csv')
    redundant=[]
    for i,a in enumerate(acoustic):
        for b in acoustic[i+1:]:
            rho=corr.loc[a,b]
            if np.isfinite(rho) and abs(rho)>=.95:redundant.append({'feature_A':a,'feature_B':b,'rho':float(rho)})
    redundant.sort(key=lambda r:abs(r['rho']),reverse=True)
    write_json(out/'redundancy.json',{'status':'exploratory; configuration-centered Spearman; correlated items unresolved',
        'acoustic_features':len(acoustic),'high_correlation_pairs':redundant,
        'interpretation':'Redundancy does not establish perceptual relevance; do not select features using absent human labels'})
    selected=METRICS[:12]
    fig,axes=plt.subplots(3,4,figsize=(17,11))
    for ax,metric in zip(axes.flat,selected):
        for config,rs in sorted(groups.items()):
            vs=sorted(v for r in rs if (v:=value(r,metric)) is not None)
            if vs:ax.step(vs,np.arange(1,len(vs)+1)/len(vs),where='post',label=display_config(config))
        if metric.endswith(('.wer','.cer')):
            ax.set_xscale('symlog',linthresh=1)
        ax.set(xlabel=metric,ylabel='Cumulative fraction');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);fig.suptitle('All 300 baseline recordings: proxy/content and descriptive acoustic distributions')
    fig.tight_layout();fig.savefig(figdir/'baseline_distributions.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,metric in zip(axes.flat,['automatic_proxy.wer','uroman_sensitivity.wer','audio.duration_s','audio.f0_range_st','audio.internal_pause_proportion','audio.rms_dbfs']):
        cs=[c for c in summary['paired_differences'] if c['metric']==metric]
        for i,c in enumerate(cs):
            v=c['mean_B_minus_A'];lo,hi=c['mean_difference_bootstrap_95ci']
            ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o')
        ax.axvline(0,color='gray',lw=.7);ax.set_yticks(range(len(cs)),[display_config(c['configuration_B'])+' minus\n'+display_config(c['configuration_A']) for c in cs],fontsize=7);ax.set_xlabel(metric)
    fig.suptitle('Matched utterance differences and exploratory bootstrap intervals');fig.tight_layout();fig.savefig(figdir/'paired_differences.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for config,rs in sorted(groups.items()):
        valid=[r for r in rs if value(r,'automatic_proxy.wer') is not None]
        axes[0].scatter([r['input_word_count'] for r in valid],[r['automatic_proxy']['wer'] for r in valid],s=16,alpha=.6,label=display_config(config))
        axes[1].scatter([r['automatic_proxy']['wer'] for r in valid],[r['uroman_sensitivity']['wer'] for r in valid],s=16,alpha=.6,label=display_config(config))
    axes[0].set(xlabel='Intended normalized token count',ylabel='Strict ASR-proxy WER');axes[1].set(xlabel='Strict ASR-proxy WER',ylabel='uroman sensitivity WER')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/'difficulty_and_normalization.png',dpi=140);plt.close(fig)
    warning_keys=['token_limit_warning','legacy_short_warning','asr_omission_warning','asr_insertion_warning','asr_repetition_warning','clipping_warning','near_silent_warning']
    fig,ax=plt.subplots(figsize=(13,5));x=np.arange(len(warning_keys))
    for i,e in enumerate(summary['configurations']):ax.bar(x+(i-1)*.24,[e['automatic_warning_counts'][k]/e['denominator'] for k in warning_keys],width=.24,label=display_config(e['configuration_id']))
    ax.set_xticks(x,[k.replace('_warning','').replace('_',' ') for k in warning_keys],rotation=15,ha='right');ax.set_ylabel('Automatic warning fraction (not confirmed failure rate)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'automatic_warnings.png',dpi=140);plt.close(fig)
    show=[k for k in METRICS if k in corr]+[k for k in acoustic if k.startswith('egemaps.')][:10]
    if show:
        fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(corr.loc[show,show],vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(len(show)),[k.replace('audio.','').replace('egemaps.','eG:') for k in show],rotation=90,fontsize=7)
        ax.set_yticks(range(len(show)),[k.replace('audio.','').replace('egemaps.','eG:') for k in show],fontsize=7)
        fig.colorbar(im,ax=ax,label='Configuration-centered Spearman rho');fig.tight_layout();fig.savefig(figdir/'metric_redundancy.png',dpi=140);plt.close(fig)
    # Four selected inputs, all configurations: preserve denominator, expose tails.
    ranked=sorted(rows,key=lambda r:value(r,'automatic_proxy.wer') if value(r,'automatic_proxy.wer') is not None else -1,reverse=True)
    examples=[]
    for r in ranked:
        if r['utterance_id'] not in examples:examples.append(r['utterance_id'])
        if len(examples)==4:break
    sections=['<h1>Full-baseline automatic evaluation</h1><p>100 inputs x 3 configurations = 300 recordings. Results are provisional/exploratory. Human listening remains pending.</p>',
        '<p>Challenge records are an overlapping diagnostic view, never additional baseline observations. Voice differences confound architecture-only interpretation.</p>',
        '<h2>What the measurements support</h2><p>ASR-proxy errors expose candidates for content problems; they cannot establish what was spoken. Raw/strict/uroman differences show normalization sensitivity, not validated equivalence. Acoustic descriptors describe delivery; greater variation or loudness is not inherently better.</p>']
    headers=['Measure']+[display_config(e['configuration_id']) for e in summary['configurations']]
    sections.append('<table border="1" cellpadding="8"><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>')
    for metric in METRICS:
        vals=[e['metrics'][metric]['median'] for e in summary['configurations']]
        sections.append('<tr><td>'+html.escape(metric)+' (median)</td>'+''.join('<td>'+('pending/missing' if v is None else f'{v:.4f}')+'</td>' for v in vals)+'</tr>')
    sections.append('</table>')
    for p in sorted(figdir.glob('*.png')):
        sections.append('<h2>'+html.escape(p.stem)+'</h2><img style="max-width:100%" src="figures/'+html.escape(p.name)+'">')
    sections.append('<h2>Interpretation and next validation</h2><ul><li>Inspect warnings and extreme cases before attributing errors to synthesis rather than ASR.</li><li>Use paired intervals and influence checks together: influential examples remain in every baseline aggregate.</li><li>Pitch/pause/voice-quality/eGeMAPS features need perceptual validation; correlated features are not independent evidence.</li><li>Language-based difficulty remains pending reliable labels; script plots must not be relabelled as Hindi/English.</li><li>Obtain human transcription and blinded ratings later to calibrate automatic measurements.</li></ul>')
    audio_dir=out/'audio';audio_dir.mkdir(exist_ok=True)
    for uid in examples:
        selected_rows=sorted([r for r in rows if r['utterance_id']==uid],key=lambda r:r['configuration_id'])
        fig,axes=plt.subplots(3,len(selected_rows),figsize=(15,8),squeeze=False)
        sections.append('<h2>Diagnostic source line '+uid+'</h2>')
        for col,r in enumerate(selected_rows):
            path=Path(r['source_audio_path'])
            sections.append('<h3>'+html.escape(display_config(r['configuration_id']))+'</h3><p><b>Intended:</b> '+html.escape(r['text_raw'])+'</p><p><b>ASR:</b> '+html.escape(r['asr_raw'] or '[missing]')+'</p>')
            if path.exists():
                target=audio_dir/(r['generation_id']+'.wav')
                if not target.exists():shutil.copyfile(path,target)
                sections.append('<audio controls preload="none" src="audio/'+target.name+'"></audio>')
                y,sr=sf.read(path)
                axes[0,col].plot(np.arange(len(y))/sr,y,lw=.3);axes[0,col].set_title(display_config(r['configuration_id']))
                if len(y)>=512:axes[1,col].specgram(y,NFFT=512,Fs=sr,noverlap=384,cmap='magma')
                track=r['audio'].get('f0_track',[]);axes[2,col].plot([t['time_s'] for t in track],[t['hz'] if t['hz'] is not None else np.nan for t in track],lw=.8);axes[2,col].set(xlabel='Seconds',ylabel='F0 Hz')
            else:axes[0,col].set_title('Audio unavailable')
        fig.suptitle('Source line '+uid+'; selected for error investigation, not representative prevalence');fig.tight_layout();name='example_'+uid+'.png';fig.savefig(figdir/name,dpi=140);plt.close(fig)
        sections.append('<img style="max-width:100%" src="figures/'+name+'">')
    sections.append('<h2>Audit-ready summaries</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SpeechLab full baseline</title><style>body{font:16px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:15px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'+''.join(sections),encoding='utf-8')
    write_json(out/'challenge_view_summary.json',{'recordings':summary['challenge_view_recordings'],
        'overlaps_baseline':True,'configuration_counts':dict(Counter(r['configuration_id'] for r in rows if r['in_challenge_view'])),
        'configuration_diagnostics':[{ 'configuration_id':config,'n':sum(r['in_challenge_view'] for r in rs),
            'metrics':{k:safe_stats([value(r,k) for r in rs if r['in_challenge_view']]) for k in METRICS},
            'warnings':{k:sum(r['automatic_warnings'].get(k) is True for r in rs if r['in_challenge_view']) for k in rs[0]['automatic_warnings']}}
            for config,rs in sorted(groups.items())],
        'interpretation':'Enriched diagnostic view; never pool as independent extra data'})
    print('Full-baseline report complete: paired uncertainty, normalization sensitivity, warnings and feature redundancy.',flush=True)
