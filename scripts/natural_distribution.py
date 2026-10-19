"""Unpaired descriptive natural/synthetic comparisons. This is NOT TTSDS.

References are selected by verified WITS filename/text/speaker joins. Original
stereo WAVs are read only; channel averaging happens in memory.
"""
from __future__ import annotations
import argparse,csv,json,random,re,sys,time,unicodedata,importlib.metadata
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import wasserstein_distance
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl
from src.evaluation.expanded import expanded_audio

def canonical(text):return ' '.join(unicodedata.normalize('NFC',text).casefold().split())

def select_references(data,count=100,seed=1234):
    data=Path(data);wits={};hashes={}
    for split in ('train','val','test'):
        path=data/(split+'_data_final_plain.json');hashes[path.name]=digest(path.read_bytes())
        for key,record in json.loads(path.read_text(encoding='utf-8')).items():
            texts=record['context_utterances']+[record['target_utterance']]
            speakers=record['context_speakers']+[record['target_speaker']]
            if len(texts)!=len(speakers):raise ValueError('WITS text/speaker mismatch')
            for i,(text,speaker) in enumerate(zip(texts,speakers)):
                wits[f'{split}_{key}_{i}.wav']={'text':text,'speaker':speaker,'conversation':split+'_'+key,
                    'split':split,'episode':record['episode_name'],'is_target':i==len(texts)-1}
    annotations=data/'fixed_with_automated_annotations.csv';hashes[annotations.name]=digest(annotations.read_bytes())
    candidates=[]
    with annotations.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            record=wits.get(r['name']);path=data/'wavfiles_utt_level'/r['name']
            if record and path.is_file() and canonical(record['text'])==canonical(r['Transcript']) and canonical(record['speaker'])==canonical(r['Speaker']):
                candidates.append({'reference_id':r['name'][:-4],'audio_path':str(path),'speaker':record['speaker'],
                    'conversation':record['conversation'],'episode':record['episode'],'split':record['split'],
                    'is_target':record['is_target'],'text_join':'NFC/casefold/whitespace exact; no transliteration',
                    'speaker_join':'NFC/casefold exact','language_labels':'automated, not validated; unused for selection'})
    # Avoid a population dominated by rare/composite speaker labels. Keep the
    # ten most frequent single-name source labels, without merging aliases.
    counts=Counter(r['speaker'] for r in candidates)
    eligible=[s for s,n in counts.most_common() if not re.search(r'\bAND\b',s.upper()) and s.upper() not in ('EVERYONE','CREW','BACHCHHE')][:10]
    candidates=[r for r in candidates if r['speaker'] in eligible]
    rng=random.Random(seed);rng.shuffle(candidates)
    # Round-robin speakers, one recording per conversation, no repeated bytes.
    buckets=defaultdict(list)
    for r in candidates:buckets[r['speaker']].append(r)
    speakers=sorted(buckets);selected=[];conversations=set();hashset=set()
    while len(selected)<count and any(buckets.values()):
        for speaker in speakers:
            while buckets[speaker]:
                r=buckets[speaker].pop()
                if r['conversation'] in conversations:continue
                sha=digest(Path(r['audio_path']).read_bytes())
                if sha in hashset:continue
                r['audio_sha256']=sha;selected.append(r);conversations.add(r['conversation']);hashset.add(sha);break
            if len(selected)==count:break
    if len(selected)!=count:raise ValueError('Insufficient independently identified references')
    return selected,{'source_hashes':hashes,'candidate_rows':len(candidates),'seed':seed,
        'selection':'top ten frequent single-name source speaker labels, round-robin; one WAV per WITS conversation; exact text+speaker; distinct audio SHA256',
        'speaker_aliases':'Source labels retained; aliases not merged; labels are not verified speaker diarization',
        'speaker_counts':dict(Counter(r['speaker'] for r in selected)),
        'split_counts':dict(Counter(r['split'] for r in selected)),
        'episodes':len({r['episode'] for r in selected}),'matched_to_baseline':False,
        'limitations':'Acted TV speech; shared episodes/characters; language mix not matched; channel/domain confounds'}

def extract(data,output):
    output=private_directory(output);output.mkdir(parents=True,exist_ok=True)
    refs,selection=select_references(data)
    if (output/'manifest.jsonl').exists() and read_jsonl(output/'manifest.jsonl')!=refs:raise ValueError('Reference selection changed')
    write_jsonl(output/'manifest.jsonl',refs);write_json(output/'selection.json',selection)
    code_root=Path(__file__).resolve().parents[1]
    extractor_hashes={name:digest((code_root/'src/evaluation'/name).read_bytes()) for name in ('audio.py','expanded.py')}
    packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','soundfile','praat-parselmouth','opensmile','pandas','audinterface','audformat','audresample')}
    def cache_path(record,policy):
        signature=digest(json.dumps({'audio_sha256':record['audio_sha256'],'policy':policy,
            'extractor_hashes':extractor_hashes,'packages':packages},sort_keys=True).encode())
        return output/'feature_cache'/(record['reference_id']+'_'+policy+'_'+signature+'.json')
    feature_rows=[];sensitivity=[];subset=set(random.Random(1234).sample([r['reference_id'] for r in refs],20))
    for i,r in enumerate(refs):
        cache=cache_path(r,'mean')
        if cache.exists():features=json.loads(cache.read_text(encoding='utf-8'))
        else:features=expanded_audio(r['audio_path'],channel_policy='mean');write_json(cache,features)
        if features.get('audio_sha256')!=r['audio_sha256']:raise ValueError('Reference WAV changed')
        feature_rows.append({**r,'audio':features})
        if r['reference_id'] in subset:
            cache=cache_path(r,'first')
            if cache.exists():first=json.loads(cache.read_text(encoding='utf-8'))
            else:first=expanded_audio(r['audio_path'],channel_policy='first');write_json(cache,first)
            if first.get('audio_sha256')!=r['audio_sha256']:raise ValueError('Reference WAV changed')
            sensitivity.append({'reference_id':r['reference_id'],'mean':features,'first':first})
        if (i+1)%10==0:print(f'Natural reference CPU extraction {i+1}/100',flush=True)
    write_jsonl(output/'features.jsonl',feature_rows);write_jsonl(output/'channel_sensitivity.jsonl',sensitivity)
    write_json(output/'provenance.json',{'extractor_hashes':extractor_hashes,'packages':packages,
        'selection_script_sha256':digest(Path(__file__).read_bytes()),'channel_policies':['mean','first'],
        'device':'CPU','egemaps':True,'original_audio_modified':False})
    write_json(output/'completion.json',{'references':len(feature_rows),'first_channel_subset':len(sensitivity),
        'primary_channel_policy':'arithmetic average in memory','source_audio_modified':False,
        'supervisor_review':'Discuss channel choice and sensitivity; alternatives are first/cleaner channel or channel-specific reporting'})

def compare(features,baseline,output):
    from src.evaluation.full_report import METRICS,flatten
    from src.evaluation.report import display_config
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not (Path(baseline)/'completion.json').is_file():raise ValueError('Baseline run must be complete before comparison')
    refs=read_jsonl(Path(features)/'features.jsonl');rows=read_jsonl(Path(baseline)/'tables/metrics.jsonl')
    if not refs or not rows:raise ValueError('Reference and synthetic populations must be nonempty')
    output=private_directory(output);output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for r in rows:groups[r['configuration_id']].append(r)
    metrics=[m for m in METRICS if m.startswith('audio.')]
    metrics += ['audio.egemaps.features.'+k for k in refs[0]['audio'].get('egemaps',{}).get('features',{})]
    def get(row,m):return flatten(row).get(m)
    rng=np.random.default_rng(1234);results=[];curves=[]
    for metric in metrics:
        rv=np.array([v for r in refs if (v:=get(r,metric)) is not None])
        if len(rv)<20:continue
        scale=float(np.quantile(rv,.75)-np.quantile(rv,.25))
        for config,items in sorted(groups.items()):
            sv=np.array([v for r in items if (v:=get(r,metric)) is not None])
            if len(sv)<20:continue
            distance=float(wasserstein_distance(rv,sv))
            episodes=sorted({r['episode'] for r in refs})
            episode_values={e:[v for r in refs if r['episode']==e and (v:=get(r,metric)) is not None] for e in episodes}
            bootstrap=[]
            for _ in range(200):
                natural_draw=[v for e in rng.choice(episodes,len(episodes),replace=True) for v in episode_values[e]]
                bootstrap.append(wasserstein_distance(natural_draw,rng.choice(sv,len(sv),replace=True)))
            split_dist={}
            for split in ('train','val','test'):
                vs=[v for r in refs if r['split']==split and (v:=get(r,metric)) is not None]
                split_dist[split]={'n':len(vs),'w1':float(wasserstein_distance(vs,sv)) if len(vs)>=5 else None}
            # Duration-nearest reference without replacement: domain sensitivity only.
            available=[r for r in refs if get(r,metric) is not None];matched=[]
            matched_items=sorted(items,key=lambda r:r['audio'].get('duration_s',0))[::2]
            for r in matched_items:
                if not available:break
                closest=min(available,key=lambda a:abs(a['audio']['duration_s']-r['audio']['duration_s']))
                available.remove(closest);matched.append(get(closest,metric))
            msv=[get(r,metric) for r in matched_items if get(r,metric) is not None]
            results.append({'metric':metric,'configuration_id':config,'natural_n':len(rv),'synthetic_n':len(sv),'w1':distance,
                'reference_iqr':scale,'w1_reference_iqr_scaled':distance/scale if scale>0 else None,
                'bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist(),
                'reference_split_sensitivity':split_dist,'duration_matched_50_sensitivity_w1':float(wasserstein_distance(matched,msv)) if matched and msv else None})
            if metric in metrics[:10]:
                for n in (20,40,60,80,100):
                    if min(len(rv),len(sv))<n:continue
                    values=[wasserstein_distance(rng.choice(rv,n,replace=False),rng.choice(sv,n,replace=False)) for _ in range(100)]
                    curves.append({'metric':metric,'configuration_id':config,'n_per_population':n,'mean_w1':float(np.mean(values)),
                        'subset_95range':np.quantile(values,[.025,.975]).tolist()})
    write_jsonl(output/'distances.jsonl',results);write_jsonl(output/'sample_size_curves.jsonl',curves)
    sensitivity=[]
    for metric in metrics:
        diffs=[]
        for r in read_jsonl(Path(features)/'channel_sensitivity.jsonl'):
            a=get({'audio':r['mean']},metric);b=get({'audio':r['first']},metric)
            if a is not None and b is not None:diffs.append(b-a)
        sensitivity.append({'metric':metric,'n':len(diffs),'first_minus_mean_median':float(np.median(diffs)) if diffs else None,
            'median_absolute_difference':float(np.median(np.abs(diffs))) if diffs else None})
    write_jsonl(output/'channel_sensitivity_summary.jsonl',sensitivity)
    show=metrics[:9];fig,axes=plt.subplots(3,3,figsize=(16,11))
    for ax,m in zip(axes.flat,show):
        for label,pop in [('Natural reference',refs),*sorted(groups.items())]:
            vals=sorted(v for r in pop if (v:=get(r,m)) is not None)
            if vals:ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=display_config(label) if label!='Natural reference' else label)
        ax.set(xlabel=m,ylabel='Cumulative fraction')
    axes.flat[0].legend(fontsize=6);fig.suptitle('Unpaired, domain-confounded natural and synthetic descriptors; NOT quality ranking');fig.tight_layout();fig.savefig(output/'natural_vs_generated.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,m in zip(axes.flat,show[:6]):
        for config in sorted(groups):
            rs=[r for r in curves if r['metric']==m and r['configuration_id']==config]
            ax.plot([r['n_per_population'] for r in rs],[r['mean_w1'] for r in rs],label=display_config(config))
        ax.set(xlabel='Recordings per population',ylabel='W1',title=m)
    axes.flat[0].legend(fontsize=6);fig.tight_layout();fig.savefig(output/'sample_size_stability.png',dpi=140);plt.close(fig)
    top=[r for r in results if r['metric'] in show]
    write_json(output/'interpretation.json',{'method':'Univariate empirical W1; NOT TTSDS/TTSDS2; no overall score',
        'bootstrap':'200 replicates: natural episode clusters, synthetic utterances; synthetic episode mapping unavailable; exploratory',
        'stability':'100 subsamples without replacement per n; subset ranges are not confidence intervals',
        'reference_selection':json.loads((Path(features)/'selection.json').read_text()),
        'limitations':['Voices/speakers, language content, acted delivery, recording domain and stereo/mono confounded',
            'Natural pitch can include background or overlapping speakers','Smaller distance does not establish better naturalness',
            'IQR scaling is descriptor-specific and is not TTSDS normalization','Natural and synthetic features extracted on different platforms; parity checked separately'],
        'core_distances':top,'supervisor_reminder':'Review stereo averaging vs first-channel results and alternatives before scientific interpretation'})
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Exploratory natural/synthetic distributions</h1><p>Not TTSDS; not naturalness rankings. Reference/domain/speaker confounds apply.</p><img width="100%" src="natural_vs_generated.png"><img width="100%" src="sample_size_stability.png"><p>See interpretation.json, distances.jsonl and channel_sensitivity_summary.jsonl for definitions and sensitivity results.</p>',encoding='utf-8')
    print(f'Distribution comparisons complete: {len(results)} feature/configuration distances',flush=True)
    write_json(output/'completion.json',{'status':'complete','feature_configuration_distances':len(results),
        'method':'Univariate W1, NOT TTSDS','source_features_sha256':digest((Path(features)/'features.jsonl').read_bytes()),
        'source_synthetic_metrics_sha256':digest((Path(baseline)/'tables/metrics.jsonl').read_bytes()),
        'script_sha256':digest(Path(__file__).read_bytes())})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['extract','compare']);p.add_argument('--data',type=Path);p.add_argument('--features',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='extract':extract(a.data,a.output)
    else:compare(a.features,a.baseline,a.output)
