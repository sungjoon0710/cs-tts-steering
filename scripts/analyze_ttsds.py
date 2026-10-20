"""Inspect official scores and bootstrap cached per-recording rate factors only.

Never bootstrap SSL frames as independent utterances. Full-ensemble confidence
intervals require per-utterance feature provenance not retained by official caches.
"""
from __future__ import annotations
import argparse,importlib.util,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.evaluation.common import digest,private_directory,read_jsonl,write_json

def export(a):
    from ttsds.util.dataset import DirectoryDataset
    from ttsds.util.cache import hash_md5
    inputs=json.loads((a.run/'inputs.json').read_text(encoding='utf-8'))
    by_config=defaultdict(list)
    for r in inputs['generations']:by_config[r['configuration_id']].append(r)
    sources={}
    for config,rs in by_config.items():
        ds=DirectoryDataset(str(Path(rs[0]['source_audio_path']).parent),name=config);ds.wavs=sorted(ds.wavs)
        if {str(p) for p in ds.wavs}!={r['source_audio_path'] for r in rs}:raise ValueError('Changed source inventory')
        sources[config]=ds
    refs=read_jsonl(a.reference_manifest);index={r['reference_id']:r for r in refs}
    natural=DirectoryDataset(str(a.natural),name='MaSaC_selected_100_stereo_mean');natural.wavs=sorted(natural.wavs);natural.sample(100)
    sources[natural.name]=natural;records={}
    for name,ds in sources.items():
        ids=[p.stem for p in ds.wavs]
        if ds.indices is not None:ids=[ids[i] for i in ds.indices]
        records[name]={'ids':ids,'episodes':[index[i]['episode'] for i in ids] if name==natural.name else None,'factors':{}}
        for feature in ['Allosaurus SR','mHubert-147 Token SR']:
            files=list((a.cache/'benchmarks'/feature).glob(hash_md5(ds)+'_*.npy'))
            if len(files)!=1:raise ValueError('Ambiguous/missing scalar cache: '+feature)
            values=np.load(files[0],allow_pickle=False)
            if values.shape!=(len(ids),) or not np.isfinite(values).all():raise ValueError('Scalar cache coverage/values invalid')
            records[name]['factors'][feature]={'values':values.tolist(),'cache_sha256':digest(files[0].read_bytes())}
    out=private_directory(a.output)
    if out.exists():raise ValueError('Use a new cache-export directory')
    write_json(out/'scalar_cache.json',{'datasets':records,'official_inputs_sha256':digest((a.run/'inputs.json').read_bytes()),
        'scope':'Per-recording rate proxies only; original natural order reconstructed from official seed42 sampling'})
    shapes={p.name:list(np.load(p,mmap_mode='r',allow_pickle=False).shape) for p in (a.cache/'benchmarks/mWhisper Activations').glob('*.npy')}
    write_json(out/'implementation_audit.json',{'multilingual_whisper_cache_shapes':shapes,
        'pooling_concern':'Official x[-1].mean(dim=1) on (batch,time,feature) states retains time positions, producing 1500 rather than 768 feature dimensions; preserve official results, review before interpretation.',
        'other_limits':['MPM caps pitch/VAD at 1000 frames (~11.6s)','DirectoryDataset peak-normalizes and pads below16000 samples','Scalar unequal-size helper does one effective subsample, not ten fresh draws','SSL/frame-pooled distributions do not retain utterance boundaries for valid cluster bootstrap']})
    write_json(out/'completion.json',{'status':'complete','scalar_factors':2,'populations':len(records),'model_inference_runs':0})

def report(a):
    import pandas as pd
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.report import display_config
    out=private_directory(a.output);out.mkdir(parents=True,exist_ok=True)
    if json.loads((a.run/'completion.json').read_text(encoding='utf-8'))['status']!='complete':raise ValueError('Official attempt is not complete')
    scores=pd.read_csv(a.run/'feature_scores.csv')
    if len(scores)!=33 or scores['benchmark_name'].nunique()!=11 or scores['dataset'].nunique()!=3 or not scores.groupby('dataset').size().eq(11).all() or scores.duplicated(['benchmark_name','dataset']).any() or not np.isfinite(scores['score']).all():raise ValueError('Expected 11 complete features x 3 configurations')
    pivot=scores.pivot(index='benchmark_name',columns='dataset',values='score').sort_index()
    pivot.to_csv(out/'feature_scores.csv');factors=scores.groupby(['dataset','benchmark_category'])['score'].mean().unstack()
    factors['equal_factor_mean']=factors.mean(axis=1);factors.to_csv(out/'factor_scores.csv')
    official=pd.read_csv(a.run/'aggregate_scores.csv');overall=official[official['benchmark_category']=='OVERALL'].set_index('dataset')['score_mean']
    error=float(np.max(np.abs(factors['equal_factor_mean']-overall)))
    if error>1e-8:raise ValueError('Official aggregation could not be reproduced')
    write_json(out/'aggregation_validation.json',{'max_absolute_difference':error,'features':11,'configurations':3,
        'official_score_std_note':'Within-category feature dispersion, not sampling uncertainty; official overall std=0 is a placeholder, not a confidence interval'})
    fig,axes=plt.subplots(1,2,figsize=(16,7),gridspec_kw={'width_ratios':[2,1]})
    im=axes[0].imshow(pivot,vmin=0,vmax=100,cmap='viridis');axes[0].set_yticks(range(len(pivot)),pivot.index,fontsize=9)
    axes[0].set_xticks(range(3),[display_config(c) for c in pivot.columns],rotation=30,ha='right',fontsize=8)
    for i,row in enumerate(pivot.values):
        for j,v in enumerate(row):axes[0].text(j,i,f'{v:.1f}',ha='center',va='center',color='black' if v>65 else 'white',fontsize=9)
    factors.drop(columns='equal_factor_mean').T.plot.bar(ax=axes[1]);axes[1].set(ylabel='Official normalized score',ylim=(0,100));axes[1].legend([display_config(c) for c in factors.index],fontsize=7)
    fig.colorbar(im,ax=axes[0]);fig.suptitle('Official multilingual TTSDS2: reference-conditioned factors, not human quality rankings');fig.tight_layout();fig.savefig(out/'official_factors.png',dpi=150);plt.close(fig)
    source=Path(a.official_distances);spec=importlib.util.spec_from_file_location('pinned_ttsds_distances',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=json.loads((a.scalar/'scalar_cache.json').read_text(encoding='utf-8'))['datasets'];ref=data.pop('MaSaC_selected_100_stereo_mean')
    # Reference-only W2 sensitivity, distinct from noise-normalized full TTSDS2.
    rng=np.random.default_rng(1234);episodes=np.array(ref['episodes']);unique=np.unique(episodes);results=[]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,feature in zip(axes,ref['factors']):
        natural=np.array(ref['factors'][feature]['values']);ax.step(np.sort(natural),np.arange(1,101)/100,label='Natural MaSaC',color='black')
        for config,d in sorted(data.items()):
            x=np.array(d['factors'][feature]['values']);w2=float(module.wasserstein_distance(x,natural));boot=[]
            for _ in range(200):
                idx=np.concatenate([np.flatnonzero(episodes==ep) for ep in rng.choice(unique,len(unique),replace=True)])
                boot.append(float(module.wasserstein_distance(x[rng.integers(0,len(x),len(x))],natural[idx])))
            stability=[]
            for n in [25,50,100]:
                values=[float(module.wasserstein_distance(x[rng.choice(len(x),n,replace=False)],natural[rng.choice(len(natural),n,replace=False)])) for _ in range(20)]
                stability.append({'n_per_population':n,'median_W2':float(np.median(values)),'range_W2':[float(min(values)),float(max(values))]})
            results.append({'feature':feature,'configuration_id':config,'reference_W2':w2,'bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
                'resampling':'200 replicates: natural episodes, generated utterances; official unequal-size W2 helper; no full-ensemble CI','sample_size_stability':stability})
            ax.step(np.sort(x),np.arange(1,101)/100,label=display_config(config))
        ax.set(title=feature,xlabel='Phone/token-event rate proxy',ylabel='Cumulative fraction');ax.legend(fontsize=7)
    fig.suptitle('Cached per-recording factors: natural-reference distributions and sensitivity');fig.tight_layout();fig.savefig(out/'rate_distributions.png',dpi=150);plt.close(fig)
    write_json(out/'scalar_sensitivity.json',{'results':results,'official_distance_source_sha256':digest(source.read_bytes()),
        'limitations':'Natural TV/speaker/channel/language differences remain. Rate proxies are not articulation rate. Full SSL-frame bootstrap invalid without per-utterance provenance.'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['export','report']);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--natural',type=Path);p.add_argument('--reference-manifest',type=Path);p.add_argument('--scalar',type=Path);p.add_argument('--official-distances',type=Path)
    a=p.parse_args();export(a) if a.mode=='export' else report(a)
