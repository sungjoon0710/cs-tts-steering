"""Reconcile upstream formulas with an immutable private baseline snapshot.

Text-only CPU calculation; no downloads, shared writes, ASR or TTS inference.
"""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,write_json,write_jsonl
from src.evaluation.legacy import legacy_content

def audit(source,output):
    output=private_directory(output)
    source=Path(source);rows=list(csv.DictReader((source/'eval/metrics.csv').open(encoding='utf-8')))
    groups=defaultdict(list);results=[]
    for r in rows:
        measured=legacy_content(r['text'],r['hyp'])
        config=r['model']+'/'+r['voice'];groups[config].append(measured)
        duration=float(r['audio_s']);expected=max(1.,len(r['text'])/12);ratio=duration/expected
        hit=r['hit_max'].lower()=='true'
        result={'utterance_id':r['line'],'configuration_id':config,'reproduced':measured,
            'wer_abs_difference':abs(measured['wer']-float(r['wer'])) if measured['wer'] is not None else None,
            'cer_abs_difference':abs(measured['cer']-float(r['cer'])) if measured['cer'] is not None else None,
            'latin_counts_match':(measured['latin_found'],measured['latin_total'])==(int(r['eng_found']),int(r['eng_total'])),
            'runaway_rule_matches':(hit or ratio>2.5)==(r['runaway'].lower()=='true'),
            'short_rule_matches':(ratio<.4)==(r['too_short'].lower()=='true')}
        results.append(result)
    import numpy as np
    summary={'recordings':len(results),'metric_input_sha256':digest((source/'eval/metrics.csv').read_bytes()),
        'upstream_evaluator_sha256':digest((Path(__file__).parent/'evaluate.py').read_bytes()),
        'text_reproduction_max_abs_difference':{k:max(r[k] for r in results if r[k] is not None) for k in ('wer_abs_difference','cer_abs_difference')},
        'latin_count_mismatches':sum(not r['latin_counts_match'] for r in results),
        'runaway_rule_mismatches':sum(not r['runaway_rule_matches'] for r in results),
        'short_rule_mismatches':sum(not r['short_rule_matches'] for r in results),
        'source_declared_asr':'openai/whisper-large-v3; float16; language=hi; task=transcribe; batch16; 16k torchaudio resample',
        'source_declared_speaker_model':'microsoft/wavlm-base-plus-sv; normalized embeddings; mean pairwise cosine within voice',
        'provenance_limit':'Source declarations do not recover original checkpoint revisions, package versions or cache-to-WAV hashes; no models run',
        'english_survival_limit':'Counts Latin regex matches length>=2, including Romanized Hindi; fuzzy threshold .5; reuses hypothesis tokens and ignores order; not PIER or verified English preservation',
        'configurations':[]}
    for config,ms in sorted(groups.items()):
        found=sum(m['latin_found'] for m in ms);total=sum(m['latin_total'] for m in ms)
        summary['configurations'].append({'configuration_id':config,'n':len(ms),
            'legacy_wer_median':float(np.median([m['wer'] for m in ms])),
            'legacy_cer_median':float(np.median([m['cer'] for m in ms])),
            'latin_found':found,'latin_total':total,'latin_fuzzy_survival':found/total if total else None})
    write_jsonl(output/'reconciliation.jsonl',results);write_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='configurations'},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.source,a.output)
