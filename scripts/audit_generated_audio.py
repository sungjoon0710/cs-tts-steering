"""CPU-only header/hash audit, versioned rate correction and acoustic spot checks."""
import argparse
import importlib.metadata as metadata
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl
from src.evaluation.expanded import expanded_audio
from scripts.lab_rate_evaluate import main as rates


def scalars(value,prefix=''):
    if isinstance(value,dict):
        for k,v in value.items():yield from scalars(v,prefix+'.'+k if prefix else k)
    elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
        yield prefix,float(value)


def main(a):
    out=private_directory(a.output)
    if out.exists():raise ValueError('Use a new audit directory')
    out.mkdir(parents=True);rows=read_jsonl(a.baseline_run/'tables/metrics.jsonl')
    if len(rows)!=300 or len({r['generation_id'] for r in rows})!=300:raise ValueError('Expected300 unique baseline records')
    headers=[]
    for r in rows:
        h={k:r[k] for k in ('generation_id','utterance_id','configuration_id')}
        try:
            p=Path(r['source_audio_path']);s=sf.info(p);sha=digest(p.read_bytes())
            h.update(audio_sha256=sha,hash_matches=sha==r['audio']['audio_sha256'],sample_rate=s.samplerate,
                     channels=s.channels,frames=s.frames,subtype=s.subtype,
                     status='ok' if sha==r['audio']['audio_sha256'] and s.samplerate==24000 and s.channels==1 and s.subtype=='PCM_16' and s.frames>0 else 'mismatch')
        except Exception as e:h.update(status='failed',error_type=type(e).__name__)
        headers.append(h)
    write_jsonl(out/'audio_hash_headers.jsonl',headers)
    if any(r['status']!='ok' for r in headers):raise ValueError('Audio association or format failed; do not reuse cached measurements')
    checks=[]
    for r in rows:
        if r['utterance_id'] not in {'0','29','43','72'}:continue
        current=expanded_audio(r['source_audio_path']);old=dict(scalars(r['audio']));new=dict(scalars(current))
        differences=[]
        for k,v in old.items():
            if k not in new or abs(new[k]-v)>1e-7+1e-5*abs(v):
                differences.append({'field':k,'old':v,'new':new.get(k)})
        checks.append({'generation_id':r['generation_id'],'utterance_id':r['utterance_id'],
            'configuration_id':r['configuration_id'],'compared_numeric_fields':len(old),
            'egemaps_fields':len(current.get('egemaps',{}).get('features',{})),
            'status':'match' if not differences else 'review_required','differences':differences})
    write_jsonl(out/'acoustic_spot_checks.jsonl',checks)
    rates(argparse.Namespace(output=out/'rate_terminal_v2',baseline_run=a.baseline_run,
        lab_script=a.lab_script,detector='terminal_v2',comparison=None))
    write_json(out/'completion.json',{'audio_records':len(headers),'audio_hash_format_failures':sum(r['status']!='ok' for r in headers),
        'acoustic_spot_checks':len(checks),'spot_check_mismatches':sum(r['status']!='match' for r in checks),
        'spot_check_input_ids':['0','29','43','72'],'spot_check_scope':'12 clips, not full feature re-extraction or perceptual validation',
        'packages':{p:metadata.version(p) for p in ['numpy','scipy','soundfile','praat-parselmouth','opensmile']},
        'source_sha256':digest(Path(__file__).read_bytes()),'new_asr_or_tts_inference':False,'gpu_used':False,
        'original_audio_evaluated':False,'unmatched_natural_rankings_reused':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--lab-script',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
