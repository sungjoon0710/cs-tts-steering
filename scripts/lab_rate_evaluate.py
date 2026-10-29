"""Reuse the lab's pinned syllable-nuclei calculation, entirely in memory.

Legacy detector outputs are unvalidated syllable/rate proxies. They do not supply
manual spoken-token counts or validated articulation boundaries. No Praat source
is bundled: provide the existing lab research script through --lab-script.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf
import parselmouth
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl

SOURCE_SHA='134beb4078fb4576f63dfdef3653c048956da86d6dedda8a084023ef3f2aadad'

def detector_body(source,version='legacy'):
    if digest(source)!=SOURCE_SHA:raise ValueError('Lab method changed; review source before execution')
    text=source.decode('utf-8');body=text[text.index('# use object ID'):].rstrip()
    if not body.endswith('endfor'):raise ValueError('Unexpected lab script structure')
    # Remove only directory iteration/I/O; keep its numerical calculation intact.
    body='silencedb = -25\nmindip = 2\nshowtext = 0\nminpause = 0.3\n'+body[:-len('endfor')]
    if version=='legacy':return body
    if version!='terminal_v2':raise ValueError('Unknown detector version')
    # Apply the existing right-dip test to the terminal candidate using the
    # remaining intensity interval. Do not add an unconditional extra nucleus.
    start=body.index('   validpeakcount = 0\n')
    stop=body.index('   # Look for only voiced parts',start)
    replacement='''   validpeakcount = 0
   if peakcount > 0
      currenttime = timepeaks1
      currentint = int1
      for p to peakcount
         if p < peakcount
            following = p + 1
            followingtime = timepeaks'following'
         else
            followingtime = end
         endif
         if followingtime > currenttime
            dip = Get minimum... 'currenttime' 'followingtime' None
            diffint = abs(currentint - dip)
            if diffint > mindip
               validpeakcount += 1
               validtime'validpeakcount' = timepeaks'p'
            endif
         endif
         if p < peakcount
            currenttime = timepeaks'following'
            currentint = Get value at time... timepeaks'following' Cubic
         endif
      endfor
   endif

'''
    return body[:start]+replacement+body[stop:]

def measure(y,sr,body,version='legacy'):
    if y.ndim==2:y=y.mean(axis=1)
    _,_,v=parselmouth.praat.run(parselmouth.Sound(y,sr),body,capture_output=True,return_variables=True)
    count=int(v['voicedcount']);sounding=float(v['speakingtot']);duration=float(v['originaldur'])
    valid=count>0 and sounding>0 and all(math.isfinite(float(v[k])) for k in ['speakingrate','articulationrate'])
    result={'status':f'computed_{version}_detector_proxy' if valid else 'no_valid_detected_nuclei',
        'detector_version':version,
        'detected_nuclei':count,'duration_s':duration,'detected_sounding_s':sounding,
        'nuclei_per_s':float(v['speakingrate']) if valid else None,
        'nuclei_per_detected_sounding_s':float(v['articulationrate']) if valid else None,
        'candidate_intensity_peaks':int(v['peakcount']),'dip_selected_peaks':int(v['validpeakcount']),
        'limitations':('Legacy adjacent-peak rule omits the terminal candidate. ' if version=='legacy' else
            'Terminal boundary uses the same right-dip, voiced and sounding rules; not calibrated on manual syllable counts. ')+
            'Nuclei and sounding intervals are not verified syllables/articulation.'}
    if version=='legacy':result['legacy_dip_selected_peaks']=result['dip_selected_peaks']
    return result

def main(a):
    out=private_directory(a.output)
    if out.exists():raise ValueError('Use a new output directory')
    out.mkdir(parents=True);start=time.time();body=detector_body(a.lab_script.read_bytes(),a.detector);rows=read_jsonl(a.baseline_run/'tables/metrics.jsonl');result=[]
    for row in rows:
        r={k:row[k] for k in ['generation_id','utterance_id','configuration_id','in_challenge_view']}
        try:
            path=Path(row['source_audio_path'])
            if digest(path.read_bytes())!=row['audio']['audio_sha256']:raise ValueError('Changed baseline audio')
            y,sr=sf.read(path,dtype='float64');r.update(audio_sha256=row['audio']['audio_sha256'],**measure(y,sr,body,a.detector))
        except Exception as e:r.update(status='extraction_failed',nuclei_per_s=None,nuclei_per_detected_sounding_s=None,error_type=type(e).__name__,error=str(e)[:300])
        result.append(r)
    write_jsonl(out/'lab_syllable_rates.jsonl',result)
    comparison=[]
    for r in read_jsonl(a.comparison) if a.comparison else []:
        try:
            y,sr=sf.read(r['audio_path'],dtype='float64');m=measure(y,sr,body,a.detector)
            comparison.append({**r,**m,'difference_from_existing_csv':m['nuclei_per_s']-r['existing_natural_rate'] if m['nuclei_per_s'] is not None else None})
        except Exception as e:comparison.append({**r,'status':'extraction_failed','error_type':type(e).__name__})
    write_jsonl(out/'natural_rate_method_comparison.jsonl',comparison)
    write_json(out/'completion.json',{'status':f'complete_{a.detector}_proxy','recordings':len(result),'successful':sum(r['status'].startswith('computed_') for r in result),
        'failures':sum(r['status']=='extraction_failed' for r in result),'runtime_s':time.time()-start,'source_sha256':SOURCE_SHA,
        'script_sha256':digest(Path(__file__).read_bytes()),'adapted_body_sha256':digest(body.encode()),'settings':{'silence_db':-25,'dip_db':2,'min_pause_s':.3,'pitch_floor':30,'pitch_ceiling':450},
        'detector_version':a.detector,'interpretation':'ASR-independent automatic nuclei proxy only; no validated articulation-rate claim'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--lab-script',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--comparison',type=Path)
    p.add_argument('--detector',choices=['legacy','terminal_v2'],default='legacy')
    main(p.parse_args())
