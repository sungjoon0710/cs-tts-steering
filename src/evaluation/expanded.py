"""Versioned CPU-only extensions; these descriptors are not naturalness scores."""
from __future__ import annotations

import importlib.metadata as metadata
import math
import warnings
from collections import Counter
from functools import lru_cache

import numpy as np
import soundfile as sf
from rapidfuzz.distance import Levenshtein
from scipy.signal import resample_poly
from scipy.stats import kurtosis, skew

from .audio import extract_audio
from .text import edit_score, normalize

VERSION = "full-exploratory-v0.2.0"


def number(x):
    return float(x) if x is not None and np.isfinite(x) else None


@lru_cache(maxsize=1)
def romanizer():
    import uroman
    return uroman.Uroman()


def romanize(text):
    return " ".join(romanizer().romanize_string(normalize(text)).casefold().split())


def raw_score(reference, hypothesis):
    import unicodedata
    if hypothesis is None:
        return {"status": "missing_asr", "wer": None, "cer": None}
    ref = unicodedata.normalize("NFC", reference); hyp = unicodedata.normalize("NFC", hypothesis)
    rt, ht = ref.split(), hyp.split(); rc, hc = "".join(rt), "".join(ht)
    return {"status": "provisional_raw_script_sensitive", "wer": Levenshtein.distance(rt, ht)/len(rt) if rt else None,
            "cer": Levenshtein.distance(rc,hc)/len(rc) if rc else None}


def content_proxies(reference, hypothesis, audio):
    strict = edit_score(reference, hypothesis)
    harmonized = edit_score(romanize(reference), romanize(hypothesis)) if hypothesis is not None else edit_score(reference, None)
    harmonized.update(status="exploratory_uroman_script_harmonization", uroman_version=metadata.version("uroman"),
                      warning="Romanization is not validated phonetic or semantic equivalence")
    tokens = normalize(hypothesis).split() if hypothesis is not None else []
    trigrams = [tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    counts = Counter(trigrams)
    repeated = sum(n-1 for n in counts.values() if n>1)
    n = strict.get("reference_words",0)
    d = strict.get("deletions",0)/n if n else None
    ins = strict.get("insertions",0)/n if n else None
    duration = audio.get("duration_s"); articulation = audio.get("articulation_time_estimate_s")
    proxies = {"asr_word_count":len(tokens) if hypothesis is not None else None,
        "asr_repeated_trigram_fraction":(repeated/len(trigrams) if trigrams else 0) if hypothesis is not None else None,
        "asr_deletion_fraction":d,"asr_insertion_fraction":ins,
        "asr_speaking_rate_wps":len(tokens)/duration if duration and hypothesis is not None else None,
        "asr_articulation_rate_proxy_wps":len(tokens)/articulation if articulation and hypothesis is not None else None,
        "rate_status":"ASR-based estimates; segmentation and recognizer unvalidated"}
    return strict, raw_score(reference,hypothesis), harmonized, proxies


@lru_cache(maxsize=1)
def smile_extractor():
    import opensmile
    return opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                          feature_level=opensmile.FeatureLevel.Functionals)


def expanded_audio(path, channel_policy="reject", egemaps=True):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base = extract_audio(path, channel_policy=channel_policy)
    base["runtime_warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    base["expanded_version"] = VERSION
    if base["status"] != "computed_unvalidated":
        return base
    x,sr=sf.read(path,always_2d=True,dtype="float64")
    y=x.mean(axis=1) if channel_policy=="mean" else x[:,0]
    f=np.array([r["hz"] for r in base.get("f0_track",[]) if r["hz"] is not None],float)
    if len(f):
        q=np.quantile(f,[.05,.25,.5,.75,.95])
        base.update(f0_mean_hz=number(f.mean()),f0_std_hz=number(f.std()),
            f0_q05_hz=number(q[0]),f0_q25_hz=number(q[1]),f0_q75_hz=number(q[3]),f0_q95_hz=number(q[4]),
            f0_skew=number(skew(f)) if len(f)>2 and f.std()>1e-9 else None,
            f0_kurtosis=number(kurtosis(f)) if len(f)>3 and f.std()>1e-9 else None)
    import parselmouth
    from parselmouth.praat import call
    sound=parselmouth.Sound(y,sampling_frequency=sr)
    try:
        intensity=sound.to_intensity(minimum_pitch=60,time_step=.01).values[0]
        base.update(intensity_praat_mean_db=number(intensity.mean()),intensity_praat_min_db=number(intensity.min()),
                    intensity_praat_max_db=number(intensity.max()),intensity_praat_iqr_db=number(np.quantile(intensity,.75)-np.quantile(intensity,.25)))
        pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=60,pitch_ceiling=600)
        strength=pitch.selected_array['strength'];freq=pitch.selected_array['frequency']
        base['pitch_voiced_autocorrelation_strength_median']=number(np.median(strength[freq>0])) if np.any(freq>0) else None
        classes=[]
        for t,hz in zip(pitch.xs(),freq):
            silent=any(r['label']=='silent' and r['start_s']<=t<r['end_s'] for r in base.get('intensity_intervals',[]))
            classes.append('intensity_silent' if silent else 'voiced_sounding' if hz>0 else 'unvoiced_sounding')
        base['joint_frame_class_fractions']={k:classes.count(k)/len(classes) for k in ('intensity_silent','voiced_sounding','unvoiced_sounding')} if classes else {}
        base['pitch_frame_coverage_s']=len(classes)*.01
    except Exception as e:base['expanded_pitch_intensity_error']=type(e).__name__
    try:
        # Parameters match /proj/speech/tools/extract_parselmouth_features.py.
        pp=call(sound,'To PointProcess (periodic, cc)',75.,600.)
        jitter=call(pp,'Get jitter (local)',0.,0.,.0001,.02,1.3)
        shimmer=call([sound,pp],'Get shimmer (local)',0.,0.,.0001,.02,1.3,1.6)
        harmonicity=call(sound,'To Harmonicity (cc)',.01,75.,.1,1.)
        hnr=call(harmonicity,'Get mean',0.,0.)
        base['lab_voice_quality']={'status':'exploratory_unvalidated_on_synthetic_speech',
            'jitter_local':number(jitter),'shimmer_local':number(shimmer),'hnr_mean_db':number(hnr),
            'method_source':'SpeechLab extract_parselmouth_features.py; 75-600Hz pulse range'}
    except Exception as e:base['lab_voice_quality']={'status':'extraction_failed','error_type':type(e).__name__}
    if egemaps:
        try:
            divisor=math.gcd(sr,16000)
            z=resample_poly(y,16000//divisor,sr//divisor,window=('kaiser',5.0)).astype('float32') if sr!=16000 else y.astype('float32')
            frame=smile_extractor().process_signal(z,16000)
            base['egemaps']={'status':'exploratory','version':metadata.version('opensmile'),'feature_set':'eGeMAPSv02',
                'feature_level':'Functionals','sample_rate':16000,'resampling':'scipy.resample_poly; Kaiser beta=5',
                'features':{k:number(v) for k,v in frame.iloc[0].items()}}
        except Exception as e:base['egemaps']={'status':'extraction_failed','error_type':type(e).__name__,'message':str(e)[:300]}
    return base
