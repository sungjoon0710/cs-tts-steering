"""Strict source associations and paired resampling of source clusters."""
from __future__ import annotations
import builtins,io,pickle,re
from pathlib import Path
import numpy as np
from .common import digest


def restricted_dataframe(path):
    import pandas as pd
    from pandas.core.internals.managers import BlockManager
    from pandas.core.indexes.base import _new_Index,Index
    from pandas.core.indexes.range import RangeIndex
    from numpy._core.multiarray import _reconstruct
    allowed={('pandas.core.frame','DataFrame'):pd.DataFrame,
        ('pandas.core.internals.managers','BlockManager'):BlockManager,
        ('pandas.core.indexes.base','_new_Index'):_new_Index,('pandas.core.indexes.base','Index'):Index,
        ('pandas.core.indexes.range','RangeIndex'):RangeIndex,
        ('numpy.core.multiarray','_reconstruct'):_reconstruct,('numpy','ndarray'):np.ndarray,
        ('numpy','dtype'):np.dtype,('builtins','slice'):builtins.slice}
    class Restricted(pickle.Unpickler):
        def find_class(self,module,name):
            if (module,name) not in allowed:raise pickle.UnpicklingError('Unapproved global: '+module+'.'+name)
            return allowed[module,name]
    data=Path(path).read_bytes()
    if len(data)>9_000_000:raise ValueError('Source table exceeds reviewed size bound')
    frame=Restricted(io.BytesIO(data)).load()
    if not isinstance(frame,pd.DataFrame):raise ValueError('Expected the reviewed source DataFrame')
    return frame


def recover_sources(baseline,frame,raw_lines,audio_lines,episode_lines):
    # Positional association is accepted only after independent text/speaker and
    # both sidecars agree with the same published DataFrame row.
    if not audio_lines or audio_lines[0]!='Audio_Filename' or not episode_lines or episode_lines[0]!='Episode_label':
        raise ValueError('Unexpected sidecar headers')
    audio_lines=audio_lines[1:];episode_lines=episode_lines[1:]
    if len({len(frame),len(raw_lines),len(audio_lines),len(episode_lines)})!=1:raise ValueError('Source lengths differ')
    if len({int(r['line']) for r in baseline})!=len(baseline):raise ValueError('Duplicate baseline line')
    result=[]
    for r in baseline:
        i=int(r['line']);row=frame.iloc[i];raw=raw_lines[i]
        speaker,separator,text=raw.partition(':')
        if not separator:speaker,_,text=raw.strip().partition(' ')
        text=' '.join(text.split());speaker=speaker.strip()
        if (speaker,text)!=(r['speaker'],r['text']):raise ValueError('Raw source and baseline disagree')
        source_text=row['text'];source_speaker=row['Speaker']
        if isinstance(source_text,str) and ' '.join(source_text.split())==text and str(source_speaker).strip()==speaker:
            association='exact_fields' if source_text==text else 'whitespace_normalization_only'
        elif source_text is None and ' '.join(str(source_speaker).split())==' '.join((speaker+' '+text).split()):
            association='malformed_source_fields_recovered_from_exact_raw_line'
        else:raise ValueError('Published source fields and baseline disagree')
        name=str(row['Audio_Filename']);episode_dialogue=str(row['Episode_label'])
        if name!=audio_lines[i] or episode_dialogue!=episode_lines[i]:raise ValueError('Source sidecar mismatch')
        if '/' in name or '\\' in name or name in ('.','..'):raise ValueError('Expected a source basename')
        m=re.fullmatch(r'(E\d+)-(D\d+)',episode_dialogue)
        if not m:raise ValueError('Unrecognized episode/dialogue identifier')
        result.append({'utterance_id':str(i),'source_row':i,'source_speaker':speaker,
            'source_episode':m.group(1),'source_dialogue':episode_dialogue,'source_audio_filename':name,
            'intended_text_sha256':digest(text.encode()),'source_association':association,
            'audio_content_alignment':'not verified by metadata alone'})
    return result


def paired_cluster_intervals(differences,clusters,repeats=2000,seed=1234):
    """One row per matched input; missing metric cells remain missing."""
    d=np.asarray(differences,float)
    if d.ndim==1:d=d[:,None]
    if len(d)!=len(clusters) or not len(d) or repeats<1:raise ValueError('Invalid paired input')
    labels,inverse=np.unique(np.asarray(clusters,str),return_inverse=True);g=len(labels)
    if g<2:raise ValueError('At least two source clusters required')
    finite=np.isfinite(d);sums=np.zeros((g,d.shape[1]));counts=np.zeros_like(sums)
    np.add.at(sums,inverse,np.where(finite,d,0));np.add.at(counts,inverse,finite.astype(float))
    rng=np.random.default_rng(seed);weights=rng.multinomial(g,np.repeat(1/g,g),size=repeats)
    denominator=weights@counts
    draws=np.divide(weights@sums,denominator,out=np.full_like(denominator,np.nan),where=denominator>0)
    results=[]
    for j in range(d.shape[1]):
        valid=draws[:,j][np.isfinite(draws[:,j])];n=int(finite[:,j].sum())
        results.append({'matched_recordings':n,'source_clusters_with_values':int((counts[:,j]>0).sum()),
            'mean_B_minus_A':float(d[finite[:,j],j].mean()) if n else None,
            'bootstrap_95ci':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
            'valid_bootstrap_replicates':len(valid),'source_clusters_sampled':g})
    return results
