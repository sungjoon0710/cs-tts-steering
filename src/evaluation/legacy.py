"""Text-only reproduction of the upstream evaluator's published formulas.

This module never imports the GPU evaluator or loads an ASR/speaker model.
Latin-token fuzzy survival is preserved for audit, not validated as English fidelity.
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from .expanded import romanizer

SOURCE_COMMIT='165ea45'
LATIN_WORD=re.compile(r'[A-Za-z]{2,}')

def legacy_romanize(text):
    s=romanizer().romanize_string(text).lower()
    return ' '.join(re.sub(r'[^a-z0-9 ]+',' ',s).split())

def latin_fuzzy_survival(reference,hypothesis_romanized,threshold=.5):
    words=[w.lower() for w in LATIN_WORD.findall(reference)]
    tokens=hypothesis_romanized.split()
    found=sum(any(SequenceMatcher(None,w,t).ratio()>=threshold for t in tokens) for w in words)
    return found,len(words)

def legacy_content(reference,hypothesis):
    import jiwer
    if hypothesis is None:return {'status':'missing_asr','wer':None,'cer':None,'latin_found':None,'latin_total':None}
    ref,hyp=legacy_romanize(reference),legacy_romanize(hypothesis)
    found,total=latin_fuzzy_survival(reference,hyp)
    return {'status':'legacy_reproduction_not_quality_validation','source_commit':SOURCE_COMMIT,
        'wer':jiwer.wer(ref,hyp) if ref else None,'cer':jiwer.cer(ref,hyp) if ref else None,
        'latin_found':found,'latin_total':total,'cer_definition':'jiwer character distance includes normalized spaces'}
