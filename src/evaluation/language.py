"""Provisional corpus-lexicon language diagnostics, separate from verified PIER.

No script-to-language inference. Label agreement is not calibrated accuracy.
Unknown words remain unknown; no row-order corpus joins or guessed labels.
"""
from __future__ import annotations
import ast,re,unicodedata
from collections import Counter,defaultdict
from .text import normalize,pier

VERSION='masac-lexicon-provisional-v1'

def lexical_tokens(text):
    result=[];current='';kind=None
    def word(c):return bool(c) and (c.isalnum() or c=='_' or unicodedata.category(c).startswith('M'))
    for i,c in enumerate(text):
        k='space' if c.isspace() else 'word' if word(c) or (c in "'\u2019:/-" and i>0 and i+1<len(text) and word(text[i-1]) and word(text[i+1])) else 'punct'
        if k!=kind or k=='space':
            if current:result.append(current)
            current='';kind=k
        if k!='space':current+=c
    if current:result.append(current)
    return result

def aligned_labels(text,cell):
    try:tags=ast.literal_eval(cell) if isinstance(cell,str) else list(cell)
    except (SyntaxError,ValueError,TypeError):return None
    if not isinstance(tags,(list,tuple)) or any(t not in ('HI','EN','OTHER') for t in tags):return None
    candidates=[re.findall(r'\w+|[^\w\s]+',text),
        re.findall(r"[\w]+(?:['\u2019:/-][\w]+)*|[^\w\s]+",text),lexical_tokens(text)]
    projections=set()
    for tokens in candidates:
        if len(tokens)!=len(tags):continue
        words=[];labels=[];valid=True
        for token,label in zip(tokens,tags):
            ns=normalize(token).split()
            if len(ns)>1:valid=False;break
            if ns:words+=ns;labels.append(label)
        if valid and words==normalize(text).split():projections.add(tuple(labels))
    return list(next(iter(projections))) if len(projections)==1 else None

def corpus_lexicon(rows,deva_rows=(),exclude_texts=()):
    exclude={normalize(t) for t in exclude_texts};paired={r['name']:r for r in deva_rows}
    counts=defaultdict(Counter);stats=Counter();seen=set()
    for row in rows:
        stats['source_rows']+=1;text=row['Transcript'];normalized=normalize(text)
        if normalized in exclude:stats['excluded_baseline_text_rows']+=1;continue
        labels=aligned_labels(text,row.get('LID_tags',''))
        if labels is None:stats['unalignable_or_ambiguous_rows']+=1;continue
        signature=(normalized,tuple(labels))
        if signature in seen:stats['duplicate_text_label_rows']+=1;continue
        seen.add(signature);stats['accepted_romanized_rows']+=1
        # One observation per word/label/row; repetitions do not boost support.
        roman_observations=set(zip(normalized.split(),labels))
        for word,label in roman_observations:counts[word][label]+=1
        deva=paired.get(row['name'])
        if deva:
            other=normalize(deva['Transcript']).split();words=normalized.split()
            if len(other)==len(words) and all(a==b for a,b,label in zip(words,other,labels) if label!='HI'):
                stats['paired_devanagari_rows']+=1
                for word,label in set(zip(other,labels))-roman_observations:counts[word][label]+=1
            else:stats['deva_pair_rejected']+=1
    return {w:dict(c) for w,c in counts.items()},dict(stats)

def tag_text(text,lexicon,minimum_support=3,minimum_agreement=.95):
    output=[]
    for word in normalize(text).split():
        c=lexicon.get(word,{});total=sum(c.values());winner=max(c,key=c.get) if c else None
        agreement=c.get(winner,0)/total if total else None
        accepted=winner in ('HI','EN') and total>=minimum_support and agreement>=minimum_agreement
        output.append({'token':word,'label':winner if accepted else 'UNKNOWN','corpus_support':total,'corpus_agreement':agreement})
    return output

def difficulty(tagged):
    labels=[r['label'] for r in tagged];n=len(labels);known=sum(l in ('HI','EN') for l in labels)
    switches=[i for i in range(n-1) if labels[i] in ('HI','EN') and labels[i+1] in ('HI','EN') and labels[i]!=labels[i+1]]
    full=known==n and n>0;en=labels.count('EN');hi=labels.count('HI')
    return {'status':'provisional_corpus_lexicon_not_human_verified','token_count':n,'known_token_count':known,
        'coverage':known/n if n else None,'fully_labeled':full,'english_proportion':en/n if full else None,
        'english_proportion_bounds':[en/n,(en+n-known)/n] if n else None,
        'observed_switch_positions':switches,'observed_switch_count_lower_bound':len(switches),
        'observed_switch_directions':[labels[i]+'->'+labels[i+1] for i in switches],
        'condition':('code_switched' if en and hi else 'english_only' if en else 'hindi_only') if full else 'partial_labels',
        'm_index':(1-((en/n)**2+(hi/n)**2))/((en/n)**2+(hi/n)**2) if full else None,
        'i_index':len(switches)/(n-1) if full and n>1 else None,
        'cmi_fraction':1-max(en,hi)/n if full else None,'strategy':'not_inferred'}

def language_errors(reference,hypothesis,tagged):
    if [r['token'] for r in tagged]!=normalize(reference).split():raise ValueError('Language tokens and reference do not align')
    if hypothesis is None:return {'status':'missing_asr','english_poi':None,'hindi_poi':None,'switch_window_poi':None}
    ref=normalize(reference).split();hyp=normalize(hypothesis).split();labels=[r['label'] for r in tagged]
    switches=difficulty(tagged)['observed_switch_positions']
    window=sorted({j for i in switches for j in range(max(0,i-1),min(len(ref),i+3))})
    return {'status':'provisional_reference_language_POI_errors','english_poi':pier(ref,hyp,[i for i,l in enumerate(labels) if l=='EN']),
        'hindi_poi':pier(ref,hyp,[i for i,l in enumerate(labels) if l=='HI']),
        'switch_window_poi':pier(ref,hyp,window),'switch_window_reference_indices':window,
        'embedded_pier':None,'embedded_pier_blocker':'Matrix language and embedded-token labels require verification',
        'interpretation':'Official alignment/insertion rules on provisional subsets; not validated synthesis language errors'}
