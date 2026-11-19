"""Freeze M1 text/document splits without reading any new model outcomes."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,random,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,write_json,write_jsonl

EMOTION=set('afraid angry annoyed anxiety anxious disappointed disappointment excited excitement fear fearful frustrated frustration glad grateful happy hate hated love loved miserable nervous pleased regret sad sadness scared sorry stressed stress terrified upset worried worry wonderful unhappy unfortunately'.split())
REFERENTIAL=set('this that these those it they them he she him her'.split())


def category(text):
    w=set(re.findall('[A-Za-z]+',text.lower()))
    return ('emotion_marked',sorted(w&EMOTION)) if w&EMOTION else ('referential_context',sorted(w&REFERENTIAL)) if w&REFERENTIAL else ('lexical_control',[])


def conllu(path):
    doc=None;file_sha=digest(path.read_bytes())
    for block in path.read_text(encoding='utf-8').split('\n\n'):
        fields={l[2:].split(' = ',1)[0]:l[2:].split(' = ',1)[1] for l in block.splitlines() if l.startswith('# ') and ' = ' in l}
        doc=fields.get('newdoc id',doc)
        if fields.get('text'):
            yield {'text':fields['text'],'source_document':doc,'source_id':fields['sent_id'],
                'source_split':path.name,'source_file_sha256':file_sha}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--text',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(20261116)
    prior=[json.loads(s) for s in (a.prior/'geometry100/cohort.jsonl').read_text().splitlines()]
    exposed_docs={r['source_document'] for r in prior};exposed_texts={r['text_SHA256'] for r in prior}
    authored=[r for r in prior if r['id'] in ('c01','c02','c03','c04')]
    rows=[]
    for r in authored:
        cat,evidence=category(r['reply'])
        rows.append({'id':r['id'],'study_split':'development','text':r['reply'],'category':cat,'category_evidence':evidence,
            'source':'prior_authored_calibration','source_document':r['source_document'],'source_id':r['id'],
            'text_sha256':r['text_SHA256'],'prior_exposure':'previous anchor, audio and/or trace analyses',
            'existing_history_key':r['id'],'source_split':r['split'],'second_history':True})
    candidates=defaultdict(list)
    for file in a.text:
        for r in conllu(file):
            t=r['text'];words=re.findall('[A-Za-z]+',t);sha=digest(t.encode())
            if not t.isascii() or not 8<=len(words)<=30 or len(t)>220 or re.search(r'\d|https?://|www\.|@',t) or not t.endswith(('.','!','?')):continue
            if r['source_document'] in exposed_docs or sha in exposed_texts:continue
            cat,evidence=category(t)
            candidates[cat].append({**r,'category':cat,'category_evidence':evidence,'text_sha256':sha,'source':'UD_English-EWT_r2.18',
                'prior_exposure':'not in prior 100-text manifest','existing_history_key':None,'second_history':False})
    counts={c:len({r['source_document'] for r in rs}) for c,rs in candidates.items()}
    # Reserve scarce emotion-marked documents first. A document is assigned to
    # exactly one study row even when it contains several eligible sentences.
    docs=set(exposed_docs);text_hashes=set(exposed_texts)
    new={}
    for cat in ('emotion_marked','referential_context','lexical_control'):
        pool=sorted(candidates[cat],key=lambda r:(r['source_document'],r['source_id']));rng.shuffle(pool)
        needed=40-sum(r['category']==cat for r in rows);selected=[]
        for r in pool:
            if r['source_document'] in docs or r['text_sha256'] in text_hashes:continue
            docs.add(r['source_document']);text_hashes.add(r['text_sha256']);selected.append(r)
            if len(selected)==needed:break
        if len(selected)!=needed:raise ValueError(f'Insufficient independent {cat} documents: {len(selected)}/{needed}')
        new[cat]=selected
    dev_index=5;confirm_index=1
    for cat in ('emotion_marked','referential_context','lexical_control'):
        dev_needed=10-sum(r['category']==cat for r in rows)
        for k,r in enumerate(new[cat]):
            split='development' if k<dev_needed else 'confirmation'
            rid=f'd{dev_index:03d}' if split=='development' else f'q{confirm_index:03d}'
            if split=='development':dev_index+=1
            else:confirm_index+=1
            rows.append({**r,'id':rid,'study_split':split})
    # History replication: 10 development (four historical + two/category new),
    # 30 confirmation (10/category). Selection precedes all new model outputs.
    for cat in ('emotion_marked','referential_context','lexical_control'):
        dev=[r for r in rows if r['study_split']=='development' and r['category']==cat and not r['existing_history_key']]
        conf=[r for r in rows if r['study_split']=='confirmation' and r['category']==cat]
        rng.shuffle(dev);rng.shuffle(conf)
        for r in dev[:2]+conf[:10]:r['second_history']=True
    for split,n in [('development',6),('confirmation',3)]:
        for cat in ('emotion_marked','referential_context','lexical_control'):
            candidates_split=sorted([r for r in rows if r['study_split']==split and r['category']==cat],key=lambda r:r['id'])
            for r in candidates_split[:n//3]:r['capture_exact']=True
    for r in rows:
        r.setdefault('capture_exact',False)
        r['numerical_repeat']=r['capture_exact']
        r['word_count']=len(re.findall('[A-Za-z]+',r['text']))
    rows=sorted(rows,key=lambda r:(r['study_split']!='development',r['id']))
    assert len(rows)==120 and sum(r['study_split']=='confirmation' for r in rows)==90
    assert len({r['source_document'] for r in rows})==120
    assert len({r['text_sha256'] for r in rows})==120
    assert sum(r['second_history'] for r in rows)==40
    assert Counter((r['study_split'],r['category']) for r in rows)==Counter({('development',c):10 for c in new}|{('confirmation',c):30 for c in new})
    write_jsonl(out/'cohort.jsonl',rows)
    write_json(out/'selection.json',{'seed':20261116,'available_distinct_documents_by_operational_category':counts,
        'counts':{s:dict(Counter(r['category'] for r in rows if r['study_split']==s)) for s in ('development','confirmation')},
        'second_histories':dict(Counter(r['study_split'] for r in rows if r['second_history'])),
        'full_tensor_capture_ids':[r['id'] for r in rows if r['capture_exact']],
        'emotion_lexicon':sorted(EMOTION),'referential_words':sorted(REFERENTIAL),
        'classification':'Emotion lexical membership takes priority over referential-word membership; otherwise lexical control. Operational, not gold emotional/neutral/context labels.',
        'confirmation_rule':'No previously analyzed text hash or source document; at most one sentence/document in all new study data.',
        'source_files':{str(f):digest(f.read_bytes()) for f in a.text},'prior_manifest_sha256':digest((a.prior/'geometry100/cohort.jsonl').read_bytes()),
        'cohort_sha256':digest((out/'cohort.jsonl').read_bytes()),
        'limitation':'Distinct documents do not guarantee distinct authors/topics. EWT split names are unrelated to M1 roles; no claim of unseen LM pretraining data.'})
    print(json.dumps({'rows':len(rows),'counts':dict(Counter(r['study_split'] for r in rows)),'second_histories':40}))


if __name__=='__main__':main()
