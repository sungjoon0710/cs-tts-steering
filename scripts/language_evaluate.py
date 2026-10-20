"""Recover provisional language diagnostics from existing annotated corpus CSVs."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,read_jsonl,write_json,write_jsonl
from src.evaluation.language import VERSION,corpus_lexicon,tag_text,difficulty,language_errors

def main(baseline,corpus,deva,output):
    output=private_directory(output)
    if output.exists():raise ValueError('Use a new output directory')
    rows=read_jsonl(baseline/'tables/metrics.jsonl');inputs={r['utterance_id']:r['text_raw'] for r in rows}
    with corpus.open(encoding='utf-8-sig',newline='') as f:annotations=list(csv.DictReader(f))
    with deva.open(encoding='utf-8-sig',newline='') as f:pairs=list(csv.DictReader(f))
    lexicon,stats=corpus_lexicon(annotations,pairs,inputs.values());write_json(output/'private_lexicon.json',lexicon)
    tagged={uid:tag_text(text,lexicon) for uid,text in inputs.items()}
    write_jsonl(output/'input_language.jsonl',[{'utterance_id':uid,'text_sha256':digest(inputs[uid].encode()),'labels':labels,'difficulty':difficulty(labels),'verified':False} for uid,labels in tagged.items()])
    result=[{'generation_id':r['generation_id'],'utterance_id':r['utterance_id'],'configuration_id':r['configuration_id'],
        'in_challenge_view':r['in_challenge_view'],'annotation_version':VERSION,
        **language_errors(r['text_raw'],r['asr_raw'],tagged[r['utterance_id']])} for r in rows]
    write_jsonl(output/'language_metrics.jsonl',result)
    sensitivity=[]
    for support,agreement in [(1,1.),(3,.95),(5,1.)]:
        all_tags=[tag_text(t,lexicon,support,agreement) for t in inputs.values()]
        sensitivity.append({'minimum_support':support,'minimum_agreement':agreement,
            'known_tokens':sum(x['label']!='UNKNOWN' for tags in all_tags for x in tags),
            'total_tokens':sum(len(tags) for tags in all_tags),'fully_labeled_inputs':sum(difficulty(tags)['fully_labeled'] for tags in all_tags)})
    write_json(output/'completion.json',{'status':'complete_provisional','recordings':len(result),'inputs':len(inputs),
        'corpus_sha256':digest(corpus.read_bytes()),'deva_corpus_sha256':digest(deva.read_bytes()),'source_stats':stats,
        'thresholds':{'minimum_support':3,'minimum_agreement':.95},'coverage_sensitivity':sensitivity,
        'verified_label_coverage':0,'matrix_language':'not_guessed','method':'lexicon labels, not script heuristics; baseline text excluded from lexicon construction'})
    print(json.dumps({'source_stats':stats,'coverage_sensitivity':sensitivity}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--deva-corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.baseline,a.corpus,a.deva_corpus,a.output)
