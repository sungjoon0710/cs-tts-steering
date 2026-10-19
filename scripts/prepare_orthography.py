"""Prepare a private orthography cohort; never generate speech or invent text variants."""
import argparse,html,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,private_directory,read_jsonl,stable_id,write_json,write_jsonl
from src.evaluation.text import normalize

def prepare(baseline,audit,output):
    baseline=Path(baseline);audit=Path(audit);output=private_directory(output)
    cohort=json.loads((baseline/'cohorts.json').read_text(encoding='utf-8'))
    originals={u['utterance_id']:u for u in read_jsonl(baseline/'tables/utterances.jsonl')}
    generations=read_jsonl(baseline/'tables/generations.jsonl')
    annotations=json.loads((audit/'remote_annotation_candidates.json').read_text(encoding='utf-8'))
    candidates=defaultdict(list)
    for d in annotations['datasets']:
        for c in d['candidates']:candidates[c['utterance_id']].append(c)
    join=[]
    for uid in sorted(originals,key=int):
        hits=candidates[uid];ids=sorted({c['corpus_id'] for c in hits})
        aligned=[len(c.get('labels_parsed',[]))==len(normalize(originals[uid]['text_raw']).split()) for c in hits]
        join.append({'utterance_id':uid,'candidate_corpus_ids':ids,'candidate_sources':sorted({c['source_path'] for c in hits}),
            'status':'no_exact_join' if not hits else 'ambiguous_corpus_identity' if len(ids)>1 else 'candidate_not_verified',
            'verified_language_labels':False,'pier':None,
            'candidate_tag_token_alignment':aligned,
            'blocker':'No verified corpus join; no row-index inheritance' if not hits else
                'Ambiguous corpus identity; manual adjudication required' if len(ids)>1 else
                'Candidate tags do not align with normalized tokens' if not any(aligned) else
                'Automated candidate labels require verification'})
    write_jsonl(output/'baseline_annotation_join.jsonl',join)
    write_json(output/'annotation_coverage.json',{'baseline_inputs':len(originals),'exact_text_candidate_inputs':len(candidates),
        'unique_candidate_identity_inputs':sum(len({c['corpus_id'] for c in cs})==1 for cs in candidates.values()),
        'ambiguous_inputs':sum(len({c['corpus_id'] for c in cs})>1 for cs in candidates.values()),
        'verified_language_coverage':0,'PIER_status':'pending; do not create baseline-wide labels from indices or script'})
    rows=[]
    for g in generations:
        uid=g['utterance_id']
        if uid not in cohort['original_pilot_ids']:continue
        for condition in ('original','devanagari_hindi_english_insertions','romanized_hindi_english_insertions'):
            rows.append({'experiment_id':'orthographic_robustness_v1_PENDING','semantic_id':'baseline:'+uid,
                'utterance_id':uid,'trial_generation_id':stable_id('orthography_v1',uid,g['configuration_id'],condition),
                'configuration_id':g['configuration_id'],'script_condition':condition,
                'original_text':originals[uid]['text_raw'],'input_text':originals[uid]['text_raw'] if condition=='original' else None,
                'equivalent_content_checked':False,'equivalence_reviewer':None,
                'source_baseline_generation_id':g['generation_id'] if condition=='original' else None,
                'audio_path':g['remote_audio_path'] if condition=='original' else None,
                'generation_status':'existing_baseline' if condition=='original' else 'pending_collaborator_audio',
                'seed':None,'decoding_settings':None,'checkpoint_revision':None,'attempt':None,
                'controls_status':'must verify voice, seed, checkpoint and decoding match before paired attribution',
                'new_generation_authorized_by_this_script':False})
    write_jsonl(output/'orthography_manifest.jsonl',rows)
    source=audit/'orthography_source_pairs.json'
    if source.exists():write_json(output/'paired_corpus_candidates.json',json.loads(source.read_text(encoding='utf-8')))
    write_json(output/'team_dependencies.json',{'Alehsan':['Indic-Speak Romanized support investigation','evaluation, annotation and orthography-equivalence validation'],
        'Sungjoon':['Orpheus FT female-voice support evidence and exact checkpoint/revision','Romanized samples with paired semantic IDs and original inputs','seed, decoding, voice prompt, attempts, audio SHA and manifests'],
        'access':['Baseline already readable; no broad permission change needed','For new samples, request read/traverse only on the specific output and manifest directories','Own evaluation workspace remains private; use a designated exchange path if team agrees'],
        'not_done':['No new TTS generations','No formal orthography comparison','No inference that Ritika label proves supported female voice']})
    cards=[]
    for uid in cohort['original_pilot_ids']:
        group=[r for r in rows if r['utterance_id']==uid]
        cards.append('<h2>Input '+html.escape(uid)+'</h2><p>'+html.escape(originals[uid]['text_raw'])+'</p><table><tr><th>Configuration</th><th>Original</th><th>Hindi in Devanagari + English insertions</th><th>Hindi in Latin + English insertions</th></tr>')
        for config in sorted({r['configuration_id'] for r in group}):
            cards.append('<tr><td>'+html.escape(config)+'</td><td>Existing baseline; remote path in manifest</td><td>Pending equivalent text and audio</td><td>Pending equivalent text and audio</td></tr>')
        cards.append('</table>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Orthographic robustness preparation</title><style>body{font:16px system-ui;margin:30px}td,th{border:1px solid #bbb;padding:9px}table{border-collapse:collapse}</style><h1>Orthographic robustness — prepared, not evaluated</h1><p>Researcher view. Intended paired comparison changes only Hindi orthography within one voice/configuration. No new samples or judgments fabricated. Fill missing manifest fields and verify equivalent content before comparison. Existing 15 Latin-only baseline inputs are not a controlled Romanized-Hindi experiment.</p>'+''.join(cards),encoding='utf-8')
    print('Prepared annotation-join audit, 20-input orthography scaffold and team checklist; no speech generated')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.baseline,a.audit,a.output)
