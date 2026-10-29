import pytest
from pathlib import Path
from types import SimpleNamespace
from scripts.prepare_matched_cohort import script_pair, choose
from scripts.matched_generate import jobs


def test_script_pair_preserves_english_case_punctuation_and_hindi_spans():
    assert script_pair('aap Meeting mein aao!', 'आप Meeting में आओ!', ['HI','EN','HI','HI']) == 'आप Meeting में आओ!'
    with pytest.raises(ValueError,match='English'):
        script_pair('aap Meeting mein aao!', 'आप party में आओ!', ['HI','EN','HI','HI'])
    with pytest.raises(ValueError,match='not converted'):
        script_pair('aap Meeting mein aao!', 'aap Meeting में आओ!', ['HI','EN','HI','HI'])
    with pytest.raises(ValueError,match='ambiguous language'):
        script_pair('mai Meeting mein aao!', 'mai Meeting में आओ!', ['EN','EN','HI','HI'])


def test_selection_is_reproducible_and_rejects_duplicate_content_conversations():
    rows=[dict(source_name=str(i),source_speaker='speaker',length_bin='short',
               conversation_id=str(i//2),canonical_hash=str(i//3)) for i in range(20)]
    first=choose(rows,4,1234)
    assert first==choose(rows,4,1234)
    assert len({r['conversation_id'] for r in first})==len(first)
    assert len({r['canonical_hash'] for r in first})==len(first)
    with pytest.raises(ValueError,match='Only'):
        choose(rows,20,1234)


def test_script_pairs_share_canonical_reference_seed_and_token_limit():
    row=dict(utterance_id='example',input_romanized='aap Meeting mein aao!',
             input_devanagari='आप Meeting में आओ!',canonical_text='आप Meeting में आओ!',pilot=True)
    planned=jobs([row])
    assert len(planned)==5 and len({r['generation_id'] for r in planned})==5
    assert len({r['max_new_tokens'] for r in planned})==1
    assert len({r['canonical_text_sha256'] for r in planned})==1
    assert {r['seed'] for r in planned}=={1234}
    changed=dict(row,input_romanized=row['input_romanized']+' please')
    assert jobs([changed])[1]['generation_id']!=planned[1]['generation_id']


def test_resume_recovers_missing_speaker_output_after_asr_was_cached(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import matched_evaluate,learned_evaluate
    from src.evaluation.common import write_jsonl
    rows=[{'generation_id':gid,'pilot':True,'audio':{'status':'computed_unvalidated','audio_sha256':gid}} for gid in ['a','b']]
    write_jsonl(tmp_path/'tables/metrics.jsonl',rows)
    prior=tmp_path/'metrics/asr-small_pilot_batch0001'
    write_jsonl(prior/'second_asr.jsonl',[{'generation_id':gid,'audio_sha256':gid,'status':'computed_asr_proxy','hypothesis_raw':'example'} for gid in ['a','b']])
    write_jsonl(prior/'speaker_embeddings.jsonl',[{'generation_id':'a','audio_sha256':'a','embedding':[1.,0.]}])
    processed=[]
    monkeypatch.setattr(learned_evaluate,'asr_speaker',lambda pending,*args,**kwargs:processed.extend(r['generation_id'] for r in pending))
    matched_evaluate.learned(SimpleNamespace(run=tmp_path,scope='full',action='asr-small',cache=tmp_path/'cache',offline=True))
    assert processed==['b']
