"""Synthetic fixtures only: no private corpus text or research judgments."""
import json
from collections import Counter

import numpy as np
import pytest
import soundfile as sf

from src.evaluation.audio import extract_audio
from src.evaluation.common import digest, private_directory, read_jsonl, write_json, write_jsonl
from src.evaluation.listening import ListeningStudy, make_trials
from src.evaluation.records import import_snapshot, select_pilot
from src.evaluation.report import paired_bootstrap, summarize_judgments
from src.evaluation.text import checked_language, edit_score, normalize, pier


def test_normalization_preserves_script_and_content():
    assert normalize('  HELLO, दुनिया!  ') == 'hello दुनिया'
    assert normalize("Don't change 3.14 or 10:30 or well-known") == "don't change 3.14 or 10:30 or well-known"
    assert normalize('हिन्दी ENGLISH १२३') == 'हिन्दी english १२३'
    assert normalize('cafe\u0301') == normalize('café')
    assert normalize('hello') != normalize('हेलो')
    assert normalize('a a a') == 'a a a'


def test_edit_counts_missing_and_silence():
    r = edit_score('one two three', 'one four')
    assert (r['substitutions'], r['deletions'], r['insertions']) == (1, 1, 0)
    assert r['wer'] == pytest.approx(2 / 3)
    assert edit_score('word', '')['wer'] == 1
    assert edit_score('word', None)['wer'] is None
    assert edit_score('', 'word')['status'] == 'invalid_empty_reference'
    assert edit_score('a', 'a b c')['wer'] == 2


@pytest.mark.parametrize('ref,hyp,poi,expected', [
    (['a', 'b'], ['a', 'x'], [1], 1),
    (['a', 'b'], ['a'], [1], 1),
    (['a', 'b'], ['x', 'a', 'b'], [0], 1),
    (['a', 'b'], ['a', 'b', 'x'], [1], 1),
    (['a', 'b'], ['a', 'b', 'x'], [0], 0),
    (['a', 'b', 'c'], ['a', 'x', 'b', 'c'], [1], 1),
])
def test_pier_official_insertion_anchoring(ref, hyp, poi, expected):
    assert pier(ref, hyp, poi)['pier'] == expected


def test_pier_no_poi_is_not_zero():
    assert pier(['a'], ['x'], [])['pier'] is None
    with pytest.raises(ValueError): pier(['a'], [], [1])


def test_annotations_fail_closed():
    text = 'नमस्ते world'
    ann = {'verified': True, 'text_sha256': digest(text.encode()), 'tokens': normalize(text).split(), 'labels': ['HI', 'EN']}
    assert checked_language(ann, text) == ['HI', 'EN']
    ann['labels'] = ['HI', 'UNCERTAIN']
    assert checked_language(ann, text) is None
    ann['text_sha256'] = 'wrong'
    with pytest.raises(ValueError): checked_language(ann, text)


def test_audio_silence_noise_and_tone(tmp_path):
    sr = 24000
    silent = tmp_path / 'silent.wav'; sf.write(silent, np.zeros(sr), sr, subtype='PCM_16')
    assert extract_audio(silent)['status'] == 'fully_silent'
    tone = tmp_path / 'tone.wav'
    t = np.arange(sr * 2) / sr
    sf.write(tone, np.concatenate((np.zeros(sr//2), .15*np.sin(2*np.pi*220*t), np.zeros(sr//2))), sr)
    r = extract_audio(tone)
    assert r['duration_s'] == 3
    assert r['f0_median_hz'] == pytest.approx(220, abs=3)
    assert r['f0_range_st'] < 1
    assert r['silence_proportion'] > .1
    noise = tmp_path / 'noise.wav'; sf.write(noise, np.random.default_rng(1).normal(0,.1,sr),sr)
    assert extract_audio(noise)['status'] != 'fully_silent'
    stereo = tmp_path / 'stereo.wav'; sf.write(stereo, np.zeros((sr,2)),sr)
    assert extract_audio(stereo)['status'] == 'unsupported_channels'
    assert extract_audio(tmp_path/'missing.wav')['status'] == 'missing_audio'
    bad = tmp_path/'bad.wav'; bad.write_bytes(b'not a wav')
    assert extract_audio(bad)['status'] == 'unreadable_audio'


def test_bootstrap_is_paired_and_deterministic():
    r = paired_bootstrap([1,2,3],[3,4,5])
    assert r['mean_B_minus_A'] == 2
    assert r['mean_difference_bootstrap_95ci'] == [2,2]
    assert r == paired_bootstrap([1,2,3],[3,4,5])


def fixtures(tmp_path):
    gens = []
    for uid in ['0','1','2']:
        for config in ['config_a','config_b','config_c']:
            gens.append({'generation_id': uid+config, 'utterance_id':uid, 'configuration_id':config,
                         'relative_audio_path': config+'/'+uid+'.wav'})
    write_json(tmp_path/'pilot.json', {'utterance_ids':['0','1','2']})
    write_jsonl(tmp_path/'tables/generations.jsonl', gens)
    write_jsonl(tmp_path/'tables/utterances.jsonl', [{'utterance_id':uid,'text_raw':'SYNTHETIC TARGET '+uid} for uid in ['0','1','2']])
    write_jsonl(tmp_path/'tables/metrics.jsonl', [{'generation_id':g['generation_id'],'audio':{'audio_sha256':'fixture-hash'}} for g in gens])
    return gens


def test_three_pair_assignment_and_balance(tmp_path):
    gens = fixtures(tmp_path); totals=Counter()
    for slot in range(15):
        trials = make_trials(gens, ['0','1','2'], 'synthetic_'+str(slot),slot)
        assert len({t['utterance_id'] for t in trials}) == 3
        for t in trials: totals[(t['utterance_id'],tuple(sorted((t['configuration_A'],t['configuration_B']))))] += 1
    assert set(totals.values()) == {5}
    assert make_trials(gens,['0','1','2'],'x') == make_trials(gens,['0','1','2'],'x')


def test_listener_blinding_lock_and_resume(tmp_path):
    fixtures(tmp_path); study=ListeningStudy(tmp_path)
    study.enroll('synthetic_listener', {'consent':True})
    view=study.view('synthetic_listener')
    assert 'transcript' not in view
    assert 'config_a' not in json.dumps(view)
    assert study.transcription_view('synthetic_listener').get('transcript') is None
    with pytest.raises(ValueError): study.finish('synthetic_listener',view['trial_id'],'a','a','')
    study.naturalness('synthetic_listener',view['trial_id'],'both_unacceptable')
    with pytest.raises(ValueError): study.naturalness('synthetic_listener',view['trial_id'],'a')
    study=ListeningStudy(tmp_path)
    assert study.view('synthetic_listener')['phase'] == 'diagnostics'
    assert 'transcript' in study.view('synthetic_listener')
    study.finish('synthetic_listener',view['trial_id'],'b','uncertain','synthetic fixture')
    r=read_jsonl(tmp_path/'human/judgments.jsonl')[0]
    assert r['naturalness']=='both_unacceptable'
    assert study.view('synthetic_listener')['trial_id'] != view['trial_id']


def test_annotation_hash_and_explicit_completeness(tmp_path):
    fixtures(tmp_path); study=ListeningStudy(tmp_path)
    study.enroll('synthetic_transcriber',{'consent':True})
    task=study.transcription_view('synthetic_transcriber')
    study.transcribe('synthetic_transcriber',task['task_id'],'synthetic spoken text',False)
    r=read_jsonl(tmp_path/'annotations/manual_transcripts.jsonl')[0]
    assert r['verified'] is False and r['audio_sha256']=='fixture-hash'


def test_four_categories_not_collapsed():
    rows=[{'configuration_A':'a','configuration_B':'b','naturalness':x} for x in
          ['a','b','both_acceptable','both_unacceptable']]
    c=summarize_judgments(rows)[0]['categories']
    assert len(c)==4 and c['both_unacceptable']==1


def test_selection_is_deterministic_and_not_language_labels():
    rows=[{'utterance_id':str(i),'script_profile':['mixed_script','devanagari_only','latin_only'][i%3],
           'normalized_words':i%30+1} for i in range(90)]
    p=select_pilot(rows)
    assert p==select_pilot(rows) and len(set(p['utterance_ids']))==20
    assert p['language_validation']=='pending'
    assert p['selection_counts']=={'mixed_script':14,'devanagari_only':3,'latin_only':3}


def test_private_outputs_rejected_in_git(tmp_path):
    (tmp_path/'.git').mkdir()
    with pytest.raises(ValueError): private_directory(tmp_path/'outputs')


def test_retry_association_and_duplicate_ids(tmp_path):
    write_jsonl(tmp_path/'utterances.jsonl',[{'line':24,'speaker':'fixture','text':'corrected text'}])
    base={'line':24,'model':'fixture','voice':'voice','wav':'/baseline/fixture/voice/24.wav'}
    write_jsonl(tmp_path/'fixture/generations.jsonl',[{**base,'text':'','error':'empty text'},{**base,'text':'corrected text'}])
    write_jsonl(tmp_path/'audio_inventory.jsonl',[{'relative_path':'fixture/voice/24.wav','status':'header_ok'}])
    write_jsonl(tmp_path/'eval/asr.jsonl',[{'wav':base['wav'],'hyp':'corrected text'}])
    _,attempts,gens,_=import_snapshot(tmp_path)
    assert len(attempts)==2 and len(gens)==1 and gens[0]['successful_retry']
    assert gens[0]['raw_generation_status']=='success'
    write_jsonl(tmp_path/'utterances.jsonl',[{'line':24,'speaker':'fixture','text':'text'}]*2)
    with pytest.raises(ValueError): import_snapshot(tmp_path)


def test_source_snapshot_tampering_rejected(tmp_path):
    p=tmp_path/'utterances.jsonl';p.write_text('original',encoding='utf-8')
    write_json(tmp_path/'snapshot.json',{'sha256':{'utterances.jsonl':digest(p.read_bytes())}})
    p.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Source snapshot changed'): import_snapshot(tmp_path)


def test_failure_review_requires_explicit_categories_and_hash(tmp_path):
    from src.evaluation.listening import FAILURES
    fixtures(tmp_path);study=ListeningStudy(tmp_path);study.enroll('synthetic_review',{'consent':True})
    task=study.review_view('synthetic_review')
    with pytest.raises(ValueError):study.review('synthetic_review',task['task_id'],{},'')
    study.review('synthetic_review',task['task_id'],{k:k=='repetition' for k in FAILURES},'synthetic fixture')
    rows=read_jsonl(tmp_path/'annotations/failure_reviews.jsonl')
    assert rows[0]['flags']['repetition'] and rows[0]['audio_sha256']=='fixture-hash'


def test_crossed_human_uncertainty_retains_four_categories():
    rows=[]
    choices=['a','b','both_acceptable','both_unacceptable']
    for listener in range(4):
        for uid in range(5):
            rows.append({'configuration_A':'a','configuration_B':'b','naturalness':choices[(listener+uid)%4],
                         'listener_id':str(listener),'utterance_id':str(uid)})
    result=summarize_judgments(rows)[0]
    assert sum(result['categories'].values())==20
    assert len(result['category_proportion_95ci'])==4
    assert all(0<=lo<=hi<=1 for lo,hi in result['category_proportion_95ci'].values())
