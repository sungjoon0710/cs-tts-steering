import io
import json
import zipfile
import numpy as np
import pytest
import soundfile as sf
from src.evaluation.audio import extract_audio
from src.evaluation.common import digest
from src.evaluation.expanded import content_proxies,expanded_audio,raw_score,romanize
from scripts.sync_evaluation import unpack_results

def test_stereo_average_is_explicit_and_does_not_change_source(tmp_path):
    sr=24000;t=np.arange(sr)/sr;y=.2*np.sin(2*np.pi*220*t)
    path=tmp_path/'stereo.wav';sf.write(path,np.column_stack([y,.5*y]),sr,subtype='PCM_16')
    before=path.read_bytes()
    assert extract_audio(path)['status']=='unsupported_channels'
    first=extract_audio(path,'first');mean=extract_audio(path,'mean')
    assert mean['rms_dbfs']-first['rms_dbfs']==pytest.approx(20*np.log10(.75),abs=.002)
    assert mean['f0_median_hz']==pytest.approx(220,abs=1)
    assert path.read_bytes()==before

def test_missing_asr_cannot_create_valid_content_score():
    strict,raw,harmonized,proxies=content_proxies('one two',None,{'duration_s':2})
    assert strict['wer'] is None and raw['wer'] is None and harmonized['wer'] is None
    assert proxies['asr_speaking_rate_wps'] is None
    assert proxies['asr_word_count'] is None and proxies['asr_repeated_trigram_fraction'] is None
    assert raw_score('Hello!','hello')['wer']==1
    assert content_proxies('Hello!','hello',{'duration_s':1})[0]['wer']==0

def test_uroman_is_script_sensitivity_not_translation():
    assert romanize('\u0928\u092e\u0938\u094d\u0924\u0947')==romanize('namaste')
    assert romanize('\u0928\u092e\u0938\u094d\u0924\u0947')!=romanize('hello')

def test_egemaps_and_voice_quality_on_known_tone(tmp_path):
    path=tmp_path/'tone.wav';sr=24000;t=np.arange(sr*2)/sr
    sf.write(path,.1*np.sin(2*np.pi*220*t),sr)
    result=expanded_audio(path)
    assert result['egemaps']['status']=='exploratory'
    assert len(result['egemaps']['features'])==88
    assert result['f0_mean_hz']==pytest.approx(220,abs=1)
    assert sum(result['joint_frame_class_fractions'].values())==pytest.approx(1)
    assert result['lab_voice_quality']['status']=='exploratory_unvalidated_on_synthetic_speech'

def bundle(name,data):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr(name,data)
        z.writestr('transfer_manifest.json',json.dumps({'sha256':{name:digest(data)}}))
    return stream.getvalue()

def test_result_sync_refuses_traversal_and_overwrite(tmp_path):
    for name in ('../escape.json','/absolute.json','C:/escape.json','x\\escape.json'):
        with pytest.raises(ValueError):unpack_results(bundle(name,b'{}'),tmp_path)
    unpack_results(bundle('data.json',b'{}'),tmp_path)
    unpack_results(bundle('data.json',b'{}'),tmp_path)
    with pytest.raises(ValueError):unpack_results(bundle('data.json',b'{"new":true}'),tmp_path)
    assert (tmp_path/'data.json').read_bytes()==b'{}'

def test_reference_join_excludes_mismatch_and_duplicate_conversation(tmp_path):
    import csv
    from scripts.natural_distribution import select_references
    (tmp_path/'wavfiles_utt_level').mkdir()
    records={};annotations=[]
    for i in range(1,6):
        records[str(i)]={'context_utterances':['same content '+str(i)],'context_speakers':['SPEAKER'],
            'target_utterance':'target '+str(i),'target_speaker':'SPEAKER','episode_name':'E1'}
        for j,text in enumerate([records[str(i)]['context_utterances'][0],records[str(i)]['target_utterance']]):
            name=f'train_{i}_{j}.wav';(tmp_path/'wavfiles_utt_level'/name).write_bytes(f'{i}:{j}'.encode())
            annotations.append({'name':name,'Transcript':text if i!=5 else 'wrong words','Speaker':'SPEAKER'})
    for split in ('train','val','test'):(tmp_path/(split+'_data_final_plain.json')).write_text(json.dumps(records if split=='train' else {}))
    with (tmp_path/'fixed_with_automated_annotations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['name','Transcript','Speaker']);writer.writeheader();writer.writerows(annotations)
    a,_=select_references(tmp_path,count=4);b,_=select_references(tmp_path,count=4)
    assert a==b
    assert len({r['conversation'] for r in a})==4
    assert all(r['conversation']!='train_5' for r in a)

def test_distribution_comparison_rejects_unfinished_baseline(tmp_path):
    from scripts.natural_distribution import compare
    with pytest.raises(ValueError,match='complete'):
        compare(tmp_path/'reference',tmp_path/'unfinished',tmp_path/'output')
    assert not (tmp_path/'output').exists()

def test_legacy_survival_is_not_one_to_one_english_preservation():
    from src.evaluation.legacy import latin_fuzzy_survival,legacy_content
    assert latin_fuzzy_survival('ram ram','tram')==(2,2)
    assert latin_fuzzy_survival('I a','i a')==(0,0)
    assert legacy_content('Hello!','hello')['wer']==0
    assert legacy_content('hello',None)['wer'] is None

def test_release_excludes_local_gpt_tools_and_account_notes(tmp_path):
    from scripts.sync_evaluation import release_bundle
    (tmp_path/'scripts').mkdir();(tmp_path/'docs').mkdir()
    (tmp_path/'scripts/build_full_handoff.py').write_text('private prompt')
    (tmp_path/'scripts/full_evaluate.py').write_text('public evaluator')
    (tmp_path/'docs/EVALUATION_STATUS.md').write_text('personal audit')
    _,data,hashes=release_bundle(tmp_path)
    assert 'scripts/full_evaluate.py' in hashes
    assert not any('handoff' in name or 'STATUS' in name for name in hashes)
