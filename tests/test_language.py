import pytest
from src.evaluation.language import aligned_labels,corpus_lexicon,tag_text,difficulty,language_errors

def test_punctuation_alignment_is_not_whitespace_length_matching():
    assert aligned_labels('Hello!',"['EN','OTHER']")==['EN']
    assert aligned_labels('hello world',"['EN']") is None
    assert aligned_labels('hello',"['/EN']") is None
    assert aligned_labels('well-known',"['EN','OTHER','EN']") is None

def test_corpus_support_is_not_doubled_for_unchanged_paired_words():
    rows=[{'name':'x','Transcript':'hello namaste','LID_tags':"['EN','HI']"}]
    paired=[{'name':'x','Transcript':'hello \u0928\u092e\u0938\u094d\u0924\u0947'}]
    lex,stats=corpus_lexicon(rows,paired)
    assert lex['hello']['EN']==1
    assert lex['namaste']['HI']==1
    assert lex['\u0928\u092e\u0938\u094d\u0924\u0947']['HI']==1
    assert corpus_lexicon(rows,paired,exclude_texts=['hello namaste'])[0]=={}

def test_unknown_is_not_english_and_switches_are_only_observed_bounds():
    tags=tag_text('a b c',{'a':{'HI':10},'c':{'EN':10}})
    d=difficulty(tags)
    assert [t['label'] for t in tags]==['HI','UNKNOWN','EN']
    assert not d['fully_labeled'] and d['english_proportion'] is None
    assert d['observed_switch_count_lower_bound']==0
    assert d['english_proportion_bounds']==pytest.approx([1/3,2/3])
    assert tag_text('latin',{})[0]['label']=='UNKNOWN'

def test_language_poi_uses_complete_alignment_and_trailing_insertions():
    tagged=[{'token':'namaste','label':'HI'},{'token':'hello','label':'EN'}]
    r=language_errors('namaste hello','namaste hello extra',tagged)
    assert r['english_poi']['pier']==1 and r['hindi_poi']['pier']==0
    assert r['embedded_pier'] is None
    with pytest.raises(ValueError):language_errors('different reference','different',tagged)
