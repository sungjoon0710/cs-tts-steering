import numpy as np
import pytest
from types import SimpleNamespace


@pytest.mark.parametrize('as_tuple',[False,True])
def test_zero_hook_masks_prefill_handles_cached_tokens_and_cleans_up(as_tuple):
    torch=pytest.importorskip('torch')
    from src.representation_probe import ZeroSteeringProbe
    class Layer(torch.nn.Module):
        def forward(self,h):return (h,'keep') if as_tuple else h
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Layer()])
        def forward(self,input_ids,past_key_values=None,position_ids=None):
            return self.model.layers[0](input_ids.float()[...,None].expand(-1,-1,4))
    lm=LM();prompt=torch.tensor([[1,2,3]])
    with ZeroSteeringProbe(lm,3,layers=(0,)) as probe:
        a=lm(input_ids=prompt)
        b=lm(input_ids=torch.tensor([[128266]]),past_key_values=SimpleNamespace(get_seq_length=lambda:3))
        if as_tuple:assert a[1]==b[1]=='keep';a,b=a[0],b[0]
        assert torch.equal(a,prompt.float()[...,None].expand(-1,-1,4))
        assert torch.equal(b,torch.full((1,1,4),128266.))
    result=probe.result()
    assert result['prefill_calls']==1 and result['audio_positions_checked']==1
    assert result['hooks_removed'] and result['all_outputs_unchanged'] and result['finite_layers']=={'0':True}
    assert not lm._forward_pre_hooks and not lm.model.layers[0]._forward_hooks
    with pytest.raises(ValueError,match='alpha=0'):ZeroSteeringProbe(lm,3,layers=(0,),alpha=.1)


def test_hook_cleanup_on_forward_failure():
    torch=pytest.importorskip('torch')
    from src.representation_probe import ZeroSteeringProbe
    lm=torch.nn.Module();lm.model=torch.nn.Module();lm.model.layers=torch.nn.ModuleList([torch.nn.Identity()])
    with pytest.raises(RuntimeError,match='fixture'):
        with ZeroSteeringProbe(lm,3,layers=(0,)):raise RuntimeError('fixture')
    assert not lm._forward_pre_hooks and not lm.model.layers[0]._forward_hooks


def test_indic_style_preserves_defaults_and_captures_tokens_before_decoding(monkeypatch):
    torch=pytest.importorskip('torch')
    import src.tts as tts
    runner=tts.IndicSpeak.__new__(tts.IndicSpeak);runner.device='cpu';seen=[];captures=[]
    def build(tok,text,voice,style=''):return [1,2] if not style else [1,2,3]
    class LM:
        def generate(self,**kw):
            seen.append(kw)
            codes=[tts.AUDIO_BASE+p*4096+1 for p in range(7)]+[98]
            return torch.cat([kw['input_ids'],torch.tensor([codes])],dim=1)
    runner._build_prompt=build
    runner.tts=SimpleNamespace(lm=LM(),tok=SimpleNamespace(eos_token_id=99),_eos=98,
        snac=SimpleNamespace(quantizer=SimpleNamespace(from_codes=lambda codes:torch.ones(1))),
        vocos=lambda z:torch.ones(1,1,240)*.1)
    a,_=runner('example','Amit',1234,77)
    b,_=runner('example','Amit',1234,77,style='')
    c,_=runner('example','Amit',1234,77,style='ANGER',token_callback=lambda p,g:captures.append((p,g)))
    assert torch.equal(seen[0]['input_ids'],seen[1]['input_ids']) and (a==b).all()
    assert seen[2]['input_ids'].tolist()==[[1,2,3]] and captures[0][0]==(1,2,3)
    assert seen[2]['repetition_penalty']==1.2 and seen[2]['max_new_tokens']==77
    runner.tts.vocos=lambda z:(_ for _ in ()).throw(RuntimeError('decode fixture'))
    with pytest.raises(RuntimeError,match='decode fixture'):
        runner('example','Amit',1234,77,token_callback=lambda p,g:captures.append((p,g)))
    assert len(captures)==2 and captures[-1][1][-1]==98


def test_stage1_manifest_and_frame_masks_are_explicit():
    from scripts.steering_pilot import plan,structure
    rows=[{'utterance_id':str(i),'source_episode':str(i),'split':'construct' if i<6 else 'heldout',
        'Romanized_input':'example input','Devanagari_input':'उदाहरण input'} for i in range(10)]
    jobs=plan(rows)
    assert len(jobs)==len({r['generation_id'] for r in jobs})==80
    for uid in range(10):
        assert len({r['max_new_tokens'] for r in jobs if r['utterance_id']==str(uid)})==1
    tokens=[128266+p*4096 for p in range(7)]+[128258]
    s=structure(tokens,{128258},100)
    assert s['normal_EOS'] and s['usable_complete_code_frames']==1 and not s['token_cap_reached']
    tokens[2]=128266
    assert structure(tokens,{128258},100)['invalid_frames']==1
    assert structure([128266],{128258},1)['incomplete_trailing_tokens']==1


def test_prompt_style_block_cannot_move_after_the_transcript():
    from scripts.steering_pilot import prompt_checks
    class Tok:
        def convert_tokens_to_ids(self,s):return {'<|style>':10,'<style|>':11,'<|start_of_speech|>':12,'<speaker|>':13}[s]
        def encode(self,s,**kw):return [20] if s=='\n' else [21]
    tok=Tok();neutral=[1,13,20,2,12]
    build=lambda *a,**kw:neutral
    correct=[1,13,20,10,21,11,20,2,12]
    moved=[1,13,20,2,10,21,11,20,12]
    assert all(prompt_checks(tok,'text','Amit','ANGER',correct,build).values())
    assert not prompt_checks(tok,'text','Amit','ANGER',moved,build)['style_placement_valid']


def test_collector_preserves_phases_positions_cache_and_unmodified_histories():
    torch=pytest.importorskip('torch')
    from src.representation_probe import code_phases,collect_phases
    valid=[128266+p*4096+1 for p in range(7)];bad=valid.copy();bad[2]=128266
    tokens=valid+bad+valid+[128266,128258]
    assert code_phases(tokens)==list(range(7))+[-1]*7+list(range(7))+[-1,-1]
    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1));self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Identity()]);self.hist=[]
        def forward(self,input_ids,attention_mask,position_ids,past_key_values,use_cache,logits_to_keep):
            seen=0 if past_key_values is None else past_key_values.get_seq_length()
            assert position_ids.tolist()==[list(range(seen,seen+input_ids.shape[1]))]
            assert attention_mask.shape==(1,seen+input_ids.shape[1]) and use_cache and logits_to_keep==1
            self.hist+=input_ids[0].tolist();h=position_ids.float()[...,None].expand(-1,-1,3)
            self.model.layers[0](h);n=seen+input_ids.shape[1]
            return SimpleNamespace(past_key_values=SimpleNamespace(get_seq_length=lambda:n))
    lm=LM().eval();prompt=[12,128257]
    a,m=collect_phases(lm,prompt,tokens,layers=(0,),chunk_size=5)
    assert lm.hist==prompt+tokens and not lm.model.layers[0]._forward_hooks
    assert (a['counts']==2).all() and np.allclose(a['means'][0,:,0],np.arange(7)+9)
    lm.hist=[];b,_=collect_phases(lm,prompt,tokens,layers=(0,),chunk_size=100)
    assert np.array_equal(a['means'],b['means']) and m['processed_tokens']==len(prompt+tokens)
    with pytest.raises(ValueError,match='empty-style'):collect_phases(lm,[156940,128257],tokens,layers=(0,))
    def fail(*a,**kw):raise RuntimeError('forward fixture')
    lm.forward=fail
    with pytest.raises(RuntimeError,match='fixture'):collect_phases(lm,prompt,tokens,layers=(0,))
    assert not lm.model.layers[0]._forward_hooks
