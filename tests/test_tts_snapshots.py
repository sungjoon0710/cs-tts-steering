"""Exercise upstream FT/PT selection and explicit snapshots without model files."""
import sys
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('model_name,pinned,voice',[
    ('OrpheusHindi',False,'none'),('OrpheusHindiPretrained',False,'none'),
    ('OrpheusHindi',True,'ऋतिका'),('OrpheusHindiPretrained',True,'none')])
def test_orpheus_snapshot_selection_preserves_pretrained_and_voice_none(monkeypatch,tmp_path,model_name,pinned,voice):
    torch=pytest.importorskip('torch')
    import src.tts as tts
    seen={}
    class Model:
        def to(self,*a):return self
        def eval(self):return self
        def generate(self,**kwargs):
            seen['limit']=kwargs['max_new_tokens']
            return torch.cat([kwargs['input_ids'],torch.tensor([[tts.OrpheusHindi.START_SPEECH]+[tts.AUDIO_BASE]*7])],dim=1)
        def decode(self,codes):return torch.zeros(1,1,240)
    class Tokenizer:
        def __call__(self,text):seen['text']=text;return SimpleNamespace(input_ids=[1,2])
    def load_tokenizer(path):seen['snapshot']=path;return Tokenizer()
    def load_codec(path):seen['codec']=path;return Model()
    monkeypatch.setitem(sys.modules,'snac',SimpleNamespace(SNAC=SimpleNamespace(from_pretrained=load_codec)))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a,**k:Model())))
    monkeypatch.setattr(tts,'cached_snapshot',lambda repo:tmp_path/repo.split('/')[-1])
    monkeypatch.setattr(tts,'snac_codes',lambda *args:[])
    cls=getattr(tts,model_name);kwargs={'snapshot':tmp_path/'pinned','codec_snapshot':tmp_path/'codec'} if pinned else {}
    model=cls(device='cpu',**kwargs)
    assert seen['snapshot']==(tmp_path/'pinned' if pinned else tmp_path/cls.repo.split('/')[-1])
    assert seen['codec']==(str(tmp_path/'codec') if pinned else 'hubertsiuzdak/snac_24khz')
    wav,_=model('example text',voice,1234,max_new_tokens=321)
    assert seen['text']==('example text' if voice=='none' else voice+': example text')
    assert seen['limit']==321 and len(wav)==240
