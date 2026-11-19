"""Real PyTorch hook/causality tests, with no checkpoint or GPU dependency."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.causal_patch import Patch,replay,intervention_vector,score_patch,norm_matched_add


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=torch.nn.Module();self.self_attn.o_proj=torch.nn.Linear(8,8);self.mlp=torch.nn.Linear(8,8)
    def forward(self,x):
        denominator=torch.arange(1,x.shape[1]+1,device=x.device,dtype=x.dtype)[None,:,None]
        x=x+self.self_attn.o_proj(x.cumsum(1)/denominator)
        return x+self.mlp(torch.tanh(x))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.embed_tokens=torch.nn.Embedding(156960,8)
        self.model.layers=torch.nn.ModuleList([Block() for _ in range(4)]);self.model.norm=torch.nn.LayerNorm(8)
        self.head=torch.nn.Linear(8,156960);self.config=SimpleNamespace(use_cache=True);self.fail=False
    def forward(self,input_ids,attention_mask,use_cache,logits_to_keep):
        x=self.model.embed_tokens(input_ids)
        for block in self.model.layers:x=block(x)
        if self.fail:raise RuntimeError('injected failure')
        return SimpleNamespace(logits=self.head(self.model.norm(x)[:,logits_to_keep]),past_key_values=None)


@pytest.fixture
def setup():
    torch.manual_seed(4);lm=TinyLM().eval();targets=[128266+(j%7)*4096 for j in range(21)]
    rows=[0,1,6,7,8,14,20];sites=[(1,'attn'),(1,'resid'),(3,'mlp'),(3,'resid')]
    a=replay(lm,[4,128257],targets,rows,sites);b=replay(lm,[5,128257],targets,rows,sites)
    return lm,targets,rows,sites,a,b


def test_identity_zero_native_and_hook_cleanup(setup):
    lm,t,rows,sites,a,b=setup;k=rows.index(7);vector=a['states'][(1,'attn')][k]
    for donor,alpha in [(vector,1),(b['states'][(1,'attn')][k],0)]:
        out=replay(lm,[4,128257],t,rows,patch=Patch(1,'attn',7,vector,donor,alpha=alpha,dtype=a['dtype']))
        assert torch.equal(out['logits'],a['logits']) and out['checks']['actual_changed_states']==0
    assert lm.config.use_cache and all(not m._forward_hooks for m in lm.modules())


def test_last_residual_reproduces_donor_and_earlier_predictions_unchanged(setup):
    lm,t,rows,sites,a,b=setup;k=rows.index(7)
    out=replay(lm,[4,128257],t,rows,patch=Patch(3,'resid',7,a['states'][(3,'resid')][k],b['states'][(3,'resid')][k]))
    assert torch.equal(out['logits'][k],b['logits'][k])
    assert torch.equal(out['logits'][:k],a['logits'][:k])
    assert not torch.equal(out['logits'][k],a['logits'][k])
    assert out['checks']['absolute_position']==8 and out['checks']['processed_phase']==6 and out['checks']['predicted_phase']==0


def test_errors_do_not_leave_hooks_or_cache_changes(setup):
    lm,t,rows,sites,a,b=setup;k=rows.index(7)
    with pytest.raises(ValueError,match='non-anchor'):replay(lm,[4,128257],t,rows,patch=Patch(1,'attn',0,np.zeros(8),np.ones(8)))
    with pytest.raises(ValueError,match='matched context'):replay(lm,[4,128257],t,rows,patch=Patch(1,'attn',7,np.zeros(8),np.ones(8)))
    lm.fail=True
    with pytest.raises(RuntimeError,match='injected'):replay(lm,[4,128257],t,rows,sites)
    assert lm.config.use_cache and all(not m._forward_hooks for m in lm.modules())
    lm.fail=False;handle=lm.model.layers[1].register_forward_hook(lambda *a:None)
    try:
        with pytest.raises(ValueError,match='existing hooks'):replay(lm,[4,128257],t,rows)
        assert len(lm.model.layers[1]._forward_hooks)==1
    finally:handle.remove()


def test_norm_matching_reverse_and_projection_are_distinct():
    a=np.arange(8,dtype=float);b=a+np.linspace(-2,2,8)
    random=intervention_vector(a,b,'random',seed=8)
    assert np.linalg.norm(random)==pytest.approx(np.linalg.norm(b-a),rel=1e-6)
    np.testing.assert_array_equal(random,intervention_vector(a,b,'random',seed=8))
    for variant in ['shared','specific']:
        v=intervention_vector(a,b,variant,direction=b-a)
        assert np.linalg.norm(v)==pytest.approx(np.linalg.norm(b-a),rel=1e-6)


def test_recovery_floor_and_both_stopping_probabilities():
    torch.manual_seed(5);base={k:torch.randn(156960).log_softmax(-1) for k in ['news','anger','happy','sad']}
    r=score_patch(base,base['anger'],'news','anger',0,128266,1e-6)
    assert r['code_normalized_recovery']==pytest.approx(1,abs=1e-6)
    assert r['code_selective_absolute_recovery']>0
    assert r['total_stop_probability']==pytest.approx(r['eos_128001_probability']+r['eos_128258_probability'])
    same={k:base['news'] for k in base};r=score_patch(same,same['news'],'news','anger',0,128266,1e-6)
    assert r['code_normalized_recovery'] is None and r['code_own_absolute_recovery']==0


def test_bf16_unrealizable_norm_is_rejected_without_relaxing_tolerance():
    original=torch.tensor([256.],dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='within 1%'):norm_matched_add(original,torch.tensor([.6]))
    assert norm_matched_add(original,torch.tensor([2.])).float().item()==258.
