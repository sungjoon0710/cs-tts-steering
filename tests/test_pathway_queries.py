"""Small CPU checks of the row selection and scoped native-attention dispatch."""
from pathlib import Path

import torch


class Dispatch:
    def get_interface(self, implementation, default):
        return default


def test_query_gate_matches_native_recipient_and_donor_rows(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts/m3"))
    from pathway_experiment import sdpa_dispatch

    torch.manual_seed(19)
    q=torch.randn(1,24,5,128,dtype=torch.bfloat16)
    k=torch.randn(1,8,5,128,dtype=torch.bfloat16)
    v=torch.randn_like(k)
    donor=(torch.randn_like(k[:,:,1]),torch.randn_like(v[:,:,1]))
    after_k,after_v=k.clone(),v.clone()
    after_k[:,:,1],after_v[:,:,1]=donor
    modifier=object()
    native_calls=[]

    def native(mod,q,k,v,mask,**kw):
        native_calls.append((q.shape,k.shape,kw))
        result=torch.nn.functional.scaled_dot_product_attention(
            q,k.repeat_interleave(3,dim=1),v.repeat_interleave(3,dim=1),is_causal=True)
        return result.transpose(1,2).contiguous(),None

    registry=Dispatch()
    original=registry.get_interface
    a,_=native(modifier,q,k,v,None)
    b,_=native(modifier,q,after_k,after_v,None)
    assert not torch.equal(a[:,3],b[:,3])
    seen=[]
    with sdpa_dispatch(registry,"sdpa",{id(modifier):0},(),None,(1,{0:donor}),[3],seen):
        interface=registry.get_interface("sdpa",native)
        mixed,_=interface(modifier,q,k,v,None,dropout=0.)
    assert registry.get_interface==original and seen==[0]
    assert torch.equal(mixed[:,[0,1,2,4]],a[:,[0,1,2,4]])
    assert torch.equal(mixed[:,3],b[:,3])
    assert torch.equal(a[:,0],b[:,0])
