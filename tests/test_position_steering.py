from types import SimpleNamespace
import pytest
import torch
from src.position_steering import AnchorIntervention


class Tiny(torch.nn.Module):
    def __init__(self, as_tuple=False):
        super().__init__()
        class Block(torch.nn.Module):
            def forward(self, h):
                return (h, 'keep') if as_tuple else h
        self.layers = torch.nn.ModuleList([Block()])
        self.model = SimpleNamespace(layers=self.layers)

    def forward(self, input_ids, past_key_values=None):
        return self.layers[0](torch.stack([input_ids.float() + i for i in range(4)], dim=-1))


@pytest.mark.parametrize('as_tuple', [False, True])
def test_exact_zero_cleanup_and_both_apis(as_tuple):
    from src.steering import SpeechLM, Steerer, extract, generate_batch
    from src.representation_probe import ZeroSteeringProbe, code_phases, collect_phases
    assert all(callable(x) for x in (SpeechLM, Steerer, extract, generate_batch,
                                     ZeroSteeringProbe, code_phases, collect_phases))
    model = Tiny(as_tuple)
    ids = torch.tensor([[8, 128257]])
    baseline = model(input_ids=ids)
    with AnchorIntervention(model, ids[0].tolist(), 0, torch.ones(4), 0) as hook:
        actual = model(input_ids=ids)
    if as_tuple:
        assert actual[1] == baseline[1] == 'keep'
        actual, baseline = actual[0], baseline[0]
    assert torch.equal(actual, baseline)
    assert hook.selected == 1 and hook.result()['hooks_removed']


def test_nonzero_only_changes_final_prefill_and_preserves_norm():
    model = Tiny()
    ids = torch.tensor([[8, 128257]])
    baseline = model(input_ids=ids)
    with AnchorIntervention(model, ids[0].tolist(), 0, torch.tensor([1., -1, 1, -1]), .02) as hook:
        changed = model(input_ids=ids)
        assert torch.equal(changed[:, :-1], baseline[:, :-1])
        assert not torch.equal(changed, baseline)
        torch.testing.assert_close(changed[:, -1].norm(), baseline[:, -1].norm())
        token = torch.tensor([[128266]])
        cached = model(input_ids=token, past_key_values=SimpleNamespace(get_seq_length=lambda: 2))
        expected = torch.stack([token.float() + i for i in range(4)], dim=-1)
        assert torch.equal(cached, expected)
    assert hook.selected == 1


def test_nonfresh_cache_and_wrong_direction_cleanup():
    model = Tiny()
    ids = torch.tensor([[8, 128257]])
    with pytest.raises(ValueError, match='fresh cache'):
        with AnchorIntervention(model, ids[0].tolist(), 0, torch.ones(4), .02):
            model(input_ids=ids, past_key_values=SimpleNamespace(get_seq_length=lambda: 2))
    assert not model._forward_pre_hooks and not model.layers[0]._forward_hooks
    with pytest.raises(ValueError, match='shape'):
        with AnchorIntervention(model, ids[0].tolist(), 0, torch.ones(3), .02):
            model(input_ids=ids)
    assert not model._forward_pre_hooks and not model.layers[0]._forward_hooks
