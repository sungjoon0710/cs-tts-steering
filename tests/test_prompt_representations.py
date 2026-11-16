import numpy as np
import pytest
import torch
from src.representation_probe import collect_anchor, collect_prefix


@pytest.fixture
def tiny_lm():
    transformers = pytest.importorskip('transformers', exc_type=ImportError)
    torch.manual_seed(1)
    config = transformers.LlamaConfig(vocab_size=156960, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    return transformers.LlamaForCausalLM(config).eval()


def test_anchor_matches_direct_post_block_states_and_exact_replay(tiny_lm):
    model = tiny_lm
    prompt = [1, 2, 128257]
    expected, handles = {}, []
    for i, layer in enumerate(model.model.layers):
        def capture(module, args, out, i=i):
            h = out[0] if isinstance(out, tuple) else out
            expected[i] = h[0, -1].detach().float().numpy().copy()
        handles.append(layer.register_forward_hook(capture))
    with torch.inference_mode():
        model(input_ids=torch.tensor([prompt]), use_cache=False)
    for h in handles:
        h.remove()
    states, meta = collect_anchor(model, prompt)
    np.testing.assert_array_equal(states['residual'], np.stack([expected[i] for i in range(2)]))
    repeated, _ = collect_anchor(model, prompt)
    np.testing.assert_array_equal(states['residual'], repeated['residual'])
    assert states['residual'].shape == (2, 8) and states['embedding'].shape == (8,)
    assert meta['fresh_cache'] and meta['hooks_removed']
    assert model.config.use_cache and not model._forward_pre_hooks


def test_prefix_checks_complete_code_phases_and_hook_conflicts(tiny_lm):
    prompt = [1, 128257]
    prefix = [128266 + i * 4096 + 1 for i in range(7)]
    arrays, meta = collect_prefix(tiny_lm, prompt, prefix)
    assert meta['frames'] == 1 and arrays['resid_mean'].shape == (2, 8)
    with pytest.raises(ValueError, match='complete'):
        collect_prefix(tiny_lm, prompt, prefix + [128258])
    handle = tiny_lm.model.layers[0].register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match='existing hooks'):
        collect_anchor(tiny_lm, prompt)
    handle.remove()
    assert not tiny_lm._forward_pre_hooks
