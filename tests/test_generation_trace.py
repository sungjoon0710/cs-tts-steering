import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.generation_trace import (TraceStore, prediction_differences, prediction_positions,
                                  state_differences, trace_context)


def codes(frames=1):
    return [128266 + (j % 7) * 4096 + j // 7 for j in range(frames * 7)]


def test_phase_is_predicted_not_processed_and_eos_is_excluded():
    rows = prediction_positions([1, 128257], codes(2))
    assert rows[0] == dict(j=0, absolute_position=1, processed_phase=-1,
                           predicted_phase=0, frame=0, input_token=128257, target_token=128266)
    assert rows[7]['processed_phase'] == 6 and rows[7]['predicted_phase'] == 0
    assert rows[7]['frame'] == 1 and rows[7]['absolute_position'] == 8
    for bad in [codes()[:-1], codes() + [128258], codes()[:1] * 7]:
        with pytest.raises(ValueError, match='complete valid'):
            prediction_positions([128257], bad)


def test_distributions_identity_symmetry_target_likelihood_and_phase_conditioning():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(7, 15)).astype('float32'), rng.normal(size=(7, 15)).astype('float32')
    targets = [j * 2 for j in range(7)]
    kw = dict(audio_base=0, codebook_size=2, eos=14)
    same = prediction_differences(a, a, targets, **kw)
    assert all(abs(r['jsd_nats']) < 1e-7 and r['delta_target_logp'] == 0 for r in same)
    ab, ba = prediction_differences(a, b, targets, **kw), prediction_differences(b, a, targets, **kw)
    for x, y in zip(ab, ba):
        assert 0 <= x['jsd_nats'] <= np.log(2)
        assert x['jsd_nats'] == pytest.approx(y['jsd_nats'], abs=1e-7)
        assert x['delta_target_logp'] == pytest.approx(-y['delta_target_logp'])
    # Alter mass outside the first phase: full JSD changes, conditional JSD=0.
    b = a.copy()
    b[0, 2:] += 5
    r = prediction_differences(a, b, targets, **kw)[0]
    assert r['jsd_nats'] > 0.01 and r['phase_jsd_nats'] < 1e-7
    assert r['delta_target_logp'] < 0


def test_zero_states_are_undefined_not_identical_directions():
    cos, l2 = state_differences(np.zeros((2, 3)), np.ones((2, 3)))
    assert np.isnan(cos).all() and np.isnan(l2).all()
    cos, l2 = state_differences(np.ones((2, 3)), np.ones((2, 3)) * 2)
    np.testing.assert_allclose(cos, 0, atol=1e-6)
    np.testing.assert_allclose(l2, 1)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(8, 8)
        self.mlp = torch.nn.Linear(8, 8)

    def forward(self, x):
        x = x + self.self_attn.o_proj(x)
        return x + self.mlp(x)


class TinyLM(torch.nn.Module):
    """No Transformers dependency; test actual registered PyTorch hook behavior."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(156960, 8)
        self.model.layers = torch.nn.ModuleList([Block(), Block()])
        self.model.norm = torch.nn.LayerNorm(8)
        self.head = torch.nn.Linear(8, 156960)
        self.config = SimpleNamespace(use_cache=True)
        self.fail = False

    @property
    def dtype(self):
        return self.head.weight.dtype

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        x = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            x = block(x)
        if self.fail:
            raise RuntimeError('injected failure')
        return SimpleNamespace(logits=self.head(self.model.norm(x)[:, -logits_to_keep:]), past_key_values=None)


def test_readonly_hooks_reconstruct_residuals_match_logits_and_cleanup_on_error():
    torch.manual_seed(3)
    lm = TinyLM().eval()
    result = trace_context(lm, [1, 128257], codes(), check_no_hook=True)
    assert result['states']['resid'].shape == (2, 7, 8)
    assert result['checks']['no_hook_logits_equal'] and result['checks']['residual_reconstruction_exact']
    assert lm.config.use_cache
    lm.fail = True
    with pytest.raises(RuntimeError, match='injected'):
        trace_context(lm, [128257], codes())
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in lm.modules())
    assert lm.config.use_cache


def test_preexisting_hooks_are_not_silently_removed():
    lm = TinyLM().eval()
    handle = lm.model.layers[0].register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match='existing hooks'):
        trace_context(lm, [128257], codes())
    assert len(lm.model.layers[0]._forward_hooks) == 1
    handle.remove()


def test_sqlite_records_exact_positions_and_hashed_external_artifacts(tmp_path):
    store = TraceStore(tmp_path / 'run', {'version': 1})
    store.context('d', 'default', 'default', prediction_positions([128257], codes()), {})
    np.savez_compressed(store.directory / 'states.npz', x=np.ones((2, 3)))
    store.artifact('states.npz', 'states', {'positions': [0, 1]})
    assert store.db.execute('SELECT count(*) FROM position').fetchone()[0] == 7
    assert store.db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert len(store.db.execute('SELECT sha256 FROM artifact').fetchone()[0]) == 64
    with pytest.raises(ValueError, match='outside'):
        store.artifact('../outside', 'unsafe')
    with pytest.raises(sqlite3.IntegrityError):
        store.comparison('bad', 'd', 'missing', {})
    store.close()
    with pytest.raises(FileExistsError):
        TraceStore(tmp_path / 'run', {})
