"""Unit tests for the CAA pilot code (no model weights needed).

Real-audio round trips through both models' own decoder functions run on kedi:
`python scripts/caa_extract.py roundtrip --model {indic-speak,orpheus-hi}`.
Run here with `python -m pytest tests/test_caa.py` or `python tests/test_caa.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.caa_stats import auc, ccc, dtw_path, friedman, make_folds, projection_cv, wilcoxon
from src.snac_encode import codes_to_tokens, tokens_to_codes
from src.tts_settings import AUDIO_BASE


def _codes(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4096, n), rng.integers(0, 4096, 2 * n), rng.integers(0, 4096, 4 * n)


def test_roundtrip_orpheus_no_dedup_matches_snac_codes():
    import torch
    from src.tts import snac_codes
    c0, c1, c2 = _codes(50)
    c0[10] = c0[9]  # duplicate c0 must be kept for Orpheus
    toks = codes_to_tokens(c0, c1, c2, AUDIO_BASE, dedup=False)
    assert len(toks) == 7 * 50
    for x, y in zip(tokens_to_codes(toks + [128258, 5], AUDIO_BASE), (c0, c1, c2)):
        assert np.array_equal(x, y)
    for x, y in zip(snac_codes(toks, "cpu"), (c0, c1, c2)):  # the decoder function used in src/tts.py
        assert np.array_equal(x[0].numpy(), y)
    assert isinstance(torch.tensor(0), torch.Tensor)


def test_roundtrip_indic_dedup_drops_consecutive_repeated_c0_only():
    base = 128266  # read from tokenizer at runtime (<|snac_0|>); same value for this checkpoint
    c0, c1, c2 = _codes(20, seed=1)
    c0[5] = c0[4]
    c0[12] = c0[10]  # non-consecutive repeat is kept
    c0[11] = 7 if c0[10] != 7 else 8
    toks = codes_to_tokens(c0, c1, c2, base, dedup=True)
    keep = np.ones(20, bool)
    keep[5] = False
    assert len(toks) == 7 * keep.sum()
    r0, r1, r2 = tokens_to_codes(toks, base)
    assert np.array_equal(r0, c0[keep])
    assert np.array_equal(r1, c1.reshape(-1, 2)[keep].reshape(-1))
    assert np.array_equal(r2, c2.reshape(-1, 4)[keep].reshape(-1))


def test_frame_layout_and_offsets():
    toks = codes_to_tokens([1], [2, 3], [4, 5, 6, 7], 100, dedup=False)
    assert toks == [100 + 1, 100 + 4096 + 2, 100 + 2 * 4096 + 4, 100 + 3 * 4096 + 5,
                    100 + 4 * 4096 + 3, 100 + 5 * 4096 + 6, 100 + 6 * 4096 + 7]


def test_auc_and_wilcoxon_and_friedman():
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1], [1]) == 0.5
    # exact two-sided Wilcoxon, n=5 all positive: p = 2/32
    r = wilcoxon([1, 2, 3, 4, 5])
    assert r["W"] == 15 and abs(r["p"] - 0.0625) < 1e-9
    f = friedman(np.array([[1, 2, 3]] * 6))
    assert abs(f["chi2"] - 12.0) < 1e-9 and f["p"] < 0.01


def test_projection_cv_separable_and_folds_grouped():
    rng = np.random.default_rng(0)
    n = 40
    groups = np.repeat(np.arange(n), 2)
    y = np.tile([1, 0], n)
    X = rng.normal(size=(2 * n, 16)) + y[:, None] * 2.0
    folds = make_folds(groups, 5, 0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(n))
    a, win, d = projection_cv(X, y, groups, folds)
    assert a > 0.95 and win > 0.9 and d > 2


def test_dtw_identity_and_ccc():
    A = np.arange(10, dtype=float)[:, None]
    assert dtw_path(A, A) == [(i, i) for i in range(10)]
    x = np.linspace(0, 1, 50)
    assert abs(ccc(x, x) - 1) < 1e-9


def test_steerer_rescales_and_leaves_unsteered_rows():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from src.steering import Steerer
    cfg = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2)
    torch.manual_seed(0)
    lm = LlamaForCausalLM(cfg).eval()
    holder = type("M", (), {"layers": lm.model.layers})()
    st = Steerer(holder)
    x = torch.randint(0, 50, (2, 5))
    seen = {}
    h = lm.model.layers[1].register_forward_hook(lambda m, i, o: seen.__setitem__("o", (o[0] if isinstance(o, tuple) else o).clone()))
    with torch.no_grad():
        lm(input_ids=x)
        base = seen["o"]
        delta = torch.zeros(2, 16)
        delta[0] = 5.0
        st.set({(0, "resid"): delta})
        lm(input_ids=x)
        steered = seen["o"]
        st.clear()
        lm(input_ids=x)
    h.remove()
    assert torch.allclose(seen["o"], base)  # hooks removed
    assert torch.allclose(steered[1], base[1], atol=1e-5)  # row without a vector unchanged
    assert not torch.allclose(steered[0], base[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
