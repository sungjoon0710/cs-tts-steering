"""CAA steering for the two Llama-based speech LMs (Indic-Speak, Orpheus Hindi FT).

- `SpeechLM`: loads a pinned model + codec, builds generation prompts and teacher-forced
  sequences (prompt + audio tokens + end-of-speech), decodes audio tokens to 24 kHz audio.
- `extract`: one teacher-forced forward pass with hooks on every layer's residual output
  ("resid") and attention o_proj output ("attn"); pools over audio-token positions
  (last token and mean) and returns per-token NLL of the audio tokens.
- `Steerer`: forward hooks that add per-row alpha*v at chosen (layer, site) and rescale each
  position back to its original L2 norm (CoCoEmo-style). Rows without a vector are untouched.
- `generate_batch`: left-padded batched sampling with per-row steering, per-row token budget.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.snac_encode import codes_to_tokens, tokens_to_codes

SITES = ("resid", "attn")
# Left-padding token for batched generation. HF's repetition penalty also scores the pad ids, so the pad must
# never be a stop token (128258 end-of-speech / 128001 end-of-text would be penalized in padded rows only).
# 128263 is <|pad|> in Indic-Speak and Orpheus's training pad (<custom_token_7>); neither model emits it.
PAD_ID = 128263
POOLS = ("last", "mean")


class SpeechLM:
    """Model-specific glue. kind in {'indic-speak', 'orpheus-hi'}."""

    def __init__(self, kind: str, snapshot: str | Path, codec_snapshot: str | Path, device="cuda"):
        self.kind, self.device = kind, device
        if kind == "indic-speak":
            from src.tts import IndicSpeak
            w = IndicSpeak(device=device, snapshot=snapshot)  # vocos weights checked weights_only=True
            t = w.tts
            self.lm, self.tok, self._t = t.lm, t.tok, t
            self._build_prompt = w._build_prompt
            self.base = self.tok.convert_tokens_to_ids("<|snac_0|>")
            self.end_speech = t._eos
            self.eos_ids = [t._eos, self.tok.eos_token_id]
            self.pad_id = PAD_ID
            self.dedup = True
            self.decoding = dict(w.decoding)
            self.snac = t.snac
        elif kind == "orpheus-hi":
            from src.tts import OrpheusHindi
            from src.tts_settings import AUDIO_BASE
            w = OrpheusHindi(device=device, snapshot=snapshot, codec_snapshot=codec_snapshot)
            self.lm, self.tok, self.snac = w.lm, w.tok, w.snac
            self._w = w
            base = self.tok.convert_tokens_to_ids("<custom_token_10>")
            if base != AUDIO_BASE:
                raise ValueError(f"Orpheus audio base {base} != {AUDIO_BASE}")
            self.base = base
            self.end_speech = w.END_SPEECH
            self.eos_ids = [w.END_SPEECH]
            self.pad_id = PAD_ID
            self.dedup = False
            self.decoding = dict(w.decoding)
        else:
            raise ValueError(kind)
        self.lm.eval()
        self.n_layers = len(self.layers)

    @property
    def layers(self):
        return self.lm.model.layers

    # ---- sequences -------------------------------------------------------------------
    def prompt(self, text: str, voice: str) -> list[int]:
        if self.kind == "indic-speak":
            return list(self._build_prompt(self.tok, text, voice))  # ends with <|start_of_speech|>
        w = self._w
        return [w.START_HUMAN] + self.tok(f"{voice}: {text}").input_ids + [w.END_TEXT, w.END_HUMAN]

    def speech_prefix(self) -> list[int]:
        """Tokens the model itself emits between the prompt and the first audio token."""
        if self.kind == "indic-speak":
            return []
        return [128261, self._w.START_SPEECH]  # <start_of_ai>, <start_of_speech> (checked on real outputs)

    def tf_sequence(self, text: str, voice: str, audio_tokens: list[int]):
        """Teacher-forced ids and the [start, end) slice of the audio-token positions."""
        head = self.prompt(text, voice) + self.speech_prefix()
        ids = head + list(audio_tokens) + [self.end_speech]
        return ids, (len(head), len(head) + len(audio_tokens))

    def audio_tokens(self, new_ids: list[int]) -> list[int]:
        if self.kind == "orpheus-hi" and self._w.START_SPEECH in new_ids:
            new_ids = new_ids[len(new_ids) - new_ids[::-1].index(self._w.START_SPEECH):]
        hi = self.base + 7 * 4096
        out = []
        for t in new_ids:
            if not self.base <= t < hi:
                break
            out.append(t)
        return out

    @torch.no_grad()
    def decode(self, audio_tokens: list[int]) -> np.ndarray:
        c0, c1, c2 = tokens_to_codes(audio_tokens, self.base)
        if len(c0) == 0:
            raise ValueError("no complete SNAC frame")
        codes = [torch.from_numpy(x).long().unsqueeze(0).to(self.device) for x in (c0, c1, c2)]
        if self.kind == "indic-speak":
            z_q = self.snac.quantizer.from_codes(codes)
            wav = self._t.vocos(z_q.float())
        else:
            wav = self.snac.decode(codes)
        return wav[0, 0].clamp(-1, 1).float().cpu().numpy()

    def encode_codes(self, c0, c1, c2) -> list[int]:
        return codes_to_tokens(c0, c1, c2, self.base, self.dedup)


def _out_tensor(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def extract(model: SpeechLM, ids: list[int], audio_slice: tuple[int, int]):
    """Returns dict with pooled activations {(site,pool): (L, d) float32 numpy}, norms and NLL."""
    a0, a1 = audio_slice
    if a1 - a0 < 1:
        raise ValueError("empty audio span")
    store = {s: [None] * model.n_layers for s in SITES}
    handles = []
    for i, layer in enumerate(model.layers):
        def hr(mod, inp, out, i=i):
            store["resid"][i] = _out_tensor(out)[0, a0:a1].float()

        def ha(mod, inp, out, i=i):
            store["attn"][i] = _out_tensor(out)[0, a0:a1].float()
        handles.append(layer.register_forward_hook(hr))
        handles.append(layer.self_attn.o_proj.register_forward_hook(ha))
    try:
        x = torch.tensor([ids], device=model.device)
        logits = model.lm(input_ids=x).logits[0]
    finally:
        for h in handles:
            h.remove()
    # NLL of audio token t predicted at position t-1
    tgt = x[0, a0:a1]
    nll = []
    for s in range(a0 - 1, a1 - 1, 512):
        e = min(s + 512, a1 - 1)
        lp = torch.log_softmax(logits[s:e].float(), dim=-1)
        nll.append(-lp.gather(1, tgt[s - a0 + 1:e - a0 + 1, None])[:, 0])
    nll = torch.cat(nll).cpu().numpy()
    pooled, norms = {}, {}
    for s in SITES:
        h = torch.stack(store[s])  # (L, T, d)
        pooled[(s, "last")] = h[:, -1].cpu().numpy()
        pooled[(s, "mean")] = h.mean(1).cpu().numpy()
        norms[s] = h.norm(dim=-1).mean(1).cpu().numpy()  # mean per-token norm per layer
    return dict(pooled=pooled, token_norm=norms, nll=nll)


class Steerer:
    """Per-row additive steering with norm rescaling at (layer, site) hooks.

    set(deltas): deltas maps (layer, site) -> tensor (B, d) of alpha*v per row (zeros = no steering).
    mode 'all' steers every position (prompt and generated); 'generated' skips the prefill pass.
    """

    def __init__(self, model: SpeechLM, mode: str = "all"):
        self.model, self.mode, self.deltas, self.handles = model, mode, {}, []

    def set(self, deltas: dict):
        self.clear()
        self.deltas = deltas
        for (layer, site), d in deltas.items():
            mod = self.model.layers[layer] if site == "resid" else self.model.layers[layer].self_attn.o_proj
            self.handles.append(mod.register_forward_hook(self._hook(d)))

    def _hook(self, delta):
        def hook(mod, inp, out):
            h = _out_tensor(out)
            if self.mode == "generated" and h.shape[1] > 1:
                return None
            hf = h.float()
            h2 = hf + delta[:, None, :].to(hf)
            h2 = h2 * (hf.norm(dim=-1, keepdim=True) / h2.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            h2 = h2.to(h.dtype)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles, self.deltas = [], {}


@torch.no_grad()
def generate_batch(model: SpeechLM, rows: list[dict], seed: int, steerer: Steerer | None = None,
                   deltas: dict | None = None):
    """rows: dicts with text, voice, budget. Returns list of (audio_tokens, info)."""
    prompts = [model.prompt(r["text"], r["voice"]) for r in rows]
    P = max(map(len, prompts))
    ids = torch.full((len(rows), P), model.pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), P), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.tensor(p)
        mask[i, P - len(p):] = 1
    budget = max(r["budget"] for r in rows)
    if steerer is not None:
        steerer.set(deltas or {})
    torch.manual_seed(seed)
    t0 = time.time()
    try:
        out = model.lm.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                                max_new_tokens=budget, eos_token_id=model.eos_ids, pad_token_id=model.pad_id,
                                do_sample=True, **model.decoding)
    finally:
        if steerer is not None:
            steerer.clear()
    gen_s = time.time() - t0
    results = []
    for i, r in enumerate(rows):
        new = out[i, P:].tolist()[: r["budget"]]
        stop = next((j for j, t in enumerate(new) if t in model.eos_ids), None)
        n_tok = (stop + 1) if stop is not None else len(new)
        audio = model.audio_tokens(new[:stop] if stop is not None else new)
        results.append((audio, dict(n_tokens=n_tok, eos=stop is not None, hit_max=stop is None,
                                    n_audio_tokens=len(audio), batch_gen_s=round(gen_s, 3), batch_size=len(rows))))
    return results
