"""Single-utterance wrappers for the baseline TTS models.

Both models are Llama-3.2-3B speech LMs that emit SNAC 24 kHz codes, 7 tokens per
frame, starting at token id 128266. Each wrapper returns (wav, info) where wav is
float32 mono at 24 kHz.
"""
from __future__ import annotations

import getpass
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from src.tts_settings import SR, AUDIO_BASE, CODEBOOK_SIZE, TOKENS_PER_SEC, token_budget

LOCAL = Path(os.environ.get("LOCAL_SP", f"/local/speech/users/{getpass.getuser()}"))
SHARED = Path("/local/speech/users/sp4050")


def hf_home() -> Path:
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"])
    own = LOCAL / "hf_cache"
    return own if own.exists() else SHARED / "hf_cache"


def cached_snapshot(repo: str) -> Path:
    """Local snapshot dir of a cached HF repo (snapshot_download() rejects it because some files were skipped)."""
    d = hf_home() / "hub" / ("models--" + repo.replace("/", "--"))
    return d / "snapshots" / (d / "refs/main").read_text().strip()


def snac_codes(ids: list[int], device) -> list[torch.Tensor]:
    """Flat audio token ids -> SNAC's 3 codebooks (1:2:4). Stops at the first non-audio token.

    Frame i is [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]],
    each offset by AUDIO_BASE + position * 4096.
    """
    hi = AUDIO_BASE + 7 * CODEBOOK_SIZE
    audio = []
    for t in ids:
        if not AUDIO_BASE <= t < hi:
            break
        audio.append(t)
    n = len(audio) // 7
    if n == 0:
        raise ValueError("no complete SNAC frame generated")
    a = np.array(audio[: n * 7], np.int64).reshape(n, 7)
    a -= AUDIO_BASE + np.arange(7) * CODEBOOK_SIZE
    a = a[np.all((a >= 0) & (a < CODEBOOK_SIZE), axis=1)]
    c1 = np.stack([a[:, 1], a[:, 4]], axis=1).reshape(-1)
    c2 = np.stack([a[:, 2], a[:, 3], a[:, 5], a[:, 6]], axis=1).reshape(-1)
    return [torch.from_numpy(x).unsqueeze(0).to(device) for x in (a[:, 0], c1, c2)]


def _info(n_new, max_new, t0, wav):
    gen_s, audio_s = time.time() - t0, len(wav) / SR
    return dict(n_tokens=n_new, hit_max=n_new >= max_new, gen_s=round(gen_s, 3),
                audio_s=round(audio_s, 3), rtf=round(gen_s / max(audio_s, 1e-6), 3))


class IndicSpeak:
    """bodhan-ai/indic-speak: repo's own prompt builder and Vocos decoder, vendor decoding config."""

    name = "indic-speak"
    decoding = dict(temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

    def __init__(self, device: str = "cuda", snapshot: str | Path | None = None):
        self.device = device
        snap = Path(snapshot) if snapshot is not None else cached_snapshot("bodhan-ai/indic-speak")
        # vocos/best.pt is a pickle that the repo loads with weights_only=False; refuse if it holds anything but plain data
        torch.load(snap / "vocos/best.pt", map_location="cpu", weights_only=True)
        sys.path.insert(0, str(snap))
        import inference
        self.tts = inference.TTS(str(snap), device=device)
        self._build_prompt = inference.build_prompt

    @torch.no_grad()
    def __call__(self, text: str, voice: str, seed: int, max_new_tokens: int | None = None):
        torch.manual_seed(seed)
        tts = self.tts
        prompt = self._build_prompt(tts.tok, text, voice)
        max_new = token_budget(text) if max_new_tokens is None else max_new_tokens
        ids = torch.tensor([prompt], device=self.device)
        t0 = time.time()
        gen = tts.lm.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
            eos_token_id=[tts._eos, tts.tok.eos_token_id], pad_token_id=tts.tok.eos_token_id,
            do_sample=True, **self.decoding)
        new = gen[0, len(prompt):].tolist()
        z_q = tts.snac.quantizer.from_codes(snac_codes(new, self.device))
        wav = tts.vocos(z_q.float())[0, 0].clamp(-1, 1).float().cpu().numpy()
        return wav, _info(len(new), max_new, t0, wav)


class OrpheusHindi:
    """canopylabs/3b-hi-ft-research_release: standard Orpheus prompt, stock SNAC decoder."""

    name = "orpheus-hi-ft"
    repo = "canopylabs/3b-hi-ft-research_release"
    decoding = dict(temperature=0.6, top_p=0.9, repetition_penalty=1.1)
    START_HUMAN, END_TEXT, END_HUMAN = 128259, 128009, 128260
    START_SPEECH, END_SPEECH = 128257, 128258

    def __init__(self, device: str = "cuda", snapshot: str | Path | None = None,
                 codec_snapshot: str | Path | None = None):
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        snap = Path(snapshot) if snapshot is not None else cached_snapshot(self.repo)
        self.tok = AutoTokenizer.from_pretrained(snap)
        self.lm = AutoModelForCausalLM.from_pretrained(
            snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.snac = SNAC.from_pretrained(str(codec_snapshot) if codec_snapshot is not None
                                        else "hubertsiuzdak/snac_24khz").to(device).eval()

    @torch.no_grad()
    def __call__(self, text: str, voice: str, seed: int, max_new_tokens: int | None = None):
        torch.manual_seed(seed)
        body = f"{voice}: {text}" if voice and voice != "none" else text
        prompt = [self.START_HUMAN] + self.tok(body).input_ids + [self.END_TEXT, self.END_HUMAN]
        max_new = token_budget(text) if max_new_tokens is None else max_new_tokens
        ids = torch.tensor([prompt], device=self.device)
        t0 = time.time()
        gen = self.lm.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
            eos_token_id=self.END_SPEECH, pad_token_id=self.END_SPEECH, do_sample=True, **self.decoding)
        new = gen[0, len(prompt):].tolist()
        n_new = len(new)
        if self.START_SPEECH in new:  # audio follows the last start-of-speech marker
            new = new[len(new) - new[::-1].index(self.START_SPEECH):]
        wav = self.snac.decode(snac_codes(new, self.device))[0, 0].clamp(-1, 1).float().cpu().numpy()
        return wav, _info(n_new, max_new, t0, wav)


class OrpheusHindiPretrained(OrpheusHindi):
    """canopylabs/3b-hi-pretrain-research_release: Hindi base model, no named voices.

    Use voice "none" (text-only prompt); the speaker is then whatever the model samples.
    """

    name = "orpheus-hi-pt"
    repo = "canopylabs/3b-hi-pretrain-research_release"


MODELS = {m.name: m for m in (IndicSpeak, OrpheusHindi, OrpheusHindiPretrained)}
