"""wav -> LM audio tokens for Indic-Speak and Orpheus (inverse of `ids_to_codes` / `snac_codes`).

Both models flatten SNAC 24 kHz codes to 7 tokens per frame:
frame i -> [c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]],
each code offset by base + position * 4096. The base id is read from each tokenizer.
Indic-Speak drops consecutive frames that repeat the previous frame's c0 (token_contract.md);
Orpheus keeps every frame.
"""
from __future__ import annotations

import numpy as np

SR = 24_000
CODEBOOK_SIZE = 4096
FRAME = 7


def codes_to_tokens(c0, c1, c2, base: int, dedup: bool) -> list[int]:
    """SNAC codebooks (1:2:4 lengths) -> flat token ids."""
    c0, c1, c2 = (np.asarray(x, np.int64).reshape(-1) for x in (c0, c1, c2))
    n = len(c0)
    if len(c1) != 2 * n or len(c2) != 4 * n:
        raise ValueError(f"codebook lengths {len(c0)},{len(c1)},{len(c2)} are not 1:2:4")
    if np.any((c0 < 0) | (c0 >= CODEBOOK_SIZE)) or np.any((c1 < 0) | (c1 >= CODEBOOK_SIZE)) \
            or np.any((c2 < 0) | (c2 >= CODEBOOK_SIZE)):
        raise ValueError("SNAC code out of range")
    frames = np.stack([c0, c1[0::2], c2[0::4], c2[1::4], c1[1::2], c2[2::4], c2[3::4]], axis=1)
    if dedup and n > 1:
        keep = np.concatenate([[True], frames[1:, 0] != frames[:-1, 0]])
        frames = frames[keep]
    frames = frames + base + np.arange(FRAME) * CODEBOOK_SIZE
    return frames.reshape(-1).tolist()


def tokens_to_codes(ids, base: int):
    """Flat token ids -> (c0, c1, c2) numpy arrays. Same contract as `snac_codes` / `ids_to_codes`."""
    hi = base + FRAME * CODEBOOK_SIZE
    audio = []
    for t in ids:
        if not base <= t < hi:
            break
        audio.append(t)
    n = len(audio) // FRAME
    a = np.array(audio[: n * FRAME], np.int64).reshape(n, FRAME) - (base + np.arange(FRAME) * CODEBOOK_SIZE)
    a = a[np.all((a >= 0) & (a < CODEBOOK_SIZE), axis=1)]
    c1 = np.stack([a[:, 1], a[:, 4]], axis=1).reshape(-1)
    c2 = np.stack([a[:, 2], a[:, 3], a[:, 5], a[:, 6]], axis=1).reshape(-1)
    return a[:, 0], c1, c2


def load_mono_24k(path):
    """Read a wav, average channels, resample to 24 kHz. Returns float32 numpy (no other processing)."""
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:  # torchaudio sinc resampling (scipy is not in the env)
        y = AF.resample(torch.from_numpy(np.ascontiguousarray(y)), sr, SR).numpy()
    return y.astype(np.float32)


def snac_encode(wav: np.ndarray, snac, device):
    """float32 mono 24 kHz -> (c0, c1, c2) numpy arrays."""
    import torch
    x = torch.from_numpy(np.asarray(wav, np.float32)).to(device)[None, None]
    with torch.inference_mode():
        codes = snac.encode(x)
    return tuple(c[0].cpu().numpy() for c in codes)


def wav_to_tokens(path, snac, base: int, dedup: bool, device):
    wav = load_mono_24k(path)
    c0, c1, c2 = snac_encode(wav, snac, device)
    return codes_to_tokens(c0, c1, c2, base, dedup), len(wav) / SR
