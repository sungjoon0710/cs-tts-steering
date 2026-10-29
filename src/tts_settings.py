"""Lightweight shared generation constants; importing these never loads a model."""
SR = 24_000
AUDIO_BASE = 128266
CODEBOOK_SIZE = 4096
TOKENS_PER_SEC = 82


def token_budget(text: str) -> int:
    return int(TOKENS_PER_SEC * max(4, len(text) / 12) * 2)
