"""Synthesize MaSaC Utterances.txt lines with one baseline model.

Writes <out>/<model>/<voice>/<line>.wav and appends one JSON record per clip to
<out>/<model>/generations.jsonl. Resumable: existing wavs are skipped.

    CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --model indic-speak --voices Kavya Amit \
        --lines 0-99 --out $LOCAL_SP/outputs/baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tts import LOCAL, MODELS, SHARED, SR  # noqa: E402

UTTERANCES = "data/masac/MSH-COMICS/Data/Raw_Utterances/Utterances.txt"


def find(rel: str) -> Path:
    for root in (LOCAL, SHARED):
        if (root / rel).exists():
            return root / rel
    raise FileNotFoundError(rel)


def parse_range(s: str) -> list[int]:
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def load_lines(idx: list[int]) -> list[tuple[int, str, str]]:
    """(0-indexed line number, speaker, text) from 'Speaker: text' / 'Speaker:text' lines.

    A few lines lack the colon ('Sahil मैं तो …'); for those the first word is the speaker.
    """
    lines = find(UTTERANCES).read_text(encoding="utf-8").splitlines()
    out = []
    for i in idx:
        speaker, sep, text = lines[i].partition(":")
        if not sep:
            speaker, _, text = lines[i].strip().partition(" ")
        out.append((i, speaker.strip(), " ".join(text.split())))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--voices", nargs="+", required=True)
    p.add_argument("--lines", default="0-99", help="inclusive 0-indexed range, e.g. 0-99")
    p.add_argument("--out", type=Path, default=LOCAL / "outputs/baseline")
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()

    utts = load_lines(parse_range(a.lines))
    model_dir = a.out / a.model
    model_dir.mkdir(parents=True, exist_ok=True)
    index_path = a.out / "utterances.jsonl"
    index = {}
    if index_path.exists():
        index = {r["line"]: r for r in map(json.loads, index_path.read_text(encoding="utf-8").splitlines())}
    index.update({i: dict(line=i, speaker=s, text=t) for i, s, t in utts})
    index_path.write_text("".join(json.dumps(index[k], ensure_ascii=False) + "\n" for k in sorted(index)),
                          encoding="utf-8")

    t0 = time.time()
    model = MODELS[a.model]()
    print(f"[{a.model}] loaded in {time.time() - t0:.0f}s", flush=True)
    meta = dict(model=a.model, seed=a.seed, decoding=model.decoding, commit=git_commit())

    with open(model_dir / "generations.jsonl", "a", encoding="utf-8") as log:
        for voice in a.voices:
            vdir = model_dir / voice
            vdir.mkdir(exist_ok=True)
            for n, (i, speaker, text) in enumerate(utts, 1):
                wav_path = vdir / f"{i}.wav"
                if wav_path.exists():
                    continue
                rec = dict(meta, voice=voice, line=i, speaker=speaker, text=text, wav=str(wav_path))
                if not text:
                    rec["error"] = "empty text"
                else:
                    try:
                        wav, info = model(text, voice, a.seed)
                        sf.write(wav_path, wav, SR)
                        rec.update(info)
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                status = rec.get("error") or f"{rec['audio_s']:.1f}s rtf {rec['rtf']:.2f}" + (" HIT MAX" if rec["hit_max"] else "")
                print(f"[{a.model}/{voice}] {n}/{len(utts)} line {i}: {status}", flush=True)
    print(f"[{a.model}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
