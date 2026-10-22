"""Reference-free evaluation of a generation run (no ground-truth audio needed).

Reads <run>/<model>/generations.jsonl and the wavs, writes
    <run>/eval/asr.jsonl        Whisper transcripts
    <run>/eval/metrics.csv      one row per clip
    <run>/eval/summary.csv      one row per model/voice
    <run>/eval/summary.md       same, as a markdown table
    <run>/eval/paired_cer.csv   CER per line, one column per model/voice
    <run>/eval/pairwise.md      head-to-head CER on shared lines for every pair of systems

Metrics: CER/WER vs. input text (both romanized with uroman), approximate English-word
survival, F0 median/range, speaking rate, pauses, edge silence, duration sanity, RTF,
and within-voice speaker consistency (WavLM x-vector cosine). Naturalness is left to
human listeners.

    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --run $LOCAL_SP/outputs/baseline
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tts import LOCAL, SR  # noqa: E402

ASR_SR = 16_000
CHARS_PER_SEC = 12        # Indic-Speak library median, used for expected duration
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


# ---------- text ----------

_uroman = None


def romanize(text: str) -> str:
    global _uroman
    if _uroman is None:
        import uroman
        _uroman = uroman.Uroman()
    s = _uroman.romanize_string(text).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(s.split())


def english_survival(ref: str, hyp_rom: str, threshold: float = 0.5) -> tuple[int, int]:
    """(found, total) Latin-script words in ref with a fuzzy match among romanized hyp tokens."""
    words = [w.lower() for w in LATIN_WORD.findall(ref)]
    toks = hyp_rom.split()
    found = sum(any(SequenceMatcher(None, w, t).ratio() >= threshold for t in toks) for w in words)
    return found, len(words)


# ---------- audio ----------

def load(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR, (path, sr)
    return wav


def to_16k(wav: np.ndarray) -> np.ndarray:
    return AF.resample(torch.from_numpy(wav), SR, ASR_SR).numpy()


def prosody(wav: np.ndarray) -> dict:
    import parselmouth

    f0 = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=SR) \
        .to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]
    f0 = f0[f0 > 0]
    out = dict(f0_median=np.nan, f0_range_st=np.nan)
    if len(f0) > 10:
        p5, p95 = np.percentile(f0, [5, 95])
        out.update(f0_median=float(np.median(f0)), f0_range_st=float(12 * np.log2(p95 / p5)))
    return out


def pauses(wav: np.ndarray, rel_db: float = -35, min_pause: float = 0.2) -> dict:
    """Energy-based: frames > rel_db below the loudest frame are silent; internal silent runs >= min_pause are pauses."""
    hop, win = SR // 100, SR // 40
    if len(wav) < win:
        return dict(lead_sil=np.nan, trail_sil=np.nan, speech_s=0.0, n_pauses=0, pause_s=0.0)
    frames = np.lib.stride_tricks.sliding_window_view(wav, win)[::hop]
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    voiced = db > db.max() + rel_db
    idx = np.flatnonzero(voiced)
    if len(idx) == 0:
        return dict(lead_sil=len(wav) / SR, trail_sil=0.0, speech_s=0.0, n_pauses=0, pause_s=0.0)
    first, last = idx[0], idx[-1]
    inner = ~voiced[first:last + 1]
    runs, n = [], 0
    for s in inner:
        if s:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    long_runs = [r / 100 for r in runs if r / 100 >= min_pause]
    return dict(lead_sil=first / 100, trail_sil=max(0.0, len(wav) / SR - (last + 1) / 100 - win / SR),
                speech_s=(last - first + 1) / 100, n_pauses=len(long_runs), pause_s=float(sum(long_runs)))


# ---------- models ----------

def transcribe(paths: list[str], device: str, batch: int = 16) -> list[str]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", dtype=torch.float16).to(device).eval()
    out = []
    for i in range(0, len(paths), batch):
        audio = [to_16k(load(p)) for p in paths[i:i + batch]]
        feats = proc(audio, sampling_rate=ASR_SR, return_tensors="pt").input_features.to(device, torch.float16)
        with torch.no_grad():
            ids = model.generate(feats, language="hi", task="transcribe")
        out += [t.strip() for t in proc.batch_decode(ids, skip_special_tokens=True)]
        print(f"  asr {min(i + batch, len(paths))}/{len(paths)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def speaker_embeddings(paths: list[str], device: str) -> np.ndarray:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    embs = []
    for p in paths:
        inp = fe(to_16k(load(p)), sampling_rate=ASR_SR, return_tensors="pt").to(device)
        with torch.no_grad():
            e = model(**inp).embeddings[0]
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(embs)


# ---------- main ----------

def main():
    import jiwer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=LOCAL / "outputs/baseline")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    # latest successful record per (model, voice, line)
    recs = {}
    for log in sorted(a.run.glob("*/generations.jsonl")):
        for r in map(json.loads, log.read_text(encoding="utf-8").splitlines()):
            if "error" not in r and (a.run / r["wav"]).exists():  # "wav" is relative to the run
                recs[(r["model"], r["voice"], r["line"])] = r
    df = pd.DataFrame(recs.values()).sort_values(["model", "voice", "line"]).reset_index(drop=True)
    print(f"{len(df)} clips", flush=True)

    out_dir = a.run / "eval"
    out_dir.mkdir(exist_ok=True)
    asr_path = out_dir / "asr.jsonl"
    cached = {}
    if asr_path.exists():
        cached = {r["wav"]: r["hyp"] for r in map(json.loads, asr_path.read_text(encoding="utf-8").splitlines())}
    todo = [w for w in df.wav if w not in cached]
    if todo:
        for w, h in zip(todo, transcribe([str(a.run / w) for w in todo], a.device)):
            cached[w] = h
        asr_path.write_text("".join(json.dumps(dict(wav=w, hyp=cached[w]), ensure_ascii=False) + "\n"
                                    for w in df.wav), encoding="utf-8")
    df["hyp"] = df.wav.map(cached)
    df["path"] = [str(a.run / w) for w in df.wav]

    rows = []
    for r in df.itertuples():
        ref_rom, hyp_rom = romanize(r.text), romanize(r.hyp)
        eng_found, eng_total = english_survival(r.text, hyp_rom)
        wav = load(r.path)
        row = dict(
            cer=jiwer.cer(ref_rom, hyp_rom) if ref_rom else np.nan,
            wer=jiwer.wer(ref_rom, hyp_rom) if ref_rom else np.nan,
            eng_found=eng_found, eng_total=eng_total,
            expected_s=max(1.0, len(r.text) / CHARS_PER_SEC),
            **prosody(wav), **pauses(wav))
        row["dur_ratio"] = r.audio_s / row["expected_s"]
        row["chars_per_s"] = len(r.text) / row["speech_s"] if row["speech_s"] > 0 else np.nan
        rows.append(row)
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    df["runaway"] = df.hit_max | (df.dur_ratio > 2.5)
    df["too_short"] = df.dur_ratio < 0.4

    print("speaker embeddings ...", flush=True)
    df["emb_idx"] = range(len(df))
    embs = speaker_embeddings(list(df.path), a.device)
    consistency = {}
    for key, g in df.groupby(["model", "voice"]):
        e = embs[g.emb_idx.to_numpy()]
        consistency[key] = float(np.mean([e[i] @ e[j] for i, j in combinations(range(len(e)), 2)]))

    cols = ["model", "voice", "line", "speaker", "text", "hyp", "cer", "wer", "eng_found", "eng_total",
            "audio_s", "expected_s", "dur_ratio", "speech_s", "chars_per_s", "n_pauses", "pause_s",
            "lead_sil", "trail_sil", "f0_median", "f0_range_st", "n_tokens", "hit_max", "runaway",
            "too_short", "rtf", "wav"]
    df[cols].to_csv(out_dir / "metrics.csv", index=False)

    summ = df.groupby(["model", "voice"]).agg(
        clips=("line", "size"),
        cer_median=("cer", "median"), cer_mean=("cer", "mean"), wer_median=("wer", "median"),
        eng_found=("eng_found", "sum"), eng_total=("eng_total", "sum"),
        f0_median_hz=("f0_median", "median"), f0_range_st=("f0_range_st", "median"),
        chars_per_s=("chars_per_s", "median"), pauses_per_clip=("n_pauses", "mean"),
        pause_s_per_clip=("pause_s", "mean"), lead_sil_s=("lead_sil", "median"), trail_sil_s=("trail_sil", "median"),
        dur_ratio=("dur_ratio", "median"), runaway=("runaway", "sum"), too_short=("too_short", "sum"),
        rtf=("rtf", "median"),
    ).reset_index()
    summ.insert(summ.columns.get_loc("eng_total") + 1, "eng_survival",
                summ.eng_found / summ.eng_total.replace(0, np.nan))
    summ["spk_consistency"] = [consistency[(m, v)] for m, v in zip(summ.model, summ.voice)]
    summ.round(3).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(summ.round(3).to_markdown(index=False) + "\n", encoding="utf-8")
    print(summ.round(3).to_string(index=False))
    compare(df, out_dir)


def compare(df: pd.DataFrame, out_dir: Path) -> None:
    """Paired CER across systems (model/voice) on the lines they share."""
    df = df.assign(system=df.model + "/" + df.voice)
    wide = df.pivot_table(index="line", columns="system", values="cer")
    wide.round(4).to_csv(out_dir / "paired_cer.csv")
    rows = []
    for a, b in combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        d = both[a] - both[b]
        rows.append(dict(A=a, B=b, lines=len(both), A_better=int((d < 0).sum()), B_better=int((d > 0).sum()),
                         ties=int((d == 0).sum()), median_cer_A=both[a].median(), median_cer_B=both[b].median(),
                         median_diff=d.median()))
    pw = pd.DataFrame(rows).round(3)
    (out_dir / "pairwise.md").write_text(pw.to_markdown(index=False) + "\n", encoding="utf-8")
    print(pw.to_string(index=False))


if __name__ == "__main__":
    main()
