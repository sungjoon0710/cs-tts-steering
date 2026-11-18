"""Re-cut natural utterance clips to their romanized transcript (MMS forced alignment).

Steps (see docs/alignment_plan.md):
  1. locate the old clip in the source audio (FFT cross-correlation)          [conversation mode]
  2. search window = old span +/- --window s                                   [conversation mode]
  3. normalize the romanized transcript (lowercase, a-z and ')
  4. MMS_FA CTC forced alignment with <star> at both ends (Pratap et al. 2024; Hwang et al. 2023)
  5. new span = first word start - pad .. last word end + pad; cut at original rate/channels
  6. QC: mean word score; Whisper-large-v3 normalized CER of old vs new clip

Modes:
  --mode clip          baseline statistics on the existing clips (no new audio is written)
  --mode conversation  source = <conversations>/<utterance_id>.wav (locate, widen, align)

Outputs in --out: pre.TextGrid, alignments.csv, words.jsonl, config.yaml; conversation mode adds
<folder>/<clip>_aligned.wav and post.TextGrid. Files:
words.jsonl, config.yaml.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
import yaml

SR = 16000
WHISPER_PROMPT = ("हाँ यार, आज office में बहुत काम था, लेकिन meeting cancel हो गई. Actually "
                  "मुझे लगा कि boss नाराज़ होंगे, पर सब ठीक-ठाक रहा. चलो, weekend पे movie देखते हैं, okay?")


# ---------------------------------------------------------------- inputs
def load_items(root: Path):
    items = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        tfile = d / "Transcripts.txt"
        nat = sorted(d.glob("01_Natural_*.wav"))
        if not tfile.exists() or not nat:
            continue
        txt = tfile.read_text()
        uid = re.search(r"^utterance_id:\s*(\S+)", txt, re.M).group(1)
        rom = re.search(r"\(0\) Original romanized transcript \(MaSaC\):\n(.+)", txt).group(1).strip()
        items.append(dict(folder=d.name, utterance_id=uid, clip=nat[0], romanized=rom))
    return items


def to_mono16k(x, sr):
    x = torch.from_numpy(x.T if x.ndim == 2 else x[None]).float().mean(0, keepdim=True)
    return AF.resample(x, sr, SR) if sr != SR else x


# ---------------------------------------------------------------- step 1
def locate(clip16, src16):
    """Offset (s) of clip inside src by FFT cross-correlation, plus normalized peak correlation."""
    a, b = clip16[0], src16[0]
    n = a.numel() + b.numel()
    nfft = 1 << (n - 1).bit_length()
    corr = torch.fft.irfft(torch.fft.rfft(b, nfft) * torch.conj(torch.fft.rfft(a, nfft)), nfft)
    corr = corr[: b.numel() - a.numel() + 1]
    k = int(torch.argmax(corr))
    seg = b[k:k + a.numel()]
    peak = float(torch.dot(a, seg) / (a.norm() * seg.norm() + 1e-9))
    return k / SR, peak


# ---------------------------------------------------------------- step 3
def normalize_words(text):
    text = text.lower().replace("’", "'")
    words = [re.sub(r"[^a-z']", "", w) for w in re.split(r"[\s\-–—/]+", text)]
    return [w for w in words if w.strip("'")]


# ---------------------------------------------------------------- step 4
class Aligner:
    def __init__(self, device):
        b = torchaudio.pipelines.MMS_FA
        self.model = b.get_model(with_star=True).to(device).eval()
        self.tok, self.align = b.get_tokenizer(), b.get_aligner()
        self.device = device

    @torch.inference_mode()
    def __call__(self, wav16, words):
        emission, _ = self.model(wav16.to(self.device))
        spans = self.align(emission[0], self.tok(["*"] + words + ["*"]))
        sec = wav16.size(1) / emission.size(1) / SR
        out = []
        for w, s in zip(["*"] + words + ["*"], spans):
            score = float(sum(t.score * (t.end - t.start) for t in s) / sum(t.end - t.start for t in s))
            out.append(dict(word=w, start=s[0].start * sec, end=s[-1].end * sec, score=score))
        return out[1:-1]


# ---------------------------------------------------------------- step 6
class Whisper:
    def __init__(self, device):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        name = "openai/whisper-large-v3"
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        self.prompt = self.proc.get_prompt_ids(WHISPER_PROMPT, return_tensors="pt").to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, wav16):
        f = self.proc(wav16[0].numpy(), sampling_rate=SR, return_tensors="pt").input_features
        ids = self.model.generate(f.to(self.device, torch.float16), language="hi", task="transcribe",
                                  num_beams=5, prompt_ids=self.prompt)
        text = self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text[len(WHISPER_PROMPT):].strip() if text.startswith(WHISPER_PROMPT) else text


_UROMAN = None


def cer_norm(text):
    """Normalized CER text, as in docs/reports/transcript_methods.pdf (uroman + spelling folding)."""
    global _UROMAN
    if _UROMAN is None:
        import uroman
        _UROMAN = uroman.Uroman()
    text = _UROMAN.romanize_string(text).lower()
    out = []
    for w in re.sub(r"[^a-z\s]", " ", text).split():
        for a, b in [("f", "ph"), ("q", "k"), ("z", "j"), ("w", "v"), ("x", "ks"), ("c", "k"),
                     ("ee", "i"), ("oo", "u"), ("ai", "e"), ("au", "o"), ("y", "i"), ("h", "")]:
            w = w.replace(a, b)
        out.append(re.sub(r"(.)\1+", r"\1", w))
    return "".join(out)


def cer(ref, hyp):
    r, h = cer_norm(ref), cer_norm(hyp)
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[-1] / max(len(r), 1)


# ---------------------------------------------------------------- outputs
def textgrid(path, dur, words, t0, label_unmatched="*"):
    """One 'words' tier; times relative to t0, clipped to [0, dur]; gaps labelled '' and the
    audio before the first / after the last aligned word labelled '*' (unmatched by <star>)."""
    iv, t = [], 0.0
    words = [(max(w["start"] - t0, 0), min(w["end"] - t0, dur), w["word"]) for w in words]
    words = [w for w in words if w[1] > w[0]]
    for i, (s, e, w) in enumerate(words):
        if s > t:
            iv.append((t, s, label_unmatched if i == 0 else ""))
        iv.append((s, e, w))
        t = e
    if t < dur:
        iv.append((t, dur, label_unmatched))
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {dur}",
             "tiers? <exists>", "size = 1", "item []:", "    item [1]:", '        class = "IntervalTier"',
             '        name = "words"', "        xmin = 0", f"        xmax = {dur}",
             f"        intervals: size = {len(iv)}"]
    for k, (s, e, w) in enumerate(iv, 1):
        lines += [f"        intervals [{k}]:", f"            xmin = {s}", f"            xmax = {e}",
                  f'            text = "{w}"']
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True, help="Listening_Examples_100_PT dir")
    ap.add_argument("--mode", choices=["clip", "conversation"], default="clip")
    ap.add_argument("--conversations", type=Path, help="dir with <utterance_id>.wav (conversation mode)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=float, default=5.0, help="s added on each side of the old span")
    ap.add_argument("--pad", type=float, default=0.10, help="s kept before first / after last word")
    ap.add_argument("--min-peak", type=float, default=0.8, help="flag if cross-correlation peak is lower")
    ap.add_argument("--max-cer", type=float, default=0.3, help="flag if new-clip CER is higher (report threshold)")
    ap.add_argument("--unmatched", type=float, default=0.05,
                    help="word score below this = text with no matching audio (crammed into the edge)")
    ap.add_argument("--min-unmatched", type=int, default=3, help="flag if at least this many unmatched words")
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.mode == "conversation" and not args.conversations:
        ap.error("--conversations required in conversation mode")

    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(Path(__file__).parent), "status", "--porcelain", "--", __file__],
                           capture_output=True, text=True).stdout.strip()
    (args.out / "config.yaml").write_text(yaml.safe_dump(
        {k: str(v) for k, v in vars(args).items()} | {"commit": commit, "script_uncommitted": bool(dirty),
                                                        "torchaudio": torchaudio.__version__, "model": "MMS_FA"}))

    items = load_items(args.items)[: args.limit]
    aligner = Aligner(dev)
    whisper = None if args.no_whisper else Whisper(dev)
    rows, wfile = [], open(args.out / "words.jsonl", "w")
    for n, it in enumerate(items, 1):
        t_start = time.time()
        clip, csr = sf.read(it["clip"], always_2d=True)
        clip16 = to_mono16k(clip, csr)
        if args.mode == "clip":
            src, ssr, src16, off, peak = clip, csr, clip16, 0.0, 1.0
        else:
            src, ssr = sf.read(args.conversations / f"{it['utterance_id']}.wav", always_2d=True)
            src16 = to_mono16k(src, ssr)
            off, peak = locate(clip16, src16)
        old = (off, off + clip16.size(1) / SR)
        win = (max(old[0] - args.window, 0.0), min(old[1] + args.window, src16.size(1) / SR)) \
            if args.mode == "conversation" else old
        words_in = normalize_words(it["romanized"])
        seg16 = src16[:, int(win[0] * SR):int(win[1] * SR)]
        words = [dict(w, start=w["start"] + win[0], end=w["end"] + win[0]) for w in aligner(seg16, words_in)]
        new = (max(words[0]["start"] - args.pad, win[0]), min(words[-1]["end"] + args.pad, win[1]))

        d = args.out / it["folder"]
        d.mkdir(exist_ok=True)
        textgrid(d / "pre.TextGrid", old[1] - old[0], words, old[0])
        scores = [w["score"] for w in words]
        row = dict(folder=it["folder"], utterance_id=it["utterance_id"], mode=args.mode, n_words=len(words),
                   old_start=round(old[0], 3), old_end=round(old[1], 3), old_dur=round(old[1] - old[0], 3),
                   # audio inside the old clip before the first / after the last aligned word (<star>)
                   lead_unaligned=round(max(words[0]["start"] - old[0], 0), 3),
                   trail_unaligned=round(max(old[1] - words[-1]["end"], 0), 3),
                   mean_word_score=round(float(np.mean(scores)), 3),
                   min_word_score=round(float(np.min(scores)), 3),
                   unmatched_words=sum(x < args.unmatched for x in scores),
                   xcorr_peak=round(peak, 3))
        if whisper:
            row["whisper_old"] = whisper(clip16)
            row["cer_old"] = round(cer(it["romanized"], row["whisper_old"]), 3)
        flags = []
        if row["unmatched_words"] >= args.min_unmatched:
            flags.append("unmatched_words")
        if whisper and row["cer_old"] > args.max_cer:
            flags.append("cer_old_high")
        if args.mode == "conversation":  # step 5: cut the new clip
            textgrid(d / "post.TextGrid", new[1] - new[0], words, new[0])
            sf.write(d / f"{it['clip'].stem}_aligned.wav", src[int(new[0] * ssr):int(new[1] * ssr)], ssr,
                     subtype="PCM_16")
            row |= dict(new_start=round(new[0], 3), new_end=round(new[1], 3))
            if whisper:
                row["whisper_new"] = whisper(src16[:, int(new[0] * SR):int(new[1] * SR)])
                row["cer_new"] = round(cer(it["romanized"], row["whisper_new"]), 3)
            if peak < args.min_peak:
                flags.append("weak_xcorr")
            if whisper and row["cer_new"] > args.max_cer:
                flags.append("cer_new_high")
        row["flags"] = ";".join(flags)
        rows.append(row)
        wfile.write(json.dumps(dict(utterance_id=it["utterance_id"], words=words), ensure_ascii=False) + "\n")
        wfile.flush()
        print(f"[{n}/{len(items)}] {it['folder']} old {old[0]:.2f}-{old[1]:.2f} "
              f"score {row['mean_word_score']:.2f} cer {row.get('cer_old', '-')}->{row.get('cer_new', '-')} "
              f"{row['flags']} ({time.time() - t_start:.1f}s)", flush=True)

    with open(args.out / "alignments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out / "alignments.csv")


if __name__ == "__main__":
    sys.exit(main())
