"""Evaluate the n-shot ICL pilot (docs/nshot_style_prompting_plan.md). Run after scripts/icl_generate.py.

Per clip (150 generated + 30 natural reference rows):
  asr      Whisper-large-v3 (language=hi, greedy, fp16) -> CER/WER vs the LLM text (src/evaluation/text.edit_score);
           shot-text leak = hypothesis words that occur in the clip's shot texts but not in its target text
           (base / natural rows use nat5's shot texts as a control floor). Hypotheses -> metrics/asr.jsonl only.
  prosody  caa_evaluate._features (Praat via src/evaluation/audio.extract_audio) + DTW-aligned f0_ccc / energy_ccc
           vs the line's natural clip (caa_evaluate.prosody logic).
  ssim     cosine of WavLM-base-plus-sv x-vectors (16 kHz) vs the mean x-vector of the speaker's OTHER 14 natural clips.
Outputs <run>/metrics/clips.csv (numbers only, no text), asr.jsonl, summary.json (speaker x condition and paired
bootstrap differences; src/evaluation/report.paired_bootstrap algorithm, 2000 resamples, seed 1234).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from icl_generate import CONDITIONS, LOCAL, SPEAKERS, read_jsonl, shot_ids, utterances  # noqa: E402

METRICS = ["cer", "wer", "f0_median_st", "f0_std_st", "f0_range_st", "intensity_std_db", "rate_wps", "pause_count",
           "pause_total_s", "duration_s", "f0_ccc", "energy_ccc", "ssim", "shot_leak_words", "shot_leak_frac"]
COMPARISONS = [("base", "nat1"), ("base", "nat3"), ("base", "nat5"), ("syn3", "nat3"), ("nat1", "nat3"),
               ("nat5", "nat3")]  # reported as B - A


def load16k(path):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = torch.from_numpy(np.ascontiguousarray(y.mean(1)))
    return AF.resample(y, sr, 16000).numpy() if sr != 16000 else y.numpy()


def items(run, utts):
    import soundfile as sf
    from src.snac_encode import load_mono_24k
    out = []
    for uid, u in utts.items():
        p = run / "natural24k" / f"{uid}.wav"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, load_mono_24k(u["natural_wav"]), 24000, subtype="PCM_16")
        out.append(dict(key=f"natural|{uid}", condition="natural", utterance_id=uid, path=str(p), shots=[]))
    gens = {(g["condition"], g["utterance_id"]): g for g in read_jsonl(run / "generations.jsonl")}
    for (c, uid), g in gens.items():
        out.append(dict(key=f"{c}|{uid}", condition=c, utterance_id=uid, shots=g["shot_ids"], gen=g,
                        path=str(run / g["wav"]) if g.get("wav") else None))
    return out


def pending(path, its):
    done = {r["key"] for r in read_jsonl(path)}
    return [i for i in its if i["key"] not in done and i["path"]]


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def asr(run, its, utts):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from src.evaluation.text import edit_score, normalize
    from src.tts import cached_snapshot
    out = run / "metrics" / "asr.jsonl"
    todo = pending(out, its)
    print(f"asr: {len(todo)} to do", flush=True)
    if not todo:
        return
    snap = cached_snapshot("openai/whisper-large-v3")
    proc = WhisperProcessor.from_pretrained(snap)
    model = WhisperForConditionalGeneration.from_pretrained(snap, dtype=torch.float16,
                                                            attn_implementation="sdpa").to("cuda").eval()
    B = 16
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        feats = proc([load16k(t["path"]) for t in batch], sampling_rate=16000, return_tensors="pt",
                     return_attention_mask=True)
        with torch.inference_mode():
            ids = model.generate(feats.input_features.to("cuda", torch.float16),
                                 attention_mask=feats.attention_mask.to("cuda"), language="hi", task="transcribe",
                                 do_sample=False, num_beams=1, max_new_tokens=440)
        for t, h in zip(batch, proc.batch_decode(ids, skip_special_tokens=True)):
            h = h.strip()
            uid = t["utterance_id"]
            s = edit_score(utts[uid]["text"], h)
            # shot-text leak: control shot set for rows without shots = nat5's shots
            shots = t["shots"] or shot_ids(uid, utts[uid]["speaker"], 5)
            tgt = set(normalize(utts[uid]["text"]).split())
            shot_vocab = set(w for s_ in shots for w in normalize(utts[s_]["text"]).split()) - tgt
            hw = normalize(h).split()
            leak = sum(w in shot_vocab for w in hw)
            append(out, dict(key=t["key"], hyp=h, wer=s["wer"], cer=s["cer"], ref_words=s.get("reference_words"),
                             hyp_words=len(hw), shot_leak_words=leak, shot_leak_frac=leak / len(hw) if hw else 0.0,
                             leak_control=not t["shots"], model="openai/whisper-large-v3"))
        print(f"asr {i + len(batch)}/{len(todo)}", flush=True)


def prosody(run, its, utts):
    from caa_evaluate import _contours, _features
    from src.caa_stats import ccc, dtw_path
    from src.evaluation.text import normalize
    out = run / "metrics" / "prosody.jsonl"
    todo = pending(out, its)
    print(f"prosody: {len(todo)} to do", flush=True)
    nat_cache = {}
    for n, t in enumerate(todo):
        uid = t["utterance_id"]
        rec = dict(key=t["key"], **_features(t["path"], len(normalize(utts[uid]["text"]).split())))
        if t["condition"] != "natural":
            if uid not in nat_cache:
                nat_cache[uid] = _contours(str(run / "natural24k" / f"{uid}.wav"))
            try:
                cs = _contours(t["path"])
            except Exception as e:
                append(out, dict(rec, f0_ccc=None, energy_ccc=None, contour_error=str(e)[:120]))
                continue
            cn = nat_cache[uid]
            path = dtw_path(cs["mfcc"][1], cn["mfcc"][1])
            ts = np.array([cs["mfcc"][0][i] for i, _ in path])
            tn = np.array([cn["mfcc"][0][j] for _, j in path])

            def at(c, tt):
                tc, v = c
                return v[np.clip(np.searchsorted(tc, tt), 0, len(v) - 1)]
            fs_, fn_ = at(cs["f0"], ts), at(cn["f0"], tn)
            ok = (fs_ > 0) & (fn_ > 0)
            f0c = ccc(12 * np.log2(fs_[ok] / np.median(fs_[ok])), 12 * np.log2(fn_[ok] / np.median(fn_[ok]))) \
                if ok.sum() >= 5 else None
            es, en_ = at(cs["en"], ts), at(cn["en"], tn)
            rec.update(f0_ccc=f0c, energy_ccc=ccc(es - es.mean(), en_ - en_.mean()), dtw_len=len(path))
        append(out, rec)
        if (n + 1) % 30 == 0:
            print(f"prosody {n + 1}/{len(todo)}", flush=True)


def ssim(run, its, utts):
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    from src.tts import cached_snapshot
    out = run / "metrics" / "speaker_sv.jsonl"
    todo = pending(out, its)
    print(f"sv: {len(todo)} to do", flush=True)
    if todo:
        snap = cached_snapshot("microsoft/wavlm-base-plus-sv")
        fe = AutoFeatureExtractor.from_pretrained(snap)
        ms = WavLMForXVector.from_pretrained(snap).to("cuda").eval()
        for t in todo:
            y = load16k(t["path"])
            padded = len(y) < 8000
            if padded:
                y = np.pad(y, (0, 8000 - len(y)))
            with torch.inference_mode():
                x = fe(y, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to("cuda")
                sv = torch.nn.functional.normalize(ms(**x).embeddings[0], dim=-1)
            append(out, dict(key=t["key"], padded=padded, sv=[round(float(v), 6) for v in sv.cpu()]))


def compute_ssim(run, utts):
    sv = {r["key"]: np.array(r["sv"]) for r in read_jsonl(run / "metrics" / "speaker_sv.jsonl")}
    res = {}
    for key, e in sv.items():
        uid = key.split("|", 1)[1]
        others = [sv[f"natural|{o}"] for o in SPEAKERS[utts[uid]["speaker"]] if o != uid]
        ref = np.mean(others, 0)
        res[key] = float(e @ ref / (np.linalg.norm(e) * np.linalg.norm(ref)))
    return res


def bootstrap(a, b, seed=1234, repeats=2000):  # = src/evaluation/report.paired_bootstrap (avoids matplotlib import)
    diff = np.asarray(b, float) - np.asarray(a, float)
    draws = np.random.default_rng(seed).integers(0, len(diff), size=(repeats, len(diff)))
    lo, hi = np.quantile(diff[draws].mean(axis=1), [.025, .975])
    return dict(n=len(diff), mean=float(diff.mean()), ci=[float(lo), float(hi)])


def table(run, its, utts):
    a_ = {r["key"]: r for r in read_jsonl(run / "metrics" / "asr.jsonl")}
    p_ = {r["key"]: r for r in read_jsonl(run / "metrics" / "prosody.jsonl")}
    s_ = compute_ssim(run, utts)
    rows = []
    for t in its:
        g = t.get("gen", {})
        r = dict(utterance_id=t["utterance_id"], speaker=utts[t["utterance_id"]]["speaker"], condition=t["condition"],
                 n_shots=len(t["shots"]), hit_max=g.get("hit_max", ""), eos_found=g.get("eos_found", ""),
                 status=g.get("status", "natural"), prompt_tokens=g.get("prompt_tokens", ""),
                 n_new_tokens=g.get("n_new_tokens", ""))
        A, P = a_.get(t["key"], {}), p_.get(t["key"], {})
        for m in METRICS:
            v = s_.get(t["key"]) if m == "ssim" else A.get(m, P.get(m))
            r[m] = "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (round(v, 5) if isinstance(v, float) else v)
        rows.append(r)
    order = {c: i for i, c in enumerate(["natural", *CONDITIONS])}
    rows.sort(key=lambda r: (order[r["condition"]], r["speaker"], r["utterance_id"]))
    out = run / "metrics" / "clips.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # summaries
    val = {(r["condition"], r["utterance_id"]): r for r in rows}
    summ = dict(cells={}, paired={}, failures={})
    for spk, uids in SPEAKERS.items():
        for c in ["natural", *CONDITIONS]:
            cell = {}
            for m in METRICS:
                xs = [val[(c, u)][m] for u in uids if (c, u) in val and val[(c, u)][m] != ""]
                cell[m] = dict(n=len(xs), mean=float(np.mean(xs)) if xs else None,
                               sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else None)
            summ["cells"][f"{spk}|{c}"] = cell
        for A, B in COMPARISONS:
            for m in METRICS:
                pairs = [(val[(A, u)][m], val[(B, u)][m]) for u in uids if (A, u) in val and (B, u) in val
                         and val[(A, u)][m] != "" and val[(B, u)][m] != ""]
                if len(pairs) >= 2:
                    summ["paired"][f"{spk}|{B}-{A}|{m}"] = bootstrap(*zip(*pairs))
    for c in CONDITIONS:
        gs = [t["gen"] for t in its if t["condition"] == c]
        lk = [val[(c, t["utterance_id"])]["shot_leak_words"] for t in its if t["condition"] == c]
        summ["failures"][c] = dict(n=len(gs), hit_max=sum(g["hit_max"] for g in gs),
                                   no_eos=sum(not g["eos_found"] for g in gs),
                                   empty=sum(g["status"] != "generated" for g in gs),
                                   first_non_audio_not_eos=sum(g["first_non_audio_token"] not in (None, 128258) for g in gs),
                                   leak_ge3=sum(1 for x in lk if x != "" and x >= 3),
                                   leak_mean=float(np.mean([x for x in lk if x != ""])) if lk else None,
                                   gen_s_total=round(sum(g["gen_s"] for g in gs), 1),
                                   audio_s_total=round(sum(g["audio_s"] for g in gs), 1))
    (run / "metrics" / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} rows) and summary.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", nargs="+", default=["asr", "prosody", "ssim", "table"])
    a = ap.parse_args()
    run = LOCAL / "outputs" / "icl_pilot" / a.run_id
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    utts = utterances()
    its = items(run, utts)
    for s in a.steps:
        dict(asr=asr, prosody=prosody, ssim=ssim)[s](run, its, utts) if s != "table" else table(run, its, utts)


if __name__ == "__main__":
    main()
