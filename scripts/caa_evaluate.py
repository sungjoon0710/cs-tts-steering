"""CAA pilot evaluation (plan §6): M1-M3, guardrails, selection, test statistics.

Feature extraction (resumable, keyed per clip; scope = natural | sweep1 | sweep2 | test):
  asr      Whisper-large-v3 (pinned), Hindi transcription, WER/CER with hi-en-nfc-v1 normalization
  embed    WavLM-base-plus layer-averaged mean-pooled embedding (M1, M2) + WavLM-SV x-vector (S-SIM)
  prosody  Praat features (src/evaluation/audio.py settings) + DTW-aligned F0/energy contours (M3)
  project  re-teacher-force each clip through its UNSTEERED model, project onto v (mechanistic check)
Analysis (CPU):
  select   dev-sweep summaries; --step 1 ranks sites for top-K; --step 2 applies the §5.2 selection rule
  test     test-set deltas, bootstrap CIs, Wilcoxon, controls, Friedman three-way comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.caa_pilot import (append_jsonl, condition_voices, hf_snapshot, load_config, load_data, load_split,
                           read_jsonl, write_json)

CONDS = ("indic-amit", "indic-kavya", "orpheus-ritika")
PROSODY_FEATS = ["f0_median_st", "f0_std_st", "f0_range_st", "intensity_std_db", "rate_wps",
                 "pause_count", "pause_total_s"]


# ---------------------------------------------------------------------------- items
def natural_24k(run, uid, utts):
    from src.snac_encode import load_mono_24k
    import soundfile as sf
    p = run / "natural24k" / f"{uid}.wav"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, load_mono_24k(utts[uid]["natural_wav"]), 24000, subtype="PCM_16")
    return p


def items(run, scope, cfg, utts):
    split = load_split()
    keep = sorted(set(split["train"]) | set(split["test"]))
    if scope == "natural":
        out = []
        for uid in keep:
            out.append(dict(key=f"nat|{uid}", path=str(natural_24k(run, uid, utts)), utterance_id=uid,
                            condition="natural"))
            for c in CONDS:
                out.append(dict(key=f"orig|{c}|{uid}", path=utts[uid]["synthetic"][c]["wav"], utterance_id=uid,
                                condition=c))
        return out
    gens = [g for g in read_jsonl(run / "generations.jsonl") if g["stage"] == scope and g["status"] == "generated"]
    return [dict(key=g["job_id"], path=str(run / g["wav"]), utterance_id=g["utterance_id"], condition=g["condition"])
            for g in gens]


def pending(path, its):
    done = {r["key"] for r in read_jsonl(path)}
    return [i for i in its if i["key"] not in done]


def load16k(path):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = torch.from_numpy(np.ascontiguousarray(y.mean(1)))
    return AF.resample(y, sr, 16000).numpy() if sr != 16000 else y.numpy()


# ---------------------------------------------------------------------------- extractors
def asr(a, cfg, utts):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from src.evaluation.text import edit_score, NORMALIZATION_VERSION
    out = a.run / "metrics" / "asr.jsonl"
    todo = pending(out, items(a.run, a.scope, cfg, utts))
    print(f"asr {a.scope}: {len(todo)} to do", flush=True)
    if not todo:
        return
    pin = cfg["pins"]["whisper"]
    snap = hf_snapshot(pin["repo"], pin["revision"])
    proc = WhisperProcessor.from_pretrained(snap)
    model = WhisperForConditionalGeneration.from_pretrained(snap, dtype=torch.float16,
                                                            attn_implementation="sdpa").to("cuda").eval()
    B = 16
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        audio = [load16k(t["path"]) for t in batch]
        feats = proc(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = model.generate(feats.input_features.to("cuda", torch.float16),
                                 attention_mask=feats.attention_mask.to("cuda"), language="hi", task="transcribe",
                                 do_sample=False, num_beams=1, max_new_tokens=440)
        hyps = proc.batch_decode(ids, skip_special_tokens=True)
        for t, h in zip(batch, hyps):
            s = edit_score(utts[t["utterance_id"]]["Devanagari"], h.strip())
            append_jsonl(out, dict(key=t["key"], hyp=h.strip(), wer=s["wer"], cer=s["cer"],
                                   ref_words=s.get("reference_words"), word_errors=None if s["wer"] is None else
                                   round(s["wer"] * s["reference_words"]),
                                   ref_chars=s.get("reference_characters"), char_errors=s.get("character_errors"),
                                   normalization=NORMALIZATION_VERSION, model=pin))
        print(f"asr {i + len(batch)}/{len(todo)}", flush=True)


def embed(a, cfg, utts):
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector, WavLMModel
    out = a.run / "metrics" / "embeddings.jsonl"
    todo = pending(out, items(a.run, a.scope, cfg, utts))
    print(f"embed {a.scope}: {len(todo)} to do", flush=True)
    if not todo:
        return
    pe, ps = cfg["pins"]["embedding"], cfg["pins"]["speaker_sv"]
    se, ss = hf_snapshot(pe["repo"], pe["revision"]), hf_snapshot(ps["repo"], ps["revision"])
    fe, fs = AutoFeatureExtractor.from_pretrained(se), AutoFeatureExtractor.from_pretrained(ss)
    me = WavLMModel.from_pretrained(se).to("cuda").eval()
    ms = WavLMForXVector.from_pretrained(ss).to("cuda").eval()
    for n, t in enumerate(todo):
        y = load16k(t["path"])
        padded = len(y) < 8000  # zero-pad clips < 0.5 s (x-vector TDNN needs a minimum length); flagged
        if padded:
            y = np.pad(y, (0, 8000 - len(y)))
        with torch.inference_mode():
            x = fe(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
            hs = me(x, output_hidden_states=True).hidden_states  # 13 x (1, T, 768)
            emb = torch.stack([h[0].mean(0) for h in hs]).mean(0)
            xs = fs(y, sampling_rate=16000, return_tensors="pt", return_attention_mask=True).to("cuda")
            sv = torch.nn.functional.normalize(ms(**xs).embeddings[0], dim=-1)
        append_jsonl(out, dict(key=t["key"], padded=padded, wavlm=[round(float(v), 5) for v in emb.cpu()],
                               sv=[round(float(v), 5) for v in sv.cpu()]))
        if (n + 1) % 100 == 0:
            print(f"embed {n + 1}/{len(todo)}", flush=True)


def _contours(path):
    import parselmouth
    import soundfile as sf
    import torch
    import torchaudio
    y, sr = sf.read(path, dtype="float64", always_2d=True)
    y = y.mean(1)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600, max_number_of_candidates=15,
                            very_accurate=False, silence_threshold=0.03, voicing_threshold=0.45, octave_cost=0.01,
                            octave_jump_cost=0.35, voiced_unvoiced_cost=0.14)
    f0 = pitch.selected_array["frequency"]
    t_f0 = pitch.xs()
    inten = snd.to_intensity(minimum_pitch=60, time_step=0.01, subtract_mean=True)
    t_in, en = inten.xs(), inten.values[0]
    mf = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13, melkwargs=dict(n_fft=int(0.025 * sr),
                                    hop_length=int(0.02 * sr), n_mels=40))(torch.from_numpy(y).float()).T.numpy()
    mf = mf - mf.mean(0)
    t_mf = np.arange(len(mf)) * 0.02
    return dict(f0=(t_f0, f0), en=(t_in, en), mfcc=(t_mf, mf))


def _features(path, ref_words):
    from src.evaluation.audio import extract_audio
    r = extract_audio(path, channel_policy="mean")
    st = lambda hz: 12 * math.log2(hz / 100) if hz else None
    art = r.get("articulation_time_estimate_s")
    return dict(f0_median_st=st(r.get("f0_median_hz")), f0_std_st=r.get("f0_std_st"), f0_range_st=r.get("f0_range_st"),
                intensity_std_db=r.get("intensity_praat_std_db"),
                rate_wps=ref_words / art if art and art > 0.2 else None,
                pause_count=r.get("internal_pause_count"), pause_total_s=r.get("internal_pause_total_s"),
                duration_s=r.get("duration_s"), status=r.get("status"))


def prosody(a, cfg, utts):
    from src.caa_stats import ccc, dtw_path
    from src.evaluation.text import normalize
    out = a.run / "metrics" / "prosody.jsonl"
    its = items(a.run, a.scope, cfg, utts)
    todo = pending(out, its)
    print(f"prosody {a.scope}: {len(todo)} to do", flush=True)
    nat_cache = {}
    for n, t in enumerate(todo):
        uid = t["utterance_id"]
        words = len(normalize(utts[uid]["Devanagari"]).split())
        f = _features(t["path"], words)
        rec = dict(key=t["key"], **f)
        if t["condition"] != "natural":
            if uid not in nat_cache:
                nat_cache[uid] = _contours(str(natural_24k(a.run, uid, utts)))
            try:
                cs = _contours(t["path"])
            except Exception as e:  # e.g. clips shorter than Praat's intensity window
                append_jsonl(out, dict(rec, f0_ccc=None, energy_ccc=None, contour_error=str(e)[:120]))
                continue
            cn = nat_cache[uid]
            path = dtw_path(cs["mfcc"][1], cn["mfcc"][1])
            ts = np.array([cs["mfcc"][0][i] for i, _ in path])
            tn = np.array([cn["mfcc"][0][j] for _, j in path])

            def at(c, tt):
                tc, v = c
                idx = np.clip(np.searchsorted(tc, tt), 0, len(v) - 1)
                return v[idx]
            fs_, fn_ = at(cs["f0"], ts), at(cn["f0"], tn)
            ok = (fs_ > 0) & (fn_ > 0)
            f0c = ccc(12 * np.log2(fs_[ok] / np.median(fs_[ok])), 12 * np.log2(fn_[ok] / np.median(fn_[ok]))) \
                if ok.sum() >= 5 else None
            es, en_ = at(cs["en"], ts), at(cn["en"], tn)
            rec.update(f0_ccc=f0c, energy_ccc=ccc(es - es.mean(), en_ - en_.mean()), dtw_len=len(path))
        append_jsonl(out, rec)
        if (n + 1) % 50 == 0:
            print(f"prosody {n + 1}/{len(todo)}", flush=True)


def project(a, cfg, utts):
    """Mechanistic M1 check: unsteered teacher-forced projection onto unit v at shortlisted sites."""
    import torch
    from src.snac_encode import wav_to_tokens
    from src.steering import SpeechLM, extract
    out = a.run / "metrics" / "projection.jsonl"
    shortlist = json.loads((a.run / "shortlist.json").read_text())
    voices = condition_voices(cfg, utts)
    its = [t for t in items(a.run, a.scope, cfg, utts) if t["condition"] in a.conditions]
    todo = pending(out, its)
    print(f"project {a.scope}: {len(todo)} to do", flush=True)
    by_model = defaultdict(list)
    for t in todo:
        by_model[cfg["conditions"][t["condition"]]["model"]].append(t)
    for kind, ts in by_model.items():
        pins = cfg["pins"]
        model = SpeechLM(kind, hf_snapshot(pins[kind]["repo"], pins[kind]["revision"]),
                         hf_snapshot(pins["codec"]["repo"], pins["codec"]["revision"]))
        vecs = {c: torch.load(a.run / "vectors" / f"{c}.pt", weights_only=True) for c in CONDS}
        for n, t in enumerate(ts):
            c = t["condition"]
            toks, _ = wav_to_tokens(t["path"], model.snac, model.base, model.dedup, model.device)
            if len(toks) < 7:
                append_jsonl(out, dict(key=t["key"], proj=None))
                continue
            ids, sl = model.tf_sequence(utts[t["utterance_id"]]["Devanagari"], voices[c], toks)
            r = extract(model, ids, sl)
            proj = {}
            for s in shortlist[c]:
                v = vecs[c][f"{s['layer']}|{s['site']}|{s['pool']}"].numpy()
                h = r["pooled"][(s["site"], s["pool"])][s["layer"]]
                proj[f"{s['layer']}|{s['site']}|{s['pool']}"] = float(h @ v / np.linalg.norm(v))
            append_jsonl(out, dict(key=t["key"], proj=proj, nll_mean=float(r["nll"].mean())))
            if (n + 1) % 100 == 0:
                print(f"project {kind} {n + 1}/{len(ts)}", flush=True)


# ---------------------------------------------------------------------------- analysis
def load_metrics(run):
    m = {}
    for name in ("asr", "embeddings", "prosody", "projection"):
        m[name] = {r["key"]: r for r in read_jsonl(run / "metrics" / f"{name}.jsonl")}
    return m


def m1_classifier(run, cfg, m):
    """Shared P(natural) classifier on WavLM embeddings: train-utterance natural vs all three conditions' synthetic."""
    from src.caa_stats import Standardize, auc, fit_logreg
    split = load_split()
    emb = m["embeddings"]

    def X(keys):
        return np.array([emb[k]["wavlm"] for k in keys], np.float64)
    tr_nat = [f"nat|{u}" for u in split["train"]]
    tr_syn = [f"orig|{c}|{u}" for u in split["train"] for c in CONDS]
    Xtr = np.concatenate([X(tr_nat), X(tr_syn)])
    ytr = np.r_[np.ones(len(tr_nat)), np.zeros(len(tr_syn))]
    st = Standardize().fit(Xtr)
    # class-balanced by duplicating naturals x3 (equal total weight)
    Xb = np.concatenate([st(X(tr_nat))] * 3 + [st(X(tr_syn))])
    yb = np.r_[np.ones(3 * len(tr_nat)), np.zeros(len(tr_syn))]
    w, b = fit_logreg(Xb, yb, l2=float(cfg["diagnostics"].get("m1_l2", 10.0)))

    def score(keys):  # log-odds of natural (P saturates near 0/1, so deltas use the logit)
        return st(X(keys)) @ w + b
    sig = lambda z: 1 / (1 + np.exp(-z))
    te_nat = score([f"nat|{u}" for u in split["test"]])
    info = dict(train_n_nat=len(tr_nat), train_n_syn=len(tr_syn), l2=float(cfg["diagnostics"].get("m1_l2", 10.0)))
    for c in CONDS:
        s = score([f"orig|{c}|{u}" for u in split["test"]])
        info[f"heldout_auc_natural_vs_{c}"] = auc(te_nat, s)
        info[f"heldout_mean_p_{c}"] = float(sig(s).mean())
        info[f"heldout_mean_logit_{c}"] = float(s.mean())
    info["heldout_mean_p_natural"] = float(sig(te_nat).mean())
    info["heldout_mean_logit_natural"] = float(te_nat.mean())
    info["score"] = "M1 = classifier logit (log-odds natural); M1_p = P(natural)"
    return score, info


def m3_distance(m, nat_iqr, key, uid):
    p, n = m["prosody"].get(key), m["prosody"].get(f"nat|{uid}")
    if not p or not n:
        return None, {}
    parts = {}
    for f in PROSODY_FEATS:
        if p.get(f) is not None and n.get(f) is not None and nat_iqr[f] > 0:
            parts[f] = abs(p[f] - n[f]) / nat_iqr[f]
    return (float(np.mean(list(parts.values()))) if parts else None), parts


def natural_iqr(m, uids):
    out = {}
    for f in PROSODY_FEATS:
        v = [m["prosody"][f"nat|{u}"][f] for u in uids if m["prosody"].get(f"nat|{u}", {}).get(f) is not None]
        out[f] = float(np.subtract(*np.percentile(v, [75, 25])))
    return out


def clip_table(run, cfg, stage, m, score, nat_iqr):
    """One row per generated clip with all metrics; baseline-relative fields filled per (cond, utt, seed)."""
    gens = [g for g in read_jsonl(run / "generations.jsonl") if g["stage"] == stage]
    emb = m["embeddings"]
    rows = []
    for g in gens:
        k, uid = g["job_id"], g["utterance_id"]
        ok = g["status"] == "generated"
        r = dict(job_id=k, condition=g["condition"], utterance_id=uid, seed=g["seed"], label=g["label"],
                 steer=g["steer"], eos=g["eos"], hit_max=g["hit_max"], duration_s=g["duration_s"], generated=ok)
        if ok and k in emb:
            e = np.array(emb[k]["wavlm"])
            nat = np.array(emb[f"nat|{uid}"]["wavlm"])
            r["M1"] = float(score([k])[0])
            r["M1_p"] = float(1 / (1 + np.exp(-r["M1"])))
            r["M2"] = float(e @ nat / (np.linalg.norm(e) * np.linalg.norm(nat)))
        if ok:
            r["M3"], parts = m3_distance(m, nat_iqr, k, uid)
            p = m["prosody"].get(k, {})
            r.update({f"pros_{f}": p.get(f) for f in PROSODY_FEATS})
            r["f0_ccc"], r["energy_ccc"] = p.get("f0_ccc"), p.get("energy_ccc")
            a_ = m["asr"].get(k)
            if a_:
                r.update(wer=a_["wer"], cer=a_["cer"], word_errors=a_["word_errors"], ref_words=a_["ref_words"],
                         char_errors=a_["char_errors"], ref_chars=a_["ref_chars"])
            pr = m["projection"].get(k)
            if pr and pr.get("proj"):
                r["proj"] = pr["proj"]
        rows.append(r)
    base = {(r["condition"], r["utterance_id"], r["seed"]): r for r in rows if r["label"] == "base"}
    ssv = {}
    for r in rows:
        b = base.get((r["condition"], r["utterance_id"], r["seed"]))
        r["base_duration_s"] = b["duration_s"] if b else None
        if b and r["generated"] and b["generated"] and r["job_id"] in emb and b["job_id"] in emb:
            r["ssim"] = float(np.dot(emb[r["job_id"]]["sv"], emb[b["job_id"]]["sv"]))
        lo, hi = cfg["guardrails"]["duration_ratio"]
        dur_bad = (b is not None and b["duration_s"] and r["duration_s"] is not None and
                   not (lo <= r["duration_s"] / b["duration_s"] <= hi))
        r["failure"] = bool((not r["generated"]) or (not r["eos"]) or r["hit_max"] or dur_bad)
    return rows


def config_summary(rows, cfg):
    """Per (condition, label) aggregates and paired per-utterance deltas vs base (seeds averaged)."""
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["label"])].append(r)
    for (c, lab), rs in by.items():
        base = {(r["utterance_id"], r["seed"]): r for r in by[(c, "base")]}
        s = dict(condition=c, label=lab, n=len(rs), steer=rs[0]["steer"])
        s["failure_rate"] = float(np.mean([r["failure"] for r in rs]))
        we = sum(r.get("word_errors") or 0 for r in rs if r.get("wer") is not None)
        ww = sum(r.get("ref_words") or 0 for r in rs if r.get("wer") is not None)
        ce = sum(r.get("char_errors") or 0 for r in rs if r.get("cer") is not None)
        cc = sum(r.get("ref_chars") or 0 for r in rs if r.get("cer") is not None)
        s["wer"] = we / ww if ww else None
        s["cer"] = ce / cc if cc else None
        ss = [r["ssim"] for r in rs if r.get("ssim") is not None]
        s["ssim"] = float(np.mean(ss)) if ss else None
        for metric in ("M1", "M1_p", "M2", "M3", "f0_ccc", "energy_ccc", "pros_f0_median_st", "wer", "cer", "duration_s"):
            per_u = defaultdict(list)
            for r in rs:
                b = base.get((r["utterance_id"], r["seed"]))
                if b and r.get(metric) is not None and b.get(metric) is not None:
                    per_u[r["utterance_id"]].append(r[metric] - b[metric])
            d = np.array([np.mean(v) for v in per_u.values()])
            s[f"d_{metric}"] = float(d.mean()) if len(d) else None
            s[f"d_{metric}_by_utt"] = {u: float(np.mean(v)) for u, v in per_u.items()}
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            s[metric] = float(np.mean(vals)) if vals else None
        out[(c, lab)] = s
    return out


def composite(summ, c, lab, base_sd):
    s = summ[(c, lab)]
    parts = []
    for metric, sign in (("M1", 1), ("M2", 1), ("M3", -1)):
        if s.get(f"d_{metric}") is None or not base_sd.get(metric):
            return None
        parts.append(sign * s[f"d_{metric}"] / base_sd[metric])
    return float(np.mean(parts))


def guard(summ, c, lab, cfg):
    s, b, g = summ[(c, lab)], summ[(c, "base")], cfg["guardrails"]
    checks = dict(wer=s["wer"] is not None and s["wer"] <= b["wer"] + g["wer_points"],
                  cer=s["cer"] is not None and s["cer"] <= b["cer"] + g["wer_points"],
                  ssim=s["ssim"] is not None and s["ssim"] >= g["ssim_min"],
                  failure=s["failure_rate"] <= max(g["failure_max"], b["failure_rate"]))  # Orpheus: own baseline
    return all(checks.values()), checks


def base_sds(rows, c):
    out = {}
    for metric in ("M1", "M2", "M3"):
        v = [r[metric] for r in rows if r["condition"] == c and r["label"] == "base" and r.get(metric) is not None]
        out[metric] = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def analysis_setup(a, cfg):
    m = load_metrics(a.run)
    split = load_split()
    score, m1info = m1_classifier(a.run, cfg, m)
    iqr = natural_iqr(m, split["train"] + split["test"])
    return m, score, m1info, iqr


def select(a, cfg, utts):
    m, score, m1info, iqr = analysis_setup(a, cfg)
    stage = f"sweep{a.step}"
    rows = clip_table(a.run, cfg, stage, m, score, iqr)
    summ = config_summary(rows, cfg)
    table, result = [], {}
    for c in CONDS:
        sd = base_sds(rows, c)
        labs = sorted({lab for (cc, lab) in summ if cc == c})
        for lab in labs:
            s = summ[(c, lab)]
            passed, checks = guard(summ, c, lab, cfg)
            table.append(dict({k: v for k, v in s.items() if not k.endswith("_by_utt")},
                              composite=composite(summ, c, lab, sd), guard_pass=passed,
                              **{f"guard_{k}": v for k, v in checks.items()}))
    sel_dir = a.run / "selection"
    write_csv(sel_dir / f"{stage}_summary.csv", [dict(r, steer=json.dumps(r["steer"])) for r in table])
    write_json(sel_dir / "m1_classifier.json", m1info)
    write_csv(sel_dir / f"{stage}_clips.csv", [dict({k: v for k, v in r.items() if k != "proj"},
                                                    steer=json.dumps(r["steer"])) for r in rows])
    if a.step == 1:
        # rank shortlisted sites: best composite over alpha among guardrail-passing configs; else best composite overall
        shortlist = json.loads((a.run / "shortlist.json").read_text())
        # Step-1 ranking ("best layers" for top-K). Usable = not structurally broken at 16-clip resolution:
        # failure <= max(5%, base) + 1 clip, S-SIM >= 0.75, WER <= base + 5 points. Rank usable sites by best
        # composite over alpha; unusable sites after, by lowest failure rate. Strict guardrails apply at step 2.
        ranking = {}
        g = cfg["guardrails"]
        for c in CONDS:
            b = summ[(c, "base")]
            scored = []
            for s in shortlist[c]:
                key = f"L{s['layer']}{s['site']}-{s['pool']}"
                cands = [r for r in table if r["condition"] == c and r["label"].startswith(key + "_")]
                usable = [r for r in cands if r["composite"] is not None
                          and r["failure_rate"] <= max(g["failure_max"], b["failure_rate"]) + 1 / 16 + 1e-9
                          and (r["ssim"] or 0) >= g["ssim_min"] and r["wer"] is not None and r["wer"] <= b["wer"] + 0.05]
                best = max((r["composite"] for r in usable), default=None)
                min_fail = min(r["failure_rate"] for r in cands)
                scored.append((bool(usable), best if usable else -min_fail, s, [r["label"] for r in usable]))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            ranking[c] = [dict(layer=s["layer"], site=s["site"], pool=s["pool"], usable=u,
                               score=sc, usable_configs=labs) for u, sc, s, labs in scored]
        write_json(sel_dir / "sweep1_ranking.json", ranking)
        # does steering effect rise with projection AUC? (CoCoEmo Fig. 7)
        corr = {}
        for c in CONDS:
            xs, ys = [], []
            for s in shortlist[c]:
                key = f"L{s['layer']}{s['site']}-{s['pool']}"
                for r in table:
                    if r["condition"] == c and r["label"].startswith(key + "_") and r["composite"] is not None:
                        xs.append(s["proj_cohen_d"]), ys.append(r["composite"])
            corr[c] = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else None
        write_json(sel_dir / "sweep1_effect_vs_separation.json", corr)
        print(json.dumps(ranking, indent=1))
    else:
        for c in CONDS:
            cands = [r for r in table if r["condition"] == c and r["label"] != "base" and r["composite"] is not None]
            passing = [r for r in cands if r["guard_pass"]]
            if passing:
                best = max(passing, key=lambda r: r["composite"])
            else:
                best = None
            entry = dict(n_candidates=len(cands), n_passing=len(passing))
            if best:
                k = best["label"].split("_a")[0]
                alphas = sorted(float(r["label"].split("_a")[1]) for r in passing if r["label"].startswith(k + "_a"))
                a_star = float(best["label"].split("_a")[1])
                milder = [x for x in alphas if x < a_star]
                a_mild = milder[-1] if milder else a_star / 2
                entry.update(label=best["label"], sites=[dict(layer=s["layer"], site=s["site"], pool=s["pool"])
                                                         for s in best["steer"]],
                             alpha_star=a_star, alpha_mild=a_mild, composite=best["composite"],
                             d_M1=best["d_M1"], d_M2=best["d_M2"], d_M3=best["d_M3"], wer=best["wer"],
                             ssim=best["ssim"], failure_rate=best["failure_rate"],
                             mild_rule="next-lower passing alpha of the same top-K; alpha*/2 if none")
            else:
                entry.update(label=None, note="no configuration passed all guardrails")
            result[c] = entry
        write_json(sel_dir / "selection_rule_output.json", result)
        print(json.dumps(result, indent=1))


def test(a, cfg, utts):
    from src.caa_stats import bootstrap_ci, friedman, wilcoxon
    m, score, m1info, iqr = analysis_setup(a, cfg)
    rows = clip_table(a.run, cfg, "test", m, score, iqr)
    summ = config_summary(rows, cfg)
    out_dir = a.run / "metrics" / "test"
    write_csv(out_dir / "clips.csv", [dict({k: v for k, v in r.items() if k != "proj"}, steer=json.dumps(r["steer"]))
                                      for r in rows])
    table, stats = [], {}
    labels = ["base", "v_star", "v_mild", "neg_star", "rand0_star", "rand1_star", "rand2_star"]
    for c in CONDS:
        for lab in labels:
            if (c, lab) not in summ:
                continue
            s = summ[(c, lab)]
            row = dict(condition=c, label=lab, n=s["n"], failure_rate=s["failure_rate"], wer=s["wer"], cer=s["cer"],
                       ssim=s["ssim"], M1=s["M1"], M1_p=s["M1_p"], M2=s["M2"], M3=s["M3"], f0_ccc=s["f0_ccc"], energy_ccc=s["energy_ccc"])
            for metric in ("M1", "M1_p", "M2", "M3", "f0_ccc", "energy_ccc", "pros_f0_median_st", "duration_s"):
                d = np.array(list(s[f"d_{metric}_by_utt"].values()))
                row[f"d_{metric}"] = s[f"d_{metric}"]
                if lab != "base" and len(d):
                    lo, hi = bootstrap_ci(d, seed=7)
                    row[f"d_{metric}_ci"] = [lo, hi]
                    row[f"d_{metric}_wilcoxon_p"] = wilcoxon(d)["p"]
            table.append(row)
        # steered vs random (per utterance: v_star - mean(rand draws)), per metric
        for metric in ("M1", "M2", "M3"):
            vs = summ.get((c, "v_star"), {}).get(f"d_{metric}_by_utt", {})
            rnd = [summ.get((c, f"rand{d}_star"), {}).get(f"d_{metric}_by_utt", {}) for d in range(3)]
            diff = [vs[u] - np.mean([r[u] for r in rnd if u in r]) for u in vs if any(u in r for r in rnd)]
            stats[f"{c}|{metric}|v_star_minus_random"] = dict(mean=float(np.mean(diff)) if diff else None,
                                                             ci=bootstrap_ci(diff, seed=11), wilcoxon=wilcoxon(diff))
    # three-way comparison of gains (v_star - base) on common utterances
    for metric in ("M1", "M2", "M3"):
        g = [summ.get((c, "v_star"), {}).get(f"d_{metric}_by_utt", {}) for c in CONDS]
        common = sorted(set.intersection(*[set(x) for x in g])) if all(g) else []
        if len(common) >= 3:
            M = np.array([[x[u] for x in g] for u in common])
            stats[f"threeway|{metric}|friedman"] = friedman(M)
            for i in range(3):
                for j in range(i + 1, 3):
                    stats[f"threeway|{metric}|{CONDS[i]}_vs_{CONDS[j]}"] = dict(
                        mean_diff=float((M[:, i] - M[:, j]).mean()), wilcoxon=wilcoxon(M[:, i] - M[:, j]))
    # mechanistic: unsteered projection onto v at the first selected site
    sel = json.loads((a.run / "selection" / "selection.json").read_text())
    for c in CONDS:
        if not sel.get(c, {}).get("sites"):
            continue
        s0 = sel[c]["sites"][0]
        key = f"{s0['layer']}|{s0['site']}|{s0['pool']}"
        per = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["condition"] == c and r.get("proj") and key in r["proj"]:
                per[r["label"]][r["utterance_id"]].append(r["proj"][key])
        b = {u: np.mean(v) for u, v in per["base"].items()}
        for lab in labels[1:]:
            d = [np.mean(v) - b[u] for u, v in per[lab].items() if u in b]
            if d:
                stats[f"{c}|proj|{lab}"] = dict(site=key, mean=float(np.mean(d)), ci=bootstrap_ci(d, seed=13),
                                                wilcoxon=wilcoxon(d))
    write_csv(out_dir / "summary.csv", table)
    write_json(out_dir / "stats.json", dict(m1_classifier=m1info, tests=stats))
    write_json(out_dir / "summary.json", table)
    print(json.dumps(table, indent=0)[:4000])


def listening(a, cfg, utts):
    """Blinded side-by-side package (plan §6.3): unsteered vs steered (alpha*), same condition and seed.

    Kept on kedi only (contains audio + transcripts). key.csv unblinds; items.csv / index.html do not."""
    import html
    import random
    import shutil
    split = load_split()
    gens = {(g["condition"], g["utterance_id"], g["seed"], g["label"]): g
            for g in read_jsonl(a.run / "generations.jsonl") if g["stage"] == "test" and g["status"] == "generated"}
    out = a.run / "listening"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    (out / "audio").mkdir(parents=True)
    rng = random.Random(cfg["seed"])
    seeds = cfg["steering"]["test_seeds"]
    items = []
    for c in CONDS:
        ctrl = set(rng.sample(split["test"], 2))  # random-vector control items per condition
        for uid in split["test"]:
            for lab in ["v_star"] + (["rand0_star"] if uid in ctrl else []):
                seed = next((s for s in seeds if (c, uid, s, "base") in gens and (c, uid, s, lab) in gens), None)
                if seed is None:
                    continue
                items.append(dict(condition=c, utterance_id=uid, seed=seed, steered_label=lab,
                                  base=gens[(c, uid, seed, "base")], steered=gens[(c, uid, seed, lab)]))
    rng.shuffle(items)
    key, public = [], []
    for n, it in enumerate(items, 1):
        iid = f"item{n:03d}"
        pair = [("base", it["base"]), (it["steered_label"], it["steered"])]
        rng.shuffle(pair)
        for side, (lab, g) in zip("AB", pair):
            shutil.copy(a.run / g["wav"], out / "audio" / f"{iid}_{side}.wav")
        shutil.copy(natural_24k(a.run, it["utterance_id"], utts), out / "audio" / f"{iid}_ref.wav")
        key.append(dict(item_id=iid, condition=it["condition"], utterance_id=it["utterance_id"], seed=it["seed"],
                        A_label=pair[0][0], B_label=pair[1][0], A_job=pair[0][1]["job_id"], B_job=pair[1][1]["job_id"],
                        control_item=it["steered_label"] != "v_star"))
        public.append(dict(item_id=iid, A=f"audio/{iid}_A.wav", B=f"audio/{iid}_B.wav", reference=f"audio/{iid}_ref.wav",
                           transcript=utts[it["utterance_id"]]["Devanagari"]))
    write_csv(out / "key.csv", key)
    write_csv(out / "items.csv", [{k: v for k, v in r.items() if k != "transcript"} for r in public])
    write_csv(out / "rating_sheet_template.csv",
              [dict(rater_id="", item_id=r["item_id"], q1_more_natural_A_B_none="", q2_mos_A_1to5="", q2_mos_B_1to5="",
                    q3_mispronounced_or_lost_A="", q3_mispronounced_or_lost_B="", notes="") for r in public])
    rows = "\n".join(
        f"<section><h2>{r['item_id']}</h2><p>A <audio controls preload='none' src='{r['A']}'></audio></p>"
        f"<p>B <audio controls preload='none' src='{r['B']}'></audio></p>"
        f"<details><summary>Q3: show transcript (only after answering Q1-Q2)</summary><p lang='hi'>{html.escape(r['transcript'])}</p></details>"
        f"<details><summary>Optional anchor: real recording (different speaker)</summary>"
        f"<audio controls preload='none' src='{r['reference']}'></audio></details></section>" for r in public)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Hinglish naturalness pairs</title>
<style>body{{font-family:sans-serif;max-width:760px;margin:auto;padding:16px}}section{{border-bottom:1px solid #ccc;padding:8px 0}}</style>
<h1>Side-by-side naturalness (bilingual Hindi-English listeners)</h1>
<p>For each item: (1) Which sounds more like natural conversational Hinglish: A, B or no difference? (2) Rate each clip's
naturalness 1-5. (3) Then open the transcript: was anything mispronounced or lost? Enter answers in
<code>rating_sheet_template.csv</code> (one copy per rater). The anchor is a real TV recording by a different speaker.</p>
{rows}""", encoding="utf-8")
    write_json(out / "README.json", dict(n_items=len(items), n_control=sum(k["control_item"] for k in key),
                                         per_condition={c: sum(k["condition"] == c for k in key) for c in CONDS},
                                         note="kedi only: audio + MaSaC transcripts. key.csv unblinds labels."))
    print(f"{len(items)} items -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["asr", "embed", "prosody", "project", "select", "test", "listening"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--scope", default="natural", choices=["natural", "sweep1", "sweep2", "test"])
    p.add_argument("--step", type=int, choices=[1, 2])
    p.add_argument("--conditions", nargs="+", default=list(CONDS))
    a = p.parse_args()
    cfg = load_config()
    utts = load_data(cfg)
    fn = dict(asr=asr, embed=embed, prosody=prosody, project=project, select=select, test=test,
              listening=listening)[a.action]
    t0 = time.time()
    fn(a, cfg, utts)
    print(f"{a.action} done in {time.time() - t0:.0f}s", flush=True)
