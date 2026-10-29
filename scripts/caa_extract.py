"""CAA pilot phase 1: data checks + split, teacher-forced extraction, diagnostics (plan §3-4).

  prepare   --run RUN [--write-split]   data checks, token-cap drop, seeded 80/20 split (+16 dev)
  extract   --run RUN --condition C     SNAC re-encode natural + synthetic clips, teacher-force, pool, NLL
  roundtrip --run RUN --model M         SNAC round-trip checks on real clips (encoder <-> model decoder)
  diagnose  --run RUN                   probes, held-out projection AUC vs permutation, geometry, NLL, shortlist
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.caa_pilot import (SPLIT, condition_voices, hf_snapshot, load_config, load_data, load_split,
                           read_jsonl, write_json)

CONDS = ("indic-amit", "indic-kavya", "orpheus-ritika")


def prepare(a, cfg):
    import soundfile as sf
    utts = load_data(cfg)
    voices = condition_voices(cfg, utts)
    checks, problems = [], []
    for uid, u in sorted(utts.items()):
        paths = [("natural", u["natural_wav"], u["natural_sha256"])] + \
                [(c, s["wav"], s["sha256"]) for c, s in u["synthetic"].items()]
        for kind, p, sha in paths:
            b = Path(p).read_bytes()
            if hashlib.sha256(b).hexdigest() != sha:
                problems.append(f"{uid}/{kind}: sha256 mismatch")
            i = sf.info(p)
            checks.append(dict(utterance_id=uid, kind=kind, sr=i.samplerate, channels=i.channels,
                               duration_s=round(i.duration, 3)))
        for c, s in u["synthetic"].items():
            if s["input_text"] != u["Devanagari"]:
                problems.append(f"{uid}/{c}: input text differs from Devanagari column")
    if len(utts) != 100 or any(len(u["synthetic"]) != 3 for u in utts.values()):
        problems.append("expected 100 utterances x 3 synthetic conditions")
    norm = lambda s: " ".join(s.casefold().split())
    if len({norm(u["Devanagari"]) for u in utts.values()}) != len(utts):
        problems.append("duplicate transcripts")
    capped = sorted({uid for uid, u in utts.items() for s in u["synthetic"].values() if s["token_cap_warning"]})
    keep = sorted(set(utts) - set(capped))
    rng = np.random.default_rng(cfg["split"]["seed"])
    order = list(rng.permutation(keep))
    n_test = int(round(cfg["split"]["test_fraction"] * len(keep)))
    test, train = sorted(order[:n_test]), order[n_test:]
    dev = sorted(train[: cfg["split"]["dev_size"]])
    split = dict(version="caa-pilot-split-v1", seed=cfg["split"]["seed"], source="Listening_Examples_100",
                 dropped_token_cap=capped, n_train=len(train), n_test=len(test), n_dev=len(dev),
                 train=sorted(train), dev=dev, test=test)
    if a.write_split:
        write_json(SPLIT, split)
    durs = {}
    for r in checks:
        durs.setdefault(r["kind"], []).append(r["duration_s"])
    summary = dict(problems=problems, voices=voices, n_utterances=len(utts), dropped_token_cap=capped,
                   formats=sorted({(r["kind"], r["sr"], r["channels"]) for r in checks}),
                   duration_quartiles={k: np.percentile(v, [0, 25, 50, 75, 100]).round(2).tolist() for k, v in durs.items()},
                   natural_speakers={s: sum(u["natural_speaker"] == s for u in utts.values())
                                     for s in sorted({u["natural_speaker"] for u in utts.values()})},
                   split_counts=dict(train=len(train), dev=len(dev), test=len(test)))
    write_json(a.run / "data_checks.json", summary)
    (a.run / "data_checks_files.jsonl").write_text("".join(json.dumps(r) + "\n" for r in checks))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if problems:
        raise SystemExit("data problems: " + "; ".join(problems))


def load_model(cfg, kind, device="cuda"):
    from src.steering import SpeechLM
    pins = cfg["pins"]
    return SpeechLM(kind, hf_snapshot(pins[kind]["repo"], pins[kind]["revision"]),
                    hf_snapshot(pins["codec"]["repo"], pins["codec"]["revision"]), device=device)


def roundtrip(a, cfg):
    """Real-audio round trip: wav -> SNAC codes -> tokens -> model's own decoder function -> same codes."""
    import torch
    from src.snac_encode import load_mono_24k, snac_encode, tokens_to_codes
    from src.tts import snac_codes
    utts = load_data(cfg)
    model = load_model(cfg, a.model)
    out = []
    for uid in sorted(utts)[:5]:
        for kind, p in [("natural", utts[uid]["natural_wav"])] + [(c, s["wav"]) for c, s in utts[uid]["synthetic"].items()]:
            c0, c1, c2 = snac_encode(load_mono_24k(p), model.snac, model.device)
            toks = model.encode_codes(c0, c1, c2)
            r0, r1, r2 = tokens_to_codes(toks, model.base)
            ref = snac_codes(toks, "cpu")  # src/tts.py decoder function (AUDIO_BASE)
            if model.kind == "indic-speak":
                import inference  # snapshot's own ids_to_codes (on sys.path after IndicSpeak load)
                ref = inference.ids_to_codes(toks, model.base, "cpu")
            if model.dedup:
                keep = np.concatenate([[True], c0[1:] != c0[:-1]])
                c0e = c0[keep]
                c1e = c1.reshape(-1, 2)[keep].reshape(-1)
                c2e = c2.reshape(-1, 4)[keep].reshape(-1)
            else:
                c0e, c1e, c2e = c0, c1, c2
            ok = all(np.array_equal(x, y) for x, y in zip((r0, r1, r2), (c0e, c1e, c2e))) and \
                all(np.array_equal(x.numpy()[0], y) for x, y in zip(ref, (c0e, c1e, c2e)))
            out.append(dict(utterance_id=uid, kind=kind, frames=len(c0), kept_frames=len(c0e), ok=bool(ok)))
    # Orpheus: what does the model emit between prompt and audio? (checks speech_prefix)
    if model.kind == "orpheus-hi":
        from src.steering import generate_batch
        u = utts[sorted(utts)[0]]
        voice = condition_voices(cfg, utts)["orpheus-ritika"]
        prompt = model.prompt(u["Devanagari"], voice)
        torch.manual_seed(0)
        ids = torch.tensor([prompt], device=model.device)
        g = model.lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=8,
                              do_sample=True, eos_token_id=model.eos_ids, pad_token_id=model.pad_id, **model.decoding)
        out.append(dict(orpheus_first_generated=g[0, len(prompt):].tolist(), speech_prefix=model.speech_prefix()))
    write_json(a.run / f"roundtrip_{a.model}.json", out)
    print(json.dumps(out, indent=0))
    if not all(r.get("ok", True) for r in out):
        raise SystemExit("round trip FAILED")


def extract(a, cfg):
    import torch
    from src.snac_encode import wav_to_tokens
    from src.steering import POOLS, SITES, extract as run_extract
    utts = load_data(cfg)
    voice = condition_voices(cfg, utts)[a.condition]
    kind = cfg["conditions"][a.condition]["model"]
    out = a.run / "activations" / a.condition
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pooled.npz").exists():
        print("exists; skip")
        return
    model = load_model(cfg, kind)
    t0 = time.time()
    tok_cache = a.run / "activations" / f"tokens_{kind}.json"
    tokens = json.loads(tok_cache.read_text()) if tok_cache.exists() else {}
    items = []
    for uid in sorted(utts):
        u = utts[uid]
        items.append((uid, "natural", u["natural_wav"]))
        items.append((uid, "synthetic", u["synthetic"][a.condition]["wav"]))
    pooled = {(s, p): [] for s in SITES for p in POOLS}
    meta = []
    for uid, cls, wav in items:
        key = f"{uid}|{cls}|{a.condition if cls == 'synthetic' else 'natural'}"
        if key not in tokens:
            toks, dur = wav_to_tokens(wav, model.snac, model.base, model.dedup, model.device)
            tokens[key] = dict(tokens=toks, duration_s=dur)
        toks = tokens[key]["tokens"]
        ids, sl = model.tf_sequence(utts[uid]["Devanagari"], voice, toks)
        r = run_extract(model, ids, sl)
        for k in pooled:
            pooled[k].append(r["pooled"][k].astype(np.float16))
        meta.append(dict(utterance_id=uid, cls=cls, n_audio_tokens=len(toks), seq_len=len(ids),
                         duration_s=round(tokens[key]["duration_s"], 3), nll_mean=float(r["nll"].mean()),
                         nll_median=float(np.median(r["nll"])),
                         token_norm_resid=r["token_norm"]["resid"].round(3).tolist(),
                         token_norm_attn=r["token_norm"]["attn"].round(3).tolist()))
    tok_cache.write_text(json.dumps(tokens))
    np.savez(out / "pooled.npz", **{f"{s}_{p}": np.stack(v) for (s, p), v in pooled.items()})
    (out / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    write_json(out / "extract_info.json", dict(condition=a.condition, voice=voice, model=kind,
                                               pin=cfg["pins"][kind], n=len(meta), runtime_s=round(time.time() - t0, 1),
                                               device=torch.cuda.get_device_name(0), dtype="bfloat16",
                                               sequence="prompt + speech_prefix + audio tokens + end_of_speech",
                                               speech_prefix=model.speech_prefix(), dedup=model.dedup))
    print(f"{a.condition}: {len(meta)} clips in {time.time() - t0:.0f}s")


def load_acts(run, cond):
    d = run / "activations" / cond
    z = np.load(d / "pooled.npz")
    meta = read_jsonl(d / "meta.jsonl")
    return {k: z[k].astype(np.float32) for k in z.files}, meta


def diagnose(a, cfg):
    """Per condition (parallelizable with --condition); `combine` merges and adds cross-condition geometry."""
    import torch
    from src.caa_stats import auc, make_folds, pair_permute, probe_cv, projection_cv
    split = load_split()
    train = set(split["train"])
    dc = cfg["diagnostics"]
    for cond in ([a.condition] if a.condition else CONDS):
        rows = []
        acts, meta = load_acts(a.run, cond)
        uid = np.array([m["utterance_id"] for m in meta])
        y = np.array([1 if m["cls"] == "natural" else 0 for m in meta])
        tr = np.isin(uid, list(train))
        folds = make_folds(uid[tr], cfg["split"]["cv_folds"], cfg["split"]["seed"])
        rng = np.random.default_rng(cfg["seed"])
        perms = [pair_permute(y[tr], uid[tr], rng) for _ in range(dc["permutations"])]
        vecs = {}
        for key, X in acts.items():
            site, pool = key.split("_")
            for layer in range(X.shape[1]):
                Xl = X[tr, layer]
                acc, pauc = probe_cv(Xl, y[tr], uid[tr], folds, pca=dc["pca_dims"], l2=dc["l2"])
                jauc, win, d = projection_cv(Xl, y[tr], uid[tr], folds)
                null = np.array([projection_cv(Xl, yp, uid[tr], folds)[0] for yp in perms])
                pnull = [probe_cv(Xl, yp, uid[tr], folds, pca=dc["pca_dims"], l2=dc["l2"])[0]
                         for yp in perms[:dc["probe_permutations"]]]
                v = Xl[y[tr] == 1].mean(0) - Xl[y[tr] == 0].mean(0)
                hn = np.linalg.norm(Xl, axis=1).mean()
                vecs[(layer, site, pool)] = v
                rows.append(dict(condition=cond, layer=layer, site=site, pool=pool, probe_acc=round(acc, 4),
                                 probe_auc=round(pauc, 4), probe_acc_perm_mean=round(float(np.mean(pnull)), 4),
                                 proj_auc=round(jauc, 4), proj_pair_win=round(win, 4), proj_cohen_d=round(d, 3),
                                 perm_auc_mean=round(float(null.mean()), 4),
                                 perm_auc_q95=round(float(np.quantile(null, 0.95)), 4),
                                 perm_p=round(float((1 + (null >= jauc).sum()) / (1 + len(null))), 4),
                                 v_norm=round(float(np.linalg.norm(v)), 4), h_norm=round(float(hn), 4),
                                 v_rel=round(float(np.linalg.norm(v) / hn), 4)))
            print(cond, key, "done", flush=True)
        (a.run / "vectors").mkdir(exist_ok=True)
        torch.save({f"{l}|{s}|{p}": torch.from_numpy(v) for (l, s, p), v in vecs.items()},
                   a.run / "vectors" / f"{cond}.pt")
        nll = np.array([m["nll_mean"] for m in meta])
        nll_info = dict(condition=cond, nll_natural_mean=float(nll[tr & (y == 1)].mean()),
                        nll_synthetic_mean=float(nll[tr & (y == 0)].mean()),
                        nll_auc_nat_vs_syn=float(auc(nll[tr & (y == 1)], nll[tr & (y == 0)])))
        # shortlist: rank by held-out projection AUC, tie-break held-out Cohen's d, one entry per layer
        cand = sorted(rows, key=lambda r: (-r["proj_auc"], -r["proj_cohen_d"]))
        seen, sl = set(), []
        for r in cand:
            if r["layer"] in seen:
                continue
            seen.add(r["layer"])
            sl.append(dict(layer=r["layer"], site=r["site"], pool=r["pool"], proj_auc=r["proj_auc"],
                           proj_cohen_d=r["proj_cohen_d"], perm_auc_q95=r["perm_auc_q95"], v_rel=r["v_rel"]))
            if len(sl) == dc["shortlist_size"]:
                break
        # NLL confound: cos(v, mean(high-NLL syn) - mean(low-NLL syn)) at shortlisted sites
        syn = tr & (y == 0)
        med = np.median(nll[syn])
        for s in sl:
            X = acts[f"{s['site']}_{s['pool']}"][:, s["layer"]]
            dn = X[syn & (nll > med)].mean(0) - X[syn & (nll <= med)].mean(0)
            v = vecs[(s["layer"], s["site"], s["pool"])]
            s["cos_v_highNLL_dir"] = round(float(dn @ v / (np.linalg.norm(dn) * np.linalg.norm(v))), 4)
        write_json(a.run / "diag" / cond / "probes.json", rows)
        write_json(a.run / "diag" / cond / "shortlist.json", sl)
        write_json(a.run / "diag" / cond / "nll.json", nll_info)
        print(json.dumps(sl, indent=1))


def combine(a, cfg):
    import csv
    import torch
    rows, shortlists, nll_rows = [], {}, []
    for cond in CONDS:
        rows += json.loads((a.run / "diag" / cond / "probes.json").read_text())
        shortlists[cond] = json.loads((a.run / "diag" / cond / "shortlist.json").read_text())
        nll_rows.append(json.loads((a.run / "diag" / cond / "nll.json").read_text()))
    va = torch.load(a.run / "vectors" / "indic-amit.pt", weights_only=True)
    vk = torch.load(a.run / "vectors" / "indic-kavya.pt", weights_only=True)
    cos = []
    for k in va:
        l, s, p = k.split("|")
        x, y = va[k].numpy(), vk[k].numpy()
        cos.append(dict(layer=int(l), site=s, pool=p,
                        cos_amit_kavya=round(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), 4)))
    for name, data in (("probes.csv", rows), ("vector_cosine_amit_kavya.csv", cos)):
        with (a.run / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    write_json(a.run / "shortlist.json", shortlists)
    write_json(a.run / "nll_by_class.json", nll_rows)
    print(json.dumps(shortlists, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "extract", "roundtrip", "diagnose", "combine"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--condition", choices=CONDS)
    p.add_argument("--model", choices=["indic-speak", "orpheus-hi"])
    p.add_argument("--write-split", action="store_true")
    a = p.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    {"prepare": prepare, "extract": extract, "roundtrip": roundtrip, "diagnose": diagnose, "combine": combine}[a.action](a, cfg)
