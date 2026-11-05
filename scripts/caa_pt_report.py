"""CAA follow-up on Orpheus Hindi PT: three-group evaluation (Natural / PT base / PT steered), figures, listening.

  nisqa      --run RUN --nisqa-source DIR   NISQA-TTS predictions (run in envs/nisqa; scripts/learned_evaluate.nisqa)
  tables     --run RUN                      per-clip table + group aggregates + steered-minus-base statistics
  figures    --run RUN                      bar charts from RUN/tables/figure_data.json (matplotlib; envs/nisqa)
  listening  --run RUN --out DIR            organized listening folder mirroring Listening_Examples_100

Metric code is reused, not re-implemented: WER/CER (src/evaluation/text.py via content_proxies), ASR rates
(src/evaluation/expanded.content_proxies formulas), Praat prosody / voice quality (expanded_audio), syllable nuclei
(scripts/lab_rate_evaluate.py terminal_v2), speaker consistency (src/evaluation/learned.speaker_consistency),
NISQA (scripts/learned_evaluate.nisqa), M1-M3 / S-SIM / failure (scripts/caa_evaluate.py).
Figures and tables hold aggregates only (no transcript text).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.caa_pilot import config_arg, load_config, load_split, read_jsonl, write_json

GROUPS = ["Natural", "PT base", "PT steered"]
CONTROL = "Random control"
COLORS = {"Natural": "#2a78d6", "PT base": "#eb6834", "PT steered": "#1baf7a", CONTROL: "#6250d6"}
COND = "orpheus-pt"

# (key, label, unit, higher_is) for every figure / table metric. Aggregate = median unless noted.
METRICS = [
    ("wer", "WER", "fraction", None), ("cer", "CER", "fraction", None),
    ("asr_word_rate", "ASR-large word rate", "words/s", None),
    ("asr_articulation_proxy", "ASR-large articulation proxy", "words/s", None),
    ("nuclei_per_s", "Syllable-nucleus rate", "nuclei/s", None),
    ("nuclei_per_sounding_s", "Nuclei / estimated sounding time", "nuclei/s", None),
    ("duration_s", "Duration", "s", None),
    ("f0_median_hz", "Median F0", "Hz", None), ("f0_range_st", "F0 range", "semitones", None),
    ("pause_pct", "Internal pause proportion", "%", None), ("rms_dbfs", "RMS level", "dBFS", None),
    ("jitter", "Praat jitter", "fraction", None), ("shimmer", "Praat shimmer", "fraction", None),
    ("hnr", "Praat HNR", "dB", None), ("nisqa", "NISQA-TTS predicted naturalness", "MOS (pred.)", None),
    ("M1", "M1 natural-vs-PT logit", "log-odds", "higher"), ("M1_p", "M1 P(natural)", "prob.", "higher"),
    ("M2", "M2 cosine to paired natural", "cosine", "higher"), ("M3", "M3 prosody distance", "IQR units", "lower"),
    ("ssim", "S-SIM to PT base (same seed)", "cosine", None),
]


# ------------------------------------------------------------------------------------------ helpers
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def clips(run):
    """[{key, group, utterance_id, path, gen}] for Natural, PT base, PT steered and the random control."""
    split = load_split()
    uids = sorted(set(split["train"]) | set(split["test"]) | set(split["dropped_token_cap"]))
    out = [dict(key=f"nat|{u}", group="Natural", utterance_id=u, path=str(run / "natural24k" / f"{u}.wav"), gen=None)
           for u in uids]
    lab = {("base", "base"): "PT base", ("final", "v_star"): "PT steered", ("final", "rand0_star"): CONTROL}
    for g in read_jsonl(run / "generations.jsonl"):
        grp = lab.get((g["stage"], g["label"]))
        if grp and g["condition"] == COND:
            out.append(dict(key=g["job_id"], group=grp, utterance_id=g["utterance_id"],
                            path=str(run / g["wav"]) if g.get("wav") else None, gen=g))
    return out


# ------------------------------------------------------------------------------------------ NISQA
def nisqa(a, cfg):
    import hashlib
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import learned_evaluate
    out_file = a.run / "metrics" / "nisqa.jsonl"
    done = {r["key"] for r in read_jsonl(out_file)}
    import soundfile as sf
    rows, short = [], []
    for c in clips(a.run):
        if c["path"] and Path(c["path"]).exists() and c["key"] not in done:
            if sf.info(c["path"]).duration < 0.3:  # NISQA needs >= 15 spectrogram windows; keep as missing
                short.append(dict(key=c["key"], nisqa=None, status="too_short_for_nisqa"))
                continue
            rows.append(dict(generation_id=c["key"], utterance_id=c["utterance_id"], configuration_id=c["group"],
                             in_challenge_view=False, source_audio_path=c["path"],
                             audio=dict(audio_sha256=hashlib.sha256(Path(c["path"]).read_bytes()).hexdigest())))
    with out_file.open("a", encoding="utf-8") as f:
        for r in short:
            f.write(json.dumps(r) + "\n")
    print(f"nisqa: {len(rows)} to do, {len(short)} too short (missing)", flush=True)
    if not rows:
        return
    n = len(list((a.run / "metrics" / "nisqa_batches").glob("batch*")))
    out = a.run / "metrics" / "nisqa_batches" / f"batch{n + 1:04}"
    out.mkdir(parents=True)
    learned_evaluate.nisqa(rows, out, a.nisqa_source)
    with out_file.open("a", encoding="utf-8") as f:
        for r in read_jsonl(out / "nisqa_tts.jsonl"):
            f.write(json.dumps(dict(key=r["generation_id"], nisqa=r["nisqa_tts_predicted_mos"],
                                    audio_sha256=r["audio_sha256"], checkpoint_sha256=r["checkpoint_sha256"])) + "\n")


# ------------------------------------------------------------------------------------------ tables
def per_clip(a, cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import caa_evaluate as ce
    ce.CONDS = (COND,)
    m, score, m1info, iqr = ce.analysis_setup(a, cfg)
    load = lambda name: {r["key"]: r for r in read_jsonl(a.run / "metrics" / f"{name}.jsonl")}
    asr, ac, nq, sv = load("asr"), load("acoustics"), load("nisqa"), load("speaker_sv")
    emb = m["embeddings"]
    lo, hi = cfg["guardrails"]["duration_ratio"]
    rows = []
    cl = clips(a.run)
    base = {c["utterance_id"]: c for c in cl if c["group"] == "PT base"}
    for c in cl:
        k, u, g = c["key"], c["utterance_id"], c["gen"]
        ok = g is None or g["status"] == "generated"
        r = dict(key=k, group=c["group"], utterance_id=u, generated=ok)
        if g is not None:
            r.update(eos=g["eos"], hit_max=g["hit_max"], n_tokens=g["n_tokens"], budget=g["budget"],
                     trailing_nonaudio=g.get("trailing_nonaudio"), prefix_ok=g.get("prefix_ids") == [128261, 128257])
        x = asr.get(k, {})
        r.update(wer=x.get("wer"), cer=x.get("cer"), word_errors=x.get("word_errors"), ref_words=x.get("ref_words"),
                 char_errors=x.get("char_errors"), ref_chars=x.get("ref_chars"))
        au, nu = ac.get(k, {}).get("audio", {}), ac.get(k, {}).get("nuclei_rate", {})
        vq = au.get("lab_voice_quality") or {}
        dur, art, nw = au.get("duration_s"), au.get("articulation_time_estimate_s"), x.get("asr_word_count")
        # content_proxies formulas: ASR token count / duration and / articulation-time estimate
        r.update(duration_s=dur, f0_median_hz=au.get("f0_median_hz"), f0_range_st=au.get("f0_range_st"),
                 pause_pct=None if au.get("internal_pause_proportion") is None else 100 * au["internal_pause_proportion"],
                 rms_dbfs=au.get("rms_dbfs"), jitter=vq.get("jitter_local"), shimmer=vq.get("shimmer_local"),
                 hnr=vq.get("hnr_mean_db"), nuclei_per_s=nu.get("nuclei_per_s"),
                 nuclei_per_sounding_s=nu.get("nuclei_per_detected_sounding_s"),
                 asr_word_rate=nw / dur if dur and nw is not None else None,
                 asr_articulation_proxy=nw / art if art and nw is not None else None,
                 nisqa=nq.get(k, {}).get("nisqa"))
        if ok and k in emb and f"nat|{u}" in emb:
            e, n = np.array(emb[k]["wavlm"]), np.array(emb[f"nat|{u}"]["wavlm"])
            r["M1"] = float(score([k])[0])
            r["M1_p"] = float(1 / (1 + np.exp(-r["M1"])))
            r["M2"] = None if c["group"] == "Natural" else float(e @ n / (np.linalg.norm(e) * np.linalg.norm(n)))
        if c["group"] != "Natural" and ok:
            r["M3"] = ce.m3_distance(m, iqr, k, u)[0]
        b = base.get(u)
        if c["group"] in ("PT steered", CONTROL) and b and ok and b["key"] in emb and k in emb:
            r["ssim"] = float(np.dot(emb[k]["sv"], emb[b["key"]]["sv"]))
        if g is not None:
            bd = b["gen"].get("duration_s") if b and c["group"] != "PT base" else None
            dur_bad = bool(bd and g.get("duration_s") is not None and not (lo <= g["duration_s"] / bd <= hi))
            r["failure"] = bool((not ok) or (not g["eos"]) or g["hit_max"] or dur_bad)
        else:
            r["failure"] = False
        r["sv"] = sv.get(k, {}).get("sv")
        rows.append(r)
    return rows, m1info


def tables(a, cfg):
    from src.caa_stats import bootstrap_ci, wilcoxon
    from src.evaluation.learned import speaker_consistency
    rows, m1info = per_clip(a, cfg)
    split = load_split()
    sets = {"test19": set(split["test"]), "all95_insample": set(split["train"]) | set(split["test"]),
            "all100_insample": set(split["train"]) | set(split["test"]) | set(split["dropped_token_cap"])}
    keys = [mk for mk, *_ in METRICS]
    out_t = a.run / "tables"
    write_csv(a.run / "metrics" / "clips_pt.csv", [{k: v for k, v in r.items() if k != "sv"} for r in rows])
    fig_data, all_stats = {}, {}
    for sname, uset in sets.items():
        summary = []
        groups = GROUPS + ([CONTROL] if sname == "test19" else [])
        by = {g: [r for r in rows if r["group"] == g and r["utterance_id"] in uset] for g in groups}
        for g, rs in by.items():
            s = dict(set=sname, group=g, n_clips=len(rs), n_generated=sum(r["generated"] for r in rs),
                     failures=sum(r["failure"] for r in rs), failure_rate=float(np.mean([r["failure"] for r in rs])))
            we, ww = sum(r["word_errors"] or 0 for r in rs if r["wer"] is not None), sum(r["ref_words"] or 0 for r in rs if r["wer"] is not None)
            ce_, cc = sum(r["char_errors"] or 0 for r in rs if r["cer"] is not None), sum(r["ref_chars"] or 0 for r in rs if r["cer"] is not None)
            s["wer_corpus"], s["cer_corpus"] = (we / ww if ww else None), (ce_ / cc if cc else None)
            for k in keys:
                v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
                s[f"{k}_n"] = len(v)
                s[f"{k}_median"] = float(np.median(v)) if v else None
                s[f"{k}_mean"] = float(np.mean(v)) if v else None
            vec = [r["sv"] for r in rs if r.get("sv")]
            if len(vec) >= 3:
                sc = speaker_consistency(np.array(vec), repeats=1000)
                s["speaker_consistency"] = sc["pairwise_mean_cosine"]
                s["speaker_consistency_ci"] = sc["pairwise_mean_bootstrap_95ci"]
            summary.append(s)
        write_csv(out_t / f"summary_{sname}.csv", [dict(s, speaker_consistency_ci=json.dumps(s.get("speaker_consistency_ci")))
                                                    for s in summary])
        # paired deltas per utterance
        idx = {(r["group"], r["utterance_id"]): r for r in rows if r["utterance_id"] in uset}
        stats = []
        pairs = [("PT steered", "PT base")]
        if sname == "test19":
            pairs += [(CONTROL, "PT base"), ("PT steered", CONTROL)]
        for A, B in pairs:
            for k in keys + ["failure"]:
                d = []
                for u in sorted(uset):
                    x, y = idx.get((A, u)), idx.get((B, u))
                    if x and y and x.get(k) is not None and y.get(k) is not None:
                        d.append(float(x[k]) - float(y[k]))
                if not d or k == "ssim":
                    continue
                lo_, hi_ = bootstrap_ci(d, seed=7)
                w = wilcoxon(d)
                stats.append(dict(set=sname, contrast=f"{A} - {B}", metric=k, n_utt=len(d), mean_diff=float(np.mean(d)),
                                  median_diff=float(np.median(d)), ci_low=lo_, ci_high=hi_, wilcoxon_n=w["n"],
                                  wilcoxon_p=w["p"]))
        write_csv(out_t / f"deltas_{sname}.csv", stats)
        # guardrails (test set: steered vs base, same rule as the dev selection)
        sm = {s["group"]: s for s in summary}
        gr = cfg["guardrails"]
        b_, s_ = sm["PT base"], sm["PT steered"]
        guard = {} if not s_["n_clips"] else dict(wer=s_["wer_corpus"] <= b_["wer_corpus"] + gr["wer_points"],
                     cer=s_["cer_corpus"] <= b_["cer_corpus"] + gr["wer_points"],
                     ssim=(s_["ssim_mean"] or 0) >= gr["ssim_min"],
                     failure=s_["failure_rate"] <= max(gr["failure_max"], b_["failure_rate"]))
        all_stats[sname] = dict(guardrails=guard, deltas=stats)
        fig_data[sname] = dict(summary=summary)
    write_json(out_t / "m1_classifier.json", m1info)
    write_json(out_t / "stats.json", all_stats)
    write_json(out_t / "figure_data.json", dict(groups=GROUPS, colors=COLORS, metrics=METRICS, sets=fig_data))
    for sname in sets:
        for s in fig_data[sname]["summary"]:
            print(sname, s["group"], "n", s["n_clips"], "fail", s["failures"], "WERmed", s["wer_median"],
                  "NISQA", s["nisqa_median"], "M1", s["M1_mean"], "spk", s.get("speaker_consistency"))


# ------------------------------------------------------------------------------------------ figures
FIGS = [  # (file, title, [(metric, stat)], ylabel)
    ("01_wer", "Whisper-large-v3 WER by group", [("wer", "median"), ("wer", "mean")], "WER"),
    ("02_speaking_rate", "Speaking rate by group (medians)",
     [("asr_word_rate", "median"), ("asr_articulation_proxy", "median"), ("nuclei_per_s", "median"),
      ("nuclei_per_sounding_s", "median")], "Rate (per second)"),
    ("03_duration", "Median duration by group", [("duration_s", "median")], "Duration (s)"),
    ("04_prosody", "Prosody by group (medians)",
     [("f0_median_hz", "median"), ("f0_range_st", "median"), ("pause_pct", "median"), ("rms_dbfs", "median")], None),
    ("05_voice_quality", "Voice quality (Praat) by group (medians)",
     [("jitter", "median"), ("shimmer", "median"), ("hnr", "median")], None),
    ("06_nisqa", "NISQA-TTS predicted naturalness by group", [("nisqa", "median")], "Median NISQA-TTS prediction"),
    ("07_speaker_consistency", "WavLM within-group speaker consistency", [("speaker_consistency", None)],
     "Mean pairwise cosine"),
    ("08_naturalness_m1_m3", "CAA naturalness metrics by group (means)",
     [("M1", "mean"), ("M2", "mean"), ("M3", "mean")], None),
]


def figures(a, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = json.loads((a.run / "tables" / "figure_data.json").read_text())
    labels = {k: (lab, unit) for k, lab, unit, _ in fd["metrics"]}
    labels["speaker_consistency"] = ("Speaker consistency", "cosine")
    out = a.run / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    set_names = {"test19": "test set (19 utterances)", "all100_insample": "all 100 utterances (in-sample)"}
    written = []
    for sname, desc in set_names.items():
        summ = {s["group"]: s for s in fd["sets"][sname]["summary"]}
        groups = [g for g in fd["groups"] if g in summ]
        for fname, title, panels, ylabel in FIGS:
            fig, axes = plt.subplots(1, len(panels), figsize=(max(5.6, 3.3 * len(panels)), 4.2), squeeze=False)
            for ax, (mk, stat) in zip(axes[0], panels):
                key = mk if stat is None else f"{mk}_{stat}"
                vals = [summ[g].get(key) for g in groups]
                if mk in ("M2", "M3"):  # paired-natural metrics are undefined for Natural itself
                    keep = [i for i, g in enumerate(groups) if g != "Natural"]
                else:
                    keep = list(range(len(groups)))
                gs, vs = [groups[i] for i in keep], [vals[i] for i in keep]
                xs = np.arange(len(gs))
                vv = [np.nan if v is None else v for v in vs]
                bars = ax.bar(xs, vv, width=0.62, color=[fd["colors"][g] for g in gs], edgecolor="#fcfcfb", linewidth=2)
                for b, v in zip(bars, vs):
                    if v is not None:
                        y = b.get_height()
                        ax.annotate(f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}", (b.get_x() + b.get_width() / 2, y),
                                    xytext=(0, 3 if y >= 0 else 4), textcoords="offset points", ha="center",
                                    fontsize=9, color="#0b0b0b")
                ax.set_xticks(xs, gs)
                lab, unit = labels[mk]
                dup = sum(p_[0] == mk for p_ in panels) > 1  # e.g. WER median and mean side by side
                ax.set_title(f"{lab}" + (f" ({stat})" if dup else ""), fontsize=10)
                ax.set_ylabel(ylabel or unit)
                ax.axhline(0, color="#52514e", lw=0.6)
                ax.grid(axis="y", color="#e6e5e1", lw=0.6)
                ax.set_axisbelow(True)
                if mk == "speaker_consistency":
                    lo = min(v for v in vv if np.isfinite(v))
                    ax.set_ylim(max(0, lo - 0.1), 1.0)
            fig.suptitle(f"{title}\n{desc}", fontsize=11)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                p = out / f"{fname}_{sname}.{ext}"
                fig.savefig(p, dpi=160)
                written.append(p.name)
            plt.close(fig)
    print("\n".join(written))


# ------------------------------------------------------------------------------------------ listening
def listening(a, cfg):
    import hashlib
    import html
    import shutil
    from urllib.parse import quote
    src, out = Path(cfg["data_dir"]), a.out
    if out.exists() and any(out.iterdir()) and not a.force:
        raise SystemExit(f"{out} exists and is not empty (use --force to refresh)")
    out.mkdir(parents=True, exist_ok=True)
    split = load_split()
    tag = {u: "test" for u in split["test"]}
    tag.update({u: "train" for u in split["train"]})
    tag.update({u: "train (dev)" for u in split["dev"]})
    tag.update({u: "excluded (pilot token-cap)" for u in split["dropped_token_cap"]})
    sel = json.loads((a.run / "selection" / "selection.json").read_text())[COND]
    with (src / "manifest.csv").open(encoding="utf-8-sig") as f:
        man = list(csv.DictReader(f))
    with (src / "utterances_100.csv").open(encoding="utf-8-sig") as f:
        utts = list(csv.DictReader(f))
    gens = {}
    for g in read_jsonl(a.run / "generations.jsonl"):
        if (g["stage"], g["label"]) in (("base", "base"), ("final", "v_star")):
            gens[(g["utterance_id"], g["label"])] = g
    with (a.run / "metrics" / "clips_pt.csv").open(encoding="utf-8") as f:
        fail = {r["key"]: r["failure"] == "True" for r in csv.DictReader(f)}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest, sections = [], []
    spec = json.dumps(dict(sites=sel["sites"], alpha=sel["alpha_star"], vector="v = mean(natural) - mean(PT base), train",
                           injection="norm-preserving, all positions, attention o_proj output"))
    for u in utts:
        folder, uid = u["folder"], u["utterance_id"]
        d = out / folder
        d.mkdir(exist_ok=True)
        nat = next(r for r in man if r["utterance_id"] == uid and r["configuration_id"].startswith("natural/"))
        shutil.copy2(src / folder / nat["filename"], d / nat["filename"])
        manifest.append(dict(folder=folder, filename=nat["filename"], utterance_id=uid, split=tag[uid],
                             source_path=str(src / folder / nat["filename"]), job_id="", seed="", steering="",
                             failure_flag="", SHA256=sha(d / nat["filename"])))
        files = [nat["filename"]]
        for lab, name, steer in (("base", "02_OrpheusPT_base.wav", "none (alpha=0)"), ("v_star", "03_OrpheusPT_steered.wav", spec)):
            g = gens[(uid, lab)]
            shutil.copy2(a.run / g["wav"], d / name)
            manifest.append(dict(folder=folder, filename=name, utterance_id=uid, split=tag[uid],
                                 source_path=str(a.run / g["wav"]), job_id=g["job_id"], seed=g["seed"], steering=steer,
                                 failure_flag=fail.get(g["job_id"]), SHA256=sha(d / name)))
            files.append(name)
        t = (src / folder / "TRANSCRIPT.txt").read_text(encoding="utf-8-sig")
        t += ("\n--- Orpheus PT CAA follow-up (" + a.run.name + ") ---\n"
              "Files in THIS folder: 01 natural recording (unchanged copy); 02_OrpheusPT_base.wav = Orpheus Hindi "
              "pretrained (canopylabs/3b-hi-pretrain-research_release), text-only prompt, Devanagari input, seed 1234, "
              "no steering; 03_OrpheusPT_steered.wav = same model, input and seed with the CAA naturalness vector added "
              f"(sites {', '.join('L%d %s-%s' % (s['layer'], s['site'], s['pool']) for s in sel['sites'])}, "
              f"alpha* = {sel['alpha_star']}). The PT model has no named voice, so the speaker varies per clip. "
              f"Split tag: {tag[uid]}. The lines above about 02-06 describe the source folder, not this one.\n")
        (d / "TRANSCRIPT.txt").write_text(t, encoding="utf-8")
        flags = "".join(f" <b>[failure flag: {n.split('_')[1]}]</b>" for n, k in
                        (("02_base", gens[(uid, 'base')]["job_id"]), ("03_steered", gens[(uid, 'v_star')]["job_id"]))
                        if fail.get(k))
        cells = "".join(f'<td>{html.escape(fn)}<br><audio controls preload="none" src="{quote(folder)}/{quote(fn)}"></audio></td>'
                        for fn in files)
        sections.append(f'<section data-split="{html.escape(tag[uid])}"><h2>{html.escape(uid)} '
                        f'<span class="tag">{html.escape(tag[uid])}</span>{flags}</h2>'
                        f'<p>{html.escape(u["Romanized"])}</p><p lang="hi">{html.escape(u["Devanagari"])}</p>'
                        f"<table><tr>{cells}</tr></table></section>")
    shutil.copy2(src / "utterances_100.csv", out / "utterances_100.csv")
    write_csv(out / "manifest.csv", manifest)
    sites = ", ".join(f"layer {s['layer']} {s['site']} ({s['pool']}-pooled v)" for s in sel["sites"])
    (out / "READ_ME.txt").write_text(
        f"100 listening examples for the CAA follow-up on Orpheus Hindi PRETRAINED (run {a.run.name}).\n"
        f"Run directory: {a.run}\n"
        "Same 100 folder names as Listening_Examples_100. Each folder: 01_Natural_<SPK>.wav (byte-identical copy of the\n"
        "original natural recording), 02_OrpheusPT_base.wav (no steering), 03_OrpheusPT_steered.wav (CAA vector),\n"
        "TRANSCRIPT.txt (copied, with a note on the PT files appended).\n"
        "Model: canopylabs/3b-hi-pretrain-research_release @ b813fee6; text-only prompt (no voice name, so the speaker\n"
        "varies per clip); Devanagari/English input; seed 1234; t 0.6, top_p 0.9, repetition penalty 1.1.\n"
        f"Steering: v = mean(natural) - mean(PT base) over the train utterances' teacher-forced activations;\n"
        f"selected sites: {sites}; alpha* = {sel['alpha_star']}; norm-preserving, all positions.\n"
        "Split tags (listen.html): test = held-out (19), train / train (dev) = used to build v (dev: used to pick\n"
        "sites/alpha), excluded = the 5 pilot token-cap utterances (not used for v or statistics).\n"
        "manifest.csv: per wav its source path, job id, seed, steering spec, failure flag and SHA-256.\n"
        "Unblinded researcher listening page; not a rating instrument. Contains MaSaC audio and transcripts: kedi only.\n",
        encoding="utf-8")
    page = ('<!doctype html><meta charset="utf-8"><title>Listening Examples 100 PT</title><style>body{font:16px system-ui;'
            'margin:24px}table{display:block;overflow:auto}audio{width:230px}td{padding:8px;border:1px solid #ccc}'
            'section{margin:30px 0}.tag{font-size:13px;font-weight:normal;padding:2px 8px;border-radius:9px;'
            'background:#e5edf5}</style><h1>100 listening examples: Natural vs Orpheus PT base vs Orpheus PT steered</h1>'
            f'<p>Run {html.escape(a.run.name)}. Steering sites: {html.escape(sites)}; alpha* = {sel["alpha_star"]}. '
            'Unblinded researcher review; PT speaker varies per clip.</p>'
            '<p><button onclick="f(\'test\')">Test only</button> <button onclick="f(\'\')">All 100</button></p>'
            '<script>function f(t){document.querySelectorAll("section").forEach(s=>s.hidden=t&&s.dataset.split!==t)}</script>'
            + "".join(sections))
    (out / "listen.html").write_text(page, encoding="utf-8")
    print(f"{len(utts)} folders, {len(manifest)} wavs -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["nisqa", "tables", "figures", "listening"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--nisqa-source", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--force", action="store_true")
    config_arg(p)
    a = p.parse_args()
    cfg = load_config(a.config)
    dict(nisqa=nisqa, tables=tables, figures=figures, listening=listening)[a.action](a, cfg)
