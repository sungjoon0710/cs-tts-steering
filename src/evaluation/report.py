from __future__ import annotations

import csv
import html
import itertools
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .common import read_jsonl, write_json


def paired_bootstrap(a, b, seed=1234, repeats=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or not len(a):
        raise ValueError("Nonempty paired observations required")
    diff = b - a
    draws = np.random.default_rng(seed).integers(0, len(diff), size=(repeats, len(diff)))
    means = diff[draws].mean(axis=1)
    return {"n_utterances": len(diff), "mean_B_minus_A": float(diff.mean()),
            "median_B_minus_A": float(np.median(diff)),
            "mean_difference_bootstrap_95ci": np.quantile(means, [.025, .975]).tolist(),
            "status": "exploratory_pilot; utterance bootstrap; conversation dependence unresolved"}


def value(row, metric):
    obj = row
    for key in metric.split("."):
        if not isinstance(obj, dict): return None
        obj = obj.get(key)
    return obj if isinstance(obj, (int, float)) and np.isfinite(obj) else None


def display_config(config):
    # Plot-only transliteration avoids Matplotlib's unsupported Devanagari shaping.
    return "Orpheus Hindi / Ritika" if config.startswith("orpheus-hi/") else config.replace("/", " / ")


def summarize_judgments(rows):
    counts = defaultdict(Counter)
    for r in rows:
        # Store responses per pair; do not pool voice-confounded comparisons.
        pair = tuple(sorted((r["configuration_A"], r["configuration_B"])))
        response = r["naturalness"]
        if response in ("a", "b"):
            chosen = r["configuration_A" if response == "a" else "configuration_B"]
            response = "first_configuration" if chosen == pair[0] else "second_configuration"
        counts[pair][response] += 1
    results = []
    for pair, c in sorted(counts.items()):
        subset = [r for r in rows if tuple(sorted((r["configuration_A"], r["configuration_B"]))) == pair]
        listeners = sorted({r.get("listener_id", "unknown") for r in subset})
        items = sorted({r.get("utterance_id", "unknown") for r in subset})
        out = {"pair": list(pair), "categories": {k: c[k] for k in
               ("first_configuration", "second_configuration", "both_acceptable", "both_unacceptable")},
               "judgments": sum(c.values()), "listeners": len(listeners), "utterances": len(items)}
        if len(listeners) >= 3 and len(items) >= 3:
            li = np.array([listeners.index(r.get("listener_id", "unknown")) for r in subset])
            ui = np.array([items.index(r.get("utterance_id", "unknown")) for r in subset])
            category = []
            for r in subset:
                choice = r['naturalness']
                if choice in ('a','b'):
                    chosen = r['configuration_A' if choice == 'a' else 'configuration_B']
                    choice = 'first_configuration' if chosen == pair[0] else 'second_configuration'
                category.append(choice)
            rng = np.random.default_rng(1234); draws = defaultdict(list)
            for _ in range(2000):
                lc = np.bincount(rng.integers(len(listeners),size=len(listeners)),minlength=len(listeners))
                uc = np.bincount(rng.integers(len(items),size=len(items)),minlength=len(items))
                weights = lc[li]*uc[ui]
                if not weights.sum(): continue
                for cat in out['categories']:
                    draws[cat].append(float(weights[np.array(category)==cat].sum()/weights.sum()))
            out['category_proportion_95ci'] = {k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()}
            out['analysis_status'] = 'exploratory crossed listener-and-utterance bootstrap; small-cluster uncertainty remains'
        else:
            out['analysis_status'] = 'insufficient listener/item clusters for uncertainty; descriptive counts only'
        results.append(out)
    return results


def report(run):
    rows = read_jsonl(run / "tables/metrics.jsonl")
    pilot = [r for r in rows if r["cohort"] == "pilot"]
    figures = run / "report/figures"; figures.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in pilot: groups[r["configuration_id"]].append(r)
    metrics = ("automatic_proxy.wer", "automatic_proxy.cer", "audio.duration_s", "audio.f0_range_st",
               "audio.internal_pause_proportion", "audio.rms_dbfs")
    summaries, comparisons = [], []
    for config, rs in sorted(groups.items()):
        entry = {"configuration_id": config, "recordings": len(rs),
                 "audio_status_counts": dict(Counter(r["audio"]["status"] for r in rs)),
                 "token_limit_warnings": sum(bool(r["hit_token_limit"]) for r in rs),
                 "reviewed_recordings": sum(r.get("failure_review", {}).get("status") == "human_reviewed" for r in rs),
                 "confirmed_failures": sum(r.get("failure_any_confirmed") is True for r in rs),
                 "confirmed_failure_lower_bound": sum(r.get("failure_any_confirmed") is True for r in rs) / len(rs),
                 "failure_rate_status": "incomplete_until_all_recordings_reviewed" if any(r.get("failure_any_confirmed") is None for r in rs) else "review_complete"}
        entry["metrics"] = {}
        for metric in metrics:
            vs = [value(r, metric) for r in rs]; vs = [v for v in vs if v is not None]
            entry["metrics"][metric] = {"available": len(vs), "median": float(np.median(vs)) if vs else None,
                                        "mean": float(np.mean(vs)) if vs else None}
        valid = [r["automatic_proxy"] for r in rs if r["automatic_proxy"].get("wer") is not None]
        entry["corpus_proxy_wer"] = sum(x["substitutions"] + x["deletions"] + x["insertions"] for x in valid) / sum(x["reference_words"] for x in valid) if valid else None
        entry["corpus_proxy_cer"] = sum(x["character_errors"] for x in valid) / sum(x["reference_characters"] for x in valid) if valid else None
        summaries.append(entry)
    for ca, cb in itertools.combinations(sorted(groups), 2):
        aa = {r["utterance_id"]: r for r in groups[ca]}; bb = {r["utterance_id"]: r for r in groups[cb]}
        for metric in metrics:
            ids = sorted(k for k in aa.keys() & bb.keys() if value(aa[k], metric) is not None and value(bb[k], metric) is not None)
            if ids:
                comparisons.append({"configuration_A": ca, "configuration_B": cb, "metric": metric,
                    "utterance_ids": ids, **paired_bootstrap([value(aa[k], metric) for k in ids], [value(bb[k], metric) for k in ids])})
    judgments = read_jsonl(run / "human/judgments.jsonl")
    output = {"status": "AUTOMATED_PILOT_EXECUTED; BILINGUAL_VALIDATION_PENDING",
        "note": "No model-quality ranking. ASR checkpoint unknown. Script strata are not language labels.",
        "configurations": summaries, "paired_comparisons": comparisons,
        "human_results": summarize_judgments(judgments) if judgments else {"status": "pending", "ratings": 0},
        "pending_counts": {"manual_transcription": sum(r["synthesis_content"]["wer"] is None for r in pilot),
                           "pier": sum(r["poi"]["status"].startswith("pending") for r in pilot),
                           "articulation_rate": sum(r["rates"]["articulation_wps"] is None for r in pilot)}}
    write_json(run / "report/summary.json", output)
    fieldnames = ["generation_id", "utterance_id", "configuration_id", "cohort", "script_profile", "input_word_count", *metrics]
    with (run / "report/metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames); writer.writeheader()
        for r in rows: writer.writerow({k: value(r, k) if "." in k else r.get(k) for k in fieldnames})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, metrics):
        for config, rs in sorted(groups.items()):
            vs = sorted(v for r in rs if (v := value(r, metric)) is not None)
            if vs: ax.step(vs, np.arange(1, len(vs) + 1) / len(vs), where="post", label=display_config(config))
        ax.set(xlabel=metric, ylabel="Empirical cumulative fraction"); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Pilot distributions — automatic proxies and descriptive acoustics, NOT quality ranks")
    fig.tight_layout(); fig.savefig(figures / "distributions.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    for ax, metric in zip(axes.flat,metrics):
        cs=[c for c in comparisons if c['metric']==metric]
        for i,c in enumerate(cs):
            v=c['mean_B_minus_A'];lo,hi=c['mean_difference_bootstrap_95ci']
            ax.errorbar(v,i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color='#194e8a')
        ax.set_yticks(range(len(cs)),[display_config(c['configuration_B'])+' minus\n'+display_config(c['configuration_A']) for c in cs],fontsize=7)
        ax.axvline(0,color='gray',linewidth=.7);ax.set_xlabel(metric);ax.grid(alpha=.15)
    fig.suptitle('Paired pilot differences, mean and exploratory 95% utterance-bootstrap interval')
    fig.tight_layout();fig.savefig(figures/'paired_differences.png',dpi=140);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for config, rs in sorted(groups.items()):
        eligible = [r for r in rs if value(r, "automatic_proxy.wer") is not None]
        axes[0].scatter([r["input_word_count"] for r in eligible], [value(r, "automatic_proxy.wer") for r in eligible], label=display_config(config), alpha=.6)
        axes[1].scatter([value(r, "audio.duration_s") for r in eligible], [value(r, "automatic_proxy.wer") for r in eligible], alpha=.6)
    axes[0].set(xlabel="Intended normalized word count", ylabel="Unvalidated ASR-proxy WER")
    axes[1].set(xlabel="Duration (seconds)", ylabel="Unvalidated ASR-proxy WER")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "difficulty.png", dpi=140); plt.close(fig)
    labeled=[r for r in pilot if r['language']['status']=='verified']
    output['language_difficulty_status']='computed_from_verified_labels' if labeled else 'pending_verified_token_language_annotations'
    if labeled:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for config in sorted(groups):
            rs=[r for r in labeled if r['configuration_id']==config and r['automatic_proxy'].get('wer') is not None]
            for ax,feature in zip(axes,('english_fraction','switch_count')):
                rs_feature=[r for r in rs if r['language'].get(feature) is not None]
                ax.scatter([r['language'][feature] for r in rs_feature],[r['automatic_proxy']['wer'] for r in rs_feature],label=display_config(config),alpha=.6)
                ax.set(xlabel=feature,ylabel='Unvalidated ASR-proxy WER')
        axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/'language_difficulty.png',dpi=140);plt.close(fig)
    output['human_automatic_disagreement_status']='pending_human_ratings'
    if judgments:
        paired=defaultdict(Counter)
        for r in judgments:
            ca,cb=sorted((r['configuration_A'],r['configuration_B']))
            if r['naturalness'] in ('a','b'):
                chosen=r['configuration_A' if r['naturalness']=='a' else 'configuration_B']
                paired[(r['utterance_id'],ca,cb)][chosen]+=1
        by_key={(r['utterance_id'],r['configuration_id']):r for r in pilot};points=[]
        for (uid,ca,cb),counts in paired.items():
            if (uid,ca) not in by_key or (uid,cb) not in by_key:continue
            va=value(by_key[(uid,ca)],'automatic_proxy.wer');vb=value(by_key[(uid,cb)],'automatic_proxy.wer')
            if va is None or vb is None:continue
            points.append({'utterance_id':uid,'configuration_A':ca,'configuration_B':cb,
                           'proxy_wer_B_minus_A':vb-va,'human_B_minus_A_margin':(counts[cb]-counts[ca])/sum(counts.values())})
        output['human_automatic_disagreement_status']='exploratory conditional A/B margins; ties remain in separate summaries'
        output['human_automatic_points']=points
        if points:
            fig,ax=plt.subplots(figsize=(8,5))
            ax.scatter([p['proxy_wer_B_minus_A'] for p in points],[p['human_B_minus_A_margin'] for p in points])
            ax.set(xlabel='ASR-proxy WER: B minus A (lower favors B)',ylabel='Conditional preference margin (higher favors B)')
            ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);fig.tight_layout();fig.savefig(figures/'human_automatic.png',dpi=140);plt.close(fig)
    # Include failures/challenges deliberately, not only attractive examples.
    selected = sorted({r["utterance_id"] for r in rows if r["hit_token_limit"]}, key=int)[:2]
    selected += [u for u in sorted({r["utterance_id"] for r in pilot}, key=int) if u not in selected][:2]
    for uid in selected:
        examples = sorted([r for r in rows if r["utterance_id"] == uid], key=lambda r: r["configuration_id"])
        fig, axes = plt.subplots(3, len(examples), figsize=(15, 8), squeeze=False)
        for col, r in enumerate(examples):
            g = next(g for g in read_jsonl(run / "tables/generations.jsonl") if g["generation_id"] == r["generation_id"])
            path = run / "source/audio" / g["relative_audio_path"]
            if not path.exists(): continue
            x, sr = sf.read(path); x = x if x.ndim == 1 else x[:, 0]
            axes[0,col].plot(np.arange(len(x)) / sr, x, linewidth=.3)
            wer = r["automatic_proxy"].get("wer")
            axes[0,col].set_title(display_config(r["configuration_id"]) + "\nASR-proxy WER=" + (str(round(wer,3)) if wer is not None else "missing"))
            for interval in r["audio"].get("intensity_intervals", []):
                if interval["label"] == "silent": axes[0,col].axvspan(interval["start_s"], interval["end_s"], color="orange", alpha=.2)
            if len(x) >= 512: axes[1,col].specgram(x, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
            track = r["audio"].get("f0_track", [])
            axes[2,col].plot([p["time_s"] for p in track], [p["hz"] if p["hz"] is not None else np.nan for p in track], linewidth=.8)
            axes[2,col].set(xlabel="Seconds", ylabel="F0 (Hz)")
        fig.suptitle("Source line " + uid + " — orange: intensity-based silence candidates; manual validation pending")
        fig.tight_layout(); fig.savefig(figures / ("example_" + uid + ".png"), dpi=140); plt.close(fig)
    sections = ["<h1>TTS evaluation: automated pilot</h1>", "<p>Human listening, transcription, language labels, and segmentation validation remain pending. These plots do not establish model quality.</p>",
                "<p>All three configurations are retained. Cross-model comparisons are voice-confounded.</p>"]
    from urllib.parse import quote
    for uid in selected:
        sections.append('<h2>Inspect source line ' + html.escape(uid) + '</h2>')
        for r in (r for r in rows if r['utterance_id']==uid):
            g=next(g for g in read_jsonl(run/'tables/generations.jsonl') if g['generation_id']==r['generation_id'])
            sections.append('<h3>'+html.escape(display_config(r['configuration_id']))+'</h3><audio controls preload="none" src="../source/audio/'+quote(g['relative_audio_path'],safe='/')+'"></audio>')
            sections.append('<p><b>Intended text:</b> '+html.escape(r['text_raw'])+'</p><p><b>Legacy ASR:</b> '+html.escape(r.get('asr_raw') or '[missing]')+'</p>')
            sections.append('<details><summary>Word-edit operations and statuses</summary><pre>'+html.escape(json.dumps({
                'alignment':r['automatic_proxy'].get('alignment'), 'failure_review':r.get('failure_review'),
                'POI':r['poi'], 'audio_status':r['audio']['status']},ensure_ascii=False,indent=2))+'</pre></details>')
    for p in sorted(figures.glob("*.png")):
        sections.append('<h2>' + html.escape(p.stem) + '</h2><img style="max-width:100%" src="figures/' + html.escape(p.name) + '">')
    sections.append("<h2>Auditable summaries</h2><pre>" + html.escape(json.dumps(output, ensure_ascii=False, indent=2)) + "</pre>")
    write_json(run / "report/summary.json", output)
    (run / "report/index.html").write_text('<!doctype html><meta charset="utf-8"><title>TTS pilot</title>' + "\n".join(sections), encoding="utf-8")
    print("Wrote private report with distributions, paired uncertainty, challenge examples, and pending statuses.")
