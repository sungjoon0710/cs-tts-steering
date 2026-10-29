"""CAA pilot steered generation (plan §5): job planning and a resumable, seeded batched runner.

  plan  --run RUN --stage {sweep1,sweep2,test}   writes RUN/jobs/<stage>.jsonl (no text; ids + steering spec)
  run   --run RUN --stage S --condition C        generates missing clips; appends RUN/generations.jsonl

Steering spec per job: list of {layer, site, pool, vec, alpha} with vec in {v, neg, rand0, rand1, ...}.
Random vectors: Gaussian direction scaled to ||v|| at the same site, seeded by (condition, site, draw).
Batches mix jobs of one condition and one seed; the batch seed is derived from the job ids, so a clip
is reproducible given the same batch composition (recorded per clip).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.caa_pilot import (append_jsonl, condition_voices, hf_snapshot, load_config, load_data, load_split,
                           read_jsonl, stable_id, write_json)

CONDS = ("indic-amit", "indic-kavya", "orpheus-ritika")


def job(stage, cond, uid, seed, label, steer):
    return dict(job_id=stable_id(stage, cond, uid, seed, steer), stage=stage, condition=cond, utterance_id=uid,
                seed=seed, label=label, steer=steer)


def site_label(s):
    return f"L{s['layer']}{s['site']}-{s['pool']}"


def plan(a, cfg):
    split = load_split()
    sc = cfg["steering"]
    jobs = []
    if a.stage == "sweep1":
        shortlist = json.loads((a.run / "shortlist.json").read_text())
        for cond in CONDS:
            for uid in split["dev"]:
                jobs.append(job("sweep1", cond, uid, cfg["seed"], "base", []))
                for s in shortlist[cond]:
                    for al in sc["sweep1_alphas"]:
                        st = [dict(layer=s["layer"], site=s["site"], pool=s["pool"], vec="v", alpha=al)]
                        jobs.append(job("sweep1", cond, uid, cfg["seed"], f"{site_label(s)}_a{al}", st))
    elif a.stage == "sweep2":
        ranking = json.loads((a.run / "selection" / "sweep1_ranking.json").read_text())
        for cond in CONDS:
            ranked = ranking[cond]  # best first: list of {layer, site, pool}
            for uid in split["dev"]:
                jobs.append(job("sweep2", cond, uid, cfg["seed"], "base", []))
                for k in sc["topk"]:
                    sites = ranked[:k]
                    for al in sc["sweep2_alphas"]:
                        st = [dict(layer=s["layer"], site=s["site"], pool=s["pool"], vec="v", alpha=al) for s in sites]
                        jobs.append(job("sweep2", cond, uid, cfg["seed"], f"top{k}_a{al}", st))
    elif a.stage == "test":
        sel = json.loads((a.run / "selection" / "selection.json").read_text())
        for cond in CONDS:
            s = sel[cond]
            sites, a_star, a_mild = s["sites"], s["alpha_star"], s["alpha_mild"]
            mk = lambda vec, al: [dict(layer=x["layer"], site=x["site"], pool=x["pool"], vec=vec, alpha=al) for x in sites]
            for uid in split["test"]:
                for seed in sc["test_seeds"]:
                    jobs.append(job("test", cond, uid, seed, "base", []))
                    jobs.append(job("test", cond, uid, seed, "v_star", mk("v", a_star)))
                    jobs.append(job("test", cond, uid, seed, "v_mild", mk("v", a_mild)))
                    jobs.append(job("test", cond, uid, seed, "neg_star", mk("neg", a_star)))
                    for d in range(sc["random_draws"]):
                        jobs.append(job("test", cond, uid, seed, f"rand{d}_star", mk(f"rand{d}", a_star)))
    path = a.run / "jobs" / f"{a.stage}.jsonl"
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists; use --force to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs -> {path}")


def vector_bank(run, cond, cfg):
    import torch
    raw = torch.load(run / "vectors" / f"{cond}.pt", weights_only=True)
    cache = {}

    def get(layer, site, pool, vec):
        key = (layer, site, pool, vec)
        if key not in cache:
            v = raw[f"{layer}|{site}|{pool}"].float()
            if vec == "v":
                out = v
            elif vec == "neg":
                out = -v
            elif vec.startswith("rand"):
                g = torch.Generator().manual_seed(int(stable_id("rand", cond, layer, site, pool, vec)[:8], 16))
                r = torch.randn(v.shape, generator=g)
                out = r / r.norm() * v.norm()
            else:
                raise ValueError(vec)
            cache[key] = out
        return cache[key]
    return get


def run(a, cfg):
    import soundfile as sf
    import torch
    from src.steering import SpeechLM, Steerer, generate_batch
    utts = load_data(cfg)
    voice = condition_voices(cfg, utts)[a.condition]
    kind = cfg["conditions"][a.condition]["model"]
    jobs = [j for j in read_jsonl(a.run / "jobs" / f"{a.stage}.jsonl") if j["condition"] == a.condition]
    log = a.run / "generations.jsonl"
    done = {r["job_id"] for r in read_jsonl(log)}
    todo = [j for j in jobs if j["job_id"] not in done]  # resumable: logged clips (incl. failures) are final
    print(f"{a.condition}/{a.stage}: {len(jobs)} jobs, {len(todo)} to do", flush=True)
    if not todo:
        return
    pins = cfg["pins"]
    model = SpeechLM(kind, hf_snapshot(pins[kind]["repo"], pins[kind]["revision"]),
                     hf_snapshot(pins["codec"]["repo"], pins["codec"]["revision"]))
    steerer = Steerer(model, mode=cfg["steering"]["mode"])
    get = vector_bank(a.run, a.condition, cfg)
    out_dir = a.run / "gen" / a.stage / a.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    B = a.batch_size or cfg["steering"]["batch_size"]
    t_start = time.time()
    n_done = 0
    for seed in sorted({j["seed"] for j in todo}):
        js = sorted([j for j in todo if j["seed"] == seed], key=lambda j: (utts[j["utterance_id"]]["budget"], j["job_id"]))
        for i in range(0, len(js), B):
            batch = js[i:i + B]
            rows = [dict(text=utts[j["utterance_id"]]["Devanagari"], voice=voice, budget=utts[j["utterance_id"]]["budget"])
                    for j in batch]
            deltas = {}
            for r, j in enumerate(batch):
                for s in j["steer"]:
                    k = (s["layer"], s["site"])
                    if k not in deltas:
                        deltas[k] = torch.zeros(len(batch), model.lm.config.hidden_size, device=model.device)
                    deltas[k][r] += s["alpha"] * get(s["layer"], s["site"], s["pool"], s["vec"]).to(model.device)
            bseed = int(stable_id("batch", a.stage, a.condition, seed, [j["job_id"] for j in batch])[:8], 16)
            res = generate_batch(model, rows, bseed, steerer, deltas)
            for j, (audio, info) in zip(batch, res):
                rec = dict(j, batch_seed=bseed, model=kind, voice=voice, revision=pins[kind]["revision"],
                           budget=utts[j["utterance_id"]]["budget"], decoding=model.decoding,
                           steer_mode=cfg["steering"]["mode"], **info)
                try:
                    wav = model.decode(audio)
                    path = out_dir / f"{j['job_id']}.wav"
                    sf.write(path, wav, 24000, subtype="PCM_16")
                    rec.update(status="generated", duration_s=round(len(wav) / 24000, 3),
                               wav=str(path.relative_to(a.run)))
                except Exception as e:
                    rec.update(status="decode_failed", error=str(e)[:200], duration_s=0.0)
                append_jsonl(log, rec)
            n_done += len(batch)
            el = time.time() - t_start
            print(json.dumps(dict(cond=a.condition, stage=a.stage, done=n_done, total=len(todo), elapsed_s=round(el),
                                  batch_gen_s=res[0][1]["batch_gen_s"])), flush=True)
    write_json(a.run / "gen" / a.stage / f"{a.condition}_completion.json",
               dict(n=len(jobs), runtime_s=round(time.time() - t_start), device=torch.cuda.get_device_name(0)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=["sweep1", "sweep2", "test"], required=True)
    p.add_argument("--condition", choices=CONDS)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    (plan if a.action == "plan" else run)(a, cfg)
