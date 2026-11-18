"""n-shot in-context style prompting pilot on Orpheus-hi-pt (docs/nshot_style_prompting_plan.md).

Conditions (all 30 targets each): base (no shots; run first), nat1/nat3/nat5 (n natural shots of the same
speaker = next n lines in list order, cyclic), syn3 (nat3's shot texts with the shot lines' base audio).

Prompt per shot:  [SOH] tok(text) [EOT EOH SOA SOS] audio [EOS EOA]
Target:           [SOH] tok(text) [EOT EOH SOA SOS]   -> model generates audio, stops at EOS (128258).

Usage (on kedi):
  python scripts/icl_generate.py --run-id 20261112-1500_maya_indr30 [--conditions base nat1 nat3 nat5 syn3]
Resumable: skips clips already recorded in generations.jsonl with a wav on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LOCAL = Path(os.environ.get("LOCAL_SP", "/local/speech/users/sp4050"))
DATA = Path("/local/speech/users/sp4050/Listening_Examples_100_PT")
SEED = 1234
SOH, EOT, EOH, SOA, SOS, EOS, EOA = 128259, 128009, 128260, 128261, 128257, 128258, 128262
DECODING = dict(temperature=0.6, top_p=0.9, repetition_penalty=1.1, do_sample=True)

SPEAKERS = {
    "MAYA": ["test_175_0", "train_1394_7", "val_131_1", "train_214_1", "train_1265_2", "train_588_5",
             "train_376_7", "train_17_2", "train_549_0", "train_240_3", "train_76_0", "train_1690_0",
             "test_173_0", "train_723_2", "train_1241_0"],
    "INDRAVARDHAN": ["train_1219_3", "train_1692_2", "test_158_0", "train_959_4", "train_406_1", "train_1181_6",
                     "train_291_0", "train_1472_2", "val_185_0", "train_1745_15", "train_1758_1", "train_734_1",
                     "train_1059_0", "train_1355_2", "train_378_4"],
}
CONDITIONS = {"base": ("none", 0), "nat1": ("natural", 1), "nat3": ("natural", 3), "nat5": ("natural", 5),
              "syn3": ("base", 3)}


def utterances():
    """uid -> dict(speaker, text, natural_wav, dir). Text = line after 'DEVANAGARI LLM INPUT'."""
    out = {}
    for spk, uids in SPEAKERS.items():
        for uid in uids:
            dirs = [d for d in DATA.iterdir() if d.is_dir() and d.name.split("_", 1)[1].startswith(uid + "_")]
            if len(dirs) != 1:
                raise SystemExit(f"{uid}: expected 1 dir, found {[d.name for d in dirs]}")
            d = dirs[0]
            lines = (d / "TRANSCRIPT.txt").read_text(encoding="utf-8").splitlines()
            i = [k for k, l in enumerate(lines) if l.strip() == "DEVANAGARI LLM INPUT"]
            if len(i) != 1 or not lines[i[0] + 1].strip():
                raise SystemExit(f"{uid}: DEVANAGARI LLM INPUT line not found")
            nat = sorted(d.glob("01_Natural_*.wav"))
            if len(nat) != 1:
                raise SystemExit(f"{uid}: natural wav not found")
            file_spk = nat[0].stem.split("_", 2)[2].upper()
            if file_spk != spk:
                raise SystemExit(f"{uid}: natural wav speaker {file_spk} != {spk}")
            out[uid] = dict(speaker=spk, text=lines[i[0] + 1].strip(), natural_wav=str(nat[0]), dir=d.name)
    return out


def shot_ids(uid, speaker, n):
    lst = SPEAKERS[speaker]
    k = lst.index(uid)
    return [lst[(k + j) % len(lst)] for j in range(1, n + 1)]


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N targets per condition (smoke test)")
    a = ap.parse_args()

    import soundfile as sf
    import torch
    from src.snac_encode import wav_to_tokens
    from src.tts import OrpheusHindiPretrained, cached_snapshot, snac_codes
    from src.tts_settings import AUDIO_BASE, CODEBOOK_SIZE, SR, token_budget

    run = LOCAL / "outputs" / "icl_pilot" / a.run_id
    run.mkdir(parents=True, exist_ok=True)
    utts = utterances()
    order = [u for spk in SPEAKERS for u in SPEAKERS[spk]]
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
    (run / "git_commit.txt").write_text(commit + "\n" + dirty, encoding="utf-8")
    cfg = dict(run_id=a.run_id, model=OrpheusHindiPretrained.repo,
               model_snapshot=str(cached_snapshot(OrpheusHindiPretrained.repo)),
               codec_snapshot=str(cached_snapshot("hubertsiuzdak/snac_24khz")), dtype="bfloat16", attn="sdpa",
               decoding=DECODING, eos=EOS, seed=SEED, max_new_tokens="token_budget(target_text)",
               prompt=dict(shot="[128259] tok(text) [128009,128260,128261,128257] audio [128258,128262]",
                           target="[128259] tok(text) [128009,128260,128261,128257]", tokenizer_call="tok(text).input_ids (incl. BOS)",
                           shot_audio="snac_encode.wav_to_tokens(base=128266, dedup=False)"),
               conditions={c: dict(shot_audio=s, n_shots=n) for c, (s, n) in CONDITIONS.items()},
               shot_rule="next n lines of the same speaker in list order, cyclic", speakers=SPEAKERS,
               data_dir=str(DATA), git_commit=commit)
    import yaml
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    dev = "cuda"
    m = OrpheusHindiPretrained(device=dev, codec_snapshot=cached_snapshot("hubertsiuzdak/snac_24khz"))
    tok, lm, snac = m.tok, m.lm, m.snac
    gen_path = run / "generations.jsonl"
    done = {(g["condition"], g["utterance_id"]) for g in read_jsonl(gen_path)
            if g.get("wav") is None or (run / g["wav"]).exists()}
    shot_cache = {}

    def shot_audio(uid, source):
        key = (uid, source)
        if key not in shot_cache:
            p = utts[uid]["natural_wav"] if source == "natural" else run / "base" / f"{uid}.wav"
            if not Path(p).exists():
                raise SystemExit(f"shot audio missing: {p} (run condition base first)")
            toks, _ = wav_to_tokens(p, snac, base=AUDIO_BASE, dedup=False, device=dev)
            shot_cache[key] = toks
        return shot_cache[key]

    hi = AUDIO_BASE + 7 * CODEBOOK_SIZE
    for cond in a.conditions:
        source, n = CONDITIONS[cond]
        (run / cond).mkdir(exist_ok=True)
        targets = order if a.limit is None else [u for spk in SPEAKERS for u in SPEAKERS[spk][:a.limit]]
        for uid in targets:
            if (cond, uid) in done:
                continue
            u = utts[uid]
            shots = shot_ids(uid, u["speaker"], n) if n else []
            assert uid not in shots
            prompt = []
            for s in shots:
                prompt += [SOH] + tok(utts[s]["text"]).input_ids + [EOT, EOH, SOA, SOS] + shot_audio(s, source) + [EOS, EOA]
            prompt += [SOH] + tok(u["text"]).input_ids + [EOT, EOH, SOA, SOS]
            max_new = token_budget(u["text"])
            ids = torch.tensor([prompt], device=dev)
            torch.manual_seed(SEED)
            t0 = time.time()
            with torch.no_grad():
                gen = lm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                  eos_token_id=EOS, pad_token_id=EOS, **DECODING)
            torch.cuda.synchronize()
            gen_s = time.time() - t0
            new = gen[0, len(prompt):].tolist()
            n_audio = next((k for k, t in enumerate(new) if not AUDIO_BASE <= t < hi), len(new))
            first_non_audio = new[n_audio] if n_audio < len(new) else None
            rec = dict(condition=cond, utterance_id=uid, speaker=u["speaker"], shot_ids=shots,
                       shot_audio=source if n else None, prompt_tokens=len(prompt), n_new_tokens=len(new),
                       max_new_tokens=max_new, hit_max=len(new) >= max_new, eos_found=EOS in new,
                       n_audio_tokens=n_audio, first_non_audio_token=first_non_audio,
                       partial_frame_tokens=n_audio % 7, seed=SEED, gen_s=round(gen_s, 3))
            try:
                with torch.no_grad():
                    wav = snac.decode(snac_codes(new, dev))[0, 0].clamp(-1, 1).float().cpu().numpy()
            except ValueError as e:  # no complete frame
                rec.update(wav=None, audio_s=0.0, status="empty_audio", error=str(e))
            else:
                p = run / cond / f"{uid}.wav"
                sf.write(p, wav, SR, subtype="PCM_16")
                rec.update(wav=f"{cond}/{uid}.wav", audio_s=round(len(wav) / SR, 3), status="generated")
            with gen_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{cond} {uid} prompt={len(prompt)} new={len(new)} eos={rec['eos_found']} "
                  f"audio={rec['audio_s']}s gen={gen_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
