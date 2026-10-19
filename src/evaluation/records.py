from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, read_jsonl, stable_id
from .text import normalize


def script_profile(text):
    deva = any("\u0900" <= c <= "\u097f" for c in text)
    latin = bool(re.search("[A-Za-z]", text))
    return "mixed_script" if deva and latin else "devanagari_only" if deva else "latin_only" if latin else "other"


def import_snapshot(source):
    source = Path(source)
    import json
    receipt = source / 'snapshot.json'
    if receipt.exists():
        for name, expected_hash in json.loads(receipt.read_text(encoding='utf-8'))['sha256'].items():
            if digest((source/name).read_bytes()) != expected_hash:
                raise ValueError('Source snapshot changed: '+name)
    inputs = read_jsonl(source / "utterances.jsonl")
    if len({r["line"] for r in inputs}) != len(inputs):
        raise ValueError("Duplicate input ID")
    utterances = [{"utterance_id": str(r["line"]), "source_line": r["line"],
                   "source": "MSH-COMICS/Raw_Utterances/Utterances.txt; corpus revision unknown",
                   "speaker_label": r["speaker"], "text_raw": r["text"],
                   "text_sha256": digest(r["text"].encode()), "script_profile": script_profile(r["text"]),
                   "normalized_words": len(normalize(r["text"]).split()),
                   "conversation_id": None, "episode_id": None, "language_status": "pending"}
                  for r in inputs]
    by_id = {r["utterance_id"]: r for r in utterances}
    inventory = {r["relative_path"]: r for r in read_jsonl(source / "audio_inventory.jsonl")}
    attempts, grouped = [], defaultdict(list)
    for model_dir in sorted(source.iterdir()):
        path = model_dir / "generations.jsonl"
        if not path.is_file():
            continue
        for index, row in enumerate(read_jsonl(path)):
            uid = str(row["line"])
            if uid not in by_id:
                raise ValueError("Generation references unknown utterance")
            config = row["model"] + "/" + row["voice"]
            aid = stable_id(path.relative_to(source).as_posix(), index, row)
            attempt = {"attempt_id": aid, "source_record_index": index, "utterance_id": uid,
                       "configuration_id": config, "record": row,
                       "status": "preprocessing_failure" if row.get("error") == "empty text" else
                                 "generation_failure" if row.get("error") else "success"}
            attempts.append(attempt)
            grouped[(uid, config)].append(attempt)
    configs = sorted({config for uid, config in grouped})
    generations = []
    for uid in by_id:
        for config in configs:
            records = grouped.get((uid, config), [])
            successes = [a for a in records if a["status"] == "success"]
            chosen = successes[-1] if successes else records[-1] if records else None
            row = chosen["record"] if chosen else {}
            rel = config + "/" + uid + ".wav"
            header = inventory.get(rel)
            if successes and row.get("text") != by_id[uid]["text_raw"]:
                raise ValueError("Selected generation/input text mismatch")
            if row.get("wav") and not row["wav"].replace("\\", "/").endswith("/" + rel):
                raise ValueError("Generation/audio association mismatch")
            generations.append({"generation_id": stable_id(uid, config, chosen["attempt_id"] if chosen else "missing"),
                                "utterance_id": uid, "configuration_id": config, "relative_audio_path": rel,
                                "remote_audio_path": row.get("wav"), "attempt_id": chosen["attempt_id"] if chosen else None,
                                "attempt_count": len(records), "successful_retry": len(records) > 1 and bool(successes),
                                "seed": row.get("seed"), "decoding": row.get("decoding"),
                                "reported_commit": row.get("commit"), "checkpoint_revision": None,
                                "hit_token_limit": row.get("hit_max"), "header": header,
                                "technical_status": "missing_audio" if header is None else header.get("status", "unknown"),
                                "content_status": "pending_listening", "raw_generation_status": chosen["status"] if chosen else "missing"})
    cache = read_jsonl(source / "eval/asr.jsonl")
    if len({r["wav"] for r in cache}) != len(cache):
        raise ValueError("Ambiguous ASR cache paths")
    return utterances, attempts, generations, {r["wav"]: r["hyp"] for r in cache}


def select_pilot(utterances, size=20, seed=1234):
    """Provisional SCRIPT strata; never pretend these are language annotations."""
    if size != 20:
        raise ValueError("This initial protocol specifies 20 utterances")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in sorted(utterances, key=lambda r: int(r["utterance_id"])):
        groups[r["script_profile"]].append(r)
    selected = []
    for profile, quota in (("mixed_script", 14), ("devanagari_only", 3), ("latin_only", 3)):
        bins = defaultdict(list)
        for r in groups[profile]:
            n = r["normalized_words"]
            bins["short" if n <= 7 else "medium" if n <= 20 else "long"].append(r)
        for rs in bins.values():
            rng.shuffle(rs)
        while quota and any(bins.values()):
            for label in ("short", "medium", "long"):
                if quota and bins[label]:
                    selected.append(bins[label].pop()["utterance_id"])
                    quota -= 1
    available = [r["utterance_id"] for r in utterances if r["utterance_id"] not in selected]
    rng.shuffle(available)
    selected.extend(available[:max(0, size - len(selected))])
    if len(selected) != size:
        raise ValueError("Not enough utterances for pilot")
    return {"utterance_ids": sorted(selected, key=int), "seed": seed,
            "sampling": "provisional_script_and_length_stratification",
            "language_validation": "pending", "protocol_frozen": False,
            "selection_counts": dict(Counter(r["script_profile"] for r in utterances if r["utterance_id"] in selected))}
