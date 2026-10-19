"""Prepare, evaluate and report an offline pilot. Never runs ASR or TTS."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import VERSION
from src.evaluation.audio import extract_audio
from src.evaluation.common import digest, private_directory, read_jsonl, utc_now, write_json, write_jsonl
from src.evaluation.records import import_snapshot, select_pilot
from src.evaluation.text import NORMALIZATION_VERSION, checked_language, edit_score, normalize, pier


def prepare(run):
    utterances, attempts, generations, cache = import_snapshot(run / "source")
    pilot = select_pilot(utterances)
    import csv
    legacy = list(csv.DictReader((run / "source/eval/metrics.csv").open(encoding="utf-8")))
    pilot["challenge_ids"] = sorted({r["line"] for r in legacy if
        r["runaway"].lower() == "true" or r["too_short"].lower() == "true"}, key=int)
    expected = {(g["configuration_id"], g["utterance_id"]) for g in generations}
    if len(legacy) != len(expected) or {(r["model"] + "/" + r["voice"], r["line"]) for r in legacy} != expected:
        raise ValueError("Legacy metric/generation association mismatch")
    by_id = {u["utterance_id"]: u for u in utterances}
    for row in legacy:
        if row["text"] != by_id[row["line"]]["text_raw"] or cache.get(row["wav"]) != row["hyp"]:
            raise ValueError("Legacy input/ASR association mismatch")
    for name, rows in (("utterances", utterances), ("attempts", attempts), ("generations", generations)):
        write_jsonl(run / "tables" / (name + ".jsonl"), rows)
    write_json(run / "pilot.json", pilot)
    import itertools
    configs = sorted({g['configuration_id'] for g in generations})
    by_key = {(g['utterance_id'],g['configuration_id']):g['generation_id'] for g in generations}
    write_jsonl(run/'tables/listening_pairs.jsonl',[
        {'utterance_id':uid,'configuration_1':a,'configuration_2':b,
         'generation_1':by_key[(uid,a)],'generation_2':by_key[(uid,b)]}
        for uid in pilot['utterance_ids'] for a,b in itertools.combinations(configs,2)])
    # Never overwrite researcher annotations when regenerating derived tables.
    labels_path = run / "annotations/input_labels.jsonl"
    if not labels_path.exists():
        write_jsonl(labels_path, [{"utterance_id": u["utterance_id"], "text_sha256": u["text_sha256"],
            "tokens": normalize(u["text_raw"]).split(), "labels": [], "matrix_language": None,
            "verified": False, "annotator_id": None, "strategy": None} for u in utterances])
    tasks = [g for g in generations if g["utterance_id"] in set(pilot["utterance_ids"])]
    write_jsonl(run / "annotations/transcription_tasks.jsonl", [{"generation_id": g["generation_id"],
        "utterance_id": g["utterance_id"], "status": "pending", "transcript": None} for g in tasks])
    write_json(run / "audit.json", {"utterances": len(utterances), "configurations": sorted({g["configuration_id"] for g in generations}),
        "generations": len(generations), "attempt_records": len(attempts),
        "attempt_status_counts": dict(Counter(a["status"] for a in attempts)),
        "technical_status_counts": dict(Counter(g["technical_status"] for g in generations)),
        "successful_retries": sum(g["successful_retry"] for g in generations),
        "cached_asr_records": len(cache), "pilot_recordings": len(tasks),
        "source_sampling": "consecutive source lines 0-99, not the planned curated sample",
        "limitations": ["unverified legacy evaluator and ASR checkpoint", "no episode/conversation mapping",
                        "no verified token-language annotations", "no human results yet"]})
    print("Prepared: 20 provisional pilot utterances; all three configurations; annotations pending.")


def unique_annotations(path, key):
    rows = read_jsonl(path)
    out = {}
    for r in rows:
        k = r[key]
        if k in out:
            raise ValueError(f"Duplicate annotation for {key}: {k}; adjudicate explicitly")
        out[k] = r
    return out


def evaluate(run, full=False):
    # Recheck immutable source data on every execution, not only during import.
    import_snapshot(run / "source")
    pilot = json.loads((run / "pilot.json").read_text(encoding="utf-8"))
    if full and not pilot.get("protocol_frozen"):
        raise ValueError("Full scaling is disabled until the pilot protocol is validated and frozen")
    utterances = {r["utterance_id"]: r for r in read_jsonl(run / "tables/utterances.jsonl")}
    generations = read_jsonl(run / "tables/generations.jsonl")
    cache = {r["wav"]: r["hyp"] for r in read_jsonl(run / "source/eval/asr.jsonl")}
    annotations = unique_annotations(run / "annotations/input_labels.jsonl", "utterance_id")
    manual = unique_annotations(run / "annotations/manual_transcripts.jsonl", "generation_id")
    failure_reviews = unique_annotations(run / "annotations/failure_reviews.jsonl", "generation_id")
    selected = set(utterances) if full else set(pilot["utterance_ids"])
    challenge = set(pilot.get("challenge_ids", [])) - selected
    results = []
    for g in generations:
        uid = g["utterance_id"]
        if uid not in selected | challenge:
            continue
        u = utterances[uid]
        result = {"generation_id": g["generation_id"], "utterance_id": uid,
            "configuration_id": g["configuration_id"], "cohort": "pilot" if uid in selected else "challenge",
            "text_raw": u["text_raw"], "script_profile": u["script_profile"],
            "input_word_count": u["normalized_words"], "metric_version": VERSION,
            "normalization_version": NORMALIZATION_VERSION, "human_validation": "pending",
            "hit_token_limit": g["hit_token_limit"], "manual_failure_status": "pending"}
        path = run / "source/audio" / g["relative_audio_path"]
        audio = extract_audio(path)
        if path.exists():
            transfer = json.loads((run / "source/audio_transfer.json").read_text(encoding="utf-8"))
            expected_hash = transfer["sha256"].get("audio/" + g["relative_audio_path"])
            if expected_hash != audio.get("audio_sha256"):
                raise ValueError("Audio differs from checked transfer receipt")
            header = g.get("header")
            if header and audio.get("frames") is not None and (header["frames"] != audio["frames"] or header["sample_rate"] != audio["sample_rate"] or header["channels"] != audio["channels"]):
                raise ValueError("Audio header differs from metadata snapshot")
        result["audio"] = audio
        review = failure_reviews.get(g["generation_id"])
        technical_failure = audio["status"] in ("missing_audio", "unreadable_audio", "empty_audio", "nonfinite_audio", "fully_silent")
        result["failure_review"] = {"status": "pending", "flags": None}
        if review and review.get("verified") is True:
            if review.get("audio_sha256") != audio.get("audio_sha256"):
                raise ValueError("Failure review audio hash mismatch")
            result["failure_review"] = {"status": "human_reviewed", "flags": review["flags"], "notes": review.get("notes", "")}
        result["failure_any_confirmed"] = True if technical_failure else any(review["flags"].values()) if review and review.get("verified") is True else None
        hyp = cache.get(g["remote_audio_path"])
        result["asr_raw"] = hyp
        has_audio = audio["status"] not in ("missing_audio", "unreadable_audio", "empty_audio", "nonfinite_audio")
        result["automatic_proxy"] = edit_score(u["text_raw"], hyp if has_audio else None)
        result["asr_provenance"] = "legacy_cached_hypothesis; checkpoint/decoding unknown; NOT validated"
        result["synthesis_content"] = {"status": "pending_manual_transcription", "wer": None, "cer": None}
        result["recognizer_error"] = {"status": "pending_manual_transcription", "wer": None, "cer": None}
        result["rates"] = {"status": "pending_verified_spoken_tokens_and_segmentation", "speech_wps": None, "articulation_wps": None}
        m = manual.get(g["generation_id"])
        if m and m.get("verified") is True:
            if m.get("audio_sha256") != audio.get("audio_sha256"):
                raise ValueError("Manual transcript audio hash mismatch")
            if m.get("complete") is True and isinstance(m.get("transcript"), str):
                result["synthesis_content"] = edit_score(u["text_raw"], m["transcript"])
                result["recognizer_error"] = edit_score(m["transcript"], hyp)
                n = len(normalize(m["transcript"]).split())
                d = audio.get("duration_s")
                result["rates"]["speech_wps"] = n / d if d else None
                if m.get("segmentation_verified") is True and audio.get("articulation_time_estimate_s", 0):
                    result["rates"]["articulation_wps"] = n / audio["articulation_time_estimate_s"]
                result["rates"]["status"] = "spoken_count_verified; articulation_requires_segmentation_validation"
        annotation = annotations.get(uid)
        labels = checked_language(annotation, u["text_raw"])
        result["language"] = {"status": "pending_verified_annotations"}
        result["poi"] = {"status": "pending_verified_annotations", "pier": None}
        if labels is not None:
            eligible = [x for x in labels if x in ("HI", "EN")]
            switches = [(a, b) for a, b in zip(labels, labels[1:]) if a in ("HI", "EN") and b in ("HI", "EN") and a != b]
            result["language"] = {"status": "verified", "english_fraction": labels.count("EN") / len(eligible) if eligible else None,
                "switch_count": len(switches), "switch_directions": [a + "->" + b for a, b in switches],
                "condition": "code_switched" if "HI" in labels and "EN" in labels else "HI" if "HI" in labels else "EN" if "EN" in labels else "OTHER",
                "strategy": annotation.get("strategy")}
            matrix = annotation.get("matrix_language")
            if matrix in ("HI", "EN") and "HI" in labels and "EN" in labels:
                indices = [i for i, label in enumerate(labels) if label == ("EN" if matrix == "HI" else "HI")]
                if hyp is not None and has_audio:
                    result["poi"] = pier(normalize(u["text_raw"]).split(), normalize(hyp).split(), indices)
            elif not ("HI" in labels and "EN" in labels):
                result["poi"] = {"status": "not_applicable", "pier": None}
            else:
                result["poi"] = {"status": "pending_matrix_language", "pier": None}
        results.append(result)
    write_jsonl(run / "tables/metrics.jsonl", results)
    packages = {}
    for name in ("numpy", "scipy", "soundfile", "praat-parselmouth", "rapidfuzz", "jiwer", "matplotlib"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    repo = Path(__file__).resolve().parents[1]
    source_hashes = {p.relative_to(repo).as_posix(): digest(p.read_bytes())
                     for p in sorted((repo / "src/evaluation").glob("*")) if p.is_file()}
    source_hashes["scripts/" + Path(__file__).name] = digest(Path(__file__).read_bytes())
    for name in ('fetch_baseline.py','listen.py'):
        source_hashes['scripts/'+name] = digest((repo/'scripts'/name).read_bytes())
    write_json(run / "provenance.json", {"executed_at_utc": utc_now(), "python": sys.version,
        "platform": platform.platform(), "packages": packages, "device": "CPU",
        "implementation_files_sha256": source_hashes, "pilot_manifest_sha256": digest((run / "pilot.json").read_bytes()),
        "source_snapshot_sha256": digest((run / "source/snapshot.json").read_bytes()),
        "status": "IMPLEMENTED_AND_EXECUTED; HUMAN_VALIDATION_PENDING", "new_asr_runs": 0, "tts_runs": 0})
    print(f"Processed {sum(r['cohort']=='pilot' for r in results)} pilot recordings and {sum(r['cohort']=='challenge' for r in results)} separate challenge recordings.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run", "report"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--full", action="store_true")
    a = p.parse_args()
    run = private_directory(a.run)
    if a.command == "prepare": prepare(run)
    elif a.command == "run": evaluate(run, a.full)
    else:
        from src.evaluation.report import report
        report(run)
