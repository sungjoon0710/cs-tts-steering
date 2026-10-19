"""Loopback-only listening service. Audio URLs never reveal configurations."""
from __future__ import annotations

import itertools
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .common import digest, read_jsonl, stable_id, utc_now, write_json

CHOICES = {"a", "b", "both_acceptable", "both_unacceptable"}
FAILURES = ("truncated", "major_omission", "repetition", "hallucinated_content",
            "severe_artifact", "unintelligible_segment", "runaway")


def make_trials(generations, pilot_ids, listener_id, slot=0, seed=1234):
    configs = sorted({g["configuration_id"] for g in generations})
    if len(configs) != 3: raise ValueError("This protocol requires three configurations")
    pairs = list(itertools.combinations(configs, 2))
    by_key = {(g["utterance_id"], g["configuration_id"]): g for g in generations}
    rng = random.Random(stable_id(seed, listener_id))
    trials = []
    for i, uid in enumerate(sorted(pilot_ids, key=int)):
        pair = list(pairs[(i + slot) % 3]); rng.shuffle(pair)
        a, b = (by_key[(uid, config)] for config in pair)
        trials.append({"trial_id": stable_id(seed, listener_id, sorted(pilot_ids,key=int), uid, pair), "utterance_id": uid,
            "generation_A": a["generation_id"], "generation_B": b["generation_id"],
            "configuration_A": pair[0], "configuration_B": pair[1], "randomization_seed": seed})
    rng.shuffle(trials)
    return trials


class ListeningStudy:
    def __init__(self, run, transcript_first=False):
        self.run = Path(run)
        self.transcript_first = transcript_first
        self.lock = threading.Lock()
        self.pilot = json.loads((self.run / "pilot.json").read_text(encoding="utf-8"))
        self.pilot_hash = digest((self.run/'pilot.json').read_bytes())
        self.generations = read_jsonl(self.run / "tables/generations.jsonl")
        self.gen = {g["generation_id"]: g for g in self.generations}
        self.text = {r["utterance_id"]: r["text_raw"] for r in read_jsonl(self.run / "tables/utterances.jsonl")}
        self.metrics = {r["generation_id"]: r for r in read_jsonl(self.run / "tables/metrics.jsonl")}
        path = self.run / "human/participants.json"
        self.participants = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.audio_ids = {stable_id("audio", g["generation_id"]): g["generation_id"] for g in self.generations}

    def append(self, path, row):
        path = self.run / path; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def enroll(self, listener, profile):
        if not isinstance(listener, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", listener):
            raise ValueError("Use a pseudonymous ID containing letters, numbers, underscore or hyphen")
        with self.lock:
            if listener in self.participants and self.participants[listener].get('pilot_sha256') != self.pilot_hash:
                raise ValueError('Pilot manifest changed; use a new study/run rather than mixing judgments')
            if listener not in self.participants:
                if profile.get("consent") is not True: raise ValueError("Consent is required")
                allowed = ("hindi_proficiency", "english_proficiency", "hinglish_frequency", "consent")
                self.participants[listener] = {"slot": len(self.participants), "enrolled_at_utc": utc_now(),
                                               'pilot_sha256':self.pilot_hash,
                                               "profile": {k: profile.get(k) for k in allowed}}
                write_json(self.run / "human/participants.json", self.participants)
        return {"status": "ready"}

    def trials(self, listener):
        if listener not in self.participants: raise ValueError("Enroll first")
        return make_trials(self.generations, self.pilot["utterance_ids"], listener, self.participants[listener]["slot"])

    def current(self, listener):
        completed = {r["trial_id"] for r in read_jsonl(self.run / "human/judgments.jsonl") if r["listener_id"] == listener}
        return next((t for t in self.trials(listener) if t["trial_id"] not in completed), None)

    def view(self, listener):
        trial = self.current(listener)
        if trial is None: return {"done": True}
        locks = [r for r in read_jsonl(self.run / "human/naturalness.jsonl") if r["listener_id"] == listener and r["trial_id"] == trial["trial_id"]]
        exposed = {"trial_id": trial["trial_id"], "done": False, "phase": "diagnostics" if locks else "naturalness",
                   "audio_A": "/audio/" + stable_id("audio", trial["generation_A"]),
                   "audio_B": "/audio/" + stable_id("audio", trial["generation_B"]),
                   "completed": sum(r["listener_id"] == listener for r in read_jsonl(self.run / "human/judgments.jsonl")),
                   "total": len(self.pilot["utterance_ids"])}
        if locks or self.transcript_first: exposed["transcript"] = self.text[trial["utterance_id"]]
        return exposed

    def naturalness(self, listener, trial_id, response):
        if response not in CHOICES: raise ValueError("Invalid response")
        with self.lock:
            trial = self.current(listener)
            if not trial or trial["trial_id"] != trial_id: raise ValueError("Not the current trial")
            locks = [r for r in read_jsonl(self.run / "human/naturalness.jsonl") if r["listener_id"] == listener and r["trial_id"] == trial_id]
            if locks: raise ValueError("Naturalness judgment is already locked")
            self.append("human/naturalness.jsonl", {"listener_id": listener, **trial, "naturalness": response,
                'pilot_sha256':self.pilot_hash,
                "timestamp_utc": utc_now(), "transcript_first": self.transcript_first})

    def finish(self, listener, trial_id, pronunciation, smoothness, notes):
        if pronunciation not in CHOICES or smoothness not in CHOICES | {"not_applicable", "uncertain"}:
            raise ValueError("Invalid diagnostic response")
        if not isinstance(notes, str) or len(notes) > 4000: raise ValueError("Invalid notes")
        with self.lock:
            trial = self.current(listener)
            if not trial or trial["trial_id"] != trial_id: raise ValueError("Not the current trial")
            locks = [r for r in read_jsonl(self.run / "human/naturalness.jsonl") if r["listener_id"] == listener and r["trial_id"] == trial_id]
            if not locks: raise ValueError("Naturalness must be recorded first")
            self.append("human/judgments.jsonl", {**locks[0], "pronunciation": pronunciation,
                "switch_smoothness": smoothness, "notes": notes, "completed_at_utc": utc_now()})

    def transcription_view(self, listener):
        if listener not in self.participants: raise ValueError("Enroll first")
        done = {r["generation_id"] for r in read_jsonl(self.run / "annotations/manual_transcripts.jsonl")}
        pending = [g for g in self.generations if g["utterance_id"] in self.pilot["utterance_ids"] and g["generation_id"] not in done]
        pending.sort(key=lambda g: stable_id("transcribe", listener, g["generation_id"]))
        if not pending: return {"done": True}
        g = pending[0]
        return {"done": False, "task_id": stable_id("transcription", g["generation_id"]),
                "audio": "/audio/" + stable_id("audio", g["generation_id"]), "remaining": len(pending)}

    def transcribe(self, listener, task_id, transcript, complete):
        if not isinstance(transcript, str) or len(transcript) > 10000 or not isinstance(complete, bool):
            raise ValueError("Invalid transcription")
        with self.lock:
            task = self.transcription_view(listener)
            if task.get("done") or task["task_id"] != task_id: raise ValueError("Not the current task")
            gid = self.audio_ids[task["audio"].split("/")[-1]]
            sha = self.metrics.get(gid, {}).get("audio", {}).get("audio_sha256")
            if not sha: raise ValueError("Audio must have been evaluated and hashed first")
            self.append("annotations/manual_transcripts.jsonl", {"generation_id": gid, "audio_sha256": sha,
                "transcript": transcript, "complete": complete, "verified": complete,
                "segmentation_verified": False, "annotator_id": listener, "timestamp_utc": utc_now(),
                "validation_scope": "single human transcription; independent adjudication recommended"})

    def review_view(self, listener):
        if listener not in self.participants: raise ValueError("Enroll first")
        done = {r["generation_id"] for r in read_jsonl(self.run / "annotations/failure_reviews.jsonl")}
        pending = sorted((g for g in self.generations if g["generation_id"] in self.metrics and g["generation_id"] not in done),
                         key=lambda g: stable_id("review", listener, g["generation_id"]))
        if not pending: return {"done": True}
        g = pending[0]
        return {"done": False, "task_id": stable_id("review", g["generation_id"]),
                "audio": "/audio/" + stable_id("audio", g["generation_id"]),
                "transcript": self.text[g["utterance_id"]], "failure_categories": list(FAILURES), "remaining": len(pending)}

    def review(self, listener, task_id, flags, notes):
        if not isinstance(flags, dict) or set(flags) != set(FAILURES) or any(type(v) is not bool for v in flags.values()):
            raise ValueError("All failure categories require an explicit boolean")
        if not isinstance(notes, str) or len(notes) > 4000: raise ValueError("Invalid review notes")
        with self.lock:
            task = self.review_view(listener)
            if task.get("done") or task["task_id"] != task_id: raise ValueError("Not the current review")
            gid = self.audio_ids[task["audio"].split("/")[-1]]
            sha = self.metrics[gid]["audio"].get("audio_sha256")
            if not sha: raise ValueError("Audio hash unavailable")
            self.append("annotations/failure_reviews.jsonl", {"generation_id": gid, "audio_sha256": sha,
                "flags": flags, "notes": notes, "verified": True, "reviewer_id": listener, "timestamp_utc": utc_now()})


def handler(study):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # Do not log transcripts or participant IDs.

        def send(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                if url.path == "/":
                    data = Path(__file__).with_name("listening.html").read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
                elif url.path.startswith("/audio/"):
                    gid = study.audio_ids.get(url.path[len("/audio/"):])
                    if not gid: self.send({"error": "Unknown audio"}, 404); return
                    g = study.gen[gid]
                    path = study.run / "source/audio" / g["relative_audio_path"]
                    if not path.is_file(): self.send({"error": "Audio unavailable"}, 404); return
                    data = path.read_bytes()
                    self.send_response(200); self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
                elif url.path in ("/api/trial", "/api/transcription", "/api/review"):
                    listener = parse_qs(url.query).get("listener", [""])[0]
                    self.send(study.view(listener) if url.path == "/api/trial" else study.review_view(listener)
                              if url.path == "/api/review" else study.transcription_view(listener))
                else: self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError) as e: self.send({"error": str(e)}, 400)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send({"error": "Cross-origin requests are not accepted"}, 403); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536: raise ValueError("Invalid request size")
                r = json.loads(self.rfile.read(length)); listener = r["listener_id"]
                if self.path == "/api/enroll": study.enroll(listener, r.get("profile", {}))
                elif self.path == "/api/naturalness": study.naturalness(listener, r["trial_id"], r["response"])
                elif self.path == "/api/finish": study.finish(listener, r["trial_id"], r["pronunciation"], r["smoothness"], r.get("notes", ""))
                elif self.path == "/api/transcribe": study.transcribe(listener, r["task_id"], r["transcript"], r["complete"])
                elif self.path == "/api/review": study.review(listener, r["task_id"], r["flags"], r.get("notes", ""))
                else: self.send({"error": "Not found"}, 404); return
                self.send({"saved": True})
            except (ValueError, KeyError, TypeError) as e: self.send({"error": str(e)}, 400)
    return Handler


def serve(run, port=8765, transcript_first=False):
    study = ListeningStudy(run, transcript_first)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(study))
    print(f"Private listening interface: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
