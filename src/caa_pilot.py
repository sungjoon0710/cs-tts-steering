"""Shared helpers for the CAA pilot: config, data manifest, split, run directory, provenance."""
from __future__ import annotations

import csv
import getpass
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/caa_pilot/pilot.yaml"
SPLIT = REPO / "configs/caa_pilot/split.json"
LOCAL = Path(os.environ.get("LOCAL_SP", f"/local/speech/users/{getpass.getuser()}"))


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def stable_id(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hf_snapshot(repo: str, revision: str) -> Path:
    home = Path(os.environ.get("HF_HOME", LOCAL / "hf_cache"))
    d = home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not d.is_dir():
        raise FileNotFoundError(f"{repo}@{revision} not in HF cache ({d})")
    return d


def token_budget_for(row) -> int:
    from src.tts_settings import token_budget
    return max(token_budget(row["Romanized"]), token_budget(row["Devanagari"]))


def load_data(cfg):
    """Returns utterances {uid: {...}} with natural and per-condition synthetic paths and token-cap flags."""
    root = Path(cfg["data_dir"])
    with (root / "manifest.csv").open(encoding="utf-8-sig") as f:
        manifest = list(csv.DictReader(f))
    with (root / "utterances_100.csv").open(encoding="utf-8-sig") as f:
        utts = {r["utterance_id"]: dict(r) for r in csv.DictReader(f)}
    for r in manifest:
        u = utts[r["utterance_id"]]
        path = root / r["folder"] / r["filename"]
        if r["configuration_id"].startswith("natural/"):
            u["natural_wav"], u["natural_speaker"], u["natural_sha256"] = str(path), r["configuration_id"].split("/")[1], r["SHA256"]
            continue
        for cond, spec in cfg["conditions"].items():
            if r["filename"] == spec["wav"]:
                u.setdefault("synthetic", {})[cond] = dict(
                    wav=str(path), sha256=r["SHA256"], configuration_id=r["configuration_id"],
                    token_cap_warning=r["token_cap_warning"] == "True", input_text=r["input_text_or_reference"])
    for u in utts.values():
        u["budget"] = token_budget_for(u)
    return utts


def condition_voices(cfg, utts):
    """Voice per condition. Read from the manifest configuration_id (model/voice/script), never retyped."""
    voices = {}
    for cond, spec in cfg["conditions"].items():
        ids = {u["synthetic"][cond]["configuration_id"] for u in utts.values()}
        if len(ids) != 1:
            raise ValueError(f"{cond}: configuration ids {ids}")
        model, voice, script = next(iter(ids)).split("/")
        if script != "devanagari" or model != spec["manifest_model"]:
            raise ValueError(f"{cond}: unexpected configuration {ids}")
        if spec.get("voice") not in (None, voice):
            raise ValueError(f"{cond}: config voice {spec.get('voice')} != manifest {voice}")
        voices[cond] = voice
    return voices


def load_split(path=SPLIT):
    return json.loads(Path(path).read_text())


def git_provenance(run: Path):
    """Records the local commit and a diff snapshot (copied by rsync as code/.provenance)."""
    prov = REPO / ".provenance"
    out = {}
    if prov.is_dir():
        for name in ("commit.txt", "status.txt"):
            if (prov / name).exists():
                out[name] = (prov / name).read_text()
    else:
        try:
            out["commit.txt"] = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True)
        except Exception:
            out["commit.txt"] = "unknown"
    return out
