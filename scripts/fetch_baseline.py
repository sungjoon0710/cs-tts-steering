"""Read-only SSH export to PRIVATE local storage; never write on Kedi.

Only metadata and selected baseline WAVs are transferred. No models or corpora.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import private_directory, read_jsonl, write_json, utc_now


def fetch(root, destination, ids=None):
    destination = private_directory(destination)
    if not root.startswith("/local/speech/") or ".." in PurePosixPath(root).parts:
        raise ValueError("Expected an explicitly authorized /local/speech baseline path")
    script = r'''
import pathlib,sys,tarfile,io,json,hashlib,wave,os
root=pathlib.Path(ROOT)
MODELS=('indic-speak','orpheus-hi-ft','orpheus-hi-pt')
names=['utterances.jsonl',*(m+'/generations.jsonl' for m in MODELS),
       'eval/asr.jsonl','eval/metrics.csv','eval/summary.csv','eval/summary.md',
       'eval.log',*(m+'.log' for m in MODELS)]
hashes={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 def add(name,data):
  info=tarfile.TarInfo(name); info.size=len(data); tar.addfile(info,io.BytesIO(data))
  hashes[name]=hashlib.sha256(data).hexdigest()
 if IDS is None:
  for n in names:
   if (root/n).exists(): add(n,(root/n).read_bytes())
  inventory=[]
  for model in MODELS:
   if not (root/model).is_dir(): continue
   for voice in sorted((root/model).iterdir()):
    if not voice.is_dir(): continue
    for p in sorted(voice.glob('*.wav')):
     if p.is_symlink(): raise RuntimeError('Refusing linked audio')
     st=p.stat(); row={'relative_path':p.relative_to(root).as_posix(),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
     try:
      with wave.open(str(p),'rb') as w:
       row.update(status='header_ok',sample_rate=w.getframerate(),channels=w.getnchannels(),
                  sample_width_bytes=w.getsampwidth(),frames=w.getnframes(),duration_s=w.getnframes()/w.getframerate())
     except (OSError,wave.Error,EOFError) as e: row.update(status='unreadable_header',error_type=type(e).__name__)
     inventory.append(row)
  add('audio_inventory.jsonl',(''.join(json.dumps(r)+'\n' for r in inventory)).encode())
 else:
  for model in MODELS:
   if not (root/model).is_dir(): continue
   for voice in sorted((root/model).iterdir()):
    if not voice.is_dir(): continue
    for uid in IDS:
     p=voice/(str(uid)+'.wav')
     if p.is_symlink(): raise RuntimeError('Refusing linked audio')
     if p.is_file(): add('audio/'+p.relative_to(root).as_posix(),p.read_bytes())
 add('transfer_hashes.json',json.dumps(hashes).encode())
'''
    script = "ROOT=" + repr(root) + "\nIDS=" + repr(ids) + "\n" + script
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "kedi", "timeout", "60s", "python3", "-u", "-B", "-"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(script.encode("utf-8")); proc.stdin.close()
    destination.mkdir(parents=True, exist_ok=True)
    transferred = {}
    advertised = None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name:
                    raise ValueError("Unsafe archive member")
                data = tar.extractfile(member).read()
                if member.name == "transfer_hashes.json":
                    advertised = json.loads(data)
                    continue
                target = destination.joinpath(*name.parts)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError("Refusing to overwrite changed snapshot file: " + member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                transferred[member.name] = hashlib.sha256(data).hexdigest()
        if proc.wait(timeout=15) != 0 or advertised != transferred:
            raise RuntimeError("Incomplete transfer or checksum mismatch")
    except BaseException:
        proc.kill(); proc.wait()
        raise
    write_json(destination / ("audio_transfer.json" if ids is not None else "snapshot.json"),
               {"source": root, "copied_at_utc": utc_now(), "sha256": transferred,
                "ids": ids, "remote_access": "read_only"})
    print(json.dumps({"files": len(transferred), "destination": str(destination), "audio_ids": ids}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--selection", type=Path, help="pilot.json; export selected WAVs and challenge items")
    args = p.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
    ids = sorted(set(selection["utterance_ids"] + selection.get("challenge_ids", [])), key=int) if selection else None
    if ids is not None and any(not str(i).isdigit() for i in ids):
        raise ValueError("Invalid audio IDs")
    fetch(args.remote_root, args.out, ids)
