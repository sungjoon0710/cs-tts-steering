# AGENTS.md

Guide for people and coding agents working on this repo: what exists, where files live, and how to run and write code. Everything is per-user: replace `<uni>` with your Columbia UNI. On kedi, `$USER` already is your UNI.

Research summary: [`README.md`](README.md). Full study notes: [`docs/study_design.md`](docs/study_design.md). Baseline experiment plan: [`docs/baseline_plan.md`](docs/baseline_plan.md).

---

## 1. Status so far

| Done | Notes |
|---|---|
| Compute access | Columbia CS speech-lab machine **kedi** (also `clac`), key-based SSH. |
| Repo | `github.com/sungjoon0710/cs-tts-steering` (**public**). Layout: `docs/`, `src/`, `scripts/`, `notebooks/`. |
| Python env | uv + Python 3.11 venv on kedi's local disk; torch 2.6.0+cu124, transformers 5.x, snac. |
| Model weights | Indic-Speak (`bodhan-ai/indic-speak`), Orpheus 3B Hindi fine-tuned (`canopylabs/3b-hi-ft-research_release`, code name `orpheus-hi-ft`) and pretrained (`canopylabs/3b-hi-pretrain-research_release`, `orpheus-hi-pt`), SNAC 24 kHz decoder. All in a personal HF cache. |
| MaSaC data | Not in the lab corpora. Cloned from the authors' GitHub (MSH-COMICS = original release; EDiReF-SemEval2024 = derived emotion labels). See §5. |
| Smoke test | `notebooks/smoke_test.ipynb`: 5 hand-picked code-switched utterances × both models. Both models load and generate. RTF ≈ 1.8 single-stream on an L40. Orpheus may truncate long inputs (utt 48). |
| Key finding | Much of MaSaC's `Utterances.txt` is **garbled automatic text**, not clean human transcripts. A clean test set needs curation (see §5). |
| Decision | Naturalness is judged **only by Hindi–English bilingual listeners** (MOS / A-B). Don't use UTMOS or SQUIM. |
| CAA pilot (2026-10-29) | Steering Indic-Speak (Amit, Kavya) and Orpheus (Ritika) toward natural speech: the direction exists and is causal, but on objective metrics it shifts speaker pitch/pausing rather than delivery. Objective no-go for all three; Indic-Kavya best candidate; listening pending. Plan: `docs/caa_pilot_plan.md`. Report: `docs/reports/preliminary_caa.pdf`. Run: `$LOCAL_SP/outputs/caa_pilot/20261029-1320_listening100/` (sp4050). |

Next: batched generation wrapper → curated ~100-utterance pilot set → pilot run (≤ 1 h) → listening package for raters. See `docs/baseline_plan.md`.

---

## 2. Access

### SSH (on your laptop)
Team UNIs: `sp4050`, `aa6109`.

Add to `~/.ssh/config`, replacing `<uni>` with your own UNI (e.g. `User aa6109`):
```
Host kedi
    HostName kedi.cs.columbia.edu
    User <uni>
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
    ServerAliveInterval 60

Host clac
    HostName clac.cs.columbia.edu
    User <uni>
    IdentityFile ~/.ssh/id_ed25519
```
Without the config, the equivalent is `ssh <uni>@kedi.cs.columbia.edu`. If you don't have a key yet, create one with `ssh-keygen -t ed25519`. Then run `ssh-copy-id kedi` once (it asks for your CS password). After that, `ssh kedi` needs no password. Close the persistent connection with `ssh -O exit kedi`.

**Editing and listening:** VS Code → Remote-SSH → `kedi`. Open the repo or output folders there. `.wav` files play in VS Code's built-in preview. Right-click → *Download…* copies a file to your laptop.

### Machine facts (kedi)
- 64 cores, 503 GB RAM, Ubuntu 22.04.
- GPUs: `0,1` = L40 48 GB, `2,3` = RTX A5500 24 GB. The NVIDIA driver supports CUDA 12.1; cu124 PyTorch wheels run on it.
- **No job scheduler.** GPUs are shared with the lab: check `nvidia-smi`, then pin a GPU with `CUDA_VISIBLE_DEVICES=N`. Run long jobs inside `tmux`.
- You must be in the `speech` Unix group (`id` shows it) to use `/local/speech` and `/proj/speech`.

---

## 3. Where things live

| What | Path | In git? |
|---|---|---|
| Repo (clone here) | `~/cs-tts-steering` (home = NFS, backed by `bronze`) | — |
| Your fast local workspace | `/local/speech/users/<uni>/` (= `$LOCAL_SP`; 7 TB NVMe, kedi-only, **not backed up**) | no |
| Python env | `$LOCAL_SP/envs/tts` | no |
| HF model cache | `$LOCAL_SP/hf_cache` (= `$HF_HOME`). There is no shared lab model cache. | no |
| uv cache / uv Pythons | `$LOCAL_SP/.cache/uv`, `$LOCAL_SP/.cache/uv-python` | no |
| Datasets | `$LOCAL_SP/data/` (MaSaC under `data/masac/`) | no |
| Experiment outputs | `$LOCAL_SP/outputs/<experiment>/<run_id>/` (smoke test: `outputs/smoke/<model>/<voice>/<line>.wav`) | no |
| Lab corpora (read-only use) | `/proj/speech/corpora/` (code-switching: `code-switching/`, `seame/`, `CSpeech/`) | no |
| 100 matched listening examples (aa6109; input for the CAA pilot) | [`/local/speech/users/aa6109/Listening_Examples_100/`](vscode://vscode-remote/ssh-remote+kedi/local/speech/users/aa6109/Listening_Examples_100): per utterance, natural + 4 Indic-Speak + 1 Orpheus wav and `TRANSCRIPT.txt`; `manifest.csv`, `utterances_100.csv`, `listen.html`. Open the path directly; the parent dir isn't listable. See [`docs/caa_pilot_plan.md`](docs/caa_pilot_plan.md) §3 | no |

Home-directory shortcuts (optional): `ln -s /proj/speech/corpora ~/corpora; ln -s /local/speech ~/local_speech`.

### Existing project workspace: `/local/speech/users/sp4050/`
The project's artifacts so far were created in sp4050's workspace. It is the **reference location**: when a file isn't in your own `$LOCAL_SP`, look here.

| What | Path |
|---|---|
| Python env (working, tested) | `/local/speech/users/sp4050/envs/tts` |
| Model weights | `/local/speech/users/sp4050/hf_cache/hub/` (Indic-Speak, Orpheus Hindi ft + pt, SNAC) |
| MaSaC (both releases) | `/local/speech/users/sp4050/data/masac/` |
| Smoke-test audio | `/local/speech/users/sp4050/outputs/smoke/<model>/<voice>/<line>.wav` |
| Executed smoke-test notebook | `/home/sp4050/cs-tts-steering/notebooks/smoke_test.ipynb` |
| Link to the 100 listening examples | `/local/speech/users/sp4050/Listening_Examples_100` → symlink to `/local/speech/users/aa6109/Listening_Examples_100` |

Rules:
- **Read** from sp4050's workspace. **Write** only to your own `$LOCAL_SP`. Never modify another member's files.
- Access depends on permissions. These directories are currently owner-only (`drwx------`), so other members get *Permission denied* until the owner grants the `speech` group read access, e.g. `chmod -R g+rX /local/speech/users/sp4050/{data,outputs,envs}`. For `hf_cache`, also keep the login token private: `chmod -R g+rX .../hf_cache && chmod 600 .../hf_cache/token`.
- In code, look in your own workspace first, then fall back to sp4050's:
  ```python
  SHARED = Path("/local/speech/users/sp4050")
  def find(rel):  # e.g. find("data/masac")
      for root in (LOCAL, SHARED):
          if (root / rel).exists():
              return root / rel
      raise FileNotFoundError(rel)
  ```

---

## 4. One-time setup for a new member (on kedi)

If sp4050's workspace is group-readable (see §3), you can reuse its env and weights instead of steps 3–5: `export HF_HOME=/local/speech/users/sp4050/hf_cache HF_HUB_OFFLINE=1` (read-only use; offline mode stops the hub client from writing to the cache) and activate `/local/speech/users/sp4050/envs/tts`. Otherwise set up your own copy:

```bash
# 1. uv (no dotfile edits by the installer)
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh

# 2. Env vars — append to ~/.bashrc, then `source ~/.bashrc`
cat >> ~/.bashrc <<'EOF'
# >>> cs-tts-steering setup >>>
export PATH="$HOME/.local/bin:$PATH"
export LOCAL_SP=/local/speech/users/$USER
export UV_CACHE_DIR=$LOCAL_SP/.cache/uv
export UV_PYTHON_INSTALL_DIR=$LOCAL_SP/.cache/uv-python
export HF_HOME=$LOCAL_SP/hf_cache
alias tts-env="source $LOCAL_SP/envs/tts/bin/activate"
# <<< cs-tts-steering setup <<<
EOF
source ~/.bashrc
mkdir -p $LOCAL_SP/{data,outputs}

# 3. Python env
uv venv $LOCAL_SP/envs/tts --python 3.11
tts-env
uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers>=5" accelerate snac soundfile numpy pandas pyyaml huggingface_hub ipykernel nbclient
uv pip install jiwer uroman praat-parselmouth tabulate   # scripts/evaluate.py

# 4. Model weights (first accept the terms on both HF model pages while logged in, then create a read token)
hf auth login
hf download bodhan-ai/indic-speak --exclude "*__pycache__*"
hf download canopylabs/3b-hi-ft-research_release \
  --exclude rng_state.pth --exclude scheduler.pt --exclude training_args.bin --exclude trainer_state.json
hf download canopylabs/3b-hi-pretrain-research_release \
  --include "*.safetensors" --include "*.json" --include README.md --exclude trainer_state.json  # skips the 15 GB FSDP duplicate
hf download hubertsiuzdak/snac_24khz

# 5. Repo + MaSaC
git clone https://github.com/sungjoon0710/cs-tts-steering.git ~/cs-tts-steering
mkdir -p $LOCAL_SP/data/masac && cd $LOCAL_SP/data/masac
git clone --depth 1 https://github.com/LCS2-IIITD/MSH-COMICS.git
git clone --depth 1 https://github.com/LCS2-IIITD/EDiReF-SemEval2024.git
```
Pass each excluded file as its own `--exclude` flag. If you write `--exclude a b c`, the CLI treats `b c` as files to download.

Sanity check: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

Note: `~/.bashrc` is **not** sourced by non-interactive `ssh kedi '<cmd>'`. In remote one-liners (and for agents), export the variables explicitly or `source ~/.bashrc` first.

---

## 5. Data: MaSaC

| Release | Path under `$LOCAL_SP/data/masac/` | Contents |
|---|---|---|
| **MSH-COMICS** (Bedi et al. 2021, original) | `MSH-COMICS/Data/Raw_Utterances/` | `Utterances.txt` (15,575 lines, `Speaker: text`, Devanagari + Latin script); `Audio_Filename.txt.txt` and `Episodewise_dialoguelabels.txt` (one header line each; per-utterance mp3 name and `E<ep>-D<dialog>`) |
| | `MSH-COMICS/Data/Pickles/Dataset.p` | Only source of the **Sarcasm / Humour** labels (pandas DataFrame: Speaker, text, Context, Sarcasm, Humour, Episode_label, Audio_Filename) |
| | raw audio | Not downloaded. Google Drive link in `MSH-COMICS/README.md` |
| **EDiReF-SemEval2024** (derived, SemEval-24 Task 10) | `EDiReF-SemEval2024/Data/Task {A,B,C}/` | JSON dialogs `{episode, speakers, utterances, emotions[, triggers]}` in **romanized** Hinglish. Task A: 8 emotions, 8,506/1,354/1,580 utterances. Separate dataset: no shared IDs with MSH-COMICS, and no audio. |

Caveats:
- **Transcript quality:** large parts of `Utterances.txt` are garbled: transliteration noise, machine-translated Hindi, scrambled word order, and the same utterance repeated in different forms. No simple rule separates clean lines from garbled ones. Example clean lines: 16, 44, 48, 126, 270 (0-indexed). Curate before using any line as a test input.
- Don't join the `.txt` files by line number without checking (15,575 vs 15,576 rows). `Dataset.p` holds text and audio filename in the same row.
- **Pickles execute code when loaded.** Load `Dataset.p` only with a restricted unpickler that allow-lists pandas/numpy classes. Inspect unknown pickles with `pickletools.dis` first.
- Speaker names are inconsistent across releases (indu / indravadan / indravardhan …).

---

## 6. Models: practical notes

| | Indic-Speak | Orpheus 3B Hindi (`orpheus-hi-ft`, `orpheus-hi-pt`) |
|---|---|---|
| Load | `inference.py` in the snapshot dir (`TTS` class, `build_prompt`, `ids_to_codes`) | `AutoModelForCausalLM` + `snac_24khz` decoder |
| Voices | Hindi: `Kavya` (F), `Amit` (M); see `voices.md` in the snapshot | **ft**: one voice, undocumented; we use `ऋतिका` (sweep in smoke test). **pt**: base model with no named voices; prompt with text only (`--voices none`), so the speaker varies per clip |
| Decoding | t 0.6, top_p 0.9, top_k 50, **rep. penalty 1.2** (vendor config; `inference.py` omits it) | t 0.6, top_p 0.9, **rep. penalty 1.1** (required) |
| Prompt | `<|start_of_human|>…<|start_of_speech|>` (built by `build_prompt`) | `128259` + tok(`"{voice}: {text}"`) + `128009, 128260`; audio after `128257`, ends at `128258`; codes from `128266` |
| Batching | `inference.py` is single-utterance; write a batched wrapper | same |
| Safety | `vocos/best.pt` passes `torch.load(weights_only=True)`. Check again if the checkpoint changes. | — |

Find a snapshot dir without network: `$HF_HOME/hub/models--<org>--<name>/snapshots/$(cat …/refs/main)`. `snapshot_download(..., local_files_only=True)` fails because we skipped some files on purpose.

---

## 7. Writing scripts and notebooks

**Layout**
- `src/`: importable library code (data loading, TTS wrappers, metrics). No hard-coded user paths.
- `scripts/`: entry points with `argparse` (`prepare_testset.py`, `generate.py`, `evaluate.py`, …). Anything GPU-heavy or long-running goes here, not in notebooks.
- `notebooks/`: review and listening only (tables, plots, `IPython.display.Audio`). Use the kernel `$LOCAL_SP/envs/tts/bin/python`.
- `configs/` (create when needed): YAML for models, decoding params, test sets.

**Conventions**
- Resolve paths from the environment, never from a UNI:
  ```python
  import getpass, os
  from pathlib import Path
  LOCAL = Path(os.environ.get("LOCAL_SP", f"/local/speech/users/{getpass.getuser()}"))
  os.environ.setdefault("HF_HOME", str(LOCAL / "hf_cache"))
  ```
- Outputs go to `$LOCAL_SP/outputs/<experiment>/<run_id>/` with `run_id = YYYYMMDD-HHMM_<testset>`. Save `config.yaml`, the git commit hash, `generations.jsonl` (per clip: model, voice, seed, tokens, stop reason, timing), audio as 24 kHz wav, and metrics files.
- Make generation **resumable** (skip clips whose wav exists) and seeded.
- Run headless in tmux: `tmux new -s gen` → `tts-env` → `CUDA_VISIBLE_DEVICES=0 python scripts/generate.py …` → detach with `Ctrl-b d`.
- To execute a notebook headlessly (embeds audio players): `python -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('X.ipynb',4); NotebookClient(nb,timeout=1800,allow_errors=True).execute(); nbformat.write(nb,'X.ipynb')"`.

**Git hygiene (repo is public)**
- Never commit audio, model weights, MaSaC transcripts, or label files. Commit utterance **IDs** and summary tables only.
- **Clear notebook outputs before committing.** Executed notebooks embed MaSaC text and base64 audio.

---

## 8. Rules for coding agents

- **Ask before bringing anything into the lab environment**: downloads, `git clone`, package installs, new directories or files on kedi. Read-only exploration (`ls`, `find`, `cat`, `nvidia-smi`, `df`) is fine.
- Check `nvidia-smi` before using a GPU, and prefer one GPU per job.
- Don't modify anything under `/proj/speech` or other users' directories.
- Use human listening for naturalness. Don't add UTMOS/SQUIM.
- Don't push to the public repo without the user's OK.
