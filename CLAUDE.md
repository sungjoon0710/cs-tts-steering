# Naturalistic code-switching TTS research

Primary dev environment is `kedi` over SSH (repo cloned there); the Mac copy is secondary.
Repo: github.com/sungjoon0710/cs-tts-steering (public). Layout: `docs/`, `src/`, `scripts/`.
Data and outputs live outside the repo at `/local/speech/users/sp4050/{data,outputs}` on kedi.

## Access
- `ssh kedi` (also `ssh clac`) — user `sp4050`, key auth via `~/.ssh/id_ed25519`, ControlMaster persists 4h.
- Run remote commands non-interactively: `ssh kedi '<cmd>'`.
- No job scheduler (no SLURM). Long jobs go in `tmux`. GPUs are shared with other lab members — check `nvidia-smi` and pin with `CUDA_VISIBLE_DEVICES`.

## kedi hardware (checked 2026-10-18)
- 64 CPU cores, 503 GB RAM, Ubuntu 22.04
- GPUs: 0,1 = L40 48GB; 2,3 = RTX A5500 24GB. Driver 530.30 → max CUDA 12.1
- CUDA toolkits: /usr/local/cuda-11.7, cuda-12.1 (`/usr/local/cuda` → nvcc present)

## Storage
| Path | Notes |
|---|---|
| `/home/sp4050` | NFS (bronze:/student), shared 12T pool, no `quota` cmd. Code/envs. |
| `/local/speech/users` | Local NVMe 7T on kedi, `speech` group writable. Fast I/O for training data. Not backed up, kedi-only. |
| `/proj/speech` | Lab NFS share, mounted on all speech/NLP machines. Group `speech` writable. Only demos/lab/projects/tools backed up. |
| `/tmp` | 125G, 76% full — avoid for large data. |

## Relevant corpora (read-only use; don't modify)
- `/proj/speech/corpora/code-switching/` (34G): miami_bangor (Spanish–English), siarad_bangor (Welsh–English), SEAME, HKUST, Fisher Spanish, etc.
- `/proj/speech/corpora/seame/` (25G): Mandarin–English conversational (LDC2015S04)
- `/proj/speech/corpora/CSpeech/` (25G)
- TTS-adjacent: `libritts`, `LJSpeech-1.1`, `arctic`, `SIWIS`; `/proj/leousers/tts` (older festival/HTS work)

## Software
- System python3 3.10 only; no conda/mamba/uv, no ffmpeg/sox/git-lfs yet → install user-space (e.g. miniforge or uv in $HOME).
- git, tmux, screen available.
