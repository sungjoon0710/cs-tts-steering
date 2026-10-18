# Baseline Plan: Zero-shot TTS on MaSaC transcripts

Goal: establish baseline quality of two open-source Hindi-capable speech LLMs on MaSaC's code-switched transcripts, to pick the base `<model>` for steering. Pilot run must finish in **≤ 1 hour** wall-clock on kedi; the same pipeline then scales to a larger test set by changing one config value.

## 1. Models under test

| | **Indic-Speak** (`bodhan-ai/indic-speak`) | **Orpheus 3B Hindi FT** (`canopylabs/3b-hi-ft-research_release`) |
|---|---|---|
| Backbone | Llama-3.2-3B (3.78B params) | Llama-3.2-3B-Instruct → Orpheus pretrain → Hindi FT |
| Audio path | LM → SNAC codes → **fine-tuned Vocos** decoder (repo `vocos/`) → 24 kHz | LM → SNAC codes → SNAC 24 kHz decoder (`hubertsiuzdak/snac_24khz`) |
| Weights | 6.6 GB bf16 + 0.47 GB Vocos | 13.2 GB **fp32** (load as bf16) |
| Voices | Fixed library (`voices.md`), ~2 Hindi voices (M/F); English voices also usable | Fixed Hindi voice(s) — names not on model card, read from repo/tokenizer after download |
| Prompt | `<speaker>`/`<style>` blocks + text (handled by repo `inference.py`) | `"{voice}: {text}"` wrapped in Orpheus start/end special tokens (per Orpheus reference notebook) |
| Code-mixing | Claimed (English words inside Indic sentences) | Unknown — this is part of what we measure |
| Deps | `transformers>=5`, `snac`, `soundfile` | `transformers`, `snac`; `repetition_penalty ≥ 1.1` required |
| License / access | Indic Open Model License v1.0, HF gated (accept terms) | Apache-2.0, HF gated (accept terms) |

Notes
- Indic-Speak's card snippet references `indic-speak-preview-v2`; the `bodhan-ai/indic-speak` repo contains `inference.py` + weights, so we use that repo and its `TTS` class. **Read `inference.py` before running it** (repo-supplied code); ignore the shipped `__pycache__/*.pyc`.
- Both models are plain HF `AutoModelForCausalLM` + SNAC code decoding, so one wrapper interface covers both: `synthesize(texts: list[str], voice, seed) -> list[np.ndarray]`.
- Inference is fully local with open weights. No vLLM for the pilot (dependency/driver risk with CUDA 12.1 driver); batched HF `generate` is fast enough at this scale.

## 2. Test set (pilot)

Source of truth: `MSH-COMICS/Data/Pickles/Dataset.p` (has `text`, `Speaker`, `Sarcasm`, `Humour`, `Episode_label`, `Audio_Filename` in one row). `Utterances.txt` vs. `Audio_Filename.txt` order/line counts don't obviously line up (15,575 vs 15,576), so don't join the text files by line number. Load the pickle **only with a restricted unpickler** (allow-list of pandas/numpy classes).

Pilot set: **100 utterances**, stratified, fixed seed, saved as an ID list:
- Must contain ≥ 1 English (Latin-script) token *and* ≥ 1 Devanagari token (true code-switching).
- Length 4–25 words (avoids one-word turns and runaway generations).
- Balanced across top ~6 speakers (Maya, Sahil, Monisha, Rosesh, Indravadan, Madhusudan) and across Sarcasm/Humour labels.
- Held-out by episode from any future steering-training split (record episode IDs now so the baseline test set stays the test set).

Text preprocessing (minimal, logged): strip `Speaker:` prefix, normalize ellipses/quotes/whitespace. Keep original mixed script as the primary condition. *Optional ablation (phase 2)*: fully-Devanagari transliteration of English words, to separate "model can't read Latin script" from "model can't code-switch".

Stored in repo (small, no transcript text since repo is public): `configs/testsets/pilot_100.txt` (utterance IDs).

## 3. Generation protocol

- Per model: 100 utts × **2 voices** (one M, one F; map to MaSaC speaker gender as secondary analysis) × **1 seed** = 200 clips/model, 400 total.
- Decoding (both): `temperature=0.6, top_p=0.9, top_k=50`, Orpheus `repetition_penalty=1.1`, `max_new_tokens` capped from text length (≈ 2× expected duration) to stop runaways.
- bf16, SDPA attention, batch size 16 (left-padded), fixed seeds; log per-clip generation time, token count, and stop reason (EOS vs. max-tokens).
- Resumable: skip clips whose wav already exists.
- No style/emotion tags in baseline (Indic-Speak `style` stays empty) — tags are a later condition.

## 4. Evaluation

Reference-free (no ground-truth audio needed — runs in the pilot):

| Metric | Tool | Notes |
|---|---|---|
| Intelligibility: CER / WER | Whisper large-v3 (lang=hi) | Compute after mapping hyp+ref to a common romanization (e.g. `uroman`), since Whisper often writes English words in Devanagari and vice versa. Report overall + on **English-token spans only** (code-switch fidelity). |
| Voice consistency | ECAPA-TDNN (SpeechBrain) | Mean pairwise cosine sim within each voice. |
| Prosody stats | parselmouth (F0 mean/range), Silero VAD (pauses), MMS forced aligner (`torchaudio` MMS_FA, romanized) | Speaking rate, pause count/duration, F0 range; and the same **within ±1 word of language-switch points** vs. elsewhere. |
| Robustness | duration ratio, stop reason | % truncated / runaway / silent clips. |
| Efficiency | wall-clock | RTF per model on L40. |

Reference-based (needs MaSaC ground-truth audio — see §6):
- Emotion SIM: emotion2vec-base cosine(gen, GT).
- Speaker SIM to the real character (ECAPA cosine).
- F0 / rate / pause deltas vs. GT; later AutoPCP and TTSDS.

**Naturalness = human judgment only.** Automatic MOS predictors (UTMOS, SQUIM) are excluded: they are trained on English and unreliable for Hindi–English. Naturalness is judged by Hindi–English bilingual listeners instead. The pilot's deliverable for this is a listening package:
- `outputs/baseline/<run_id>/listening/`: clips renamed to anonymous IDs (model/voice hidden), loudness-normalized, 24 kHz wav; `key.csv` (ID → model, voice, utt) kept separate from what raters see.
- `ratings_template.csv`: one row per clip with transcript, columns for naturalness MOS (1–5) and "sounds like natural conversational code-switching" (1–5), plus an A/B sheet pairing the two models on the same utterance.
- Design: all 100 utts, both models, one voice each (200 clips) for MOS; randomized order per rater; ≥ 3 raters per clip if possible; report MOS with 95% CI and A/B preference rate.

## 5. Where things run and live

Everything runs on **kedi**, inside `tmux`, one model per GPU in parallel:
`CUDA_VISIBLE_DEVICES=0` → Indic-Speak, `CUDA_VISIBLE_DEVICES=1` → Orpheus (both L40 48 GB; each needs ~8–10 GB in bf16, so A5500s are fallback). Check `nvidia-smi` first; GPUs are shared.

| Artifact | Location | Why |
|---|---|---|
| Code, configs, test-set ID lists, small summary tables | repo `~/cs-tts-steering` (git) | versioned; repo is public → no audio, transcripts, or weights |
| Python env | `/local/speech/users/sp4050/envs/tts` (uv venv) | torch+CUDA wheels are large; local NVMe is faster than NFS home |
| Model weights (HF cache) | `HF_HOME=/local/speech/users/sp4050/hf_cache` (~25 GB incl. eval models) | not home (NFS), not `/tmp` (76% full) |
| MaSaC source data | `/local/speech/users/sp4050/data/masac/` | already there |
| Generated audio | `/local/speech/users/sp4050/outputs/baseline/<run_id>/audio/<model>/<voice>/<utt_id>.wav` | |
| Run metadata | `…/<run_id>/{config.yaml, generations.jsonl, metrics.csv, summary.json, logs/}` | `config.yaml` snapshot + git commit hash for reproducibility |

`run_id = YYYYMMDD-HHMM_<testset>`.

Env (**installed 2026-10-18**): uv venv at `/local/speech/users/sp4050/envs/tts` (alias `tts-env`), uv-managed Python 3.11, `torch 2.6.0+cu124` (transformers 5 requires torch ≥ 2.6; cu121 wheels stop at 2.5.1, and cu124 runs on the CUDA 12.1 driver via minor-version compatibility — verified matmul + SDPA on GPU), `transformers 5.20.1`, `accelerate`, `snac`, `soundfile`, `pandas`, `huggingface_hub`, `ipykernel`. Eval deps still to add: HF Whisper, `speechbrain`, `funasr` (emotion2vec), `praat-parselmouth`, `uroman`, Silero VAD. If Orpheus misbehaves under transformers 5, split into two envs.

No shared model-weights cache exists on kedi (lab members keep personal caches), so `HF_HOME=/local/speech/users/sp4050/hf_cache` (set in `~/.bashrc`, along with `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` under `/local/speech/users/sp4050/.cache`).

## 6. Delivery

- **Python scripts for anything compute-heavy** (reproducible, run headless in tmux, resumable):
  - `scripts/prepare_testset.py` → loads MaSaC safely, builds stratified ID list
  - `scripts/generate.py --model {indic_speak,orpheus} --testset pilot_100` → wavs + `generations.jsonl`
  - `scripts/evaluate.py --run <run_id>` → `metrics.csv`, `summary.json`
  - Library code in `src/` (`data.py`, `tts.py` with both wrappers, `metrics.py`).
  - One `configs/baseline.yaml` (models, voices, decoding params, paths).
- **One notebook for review**: `notebooks/baseline_review.ipynb` — reads `metrics.csv`, shows comparison tables/plots and inline audio players (text + both models side by side) for listening. Opened via VS Code Remote-SSH on kedi with the venv as kernel. The notebook never runs generation.
- Commit summary tables (`results/baseline_<run_id>.md`) to the repo; audio stays on kedi.

## 7. Time budget (pilot, both GPUs in parallel)

| Step | Est. |
|---|---|
| Env setup + weight downloads (one-time) | 20–30 min (not counted in run) |
| Test set prep | < 2 min |
| Generation: 200 clips/model, ~4 s audio each (~350 SNAC tokens), batch 16 on L40 | ~5–10 min per model, parallel |
| Whisper large-v3 ASR on 400 clips | ~5 min |
| ECAPA, VAD, F0, alignment | ~10 min |
| Build listening package | < 2 min |
| **Total run** | **~25–35 min** (headroom under 1 h) |

Scale-up: change `testset` to a larger stratified list (e.g. 1,000 utts); generation scales linearly (~1–1.5 h/model), or switch to vLLM for throughput.

## 8. Dependencies / approvals before running

User actions:
1. Accept gated terms on HF for both repos; create an HF read token; `huggingface-cli login` on kedi.
2. Decide on fetching MaSaC GT audio (Google Drive, from MSH-COMICS README) — **only the ~100 pilot clips**, for reference-based metrics. Pilot can run reference-free without it.

Needs explicit OK before Claude does it on kedi:
- Install uv + create env in `/local/speech/users/sp4050/envs/tts` (~8 GB).
- Download model weights + eval models into `/local/speech/users/sp4050/hf_cache` (~25 GB).
- Load `Dataset.p` with restricted unpickler.
- Download GT audio subset.

## 9. Open questions to resolve on first contact
- Orpheus Hindi: special tokens confirmed as standard Orpheus layout (`<custom_token_N>` = 128256+N; audio codes from 128266, 7×4096). **Voice name still unknown** — not in model card, local README, or tokenizer. Smoke test: compare no-voice prefix vs. candidate names and pick the most stable voice. Defaults: `temperature 0.6, top_p 0.9`; add `repetition_penalty 1.1` (required per Orpheus README).
- ~~Indic-Speak Hindi voice names and batching~~ — resolved from the downloaded repo:
  - Hindi voices: **Kavya** (F, ~215 Hz, 12.9 chars/s) and **Amit** (M, ~152 Hz, 12.6 chars/s). All voices are cross-lingual.
  - `inference.py` generates **one utterance at a time** (no batching) → write a thin batched wrapper reusing its `build_prompt` / `ids_to_codes`; left-pad prompts.
  - Vendor's benchmark config is `temperature 0.6, top_p 0.9, repetition_penalty 1.2`, but `inference.py` doesn't pass `repetition_penalty` — set it explicitly (1.2) in our wrapper and record it.
  - Audio rate ≈ 82 tokens/s → `max_new_tokens` ≈ 82 × expected seconds × 2.
  - Model derives pauses from punctuation — keep MaSaC punctuation intact in preprocessing.
  - `vocos/load.py` uses `torch.load(..., weights_only=False)` on `vocos/best.pt` (a pickle from the model publisher). Try `weights_only=True` first in our loader; fall back only if it fails.
  - Style tags available for later conditions: `happy, sad, anger, fear, surprise, disgust`, plus genre tags.
- Does Orpheus Hindi accept Latin-script English, or does it need transliteration? (first 5-utt smoke test answers this before the full pilot).
