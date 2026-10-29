# Preliminary study: CAA steering of TTS speech LMs toward natural code-switched speech

Status: plan (2026-10-29). Method and evaluation follow CoCoEmo (Wang et al., ICML 2026; see [`papers.md`](papers.md)), adapted from emotion to naturalness.

**Question.** Do the language models of Indic-Speak and Orpheus Hindi contain a linear "natural vs. synthetic" direction that we can add at inference to make their Hindi–English code-switched speech sound more natural, without hurting intelligibility or changing the voice?

**Three-way comparison.** Each condition gets its own steering vector:

| Condition | Model | Voice |
|---|---|---|
| **Indic-Amit** | Indic-Speak (`bodhan-ai/indic-speak`) | Amit (M) |
| **Indic-Kavya** | Indic-Speak | Kavya (F) |
| **Orpheus-Ritika** | Orpheus 3B Hindi FT (`canopylabs/3b-hi-ft-research_release`) | ऋतिका |

All three use the mixed Devanagari/English input.

This is a feasibility study. It needs a clear go/no-go answer per condition (§8), not a polished system.

---

## 1. Decisions

| Topic | Decision | Source |
|---|---|---|
| Contrast | 100 natural recordings vs. 100 default outputs per condition, paired on the same transcript | user |
| Input script | **Devanagari/English mixed only** (the `*_Devanagari.wav` files). Romanized files are not used | user |
| Conditions | Three steered conditions: Indic-Amit, Indic-Kavya, Orpheus-Ritika. One vector per condition | user |
| Speaker mismatch | Natural clips come from many speakers; synthetic clips are a fixed TTS voice. Accepted for now. Measured with S-SIM and listening, not removed | user (3a) |
| Natural audio cleaning | None beyond required format conversion (mono, 24 kHz) | user (3a) |
| Split | 80 train / 20 test at the **utterance** level. All versions of an utterance (natural + all three conditions) stay in the same split. **One split shared by all conditions.** No grouping by episode or dialogue | user |
| Objective naturalness metrics | All three: P(natural) classifier, embedding similarity to the paired natural clip, prosody distance (§6) | user |
| Human evaluation | Bilingual listeners compare clips side by side after steering (§6.3) | user |
| Data | Partner's structured folder `Listening_Examples_100` (§3) | user |
| Model selection without a test leak | **Default:** layers, sites and α are chosen per condition, only on the 80 training utterances (5-fold CV for probes; a 16-utterance dev subset for steering sweeps). The 20 test utterances are used once | default, open |
| Baselines | **Default:** no steering (α=0), random vector of the same norm, reversed vector (−v) | default, open |

---

## 2. How CoCoEmo works and how we adapt it

| | CoCoEmo | This study |
|---|---|---|
| Model | CosyVoice2 / IndexTTS2: speech LM + flow matching. Steer only the LM | Indic-Speak and Orpheus Hindi FT: both Llama-3.2-3B LMs (28 layers, d=3072) → SNAC codes → audio decoder (Vocos for Indic-Speak, stock SNAC for Orpheus). Only the LM is steered, which matches CoCoEmo's main finding |
| Contrast | Emotional vs. neutral recordings, same speaker and same transcript | Natural vs. synthetic recordings, same transcript, different speakers |
| How audio enters the LM | Emotional clip given as the reference prompt | **Teacher forcing:** text prompt + the clip's SNAC tokens (§4.1) |
| Activation | Last token, 10 hook sites per layer | Residual stream and attention output (`o_proj`) per layer; last audio token **and** mean over audio tokens |
| Vector | mean(emotion) − mean(neutral) | v = mean(natural) − mean(synthetic), per condition |
| Site selection | Linear-probe accuracy, then layer-wise steering sweep; top-K, K=1–5 | Same procedure per condition, run only on training utterances |
| Injection | h ← h + α·v, rescaled to ‖h‖, all positions | Same |
| α choice | Sweep 0.5–7.0; keep the strongest α whose WER stays within +0.5 of no steering | Same idea; guardrails in §5.2 |
| Metrics | E-SIM, TEP, S-SIM, WER, N-MOS | P(natural), similarity to the paired natural clip, prosody distance, S-SIM, WER/CER, side-by-side listening |

CoCoEmo compared two backbones (CosyVoice2, IndexTTS2) and found that the best layers differ by model. Our Indic-Speak vs. Orpheus comparison is the same kind of comparison: two backbones with the same architecture but different training. The two Indic-Speak voices add a within-model comparison.

---

## 3. Data intake

**Data:** [`/local/speech/users/aa6109/Listening_Examples_100/`](vscode://vscode-remote/ssh-remote+kedi/local/speech/users/aa6109/Listening_Examples_100) (kedi). It is also linked from `/local/speech/users/sp4050/Listening_Examples_100` (symlink). Open it in VS Code Remote-SSH or with `ssh kedi`; the `listen.html` inside is a player page for all clips.

| File | Contents |
|---|---|
| `NN_<utterance_id>_<first words>/` (100 folders) | `01_Natural_<SPEAKER>.wav` (original stereo), `02_IndicSpeak_Amit_Devanagari.wav`, `03_..._Amit_Romanized`, `04_..._Kavya_Devanagari`, `05_..._Kavya_Romanized`, `06_OrpheusHindi_Ritika_Devanagari.wav`, `TRANSCRIPT.txt` |
| `utterances_100.csv` | One row per utterance: `utterance_id`, `source_character_label`, `Romanized`, `Devanagari` |
| `manifest.csv` | 600 rows: per wav, `configuration_id`, `generation_id`, original `source_path`, SHA-256, `token_cap_warning`, input text |
| `READ_ME.txt`, `publication.json`, `source_identity.json` | Provenance. The wavs are byte-identical copies of the `v1_supplied_spellings` run; nothing was regenerated or normalized |

**Files used in this study:** `01` (natural), `02` (Indic-Amit), `04` (Indic-Kavya), `06` (Orpheus-Ritika). `03` and `05` (romanized) are not used.

Facts checked on 2026-10-29:
- Access (as sp4050): the folder and its files are readable **and writable** through an ACL. The parent `/local/speech/users/aa6109/` cannot be listed, but this path can be opened directly.
- Natural speakers: Indravardhan 18, Maya 18, Monisha 17, Rosesh 16, Sahil 16, Dushyant 15. That is about 35 female and 65 male clips.
- Token-cap warnings in the files we use: **4 Orpheus-Ritika clips** and **1 Indic-Amit clip**. These may be cut off (AGENTS.md notes Orpheus truncating long inputs). Listen to them. For the main analysis, drop those utterances from all conditions, so the three conditions keep the same utterance set.
- The `train_`/`val_`/`test_` prefixes in the IDs (88/6/6) are the **source corpus's** split labels, not ours. Our 80/20 split is drawn separately.
- The Devanagari inputs are machine-transliterated and contain spelling errors. Natural and synthetic passes get the same string, but keep it in mind when reading WER.
- Don't write into this folder; treat it as read-only input. Our outputs go to `$LOCAL_SP`.

`manifest.csv` + `utterances_100.csv` already cover the fields below. The table is kept as the contract for any future data version.

| Field | Notes |
|---|---|
| `utterance_id` | Stable ID. Only IDs and split lists go into git |
| `text_devanagari` | The exact string used for generation. The natural clip is paired with the same string |
| `natural_wav` | Path on kedi |
| `natural_speaker` | Source label. For reporting only |
| `synthetic_wav[condition]` | Paths to the Indic-Amit, Indic-Kavya and Orpheus-Ritika Devanagari outputs |
| `generation` | Seed, decoding settings, checkpoint revision (from the original run's records) |

Checks before any extraction:
- 100 natural clips and 300 synthetic clips (3 conditions). Every row resolves to readable files.
- No duplicate transcripts. If two `utterance_id`s share a normalized transcript, they go to the same split.
- Log natural-clip channels, sample rates and durations (reuse `scripts/audit_natural_channels.py`).

**Split:** shuffle once with a fixed seed, after dropping the token-cap utterances. The test IDs go in `configs/caa_pilot/split.json` (IDs only). A 16-utterance **dev** subset of the training utterances is used for steering sweeps.

---

## 4. Phase 1: extract activations and run diagnostics (cheap, no generation)

### 4.1 Teacher forcing

Each clip is fed through the model of the condition, with that condition's voice:

| Model | Input sequence |
|---|---|
| Indic-Speak | `build_prompt(text, speaker=v)` + audio tokens + `<|end_of_speech|>` |
| Orpheus | `128259` + tok(`"ऋतिका: {text}"`) + `128009, 128260` + `128257` + audio tokens + `128258` |

- **Audio tokens:** wav → mono, 24 kHz → SNAC encoder (`snac_24khz`) → 7-token frame interleave, each code offset by `base + position·4096`. The frame layout is the same in both models (`ids_to_codes` in Indic-Speak's `inference.py`, `snac_codes` in `src/tts.py`).
- **Model-specific details:**
  - Indic-Speak removes consecutive duplicate frames (same `c0`) at encode time (`token_contract.md`). Orpheus does not.
  - The audio-token base id differs between the two models. Read it from each tokenizer and don't hard-code it.
  - Test the encoder by round trip for each model: tokens → decoder function → same codes.
- **Synthetic clips are also re-encoded from their wavs**, not taken from the generated tokens. Both classes then go through the same SNAC encoding.
- The natural pass uses the same voice name as the synthetic one. Within a condition, the only difference between the two inputs is the audio.
- Natural activations are extracted **separately for each condition**: each model, and each speaker tag, gives different activations for the same audio. Each condition therefore has its own natural mean and its own v.

### 4.2 What to record

Per (condition, clip), one forward pass with hooks:
- Every layer: residual-stream output and attention `o_proj` output.
- Two poolings: last audio token, and mean over all audio tokens.
- Per-token NLL of the audio tokens. This is a diagnostic: natural audio will look unusual to the model, so v may partly encode "this audio is unusual for the model".

Size: 3 conditions × (100 natural + 100 synthetic) = 600 forward passes. Under 15 minutes on one GPU, including loading both models.

### 4.3 Diagnostics (train utterances only, per condition)

1. **Linear probes.** For each (layer, site, pooling): logistic regression, L2-regularized, features standardized, optional PCA to 64 dims, 5-fold CV. Folds are by utterance, so the natural and synthetic versions of a transcript are never on opposite sides.
   - Expect near-100% accuracy almost everywhere, because speaker and recording channel make the classes easy to separate. This is why probes alone can't choose the site (CoCoEmo App. B makes the same point).
2. **Difference-of-means projection.** Compute v on 4 folds. Project the held-out fold onto v and report AUC. This tests the exact direction we will inject.
3. **Permutation control.** Shuffle the natural/synthetic labels within each utterance pair and recompute. This gives the chance level for 1 and 2.
4. **Vector geometry.**
   - ‖v‖ per layer relative to the typical ‖h‖.
   - Cosine between Indic-Amit's and Indic-Kavya's vectors (same model). High cosine means the direction does not depend on the voice.
   - Indic-Speak vs. Orpheus: the two models have the same architecture but different weights, so a raw cosine between their vectors is not meaningful. Compare their layer profiles instead: which layers separate the classes, and the size of v.

**Output:** for each condition, a shortlist of 5–8 candidate layer/site/pooling choices, ranked by projection AUC.

---

## 5. Phases 2–3: steered generation

### 5.1 Injection

A forward hook at the chosen sites computes `h ← h + α·v`, then rescales to the original ‖h‖.
- **Default:** apply at every position, prompt and generated tokens (as CoCoEmo).
- **Ablation:** generated tokens only.
- Decoding stays at each model's defaults, with the same token budget as the baseline run, fixed seeds, and resumable runs (skip a clip whose wav exists):

| Model | Decoding |
|---|---|
| Indic-Speak | t 0.6, top_p 0.9, top_k 50, repetition penalty 1.2 |
| Orpheus | t 0.6, top_p 0.9, repetition penalty 1.1 |

### 5.2 Choosing sites and α (dev subset, training utterances only, per condition)

1. Layer-by-layer sweep on the dev subset: 16 utterances × shortlisted sites × α ∈ {2, 4} × 1 seed. Score with M1 and WER (§6). Check whether steering effect rises with projection AUC (CoCoEmo Fig. 7).
2. Top-K: K ∈ {1, 2, 3} from the best layers. α ∈ {0, 0.5, 1, 2, 3, 4, 6} on the dev subset.
3. **Selection rule** (fixed before looking at test, same rule for all three conditions):
   - Keep only configurations that pass all three guardrails:
     - WER/CER at most the condition's own baseline +2 points absolute. n is small, so this is looser than CoCoEmo's +0.5.
     - S-SIM to the unsteered voice ≥ 0.75.
     - Failure rate ≤ 5%. A failure is a missing end-of-speech token, hitting the token budget, or a duration outside 0.5–2× the baseline.
   - Among the configurations that pass, pick the one with the best average improvement across M1–M3.
   - Bring two α values to test: the selected one and a milder one.
   - Orpheus has a weaker, truncation-prone baseline. Its guardrails are measured against its own baseline, not Indic-Speak's.

### 5.3 Test run

Test utterances (~20) × 3 conditions × 3 seeds × these generation conditions:

| Generation condition | Purpose |
|---|---|
| α=0 | Baseline. Re-generated with the same seeds, so it differs from the original run only by seed handling |
| v at α\* and at the mild α | Main result |
| Random vector, same norm, same sites, α\* (3 draws) | Is it the direction, or any perturbation? |
| −v at α\* | Sanity check: should become *less* natural |

About 20 × 3 × 3 × 7 ≈ 1,260 clips. At RTF ≈ 1.8 single-stream on an L40, this is about 3 h, and less with batching or running the two models on two GPUs. The dev sweeps add about 1.5 h.

---

## 6. Evaluation

All objective metrics are computed against the **paired natural clip** of each test utterance where relevant, and reported **per condition** as the change vs. that condition's α=0. Statistics:
- 95% bootstrap CIs, resampling test utterances (seeds averaged within an utterance).
- Paired Wilcoxon test, steered vs. baseline and steered vs. random.

**Three-way comparison.** Compare the *gains* (steered − baseline) across Indic-Amit, Indic-Kavya and Orpheus-Ritika on the same test utterances (Friedman test, then pairwise Wilcoxon). Compare gains, not absolute scores: the baselines differ in quality, and a weaker baseline may have more room to improve.

### 6.1 Naturalness (in place of CoCoEmo's E-SIM and TEP)

| ID | Metric | Details |
|---|---|---|
| **M1** | P(natural) | Logistic-regression classifier on **external** audio embeddings (not TTS-model activations, to avoid circularity). Trained on the training utterances' natural clips vs. **all three conditions' synthetic clips**, so one shared classifier scores every condition. Report mean P(natural) per condition, and AUC against the held-out natural clips. Secondary, mechanistic check: re-teacher-force each steered output through its *unsteered* model and project onto v |
| **M2** | Similarity to paired natural clip | Cosine similarity of utterance embeddings, steered vs. paired natural, minus unsteered vs. paired natural |
| **M3** | Prosody distance | F0 (semitones: median, SD, range), energy, speaking rate, pause count and total pause time: absolute difference from the paired natural clip, scaled by the natural set's IQR. F0 and energy contour correlation after DTW alignment (CoCoEmo-style CCC). Reuse the Praat settings in `docs/EVALUATION_FULL.md`. Optional: the same features in windows around code-switch points, if word alignment is available |

Embedding for M1/M2: **default** WavLM-base-plus mean-pooled hidden states. Using emotion2vec (CoCoEmo's choice) needs a new download, so ask first. UTMOS/SQUIM are not used (project rule).

Caveats:
- M1 and M2 can reward moving toward the natural speakers' **voice or recording channel** instead of a more natural delivery. Read them together with S-SIM and listening.
- The natural speakers are ~65% male. Amit is male; Kavya and Ritika are reported female (Ritika's gender is unverified, see `docs/model_inputs.md`). A gender-related shift in pitch could therefore show up as a "naturalness gain" for the female voices. Report M3's F0 features separately and check S-SIM per condition.

### 6.2 Guardrails

| Metric | Details |
|---|---|
| WER / CER | Whisper-large-v3 with the repo's existing normalization (`hi-en-nfc-v1`), against the input transcript |
| S-SIM | WavLM-SV cosine (already pinned in `docs/EVALUATION_LEARNED.md`) between the steered clip and the unsteered clip of the same condition and seed. Measures how much the voice drifts toward the natural speakers |
| Failure rate | Missing end-of-speech token, hitting the token budget, or abnormal duration. Watch truncation in Orpheus especially |

### 6.3 Human side-by-side listening

- **Raters:** Hindi–English bilingual listeners only.
- **Items:** ~20 test utterances × 3 conditions = ~60 pairs. Each pair is unsteered vs. steered (α\*), same condition and seed, order randomized, model labels hidden.
- **Questions:**
  1. Which sounds more like natural conversational Hinglish? (A / B / no difference)
  2. Naturalness of each clip on a 1–5 scale (CoCoEmo's N-MOS).
  3. After the transcript is shown: was anything mispronounced or lost?
- **Optional:** the natural recording as an anchor at the end of each item (different speaker; listeners are told so).
- **Setup:** reuse the existing blinded comparison interface (`docs/EVALUATION_FULL.md`, "Human naturalness"). Mix pairs from all three conditions in random order. Also include a few random-vector items as a control.
- **Analysis:** steered preference rate per condition with a binomial CI (ties reported separately), and inter-rater agreement. The steered N-MOS of the three conditions gives the cross-model comparison.
- **Size:** ~60 pairs, about 20–25 minutes per rater. Aim for ≥ 5 raters.

---

## 7. Outputs and code

Outputs go to `$LOCAL_SP/outputs/caa_pilot/<run_id>/`, following AGENTS.md §7:
- `config.yaml`, git commit
- `activations/<condition>/` (pooled, float16)
- `vectors/<condition>.pt`
- `probes.csv`
- `generations.jsonl` and wavs
- `metrics/`
- `listening/`

`<condition>` ∈ {`indic-amit`, `indic-kavya`, `orpheus-ritika`}.

Planned code (nothing written yet):

| File | Contents |
|---|---|
| `src/steering.py` | Hook registration, teacher-forced extraction, injection with norm rescaling. Model-agnostic: both models are `LlamaForCausalLM` |
| `src/snac_encode.py` | wav → audio tokens for each model (inverse of `ids_to_codes` / `snac_codes`; deduplication for Indic-Speak only), with round-trip tests |
| `scripts/caa_extract.py` | Phase 1 extraction and diagnostics |
| `scripts/caa_generate.py` | Dev sweep and test run (resumable, seeded). Builds on `src/tts.py` (`IndicSpeak`, `OrpheusHindi`) |
| `scripts/caa_evaluate.py` | M1–M3, guardrails, three-way comparison. Reuses `src/evaluation/` |

Commit only IDs, configs and summary tables. No audio and no transcripts.

---

## 8. Go/no-go criteria for the full study

Judged **per condition** on the test utterances. Steering is **feasible** for a condition if:
1. **Direction exists:** the held-out projection AUC is clearly above the permutation control (§4.3).
2. **Direction is causal:** at α\*, M1 and at least one of M2/M3 improve over both the α=0 and random-vector conditions (95% CI excludes 0), and −v moves them the other way.
3. **Model stays usable:** all §5.2 guardrails hold on test.
4. **Listeners agree:** listeners prefer the steered clip more often than the unsteered one, excluding ties.

The study is a **go** if at least one condition passes. The three-way comparison then says which backbone and voice to take forward. A pass for both Indic-Speak voices but not for Orpheus, or the reverse, points to a model difference. A pass for only one Indic-Speak voice points to a voice difference, possibly the gender mismatch in §6.1.

If 1 and 2 pass but 4 fails, the vector likely captures speaker or recording channel rather than naturalness. The next step would then be cleaner contrasts: speaker-matched natural data, denoised natural audio, or synthetic-vs-synthetic controls.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| The vector encodes speaker identity or recording channel (TV audio, laugh track) instead of naturalness | Accepted for the pilot. Detected by S-SIM, by M1 vs. listening disagreement, and by comparing the Amit and Kavya vectors |
| Natural speakers are mostly male; two of three TTS voices are female | Report per condition; look at F0 shifts separately (§6.1) |
| Orpheus baseline is weaker and truncates some inputs | Drop token-cap utterances from all conditions; compare gains, not absolute scores; Orpheus guardrails relative to its own baseline |
| Natural audio is unusual input for the model; the vector partly encodes "surprising audio" | Track NLL per class; compare v with the direction separating high- and low-NLL synthetic clips |
| Small n (~80 per class, d=3072) | Difference-of-means is stable at this size; probes are regularized; CIs come from bootstrap over utterances |
| Teacher-forcing re-encoded clips ≠ the model's own tokens (Indic-Speak's Vocos output, Orpheus's SNAC output) | Both classes go through the same path. Optional check: compare with the original generated tokens if the run saved them |
| Picking α on test | Selection rule fixed in advance and run on dev only |

---

## 10. Order of work

1. Data checks on `Listening_Examples_100` (§3), including listening to the 5 token-cap clips.
2. SNAC encoder with round-trip tests for both models. Extraction for all three conditions. Diagnostics (§4). **First feasibility read** at this point, with no generation yet.
3. Dev sweeps (§5.2) and selection, per condition.
4. Test generation (§5.3) and objective evaluation (§6.1–6.2).
5. Listening package and ratings (§6.3).
6. Short report with a go/no-go decision per condition and the three-way comparison (§8).
