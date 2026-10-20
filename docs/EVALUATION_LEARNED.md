# Additional automatic diagnostics

These methods extend the existing evaluation; they do not rerun TTS or replace
bilingual listening. All three model–voice configurations remain separate.
An overlapping challenge view is not an independent extra cohort.

## Language-specific error analysis

`language_evaluate.py` reuses `text.pier` and reads existing MaSaC `Transcript`,
`name`, and `LID_tags` tables. It tries the lab's punctuation tokenizers and a
Unicode letter/mark tokenizer. It accepts a projection only when it uniquely
reproduces the versioned normalized token sequence. It rejects ambiguous or
length-inconsistent rows rather than padding labels.

The optional paired Devanagari table joins by `name`, not row order. Projection
requires equal word counts and unchanged non-Hindi words. This is an automated
projection, not verified alignment. Corpus evidence is deduplicated by normalized
text and label sequence, and repeated words in a row do not inflate support.
Exact baseline texts are excluded when constructing the lexicon. Conflicting
corpus evidence contributes to disagreement, not a forced label.

Primary provisional thresholds are support >=3 rows and agreement >=0.95;
sensitivity settings are (1,1.0) and (5,1.0). Agreement is not calibrated accuracy.
Latin script is never treated as evidence of English. Unknowns remain unknown.

The complete reference/hypothesis token sequences are aligned before selecting
errors at provisional English, Hindi, or observed-switch windows. Switch windows
include the two boundary tokens and one neighboring token on either side.
Insertion anchoring follows the pinned [official PIER code](https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/tree/09535fa0311cfec98176d1a4f7bec82094bb0d4f).
These are **provisional POI diagnostics**, not verified English WER or validated
embedded-language PIER. No matrix language is guessed. True embedded-language
PIER requires verified token and matrix/embedded-language annotations.

Difficulty outputs include coverage, English-proportion bounds, observed-switch
lower bounds and directions. Language condition, CMI, M-index and I-index are
available only for completely labeled inputs. Strategy is not inferred.

## Recognizer sensitivity

`learned_evaluate.py asr` uses `openai/whisper-small` revision
`973afd24965f72e36ca33b3055d56a652f456b4d`: 16 kHz resampling, Hindi transcription,
float16, batch 1, greedy decoding, attention mask, timestamps, no previous-token
conditioning, no input truncation, and `max_new_tokens=440` (within its positional
limit including prompt tokens). Raw and normalized hypotheses are stored
separately. The legacy recognizer was Whisper-large-v3; its exact checkpoint
revision was not recorded. Its batch, masking and long-input handling differ.
This is a second practical recognizer analysis, not a controlled model-size study.

The existing versioned WER/CER and uroman sensitivity functions are reused.
Cross-recognizer WER is a directional disagreement measure, not ground truth.
Two recognizers agreeing does not validate the spoken content. Failed extraction
is null, never zero. The disagreement queue is for manual transcription.

## Learned quality and speaker diagnostics

NISQA uses the official **NISQA-TTS v1** checkpoint, not the transmission-quality
model. The approved execution uses official source commit
`fe84f0f252abec382b24367d5b22498a7ce34dbb`.
The checkpoint is approximately 1.24 MB, and is loaded weights-only. Official
prediction is CPU batch 1. Its checkpoint settings retain native sample rate,
8 kHz maximum mel frequency, 48 mel bins, 20 ms window, 10 ms hop, segment length
15 frames and maximum 6000 segments. Preserve the entire checkpoint arguments
and SHA256 in the private provenance manifest.

[NISQA documentation](https://github.com/gabrielmittag/NISQA) describes the TTS
checkpoint's prediction target. Its Hindi–English validity is not established by
running it here. Scores are model predictions, not human MOS. Checkpoint weights
have CC BY-NC-SA 4.0 licensing; do not redistribute them with this repository.

Speaker embeddings use `microsoft/wavlm-base-plus-sv` revision
`feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`, 16 kHz input, float32, batch 1,
attention mask, and L2 normalization. The [model card](https://huggingface.co/microsoft/wavlm-base-plus-sv)
documents VoxCeleb1 fine-tuning. Report within-configuration mean pairwise cosine
and each recording's leave-one-out centroid cosine. Resample utterances for
uncertainty; the thousands of pairwise similarities are not independent samples.
Same-original-record pairs are excluded in bootstrap replicates. Language,
content, duration and recording domain can affect embeddings. Cosine similarity
is not naturalness and is not a cross-architecture ranking.

## Official TTSDS2 attempt

`official_ttsds_evaluate.py` targets official release 2.1.3, source commit
`8ce85ad9e1072a542dc2a6735932a991ebf6e442`, in an isolated environment. Explicit
approval is required for downloads and shared-GPU execution. Keep a process
timeout outside this entry point. It pins and inventories selected checkpoint
files, limits additional downloads, and runs offline after prefetching.

Use the multilingual ensemble with environment excluded: generic mHuBERT/XLSR/
WavLM; prosody MPM/WORLD pitch/mHuBERT-token rate/Allosaurus phone rate;
WeSpeaker/dvector speaker features; XLSR-phone and multilingual Whisper
activations. Four factor weights are 0.25 each. This is not the older univariate
feature-W1 analysis, whose results must remain intact.
Use one suite worker: official caches and scalar-distance global RNG resets are
not suitable for concurrent benchmark tasks. BLAS/PyTorch CPU threads may be two.

The actual official distance path uses scalar empirical W2 and multivariate
Gaussian Fréchet distance (squared W2), then 100*d_noise/(d_reference+d_noise),
using nearest reference/noise populations. The unused `frechet_distance_fast`
function has a different trace calculation and is not called. For unequal scalar
sample sizes, the official helper reduces the larger sample once; its loop does
not produce ten fresh resamples. Report this implementation limitation.
Inspect feature shapes before interpreting the ensemble: the pinned multilingual
Whisper code indexes each encoder state with `x[-1]` and averages dimension 1.
For `(batch,time,feature)` states this averages features and retains time, then
averages layers. Verify the resulting cached dimension rather than silently
replacing the official calculation. Its frontend also uses default 30-second
padding/truncation. MPM takes at most 1000 256-sample frames (about 11.6 seconds
at 22050 Hz). Other factor populations can weight recordings by their number of
frames. Report temporal coverage and pooling concerns alongside official scores.

Use official DirectoryDataset for both populations, with manifest-exact WAV
coverage. The official WavListDataset embeds absolute paths in cache keys and
can attempt cache writes beside source audio; it is deliberately avoided.
DirectoryDataset averages channels, resamples to 22050 Hz, peak-normalizes, and
pads clips shorter than 16000 samples. Noise TarDataset instead trims edges.
These transformations differ from our native-rate Praat/eGeMAPS extraction.

The pinned WeSpeaker Lightning checkpoint contains four metadata types:
`TorchVersion`, and pyannote `Specifications`, `Problem`, `Resolution`. The
runner verifies the required globals and scopes their allowlist to construction,
retaining weights-only loading. It never enables unrestricted checkpoint loading.
Any unknown class or unresolved compatibility failure terminates the attempt.

Natural-reference speaker, acting style, soundtrack, language mix, segmentation,
channels, and population-size differences can dominate results. Neither a smaller
distance nor a higher official score establishes greater human naturalness.
See [official implementation](https://github.com/ttsds/ttsds) and
[TTSDS2 paper](https://arxiv.org/abs/2506.19441). Preserve source/checkpoint/runtime
manifests and any failed attempts; do not label partial factors as full TTSDS2.

## Execution and validation

`lab_rate_evaluate.py` reuses the existing lab `speakingRate.praat` calculation
through a pinned source hash. Directory I/O is removed, and the calculation runs
on an in-memory Sound. No legacy source is bundled. Settings remain -25 dB,
2 dB dip, 300 ms pause, pitch 30–450 Hz. This detector omits the terminal candidate:
a ten-pulse probe produces nine nuclei; one pulse produces zero. Preserve invalid
detections as null. Its results are legacy nuclei/sounding-time proxies, not
validated articulation. A 25-clip natural-rate comparison is method-provenance
evidence, not justification for transferring natural timing to synthesized audio.

Use the existing CPU environment for language analysis and report building;
keep learned-model/TTSDS dependencies separate from required CPU dependencies.
The observed learned runtime stack was torch/torchaudio 2.8.0, transformers
4.57.1, huggingface-hub 0.36.0, librosa 0.11.0 and NumPy 2.2.6. The isolated
TTSDS stack uses NumPy 1.26.4 and pyannote.audio 3.3.2. Record all installed
versions, including transitive dependencies, for each released run.

`inventory_evaluation_resources.py` inventories only top-level research tables
and notebook source metadata, never executing notebooks. Workbook rows are
counted by actual content, not formatted worksheet dimensions. No transcript rows
are exported by the inventory. `audit_natural_channels.py` adds correlation and
energy diagnostics without altering the approved mean/first-channel extraction.

`analyze_missing_metrics.py` checks generation IDs and waveform hashes before
joining results, writes a separate versioned extension, and preserves missing
statuses. Source/data/results remain separate; all private records and figures
live outside Git. Run deterministic tests before publishing source changes.
