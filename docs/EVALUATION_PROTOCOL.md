# Baseline evaluation protocol, version 0.1

Historical pilot protocol. The pilot CLI is `scripts/pilot_evaluate.py`.
The separate full automatic extension is described in [evaluation.md](evaluation.md)
and [EVALUATION_FULL.md](EVALUATION_FULL.md). It can run before human validation,
while preserving provisional/exploratory statuses. `scripts/evaluate.py` is the
upstream model-based evaluator and is a different entry point.

The experiment compares three **model–voice configurations**, not three
architectures: Indic-Speak/Amit, Indic-Speak/Kavya, and Orpheus Hindi/ऋतिका.
Cross-model differences also involve voices. No architecture-only causal claim
is supported. The current baseline consists of consecutive MSH-COMICS source
lines 0–99, not the curated/episode-held-out test set originally proposed.

## Outcomes and validation

Primary evidence is bilingual human preference for natural conversational
delivery. Naturalness is collected before the intended text is shown by default;
`--transcript-first` is an explicit, logged pilot alternative. Naturalness locks
before pronunciation/content and switch-smoothness judgments. Four response
categories remain separate: A better, B better, both acceptable, both unacceptable.
Choose a preferred side when there is one; “both” indicates no preference and an
acceptability judgment. Switch judgments additionally allow not applicable and
uncertain. Do not infer emotion or empathy from these judgments.

Twenty provisional utterances cover all configurations (60 recordings). Sampling
uses 14 mixed-script, 3 Devanagari-only, and 3 Latin-only inputs, distributed over
length bins (<=7, 8–20, >20 normalized tokens), seed 1234. **Script strata are not
language labels**. A bilingual researcher must check labels, input quality, and
coverage before finalizing selection. If the manifest changes, regenerate trials
before collecting ratings; never mix ratings across selections.

All three unordered configuration pairs are retained. Each participant receives
one pair per utterance (20 pairs), with shuffled order and randomized A/B side.
Enrollment slots cycle through all pairs for each item. Fifteen complete listeners
would give five judgments per item/pair; this is an operational target, not a
power calculation. Partial participation and listener proficiency must be reported.
The interface is loopback-only; no remote hosting or participant recruitment occurs.

Challenge inspection includes all six legacy token-limit warnings and two
short-duration warnings. Items outside the representative pilot are a separate
cohort. Do not estimate population failure frequency from enriched challenge items.

## Content preservation

Preserve raw intended and ASR texts. Normalization `hi-en-nfc-v1` applies NFC,
Latin case folding, whitespace collapse, curly-apostrophe/nonbreaking-hyphen
glyph mappings, and replaces punctuation with spaces except internal apostrophes,
hyphens, and numeric separators `. , : /` between digits. Hindi letters and marks,
English insertions, repetitions, and digits are preserved. There is no translation,
number verbalization, token deletion, or implicit script conversion. Words are
whitespace tokens after this normalization. CER uses non-whitespace Unicode code
points (combining marks count), not grapheme clusters. This conservative score can
penalize legitimate cross-script spelling variants: report that limitation until
bilingual reviewed equivalences are available; never silently relax the score.

WER = (S+D+I)/reference-word count; CER uses character edit distance/reference
character count. Preserve full token alignments, operation counts, denominators,
and software versions. WER can exceed 1 because of insertions. Report both corpus
ratios and item distributions. Missing inputs yield null/status, not WER=0.
An empty hypothesis with nonempty reference yields deletions; an empty reference
has undefined WER and an explicit invalid-reference status.

Manual transcription is collected without showing the target text, tied to the
audio SHA256. Record completeness and reviewer ID. A complete human transcription
enables intended→spoken and spoken→ASR comparisons, alongside intended→ASR.
These are not additive. A single reviewer is not an adjudicated ground truth;
independent review is recommended. Incomplete annotations remain pending.

Legacy ASR hypotheses are reused; no new ASR backend has been selected or run.
Checkpoint, decoding, and cache-to-audio hash provenance are unknown. Log warnings
include absent attention masks. Proxy results are **computed, not validated**.

## PIER and language annotations

Reference: Ugan et al., ICASSP 2025, https://arxiv.org/abs/2501.09512.
Implementation checked at official commit
`09535fa0311cfec98176d1a4f7bec82094bb0d4f`, `jiwer/process.py:get_idsh`.

Align the complete normalized reference and hypothesis using RapidFuzz editops.
Count substitutions/deletions/insertions whose reference index is a POI; count
trailing insertions when the final reference token is a POI. Divide by POI tokens.
Store fractions internally (official helper reports percentages). Empty POI sets
are not applicable, not zero. Pin RapidFuzz because alignment tie-breaking matters.

PIER requires verified token labels, exact normalized tokens and source text hash.
HI/EN/OTHER are supported; MIXED or UNCERTAIN keep language metrics pending.
For code-switched inputs, a researcher must explicitly identify matrix language;
the embedded-language tokens form POIs. Monolingual items are N/A. Latin script
alone is never an English label. Optional switch-window POIs are a future named
variant, not silently substituted for embedded-word PIER. Strategy is never inferred.
English proportion uses HI+EN tokens as denominator. Switch counts use adjacent
HI/EN tokens; OTHER interrupts adjacency. No utterance/conversation row-order joins.

## Reliability and acoustics

Keep preprocessing failures, generation failures, retries, technical audio failures,
token-limit warnings, and human-confirmed failures separate. Legacy “runaway” is
observationally equal to hit_max; it is not a confirmed perceptual failure. Legacy
expected duration equals max(1, input characters/12); short flags match ratio<0.4.
Keep those warning definitions without promoting them to ground truth.

Human reviews explicitly record truncation, major omission, repetition, hallucinated
content, severe artifacts, unintelligible segments and runaway behavior. Pending
reviews are not passes. Show reviewed coverage, confirmed-failure lower bounds,
and all selected recordings in denominators. Attempt counts are separate from
utterance/configuration counts. In particular, line 24 had input-parser failures
and successful retries in all configurations.

Acoustics use original PCM amplitudes, without loudness normalization. Record file
hash, sample rate, channels, duration, RMS dBFS, peak and clipping fraction. Stereo
is explicitly unsupported until a conversion policy is chosen. Exact zero-energy
audio is fully silent; unvoiced speech is not silence.

Praat/Parselmouth autocorrelation pitch: 10 ms step, 60–600 Hz, voicing threshold
0.45, silence threshold 0.03, 15 candidates; octave/jump/voiced-unvoiced costs
0.01/0.35/0.14. Store voiced median Hz, semitone SD and Q95−Q05 range, and pitch
tracks. Flag near-bound estimates; range/SD are relative to clip median and do
not compare absolute male/female pitch as quality. Praat intensity is in its
uncalibrated digital-input convention, not calibrated physical loudness.

Pause candidates use Praat intensity-based silence segmentation (-25 dB relative
threshold, minimum silence 150 ms, minimum sounding 100 ms, 10 ms frame step).
This is not validated linguistic VAD. Preserve intervals, distinguish edge silence
from internal pauses, use 200 ms as primary internal-pause threshold and show
150/250 ms sensitivity. Bilingual/manual boundary spot-checks remain necessary.

Intended character count is never labelled speaking/articulation rate. Spoken
words/total duration require a complete verified human transcript. Articulation
rate additionally requires verified segmentation; its denominator is the first-to-
last sounding span minus qualifying internal pauses. Record missing values/statuses.

## Analysis and stopping criteria

Distributions, paired B−A differences, and utterance-block bootstrap intervals are
exploratory pilot diagnostics. All configurations for each sampled utterance remain
together. Episode/conversation dependence is unresolved because those IDs are
missing. Human summaries retain all four categories; with enough listener/item
clusters, crossed listener-and-utterance bootstrap weights account for both repeated
dimensions. Small-cluster intervals remain exploratory, not confirmatory evidence.

Difficulty reports may use intended length now. English proportion, switch direction,
count and strategy remain pending verified annotations. Text indices do not rank
audio quality. Human/automatic disagreement analysis requires actual human ratings.
No composite quality score, predicted MOS, emotion/empathy score, steering,
speaker-embedding metric, TTSDS or switch-boundary metric is included.

The automated pilot passing tests/extraction is not human validation. Freeze the
protocol and enable full scaling only after ASR/manual transcription, annotation,
pitch/pause, failure-review and listening-rubric checks. The CLI rejects `--full`
until `pilot.json` explicitly records `protocol_frozen: true`; change it only after
documenting the human validation decision. No TTS regeneration is required.
