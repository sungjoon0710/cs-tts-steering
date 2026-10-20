# Full automatic evaluation, version 0.2

This extends the preserved v0.1 pilot. The user authorized automatic evaluation of
the whole baseline before human validation. `pilot_evaluate.py` retains its original
pilot gate; `full_evaluate.py` is the separate automatic entry point. Automatic
results are provisional, acoustic extensions exploratory, and human data pending.
Three configurations are two architectures with different voices. There is no
overall quality score or architecture-only ranking.

## Records, provenance and exclusions

The baseline is 100 inputs × three configurations. An 18-recording challenge view
overlaps that baseline; it is never added as extra independent data. The original
pilot includes 60 of the same 300 recordings. Generation attempts, successful
retries, selected audio, ASR and measurements have separate records. Keep intended
outputs in denominators even when measurements are unavailable. Missing audio/ASR
produces null content scores, never WER zero. Report metric coverage separately.

Snapshots hash allowlisted metadata and inventory WAV headers; audio is read in
place and SHA256 checked. Feature cache keys include audio, extractor code hashes,
package versions and settings. Code releases are immutable content-addressed
copies, verified before execution. Results are checksum verified on retrieval;
conflicting existing local files require a new destination. No deletion sync.
Code/config/tests/docs belong in Git; private texts, labels, audio, manifests with
text, environments and results stay outside this public checkout.

## Metric registry

| Dimension / question | Operational definition | Input, validation and interpretation |
|---|---|---|
| Content proxy: do ASR and intended words agree? | WER = (S+D+I)/N; full token edit alignment retained. CER = character edits / reference Unicode code points after whitespace removal. | Cached legacy ASR; newly published upstream source specifies Whisper large-v3, Hindi transcription. Original cache checkpoint revision/hash unverified. Recognizer and synthesis errors confounded. Manual intended↔spoken, spoken↔ASR, intended↔ASR comparisons needed; rates are not additive. |
| Normalization sensitivity | Raw NFC/case/punctuation-sensitive errors; primary `hi-en-nfc-v1` case/punctuation normalization; separate uroman 1.3.1.1 romanization sensitivity. | Preserve original/normalized strings. No translation or deletion of disfluencies. uroman is neither validated phonetic equivalence nor a Romanized-Hindi recognizer. Devanagari CER counts combining code points, not grapheme clusters. |
| Code-switch-specific content | Official PIER reference-token POIs, edit-aligned substitutions/deletions and anchored insertions; trailing insertion assigned to last reference token if POI. | Existing deterministic implementation tested. Verified aligned token-language labels required for primary PIER. Sparse exact matches and mismatched tag/token counts cannot support baseline-wide PIER. Latin script is not English. |
| Reliability | Missing/unreadable/empty/fully silent; PCM clipping; RMS < −50 dBFS near-silence warning; retained generation token-cap and original short flags. | Warnings are not confirmed perceptual failures. Legacy short rule observed as duration / max(1,raw-character-count/12) < .4. New excessive-duration ratio >2.5 is an exploratory queue rule, not a validated threshold. |
| ASR warning proxies | D/N ≥ .25; I/N ≥ .50; repeated normalized trigram occurrences beyond first / total trigrams ≥ .25. | Possible omission, extra/repeated content; ASR can cause all three. Zero trigrams with available ASR gives repetition zero; missing ASR remains null. |
| Pitch | Praat autocorrelation, 10 ms step, 60–600 Hz, other parameters frozen in `audio.py`. Voiced median/mean, SD Hz/semitones, Q05–Q95 robust range, contours, edge fraction and autocorrelation strength. | Voiced frames only for pitch statistics. Check octave errors/overlapping speakers and floor/ceiling sensitivity. No direction is inherently better. |
| Energy/intensity | Digital RMS dBFS, peak/clipping; Praat intensity median/mean/SD/range/IQR using minimum pitch 60 Hz, 10 ms. | Digital recordings lack SPL calibration; Praat intensity is not calibrated real-world loudness. Level and recording-domain confounds remain. |
| Silence/pause | Praat relative intensity threshold −25 dB, minimum silence .15 s, sounding .10 s; internal pauses ≥.20 s, sensitivity .15/.25 s. | Silence segmentation is not VAD ground truth. Unvoiced sounding and intensity-silent frames separately classified; silence can still contain pitch from background signals. |
| Rate proxies | Normalized ASR word count / full duration; articulation proxy denominator = first-to-last sounding span minus internal pauses ≥.20 s. | Neither is validated spoken-token rate. Intended word counts never substituted. Segmentation, short pauses and ASR omissions confound these estimates. |
| Lab voice quality | Praat pulse extraction 75–600 Hz; local jitter period bounds .0001–.02, maximum factor1.3; local shimmer maximum amplitude factor1.6; HNR cc .01 s/75 Hz/.1/1period. | Reuses shared SpeechLab extractor settings. Exploratory on connected synthetic/TV speech; not naturalness or clinical voice scores. Undefined values retained null. |
| eGeMAPSv02 | All 88 openSMILE functionals, 16 kHz float32, SciPy polyphase resampling Kaiser beta5. | Version2.6.0 recorded. No learned checkpoint. Configuration-centered rank correlations identify redundancy, not perceptual utility. Full DisVoice deferred until a specific nonredundant hypothesis justifies new dependencies. |
| Distributional question | Per-feature empirical univariate W1; optional division by natural-reference IQR, never arbitrary averaging. | Unpaired natural population, actor/channel/language/domain confounds. Not TTSDS; lower distance is not established higher quality. |
| Human naturalness | Existing blinded three-pair UI; naturalness before transcript; pronunciation/switch smoothness after transcript; all four preference categories retained. | Pending bilingual listeners. Repeated utterances/listeners need clustered or mixed-effects analysis. No fake scores. |

Edit distances use [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html).
Acoustics follow [Praat](https://www.fon.hum.uva.nl/praat/manual/Intro.html) and
the accessible lab extractor. Feature definitions/configuration follow
[openSMILE Python](https://audeering.github.io/opensmile-python/) and
[eGeMAPS](https://doi.org/10.1109/TAFFC.2015.2457417). Published validation of
these descriptors for other tasks does not validate them as Hinglish TTS quality.

## Statistics and natural-reference population

Configuration distributions, medians/means/Q05/Q95, corpus-weighted content errors,
paired utterance mean differences (2,000 bootstrap replicates), and leave-one-out
influence are separate. Influential examples stay in primary aggregates.
Conversation mapping for the baseline is unverified, so paired uncertainty is
exploratory and cannot claim conversation-level independence. Difficulty views use
intended length bins and script profiles; validated language conditions pending.

Natural references require filename-based WITS joins plus exact NFC/casefold/
whitespace text and speaker agreement. The revised exploratory population samples
100 distinct WAV hashes, one per conversation, round-robin across the ten most
frequent single-name source speaker labels, without merging aliases. Source labels
are not verified diarization. Natural stereo channels are averaged **in memory**;
20 seeded references are repeated using the first channel. Original WAVs unchanged.

Distribution uncertainty: 200 bootstrap replicates resample natural episode
clusters and synthetic utterances. Synthetic episodes are unknown. Reference
train/val/test subset and duration-nearest 50-item views are sensitivity checks,
not replacements for the primary population. Sample-size curves use 100 subsamples
without replacement at n=20,40,60,80,100; their ranges are subset variability, not
confidence intervals. At n=100 no subsampling variation remains by construction.
Discuss channel averaging, source-speaker selection and domain mismatch with the
supervisor before interpreting these reference comparisons.

## Execution

Use `scripts/sync_evaluation.py --help` for immutable release upload, CPU launch,
status and verified result retrieval. Remote processes disable CUDA and limit
BLAS/OpenMP to two threads. No TTS or ASR inference runs. `research-push` transfers
top-level feature JSON/JSONL only, not audio. `compare` runs descriptive distribution
analysis against a completed baseline in the caller's workspace.

Local natural extraction: `natural_distribution.py extract --data DATA --output PRIVATE`.
Prepare separate orthography records with `prepare_orthography.py`; it creates
pending fields, not transliterations or speech. Equivalent content, seed, exact
voice/checkpoint and decoding must be checked before script-effect claims.

Kedi environment reuses readable shared Python3.10 packages through an own-env
`.pth`; additional small CPU dependencies live only in the user's environment.
This is reproducible by recorded versions/hashes but not hermetic: shared packages
can change. Capture provenance per run; a future locked standalone environment
would remove that dependency. Local/remote dependency versions can differ; perform
numerical parity checks rather than assuming binary equivalence.

## Additional automatic diagnostics

The CPU-only execution statements above describe `full_evaluate.py` and
`natural_distribution.py`. Separately approved learned-model entry points use
their own run directories and budgets; they do not change the existing cached
CPU measurements. See [EVALUATION_LEARNED.md](EVALUATION_LEARNED.md) for language
POI recovery, second-ASR sensitivity, NISQA-TTS, within-voice embeddings and the
isolated official TTSDS2 attempt. Run-specific metric registries remain beside
private results, outside this public repository.
