# Reproducible offline evaluation

This evaluates existing baseline WAVs and cached ASR without loading speech models.
The three model–voice configurations are matched by utterance ID; voice differences
confound architecture-only conclusions. Human bilingual listening remains the
primary evidence of conversational naturalness. No composite quality ranking.

## Entry points

| Script | Purpose |
|---|---|
| `evaluate.py` | Upstream evaluator: may download/load Whisper and WavLM, run GPU inference and write into its input run. Use only with separately authorized compute and an owned output directory. |
| `audit_legacy_evaluation.py` | Reproduce upstream text metrics and flags from cached CSV, CPU only. No import/execution of the model-based evaluator. |
| `fetch_baseline.py` | Read-only SSH export of allowlisted metadata and selected WAVs; verifies hashes. |
| `pilot_evaluate.py` | Prepare provisional 20-input pilot, score cached ASR, incorporate reviewed annotations and create reports. |
| `full_evaluate.py` | Full automatic baseline analysis with explicit provisional/exploratory status, independent of pending human validation. |
| `listen.py` | Existing loopback-only blinded three-pair pilot UI, transcription and failure-review tasks. |
| `natural_distribution.py` | Select joined WITS/MaSaC reference WAVs, extract features and compare descriptor distributions; not TTSDS. |
| `prepare_orthography.py` | Private paired-input scaffold. Does not invent transliterations, equivalent-content reviews or generated samples. |
| `sync_evaluation.py` | Immutable own-workspace Kedi code/feature sync, CPU execution, result hashes and non-overwrite checks. |

## Environment and local checks

Python3.10+; small CPU dependencies are pinned in `requirements-evaluation.txt`.
Create a personal environment and install those requirements only where authorized.
Existing shared environments can be reused, but record actual versions per run.
No model checkpoints are required. The optional browser smoke test uses an existing
Edge/Playwright installation and skips when unavailable; it downloads no browser.

```sh
python -m pytest -q
python scripts/full_evaluate.py --baseline /path/to/readable/baseline --run ../evaluation_runs/full_run --cache ../evaluation_runs/cache
python scripts/audit_legacy_evaluation.py --source ../evaluation_runs/full_run/source --output ../evaluation_runs/legacy_audit
```

Keep **all outputs outside every Git checkout**; evaluator entry points enforce
this. The source baseline remains read only. Each full run captures allowlisted
manifests/logs, WAV header inventory, selected generation/attempt associations,
audio SHA256, code hashes, configuration, package versions and timestamps. Results
include per-record alignments, denominator/coverage, distributions, paired bootstrap
intervals, outlier influence, warning counts, acoustic redundancy and diagnostics.

Pilot after fetching metadata to an owned private `source/`:

```sh
python scripts/pilot_evaluate.py prepare --run ../evaluation_runs/pilot
python scripts/fetch_baseline.py --remote-root /local/speech/users/OWNER/outputs/baseline --out ../evaluation_runs/pilot/source --selection ../evaluation_runs/pilot/pilot.json
python scripts/pilot_evaluate.py run --run ../evaluation_runs/pilot
python scripts/pilot_evaluate.py report --run ../evaluation_runs/pilot
python scripts/listen.py --run ../evaluation_runs/pilot
```

Listening binds to `127.0.0.1:8765`. Do not expose this local service publicly. Each
listener sees one randomized pair per item, with all three configuration pairs
balanced across enrollment slots. Naturalness is locked before transcript reveal
by default; an explicit pilot option can change reveal order. All four categories
are retained. Human records use pseudonymous IDs and remain private. No scores or
model identities appear in the blinded trial API. Do not change an active pilot
selection while collecting responses.

## Metrics and validation

[EVALUATION_FULL.md](EVALUATION_FULL.md) defines normalization, alignment, failure
handling, pitch/intensity/segmentation, voice quality, eGeMAPS and statistics.
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) records the historical v0.1 pilot.

- Raw, conservative normalized and uroman-harmonized content errors are different
  representations. CER in our evaluator removes whitespace and counts Unicode
  code points; upstream CER includes normalized spaces. Do not pool them.
- Intended↔manual speech, manual speech↔ASR and intended↔ASR comparisons isolate
  different questions; their rates cannot be added/subtracted to recover truth.
- PIER requires aligned verified token-language labels and the official insertion
  policy. Latin script is not English; corpus row index is not a verified join.
- Missing/invalid audio remains in intended-output denominators with null content
  metrics. Token limits, duration and ASR warnings are review candidates, not
  confirmed perceptual failure labels. Failed attempts and retries are separate.
- Acoustic descriptors/eGeMAPS are diagnostic rather than naturalness scores.
  ASR-word rates use explicit proxy labels. Unvoiced speech and intensity silence
  are distinct. Human review, segmentation and pitch tracking need validation.
- A challenge view can overlap the baseline; never add it as extra independent
  recordings. Paired intervals resample utterances; unknown conversation clustering
  limits inference. Repeated listeners/items need the appropriate clustered analysis.

## Newly published upstream formulas

Upstream commit `165ea45` identifies Whisper large-v3 with `language=hi`,
`task=transcribe`, float16, batch16 and torchaudio 16k resampling. WavLM
`microsoft/wavlm-base-plus-sv` computes within-voice mean pairwise cosine of
normalized embeddings. Those declarations do not recover original cache-to-audio
hashes or checkpoint revisions. The audit script can reproduce existing text
metrics without either model.

Legacy text processing is uroman → lowercase → replace non-ASCII alphanumeric
characters with spaces → collapse whitespace. Its “English survival” extracts
Latin regex words of at least two letters and independently fuzzy-matches each
against any romanized hypothesis token at SequenceMatcher ratio≥.5. It counts
Romanized Hindi, ignores token order and can reuse one recognized token. Treat it
as a **legacy Latin-token fuzzy-match diagnostic**, not verified English-content
preservation or PIER. Legacy runaway is `hit_max OR duration_ratio>2.5`, and short
is `duration_ratio<.4`, with expected duration `max(1,len(raw_text)/12)`.

## Private reference comparisons and orthography

Natural stereo handling must be explicit: reject, first channel, or channel average
in memory. Averaging is not universally correct; inspect channel sensitivity and
pitch/segmentation outliers with the supervisor. Original WAVs remain unchanged.
The natural reference selector uses filename and exact normalized text/speaker
joins, one clip per conversation, duplicate-audio exclusion and a documented
speaker-label population. Speaker aliases, acted delivery, language, background
and recording domain remain possible confounds.

Univariate W1, reference-IQR scaling, episode-cluster bootstrap and sample/reference
subset checks are separate descriptive diagnostics. They are not full TTSDS or
perceptual rankings. Full TTSDS, speaker embeddings, learned MOS and additional ASR
need checkpoint/domain/compute validation and separate execution authorization.

Orthography experiments require semantic IDs, equivalent-content review, matched
voice/checkpoint/seed/decoding and separate output identifiers. Existing Latin-only
inputs do not prove Romanized-Hindi support. No preparation script regenerates TTS.

## What belongs in Git

Commit reusable code, tests with synthetic examples, configurations and methodology.
Keep audio, weights, corpus transcripts/labels, listener records, per-record results,
private audits and GPT handoffs outside Git. `.gitignore` is a secondary guard;
stage explicit allowlisted paths and inspect the complete staged diff before push.
No datasets or generated results are included with this implementation.

Kedi sync operates under the connected user's own `~/speechlab_tts_eval/` and
never changes collaborator permissions. Excluded personal report/GPT tooling is
not included in new code releases. Feature/owned-result transfers require explicit
commands. Sync refuses path traversal, checksum mismatches and differing existing
files; it never deletes remote/local files or silently overwrites a newer run.
