# SpeechLab Model Explorer

Local, CPU-only exploration of **saved** computations. No checkpoint is loaded,
no extraction or synthesis is started, and nothing is sent to an external API.
Vanilla HTML/CSS/Canvas/JavaScript and Python's standard-library HTTP server are
used with the existing NumPy installation. No new dependencies or CDNs.

## Launch on Windows

From the integrated worktree, using the existing environment:

```powershell
& ..\..\cs-tts-steering\.venv\Scripts\python.exe scripts/model_explorer.py serve --catalog ..\..\evaluation_runs\model_explorer_20261116\catalog.sqlite --port 8766 --open
```

The browser opens at `http://127.0.0.1:8766`. Keep the terminal open; Ctrl+C stops
the foreground server. **No Kedi, SSH, VPN or internet connection is needed.**
The delivered local run also has `START_MODEL_EXPLORER.ps1`, which opens an already
running explorer or launches it. The service binds only to loopback. APIs are
read-only; registering data is a separate explicit CLI operation.

## A useful first investigation

1. Open c03 and its first registered fixed-history prompt comparison.
2. Choose layer 14, j=175, phase 0. The summary maps and next-code predictions
   exist; full hidden vectors **do not** exist there. The UI reports this gap.
3. Explicitly choose a captured position to inspect attention, MLP and residual
   vectors. Step through layers without changing that position. Pin 5 and 26.
4. Compare unit-normalized profiles across pinned layers or fit **one shared PCA**
   to those layers and both contexts. Clicking a PCA point selects its recorded
   layer/position without refitting the coordinates.
5. Inspect predictions, codebook lookups, quantized latent channels and executed
   Vocos outputs. Seeking uses the enclosing code-frame grid, not word timing.
6. Change from a fixed-history prompt comparison to a fixed-prompt history
   comparison and observe the different scientific interpretation.

Blue/orange identify the **selected A/B contexts**, not permanent emotion or voice
labels. Heatmaps support layer/position ranges, zoom, pan, phase filters, hover,
selection, normalized L2/cosine distance, marginal curves, labeled percentile
clipping and locked scales. Decoder heatmaps display bounded, exact channel/time
slices, not invented full tensors. Spectral phase uses a cyclic color scale.

## Storage and compatibility

The original five-table trace SQLite schema describes a single completed run.
It lacks an experiment collection and independent condition/recording identities.
Rather than altering it, a **separate version-1 catalog** adds dataset, model,
experiment, utterance, condition, recording, audio history, replay context,
capture, source and comparison entities. Historical SQLite/NPZ/CSV/audio files
remain authoritative and unchanged. Catalog identifiers hash canonical semantic
metadata; labels/filenames are not context identities.

Large arrays stay in their existing compressed files. The loopback server loads
only requested contexts/comparisons/decoder files and returns bounded slices.
The six-file NPZ LRU is bounded; JSON catalog entries omit full position/token
arrays until their context is selected. Browser requests never run a model.
The catalog stores paths, so on another machine register that machine's copies;
do not assume the original absolute paths exist there.

Legacy registration reads and verifies the old database, foreign keys, artifact
paths, sizes and SHA256 hashes. It checks exact source audio hashes, rebuilt
decoder-code histories, vector dimensions and condition associations. A legacy
bundle must contain one checkpoint/voice. Model-compatible cross-voice data can
instead use separate conditions in the generic capture manifest.

```powershell
python scripts/model_explorer.py register --catalog NEW_PRIVATE_CATALOG.sqlite --trace EXISTING_TRACE_1 --trace EXISTING_TRACE_2 --experiment "Shared experiment label" --dataset "Shared dataset label" --language en --settings EXISTING_EFFECTIVE_CONFIGURATION.json --acoustics EXISTING_PER_RECORDING.csv
python scripts/model_explorer.py verify --catalog NEW_PRIVATE_CATALOG.sqlite
```

Keep the catalog outside Git. Reusing the same labels, metadata and identical
artifacts on a colleague's machine preserves stable view identities.

## Registering new styles, speakers, languages and interventions

No frontend edits are required. The available conditions come from the catalog.
There is no frontend enumeration of Default, ANGER, Happy, Sad or other styles.

For a **metadata-only** prospective condition, create a JSON specification with
`experiment_id`, `label` and `metadata`, then run:

```powershell
python scripts/model_explorer.py register-condition --catalog CATALOG.sqlite --spec CONDITION.json
```

Required metadata: exact `style` string (possibly empty), `speaker`, `language`,
`model_id`, `model_revision`, `prompt_template` identity and `settings`. Optional
script, codec, seed and intervention fields can be added. Registering a label
creates **no results**: it appears disabled until an actual replay is registered.
The current production catalog contains only the two real conditions.

For new completed traces, use `register-manifest --catalog ... --manifest ...`.
This adapter reuses the saved numerical format while allowing arbitrary filenames
and condition/history names. The UTF-8 JSON manifest has:

|Field|Required contents|
|---|---|
|`schema_version`|1|
|`experiment`|`name`, `dataset`, `model` with revision, codec_revision, layers, hidden_size and dtype; reuse the exact registered model metadata for compatible coordinates|
|`utterance`|Stable source `id`, canonical `text`, metadata including language/provenance|
|`conditions`|Dictionary of keys → `{label, metadata}`; exact style, speaker, language, prompt_template and settings; model identity is attached from the experiment|
|`histories`|Keys → label, recording (`audio` artifact reference and metadata), `decoder_npz`, `decoder_json`|
|`contexts`|Records with key, condition key, history key, metadata, `arrays`, `predictions` and capture_levels|
|Context metadata|Exact prompt_ids, history_tokens, positions; prompt_length and evidence; token pieces where captured|
|`comparisons`|Optional records with a/b context keys, arrays/predictions artifact references, and original summary metadata|
|Artifact reference|`{path: relative_path, sha256: actual_hash}`; paths must remain inside the manifest's directory|

The context NPZ uses the existing `resid`, `attn`, `mlp`, `prediction_rows`,
`prompt_embeddings`, `final_norm` keys. Predictions JSON uses `top_predictions`.
Comparison NPZ/CSV and decoder NPZ/JSON retain the existing trace contracts.
All fields refer to **actually saved** data. Test fixtures are explicitly synthetic
and never registered in the production catalog.

Future causal captures must explicitly record `evidence: causal_intervention` and
`intervention.executed: true`, with the actual intervention metadata. The viewer
does not infer causal execution from a condition's name. Current traces have no
intervention. Mixed condition/history changes remain flagged as confounded.

## What the current data support

|Measurement/component|Availability and treatment|
|---|---|
|All 28 layers, all compared positions|Original normalized L2/cosine-distance maps and marginal summaries|
|Actual hidden vectors|Only first/last 7 prediction rows in each context; all layers at those 14 rows|
|Incoming residual|Previous saved block output; layer 0 only has the saved prompt-anchor embedding at j=0|
|After-attention residual|BF16 addition of incoming residual and attention output; exposed only after exact final-residual verification using the saved MLP branch|
|Attention/MLP normalization inputs|Not captured; normalization weights unavailable; not reconstructed|
|Final model RMSNorm|Exact selected rows; independent of selected Transformer block|
|Token probabilities|Recorded final-LM statistics and top five; no per-layer logits or reconstruction of full distributions from top five|
|SNAC code IDs/lookups|Actual three code streams and all 8 lookup dimensions, strides 4:2:1|
|Projected codec branches|First 16 of 768 channels; first-16 stride/sum reconstruction can be checked against full z_q|
|Quantized latent|All 768 channels; selectable bounded slices and full selected-frame vectors|
|Vocos|Recorded stem, final pre block, up-conv, final post block, final norm and head projection; selected 16 channels (head projection originally 514)|
|Other individual Vocos blocks|Executed according to saved architecture, but their intermediate tensors were not captured|
|Predicted spectrum|All 257 saved log-magnitude/phase bins; exact slices; no extra STFT extraction|
|Waveform/acoustics|Saved decoder replay and original WAV, plus existing acoustic rows joined by exact SHA256|

Reconstruction is not inverse subtraction through rounding. Raw A/B vector values
and differences can be exported; pinned layer profiles and shared PCA use unit
normalization to avoid treating raw scale as comparable evidence. PCA is a
descriptive projection over dependent captured positions, not an emotion model.
No reference direction is defined here, so no invented directional-alignment
score is displayed. Future selective recapture is separate work; this application
never invokes it.

### Verified legacy EOS correction

The original v1 extractor wrote `lp[eos]` into columns called
`*_eos_probability`. Those saved columns contain **log p**, confirmed from source
and the negative values. The adapter preserves them as `*_eos_probability_raw`,
also exposes `*_eos_logp`, and displays `exp(log p)` as the probability. Exports
include the encoding note and both representations. No historical file or
extractor was changed. Existing JSD, likelihood, layer maps and audio values are
unchanged. This is a deterministic compatibility conversion, not new inference.

### Comparisons and interpretation

Only a registered ordered comparison supplies all-position normalized L2 and
full-distribution statistics. Reverse or unrecorded comparisons cannot be invented
from summary distances/top-five values; the UI reports their absence. Compatible
selected exact vectors can still be compared directly in either order.
Different utterances/checkpoints or incompatible phase grids are rejected.

Holding an audio history fixed controls code IDs; it does not remove prompt-length
or conditioning-context effects. Changing histories does not align phonemes,
words or linguistic timing. The deterministic decoder receives codes, **not** the
style prompt. It is expected to be identical when those inputs/settings are
identical. RMS of hidden states and spectral values is not loudness or emotion.
No RMS score is invented for wrapped spectral phase.

## Saved views and exports

Save/open/restore a lightweight view JSON with experiment/context IDs, layer/site,
prediction position, phase/window/scaling settings and up to two pinned layers.
Unknown IDs/modes are rejected. A colleague needs the corresponding registered
artifacts; views contain no model, audio or activation arrays. Local browser
storage is used only for the most recently saved view.

PNG exports contain the current scientific canvas. CSV/JSON exports preserve
available numerical data and selection/provenance. Missing vectors do not cause
an old hidden plot to be exported as current data. Full numerical data from an
unseen vocabulary distribution are not fabricated.

## Verification

```powershell
$env:SPEECHLAB_TRACE_ROOT="PATH_TO_SAVED_NOVEMBER_15_TRACE_RUN"
python -m pytest tests/test_model_explorer.py tests/test_generation_trace.py tests/test_trace_viewer.py -q
```

Real-artifact tests are opt-in via that variable and CPU-only. Standalone unit
tests use explicitly labeled test fixtures. Browser acceptance checks exercised
c03/c04, every layer, pinned layers 5/26, j175/phase0, missing vectors, shared PCA,
SNAC/Vocos, audio seeking, saved views, exports, metadata-only discovery and mobile
layout. Full historical inventories are verified before serving.
