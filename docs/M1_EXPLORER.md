# M1 saved-trace explorer integration

M1 replays saved audio-token histories. It does not decode new WAVs or execute the
SNAC/Vocos decoder. The existing viewer now accepts these histories without a
recording, shows their real representation/probability results, and explicitly
marks waveform and decoder views unavailable. Original c03/c04 decoder and audio
views retain their existing data paths.

## CPU export

Run the exporter after the selected source subjects have completed. It accepts
the existing M1 output tree, not checkpoints or a model-loading configuration:

```powershell
python scripts/export_m1_explorer.py --root M1_ROOT --output NEW_PACK --model-metadata M1_ROOT/inputs/model_metadata.json --exact-ids FROZEN_ID_1 FROZEN_ID_2 --vector-row-limit 7 --archive NEW_PACK.zip
```

Replace the placeholder IDs with prespecified cohort IDs having saved full
vectors. Without `--ids`, all cohort subjects and all their completed histories
are exported. `--ids ID ...` explicitly restricts an export; its summary records
the exact coverage. Without `--exact-ids`, every context is summary-only. Exact
vectors are copied only for primary history h0, retaining at most the final seven
*already captured* prediction rows, all model layers and the three saved sites.
They are not newly collected. Missing exact captures are listed in the summary.
Verified reference-token-cap failures are listed in `unavailable_subjects` with
their evidence hashes, without an invented capture. Unattempted subjects and
unresolved technical failures still reject a final export.

The exporter copies scalar heatmap NPZs, per-position probability CSVs and context
JSON unchanged and verifies their completion hashes. It makes smaller derived
vector NPZs and records both source and exported hashes, row IDs, actual model
metadata, protocol hash, cohort hash and context associations. It does not copy
phase-contrast tensors, full logits, model weights or audio.

`export_summary.json` lists uncompressed bytes and exact coverage. The optional
archive's companion `.zip.summary.json` and terminal output include measured
compressed bytes, archive SHA256 and a 256 MiB target-budget flag. An over-budget
flag does **not** remove scientific arrays or alter values. Decide transfer scope
after inspecting this result. `--compression lzma` is an optional standard-library
alternative; default ZIP deflate is faster.

## Registration and launch

Each `NEW_PACK/<utterance>/h<index>/manifest.json` is an ordinary
`capture-manifest-v1` registration source. Artifact paths stay within its directory.
Register all manifests through the existing `Catalog.register_spec` API into the
same local explorer catalog; do not edit frontend condition lists. Conditions are
discovered from exact style, speaker, prompt template, model and replay metadata.
Launch using the existing `scripts/model_explorer.py` serve command documented in
the main Model Explorer README.

The additive token-only history declaration is:

```json
{"label":"reference history", "recording":null,
 "tokens":{"path":"history.json", "sha256":"ACTUAL_FILE_SHA256"}}
```

The token JSON must contain complete seven-phase SNAC frames followed by one
normal EOS token. Registration checks each forced prefix and its explicit
processed/predicted phase, input/target token and absolute prompt offset. It does
not fabricate missing vectors from distances. Decoder/waveform API responses use
`available:false` with a reason rather than an error or synthetic output.

Viewer comparison summaries include the speech-start row because that is the
existing visualization convention. The exported metadata separately preserves
M1's generation-only JSD and top-token statistics, excluding j=0. Positions within
one utterance are dependent; the viewer's curves are descriptive, not a substitute
for utterance-level confirmation analyses.

## Validation

`tests/test_model_explorer.py` includes explicit synthetic unit fixtures for
token-only registration, missing decoder/audio, exact versus summary-only vectors,
phase rejection, byte-preserved scalar files and archive-size reporting. Set
`SPEECHLAB_TRACE_ROOT` to the saved November 15 experiment to include immutable
c03/c04 CPU regression checks. No inference, network or checkpoint is needed.
