# Shared representation and steering tools

The upstream CAA API stays in `src/steering.py`. Local inspection utilities have
moved to `src/representation_probe.py`; import them from there rather than
replacing the CAA module. Both APIs are exercised by CPU tests.

|Module|Purpose|
|---|---|
|`src.steering`|Existing `SpeechLM`, `extract`, `Steerer`, `generate_batch`; upstream behavior unchanged|
|`src.representation_probe`|`ZeroSteeringProbe`, `code_phases`, `collect_phases`, plus shared `collect_anchor` and `collect_prefix`|
|`src.position_steering`|`AnchorIntervention`: reuse the CAA norm-preserving hook at the final speech-start prompt state only|
|`src.representation_geometry`|Normalized differences, cosine/consistency, calibration-only means and PCA; reuses upstream PCA|

`IndicSpeak(...)(text, voice, seed, cap, style="", token_callback=None)` retains
the original default decoding. The optional callback receives exact prompt and
generated token IDs before decoding, including when decoding subsequently fails.
Independent speaker embedding extraction is available through the existing
`scripts.learned_evaluate.speaker_embeddings` helper.

## Inspect prompts without generating audio

Keep input manifests and outputs outside Git. A JSONL manifest needs `id`, `text`,
`split` (`calibration`, `heldout` or `validation`) and optionally
`source_document`. IDs use letters, digits, underscores and hyphens. For example,
an **authored fixture**, not a corpus extract:

```json
{"id":"example01","text":"This is an example sentence.","split":"calibration","source_document":"example_doc"}
```

Pass existing pinned snapshot directories explicitly. The runner checks revision
names, vendor source hash and the codec cache ref; it runs offline and does not
download weights. `configs/representation_pilot.json` records the tested model,
voice and labels. The snapshot options below refer to your own cache paths.

```powershell
python scripts/inspect_representations.py prompts --manifest INPUT.jsonl --snapshot MODEL_SNAPSHOT --codec-snapshot CODEC_SNAPSHOT --device cuda --output outputs/representation_run
python scripts/inspect_representations.py analyze --source outputs/representation_run --output outputs/representation_analysis --pca-layers 3 26
```

Only the first command loads a checkpoint; the second reads saved arrays on CPU.
Extraction defaults to CPU unless `--device cuda` is explicit. Check shared GPU
availability before choosing CUDA. No command generates speech.

The matched index saves IDs, conditions, splits, model revision, original prompt
IDs, input hashes and NPZ hashes. Analysis rejects incomplete conditions, altered
files, mixed checkpoints/codecs/voices and mismatched text/split associations. It fits means
and optional PCA on calibration data only. Consistency summaries keep splits
separate and group repeated source documents; pair counts are not independent
sample counts. Zero directions give undefined cosines, never perfect matches.

To isolate equal-length style-word contrasts, analyze with
`--reference control_news --conditions anger happy sad`; optional PCA then uses
only those selected contrasts. Keep one model/codec/voice configuration per run.

## Inspect generated-speech phases

Use the saved **empty-style** prompt for both continuations in a style pair:

```powershell
python scripts/inspect_representations.py phases --neutral-tokens DEFAULT_TOKENS.json --continuation-tokens ANGER_TOKENS.json --snapshot MODEL_SNAPSHOT --codec-snapshot CODEC_SNAPSHOT --layers 8 14 20 --output outputs/phase_run
```

Neutral records accept `prompt_ids` or the older `prompt_token_ids` key;
continuations use `generated_token_ids`. Whole histories remain intact. Prompt,
EOS, invalid and incomplete frames are excluded from statistics, with seven SNAC
phases pooled separately. A processed audio-token state predicts its **next**
token; this is not word alignment. `collect_prefix` instead compares an identical
valid prefix under different prompts, which is a context control.

The preserved `scripts/steering_pilot.py` and `collect_style_smoke.py` remain
available for the frozen Stage 1 workflow. Imports and provenance now reference
`representation_probe`. Their model paths are explicit: Stage 1 requires
`--snapshot` and `--codec-snapshot`; the smoke collector requires `--snapshot`.
They no longer read a personal `model_access.json` file. Their generation
commands are not integration checks and were not run during this integration.

## Position-specific intervention

```python
from src.position_steering import AnchorIntervention

# lm is an already loaded/eval model; prompt is the exact empty-style prompt.
with AnchorIntervention(lm, prompt, layer=26, direction=vector, epsilon=0.02) as hook:
    output = lm.generate(...)  # same input, seed and decoding as the control
assert hook.result()["hooks_removed"]
```

The additive delta is 2% of that selected state's norm before norm preservation.
Only the final prompt state changes, once; other prompt and cached audio states
are untouched. Zero strength bypasses arithmetic for exact BF16 compatibility.
This differs from upstream `Steerer` all-position/generated-position modes.
Use fresh caches, explicit masks and matched zero/random controls. A moved state
does not establish a perceived emotion. The snippet is usage documentation,
not an automatically launched experiment.

## CPU verification

```powershell
$env:CUDA_VISIBLE_DEVICES=""
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m pytest tests/test_caa.py tests/test_representation_probe.py tests/test_representation_geometry.py tests/test_position_steering.py tests/test_prompt_representations.py tests/test_representation_cli.py -q
```

These tests use synthetic fixtures/tiny randomly initialized models, not
checkpoints. Keep Torch/Transformers/Hub versions mutually compatible; the
original Windows environment had an existing Transformers/Hub version conflict.
Tiny-Transformers tests skip when the package cannot import; report that
limitation instead of treating a skip as a real-model pass. No dependency change
is required by the new pure geometry functions.

Audio, checkpoint files, arrays (`.npy`/`.npz`), private manifests, listener records,
GPT handoffs and run outputs stay outside the public repository. Extraction and
analysis refuse existing output directories. Completed historical runs are not
migrated, regenerated or overwritten by this integration.
