# Generation-time introspection

This extends the shared probes without changing `src/steering.py`, historical
outputs, inference settings or model weights. `src/generation_trace.py` provides
read-only all-layer hooks and a SQLite artifact index. `scripts/trace_generation.py`
replays saved tokens; `scripts/view_generation_trace.py` verifies and visualizes
saved results on CPU. The viewer is a self-contained HTML file, not a service.

## Controlled comparison

Start with one saved English default/ANGER pair from the same checkpoint, voice,
text, seed and decoding settings. Each source needs its exact prompt IDs, generated
token IDs, audio hash and generation manifest. The runner checks them against the
rebuilt pinned prompts. It refuses existing output directories, altered inputs,
invalid/incomplete selected frames, nonfinite states and unexpected hooks.

Replay a 2×2 design, with **no new LM sampling**:

|Context|Prompt|Teacher-forced audio history|
|---|---|---|
|default__default|Empty style|Saved default continuation|
|anger__default|ANGER|Exactly the same default continuation|
|default__anger|Empty style|Saved ANGER continuation|
|anger__anger|ANGER|Exactly the same ANGER continuation|

Prompt contrasts hold every audio-prefix ID fixed. History contrasts hold the
prompt fixed but compare different generated histories at the same serialized
code index. Those histories are **not phonetic or word alignments**. Neither
contrast isolates emotion from all possible conditioning/content effects.
The ANGER prompt includes four extra tokens in the verified pilot. Absolute
positions remain those used by the original inference; they are explicitly saved.

For N target codes, feed `prompt + history[:N-1]`. Row j is the state predicting
target code j: row 0 processes speech-start, row j>0 processes code j−1. Save both
processed and predicted phases. N is a multiple of seven, bounded by the shorter
valid continuation and `--max-frames`. EOS, prompt states other than the anchor,
invalid frames and incomplete selected frames are not target codes.

## What is captured

- All model blocks: post-block residual, projected attention branch (`o_proj`),
  and MLP branch, before their residual additions. Final model RMSNorm is separate.
- Input IDs/token pieces and prompt embeddings. No attention probability matrix
  is requested from SDPA. Branch outputs are not attention weights.
- For every target position: full-vocabulary JSD, conditional predicted-phase
  JSD, entropy, expected-phase probability mass, EOS probability, common-target
  log probabilities and argmax agreement. Top five raw predictions are retained.
- Exact selected hidden tensors: first seven and last seven prediction rows at
  every layer/site, plus prompt embeddings and selected final-normalized states.
  All-position distance maps remain available. Other full hidden states and
  full-vocabulary logits are temporary CPU data, not persisted.
- The **complete original continuations** are decoded separately, even when
  the aligned LM comparison uses a shorter prefix. Capture the code IDs, actual
  selected codebook lookup vectors at `out_proj`'s input, projected-branch
  summaries, summed quantized latent, selected Vocos stages and predicted complex
  spectrum (log magnitude and phase), plus the replay waveform.
- Only executed decoder sites are listed. The verified checkpoint uses three
  quantizers with strides 4:2:1 and a Vocos iSTFT head. Its stock SNAC waveform
  decoder and disabled noise modules are not traced or claimed to execute.

The runtime checks exact residual reconstruction in the model dtype, hook cleanup,
fresh uncached passes, finite states, exact no-hook logits in the first context,
quantizer stride/sum reconstruction, waveform dimensions, and replay discrepancy
against the saved PCM. A one-PCM-step difference is expected when comparing float
replay to a quantized original; report the actual error, not bit-identical audio.

## Numerical definitions

`normalized_L2 = ||changed-reference||₂ / ||reference||₂`.
`cosine_distance = 1 - cosine(reference, changed)`; zero directions are undefined.
JSD uses natural logarithms and **the full vocabulary**, with a separate JSD after
conditioning both distributions on the expected next-phase codebook interval.
For log-probability deltas the target token is identical on both sides:
the saved history for prompt contrasts, the saved default target for history
contrasts. A history contrast's poor default-target likelihood is expected under
a different continuation and is not a content-error score.

These are **raw LM distributions**. They precede temperature, repetition penalty,
top-k and top-p transformations; they are not the historical sampler's probability
of its draw. Top-token changes are diagnostic, not newly generated decisions.
The exact historical draws remain saved input IDs. Tokens/frames within one
sentence are dependent; do not treat them as independent experimental repetitions.

## Storage and commands

Keep outputs outside Git. The SQLite database contains run/context metadata,
every aligned position, comparisons and hashed external artifacts. NPZ files are
compressed and organized per context/comparison/decoder, so another tensor
database dependency is unnecessary for this bounded pilot. SQLite is metadata,
not a store for model weights or giant tensor blobs. Read-only verification checks
its integrity, foreign keys, file paths, byte counts and SHA256 values.

```powershell
# On Kedi, choose an idle GPU, offline caches and the existing TTS Python.
python scripts/trace_generation.py --source SAVED_RUN --id c03 --snapshot MODEL_SNAPSHOT --codec-snapshot CODEC_SNAPSHOT --output NEW_PRIVATE_TRACE --max-frames 64 --device cuda --base-commit 42e3a159f2c61f5f3e2c94679bcdd15d24b61b29

# Local CPU-only analysis; repeat --trace for a second small case.
python scripts/view_generation_trace.py --trace PRIVATE_TRACE --output NEW_ANALYSIS --metrics EXISTING_PER_RECORDING_CSV --source-configuration EXISTING_EFFECTIVE_CONFIGURATION_JSON --source-run EXISTING_LOCAL_RUN_MIRROR
python -m pytest tests/test_generation_trace.py tests/test_trace_viewer.py -q
```

`--source-run` additionally verifies exact saved prompt/history IDs and inverts
the stored SNAC codebooks to check every decoded code against the original tokens.
`--metrics` joins existing acoustic results by generation ID **and audio SHA256**.
No ASR or acoustic extractor reruns in the viewer. The original generation config
is included as supplemental provenance without modifying raw extraction outputs.

## Interpretation boundary

Large hidden displacement does not establish a causal control point. A useful
next test is a single-position residual patch on a fixed history, at a specified
next-code phase, with zero and norm-matched random controls. Compare its immediate
logit effect and downstream propagation before sampling more speech. Observational
tracing neither fits an emotion vector nor validates a perceptual emotion claim.
