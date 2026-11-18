# n-shot style prompting with natural MaSaC speech (Orpheus-hi-pt)

Status: design (2026-11-12). Nothing generated yet.

**Question.** If Orpheus-hi-pt's prompt contains *n* natural recordings of a speaker (with their text), does its output take on that speaker's prosody, more than with 3 of its own synthetic clips?

**Hypothesis.** 3 shots works best (vs 1 and 5) for prosodic match to the speaker and perceived naturalness.

**Model:** `canopylabs/3b-hi-pretrain-research_release` only. **Indic-Speak is excluded**: its README states "No voice cloning. Voices come from the library only", and its license forbids voice clones of real people.

## Data

30 clips: the 15 unflagged clips each of Maya and Indravardhan (transcript audit, `docs/reports/transcript_methods.pdf`). Text = `DEVANAGARI LLM INPUT` from each clip's `TRANSCRIPT.txt` in `Listening_Examples_100_PT`.

- Maya: `test_175_0`, `train_1394_7`, `val_131_1`, `train_214_1`, `train_1265_2`, `train_588_5`, `train_376_7`, `train_17_2`, `train_549_0`, `train_240_3`, `train_76_0`, `train_1690_0`, `test_173_0`, `train_723_2`, `train_1241_0`
- Indravardhan: `train_1219_3`, `train_1692_2`, `test_158_0`, `train_959_4`, `train_406_1`, `train_1181_6`, `train_291_0`, `train_1472_2`, `val_185_0`, `train_1745_15`, `train_1758_1`, `train_734_1`, `train_1059_0`, `train_1355_2`, `train_378_4`

## Steps (150 clips, seed 1234, ≈15 min on one GPU)

| Step | What | Clips |
|---|---|---|
| 1. `base` | Regenerate PT output for each of the 30 lines from the LLM text, text-only prompt. This is the baseline and the source of the synthetic shots | 30 |
| 2. `nat-1`, `nat-3`, `nat-5` | Each line, with 1, 3 or 5 **natural** clips of the same speaker in the prompt as shots | 90 |
| 3. `syn-3` | Each line, with the **same 3 shots**, but their audio taken from step 1 | 30 |

**Shots for a line:** the next *n* clips of the same speaker in the list above (folder order of the 100 listening examples), wrapping around at the end. The clips are unrelated lines from different episodes, so "next" is an arbitrary, fixed choice of other clips, not dialogue context. The line itself is never one of its own shots.

## Prompt design

Each shot is a text–audio pair given as one completed turn. The target text follows as an open turn, and the model generates its audio. Example: 3-shot, target = Maya line 4, shots = lines 1–3:

```
[SOH] maya_1_txt [EOT EOH SOA SOS] maya_1_wav [EOS EOA]
[SOH] maya_2_txt [EOT EOH SOA SOS] maya_2_wav [EOS EOA]
[SOH] maya_3_txt [EOT EOH SOA SOS] maya_3_wav [EOS EOA]
[SOH] maya_4_txt [EOT EOH SOA SOS]  → model generates maya_4_wav, stops at [EOS]
```

- **Text** (`*_txt`): the `DEVANAGARI LLM INPUT` line, tokenized with the model's Llama-3 tokenizer. No voice name or other prefix.
- **Audio** (`*_wav`): mono, resampled to 24 kHz, encoded with SNAC 24 kHz into 7 tokens per frame from id 128266 (`src/snac_encode.py`, `dedup=False`; ≈ 82 tokens/s).
- **Conditions** differ only in the shot block:
  - `base`: no shots.
  - `nat-1`, `nat-3`, `nat-5`: 1, 3 or 5 natural pairs.
  - `syn-3`: the same 3 texts as `nat-3`, with the step 1 `base` audio in place of the natural recordings.
- **The target's own recording is never in the prompt.** It is used only as the reference in evaluation.

Canopy documents in-context "text-speech pairs in the prompt" for the pretrained Orpheus, but not the token sequence. The layout above follows Orpheus's standard single-turn TTS format, repeated per shot.

## Evaluation

Per clip, with the existing `caa_evaluate.py` tools: Whisper CER, F0 SD/range (semitones), speaking rate, pauses, F0 contour correlation with the line's natural clip, and speaker similarity to the speaker's natural clips. Compare `nat-1` / `nat-3` / `nat-5` (does 3 win?), `nat-3` vs `syn-3`, and each vs `base`, per speaker. Then listen (perceived naturalness).

**Read-out:** if `nat-3` only raises speaker similarity, the shots copied the voice, not the delivery. It counts as style if `nat-3` moves the prosody toward the natural clips more than `syn-3` does, without a CER jump.

Outputs: `$LOCAL_SP/outputs/icl_pilot/<run_id>/{base,nat1,nat3,nat5,syn3}/<utterance_id>.wav` + `generations.jsonl`.
