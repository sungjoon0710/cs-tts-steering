# Re-cutting the 100 natural clips to their transcripts

Goal: replace each `Listening_Examples_100_PT/<folder>/01_Natural_<SPK>.wav` with a clip that starts and
ends where its romanized MaSaC transcript (O) starts and ends. Motivation: `docs/reports/transcript_methods.pdf`
flags 16/100 clips with boundary errors (line cut off, or another speaker included).

Script: `scripts/realign.py`. Outputs: `$LOCAL_SP/outputs/align/<run_id>/`.

## Inputs
| Input | Source |
|---|---|
| Old clip | `01_Natural_<SPK>.wav` (44.1 kHz stereo) |
| Romanized transcript O | line `(0)` of `<folder>/Transcripts.txt` |
| Conversation wav (conversation mode only) | `<conversations>/<utterance_id>.wav`, supplied separately |

## Steps
| # | Step | Tool | Why | Source |
|---|---|---|---|---|
| 1 | Locate the old clip in the conversation | FFT cross-correlation (torch) | Recovers the old boundaries exactly; a weak peak means a wrong file | standard signal processing |
| 2 | Search window = old span ± 5 s | — | Boundary errors sit at the edges; window catches dropped words and prevents matching the same phrase elsewhere | — |
| 3 | Normalize O to lowercase a–z and `'` | — | MMS_FA vocabulary is romanized letters; O is already Latin, so no uroman needed | torchaudio multilingual alignment tutorial |
| 4 | CTC forced alignment, `<star>` at both ends | torchaudio `MMS_FA` + `forced_align` | One model for 1,100+ languages; `<star>` absorbs laughter / other speakers instead of forcing them onto our words | Pratap et al. 2024, JMLR 25(97), arXiv 2305.13516; Hwang et al. 2023, TorchAudio 2.1, ASRU, arXiv 2310.17864 |
| 5 | New span = first word start − 0.1 s … last word end + 0.1 s; cut at original rate/channels | soundfile | Boundaries come straight from the word alignment; padding avoids clipping onsets/releases | — |
| 6 | QC: word scores; Whisper normalized CER of old vs new clip | Whisper-large-v3, metric of transcript_methods.pdf | Alignment always returns *something*; QC shows whether the cut helped and flags lines that need listening | Radford et al. 2023, ICML; confidence-based flagging as in Kürzinger et al. 2020 (CTC-segmentation, SPECOM) |

Flags (provisional thresholds): `unmatched_words` (≥ 3 words with score < 0.05: text with no audio, which
the aligner crams into the clip edge, so the line is cut off; a single such word is common on clean clips),
`cer_old_high` / `cer_new_high` (CER > 0.3, the report's threshold),
`weak_xcorr` (peak < 0.8).

## Modes
- `--mode clip` (baseline statistics): aligns O inside each existing clip and records how well it fits
  (unaligned audio at the edges, unmatched words, Whisper CER). It writes no new audio.
- `--mode conversation`: full steps 1–6; recovers words cut off by the old boundaries.

## Outputs
`<folder>/01_Natural_<SPK>_aligned.wav`, `<folder>/pre.TextGrid`, `<folder>/post.TextGrid` (word tier, `*` =
audio absorbed by `<star>`), `alignments.csv` (old/new span, scores, CER, flags), `words.jsonl`, `config.yaml`.

## Figures (5 examples)
Pre vs post side by side, Praat-style: waveform, spectrogram + pitch, word tier. Rendered with
Parselmouth (Praat's algorithms; Jadoul, Thompson & de Boer 2018, J. Phonetics 71:1–15) and matplotlib.
Examples fixed in advance: the first five large-mismatch boundary errors in the report
(03_train_1440_3, 06_train_685_4, 12_val_205_5, 14_train_1753_4, 56_train_1421_3).

## Known limits
Overlapping speech cannot be fixed by cutting. Clips whose transcript is wrong ("audio says something
else", garbled) need re-transcription; QC flags them rather than hiding them.
