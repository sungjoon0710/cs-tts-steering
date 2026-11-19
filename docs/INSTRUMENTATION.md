# Indic-Speak research instrumentation

The pinned model, codec and vendor inference digest are in `configs/representation_pilot.json`. Point research entry points at the existing snapshots. `scripts/inspect_representations.py --help` lists prompt extraction, continuation phase pooling and CPU geometry analysis.

| Instrument | Entry point | Scope |
| --- | --- | --- |
| Team CAA API | `src.steering.SpeechLM`, `extract`, `Steerer`, `generate_batch` | Shared model adapters, extraction and additive intervention |
| Representation probes | `src.representation_probe` | Final speech-start anchor, fixed prefix, seven-phase pooling and zero-strength checks |
| Prompt-only intervention | `src.position_steering.AnchorIntervention` | One final prompt position, fresh prefill and norm-preserving team hook |
| Geometry | `src.representation_geometry` | Paired contrasts, directional consistency and calibration-fit PCA |
| Reference generator | `src.tts.IndicSpeak` | Optional style and token callback capture the historical prompt and generated token sequence |

The anchor and a speech-code position answer different research questions. Use the same semantic anchor for prompt geometry. For comparisons during generation, record the forced history and the exact prediction position. Keep fitting and selection on development or calibration data. Report missing generations and preserve original experiment manifests and releases.

Run CPU tests from the repository with the evaluation dependencies and an installed PyTorch build. No checkpoints are needed for this check:

```powershell
python -m pytest -q tests/test_caa.py tests/test_position_steering.py tests/test_representation_probe.py tests/test_representation_geometry.py tests/test_prompt_representations.py tests/test_representation_cli.py tests/test_tts_snapshots.py
```

Keep raw recordings, checkpoints, source texts, execution archives and credential state in separately managed storage. Git attributes preserve LF for the frozen code and disable newline conversion for the archived protocol JSON files, which include original CRLF bytes.
