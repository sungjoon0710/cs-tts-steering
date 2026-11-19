# November 12 prompt-anchor study

The study compared 100 English texts with Amit under default, ANGER, happy, sad and an equal-length News control. Four authored calibration texts fit the directions. Two authored held-out texts were used in the anchor intervention pilot. A separate set of 94 documents tested directional consistency and displacement. The largest mean displacement for the three emotion labels was at zero-based layer 26. Validation directional consistency peaked at layer 5. The two-text pilot compared layers 3 and 26 at the final speech-start prompt state and used positive, negative, random and zero controls. A change in tokens and downstream states did not establish audible anger or reproduce the prompted pitch pattern.

The geometry workflow uses `src.representation_probe` and `src.representation_geometry`. `scripts/inspect_representations.py prompts` can collect prompt states for a supplied manifest. `nov12_geometry.py` reads the original `[100,5,28,3072]` saved array, with conditions in the order default, anger, happy, sad and control_news, and retains the original 1,000-draw genre-stratified document bootstrap. Pass `--reference control_news` for the equal-format contrasts. The historical EWT split uses release r2.18 at `b7711cce01cdd4f5fcc0a8199b8a50d951b16c0c`. The original cohort, snapshots and results stay in the November 12 private run.

```powershell
python scripts/nov12_geometry.py summarize --states SAVED_STATES.npy --cohort FROZEN_COHORT.jsonl --output NEW_SUMMARY
python scripts/nov12_geometry.py plot --tables SAVED_SUMMARY_DIRECTORY --output NEW_PLOTS
```

`nov12_anchor_pilot.py` adapts the historical `controlled_extension.py` runner around the tracked `AnchorIntervention` adapter and the team's norm-preserving CAA hook. The default design is the saved November 12 design. Use the pinned offline model and codec cache and the private screen-run root with its saved config, scenario manifest, activation files, audio and token receipts. Set a separate output root. Run `expand`, then `smoke`, and evaluate the smoke output against the ASR guardrail in `configs/research/nov12/anchor_intervention.json` before `rest`. These commands would run model inference and generate audio. The checked-in code preserves the algorithm, while original execution hashes and output evidence remain with the archived release.

```powershell
python scripts/nov12_anchor_pilot.py --help
```
