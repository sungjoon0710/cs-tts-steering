# M3 research readiness

Start from M2's confirmed native layer-27 MLP effect and the divergent own-target and selectivity results for fitted directions. The existing code supports matched, single-position interventions under a fixed audio history. It provides exact source provenance, prediction phase and offset checks, realized-norm random controls, earlier-position invariance and utterance-level statistics.

| Question | Reusable infrastructure | Prerequisite |
| --- | --- | --- |
| Which computation makes the late effect possible? | `src/causal_patch.py` and the attention, MLP and residual trace conventions | Add carefully matched L27 MLP input/output interventions and a final normalization/readout comparison. Preserve BF16 ordering and control for proximity to the head. |
| Does a particular head or source position mediate the effect? | `src/generation_trace.py` and exact hook/position controls | Expose per-head tensors before output projection and define source positions, masks and causal receptive fields. Existing attention captures are post-projection. |
| Does a downstream path carry the signal? | Native/reverse controls, fixed histories and downstream observations | Specify sender and receiver sites, isolation conditions and restoration controls. Verify patch isolation before testing a bounded pathway. |
| Can control work across contexts without donor activations? | Development/confirmation separation, fitted basis utilities and M2 endpoints | Calibrate sign, norm and timing on development contexts. Freeze the method and require positive own-target recovery and selectivity on untouched texts and histories. |
| What changes in speech? | Indic-Speak wrapper, token callback, SNAC/Vocos tracing and audio evaluation utilities | Generate valid code sequences under a separately approved design. Verify stopping, content and speaker controls before acoustic and blinded perceptual evaluation. |

## Fixed-history development runs

`scripts/m3/run_mechanism.py` compares late MLP input/output swaps, the residual bypass, and the final readout using fresh uncached replays. It checks the two native BF16 residual additions and exact agreement for the unmodified and joint-donor controls. `scripts/m3/run_source_paths.py` replaces the donor's K/V projection outputs at the changed style slot, before RoPE, across layer bands. It checks exact identity and earlier-position invariance and can clamp the News MLP output at the scored rows of layer 27.

`configs/research/m3/design.json` records the development split and model, codec, and M2 source revision. The runners take the original M2 source release, private M2 run root, model configuration, and checkpoint paths explicitly. Use an unused output directory. The saved six-text development comparison recovered about 89% of the phase-conditional donor difference with an unrestricted layer 0–8 K/V swap. It addresses next-code probabilities under a forced history. The audible effect and downstream route require separate evidence.

```powershell
python scripts/m3/run_mechanism.py --help
python scripts/m3/run_source_paths.py --help
python scripts/m3/summarize_mechanism.py SAVED_ASSESSMENT NEW_SUMMARY
```

## Query-scoped attention

`scripts/m3/pathway_experiment.py` runs native scaled dot-product attention with original K/V and with donor K/V at the matched style slot after RoPE. It uses the donor attention output only for the chosen query rows before output projection. The experiment checks identity, exact agreement with an unrestricted native projection swap, earlier-position invariance, and realized random-control norms. The frozen `configs/research/m3/plan.json` records the six development texts, layers 0–8 and scored rows. Pass it with `--plan` and use `--limit 1` to run the smoke test before the six-text run. Set `--delimiter-focus` for the exploratory split.

The two query positions immediately after the style word recovered 84% of the phase-conditional donor difference on the same six texts. Recovery fell to 42% with News MLP outputs held fixed at the scored layer-27 rows. This localizes admission of style-slot information under one prompt format and leaves its downstream route unresolved. Observational comparisons involving Default align the same prediction row across prompts, since Default omits four style-block tokens.

```powershell
python scripts/m3/pathway_experiment.py --help
python scripts/m3/summarize_pathways.py SAVED_PRIMARY NEW_PRIMARY_SUMMARY
python scripts/m3/summarize_delimiters.py SAVED_PRIMARY SAVED_FOLLOWUP NEW_DELIMITER_SUMMARY
```
