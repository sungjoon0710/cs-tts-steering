# M3 research readiness

Start from M2's confirmed native layer-27 MLP effect and the divergent own-target and selectivity results for fitted directions. The existing code supports matched, single-position interventions under a fixed audio history. It provides exact source provenance, prediction phase and offset checks, realized-norm random controls, earlier-position invariance and utterance-level statistics.

| Question | Reusable infrastructure | Prerequisite |
| --- | --- | --- |
| Which computation makes the late effect possible? | `src/causal_patch.py` and the attention, MLP and residual trace conventions | Add carefully matched L27 MLP input/output interventions and a final normalization/readout comparison. Preserve BF16 ordering and control for proximity to the head. |
| Does a particular head or source position mediate the effect? | `src/generation_trace.py` and exact hook/position controls | Expose per-head tensors before output projection and define source positions, masks and causal receptive fields. Existing attention captures are post-projection. |
| Does a downstream path carry the signal? | Native/reverse controls, fixed histories and downstream observations | Specify sender and receiver sites, isolation conditions and restoration controls. Verify patch isolation before testing a bounded pathway. |
| Can control work across contexts without donor activations? | Development/confirmation separation, fitted basis utilities and M2 endpoints | Calibrate sign, norm and timing on development contexts. Freeze the method and require positive own-target recovery and selectivity on untouched texts and histories. |
| What changes in speech? | Indic-Speak wrapper, token callback, SNAC/Vocos tracing and audio evaluation utilities | Generate valid code sequences under a separately approved design. Verify stopping, content and speaker controls before acoustic and blinded perceptual evaluation. |

The first proposed experiment compares MLP input and output interventions and the final head readout. It can distinguish a contribution from the MLP transformation from direct access to output-relevant coordinates. Individual-head, path and generation experiments require their own engineering controls and frozen hypotheses.
