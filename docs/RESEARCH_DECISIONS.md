# Research decisions

Updated 15 November 2026. This records completed work and the intended direction;
it is not evidence that a new experiment has run during code integration.

## What is implemented and what we learned

- The Hindi–English generation/evaluation work compares **model–voice
  configurations**. Amit and Kavya are two Indic-Speak voices, not independent
  architectures. Automatic content/acoustic scores remain proxies; provisional
  labels and transcript uncertainties stay explicit.
- The team's natural-versus-synthetic CAA pilots provide model adapters,
  extraction, norm-preserving hooks, grouped statistics and controlled generation.
  Their reports identify speaker/channel/NLL and text–audio mismatch confounds.
  Classifier movement did not establish naturalness improvement. See
  `preliminary_caa.tex` and `preliminary_caa_pt.tex`; preserve held-out versus
  in-sample and pre-registered versus revised selection results.
- Style-conditioning pilots generated default/ANGER controls and validated
  prompt placement, code phases, fresh-cache masks, exact alpha-zero behavior
  and hook removal. These checks establish engineering readiness, not emotion.
- The English multi-emotion study measured 100 matched texts under default,
  `ANGER`, `happy`, `sad` and an equal-length `news` control: 500 contexts over
  28 layers. Four authored calibration sentences constructed directions; 94
  texts from distinct EWT source documents tested generalization. Two original
  held-out sentences supported the small causal generation check. Raw data and
  activation arrays stay in private experiment storage.
- In the 94-text validation set, largest mean displacement occurred at
  zero-based layer **26** for the three emotion labels. Strongest directional
  consistency occurred at **5**, whereas the four-text calibration maximum had
  been **3**. Magnitude, consistency and causal utility are different criteria.
- The fixed 2% one-anchor intervention compared layers 3 and 26, positive and
  negative directions and a norm-matched random control on two held-out texts.
  At layer 26, final-state alignment with the ANGER-prompt displacement averaged
  +0.816 / −0.766 for positive/negative directions, versus +0.105 for random.
  The positive layer-3 alignment did not persist. All 12 interventions changed
  audio, including random controls; zero strength was exact.
- Geometric movement **did not reproduce the prompted pitch pattern**.
  Eleven of twelve nonzero outputs had zero normalized Whisper WER; one had
  extra ASR-transcribed words. A sadness prompt also produced a repetition flag
  in the control cohort. Neither content proxies nor geometric alignment proves
  successful vocal emotion. Perceived expression was not validated.

## Boundaries we keep

1. Identify the model revision, voice, text, token position, hook site and phase.
   A layer index alone is not a portable control prescription.
2. Match semantic anchors across prompts and keep instruction length/context
   effects explicit. Equal-length style-word controls do not establish semantic
   emotion disentanglement.
3. Separate prompt-context directions from speech-continuation directions. Do
   not inject a vector at unrelated audio-token positions merely because its
   width matches.
4. Fit on calibration data only. Preserve held-out IDs, failed cases, negative
   controls and source-document grouping. Ninety-four validation documents do
   not turn a two-text causal pilot into a large speech-control study.
5. Do not rank perceived naturalness or emotion from WER, F0, classifier logits,
   PCA separation or activation distance alone. Natural-reference analyses also
   require appropriate source mapping and explicit domain/speaker limitations.

## Next planned direction

Trace **style conditioning → hidden states → audio-token predictions → speech**.
Reuse saved default/style continuations under the **same empty-style prompt**,
with complete seven-code phases and preserved histories. Compare those
continuation-associated directions with prompt-anchor directions. Collect at
the position/site where an intervention will act; inspect next-token effects
and downstream propagation with explicit zero and random controls.

Replicate a frozen small protocol across more held-out texts and seeds before
claiming reliable control. Keep this separate from future Hindi–English and
cross-voice generalization. Reuse shared modules rather than introducing a new
steering framework or repeating completed baseline generation.

Foundational reference: [Rimsky et al., CAA, ACL 2024](https://aclanthology.org/2024.acl-long.828/).
The prompt-anchor protocol is a documented adaptation, not the paper's
all-generated-position intervention. EWT provenance:
[Universal Dependencies English EWT](https://github.com/UniversalDependencies/UD_English-EWT),
r2.18 revision `b7711cce01cdd4f5fcc0a8199b8a50d951b16c0c`.
