# Frozen M1 CPU analysis

`configs/research/m1/protocol.json` preserves the original design. The adjacent
cohort index contains 120 text-free rows with the frozen development/confirmation
split. M1 has 119 usable texts, 159 histories and 795 replay contexts. Keep q024's
unusable reference and all requested denominators when comparing results.

For a new replay using the original private inputs, see `prepare_m1.py`,
`m1_sequence.py --help`, and `run_m1.py --help`. The sequencer accepts the prior
November 12 token input directory via `--existing` and a GPU index via `--gpu`.
Supply `--prior-traces` to preserve the c03/c04 comparison against the original
November 15 traces. The original manifest, fitted basis and execution-source
receipts stay with the archived run. The text-free index is a reference for
identity checks, not a raw replay input.

Run from the shared worktree with the existing Python environment:

```powershell
python scripts/analyze_m1.py --root <export-root> --output <new-directory> --allow-missing-tensors
```

The input root contains `inputs/`, `study/`, and `fit/` copied from the immutable
M1 run. The output directory must not exist. No GPU inference, model fitting,
audio synthesis, or steering occurs. Missing optional large NPZ tensors may stay
on Kedi; present artifacts and frozen basis/shortlist/protocol/cohort hashes are
checked. A final run rejects unattempted subjects and incomplete requested second
histories. `--allow-partial` is explicitly an interim engineering analysis; it
also permits a small bootstrap override for synthetic tests.

## Inferential units and frozen hypotheses

The 90 confirmation utterances, not positions, phases, labels, or layers, are the
independent observational units. The history outcome has at most 30 confirmation
utterances. The ten predefined outcomes are reused from `fit_m1.global_outcomes`.
Each contrasts signed utterance aggregates against zero using an approximate
two-sided one-sample t test; Holm correction always retains all ten slots.
Undefined outcomes are missing, with explicit coverage. Constant/numerically
degenerate samples receive no variance-based test, and occupy a conservative
p=1 slot. Significant negative means fail the proposed positive pattern.

Percentile 95% intervals use 2,000 utterance bootstrap draws, stratified by the
three operational text categories, with the frozen analysis seed. All cells in
a map use the same resampled rows. The development basis remains fixed: these
intervals omit uncertainty from fitting a different development cohort.

For each historical/shared/specificity map family, approximate simultaneous
bands use the bootstrap maximum absolute centered mean deviation divided by the
original cell standard error. These are approximate fixed-SE standardized bands,
not a fully studentized bootstrap-t or a strong mechanistic localization test.
Pointwise figure ribbons and map-wide bands are labeled separately. Very small
or undefined standard errors do not generate a discovery threshold.

## Alignment and interpretation

Primary generation summaries exclude j=0 and weight seven predicted SNAC phases
equally. Historical reference directions exist only for post-block residuals.
The cosine of a phase-averaged contrast differs from the supplementary mean of
pointwise cosines; both retain distinct names. Chronological frame tertiles are
not word or phoneme alignments. Final-LM token distributions are not per-layer
probabilities. Residual decomposition retains the actual dtype-rounding remainder;
signed projections may exceed one or cancel and are not causal fractions.

The development-only M2 layer/site and phase choices are checked against the
original first-flat-index exact-tie rule, never reselected. Their confirmation
intervals are descriptive outside the fixed ten-test family. Identical inspected
secondary prefixes remain in coverage but are excluded from independent-history
inference. No reseeding is used to force diversity.

Category screens are lexical, not validated emotional states or discourse labels.
Genre, length, document selection, and category can be confounded. Equal-format
News contrasts control the style block but do not isolate emotional semantics.
Default lacks that block and is secondary. Effects concern inspected code prefixes
from English Amit at one frozen checkpoint. M1 supplies observational hypotheses,
not emotional delivery, shared emotion circuits, causal control, or perceptual
quality evidence.

Methods: [SciPy one-sample t test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)
and [Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
