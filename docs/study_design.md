# Study Design (preliminary)

Building activation steering vectors for naturalistic code-switched speech generation using MaSaC.

## 1. Research question and task

**Task:** Generation of naturalistic speech from an LLM-generated code-switched transcript.

**Motivation:** CAA is light to train and modular at inference time. We want to generate naturalistic code-switched speech but stay within narrow (<10B params) compute constraints. CAA steering of transformer-based TTS models has shown promise in recent studies, as seen in the CoCoEmo (Wang et al., 2026) paper. `<model>` is suitable because it provides efficient inference while providing multilingual support in both English and Hindi.

**Research questions:**
- **RQ1:** Can we do CAA on empathetic / neutral examples of the MaSaC corpus to get `<model>`, an LLM-based speech generation model, to generate naturalistic code-switched speech in unseen / out-of-distribution examples?
- **RQ2:** Can we encode personality vectors into CAA based on the different characters of the MaSaC corpus to get `<model>` to generate naturalistic, character-similar code-switched speech in left-out transcripts from the MaSaC dataset?

**Hypothesis:** RQ1 and RQ2 will be possible with thoughtful tuning of which layers to control and the train / test split.

## 2. Datasets

Positive examples come from the MaSaC corpus, a specialized dataset designed to study sarcasm and humor within a code-mixed and code-switched (Hindi–English) conversational environment (Bedi et al., 2021).

Neutral examples are generated from MaSaC's human-transcribed transcripts using an open-source TTS model (either base OrpheusTTS or a simple open-source TTS model such as Kokoro-82M).

## 3. Models

We utilize `<model>`, an open-source text-to-speech (TTS) model built around a post-trained `<backbone>` LLM.

## 4. Novel extensions

Applying BiPO (Cao et al., 2024) to TTS activation steering has not been tried before.

## 5. Baselines

_TBD_

## 6. Evaluation and error analysis

- **Acoustic naturalness:** Pitch (F0), speaking rate, pause placement, and syllable duration near language switches, compared with reference speech.
- **Emotion similarity (SIM):** use the emotion2vec-base model to extract emotion embeddings from ground-truth and generated audio, compute cosine similarity, and average across the test set for the EMO SIM score.
- **Prosody similarity (SIM):**
  - Could use AutoPCP as an utterance-level estimator to quantify prosody similarity between generated and ground-truth speech.
  - Could compute similarity to a given reference speaker, e.g. using a MaSaC speaker as the reference.
- **Naturalness:**
  - Turn-based and conversation-based evaluations need to complement each other.
  - UTMOS naturalness metric (0–5, higher is better) (Saeki et al., 2022).
- **Linguistic fidelity:** Hindi and English pronunciation accuracy, word preservation, and language-specific intelligibility (WER, etc.).
- **Human preference:** ask Hindi–English bilingual listeners which sample sounds more like natural conversational code-switching.
- **Understanding metrics:** evaluating a system's understanding of input speech as a sanity check.
  - Perplexity is one option.
  - Evaluation of translations to English is another.

## 7. Experiments

- **Baseline experiment:** set up baselines using the base `<model>`; establish key metrics such as speaker similarity, naturalness, intelligibility, WER, etc., to select the correct base model to work from.
- **Preliminary experiment:** train an activation steering vector using positive vs. neutral speech examples.

## 8. Feasibility, milestones, and responsibilities

Training an activation steering vector is considered feasible based on a preliminary literature review.

## 9. References

See the [README](../README.md#references).
