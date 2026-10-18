# Activation Steering for Naturalistic Code-Switched Speech Generation

Building activation steering vectors for naturalistic Hindi–English code-switched speech generation using the MaSaC corpus.

> **Status:** preliminary study design. The TTS model and its LLM backbone are not yet selected (shown as `<model>` / `<backbone>` below). See [`docs/study_design.md`](docs/study_design.md) for the full notes.

## Overview

**Task:** generate naturalistic speech from an LLM-generated code-switched transcript.

**Motivation:** Contrastive Activation Addition (CAA) is cheap to train and can be swapped in and out at inference time. That makes it a fit for producing naturalistic code-switched speech with a small model (<10B parameters). Recent work such as CoCoEmo (Wang et al., 2026) shows CAA steering working on transformer-based TTS models. `<model>` is a candidate because it runs inference efficiently and supports both English and Hindi.

## Research questions

- **RQ1:** Can CAA, trained on contrasting empathetic and neutral examples from MaSaC, steer `<model>` (an LLM-based speech generation model) to produce naturalistic code-switched speech on unseen / out-of-distribution examples?
- **RQ2:** Can CAA encode per-character "personality" vectors from the MaSaC characters, so that `<model>` generates naturalistic, character-like code-switched speech for held-out MaSaC transcripts?

**Hypothesis:** both are achievable with careful choice of steered layers and train/test split.

**Novel extension:** applying Bi-directional Preference Optimization (BiPO; Cao et al., 2024) to TTS activation steering, which has not been tried before.

## Data

| Role | Source |
|---|---|
| Positive examples | **MaSaC** (Bedi et al., 2021): Hindi–English code-mixed conversational corpus annotated for sarcasm and humor |
| Neutral examples | Synthesized from MaSaC's human transcripts with an open-source TTS model (base OrpheusTTS or Kokoro-82M) |

Audio, model weights, and experiment outputs are not tracked in git; they live on the compute machine (see [Compute](#compute)).

## Evaluation

- **Acoustic naturalness:** F0, speaking rate, pause placement, syllable duration near language switches, compared with reference speech
- **Emotion similarity:** cosine similarity of emotion2vec-base embeddings between generated and ground-truth audio
- **Prosody similarity:** AutoPCP utterance-level similarity; similarity to a reference MaSaC speaker
- **Naturalness:** judged by Hindi–English bilingual listeners (MOS and A/B preference), at both turn and conversation level. Automatic MOS predictors (UTMOS, SQUIM) are not used because they are trained on English and are unreliable for Hindi–English speech.
- **Linguistic fidelity:** Hindi/English pronunciation accuracy, word preservation, WER
- **Human preference:** Hindi–English bilingual listeners judge which sample sounds more like natural conversational code-switching
- **Understanding:** sanity checks such as perplexity or English-translation quality

## Planned experiments

1. **Baseline:** benchmark candidate base models (speaker similarity, naturalness, intelligibility, WER) and pick the base `<model>`.
2. **Preliminary:** train a CAA steering vector from positive vs. neutral speech examples.

## Repository layout

```
docs/       study design notes
src/        library code (data loading, steering, TTS wrappers, metrics)
scripts/    entry points (data prep, training, evaluation)
```

## Compute

Developed and run on the Columbia CS speech-lab GPU machine `kedi` (2× L40 48GB, 2× RTX A5500 24GB). Data and outputs are kept outside the repo on kedi's local disk:

```
/local/speech/users/sp4050/
  data/       MaSaC audio/transcripts, synthesized neutral examples
  outputs/    steering vectors, generated audio, logs, results
```

## References

- Bedi et al. (2021). *Multi-modal Sarcasm Detection and Humor Classification in Code-mixed Conversations.* [arXiv:2105.09984](https://arxiv.org/abs/2105.09984)
- Saeki et al. (2022). *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022.* [arXiv:2204.02152](https://arxiv.org/abs/2204.02152)
- Yang et al. (2024). *InstructTTS: Modelling Expressive TTS in Discrete Latent Space With Natural Language Style Prompt.* [PDF](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/InstructTTS_Modelling_Expressive_TTS_in_Discrete_Latent_Space_With_Natural_Language_Style_Prompt.pdf)
- Cao et al. (2024). *Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization.* [arXiv:2406.00045](https://arxiv.org/abs/2406.00045)
- Gao et al. (2024). *Emo-DPO: Controllable Emotional Speech Synthesis through Direct Preference Optimization.* [arXiv:2409.10157](https://arxiv.org/abs/2409.10157)
- Wang et al. (2026). *CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering.* [PMLR v306](https://proceedings.mlr.press/v306/wang26he.html)
- Minixhofer et al. (2024). *TTSDS – Text-to-Speech Distribution Score.* [arXiv:2407.12707](https://arxiv.org/abs/2407.12707)
