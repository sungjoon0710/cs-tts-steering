# Model input/output spec (research notes, 2026-10-18)

What each baseline model expects as input for Hindi–English code-switched text, what it returns, and what is still unverified. Each claim is labeled by source: **[repo]** = model's own files, **[community]** = third-party code/discussion, **[ours]** = our smoke test.

## Summary

| | Indic-Speak (`bodhan-ai/indic-speak`) | Orpheus 3B Hindi (`canopylabs/3b-hi-ft-research_release`) |
|---|---|---|
| Hindi script | Devanagari **[repo: all examples]** | Devanagari (presumed). Romanized Hinglish reported "pretty bad" **[community: HF discussion on the hi-pretrain model]** |
| English inside Hindi | Explicitly supported: "English terms, acronyms, and numerals sitting inside an Indic sentence are read in the surrounding voice and accent" **[repo]**. Script for English words **not specified**. | Not documented. One long Latin-heavy line came out ~2.7 s vs ~10 s from Indic-Speak (possible truncation) **[ours]** |
| Text normalization | None in tokenizer (`normalizer: null`); Llama-3 byte-level BPE, so Latin and Devanagari both tokenize **[repo]** | Llama-3 tokenizer (same) |
| Punctuation | Drives pauses/intonation; "place commas wherever a speaker would breathe" **[repo: voices.md]** | Not documented |
| Voices | Hindi: `Kavya` (F, ~215 Hz), `Amit` (M, ~152 Hz). Same artists also recorded English, which "is what makes code-mixed text work" **[repo]** Reportedly one voice, `ऋतिका` **[community: orpheus-cpp, FlashTTS, phoonnx list only this name, likely all copied from Canopy's release page, so not independent]**. "Female" comes only from one quantized-copy model card, so **gender is unverified**. Voice names aren't stored in the model files. To verify: listen to the smoke-test clips and try more names. |
| Prompt text | `<|speaker>Kavya<speaker|>\n[<|style>…<style|>\n]{text}` (built by `inference.build_prompt`) | `ऋतिका: {text}` wrapped in `128259 … 128009 128260` |
| Style / emotion | `style` block: `happy, sad, anger, fear, surprise, disgust` + genre tags; "preview-quality, not reliable" **[repo]** | **No emotive tags for Hindi** (FlashTTS lists `tags: []`) **[community]** |
| Non-verbals | 16 `<|nv_*|>` tokens exist in tokenizer, but README says "No laughter, sighs, or breaths" → don't rely on them **[repo]** | None for Hindi |
| Multi-turn | Training format exists for conversations: `<|speaker>A<speaker|>\nturn\n\n<|speaker>B<speaker|>\nturn` **[repo: token_contract.md §5]**. Whether `A`/`B` can be voice names is untested | No |
| Decoding | t 0.6, top_p 0.9, top_k 50, rep. penalty 1.2 (vendor benchmark config) **[repo]** | t 0.6, top_p 0.9, rep. penalty ≥ 1.1 (required) **[Orpheus README]** |
| Output | 24 kHz mono float32; SNAC codes → **Vocos** decoder; ≈ 82 tokens/s; duplicate frames removed so tokens ∝ duration only roughly **[repo]** | 24 kHz mono; SNAC codes → stock SNAC decoder |
| Quality claims | WER 26.5 / CER 17.9, MOS 4.46 (vendor) **[repo]** | Canopy reportedly rates its own Hindi fine-tune as **"poor"** **[community: phoonnx docs; not confirmed on Canopy's page]** |

## Training data hints

- **Indic-Speak** prompt templates are named after their training sets **[repo: token_contract.md]**:
  - `rasa`: AI4Bharat Rasa, studio-recorded single speakers per Indic language, Wikipedia/IndicTTS readings, commands, conversations, 6 emotions. Native-script text presumed (dataset is gated, so not checked).
  - `globe`: English accented speech.
  - `bhili`: Bhili voices.
  - `gemini_vc_conversational`: synthetic multi-speaker conversations.

  So Hindi is likely learned from native-script Devanagari text, and English from Latin script. How code-mixed sentences were written in training is unknown.
- **Orpheus Hindi**: undocumented. Canopy's release page (voices, tags, training guide) is JS-only and couldn't be fetched. Pretrain → fine-tune on a single voice.

## Discrepancies / gotchas

- **Indic-Speak prompt end:** `token_contract.md` says the prompt ends at `<|start_of_ai|>` and the model emits `<|start_of_speech|>` itself, but `inference.py` and the README include `<|start_of_speech|>` in the prompt. We follow `inference.py` (vendor runtime) and note this.
- **Orpheus voice:** the smoke test ran all 5 utterances with *no* voice prefix and tried `ऋतिका` on utterance 16 only. Baseline runs of `orpheus-hi-ft` use `ऋतिका:`; `orpheus-hi-pt` (pretrained) has no named voices and gets a text-only prompt.
- **Gender coverage (unverified):** if Orpheus really has a single (female) voice, male MaSaC characters (Indravadan, Sahil, Rosesh, …) can't be gender-matched. Indic-Speak can (Kavya/Amit). Confirm by listening before relying on this.
- **MaSaC text formats:** MSH-COMICS clean lines = Devanagari Hindi + Latin English, which matches Indic-Speak's documented use case. EDiReF = fully romanized, which neither model is documented for, and Orpheus reportedly handles it badly.

## Proposed input conditions (to verify with a small test before the pilot)

| Condition | Example (invented sentence, not from MaSaC; the test itself uses MaSaC line 126) | Rationale |
|---|---|---|
| **A. As-is (primary)** | `seriously? मैं meeting में late नहीं था, traffic बहुत था.` | MaSaC's own clean format; Indic-Speak's documented use case |
| B. Full Devanagari | `सीरियसली? मैं मीटिंग में लेट नहीं था, ट्रैफ़िक बहुत था.` | Isolates script handling from code-switching; likely closest to native-script training text |
| C. Full roman | `seriously? main meeting mein late nahin tha, traffic bahut tha.` | EDiReF-style; expected to fail on Orpheus, included only to confirm |

Test design: 5 clean lines × 3 conditions × 2 models (Indic-Speak `Kavya` + `Amit`, Orpheus `ऋतिका`) ≈ 45 clips, < 10 min on one GPU. Judge by ear and by Whisper transcript (does each English word survive?). Conditions B and C are hand-written for these 5 lines, not machine-transliterated.

## Open questions

1. Script for English words in Indic-Speak's training text (Latin vs Devanagari). Checkable from Rasa samples (gated; needs accepting its terms) or by the test above.
2. Whether Orpheus Hindi truncates or skips Latin-script spans. The test above with `ऋतिका:` answers it.
3. Whether Indic-Speak's multi-turn format accepts real voice names, for conversation-level generation (relevant to "turn- vs conversation-based" evaluation).

## Sources
- Indic-Speak repo files: `README.md`, `voices.md`, `token_contract.md`, `inference.py`, `tokenizer.json` (local snapshot on kedi)
- Orpheus Hindi pretrain discussion: https://huggingface.co/canopylabs/3b-hi-pretrain-research_release/discussions/1
- Orpheus voice lists: freddyaboulton/orpheus-cpp (`src/orpheus_cpp/__main__.py`), HuiResearch/FlashTTS (`flashtts/engine/orpheus_engine.py`), TigreGotico/phoonnx (`docs/orpheus.md`)
- Orpheus README: https://github.com/canopyai/Orpheus-TTS
- Rasa dataset card: https://huggingface.co/datasets/ai4bharat/Rasa
