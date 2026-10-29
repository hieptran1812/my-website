---
title: "Building a Modern ASR System End to End: Data Engine, Fine-Tuning, Evaluation and Deployment"
date: "2026-10-29"
publishDate: "2026-10-29"
description: "A step-by-step playbook for shipping speech recognition in 2026: pick a base model, build the test set first, run a pseudo-labeling data engine, fine-tune without forgetting, handle new words and proper nouns, evaluate per slice, and close the loop from production."
tags: ["asr", "speech-recognition", "speech-to-text", "data-engine", "pseudo-labeling", "contextual-biasing", "wer", "evaluation", "fine-tuning", "whisper"]
category: "machine-learning"
subcategory: "Speech-to-Text"
author: "Hiep Tran"
featured: true
readTime: 11
---

Most teams that "build ASR" in 2026 never train a model from scratch, and the ones that fail rarely fail on architecture. They fail because their test set came from the same speakers as their training set, because a text normalizer turned a perfect transcript into 160% WER, or because the system wrote "Việt Com Banh" for the bank whose name the whole call was about. The model is the easy part. Everything around it decides whether the product works.

![The ASR loop: raw audio flows through a data engine into fine-tuning, an eval gate and deployment, and hard production audio flows back](/imgs/blogs/modern-asr-system-end-to-end-1.webp)

The diagram above is the mental model: a loop, not a pipeline. Audio enters a data engine, becomes training pairs, passes through fine-tuning and an evaluation gate, ships, and the clips production found hard flow back in. The running example throughout is a **Vietnamese call-center recognizer**: 8 kHz phone audio, regional accents, English product terms mixed into Vietnamese sentences, and a catalog of bank and product names that must be spelled exactly. Every number below is an illustrative scenario, not a benchmark.

## Step 0: define the target before touching data

Write down four things first, because each one changes later choices:

- **Domain and audio channel.** Phone audio at 8 kHz loses everything above 4 kHz; a model trained on 16 kHz podcasts will underperform on it before any other problem appears.
- **Streaming or batch.** Live agent assist needs partial words within a few hundred milliseconds. Post-call analytics can wait minutes. This one answer picks the model family, as explained in [CTC vs RNN-T vs attention encoder-decoder](/blog/machine-learning/speech-to-text/ctc-vs-rnnt-vs-attention-asr-alignment).
- **The metric that decides shipping.** For a call center it is rarely raw WER. It is whether account names, amounts and product codes come out right.
- **Languages and code-switching.** "Em muốn upgrade lên gói premium" is one sentence in two languages, and the model must keep both.

## Step 1: pick a base model, not an architecture

| Base model family | Strength | Weakness | Pick it when |
| --- | --- | --- | --- |
| Whisper large-v3 / turbo | 99 languages, strong formatting, huge ecosystem | Offline, hallucinates on silence | Batch transcription, many languages |
| Parakeet / Canary (FastConformer) | Very fast, streaming variants, CTC/TDT heads | Fewer languages out of the box | Real-time, high throughput |
| LLM-based (Qwen3-ASR and similar) | Prompt-based context, strong multilingual | Heavy, slow, LLM-style hallucination | Rich context, offline, GPU budget available |

Our call center needs live agent assist, so the default is a streaming FastConformer transducer, fine-tuned for Vietnamese. If the tokenizer of the base model was trained only on English, that decision has a consequence covered in Step 5. The architectures are unpacked in [Whisper under the hood](/blog/machine-learning/speech-to-text/whisper-under-the-hood) and [Canary and Parakeet](/blog/machine-learning/speech-to-text/canary-parakeet-fastconformer-asr).

## Step 2: build the test set first

Before collecting a single training hour, have humans transcribe about 20 hours of real production calls for test and 10 for dev, under a written style guide: how numbers are written, how English terms are spelled, how to mark disfluencies and unintelligible speech.

Three rules make this set trustworthy:

1. **Speaker-disjoint.** No speaker in test may appear in training. Call centers have a few hundred agents who talk on thousands of calls; a random split leaks their voices and inflates every result.
2. **Stratified by slice.** Tag every clip with channel, accent, noise, code-switching and entity density, so you can measure each slice separately later.
3. **Frozen.** The test set changes only by versioned additions. If it moves every week, no two results are comparable.

## Step 3: the data engine

Human transcription at scale is slow and expensive, so modern pipelines manufacture labels. The pattern, used by Distil-Whisper and most production teams, is: run strong teacher models over raw audio, keep only what the teachers agree on, and normalize the text.

![Data funnel: 10,000 hours of raw audio shrink to 3,900 hours of trusted training data](/imgs/blogs/modern-asr-system-end-to-end-2.webp)

The funnel shows where the hours go:

1. **VAD and segmentation.** A voice activity detector (Silero, or the VAD built into NeMo and faster-whisper) cuts hold music, ringing and silence, and splits long calls into 5 to 20 second segments at pauses. In our scenario 10,000 raw hours become 6,200 hours of speech.
2. **Language ID and dedupe.** Drop clips in the wrong language and near-duplicate IVR prompts that would otherwise appear thousands of times. 5,100 hours remain.
3. **Teacher pseudo-labels with an agreement filter.** Transcribe every segment with two different teachers, for example Whisper large-v3 and a large Vietnamese CTC model. Keep a segment only if their normalized WER against each other is at most 10%. Disagreement usually means noise, overlap or an unusual word, exactly the cases where a pseudo-label is wrong. 3,900 hours survive.
4. **Text normalization.** Convert every label to one canonical form that matches the test style guide. Mixed conventions in training labels teach the model to be randomly inconsistent.

The segments filtered out are not waste. They are the hardest audio you own, and the best candidates for paid human labeling. More on the alignment side of this in [speech data for ASR](/blog/machine-learning/training-data/speech-data-for-asr) and on low-resource tricks in [weak supervision and low-resource speech](/blog/machine-learning/training-data/weak-supervision-and-low-resource-speech).

## Step 4: fine-tune without forgetting

The recipe is unglamorous and mostly about data:

- **Match the channel.** Downsample general 16 kHz data to 8 kHz and back, or run it through a telephony codec, so the model sees phone-quality audio during training.
- **Augment.** SpecAugment (time and frequency masks), speed perturbation at 0.9, 1.0 and 1.1, additive noise from real call-center background, and room impulse responses.
- **Low learning rate, short schedule.** The base model already knows acoustics. A learning rate around one tenth of pretraining, with warmup, is a typical start; tune it on dev.
- **Mix in general data.** This one matters most.

![In-domain WER falls in both runs, but general WER rises unless general data is mixed in](/imgs/blogs/modern-asr-system-end-to-end-3.webp)

Both runs improve in-domain WER from 24% to 11%. The run trained only on call audio forgets: general-domain WER climbs from 8% to 14%, and that shows up the first time a customer talks about anything off-script. Mixing roughly 30% general data holds it at 8.6%. Track a general-domain set alongside the in-domain one from the first step. The full training recipe, including tokenizer and schedule choices, is in [training ASR models](/blog/machine-learning/speech-to-text/training-asr-models).

## Step 5: new words, proper nouns and unknown tokens

This is where call-center ASR is won or lost, and it is three separate problems.

### Problem 1: the tokenizer cannot write the word at all

Modern subword tokenizers rarely produce a literal `<unk>` for English text, but it still happens in two situations. A SentencePiece model trained without `byte_fallback` maps any unseen character to `<unk>`, so an English-only tokenizer fed Vietnamese emits `<unk>` for every "ệ" and "ư". And a CTC or transducer head can only output tokens in its vocabulary, so no amount of fine-tuning will make it spell characters the tokenizer lacks.

The fixes, in order of preference: choose a base model whose tokenizer already covers the script (Whisper's byte-level BPE can encode any string). Otherwise retrain the tokenizer on in-domain text and re-initialize the output layer, accepting a longer fine-tune. Extending an existing vocabulary with new tokens works too, but the new output rows start untrained and need enough examples to learn. Always run a check before training: tokenize every label in the training set and assert that zero `<unk>` tokens appear.

### Problem 2: the word is spellable but rare

"Vietcombank" decomposes into ordinary subwords, so the model *can* write it. It just rarely does, because the language-model prior inside the decoder favors common words that sound similar. The tools, from cheapest to most expensive:

1. **Contextual biasing at decode time.** Give the decoder a list of phrases and boost their score during beam search.

![Biasing reranks the beam so the listed bank name moves from second to first](/imgs/blogs/modern-asr-system-end-to-end-4.webp)

The figure shows why this works. The correct spelling was already in the beam at rank 2; a +1.5 boost for a full match lifts it to first. Biasing can only rescue words the acoustic model nearly heard, which is why it pairs well with the methods below. Keep the list per call context (the customer's own products, the agent's queue), because a 50,000-entry global list starts boosting false matches.

```python
from faster_whisper import WhisperModel
from pyctcdecode import build_ctcdecoder

BIAS = ["Vietcombank", "Techcombank", "MoMo", "gói Premium"]

# Attention decoder: steer with a prompt and hotwords (offline path).
whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, _ = whisper.transcribe(
    "call_0193.wav", language="vi", vad_filter=True,
    initial_prompt="Khách hàng hỏi về chuyển khoản Vietcombank và gói Premium.",
    hotwords=" ".join(BIAS),
)

# CTC model: boost phrases inside beam search, with a domain KenLM.
def decode_ctc(logits, vocab, kenlm_path="callcenter_5gram.arpa"):
    decoder = build_ctcdecoder(vocab, kenlm_model_path=kenlm_path,
                               alpha=0.5, beta=1.0)
    return decoder.decode(logits, beam_width=64,
                          hotwords=BIAS, hotword_weight=8.0)
```

2. **A domain language model.** Shallow fusion with an n-gram LM trained on call transcripts, product catalogs and FAQ text raises the prior of domain words everywhere, not only listed ones.
3. **Synthetic training audio.** Generate sentences containing new product names with a multi-speaker TTS, add channel augmentation, and mix a small amount into fine-tuning. This is how teams prepare for a product launch before any customer has said the name.
4. **Post-correction against a catalog.** After decoding, fuzzy-match spans against the entity catalog on phonetic similarity ("việt com banh" to "Vietcombank"), or let an LLM correct the n-best list with the catalog in its prompt. Always constrain the LLM to the catalog; free rewriting introduces its own hallucinations.

### Problem 3: the word is new tomorrow

Products launch weekly. Treat the bias list and the domain LM as **configuration**, deployed without retraining, and add launched terms to the next fine-tune once real audio exists. Keep a small "entity regression set" of clips containing every important name, and run it on every model and every bias-list change.

## Step 6: evaluate per slice, after normalization

WER is substitutions, deletions and insertions divided by reference words. Before computing it, both reference and hypothesis must pass through the **same normalizer**.

![Without normalization, a perfect transcript scores 160% WER; after it, 0%](/imgs/blogs/modern-asr-system-end-to-end-5.webp)

The reference wrote "10%" and "2.500.000"; the model spoke the amount in words. Eight edits over five reference words give 160% WER for a transcript a human would call perfect. This is why public leaderboards normalize every system with the same text normalizer before scoring. For Vietnamese, also report CER, since word boundaries in the written language split syllables, not words.

Then break the score down:

![Aggregate WER of 9.8% hides code-switched and entity-heavy slices that fail badly](/imgs/blogs/modern-asr-system-end-to-end-6.webp)

An aggregate 9.8% looks shippable. The code-switched slice runs at 18.7% WER and misses 31% of entities, and the silence slice invents text on 3.2% of clips. Those are the rows customers notice.

```python
import jiwer
from collections import defaultdict

def normalize(text: str) -> str:
    # Project-specific: lowercase, strip punctuation, verbalize numbers, unify spellings.
    return vi_text_normalizer(text)

def evaluate(rows):  # rows: dicts with ref, hyp, slice, entities, is_silence
    by_slice, ent_hit, ent_tot, halluc, silent = defaultdict(lambda: ([], [])), 0, 0, 0, 0
    for r in rows:
        if r["is_silence"]:
            silent += 1
            halluc += bool(normalize(r["hyp"]).strip())   # any text on silence counts
            continue
        refs, hyps = by_slice[r["slice"]]
        refs.append(normalize(r["ref"])); hyps.append(normalize(r["hyp"]))
        for e in r["entities"]:
            ent_tot += 1
            ent_hit += normalize(e) in normalize(r["hyp"])
    report = {s: jiwer.wer(rf, hy) for s, (rf, hy) in by_slice.items()}
    report["entity_error_rate"] = 1 - ent_hit / max(ent_tot, 1)
    report["hallucination_rate"] = halluc / max(silent, 1)
    return report
```

Add two systems metrics: **RTFx** (seconds of audio processed per second of compute) for cost, and **latency at p50 and p95** for streaming. Turn all of it into a release gate: a new model ships only if no slice regresses beyond a set tolerance, entity error falls, and hallucination rate does not rise. Track a confidence score as well: it does not need to be perfect, only correlated enough to route clips back to the data engine.

## Step 7: deploy and close the loop

![Streaming latency splits into capture, compute, network and endpointing](/imgs/blogs/modern-asr-system-end-to-end-7.webp)

In the streaming path, partial text appears about 225 ms after speech, most of it the 160 ms chunk itself. The final transcript waits for the endpointer to see 400 ms of silence, which is the knob most teams end up tuning: shorter feels snappier but cuts off customers who pause mid-sentence.

INT8 or FP8 quantization of the encoder usually costs little accuracy; verify it on the slice table, not only the aggregate. Then close the loop: log low-confidence segments, agent corrections and clips where entity post-correction fired, and feed them back to Step 3. That stream is the most valuable training data you will ever get, because it is exactly the distribution your model fails on.

## Edge cases that break real systems

| Edge case | Symptom | Fix |
| --- | --- | --- |
| Silence, hold music, ringing | Invented sentences, "thank you for watching" | VAD before decoding, no-speech threshold, hallucination metric in gate |
| Long audio | Words dropped or duplicated at chunk boundaries | Overlapping chunks, merge on timestamps, cut at VAD pauses |
| Code-switching | English terms transliterated or dropped | Code-switched data in fine-tune, both scripts in tokenizer, bias list |
| Numbers, dates, money | Correct speech, wrong written form | Explicit inverse text normalization stage, test slice for numbers |
| Overlapping speakers | Merged or missing turns | Channel separation (stereo calls), diarization before ASR |
| Very short utterances ("ừ", "dạ") | Deleted or hallucinated filler | Keep short clips in training, tune minimum segment length |
| Wrong language ID | Whole call transcribed in the wrong language | Force the language when known, LID per segment when not |
| Clipping, codec artifacts | Error spikes on one carrier or device | Channel augmentation, slice by carrier in eval |

## Best-practice checklist

- Build a speaker-disjoint, slice-tagged test set before any training data.
- Fix one normalization style guide and apply it to labels, outputs and metrics.
- Filter pseudo-labels by teacher agreement; send the rejects to human labeling.
- Mix general data into every fine-tune and track a general-domain set.
- Assert zero `<unk>` tokens over all training labels before the first step.
- Treat bias lists and domain LMs as deployable configuration with their own regression set.
- Gate releases on per-slice WER, entity error, hallucination rate and latency, never on one number.
- Route low-confidence production audio back into the data engine every cycle.

## Case studies

**1. Careless Whisper (2024).** Koenecke and colleagues found that roughly 1% of Whisper transcriptions in their study contained hallucinated phrases absent from the audio, some of them harmful, and that hallucinations clustered around long non-speech pauses, which made speakers with aphasia disproportionately affected. The lesson for any pipeline: measure hallucination on silence explicitly and strip non-speech with VAD before an attention decoder ever sees it.

**2. Distil-Whisper (2023).** Hugging Face trained a smaller, faster Whisper on pseudo-labels from Whisper large-v2, discarding any segment whose pseudo-label exceeded 10% WER against the dataset's own transcript. The filter mattered: it removed the noisy and misaligned samples that would otherwise teach the student the teacher's errors. Step 3's agreement filter is the same idea when no reference transcript exists.

**3. The normalizer that moves leaderboards.** The Whisper authors devoted an appendix to text normalization because formatting differences (contractions, numerals, spelling variants), not recognition errors, were inflating measured WER on standard datasets, and they released the normalizer so others could score the same way. Comparisons across systems without a shared normalizer measure formatting, not hearing.

## When to fine-tune, and when not to

Fine-tune when the channel, language or vocabulary is far from the base model's training data and you can build a trusted test set to prove the gain. Do not fine-tune to fix a handful of names: a bias list, a domain LM and catalog post-correction get there in a day, ship as configuration and roll back instantly. Retrain the model when the slice table says the acoustics are wrong; reconfigure the decoder when only the words are.
