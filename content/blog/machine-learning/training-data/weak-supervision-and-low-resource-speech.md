---
title: "Weak Supervision and Low-Resource Speech: Scaling ASR Data Past Human Transcription"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A principal-engineer field guide to building speech data without paying humans per minute: Whisper's 680k-hour weak-supervision recipe, why it deliberately filtered machine-generated transcripts, noisy-student pseudo-labeling with confidence and agreement filters, and the low-resource and multilingual playbook (MMS, mining, cross-lingual transfer) — with runnable pipelines, a worked yield calculation, and a symptom-to-fix troubleshooting section for hallucination and feedback loops."
tags:
  - training-data
  - weak-supervision
  - speech-recognition
  - whisper
  - pseudo-labeling
  - noisy-student
  - low-resource
  - multilingual
  - self-training
  - mms
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 33
---

The fastest way to feel the economics of speech data is to price it. A careful human transcriber, with a quality-assurance second pass, runs somewhere around one dollar per audio-minute for general-domain speech — call it \$60 per audio-hour. That number is not a rounding error you can optimize away; it is the wall. Whisper trained on 680,000 hours of audio. At \$60 per hour, transcribing that corpus by hand would cost roughly forty million dollars and occupy an army of annotators for years, and you would still have to do it again for the next language. Nobody built Whisper that way, because nobody could.

So the interesting question in modern speech is not "how do we label audio well?" It is "how do we get away with barely labeling it at all?" The answer is a family of techniques that trade label *cleanliness* for label *quantity* at enormous scale: weak supervision (scrape audio that already ships with a transcript, however messy), and self-training or pseudo-labeling (let a model label audio for the next model). This post is the working engineer's tour of that toolkit — how it works, the one filtering step everyone forgets, the failure modes that quietly poison a corpus, and what changes when your target language has forty hours of data instead of forty thousand.

![Human transcription versus large-scale weak supervision: paying per audio-minute caps a corpus at thousands of hours, while scraped weak labels reach hundreds of thousands.](/imgs/blogs/weak-supervision-and-low-resource-speech-1.webp)

The diagram above is the mental model for the whole article. The left column is the world we are leaving: pay per minute, get clean labels, hit a ceiling somewhere in the low thousands of hours, and stay stuck in a narrow domain because that is all you could afford to annotate. The right column is where the field went: scrape audio that already has a transcript attached, pay essentially nothing per label, reach hundreds of thousands of hours, and accept noisy labels across a broad domain — which, counterintuitively, is exactly what makes the resulting model robust. Every section below is a stop on the arrow between those two columns.

This is the twenty-second post in the Training Data series. It assumes you have read, or can skim, the companion piece on [speech data for ASR](/blog/machine-learning/training-data/speech-data-for-asr) — that post covers the clean-data side (formats, sampling rates, forced alignment, VAD segmentation); this one covers what to do when you cannot afford clean data. It is also a sibling of [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), because pseudo-labeling is nothing more than synthetic labels for real audio.

## Why weak supervision is different from what you expect

Most engineers arrive at speech data with a supervised-learning reflex: more labels, cleaner labels, higher accuracy. Speech at scale breaks two of those three intuitions.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| Cleaner labels are always better | Hand-corrected transcripts beat noisy ones | Below a noise floor, *quantity and diversity* dominate label purity; a 680k-hour noisy corpus beats a 5k-hour pristine one out-of-distribution |
| Robustness comes from more parameters | Bigger models generalize better | Robustness comes from *broad weakly-labeled data*; a model trained on one clean corpus can be superhuman on its test set and brittle everywhere else |
| Machine-labeled data is free accuracy | Let a good ASR label everything and train on it | Unfiltered machine labels teach the *other model's mistakes*; you must filter, and Whisper deliberately removed transcripts that looked machine-generated |
| Confidence tells you what to keep | High softmax probability means correct | Neural ASR is systematically over-confident, and it is *most* confident exactly when it hallucinates on silence |
| Low-resource is just "less data" | Same recipe, smaller dataset | Below a few hundred hours the whole strategy changes: transfer, mining, and alignment replace collection |

Hold onto the second row, because it is the thesis that justifies the entire enterprise. The reason to prefer 680,000 messy hours over 5,000 clean ones is not that noise is good — it is that **breadth buys robustness, and you can only afford breadth if you stop paying per label.** A model fine-tuned on LibriSpeech can post a word-error rate on the LibriSpeech test set that beats human transcribers, then fall apart on a phone call, a podcast, an accent, or a noisy café. Weak supervision spends its budget on covering the world instead of perfecting one corner of it.

> The senior instinct here is inverted from the rest of ML: your job is not to make the labels cleaner, it is to make the *distribution* wider and then defend the training set against the specific ways cheap labels lie.

## 1. The scale problem: transcription cost is the whole story

**Rule of thumb: if a data strategy requires a human to touch every second of audio, it does not scale past the low thousands of hours, full stop.**

Let us make the wall concrete. Suppose you want to build an ASR system for a new domain — say, medical dictation in English. You price out professional transcription:

- General transcription: about \$1.00–\$1.50 per audio-minute, or \$60–\$90 per hour.
- Specialized (medical, legal) with terminology QA: \$2–\$4 per audio-minute.
- Turnaround: days to weeks for large batches, because it is human-rate-limited.

Now do the arithmetic for a modest target of 10,000 hours — small by modern standards:

```
10,000 hours × 60 min/hour × $1.00/min = $600,000   (general)
10,000 hours × 60 min/hour × $2.50/min = $1,500,000  (medical)
```

Six hundred thousand dollars for a corpus that a research lab would consider a starter dataset. And that buys you 10,000 hours in *one* domain and *one* language. Whisper's 680,000 hours across 96 languages is simply not reachable on this cost curve — it is off the top of the chart by two orders of magnitude.

There is a second, subtler cost that the dollar figure hides: **humans are also a bottleneck on diversity.** When you pay per minute, you rationally spend on the cleanest, easiest-to-transcribe audio, because messy audio costs more to transcribe accurately. So the manual-transcription reflex does not just cap your hours — it biases your corpus toward exactly the clean, read, studio-quality speech that produces brittle models. The cost wall and the robustness problem are the same wall seen from two sides.

The escape is to find audio that *already comes with a transcript someone else produced for a different reason*: closed captions on videos, subtitles on films, transcripts of parliamentary proceedings, lyrics, audiobook–ebook pairs, podcast show-notes. The label was a byproduct of publishing, so its marginal cost to you is zero. It is noisy, misaligned, sometimes wrong, sometimes machine-generated — and there is an ocean of it. That is weak supervision.

## 2. Whisper's recipe: 680,000 hours of weak supervision

**Rule of thumb: minimal filtering plus one deliberate, non-obvious cut turns a web-scale mess into a robust model — the discipline is in the one cut, not the many.**

Whisper (Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022) is the canonical demonstration that the right column of our mental model actually works. The recipe, stripped to its load-bearing parts:

![Whisper's weak-supervision data pipeline: minimal filtering plus one deliberate cut (machine transcripts) turns messy web audio into a robust model.](/imgs/blogs/weak-supervision-and-low-resource-speech-2.webp)

1. **Scrape audio paired with transcripts from the internet.** 680,000 hours total: about 438,000 hours of English transcription, roughly 117,000 hours covering 96 other languages, and about 125,000 hours of "translate this non-English audio into English text" pairs. The multitask framing (transcribe *or* translate, in any of many languages) falls out of just collecting whatever the web already paired together.
2. **Language identification and alignment.** Run a spoken-language detector on the audio and check it against the transcript's language. If they disagree, the pair is not a valid transcription example — unless the transcript is English, in which case it becomes a *translation* example. Segment long audio into 30-second windows, each paired with the transcript text falling in that window.
3. **Deliberately drop machine-generated transcripts** — the step everyone underweights, covered in its own section below. This is the "one cut."
4. **Light deduplication and heuristics** — fuzzy-dedup near-identical transcripts, drop obviously broken pairs.
5. **Train a plain sequence-to-sequence Transformer** to predict the raw text. No pronunciation lexicon, no separate language model, no dataset-specific normalization at training time.
6. **Ship a model that works zero-shot** — no fine-tuning on the target benchmark. This is the part that made people pay attention.

The result reframed the field. On standard benchmarks like LibriSpeech test-clean, a well-tuned supervised model can beat Whisper's raw number. But move off-distribution — different microphones, accents, background noise, spontaneous speech — and the supervised model's error rate explodes while Whisper degrades gracefully. Averaged across a suite of diverse datasets, Whisper approached human-level *robustness* without ever seeing those datasets in training. That is the payoff of breadth: **the model was never allowed to overfit one corpus, because it never saw one corpus.**

A useful way to internalize this: a LibriSpeech-tuned model is a specialist who has memorized one textbook; Whisper is a generalist who has read a messy, contradictory, enormous library. On the exam that matches the textbook, the specialist wins. On everything else — which is what "production" means — the generalist wins by a mile.

## 3. Filtering machine-generated transcripts

**Rule of thumb: never train on another ASR system's output as if it were ground truth — you will inherit its accent, its vocabulary gaps, and its exact hallucinations.**

Here is the counterintuitive discipline at the center of Whisper's recipe, and the thing most teams skip when they build a scraping pipeline. A large fraction of "transcripts" on the internet are not human-written at all — they are the output of *some other ASR system*. YouTube auto-captions are the obvious example, but it is everywhere: auto-generated subtitles, transcription-service exports, phone-system logs.

Why is that poison? Because if you train Model B on Model A's outputs, Model B does not learn to transcribe speech — it learns to *imitate Model A*, including everything Model A gets wrong. Model A's systematic mistakes (its confusion between similar-sounding words, its handling of numbers, its behavior on accents it was weak on) become Model B's targets. You are laundering one system's error profile into your training labels and calling it supervision. The Whisper authors found this measurably hurt performance and built heuristics specifically to detect and remove machine-generated transcripts.

![Detecting machine-generated transcripts: casing, punctuation, and repetition signals feed a classifier that keeps human captions and drops ASR-shaped ones.](/imgs/blogs/weak-supervision-and-low-resource-speech-3.webp)

How do you tell a human transcript from a machine one, at scale, without listening to any of it? You lean on the fact that **humans and ASR systems format text differently:**

- **Casing.** A transcript that is entirely uppercase, or entirely lowercase, is very unlikely to be human. People write "The meeting starts at 9." Old ASR pipelines often emit "the meeting starts at nine" or "THE MEETING STARTS AT NINE".
- **Punctuation.** Many ASR systems, especially older ones, produce unpunctuated streams. A long transcript with zero commas, periods, or question marks is a strong machine signal.
- **Normalization artifacts.** Spelled-out numbers everywhere ("two thousand twenty three"), no capitalization of proper nouns, no apostrophes in contractions — these are the fingerprints of text that was generated, then normalized, by a machine.
- **Repetition and looping.** ASR output on hard audio tends to repeat n-grams; human transcribers do not loop.

None of these is decisive alone, so you combine them. Here is a compact, runnable detector that turns those signals into a keep/drop decision — the same heuristics double as a *hallucination* detector later, because a looping ASR output and a machine-generated transcript share a signature:

```python
import re, zlib
from collections import Counter

def compression_ratio(text: str) -> float:
    """gzip ratio: highly repetitive text compresses far better, so a high
    ratio flags looping/degenerate output. Whisper uses 2.4 as its threshold."""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0

def max_ngram_repeat(text: str, n: int = 3) -> float:
    """fraction of the most-repeated n-gram; loops push this toward 1.0."""
    toks = text.split()
    if len(toks) < n:
        return 0.0
    grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    return max(counts.values()) / len(grams)

def looks_machine_or_hallucinated(text, no_speech_prob=0.0, avg_logprob=0.0):
    reasons = []
    if compression_ratio(text) > 2.4:
        reasons.append("high-compression (repetition)")
    if max_ngram_repeat(text) > 0.25:
        reasons.append("looping n-gram")
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        reasons.append("no-speech segment with low logprob")
    stripped = text.strip()
    uniform_case = stripped and (stripped == stripped.upper() or stripped == stripped.lower())
    if uniform_case and not re.search(r"[.?!,;:]", stripped) and len(stripped.split()) > 8:
        reasons.append("uniform case + no punctuation (ASR-style)")
    return reasons
```

For a production pipeline you graduate from hand-tuned thresholds to a learned classifier: featurize each transcript (casing ratio, punctuation density, compression ratio, out-of-vocabulary rate, character n-gram entropy), label a few thousand examples by hand as human or machine, and train a gradient-boosted classifier. It costs a day and it is the single highest-leverage filter in a scraping pipeline. The [classifier-and-perplexity-based quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering) post covers the same pattern for text corpora — the machinery transfers directly.

> If you remember one thing from this post: scraping audio-with-captions and training on all of it is a trap, because a large slice of those captions are a robot impersonating a human. The value is in the human captions; the whole game is separating them.

## 4. Pseudo-labeling and noisy-student self-training

**Rule of thumb: a model good enough to be useful is good enough to be a teacher — as long as you filter what it teaches.**

Weak supervision harvests labels the world already made. *Self-training* manufactures new ones: take a model that works reasonably well, point it at unlabeled audio (of which there is effectively infinite), let it produce transcripts, keep the ones you trust, and train the next model on the union of your real labels and these *pseudo-labels*. Then the improved model becomes the teacher and you go around again. This is the noisy-student recipe — "noisy" because the student trains with heavy augmentation (SpecAugment, speed perturbation, added noise) so it cannot simply memorize the teacher.

<figure class="blog-anim">
<svg viewBox="0 0 720 400" role="img" aria-label="A noisy-student self-training loop: a teacher ASR labels unlabeled audio, a confidence filter keeps the best pseudo-labels, a student trains on them and becomes the next round's teacher" style="width:100%;height:auto;max-width:760px">
<style>
.wl-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#9aa3af);stroke-width:2}
.wl-t{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.wl-s{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.wl-arr{stroke:var(--text-secondary,#9aa3af);stroke-width:2;fill:none;marker-end:url(#wl-head)}
.wl-dot{fill:var(--accent,#6366f1)}
.wl-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
@keyframes wl-move{0%{offset-distance:0%}100%{offset-distance:100%}}
.wl-dot{offset-path:path('M 360 57 L 600 183 L 360 333 L 120 183 Z');animation:wl-move 9s linear infinite}
@media (prefers-reduced-motion:reduce){.wl-dot{animation:none}}
</style>
<defs>
<marker id="wl-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0 0 L10 5 L0 10 z" fill="var(--text-secondary,#9aa3af)"/>
</marker>
</defs>
<line class="wl-arr" x1="410" y1="72" x2="558" y2="162"/>
<line class="wl-arr" x1="558" y1="204" x2="410" y2="318"/>
<line class="wl-arr" x1="310" y1="318" x2="162" y2="204"/>
<line class="wl-arr" x1="162" y1="162" x2="310" y2="72"/>
<rect class="wl-box" x="250" y="24" width="220" height="66" rx="10"/>
<text class="wl-t" x="360" y="52">teacher ASR</text>
<text class="wl-s" x="360" y="74">round k</text>
<rect class="wl-box" x="500" y="150" width="200" height="66" rx="10"/>
<text class="wl-t" x="600" y="178">pseudo-labels</text>
<text class="wl-s" x="600" y="200">on unlabeled audio</text>
<rect class="wl-box" x="250" y="300" width="220" height="66" rx="10"/>
<text class="wl-t" x="360" y="328">confidence filter</text>
<text class="wl-s" x="360" y="350">keep the best ~62%</text>
<rect class="wl-box" x="20" y="150" width="200" height="66" rx="10"/>
<text class="wl-t" x="120" y="178">student ASR</text>
<text class="wl-s" x="120" y="200">round k+1</text>
<circle class="wl-dot" r="10"/>
<text class="wl-cap" x="360" y="205">iterate</text>
</svg>
<figcaption>The self-training loop: the teacher labels unlabeled audio, a confidence filter keeps only the best pseudo-labels, the student trains on them and is promoted to teacher for the next round.</figcaption>
</figure>

The animation traces the loop. The single most important box is "confidence filter." Skip it and you are back to the machine-transcript trap of the previous section — you are training on the teacher's raw output, mistakes and all. The filter is what turns self-training from a copy machine into an improvement engine.

### Scoring a pseudo-label: confidence and agreement

You need a per-clip quality score that correlates with actual correctness. Two signals do most of the work:

1. **Model confidence.** For a sequence-to-sequence model, the natural score is the geometric mean of per-token probabilities — equivalently, the exponential of the average token log-probability:

$$c(x) = \exp\!\left(\frac{1}{T}\sum_{t=1}^{T}\log p(y_t \mid y_{\lt t}, x)\right)$$

where $x$ is the audio, $y_t$ is the $t$-th predicted token, and $T$ is the transcript length. A value near 1 means the model was rarely surprised by its own output; a low value means it hedged. (Whisper exposes this as `avg_logprob` per segment.)

2. **Cross-model agreement.** Run a *second, architecturally different* model on the same audio and measure how far its transcript is from the first, using character error rate (CER). When two independent models agree, the transcript is far more likely correct than when either is confident alone. Agreement is the antidote to the calibration problem, because two models rarely hallucinate the *same* wrong text.

Combine them with a hard gate, and add the no-speech and compression signals from the detector so you also throw out hallucinations:

```python
# pseudo_label.py — one pass of the self-training loop:
# teacher transcribes unlabeled audio; we score each clip and keep survivors.
import math
from dataclasses import dataclass
from faster_whisper import WhisperModel   # pip install faster-whisper
import jiwer                               # pip install jiwer

teacher = WhisperModel("large-v3", device="cuda", compute_type="float16")
checker = WhisperModel("medium",   device="cuda", compute_type="float16")  # 2nd model

@dataclass
class Scored:
    path: str; text: str; conf: float
    no_speech_prob: float; compression_ratio: float; agree_cer: float

def transcribe(model, path):
    segments, _info = model.transcribe(path, beam_size=5, vad_filter=True)
    segs = list(segments)
    dur = sum(s.end - s.start for s in segs) or 1e-6
    text = " ".join(s.text.strip() for s in segs)
    avg_lp = sum((s.end - s.start) * s.avg_logprob       for s in segs) / dur
    nsp    = max((s.no_speech_prob for s in segs), default=1.0)
    comp   = sum((s.end - s.start) * s.compression_ratio for s in segs) / dur
    return text, math.exp(avg_lp), nsp, comp

def score(path) -> Scored:
    t_text, conf, nsp, comp = transcribe(teacher, path)
    c_text, *_ = transcribe(checker, path)
    agree_cer = jiwer.cer(t_text, c_text) if (t_text and c_text) else 1.0
    return Scored(path, t_text, conf, nsp, comp, agree_cer)

def keep(s: Scored) -> bool:
    return (s.conf >= 0.85               # teacher was confident
            and s.agree_cer <= 0.15      # a 2nd model largely agrees
            and s.no_speech_prob < 0.5   # there is actually speech
            and s.compression_ratio < 2.4)  # not a loop / hallucination
```

Run it over a corpus with a plain shell loop, writing survivors to a manifest the trainer reads:

```bash
# Fan out over shards; each worker appends kept (audio, pseudo-label) rows.
find /data/unlabeled -name '*.wav' | \
  parallel -j 8 python pseudo_label.py --keep-only --out kept/{%}.jsonl {}
cat kept/*.jsonl > train_pseudo.jsonl
wc -l train_pseudo.jsonl   # how many survived the filter
```

### Worked scenario: what the filter actually costs and buys

Numbers make this real. Suppose you have **50,000 hours of unlabeled English podcast audio** and **10,000 hours of clean, human-labeled data** in a matching domain. You run Whisper large-v3 as the teacher over all 50,000 hours, score each clip, and bin by confidence:

![Confidence-threshold triage of pseudo-labels: keeping only clips whose confidence clears the bar trades quantity for label quality — about 62 percent survive.](/imgs/blogs/weak-supervision-and-low-resource-speech-4.webp)

Read the table top to bottom. The clips where the teacher was most confident (confidence at least 0.95) are 38% of the corpus and have a genuinely low measured CER of about 3.1% against a held-out human check, with 97% cross-model agreement — keep them without hesitation. The next band (0.85–0.95) is another 24%, CER around 8.4%, still good — keep. The 0.70–0.85 band is where quality falls off a cliff (CER ~17%, agreement drops to 71%); route it to spot-check or drop. Below 0.70, the transcripts are junk (CER ~39%) and full of hallucinations — drop unconditionally.

Set the threshold at confidence ≥ 0.85 **and** agreement CER ≤ 0.15, and you keep the top two bands: roughly **62% of clips survive**, giving you about **31,000 hours of pseudo-labeled audio** at a blended CER in the single digits. The arithmetic of the tradeoff:

```
Unlabeled pool:            50,000 hr
Kept at threshold (62%):   31,000 hr  (blended CER ~5.5%)
Dropped (38%):             19,000 hr  (would have added CER ~28%)

Student A: 10,000 hr clean only                 -> held-out WER 9.1%
Student B: 10,000 clean + 31,000 kept pseudo    -> held-out WER 7.4%
Student C: 10,000 clean + all 50,000 unfiltered -> held-out WER 8.6%
```

Two lessons hide in those three rows. First, filtered self-training bought a 1.7-point WER reduction — real money on a hard out-of-domain test. Second, and more important: **the unfiltered variant (Student C) was worse than the filtered one despite having 19,000 more hours.** The dropped 38% did not merely add nothing; they actively dragged the model toward the teacher's error profile. This is the entire argument for the filter compressed into one comparison. Quantity past the noise floor is not free — it can be negative.

Iterate the loop two or three times and the yield curve shifts: as the student improves, more clips clear the confidence bar next round, and the CER within each band drops. Diminishing returns usually set in by the third generation. Do not iterate forever; watch a held-out human-labeled dev set, and stop when it stops moving.

## 5. Low-resource and multilingual: when there is no ocean to scrape

**Rule of thumb: below a few hundred hours, stop thinking "collect more" and start thinking "transfer, mine, and align" — the recipe inverts.**

Everything so far assumed abundance: an ocean of captioned English video, a large clean seed set. Most of the world's ~7,000 languages have neither. For a language with forty hours of audio and no captioned web presence, weak supervision has nothing to scrape and self-training has no competent teacher to bootstrap from. The strategy changes completely.

![Strategies for low-resource and multilingual speech: no single tactic covers 1000+ languages; MMS-style systems layer transfer, mining, and community data together.](/imgs/blogs/weak-supervision-and-low-resource-speech-5.webp)

The tree above lays out the four moves that actually cover the long tail. None of them works alone; systems like Meta's MMS combine all four.

**Cross-lingual transfer.** Pre-train a single self-supervised encoder (wav2vec 2.0 style) on massive *unlabeled* multilingual audio — no transcripts needed, just raw speech in hundreds of languages. Speech shares acoustic structure across languages (phonation, formants, prosody), so the encoder learns representations that transfer. Then fine-tune with a tiny language-specific head (or an adapter) on the forty hours you do have. The heavy lifting is done by languages you are *not* targeting. This is why a low-resource language can reach usable accuracy on hours of labeled data instead of thousands.

**Speech–text mining and alignment.** For many under-served languages the one text-with-audio resource that exists is *read religious text* — recordings of people reading scripture, available in over a thousand languages. That is the data engine behind MMS (Pratap et al., *Scaling Speech Technology to 1,000+ Languages*, 2023). The problem is that these recordings are long (whole chapters) and only loosely aligned to the text. The fix is a **forced-alignment model**: a CTC-based aligner that maps a long audio recording to its known transcript, cutting it into short, precisely-aligned (audio, text) segments you can train on. Mining found text and aligning it to found audio manufactures supervised pairs where none were annotated by hand. MMS released the aligner as a tool precisely because alignment is the reusable primitive.

**Community collection.** Sometimes you just need people to donate speech. Mozilla's Common Voice is the reference model: volunteers read prompted sentences (and validate each other's recordings) under a permissive license, across scores of languages. It is slow and skews toward read rather than spontaneous speech, but for a language with nothing, a few dozen validated hours is the difference between an impossible and a merely hard problem. Pair it with transfer and it goes further than its hour count suggests.

**Self-training, again.** Once transfer plus mining plus community data gets you a first usable model, you can point it at unlabeled radio broadcasts, phone data, or web audio in that language and run the pseudo-labeling loop from Section 4 — now bootstrapped in a language that started with nothing. The loop is language-agnostic; it just needs a teacher good enough to start.

Here is the transfer move in code — fine-tuning a pretrained multilingual model on a small labeled set with Hugging Face:

```python
from transformers import (Wav2Vec2ForCTC, AutoProcessor,
                          TrainingArguments, Trainer)

# MMS ships with per-language adapters; load the base and target one language.
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/mms-1b-all", target_lang="lug",   # e.g. Luganda
    ignore_mismatched_sizes=True)
model.init_adapter_layers()          # train only the small adapter + head
processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")

# Freeze the 1B-parameter backbone; adapt on ~tens of hours, not thousands.
model.freeze_base_model()

args = TrainingArguments(
    output_dir="mms-lug", per_device_train_batch_size=16,
    learning_rate=1e-3, num_train_epochs=4, fp16=True,
    gradient_checkpointing=True, eval_strategy="steps", eval_steps=200)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=dev_ds,
                  data_collator=collator, tokenizer=processor.feature_extractor)
trainer.train()
```

The whole design is right there: a billion-parameter backbone frozen, a small adapter trained on a handful of hours, and cross-lingual transfer carrying the rest. MMS used this to cover over 1,100 languages for ASR and to build language identification for 4,000+ — roughly an order of magnitude more languages than Whisper, by leaning on mining and transfer instead of scraping captions that, for most of these languages, simply do not exist. For a broader treatment of squeezing signal from small corpora, see [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning).

## 6. Accent and dialect balance: the bias weak supervision bakes in

**Rule of thumb: scraped data mirrors who publishes, not who speaks — measure accuracy per accent slice or you will ship a model that fails the people it was not trained on.**

Weak supervision has a demographic problem, and it is worth naming because it is easy to miss when you are staring at an aggregate WER. The captioned audio on the web is not a representative sample of human speech. English on the internet skews heavily toward North American and British accents, toward broadcast and professional voices, toward standard dialects. Scrape it and your corpus inherits that skew; train on the corpus and your model is quietly worse for a Nigerian, Scottish, Indian, or Southern US speaker than for a network anchor.

The failure is invisible in a single number. An aggregate WER of 7% can hide a 4% error rate on the majority accent and a 20% error rate on an underrepresented one. The fix is measurement discipline first, data second:

- **Evaluate on stratified slices.** Build or buy accent- and dialect-labeled test sets and report WER *per slice*, never only in aggregate. A model is not "7% WER"; it is "4% here, 20% there," and the second number is the one that matters for fairness and for the tail of your users.
- **Reweight or resample during training.** If your corpus is 80% majority accent, upweight the minority slices so the loss does not ignore them. This is the same domain-weighting logic covered in [data mixing, domain weighting, and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum), applied to accent as the domain axis.
- **Targeted collection for the gaps.** This is the one place where paying for data still makes sense: a few hundred hand-collected hours in an underrepresented accent, added to a huge weak-supervision base, closes slice gaps far more cost-effectively than trying to fix the whole corpus.

The general principle: weak supervision is fantastic at breadth but inherits the *shape* of what got published, and that shape has holes. You cannot scrape your way out of a demographic gap — you have to measure it and patch it deliberately.

## 7. Troubleshooting: how weak-supervision pipelines fail

Every technique above has a characteristic failure mode. Here is the field guide — symptom, cause, fix — for the four that will actually bite you.

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Fluent text on silence/music; "Thank you for watching!" | Decoder is a language model with no "say nothing" option | Gate on `no_speech_prob` + VAD; drop segments; log-prob threshold |
| Same phrase repeated 20× | Autoregressive loop on ambiguous/hard audio | Compression-ratio + n-gram-repeat filter; temperature fallback |
| Pseudo-labels confidently wrong | ASR is systematically over-confident | Cross-model agreement, not confidence alone; calibrate on a dev set |
| Student plateaus, then regresses | Feedback loop: student inherits teacher's systematic errors | Keep a fixed human dev set; hold out real labels; cap iterations |
| Low-resource WER stuck high | Not enough data, wrong recipe | Transfer + mining + alignment, not more scraping |

### 7.1 Hallucinated transcripts

**Symptom.** The teacher emits fluent, grammatical, completely fabricated text over segments that contain no speech — silence, music, applause, room tone. The canonical artifact is Whisper writing "Thank you for watching!" or "Please subscribe to my channel" over silence, because those phrases are over-represented at the end of the YouTube videos it trained on. On hard audio it does the other failure: it locks into a loop and repeats a word or phrase dozens of times.

**Detection.** This is the single most important failure to catch, because a hallucination is *fluent* — it will not look wrong to a text-quality filter. Three signals catch it:

![How an ASR teacher hallucinates on non-speech: on silence and music the decoder invents fluent text or loops; the anomaly shows up in no-speech and compression signals.](/imgs/blogs/weak-supervision-and-low-resource-speech-6.webp)

The figure shows the three columns you check. On the genuine-speech segment the transcript is correct, the average log-probability is healthy (around −0.3), and nothing fires. On silence, the model invents a plausible sentence but the `no_speech_prob` spikes to ~0.9. On music, it loops, and the compression ratio blows up to ~6 (the `looks_machine_or_hallucinated` function from Section 3 catches both). The point is that hallucinations are undetectable in the *text* but obvious in the *decoder's own uncertainty signals* and the *repetition structure*.

**Fix.** Layer three defenses: (1) run a voice-activity detector first and never feed non-speech segments to the model — `vad_filter=True` in faster-whisper, or a standalone Silero/WebRTC VAD; (2) gate on `no_speech_prob` and `avg_logprob` and drop segments that fail; (3) apply the compression-ratio and n-gram-repeat filters to kill loops. In self-training, hallucinated pseudo-labels are especially toxic because they are confident, so they sail through a naive confidence filter — which is exactly why you also need cross-model agreement. This failure mode is covered from the fine-tuning side in [debugging ASR fine-tuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning), and the mechanism of why the decoder does this is in [Whisper Under the Hood](/blog/machine-learning/signal-processing/whisper-under-the-hood).

### 7.2 Confidence miscalibration in pseudo-labels

**Symptom.** You filter pseudo-labels at confidence ≥ 0.9 expecting near-perfect transcripts, but the kept set still has an unexpectedly high error rate — the model is confidently wrong, not confidently right.

**Cause.** Modern neural ASR is systematically over-confident: softmax probabilities are not calibrated to actual accuracy, and beam search amplifies this by selecting high-probability paths. The model is *most* confident on exactly the degenerate outputs (repetitions, common phrases) you most want to reject. Raw confidence is a biased estimator of correctness.

**Fix.** Do not trust a single model's confidence as a correctness estimate. Three moves, in order of leverage: (1) **cross-model agreement** — two architecturally different models agreeing is far stronger evidence than one model's confidence, because independent models rarely make the *same* mistake; (2) **calibrate the threshold empirically** — transcribe a few hundred clips, get human CER for each, and plot CER against confidence to find the confidence level that actually delivers your target error rate (do not assume 0.9 means 90% accurate); (3) **round-trip consistency** — check that the transcript, re-synthesized or re-scored, is stable. Agreement is the workhorse; confidence alone is a trap.

### 7.3 Label-noise feedback loops

**Symptom.** Self-training improves for one or two generations, then stalls, and by the third or fourth generation the model *regresses* on your real held-out test set even as the pseudo-labels look ever more confident.

**Cause.** This is the defining risk of self-training and it is subtle. The student learns the teacher's *systematic* errors — say, the teacher consistently mistranscribes a technical term or drops a particular accent's vowel. Those errors become training targets, the student reproduces them more confidently, they clear the confidence filter more easily next round, and the loop amplifies its own mistakes. You have built a machine for compounding a bias. It is the ASR analogue of model collapse in synthetic text data, covered in [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation).

**Fix.** Break the loop's grip on reality: (1) **always retain the real human-labeled data** in every generation's training mix — never train on pseudo-labels alone, so the model is continually pulled back toward ground truth; (2) **hold out a fixed human-labeled dev set** that is *never* pseudo-labeled and gate every generation on it — the moment it stops improving, stop; (3) **cap iterations** at two or three generations, because the gains diminish and the risk grows; (4) **inject diversity** — use a *different* architecture as the teacher across rounds, or add fresh unlabeled data, so the loop cannot converge onto one error profile. The dev set is the load-bearing defense: without an independent measurement of reality, a feedback loop feels like progress right up until it isn't.

### 7.4 Low-resource data scarcity

**Symptom.** You apply the standard recipe — collect data, fine-tune — to a low-resource language and the WER is stuck at an unusable level no matter how you tune. Self-training does not help because your teacher is too weak to produce usable pseudo-labels.

**Cause.** You are trying to run an abundance recipe in a scarcity regime. There is no ocean to scrape and no competent teacher to bootstrap. More of the same effort does not move the needle because the bottleneck is not tuning — it is data that does not exist.

**Fix.** Invert the recipe, per Section 5: (1) start from a **massively multilingual self-supervised encoder** (MMS, wav2vec 2.0 XLSR) and fine-tune a small adapter rather than training from scratch — let cross-lingual transfer do the heavy lifting; (2) **mine and align** whatever found text-with-audio exists (read texts, broadcasts) using a forced-alignment model to manufacture supervised segments; (3) **collect a small, high-value seed** via community efforts (Common Voice) to anchor the fine-tune; (4) *only then* run self-training, now that you have a teacher worth iterating. The order matters: transfer and mining come first; self-training is the last step, not the first.

## Case studies

### 1. Whisper — weak supervision at 680k hours

Whisper is the existence proof for the right column of our mental model. OpenAI scraped 680,000 hours of audio-with-transcripts from the web — 438k hours English transcription, 117k hours across 96 other languages, 125k hours of translation pairs — applied *minimal* filtering with one deliberate exception (removing machine-generated transcripts, plus language-match checks and dedup), and trained a plain encoder-decoder Transformer to predict raw text. The headline result was not a benchmark record; a supervised model can beat Whisper on LibriSpeech test-clean. The result was *robustness*: across a suite of diverse out-of-distribution datasets it had never seen, Whisper approached human-level error rates zero-shot, while supervised models fine-tuned on a single corpus degraded sharply off-distribution. The lesson that reshaped the field: for real-world robustness, breadth of weak supervision beats depth of clean supervision. The one non-obvious discipline — filtering machine transcripts — is what kept the breadth from becoming garbage.

### 2. MMS — 1,100+ languages via alignment and mining

Meta's Massively Multilingual Speech project (2023) is the low-resource playbook executed at extreme scale. Where Whisper scraped captions (which exist mainly for high-resource languages), MMS confronted the fact that most of the world's languages have no captioned web presence at all. Its data engine was *found read text* — recordings of people reading religious texts, available in over a thousand languages — combined with self-supervised pre-training (wav2vec 2.0) on large unlabeled multilingual audio. The critical primitive was a **forced-alignment model** that turned long, loosely-labeled recordings into short precisely-aligned training segments; Meta released it as a reusable tool. Layering self-supervised transfer, mining-plus-alignment, and per-language adapters, MMS built ASR for 1,100+ languages and language ID for 4,000+ — roughly 10× Whisper's language coverage — and reported substantially lower error rates than Whisper on many overlapping languages. The takeaway: below the scraping threshold, *alignment and transfer are the scaling levers*, not collection.

### 3. Improved Noisy Student — iterative pseudo-labeling on LibriLight

Google's noisy-student work on ASR (Park et al., 2020) is the canonical demonstration that self-training, done carefully, sets records. Starting from a model trained on the labeled LibriSpeech (960 hours), they pseudo-labeled the ~60,000 unlabeled hours of LibriLight, filtered and normalized the labels, trained a student with heavy SpecAugment and a language-model fusion, promoted it to teacher, and iterated. The result was state-of-the-art LibriSpeech WER (on the order of 1.4% test-clean / 2.6% test-other) — a large gain from *unlabeled* audio and a strong teacher. The load-bearing details were exactly the ones this post emphasizes: strong augmentation on the student so it does not merely copy, filtering and normalization of pseudo-labels, and a bounded number of generations. It is the clean, controlled version of the loop in Section 4.

### 4. wav2vec 2.0 + self-training — labels from ten minutes

The wav2vec 2.0 line (Baevski et al., 2020) pushed the low-resource case to its limit: self-supervised pre-training on unlabeled audio, then fine-tuning on as little as *ten minutes* of labeled speech, reaching usable WER — and combining self-supervision with self-training (pseudo-labeling the rest) pushed it further. It is the proof that the encoder can learn almost everything about the acoustics of speech from raw audio, leaving only a tiny mapping-to-text to be supervised. This is the mechanism underneath the cross-lingual transfer branch of the low-resource tree: pre-train once on unlabeled multilingual audio, adapt cheaply per language.

### 5. Common Voice — community collection as infrastructure

Mozilla's Common Voice is the reference for the community-collection branch. Volunteers read prompted sentences and validate each other's recordings, all under a CC0 license, across scores of languages. It skews toward read (not spontaneous) speech and grows slowly, but it created labeled data for languages that had essentially none, and its permissive license made it a base layer that everyone — including the multilingual models above — could build on. The lesson: for the true long tail, sometimes the answer is not an algorithm but an institution that makes volunteering to donate speech easy and legally clean.

### 6. The "thank you for watching" epidemic — a hallucination post-mortem

When Whisper shipped, practitioners quickly noticed it writing "Thank you for watching!", "Please subscribe", and similar YouTube-outro phrases over silence, and looping words over music and noise. This is not a bug in the decoder so much as a data artifact: those phrases were over-represented in the training transcripts (they end countless videos), so on a segment with no speech signal to constrain it, the decoder's language-model prior fills the vacuum with its most likely completion. The post-mortem lesson for anyone using Whisper *as a teacher* in self-training is direct: these fluent fabrications are confident, so they pass a naive confidence filter and poison the student. The defense is the layered detector from Section 3 plus VAD plus cross-model agreement — never confidence alone. It is the most instructive failure in the toolkit because it shows that the most dangerous errors are the fluent ones.

## When to reach for weak supervision — and when not to

**Reach for weak supervision and self-training when:**

- You need **breadth and robustness** across domains, accents, and conditions more than a record on one benchmark — this is the Whisper case, and it is most production ASR.
- **Audio-with-transcripts already exists** at scale for your language (captioned video, subtitled media, proceedings) and you can scrape it legally.
- You have a **large pool of unlabeled audio** and a **decent seed model** to bootstrap self-training — the pool is cheap; the teacher makes it usable.
- Your target language is **low-resource** but a **multilingual pretrained encoder** covers it — transfer plus a small adapter beats collecting thousands of hours.
- Your budget cannot absorb **six figures of human transcription** for a starter corpus — which is almost always.

**Skip it, or use it carefully, when:**

- You are in a **narrow, high-stakes, high-accuracy** domain (medical dictation, legal, safety-critical commands) where you need clean labels and can justify paying for them — here, human transcription of a focused corpus is the right call.
- You have **no unlabeled audio and no captioned web presence** in the target language and cannot mine found text — you may be stuck with community collection as the only path, slow as it is.
- You would be **training on another ASR system's raw output** without filtering — do not; you are laundering someone else's errors into your labels.
- You cannot **measure quality on an independent human-labeled dev set** — without it, self-training's feedback loop will feel like progress while it quietly regresses, and you will not know until production.
- You need to **serve underrepresented accents fairly** and have not built per-slice evaluation — fix the measurement before you scale the data, because weak supervision will bake in the web's demographic skew.

The through-line of this entire post is a single inversion of the supervised-learning instinct. In classic supervised learning you make labels cleaner. In weak supervision you make the *distribution wider*, accept that the labels are dirty, and spend your engineering effort on *defending the training set from the specific ways cheap labels lie* — machine-transcript contamination, hallucination on silence, over-confident pseudo-labels, and feedback loops that amplify a teacher's mistakes. Get those four defenses right and you can build a robust multilingual ASR system for a small fraction of what human transcription would cost. Get them wrong and you build a very expensive, very confident machine for reproducing someone else's errors.

## Further reading

- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper, 2022) — the 680k-hour recipe and the robustness result.
- Pratap et al., *Scaling Speech Technology to 1,000+ Languages* (MMS, 2023) — alignment, mining, and per-language adapters.
- Park et al., *Improved Noisy Student Training for Automatic Speech Recognition* (2020) — the iterative pseudo-labeling loop done right.
- Baevski et al., *wav2vec 2.0* (2020) — self-supervised pre-training that makes ten-minute fine-tuning viable.
- Companion posts: [speech data for ASR](/blog/machine-learning/training-data/speech-data-for-asr), [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), [Whisper Under the Hood](/blog/machine-learning/signal-processing/whisper-under-the-hood), and [debugging ASR fine-tuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning).
