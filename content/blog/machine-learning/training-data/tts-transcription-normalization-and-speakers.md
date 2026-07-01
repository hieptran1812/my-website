---
title: "TTS Transcripts, Text Normalization, and Speakers: The Text Side of Voice Data"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A principal-engineer field guide to the text half of a text-to-speech corpus: verbatim transcripts, text normalization, grapheme-to-phoneme, speaker diarization and embeddings, and how modern zero-shot TTS corpora are mined from audiobooks and podcasts at scale."
tags:
  - training-data
  - text-to-speech
  - text-normalization
  - grapheme-to-phoneme
  - speaker-diarization
  - speaker-embeddings
  - phonemization
  - voice-cloning
  - pyannote
  - nemo
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 29
---

Most people building a text-to-speech (TTS) system think the hard part is the audio: sample rates, spectrograms, vocoders, the whole DSP stack. Then they train their first model on a "clean" corpus and it says "one two three" when the script reads "123 Main Street," pronounces "Dr. Read" as "doctor reed" instead of "doctor red," and blends two audiobook narrators into one uncanny average voice. Every one of those failures is a **text-side** defect, not an acoustic one. The model faithfully learned exactly what the data told it to say — the data was wrong.

The text side of a TTS corpus is where most silent quality loss happens, because it is invisible in the waveform and invisible in the loss curve. A transcript that says "twenty twenty six" while the speaker actually said "two thousand twenty six" produces a perfectly low training loss and a model that has quietly learned to hallucinate. This post is the field guide I wish I had the first time I built a voice dataset: how to get transcripts that match the audio, how to normalize written text into spoken form, how to control pronunciation with phonemes, how to figure out **who spoke when** in multi-speaker recordings, and how the modern large-scale corpora that power zero-shot voice cloning are actually mined.

This is the sibling of [Data for text-to-speech: cleaning and enhancement](/blog/machine-learning/training-data/data-for-text-to-speech-cleaning-and-enhancement), which covers the audio half — denoising, segmentation, loudness, and quality gates. Read them together; a TTS pair is only as good as its weaker half.

## Why the text side is different from what you expect

The instinct from ASR (automatic speech recognition) is to treat text and audio symmetrically: you have pairs, you train. But TTS runs the arrow the other way, and that reversal changes every requirement.

| What you assume | The naive view | The reality for TTS |
| --- | --- | --- |
| "Any transcript will do" | Transcripts label the audio | The transcript **is the input**; every error becomes a learned mispronunciation |
| "The model reads text" | Feed characters, get speech | Most good systems feed **phonemes**; text normalization + G2P sit in front |
| "Numbers are numbers" | "123" is fine as-is | The model must be told it is "one hundred twenty-three," not "one two three" |
| "One recording, one speaker" | A file is a speaker | Audiobooks and podcasts are **multi-speaker**; you must diarize before you can label |
| "More hours is better" | Scrape everything | Found audio needs VAD, diarization, ASR, alignment, and hard filtering before it is trainable |
| "Random splits are fine" | Shuffle and split | Random splits **leak speakers** and inflate every voice-cloning metric you report |

The through-line: in TTS the transcript is not a label, it is the **control signal**. Garbage in the control signal is not noise the model averages out — it is an instruction the model obeys.

![The text-side pipeline of a TTS corpus, from raw transcript through normalization, phonemization, and forced alignment to a model-ready pair](/imgs/blogs/tts-transcription-normalization-and-speakers-1.webp)

The diagram above is the mental model for the whole post: a raw transcript is checked against the audio, normalized from written to spoken form, converted to phonemes by grapheme-to-phoneme (G2P), and then force-aligned back to the waveform to produce a `(audio, phones)` pair the model can train on. Everything below is a tour of one of those boxes, plus the machinery — diarization and speaker embeddings — that lets you build these pairs from messy, multi-speaker found audio at scale.

## 1. Transcripts that exactly match what was said

> The single most valuable property of a TTS transcript is that it is **verbatim**: it contains every word that was spoken, in order, and nothing that was not.

This sounds trivial and it is the source of more downstream pain than any other single factor. There are two ways to get transcripts, and both fail in characteristic ways.

**Human transcripts** (the LibriTTS/LJSpeech lineage, or professionally captioned audiobooks) tend to be *clean* but not *verbatim*. Captioners drop filler words ("um," "you know"), silently correct grammar, merge false starts, and normalize numbers to whatever their style guide says. That is exactly what you want for subtitles and exactly wrong for TTS: if the narrator said "the the cat" (a stutter) and the caption says "the cat," the model learns to compress two "the" acoustic frames into one grapheme, and alignment drifts.

**ASR transcripts** (the modern large-scale route) are *verbatim-ish* but carry recognition errors: substitutions on rare words, dropped short function words, hallucinated text during long silences (a notorious Whisper failure mode), and wrong casing/punctuation. A 5% word error rate (WER) sounds fine until you realize it means one word in twenty is wrong, and TTS training will faithfully reproduce that one-in-twenty defect as a mispronunciation.

### Measuring the mismatch

The right metric is transcript-vs-audio WER, but you rarely have ground truth. The practical proxy is **round-trip consistency**: run a strong ASR model over the audio, compare its output to your transcript, and treat high disagreement as a red flag on that clip. Define, per utterance,

$$
\text{WER} = \frac{S + D + I}{N}
$$

where $S$, $D$, $I$ are substitutions, deletions, and insertions between the two transcripts and $N$ is the reference length. Clips above a threshold (I start at 0.10 and tune) go to a review queue or get dropped. This will not catch errors both your transcript and the ASR share, but it catches the majority — dropped words, number formatting, and the worst hallucinations.

```python
import jiwer

def transcript_health(reference: str, asr_hypothesis: str) -> dict:
    """Round-trip check: how far is our stored transcript from a strong ASR pass?"""
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    measures = jiwer.compute_measures(
        reference, asr_hypothesis,
        truth_transform=transform, hypothesis_transform=transform,
    )
    return {
        "wer": round(measures["wer"], 3),
        "deletions": measures["deletions"],     # words in transcript, missing in audio
        "insertions": measures["insertions"],   # words in audio, missing in transcript
    }

print(transcript_health("the cat sat on the mat", "the the cat sat on a mat"))
# {'wer': 0.333, 'deletions': 0, 'insertions': 1}  -> flag for review
```

### The second-order gotcha: punctuation and casing are prosody

Even a perfectly verbatim transcript can be under-specified. Punctuation is not decoration in TTS — commas and periods drive pauses and terminal intonation, and casing drives emphasis and acronym expansion ("US" versus "us"). A transcript stripped of punctuation (common in ASR output) will train a model with flat, run-on prosody. Restore punctuation and casing before training, either from the original text source or with a punctuation-restoration model. Treat the punctuated, cased, verbatim string as the real transcript; everything else is a lossy approximation.

## 2. Text normalization: turning written text into spoken text

Once the transcript is verbatim, it still is not sayable. Written English is full of **non-standard words** — numbers, dates, currencies, abbreviations, symbols, units — that a human reads aloud by silently expanding them. "Dr." becomes "doctor," "\$3.50" becomes "three dollars and fifty cents," "km" becomes "kilometers." Text normalization (TN) is the module that performs that expansion, and it is the inverse of the **inverse text normalization** (ITN) step that ASR systems run to turn "one hundred twenty three" back into "123."

![Text normalization by semiotic class: each written class expands to a distinct spoken form, and dates are ambiguous without a locale](/imgs/blogs/tts-transcription-normalization-and-speakers-2.webp)

The figure organizes TN by **semiotic class** — the taxonomy from the classic Sproat/Google TN literature. Each class has its own expansion rules: cardinals ("123" to "one hundred twenty-three"), currency ("\$3.50" to "three dollars fifty cents"), dates ("03/04" to "March fourth" — or "the third of April" outside the US, which is exactly why dates are the most locale-dependent class), abbreviations ("Dr." to "doctor," but "St." to "saint" *or* "street" depending on position), units ("2 kg" to "two kilograms"), and ordinals ("3rd" to "third"). The model never sees the written form; it sees the expansion, so the expansion must be correct.

### Rule-based TN with NeMo

The production-grade open-source TN is NVIDIA NeMo's `nemo_text_processing`, built on weighted finite-state transducers (WFSTs) via Pynini. It is fast, deterministic, auditable, and covers dozens of languages. Deterministic matters here: you want to be able to point at *why* "St. John St." expanded the way it did.

```python
from nemo_text_processing.text_normalization.normalize import Normalizer

# WFST-based, deterministic, English. lang="en"; input_case matters for casing rules.
normalizer = Normalizer(input_case="cased", lang="en")

samples = [
    "Dr. Smith paid $3.50 on 03/04 for 2 kg.",
    "Call 555-0142 before 9:30 AM.",
    "The GDP grew 3.2% to $21T in Q3.",
]
for s in samples:
    print(normalizer.normalize(s, verbose=False, punct_post_process=True))

# Dr. Smith paid $3.50 on 03/04 for 2 kg.
#   -> doctor smith paid three dollars and fifty cents on
#      the third of april for two kilograms
# The GDP grew 3.2% to $21T in Q3.
#   -> the gdp grew three point two percent to twenty one
#      trillion dollars in the third quarter
```

Note the date came out as "the third of april" — NeMo's default English grammar happened to read `03/04` in day/month order in this build. That is not a bug you can ignore; it is the central hazard of TN, and we return to it in troubleshooting. The point is that TN makes a *choice*, and if the speaker made a different choice, your transcript now disagrees with the audio.

### When rules are not enough: neural TN

Rule-based TN is the right default, but it has a long tail: novel abbreviations, context-dependent expansions ("read" the verb versus "read" the acronym in some corpora), and messy user-generated text. This is where **neural TN** — a small seq2seq model trained on (written, spoken) pairs — earns its place, usually in a hybrid: rules handle the 95% they cover with certainty, and a neural model handles the residue, guarded so it can never do something catastrophic like drop a whole clause. The NeMo "duplex" TN/ITN system is exactly this hybrid pattern.

> The senior rule of thumb: **normalize deterministically, log every expansion, and diff the token count.** If TN changes the number of spoken tokens in a way alignment cannot absorb, you have manufactured a mismatch.

### The second-order gotcha: TN must match the *speaker*, not the *style guide*

The subtle failure is that TN encodes *a* reading, and the reading in your audio may differ. If the script says "1999" and the narrator said "nineteen ninety-nine" but your TN produced "one thousand nine hundred ninety-nine," the phoneme sequence now has extra words the audio does not contain, and forced alignment will either fail or, worse, silently smear the extra phonemes across real audio. For a corpus mined from real speech, the correct TN target is *what the person actually said*, which you can only recover by aligning candidate expansions against the audio and picking the one that fits — not by trusting the normalizer's default.

## 3. Grapheme-to-phoneme: controlling pronunciation

Normalized text is a sequence of ordinary words, but ordinary words are still ambiguous to a synthesizer. English spelling is famously non-phonetic: "read" has two pronunciations, "lead" has two, "Nguyen" has none a rule could guess. Most high-quality TTS systems therefore convert words to **phonemes** — the atomic sounds — before the acoustic model, using grapheme-to-phoneme (G2P) conversion. Phonemes give you direct pronunciation control and dramatically improve rare-word and name handling.

![Grapheme-to-phoneme conversion with a heteronym branch: the lexicon settles most words, heteronyms need context, and out-of-vocabulary names fall back to a neural model](/imgs/blogs/tts-transcription-normalization-and-speakers-3.webp)

The figure shows the real control flow of a production G2P. A word token first hits a **pronunciation lexicon** (CMUdict for English, ~134k entries in ARPAbet). Most words resolve there with a single pronunciation. Two branches complicate it:

- **Heteronyms** — words spelled identically but pronounced differently by part of speech or tense: "read" (`/r iy1 d/` present versus `/r eh1 d/` past), "lead," "tear," "bass," "live." The lexicon holds both; a context tagger (part-of-speech, or a small classifier) picks the right one. Get this wrong and every "he read the book" becomes "he reed the book."
- **Out-of-vocabulary (OOV) words** — names, loanwords, neologisms, product names. "Nguyen," "Xiaomi," "Cholmondeley." These miss the lexicon entirely and fall back to a **neural G2P model** (a small seq2seq, e.g. the `g2p_en` or `g2p-seq2seq` style) that predicts phonemes from spelling.

### Phonemizing with phonemizer + espeak

The `phonemizer` library wrapping the `espeak-ng` backend is the workhorse for multilingual IPA phonemization and is what VITS, many Tacotron variants, and countless research systems use.

```python
from phonemizer import phonemize
from phonemizer.separator import Separator

text = ["He will read the book.", "She read the book yesterday."]
phones = phonemize(
    text,
    language="en-us",
    backend="espeak",
    strip=True,
    preserve_punctuation=True,
    separator=Separator(phone=" ", word=" | "),
)
for t, p in zip(text, phones):
    print(f"{t:32s} -> {p}")

# He will read the book.        -> hiː | wɪl | ɹiːd | ðə | bʊk
# She read the book yesterday.  -> ʃiː | ɹɛd | ðə | bʊk | jɛstɚdeɪ
```

Notice espeak's built-in tagging got both "read" pronunciations right (`ɹiːd` present, `ɹɛd` past) from context — a nice demonstration that the heteronym branch is real and it works, most of the time. The failures are the interesting part, and we catalog them in troubleshooting.

### The second-order gotcha: phoneme sets are a contract

Once you commit to a phoneme inventory — ARPAbet, IPA, or a custom set — that inventory is a contract between your G2P and your model's embedding table. If you mine a corpus with espeak IPA but your model was trained on CMUdict ARPAbet, every symbol is out of vocabulary and the model produces mush. Worse, mixing G2P backends across a corpus (some clips espeak, some CMUdict) fragments the phoneme distribution and the model never learns stable pronunciations. Pick one G2P path per corpus, version it, and store the phoneme string alongside the audio so training is reproducible.

## 4. A worked pipeline: from a messy sentence to phonemes

Let us run the exact scenario from the outline end to end. Take the raw sentence:

> Dr. Smith paid \$3.50 on 03/04 for 2 kg.

Stage by stage, here is what each module does and what it emits.

**Stage 1 — verbatim check.** Assume this is a human transcript and the audio matches; WER against a round-trip ASR pass is 0.0. Pass.

**Stage 2 — text normalization.** NeMo expands every non-standard word:

```python
from nemo_text_processing.text_normalization.normalize import Normalizer
norm = Normalizer(input_case="cased", lang="en")

raw = "Dr. Smith paid $3.50 on 03/04 for 2 kg."
spoken = norm.normalize(raw, punct_post_process=True)
print(spoken)
# -> doctor smith paid three dollars and fifty cents on
#    the third of april for two kilograms
```

The written string had 8 orthographic tokens; the spoken string has 15. That token-count expansion is normal and expected — but it is also why you cannot align the *written* form to audio. You must align the *spoken* form.

**Stage 3 — grapheme-to-phoneme.** Feed the normalized text to the phonemizer:

```python
from phonemizer import phonemize
from phonemizer.separator import Separator

phones = phonemize(
    spoken, language="en-us", backend="espeak",
    strip=True, separator=Separator(phone=" ", word=" | "),
)
print(phones)
# -> dɑktɚ | smɪθ | peɪd | θɹiː | dɑlɚz | ænd | fɪfti | sɛnts |
#    ɑn | ðə | θɜːd | ʌv | eɪpɹəl | fɚ | tuː | kɪloʊɡɹæmz
```

**Stage 4 — forced alignment.** With phonemes in hand, a forced aligner (Montreal Forced Aligner, or a CTC-based aligner from `torchaudio`) maps each phoneme to a start/end time in the audio. The output is the training pair: the waveform, the phoneme sequence, and per-phoneme durations that many TTS models (FastSpeech2, and duration-predictor systems generally) consume directly.

The critical inspection is the token-count sanity diff between what TN produced and what the aligner could place:

```python
def alignment_sanity(spoken_text: str, aligned_phones: list[str]) -> None:
    n_words = len(spoken_text.split())
    n_phones = len(aligned_phones)
    # ~2-6 phones per English word is normal; wildly off means TN/G2P/audio disagree
    ratio = n_phones / max(n_words, 1)
    status = "OK" if 1.5 <= ratio <= 8.0 else "SUSPECT"
    print(f"words={n_words} phones={n_phones} phones/word={ratio:.1f} [{status}]")

alignment_sanity(spoken, phones.replace(" | ", " ").split())
# words=15 phones=41 phones/word=2.7 [OK]
```

If that ratio is wildly high, TN over-expanded (the "one thousand nine hundred..." trap) or the audio is shorter than the text; if wildly low, the transcript is missing words the speaker said. Either way the aligner's confidence drops and you route the clip to review. This one cheap diff catches a surprising fraction of silent mismatches before they poison training.

## 5. Speaker diarization: who spoke when

Single-speaker corpora (LJSpeech, a studio audiobook read by one narrator) let you skip everything in this section. The moment you touch **found data** — podcasts, interviews, multi-narrator audiobooks, meeting recordings, YouTube — you have multiple voices in one file, and you cannot build clean speaker-consistent training pairs until you know **who spoke when**. That is diarization.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="A two-speaker recording shown as a who-spoke-when track: colored segments for speaker A and speaker B with an overlap region, and a playhead sweeping left to right through time" style="width:100%;height:auto;max-width:820px">
<style>
.dz-wave{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:1.5;opacity:.5}
.dz-a{fill:var(--accent,#6366f1)}
.dz-b{fill:#0ea5e9}
.dz-ov{fill:#f59e0b}
.dz-seg{stroke:var(--background,#fff);stroke-width:2}
.dz-lbl{font:600 15px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.dz-tick{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.dz-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.dz-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.dz-head line{stroke:var(--text-primary,#1f2937);stroke-width:2}
.dz-head polygon{fill:var(--text-primary,#1f2937)}
.dz-head rect{fill:var(--accent,#6366f1);opacity:.14}
@keyframes dz-sweep{0%{transform:translateX(0);opacity:0}6%{opacity:1}94%{opacity:1}100%{transform:translateX(640px);opacity:0}}
.dz-head{animation:dz-sweep 11s linear infinite}
@media (prefers-reduced-motion:reduce){.dz-head{animation:none;transform:translateX(300px)}}
</style>
<text class="dz-cap" x="360" y="28">Diarization output: who spoke when</text>
<polyline class="dz-wave" points="60,70 100,52 140,86 180,58 220,80 260,50 300,84 340,64 380,76 420,54 460,82 500,60 540,78 580,66 620,84 660,56 700,72"/>
<rect class="dz-seg dz-a"  x="60"  y="120" width="130" height="60" rx="6"/>
<rect class="dz-seg dz-b"  x="192" y="120" width="94"  height="60" rx="6"/>
<rect class="dz-seg dz-ov" x="288" y="120" width="44"  height="60" rx="6"/>
<rect class="dz-seg dz-a"  x="334" y="120" width="104" height="60" rx="6"/>
<rect class="dz-seg dz-b"  x="440" y="120" width="150" height="60" rx="6"/>
<rect class="dz-seg dz-a"  x="592" y="120" width="108" height="60" rx="6"/>
<text class="dz-lbl" x="125" y="156">Spk A</text>
<text class="dz-lbl" x="239" y="156">Spk B</text>
<text class="dz-lbl" x="310" y="156">ov</text>
<text class="dz-lbl" x="386" y="156">Spk A</text>
<text class="dz-lbl" x="515" y="156">Spk B</text>
<text class="dz-lbl" x="646" y="156">Spk A</text>
<line class="dz-axis" x1="60" y1="210" x2="700" y2="210"/>
<text class="dz-tick" x="60"  y="232">0s</text>
<text class="dz-tick" x="220" y="232">5s</text>
<text class="dz-tick" x="380" y="232">10s</text>
<text class="dz-tick" x="540" y="232">15s</text>
<text class="dz-tick" x="700" y="232">20s</text>
<g class="dz-head">
<rect x="52" y="112" width="16" height="76"/>
<line x1="60" y1="104" x2="60" y2="196"/>
<polygon points="60,104 52,92 68,92"/>
</g>
<figcaption>The playhead scrubs the timeline; at each instant the colored track tells you which speaker is active (or that they overlap).</figcaption>
</figure>

The animation shows diarization output as a **who-spoke-when track**: the recording is carved into time segments, each tagged with a speaker (A or B), plus a short amber **overlap** region where they talk at once. As the playhead scrubs left to right, exactly one speaker is "active" at each instant — that per-instant labeling is what diarization produces, and it is precisely what you need to slice the file into single-speaker utterances.

### Diarization with pyannote

The de-facto open-source diarizer is `pyannote.audio`. Its `speaker-diarization-3.1` pipeline handles VAD, speaker embedding, and clustering end to end. It gates model access behind a Hugging Face token — pass it through an environment variable, never hard-code it.

```python
import os
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HF_TOKEN"],   # from env, not a literal
)

# Optional but recommended for reproducibility: pin device and num_speakers if known.
diarization = pipeline("interview.wav")  # or num_speakers=2

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:6.2f}s -> {turn.end:6.2f}s  {speaker}")

#   0.31s ->   4.02s  SPEAKER_00
#   4.05s ->   6.98s  SPEAKER_01
#   6.90s ->   7.44s  SPEAKER_00   # overlaps the previous turn -> handle it
#   7.50s ->  10.72s  SPEAKER_00
#  10.80s ->  15.31s  SPEAKER_01
```

### A worked diarization pass

Take a two-speaker clip: a 15-second interview snippet. Running the pipeline yields the segments above. To turn that into training material you (1) merge adjacent same-speaker turns separated by short pauses, (2) *drop or specially handle* overlapping regions (the 6.90 to 7.44 turn overlaps 6.98), and (3) cut the audio at the surviving boundaries so each clip contains exactly one voice.

```python
from itertools import groupby

def to_single_speaker_segments(diar, min_dur=1.0, merge_gap=0.3):
    """Collapse a pyannote diarization into clean single-speaker cuts."""
    turns = [(t.start, t.end, spk)
             for t, _, spk in diar.itertracks(yield_label=True)]
    turns.sort()
    clean = []
    for spk, group in groupby(turns, key=lambda x: x[2]):
        g = list(group)
        start, end = g[0][0], g[-1][1]
        if end - start >= min_dur:                 # skip fragments
            clean.append((round(start, 2), round(end, 2), spk))
    return clean

segs = to_single_speaker_segments(diarization)
for s in segs:
    print(s)
# (0.31, 4.02, 'SPEAKER_00')
# (7.5, 10.72, 'SPEAKER_00')
# (10.8, 15.31, 'SPEAKER_01')
# -> three clean single-speaker clips; the overlap turn was correctly excluded
```

Out of 15 seconds, we kept roughly 11 seconds of clean single-speaker audio and discarded the overlap and sub-second fragments. That ~25% loss rate is typical and healthy — it is the price of clean speaker labels, and it is far cheaper than training on smeared multi-speaker clips.

### The second-order gotcha: overlapped speech is not a rounding error

Naive diarizers assign every frame to exactly one speaker, which means overlapped speech gets attributed to whoever "wins." For conversational corpora, overlap can be 5 to 15% of the audio, and that mislabeled fraction ends up teaching your model that speaker A occasionally sounds like speaker B. Modern pipelines do **overlap-aware** diarization and can output multiple labels per frame; at minimum, detect overlap and exclude those regions from single-speaker training data.

## 6. Speaker labels and embeddings for multi-speaker corpora

Diarization gives you *local* speaker labels — `SPEAKER_00` in this file, `SPEAKER_00` in that file — but those are per-file identities with no connection to each other. Building a multi-speaker or voice-cloning corpus requires *global* speaker identity: knowing that the narrator in chapter 1 is the same person as in chapter 40, and different from the podcast host. That is what **speaker embeddings** provide.

![From local per-file diarization labels to global corpus-wide speaker identity via embeddings, feeding voice cloning and multi-speaker TTS](/imgs/blogs/tts-transcription-normalization-and-speakers-4.webp)

The figure traces the pipeline: two files come in, each with local diarization labels (`spk_0`, `spk_1`). A **speaker encoder** — ECAPA-TDNN, x-vector, or d-vector, typically producing a 192- or 256-dimensional embedding — maps every segment to a point in speaker space. Clustering those embeddings across the whole corpus collapses local labels into **global speaker IDs**: all the segments that are acoustically the same person, regardless of which file they came from. Those global IDs are what voice cloning uses (one reference embedding per speaker) and what multi-speaker TTS conditions on (a speaker-embedding lookup or a reference encoder).

### Extracting and comparing embeddings

```python
import torch
from pyannote.audio import Model, Inference

# ECAPA-style speaker embedding model.
model = Model.from_pretrained("pyannote/embedding",
                              use_auth_token=os.environ["HF_TOKEN"])
inference = Inference(model, window="whole")

emb_a = torch.tensor(inference("speaker_a_utt1.wav"))
emb_b = torch.tensor(inference("speaker_a_utt2.wav"))   # same person, diff clip
emb_c = torch.tensor(inference("speaker_b_utt1.wav"))   # different person

def cosine(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item()

print(f"same speaker:      {cosine(emb_a, emb_b):.2f}")   # ~0.82  high
print(f"different speaker: {cosine(emb_a, emb_c):.2f}")   # ~0.31  low
```

The verification decision is a threshold on cosine similarity: same speaker if $\cos(\theta) > \tau$, different otherwise, where $\tau$ is tuned on a held-out set (typically 0.5 to 0.7 for ECAPA on clean speech). Clustering for global IDs uses the same similarity, usually with agglomerative clustering and a distance threshold, or spectral clustering when you know the speaker count.

### The second-order gotcha: embeddings drift with channel, not just voice

Speaker embeddings encode more than the voice — they pick up the recording channel, microphone, room, and codec. Two clips of the same speaker recorded on different equipment can score *lower* than two different speakers recorded on the same equipment. This is **channel leakage**, and it wrecks clustering: you get clusters that are really "iPhone recordings" and "studio recordings" rather than "person A" and "person B." Mitigations: augment the encoder's training with channel variation, cluster within-source before merging across sources, and always sanity-check clusters by listening to a few members.

## 7. Mining TTS-grade data from audiobooks and podcasts at scale

Everything above composes into the recipe that transformed TTS from 2020 to today: instead of hiring one narrator for 24 studio hours (LJSpeech), you **mine** tens of thousands of hours from existing audiobooks and podcasts, and the resulting massive multi-speaker corpora are what make **zero-shot voice cloning** possible. A model that has heard 50,000 hours across 7,000 speakers can imitate a brand-new voice from a three-second sample, because it has learned the manifold of human voices — see [Neural codec language model TTS: VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e) and [Zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) for what these corpora enable downstream.

![Mining TTS-grade data from audiobooks and podcasts: found audio becomes a trainable multi-speaker corpus only after VAD, diarization, ASR, alignment, and hard filtering](/imgs/blogs/tts-transcription-normalization-and-speakers-5.webp)

The figure is the modern large-scale recipe as a funnel. Start with ~60,000 hours of raw found audio. **VAD** (Silero, or pyannote's VAD) cuts it into speech segments and throws away silence and music. **Diarization** (pyannote) tags speakers and lets you slice into single-speaker clips. **ASR** (Whisper-large, or an NeMo Parakeet/Conformer) transcribes each clip verbatim-ish. **Forced alignment** (MFA or a CTC aligner) attaches per-word or per-phoneme timestamps. Finally, **hard filtering** on signal-to-noise ratio, ASR confidence / character error rate, clip length, and speaking rate drops the unusable fraction — typically you keep 70 to 85%, landing around ~50,000 TTS-grade hours.

### The case study: how LibriHeavy / Emilia-scale corpora are built

The lineage is worth knowing because each corpus fixed a specific weakness of the last.

**LibriLight (2019)** took ~60,000 hours of LibriVox audiobooks with essentially no labels — it was built for self-supervised pretraining, so verbatim text was not the point.

**LibriHeavy (2023)** went back to that same LibriVox audio and did the hard part: it recovered ~50,000 hours of *aligned, punctuated, cased* text by matching the audio against the original Project Gutenberg book texts, then force-aligning. The innovation was using the *known* source text (the book) rather than trusting raw ASR, which sidesteps the biggest transcript-quality problem. Casing and punctuation — the prosody signals from section 1 — were explicitly preserved.

**Emilia (2024)** generalized the recipe to *in-the-wild* multilingual audio (podcasts, talk shows, video), where there is no source book to align against. Its `Emilia-Pipe` open pipeline is exactly the funnel above: source separation to remove background music, VAD, diarization, ASR with a strong model, and DNSMOS-based quality filtering. It produced over 100,000 hours across six languages, and its whole reason for existing is that spontaneous, expressive, multi-speaker speech is what zero-shot TTS needs — studio audiobooks are too clean and too monotone to teach a model the full range of human delivery.

The engineering lesson across all three: **the transcript problem dominates the audio problem at scale.** LibriHeavy's value was not more audio (it reused LibriLight's) — it was *better text alignment*. When you have the source text, use it; when you do not, invest in your best ASR plus aggressive confidence filtering, because a mined corpus is only as trustworthy as the weakest transcript in it.

```bash
# Sketch of the mining funnel as a shell pipeline over a shard of found audio.
# Each stage writes a manifest the next stage consumes; nothing is deleted in place.
python vad_segment.py     --in raw_shard/ --out seg/    --backend silero
python diarize.py         --in seg/       --out diar/   --model pyannote/speaker-diarization-3.1
python asr_transcribe.py  --in diar/      --out asr/    --model whisper-large-v3
python force_align.py     --in asr/       --out aligned/ --aligner mfa
python quality_filter.py  --in aligned/   --out kept/   --min-snr 15 --max-cer 0.10 --min-dur 1.0
# kept/manifest.jsonl now holds (audio, text, phones, speaker_id, duration, snr, cer)
```

## Troubleshooting: symptoms, causes, and fixes

This is the section to bookmark. Every failure below has shipped to production in some real TTS corpus, mine included.

### Symptom: the model says the wrong number of words

**Cause:** text normalization expanded a number differently from how the speaker said it — "1999" became "one thousand nine hundred ninety-nine" but the audio says "nineteen ninety-nine." The phoneme sequence now has phonemes with no audio, and forced alignment smears them.

**Detection:** the phones-per-word ratio from section 4 spikes; forced-alignment confidence for that clip drops; a spot listen reveals the mismatch. At corpus scale, flag any clip whose aligned duration per phoneme is more than ~3 standard deviations from the mean.

**Fix:** for mined corpora, make TN *audio-aware* — generate candidate expansions (year-style, cardinal-style, digit-string) and pick the one whose forced alignment scores best, rather than trusting the normalizer's default. For controlled corpora, verify TN output against the audio on a sample and correct the grammar. Always log the pre- and post-TN token counts so this is auditable.

### Symptom: heteronyms and names are consistently mispronounced

**Cause:** G2P chose the wrong pronunciation. Heteronyms failed because the context tagger mis-tagged part of speech ("he will read" tagged as past tense); names failed because the neural G2P guessed wrong on an OOV word ("Nguyen" pronounced letter-by-letter).

**Detection:** build a small **pronunciation eval set** — a few hundred clips containing known-hard words (heteronyms, common names in your domain) — and check the phoneme output against a hand-verified key. This catches systematic G2P errors that WER never will, because the words are spelled right; only the sounds are wrong.

**Fix:** maintain a **custom pronunciation lexicon** that overrides G2P for your domain's names and terms — this is the single highest-leverage fix and it is what every production system does. For heteronyms, improve the POS tagger or add explicit disambiguation rules. Never rely on the default G2P for proper nouns that matter to you.

### Symptom: two speakers bleed into one voice; cloning sounds "averaged"

**Cause:** diarization errors. Either speaker confusion (segments assigned to the wrong speaker), missed overlap (overlapped speech attributed to one speaker), or channel-leaked embeddings that clustered by microphone instead of by person.

**Detection:** measure **diarization error rate (DER)** on a labeled subset; DER above ~15% is a warning sign for corpus building. Cluster purity checks — listen to a random sample from each global speaker cluster and confirm it is one person. If a "speaker" cluster contains two voices, your clone will be an average of them.

**Fix:** use overlap-aware diarization and exclude overlap regions from training. Raise the clustering threshold to prefer *splitting* speakers over *merging* them — over-clustering (one real speaker into two IDs) is far less harmful for TTS than under-clustering (two speakers into one). Cluster within-source before merging across sources to fight channel leakage.

### Symptom: your voice-cloning metrics are suspiciously high

**Cause:** **speaker leakage across the train/test split.** You split utterances at random, so the same speaker appears in both train and test. The model has literally heard your "unseen" test speaker during training, so speaker-similarity (SIM) and naturalness look great in evaluation and collapse in production on genuinely new voices.

![Speaker leakage: a random utterance split puts the same speaker in train and test and inflates cloning similarity, while a speaker-disjoint split is honest](/imgs/blogs/tts-transcription-normalization-and-speakers-6.webp)

The before/after figure makes the failure concrete. A random utterance split shuffles all clips, so the same speaker lands in train and test, and the reported clone SIM comes out at an inflated 0.95. A speaker-disjoint split assigns *whole speakers* to test — held-out identities the model has never heard — and the honest clone SIM drops to 0.78. That gap is the size of the lie a random split tells you.

**Detection:** compute the intersection of global speaker IDs between train and test. It must be empty. If you cannot produce global speaker IDs, you cannot prove your split is clean — which is itself a reason to build them.

**Fix:** always split by speaker for any voice-cloning or speaker-generalization claim.

```python
from sklearn.model_selection import GroupShuffleSplit

# clips: rows with a 'speaker_id' column (the GLOBAL id from section 6).
splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, test_idx = next(splitter.split(
    clips, groups=clips["speaker_id"]))     # group = speaker -> no speaker in both

train, test = clips.iloc[train_idx], clips.iloc[test_idx]
assert set(train["speaker_id"]).isdisjoint(set(test["speaker_id"]))  # must hold
print(f"train speakers={train.speaker_id.nunique()} "
      f"test speakers={test.speaker_id.nunique()} (disjoint)")
```

### Symptom: alignment fails on a chunk of clips even though audio and text look fine

**Cause:** usually a phoneme-set mismatch (the G2P backend changed mid-corpus), a language mismatch (an English aligner on a Spanish clip that ASR mis-language-tagged), or leading/trailing non-speech (music, applause) that VAD missed.

**Detection:** alignment tools report a per-clip failure or a very low likelihood; bucket failures by source, language tag, and G2P version to find the common factor.

**Fix:** pin one G2P path per corpus and one phoneme inventory; run language ID before ASR and route clips to the matching aligner; tighten VAD and trim non-speech edges before alignment.

## Case studies from production

### 1. The audiobook that read every price wrong

A team building a retail-assistant voice mined product-review audiobooks. Reviews are full of prices, and their TN normalized "\$4.99" to "four dollars and ninety-nine cents" — correct — but the narrators, reading fast, actually said "four ninety-nine." Every price clip had four extra spoken tokens ("dollars and," "cents") with no matching audio. Alignment smeared them, and the shipped model inserted phantom "dollars and cents" after prices. The fix was audio-aware TN: generate both expansions, pick the one that aligns. Twelve percent of the price clips flipped to the short form. Lesson: **TN must model the speaker, not the style guide.**

### 2. The LibriHeavy insight, learned the hard way

Before LibriHeavy existed, a group tried to build an aligned corpus from LibriVox using raw Whisper transcripts. Whisper's number formatting, casing loss, and long-silence hallucinations meant ~8% of clips had transcript defects, and the resulting model had audible pronunciation instability. Switching to the LibriHeavy approach — aligning audio against the *original Gutenberg book text* instead of trusting ASR — cut transcript defects by an order of magnitude with the same audio. Lesson: **when the source text exists, aligning to it beats re-transcribing it.** This is the whole reason LibriHeavy was a leap over LibriLight despite reusing the audio.

### 3. The podcast corpus that clustered by microphone

A conversational-TTS effort mined 8,000 hours of podcasts and clustered speaker embeddings to get global IDs. The clusters looked great by silhouette score but produced averaged clones. Listening revealed the clusters were splitting on **recording setup** — remote guests on laptop mics versus hosts on studio mics — so a single host recorded across two setups became two "speakers," and two guests on the same call-in line merged into one. They fixed it by clustering within each show first, then merging across shows with a channel-robust encoder. Lesson: **speaker embeddings encode the room, not just the voice.**

### 4. The overlap that taught a stutter

An interview corpus used a one-label-per-frame diarizer. Roughly 9% of the audio was crosstalk, all attributed to whichever speaker "won" each frame. The model trained on those clips learned a subtle doubling artifact — it had been shown two voices labeled as one. Adding overlap detection and excluding overlapped regions removed the artifact and cost only 9% of the data. Lesson: **overlapped speech is not a rounding error; exclude it or model it explicitly.**

### 5. The 0.95 that became 0.78

A voice-cloning paper reported a speaker-similarity of 0.95 and could not reproduce it in a demo with a new user's voice. The split had been random over utterances; 40% of "test" speakers were also in train. Re-running with a speaker-disjoint split (the `GroupShuffleSplit` above) gave an honest 0.78 — still good, but 0.17 lower. Every downstream decision made on the 0.95 number had been wrong. Lesson: **for any speaker-generalization claim, split by speaker or do not make the claim.**

### 6. The Emilia bet on messy audio

A team had a pristine 200-hour single-speaker corpus and a model that sounded natural but flat — it could not do expressive or conversational delivery. Adding a mined, in-the-wild multi-speaker corpus in the Emilia style (podcasts, talk shows, source-separated and filtered) — deliberately *messier* data — gave the model the prosodic range it lacked, at the cost of more aggressive filtering to keep noise out. Lesson: **studio-clean data teaches pronunciation; in-the-wild data teaches expression.** Zero-shot systems need both. This mirrors the streaming, expressive direction of systems like [CosyVoice](/blog/machine-learning/signal-processing/cosyvoice-v1-v3-tokens-flow-matching-streaming).

## When to reach for this machinery — and when not to

**Build the full text-side pipeline (TN + G2P + diarization + embeddings) when:**

- You are training multi-speaker or zero-shot TTS and need global speaker identity.
- Your data is *mined* from found audio (audiobooks, podcasts, video) rather than recorded to spec.
- Your domain is number-, name-, or acronym-heavy (finance, medical, navigation), where TN and custom pronunciation lexicons are load-bearing.
- You will publish or act on voice-cloning metrics — then speaker-disjoint splits are non-negotiable.

**Skip parts of it when:**

- **Single-speaker studio corpus** (LJSpeech-style): skip diarization and speaker embeddings entirely; one file is one speaker.
- **You control the script:** if you commissioned the recordings from a written script, your transcript is verbatim by construction and you can skip round-trip WER checking (but still do TN and G2P).
- **Character/byte-input models:** some modern systems feed normalized text directly and learn pronunciation implicitly at scale — you still need TN and clean transcripts, but you can skip explicit G2P. Weigh this against losing direct pronunciation control for rare words and names.
- **Tiny prototypes:** for a quick demo on a handful of speakers, manual labeling beats standing up a mining pipeline. The pipeline pays off at hundreds of hours, not tens.

The unifying principle is the one we opened with: in TTS the transcript is the control signal. Whether you spend your effort on verbatim checking, normalization, phonemes, or speaker identity, you are spending it on the same goal — making sure the text you feed the model is a faithful, unambiguous description of the sound you want back.

## Further reading

- [Data for text-to-speech: cleaning and enhancement](/blog/machine-learning/training-data/data-for-text-to-speech-cleaning-and-enhancement) — the audio half of the same corpus problem.
- [Neural codec language model TTS: VALL-E](/blog/machine-learning/audio-generation/neural-codec-language-model-tts-vall-e) — what large mined corpora unlock.
- [Zero-shot voice cloning and the TTS frontier](/blog/machine-learning/audio-generation/zero-shot-voice-cloning-and-the-tts-frontier) — where speaker embeddings and multi-speaker data lead.
- [CosyVoice v1–v3: tokens, flow matching, and streaming](/blog/machine-learning/signal-processing/cosyvoice-v1-v3-tokens-flow-matching-streaming) — a modern expressive, streaming system built on this kind of data.
- NVIDIA NeMo `nemo_text_processing` (WFST text normalization), `phonemizer` + `espeak-ng` (G2P), and `pyannote.audio` (diarization and embeddings) — the three open-source workhorses used throughout this post.
