---
title: "Debugging ASR Finetuning: Whisper, wav2vec2, and the WER That Lies"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Find the silent failures that ruin a speech-recognition finetune — the wrong language token, the SpecAugment that underfits, the processor that doesn't match the checkpoint — and learn why your WER is often measuring your scorer, not your model."
tags:
  [
    "debugging",
    "model-training",
    "speech",
    "asr",
    "whisper",
    "wav2vec2",
    "finetuning",
    "pytorch",
    "evaluation",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/debugging-asr-finetuning-1.png"
---

A team I helped had finetuned `whisper-small` on about 40 hours of customer-support phone calls. The training loss fell cleanly, the run finished overnight, and the evaluation script printed a word error rate of 61%. Sixty-one percent. For context, the base model — the one they were trying to *improve* — scored about 14% on the same audio out of the box. They had spent two weeks and a few hundred dollars of GPU time making a state-of-the-art model four times worse. Someone proposed throwing away the dataset. Someone else proposed a different architecture. Both were wrong, and both would have wasted another two weeks.

When I sat down and printed five transcriptions next to their references, the model was nearly perfect. It heard "I'd like to check on order number five eight two" and wrote exactly that. The reference said "I would like to check on order #582." Every single difference between prediction and reference was a contraction, a number written as a word, a piece of punctuation, or a capital letter — and the scorer was counting every one of them as a word error. The model was fine. The *scorer* was broken. After we ran the predictions through a standard text normalizer before computing WER, the same checkpoint scored 8%. Nothing about the model changed. The WER had been lying.

![Side-by-side figure showing raw word error rate of 60 percent collapsing to a true 8 percent on identical predictions after lowercasing, stripping punctuation, and standardizing number words](/imgs/blogs/debugging-asr-finetuning-1.png)

This is the defining trap of speech recognition, and it is the reason ASR finetuning fails differently from every other modality in this series. In a vision or tabular run, a wrong number on the dashboard usually means the model is wrong. In ASR, a wrong number routinely means the *measurement* is wrong — and even when the model genuinely is broken, the cause is almost never the place beginners look. It is a feature extractor that doesn't match the checkpoint, a single wrong special token in Whisper's decoder prompt, a SpecAugment policy that masks so much of the spectrogram the model can't learn, a loss that trains on the prompt instead of masking it, or a long-audio inference path that silently truncates everything past 30 seconds. By the end of this post you will be able to take any stalled or "got worse after finetuning" ASR run — Whisper, wav2vec2, or a Conformer-style CTC model — and localize the bug in minutes: decide whether the problem is the scorer, the prompt, the features, or the acoustic model itself, confirm it with one print statement, and fix it. You will have a runnable Hugging Face processor setup, a WER harness that uses a real normalizer, a SpecAugment audit, and a decoder-prompt inspector.

This is one post in a series whose spine is simple: a training bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and you *bisect* to the right one before touching code. ASR is unusual because it concentrates the danger in **evaluation** (the WER that lies) and **model code / data** (the processor and prompt contract) rather than in raw optimization. If you have not seen the master map, start with [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the [capstone playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) ties every track together. We will also lean on a few siblings: the [audio input bugs](/blog/machine-learning/debugging-training/audio-input-bugs) post for sample-rate and mel-spectrogram mismatches that feed straight into this one, the [CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) post for the wav2vec2 loss path, and [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) for the general version of the scorer problem.

## 1. Word error rate is not what you think it is

Before we debug anything, we have to be precise about the instrument, because half the bugs in this post are *about* the instrument. Word error rate is defined as the minimum number of single-word edits — substitutions, insertions, and deletions — needed to turn the hypothesis (the model's output) into the reference (the ground truth), divided by the number of words in the reference. Formally, if $S$ is the number of substitutions, $D$ deletions, $I$ insertions, and $N$ the number of words in the reference:

$$\text{WER} = \frac{S + D + I}{N}$$

This comes straight from the Levenshtein edit distance computed at the word level. It is a perfectly good metric. The problem is the words. WER treats two strings as a sequence of tokens split on whitespace, and it compares them *exactly*. The string "582" is not the same token as "five hundred eighty two". "don't" is not the same token as "do not". "Hello." with a period is not the same token as "hello" without one. "OK" and "okay" are different. A trailing space, a double space, a non-breaking space — all of these change the tokenization and therefore the edit distance.

Here is the part that makes WER uniquely treacherous as a debugging signal: these surface differences are *not small*. They are not a fraction of a percent of noise on top of a meaningful number. In conversational speech, contractions, numbers, and punctuation are everywhere. A reference transcript written by a human who follows ordinary writing conventions ("I'll call you at 3:30 about the \$40 charge") will differ from a faithful, correct ASR output ("I will call you at three thirty about the forty dollar charge" or "i'll call you at 3 30 about the 40 charge") in nearly every clause. If you score raw, those differences can dominate the error count. This is why a model that a human would call excellent can post a 60% WER, and why the very first thing you do when a WER looks wrong is *not* touch the model. You look at the transcriptions.

### Why normalization dominates the number

Let us make the magnitude concrete, because "normalization matters" is the kind of sentence people nod at and then ignore. Take a single reference sentence with 12 words: "I would like to check on order number five eight two please." Suppose the model outputs, correctly by ear, "I'd like to check on order #582 please" — 8 whitespace tokens. Now run the edit distance. "I would" became "I'd" — that's a substitution and a deletion if you align greedily, or two edits. "number five eight two" (four tokens) became "#582" (one token) — that's at minimum one substitution and three deletions. Already you have on the order of 5–6 edits against a 12-word reference: a WER north of 40% on a transcription that is *phonetically perfect*. The model heard every word. The scorer punished it for spelling conventions.

Now normalize both sides first: lowercase, strip punctuation, expand contractions, convert digits to a canonical form (either spell numbers out or collapse number words to digits — the point is to make both sides use the *same* convention), and squash whitespace. After normalization both strings become, say, "i would like to check on order number five eight two please" on both sides. Edit distance: zero. WER: 0%. The model didn't change. The measurement became honest.

The lesson, which we will return to repeatedly, is structural: **WER is the composition of two functions — a model and a normalizer — and a bug in either one moves the number by tens of points.** When you see a surprising WER, you are debugging a composition, and you must isolate which half is responsible before you spend a single GPU-hour. This is the speech-specific instance of a general rule from [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying): a metric is code, code has bugs, and a metric bug looks exactly like a model bug on the dashboard.

#### Worked example: the 61% that was 8%

The support-call run from the intro is the canonical case, so let us put real numbers on it. Their eval set was 1,200 utterances, average reference length 11 words, so roughly $N = 13{,}200$ reference words total. Scored raw, the harness reported 61% WER, which means about 8,050 edits. We pulled the edit breakdown — most WER libraries can return $S$, $D$, $I$ separately — and it was overwhelmingly substitutions and insertions concentrated on three patterns: digits written as words by the model vs digits-as-numerals in the reference (about 31% of all edits), contraction expansion (about 24%), and punctuation tokens like commas and periods that the reference had and the model omitted (about 28%). That is 83% of the "errors" living in three categories that have nothing to do with whether the model heard the words.

We applied the Whisper text normalizer (which lowercases, removes punctuation, and has a number-spelling normalization) to both hypotheses and references and re-scored. The same predictions came back at 8.2% WER — about 1,080 genuine edits, and when we eyeballed those they were real: a few homophone confusions, a couple of proper nouns the model didn't know, the occasional dropped filler word. The model was, in fact, a solid improvement over the 14% base model. The "fix" was four lines of code in the scorer and zero lines in the model. Two weeks of proposed model-architecture changes evaporated because someone finally printed the transcriptions.

The reason this is worth dwelling on is that it is the *most common* ASR debugging mistake by a wide margin, and it is invisible if you only look at the loss curve. Cross-entropy or CTC loss is computed on the model's *tokens*, which already include the model's spelling and casing conventions — the loss can be perfectly healthy while the WER scorer, comparing against differently-formatted references, screams. Loss and WER disagreeing is the signature. We will build the harness that catches it in Section 8.

### Two normalization subtleties that bite even careful teams

Even teams that know to normalize get tripped up by two specifics. The first is **over-normalization that hides real errors.** If your normalizer collapses everything aggressively — stripping not just punctuation but also, say, mapping every digit string to a single token, or removing all whitespace — you can make two genuinely different transcriptions score as identical and report a WER that is *too low.* A scorer can lie in both directions. The discipline is to use a *standard, published* normalizer (the Whisper normalizer, or a well-understood `jiwer` transform chain) rather than hand-rolling one, because a hand-rolled normalizer is itself untested code in the most load-bearing position in your evaluation. When the normalizer is custom, you are now debugging a third function in the composition, and the bug can hide there.

The second subtlety is the **empty-reference and empty-hypothesis edge case.** If a reference normalizes to an empty string (it was pure punctuation, or all of it got stripped), most WER libraries either error out or divide by zero. If a hypothesis is empty but the reference is not, the WER for that utterance is 100% (all deletions), and a handful of those — say, the model emitting nothing on the few clips where it choked — can swing your aggregate WER by several points even though the bulk of the corpus is fine. The fix is to handle these cases explicitly (drop empty-reference pairs, and *log* empty hypotheses separately so you notice them) rather than letting the library silently mishandle them. This matters because an aggregate WER is a mean, and a mean is dominated by its outliers; a clean way to debug is to compute *per-utterance* WER and look at the distribution, not just the headline mean. A bimodal distribution — most utterances near 5%, a cluster at 100% — tells you the average is hiding a specific failure (empty outputs, a wrong language on a subset, clips past the 30-second window) rather than uniform mediocrity.

The deeper structural point, again, is that **WER is a summary statistic, and summary statistics hide structure.** The mean WER over 1,200 utterances can be 60% because every utterance is 60% wrong, or because 90% of utterances are 5% wrong and 10% are catastrophic. Those are completely different bugs with completely different fixes, and the headline number can't tell them apart. Whenever a WER surprises you, the second thing you do (after eyeballing transcripts) is plot the per-utterance distribution. We will instrument exactly that in Section 8.

## 2. The Whisper decoder prompt: a stack of special tokens

If the WER is honest and the model is genuinely producing wrong text, the next most common cause for Whisper specifically is the decoder prompt. Whisper is an encoder-decoder model, and unlike a from-scratch seq2seq model you might write, its decoder does not start from a blank slate. It starts from a fixed sequence of special tokens that *tell it what task to do*. Get one of those tokens wrong and the model will confidently do the wrong task — transcribe in the wrong language, translate instead of transcribe, or emit timestamps you didn't ask for — while the loss looks completely normal because, during training, the model was perfectly happy predicting the next token given whatever prefix you handed it.

![Vertical stack figure showing the Whisper decoder prompt as start-of-transcript, language, task, and timestamp tokens above the text tokens and end-of-text, with the language token flagged as a common failure](/imgs/blogs/debugging-asr-finetuning-2.png)

The standard Whisper decoder prompt for English transcription without timestamps is, in order:

```bash
<|startoftranscript|>   # always first; marks the beginning
<|en|>                  # the LANGUAGE token (one per supported language)
<|transcribe|>          # the TASK token: transcribe (vs <|translate|>)
<|notimestamps|>        # suppress timestamp tokens (vs emitting them)
... actual transcription tokens ...
<|endoftext|>           # EOS; the model must learn to emit this to stop
```

These are not decorative. The language token genuinely switches the model's behavior. If you finetune English data but the decoder prompt carries a `<|fr|>` token — easy to do if you copied a multilingual config, or if the processor's default language is set wrong — the model will try to transcribe English audio under a French-language prior and produce fluent-looking garbage. The task token is the difference between writing down what was said (`<|transcribe|>`) and translating it to English (`<|translate|>`); a translate token on a transcription task is a disaster that looks like the model "paraphrasing." The timestamps token controls whether the model emits `<|0.00|>`-style time markers interleaved with text; if your training data has no timestamps but the prompt says to emit them, or vice versa, you get either spurious timestamp tokens polluting the output or a model that never learned to align.

### Why a wrong prompt token survives training silently

Here is the mechanism, and it is worth understanding because it explains why this bug is so quiet. During finetuning, Whisper is trained with teacher forcing: at each position the model is given the *correct* previous tokens (including the special prompt) and asked to predict the next token, and the loss is cross-entropy over that prediction. If your prompt is internally consistent — every training example uses the same `<|fr|>` token, and the labels are masked correctly — the model learns a perfectly self-consistent mapping: "given this French-flavored prefix and this audio, predict these English words." The training loss goes down. The model is *learning*. It is just learning under the wrong conditioning, and at inference, if you (or the generation defaults) use a *different* prompt than training used, or if the prompt itself was semantically wrong, the output degrades.

There are actually two distinct failure modes hiding here, and they have opposite fixes:

1. **Train/infer prompt skew.** You trained with one prompt and generate with another. The model is fine; the inference config is wrong. Fix the generation call to match training.
2. **A semantically wrong prompt used consistently in both.** You both trained and infer with `<|fr|>` on English. The model learned a degraded mapping. Fix the prompt everywhere and re-finetune (or at least re-evaluate; sometimes the encoder is robust enough that just fixing inference recovers most of it).

You distinguish them the same way you distinguish everything in this series: by reading the instrument. Print the exact decoder input IDs.

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# What forced-decoder-ids does the processor produce for your config?
forced = processor.get_decoder_prompt_ids(language="en", task="transcribe")
print("forced_decoder_ids:", forced)

# Decode each id back to its token string so a human can read it
tok = processor.tokenizer
for position, token_id in forced:
    print(f"  pos {position}: id={token_id}  token={tok.convert_ids_to_tokens(token_id)!r}")

# Sanity-check the special tokens you THINK you're using
for name in ["<|startoftranscript|>", "<|en|>", "<|transcribe|>",
             "<|notimestamps|>", "<|endoftext|>"]:
    print(f"{name}: id={tok.convert_tokens_to_ids(name)}")
```

If that printout shows `<|fr|>` where you expected `<|en|>`, you have found your bug without launching a single training step. This is the make-it-fail-small discipline applied to configuration: the smallest possible reproducer is *one print statement*, and it costs nothing.

#### Worked example: the model that "paraphrased"

A client reported that their finetuned Whisper "paraphrased" — it produced grammatically clean English that captured the *gist* of the audio but rarely the exact words, scoring around 38% WER even after normalization. Paraphrasing is not a thing transcription models do; it was a tell. We printed the forced decoder IDs and found `<|translate|>` instead of `<|transcribe|>`. Their audio was already English, so "translate English to English" became, in effect, "produce a fluent English rendering" — which is exactly what paraphrasing looks like. The translate token had crept in because they had copied a generation config from a multilingual translation demo. We set `task="transcribe"` in both the data-prep prompt and the generation call, re-ran evaluation on the existing checkpoint (no retraining needed, because the encoder representations were unchanged and the decoder could follow the corrected prompt), and WER dropped from 38% to 9%. The instrument was a print statement; the fix was one keyword argument.

## 3. The feature extractor and tokenizer must match the checkpoint

The third place ASR finetunes break is the processor — the pairing of feature extractor (audio → log-mel spectrogram for Whisper, or raw-waveform normalization for wav2vec2) and tokenizer (text → token IDs). In Hugging Face these are bundled in a `WhisperProcessor` or `Wav2Vec2Processor`, and the cardinal rule is that **the processor must come from the same checkpoint as the model.** This sounds obvious and is violated constantly, because it is easy to instantiate a default processor and not notice it doesn't match.

![Graph figure showing a checkpoint fanning into a matching feature extractor and tokenizer that feed the encoder-decoder, contrasted with a default extractor on wrong-sample-rate audio producing garbage output](/imgs/blogs/debugging-asr-finetuning-5.png)

Why does a mismatch matter so much? Because the model was trained on features with *exact* statistical properties, and the feature extractor reproduces those properties. Whisper expects an 80-channel (or 128-channel for `large-v3`) log-mel spectrogram computed with a specific FFT size, hop length, and mel filterbank, on audio resampled to 16 kHz, padded or trimmed to 30 seconds, and normalized in a specific way. If you feed it a spectrogram with a different number of mel bins, a different hop, or audio at the wrong sample rate, you are presenting the encoder with inputs from a distribution it has never seen. The encoder's learned filters expect energy in particular frequency bins at particular frame rates; shift those and the representations are noise. The model will still produce *some* output, the loss will be high and flat, and it will look like "the model isn't learning" — when really the model is being fed garbage. This is the deep tie to [audio input bugs](/blog/machine-learning/debugging-training/audio-input-bugs): a sample-rate or mel-parameter mismatch is upstream of everything, and it presents in finetuning as a loss that won't move.

The tokenizer half is just as load-bearing. Whisper's tokenizer is a byte-level BPE with a specific vocabulary including all those special tokens. If you instantiate a tokenizer from a different Whisper size or, worse, a different model family, the special-token IDs won't line up, your `-100` masking will mask the wrong positions, and the decoder prompt will be built from the wrong IDs. For wav2vec2 CTC, the tokenizer is typically a character or BPE vocabulary you may have *built yourself* from your dataset — and there the failure mode is a vocabulary that doesn't include a character present in your transcripts, which silently maps to `<unk>` and caps your achievable accuracy.

Here is the diagnostic: load the processor from your checkpoint, not from a default, and assert the audio is at the right rate.

```python
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

CHECKPOINT = "openai/whisper-small"

# Load BOTH from the same checkpoint. Never mix.
processor = WhisperProcessor.from_pretrained(CHECKPOINT)
model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT)

fe = processor.feature_extractor
print("expected sampling rate:", fe.sampling_rate)      # 16000
print("expected mel bins      :", fe.feature_size)       # 80 (small) / 128 (large-v3)

# Load an example and RESAMPLE to the expected rate if needed.
waveform, sr = torchaudio.load("example.wav")
if sr != fe.sampling_rate:
    print(f"WARNING resampling {sr} -> {fe.sampling_rate}")
    waveform = torchaudio.functional.resample(waveform, sr, fe.sampling_rate)
if waveform.shape[0] > 1:                                # downmix stereo to mono
    waveform = waveform.mean(dim=0, keepdim=True)

features = fe(waveform.squeeze().numpy(),
             sampling_rate=fe.sampling_rate,
             return_tensors="pt").input_features
print("feature tensor shape:", features.shape)           # (1, 80, 3000) for 30s @ small

# A 30s window at 16kHz with hop 160 gives 3000 frames; assert it, so a
# wrong hop/sr shows up immediately instead of as mysterious high loss.
assert features.shape[1] == fe.feature_size, "mel-bin mismatch -> wrong extractor"
```

That assert on the mel-bin count is the kind of cheap guardrail that converts a silent two-week bug into an immediate crash. The series-wide theme: spend a line of code to make a silent failure loud.

The wav2vec2 side has its own version of the same contract, and the failure surface is a little different because the "feature extractor" is much simpler (it normalizes the raw waveform rather than computing a mel spectrogram) while the "tokenizer" is often something *you build*, which is where the danger concentrates. Here is the matched-processor pattern for wav2vec2, plus the vocabulary-coverage check that catches the silent `<unk>` ceiling:

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

CHECKPOINT = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT)
model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT)

fe = processor.feature_extractor
print("expected sampling rate :", fe.sampling_rate)          # 16000
print("do_normalize           :", fe.do_normalize)           # waveform z-norm
print("return_attention_mask  :", fe.return_attention_mask)  # MUST be True for padded batches

# wav2vec2 normalizes the raw waveform to zero mean / unit variance.
# Skipping this (feeding raw int16-scaled audio) shifts the input
# distribution and the encoder produces noise -> flat high loss.
import numpy as np
wav = np.random.randn(16000).astype("float32")
inp = processor(wav, sampling_rate=16000, return_tensors="pt")
print("normalized input mean/std:",
      float(inp.input_values.mean()), float(inp.input_values.std()))
```

Two things in that printout are load-bearing. `do_normalize` must be on, because wav2vec2 was trained on zero-mean/unit-variance waveforms; feeding it un-normalized audio (e.g. raw `int16` rescaled to float without standardizing) shifts the input distribution and the encoder, again, sees out-of-distribution input and the loss won't move. And `return_attention_mask` must be `True` whenever you pad variable-length clips into a batch, or the CTC head attends to padded frames and the alignment drifts — a batch-composition-dependent WER we will revisit in the case studies.

#### Worked example: the loss that wouldn't move

A run on Vietnamese podcast audio sat at a flat training loss around 6.2 for 2,000 steps — the textbook "model isn't learning" curve. The team assumed a learning-rate problem and started sweeping. The actual bug: their audio files were 44.1 kHz, and their data-prep code passed the raw arrays to the feature extractor *without resampling*, while telling the extractor `sampling_rate=16000`. The extractor dutifully computed a mel spectrogram as if the audio were 16 kHz, which stretched every formant to the wrong frequency bin — the encoder saw audio "pitched up" by a factor of 2.76 (44100/16000). The fix was a single `torchaudio.functional.resample` call in the collator. Loss dropped from 6.2 flat to 1.1 in the first 200 steps and the finetune converged to 11% WER. No learning-rate change was needed; the LR had never been the problem. This is exactly the bisection lesson: *before* you blame optimization, prove the data pipeline is feeding the model what it expects. The overfit-one-batch test (Section 8) would have caught this in five minutes, because a model fed garbage features cannot overfit even a single batch.

## 4. SpecAugment: the regularizer that can starve the model

SpecAugment is the standard data augmentation for speech: it randomly masks contiguous bands of time and frequency in the spectrogram so the model can't rely on any single region and learns more robust features. It is genuinely effective — the original SpecAugment paper (Park et al., 2019) showed large WER improvements on LibriSpeech — and it is on by default in many wav2vec2 finetuning configs and easy to add to Whisper. But it has two failure modes that are mirror images of each other, and both show up as a worse-than-expected WER.

![Before-after figure contrasting an over-aggressive SpecAugment applied at eval producing high error with a moderate train-only policy producing low error](/imgs/blogs/debugging-asr-finetuning-4.png)

The first failure is **too aggressive**. SpecAugment has knobs: the number of time masks, the maximum width of each time mask (in frames), the number of frequency masks, and the maximum width of each frequency mask (in mel bins). If you mask too much — say, two time masks each up to 100 frames on a 3,000-frame window plus aggressive frequency masking — you can routinely obliterate so much of the spectrogram that the remaining signal is insufficient to recognize the words. The model can't fit the training data because you've thrown away the evidence. The signature is *underfitting*: training WER stays high and won't come down no matter how long you train. This is the opposite of the usual augmentation intuition (more augmentation = better generalization); past a threshold, augmentation is just noise injection that destroys signal. wav2vec2's default masking is fairly strong because it was tuned for large pretraining; on a small finetune it can be too much.

The second failure is the **train/eval leak**: applying SpecAugment at evaluation time. Augmentation is a *training-only* operation. At eval you want the clean spectrogram. If SpecAugment is wired into the forward pass without a guard on training mode, every evaluation runs on randomly-masked spectrograms, which inflates eval WER and — worse — makes it *nondeterministic*, so your WER changes run to run for the same checkpoint. This is the speech-specific instance of the classic [train/eval-mode bug](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs): an operation that should respect `model.training` doesn't, and your eval numbers are quietly corrupted.

### The science: why masking helps up to a point and then hurts

The intuition here is a bias-variance argument made concrete. Masking removes information from each training example, which has two effects. It increases the *effective diversity* of the training set — the model sees many partially-occluded versions of each utterance, which prevents it from memorizing exact spectral patterns and forces it to use redundant cues (a word is recognizable from multiple frequency bands and time spans). That reduces variance and improves generalization. But masking also reduces the *mutual information* between the input and the target: if you mask the frames where a word's phonemes actually occur, no model can recover that word, and you have injected label noise. The optimum is the masking dose where you have removed enough redundancy to force robustness but not so much that you've destroyed the discriminative signal.

You can reason about the ceiling quantitatively. If a phoneme occupies roughly 80–120 ms and you mask time spans up to 100 frames at a 10 ms hop (1,000 ms = one full second), a single time mask can swallow several entire words. Two such masks can remove 2 seconds from a 5-second utterance — 40% of the audio. At that dose the model is being asked to transcribe words it literally cannot hear in a large fraction of examples, and training WER plateaus at whatever fraction of words happen to survive the masks. The fix is to cap mask widths to a fraction of the utterance (the paper uses an adaptive cap proportional to the utterance length, `time_mask_param` scaled by a fraction `p` of the number of frames), so a mask never swallows a meaningful fraction of any single example.

Here is a SpecAugment audit you can drop into a training run to confirm the policy is both moderate and train-only:

```python
import torch
import torchaudio.transforms as T

class GuardedSpecAugment(torch.nn.Module):
    """SpecAugment that is provably train-only and dose-capped."""
    def __init__(self, n_time_masks=2, time_mask_frac=0.05,
                 n_freq_masks=2, freq_mask_param=27):
        super().__init__()
        self.n_time_masks = n_time_masks
        self.time_mask_frac = time_mask_frac      # cap each mask to 5% of frames
        self.freq = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.n_freq_masks = n_freq_masks

    def forward(self, spec):                       # spec: (batch, mels, frames)
        if not self.training:                      # HARD GUARD: no aug at eval
            return spec
        n_frames = spec.shape[-1]
        time_mask_param = max(1, int(self.time_mask_frac * n_frames))
        time = T.TimeMasking(time_mask_param=time_mask_param)
        for _ in range(self.n_time_masks):
            spec = time(spec)
        for _ in range(self.n_freq_masks):
            spec = self.freq(spec)
        return spec

aug = GuardedSpecAugment()

# AUDIT 1: prove it is a no-op in eval mode.
aug.eval()
x = torch.randn(1, 80, 3000)
assert torch.equal(aug(x), x), "SpecAugment is modifying eval-mode input!"

# AUDIT 2: measure how much signal a train-mode pass destroys.
aug.train()
y = aug(x.clone())
masked_frac = (y == 0).float().mean().item()
print(f"fraction of spectrogram zeroed in train mode: {masked_frac:.1%}")
# A healthy dose is single-digit percent. >25% means you are underfitting on purpose.
```

The two asserts are the whole point. The first proves SpecAugment respects eval mode; the second quantifies the dose so "too aggressive" stops being a vibe and becomes a number you can compare against a budget.

### Why wav2vec2's default masking is so easy to over-apply

It is worth being specific about wav2vec2's masking because the default config is a trap for small finetunes. wav2vec2 doesn't mask a fixed number of contiguous spans; it samples *starting positions* with probability `mask_time_prob` and masks a span of `mask_time_length` frames from each. When `mask_time_prob` is high (the pretraining default is around 0.65), the sampled spans overlap heavily, and the *total* fraction of frames masked is much larger than `mask_time_prob × mask_time_length / sequence_length` would naively suggest — because overlapping masks compound. The expected masked fraction under independent span starts is approximately $1 - (1 - p)^{L}$ for a per-frame inclusion that depends on the span length $L$ and probability $p$, which saturates toward "almost everything" as either grows. That non-linearity is exactly why a config that is perfect for self-supervised pretraining on thousands of hours destroys a 40-hour finetune: pretraining *wants* an aggressive masking ratio because the pretext task is to reconstruct masked content from context, but a supervised finetune wants to actually *see* the audio it is transcribing. Copying the pretraining masking ratio into a finetune is a category error, and it is the single most common reason a wav2vec2 finetune underfits.

The practical rule: for a finetune, measure the realized masked fraction (the audit in the snippet above does this directly) and keep it in the single-digit-to-low-teens percent range. Don't trust the config knobs to tell you the dose; measure the dose.

#### Worked example: the finetune that wouldn't beat the baseline

A wav2vec2 finetune on read speech stalled at 19% WER on both train and eval — suspiciously, the train WER was barely better than eval, which rules out overfitting and points at underfitting or a data problem. We ran the audit above and found that train-mode passes were zeroing 31% of every spectrogram, because the config used `mask_time_length=10` with `mask_time_prob=0.65` (wav2vec2's masking is probabilistic over starting positions, and at that probability the masks overlap heavily). Thirty-one percent of the signal gone, on a small finetune, is too much. We dropped `mask_time_prob` to 0.20 and the audit showed about 8% zeroed; train WER fell to 7% and eval to 8% over the same number of steps. The before/after is exactly Figure 4: a model that was underfitting because we were starving it became a model that learned. The diagnostic was a one-line measurement of masked fraction — not a hyperparameter sweep.

## 5. Loss masking and the decoder labels

We have covered the scorer, the prompt, the processor, and augmentation. The fifth place ASR finetunes break is the loss — specifically, *what positions the loss is computed over.* For Whisper's seq2seq cross-entropy, the labels are the decoder target tokens, and you must mask two kinds of positions to `-100` (PyTorch's `ignore_index`, the value cross-entropy skips): the special-token prompt prefix and the padding tokens. Get this wrong and you either train the model on the prompt (teaching it to "predict" tokens that are always supplied at inference, which wastes capacity and can cause the model to emit prompt tokens in its output) or you train on padding (teaching it to predict pad tokens, which biases length and can make the model stop early or never stop).

The mechanism is straightforward once you see it. Cross-entropy loss at position $t$ is $-\log p_\theta(y_t \mid y_{\lt t}, \text{audio})$, summed (or averaged) over all positions $t$ where the label is not `-100`. The special prompt tokens (`<|startoftranscript|>`, the language token, etc.) are *given* at inference; the model never has to predict them, so including them in the loss is at best wasted gradient and at worst harmful — if the model learns to assign high probability to emitting `<|transcribe|>` mid-sequence, it can leak that token into transcriptions. Padding tokens are even worse: in a batch, shorter sequences are padded to the longest length, and those pad positions carry no information; computing loss on them teaches the model statistics of your batching, not of speech.

Here is the correct label construction for a Whisper finetune, the way the standard Hugging Face data collator does it:

```python
import torch
from dataclasses import dataclass
from transformers import WhisperProcessor

@dataclass
class WhisperCollator:
    processor: WhisperProcessor

    def __call__(self, features):
        # 1) pad the input mel features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2) pad the label token ids
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 3) replace PAD positions with -100 so loss ignores them
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100)

        # 4) if the tokenizer prepended the BOS/startoftranscript token to EVERY
        #    label, strip it -- the model adds it as the decoder_start_token_id,
        #    so leaving it in double-counts the prompt.
        bos = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == bos).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
```

The subtle bug lives in step 4. Whisper's `WhisperForConditionalGeneration` automatically prepends the decoder start token internally; if your tokenizer *also* put `<|startoftranscript|>` at the front of every label sequence, you have it twice, the labels are shifted by one relative to what the model predicts, and the model learns a corrupted alignment. This is the speech version of the off-by-one [loss-masking bug](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) that haunts LLM finetuning: a one-token shift between what you supply and what you score is enough to wreck a run while the loss still goes down (just to a worse floor).

### The "won't stop" bug

A specific, common, and maddening variant: the model never stops generating. It transcribes the speech correctly and then keeps going — repeating the last phrase, hallucinating, or running to the max length. This is an EOS-handling bug. The model stops at inference when it predicts `<|endoftext|>`; it can only learn to do that if `<|endoftext|>` appears at the end of the labels *and is included in the loss* (not masked to `-100`). If your label construction strips or masks the final EOS, the model is never trained to predict it, and at inference it has no learned signal to stop. The diagnostic is to print one label sequence and confirm the last non-masked token is the EOS ID:

```python
sample = labels[0]
non_masked = sample[sample != -100]
eos = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
print("last trained token:", non_masked[-1].item(),
      "(EOS is", eos, ")")
assert non_masked[-1].item() == eos, "EOS not in loss -> model won't learn to stop"
```

If that assert fires, you have explained the runaway generation without watching a single decode. The contrast between the two loss paths — Whisper's prompt-and-EOS-laden cross-entropy versus wav2vec2's CTC — is worth seeing in one image, because they fail in different vocabularies.

![Before-after figure comparing the Whisper sequence-to-sequence cross-entropy loss path with the wav2vec2 CTC loss path, each with its own distinct failure signatures](/imgs/blogs/debugging-asr-finetuning-6.png)

## 6. wav2vec2 and CTC: a different loss, a different bug set

Whisper is encoder-decoder with cross-entropy. wav2vec2 (and Conformer-CTC, and most "encoder-only" ASR) is an encoder followed by a linear head that emits a probability distribution over the vocabulary (plus a special **blank** token) at every audio frame, trained with Connectionist Temporal Classification (CTC) loss. Because the loss is fundamentally different, the bugs are fundamentally different, and you debug them with a different toolkit. This section is the bridge to the dedicated [CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) post; here we cover the parts that bite specifically during *finetuning*.

CTC's job is to align a frame-level output sequence (say, 750 frames for a 15-second clip) to a much shorter label sequence (say, 40 characters) without requiring a per-frame label. It does this by summing the probability over all possible alignments that collapse to the target, where collapsing means "remove repeated tokens, then remove blanks." The blank token is what lets the model say "no new output here" and what separates genuine repeated characters (the two l's in "hello" need a blank between them or they collapse to one). Understanding the blank is essential because two of the three big CTC finetuning bugs are about it.

Make the loss concrete. The model outputs, at each of $T$ frames, a probability distribution over the vocabulary plus blank. An *alignment* $\pi$ is one choice of token (possibly blank) per frame — a path of length $T$. A many-to-one collapse function $\mathcal{B}$ maps a path to a label string by merging adjacent duplicates and then deleting blanks; so `B(a a _ b b) = ab` and `B(a a a) = a`. The probability of a target label sequence $\mathbf{y}$ is the sum over *every* path that collapses to it:

$$p(\mathbf{y} \mid \text{audio}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^{T} p_t(\pi_t)$$

and the CTC loss is $-\log p(\mathbf{y} \mid \text{audio})$, computed efficiently with a forward-backward dynamic program over a lattice rather than by enumerating the exponentially many paths. The key consequence for debugging is the support of that sum: if there is *no* path of length $T$ that can possibly collapse to $\mathbf{y}$ — which happens precisely when $T$ is too small — then $\mathcal{B}^{-1}(\mathbf{y})$ is empty, the sum is zero, and the loss is $-\log 0 = +\infty$. This is not a numerical fluke you can clip away; it is the loss correctly reporting that the target is unreachable. That single fact explains the most violent CTC finetuning failure, which we hit below.

### Bug 1: input shorter than target gives infinite loss

CTC has a hard mathematical constraint: the input (frame) sequence must be at least as long as the target, and strictly, after accounting for the blanks needed between repeated characters. The number of frames $T$ must satisfy $T \geq 2L + 1$ in the worst case for a target of length $L$ with all-repeated characters (you need a blank between each pair plus capacity for each character). If $T < L$, there is *no valid alignment*, the probability of the target is zero, and the loss is $-\log 0 = +\infty$. In practice this happens when:

- Your audio is downsampled aggressively by the encoder (wav2vec2 downsamples by ~320x, so 1 second of 16 kHz audio is only 50 frames), and your transcript is long relative to a short clip.
- A data bug truncates the audio but not the transcript.
- You forgot to filter out examples where the transcript is longer than the (downsampled) audio.

The signature is `loss = inf` (often appearing as NaN after a backward pass) on specific batches, not all of them — which is the tell that it's data-dependent, not a global numerics problem. The fix is to filter the dataset: drop or flag examples where the frame count is less than the label length. Here is the guard:

```python
def ctc_length_ok(num_audio_frames, label_len, downsample=320):
    # wav2vec2 conv stack reduces 16kHz samples by ~320x to feature frames.
    feature_frames = num_audio_frames // downsample
    # need at least one frame per label token, with blanks for repeats.
    return feature_frames >= label_len

# Apply as a dataset filter so no inf-loss batch ever reaches the optimizer.
ds = ds.filter(lambda ex: ctc_length_ok(len(ex["audio"]["array"]),
                                        len(ex["labels"])))
```

For the deep version of why CTC returns `inf` and how to hunt the offending batch, see [hunting NaNs and infs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) via the taxonomy — the bisection-by-step technique applies directly.

### Bug 2: the vocabulary doesn't cover the transcripts

For wav2vec2 CTC you usually build the tokenizer vocabulary from your dataset's characters. If a character appears in your transcripts but not in your vocab (an accented letter, a special punctuation mark, a non-Latin script character), it maps to `<unk>`, and the model can never predict it correctly. Your WER floors at the rate those characters appear. The diagnostic is a set difference:

```python
vocab = set(processor.tokenizer.get_vocab().keys())
chars_in_data = set("".join(ex["sentence"] for ex in ds))
missing = chars_in_data - vocab
print("characters in data but NOT in vocab:", sorted(missing))
```

If that set is non-empty, you have a hard ceiling on accuracy that no amount of training removes. Add the characters to the vocab and rebuild the head.

### Bug 3: repeated-token collapse and the blank

A subtler CTC bug: the model outputs the right characters but the *collapse* produces wrong text because of how repeats and blanks interact. If the model under-predicts blanks, genuinely repeated characters merge ("bookkeeper" loses its doubles). If it over-predicts blanks, it may fragment words. During finetuning this usually self-corrects, but if you see systematic doubling/de-doubling errors, inspect the raw (pre-collapse) argmax path against the collapsed output to confirm the blank behavior. That inspection — printing the frame-level argmax and watching the collapse — is the CTC analog of printing the decoder prompt for Whisper, and it's covered in depth in the [CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) post.

### A note on Conformer-CTC and the same constraints

Conformer-based CTC models (the convolution-augmented transformers common in toolkits like NeMo and ESPnet) finetune with the *same* CTC machinery, so the same three bugs apply with the same fixes, but two details shift. First, the downsampling factor differs — many Conformer front-ends downsample by a factor of 4 or 8 via strided convolutions rather than wav2vec2's ~320, so the $T \geq L$ frame budget is more generous and the `inf`-loss trap is rarer (though still real for very short clips with long transcripts). Always compute the *actual* downsample factor from your front-end config rather than assuming, because that number is the entire frame budget. Second, Conformer pipelines frequently apply SpecAugment on the mel features as a first layer, which means the train/eval gating bug from Section 4 is wired in at the model level rather than the data level — so the place you assert "no augmentation in eval mode" moves from the collator into the model's forward, but the assert is identical. The point is that the core ideas transfer directly: blank token, length constraint, vocabulary coverage, train-only augmentation, matched processor. Learn them once on wav2vec2 and you debug any CTC model.

#### Worked example: the inf-loss that came and went

A Conformer-CTC finetune on lecture audio threw NaN loss intermittently — clean for a few hundred steps, then NaN, then clean again after a restart with a different shuffle. Intermittent-and-shuffle-dependent is the fingerprint of a *bad batch*, not a global numerics bug (a global bug, like too-high LR, gives a smooth-then-NaN curve at a consistent step; see [loss spikes and divergence](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs)). We added a per-batch check that logged the example IDs whenever loss was non-finite and found 14 examples where the transcript was longer than the downsampled frame count — short clips that had been mis-segmented to include a long transcript. They produced `inf` CTC loss whenever they landed in a batch. The `ctc_length_ok` filter removed all 14, and the run went from "NaN somewhere in the first 500 steps, every time" to clean for the full 20,000 steps. The lesson is the series' core bisection move: an intermittent failure that depends on data order is a *data* bug, and you find it by logging which examples were in the failing batch.

## 7. The diagnostic flow: a bisection for high WER

Now we assemble the pieces into a procedure. When a finetune reports a bad WER, you do not start retraining. You bisect, in this order, because each step is cheaper than the next and each rules out a whole class of bug.

![Tree figure routing a high word error rate to one of three suspects, the scorer, the decoder prompt, or the acoustic model, with a confirming test under each branch](/imgs/blogs/debugging-asr-finetuning-7.png)

**Step 0 — Eyeball five transcriptions next to references.** This costs thirty seconds and rules out the single most common bug. Print five (hypothesis, reference) pairs. If the hypotheses look correct to a human and the only differences are casing, punctuation, contractions, and number formats, your bug is the *scorer*, not the model. Go to Step 1. If the hypotheses are fluent but in the wrong language or are translations/paraphrases, your bug is the *prompt*. Go to Step 2. If the hypotheses are phonetic garbage (wrong words, nonsense), your bug is the *acoustic model* (or its features). Go to Step 3.

**Step 1 — Run raw vs normalized WER side by side.** If normalized WER is dramatically lower than raw, you have confirmed a scorer bug; switch to a standard normalizer and you are done. If normalized WER is also bad, the model genuinely struggles — continue to Step 2 or 3 based on what the transcripts look like.

**Step 2 — Print the decoder prompt / forced decoder IDs.** Confirm the language and task tokens. Fix any skew between training and inference. Re-evaluate the *existing* checkpoint first; a prompt-only bug often needs no retraining.

**Step 3 — Audit features and the acoustic path.** Confirm the processor matches the checkpoint, the sample rate is right, the mel-bin count asserts pass, and SpecAugment is moderate and train-only. Then run the overfit-one-batch test: can the model drive loss to near zero on a single batch? If yes, the acoustic stack works and the problem is data quantity/quality or LR; if no, the model can't even fit one example and the bug is upstream (features, masking, a frozen encoder, or a wrong LR destroying the pretrained weights).

That last point — LR destroying pretrained weights — connects straight to [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it): a pretrained acoustic model sits in a sharp loss basin, and a from-scratch learning rate (say `1e-3`) will detonate it in the first dozen steps, producing exactly the "phonetic garbage" symptom. Whisper finetunes typically want `1e-5` to `1e-6`; wav2vec2 wants something like `1e-4` to `3e-4` for the head with a much smaller (or zero, early on) rate for the pretrained encoder. The same Hessian-curvature argument applies: a stable Adam step needs $\eta \lesssim 2/\lambda_{\max}$ where $\lambda_{\max}$ is the largest curvature near the pretrained minimum, and that bound is tiny for a converged model.

To see why the bound is so much smaller for a finetune than for from-scratch training, picture the loss locally as a quadratic bowl $L(\theta) \approx L(\theta^\*) + \tfrac{1}{2}(\theta - \theta^\*)^\top H (\theta - \theta^\*)$ around a minimum $\theta^\*$, where $H$ is the Hessian. Gradient descent with step $\eta$ along an eigendirection of curvature $\lambda$ multiplies the error in that direction by $(1 - \eta\lambda)$ each step; the update is stable only if $|1 - \eta\lambda| < 1$, i.e. $\eta < 2/\lambda$. A randomly-initialized model starts in a flat, high-loss region where the largest eigenvalue $\lambda_{\max}$ is small, so large steps are stable and even beneficial. A pretrained model sits at the bottom of a *sharp* basin where $\lambda_{\max}$ is large; the stable-step bound $2/\lambda_{\max}$ collapses, and a step that was fine from scratch now has $\eta\lambda_{\max} > 2$, so the error in the sharpest direction *grows* every step. That is the loss spike you see in the first dozen steps of an over-LR'd finetune — the model is being kicked out of the basin it spent a million GPU-hours finding. The fix is not subtle: drop the LR by one to two orders of magnitude, and add warmup so the first steps, where the optimizer's second-moment estimates are still cold and the effective step is largest, stay inside the basin.

Here is the eyeball-and-bisect harness as code, the thing you actually run first:

```python
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("./my-finetuned-whisper")
model = WhisperForConditionalGeneration.from_pretrained("./my-finetuned-whisper").eval()

def transcribe(audio_array, sr=16000, language="en", task="transcribe"):
    feats = processor.feature_extractor(
        audio_array, sampling_rate=sr, return_tensors="pt").input_features
    forced = processor.get_decoder_prompt_ids(language=language, task=task)
    with torch.no_grad():
        ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=200)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# STEP 0: eyeball five (hyp, ref) pairs -- the highest-leverage 30 seconds.
for ex in eval_set[:5]:
    hyp = transcribe(ex["audio"]["array"])
    print("REF:", ex["sentence"])
    print("HYP:", hyp)
    print("---")
```

If those five pairs look right but the scorer disagrees, you have already solved 60% of real ASR debugging cases before computing a single WER.

## 8. A WER harness that does not lie

The fix for the most common bug is a scorer you can trust. The standard tool is `jiwer`, and the standard normalizer for Whisper-family models is the one shipped in the `transformers` library (`BasicTextNormalizer` for general use, or the English-specific `EnglishTextNormalizer` that also handles number words, contractions, and common abbreviations the way the Whisper paper's evaluation did). The rule: **always report normalized WER, and report raw WER next to it during debugging so the gap is visible.** A large gap is itself a diagnostic signal — it tells you formatting, not phonetics, dominates your error.

```python
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# Build the English normalizer the way Whisper's eval does.
normalizer = EnglishTextNormalizer({})   # pass an abbreviations dict if you have one

def compute_wer(hyps, refs):
    # 1) RAW: exactly as the model and references are written.
    raw = jiwer.wer(refs, hyps)

    # 2) NORMALIZED: lowercase, strip punctuation, expand contractions,
    #    standardize numbers -- then score.
    norm_hyps = [normalizer(h) for h in hyps]
    norm_refs = [normalizer(r) for r in refs]
    # drop pairs that normalize to empty (jiwer errors on empty references)
    pairs = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
    nh, nr = zip(*pairs)
    normalized = jiwer.wer(list(nr), list(nh))

    # 3) the gap is itself a signal: a big gap == formatting, not phonetics
    return {"raw_wer": raw, "normalized_wer": normalized,
            "gap": raw - normalized}

result = compute_wer(hypotheses, references)
print(f"raw WER        : {result['raw_wer']:.1%}")
print(f"normalized WER : {result['normalized_wer']:.1%}")
print(f"formatting gap : {result['gap']:.1%}")   # >0.10 => fix the scorer, not the model
```

For Hugging Face's `Seq2SeqTrainer`, wire this into `compute_metrics` so every eval step reports the honest number:

```python
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # restore -100 -> pad so decoding works
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    hyps = processor.batch_decode(pred_ids, skip_special_tokens=True)
    refs = processor.batch_decode(label_ids, skip_special_tokens=True)
    out = compute_wer(hyps, refs)
    return {"wer": out["normalized_wer"], "raw_wer": out["raw_wer"]}

args = Seq2SeqTrainingArguments(
    output_dir="./whisper-ft",
    per_device_train_batch_size=16,
    learning_rate=1e-5,            # finetune rate, NOT 1e-3
    warmup_steps=500,
    max_steps=4000,
    predict_with_generate=True,    # generate full sequences for eval, not teacher-forced
    fp16=True,
    eval_strategy="steps",
    eval_steps=500,
)
```

Two flags in that config are load-bearing. `predict_with_generate=True` makes evaluation run real autoregressive generation rather than teacher-forced next-token prediction — without it, your "WER" is computed on teacher-forced outputs, which are far better than real generation and give you a falsely optimistic number (a subtle train/infer mismatch). And `learning_rate=1e-5` is the finetune rate; the tutorial default of `1e-3` is the from-scratch rate that destroys the pretrained encoder.

The other instrument I promised back in Section 1 is the **per-utterance WER distribution.** The aggregate WER is a mean, and a mean hides structure; the distribution does not. Computing it costs a few lines and routinely turns "the model is 30% wrong" into "the model is 5% wrong except on this identifiable cluster":

```python
import numpy as np

def per_utterance_wer(hyps, refs, normalize=True):
    fn = normalizer if normalize else (lambda x: x)
    wers = []
    for h, r in zip(hyps, refs):
        nr = fn(r)
        if not nr.strip():
            continue                      # skip empty references
        wers.append(jiwer.wer(nr, fn(h)))
    wers = np.array(wers)
    print(f"median per-utt WER : {np.median(wers):.1%}")
    print(f"90th pct WER       : {np.quantile(wers, 0.90):.1%}")
    print(f"utts at 100% WER   : {(wers >= 1.0).mean():.1%}")  # empty/garbage outputs
    return wers

wers = per_utterance_wer(hypotheses, references)
```

A median near 5% with a 90th percentile near 100% is the unmistakable signature of a *subset* failure: most of the corpus is fine, and a specific cluster is catastrophic. That cluster is almost always one identifiable thing — clips longer than 30 seconds (the windowing bug), clips in a second language (a prompt bug), or clips the model returned empty for (a choke on silence or noise). The mean would have told you "30% WER, model is mediocre"; the distribution tells you "model is excellent on 90% of clips and you have one specific bug to find." That is the difference between a week of aimless retraining and an afternoon of targeted fixing.

The overfit-one-batch test deserves its own snippet, because it is the single most decisive sanity check in this whole post — it cleanly separates "the model and pipeline work" from "something upstream is broken":

```python
# Take ONE batch. Disable augmentation and shuffling. Train only on it.
# A healthy model+pipeline drives loss to near zero within ~100 steps.
one_batch = next(iter(train_loader))
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step in range(100):
    opt.zero_grad()
    out = model(**{k: v.to(model.device) for k, v in one_batch.items()})
    out.loss.backward()
    opt.step()
    if step % 20 == 0:
        print(f"step {step}: loss {out.loss.item():.4f}")
# If loss does NOT approach ~0, the bug is upstream (features, masking,
# frozen encoder, wrong LR) -- NOT data quantity. Stop blaming the dataset.
```

This is [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) applied to speech. If a Whisper finetune cannot drive a single batch to near-zero loss, you have a structural bug — a mismatched processor, a frozen encoder, a wrong LR, or labels that don't line up — and no amount of more data will help. The whole bug-class-to-symptom-to-fix mapping is summarized in the matrix below; print it next to your run.

![Matrix figure mapping five ASR finetuning bugs to their symptom, a one-line confirming check, and the targeted fix](/imgs/blogs/debugging-asr-finetuning-3.png)

## 9. Long-form inference: trained on 30 seconds, tested on 12 minutes

The last big ASR finetuning trap is a *train/inference* mismatch that has nothing to do with the model weights and everything to do with how you feed audio at serving time. Whisper's encoder is built for exactly 30 seconds of audio: every input is padded or trimmed to a 3,000-frame (30s) window. The model literally cannot attend to audio beyond that window in a single forward pass. So what happens when you take your beautifully finetuned model and run it on a 12-minute podcast in one call? Everything past the first 30 seconds is silently dropped, and your WER on long files collapses — not because the model is bad, but because the inference path threw away 95% of the audio.

![Before-after figure showing a long audio file fed in one pass truncating past 30 seconds versus chunked 30-second decoding with overlap recovering the full transcription](/imgs/blogs/debugging-asr-finetuning-8.png)

The fix is chunked long-form decoding: split the audio into 30-second windows (with a few seconds of overlap to avoid cutting words at boundaries), transcribe each, and stitch the results, using the timestamp tokens or the overlap to merge correctly. The Hugging Face `pipeline` does this for you if you ask:

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition",
               model="./my-finetuned-whisper",
               chunk_length_s=30,        # 30s windows -- match training
               stride_length_s=5,        # 5s overlap to stitch boundaries
               return_timestamps=True)

# Now a 12-minute file is chunked, decoded, and stitched automatically.
result = asr("long_podcast.wav")
print(result["text"])
```

The deeper point is that ASR has *two* kinds of train/inference mismatch and you must rule out both. The first is the prompt skew from Section 2 (you generate with a different special-token prefix than you trained with). The second is this windowing mismatch (you trained on 30s segments and infer on raw long files). Both produce a model that "works in eval but fails in production," and both are invisible in the training loss. The general pattern — works in teacher-forced training, breaks in free-running inference — is the [train/infer mismatch](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) story that recurs across modalities; in ASR it just wears a 30-second costume.

#### Worked example: the model that aced eval and bombed production

A finetuned Whisper scored 7% WER on the eval set and then posted 50%+ WER in production on full call recordings. The eval set was built from 30-second clips (matching training); production fed entire 8-minute calls in one shot. The model transcribed the first 30 seconds of each call perfectly and emitted nothing for the remaining 7.5 minutes, so against the full reference it "deleted" the vast majority of words — a deletion-heavy WER over 50%. Switching the inference path to the chunked pipeline with `chunk_length_s=30` and a 5-second stride brought production WER to 9%, in line with eval. The model had been correct the entire time; the serving harness was throwing away 94% of every call. The diagnostic was to look at *where* the errors were (almost all deletions, almost all after the 30-second mark) — error *position* and *type* are instruments too.

## 10. Putting the six places to work for ASR

It is worth stepping back to map ASR bugs onto the series' six-places frame, because it makes the bisection automatic. The whole point of the framework is that you don't guess — you locate.

- **Data**: sample-rate/mel mismatch (Section 3), transcripts that don't normalize, characters missing from the wav2vec2 vocab (Section 6), CTC examples where input is shorter than target.
- **Optimization**: the from-scratch LR that destroys the pretrained acoustic model (Section 7), warmup absent so the first steps spike.
- **Model code**: the wrong special token in the decoder prompt (Section 2), the double-BOS / off-by-one in labels, EOS not in the loss (Section 5).
- **Numerics**: fp16 underflow on a quiet finetune (rare for Whisper-small but real at scale), CTC `inf` loss (Section 6).
- **Systems**: long-form windowing at inference (Section 9), train/infer prompt skew.
- **Evaluation**: the WER that lies — raw vs normalized, teacher-forced vs generated eval (Sections 1 and 8).

Notice how heavily ASR loads the **evaluation** and **data/model-code** columns relative to, say, a from-scratch vision run. That is the structural insight of this post: for ASR finetuning, the prior should be "it's the scorer or the contract," not "it's the optimizer." The optimizer is almost always fine. You earn the right to touch the LR only after the overfit-one-batch test fails.

### A diagnostic table to keep next to your run

| Symptom | Most likely place | Confirming test | Fix |
|---|---|---|---|
| WER 60% but transcripts look right | Evaluation | raw vs normalized WER gap | use a standard normalizer (jiwer + Whisper normalizer) |
| Fluent output, wrong language/translated | Model code (prompt) | print forced_decoder_ids | set language and task in generate/data-prep |
| Loss flat and high from step 1 | Data (features) | assert mel-bin count, check sample rate | resample to 16 kHz, load processor from checkpoint |
| Train WER won't drop below ~18% | Data (augmentation) | measure masked fraction of spectrogram | cap SpecAugment dose; gate on model.training |
| Eval WER worse than train, nondeterministic | Evaluation/model code | assert SpecAugment no-op in eval | guard augmentation on training mode |
| Model never stops generating | Model code (loss) | check last non-masked label is EOS | include EOS in labels, not in the -100 mask |
| CTC loss = inf on some batches | Numerics/data | log example IDs of non-finite batches | filter input-shorter-than-target examples |
| Great eval, terrible on long files | Systems | check error position (all deletions after 30s) | chunked long-form decoding (30s + overlap) |
| Phonetic garbage from step 1 | Optimization | overfit one batch; check LR | drop LR to 1e-5 (Whisper) / proper head/encoder split |

Every row is "symptom → place → one-line test → fix." That is the series' decision tree, instantiated for speech.

## 11. Case studies and real signatures

A few well-known patterns, accurately stated, to calibrate your intuition.

**The Whisper normalizer in the original evaluation.** OpenAI's Whisper paper (Radford et al., 2022) was explicit that they applied text normalization before computing WER, precisely because raw WER conflates transcription quality with formatting conventions, and different reference datasets format text differently. Their normalizer handles casing, punctuation, common contractions and abbreviations, and number formatting. The practical takeaway is not a specific number but a discipline: the people who built the model report normalized WER, and if you report raw WER you are not measuring the same thing they were. When you compare your finetune against the base model, score both with the *same* normalizer or the comparison is meaningless.

**The "left-padding-breaks-generation" cousin.** In decoder-only LLM generation, left vs right padding famously changes outputs because position IDs and attention masks interact with the cache. ASR has an analogous, less-discussed version: padding and attention masks on the *encoder* side. For wav2vec2, the attention mask that tells the model which frames are real vs padding must be correct, or the model attends to padding and CTC alignments drift. The signature is a model that does fine on full-length batches and degrades on heavily-padded short clips — a batch-composition-dependent WER. The fix is to pass the attention mask through (many quick scripts forget it) so padded frames are excluded.

**SpecAugment's published gains and its dosage.** The SpecAugment paper (Park et al., 2019) reported large relative WER improvements on LibriSpeech — on the order of a substantial fraction of the error — by masking time and frequency, and crucially they *scaled* the time-mask width to the utterance length rather than using a fixed absolute width. The lesson for finetuning is to copy the *adaptive* policy, not a fixed `mask_time_length` borrowed from large-scale pretraining; a fixed width that is fine on long LibriSpeech utterances is catastrophic on short conversational clips. The bug is not "SpecAugment is bad" — it is "SpecAugment with a width tuned for a different data regime starves a small finetune."

**The CTC length constraint as a hard failure, not a soft one.** The CTC paper (Graves et al., 2006) defines the loss as a sum over alignments, and the constraint $T \geq L$ (and stricter with repeats) is mathematical, not heuristic. This is why the `inf` loss is so abrupt — it is not a large finite number that gradient clipping can save, it is genuinely $-\log 0$. You cannot clip your way out of it; you must filter the offending examples. Teams that try to "stabilize" intermittent CTC NaNs with gradient clipping or lower LR are treating a symptom; the cause is a handful of examples that make the target probability exactly zero.

**The economics of getting this wrong.** It is worth naming the cost, because it is what makes these bugs matter beyond a wrong number. A `whisper-small` finetune on 40 hours of audio is maybe 6–10 GPU-hours; at roughly \$2 per A100-hour on a typical cloud that is a \$15–20 run. That is cheap to re-do — *if you know it's broken.* The expensive part is the human time spent chasing the wrong suspect: the two weeks of proposed architecture changes in the intro example were perhaps \$8,000–10,000 of engineer time spent because nobody printed five transcriptions. The asymmetry is the whole argument for the discipline in this post: the diagnostics cost minutes and dollars; the misattributions cost weeks and thousands. The cheapest possible thing to do when a WER surprises you is also the most decisive — look at the data, print the prompt, measure the dose — and the most expensive thing you can do is assume the model is wrong and start a sweep. Every diagnostic in this post is designed to be run before you spend money, not after.

## 12. When this is (and isn't) your bug

Be decisive about ruling things in and out, because misattribution is what wastes the weeks.

**If the transcripts look right to a human, it is the scorer — stop blaming the model.** A 60% WER with phonetically-perfect transcriptions is a normalization bug 95% of the time. Do not retrain. Do not change architectures. Fix the scorer and re-measure.

**If the output is fluent but wrong-language or paraphrased, it is the prompt — stop blaming the data.** Transcription models don't paraphrase; that's a translate token. They don't switch languages randomly; that's a language token. Print the forced decoder IDs before touching anything else.

**If the loss is flat from step 1 and won't move, it is the features — stop sweeping the learning rate.** A model fed out-of-distribution spectrograms (wrong sample rate, wrong mel count, wrong processor) cannot learn regardless of LR. The overfit-one-batch test will confirm this in five minutes; if it fails, the bug is upstream, not in the optimizer.

**If the WER is great on clips and terrible on long files, it is the inference windowing — not the weights.** Look at where the errors are. If they're all deletions after the 30-second mark, the model is correct and the serving path is truncating. Chunk the audio.

**Conversely, if normalized WER is also bad, the transcripts are genuinely wrong, the prompt is correct, and overfit-one-batch passes — *now* the model is your bug.** Then you are in ordinary finetuning territory: not enough data, wrong LR magnitude, too few or too many epochs, a domain gap between training and eval audio. That is real, and it's where the [finetuning recipe](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) lessons apply. But you only earn that conclusion after the cheaper tests have ruled out the scorer, the prompt, the features, and the windowing.

**A note on what is *not* usually the bug in ASR finetuning.** People reach for exotic explanations — the architecture is wrong, the attention mechanism can't handle their accent, they need a bigger model. In dozens of these runs, the cause was almost never exotic. It was a normalizer, a token, a sample rate, a mask, or a windowing call. The boring explanations are the right ones, and the discipline that finds them is the same one from the start of this series: read the instrument before you touch the code.

## Key takeaways

- **WER is a composition of a model and a normalizer; a bug in either moves the number by tens of points.** Always report normalized WER, and report raw next to it while debugging so the formatting gap is visible.
- **Eyeball five transcriptions before computing any metric.** If they look right but the scorer disagrees, your bug is the scorer, not the model — this alone resolves the majority of "got worse after finetuning" reports.
- **Print the Whisper forced decoder IDs.** A wrong language or task token (`<|fr|>`, `<|translate|>`) produces fluent garbage with a perfectly healthy training loss; fix it before retraining, often with no retraining at all.
- **Load the processor from the checkpoint and assert the mel-bin count and sample rate.** A wrong feature extractor (often a missing resample to 16 kHz) shows up as a flat, high loss that looks like an LR problem but isn't.
- **Measure SpecAugment's dose and gate it on `model.training`.** Masking more than ~25% of the spectrogram starves a small finetune (underfitting); applying augmentation at eval inflates and randomizes WER.
- **Mask the prompt and padding to `-100`, but keep EOS in the loss.** A double-BOS off-by-one corrupts alignment; an EOS dropped from the loss produces a model that never stops generating.
- **CTC's `inf` loss is a hard length constraint, not a numerics wobble.** When input frames are fewer than the target length, the target probability is exactly zero; filter those examples rather than clipping gradients.
- **Use `predict_with_generate=True` and a finetune LR (`1e-5` for Whisper).** Teacher-forced eval gives a falsely optimistic WER; a from-scratch LR detonates the pretrained acoustic model.
- **For long audio, chunk to 30-second windows with overlap.** A model that aces clip-level eval and bombs full files isn't broken — the inference path is truncating everything past the 30-second window.
- **Map every symptom to one of six places.** ASR loads evaluation and data/model-code heavily; the prior should be "it's the scorer or the contract," and you only touch the optimizer after overfit-one-batch fails.

## Further reading

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022 — the decoder-prompt structure, the multitask token format, and the use of text normalization in evaluation.
- Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," 2020 — the encoder/CTC-head finetuning recipe and masking.
- Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," 2019 — time/frequency masking and the adaptive, utterance-length-scaled masking policy.
- Graves et al., "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," 2006 — the CTC loss, the blank token, and the input-length constraint.
- Hugging Face documentation: `WhisperProcessor`, `Wav2Vec2Processor`, `Seq2SeqTrainer`, and the "Fine-Tune Whisper" guide — the canonical data collator, `-100` masking, and `predict_with_generate`.
- The `jiwer` library docs and the `transformers` `EnglishTextNormalizer` / `BasicTextNormalizer` — the standard WER computation and normalization.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master decision tree), [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) (the capstone), [audio input bugs](/blog/machine-learning/debugging-training/audio-input-bugs) (sample-rate and mel mismatches), [debugging CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) (the wav2vec2 loss path), [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) (the general scorer problem), and [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) (the LR-destroys-pretrained-weights mechanism).
