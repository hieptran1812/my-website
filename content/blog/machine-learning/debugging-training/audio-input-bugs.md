---
title: "Audio Input Bugs: Sample Rates, Mel-Spectrograms, and the Feature Mismatch"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Find the speech preprocessing bugs that wreck recognition without a crash, by learning to print and visualize the exact spectrogram your model receives and compare it to the spec it was pretrained on."
tags:
  [
    "debugging",
    "model-training",
    "speech",
    "audio",
    "preprocessing",
    "torchaudio",
    "whisper",
    "finetuning",
    "pytorch",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/audio-input-bugs-1.png"
---

Here is a run that should have worked. You take Whisper-small, a model that transcribes English at a few percent word error rate out of the box, and finetune it on your own call-center recordings. The loss falls. The training curve is smooth and unremarkable. Then you run inference and the transcripts are nonsense — not "slightly wrong," but confidently wrong, a stream of plausible English words that have nothing to do with what was said. You did not change the architecture. You did not pick a bad optimizer. You broke the model before the first encoder layer ran, because the audio you fed it was sampled at 44.1 kHz and Whisper expects exactly 16 kHz, and nobody resampled it. To the model, every vowel sits 2.76 times too high in frequency. The formants that distinguish "cat" from "cot" landed in the wrong mel bins. Nothing crashed. The waveform was "audio," the model trained on "audio," and the recognition evaporated into a sample-rate that nobody printed.

This is the defining property of audio input bugs: **they almost never raise an exception, and the modality hides them better than any other.** A shape mismatch crashes and you fix it in a minute. A `NaN` poisons the loss and screams. But hand a speech model audio at the wrong sample rate, compute the mel-spectrogram with a different FFT size than the pretrained encoder expects, skip the normalization the model was trained with, or silently average a stereo file down to mono in a way the training pipeline never did — and the whole thing runs to completion and reports numbers. The numbers measure a different experiment than the one you meant to run. Worse, you cannot *see* the bug the way you can stare at an image and notice the colors are wrong. The bug lives in a 2-D float tensor of mel energies that looks like noise to the naked eye. In the six-places framing this series is built on — a bug hides in **data, optimization, model code, numerics, systems, or evaluation** — every bug in this post lives in *data*, specifically in the feature-extraction layer, and that entire layer is invisible to your loss curve. Figure 1 is the map: between a waveform on disk and the log-mel tensor that reaches the encoder, there are six transform stages, and a defect at any one of them rewrites the feature without leaving a trace.

![Stack diagram showing the six transform stages an audio batch passes through, from a waveform of unknown sample rate and dtype, to resampling to the rate the model expects, to the short-time Fourier transform, to the mel filterbank, to log compression, to normalization, ending at the encoder which sees a feature that may not match what it learned](/imgs/blogs/audio-input-bugs-1.png)

By the end of this post you will be able to take any speech model that "trained fine but transcribes garbage" and decide in a few minutes whether the input pipeline is the culprit, then localize *which* of seven specific bugs it is. The single discipline that makes this fast is one you can build today: **print and plot the exact feature the model receives, then compare it to the reference spec.** Not the file on disk, not the waveform you think you loaded — the literal `[n_mels, T]` log-mel tensor at the moment it enters the encoder, with its sample rate, shape, value range, mean, and standard deviation printed beside it, and the spectrogram rendered so you can compare it to a known-good clip. We will write that reusable "model's-eye view" function once and use it to catch every bug below. The spine, as always, is the two master tools of the series: **make-it-fail-small** (one clip, one feature, one forward pass you can inspect) and **read the instruments** (here the instrument is the feature tensor itself). This post is G1, the start of the speech and audio track. It instantiates the [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the audio-data branch, shares its print-the-batch discipline with [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) and its sibling [computer vision input-pipeline bugs](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs), and feeds directly into [debugging CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment), [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning), and the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

## 1. The symptom: a speech model that trains fine and transcribes garbage

Let me make the running example concrete, because we are going to debug it all the way down. You are finetuning Whisper-small on a domain corpus — call it the **call-center finetune**, English customer-support calls, recorded as 16-bit WAV files. The training loop is textbook: the Hugging Face `Seq2SeqTrainer`, a sane learning rate of `1e-5`, a cosine schedule, three epochs. The loss falls smoothly from `1.9` to `0.4`. Nothing in the dashboard looks wrong. Then you transcribe a held-out call and read: *"the the and to a the of"* — repetitive, hallucinated, untethered to the audio. Or, in a subtler variant, the transcript is *almost* right but every other word is a near-homophone, and the word error rate sits at `40%` when the base model alone managed `9%` on the same calls.

That last variant is the trap. Audio input bugs come in two flavors, and you need to recognize both:

1. **The loud failure** — transcripts are total nonsense, word error rate is `60–90%`, and the model has obviously never seen anything like the input. This is usually a *gross* mismatch: wrong sample rate, completely wrong feature shape, or features off-scale by orders of magnitude.
2. **The quiet failure** — the model produces fluent, mostly-plausible text that is a few points worse than it should be, forever, for a reason that lives entirely upstream of the gradient. This is a *subtle* mismatch: a slightly different mel parameter, a normalization that is close-but-not-exact, a resampling method that introduced aliasing.

Both flavors share one thing: **the optimizer is doing its job.** Gradients flow, weights update, the loss falls. That is exactly why these bugs are corrosive — they survive the one check everyone runs (does the loss go down?) and they hide from the loss curve entirely, because the loss curve is a thousand-step summary and the bug is in the *content* of every feature, not in the optimization dynamics. Here are the signatures you will actually see, and where each one points:

| Signature | What you observe | First suspect | Why |
| --- | --- | --- | --- |
| Total nonsense, every file | WER 60%+, hallucinated tokens | Sample-rate mismatch | Content shifted in frequency, encoder lost |
| WER off by 10–30 pts | Fluent but wrong words | Mel-param disagreement | Feature has wrong shape or resolution |
| Soft, never clean | WER a few points high forever | Normalization mismatch | Features off the scale the model expects |
| Only some files fail | A subset garbled | Stereo / channel handling | Some files are 2-channel, downmixed wrong |
| First loss enormous | Loss `1e3+` at step 1 | int16 not scaled to float | Amplitudes 32768x too large |
| Words drop at clip edges | Start or end of clips lost | Padding / trimming / mask | Trimmed the audio or fed no attention mask |

Notice the common thread: in every row the *optimizer is doing its job*. The disciplined move is to stop staring at the loss curve and **go look at the feature.** Before any fix, internalize the bisection: the encoder's output is a function of its parameters and the input feature. If the parameters are a known-good pretrained checkpoint, then a model that transcribes garbage and a model that works *differ only in the input feature.* So we hold the weights fixed and interrogate the feature directly. Everything below is a specific question to ask the feature and the specific bug a wrong answer reveals.

### 1.1 Why audio is harder to debug than images

It is worth dwelling on *why* this modality hides bugs so well, because the answer dictates the whole diagnostic strategy. With an image, the tensor that reaches the model is a denormalized version of something your visual system evolved to parse: you display it, your eyes catch the swapped colors or the upside-down crop in a fraction of a second. With audio, the tensor that reaches the model is a **log-mel spectrogram** — a 2-D array of filterbank energies on a logarithmic frequency axis, log-compressed in magnitude. A human cannot look at that array and say "this is sampled too high" the way they can say "this image is blue where it should be red." Two spectrograms that differ only by a sample-rate factor look like two patches of textured noise. Your eyes are useless; you need *instruments*.

That is the entire reason this post leans so hard on printing numbers and comparing to a reference. The feature is not human-legible, so we make it machine-comparable: print the sample rate, the shape, the value range, the mean and standard deviation; render the spectrogram and put it side by side with a known-good clip from the same model's documentation; round-trip a sine wave of known frequency and check that the energy lands in the bin you predicted. The audio bug does not announce itself, so you build a battery of cheap, decisive checks and run them in cost order, exactly as the [taxonomy post](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) prescribes for every class of training bug.

There is a second reason audio is treacherous that has nothing to do with legibility: the feature pipeline has *more independent parameters than any other modality's*. An image pipeline has a handful of knobs — channel order, resize method, normalization mean and std, layout. An audio pipeline has at least a dozen — sample rate, `n_fft`, `win_length`, `hop_length`, window function, `n_mels`, `fmin`, `fmax`, power-vs-magnitude, log base, the dynamic-range clamp, and the normalization scheme — and each one can be defaulted differently by `torchaudio`, `librosa`, `soundfile`, `kaldi`, and the various Hugging Face extractors. The combinatorics are brutal: a dozen knobs that each have two or three plausible settings means hundreds of slightly-different feature pipelines, only one of which matches your pretrained model. The defense is not to memorize the right value of every knob; it is to *stop choosing knobs* and delegate the entire feature computation to the extractor that shipped with the checkpoint, then verify the result matches. The whole strategy of this post collapses to one sentence: do not reimplement the feature, and prove the feature you feed equals the feature the model's own extractor produces.

## 2. The science: the STFT to mel to log pipeline, and why every parameter must match

Before we hunt individual bugs, we need the mechanism, because the mechanism is what makes the bugs *predictable*. A speech model does not consume a raw waveform. It consumes a **feature** computed by a fixed deterministic pipeline, and the pretrained encoder's weights were fit to the exact statistics that pipeline produces. Change any parameter of the pipeline and you change the feature; change the feature and you hand the encoder an input it never saw. Figure 2 below will show the headline result; first, here is the chain.

A waveform $x[n]$ is a sequence of amplitude samples taken at a **sample rate** $f_s$ (samples per second, in Hz). For 16 kHz audio, $f_s = 16000$, so one second of audio is 16,000 numbers. The highest frequency that can be represented is the **Nyquist frequency**, $f_s / 2$ — for 16 kHz that is 8 kHz, which is enough for speech intelligibility (most speech energy lives below 8 kHz). This is the first place the math bites: the sample rate is not just metadata, it *defines the frequency axis of everything downstream.*

### 2.1 Step one: the short-time Fourier transform

Speech is non-stationary — the spectrum changes from phoneme to phoneme — so we cannot take one Fourier transform of the whole clip. Instead we slide a window across the waveform and take a Fourier transform of each short frame. This is the **short-time Fourier transform (STFT)**. Three parameters define it:

- `n_fft` — the FFT size, the number of samples in the transform. It sets the **frequency resolution**: the FFT produces `n_fft/2 + 1` frequency bins spanning 0 to Nyquist, so each bin is $f_s / \text{n\_fft}$ Hz wide. For `n_fft = 400` at 16 kHz, each bin is 40 Hz.
- `win_length` — the analysis window length in samples (often equal to `n_fft`). The waveform is multiplied by a window function (Hann is standard) of this length before the FFT, to reduce spectral leakage. At 16 kHz, `win_length = 400` is a 25 ms window.
- `hop_length` — how many samples the window advances between frames. It sets the **time resolution**: the number of frames is roughly `(len(x) - win_length) / hop_length + 1`. At 16 kHz, `hop_length = 160` is a 10 ms hop, the standard frame rate for speech (100 frames per second).

The STFT of frame $m$ is $S[m, k] = \sum_{n} x[n + m \cdot \text{hop}] \, w[n] \, e^{-i 2\pi k n / \text{n\_fft}}$, a complex number for each frequency bin $k$. We then take the magnitude $|S|$ or the **power** $|S|^2$. This is the first hidden parameter that bites: *power versus magnitude*. The mel filterbank applied to $|S|^2$ produces different energies than applied to $|S|$, and the log of one is twice the log of the other plus a constant — because $\log |S|^2 = 2 \log |S|$. If your training pipeline used power and the pretrained model used magnitude (or vice versa), every feature value is off by a factor of two in the log domain that the encoder's first LayerNorm will partly absorb and partly not.

It is worth grounding these numbers with one concrete frame so the formulas become arithmetic. Take 16 kHz audio, `n_fft = 400`, `hop_length = 160`. Each frame spans `400 / 16000 = 25` ms of audio, and consecutive frames start `160 / 16000 = 10` ms apart, so adjacent frames overlap by 15 ms — a 60% overlap that smooths the spectral estimate over time. The FFT produces `400 / 2 + 1 = 201` complex frequency bins spanning 0 Hz to the Nyquist 8 kHz, so each bin is `8000 / 200 = 40` Hz wide. A one-second clip yields about `(16000 - 400) / 160 + 1 ≈ 98` frames. Now change the sample rate to 8 kHz without changing `n_fft`: each bin is suddenly `4000 / 200 = 20` Hz wide and the same 25 ms window now spans only 200 samples, so *the exact same parameter values produce a different feature on differently-sampled audio.* This is precisely why the sample rate is not separable from the rest of the pipeline — it silently reparameterizes the STFT. When you debug an audio feature, the sample rate is always the first number to print, because every downstream parameter inherits its meaning from it.

### 2.2 Step two: the mel filterbank

The raw STFT has `n_fft/2 + 1` linearly spaced frequency bins — for `n_fft = 400`, that is 201 bins. But human hearing is logarithmic in frequency: we discriminate low frequencies finely and high frequencies coarsely. The **mel scale** approximates this. A common formula is

$$
m(f) = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right),
$$

which maps Hz to mels. The **mel filterbank** is a set of `n_mels` triangular filters spaced evenly on the mel axis, each summing the STFT power across a band of linear bins. The output is an `n_mels`-dimensional vector per frame — for Whisper, `n_mels = 80`, so the 201 linear bins collapse to 80 mel channels. Two filterbank parameters bite:

- `n_mels` — the number of mel channels. This is the *dimensionality of the feature*, and it is baked into the encoder's input projection. Whisper's encoder expects exactly 80 (the large-v3 checkpoint expects 128). Feed it 64 or 128 and the shapes will either crash or, worse, get silently reshaped into garbage.
- `fmin` / `fmax` — the lowest and highest frequencies the filterbank covers. The default is often `fmin = 0`, `fmax = f_s / 2`. If your pipeline clips to `fmin = 20`, `fmax = 7600` and the pretrained model used `0` to `8000`, every triangular filter sits at a slightly different center frequency, and the mel channels no longer mean what the encoder learned.

To make the mel compression concrete, plug a few frequencies into the mel formula. The lowest formant region of speech, around 200 Hz, maps to $m(200) = 2595 \log_{10}(1 + 200/700) ≈ 283$ mels. A consonant burst at 4 kHz maps to $m(4000) = 2595 \log_{10}(1 + 4000/700) ≈ 2146$ mels. The full 0–8000 Hz range spans roughly 0 to 2840 mels. Spreading 80 triangular filters evenly across that mel range means the low-frequency filters are *narrow* in Hz (a few tens of Hz each, where speech is information-dense) and the high-frequency filters are *wide* (hundreds of Hz each, where speech carries less linguistic content). This non-uniform spacing is the whole point of the mel scale, and it is also why a wrong `fmin`/`fmax` is corrosive: shifting `fmax` from 8000 to 7600 Hz recomputes the mel range, which re-spaces *all 80* filters, not just the top one. Every mel channel now integrates a slightly different band of linear bins than the encoder learned. There is also a hidden normalization choice inside the filterbank — whether each triangular filter is area-normalized (the "Slaney" convention) or peak-normalized — and the two produce energies that differ by a per-filter constant. `librosa` and `torchaudio` default this differently, which is the single most common reason two pipelines with identical `n_fft`/`hop`/`n_mels` still produce different features.

### 2.3 Step three: log compression

Filterbank energies span many orders of magnitude — a loud vowel can be a million times more energetic than a quiet fricative. Neural networks train poorly on inputs spanning six decades, so we compress with a logarithm:

$$
\text{logmel}[m, j] = \log\!\left(\text{mel}[m, j] + \epsilon\right),
$$

where $\epsilon$ is a small floor to avoid $\log 0$. But *which* log? Natural log (`ln`), base-10 (`log10`), or the decibel convention $10 \log_{10}(\cdot)$ — these differ by constant multiplicative factors ($\ln x = 2.303 \log_{10} x$). Whisper, specifically, clamps the log-mel to a dynamic range of 80 dB below the maximum and then rescales: `log_spec = (log_spec + 4.0) / 4.0`. Reproduce the log step with the wrong base or skip the clamp-and-rescale, and the feature is on a different scale than the encoder's weights expect.

The takeaway from the whole chain is the scientific spine of this post: **a pretrained speech encoder has memorized a specific feature distribution, produced by a specific deterministic pipeline with specific parameters.** The encoder's first projection and LayerNorm were fit to features with a particular shape, a particular value range, a particular frequency resolution. Hand it a feature from a different pipeline and you are doing an uncontrolled domain shift at the input — sometimes catastrophic (wrong sample rate), sometimes a slow bleed (wrong log base). The whole point of the diagnostics below is to verify, parameter by parameter, that the pipeline you run matches the pipeline the model was trained with.

![Graph diagram showing how the short-time Fourier transform, the mel filterbank, and the log step each carry parameters, with branches for power versus magnitude and for the choice of log base, all converging on a final log-mel feature that must match the spec the encoder learned](/imgs/blogs/audio-input-bugs-3.png)

## 3. Bug one: the sample-rate mismatch, and why content shifts in frequency

The single most common and most destructive audio input bug is the sample-rate mismatch. It is also the most instructive, because the math tells you *exactly* what goes wrong, and the fix has a subtle trap of its own.

### 3.1 The science: feeding the wrong rate compresses or stretches the spectrum

Suppose your model was trained on 16 kHz audio and you hand it a 44.1 kHz waveform *without resampling* — that is, you pass the raw 44,100-sample-per-second array and the feature extractor treats it as if each sample were $1/16000$ of a second apart. What happens? The STFT computes frequency bins as $k \cdot f_s / \text{n\_fft}$, but the extractor uses the *assumed* $f_s = 16000$, while the actual samples carry content placed by a $44100$ Hz clock. The effect is that **every frequency in the signal is scaled by the ratio of the rates**:

$$
\text{apparent frequency} = \text{true frequency} \times \frac{44100}{16000} = \text{true frequency} \times 2.76.
$$

A vowel whose first formant is at 700 Hz now appears at 1,932 Hz. The whole spectrum is squeezed upward (or, in the opposite direction — feeding 8 kHz audio to a 16 kHz model — stretched downward) by 2.76x. The mel bins that the encoder learned to associate with "this is the vowel /a/" now contain the energy of a completely different sound. The model is not broken; it is being asked to read a foreign language. Recognition collapses. Empirically, on a clean test set, this kind of gross rate mismatch pushes word error rate from single digits to the **60–80%** range — the transcripts become fluent hallucination, because the encoder produces *some* representation and the decoder, a strong language model, papers over the nonsense with plausible English.

There is also the silent-resample variant, which is subtler and arguably more dangerous because it does not look obviously broken. Suppose someone *did* resample, but used a low-quality method — nearest-neighbor or naive linear interpolation — without an anti-aliasing filter. Here the math is unforgiving. The Nyquist–Shannon sampling theorem says a signal sampled at rate $f_s$ can only faithfully represent frequencies below $f_s / 2$. When you downsample from 44.1 kHz to 16 kHz, the new Nyquist drops from 22,050 Hz to 8,000 Hz. Any energy in the original signal between 8 kHz and 22 kHz has nowhere to go in the new representation — and without a low-pass filter to remove it first, it **aliases**: a tone at frequency $f$ above the new Nyquist folds back to appear at $|f_s - f|$, masquerading as a lower frequency. A 10 kHz component in the original folds to $16000 - 10000 = 6000$ Hz in the downsampled signal, adding a false 6 kHz tone that was never spoken. The waveform is now at the right rate but corrupted by phantom energy, and the feature carries artifacts the encoder never saw. This is the audio analogue of resizing an image with the wrong interpolation: the rate is right, the *content* is subtly wrong. The fix is to always low-pass filter to below the new Nyquist *before* decimating — which a proper resampler (windowed-sinc, polyphase) does automatically and a naive `array[::3]` decimation does not.

The asymmetry is worth internalizing: upsampling (16 kHz to 44.1 kHz) is benign because you are not discarding bandwidth, only interpolating new sample points; the worst it does is introduce minor interpolation ripple. Downsampling is where aliasing lurks, because you are throwing away half the frequency axis and must filter first. So the dangerous direction is *high-rate audio fed to a low-rate model* — exactly the common case, since recording equipment defaults to 44.1 or 48 kHz and speech models want 16 kHz. When you see a resampling-related bug, it is almost always a downsample done without the anti-aliasing filter.

![Before and after diagram contrasting a pipeline that feeds 44.1 kHz audio to a 16 kHz model so all content is squeezed up by 2.76x and word error rate is 78 percent, against a pipeline that resamples to 16 kHz so formants land in the mel bins the encoder learned and word error rate returns to 6.5 percent](/imgs/blogs/audio-input-bugs-2.png)

### 3.2 The diagnostic: print the sample rate, do not trust the loader

You cannot confirm a sample rate by reading the loader code, because the bug is usually a missing resample line, and a missing line is invisible. The only reliable confirmation is to **print the sample rate of every clip and assert it matches the model's expected rate.** Here is the reusable check using `torchaudio`:

```python
import torchaudio

EXPECTED_SR = 16000  # Whisper, wav2vec2 base, most ASR models

def load_and_check(path, expected_sr=EXPECTED_SR):
    waveform, sr = torchaudio.load(path)   # waveform: [channels, samples]
    print(f"{path}")
    print(f"  sample rate : {sr} Hz   (model expects {expected_sr})")
    print(f"  shape       : {tuple(waveform.shape)}  [channels, samples]")
    print(f"  duration    : {waveform.shape[-1] / sr:.2f} s")
    print(f"  dtype       : {waveform.dtype}")
    print(f"  amplitude   : min {waveform.min():.3f}  max {waveform.max():.3f}")
    if sr != expected_sr:
        print(f"  >>> SAMPLE-RATE MISMATCH: resample {sr} -> {expected_sr}")
    return waveform, sr

wav, sr = load_and_check("data/call_0001.wav")
```

When you run this across your dataset and see `sample rate : 44100 Hz (model expects 16000)`, you have found the bug in one line. The amplitude print also catches the int16-versus-float bug we will get to in Section 7 — if `max` is `32767` instead of something near `1.0`, the audio is unscaled integer data.

### 3.3 The fix: resample with a proper anti-aliasing kernel

The fix is to resample to the expected rate using a method with an anti-aliasing filter. `torchaudio.transforms.Resample` does this correctly with a windowed-sinc kernel:

```python
import torchaudio
import torchaudio.functional as F

def to_expected_rate(waveform, sr, expected_sr=16000):
    if sr == expected_sr:
        return waveform, sr
    # Resample uses a windowed-sinc kernel with anti-aliasing by default.
    waveform = F.resample(waveform, orig_freq=sr, new_freq=expected_sr,
                          lowpass_filter_width=64,        # sharper anti-alias
                          rolloff=0.99, resampling_method="sinc_interp_kaiser")
    return waveform, expected_sr

wav, sr = torchaudio.load("data/call_0001.wav")
wav, sr = to_expected_rate(wav, sr)   # now guaranteed 16 kHz, anti-aliased
```

The cleaner habit, though, is to never resample by hand at all in the Hugging Face path — let the `FeatureExtractor` know the input rate and have it handle resampling, which we cover in Section 5. The single most important guardrail is the assert: refuse to train or infer on audio whose sample rate you have not verified equals the model's expected rate.

#### Worked example: a 44.1 kHz dataset fed to Whisper

A team finetunes Whisper-small on a podcast corpus distributed as 44.1 kHz MP3s. They decode with a library that returns the native rate, build features, and train. The training loss falls normally from `1.8` to `0.5` over three epochs — the model is genuinely learning *something*, namely how to map the squeezed-up features to the reference text in the training set, which is why the loss curve looks healthy. On the held-out set, word error rate is `78%`: fluent gibberish. The on-call engineer adds the `load_and_check` print and immediately sees `sample rate : 44100 Hz (model expects 16000)`. They insert `F.resample` to 16 kHz before feature extraction and retrain. Word error rate drops to `6.5%` — a **71.5-point** improvement, from one missing resample line. The lesson: the smooth training loss was the lie; the model learned to read a feature nobody else would ever produce. Confirming the rate would have taken thirty seconds and saved a multi-hour run on the wrong features.

## 4. Bug two: mel-spectrogram parameter disagreement

Once the sample rate is right, the next place the feature diverges is the spectrogram parameters themselves. This is the subtler, quiet-failure bug, and it is endemic because there are so many parameters and so many libraries that default them differently.

### 4.1 The science: every parameter changes the feature the encoder reads

Recall the chain from Section 2. The feature the encoder consumes is fully determined by: `n_fft`, `win_length`, `hop_length`, `n_mels`, `fmin`, `fmax`, power-vs-magnitude, and the log convention. The pretrained encoder's weights are fit to features with one specific setting of all eight. Whisper's spec, which is a useful anchor because it is precisely documented, is: **16 kHz, `n_fft = 400`, `hop_length = 160`, a 25 ms Hann window, `n_mels = 80`** (128 for large-v3), magnitude-then-power with an 80 dB dynamic-range clamp, and the `(log_spec + 4) / 4` rescale. wav2vec2 is different — it consumes the **raw normalized waveform** directly, no mel-spectrogram at all, which is itself a common source of confusion when people try to hand it a spectrogram.

To make the "match the spec exactly" advice concrete, here is the feature contract for the three model families you are most likely to finetune. Notice how different they are — the wav2vec2 row has no mel parameters at all, which is the single biggest source of cross-family confusion:

| Parameter | Whisper (small/base) | Whisper large-v3 | wav2vec2 / HuBERT |
| --- | --- | --- | --- |
| Input to model | log-mel spectrogram | log-mel spectrogram | raw normalized waveform |
| Sample rate | 16 kHz | 16 kHz | 16 kHz |
| n_mels | 80 | 128 | none (no mel) |
| n_fft / window | 400 / 25 ms Hann | 400 / 25 ms Hann | none |
| hop_length | 160 (10 ms) | 160 (10 ms) | none |
| Fixed length | 30 s (3000 frames) | 30 s (3000 frames) | variable, pad + mask |
| Normalization | log clamp + (x+4)/4 | log clamp + (x+4)/4 | per-utterance zero-mean unit-var |

The table makes the cross-family trap obvious: hand a log-mel feature to wav2vec2 and it is wrong by construction, because wav2vec2 never consumes a spectrogram. Mix the 80-mel small extractor with the 128-mel large-v3 model and the dimensionality disagrees. The only safe move is to load the extractor from the same checkpoint string as the model, every time.

Why does each parameter matter quantitatively? Two illustrative cases:

- **Wrong `n_mels`.** If you compute 128 mels and the encoder expects 80, the feature has the wrong dimensionality. In the best case it crashes at the input projection (a loud, easy bug). In the worse case some glue code reshapes or truncates it, and the encoder reads 80 channels that are not the 80 it learned — every channel is now a different slice of the mel axis. This is the `41%` WER case in Figure 5.
- **Wrong `hop_length`.** If you use `hop = 256` instead of `160`, you produce features at the wrong frame rate — fewer frames per second of audio. Whisper internally assumes 100 frames per second and pads or truncates to exactly 3,000 frames (30 seconds). A wrong hop produces the wrong number of frames per second, so a 5-second utterance occupies a different fraction of the 3,000-frame window than the encoder expects, and the temporal positions no longer align with the positional encodings.

The general principle: the feature pipeline is a contract, and the encoder is the other party. Break any clause and you breach the contract silently.

![Before and after diagram contrasting a pipeline that recomputes features with a 1024-point FFT and 128 mels producing a feature of the wrong shape and resolution with word error rate 41 percent, against a pipeline that uses Whisper's exact spec of a 400-point FFT, 80 mels, 160-sample hop, and 25 millisecond window, producing the feature shape the encoder learned with word error rate 7.2 percent](/imgs/blogs/audio-input-bugs-5.png)

### 4.2 The diagnostic: compute the feature and diff it against the reference

The decisive test is to compute the feature yourself with `torchaudio` *and* with the model's own `FeatureExtractor`, then diff them. If they disagree, your hand-rolled parameters are wrong. Here is the comparison for Whisper:

```python
import torch
import torchaudio
from transformers import WhisperFeatureExtractor

# --- Reference: the model's own feature extractor (the ground truth) ---
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
print("Whisper expects:")
print(f"  sampling_rate     : {fe.sampling_rate}")     # 16000
print(f"  n_fft             : {fe.n_fft}")             # 400
print(f"  hop_length        : {fe.hop_length}")        # 160
print(f"  feature_size      : {fe.feature_size}")      # 80  (n_mels)
print(f"  chunk_length      : {fe.chunk_length}")      # 30 seconds

wav, sr = torchaudio.load("data/call_0001.wav")
wav = torchaudio.functional.resample(wav, sr, 16000).mean(0)  # mono, 16 kHz
ref = fe(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_features
print("reference feature shape:", tuple(ref.shape))   # (1, 80, 3000)

# --- Your hand-rolled mel (deliberately wrong here, to show the diff) ---
mine = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128,  # WRONG
)(wav)
mine = torch.log(mine + 1e-6)
print("homemade feature shape :", tuple(mine.shape))   # (128, ...) -- mismatch!
```

The shapes alone — `(80, 3000)` versus `(128, T)` — reveal the disagreement instantly. When the shapes *match* but the values differ, the cheapest decisive check is the value range and the mean: print `ref.mean()`, `ref.std()`, `mine.mean()`, `mine.std()` and compare. A different log base shows up as a constant scale factor; a power-vs-magnitude difference shows up as a factor of 2 in the log domain.

### 4.3 The fix: use the model's FeatureExtractor, do not reimplement it

The fix for mel-parameter disagreement is almost never "carefully match all eight parameters by hand." It is "stop hand-rolling the feature and use the extractor that ships with the checkpoint." Every Hugging Face speech model has a paired `FeatureExtractor` (or a `Processor` that wraps it together with the tokenizer) whose job is to produce *exactly* the feature the model was trained on. Use it:

```python
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def make_features(wav_16k_mono):
    # processor.feature_extractor handles n_fft, hop, n_mels, log, pad-to-30s.
    return processor.feature_extractor(
        wav_16k_mono, sampling_rate=16000, return_tensors="pt"
    ).input_features  # (1, 80, 3000) guaranteed to match the model
```

We will return to this `FeatureExtractor != model` pairing as its own bug class in Section 6, because using the *wrong model's* extractor is its own trap.

## 5. Bug three: normalization, the silent off-scale feature

Even with the right rate and the right mel parameters, the feature can still be on the wrong *scale* if normalization disagrees. Normalization is the step most often forgotten, because the feature "looks fine" without it — it is on a plausible numeric range, just not the one the model expects.

### 5.1 The science: why off-scale features hurt a pretrained encoder

A neural network's first layer is a linear map followed by a nonlinearity (and usually a LayerNorm for transformers). The weights of that layer were fit assuming inputs with a particular mean and variance. If the pretrained encoder was trained on log-mel features that were normalized to roughly zero mean and unit variance — via **cepstral mean and variance normalization (CMVN)**, the standard for speech features — and you feed it raw log-mel features with a mean of `-4.1` and a standard deviation of `2.7`, every neuron sees an input shifted off the range its weights expect. For models with a leading LayerNorm (Whisper), part of this is absorbed, but the per-channel statistics still differ, and the model lands a few points worse. For models *without* a leading normalization, the damage is larger.

There are two common normalization schemes, and confusing them is a bug:

- **Per-utterance CMVN** — compute the mean and variance of *this clip's* features and normalize by them. Simple, no external state, but the normalization differs per clip.
- **Global CMVN** — compute the mean and variance over the *entire training corpus* once, store them, and apply the same fixed statistics to every clip at train and inference time. This is what most production ASR pipelines use, and the statistics must be saved and shipped with the model.

The bug is a mismatch: training with global CMVN and inference with per-utterance CMVN (or none), or computing global statistics on the training set and forgetting to apply them at inference. wav2vec2's feature extractor, notably, applies per-utterance zero-mean-unit-variance normalization to the *raw waveform* (controlled by `do_normalize=True`); skip it and the model degrades.

### 5.2 The diagnostic: print the feature statistics

The check is to print the mean and standard deviation of the feature you actually feed the model and confirm they match what the model expects:

```python
import torch

def feature_stats(feat, name="feature"):
    print(f"{name}: shape {tuple(feat.shape)}")
    print(f"  mean {feat.mean():+.3f}  std {feat.std():.3f}")
    print(f"  min  {feat.min():+.3f}  max {feat.max():.3f}")

# wav2vec2 expects do_normalize=True on the waveform
from transformers import Wav2Vec2FeatureExtractor
fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
print("wav2vec2 do_normalize:", fe.do_normalize)   # True -- must be applied

raw = wav  # 16 kHz mono float waveform
no_norm = raw
normed = (raw - raw.mean()) / (raw.std() + 1e-7)
feature_stats(no_norm, "no normalization")   # mean far from 0
feature_stats(normed,  "per-utterance CMVN") # mean ~0, std ~1
```

If `do_normalize` is `True` on the extractor but your pipeline skipped it (because you built the input tensor by hand), you have found the bug. The fix is to apply the same normalization the extractor would.

If your model uses **global** CMVN, the statistics are part of the model and must be computed once over the training corpus and then frozen — computing them per-batch at inference is itself a bug, because the inference statistics drift from the training ones. Here is how to compute and persist global statistics correctly, accumulating in a single streaming pass so you never hold the whole corpus in memory:

```python
import torch, torchaudio

def compute_global_cmvn(file_list, feature_fn):
    # Streaming accumulation of sum and sum-of-squares per mel channel.
    n = 0
    s = None      # running sum,      shape [n_mels]
    s2 = None     # running sum of x^2, shape [n_mels]
    for path in file_list:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)                          # mono
        wav = torchaudio.functional.resample(wav, sr, 16000)
        feat = feature_fn(wav)                     # [n_mels, T]
        if s is None:
            s  = torch.zeros(feat.shape[0])
            s2 = torch.zeros(feat.shape[0])
        s  += feat.sum(dim=1)
        s2 += (feat ** 2).sum(dim=1)
        n  += feat.shape[1]
    mean = s / n
    var  = s2 / n - mean ** 2
    std  = var.clamp_min(1e-8).sqrt()
    torch.save({"mean": mean, "std": std}, "cmvn_stats.pt")  # ship with model
    return mean, std
```

The critical discipline: save `cmvn_stats.pt` alongside the checkpoint and apply *those exact* statistics at inference. The single most common normalization bug is computing global statistics during training and then forgetting to load and apply them at serve time, so inference falls back to per-utterance (or no) normalization — a train-serve skew that the [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) files under "the pipeline differs between train and inference."

![Before and after diagram contrasting raw log-mel features with mean negative 4.1 and standard deviation 2.7 that sit off the scale the encoder expects with word error rate 14.8 percent, against features after global cepstral mean and variance normalization with mean zero and standard deviation one that land where the weights expect with word error rate 6.9 percent](/imgs/blogs/audio-input-bugs-8.png)

#### Worked example: forgetting CMVN on a wav2vec2 finetune

A team finetunes `wav2vec2-base` for a phone-call domain. They write a custom collate function that loads waveforms, pads them to the batch max, and stacks them — but they build the input tensor directly and never call the feature extractor, so `do_normalize=True` is never applied. The waveforms reach the model with their native amplitude statistics: mean near zero (audio is roughly zero-mean) but standard deviation that varies wildly between loud and quiet clips. Training proceeds; the loss falls; word error rate plateaus at `14.8%`, soft and never clean. The fix is one line — route every waveform through `Wav2Vec2FeatureExtractor` (or apply the equivalent per-utterance zero-mean-unit-variance), so each clip is normalized exactly as in pretraining. Word error rate drops to `6.9%`, a **7.9-point** gain. The diagnostic that found it: printing the feature mean and standard deviation per batch and noticing they were not normalized, plus checking `fe.do_normalize` on the model's own extractor.

## 6. Bug four: the FeatureExtractor and model are not paired

Hugging Face makes it dangerously easy to load a feature extractor from one checkpoint and a model from another. When they disagree, you get a quiet feature mismatch with no error.

### 6.1 The science: the extractor encodes the model's training contract

Every speech checkpoint ships a `preprocessor_config.json` that records the exact feature parameters the model was trained on — sample rate, `n_mels`, `n_fft`, hop, normalization flags. The `FeatureExtractor` reads that config and produces features to spec. If you load `WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")` (which uses **128** mels) but load the **small** model (which expects **80**), the features and the model disagree on dimensionality. Or if you load a generic `Wav2Vec2FeatureExtractor` with default `do_normalize` for a model that was trained without normalization, the scale is wrong. The extractor is not a generic utility; it is part of the model's training contract.

### 6.2 The diagnostic: assert the extractor matches the model card

The defensive habit is to load the extractor and the model from the *same* checkpoint string, and to assert the key parameters match the model's config:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

CKPT = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(CKPT)
model = WhisperForConditionalGeneration.from_pretrained(CKPT)

fe = processor.feature_extractor
# The encoder's expected mel count must equal the extractor's feature_size.
assert fe.feature_size == model.config.num_mel_bins, (
    f"FEATURE MISMATCH: extractor produces {fe.feature_size} mels, "
    f"model expects {model.config.num_mel_bins}"
)
assert fe.sampling_rate == 16000, f"unexpected sr {fe.sampling_rate}"
print(f"OK: {CKPT} extractor {fe.feature_size} mels @ {fe.sampling_rate} Hz "
      f"matches model {model.config.num_mel_bins} mels")
```

This assert turns a silent feature mismatch into a loud, immediate failure at load time — which is exactly the trade we always want when debugging: convert a quiet bug into a crash you cannot ignore. The same discipline applies to the tokenizer side, which the [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning) post covers in depth.

Here is the diagnostic table that ties the feature-stage bugs to their cheapest confirming tests — the audio instantiation of the series' master symptom-to-fix lookup.

![Matrix diagram mapping five audio failure signatures, namely huge word error rate on all files, error from a wrong log scale, failure only on quiet files, only the left channel working, and dropped words at clip edges, each to its likely feature-stage cause, the cheapest test that confirms it, and the fix direction](/imgs/blogs/audio-input-bugs-4.png)

## 7. The remaining feature-stage bugs: channels, amplitude, padding, and masks

Three more bugs round out the feature stage. Each is small to fix and easy to miss.

### 7.1 Mono versus stereo and the silent downmix

Speech models expect **mono** (single-channel) audio. Many recordings — especially anything from a video, a meeting platform, or a stereo microphone — are **stereo** (two channels), shape `[2, samples]`. If your pipeline assumes mono and indexes `waveform[0]`, you silently keep only the left channel and discard the right. On a call where the two speakers are panned to different channels, you may drop one speaker entirely. The correct downmix is to average the channels: `waveform.mean(dim=0)`. The bug is subtle because *most* of your dataset may be mono and only a stereo subset fails, producing the "only some files are garbled" signature in the table above.

```python
def to_mono(waveform):
    # waveform: [channels, samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # average, do not drop
    return waveform  # [1, samples]
```

### 7.2 Amplitude scaling: int16 versus float

WAV files store samples as 16-bit integers in the range $[-32768, 32767]$. Models expect floats in $[-1, 1]$. `torchaudio.load` returns floats already scaled to $[-1, 1]$ by default, but if you read raw bytes, use a library that returns int16, or load with `normalize=False`, you get integer amplitudes 32,768 times too large. The feature pipeline then produces enormous log-mel values, and the first training loss is `1e3` or larger — the "loss enormous at step 1" signature. The fix is to scale: `waveform = waveform.float() / 32768.0`. The diagnostic is the amplitude print from Section 3.2: if `max` is in the thousands, you have int16 data masquerading as float.

There is also **clipping** to watch for: if a recording was already at full scale and you apply gain (or a buggy normalization multiplies it up), samples saturate at $\pm 1$, flattening the waveform peaks into a square wave that adds harmonic distortion the model never saw. Check the fraction of samples at exactly $\pm 1.0$; more than a fraction of a percent is a clipping red flag.

#### Worked example: int16 amplitudes and a loss of 1,400 at step 1

A team builds a custom dataset class that reads WAV bytes with a low-level library and stacks the raw int16 arrays into a tensor, forgetting to scale. The first training step prints a loss of `1,427` — absurd, because the model and labels are fine; the *features* are enormous. They add the amplitude print from the model's-eye-view function and see `amp min -29314 max 31002` instead of values near $\pm 1$. The waveform is unscaled int16, so the log-mel features are larger than the encoder ever saw by a factor of $\log(32768) ≈ 10.4$ in the log domain per channel, and the loss explodes. The fix is one line — `waveform = waveform.float() / 32768.0` — after which the first loss is a sane `1.8` and training proceeds normally. The diagnostic took ten seconds; without the amplitude print, the team might have spent an afternoon lowering the learning rate to "fix" a loss that was never an optimization problem. This is a recurring theme of the series: a wildly wrong instrument reading (loss in the thousands) usually means a *data-scale* bug, not an optimization one, and the fastest discriminator is to print the input range before touching the learning rate.

The attention-mask bug deserves a closer look at *why* it matters numerically, because it is easy to dismiss as a formality. When you pad a batch of variable-length clips to the batch maximum, the padded region is zeros. For a model with self-attention over the time axis (Whisper's encoder, wav2vec2's transformer), every real frame attends to every other frame *including the padding*. Without a mask, the softmax over attention scores includes the pad positions, so each real frame's representation is a weighted average that is diluted by however many pad frames there are. A clip that is 5 seconds in a batch padded to 30 seconds has 25 seconds of pad — the real content is one-sixth of the sequence, and five-sixths of the attention budget can leak onto zeros. The `attention_mask` sets the pad positions' attention scores to $-\infty$ before the softmax, so they contribute exactly zero weight. Skip the mask and the encoder's pooled representation is systematically contaminated, worse for shorter clips, which is why the bug often shows up as "short utterances transcribe worse than long ones."

### 7.3 Padding, trimming, and the attention mask

Speech clips have variable length, but a batch must be rectangular, so you pad shorter clips to the batch maximum (or, for Whisper, pad/trim every clip to exactly 30 seconds). Two bugs hide here:

- **Trimming the answer.** If you truncate to a fixed length that is shorter than some utterances, you cut off the end of long clips — and the words you cut never reach the model. The transcript is missing the tail. This is the "words drop at clip edges" signature.
- **The missing attention mask.** When you pad, you must tell the model which frames are real and which are padding, via an `attention_mask` (Whisper handles the 30 s pad internally, but for variable-length models like wav2vec2 the mask is essential). Without it, the model attends to the zero-padding as if it were silence-that-means-something, and the encoder's pooled representation is contaminated by the pad. Pad tokens leaking into the computation is the audio cousin of the loss-masking bugs that plague LLM finetuning.

```python
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
batch = processor(
    [w.numpy() for w in waveforms],   # list of 1-D mono waveforms
    sampling_rate=16000,
    padding=True,                     # pad to batch max
    return_attention_mask=True,       # MUST pass the mask downstream
    return_tensors="pt",
)
# Pass BOTH input_values AND attention_mask to the model:
out = model(input_values=batch.input_values, attention_mask=batch.attention_mask)
```

The rule: whenever you pad, return and pass the attention mask, and never trim to a length shorter than your longest real utterance unless you intend to.

## 8. The unified diagnostic: the model's-eye view function

Now we assemble everything into one reusable function — the audio equivalent of "display the exact tensor the model receives." It loads a clip, runs the *exact* preprocessing the model expects, prints every statistic that can reveal a bug, and renders the spectrogram so you can compare it to a reference.

```python
import torch, torchaudio
import matplotlib.pyplot as plt
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
EXPECTED_SR = processor.feature_extractor.sampling_rate  # 16000
EXPECTED_MELS = processor.feature_extractor.feature_size  # 80

def model_eye_view(path):
    wav, sr = torchaudio.load(path)                      # [C, N], float [-1,1]
    print(f"=== {path} ===")
    print(f"sr {sr}  channels {wav.shape[0]}  "
          f"samples {wav.shape[1]}  dtype {wav.dtype}")
    print(f"amp min {wav.min():+.3f}  max {wav.max():+.3f}")

    # 1) downmix to mono
    if wav.shape[0] > 1:
        print("  downmixing stereo -> mono (mean over channels)")
        wav = wav.mean(0, keepdim=True)
    # 2) resample to expected rate
    if sr != EXPECTED_SR:
        print(f"  resampling {sr} -> {EXPECTED_SR}")
        wav = torchaudio.functional.resample(wav, sr, EXPECTED_SR)
    wav = wav.squeeze(0)

    # 3) build the feature with the MODEL'S OWN extractor (the ground truth)
    feat = processor.feature_extractor(
        wav.numpy(), sampling_rate=EXPECTED_SR, return_tensors="pt"
    ).input_features                                     # (1, 80, 3000)
    print(f"feature shape {tuple(feat.shape)}  "
          f"(expect 1x{EXPECTED_MELS}x3000)")
    print(f"feature mean {feat.mean():+.3f}  std {feat.std():.3f}  "
          f"min {feat.min():+.3f}  max {feat.max():+.3f}")
    assert feat.shape[1] == EXPECTED_MELS, "MEL COUNT MISMATCH"

    # 4) render the spectrogram to compare visually to a reference clip
    plt.figure(figsize=(10, 3))
    plt.imshow(feat[0].numpy(), aspect="auto", origin="lower")
    plt.title(f"{path}  ({EXPECTED_MELS} mel x time)")
    plt.xlabel("frame (10 ms each)"); plt.ylabel("mel bin")
    plt.tight_layout(); plt.savefig("feature_view.png", dpi=110)
    return feat

feat = model_eye_view("data/call_0001.wav")
```

This single function catches the majority of the bugs in this post: wrong rate (printed), wrong channels (printed and fixed), unscaled amplitude (printed), wrong mel count (asserted), off-scale features (mean and std printed), and a renderable spectrogram you can eyeball against the model's documentation. Run it on a clip from the model's own test set first to learn what "good" looks like, then run it on your data and compare. The two spectrograms should be visually similar in scale and resolution; if yours looks washed out, banded, or wrong-shaped, the difference is your bug.

### 8.1 The round-trip test: a sine wave of known frequency

The most rigorous single check — the audio analogue of "feed a known input and verify the known output" — is to synthesize a pure tone at a known frequency, run it through your feature pipeline, and verify the energy lands in the mel bin you can compute by hand. If you generate a 1 kHz sine at 16 kHz and the energy peak is *not* in the mel bin corresponding to 1 kHz, your pipeline has a rate or filterbank bug.

```python
import torch, torchaudio

sr = 16000
t = torch.arange(0, sr) / sr            # 1 second
tone = 0.5 * torch.sin(2 * torch.pi * 1000 * t)   # pure 1 kHz tone

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=400, hop_length=160, n_mels=80,
)(tone)                                  # (80, frames)
peak_bin = mel.mean(dim=1).argmax().item()
print(f"1 kHz tone peaks at mel bin {peak_bin} of 80")
# Expected: a bin in the low-middle of the mel axis. If it peaks near the top,
# the pipeline thinks 1 kHz is a high frequency -> sample-rate bug.
```

If you fed this tone but told the pipeline `sample_rate=44100`, the peak would shift dramatically, confirming the rate handling. This test takes seconds and is decisive because the answer is computable from first principles — there is no dataset noise to hide behind.

### 8.2 Learning to read a spectrogram so the eye becomes useful

I said earlier that the eye is useless on a spectrogram, and that is true *until you train it on what correct looks like*. Once you have rendered a few good ones, several gross bugs become visible at a glance, which is why the `model_eye_view` function plots the feature. Here is the field guide for a 16 kHz, 80-mel log-spectrogram of clean speech:

- **The mel axis (vertical) should show structure in the lower two-thirds.** Speech energy concentrates below ~4 kHz, which is the lower portion of the mel axis. If your spectrogram has all its energy crammed into the *top* few mel bins, the content has been shifted up — the classic sample-rate-too-low signature, because feeding high-rate audio as if it were low-rate pushes everything toward Nyquist.
- **Horizontal bands that drift and break are formants.** Clean voiced speech shows two or three roughly horizontal energy bands (the formants) that move as the vowel changes, punctuated by vertical streaks at consonant bursts. If you see no horizontal structure at all — just uniform texture or vertical noise — the audio may be silence, noise, or a feature computed with the wrong parameters.
- **The dynamic range should span the colormap.** A correct log-mel uses most of its value range; if the whole image is one flat color, the log compression or normalization is wrong (everything clamped to one level), or the clip is silent.
- **The aspect should be right.** For Whisper, every clip pads to exactly 3,000 frames (30 seconds at 100 frames/s). A 5-second clip should show 500 frames of speech followed by 2,500 frames of padding (a flat region). If the speech fills the whole width when the clip is short, the hop length or padding is wrong.

The workflow is: render one clip from the *model's own* documentation or test set, learn what its spectrogram looks like, then render your clip beside it. The two should be visually comparable in scale, resolution, and structure. A side-by-side that looks obviously different — yours washed out, banded, squeezed into the top bins, or the wrong width — localizes the bug to the feature stage in seconds, even though no single number told you so. This is the modality-specific version of the series' core move: make the invisible visible, then compare to a known-good reference.

![Timeline diagram showing the five audio feature-audit checks in cost order, beginning with printing the sample rate, then the shape for channels and length, then the value range to catch int16 versus float, then confirming the feature extractor matches the model card, and finally plotting the spectrogram to compare against the reference spec](/imgs/blogs/audio-input-bugs-7.png)

## 9. The bisection in action: localizing the bug step by step

Let me walk the full debugging narrative the way you would actually do it, applying the series' bisection discipline. You have the call-center finetune from Section 1: smooth loss, `40%` word error rate, the base model alone managed `9%`. You have not touched a line of code yet. You bisect.

**Step 0 — make it repeatable and small.** Pick one clip from the failing eval set. Everything from here runs on that one clip in a notebook, so each test is instant. This is make-it-fail-small for audio: one clip, one feature, one forward pass.

**Step 1 — is it the rate?** Run `model_eye_view` on the clip. Read the first line. If `sr` is not `16000`, you are done — resample and re-eval. In our case it prints `sr 16000`, so the rate is clean; move on. (This one print would have caught the Section 3 worked example outright.)

**Step 2 — is it the channels or amplitude?** Same print: `channels 1`, `amp min -0.82 max 0.79`. Mono, properly scaled to $[-1, 1]$. Both clean. Move on.

**Step 3 — is it the feature parameters?** The function uses the model's own extractor, so the feature shape is `(1, 80, 3000)` — correct by construction. But your *training* pipeline might have used a hand-rolled mel. Diff the two: compute the feature your training collate produced for this clip and compare its shape and statistics to `model_eye_view`'s. Here is the tell — the training feature is `(1, 128, T)`. Someone wrote a custom `MelSpectrogram` with `n_mels=128` during data prep, and a reshape downstream forced it to 80 by truncation. The model trained on a *mangled* 80-channel slice of a 128-mel feature. That is the bug.

**Step 4 — confirm with the before/after.** Replace the hand-rolled feature with `processor.feature_extractor` in the training pipeline, retrain, re-eval. Word error rate drops from `40%` to `7.2%` (Figure 5). The fix is confirmed; the suspect is closed.

**Step 5 — stress-test.** Now ask the series' standard stress questions. *What if it were the rate instead?* The Section 3 path — resample and re-eval. *What if it were normalization?* The Section 5 path — print feature mean and std, apply CMVN. *What if only some files failed?* The channel path — check for stereo. *What if the loss had been NaN rather than just high?* That is a numerics story, not a feature story — it routes to [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs), not here. The decision tree in Figure 6 encodes exactly these branches.

Notice what made this bisection fast: at no point did we read the training code looking for the bug. Reading code is the slowest way to find an input bug, because the bug is usually a *missing* line — a resample that never happened, a normalization that was skipped, an extractor that was never called — and a missing line is invisible to a code review. Instead we *observed the data* at the point it enters the model and compared it to ground truth (the extractor's output). This is the difference between deductive debugging (reason about what the code should do) and empirical debugging (measure what the data actually is). For input-pipeline bugs, empirical always wins, because the failure lives in the gap between what the code appears to do and what it actually produces. The model's-eye-view function is the instrument that closes that gap.

One more discriminator worth internalizing for the bisection: the *shape* of the WER tells you the *class* of bug before you even open the data. A uniform catastrophic WER across all files points to a global mismatch — sample rate, mel parameters, or normalization that is wrong for every clip. A *bimodal* WER, where most files are fine and a subset is terrible, points to a property that *varies* across the dataset — channel count, source sample rate, encoding format. And a WER that is *uniformly a few points high* with no catastrophic failures points to a *subtle* mismatch — a slightly wrong log base, a double resample, a normalization that is close but not exact. Reading the distribution of the error, not just its mean, routes you to the right branch of the tree before you touch a single clip.

![Tree diagram routing a garbled transcription from the symptom through three feature stages, the rate stage tested by printing the sample rate, the feature stage tested by diffing against the reference extractor, and the channel stage tested by printing the shape, each leading to the cheapest test that confirms the wrong knob](/imgs/blogs/audio-input-bugs-6.png)

#### Worked example: bisecting a wav2vec2 run that fails on a subset

A different team finetunes wav2vec2 on meeting recordings. Overall word error rate is a decent `11%`, but a subset of about 20% of files is catastrophically wrong — `70%+` on those, fine on the rest. The smooth aggregate hides a bimodal failure. They bisect on a *failing* clip. `model_eye_view` prints `channels 2` — the failing clips are stereo, and the training collate did `waveform[0]` to "get a 1-D array," silently keeping only the left channel. On meetings where the active speaker was panned right, the left channel was near-silent, so the model transcribed noise. The fix: `waveform.mean(0)` to downmix. The subset's word error rate drops from `70%` to `12%`, and the aggregate from `11%` to `7%`. The lesson the bisection teaches: when only *part* of a dataset fails, the bug is almost always in a *property that varies across the dataset* — here, channel count — and the fastest path is to inspect a failing example, not the aggregate metric.

## 10. Before and after: the evidence table

Pulling the worked examples and figures together, here is the consolidated before→after evidence for the four headline bugs. Every number is the kind you would measure on a held-out set with a fixed text normalizer (lowercasing, punctuation stripping) so the word error rate reflects content, not formatting. These figures are illustrative of the *magnitude and direction* you should expect when you fix each bug on a real finetune; measure your own, but expect changes on this scale.

| Bug | Symptom | Confirming test | After fix | Delta |
| --- | --- | --- | --- | --- |
| Sample-rate mismatch | WER 78%, fluent gibberish | `print(sr)` shows 44100 | WER 6.5% | -71.5 pts |
| Mel-param disagreement | WER 41%, wrong words | feature shape 128 vs 80 | WER 7.2% | -33.8 pts |
| Missing normalization | WER 14.8%, soft forever | feature mean/std off | WER 6.9% | -7.9 pts |
| Stereo kept-left only | 20% subset at 70% WER | `print(channels)` shows 2 | subset WER 12% | -58 pts (subset) |

How would you *confirm* these honestly rather than just trusting the table? Two ways, both cheap. First, the held-out word error rate before and after the one-line fix, with everything else held constant and the same random seed — the series' [reproducibility discipline](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) makes this an apples-to-apples comparison. Second, the feature diff itself: print the mean, std, and shape of the feature before and after, and show they moved from "disagrees with the extractor" to "matches the extractor." The feature-level proof is stronger than the metric-level proof, because it confirms you fixed the *cause*, not that the metric happened to move.

## 11. Case studies and real signatures

A few patterns are worth naming because they recur across teams and tools.

**Whisper's exact spec as the canonical anchor.** Whisper (Radford et al., OpenAI, 2022) is unusually well-documented in its feature spec, which makes it the reference everyone should check against: 16 kHz, 80 mel channels (128 for large-v3), a 400-point FFT with a 25 ms Hann window, a 160-sample (10 ms) hop, and a fixed 30-second (3,000-frame) input window with a `(log_spec + 4) / 4` rescale. The most common Whisper finetuning bug in the wild is feeding non-16-kHz audio; the second most common is using the wrong checkpoint's mel count (80 vs 128) when mixing model sizes. The `WhisperFeatureExtractor` exists precisely so you never reimplement this — and the recurring lesson is that people reimplement it anyway and get a parameter wrong.

**wav2vec2 consumes the raw waveform, not a spectrogram.** wav2vec2 (Baevski et al., 2020) and HuBERT learn features from the raw normalized waveform via a convolutional feature encoder; there is no mel-spectrogram in the input path at all. A frequent confusion is to hand these models a log-mel feature because "that's how speech models work" — it is not how *these* work, and the result is garbage. The matching bug is forgetting `do_normalize=True` on the waveform, which we covered in Section 5.

**The librosa-versus-torchaudio default drift.** `librosa.feature.melspectrogram` and `torchaudio.transforms.MelSpectrogram` default several parameters differently — notably the mel-scale formula (`librosa` defaults to the "slaney" mel scale and area-normalized filters; `torchaudio` defaults differ), the power exponent, and the windowing. Compute the "same" feature in both libraries and you can get visibly different arrays. If your data prep used `librosa` and the model's extractor uses a different convention, you have a quiet feature mismatch even with identical `n_fft`, `hop`, and `n_mels`. The defense is the same: do not hand-roll; diff against the model's extractor.

**The WER normalizer that lies.** A subtler "feature" bug lives in evaluation: word error rate is enormously sensitive to text normalization. If your reference transcripts keep punctuation and casing but the model outputs lowercase unpunctuated text (or vice versa), the WER can be inflated by tens of points for purely formatting reasons, masking a model that is actually fine — or hiding a model that is actually broken. This is the audio cousin of "your metric is lying," and the [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning) post treats it in full. The point here: when WER looks bad, confirm the *feature* is right before you conclude the *model* is broken, and confirm the *normalizer* is right before you conclude either.

**The 8 kHz telephony trap.** A specific, common variant deserves its own mention because it bites teams working with real call data. Traditional telephony is sampled at 8 kHz (narrowband), so audio from phone systems, old call recordings, and some VoIP pipelines arrives at 8 kHz with content only up to 4 kHz. Whisper expects 16 kHz. Naively upsampling 8 kHz to 16 kHz is *safe* in the aliasing sense (you are not discarding bandwidth), but it does not add the missing 4–8 kHz content — that information was never recorded. The model trained on full-band 16 kHz audio now sees clips with an empty upper half of the spectrum, a distribution it underperforms on. This is not strictly a *bug* in your pipeline (you did the resample correctly) but a *data-distribution* issue you must recognize: telephony audio is fundamentally band-limited, and the fix is either to finetune on telephony-band audio or to accept the degradation, not to keep hunting for a feature mismatch that is not there. Recognizing when the "bug" is actually the data's nature is part of knowing when to stop debugging.

**The double-resample.** A pipeline can resample *twice* — once in the loader (to 16 kHz) and again inside the `FeatureExtractor` (which also resamples if you pass the original rate). Each resample is a lossy filtering operation, so two of them stack two low-pass filters and soften the audio more than the model saw. The tell is that everything *looks* right — rate is 16 kHz, feature shape correct — but WER is mysteriously a point or two high. The fix is to resample exactly once: either in the loader, then tell the extractor the audio is already at 16 kHz, or hand the extractor the original rate and let *it* do the single resample. Never both.

## 12. When this is (and isn't) your bug

A decisive section, because the worst outcome is to spend a day in the feature pipeline when the bug lives elsewhere.

**It IS a feature bug when:** the base model worked and your finetune does not on the same audio; the loss falls smoothly but transcripts are wrong; the failure is *systematic* across all files (or a clean subset defined by a file property like channel count or sample rate); printing the sample rate, shape, range, or feature statistics reveals a value that disagrees with the model's extractor. The signature is "the optimizer is healthy but the input is foreign."

**It is NOT a feature bug — look elsewhere — when:**

- **The loss is NaN or Inf, not just high.** A smooth-then-NaN curve is numerics, not features. Route to [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) and [mixed-precision debugging](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16). A feature bug produces *wrong* numbers, not *invalid* ones.
- **CTC loss is `inf` at step 1.** That is the input-shorter-than-target trap, a property of the CTC objective and your label lengths, not the spectrogram. Route to [debugging CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment).
- **The feature matches the extractor exactly but WER is still high.** Then the feature is *not* your bug. Suspect the tokenizer, the language/task tokens, the decoder prompt, an over-aggressive SpecAugment, or the learning rate — all covered in [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning).
- **Training works, only streaming inference breaks.** That is a train-infer mismatch (chunking, lookahead, missing normalization stats at serve time), not a static feature bug.
- **`overfit-one-batch` fails.** If the model cannot drive loss to near-zero on a single batch of *correctly-featured* audio, the bug is in the model or optimization, not the input. Run that test first, exactly as the [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) prescribes.

The cleanest single discriminator: **does the feature you feed the model match the feature its own `FeatureExtractor` produces?** If yes, stop blaming the input and move to the model, optimization, tokenizer, or eval. If no, you have found your bug and you can stop looking.

## 13. Key takeaways

- **Print the sample rate of every clip and assert it equals the model's expected rate.** A wrong rate scales every frequency by the rate ratio (44.1 kHz into a 16 kHz model squeezes content up 2.76x) and produces fluent gibberish at `60–80%` WER. One print, one assert, bug gone.
- **Do not hand-roll the mel-spectrogram — use the model's `FeatureExtractor`.** `n_fft`, `hop_length`, `win_length`, `n_mels`, `fmin`/`fmax`, power-vs-magnitude, and the log base are a contract the encoder's weights depend on. Whisper is 16 kHz / 80 mels / 400 FFT / 160 hop / 25 ms window; reproduce it by *calling the extractor*, not by matching eight parameters by hand.
- **Print the feature mean and std and confirm normalization matches training.** Off-scale features (raw log-mel with mean `-4.1` instead of CMVN'd to zero-mean-unit-variance) cost several points of WER. Check `do_normalize` on the extractor.
- **Assert the `FeatureExtractor` and model came from the same checkpoint.** `assert fe.feature_size == model.config.num_mel_bins` turns a silent dimensionality mismatch into a load-time crash.
- **Downmix stereo to mono by averaging, never by `waveform[0]`.** Keeping one channel drops a panned speaker and produces a failing *subset*, which hides in the aggregate metric. When part of a dataset fails, inspect a failing example.
- **Scale int16 to float `[-1, 1]` and watch the amplitude print.** A `max` of `32767` means unscaled integers, and the first training loss will be `1e3+`.
- **When you pad, pass the attention mask; never trim shorter than your longest utterance.** A missing mask leaks padding into the encoder; over-trimming cuts off the tail of long clips.
- **Build the model's-eye-view function and the sine-wave round-trip test.** Render the spectrogram and compare to a reference; feed a 1 kHz tone and verify the energy lands in the bin you computed by hand. The feature is not human-legible, so make it machine-comparable.
- **Confirm the feature, not just the metric.** A NaN loss is numerics; an `inf` CTC loss is the length constraint; a feature that matches the extractor but still gives high WER points to the tokenizer or eval — not the spectrogram.

## 14. Further reading

- **"Robust Speech Recognition via Large-Scale Weak Supervision"** — Radford, Kim, Xu, Brockman, McLeavey, Sutskever (OpenAI, 2022). The Whisper paper; the canonical feature spec (16 kHz, 80-mel log-magnitude, 30-second window) every finetune should match.
- **"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"** — Baevski, Zhou, Mohamed, Auli (2020). Why these models consume the raw normalized waveform, not a mel-spectrogram.
- **torchaudio documentation** — `torchaudio.transforms.MelSpectrogram`, `torchaudio.transforms.Resample`, and `torchaudio.functional.resample`; the parameter defaults that drift between libraries.
- **Hugging Face `transformers` audio docs** — `WhisperFeatureExtractor`, `WhisperProcessor`, `Wav2Vec2FeatureExtractor`, `Wav2Vec2Processor`; the `preprocessor_config.json` that encodes each model's feature contract.
- **librosa documentation** — `librosa.feature.melspectrogram` and the Slaney-vs-HTK mel-scale conventions, the most common source of silent feature drift versus torchaudio.
- **Within this series:** [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) (the master symptom→suspect→test→fix frame), [the input pipeline is lying to you](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) and [computer vision input-pipeline bugs](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs) (the print-the-batch discipline in other modalities), [debugging CTC and alignment](/blog/machine-learning/debugging-training/debugging-ctc-and-alignment) and [debugging ASR finetuning](/blog/machine-learning/debugging-training/debugging-asr-finetuning) (the next steps once the feature is correct), and the capstone [training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).
