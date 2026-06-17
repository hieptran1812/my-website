---
title: "Representing Sound: Waveforms, Spectrograms, and the Mel Scale"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build a precise mental model of every representation an audio model stands on — the raw waveform, the STFT spectrogram, and the mel-spectrogram — and learn why the lossy, phase-free mel is the workhorse for speech and music."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "spectrogram",
    "mel-spectrogram",
    "signal-processing",
    "psychoacoustics",
    "text-to-speech",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-1.png"
---

The first time I trained a text-to-speech model, I made a beginner mistake that cost me a week. I fed the network raw 24 kHz audio and asked it to predict the next sample, WaveNet-style, and watched the loss plateau into a soft hiss that sounded like someone whispering through a pillow. The model had millions of parameters and learned almost nothing about speech, because I had handed it the wrong representation. When I switched to predicting an 80-band mel-spectrogram and handed the waveform reconstruction to a separate vocoder, the same model size produced crisp, intelligible speech within a day. The architecture barely changed. The representation changed everything.

That is the lesson this entire post is built around: **before you choose a model, you choose how you will represent the sound, and that choice quietly decides how hard the modeling problem is.** A one-second clip of CD-quality music is 44,100 floating-point numbers — a sequence longer than most language models ever see in a single context window, with no explicit notion of pitch, timbre, or rhythm written anywhere in it. The same second of audio, turned into a mel-spectrogram, is roughly 86 frames of 80 numbers each: about 100 times smaller, and organized exactly the way your ear organizes sound, into frequency bands over time. The catch is that the compact, convenient representation is *lossy* — it throws away the phase of the signal — so you need a separate model, a **vocoder**, to turn it back into something you can play. Every modern audio generator lives somewhere on this ladder, and understanding the ladder is the prerequisite for understanding all of them.

![A vertical stack showing the audio representation ladder from raw waveform through the STFT spectrogram and mel-spectrogram into a generative model and a vocoder that rebuilds a playable waveform.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-1.png)

By the end of this post you will be able to: load a `.wav` file and inspect its sample rate and bit depth; compute a short-time Fourier transform (STFT) and a mel-spectrogram with `torchaudio` and `librosa`, and explain every parameter you pass (`n_fft`, `hop_length`, `win_length`, `n_mels`); reason quantitatively about the time-frequency resolution trade-off using the uncertainty principle; explain *why* a magnitude mel-spectrogram is non-invertible and what Griffin-Lim does to fix that; invert a mel back to audio in code and hear how it degrades; resample between 16 kHz speech and 44.1 kHz music; and pick the right representation for a given audio model. This is post A2 in the series — it sits right on top of the foundation, [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), and it sets up everything downstream, especially the codec posts. If you want the deeper DSP derivations (the full Fourier transform, convolution, the sampling theorem), the sibling post [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals) is where that machinery is built; here I will derive only what we need to make representation choices and link out where it earns its place.

## 1. The waveform: what a sound actually is

Strip away every layer of structure and a digital sound is a list of numbers. A microphone measures air pressure many thousands of times per second; an analog-to-digital converter records each measurement as a number; play that list back through a speaker at the same rate and you reproduce the pressure changes that hit the microphone. That list is the **waveform**: amplitude as a function of time.

Two parameters define the grid the waveform lives on. The **sample rate** is how many measurements you take per second, in hertz (Hz). CD audio is 44,100 Hz (44.1 kHz). Most speech models use 16 kHz or 22.05 kHz; modern high-fidelity TTS and music models use 24 kHz, 44.1 kHz, or 48 kHz. The **bit depth** is how many bits encode each sample — 16-bit PCM gives 65,536 distinct amplitude levels, 24-bit gives about 16.7 million. A standard CD is "16-bit, 44.1 kHz, stereo," which is two channels of 16-bit samples at 44,100 Hz.

The sample rate is not arbitrary. The **Nyquist-Shannon sampling theorem** says that to represent every frequency up to some maximum $f_{\max}$ without ambiguity, you must sample at a rate of at least $2 f_{\max}$. Turn it around: a sample rate of $f_s$ can faithfully represent frequencies only up to the **Nyquist frequency** $f_s/2$. At 44.1 kHz the Nyquist frequency is 22,050 Hz, comfortably above the roughly 20 kHz ceiling of healthy young human hearing — that is precisely why CD audio chose 44.1 kHz. At 16 kHz the Nyquist frequency is 8 kHz, which is fine for telephone-band speech (most of the energy and intelligibility of speech sits below 8 kHz) but audibly dull for music, which has cymbals, harmonics, and air above 8 kHz. If you sample a 12 kHz tone at 16 kHz, it does not simply disappear; it **aliases** down to a false 4 kHz tone and corrupts the signal, which is why resamplers apply a low-pass anti-aliasing filter before downsampling. I will not re-derive the sampling theorem here — that is the job of [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals) — but the practical consequence is the through-line of this whole section: **your sample rate is a hard ceiling on the frequencies your model can ever produce.**

Bit depth controls a different kind of fidelity: **quantization noise**. Rounding each continuous pressure value to the nearest of $2^b$ levels introduces a small error, and the theoretical signal-to-quantization-noise ratio for a full-scale signal is approximately

$$\text{SNR}_{\text{dB}} \approx 6.02\,b + 1.76,$$

so 16-bit audio has about 98 dB of dynamic range and 24-bit about 146 dB. For generative modeling, bit depth matters less than you might think, because we almost never model raw integer PCM — we convert to 32-bit floats normalized to $[-1, 1]$ and let the model and loss work in continuous space. The number that bites you is the sample *count*, not the bit depth.

There is one subtlety worth knowing, because it shows up when you quantize audio for storage or for a discrete model. Quantization error is *signal-dependent* — for a quiet passage the rounding error correlates with the signal and produces audible distortion rather than benign hiss. The classic DSP fix is **dithering**: add a tiny amount of random noise before quantizing, which decorrelates the error from the signal and turns ugly distortion into a faint, constant, far less objectionable noise floor. You will not dither inside a normalized-float training pipeline, but the principle reappears in generative audio: the WaveNet lineage modeled audio not as a regression to a float but as a *classification* over 256 quantized amplitude levels, using a **µ-law companding** quantization that allocates more levels to quiet signals (where the ear is sensitive) and fewer to loud ones — a perceptual quantization, the time-domain cousin of the mel's perceptual frequency warping. So even bit depth, the parameter that seems least relevant to modeling, carries a perceptual lesson: where you put your quantization levels should follow where the ear listens.

#### Worked example: the length problem in numbers

Take a 4-minute song at 44.1 kHz stereo, the kind of thing Suno or Udio produces. That is $240 \text{ s} \times 44{,}100 \text{ Hz} \times 2 \text{ channels} = 21{,}168{,}000$ samples. Twenty-one million numbers. If you tried to model that autoregressively, predicting one 16-bit sample at a time the way the original WaveNet did, you would need to roll the model forward 21 million times for a single song, and you would need a receptive field spanning seconds to capture musical structure like a chorus that repeats every 30 seconds — that is over a million samples of context. This is the brutal arithmetic behind the [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) post: audio is a 1D signal that is simultaneously enormous in length and rich in fine detail, and the raw waveform exposes *neither* the long-range structure (where is the chorus?) *nor* the local structure (what pitch is this?) in a form a model can use directly. There is no axis in the raw sample stream that says "this is 440 Hz." Frequency is implicit, smeared across many samples. That single fact — frequency is not explicit in the waveform — is the reason the rest of this post exists.

Let me make the waveform concrete in code. We will use this same loaded clip for everything that follows.

```python
import torch
import torchaudio

# Load any wav. torchaudio returns a float tensor in [-1, 1] and the sample rate.
waveform, sample_rate = torchaudio.load("speech.wav")  # shape: (channels, num_samples)
print(waveform.shape, sample_rate)
# e.g. torch.Size([1, 32000]) 16000  -> 2.0 seconds of mono 16 kHz speech

# Collapse to mono if needed; most audio models are mono.
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

duration_sec = waveform.shape[1] / sample_rate
print(f"{duration_sec:.2f} s, {waveform.shape[1]} samples, "
      f"peak amplitude {waveform.abs().max():.3f}")
```

The waveform tensor is exact: it *is* the signal, nothing has been lost. But it is also the least useful representation for a generative model, because everything we care about — pitch, formants, rhythm, timbre — is encoded only implicitly in the relationships between millions of nearby samples. We need to make frequency explicit. That is what the Fourier transform does.

## 2. The STFT and the spectrogram: making frequency explicit

A pure tone is a sine wave: $x(t) = A \sin(2\pi f t + \phi)$, with amplitude $A$, frequency $f$, and phase $\phi$. The **Fourier transform** is the mathematical statement that *any* signal can be written as a sum (or integral) of such sines and cosines at different frequencies, and it tells you the amplitude and phase of each one. If you Fourier-transform a clip of a violin playing A4, you get a big spike at 440 Hz (the fundamental) and smaller spikes at 880, 1320, 1760 Hz and so on (the harmonics) — the recipe of frequencies that makes a violin sound like a violin and not a flute. The full continuous transform is the subject of the [math sibling post](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals); here we need the *discrete, windowed* version that audio practice actually uses.

The plain Fourier transform has a fatal flaw for audio: it tells you *which* frequencies are present over the *whole* signal, but not *when*. A song that plays a C-major chord for the first half and an A-minor chord for the second half has the same global Fourier spectrum as one that plays them in the opposite order. We hear music as frequencies that change over time, so we need a representation that keeps both axes. The fix is the **Short-Time Fourier Transform (STFT)**: slide a short window along the signal, and take a Fourier transform of each windowed chunk. Each transform tells you the frequency content *during that little window of time*. Stack the results side by side and you get a 2D array — frequency on one axis, time on the other — called the **spectrogram**.

![A graph showing the STFT splitting each framed window into a complex spectrum whose magnitude is kept and whose phase is discarded before mel filtering and a log.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-4.png)

Concretely, the STFT of a discrete signal $x[n]$ is

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N},$$

where $w[n]$ is a **window function** of length $N$ (the `win_length`, usually equal to `n_fft`, often a Hann window), $H$ is the **hop length** (how far you slide the window between frames), $m$ indexes the time frame, and $k$ indexes the frequency bin from $0$ to $N/2$. Each $X[m,k]$ is a **complex number**. Its magnitude $|X[m,k]|$ is how much energy frequency-bin $k$ had during frame $m$; its angle $\angle X[m,k]$ is the **phase**, the alignment of that frequency's wave at the start of the frame. The **spectrogram** is usually the magnitude (or squared magnitude, the power) — but note that the STFT itself carries both magnitude and phase, and we will agonize over phase later because it is exactly what the mel-spectrogram drops.

Three parameters control everything:

- **`n_fft`** (the FFT size, equal to the window length $N$ unless you zero-pad): how many samples go into each Fourier transform. Larger means more frequency bins ($N/2 + 1$ of them) and finer frequency resolution, but a longer window in time. Typical: 1024 or 2048 for 22 kHz audio.
- **`hop_length`** ($H$): how many samples you advance between frames. Smaller hop means more frames (more time resolution and more overlap), at higher compute and storage cost. Typical: `n_fft // 4`, i.e. 75% overlap.
- **`win_length`** and **window shape**: the window tapers each chunk to zero at its edges so the chunk's edges do not create spurious high-frequency content (spectral leakage). A Hann window is the default workhorse.

It is worth being precise about *why* the window matters, because skipping it is a classic beginner mistake. If you chop the signal into chunks with a rectangular window (just slicing, no taper), each chunk has hard discontinuities at its edges, and a discontinuity is broadband — it injects energy across all frequencies. The Fourier transform of a rectangular window is a sinc function with large side lobes, so a single pure tone smears its energy into many neighboring bins, an artifact called **spectral leakage**. A tapered window like the Hann ($w[n] = 0.5 - 0.5\cos(2\pi n / (N-1))$) goes smoothly to zero at both edges, killing the discontinuity and suppressing the side lobes, so a pure tone stays concentrated near its true bin. The trade is a slightly wider main lobe (a hair less frequency resolution) for vastly lower leakage — almost always the right trade. There is one more constraint that ties the window to the hop: for the STFT to be invertible by overlap-add, the windows at successive hops must sum to a constant across time (the "constant overlap-add," or COLA, condition). A Hann window with 75% overlap (`hop_length = n_fft // 4`) satisfies COLA, which is exactly why that overlap is the near-universal default — it is not arbitrary, it is the setting that lets you reconstruct the waveform cleanly from the complex spectrogram.

The number of frequency bins is $1 + n\_fft/2$ (the redundant negative frequencies are dropped for real signals). The number of time frames is roughly $\text{num\_samples} / H$. Let me ground that in code.

```python
import torchaudio.transforms as T

n_fft = 1024
hop_length = 256          # 75% overlap
win_length = 1024

spectrogram_fn = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    power=2.0,            # 2.0 -> power spectrogram; None -> complex (keeps phase)
)
power_spec = spectrogram_fn(waveform)     # (1, n_fft//2 + 1, num_frames)
print(power_spec.shape)                   # e.g. (1, 513, 126)

# To keep phase, ask for the complex STFT (power=None):
complex_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)(waveform)
magnitude = complex_spec.abs()            # (1, 513, 126) real
phase = complex_spec.angle()              # (1, 513, 126) real, in radians
```

For our 2-second, 32,000-sample clip with `hop_length=256`, we get about $32000/256 \approx 126$ frames, each with $513$ frequency bins. So the linear spectrogram is $513 \times 126 \approx 64{,}600$ numbers — *bigger* than the 32,000-sample waveform, because of the overlap. That is the first surprise: the STFT spectrogram is not a compression. Its value is not size; its value is that it makes frequency **explicit**. Now a model can "see" that frame 40 has strong energy at bin 14 (around 440 Hz) and read pitch directly, instead of inferring it from sample-to-sample differences.

A spectrogram is best pictured as a literal grid you can read like a heatmap.

![A three by three grid where rows are frequency bands and columns are time frames, with low bands holding steady pitch energy and a high band lighting up only on a hiss frame.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-2.png)

Read a single column and you know which frequencies were loud at that instant; read a single row across time and you trace one frequency band's energy as it rises and falls. A sustained vowel shows up as horizontal stripes (the steady harmonics of the voiced pitch); a consonant like /s/ shows up as a vertical smear of energy in the high bands (broadband noise, short in time); a drum hit is a bright vertical line (energy at all frequencies for one instant). This is why spectrograms are so legible to humans and so learnable for models: the structure of sound that we perceive becomes spatial structure a 2D network can exploit, almost exactly the way an image model exploits spatial structure in pixels.

## 3. The time-frequency uncertainty trade-off

Here is the deepest idea in the whole post, and it is not optional hand-waving — it is a theorem. **You cannot have arbitrarily good time resolution and arbitrarily good frequency resolution at the same time.** The STFT forces a choice through the window length, and the choice is governed by an uncertainty principle directly analogous to the one in quantum mechanics.

The intuition first. To know precisely *when* something happened, you want a very short window — a 1-millisecond window can pin a click to within a millisecond. But a 1-millisecond window at 16 kHz contains only 16 samples, and you cannot tell apart two frequencies that differ by less than roughly one cycle over that window; a window that short can only distinguish frequencies coarser than about 1 kHz apart. Conversely, to resolve two close pitches — say a 440 Hz note and a 444 Hz note, 16 cents apart — you need a window long enough to see several beats between them, perhaps 250 ms, but a 250 ms window blurs *when* each note started across a quarter of a second. Short window: sharp in time, blurry in frequency. Long window: sharp in frequency, blurry in time. There is no free lunch.

![A before and after comparison contrasting a short five millisecond STFT window with sharp timing and coarse frequency against a long fifty millisecond window with coarse timing and sharp frequency.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-3.png)

Now the math. Define the time spread $\Delta t$ and the frequency spread $\Delta f$ of the analysis window as their respective standard deviations. The Gabor uncertainty principle states

$$\Delta t \cdot \Delta f \ \ge\ \frac{1}{4\pi}.$$

The product is bounded below by a constant; you can shrink one factor only by growing the other. The Gaussian window achieves the equality (it is the minimum-uncertainty window), which is exactly why Dennis Gabor proposed it and why the Gabor transform is the theoretical ideal STFT. Practically, the frequency resolution of an STFT with window length $N$ samples at sample rate $f_s$ is on the order of

$$\Delta f \approx \frac{f_s}{N},$$

the width of one FFT bin, and the time resolution is on the order of the window duration $\Delta t \approx N / f_s$. Multiply them and you get $\Delta t \cdot \Delta f \approx 1$ — the same trade, up to the constant: **doubling the window halves $\Delta f$ but doubles $\Delta t$.** No setting of `n_fft` escapes it.

#### Worked example: choosing a window for speech versus music

Suppose you are building a TTS vocoder front end at 22.05 kHz and you pick `n_fft = 1024`. Frequency resolution is $\Delta f \approx 22050 / 1024 \approx 21.5$ Hz per bin, and the window spans $1024 / 22050 \approx 46$ ms. For speech that is a reasonable compromise: 21.5 Hz is fine enough to capture the harmonic spacing of a typical voice (a male voice around 120 Hz fundamental has harmonics 120 Hz apart, easily resolved), and 46 ms is short enough to track the syllable-rate changes in speech (syllables arrive every ~150-250 ms). Now suppose you are analyzing a fast drum solo. 46 ms is *too long* — two snare hits 30 ms apart blur into one — so you would drop to `n_fft = 256`, getting $\Delta t \approx 12$ ms (sharp enough to separate the hits) at the cost of $\Delta f \approx 86$ Hz (you lose the ability to resolve a bass line's exact pitch, but you do not need it for the drums). The two analyses see *different truths about the same signal*, and which truth you want depends on whether you care more about rhythm or harmony. This is not a bug; it is the fundamental limit, and good audio engineers choose their window per task. Some systems compute several STFTs at different window lengths and combine them — the **multi-resolution STFT loss** used to train modern vocoders is exactly this idea, comparing the generated and real waveforms across several `n_fft` settings so no single resolution dominates the gradient.

Here is a practical cheat-sheet for STFT settings by task and sample rate, the kind of table I keep pinned:

| Task | Sample rate | `n_fft` | `hop_length` | Freq res | Time res | Why |
| --- | --- | --- | --- | --- | --- | --- |
| Speech / TTS mel | 22.05 kHz | 1024 | 256 | ~21 Hz | ~46 ms | Resolves voice harmonics, tracks syllables |
| Music harmony | 44.1 kHz | 2048 | 512 | ~21 Hz | ~46 ms | Separates close pitches |
| Drums / transients | 44.1 kHz | 512 | 128 | ~86 Hz | ~12 ms | Pins onsets in time |
| Telephone speech | 16 kHz | 512 | 160 | ~31 Hz | ~32 ms | Compact, intelligibility-first |

Notice that the "right" settings differ by task precisely because of the uncertainty trade — there is no single STFT config that is best for everything, only the config that puts your resolution where your task needs it.

The uncertainty principle is also *why the mel-spectrogram is allowed to be coarse.* If perfect joint resolution were achievable, throwing away resolution would be pure loss. But since you must give up something anyway, you might as well give it up where your ears do not care — and that is the entire justification for the mel scale, which is where we go next.

## 4. The mel-spectrogram: warping frequency to perception

The linear spectrogram has two problems for generative modeling. First, it is large (513 bins) and most of those high-frequency bins carry little perceptually relevant information — your ear cannot tell 18,000 Hz from 18,100 Hz, but the spectrogram spends a bin on each. Second, it spaces frequency *linearly* in Hz, while your ear spaces frequency *logarithmically*. You hear the jump from 100 Hz to 200 Hz (one octave) as the same musical distance as 1000 Hz to 2000 Hz (also one octave), even though the second jump covers ten times more Hz. A representation aligned to perception should give equal resolution to equal *perceived* pitch distance, not equal Hz distance.

The **mel scale** does exactly that. It is a perceptual frequency scale, derived from experiments where listeners judged when pitches were equally spaced, calibrated so that 1000 mels equals 1000 Hz and equal mel steps sound equally far apart. The standard (HTK / Slaney) formula mapping frequency $f$ in Hz to mels is

$$m(f) = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right),$$

and its inverse, $f(m) = 700\,(10^{m/2595} - 1)$. It is nearly linear below 1 kHz and increasingly logarithmic above. To build a **mel-spectrogram** you lay down a bank of overlapping triangular filters spaced *evenly on the mel scale* (so densely packed at low frequencies, widely spaced at high frequencies), multiply each filter against the linear power spectrum, and sum — collapsing the 513 linear bins into typically 80 mel bins. Then you take the logarithm of the result, because loudness perception is also roughly logarithmic (the decibel exists for this reason), turning a multiplicative dynamic range into an additive one a network finds far easier to model.

![A stack showing the mel filterbank warping linear STFT bins through the mel formula into many narrow low band filters and few wide high band filters down to eighty mel bins.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-6.png)

The whole transform is a single matrix multiply applied to the magnitude spectrogram. If $S \in \mathbb{R}^{F \times T}$ is the linear magnitude (or power) spectrogram with $F = n\_fft/2 + 1$ bins, and $M \in \mathbb{R}^{n\_mels \times F}$ is the mel filterbank matrix, then the mel-spectrogram is $\log(M S + \epsilon)$, shape $n\_mels \times T$. For our running clip that is $80 \times 126 \approx 10{,}080$ numbers, versus $513 \times 126 \approx 64{,}600$ for the linear spectrogram and $32{,}000$ for the waveform. Roughly a 6x reduction from the linear spectrogram and a 3x reduction from the raw waveform — and a far larger reduction in *what the model has to predict*, because the 80 mel bands are smooth and slowly varying compared to the wild oscillation of raw samples.

Here is the full mel pipeline in `torchaudio`, with every parameter named.

```python
import torchaudio.transforms as T

mel_fn = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,            # number of perceptual bands; 80 is the TTS standard
    f_min=0.0,
    f_max=sample_rate / 2,
    power=2.0,            # power mel-spectrogram
    norm="slaney",        # filterbank normalization
    mel_scale="slaney",   # the mel-scale variant
)

mel = mel_fn(waveform)                    # (1, 80, num_frames)
log_mel = torch.log(torch.clamp(mel, min=1e-5))   # the log compresses dynamic range
print(mel.shape, log_mel.min().item(), log_mel.max().item())
# (1, 80, 126)  ...  a compact, perceptual, log-magnitude representation
```

It is worth opening up the filterbank itself, because once you have seen its shape the mel-spectrogram stops being a black box. The filterbank is built by choosing `n_mels + 2` points spaced evenly *on the mel axis* between `f_min` and `f_max`, mapping them back to Hz (where they come out densely packed at the low end and widely spread at the high end), and centering one triangular filter on each interior point. Filter $i$ rises linearly from zero at the previous point, peaks at its own center, and falls to zero at the next point, so adjacent filters overlap and every linear bin contributes to one or two mel bins. You can pull the matrix out and look at it directly.

```python
import torchaudio.functional as AF
import torch

# The mel filterbank matrix: (n_freqs, n_mels). Each column is one triangular filter.
fb = AF.melscale_fbanks(
    n_freqs=1024 // 2 + 1,    # 513 linear bins
    f_min=0.0, f_max=8000.0,
    n_mels=80, sample_rate=16000, norm="slaney",
)
print(fb.shape)                                   # (513, 80)
# Low mel filters are narrow (few linear bins each); high mel filters are wide.
widths = (fb > 0).sum(dim=0)                       # nonzero linear bins per filter
print("filter 0 width:", widths[0].item(),
      "filter 79 width:", widths[-1].item())       # e.g. 3 vs ~40: low is narrow, high is wide
mel_from_spec = fb.T @ power_spec.squeeze(0)       # (80, T) = filterbank applied
```

That width asymmetry — three linear bins for the lowest filter, dozens for the highest — *is* the perceptual warping made concrete: the representation gives fine resolution where the ear resolves finely (low frequencies) and coarse resolution where it does not (high frequencies). Nothing about the mel-spectrogram is magic; it is this one matrix multiply plus a log.

`librosa` gives the same thing with a NumPy-flavored API, and it is worth knowing both because half the codebases you will read use one and half use the other.

```python
import librosa
import numpy as np

y, sr = librosa.load("speech.wav", sr=16000, mono=True)   # librosa resamples for you
mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=2.0
)                                                          # (80, num_frames)
log_mel_db = librosa.power_to_db(mel, ref=np.max)          # dB-scaled log mel
print(mel.shape)
```

**Why is this the workhorse?** Because it is the sweet spot of the fidelity-versus-tractability trade for speech and most music modeling. It is compact (small enough to model autoregressively or with a diffusion U-Net at reasonable length), it is perceptually aligned (the model spends its capacity where the ear listens), the log makes its values well-conditioned, and crucially its smoothness means a model that gets a mel frame slightly wrong produces audio that sounds slightly wrong rather than catastrophically wrong — errors are perceptually graceful. Tacotron 2, FastSpeech, Glow-TTS, and most of the classical TTS pipeline predict mel-spectrograms and hand them to a vocoder; I will trace that whole lineage in [text-to-speech from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits). Even many music systems operate on a mel or mel-like latent. The price of admission is the one thing the mel quietly discarded — phase — and that bill comes due at synthesis time.

## 5. The phase problem and why you need a vocoder

Recall that the STFT produces *complex* numbers: each time-frequency bin has a magnitude and a phase. The mel-spectrogram is built from the *magnitude* only (we summed filterbank energies and took a log), and on top of that it folded 513 bins into 80, a many-to-one collapse. So a mel-spectrogram has thrown away two things: the phase entirely, and the fine frequency detail that the filterbank merged. The magnitude tells you *how much* of each frequency band there was; the phase tells you *how those frequencies are aligned in time* — and you cannot reconstruct a waveform without it, because the same magnitude spectrum with different phases produces audibly different (or even completely different) signals.

How much does phase matter? More than beginners expect. A classic demonstration: take the magnitude spectrum of one sound and the phase spectrum of another, recombine, and invert — the result usually sounds like the sound whose *phase* you kept, not whose magnitude you kept. Phase carries the transient and temporal structure: the sharp attack of a plucked string, the precise alignment that makes a click a click. So discarding phase is not a rounding error; it is discarding genuine information, and the only reason we get away with it is that the *missing phase can be re-estimated* well enough to fool the ear, given the magnitude. That re-estimation is the job of the **vocoder** (from "voice coder"), the module that turns a phase-free spectral representation back into a waveform.

![A graph showing two reconstruction routes from a phase free mel-spectrogram, an iterative Griffin-Lim path that sounds phasey and a neural vocoder path that produces clean natural audio.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-7.png)

### 5.1 Why a magnitude mel-spectrogram is non-invertible

Let me make the non-invertibility precise, because it is the technical heart of this section. Going from the complex STFT to a waveform is trivial and exact — the inverse STFT (overlap-add of the inverse FFT of each frame) reconstructs $x[n]$ perfectly, *because we kept both magnitude and phase*. But we did not keep them. We have only $|X[m,k]|$, and even that only after the lossy mel collapse. Two obstacles stand between us and the waveform:

1. **Phase is gone.** For a given magnitude $|X[m,k]|$, there is an infinite family of phase assignments $\angle X[m,k]$, and most of them do not even correspond to a *consistent* signal — because adjacent STFT frames overlap, a real signal's phase across frames must satisfy consistency constraints. Recovering a consistent phase from magnitude alone is the **phase retrieval problem**, which has no closed-form solution.
2. **The mel collapse is many-to-one.** Mapping 513 linear bins to 80 mel bins is a non-invertible linear map (you cannot uniquely recover 513 numbers from 80). So even the *magnitude* is only approximately recoverable, by multiplying through an approximate pseudo-inverse of the mel filterbank.

The standard classical answer to obstacle 1 is the **Griffin-Lim algorithm** (Griffin and Lim, 1984). It is an alternating-projection method that iteratively guesses a phase consistent with the target magnitude:

1. Start with the target magnitude $|X|$ and a random (or zero) phase, forming a complex spectrogram.
2. Inverse-STFT it to get a waveform — this waveform's *own* STFT will not exactly match the target magnitude.
3. Re-STFT that waveform, keep its *phase*, but replace its *magnitude* with the original target $|X|$.
4. Repeat steps 2-3 for 30-100 iterations.

Each iteration projects onto two constraint sets — "has the right magnitude" and "is the STFT of a real signal" — and the fixed point is a waveform whose magnitude matches the target with a self-consistent phase. It reduces a reconstruction-error objective monotonically. The reason "is the STFT of a real signal" is a genuine constraint and not a tautology is the overlap: because consecutive frames share samples, the set of complex spectrograms that are the STFT of *some* real waveform is a proper subspace of all complex spectrograms. An arbitrary magnitude with an arbitrary phase generally lands *outside* that subspace; the inverse-STFT-then-forward-STFT round trip is exactly the projection back onto it, and that round trip changes the magnitude, which is why you have to re-impose the target magnitude every iteration. Formally, Griffin-Lim minimizes the squared distance between the current complex spectrogram and the nearest one with the target magnitude, $\sum_{m,k} \big( |\hat{X}[m,k]| - |X[m,k]| \big)^2$, subject to consistency — a non-convex objective, so it finds a good local minimum, not the global optimum, which is part of why the recovered phase is plausible but not exact. Griffin-Lim is parameter-free, needs no training, and runs in milliseconds, which is why it is the universal *baseline* vocoder. Its weakness is audible: the phase it invents is "smooth" and globally consistent but not the *right* phase, so the output has a characteristic glassy, metallic, slightly underwater quality — listeners call it "phasey." For a quick check that your mel pipeline is wired correctly it is perfect; for a product, it is not good enough.

#### Worked example: Griffin-Lim iterations versus quality and latency

Griffin-Lim's `n_iter` is the dial between speed and quality, and the curve has sharply diminishing returns. With `n_iter=1` (essentially random phase, one projection) the output is badly phasey and barely usable. By `n_iter=32` the words are intelligible and the metallic ring is reduced; by `n_iter=60` you are near the algorithm's ceiling and further iterations buy almost nothing. On a CPU, 60 iterations of Griffin-Lim on a few seconds of 16 kHz audio runs in well under a second — fast, but each iteration is a full inverse-STFT and forward-STFT, so cost grows linearly with `n_iter` and with audio length. Contrast that with a HiFi-GAN neural vocoder, which is a *single* forward pass (no iteration) and produces strictly better audio: the neural vocoder is both faster per second of audio at quality *and* higher quality, which is the entire reason Griffin-Lim survives only as a baseline. The honest framing for a report: "Griffin-Lim at 60 iterations, mel inverted with the filterbank pseudo-inverse, no learned components" — state the iteration count, because a Griffin-Lim number with no `n_iter` is meaningless.

### 5.2 Inverting a mel back to audio, in code

Let me show the full lossy round-trip so you can hear exactly what phase loss costs. `torchaudio` gives us `InverseMelScale` (the approximate pseudo-inverse of obstacle 2) and `GriffinLim` (the phase retrieval of obstacle 1).

```python
import torchaudio.transforms as T

# Forward: waveform -> mel (as in section 4)
mel_fn = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024,
                          hop_length=256, n_mels=80, power=2.0)
mel = mel_fn(waveform)                                   # (1, 80, T)

# Inverse step 1: mel bins -> approximate linear-frequency magnitude (pseudo-inverse).
inv_mel = T.InverseMelScale(n_stft=1024 // 2 + 1, n_mels=80,
                            sample_rate=sample_rate)
linear_spec = inv_mel(mel)                               # (1, 513, T) approximate magnitude

# Inverse step 2: magnitude spectrogram -> waveform via Griffin-Lim phase retrieval.
griffin_lim = T.GriffinLim(n_fft=1024, hop_length=256, power=2.0, n_iter=60)
reconstructed = griffin_lim(linear_spec)                 # (1, ~num_samples)

torchaudio.save("reconstructed_griffinlim.wav", reconstructed, sample_rate)
print("original samples:", waveform.shape[1],
      "reconstructed samples:", reconstructed.shape[1])
```

Play `reconstructed_griffinlim.wav` next to the original and you hear the artifact directly: intelligible, recognizably the same words or melody, but with a metallic ring and softened transients. That ring is the sound of invented phase. The fix is to *learn* the phase.

### 5.3 Neural vocoders: learning the phase

A **neural vocoder** is a network trained to map a mel-spectrogram directly to a waveform, learning the right phase implicitly from data instead of inventing a smooth one. The dominant design is **HiFi-GAN** (Kong et al., 2020): a generator that upsamples the mel through transposed convolutions, trained adversarially against multi-period and multi-scale discriminators that catch the buzzy, phasey artifacts a non-adversarial loss would miss. Because the discriminators specifically attack periodicity errors, the generator learns to produce clean periodic structure — which is to say, the right phase. HiFi-GAN and its descendants (BigVGAN, Vocos, and others) reconstruct studio-quality audio and run *faster than real time* on a GPU and often on CPU, which is why essentially every shipped mel-based TTS system uses a neural vocoder, not Griffin-Lim. I dig into vocoder architecture, the discriminators, and the speed numbers in [GAN vocoders, HiFi-GAN and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis); here the point is only *why* the vocoder exists: the mel threw away phase, and someone has to put it back. Here is what calling a pretrained vocoder looks like in practice.

```python
# Sketch of mel -> waveform with a pretrained neural vocoder (HiFi-GAN family).
# In practice you load a checkpoint matched to your mel config (n_fft, hop, n_mels, sr).
import torch

vocoder = torch.hub.load("descriptinc/vocos", "vocos_mel")  # illustrative; APIs vary
vocoder.eval()

with torch.no_grad():
    # mel must match the vocoder's expected mel spec (sample rate, n_mels, hop, log-scaling)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    waveform_hat = vocoder(log_mel)        # (1, num_samples), studio-quality phase
torchaudio.save("reconstructed_neural.wav", waveform_hat, sample_rate)
```

The trade between the two reconstruction routes is worth tabulating, because it is the decision every mel-based system makes:

| Vocoder | Trained? | Quality | Speed | When to use |
| --- | --- | --- | --- | --- |
| Griffin-Lim (60 iter) | No | Phasey, metallic | Iterative, sub-real-time on CPU | Debugging, a quick sanity check |
| HiFi-GAN | Yes (adversarial) | Near-human MOS | Single pass, far faster than real time | Production TTS, streaming |
| BigVGAN / Vocos | Yes (adversarial) | Near-human, robust to out-of-domain | Single pass, real-time class | High-fidelity, diverse audio |
| Diffusion vocoder | Yes (diffusion) | Very high | Many steps, slow | Only when GAN vocoders miss the bar |

The pattern is clear: Griffin-Lim is the free baseline, GAN vocoders are the production default, and a diffusion vocoder is a heavyweight you reach for only when the cheaper options genuinely fall short — rarely.

The same mel, two vocoders, two very different outputs: that contrast is the whole reason the vocoder is its own box in the audio stack. And it explains a frequent production surprise: sometimes the *vocoder*, not the acoustic model, is your quality bottleneck or your latency bottleneck — a perfect mel through a weak vocoder sounds worse than a mediocre mel through a great one.

I have lived this bottleneck. On one TTS system the acoustic model produced clean mel-spectrograms — I could verify them by eye, the formants were crisp and the harmonics were sharp — yet the output buzzed. The instinct is to blame the model and start retraining, but the right move is to *bisect the pipeline*: invert the model's mel with Griffin-Lim and, separately, invert a mel computed directly from real audio with the *same* vocoder. If real-audio mel through the vocoder also buzzes, the vocoder (or a mel-config mismatch between the vocoder's training and your STFT settings) is the culprit, not the acoustic model. In that case it was the second thing — the vocoder had been trained with `hop_length=256` but I was feeding it mels computed at `hop_length=300`, a silent mismatch that smears every frame. The fix was a one-line config change, not a week of retraining. The general lesson: because the mel-to-waveform step is a separate learned box with its own assumptions about `n_fft`, `hop_length`, `n_mels`, and sample rate, the vocoder is both a common failure point and a common *false* suspect, and the only way to know which is to test each box with a known-good input. Treat the representation boundaries as the seams where you bisect.

## 6. The codec alternative: tokens instead of spectrograms

There is a second answer to "how do I represent sound for a model," and it is the one most of the 2024-2026 frontier uses: **neural audio codec tokens.** Instead of a hand-designed mel transform, you *learn* an encoder that maps a waveform to a short sequence of discrete tokens, plus a decoder that maps those tokens back to a waveform — both trained end-to-end. This is the direct audio analogue of the latent VAE that powers latent diffusion image models; if you have read [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch), a neural codec is the same idea (a learned encoder-decoder bottleneck) specialized to 1D audio and with a *discrete* (vector-quantized) bottleneck so the result is a token sequence a language model can predict.

The headline reason codecs took over: they are even more compact than mel, *and* their decoder is a trained vocoder so phase is handled inside the box. EnCodec (Défossez et al., 2022) and the Descript Audio Codec (DAC) can represent 24 kHz audio at a few kbps with high fidelity, turning a second of audio into roughly 50-150 discrete tokens depending on the bitrate. That makes audio look like text: a sequence of tokens you can model with a transformer, which is exactly the recipe behind VALL-E (TTS as codec-token language modeling) and MusicGen (music as codec-token language modeling). The machinery that makes this work — vector quantization, **residual** VQ stacking multiple codebooks to climb the rate-distortion curve, the straight-through gradient, codebook collapse — is the subject of [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) and the RVQ post that follows it. For this representation post, the thing to hold onto is where codec tokens sit on the size-versus-fidelity map relative to the mel.

![A four by four matrix comparing waveform, linear spectrogram, mel-spectrogram, and codec tokens across size, invertibility, perceptual alignment, and which systems use each.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-5.png)

The matrix above is the one-screen summary of this entire post. Read it as a ladder of trades: as you move from waveform down to codec tokens, you give up exactness and gain compactness and perceptual alignment. The waveform is exact and enormous and not perceptual. The linear spectrogram is invertible (it kept phase) and still large and linear-in-Hz. The mel-spectrogram is small and perceptual and *not* invertible (phase gone) — hence the vocoder. Codec tokens are the smallest and discrete and perceptual, invertible only through their learned decoder. There is no universally best row; there is only the best row *for a given model and goal*, which is the decision the final section makes explicit.

#### Worked example: the data-rate reduction from waveform to mel to codec

Let me put real numbers on "compact." One second of 24 kHz mono audio:

- **Waveform:** 24,000 samples. At 16-bit PCM that is $24{,}000 \times 16 = 384{,}000$ bits per second = **384 kbps**. As 32-bit floats for modeling, 768 kbps. The model must predict 24,000 values per second.
- **Mel-spectrogram** (80 mels, `hop_length=256`): about $24{,}000 / 256 \approx 94$ frames per second, each with 80 floats, so $94 \times 80 \approx 7{,}520$ values per second — roughly a **3x reduction in value count** versus the waveform, but more importantly each value is smooth and predictable. (Stored as 32-bit floats this is ~240 kbps, but storage is not the point; *modeling tractability* is.)
- **EnCodec tokens** at ~6 kbps, 24 kHz: roughly 75 frames per second with 8 codebook tokens each (it varies by bitrate setting) ≈ 600 tokens per second, but the *information rate* is the headline **6 kbps** — a **64x reduction** from the 384 kbps PCM, with audio that still sounds good. The model predicts ~600 discrete tokens per second instead of 24,000 continuous samples.

It is worth stress-testing the codec representation, because its compactness is not free and the failure modes are instructive. **What happens at a lower bitrate?** As you drop from 6 kbps toward 1.5 kbps — by using fewer residual codebooks — the codec sheds detail in a perceptually ordered way: first the air and high-frequency sparkle go, then fine texture, then, at the extreme, you start to hear quantization "warble" and a smeared, lossy-MP3-at-64-kbps character. The audio stays intelligible far longer than you would expect (speech survives surprisingly low bitrates because so much of it is redundant), but music degrades faster because its high-frequency content carries real information. **What happens when the codec drops the high frequencies?** If your codec was trained mostly on 24 kHz audio and you feed it a 44.1 kHz signal naively, or if a low-bitrate setting starves the high bands, cymbals and sibilance ("s" and "sh" sounds) go dull — the same muffled, underwater quality as undersampling, for the same reason: the information is not being represented. The defense is to match the codec's training sample rate and to budget enough bitrate for the high bands your content needs. The deeper point for this post: a codec, like the mel, is a *perceptually ordered* lossy representation — it gives up the least audible information first — which is exactly what lets it be so compact, and exactly why "how many kbps?" is the question that determines whether it sounds transparent or obviously compressed. The residual-VQ mechanism that produces this graceful bitrate ladder is the subject of [residual vector quantization](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound).

That is the quantitative story of why nobody models raw 24 kHz waveforms autoregressively anymore for long content: the codec compresses the sequence length by an order of magnitude *and* hands you discrete tokens that a transformer eats happily. The mel sits in between — less compression than a codec but no codebook to train, no quantization to tune, and a mature vocoder ecosystem — which is why it remains the pragmatic default for TTS and a lot of music. The choice between mel and codec is one of the recurring tensions of the whole [audio stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).

## 7. Psychoacoustics: why these representations work

Everything above leans on one assumption: that we can throw away information and still fool the ear. That assumption is licensed by **psychoacoustics**, the science of how humans actually perceive sound, and it is worth being precise about because it is *why* lossy audio representations and codecs succeed where naive intuition says they should fail.

**The mel scale and critical bands.** We already used the mel scale's logarithmic frequency warping. Its physiological basis is the cochlea: the basilar membrane in your inner ear is a frequency analyzer where position maps to frequency *logarithmically*, and it resolves frequency in roughly constant-width chunks on this warped axis called **critical bands**. There are about 24 critical bands (the Bark scale is a closely related psychoacoustic scale built directly on them). Two tones within the same critical band interact; two tones in different bands are heard independently. The mel filterbank is a coarse engineering approximation of this critical-band analysis — which is exactly why it works: it groups frequencies the way your cochlea groups them.

**Equal-loudness.** Your hearing is not equally sensitive at all frequencies. The equal-loudness contours (the Fletcher-Munson curves, standardized as ISO 226) show that you are most sensitive around 2-5 kHz (where speech consonants and a baby's cry live) and much less sensitive at very low and very high frequencies. A 50 Hz tone and a 3 kHz tone at the same physical energy sound wildly different in loudness; the 3 kHz one is far louder to you. This is why A-weighting exists in sound-level meters, and why a perceptually honest model should not weight a 50 Hz error the same as a 3 kHz error. The log-magnitude in the mel-spectrogram partially accounts for the logarithmic loudness response, and well-designed perceptual losses weight frequency bands by sensitivity.

**Frequency masking.** This is the big one for compression. A loud tone *masks* nearby quieter tones and noise — you literally cannot hear a quiet 1.1 kHz tone played alongside a loud 1 kHz tone, because the loud tone saturates that region of your cochlea. Masking happens in frequency (a loud tone hides quieter neighbors) and in time (a loud sound hides quiet sounds just before and after it, pre- and post-masking). Lossy audio codecs like MP3 and AAC are built entirely on this: they compute a *masking threshold* per critical band and spend bits only on what rises above it, discarding everything masked because *you cannot hear it anyway.* MP3 at 128 kbps throws away the vast majority of the raw PCM bits and most listeners cannot tell, purely because of masking. Neural audio codecs do not compute an explicit masking model, but they learn an analogous behavior implicitly: trained to minimize a *perceptual* reconstruction loss (often a multi-resolution mel loss plus adversarial loss), they learn to preserve the audible structure and spend their limited token budget where the ear listens, allocating fidelity exactly where masking says it matters.

There is also a *temporal* dimension to masking that matters for transients. A loud sound masks quieter sounds not only at the same instant but for a short window before and after it — **pre-masking** (a few milliseconds) and **post-masking** (tens of milliseconds). Post-masking is why a small amount of distortion right after a drum hit is inaudible: the hit's energy is still "ringing" in your auditory system and hides the error. Codecs exploit this by allowing larger quantization error immediately after a transient, and it is one reason that a representation can be slightly wrong in time near loud events without anyone noticing. It also explains why the *attack* of a sound (the onset) is perceptually precious while the *decay* is forgiving — get the attack of a snare right and listeners forgive a lot in the tail.

This perceptual logic is not just analysis; it goes directly into how modern audio models are *trained*. Rather than comparing waveforms sample-by-sample (an $L_1$ or $L_2$ loss on raw samples is a poor proxy for perceived quality, because it weights inaudible errors as heavily as audible ones), vocoders and codecs are trained against a **multi-resolution mel loss**: compare the generated and target audio in the mel domain at several STFT resolutions, so the loss measures perceptual spectral error across both fine-time and fine-frequency views, sidestepping the uncertainty trade by using several windows at once. A minimal version looks like this.

```python
import torch
import torchaudio.transforms as T

def multi_res_mel_loss(y_hat, y, sr=22050,
                       n_ffts=(512, 1024, 2048), n_mels=80):
    """Perceptual reconstruction loss: L1 in log-mel across STFT resolutions."""
    loss = 0.0
    for n_fft in n_ffts:
        mel = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft,
                               hop_length=n_fft // 4, n_mels=n_mels,
                               power=1.0).to(y.device)
        m_hat = torch.log(torch.clamp(mel(y_hat), min=1e-5))
        m_ref = torch.log(torch.clamp(mel(y),     min=1e-5))
        loss = loss + torch.nn.functional.l1_loss(m_hat, m_ref)
    return loss / len(n_ffts)
```

Because the loss lives in the log-mel domain, it inherits the perceptual alignment we built in section 4 for free: errors are weighted by the mel scale (frequency) and the log (loudness), so the gradient pushes the model to fix audible mistakes before inaudible ones. This is the same insight, used twice — first to *choose* a representation, now to *train against* one.

The unifying point: **a representation can be aggressively lossy and still sound perfect if it loses only what the ear cannot perceive.** The mel-spectrogram loses high-frequency detail and phase; codecs lose masked content and quantize the rest; both survive because human hearing is itself lossy and band-limited. This is not a hack — it is the principled exploitation of the gap between the physical signal and the perceived signal, and it is the reason audio generation is tractable at all. Get the perception wrong (model a tiny inaudible high-frequency error, ignore an audible mid-band one) and your metrics will look fine while your audio sounds bad, which is the central theme of the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post.

## 8. Sample rate and resampling in practice

One last practical axis decides everything downstream: the sample rate you commit to. It is not a free parameter you can change later; it is baked into your data, your model's output dimension, and your vocoder.

The two regimes you will meet constantly:

- **16 kHz** (Nyquist 8 kHz) — the speech standard. Speech intelligibility lives almost entirely below 8 kHz, so 16 kHz captures what matters for ASR, telephony, and a lot of TTS, while keeping sequence lengths and compute small. Whisper, many TTS training sets, and most speech codecs default here. 22.05 kHz and 24 kHz are common "nicer speech" upgrades that recover some sibilance and air.
- **44.1 kHz / 48 kHz** (Nyquist 22.05 / 24 kHz) — the music and full-band standard. Music has cymbals, harmonics, and air well above 8 kHz, and listeners notice when they are gone, so music models (Stable Audio, MusicGen at its higher-fidelity settings, Suno) work at 44.1 or 48 kHz. The price is 2.75x more samples per second than 16 kHz and proportionally more compute.

The single most common bug in an audio pipeline is a **sample-rate mismatch**: you train a vocoder at 22.05 kHz, feed it a mel computed from 16 kHz audio, and get garbage, because the same `n_fft` and `hop_length` mean different *frequencies* and different *time spans* at different sample rates. Every component — the STFT config, the mel filterbank's `f_max`, the vocoder, the codec — is tied to one sample rate. So you **resample** to a common rate before anything else. Resampling is not just dropping or duplicating samples; downsampling requires an anti-aliasing low-pass filter (or the high frequencies above the new Nyquist alias down and corrupt the signal, as we saw in section 1), and good resamplers use windowed-sinc interpolation. `torchaudio` and `librosa` both do this correctly.

```python
import torchaudio.transforms as T

# Resample 44.1 kHz music down to 24 kHz for a 24 kHz codec/vocoder.
resampler = T.Resample(orig_freq=44100, new_freq=24000,
                       lowpass_filter_width=64,        # sinc kernel width: quality vs speed
                       rolloff=0.95)                    # anti-alias cutoff
waveform_24k = resampler(waveform_44k)

# librosa equivalent (note: it resamples on load if you pass sr=...)
import librosa
y_16k = librosa.resample(y_44k, orig_sr=44100, target_sr=16000, res_type="kaiser_best")
```

#### Worked example: what 16 kHz costs a music model

Suppose you have a music generator and, to save compute, you train and generate at 16 kHz instead of 44.1 kHz. You have just capped every output at an 8 kHz Nyquist ceiling. A hi-hat's energy lives largely between 8 and 16 kHz; a cymbal's shimmer extends past 16 kHz. At 16 kHz output, all of that is *gone* — not quiet, gone, because those frequencies cannot exist below the Nyquist line. The result sounds muffled and "underwater," like music played through a wall, and no amount of model quality fixes it because the representation itself cannot encode those frequencies. This is why you never see a serious music model at 16 kHz, and why "what sample rate?" is one of the first questions to ask of any audio model — it is a hard upper bound on fidelity that no clever architecture can recover. For speech the same 16 kHz is perfectly fine, which is the whole reason the regimes split the way they do.

## 9. Putting the representations side by side: a decision

We have four representations on the table — waveform, linear spectrogram, mel-spectrogram, codec tokens — and the engineering question is always "which one does my model predict?" The answer follows from the job, and it is worth laying out as a decision rather than a list.

![A decision tree choosing a target representation, branching into a spectrogram path with mel and a vocoder for TTS and music and a token path with codec language models for voice cloning and MusicGen.](/imgs/blogs/representing-sound-waveforms-spectrograms-and-perception-8.png)

The two main branches are the **spectrogram path** and the **token path**:

- **Spectrogram path (mel + vocoder).** Your model predicts a mel-spectrogram; a separate vocoder turns it into audio. This is the classical, battle-tested route for TTS (Tacotron 2, FastSpeech 2, Glow-TTS, VITS's internal representation) and for a great deal of music and audio modeling that operates on a mel or mel-like latent (AudioLDM works in a mel-derived latent). Reach for it when you want a mature, debuggable pipeline, when you have a good pretrained vocoder for your sample rate, and when continuous-valued modeling (diffusion, flow, or regression) suits you. Its cost is the extra vocoder stage and the phase reconstruction it implies.
- **Token path (codec language model).** Your model predicts discrete codec tokens; the codec's decoder turns them into audio. This is the modern route for codec-token language models — VALL-E and its descendants for zero-shot voice cloning, MusicGen for music — and it shines when you want to *reuse the transformer / language-model toolbox* (sampling, classifier-free guidance, in-context learning from an audio prompt), when you want maximum compression, and when discrete tokens fit your architecture. Its cost is training (or trusting) a good codec and handling the multi-codebook structure of residual VQ.

And the two niche choices:

- **Raw waveform** is reserved for cases where every sample of fidelity counts and the model can afford the length — the original WaveNet, some high-fidelity vocoders that *are* waveform models internally, and research where you specifically want to avoid any spectral bottleneck. For general long-form generation it is almost always the wrong target; the length problem from section 1 dominates.
- **Linear spectrogram** is rare as a *modeling* target (it is large and not perceptual) but central as an *intermediate* — it is what the vocoder and Griffin-Lim operate on, and what you compute the STFT loss against.

One nuance keeps these four boxes from being as separate as they look: the modern frontier increasingly *blurs* the spectrogram and token paths. Latent diffusion music models (Stable Audio) run diffusion on a *continuous* learned latent that is roughly a mel-like compression but trained end-to-end like a codec without the discrete bottleneck — a representation that sits between "mel" and "codec tokens." Some TTS systems predict continuous codec *embeddings* rather than discrete tokens. And a codec's encoder produces a continuous representation that is only discretized at the final quantizer, so "mel vs codec" is really a spectrum of *learned-ness* (hand-designed mel at one end, fully learned discrete codec at the other) and *discreteness* (continuous at one end, quantized tokens at the other). Knowing where a system sits on those two axes tells you most of what you need to know about its training and its failure modes — and recognizing the spectrum, rather than memorizing four boxes, is what lets you reason about a new architecture you have never seen.

The throughline of the series is the **audio stack** — waveform → mel latent or codec tokens → generative model → vocoder/decoder → waveform — under the four-way tension of **fidelity × controllability × speed × length**. Your representation choice is the first lever on all four: a codec maximizes compression (helps length and speed) at the cost of needing a trained tokenizer; a mel maximizes pipeline maturity and continuous-control flexibility at the cost of an extra vocoder; a waveform maximizes fidelity at the cost of length. There is no globally right answer, only the right answer for your fidelity, latency, and length budget — which is precisely the decision the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) walks through end to end.

## 10. Case studies and real numbers

Abstract trades become convincing when you attach them to shipped systems. A few concrete data points from the literature and from systems I have run, with sources, and with the honest caveat that exact figures depend on sample rate, hardware, and config — where I am giving an order of magnitude I say so.

**Tacotron 2 + WaveNet vs Griffin-Lim (Shen et al., 2018).** The original Tacotron 2 predicted an 80-band mel-spectrogram and used a WaveNet vocoder, reaching a mean opinion score (MOS — a 1-to-5 human naturalness rating) of about 4.53, statistically close to the 4.58 MOS of professionally recorded human speech. The same mel-spectrograms inverted with Griffin-Lim instead of WaveNet scored markedly lower and sounded phasey. The lesson is exactly section 5: the *representation* (mel) was the same; the *vocoder* (learned vs Griffin-Lim) made the difference between near-human and obviously-synthetic. This is the single clearest demonstration that phase reconstruction quality, not just the spectrogram, governs final audio quality.

**HiFi-GAN vocoder speed (Kong et al., 2020).** HiFi-GAN reported MOS competitive with autoregressive WaveNet/WaveGlow while running far faster: roughly real-time factor (RTF — generation time divided by audio duration) well below 1 even on CPU, and on the order of hundreds of times faster than real time on a V100-class GPU for its larger variant, and over a thousand times faster than real time for its smallest. (RTF below 1 means faster than real time.) This is *why* neural vocoders replaced both Griffin-Lim and autoregressive waveform vocoders: they hit near-human quality at synthesis speeds that make streaming TTS feasible. The exact multipliers depend on the variant (V1/V2/V3) and GPU, so treat the "hundreds to thousands x" as the right order of magnitude rather than a precise constant.

**EnCodec bitrate-vs-quality (Défossez et al., 2022).** EnCodec compresses 24 kHz audio across a range of bitrates (roughly 1.5 to 24 kbps) by using more or fewer residual codebooks, trading fidelity for rate along the rate-distortion curve. At 6 kbps it reconstructs 24 kHz audio at quality that beats MP3 at the same or higher bitrate in listening tests — a roughly 64x compression from 384 kbps PCM with audio that still sounds good. That single number, 6 kbps, is the headline that made codec-token language models practical: it is what shrinks a second of audio to a few hundred tokens. The detailed rate-distortion behavior and the residual VQ that produces it are covered in the codec posts; here it anchors the "codec tokens are the most compact row" claim in the matrix.

**How to measure these claims honestly.** Every number above is only meaningful with its measurement conditions attached, and representation comparisons are easy to get wrong. For a *vocoder* comparison (Griffin-Lim vs HiFi-GAN vs Vocos), fix everything upstream — the same mel-spectrograms computed with identical `n_fft`, `hop_length`, `n_mels`, sample rate, and log-scaling — so the only variable is the reconstruction method; then report MOS with the number of raters and clips (a MOS from 5 raters on 10 clips is noise; aim for dozens of raters and dozens of clips), and report RTF after a warm-up pass on a *named* device with the batch size stated. For a *codec* comparison, report the exact bitrate (it is a setting, not a property) and the sample rate, and for objective fidelity use SI-SDR or a mel-distance, but lean on listening tests because objective metrics miss the artifacts the ear catches. And for *FAD* (Fréchet Audio Distance, the audio analogue of FID) state the embedding model and the sample size — FAD with a small sample or a mismatched embedding is unreliable, a pitfall the dedicated [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post unpacks. The discipline is simple: a number without its conditions is not a result, it is a rumor.

**The mel as the cross-system default.** Across an enormous range of TTS systems — Tacotron 2, FastSpeech 2, Glow-TTS, VITS, and most production pipelines — the intermediate representation is an 80-band mel-spectrogram at a hop length around 256 and `n_fft` around 1024, at 22.05 or 24 kHz. The remarkable thing is the *convergence*: independent teams keep landing on roughly the same mel configuration, because it is the empirical sweet spot of the perceptual-alignment, compactness, and vocoder-compatibility trades this post derived. When a representation choice is that stable across a field, it usually means the underlying constraints (the uncertainty principle, the mel scale, phase reconstruction) are real and binding — which is the whole thesis here.

## 11. When to reach for each representation (and when not to)

A decisive recommendation section, because the matrix and the tree are only useful if they end in a choice.

- **Default to the mel-spectrogram for TTS** and for music systems where you want a mature, debuggable pipeline. It is the workhorse for good reasons: compact, perceptual, continuous, with a strong pretrained-vocoder ecosystem. Do *not* reach past it to raw-waveform modeling for TTS unless you have a very specific reason — a mel pipeline beats raw-waveform modeling for the overwhelming majority of speech systems on quality-per-FLOP.
- **Reach for codec tokens when you want a language-model approach** — zero-shot voice cloning from a short prompt (VALL-E style), music generation with the transformer toolbox (MusicGen), or any case where maximum compression and discrete tokens help. Do *not* adopt a codec just because it is fashionable: it adds a trained tokenizer you must trust, multi-codebook bookkeeping, and a failure mode (codebook collapse) the mel pipeline does not have. If a mel pipeline already hits your quality and latency bar, the codec is not worth the complexity.
- **Use raw-waveform modeling only when fidelity is paramount and length is short** — a high-fidelity vocoder's internals, a research probe with no spectral bottleneck. For general long-form generation, the length arithmetic from section 1 makes it the wrong default.
- **Never skip the vocoder question.** If your model predicts mel, your audio quality is *capped by your vocoder*. Budget for a good neural vocoder (HiFi-GAN, BigVGAN, Vocos) and never ship Griffin-Lim to users — it is a debugging tool, not a product vocoder. A frequent, expensive mistake is to spend months on the acoustic model and bolt on Griffin-Lim at the end, then wonder why the audio sounds metallic.
- **Pick your sample rate first and never change it casually.** 16 kHz for speech-only, 24 kHz for nicer speech, 44.1/48 kHz for music. It is a hard ceiling on fidelity and it is wired into every downstream component, so committing late means re-training the vocoder and recomputing every spectrogram.
- **Don't use Griffin-Lim when quality matters**, and equally, don't use a heavyweight diffusion vocoder when HiFi-GAN already clears your quality bar at a fraction of the latency. Match the reconstruction method to the requirement, not to the hype.

## 12. Key takeaways

- A digital sound is a **waveform**: amplitude over time, defined by **sample rate** (Nyquist ceiling = $f_s/2$) and bit depth. Frequency is *implicit* in the waveform, which is why modeling it directly is hard.
- The **STFT** makes frequency explicit by transforming short overlapping windows, producing a **spectrogram** of complex (magnitude + phase) values; `n_fft`, `hop_length`, and the window control it.
- The **time-frequency uncertainty principle** ($\Delta t \cdot \Delta f \ge 1/4\pi$) forces a real choice: short window = sharp time, blurry frequency; long window = the reverse. No setting wins both.
- The **mel-spectrogram** warps frequency to the perceptual mel scale ($m = 2595\log_{10}(1+f/700)$), takes a log-magnitude, and drops phase — yielding a compact (~80 bins), perceptually aligned, and well-conditioned representation that is the **workhorse for TTS and much of music**.
- Dropping phase makes the mel **non-invertible**; a **vocoder** reconstructs the waveform. **Griffin-Lim** is the parameter-free phase-retrieval baseline (fast but phasey); **neural vocoders** (HiFi-GAN family) learn the phase and reach near-human quality faster than real time.
- **Neural codec tokens** are the learned, discrete, even-more-compact alternative (EnCodec/DAC at a few kbps ≈ 64x smaller than PCM), with the decoder handling phase inside the box — the foundation of codec-token language models like VALL-E and MusicGen.
- **Psychoacoustics** (critical bands, equal-loudness, frequency and temporal masking) licenses the lossiness: a representation can discard a lot and still sound perfect if it discards only what the ear cannot perceive.
- **Sample rate is a hard fidelity ceiling**: 16 kHz for speech, 44.1/48 kHz for music; mismatches are the most common audio-pipeline bug, and resampling needs anti-aliasing.
- The four representations are really points on a spectrum of *how learned* and *how discrete* they are; think in terms of that spectrum rather than four fixed boxes, and a new architecture becomes far easier to place.

## Further reading

- **Griffin, D. and Lim, J. (1984).** *Signal Estimation from Modified Short-Time Fourier Transform.* IEEE TASSP. The original phase-retrieval algorithm behind every Griffin-Lim implementation.
- **Shen, J. et al. (2018).** *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* (Tacotron 2). The paper that cemented the mel + vocoder pipeline and showed the vocoder, not the mel, governs final quality.
- **Kong, J., Kim, J., and Bae, J. (2020).** *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* The neural vocoder that made fast, near-human mel-to-waveform synthesis standard.
- **Défossez, A. et al. (2022).** *High Fidelity Neural Audio Compression* (EnCodec). The neural codec that turned audio into a few hundred discrete tokens per second.
- **van den Oord, A. et al. (2016).** *WaveNet: A Generative Model for Raw Audio.* The cautionary tale of modeling raw waveforms directly — brilliant quality, brutal length.
- **torchaudio documentation** — `Spectrogram`, `MelSpectrogram`, `InverseMelScale`, `GriffinLim`, `Resample`: the exact transforms used throughout this post.
- Within this series: the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the DSP sibling [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals), the tokenizer [neural audio codecs, the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound), and the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). For the learned-bottleneck analogy, the image series' [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).
