---
title: "The Mathematics of Audio Signals: Fourier, Sampling, and Convolution"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn the signal-processing math the whole audio-generation stack stands on — the Fourier transform, the sampling theorem, the STFT uncertainty trade, and convolution — with every law derived and every number worked, in code you can run."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "digital-signal-processing",
    "fourier-transform",
    "sampling-theorem",
    "convolution",
    "spectrogram",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-mathematics-of-audio-signals-1.png"
---

The first time I tried to debug a vocoder that buzzed, I stared at the waveform for an hour and learned nothing. A waveform is a million numbers a second going up and down; the buzz was in there somewhere, but the time domain hides everything that matters about audio. The moment I ran a Fourier transform and looked at the spectrum, the bug was obvious: a parasitic spike at 8 kHz, exactly half the model's sample rate, where the upsampling layer was folding energy back into the audible band. The waveform had been screaming this at me the whole time. I just had been reading it in the wrong language.

This post is the language. It is the rigorous companion to [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) — where that post builds intuition for *what* the representations are, this one derives *why* they work and proves the laws that govern them. Every neural codec, every vocoder, every diffusion-on-mel pipeline, every 1D convolution in an audio network rests on four pieces of digital signal processing: the Fourier transform (any signal is a sum of sinusoids), the sampling theorem (a continuous wave becomes a finite list of numbers without losing anything below a known limit), the short-time Fourier transform and its hard uncertainty trade (you cannot have sharp time and sharp frequency at once), and convolution with its theorem (filtering in time is multiplication in frequency). Master these and the rest of the [audio-generation series](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) stops being a pile of magic incantations and becomes engineering you can reason about.

![A before-and-after figure showing a tangled three-tone time-domain signal on the left resolving into three sharp frequency peaks at 220, 440, and 880 hertz on the right.](/imgs/blogs/the-mathematics-of-audio-signals-1.png)

I am writing this for the ML engineer who knows backprop cold but has never taken a DSP course. You do not need any signal-processing background. By the end you will be able to: read an FFT and find the frequencies in any sound; choose a sample rate and bit depth from first principles and predict the resulting noise floor in decibels; pick an FFT size and hop length knowing exactly what you trade; explain why 44.1 kHz captures up to ~22 kHz and what aliasing does when you get it wrong; and understand why dilated causal convolutions (WaveNet) and 1D convolutions are everywhere in audio networks. This is the math the rest of the series — codecs, vocoders, TTS, music models, and the [capstone stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) — quietly assumes you already know. It pairs with the image series' [mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions): that post is the math of the *data* in two dimensions; this is the math of the *signal* in one dimension at very high rate.

Throughout I will keep returning to one running example: a synthesized A note, 440 Hz, plus a couple of its harmonics — the simplest sound that is interesting enough to teach every concept. We will sample it, transform it, window it, filter it, and undersample it on purpose to break it. Let us start with what a signal even is.

## 1. A signal is a function, then a sequence

To a physicist, sound is a pressure wave: at every instant $t$ the air pressure at your eardrum takes some value $x(t)$. That is a **continuous-time signal** — a function from the real line (time, in seconds) to the reals (amplitude, in pascals or, after a microphone, volts). Our 440 Hz tone is the cleanest possible example:

$$
x(t) = A \sin(2\pi f t + \phi)
$$

where $A$ is amplitude, $f = 440$ Hz is frequency (cycles per second), and $\phi$ is the phase (where in its cycle the wave starts). The argument $2\pi f t$ sweeps through $2\pi$ radians — one full cycle — exactly $f$ times per second. Frequency is the single most important property of an audio signal because the human ear is, physically, a frequency analyzer: the cochlea is a coiled membrane whose position along its length resonates at different frequencies, so your ear delivers a frequency decomposition to your brain before any conscious processing happens. This is the deep reason the frequency domain is the *right* lens for audio and not merely a convenient one — it is the representation your own hearing uses.

A computer cannot store a function of a continuous variable. It stores a **discrete-time signal**: a sequence of numbers $x[n]$ for integer indices $n = 0, 1, 2, \dots$, obtained by reading the continuous signal at evenly spaced instants. If we sample every $T_s$ seconds, then $x[n] = x(n T_s)$. The **sample rate** is $f_s = 1/T_s$, measured in samples per second (hertz). At CD quality $f_s = 44{,}100$ Hz, so one second of mono audio is 44,100 floating-point numbers, and a three-minute song is about 8 million samples per channel. That number — millions of samples for seconds of sound — is the first thing that makes audio generation hard, and it is why every modern system compresses the waveform into a much shorter token sequence before a generative model ever touches it. The [neural-codec post](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) is entirely about that compression; here we care about the raw signal it starts from.

Two questions immediately arise and the rest of this post answers them. First: when I replace the continuous $x(t)$ with the sampled sequence $x[n]$, what do I lose? (Answer: nothing, below a precise frequency limit — the sampling theorem, Section 4.) Second: how do I see the frequencies inside $x[n]$ so I can analyze and manipulate them? (Answer: the discrete Fourier transform, next.)

A useful piece of vocabulary before we go on, because it threads through the whole series: an **amplitude** of $\pm 1.0$ is the digital full scale (DBFS), the largest number the format can hold; a real recording sits below that with headroom. Loudness is measured in **decibels**, a logarithmic ratio: a sound at amplitude $a$ relative to full scale is $20\log_{10}(a)$ dBFS, so half the amplitude is $-6$ dB, a tenth is $-20$ dB, and silence is $-\infty$. We use logarithms everywhere in audio for the same reason the ear does — perception of both loudness and pitch is roughly logarithmic, so a $\times 10$ change in physical amplitude feels like a fixed perceptual step. Keep that in mind: nearly every "spectrogram" you see is a *log*-magnitude spectrogram, because the linear magnitudes span six orders of magnitude and the quiet detail that carries timbre would otherwise be invisible.

Let us load the running example in code so it is concrete. We will use `torchaudio`, the audio companion to PyTorch, throughout.

```python
import torch
import torchaudio

fs = 16000          # sample rate in Hz (16 kHz is plenty for a demo tone)
dur = 1.0           # seconds
n = torch.arange(int(fs * dur))
t = n / fs          # the sample times in seconds

# A 440 Hz fundamental plus two harmonics — a slightly "richer" A note
x = (1.0 * torch.sin(2 * torch.pi * 440 * t)
     + 0.5 * torch.sin(2 * torch.pi * 880 * t)
     + 0.25 * torch.sin(2 * torch.pi * 1320 * t))
x = x / x.abs().max()                     # normalize to [-1, 1]

torchaudio.save("a_note.wav", x.unsqueeze(0), fs)
print(x.shape)        # torch.Size([16000])  -> 16,000 samples for 1 second
```

That tensor of 16,000 numbers *is* the signal. Everything below is about reading it.

## 2. The Fourier transform: any signal is a sum of sinusoids

The central claim of Fourier analysis is audacious and exactly true: **any reasonable signal can be written as a sum (or integral) of pure sinusoids of different frequencies, amplitudes, and phases.** The Fourier transform is the change of coordinates that finds those amplitudes and phases. It takes a signal indexed by time and returns a function indexed by frequency — the **spectrum**.

### 2.1 The continuous Fourier transform

For a continuous signal $x(t)$, the Fourier transform is

$$
X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-i 2\pi f t}\, dt
$$

and the inverse, which rebuilds the signal from its spectrum, is

$$
x(t) = \int_{-\infty}^{\infty} X(f)\, e^{i 2\pi f t}\, df.
$$

The complex exponential $e^{i 2\pi f t} = \cos(2\pi f t) + i\sin(2\pi f t)$ is just a sinusoid of frequency $f$ packaged with a quadrature partner so that magnitude and phase fall out of one complex number. The integral is a **correlation**: it slides a probe sinusoid of frequency $f$ against the signal and measures how much they overlap. Where the signal contains energy at $f$, the product stays in phase and the integral accumulates a large value; where it does not, the product oscillates and integrates to roughly zero. That is the whole mechanism. $X(f)$ is a complex number whose **magnitude** $|X(f)|$ says *how much* of frequency $f$ is present and whose **phase** $\angle X(f)$ says *where in its cycle* that component sits.

Why complex numbers? Because a real sinusoid at frequency $f$ has two degrees of freedom — amplitude and phase — and one real number cannot carry both. A complex value does: $X(f) = |X(f)| e^{i\angle X(f)}$ stores amplitude in its modulus and phase in its argument. This phase versus magnitude split matters enormously for audio. The magnitude spectrum is what a mel-spectrogram keeps and what most generative models predict; the phase is what they throw away and what a vocoder must reconstruct, because human hearing is far more sensitive to magnitude structure than to absolute phase — but get phase relationships between frequencies wrong and you get the metallic, buzzy artifacts I chased in the intro. We return to this when we discuss the mel pipeline in Section 8 and it is the central subject of the [representations post](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception).

### 2.2 The discrete Fourier transform (DFT), derived

The continuous transform is an integral over all time, which a computer cannot evaluate. For a finite sequence of $N$ samples $x[0], \dots, x[N-1]$ we use the **discrete Fourier transform**:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i 2\pi k n / N}, \qquad k = 0, 1, \dots, N-1.
$$

Let me derive where the frequencies $k$ live, because this is the part that trips people up. The DFT does not test arbitrary frequencies — it tests exactly $N$ of them, the ones that complete a whole number of cycles in the $N$-sample window. The $k$-th probe is $e^{i 2\pi k n / N}$, which over $n = 0 \dots N-1$ completes exactly $k$ full cycles. Its physical frequency, in hertz, is

$$
f_k = \frac{k}{N} f_s, \qquad k = 0, 1, \dots, N-1.
$$

So bin $k=0$ is DC (the mean, zero frequency); bin $k=1$ is one cycle per window, i.e. $f_s/N$ Hz; the **bin spacing** is $\Delta f = f_s/N$ Hz. This single quantity, $\Delta f = f_s/N$, is the **frequency resolution** and it will reappear as one arm of the uncertainty trade in Section 5. With $f_s = 16{,}000$ Hz and $N = 2048$, each bin is $16000/2048 \approx 7.8$ Hz wide.

The DFT is its own inverse up to a constant and a sign:

$$
x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\, e^{i 2\pi k n / N}.
$$

Why does this invert correctly? Because the probe sinusoids are **orthogonal**: for integer $k, m$,

$$
\sum_{n=0}^{N-1} e^{i 2\pi (k-m) n / N} =
\begin{cases} N & k = m \pmod N \\ 0 & \text{otherwise.} \end{cases}
$$

The $k=m$ case is a sum of $N$ ones. The $k \neq m$ case is a geometric series with ratio $r = e^{i 2\pi(k-m)/N} \neq 1$, summing to $(1 - r^N)/(1 - r) = 0$ since $r^N = e^{i 2\pi(k-m)} = 1$. Plug the analysis equation into the synthesis equation and this orthogonality collapses the double sum to exactly $x[n]$. The DFT is therefore an exact, lossless, invertible change of basis — a rotation of the $N$-dimensional signal vector onto a basis of complex sinusoids. Nothing is lost; you are looking at the same vector from a different angle.

### 2.3 The FFT: why we can afford this

Computed directly, the DFT is $O(N^2)$: $N$ output bins, each a sum over $N$ inputs. For $N = 2048$ that is 4 million complex multiply-adds per frame, and audio has hundreds of frames per second. The **Fast Fourier Transform** (Cooley–Tukey, 1965) computes the identical result in $O(N \log N)$ by recursively splitting the sum into even- and odd-indexed terms, each a half-size DFT, and combining them with $N/2$ "butterfly" operations. For $N = 2048$ that drops 4 million operations to about $2048 \times 11 \approx 22{,}000$ — a 180× saving — which is the only reason real-time spectral audio processing exists at all. The FFT is so much faster than the direct DFT that, as Section 7 shows, even *convolution* is done by FFT. Every `torch.fft.rfft` call below dispatches to an FFT internally.

### 2.4 Reading off the peak frequency in code

Let us take the FFT of our A note and confirm the math says what we built. We use `rfft` (real FFT), which returns only the non-redundant half of the spectrum — for a real input, $X[N-k] = \overline{X[k]}$, so the upper half is the complex conjugate mirror of the lower half and carries no new information.

```python
import torch

N = x.shape[0]                       # 16000 samples
X = torch.fft.rfft(x)                # complex spectrum, length N//2 + 1
freqs = torch.fft.rfftfreq(N, d=1/fs)  # the f_k = k * fs / N in Hz

mag = X.abs()                        # magnitude spectrum |X[k]|
# Find the three strongest bins (ignore the DC bin at index 0)
top = torch.topk(mag[1:], k=3).indices + 1
peak_freqs = freqs[top]
print(sorted(peak_freqs.tolist()))
# -> approximately [440.0, 880.0, 1320.0]  -> our fundamental + 2 harmonics
```

The FFT recovers exactly the three frequencies we synthesized, with relative magnitudes 1.0, 0.5, 0.25 as built. This is figure 1 made real: a signal that looks like a tangled wiggle in time is three clean spikes in frequency. The magnitude spectrum makes the structure legible, which is why we analyze audio here.

One subtlety worth flagging now because it bites everyone the first time: the bins are *discrete*, so a frequency that does not land exactly on a bin center smears across neighboring bins. Our 440 Hz tone with $f_s = 16{,}000$ and $N = 16{,}000$ happens to land exactly on bin 440 ($\Delta f = 1$ Hz here), so it is a single clean spike. Had we used $N = 1024$, the bin spacing would be $16000/1024 \approx 15.6$ Hz and 440 Hz falls between bins 28 (437.5 Hz) and 29 (453.1 Hz), so the energy spreads across both and a few neighbors — this is the leakage we will tame with windowing in Section 6. The fix is not "more samples of the same tone" but a longer analysis window or a window function; the resolution is $f_s/N$ and nothing about the signal changes it. Internalizing that $\Delta f = f_s/N$ — and that it is a property of your *analysis*, not of the *sound* — is half of using the FFT competently.

It is also worth seeing the phase, since the series leans on it repeatedly. The same `X` tensor carries it:

```python
import torch

phase = torch.angle(X)          # phase angle of each complex bin, in radians
# At the fundamental bin, the phase encodes WHERE in its cycle the sine started.
# Two signals with identical |X| but scrambled phase sound very different on
# transients, which is exactly why vocoders must reconstruct phase, not just |X|.
print(phase[440].item())        # the phase of the 440 Hz component
```

The magnitude told us *which* frequencies; the phase tells us *how they line up*, which governs the shape of the waveform in time. A mel-spectrogram keeps only the magnitude and discards this phase array entirely — the single biggest reason audio generation needs a vocoder, and a thread we pick up in Section 8.

## 3. Why frequency is the right lens (and a worked spectrum)

It is worth pausing on *why* this transform is so central rather than treating it as a formula. Three reasons, each load-bearing for the series.

**Hearing is frequency analysis.** As noted, the cochlea decomposes sound by frequency mechanically. So perceptually meaningful operations — equalization, pitch, timbre, masking — are natural in the frequency domain and awkward in time. The [perceptual representations post](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) builds the mel scale on exactly this fact.

**Convolution becomes multiplication.** Filtering, reverberation, and the action of any linear time-invariant system are convolutions in time, and Section 7 proves these become simple multiplications in frequency. A graphic equalizer is literally a per-bin multiply.

**Structure is sparse in frequency.** A musical note is a handful of harmonics — a few nonzero bins — but thousands of nonzero time samples. Speech is shaped by a few resonant formant peaks. Compression, denoising, and generative modeling all exploit this sparsity, which is invisible in the time domain and obvious in the spectrum.

There is a fourth reason that is worth spelling out because it explains *why* musical sounds are sparse in frequency at all: **any periodic signal is a sum of harmonics.** This is the Fourier *series* (the periodic ancestor of the transform). A note of pitch $f_0$ repeats every $1/f_0$ seconds, and any periodic waveform of that period can be written as

$$
x(t) = \sum_{k=1}^{\infty} A_k \sin(2\pi k f_0 t + \phi_k),
$$

a sum over *integer multiples* of the fundamental $f_0$ — the harmonics. A bowed string, a blown pipe, a vibrating vocal fold: all are (nearly) periodic, so all have spectra that are combs of harmonics at $f_0, 2f_0, 3f_0, \dots$. The *pitch* you hear is $f_0$ (where the comb starts and how tightly spaced its teeth are); the *timbre* is the envelope $\{A_k\}$ (how tall each tooth is). This is not a metaphor — it is the literal decomposition the FFT recovers, and it is why two instruments at the same pitch occupy the same bin positions but different bin heights. Percussion and noise (a cymbal, a fricative "sss") are the opposite: aperiodic, so their energy spreads continuously across frequency rather than concentrating on a comb. The FFT shows you instantly which kind of sound you have — harmonic comb versus broadband smear — a distinction that is the basis of pitch detection, voiced/unvoiced classification in speech, and the harmonic-plus-noise models that classical vocoders used before neural nets. When later TTS posts in this series talk about modeling "voiced" versus "unvoiced" frames, this comb-versus-smear distinction is exactly what they mean.

#### Worked example: reading a real spectrum

Take a 0.5-second clip of a violin playing concert A, sampled at $f_s = 44{,}100$ Hz, and FFT a $N = 4096$-sample frame. The bin spacing is $\Delta f = 44100/4096 \approx 10.8$ Hz. The fundamental at 440 Hz lands in bin $k = 440/10.8 \approx 41$. You will see a tall peak at bin 41, then progressively shorter peaks at bins 82 (880 Hz), 123 (1320 Hz), 164 (1760 Hz), and so on — the **harmonic series**, integer multiples of the fundamental, whose *relative* heights are the violin's timbre. The same note on a flute has the same peak positions but different relative heights, which is exactly why a flute and a violin sound different playing the same pitch: pitch is *where* the peaks are, timbre is *how tall* they are. No amount of staring at the two waveforms would have told you that; one FFT does. That separation — pitch versus timbre — is also what the cepstrum in Section 8 makes explicit.

## 4. The sampling theorem: when a list of numbers is enough

Now the foundational result that makes digital audio possible at all. We replaced the continuous $x(t)$ with samples $x[n] = x(nT_s)$. When is that safe? The **Nyquist–Shannon sampling theorem** gives the exact answer.

> **Theorem (Nyquist–Shannon).** If a continuous signal $x(t)$ contains no frequency components at or above $f_s/2$, then it is *completely determined* by its samples taken at rate $f_s$, and can be reconstructed exactly from them.

The quantity $f_N = f_s/2$ is the **Nyquist frequency**. At $f_s = 44{,}100$ Hz, $f_N = 22{,}050$ Hz — comfortably above the ~20 kHz ceiling of adult human hearing, which is precisely why CD audio chose 44.1 kHz. You sample at *twice* the highest frequency you care about because, as we will see, you need at least two samples per cycle to pin down a sinusoid's frequency and phase.

### 4.1 Why the factor of two — the sketch of the proof

Here is the intuition made rigorous. Sampling in time is multiplication by an impulse train $\sum_n \delta(t - nT_s)$. The Fourier transform of an impulse train is itself an impulse train, spaced $f_s$ apart in frequency. By the convolution theorem (Section 7), multiplying in time *convolves* in frequency, so sampling **replicates the signal's spectrum at every integer multiple of $f_s$**:

$$
X_{\text{sampled}}(f) = \frac{1}{T_s}\sum_{m=-\infty}^{\infty} X(f - m f_s).
$$

Picture the original spectrum, which is nonzero out to some maximum frequency $f_{\max}$, now stamped down repeatedly every $f_s$ along the frequency axis. If $f_{\max} < f_s/2$, the copies are spaced far enough apart that they do not touch — there is a clean gap between each replica — and an ideal low-pass filter that keeps only the band $[-f_s/2, f_s/2]$ recovers the original spectrum *exactly*, hence the original signal exactly. That low-pass reconstruction is the content of the theorem. But if $f_{\max} \geq f_s/2$, the replicas **overlap**: the tail of one copy lands on top of the next. Overlapping spectra add, and once added they cannot be separated. The high frequencies have masqueraded as lower ones. This overlap is **aliasing**, and it is irreversible.

### 4.2 Aliasing made concrete

When a frequency $f > f_N$ is sampled, it folds back into the audible band and appears as a phantom tone at

$$
f_{\text{alias}} = |f - f_s \cdot \text{round}(f/f_s)|.
$$

For a tone just above Nyquist the simpler form is $f_{\text{alias}} = f_s - f$.

#### Worked example: the 30 kHz tone that becomes 14.1 kHz

Sample a pure 30,000 Hz tone at $f_s = 44{,}100$ Hz with no anti-alias filter. Nyquist is 22,050 Hz, so 30 kHz is above it and *will* alias. Its mirror image is $f_{\text{alias}} = 44{,}100 - 30{,}000 = 14{,}100$ Hz. A tone you cannot even hear (most adults' hearing rolls off before 20 kHz, and 30 kHz is ultrasonic) reappears at 14.1 kHz, squarely audible, as a whistling artifact nobody asked for. The energy did not vanish; it folded down. This is why every analog-to-digital converter places an **anti-alias filter** — a steep analog low-pass cutting everything above $f_N$ — *before* the sampler. Filter first, then sample. Get the order wrong and no later processing can undo it.

![A before-and-after figure contrasting a 30 kilohertz tone that folds down to a phantom 14.1 kilohertz tone without an anti-alias filter against a clean band when a low-pass filter is applied before sampling.](/imgs/blogs/the-mathematics-of-audio-signals-4.png)

The same folding hurts generative models, not just converters. A neural vocoder or codec decoder that upsamples a low-rate feature to a high-rate waveform can introduce energy above its internal Nyquist, which aliases into audible buzz — exactly the 8 kHz spike from the intro, half of that model's 16 kHz rate. Modern designs (BigVGAN's anti-aliased activations, filtered upsampling) exist specifically to suppress this. The math you are reading is the reason those design choices are not optional.

### 4.3 Demonstrating aliasing by undersampling, in code

Let us break it on purpose. We synthesize a clean 6 kHz tone at a high rate, then naively keep every other sample (decimate by 2 without filtering) so the effective rate is too low, and watch the peak move to the wrong place.

```python
import torch

fs_hi = 16000
t = torch.arange(int(fs_hi)) / fs_hi
tone = torch.sin(2 * torch.pi * 6000 * t)   # a 6 kHz tone, fine at 16 kHz

# Naive downsample by 2 WITHOUT an anti-alias filter -> effective fs = 8000
bad = tone[::2]
fs_lo = fs_hi // 2                            # 8000 Hz; Nyquist now 4000 Hz

Xb = torch.fft.rfft(bad)
fb = torch.fft.rfftfreq(bad.shape[0], d=1/fs_lo)
peak = fb[Xb.abs().argmax()]
print(float(peak))     # ~2000.0 Hz  -> the 6 kHz tone aliased to 8000 - 6000 = 2000 Hz
```

The 6 kHz tone, now above the new 4 kHz Nyquist, reappears at $8000 - 6000 = 2000$ Hz. Had we applied a proper low-pass filter before decimating (which is exactly what `torchaudio.transforms.Resample` does internally), the tone would have been removed rather than folded. The lesson generalizes to every place you change sample rate in an audio pipeline: never decimate without filtering first.

The right way, for contrast, is one line:

```python
import torchaudio

# Resample 16 kHz -> 8 kHz CORRECTLY: this low-pass filters before decimating,
# so frequencies above the new 4 kHz Nyquist are removed, not aliased.
resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
good = resampler(tone.unsqueeze(0)).squeeze(0)
# The 6 kHz tone is now GONE (it was above 4 kHz) rather than folded to 2 kHz.
```

There is one more practical wrinkle that the sampling theorem warns about: **real anti-alias filters are not ideal "brick walls".** The theorem assumes a perfect low-pass that passes everything below $f_N$ and kills everything above with a vertical cliff. Real analog and digital filters have a finite *transition band* — they roll off over some range — so in practice you sample a little faster than twice your highest frequency to leave room for the filter to do its work. This is exactly why 44.1 kHz (not a tidy 40 kHz for 20 kHz hearing) was chosen: the extra 4 kHz of Nyquist headroom above 20 kHz gives the anti-alias filter room to transition without touching the audible band. The "magic" number 44.1 kHz is the sampling theorem plus a real filter's slope, nothing more.

## 5. Bit depth and quantization noise: the 6.02b + 1.76 dB law

Sampling discretizes *time*. We also have to discretize *amplitude*, because a sample is stored as a finite-precision number. This is **quantization**, controlled by the **bit depth** $b$: with $b$ bits per sample you have $2^b$ available amplitude levels (16-bit audio has 65,536 levels; 24-bit has about 16.7 million). Rounding each true amplitude to the nearest level introduces an error — **quantization noise** — and we can derive exactly how loud it is.

### 5.1 Deriving the signal-to-noise ratio

Model the signal as filling a full-scale range of width $2A$ (peak-to-peak $2A$, so amplitude $\pm A$). With $b$ bits, the $2^b$ levels are spaced

$$
\Delta = \frac{2A}{2^b}.
$$

Assume the quantization error $e$ on each sample is uniformly distributed over one step, $e \in [-\Delta/2, +\Delta/2]$ — a standard and well-justified model for a busy signal. The mean-square error (noise power) of a uniform distribution of width $\Delta$ is its variance:

$$
\sigma_e^2 = \frac{1}{\Delta}\int_{-\Delta/2}^{\Delta/2} e^2\, de = \frac{\Delta^2}{12}.
$$

For the signal, take a full-scale sinusoid of amplitude $A$. Its power (mean square) is

$$
\sigma_s^2 = \frac{A^2}{2}.
$$

The signal-to-noise ratio is the ratio of these powers. Substitute $\Delta = 2A/2^b$:

$$
\text{SNR} = \frac{\sigma_s^2}{\sigma_e^2}
= \frac{A^2/2}{\Delta^2/12}
= \frac{A^2/2}{(2A/2^b)^2 / 12}
= \frac{A^2/2 \cdot 12}{4A^2 / 2^{2b}}
= \frac{12}{8}\, 2^{2b} = \frac{3}{2}\, 2^{2b}.
$$

Now convert to decibels ($10\log_{10}$ of a power ratio):

$$
\text{SNR}_{\text{dB}} = 10\log_{10}\!\left(\frac{3}{2}\,2^{2b}\right)
= 10\log_{10}(1.5) + 2b \cdot 10\log_{10}(2).
$$

Since $10\log_{10}(2) \approx 3.0103$ and $10\log_{10}(1.5) \approx 1.7609$,

$$
\boxed{\ \text{SNR}_{\text{dB}} \approx 6.02\, b + 1.76\ \text{dB}.\ }
$$

This is the celebrated **6 dB per bit** rule. Each extra bit of depth doubles the number of levels, halves the step size, and buys you about 6 dB of dynamic range. The constant 1.76 dB is the bonus from the full-scale sinusoid assumption.

#### Worked example: 16-bit versus 24-bit noise floor

For 16-bit CD audio: $\text{SNR}_{\text{dB}} \approx 6.02 \times 16 + 1.76 = 98.08$ dB. That is the gap between the loudest signal the format holds and its noise floor — roughly the difference between a jet engine and a whisper, which is why 16 bits is transparent for playback. For 24-bit studio audio: $6.02 \times 24 + 1.76 = 146.2$ dB, far exceeding any analog electronics' own noise, which is why 24-bit is used in production (headroom for editing) but is overkill for delivery. For 8-bit: $6.02 \times 8 + 1.76 = 49.9$ dB, audibly hissy — the gritty sound of old samplers. The takeaway for a generative pipeline: train and generate at 16-bit equivalent precision and your quantization noise sits ~98 dB down, inaudible; the artifacts you hear in a bad model are *never* quantization — they are spectral, from the model or the vocoder. Knowing the noise floor tells you where *not* to look.

### 5.2 Two refinements that matter in practice: dither and floating point

The uniform-error model above assumes the signal is "busy" enough that successive quantization errors are uncorrelated. For very quiet or very pure signals that assumption breaks — the error becomes *correlated* with the signal and turns into audible harmonic distortion rather than benign hiss. The fix, **dither**, is delightfully counterintuitive: you add a tiny amount of random noise (about one quantization step) *before* quantizing. This decorrelates the error, trading a faint constant hiss (which the ear ignores) for distortion (which the ear catches). It is the reason a well-mastered 16-bit file sounds clean even on a fade-to-silence where the signal uses only the bottom few bits. The deeper point for ML: when you quantize a neural codec's latent (the [RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) covers this), the *same* correlated-error problem appears, and the *same* idea — injecting noise during training, or using a straight-through estimator that behaves like dither — keeps the gradients honest.

Most audio ML actually runs in **32-bit floating point**, not fixed-point integers, inside the model. Floating point spends its bits differently: it gives roughly *constant relative* precision across amplitudes (a fixed number of significant figures) rather than constant absolute step size, so its effective SNR is enormous and signal-dependent. You only meet the $6.02b + 1.76$ law at the boundaries — the 16-bit WAV you load and the 16-bit WAV you save. Inside the pipeline, quantization noise is a non-issue; the law's value is telling you the *format's* ceiling so you can confirm, as above, that your audible problems live in the spectrum and the model, not in the number format.

#### Worked example: a fade-out and the bottom bits

Take a note fading to silence over two seconds at 16-bit. Near the end, the amplitude drops to $-80$ dBFS, which is $80/6.02 \approx 13$ bits below full scale — so only the bottom 3 bits of the 16 are doing any work. Without dither, those 3 bits quantize a smooth sinusoid into a visibly stepped staircase whose error is harmonically related to the tone: you hear "granular" distortion, a faint buzzing on the tail. Add 1-LSB triangular dither and the staircase dissolves into a smooth tone riding on inaudible hiss roughly 98 dB down. The numbers are the whole argument: 3 working bits is $6.02 \times 3 + 1.76 \approx 20$ dB of SNR locally — clearly audible error — which dither converts into perceptually transparent noise. This is the audio-domain version of why generative models add noise during training to smooth a hard discretization.

## 6. The STFT and the time-frequency uncertainty trade

The plain FFT gives one spectrum for an entire signal — useless for audio that *changes*, like speech or music, where the whole point is that the frequencies evolve over time. The fix is the **short-time Fourier transform** (STFT): chop the signal into short overlapping frames, FFT each one, and stack the results into a time-frequency image — the **spectrogram**, the single most important feature in audio ML.

![A vertical stack figure showing the short-time Fourier transform pipeline from waveform through framing, Hann windowing, per-frame FFT, magnitude, to the final frequency-by-time spectrogram image.](/imgs/blogs/the-mathematics-of-audio-signals-2.png)

### 6.1 The STFT formalized

Formally, with a window function $w[n]$ of length $N$, a hop (stride) of $H$ samples between frames, and frame index $m$:

$$
X[m, k] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-i 2\pi k n / N}.
$$

It is just the DFT of each windowed frame. Three parameters define it, and each is a knob you will set in every audio project:

- **`n_fft` = $N$**, the frame and FFT length. It sets the frequency resolution $\Delta f = f_s/N$ (more bins, finer pitch) and the time extent of each frame, $N/f_s$ seconds.
- **`hop_length` = $H$**, the stride between frames. It sets the time resolution / frame rate $f_s/H$ frames per second and the overlap. Common choice: $H = N/4$ (75% overlap).
- **`window` = $w[n]$**, the taper applied to each frame, usually a **Hann** window $w[n] = 0.5\,(1 - \cos(2\pi n/N))$.

### 6.2 Why window at all — spectral leakage

Why taper each frame instead of using the raw samples (a rectangular window)? Because chopping a signal abruptly is itself a multiplication by a rectangular box in time, which **convolves the true spectrum with the box's transform** — a sinc function with large side-lobes (the convolution theorem again, Section 7). A pure 440 Hz tone that falls between bins then smears its energy across many bins through those side-lobes; this is **spectral leakage**. A Hann window tapers smoothly to zero at the frame edges, so its transform has much smaller side-lobes (about -31 dB versus the rectangle's -13 dB), trading a slightly wider main peak for far less leakage. The cost is that the edges of each frame are attenuated — which is why frames overlap, so every sample is well-represented in *some* frame.

### 6.3 The COLA constraint: when can you reconstruct?

If you want to *invert* an STFT — modify a spectrogram and turn it back into a waveform, as every vocoder and many editing tools do — the window and hop must satisfy the **Constant Overlap-Add (COLA)** condition: the shifted, squared windows must sum to a constant across all sample positions,

$$
\sum_{m} w^2[n - mH] = C \quad \text{for all } n.
$$

When COLA holds, the overlap-add of inverse-FFT'd frames reconstructs the signal exactly (perfect reconstruction). A Hann window with 75% overlap ($H = N/4$) satisfies COLA, which is the practical reason that hop is the default. Violate COLA and inverse-STFT introduces amplitude ripple — a subtle warble you can hear. `torchaudio.transforms.InverseSpectrogram` assumes you respected it.

There is a crucial asymmetry here that motivates half of the audio-generation field. Inverting a *complex* STFT (magnitude **and** phase) under COLA is exact and trivial — overlap-add the inverse FFTs. But a model that predicts a **magnitude** spectrogram (or a mel-spectrogram) has thrown the phase away, and now you must *guess* a phase that is consistent with the predicted magnitudes. The classical guesser is the **Griffin-Lim** algorithm: start from random phase, inverse-STFT to a waveform, forward-STFT back, keep the model's magnitudes but adopt the *resulting* phase, and iterate. It converges slowly to a phase that is approximately consistent, but the result is the slightly metallic, "robotic" timbre that gives Griffin-Lim audio its dated sound — because no phase perfectly reconstructs from magnitude alone. This is exactly why neural vocoders (HiFi-GAN, BigVGAN, Vocos) replaced Griffin-Lim: they *learn* to produce a perceptually correct waveform from the magnitude (or mel) directly, phase included, rather than iterating toward a compromise. The COLA math tells you when reconstruction is exact; the phase problem tells you why, in practice, it usually is not, and why a learned vocoder earns its place.

```python
import torchaudio

# Griffin-Lim: recover a waveform from a MAGNITUDE spectrogram by iterating
# toward a consistent phase. Workable, but the classic "robotic" artifact is here.
spec_mag = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=1.0)(x)
gl = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256, n_iter=32)
x_recon = gl(spec_mag)          # approximate; phase was guessed, not known
print(x_recon.shape)            # a waveform back, minus the original phase fidelity
```

### 6.4 The uncertainty principle, derived

Here is the deep law that governs every spectrogram, and the one most worth internalizing: **you cannot have arbitrarily good time resolution and frequency resolution at the same time.** Make the window short and you locate events precisely in time but the few samples give you coarse, blurry frequency bins. Make it long and you resolve pitch finely but smear everything in time. This is not an engineering limitation; it is a mathematical theorem, the signal-processing form of the Heisenberg uncertainty principle.

Define the time spread $\Delta t$ and frequency spread $\Delta f$ of a window as the standard deviations of $|w(t)|^2$ and $|W(f)|^2$ respectively (their effective widths). The Gabor limit states:

$$
\Delta t \cdot \Delta f \ \geq\ \frac{1}{4\pi}.
$$

The product is bounded below by a constant. Sketch of why: the time spread is $\Delta t^2 = \int t^2 |w(t)|^2 dt / \int |w(t)|^2 dt$ and the frequency spread is the analogous integral for $W(f)$. A frequency-domain factor of $f$ corresponds, by the Fourier derivative property, to a derivative $\frac{d}{dt}$ in time. The product $\Delta t \cdot \Delta f$ then becomes, via the **Cauchy–Schwarz inequality** applied to $w(t)$ and its derivative $w'(t)$ (this is the same algebra that proves Heisenberg's principle for $x$ and $p$ in quantum mechanics), bounded below by a constant, with equality only for a **Gaussian** window — which is precisely why the Gaussian is the optimal time-frequency localization and why the "Gabor transform" uses it. The discrete consequence is brutally simple and you can read it straight off the parameters:

$$
\Delta t = \frac{N}{f_s} \ (\text{seconds per frame}), \qquad \Delta f = \frac{f_s}{N}\ (\text{Hz per bin}), \qquad \Delta t \cdot \Delta f = 1.
$$

Their product is *exactly 1 and independent of $N$* — every sample rate, every FFT size, the same fixed budget. Doubling $N$ halves $\Delta f$ (better pitch) but doubles $\Delta t$ (worse timing). You are sliding along a fixed hyperbola, never beating it.

![A two-by-three grid figure showing short windows giving fine time and coarse frequency tiles good for transients, versus long windows giving coarse time and fine frequency tiles good for resolving pitch.](/imgs/blogs/the-mathematics-of-audio-signals-3.png)

#### Worked example: choosing N for speech versus music

For **speech** at $f_s = 16{,}000$ Hz you care about catching fast consonant onsets, so you want fine time resolution. Pick $N = 400$ (25 ms): $\Delta t = 400/16000 = 25$ ms, $\Delta f = 16000/400 = 40$ Hz. Crisp timing, frequency bins coarse but fine enough for speech formants. This 25 ms / 10 ms-hop frame is the near-universal speech-recognition and TTS default. For **music** at $f_s = 44{,}100$ Hz where you need to separate closely spaced notes (a semitone at low pitch is only a few Hz apart), you want fine frequency. Pick $N = 4096$ (93 ms): $\Delta t = 93$ ms, $\Delta f = 44100/4096 \approx 10.8$ Hz. Now you resolve individual notes in a chord, at the cost of blurring fast attacks. There is no universally right $N$; there is only the trade, and the trade is the theorem. A practical dodge when you genuinely need both — multi-resolution STFT, several window sizes in parallel — is exactly why GAN vocoders use a *multi-resolution STFT loss*, computing the spectral error at several $N$ at once. That design choice is the uncertainty principle showing up in a loss function.

### 6.5 Building an STFT from framed FFTs, in code

To make the STFT concrete, build it by hand from per-frame FFTs and confirm it matches `torch.stft`.

```python
import torch

def manual_stft(x, n_fft=1024, hop=256):
    win = torch.hann_window(n_fft)
    frames = []
    for start in range(0, len(x) - n_fft + 1, hop):
        frame = x[start:start + n_fft] * win     # window the frame
        frames.append(torch.fft.rfft(frame))     # FFT it -> n_fft//2 + 1 bins
    return torch.stack(frames, dim=1)            # shape: (freq, time)

S_manual = manual_stft(x, n_fft=1024, hop=256)

# torchaudio's built-in, which we trust as ground truth
S_ref = torch.stft(x, n_fft=1024, hop_length=256,
                   window=torch.hann_window(1024),
                   center=False, return_complex=True)

err = (S_manual - S_ref).abs().max()
print(S_manual.shape, float(err))   # (513, ~58)  and err ~ 1e-6 (they match)
```

The hand-rolled framing-windowing-FFT loop reproduces the library STFT to floating-point precision. There is no magic in `torch.stft` — it is exactly figure 2's pipeline: frame, window, FFT, stack. For mel features you would then multiply this by a mel filterbank (Section 8).

A note on the output shape, because it confuses people coming from images: a spectrogram from `n_fft=1024` has $1024/2 + 1 = 513$ frequency rows (the non-redundant half of the FFT plus DC and Nyquist) and however many time columns the framing produced. For a 1-second 16 kHz signal with `hop=256` that is about $16000/256 \approx 62$ columns. So a spectrogram is a $513 \times 62$ "image" — far fewer pixels than a photo, which is one reason 2D convolutional architectures borrowed from vision (and the diffusion machinery from the [image series](/blog/machine-learning/image-generation/diffusion-from-first-principles)) port so naturally onto mel-spectrograms. The catch that the [representations post](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) dwells on: unlike an RGB image, the spectrogram is a *transform* of a signal that must be inverted back to a waveform, and that inversion is where phase reconstruction — and vocoders — enter.

In practice you would call the optimized transform:

```python
import torchaudio

spec = torchaudio.transforms.Spectrogram(
    n_fft=1024, hop_length=256, power=2.0)(x)   # magnitude-squared spectrogram
print(spec.shape)    # torch.Size([513, ~60])  -> 513 freq bins x ~60 time frames
```

## 7. Convolution and the convolution theorem: filtering is multiplication

The last pillar, and the one that connects DSP directly to the convolutional layers in audio networks. **Convolution** is the operation that describes how a linear time-invariant (LTI) system — any fixed filter, the acoustics of a room, the response of a microphone — transforms an input signal. For discrete signals it is

$$
(x * h)[n] = \sum_{m} x[m]\, h[n - m].
$$

The sequence $h$ is the system's **impulse response**: literally what comes out when you feed in a single unit impulse $\delta[n]$ (a 1 at $n=0$, zeros elsewhere). Because an LTI system is fully characterized by its impulse response, convolving any input with $h$ tells you the output — that is the defining property of LTI systems. A room's reverb is its impulse response (clap once, record the decay, and you can convolve any dry signal to place it in that room — this is exactly how convolution reverb plugins work). A low-pass filter is a convolution with a sinc-like kernel. The cochlea's frequency channels are approximately convolutions with band-pass filters.

### 7.1 The convolution theorem, proved

Here is the result that makes the frequency domain indispensable: **convolution in time equals multiplication in frequency.**

$$
\boxed{\ x * h \ \xleftrightarrow{\ \mathcal{F}\ }\ X \cdot H\ }
\qquad\text{i.e.}\qquad \mathcal{F}\{x * h\}[k] = X[k]\, H[k].
$$

The proof is short. Take the DFT of the convolution (using circular convolution, which the DFT implies, on length-$N$ sequences):

$$
\mathcal{F}\{x*h\}[k] = \sum_{n=0}^{N-1}\left(\sum_{m=0}^{N-1} x[m]\,h[n-m]\right) e^{-i 2\pi k n / N}.
$$

Swap the order of summation and substitute $\ell = n - m$ (so $n = \ell + m$, and the exponential factors):

$$
= \sum_{m=0}^{N-1} x[m]\, e^{-i 2\pi k m / N} \sum_{\ell} h[\ell]\, e^{-i 2\pi k \ell / N}
= X[k] \cdot H[k].
$$

The exponential $e^{-i 2\pi k (\ell+m)/N}$ factored cleanly into a part depending only on $m$ and a part depending only on $\ell$, decoupling the double sum into the product of two single sums — which are exactly $X[k]$ and $H[k]$. That clean factorization is the entire reason the theorem holds, and it is the same factorization that made the DFT inverse work in Section 2.

### 7.2 Why this matters: fast convolution and graphic EQ

Two consequences. First, **fast convolution**: a direct convolution of an $N$-sample signal with an $M$-tap filter costs $O(NM)$. But by the theorem you can instead FFT both ($O(N\log N)$), multiply pointwise ($O(N)$), and inverse-FFT ($O(N\log N)$) — total $O(N\log N)$, dramatically faster for long filters. Long reverbs (impulse responses of tens of thousands of taps) are *only* feasible this way. Second, **filtering is a per-bin gain**: a graphic equalizer that boosts the bass and cuts the treble is, in the frequency domain, just multiplying each bin by a gain — $H[k]$ is the EQ curve. To design a filter, you draw its frequency response $H[k]$ and inverse-FFT to get the kernel $h[n]$.

![A before-and-after figure showing convolution computed as a costly sliding sum in the time domain versus the cheaper FFT, pointwise multiply, and inverse FFT path in the frequency domain producing the identical output.](/imgs/blogs/the-mathematics-of-audio-signals-6.png)

### 7.3 Verifying the theorem in code

Let us prove it numerically: convolve in time, multiply in frequency, and show they agree.

```python
import torch

sig = torch.randn(2048)
kernel = torch.tensor([0.25, 0.5, 0.25])      # a tiny smoothing (low-pass) filter

# (a) Direct convolution in the time domain
y_time = torch.nn.functional.conv1d(
    sig.view(1, 1, -1),
    kernel.flip(0).view(1, 1, -1),            # conv1d cross-correlates; flip for true conv
    padding=len(kernel) - 1
).flatten()[:sig.shape[0] + len(kernel) - 1]

# (b) Multiplication in the frequency domain (linear conv via zero-padded FFT)
L = sig.shape[0] + kernel.shape[0] - 1
Y = torch.fft.rfft(sig, n=L) * torch.fft.rfft(kernel, n=L)
y_freq = torch.fft.irfft(Y, n=L)

print(float((y_time - y_freq).abs().max()))   # ~1e-6  -> they match
```

The two paths agree to floating-point precision: time-domain convolution and frequency-domain multiplication are the same operation, exactly as the theorem promises. (Note the zero-padding to length $N+M-1$ — that converts the DFT's *circular* convolution into the *linear* convolution we actually want, avoiding wrap-around at the edges.)

#### Worked example: when to FFT a convolution and when not to

Convolution reverb places a dry voice into a real room by convolving it with the room's impulse response. A concert-hall impulse response at 44.1 kHz is about 3 seconds long — roughly $M = 132{,}000$ taps. Convolving a 10-second clip ($N = 441{,}000$ samples) directly costs on the order of $N \times M \approx 5.8 \times 10^{10}$ multiply-adds — tens of seconds of CPU work for a few seconds of audio, an RTF far above 1. The FFT path costs $O((N+M)\log(N+M)) \approx 5.7 \times 10^5 \times 19 \approx 1.1 \times 10^7$ operations — roughly **5,000× fewer**. This is not a marginal optimization; it is the difference between feasible and infeasible, and it is why every convolution-reverb plugin and every long-filter audio operation uses FFT-based (overlap-add or overlap-save) convolution. The flip side: for a *short* filter — a 3-tap smoother, or a small conv kernel in a network — the FFT's overhead is not worth it and direct convolution wins. The crossover is roughly when the filter exceeds ~64 taps. Neural networks live on the short side (kernels of 3–7), which is exactly why deep-learning frameworks implement `conv1d` as a direct (im2col / GEMM) operation, not an FFT.

### 7.4 Why convolutions are everywhere in audio networks

This is where DSP meets deep learning. A 1D convolutional layer in an audio network is *exactly* a bank of learnable filters convolving the signal — the network learns its own $h$ kernels instead of you hand-designing them. So a 1D conv stack is a learnable, multi-band filterbank, which is why convolutional front-ends dominate audio models: they are the natural learnable version of the spectral analysis we have been doing by hand.

The killer application is **WaveNet**'s dilated causal convolution. Two properties matter. **Causal**: the filter only looks at past samples ($h[n]$ for $n \geq 0$), never the future, so the model can generate sample by sample in order — essential for autoregressive synthesis. **Dilated**: the filter skips input positions with a gap (dilation) that *doubles* each layer (1, 2, 4, 8, …). A stack of $L$ dilated layers with kernel size 2 has a receptive field of $2^L$ samples — exponential reach from linear depth. With $L = 10$ you see $2^{10} = 1024$ past samples (64 ms at 16 kHz); stack a few such blocks and you cover hundreds of milliseconds, enough for the model to capture both the fine waveform shape and the longer-range structure of speech. This is how WaveNet modeled raw 16 kHz audio directly, sample by sample, before codecs made that affordable. The [autoregressive audio post](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm) builds the full model on this primitive.

Why dilation rather than just deeper plain convolutions or bigger kernels? A plain causal conv stack grows its receptive field only *linearly* in depth — $L$ layers of kernel size $k$ reach $L(k-1)+1$ samples. To cover 1024 samples with kernel-2 layers you would need ~1023 layers, which is untrainable. Dilation buys the same reach in $\log_2(1024) = 10$ layers. And unlike simply using a kernel of 1024 (which would have 1024 weights per channel and blur over everything in between), the dilated stack keeps each layer's kernel tiny (2 weights) while the *composition* of skips covers the full span — far fewer parameters, and a hierarchy where early layers see fine local detail and deep layers see coarse long-range context. That hierarchical receptive field is the same idea WaveNet, then dilated TCNs, then the convolutional codec encoders (EnCodec, DAC) all reuse. It is the audio answer to the question "how do I see far without paying quadratically," and it predates the attention-based answer that transformers give for the same problem on token sequences.

Here is the receptive-field arithmetic made concrete as a small table you can reason from:

| Dilations | Layers | Receptive field (samples) | At 16 kHz | At 24 kHz |
|---|---|---|---|---|
| 1, 2, 4, 8 | 4 | 16 | 1.0 ms | 0.67 ms |
| 1…256 | 9 | 512 | 32 ms | 21 ms |
| 1…512 | 10 | 1024 | 64 ms | 43 ms |
| 3 × (1…512) | 30 | ~3072 | 192 ms | 128 ms |

The last row — three stacked dilation blocks, WaveNet's actual configuration — reaches nearly 200 ms of context at 16 kHz with only 30 layers, enough to span a syllable. That is the dilated-convolution receptive-field math doing exactly what figure 7 shows.

![A branching graph figure showing stacked dilated causal convolution layers with dilation doubling from one to two to four, growing the receptive field exponentially while summing through gated skip connections to predict the next sample.](/imgs/blogs/the-mathematics-of-audio-signals-7.png)

## 8. Mel filterbank and cepstrum: derived transforms on top of the FFT

The raw magnitude spectrum has hundreds of linearly spaced bins, but human pitch perception is roughly *logarithmic* — we hear the octave from 100 to 200 Hz as the same musical distance as 1000 to 2000 Hz, even though one spans 100 Hz and the other 1000 Hz. The **mel scale** warps frequency to match this. A common formula is

$$
m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right),
$$

linear below ~1 kHz and logarithmic above. The **mel-spectrogram** — the single most-used feature in all of audio ML — is built by multiplying the magnitude spectrum by a bank of (typically 80) overlapping triangular filters spaced evenly on the mel scale, then taking the log. That is two simple, fixed, differentiable operations stacked on the FFT: a linear projection (the filterbank is just a matrix multiply) and a log. Cheap, perceptually aligned, and the reason nearly every TTS and music model predicts a mel-spectrogram rather than a raw waveform — it is a compact (80 bins versus 513), perceptually weighted, smooth target. Crucially the mel-spectrogram is **magnitude-only**: it discards phase, which is why it needs a vocoder (Griffin-Lim or a neural net like HiFi-GAN) to turn it back into a waveform, reconstructing the missing phase. Why throwing away phase still works perceptually is the subject of the [representations post](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception).

![A vertical stack figure showing the magnitude FFT feeding a mel filterbank and log to produce the mel-spectrogram, and separately a discrete cosine transform of the log spectrum producing the MFCC cepstrum that separates pitch from timbre.](/imgs/blogs/the-mathematics-of-audio-signals-8.png)

The **cepstrum** goes one transform further and is a beautiful idea. Recall from Section 3 that a sound's spectrum is a product of a slowly varying envelope (the timbre — where the broad peaks sit, the resonances) times a fast harmonic comb (the pitch — the closely spaced harmonics). Taking the **log** turns that product into a *sum*: $\log(\text{envelope} \times \text{harmonics}) = \log(\text{envelope}) + \log(\text{harmonics})$. Now apply a discrete cosine transform (DCT, a real cousin of the FFT) to the log-spectrum. The slowly varying envelope becomes low "quefrency" coefficients; the fast harmonic comb becomes a high-quefrency peak whose position is the pitch period. The cepstrum has thus **separated pitch from timbre** into different coefficients — exactly the violin-versus-flute distinction from Section 3, now made explicit and machine-readable. Keep the first 13 or so coefficients and you get the **MFCCs** (mel-frequency cepstral coefficients) that powered decades of speech recognition. It is FFT → log → DCT, three fixed transforms, and it cleanly factors the two perceptual properties we care about most.

```python
import torchaudio

# Mel-spectrogram: the workhorse feature, ~80 perceptually-spaced bins
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=fs, n_fft=1024, hop_length=256, n_mels=80
)(x)
log_mel = torch.log(mel + 1e-6)        # log-compress, as models expect
print(log_mel.shape)                    # torch.Size([80, ~60])

# MFCC cepstrum: FFT -> mel -> log -> DCT, keep 13 coefficients
mfcc = torchaudio.transforms.MFCC(
    sample_rate=fs, n_mfcc=13,
    melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80}
)(x)
print(mfcc.shape)                       # torch.Size([13, ~60])
```

## 9. The DSP parameters as a control panel

Step back and see the four knobs together, because choosing them is the very first decision in any audio project, before you pick a model.

![A matrix figure mapping the four core DSP parameters of sample rate, bit depth, FFT size, and hop length to what each one controls, the effect of raising it, and the trade-off it carries.](/imgs/blogs/the-mathematics-of-audio-signals-5.png)

| Parameter | Symbol | Controls | Raise it → | The cost |
|---|---|---|---|---|
| Sample rate | $f_s$ | Top frequency = $f_s/2$ | Capture brighter highs (16→44.1 kHz adds "air") | More samples/sec → more compute and longer sequences |
| Bit depth | $b$ | Dynamic range, $\approx 6.02b + 1.76$ dB | Lower noise floor (16-bit → 98 dB) | Bigger files; beyond ~20 bits it is below analog noise |
| FFT size | $N$ | Frequency resolution $\Delta f = f_s/N$ | Sharper pitch, separate close notes | Worse time resolution $\Delta t = N/f_s$ (smear onsets) |
| Hop length | $H$ | Frame rate $f_s/H$ | Smoother time tracking, more overlap | More frames → slower, heavier spectrogram |

A few canonical settings worth memorizing, all of which fall straight out of the math above:

| Task | $f_s$ | $N$ (`n_fft`) | $H$ (`hop`) | `n_mels` | Why |
|---|---|---|---|---|---|
| Speech (TTS / ASR) | 16–24 kHz | 1024 (≈42 ms) | 256 | 80 | Speech tops out ~8 kHz; fine time for consonants |
| Music / general audio | 44.1 kHz | 2048 (≈46 ms) | 512 | 128 | Need the full band and pitch resolution |
| Neural codec input | 24–48 kHz | (model-internal) | — | — | Codec learns its own filters; rate sets quality |
| Transient analysis | any | 256–512 | 64–128 | — | Short window = sharp time for clicks/onsets |

These are starting points, not laws. The only law is the uncertainty trade: you do not get sharp time and sharp frequency for free, ever.

## 10. Case studies: the math showing up in real systems

These are not toy facts — every modern audio system is a place where this DSP is load-bearing. A few concrete, citable examples.

**WaveNet (van den Oord et al., 2016)** modeled raw audio at 16 kHz directly with stacked dilated causal convolutions, dilations doubling 1, 2, 4, …, 512, giving a receptive field of thousands of samples — the exponential-reach math from Section 7 is the entire reason it could capture both fine waveform detail and longer speech structure. It was famously slow at inference (autoregressive over every single sample, real-time factor far above 1, i.e. far slower than real time), which is precisely why the field moved to codec tokens and parallel vocoders.

**HiFi-GAN (Kong et al., 2020)** is a mel-spectrogram-to-waveform vocoder built on transposed 1D convolutions, and its training loss is a **multi-resolution STFT loss** — it computes the spectral error at several FFT sizes simultaneously (e.g. $N$ = 512, 1024, 2048). That is a direct engineering response to the uncertainty principle from Section 6: no single $N$ captures both transients and pitch, so you supervise at several resolutions at once. On a V100 GPU HiFi-GAN synthesizes audio far faster than real time (reported real-time factors well above 100×), which is why it became the default vocoder and why the [GAN vocoder post](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis) treats it as the speed workhorse.

**EnCodec (Défossez et al., 2022)** and **DAC (Descript Audio Codec, 2023)** are convolutional encoder-decoder codecs whose entire job is to push the bitrate-quality trade — representing 24–44.1 kHz audio at a few kbps via residual vector quantization. Their decoders are upsampling convolution stacks, and a real engineering concern is exactly the upsampling-aliasing from Section 4: DAC adopted a **snake activation** and careful anti-aliased upsampling specifically to suppress the high-frequency artifacts that naive upsampling folds into the audible band. The codecs are the subject of the [codec post](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound); the math here is *why* their design choices exist.

**BigVGAN (Lee et al., 2023)** made the anti-aliasing point explicit: it adds anti-aliased (filtered) nonlinear activations inside the generator because passing a band-limited signal through a pointwise nonlinearity *generates new high frequencies* that then alias — the Nyquist math from Section 4 applied inside the network, not just at the converter. Fixing it audibly cleaned up the high band.

To make the trade concrete, here is how the DSP choices propagate into the numbers you would actually report when comparing systems. These are representative figures from the literature and standard configurations, not measurements of one specific run — treat them as order-of-magnitude anchors and re-measure on your own hardware with a fixed seed and warm-up.

| System / setting | Sample rate | Representation | Bitrate / size | Quality anchor | Speed (RTF, named device) |
|---|---|---|---|---|---|
| Raw PCM (CD) | 44.1 kHz | 16-bit samples | 1,411 kbps | transparent (98 dB floor) | n/a (storage) |
| EnCodec (Défossez 2022) | 24 kHz | RVQ tokens | ~6 / 12 / 24 kbps | high at 12+ kbps | real-time on CPU (reported) |
| DAC (Kumar 2023) | 44.1 kHz | RVQ tokens | ~8 kbps | near-transparent | GPU real-time |
| WaveNet (2016) | 16 kHz | raw waveform AR | n/a | high MOS | RTF ≫ 1 (slower than real time) |
| HiFi-GAN (2020) | 22.05 kHz | mel → waveform | n/a (vocoder) | high MOS | RTF ≪ 1 (≫100× real time, V100) |

Read the table through the series' frame of **fidelity × controllability × speed × length**: raw PCM maximizes fidelity at a ruinous 1,411 kbps and no controllability; codecs trade a little fidelity for a ~100× shorter sequence (the *length* axis), which is what makes a generative model over them tractable; WaveNet proves you *can* model raw samples but pays for it in speed; HiFi-GAN shows that a fixed mel-to-waveform map is the fast last mile. Every cell in that table is governed by the math in this post — the bitrate by the sampling and quantization theorems, the representation by the STFT and mel transforms, the receptive field and speed by convolution arithmetic.

The honest measurement note that ties to the [metrics post](/blog/machine-learning/audio-generation/audio-quality-metrics): when you compare these systems you report bitrate in kbps, real-time factor (RTF = generation time ÷ audio duration) on a *named* device, and a perceptual or distributional quality number (MOS from human raters, or Fréchet Audio Distance from an embedding). All three of those rest on the representations derived here — the spectral features the metrics are computed on, the sample rate and bit depth that bound the fidelity ceiling, and the STFT settings that define the comparison. Measure RTF only after a warm-up pass (the first GPU call pays compilation and allocation costs that are not representative), hold the text or seed fixed across systems, and state which FAD embedding (VGGish vs CLAP) you used, because the same audio gives different FAD under different embeddings.

## 11. When to reach for which transform (and when not to)

A decisive guide, because half of using DSP well is *not* over-engineering it.

**Use a plain FFT** when the signal is stationary over the window you care about — a sustained tone, a steady hum you want to find and notch, a single chord. One spectrum tells you everything. Do *not* FFT a whole 3-minute song at once expecting to read it; you will get one averaged spectrum with all time information gone.

**Use the STFT / spectrogram** for anything that changes over time, which is essentially all real audio — speech, music, environmental sound. It is the default input to nearly every audio model. Choose $N$ from the time-versus-frequency trade: short for transient-heavy material (drums, consonants), long for pitch-heavy material (sustained music, choirs). When you genuinely need both, use multi-resolution rather than agonizing over one $N$.

**Use the mel-spectrogram** as your model feature unless you have a specific reason not to. It is perceptually weighted, compact (80 bins), and the established target for TTS and music vocoders. Do *not* feed a raw linear spectrogram to a generative model by default — it wastes capacity on inaudible high-frequency precision the mel scale correctly de-emphasizes.

**Use MFCCs** for classical speech/audio classification and as a lightweight feature, but *not* as a generative target — the DCT and coefficient truncation are lossy and you cannot invert them cleanly back to good audio. They are an analysis feature, not a synthesis representation.

**Reach for raw-waveform modeling** (WaveNet-style) only when you truly need sample-level fidelity and can afford the cost — and in 2026 you usually should not, because neural codecs give you most of the fidelity at a fraction of the sequence length, which is the whole premise of the modern stack. Do *not* autoregress raw 44.1 kHz samples for a long clip; that is millions of steps and an RTF far above 1.

**On bit depth and sample rate**: do not chase 24-bit/96 kHz for a generative pipeline. 16-bit puts the quantization noise floor ~98 dB down (Section 5) — inaudible — and 44.1 kHz captures the full audible band (Section 4). Higher numbers cost compute and sequence length for fidelity nobody can hear. Spend that budget on the model, not the format.

## 12. Key takeaways

- A signal is a function of time, then a sequence of samples; the **sample rate** $f_s$ and **bit depth** $b$ define how faithfully the digits capture the wave.
- The **Fourier transform** rewrites any signal as a sum of sinusoids; the **DFT** does it exactly for $N$ samples with bin spacing $\Delta f = f_s/N$, and the **FFT** computes it in $O(N\log N)$, which is why spectral audio processing is feasible.
- **Frequency is the right lens** because hearing is itself frequency analysis, convolution becomes multiplication, and audio structure is sparse in frequency (a few harmonic peaks).
- The **sampling theorem** says samples at rate $f_s$ perfectly capture everything below $f_s/2$ (Nyquist); cross that line and frequencies **alias** down irreversibly — always low-pass *before* you sample or downsample.
- **Quantization noise** follows $\text{SNR} \approx 6.02\,b + 1.76$ dB — 6 dB per bit; 16-bit gives a 98 dB floor, so artifacts you hear are spectral, never quantization.
- The **STFT** trades time for frequency resolution under the hard law $\Delta t \cdot \Delta f \geq \tfrac{1}{4\pi}$ (and $= 1$ for the discrete $N/f_s$ and $f_s/N$) — short windows catch transients, long windows resolve pitch, and you cannot have both, which is why vocoders use a multi-resolution STFT loss.
- The **convolution theorem** ($x * h \leftrightarrow X \cdot H$) makes filtering a per-bin multiply and long convolutions fast via the FFT; it is also why **1D and dilated causal convolutions** (WaveNet) are the natural learnable filterbanks of audio networks.
- The **mel-spectrogram** (filterbank + log) and **cepstrum/MFCC** (+ DCT) are cheap fixed transforms on the FFT; the mel feature is the workhorse target precisely because it is perceptual, compact, and magnitude-only.
- Inverting a magnitude-only spectrogram requires *guessing* phase (Griffin-Lim is the classical, slightly robotic answer), which is the entire reason a learned neural vocoder earns its place: it produces a perceptually correct waveform, phase and all, from the magnitude in one fast pass.

You now have the DSP the rest of this series stands on. When the [codec post](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) talks about quantizing a latent, you know what quantization noise is; when a vocoder post derives a multi-resolution STFT loss, you know which law it is fighting; when a TTS model predicts a mel-spectrogram, you know what it kept and what the vocoder must restore; and when the [capstone](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) assembles the full pipeline, every box in it speaks this language. Audio generation is, at bottom, the art of getting these transforms and their inverses right.

## Further reading

- **The Scientist and Engineer's Guide to Digital Signal Processing** — Steven W. Smith (free online). The most readable DSP book ever written; the chapters on the DFT, convolution, and sampling are the gold-standard intuition build.
- **Cooley, J. W. & Tukey, J. W. (1965)** — *An Algorithm for the Machine Calculation of Complex Fourier Series.* The FFT paper; the algorithm that made all of this practical.
- **Oppenheim, A. V. & Schafer, R. W.** — *Discrete-Time Signal Processing.* The rigorous reference for the sampling theorem, the DTFT/DFT, and LTI/convolution theory.
- **van den Oord et al. (2016)** — *WaveNet: A Generative Model for Raw Audio.* Dilated causal convolutions for raw-waveform modeling.
- **Kong, J., Kim, J. & Bae, J. (2020)** — *HiFi-GAN.* The multi-resolution STFT loss as an answer to the time-frequency uncertainty trade.
- **Défossez et al. (2022)** *EnCodec* and **Kumar et al. (2023)** *DAC* — convolutional neural codecs where the sampling and anti-aliasing math is load-bearing.
- **`torchaudio` documentation** — `Spectrogram`, `MelSpectrogram`, `MFCC`, `Resample`, `GriffinLim`, and `InverseSpectrogram`: the practical APIs for everything derived here.
- Within this series: the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the sibling [representing sound](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), the forward link [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics), and the capstone [building an audio-generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). The parallel data-math post for images: [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions).
