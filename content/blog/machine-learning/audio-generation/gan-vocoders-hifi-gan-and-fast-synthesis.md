---
title: "GAN Vocoders: HiFi-GAN and the Fast Path From Mel to Waveform"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn why GAN vocoders like HiFi-GAN, BigVGAN, and Vocos turn a phase-free mel-spectrogram back into a clean waveform hundreds of times faster than real time, and build the intuition for the multi-period and multi-scale discriminators that kill the buzzing artifacts."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "vocoder",
    "hifi-gan",
    "text-to-speech",
    "neural-audio-codec",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-1.png"
---

The first vocoder I shipped buzzed. Not loudly, not on every utterance, but on sustained vowels and held musical notes there was a faint metallic ring underneath the sound, like a cheap synthesizer trying to imitate a voice. The acoustic model in front of it was producing perfectly good mel-spectrograms — I checked them, they looked clean — and yet the audio that came out the other end had this electronic edge that no amount of retraining the acoustic model fixed, because the acoustic model was not the problem. The vocoder was. It was the last box in the pipeline, the one that takes the mel-spectrogram and turns it back into a playable waveform, and it was the box quietly deciding whether the whole system sounded human or synthetic.

That box is the subject of this post. A **vocoder** is the model that inverts a mel-spectrogram (or a codec latent, or any compact acoustic representation) back into a raw waveform you can actually play through a speaker. It is the workhorse at the very end of almost every text-to-speech system, the decoder inside every neural codec, and the difference between a TTS demo that sounds like a person and one that sounds like a 1990s answering machine. And for the last several years the vocoders that dominate production — the ones that run on your phone, inside ElevenLabs-class speech, behind every open TTS model worth using — are not the high-quality-but-glacial autoregressive models that came first. They are **GAN vocoders**: fully-convolutional generators trained adversarially that turn a mel into a waveform in a *single forward pass*, hundreds of times faster than the audio plays in real time.

![A vertical stack showing a mel-spectrogram fed through an input convolution, four transposed-convolution upsample blocks with a residual multi-receptive-field stack, and a final convolution producing a 24 kHz waveform.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-1.png)

By the end of this post you will understand the **vocoding problem** precisely — why turning a mel back into a waveform is fundamentally a *phase-reconstruction* problem, and why the classical signal-processing answer (Griffin-Lim) sounds robotic. You will know why the WaveNet and WaveGlow vocoders that came next bought quality at a brutal speed cost, and how MelGAN and then **HiFi-GAN** broke the trade by going adversarial and feed-forward. You will understand the single most important idea in the field — the **multi-period discriminator** and the **multi-scale discriminator**, and *why those two specific critics* fix audio artifacts that a single discriminator never could. You will see the **real-time-factor math** that makes "hundreds of times faster than real time" a number you can compute, run a HiFi-GAN / BigVGAN / Vocos vocoder on a mel in PyTorch and time it yourself, and walk away with a clear recommendation about when a GAN vocoder is the right tool and when it will quietly let you down. This is post C4 in the series. It assumes you know what a mel-spectrogram and phase are — if not, the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard) and the representations post [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception) build exactly that, and I will lean on both.

This post sits at a specific spot on the series spine — the **audio stack** of waveform → codec/mel latent → generative model → vocoder → waveform, balancing fidelity, controllability, speed, and length. The vocoder is the final arrow, the one that turns a latent representation back into something you can hear. It is where the *speed* axis is usually won or lost, because the generative model in front of it can be slow and clever, but the vocoder has to run every single time and cannot afford to be the bottleneck.

## 1. The vocoding problem: a mel has no phase

Let me restate exactly what a vocoder is being asked to do, because the difficulty is hidden inside an innocent-looking type signature: mel-spectrogram in, waveform out.

A waveform is a 1D list of amplitude samples — at 24 kHz, twenty-four thousand floating-point numbers per second. A mel-spectrogram is a 2D array: frequency bands (typically 80) on one axis, time frames (typically about 86 per second at a 256-sample hop) on the other, with each cell holding the *magnitude* of energy in that band at that moment. The mel is roughly a hundred times smaller than the waveform and is organized the way the ear hears, which is exactly why acoustic models predict it instead of raw samples. But that compression is not free, and the price is the whole game.

To build a mel, you take the **short-time Fourier transform** (STFT) of the waveform: you slide a window across the signal, and for each window you compute a Fourier transform, which gives you a complex number per frequency bin — a magnitude (how much of that frequency is present) and a **phase** (the alignment of that frequency's wave within the window). Then you throw the phase away. The mel keeps only magnitudes, warped onto the perceptual mel frequency scale and log-compressed. The reason this is fine for *modeling* is that the human ear is far more sensitive to the magnitude spectrum than to absolute phase — two signals with identical magnitude spectra and scrambled phase can sound similar in some cases. The reason it is a problem for *inversion* is that to reconstruct a playable waveform you need the phase back, and it is gone.

So the vocoding problem is, at its heart, a **phase-reconstruction problem**: given the magnitude mel-spectrogram, recover a full complex spectrogram (magnitude *and* a consistent phase) whose inverse STFT is a clean, natural-sounding waveform. This is underdetermined — many phase assignments are consistent with the same magnitudes — and most of them sound terrible. The art of every vocoder, classical or neural, is choosing the phase well.

There is a second subtlety on top of the missing phase: the mel is **lossy in frequency too**. Going from a linear-frequency STFT magnitude (say 1025 bins) down to 80 mel bands collapses many linear bins into each mel band, so even the magnitude information is reduced. A good vocoder does not just hallucinate phase; it also has to plausibly *up-resolve* the spectral detail that the mel binning blurred. This is why a vocoder trained on speech can sound subtly wrong on music or on an unseen speaker: it learned a prior over "what waveforms produce mels like this," and outside that prior it fills in detail that may not match the truth. Hold that thought — it is the central caveat we return to at the end.

Why is phase specifically *hard* to predict, as opposed to just missing? Two reasons. First, phase is **circular and unstable**: it lives on $[-\pi, \pi)$ with a wraparound discontinuity, so a tiny change in the underlying signal can flip a phase value from near $+\pi$ to near $-\pi$, which makes phase a brutal regression target — a small numerical error produces a large apparent error, and a naive L2 loss on raw phase is nearly useless. Second, what actually matters perceptually is not the absolute phase of each bin but the **phase coherence across bins and across frames** — whether the harmonics of a voiced sound stay aligned over time so the periodic waveform repeats cleanly. Two signals can have wildly different per-bin phase values yet sound identical because their *relative* phase structure is the same, and a signal can have plausible-looking per-bin phase yet sound buzzy because the relative structure is broken. This is the precise reason a learned, time-domain model beats a per-bin phase regressor: by generating the *waveform* directly (or, in Vocos, by predicting a phase the iSTFT will reconcile), the model optimizes for the coherent relative phase the ear cares about, not the unstable absolute phase the spectrogram would have you chase. The discriminators, judging the raw waveform, score exactly that coherence. Phase is hard because the easy way to represent it (per-bin angles) is the wrong way to optimize it.

#### Worked example: how much information is missing

Take one second of 24 kHz audio: 24,000 samples. Compute an STFT with `n_fft = 1024` and `hop_length = 256`. That gives about 94 frames, each with $1024/2 + 1 = 513$ complex bins, so $94 \times 513 \approx 48{,}000$ complex numbers — magnitude *and* phase. Now build the mel: keep only magnitude, project the 513 linear bins onto 80 mel bands. You are left with $94 \times 80 \approx 7{,}500$ real numbers. You went from roughly 96,000 real numbers (magnitude + phase) to 7,500 — you discarded about 92% of the representation, and almost all of the discarded part is phase plus fine frequency structure. The vocoder's job is to put those 7,500 numbers back into 24,000 samples that sound right. That gap is exactly why a naive inversion sounds wrong and why a learned model wins: the model has *seen* what real waveforms look like and can fill the gap with a plausible, learned prior rather than an arbitrary guess.

## 2. The classical baseline: Griffin-Lim and why it buzzes

Before neural vocoders, the standard way to invert a magnitude spectrogram was the **Griffin-Lim algorithm** (Griffin and Lim, 1984). It is worth understanding because it makes the phase problem concrete and because it is still the right tool in a few narrow cases.

Griffin-Lim is an iterative fixed-point procedure. You have the target magnitude spectrogram $|S|$ and you want a waveform. Start by assigning a random (or zero) phase $\phi_0$, giving a complex spectrogram $S_0 = |S| e^{i\phi_0}$. Then repeat:

1. Inverse STFT $S_k$ to get a candidate waveform $x_k$.
2. Forward STFT $x_k$ to get a new complex spectrogram $\hat{S}_k$.
3. Keep the *phase* of $\hat{S}_k$ but force the *magnitude* back to the target $|S|$: $S_{k+1} = |S| e^{i \angle \hat{S}_k}$.

The trick in step 3 is that the inverse-then-forward STFT round trip is not the identity for overlapping windows — adjacent frames share samples, so they impose **consistency constraints** on each other. Each iteration nudges the phase toward an assignment that is *self-consistent* across overlapping windows. The algorithm minimizes the squared error between the candidate's magnitude and the target magnitude, and it provably does not increase that error each step, so it converges to a consistent phase estimate.

It converges — but to *a* consistent phase, not necessarily the *natural* one. And that is exactly why Griffin-Lim sounds the way it does. Here is what it costs you in practice.

```python
import torch
import torchaudio
import torchaudio.transforms as T

# Load a clip; resample to 24 kHz for a typical TTS pipeline.
wav, sr = torchaudio.load("speech.wav")
if sr != 24000:
    wav = T.Resample(sr, 24000)(wav)
    sr = 24000

n_fft, hop, n_mels = 1024, 256, 80

# Forward: waveform -> mel (magnitude only; phase is discarded here).
mel_spec = T.MelSpectrogram(
    sample_rate=sr, n_fft=n_fft, hop_length=hop,
    n_mels=n_mels, power=1.0,  # magnitude, not power
)(wav)

# To invert, first go mel -> linear-frequency magnitude (lossy approximation),
# then Griffin-Lim guesses the phase iteratively.
inv_mel = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sr)
lin_mag = inv_mel(mel_spec)

griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=hop, power=1.0, n_iter=60)
wav_gl = griffin_lim(lin_mag)

torchaudio.save("speech_griffinlim.wav", wav_gl, sr)
```

Run that and listen. On speech you get something intelligible but unmistakably **robotic and buzzy**, with a hollow, phasey quality on vowels — a mean opinion score (MOS) somewhere around 2.5–3.0 out of 5 on a typical setup, well below the ~4.4 of the original. On music it is worse: transients smear, cymbals turn to mush, and sustained tones get a metallic ring.

Why does it buzz? Three reasons, all instructive. First, **Griffin-Lim has no learned prior** — it knows nothing about what real speech or music waveforms look like, only that the magnitudes should match and the frames should be consistent. Among the many waveforms with the right magnitudes, it picks a consistent one, not a *natural* one, and the unnatural phase reads to the ear as artifact. Second, the **mel-to-linear step is itself lossy and approximate** — `InverseMelScale` solves an underdetermined least-squares problem to guess 513 linear bins from 80 mel bands, smearing fine harmonic structure before Griffin-Lim ever runs. Third, it is **slow**: 60 iterations of full STFT/iSTFT over the whole clip, and quality keeps improving (slowly) with more iterations, so you trade time for marginal quality with no good stopping point.

The lesson generalizes: signal-processing inversion is principled but priorless, and a phase-reconstruction problem with no prior is exactly the kind of underdetermined problem where a learned model should win. The only question was how to learn it without paying an unacceptable speed cost — and the first neural answers got the quality but not the speed.

## 3. The slow-but-good era: WaveNet and WaveGlow vocoders

The first neural vocoders that decisively beat Griffin-Lim were **autoregressive** and **flow-based**, and both bought their quality with a heavy speed tax.

**WaveNet** (van den Oord et al., 2016), used as a vocoder, models the waveform one sample at a time: $p(x_t \mid x_{<t}, \text{mel})$, a categorical distribution over 256 mu-law-quantized amplitude levels, conditioned on the mel-spectrogram as a local conditioning signal. Architecturally it is a stack of dilated causal convolutions — dilation doubling each layer (1, 2, 4, 8, …, 512) so the receptive field grows exponentially without an explosion of layers, letting a few dozen layers see thousands of past samples. The quality is superb; WaveNet as a vocoder hits MOS around 4.4, essentially indistinguishable from natural speech in the original studies. The catch is in the type signature: **autoregressive means one forward pass per output sample.** To synthesize one second of 24 kHz audio you run the network 24,000 times, each step depending on the last, with no parallelism across time. A vanilla WaveNet vocoder runs *hundreds of times slower than real time* on a GPU — a real-time factor (RTF) far greater than 1, meaning it takes longer to generate the audio than to play it. That is a non-starter for any interactive system.

The field tried two escape routes. The first was **distillation into a parallel student**: Parallel WaveNet and ClariNet trained a feed-forward inverse-autoregressive-flow student to match a WaveNet teacher, recovering parallelism but at the cost of a fragile, multi-loss training recipe. The second was **normalizing flows** done natively: **WaveGlow** (Prenger et al., 2018) and **FloWaveNet** built an invertible flow that maps a simple Gaussian to the waveform conditioned on the mel, so sampling is a single parallel forward pass through the flow. WaveGlow's quality approaches WaveNet's and it is genuinely parallel, so its RTF on a GPU is well below 1 — but the price is **size and memory**: WaveGlow has on the order of 90 million parameters and a deep stack of invertible coupling layers with large 1×1 convolutions, because the invertibility constraint forces the architecture to be wider and heavier than an unconstrained network would need to be. You got speed, but you paid for it in parameters and VRAM, and the flow's invertibility constraint capped how efficient the model could be.

It is worth naming *why* the likelihood requirement was so costly, because that is the constraint the GAN drops. WaveNet's autoregression and WaveGlow's flow both exist to make the model a valid probability density you can train by maximum likelihood — WaveNet factorizes $p(x) = \prod_t p(x_t \mid x_{<t})$, which forces sequential generation, and a normalizing flow requires every layer to be invertible with a tractable Jacobian, which forces a wide, constrained architecture. Both pay a structural tax for the privilege of writing down an exact likelihood. But for a *conditional* problem with a strong conditioning signal — give me the waveform for *this* mel — you do not actually need a calibrated density over all waveforms; you need one good waveform per mel. The likelihood machinery is overkill, and it is exactly the overkill that makes these models slow or heavy.

So by 2019 the landscape was: Griffin-Lim (fast, priorless, buzzy), WaveNet (gorgeous, unusably slow), distilled/parallel WaveNet (fast but fragile), WaveGlow (fast and good but heavy). The opening was obvious in hindsight. What if you dropped the requirement that the model represent an exact likelihood — no autoregression, no invertibility — and instead trained a plain feed-forward convolutional network to map mel to waveform, judged not by likelihood but by an **adversarial critic**? That is the GAN vocoder, and it is what changed everything.

![A dataflow graph showing a mel-spectrogram feeding a feed-forward generator that produces a fake waveform, which is then judged by both a multi-period discriminator and a multi-scale discriminator whose outputs combine into the adversarial, feature-matching, and mel-reconstruction loss.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-2.png)

## 4. The GAN vocoder breakthrough: MelGAN then HiFi-GAN

A **generative adversarial network** trains two networks against each other: a **generator** $G$ that produces samples, and a **discriminator** $D$ that tries to tell real from generated. The generator's gradient comes not from a reconstruction target but from "fool the discriminator," which means the generator is pushed toward outputs that are *indistinguishable from real data* under whatever features the discriminator learned to inspect. GANs famously struggled and ultimately lost the image-generation crown to diffusion — the full story is in the image series' [generative adversarial networks and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost) — but for *vocoding* the GAN's one decisive advantage, **a single fast feed-forward forward pass at inference**, turned out to be exactly what the problem needed, and the conditioning on a mel-spectrogram sidesteps the mode-collapse and instability that plague unconditional image GANs. The vocoder is a conditional generation problem with a strong conditioning signal, which is the friendliest possible setting for a GAN.

**MelGAN** (Kumar et al., 2019) was the first to make it work end to end. Its generator is a stack of transposed convolutions that upsample the mel to the waveform, with residual blocks of dilated convolutions in between; the whole thing is non-autoregressive, so synthesis is one parallel pass. The key MelGAN insight was the **discriminator design**: a single discriminator looking at the raw waveform at one resolution is not enough, because audio has structure at many time scales at once. MelGAN used a **multi-scale discriminator** — three copies of the same discriminator architecture, each looking at the waveform at a different downsampling (raw, 2×-downsampled, 4×-downsampled) — so the critic judges both fine, high-frequency detail and coarse, low-frequency envelope. It also introduced a **feature-matching loss** (more on that below). MelGAN was fast and reasonably good — MOS around 3.6–4.0 depending on the setup — but it left a quality gap to WaveNet, especially a residual artifact on the periodic structure of voiced speech.

**HiFi-GAN** (Kong, Kim, and Bae, 2020) closed that gap and became the default. It kept the feed-forward generator idea, refined it with a **multi-receptive-field fusion** module (which I will dissect in the next section), and — most importantly — added a second, complementary discriminator that was the missing piece: the **multi-period discriminator**. With its multi-period plus multi-scale discriminator pair, a feature-matching loss, and a mel-reconstruction loss, HiFi-GAN reached MOS around 4.3 — within the confidence interval of ground-truth audio in the original listening tests — at an RTF on the order of 0.004 on a V100 GPU, meaning it generates audio *roughly 250× faster than real time*. That is the headline: WaveNet-class quality, hundreds of times faster than real time, in a 14-million-parameter model that fits comfortably on a phone. The rest of this post is the *why*.

## 5. Inside the generator: upsampling a mel into a waveform

Let me open up the HiFi-GAN generator, because its structure is the template every later GAN vocoder follows.

The generator is **fully convolutional** and its job is pure upsampling: the input is an 80-channel mel at the frame rate (about 86 frames per second), and the output is a 1-channel waveform at the sample rate (24,000 samples per second). The ratio is the **hop length** — at `hop = 256`, you need to expand the time axis by 256×. HiFi-GAN does this in a few **upsample blocks**, each a transposed convolution that multiplies the time resolution, and the *product* of the upsample factors equals the hop length. A common configuration for 22.05 kHz is upsample rates `[8, 8, 2, 2]`, whose product is $8 \times 8 \times 2 \times 2 = 256$. Channels shrink as time grows (512 → 256 → 128 → 64 → 32), so the tensor stays a manageable size as it expands.

![A vertical stack of the HiFi-GAN generator showing the 80-channel mel input, four upsample-plus-MRF stages that progressively double the time axis while halving channels, and a final convolution with tanh producing a single-channel 24 kHz waveform.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-5.png)

After each transposed convolution comes the part that makes HiFi-GAN sound good: the **multi-receptive-field fusion (MRF)** module. A single dilated convolution sees one time scale; speech and music have structure at many scales at once (the fine wiggle of a single pitch period, the slower swell of a vowel, the envelope of a syllable). The MRF runs **several residual blocks in parallel**, each with a different kernel size and dilation pattern, and **sums** their outputs. So every sample of the generator's output is computed from a fusion of several receptive fields — some narrow and detail-focused, some wide and context-aware. This is the architectural answer to "audio has structure at many time scales," and it is why HiFi-GAN's generator beats MelGAN's plainer residual stack.

Here is the generator sketch in PyTorch — simplified but structurally faithful, so you can see exactly how a mel becomes a waveform.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """One MRF branch: dilated convs with residual connections."""
    def __init__(self, ch, kernel=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel, dilation=d,
                      padding=(kernel - 1) * d // 2)
            for d in dilations
        ])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, 0.1))  # residual
        return x

class Generator(nn.Module):
    def __init__(self, n_mels=80, up_rates=(8, 8, 2, 2),
                 up_kernels=(16, 16, 4, 4), ch=512):
        super().__init__()
        self.conv_pre = nn.Conv1d(n_mels, ch, 7, padding=3)
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        for i, (r, k) in enumerate(zip(up_rates, up_kernels)):
            in_ch, out_ch = ch // (2 ** i), ch // (2 ** (i + 1))
            # transposed conv upsamples the time axis by r
            self.ups.append(nn.ConvTranspose1d(
                in_ch, out_ch, k, stride=r, padding=(k - r) // 2))
            # MRF = several ResBlocks of different kernels, summed
            self.mrfs.append(nn.ModuleList([
                ResBlock(out_ch, ker, dil)
                for ker, dil in [(3, (1, 3, 5)), (7, (1, 3, 5)),
                                 (11, (1, 3, 5))]
            ]))
        self.conv_post = nn.Conv1d(ch // (2 ** len(up_rates)), 1, 7, padding=3)

    def forward(self, mel):
        x = self.conv_pre(mel)                       # (B, 512, T)
        for up, mrf in zip(self.ups, self.mrfs):
            x = up(F.leaky_relu(x, 0.1))             # time x r, channels / 2
            x = sum(block(x) for block in mrf) / len(mrf)  # MRF fusion
        x = self.conv_post(F.leaky_relu(x, 0.1))
        return torch.tanh(x)                         # waveform in [-1, 1]
```

Two details earn their place. The final `tanh` bounds the output to $[-1, 1]$, the natural range of a normalized waveform. And the whole forward pass is **a single sweep of convolutions with no recurrence and no autoregression** — every output sample is computed in parallel, which is the entire reason this runs faster than real time. There is no loop over time at inference. You hand it a mel, you get a waveform, once.

The MRF dilations are worth a second look because they are doing real work. A stack of dilated convolutions with dilation rates $1, 3, 5$ and kernel size $k$ has a **receptive field** that grows as roughly $1 + \sum_i (k-1) d_i$ samples. For $k = 3$ and dilations $1, 3, 5$ that is $1 + 2(1 + 3 + 5) = 19$ samples per branch at the post-upsampling resolution; chain a few of these across the upsample stages and a single output sample's receptive field reaches hundreds of input samples — several pitch periods of context — without the parameter cost of an equally wide dense kernel. This is the same dilation trick WaveNet used to grow its receptive field exponentially, but here it runs *non-causally and in parallel* rather than one sample at a time, which is the architectural difference that turns WaveNet's per-sample cost into HiFi-GAN's one-pass cost. The MRF then fuses three branches of *different* kernel sizes (3, 7, 11), so the model simultaneously sees narrow detail (kernel 3) and wide context (kernel 11) and learns to weight them. The summation is what "multi-receptive-field fusion" means literally: add the outputs of receptive fields of different widths so no single scale dominates.

One more structural choice matters for quality: **weight normalization** on every convolution. HiFi-GAN applies `weight_norm` throughout training to stabilize the adversarial optimization, then *folds it away* (`remove_weight_norm`) before inference so it costs nothing at deployment. This is why the BigVGAN snippet later calls `remove_weight_norm()` — it is a free inference speedup that bakes the normalization into the raw weights. Forgetting it leaves a small but real per-layer overhead on every forward pass, which adds up when the vocoder runs millions of times in production.

#### Worked example: the upsample arithmetic

Suppose your acoustic model emits an 80-band mel for a 2-second utterance at `hop = 256`, `sr = 24000`. The mel has $T = \lceil 24000 \times 2 / 256 \rceil \approx 188$ frames, shape `(80, 188)`. With upsample rates `[8, 8, 2, 2]`, the time axis after each block is $188 \to 1504 \to 12032 \to 24064 \to 48128$ — and $48128 \approx 24000 \times 2$, the right number of samples for 2 seconds. The channel dimension runs $512 \to 256 \to 128 \to 64 \to 32 \to 1$. Notice the tensor never explodes: time grows but channels shrink in lockstep, so the largest intermediate tensor is on the order of a few million floats, trivial for a GPU. This balance is deliberate — it is why a 14M-parameter generator can produce a full-rate waveform in one cheap pass.

## 6. The key idea: why the multi-period and multi-scale discriminators

This is the heart of the post. HiFi-GAN's quality does not come mainly from the generator — plenty of architectures upsample a mel. It comes from the **discriminators**, and specifically from a pair that attack two different failure modes of generated audio. Understanding *why these two specific critics* is the single most valuable thing to take away.

Start from the failure they fix. Early GAN vocoders produced a faint **buzzing and metallic artifact**, worst on voiced, periodic sounds — sustained vowels, held musical notes. The root cause is **periodicity**. Voiced speech and pitched music are *quasi-periodic*: the vocal folds (or a vibrating string) repeat a waveform shape at the fundamental frequency, hundreds of times per second. A 200 Hz voiced sound repeats its basic waveform every 120 samples at 24 kHz. If the generator gets the period *slightly* wrong — drifts the phase across cycles, or smears the sharp glottal pulse that starts each period — the ear hears it instantly as roughness or buzz, because human hearing is exquisitely tuned to pitch periodicity. A discriminator that looks at the waveform as one long 1D sequence (a plain 1D convnet) has a hard time *seeing* this periodic structure: the repeating pattern is spread thinly across thousands of samples, and a 1D conv's receptive field mixes it with everything else.

The **multi-period discriminator (MPD)** is a beautifully simple fix. Take the 1D waveform and **reshape it into 2D by folding it at a period** $p$: a waveform of length $L$ becomes a 2D tensor of shape roughly $(L/p, p)$, where each *row* is one chunk of $p$ consecutive samples. Now apply a 2D convolution with width-1 kernels along the period axis. The effect is that the discriminator looks at every $p$-th sample together — it isolates exactly the structure that repeats with period $p$. HiFi-GAN uses **several MPD sub-discriminators with prime periods** $p \in \{2, 3, 5, 7, 11\}$. Why primes? Because primes are mutually non-overlapping in the periodicities they sample — using 2 and 4 would have 4's structure partly captured by 2, wasting a discriminator, whereas coprime periods tile the space of periodicities efficiently. With prime periods, the MPD ensemble can catch periodic errors at many fundamental frequencies, and because it judges each period in isolation it punishes the generator hard for any phase drift or pulse-shape error in the periodic structure. **That is what kills the buzz.**

```python
class PeriodDiscriminator(nn.Module):
    """One MPD sub-disc: fold the 1D waveform at period p, then 2D-conv."""
    def __init__(self, period, ch=32):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            # kernel (5, 1): wide in time-within-period, width 1 across periods
            nn.Conv2d(1, ch, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(ch, ch * 4, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(ch * 4, ch * 16, (5, 1), (3, 1), padding=(2, 0)),
        ])
        self.post = nn.Conv2d(ch * 16, 1, (3, 1), padding=(1, 0))

    def forward(self, x):                  # x: (B, 1, L)
        b, c, t = x.shape
        if t % self.period:                # pad so L divides by period
            x = F.pad(x, (0, self.period - t % self.period), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)  # fold to 2D
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)                # keep activations for feature matching
        x = self.post(x)
        feats.append(x)
        return x, feats                    # logits, intermediate features
```

Two more details make the MPD precise. First, the convolution kernels are *tall and narrow* — shape `(5, 1)` — so they look across **five consecutive samples within a period** but never mix across the period axis at width greater than one. That keeps the per-period structure isolated; the discriminator literally cannot blur one period's evidence into the next, which is what lets it localize a phase drift to a specific period. Second, the periods are deliberately **coprime**. Any waveform periodicity that is a multiple of a chosen period $p$ is partly visible to that sub-disc, so if you picked 2 and 4 the period-4 structure would already leak into the period-2 view and the second discriminator would be largely redundant. Primes $\{2, 3, 5, 7, 11\}$ share no common factors, so each samples a genuinely different slice of the periodicity space, and their least common multiple (2310) is large enough that the ensemble jointly resolves periodicities far longer than any single member. This is a small, almost number-theoretic design choice that buys a lot of coverage for five cheap sub-discriminators.

The **multi-scale discriminator (MSD)**, inherited from MelGAN, attacks the complementary failure: **structure across resolutions**. The MSD is three copies of a 1D convolutional discriminator, each operating on the waveform at a different scale — the raw waveform, a 2×-average-pooled version, and a 4×-average-pooled version. The raw-scale disc judges fine, high-frequency detail; the pooled-scale discs judge coarser, lower-frequency envelope and longer-range consistency. Where the MPD looks at *periodicity* (vertical, across cycles), the MSD looks at *broadband texture at multiple zoom levels* (horizontal, across time scales). Average-pooling before each scale acts as an anti-aliasing low-pass, so the pooled discriminators genuinely see the low-frequency content without high-frequency leakage. The MSD sub-discs also use **grouped convolutions** with large kernels (up to 41) and strided downsampling, so a single MSD pass spans a wide receptive field over the raw waveform — wide enough to judge whether a whole syllable's envelope is coherent, not just whether neighboring samples are smooth.

There is a deeper reason both discriminators operate directly on the **raw waveform** rather than on a spectrogram. If you judged generated audio only by comparing spectrograms, you would be blind to phase by construction — the spectrogram magnitude is phase-free, which is the very thing we are trying to reconstruct. A discriminator that ingests the raw time-domain waveform *can* see phase, because phase is exactly the alignment of samples in time. The MPD's period-folding is the sharpest possible time-domain phase probe: it lines up samples that should be in lockstep across cycles and checks whether they are. So the choice to discriminate in the time domain is not incidental — it is the only place a critic can catch the phase errors that a mel threw away and Griffin-Lim guessed wrong. The discriminators see what the mel cannot.

The two together cover the space of audio artifacts. **MPD catches what is wrong with the pitch and periodicity; MSD catches what is wrong with the broadband texture and envelope.** A generator that fools both has gotten the periodic structure right (no buzz) *and* the multi-resolution texture right (no smearing, no muffling). This is the precise, mechanistic reason HiFi-GAN sounds clean where MelGAN (MSD only) had a residual artifact: MelGAN had no critic specialized for periodicity, so the generator was never strongly punished for the exact error the ear is most sensitive to.

#### Worked example: catching a 5-sample-period error

Imagine the generator produces a 240 Hz voiced sound at 24 kHz, which should repeat every 100 samples, but the generator's output drifts so the effective period wobbles by a couple of samples across cycles — inaudible in the magnitude spectrum, but audible as roughness. A plain 1D discriminator sees the waveform as a 24,000-long sequence and the wobble is buried. The MPD sub-disc with period $p = 5$ folds the waveform so every 5th sample lines up in a column; across the 20 columns spanning one 100-sample period, the wobble shows up as a misalignment the 2D conv can localize, and the disc returns a low "real" score. The gradient flows back to the generator as "your period-5 structure is off," and over training the generator tightens its periodicity. No single-resolution 1D discriminator gives that targeted a signal. This is the whole argument in miniature: *the right discriminator geometry makes an inaudible-in-magnitude, audible-to-the-ear error visible to a gradient.*

## 7. The three-part loss that actually kills the artifact

The discriminators define *what* to judge; the loss defines *how* the generator learns from that judgment. HiFi-GAN's generator loss is a weighted **sum of three terms**, and the combination matters as much as any single piece — drop any one and quality degrades in a specific way.

![A dataflow graph showing the generated waveform feeding three loss terms in parallel — an adversarial least-squares term, a feature-matching L1 term over discriminator layers, and a mel-reconstruction L1 term weighted by 45 — that sum into the total generator loss.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-6.png)

**The adversarial loss.** HiFi-GAN uses the **least-squares GAN (LSGAN)** objective rather than the original cross-entropy, because least-squares gives smoother, more stable gradients and avoids the vanishing-gradient pathology of the saturating GAN loss. For each discriminator $D_k$ (every MPD and MSD sub-disc), the discriminator minimizes
$$\mathcal{L}_D = \mathbb{E}\big[(D_k(x) - 1)^2\big] + \mathbb{E}\big[D_k(G(s))^2\big],$$
pushing real audio $x$ toward 1 and generated audio $G(s)$ (from mel $s$) toward 0. The generator's adversarial term pushes the other way:
$$\mathcal{L}_{\text{adv}} = \mathbb{E}\big[(D_k(G(s)) - 1)^2\big],$$
that is, "make every discriminator score my output as real." This is the term that sharpens realism and gives the audio its naturalness, but on its own it is unstable and can chase realism into the wrong content.

**The feature-matching loss.** This is the stabilizer, and it is doing more work than people credit. For a real and a generated sample, run both through each discriminator and compare the *intermediate activations*, not just the final logit:
$$\mathcal{L}_{\text{fm}} = \mathbb{E}\Big[\sum_{l} \frac{1}{N_l}\big\|D_k^{(l)}(x) - D_k^{(l)}(G(s))\big\|_1\Big],$$
summed over discriminator layers $l$. In words: the generated audio should produce *the same internal features* in the discriminator as real audio does, layer by layer. This is a perceptual loss in the discriminator's own learned feature space — and because the discriminator's features are tuned to exactly the periodicity and multi-scale structure that matters, matching them is a strong, stable signal that does not collapse the way a bare adversarial loss can. Feature matching is the quiet reason GAN vocoders train reliably where image GANs were finicky.

**The mel-reconstruction loss.** Finally, an explicit content anchor: take the generated waveform, compute *its* mel-spectrogram, and L1-match it to the input mel:
$$\mathcal{L}_{\text{mel}} = \mathbb{E}\big[\|\,\phi(x) - \phi(G(s))\,\|_1\big],$$
where $\phi$ is the mel transform. This term, weighted heavily (the paper uses $\lambda_{\text{mel}} = 45$), guarantees the output actually *says the right thing* — the adversarial and feature-matching terms make it sound real, but the mel-recon term makes it sound real *and correct*, locking the spectral envelope to the conditioning. Without it, a GAN vocoder can produce gorgeous, natural-sounding audio that drifts from the intended content.

The full generator objective is
$$\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \lambda_{\text{fm}}\,\mathcal{L}_{\text{fm}} + \lambda_{\text{mel}}\,\mathcal{L}_{\text{mel}},$$
with $\lambda_{\text{fm}} = 2$ and $\lambda_{\text{mel}} = 45$ in the original. The takeaway to keep: **adversarial makes it real, feature-matching makes it stable, mel-recon makes it correct** — and the buzz dies because the MPD-driven adversarial gradient specifically punishes periodic error while the other two terms keep training from wandering. No one term does it alone.

The *training dynamics* are worth a paragraph because they explain why the weights are what they are. Early in training, the discriminators are weak and the adversarial signal is nearly useless — if the generator relied on it alone, it would produce noise. The heavily-weighted mel-reconstruction term ($\lambda_{\text{mel}} = 45$ dwarfs the others) acts as a strong supervised anchor that gets the generator producing roughly-correct audio *immediately*, so the adversarial game starts from a sane place. As the discriminators sharpen, the adversarial and feature-matching gradients take over the fine perceptual polishing — the parts the L1 mel loss cannot capture, like the exact phase and the high-frequency texture, because an L1 on a mel is itself phase-blind and over-smoothed. So the loss schedule is implicitly curriculum: mel-recon dominates the early "get it roughly right" phase, adversarial plus feature-matching dominate the late "make it indistinguishable" phase. That ordering is why GAN vocoders train in a day or two and do not collapse: there is always a strong, stable gradient (mel-recon) keeping the generator on the manifold of plausible speech while the adversarial terms do the risky, high-variance work of chasing realism.

A subtle failure mode lurks here. If you set $\lambda_{\text{mel}}$ too high, the generator over-relies on the L1 mel loss and produces *over-smoothed* audio — because L1 in mel space, like any reconstruction loss, regresses toward the conditional mean, which for the phase-free, multi-modal waveform distribution is a blurry average that sounds muffled. If you set it too low, training destabilizes and the buzz comes back because the adversarial term is chasing realism without an anchor. The default $45$ is a balance found empirically, and it is one of the first knobs to tune if your vocoder sounds either muffled (lower it) or rough (raise it). This is the same mean-seeking pathology that makes a pure L2/L1 image autoencoder produce blurry images and why GANs were brought in at all — the adversarial term exists precisely to escape the blurry conditional mean.

#### Worked example: diagnosing a muffled versus a buzzy vocoder

You train a HiFi-GAN and the output is intelligible but *muffled* — like speech through a wall, with the high frequencies rolled off. The instinct is "the model is too small," but the more likely culprit is the loss balance: with $\lambda_{\text{mel}}$ dominating, the generator regressed to the over-smoothed conditional-mean waveform and the adversarial term never got enough say to restore the crisp high-frequency texture. The fix is to *lower* $\lambda_{\text{mel}}$ (say from 45 to 20) or train longer so the discriminators catch up. Conversely, if the output is crisp but *buzzy and rough* on voiced sounds, the adversarial term is over-driving without a strong enough anchor — *raise* $\lambda_{\text{mel}}$ or check that the MPD is actually learning (a collapsed discriminator that outputs a constant gives no useful gradient). Knowing which artifact maps to which knob — muffled → too much mel-recon, buzzy → too little — turns vocoder debugging from guesswork into a two-line decision, and it is exactly the kind of thing you only learn by chasing the buzz a few times yourself.

![A before-and-after comparison contrasting a Griffin-Lim reconstruction that is robotic and buzzing with a low MOS against a HiFi-GAN reconstruction that is clean and natural with a high MOS at a tiny real-time factor.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-3.png)

## 8. The real-time-factor math: what "faster than real time" means

"Hundreds of times faster than real time" is not a vibe; it is a number you can compute, and you should always report it. The metric is the **real-time factor (RTF)**:
$$\text{RTF} = \frac{\text{generation time (seconds)}}{\text{audio duration (seconds)}}.$$
An RTF of 1 means the model takes exactly as long to generate as the audio lasts — barely usable for streaming, hopeless for batch. An RTF below 1 means faster than real time; an RTF of 0.004 means it generates one second of audio in four milliseconds, i.e. **250× faster than real time**. Its reciprocal, $1/\text{RTF}$, is the "× faster than real time" headline. For a GAN vocoder, RTF ≪ 1; for autoregressive WaveNet, RTF ≫ 1 (it takes hundreds of times *longer* than the audio plays).

Why is the GAN vocoder's RTF so low? Because the work is a fixed, parallel stack of convolutions whose cost scales linearly with output length and runs entirely in parallel on a GPU — there is no sequential dependency across the 24,000 output samples. WaveNet's cost also scales with length, but each of the 24,000 samples is a *separate sequential forward pass* that cannot start until the previous one finishes, so the wall-clock time is length × per-step latency with zero time parallelism. That sequential dependency, not the FLOP count, is what makes autoregression slow. A GAN vocoder may do comparable total FLOPs but does them all at once.

Here is how to measure RTF honestly, with a warm-up to exclude one-time CUDA/compilation costs and a synchronize so you time the GPU, not just the kernel launch.

```python
import time, torch

def measure_rtf(vocoder, mel, sr=24000, hop=256, n_warmup=3, n_runs=20):
    device = next(vocoder.parameters()).device
    mel = mel.to(device)
    audio_seconds = mel.shape[-1] * hop / sr   # frames * hop / sample_rate

    with torch.inference_mode():
        for _ in range(n_warmup):              # warm up: skip JIT/cuDNN autotune
            _ = vocoder(mel)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            wav = vocoder(mel)
        if device.type == "cuda":
            torch.cuda.synchronize()           # wait for GPU before stopping clock
        gen_seconds = (time.perf_counter() - t0) / n_runs

    rtf = gen_seconds / audio_seconds
    print(f"audio={audio_seconds:.2f}s  gen={gen_seconds*1000:.1f}ms  "
          f"RTF={rtf:.4f}  ({1/rtf:.0f}x faster than real time)")
    return rtf
```

Three honesty rules for RTF, learned the hard way. **Warm up** — the first call pays JIT, cuDNN autotuning, and memory-allocation costs that have nothing to do with steady-state speed, so discard the first few runs. **Synchronize** — CUDA kernels are asynchronous, so without `torch.cuda.synchronize()` you are timing kernel *launch*, not *execution*, and will report an absurdly low RTF. **Name the device and batch size** — RTF on an A100 with batch 1 is a different number from RTF on a CPU or at batch 32, and "RTF 0.004" is meaningless without "on a V100, batch 1, fp32." Report all three or the number is noise.

#### Worked example: where the vocoder sits in a TTS budget

Suppose you run a non-autoregressive TTS acoustic model that emits a mel for a 3-second utterance in 40 ms on an A100, and you pair it with HiFi-GAN at RTF 0.004. The vocoder takes $0.004 \times 3 = 12$ ms. Total: about 52 ms to synthesize 3 seconds of speech — an end-to-end RTF of $0.052 / 3 \approx 0.017$, roughly 58× faster than real time, comfortably inside a real-time conversational budget. Now swap in a WaveNet vocoder at RTF 200: the vocoder alone takes $200 \times 3 = 600$ seconds — ten minutes to voice a 3-second sentence. *The vocoder choice, not the acoustic model, is what decides whether the system can run interactively.* This is exactly why GAN vocoders won production: the rest of the stack can be as clever as it likes, but the vocoder runs every time and must be cheap.

## 9. Running a real vocoder: HiFi-GAN, BigVGAN, Vocos

Enough theory — let me run three real, pretrained vocoders on a mel and time them, because the whole point is that this is a few lines of code today. I will keep the running thread of a single utterance's mel-spectrogram and decode it three ways.

**HiFi-GAN via the bundled torchaudio / a TTS pipeline.** The simplest path is to let a TTS library hand you a matched mel + HiFi-GAN pair (the vocoder must match the mel parameters it was trained on — sample rate, `n_fft`, `hop`, `n_mels`, mel `fmin`/`fmax` — or you get garbage). Here I use the 🤗 `transformers` SpeechT5 stack, whose vocoder *is* a HiFi-GAN, to show a clean mel → waveform handoff.

```python
import torch, soundfile as sf
from transformers import SpeechT5HifiGan

# SpeechT5's vocoder is a HiFi-GAN. Feed it an 80-band log-mel at the
# parameters it was trained on (16 kHz, n_fft=1024, hop=256, n_mels=80).
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()

# `mel` here comes from your acoustic model; shape (frames, 80) for this API.
# For a standalone test, you can mel-transform a real clip and round-trip it.
with torch.inference_mode():
    waveform = vocoder(mel)            # (num_samples,) at 16 kHz, one pass
sf.write("hifigan_out.wav", waveform.numpy(), 16000)
```

**BigVGAN — the large universal vocoder.** HiFi-GAN, trained on one speaker or one dataset, generalizes imperfectly to unseen speakers and to music. **BigVGAN** (Lee et al., NVIDIA, 2022) is the answer: a much larger HiFi-GAN-style generator (up to ~112M parameters) trained on a large, diverse multi-speaker, multi-lingual, music-and-environmental corpus to be a **universal vocoder** — one model that vocodes any mel, seen speaker or not. Two architectural changes make the scale pay off. First, the **snake activation** $x + \frac{1}{\alpha}\sin^2(\alpha x)$, which has a built-in *periodic* inductive bias — unlike a ReLU or leaky-ReLU, a periodic activation naturally produces the repeating structure that pitched audio needs, so the network does not have to learn periodicity from scratch. Second, **anti-aliased upsampling**: naive transposed convolutions introduce aliasing artifacts (spurious high-frequency content) when they upsample, so BigVGAN borrows the low-pass-filtered up/down-sampling from the alias-free GAN literature to keep the spectrum clean as it expands. The result is a vocoder that holds up on out-of-distribution audio where vanilla HiFi-GAN gets brittle.

```python
import torch
# pip install bigvgan  (NVIDIA's official package), or use the HF hub repo.
import bigvgan

model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False).eval()
model.remove_weight_norm()             # fold weight-norm for faster inference

with torch.inference_mode():
    # mel: (1, 100, frames) log-mel at the model's exact params (24 kHz here)
    wav = model(mel)                    # (1, 1, num_samples), one pass
```

**Vocos — predict the spectrum, not the samples.** **Vocos** (Siuzdak, 2023) takes a different and faster route. Instead of upsampling the mel through a deep stack of transposed convolutions in the *time* domain, Vocos keeps a **constant time resolution** — its backbone is a ConvNeXt network running at the spectrogram frame rate — and predicts the **STFT coefficients** (magnitude and phase) directly. Then a single **inverse STFT** does all the upsampling to the waveform in one cheap, exact, non-learned step. Because the heavy network never operates at the full sample rate (it works at the ~86 Hz frame rate, not 24 kHz), Vocos is even faster than HiFi-GAN while matching its quality — and it sidesteps the aliasing pitfalls of learned transposed-conv upsampling because the iSTFT is a perfect reconstruction operator. It is, in effect, a neural Griffin-Lim that learned the phase in one shot.

```python
import torch, torchaudio
from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

wav, sr = torchaudio.load("speech.wav")
if sr != 24000:
    wav = torchaudio.transforms.Resample(sr, 24000)(wav)

with torch.inference_mode():
    mel = vocos.feature_extractor(wav)   # the mel Vocos expects
    recon = vocos.decode(mel)            # mel -> STFT coeffs -> iSTFT, one pass
torchaudio.save("vocos_out.wav", recon, 24000)
```

![A before-and-after comparison contrasting time-domain HiFi-GAN that upsamples with stacked transposed convolutions against Vocos that predicts STFT magnitude and phase at the frame rate and inverts with a single iSTFT.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-8.png)

The Vocos idea — **predict a representation the iSTFT can invert exactly, instead of regressing raw samples** — is quietly important. It decouples the *learning* problem (estimate phase and magnitude at low resolution) from the *upsampling* problem (a fixed, exact iSTFT), and that decoupling is why it is both fast and clean. It is the same move that makes latent diffusion efficient — do the hard learning in a compact space and let a fixed decoder handle the expansion.

## 10. The vocoder zoo, measured

Now the comparison the whole post has been building toward. Here is the practical frontier, with the caveat that exact numbers depend on dataset, sample rate, and hardware — I cite order-of-magnitude figures from the original papers and standard benchmarks and flag them as approximate. MOS is on a 5-point scale; RTF is GPU, batch 1, and the reciprocal is the speedup over real time.

![A matrix comparing Griffin-Lim, WaveNet, HiFi-GAN, BigVGAN, and Vocos across approximate MOS, GPU real-time factor, parameter count, and whether the vocoder is universal across speakers and domains.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-4.png)

| Vocoder | Approach | MOS (approx) | RTF (GPU, batch 1) | Params | Universal? |
| --- | --- | --- | --- | --- | --- |
| Griffin-Lim | Iterative phase recovery | ~2.5–3.0 | fast (CPU) | 0 | n/a |
| WaveNet | Autoregressive, per-sample | ~4.4 | ~200–500 (slower than RT) | ~24M | no |
| WaveGlow | Normalizing flow | ~4.2 | ~0.01 | ~90M | no |
| MelGAN | GAN, MSD only | ~3.6–4.0 | ~0.005 | ~4M | no |
| HiFi-GAN V1 | GAN, MPD + MSD | ~4.3 | ~0.004 | ~14M | no |
| BigVGAN | Large GAN, snake + anti-alias | ~4.4 | ~0.01 | ~112M | yes |
| Vocos | GAN, Fourier-head | ~4.3 | ~0.002 | ~13M | yes |

Read the table as a frontier. **Griffin-Lim** is the priorless baseline — fast and free but audibly buzzy. **WaveNet** sets the quality ceiling but at an RTF *above* 1, meaning slower than real time by two to three orders of magnitude — beautiful and unusable for interactive work. The **GAN row** (MelGAN → HiFi-GAN → BigVGAN → Vocos) is where the field lives: WaveNet-class MOS at RTF three to four orders of magnitude lower, in models small enough to ship. **HiFi-GAN** is the default workhorse; **BigVGAN** trades size for universality (one model for any speaker or domain); **Vocos** trades a little architectural exoticism for the lowest RTF of all by moving the upsampling into a fixed iSTFT.

![A taxonomy tree splitting vocoders into classical signal-processing inversion, slow likelihood-based neural models, and the fast GAN branch, with HiFi-GAN and BigVGAN as time-domain GAN vocoders and Vocos as a Fourier-head GAN vocoder.](/imgs/blogs/gan-vocoders-hifi-gan-and-fast-synthesis-7.png)

The **GAN-vs-WaveNet speedup** is the cleanest single statistic. WaveNet at RTF ~250 (slower than real time) versus HiFi-GAN at RTF ~0.004 (faster than real time) is a wall-clock speedup of roughly $250 / 0.004 \approx 60{,}000\times$ for the same quality. That is not a typo — moving from per-sample autoregression to a parallel feed-forward pass is a five-orders-of-magnitude change in throughput, and it is *the* reason neural vocoders are deployable at all. The quality was always achievable; the GAN made it cheap.

## 11. How vocoders compose with TTS and with codecs

A vocoder almost never runs alone. It is the back half of two very common stacks, and seeing both clarifies what it is for.

**Vocoder + TTS acoustic model.** The classic two-stage TTS pipeline is text → mel → waveform: an **acoustic model** (Tacotron 2, FastSpeech 2, the acoustic half of VITS) turns text into a mel-spectrogram, and a **vocoder** turns the mel into audio. The split is deliberate and load-bearing: the acoustic model handles the *linguistic* problem (what should the prosody and content be?) in the compact, ear-aligned mel space, and the vocoder handles the *signal* problem (what does the waveform look like?). Because the two are decoupled, you can swap a better vocoder into an existing TTS system and improve audio quality without retraining the acoustic model — exactly the upgrade that fixed my buzzing system. The forward post [text-to-speech: from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits) builds the acoustic-model half in full; here the takeaway is that the vocoder is the swappable, speed-critical back end, and that mismatched mel parameters between the acoustic model and the vocoder are the single most common cause of "I plugged in a vocoder and got noise." One subtlety worth flagging: a vocoder trained on *ground-truth* mels can degrade on the *predicted* mels an acoustic model emits, because predicted mels are slightly over-smoothed — fine-tuning the vocoder on the acoustic model's own outputs (or training them jointly, as VITS does) closes that gap.

**The codec decoder is a GAN vocoder.** Here is the unifying insight that ties this post to the codec track. A neural audio codec — EnCodec, DAC, Mimi — is an encoder → residual-VQ quantizer → decoder, and **the decoder is, architecturally, a GAN vocoder.** It upsamples a compact latent (the quantized codes, rather than a mel) back to a waveform, and it is trained with *exactly the same machinery*: a HiFi-GAN-style transposed-conv generator, multi-scale and multi-period (or multi-resolution STFT) discriminators, feature-matching, and a multi-scale mel/spectral reconstruction loss. The codec post [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec) goes deep on those shared discriminators. The mental shift is this: a **mel-vocoder** maps mel → waveform; a **codec decoder** maps codec-tokens → waveform; both are the same kind of network solving the same "expand a compact acoustic latent into a clean full-rate waveform" problem with the same adversarial recipe. When you see an AR audio language model (VALL-E, MusicGen) "generate audio," what generates the *tokens* is the language model, but what turns those tokens back into a *waveform* is a GAN-vocoder-class decoder. The GAN vocoder is the universal back end of modern audio generation, whether the front end is a mel acoustic model or a codec language model.

This is also the cleanest way to place GAN vocoders against the **diffusion** alternative. Diffusion vocoders (DiffWave, WaveGrad, PriorGrad) and diffusion codecs exist and can hit excellent quality, but they need multiple denoising steps, so their RTF is typically an order of magnitude or more worse than a one-pass GAN vocoder — the sibling post [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio) covers that family. The honest rule, which I will sharpen in the recommendations: when a GAN vocoder hits your quality bar, it is faster than a diffusion vocoder by a large margin, and you should use it.

It helps to see why the two families differ so sharply on speed even though both are non-autoregressive. A diffusion vocoder starts from noise and *iterates* a denoising network $N$ times to refine its way to a clean waveform; each of those $N$ steps is a full forward pass, so its cost is roughly $N$ times a single GAN-vocoder pass. Even an aggressively distilled diffusion vocoder running in, say, 6 steps is 6× the work of HiFi-GAN's single step, and the early diffusion vocoders used dozens of steps. The GAN vocoder amortizes *all* of that refinement into the adversarial training — the discriminators teach the generator, during training, to produce a clean waveform in one shot, so inference pays nothing for refinement. That is the trade in one sentence: diffusion moves the cost to inference (many steps, every time you generate), GAN moves it to training (a hard adversarial game, paid once). For a back end that runs on every single utterance forever, paying once at training time is overwhelmingly the right side of that trade, which is the structural reason GAN vocoders, not diffusion vocoders, became the production default — and why the few-step consistency and distillation ideas from the image series' [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) are what diffusion vocoders must borrow to even compete on speed.

## 12. Stress-testing the GAN vocoder: where it breaks

A principal engineer's job is knowing where the tool fails, so let me push on it.

**What happens at an unseen speaker?** A HiFi-GAN trained on a single speaker (or LJSpeech) often sounds subtly off on a voice it never heard — a slightly wrong timbre, a faint metallic edge on the new speaker's vowels — because it learned a prior specific to its training distribution and is now extrapolating. This is the **generalization gap**, and it is the main reason BigVGAN exists: scale the model and the data until the prior covers "all speakers, all domains." If your application is multi-speaker or open-domain (a TTS that clones arbitrary voices, a codec decoder behind a music model), reach for a universal vocoder (BigVGAN, Vocos) over a single-domain HiFi-GAN.

**What happens to fine detail it didn't see?** GAN vocoders can **hallucinate** plausible-but-wrong high-frequency detail. Because the mel discarded fine spectral structure and the generator fills it with a learned prior, on out-of-distribution content the filled-in detail may be confident and natural-sounding but *not* what the original had — a vocoder might render a fricative or a cymbal with texture that sounds right in general but is wrong in particular. This is usually inaudible and harmless for speech, but it is a real limitation for high-fidelity music and for any application where faithfulness to a specific source matters more than general naturalness. It is the flip side of "no exact likelihood": the GAN optimizes for *realistic*, not *faithful*.

**What happens when the input mel is bad?** A vocoder is only as good as the mel it is handed. If the acoustic model emits an over-smoothed or out-of-distribution mel, the vocoder faithfully renders garbage into clean-sounding garbage. The fix is co-adaptation — fine-tune the vocoder on the acoustic model's predicted mels, or train end-to-end. A vocoder cannot rescue a broken front end; it can only render what it is given.

**What happens when you need it on CPU or on-device?** RTF 0.004 is a *GPU* number. On a phone CPU the same HiFi-GAN might run at RTF 0.1–0.5 — still faster than real time, but the margin shrinks, and a heavy BigVGAN can blow the budget. This is where the small, efficient vocoders (HiFi-GAN, Vocos) and weight-norm folding and `torch.compile` earn their keep, and where Vocos's frame-rate (rather than sample-rate) backbone gives it a structural advantage on-device — because its heavy ConvNeXt network processes only ~86 frames per second of audio rather than 24,000 samples, the per-second compute is dramatically lower, which is exactly the property you want when the CPU is the constraint. The standard compression toolkit applies directly: int8 quantization of the convolutions, operator fusion, and pruning the widest channels typically shave a HiFi-GAN's CPU latency by 2–4× with negligible MOS loss, because a vocoder is convolution-bound and convolutions quantize well. The edge-AI techniques in [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — quantization, operator fusion, structured pruning — apply to a vocoder almost without modification, and a quantized HiFi-GAN or Vocos is a realistic on-device TTS back end today.

**What happens when the vocoder is the bottleneck, not the model?** In a fast non-autoregressive TTS, the acoustic model can emit a mel in tens of milliseconds, and then a *slow* vocoder dominates the latency. People sometimes reach for a diffusion vocoder for the last few tenths of a MOS point and quietly 10–50× their latency. Profile end to end before optimizing: if the vocoder is 90% of your latency, the right move is a faster vocoder (Vocos), not a faster acoustic model.

## 13. Case studies: real numbers from the literature

A few concrete, citable results to ground the claims, with the usual caveat that exact figures depend on the listening-test setup and hardware.

**HiFi-GAN (Kong et al., 2020).** On LJSpeech, HiFi-GAN V1 reported MOS of about 4.36 against a ground-truth MOS of about 4.45 — statistically close to natural — while running at an RTF on the order of 0.004 on a V100 (about 167× faster than real time) and a remarkable RTF well below 1 even on CPU. The smaller V2/V3 variants trade a little quality for even more speed and a footprint small enough for mobile. The decisive comparison in the paper was against WaveNet and WaveGlow: comparable or better MOS at a fraction of the parameters and orders of magnitude more throughput.

**BigVGAN (Lee et al., NVIDIA, 2022).** Trained on a large, diverse corpus, BigVGAN demonstrated strong **zero-shot generalization** — high quality on unseen speakers, languages, and even non-speech audio (singing, instruments, environmental sound) where single-domain vocoders degrade. The snake activation and anti-aliased sampling were ablated to show each contributes to out-of-distribution robustness. The cost is size (the largest variant is ~112M parameters) and a correspondingly higher but still-sub-real-time RTF.

**Vocos (Siuzdak, 2023).** Vocos matched HiFi-GAN's quality on standard benchmarks while reporting a *lower* RTF (it cites roughly an order-of-magnitude speedup over comparable time-domain GAN vocoders in some settings) precisely because its heavy network runs at the frame rate and the iSTFT does the upsampling. It also provided a mel-conditioned *and* an EnCodec-token-conditioned variant, underscoring the point that a vocoder and a codec decoder are the same kind of object.

**The codec connection (EnCodec / DAC).** EnCodec (Défossez et al., 2022) and the Descript Audio Codec (Kumar et al., 2023) both use HiFi-GAN-lineage decoders trained with multi-scale and multi-period/multi-resolution-STFT discriminators and feature-matching — the exact recipe in this post — to reconstruct waveforms from quantized codes at high quality and low bitrate. DAC in particular reported strong reconstruction at ~8 kbps across speech, music, and general audio, and its decoder is, for all practical purposes, a universal GAN vocoder conditioned on RVQ tokens instead of a mel.

## 14. When to reach for a GAN vocoder (and when not to)

A decisive recommendation section, because the point of all this is to choose well.

**Reach for HiFi-GAN** when you need a fast, high-quality vocoder for a known domain — a single-speaker or small-multi-speaker TTS, a fixed music style, anything where the training distribution matches deployment. It is small, battle-tested, runs faster than real time on a phone, and there are good pretrained checkpoints. It is the default; start here.

**Reach for BigVGAN** when you need **universality** — vocoding arbitrary unseen speakers, multiple languages, or music plus speech with one model, and you can afford ~112M parameters and a slightly higher RTF. If your TTS clones arbitrary voices or your codec/music model produces open-domain audio, the generalization is worth the size.

**Reach for Vocos** when **latency is the constraint**, especially on-device or for streaming, and you want HiFi-GAN-class quality at a lower RTF. Its frame-rate backbone plus iSTFT upsampling is the structurally fastest option, and it comes in both mel- and codec-conditioned flavors.

**Do not reach for Griffin-Lim** for any production audio — it is the right tool only for a quick, dependency-free, learning-free debug listen of a mel, or as a phase initializer. Its buzz is disqualifying for anything a user hears.

**Do not reach for an autoregressive (WaveNet) vocoder** in 2026 unless you have an unusual reason — a GAN vocoder matches its quality at tens of thousands of times the speed. WaveNet's place now is historical and pedagogical.

**Do not reach for a diffusion vocoder** when a GAN vocoder hits your quality bar. Diffusion vocoders can edge out GANs on the last fraction of a MOS point in some studies, but they pay 10–50× in latency for multiple denoising steps. Only consider one when you have a *measured* quality deficit that a GAN cannot close *and* the latency budget to absorb the cost — which, for most speech and most music, you do not. When in doubt, the GAN vocoder is the right default, and you should make the diffusion vocoder *prove* its quality advantage on your data before paying for it.

**Do not assume a vocoder is plug-and-play across mel definitions.** The single most common operational failure is a mel-parameter mismatch (sample rate, `n_fft`, `hop`, `n_mels`, `fmin`/`fmax`, log vs linear, normalization) between whatever produces the mel and the vocoder. Match them exactly, and fine-tune the vocoder on your acoustic model's *predicted* mels if you can.

## 15. Key takeaways

- **Vocoding is phase reconstruction.** A mel-spectrogram threw away phase (and blurred fine frequency detail); the vocoder's job is to invert it into a clean, natural waveform, which is an underdetermined problem that a learned prior solves far better than signal processing.
- **Griffin-Lim is the priorless baseline and it buzzes.** It recovers *a* consistent phase, not a *natural* one, so it sounds robotic and metallic. Use it only for a quick debug listen.
- **GAN vocoders broke the quality-speed trade.** A feed-forward convolutional generator produces a waveform in *one parallel pass*, hitting WaveNet-class MOS at an RTF on the order of 0.004 — hundreds of times faster than real time, versus WaveNet's hundreds of times *slower*.
- **The two discriminators are the whole trick.** The multi-period discriminator folds the waveform at prime periods to catch pitch periodicity (killing the buzz); the multi-scale discriminator catches broadband texture at multiple resolutions. Together they make audible-but-magnitude-invisible errors visible to a gradient.
- **The three-part loss matters as a whole.** Adversarial makes it *real*, feature-matching makes it *stable*, mel-reconstruction makes it *correct*; drop any one and quality degrades in a specific way.
- **Always report RTF with the device, batch size, warm-up, and synchronize.** "Faster than real time" is $1/\text{RTF}$; without the device and batch it is meaningless.
- **The codec decoder is a GAN vocoder.** EnCodec, DAC, and Mimi decoders use the same generator-plus-discriminators recipe to turn RVQ tokens (instead of a mel) into a waveform. The GAN vocoder is the universal back end of modern audio generation.
- **Choose by domain and latency.** HiFi-GAN for a known domain, BigVGAN for universality, Vocos for the lowest latency; never autoregress, and only pay for diffusion when it *measurably* beats the GAN on your data.
- **The GAN vocoder is fast and clean but not faithful.** It optimizes for *realistic*, not *faithful*, so it can hallucinate plausible-but-wrong fine detail and generalize imperfectly to unseen speakers — match the training distribution or go universal.

## 16. Further reading

- **Kong, Kim, Bae (2020), "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"** — the multi-period + multi-scale discriminator and MRF generator; the reference design this whole post builds on.
- **Kumar et al. (2019), "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis"** — the first practical GAN vocoder, the multi-scale discriminator, and the feature-matching loss.
- **Lee et al. (2022), "BigVGAN: A Universal Neural Vocoder with Large-Scale Training"** — the snake activation, anti-aliased sampling, and universal-vocoder result.
- **Siuzdak (2023), "Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders"** — predicting STFT coefficients and inverting with a single iSTFT for the lowest RTF.
- **van den Oord et al. (2016), "WaveNet: A Generative Model for Raw Audio"** — the autoregressive vocoder that set the quality bar and the speed problem GAN vocoders solved.
- **Griffin and Lim (1984), "Signal Estimation from Modified Short-Time Fourier Transform"** — the classical iterative phase-reconstruction baseline.
- **Défossez et al. (2022), "High Fidelity Neural Audio Compression" (EnCodec)** and **Kumar et al. (2023), "High-Fidelity Audio Compression with Improved RVQGAN" (DAC)** — the codec decoders that are GAN vocoders conditioned on RVQ tokens.
- Within this series: the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), the representations post [representing sound: waveforms, spectrograms, and perception](/blog/machine-learning/audio-generation/representing-sound-waveforms-spectrograms-and-perception), the codec post [EnCodec, DAC, and the modern codec](/blog/machine-learning/audio-generation/encodec-dac-and-the-modern-codec), the diffusion-vocoder alternative [diffusion for audio](/blog/machine-learning/audio-generation/diffusion-for-audio), the forward TTS post [text-to-speech: from Tacotron to VITS](/blog/machine-learning/audio-generation/text-to-speech-from-tacotron-to-vits), the conditioning post [conditioning and control in audio generation](/blog/machine-learning/audio-generation/conditioning-and-control-in-audio-generation), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack). For the GAN parallel from the image side, see [generative adversarial networks and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost).
