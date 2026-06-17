---
title: "EnCodec, DAC, and the Modern Neural Codec"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A deep dive into the production neural codecs — SoundStream, EnCodec, DAC, and Mimi — that turn waveforms into the tokens every modern audio language model actually generates, with runnable code and real numbers."
tags:
  [
    "audio-generation",
    "audio-synthesis",
    "neural-audio-codec",
    "encodec",
    "descript-audio-codec",
    "residual-vector-quantization",
    "generative-ai",
    "deep-learning",
    "music-generation",
  ]
category: "machine-learning"
subcategory: "Audio Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/encodec-dac-and-the-modern-codec-1.png"
---

The first time I swapped a codec under a music model, nothing else in the system changed — same transformer, same training data, same sampling code — and the generated clips went from "promising but slightly underwater" to "I would not have known this was synthetic." I had not touched the generative model at all. All I did was retrain it on tokens from a better codec. That experience reorganized how I think about the whole audio stack: the codec is not a preprocessing detail you pick once and forget. It is the alphabet your model writes in, and a richer alphabet lets the same model say more. A blurry, half-collapsed codebook caps your quality no matter how big the language model on top gets, because the model can only ever reconstruct what the codec's decoder can express.

This post is about the codecs that sit at that chokepoint in 2024–2026: Meta's **EnCodec**, the **Descript Audio Codec (DAC)**, and **Mimi**, the codec inside Moshi. They are the reason a text-to-music model or a voice-cloning model can treat sound like a sequence of discrete symbols at all. By the end you will be able to encode and decode audio at several bitrates with real tooling, read the codes' shape and compute the bitrate yourself, run the codebook-usage diagnostic that explains *why* DAC beats EnCodec at the same number of bits, and choose a codec deliberately for streaming speech versus offline music. We will go deep on the science — why factorized, normalized codes raise codebook usage, what each loss term in the training recipe actually fixes, and why a causal encoder buys you latency — and we will keep it concrete with code you can paste and run.

![A dataflow graph of the modern neural codec showing a waveform entering a conv encoder, a residual vector quantizer, and a conv decoder, with spectral discriminators and reconstruction losses feeding a total objective](/imgs/blogs/encodec-dac-and-the-modern-codec-1.png)

If you have not read the two posts this one builds on, skim them first. [Neural audio codecs: the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound) sets up the encoder→quantizer→decoder shape and the VAE analogy, and [residual vector quantization (RVQ)](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) derives the multi-codebook quantizer, the rate-distortion trade, the straight-through gradient, and codebook collapse — the machinery I will lean on without re-deriving. This post is the next step in that arc: how the field turned the RVQ template into production codecs that actually ship. It sits in the recurring frame of this series, the **audio stack** (waveform → neural-codec tokens / mel latent → generative model → vocoder/decoder → waveform) under the tension of **fidelity × controllability × speed × length**. The codec owns the first and last arrows of that stack, and the better it is, the better everything downstream can be.

## 1. SoundStream: the template everything copies

Before EnCodec there was **SoundStream** (Zeghidour et al., 2021), and almost every codec since is a variation on its three ideas. It is worth understanding SoundStream not as a historical footnote but as the blueprint, because once you see its shape you see it everywhere.

SoundStream is, at heart, three components wired in series. First, a **fully convolutional encoder**: a stack of strided 1D convolutions that takes a 24 kHz waveform — 24,000 floating-point samples per second — and downsamples it by a large factor (320× in the canonical configuration) into a sequence of latent vectors. That 320× striding turns 24,000 samples per second into **75 latent frames per second**, each frame a vector summarizing about 13 milliseconds of audio. Second, a **residual vector quantizer (RVQ)**: instead of snapping each latent frame to a single codebook entry, RVQ snaps it to a *sum* of entries drawn from a stack of codebooks, where each codebook quantizes the residual error left by the ones before it. This is the rate-distortion lever — more codebooks means more bits per frame and lower distortion, fewer means a smaller bitrate. Third, a **fully convolutional decoder** that mirrors the encoder with transposed convolutions, upsampling the quantized frames back to 24,000 samples per second.

The piece that made SoundStream sound *good* rather than merely *compressed* was **adversarial training**. A plain autoencoder trained only to minimize reconstruction error on the waveform or spectrogram learns to produce an output that is, in a mean-squared sense, close to the target — and mean-squared-close audio sounds muffled and buzzy, because the loss happily averages away the high-frequency detail and phase structure that the ear cares about most. SoundStream added a **discriminator** — a separate network trained to tell real audio from the codec's reconstructions — and trained the codec to fool it. The adversarial signal pushes the decoder toward outputs that are *perceptually* indistinguishable from real audio, not just numerically close, and that is what kills the underwater quality. We will return to exactly why in section 5; for now, the headline is that SoundStream fused three ideas — a conv autoencoder, residual VQ, and a GAN — into a single end-to-end codec, and trained it to compress 24 kHz speech and music down to roughly 3 to 18 kbps with quality that beat the hand-engineered Opus codec at comparable rates.

The figure at the top of this post is the SoundStream shape, generalized: a conv encoder, a quantizer, a conv decoder, and a discriminator that only exists at training time. Hold that picture. EnCodec, DAC, and Mimi are all this diagram with different choices bolted into each box. The art of modern codec design is almost entirely about *which* choices, and the rest of this post is a tour of them.

One number to anchor on, because it recurs: the **frame rate** (latent frames per second) and the **bitrate** (bits per second) are different knobs, and conflating them is a common confusion. Frame rate is set by the encoder's total striding — 320× striding on 24 kHz audio gives 75 fps, full stop. Bitrate is set by how many codebooks the RVQ uses *and* how big each codebook is. With $N_q$ codebooks of size 1024 (so $\log_2 1024 = 10$ bits each) at 75 fps, the bitrate is $75 \times N_q \times 10$ bits per second. One codebook is 750 bps; eight codebooks is 6 kbps. Same encoder, same frame rate, a 8× range of bitrates — just by choosing how deep to read the RVQ stack. That decoupling is the source of EnCodec's multi-bitrate flexibility, which we turn to next.

## 2. Meta EnCodec: the codec that became the default

When people say "the codec," in a MusicGen or AudioGen or VALL-E-style context, they usually mean **EnCodec** (Défossez et al., Meta AI, 2022). It is the SoundStream template, refined and packaged into something robust enough that an entire generation of audio language models tokenized against it. Understanding EnCodec in detail pays off twice: once because it is genuinely good, and once because it is the baseline DAC was built to beat, so its weaknesses are the setup for the next section.

### The architecture

EnCodec ships in two main configurations: a **24 kHz mono** model and a **48 kHz stereo** model. The 24 kHz model is the one almost everyone uses for generative work, so I will describe it concretely. Its encoder is a **streaming convolutional encoder** — a stack of residual conv blocks with strided downsampling that achieves the same 320× reduction as SoundStream, producing **75 frames per second** of 128-dimensional latents. The word "streaming" is load-bearing and we will dig into it in section 8: the convolutions are **causal** (padded only on the left, never looking into the future), which means you can feed audio in and get tokens out as it arrives, without waiting for the whole clip. The decoder mirrors this with causal transposed convolutions.

In the middle sits the **residual vector quantizer**, and here EnCodec makes its first deliberate choice: the RVQ is trained to support a **variable number of codebooks at inference time**. During training, EnCodec randomly samples how many of its (up to 32) codebooks to use on each step, so the model learns to produce a coherent reconstruction whether you read 1 codebook or all 32. At inference you pick the depth, and that picks the bitrate. EnCodec's published bitrates for the 24 kHz model are **1.5, 3, 6, 12, and 24 kbps**, corresponding to roughly 2, 4, 8, 16, and 32 codebooks respectively (each codebook is size 1024 = 10 bits, at 75 fps). This is the single most useful property of EnCodec for a downstream LM: you can trade tokens-per-second against fidelity without retraining the codec, and you can decide how many of the codec's "channels" your generative model has to predict.

![A layered stack diagram of EnCodec showing the streaming conv encoder, the variable-depth residual VQ, the token grid, the transformer entropy model, and the decoder](/imgs/blogs/encodec-dac-and-the-modern-codec-2.png)

The figure shows the second deliberate choice, the one that distinguishes EnCodec from a plain SoundStream: a **small transformer-based entropy model**. After RVQ produces its grid of discrete codes, those codes are not equally likely — some codebook entries fire far more often than others, and there is temporal structure (this frame's code is correlated with the last frame's). A lightweight transformer is trained to *predict the next code given the past*, and its predictions drive an **arithmetic coder** that losslessly compresses the code stream. Because predictable symbols cost fewer bits under entropy coding, this shaves the *transmitted* bitrate further — Meta reports roughly a **25% to 40% reduction** in bitrate from the entropy model, depending on content, for free in quality terms (it is lossless; the reconstruction is identical, you just store the tokens more compactly). For *generative* use this stage is usually bypassed — the language model wants the raw token grid, not an arithmetic-coded bitstream — but for *compression* use (EnCodec as an actual file codec) it matters a great deal, and it is a real architectural difference from SoundStream worth knowing.

### Why it became the default

EnCodec became the de facto codec for open audio generation for a few unglamorous reasons that matter more than any single benchmark. It was **released with strong pretrained checkpoints** and clean code, integrated into `audiocraft` (so it came for free with MusicGen and AudioGen) and into 🤗 `transformers` as `EncodecModel`. It was **robust across content** — speech, music, environmental sound — rather than tuned for one domain. Its **multi-bitrate** design let MusicGen choose a token budget. And it was **causal/streaming**, which mattered for the real-time ambitions of the models built on it. When VALL-E reframed text-to-speech as "predict EnCodec tokens with a language model," it inherited all of that. The lesson is one I have seen repeat across ML: the artifact that wins is rarely the one with the single best number; it is the one that is good *enough*, available, robust, and easy to build on. EnCodec was that artifact for audio tokens.

But "good enough and everywhere" left a real quality gap on the table, and the gap had a specific, diagnosable cause: EnCodec does not use its codebooks well. Many of its codebook entries are effectively dead — never selected, or selected so rarely they contribute almost nothing — which means the bitrate you are *paying for* is larger than the information you are *actually transmitting*. That inefficiency is exactly what the Descript Audio Codec was built to fix, and fixing it is the most instructive story in this whole post.

## 3. The science of codebook usage: why DAC beats EnCodec at equal bitrate

Here is the central scientific claim of this post, stated plainly: **a codec's effective fidelity at a fixed bitrate is bounded by how much of its codebook it actually uses, and DAC's main advantage over EnCodec is that it uses nearly all of its codebook while EnCodec uses a fraction of it.** Everything else — snake activations, better discriminators — adds polish, but the codebook-usage fix is the load-bearing wall. Let me build up why from first principles, because it is genuinely elegant and it changes how you think about quantization.

### Bits paid for versus bits used

Start with information theory. A codebook of size $K$ can in principle carry $\log_2 K$ bits per selection — for $K = 1024$, that is 10 bits. But that maximum is only achieved if all $K$ entries are used *equally often*. The actual information a codebook carries is the **entropy** of its usage distribution:

$$
H = -\sum_{k=1}^{K} p_k \log_2 p_k \quad \text{bits per frame},
$$

where $p_k$ is the probability that codebook entry $k$ is selected. Entropy is maximized — equal to $\log_2 K$ — exactly when the distribution is uniform ($p_k = 1/K$ for all $k$), and it collapses toward zero as the distribution concentrates on a few entries. If half your codebook is never selected, you have at most $\log_2(K/2) = 9$ bits of real capacity in a slot you are paying 10 bits to store. If only a quarter is alive, you are down to 8 effective bits in a 10-bit slot. **You are paying full freight and shipping a fraction of the cargo.**

A useful single number that captures this is **codebook usage** (or perplexity): the fraction of entries that get selected with non-negligible frequency over a representative batch, or equivalently the **perplexity** $2^{H}$ expressed as a fraction of $K$. A perplexity of 1024 out of 1024 means every entry pulls its weight; a perplexity of 200 means three-quarters of your codebook is dead weight. The DAC paper's measurements put EnCodec's per-codebook usage well below full and DAC's near complete — and that delta, propagated across all the codebooks in the RVQ stack, is most of the quality difference.

![A before-and-after comparison showing EnCodec's full-dimension low-usage codes on one side and DAC's factorized, L2-normalized, near-100-percent-usage codes on the other](/imgs/blogs/encodec-dac-and-the-modern-codec-4.png)

### Why EnCodec wastes its codebook — and DAC's two fixes

So why does a normally-trained RVQ leave most of its codebook unused? Two mechanisms, and DAC has a targeted fix for each.

**Mechanism one: high-dimensional lookup makes most entries unreachable.** EnCodec quantizes in the *full* latent dimension — it does a nearest-neighbor lookup of a 128-dimensional encoder vector against a 128-dimensional codebook. In high dimensions, distances behave badly: the encoder's outputs occupy a thin, curved manifold, and a codebook initialized to fill the high-dimensional space has most of its entries sitting in regions the encoder never visits. Those entries are *geometrically unreachable* — no encoder output is ever closest to them — so they die and stay dead. This is the curse of dimensionality biting quantization directly.

DAC's fix is **factorized codes**. Before the nearest-neighbor lookup, DAC projects each encoder frame *down* to a tiny code dimension — on the order of **8 dimensions** — does the codebook lookup in that low-dimensional space, and projects the chosen code *back up* to the full latent dimension afterward. The codebook lives in the 8-D space, not the 1024-D space. In 8 dimensions the encoder's outputs and the codebook entries can actually fill the space; there are no vast unreachable regions; entries that would have died in 128-D now get used. This is the single most important trick, and it is almost embarrassingly cheap — two extra linear projections around the lookup.

![A dataflow graph of DAC's factorized lookup, showing the encoder frame projected down, both the frame and the codebook L2-normalized, a cosine nearest-neighbor match, and near-100-percent codebook usage](/imgs/blogs/encodec-dac-and-the-modern-codec-6.png)

**Mechanism two: Euclidean lookup is dominated by magnitude, not direction.** When you match an encoder vector to a codebook by ordinary Euclidean distance, vectors with large magnitude dominate the distance, and the lookup ends up sensitive to *how loud* a frame is rather than *what it contains*. This skews selection toward a few high-magnitude codes.

DAC's fix is **L2-normalized codes**. It normalizes both the (projected) encoder frame and the codebook entries to unit length before the lookup, which turns the nearest-neighbor search into a **cosine-similarity** match — it picks the code pointing in the most similar *direction*, ignoring magnitude. On the unit sphere, codes spread out evenly and selection is governed by the content's direction, which is the perceptually meaningful part. Combined with factorization, this is what drives usage toward 100%. The figure above traces the path: project down, normalize both sides, cosine-match, and the result is a codebook with no dead entries.

The reason this matters downstream is worth stating explicitly, because it is the whole point. **A fully-used codebook has more effective bits at the same nominal bitrate.** If EnCodec's 10-bit codebook is really carrying 8 bits and DAC's is carrying close to 10, then at the *same* number of codebooks and the *same* nominal kbps, DAC is transmitting meaningfully more information about the waveform — and more information means a more faithful reconstruction. That is why DAC, in its paper, reconstructs 44.1 kHz audio at roughly **8 kbps** with quality that EnCodec needs a substantially higher bitrate to match. You did not get something for nothing; you got something for *using what you were already paying for*.

#### Worked example: counting the effective bits

Make this concrete with a back-of-envelope. Take an RVQ with 8 codebooks, each size 1024, at 75 fps. Nominal bitrate: $8 \times 10 \times 75 = 6{,}000$ bps = 6 kbps. Now suppose EnCodec's per-codebook usage gives an average entropy of about 8 bits (a plausible figure for a poorly-utilized 1024-entry codebook). Effective information rate: $8 \times 8 \times 75 = 4{,}800$ bps — you are paying 6 kbps to ship 4.8 kbps of real signal, a 20% tax. Now suppose DAC's factorized, normalized codes push average entropy to 9.7 bits. Effective rate: $8 \times 9.7 \times 75 = 5{,}820$ bps — almost the full 6 kbps does real work. The codecs are *nominally* identical in bitrate, but DAC is moving about 21% more information per second through the same channel. That extra information is exactly the high-frequency detail and transient sharpness you hear as "better." This is approximate — I am illustrating the mechanism with round numbers, not quoting a measured DAC entropy — but the *direction* and rough magnitude are right, and the framing is the thing to keep: **usage is effective bitrate.**

This is also the place to retire a tempting but wrong intuition. You might think "just make the codebook bigger" — go from 1024 to 8192 entries for 13 bits instead of 10. But if your *usage* is the problem, a bigger codebook makes it worse: you have added more entries for the encoder to fail to reach, and the geometric unreachability that killed entries in the 1024 case kills even more of them in the 8192 case. The win is not capacity, it is *utilization*. DAC's small 8-D factorized codebook beats EnCodec's larger-dimensional one not because it can represent more, but because it wastes almost none of what it can represent. If you remember one thing from this post, make it that.

## 4. The third fix: snake activations for periodic signals

DAC's other notable departure from EnCodec is small but tells you something deep about audio. EnCodec, like most conv nets, uses standard activations (ELU in its case) between conv layers. DAC replaces them with the **snake activation**, introduced for periodic signals by Liu et al. (2020) and adopted by the BigVGAN vocoder:

$$
\text{snake}(x) = x + \frac{1}{\alpha}\sin^2(\alpha x),
$$

where $\alpha$ is a learnable per-channel parameter. The intuition is direct and physical. Audio is, more than almost any other signal, made of **periodicity** — a sung note, a violin string, a vowel, a synth pad are all approximately periodic waveforms, and a good chunk of "this sounds real" is "the periodic structure is clean." A standard ReLU or ELU is monotonic and has no built-in notion of repetition; to reproduce a periodic pattern it has to laboriously stack many piecewise-linear segments, and the seams between those segments show up as exactly the kind of high-frequency grunge and aliasing that the ear flags as "synthetic." The snake activation has a **periodic term baked in** — the $\sin^2(\alpha x)$ — so a single unit can natively produce a repeating output, and the network gets a strong *inductive bias toward periodic signals*. The learnable $\alpha$ lets each channel tune its own period.

The practical effect is that DAC's decoder reproduces sustained tones, harmonics, and vowel formants more cleanly than an ELU decoder of the same size, with less of the buzzing and metallic edge that plague codecs and vocoders. It is the same reason BigVGAN beats HiFi-GAN on out-of-distribution instruments — we will revisit that when we get to [GAN vocoders](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis), which share this exact lineage of discriminators and activations. The takeaway: **the structure of audio (it is periodic) should be reflected in the architecture (use a periodic nonlinearity)**, and snake is the cheapest way to do that.

### How the gradient gets through the quantizer

There is a question lurking under all of this that you must answer to train a codec at all: the nearest-neighbor lookup that picks a codebook entry is a *non-differentiable* operation — `argmin` has no useful gradient — so how does the encoder ever learn? The codecs inherit the answer from VQ-VAE, and it is worth stating precisely because it is the seam where most custom-codec training goes wrong. The trick is the **straight-through estimator**: on the forward pass you quantize honestly (snap the encoder output $z_e$ to its nearest codebook entry $z_q$), but on the backward pass you *pretend the quantizer was the identity* and copy the decoder's gradient straight onto the encoder, $\frac{\partial \mathcal{L}}{\partial z_e} \approx \frac{\partial \mathcal{L}}{\partial z_q}$. In code this is the one-liner $z_q = z_e + \text{stop\_grad}(z_q - z_e)$ — the value is the quantized one, the gradient flows as if it were continuous. It is biased (the encoder is being told its continuous output mattered when actually only the snapped code did), but it works because the **commitment loss** keeps $z_e$ close to $z_q$, so the lie is small. The [RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) derives this; the reason it matters *here* is that DAC's factorization changes where the estimator operates — the straight-through pass now runs through the low-dimensional projected space, which is part of why DAC's gradients are better behaved and its codebooks stay alive.

The codebook entries themselves are updated one of two ways, and the choice has real consequences. The first is to treat them as ordinary parameters and let the optimizer move them via the codebook loss (pull each entry toward the encoder outputs assigned to it). The second, used by many strong codecs, is an **exponential moving average (EMA)** update: each codebook entry is maintained as a running average of the encoder vectors that selected it, decayed with a momentum like 0.99, outside the optimizer entirely. EMA is more stable — it does not fight the encoder's learning-rate schedule — and it is the natural home for the **dead-code reset** trick from section 7: when an entry's EMA cluster size falls below a threshold, you reassign it to a random recent encoder output, giving it a fresh chance to become useful. If you ever train your own codec, the EMA-plus-reset recipe is the difference between a codebook that fills up and one that quietly collapses to a handful of live entries while your reconstruction loss looks deceptively healthy.

## 5. The loss recipe: what each term fixes

A codec is defined as much by *how* it is trained as by its architecture, and the loss is where the perceptual quality is won or lost. Both EnCodec and DAC are trained on a weighted sum of several complementary losses, and the instructive way to learn them is to ask, for each one, *what artifact appears if you remove it?* That framing turns an intimidating list into a set of debugging tools.

![A layered stack of the codec loss components — time-domain L1, multi-scale mel, adversarial, feature matching, and VQ commitment — summing to a balanced objective](/imgs/blogs/encodec-dac-and-the-modern-codec-3.png)

**Time-domain reconstruction (L1 on the waveform).** This is the simplest term: the L1 distance between the input waveform and the reconstruction, sample by sample. It anchors the coarse shape — overall amplitude envelope, gross timing — and without it training is unstable because the spectral losses below are invariant to certain time shifts. But on its own it produces muffled, lifeless audio, because matching a waveform sample-by-sample in L1 cannot capture phase and fine spectral detail; the optimizer settles for a smooth average. It is necessary but nowhere near sufficient.

**Multi-scale mel reconstruction.** This is the workhorse fidelity term, and it deserves the most attention. You compute mel-spectrograms of both the target and the reconstruction at **several STFT window sizes** — short windows (say 32 or 64 samples) for sharp time resolution, long windows (up to 2048 samples) for sharp frequency resolution — and penalize their difference (L1 and L2). Why *multi-scale*? Because of the time-frequency uncertainty principle we derived in [the mathematics of audio signals](/blog/machine-learning/audio-generation/the-mathematics-of-audio-signals): a single STFT window cannot resolve both fast transients (drum hits, plosive consonants) and fine pitch simultaneously — a short window blurs frequency, a long window blurs time. By summing the loss over many window sizes, you force the codec to get *both* right: the short-window terms police transient sharpness, the long-window terms police tonal accuracy. Remove the multi-scale aspect and use one window, and you get a codec that is crisp on transients but smeared on pitch, or vice versa. This term is why the reconstruction has the right *timbre*.

**Adversarial loss (the spectral discriminators).** This is where the buzz dies. The discriminator is a network — or in modern codecs, *several* networks — trained to distinguish real audio from reconstructions, and the codec is trained to fool it. The crucial design choice in EnCodec and DAC is that the discriminators operate on **spectral representations at multiple scales** (a multi-scale STFT discriminator), not on the raw waveform alone. Here is why that matters: the reconstruction losses above are *averaging* losses — they penalize the mean error, and the mean-optimal output of an averaging loss is a blurry, over-smoothed signal, because blurring reduces average error even though it sounds wrong. A blurry spectrogram is *exactly* the buzzing, watery artifact you hear from a codec trained only on reconstruction. The adversarial loss breaks this: a discriminator can easily tell a blurry, over-smooth spectrogram from a real sharp one, so to fool it the decoder *must* produce sharp, realistic high-frequency structure with the right stochastic texture. The discriminator does not care about average error; it cares about *realism*, and realism is what the ear cares about too. Remove the adversarial term and the codec immediately regresses to muffled, buzzy, "underwater" audio — this single term is the difference between a codec that sounds compressed and one that sounds transparent.

**Feature matching.** GANs are notoriously unstable to train; the discriminator can overpower the generator and provide useless gradients. The feature-matching loss stabilizes this: it asks the *intermediate activations* of the discriminator to match between real and reconstructed audio, giving the generator a smoother, more informative gradient than the raw real/fake signal. It does not change *what* the codec learns to produce so much as make the learning *converge*. Remove it and training gets jittery and slow; the final quality suffers because the GAN never settles.

**VQ commitment loss.** This is the quantizer's own regularizer, inherited from VQ-VAE and covered in the [RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq). It is a term that pulls the encoder's continuous outputs *toward* the codebook entries they get snapped to, so the encoder "commits" to the discrete vocabulary rather than drifting away from it (which would make the straight-through gradient meaningless). It also includes the codebook update that pulls entries toward the encoder outputs assigned to them. This term, plus DAC's factorization and normalization, is what keeps the codebook *alive and used* — remove it and codes drift, codebook collapse accelerates, and your effective bitrate craters.

DAC's improvements to this recipe are mostly in the **discriminators**. Where EnCodec used a multi-scale STFT discriminator, DAC uses a **multi-band, multi-scale** discriminator bank — discriminators that look at different frequency bands separately as well as different time scales. The reasoning is that artifacts often live in specific bands (a buzz in the high end, a muddiness in the low-mids), and a discriminator that examines each band on its own is harder to fool with a band-specific defect than one that pools all frequencies together. Better discriminators mean a more demanding adversary, and a more demanding adversary forces a better decoder. This, plus snake plus factorized codes, is the full DAC story: same loss *shape* as EnCodec, sharper teeth.

#### Worked example: silencing one loss at a time

Suppose you are debugging a codec you trained and the output has an audible high-frequency buzz. The loss-by-loss frame gives you a diagnostic protocol. First, check whether the **adversarial** term is contributing — if your discriminator has collapsed (loss pinned, gradients near zero), the decoder has no pressure to be sharp and will produce exactly that buzz; restart the GAN with feature matching weighted up. If the discriminator is healthy but the buzz persists in a *specific band*, your discriminators may not be looking at that band — this is the case DAC's multi-band discriminators were designed for, and adding a band-specific discriminator there is the fix. If instead the problem is *muddiness* rather than buzz — smeared transients, soft drum hits — suspect the **multi-scale mel** term: your shortest STFT window may be too long to resolve the transients, so add a shorter window to the loss. And if the *codebook usage* diagnostic (section 7) shows half your codes dead, no loss tuning will save you — you need DAC's factorization and normalization. Each artifact maps to a term; that mapping is the most useful thing to carry out of this section.

## 6. Encode and decode in code: EnCodec via 🤗 transformers

Enough theory — let us actually run a codec. The cleanest entry point is 🤗 `transformers`, which wraps EnCodec as `EncodecModel`. We will load audio, encode it at several bitrates, inspect the token grid's shape, compute the bitrate ourselves to confirm we understand it, decode back to a waveform, and measure reconstruction error.

```bash
pip install "transformers>=4.40" torchaudio soundfile
# DAC has its own package, used later in this section:
pip install descript-audio-codec
```

Start with loading and the multi-bitrate encode. The thing to watch is the **shape of the codes** — it tells you the frame rate and the number of codebooks directly.

```python
import torch
import torchaudio
from transformers import EncodecModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# 24 kHz mono EnCodec — the one used for generative work.
model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Load a wav and resample to the codec's native rate.
wav, sr = torchaudio.load("speech.wav")          # (channels, samples)
wav = torchaudio.functional.resample(wav, sr, 24000)
wav = wav.mean(0, keepdim=True)                   # force mono: (1, samples)
duration_s = wav.shape[-1] / 24000

inputs = processor(raw_audio=wav.squeeze(0).numpy(),
                   sampling_rate=24000,
                   return_tensors="pt").to(device)

# Encode at several bitrates and inspect the token grid.
for bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0]:   # kbps
    with torch.no_grad():
        enc = model.encode(inputs["input_values"],
                           inputs["padding_mask"],
                           bandwidth=bandwidth)
    codes = enc.audio_codes              # (1, 1, n_q, n_frames)
    n_q = codes.shape[-2]
    n_frames = codes.shape[-1]
    fps = n_frames / duration_s
    # Each codebook entry is log2(1024) = 10 bits.
    measured_kbps = (n_q * 10 * fps) / 1000
    print(f"bandwidth={bandwidth:>4} kbps | n_q={n_q:>2} | "
          f"frames={n_frames} | fps~{fps:.1f} | "
          f"computed={measured_kbps:.1f} kbps | shape={tuple(codes.shape)}")
```

Run this and you will see the frame rate stay pinned near **75 fps** at every bandwidth (the encoder striding does not change), while `n_q` — the number of codebooks read — climbs from 2 at 1.5 kbps to 32 at 24 kbps. The `computed` kbps you print from $n_q \times 10 \times \text{fps}$ should match the requested bandwidth, which confirms the bitrate arithmetic from section 1 is exactly what the model is doing: bitrate is codebooks times bits-per-code times frames-per-second, nothing more mysterious. This is the single most clarifying thing to run when you are learning a codec — *see the token grid resize as you turn the bitrate knob.*

Now decode and measure how much you lost. The honest reconstruction metric here is not a single number but a small basket — see [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) for why — but for a quick sanity check, an SI-SDR or a mel-spectrogram L1 tells you whether the codec is behaving.

```python
import torch.nn.functional as F

def reconstruct(model, processor, wav, bandwidth, device):
    inp = processor(raw_audio=wav.squeeze(0).numpy(),
                    sampling_rate=24000, return_tensors="pt").to(device)
    with torch.no_grad():
        enc = model.encode(inp["input_values"], inp["padding_mask"],
                           bandwidth=bandwidth)
        dec = model.decode(enc.audio_codes, enc.audio_scales,
                           inp["padding_mask"])[0]
    return dec.cpu()                              # (1, 1, samples)

def mel_l1(a, b, sr=24000):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)
    la = torch.log(mel(a).clamp(min=1e-5))
    lb = torch.log(mel(b).clamp(min=1e-5))
    n = min(la.shape[-1], lb.shape[-1])
    return F.l1_loss(la[..., :n], lb[..., :n]).item()

orig = wav.unsqueeze(0)                           # (1, 1, samples)
for bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0]:
    rec = reconstruct(model, processor, wav, bandwidth, device)
    n = min(orig.shape[-1], rec.shape[-1])
    err = mel_l1(orig[..., :n], rec[..., :n])
    print(f"{bandwidth:>4} kbps  mel-L1 = {err:.4f}")
```

You will see the mel-L1 error fall monotonically as bitrate rises — more codebooks, lower distortion, exactly the rate-distortion curve from the RVQ post. The interesting part is *where the curve flattens*: somewhere around 6 kbps for EnCodec-24k, the marginal quality per extra kbps drops off, which is why 6 kbps is such a common operating point for generative models — it is the knee of the curve, the most quality per token. Below it, quality falls off fast; above it, you are paying tokens for diminishing returns. Knowing where your codec's knee is, *for your content*, is one of the highest-leverage measurements you can make before training a generative model on top of it.

Two practical gotchas bite people on their first encode/decode round-trip, and both are easy to miss. The first is the **`audio_scales`** return value: EnCodec normalizes each input chunk's loudness before encoding and stores the scale factor so the decoder can restore the original level — if you pass the codes to `decode` without the matching `audio_scales`, your reconstruction comes back at the wrong amplitude (often dramatically quiet or clipped), and it looks like a quality bug when it is really a bookkeeping bug. Always carry the scales alongside the codes. The second is **padding and length mismatch**: the codec works in frames, so a reconstruction is generally not sample-for-sample the same length as the input (it rounds to a frame boundary), which is why every comparison snippet above truncates both signals to `min(len_a, len_b)` before computing a metric. Forget that truncation and your mel-L1 or SI-SDR will be dominated by a one-frame misalignment rather than by real codec error — a classic way to convince yourself a perfectly good codec is broken. Both of these are the kind of detail that never shows up in a paper but eats an afternoon the first time you hit it, which is exactly why the runnable snippets here handle them explicitly.

For **DAC**, the API is different but the shape of the workflow is identical:

```python
import dac
from audiotools import AudioSignal

# Download + load the 44.1 kHz DAC model.
dac_path = dac.utils.download(model_type="44khz")
dac_model = dac.DAC.load(dac_path).to(device).eval()

signal = AudioSignal("music.wav")                 # auto-handles resampling
signal.to(device)

with torch.no_grad():
    x = dac_model.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, _, _ = dac_model.encode(x)
    y = dac_model.decode(z)

print("DAC codes shape:", tuple(codes.shape))     # (B, n_codebooks, n_frames)
recon = AudioSignal(y.cpu(), dac_model.sample_rate)
recon.write("music_dac.wav")
```

Note DAC's native rate is **44.1 kHz** and its frame rate is higher (~90 fps with a hop of 512), so its token grid is denser than EnCodec-24k's — a real consideration for the downstream LM, which has more tokens per second to predict. That density buys the higher sample rate and fidelity; whether you want it depends on whether your content needs 44.1 kHz. For speech, 24 kHz is plenty and EnCodec's lower frame rate is a feature. For music with airy high frequencies, DAC's 44.1 kHz earns its tokens.

## 7. The codebook-usage diagnostic you should always run

Section 3 argued that codebook usage *is* effective bitrate. Here is the code to measure it, because a claim about usage you cannot measure is a claim you cannot trust. The diagnostic is simple: encode a representative batch of audio, histogram which codebook entries get selected per codebook, and compute the entropy and the fraction of entries that fire.

```python
import torch
from collections import defaultdict

@torch.no_grad()
def codebook_usage(model, processor, wav_iter, bandwidth, device,
                   codebook_size=1024):
    # counts[q] = tensor of selection counts for codebook q
    counts = defaultdict(lambda: torch.zeros(codebook_size))
    for wav in wav_iter:                          # iterable of (1, samples)
        inp = processor(raw_audio=wav.squeeze(0).numpy(),
                        sampling_rate=24000, return_tensors="pt").to(device)
        codes = model.encode(inp["input_values"], inp["padding_mask"],
                             bandwidth=bandwidth).audio_codes
        codes = codes.squeeze(0).squeeze(0)       # (n_q, n_frames)
        for q in range(codes.shape[0]):
            idx, c = torch.unique(codes[q].cpu(), return_counts=True)
            counts[q][idx] += c.float()

    report = []
    for q in sorted(counts):
        p = counts[q] / counts[q].sum().clamp(min=1)
        nz = (counts[q] > 0).float().mean().item()           # fraction alive
        ent = -(p[p > 0] * p[p > 0].log2()).sum().item()     # bits
        perplexity = 2 ** ent
        report.append((q, nz, ent, perplexity))
        print(f"codebook {q:>2}: alive={nz*100:5.1f}%  "
              f"entropy={ent:5.2f} bits  perplexity={perplexity:6.1f} "
              f"/ {codebook_size}")
    return report
```

Run this over a few hundred clips for EnCodec and for DAC at a matched bitrate and compare the columns. The pattern the DAC paper documents, and that you will reproduce, is that EnCodec's deeper codebooks (the later RVQ stages, quantizing small residuals) show **low `alive` fractions and entropy well below 10 bits** — many entries never fire, the residual is small and concentrated — while DAC's codebooks stay **near fully alive with entropy close to 10 bits all the way down the stack.** That difference, summed over the stack, is the effective-bitrate gap from section 3, now measured rather than asserted.

This diagnostic is not academic. If you are training your *own* codec or fine-tuning one, codebook usage is the first thing to watch, because **collapse is silent** — your reconstruction loss can look fine while half your codebook quietly dies, and you only discover it when the downstream model underperforms for reasons you cannot localize. Watching `alive` per codebook during training catches it early. The fixes, in order of leverage: factorize and normalize the codes (DAC's approach, the biggest lever), add or strengthen the commitment loss, periodically **re-initialize dead codes** to encoder outputs from the current batch (a common trick — reassign an unused entry to a random recent encoder vector so it gets a second life), and lower the codebook size if you genuinely have more capacity than your data needs. The [RVQ post](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) covers collapse mechanics in depth; the point here is that the diagnostic above is how you *see* it.

## 8. The streaming/causal constraint and the latency it buys

The word "streaming" attached to EnCodec and Mimi is not marketing; it is a hard architectural constraint with a precise payoff, and understanding it is essential for any real-time audio application. Let us derive the latency benefit from first principles.

A convolutional layer with a kernel that spans $k$ samples normally pads symmetrically — $\lfloor k/2 \rfloor$ samples on each side — so its output at time $t$ depends on inputs from $t - \lfloor k/2 \rfloor$ to $t + \lfloor k/2 \rfloor$. That dependence on *future* inputs ($t + \lfloor k/2 \rfloor$) is the problem: to compute the output now, you must wait for audio that has not arrived yet. Stack many such layers and the total look-ahead — the **algorithmic latency** — grows to a substantial fraction of a second. A non-causal codec literally cannot produce its first token until it has buffered enough future audio to fill every layer's right-side context. For a live conversation, that buffering *is* the latency you feel as lag.

A **causal** convolution pads only on the *left*: the output at time $t$ depends only on inputs from $t - (k-1)$ to $t$, never on the future. Stack causal layers and the look-ahead is **zero** — every output depends only on past and present samples. The codec can emit a token the moment a frame's worth of audio arrives, with no future buffering. The price is a small amount of modeling power (the network cannot use future context to disambiguate the present), which in practice costs a little fidelity, but for streaming applications it is the difference between a usable real-time system and an unusable one. This is the same causal-convolution idea WaveNet pioneered, which we cover in [autoregressive audio models](/blog/machine-learning/audio-generation/autoregressive-audio-models-wavenet-to-audiolm).

The latency that remains in a causal codec is just the **frame duration** plus compute: you must accumulate one frame of samples before you can quantize it. For EnCodec-24k at 75 fps, a frame is $1/75 \approx 13.3$ ms; for Mimi at 12.5 fps, a frame is $1/12.5 = 80$ ms. That frame duration is the *floor* on a causal codec's latency, and it is why Mimi quotes a latency around 80 ms — it is fundamentally a one-frame system. The trade is visible here: Mimi's low frame rate (12.5 fps) is great for bitrate and for the downstream LM (fewer tokens to predict per second), but it sets a higher latency floor than EnCodec's 75 fps would. Frame rate, bitrate, and latency are three faces of the same striding choice, and you cannot independently maximize all three.

#### Worked example: the latency budget of a voice agent

Say you are building a full-duplex voice agent and you want end-to-end response latency under 300 ms — the threshold below which a conversation feels natural. Budget it. The codec's encode latency is one frame: 80 ms for Mimi. The language model must then produce its first response token, which on a well-optimized 7B-class model on an A100 might be 50–100 ms to first token. The codec's decode latency is another frame, 80 ms, before the first audio comes out. That is already roughly $80 + 75 + 80 = 235$ ms before you add network round-trips. The lesson: in a real-time voice stack the *codec's frame rate shows up twice* (encode and decode), so a codec with a 13 ms frame (EnCodec-75fps) leaves far more headroom than one with an 80 ms frame (Mimi-12.5fps) — *but* the 80 ms frame buys Mimi a far lower bitrate and token rate, which makes the LM cheaper and faster. There is no free lunch; the right choice depends on whether your bottleneck is latency or LM compute. This is precisely the kind of trade [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack) walks through end to end.

## 9. Mimi: the split semantic+acoustic codec for streaming speech

Mimi (Défossez et al., the codec inside **Moshi**, 2024) is the newest of the production codecs and it makes a design move the others do not, one that points squarely at the next post in this series. It is a **split codec**: its token stream is factored into a *semantic* part and an *acoustic* part, in a single model.

![A dataflow graph of Mimi splitting its tokens, with one encoder routing the first codebook to a distilled semantic token and the rest to acoustic RVQ codebooks before a causal decoder](/imgs/blogs/encodec-dac-and-the-modern-codec-7.png)

Here is the idea. A normal codec's tokens are purely *acoustic* — they describe the sound (timbre, pitch, texture) but carry no explicit notion of *linguistic content*. That is fine for compression, but for a speech *language model* it is wasteful: the model has to learn the mapping from acoustics to meaning from scratch, inside the LM. Mimi instead makes its **first codebook carry a semantic token** — trained by distillation from a self-supervised speech model (WavLM-style) so that the first token captures *what is being said* — while the **remaining codebooks carry acoustic detail** — *how it sounds*. The result is a single 12.5 fps, roughly **1.1 kbps** stream where the first channel is meaning and the rest is fidelity. This is the cleanest of the production codecs at extremely low bitrate, tuned specifically for streaming speech, and it is causal/streaming so it slots into a real-time loop.

Why does splitting help so much for a speech LM? Because it lets the language model spend its capacity efficiently: the semantic token gives it a stable, linguistically-meaningful backbone to model, and the acoustic tokens fill in the realization. This is the **semantic-vs-acoustic token paradigm** that powers AudioLM, MusicLM, and VALL-E-style models — the next post, [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens), is entirely about why you need both and how they compose, so I will leave the deep treatment there. The preview to carry forward: a codec is not just a compressor; the *structure* of its tokens shapes how easy the downstream model's job is, and Mimi's split structure is a codec designed hand-in-hand with the language model that consumes it. That co-design — codec and LM evolving together — is where the frontier is heading.

## 10. The design axes that matter, and how to choose

Step back from the individual codecs and look at the **axes** along which they differ, because once you can name the axes you can choose a codec for a new problem without memorizing models. Six axes carry almost all the decision.

![A comparison matrix of SoundStream, EnCodec, DAC, and Mimi across sample rate, frame rate, bitrate, codebook usage, and reconstruction quality](/imgs/blogs/encodec-dac-and-the-modern-codec-5.png)

**Bitrate (kbps).** The headline. Lower bitrate means fewer tokens for the downstream model to predict — faster, cheaper generation — but lower fidelity. The relationship is the rate-distortion curve; find its knee for your content and operate there.

**Frame rate (fps).** How many tokens per second of audio. This is the downstream model's *sequence length per second*, and it dominates the LM's compute and its ability to model long-range structure. A lower frame rate (Mimi's 12.5 vs EnCodec's 75) is a *huge* win for the LM — six times fewer tokens to predict for a given duration — which is a big part of why low-frame-rate codecs are the streaming-speech trend. But frame rate also sets the latency floor (section 8) and limits how fast a transient the codec can resolve.

**Codebook usage.** The efficiency multiplier from section 3. Two codecs at the same nominal bitrate are not equal if one uses its codebook fully and the other does not. Always check usage before comparing kbps across codecs.

**Streaming / causal.** Whether the codec can run in real time with bounded latency. Required for live conversation and full-duplex; irrelevant for offline music rendering where you have the whole clip up front and can use a non-causal codec for a fidelity bump.

**Sample rate.** 24 kHz is plenty for speech (human speech energy is mostly below 8 kHz); 44.1 kHz matters for music with airy highs (cymbals, breath, room). A higher sample rate means a denser token grid for the same frame duration, so it costs the LM more.

**Channels (mono / stereo).** Speech is mono; music is often stereo. Stereo doubles your token budget (or requires a codec that models the channels jointly). Most generative work is mono first, stereo as a later refinement.

The decision tree falls out of these axes directly.

![A decision tree for choosing a codec, branching from the use case into streaming speech, general audio LM, and music or high-fidelity leaves selecting Mimi, EnCodec, and DAC](/imgs/blogs/encodec-dac-and-the-modern-codec-8.png)

For **streaming conversational speech** where latency is the constraint, reach for **Mimi**: its low frame rate, low bitrate, causal design, and semantic split are all tuned for exactly that, and its tokens make the speech LM's job easier. For a **general-purpose audio LM** where you want a robust, well-supported default and the ecosystem matters more than the last decibel of fidelity, reach for **EnCodec**: it is everywhere, battle-tested, multi-bitrate, and the tooling is mature. For **offline music or high-fidelity audio** where quality at a given bitrate is the whole game and you can afford 44.1 kHz, reach for **DAC**: its near-full codebook usage and snake-activation periodicity handling make it the best reconstructor at equal bits. And if you are training your own model, the meta-rule: **pick the codec first, because it sets the ceiling, then build the LM to its token format.** Swapping the codec later means retraining the LM from scratch — the tokens are a different language.

## 11. Case studies and real numbers

Let me ground the comparisons in named results from the literature and shipped systems. I will flag where a number is approximate; the prior posts in this series are emphatic that false precision is the field's besetting sin, and codecs are no exception.

**EnCodec vs DAC at equal bitrate (Kumar et al., DAC paper, 2023).** The headline result of the DAC paper is that DAC reconstructs **44.1 kHz audio at ~8 kbps** with quality (measured by a battery of objective metrics and listening tests) that EnCodec requires a substantially higher bitrate to match — the paper frames DAC as beating EnCodec "at all bitrates" on its universal-audio benchmark, with the gap attributed primarily to codebook usage and the improved discriminators. The mechanism is the one we derived: DAC's per-codebook usage is near complete while EnCodec's is partial, so DAC's nominal bitrate does more real work. The lesson to internalize is that "8 kbps" is not a fixed quality — it is a quality *given how well the codec uses those bits*, and DAC uses them better.

**EnCodec's role as the generative default (Défossez et al., 2022; Copet et al., MusicGen, 2023; Wang et al., VALL-E, 2023).** MusicGen generates music by predicting EnCodec tokens with a single-stage transformer, using a codebook-interleaving pattern to handle the multiple RVQ codebooks in one autoregressive stream. VALL-E reframes TTS as predicting EnCodec tokens conditioned on a text prompt and a 3-second speaker prompt. In both cases the codec is EnCodec, operating around **6 kbps** (the knee of its rate-distortion curve), and the *quality ceiling of the generative model is the reconstruction quality of EnCodec at that bitrate*. This is the practical payoff of the whole post: when MusicGen's outputs have a slightly compressed sheen, part of that is the model and part is the codec — and you cannot fix the codec part by scaling the model. (Approximate: exact bitrates vary by configuration and model size.)

**Mimi inside Moshi (Défossez et al., 2024).** Moshi, the full-duplex spoken-dialogue model, runs on Mimi at **12.5 fps, ~1.1 kbps**, with the semantic-token split described in section 9 and a quoted theoretical latency around **80 ms** from its frame rate. This is the codec engineered for the streaming-speech frontier: extremely low bitrate, low token rate (which keeps the speech LM tractable in real time), causal, and structurally aligned to the LM via the semantic split. It is the clearest example in production of a codec co-designed with its language model.

**SoundStream as the origin (Zeghidour et al., 2021).** SoundStream demonstrated that a single learned codec at **3 kbps** could beat Opus at **6 kbps** and match it at lower rates on subjective tests — the result that proved neural codecs were not just a research curiosity but could outcompete decades of hand-engineered signal-processing codecs. Everything since is a refinement of its template. (Approximate: subjective comparisons depend on content and test protocol.)

#### Worked example: an honest A/B between two codecs

You want to decide, for a music-generation project, whether to retrain on DAC tokens or stay on EnCodec. Here is the protocol I would run, with the kind of numbers to expect. Fix a held-out set of 2,000 music clips at 44.1 kHz, about 10 seconds each. Reconstruct each clip through EnCodec-24k at 6 kbps (resampling as needed) and through DAC-44k at a matched ~6–8 kbps, with a fixed random seed and identical preprocessing. Now measure three things, not one. First, **codebook usage** via the section-7 diagnostic — expect EnCodec's deeper codebooks to read well under full while DAC's read near complete; this is the *mechanism* and you want to confirm it holds on your content. Second, a **distributional metric**: FAD with a stated embedder — and here the embedder choice matters enormously, because VGGish-FAD at 16 kHz is structurally deaf above 8 kHz and will under-credit DAC's high-frequency advantage, so report CLAP-FAD at 48 kHz where the airy detail actually counts (the [audio quality metrics](/blog/machine-learning/audio-generation/audio-quality-metrics) post is emphatic about this). Third, a **small listening test**: 15–20 raters, CMOS (comparative mean opinion score) on paired A/B clips, asking "which sounds more faithful to the original." A plausible outcome consistent with the DAC paper's direction is something like a CMOS in DAC's favor of around +0.3 to +0.5 on full-band music, concentrated on clips with prominent cymbals, strings, and breathy vocals — exactly the high-frequency content EnCodec's usage gap costs it. The decision rule: if the CMOS gap is positive *and* the codec is not the only thing changing (you will have to retrain the LM on DAC's denser 44.1 kHz token grid, which costs more compute per second of audio), weigh the quality gain against the higher token rate. For full-band music it usually wins; for speech, where 24 kHz is plenty, it usually does not. Numbers here are approximate and meant to size the decision, not to quote a controlled benchmark — but the *shape* of the result, DAC ahead on high-frequency-rich content via better usage, is robust.

Here is the consolidated comparison. Treat the quality and usage columns as *directional* — they are synthesized from the papers' reported trends, not a single controlled benchmark, and I have marked the soft ones.

| Codec | Sample rate | Frame rate | Typical bitrate | Codebook usage | Streaming | Reconstruction (relative) | Primary use |
|---|---|---|---|---|---|---|---|
| SoundStream | 24 kHz | ~75 fps | 3–18 kbps | Moderate | Yes (causal) | Good (the template) | The original neural codec |
| EnCodec 24k | 24 kHz mono | 75 fps | 1.5–24 kbps | Low (partial) | Yes (causal) | Good — generative default | MusicGen, AudioGen, VALL-E |
| EnCodec 48k | 48 kHz stereo | 150 fps | up to 24 kbps | Low (partial) | Non-causal | Good (stereo, hi-rate) | High-rate stereo compression |
| DAC 44.1k | 44.1 kHz | ~90 fps | ~8 kbps universal | Near 100% | Non-causal | Best at equal bitrate | Offline hi-fi, music |
| Mimi (Moshi) | 24 kHz | 12.5 fps | ~1.1 kbps | High (split) | Yes (causal) | Speech-tuned, very low rate | Streaming speech LMs |

And the downstream-quality table that ties the codec to the model it feeds — the single most important relationship in this post:

| Generative model | Codec used | Codec bitrate (approx) | What the codec caps |
|---|---|---|---|
| MusicGen | EnCodec 24k | ~6 kbps | Music fidelity ceiling, that compressed sheen |
| AudioGen | EnCodec 24k | ~6 kbps | Sound-effect realism |
| VALL-E | EnCodec 24k | ~6 kbps | Speech naturalness & speaker fidelity |
| Moshi | Mimi | ~1.1 kbps | Real-time speech quality & latency |

The pattern across the table is the thesis of the post in one glance: **a better codec is a better audio LM, because the LM can only ever reconstruct what the codec's decoder can express.** Improve the codec — raise its codebook usage, sharpen its discriminators, give it periodic activations — and every model trained on its tokens inherits the gain, for free, the next time you retrain. That is why so much frontier effort has gone into codecs that look, from the outside, like a solved problem.

## 12. When to reach for which codec (and when not to)

A decisive recommendation, because surveys without verdicts waste your time.

**Reach for EnCodec when** you want a robust, well-supported default and you are building on the open ecosystem — MusicGen, AudioGen, anything in `audiocraft` or 🤗 `transformers`. It is the safe choice precisely because everyone uses it: the tooling is mature, the checkpoints are good, and "EnCodec at 6 kbps" is a known quantity your collaborators will recognize. **Do not** reach for it if you are chasing the last decibel of music fidelity at a fixed bitrate — its codebook underutilization leaves quality on the table that DAC recovers.

**Reach for DAC when** reconstruction quality at a given bitrate is the whole game: offline music generation, high-fidelity audio, anything where 44.1 kHz matters and you can afford the denser token grid. Its near-full codebook usage is a genuine, measurable advantage, not a marketing claim — run the diagnostic from section 7 and see it. **Do not** reach for it for low-latency streaming speech: it is non-causal in its standard configuration and its 44.1 kHz frame grid is denser than a speech LM needs.

**Reach for Mimi when** you are building streaming, low-latency, conversational speech and you want the codec and the LM to fit together — its low frame rate keeps the speech model tractable in real time and its semantic split gives the LM a meaningful backbone. **Do not** reach for it for music or general audio: it is speech-tuned at a very low bitrate, and that low bitrate that serves speech so well will starve music of the fidelity it needs.

**Do not train your own codec** unless you have a specific reason the existing three cannot serve — a wildly different sample rate, a domain (bioacoustics, industrial sound) far from speech and music, or a research goal in codec design itself. Codecs are expensive and finicky to train (GAN instability, codebook collapse, weeks of compute), and EnCodec/DAC/Mimi cover the common cases well. The far more common and higher-leverage move is to *pick the right existing codec for your downstream model* and spend your training budget on the model, not the tokenizer. When you do need a custom codec, start from DAC's recipe — factorized normalized codes, snake activations, multi-band discriminators — because it is the current state of the art and the open implementation is clean.

One more anti-pattern worth naming: **do not compare codecs by nominal bitrate alone.** "Codec X at 6 kbps beats codec Y at 6 kbps" is meaningless without codebook usage, sample rate, and the same evaluation protocol — the bits are not interchangeable. The honest comparison is reconstruction quality (a metrics basket: FAD with a stated embedder, a mel-distance, and a small listening test) at a *matched* bitrate *and* matched content, with usage reported. Anything less and you are comparing labels, not codecs.

## 13. Stress-testing the choice

Pose the engineering problem honestly and push on it. You have chosen a codec; now break it.

**What happens at a lower bitrate?** As you drop codebooks, the codec drops information, and it drops the *least perceptually important* information first — usually the high frequencies and the fine stereo image — because the reconstruction and adversarial losses were weighted toward what the ear notices most. So a codec at half its operating bitrate does not sound *half as good*; it sounds *band-limited and slightly hollow*, like the high end was rolled off. For a generative model this can actually be acceptable — a slightly band-limited token space is easier for the LM to model, and you may prefer the cheaper, faster generation. The stress test is to listen at the *low* end of your bitrate range and decide whether the artifact is one your users will tolerate.

**What happens when the codec drops the high frequencies?** This is the most common low-bitrate failure and it has a signature: cymbals lose their shimmer, sibilants ("s" sounds) get muffled, and the whole thing sounds like it is behind a curtain. If your downstream model's outputs have this signature and the model is fine, the codec bitrate is your culprit — raise it, or switch to a codec (DAC) whose better usage retains more high-frequency detail at the same rate.

**What happens when you ask for 4 minutes?** The codec itself does not care about length — it is convolutional, it streams, it will happily encode an hour. The length problem is the *downstream LM's*, not the codec's: at 75 fps, four minutes is 18,000 frames, and at 6 kbps with 8 interleaved codebooks that is a very long token sequence for an autoregressive model to keep coherent. This is exactly why a low-frame-rate codec (Mimi at 12.5 fps, or a codec tuned for long-form) is attractive for long generation — fewer tokens per second means the LM's context covers more *time*. The codec choice and the length ambition are coupled through the frame rate.

**What happens when the codec is the bottleneck, not the model?** Sometimes you scale the generative model, retrain, and quality barely moves. Before blaming the model, *reconstruct your training audio through the codec alone and listen.* If the codec's own reconstruction already has the artifact you are trying to fix, the model can never beat it — you are at the codec's ceiling, and the fix is a better codec, not a bigger model. This single diagnostic — "what does the codec alone do to my audio?" — would save an enormous amount of wasted model-scaling effort, and almost nobody runs it first.

The thread through all four stress tests: the codec sets boundaries the rest of the stack lives inside. Its bitrate bounds fidelity, its frame rate bounds length and latency, its codebook usage bounds effective quality, and its reconstruction is a hard ceiling on everything downstream. Choosing it well, and measuring it honestly, is the highest-leverage decision in the early design of an audio generation system — which is why this post exists between the [RVQ](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq) machinery and the [semantic-vs-acoustic](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens) paradigm that builds on it. The codec is the alphabet; the rest of the stack is what you write with it.

It is worth seeing the codec as the audio analogue of the latent autoencoder in image generation. A latent diffusion model does not denoise pixels; it denoises a VAE's compressed latent, and the quality of that VAE bounds the whole system — the parallel is exact and we drew it deliberately. If you want the image-side version of "the compressor sets the ceiling," [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) is the same lesson in pixels: a learned compressor turns an intractable signal into a short code a generative model can handle, and the compressor's fidelity is the system's fidelity. Audio's codec is image's VAE, and the design pressures rhyme.

## 14. Key takeaways

- **The codec is the alphabet your audio LM writes in.** A better codec lifts every model trained on its tokens, for free, the next time you retrain — and no amount of scaling the LM can beat the codec's reconstruction ceiling.
- **SoundStream is the template**: conv encoder + residual VQ + conv decoder + adversarial training. EnCodec, DAC, and Mimi are all that diagram with different choices in each box.
- **Bitrate, frame rate, and latency are three faces of one striding choice.** You cannot independently maximize all three; a low frame rate helps the LM and the bitrate but raises the latency floor.
- **Codebook usage is effective bitrate.** Two codecs at the same nominal kbps are not equal if one uses its codebook fully and the other does not — always measure usage before comparing.
- **DAC beats EnCodec at equal bitrate mainly through codebook usage**: factorized (low-dimensional) codes plus L2-normalized cosine lookup keep nearly every entry alive, while snake activations handle audio's periodicity and multi-band discriminators sharpen the high end.
- **Each loss term fixes a named artifact**: time-L1 anchors the envelope, multi-scale mel fixes timbre across the time-frequency trade, the adversarial term kills the buzz, feature matching stabilizes the GAN, and commitment keeps codes alive.
- **Causal convolutions buy zero look-ahead latency** at a small fidelity cost; the remaining latency floor is one frame duration, which is why Mimi's 12.5 fps quotes ~80 ms.
- **Choose the codec first, then build the LM to its tokens** — swapping codecs later means retraining the model from scratch.
- **Match the codec to the job**: Mimi for streaming speech, EnCodec for the robust general default, DAC for offline high-fidelity music. Do not train your own unless the three genuinely cannot serve you.
- **Reconstruct your audio through the codec alone and listen** before blaming the generative model — the most common silent bottleneck is the codec at its ceiling.

## 15. Further reading

- **Zeghidour, Luebs, Omran, Skoglund, Tagliasacchi — "SoundStream: An End-to-End Neural Audio Codec" (2021).** The template: conv autoencoder + residual VQ + adversarial training, beating Opus at lower bitrates.
- **Défossez, Copet, Synnaeve, Adi — "High Fidelity Neural Audio Compression" (EnCodec, Meta AI, 2022).** The streaming multi-bitrate codec with the transformer entropy model that became the generative default.
- **Kumar, Seetharaman, Luebs, Kumar, Kumar — "High-Fidelity Audio Compression with Improved RVQ" (Descript Audio Codec, 2023).** The codebook-usage fix: factorized + L2-normalized codes, snake activations, better discriminators — beating EnCodec at equal bitrate.
- **Défossez et al. — "Moshi: a speech-text foundation model for real-time dialogue" (2024).** Introduces Mimi, the split semantic+acoustic streaming codec at 12.5 fps and ~1.1 kbps.
- **van den Oord, Vinyals, Kavukcuoglu — "Neural Discrete Representation Learning" (VQ-VAE, 2017).** The origin of the discrete codebook, commitment loss, and straight-through estimator the codecs inherit.
- **Copet et al. — "Simple and Controllable Music Generation" (MusicGen, 2023)** and **Wang et al. — "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (VALL-E, 2023).** The two clearest examples of generative models built directly on EnCodec tokens.
- **🤗 `transformers` `EncodecModel` docs** and the **`descript-audio-codec` (`dac`) repository** — the working APIs used in this post.
- Within this series: [neural audio codecs: the tokenizer of sound](/blog/machine-learning/audio-generation/neural-audio-codecs-the-tokenizer-of-sound), [residual vector quantization](/blog/machine-learning/audio-generation/residual-vector-quantization-rvq), [semantic vs acoustic tokens](/blog/machine-learning/audio-generation/semantic-vs-acoustic-tokens), [GAN vocoders: HiFi-GAN and fast synthesis](/blog/machine-learning/audio-generation/gan-vocoders-hifi-gan-and-fast-synthesis), the foundation [why audio generation is hard](/blog/machine-learning/audio-generation/why-audio-generation-is-hard), and the capstone [building an audio generation stack](/blog/machine-learning/audio-generation/building-an-audio-generation-stack).
