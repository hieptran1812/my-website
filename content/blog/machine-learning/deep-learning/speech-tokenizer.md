---
title: "Speech Tokenizers: Turning Audio Into Discrete Tokens for Generative Speech Models"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "speech-tokenizer",
    "audio-codec",
    "vq-vae",
    "rvq",
    "fsq",
    "encodec",
    "soundstream",
    "cosyvoice",
    "tts",
    "asr",
    "audio",
    "deep-learning",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A deep dive into speech tokenizers — the component that makes LLM-style generative speech possible. Covers VQ-VAE, RVQ, FSQ, acoustic vs semantic tokens, EnCodec, SoundStream, HuBERT, WavTokenizer, CosyVoice's S³ tokenizer, training objectives, evaluation, and the engineering trade-offs that matter in production."
---

## Why Speech Tokenizers Exist

A raw audio waveform at 16 kHz is 16,000 floats per second. A language model cannot directly predict the "next float" — the output space is continuous, unbounded, and phenomenally high-dimensional. The entire paradigm of LLM-based speech generation (VALL-E, Whisper-style decoders, CosyVoice, Tortoise, XTTS, AudioLM, MusicGen) rests on a single trick:

> Convert the continuous audio signal into a sequence of **discrete tokens** from a finite codebook, then train a Transformer to model that sequence like text.

This is what a **speech tokenizer** does. It is the bridge between signal processing and language modeling, and its quality sets the ceiling for everything downstream — generation fidelity, voice cloning accuracy, streaming latency, and how much data the LLM needs to train.

```
Raw waveform (1s at 16kHz = 16,000 floats)
                    ↓
         [ Speech Tokenizer (encoder) ]
                    ↓
       Discrete tokens: [382, 1521, 88, 4093, ...]
                    ↓
            [ Transformer LLM ]
                    ↓
       Predicted tokens: [382, 1521, 88, 4093, ...]
                    ↓
         [ Speech Tokenizer (decoder) ]
                    ↓
          Reconstructed waveform
```

The compression ratio is staggering. EnCodec at 6 kbps represents 1 second of 24kHz audio with 600 tokens — a **40× reduction** versus raw samples, and **1000×+** versus float32 storage. Yet it reconstructs speech at near-perceptual quality.

## Two Families: Acoustic vs Semantic Tokens

All speech tokenizers fall into one of two philosophical camps — and many modern systems use **both** in parallel.

### Acoustic Tokens (Reconstruction-First)

**Goal**: Losslessly compress audio into tokens so the original waveform can be reconstructed with high fidelity.

**How**: Train an autoencoder end-to-end with a reconstruction loss (time-domain + spectrogram + adversarial). The bottleneck is quantized into discrete indices.

**What they encode**: Everything — content, speaker identity, prosody, channel noise, reverb, background sounds. The decoder needs all of it to reconstruct the signal.

**Examples**: SoundStream, EnCodec, DAC (Descript Audio Codec), HiFi-Codec, WavTokenizer.

**Used for**: The final stage in LLM-based TTS — predicting the acoustic tokens that a vocoder (the tokenizer's decoder) turns back into waveform. Also general-purpose neural audio compression (music, effects).

### Semantic Tokens (Content-First)

**Goal**: Extract tokens that represent **what** is being said, not **how** it sounds. Speaker-invariant, prosody-lean representations.

**How**: Cluster or quantize hidden states from a self-supervised or supervised speech model (HuBERT, Wav2Vec 2.0, Whisper encoder) that was trained for content-level tasks like masked prediction or ASR.

**What they encode**: Phonetic content, linguistic structure, coarse prosody. Speaker timbre is deliberately suppressed.

**Examples**: HuBERT k-means tokens, Wav2Vec 2.0 quantized codes, CosyVoice's S³ (Supervised Semantic Speech) tokenizer, AudioLM's semantic stream.

**Used for**: The first stage in LLM-based TTS — the text-conditioned LLM predicts semantic tokens (which are easier because they're phonetically regular). Then a second stage fills in the acoustic detail. Also: voice cloning (semantic tokens are content, speaker embedding is timbre), speech editing.

### The Two-Stage Pattern

Modern systems (AudioLM, VALL-E's variants, CosyVoice, SPEAR-TTS) exploit the division of labor:

```
Text → [LLM 1] → Semantic tokens → [LLM 2 + speaker] → Acoustic tokens → [Vocoder] → Audio
         ↑                            ↑
     content decisions           acoustic realization
     (what to say)               (how to say it)
```

Semantic tokens are a shorter, lower-entropy sequence — the content LLM learns from less data. Acoustic tokens are longer but conditionally easier once semantics are fixed. This factorization is a major reason LLM-TTS works at reasonable data scales.

## The Core Machinery: Vector Quantization

Every speech tokenizer you will encounter is, at heart, an encoder + a **quantizer** + a decoder. The quantizer is what makes the output discrete. Four variants dominate, and knowing their trade-offs is the single most useful thing in this article.

### Vanilla VQ (Vector Quantization)

Given an encoder output $z \in \mathbb{R}^d$, find the nearest code in a learned codebook $\{e_1, e_2, \ldots, e_K\}$:

$$q(z) = e_k \quad \text{where} \quad k = \arg\min_j \| z - e_j \|_2$$

The index $k$ is the token. A codebook of size $K = 1024$ gives 10-bit tokens.

**The gradient problem**: $\arg\min$ has zero gradient. VQ-VAE solves this with the **straight-through estimator** — copy the gradient through the quantizer as if it were identity — plus two auxiliary losses:

$$\mathcal{L}_{\text{VQ}} = \| \text{sg}[z] - e_k \|_2^2 + \beta \| z - \text{sg}[e_k] \|_2^2$$

where $\text{sg}[\cdot]$ is stop-gradient. The first term pulls the codebook toward the encoder outputs; the second ($\beta \approx 0.25$) is the **commitment loss** that keeps the encoder from drifting away from the codebook.

**The dead-code problem**: Early in training, some codes are never selected and receive no gradient — they die. Fixes:
- **EMA updates** (van den Oord et al.): maintain codebook as exponential moving average of encoder outputs assigned to each code, instead of gradient descent. More stable.
- **Random restarts**: periodically re-initialize unused codes to a random encoder output from the current batch.
- **Codebook reset**: if utilization drops below a threshold, reset the entire codebook.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes=1024, code_dim=256, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_codes, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):
        # z: (B, T, D)
        flat = z.reshape(-1, self.code_dim)                    # (BT, D)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )                                                        # (BT, K)
        idx = dist.argmin(dim=1)                                 # (BT,)
        onehot = F.one_hot(idx, self.num_codes).type(flat.dtype) # (BT, K)
        q = self.embed[idx].view_as(z)                           # (B, T, D)

        if self.training:
            # EMA updates — no gradient needed
            self.cluster_size.mul_(self.decay).add_(
                onehot.sum(0), alpha=1 - self.decay
            )
            embed_sum = onehot.t() @ flat                        # (K, D)
            self.embed_avg.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )
            n = self.cluster_size.sum()
            smoothed = (
                (self.cluster_size + self.eps) /
                (n + self.num_codes * self.eps) * n
            )
            self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

        # Commitment loss + straight-through estimator
        commit_loss = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()                                 # STE
        return q, idx.view(z.shape[:-1]), commit_loss
```

### Residual Vector Quantization (RVQ)

A single codebook with $K$ entries gives $\log_2 K$ bits per frame — not enough for high-fidelity audio. Scaling $K$ up hits the dead-code problem hard. **RVQ** solves this by stacking $N$ quantizers, each coding the residual of the previous:

$$r_0 = z, \quad k_i = \arg\min_j \| r_{i-1} - e_j^{(i)} \|_2, \quad r_i = r_{i-1} - e_{k_i}^{(i)}$$

Final quantized output: $q = \sum_{i=1}^{N} e_{k_i}^{(i)}$.

With $N=8$ codebooks of size $K=1024$ each, you get $8 \times 10 = 80$ bits per frame — enough for 6 kbps at 75 Hz frame rate, which is roughly EnCodec's setup. Each frame becomes $N$ parallel tokens.

**Coarse-to-fine structure**: Codebook 1 captures the gross signal; codebook 8 captures fine acoustic detail. This lets you **drop codebooks** at inference for bitrate scaling (generate only the first few and still get intelligible speech, losing detail).

**Downside for LLMs**: A single frame is now $N$ tokens (one per codebook level), so sequence length multiplies. LLM-TTS systems either (a) flatten into $N \times T$ tokens, (b) predict the $N$ codebooks in parallel with a separate head per level, or (c) use a hierarchical model (coarse LLM + fine LLM, as in VALL-E).

### Finite Scalar Quantization (FSQ)

RVQ is powerful but finicky — codebook collapse, EMA tuning, multiple levels to coordinate. **FSQ** (Mentzer et al., 2023) replaces learned codebooks with something almost embarrassingly simple:

1. Project encoder output to $d$ dimensions (e.g., $d = 6$).
2. Apply $\tanh$ to bound each channel to $[-1, 1]$.
3. Round each channel to one of $L$ evenly-spaced levels.
4. Reinterpret the rounded $d$-tuple as an integer index.

Codebook size is $\prod_i L_i$ implicitly — e.g., levels $[8, 6, 5, 5, 5]$ give $8 \times 6 \times 5 \times 5 \times 5 = 6000$ codes with **no codebook to train**. The "codes" are just a grid in low-dimensional space.

```python
class FSQ(nn.Module):
    def __init__(self, levels=(8, 6, 5, 5, 5)):
        super().__init__()
        self.levels = torch.tensor(levels)
        self.num_codes = int(self.levels.prod())
        self.dim = len(levels)

    def forward(self, z):
        # z: (B, T, d) — already projected to dim
        z = torch.tanh(z)
        # Map tanh output [-1, 1] to level indices {0, ..., L-1}
        half_l = (self.levels - 1) / 2
        z_scaled = z * half_l
        z_rounded = z_scaled.round()
        # Straight-through
        z_q = z_scaled + (z_rounded - z_scaled).detach()
        z_q = z_q / half_l
        # Flatten to single integer index per frame
        shift = z_rounded + half_l                           # (B, T, d) in {0..L-1}
        basis = torch.cumprod(
            torch.cat([torch.tensor([1]), self.levels[:-1]]), 0
        )
        idx = (shift * basis).sum(-1).long()                 # (B, T)
        return z_q, idx
```

**Why FSQ matters**: No commitment loss, no codebook EMA, no dead codes (every grid cell exists whether used or not — high utilization is a free side effect). In many settings, FSQ matches or beats RVQ on reconstruction while being dramatically simpler to train. It's winning ground in new tokenizers (WavTokenizer, some CosyVoice 2 variants).

### Lookup-Free Quantization (LFQ)

A close cousin of FSQ that appeared in MAGVIT-v2. Binary quantization per channel: each of $d$ channels is rounded to $\{-1, +1\}$, giving $2^d$ codes implicit in a $d$-bit binary index. Used mainly in image/video tokenizers but applicable to audio. Extremely simple, very large codebook capacity ($d = 18$ gives 262,144 codes).

### Comparison at a Glance

| Method | Codebook | Training | Dead codes | LLM-friendly | Typical use |
|--------|----------|----------|-----------|--------------|-------------|
| Vanilla VQ | Learned, 1 level | Commitment + STE | Common problem | 1 token/frame ✓ | Simple VQ-VAE |
| VQ + EMA | Learned, EMA updates | EMA | Rare | 1 token/frame ✓ | Stable VQ-VAE |
| RVQ | Learned, $N$ levels | EMA, per-level | Per-level tuning | $N$ tokens/frame ✗ | EnCodec, SoundStream, DAC |
| FSQ | None (grid) | None for codebook | None by design | 1 token/frame ✓ | WavTokenizer, CosyVoice 2 |
| LFQ | None (binary) | None for codebook | None by design | 1 token/frame ✓ | MAGVIT-v2, experimental audio |

## Landmark Speech Tokenizers

### SoundStream (Google, 2021)

The first end-to-end neural audio codec to hit perceptually competitive quality at low bitrates. Architecture:

- **Encoder**: Strided 1D convolutions downsample waveform → 75 Hz latent.
- **Quantizer**: RVQ with up to 8 levels of 1024 codes.
- **Decoder**: Mirror of encoder with transposed convolutions.
- **Discriminator**: Multi-scale STFT + waveform discriminators (adversarial training for realism).

Losses: time-domain L1 + multi-scale mel L1 + adversarial + feature matching + VQ commitment. Trained on speech, music, and general audio — a **general-purpose** codec.

### EnCodec (Meta, 2022)

Improved SoundStream with:
- LSTM layers in the encoder/decoder for longer context.
- Better discriminator design.
- **Bandwidth scalability**: train with random codebook dropout so you can decode with any subset of codebooks at inference. One model serves 1.5, 3, 6, 12, and 24 kbps.
- Optional language-model entropy coding on top of the discrete tokens (further 25-40% bitrate reduction at the cost of sequential decode).

EnCodec is the de facto research baseline for acoustic tokenization in 2023-2025. MusicGen and many VALL-E reproductions use it directly.

### DAC — Descript Audio Codec (Descript, 2023)

Pushed fidelity further at lower bitrates. Key changes:
- **Snake activation** (from BigVGAN) for better periodic modeling.
- **Factorized codes**: project to low dim before VQ (12D codebook input), better utilization.
- **L2-normalized code vectors** and encoder outputs (cosine similarity lookup) — stabilizes training.
- Aggressive discriminator schedule.

At 8 kbps on speech, DAC is near-transparent. The factorized-low-dim-+-cosine trick is now copied almost everywhere.

### HuBERT Tokens (Meta, 2021)

Not a codec — a **semantic tokenizer**. Take a pretrained HuBERT model, extract hidden states from a middle layer (layer 6-9 depending on variant), run **k-means** with $K = 100 \ldots 2000$ on a sample of hidden states, then use cluster IDs as tokens.

Frame rate: 50 Hz. No decoder — these tokens aren't for reconstruction. They're for content-focused modeling (AudioLM uses them as the first stream).

**Why a middle layer?** Early layers are acoustic; final layers specialize for the self-supervised pretext task. Middle layers have the strongest phonetic structure — empirically validated by phone purity / WER probing.

### WavTokenizer (2024)

A recent single-codebook acoustic tokenizer aimed at LLM-TTS. Key properties:
- **One codebook, ~4096 codes, ~75 Hz frame rate** — a single token per frame, ideal for LLM prediction.
- Uses FSQ or improved VQ with large code dim.
- Competitive reconstruction with EnCodec at similar bitrates despite the single-codebook constraint.

The motivation: multi-codebook RVQ complicates LLM design. One-codebook tokenizers let you drop in any Transformer and treat speech like text.

### CosyVoice S³ Tokenizer (Alibaba, 2024)

A **supervised semantic** tokenizer specifically for TTS. Instead of unsupervised k-means on HuBERT, S³ is trained with an ASR-style objective:

- Encoder (Whisper-style) → FSQ → decoder that must predict text tokens.
- The FSQ bottleneck forces the encoder to compress audio into a content-rich discrete sequence; the ASR objective ensures the tokens preserve transcription information.
- Frame rate: 25 Hz (40 ms per token) — very short sequences, great for LLM modeling.

CosyVoice 2 refines this with FSQ + additional prosody disentanglement. The TTS LLM then predicts S³ tokens from text; a second-stage flow-matching model converts S³ tokens + speaker embedding → mel spectrogram → waveform.

## Training a Speech Tokenizer

### Architecture

A canonical neural audio codec:

```python
class NeuralAudioCodec(nn.Module):
    def __init__(self, sample_rate=24000, frame_rate=75, num_codebooks=8, codebook_size=1024):
        super().__init__()
        # Encoder: downsample 24000 Hz → 75 Hz (factor 320)
        # Typical strides: 2, 4, 5, 8 → product 320
        self.encoder = ConvEncoder(
            channels=[32, 64, 128, 256, 512],
            strides=[2, 4, 5, 8],
            activation="snake",
        )
        self.rvq = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            code_dim=512,
        )
        # Decoder: upsample 75 Hz → 24000 Hz
        self.decoder = ConvDecoder(
            channels=[512, 256, 128, 64, 32],
            strides=[8, 5, 4, 2],
            activation="snake",
        )

    def forward(self, wav):
        z = self.encoder(wav)                        # (B, 512, T=75*seconds)
        z_q, indices, vq_loss = self.rvq(z)          # z_q: same shape, indices: (B, N, T)
        recon = self.decoder(z_q)                    # (B, 1, samples)
        return recon, indices, vq_loss
```

### The Loss Stack

Reconstruction alone (L1/L2 in the time domain) gives muddy, over-smoothed audio — the classic "neural vocoder" artifact. Modern codecs combine **five** losses:

1. **Time-domain**: $L_1(\hat{x}, x)$. Cheap sanity check.
2. **Multi-scale mel L1**: $\sum_s \| \text{Mel}_s(\hat{x}) - \text{Mel}_s(x) \|_1$ over STFT window sizes 32, 64, 128, 256, 512, 1024, 2048. Captures both transients and steady tones.
3. **Adversarial loss** (hinge GAN): multiple discriminators — multi-period (MPD), multi-scale (MSD), multi-scale STFT (MS-STFT). Produces sharp, natural-sounding audio.
4. **Feature matching**: $\sum_\ell \| D^{(\ell)}(\hat{x}) - D^{(\ell)}(x) \|_1$ across discriminator layers. Stabilizes the adversarial game.
5. **VQ commitment / codebook**: from the quantizer. Typically weighted at 0.25-1.0.

Total: $\mathcal{L} = \lambda_t \mathcal{L}_{\text{time}} + \lambda_m \mathcal{L}_{\text{mel}} + \lambda_a \mathcal{L}_{\text{adv}} + \lambda_f \mathcal{L}_{\text{feat}} + \lambda_q \mathcal{L}_{\text{vq}}$.

Typical weights: $\lambda_t = 0.1, \lambda_m = 15, \lambda_a = 3, \lambda_f = 3, \lambda_q = 1$ (EnCodec-ish).

### Training Tips That Actually Matter

- **Warm up without adversarial loss** for the first 5-10K steps. GANs destabilize early training before the generator can produce anything remotely structured.
- **Normalize audio** to peak = -1 dB before training. Clipping or wildly varying loudness ruins L1 losses.
- **Balance speakers/domains**. A codec trained on LibriSpeech only will crumble on noisy real-world audio. Mix LibriLight, Common Voice, DNS challenge noise, and music if you want general-purpose.
- **Codebook utilization monitoring**: track the fraction of codes used per batch. Healthy = >90%. If <50%, something is wrong (learning rate, EMA decay, commitment weight).
- **Gradient clipping at 1.0**. RVQ training can spike gradients when codes switch.
- **Discriminator update ratio**: 1:1 with generator is standard; some papers use 2:1. Too fast a discriminator flattens the generator loss.
- **Bitrate dropout** (EnCodec trick): during training, randomly use $n \leq N$ codebooks with $n$ sampled uniformly. Model learns to handle arbitrary bitrates in one checkpoint.

### Data and Compute

A usable neural audio codec needs:
- **1,000-10,000 hours** of clean-ish speech (and possibly music/noise).
- Training: roughly 500K-2M steps at batch size 32-128, on 8× A100 for 3-10 days.

This is substantially cheaper than training a TTS LLM, which is why codecs are often shared across many downstream systems — train once, reuse everywhere.

## Evaluating Speech Tokenizers

A speech tokenizer has two jobs: **reconstruct** well and be **modelable** by a downstream LLM. Evaluation covers both.

### Reconstruction Quality

| Metric | What it measures | Range / direction |
|--------|------------------|-------------------|
| PESQ | Perceptual speech quality (MOS-predicted) | 1.0-4.5, higher better |
| STOI | Short-time intelligibility | 0-1, higher better |
| ViSQOL | Virtual speech quality objective listener | 1-5, higher better |
| Mel distance | L1 between mel spectrograms of original/recon | Lower better |
| SI-SDR | Scale-invariant signal-to-distortion ratio (dB) | Higher better |
| MOS (human) | Subjective listening test (gold standard) | 1-5, higher better |

Run these on held-out test data. PESQ and STOI are the quick day-to-day metrics; MOS is the final judge before shipping.

### Content Preservation

If tokens are for TTS/ASR, the critical question is: does the token sequence still contain the linguistic content?

- **ASR-WER on reconstructed audio**: run an off-the-shelf ASR (Whisper) on the original and the reconstruction. Compute WER delta. <1% absolute increase = excellent; >3% means content is being lost.
- **Phone purity** (for semantic tokens): how well tokens cluster by phoneme label. Measures content-focus.
- **Speaker leakage** (for semantic tokens): classify speaker ID from tokens alone. For a clean semantic tokenizer, speaker ID accuracy should be close to chance — otherwise the LLM will overfit to speaker in unintended ways.

### Modelability

You can have beautiful reconstruction and still train a bad LLM on top. Check:

- **Token entropy**: average bits per token should be close to $\log_2 K$ if usage is uniform. Low entropy = codebook collapse.
- **Token sequence perplexity** under a simple n-gram or small Transformer. Low perplexity = predictable = easier to model.
- **Downstream WER/MOS** after training a TTS system on the tokens. The only evaluation that really matters.

### Bitrate and Latency

- **Bitrate** = frame_rate × bits_per_frame. EnCodec at 6 kbps: 75 × 80 bits.
- **Frame rate** determines LLM sequence length. 25 Hz (CosyVoice S³) is 3× friendlier than 75 Hz (EnCodec) for long-form generation.
- **Encoder latency**: for streaming, the encoder must be **causal** — no future-frame dependence. Chunked / lookahead-constrained variants exist for all major codecs.

## Design Decisions: What to Pick for What

### Building a TTS system from scratch
- **Semantic stage**: HuBERT k-means (public, easy) or train an S³-style supervised semantic tokenizer (more control, needs ASR data).
- **Acoustic stage**: EnCodec (battle-tested) or WavTokenizer (single-codebook, simpler LLM).

### Zero-shot voice cloning
- You need the semantic/acoustic split. Semantic tokens from 3-sec reference are useless; **speaker embedding** (ECAPA-TDNN or similar) captures timbre, semantic tokens capture content, acoustic stage fuses them.

### Streaming TTS
- Frame rate matters. 25 Hz tokens + chunked causal encoder + chunk-aware flow-matching decoder (CosyVoice 2 pattern).
- Single-codebook tokenizers win over RVQ: no $N$-head prediction to serialize.

### On-device speech
- Bitrate budget often forces <3 kbps. DAC or heavily-tuned EnCodec.
- Quantize encoder/decoder to int8. Replace large conv stacks with depthwise-separable.

### Music or general audio
- DAC > EnCodec > SoundStream in 2025 for music fidelity. All need 16+ kbps for instrument separation to survive.

## Common Pitfalls

- **Codebook collapse**: 80% of codes unused. Fix: EMA updates, commitment weight, random restarts, or switch to FSQ.
- **Over-smoothed audio**: only time-domain L1 loss. Fix: add mel L1 + adversarial + feature matching.
- **Content loss**: perfect PESQ but ASR WER triples after reconstruction. Likely a bottleneck-too-narrow problem (too few codebooks or too aggressive downsampling). Also check that training data covers your target domain.
- **Speaker leakage in semantic tokens**: semantic tokenizer memorizes speaker. Fix: speaker perturbation augmentation (pitch shift, formant shift) during tokenizer training, or an explicit adversarial speaker-unlearning head.
- **Streaming artifacts**: clicks/pops at chunk boundaries when moving from offline to streaming. Fix: overlap-add at decoder, causal-convolution-only architecture, retrain with streaming constraints (don't just chop the offline model).
- **Bitrate mismatch**: training at 6 kbps then decoding a sequence the LLM produced at 1.5 kbps → muddy output. Use EnCodec-style bitrate dropout so one model covers all targets.

## Interview-Grade Summary

- A speech tokenizer converts continuous audio into discrete tokens so a Transformer can model speech like text.
- Two families: **acoustic** (reconstruction-first, encodes everything) and **semantic** (content-first, speaker-invariant). Modern systems often use both.
- The quantizer is the core: **VQ**, **RVQ** (dominant for acoustic), **FSQ** (simpler, winning share), **LFQ**.
- Landmark systems: SoundStream → EnCodec → DAC for acoustic; HuBERT k-means → CosyVoice S³ for semantic; WavTokenizer for single-codebook LLM-friendly acoustic.
- Losses combine time-domain + multi-scale mel + adversarial + feature matching + VQ commitment.
- Evaluation: PESQ/STOI/MOS for reconstruction, ASR-WER for content preservation, codebook utilization for quantizer health, downstream WER/MOS for modelability.
- Design drivers: single-codebook vs RVQ (LLM sequence length), frame rate (streaming latency), semantic vs acoustic (voice cloning architecture), bitrate budget (on-device vs server).

## References

1. van den Oord, A., et al. "Neural Discrete Representation Learning (VQ-VAE)." NeurIPS 2017.
2. Razavi, A., et al. "Generating Diverse High-Fidelity Images with VQ-VAE-2." NeurIPS 2019.
3. Zeghidour, N., et al. "SoundStream: An End-to-End Neural Audio Codec." IEEE/ACM TASLP 2021.
4. Défossez, A., et al. "High Fidelity Neural Audio Compression (EnCodec)." TMLR 2023.
5. Kumar, R., et al. "High-Fidelity Audio Compression with Improved RVQGAN (DAC)." NeurIPS 2023.
6. Mentzer, F., et al. "Finite Scalar Quantization: VQ-VAE Made Simple." ICLR 2024.
7. Yu, L., et al. "Language Model Beats Diffusion — Tokenizer is Key to Visual Generation (MAGVIT-v2, LFQ)." ICLR 2024.
8. Hsu, W.-N., et al. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." IEEE/ACM TASLP 2021.
9. Borsos, Z., et al. "AudioLM: A Language Modeling Approach to Audio Generation." IEEE/ACM TASLP 2023.
10. Wang, C., et al. "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)." 2023.
11. Du, Z., et al. "CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens." 2024.
12. Du, Z., et al. "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models." 2024.
13. Ji, S., et al. "WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling." 2024.
