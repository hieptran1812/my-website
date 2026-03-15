---
title: "HiFi-GAN: High-Fidelity Neural Vocoder for Real-Time Speech Synthesis"
publishDate: "2026-03-15"
category: "machine-learning"
subcategory: "Signal Processing"
tags:
  [
    "HiFi-GAN",
    "vocoder",
    "speech-synthesis",
    "GAN",
    "signal-processing",
    "deep-learning",
    "audio-generation",
    "TTS",
  ]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/hifi-gan-20260315180007.png"
aiGenerated: true
excerpt: "A comprehensive guide to HiFi-GAN — a generative adversarial network that synthesizes high-fidelity audio from mel-spectrograms in real-time. We explore its architecture, training strategy, code implementation, and real-world applications."
---

## Introduction

Imagine you have a text-to-speech (TTS) system that converts text into a mel-spectrogram — a compact representation of audio. The final step is converting that mel-spectrogram back into an actual waveform you can hear. This step is called **vocoding**, and it's one of the hardest parts of the speech synthesis pipeline.

Traditional vocoders like Griffin-Lim produce robotic-sounding audio. Autoregressive models like WaveNet produce amazing quality but are painfully slow — generating one audio sample at a time. WaveGlow is faster but requires a huge model with hundreds of millions of parameters.

**HiFi-GAN** (High Fidelity Generative Adversarial Network), introduced by Kong et al. in 2020, changed the game. It generates audio that is:

- **High quality** — matching or exceeding autoregressive models
- **Fast** — up to 167x faster than real-time on a single GPU
- **Lightweight** — the smallest variant has only ~0.92M parameters

The paper: [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)

## Background: What Problem Does HiFi-GAN Solve?

### The Mel-Spectrogram Gap

A mel-spectrogram is a 2D representation of audio where:

- The x-axis represents **time**
- The y-axis represents **frequency** (on the mel scale, which mimics human hearing)
- The intensity represents **energy/amplitude**

A mel-spectrogram typically has 80 frequency bins and captures audio at ~256 samples per frame. To reconstruct 22,050 Hz audio, the vocoder must **upsample** from ~86 frames/second to 22,050 samples/second — a **256x expansion**.

This is hard because:

**1. The mel-spectrogram discards phase information — you only have magnitude**

Sound is a wave, and any wave has two components: **amplitude** (how loud) and **phase** (where in the cycle the wave is at any given moment). A mel-spectrogram only keeps the amplitude information — it tells you *how much energy* exists at each frequency, but not *when* each frequency cycle starts or how different frequencies align with each other.

Why does this matter? Consider two sounds that have identical mel-spectrograms but completely different phases. They would sound noticeably different to your ears. When a vocoder reconstructs audio from a mel-spectrogram, it must **invent** plausible phase information from scratch. There are infinitely many valid phase configurations for a given magnitude spectrum, and most of them sound terrible — producing metallic, hollow, or buzzy artifacts. This is why simple methods like the Griffin-Lim algorithm (which iteratively estimates phase) produce robotic-sounding audio. The vocoder essentially needs to solve an ill-posed inverse problem: recovering a complete signal from an incomplete representation.

**2. Small errors compound — a tiny artifact in one sample creates audible distortion**

Audio operates at an extremely fine granularity. At 22,050 Hz sampling rate, the vocoder generates **22,050 individual amplitude values per second**. Each sample is spaced just ~45 microseconds apart. A single misplaced sample — even by a small amount — can create an audible "click" or "pop". Worse, because audio is a continuous signal, errors don't stay isolated:

- A spike in one sample affects the perceived waveform shape of neighboring samples
- The human ear is incredibly sensitive to discontinuities — we can detect distortions as short as 1-2 milliseconds
- Errors in the high-frequency components (consonants like "s", "t", "f") are especially noticeable because these sounds have sharp, precise temporal structures

This is fundamentally different from image generation, where a single wrong pixel is virtually invisible. In audio, a single wrong sample at the wrong moment can ruin the perception of an entire syllable.

**3. Speech has patterns at multiple time scales simultaneously**

Speech is a hierarchical signal with structures spanning several orders of magnitude in time:

- **Pitch cycles (~2-16ms)**: The vocal cords vibrate at a fundamental frequency (typically 85-300 Hz). Each vibration cycle creates a single "pulse" in the waveform. The vocoder must reproduce these micro-level oscillations precisely, or the voice sounds hoarse or unnatural.
- **Phonemes (~30-100ms)**: Individual speech sounds like vowels ("ah", "ee") and consonants ("t", "s") each have distinctive spectral patterns. The transition between phonemes — called **coarticulation** — involves smooth spectral changes that must be captured faithfully.
- **Syllables and words (~150-500ms)**: Prosody (the rhythm and intonation of speech) operates at this scale. Stress patterns, rising/falling pitch contours, and speaking rate all encode meaning. A question sounds different from a statement primarily because of patterns at this timescale.
- **Phrases and sentences (~1-5s)**: Long-range dependencies like breath pauses, emphasis patterns, and emotional tone span entire utterances.

The vocoder must capture **all of these scales simultaneously** in a single waveform output. A model with a small receptive field might nail the pitch cycles but miss the prosody. A model with only a large receptive field might capture prosody but produce blurry, indistinct phonemes. This multi-scale challenge is exactly what HiFi-GAN's architecture — with its Multi-Receptive Field Fusion in the generator and Multi-Period/Multi-Scale discriminators — is designed to address.

### Why GANs?

GANs work by training two networks against each other:

- **Generator**: Creates audio from mel-spectrograms
- **Discriminator**: Tries to distinguish generated audio from real audio

The adversarial training pushes the generator to produce increasingly realistic audio. HiFi-GAN's key insight is that **audio patterns exist at multiple scales**, so the discriminators should also operate at multiple scales and periods.

## Architecture Deep Dive

HiFi-GAN has three main components:

### 1. Generator

The generator takes a mel-spectrogram and upsamples it to a raw waveform through a series of **transposed convolutions** and **residual blocks**.

```
Mel-spectrogram (80 x T)
         │
    [Conv1d]  ──  Initial projection
         │
    [Upsample Block 1]  ──  8x upsample
         │
    [Upsample Block 2]  ──  8x upsample
         │
    [Upsample Block 3]  ──  2x upsample
         │
    [Upsample Block 4]  ──  2x upsample
         │
    [Conv1d + Tanh]  ──  Final projection to waveform
         │
  Waveform (1 x T*256)
```

Each upsample block consists of:

- A **transposed convolution** that increases the temporal resolution
- A **Multi-Receptive Field Fusion (MRF)** module

#### Multi-Receptive Field Fusion (MRF)

This is the core innovation. Instead of using a single receptive field, MRF uses **multiple residual blocks with different kernel sizes and dilation rates**, then sums their outputs.

Why? Because audio has patterns at different time scales:

- **Small kernels** (size 3) capture fine details like consonant bursts
- **Medium kernels** (size 7) capture pitch period patterns
- **Large kernels** (size 11) capture broader phonetic structures

Each residual block uses **dilated convolutions** to further expand the receptive field without increasing parameters:

```
ResBlock with kernel_size=3, dilations=[1, 3, 5]:

Input ──> Conv1d(k=3, d=1) ──> LeakyReLU ──> Conv1d(k=3, d=1) ──> + ──> Output
  │                                                                 │
  └─────────────────────── (skip connection) ──────────────────────┘

Input ──> Conv1d(k=3, d=3) ──> LeakyReLU ──> Conv1d(k=3, d=1) ──> + ──> Output
  │                                                                 │
  └─────────────────────── (skip connection) ──────────────────────┘

Input ──> Conv1d(k=3, d=5) ──> LeakyReLU ──> Conv1d(k=3, d=1) ──> + ──> Output
  │                                                                 │
  └─────────────────────── (skip connection) ──────────────────────┘
```

The dilation factor `d` means the convolution "skips" every `d-1` samples, allowing the network to see a wider range without extra computation.

### 2. Multi-Period Discriminator (MPD)

The MPD is designed to capture **periodic patterns** in audio — which makes sense because speech is quasi-periodic (pitch cycles repeat regularly).

The trick: reshape the 1D audio into 2D by folding it at different periods.

For example, with period p=2:

```
Original:  [s0, s1, s2, s3, s4, s5, s6, s7]

Reshaped (p=2):
  [s0, s2, s4, s6]
  [s1, s3, s5, s7]
```

HiFi-GAN uses **5 sub-discriminators** with periods [2, 3, 5, 7, 11] (prime numbers to avoid overlapping patterns). Each sub-discriminator applies 2D convolutions on the reshaped audio, effectively looking for periodic structures at different fundamental frequencies.

### 3. Multi-Scale Discriminator (MSD)

While MPD captures periodic patterns, MSD captures **non-periodic patterns** and overall audio structure at different time scales.

MSD uses **3 sub-discriminators** operating on:

1. Raw audio (original scale)
2. 2x downsampled audio (average pooling)
3. 4x downsampled audio (average pooling)

Each sub-discriminator is a stack of **grouped 1D convolutions** with increasing dilation. The downsampled versions help the discriminator evaluate long-range structure and overall audio quality.

## Training Strategy

### Loss Functions

HiFi-GAN uses a combination of three losses:

#### 1. Adversarial Loss (GAN Loss)

The standard GAN objective, but using **least squares GAN** instead of the original cross-entropy formulation for more stable training:

$$L_{adv}(G) = \mathbb{E}\left[(D(G(s)) - 1)^2\right]$$

$$L_{adv}(D) = \mathbb{E}\left[(D(x) - 1)^2 + D(G(s))^2\right]$$

where $x$ is real audio, $s$ is the mel-spectrogram, and $G(s)$ is the generated audio.

This loss is computed for **each sub-discriminator** in both MPD and MSD (8 discriminators total).

#### 2. Mel-Spectrogram Loss

This reconstruction loss ensures the generated audio matches the input mel-spectrogram:

$$L_{mel} = \mathbb{E}\left[\|M(x) - M(G(s))\|_1\right]$$

where $M(\cdot)$ extracts the mel-spectrogram. This is crucial for:

- **Stabilizing early training** — gives a strong gradient signal before the discriminator is effective
- **Ensuring spectral accuracy** — the GAN loss alone might produce realistic-sounding but spectrally inaccurate audio

#### 3. Feature Matching Loss

For each discriminator layer $l$, compare intermediate features between real and generated audio:

$$L_{fm} = \mathbb{E}\left[\sum_{l=1}^{L} \frac{1}{N_l}\|D^l(x) - D^l(G(s))\|_1\right]$$

This helps the generator learn to match internal representations, not just fool the final output. Think of it as teaching the generator to match the "texture" of real audio at multiple levels of abstraction.

#### Final Combined Loss

$$L_G = \sum_{k=1}^{K} L_{adv}(G; D_k) + \lambda_{fm} \sum_{k=1}^{K} L_{fm}(G; D_k) + \lambda_{mel} L_{mel}(G)$$

The paper uses $\lambda_{fm} = 2$ and $\lambda_{mel} = 45$.

## Implementation

Let's implement HiFi-GAN step by step in PyTorch.

### Generator

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels,
                        kernel_size, stride=1,
                        dilation=d,
                        padding=self._get_padding(kernel_size, d),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels,
                        kernel_size, stride=1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    )
                )
            )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def _get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x  # Skip connection
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Module):
    """HiFi-GAN Generator.

    Args:
        in_channels: Number of mel-spectrogram channels (typically 80)
        upsample_initial_channel: Initial hidden channels (512 for V1)
        upsample_rates: Upsampling factor for each block [8, 8, 2, 2]
        upsample_kernel_sizes: Kernel size for each transposed conv [16, 16, 4, 4]
        resblock_kernel_sizes: Kernel sizes for MRF blocks [3, 7, 11]
        resblock_dilation_sizes: Dilation rates for each kernel [[1,3,5], [1,3,5], [1,3,5]]
    """

    def __init__(
        self,
        in_channels=80,
        upsample_initial_channel=512,
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Initial convolution
        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        )

        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2,
                        k, stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )
            ch = ch // 2

        # Multi-Receptive Field Fusion (MRF) blocks
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch_i = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch_i, k, d))

        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(ch_i, 1, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: Mel-spectrogram tensor of shape (B, 80, T)
        Returns:
            Waveform tensor of shape (B, 1, T * prod(upsample_rates))
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # MRF: sum outputs from all residual blocks
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
```

### Multi-Period Discriminator

```python
class PeriodDiscriminator(nn.Module):
    """Sub-discriminator that reshapes audio by a given period."""

    def __init__(self, period):
        super().__init__()
        self.period = period

        # Stack of 2D convolutions
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x: Audio tensor of shape (B, 1, T)
        Returns:
            output: Discriminator score
            fmap: List of intermediate feature maps (for feature matching loss)
        """
        fmap = []
        b, c, t = x.shape

        # Pad to make length divisible by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        # Reshape: (B, 1, T) -> (B, 1, T/p, p)
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Discriminator that evaluates audio at multiple periods."""

    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(self, y, y_hat):
        """
        Args:
            y: Real audio (B, 1, T)
            y_hat: Generated audio (B, 1, T)
        Returns:
            Lists of discriminator outputs and feature maps for real and generated
        """
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
```

### Multi-Scale Discriminator

```python
class ScaleDiscriminator(nn.Module):
    """Sub-discriminator operating at a single scale."""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = nn.utils.spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Discriminator that evaluates audio at multiple time scales."""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),  # Raw audio
            ScaleDiscriminator(),                         # 2x downsampled
            ScaleDiscriminator(),                         # 4x downsampled
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
```

### Loss Functions

```python
def feature_matching_loss(fmap_r, fmap_g):
    """Feature matching loss across all discriminator layers."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """LSGAN discriminator loss."""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr - 1) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
    return loss


def generator_loss(disc_outputs):
    """LSGAN generator loss."""
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((dg - 1) ** 2)
    return loss
```

### Training Loop

```python
import torchaudio
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR


def mel_spectrogram(y, n_fft=1024, hop_size=256, win_size=1024, n_mels=80,
                    fmin=0, fmax=8000, sample_rate=22050):
    """Compute mel-spectrogram from waveform."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_size,
        win_length=win_size, n_mels=n_mels, f_min=fmin, f_max=fmax,
        power=1, norm="slaney", mel_scale="slaney",
    ).to(y.device)
    mel = mel_transform(y)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def train_step(generator, mpd, msd, optim_g, optim_d,
               real_audio, mel, lambda_fm=2, lambda_mel=45):
    """Single training step for HiFi-GAN.

    Args:
        real_audio: Ground truth waveform (B, 1, T)
        mel: Input mel-spectrogram (B, 80, T_mel)
    """
    # ---------------------
    # Train Discriminators
    # ---------------------
    optim_d.zero_grad()

    with torch.no_grad():
        y_hat = generator(mel)

    # Multi-Period Discriminator
    y_dr_mpd, y_dg_mpd, _, _ = mpd(real_audio, y_hat.detach())
    loss_d_mpd = discriminator_loss(y_dr_mpd, y_dg_mpd)

    # Multi-Scale Discriminator
    y_dr_msd, y_dg_msd, _, _ = msd(real_audio, y_hat.detach())
    loss_d_msd = discriminator_loss(y_dr_msd, y_dg_msd)

    loss_d = loss_d_mpd + loss_d_msd
    loss_d.backward()
    optim_d.step()

    # -----------------
    # Train Generator
    # -----------------
    optim_g.zero_grad()

    y_hat = generator(mel)

    # Mel reconstruction loss
    mel_hat = mel_spectrogram(y_hat.squeeze(1))
    mel_real = mel_spectrogram(real_audio.squeeze(1))
    # Trim to match lengths
    min_len = min(mel_hat.shape[-1], mel_real.shape[-1])
    loss_mel = F.l1_loss(mel_hat[..., :min_len], mel_real[..., :min_len])

    # Discriminator outputs for generator loss
    y_dr_mpd, y_dg_mpd, fmap_r_mpd, fmap_g_mpd = mpd(real_audio, y_hat)
    y_dr_msd, y_dg_msd, fmap_r_msd, fmap_g_msd = msd(real_audio, y_hat)

    # Feature matching losses
    loss_fm_mpd = feature_matching_loss(fmap_r_mpd, fmap_g_mpd)
    loss_fm_msd = feature_matching_loss(fmap_r_msd, fmap_g_msd)

    # Generator adversarial losses
    loss_g_mpd = generator_loss(y_dg_mpd)
    loss_g_msd = generator_loss(y_dg_msd)

    # Combined generator loss
    loss_g = (
        loss_g_mpd + loss_g_msd
        + lambda_fm * (loss_fm_mpd + loss_fm_msd)
        + lambda_mel * loss_mel
    )
    loss_g.backward()
    optim_g.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g": loss_g.item(),
        "loss_mel": loss_mel.item(),
    }


# ---- Example usage ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    optim_g = AdamW(generator.parameters(), lr=2e-4, betas=(0.8, 0.99))
    optim_d = AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=2e-4, betas=(0.8, 0.99),
    )
    scheduler_g = ExponentialLR(optim_g, gamma=0.999)
    scheduler_d = ExponentialLR(optim_d, gamma=0.999)

    # Training loop (simplified)
    num_epochs = 3000
    for epoch in range(num_epochs):
        # In practice, iterate over your DataLoader here
        # real_audio: (B, 1, T) waveform
        # mel: (B, 80, T//256) mel-spectrogram

        # Dummy data for illustration
        real_audio = torch.randn(4, 1, 8192).to(device)
        mel = mel_spectrogram(real_audio.squeeze(1)).to(device)

        losses = train_step(
            generator, mpd, msd, optim_g, optim_d,
            real_audio, mel,
        )

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D={losses['loss_d']:.4f}, "
                  f"G={losses['loss_g']:.4f}, Mel={losses['loss_mel']:.4f}")

        scheduler_g.step()
        scheduler_d.step()
```

## Model Variants

HiFi-GAN comes in three configurations, trading off quality for speed:

| Variant | Hidden Channels | Upsample Rates | Parameters | Speed (RTF) | MOS  |
| ------- | --------------- | -------------- | ---------- | ----------- | ---- |
| **V1**  | 512             | [8, 8, 2, 2]   | 13.92M     | 167.9x      | 4.36 |
| **V2**  | 128             | [8, 8, 2, 2]   | 0.92M      | 1186.8x     | 4.23 |
| **V3**  | 256             | [8, 8, 4]      | 1.46M      | 1049.3x     | 4.05 |

- **V1**: Best quality, suitable for offline rendering or powerful GPUs
- **V2**: Fastest, great for real-time on edge devices
- **V3**: Balanced, uses 3 upsample blocks instead of 4

MOS (Mean Opinion Score) is rated 1-5, with 5 being perfect. Ground truth audio scored 4.45, so V1's 4.36 is remarkably close.

## Using Pre-trained HiFi-GAN

You can use HiFi-GAN directly with popular TTS libraries:

### With the Official Repository

```python
import json
import torch
from env import AttrDict
from models import Generator

# Load config
with open("config_v1.json") as f:
    config = json.load(f)
h = AttrDict(config)

# Load generator
generator = Generator(h).cuda()
state_dict = torch.load("generator_v1", map_location="cuda")
generator.load_state_dict(state_dict["generator"])
generator.eval()
generator.remove_weight_norm()

# Inference
mel = torch.randn(1, 80, 100).cuda()  # Your mel-spectrogram
with torch.no_grad():
    audio = generator(mel)
    audio = audio.squeeze().cpu().numpy()
```

### With SpeechBrain

```python
import torchaudio
from speechbrain.pretrained import HIFIGAN

hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/tts-hifigan-ljspeech",
)

# Generate waveform from mel-spectrogram
mel = torch.randn(1, 80, 100)  # Replace with real mel
waveform = hifi_gan.decode_batch(mel)
torchaudio.save("output.wav", waveform.squeeze(1), 22050)
```

### With ESPnet

```python
from espnet2.bin.tts_inference import Text2Speech

# ESPnet bundles HiFi-GAN as the default vocoder
tts = Text2Speech.from_pretrained(
    model_tag="kan-bayashi/ljspeech_vits",
    vocoder_tag="none",  # VITS has built-in vocoder
)

# Or use standalone HiFi-GAN vocoder
tts = Text2Speech.from_pretrained(
    model_tag="kan-bayashi/ljspeech_tacotron2",
    vocoder_tag="parallel_wavegan/ljspeech_hifigan.v1",
)

speech = tts("Hello, this is a test of HiFi-GAN vocoding.")
```

## Applications

### 1. Text-to-Speech (TTS) Systems

HiFi-GAN is the most common vocoder in modern TTS pipelines:

```
Text → Acoustic Model (Tacotron2/FastSpeech2) → Mel-Spectrogram → HiFi-GAN → Waveform
```

It is used as the default vocoder in systems like:

- **VITS** (combines acoustic model + HiFi-GAN in a single end-to-end model)
- **Coqui TTS** (open-source TTS toolkit)
- **ESPnet** and **SpeechBrain** TTS modules

### 2. Voice Conversion

Convert one speaker's voice to sound like another while preserving the linguistic content:

```
Source Audio → Extract Mel → Modify Speaker Embedding → HiFi-GAN → Target Voice Audio
```

HiFi-GAN's ability to generate high-quality audio from modified mel-spectrograms makes it ideal for voice conversion tasks.

### 3. Speech Enhancement and Denoising

Use HiFi-GAN to reconstruct clean audio from noisy mel-spectrograms:

```
Noisy Audio → Mel-Spectrogram → Denoising Model → Clean Mel → HiFi-GAN → Clean Audio
```

### 4. Music Generation

While originally designed for speech, HiFi-GAN has been adapted for music synthesis:

- **JukeBox** variants use modified HiFi-GAN for music generation
- **MusicGen** and related systems use similar GAN-based vocoders
- Singing voice synthesis (SVS) systems frequently use HiFi-GAN

### 5. Audio Super-Resolution

Upsample low-quality audio (e.g., 8kHz phone audio) to high-quality (22kHz+):

```
Low-Res Audio → Low-Res Mel → Bandwidth Extension Model → Full Mel → HiFi-GAN → Hi-Res Audio
```

### 6. Real-Time Applications

Thanks to V2's tiny size (~0.92M parameters) and 1186x real-time speed:

- **Mobile TTS**: On-device speech synthesis for virtual assistants
- **Game engines**: Real-time NPC dialogue generation
- **Accessibility**: Screen readers with natural-sounding voices
- **Telecommunications**: Real-time voice modification and enhancement

## Key Takeaways

1. **Multi-scale thinking**: Audio has structure at many time scales. The MRF module in the generator and the multi-period/multi-scale discriminators each address this from different angles.

2. **Periodic inductive bias**: The Multi-Period Discriminator exploits the quasi-periodic nature of speech — a domain-specific insight that significantly boosts quality.

3. **Loss function cocktail**: The combination of adversarial, feature matching, and mel-spectrogram losses provides complementary training signals. No single loss would work as well alone.

4. **Efficiency through design**: V2 achieves near-V1 quality with 15x fewer parameters by simply reducing the hidden channel dimension — proving that the architecture design (not just scale) is what matters.

5. **Universal vocoder potential**: HiFi-GAN generalizes well to unseen speakers and even other domains (music, environmental sounds) with minimal fine-tuning.

## References

- Kong, J., Kim, J., & Bae, J. (2020). [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646). NeurIPS 2020.
- [Official HiFi-GAN Repository](https://github.com/jik876/hifi-gan)
- [SpeechBrain HiFi-GAN](https://huggingface.co/speechbrain/tts-hifigan-ljspeech)
- Kim, J., Kong, J., & Son, J. (2021). [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)](https://arxiv.org/abs/2106.06103)
