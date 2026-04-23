---
title: "Training CosyVoice: A Complete Guide to LLM-Based Text-to-Speech"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "cosyvoice",
    "tts",
    "text-to-speech",
    "speech-synthesis",
    "flow-matching",
    "deep-learning",
    "voice-cloning",
    "audio",
    "llm",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A comprehensive guide to training CosyVoice and CosyVoice 2 — Alibaba's LLM-based TTS systems. Covers the full pipeline: speech tokenization (VQ/FSQ), autoregressive LLM for text-to-token, flow matching for token-to-mel, streaming via chunk-aware causal decoding, and interview-ready depth."
---

## What Is CosyVoice?

![CosyVoice 4-stage TTS pipeline: tokenizer -> AR LLM -> flow-matching decoder -> vocoder, with speaker-embedding path and training stages](/imgs/blogs/training-cosyvoice-diagram.png)

CosyVoice is an LLM-based text-to-speech (TTS) system developed by Alibaba's FunAudioLLM team. It generates natural, expressive speech from text with zero-shot voice cloning — give it a 3-second audio sample of any speaker and it can synthesize new speech in that person's voice.

The system follows a two-stage pipeline:

1. **Text → Speech Tokens**: An autoregressive LLM converts text into discrete speech tokens (like how a language model generates text tokens, but the tokens represent speech)
2. **Speech Tokens → Audio**: A conditional flow matching model converts those speech tokens into a mel spectrogram, and a vocoder produces the final waveform

```
                         CosyVoice Pipeline
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Text: "Hello, how are you?"                                │
│     │                                                       │
│     ▼                                                       │
│  ┌──────────────────────────┐                                │
│  │  BPE Tokenizer            │                                │
│  │  "Hello" → [15496, ...]   │                                │
│  └────────────┬─────────────┘                                │
│               │                                              │
│               ▼                                              │
│  ┌──────────────────────────┐   ┌────────────────────────┐  │
│  │  Autoregressive LLM       │   │  Speaker Embedding     │  │
│  │  (Qwen2.5-0.5B in v2)    │   │  (CAM++, 192-dim)      │  │
│  │                           │   └──────────┬─────────────┘  │
│  │  Generates speech tokens  │              │               │
│  │  at 25 Hz (25 per second) │              │               │
│  └────────────┬─────────────┘              │               │
│               │                             │               │
│               ▼                             ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Conditional Flow Matching Decoder                    │   │
│  │  Speech tokens + speaker embedding → mel spectrogram  │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────┐                                │
│  │  HiFT Vocoder             │                                │
│  │  Mel spectrogram → audio  │                                │
│  └────────────┬─────────────┘                                │
│               │                                              │
│               ▼                                              │
│  Audio: ~~~∿∿∿∿∿~~~∿∿∿∿∿~~~  (24 kHz waveform)            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

CosyVoice 2 (December 2024) introduced three key upgrades: **Finite Scalar Quantization (FSQ)** for better speech tokens, a **pre-trained Qwen2.5-0.5B** backbone replacing the custom-trained LLM, and **chunk-aware causal flow matching** enabling streaming synthesis with latency as low as 150ms.

## Stage 0: Speech Tokenizer — Converting Audio to Discrete Tokens

Before we can train the LLM, we need a way to represent speech as discrete tokens — the same way text is represented as BPE tokens for language models.

### Why Discrete Tokens?

LLMs operate on discrete token sequences. To use an LLM for speech, we need to convert continuous audio into a sequence of integers. The tokenizer must capture enough information for the downstream models to reconstruct intelligible, natural speech, while being compact enough for the LLM to model effectively.

### CosyVoice 1: Supervised Semantic Tokens with Vector Quantization (VQ)

CosyVoice v1 introduced **supervised semantic tokens (S³)** — a novel approach that uses an ASR model to derive speech tokens, rather than self-supervised models like HuBERT.

**Architecture**: Take a pretrained ASR encoder (SenseVoice-Large), split it into two halves (Encoder₁ and Encoder₂), and insert a **vector quantization (VQ)** bottleneck between them. The ASR decoder sits on top, so the quantized representations must preserve enough information for speech recognition — this forces the tokens to capture semantic content.

```
Audio → [Encoder₁ (6 layers)] → continuous features
                                    ↓
                              [Vector Quantization]
                              Codebook: 4,096 entries
                              Find nearest codebook vector
                                    ↓
                              Quantized features (discrete tokens)
                                    ↓
        [Encoder₂ (remaining layers)] → [ASR Decoder] → text
```

**How VQ works**:

Given a continuous feature vector $h_l$ at frame $l$, VQ finds the nearest entry in a learnable codebook $C = \{c_1, c_2, ..., c_N\}$:

$$\mu_l = \arg\min_{c_n \in C} \| h_l - c_n \|_2$$

The codebook is updated via Exponential Moving Average (EMA):

$$c_{\mu_l} \leftarrow \alpha \cdot c_{\mu_l} + (1 - \alpha) \cdot h_l$$

Gradients flow through the quantization step using the **straight-through estimator** — the forward pass uses the quantized vector, but the backward pass passes gradients as if quantization didn't happen.

**Problem**: Only 23% of the 4,096 codebook entries were actually used. Most codes were dead — never selected as the nearest neighbor. This wastes representational capacity.

### CosyVoice 2: Finite Scalar Quantization (FSQ)

CosyVoice 2 replaces VQ with **Finite Scalar Quantization (FSQ)** — a simpler, deterministic quantization scheme that achieves 100% codebook utilization.

**How FSQ works**:

1. Project the continuous feature vector down to a low-dimensional space (e.g., $D$ dimensions)
2. Round each dimension to the nearest integer in a bounded range $[-K, K]$
3. Project back up to the original dimension

$$\bar{H} = \text{round}(\text{Proj}_\text{down}(H))$$
$$\hat{H} = \text{Proj}_\text{up}(\bar{H})$$

Each dimension can take $2K + 1$ discrete values. With $D$ dimensions, the total codebook size is $(2K+1)^D$. The token index is computed as:

$$\mu_i = \sum_{j=0}^{D-1} \bar{h}_{i,j} \cdot (2K+1)^j$$

```python
class FiniteScalarQuantization(nn.Module):
    def __init__(self, input_dim, num_dimensions=4, num_levels=9):
        """
        FSQ: Finite Scalar Quantization.
        
        With D=4 dimensions and 9 levels per dimension:
        Codebook size = 9^4 = 6,561 entries
        """
        super().__init__()
        self.num_dimensions = num_dimensions
        self.num_levels = num_levels  # 2K+1
        self.K = (num_levels - 1) // 2
        
        self.proj_down = nn.Linear(input_dim, num_dimensions)
        self.proj_up = nn.Linear(num_dimensions, input_dim)
    
    def forward(self, h):
        # Project to low-dimensional space
        z = self.proj_down(h)                         # (B, T, D)
        
        # Bound to [-K, K] using tanh
        z = self.K * torch.tanh(z)
        
        # Round to nearest integer (straight-through estimator)
        z_hat = z + (z.round() - z).detach()          # forward: rounded, backward: smooth
        
        # Project back up
        h_hat = self.proj_up(z_hat)                   # (B, T, input_dim)
        
        # Compute token indices
        z_int = z_hat.round().long() + self.K         # shift to [0, 2K]
        indices = torch.zeros(z_int.shape[:-1], dtype=torch.long, device=h.device)
        for j in range(self.num_dimensions):
            indices += z_int[..., j] * (self.num_levels ** j)
        
        return h_hat, indices
```

**FSQ vs VQ comparison**:

| Aspect | VQ (v1) | FSQ (v2) |
|--------|---------|----------|
| Codebook size | 4,096 | 6,561 |
| Codebook utilization | 23% | **100%** |
| ASR error rate (CommonVoice EN) | 18.26% | **10.67%** |
| Learning | Codebook vectors learned via EMA | Projections learned, rounding is deterministic |
| Codebook collapse risk | High (dead codes) | **None** (all entries reachable by construction) |
| Token rate | Variable | **25 Hz** (25 tokens per second) |

**Training the tokenizer**:

The tokenizer is trained end-to-end as an ASR model — the VQ/FSQ bottleneck is inserted into the encoder, and the whole system is trained with ASR loss (CTC + attention). The speech tokens that emerge are supervised by the ASR objective, ensuring they capture semantic content.

```
Training data: 200,000 hours (110,884h Chinese + 99,918h English)
Base model: SenseVoice-Large (pretrained ASR)
Training: 210,000 steps on 8× A800 GPUs
```

## Stage 1: Autoregressive LLM — Text to Speech Tokens

Once we have a speech tokenizer, the core TTS task becomes a **sequence-to-sequence language modeling problem**: given text tokens, generate speech tokens autoregressively.

### CosyVoice 1 LLM

The v1 LLM is a custom Transformer decoder with a separate text encoder:

```
Input sequence:
[SOS] [speaker_embedding] [text_encodings] [TOS] [speech_tokens] [EOS]

Where:
  SOS = start of sequence
  speaker_embedding = 192-dim x-vector projected to LLM dimension
  text_encodings = BPE text → 6-layer Conformer encoder
  TOS = turn of speech (separator between text and speech)
  speech_tokens = target speech token sequence
  EOS = end of sequence
```

The LLM is trained with **next-token prediction loss on speech tokens only** — text tokens serve as context but don't contribute to the loss:

$$\mathcal{L}_\text{LM} = -\frac{1}{L+1} \sum_{l=1}^{L+1} \log q(\mu_l)$$

Where $\mu_l$ are speech tokens and $q(\mu_l)$ is the model's predicted probability.

### CosyVoice 2 LLM: Three Key Simplifications

CosyVoice 2 makes the architecture dramatically simpler:

**1. Pre-trained Qwen2.5-0.5B replaces custom LLM**

Instead of training an LLM from scratch, CosyVoice 2 initializes with a pre-trained Qwen2.5-0.5B (896-dimensional, 24 layers). The pre-trained language model already understands text structure, grammar, and semantics — it just needs to learn the text-to-speech-token mapping.

**2. Text encoder removed entirely**

The paper argues that a pre-trained LLM is powerful enough to align text and speech tokens without a separate text encoder. Raw BPE tokens are fed directly to the LLM.

**3. Speaker embedding removed from LLM input**

In v1, the speaker embedding was fed to both the LLM and the flow matching decoder. In v2, it's only used in the flow matching decoder. This prevents "information leakage" — the LLM should focus on linguistic content, while the flow matching decoder handles speaker identity.

**Ablation results (each change improves quality)**:

| Modification | test-zh CER | test-hard WER |
|-------------|------------|---------------|
| CosyVoice v1 baseline | 3.63% | 11.75% |
| + LLM init (Qwen2.5) | 2.96% | 9.94% |
| + Drop speaker embedding | 2.56% | 9.66% |
| + FSQ (replace VQ) | **1.45%** | **6.83%** |

### Sequence Formats

**Non-streaming (offline TTS)**:

```
[SOS] [text_token_1, ..., text_token_N] [TOS] [speech_token_1, ..., speech_token_L] [EOS]
```

The LLM sees all text first, then generates all speech tokens autoregressively.

**Streaming (bistream format)**:

```
[SOS] [5_text, 15_speech, FILL, 5_text, 15_speech, FILL, ...] [TOS] [remaining_speech] [EOS]
```

Text and speech tokens are interleaved in blocks of N text tokens followed by M speech tokens (default N=5, M=15). A special `FILL` token marks block boundaries. This enables the LLM to start generating speech before seeing all the text — critical for real-time voice applications.

```python
# CosyVoice 2 LLM input construction (simplified)
def build_input_sequence(text_tokens, speech_tokens, mode="non_streaming"):
    sos = torch.tensor([0])    # start of sequence
    tos = torch.tensor([1])    # turn of speech (task_id)
    eos = torch.tensor([6561]) # end of sequence = speech_token_size
    
    if mode == "non_streaming":
        # All text, then all speech
        input_ids = torch.cat([sos, text_tokens, tos, speech_tokens, eos])
    
    elif mode == "streaming":
        # Interleave text and speech in N:M blocks
        fill = torch.tensor([6563])  # fill token = speech_token_size + 2
        N, M = 5, 15  # text:speech ratio
        
        blocks = []
        text_pos, speech_pos = 0, 0
        
        while text_pos < len(text_tokens) or speech_pos < len(speech_tokens):
            # N text tokens
            text_chunk = text_tokens[text_pos:text_pos + N]
            text_pos += N
            # M speech tokens
            speech_chunk = speech_tokens[speech_pos:speech_pos + M]
            speech_pos += M
            
            blocks.extend([text_chunk, speech_chunk, fill])
        
        # Remaining speech after all text is consumed
        remaining = speech_tokens[speech_pos:]
        input_ids = torch.cat([sos] + blocks + [tos, remaining, eos])
    
    return input_ids
```

### Repetition Aware Sampling (RAS)

Autoregressive TTS models often suffer from **looping** — the model gets stuck repeating the same tokens indefinitely. CosyVoice 2 addresses this with Repetition Aware Sampling:

- Standard sampling: top-p=0.8, top-k=25
- RAS monitors a sliding window of recent tokens (window=10)
- If a token has been repeated too frequently within the window, its probability is reduced by factor $\tau_r = 0.1$

This simple heuristic dramatically reduces looping without degrading natural repetitions in speech (like "um" or "uh").

### Training Configuration

```yaml
# CosyVoice 2 LLM training
optimizer: adam
learning_rate: 1e-5          # low LR — pre-trained LLM, don't destroy weights
warmup_steps: 2500
scheduler: constant_lr       # constant after warmup
gradient_clip: 5.0
accumulation_steps: 2
max_epochs: 200
precision: bf16
loss: cross_entropy           # on speech tokens only
label_smoothing: 0.0
```

Note the very low learning rate (1e-5) — this is because we're fine-tuning a pre-trained Qwen2.5-0.5B. Higher learning rates would destroy the pre-trained text representations.

## Stage 2: Flow Matching Decoder — Speech Tokens to Mel Spectrogram

The flow matching decoder converts the discrete speech tokens (from the LLM) into a continuous mel spectrogram. This is the most technically interesting component.

### Why Flow Matching?

The LLM outputs discrete tokens at 25 Hz, but the mel spectrogram has 50 Hz frame rate and 80 continuous dimensions per frame. We need a model that:

1. Upsamples from 25 Hz tokens to 50 Hz mel frames
2. Generates continuous 80-dimensional vectors from discrete inputs
3. Captures the fine acoustic details that discrete tokens can't represent (timbre, prosody, micro-intonation)

Flow matching is ideal because it generates high-quality continuous outputs without the training instability of GANs or the blurriness of simple regression.

### Optimal Transport Conditional Flow Matching (OT-CFM)

CosyVoice uses OT-CFM. The goal is to learn a velocity field $v_\theta$ that transports samples from Gaussian noise $\mathcal{N}(0, I)$ to the mel spectrogram distribution.

**The interpolation path** (optimal transport):

$$\phi_t(X_0, X_1) = (1 - (1-\sigma_\text{min})t) \cdot X_0 + t \cdot X_1$$

Where $X_0 \sim \mathcal{N}(0, I)$ is noise, $X_1$ is the target mel spectrogram, $t \in [0, 1]$, and $\sigma_\text{min} = 10^{-6}$.

**The target velocity**:

$$\omega_t = X_1 - (1-\sigma_\text{min}) \cdot X_0$$

**The training loss** (L1 regression):

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, X_0 \sim \mathcal{N}(0,I),\, X_1 \sim q} \left\| \omega_t - v_\theta(\phi_t; \mu, \tilde{X}_1, v, t) \right\|_1$$

Where the velocity field $v_\theta$ is conditioned on:
- $\mu$ = speech tokens (from the LLM)
- $\tilde{X}_1$ = masked mel spectrogram (prompt audio for voice cloning)
- $v$ = speaker embedding (192-dim from CAM++)
- $t$ = timestep

```python
def flow_matching_loss(model, x1, speech_tokens, speaker_emb, prompt_mel):
    """
    CosyVoice flow matching training step.
    
    x1: target mel spectrogram, (B, 80, T_mel)
    speech_tokens: discrete tokens from tokenizer, (B, T_tok)
    speaker_emb: 192-dim speaker vector, (B, 192)
    prompt_mel: mel spectrogram of prompt audio, (B, 80, T_prompt)
    """
    batch_size = x1.shape[0]
    sigma_min = 1e-6
    
    # Sample random timestep
    t = torch.rand(batch_size, device=x1.device)
    
    # Sample noise
    x0 = torch.randn_like(x1)
    
    # Interpolated state (OT path)
    phi_t = (1 - (1 - sigma_min) * t[:, None, None]) * x0 + t[:, None, None] * x1
    
    # Target velocity
    target_v = x1 - (1 - sigma_min) * x0
    
    # Classifier-free guidance: drop conditions with probability 0.2
    if random.random() < 0.2:
        speech_tokens = torch.zeros_like(speech_tokens)
        speaker_emb = torch.zeros_like(speaker_emb)
        prompt_mel = torch.zeros_like(prompt_mel)
    
    # Predict velocity
    pred_v = model(phi_t, t, speech_tokens, speaker_emb, prompt_mel)
    
    # L1 loss (masked for variable-length sequences)
    loss = F.l1_loss(pred_v, target_v)
    
    return loss
```

### Classifier-Free Guidance (CFG)

During training, all conditions (speech tokens, speaker embedding, prompt mel) are randomly dropped with probability 0.2. During inference, the model runs twice — once with conditions and once without — and the outputs are extrapolated:

$$\tilde{v}_t = (1 + \beta) \cdot v_t(\text{conditioned}) - \beta \cdot v_t(\text{unconditioned})$$

Where $\beta = 0.7$ is the guidance strength.

### Architecture: Conditional UNet

The flow matching decoder uses a **conditional UNet** architecture:

```
Input: [mel (80) | speech_tokens (80) | speaker_emb (80)] = 240 channels
                    ↓
            ┌───────────────┐
            │  UNet Decoder  │
            │  4 blocks      │
            │  12 mid blocks │
            │  8 attn heads  │
            │  256 channels   │
            └───────┬───────┘
                    ↓
            Output: mel (80 channels)
```

The speech tokens (at 25 Hz) are upsampled to match the mel frame rate (50 Hz) via a **length regulator** (learned interpolation). The speaker embedding is expanded temporally to match the sequence length. All three are channel-concatenated as the input to the UNet (240 = 80 + 80 + 80 channels).

### CosyVoice 2: Chunk-Aware Causal Flow Matching

This is the core innovation that enables **streaming synthesis** — generating and playing audio while the LLM is still producing speech tokens.

**The problem**: Standard flow matching uses non-causal convolutions and full attention in the UNet. Each frame can see all other frames. This means you must have the complete speech token sequence before running the decoder — no streaming.

**The solution**: Replace all convolutions with **causal convolutions** (left-padded, so each frame only sees past frames) and train with multiple attention masks simultaneously.

**Four attention masks** (randomly sampled during training):

```
1. Non-causal:  Each frame sees ALL frames
   [████████████████████]
   Best quality, offline only
   
2. Full-causal: Each frame sees only PAST frames
   [█░░░░░░░░░░░░░░░░░░]
   [██░░░░░░░░░░░░░░░░░]
   [███░░░░░░░░░░░░░░░░]
   Lowest latency, real-time streaming

3. Chunk-M:     Each frame sees past + M future frames (first chunk)
   [████████░░░░░░░░░░░]
   Low latency with some lookahead

4. Chunk-2M:    Each frame sees past + 2M future frames (subsequent chunks)
   [████████████████░░░]
   Better quality than Chunk-M
```

During training, one of the four masks is **randomly sampled** per example. This acts as **implicit self-distillation** — the non-causal mode produces the best representations and implicitly teaches the causal modes through shared parameters:

```python
class ChunkAwareCausalFlowMatching(nn.Module):
    def __init__(self, unet, chunk_size_M=15):
        super().__init__()
        self.unet = unet  # CausalConditionalDecoder (causal convs)
        self.M = chunk_size_M
    
    def get_random_mask(self, seq_len, device):
        """Sample one of 4 attention masks randomly during training."""
        mask_type = random.choice(["non_causal", "full_causal", "chunk_M", "chunk_2M"])
        
        if mask_type == "non_causal":
            # Full bidirectional attention
            mask = torch.zeros(seq_len, seq_len, device=device)
        
        elif mask_type == "full_causal":
            # Strictly causal
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
        
        elif mask_type == "chunk_M":
            # Causal + M frames lookahead
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=self.M + 1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
        
        elif mask_type == "chunk_2M":
            # Causal + 2M frames lookahead
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=2 * self.M + 1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
        
        return mask
```

**Streaming inference procedure**:

1. LLM generates M speech tokens (one chunk)
2. Flow matching decoder processes the chunk with Chunk-M mask (first chunk) or Chunk-2M mask (subsequent chunks)
3. Vocoder converts the chunk's mel spectrogram to audio
4. Audio is streamed to the user while the next chunk is being generated

**Critical implementation detail**: The flow matching process starts from random noise. For streaming, each chunk needs consistent noise — otherwise there would be discontinuities at chunk boundaries. CosyVoice 2 pre-allocates a **deterministic noise buffer** seeded at 0:

```python
# Pre-allocate noise for consistent streaming
self.rand_noise = torch.randn(1, 80, 15000, generator=torch.Generator().manual_seed(0))
```

**Streaming quality** (essentially lossless):

| Mode | test-zh CER | test-en WER |
|------|------------|-------------|
| Non-streaming LM + Non-streaming FM | 1.45% | 2.57% |
| Streaming LM + Streaming FM | 1.45% | 2.38% |

### Inference: Euler ODE Solver with Cosine Schedule

At inference, the flow matching ODE is solved using the Euler method with a **cosine time schedule**:

$$t_\text{scheduled} = 1 - \cos\left(\frac{\pi \cdot t}{2}\right)$$

This allocates more steps to the beginning of the ODE trajectory (where the signal is noisiest and needs finer resolution). Default: 10 Euler steps.

```python
@torch.no_grad()
def solve_euler(model, shape, conditions, n_steps=10, cfg_strength=0.7):
    """Generate mel spectrogram via Euler ODE integration."""
    # Start from noise
    x = torch.randn(shape) * temperature
    
    # Cosine time schedule
    t_span = torch.linspace(0, 1, n_steps + 1)
    t_span = 1 - torch.cos(t_span * math.pi / 2)
    
    for i in range(n_steps):
        t = t_span[i]
        dt = t_span[i + 1] - t_span[i]
        
        # Classifier-free guidance: run conditioned and unconditioned
        v_cond = model(x, t, **conditions)
        v_uncond = model(x, t, null_conditions)
        
        v_guided = (1 + cfg_strength) * v_cond - cfg_strength * v_uncond
        
        # Euler step
        x = x + dt * v_guided
    
    return x  # mel spectrogram
```

## Stage 3: Vocoder — Mel Spectrogram to Waveform

CosyVoice uses **HiFT** (a HiFi-GAN variant with harmonic modeling) to convert mel spectrograms to audio waveforms.

### HiFT Architecture

```
Mel spectrogram (80-dim, 50 Hz)
    ↓
[Upsampling blocks]
  Upsample rates: [8, 5, 3] → total 120× upsampling
  50 Hz × 120 = 6000 intermediate frames
    ↓
[Harmonic generator]
  8 harmonics modeled via iSTFT
  n_fft=16, hop_length=4
    ↓
[Neural Source Filter (NSF)]
  F0 predictor: ConvRNNF0Predictor
    ↓
Audio waveform (24,000 Hz)
```

**GAN training**: HiFT is trained adversarially with Multi-Period Discriminator (MPD) + Multi-Resolution Spectrogram Discriminator (MRD), using the combined losses:
- Generator: L1 mel loss + feature matching loss + adversarial loss
- Discriminator: adversarial loss

## Speaker Embedding

CosyVoice uses **CAM++** from the 3D-Speaker project for speaker embedding extraction:

- **Model**: CAM++ (Class-Attentive Multi-view)
- **Output**: 192-dimensional speaker vector
- **Extracted from**: Prompt audio (the reference speaker audio for voice cloning)
- **Used in v1**: Fed to both LLM and flow matching decoder
- **Used in v2**: Fed **only** to flow matching decoder (removed from LLM to prevent information leakage — the LLM should generate content-appropriate tokens regardless of speaker identity)

## Training Data

### Scale

| Component | Data | Scale |
|-----------|------|-------|
| Speech tokenizer | Chinese + English | 200,000 hours |
| TTS model (LLM + flow matching) | Chinese, English, Japanese, Korean | ~167,000 hours |
| Instructed generation | Emotion, speed, dialect, personas | 1,500 hours |

### Data Processing Pipeline

Raw web-scraped audio goes through:

1. **Speech detection** — filter out non-speech segments
2. **SNR estimation** — reject low-quality audio (noisy recordings)
3. **Speaker diarization** — separate multiple speakers
4. **Pseudo-labeling** — generate transcripts using Paraformer (Chinese) and SenseVoice (other languages)
5. **Force-alignment** — align text to audio for quality filtering and punctuation
6. **Filtering** — remove mismatched pairs, too-short/too-long utterances

## Reinforcement Learning (CosyVoice 2)

Beyond supervised training, CosyVoice 2 applies RL to further improve quality:

### ASR Reward Loss

Use a differentiable ASR model as a reward — the generated speech should be accurately transcribable:

$$\mathcal{L}_\text{ASR} = -\log P(Y | \hat{H}; \theta_\text{ASR})$$

Where $\hat{H}$ is the speech representation from the generated mel, using Gumbel-softmax for differentiability through the discrete token layer.

### DPO (Direct Preference Optimization)

Generate multiple speech outputs for the same text, score them with WER and speaker similarity, create preference pairs (chosen/rejected), and train with DPO loss.

**Results of RL fine-tuning**:

| Configuration | WER | Speaker Similarity |
|--------------|-----|--------------------|
| Base model | 5.34% | 0.721 |
| + SFT | 7.15% | 0.795 |
| + ASR reward | 6.79% | 0.795 |
| + DPO | 6.83% | 0.792 |
| Combined | **6.64%** | **0.796** |

## Inference Modes

CosyVoice supports multiple inference modes:

### 1. Zero-Shot Voice Cloning (In-Context Learning)

Provide a reference audio clip + its transcript + target text:

```python
# Zero-shot voice cloning
output = cosyvoice.inference_zero_shot(
    tts_text="Hello, welcome to our service.",
    prompt_text="The weather today is really nice.",  # transcript of prompt audio
    prompt_speech=load_audio("speaker_sample.wav"),
)
```

The LLM input is: `[SOS] [prompt_text_tokens] [target_text_tokens] [TOS] [prompt_speech_tokens]` → generate target speech tokens.

### 2. Cross-Lingual Synthesis

Clone a speaker's voice into a different language. Omit the prompt text to prevent source-language prosody transfer:

```python
# Chinese speaker → English speech
output = cosyvoice.inference_cross_lingual(
    tts_text="This is spoken in English with a Chinese speaker's voice.",
    prompt_speech=load_audio("chinese_speaker.wav"),
)
```

### 3. Instruct Mode

Control emotion, speaking rate, dialect via natural language instructions:

```python
output = cosyvoice.inference_instruct(
    tts_text="I'm so excited about this news!",
    speaker="Speaker_A",
    instruct_text="Speak with excitement and enthusiasm, at a fast pace.",
)
```

Supported instructions: 8 emotions (happy, sad, angry, surprised, fearful, disgusted, calm, serious), speaking rate (fast/slow), 18+ Chinese dialects, role-playing personas.

### 4. Fine-Grained Control

Embed control tags directly in the text:

```python
# Fine-grained paralinguistic control
output = cosyvoice.inference_sft(
    tts_text="Well [laughter] that's really funny [breath] let me think about it.",
    speaker="Speaker_A",
)
```

## Evaluation Metrics

| Metric | What It Measures | Tool |
|--------|-----------------|------|
| CER | Character Error Rate (Chinese) | Paraformer ASR |
| WER | Word Error Rate (English) | Whisper-Large-V3 |
| Speaker Similarity (SS) | Cosine similarity of speaker embeddings | ERes2Net |
| NMOS | Naturalness Mean Opinion Score | Human evaluation |

### CosyVoice 2 Results (LibriSpeech test-clean)

| Model | WER | NMOS | Speaker Sim |
|-------|-----|------|-------------|
| Human | 2.66% | 3.84 | 0.697 |
| CosyVoice 2 | **2.47%** | **3.96** | **0.745** |
| CosyVoice 2 (streaming) | 2.45% | 3.90 | 0.751 |

CosyVoice 2 actually outperforms human recordings on WER and naturalness — the model produces cleaner, more consistent speech than the noisy human recordings in the test set.

## Interview Questions and Answers

### Q: Explain the CosyVoice architecture at a high level. What are the main components?

CosyVoice is a two-stage TTS system:

**Stage 1 — Text to speech tokens**: An autoregressive LLM (Qwen2.5-0.5B in v2) converts BPE text tokens into discrete speech tokens at 25 Hz. The input sequence is `[SOS] [text_tokens] [TOS] [speech_tokens] [EOS]`. Only speech tokens contribute to the cross-entropy loss. The LLM conditions on text context and generates speech tokens that capture linguistic content, prosody, and rhythm.

**Stage 2 — Speech tokens to mel spectrogram**: A conditional flow matching decoder converts the discrete tokens into a continuous 80-dimensional mel spectrogram at 50 Hz. It conditions on speech tokens (upsampled 2x from 25→50 Hz), speaker embedding (192-dim from CAM++), and prompt mel (for voice cloning). A HiFT vocoder then converts the mel spectrogram to a 24 kHz waveform.

The speech tokenizer is trained separately as an ASR model with a quantization bottleneck (VQ in v1, FSQ in v2), ensuring tokens capture semantic content.

### Q: What is Finite Scalar Quantization (FSQ) and why is it better than Vector Quantization (VQ)?

VQ maintains a codebook of $N$ learned embedding vectors. For each input, it finds the nearest codebook entry. Problem: **codebook collapse** — most entries are never selected as the nearest neighbor. CosyVoice v1 only used 23% of its 4,096 codes.

FSQ takes a different approach: project the input down to $D$ dimensions, round each dimension to the nearest integer in $[-K, K]$, and project back up. The codebook is **implicit** — every combination of rounded values corresponds to a valid code. With $D=4$ dimensions and 9 levels, the codebook size is $9^4 = 6,561$.

**Why FSQ is better**: (1) **100% utilization** — every code is reachable by construction, no dead codes. (2) **Better downstream quality** — CER on CommonVoice dropped from 18.26% (VQ) to 10.67% (FSQ). (3) **No codebook learning** — only the projection layers are learned; rounding is deterministic. (4) **No EMA updates** — simpler training, no commitment loss needed.

### Q: How does the flow matching decoder work? What is OT-CFM?

Optimal Transport Conditional Flow Matching (OT-CFM) learns a velocity field that transports samples from Gaussian noise to the mel spectrogram distribution along straight-line paths.

**Training**: Sample noise $X_0 \sim \mathcal{N}(0,I)$, a target mel $X_1$, and time $t \sim \mathcal{U}[0,1]$. Interpolate: $\phi_t = (1-t)X_0 + tX_1$ (simplified). The target velocity is $X_1 - X_0$. Train the network to predict this velocity given the interpolated state, speech tokens, speaker embedding, and prompt mel. Loss is L1 between predicted and target velocity.

**Inference**: Start from noise, solve the ODE $dx/dt = v_\theta(x, t)$ using 10 Euler steps with a cosine time schedule. Classifier-free guidance ($\beta=0.7$) amplifies the conditioning signal.

The conditioning is channel-concatenated: speech token embeddings (80-dim, upsampled to 50 Hz) + speaker embedding (80-dim, temporally expanded) + prompt mel (80-dim) = 240-channel input to the UNet.

### Q: How does CosyVoice 2 enable streaming synthesis?

Two innovations enable streaming:

**1. Bistream LLM format**: Text and speech tokens are interleaved in N:M blocks (5 text tokens, 15 speech tokens, fill marker, repeat). The LLM can start generating speech after seeing just 5 text tokens, instead of waiting for all text.

**2. Chunk-aware causal flow matching**: The UNet uses causal convolutions (left-padded). During training, four attention masks are randomly sampled per example: non-causal (offline, best quality), full-causal (real-time), chunk-M (first chunk with M-frame lookahead), and chunk-2M (subsequent chunks with 2M-frame lookahead).

This random mask sampling acts as **implicit self-distillation** — the non-causal mode produces the best representations and guides the causal modes through shared weights. At inference, streaming uses the causal masks; offline uses non-causal.

**Streaming procedure**: The LLM generates M tokens → flow matching processes the chunk → vocoder produces audio → stream to user → repeat. Latency as low as 150ms. Quality is essentially lossless compared to offline mode.

### Q: What is classifier-free guidance (CFG) in the context of CosyVoice?

During flow matching training, all conditions (speech tokens, speaker embedding, prompt mel) are randomly dropped with probability $p=0.2$ — replaced with zeros. This teaches the model to operate both with and without conditions.

At inference, the model runs twice per step: once with conditions (conditioned prediction $v_\text{cond}$) and once without (unconditioned prediction $v_\text{uncond}$). The final velocity is:

$$\tilde{v} = (1 + \beta) \cdot v_\text{cond} - \beta \cdot v_\text{uncond}$$

With $\beta=0.7$, this amplifies the conditioning signal — the generated mel spectrogram better matches the target speaker and content. Higher $\beta$ = more faithful to conditions but potentially less natural. Lower $\beta$ = more natural but may drift from the target speaker/content.

### Q: Why did CosyVoice 2 remove the speaker embedding from the LLM input?

In v1, the speaker embedding was fed to both the LLM and the flow matching decoder. This caused **information leakage**: the LLM could condition its speech token generation on the speaker's identity, potentially ignoring linguistic content in favor of speaker-specific patterns.

By removing the speaker embedding from the LLM, the architecture enforces a clean separation of concerns:
- **LLM**: Responsible for linguistic content, prosody, and rhythm (speaker-agnostic)
- **Flow matching decoder**: Responsible for speaker identity, timbre, and acoustic details

Ablation shows this improves quality: CER dropped from 2.96% to 2.56% on test-zh just from removing the speaker embedding from the LLM.

### Q: How does zero-shot voice cloning work in CosyVoice?

The user provides: (1) a reference audio clip (3+ seconds), (2) the transcript of that reference, and (3) the target text to synthesize.

**Step 1**: Extract a 192-dim speaker embedding from the reference audio using CAM++.

**Step 2**: The LLM receives `[SOS] [prompt_transcript_tokens] [target_text_tokens] [TOS] [prompt_speech_tokens]` and generates target speech tokens. The prompt speech tokens teach the LLM the prosodic patterns; the prompt transcript provides text alignment.

**Step 3**: The flow matching decoder converts the target speech tokens into a mel spectrogram, conditioned on the speaker embedding and the prompt mel spectrogram. The prompt mel provides fine-grained acoustic conditioning (timbre, recording environment).

**Step 4**: The HiFT vocoder produces the final waveform.

For cross-lingual cloning (e.g., Chinese speaker → English speech), the prompt transcript is omitted to prevent source-language prosody from bleeding into the target language.

### Q: Compare CosyVoice with other TTS systems (VALL-E, XTTS, etc.).

| Aspect | VALL-E | XTTS (Coqui) | CosyVoice 2 |
|--------|--------|-------------|-------------|
| Tokenizer | EnCodec (neural codec, 8 codebooks) | VQ-VAE | FSQ (single codebook, ASR-supervised) |
| LLM | Custom AR + NAR | GPT-2 variant | Qwen2.5-0.5B (pre-trained) |
| Decoder | NAR codec decoder | decoder + HiFi-GAN | Flow matching + HiFT |
| Streaming | No | Limited | Yes (chunk-aware causal) |
| Voice cloning | 3s prompt | 6s prompt | 3s prompt |
| Languages | English | 16 | 9 + 18 Chinese dialects |
| Training data | 60K hours (LibriLight) | Proprietary | 167K hours |
| Open source | No | Yes (AGPL) | Yes (Apache 2.0) |

CosyVoice 2's key advantages: (1) FSQ with ASR supervision produces more semantically meaningful tokens than neural codec tokens (lower WER in generated speech). (2) Pre-trained Qwen backbone provides better text understanding. (3) Chunk-aware causal flow matching enables low-latency streaming in a unified model.

### Q: What are the main challenges in training CosyVoice and how are they addressed?

**1. Codebook collapse in speech tokenization**
- Problem: VQ codebooks have dead entries (77% unused in v1)
- Solution: FSQ — deterministic quantization with 100% utilization by construction

**2. LLM repetition/looping**
- Problem: Autoregressive TTS models get stuck repeating tokens
- Solution: Repetition Aware Sampling (RAS) — penalize recently repeated tokens

**3. Streaming quality degradation**
- Problem: Causal constraints degrade generation quality
- Solution: Chunk-aware training with random mask sampling — non-causal mode implicitly distills knowledge to causal modes through shared parameters

**4. Speaker identity vs linguistic content entanglement**
- Problem: LLM might rely on speaker identity instead of text content
- Solution: Remove speaker embedding from LLM input; only condition the flow matching decoder on speaker

**5. Data quality at scale**
- Problem: 167K hours of web-scraped audio is noisy
- Solution: Multi-stage filtering pipeline (VAD → SNR → diarization → pseudo-labeling → force-alignment → quality filtering)

### Q: How would you fine-tune CosyVoice for a custom voice or domain?

**Single-speaker fine-tuning** (e.g., creating a custom voice assistant):

1. Collect 1-10 hours of clean, single-speaker recordings with transcripts
2. Extract speech tokens using the frozen tokenizer
3. Fine-tune the LLM with SFT format: `[SOS] "Speaker_Custom<|endofprompt|>" [text_tokens] [TOS] [speech_tokens] [EOS]`
4. Fine-tune the flow matching decoder on the same data
5. Use very low learning rate (1e-5 or lower) to preserve the base model's capabilities

**Multi-speaker fine-tuning (mSFT)**: Tag each utterance with a speaker identifier `"Speaker_A<|endofprompt|>"` to prevent timbre confusion across speakers.

**Domain adaptation** (e.g., medical dictation, financial reports): Fine-tune primarily the LLM on domain-specific text-speech pairs. The flow matching decoder and vocoder usually don't need domain-specific adaptation.

**Key practical tips**:
- Always use bf16, not fp16 (numerical stability)
- Gradient clipping at 5.0 (the LLM loss can spike)
- Monitor generated audio quality every few hundred steps — loss curves don't always correlate with perceptual quality
- Start from the official CosyVoice2-0.5B checkpoint, don't train from scratch

## References

1. Du, Z., et al. "CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer Using Supervised Semantic Tokens." 2024. arXiv:2407.05407
2. Du, Z., et al. "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models." 2024. arXiv:2412.10117
3. [CosyVoice GitHub Repository](https://github.com/FunAudioLLM/CosyVoice)
4. [CosyVoice2-0.5B on Hugging Face](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
5. Mentzer, F., et al. "Finite Scalar Quantization: VQ-VAE Made Simple." ICLR 2024.
6. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023.
7. Yang, D., et al. "Qwen2.5 Technical Report." 2024.
8. Wang, C., et al. "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)." 2023.
