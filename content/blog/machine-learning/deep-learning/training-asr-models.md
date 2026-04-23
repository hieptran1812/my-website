---
title: "Training Automatic Speech Recognition Models: A Complete Guide"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "asr",
    "speech-recognition",
    "deep-learning",
    "whisper",
    "ctc",
    "transducer",
    "conformer",
    "audio",
    "transformer",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A comprehensive guide to training ASR models — from audio fundamentals and feature extraction, through CTC, RNN-T, and attention-based architectures, to modern foundation models like Whisper. Covers data pipelines, training recipes, evaluation, and interview-ready depth."
---

## What Is ASR?

![Three ASR architectures: CTC, RNN-T, and encoder-decoder, with their components and tradeoffs](/imgs/blogs/training-asr-models-diagram.png)

Automatic Speech Recognition (ASR) converts spoken language into text. You speak "What's the weather today?" and the model outputs the string `"What's the weather today?"`. It's the technology behind Siri, Google Assistant, Alexa, live captions, meeting transcription, and voice-controlled interfaces.

At its core, ASR is a **sequence-to-sequence** problem — but with a twist. The input (audio waveform) and output (text) have very different lengths and structures:

- Input: A waveform sampled at 16,000 Hz → 1 second of audio = 16,000 numbers
- Feature extraction: Convert to 80-dimensional mel spectrogram frames every 10ms → 1 second = 100 frames
- Output: Text with ~3-5 words → maybe 20-30 characters

The input is 100x longer than the output, and there's no simple one-to-one alignment between frames and characters. This **alignment problem** is the central challenge of ASR, and the three major architectures (CTC, RNN-T, Attention) each solve it differently.

```
Audio waveform (16kHz, 3 seconds):
  ~~~∿∿∿~~~∿∿∿∿∿∿~~~∿∿∿∿∿~~~∿∿∿∿~~~∿∿∿~~~  (48,000 samples)
                    ↓
Mel spectrogram (80-dim, 10ms frames):
  [████████████████████████████████]           (300 frames)
                    ↓
Text output:
  "Hello world"                                (11 characters)
```

## Audio Fundamentals for ASR

### The Audio Signal

Sound is a pressure wave. A microphone converts it into a continuous electrical signal, which is **digitized** by sampling at regular intervals:

- **Sample rate**: How many times per second we measure the signal. Standard for ASR: **16,000 Hz** (16 kHz). Each second produces 16,000 float values.
- **Bit depth**: Precision of each sample. Typically 16-bit integers (range -32,768 to 32,767) or 32-bit floats.
- **Channels**: Mono (1 channel) is standard for ASR. Stereo audio is mixed to mono before processing.

```python
import torchaudio

# Load audio file
waveform, sample_rate = torchaudio.load("speech.wav")
# waveform: shape (1, num_samples), e.g., (1, 48000) for 3 seconds at 16kHz
# sample_rate: 16000

# Resample if needed (many recordings are 44.1kHz or 48kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
```

### Feature Extraction: From Waveform to Mel Spectrogram

Raw waveforms are high-dimensional and redundant. We compress them into **mel spectrograms** — a time-frequency representation that approximates human auditory perception.

**Step 1: Short-Time Fourier Transform (STFT)**

Slice the waveform into overlapping windows (typically 25ms windows every 10ms) and compute the frequency content of each window via FFT:

$$X[t, f] = \sum_{n=0}^{N-1} x[t \cdot H + n] \cdot w[n] \cdot e^{-2\pi i f n / N}$$

Where $w[n]$ is a window function (Hann window), $H$ is the hop size, and $N$ is the FFT size.

**Step 2: Mel Filter Bank**

The mel scale compresses high frequencies (humans are less sensitive to pitch differences at high frequencies). Apply triangular filters spaced on the mel scale to the power spectrum:

$$\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

Typically 80 mel filter banks for modern ASR models.

**Step 3: Log Compression**

Apply logarithm to compress the dynamic range (quiet sounds and loud sounds are treated more equally):

$$\text{log-mel}[t, m] = \log(\text{mel-filterbank}[t, m] + \epsilon)$$

The result: a 2D representation of shape `(T, 80)` where `T` is the number of frames (1 frame per 10ms).

```python
import torch
import torchaudio.transforms as T

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
        """
        Standard ASR feature extraction.
        
        n_fft=400: 25ms window at 16kHz
        hop_length=160: 10ms hop at 16kHz
        n_mels=80: 80 mel filter banks
        """
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
    
    def __call__(self, waveform):
        # waveform: (1, num_samples)
        mel = self.mel_spec(waveform)           # (1, 80, T)
        log_mel = torch.log(mel.clamp(min=1e-9)) # log compression
        return log_mel.squeeze(0).transpose(0, 1) # (T, 80)

# Example: 3 seconds of audio → 300 frames × 80 mel bins
extractor = AudioFeatureExtractor()
features = extractor(waveform)  # shape: (300, 80)
```

### Other Feature Types

| Feature | Dimensions | Used By | Notes |
|---------|-----------|---------|-------|
| Log-mel spectrogram | 80 | Whisper, Conformer, most modern models | Standard choice |
| MFCC (Mel-Frequency Cepstral Coefficients) | 13-40 | Legacy systems (Kaldi) | Decorrelated features, less common now |
| Filter bank energies (fbank) | 40-80 | Many research systems | Similar to log-mel without log |
| Raw waveform | 1 | Wav2Vec 2.0, HuBERT | Model learns its own features |

Modern architectures overwhelmingly use **80-dimensional log-mel spectrograms** or learn directly from raw waveforms. MFCCs are considered legacy.

## The Three ASR Architectures

### 1. CTC (Connectionist Temporal Classification)

**Core idea**: The encoder processes the entire audio input and produces one output per frame. A special **blank token** $\langle b \rangle$ handles the alignment — the model can output blank at frames where no new character should be emitted. Multiple paths through blanks and repeated characters can produce the same output text.

```
Audio frames:  [f1] [f2] [f3] [f4] [f5] [f6] [f7] [f8] [f9] [f10]
CTC output:     h    h   <b>   e    l    l   <b>   l    o    <b>
                ↓ collapse repeated characters and remove blanks
Text:          "hello"
```

**The CTC collapsing rule**: Remove consecutive duplicate characters, then remove all blanks. This means multiple CTC paths map to the same output:

```
"hh<b>ell<b>lo<b>" → "hello"
"h<b>e<b>ll<b>lo"  → "hello"  (different path, same output!)
"<b>hello<b><b><b>" → "hello"
```

**The CTC loss** marginalizes over all valid alignments:

$$P(y | x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} P(\pi_t | x_t)$$

Where $\mathcal{B}^{-1}(y)$ is the set of all CTC paths that collapse to text $y$. This sum is computed efficiently using the **forward-backward algorithm** (dynamic programming).

```python
import torch.nn as nn

class CTCModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, vocab_size=29):
        """
        Simple CTC ASR model.
        vocab_size = 26 letters + space + apostrophe + CTC blank
        """
        super().__init__()
        self.encoder = nn.Sequential(
            # Convolutional frontend for downsampling
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Transformer or RNN encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, features, feature_lengths):
        # features: (B, T, 80)
        x = features.transpose(1, 2)         # (B, 80, T)
        x = self.encoder(x)                  # (B, hidden, T//4)
        x = x.transpose(1, 2)               # (B, T//4, hidden)
        x = self.transformer(x)             # (B, T//4, hidden)
        logits = self.projection(x)          # (B, T//4, vocab_size)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs

# Training with CTC loss
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

log_probs = model(features, feature_lengths)    # (B, T', vocab_size)
input_lengths = feature_lengths // 4            # after 2x downsampling twice
loss = ctc_loss(
    log_probs.transpose(0, 1),  # (T', B, vocab_size) — CTC expects time-first
    targets,                     # (B, S) — target text as integer indices
    input_lengths,               # (B,) — length of each input sequence
    target_lengths,              # (B,) — length of each target text
)
```

**Pros**:
- Simple — encoder-only, no autoregressive decoder
- Fast inference — no sequential token generation, just a forward pass + greedy decode or beam search
- Streamable — can output tokens as audio arrives

**Cons**:
- **Conditional independence assumption**: CTC assumes each frame's output is independent of all other frames' outputs (given the encoder output). This means CTC can't model language-level dependencies well — it might output "their" when "there" is grammatically correct
- **Monotonic alignment only**: CTC assumes input and output are monotonically aligned (left-to-right). This prevents reordering, which matters for some languages
- Requires external language model for best results

**Used by**: DeepSpeech 2, Wav2Vec 2.0 + CTC head, NeMo Conformer-CTC.

### 2. RNN-T (RNN Transducer) / Transducer

**Core idea**: Combine an **encoder** (processes audio) with a **prediction network** (models text history, like a language model) and a **joint network** (combines both to produce output probabilities). Unlike CTC, the transducer can condition on previously emitted tokens, solving the conditional independence problem.

```
                    ┌─────────────────┐
Audio frames  →     │    Encoder       │ → encoder features h_t
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Joint Network  │ → P(y | t, u)
                    └────────▲────────┘
                             │
Previous text →     │ Prediction Net  │ → prediction features g_u
                    └─────────────────┘
```

The joint network combines encoder output $h_t$ (at audio frame $t$) and prediction network output $g_u$ (at text position $u$):

$$z_{t,u} = \text{Joint}(h_t, g_u) = \text{Linear}(\tanh(W_h h_t + W_g g_u))$$

$$P(y_{t,u} | t, u) = \text{softmax}(z_{t,u})$$

The output at each $(t, u)$ point is either a character/token or the **blank** symbol $\langle b \rangle$. Emitting blank advances the time step $t$ without emitting a character. Emitting a character advances the text position $u$.

```python
class RNNTransducer(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, vocab_size=29):
        super().__init__()
        # Audio encoder (Conformer or Transformer)
        self.encoder = ConformerEncoder(input_dim, hidden_dim, num_layers=12)
        
        # Prediction network (text history LM)
        self.prediction = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.pred_embed = nn.Embedding(vocab_size, vocab_size)
        
        # Joint network
        self.joint = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )
    
    def forward(self, features, feature_lengths, targets, target_lengths):
        # Encoder: process audio
        enc_out = self.encoder(features)             # (B, T, hidden)
        
        # Prediction network: process previous text
        target_embed = self.pred_embed(targets)      # (B, U, vocab_size)
        pred_out, _ = self.prediction(target_embed)  # (B, U, hidden)
        
        # Joint network: combine at every (t, u) pair
        # Expand for broadcasting: (B, T, 1, H) + (B, 1, U, H) → (B, T, U, H)
        enc_expanded = enc_out.unsqueeze(2)          # (B, T, 1, hidden)
        pred_expanded = pred_out.unsqueeze(1)        # (B, 1, U, hidden)
        
        joint_input = torch.cat([
            enc_expanded.expand(-1, -1, pred_out.size(1), -1),
            pred_expanded.expand(-1, enc_out.size(1), -1, -1),
        ], dim=-1)                                   # (B, T, U, 2*hidden)
        
        logits = self.joint(joint_input)             # (B, T, U, vocab_size)
        return logits
```

**The transducer loss** (also computed via forward-backward dynamic programming) is more complex than CTC because it operates on a 2D lattice $(T \times U)$ instead of CTC's 1D path.

**Pros**:
- Models text dependencies (prediction network acts as implicit LM)
- Streamable — can emit tokens as audio arrives
- Better accuracy than CTC without an external LM

**Cons**:
- More complex to implement and train
- The 2D lattice makes the loss computation memory-intensive ($O(T \times U)$)
- Slower decoding (autoregressive prediction network)

**Used by**: Google's speech recognition (Pixel phones), NeMo Conformer-Transducer, most production on-device ASR.

### 3. Attention-Based Encoder-Decoder (AED)

**Core idea**: Use a full encoder-decoder architecture with cross-attention, similar to machine translation. The decoder autoregressively generates text tokens, attending to the full encoded audio at each step.

```
Audio → [Encoder] → encoded features → [Cross-Attention Decoder] → text tokens
                                              ↑
                                    (attends to all audio frames)
```

This is the architecture used by **Whisper**, the most popular ASR model today.

```python
class AttentionASR(nn.Module):
    def __init__(self, input_dim=80, d_model=512, vocab_size=50257):
        super().__init__()
        # Audio encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=2048, batch_first=True),
            num_layers=12,
        )
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Text decoder with cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, features, tokens):
        # Encode audio
        x = features.transpose(1, 2)                    # (B, 80, T)
        x = self.conv_stem(x).transpose(1, 2)           # (B, T//4, d_model)
        memory = self.encoder(x)                         # (B, T//4, d_model)
        
        # Decode text (teacher-forced during training)
        tgt = self.token_embed(tokens)                   # (B, S, d_model)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1))
        output = self.decoder(tgt, memory, tgt_mask=causal_mask)
        logits = self.output_proj(output)                # (B, S, vocab_size)
        return logits

# Training with standard cross-entropy loss (like language modeling)
loss = F.cross_entropy(logits.view(-1, vocab_size), target_tokens.view(-1))
```

**Pros**:
- Highest accuracy — cross-attention can learn flexible, non-monotonic alignments
- Standard cross-entropy training — simpler than CTC/transducer losses
- Can handle tasks beyond transcription: translation, language ID, timestamps
- Benefits from large-scale pretraining (Whisper paradigm)

**Cons**:
- **Not streamable**: The decoder must attend to the full encoded audio, requiring the entire utterance before decoding starts
- Slower inference: autoregressive token generation
- Attention can fail on very long audio (>30 seconds without chunking)
- Requires paired audio-text data (no self-supervised pretraining like Wav2Vec)

**Used by**: Whisper, ESPnet encoder-decoder models, early Listen-Attend-Spell (LAS).

### Architecture Comparison

| Aspect | CTC | RNN-T | Attention (AED) |
|--------|-----|-------|-----------------|
| Alignment | Monotonic, learned via DP | Monotonic, 2D lattice | Flexible, via cross-attention |
| Decoder | None (encoder-only) | Prediction network | Full autoregressive decoder |
| Streaming | Yes | Yes | No (needs full audio) |
| Training loss | CTC loss | Transducer loss | Cross-entropy |
| Accuracy (no ext. LM) | Moderate | Good | Best |
| Accuracy (with ext. LM) | Good | Very good | Best |
| Inference speed | Fastest | Medium | Slowest |
| Complexity | Simplest | Medium | Simplest (standard seq2seq) |
| Best for | Streaming, on-device | Production streaming | Offline, highest quality |

## The Conformer: The Dominant Encoder

The **Conformer** (Gulati et al., 2020) is the standard encoder architecture for ASR, combining the strengths of CNNs (local feature extraction) and Transformers (global context):

```
Conformer Block:
  Input
    ↓
  [Feed-Forward Module (½)] ← first half-step FFN
    ↓
  [Multi-Head Self-Attention] ← global context
    ↓
  [Convolution Module]       ← local patterns (kernel_size=31)
    ↓
  [Feed-Forward Module (½)] ← second half-step FFN
    ↓
  [Layer Norm]
    ↓
  Output
```

The **convolution module** captures local patterns (phoneme boundaries, formant transitions) that attention is inefficient at modeling. The **self-attention** captures global dependencies (long-range context, prosody).

```python
class ConformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, conv_kernel_size=31, ff_expansion=4):
        super().__init__()
        # Half-step Feed-Forward
        self.ff1 = FeedForward(d_model, d_model * ff_expansion)
        self.ff2 = FeedForward(d_model, d_model * ff_expansion)
        
        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Convolution Module
        self.conv = ConvolutionModule(d_model, conv_kernel_size)
        
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Macaron-style: FFN → Attention → Conv → FFN
        x = x + 0.5 * self.ff1(x)
        
        residual = x
        x = self.attn_norm(x)
        x_attn, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = residual + x_attn
        
        x = x + self.conv(x)
        
        x = x + 0.5 * self.ff2(x)
        
        return self.final_norm(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)  # GLU gate
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, 
            padding=kernel_size // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
    
    def forward(self, x):
        residual = x
        x = self.norm(x).transpose(1, 2)      # (B, D, T)
        x = self.pointwise1(x)                 # (B, 2D, T)
        x = x.chunk(2, dim=1)                  # GLU activation
        x = x[0] * torch.sigmoid(x[1])         # (B, D, T)
        x = self.depthwise(x)                  # (B, D, T)
        x = self.batch_norm(x)
        x = x * torch.sigmoid(x)               # Swish activation
        x = self.pointwise2(x)                 # (B, D, T)
        return residual + x.transpose(1, 2)
```

## Whisper: The Foundation Model Approach

Whisper (OpenAI, 2022) changed ASR by showing that **large-scale supervised pretraining** on 680,000 hours of labeled audio from the internet can produce a single model that handles:

- Speech recognition in 99 languages
- Speech translation (any language → English)
- Language identification
- Voice activity detection
- Timestamp prediction

### Architecture

Whisper uses a standard Transformer encoder-decoder with:
- Audio encoder: 80-mel spectrogram input, 2 Conv1d layers for downsampling, then Transformer layers
- Text decoder: Standard autoregressive Transformer decoder with cross-attention
- Special tokens: `<|startoftranscript|>`, `<|en|>`, `<|transcribe|>`, `<|translate|>`, `<|notimestamps|>`

| Model | Layers (Enc/Dec) | Width | Heads | Parameters |
|-------|-----------------|-------|-------|------------|
| tiny | 4/4 | 384 | 6 | 39M |
| base | 6/6 | 512 | 8 | 74M |
| small | 12/12 | 768 | 12 | 244M |
| medium | 24/24 | 1024 | 16 | 769M |
| large-v3 | 32/32 | 1280 | 20 | 1.55B |
| large-v3-turbo | 32/4 | 1280 | 20 | 809M |

### Fine-Tuning Whisper

Fine-tuning Whisper on domain-specific data is the most practical path to a production ASR system:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# Load pretrained Whisper
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Prepare dataset
def preprocess(example):
    audio = example["audio"]["array"]
    input_features = processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features[0]
    
    labels = processor.tokenizer(example["text"]).input_ids
    return {"input_features": input_features, "labels": labels}

dataset = dataset.map(preprocess)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    predict_with_generate=True,
    generation_max_length=225,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

## Self-Supervised Pretraining: Wav2Vec 2.0 and HuBERT

An alternative paradigm: **pretrain on unlabeled audio** (which is abundant), then fine-tune with a small amount of labeled data.

### Wav2Vec 2.0

1. **Feature encoder**: CNN processes raw waveform into latent representations (one per ~20ms)
2. **Quantization module**: Discretizes latent representations into a finite codebook (like VQ-VAE)
3. **Context network**: Transformer processes the latent sequence
4. **Pretraining objective**: Mask some latent representations and predict the correct quantized target from a set of distractors (**contrastive learning**)

```
Raw waveform → [CNN encoder] → latent z_t → [mask some positions]
                                    ↓
                    [Quantized target q_t]     [Transformer context network]
                                    ↓                      ↓
                    Contrastive loss: predict q_t from context output c_t
```

After pretraining, add a CTC head on top and fine-tune with as little as **10 minutes** of labeled data to get usable ASR.

### HuBERT

Similar to Wav2Vec 2.0 but uses **offline clustering** (k-means) to create pseudo-labels instead of online contrastive learning. Iterative: cluster → train → recluster with better features → retrain.

## Tokenization for ASR

Tokenization converts the transcript text into a sequence of integer IDs that the model can predict. The choice of tokenizer directly shapes vocabulary size, sequence length, model size (embedding and output projection), and how well the system handles rare words, proper nouns, and out-of-vocabulary (OOV) items. Unlike NLP where tokens are sometimes treated as a detail, in ASR the tokenizer is a **first-class architectural decision** because:

1. The output vocabulary size defines the final classification head dimension — for CTC and transducer, this is also the blank-inclusive softmax size per frame.
2. The tokenizer determines the target sequence length $U$, which drives the transducer's $O(T \times U)$ lattice memory.
3. Rare-word coverage (names, numbers, code-switched terms) is bounded entirely by what the tokenizer can represent.

### Tokenization Granularities

| Granularity | Vocab Size | Sequence Length | OOV Handling | Used By |
|-------------|-----------|-----------------|--------------|---------|
| Character | 28-100 | Longest (one token per char) | Perfect — any word decomposable | DeepSpeech 2, early CTC |
| Phoneme | 40-60 | Long | Needs pronunciation lexicon | Hybrid HMM systems, Kaldi |
| Byte-level | 256 | Long | Perfect — any Unicode byte | ByT5, experimental ASR |
| Subword (BPE) | 500-16K | Medium | Graceful — splits into known pieces | Conformer-CTC, NeMo, Whisper |
| SentencePiece (unigram) | 1K-10K | Medium | Graceful — probabilistic segmentation | ESPnet, Wav2Vec 2.0 + CTC |
| Word | 10K-200K | Shortest | Catastrophic — `<unk>` for OOV | Legacy systems only |

**Rule of thumb**:
- English / Latin-alphabet single-language: **BPE with 1K-5K vocab**
- Multilingual (>5 languages): **SentencePiece with 10K-50K vocab**
- Chinese, Japanese, Korean (CJK): **character-level** (characters are already semantic units) or **SentencePiece on raw text**
- Very low-resource (<10h labeled): **character-level** (fewer parameters in output head, less overfitting)

### Character-Level Tokenization

The simplest approach: one token per character, plus a few special tokens. The entire vocabulary might be:

```
{<blank>, <pad>, <unk>, <sos>, <eos>,
 ' ', "'", '.', ',', '?', '!',
 a, b, c, d, ..., z}
```

Typical English vocab size: **28-32 tokens**. The output head is tiny (hidden_dim × 32), and the model can never produce an OOV because every word is just a sequence of characters.

```python
class CharTokenizer:
    def __init__(self):
        self.blank = 0
        self.pad = 1
        self.unk = 2
        self.sos = 3
        self.eos = 4
        # Printable characters we care about
        chars = " abcdefghijklmnopqrstuvwxyz'.,?!-"
        self.char_to_id = {c: i + 5 for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars) + 5  # 37 for this setup

    def encode(self, text: str) -> list[int]:
        text = text.lower().strip()
        return [self.char_to_id.get(c, self.unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "") for i in ids
                       if i not in {self.blank, self.pad, self.sos, self.eos})

tok = CharTokenizer()
ids = tok.encode("hello world")      # [13, 10, 17, 17, 20, 5, 28, 20, 22, 17, 9]
text = tok.decode(ids)                # "hello world"
```

**Trade-off**: Short utterances produce long token sequences. A 10-word sentence might be 60 characters → 60 tokens. This makes the transducer lattice expensive and slows attention decoding (more decoder steps).

### Byte-Pair Encoding (BPE)

BPE starts with a character vocabulary and **iteratively merges the most frequent adjacent pair** until the target vocab size is reached. The result: common substrings (like "ing", "tion", "th") become single tokens while rare words fall back to character-level.

**Algorithm**:
1. Initialize vocab with all characters in the training corpus.
2. Count adjacent pair frequencies across the corpus.
3. Merge the most frequent pair (e.g., `("e", "r") → "er"`) and add to vocab.
4. Repeat until vocab size = target.

Worked example training BPE on a tiny corpus:

```
Corpus: "low lower lowest newer"

Step 0: vocab = {l, o, w, e, r, s, t, n, ' '}        (9 tokens)
Step 1: most frequent pair = (l, o) → merge to "lo"
        vocab += {"lo"}                              (10 tokens)
Step 2: most frequent pair = (lo, w) → merge to "low"
        vocab += {"low"}                             (11 tokens)
Step 3: most frequent pair = (e, r) → merge to "er"
        vocab += {"er"}                              (12 tokens)
...
```

**Training BPE with HuggingFace tokenizers**:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=2000,
    special_tokens=["<blank>", "<pad>", "<unk>", "<sos>", "<eos>"],
    min_frequency=2,
    show_progress=True,
)

# Train on the transcript corpus (plain text, one utterance per line)
tokenizer.train(files=["librispeech_transcripts.txt"], trainer=trainer)
tokenizer.save("bpe_2000.json")

# Usage
ids = tokenizer.encode("the quick brown fox").ids
# e.g., [145, 892, 1423, 67]  — 4 tokens for 4 words (all common)

ids = tokenizer.encode("antidisestablishmentarianism").ids
# e.g., [12, 438, 1891, 54, 33]  — broken into subwords, no <unk>
```

### SentencePiece (Unigram Language Model)

SentencePiece treats the input as a **raw byte stream** (no pre-tokenization on whitespace), which is important for:
- Languages without spaces (Chinese, Japanese, Thai)
- Multilingual models (consistent behavior across scripts)
- Preserving leading spaces as part of tokens (the `▁` marker)

It uses the **unigram language model** algorithm: start with a large candidate vocabulary, then iteratively remove tokens that contribute least to the corpus likelihood until the target size is reached. At inference, tokenization is the Viterbi-best segmentation under the learned unigram LM.

```python
import sentencepiece as spm

# Training
spm.SentencePieceTrainer.train(
    input="transcripts.txt",
    model_prefix="spm_5000",
    vocab_size=5000,
    model_type="unigram",             # or "bpe"
    character_coverage=1.0,           # 1.0 for Latin, 0.9995 for CJK
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    user_defined_symbols=["<blank>"], # CTC blank as id=4
    normalization_rule_name="nmt_nfkc_cf",  # case-fold + NFKC normalization
)

# Inference
sp = spm.SentencePieceProcessor(model_file="spm_5000.model")
ids = sp.encode("I love NYC")
# → [8, 124, 43, 9, 1827]  (▁I, ▁love, ▁N, Y, C)
text = sp.decode(ids)                  # "I love NYC"
```

The leading `▁` (U+2581) marks word boundaries, so detokenization is exact and reversible — critical when you need to reconstruct casing, spacing, and punctuation faithfully.

### Special Tokens by Architecture

| Token | CTC | Transducer | Attention |
|-------|-----|-----------|-----------|
| `<blank>` | Required (alignment) | Required (emit vs advance) | Not used |
| `<pad>` | Batch padding | Batch padding | Batch padding |
| `<sos>` / `<bos>` | No | Prediction network init | Decoder start |
| `<eos>` | No | Emission-end signal | Decoder stop condition |
| `<unk>` | Rare (char-level has none) | Rare | Rare |

**Critical detail**: For CTC, the blank token must be reserved (typically at index 0) and **excluded from the target sequence** passed to the loss — CTC targets contain only the real characters; the blank is inserted implicitly by the forward-backward algorithm.

```python
# Correct CTC target preparation
targets = tokenizer.encode("hello world")  # [h, e, l, l, o, ' ', w, o, r, l, d]
# DO NOT insert <blank> into targets — CTC loss handles that

# For attention decoder, prepend <sos> and append <eos>
decoder_input = [SOS] + targets              # teacher-forcing input
decoder_target = targets + [EOS]             # labels (shifted by 1)
```

### The Whisper Tokenizer

Whisper uses a **multilingual BPE tokenizer with 50,257 tokens** (extending GPT-2's tokenizer) plus **~1,500 special tokens** for language IDs, task markers, and timestamps. This is a much larger vocab than typical ASR models because Whisper covers 99 languages in a single model.

Key special tokens:

```
<|startoftranscript|>     — begin sequence
<|en|>, <|zh|>, <|es|>... — 99 language tokens
<|transcribe|>            — task: transcription
<|translate|>             — task: translation to English
<|notimestamps|>          — disable timestamp prediction
<|0.00|>, <|0.02|>, ...   — timestamp tokens (20ms granularity, 1500 of them)
<|endoftext|>             — end sequence
```

A Whisper training target for 5-second English audio with timestamps looks like:

```
<|startoftranscript|> <|en|> <|transcribe|> <|0.00|>
  Hello_world <|2.40|> <|2.60|> this_is_a_test <|5.00|>
<|endoftext|>
```

```python
from transformers import WhisperTokenizer

tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="en", task="transcribe")

# Encode with special tokens
ids = tok("Hello world", add_special_tokens=True).input_ids
# → [50258, 50259, 50359, 50363, 15947, 1002, 50257]
#     sot   <en>   transcribe notimestamps  hello world eot

# Decode, skipping special tokens for display
tok.decode(ids, skip_special_tokens=True)  # "Hello world"
```

**Fine-tuning Whisper on a new language**: You typically do NOT retrain the tokenizer. The BPE vocab is large enough that even new languages get reasonable subword coverage through byte-level fallback. Retraining breaks compatibility with the pretrained embedding matrix (which is where most of the learned knowledge lives).

### Vocabulary Size Trade-offs

Choosing vocab size is an empirical trade-off — there's no single right answer:

| Factor | Small vocab (500-1K) | Large vocab (8K-50K) |
|--------|---------------------|----------------------|
| Output head parameters | Tiny (hidden × 1K) | Large (hidden × 50K) |
| Sequence length $U$ | Long (more tokens/word) | Short |
| Transducer lattice memory | High (big $T \times U$) | Lower |
| OOV robustness | Excellent (fine-grained pieces) | Worse (rare words fragmented) |
| Attention decoder speed | Slow (more steps) | Fast |
| Overfitting risk | Low | Higher (more params) |
| Low-resource fine-tune | Better | Worse (embeddings undertrained) |

**Empirical defaults that work well**:
- LibriSpeech-scale English (960h): **BPE 1024-5000**
- Conformer-CTC production: **BPE 128-500** (smaller favors streaming)
- Conformer-Transducer production: **BPE 1024** (balances lattice cost)
- Whisper-style multilingual: **BPE 50K+** (necessary for language coverage)

### Tokenizer Training: Best Practices

1. **Train on transcripts only**, not on general text corpora. The ASR vocabulary should match the distribution of spoken language (short utterances, contractions, filler words).

2. **Normalize before training**: Lowercase, strip unusual punctuation, expand numerics ("2024" → "twenty twenty four"), and decide whether to keep punctuation at all (CTC models usually omit it; Whisper keeps it).

3. **Reserve special token IDs upfront**. Changing the blank or pad ID after training breaks checkpoints. A robust convention: `<blank>=0, <pad>=1, <unk>=2, <sos>=3, <eos>=4`, real tokens start at 5.

4. **Match casing policy to your users**. Lowercase-only gives lower WER on benchmarks but produces ugly output. Mixed-case tokenizers roughly double the effective vocab for letters but provide readable transcripts without a separate truecasing step.

5. **Guard against tokenization drift** between training and inference. Save the exact tokenizer file with every model checkpoint — a vocab mismatch silently produces garbage predictions.

```python
# Recommended checkpoint bundle
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "step": step,
    "config": config,
    "tokenizer_path": "bpe_2000.json",  # ship alongside the checkpoint
    "tokenizer_hash": sha256_of_file("bpe_2000.json"),  # verify at load
}, "checkpoint.pt")
```

## Data Pipeline

### Datasets

| Dataset | Hours | Languages | Type | Access |
|---------|-------|-----------|------|--------|
| LibriSpeech | 960 | English | Read speech | Free |
| Common Voice | 30,000+ | 100+ languages | Crowd-sourced | Free |
| GigaSpeech | 10,000 | English | Diverse (YouTube, podcasts) | Free |
| VoxPopuli | 400K (unlabeled) + 1.8K (labeled) | 23 EU languages | European Parliament | Free |
| SPGISpeech | 5,000 | English | Financial earnings calls | Free |
| MLS (Multilingual LibriSpeech) | 50,000 | 8 languages | Read audiobooks | Free |
| Whisper training data | 680,000 | 99 languages | Web-scraped | Not public |

### Data Preparation Pipeline

```python
import torch
from torch.utils.data import Dataset, DataLoader

class ASRDataset(Dataset):
    def __init__(self, audio_paths, transcripts, sample_rate=16000, max_duration=30.0):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.feature_extractor = AudioFeatureExtractor()
        self.tokenizer = CharTokenizer()  # or BPE tokenizer
    
    def __getitem__(self, idx):
        # Load and preprocess audio
        waveform, sr = torchaudio.load(self.audio_paths[idx])
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Mix to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Truncate to max duration
        waveform = waveform[:, :self.max_samples]
        
        # Extract features
        features = self.feature_extractor(waveform)  # (T, 80)
        
        # Tokenize transcript
        tokens = self.tokenizer.encode(self.transcripts[idx])
        
        return {
            "features": features,
            "feature_length": features.size(0),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "token_length": len(tokens),
        }
    
    def __len__(self):
        return len(self.audio_paths)


def collate_fn(batch):
    """Pad sequences to the longest in the batch."""
    features = torch.nn.utils.rnn.pad_sequence(
        [b["features"] for b in batch], batch_first=True
    )
    tokens = torch.nn.utils.rnn.pad_sequence(
        [b["tokens"] for b in batch], batch_first=True, padding_value=-1
    )
    feature_lengths = torch.tensor([b["feature_length"] for b in batch])
    token_lengths = torch.tensor([b["token_length"] for b in batch])
    
    return features, feature_lengths, tokens, token_lengths
```

### Data Augmentation

Data augmentation is critical for ASR — models overfit quickly without it.

**SpecAugment** (Park et al., 2019): The most important augmentation for ASR. Applies masking directly to the mel spectrogram:

```python
class SpecAugment:
    def __init__(self, freq_mask_param=27, time_mask_param=100, 
                 num_freq_masks=2, num_time_masks=2):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spec):
        # spec: (T, n_mels) or (n_mels, T)
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec)
        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec)
        return spec
```

**Other augmentations**:
- **Speed perturbation**: Randomly speed up/slow down by 0.9x-1.1x. Changes pitch and duration.
- **Noise injection**: Add background noise (room noise, babble, music) at random SNR (5-20 dB).
- **Room impulse response (RIR) simulation**: Convolve with RIR to simulate reverb/echo.
- **Volume perturbation**: Random gain adjustment (±10 dB).

## Training Recipe

### Complete Training Configuration

```yaml
# config.yaml — Conformer-CTC training
model:
  encoder: conformer
  num_layers: 12
  d_model: 512
  num_heads: 8
  conv_kernel_size: 31
  ff_expansion: 4
  dropout: 0.1
  
  # Subsampling: 2 Conv layers with stride 2 → 4x downsampling
  # 10ms frames → 40ms per encoder output
  subsample: conv2d  # or vgg
  subsample_factor: 4
  
  decoder: ctc  # or transducer, attention
  vocab_size: 5000  # BPE vocabulary

data:
  train_datasets:
    - librispeech_960h
    - common_voice_en
  sample_rate: 16000
  max_duration: 30.0
  min_duration: 0.5
  features: log_mel_80
  augmentation:
    spec_augment: true
    speed_perturb: [0.9, 1.0, 1.1]
    noise_injection: true
    noise_snr_range: [5, 20]

training:
  batch_size: 32
  accumulate_grad_batches: 4
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 200000
  lr_scheduler: warmup_cosine
  precision: bf16
  gradient_clip: 5.0
  
  # Dynamic batching by audio duration
  batch_duration: 120  # seconds of audio per batch
```

### The Training Loop

```python
def train_asr(model, train_loader, optimizer, scheduler, config):
    model.train()
    scaler = torch.amp.GradScaler()
    
    for step, (features, feat_lens, tokens, tok_lens) in enumerate(train_loader):
        features = features.cuda()
        tokens = tokens.cuda()
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if config.decoder == "ctc":
                log_probs = model(features, feat_lens)
                loss = ctc_loss(log_probs, tokens, feat_lens // 4, tok_lens)
            
            elif config.decoder == "attention":
                logits = model(features, tokens[:, :-1])  # teacher forcing
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    ignore_index=-1,
                )
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        
        if step % 2000 == 0:
            evaluate(model, val_loader)
```

### Dynamic Batching

ASR data has highly variable lengths (0.5s to 30s+). Static batch sizes waste memory (short utterances padded to the longest). **Dynamic batching** groups utterances by duration:

```python
class DurationBatchSampler:
    """Sample batches with roughly equal total audio duration."""
    
    def __init__(self, durations, max_batch_duration=120, shuffle=True):
        self.durations = durations
        self.max_duration = max_batch_duration
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.durations)))
        if self.shuffle:
            random.shuffle(indices)
        
        batch = []
        batch_duration = 0
        
        for idx in indices:
            dur = self.durations[idx]
            if batch_duration + dur > self.max_duration and len(batch) > 0:
                yield batch
                batch = []
                batch_duration = 0
            batch.append(idx)
            batch_duration += dur
        
        if batch:
            yield batch
```

## Evaluation: WER and CER

### Word Error Rate (WER)

The standard ASR metric. Measures the edit distance between the predicted and reference transcripts at the word level:

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

Where:
- $S$ = substitutions (wrong words)
- $D$ = deletions (missing words)
- $I$ = insertions (extra words)
- $N$ = total words in the reference

```python
import jiwer

reference = "the cat sat on the mat"
hypothesis = "the cat sit on a mat"

wer = jiwer.wer(reference, hypothesis)
# Alignment:
#   REF: the cat sat on the mat
#   HYP: the cat sit on  a  mat
#              ^^^      ^^^
#              S=1      S=1     → WER = 2/6 = 33.3%
```

### Character Error Rate (CER)

Same formula but at the character level. More meaningful for languages without clear word boundaries (Chinese, Japanese, Thai) or when WER is very high:

$$\text{CER} = \frac{S_\text{char} + D_\text{char} + I_\text{char}}{N_\text{char}} \times 100\%$$

### Decoding Strategies

**Greedy decoding** (CTC): Take the argmax at each frame, then apply CTC collapsing. Fast but suboptimal.

**Beam search** (CTC): Maintain top-$k$ hypotheses, merging paths that produce the same text. Significantly better than greedy.

**Beam search with language model** (CTC/RNN-T): Integrate an external language model during decoding:

$$\text{score}(y) = \log P_\text{AM}(y|x) + \alpha \log P_\text{LM}(y) + \beta |y|$$

Where $\alpha$ is the LM weight and $\beta$ is a word insertion bonus (prevents the model from preferring short outputs).

```python
# CTC beam search with language model (using pyctcdecode)
from pyctcdecode import build_ctcdecoder

decoder = build_ctcdecoder(
    labels=vocab,
    kenlm_model_path="4gram_lm.arpa",  # n-gram language model
    alpha=0.5,   # LM weight
    beta=1.0,    # word insertion bonus
)

# logits: (T, vocab_size) — CTC output
text = decoder.decode(logits.numpy())
```

## Streaming ASR

For real-time applications, the model must output text **while the user is still speaking**. This constrains the architecture:

- **CTC**: Naturally streamable — process chunks of audio and decode incrementally
- **RNN-T**: Streamable — encoder processes audio frames as they arrive, prediction network generates tokens
- **Attention (AED)**: **Not streamable** — decoder attends to the full encoder output

### Chunked Streaming

Process audio in fixed-size chunks (e.g., 640ms) with limited future context:

```python
class StreamingASR:
    def __init__(self, model, chunk_size_ms=640, lookahead_ms=320):
        self.model = model
        self.chunk_frames = chunk_size_ms // 10  # 64 frames
        self.lookahead = lookahead_ms // 10      # 32 frames
        self.buffer = []
        self.state = None
    
    def process_chunk(self, audio_chunk):
        """Process one chunk of audio and return partial transcript."""
        features = extract_features(audio_chunk)  # (chunk_frames, 80)
        self.buffer.append(features)
        
        # Use current chunk + limited lookahead from next chunk
        context = torch.cat(self.buffer[-3:], dim=0)  # last 3 chunks for context
        
        output, self.state = self.model.streaming_forward(
            context, self.state
        )
        
        return decode_ctc(output)
```

### Latency vs Accuracy Trade-off

| Configuration | Latency | WER Impact |
|--------------|---------|------------|
| Full-context (offline) | N/A | Baseline |
| Chunk 1280ms + 640ms lookahead | ~1.9s | +0.5-1% |
| Chunk 640ms + 320ms lookahead | ~1.0s | +1-3% |
| Chunk 320ms + 160ms lookahead | ~0.5s | +3-7% |
| Frame-by-frame (no lookahead) | ~40ms | +5-15% |

## Interview Questions and Answers

### Q: What is the difference between CTC, RNN-T, and Attention-based ASR models?

**CTC (Connectionist Temporal Classification)**: Encoder-only. Outputs one prediction per audio frame, using a special blank token for alignment. Collapses repeated characters and blanks to produce text. Key limitation: assumes conditional independence between frame outputs — each frame's prediction is independent given the encoder output. Fast, streamable, but needs an external language model for best accuracy.

**RNN-T (Transducer)**: Encoder + prediction network + joint network. The prediction network conditions on previously emitted tokens, solving CTC's independence problem. Operates on a 2D lattice of (audio frame, text position) pairs. Streamable, better accuracy than CTC without an external LM, but more complex to train (memory-intensive 2D lattice loss).

**Attention-based (AED)**: Full encoder-decoder with cross-attention. The decoder autoregressively generates tokens while attending to the entire encoded audio. Highest accuracy because it can learn flexible, non-monotonic alignments. Standard cross-entropy training (simplest). But **not streamable** — requires the full audio before decoding. Used by Whisper.

### Q: Explain the CTC loss function. How does the forward-backward algorithm compute it?

CTC defines a mapping $\mathcal{B}$ from frame-level CTC paths (with blanks and repeated characters) to text. The CTC loss is the negative log-probability of the target text, summed over all valid CTC paths:

$$L = -\log P(y|x) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T P(\pi_t|x_t)$$

Direct enumeration is intractable (exponentially many paths). The **forward-backward algorithm** computes this efficiently in $O(T \times |y|)$ using dynamic programming:

- Define $\alpha(t, s)$ = total probability of all CTC paths that output the first $s$ characters of $y$ at frame $t$
- Recurrence: at each $(t, s)$, the path either stayed at the same character, moved from blank, or moved from the previous character
- $P(y|x) = \alpha(T, |y'|)$ where $y'$ is $y$ with blanks inserted between every character

The backward pass computes $\beta(t, s)$ similarly from right to left. The gradient at each frame uses both $\alpha$ and $\beta$ (like Forward-Backward in HMMs).

### Q: What is SpecAugment and why is it so important for ASR?

SpecAugment applies two types of masking directly to the log-mel spectrogram during training:

1. **Frequency masking**: Randomly zero out $f$ consecutive mel bins (e.g., $f \leq 27$ out of 80). Forces the model to not rely on any single frequency band.
2. **Time masking**: Randomly zero out $t$ consecutive time frames (e.g., $t \leq 100$). Forces the model to handle missing audio segments.

Why it's so effective:
- Regularization that's specific to the audio domain — random dropout doesn't capture the structured nature of spectrograms
- Simulates real-world conditions: frequency masking approximates filtered audio, time masking approximates interruptions
- Computationally free — just zeroing out regions, no additional data processing
- The original paper showed 10-20% relative WER reduction across benchmarks

SpecAugment is used in virtually every modern ASR system and is considered mandatory, not optional.

### Q: Explain the Conformer architecture. Why does it outperform pure Transformers for ASR?

The Conformer interleaves self-attention with depthwise separable convolutions in each block: FFN(½) → Attention → Convolution → FFN(½) → LayerNorm.

**Why convolutions help**: Speech has strong **local patterns** — phoneme boundaries, formant transitions, pitch contours — that span 10-100ms. Self-attention is global and treats all positions equally, requiring many layers to learn these local patterns. Convolutions directly capture them with a local receptive field (kernel size 31 ≈ 310ms). The combination gives both local feature extraction (convolution) and global context (attention).

**Why the "macaron" FFN structure**: Two half-step FFN modules before and after the attention+conv block. Empirically this works better than a single full FFN, likely because it provides two separate nonlinear transformations that bracket the core attention/conv computation.

### Q: How does Whisper work? What makes it different from traditional ASR models?

Whisper is a standard Transformer encoder-decoder trained on **680,000 hours of weakly supervised audio-text pairs** scraped from the internet. Its key innovations are not architectural but in the training paradigm:

1. **Massive weakly supervised data**: Instead of clean, manually transcribed data, Whisper uses web-scraped audio with approximate transcripts. Quality is lower per-sample but the sheer scale (10x more data than any previous ASR system) compensates.

2. **Multitask training**: A single model handles transcription, translation, language ID, and timestamp prediction, controlled by special tokens (`<|en|>`, `<|transcribe|>`, `<|translate|>`, `<|notimestamps|>`).

3. **Robustness**: By training on diverse web audio (podcasts, YouTube, lectures, noisy environments), Whisper is far more robust to real-world conditions than models trained on clean read speech.

4. **Zero-shot generalization**: Whisper achieves competitive WER on benchmarks it was never explicitly trained on, approaching the performance of specialized fine-tuned models.

### Q: What is the difference between WER and CER? When would you use each?

**WER (Word Error Rate)**: Edit distance at the word level. $\text{WER} = (S + D + I) / N$ where S=substitutions, D=deletions, I=insertions, N=total reference words.

**CER (Character Error Rate)**: Same formula but at the character level.

Use **WER** for: languages with clear word boundaries (English, Spanish, German). It's the standard metric and directly measures how many words the user would need to correct.

Use **CER** for: languages without spaces between words (Chinese, Japanese, Thai) where "word" is ambiguous and depends on the segmentation algorithm. Also useful when WER is very high (>50%) — CER provides more granular signal because it can give partial credit for nearly-correct words.

**Gotcha**: WER can exceed 100% (if insertions outnumber correct words). CER is always more optimistic than WER because getting most characters right in a word still counts as a word error in WER.

### Q: How do you build a streaming ASR system? What are the trade-offs?

A streaming system must output text while audio is still arriving. This requires:

1. **Architecture choice**: CTC or RNN-T (not attention-based, which needs full audio). RNN-T is preferred for quality; CTC for simplicity.

2. **Chunked processing**: Audio is processed in fixed-size chunks (typically 320-1280ms). The encoder processes each chunk with limited future context (lookahead). Smaller chunks = lower latency but higher WER.

3. **Encoder modification**: Standard self-attention sees the full sequence. For streaming, use **causal attention** (each frame only attends to past frames) or **chunk-based attention** (attend within current + previous chunks).

4. **Endpointing**: Detect when the user has stopped speaking. Too aggressive = cuts off mid-sentence. Too conservative = long pauses before response. Typically uses a voice activity detector (VAD) + silence duration threshold.

**Trade-offs**: Latency vs accuracy is the fundamental trade-off. More lookahead and larger chunks improve accuracy (the model sees more context) but increase latency. Production systems typically target 500ms-1s latency with 1-3% WER degradation vs offline models.

### Q: How do self-supervised ASR models (Wav2Vec 2.0, HuBERT) work? When would you use them?

Both follow a two-stage approach:

**Stage 1 — Self-supervised pretraining on unlabeled audio**: Learn general speech representations without any transcripts. Wav2Vec 2.0 uses contrastive learning (predict the correct quantized latent from distractors). HuBERT uses offline clustering to create pseudo-labels, then trains a masked prediction task.

**Stage 2 — Fine-tuning with labeled data**: Add a CTC head and fine-tune on a small amount of transcribed audio. Wav2Vec 2.0 achieved usable ASR with only 10 minutes of labeled data.

**When to use them**:
- **Low-resource languages**: When you have thousands of hours of unlabeled audio but only 1-10 hours of transcribed data
- **Domain adaptation**: Pretrain on unlabeled in-domain audio (e.g., medical dictation), then fine-tune with small labeled set
- **Feature extraction**: Use the pretrained encoder as a frozen feature extractor for downstream tasks (emotion recognition, speaker ID)

**When NOT to use them**: If you have abundant labeled data (>1000 hours), Whisper-style supervised pretraining typically outperforms self-supervised approaches. The self-supervised advantage disappears when labeled data is plentiful.

### Q: What are the most important hyperparameters for ASR training?

| Parameter | Typical Range | Impact |
|-----------|--------------|--------|
| Learning rate | 1e-4 to 1e-3 (from scratch), 1e-5 to 5e-5 (fine-tuning) | Most critical — too high causes divergence, too low wastes compute |
| Warmup steps | 5K-25K | Essential — ASR models are sensitive to early training instability |
| SpecAugment freq masks | 2 masks, width ≤27 | High impact — mandatory for regularization |
| SpecAugment time masks | 2-10 masks, width ≤100 | High impact — more aggressive = more robust but slower convergence |
| Gradient clipping | 1.0-5.0 | Important for CTC/transducer (loss can spike) |
| Batch strategy | Duration-based (60-120s per batch) | Better than fixed batch size for variable-length audio |
| Speed perturbation | 0.9x, 1.0x, 1.1x | 5-10% relative WER improvement, almost free |
| Encoder downsampling | 4x (standard) or 8x (faster) | Trade-off: accuracy vs speed and memory |

### Q: How would you deploy an ASR model in production?

**Model selection**: Start with Whisper large-v3-turbo (809M params, good accuracy, 3x faster than large-v3). Fine-tune on your domain if needed.

**Inference optimization**:
- **FP16/BF16 inference**: 2x memory reduction, ~1.5x speedup
- **CTranslate2 or faster-whisper**: Optimized Whisper inference (4-8x faster than HuggingFace)
- **Batched inference**: Process multiple audio files in parallel
- **VAD preprocessing**: Use Silero VAD to detect speech segments, skip silence
- **Chunking**: For long audio (>30s), chunk into 30s segments with overlap

**Streaming**: If real-time needed, use Conformer-Transducer (NVIDIA NeMo) or streaming Whisper variants.

**Infrastructure**:
```python
# Production setup with faster-whisper
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    "audio.wav",
    beam_size=5,
    language="en",
    vad_filter=True,         # skip silence
    word_timestamps=True,    # per-word timing
)

for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
```

**Monitoring**: Track WER on a held-out test set regularly (distribution shift), latency P99, and throughput (audio-seconds processed per second — target >1x for real-time).

### Q: Compare CTC, Transducer, and Attention for a production ASR system.

| Criterion | CTC | Transducer | Attention (Whisper) |
|-----------|-----|-----------|-------------------|
| Streaming | Yes | Yes | No |
| Accuracy (no ext. LM) | Moderate | Good | Best |
| Accuracy (with ext. LM) | Good | Very good | Best |
| Inference speed | Fastest | Medium | Slowest |
| Training complexity | Simple | Complex (2D lattice) | Simple (cross-entropy) |
| On-device deployment | Best | Good | Poor (too large/slow) |
| Multilingual | Needs per-language model | Needs per-language model | Single model (Whisper) |
| Best use case | On-device, streaming | Production streaming | Offline, highest quality |

**Decision framework**:
- **Need streaming**: CTC (simplest) or Transducer (best quality)
- **Need highest accuracy, offline OK**: Whisper/Attention
- **On-device / edge**: CTC with small Conformer (~30M params)
- **Multilingual**: Whisper (covers 99 languages in one model)
- **Low-resource language**: Wav2Vec 2.0 pretrain + CTC fine-tune

## References

1. Graves, A., et al. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." ICML 2006.
2. Graves, A. "Sequence Transduction with Recurrent Neural Networks." ICML Workshop 2012.
3. Chan, W., et al. "Listen, Attend and Spell." ICASSP 2016.
4. Gulati, A., et al. "Conformer: Convolution-augmented Transformer for Speech Recognition." Interspeech 2020.
5. Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." Interspeech 2019.
6. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)." ICML 2023.
7. Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS 2020.
8. Hsu, W., et al. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." IEEE/ACM TASLP 2021.
9. He, Y., et al. "Streaming End-to-End Speech Recognition for Mobile Devices." ICASSP 2019.
10. NVIDIA NeMo ASR Documentation. https://docs.nvidia.com/nemo/asr/
