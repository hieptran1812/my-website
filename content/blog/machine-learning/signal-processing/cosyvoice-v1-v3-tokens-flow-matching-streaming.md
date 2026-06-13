---
title: "Inside the CosyVoice Series: Supervised Speech Tokens, Flow Matching, and Streaming TTS from v1 to v3"
date: "2026-06-13"
publishDate: "2026-06-13"
description: "A technique-by-technique deep dive through three CosyVoice papers — supervised S3 tokens, FSQ, optimal-transport flow matching, unified streaming, and DiffRO post-training — and the engineering trade-offs behind each decision."
tags:
  [
    "cosyvoice",
    "text-to-speech",
    "speech-synthesis",
    "flow-matching",
    "fsq",
    "tokenizer",
    "streaming",
    "reinforcement-learning",
    "voice-cloning",
    "llm",
    "audio",
    "tts",
  ]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most engineers meet a text-to-speech system as a black box: text goes in, a `.wav` comes out, and the only knob anyone touches is the voice preset. That framing was roughly true in the Tacotron era. It is badly wrong for the generation of systems that CosyVoice belongs to — and the gap between the two mental models is exactly where the interesting engineering lives.

The CosyVoice family from Alibaba's Tongyi Speech Lab is a good case to study because it ships as three papers across roughly eighteen months — [CosyVoice 1](https://arxiv.org/abs/2407.05407), [CosyVoice 2](https://arxiv.org/abs/2412.10117), and [CosyVoice 3](https://arxiv.org/abs/2505.17589) — and each one changes exactly one or two load-bearing components while keeping the skeleton fixed. That makes it a controlled experiment in TTS systems design. You can watch a single decision — the speech tokenizer, the streaming scheme, the post-training objective — get isolated, swapped, and measured. This post walks that arc technique by technique, and at every step asks the three questions that matter for a staff-level read: why does this work, when does it fail, and what is the second-order consequence nobody mentions in the abstract.

If you want the companion "here is how you would actually train one" walkthrough with interview questions, I wrote that separately in [Training CosyVoice](/blog/machine-learning/deep-learning/training-cosyvoice). This piece is the opposite lens: a retrospective on *why* the architecture is shaped the way it is, with CosyVoice 3's net-new techniques — a multi-task tokenizer, differentiable reward optimization, and million-hour scaling — as the centerpiece.

## Why a modern TTS stack breaks the black-box assumption

Here is the mismatch that trips people up. Everyone's intuition for "the hard part of TTS" is the vocoder — turning a spectrogram into audio. In a CosyVoice-class system the vocoder is the *least* interesting component. The hard parts moved up the stack.

| What you assume | The naive view | What CosyVoice actually does |
| --- | --- | --- |
| The vocoder is the bottleneck | A good neural vocoder makes good speech | HiFi-GAN is a fixed, almost stock component; quality is set upstream |
| Tokens are an implementation detail | Any discrete code of the audio will do | The *token* is the central design object — supervised vs unsupervised changes everything |
| Streaming is a serving-layer concern | Chunk the output, stream the bytes | Streaming is baked into the token *sequence* and the decoder's *attention mask* |
| Prosody comes from the acoustic model | The spectrogram model decides intonation | Prosody is decided by the autoregressive LM; the acoustic model only paints timbre |
| Quality plateaus with data | More hours, better speech | Past a point, post-training (RL) moves the needle more than raw data |

Every row of that table is a decision CosyVoice got to make, measure, and sometimes get wrong on the first try. The rest of this article is a tour of those decisions.

## The mental model: a two-stage semantic-to-acoustic hybrid

![CosyVoice mental model: text feeds an autoregressive LM that emits 25 Hz speech tokens, which a flow-matching decoder turns into a Mel spectrogram, which a HiFi-GAN vocoder turns into a 24 kHz waveform](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-1.webp)

The diagram above is the mental model, and the entire series is a tour of it. Read it left to right, top row then bottom: text is tokenized, an autoregressive language model converts text tokens into a sequence of discrete *speech tokens*, a flow-matching model turns those tokens into a Mel spectrogram, and a HiFi-GAN vocoder turns the Mel into a waveform. Two stages, cleanly split by responsibility.

The split is the whole idea, and it survives unchanged across all three papers. The language model owns **semantics**: what words are said, in what order, with what rhythm and emphasis. The flow-matching decoder owns **acoustics**: the timbre of the specific speaker, the room, the microphone. The speech token in the middle is the contract between them — a 25 Hz stream (25 tokens per second of audio) that is rich enough to reconstruct intelligible, well-prosodied speech but stripped of speaker identity.

That decoupling buys three concrete things:

- **Zero-shot voice cloning is a conditioning trick, not a retraining job.** Because timbre is injected only at the acoustic stage, you clone a voice by handing the flow-matching model a three-second reference clip and a speaker embedding. The language model never sees the target speaker.
- **The two stages can use the generative model each is best at.** Autoregressive sampling is excellent at producing varied, natural prosody token-by-token. Flow matching is excellent at high-fidelity, parallel, few-step spectrogram generation. You do not want one model doing both.
- **Errors localize.** When content is wrong (a dropped word, a hallucinated syllable), it is almost always the LM. When timbre drifts or the room sounds off, it is the flow-matching model. The CosyVoice 2 ablations confirm this cleanly, and we will use it as a debugging principle later.

Hold onto the semantic/acoustic split. Half the design decisions in the series are downstream of asking "which stage should own this?" and the answer is rarely the obvious one — the speaker embedding, for instance, started life in the language model and got evicted to the acoustic stage in version 2, with measurable gains.

## 1. The bet that started it all: supervised semantic tokens

Every LLM-based TTS system has to answer one question first: *what is a speech token?* You need to discretize continuous audio into a sequence an autoregressive model can predict. The pre-CosyVoice answer, from systems like VALL-E, was an unsupervised neural codec — train an autoencoder (EnCodec, SoundStream) to compress and reconstruct the waveform, and use its residual-VQ codes as tokens. Those codes are optimized for *reconstruction*. Nothing forces them to align with the text.

CosyVoice 1's central claim is that this is the wrong objective. If your downstream task is "predict tokens from text," you want tokens that already correlate with text. So instead of training a codec from scratch, CosyVoice takes a *supervised* automatic speech recognition (ASR) model and steals representations from inside it.

![A quantizer inserted between Encoder1 and Encoder2 of a supervised ASR model taps out speech tokens at 25 Hz; the rest of the ASR (Encoder2 and the text decoder) is used only at training time](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-2.webp)

The construction in the figure is the **S3 tokenizer** (Supervised Semantic Speech tokenizer). Take a strong multilingual ASR model — a fine-tuned SenseVoice-Large in the large-scale setup, an ESPnet Conformer in the small-scale ablations — and split its encoder in two. After the first six encoder layers, insert a quantizer. The first half plus the quantizer become the tokenizer; the second half plus the ASR text decoder are kept *only during training*, to supply the gradient signal that keeps the tokens semantically meaningful.

Mechanically, given a Mel spectrogram $X$, the first encoder produces a context-aware hidden sequence:

$$H = \text{Encoder}_1(\text{PosEnc}(X))$$

In version 1 the quantizer is a single-codebook vector quantizer (VQ) with 4096 entries. For each frame hidden vector $h_l$, the token is the index of its nearest codebook entry:

$$\mu_l = \text{VQ}(h_l, C) = \arg\min_{c_n \in C} \lVert h_l - c_n \rVert_2$$

and the codebook entries are updated during training with an exponential moving average (EMA), $c_{\mu_l} \leftarrow \alpha c_{\mu_l} + (1-\alpha) h_l$. The quantized vectors then flow through the remaining encoder layers and the ASR decoder, which predicts the transcript. That decoder loss is the entire point: it forces the codes to carry enough phonetic and lexical information to recover the words.

### Why this works: tokens that already know what was said

The evidence that the bet pays off is simple. Inserting a quantizer into an ASR encoder barely degrades recognition. On LibriTTS, the VQ-augmented Conformer goes from 2.89% to 3.18% word error rate (WER) on test-clean — a token that still recognizes speech almost as well as the un-quantized model is a token that preserves content. On Common Voice Chinese, the S3 tokens even *beat* Whisper-Large V3's recognition, a 4.14% relative error reduction, with a single 4096-entry codebook.

Compare that against unsupervised tokens on the downstream TTS task. CosyVoice 1's ablation swaps tokenizers while holding everything else fixed:

| Text token | Speech token | WER (%) | Speaker sim |
| --- | --- | --- | --- |
| Phone | HuBERT (unsupervised) | 7.41 | 0.679 |
| Phone | S3 (supervised) | 5.05 | 0.679 |
| BPE | S3 (supervised) | 3.93 | 0.679 |
| BPE | S3, large-scale data | 3.17 | 0.695 |

The jump from HuBERT to S3 at fixed everything-else is a 32% relative WER reduction. Speaker similarity is unchanged — because, remember, the token deliberately does *not* carry speaker identity; that is the flow-matching model's job. This is the first concrete payoff of the semantic/acoustic split: you can change the tokenizer and move content consistency without touching speaker similarity at all.

### The second-order consequence: you inherit the ASR model's blind spots

Supervised tokens are not free. Because the tokenizer is literally the front half of an ASR model, it inherits that model's failure modes. If the ASR backbone confuses two homophones, the tokens will too. If the ASR model was trained mostly on Chinese and English, the tokens will be weaker on Japanese — which is exactly the cross-lingual wall CosyVoice 2 hits and CosyVoice 3 has to engineer around. We will return to this in the case studies. The lesson to bank now: choosing a supervised tokenizer ties your synthesis quality to an ASR model's coverage, and "improve the tokenizer" later means "improve or replace the ASR backbone," which is precisely the move CosyVoice 3 makes when it jumps from SenseVoice to MinMo.

## 2. From vector quantization to finite scalar quantization

CosyVoice 1's tokenizer worked, but it had a quiet inefficiency that CosyVoice 2 measured and killed. A vector-quantized codebook of 4096 entries does not actually *use* 4096 entries. Most of them go dead.

![Before/after comparison: vector quantization uses 963 of 4096 codes (23 percent) with 18.3 percent ASR error, while finite scalar quantization uses all 6561 codes (100 percent) with 10.7 percent ASR error](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-3.webp)

The numbers in the figure are the headline. With VQ, only 963 of 4096 codes are ever selected — 23% utilization. The other 77% are dead weight: codebook entries that, early in training, lost the nearest-neighbor lottery and never got pulled back toward the data because EMA only updates codes that win. This is the classic codebook-collapse problem, and it means your "4096-entry codebook" is effectively a ~1000-entry codebook with a lot of wasted index space.

CosyVoice 2 replaces VQ with **finite scalar quantization** (FSQ), the technique from Mentzer et al.'s "VQ-VAE made simple." The idea is to throw away the learned codebook entirely and quantize each dimension of a low-rank projection onto a fixed integer grid. Given the encoder hidden $H$:

$$\bar{H} = \text{ROUND}(\text{Proj}_{\text{down}}(H)), \qquad \hat{H} = \text{Proj}_{\text{up}}(\bar{H})$$

The projection drops $H$ into a small $D$-dimensional space; each dimension is rounded to an integer in $[-K, K]$ with a bounded round operation; then it is projected back up. Gradients pass through the round with the straight-through estimator. The token index is read off as a number in a $(2K+1)$-ary system:

$$\mu_i = \sum_{j=0}^{D-1} \bar{h}_{i,j}\,(2K+1)^j$$

With $K=1$ and $D=8$ you get $(2 \cdot 1 + 1)^8 = 3^8 = 6561$ codes, and here is the property that matters: **every one of them is reachable**. There is no codebook to collapse. Each grid cell is a valid token by construction, so utilization is 100% by definition, not by training luck.

Here is the implementation, compact enough to drop into a tokenizer:

```python
import torch
import torch.nn as nn

class FSQ(nn.Module):
    """Finite scalar quantization. Codebook size = (2K+1) ** D, always 100% used."""
    def __init__(self, dim: int, d_low: int = 8, k: int = 1):
        super().__init__()
        self.k = k
        self.levels = 2 * k + 1                 # 3 levels per dim: {-1, 0, 1}
        self.proj_down = nn.Linear(dim, d_low)  # H -> low-rank
        self.proj_up = nn.Linear(d_low, dim)    # low-rank -> H
        # radix weights for the (2K+1)-ary index: [1, 3, 9, 27, ...]
        radix = self.levels ** torch.arange(d_low)
        self.register_buffer("radix", radix)

    def forward(self, h: torch.Tensor):
        z = self.proj_down(h)                          # (B, T, d_low)
        # bounded round to {-K, ..., K}, straight-through gradient
        z_q = torch.clamp(z, -self.k, self.k)
        z_q = z_q + (z_q.round() - z_q).detach()       # STE: forward rounds, backward is identity
        h_hat = self.proj_up(z_q)                      # (B, T, dim) -> rest of encoder
        # token index in [0, (2K+1)**d_low)
        idx = ((z_q + self.k) * self.radix).sum(-1)    # (B, T)
        return h_hat, idx.long()
```

### Why this matters beyond the utilization number

A fully-used codebook is not just tidier; it carries more information. CosyVoice 2's measurement: with the same SenseVoice-Large encoder, the FSQ tokenizer's ASR error on Common Voice English drops from 18.26% (VQ) to 10.67% (FSQ), and on Common Voice Chinese from 11.56% to 7.29%. More reachable codes means the tokenizer can encode finer phonetic and contextual distinctions, which means the language model has a richer, less ambiguous target to predict. In the end-to-end ablation, swapping VQ for FSQ takes SEED test-zh character error rate (CER) from 2.56% down to 1.45% — the single largest jump in the entire v1-to-v2 modular study.

There is a subtler win that the paper proves with a t-SNE plot and a probing experiment: FSQ tokens are *more* speaker-independent than VQ tokens. The authors take 100 clips from each of three VoxCeleb speakers and visualize the representations. Before quantization, the three speakers form distinct clusters; after FSQ quantization, the distributions are nearly indistinguishable. They then train a speaker-identification probe on the quantized tokens — and it fails to converge. That non-convergence is the goal: the tokenizer has scrubbed speaker identity out of the token, leaving it for the flow-matching model to repaint. This is the semantic/acoustic split being enforced *at the representation level*, and it is why removing the speaker embedding from the language model (next section) works without hurting speaker similarity.

The trade-off FSQ accepts: a fixed grid is less expressive *per code* than a learned codebook could be in principle. If you had a perfectly trained, fully-utilized VQ codebook, it could place codes exactly where the data density is. FSQ spreads codes uniformly on a grid regardless of data density. In practice, the collapse problem makes "perfectly trained VQ" a fiction, and the robustness of FSQ wins. CosyVoice 2 keeps FSQ; CosyVoice 3 keeps it too, just moves it into a stronger encoder.

## 3. The flow-matching decoder: optimal-transport conditional flow matching

The acoustic stage's job is to turn the 25 Hz token stream into a 50 Hz Mel spectrogram, conditioned on a target speaker. CosyVoice uses **optimal-transport conditional flow matching** (OT-CFM) for this, and the choice is deliberate: flow matching trains faster and generates in far fewer steps than the denoising diffusion models that earlier TTS systems (TorToiSe) used.

![Four conditions — a Gaussian prior, the speaker embedding, the speech tokens from the LM, and a masked reference Mel — fan into one vector-field network, which an Euler ODE solver runs for about ten steps to produce the Mel spectrogram](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-4.webp)

The figure shows the data flow. Four things feed a single vector-field network: a Gaussian prior sample $x_0 \sim \mathcal{N}(0, I)$, the speaker embedding $v$, the speech tokens $\mu$ from the language model, and a masked reference Mel $\tilde{X}$. An ODE solver integrates the learned vector field from the prior to a Mel spectrogram, which then goes to HiFi-GAN. Let me unpack the math, because the details are where flow matching earns its speed.

### The intuition, then the equations

Flow matching learns a velocity field that transports a simple distribution (Gaussian noise) to a complex one (Mel spectrograms). At training time you pick a random point in time $t \in [0,1]$, place a sample on the straight line between a noise sample $x_0$ and a real Mel $x_1$, and train a network to predict the velocity that would move it along that line. The optimal-transport path is a literal straight line:

$$\phi_t^{\text{OT}}(x_0, x_1) = (1 - (1-\sigma)t)\, x_0 + t\, x_1, \qquad \omega_t = x_1 - (1-\sigma)x_0$$

The network $v_\theta$ is trained to match that target velocity, conditioned on everything:

$$\mathcal{L} = \mathbb{E}_{t,\, x_0,\, x_1} \left\lVert \omega_t - v_\theta\big(\phi_t^{\text{OT}}(x_0, x_1),\, t;\, v, \mu, \tilde{X}\big) \right\rVert$$

At inference you start from pure noise and integrate the ODE $\frac{d}{dt}\phi_t = v_\theta(\phi_t, t)$ forward. Because the training target is a straight line, the field is nearly straight, and a handful of Euler steps suffice — CosyVoice uses **NFE = 10** (ten network function evaluations). Diffusion models of comparable quality often need 25 to 100. That 2.5-to-10x reduction in decoder calls is most of why CosyVoice can stream.

Here is the training step, stripped to its essentials:

```python
import torch

def ot_cfm_loss(net, x1, cond, sigma_min=1e-4):
    """One optimal-transport conditional flow-matching training step.
    x1:   (B, n_mel, T) ground-truth Mel
    cond: dict of conditions {v: speaker emb, mu: speech tokens, x_masked: ref Mel}
    """
    b = x1.size(0)
    t = torch.rand(b, device=x1.device)          # uniform t ~ U[0, 1] at train time
    t_ = t.view(b, 1, 1)
    x0 = torch.randn_like(x1)                     # prior sample N(0, I)
    # point on the straight OT path from noise to data
    xt = (1 - (1 - sigma_min) * t_) * x0 + t_ * x1
    target = x1 - (1 - sigma_min) * x0            # the OT velocity
    pred = net(xt, t, **cond)                     # vector field v_theta
    return torch.nn.functional.l1_loss(pred, target)
```

### The three tricks that make it sound good

Three modifications turn vanilla flow matching into something with human-parity quality:

**Masked-Mel conditioning.** During training, the reference Mel $\tilde{X}$ is the ground-truth Mel with 70%–100% of its final frames zeroed out. At inference, $\tilde{X}$ is the *reference speaker's* Mel. This teaches the model to continue a spectrogram in a consistent voice and room — it is in-context learning for acoustics. The speaker embedding $v$ gives a global timbre vector; the masked Mel gives local acoustic texture. Together they pin down the voice.

**Cosine timestep schedule.** Generation is hardest at the start, when the sample is pure noise. So inference does not space the ten Euler steps uniformly in $t$; it uses a cosine schedule $t := 1 - \cos(\frac{1}{2} t\pi)$ that packs more steps near $t=0$. More compute where the field is most curved.

**Classifier-free guidance (CFG).** During training, the conditions $\Psi = \{v, \mu, \tilde{X}\}$ are dropped with probability 0.2, so the model learns both a conditional and an unconditional field. At inference, the two are combined to sharpen the result:

$$\tilde{v}_t = (1 + \beta)\, v_t(\,\cdot\,|\,\Psi) - \beta\, v_t(\,\cdot\,)$$

with guidance strength $\beta = 0.7$. Each step costs two network evaluations (conditional and unconditional), so "NFE = 10" with CFG is really 20 forward passes — still cheap next to diffusion. The whole inference loop is short enough to read in one block:

```python
import torch

@torch.no_grad()
def cfm_sample(net, cond, n_mel, T, nfe=10, beta=0.7, device="cuda"):
    """Sample a Mel spectrogram with OT-CFM + classifier-free guidance.
    cond: {v, mu, x_masked}; null_cond is the same with conditions dropped."""
    x = torch.randn(1, n_mel, T, device=device)            # x0 ~ N(0, I)
    # cosine schedule: pack more steps near t=0 where the field is most curved
    ts = 1 - torch.cos(0.5 * torch.linspace(0, 1, nfe + 1, device=device) * torch.pi)
    for i in range(nfe):
        t = ts[i].expand(1)
        dt = (ts[i + 1] - ts[i])
        v_cond = net(x, t, **cond)                          # conditional velocity
        v_null = net(x, t, **null_cond(cond))               # unconditional velocity
        v = (1 + beta) * v_cond - beta * v_null             # CFG extrapolation
        x = x + dt * v                                      # Euler step along the ODE
    return x                                                # -> HiFi-GAN
```

Two things to notice. The step sizes `dt` are unequal because the cosine schedule spaces `ts` non-uniformly — small steps early, larger steps late. And each iteration does two `net` calls, which is the CFG tax made concrete.

The second-order consequence worth flagging: CFG doubles your decoder compute. For an offline batch job that is fine. For streaming, where the flow-matching cost $d_{\text{fm}}$ is in the first-packet latency budget, it is a real tax — and it is one reason CosyVoice 3 moves to a more efficient DiT backbone and shrinks the gap. We will quantify the latency budget shortly.

## 4. Rebuilding the language model on a pretrained backbone

CosyVoice 1's language model was a from-scratch transformer with two auxiliary parts: a text encoder (to align text and speech token spaces) and an x-vector speaker embedding (to give the LM a sense of who is speaking). CosyVoice 2 deletes both and initializes from a pretrained text LLM. This is the single highest-leverage change in the v1-to-v2 jump, and it is worth understanding why each deletion helps.

![Before/after: CosyVoice v1's LM has a text encoder, a leaky x-vector speaker embedding, and random initialization, hitting 3.63 percent CER; v2 removes the text encoder and speaker embedding and uses a Qwen2.5-0.5B backbone, hitting 1.45 percent CER](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-5.webp)

The before/after is in the figure. Three changes, each measured in the ablation.

### Deletion 1: the text encoder

In version 1, raw text went through a BPE tokenizer and then a dedicated text *encoder* whose job was to project text into a space aligned with the speech tokens. The reasoning was that text and speech "live at different semantic levels" and need alignment before the LM can model them jointly.

Version 2's finding: a pretrained LLM does not need the help. Qwen2.5-0.5B has already learned rich text representations; feeding it raw BPE text and letting it learn the text-to-speech-token mapping end-to-end works better than pre-aligning with a bolt-on encoder. The text encoder is deleted. The model is simpler and the context understanding improves because you are now standing on top of a real language model's pretraining.

### Deletion 2: the speaker embedding (the leak)

This one is subtle and it is my favorite decision in the series. Version 1 fed an utterance-level x-vector speaker embedding $v$ into the language model, on the theory that the LM should know the speaker to produce appropriate prosody.

Version 2 removes it from the LM entirely — and *keeps* it only in the flow-matching model. The stated reason is that the utterance-level vector "contains not only speaker identity but also language and paralanguage information," and injecting that into the LM harms prosody naturalness and cross-lingual ability. In plain terms: the speaker embedding is a *leak*. It smuggles language and style information into the model that is supposed to be modeling pure semantics, and that contamination shows up as flatter prosody and worse cross-lingual transfer. Evicting the embedding to the acoustic stage — where timbre belongs — cleans up the LM's job. The ablation shows content error dropping while speaker similarity holds, confirming that "speaker information is mainly recovered by the flow matching model," exactly as the architecture intends.

### Deletion 3: random initialization

With the text encoder and speaker embedding gone, the LM is simple enough to *be* a pretrained LLM. CosyVoice 2 uses Qwen2.5-0.5B as the backbone. The ablation isolates this:

| Model | test-zh CER | test-en WER | test-hard CER |
| --- | --- | --- | --- |
| CosyVoice 1 (baseline) | 3.63 | 4.29 | 11.75 |
| + LLM initialization | 2.96 | 4.57 | 9.94 |
| + drop speaker embedding | 2.56 | 3.81 | 9.66 |
| + FSQ tokenizer (= CosyVoice 2) | 1.45 | 2.57 | 6.83 |
| + pitch loss (experimental) | 1.19 | 2.40 | 6.29 |

LLM initialization alone buys an 18% relative CER reduction on test-zh and a 15% reduction on test-hard. Dropping the speaker embedding adds more. FSQ adds the most. The table is a clean illustration of a principle: in a hybrid TTS stack, the wins come from *removing* the wrong inductive biases (the text encoder, the leaky embedding) and *inheriting* the right pretraining (the LLM), not from adding clever new modules.

### The token-tokenizer detail nobody mentions

One small but real engineering decision: CosyVoice 2 masks out multi-character BPE tokens. A standard text BPE tokenizer might encode a common two-character Chinese word as a single token. That is fine for a text LLM, but for TTS it creates a problem — the pronunciation of one token becomes "excessively long," and rare multi-character tokens create data-sparsity corner cases. So if a BPE token encodes more than one Chinese character, CosyVoice masks it and forces each character to be encoded separately. English, Japanese, and Korean are left alone. This is the kind of detail that does not make the abstract but absolutely matters when you are debugging why a specific phrase comes out garbled.

## 5. Streaming without a second model

The marquee feature of CosyVoice 2 is bidirectional streaming: the system can start emitting audio before it has seen the full text, with first-packet latency low enough for live voice chat. The elegant part is that streaming is not a separate model or a separate serving path. It is the *same* language model, fed a different token sequence.

![Two token tapes: non-streaming concatenates the start token, all text tokens, a turn-of-speech token, all speech tokens, and end; streaming interleaves 5 text tokens with 15 speech tokens repeatedly, with filling tokens marking where the next text batch splices in](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-6.webp)

The two tapes in the figure are the entire mechanism. In **non-streaming** mode, you build the sequence by concatenation: start-of-sequence, all text tokens, a turn-of-speech token, all speech tokens, end-of-sequence. The LM needs the whole text before it produces any speech. In **streaming** mode, you interleave: every $N$ text tokens are followed by $M$ speech tokens, repeating, with the parameters set to $N=5$, $M=15$. When the model would emit a text token but there are no more text tokens ready, it emits a *filling token* — a placeholder that says "splice the next $N$ text tokens in here at inference." Once text runs out, you append the turn-of-speech token and the remaining speech tokens.

That is it. Both sequences are trained simultaneously, so one set of weights learns both modes. There is no second model to deploy, no separate streaming checkpoint, no divergence between what you tested offline and what you serve live. Here is the sequence builder for the streaming case:

```python
def build_streaming_sequence(text_tokens, speech_tokens, n=5, m=15,
                             SOS="<sos>", TURN="<turn>", EOS="<eos>", FILL="<fill>"):
    """Interleave N text : M speech tokens into one unified LM sequence.
    Mirrors CosyVoice 2's ICL-streaming construction."""
    seq, ti, si = [SOS], 0, 0
    while ti < len(text_tokens) or si < len(speech_tokens):
        # emit up to N text tokens; pad with FILL if text has run out mid-stream
        if ti < len(text_tokens):
            chunk = text_tokens[ti:ti + n]
            ti += n
            seq.extend(chunk + [FILL] * (n - len(chunk)))
        # emit up to M speech tokens
        chunk = speech_tokens[si:si + m]
        si += m
        seq.extend(chunk)
    seq.append(TURN)                       # text exhausted -> switch to pure speech
    seq.extend(speech_tokens[si:])         # any remaining speech tokens
    seq.append(EOS)
    return seq
```

### The latency budget, written out

Why does this give low latency? Because the model emits a chunk of $M$ speech tokens after seeing only $N$ text tokens, you can start vocoding after the first chunk instead of after the whole utterance. CosyVoice 2 writes the first-packet latency as a sum of three per-token costs:

$$L_{\text{TTS}} = M \cdot d_{\text{lm}} + M \cdot d_{\text{fm}} + M \cdot d_{\text{voc}}$$

where $d_{\text{lm}}$ is the time for the LM to generate one speech token, $d_{\text{fm}}$ the flow-matching time per token of Mel, and $d_{\text{voc}}$ the vocoder time per token of waveform. With $M=15$, your first audio packet costs fifteen tokens through all three stages — a fixed, small number independent of utterance length. In a full voice-chat setting where an upstream text LLM is also generating, you add its cost for the first $N$ text tokens:

$$L_{\text{chat}} \le N \cdot d_{\text{llm}} + L_{\text{TTS}}$$

This formula is the reason the streaming design is shaped the way it is. Every term is something you can attack: shrink $M$ to cut latency (at some quality cost), make $d_{\text{fm}}$ smaller with a better decoder, make $d_{\text{voc}}$ smaller with a lighter vocoder. The streaming scheme turns latency into an arithmetic you can optimize rather than a property you measure and hope is acceptable.

To make the arithmetic concrete, plug in plausible numbers for a single GPU. Suppose the 0.5B language model emits one speech token in $d_{\text{lm}} \approx 4$ ms, the flow-matching decoder produces one token's worth of Mel in $d_{\text{fm}} \approx 3$ ms, and the vocoder turns one token of Mel into waveform in $d_{\text{voc}} \approx 1$ ms. With $M = 15$, the first audio packet costs $15 \times (4 + 3 + 1) = 120$ ms — and because $M=15$ tokens is 0.6 s of audio at 25 Hz, you have produced 600 ms of speech for a 120 ms wait, so the stream stays ahead of real time after the first packet. Now halve the chunk to $M = 8$ and the first packet drops to 64 ms at the cost of a thinner lookahead for the decoder; double it to $M = 30$ and you trade 240 ms of latency for steadier prosody on long sentences. That is the entire latency/quality dial, and it is a single integer you set per request — no redeploy, no separate model.

### Why streaming the LM is riskier than streaming the decoder

The streaming sequence has a cost: in streaming mode the LM sees less future context when predicting each speech token, which can hurt content consistency on hard inputs. The modular ablation isolates this and it is one of the most instructive tables in the series — we will dissect it in the case studies. The short version: streaming the *language model* costs you on hard cases (tongue-twisters, repeated words), while streaming the *flow-matching model* is nearly free, because the semantic/acoustic split means the acoustic stage was never the part doing the hard reasoning.

## 6. Chunk-aware causal flow matching

Streaming the language model is half the problem. The flow-matching decoder also has to stream, and that is harder, because flow matching is natively an *offline* operation: you need all the speech tokens before you can integrate the ODE over the whole spectrogram. CosyVoice 2's chunk-aware flow matching solves this with a trick: treat the multi-step ODE solve as a deep network and make *that* network causal.

![A capability matrix of four attention masks — non-causal, full-causal, chunk-M, chunk-2M — scored on frames attended, first-packet latency, quality, and use case; richer-context masks give better quality at higher latency](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-7.webp)

The matrix in the figure lays out the four attention masks the decoder is trained on. The ten Euler steps of the ODE solve are unfolded into a stack of ten UNet evaluations; by making that unfolded stack respect a causal attention mask, each chunk of Mel can be generated from past chunks plus a bounded lookahead, so generation can begin before all tokens arrive. The four masks span the latency/quality frontier:

- **Non-causal** attends to all past and future frames. Best quality, highest latency. Use it for offline batch jobs.
- **Full-causal** attends only to the past. Lowest latency, lower quality. Use it when latency is non-negotiable.
- **Chunk-M** attends to the past plus $M$ future frames. Low latency, good quality. Use it for the first chunk.
- **Chunk-2M** attends to the past plus $2M$ future frames. Medium latency, near-offline quality. Use it for later chunks where you can afford a bit more lookahead.

The masks are also given a few structural supports: before the upsampling that aligns the 25 Hz tokens to the 50 Hz Mel (a 2x upsample), a *lookahead convolution* — a right-padded 1-D conv with pad size $P$ and kernel size $P+1$ — gives the causal modules a small window of future information.

### The self-distillation that comes for free

Here is the clever bit. For each training example in a mini-batch, the mask is sampled *uniformly at random* from the four options. This means one model learns all four context regimes at once. And because the richer-context masks (non-causal, chunk-2M) produce better outputs than the leaner ones (full-causal, chunk-M) on the same input, the rich masks act as an *implicit teacher* for the lean masks during training. The paper calls it an "implicit self-distillation scheme": the model trained to do well with full context pulls up its own performance when it has less context, because they share weights and see the same data. You get a quality floor for free on the low-latency masks.

The modular study confirms the payoff: streaming the flow-matching model barely moves content consistency (and even slightly *improves* speaker similarity, because the initial chunks in streaming mode have a higher prompt-to-generation ratio than the heavily-padded offline mode). This is the chunk-aware design doing its job — covering the entire latency/quality frontier with a single set of weights.

## 7. Post-training: DiffRO and the differentiable reward

CosyVoice 2 already experimented with reinforcement learning — direct preference optimization (DPO) with speaker-similarity and WER rewards, plus a differentiable ASR reward. CosyVoice 3 turns that experiment into a general method: **DiffRO** (Differentiable Reward Optimization). It is the most transferable idea in the whole series, because it applies to *any* discrete-token TTS model, not just CosyVoice.

![DiffRO data flow: input text feeds the LM, which branches to a Gumbel-softmax token sampler and a token-level KL term; the sampled tokens feed a frozen Token2Text ASR model that emits a log-probability reward, and the reward plus KL penalty backpropagate to update the LM — no vocoder and no RL loop](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-8.webp)

The figure shows why DiffRO is different from standard RLHF-style post-training. The problem with applying RL to TTS is structural: to score a generated utterance you normally have to run the full pipeline — language model, flow matching, vocoder — to get audio, then score the audio. That is expensive (the downstream models are large), and worse, after the flow-matching and vocoder stages the audio quality is so uniformly high that telling "good" from "bad" samples for reward modeling is genuinely hard. DPO needs four forward passes per training step and synthesizes audio repeatedly.

DiffRO sidesteps all of it by scoring the *tokens directly*, before any audio exists. It trains an ASR-like Token2Text model — a model that reads speech tokens and predicts text — and uses its log-posterior as the reward:

$$\tilde{\mu}_t = \text{GumbelSoftmax}\, P_{\pi_\theta}(\mu_t \mid \mu_{1:t-1};\, Y), \qquad R_{\text{ASR}}(Y) = \log P_{\text{ASR}}(\tilde{Y} = Y \mid \tilde{\mu}_{1:T})$$

The Gumbel-softmax makes the token sampling differentiable, so the ASR reward's gradient flows straight back into the language model's parameters. No reinforcement-learning loop, no vocoder in the gradient path. The objective is a reward term plus a KL penalty to keep the model near its reference, with one twist: the KL is computed at the *token-logit level*, not the sequence level:

$$\pi_\theta^* = \max_{\pi_\theta} \mathbb{E}[R(Y)] - \beta\, D_{\text{KL}}\big[\pi_\theta(\mu \mid Y) \,\Vert\, \pi_{\text{ref}}(\mu \mid Y)\big]$$

Here is the reward step:

```python
import torch
import torch.nn.functional as F

def diffro_step(lm, asr_token2text, ref_lm, text, mu_prev, tau=1.0, beta=0.1):
    """One DiffRO update: differentiable ASR reward + token-level KL.
    lm:             text-speech LM being optimized (pi_theta)
    asr_token2text: frozen model mapping speech tokens -> text posterior
    ref_lm:         frozen reference policy (pi_ref)
    """
    logits = lm(text, mu_prev)                       # (B, T, vocab) next-token logits
    # differentiable sample of speech tokens via Gumbel-softmax (soft one-hot)
    mu_soft = F.gumbel_softmax(logits, tau=tau, hard=False)   # (B, T, vocab)
    # ASR reads the (soft) tokens and scores how well they recover the text
    asr_logp = asr_token2text.log_prob(text, mu_soft)         # scalar per batch
    reward = asr_logp.mean()
    # token-level KL to the frozen reference (not sequence-level)
    with torch.no_grad():
        ref_logits = ref_lm(text, mu_prev)
    kl = F.kl_div(F.log_softmax(logits, -1),
                  F.softmax(ref_logits, -1), reduction="batchmean")
    loss = -reward + beta * kl                       # maximize reward, stay near ref
    return loss
```

### What DiffRO buys, and what it hacks

The wins are large and concentrated where you would hope. Across CosyVoice 2 and 3, DiffRO delivers 20%–50% relative WER improvement, and the gains are biggest in low-resource languages and cross-lingual settings — CosyVoice 3-0.5B shows a 68.7% relative WER improvement on Korean. RL post-training also shrinks the gap between the 0.5B and 1.5B models, which tells you something: a chunk of what model scale buys is just "predicting tokens that recover the text," and a targeted reward can supply that without more parameters.

The honest part of the paper is that it documents the hacking. Because the reward optimizes "do the tokens recover the text," chasing it slightly *degrades* speaker similarity — the model spends capacity on intelligibility at a small cost to timbre fidelity. The proposed mitigation is a multi-task reward (MTR): besides the Token2Text model, DiffRO can use speech-emotion-recognition, MOS-prediction, and audio-event-detection models as additional reward heads, letting you steer attributes like emotion:

$$R_{\text{MTR}}(Y, \{A_i\}) = \sum_i \log P_{\text{task}_i}(\tilde{A}_i = A_i \mid \tilde{\mu})$$

Adding the SER reward (DiffRO-EMO) pushes CosyVoice 3 to top emotion accuracy across most categories — but the paper notes this *hurts* pronunciation, surfacing the central tension of multi-objective post-training: rewards compete, and balancing them is unsolved. We will return to the reward-hacking failure mode in the case studies, because it is a clean example of an optimization doing exactly what you asked rather than what you wanted.

## 8. Scaling to the wild: data, model, and a stronger tokenizer

CosyVoice 3's thesis is in its subtitle: "towards in-the-wild speech generation via scaling-up and post-training." Where versions 1 and 2 worked well in clean broadcast Chinese and English, version 3 targets noisy, multilingual, real-world audio. Three scaling moves and a data pipeline make it happen.

### A multi-task tokenizer on a bigger brain

Recall the second-order consequence of supervised tokens: you inherit your ASR backbone's coverage. CosyVoice 3 acts on it by replacing the SenseVoice-Large backbone with **MinMo**, a large-scale audio-understanding model trained on over 1.4 million hours of speech. The FSQ module is inserted into MinMo's voice encoder (12 transformer blocks with rotary position embeddings instead of SenseVoice's 6), and crucially the tokenizer is trained with *multi-task* supervision over 530,000 hours: multilingual ASR (365k hours), language identification (85k), speech emotion recognition (48k), audio event detection (21k), and speaker analysis (11k).

The multi-task supervision is the point. A tokenizer trained only on ASR captures *what was said*. Adding emotion and event recognition pushes the tokens to also capture *how it was said* — paralinguistic information like emotional tone and pronunciation style. This directly serves expressive, in-the-wild synthesis, where a flat ASR-only token would strip out exactly the prosodic richness you want to clone.

The tokenizer ablation is one of the more illuminating experiments in the paper, because it compares supervised, self-supervised, and unsupervised tokens head-to-head on downstream TTS at two data scales:

| Tokenizer | Type | test-zh CER | test-en WER | speaker sim |
| --- | --- | --- | --- | --- |
| SoundStream (1st VQ) | unsupervised acoustic | 14.19 | 25.34 | 0.457 |
| HuBERT | self-supervised | 18.68 | 6.50 | 0.716 |
| W2v-BERT 2.0 | self-supervised | 2.62 | 6.72 | 0.381 |
| CosyVoice 2 (FSQ-SenseVoice) | supervised | 1.92 | 7.21 | 0.668 |
| CosyVoice 3 (FSQ-MinMo) | supervised | 1.68 | 6.60 | 0.710 |

(All trained on the same 3,000-hour TTS set for a fair tokenizer comparison.) The story: unsupervised acoustic tokens (SoundStream) are terrible for content because they carry no text alignment. Self-supervised tokens are a mixed bag — HuBERT collapses on Chinese, W2v-BERT keeps too much acoustic information and tanks speaker similarity (it leaks the *source* speaker into the token, which fights the flow-matching model). Supervised semantic tokens win on the combination of content and speaker independence, and the MinMo backbone edges out SenseVoice. The bet from CosyVoice 1 — supervised beats unsupervised — holds three papers later, now with the strongest available ASR brain behind it.

### Data and model scale, and where it plateaus

The data scales from roughly ten thousand to **one million hours**, covering 9 languages (Chinese, English, Japanese, Korean, Russian, French, German, Spanish, Italian) and 18 Chinese dialects. The language model grows from 0.5B to 1.5B parameters; the flow-matching model grows from 100M to 300M and switches its backbone from the UNet to a **diffusion transformer** (DiT). The DiT is efficient enough that the text encoder and the explicit length-regularization module are dropped — a simple interpolation handles the token-to-Mel frame-rate mismatch that earlier versions solved with dedicated machinery.

The headline results: on SEED test-zh, CER drops from CosyVoice 2's 1.45% to 0.71% (the RL-tuned 1.5B model) — a 44% relative gain; test-en WER drops 51%. But the paper is refreshingly honest about diminishing returns. The 1.5B model does *not* uniformly beat the 0.5B model; on the hard test set the 0.5B-RL model (5.09% CER) edges out the 1.5B-RL model (5.66%), because, in the authors' words, the dataset is too small to train the larger model effectively on challenging cases. Going from 3,000 to 170,000 hours gives 63%–75% relative error reductions; going from 170,000 to one million hours helps but "begins to plateau." This is the kind of result that should recalibrate anyone who assumes more parameters and more data are always the answer — past a point, the bottleneck moves to *data diversity on hard cases* and to post-training, not raw scale.

### Two pragmatic techniques: pronunciation inpainting and self-trained text normalization

Two smaller CosyVoice 3 techniques are worth a mention because they solve real production pain.

**Pronunciation inpainting** addresses the BPE tokenizer's weakness: it cannot fix a mispronounced polyphonic character or rare word, because there is no phoneme-level handle. CosyVoice 3 extends the vocabulary to model mixed sequences of words *and* phonemes, building an auxiliary training set that replaces Chinese monophonic characters with pinyin and English words with CMU-dictionary phonemes. At inference you can splice a phoneme spelling into the text to force a pronunciation. The best variant (replace monophonic characters, mix in the phoneme) hits a 100% correction rate on a polyphone test set.

**Self-trained text normalization** removes the hand-crafted rule engine that converts "$5" or "2026" into spoken form. CosyVoice 3 builds paired training data three ways — run raw text through a rule-based normalizer then synthesize, prompt Qwen-Max to normalize then synthesize, and prompt Qwen-Max to *de*-normalize existing pairs to recover raw text — so the model learns to speak raw, un-normalized text directly. It is text normalization absorbed into the model rather than bolted in front of it.

### Instruction control scales too

Controllability followed the same one-component-at-a-time arc as everything else. CosyVoice 1 shipped a *separate* model, CosyVoice-instruct, fine-tuned without the speaker embedding to follow natural-language style prompts. It supported speaker identity, speaking style (emotion, gender, rate, pitch), and fine-grained paralinguistics through inline markup. The control vocabulary is worth seeing, because it is how you actually drive the system:

```
Natural-language instruction (prepended, ends with <|endofprompt|>):
  "A happy girl with high tone and quick speech.<|endofprompt|>The sun is shining today."

Fine-grained inline markup (spliced into the text itself):
  "Well that's kind of scary [laughter]."
  "The team's <strong>unity</strong> and <strong>resilience</strong> won them the title."
  "Speaking while <laughter>telling the joke</laughter> made it land."
```

CosyVoice 2's move was to *merge* the instruct model back into the base model: instruction-following and zero-shot capability now live in one set of weights, trained on 1,500 hours of instructed data. The payoff is in the instruction-MOS metric (MOS-I, accuracy and naturalness of following an instruction), which jumps from CosyVoice-instruct's 3.09 to CosyVoice 2's 4.06. The ablation also makes a point that is easy to miss: when you *remove* the instruction from CosyVoice 2's input, content consistency and speaker similarity *improve* (CER 1.52 to 0.97) while MOS-I collapses to 2.28. Controllability is not free — asking the model to act emotional or speak a dialect costs you a little intelligibility and timbre stability, which is the same Pareto tension DiffRO's multi-task reward runs into.

CosyVoice 3 then *scaled* the merged capability: instruction data grew from 1,500 to 5,000 hours, and the number of supported styles passed 100 — adventurous, merciless, Cantonese dialect, Sichuan dialect, robot, Peppa, and dozens more, plus Chinese-English, Indian-English, and Russian-English accents. The mechanism never changed (a natural-language prompt before `<|endofprompt|>`, inline tags inside the text); only the coverage grew. The honest limitation the papers keep flagging: timbre is *not* controllable by instruction, because timbre lives in the flow-matching model and the instruction conditions the language model. Editing a voice's timbre with text — "make this voice raspier" — remains an open problem precisely because of the semantic/acoustic split that makes everything else clean.

### The data pipeline that makes in-the-wild audio usable

None of the scaling works without clean data, and in-the-wild audio is filthy. CosyVoice 3's six-stage pipeline is the unglamorous engine behind the whole thing.

![A six-stage data pipeline: speaker diarization plus VAD plus audio-event detection cutting 30-second segments, MossFormer2 noise reduction, three-ASR cross-validation requiring pairwise WER under 15 percent, Montreal Forced Aligner punctuation, volume standardization, and length-ratio filtering — producing training-grade multilingual audio](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-9.webp)

Each stage in the figure removes a specific failure mode from raw audiobooks, podcasts, and videos:

1. **Speaker diarization + VAD + audio-event detection** cuts the stream into speaker-coherent segments under 30 seconds, with non-speech removed.
2. **MossFormer2 noise reduction** denoises, and segments that start or end mid-word (from bad truncation) are screened out by leading/trailing energy.
3. **Three-ASR cross-validation** is the clever one: transcribe each clip with three independent ASR models (Faster-Whisper Large-V3, NVIDIA NeMo Canary-1B, Meta seamlessM4T-V2-large) and keep a transcript only if the average pairwise WER among the three is below 15%. Disagreement among three strong ASR systems is a reliable signal that a clip is hard, noisy, or mis-transcribed — so you drop it.
4. **Montreal Forced Aligner punctuation** fixes punctuation to match real pauses: add a comma where there is a $\ge 300$ ms gap, remove pause-implying punctuation where the gap is $\le 50$ ms.
5. **Volume standardization** normalizes amplitude with a simple peak normalization, $\text{wav}_{\text{norm}} = \frac{\text{wav}}{\max(\text{wav})} \times 0.6$.
6. **Length-ratio filtering** computes the ratio of speech-token length to text-token length per clip and discards the bottom 1% and top 5% — catching mismatches like a long text paired with near-silent audio.

The three-ASR consensus filter is the technique to steal. When you do not have ground-truth labels and your pseudo-labeler (ASR) is itself imperfect, using *disagreement across independent labelers* as a quality gate is a robust, cheap way to throw out the data that would poison training. It generalizes well beyond speech.

## Cross-cutting concerns

A few themes cut across the whole stack and deserve their own treatment.

### Latency and the streaming/quality trade-off

We wrote the first-packet budget $L_{\text{TTS}} = M(d_{\text{lm}} + d_{\text{fm}} + d_{\text{voc}})$ earlier. The strategic point is that CosyVoice's unified streaming framework means you do not choose latency at deployment by picking a model; you choose it by picking $M$ and the flow-matching mask, on the *same* weights. A latency-sensitive voice agent uses a small $M$ and the chunk-M mask for the first packet; a podcast-generation batch job uses non-causal and large chunks. One checkpoint, the whole frontier.

### Evaluating expressive TTS is harder than it looks

CosyVoice 3 introduces the **CV3-Eval** benchmark precisely because the standard ones stopped discriminating. On clean benchmarks like LibriSpeech, top systems now beat the ground-truth audio on WER, which means the benchmark is saturated and measures noise. CV3-Eval is built from genuinely in-the-wild reference audio (Common Voice, FLEURS, EmoBox, web-crawled clips), spans 9 languages and 18 dialects, and includes cross-lingual and emotion-cloning subsets. It also exposes a measurement trap we will dissect in the case studies: an ASR-based WER metric penalizes emotional and accented speech, so "content consistency" numbers can punish a system for being *more* expressive.

### Multilingual robustness is a coverage problem, not a modeling problem

The recurring failure across the series is not architectural — it is coverage. Japanese underperforms because its character set overlaps Chinese and the tokenizer leaks Chinese pronunciations. Korean does better because it has no such overlap. The fixes are coverage fixes: more data, kana conversion, a broader tokenizer. The architecture was right; the data distribution was the constraint. That is worth internalizing, because it predicts where the next version's effort goes — not new modules, but more languages and harder cases.

The cross-lingual numbers make the coverage story vivid, and they also show post-training carrying the weakest languages. On the CV3-Eval cross-lingual subset, where the reference voice and target text are different languages, DiffRO is transformative: cloning into Korean (to-ko) drops from a barely-usable 24.8% WER on CosyVoice 2 to 16.9% on CosyVoice 3-0.5B and then to 14.4% with the 1.5B model — but the single largest jumps come from DiffRO, which on some pairs more than halves the error. The Korean-to-Chinese (ko-to-zh) pair falls from 7.70% to 1.03% with DiffRO applied. The pattern is consistent: the rarer the language and the harder the transfer, the more post-training matters, because the base model simply has not seen enough of that distribution and a reward that says "make the tokens recover the text" supplies the missing supervision cheaply. The strategic reading is that for a long-tail multilingual system, you should budget for *both* data coverage and reward-based post-training — they attack the same coverage gap from different ends, and the CosyVoice 3 results show they stack rather than overlap.

## Case studies from the papers

Each of these is a concrete decision, failure, or measurement from the three papers, with the numbers and the lesson.

### 1. The speaker-embedding leak

CosyVoice 1 fed an x-vector speaker embedding into the language model to give it speaker awareness. It seemed reasonable: the LM produces prosody, prosody depends on the speaker, so tell the LM the speaker. The bug was that an utterance-level embedding is not a clean speaker signal — it also encodes the language being spoken and paralinguistic style. By conditioning the *semantic* model on that vector, version 1 contaminated its prosody and crippled cross-lingual transfer; the LM had learned to associate certain prosodic patterns with whatever language the embedding implied.

CosyVoice 2's fix is a deletion: remove the embedding from the LM, keep it only in the flow-matching model where timbre belongs. The ablation shows content error dropping (test-hard CER 2.96 to 2.56 after the removal, on top of the LLM-init gain) while speaker similarity holds steady, proving the LM was never the right home for speaker identity. The lesson is a design heuristic you can reuse: when you condition a module on a feature, ask whether that feature is *pure* for that module's job. A speaker embedding bundles identity, language, and style; feed it to the part of the system that only needs identity, and nowhere else. Leaks like this are invisible in aggregate metrics until you run the controlled ablation — which is exactly why the controlled, one-change-at-a-time structure of these papers is so valuable.

### 2. The dead codebook

The vector-quantized codebook in CosyVoice 1 looked like 4096 tokens of capacity. It was really about 1000. Codebook collapse — where most entries lose the nearest-neighbor competition early and never recover, because EMA only updates winners — left 77% of the codes dead, a 23% utilization rate measured directly in CosyVoice 2.

The fix, FSQ, is almost anticlimactic: delete the learned codebook and round each dimension of a low-rank projection onto a fixed integer grid, making every code reachable by construction. Utilization goes to 100%, ASR error through the tokenizer nearly halves (18.26% to 10.67% on Common Voice English), and end-to-end CER on test-zh drops from 2.56% to 1.45%. The lesson is broader than TTS: a learned discrete bottleneck has a failure mode (collapse) that a *parameter-free* discretization simply does not have. When a learned codebook is fighting you, ask whether you need it learned at all. Often a fixed grid with enough cells, plus a good continuous encoder feeding it, beats a clever-but-collapsing codebook. FSQ is now the default in CosyVoice and a growing number of audio and image tokenizers for exactly this reason.

### 3. Japanese and Chinese, and the character-overlap collapse

CosyVoice 2 supports Japanese, but badly: Japanese CER is 18.79% versus Korean's 7.98%, and the cross-lingual case is worse — cloning a Japanese voice to speak Chinese (ja-to-zh) hits 48.1% CER. The root cause is character-set overlap. Japanese kanji share glyphs with Chinese hanzi, and the supervised tokenizer — trained mostly on Chinese and English — maps those shared glyphs to Chinese pronunciations. The model literally speaks Japanese text with a Chinese accent because the tokens told it to.

This is the supervised-token second-order consequence biting. CosyVoice 3's fix is targeted: convert all Japanese characters to kana before tokenization, removing the glyph ambiguity. The ja-to-zh cross-lingual CER drops from 48.1% to 6.86% in the 0.5B model. The lesson is that a supervised tokenizer's failures are *predictable from its training distribution* — if two languages share symbols and your ASR backbone favored one, the tokenizer will too, and the fix lives in preprocessing (disambiguate the symbols) as much as in the model. When you adopt a supervised representation, audit its training data for exactly these collisions before you are surprised by them in production.

### 4. DPO punishes tongue-twisters

CosyVoice 2 tried DPO for post-training: synthesize pairs, label the better one by WER and speaker similarity, optimize the preference. It worked on average — but it *hurt* the hard test set. The diagnosis is sharp: hard cases contain repeated words and tongue-twisters, and a sample that correctly repeats a word looks, to a WER-based preference labeler, a lot like a "rejected" sample full of repetitions. DPO learned to avoid the very patterns the hard cases require. The numbers show DPO helping Chinese and English subsets while degrading the hard subset.

The differentiable ASR reward (the seed of DiffRO) does not have this failure, because it does not do pairwise preference at all — it directly maximizes the ASR posterior of the *correct* text given the tokens, so a legitimately-repeated word is rewarded for being recoverable rather than penalized for looking like a repetition. On the hard sets, the differentiable reward generalizes where DPO regresses, and the paper's recommendation is to combine them: $\mathcal{L}_{\text{ASR}} + \mathcal{L}_{\text{DPO}}$ gives the best of both. The lesson is about reward design under distribution shift: a preference signal built from a proxy metric (WER) inherits the proxy's confusions (repetition looks like an error), and a *direct* differentiable objective can be more robust precisely because it is tied to the true target rather than a comparison.

### 5. Reward hacking in DiffRO

DiffRO does exactly what its reward says, which is the problem. The reward maximizes "do these tokens recover the text," so the optimizer pours capacity into intelligibility — and the paper measures the cost: speaker similarity drops slightly across most datasets after RL. The model learned to hack its reward, sacrificing a little timbre fidelity for ASR-recoverability, because nothing in the objective protected timbre.

The mitigation is the multi-task reward — add speaker-similarity, emotion, or MOS reward heads so the optimizer cannot ignore them. But this exposes the next layer of the problem: when DiffRO-EMO adds an emotion reward to boost expressiveness, it *degrades pronunciation*, because emotional speech and clean pronunciation pull in different directions. There is no free lunch; you are trading along a Pareto frontier and the reward weights pick your point on it. The lesson is the oldest one in optimization, freshly illustrated: a single scalar reward will be gamed along whatever axis it ignores, and stacking rewards turns a hacking problem into a balancing problem that has no clean solution — only an explicit choice of trade-off. If you deploy reward-based post-training, instrument *every* metric you care about, not just the one you are optimizing, because the one you forgot is the one that will move.

### 6. Bigger is not always better

Conventional wisdom says the 1.5B model should beat the 0.5B model everywhere. It does not. On the SEED hard test set, CosyVoice 3-0.5B-RL scores 5.09% CER and the 1.5B-RL scores 5.66% — the smaller model wins on the hardest cases. The authors' explanation is that the pretraining and post-training data is too limited to exploit the larger model's capacity on challenging scenarios; the 1.5B model has more parameters than the hard-case data can usefully constrain.

The general pattern from the scaling study reinforces this: 3,000 to 170,000 hours buys 63%–75% relative error reduction, but 170,000 to one million "begins to plateau." Past a point, the binding constraint is not model size or even total data volume — it is *diversity on the hard tail*. The lesson for anyone scaling a system: measure where your returns plateau before you buy the bigger model, and check the *hard* slice separately, because aggregate metrics hide the place where a bigger model can actively regress for lack of data to ground it. CosyVoice 3 explicitly defers the larger model's full payoff to a future "tens of millions of hours" dataset — an admission that the 1.5B model is data-starved, not under-parameterized.

### 7. Streaming hurts the LM but not the decoder

The modular study that isolates streaming is the cleanest demonstration of the semantic/acoustic split paying off. The authors run all four combinations of streaming and offline for the language model and the flow-matching model:

| Config | LM | Flow matching | test-hard CER | test-hard SS |
| --- | --- | --- | --- | --- |
| M1 | offline | offline | 6.83 | 0.776 |
| M2 | offline | streaming | 7.12 | 0.788 |
| M3 | streaming | offline | 7.88 | 0.773 |
| M4 | streaming | streaming | 8.08 | 0.785 |

Streaming the language model (M1 to M3) costs over a point of CER on hard cases — the LM loses future context and stumbles on tongue-twisters. Streaming the flow-matching model (M1 to M2) barely moves content and actually *raises* speaker similarity, because streaming's first chunks have a higher prompt-to-generation ratio than offline's padding-heavy sequences. The decoder streams almost for free; the LM does not.

This is the architecture's central thesis validated by ablation: because content lives in the LM and acoustics live in the flow-matching model, the cost of streaming localizes to the stage doing the hard semantic reasoning, and the acoustic stage — which was never doing that reasoning — pays almost nothing. The practical takeaway: if you must cut latency, cut it in the decoder first (chunk-aware flow matching is nearly lossless) and only stream the LM when you truly need the lower first-token latency, knowing it taxes hard cases.

### 8. The ASR-bias evaluation trap

CosyVoice 3 is *more* expressive than its predecessor, and on one metric it looks *worse* for it. On the Expresso expressive-speech benchmark, CosyVoice 3-1.5B records a higher WER (13.43%) than CosyVoice 2 (9.42%) — but also a higher style similarity (68.25 vs 60.98). The paper's diagnosis: the ASR model used to compute WER is biased toward standard, neutral pronunciation, so it transcribes emotional and accented speech *worse*, inflating the WER of the more expressive system. The ground-truth human recordings score a 10.0% WER on the same metric — worse than CosyVoice 2's synthesis — which is the tell that the metric, not the speech, is the problem.

The lesson is about benchmark validity. WER is a fine proxy for content consistency *when pronunciation is neutral*, and a misleading one when it is not, because your evaluator (an ASR model) has its own distribution and its own biases. A system optimized to lower WER will drift toward the neutral pronunciation the ASR model prefers, which is the opposite of expressive — the metric actively fights the goal. This is why CosyVoice 3 leans on subjective MOS and style-similarity metrics for expressive tasks, and why CV3-Eval separates text-related from text-unrelated emotion cloning. When your metric is itself a learned model, audit its biases before you optimize against it, or you will cheerfully optimize your system into blandness.

## The arc: what each generation actually bought

![Timeline of three CosyVoice generations: v1 (2024) supervised S3 tokens with VQ and OT-CFM at 3.63 percent CER; v2 (2024) FSQ plus Qwen2.5-0.5B with 5:15 streaming and chunk-aware flow matching at 1.45 percent CER; v3 (2025) FSQ-MinMo plus DiffRO plus DiT scaling to 1M hours and 1.5B parameters at 0.71 percent CER](/imgs/blogs/cosyvoice-v1-v3-tokens-flow-matching-streaming-10.webp)

The timeline makes the discipline of the series visible: each version pushed one axis and measured it cleanly.

| Dimension | CosyVoice 1 | CosyVoice 2 | CosyVoice 3 |
| --- | --- | --- | --- |
| Tokenizer | VQ in SenseVoice (4096, 23% used) | FSQ in SenseVoice (6561, 100%) | FSQ in MinMo, multi-task (530k hrs) |
| LM backbone | from-scratch transformer | Qwen2.5-0.5B | 1.5B (scaled from 0.5B) |
| Text/speaker handling | text encoder + x-vector | both removed | both removed, kana fix |
| Acoustic decoder | OT-CFM UNet | chunk-aware causal UNet | DiT (100M to 300M) |
| Streaming | no | unified 5:15 interleave | unified, refined |
| Post-training | none | DPO + diff. ASR reward | DiffRO (general) |
| Training data | ~170k hrs, 5 langs | ~170k hrs, 4 langs | 1M hrs, 9 langs + 18 dialects |
| SEED test-zh CER | 3.63% | 1.45% | 0.71% |

Read down any column and you see a coherent system; read across any row and you see a single technique being isolated and improved. CosyVoice 1 established the bet (supervised tokens, hybrid LM-plus-flow-matching). CosyVoice 2 made it efficient and streamable (FSQ, pretrained backbone, unified streaming, chunk-aware decoder). CosyVoice 3 made it scale and post-trainable (multi-task tokenizer, DiffRO, a million hours, DiT). The CER trajectory — 3.63 to 1.45 to 0.71 — is the cumulative result of never changing more than a couple of things at once.

## When to reach for this architecture, and when not to

The CosyVoice design is not the only way to build a TTS system, and it is the right choice for a specific shape of problem.

**Reach for an LM-plus-flow-matching hybrid like CosyVoice when:**

- You need **zero-shot voice cloning** from a few seconds of reference audio without per-speaker training. The semantic/acoustic split makes cloning a conditioning operation.
- You need **streaming with low first-packet latency** for interactive voice agents. The unified streaming scheme and chunk-aware decoder are built for exactly this, on one checkpoint.
- You want **prosody to come from a real language model**. Autoregressive token sampling on top of a pretrained LLM produces varied, context-aware intonation that non-autoregressive duration-predictor systems struggle to match.
- You have, or can build, a **strong multilingual ASR model** to supply supervised tokens. The whole approach is downstream of that asset.
- You need **fine-grained instruction control** — emotion, dialect, role, paralinguistic bursts — expressed in natural language alongside the text.

**Skip it, or pick a different design, when:**

- You synthesize **one or a few fixed speakers** and never need cloning. A classic non-autoregressive model (FastSpeech-style with a duration predictor) is simpler, fully parallel, and has no autoregressive instability to manage.
- You need **hard guarantees on timing or duration** (dubbing to a fixed video length, for instance). Autoregressive token generation does not give you precise output-length control the way an explicit duration model does.
- You **cannot tolerate autoregressive failure modes** — the occasional dropped or hallucinated token, the speaker-leakage corner cases — in a setting with no human in the loop. Pure non-autoregressive diffusion systems trade prosody variety for more predictable behavior.
- You are **latency-bound to a degree that even chunk streaming cannot meet**, or you have no GPU at inference. A small CNN vocoder with a parametric front-end will beat any LLM-based stack on raw cost.
- You need **singing or precise musical control**. The CosyVoice papers explicitly note singing is unsupported — the semantic-token representation was not built for sustained pitch control.

The deeper point is that CosyVoice's three papers are a master class in *constrained iteration*: hold the architecture fixed, change one load-bearing component per release, and measure it against a stable benchmark. The result is not just a good TTS system but a legible one — you can point at any number in the final model and trace it back to the specific decision that produced it. That legibility is rarer than the performance, and it is the part worth copying whether or not you ever build a speech model. Pick your own stable benchmark, change one thing at a time, and let the numbers tell you which of your instincts were wrong — that discipline travels to any system you are responsible for.

## Further reading

- [CosyVoice 1: A Scalable Multilingual Zero-shot TTS Synthesizer based on Supervised Semantic Tokens](https://arxiv.org/abs/2407.05407) — the original supervised-token bet and the OT-CFM decoder.
- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/abs/2412.10117) — FSQ, the Qwen2.5 backbone, unified streaming, and chunk-aware flow matching.
- [CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training](https://arxiv.org/abs/2505.17589) — the multi-task tokenizer, DiffRO, the million-hour data pipeline, and CV3-Eval.
- [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505) — the FSQ technique CosyVoice 2 adopts.
- [Training CosyVoice: a complete guide](/blog/machine-learning/deep-learning/training-cosyvoice) — the companion how-to-train walkthrough with interview questions.
- [Orpheus TTS and SNAC: LLM speech with codec tokens](/blog/machine-learning/signal-processing/orpheus-tts-llm-speech-snac) — a sibling LLM-codec TTS system for contrast.
- [HiFi-GAN, the GAN vocoder explained](/blog/machine-learning/signal-processing/hifi-gan) — the vocoder CosyVoice uses as its final stage.
- [Whisper internals: the ASR encoder](/blog/machine-learning/signal-processing/whisper-under-the-hood) — the ASR-encoder lineage that supervised speech tokens are tapped from.
- [Fine-tuning LLMs with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — background on the preference-optimization method DiffRO improves upon.
