---
title: "Autoregressive Image Models: From PixelCNN to the GPT-Image Comeback"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build an image generator the way GPT builds a sentence — one token at a time — and follow the family from per-pixel networks to next-scale VAR, quantization-free MAR, and the multimodal models that brought autoregression back."
tags:
  [
    "image-generation",
    "diffusion-models",
    "autoregressive-models",
    "vq-vae",
    "transformers",
    "generative-ai",
    "deep-learning",
    "tokenization",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/autoregressive-image-models-1.png"
---

For about four years, if you told a generative-modeling researcher that you wanted to build a state-of-the-art image generator by predicting one pixel at a time, left to right, top to bottom, the way GPT predicts the next word, you would have gotten a polite smile. Diffusion had won. Autoregressive (AR) image models were the thing you taught in the first week of a course to explain the chain rule, then quietly retired before the interesting results. They were slow, they produced blurry low-resolution samples, and they had been thoroughly lapped by DDPM and latent diffusion.

Then 2024 happened. A paper called **Visual Autoregressive modeling (VAR)** won the NeurIPS 2024 best paper award by changing what "next" means — predicting the next *resolution scale* instead of the next pixel — and reported an ImageNet 256×256 FID of **1.73**, beating the diffusion transformers that had been the gold standard, with *better* scaling-law behavior. A second paper, **MAR**, threw out the discrete codebook that everyone assumed autoregressive image models needed and modeled continuous tokens with a tiny diffusion loss, landing FID around **1.55**. And the largest commercial image models of 2025 — OpenAI's GPT-Image-1, Tencent's HunyuanImage-3.0, Lumina-mGPT — turned out to be autoregressive transformers that share their weights with a language model, which is *why* they can spell, count, follow a six-clause prompt, and reason about what they are drawing.

This post is the full arc of that comeback. We will start where every textbook starts — factorizing the joint pixel distribution $p(x)$ by the chain rule — and earn the right to call it slow by writing the masked-convolution math that makes PixelCNN work and counting its sampling cost. Then we will follow the single idea that unlocked everything: **discrete tokenization** with VQ-VAE/VQ-GAN, which turns a 256×256 image into a sequence of a few hundred codebook indices so a transformer can model it like text. From there we climb to the 2024–2026 frontier: VAR's next-scale prediction, MAR's quantization-free continuous tokens, and the native-multimodal models that unified image generation with language. By the end you will have VQ-tokenized an image, trained a tiny causal transformer over those tokens, sampled from it with temperature and top-k, and you will be able to make a defensible call on **when AR beats diffusion and when it doesn't**.

![A horizontal timeline tracing autoregressive image models from PixelRNN and PixelCNN in 2016 through Image GPT, DALL-E 1 with VQ-GAN, Parti and Muse, to the 2024 best-paper VAR and MAR, and 2025 multimodal GPT-Image and HunyuanImage](/imgs/blogs/autoregressive-image-models-1.png)

This is one family in the four-family map of generative models we laid out in [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard): VAE, GAN, autoregressive, and diffusion/flow. It is worth holding the **generative trilemma** in mind the whole way — sample quality, mode coverage/diversity, and sampling speed, where you usually only get to pick two. Autoregressive models historically bought *quality and exact-likelihood coverage* at a brutal cost in *speed*. The whole story of this post is the series of tricks — tokenization, next-scale prediction, masked parallel decoding — that chipped away at that speed cost until AR became competitive again.

## The chain rule: why "one pixel at a time" is exactly correct

Start from the only thing we actually want: a model of the probability distribution over images, $p(x)$, where $x$ is a vector of $N$ pixel values (for a 256×256 RGB image, $N = 256 \times 256 \times 3 = 196{,}608$ numbers). If we had $p(x)$ we could sample new images, score how plausible a given image is, and compute exact likelihoods. The trouble is that $N$ is enormous and the pixels are wildly correlated — a cat's left eye constrains its right eye — so we cannot just model each pixel independently.

The chain rule of probability gives us an *exact*, assumption-free way to break the joint distribution into a product of one-dimensional conditionals. Pick any ordering of the pixels (say raster order: row by row, left to right) and write

$$
p(x) = p(x_1, x_2, \ldots, x_N) = \prod_{i=1}^{N} p(x_i \mid x_1, \ldots, x_{i-1}) = \prod_{i=1}^{N} p(x_i \mid x_{\lt i}).
$$

This is not an approximation. It is the chain rule, and it holds for *any* ordering. Every term $p(x_i \mid x_{\lt i})$ is a distribution over a single pixel value given all the pixels that came before it. If each conditional is, say, a softmax over 256 possible intensity values, then we have decomposed an impossible 196,608-dimensional problem into 196,608 tractable 256-way classification problems. That is the entire autoregressive idea: **model the next thing given everything before it.**

![A branching graph showing an image factored by the chain rule into conditional terms p of x1, p of x2 given x1, up to p of xN given x less than N, all feeding into a product that equals p of x with exact likelihood, and a caution node marking O of N sequential sampling cost](/imgs/blogs/autoregressive-image-models-2.png)

The payoff is real and worth stating plainly, because it is exactly what diffusion does *not* give you. Because the model directly parameterizes each conditional, you can compute the **exact log-likelihood** of any image:

$$
\log p(x) = \sum_{i=1}^{N} \log p(x_i \mid x_{\lt i}).
$$

There is no variational lower bound here, no ELBO gap as in a VAE, no intractable normalizing constant as in an energy-based model. You feed an image in, run one forward pass with the right masking, sum the log-probabilities of the actual pixel values, and you have $\log p(x)$ to the precision of your floating point. This is why AR models are the natural home of **density estimation** and why, historically, they reported the best test-set likelihoods (bits-per-dimension) on CIFAR-10 and ImageNet. If your downstream task cares about likelihood — anomaly detection, compression, calibrated uncertainty — AR is the family that gives it to you directly.

The cost is equally plain and it is the villain of the whole story. To *sample* a new image you must draw $x_1$, feed it back in to get the distribution over $x_2$, draw $x_2$, feed both back in to get $x_3$, and so on. Sampling is inherently **sequential**: $N$ forward passes, one per pixel, each depending on the last. For a 256×256 image that is ~196,608 sequential network evaluations. Training, by contrast, is fully parallel — you have the whole real image, so you can predict all conditionals at once with teacher forcing and a single masked forward pass — but generation is the bottleneck that diffusion, with its fixed ~20–50 step budget regardless of resolution, would later exploit to win.

#### Worked example: counting the sampling cost

Take a tiny 32×32 RGB image, the size of CIFAR-10. That is $32 \times 32 \times 3 = 3{,}072$ pixels. A pixel-space AR model needs 3,072 sequential forward passes to generate one image. If each forward pass through a modest PixelCNN takes 2 ms on a GPU, that is **~6.1 seconds per image** — for a 32×32 thumbnail. Scale to 256×256 and you are at ~196,608 passes, on the order of **6–7 minutes per image** at the same per-step cost. A 50-step diffusion model generates a 256×256 image in well under a second on the same hardware. That single ratio — thousands of sequential steps versus tens of parallel-per-step iterations — is the entire reason AR lost the first round, and every advance in this post is an attack on that number.

## PixelRNN and PixelCNN: making the conditionals causal

The chain rule tells us *what* to compute; PixelRNN and PixelCNN (van den Oord et al., 2016) tell us *how* to compute all those conditionals efficiently with one network. The central engineering problem is **causality**: when the network predicts pixel $x_i$, it must see $x_{\lt i}$ but absolutely not $x_i$ or any later pixel, or it would be cheating — it would just copy the answer and learn nothing useful for generation.

PixelRNN solves this with recurrent connections (row LSTMs, diagonal BiLSTMs) that propagate information along the raster order, but it is slow to train because recurrence is sequential. PixelCNN solves it with a far more practical trick: **masked convolutions**. Take an ordinary convolution and zero out the weights in the kernel that correspond to the current pixel and all "future" pixels in raster order. Concretely, for a 3×3 kernel centered on the pixel being predicted, you keep the connections to the row above and to the pixels to the left in the current row, and you zero the center and everything to the right and below.

There is a subtlety that trips up everyone who implements this from scratch, and it is worth getting right because it reveals how careful the causality bookkeeping has to be. There are two mask types:

```python
import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    """Causal convolution for PixelCNN.
    mask_type 'A': used only in the first layer; excludes the center pixel
                   (so the very first prediction cannot see its own value).
    mask_type 'B': used in all later layers; includes the center pixel
                   (allowed, because by then the center carries features of x_<i, not x_i).
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.shape
        yc, xc = kH // 2, kW // 2
        # Zero out everything strictly after the center in raster order.
        self.mask[:, :, yc, xc + (mask_type == "B"):] = 0.0
        self.mask[:, :, yc + 1:, :] = 0.0

    def forward(self, x):
        self.weight.data *= self.mask  # enforce causality every forward
        return super().forward(x)
```

The first convolutional layer uses a **type-A** mask (excludes the center pixel) because at that point the center *is* the pixel we are predicting — letting it through would be a one-line information leak that turns your impressive bits-per-dimension into a fraud. Every subsequent layer uses a **type-B** mask (includes the center) because by then the center activation is a learned feature computed only from $x_{\lt i}$, so it is safe to use. A whole class of "my likelihood is suspiciously good" bugs come from getting this one boolean wrong.

The original PixelCNN had a real architectural wart called the **blind spot**: stacking masked convolutions creates a triangular receptive field that fails to cover some pixels that are legitimately in the past, so the model literally cannot condition on them. **Gated PixelCNN** fixed this by splitting the network into a *vertical stack* (everything in the rows above) and a *horizontal stack* (the pixels to the left in the current row), combining them so the receptive field finally covers the entire valid history without leaking the future. It also swapped the ReLU for a gated activation, $\tanh(W_f * x) \odot \sigma(W_g * x)$, borrowed from the LSTM gate, which gave a measurable likelihood bump.

There is one more piece of the PixelCNN story that is pure science and worth a paragraph, because it is the first appearance of a tension that runs through the whole family: **how do you parameterize the per-pixel distribution?** The original PixelRNN used a 256-way softmax over the integer intensities $\{0, \ldots, 255\}$ — clean, but it wastes capacity, because it treats intensity 127 and 128 as completely unrelated categories when they are obviously adjacent. PixelCNN++ replaced it with a **discretized mixture of logistics**: the model outputs the parameters of a small mixture of logistic distributions over the *continuous* intensity, and the probability of a discrete pixel value $v$ is the mass the mixture assigns to the interval around it,

$$
p(v \mid x_{\lt i}) = \sum_{m=1}^{M} \pi_m \left[\, \sigma\!\left(\frac{v + 0.5 - \mu_m}{s_m}\right) - \sigma\!\left(\frac{v - 0.5 - \mu_m}{s_m}\right) \right],
$$

where $\sigma$ is the logistic CDF, $\mu_m, s_m$ are the mixture means and scales, and $\pi_m$ the mixture weights, all predicted by the network. This costs a handful of parameters per pixel instead of 256 logits, exploits the ordinal structure of intensity, and converges faster. The choice echoes forward: it is exactly the "what distribution do we put on each element" question that token-AR answers with a categorical-over-codes, and that MAR answers with a per-token diffusion model. The factorization is fixed by the chain rule; the *per-element distribution* is a free design choice, and the history of AR image models is largely a history of better answers to it.

How well did this actually work? On CIFAR-10, Gated PixelCNN reached about **3.03 bits per dimension** (lower is better; this was state-of-the-art likelihood at the time), and PixelCNN++ (Salimans et al., 2017) pushed it to **2.92 bpd** with the discretized logistic mixture output and other refinements. Those are genuinely strong density-estimation numbers. But sample *quality* — what humans actually look at — lagged badly: 32×32 samples that were locally plausible but globally incoherent, and the O(N) sampling cost from the worked example above made scaling past 64×64 impractical. Great likelihood, mediocre pictures, glacial sampling. The trilemma, exactly.

## Image GPT: pixels are just a sequence, so use a transformer

The next move (Chen et al., OpenAI, 2020) was almost defiantly simple. Transformers had taken over language modeling by treating text as a sequence of tokens and learning $p(\text{token}_i \mid \text{token}_{\lt i})$ with masked self-attention. An image, after the chain rule, is *also* just a sequence of conditionals. So **iGPT** flattened an image into a 1D sequence of pixels in raster order and trained a GPT-2-style transformer on it, with exactly the same causal attention mask, the same next-token objective, no image-specific inductive bias at all.

The causality mechanism moves from masked convolutions to the transformer's **causal attention mask**: in the attention matrix, position $i$ is forbidden from attending to any position $j > i$. You implement it by adding $-\infty$ to the pre-softmax attention scores above the diagonal:

```python
import torch
import torch.nn.functional as F

def causal_self_attention(q, k, v):
    # q, k, v: (batch, heads, seq, dim)
    seq = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    # Lower-triangular mask: position i may attend only to j <= i.
    mask = torch.tril(torch.ones(seq, seq, device=q.device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v
```

iGPT could not feed raw 256-valued pixels into a transformer at full resolution — the sequence length and the vocabulary would have been ruinous — so it did two telling things. It worked at tiny resolutions (32×32, 48×48, 64×64) and it **clustered the RGB color space into a 512-entry palette** with k-means, so each pixel became one of 512 discrete tokens. That palette step is a quiet foreshadowing of everything to come: the realization that you do not need to model pixels in their raw continuous form, you need a *discrete vocabulary* over which a transformer can do classification.

The headline result of iGPT was not its sample FID, which was unremarkable, but its **representations**. A linear probe on the features of iGPT-L reached **96.3%** accuracy on CIFAR-10 and competitive numbers on ImageNet — showing that a model trained purely to predict the next pixel learns genuinely useful visual features, the image analog of GPT learning grammar and facts from next-word prediction. iGPT was the proof of concept that the transformer-plus-next-token recipe transfers from text to pixels. What it also proved, painfully, was that doing it in pixel space does not scale: a 64×64 image is a 12,288-long sequence, attention is $O(L^2)$, and you are still paying the O(N) sequential sampling tax. The obvious fix was staring everyone in the face from the iGPT palette trick — make the tokens carry *much* more than one pixel each.

## VQ-VAE and VQ-GAN: the tokenization that changed everything

Here is the hinge of the entire post. If a 256×256 image is 196,608 pixel-tokens, a transformer is hopeless. But what if we could compress the image into a small grid of, say, 16×16 = **256 discrete tokens**, each one a rich code that captures a whole patch of texture and structure? Then the transformer models a 256-long sequence — the length of a short paragraph — and everything that worked for language modeling suddenly works for images. This is what **VQ-VAE** (van den Oord et al., 2017) and its sharper successor **VQ-GAN** (Esser et al., 2021) provide, and it is the same discrete-latent autoencoder we build in detail in the sibling post [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).

The mechanism is vector quantization. An encoder CNN maps the image down to a grid of continuous feature vectors $z_e \in \mathbb{R}^{h \times w \times d}$ (for a 16× downsample of a 256×256 image, $h = w = 16$). We maintain a learned **codebook** $\{e_1, \ldots, e_K\}$ of $K$ embedding vectors. Each spatial location's continuous vector is *snapped to its nearest codebook entry*:

$$
z_q(x)_{ij} = e_k, \quad \text{where} \quad k = \arg\min_{m \in \{1, \ldots, K\}} \lVert z_e(x)_{ij} - e_m \rVert_2.
$$

After this nearest-neighbor lookup, every location is represented by a single integer — the index $k$ of its chosen codebook vector — so the image becomes a 16×16 grid of integers in $\{1, \ldots, K\}$, which we read out in raster order as a sequence of 256 tokens. A decoder CNN maps the quantized grid $z_q$ back to pixels. VQ-GAN improves the reconstruction quality of this autoencoder dramatically by adding a **perceptual loss** (LPIPS) and an **adversarial discriminator**, which is why its codes preserve crisp textures that the plain L2-trained VQ-VAE blurs.

![A vertical stack showing pixels at 256x256x3 flowing into a VQ-GAN encoder that downsamples 16x to a continuous 16x16 grid, then a nearest-code lookup against a codebook of 16384 entries producing a 16x16 grid of 256 token indices, which a causal transformer then models](/imgs/blogs/autoregressive-image-models-3.png)

There is one piece of math that everyone hits when they first implement VQ: the $\arg\min$ is not differentiable, so how does gradient descent train the encoder? The answer is the **straight-through estimator**. On the forward pass you quantize normally. On the backward pass you pretend the quantization was the identity function and copy the gradient from the decoder input straight through to the encoder output, skipping the non-differentiable lookup entirely. In one line of PyTorch it is a beautiful little trick:

```python
import torch

def vq_straight_through(z_e, codebook):
    """z_e: (B, H, W, D) continuous encoder output.
       codebook: (K, D) learned embeddings. Returns quantized z_q and the indices."""
    B, H, W, D = z_e.shape
    flat = z_e.reshape(-1, D)                         # (B*H*W, D)
    # Squared distances to every code, then nearest.
    dists = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ codebook.t()
             + codebook.pow(2).sum(1))
    idx = dists.argmin(1)                             # (B*H*W,) the discrete tokens
    z_q = codebook[idx].reshape(B, H, W, D)
    # Straight-through: forward uses z_q, backward passes grad to z_e unchanged.
    z_q = z_e + (z_q - z_e).detach()
    return z_q, idx.reshape(B, H, W)
```

The full VQ-VAE objective has three terms: the reconstruction loss $\lVert x - \text{decode}(z_q) \rVert$, a **codebook loss** $\lVert \text{sg}[z_e] - e \rVert^2$ that pulls codebook vectors toward the encoder outputs they get assigned to, and a **commitment loss** $\beta \lVert z_e - \text{sg}[e] \rVert^2$ that keeps the encoder from oscillating between codes, where $\text{sg}[\cdot]$ is the stop-gradient operator (PyTorch's `.detach()`). That commitment term, weighted by $\beta \approx 0.25$, is the difference between a codebook that trains stably and one where most codes go unused — the dreaded **codebook collapse** where the encoder learns to use only a handful of the $K$ entries.

Now the two-stage recipe that defined a generation of models (DALL·E 1, VQGAN-Transformer, Parti, Muse): **Stage 1**, train the VQ-GAN autoencoder so you can losslessly-enough convert images to-and-from 256-token sequences. **Stage 2**, freeze the VQ-GAN, convert your whole image dataset to token sequences, and train a plain causal transformer to model $p(\text{token}_i \mid \text{token}_{\lt i})$ over those sequences — exactly iGPT, but now each token is a 16×16-pixel patch's worth of information instead of one pixel. DALL·E 1 (Ramesh et al., 2021) did precisely this, concatenating BPE text tokens in front of the image tokens so the transformer learned $p(\text{image tokens} \mid \text{text tokens})$ and you got text-to-image from a single autoregressive sequence. The text-tokenization half of that pipeline is the same byte-pair encoding we dissect in [the BPE tokenizer](/blog/machine-learning/large-language-model/bpe-tokenizer).

#### Worked example: how much did tokenization buy us?

Concretely: a 256×256 RGB image is 196,608 pixel values. A type-f16 VQ-GAN with a 16× spatial downsample produces a **16×16 = 256-token** grid. That is a **768× reduction in sequence length** (196,608 / 256), and it moves the transformer's $O(L^2)$ attention cost down by a factor of $768^2 \approx 590{,}000$. Sampling drops from ~196,608 sequential steps to **256 sequential steps** — still sequential, still the trilemma's speed cost, but now in the same ballpark as a slow diffusion sampler rather than thousands of times worse. The price you pay is the **quantization error**: the VQ-GAN reconstruction is not perfect, so there is a hard ceiling on fidelity set by how good your autoencoder is, no matter how good your transformer becomes. That ceiling is exactly the problem MAR will later attack by removing quantization altogether.

## Why AR faded — and the precise reasons it came back

By 2022 the token-AR recipe had produced genuinely good models. **Parti** (Google, 2022) scaled the transformer to 20B parameters and produced photorealistic, text-aligned images; **Muse** (Google, 2023) used *masked* token prediction (more on that shortly) for fast parallel decoding. And yet the field's center of gravity moved decisively to diffusion — Stable Diffusion, Imagen, SDXL. Why? Three honest reasons.

First, **sampling speed**. Even at 256 tokens, AR sampling is strictly sequential, and you cannot trade quality for speed by simply running fewer steps the way diffusion lets you with DPM-Solver — you need every token. Second, the **quantization ceiling**: the VQ-GAN bottleneck threw away fine detail, and at the resolutions people wanted, diffusion in a continuous VAE latent (latent diffusion) reconstructed sharper images. Third, **raster order is a bad inductive bias for pixels**: predicting an image token-by-token in row-major order imposes a one-dimensional causal structure on a fundamentally two-dimensional, roughly-isotropic signal, so the model spends capacity fighting an ordering that does not match how images are organized.

The comeback came from attacking each of these directly, and it is worth being precise about which paper attacks which weakness, because that is the real intellectual content of 2024–2026.

## MaskGIT and Muse: parallel decoding through masking

Before VAR and MAR, there was a quieter but pivotal idea that attacked the *sampling-speed* weakness without touching the tokenizer: **masked generative transformers**. MaskGIT (Chang et al., 2022) and its text-to-image successor **Muse** (Chang et al., 2023) start from the same VQ token grid as token-AR, but they refuse the strict left-to-right order. Instead they train a *bidirectional* transformer with a BERT-style masked objective: randomly mask a fraction of the image tokens and predict the masked ones from the visible ones in parallel. At inference, they decode the whole image in a small number of rounds — start from an all-masked grid, predict every token at once, *keep* the most confident predictions, re-mask the rest, and repeat.

The math that makes this fast is the **confidence-based unmasking schedule**. Let $\gamma(r)$ be a function (cosine in MaskGIT) of the fraction of tokens that should remain masked at round $r$ out of $R$ total rounds. At round $r$ you predict logits for all currently-masked positions, take the per-token max-softmax confidence, and unmask the top $(1-\gamma(r))$ fraction by confidence, committing those tokens permanently. Because $R$ is small (MaskGIT used ~8–12 rounds for a 256-token grid; Muse used ~24 for a larger grid across two stages), you replace 256 strictly-sequential steps with ~10 *parallel* steps — each step predicts hundreds of tokens at once. The trade is the same one diffusion makes: a handful of refinement iterations instead of one pass per element.

```python
import torch
import math

@torch.no_grad()
def maskgit_decode(model, seq_len, rounds=10, temperature=1.0, mask_id=-1, device="cuda"):
    """Confidence-based parallel decoding over a VQ token grid."""
    seq = torch.full((1, seq_len), mask_id, device=device, dtype=torch.long)
    for r in range(rounds):
        logits = model(seq.clamp(min=0))                    # bidirectional, all positions
        probs = (logits / temperature).softmax(-1)
        conf, pred = probs.max(-1)                          # confidence + argmax per token
        # Cosine schedule: how many tokens stay masked after this round.
        ratio = math.cos(math.pi / 2 * (r + 1) / rounds)    # 1 -> 0 over the rounds
        n_keep_masked = int(seq_len * ratio)
        masked = (seq == mask_id)
        conf = conf.masked_fill(~masked, float("inf"))      # never re-pick committed tokens
        # Unmask the lowest-ranked-by-staying-masked = highest confidence tokens.
        thresh = conf.view(-1).sort().values[n_keep_masked] if n_keep_masked < seq_len else float("inf")
        commit = masked & (conf >= thresh)
        seq = torch.where(commit, pred, seq)
    return seq
```

Muse layered this on a frozen T5-XXL text encoder and a two-stage (low-res then super-res) token pipeline, and it matched the image quality of contemporary diffusion models while being **roughly an order of magnitude faster at inference** than autoregressive Parti and faster than diffusion at the time, precisely because of the parallel masked decoding. The reason MaskGIT/Muse matters for *this* post is that its masked, confidence-scheduled, parallel decoding is the direct ancestor of **MAR's** sampling procedure. When MAR later swaps the discrete tokens for continuous ones and the categorical head for a diffusion head, it keeps this masked parallel decoding loop almost wholesale. So the lineage is clean: token-AR gave us the tokenizer, MaskGIT/Muse gave us parallel masked decoding, VAR gave us next-scale, and MAR fused masked decoding with a diffusion head over continuous tokens.

![A before-and-after comparison contrasting raster autoregressive generation that predicts tokens one by one in row-major order over 256 sequential steps with broken 2D locality, against next-scale autoregressive generation in VAR that predicts a full resolution map at each of about ten scales from 1x1 to 16x16 and reaches FID 1.73 on ImageNet](/imgs/blogs/autoregressive-image-models-4.png)

## VAR: next-scale prediction and the best-paper result

**Visual Autoregressive modeling (VAR)** (Tian et al., NeurIPS 2024 best paper) made one change to the definition of "next," and it fixed both the speed problem and the raster-order problem at once. Instead of predicting the next *token* in raster order, VAR predicts the next *scale*: it generates the image as a sequence of progressively finer token maps, coarse to fine.

Here is the construction. A multi-scale VQ-GAN encodes an image into a *pyramid* of token maps: a 1×1 map, then 2×2, then 3×3, up through 16×16 (typically ~10 scales). The 1×1 map is the coarsest possible summary — one token for the whole image. Each subsequent scale adds the *residual* detail that the previous scales could not represent. The autoregressive factorization is now over **scales**, not tokens:

$$
p(r_1, r_2, \ldots, r_S) = \prod_{s=1}^{S} p(r_s \mid r_{\lt s}),
$$

where $r_s$ is the entire token map at scale $s$. The crucial difference: within a scale, *all tokens are predicted in parallel* (they only need to condition on the coarser scales, not on each other), so generating scale $s$ is a single forward pass. The sequential dependency is only across the ~10 scales.

This is a genuinely large complexity win, and it is worth doing the arithmetic. Raster AR over a 16×16 grid is $O(N) = O(256)$ sequential steps. VAR's scales have side lengths $1, 2, \ldots, 16$, so the number of scales $S \approx 16$ and the sequential cost is $O(S) = O(\log_2 N)$-ish — about **10 sequential forward passes** instead of 256, a ~25× reduction in sampling steps for the same final resolution, with the per-scale work parallelized. And because each scale is a 2D map predicted as a whole, the model conditions on a genuine coarse-to-fine 2D structure rather than a row-major 1D crawl, which is the better inductive bias for images.

The piece that makes this work — and the piece people gloss over — is the **multi-scale residual tokenizer**. You cannot just naively downsample the image to each resolution and quantize independently; the scales would carry redundant information and the model could not cleanly "add detail." VAR's tokenizer instead builds a residual pyramid. Encode the image to the finest feature map $f$. Then for each scale $s$ from coarse to fine, downsample the *current residual* to that scale's resolution, quantize it against a shared codebook to get the token map $r_s$, upsample the quantized result back, and subtract it from the residual before moving to the next finer scale. Formally, with $f_0 = f$ and $\downarrow_s, \uparrow_s$ the resize operators for scale $s$:

$$
r_s = \mathcal{Q}\!\left( \downarrow_s f_{s-1} \right), \qquad f_s = f_{s-1} - \uparrow_s \big( \text{lookup}(r_s) \big).
$$

Each scale therefore encodes only what the coarser scales could not represent — exactly the structure that lets the transformer treat "next scale" as "add the next level of detail." The decoder sums the upsampled quantized maps across all scales to reconstruct $f$, then the VQ decoder turns $f$ into pixels. This residual construction is why VAR's autoregression over scales is meaningful: $p(r_s \mid r_{\lt s})$ is genuinely "predict the residual detail at this resolution given the coarser image so far."

The results justified the best-paper award. On ImageNet 256×256 class-conditional generation, VAR reported **FID 1.73** with a 2B-parameter model, beating the strong **DiT-XL/2** diffusion transformer (FID ~2.27) while being roughly **an order of magnitude faster to sample**. More striking than the single number was the **scaling law**: VAR exhibited a clean power-law relationship between model size and FID, $\text{FID} \propto N^{-\alpha}$, with a strong fit ($R^2$ near 0.99 in the paper) — the same kind of predictable, smooth scaling behavior that made large language models a sound engineering bet, now demonstrated for image generation. That scaling claim is arguably the most consequential part of the paper: it says you can buy better images with more compute *predictably*, which is exactly what you want before you spend a GPU budget.

```python
# Conceptual sketch of VAR's next-scale generation loop (not a full impl).
# A real VAR uses a multi-scale VQ tokenizer; here we show the control flow.
import torch

@torch.no_grad()
def var_sample(transformer, scales, class_id, device="cuda"):
    """scales: list of (h, w) per resolution, e.g. [(1,1),(2,2),...,(16,16)]."""
    context = transformer.class_embed(torch.tensor([class_id], device=device))
    token_maps = []
    for (h, w) in scales:
        # Predict the ENTIRE token map at this scale in one parallel forward pass,
        # conditioned on all coarser maps already generated.
        logits = transformer(context, token_maps, target_hw=(h, w))  # (1, h*w, K)
        probs = logits.softmax(dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, h, w)
        token_maps.append(tokens)
        # The coarse map is upsampled and fed forward as conditioning for finer scales.
    return token_maps  # decode the finest map with the multi-scale VQ decoder
```

The honest caveats: VAR still inherits the VQ quantization ceiling (its multi-scale tokenizer is still a VQ-GAN), and the multi-scale tokenizer is more complex to train than a single-scale one. But it broke the two assumptions — that AR must be raster-ordered and must be O(N)-slow — that had written the family off, and it did so while *beating* diffusion on the field's standard benchmark. That is why the room woke up.

## MAR: autoregression without vector quantization

If VAR fixed the *order* and *speed*, **MAR** (Li et al., 2024, "Autoregressive Image Generation without Vector Quantization") fixed the *fidelity ceiling* by attacking the assumption that nobody had questioned: that autoregressive image models need discrete tokens at all. The discreteness was always a means to an end — it let us use a categorical softmax and cross-entropy, the comfortable machinery of language modeling. But it costs you the quantization error, and it caps how good your reconstructions can be.

MAR's insight: an autoregressive model is defined by the *factorization* $p(x) = \prod_i p(x_i \mid x_{\lt i})$, not by how each conditional $p(x_i \mid x_{\lt i})$ is parameterized. Token AR parameterizes it as a softmax over a codebook. But $x_i$ can just as well be a **continuous** vector (a continuous latent token from a *non*-quantized autoencoder), and the per-token conditional can be any distribution over $\mathbb{R}^d$. MAR models that continuous conditional with a **small diffusion model** — a tiny per-token denoising MLP that, conditioned on the transformer's output vector $z_i$ for position $i$, learns to sample the continuous token $x_i$. The transformer produces the *condition*; the little diffusion head turns that condition into a continuous token via a few denoising steps.

![A before-and-after comparison contrasting token autoregression using a VQ codebook with a softmax cross-entropy loss and a quantization-error fidelity cap, against continuous masked autoregression in MAR using no codebook, a tiny per-token diffusion head, and FID 1.55 on ImageNet with no quantization loss](/imgs/blogs/autoregressive-image-models-5.png)

The training objective is a clean composition. The big transformer predicts a conditioning vector $z_i = f_\theta(x_{\lt i})$ for each position. The small diffusion head defines a loss on the *continuous* ground-truth token $x_i$:

$$
\mathcal{L}(z_i, x_i) = \mathbb{E}_{\varepsilon, t} \left[ \lVert \varepsilon - \varepsilon_\phi(x_i^t \mid t, z_i) \rVert^2 \right],
$$

where $x_i^t = \sqrt{\bar\alpha_t}\, x_i + \sqrt{1 - \bar\alpha_t}\, \varepsilon$ is the noised token and $\varepsilon_\phi$ is the small denoiser conditioned on $z_i$. This is exactly the DDPM noise-prediction loss from [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), but applied *per token* rather than to the whole image, with the autoregressive transformer providing the conditioning. At sampling time you draw each token by running the few-step diffusion head conditioned on $z_i$, then feed the sampled token back into the transformer for the next position — autoregression, but the conditional is a diffusion process instead of a softmax.

The second ingredient is in the name: **masked** autoregression. Rather than a strict left-to-right order, MAR generalizes to **random-order masked prediction** (in the spirit of MaskGIT and Muse): at each step the model predicts a *set* of masked tokens in parallel, conditioned on the currently-known tokens, and you iterate, unmasking more tokens each round. This makes sampling far more parallel than strict raster AR — on the order of ~64 iterations rather than 256 — while still being autoregressive in the sense that each round conditions on the previously committed tokens.

Concretely, the per-token diffusion head is tiny — a few-layer MLP, often well under 1% of the transformer's parameters — and it shares the noise-prediction machinery with full-image diffusion, just operating on a single $d$-dimensional token at a time conditioned on $z_i$:

```python
import torch
import torch.nn as nn

class DiffusionHead(nn.Module):
    """Per-token denoiser. Conditioned on the transformer's vector z_i, it predicts
    the noise added to a single continuous token. This replaces the softmax-over-codes."""
    def __init__(self, token_dim, cond_dim, width=1024, depth=3):
        super().__init__()
        self.t_embed = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.cond_proj = nn.Linear(cond_dim, width)
        self.in_proj = nn.Linear(token_dim, width)
        self.net = nn.Sequential(*[nn.Sequential(nn.SiLU(), nn.Linear(width, width))
                                   for _ in range(depth)])
        self.out = nn.Linear(width, token_dim)

    def forward(self, x_t, t, z):
        # x_t: noised token (B, token_dim); t: (B,1); z: condition from transformer (B, cond_dim)
        h = self.in_proj(x_t) + self.t_embed(t) + self.cond_proj(z)
        return self.out(self.net(h))                # predicted noise eps_phi

def mar_token_loss(head, z, x0):
    """DDPM noise-prediction loss, applied PER TOKEN. abar: cumprod of (1-beta)."""
    B = x0.size(0)
    t = torch.rand(B, 1, device=x0.device)         # continuous time in [0,1]
    abar = torch.cos(t * torch.pi / 2) ** 2        # a simple cosine alpha-bar schedule
    eps = torch.randn_like(x0)
    x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
    eps_hat = head(x_t, t, z)
    return ((eps - eps_hat) ** 2).mean()           # the entire MAR per-token objective
```

At sampling time the two loops nest: the *outer* loop is MaskGIT-style masked parallel decoding over token positions (predict a set of masked tokens' conditioning vectors $z_i$ in parallel each round), and the *inner* loop, for each newly-decoded position, runs a short few-step diffusion sampler on the head to turn $z_i$ into a concrete continuous token. So MAR is, almost literally, MaskGIT's outer decoder wrapped around a per-token DDPM. The number of *outer* rounds (~64) governs the speed–quality trade, and the *inner* diffusion steps (a handful) govern per-token fidelity.

The payoff is a model that is "autoregressive" in factorization yet free of the discrete bottleneck. MAR reported ImageNet 256×256 FID around **1.55** (its best variant), competitive with or beating both VAR and the top diffusion transformers, and crucially it demonstrated that *the diffusion loss and autoregression are not competitors but composable parts*. That is the deep point this post keeps circling: MAR is literally an autoregressive transformer whose output distribution is a diffusion model. The two families everyone treated as rivals turn out to be two layers of the same stack.

## A tiny but real token-AR model you can actually run

Enough theory. Here is the smallest end-to-end token-AR pipeline that is still *real*: load a pretrained VQ tokenizer from 🤗 `diffusers`, encode an image to tokens, train a small causal transformer over those tokens, and sample with temperature and top-k. The pieces are exactly the production pieces, just sized for a single GPU.

First, tokenize an image with a pretrained `VQModel` (the taming-transformers VQ-GAN, available through `diffusers`):

```python
import torch
from diffusers import VQModel
from PIL import Image
import torchvision.transforms as T

device = "cuda"
# A pretrained VQ-GAN (f16, codebook ~16384). This is the Stage-1 autoencoder.
vq = VQModel.from_pretrained(
    "CompVis/ldm-celebahq-256", subfolder="vqvae", torch_dtype=torch.float32
).to(device).eval()

tfm = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor(),
                 T.Normalize([0.5] * 3, [0.5] * 3)])  # to [-1, 1]
img = tfm(Image.open("cat.jpg").convert("RGB")).unsqueeze(0).to(device)

with torch.no_grad():
    h = vq.encode(img).latents                 # continuous encoder grid
    # quantize() returns (z_q, loss, (perplexity, min_encodings, indices))
    z_q, _, (_, _, indices) = vq.quantize(h)
    tokens = indices.view(1, -1)               # (1, H*W) discrete token sequence
print("sequence length:", tokens.shape[-1])    # e.g. 256 for a 16x16 grid
```

Those `tokens` are the discrete sequence the transformer learns to model. Now a minimal causal transformer — a GPT in miniature — over the token vocabulary:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyImageGPT(nn.Module):
    def __init__(self, vocab, seq_len, dim=512, heads=8, layers=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        block = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, activation="gelu", norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        B, L = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :L]
        # Causal mask: position i attends only to j <= i (the chain rule, enforced).
        mask = torch.triu(torch.ones(L, L, device=idx.device), diagonal=1).bool()
        x = self.blocks(x, mask=mask)
        return self.head(self.norm(x))         # (B, L, vocab) next-token logits

# Training step is plain next-token cross-entropy with teacher forcing (parallel!).
def train_step(model, tokens, opt):
    logits = model(tokens[:, :-1])             # predict token i from tokens < i
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
    )
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

Note the symmetry with the chain rule: the model predicts `tokens[:, 1:]` (every token) from `tokens[:, :-1]` (everything before it) in a *single parallel forward pass* thanks to the causal mask. Training is fast; only generation is sequential. Here is the generation loop, with the temperature and top-k controls that govern the quality-diversity trade-off:

```python
@torch.no_grad()
def generate(model, vq, n_tokens, temperature=1.0, top_k=100, device="cuda"):
    model.eval()
    seq = torch.zeros(1, 1, dtype=torch.long, device=device)  # start token / BOS
    for _ in range(n_tokens):
        logits = model(seq)[:, -1] / temperature              # last-position logits
        if top_k is not None:                                 # keep only top-k codes
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)                     # sample next token
        seq = torch.cat([seq, nxt], dim=1)
    grid = seq[:, 1:].view(1, 16, 16)                          # drop BOS, reshape
    # Look up the codebook vectors and decode back to pixels.
    z_q = vq.quantize.embedding(grid).permute(0, 3, 1, 2)      # (1, D, 16, 16)
    return vq.decode(z_q).sample                               # (1, 3, 256, 256) image
```

Two knobs do the real work and both are worth understanding mechanically. **Temperature** divides the logits before the softmax: $T < 1$ sharpens the distribution toward the mode (higher fidelity, lower diversity, risk of repetitive/over-smooth images), $T > 1$ flattens it (more diversity, more artifacts). **Top-k** truncates the distribution to the $k$ most likely tokens before sampling, killing the long tail of low-probability codes that cause splotchy artifacts — this is the same nucleus/top-k sampling that keeps language-model generations coherent, doing the same job for image codes. In practice a token-AR image model with $T \approx 0.9$ and top-k around 100–600 lands in a sensible quality-diversity spot; cranking $T$ toward 1.2 visibly degrades coherence, which is your hands-on demonstration of the trilemma's quality-vs-diversity edge.

#### Worked example: the temperature knob as a trilemma dial

Sample the same TinyImageGPT three times with fixed seed and varying temperature. At $T = 0.7$, top-k 50: clean, slightly conservative samples, lower diversity — you will see the same handful of "modes" repeated. At $T = 1.0$, top-k 200: a good balance, diverse and coherent. At $T = 1.3$, no top-k: maximal diversity but visibly more incoherent samples — broken textures, color splotches where a low-probability code slipped through. You have just traced one axis of the generative trilemma with a single scalar, no retraining. The diffusion analog is the classifier-free guidance scale; here it is temperature and top-k. Same trade-off, different family.

## Stress-testing the token-AR model: where it breaks and how you'd measure it

A model you cannot break is a model you do not understand. Let me pose the real engineering problem — *you have trained the TinyImageGPT above and the samples look off* — and reason through the failure modes you will actually hit, because each one points at a structural property of the AR family.

**The samples drift and lose global coherence halfway through.** This is the signature failure of raster AR: by the time the model is generating the bottom rows, the conditioning context is hundreds of tokens long, and any error early in the sequence (a slightly wrong color in the top-left) compounds — the model conditions on its *own* imperfect past, a regime it never saw during teacher-forced training. This is **exposure bias**, and it is worse for images than for text because there is no "grammar" to snap back to. How you would measure it: generate at increasing sequence lengths and plot FID against position-in-sequence, or compute the per-token entropy along the raster and watch it spike where coherence breaks. The structural fixes are exactly VAR (coarse-to-fine means the global structure is fixed *first*, at the 1×1 scale, before any detail) and masked decoding (commit high-confidence tokens globally rather than left-to-right), which is one more reason those approaches won.

**Cranking the codebook size does not help and training gets unstable.** If you scale the VQ-GAN's $K$ from 1,024 to 16,384 hoping for sharper reconstructions, you often find **codebook collapse**: a large fraction of codes are never selected, the effective vocabulary is tiny, and the transformer's job gets *easier* in a bad way (it just predicts the few live codes). Measure it with codebook **perplexity** (the `perplexity` value `vq.quantize` returns) — if it is far below $K$, most of your codebook is dead. The fixes are the commitment loss weighting, codebook EMA updates, and code re-initialization of dead entries; this is exactly the instability MAR sidesteps by dropping quantization entirely.

**You give it a prompt with three objects and the count is wrong.** Compositional failures — "three red cubes and two blue spheres" rendering as two-ish cubes of muddled color — are not unique to AR, but classical token-AR is *especially* prone because the raster order makes long-range "I already drew two cubes, I need one more" bookkeeping hard. This is precisely the weakness that **native-multimodal AR** (next section) fixes by sharing weights with a language model that *knows how to count in text*, which is why GPT-Image-class models improved so visibly on counting and multi-object prompts. The honest measurement is a compositional benchmark like GenEval or T2I-CompBench rather than FID, which is blind to whether the count is right.

**At very low temperature the model produces near-duplicate, over-smooth images.** Pushing $T \to 0$ collapses every conditional to its mode, and the autoregressive product of modes is a single high-probability image repeated with minor variation — mode collapse by another route. This is the diversity corner of the trilemma failing, and the cure is the temperature/top-k balance from the worked example. The lesson for serving: never ship $T = 0$ for a creative image model; you trade away the diversity that makes it useful.

The throughline of all four failures is that they are *structural*, traceable to the chain-rule factorization and the raster order, and each one motivated a specific advance — which is the most satisfying way to read the 2024–2026 papers: not as random wins but as targeted fixes to a known fault list.

## The 2025–2026 frontier: native-multimodal AR

The deepest reason autoregression came back is not VAR's FID or MAR's continuous tokens — it is **multimodality**. A causal transformer over tokens does not care whether the tokens came from text or from an image. So you can put them in *one sequence*, train *one transformer*, and get a model that reads a prompt, *reasons in text* about what it should draw, and then emits image tokens — all in a single autoregressive pass with shared weights. That is something diffusion, with its separate denoiser and bolted-on text encoder, does not do natively.

![A branching graph showing text tokens from a BPE vocabulary and image tokens from a VQ or continuous tokenizer interleaved into one sequence that feeds a shared causal transformer, which routes through a text head for next words and an image head for next image tokens, producing an image with reasoning as in GPT-Image and HunyuanImage](/imgs/blogs/autoregressive-image-models-7.png)

This is the architecture behind the 2025 wave. **GPT-Image-1** (and its 2025 successor) is an autoregressive model in OpenAI's GPT family that generates images as token sequences sharing the transformer stack with the language model — which is *why* it is dramatically better at rendering legible text in images, following multi-clause prompts, and counting objects than the diffusion models that preceded it. **HunyuanImage-3.0** (Tencent, 2025) is an open large native-multimodal AR model that interleaves text and image generation. **Lumina-mGPT** demonstrated the recipe in the open-source world. **Fluid** (Google, 2024) scaled a continuous-token MAR-style model to **10.5B parameters** and showed that, like VAR, AR image generation scales smoothly and predictably with parameters and data — the scaling-law argument that makes the whole approach a sound bet for the next round of frontier models.

The mechanism for "world knowledge helping generation" is concrete and worth spelling out. Because the same weights that learned, from a trillion text tokens, that a "golden retriever has a long muzzle and floppy ears" are the weights generating the image tokens, the model brings linguistic/factual knowledge directly to bear on what it draws. A diffusion model has to learn that visual fact from images alone (or from a frozen text encoder's embedding); a unified AR model gets it from its text pretraining. This is the source of the reasoning, counting, and text-rendering gains people noticed in GPT-Image — it is a language model that happens to also speak in pixels.

Two practical consequences fall out of this design and they are worth naming because they are where the unified models genuinely pull ahead of diffusion. First, **in-context image-to-image** comes nearly for free: because images are just tokens in the sequence, you can prepend a reference image's tokens, then an instruction in text, then let the model autoregress the edited image — no separate ControlNet or IP-Adapter, no architecture change, just a longer prompt. This is the substrate behind the 2025 conversational-editing wave (GPT-Image edits, FLUX-Kontext-style instruction editing), where you say "make the car red and add snow" and the model edits in place. Second, **legible text rendering** improves sharply, and the reason is mechanical: rendering the word "OPEN" on a sign is, for a unified model, close to a spelling task it already mastered in text pretraining, decoded into glyph-shaped image tokens — whereas a diffusion model has to hallucinate letter shapes from a CLIP embedding with no token-level spelling prior, which is exactly why pre-2024 diffusion models produced famously garbled text. The honest caveat is cost: these are large models (Fluid at 10.5B, HunyuanImage-3.0 larger still), the sequential image-token decode is slower per image than a few-step distilled diffusion model, and serving them is a frontier-scale undertaking. You adopt native-multimodal AR when the unification, reasoning, editing, and text-rendering wins justify the inference bill — which, for a general-purpose "draw and reason" assistant, they increasingly do.

The architecture splits into two design choices for the image tokens. **Chameleon** (Meta, 2024) uses *discrete* VQ image tokens in a single early-fusion transformer trained on interleaved text-and-image with one next-token loss over a mixed vocabulary — the purest "everything is one sequence" design. **Transfusion** (2024) instead keeps image tokens *continuous* and trains the single transformer with a **language-modeling loss on text tokens and a diffusion loss on image tokens simultaneously** — the same transformer, two losses, autoregressive over text and diffusion over images. Transfusion is the clearest statement of the convergence thesis: the unified model of 2025 is not "AR beats diffusion," it is "one transformer, an AR loss where the signal is discrete and a diffusion loss where it is continuous." MAR made that argument at the token level; Transfusion makes it at the architecture level.

The unified training loop is worth seeing concretely, because it dispels the magic. You assemble a batch of interleaved sequences — a caption's BPE tokens followed by its image's tokens, repeated across documents — and you apply a *per-position* loss mask: cross-entropy where the position is a text (discrete) token, and the image loss (cross-entropy over codes for Chameleon, or the per-token diffusion loss for Transfusion/MAR) where the position is an image token. One backward pass, two loss types, summed:

```python
import torch
import torch.nn.functional as F

def unified_loss(model, seq, is_image, diffusion_head=None):
    """seq: (B, L) token ids. is_image: (B, L) bool marking image positions.
    Text positions use next-token cross-entropy; image positions use either
    code cross-entropy (Chameleon-style) or a per-token diffusion loss (Transfusion)."""
    hidden = model.backbone(seq[:, :-1])                 # shared causal transformer
    target = seq[:, 1:]
    text_mask = ~is_image[:, 1:]
    # Text loss: standard language-modeling cross-entropy on text positions only.
    text_logits = model.text_head(hidden)
    loss_text = F.cross_entropy(
        text_logits[text_mask], target[text_mask], reduction="mean"
    )
    if diffusion_head is None:                            # Chameleon: discrete image codes
        img_logits = model.image_head(hidden)
        loss_img = F.cross_entropy(
            img_logits[is_image[:, 1:]], target[is_image[:, 1:]], reduction="mean"
        )
    else:                                                 # Transfusion/MAR: continuous tokens
        z = hidden[is_image[:, 1:]]                       # conditions for image positions
        x0 = model.continuous_tokens(target)[is_image[:, 1:]]
        loss_img = mar_token_loss(diffusion_head, z, x0)  # the per-token DDPM loss from earlier
    return loss_text + loss_img
```

One more frontier detail that matters in practice: **classifier-free guidance comes along for free** in these AR models, exactly as it does in diffusion. You train with the conditioning (the text prompt, or the class) dropped some fraction of the time so the model learns both a conditional and an unconditional distribution; at sampling you extrapolate the logits, $\ell_\text{guided} = \ell_\text{uncond} + w \cdot (\ell_\text{cond} - \ell_\text{uncond})$, before the softmax (token-AR) or inside the diffusion head (MAR/Transfusion). VAR and MAR both report meaningful FID improvements from CFG, the same diversity-for-fidelity trade we know from [classifier-free guidance](/blog/machine-learning/image-generation/diffusion-from-first-principles) in diffusion — yet another place the two families share machinery. The 2025 multimodal models lean on this heavily: the guidance scale is a primary quality knob even though the backbone is an autoregressive transformer.

## A taxonomy of the AR family, by what each variant predicts

It helps to lay the whole family on one grid, organized by the single dimension that actually distinguishes them: **what is the unit of prediction**, and what does that choice cost you in sampling steps and fidelity. Pixel AR predicts one pixel and pays O(N) for exact pixels. Token AR predicts one VQ token and pays O(N) but at a much smaller N, capped by quantization. VAR predicts a whole scale map and pays only O(log N). MAR predicts a continuous vector and escapes the quantization cap entirely.

![A matrix comparing four autoregressive families across unit of prediction, sampling steps, and fidelity ceiling: pixel AR predicting one pixel at O of N around 196k with exact pixels, token AR predicting one VQ token at O of N around 256 with a VQ-limited ceiling, next-scale VAR predicting a scale map at O of log N around 10 also VQ-limited, and masked MAR predicting a continuous vector in about 64 masked steps with no VQ loss](/imgs/blogs/autoregressive-image-models-6.png)

Reading the grid top to bottom is reading the history of the field's attack on the trilemma's speed axis: each row pushes the sampling-steps column down (196k → 256 → 10 → ~64-but-parallel) while the fidelity column climbs (quantization ceiling lifted entirely by MAR's continuous tokens). The one thing every row shares is the chain-rule factorization — that is what makes them all "autoregressive" despite predicting wildly different units. And every row except the last keeps the *exact-likelihood* property that is AR's structural advantage over diffusion: a discrete token-AR model can report $\log p(x)$ exactly, which diffusion can only bound.

## AR vs diffusion, honestly — and why they are converging

Let me put the two families side by side without cheerleading, because the interesting truth in 2026 is that the line between them is dissolving.

| Property | Token AR (VQ-GAN) | Diffusion (DiT/LDM) | MAR (continuous) |
| --- | --- | --- | --- |
| Likelihood | **Exact** $\log p(x)$ | ELBO lower bound only | Approximate |
| Sampling | Sequential, N steps | Parallel-per-step, ~20–50 iters | Masked iters, ~64 |
| Quality ceiling | VQ-capped | VAE-latent-capped | No quantization cap |
| Multimodality | **Native** (one sequence) | Bolt-on text encoder | **Native** |
| Controllability | Prompting / in-context | CFG, ControlNet, inpainting | CFG-style |
| ImageNet 256 FID (class-cond.) | ~3–5 (VQGAN-AR) | ~2.27 (DiT-XL/2) | **~1.55** |
| Best scaling-law evidence | VAR (clean power law) | DiT (clean power law) | Fluid 10.5B |

A few honest observations. On **raw quality** at ImageNet scale, the gap is gone — VAR (1.73) and MAR (1.55) are at or below DiT-XL/2 (2.27). On **likelihood**, AR wins structurally and always will: it parameterizes the density directly. On **sampling speed**, classical token AR loses (sequential), VAR is competitive (O(log N) scales), and diffusion's fixed ~20–50-step budget is its enduring advantage. On **controllability**, diffusion has the mature ecosystem (ControlNet, IP-Adapter, inpainting, SDEdit), while AR's lever is *prompting and in-context conditioning* — which is suddenly very powerful in the multimodal-AR setting where you can interleave reference images and instructions in the sequence. On **unified multimodality**, AR wins decisively: it is one transformer for text and images, which is why the frontier commercial models went this way.

![A matrix comparing autoregressive, diffusion, and MAR on exact likelihood, unification with text, sampling structure, and ImageNet FID class, showing token AR with exact likelihood and native text unification but sequential sampling, diffusion with an ELBO bound and bolt-on text and parallel iterative sampling near FID 2.3, and MAR with approximate likelihood, native text, masked iterative sampling, and FID near 1.55](/imgs/blogs/autoregressive-image-models-8.png)

And here is the convergence, stated as plainly as I can. **MAR** is an autoregressive transformer whose per-token output distribution is a diffusion model — AR on the outside, diffusion on the inside. **Transfusion** is a single transformer trained with an AR loss on text and a diffusion loss on images — both losses, one network. The clean dichotomy "autoregressive *or* diffusion" was always an artifact of how we parameterized the conditional, not a fundamental fork. The factorization (chain rule) and the per-element distribution (softmax over codes, or a diffusion process over continuous vectors) are *independent design choices*. The frontier is mixing and matching them: AR factorization for the global structure and multimodal unification, diffusion for the continuous per-element fidelity. We pick up exactly this convergence in the dedicated showdown post [autoregressive vs diffusion: the 2026 showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown).

## The science of exact likelihood, and why it is AR's one structural moat

I keep claiming AR's exact-likelihood property as a real advantage, so let me make it rigorous rather than asserted, because it is the one thing diffusion structurally cannot match. For a *discrete* token-AR model the likelihood is unambiguous: the model outputs, for each position, a categorical distribution over the $K$ codes, and the log-likelihood of a token sequence is just the sum of the log-probabilities of the true tokens,

$$
\log p(\mathbf{t}) = \sum_{i=1}^{L} \log p_\theta(t_i \mid t_{\lt i}),
$$

with no approximation anywhere — one masked forward pass gives you every conditional, you index the true token in each, and sum. Contrast this with diffusion, where the model only gives you a *lower bound* on $\log p(x)$ via the evidence lower bound (the ELBO), and computing a tight likelihood requires the probability-flow ODE and an expensive trace estimate. AR hands you the exact number for the price of one forward pass.

To compare across models people report **bits per dimension** (bpd), which normalizes the negative log-likelihood by the number of pixel-dimensions and converts to base-2:

$$
\text{bpd} = \frac{-\log_2 p(x)}{N} = \frac{-\log p(x)}{N \cdot \ln 2}.
$$

The interpretation is genuinely concrete: bpd is the average number of bits an optimal entropy coder, using this model as its probabilistic model, would need to *losslessly compress* each pixel-dimension of held-out data. A model at 3.0 bpd on CIFAR-10 would compress those images to about 3 bits per sub-pixel; a worse model needs more bits. This is why PixelCNN-family numbers (2.92 bpd on CIFAR-10) were a big deal — they were, almost literally, a statement about the best lossless compressor you could build with that model. And it is why AR models, despite their sampling cost, remained the reference for *density estimation* long after they lost the *sample-quality* race: bits-per-dimension is exactly the quantity they optimize, and they optimize it exactly.

There is one honest asterisk, and it is the reason the comparison table marked MAR's likelihood "approximate." Once you move to *continuous* tokens with a diffusion head (MAR) or train with a diffusion loss (Transfusion), you give up the clean categorical likelihood — the per-token diffusion model has the same ELBO-only situation as full-image diffusion. So the convergence has a cost the cheerleading usually omits: when AR adopts diffusion's continuous per-element distribution to escape the quantization ceiling, it *also* inherits diffusion's loss of exact likelihood. You cannot have both the no-quantization fidelity and the exact density in one model; pick the property your application needs. For pure sample quality, MAR's trade is clearly right. For compression or calibrated anomaly detection, discrete token-AR's exact $\log p(x)$ is still the only game in town.

## Case studies: the real numbers

A few concrete, sourced results so you can calibrate. Treat the FID figures as the values reported in the respective papers on **ImageNet 256×256 class-conditional generation** unless noted; FID is computed against the standard 50k reference set, lower is better, and small differences (under ~0.2) are within noise, so read these as classes not exact rankings.

**PixelCNN++ (Salimans et al., 2017).** Density estimation, not pretty pictures: ~**2.92 bits/dim** on CIFAR-10, then-SOTA likelihood. Samples were 32×32 and globally incoherent. The lesson: great likelihood does not imply great samples, and O(N) per-pixel sampling does not scale.

**Image GPT (Chen et al., 2020).** The representation-learning proof: a linear probe on iGPT-L features hit ~**96.3%** on CIFAR-10. It worked at 32–64px with a 512-color palette. Lesson: next-token prediction learns strong visual features, but pixel-space sequences are too long to scale.

**VQGAN + Transformer (Esser et al., 2021)** and **Parti (Yu et al., 2022).** The two-stage token-AR recipe at scale. Parti at **20B** parameters produced strong text-to-image with state-of-the-art-at-the-time text alignment, demonstrating that AR text-to-image scales with parameters. Lesson: tokenization made transformer-AR over images practical and scalable, but the VQ ceiling and sequential sampling kept it behind latent diffusion for a while.

**VAR (Tian et al., NeurIPS 2024 best paper).** **FID 1.73** on ImageNet 256 with a 2B model, beating DiT-XL/2 (~2.27), ~**10–20× faster sampling** via ~10 scale-steps instead of 256 token-steps, and a clean **power-law scaling** of FID with model size ($R^2 \approx 0.99$ in the paper). Lesson: changing "next token" to "next scale" fixed both the speed and the raster-order weaknesses, and AR beat diffusion on the field's standard benchmark.

**MAR (Li et al., 2024).** Up to **FID ~1.55** on ImageNet 256 with **no vector quantization**, using a per-token diffusion loss. Lesson: discreteness was never required; removing it lifts the fidelity ceiling, and the diffusion loss composes cleanly inside autoregression.

**Fluid (Fan et al., Google, 2024).** Scaled continuous-token AR to **10.5B** parameters and reported smooth, predictable scaling of generation quality with size — the scaling-law evidence that makes large AR image models a defensible bet alongside DiT.

**Native-multimodal (GPT-Image-1 / HunyuanImage-3.0, 2025).** Not clean single-number benchmarks but the qualitative frontier: markedly better text rendering, instruction following, object counting, and in-image reasoning than prior diffusion models — directly attributable to sharing weights with a language model. Lesson: the strongest argument for AR in 2025–2026 is unification, not FID.

## When to reach for autoregressive image generation (and when not to)

Decisive guidance, because every choice is a cost.

**Reach for AR when you need unified multimodality** — one model that reasons in text and generates images, follows long compositional prompts, renders legible text, or interleaves images and instructions in a conversation. This is AR's structural home turf in 2025–2026, and it is why the frontier commercial models are AR. If your product is "chat that also draws," AR is the natural substrate.

**Reach for AR when you need exact likelihood** — anomaly/out-of-distribution detection, lossless or near-lossless compression, calibrated density estimation. Diffusion only gives you a bound; token-AR gives you $\log p(x)$ directly. This is a niche but a real one.

**Reach for next-scale AR (VAR) or MAR when you want frontier ImageNet-class quality with good scaling** and you are training from scratch on a large dataset — the scaling-law evidence says your compute buys predictable quality, and the FID numbers are at the frontier.

**Do NOT reach for classical raster token-AR for high-resolution single-image latency** — sequential O(N) sampling is its Achilles heel, and a 20–50-step diffusion model with DPM-Solver or a few-step distilled model will be faster per image. If latency on a single 1024×1024 image is your hard constraint and you do not need multimodality, diffusion (especially distilled few-step diffusion) is still the pragmatic choice.

**Do NOT reach for AR when you need the mature control ecosystem today** — ControlNet, IP-Adapter, inpainting/outpainting, SDEdit, regional prompting, the deep ComfyUI tooling. That ecosystem grew up around diffusion. AR's in-context control is powerful and growing but the off-the-shelf control toolkit is thinner.

**Do NOT pay for a multi-scale or continuous-token tokenizer if a plain latent-diffusion VAE plus a DiT gets you there** — VAR's multi-scale VQ and MAR's diffusion head add real training complexity. For many practical text-to-image products, a well-tuned latent-diffusion or flow-matching DiT is the lower-risk path; AR earns its complexity when you specifically need unification, likelihood, or its scaling story.

## Key takeaways

- **The chain rule is exact, not an approximation.** $p(x) = \prod_i p(x_i \mid x_{\lt i})$ holds for any ordering and gives autoregressive models their two defining traits: exact likelihood (a structural win over diffusion's ELBO) and strictly sequential sampling (their historic curse).
- **Causality is the whole implementation challenge.** Masked convolutions (type-A first layer, type-B after) in PixelCNN, the lower-triangular attention mask in transformers — get the masking wrong and your likelihood is a fraud.
- **Tokenization with VQ-VAE/VQ-GAN is the hinge.** Compressing a 256×256 image to ~256 discrete tokens (a 768× shorter sequence) is what made transformer-AR over images tractable and gave us DALL·E 1, Parti, and Muse. The straight-through estimator is how you backprop through the non-differentiable $\arg\min$.
- **AR faded for three concrete reasons** — sequential sampling speed, the VQ fidelity ceiling, and raster order being a bad 2D inductive bias — and the comeback fixed each one specifically.
- **VAR changed "next token" to "next scale"** (coarse→fine), turning O(N) sampling into ~O(log N) and fixing the raster-order bias, winning NeurIPS 2024 best paper with FID 1.73 on ImageNet and a clean scaling law that beat DiT.
- **MAR removed vector quantization** by modeling continuous tokens with a tiny per-token diffusion loss, lifting the fidelity ceiling to ~FID 1.55 and proving the diffusion loss composes inside autoregression.
- **The real 2025–2026 story is multimodal unification** — one transformer over text and image tokens (GPT-Image, HunyuanImage-3.0, Lumina-mGPT, Fluid) that brings language-pretrained world knowledge to image generation, which is why these models reason, count, and spell.
- **AR and diffusion are converging, not competing.** MAR is AR-outside/diffusion-inside; Transfusion is one transformer with an AR loss on text and a diffusion loss on images. The factorization and the per-element distribution are independent design choices, and the frontier mixes them.

## Further reading

- van den Oord, Kalchbrenner, Kavukcuoglu, **"Pixel Recurrent Neural Networks"** (2016) and van den Oord et al., **"Conditional Image Generation with PixelCNN Decoders"** (2016) — the per-pixel foundation and gated/masked-convolution causality.
- Chen, Radford, Child et al., **"Generative Pretraining from Pixels" (Image GPT)** (OpenAI, 2020) — pixels as a transformer sequence and the representation-learning result.
- van den Oord, Vinyals, Kavukcuoglu, **"Neural Discrete Representation Learning" (VQ-VAE)** (2017) and Esser, Rombach, Ommer, **"Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN)** (2021) — the discrete tokenization that unlocked token-AR.
- Ramesh et al., **"Zero-Shot Text-to-Image Generation" (DALL·E 1)** (2021); Yu et al., **"Scaling Autoregressive Models for Content-Rich Text-to-Image Generation" (Parti)** (2022); Chang et al., **"Muse: Text-To-Image Generation via Masked Generative Transformers"** (2023).
- Tian et al., **"Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction" (VAR)** (NeurIPS 2024, best paper) — next-scale prediction, FID 1.73, the scaling law.
- Li et al., **"Autoregressive Image Generation without Vector Quantization" (MAR)** (2024) — continuous tokens with a per-token diffusion loss.
- Zhou et al., **"Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"** (2024); Team Chameleon (Meta), **"Chameleon: Mixed-Modal Early-Fusion Foundation Models"** (2024); Fan et al., **"Fluid: Scaling Autoregressive Text-to-Image Generative Models with Continuous Tokens"** (Google, 2024).
- Within this series: [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) (the four-family map and the trilemma), [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) (the VQ tokenizer in depth), [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) (the per-token diffusion loss MAR reuses), [autoregressive vs diffusion: the 2026 showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
