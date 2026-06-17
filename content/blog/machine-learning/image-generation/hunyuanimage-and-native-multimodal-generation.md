---
title: "HunyuanImage and Native Multimodal Generation: One Transformer for Text and Pixels"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How Tencent's HunyuanImage 3.0 generates images token-by-token in the same transformer that handles language — gaining world knowledge and reasoning a diffusion model conditioned on a frozen encoder simply cannot reach."
tags:
  [
    "image-generation",
    "diffusion-models",
    "autoregressive-models",
    "multimodal",
    "hunyuanimage",
    "mixture-of-experts",
    "generative-ai",
    "deep-learning",
    "world-knowledge",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/hunyuanimage-and-native-multimodal-generation-1.png"
---

Here is a prompt that quietly breaks most image generators: *"A vintage 1960s American magazine advertisement for a washing machine, with a tagline and a price in old dollars, the brand logo top-left, drawn in the flat illustration style of that era."* A strong diffusion model will give you something that *looks* like a 1960s ad — but read it closely. The "tagline" is gibberish letters. The price says something like "\$3,47.99" with the comma in the wrong place. The illustration style is roughly right but the layout convention — logo placement, the way mid-century ads stacked a headline over a product hero shot — is generic. The model has *seen* thousands of ads, but it does not *know* what a 1960s ad is. It has no facts. It was handed a frozen text embedding and asked to paint.

Now give the same prompt to a model that is, underneath, a large language model that happens to also speak in pixels. Before it draws anything, it can reason: *1960s ads used Futura-style sans-serif, prices were typically under ten dollars for appliances on installment plans, the layout put the headline at top with a "before/after" benefit line, washing machines of that era were top-loading with a wringer or early automatic.* It writes that reasoning out as text tokens, then — in the *same forward pass, the same weights, the same sequence* — switches to emitting image tokens that render a coherent, knowledge-grounded ad. That is the difference between a model that has memorized the *appearance* of the world and a model that has memorized the *facts* of the world and can draw them.

That second model is roughly what **HunyuanImage 3.0** is: Tencent's open-weights, ~80B-parameter Mixture-of-Experts (MoE), **native multimodal autoregressive** image generator — a single decoder-only transformer that models interleaved text and image tokens, generating pictures one token at a time in the very same network that handles language. It is the open model that most directly embodies the thesis running through this whole series' frontier chapters: *unify generation with an LLM, and you inherit the LLM's world knowledge and reasoning.* In this post we'll build up to it from its diffusion sibling **HunyuanImage 2.1** (a strong, fast MM-DiT with an aggressive high-compression VAE that does 2K), then dig into the science of native-AR generation — the chain rule over a unified token sequence, the image tokenizer, the MoE routing — and the brutally honest trade it makes: world knowledge and instruction following in exchange for sequential decoding speed and an 80B memory footprint.

![A single decoder-only Mixture-of-Experts transformer takes prompt tokens and emits both reasoning text tokens and image tokens, which a VAE detokenizer turns into a final 1024px image](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-1.png)

By the end you will understand *why* sharing one transformer with language gives world knowledge that a diffusion model conditioned on a frozen text encoder lacks, *how* the unified token space and MoE actually work (with a runnable PyTorch sketch of interleaved text+image token generation), and *when* native-AR is worth its cost versus reaching for a diffusion model like FLUX or HunyuanImage 2.1. We'll tie it back, as always, to the **generative trilemma** (quality × diversity × speed) and the diffusion stack — because native-AR makes a very specific, very deliberate bet on which corner of that triangle to sacrifice.

If you want the broader map of where autoregressive image generation sits, read [Autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) for the token paradigm and [Autoregressive vs Diffusion: The 2026 Showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown) for the head-to-head. This post is the deep-dive on the single open model that pushes the AR side of that showdown the hardest.

## 1. The setup: two paradigms wearing the same brand

"HunyuanImage" is not one model — it is a family with two fundamentally different engines under the hood, and conflating them is the first mistake people make. Let me separate them cleanly, because the rest of the post depends on keeping them apart.

**HunyuanImage 2.1** is a *diffusion* model. Architecturally it is a multimodal diffusion transformer (MM-DiT, the SD3/FLUX lineage we covered in [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe)). Its headline tricks are an aggressively high-compression VAE — squeezing the image into a very small latent so the transformer operates on far fewer tokens — and native 2K (2048px-class) generation. It is fast, it is aesthetic, and it conditions on text the way every diffusion model does: a frozen text encoder turns your prompt into an embedding, and the denoiser cross-attends to that embedding while it iteratively removes noise. It is a *very good* member of the family we already understand.

**HunyuanImage 3.0** is something else entirely. It is a *native multimodal autoregressive* model — a single decoder-only transformer (the same kind of architecture as a GPT-style LLM) with **Mixture-of-Experts** layers, totaling on the order of 80 billion parameters. It does not denoise. It does not run a separate frozen text encoder bolted onto a separate image network. Instead, it treats an image as a *sequence of tokens* drawn from the same vocabulary space as text, and it generates that sequence the way an LLM generates a paragraph: one token at a time, each token conditioned on everything before it. Because the *same weights* that learned language also produce the image tokens, the model carries genuine world knowledge into the act of drawing.

That is the whole story in one sentence: **HunyuanImage 2.1 is a fast diffusion painter; HunyuanImage 3.0 is a reasoning LLM that learned to paint.**

![A matrix comparing HunyuanImage 2.1 and 3.0 across paradigm, parameters, world knowledge, speed, and license, showing the diffusion-versus-native-AR split](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-4.png)

| | HunyuanImage 2.1 | HunyuanImage 3.0 |
|---|---|---|
| Paradigm | Diffusion (MM-DiT) | Native autoregressive (MoE transformer) |
| How it conditions | Frozen text encoder → cross-attention | Prompt tokens in the same sequence |
| Generation | Parallel iterative denoising | Sequential next-token decoding |
| Params (approx) | ~17B dense | ~80B total MoE (few B active/token) |
| World knowledge | Limited (encoder-bound) | LLM-grade (reasons + infers) |
| Speed | Fast, few-step capable | Slower (sequential decode) |
| Strength | 2K aesthetics, latency | Instruction following, knowledge, text-in-image |
| License | Open weights | Open weights |

Numbers like "~17B" and "~80B" are approximate and reflect the public framing of these releases; treat parameter counts as order-of-magnitude unless you check the exact config in the released weights. The *qualitative* split — diffusion vs native-AR — is the part that matters and is not in doubt.

Why does Tencent ship both? Because they sit at opposite corners of the generative trilemma. Diffusion's parallel denoising is the speed play; native-AR's shared-transformer reasoning is the *fidelity-to-intent* play. If you want a beautiful 2K render of an aesthetic concept *fast*, 2.1 is the tool. If you want an image that is *correct* — that respects facts, follows a multi-clause instruction, spells the words right, and infers the things you didn't say — 3.0 is the tool, and you pay for it in latency.

## 2. Why a frozen encoder caps what a diffusion model can know

To feel why native-AR matters, you have to feel the *limit* of the diffusion conditioning recipe. Let's make it concrete.

A standard text-to-image diffusion model — Stable Diffusion, FLUX, HunyuanImage 2.1 — has two separate brains. One is a **text encoder** (CLIP, T5, sometimes a small LLM), which is *frozen* during diffusion training or at most lightly tuned. It maps your prompt to a fixed-length sequence of embedding vectors. The other is the **denoiser** (a U-Net or DiT), which never reads raw text — it only ever sees those embedding vectors, which it cross-attends to while denoising.

This factorization is efficient and it works astonishingly well for *style and composition*. But it has a hard ceiling, and the ceiling is this: **the denoiser can only use the information the encoder bothered to put in the embedding, and the encoder was never trained to reason.** CLIP was trained to match images to captions; it knows that the *string* "Eiffel Tower" co-occurs with a certain shape, but it does not *know* the Eiffel Tower is 330 meters tall, was built for the 1889 World's Fair, or that its silhouette has four legs that curve inward. T5 is a better text model but it is still frozen, still summarizing the prompt into vectors, still not running any inference at generation time about facts the prompt left implicit.

![A diffusion model sees only a frozen text embedding and guesses unstated facts, while the native-AR model reads, reasons over its own world knowledge, and infers unstated details before drawing](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-2.png)

The way this shows up in practice is a family of failures that no amount of denoiser scaling fully fixes:

- **Knowledge-dependent content.** "Draw the flag of Bhutan." A diffusion model has probably seen few Bhutanese flags; it will produce a plausible-looking flag that is *wrong* (Bhutan's flag is a diagonal split with a white dragon — easy to get wrong). The model has no way to *look up* the fact; it can only interpolate appearances.
- **Text rendering.** "A storefront with the sign 'FRESH BAKED BREAD'." Diffusion models infamously produce garbled text because spelling is a *symbolic, sequential* task and the denoiser is a *spatial, parallel* one. The frozen encoder hands over a blurry notion of "there is a sign with words" and the denoiser hallucinates letter-shaped blobs.
- **Multi-clause instructions.** "A red cube on top of a blue sphere, to the left of a green cone, with exactly four cones in the background." Counting and relative positioning require something closer to *reasoning* over the prompt. Cross-attention on a frozen embedding routinely drops a clause, miscounts, or swaps attributes (the "attribute binding" failure — see [Text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning)).
- **Unstated inference.** "A breakfast that a marathon runner would eat the night before a race." The *right* answer requires knowing that runners carb-load — pasta, rice, bread — the night before. A diffusion model has no mechanism to perform that inference; it pattern-matches "breakfast" and "runner" into a generic plate of eggs.

The common thread: every one of these needs the model to *reason about facts at generation time*, and the diffusion recipe structurally cannot, because the part that knows facts (a big language model) is not the part that draws (the denoiser), and the bridge between them is a frozen, low-bandwidth embedding.

Native-AR collapses that gap by making the part that knows facts and the part that draws *the same network*. That is the entire bet.

## 3. The science: native-AR is just the chain rule over a longer alphabet

Strip away the marketing and a native-AR image generator is doing the most classical thing in generative modeling: **factorizing a joint distribution with the chain rule and learning each conditional with a neural net.** The only twist is that the "alphabet" includes image tokens alongside words.

Let a full multimodal sequence be a list of discrete tokens $x = (x_1, x_2, \dots, x_T)$, where each $x_t$ comes from a single shared vocabulary $\mathcal{V}$ that contains *both* text tokens (BPE subwords) *and* image tokens (we'll see where those come from in a moment). The autoregressive model factorizes the joint probability of the whole sequence exactly:

$$p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_1, \dots, x_{t-1}).$$

There is no approximation here — this is the chain rule of probability, true for any joint distribution. The modeling choice is to parameterize each conditional $p_\theta(x_t \mid x_{<t})$ with one decoder-only transformer that reads the prefix $x_{<t}$ and outputs a softmax over $\mathcal{V}$. Training is the same maximum-likelihood objective an LLM uses — minimize the negative log-likelihood, i.e. the next-token cross-entropy:

$$\mathcal{L}(\theta) = -\,\mathbb{E}_{x \sim \mathcal{D}} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).$$

The data $\mathcal{D}$ is a mix of pure text, image-caption pairs, and *interleaved* documents (text and images woven together, like a webpage). When the model is asked to generate an image, the prompt is a text prefix and the image is the suffix the model must predict, token by token.

Here is the conceptual payoff that the math makes inevitable. Because $p_\theta(x_t \mid x_{<t})$ conditions on the *entire prefix*, every image token the model emits is conditioned on:

1. the text prompt,
2. *any reasoning text the model itself generated* (it can write out a plan before drawing), and
3. every image token already drawn.

That second point is the magic. The model can emit a chain of reasoning tokens — "this is a 1960s ad, so the price should be under ten dollars and the font should be a geometric sans-serif" — and then condition the image tokens on that reasoning. It is, quite literally, **chain-of-thought for pixels.** A diffusion model cannot do this because it has no autoregressive prefix to put a plan into; it commits to a noisy latent and denoises in parallel.

Contrast the two objectives directly. Diffusion (from [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm)) learns to predict noise:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{x_0, \epsilon, t}\big[\,\lVert \epsilon - \epsilon_\theta(x_t, t, c)\rVert^2\,\big],$$

where $c$ is the *frozen* conditioning embedding. Notice $c$ enters as a fixed input — the network cannot grow or revise it. The AR objective has no such $c$; the "condition" is the *generated prefix itself*, which the model writes and can reason within. That structural difference is the whole reason native-AR inherits world knowledge: the conditioning is *live computation*, not a frozen vector.

### Why the joint factorization is the source of the knowledge

It is worth making the world-knowledge claim *provable* rather than asserted, because it is the crux of the whole post. Consider the joint distribution the native-AR model learns over a text prompt $c$ and an image token sequence $y = (y_1, \dots, y_M)$. The chain rule lets us write it as

$$p_\theta(c, y) = \underbrace{p_\theta(c)}_{\text{a language model}} \cdot \underbrace{\prod_{m=1}^{M} p_\theta(y_m \mid c, y_{<m})}_{\text{an image model conditioned on text}}.$$

The single network $\theta$ is trained to maximize the likelihood of *both* factors at once — the same parameters are pushed to be a good language model *and* a good conditional image model. Gradient signal from the text factor literally shapes the weights that the image factor reuses. So when the model conditions an image token on the prompt, the representation it conditions on is the *same* representation that learned, from billions of text tokens, what a "1960s ad" or "Bhutan's flag" or "a marathon runner's pre-race meal" actually *is*.

A diffusion model factorizes differently and that difference is the ceiling. It models $p_\phi(y \mid c)$ where $c = f_\psi(\text{prompt})$ is produced by a *separate, frozen* encoder $\psi \neq \phi$. The image network $\phi$ never receives gradient that would teach it facts; it only learns to map a fixed embedding to pixels. The knowledge that exists lives in $\psi$, was frozen, and was never trained to *reason* — only to *embed*. There is no path for generation-time inference because there is no autoregressive prefix and no shared parameters. The math makes the limit structural, not incidental: **you cannot reason at generation time with a network that has no generated prefix to reason within and no parameters shared with a language model.**

### What does native-AR cost, exactly?

The chain rule is a blessing for knowledge and a curse for speed. To generate $T$ image tokens you need $T$ sequential forward passes — each token waits for the previous one, because $x_t$ depends on $x_{<t}$. With KV-caching the per-step cost is roughly the cost of one transformer forward over the cached context, but you still pay $O(T)$ steps that *cannot be parallelized across token positions*.

Diffusion, by contrast, generates *all* spatial positions in parallel each step and needs only $N$ steps where $N$ is the number of denoising steps (20–50 for good samplers, or 1–4 after distillation — see [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it)). For a 1024px image an AR model might emit on the order of a few thousand image tokens; that is a few thousand sequential steps versus a few dozen parallel ones. This is the central latency tax of native-AR, and no amount of cleverness fully erases it (though speculative decoding, parallel/blockwise token prediction, and next-scale schemes like VAR chip at it).

#### Worked example: counting the sequential steps

Suppose HunyuanImage 3.0 emits a $64 \times 64$ grid of image tokens for a 1024px image (a 16× spatial compression from the VAE), so $T_\text{img} = 4096$ image tokens, plus say 300 reasoning/prompt tokens, giving ~4,400 sequential decode steps. At a hypothetical 50 ms per step on a single high-end GPU with the full 80B MoE resident, that is $4400 \times 0.05 \approx 220$ seconds — call it three to four minutes per image, ballpark. A diffusion model like HunyuanImage 2.1 at 25 steps with parallel denoising might land a 1024px image in single-digit seconds on the same hardware. That ~30–60× latency gap is the price of admission for the reasoning, and it is why you do not reach for native-AR when you need a cheap thumbnail. (Every number here is an order-of-magnitude illustration to make the *scaling* visible, not a measured benchmark — exact latency depends on batch size, expert count, sequence length, and serving stack.)

## 4. Where image tokens come from: the tokenizer is the whole ballgame

An LLM tokenizes text with byte-pair encoding — a deterministic, lossless-ish map from strings to integer ids. To make an *image* into tokens you need an analogous map from pixels to a sequence of integer ids that (a) is short enough to be tractable and (b) can be *inverted* back to pixels with acceptable quality. This is the **image tokenizer**, and it is doing most of the heavy lifting that makes native-AR possible at all.

![Text passes through a BPE tokenizer and images through a VAE plus quantizer, both producing token ids that join one shared vocabulary the MoE decoder predicts over](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-5.png)

The standard recipe, which we covered in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) and which goes back to VQ-VAE/VQ-GAN, is:

1. **Encode** the image with a convolutional encoder into a small spatial grid of continuous feature vectors — e.g. a $1024 \times 1024$ image becomes a $64 \times 64$ grid of $d$-dimensional vectors. That's a steep spatial compression, which is exactly what you want: fewer positions to generate.
2. **Quantize** each vector to the nearest entry in a learned codebook of $K$ vectors (vector quantization). Now each grid cell is a single integer in $\{1, \dots, K\}$ — an *image token*. The codebook is the "visual vocabulary."
3. **Decode** a grid of codebook ids back to pixels with a convolutional decoder (often trained with a GAN loss, hence VQ-*GAN*, to keep textures sharp).

The autoregressive transformer never touches pixels. It only ever sees and predicts *codebook ids*, exactly as it predicts BPE ids for text. The shared vocabulary $\mathcal{V}$ is literally the concatenation: text BPE ids occupy id range $[0, V_\text{text})$ and image codebook ids occupy $[V_\text{text}, V_\text{text} + K)$, plus a few special tokens like `<image_start>` and `<image_end>` that mark mode switches. To the transformer it is all just one big softmax.

A few engineering realities that decide whether this works:

- **The codebook size $K$ and the compression ratio set a hard quality/length trade.** A bigger codebook and a finer grid preserve more detail but explode the sequence length (more tokens to generate, slower) and the softmax width (harder to learn). HunyuanImage 2.1's claim to fame on the *diffusion* side is an aggressive high-compression VAE; on the *AR* side, the analogous lever is how aggressively the tokenizer compresses, because every factor of compression directly cuts the number of sequential decode steps.
- **Quantization is lossy and the loss is *structured*.** VQ throws away the residual between a vector and its nearest codebook entry. That shows up as a ceiling on reconstructable detail: even a *perfect* AR model can only produce images as good as the tokenizer can reconstruct. This is why some 2026-era native-AR systems move toward continuous or hybrid tokenizers and why "no-VQ" approaches (like MAR, masked autoregression over continuous tokens) exist — to dodge the quantization ceiling.
- **The tokenizer is trained *first and frozen*** (or nearly so) before the big AR transformer trains on top. So in a real sense the tokenizer defines the "image language" and the 80B transformer learns to *speak* it fluently, the way you'd learn to write in a fixed alphabet.

The exact tokenizer details of HunyuanImage 3.0 (codebook size, whether it is pure VQ or a continuous/hybrid scheme, the precise compression factor) are model-specific and you should read the released config rather than trust a blog's recollection — but the *shape* of the design, a VAE-plus-quantizer feeding a shared vocabulary, is the standard native-AR pattern and is the right mental picture.

### A toy tokenizer round-trip in PyTorch

Here is the tokenize → detokenize round-trip in miniature, so the design has weight. This is a sketch of the *interface*, not the production model:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQImageTokenizer(nn.Module):
    """Toy VQ tokenizer: pixels <-> codebook ids. The AR transformer
    only ever sees the integer ids this returns."""
    def __init__(self, in_ch=3, latent_dim=256, codebook_size=16384, downscale=16):
        super().__init__()
        # Encoder: image -> small grid of latent_dim vectors (downscale x).
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),  # /16 total
        )
        # Learned visual vocabulary (the codebook).
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        # Decoder: codebook vectors -> pixels (GAN-trained in practice).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, in_ch, 4, stride=2, padding=1),
        )

    def tokenize(self, x):                      # x: [B, 3, H, W] in [-1, 1]
        z = self.encoder(x)                     # [B, D, H/16, W/16]
        B, D, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)        # [B*h*w, D]
        # Nearest codebook entry per spatial cell -> integer ids.
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        ids = dist.argmin(dim=1).view(B, h, w)               # [B, h, w] ints
        return ids                                           # <- image tokens

    def detokenize(self, ids):                  # ids: [B, h, w] integers
        z_q = self.codebook(ids).permute(0, 3, 1, 2)         # [B, D, h, w]
        return self.decoder(z_q)                             # [B, 3, H, W]

tok = VQImageTokenizer()
img = torch.randn(1, 3, 256, 256)
ids = tok.tokenize(img)            # e.g. [1, 16, 16] -> 256 image tokens
recon = tok.detokenize(ids)        # back to [1, 3, 256, 256]
print(ids.shape, ids.min().item(), ids.max().item(), recon.shape)
```

The two functions you care about are `tokenize` (the AR model's *input/target* for images) and `detokenize` (the final step that turns the model's predicted ids back into a picture). Everything the 80B transformer does happens in id-space *between* these two calls.

## 5. The unified transformer: one decoder-only stack, two modalities

With a tokenizer in hand, the architecture of HunyuanImage 3.0 is, at the block level, just an LLM. A decoder-only transformer: token embedding table (now spanning text + image ids), rotary or learned positional encodings, a stack of causal self-attention + feed-forward blocks, and a final linear head projecting to the shared vocabulary softmax. The two things that make it a *frontier* model rather than a textbook one are (1) it is huge and uses Mixture-of-Experts to be huge affordably, and (2) it is trained on interleaved multimodal data so the *same weights* model both modalities.

![Generation walks from prompt tokens through a text reasoning phase, a mode-switch token, autoregressive image-token decoding, and a VAE detokenize step to the final image](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-3.png)

Let's walk the generation flow top to bottom, because the interleaving is the part that's genuinely new relative to a pure text LLM.

1. **Prompt tokens in.** Your text prompt (plus any system instruction) is BPE-tokenized into the shared vocabulary and fed as the prefix.
2. **Reasoning phase (optional "think").** The model can emit text tokens that plan the image — inferring unstated details, resolving ambiguity, recalling facts. This is ordinary text generation; it is conditioned on the prompt and produces a free-form plan. Because it is the *same* weights as the language model, this plan is genuinely informed.
3. **Mode switch.** A special token (`<image_start>` or equivalent) tells the model to begin emitting image tokens. From here the softmax mass concentrates on the image-id range of the vocabulary.
4. **Autoregressive image decode.** The model emits image tokens one at a time, each conditioned on the prompt, the reasoning, and all prior image tokens. This is the slow part — thousands of sequential steps — but it is also where the conditioning pays off, because every patch is informed by the full prefix.
5. **Detokenize.** The grid of image ids is reshaped and passed through the VAE decoder to produce pixels.

The crucial difference from a diffusion model: in diffusion, *conditioning is an input vector* that the network attends to. In native-AR, *conditioning is the prefix the model itself wrote.* The model is, in a real sense, talking to itself — reasoning in words, then realizing that reasoning in pixels — within a single causal sequence.

### Why Mixture-of-Experts, and what it changes

An 80B dense transformer would be brutal to train and serve. **Mixture-of-Experts** is the standard way to get the *capacity* of a huge model while paying only a fraction of the *compute* per token. The idea: replace the single feed-forward network (FFN) in each transformer block with $E$ parallel FFNs ("experts") plus a small **router** that, for each token, picks the top-$k$ experts (often $k=1$ or $2$) to actually run. The token's FFN output is a weighted combination of just those $k$ experts.

The accounting is what makes it worth it. If you have $E=64$ experts but route each token to only $k=2$ of them, you have ~32× the FFN parameters of a dense model but only ~2 experts' worth of FFN compute per token. So an "80B" MoE might activate only a handful of billion parameters per token — closer to the compute of a much smaller dense model, with the knowledge capacity of a much larger one. That capacity is exactly what you want for world knowledge: more parameters means more facts can be stored.

![A Mixture-of-Experts router sends each token to top-k of many experts, keeping active parameters small per token while total weights, sequential decoding, and memory set the real serving cost](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-8.png)

But MoE moves the cost from *compute* to *memory and complexity*:

- **All experts must be resident in memory** even though only a few run per token, because you don't know in advance which the router will pick. So the *memory* footprint is the full 80B, which is why you serve a model like this across multiple GPUs.
- **Routing must be load-balanced.** If the router sends most tokens to a few favorite experts, the rest sit idle and you've wasted capacity. Training uses an auxiliary load-balancing loss to spread tokens across experts. This is a well-known MoE failure mode and a real tuning headache.
- **Sequential decoding still dominates latency.** MoE cuts per-token compute, but it does nothing about the $O(T)$ sequential steps of AR image generation. The two costs are orthogonal: MoE makes each step cheaper; it does not make there be fewer steps.

So the net "compute reality" of HunyuanImage 3.0 is: total weights ~80B (sets memory), active params a few billion per token (sets per-step compute), and a few thousand sequential steps per image (sets wall-clock latency). It is a model you serve on a multi-GPU node, not a laptop — the opposite end of the spectrum from HunyuanImage 2.1's efficiency story.

### Two-dimensional positions for a one-dimensional sequence

A subtle but important wrinkle: text is naturally one-dimensional (a line of tokens) but an image is two-dimensional (a grid of patches), and the transformer flattens the image grid into the same 1D sequence as the text. If you just use ordinary 1D positional encodings, the model has no clean signal that token at flattened position 100 is *directly below* token at position 36 in a $64$-wide grid — that spatial adjacency is buried in arithmetic the model has to learn the hard way. Modern native-multimodal models address this with **2D positional encodings** for the image span: each image token carries its $(row, col)$ position, often via a 2D variant of rotary position embeddings (RoPE), so the model knows the grid geometry explicitly. This is the AR analog of how a DiT patchifies and position-encodes a 2D latent. The exact scheme HunyuanImage 3.0 uses is in its config, but the *need* is universal: flattening a grid into a line loses spatial structure, and you have to put it back through the position encoding or the model wastes capacity rediscovering that "next row" is a jump of one grid-width in the flattened index.

### What the model trains on

The world knowledge does not appear by magic — it comes from the data mixture. A native-multimodal model like this trains on three kinds of sequences, and the *interleaving* is what fuses the modalities:

- **Pure text.** Ordinary language-model data — books, web text, code. This is where the bulk of the *factual and reasoning* capability is laid down, exactly as in any LLM.
- **Image-caption pairs.** A caption (text tokens) followed by the image's tokens, or vice versa. This teaches the alignment between words and visual tokens — the grounding that lets "red mug" produce red-mug image tokens.
- **Interleaved documents.** Webpage-like data where text and images alternate naturally (an article with embedded figures). This is the secret sauce: it teaches the model to *switch modes fluidly* and to use surrounding text to inform an image and vice versa — the same skill that, at inference, lets it write a reasoning plan and then draw it.

Because all three flow through one next-token objective, the model never learns "text mode" and "image mode" as separate skills bolted together — it learns *one* generative process over a mixed alphabet. That unity is precisely why reasoning transfers into drawing: there is no seam to cross.

#### Worked example: MoE active-parameter accounting

Take a stylized MoE block with hidden size $d = 6144$, an FFN expansion of $4\times$ (so each expert FFN is two matrices of shape roughly $6144 \times 24576$), $E = 64$ experts, and top-$k = 2$ routing. Each expert FFN holds about $2 \times 6144 \times 24576 \approx 3.0 \times 10^8$ params, so 64 experts hold ~$1.9 \times 10^{10}$ params *in that one block's FFN* — and the model has many blocks, which is how you reach tens of billions total. But per token, only $k=2$ experts run, so the *active* FFN compute is $2 \times 3.0\times10^8 = 6.0\times10^8$ params' worth — about $1/32$ of the block's FFN capacity. That ratio, repeated across the stack, is exactly how an 80B-total model can run at the per-token compute of a single-digit-billion dense model. The catch, again: you still hold all $1.9\times10^{10}$ per block in memory. (These dimensions are illustrative round numbers chosen to show the arithmetic, not HunyuanImage's exact config.)

## 6. A runnable sketch: interleaved text + image token generation

Let's make the inference flow concrete with a PyTorch sketch of the *generation loop*. This is deliberately a minimal decoder-only model that emits interleaved reasoning text and image tokens, then detokenizes — the conceptual skeleton of native-AR. It will not produce art (it's untrained and tiny), but it shows exactly where the text/image mode switch lives and how detokenization closes the loop.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Shared vocabulary layout ----
V_TEXT      = 32000                 # text BPE ids: [0, 32000)
K_IMAGE     = 16384                 # image codebook ids
IMG_OFFSET  = V_TEXT               # image ids live at [V_TEXT, V_TEXT + K_IMAGE)
IMG_START   = V_TEXT + K_IMAGE     # special: begin image
IMG_END     = V_TEXT + K_IMAGE + 1 # special: end image
VOCAB       = V_TEXT + K_IMAGE + 2

class TinyMoEDecoder(nn.Module):
    """A miniature decoder-only transformer over the shared vocab.
    One model, one softmax, both modalities."""
    def __init__(self, vocab=VOCAB, d=512, n_layers=6, n_heads=8, n_experts=8, top_k=2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(8192, d)
        self.blocks  = nn.ModuleList([MoEBlock(d, n_heads, n_experts, top_k)
                                      for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)   # predicts text AND image ids

    def forward(self, ids):                            # ids: [B, T]
        T = ids.size(1)
        pos = torch.arange(T, device=ids.device)
        h = self.tok_emb(ids) + self.pos_emb(pos)[None]
        mask = torch.full((T, T), float("-inf"), device=ids.device).triu(1)  # causal
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.norm(h))                 # [B, T, VOCAB]

class MoEBlock(nn.Module):
    def __init__(self, d, n_heads, n_experts, top_k):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
                                                    nn.Linear(4*d, d))
                                      for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts)
        self.top_k = top_k

    def forward(self, h, mask):
        a, _ = self.attn(self.n1(h), self.n1(h), self.n1(h), attn_mask=mask)
        h = h + a
        x = self.n2(h)
        # --- top-k MoE routing: each token picks top_k experts ---
        logits = self.router(x)                        # [B, T, E]
        weights, idx = logits.softmax(-1).topk(self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            for e, expert in enumerate(self.experts):
                sel = (idx[..., slot] == e)            # tokens routed to expert e
                if sel.any():
                    out[sel] += weights[..., slot][sel, None] * expert(x[sel])
        return h + out
```

And the generation loop — the part that actually interleaves reasoning text and image tokens:

```python
@torch.no_grad()
def generate(model, prompt_ids, n_think=64, n_image=256, temp=0.9, device="cuda"):
    """Emit some reasoning text, switch modes, then decode image tokens."""
    model.eval()
    ids = prompt_ids.to(device)                        # [1, T_prompt]

    # 1) Reasoning phase: free-form text tokens (the 'think before drawing').
    for _ in range(n_think):
        logits = model(ids)[:, -1]                     # next-token distribution
        logits[:, V_TEXT:] = float("-inf")             # restrict to TEXT ids
        nxt = torch.multinomial((logits / temp).softmax(-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == IMG_START:                    # model chose to start drawing
            break

    # 2) Force the mode switch if it didn't emit IMG_START on its own.
    if ids[0, -1].item() != IMG_START:
        ids = torch.cat([ids, torch.tensor([[IMG_START]], device=device)], dim=1)

    # 3) Image phase: decode image tokens (restricted to the image id range).
    img_ids = []
    for _ in range(n_image):
        logits = model(ids)[:, -1]
        logits[:, :V_TEXT] = float("-inf")             # restrict to IMAGE ids
        logits[:, IMG_START:] = float("-inf")
        nxt = torch.multinomial((logits / temp).softmax(-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        img_ids.append(nxt.item() - IMG_OFFSET)        # back to [0, K_IMAGE)

    return ids, torch.tensor(img_ids)                  # full seq + raw codebook ids

# Detokenize the codebook ids back to pixels with the VQ tokenizer from section 4.
# h = w = int(len(img_ids) ** 0.5)
# image = tokenizer.detokenize(img_ids.view(1, h, w))   # [1, 3, H, W]
```

Three things to notice, because they are the whole architecture in miniature:

- **One model, one `head`, two modes.** The same `TinyMoEDecoder` predicts text ids and image ids from one softmax. The only thing that changes between phases is *which slice of the vocabulary we allow* (the `-inf` masking). Real systems learn to switch modes via the `<image_start>` token rather than hard masking, but the mechanism is the same: it is all one sequence.
- **The reasoning prefix conditions the image.** Step 3's image tokens are generated with steps 1–2 still in the context. Anything the model "thought" in the text phase is visible to every image token it draws. That is the chain-of-thought-for-pixels mechanism, made literal.
- **Detokenization closes the loop.** The model's output is just integers; the VQ decoder from section 4 is what makes them pixels. This is the architectural boundary between "the LLM" and "the renderer," and it is the *only* non-transformer piece in the whole pipeline.

### The KV-cache is what makes sequential decoding survivable

The naive generation loop above recomputes attention over the entire prefix at every step — that is $O(T^2)$ total work to emit $T$ tokens and would be hopeless for thousands of image tokens. Every real AR serving stack uses a **KV-cache**: after processing a token, you store its key and value vectors per layer, so the next step only computes the query for the *one new* token and attends against the cached keys/values. That turns each step into roughly one transformer forward over a single new position (plus an attention read over the cache), making the per-step cost roughly constant in the prefix length rather than growing quadratically. My sketch omits it for clarity; a production loop threads a `past_key_values` object through `model.generate` and the framework handles it.

But the KV-cache also reveals where native-AR's *memory* pressure comes from at inference, separate from the weight memory. For a sequence of length $T$, the cache holds $T$ key and value vectors per layer per attention head. For thousands of image tokens across dozens of layers, that cache is large — and it grows with every token emitted. So serving HunyuanImage 3.0 is a two-front memory battle: the 80B of MoE weights (constant, shardable across GPUs) *plus* a KV-cache that swells as the image decodes. This is why batching many image generations on one node is harder than batching short text completions: each in-flight image holds a big, growing cache. It is one more reason the economics of native-AR favor *fewer, higher-value* generations over high-throughput streams.

There is active research on shrinking this: parallel/blockwise decoding (emit several image tokens per step), speculative decoding (a small model proposes, the big one verifies), and next-scale prediction (VAR-style, where each step produces a whole resolution level). All of them attack the same enemy — the $O(T)$ sequential steps — and all of them are the reason to expect native-AR latency to keep falling even though the paradigm is intrinsically sequential.

### Where the real weights load

You don't reimplement this — you load the released checkpoint. The open HunyuanImage weights are distributed on the Hugging Face Hub under Tencent's organization, and the practical loading path mirrors any large multimodal model:

```python
# Conceptual loading flow for the open HunyuanImage 3.0 weights.
# Check the official model card for the exact repo id, class, and any
# trust_remote_code / custom-pipeline requirements before running.
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

REPO = "tencent/HunyuanImage-3.0"   # confirm the exact id on the HF model card

processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    REPO,
    torch_dtype=torch.bfloat16,     # bf16 to fit; 80B MoE needs multi-GPU
    device_map="auto",              # shard experts across available GPUs
    trust_remote_code=True,         # native-multimodal models ship custom code
)

inputs = processor(text="A vintage 1960s washing-machine ad with a price and tagline",
                   return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4500)  # text plan + image tokens
image = processor.decode_image(out)                  # detokenize to pixels
image.save("ad.png")
```

The exact class names, processor method (`decode_image` is a stand-in), and whether it uses `AutoModelForCausalLM` or a model-specific class depend on how Tencent packaged the release — **read the model card.** The shape is what's portable: a `from_pretrained` over a sharded 80B MoE, `device_map="auto"` to spread experts across GPUs, `bfloat16` to halve memory, and a `generate` call whose output includes image tokens that a processor detokenizes. Plan for multi-GPU; this is not a single-24GB-card model.

## 7. The payoff: what world knowledge buys on real prompts

Theory is nice; let's talk about what actually changes in the outputs. The category of prompts where native-AR pulls decisively ahead of diffusion is **knowledge-dependent and instruction-heavy** generation. Here are the buckets, with honest caveats.

**Text rendering inside images.** Because the model spells using the same machinery it uses to write text, native-AR systems (and the closed GPT-Image / Nano Banana lineage they parallel — see [the closed frontier](/blog/machine-learning/image-generation/gpt-image-and-nano-banana-the-closed-frontier)) render legible, correctly-spelled text far more reliably than classic diffusion. "A poster that says 'GRAND OPENING SATURDAY 9AM'" comes out readable. Diffusion has improved here (FLUX and Qwen-Image are notably better than SD1.5), but it remains a structural advantage of the AR side because spelling is intrinsically a sequential, symbolic task.

**Knowledge-grounded content.** Flags, logos, maps, scientific diagrams, period-accurate detail, brand-correct color schemes — anything where the *right* answer is a fact, not an aesthetic. A model that *knows* the fact (because the same weights answer the text question "what does Bhutan's flag look like?") can draw it; a diffusion model conditioned on a frozen encoder can only interpolate appearances and will confidently produce a wrong-but-plausible flag.

**Multi-clause instruction following.** "A wooden desk with, from left to right, a red mug, a closed silver laptop, and exactly three yellow sticky notes; a window behind it showing rain." The reasoning prefix lets the model parse the clause structure, plan the layout, and check the count before committing. Diffusion routinely drops a clause or miscounts; this is the [GenEval / T2I-CompBench](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) compositionality story.

**Unstated inference.** The marathon-breakfast prompt from the intro. The model infers what you meant, not just what you said, because inference is what LLMs do.

#### Worked example: tracing the reasoning-then-draw on the 1960s ad

Walk the intro prompt through the native-AR pipeline concretely. The prompt tokens enter: *"A vintage 1960s American magazine advertisement for a washing machine, with a tagline and a price in old dollars, the brand logo top-left, drawn in the flat illustration style of that era."* In the reasoning phase, the model emits text tokens that read something like an internal note: *the era implies a geometric sans-serif headline; appliance prices in 1960 were typically under \$300 and often advertised with a "$X down" installment line; mid-century ad layout puts the headline across the top, the product hero shot center, the benefit line and price lower-right, the logo upper-left as requested; the illustration style is flat color with bold outlines and limited palette.* None of that was in the prompt — it is recalled and inferred from the language model's training. Then the mode-switch token fires and the model decodes image tokens *conditioned on that plan*, so the headline font, the price format, the logo placement, and the palette all come out era-consistent. A diffusion model gets none of this scaffolding: it maps a frozen "1960s washing machine ad" embedding straight to a noisy latent and denoises, so the price string and tagline are decorative blobs and the layout is whatever the training-set average looked like. The *same prompt* produces a factually-grounded composition from the AR model and a vibes-only one from the diffusion model — and the difference is entirely the reasoning prefix.

![A matrix comparing native-AR and diffusion across world knowledge, instruction following, aesthetic polish, latency per image, and text-in-image rendering, showing the trade-off split](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-6.png)

Now the honest other side, because this series does not do hype:

- **Raw aesthetic polish and texture** is often *still* a diffusion strength. Years of sampler tuning, guidance tricks, and aesthetic fine-tuning have made diffusion models extraordinarily good at "make this beautiful." A native-AR model can match or exceed them on *correctness* while sometimes lagging on the last 10% of painterly polish — though this gap is closing fast.
- **Latency.** As established, native-AR is sequential and slow. For a high-throughput service generating millions of thumbnails, diffusion's parallel denoising (especially distilled to 1–4 steps) is dramatically cheaper.
- **Diversity per prompt.** Diffusion's stochastic sampling naturally gives varied outputs across seeds. AR sampling is varied too (via temperature/top-p), but the relationship between sampling temperature and image diversity is less well-understood and easier to get wrong (too high → incoherent, too low → repetitive).
- **The quantization ceiling.** If the tokenizer is pure VQ, the AR model is capped by the codebook's reconstruction quality, full stop.

The right framing is not "AR beats diffusion" or vice versa — it's that they sit at different corners of the trilemma. Native-AR trades **speed** for **fidelity-to-intent and world knowledge**. Diffusion trades **world knowledge** for **speed and aesthetic tunability**. HunyuanImage shipping *both* is Tencent hedging across that trade explicitly.

## 8. HunyuanImage 2.1 as context: the diffusion sibling done well

It's worth spending a section on 2.1, because it is both a strong model in its own right and the foil that makes 3.0's choices legible.

HunyuanImage 2.1 is a diffusion MM-DiT in the [SD3/FLUX lineage](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe): joint attention over image and text tokens inside transformer blocks, flow-matching-style training, and — its distinguishing feature — an aggressively **high-compression VAE** that lets the transformer operate on a very small latent grid even for high-resolution images. The compression is the speed lever: fewer latent tokens means cheaper attention (which is quadratic in token count) and fewer positions to denoise, which is how it reaches native 2K (2048px-class) generation without the cost exploding.

Architecturally, the recurring tension in high-compression VAEs is the *reconstruction ceiling*: the more you compress, the more the VAE decoder has to hallucinate fine detail, and the easier it is to get mushy textures or lost high-frequency content. The art is pushing compression as far as quality allows. HunyuanImage 2.1's contribution on this axis is real engineering — getting a very high compression ratio to still reconstruct crisp 2K images. (We covered the general principle in [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) and the perceptual-compression argument in [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).)

Here's the thing to internalize: **2.1's VAE compresses for the *denoiser's* benefit (fewer tokens to denoise in parallel), while 3.0's tokenizer compresses for the *autoregressor's* benefit (fewer tokens to decode sequentially).** Same VAE-compression idea, opposite paradigm, and the *reason* compression matters is different in each. In diffusion, compression buys cheaper parallel attention. In native-AR, compression buys *fewer sequential steps* — which is even more precious, because sequential steps are the thing you cannot parallelize away.

Practically, if you wanted to *use* 2.1, it slots into the `diffusers` mental model you already have — a pipeline with a transformer, a VAE, text encoders, and a flow-matching scheduler. A sketch of the shape (confirm exact class names on the model card):

```python
import torch
from diffusers import DiffusionPipeline

# HunyuanImage 2.1: a diffusion MM-DiT pipeline. Confirm the exact pipeline
# class and repo id on the model card; the *shape* is a standard diffusers call.
pipe = DiffusionPipeline.from_pretrained(
    "tencent/HunyuanImage-2.1",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe.enable_model_cpu_offload()        # fit the model on a single big GPU
# pipe.enable_vae_tiling()             # for very high-res VAE decode

image = pipe(
    prompt="A serene mountain lake at golden hour, ultra-detailed, 2K",
    num_inference_steps=25,            # diffusion: parallel denoising steps
    guidance_scale=4.5,                # classifier-free guidance
    height=2048, width=2048,
).images[0]
image.save("lake.png")
```

Notice everything that is *absent* compared to the 3.0 generation loop: no reasoning phase, no token-by-token decode, no mode switch. You set a step count, the scheduler runs $N$ parallel denoising passes, the VAE decodes once, done. That is the entire ergonomic and computational difference between the two paradigms, sitting side by side under one brand.

![A tree of the HunyuanImage family forking into a diffusion branch with the 2.1 MM-DiT and refiner, and an autoregressive branch with the 3.0 MoE model and its think mode](/imgs/blogs/hunyuanimage-and-native-multimodal-generation-7.png)

## 9. Case studies and real numbers

Let me ground this in the broader 2026 native-AR landscape, because HunyuanImage 3.0 is one point in a fast-moving field and the comparisons sharpen the picture. I'll be explicit about what is well-established versus what is approximate.

**Native-AR is the GPT-Image strategy, made open.** OpenAI's GPT-Image and Google's Nano Banana (Gemini's image generation) are widely understood to be *native multimodal* generators — image generation built into a large multimodal model rather than a separate diffusion network — which is exactly why they are so strong at text rendering and instruction following. HunyuanImage 3.0 is the most prominent *open-weights* model pursuing the same architecture, which is its real significance: it lets the research community study and build on native-AR generation without a closed API. (See [the closed frontier](/blog/machine-learning/image-generation/gpt-image-and-nano-banana-the-closed-frontier) for the GPT-Image / Nano Banana side.)

**The unified-model research lineage.** HunyuanImage 3.0 sits downstream of a clear research arc: **Chameleon** (Meta, 2024) showed a single transformer over interleaved image+text tokens; **Transfusion** (Meta, 2024) combined AR text with diffusion *within* one transformer; **Janus / Janus-Pro** (DeepSeek) decoupled the visual encoders for understanding vs generation; **Emu3** trained a single next-token model across text, image, and video. HunyuanImage 3.0 scales the "pure native-AR generation" branch of this tree to ~80B with MoE. We surveyed this convergence in [the 2026 showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown); this post is the deep-dive on its strongest open exemplar.

**The VAR / next-scale speed angle.** One concrete answer to native-AR's latency tax is **VAR (Visual Autoregressive)**, which predicts images *next-scale* (coarse-to-fine resolution) rather than next-token (raster order). VAR reported strong ImageNet FID (in the low single digits at $256^2$, competitive with diffusion) while needing far fewer autoregressive steps than raster-order AR, because each "step" predicts a whole scale rather than one patch. This matters for HunyuanImage's family because next-scale and parallel/blockwise decoding are the most promising routes to making native-AR fast enough for production. (Numbers like "low-single-digit FID at $256^2$" are from the VAR paper's reported results; verify the exact figures against the paper.)

**The open-release reality check.** Releasing an 80B MoE openly is a genuine contribution, but it comes with a sober footnote: most people *cannot run it.* The weights alone, even in bfloat16, are on the order of 160 GB (roughly two bytes per parameter), which does not fit on a single 24 GB or 48 GB consumer card or even a single 80 GB datacenter GPU. You need a multi-GPU node with tensor/expert parallelism to hold the experts, plus headroom for a KV-cache that grows through the image decode. Quantization to 4-bit can roughly quarter the weight memory (toward ~40 GB) and brings it within reach of a small number of high-end cards, at some quality cost — the same quantization story we tell for diffusion in [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference), but harder here because the model is so much larger and the sequential decode means quantization error can compound token-over-token. So "open weights" for a model this size means *open to organizations and serious labs*, not *open to a hobbyist's gaming PC*. That is a real difference from FLUX or SDXL, which a determined hobbyist can run on a single 24 GB card. The openness matters most for *research* — being able to inspect and fine-tune the architecture — more than for *casual local generation*.

**The honest aesthetic comparison.** On pure aesthetic benchmarks and human-preference scores, the best diffusion models (FLUX, the SD3.5 line, Midjourney) and the best native-AR models trade blows, and the ranking flips depending on the prompt distribution. On *knowledge-dependent and text-heavy* prompts, native-AR (GPT-Image-class, and HunyuanImage 3.0 among open models) tends to lead. On *pure aesthetic, fast, stylized* generation, diffusion tends to lead. This is the central finding of the showdown and it is *not* going to resolve into a single winner soon — the frontier is fusion (models that do both), not a knockout.

| Model | Paradigm | Params (approx) | Open? | Strongest at |
|---|---|---|---|---|
| HunyuanImage 2.1 | Diffusion MM-DiT | ~17B dense | Yes | Fast 2K aesthetics |
| HunyuanImage 3.0 | Native-AR MoE | ~80B total | Yes | Knowledge, instructions, text |
| FLUX.1 / FLUX.2 | Diffusion (rectified flow) | ~12B | Yes (dev) | Aesthetics, prompt fidelity |
| Qwen-Image | Diffusion MM-DiT | ~20B | Yes | Text rendering, editing |
| GPT-Image | Native multimodal | Undisclosed | No | Instructions, text, knowledge |
| Nano Banana (Gemini) | Native multimodal | Undisclosed | No | Editing, instructions, knowledge |

Treat all parameter counts as approximate / order-of-magnitude; the closed models are undisclosed and the open ones should be checked against their configs. The *paradigm* column is the load-bearing one.

#### Worked example: choosing a model for a real job

Say you're building a feature that generates **product packaging mockups with the brand name and a regulatory text block printed on them**. This is text-heavy and instruction-heavy: the brand name must be spelled correctly, the layout must follow packaging conventions, and the legal text must be legible. A diffusion model will mangle the text and may invent a wrong layout. A native-AR model — HunyuanImage 3.0 in the open camp, or GPT-Image in the closed camp — will render the text correctly and reason about the layout. The latency cost (a few minutes per image versus a few seconds) is *acceptable* here because mockups are generated occasionally, not at scale, and correctness is the whole point. **Decision: native-AR.** Flip the scenario to "generate 50,000 decorative background textures for a wallpaper app" — no text, no facts, pure aesthetics, high volume — and the decision flips to **distilled diffusion**, because speed and cost dominate and there's no knowledge to get right. The paradigm choice falls straight out of *whether the prompt needs facts*.

## 10. Stress-testing the native-AR bet

Good engineering means poking at where a design breaks. Let's stress-test native-AR generation the way you'd interrogate it before betting a product on it.

**What happens when the image gets bigger?** Sequence length scales with the *number of image tokens*, which scales with resolution (modulo tokenizer compression). A 4× larger image is ~4× more tokens, which is ~4× more sequential steps *and* longer attention context (quadratic in the worst case, though modern attention and KV-caching soften this). This is why native-AR systems lean so hard on aggressive tokenizer compression and on next-scale / blockwise schemes — the raw raster-order cost at high resolution is punishing. If you need native 4K fast, native-AR is currently the wrong tool.

**What happens when the tokenizer is the bottleneck?** If the VQ codebook can't represent fine detail, *no* AR model on top can produce it — you'll see a texture ceiling, banding, or loss of small text legibility regardless of how good the transformer is. The fix is a better tokenizer (bigger/finer codebook, continuous or hybrid quantization), not a bigger transformer. Always profile the tokenizer's reconstruction quality *first*; it's the silent cap on the whole system.

**What happens when routing collapses?** If the MoE router degenerates and sends most tokens to a few experts, you lose effective capacity — the model behaves like a much smaller dense model and world knowledge suffers. This is a training-time failure (insufficient load-balancing loss, or a learning-rate/warmup mishap) and it shows up as worse-than-expected factual accuracy. It's why MoE training is finicky and why the load-balancing auxiliary loss is non-negotiable.

**What happens when sampling temperature is wrong?** Too high and the image tokens become incoherent (the AR analog of a diffusion model with too little guidance — noisy, structureless). Too low and you get repetitive, mode-collapsed outputs with little diversity across seeds. There's no universal sweet spot; it interacts with the prompt and the tokenizer, and it's one of the less-mature knobs in native-AR serving compared to the well-charted guidance-scale behavior of diffusion (see [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance)).

**What happens to editing?** Native multimodal models are *naturally* good at in-context editing — you put the source image's tokens in the prefix and ask for a change, and the model conditions on them, the same way it conditions on text. This is exactly the conversational-editing capability we covered in [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing), and it's a genuine *strength* of the AR paradigm: editing is "just" more conditioning in the sequence, no special inversion machinery (unlike DDIM-inversion-based diffusion editing, which is fiddly). If your product is editing-heavy, the native-AR architecture is working *for* you.

**What happens when the model's "knowledge" is wrong?** Inheriting an LLM's world knowledge inherits the LLM's *hallucination* too. If the language model confidently believes a false fact — a wrong flag, an outdated logo, a misremembered historical detail — the native-AR generator will draw the false fact just as confidently, because the same weights that hold the misconception produce the image. This is a *new* failure mode that diffusion models largely don't have: a diffusion model that's never seen Bhutan's flag produces a vague wrong flag and looks uncertain; a native-AR model that *misremembers* it produces a crisp, confident, wrong flag. The reasoning prefix can even rationalize the error in text before drawing it. So native-AR's knowledge is a double-edged sword: it gets the *common, well-attested* facts right far more often than diffusion, but when it's wrong it's wrong with conviction, and there's no easy "the model wasn't sure" signal in the pixels. For high-stakes factual content (medical, legal, official insignia), you still verify the output against ground truth — the model is a strong prior, not an oracle.

**What happens to cost at scale?** A multi-GPU 80B MoE doing minutes-per-image is expensive to serve at volume. If your workload is high-throughput, the per-image cost can be one to two orders of magnitude above a distilled diffusion model. This is the single biggest reason native-AR is not the default for every job — the economics only make sense when correctness is worth the premium.

The pattern across all of these: native-AR's weaknesses are *speed, cost, and tokenizer ceiling*; its strengths are *knowledge, instructions, text, and editing*. Know which axis your problem lives on and the choice is usually clear.

## 11. How to actually evaluate which paradigm wins for you

Benchmarks lie in characteristic ways, and the native-AR-vs-diffusion comparison is especially prone to it, so let me give you a *measurement protocol* rather than a leaderboard you'll over-trust. The deep dive on this is [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly), but here's the version specific to choosing a paradigm.

First, **FID is the wrong metric for this decision.** FID (Fréchet Inception Distance) measures the distance between the distribution of generated images and a reference set in an Inception feature space:

$$\text{FID} = \lVert \mu_g - \mu_r \rVert^2 + \operatorname{Tr}\!\big(\Sigma_g + \Sigma_r - 2(\Sigma_g \Sigma_r)^{1/2}\big),$$

where $(\mu_g, \Sigma_g)$ and $(\mu_r, \Sigma_r)$ are the mean and covariance of generated and reference features. It is a *distributional aesthetic/realism* metric — it rewards images that look like the reference set on average. It says almost nothing about whether a *specific* prompt's facts were rendered correctly. A model can have a beautiful FID and still spell "GRAND OPENING" as "GRAMD OPEMING." So if your decision hinges on knowledge and instruction following, FID will actively mislead you toward the prettier diffusion model.

The metrics that *do* discriminate the paradigms are the **compositional and text-rendering** ones:

- **GenEval / T2I-CompBench** score whether the right objects, counts, colors, and spatial relations appear — exactly the multi-clause-instruction axis where native-AR pulls ahead. Run your candidate models on these and the AR advantage on instruction following becomes visible in a way FID hides.
- **OCR-based text-rendering accuracy** — generate images with specified text, run OCR on the outputs, measure exact-match or character-error-rate. This is the cleanest way to quantify the text-rendering gap, and it's where native-AR's spelling advantage shows up as a hard number.
- **Human preference on knowledge-dependent prompts** — assemble a prompt set that *requires facts* (flags, period detail, brand specifics) and run a blind A/B. Aggregate metrics will not capture "the flag is correct"; only a targeted eval will.

Second, **control the confounds.** When you compare HunyuanImage 2.1 vs 3.0 (or any diffusion-vs-AR pair), fix the prompt set, fix the resolution, fix the random seed where the API allows, and warm up the model before timing (the first forward pass includes compilation/loading overhead that pollutes latency numbers). Report latency *and* a quality metric *and* the hardware — a single number is meaningless. The honest report is a small table: paradigm × (GenEval score, OCR accuracy, human-pref win-rate, s/image on a named GPU), built on *your* prompt distribution, not a generic benchmark.

#### Worked example: a prompt-set-driven A/B you can run

Suppose you have 200 prompts split into three buckets — 70 "aesthetic" (landscapes, portraits, no text/facts), 70 "instruction-heavy" (multi-object, counts, relations), 60 "text-and-knowledge" (signage, flags, brand mockups). Generate all 200 from a diffusion model (say HunyuanImage 2.1 or FLUX) and a native-AR model (HunyuanImage 3.0), then score: FID against a reference set for the aesthetic bucket, GenEval for the instruction bucket, OCR character-error-rate for the text bucket, and a blind human A/B for knowledge. The pattern you should expect — and that you can verify rather than trust — is diffusion competitive-or-ahead on the aesthetic bucket and fast, native-AR clearly ahead on the instruction and text/knowledge buckets but several-fold slower per image. That table *is* your decision: route aesthetic, high-volume traffic to diffusion and knowledge/text/instruction traffic to native-AR. The point is that *no single metric* makes this call; you have to measure on the axis your product actually lives on.

The broader trend the measurement reveals is **convergence, not conquest.** The frontier is moving toward models that do *both* — Transfusion-style hybrids that AR-generate text and diffuse images in one transformer, diffusion models bolting on LLM-grade text encoders, and native-AR models adopting next-scale/parallel decoding to claw back speed. HunyuanImage shipping a diffusion model and a native-AR model under one brand is a snapshot of that convergence in progress: the *same organization* hedging across the trade because neither corner of the trilemma is strictly dominant yet.

## 12. When to reach for native-AR (and when not to)

Here's the decisive recommendation, because every architecture is a cost and pretending otherwise wastes your time.

**Reach for native-AR (HunyuanImage 3.0, or GPT-Image / Nano Banana if closed is fine) when:**

- The prompt is **knowledge-dependent** — flags, logos, maps, brand-correct details, scientific/period accuracy, anything where the right answer is a *fact*.
- The image must contain **legible, correctly-spelled text** — posters, packaging, UI mockups, signage.
- The instruction is **multi-clause or compositional** — specific counts, relative positions, several constrained objects.
- You need the model to **infer unstated intent** — "what a runner eats before a race," "a tasteful corporate logo for a marine-biology startup."
- Your workload is **editing-heavy** and you want conversational, in-context edits without inversion machinery.
- **Latency is not the binding constraint** — occasional high-value generations, not high-throughput.

**Do NOT reach for native-AR when:**

- You need **high throughput or low latency** — thumbnails, real-time, millions of images. Use distilled diffusion (1–4 step LCM/Turbo/DMD — see [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation)). Native-AR's sequential decode will crush your unit economics.
- The job is **pure aesthetics with no facts** — decorative art, textures, stylized concept work. Diffusion's aesthetic tuning and speed win.
- You need to run **on a single consumer GPU** — an 80B MoE wants multi-GPU memory. HunyuanImage 2.1 or a quantized FLUX is the on-device-friendlier choice (see [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference)).
- You need **native very-high-resolution fast** — the sequential cost at 2K–4K is currently prohibitive for AR; diffusion with a high-compression VAE (like 2.1) is the better high-res-fast play.

The meta-rule: **native-AR is the tool when correctness matters more than speed; diffusion is the tool when speed and aesthetics matter more than facts.** HunyuanImage shipping both 2.1 and 3.0 is a clean expression of exactly this fork, which is why it's such a good case study for the whole AR-vs-diffusion question.

## 13. Key takeaways

- **HunyuanImage is two models, not one.** 2.1 is a fast diffusion MM-DiT with a high-compression VAE doing 2K; 3.0 is a ~80B MoE *native autoregressive* multimodal model. Don't conflate them.
- **Native-AR is the chain rule over a shared text+image vocabulary.** One decoder-only transformer factorizes $p(x) = \prod_t p(x_t \mid x_{<t})$ over a vocabulary that includes both BPE text ids and quantized image ids, trained with plain next-token cross-entropy.
- **World knowledge comes from sharing weights with language.** Because the same network that answers text questions also draws, the model carries genuine facts and reasoning into generation — and can write a *plan in text* that conditions the image tokens. A diffusion model conditioned on a frozen encoder structurally cannot do this.
- **The tokenizer sets the ceiling.** A VAE-plus-quantizer maps pixels to codebook ids; the model only ever predicts ids and detokenizes at the end. Compression cuts sequential steps; quantization caps reconstructable detail.
- **MoE buys capacity, not speed.** Top-$k$ routing gives an 80B-total model the per-token compute of a few-billion dense model — more knowledge capacity — but the *full* weights stay resident (multi-GPU) and image decoding is still $O(T)$ sequential.
- **The trade is fidelity-to-intent vs latency.** Native-AR wins on knowledge, instruction following, text rendering, and editing; it loses on speed, cost, and (often) the last bit of aesthetic polish. Diffusion is the mirror image.
- **Choose by whether the prompt needs facts.** Knowledge-/text-/instruction-heavy → native-AR. High-throughput, pure-aesthetic, on-device → distilled diffusion. The decision falls straight out of the workload.
- **HunyuanImage 3.0's significance is openness.** It's the strongest *open-weights* embodiment of the GPT-Image-style native-multimodal strategy, letting the community study an architecture the closed frontier otherwise hides.

## Further reading

- **HunyuanImage** — Tencent's official model cards and technical reports on the Hugging Face Hub (HunyuanImage 2.1 and HunyuanImage 3.0). Read these for the exact parameter counts, tokenizer details, MoE config, and loading code — the authoritative source over any secondhand summary.
- **Chameleon: Mixed-Modal Early-Fusion Foundation Models** — Chameleon Team, Meta, 2024. The single-transformer-over-interleaved-tokens design that the native-AR generation lineage builds on.
- **Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model** — Zhou et al., 2024. AR text + diffusion image within one transformer; the hybrid point on the spectrum.
- **Emu3: Next-Token Prediction is All You Need** — BAAI, 2024. A single next-token model across text, image, and video — the "pure AR" thesis at scale.
- **Visual Autoregressive Modeling (VAR)** — Tian et al., 2024. Next-scale prediction as the route to fast, high-quality autoregressive image generation; the most promising answer to native-AR's latency tax.
- **Janus / Janus-Pro** — DeepSeek, 2024–2025. Decoupling visual encoders for understanding vs generation within a unified multimodal model.
- Within this series: [Autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) (the token paradigm and tokenizers), [Autoregressive vs Diffusion: The 2026 Showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown) (the head-to-head), [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) (the diffusion side HunyuanImage 2.1 belongs to), and the capstone [Building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) (where this slots into a real pipeline). For the closed-frontier counterparts, see [GPT-Image and Nano Banana: the closed frontier](/blog/machine-learning/image-generation/gpt-image-and-nano-banana-the-closed-frontier), and for the full buyer's-guide synthesis, [The 2026 image model landscape](/blog/machine-learning/image-generation/the-2026-image-model-landscape).
