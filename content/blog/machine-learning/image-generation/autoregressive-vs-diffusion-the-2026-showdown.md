---
title: "Autoregressive vs Diffusion: The 2026 Showdown for Image Generation"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A head-to-head on the axes that decide image generation in 2026 — quality, prompt fidelity, sampling cost, and unification — and why the frontier is fusion, not a winner."
tags:
  [
    "image-generation",
    "diffusion-models",
    "autoregressive-models",
    "multimodal",
    "generative-ai",
    "deep-learning",
    "var",
    "mar",
    "transfusion",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-1.png"
---

Here is a question that, in 2022, had a boring answer: if you wanted to generate a 1024×1024 photograph of "a golden retriever wearing aviator sunglasses, sitting in a 1970s diner booth," what architecture would you reach for? The answer was diffusion. It was always diffusion. GANs had lost, autoregressive image models were a research curiosity that produced blurry 32×32 thumbnails, and every serious text-to-image system — Stable Diffusion, DALL·E 2, Imagen, Midjourney — was iterative denoising under the hood. The debate was settled before it started.

That answer is no longer boring. By 2024, an autoregressive model called VAR won the NeurIPS best-paper award by *beating* diffusion on ImageNet generation with a cleaner scaling law. A sibling, MAR, dropped the discrete tokenizer that everyone thought autoregressive image models required and instead modeled each token with a tiny diffusion head — a literal fusion of the two paradigms. Meanwhile OpenAI shipped GPT-Image-1, a natively multimodal autoregressive model whose prompt-following and world-knowledge made diffusion systems look semantically clumsy, and a wave of unified models (Transfusion, Chameleon, Janus-Pro, Emu3, Show-o) started doing image *understanding* and *generation* in a single transformer. The settled question reopened.

This post is the head-to-head. By the end you will be able to: explain the two likelihoods that define each paradigm (the autoregressive chain rule vs the diffusion denoising objective) and why they lead to such different engineering trade-offs; describe in detail *why* next-scale prediction (VAR) scales and *why* MAR can drop vector quantization; write a minimal next-token image sampler and a diffusion sampler side by side and see exactly where they diverge; and make a defensible call on which paradigm to reach for, given that the honest 2026 answer is "it depends, and the frontier is fusion." We will keep the series' spine in view the whole time — the generative trilemma (quality × diversity × sampling speed) and the diffusion stack — because the AR-vs-diffusion question is really a question about *where on that trilemma each paradigm sits*.

![A four-by-two comparison matrix showing diffusion and autoregressive paradigms scored on aesthetic quality, prompt fidelity, sampling cost, and unification with language, with each paradigm leading on different axes](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-1.png)

If you have read the foundations of this series — [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) and [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) — you already have the diffusion side of the picture. This post completes the map by giving the autoregressive side equal weight and then showing where the two are quietly merging. It is the natural sequel to [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models), which built the VAR/MAR/GPT-Image foundation, and a sibling to [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly), which supplies the benchmarks we will argue over.

## 1. Two likelihoods, two worlds

Everything downstream — the quality numbers, the latency, the controllability, the unification story — falls out of one choice: *how do you factorize the probability of an image?* This is not a detail. It is the fork in the road, and the rest of the showdown is the two branches playing out.

Both paradigms want to model a distribution $p(x)$ over images $x$, and both want to sample from it. They disagree on how to write $p(x)$ down in a way you can train.

**The autoregressive factorization.** Pick an ordering of the image into a sequence of tokens $x_1, x_2, \ldots, x_N$ — raster order (top-left to bottom-right), or some smarter order we will discuss. Then the chain rule of probability gives you, *exactly and with no approximation*:

$$
p(x) = \prod_{i=1}^{N} p(x_i \mid x_1, \ldots, x_{i-1}).
$$

You model each conditional with a transformer that, given the tokens so far, outputs a distribution over the next token. Training is maximum likelihood: minimize the negative log-likelihood, which for discrete tokens is just cross-entropy. This is *literally the same objective a language model uses* — next-token prediction. That single fact is why autoregressive image generation is the natural path to unification with language: it is the same machine.

**The diffusion factorization.** Diffusion refuses to pick an ordering. Instead it defines a forward process that gradually adds Gaussian noise to the image over $T$ steps, turning $x_0$ into pure noise $x_T \sim \mathcal{N}(0, I)$, and learns to reverse it. The marginal at step $t$ has a closed form,

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),
$$

where $\bar\alpha_t$ is the cumulative noise schedule. The model $\epsilon_\theta(x_t, t)$ is trained to predict the noise that was added, with the famous simplified loss

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{x_0, \epsilon, t}\big[\, \lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2 \,\big].
$$

Sampling runs the reverse process: start from noise and denoise step by step. Crucially, *every pixel (or latent) is updated at every step, in parallel*. There is no left-to-right. The "sequence" in diffusion is over noise levels, not over spatial position.

So the two worlds:

- **AR** factorizes over *space* (or scale, or a masking order). It commits tokens one chunk at a time, each conditioned on its own past. The number of forward passes scales with the number of tokens.
- **Diffusion** factorizes over *noise level*. It refines the whole canvas in parallel, many times. The number of forward passes scales with the number of denoising steps, independent of resolution-as-tokens.

![A before and after diagram contrasting diffusion which updates all latents in parallel each step against autoregressive which predicts tokens sequentially conditioned on past tokens](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-2.png)

That single structural difference — parallel-over-noise vs sequential-over-space — is the engine behind every trade-off we are about to measure. Hold onto it.

### Why the objectives feel so different

There is a deeper way to see the split, and it connects to the [score-based view](/blog/machine-learning/image-generation/diffusion-from-first-principles). The diffusion objective is, up to weighting, denoising score matching: $\epsilon_\theta$ is (a rescaling of) an estimate of the score $\nabla_{x_t} \log p_t(x_t)$, the gradient of the log-density of the noised data. Diffusion never evaluates a likelihood during training; it learns a *vector field* that points toward higher density, and sampling is following that field down from noise. It is an implicit, simulation-based way to model $p(x)$.

The autoregressive objective is the opposite temperament: it is *explicit* likelihood. Cross-entropy on next-token prediction is exactly the negative log-likelihood of the data under the model's factorization. You can read off $\log p(x)$ for any image. This gives AR models a clean, calibrated training signal and the same loss landscape that made transformer language models scale so predictably — which, as we will see with VAR, turns out to matter enormously.

Neither is "more correct." They are two valid factorizations of the same intractable object. The whole 2026 debate is about which factorization gives you the better *engineering* on the axes you care about — and, increasingly, whether you can have both in one model.

### Both are lower-bounding the same log-likelihood

It is worth making the connection rigorous, because it explains why fusion is even possible. Diffusion's loss is not arbitrary — it is a (reweighted) variational lower bound on $\log p(x)$, exactly like a VAE's ELBO. The diffusion ELBO decomposes into a sum of per-step KL terms,

$$
\log p(x_0) \ge \mathbb{E}_q\Big[ \log p(x_0 \mid x_1) - \sum_{t>1} D_\text{KL}\big(q(x_{t-1}\mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\big) - D_\text{KL}\big(q(x_T\mid x_0)\,\Vert\,p(x_T)\big)\Big],
$$

and after Ho et al.'s simplification each KL term collapses (up to a constant weight) to the squared-error noise-prediction loss $\lVert \epsilon - \epsilon_\theta\rVert^2$. So *both* objectives are bounds (AR's is exact, diffusion's is variational) on the same quantity, the data log-likelihood. That shared target is precisely why you can add them: $\mathcal{L}_\text{LM} + \lambda\,\mathcal{L}_\text{diffusion}$ is a sum of two estimators of the same kind of thing — negative log-likelihood — over two different token types. If diffusion's loss were some unrelated objective, the Transfusion sum would be meaningless. It is not; that is the deep reason the fusion in Section 6 works.

The difference that remains is the *factorization order* of the bound. AR's chain rule is sequential and exact but pays the sequential-decode tax at sampling time. Diffusion's bound is over noise levels and is parallel-in-space but loose (variational) and needs many steps to be tight. You are trading exactness-and-sequentiality against parallelism-and-iteration. Every number in the rest of this post is a measurement of that trade.

### A quick sanity check on "which sees the whole image"

One more intuition before we score axes, because it is the source of a lot of confusion. People say "diffusion sees the whole image, AR only sees the past." That is half-true. During *training*, both see the whole image — diffusion noises it, AR masks the future with a causal mask, but the ground-truth full image is available for the loss. The asymmetry is at *generation* time: a diffusion model always has a (noisy) estimate of every pixel and refines all of them, whereas a strictly causal AR model genuinely has not yet decided the future tokens. VAR and MAR soften exactly this: VAR has a full coarse image early (just low-resolution), and MAR's masked order lets it condition on a scattered set of already-placed tokens rather than only a causal prefix. So the resurgence is partly a campaign to give AR models more of diffusion's "I can see a draft of the whole image" property without giving up the explicit likelihood.

## 2. The axes that actually matter

Let me name the axes before we score them, because vague claims like "AR is better at prompts" or "diffusion is faster" hide more than they reveal. There are six axes that decide a real deployment, and the two paradigms genuinely split across them.

1. **Aesthetic quality at high resolution.** Can it make a clean, sharp, photorealistic 1024×1024 image with pleasing composition? This is measured imperfectly by FID and FID-DINOv2, and more honestly by human preference scores like HPSv2 and PickScore (see [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly)).
2. **Prompt adherence and world-knowledge.** Does it put exactly three apples when you ask for three, bind "red hat, blue scarf" to the right objects, and *know* what a 1970s diner booth looks like? Measured by GenEval, T2I-CompBench, and DPG-Bench.
3. **Sampling cost / latency.** Wall-clock seconds and FLOPs per image on a named GPU. This is where the trilemma's speed axis lives.
4. **Training stability and scaling behavior.** Does loss go down predictably as you add compute and data? Is there a clean scaling law you can extrapolate?
5. **Unification with language.** Can one model both *understand* images (answer questions, follow instructions) and *generate* them, sharing weights? This is the axis AR was built for.
6. **Controllability and editing.** How easily can you condition, inpaint, do instruction-based edits, preserve identity?

The opening figure scored these at a glance. The honest summary: diffusion leads on (1) and (3), AR leads on (2) and (5), and (4) and (6) are contested in interesting ways. Let me now defend each of those claims with mechanism and numbers, because a scorecard without reasons is just an opinion.

### Why diffusion leads aesthetic quality (for now)

Diffusion's parallel refinement is a gift for high-frequency detail. Because every latent is revisited at every step, the model can keep correcting fine texture — hair, skin pores, fabric weave — across the whole canvas right up to the last step. The iterative process is a built-in error-correction loop: a slightly-off latent at step 20 gets nudged back toward the data manifold at step 21. Combined with latent diffusion (denoising in a VAE's compressed space, see [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/diffusion-from-first-principles)) and the [MM-DiT recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe), this is why FLUX and SD3 still produce the cleanest open high-res aesthetics in 2026.

Autoregressive models historically struggled here for a concrete reason: **error accumulation**. If you commit token 50 with a small mistake, tokens 51 through 4096 all condition on that mistake and there is no going back. Raster-order generation also has a perceptual problem — the model decides the bottom-right corner having already locked the top-left, which is an unnatural way to compose an image. VAR and MAR both attack exactly this, which is the heart of the AR resurgence.

There is a second, quieter reason diffusion holds the aesthetic crown: its objective is a *perceptual* fit. Because diffusion trains in a VAE latent space tuned (often with a perceptual and adversarial loss in the VAE itself) to look good to humans, and because the iterative sampler spends most of its late steps on exactly the high-frequency band humans judge as "sharp" or "detailed," the whole stack is implicitly optimized for human aesthetic judgment, not just distribution matching. A classic discrete-token AR model is optimized for codebook cross-entropy, which is a *uniform* loss over tokens — it does not know that getting the eyes right matters more than the background blur. This is partly why AR's ImageNet FID wins (FID is a distribution metric, blind to where the detail goes) did not immediately translate into aesthetic wins (human raters care intensely about where the detail goes). MAR's continuous tokens and the native-multimodal models' scale are narrowing this, but the perceptual-objective advantage is a real, underrated part of why diffusion still looks best at high resolution.

### Why AR leads prompt fidelity and world-knowledge

This is the surprising one, and the mechanism is worth understanding. Autoregressive image models — especially the natively multimodal ones like GPT-Image — are trained on the *same token stream* as text. The model that wrote you a paragraph about 1970s diners is the model generating the diner. World-knowledge, compositional reasoning, and instruction-following transfer from the language side because there is no architectural seam between "thinking" and "drawing."

Diffusion models, by contrast, bolt a frozen text encoder (CLIP, T5, or an LLM) onto a denoiser and inject text via cross-attention. The denoiser never *reasons* in text; it consumes a fixed embedding. This is why diffusion notoriously fails at counting ("three apples" → four), at attribute binding, and at spatial relations — the famous compositionality gap that [GenEval and T2I-CompBench](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) were built to expose. An AR model that *is* a language model inherits the language model's grip on these.

It is worth being precise about *why* cross-attention conditioning leaks. When you encode "three red apples and two green pears" with a frozen text encoder, you get a sequence of token embeddings, and the denoiser cross-attends to them. But nothing in that mechanism *enforces* the count or the binding — the denoiser is free to attend to "red," "green," "apples," and "pears" and paint a plausible fruit bowl that satisfies the embedding's *gist* without satisfying its *logic*. Counting and binding are compositional, symbolic constraints, and a cross-attention soft-lookup over embeddings is a poor instrument for hard symbolic constraints. A native-multimodal AR model, generating image tokens in the same autoregressive stream where it just processed "three," carries the symbolic state forward the way it carries any other discourse state — it has, in effect, a working register that says "I have placed two apples, one to go." That is the mechanistic root of the prompt-fidelity gap, and it is why bolting a *bigger* text encoder onto diffusion helps only partially (SD3's T5 helped, but the gap to native-multimodal persists): the problem is not encoder capacity, it is that the denoiser and the language live in separate representational worlds joined by a lossy soft-lookup.

#### Worked example: the counting test

Take the prompt "exactly five red dice arranged in a cross pattern on a white table." On GenEval-style counting tasks, open diffusion models like SDXL land around 0.30–0.40 accuracy on the "counting" sub-score (approximate, from the GenEval paper and follow-ups), and even strong 2024 diffusion systems struggle past ~0.5 without special tricks. Natively multimodal AR systems and the strongest unified models report markedly higher counting and binding accuracy on these splits, often above 0.6–0.7 (approximate; exact numbers vary by model and prompt set). The mechanism is exactly what we said: the AR model is "counting" in the same latent space it counts words, while the diffusion model is trying to satisfy a counting constraint through a denoiser that has no symbolic handle on "five." Mark these numbers as approximate and benchmark-dependent — the *direction* is robust and reproducible; the precise decimals are not.

## 3. The AR resurgence, in detail

For years, autoregressive image generation meant one recipe: tokenize an image into a grid of discrete codes with a VQ-VAE or VQ-GAN, flatten in raster order, and run a GPT over the codes. Image-GPT did it on pixels; VQ-GAN + transformer did it on codes. It worked, it scaled poorly compared to diffusion, and it lost. The 2024 resurgence is the story of *changing the recipe* in two independent, brilliant ways.

### VAR: next-scale prediction

VAR (Visual Autoregressive modeling, Tian et al., NeurIPS 2024 best paper) made one change with outsized consequences: **stop generating in raster order; generate in scale order.** Instead of predicting tokens one spatial position at a time, VAR predicts an entire coarse token map first (a 1×1 map, then a 2×2, then 4×4, and so on up to the full resolution), with each scale conditioned on all coarser scales.

![A branching graph showing VAR encoding an image into K multi-scale token maps with a multi-scale VQ tokenizer then predicting each scale conditioned on coarser ones before decoding to a low FID result](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-3.png)

Why does this scale so well? Three reasons, all of which trace back to Section 1's likelihood story.

First, **the autoregressive step count drops from $O(N)$ tokens to $O(\log N)$ scales.** A 16×16 token map (256 tokens) raster-style is 256 sequential steps; as a coarse-to-fine pyramid it is roughly 10 scales. Within a scale, all tokens are predicted *in parallel*, so VAR recovers some of diffusion's parallelism while keeping the AR likelihood. This is the key structural insight: next-scale prediction is autoregressive *across* scales but parallel *within* a scale.

Second, **coarse-to-fine matches how images are actually structured.** Global composition (where the dog is, the lighting) lives at coarse scales; texture lives at fine scales. Predicting global structure first and refining it is a far more natural generative order than committing the top-left corner before you know what the image is. This drastically reduces the error-accumulation problem that hurt raster AR.

Third — and this is what won the award — **VAR exhibits a clean power-law scaling law** like a language model. As you scale parameters and compute, validation loss and FID improve along a smooth power law, with the kind of predictable extrapolation that diffusion's FID curves do not always offer. On ImageNet 256×256, VAR reports an FID around **1.8** (approximate, from the paper) with a 2-billion-parameter model, beating the comparable DiT diffusion baseline (DiT-XL/2 reported FID ≈ 2.27) *and* sampling faster. An AR model beat a diffusion model at diffusion's home benchmark. That is the shot that reopened the debate.

To see why the scaling law is so clean, it helps to write out what VAR is actually predicting. Let the image be encoded into $K$ token maps $r_1, r_2, \ldots, r_K$ of increasing resolution, where $r_k$ is an $h_k \times w_k$ grid of discrete codes. VAR factorizes:

$$
p(r_1, \ldots, r_K) = \prod_{k=1}^{K} p(r_k \mid r_1, \ldots, r_{k-1}),
$$

and within a scale it predicts *all* tokens of $r_k$ in parallel (with a block-causal attention mask: tokens within scale $k$ can attend to all coarser scales but the loss treats the scale as one parallel prediction). The crucial property is that the conditioning $r_1, \ldots, r_{k-1}$ is a *complete lower-resolution image*, not a partial raster prefix. That makes each conditional a well-posed, low-variance prediction problem (predict the residual detail at the next resolution given the full coarse image) — and well-posed prediction problems are exactly what transformers scale smoothly on. Raster AR's conditionals, by contrast, are high-variance ("given the top half of an image, predict the next pixel of the bottom-left") and that variance is part of why raster AR scaled badly.

Here is the shape of a VAR sampling loop, to make the scale-by-scale structure concrete:

```python
import torch

# Sketch of VAR-style next-scale sampling.
# transformer: predicts logits for ALL tokens of the next scale, in parallel.
# scales: list of (h_k, w_k) resolutions, coarse -> fine.
# decode: multi-scale VQ decoder mapping all token maps -> image.
@torch.no_grad()
def var_sample(transformer, scales, vocab_size, device="cuda"):
    token_maps = []                          # accumulates r_1 ... r_K
    for (h, w) in scales:                    # LOOP OVER SCALES, not tokens
        # condition on ALL coarser maps already produced
        logits = transformer(token_maps, target_hw=(h, w))   # shape [1, h*w, vocab]
        probs = logits.softmax(dim=-1)
        # sample the WHOLE map at once -> this is the parallelism win
        r_k = torch.multinomial(probs.view(-1, vocab_size), 1).view(1, h, w)
        token_maps.append(r_k)
    return decode(token_maps)                # ~K (≈10) sequential steps total
```

Count the sequential steps: `len(scales)` ≈ 10, not 256 or 4096. That single change — looping over ~10 scales instead of thousands of positions — is the whole latency story, and the "predict a full map in parallel" line is the whole quality story.

### MAR: masked autoregression with a diffusion loss, no VQ

MAR (Masked Autoregressive, Li et al., 2024) attacked a different assumption. Everyone "knew" autoregressive image models needed discrete tokens — you predict a categorical distribution over a codebook with a softmax. MAR asked: *why?* Vector quantization throws away information (every continuous latent is snapped to its nearest codebook entry, and the quantization error caps your maximum fidelity), and it is finicky to train (codebook collapse, low utilization). What if you kept the tokens *continuous*?

The problem: you can't put a softmax over a continuous space. The categorical cross-entropy that makes AR training clean assumes a finite vocabulary. MAR's answer is the fusion move that defines 2026:

**Model each continuous token with a small per-token diffusion process.** The transformer produces a conditioning vector for token $i$ (from the visible/already-generated tokens), and a tiny diffusion MLP — a "diffusion head" — generates the continuous token value conditioned on that vector. The loss for each token is the diffusion denoising loss, not cross-entropy:

$$
\mathcal{L}_\text{MAR} = \mathbb{E}_{i, \epsilon, t}\big[\, \lVert \epsilon - \epsilon_\theta(z_i^{(t)}, t \mid c_i)\rVert^2 \,\big],
$$

where $z_i$ is the continuous token, $c_i$ is the transformer's conditioning for position $i$, and $\epsilon_\theta$ is the small diffusion head. The big transformer still does autoregressive conditioning (in a masked, BERT-like generation order rather than strict raster), but the *per-token output distribution* is a diffusion model, not a categorical.

![A before and after diagram contrasting classic autoregressive models that quantize to a discrete codebook and lose fidelity against MAR which keeps continuous tokens and models them with a small diffusion head](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-6.png)

This is not "AR or diffusion." It is *AR for the sequence structure, diffusion for the token distribution.* MAR reports an ImageNet 256×256 FID around **1.55** (approximate, from the paper) — better than VAR and competitive with or beating strong diffusion baselines — with no vector quantization at all. It is the cleanest single piece of evidence that the paradigms are not rivals but ingredients.

The mechanics of the per-token diffusion head are worth seeing, because "a diffusion model that generates one token" sounds stranger than it is. The big transformer produces a conditioning vector $c_i$ for each masked position. To *sample* the token value at position $i$, you run a miniature reverse diffusion — a few denoising steps of a tiny MLP — conditioned on $c_i$. To *train*, you noise the ground-truth continuous token and ask the small MLP to predict the noise, with $c_i$ as conditioning. In code, the head is almost trivially small:

```python
import torch
import torch.nn as nn

class DiffusionHead(nn.Module):
    """Per-token diffusion: maps (noisy token, t, transformer cond) -> noise."""
    def __init__(self, token_dim, cond_dim, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim + cond_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, token_dim),               # predicts epsilon
        )

    def loss(self, z0, cond):                           # training: one token's loss
        t = torch.rand(z0.size(0), 1, device=z0.device)
        eps = torch.randn_like(z0)
        zt = (1 - t).sqrt() * z0 + t.sqrt() * eps       # simple noising
        pred = self.net(torch.cat([zt, cond, t], dim=-1))
        return ((pred - eps) ** 2).mean()               # diffusion MSE, NOT cross-entropy

    @torch.no_grad()
    def sample(self, cond, token_dim, steps=20):        # inference: a few mini-steps
        z = torch.randn(cond.size(0), token_dim, device=cond.device)
        for i in range(steps, 0, -1):
            t = torch.full((cond.size(0), 1), i / steps, device=cond.device)
            eps = self.net(torch.cat([z, cond, t], dim=-1))
            z = z - (1.0 / steps) * eps                 # crude Euler reverse step
        return z                                        # a CONTINUOUS token value
```

The whole MAR idea is in the contrast between `loss` (MSE on noise — diffusion) and what a classic AR head would do (cross-entropy over a codebook — categorical). Everything else — the big transformer, the masked generation order, the attention — is shared with ordinary autoregressive models. MAR is therefore the single most literal "fusion" model: not two networks glued together, but one network whose *output distribution per token* is itself a diffusion model. When people say the paradigms are converging, this is the convergence at its most concrete.

### The native-multimodal AR models

The third strand of the resurgence is not about beating FID on ImageNet — it is about *unification and instruction-following*. These are large models trained on interleaved text-and-image token streams:

- **GPT-Image-1 / GPT-Image-1.5** (OpenAI, 2025) — natively multimodal autoregressive image generation inside a multimodal model; the prompt-following, text rendering, and editing quality reset expectations for what "follows the prompt" means.
- **HunyuanImage-3.0** (Tencent, 2025) — a large open native-multimodal generator.
- **Lumina-mGPT** — a decoder-only multimodal model generating images as tokens.
- **Fluid** (Google, ~10.5B) — a scaling study showing that *continuous-token, random-order* autoregressive models (MAR-style) scale better than discrete-token raster ones, with the gap between AR and diffusion closing as you scale.

The common thread: these models lead on prompt fidelity and world-knowledge precisely because they are language models that also emit image tokens. They pay for it in sampling cost (many tokens, sequential) — which is exactly the trade-off the opening matrix encoded.

The Fluid result deserves a closer look because it is the clearest "controlled experiment" the field has on the AR-vs-diffusion question. Fluid varies two binary choices independently — discrete-VQ vs continuous tokens, and raster vs random generation order — and measures scaling. The finding: continuous tokens beat discrete (the VQ ceiling again), random order beats raster (less error accumulation, more diffusion-like "see a draft of the whole image"), and crucially the *gap to diffusion closes as you scale*. That last point matters for forecasting. If AR and diffusion converged to similar quality at small scale and diverged at large scale, you would bet on the diverging winner. Instead they *converge* with scale, which is exactly the signature you would expect if both are good estimators of the same log-likelihood (Section 1) and the differences are second-order effects of factorization that wash out with enough capacity and data. It is the empirical backbone of this post's thesis: at the frontier, the paradigm matters less than the scale, the data, and which objective you put on which token.

A practical note on what "native-multimodal" buys you that a diffusion model cannot easily replicate: *in-context generation*. Because the image is in the same token stream as text, you can give the model a few example image-caption pairs and have it generalize the style or task in-context, with no fine-tuning — the same few-shot behavior language models have. A diffusion model has no analog; you would need a LoRA or an IP-Adapter. This in-context capability is subtle but is one of the strongest long-run arguments for the AR/unified direction: it inherits *all* of the language-model paradigm's emergent capabilities, in-context learning included, for free.

## 4. A minimal sampler, side by side

Nothing makes the structural difference concrete like code. Here are two stripped-down samplers — a next-token AR image sampler and a diffusion sampler — written to highlight the one thing that differs: **the loop is over positions in one, over noise levels in the other.** These are illustrative PyTorch skeletons, not production code; the real models live behind `diffusers` and `transformers`, which we wire up afterward.

```python
import torch
import torch.nn.functional as F

# ---- Autoregressive image sampler (discrete tokens, raster order) ----
# model: a transformer that maps a sequence of token ids -> logits over vocab
# Generates an image as a flat sequence of N codebook tokens, one at a time.
@torch.no_grad()
def ar_sample(model, n_tokens: int, vocab_size: int, temperature: float = 1.0,
              top_k: int = 100, device: str = "cuda") -> torch.Tensor:
    tokens = torch.empty(1, 0, dtype=torch.long, device=device)
    for _ in range(n_tokens):                 # LOOP OVER SPATIAL POSITIONS
        logits = model(tokens)[:, -1, :]      # distribution for the NEXT token
        logits = logits / temperature
        if top_k is not None:                 # truncate the tail
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)   # sample one token
        tokens = torch.cat([tokens, nxt], dim=1)        # APPEND, then condition on it
    return tokens                             # N forward passes total
```

```python
# ---- Diffusion sampler (DDIM-style, latents updated in parallel) ----
# eps_model: predicts the noise eps given (x_t, t)
# alpha_bar: precomputed cumulative product of (1 - beta_t), length T
@torch.no_grad()
def diffusion_sample(eps_model, shape, alpha_bar, steps, device: str = "cuda"):
    x = torch.randn(shape, device=device)     # start from pure noise (full canvas)
    ts = torch.linspace(len(alpha_bar) - 1, 0, steps).long()
    for i in range(len(ts) - 1):              # LOOP OVER NOISE LEVELS
        t, t_next = ts[i], ts[i + 1]
        ab_t, ab_next = alpha_bar[t], alpha_bar[t_next]
        eps = eps_model(x, t)                 # predict noise for EVERY latent at once
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()      # predicted clean image
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps  # step to next noise level
    return x                                  # `steps` forward passes total, any resolution
```

Read the two loops next to each other and the entire debate is visible in the control flow:

- The AR loop runs `n_tokens` times. For a 16×16 latent that is 256 forward passes; for a 64×64 token grid it is 4096. Each pass conditions on the growing sequence (in practice you cache keys/values so each step is cheap, but you still need $N$ of them, and they are *sequential* — step $i+1$ cannot start until step $i$ commits).
- The diffusion loop runs `steps` times — 20, 30, maybe 4 after distillation — *regardless of resolution*. Each pass touches every latent in parallel. The cost is set by the step count, not the token count.

This is why diffusion's killer move is *step reduction* (DDIM, DPM-Solver, consistency models, DMD — see [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/diffusion-from-first-principles)) and AR's killer move is *reducing the token count or parallelizing token prediction* (exactly what VAR's next-scale and MAR's masked-parallel decoding do).

### Loading the real thing

In practice you do not write these loops; you load a pipeline. Diffusion in 🤗 `diffusers` is one call:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()             # fit on a 24GB GPU
image = pipe(
    "a golden retriever wearing aviator sunglasses in a 1970s diner booth",
    num_inference_steps=28,                  # the diffusion step budget
    guidance_scale=3.5,                      # classifier-free guidance
    height=1024, width=1024,
).images[0]
```

An autoregressive or unified model is loaded through `transformers` and *generated* like a language model — same `generate()` machinery, because image tokens are just more tokens in the stream:

```python
from transformers import AutoModelForCausalLM, AutoProcessor

# Conceptual: a native-multimodal AR model that emits image tokens.
# The image is produced by the SAME autoregressive generate() loop as text.
model = AutoModelForCausalLM.from_pretrained(
    "some-org/unified-mm-model", torch_dtype="bfloat16", device_map="auto"
)
processor = AutoProcessor.from_pretrained("some-org/unified-mm-model")

inputs = processor(
    text="Generate an image: three red dice on a white table.",
    return_tensors="pt",
).to(model.device)

# generate() emits a sequence containing image tokens; a detokenizer/VAE
# decoder then turns those tokens back into pixels.
out = model.generate(**inputs, max_new_tokens=1024, do_sample=True, top_p=0.95)
image = processor.decode_image(out)          # model-specific image detokenization
```

Look at the symmetry: the diffusion model has a `num_inference_steps` knob (the noise-level loop) and the AR model has a `max_new_tokens` knob (the position loop). Those two knobs are the two loops from the samplers above, surfaced as APIs. The structural difference from Section 1 is right there in the function signatures.

## 5. The token-budget and latency comparison

Let me make the cost difference quantitative, because "AR is slower" is too lazy a claim — it depends entirely on the token budget and on parallelism within a step.

Define the cost of generating one image as forward passes times per-pass FLOPs. For a transformer of $d$ model dimension processing a sequence of length $L$, a forward pass is roughly $O(L^2 d)$ for attention plus $O(L d^2)$ for the MLPs. Now compare:

- **Diffusion (latent, DiT-style).** The sequence length is the number of latent patches, e.g. a 64×64 latent patchified at 2×2 gives $L = 1024$ patches. You run the full transformer $S$ times where $S$ is the step count. Total ≈ $S \cdot O(L^2 d)$. With distillation $S$ can be 4 or even 1. So a distilled diffusion model is ~4 full forward passes over 1024 patches.
- **Classic raster AR.** You run the transformer once per token, $L$ times, with KV-caching so pass $i$ attends over $i$ cached tokens: total ≈ $\sum_{i=1}^{L} O(i \cdot d) \approx O(L^2 d)$ for attention — comparable *total* FLOPs to one diffusion pass over the sequence, *but spread across $L$ sequential steps*. The latency killer is not FLOPs; it is the $L$ sequential dependencies (256, 1024, 4096…) that cannot be parallelized and that each incur kernel-launch and memory overhead.
- **VAR (next-scale).** Sequential steps drop from $L$ tokens to ~$\log L$ scales (≈10), with all tokens in a scale predicted in parallel. This is why VAR is *faster* than both raster AR and comparable diffusion at matched quality — it converts most of the sequential budget into parallel within-scale prediction.
- **MAR (masked parallel).** Generates tokens in a few dozen masked-prediction rounds (e.g. 64 steps), each round filling in many tokens at once, with a tiny extra diffusion-head cost per token. Far fewer sequential steps than raster AR.

#### Worked example: latency on an A100 80GB

Take a 1024×1024 target. A distilled FLUX-class diffusion model at 4 steps generates in roughly **0.3–0.6 s** on an A100 80GB (approximate; depends on VAE, attention kernels, batch size). An undistilled 28-step FLUX is closer to **3–6 s**. A *raster* autoregressive model emitting ~4096 image tokens pays ~4096 sequential decoder steps — even at a fast 50–100 tokens/s for a large model that is tens of seconds per image, which is why nobody ships raster AR at high resolution. VAR's next-scale ordering and MAR's masked parallel decoding cut that to roughly **1–3 s** at comparable resolution (approximate), and large native-multimodal models trade latency for their world-knowledge advantage. The takeaway: *AR's latency problem is a token-count problem, and the resurgence is largely a campaign to shrink the sequential token budget.* Diffusion's latency advantage is real today but is mostly an artifact of aggressive step distillation, a lever AR variants are starting to pull too.

### Training stability and the scaling-law axis

Latency is the axis people quote, but training behavior is the one that quietly decides which paradigm a frontier lab bets on, and it is where AR has a genuine, underappreciated edge. Recall from Section 1 that AR trains with cross-entropy on an explicit likelihood. That objective is convex in the logits, well-conditioned, and — critically — has the *same* loss landscape that the entire language-modeling field spent five years learning to scale. Every trick that stabilized large language model training (careful initialization, learning-rate warmup, $\mu$P-style width scaling, AdamW with the now-standard hyperparameters, z-loss for logit stability) transfers directly to autoregressive image models, because the objective is identical. You are not discovering how to scale a new loss; you are reusing a solved one.

Diffusion training, by contrast, has its own folklore of stabilizers that took years to mature: the choice of noise schedule (linear vs cosine vs the zero-terminal-SNR fix), the loss weighting (min-SNR weighting to stop high-noise timesteps from dominating the gradient), the parameterization ($\epsilon$-pred vs v-pred vs x0-pred), and EMA of the weights to get a usable sampler. None of these are hard once you know them — see [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/diffusion-from-first-principles) — but they are diffusion-specific, and getting them wrong gives you gray mush or color shifts rather than a clean error message. The practical consequence: a team that already trains large language models can stand up a credible autoregressive image model faster than they can master the diffusion-training folklore, because the AR objective is the one they already operate at scale.

This is the real subtext of VAR's "clean power-law" claim. Diffusion *does* scale — DiT showed a clear compute-to-FID trend — but the AR scaling law is the *same shape* as the language scaling law, which means a lab can borrow its compute-allocation intuitions (how to trade parameters against tokens, when to expect diminishing returns) wholesale. That transferability, more than any single FID number, is why native-multimodal labs leaned AR: the scaling is predictable with tools they already own.

### The diversity and mode-coverage axis

The series' trilemma has a third corner we should not skip: mode coverage. Both paradigms are, in principle, good at diversity — both are likelihood-based (AR exactly, diffusion variationally), and likelihood objectives are mode-covering by construction (they pay an infinite penalty for assigning zero probability to real data, which is what prevents the mode collapse that killed [GANs](/blog/machine-learning/image-generation/diffusion-from-first-principles)). This is a real shared win over GANs and a reason both paradigms beat them. The subtlety is in *how sampling temperature interacts with coverage*. AR exposes a clean temperature and top-p/top-k knob: lower temperature trades diversity for fidelity, exactly as in language. Diffusion's analogous knob is classifier-free guidance, and high guidance is well known to collapse diversity and over-saturate (CFG above ~7–8 on many models). So both can trade coverage for fidelity, but the *mechanism and failure mode* differ: AR over-sharpens its categorical (or shrinks its diffusion-head variance), diffusion over-extrapolates the guidance vector. When you measure precision/recall (the diversity-aware metric pair from [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly)), you are largely measuring how each paradigm's fidelity knob was set, not a fixed property of the paradigm.

## 6. Unified multimodal architectures: doing both

The most interesting models of 2024–2026 are not "an AR model" or "a diffusion model." They are *unified multimodal models* that do image understanding and image generation in one network — and they are where the convergence becomes architectural rather than incidental. There are three broad designs, and it is worth seeing them as a small taxonomy because the design choice determines everything about how the model behaves.

![A taxonomy tree of unified multimodal models splitting into early-fusion token models, decoupled-encoder models, and mixed-objective models with example systems under each branch](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-7.png)

### Early-fusion token models: Chameleon, Emu3

The purest unification: tokenize *everything* — text and images — into one discrete vocabulary, interleave them in a single sequence, and train one transformer with one next-token objective over the mixed stream. **Chameleon** (Meta, 2024) does exactly this: images become VQ tokens, text becomes BPE tokens, and the model autoregresses over the union. **Emu3** (BAAI, 2024) pushes the same idea to a single next-token objective for text, image, *and* video. The beauty is conceptual simplicity — there is only one loss, one architecture, one inference loop, and the model can freely interleave reasoning and generation. The cost is image quality: discrete VQ tokens cap fidelity (the same VQ problem MAR diagnosed), and these models historically trailed diffusion on raw aesthetics.

There is also a subtle training-stability wrinkle that the Chameleon paper itself flagged and that is instructive: mixing two modalities in one softmax can destabilize training, because the logit norms of image tokens and text tokens drift apart and the shared final layer-norm and output projection get pulled in two directions. Chameleon's fix was query-key normalization and careful re-ordering of norms — a reminder that "just put everything in one vocabulary" is conceptually clean but operationally has its own folklore, mirroring (not escaping) diffusion's stabilizer folklore. Unification does not make the engineering free; it relocates it. The payoff is the interleaving: a Chameleon-style model can read an image, reason about it in text, and emit a new image in one continuous generation, which is exactly the behavior that makes native-multimodal models feel qualitatively different from a diffusion model with a text encoder bolted on.

### Decoupled-encoder models: Janus, Janus-Pro

The insight here is that *understanding* and *generating* images want different visual representations. Understanding wants a high-level semantic encoder (a SigLIP/CLIP-style encoder that captures "what is in this image"); generation wants a low-level encoder that preserves pixel detail. **Janus** and **Janus-Pro** (DeepSeek, 2024–2025) *decouple* the vision pathways: one encoder for understanding, a separate one for generation, both feeding a shared autoregressive transformer. Janus-Pro-7B reports strong results on both multimodal understanding benchmarks and text-to-image generation benchmarks (GenEval, DPG-Bench), showing that decoupling resolves the tension that hurt early-fusion models. It is still autoregressive at the core; the cleverness is in *not forcing one visual tokenizer to do two jobs.*

Why does the single-encoder approach create tension in the first place? A semantic encoder trained to be good at "what is in this image" deliberately *throws away* low-level detail — it wants "dog" to map to the same region of latent space regardless of fur texture, because that invariance is what makes understanding robust. But a generator needs exactly that thrown-away detail to reconstruct pixels. Force one encoder to serve both and you get a representation that is too lossy to generate well or too detail-bound to understand robustly; you are optimizing one tokenizer against two contradictory pressures. Janus-Pro's decoupling is the principled fix: let the understanding path be lossy-and-semantic and the generation path be detailed-and-reconstructive, and let the shared transformer do the reasoning that bridges them. It is a nice illustration that "unified" does not have to mean "one component for everything" — it can mean "one reasoning core, specialized perception and rendering paths," which is arguably closer to how the whole field is settling.

### Mixed-objective models: Transfusion, Show-o

This is the most direct fusion, and the most relevant to our thesis. **Transfusion** (Meta, 2024) trains *one transformer* on *two objectives at once*: text tokens get the next-token cross-entropy loss, and image patches get the *diffusion* loss. In a single sequence, the model attends causally over text (AR) and bidirectionally over the image patches it is denoising (diffusion), with every weight shared.

![A branching and merging graph of a Transfusion backbone where text tokens flow to a next-token head and image patches flow to a diffusion head sharing one transformer to produce a joint understand-and-generate model](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-4.png)

The training signal is literally the sum of two losses:

$$
\mathcal{L}_\text{Transfusion} = \mathcal{L}_\text{LM}(\text{text tokens}) + \lambda\,\mathcal{L}_\text{diffusion}(\text{image patches}).
$$

The text half is the chain-rule likelihood from Section 1; the image half is the denoising-score-matching objective. One backbone, both worlds. The one architectural detail that makes this work is the *attention pattern*: text tokens use a causal mask (token $i$ sees tokens $< i$, as in any language model), but the image patches inside an image use *bidirectional* attention among themselves (every patch sees every other patch), because that is what diffusion wants — a denoiser needs the whole noisy canvas to estimate the score. Transfusion runs both mask types in one sequence: causal between text and image blocks, bidirectional within each image block. That single hybrid attention mask is the mechanical heart of the fusion — it lets one set of weights serve the sequential text objective and the parallel image objective without contradiction. Transfusion reports image-generation quality competitive with comparable diffusion-only models *and* language modeling competitive with text-only models at matched scale — strong evidence that you do not have to choose. **Show-o** (2024) does a related thing with a unified transformer that mixes autoregressive text modeling and (discrete) masked-token image generation in one model. **MAR** (Section 3) is the per-token version of the same fusion.

If you want one sentence for the whole section: *the frontier stopped asking "AR or diffusion?" and started asking "which objective for which token, in one shared transformer?"*

## 7. Where they are converging

Step back and the convergence is unmistakable. It is happening on three fronts simultaneously, and each one erodes a supposed "fundamental" difference between the paradigms.

**The backbone converged first.** It is transformers all the way down. Diffusion abandoned the U-Net for the [DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit) and then [MM-DiT](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe); autoregressive image models were always transformers. The architectural moat between "diffusion architecture" and "AR architecture" is gone — both are transformers with attention over a token sequence. What differs is the *training objective and the generation order*, not the network.

**The loss is converging.** MAR puts a diffusion loss inside an autoregressive model. Transfusion sums an AR loss and a diffusion loss over one transformer. The continuous-token, flow/diffusion-on-tokens idea is showing up everywhere. The clean dichotomy "AR uses cross-entropy, diffusion uses MSE-on-noise" no longer holds — modern models mix them per token or per modality.

**The generation order is converging.** VAR's coarse-to-fine next-scale prediction is *structurally* close to diffusion's coarse-to-fine denoising (diffusion also resolves global structure early in sampling and detail late). MAR's masked, few-round parallel decoding looks more like a handful of diffusion steps than like raster AR. From the other side, distilled few-step diffusion is converging toward AR's "a small number of forward passes" regime. The two are meeting in the middle: a modest number of forward passes, each refining a partially-formed image, on a transformer, with a likelihood-flavored loss.

![A timeline of the autoregressive versus diffusion arc from DDPM in 2020 through latent diffusion and DiT scaling diffusion, the VAR and MAR resurgence, Transfusion and Chameleon fusion, GPT-Image native multimodal, to the 2026 fusion frontier](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-5.png)

The timeline tells the story: diffusion's sweep (2020–2023), the AR resurgence and fusion models (2024), native-multimodal scale (2025), and a 2026 frontier where the question is no longer "which paradigm" but "which mixture." If you are building today, you are not picking a team; you are picking a point in a continuous design space that spans pure diffusion to pure AR with everything in between.

It is worth naming the *one* difference that has not converged, because it is the most likely axis of future divergence: the **fundamental sampling unit**. Diffusion's atom is a denoising step over the whole canvas; AR's atom is a token (or a scale, or a masked round). Distillation pushes diffusion toward fewer, bigger steps; parallel decoding pushes AR toward fewer, bigger rounds. They are converging on "a handful of forward passes," but the *shape* of each pass still differs — diffusion's pass is a full-canvas score estimate, AR's is a conditional-distribution prediction. Whether these collapse into one unit (some future model where "denoise the whole canvas a bit" and "predict the next scale" are literally the same operation) is, to me, the open research question that decides whether 2028's models are recognizably "diffusion" or "AR" at all, or simply *generative transformers* with a tunable objective per token. The honest read of the trend line is the latter.

## 8. The 2026 scorecard: real numbers

Now the results section the series demands — named models, paradigms, benchmark standing, sampling cost, and whether they unify with language. As always with generative metrics, treat FID as a coarse proxy (it correlates imperfectly with human preference; see [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly)) and treat every decimal here as approximate and benchmark-conditioned. The *rankings and directions* are robust; the exact figures shift with sample size, reference set, and prompt distribution.

![A six-by-four scorecard matrix mapping FLUX SD3, VAR, MAR, Chameleon, Transfusion, and GPT-Image to paradigm, benchmark standing, sampling cost, and whether each unifies with language](/imgs/blogs/autoregressive-vs-diffusion-the-2026-showdown-8.png)

Here is the same data with sources and caveats, as a table you can act on:

| Model | Paradigm | Headline number | Sampling cost | Unified? |
|---|---|---|---|---|
| SDXL | Diffusion (latent U-Net) | COCO FID ≈ 8 (text-to-image, approx) | 25–50 steps; ~4 s/img A100 | No |
| SD3 / FLUX | Diffusion (MM-DiT, flow) | Top open aesthetics; HPSv2 leaders | 20–28 steps; ~3–6 s, or ~0.5 s distilled | No |
| VAR-d30 (~2B) | AR (next-scale) | ImageNet 256 FID ≈ 1.8 | ~10 scales; fast | No |
| MAR (~0.9B) | AR + per-token diffusion (no VQ) | ImageNet 256 FID ≈ 1.55 | ~64 masked rounds | Partial |
| DiT-XL/2 (~675M) | Diffusion (transformer) | ImageNet 256 FID ≈ 2.27 | 250 steps (paper) | No |
| Chameleon | AR (early-fusion tokens) | Strong on mixed text-image tasks | Many tokens; sequential | Yes |
| Transfusion | AR + diffusion (one transformer) | Competitive image + text at scale | Mixed; diffusion-step + token | Yes |
| GPT-Image-1 | AR (native multimodal) | Top instruction-following & editing | Many tokens; slower | Yes |

A few honest readings of this table:

- **On ImageNet 256 FID specifically, AR variants now lead.** MAR ≈ 1.55 and VAR ≈ 1.8 beat DiT-XL/2's ≈ 2.27. This is the strongest *quantitative* evidence for the resurgence — and it is on diffusion's classic home turf.
- **On open high-res aesthetic text-to-image, diffusion still leads.** FLUX/SD3 produce the cleanest 1024×1024 open-weights results and dominate human-preference leaderboards. ImageNet FID and "does this 1024px portrait look gorgeous" are different questions; AR's FID win does not yet translate to an open aesthetic win at high resolution.
- **On unification, AR and fusion models lead by construction.** Diffusion needs a bolt-on (Transfusion is the bolt-on done right); pure-AR and mixed-objective models do understanding and generation natively.
- **On sampling cost, distilled diffusion is the speed champion.** A 1-to-4-step distilled diffusion model is the fastest path to a high-res image in 2026; AR's resurgence narrows but does not erase this for high-resolution aesthetic generation.

### The controllability and editing axis, in practice

The sixth axis — controllability and editing — is where the paradigms split in a way that does not show up in any FID table but matters enormously to what you can actually ship. Diffusion's iterative, latent-space process turns out to be remarkably hospitable to *injected control*. Because the model revisits the whole latent at every step, you can intervene mid-process: ControlNet adds a parallel branch that nudges every denoising step toward a depth map or pose skeleton; SDEdit starts the denoising from a partially-noised version of a reference image; inpainting masks part of the latent and only denoises the rest; IP-Adapter injects a reference image's embedding through decoupled cross-attention. All of these exploit the same property — *there is a continuous latent you can reach into at every one of many steps.* That is why diffusion accumulated a vast, mature control ecosystem (the subject of much of Track D in this series).

Autoregressive editing works differently and, in the native-multimodal case, more naturally for *instruction* edits. You do not reach into a latent; you *keep talking.* "Make the sky purple" is just more tokens appended to a conversation that already contains the image's tokens, and the model regenerates conditioned on the instruction plus the original. This is why the 2025 conversational-editing wave — GPT-Image, FLUX-Kontext, Nano Banana, covered in [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) — felt like a step change: instruction editing is the *native* mode of a model that treats images as tokens in a dialog. The trade-off is precision. Diffusion's latent-space control (a depth map, an exact mask, a precise reference) is more surgical than a natural-language instruction; if you need pixel-exact structural control, the diffusion + ControlNet path is still more reliable, while if you need "edit this the way a person would describe it," the native-multimodal AR path is more natural. A fusion model, again, wants to offer both — latent-reachable image patches *and* a conversational instruction interface — which is one more reason fusion is the interesting frontier.

#### Worked example: cost per image across paradigms

Put rough dollar figures on it, because cost decides production choices. On a cloud A100 80GB at roughly \$2/hr (approximate, spot-dependent), a distilled 4-step diffusion model at ~0.5 s/image generates about 7,200 images/hr, or roughly \$0.0003 per image. An undistilled 28-step model at ~4 s/image is about 900 images/hr, ~\$0.002 per image — still cheap. A large native-multimodal AR model emitting thousands of tokens at, say, 5 s/image is ~720 images/hr, ~\$0.003 per image, *plus* the larger model often needs more or bigger GPUs, pushing the effective cost higher. So the cost ranking tracks the latency ranking: distilled diffusion is cheapest per image, AR native-multimodal is the most expensive, and you are paying that premium specifically for world-knowledge and unification. If your product generates millions of routine images, the diffusion economics are decisive; if it generates fewer images that each need to *exactly* follow a complex instruction, the AR premium is easily worth it. The numbers are approximate and move with hardware and batching, but the *ordering* is the durable takeaway.

## 9. The honest verdict, and when to reach for each

Time for the decisive recommendation this series insists on. No fence-sitting — here is what I would actually pick, and when I would *not* pick it.

**Reach for diffusion when** you want the best open high-resolution aesthetic quality with the lowest latency, and you do not need tight instruction-following or unification. Text-to-image for art, product imagery, and stylized generation; anything where a distilled few-step model's speed matters; anything where you want the mature ecosystem of [ControlNet](/blog/machine-learning/image-generation/diffusion-from-first-principles), LoRA, IP-Adapter, and inpainting tooling. In 2026, if someone asks for a 1024×1024 image generator to ship next week, the default is still FLUX/SD3-class diffusion. **Do not reach for diffusion when** your core problem is compositional reasoning, counting, precise instruction-following, or interleaving generation with chat — that is where the bolt-on text encoder leaks.

**Reach for autoregressive / native-multimodal when** prompt fidelity, world-knowledge, text rendering, and *unification with a language model* are the point. Conversational image editing, agents that reason and then draw, products where the same model answers questions about an image and generates a new one, anything that benefits from the language model's grip on "exactly five" and "the red one on the left." **Do not reach for raster AR** at high resolution if latency matters — the sequential token budget will bury you; reach for a next-scale (VAR-style) or masked (MAR-style) variant instead, or accept the latency for the semantic win.

**Reach for fusion (MAR / Transfusion-style) when** you are building something new and want both: a single transformer that understands and generates, with continuous tokens to dodge VQ's fidelity ceiling and a diffusion loss to get clean per-token distributions. This is where I would place a research bet for 2026–2027. **Do not reach for fusion** if you need a battle-tested, well-tooled production system *today* — the fusion models are newer, the tooling is thinner, and the operational knowledge is less mature than the diffusion stack.

### Stress-testing the verdict

Let me poke at my own recommendation, the way the series demands.

- *What if you need both top aesthetics AND counting?* Today you compose: a strong diffusion generator with a language-model planner that decomposes the prompt (lay out "five dice" as explicit regions) or a fusion model accepting slightly lower peak aesthetics for much better adherence. There is no single model that dominates both axes yet — that gap is exactly the frontier.
- *What happens to AR at 4K?* The token budget explodes. Even next-scale prediction has many fine-scale tokens at 4K, and the cost grows fast. Diffusion's resolution-independence (cost set by steps, not tokens) is a real structural advantage at very high resolution. Watch for next-scale + latent-space tricks to close this.
- *What if the FID win is a benchmark artifact?* It partly is — ImageNet 256 class-conditional FID is a narrow, saturated benchmark, and small FID differences there are not human-meaningful (a point [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) hammers). The AR FID win is real and important as a *scaling and architecture* signal, but do not over-read 1.55 vs 2.27 as "AR images look better to people." On open high-res human preference, diffusion still leads.
- *What if distillation comes to AR?* It is starting to (parallel decoding, speculative decoding for image tokens, fewer-scale variants). If AR's sequential-token tax falls the way diffusion's step tax fell to distillation, the speed gap narrows further and the case for AR/fusion strengthens.
- *What if your "AR is better at prompts" claim is just GPT-Image being a huge, heavily-trained model?* Fair challenge — and partly true. The native-multimodal models that lead on prompt fidelity are also enormous and trained on vastly more multimodal data than open diffusion models, so some of the gap is scale and data, not paradigm. The cleaner controlled evidence is benchmarks like GenEval where matched-scale unified models still tend to edge matched-scale diffusion on counting and binding, plus the *mechanistic* argument (Section 2) that cross-attention soft-lookup is a poor instrument for symbolic constraints. So the direction is paradigmatic, but do not attribute the *entire* observed gap to the paradigm — confounded by scale, it is real but smaller than the flashiest demos suggest.
- *What if you are wrong and one paradigm just wins outright by 2028?* Possible. If a distillation breakthrough makes AR/fusion both fastest *and* best-at-everything, the "fusion frontier" framing collapses into "AR won." The hedge is the build advice: keep the generator behind a stable interface so the bet is cheap to revise. I would put more probability on continued fusion than on a clean AR or diffusion victory, but I hold that view loosely — this is a field that reopened a settled question once already.

## 10. Case studies: three results worth knowing

To ground all of this, three specific, citable results from the literature that capture the showdown.

**VAR beats DiT on ImageNet (Tian et al., NeurIPS 2024).** A ~2B-parameter next-scale autoregressive model reaches ImageNet 256×256 FID ≈ 1.8 (approximate) with a clean power-law scaling law in parameters and compute, surpassing the DiT-XL/2 diffusion baseline (FID ≈ 2.27) while sampling faster thanks to $O(\log N)$ scales instead of $O(N)$ tokens. The significance is less the FID and more the *scaling law*: AR image models scale like language models, which is a property diffusion's FID curves do not always cleanly exhibit. This is the result that won the best-paper award and reopened the debate.

**MAR removes vector quantization (Li et al., 2024).** By modeling continuous tokens with a small per-token diffusion head instead of a categorical softmax over a codebook, MAR reaches ImageNet 256×256 FID ≈ 1.55 (approximate) with *no VQ*. This is the cleanest demonstration that the AR/diffusion split is a false binary — the per-token output distribution is literally a diffusion model living inside an autoregressive transformer. It also fixes a real engineering pain (codebook collapse, low utilization) that plagued discrete-token image models.

**Transfusion unifies the objectives (Meta, 2024).** One transformer, two losses (next-token for text, diffusion for image patches), every weight shared. Transfusion reports image-generation quality competitive with comparable diffusion-only models and language modeling competitive with text-only models at matched scale — quantitative evidence that the fusion does not force a quality tax on either modality. Pair this with Chameleon (early-fusion discrete tokens) and Janus-Pro (decoupled encoders) and you have the three-way taxonomy of how to build a model that both *sees* and *draws*.

#### Worked example: estimating a fusion model's training cost split

Suppose you train a Transfusion-style 7B model on a corpus that is 80% text tokens and 20% image patches, with the diffusion loss weight $\lambda = 5$ to balance gradient magnitudes (image patches are noisier targets). The effective gradient contribution from images is then roughly $0.2 \times 5 = 1.0$ relative to text's $0.8 \times 1 = 0.8$ — so despite being only a fifth of the tokens, the image objective drives slightly *more* of the update, which is what you want if generation quality is the goal. If you flipped $\lambda$ to 1, images would contribute only $0.2$ vs text's $0.8$ and the model would barely learn to generate. This single hyperparameter — the loss-weight balance between the AR and diffusion objectives — is one of the most important knobs in a fusion model, and it is a knob that *only exists because the two objectives now live in one model.* That is the convergence made concrete: a hyperparameter that would have been meaningless in 2022.

## 11. What this means for the rest of the stack

The showdown is not just academic taxonomy; it changes how you build. If you are assembling a real system — the subject of the capstone, [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) — the paradigm choice ripples through every layer.

**Conditioning and editing.** Native-multimodal AR models made conversational, instruction-based editing natural (you just keep talking in the same token stream), which is why the 2025 editing wave — covered in [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) — leaned heavily on GPT-Image-style models alongside FLUX-Kontext. Diffusion editing needs more machinery (inversion, prompt-to-prompt, IP-Adapter); AR editing is "more conversation."

**Serving.** A diffusion server's hot path is the denoising loop and the VAE decode; you optimize step count, attention kernels, and VAE tiling. An AR server's hot path is the autoregressive decode loop; you optimize KV-cache management, parallel/speculative decoding, and the image detokenizer. They are different serving problems with different bottlenecks, and a fusion model is *both* at once — which is why fusion models are operationally harder to serve today. The practical implication for capacity planning is sharper than it looks. A diffusion server's throughput is dominated by GPU compute and is *batch-friendly* — you can pack many images into a batch and amortize the fixed step count, so GPU utilization stays high. An AR server's throughput is dominated by the sequential decode and is *memory-bandwidth-bound* per request (each token step reads the whole KV cache), so the optimization toolkit is the language-model serving toolkit: continuous batching, paged attention, speculative decoding. If your team already runs a production LLM serving stack, you can serve a native-multimodal AR model on the same infrastructure — another quiet reason the AR direction is attractive to organizations that already operate language models at scale. A diffusion model, by contrast, wants a different serving stack tuned for the denoising loop.

**Evaluation.** The paradigm shift is part of why evaluation is in crisis. FID rewards distribution matching (diffusion's strength) and under-rewards instruction-following (AR's strength), so a benchmark suite that is FID-heavy will systematically mis-rank the two. You need GenEval/T2I-CompBench/DPG-Bench for compositionality and human-preference scores for aesthetics to see the real trade-off — the central argument of [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly). Choosing a paradigm without choosing the right evaluation is how teams ship the wrong model.

**Fine-tuning and personalization.** The customization story also splits. Diffusion has the richest personalization toolkit in generative modeling — DreamBooth, Textual Inversion, LoRA, DoRA, all mature, all with thriving community ecosystems (a model has tens of thousands of community LoRAs). Adapting an AR or unified model to a new subject or style is, in principle, *structurally simpler* (it is just more fine-tuning of a language-model-shaped network, with the same LoRA machinery from `peft`) but the ecosystem is far thinner and the recipes less settled. If your product depends on cheap, abundant community-style personalization today, that pulls hard toward diffusion; if you are building a closed system and will own the fine-tuning, the AR path is fine and gets you in-context adaptation as a bonus.

The throughline back to the series' spine: the generative trilemma still rules. Diffusion sits at "high quality, parallel-but-iterative speed, excellent diversity" with a known path to fast sampling via distillation. Pure AR sits at "excellent semantics and unification, sequential speed cost, strong likelihood-based diversity" with a known path to faster sampling via next-scale/masked decoding. Fusion is the attempt to sit at a better point on the trilemma than either corner — and so far, on the evidence, it is working. The honest forecast for 2027: the frontier closed-source systems will be increasingly fusion/native-multimodal because unification and instruction-following are where user-facing value concentrates, while open high-resolution aesthetic generation stays diffusion-led until the AR/fusion tooling and distillation catch up. Bet on convergence, not on a single winner — and build so you can swap the generator behind a stable interface, because the paradigm under it is going to keep moving.

## 12. Key takeaways

- **The fork is the factorization.** AR factorizes $p(x)$ over space/scale (chain-rule next-token, explicit likelihood); diffusion factorizes over noise level (denoising score matching, implicit). Every trade-off downstream falls out of that one choice.
- **Diffusion still leads open high-res aesthetics and distilled speed.** FLUX/SD3-class models produce the cleanest 1024×1024 open results and, distilled, the fastest. That is the 2026 default for "make me a beautiful image fast."
- **AR leads prompt fidelity, world-knowledge, and unification.** A model that *is* a language model counts, binds attributes, renders text, and follows instructions better — and unifies understanding with generation natively.
- **The AR resurgence is real and quantified.** VAR (FID ≈ 1.8) and MAR (FID ≈ 1.55) beat DiT-XL/2 (≈ 2.27) on ImageNet 256, with VAR showing a clean power-law scaling law. Next-scale and masked decoding fixed AR's error-accumulation and latency problems.
- **MAR is a literal fusion.** It drops vector quantization by modeling each continuous token with a small diffusion head — AR for the sequence, diffusion for the token distribution.
- **Unified models split three ways.** Early-fusion tokens (Chameleon, Emu3), decoupled encoders (Janus-Pro), mixed objectives (Transfusion, Show-o). The design choice sets the quality-vs-unification trade-off.
- **The paradigms are converging hard.** Same transformer backbone, mixing losses (MAR, Transfusion), and converging generation orders (coarse-to-fine AR ≈ coarse-to-fine denoising; few-step diffusion ≈ few-round AR).
- **The frontier is fusion, not a winner.** The right 2026 question is not "AR or diffusion?" but "which objective for which token, in one shared transformer?" — and the loss-weight $\lambda$ between them is the new key knob.
- **Pick by the axis you actually need.** Aesthetics-and-speed → diffusion; semantics-and-unification → AR/native-multimodal; building-new-and-want-both → fusion. Match your evaluation to that axis or you will mis-rank the models.

## Further reading

- Tian et al., **"Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction" (VAR)**, NeurIPS 2024 (best paper) — the next-scale resurgence and the AR scaling law.
- Li et al., **"Autoregressive Image Generation without Vector Quantization" (MAR)**, 2024 — continuous tokens with a per-token diffusion loss; the fusion move.
- Zhou et al., **"Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"**, Meta 2024 — one transformer, two objectives.
- Team, **"Chameleon: Mixed-Modal Early-Fusion Foundation Models"**, Meta 2024, and **Emu3**, BAAI 2024 — early-fusion token unification.
- Chen et al., **"Janus / Janus-Pro: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation"**, DeepSeek 2024–2025 — the decoupled-encoder design.
- Fan et al., **"Fluid: Scaling Autoregressive Text-to-image Generative Models with Continuous Tokens"**, Google 2024 — the AR scaling study (continuous + random-order wins).
- Peebles & Xie, **"Scalable Diffusion Models with Transformers" (DiT)**, ICCV 2023, and Esser et al., **"Scaling Rectified Flow Transformers" (SD3)**, 2024 — the diffusion baselines this post compares against.
- Within this series: [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) (the VAR/MAR/GPT-Image foundation), [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit), [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe), [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing), [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
