---
title: "FLUX Deep-Dive: Inside Black Forest Labs' Flow-Matching Transformer"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A full teardown of FLUX.1 and FLUX.2 from Black Forest Labs: the 12B rectified-flow transformer with its hybrid double-then-single-stream blocks, the CLIP+T5 encoder stack, guidance and timestep distillation, the pro/dev/schnell tiers, the Kontext and Fill/Redux/Canny/Depth control suite, with runnable diffusers code and honest numbers."
tags:
  [
    "image-generation",
    "diffusion-models",
    "flux",
    "rectified-flow",
    "flow-matching",
    "mm-dit",
    "text-to-image",
    "guidance-distillation",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/flux-architecture-deep-dive-1.png"
---

In August 2024, a model called FLUX.1 appeared on Hugging Face, downloaded a few hundred thousand times in its first week, and quietly reset everyone's expectations of what an *open* text-to-image model could do. Type "a storefront window at dusk, gold-leaf lettering that reads FRESH BREAD & PASTRIES, a tabby cat asleep on the sill, the whole scene reflected in the wet pavement" into FLUX.1-dev and you get back legible gold lettering, a believable cat with the right number of legs, and a reflection that respects the geometry of the scene. The same prompt into SDXL — the open frontier of eight months earlier — gives you a moody storefront with a sign that reads something like "FRSEH BARED" and a cat with a suspicious extra paw. FLUX did not invent a new paradigm. It took the MM-DiT recipe that Stable Diffusion 3 introduced, redesigned the transformer, scaled it to roughly twelve billion parameters, distilled the guidance into the weights, and shipped it under an Apache license at the small end. The result was the model that set the 2024–2025 quality bar.

This post is a full teardown of that model. We will trace the lineage — the same researchers who designed SD3 at Stability AI left to found Black Forest Labs (BFL) and built FLUX as the next step. We will dissect the architecture, which is the interesting part: FLUX is not a plain stack of MM-DiT blocks. It is a *hybrid* — a run of **double-stream** blocks where the image and text token streams keep separate weights and co-attend, followed by a longer run of **single-stream** blocks where the two streams are concatenated into one sequence and pushed through a unified transformer. We will work out *why* that hybrid exists, what each half buys you, and what the parameter and compute accounting looks like. We will explain **guidance distillation** — the trick that lets FLUX.1-dev bake classifier-free guidance into the weights so that the guidance scale becomes a conditioning *input* rather than a second forward pass — and **timestep distillation**, which compresses FLUX.1-schnell down to one-to-four sampling steps. And we will cover the wider FLUX ecosystem: the pro/dev/schnell tiers, FLUX.1 Kontext for in-context editing, the Fill/Redux/Canny/Depth control suite, and the FLUX.2 line that lifted resolution, text rendering, and parameter count.

![A dataflow figure showing image patch tokens and text tokens entering a run of double-stream blocks, then being concatenated into one sequence for a longer run of single-stream blocks, ending in a predicted velocity field](/imgs/blogs/flux-architecture-deep-dive-1.png)

Throughout, I will hold to the spine of this whole series: the **generative trilemma** (sample quality versus mode coverage versus sampling speed) and the **diffusion stack** (data → VAE latent → forward noising → denoiser net → ODE/SDE sampler → guidance → image). FLUX is a particular set of answers to that trilemma — a big, slow, extremely high-quality denoiser, with a distilled variant that trades a little quality for a ten-times speedup. Because FLUX is open, we do not have to guess at the numbers. We can read the config, run the weights on a 4090, and measure. Where I am genuinely unsure of a figure — FLUX.1-pro's exact parameter count, FLUX.2's internal details — I will mark it approximate rather than invent a precise number. By the end you will understand the architecture well enough to sketch a double-stream-to-single-stream block in PyTorch, run all three open tiers in 🤗 `diffusers`, and make a defensible decision about when FLUX is the right tool and when SANA or SD3.5 is the better one.

If you want the foundations under this post — flow matching, MM-DiT, the diffusion transformer, distillation — they each have their own deep-dive in this series, linked in place. This one assumes you have seen a `diffusers` pipeline before and want to understand the specific model that everyone in the open-source image world spent 2024 and 2025 building on top of.

## 1. The lineage: from SD3 to Black Forest Labs

You cannot understand FLUX's design without understanding where its authors came from, because FLUX is a deliberate *answer* to the limitations they hit shipping Stable Diffusion 3. The core of the team — Robin Rombach, Andreas Blattmann, Patrick Esser, Dominik Lorenz, and others — were the authors of the original Latent Diffusion Models paper (Rombach et al., 2022), the paper that became Stable Diffusion. They then led the work on Stable Diffusion 3, whose key paper, "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Esser et al., 2024), introduced the MM-DiT block and made the case for flow matching over the classic DDPM objective. In early 2024 that group left Stability AI and founded Black Forest Labs, named for the region of Germany, and raised a substantial seed round to build the next generation of open image models.

What they carried forward from SD3 is the *recipe*: diffusion (really, flow matching) in a VAE latent space, a transformer backbone rather than a U-Net, joint attention between text and image tokens, and a strong dual text-encoder stack of CLIP plus T5-XXL. If you have read the [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) post in this series, all of that is familiar territory. What they *changed* is the transformer itself, the scale, and the distribution strategy. SD3 in its largest released form is an 8B-parameter MM-DiT where every block keeps the image and text streams separate. FLUX is a ~12B-parameter transformer that runs the separate-stream design for only the first portion of its depth and then switches to a concatenated single-stream design for the rest. And where SD3 shipped as a single model that you sample with classifier-free guidance, FLUX shipped as a *family* — pro, dev, schnell — with guidance and step count distilled to different operating points.

There is a real engineering story in that change, and it is worth being precise about the motivation. SD3's MM-DiT is parameter-heavy: keeping separate query/key/value projections, separate MLPs, and separate normalization for two streams at every layer roughly doubles the parameter cost of each block compared to a single-stream transformer of the same width. That is *expensive*, and the question the BFL team clearly asked themselves is: do you actually need separate-stream processing at every layer, or only at the layers where modality-specific feature extraction matters most? FLUX's hybrid is the empirical answer — separate streams early, merged streams late — and the rest of this post is, in large part, about why that answer makes sense.

#### Worked example: the parameter cost of "separate streams everywhere"

Consider a transformer block at hidden width $d = 3072$ (FLUX's width). A standard single-stream block has roughly $4d^2$ parameters in attention (Q, K, V, and output projections) and roughly $8d^2$ in a typical MLP with a 4× expansion (up and down projections), for about $12d^2 \approx 113$M parameters per block. A double-stream block that keeps *separate* weights for the image and text streams pays that cost twice — about $24d^2 \approx 226$M parameters per block — for the same width. If you built a 57-block transformer with double-stream blocks throughout, you would be at roughly $57 \times 226\text{M} \approx 12.9$B parameters from the blocks alone. FLUX instead uses (approximately) 19 double-stream blocks and 38 single-stream blocks. The single-stream blocks are *also* tuned to be parameter-efficient — FLUX fuses the attention and MLP input projections and uses a parallel attention-MLP layout — so they cost less than a naive $12d^2$. The hybrid lands the whole transformer near 12B instead of the ~13B+ a separate-stream-everywhere design of the same depth would hit, while the team's ablations indicated the early double-stream layers carry most of the modality-specific benefit. The numbers above are order-of-magnitude bookkeeping to show the *shape* of the trade-off; the exact per-block counts depend on the MLP ratio and the fused-projection details.

That is the lineage and the central design tension in one paragraph: SD3 proved the MM-DiT recipe; FLUX kept the recipe and re-engineered the transformer to spend its parameter budget where it earns the most quality. Now let us open up the architecture and see exactly how.

## 2. The big picture: a rectified-flow transformer

Before we get into blocks, let us situate FLUX in the diffusion stack so the moving parts have somewhere to live. FLUX is a latent generative model. It does not operate on pixels; it operates on a compressed latent produced by a VAE, exactly as Latent Diffusion does ([latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) covers that compression argument in depth). FLUX's VAE encodes a $1024 \times 1024$ RGB image into a latent of shape $16 \times 128 \times 128$ — that is, 16 channels at an 8× spatial downsample. The transformer then *patchifies* this latent into tokens. FLUX uses a $2 \times 2$ patch, so a $128 \times 128$ latent becomes a $64 \times 64$ grid of tokens, $4096$ image tokens, each a vector formed by flattening a $2 \times 2 \times 16 = 64$-dimensional patch and projecting it up to the transformer width.

On the text side, FLUX runs two encoders. A CLIP text encoder (the ViT-L/14 CLIP used by SD and SDXL) produces a single *pooled* embedding — a global summary of the prompt, one vector. A T5-XXL encoder (the 4.7B-parameter encoder-only T5) produces a *sequence* of token embeddings, typically up to 256 or 512 tokens, that carry the fine-grained, position-aware meaning of the prompt. The pooled CLIP vector is added into the timestep/conditioning signal that modulates every block (the same role AdaLN-Zero plays in a DiT — see [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit) for how that modulation works). The T5 sequence becomes the *text tokens* that the image tokens attend to. So when you read "text tokens" in FLUX, think T5-XXL output; when you read "conditioning vector," think pooled CLIP plus the timestep embedding plus, for the distilled models, the guidance embedding.

The training objective is **rectified flow**, a particular instance of flow matching. Rather than learning to predict the noise $\epsilon$ added at each diffusion step (the DDPM/ε-prediction objective), FLUX learns a *velocity field*. You define a straight-line path in latent space between a sample of pure noise $x_1 \sim \mathcal{N}(0, I)$ and a real data latent $x_0$: $x_t = (1-t)\,x_0 + t\,x_1$ for $t \in [0,1]$. The velocity along this path is constant, $\frac{dx_t}{dt} = x_1 - x_0$, and the network $v_\theta(x_t, t, c)$ is trained to regress that velocity given the noisy point, the time, and the conditioning $c$. At sampling time you start from noise at $t=1$ and integrate the learned velocity field back to $t=0$ with an ODE solver. Because the target paths are straight, the ODE is close to linear and needs few steps. The full derivation lives in [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow); here the thing to hold onto is that FLUX's network outputs a velocity, the loss is a simple regression, and the straight-line geometry is exactly what makes few-step distillation (the schnell tier) feasible later.

So the FLUX forward pass, end to end, is: encode the prompt with CLIP and T5, encode the noisy latent into image tokens via patchify, run the hybrid transformer to predict a per-token velocity, then unpatchify and let the ODE sampler take a step. Repeat for the chosen number of steps, then VAE-decode the final latent to pixels. Everything novel about FLUX is in the middle box — the hybrid transformer — so that is where we spend the most time.

## 3. The hybrid transformer: double-stream then single-stream

Here is the architectural heart of FLUX, and the thing that distinguishes it from SD3. FLUX's transformer is split into two phases. The first phase is a run of **double-stream blocks** (often called MM-DiT blocks, the SD3 design): the image tokens and the text tokens flow through the block in two *separate* streams, each with its own normalization, its own QKV projection, and its own MLP, but they meet in a single *joint attention* operation where the concatenated image-and-text tokens attend to each other. The second phase is a longer run of **single-stream blocks**: the image and text tokens are concatenated into one flat sequence and pushed through a unified transformer block with one shared set of weights for everything.

The released FLUX.1 models use approximately 19 double-stream blocks followed by approximately 38 single-stream blocks. (These counts come from the public model config and community analysis of the open weights; I am stating them as the released configuration.) That is roughly a 1:2 ratio — a third of the depth keeps the streams separate, two-thirds merges them. The hidden width is 3072 across both phases, and attention uses 24 heads of dimension 128.

![A dataflow figure of one double-stream block showing the image and text streams each with their own QKV projection feeding a single joint attention with rotary embeddings, then splitting back into separate per-stream MLP paths](/imgs/blogs/flux-architecture-deep-dive-3.png)

Let us be precise about what a **double-stream block** does, because the word "separate" is doing a lot of work. Inside one double-stream block:

1. The image tokens get their own adaptive layer-norm, modulated by the conditioning vector (timestep + pooled CLIP + guidance), and are projected to image queries, keys, and values by *image-specific* weights. The text tokens do the same with *text-specific* weights.
2. Rotary position embeddings (RoPE) are applied to the image and text queries and keys. FLUX uses a 2-D RoPE for the image tokens (so an image token's position encodes its row and column in the $64\times64$ grid) and a positional scheme for text tokens. RoPE is what tells the attention *where* each token sits without adding learned absolute position embeddings.
3. The image and text queries/keys/values are *concatenated* along the sequence dimension and a **single joint attention** is computed over the whole $4096 + |\text{text}|$ token set. This is the crucial part: an image patch can attend to any text token and vice versa, in one attention matrix. After attention, the result is split back into the image part and the text part.
4. Each stream then goes through its *own* MLP — image MLP for image tokens, text MLP for text tokens — with residual connections.

So a double-stream block is "separate everywhere except the one place where the two modalities have to actually talk to each other." The separate QKV and MLP weights let the model learn modality-specific transformations: image features and text features genuinely are different kinds of object, and giving each its own parameters early lets the network specialize. The shared attention is where the binding happens — where "the sign reads FRESH BREAD" gets connected to the patches that will become the sign.

A **single-stream block** drops the separation. The image and text tokens are concatenated into one sequence of length $4096 + |\text{text}| \approx 4250$ tokens and treated uniformly: one normalization, one QKV projection, one attention, one MLP, all with weights shared across both modalities. FLUX's single-stream blocks also use a *parallel* attention-and-MLP design (the attention and the MLP read the same normalized input and their outputs are summed, rather than the MLP reading the attention output) and *fuse* the attention QKV projection with the MLP input projection into one big linear layer. Both of those are efficiency tricks borrowed from large language model transformers — they reduce the number of separate matrix multiplies and improve hardware utilization.

![A layered figure of a single-stream block showing merged image and text tokens passing through a modulated norm, a fused shared projection, a self-attention branch and a parallel MLP branch, then a fused output projection](/imgs/blogs/flux-architecture-deep-dive-4.png)

Why this order — double first, then single? Two reasons, one about *features* and one about *parameters*.

The **feature argument**: early layers of the network are where low-level, modality-specific structure is extracted. Image patches need to learn local texture, edges, and color statistics; text tokens need to resolve syntax and word meaning. These are different jobs and benefit from different weights, so the double-stream blocks sit early. By the time you are deep in the network, the representations have become more *shared and high-level* — both streams are now carrying semantic content about *what the image should contain*, and the distinction between "this is an image token" and "this is a text token" matters less. At that point a single shared set of weights operating over the merged sequence is sufficient, and the joint processing may even help by letting information flow freely between modalities without the bottleneck of separate MLPs. There is a useful analogy to encoder-decoder transformers in language: the early, modality-specialized layers are like separate encoders that each understand their own input, and the late, merged layers are like a fused decoder that reasons over everything at once. FLUX just makes the hand-off learnable and puts it at a specific depth rather than at a hard architectural boundary.

There is also a subtler benefit to merging late. In the double-stream phase, the text tokens are *updated* by attention — they are not frozen the way a U-Net's cross-attention keys are. So by the time the streams merge, the text representation has already adapted to the image it is helping to build. The single-stream phase then operates on text tokens that are no longer raw T5 outputs but image-aware semantic tokens, which is part of why FLUX binds attributes to the right objects so reliably: the text has, in effect, been "told" what the image looks like before the final layers commit the details. This is the same bidirectional-attention advantage that MM-DiT introduced over U-Net cross-attention, carried through FLUX's whole stack and then concentrated in the cheaper single-stream layers.

The **parameter argument** is the one from the worked example in §1: double-stream blocks cost roughly twice the parameters of a single-stream block of the same width, because they keep duplicate weights. If you used double-stream blocks for the whole depth, you would either blow your parameter budget or have to make the model shallower. By using single-stream blocks for the bulk of the depth, FLUX gets *more layers of processing per parameter* — the single-stream blocks are the cheap way to add depth. So the hybrid is a deliberate allocation: spend the expensive separate-stream parameters early where modality specialization pays off, and spend the cheap shared-stream parameters on raw depth later where it does not. It is the same instinct as putting your most expensive engineers on the hardest part of the problem and a larger, cheaper team on the bulk of the work.

#### Worked example: counting FLUX's compute, double vs single

Attention is the same cost in both block types because both compute one joint attention over the full $\sim 4250$-token sequence — that is $O(N^2 d)$ with $N \approx 4250$ and $d = 3072$, dominated by the image tokens. The *difference* is in the projections and MLPs. A double-stream block does two separate sets of QKV projections and two separate MLPs; a single-stream block does one fused set. For the linear (non-attention) part of a block, the double-stream block does roughly twice the matmul work of the single-stream block. With 19 double-stream and 38 single-stream blocks, the *linear* compute is split roughly $19 \times 2 : 38 \times 1 = 38 : 38$ — that is, the double-stream third of the network and the single-stream two-thirds cost about the same in linear FLOPs, while the quadratic attention cost is uniform across all 57 blocks. The practical consequence: at $1024\times1024$ the attention over 4096 image tokens dominates per-step latency, which is why feature-caching tricks like TeaCache (covered in [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference)) target the attention path. This is approximate FLOP accounting, not a profiler trace, but it correctly predicts where the time goes.

## 4. Rotary position embeddings and why FLUX uses them

A small but important detail: FLUX uses **rotary position embeddings (RoPE)** for both image and text tokens, rather than the learned absolute position embeddings that the original DiT and SD3 used. RoPE, introduced for language models (Su et al., 2021), encodes position by *rotating* the query and key vectors by an angle proportional to their position before computing attention. Because attention depends on the dot product of a query and a key, and rotating both by an amount proportional to their *relative* offset changes that dot product in a position-dependent way, RoPE effectively makes attention sensitive to *relative* position without ever storing an absolute position table.

For image generation this has two practical benefits. First, it generalizes across resolutions and aspect ratios better than a fixed learned table: a learned absolute embedding has one slot per position and breaks when you ask for a resolution it never saw, whereas RoPE's rotation formula extends smoothly to new positions. FLUX can therefore generate at a range of resolutions and aspect ratios more gracefully. Second, the 2-D RoPE FLUX uses for image tokens encodes the row and column of each patch directly into the rotation, giving the attention a clean, parameter-free notion of spatial layout — useful when the model has to keep a sign's letters in the right order or a face's features in the right arrangement. The text tokens get their own positional treatment so the model knows token order in the prompt. None of this is unique to FLUX — RoPE is now standard in many transformers — but it is part of why FLUX handles varied resolutions and crisp spatial structure as well as it does.

## 5. The text-encoder stack: CLIP plus T5-XXL

FLUX inherits SD3's dual text-encoder design, and it is worth understanding why two encoders rather than one. CLIP and T5 do different jobs.

**CLIP** (specifically the OpenAI ViT-L/14 text encoder) was trained contrastively to align text with images. Its embeddings are excellent at capturing the *global semantic gist* of a prompt — the overall subject, style, and mood — because that is what the contrastive image-text objective rewards. But CLIP's text encoder has a short context (77 tokens) and a relatively weak grasp of fine-grained compositional structure and exact wording. FLUX uses CLIP's *pooled* output — a single global vector — and feeds it into the conditioning signal that modulates every block via AdaLN. So CLIP sets the overall direction.

**T5-XXL** (the 4.7B encoder-only T5) was trained as a general language model on text-to-text tasks, never on images. It has a much stronger, more precise representation of language: word order, negation, attribute binding, exact spellings. FLUX uses T5's full *token sequence* — up to 256 or 512 tokens depending on the configuration — as the text tokens that the image tokens attend to in joint attention. So T5 carries the detailed, position-aware meaning that lets FLUX render legible text and obey multi-clause prompts.

The division of labor is the point: CLIP gives a cheap global summary that conditions the whole network, and T5 gives a rich, long, precise sequence that the attention can query token-by-token. The [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) post in this series goes deep on this CLIP-versus-T5 trade-off and the attribute-binding failure modes that motivate it; here it is enough to know that FLUX's strong text rendering owes a great deal to T5-XXL doing the heavy lifting on language.

One practical consequence worth flagging: T5-XXL is a 4.7B-parameter model, and it lives in memory alongside the 12B transformer and the VAE. That is why running FLUX naively wants a lot of VRAM, and why `enable_model_cpu_offload()` — which keeps the text encoder, transformer, and VAE on CPU and moves each to GPU only when it runs — is so important on consumer cards. We will use it in the code.

## 6. Flow matching and rectified flow, briefly

We covered the objective in §2, but let us make the *why few steps* argument concrete because it is what makes the schnell tier possible. In classic DDPM the model learns to reverse a curved, stochastic noising process; the reverse trajectory through latent space wiggles, and a coarse ODE/SDE solver with few steps accumulates error and produces artifacts, which is why naive DDPM wanted hundreds of steps. Rectified flow instead trains the model so that the *transport path* from noise to data is as close to a straight line as possible. The training target is the constant velocity $x_1 - x_0$ along the straight interpolation $x_t = (1-t)x_0 + t x_1$. If the learned velocity field were exactly the straight-line field, you could integrate it in a *single* Euler step and land exactly on the data. In practice the field is only approximately straight — averaging over many noise-data pairs bends it — but it is straight *enough* that a handful of steps suffices, and far straighter than a DDPM trajectory.

This is the geometric reason FLUX samples well in 20–50 steps for the full models and the reason FLUX.1-schnell can be distilled to 1–4 steps: the underlying flow is already nearly linear, so there is little curvature for a distillation procedure to have to approximate away. The full derivation, including reflow (the procedure that iteratively straightens the paths) and the connection to optimal transport, is in [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow). The one formula to carry forward is the loss:

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t, x_0, x_1}\left[\;\big\lVert\, v_\theta\big((1-t)x_0 + t x_1,\; t,\; c\big) - (x_1 - x_0) \,\big\rVert^2\,\right].
$$

It is a plain mean-squared-error regression of the network's predicted velocity onto the constant straight-line velocity. No variational bound, no noise schedule to tune beyond the time sampling distribution — which, in SD3 and FLUX, is a logit-normal weighting that puts more training emphasis on the middle of the trajectory where the model has the most to learn. That simplicity is part of why the recipe scaled so cleanly.

One subtlety that matters at high resolution: **timestep shifting**. The amount of noise that corresponds to a given $t$ does not have the same visual effect at $256\times256$ as at $1024\times1024$, because a higher-resolution latent has more tokens and the signal-to-noise ratio per token shifts with resolution. SD3 and FLUX correct for this by *shifting* the sampling timesteps as a function of the image resolution (the larger the image, the more the schedule is pushed toward higher noise levels early), so that the effective denoising difficulty is matched across resolutions. In `diffusers` this shows up as the `FlowMatchEulerDiscreteScheduler`'s `shift` parameter and the resolution-dependent shifting that `FluxPipeline` applies automatically. If you ever generate at an unusual resolution and the images come out under- or over-detailed, the timestep shift is the knob behind it. This is the kind of detail that does not change the big picture but quietly decides whether a $1536\times1536$ generation looks right — a reminder that the schedule, not just the architecture, carries quality.

The VAE is also worth a paragraph, because FLUX's is part of why it looks the way it does. FLUX uses a 16-channel VAE (versus the 4-channel VAE of SD1.5/SDXL), and the channel count of the latent space directly bounds how much fine detail the model can represent. A 4-channel latent at an 8× spatial downsample compresses a $512\times512\times3$ image into a $64\times64\times4$ latent — a 12× compression that throws away a lot of high-frequency information, which is part of why SD1.5 struggled with fine text and small faces. FLUX's 16-channel latent (the same channel count SD3 adopted) quadruples the per-pixel latent capacity, so the latent retains far more high-frequency structure for the transformer to work with and for the VAE to decode back into crisp edges and legible letters. This is not a flashy architectural trick, but it is one of the unglamorous reasons FLUX renders text and fine detail so much better than the SD1.5 generation — there is simply more information surviving the compression. The trade is that a 16-channel latent is more expensive to denoise (more channels per token), but at FLUX's scale that cost is in the noise compared to the 12B transformer.

## 7. The three tiers: pro, dev, and schnell

FLUX.1 did not ship as one model. It shipped as three, each a different point on the quality-versus-speed-versus-license frontier, and understanding them is the most practically useful part of this post because *which tier you pick* determines your cost, latency, and legal options.

![A comparison grid of the FLUX tiers with rows for pro, dev, schnell, and FLUX.2 and columns for parameters, steps, guidance distillation, license, and access](/imgs/blogs/flux-architecture-deep-dive-5.png)

**FLUX.1-pro** is the flagship, available only through the BFL API (and partner platforms). It is the highest-quality tier, uses standard classifier-free guidance, and its weights are closed. You reach it over an API and pay per image. Think of it as the reference point: the best FLUX can do, with no open weights. (BFL later shipped FLUX1.1-pro, faster and higher quality, and ultra/raw variants, all API-only.)

**FLUX.1-dev** is the open-weights model that most of the open-source ecosystem actually uses. It is *guidance-distilled*: the classifier-free guidance behavior of the pro model has been baked into the weights, so FLUX.1-dev runs a single forward pass per step (not the two passes CFG normally requires) and takes a `guidance_scale` as a *conditioning input* rather than as an extrapolation coefficient. We will unpack guidance distillation in detail in §8. FLUX.1-dev ships under a non-commercial community license — you can use it freely for research and personal work and it is the basis for the vast majority of community fine-tunes and LoRAs, but commercial use requires a separate BFL license. It samples well in roughly 20–50 steps.

**FLUX.1-schnell** ("schnell" is German for "fast") is the open, Apache-2.0 tier, and it is the one you can use commercially without restriction. On top of guidance distillation it is *timestep-distilled* to generate good images in **1 to 4 steps**. That is a ten-to-fifty-times reduction in sampling cost versus the dev model, at some cost in fine detail and diversity. The Apache-2.0 license is significant: it makes schnell the default choice for products and for derivative models that need a clean commercial license. The trade is quality — schnell is genuinely good but not quite at dev's level of fine detail and prompt adherence, exactly the quality-versus-speed point on the trilemma you would expect from a step-distilled model.

Here is the same information as a table, with the caveats stated:

| Tier | Params | Steps | Guidance | License | Access | Best for |
|------|--------|-------|----------|---------|--------|----------|
| FLUX.1-pro | ~12B (approx) | ~25–50 | CFG | Closed | API only | Highest quality, no weights needed |
| FLUX.1-dev | ~12B | ~20–50 | Distilled (input) | Non-commercial | Open weights | Research, fine-tuning, best open quality |
| FLUX.1-schnell | ~12B | 1–4 | Distilled + timestep | Apache-2.0 | Open weights | Fast/commercial, products, derivatives |
| FLUX.2-dev | larger (approx) | ~20–50 | Distilled (input) | Non-commercial | Open weights | Higher res, better text, more params |

The "~12B" figures are the widely-reported transformer parameter count for the FLUX.1 line; the pro count is not officially broken out, so I mark it approximate. The FLUX.2 parameter count is larger than FLUX.1 but I will not assert a precise number — treat it as approximate and bigger. The step ranges are typical operating points, not hard limits.

The thing to internalize is that all three FLUX.1 tiers are *the same backbone* at *different operating points*. Pro is the undistilled reference, dev is guidance-distilled and open, schnell is additionally timestep-distilled and Apache. The distillation is what differentiates them, so let us understand it properly.

## 8. Guidance distillation: baking CFG into the weights

This is the single most important *technique* in the FLUX family to understand, because it changes how you sample and why FLUX.1-dev is so much cheaper than it looks. Start from classifier-free guidance, the technique covered in full in [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance). Standard CFG works like this: at each sampling step you run the network *twice* — once with the prompt conditioning $c$ and once with a null/empty conditioning $\varnothing$ — and you extrapolate:

$$
\tilde{v}_\theta(x_t, t, c) = v_\theta(x_t, t, \varnothing) + s\,\big(v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing)\big),
$$

where $s$ is the guidance scale. Pushing $s$ above 1 amplifies the direction that the prompt adds, sharpening prompt adherence and fidelity at the cost of diversity. The catch is the *two forward passes per step*: CFG literally doubles your inference compute. For a 12B model at $1024\times1024$, that is a serious tax.

**Guidance distillation** removes the second pass. The idea: train a *student* network that, given the noisy latent, the time, the conditioning, *and the guidance scale $s$ as an extra input*, directly predicts the CFG-extrapolated velocity $\tilde{v}_\theta(x_t, t, c)$ that the teacher would have produced with that scale — in a single forward pass. The guidance scale stops being an extrapolation coefficient you apply *outside* the network and becomes a *conditioning input* you feed *into* the network, encoded as a sinusoidal embedding and added to the timestep/conditioning signal exactly like the timestep is. The student learns to emulate "run me twice and extrapolate at scale $s$" with one pass.

![A two-column figure contrasting classic classifier-free guidance, which runs two forward passes and extrapolates, against guidance distillation, which runs a single forward pass with the guidance scale fed in as a conditioning input](/imgs/blogs/flux-architecture-deep-dive-2.png)

The consequences are exactly what you would want. FLUX.1-dev does **one** forward pass per step instead of two, halving per-step compute relative to a CFG model of the same size. The `guidance_scale` you pass in `diffusers` is *not* doing CFG extrapolation — it is being fed to the network as the conditioning value the distillation taught it to respond to. This is why FLUX.1-dev's effective guidance values look different from SD's: a typical FLUX-dev `guidance_scale` is around 3.5, and pushing it the way you would push SD's CFG to 7.5 over-cooks the image, because the meaning of the number changed. It is also why, technically, FLUX.1-dev does not support a "true" negative prompt in the CFG sense out of the box — there is no second uncond pass to steer with. (Community tooling adds CFG-style negative prompting back on top, at the cost of the second pass, but the model as shipped does not need it.)

How is the student trained? In broad strokes, you take the teacher (the undistilled model that does the two-pass CFG) and, for many noisy latents and many sampled guidance scales, you have it produce the CFG-extrapolated target $\tilde{v}$. You then train the student to match that target in one pass, with the sampled scale $s$ given as an input. The student therefore internalizes a whole *family* of behaviors — "what would the CFG output be at scale 2, at scale 3.5, at scale 5" — and you select among them at inference by passing the scale. The loss is again a regression of the student's single-pass output onto the teacher's two-pass output, so it is the same MSE machinery as the base training, just with a teacher providing the targets and the scale threaded through as a condition. The practical upshot is that the released FLUX.1-dev weights are *already* a guidance-distilled student; you do not run CFG, you run the student and tell it the scale. (BFL has not published every training detail, so treat the procedure sketch as the well-understood shape of guidance distillation rather than a line-by-line account of FLUX's exact recipe.)

For **FLUX.1-schnell**, guidance distillation is combined with **timestep distillation** (in the spirit of the consistency and adversarial-distillation methods covered in [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) and [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation)). Timestep distillation trains the network so that a *single* (or a handful of) large ODE steps reproduce what the teacher achieves with many small steps. Because rectified flow's trajectories are already nearly straight (§6), the distillation has little curvature to fight, which is exactly why schnell can land at 1–4 steps without collapsing into mush. The combination — guidance baked in *and* steps collapsed — is what makes schnell roughly an order of magnitude faster than dev. The cost, honestly stated, is some loss of fine detail, occasional reduced diversity, and slightly weaker handling of the hardest compositional prompts. That is the quality-for-speed trade the trilemma guarantees you cannot escape; schnell just picks a very aggressive point on it and picks it well.

There is a deeper reason the distilled tiers matter beyond raw speed: they change the *economics* of who can run the model. An undistilled 12B model that needs two forward passes per step at 50 steps is 100 transformer evaluations per image — a serious bill on cloud GPUs and a non-starter on a laptop. A guidance-distilled model at 28 steps is 28 evaluations. A timestep-distilled model at 4 steps is 4. That is the difference between "you need a datacenter GPU and a budget" and "you can run this on a gaming laptop in a couple of seconds." Distillation is therefore not just a quality knob; it is the lever that turned FLUX from a model you call over an API into a model the entire open-source community could run, fine-tune, and build products on. The fact that schnell is *also* Apache-2.0 sealed it: cheap to run *and* legally clean. That combination is a large part of why FLUX, rather than any single closed model, became the center of gravity for open image generation in 2024–2025.

#### Worked example: the compute you save with distillation

Take FLUX.1-dev at 28 steps and compare against a hypothetical undistilled CFG model of the same size, also at 28 steps. The CFG model does $28 \times 2 = 56$ forward passes through the 12B transformer. FLUX.1-dev does $28 \times 1 = 28$ passes — a 2× saving from guidance distillation alone. Now take FLUX.1-schnell at 4 steps: $4 \times 1 = 4$ passes. Relative to the 56-pass CFG baseline, schnell does **14× fewer** transformer evaluations. On an RTX 4090, where a single $1024\times1024$ FLUX transformer pass is on the order of a few hundred milliseconds, that is the difference between a ~20-second image and a ~1.5-second image (approximate, hardware- and precision-dependent). The point of the arithmetic: the FLUX tier you choose is, at its core, a choice of how many 12B-transformer forward passes you are willing to pay per image — 56 (CFG-equivalent), 28 (dev), or 4 (schnell).

## 9. Running FLUX in diffusers

Enough theory — let us run all three open tiers. 🤗 `diffusers` has first-class FLUX support via `FluxPipeline` and `FluxTransformer2DModel`, with the `FlowMatchEulerDiscreteScheduler` as the default sampler. Here is FLUX.1-schnell at 4 steps in bf16 with CPU offload, the configuration that runs comfortably on a 24 GB card:

```python
import torch
from diffusers import FluxPipeline

# FLUX.1-schnell: Apache-2.0, timestep-distilled, 1-4 steps.
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,   # FLUX is trained and released in bf16
)
# Keep the T5 (4.7B), transformer (12B), and VAE on CPU and stream each to
# GPU only while it runs. This is what makes 24 GB enough.
pipe.enable_model_cpu_offload()

prompt = (
    "a storefront window at dusk, gold-leaf lettering that reads "
    "FRESH BREAD & PASTRIES, a tabby cat asleep on the sill, the whole "
    "scene reflected in the wet pavement, cinematic lighting"
)

image = pipe(
    prompt,
    num_inference_steps=4,        # schnell is distilled to 1-4 steps
    guidance_scale=0.0,           # schnell ignores guidance; it is baked out
    height=1024,
    width=1024,
    max_sequence_length=256,      # T5 token budget
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("flux_schnell.png")
```

Two things to notice. First, `guidance_scale=0.0` for schnell: the timestep-distilled model does not use guidance at all, so you pass zero. Second, `max_sequence_length=256` caps the T5 token count — a real knob, because longer prompts cost more attention compute over the text tokens.

For **FLUX.1-dev**, the guidance-distilled open model, the call is almost identical but you raise the step count and set a *small* guidance scale, because here `guidance_scale` is the distilled conditioning input:

```python
import torch
from diffusers import FluxPipeline

# FLUX.1-dev: guidance-distilled, non-commercial license, best open quality.
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()          # tile the VAE decode to save VRAM at 1024px

image = pipe(
    prompt="a vintage diner at night, a neon sign that reads OPEN 24 HOURS, "
           "rain on the asphalt reflecting the letters, photorealistic",
    num_inference_steps=28,       # dev's sweet spot is ~20-50
    guidance_scale=3.5,           # distilled guidance INPUT, not CFG; ~3.5 is typical
    height=1024,
    width=1024,
    max_sequence_length=512,      # dev supports up to 512 T5 tokens
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]

image.save("flux_dev.png")
```

The `guidance_scale=3.5` is doing something fundamentally different from SD's CFG, as §8 explained — it is the conditioning value the distillation taught the model to respond to, fed in once, not an extrapolation between two passes. If you crank it to 7.5 expecting SD-style behavior, you will over-saturate and harden the image. Around 3.0–4.0 is the usable band for dev.

If you are tight on VRAM, you can push further. Load the transformer in 8-bit or 4-bit with `bitsandbytes` (FLUX quantizes well, and [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) covers SVDQuant-style 4-bit diffusion that was demonstrated on FLUX specifically), or swap the full VAE for `AutoencoderTiny` to cut decode cost:

```python
import torch
from diffusers import FluxPipeline, AutoencoderTiny

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
# Tiny autoencoder: a much smaller VAE decoder, slightly lower fidelity,
# far less VRAM and faster decode. Good for previews and tight budgets.
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taef1", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

image = pipe(
    "an isometric tiny island with a lighthouse, soft pastel palette",
    num_inference_steps=4,
    guidance_scale=0.0,
    height=1024, width=1024,
    generator=torch.Generator("cpu").manual_seed(7),
).images[0]
image.save("flux_schnell_taef1.png")
```

These are the real APIs and flags, copy-and-adapt ready. The pattern to remember: `FluxPipeline`, bf16, `enable_model_cpu_offload()` for memory, `FlowMatchEulerDiscreteScheduler` as the (default) sampler, and the schnell-versus-dev distinction living entirely in `num_inference_steps` and `guidance_scale`.

## 10. Sketching the hybrid block in PyTorch

To cement the architecture, here is a stripped-down sketch of the double-stream-to-single-stream structure. This is not the real FLUX code — it omits RoPE construction, the exact modulation, and many details — but it shows the *shape* of the hybrid so you can map the official `FluxTransformer2DModel` onto it:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleStreamBlock(nn.Module):
    """Separate weights per modality; one shared joint attention."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        # Separate QKV + MLP for image and text streams.
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.txt_qkv = nn.Linear(dim, dim * 3)
        self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                     nn.Linear(dim * 4, dim))
        self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                     nn.Linear(dim * 4, dim))
        self.img_norm = nn.LayerNorm(dim)
        self.txt_norm = nn.LayerNorm(dim)

    def forward(self, img, txt, rope_img, rope_txt):
        # Project each stream with its own weights.
        iq, ik, iv = self.img_qkv(self.img_norm(img)).chunk(3, dim=-1)
        tq, tk, tv = self.txt_qkv(self.txt_norm(txt)).chunk(3, dim=-1)
        iq, ik = apply_rope(iq, ik, rope_img)   # 2-D RoPE for image tokens
        tq, tk = apply_rope(tq, tk, rope_txt)   # positional RoPE for text
        # Concatenate and run ONE joint attention over image + text tokens.
        q = torch.cat([iq, tq], dim=1)
        k = torch.cat([ik, tk], dim=1)
        v = torch.cat([iv, tv], dim=1)
        attn = sdpa(q, k, v, self.heads)        # scaled_dot_product_attention
        ai, at = attn.split([img.shape[1], txt.shape[1]], dim=1)
        # Each stream updates with its OWN MLP.
        img = img + ai + self.img_mlp(img + ai)
        txt = txt + at + self.txt_mlp(txt + at)
        return img, txt

class SingleStreamBlock(nn.Module):
    """One shared weight set over the merged sequence; parallel attn+MLP."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        # Fused projection feeds both attention QKV and the MLP input.
        self.qkv_mlp_in = nn.Linear(dim, dim * 3 + dim * 4)
        self.out = nn.Linear(dim + dim * 4, dim)   # fuse attn-out and MLP-out

    def forward(self, x, rope):
        h = self.norm(x)
        qkv, mlp = self.qkv_mlp_in(h).split([x.shape[-1] * 3, x.shape[-1] * 4], -1)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = apply_rope(q, k, rope)
        attn = sdpa(q, k, v, self.heads)
        mlp = F.gelu(mlp)
        # Parallel: attention and MLP read the SAME input, outputs concatenated.
        return x + self.out(torch.cat([attn, mlp], dim=-1))

class FluxLikeTransformer(nn.Module):
    def __init__(self, dim=3072, heads=24, n_double=19, n_single=38):
        super().__init__()
        self.double = nn.ModuleList(DoubleStreamBlock(dim, heads)
                                    for _ in range(n_double))
        self.single = nn.ModuleList(SingleStreamBlock(dim, heads)
                                    for _ in range(n_single))

    def forward(self, img, txt, rope_img, rope_txt, rope_all):
        # Phase 1: separate streams co-attend.
        for blk in self.double:
            img, txt = blk(img, txt, rope_img, rope_txt)
        # Merge and run phase 2 on one sequence.
        x = torch.cat([img, txt], dim=1)
        for blk in self.single:
            x = blk(x, rope_all)
        return x[:, :img.shape[1]]   # the image-token velocities
```

The shape is the lesson: `DoubleStreamBlock` carries duplicate `img_*` and `txt_*` weights and merges only for the attention; `SingleStreamBlock` carries one fused weight set and runs a parallel attention-plus-MLP over the concatenated sequence; the top-level module runs 19 of the first then 38 of the second and returns the image-token velocities. Modulation (the AdaLN that injects timestep, pooled CLIP, and guidance) is omitted for clarity but in the real model wraps every norm. Map this onto the official `FluxTransformer2DModel` and the `FluxTransformerBlock` / `FluxSingleTransformerBlock` classes in `diffusers` and you will recognize every piece.

## 11. The control and editing suite

FLUX is not just a base text-to-image model; BFL built an ecosystem of conditioned variants around it, each a fine-tune of the same 12B backbone for a specific control or editing task. Knowing what exists saves you from reinventing it.

![A taxonomy figure of the FLUX control suite branching from the base backbone into an editing family with Kontext, Fill, and Redux, and a structural-control family with Canny and Depth](/imgs/blogs/flux-architecture-deep-dive-8.png)

**FLUX.1 Kontext** is the most significant addition: an *in-context editing* model. Instead of conditioning only on text, Kontext takes an input image *and* an instruction and produces an edited output, in the conversational-editing paradigm covered in [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing). You give it a photo and say "make it winter" or "change the car to red" or "put the subject in a business suit," and it edits the relevant region while preserving the rest — including, notably, character identity across edits, which is the hard part. Kontext works by feeding the reference image's tokens into the same joint attention as in-context tokens, so the model attends to the source image and the instruction together. It came in dev (open) and pro (API) variants and was a major part of why FLUX stayed at the frontier of editing through 2025.

The rest of the suite is structural and inpainting control, the FLUX analogue of [ControlNet and structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control):

- **FLUX.1 Fill** — inpainting and outpainting. You provide an image and a mask and a prompt; Fill regenerates the masked region coherently with the surroundings. It is the FLUX-quality inpainting model.
- **FLUX.1 Redux** — image variation / "remix." Given an input image, Redux produces variations that preserve the overall content and style, useful for generating alternatives around a reference.
- **FLUX.1 Canny** and **FLUX.1 Depth** — structural control conditioned on a Canny edge map or a depth map respectively, so you can pin the composition (the edges or the 3-D layout) while regenerating texture and content. These ship as full fine-tuned models and as LoRA adapters.

The architectural point that ties the suite together: these are not separate architectures. They are the *same hybrid transformer*, fine-tuned with additional conditioning tokens (the reference image, the mask, the edge/depth map) fed into the joint attention. That uniformity is why the FLUX ecosystem grew so fast — once the community had the base model and the LoRA tooling, every control variant was "the same model with extra conditioning," and `diffusers` exposes them through `FluxFillPipeline`, `FluxControlPipeline`, `FluxKontextPipeline`, and friends with the same `FluxPipeline` ergonomics.

## 12. FLUX.2: what changed

In late 2025 BFL released the **FLUX.2** line, the successor generation. I will be careful here because some FLUX.2 internals are not as thoroughly documented in the open literature as FLUX.1's, so I will state what is well-established and mark the rest approximate.

FLUX.2 is a larger, higher-quality model that pushes on three axes the user community most wanted improved. First, **resolution and detail**: FLUX.2 generates cleanly at higher native resolutions with better fine detail than FLUX.1, which topped out comfortably around 1–2 megapixels. Second, **text rendering**: already FLUX.1's strongest suit, text legibility and the handling of longer strings of text improved further in FLUX.2, narrowing the gap with the closed frontier (GPT-Image and Nano Banana) on dense typography. Third, **parameters and capability**: FLUX.2 is a bigger model — the parameter count is larger than FLUX.1's ~12B, though I will not assert a precise figure — with correspondingly stronger prompt adherence and world knowledge, and it unifies generation and editing more tightly so a single model handles both text-to-image and image-editing without a separate Kontext model.

![A timeline figure tracing the lineage from SD3 through FLUX.1 dev and schnell, FLUX1.1 pro, FLUX.1 Kontext, and FLUX.2](/imgs/blogs/flux-architecture-deep-dive-7.png)

FLUX.2 keeps the recipe: a flow-matching transformer in latent space with a strong text-encoder stack, guidance distillation for the open dev tier, and the pro/dev/schnell-style tiering. The architectural DNA — hybrid transformer, RoPE, joint attention — carries over. The story of FLUX.2 is less "new paradigm" and more "the same recipe, scaled and refined," which is exactly what you would expect from a team that found a recipe that works and is now riding the scaling curve. For the precise FLUX.2 numbers, the BFL release notes are the source of truth; treat my figures here as the well-supported shape of the improvement rather than exact specifications.

There is a broader lesson in the FLUX.1-to-FLUX.2 progression that is worth naming, because it is the same lesson the language-model world learned. Once you have a clean, scalable architecture and a simple, well-behaved training objective — a transformer plus flow matching, in this case — the dominant way to improve quality is to make the model bigger, feed it more and better data, and train it longer. You do not need to reinvent the block design every generation; you need to scale what works. FLUX.1 already proved the hybrid transformer and the distillation strategy; FLUX.2 mostly turned the scale dials and refined the data and the VAE. This is why I keep emphasizing the *recipe* framing throughout this post: the value of understanding FLUX.1 deeply is that FLUX.2, and likely FLUX.3, are the same machine at larger scale, so the architecture you learned here keeps paying off. The exceptions — when a genuinely new architectural idea beats scaling — are real but rare, and when they happen (linear attention in SANA, autoregressive token models in the GPT-Image line) they tend to be about a *different* point on the trilemma, not a strictly better one. FLUX's bet is that the MM-DiT-derived flow-matching transformer is the right backbone to scale for top quality, and through 2024–2025 that bet paid off.

## 13. Honest strengths and limits

A deep-dive that only lists strengths is marketing. Here is the honest assessment, the kind you would give a colleague deciding whether to build on FLUX.

**Where FLUX genuinely leads:**

- **Text rendering.** This is FLUX's signature strength and it is real. Legible signs, labels, and short strings of text come out right far more often than with SDXL or SD1.5, owing to T5-XXL and the joint-attention recipe. It is not perfect at paragraph-length text, but for the words-on-a-sign case it set the open-model bar.
- **Prompt adherence.** Multi-clause prompts — three objects, specified colors, specified spatial relations — are obeyed more reliably than in earlier open models. The joint bidirectional attention between text and image tokens is the mechanism, and it shows up in GenEval-style compositional benchmarks.
- **Hands and anatomy.** FLUX is notably better at the classic failure case of hands than the SD lineage was. Not flawless — you will still see the occasional bad hand — but the base rate of mangled anatomy dropped sharply, which is one of the most visible quality jumps for end users.
- **Aesthetic quality at the dev tier.** FLUX.1-dev produces images that are, to most viewers, simply more polished and coherent than SDXL's, which is why it became the default base for community fine-tunes.

**Where FLUX is limited or costly:**

- **Size and speed.** It is a 12B model with a 4.7B T5 alongside it. Without offload or quantization it does not fit on smaller consumer cards, and even on a 4090 the dev model at 28 steps is not fast (tens of seconds). This is the trilemma tax for the quality — and exactly why SANA exists as the efficient alternative ([SANA's deep-compression AE and linear attention](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) discussion lives in the MM-DiT recipe post).
- **License friction at the dev tier.** FLUX.1-dev's non-commercial license means you *cannot* ship it in a commercial product without a BFL agreement. Only schnell (Apache-2.0) is clean for commercial use, and schnell trades some quality. This is a genuine constraint that pushes many products toward schnell or toward SD3.5/SANA.
- **Guidance behavior surprises.** Because dev's guidance is distilled, the `guidance_scale` does not behave like SD's CFG, and there is no native negative prompt. Engineers coming from SD trip on this constantly. It is learnable but it is a footgun.
- **Schnell's quality ceiling.** The 1–4 step model is excellent for its speed but visibly behind dev on fine detail, the hardest compositional prompts, and diversity. If you need the best quality, you pay the steps.
- **The usual generative limits.** Counting beyond a few objects, very long text strings, perfect physical consistency in reflections and shadows — these remain hard, as they do for every model in this class. FLUX moved the base rates but did not solve them.

It is worth stress-testing a couple of these failure modes concretely, because knowing *where* a model breaks is as useful as knowing where it shines. Ask FLUX for "exactly seven red apples in a wooden bowl" and you will often get six, or eight, or seven apples of which one is suspiciously orange — counting is bounded by what the attention can track over the image tokens, and beyond a handful of instances the model loses the exact count even though it nails the *kind* of object and the *style*. Ask it to render a full paragraph of text on a poster and you will get the first line crisp and the rest degrading into plausible-looking but wrong letters — text rendering is excellent for short strings and falls off with length, because the model has learned the shapes of common words and short phrases far better than arbitrary long-form text. Ask for a complex reflection — "a chrome teapot reflecting a checkerboard floor and a window" — and the reflection will look convincing at a glance but violate the physics on close inspection, because the model has no explicit geometry, only a learned prior over what reflections tend to look like. None of these are FLUX-specific bugs; they are the current frontier's shared limits, and FLUX sits at or near the best of that frontier on each. The engineering lesson is to design around them: constrain counts in the prompt to small numbers, render long text as a post-process overlay rather than asking the model for it, and do not rely on generated reflections for anything that has to be physically correct.

## 14. Case studies and real numbers

Let us put numbers on the comparisons, with sources and honesty about precision. The figure below is the open-field comparison; the discussion fills in the caveats.

![A comparison grid placing FLUX against SDXL, SD3.5-Large, and SANA across parameters, text rendering, prompt adherence, step count, and speed](/imgs/blogs/flux-architecture-deep-dive-6.png)

**FLUX.1-dev vs SDXL (parameters and quality).** SDXL is a ~2.6B-parameter U-Net; FLUX.1-dev is a ~12B transformer plus a 4.7B T5. That is roughly a 5× jump in the denoiser alone, plus a far stronger text encoder. The visible payoff is text rendering and prompt adherence, where FLUX is not incrementally but *categorically* better — the storefront-sign example at the top of this post is representative, not cherry-picked. On raw aesthetic quality FLUX-dev is also clearly ahead for most prompts. The cost is the 5× parameters and the corresponding latency and VRAM. SDXL remains the right choice when you need speed and a clean commercial license and can live with weaker text — it is much lighter.

**FLUX.1-schnell vs SDXL-Turbo / SD-Turbo (few-step models).** Both schnell and the Turbo line are distilled few-step models, and both are good for their speed. Schnell at 4 steps produces noticeably higher quality and far better text than SDXL-Turbo at 1–4 steps, because schnell is a distilled 12B FLUX while Turbo is a distilled 2.6B SDXL. Schnell is heavier and a bit slower per step, but the quality gap is real. The honest summary: schnell is the higher-quality few-step open model; Turbo is lighter. Both are Apache-friendly for commercial use (check the specific Turbo license).

**FLUX vs SD3.5-Large.** SD3.5-Large is the 8B MM-DiT from Stability, the direct descendant of the SD3 the FLUX team originally built. The two are close in spirit — same MM-DiT-derived recipe, same flow matching, same CLIP+T5 stack — and close in quality, with FLUX.1-dev generally rated a bit ahead on text and fine detail and SD3.5 competitive on overall aesthetics, and SD3.5 carrying a more permissive community license. If license is your binding constraint, SD3.5 is often the more practical "best open" choice; if pure quality is, FLUX-dev usually edges it.

**FLUX vs SANA.** SANA (NVIDIA, covered in [the MM-DiT recipe post](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe)) is the efficiency play: a 0.6–1.6B linear-attention DiT with a 32× deep-compression autoencoder, designed to generate 1024px images fast enough to run on a laptop. SANA is *dramatically* smaller and faster than FLUX — orders of magnitude fewer parameters and far lower latency — and produces good images, but it does not match FLUX's text rendering or top-end fidelity. This is the cleanest illustration of the trilemma in the open field: FLUX picks quality, SANA picks speed and footprint, and you choose based on which constraint binds.

**Fine-tuning FLUX (LoRA).** A large fraction of FLUX's real-world impact is in the fine-tunes the community built on top of it, and the relevant numbers there are about *trainability*, not just inference. Because the FLUX transformer is a clean attention-MLP stack, LoRA fine-tuning (the technique covered in [personalization with DreamBooth, Textual Inversion, and LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora)) works well: you freeze the 12B backbone and train low-rank adapters on the attention and MLP projections, typically a few tens of millions of trainable parameters. A character or style LoRA on FLUX.1-dev can be trained on a single 24 GB consumer GPU with gradient checkpointing and 8-bit optimizers in a few hours on a few dozen images, and the resulting adapter is a small file (tens of megabytes) you load on top of the base model. The practical caveat: because FLUX-dev is guidance-distilled, you generally fine-tune *with* the distilled guidance behavior intact, and you train on dev (the open, fine-tunable tier) rather than schnell, whose aggressive timestep distillation makes it a worse base for adding new concepts. The thousands of FLUX LoRAs on community hubs are the evidence that this works — and a reason FLUX-dev, despite its license, became the default research and personalization base.

#### Worked example: choosing a tier for a real product

Say you are building a marketing-image generator. Users type a product description and a short on-image headline (so text rendering matters), and they expect a result in a few seconds. Walk the decision: you need good text rendering — that rules out SDXL and points at the FLUX/SD3.5 family. You need a *clean commercial license* — that rules out FLUX.1-dev (non-commercial) unless you license it, and points at FLUX.1-schnell (Apache-2.0) or SD3.5 (community license, check terms). You need *low latency at volume* — schnell's 4-step sampling is roughly 6× cheaper per image than dev (from the earlier worked example), which at thousands of images a day is a real cost line. So the defensible call is **FLUX.1-schnell**: it keeps FLUX-class text rendering, it is Apache-2.0, and it is fast. You accept that the hardest hero images will be a notch below dev — and you mitigate by offering a "high quality" mode that calls the FLUX.1-pro API for those. That is a complete, constraint-driven decision: schnell for the bulk, pro-API for the premium tier, and never dev in the commercial path without a license. Notice that the *architecture* did not decide this — the tier's license and step count did, which is exactly why understanding the tiers matters as much as understanding the blocks.

#### Worked example: a serving cost back-of-envelope

Suppose you serve text-to-image at scale and want to compare FLUX.1-dev against FLUX.1-schnell on cost. Assume an A100 80GB at roughly \$2/hr (spot pricing varies; this is illustrative). FLUX.1-dev at 28 steps might take ~5–8 seconds per $1024\times1024$ image on an A100 (approximate, depends on attention kernels and offload). At, say, 6 seconds, that A100 produces ~600 images/hour, so the marginal cost is about \$2 ÷ 600 ≈ \$0.0033 per image. FLUX.1-schnell at 4 steps might take ~1 second per image, ~3600 images/hour, so about \$2 ÷ 3600 ≈ \$0.00056 per image — roughly **6× cheaper per image**. The decision is then: is the quality gap between dev and schnell worth a 6× cost difference for your use case? For a thumbnail or a draft, no — use schnell. For a hero image a user will scrutinize, often yes — use dev (with a commercial license) or pro. These per-image latencies are order-of-magnitude figures to show the *method*; measure your own with a fixed seed, a warm pipeline (discard the first run), and your actual resolution and step count before committing to a number.

How would you measure quality honestly to back this up? For FID, you would generate a few thousand images from a fixed prompt set against a fixed reference distribution (and remember FID on photoreal-vs-stylized sets can mislead — [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) is the companion on why). For text rendering specifically, FID is nearly useless; you want a targeted benchmark (render-this-string accuracy) or human eval. For prompt adherence, GenEval or T2I-CompBench. The headline FLUX numbers in BFL's materials use exactly these kinds of compositional and human-preference benchmarks rather than FID alone, which is the right call for a model whose strengths are text and adherence.

## 15. When to reach for FLUX (and when not to)

A decisive recommendation, because every choice is a cost.

**Reach for FLUX.1-dev when:** you need the best open-weights quality, especially text rendering and prompt adherence; you are doing research, personal projects, or building fine-tunes/LoRAs (the community ecosystem is overwhelmingly FLUX-dev based); and you can either tolerate the non-commercial license or get a BFL commercial license. It is the default "best open image model" for quality through 2024–2025.

**Reach for FLUX.1-schnell when:** you need speed *and* a clean Apache-2.0 commercial license; you are building a product or a derivative model; or you are doing high-volume generation where the 6×-ish cost saving from 4-step sampling dominates. Accept the modest quality drop versus dev.

**Reach for FLUX.1-pro (API) when:** you want the highest FLUX quality with no infrastructure, you are fine paying per image, and you do not need the weights. Good for low-volume, high-quality needs without GPU ops.

**Do NOT reach for FLUX when:** you need to run on a small consumer GPU or a laptop with tight VRAM and cannot use heavy offload/quantization — reach for **SANA** instead, which is built for exactly that footprint. Do not use FLUX-dev in a commercial product without sorting the license — use **schnell** or **SD3.5** (more permissive license) instead. Do not reach for a 12B model when SDXL or a smaller model already clears your quality bar at a fraction of the cost — FLUX is overkill for many simple generation tasks. And do not expect SD-style CFG behavior from FLUX-dev — the distilled guidance is a different knob, and if you need true negative prompting out of the box, that is a point against dev.

**The honest one-line rule:** FLUX is the quality king of the open frontier, and you pay for that crown in parameters, VRAM, latency, and (for dev) license friction. Pick the tier that matches your binding constraint — quality (pro/dev), commercial speed (schnell), or footprint (use SANA instead) — and do not pay for quality you will not see.

## 16. Key takeaways

- **FLUX is the SD3 recipe, re-engineered and scaled.** Same MM-DiT-derived joint-attention recipe, same flow matching, same CLIP+T5 stack — but a redesigned ~12B hybrid transformer from the original SD3 authors at Black Forest Labs.
- **The hybrid is the headline.** FLUX runs ~19 double-stream blocks (separate image/text weights, one joint attention) for modality-specific feature learning, then ~38 single-stream blocks (merged tokens, shared weights) for parameter-efficient depth. Expensive specialization early, cheap depth late.
- **Guidance distillation removes the second forward pass.** FLUX.1-dev bakes CFG into the weights, so `guidance_scale` is a *conditioning input* (typical ~3.5), not a 2× extrapolation. This halves per-step compute and changes how you tune it — there is no native negative prompt.
- **Timestep distillation gives schnell 1–4 steps.** Rectified flow's near-straight trajectories make aggressive step distillation feasible; schnell is ~10× faster than dev for a modest quality cost, and it is Apache-2.0.
- **Three tiers, one backbone.** pro (closed, API, CFG, top quality), dev (open weights, guidance-distilled, non-commercial, best open quality), schnell (open, Apache, timestep-distilled, fast/commercial). Choose by your binding constraint.
- **The control suite is the same model fine-tuned.** Kontext (in-context editing), Fill (inpaint/outpaint), Redux (variation), Canny/Depth (structural control) are all the hybrid backbone with extra conditioning tokens — which is why the ecosystem grew so fast.
- **FLUX leads on text, adherence, and hands; it is big and slow.** Strengths are real and category-defining for open models; the cost is 12B+4.7B parameters, VRAM, latency, and license friction at the dev tier.
- **FLUX.2 is the recipe scaled, not a new paradigm.** Higher resolution, better text, more parameters, tighter generation+editing unification — riding the scaling curve, not reinventing it.
- **Choose against the trilemma.** FLUX picks quality; SANA picks speed and footprint; SD3.5 picks a permissive license at near-FLUX quality. Match the model to the constraint that binds you.

## 17. Further reading

- **Esser, Kulal, Blattmann, Rombach, et al. (2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"** — the SD3 paper that introduced MM-DiT and made the flow-matching case; the direct ancestor of FLUX, by the same core authors.
- **Lipman, Chen, Ben-Hamu, Nickel, Le (2023), "Flow Matching for Generative Modeling"** — the flow-matching foundation; and Liu, Gong, Liu (2022), "Flow Straight and Fast: Rectified Flow," for the rectified-flow straightening that FLUX's few-step sampling relies on.
- **Ho & Salimans (2022), "Classifier-Free Guidance"** — the guidance technique that FLUX-dev distills into its weights; essential for understanding what guidance distillation is removing.
- **Peebles & Xie (2023), "Scalable Diffusion Models with Transformers (DiT)"** — the transformer-backbone-for-diffusion paper underneath the whole MM-DiT/FLUX line, with the AdaLN-Zero conditioning FLUX inherits.
- **Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"** — the RoPE that FLUX uses for image and text tokens.
- **Black Forest Labs FLUX.1 and FLUX.2 announcements and model cards** (Hugging Face: `black-forest-labs/FLUX.1-dev`, `FLUX.1-schnell`) — the source of truth for tier specifications, licenses, and the FLUX.2 numbers; consult these for exact, current figures.
- **🤗 `diffusers` FLUX documentation** — `FluxPipeline`, `FluxTransformer2DModel`, `FlowMatchEulerDiscreteScheduler`, and the Fill/Control/Kontext pipelines, with current API details and memory-optimization recipes.
- **Within this series:** [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe), [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance), [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation), [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing). Forward to [the 2026 image model landscape](/blog/machine-learning/image-generation/the-2026-image-model-landscape) for where FLUX sits against the closed frontier, and to the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) for wiring FLUX into a real serving pipeline.
