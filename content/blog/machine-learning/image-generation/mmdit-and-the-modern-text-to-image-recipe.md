---
title: "MM-DiT and the Modern Text-to-Image Recipe: Inside SD3, FLUX, and SANA"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The architecture recipe behind the 2024-2026 open frontier: MM-DiT's joint text-image attention, QK-norm and flow matching, the CLIP+T5/LLM encoder stack, FLUX's double-then-single-stream design, and SANA's 32x deep-compression autoencoder, with runnable diffusers code and measured numbers."
tags:
  [
    "image-generation",
    "diffusion-models",
    "mm-dit",
    "stable-diffusion-3",
    "flux",
    "sana",
    "flow-matching",
    "text-to-image",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-1.png"
---

Type "a vintage diner at night, a neon sign that reads OPEN 24 HOURS, rain on the asphalt reflecting the letters" into a 2021-era Stable Diffusion model and you will get a moody diner, rain, neon — and a sign that says something like "0PEN 24 H0URZ" or "OEPN 24 OURS". The model nails the *vibe* and butchers the *text*, because the text was the one thing the architecture was structurally bad at. Type the same prompt into FLUX.1 or Stable Diffusion 3.5 in 2025 and the sign reads, crisply, OPEN 24 HOURS, the rain reflects the right letters, and the composition obeys every clause of the prompt. The image looks like the model actually *read* the sentence rather than pattern-matching a few keywords. That jump — from "draws the vibe, mangles the details" to "renders legible text and follows three-clause prompts" — is not one trick. It is a *recipe*: a specific stack of architectural choices that the open frontier converged on between 2024 and 2026, and that recipe is what this post is about.

![A dataflow figure showing text tokens and image patch tokens each going through separate per-modality projections before being concatenated into one shared joint self-attention, then split back into two per-stream MLP paths](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-1.png)

By the end of this post you will be able to explain, and partly reimplement, the four ingredients that define a modern text-to-image model: (1) **MM-DiT**, the Multimodal Diffusion Transformer from Stable Diffusion 3 (Esser et al., 2024), where text tokens and image-latent patch tokens flow through a *joint* attention with *separate per-modality weights* — two streams that read each other; (2) the **text-encoder stack** — why SD3 carries CLIP-L plus CLIP-G plus a T5-XXL language model, what each buys, and what breaks when you drop T5 to save memory; (3) **FLUX.1** (Black Forest Labs), the 12-billion-parameter model whose blocks go double-stream then single-stream, with rotary position embeddings and *guidance distillation* so the schnell tier samples in 1-4 steps; and (4) **SANA** (Xie et al., 2024), which throws out the usual 8x latent for a **32x deep-compression autoencoder**, swaps quadratic attention for **linear attention**, and uses a decoder-only LLM as its text encoder — enough to generate 1024px images on a laptop GPU. We will also glance at the *other* branch of the frontier, the native-autoregressive challengers (HunyuanImage-3.0, GPT-Image), and tie it all together: modern text-to-image = flow matching + (MM-)DiT + a strong text encoder + aggressive latent compression.

This is the capstone of the architecture track. It builds directly on three siblings: [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit) gave us the transformer backbone and AdaLN-Zero conditioning; [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) gave us the VAE latent and SDXL's dual encoders; and [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) gave us the straight-line velocity objective that all three frontier models train with. If those words feel shaky, skim them first. And recall our running frame: the **diffusion stack** (data → VAE latent → forward noising → denoiser net → ODE/SDE sampler → guidance → image) and the **generative trilemma** (quality × diversity × speed). The recipe in this post is a coordinated attack on all three faces of the trilemma at once, which is exactly why it took the whole field a few years to assemble.

## Where the old recipe ran out of road

Let me set the stage by naming what the *previous* recipe was, because the modern one is best understood as a list of fixes to its specific failures. The 2022-2023 recipe — SD 1.5, SD 2, SDXL — was: a **U-Net** denoiser, operating in an **8x VAE latent**, conditioned on a **frozen CLIP text encoder** via **cross-attention**, trained with the **ε-prediction** DDPM loss, sampled with a DDIM/DPM-Solver ODE solver, steered with classifier-free guidance. It worked, spectacularly, and most of the art you have seen from 2022-2023 came out of it. But it had four structural weaknesses, and each one maps to a fix in the modern recipe.

First, **cross-attention is one-way**. In the U-Net, the image features (the queries) attend *to* the text tokens (the keys and values), but the text never attends back to the image. The text encoder ran once, up front, produced a fixed sequence of embeddings, and those embeddings sat frozen for the entire denoising loop. The image could *read* the text, but the text representation could never adapt to what the image was becoming. That asymmetry is fine for "a photo of a cat" but it is exactly wrong for "the word OPEN in neon," because rendering legible text needs the text and image representations to negotiate, glyph by glyph, throughout denoising.

Second, **CLIP is a weak language model**. CLIP's text encoder is a small transformer trained on a contrastive image-text objective with a 77-token context. It is excellent at *gist* — "is this caption about a cat or a dog" — and poor at *syntax*, *counting*, *negation*, and *spelling*. Ask CLIP to distinguish "a red cube on a blue sphere" from "a blue cube on a red sphere" and its embeddings barely move. The denoiser inherits that blindness. So compositional prompts and text rendering failed not in the U-Net but upstream, in an encoder that never understood the sentence.

Third, **the U-Net is an awkward fit for scaling**. Its hand-designed multi-resolution structure — ResBlocks, downsampling, upsampling, attention only at the lower resolutions — does not scale as cleanly as a plain transformer, and it bakes in a 2D-grid inductive bias that makes it hard to mix in non-image tokens (like text) as first-class citizens. The [DiT paper](/blog/machine-learning/image-generation/diffusion-transformers-dit) showed that a pure transformer on patch tokens scales better (lower FID at the same compute, clean power-law scaling), which is why every frontier model replaced the U-Net with a transformer.

Fourth, **the 8x latent is still a lot of tokens**. A 1024x1024 image in an 8x VAE latent is a 128x128x4 (or x16) grid. Patchify that and you get 16,384 image tokens (at patch size 1; SD3 uses patch 2 to bring it to 4096). Attention is quadratic in token count, so that is the single biggest cost in the network. The old recipe just paid it. SANA's whole thesis is that you do not have to.

There is also a fifth, subtler limitation that the modern recipe quietly fixed: **how conditioning enters the network**. In the SDXL U-Net, the timestep and the pooled text vector entered through FiLM/AdaGN — a learned scale-and-shift applied to the normalized activations — while the per-token text entered through cross-attention. DiT sharpened this into **AdaLN-Zero**: the conditioning vector predicts the LayerNorm scale, shift, *and* a residual gate that is initialized to zero, so every block starts as an identity function and the network learns to "turn on" its blocks gradually. That zero-init is a real stabilizer — it lets you train very deep transformers without the early-training instability of all blocks firing at once. MM-DiT keeps AdaLN-Zero and applies it *per modality* (separate modulation for the text and image streams, visible as `img_mod` and `txt_mod` in the block code above). So the modern recipe did not just change *what* attends to what; it changed *how the timestep and prompt-gist steer the network*, and AdaLN-Zero is part of why these models train stably at depth. That is the conditioning machinery the [DiT post](/blog/machine-learning/image-generation/diffusion-transformers-dit) derives in full.

So the modern recipe is: replace cross-attention with **joint attention** (fix one); replace CLIP-only with a **CLIP+T5/LLM stack** (fix two); replace the U-Net with a **DiT/MM-DiT** transformer and train it with **flow matching** (fix three); and, in SANA's branch, replace the 8x latent with a **32x latent + linear attention** (fix four). Let me take them one at a time, with the math and the code.

It is worth dwelling for a moment on *why these four fixes arrived together* rather than one at a time, because it tells you something about how the field actually moves. They are not independent — they enable each other. Flow matching only became worth adopting once the backbone was a transformer that could scale cleanly to exploit the simpler objective; the joint-attention design only became affordable once you had a transformer where adding ~160 text tokens to a 4096-token image sequence was nearly free; and the strong T5/LLM encoder only paid off once joint attention gave those rich text tokens a two-way channel into the image. The U-Net could not have used a T5 encoder nearly as well, because cross-attention would have throttled all that linguistic detail through a one-way pipe. So the recipe is a *system*: each ingredient is mediocre alone and excellent in combination. That is the recurring shape of progress in generative modeling — not a single breakthrough but a co-adapted bundle of choices that only works as a set. Keep that in mind as we dissect the parts; the parts are real, but the magic is the assembly.

## Joint attention: the heart of MM-DiT

Here is the single most important idea in the post. In the U-Net's cross-attention, the image and text live in separate worlds that touch only through a one-way query. In **MM-DiT**, they live in the *same* sequence.

![A before-and-after comparison contrasting one-way cross-attention where the image reads frozen text against joint bidirectional attention where both streams update each other inside a single concatenated attention](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-2.png)

Concretely: you take your text tokens (the output of the encoder stack — call it a sequence of $L_\text{txt}$ vectors) and your image-latent patch tokens ($L_\text{img}$ vectors), and you **concatenate them into one sequence** of length $L = L_\text{txt} + L_\text{img}$. Then you run ordinary self-attention over the whole sequence. Every token — text or image — produces a query, a key, and a value, and attends to *every other token*, regardless of modality. The image patches attend to the text (as in cross-attention), but now the text tokens *also* attend to the image patches, and to each other in the context of the image. It is bidirectional, full, and symmetric in *connectivity*.

The clever part — the "MM" in MM-DiT — is that the connectivity is shared but the **weights are not**. Each modality gets its own projection matrices. Text tokens are projected to queries/keys/values by $W_Q^\text{txt}, W_K^\text{txt}, W_V^\text{txt}$; image tokens by a *different* set $W_Q^\text{img}, W_K^\text{img}, W_V^\text{img}$. Same for the output projection, the MLP, and the AdaLN modulation. So MM-DiT is *two transformers that share one attention operation*. The text stream and the image stream each have their own parameters, but the attention step lets them mix freely. This is why people draw it as "two streams that attend to each other."

Why does this beat cross-attention? Two reasons, one representational and one empirical. Representationally, joint attention lets the text representation become *image-conditioned*. The token for "OPEN" can refine itself based on where the sign is, how big it is, what font the rest of the image implies — across every layer and every denoising step. Cross-attention froze the text after the encoder; joint attention keeps it live. Empirically, the SD3 paper ablates exactly this and reports that the MM-DiT (separate weights, joint attention) beats both a single-stream design (shared weights for both modalities, which under-fits because text and image statistics differ) and the cross-attention DiT baseline, on validation loss, on CLIP-score, and on GenEval (the compositional benchmark). Joint attention is *the* reason SD3 and FLUX render text and bind attributes so much better than SDXL.

### The token budget, quantified

Let me make "a lot of tokens" precise, because the token count drives everything downstream — memory, FLOPs, and the architecture choices in FLUX and SANA.

Take SD3 at 1024x1024. The VAE is 8x, so the latent is 128x128. SD3 uses a patch size of 2, so patchify groups 2x2 latent cells into one token: that is $(128/2)^2 = 64^2 = 4096$ image tokens. (At the 1024 training resolution, with the 16-channel VAE; the exact count depends on resolution and patch size.) The text side: SD3 concatenates a pooled CLIP vector and the per-token T5 sequence, padded to 256 tokens, plus the CLIP per-token sequences — on the order of $L_\text{txt} \approx 154$ to $256$ tokens depending on configuration. So the joint sequence is roughly $4096 + 160 \approx 4250$ tokens, overwhelmingly image. Self-attention cost scales as $O(L^2 d)$, so the image tokens dominate the compute by a factor of $(4096/160)^2 \approx 650$. The text tokens are nearly free; the image tokens are the whole bill. Hold that thought — it is exactly the lever SANA pulls.

Let me push the FLOP accounting one level deeper, because it is the quantitative backbone of the rest of the post. For a transformer with hidden width $d$, the cost of one block is dominated by two terms: the attention, which is $O(L^2 d)$ for the score matrix plus $O(L^2 d)$ for the value aggregation, and the MLP, which is $O(L d^2)$ (with the standard 4x expansion, $\approx 8 L d^2$). The crossover — where attention overtakes the MLP — happens when $L \approx d$. For SD3-medium, $d = 1536$ and $L \approx 4250$, so $L > d$: attention is the larger term, and it grows *quadratically* while everything else grows linearly in $L$. This is why, at high resolution, reducing the token count is worth far more than reducing the width or the depth. Double the resolution (4x the tokens) and attention cost goes up 16x; halve the tokens (deeper compression) and it drops 4x. The asymmetry is the entire argument for SANA, and it is why "just train a bigger denoiser" is the wrong lever once you are token-bound.

There is one more wrinkle the modern recipe exploits. Because the text tokens are so few (~160 of ~4250), the *extra* cost of joint attention over plain image self-attention is tiny: the attention matrix grows from $4096^2$ to $4250^2$, about 8% more entries, for a large gain in prompt adherence. Joint attention is, in FLOP terms, nearly free — you are letting ~160 text tokens into a sequence already dominated by 4096 image tokens. That cheapness is *why* the field could afford to switch from cross-attention to joint attention without blowing up the compute budget. The expensive thing was never the text; it was always the image grid.

### QK-normalization: the stability fix that made it trainable

One small but load-bearing detail. When you scale MM-DiT up and train in `bfloat16` mixed precision, the attention **logits** — the dot products $q \cdot k$ before the softmax — can grow without bound during training. If a few entries of $q$ and $k$ drift large and aligned, $q \cdot k$ blows up, the softmax saturates to a one-hot, the gradient through it vanishes, and (worse) the activation magnitudes in later layers explode into `NaN`. The SD3 team hit exactly this: past a certain scale, training would diverge with the attention-entropy collapsing.

The fix is **QK-normalization**: apply RMSNorm (or LayerNorm) to the queries and keys *per head* before the dot product, so

$$
\text{attn}(Q,K,V) = \text{softmax}\!\left(\frac{\text{RMSNorm}(Q)\,\text{RMSNorm}(K)^\top}{\sqrt{d_k}}\right) V .
$$

Because RMSNorm fixes the norm of each query and key vector to a learned scale, the dot product $\hat q \cdot \hat k$ is bounded by the (now controlled) magnitudes times $\cos\theta$, so the logits cannot run away. The softmax stays in its sensitive regime, gradients keep flowing, and the model trains stably at 8B parameters in `bf16`. QK-norm costs almost nothing — two RMSNorms per attention — and it is the difference between a model that trains and one that diverges at scale. It originated in the LLM literature (Dehghani et al.'s ViT-22B, and others) and SD3 brought it into image diffusion. Every frontier MM-DiT uses it.

### A joint-attention block in PyTorch

Here is a stripped-down but faithful sketch of one MM-DiT joint-attention block: separate projections per modality, a single concatenated attention, QK-norm, and AdaLN modulation from the timestep/pooled-text embedding. This is not the production code, but it is the right shape.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # normalize over the last (head) dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class MMDiTBlock(nn.Module):
    """One MM-DiT block: separate weights per modality, one joint attention."""
    def __init__(self, dim, n_heads, cond_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # SEPARATE projections per modality (this is the "MM")
        self.img_qkv = nn.Linear(dim, 3 * dim)
        self.txt_qkv = nn.Linear(dim, 3 * dim)
        self.img_proj = nn.Linear(dim, dim)
        self.txt_proj = nn.Linear(dim, dim)
        self.img_mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.txt_mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        # QK-norm, per modality, applied per head
        self.img_q_norm = RMSNorm(self.head_dim)
        self.img_k_norm = RMSNorm(self.head_dim)
        self.txt_q_norm = RMSNorm(self.head_dim)
        self.txt_k_norm = RMSNorm(self.head_dim)

        # AdaLN-Zero modulation from the conditioning vector (time + pooled text)
        self.img_mod = nn.Linear(cond_dim, 6 * dim)
        self.txt_mod = nn.Linear(cond_dim, 6 * dim)
        self.img_ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_ln = nn.LayerNorm(dim, elementwise_affine=False)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # B, H, L, d

    def forward(self, img, txt, cond):
        B, Li, _ = img.shape
        Lt = txt.shape[1]

        # AdaLN: shift/scale/gate for attention and MLP, from the conditioning vector
        i_sa, i_ss, i_g1, i_sm, i_ms, i_g2 = self.img_mod(cond).chunk(6, dim=-1)
        t_sa, t_ss, t_g1, t_sm, t_ms, t_g2 = self.txt_mod(cond).chunk(6, dim=-1)

        img_n = self.img_ln(img) * (1 + i_ss[:, None]) + i_sa[:, None]
        txt_n = self.txt_ln(txt) * (1 + t_ss[:, None]) + t_sa[:, None]

        # per-modality QKV
        iq, ik, iv = self.img_qkv(img_n).chunk(3, dim=-1)
        tq, tk, tv = self.txt_qkv(txt_n).chunk(3, dim=-1)
        iq, ik, iv = map(self._split_heads, (iq, ik, iv))
        tq, tk, tv = map(self._split_heads, (tq, tk, tv))

        # QK-normalization (the stability fix)
        iq, ik = self.img_q_norm(iq), self.img_k_norm(ik)
        tq, tk = self.txt_q_norm(tq), self.txt_k_norm(tk)

        # CONCATENATE both modalities along the sequence, then ONE joint attention
        q = torch.cat([tq, iq], dim=2)   # B, H, Lt+Li, d
        k = torch.cat([tk, ik], dim=2)
        v = torch.cat([tv, iv], dim=2)
        out = F.scaled_dot_product_attention(q, k, v)  # FlashAttention under the hood
        out = out.transpose(1, 2).reshape(B, Lt + Li, -1)

        # split back to the two streams, separate output projections + gates
        t_out, i_out = out[:, :Lt], out[:, Lt:]
        img = img + i_g1[:, None] * self.img_proj(i_out)
        txt = txt + t_g1[:, None] * self.txt_proj(t_out)

        # per-modality MLP with its own modulation
        img = img + i_g2[:, None] * self.img_mlp(self.img_ln(img) * (1 + i_ms[:, None]) + i_sm[:, None])
        txt = txt + t_g2[:, None] * self.txt_mlp(self.txt_ln(txt) * (1 + t_ms[:, None]) + t_sm[:, None])
        return img, txt
```

Read the `forward` top to bottom and you have the whole idea: two streams, separate projections, QK-norm, **one** `scaled_dot_product_attention` over the concatenated sequence, then split and recombine. The conditioning vector `cond` (timestep embedding plus the pooled CLIP text vector) drives AdaLN modulation per modality — that is how the timestep and the global prompt gist steer the block, exactly as in DiT but doubled. `F.scaled_dot_product_attention` dispatches to FlashAttention, so the quadratic attention runs memory-efficiently. The actual `SD3Transformer2DModel` and `FluxTransformer2DModel` in `diffusers` are this block, parameterized and stacked 24 (SD3) to 57 (FLUX) deep.

#### Worked example: counting the parameters that joint attention costs

Separate weights per modality roughly *double* the per-block parameter count of the projections and MLPs relative to a shared-weight single-stream block. Is that worth it? Take SD3-medium's hidden size $d = 1536$, with 24 MM-DiT blocks. The attention QKV projections are $3 d^2$ per modality, the output projection $d^2$, and the MLP $2 \cdot 4 d^2 = 8 d^2$, so roughly $12 d^2$ per modality per block, doubled to $\approx 24 d^2$ for both streams: $24 \times 1536^2 \approx 5.7 \times 10^7$ parameters per block, times 24 blocks $\approx 1.36$B in the blocks, with embeddings and the final layer bringing SD3-medium to about 2B total. The ablation in the SD3 paper says the separate-weight design buys roughly a 0.1-0.2 CLIP-score improvement and a clear GenEval gain over shared weights at matched compute — a real, measurable jump in prompt adherence for the extra parameters. For text rendering specifically the gap is larger and obvious to the eye. So yes: the doubling pays for itself.

## The text-encoder stack: why three encoders

Now the upstream half of fix two. SD3 does not use one text encoder; it uses **three**, in parallel: **CLIP-L** (the ViT-L/14 text tower, ~123M params), **CLIP-G** (the bigGCLIP / OpenCLIP-bigG text tower, ~695M), and **T5-XXL** (the 4.7B encoder of Google's T5 language model). Why carry 5.5B of text encoders for a 2B denoiser?

![A layered stack figure showing the text-encoder stack feeding conditioning tokens into the MM-DiT backbone, then a flow-matching sampler producing a clean latent that the VAE decodes once to a 1024px image](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-3.png)

Each encoder contributes something different. The two CLIP encoders give a **pooled** global vector (a single embedding summarizing the whole prompt — great for the AdaLN modulation that sets the overall "vibe") and a short per-token sequence aligned to the visual world by contrastive pretraining. The T5-XXL gives a long, **syntax-aware** per-token sequence: T5 was trained as an actual language model on text-to-text tasks, so it understands word order, clauses, negation, counting, and — critically — *spelling*. When your prompt is "a sign reading OPEN 24 HOURS," the T5 tokens carry the character-level and syntactic structure that CLIP throws away. The MM-DiT attends to all of them jointly.

The mechanics: SD3 takes the pooled CLIP-L and CLIP-G vectors, concatenates them, and adds them to the timestep embedding to form the AdaLN conditioning vector (the "global" path). Separately it concatenates the per-token CLIP-L, CLIP-G, and T5 sequences (projecting them to the model width and zero-padding the CLIP sequences to match T5's length) into the text-token stream that enters joint attention (the "fine-grained" path). So the global gist comes from CLIP pooling; the fine-grained, spell-it-out detail comes from T5. That division of labor is the whole point.

Here is the encoder wiring made concrete — the two paths, pooled and per-token, exactly as SD3 assembles them. This is the shape of `StableDiffusion3Pipeline.encode_prompt`, simplified so you can see the structure.

```python
import torch

def encode_prompt(prompt, clip_l, clip_g, t5, tok_l, tok_g, tok_t5, device):
    """Build SD3's two conditioning paths: pooled (global) and sequence (fine-grained)."""
    # --- CLIP-L and CLIP-G: take BOTH the pooled vector and the per-token states ---
    il = tok_l(prompt, padding="max_length", max_length=77, truncation=True,
               return_tensors="pt").to(device)
    out_l = clip_l(**il, output_hidden_states=True)
    pooled_l = out_l.pooler_output                     # (B, 768)   global
    seq_l = out_l.hidden_states[-2]                    # (B, 77, 768) per-token

    ig = tok_g(prompt, padding="max_length", max_length=77, truncation=True,
               return_tensors="pt").to(device)
    out_g = clip_g(**ig, output_hidden_states=True)
    pooled_g = out_g.text_embeds                       # (B, 1280)  global
    seq_g = out_g.hidden_states[-2]                    # (B, 77, 1280) per-token

    # --- T5-XXL: per-token only, long context, the syntax/spelling path ---
    it = tok_t5(prompt, padding="max_length", max_length=256, truncation=True,
                return_tensors="pt").to(device)
    seq_t5 = t5(**it).last_hidden_state                # (B, 256, 4096) per-token

    # POOLED path -> drives AdaLN modulation (the "global vibe")
    pooled = torch.cat([pooled_l, pooled_g], dim=-1)   # (B, 2048)

    # SEQUENCE path -> enters joint attention as the text stream.
    # CLIP seqs are projected to T5 width and zero-padded, then concatenated.
    clip_seq = torch.cat([seq_l, seq_g], dim=-1)       # (B, 77, 2048)
    clip_seq = torch.nn.functional.pad(clip_seq, (0, 4096 - 2048))  # -> width 4096
    seq = torch.cat([clip_seq, seq_t5], dim=1)         # (B, 77+256, 4096)
    return pooled, seq
```

Read the return values: `pooled` (the concatenated CLIP pooled vectors) feeds AdaLN, and `seq` (CLIP per-token padded, plus the full T5 sequence) is the text stream that enters joint attention. If you set `t5=None` and skip the T5 block, `seq` is just the 77 CLIP tokens — the model still runs, but it has lost the 256 syntax-and-spelling-rich T5 tokens, which is exactly why text rendering degrades. The wiring *is* the trade-off.

### Dropping T5 to save memory — and what it costs

T5-XXL is 4.7B parameters and, in `fp16`, about 9.5 GB of weights — often *larger than the denoiser it conditions*. On a 12 GB or 16 GB consumer GPU that is brutal. So both SD3 and FLUX support **dropping T5 at inference**: run with CLIP only, leave the T5 slot empty. The SD3 report measures this honestly. Without T5: overall sample quality and aesthetics are *barely affected* on typical prompts, GenEval drops a little, and **text-rendering quality collapses** — the legible-sign superpower is mostly a T5 effect. So the rule is: if you do not need text in the image and your prompts are simple, drop T5 and reclaim ~9 GB; if you need legible typography or complex multi-clause prompts, keep it. This is a clean, actionable trade-off, and it is *why* you see "with/without T5" toggles in the tooling.

```python
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

# Option A: load SD3.5 normally (needs ~18 GB+ for all three encoders + transformer in fp16)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # stream modules to GPU as needed; fits ~12 GB

# Option B: quantize the heavy T5 encoder to 4-bit instead of dropping it
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
t5_4bit = T5EncoderModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    subfolder="text_encoder_3",
    quantization_config=quant,
    torch_dtype=torch.bfloat16,
)
pipe_q = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    text_encoder_3=t5_4bit,
    torch_dtype=torch.bfloat16,
)
pipe_q.enable_model_cpu_offload()

image = pipe_q(
    prompt='a vintage diner at night, a neon sign reading "OPEN 24 HOURS", rain on the asphalt',
    num_inference_steps=28,
    guidance_scale=4.5,        # flow-matching models like a LOWER CFG than SDXL
    height=1024, width=1024,
).images[0]
image.save("diner.png")
```

Two things to notice. `guidance_scale=4.5` — flow-matching MM-DiTs want a *lower* CFG than the 7-12 you used with SDXL; the straighter probability path is already sharp, so over-guiding washes it out fast. And quantizing T5 to 4-bit (Option B) keeps the text-rendering ability while cutting the encoder from ~9.5 GB to ~3 GB, which is usually the better move than dropping it outright. We go deeper on this in the [quantization and efficient-inference post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference).

## Flow matching: the training objective underneath all three

I will keep this short because [the flow-matching post](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) does the full derivation, but you cannot understand why SD3/FLUX/SANA sample in 4-28 steps without it. The old recipe trained the network to predict noise ($\epsilon$-prediction) under the DDPM forward process, whose probability path is *curved* — the ODE trajectory from noise to data bends, so a coarse-step solver overshoots and you need many steps.

Flow matching trains the network to predict a **velocity** along a *straight* path. Define the interpolation between a data latent $x_1$ and a noise sample $x_0 \sim \mathcal{N}(0, I)$ as the straight line

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad t \in [0, 1],
$$

so the target velocity is simply the constant $\frac{dx_t}{dt} = x_1 - x_0$. The network $v_\theta(x_t, t, c)$ is trained to regress that velocity, conditioned on the text $c$, with the conditional-flow-matching loss

$$
\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\,x_0,\,x_1}\big[\,\lVert v_\theta(x_t, t, c) - (x_1 - x_0)\rVert^2\,\big].
$$

Because the target path is a straight line, the learned ODE $\dot x = v_\theta$ is much closer to straight than the DDPM one, so an Euler solver with few steps tracks it well. SD3 uses a *logit-normal* timestep sampling (sample more often near the middle of the path, where the velocity field is hardest to learn) which sharpens this further. The practical payoff: SD3 and FLUX-dev produce strong samples in ~20-28 steps where SD1.5 needed 50, and FLUX-schnell — after *guidance distillation* — does it in 1-4. The straight path is the reason the whole recipe is fast.

#### Worked example: why 28 steps not 50

Step count is set by how curved the ODE is. With $\epsilon$-prediction on the DDPM cosine schedule, the velocity field changes direction along the trajectory, so a fixed-step Euler solver accumulates truncation error proportional to the path curvature times the step size; you need ~50 steps with DDIM (or ~20 with a 2nd-order DPM-Solver that corrects for the curvature). With flow matching, the conditional path is *exactly* straight and the marginal (the one the network actually learns) is *nearly* straight, so first-order Euler at ~28 steps already lands inside the data manifold — and a few-step distilled model can cut it to single digits. The measured FID-vs-steps curve for SD3 flattens by ~25-28 steps; pushing to 50 buys almost nothing. That is the step↔quality Pareto point: **set SD3/FLUX-dev to ~28 steps and stop.**

There is a subtlety worth making explicit, because it trips people coming from DDPM. The conditional path — the line between *one* specific noise sample $x_0$ and *one* specific data point $x_1$ — is perfectly straight by construction. But the network never sees that pairing; it only sees the noisy point $x_t$ and must predict the *expected* velocity averaged over all data points that could have produced $x_t$. That marginal velocity field $u(x_t, t) = \mathbb{E}[x_1 - x_0 \mid x_t]$ is what the network regresses, and it is *not* perfectly straight — different data points pull $x_t$ in different directions, so the average curves. Flow matching's claim is the weaker, true one: the marginal path is *much straighter* than the DDPM marginal, straight enough that low-order solvers track it. **Rectified flow** (Liu et al.) makes it straighter still by a "reflow" procedure — generate pairs $(x_0, x_1)$ with the current model, then retrain on those straightened pairs — which is part of why FLUX-schnell can go so low on steps. The takeaway: straightness is a spectrum, and the whole frontier is a race to straighten the marginal path so the solver can take bigger steps.

Here is what a flow-matching sampling loop actually looks like, stripped to its essence. It is an Euler integration of $\dot x = v_\theta$ from $t=0$ (pure noise) to $t=1$ (data), with the timestep grid shifted toward the noisy end (SD3/FLUX use a *resolution-dependent timestep shift* — higher resolution needs more steps spent at high noise because there is more signal to place). Notice there is no `beta` schedule, no `alpha_cumprod`, none of the DDPM bookkeeping — flow matching is dramatically simpler to implement.

```python
import torch

@torch.no_grad()
def flow_match_sample(transformer, text_embeds, pooled, shape, num_steps=28,
                      shift=3.0, guidance_scale=4.5, device="cuda", dtype=torch.bfloat16):
    """Minimal flow-matching (rectified-flow) Euler sampler for an MM-DiT."""
    # start from pure noise at t=0
    x = torch.randn(shape, device=device, dtype=dtype)

    # uniform t grid in [0,1], then apply the resolution-dependent shift toward high noise
    t = torch.linspace(0, 1, num_steps + 1, device=device)
    t = shift * t / (1 + (shift - 1) * t)          # SD3/FLUX timestep shift
    dt = t[1:] - t[:-1]                              # per-step size (not uniform)

    for i in range(num_steps):
        t_cur = t[i].expand(shape[0])
        # one network call predicts the velocity at (x, t); for classic CFG you'd
        # also call the unconditional branch and extrapolate. Guidance-distilled
        # models (FLUX-dev) take guidance as an INPUT and need only this one call.
        v = transformer(x, timestep=t_cur, encoder_hidden_states=text_embeds,
                        pooled_projections=pooled, guidance=guidance_scale).sample
        x = x + dt[i] * v                            # Euler step along the velocity
    return x                                          # clean latent z0, hand to the VAE decoder
```

That loop is the entire sampler. The only model-specific magic is the timestep `shift` and whether guidance is a separate pass (classic CFG) or an input (distilled). Everything else is `x += dt * v`. Compare that to the DDPM posterior-mean-and-variance update and you see why the field happily switched objectives: flow matching is both straighter (fewer steps) *and* simpler (less code, fewer ways to get the schedule wrong).

The *training* step is just as clean. Sample a data latent, sample noise, sample a timestep from the logit-normal distribution, linearly interpolate, and regress the velocity. That is the whole loss — no noise-schedule lookup, no SNR weighting table:

```python
import torch

def flow_match_loss(transformer, z1, text_embeds, pooled):
    """One MM-DiT flow-matching training step. z1 = clean VAE latent of a real image."""
    B = z1.shape[0]
    z0 = torch.randn_like(z1)                          # noise endpoint

    # logit-normal timestep sampling: more mass in the middle of the path
    t = torch.sigmoid(torch.randn(B, device=z1.device))   # ~ logit-normal(0,1) in [0,1]
    t_ = t.view(B, 1, 1, 1)

    zt = (1 - t_) * z0 + t_ * z1                       # point on the straight path
    target = z1 - z0                                   # constant target velocity

    v = transformer(zt, timestep=t, encoder_hidden_states=text_embeds,
                    pooled_projections=pooled).sample
    return torch.nn.functional.mse_loss(v, target)     # L_CFM
```

Five meaningful lines. The logit-normal `t` is the one non-obvious choice: sampling timesteps with more density in the middle of the path concentrates training where the marginal velocity field is hardest (near the endpoints it is nearly deterministic). Drop that and you can still train, but convergence is slower and the mid-path region — where most of the perceptual content is decided — gets under-trained. This is the flow-matching analogue of DDPM's min-SNR weighting, and it is one of the quiet choices that decides final quality.

### Why the math says joint attention should help binding

We asserted that joint attention improves attribute binding ("a red cube on a blue sphere"). Let me make the mechanism a little more rigorous, because it is not hand-waving. Attribute binding is, fundamentally, a problem of *routing the right adjective to the right noun and the right region*. In cross-attention, the routing is one-way: image query $q_i$ (a patch) attends over text keys, picks up "red," "cube," "blue," "sphere," and must sort out, locally and without feedback, which words apply to it. The text tokens have no way to specialize — the token "red" is the same vector whether it is being read by a cube-patch or a sphere-patch.

In joint attention the text token "red" produces a query too, and it attends over the *image* patches. So across layers it can localize: "red" finds the cube-region, strengthens its key there, and the cube-patches in turn find a "red" that has already committed to them. The fixed point of this two-way negotiation is a binding. Formally, the joint attention matrix has four blocks — image→image, image→text, text→image, text→text — and the U-Net's cross-attention keeps only the image→text block (with image→image self-attention in a separate layer). MM-DiT keeps all four. The text→image and text→text blocks are exactly the new capacity, and they are precisely the capacity needed to let attributes and nouns *co-localize* rather than be sorted out one-sidedly. The SD3 GenEval numbers — where binding and counting are the hard sub-scores — are the empirical shadow of those two extra attention blocks.

## FLUX.1: double-stream, then single-stream

FLUX.1, from Black Forest Labs (several of whom authored the original Latent Diffusion and SD3 work), is the 12B-parameter open-weight model that, as of 2024-2025, set the bar for open text-to-image. Architecturally it is an MM-DiT with one important twist and several refinements.

![A dataflow figure showing FLUX running double-stream blocks with separate text and image weights, then merging both streams into a single concatenated sequence processed by shared-weight single-stream blocks to produce the image latent](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-4.png)

**The double-then-single twist.** FLUX's first ~19 blocks are MM-DiT-style **double-stream** blocks (separate weights per modality, exactly as above). Then it concatenates the two streams into a *single* sequence and runs ~38 **single-stream** blocks, where text and image tokens share one set of weights and flow through a unified attention+MLP. The intuition: the early layers need modality-specific processing (text statistics and image statistics differ a lot), but once the representations have been aligned by the double-stream blocks, the later layers can treat the tokens uniformly and share parameters. Single-stream blocks are cheaper per parameter (one MLP path, not two) and let FLUX spend its 12B budget on *depth*. The result is 19 double + 38 single = 57 blocks, far deeper than SD3-medium's 24.

**Rotary position embeddings (RoPE).** Instead of learned or sinusoidal absolute positions, FLUX uses RoPE on the image tokens — rotating the query/key vectors by an angle proportional to their 2D position. RoPE is what LLMs use, it extrapolates to unseen resolutions more gracefully, and it injects relative-position information directly into attention. For an image model that wants to generate at multiple aspect ratios and resolutions, relative positions are the right inductive bias. The 2D twist matters: an image patch has a row and a column, so FLUX applies RoPE with the angle derived from the 2D grid position (splitting the head dimension between the two axes), so the attention is aware that two patches are "three rows apart and one column apart," not just "some distance apart in a flattened sequence." Absolute learned embeddings, by contrast, are tied to the training resolution — generate at a new aspect ratio and they are out of distribution. RoPE's relative encoding degrades gracefully instead. This is a direct import from the LLM playbook (long-context LLMs lean on RoPE for the same extrapolation reason), and it is one more place where the image and text frontiers have converged on the same primitive.

**Guidance distillation — the schnell trick.** This is the headline. Classifier-free guidance, recall, runs the network *twice* per step (conditional and unconditional) and extrapolates — so it doubles inference cost. FLUX **distills the guidance into the weights**: a student network is trained to directly produce the *already-guided* velocity for a given guidance scale, so at inference you run the network *once* per step with no separate unconditional pass. FLUX-dev is guidance-distilled (you pass a `guidance_scale` that the model consumes as an input, not as a two-pass extrapolation). FLUX-schnell goes further with *step* distillation (an adversarial/consistency-style objective) so it samples in **1-4 steps**. The three tiers:

- **FLUX.1-pro** — the best quality, API-only, closed weights.
- **FLUX.1-dev** — open weights (non-commercial license), guidance-distilled, ~20-50 steps, the quality workhorse.
- **FLUX.1-schnell** — open weights (Apache-2.0), guidance + step distilled, **1-4 steps**, the speed tier.

```python
import torch
from diffusers import FluxPipeline

# FLUX.1-schnell: Apache-2.0, 1-4 step sampling, no CFG (guidance is distilled in)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()        # 12B model streams across a 16-24 GB GPU
# pipe.enable_sequential_cpu_offload() # even tighter VRAM, slower

image = pipe(
    prompt="a macro photo of a bee on a sunflower, golden hour, sharp focus",
    guidance_scale=0.0,        # schnell is guidance-DISTILLED -> CFG is a no-op, keep it 0
    num_inference_steps=4,     # the whole point: 4 steps
    max_sequence_length=256,   # T5 context
    height=1024, width=1024,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("bee.png")
```

Note `guidance_scale=0.0` and `num_inference_steps=4`. Because guidance is baked into the weights, the usual CFG knob does nothing on schnell — that is a *feature*: you get the guided look at single-pass cost. On an RTX 4090 (24 GB), `FLUX.1-schnell` at 4 steps and 1024px generates in roughly 1-2 seconds after warm-up; `FLUX.1-dev` at 28 steps is closer to 10-20 seconds. The distillation is the difference. (The exact latency depends on offload settings, `torch.compile`, and attention backend; treat these as order-of-magnitude.)

#### Worked example: the cost of distilling guidance

Standard CFG runs $2N$ network evaluations for $N$ steps (two passes per step). Guidance distillation makes it $N$. Step distillation then cuts $N$ from ~28 to ~4. Compose them: FLUX-dev at 28 steps with classical CFG would be $2 \times 28 = 56$ forward passes; FLUX-schnell does $1 \times 4 = 4$. That is a **14x** reduction in forward passes through a 12B network — the bulk of why schnell feels instant. The honest cost: step-distilled models lose a little diversity and fine detail versus the 28-step teacher, and they bake in *one* guidance level (you cannot dial CFG at inference). For drafts, thumbnails, and real-time tools, that trade is overwhelmingly worth it; for a hero image you want maximum fidelity, reach for dev or pro.

## SANA: attack the token count, not the step count

FLUX makes the network cheaper *per step* and cuts steps. SANA (Xie et al., NVIDIA + MIT, 2024) attacks a different axis entirely: the **number of tokens**. Recall the token-budget arithmetic — at 1024px the image tokens dominate, and they come from the 8x latent. SANA's thesis: compress harder, get fewer tokens, and the whole network gets cheaper *everywhere*, not just per step.

![A dataflow figure showing SANA's pipeline where a 32x deep-compression autoencoder produces few latent tokens that feed a linear-attention DiT conditioned by a decoder-only LLM text encoder to generate 1024px images at low VRAM](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-5.png)

**The deep-compression autoencoder (32x).** The standard VAE downsamples 8x spatially. SANA's autoencoder downsamples **32x** — a 1024x1024 image becomes a 32x32 latent. Patchify with patch size 1 and that is 1024 image tokens, versus FLUX/SD3's ~4096 (at 8x, patch 2) or 16,384 (at 8x, patch 1). The hard part is training a 32x autoencoder that *does not destroy detail* — compress too much and the decoder cannot recover fine texture and the latent loses the high-frequency information the diffusion model needs. SANA's deep-compression AE uses more channels (a higher-dimensional latent per spatial cell) and careful adversarial+perceptual training to keep reconstruction quality high at 32x. The net effect: **16x fewer tokens** than an 8x latent at patch 2 (and 256x fewer than 8x at patch 1).

**Why fewer tokens helps super-linearly.** Self-attention is $O(N^2)$ in tokens. Cut $N$ by 16x and naive attention cost drops by $16^2 = 256$x. Even the linear parts of the transformer (the MLPs) drop 16x. So deep compression is the highest-leverage knob in the whole stack — it is the one place where you get a quadratic return.

![A comparison matrix showing how 8x, 16x, and 32x autoencoders map to latent grid size, image-token count, quadratic attention cost, and relative compute, with the 32x row cheapest](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-7.png)

**Linear attention.** Even at 1024 tokens, SANA replaces most of the quadratic softmax attention with **linear attention**, which approximates attention with a kernel feature map so cost is $O(N)$ instead of $O(N^2)$. The trick is associativity. Softmax attention computes $\text{softmax}(QK^\top)V$, and the $QK^\top$ is an $N\times N$ matrix — quadratic. Linear attention drops the softmax and instead applies a feature map $\phi(\cdot)$ to queries and keys, so attention becomes $\phi(Q)\,(\phi(K)^\top V)$. Because matrix multiplication is associative, you compute $\phi(K)^\top V$ first — a $d \times d$ matrix, *independent of $N$* — and then multiply by $\phi(Q)$. The cost is $O(N d^2)$ instead of $O(N^2 d)$: linear in the token count. For SANA at high resolution, where $N$ can reach the thousands, that is the difference between feasible and not.

The catch is that dropping the softmax removes its sharp, winner-take-most selectivity — linear attention spreads its weight more uniformly, which blurs fine local detail. Historically this is why linear attention underperformed on images. SANA's fix is a **Mix-FFN**: it inserts a depthwise 3x3 convolution into the feed-forward block, so the *local* spatial structure that linear attention smears out is recovered by the convolution. The division of labor mirrors the encoder stack: linear attention handles cheap global mixing, the depthwise conv handles local detail. The combination — deep-compression AE + linear attention + Mix-FFN — is what lets SANA scale to high resolution cheaply without the texture collapse that naive linear attention would cause.

**A decoder-only LLM as the text encoder.** Instead of CLIP or T5, SANA uses a small **decoder-only LLM** (Gemma) as its text encoder, with prompt engineering and a "complex human instruction" setup to extract strong conditioning. A modern instruction-tuned LLM is a far better language model than CLIP, so SANA gets strong prompt adherence from a relatively small encoder.

**The result.** SANA generates 1024x1024 images with a 0.6B-1.6B model and can run on a **16 GB laptop GPU**, with reported throughput dramatically higher than SDXL at similar or better quality on automated metrics. The headline from the paper: SANA-0.6B is competitive with much larger models while being roughly **20-40x** faster than SDXL at 1024px and able to do 4096px generation, because the token count — not the parameter count — was the real bottleneck.

There is a deeper lesson in why SANA's deep-compression AE was the hard part. You cannot simply train an 8x VAE and bolt a "32x mode" onto it; you have to retrain the autoencoder from scratch with a higher channel count and a much heavier perceptual+adversarial objective, because at 32x the reconstruction problem is genuinely harder — each latent cell must encode a 32x32x3 pixel block (3072 values) into a single spatial position. SANA pushes the channel dimension up to compensate (more channels per cell means more information survives), and the reconstruction quality of *that autoencoder* sets the ceiling on the whole model: the diffusion transformer can never produce detail the decoder cannot reconstruct. This is the inversion I flagged in the stress-test section — in the deep-compression regime, the autoencoder, not the denoiser, is the quality bottleneck, and improving SANA means improving its AE more than its DiT.

```python
import torch
from diffusers import SanaPipeline

# SANA: 32x deep-compression AE + linear attention -> 1024px on a laptop GPU
pipe = SanaPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
# the decoder-only LLM text encoder can run in bf16; the AE + DiT are tiny vs FLUX

image = pipe(
    prompt="a watercolor fox curled asleep in autumn leaves, soft morning light",
    height=1024, width=1024,
    guidance_scale=4.5,
    num_inference_steps=20,
    generator=torch.Generator("cuda").manual_seed(7),
).images[0]
image.save("fox.png")
```

#### Worked example: SANA's token math vs SDXL

SDXL at 1024px in its 8x latent (patch 1, in the U-Net's lowest-resolution attention) processes attention over a 128x128 = 16,384-element map at the coarsest stage and proportionally more at finer stages. SANA at 1024px works on 32x32 = 1024 tokens, end to end. The attention FLOPs scale as the square of the token count, so SANA's attention is on the order of $(16384/1024)^2 = 256$x cheaper *per attention layer* than operating on the 8x map at full resolution — before you even count linear attention turning the remaining $N^2$ into $N$. That compounding is why a sub-2B SANA outruns a 3.5B SDXL by more than 10x in throughput. The lesson generalizes: **in a diffusion transformer, your token count is your budget; halve it before you optimize anything else.**

## Putting the recipe together

Step back and look at the whole thing. The modern text-to-image model is four coordinated choices, and you can read off any frontier model as a point in that design space.

![A comparison matrix of SD3, FLUX.1-dev, FLUX.1-schnell, and SANA across parameter count, text encoders, training objective, sampling steps, and license](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-6.png)

| Model | Params (denoiser) | Backbone | Text encoders | Objective | Steps | Native res | License |
|---|---|---|---|---|---|---|---|
| **SD3 / SD3.5** | 2-8B | MM-DiT (24-38 blocks) | CLIP-L + CLIP-G + T5-XXL | Flow matching | ~28 | 1024 | Community / non-commercial-ish |
| **FLUX.1-dev** | 12B | Double+single-stream DiT | CLIP-L + T5-XXL | Flow matching, guidance-distilled | ~20-50 | 1024 | Non-commercial |
| **FLUX.1-schnell** | 12B | Double+single-stream DiT | CLIP-L + T5-XXL | Guidance + step distilled | **1-4** | 1024 | Apache-2.0 |
| **SANA** | 0.6-1.6B | Linear-attention DiT | Decoder-only LLM (Gemma) | Flow matching | ~20 | 1024-4096 | Permissive (research) |
| **SDXL** (for reference) | 2.6B U-Net | U-Net | CLIP-L + CLIP-G | ε-pred (DDPM) | ~30-50 | 1024 | OpenRAIL |

Notice the common ground: **every frontier model is a transformer (DiT-family) trained with flow matching, conditioned by a strong text encoder, sampling in few steps.** That is the recipe. The differences are *where each one spends its budget*: SD3 on encoder breadth (three encoders) and joint-attention quality; FLUX on depth (57 blocks, 12B) and distillation; SANA on compression (32x AE, linear attention, tiny denoiser). Three ways to win the trilemma.

And the licenses, which I keep flagging, are not a footnote — they often decide which model you can actually ship. FLUX.1-dev is the open quality leader but is non-commercial, so a startup cannot put it behind a paid product without a separate license; FLUX.1-schnell is Apache-2.0 and *can* be shipped commercially, which is a large part of why it became the default for hosted tools despite dev's higher ceiling. SD3.5's community license has its own thresholds. SANA's release is permissive enough for research and increasingly for products. So the practical model-selection question is rarely "which has the lowest FID" — it is "which combination of license, VRAM budget, latency target, and control tooling fits my deployment," and the recipe's four axes (plus the license axis) are the coordinates you reason in. A model that is 5% better on a benchmark but cannot be licensed for your use is not better for you at all.

![A timeline showing the recipe converging year by year from LDM and Stable Diffusion to the DiT backbone, then SD3's MM-DiT with flow matching, FLUX, SANA's deep compression, and the 2025 autoregressive challengers](/imgs/blogs/mmdit-and-the-modern-text-to-image-recipe-8.png)

The timeline makes the convergence vivid. 2022: latent diffusion + U-Net (SD). 2023: the DiT backbone shows transformers scale better. 2024: SD3 fuses DiT + flow matching + joint attention into MM-DiT; FLUX scales it to 12B and distills it; SANA compresses it. 2025: the next branch opens — native-autoregressive image models.

## Adapting the frontier: LoRA on an MM-DiT

One reason the recipe matters in practice is that you rarely run a base model untouched — you fine-tune it for a subject, a style, or a brand. The good news is that the MM-DiT backbone is just a transformer, so the LoRA machinery you know from LLMs transfers almost unchanged: attach low-rank adapters to the attention and MLP projection matrices, freeze everything else, and train on a handful of images. The separate-weights design means you can adapt the *image* stream's projections while leaving the *text* stream alone, or both — a flexibility the U-Net never gave you cleanly.

```python
import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
)
transformer = pipe.transformer

# attach LoRA to the joint-attention and MLP projections of the MM-DiT blocks
lora_cfg = LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0",        # attention
                    "ff.net.0.proj", "ff.net.2",               # MLP
                    "ff_context.net.0.proj", "ff_context.net.2"],  # text-stream MLP
    init_lora_weights="gaussian",
)
transformer = get_peft_model(transformer, lora_cfg)
transformer.print_trainable_parameters()   # typically <1% of the base model

# ... standard flow-matching training loop on your subject images (see flow_match_loss) ...
# then save just the adapter:
transformer.save_pretrained("my-sd3-lora")     # a few MB, not a few GB
```

The `ff_context` target is the giveaway that this is an MM-DiT: `ff` is the image-stream MLP and `ff_context` is the *text*-stream MLP, two separate modules because the weights are separate. A rank-16 LoRA on these is a few megabytes and trains in minutes on a single 24 GB GPU for a subject of 10-20 images. The same recipe works on `FluxTransformer2DModel` (the FLUX LoRA ecosystem is enormous) and on SANA's linear-attention DiT, with the target module names changed to match. The architecture being a clean transformer is *why* the adapter tooling is so portable — another quiet dividend of replacing the U-Net.

A word on what *not* to do. Do not full-fine-tune a 12B FLUX model to teach it your dog; a rank-16 LoRA captures a subject just as well and costs 1000x less to train and store. Do not crank the LoRA rank to 128 "to be safe" — for a single subject, high rank overfits and induces language drift (the model forgets how to draw anything *but* your subject). And do not fine-tune on 5 images and expect generalization — the documented DreamBooth failure mode is exactly this, and joint attention does not rescue you from too little data. We cover the full personalization playbook in the editing and fine-tuning posts of the next track.

## The other branch: native-autoregressive challengers

Everything above is the *diffusion* branch of the frontier. There is a second branch, and a serious one: **native-autoregressive** image models that generate image tokens the way an LLM generates text tokens — one (or one *scale*) at a time, with a next-token objective. **GPT-Image** (OpenAI) and **HunyuanImage-3.0** (Tencent, an open ~80B-parameter unified multimodal model) are the prominent 2025 examples. The pitch: a single transformer that does text *and* image in one autoregressive stack inherits the LLM's reasoning, in-context learning, and instruction-following — so it edits conversationally ("make the sky stormier, keep everything else") and handles complex compositional prompts by *reasoning* about them, not just diffusing.

The trade-off is the classic one. Autoregressive generation is sequential — you decode tokens one after another — so it is intrinsically harder to parallelize than diffusion's batched denoising, and naive raster-order AR over thousands of image tokens is slow. The field is closing that gap with next-*scale* prediction (VAR), masked parallel decoding (MAR), and unified architectures (Transfusion, Chameleon, Janus) that mix a diffusion head onto an AR backbone. We give this its own treatment in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) and the head-to-head in the forthcoming [autoregressive-vs-diffusion 2026 showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown). For this post the point is just: MM-DiT-style diffusion is *one* frontier recipe, and it currently owns the open-weight, controllable, fast-sampling niche; native-AR owns the reasoning-and-editing niche; and the two are visibly converging.

It is worth noticing how much the two branches now *share*, because that is the real story of the frontier. Both are transformers. Both consume a strong language understanding of the prompt — MM-DiT via a T5/LLM encoder, native-AR via its own LLM backbone. Both increasingly use the same position embeddings (RoPE) and the same stabilizers (QK-norm). The difference is narrowing to the *output head and the sampling process*: diffusion predicts a continuous velocity and integrates an ODE; AR predicts a categorical distribution over the next token (or next scale) and samples it. Unified models like Transfusion literally put both heads on one backbone — a diffusion loss on the image tokens and a next-token loss on the text tokens — and train them jointly. So "MM-DiT vs autoregressive" is less a war than a spectrum, and the MM-DiT recipe in this post is the diffusion end of it. The ingredients — joint/shared attention over multimodal tokens, a strong text representation, aggressive compression of the image into few tokens — are exactly the ingredients the AR side reaches for too. Learn this recipe and you have learned most of the AR recipe by transfer.

## Case studies: real numbers from the literature

Let me ground the claims in published results. Be aware these are headline numbers from the respective papers and reports, measured under each team's own protocol (reference set, sample count, resolution) — they are directionally reliable but not perfectly cross-comparable, and I flag the soft ones.

**SD3 (Esser et al., 2024).** The ablation that matters: MM-DiT (joint attention, separate weights) beats the cross-attention DiT and the single-stream variant on validation loss and on GenEval, with the gap widening as the model scales — clean evidence that joint attention is the win, not just a reshuffle. SD3 reports state-of-the-art GenEval among open models at release and a large jump in text-rendering and prompt-following over SDXL. The text-encoder ablation: dropping T5 leaves aesthetics roughly intact but tanks typography and trims GenEval — quantified proof that T5 is the spelling/compositional contributor.

**FLUX.1 (Black Forest Labs, 2024).** 12B parameters, double-stream-then-single-stream, RoPE, guidance + step distillation. FLUX.1-dev set the open-model quality bar on human-preference benchmarks (ELO on prompt following and aesthetics) at release; FLUX.1-schnell delivered comparable-tier quality in **1-4 steps** under Apache-2.0, which is what made it the default for real-time and on-device tools. The schnell-vs-dev gap is the price of step distillation: a little diversity and texture for a ~7-14x speedup in forward passes.

**SANA (Xie et al., 2024).** The deep-compression AE (32x) + linear attention design generates 1024px with a 0.6B model and reports roughly **20-40x** higher throughput than SDXL at 1024px on the same GPU class, while running on a 16 GB laptop GPU and extending to 4096px — all because the token count, not the parameters, was the bottleneck. On automated metrics SANA is competitive with much larger models; on the hardest text-rendering and fine-texture cases the deep compression does cost some fidelity, which is the honest limit of compressing 32x.

**Text rendering, SDXL vs SD3 (the qualitative jump).** This is the case most readers have seen with their own eyes, so it is worth stating as a result. Prompt either model with "a storefront with a sign that reads 'FRESH COFFEE'." SDXL (CLIP-only, cross-attention) gets the storefront and renders the sign as plausible-looking gibberish — the right *shape* of letters, the wrong letters. SD3 (CLIP+T5, joint attention) renders FRESH COFFEE legibly a large fraction of the time, and SD3.5 and FLUX better still. There is no single FID number that captures this — text rendering is a *compositional* capability that aggregate distribution metrics miss entirely, which is part of why the field moved to GenEval and human preference. But the qualitative jump is the most visible single payoff of the whole recipe, and it traces directly to two ingredients: T5's character-and-syntax awareness in the encoder stack, and joint attention letting the glyph tokens negotiate with the image region throughout denoising. Remove either and the legibility collapses — which is exactly what the T5-ablation measures.

Put the four case studies together and a pattern emerges: **every measured win in this recipe traces to one of the four ingredients, and you can predict which.** Better binding and text? Joint attention plus T5. Fewer steps? Flow matching's straight path. Real-time generation? Guidance and step distillation. Laptop-scale 1024px? Deep compression plus linear attention. The recipe is not a bag of unrelated tricks; it is four orthogonal levers on the trilemma, and a frontier model is a choice of how hard to pull each one.

**The throughput comparison, side by side** (1024px, single high-end consumer/datacenter GPU, order-of-magnitude — exact numbers depend on attention backend, `torch.compile`, dtype, and offload):

| Model | Params | Steps | Forward passes | Relative speed @1024px | Notes |
|---|---|---|---|---|---|
| SDXL | 2.6B U-Net | ~30 + CFG | ~60 | 1x (baseline) | 8x latent, ε-pred |
| SD3.5-medium | ~2.5B MM-DiT | ~28 | ~28 (distilled CFG) | ~1-2x | joint attention, flow matching |
| FLUX.1-dev | 12B | ~28 | ~28 | ~0.3-0.5x (bigger net) | best open quality, guidance-distilled |
| FLUX.1-schnell | 12B | **4** | **4** | ~2-4x | step-distilled, Apache-2.0 |
| SANA-1.6B | 1.6B | ~20 | ~20 | **~10-40x** | 32x AE + linear attention |

Read the table as a map of the trilemma. SANA wins *speed* by attacking tokens. FLUX-dev wins *quality* by spending parameters. FLUX-schnell wins *speed* by distilling steps. SD3.5 sits in the balanced middle. There is no free lunch — each model paid for its win on a different face of the triangle.

## When to reach for each (and when not to)

A decisive recommendation, because "it depends" helps no one.

- **Want the best open quality, money/VRAM no object?** FLUX.1-dev (or pro via API). 12B, 28 steps, ~20+ GB with offload. Do *not* reach for it if you need commercial rights (dev is non-commercial) or real-time speed.
- **Need real-time / 1-4 step / on consumer hardware / commercial-OK?** FLUX.1-schnell (Apache-2.0). Do *not* expect to dial CFG or squeeze out maximum fine detail — the distillation fixed those.
- **Need legible text in the image or hard compositional prompts?** Keep T5 in the stack (SD3.5 or FLUX-dev) — quantize it to 4-bit rather than dropping it. Do *not* drop T5 and then complain the sign is misspelled; that is the documented failure mode.
- **Running on a laptop / 16 GB GPU, or need very high resolution cheaply?** SANA. Its 32x AE + linear attention is the only design here built for that regime. Do *not* expect FLUX-dev-level texture on the hardest cases — deep compression has a quality ceiling.
- **Building a controllable production pipeline (LoRA, ControlNet, editing)?** As of 2025 the SDXL and FLUX ecosystems have the richest adapter tooling; SD3.5 and SANA are catching up. Pick the model your *control* tooling supports, not just the one with the best base FID.
- **Need conversational editing or LLM-style reasoning over the prompt?** That is the AR branch (GPT-Image / Hunyuan-3) — a different tool, covered in the AR posts.

A general rule: **do not pay for ingredients you will not taste.** If your prompts are short and text-free, the T5 encoder, the 12B FLUX denoiser, and 28 steps are all over-spend — a SANA or a schnell will serve you at a fraction of the cost. Match the recipe to the workload.

## Stress-testing the recipe

Where does it break?

**At 1-2 steps, even schnell degrades.** Step distillation buys you 4 steps cleanly; push to 1 step and you start to see the smoothing and detail loss characteristic of aggressive distillation (the [distillation post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) covers DMD-style one-step methods that do better). The lesson: 4 steps is the sweet spot for schnell, not 1.

**When the VAE is the bottleneck.** With SANA's tiny linear-attention denoiser, the 32x autoencoder's *decode* can become a meaningful fraction of total latency and the limiting factor on fine texture. If your images look soft, suspect the AE before the denoiser — exactly the inverse of the SDXL regime where the U-Net dominated.

**Three objects and counting.** Joint attention plus T5 dramatically improves attribute binding and counting versus SDXL, but "exactly seven red apples and three green pears" still fails sometimes — compositionality is improved, not solved. GenEval scores went up; they did not hit 100%. If exact counts matter, you are still in the territory where the AR-reasoning branch or explicit layout control helps.

**Dropping T5 then prompting for text.** Covered above, but worth repeating as a stress test because it is the single most common "why is my sign gibberish" support question: the spelling ability lives substantially in T5. No T5, no reliable text.

**Over-guiding a flow-matching model.** Bring your SDXL habit of `guidance_scale=7.5` to SD3 and you over-saturate and posterize. Flow-matching models want ~3.5-5. The straighter path is already sharp; guidance is a smaller correction, not a big one.

**Mixing a schnell habit onto a dev model (and vice versa).** A common operational mistake: copy a working FLUX-schnell call (`guidance_scale=0.0, num_inference_steps=4`) and point it at FLUX-dev. Dev is *not* step-distilled, so 4 steps gives you a blurry, half-denoised mess; and dev *does* consume guidance, so `0.0` flattens the prompt adherence. The inverse — running schnell at 28 steps with `guidance_scale=4.5` — wastes 7x the compute for no quality gain, because schnell's distillation already baked the guidance in and saturates by step 4. The rule is that the *tier dictates the call*: schnell is 1-4 steps at guidance 0; dev is ~28 steps at guidance ~3.5. Treat the step count and guidance as properties of the *checkpoint*, not knobs you carry between checkpoints. This single confusion is responsible for a large share of "FLUX looks bad for me" reports, and it is purely an inference-config error, not a model failure.

## Key takeaways

- **Modern text-to-image = flow matching + (MM-)DiT + a strong text encoder + aggressive latent compression.** Memorize that sentence; every frontier model is a point in that four-axis space.
- **Joint attention beats cross-attention** because it is bidirectional: text and image tokens sit in one sequence and update each other, which is *why* SD3/FLUX render legible text and bind attributes far better than SDXL.
- **Separate weights, shared attention** is the "MM" in MM-DiT — two streams of parameters, one attention operation. The doubled projection/MLP cost buys a measurable GenEval and CLIP-score gain.
- **QK-normalization** (RMSNorm on queries and keys) bounds the attention logits and is what makes a multi-billion-parameter MM-DiT train stably in `bf16`. Cheap, mandatory.
- **The text-encoder stack divides labor**: CLIP pooling sets the global gist (AdaLN), T5-XXL supplies syntax and spelling. Drop T5 to save ~9 GB and you keep the look but lose legible text — quantize it to 4-bit instead.
- **Flow matching's straight path** is why SD3/FLUX-dev sample well in ~28 steps where DDPM needed 50, and why distillation can push FLUX-schnell to 1-4. Use a *lower* CFG (~3.5-5) than you would with SDXL.
- **FLUX = depth + distillation**: double-stream then single-stream, 57 blocks, 12B, RoPE; guidance distillation makes it single-pass and step distillation makes schnell real-time. The tiers (pro/dev/schnell) trade quality, license, and speed.
- **SANA = attack the token count**: a 32x deep-compression AE cuts image tokens ~16x, which cuts quadratic attention ~256x per layer; linear attention and a decoder-only LLM encoder finish the job, yielding 1024px on a laptop GPU. Your token count is your budget — cut it first.
- **The AR branch (GPT-Image, Hunyuan-3) is the other frontier** — reasoning and conversational editing at the cost of sequential decoding; diffusion and AR are converging on shared primitives (transformers, RoPE, QK-norm, an LLM-grade text representation).
- **The four levers are orthogonal and the license is a fifth axis.** Every measured win traces to one lever (joint-attention+T5 for adherence, flow matching for steps, distillation for speed, deep compression for memory), and the right model is the one whose lever settings *and license* match your deployment — not the one with the lowest benchmark number.
- **The recipe is a co-adapted system, not a bag of tricks.** Flow matching needs the transformer, joint attention needs the cheap text tokens, the strong encoder needs the two-way channel. Each ingredient is ordinary alone and excellent in the assembly, which is why the whole field converged on the same recipe within a single year.

## Further reading

- **Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Stable Diffusion 3), 2024** — the MM-DiT paper: joint attention, the three-encoder stack, QK-norm, logit-normal flow matching, the ablations cited here.
- **Black Forest Labs, "FLUX.1" (model card and technical notes), 2024** — the double-stream/single-stream design, RoPE, guidance distillation, the pro/dev/schnell tiers.
- **Xie et al., "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers," 2024** — the 32x deep-compression autoencoder, linear attention + Mix-FFN, the decoder-only LLM text encoder, the throughput numbers.
- **Lipman et al., "Flow Matching for Generative Modeling," 2023** and **Liu et al., "Rectified Flow," 2023** — the training objective underneath all three models.
- **Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023** — the backbone MM-DiT extends; see the sibling post [diffusion transformers (DiT)](/blog/machine-learning/image-generation/diffusion-transformers-dit).
- **Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," 2022** — the latent-space foundation; see [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).
- **Ho & Salimans, "Classifier-Free Guidance," 2022** — the guidance that FLUX distills; see [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance).
- **🤗 `diffusers` docs** — `StableDiffusion3Pipeline`, `FluxPipeline`, `SanaPipeline`, and the `SD3Transformer2DModel` / `FluxTransformer2DModel` model classes.

When you are ready to assemble all of this into a serving pipeline — model choice, encoder/quantization trade-offs, sampler and step budget, LoRA and ControlNet, cost per image — that is the [capstone, building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack). And for the math of the distributions all of this is approximating, the foundation post is [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions).
