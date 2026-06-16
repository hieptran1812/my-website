---
title: "Diffusion Transformers: Why DiT Replaced the U-Net"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Trace the Diffusion Transformer from patchify to AdaLN-Zero to its clean compute-to-FID scaling law, derive the token count and quadratic attention cost from first principles, build a DiT block in PyTorch, and see exactly why DiT-XL/2 beat the best U-Nets and became the trunk under SD3, FLUX, and Sora."
tags:
  [
    "image-generation",
    "diffusion-models",
    "dit",
    "diffusion-transformer",
    "adaln",
    "transformer",
    "scaling-laws",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/diffusion-transformers-dit-1.png"
---

For five years, the picture in your head when someone said "diffusion model" was a U-Net. The encoder-decoder with the famous skip connections, the residual blocks at each resolution, the self-attention bolted onto the 16×16 and 8×8 feature maps, the time embedding fanned out through every block. That architecture trained DDPM, it trained ADM, it trained Stable Diffusion. It was so synonymous with diffusion that for a while the two ideas — "denoise an image" and "use a U-Net" — were treated as one thing. Then in late 2022 a paper from William Peebles and Saining Xie asked a question that, in hindsight, was obvious and, at the time, was heresy: what if the denoiser doesn't need to be a U-Net at all? What if it's just a transformer?

The answer turned out to be the most consequential architectural shift in image generation since latent diffusion itself. The Diffusion Transformer — DiT — threw away the convolutions, the multi-resolution stages, the carefully tuned skip connections, and replaced all of it with the same plain transformer that powers every large language model: patch the input into tokens, run a stack of identical attention-plus-MLP blocks, project back out. And it didn't just match the U-Net. DiT-XL/2 reached **FID 2.27** on class-conditional ImageNet 256×256, beating the best convolutional models (ADM, LDM) that the field had spent years tuning. More importantly, it scaled *cleanly*: every time you added compute, FID went down, on a tidy log-linear trend, with none of the diminishing-returns wall that plagued bigger U-Nets. That single property — predictable scaling — is why DiT, not the U-Net, is the trunk underneath SD3, FLUX, PixArt, and Sora today.

![A vertical stack diagram showing the DiT forward pass from a noisy latent through patchify into tokens, a stack of DiT blocks fed by timestep and class conditioning, then unpatchify back to a predicted noise and covariance output.](/imgs/blogs/diffusion-transformers-dit-1.png)

Figure 1 is the whole architecture in one column, and it's worth pinning to the top of the page. A noisy latent comes in from the VAE (we are working in latent space — see [latent diffusion and stable diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) for why we never run DiT on raw pixels). **Patchify** chops it into a short sequence of tokens. A linear layer plus positional embeddings turns those into a transformer's input. A stack of **DiT blocks** — each one self-attention, then an MLP, with the timestep and class conditioning injected through *adaptive layer norm* rather than through extra tokens — processes the sequence. Then a final linear layer and **unpatchify** fold the tokens back into a latent-shaped tensor: the predicted noise. That is the entire model. No U-Net. No convolutions. No skip connections across resolutions.

By the end of this post you will be able to: derive the exact token count from a latent resolution and patch size, and the quadratic attention FLOPs that follow; explain what AdaLN-Zero is, why initializing the residual gates to zero makes each block start as the identity, and why that one trick stabilizes training at scale; read the DiT compute↔FID scaling trend and the S/B/L/XL sweep the way you'd read an LLM scaling law; implement a complete DiT block in PyTorch — patch embedding, the adaLN-zero modulation producing shift/scale/gate, the gated attention and MLP residuals; load the equivalent in 🤗 `diffusers` with `DiTTransformer2DModel` and `SD3Transformer2DModel`; and state precisely when the transformer wins over the U-Net and when its quadratic cost bites back. We'll keep tying it to the series spine — the **generative trilemma** (quality × diversity × speed) and the **diffusion stack** (data → VAE latent → noising → *denoiser* → sampler → guidance → image) — because DiT changes exactly one box in that stack, the denoiser, and changes everything downstream by doing so.

A note on prerequisites. You should be comfortable with the diffusion forward/reverse processes and the ε-prediction loss — if not, read [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm) first; DiT changes the *network* that predicts ε, not the objective it's trained on. You should also know what a transformer block is (multi-head self-attention plus a feed-forward MLP, residual connections, layer norm). If you've read a single ViT or GPT explainer you have enough. Everything DiT-specific — patchify for latents, AdaLN-Zero, the scaling story — we build here.

## 1. The setup: a denoiser is just a function on a tensor

Let's be precise about what job the denoiser does, because the whole argument for DiT rests on noticing that the job has no convolutional requirement baked into it. In latent diffusion, the VAE encoder compresses a 256×256×3 image down to a latent $z_0 \in \mathbb{R}^{32\times32\times4}$ — a 4-channel grid one-eighth the spatial size (a 48× reduction in element count). The forward process noises this latent: at timestep $t$,

$$
z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\, \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon}\sim\mathcal N(\mathbf 0, \mathbf I).
$$

The denoiser is a function $\boldsymbol\epsilon_\theta(z_t, t, y)$ that takes the noisy latent, the timestep, and a conditioning signal $y$ (a class label, or later a text embedding), and predicts the noise $\boldsymbol\epsilon$ that was added. That's it. The loss is the same $\mathcal L_\text{simple} = \mathbb E\,\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(z_t,t,y)\|^2$ you derived in the DDPM post. DiT changes *nothing* about this objective. It changes only the parametric form of $\boldsymbol\epsilon_\theta$.

Now ask: what does the function $\boldsymbol\epsilon_\theta$ actually *need*? It needs to map a $32\times32\times4$ tensor to a $32\times32\times4$ tensor (the predicted noise, same shape as the input; ADM and DiT additionally predict a per-element variance, doubling the output channels — more on that later). It needs to condition on a scalar timestep and a label. It needs enough capacity and the right inductive biases to learn the denoising function across all noise levels. The U-Net's answer to "right inductive biases" was: locality (convolutions), multi-scale processing (downsample/upsample stages), and long-range mixing only at low resolution (attention at 16×16 and below). Those are *good* biases for natural images. But they are not the *only* biases that work, and — crucially — they come with a ceiling.

Here is the tension that motivated DiT, stated plainly. Convolutions have a strong locality prior, which is sample-efficient at small scale but becomes a *constraint* at large scale: a conv kernel can only ever look at a fixed neighborhood, and you bolt on attention as an afterthought to get global mixing. The U-Net's multi-resolution structure is hand-designed — how many stages, how many ResBlocks per stage, where to put attention, the channel multipliers — and every one of those is a hyperparameter someone tuned by hand on a specific dataset and resolution. When you want a bigger model, you don't have one clean knob; you have a dozen interacting ones. Transformers, by contrast, have almost no spatial inductive bias (just positional embeddings) and *one* clean scaling recipe inherited from years of LLM research: make it wider, make it deeper, add tokens. The bet DiT made was that for latent diffusion — where the VAE has already done the heavy perceptual compression — you don't *need* the conv biases, and giving them up buys you the transformer's clean scalability. That bet paid off.

It's worth dwelling on *why* the VAE makes the conv prior dispensable, because this is the load-bearing observation and it's easy to skim past. A convolution's locality prior is valuable when neighboring elements are strongly correlated and structure is local — which is exactly true of raw pixels (adjacent pixels are nearly identical, edges are local, textures repeat over small windows). But a VAE latent is *not* raw pixels. It is a learned, compressed, decorrelated representation where each of the 1,024 spatial positions already summarizes an 8×8 pixel block, and where the encoder has been trained specifically to pack perceptually relevant information densely. In that latent space the "neighbors are nearly identical" assumption is much weaker — adjacent latent cells can encode quite different content — so the conv locality prior buys less, and the cost of *not* having it (needing global attention to relate distant cells) is cheap because there are only 1,024 cells, not 65,536 pixels. The VAE, in other words, has already done the spatial-compression job that the U-Net's downsampling stages were doing in pixel space. Stacking a U-Net's multi-resolution machinery on top of an already-compressed latent is partly redundant. That redundancy is the slack DiT exploits: drop the conv biases, and a flat transformer over 256 tokens has enough capacity and reach to do the whole denoising job, with the VAE handling the perceptual compression the conv stages used to provide.

There's a second, more pragmatic reason the field wanted out of the U-Net, and it's about *engineering*, not just FID. A U-Net is a tangle of skip connections, resolution-specific blocks, and asymmetric encoder/decoder paths that is genuinely awkward to scale on modern parallel hardware: the skip connections create cross-stage memory dependencies, the varying spatial sizes make activation memory hard to balance, and the bespoke structure resists the off-the-shelf parallelism (tensor, sequence, pipeline) that the LLM world had spent years perfecting. A transformer is, by contrast, *embarrassingly* uniform — N identical blocks, each a matmul-heavy attention plus MLP, the exact shape that FlashAttention, `torch.compile`, and every distributed-training framework are optimized for. When you choose DiT you are not just choosing a different inductive bias; you are choosing to run your diffusion model on the same battle-tested infrastructure that trains 70B-parameter language models. That alignment with existing infrastructure is a quiet but enormous part of why the industry switched so fast — the moment DiT showed parity on quality, the operational case for the transformer was already overwhelming.

We'll spend the rest of the post making each piece of that argument concrete and measured. Start with the very first operation, the one that lets a transformer touch an image latent at all: patchify.

## 2. Patchify: turning a latent into a sequence of tokens

A transformer operates on a *sequence* of vectors (tokens), each of dimension $d$ (the "hidden size" or "width"). An image latent is a *grid* of vectors. Patchify is the bridge. It is exactly the operation from Vision Transformers (ViT, Dosovitskiy et al. 2020), applied here to a noised latent instead of a clean image.

![A grid diagram showing a 32 by 32 latent split into non-overlapping 2 by 2 patches, each patch flattened into one of 256 ordered tokens numbered across the rows.](/imgs/blogs/diffusion-transformers-dit-3.png)

The recipe, shown in figure 3: take the $32\times32\times4$ latent and a **patch size** $p$. Tile the spatial grid into non-overlapping $p\times p$ patches. Each patch is a little $p\times p\times 4$ block; flatten it into a vector of length $p^2\cdot 4$, and project it with a single learned linear layer (equivalently, a conv with kernel $=$ stride $=p$) up to the transformer's hidden size $d$. Add a positional embedding so the model knows where each patch sat. The result is a sequence of $T$ tokens, each a $d$-dimensional vector, ready for a plain transformer.

The token count falls straight out of the arithmetic. If the latent is $H\times W$ spatially and the patch size is $p$, then

$$
T \;=\; \frac{H}{p}\cdot\frac{W}{p} \;=\; \frac{HW}{p^2}.
$$

This is the single most important equation for understanding DiT's cost. Plug in the standard configuration — a $32\times32$ latent (from a 256×256 image through an 8× VAE) and patch size $p=2$ — and you get

$$
T \;=\; \frac{32\cdot 32}{2^2} \;=\; \frac{1024}{4} \;=\; 256 \text{ tokens.}
$$

That's why the canonical DiT is written **DiT-XL/2**: the "/2" is the patch size, and it sets the sequence length to 256. If instead you used $p=4$, you'd get $T = 1024/16 = 64$ tokens — a 4× shorter sequence, much cheaper, but coarser (each token now summarizes a $4\times4$ latent region, throwing away fine spatial detail). If you used $p=8$, $T=16$ tokens, almost a thumbnail. Patch size is the dial that trades spatial granularity for sequence length, and because attention is quadratic in sequence length, it is also the dial that trades quality for speed. In the DiT paper, $p=2$ was consistently the best for quality — smaller patches mean more tokens mean more compute mean lower FID. The "/2" suffix is doing real work; DiT-XL/4 and DiT-XL/8 exist and are progressively worse and cheaper.

#### Worked example: token counts at three resolutions

Suppose you want to generate at three resolutions with the same 8× VAE and $p=2$. A 256×256 image → 32×32 latent → $T=256$ tokens. A 512×512 image → 64×64 latent → $T = 64^2/4 = 1024$ tokens. A 1024×1024 image → 128×128 latent → $T = 128^2/4 = 4096$ tokens. Notice the brutal scaling: doubling the image side quadruples the token count, and we'll see in a moment that attention cost grows with the *square* of token count. From 256→1024 image resolution, tokens go up 16×, and attention FLOPs go up roughly **256×**. This is the central cost story of DiT and the reason high-resolution generation forces you to either (a) lean even harder on the VAE to compress more aggressively (SANA's 32× autoencoder, which we cover in [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe)), or (b) replace softmax attention with a linear-attention variant. Hold that thought; it's section 6.

Patchify is also where DiT inherits a lovely flexibility the U-Net lacks: **variable resolution is just a different number of tokens**. A U-Net trained at 256×256 has its architecture partly frozen to that resolution (the number of downsampling stages, the spatial sizes where attention lives). A transformer doesn't care — give it 256 tokens or 1024 tokens, the same blocks process either, you only need positional embeddings that extend (interpolated or 2D-RoPE) to the longer sequence. This is why modern DiT-based models (SD3, FLUX) handle multiple aspect ratios and resolutions so naturally: it's all just sequences of patch tokens.

Here is patchify in PyTorch, the way DiT actually implements it — a single strided convolution does the tiling and projection in one shot:

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Latent grid -> sequence of patch tokens (ViT-style)."""
    def __init__(self, latent_size=32, patch_size=2, in_chans=4, embed_dim=1152):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (latent_size // patch_size) ** 2  # 256 for 32/2
        # conv with kernel = stride = patch_size both tiles AND projects
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                 # x: (B, 4, 32, 32)
        x = self.proj(x)                  # (B, 1152, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 1152)  -> tokens
        return x

# sanity check
pe = PatchEmbed()
z = torch.randn(2, 4, 32, 32)            # a batch of two noisy latents
tokens = pe(z)
print(tokens.shape)                       # torch.Size([2, 256, 1152])
print("tokens per image:", pe.num_patches)  # 256
```

The positional embedding in DiT is a fixed 2D sin-cos embedding (not learned) added to these tokens, exactly as in the original ViT for the absolute-position case. The "2D" part matters: a patch token needs to know both its row and its column in the grid, so DiT builds the embedding by concatenating a 1D sin-cos encoding of the row index with a 1D sin-cos encoding of the column index. This is what tells the otherwise permutation-invariant attention that token 17 sits directly below token 1 and not somewhere arbitrary — without it, the transformer would treat the latent as an unordered bag of patches and lose all spatial structure. The choice of a *fixed* (non-learned) embedding is deliberate: it generalizes to sequence lengths not seen at training time, which is precisely what you want when you later sample at a different resolution.

This is the first place DiT's flexibility shows up concretely, and it's worth making the contrast with the U-Net sharp. A U-Net trained at 256×256 has spatial sizes baked into its architecture: the number of downsampling stages and the feature-map resolutions where attention lives are *fixed at construction time*. To generate at 512×512 you typically retrain or at least fine-tune, and the architecture itself has to change. DiT changes nothing about its blocks — a different resolution is just a different number of patch tokens, and the 2D sin-cos positional embedding extends to the longer grid (the modern variants use interpolated absolute embeddings or 2D rotary embeddings, RoPE, which extrapolate even more gracefully). The same 28 blocks that process 256 tokens at 256×256 process 1,024 tokens at 512×512 with no structural change. This is why DiT-lineage models handle arbitrary aspect ratios and multiple resolutions so naturally, and it is a direct, practical consequence of patchify plus a resolution-agnostic positional scheme. The U-Net's resolution-baked structure is exactly the kind of hand-design the transformer dissolves.

With the sequence built, every block from here on is a transformer block. The only question — and the heart of the DiT design — is how you inject the conditioning.

## 3. The DiT block and AdaLN: how conditioning gets in

A language transformer's block is simple: `x = x + Attn(LN(x))`, then `x = x + MLP(LN(x))`. Self-attention mixes information across tokens; the MLP processes each token; layer norm stabilizes; residual connections let gradients flow. DiT keeps this skeleton exactly. The whole design problem is: where do the timestep $t$ and the class/text condition $y$ enter?

You have a few options, and Peebles & Xie tried all of them. (1) **In-context conditioning**: append $t$ and $y$ as extra tokens in the sequence, like a `[CLS]` token, and let attention mix them in. Simple, but the conditioning competes with image tokens for attention and the paper found it weakest. (2) **Cross-attention**: add a cross-attention layer in each block that attends from image tokens to the conditioning, the way the U-Net does text conditioning. Works, but adds the most parameters and FLOPs. (3) **Adaptive layer norm (AdaLN)**: don't add tokens or layers at all — instead, *modulate* the existing layer norms with parameters computed from the conditioning. This is the cheapest and, with one critical modification, the best.

It helps to see these four mechanisms (the three above plus AdaLN's zero-init variant) side by side, because the ranking is counterintuitive — the *cheapest* mechanism wins:

| Mechanism | Extra cost per block | Conditioning path | DiT-XL/2 FID (no CFG) |
|---|---|---|---|
| In-context tokens | +2 tokens in attention | competes with image tokens | ~35 (worst) |
| Cross-attention | a full extra attention layer | image tokens attend to cond | ~28 |
| AdaLN | one MLP → 6 modulation tensors | modulates layer norm | ~19.5 |
| **AdaLN-Zero** | same MLP, **gate init 0** | modulates + identity-init gate | **9.62 (best)** |

The pattern is the lesson: adding *capacity* to the conditioning path (cross-attention's whole extra attention layer) helps less than adding the right *inductive structure* (AdaLN's modulation, plus identity init). More parameters in the wrong place lose to fewer parameters in the right place. That is a recurring theme in architecture design, and DiT is a clean demonstration of it.

![A branching graph of one DiT block showing token input and conditioning input, an adaptive layer-norm MLP that emits six modulation parameters, and the self-attention and MLP sublayers each gated before adding back to the residual stream.](/imgs/blogs/diffusion-transformers-dit-2.png)

Figure 2 shows the AdaLN-modulated DiT block. Let's unpack it. Standard layer norm normalizes a token vector and then applies a *learned* scale $\gamma$ and shift $\beta$: $\text{LN}(x) = \gamma\odot \frac{x-\mu}{\sigma} + \beta$, where $\gamma,\beta$ are fixed parameters of the layer. **Adaptive** layer norm makes $\gamma$ and $\beta$ *functions of the conditioning* instead of fixed: $\gamma(c), \beta(c)$ where $c$ is the conditioning vector. So the same normalization is dynamically rescaled and shifted by a small network that reads the timestep and label. This is the FiLM idea (Feature-wise Linear Modulation, Perez et al. 2018) — the U-Net's own AdaGN time-conditioning is a sibling — adapted to a transformer's layer norm. We discuss the U-Net's version in [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet).

DiT goes one step further than plain AdaLN. Each DiT block has *two* sublayers (attention, MLP), and AdaLN-DiT computes **six** modulation parameters from the conditioning, not two: a scale and shift for the attention's layer norm, a scale and shift for the MLP's layer norm, and — the key addition — a **gate** ($\alpha$) that multiplies the *output* of each sublayer before it's added back to the residual stream. So a DiT block computes:

$$
\begin{aligned}
(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) &= \text{MLP}_\text{adaLN}(c), \\
x &\leftarrow x + \alpha_1 \odot \text{Attn}\big(\gamma_1 \odot \text{LN}(x) + \beta_1\big), \\
x &\leftarrow x + \alpha_2 \odot \text{MLP}\big(\gamma_2 \odot \text{LN}(x) + \beta_2\big).
\end{aligned}
$$

The conditioning vector $c = \text{embed}(t) + \text{embed}(y)$ is a single vector per image; the adaLN MLP maps it to $6d$ numbers (six $d$-dimensional vectors), reshaped into the six modulation tensors. Every block has its own adaLN MLP, so different layers can modulate differently. Critically, conditioning here costs almost nothing in the attention itself — the image tokens never attend to condition tokens; the condition only reshapes the per-token normalization and gates. That's why AdaLN is the cheapest of the three options: a handful of extra linear layers, no extra attention.

That's the *what*. The *why it's the best* hides in the word "gate" and one initialization choice, which deserves its own section because it is the single most important trick in the whole paper.

## 4. AdaLN-Zero: identity initialization and why it matters

The gates $\alpha_1, \alpha_2$ multiply each sublayer's output before the residual add. Now ask: what if you initialize the adaLN MLP so that, *at the start of training*, those gates output exactly **zero**?

If $\alpha_1 = \alpha_2 = 0$ at initialization, then look at the block equations: $x \leftarrow x + 0\cdot\text{Attn}(\cdots) = x$, and $x \leftarrow x + 0\cdot\text{MLP}(\cdots) = x$. The block is the **identity function**. It passes its input straight through, untouched. Stack 28 such blocks and the entire DiT, at initialization, is one giant identity map from input to (the final layer of) output. This is **AdaLN-Zero**, and it is the reason DiT trains stably at scale.

![A before-and-after comparison contrasting plain AdaLN with random gate initialization against AdaLN-Zero with the gate initialized to zero, showing the identity-initialized version reaching a much lower DiT-XL FID.](/imgs/blogs/diffusion-transformers-dit-7.png)

Why does identity-at-init help so much? Figure 7 contrasts the two. The intuition comes from residual network theory (and the ReZero / Fixup / SkipInit line of work): a deep residual network is easiest to optimize if it starts close to the identity, because then the signal and the gradient pass through cleanly without being scrambled by 28 layers of randomly-initialized transforms. If instead every block starts perturbing the residual stream with random attention and MLP outputs (as plain AdaLN with random gate init does), then early in training the forward signal is mangled by deep random transforms and the gradients are noisy — and the bigger the model, the worse this gets, because there are more random layers compounding. Identity init says: start the model as a no-op, and let gradient descent *gradually* dial up each block's contribution by raising its gate off zero only where it helps. Depth is added on demand rather than imposed from step zero.

There's a cleaner way to see why this is *quantitatively* important and not just a vibe. Consider the variance of the residual stream as it flows through the stack. In a standard pre-norm transformer, each block adds its sublayer output to the residual, and those additions *accumulate variance*: after $L$ blocks the residual-stream variance has grown roughly in proportion to $L$ (each block contributes some independent variance). For a 28-layer model that's a 28-fold growth in the scale of activations from input to output at initialization — which means the final layer sees activations on a wildly different scale than the first, the normalization statistics drift, and the effective learning rate per layer is mismatched. Now zero the gates: every block contributes *exactly zero* variance at init, so the residual stream's scale is *preserved* end to end — the activation at layer 28 has the same scale as the input. The network is perfectly conditioned at step zero, and as training raises the gates off zero, the variance grows *gradually and under the optimizer's control* rather than being dumped on the model all at once. This variance-preservation argument is why identity init matters *more* the deeper and wider the model gets — which is precisely the regime (DiT-L, DiT-XL) where the ablation shows the biggest AdaLN-Zero win. It's not a coincidence that the trick the paper found indispensable is the one whose benefit grows with scale; that's the same reason it carried over unchanged into the much larger MM-DiT and FLUX models.

A useful way to hold this: AdaLN-Zero is the diffusion-transformer instance of a general principle — *make deep residual networks start as the identity and let them earn their depth*. The same principle shows up as zero-initialized final convolutions in ControlNet's zero-convolutions, as ReZero's learnable per-block scalar, as Fixup's careful residual rescaling. DiT's contribution was to fold it into the *conditioning* mechanism itself: the gate that implements identity init is the same gate that injects the timestep and class signal. One mechanism, two jobs — modulation and stable initialization — which is the kind of economical design that tends to survive scaling.

The empirical effect is dramatic and it is the headline ablation of the paper. Compare DiT-XL/2 trained with the three conditioning mechanisms, all else equal, after 400K training steps (FID-50K, lower is better):

| Conditioning mechanism | DiT-XL/2 FID-50K |
|---|---|
| In-context (extra tokens) | 35.0 (approx) |
| Cross-attention | 28.0 (approx) |
| AdaLN (random gate init) | 19.5 (approx) |
| **AdaLN-Zero (gate init 0)** | **9.6** |

AdaLN-Zero roughly *halves* the FID of plain AdaLN and is more than 3× better than in-context conditioning, while being the cheapest mechanism in FLOPs. (These are the relative numbers from the DiT ablation; treat the non-AdaLN-Zero figures as approximate readings off the paper's curves — the exact published headline is AdaLN-Zero's 9.62, and the ordering AdaLN-Zero ≪ AdaLN ≪ cross-attn ≪ in-context is the robust, reproducible finding.) Identity initialization is not a minor tweak. It is the difference between a transformer that scales and one that fights you the whole way up.

Here is the modulation logic in PyTorch, with the all-important zero initialization made explicit:

```python
import torch.nn as nn

def modulate(x, shift, scale):
    # x: (B, T, d); shift/scale: (B, d) -> broadcast over the T token axis
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNZeroModulation(nn.Module):
    """Maps conditioning c -> six modulation tensors, zero-initialized."""
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.lin = nn.Linear(dim, 6 * dim, bias=True)
        # THE trick: zero-init the final linear so all 6 outputs start at 0.
        # scale starts 0 -> modulate() multiplies by (1+0)=1 (identity LN);
        # gate starts 0 -> the residual update is multiplied by 0 (identity block).
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, c):                       # c: (B, d)
        h = self.lin(self.act(c))               # (B, 6d)
        return h.chunk(6, dim=1)                # 6 tensors of (B, d)
```

Note the subtlety in `modulate`: we apply `(1 + scale)`, not `scale`, so that a zero scale means *multiply by 1* (a no-op normalization), and the shift starts at 0 (no shift). Combined with the gate starting at 0, the block is a perfect identity at init. This `(1 + scale)` convention is standard across DiT, MM-DiT, and the 🤗 `diffusers` implementations — get it wrong (use bare `scale`) and a zero-init scale would *blank the normalized features to zero*, which is catastrophic. Small detail, load-bearing.

## 5. Assembling the full DiT block in PyTorch

Now stitch patchify, AdaLN-Zero modulation, attention, and MLP into a complete, runnable DiT block. This is the core of the model; everything else (the N-fold stack, the final layer, unpatchify) is wiring around it. I'll use PyTorch's built-in `scaled_dot_product_attention` (which dispatches to FlashAttention internally when available) so the attention is efficient and the code is short.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):                      # x: (B, T, d)
        B, T, d = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each (B, H, T, hd)
        # FlashAttention / SDPA: the quadratic-in-T step lives here
        out = F.scaled_dot_product_attention(q, k, v)    # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class DiTBlock(nn.Module):
    """One DiT block: adaLN-zero conditioned self-attention + MLP."""
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        # elementwise_affine=False: LN has NO learned scale/shift of its own;
        # adaLN supplies them dynamically from the conditioning.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp   = Mlp(dim, mlp_ratio)
        self.ada   = AdaLNZeroModulation(dim)   # from section 4

    def forward(self, x, c):                    # x: (B,T,d)  c: (B,d)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c)
        # attention sublayer, gated
        x = x + gate1.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift1, scale1))
        # MLP sublayer, gated
        x = x + gate2.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift2, scale2))
        return x

# smoke test: identity at init (all gates start at 0)
block = DiTBlock(dim=1152, n_heads=16)
x = torch.randn(2, 256, 1152)
c = torch.randn(2, 1152)
y = block(x, c)
print("max |y - x| at init:", (y - x).abs().max().item())  # ~0.0: identity!
```

Run that smoke test and the final print is essentially zero: at initialization the block returns its input unchanged, because the zero-initialized gates kill both residual updates. That is AdaLN-Zero working exactly as designed, and it's a one-line check you should always run after implementing it — if it's not near zero, your gate init is wrong.

The full DiT wraps a stack of these. Conditioning is built once at the top (timestep embedding plus label embedding), passed to every block. The final layer is a special "FinalLayer" that does one last adaLN-modulated norm and a linear projection back to patch-pixel space, then unpatchify reshapes the token sequence back to the latent grid. Here is the skeleton:

```python
class DiT(nn.Module):
    def __init__(self, latent_size=32, patch_size=2, in_chans=4,
                 dim=1152, depth=28, n_heads=16, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(latent_size, patch_size, in_chans, dim)
        self.t_embed = TimestepEmbedder(dim)         # sinusoid + 2-layer MLP
        self.y_embed = nn.Embedding(num_classes + 1, dim)  # +1 for CFG null
        self.pos = nn.Parameter(get_2d_sincos_pos(dim, latent_size // patch_size),
                                requires_grad=False)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, n_heads) for _ in range(depth)])
        # output channels = 2*in_chans: predict eps AND a per-element variance
        self.final = FinalLayer(dim, patch_size, 2 * in_chans)

    def forward(self, z, t, y):                       # z:(B,4,32,32) t:(B,) y:(B,)
        x = self.patch_embed(z) + self.pos            # (B, 256, dim)
        c = self.t_embed(t) + self.y_embed(y)         # (B, dim)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.final(x, c)                          # (B, 256, p*p*2*4)
        return unpatchify(x)                          # (B, 8, 32, 32) -> eps, Sigma
```

`DiT-XL/2` is exactly this with `dim=1152, depth=28, n_heads=16, patch_size=2` — 675M parameters. The `2 * in_chans` output is the "learn-sigma" choice inherited from ADM (Nichol & Dhariwal): the network predicts both the noise *and* a per-dimension variance interpolation for the reverse process, which improves log-likelihood. If you only want ε-prediction, output `in_chans` channels instead.

Now let's build the conditioning path more carefully, because the way $t$ and $y$ become a single vector $c$ is worth seeing in full.

## 6. Conditioning: from timestep and label to six modulation tensors

The conditioning vector $c$ feeds every block's adaLN MLP, so getting it right matters. Figure 6 shows the path.

![A branching graph showing a sinusoidal timestep embedding and a class label embedding summing into one conditioning vector, which an adaptive layer-norm MLP expands into separate attention and MLP modulation parameter sets.](/imgs/blogs/diffusion-transformers-dit-6.png)

The **timestep** $t$ (an integer in $[0, 1000)$) becomes a vector via a sinusoidal embedding — the same Fourier-feature trick as the transformer's positional encoding, mapping a scalar to a smooth high-dimensional vector — followed by a small two-layer MLP. The **class label** $y$ (an integer in $[0, 1000)$ for ImageNet) becomes a vector via a learned embedding table. For classifier-free guidance, the table has one extra "null" row: during training you randomly drop the label (replace $y$ with the null token) ~10% of the time, so the model learns both conditional and unconditional denoising and you can extrapolate between them at sampling time. (CFG is essential to DiT's headline numbers — the FID 2.27 is *with* guidance; see [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) for the derivation and the scale↔quality trade-off.) The two embeddings are simply **summed**: $c = \text{embed}(t) + \text{embed}(y)$. That single vector is then handed to every block.

```python
import math, torch
import torch.nn as nn

class TimestepEmbedder(nn.Module):
    """Scalar timestep -> sinusoidal features -> MLP, the standard DiT path."""
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):                              # t: (B,) integers
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)                           # (B, dim)
```

For *text* conditioning (the move from class-conditional DiT to text-to-image PixArt/SD3), the label embedding is replaced by a pooled text embedding from a CLIP/T5 encoder for the adaLN path, and — in the more powerful designs — the full sequence of text tokens is also fed into the attention via cross-attention or joint attention. That joint-attention generalization is exactly what MM-DiT (SD3) is, and we cover it in [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe). For this post, class-conditional ImageNet is the clean setting where DiT's properties were first measured, so we stay there.

#### Worked example: counting the conditioning parameters

How much does AdaLN-Zero conditioning actually cost in parameters? Each block's adaLN MLP is a single `Linear(dim, 6*dim)`. For DiT-XL with $d=1152$: that's $1152 \times (6\times1152) = 1152\times 6912 \approx 7.96\text{M}$ weights per block, times 28 blocks $\approx 223\text{M}$ parameters. That sounds like a lot — about a third of the 675M total! — and it is the one place AdaLN is *not* cheap in parameter count (though it remains cheap in FLOPs, since it's applied once per block to a single vector $c$, not per token). The paper notes this and a common efficiency variant (used in later models) shares the adaLN MLP across blocks or uses a low-rank factorization to cut that cost. The point stands: AdaLN buys its quality with parameters in the conditioning MLP, not with extra attention FLOPs over the token sequence — which is the right trade, because the token-sequence FLOPs are what blow up with resolution. That's the cost story we turn to next.

## 7. The cost: attention is quadratic in tokens

DiT's clean scalability has a price, and it's the same price every transformer pays: self-attention is quadratic in sequence length. Let's derive it, because the constant matters when you're deciding whether DiT is affordable at a target resolution.

A self-attention layer over $T$ tokens of dimension $d$ does, per the standard count: the QKV projections are $3\times T\times d\times d$ multiply-adds (linear in $T$); the attention score matrix $QK^\top$ is $T\times T\times d$ (this is the **quadratic** term, $T^2 d$); the softmax-weighted value aggregation is another $T^2 d$; the output projection is $T d^2$. The MLP (ratio 4) is $\approx 8 T d^2$. So per block, the dominant terms are

$$
\underbrace{\;\sim 12\,T d^2\;}_{\text{linear in }T:\ \text{QKV, out-proj, MLP}} \;+\; \underbrace{\;\sim 2\,T^2 d\;}_{\text{quadratic in }T:\ \text{attention scores + aggregation}}.
$$

For small $T$ relative to $d$, the *linear* term dominates and the model behaves like a dense net whose cost grows linearly with tokens. But as $T$ grows past $\sim d$, the $T^2 d$ term takes over and cost explodes quadratically. This is the crossover that decides everything about high resolution.

Plug in DiT-XL/2 at 256×256: $T=256$, $d=1152$. The linear term per block $\sim 12 \cdot 256 \cdot 1152^2 \approx 4.1\times10^9$; the quadratic term $\sim 2\cdot 256^2\cdot 1152 \approx 1.5\times10^8$. At this resolution the quadratic term is *small* — only ~4% of the linear cost — because $T=256 \ll d=1152$. This is exactly why DiT at 256×256 is so efficient: the sequence is short enough that attention isn't the bottleneck, and the whole model is a modest 118.6 Gflops per forward pass.

Now scale up. At 512×512, $T=1024$: the quadratic term becomes $\sim 2\cdot 1024^2\cdot 1152 \approx 2.4\times10^9$ per block, now comparable to the linear term. At 1024×1024, $T=4096$: the quadratic term is $\sim 2\cdot 4096^2\cdot 1152 \approx 3.9\times10^{10}$ per block — now it *dominates*, an order of magnitude over the linear cost. The model's compute is now governed by $T^2$, and since $T\propto$ (image side)$^2$, total attention cost scales as (image side)$^4$. That fourth-power scaling is why naive high-resolution DiT is brutally expensive and why the frontier responds with two fixes, both of which shrink $T$ or break the quadratic:

1. **Compress harder in the VAE.** SANA uses a 32× autoencoder instead of 8×, so a 1024×1024 image becomes a 32×32 latent (not 128×128) — back to $T=256$ at $p=2$. The token count, and thus the attention cost, is set by the *latent* resolution, so a deeper AE is the most direct lever. Covered in [the modern recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).
2. **Replace softmax attention with linear attention.** Linear attention (kernel-feature or other variants) computes the same mixing in $O(Td)$ instead of $O(T^2 d)$, removing the quadratic term entirely at some quality cost. SANA again is the production example.

This is the honest trade-off the DiT design rests on: the transformer's uniformity and scalability are wonderful *as long as the sequence is short*, which in latent diffusion it is — that's the entire reason DiT operates on VAE latents and not pixels. Try to run a DiT on raw 256×256×3 pixels with $p=2$ and you'd have $T = 128^2/4 = 4096$ tokens and the fourth-power wall hits immediately. Latent compression isn't an optimization bolted onto DiT; it's the precondition that makes DiT affordable at all. The U-Net, with its conv locality, degrades more gracefully to high resolution — which is the one regime where the U-Net's inductive bias still earns its keep.

Let me stress-test this cost model, because the way it breaks tells you exactly where to intervene. **What happens at 2048×2048?** With an 8× VAE the latent is 256×256, so $T = 256^2/4 = 16{,}384$ tokens, and the attention score matrix alone is $16{,}384^2 \approx 2.7\times10^8$ entries *per head per layer* — at fp16 that single matrix is over half a gigabyte, and you have 16 heads and 28 layers. You will run out of memory long before you run out of patience, and no amount of FlashAttention (which avoids materializing the full matrix but not the $T^2$ *compute*) changes the asymptotics. This is not a tuning problem; it is a structural one, and the structural fix is to shrink $T$: a 32× VAE turns that same 2048×2048 image into a 64×64 latent and $T = 1024$, a 16× reduction in tokens and a 256× reduction in attention compute. **What happens if you instead just make the model smaller** to fit? You move down the scaling curve and your FID gets worse — you've traded the quality DiT was supposed to give you. The lesson the cost model teaches, sharply, is that at high resolution the *first* lever to pull is the autoencoder's compression ratio, not the transformer's size, because the autoencoder sets $T$ and $T$ is what's squared. Reach for linear attention only when you've already compressed as hard as quality allows and still need more headroom. Getting this ordering right — VAE compression first, attention variant second, model shrink last — is the difference between a DiT that scales to 4K and one that OOMs at 1K.

## 8. The scaling story: compute in, FID out

Here is the result that made DiT the default. Peebles & Xie swept four model sizes — S, B, L, XL — and three patch sizes — /8, /4, /2 — and plotted the achieved FID against the model's forward-pass Gflops. The finding: **FID is a clean, smooth, decreasing function of Gflops, and almost nothing else.** Two DiT configs with the same Gflops reach nearly the same FID even if one is "wide and shallow" and the other "narrow with small patches." Compute is the master variable. More Gflops → lower FID, on a tidy trend, with no plateau in the range they explored.

![A matrix comparing DiT-S, B, L, and XL at patch size two across parameters, Gflops at 256 resolution, and ImageNet FID, showing FID falling monotonically as model size and compute rise.](/imgs/blogs/diffusion-transformers-dit-4.png)

Figure 4 lays out the /2 sweep. The numbers (FID-50K on class-conditional ImageNet 256×256, no classifier-free guidance, at the paper's training budget):

| Model | Params | Gflops (256²) | FID-50K (no CFG) |
|---|---|---|---|
| DiT-S/2 | 33M | 6.1 | 68.4 |
| DiT-B/2 | 130M | 23.0 | 43.5 |
| DiT-L/2 | 458M | 80.7 | 23.3 |
| DiT-XL/2 | 675M | 118.6 | 9.62 |

Read the pattern: from S to XL, Gflops rises ~19× and FID falls from 68 to 9.6 — a 7× improvement, monotone, no diminishing returns visible. Hold the *size* fixed and shrink the *patch* (which raises Gflops by adding tokens) and FID falls the same way: DiT-XL/8 → DiT-XL/4 → DiT-XL/2 improves steadily as the patch shrinks and the token count grows. Both knobs — bigger model, more tokens — are just "more compute," and both pay off identically. **That collapse of two different knobs onto one compute axis is the scaling law**, and it's exactly the kind of predictable relationship that LLM practitioners had been exploiting for years. DiT imported it to diffusion.

#### Worked example: reading the scaling law as a budget decision

Say you have a fixed training-compute budget and want the lowest FID. The scaling result tells you: don't agonize over the exact (width, depth, patch) combination — pick whatever lands you at the highest Gflops you can afford to *train to convergence*, because FID tracks Gflops almost regardless of how you spend them. Concretely, if DiT-L/2 (80.7 Gflops, FID 23.3) and a hypothetical "DiT-XL/4" land at similar Gflops, expect similar FID. The practical corollary the paper draws: small models, even trained for many more steps, *cannot* close the gap to a large model — compute spent on a bigger model is more FID-efficient than the same compute spent training a small model longer. This is the diffusion echo of the Chinchilla-era lesson that model size and data/compute must scale together; if you've read the LLM [scaling-laws](/blog/machine-learning/scaling-laws) posts, the shape of the curve will feel familiar, because it is the same shape. The deep reason DiT scales like this and a tuned U-Net doesn't: the transformer has no architectural ceiling you hit — you just add width, depth, tokens — whereas a U-Net's gains saturate as you fight its fixed multi-resolution structure and have to re-tune a dozen interacting hyperparameters at every scale.

A caveat on honesty, since this series insists on it: these FID numbers are at a *specific* training budget (400K–7M steps depending on the figure) and the no-CFG column. The absolute values shift with training length and guidance; what's robust is the *trend* (monotone decrease with Gflops) and the *ordering* (XL ≪ L ≪ B ≪ S). The single most-quoted headline — DiT-XL/2 at **FID 2.27** — comes with classifier-free guidance (scale ≈ 1.5) and the full 7M-step training run, which is why it's lower than the 9.62 no-CFG number above. Always check whether a quoted FID includes guidance; it's the difference between 2.27 and 9.62 for the *same model*.

## 9. DiT versus the U-Net: the head-to-head

The whole point was to beat the convolutional incumbents. Did it? Here is the comparison that retired the U-Net as the *default* diffusion backbone, on class-conditional ImageNet 256×256, FID-50K, all with their respective best guidance settings.

![A before-and-after comparison of the convolutional U-Net backbone against the DiT transformer backbone, contrasting hand-designed multi-resolution stages and skip connections with a uniform stack of patch-token blocks and a lower final FID.](/imgs/blogs/diffusion-transformers-dit-5.png)

Figure 5 frames the architectural contrast; here are the numbers:

| Model | Backbone | Params | FID-50K (256²) | Notes |
|---|---|---|---|---|
| ADM | U-Net (pixel) | 554M | 10.94 | Dhariwal & Nichol 2021, no guidance |
| ADM-G, ADM-U | U-Net (pixel) | 554M+ | 3.94 | with classifier guidance + upsampler |
| LDM-4 | U-Net (latent) | 400M | 10.56 | Rombach et al. 2022, no guidance |
| LDM-4-G | U-Net (latent) | 400M | 3.60 | with classifier-free guidance |
| **DiT-XL/2** | **Transformer (latent)** | **675M** | **2.27** | **with CFG, the new SOTA at the time** |

DiT-XL/2's 2.27 beat the best U-Net result (LDM-4-G's 3.60, ADM-G's 3.94) by a clear margin, and it did so with a *simpler*, more uniform architecture that has one scaling recipe instead of a dozen hand-tuned knobs. Notice both DiT and LDM operate in the same VAE latent space (the 32×32×4 latent) — so this is a clean apples-to-apples *backbone* comparison: same latent, same diffusion objective, swap the U-Net for a transformer, FID drops from 3.60 to 2.27. That controlled comparison is what made the result so persuasive. It wasn't "transformers plus a hundred other changes win"; it was "hold everything fixed, change only the denoiser net, and the transformer wins *and* scales better."

The honest caveats, because they shape when you'd still pick a U-Net:

- **Matched-compute fairness.** DiT-XL/2 is a bit bigger than LDM's U-Net (675M vs 400M). The scaling-law section is the real argument: at *matched* Gflops, the transformer is on a better FID-vs-compute curve, and it keeps improving with scale where the U-Net saturates. But the headline 2.27-vs-3.60 isn't perfectly iso-parameter; it's iso-latent-and-objective. Read it as "the transformer backbone is at least as good and scales better," not "transformers are 1.6× better per FLOP at every size."
- **Convergence cost.** DiT-XL/2's best number comes from a *long* training run (7M steps). Transformers are sometimes slower to get going than a well-tuned U-Net at small scale, precisely because they lack the conv prior — they have to *learn* spatial structure that the U-Net gets for free. At small data/compute, the U-Net's inductive bias can still win. DiT's advantage compounds with scale.
- **High resolution.** As section 7 showed, the U-Net's locality degrades to high resolution more gracefully than DiT's quadratic attention. That's why production DiT models pair the transformer with aggressive latent compression or linear attention — the U-Net's one remaining structural advantage is handled by fixing DiT's sequence length, not by going back to convolutions.

### What each U-Net piece becomes in DiT

It clarifies the swap to map every component of the U-Net denoiser onto its DiT counterpart, because DiT doesn't *delete* the U-Net's jobs — it re-implements each one in the transformer idiom. The U-Net's **ResBlocks** (local feature extraction) become the per-token **MLP** sublayers — both are position-wise nonlinear transforms, except the MLP acts on a patch token instead of a conv feature. The U-Net's **down/up-sampling stages** (multi-resolution processing) are gone entirely; their job — relating information across spatial scales — is handled by the VAE's compression upstream and by global self-attention within the single token resolution. The U-Net's **attention layers** (bolted on only at 16×16 and 8×8 because full-resolution attention was too expensive in pixel space) become the **self-attention** in *every* DiT block, affordable now because the sequence is only 256 tokens. The U-Net's **skip connections** (carrying high-resolution detail across the encoder/decoder bottleneck) have no analog and need none — there is no bottleneck to skip across, because the token resolution never changes through the stack. And the U-Net's **time/condition embedding via AdaGN** (FiLM-style group-norm modulation) becomes **AdaLN-Zero** — the same modulation idea, moved from group norm to layer norm and given the identity-init gate. The mapping is almost one-to-one in *function*, and radically simpler in *form*: a tangle of resolution-specific, skip-connected, asymmetric blocks collapses into a flat stack of identical transformer blocks. That collapse is the architectural essence of DiT.

#### Worked example: estimating latency on an A100

Put rough numbers on what DiT-XL/2's 118.6 Gflops means in wall-clock time, because "Gflops" is abstract until it's seconds. An A100 (80GB) delivers on the order of 300 TFLOP/s in fp16 for matmul-heavy workloads in practice (well below its 312 TFLOP/s peak, since real kernels don't hit peak). A single DiT-XL/2 forward pass at 256×256 is ~118.6 Gflops $= 0.119$ TFLOP, so one forward pass is roughly $0.119 / 300 \approx 0.4$ ms of pure compute (in reality a few ms once you count overhead, kernel launches, and the memory-bound bits). With classifier-free guidance you run the model *twice* per step (conditional + unconditional), so ~2× that. Sampling with a 25-step DPM-Solver is then $25 \times 2 \approx 50$ forward passes $\approx$ a couple hundred milliseconds to a second per image on an A100, batch size 1 — fast enough for interactive use, and a good chunk of why DiT at 256×256 is so practical. Now contrast 1024×1024 ($T=4096$, ~16× the tokens, and the quadratic term now dominant): the per-pass cost balloons by well over an order of magnitude, and the same 50-pass sampling loop that was sub-second at 256² becomes many seconds — the exact wall the frontier addresses with deeper VAE compression. The takeaway you can act on: at 256×256 DiT is cheap enough to ignore the cost; at ≥1024×1024 the cost model in section 7 is the thing you must engineer around before anything else.

## 10. Using DiT in 🤗 diffusers

You don't have to implement DiT from scratch to use it — the 🤗 `diffusers` library ships the class-conditional DiT and its descendants. Here's the original DiT for ImageNet, via `DiTPipeline`:

```python
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

# DiT-XL/2 at 256x256, class-conditional ImageNet, pretrained by the authors
pipe = DiTPipeline.from_pretrained(
    "facebook/DiT-XL-2-256", torch_dtype=torch.float16
).to("cuda")
# swap in a fast multistep solver (see the samplers post for why this is safe)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# generate a few ImageNet classes: 207=golden retriever, 387=red panda, 88=macaw
class_ids = pipe.get_label_ids(["golden retriever", "red panda", "macaw"])
images = pipe(
    class_labels=class_ids,
    guidance_scale=4.0,          # CFG; DiT's quality knob
    num_inference_steps=25,      # 25 DPM-Solver steps is plenty
    generator=torch.manual_seed(0),
).images
images[0].save("dit_golden_retriever.png")
```

The denoiser inside that pipeline is a `DiTTransformer2DModel` — the `diffusers` name for the architecture we built in sections 2–6. You can instantiate it directly to see the exact config of DiT-XL/2:

```python
from diffusers import DiTTransformer2DModel

dit = DiTTransformer2DModel(
    sample_size=32,          # latent spatial size (256/8)
    patch_size=2,            # the "/2": 256 tokens
    in_channels=4,           # VAE latent channels
    num_layers=28,           # depth
    num_attention_heads=16,
    attention_head_dim=72,   # 16 * 72 = 1152 = hidden size
    num_embeds_ada_norm=1000 + 1,  # ImageNet classes + null for CFG
    norm_type="ada_norm_zero",     # <-- AdaLN-Zero, the key flag
)
print(sum(p.numel() for p in dit.parameters()) / 1e6, "M params")  # ~675M
```

The `norm_type="ada_norm_zero"` flag is literally the AdaLN-Zero conditioning we derived in section 4; `diffusers` exposes it as a first-class option because it's that important. For the text-to-image successor, the SD3 backbone is `SD3Transformer2DModel`, which is DiT generalized to MM-DiT (joint text-image attention plus flow-matching velocity prediction):

```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_model_cpu_offload()   # fits the T5 + MM-DiT on a 24GB card

image = pipe(
    "a red panda astronaut floating in a nebula, cinematic, 8k",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save("sd3_panda.png")
```

The trunk of that pipeline — `pipe.transformer`, an `SD3Transformer2DModel` — is a direct descendant of the DiT block you implemented: same patchify, same AdaLN-style conditioning, same uniform stack of attention+MLP blocks, generalized to two modalities and trained with a flow-matching objective ([flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) covers the objective change; [the modern recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) covers the MM-DiT generalization). FLUX's `FluxTransformer2DModel` is the same lineage with double-stream and single-stream block variants. Once you understand the DiT block, you understand the trunk of every frontier text-to-image model shipping in 2025–2026.

## 11. Training a DiT: the practical step

The training loop is the standard diffusion loop — DiT changes the network, not the objective — but a few DiT-specific details matter for reproducing the paper's stability and numbers. Here's a complete training step:

```python
import torch
import torch.nn.functional as F

def dit_training_step(model, vae, batch, scheduler, label_drop=0.1):
    images, labels = batch                       # images:(B,3,256,256)
    with torch.no_grad():
        # encode to the 32x32x4 latent; scale per the VAE's convention
        z0 = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

    # classifier-free guidance: randomly null the label ~10% of the time
    null_id = model.config.num_embeds_ada_norm - 1   # the extra null row
    drop = torch.rand(labels.shape[0], device=labels.device) < label_drop
    labels = torch.where(drop, torch.full_like(labels, null_id), labels)

    # sample a timestep and noise; build z_t with the closed-form marginal
    t = torch.randint(0, scheduler.config.num_train_timesteps,
                      (z0.shape[0],), device=z0.device)
    noise = torch.randn_like(z0)
    z_t = scheduler.add_noise(z0, noise, t)

    # predict; DiT-XL outputs 8 channels: split into eps and the Sigma logits
    pred = model(z_t, t, labels).sample           # (B, 8, 32, 32)
    eps_pred, _sigma = pred.chunk(2, dim=1)        # learn-sigma: keep eps half

    loss = F.mse_loss(eps_pred, noise)             # L_simple
    return loss
```

The DiT-specific notes: (1) **No weight decay** on the model — the paper trains with AdamW at a constant `1e-4` learning rate and *zero* weight decay, which is unusual but matters. (2) **EMA of the weights** with decay 0.9999 — all reported FIDs use the EMA model, not the raw weights, and the gap is large. (3) **The constant LR with no warmup** works precisely *because* of AdaLN-Zero: the identity init means you don't need a warmup to avoid early divergence; the model starts as a stable no-op and ramps itself. (4) The `learn_sigma` output means the loss on ε uses only half the channels; the variance channels are trained with the full variational bound term if you want the better log-likelihood, or you can drop them and predict ε only. These four together — no weight decay, EMA, constant LR, AdaLN-Zero — are why DiT training is famously *boring*: it just goes down, no babysitting, which is the whole promise of importing the transformer recipe.

#### Worked example: the cost of a DiT-XL/2 training run

What does it actually take to reproduce DiT-XL/2's 2.27? The paper trains for ~7M steps at batch size 256, which is on the order of $7\text{M}\times 256 \approx 1.8$ billion images seen. On a node of 8×A100, the original training took on the order of weeks. At a rough cloud rate of \$2/hr per A100, an 8-GPU run over ~2 weeks is on the order of \$5,000–\$10,000 — a real but not absurd number, and the reason most people *fine-tune* a released DiT/SD3 checkpoint rather than train from scratch. The headline lesson for your own budget: the 2.27 figure is a long-training, EMA, with-CFG result; a 400K-step run gets you the 9.62 no-CFG number, which is perfectly good for most purposes and an order of magnitude cheaper. Decide which number you actually need before you light up a GPU cluster, because the last ~7 FID points cost most of the money.

## 12. Why this generalized: the trunk under SD3, FLUX, and Sora

The reason DiT matters far beyond its own ImageNet numbers is that it turned out to be the *right trunk* for everything that came after. Once you have a denoiser that (a) is a plain transformer, (b) scales cleanly with compute, (c) handles variable resolution as just more tokens, and (d) conditions through cheap AdaLN — you have a backbone you can grow in every direction the field wanted to grow.

![A timeline tracing the backbone from U-Net diffusion through DiT and its adaln-zero design to PixArt, SD3 MM-DiT, FLUX, and Sora video generation, all building on the transformer trunk.](/imgs/blogs/diffusion-transformers-dit-8.png)

Figure 8 traces the lineage. **PixArt-α** (2023) was the first to show DiT could be a strong *text-to-image* model, not just class-conditional, by swapping the class embedding for T5 text features fed through cross-attention — and it did so at a fraction of Stable Diffusion's training cost, leaning on DiT's compute efficiency. **SD3** (Esser et al. 2024) generalized the DiT block into **MM-DiT**: instead of conditioning text through AdaLN only, it gives text and image tokens *separate* weight streams that attend to each other in joint attention, so text and image co-evolve through the stack. **FLUX** (Black Forest Labs 2024) took the same MM-DiT lineage to 12B parameters with double-stream and single-stream block designs and became the open-weights quality leader. **Sora** (OpenAI 2024) applied the exact DiT recipe to *video*: patchify spacetime into tubelets, run a transformer over the much longer token sequence, and the same compute↔quality scaling that DiT demonstrated for images held for video. Every one of these is "DiT plus a generalization," and none of them would have been the obvious move if DiT hadn't first proved that a plain transformer scales for diffusion. The U-Net could not have grown into any of them without re-deriving its multi-resolution structure for each new modality and scale; the transformer just needed more tokens and more layers.

This is the deepest reason the field switched. It's not only that DiT-XL/2 hit a lower FID on one benchmark. It's that DiT replaced a *bespoke, hand-tuned, modality-specific* architecture (the U-Net) with a *uniform, scalable, modality-agnostic* one (the transformer), and in doing so let image generation inherit the entire decade of transformer scaling, infrastructure, and tooling that the LLM world had built — distributed training, FlashAttention, tensor/sequence parallelism, the scaling-law methodology, the whole stack. That inheritance, more than any single FID, is why DiT won.

It's worth closing the loop on the series spine, the **generative trilemma** (quality × diversity × speed), because DiT's contribution lands squarely on one axis and leaves the others for later tracks. DiT improves *quality at a given compute budget* and, more importantly, makes quality *predictably scalable* — push more compute through the transformer and FID falls on a known curve, which is the single most valuable property when you're deciding how to spend a training budget. It does *not*, by itself, touch sampling *speed*: a DiT still needs the same 20–50 sampler steps as a U-Net diffusion model, because the number of denoising steps is a property of the *sampler and objective*, not the *backbone* (that's the territory of [DDIM](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling), flow matching, and the distillation track). And DiT keeps diffusion's strong *mode coverage* — the full-distribution property that GANs sacrificed — because it changes only the network, not the likelihood-based training that gives diffusion its coverage. So in trilemma terms, DiT is a *quality-and-scalability* move that is neutral on speed and coverage: it makes the denoiser box better and more scalable without disturbing the rest of the stack. That clean separation — improve one box, leave the interfaces intact — is exactly why DiT slotted into the existing diffusion pipeline so frictionlessly, and why every speed and control technique built for U-Net diffusion (samplers, CFG, ControlNet-style conditioning) transferred onto DiT with minimal change.

## 13. When to reach for DiT (and when not to)

Architecture choices are trade-offs, and the honest answer is that DiT is not free or universally best. Here's the decision.

**Reach for a DiT / transformer backbone when:**

- You're operating in a **compressed latent space** (8× VAE or deeper), so the token sequence is short ($T \lesssim 1024$) and the quadratic attention term isn't the bottleneck. This is the regime where DiT shines and where every frontier model lives.
- You want to **scale** — bigger models, more data, more compute, multiple resolutions and aspect ratios — and want *predictable* returns. The compute↔FID law and the "just add tokens/width/depth" recipe are DiT's killer feature.
- You want to **inherit the transformer ecosystem**: FlashAttention, `torch.compile`, sequence/tensor parallelism, LLM-style training infra, and the option to extend to text (MM-DiT) or video (Sora-style) without redesigning the backbone.
- You're building **text-to-image or text-to-video at the frontier** in 2025–2026. The answer here is just "yes" — SD3, FLUX, PixArt, Sora are all DiT-lineage, and that's where the tooling and checkpoints are.

**Stick with (or consider) a U-Net when:**

- You're at **small scale / small data**, where the conv locality prior is a genuine sample-efficiency advantage and the transformer's lack of inductive bias means it converges slower or needs more data to match.
- You must run at **high pixel resolution without aggressive latent compression**, where DiT's $T^2$ attention cost (scaling as the fourth power of image side) becomes prohibitive and the U-Net's local convolutions degrade more gracefully. (The frontier's real answer is "use DiT but compress harder," but if you can't change the VAE, the U-Net's locality is a legitimate fallback.)
- You're **fine-tuning an existing U-Net model** (SD1.5, SDXL) where a vast ecosystem of LoRAs, ControlNets, and community tooling already exists. The architecture you *should* have started with and the architecture with the tooling you need today are sometimes different; don't rebuild SDXL's ecosystem on a DiT just for architectural purity.

The non-obvious failure mode to watch: **running DiT on too long a sequence.** If you naively push DiT to 1024×1024 generation with an 8× VAE ($T=4096$), you hit the quadratic wall and your latency and memory explode — and the fix is *not* "go back to a U-Net," it's "compress the latent more (32× AE) or use linear attention so $T$ or the $T^2$ term shrinks." Getting this wrong — treating DiT's quadratic cost as a reason to abandon transformers rather than a reason to manage sequence length — is the single most common architectural misstep I see. The transformer is right; the sequence length is the thing to engineer.

## 14. Case studies: the real numbers

Four concrete results from the literature, stated with their conditions so you can trust them.

**DiT-XL/2 on ImageNet 256×256 (Peebles & Xie 2023).** FID-50K **2.27** with classifier-free guidance (scale ≈ 1.5) and EMA weights after the full ~7M-step training run; **9.62** without guidance at 400K steps. 675M parameters, 118.6 Gflops per forward pass, 256 tokens, 28 layers, hidden 1152. This beat ADM-G (U-Net, 3.94) and LDM-4-G (U-Net, 3.60) — the first time a transformer backbone took the class-conditional ImageNet diffusion crown. The same paper reports DiT-XL/2 at 512×512 reaching FID 3.04 with guidance, again state-of-the-art at the time.

**The AdaLN-Zero ablation (same paper).** Holding DiT-XL/2 fixed and changing only the conditioning mechanism, AdaLN-Zero (FID 9.62, no CFG) beat plain AdaLN (~19.5), cross-attention (~28), and in-context tokens (~35) — the cheapest mechanism was also the best, and identity initialization roughly halved AdaLN's FID. This is the result that established AdaLN-Zero as the default and made `norm_type="ada_norm_zero"` a standard knob.

**SD3 / MM-DiT (Esser et al. 2024).** Generalizing DiT to MM-DiT (separate text/image weight streams, joint attention) plus a flow-matching objective, SD3 scales from 800M to 8B parameters and shows the *same* clean scaling behavior DiT demonstrated — validation loss and human-preference scores improve smoothly with model size, with no plateau in their range. SD3-8B leads on prompt-following benchmarks (GenEval). The lesson DiT taught — transformers scale predictably for diffusion — held at the 8B scale and on text-to-image, not just class-conditional 675M.

**Sora (OpenAI 2024).** The clearest proof of DiT's generality: applying the patchify-then-transformer recipe to *video* (spacetime patches), Sora's technical report explicitly frames the model as a diffusion transformer and shows sample quality improving with training compute on the same kind of curve DiT showed for images. A U-Net would have required re-engineering its multi-resolution structure for the temporal dimension; the transformer just took longer token sequences. This is the payoff of a modality-agnostic backbone, and it's why "DiT" is now shorthand for the whole approach, not one ImageNet model.

## 15. Key takeaways

- **DiT replaces the U-Net denoiser with a plain transformer**: patchify the latent into tokens, run a uniform stack of self-attention + MLP blocks, unpatchify back. Same diffusion objective, different network.
- **Patchify sets everything.** Token count is $T = HW/p^2$; patch size $p$ trades spatial granularity for sequence length. The canonical DiT-XL/2 is patch size 2 over a 32×32 latent → 256 tokens.
- **AdaLN injects conditioning by modulating layer norm** with scale/shift/gate computed from the timestep + label — cheaper than cross-attention because image tokens never attend to condition tokens.
- **AdaLN-Zero is the load-bearing trick.** Zero-initialize the residual gates so every block starts as the identity; the model begins as a stable no-op and ramps each block off zero, which roughly halves FID at scale and removes the need for LR warmup.
- **The cost is quadratic attention**, $\sim T^2 d$. Cheap at 256 tokens (attention is ~4% of FLOPs), brutal at high resolution (cost scales as the fourth power of image side). The fix is harder latent compression or linear attention, *not* abandoning the transformer.
- **DiT scales cleanly: more Gflops → lower FID**, on a tidy trend with no plateau, and compute (not the specific width/depth/patch split) is the master variable. DiT-XL/2 hit FID 2.27 (with CFG), beating the best U-Nets.
- **The transformer's win is the ecosystem.** Importing the LLM scaling recipe and infrastructure — and a modality-agnostic backbone — is why SD3 (MM-DiT), FLUX, PixArt, and Sora all build on DiT, and why the U-Net stopped being the default.
- **Pick the U-Net only** at small scale/data (conv prior helps), at high resolution without latent compression, or when you need an existing model's LoRA/ControlNet ecosystem.

## 16. Further reading

- **Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), ICCV 2023** — the source paper. The patchify, AdaLN-Zero, and Gflops↔FID scaling results all come from here. Read the ablation tables directly.
- **Dosovitskiy et al., "An Image Is Worth 16×16 Words" (ViT), 2020** — where patchify comes from; DiT applies it to noised latents.
- **Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," 2018** — the feature-wise modulation idea that AdaLN generalizes.
- **Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), 2024** — the MM-DiT generalization of DiT and the flow-matching objective.
- **Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (ADM), 2021** — the strong U-Net baseline DiT had to beat; also the source of learn-sigma and classifier guidance.
- **🤗 `diffusers` documentation** — `DiTTransformer2DModel`, `SD3Transformer2DModel`, `FluxTransformer2DModel`, and the `DiTPipeline` / `StableDiffusion3Pipeline` APIs used above.
- **Within this series:** [latent diffusion and stable diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) (the latent space DiT runs in), [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet) (the incumbent it replaced), [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) and [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) (where DiT goes next), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
