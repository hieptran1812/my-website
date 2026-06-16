---
title: "Personalization: Textual Inversion, DreamBooth, and LoRA for Your Own Subjects"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Teach a frozen text-to-image model a brand-new subject or style from a handful of photos — derive the Textual Inversion objective, DreamBooth's prior-preservation loss, and the LoRA low-rank decomposition, then fine-tune one cleanly in diffusers and peft with a decision guide for picking among them."
tags:
  [
    "image-generation",
    "diffusion-models",
    "lora",
    "dreambooth",
    "textual-inversion",
    "personalization",
    "peft",
    "fine-tuning",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/personalization-dreambooth-textual-inversion-lora-1.png"
---

You have five photos of your dog. You want a model that can put *that* dog — same brindle coat, same crooked ear — onto a beach, into a Renaissance oil painting, riding a tiny motorcycle. Stable Diffusion already knows what "a dog on a beach" looks like. It has seen millions of dogs. What it has never seen is *your* dog, and no prompt you can type will conjure those exact markings, because the information simply is not in the weights.

This is the personalization problem, and it sits at the heart of every custom avatar generator, every artist's signature-style model on Civitai, every "make me as an astronaut" app. The base model is a vast, frozen reservoir of visual knowledge. You have a tiny puddle of new information — three to five images — and you need to pour it in without spilling, without forgetting, and ideally without copying a 7 GB checkpoint around to share it. The art is doing surgery on a model that cost millions of dollars to train using a training set you could fit in an email.

There are three classic ways to do it, and they span four orders of magnitude in what they touch. **Textual Inversion** freezes the entire model and learns a single new word — one embedding vector, a few kilobytes — that points at your subject. **DreamBooth** fine-tunes the whole denoiser and adds a clever second loss to keep it from forgetting what a "dog" is in general. **LoRA** threads tiny low-rank matrices into the attention layers and trains only those, landing in the sweet spot: megabytes, not gigabytes, with most of DreamBooth's fidelity. By the end of this post you will understand the math behind each, be able to run a LoRA fine-tune in 🤗 `diffusers` and `peft`, read the rank-versus-quality trade-off off a table, and pick the right tool without guessing. Figure 1 lays out the three methods on the axes that actually decide which one you reach for.

![Matrix comparing Textual Inversion, DreamBooth, and LoRA across trainable size, fidelity, and portability](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-1.png)

This connects directly to the recurring frame of the series. Personalization is the diffusion stack — data to VAE-latent to denoiser to sampler to guidance to image — with a thumb pressed on exactly one stage. Textual Inversion edits the *conditioning* (a new text embedding). DreamBooth and LoRA edit the *denoiser* itself (the U-Net or DiT). If you have read [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet) post you already know where the attention layers live; this post is largely about injecting a few new parameters right there. If you want the foundations under all of it, [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) sets up the noise-prediction loss we will keep reusing as the training signal.

## 1. The shared setup: every method optimizes the same diffusion loss

Before the three methods diverge, they share one engine. All of them are trained with the ordinary denoising objective — the same loss the base model was pretrained on. Recall the simplified DDPM loss. You take a clean image, encode it to a latent $z_0$ (latent diffusion works in the VAE's compressed space, not pixels), sample a timestep $t$, sample Gaussian noise $\epsilon$, and form the noised latent

$$
z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,
$$

where $\bar\alpha_t$ is the cumulative noise schedule. The network $\epsilon_\theta$ — conditioned on the timestep and on a text embedding $c$ — tries to predict the noise that was added, and we minimize

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{z_0, c, \epsilon, t}\Big[\, \big\| \epsilon - \epsilon_\theta(z_t, t, c) \big\|_2^2 \,\Big].
$$

Nothing here is new — it is the loss derived in [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm). What changes between the three personalization methods is *which variables in this expression carry gradients*:

- **Textual Inversion** holds $\theta$ fixed and lets gradients flow only into a small part of $c$ — specifically the embedding of one new token.
- **DreamBooth** lets gradients flow into all of $\theta$ (the full network), and adds a second copy of $\mathcal{L}_\text{simple}$ over generated class images.
- **LoRA** holds $\theta$ fixed but adds a small set of new trainable parameters $\Delta\theta = BA$ inside the attention projections, and gradients flow only into $B$ and $A$.

That is the whole story at the level of the loss. The differences in fidelity, size, and failure modes all fall out of those three choices. Let me take them one at a time, deriving the *why* for each, then we will write code and look at numbers.

A note on the running example. Throughout I will use the canonical DreamBooth setup: a subject (a dog) labeled by a rare token, and a class noun ("dog"). The rare token in the original paper is `sks` — deliberately chosen because it is an obscure sequence the tokenizer barely uses, so it carries almost no prior meaning to overwrite. We will generate "a photo of `sks` dog" and want it to render *our* dog. The same recipe personalizes a face, a product, an art style, or a font; only the captions change.

### A small but important point: personalization is not "training a model"

It is worth being precise about what we are *not* doing, because the mental shortcut "I'll train a model on my photos" leads people astray. We are not training a generative model from scratch — that takes hundreds of millions of images and a cluster, and five photos would produce nothing but memorized noise. We are *adapting* a model that already knows almost everything. The base model has, in its weights, the visual grammar of fur, skin, lighting, lenses, perspective, and ten thousand dog breeds. Your contribution is a tiny correction: "of all the dogs you can draw, learn to draw *this* one when you see this word." Personalization is a steering operation on a pretrained prior, not a learning-from-data operation. This reframing explains every property we will observe — why so few images suffice (you are selecting, not learning, most of the content), why overfitting is so easy (you have far more model capacity than data), and why methods that change less (Textual Inversion, LoRA) forget less (they perturb the prior less).

### Where the gradient actually goes

The other shared fact worth internalizing is the *gradient path*. In all three methods, the loss is computed at the U-Net's output (predicted noise versus true noise), and gradients flow backward through the entire computation graph: from the loss, back through the U-Net's layers, back through the cross-attention that injected the text condition, back through the text encoder, all the way to the embedding table. This full path is differentiable. The *only* thing that distinguishes the methods is where, along this path, we have placed trainable parameters and where we have called `requires_grad_(False)`. Textual Inversion puts the one trainable parameter at the very start of the path (the embedding); DreamBooth makes the whole U-Net trainable in the middle; LoRA inserts small trainable detours inside the attention blocks. The backward pass is identical machinery in every case — `loss.backward()` — and PyTorch simply accumulates gradients onto whichever tensors have `requires_grad=True`. Hold that picture and the rest is detail.

## 2. Textual Inversion: learn a word, freeze the model

The cleverest thing about Textual Inversion (Gal et al., 2022, *"An Image is Worth One Word"*) is what it refuses to touch. The U-Net, the VAE, the text encoder — all frozen, every weight exactly as pretrained. The only thing that learns is a single vector in the text encoder's embedding table.

Here is the mechanism. A text encoder like CLIP turns a prompt into tokens, looks each token up in an embedding table $E \in \mathbb{R}^{V \times d}$ (vocabulary size $V$, embedding dimension $d$, typically 768), and feeds those embeddings through transformer layers to produce the conditioning $c$. Textual Inversion adds one new row to that table — a pseudo-word we will write $S_*$ — and initializes it (often from the embedding of a coarse describing word like "dog" or "toy"). Then it optimizes *only that row*:

$$
v_* = \arg\min_{v}\; \mathbb{E}_{z_0, \epsilon, t}\Big[\, \big\| \epsilon - \epsilon_\theta\big(z_t, t, \, c_\phi(\text{"a photo of } S_* \text{"}, v)\big) \big\|_2^2 \,\Big],
$$

where $c_\phi$ is the frozen text encoder with parameters $\phi$, and $v$ is the embedding we slot in for $S_*$. The gradient $\nabla_v \mathcal{L}$ is computed by ordinary backprop — it flows from the diffusion loss, back through the frozen U-Net, back through the frozen text encoder, and lands on $v$. Everything in the path is differentiable; we just zero out (or never create) gradients for everything except $v$. Figure 2 shows the flow: photos go in as the denoising target, the new token rides through the frozen encoder as the condition, and the loss back-propagates to exactly one vector.

![Stack showing Textual Inversion freezing the encoder and U-Net while gradients update only the new token embedding](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-2.png)

### Why this is so cheap — and so limited

The learned artifact is one vector of dimension $d = 768$, stored in `float32`, which is $768 \times 4 = 3072$ bytes — about 3 KB. You can attach it to any prompt: "a watercolor of $S_*$", "$S_*$ wearing a tiny hat". It composes with the model's full prior because the model never changed; you only added a word it can now use.

The limitation is equally direct, and it is a capacity argument. You are asking $768$ numbers to capture everything that distinguishes your subject from the class average. That is enough to nudge color, broad shape, and overall vibe — Textual Inversion is genuinely good at *styles* and at subjects that are "a recognizable variation on a common thing." It is not enough to pin down a precise face or fine texture, because the rest of the network — the part that actually draws fur and reflections — is frozen and was never told about your specific subject. There is simply no place to store high-frequency identity detail. In practice Textual Inversion embeddings often drift toward "the right palette and silhouette" while missing the exact identity, and pushing training longer overfits the embedding to the few training poses rather than improving identity.

#### Worked example: the bytes and the capacity ceiling

Take SD 1.5. Its CLIP text encoder has $d = 768$. A Textual Inversion embedding is one such vector: 3 KB on disk (some implementations learn 2–8 vectors per concept to add headroom, so 6–24 KB). Compare that to the U-Net, which is about **860M** parameters, roughly **1.7 GB** in `float16`. So Textual Inversion trains a fraction $768 / 860{,}000{,}000 \approx 9 \times 10^{-7}$ of the model — under one part in a million. It is astonishing it works at all, and the reason it does is that the frozen model already knows how to draw almost everything; the embedding just has to *select* a region of the model's existing capability. When the subject lives outside that region (a face the model has truly never approximated), 768 numbers cannot get you there, and you graduate to DreamBooth or LoRA.

Training is fast and light: a few thousand steps, single-digit GB of VRAM, ten to thirty minutes on a consumer GPU. You pay almost nothing and you get a portable word. Reach for it when the budget is *style* or *broad concept*, not photoreal identity.

### The optimization, in code

The training loop for Textual Inversion is the same denoising loop we will write for LoRA, with one difference: the trainable parameter is the embedding, and nothing else. The crux is wiring the optimizer to a *single row* of the embedding table while leaving the rest frozen:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
)
tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder

# 1. Add the placeholder token and resize the embedding table by one row.
placeholder = "<my-subject>"
num_added = tokenizer.add_tokens(placeholder)            # returns 1
text_encoder.resize_token_embeddings(len(tokenizer))
placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)

# 2. Initialize the new row from a coarse describing word ("toy", "dog", ...).
init_id = tokenizer.convert_tokens_to_ids("toy")
embeds = text_encoder.get_input_embeddings().weight.data
embeds[placeholder_id] = embeds[init_id].clone()

# 3. Freeze the whole model. Only the embedding matrix carries grad,
#    and we will mask the gradient to the single new row each step.
for p in text_encoder.parameters():
    p.requires_grad_(False)
text_encoder.get_input_embeddings().weight.requires_grad_(True)
```

During each step, after `loss.backward()`, you zero out the gradient for *every* row except the placeholder before calling `optimizer.step()` — that mask is what keeps Textual Inversion from accidentally drifting the rest of the vocabulary:

```python
# After backward(), keep the gradient ONLY for the new token's row.
grad = text_encoder.get_input_embeddings().weight.grad
mask = torch.ones(grad.shape[0], dtype=torch.bool)
mask[placeholder_id] = False
grad[mask] = 0.0          # all other rows: no update
optimizer.step()
optimizer.zero_grad()
```

That gradient mask is the entire enforcement of "freeze everything but one word." 🤗 `diffusers` ships `textual_inversion.py` that does exactly this with mixed precision and EMA on the embedding; the snippet above is its core so you can see there is no magic.

### Multi-vector Textual Inversion and its ceiling

A natural extension when one vector is not enough: learn *several* placeholder vectors for the same concept ("`<my-subject>`" expands to 2–8 embedding rows that are inserted together). This multiplies capacity linearly — eight vectors is $8 \times 768$ numbers, ~24 KB — and noticeably improves fidelity over a single vector, at the cost of eating more of the 77-token prompt budget and being slightly harder to compose. But there is a hard ceiling no amount of vectors can break: every Textual Inversion vector still has to express itself *through the frozen network*. If the model's weights cannot draw a particular texture or a particular face geometry, no input embedding can summon it, because the drawing happens downstream in frozen layers. This is precisely the wall that DreamBooth and LoRA climb over by making the *network* trainable — which is where we go next.

## 3. DreamBooth: fine-tune the model, but stop it from forgetting

DreamBooth (Ruiz et al., 2023) takes the opposite bet. Instead of freezing the model and learning a word, it fine-tunes the *whole denoiser* (or at least the U-Net) on your handful of images, binding them to a rare token. Because the network itself updates, it can store the high-frequency detail Textual Inversion could not — the exact face, the exact texture. The fidelity is the best of the three. But fine-tuning a giant model on five images invites two disasters, and DreamBooth's real contribution is the trick that prevents the worse one.

### The two failure modes of naive fine-tuning

**Overfitting.** Five images is nothing. Train long enough on them and the model memorizes the *poses and backgrounds*, not just the subject — ask for "$sks$ dog on the moon" and you get your dog in the exact crouch from photo #3, on your living-room rug, with the moon pasted behind.

**Language drift.** This is the subtle one. You are binding the subject to "$sks$ dog." But every gradient step that sharpens "$sks$ dog" also tugs on the shared weights that draw *all* dogs — and the word "dog" itself. After enough steps the model's general notion of "dog" collapses toward *your* dog. Prompt "a dog" and you no longer get a random dog; you get yours, or some mangled average. The class prior has drifted. The model forgot what dogs in general look like. This is a specific instance of **catastrophic forgetting**, and it is what makes naive full fine-tuning on a tiny set so destructive.

### The prior-preservation loss

DreamBooth's fix is **class-prior-preservation**. Before training, you ask the *frozen base model* to generate a few hundred images of the generic class — "a photo of a dog" — using its own prior. These become a second, self-supervised dataset. During training you optimize two losses at once:

$$
\mathcal{L}_\text{DreamBooth} = \underbrace{\mathbb{E}\big[\| \epsilon - \epsilon_\theta(z_t, t, c_\text{inst}) \|^2\big]}_{\text{instance loss: learn your subject}} \;+\; \lambda\, \underbrace{\mathbb{E}\big[\| \epsilon' - \epsilon_\theta(z'_{t'}, t', c_\text{cls}) \|^2\big]}_{\text{prior loss: don't forget the class}},
$$

where $c_\text{inst}$ is the conditioning for "a photo of $sks$ dog" (your images $z$), and $c_\text{cls}$ is "a photo of a dog" (the *model-generated* class images $z'$). The prior term has a beautiful logic: it asks the model, while it is busy learning your dog, to *keep reproducing its own original outputs* for the generic class. It anchors "dog" to where the base model put it, so the gradient updates from your five images cannot wander the class meaning off its mark. The weight $\lambda$ (commonly 1.0) balances the two; the number of class images (200–1000) sets how strong the anchor is. Figure 3 shows the two branches merging into the combined loss that then updates the full U-Net.

![Graph showing DreamBooth combining an instance loss and a class-prior-preservation loss into a total loss that fine-tunes the full U-Net](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-3.png)

### Why a *rare* token matters

The choice of `sks` (or any low-frequency token) is not arbitrary. If you bind your subject to a common word — say you used "dog" itself as the trigger — you would be overwriting a heavily-used, richly-connected embedding, and the damage to the model's general competence would be severe. A rare token is a nearly-blank slot: it has weak associations to overwrite, so binding your subject to it disturbs the rest of the model's language minimally. You want the trigger word to be a fresh hook, not a load-bearing beam.

### Generating the class images, in code

The prior-preservation dataset is self-supervised — the model generates it from itself before training:

```python
# One-time, before training: sample class images from the FROZEN base model.
prompts = ["a photo of a dog"] * 200
class_images = []
for i in range(0, len(prompts), 8):
    batch = pipe(prompts[i:i+8], num_inference_steps=30,
                 guidance_scale=7.5).images
    class_images.extend(batch)
# Save to ./dog_class — these anchor the class meaning during fine-tuning.
```

These 200 images are not your dog and not curated — they are the model's *own idea* of "a dog," and that is the point: by continuing to reproduce them during training, the model is forced to keep "a dog" pointing where it already pointed. You generate them once, then reuse them. More class images is a stronger anchor (less drift) but slower training; 200–1000 is the usual range, with $\lambda = 1.0$ weighting the prior term equally against the instance term.

### Why prior preservation is the *right* fix, not just *a* fix

It is worth seeing why this specific construction prevents drift rather than merely diluting it. Without the prior term, the only gradient signal is "make `sks` dog look like these five photos," and the shortest path to that for a high-capacity model runs straight through the shared "dog" machinery — the model is happy to repurpose the general dog circuitry for your dog because nothing penalizes it for doing so. The prior term installs exactly that penalty: every step, the model is *also* scored on reproducing generic dogs, so any update that degrades general dogs is now penalized in proportion. The two gradients pull against each other, and the equilibrium is "learn the new subject *in the spare capacity*, leave the general class intact." That is the behavior you want, and it falls directly out of adding a self-distillation term on the class. It is a small, elegant idea — use the model's own outputs as a regularizer — and it generalizes well beyond DreamBooth.

### The cost

DreamBooth's price is everything Textual Inversion saved. You fine-tune the whole U-Net, so the artifact is a full checkpoint — 2 GB (SD 1.5 `float16`) to 7 GB (SDXL). That is what you have to store and share per subject. Training is heavier: you are computing gradients for ~860M (SD 1.5) or ~2.6B (SDXL) parameters, plus generating the class images up front. VRAM climbs (you need gradients and optimizer states for the full model), and you must watch the step count closely — DreamBooth overfits fast, often in 800–1500 steps for a subject. The fidelity is unmatched, but you are carrying a piano to play one tune.

In modern practice almost nobody ships raw DreamBooth checkpoints anymore. Instead they run **DreamBooth's training recipe** (the rare token, the prior-preservation loss) but apply the *weight updates through LoRA* — which is exactly the bridge to the next section. "DreamBooth LoRA" is the single most common personalization workflow today, and it is DreamBooth's loss with LoRA's parameter budget.

## 4. LoRA: the low-rank update that captures fine-tuning cheaply

LoRA — Low-Rank Adaptation (Hu et al., 2021, originally for language models, now the default for diffusion) — is the idea that makes everything practical. Its premise is a hypothesis about fine-tuning itself: when you adapt a big pretrained model to a narrow task, the *change* to the weights, $\Delta W$, is low-rank. You are not rewiring the model; you are nudging it along a few directions. If $\Delta W$ is low-rank, you should not store or train a full dense matrix of updates — you should store its cheap factorization.

### The decomposition

Take any weight matrix $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ in the model — in diffusion, the attention layers' query, key, value, and output projections are the prime targets. Full fine-tuning would learn an update $\Delta W$ of the same shape, costing $d_\text{out} \times d_\text{in}$ parameters. LoRA instead writes the update as a product of two thin matrices:

$$
\Delta W = B A, \qquad B \in \mathbb{R}^{d_\text{out} \times r}, \quad A \in \mathbb{R}^{r \times d_\text{in}}, \quad r \ll \min(d_\text{in}, d_\text{out}).
$$

The adapted layer computes

$$
h = W_0 x + \Delta W x = W_0 x + B A x,
$$

and during training $W_0$ is **frozen** — only $A$ and $B$ get gradients. Figure 4 draws this: the frozen weight and the trainable low-rank detour run in parallel and their outputs add.

![Graph of the LoRA low-rank branch B times A running parallel to the frozen attention weight and summing into the output](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-4.png)

### Why low rank captures fine-tuning so cheaply

Count the parameters. A dense update is $d_\text{out} \times d_\text{in}$. The LoRA factorization is

$$
r \cdot d_\text{in} + d_\text{out} \cdot r = r\,(d_\text{in} + d_\text{out}).
$$

For a square $768 \times 768$ attention projection, dense is $768^2 = 589{,}824$ parameters. With rank $r = 8$ the LoRA update is $8 \times (768 + 768) = 12{,}288$ parameters — a **48×** reduction *on that one matrix*. Across an entire U-Net with hundreds of such projections, that is the difference between a 2 GB checkpoint and a 10–40 MB adapter. The compression is exact in the sense that you are not approximating $W_0$ at all (it is untouched and full-rank); you are only constraining the *update* to lie in a rank-$r$ subspace. The bet is that this subspace is enough to express "draw this subject" — and empirically, for personalization, it overwhelmingly is.

### The intrinsic-rank argument, a little more rigorously

Why should we believe the update is low-rank? The original LoRA paper grounds this in earlier work on the *intrinsic dimension* of fine-tuning: large pretrained models can be adapted to downstream tasks by optimizing in a surprisingly low-dimensional random subspace, which says the task-specific information needed to adapt the model is small. LoRA takes the next step — instead of a random subspace, it learns the *best* rank-$r$ subspace per weight matrix by gradient descent.

There is a clean way to see why a *single subject* needs so little. Think of the full weight matrix $W_0$ as already encoding "how to draw the entire visual world." Teaching it your specific dog does not require touching most of that machinery — the parts that draw fur, eyes, grass, and bokeh are all reusable. The change you need is a narrow re-aiming: shift a handful of internal feature directions so that the trigger token routes to *your* dog's particular combination of features. A narrow re-aiming is, almost by definition, a low-rank perturbation — it lives in a few directions, not all $d^2$ of them. Empirically, ablations in the LoRA paper show that even rank $r=1$ or $r=2$ recovers most of the benefit on many tasks, and that the top singular directions learned at high rank are largely already present at low rank — strong evidence that the useful part of $\Delta W$ genuinely is low-rank, and the rest is noise you are better off not fitting.

This is also why LoRA *forgets less* than full fine-tuning. A full update can move $W_0$ anywhere in $\mathbb{R}^{d_\text{out} \times d_\text{in}}$, including directions that destroy unrelated capabilities. A rank-$r$ update can only move along $r$ directions, so the damage it can do to the rest of the model is bounded by construction. Constraining the update *is* the regularization — it is the same reason a low-rank model overfits less than a full one, applied to the fine-tuning delta rather than the model itself.

### Where LoRA injects in a diffusion model, concretely

In a U-Net (SD 1.5, SDXL) the attention lives in the transformer blocks scattered through the down, mid, and up stages. Each block has **self-attention** (the latent attends to itself, capturing spatial structure) and **cross-attention** (the latent attends to the text tokens, injecting the prompt). Both have query/key/value/output projections — the `to_q`, `to_k`, `to_v`, `to_out` modules we target. Personalization gets most of its leverage from the *cross-attention* projections, because that is where the trigger token's meaning enters the image; but training all attention projections (self and cross) is standard and works well.

In a DiT/MM-DiT backbone (SD3, FLUX) the picture is the same in spirit — the model is a stack of transformer blocks with attention and MLP sub-layers — but the module names differ (`attn.to_q`, `attn.add_q_proj` for the joint text-image attention in MM-DiT, etc.), and FLUX LoRAs commonly target both the attention and the MLP projections because FLUX's blocks carry more of their capacity in the feed-forward path. The principle is unchanged: find the linear layers where conditioning has leverage, wrap each with a rank-$r$ $BA$ detour, and train only those. The U-Net post in this series, [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet), maps out exactly where these attention blocks sit if you want the floor plan.

There are two more details that make LoRA train stably:

- **Initialization.** $A$ is initialized from a small random Gaussian and $B$ is initialized to **zero**. So at step 0, $BA = 0$ and the adapter is a no-op — the model behaves exactly like the frozen base, and training starts from the pretrained solution rather than a random perturbation. This is why LoRA fine-tunes are so well-behaved: you begin at the base and move outward only as the gradient warrants.
- **The scaling factor $\alpha$.** The update is scaled: $h = W_0 x + \frac{\alpha}{r} B A x$. The factor $\alpha/r$ decouples the *learning-rate-like strength* of the adapter from its rank. A common convention is $\alpha = r$ (so the scale is 1) or $\alpha = 2r$. When you load a LoRA at inference you can additionally multiply by a user weight (the "LoRA strength" slider in ComfyUI), turning the effect up or down — which is impossible with a baked-in DreamBooth checkpoint.

### DoRA: a refinement worth knowing

DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al., 2024) splits each weight into a **magnitude** and a **direction**, then applies LoRA only to the direction while learning the magnitude separately: $W = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|}$. The intuition is that full fine-tuning changes both how *much* a weight fires and *which way* it points, while plain LoRA entangles the two; separating them lets DoRA match full fine-tuning more closely at the same rank, at a small extra cost. In `peft` it is a one-flag change (`use_dora=True`). For most personalization jobs plain LoRA is fine; reach for DoRA when you are squeezing the last bit of fidelity at low rank.

### Why LoRA won

LoRA became the default for sharing custom models for reasons that are social as much as technical. The adapter is small enough to host and download freely (megabytes). It is *composable*: because the update is additive ($W_0 + \sum_i w_i B_i A_i$), you can stack multiple LoRAs — a character LoRA plus a style LoRA — at inference, each with its own strength. It is *reversible*: unload the adapter and you have the pristine base back, so one base model serves every personalization. And it is *cheap to train*: only $A$ and $B$ carry gradients and optimizer state, so VRAM drops dramatically versus full fine-tuning, letting people train on consumer 12–24 GB GPUs. None of these is true of a DreamBooth checkpoint, and that is why an entire ecosystem of shared subject and style models is LoRAs.

## 5. The training loop, concretely: a DreamBooth-LoRA fine-tune in diffusers + peft

Enough theory. Here is the loop that runs the dominant workflow — DreamBooth's recipe (rare token, prior preservation optional) delivered through LoRA adapters — using 🤗 `diffusers` and `peft`. Figure 5 is the shape of a single training step before we write it: load a photo, encode and noise it, denoise through the LoRA-wrapped U-Net, and update only the adapter.

![Stack of one LoRA fine-tune training step from sampling a photo through encoding, the LoRA forward pass, the loss, and updating only the adapter](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-5.png)

First, attach LoRA adapters to the U-Net's attention projections with `peft`. The key decisions are the rank, the alpha, and which modules to target.

```python
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
unet, vae, text_encoder = pipe.unet, pipe.vae, pipe.text_encoder
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# Freeze everything; only the LoRA adapters will train.
for module in (unet, vae, text_encoder):
    module.requires_grad_(False)

lora_config = LoraConfig(
    r=16,                 # rank: capacity vs size, see the rank table below
    lora_alpha=16,        # scale alpha/r = 1.0
    init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # attention projections
)
unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()
# e.g. trainable params: 1.6M || all params: 861M || trainable%: 0.19
```

Note the `target_modules`: we inject into the attention query/key/value/output projections, which is where conditioning information is mixed and where personalization has the most leverage. Some recipes also target the feed-forward MLP (`ff.net.0.proj`, `ff.net.2`) for more capacity, or — important for SDXL and SD3 — also train a LoRA on the *text encoder*, which often improves how strongly the trigger token binds. Start with attention-only; add the rest if identity is weak.

Now the training step. This is the loss from Section 1, made executable:

```python
import torch.nn.functional as F

optimizer = torch.optim.AdamW(
    [p for p in unet.parameters() if p.requires_grad], lr=1e-4
)

for step, (images, captions) in enumerate(dataloader):
    # 1. Encode images to latents (VAE is frozen; no grad needed here).
    with torch.no_grad():
        latents = vae.encode(images.half()).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        enc = pipe.tokenizer(captions, padding="max_length",
                             max_length=77, return_tensors="pt").input_ids
        cond = text_encoder(enc.to(latents.device))[0]

    # 2. Sample noise and a timestep, then form the noised latent z_t.
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                      (bsz,), device=latents.device).long()
    noisy = noise_scheduler.add_noise(latents, noise, t)

    # 3. Predict the noise through the LoRA-wrapped U-Net.
    eps_hat = unet(noisy, t, encoder_hidden_states=cond).sample

    # 4. The simple diffusion loss. Only A, B receive gradients.
    loss = F.mse_loss(eps_hat.float(), noise.float(), reduction="mean")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

That is the entire fine-tune. The frozen VAE and text encoder run under `torch.no_grad()`; the U-Net forward goes through the LoRA branches; `loss.backward()` populates gradients on `A` and `B` only (everything else has `requires_grad=False`); `optimizer.step()` updates a couple of million parameters instead of 860 million. To add **prior preservation** (the DreamBooth half), you pre-generate class images, build a second dataloader of "a photo of a dog" pairs, run them through the same step, and add `lambda * prior_loss` before `backward()`.

### Saving and loading the adapter

The payoff is in how small the saved artifact is:

```python
# Save just the LoRA weights — a few MB, not a full checkpoint.
unet.save_pretrained("my-dog-lora")           # peft adapter dir
# or diffusers' helper for inference-ready LoRA:
# pipe.save_lora_weights("my-dog-lora", unet_lora_layers=...)

# Later, at inference: load the base ONCE, attach the adapter on demand.
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights("my-dog-lora")
pipe.set_adapters(["default"], adapter_weights=[0.8])  # strength slider

image = pipe("a photo of sks dog riding a tiny motorcycle, studio light",
             num_inference_steps=30, guidance_scale=7.0).images[0]
```

The `adapter_weights=[0.8]` is the inference-time strength — the $\alpha/r$ scaling we derived, exposed as a dial. Turn it down if the subject is overpowering the prompt; turn it up if identity is weak.

This is also the foundation of multi-tenant serving. Because attaching and detaching a LoRA is just adding or removing a small additive term on a resident base, a single GPU process can serve many subjects by hot-swapping adapters between requests:

```python
# One base model resident; swap megabyte adapters per request.
pipe.load_lora_weights("alice-lora", adapter_name="alice")
pipe.load_lora_weights("bob-lora",   adapter_name="bob")

# Request for Alice:
pipe.set_adapters(["alice"], adapter_weights=[0.85])
img_a = pipe("a portrait of sks person, cinematic", num_inference_steps=30).images[0]

# Next request for Bob — no reload of the 7 GB base, just flip the adapter:
pipe.set_adapters(["bob"], adapter_weights=[0.85])
img_b = pipe("a portrait of sks person, watercolor", num_inference_steps=30).images[0]
```

The base stays hot in VRAM; only the active adapter changes, and switching is near-instant because the adapters are megabytes. This is the architectural reason production avatar and style services standardized on LoRA: it turns "a model per user" into "one model plus a tiny file per user," which is the difference between a service that fits on a handful of GPUs and one that does not fit at all. Frameworks built for this — adapter-aware serving stacks — keep a pool of LoRAs in memory and route each request to its adapter, so thousands of personalized models share a single resident base. You can also do the full DreamBooth-LoRA training from the command line with the official `diffusers` script, which handles prior preservation, mixed precision, and gradient checkpointing for you:

```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir="./dog" \
  --instance_prompt="a photo of sks dog" \
  --class_data_dir="./dog_class" \
  --class_prompt="a photo of a dog" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --rank=16 --resolution=1024 --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --learning_rate=1e-4 --lr_scheduler="constant" \
  --max_train_steps=1000 --mixed_precision="bf16" \
  --output_dir="sks-dog-lora-sdxl"
```

On an RTX 4090 (24 GB) this SDXL DreamBooth-LoRA runs comfortably with gradient checkpointing and `bf16`; on smaller cards drop the resolution to 768, enable `enable_model_cpu_offload`, or use 8-bit Adam (`bitsandbytes`) to fit. SD 1.5 LoRA trains on as little as 8–10 GB.

### The training tricks that quietly decide quality

The loop above will run, but whether it produces a clean adapter or a memorized mess depends on a handful of choices the script exposes. These are the levers I actually turn:

- **Learning rate.** `1e-4` is a sane default for attention-only LoRA at rank 8–16; the text-encoder LoRA, if you train one, wants a *lower* rate (often `5e-5` or less) because the text encoder is more fragile and over-training it is a fast route to language drift. Too high and the adapter slams the subject in at the cost of everything else; too low and 1000 steps is not enough.
- **Step count.** This is the single most important number and there is no universal value — it scales with image count, rank, and learning rate. For 5–10 images at rank 16, `1e-4`, somewhere in 800–1500 steps is typical. The honest method is not to trust a number but to **save a checkpoint every ~250 steps and sample a fixed prompt grid from each**, then pick the checkpoint where identity is clearly learned but style/scene prompts still steer. That window is the sweet spot, and it is narrow.
- **Min-SNR loss weighting.** The plain MSE loss weights all timesteps equally, but the easy, high-noise timesteps dominate the gradient and the model under-learns the detail-bearing low-noise steps. Min-SNR weighting (`--snr_gamma=5.0` in the script) re-weights the loss by $\min(\text{SNR}_t, \gamma)/\text{SNR}_t$, which empirically speeds convergence and sharpens detail. This connects to [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo) if you want the full derivation of why timestep weighting matters.
- **Gradient checkpointing + 8-bit Adam.** Both trade a little speed for a lot of VRAM. Gradient checkpointing recomputes activations in the backward pass instead of storing them; 8-bit Adam (`--use_8bit_adam`) halves the optimizer-state memory. Together they are what let a 24 GB card train an SDXL LoRA at 1024 resolution.
- **EMA on the adapter.** Keeping an exponential moving average of the LoRA weights and sampling from the EMA copy smooths out late-training noise. Optional, but it can buy a cleaner final adapter.

### Data preparation and captioning: the half nobody talks about

The most common reason a personalization fails is not the method — it is the *data and the captions*. Two rules carry most of the weight.

**Curate for variety, not quantity.** Five to fifteen images is plenty, but they should vary in *everything except the subject*: different backgrounds, lighting, angles, distances, and expressions, with the subject consistent. If all your photos share a gray studio backdrop, the model cannot tell whether "gray studio backdrop" is part of the subject or part of the scene, so it bakes the backdrop into the identity — that is style bleed, created at data time. Variety in the context is what lets the model attribute the *invariant* part (the subject) to your trigger and the *variable* parts (background, lighting) to ordinary words.

**Caption to disentangle.** This is the lever most people miss. The caption tells the model what each part of the image *should be attributed to*. For a subject LoRA you have a choice:

- **Minimal captions** ("a photo of `sks` dog") bind *everything* in the image to the trigger token. This maximizes identity strength but also drags context into the token — good for a single clean subject, risky if backgrounds vary.
- **Descriptive captions** ("a photo of `sks` dog sitting on a red couch, sunlight from the left") explicitly attribute the couch and the light to *those words*, leaving the trigger to carry only the dog. This produces a more *flexible* adapter — the dog composes into new scenes better — at the cost of slightly weaker raw identity. For style LoRAs, descriptive captions of the *content* (so the trigger carries only the *style*) are almost always right.

The choice between minimal and descriptive captioning is, in effect, choosing what your trigger token *means*. Get it wrong and you will fight overfitting and style bleed that no hyperparameter can fix; get it right and a rank-8 LoRA on ten images behaves beautifully. Automatic captioners (BLIP, a vision-language model) can draft descriptive captions you then edit to insert the trigger — a standard, time-saving step.

#### Worked example: the captioning fork on a product LoRA

You are making a LoRA for a specific sneaker from twelve product shots, all on a white background. With **minimal captions** ("a photo of `sks` sneaker"), the white background fuses into the identity, and every generation — even "`sks` sneaker on a muddy trail" — fights to reinstate a white void behind the shoe. With **descriptive captions** ("a photo of `sks` sneaker on a plain white background, product photography"), you have told the model that "white background" belongs to *those words*; drop them at inference and the sneaker composes onto the muddy trail cleanly. Same twelve images, same rank, same steps — the captions alone decided whether the adapter is usable. This is why "the data half" deserves as much care as the method half.

## 6. What you get: base versus personalized

The point of all this is a single, sharp behavioral change. Before training, the trigger phrase means nothing special and the model gives you a generic class member. After training, the trigger reliably summons *your* subject's identity while the rest of the model's compositional skill — putting the subject in new scenes, styles, and lighting — stays intact, because you changed so little. Figure 6 contrasts the two regimes on the same prompt skeleton.

![Before and after comparison of a base model producing a generic dog versus a personalized model producing the specific subject while keeping scene control](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-6.png)

The "scenes still work" column on the right is the whole game. A good personalization adds identity *without* costing generalization. When it goes wrong — too many steps, rank too high, no prior preservation — the right column collapses: the subject appears, but every scene looks like a training photo, the model "forgets" how to compose, and prompts stop steering. That collapse is the visible signature of overfitting and forgetting, and the techniques in the next section are how you avoid it.

There is a deeper reason the right column *can* stay intact, and it is the payoff of everything we derived. Because LoRA and Textual Inversion change so little of the model, the vast compositional machinery — the part that knows how to place a subject on a beach, render it in oil, light it from the side — is left exactly as the base model trained it. You are not teaching the model *how to compose*; it already knows. You are only teaching it *one new noun*, and letting its existing composition skill carry that noun into every scene. This is why personalization feels almost magical when it works: you supply five photos and get infinite scenes, because the scenes were never yours to supply — they were always in the base model, waiting for a word to point at your subject. The discipline of personalization is, in one sentence, *change the smallest thing that teaches the subject, so the model's general competence survives to compose it.*

#### Worked example: reading success and failure off the outputs

Suppose you train an SD 1.5 LoRA on five photos of a person at rank 16 for 1200 steps, learning rate `1e-4`. You then generate a fixed grid of test prompts at a fixed seed: "a photo of $sks$ person", "$sks$ person as an oil painting", "$sks$ person on a snowy mountain". Healthy result: the face is recognizable in all three, the oil painting actually looks painted, the mountain is a mountain. Overfit result (push to 3000 steps): the face is sharper but the "oil painting" comes out as a *photo* of the person with a faint painterly filter, and the "snowy mountain" keeps inserting the gray studio backdrop from the training set. The diagnostic is not the identity — it is whether the *non-identity* parts of the prompt still steer. If style and scene prompts stop working, you have trained too long or at too high a rank. The fix is fewer steps or lower rank, which the next section quantifies.

## 7. The rank knob: capacity, size, and overfitting

LoRA gives you one dominant hyperparameter — the rank $r$ — and it directly trades capacity against size and overfitting risk. Higher rank means more trainable directions in the update subspace, which means more capacity to capture fine identity detail, a larger file, and a faster path to overfitting a tiny set. Figure 7 lays the trade-off out across the common rank choices.

![Matrix of LoRA rank four, eight, sixteen, and sixty-four against adapter size, capacity, and overfit risk](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-7.png)

The numbers (SDXL, attention-only LoRA, `float16`) are roughly:

| Rank $r$ | Adapter size (SDXL) | Trainable params | Best for | Overfit risk |
| --- | --- | --- | --- | --- |
| 4 | ~9 MB | ~2.3M | styles, broad concepts | low |
| 8 | ~18 MB | ~4.6M | most subjects | low |
| 16 | ~37 MB | ~9.3M | faces, fine detail | moderate |
| 32 | ~75 MB | ~18.6M | hard subjects, multi-concept | moderate–high |
| 64 | ~150 MB | ~37M | rarely needed | high (memorizes set) |

The pattern is the diminishing-returns curve you would expect from a low-rank approximation: going from rank 4 to 16 buys real identity fidelity; going from 16 to 64 mostly buys file size and overfitting. The reason is the hypothesis we started with — the true fine-tuning update for "draw this one subject" *is* low-rank, so once $r$ exceeds that intrinsic rank you are adding capacity the task does not need, and the extra directions get spent memorizing training noise instead of identity. For a single subject from 3–10 images, **rank 8–16 is the workhorse**. Reserve rank 32+ for genuinely hard or multi-concept adapters, and watch the step count even harder there.

#### Worked example: picking rank for a style versus a face

A *style* LoRA (teach the model a flat-shaded vector-illustration look from 30 example images) lives almost entirely in low-frequency statistics — palette, edge style, shading. Rank 4–8 captures it, the adapter is under 20 MB, and a high rank would just start copying specific training drawings. A *face* LoRA (one person, 8 close-up photos) needs higher frequency identity detail — the precise geometry that makes a face *that* face — so rank 16 (sometimes with a text-encoder LoRA added) is the right call, accepting the moderate overfit risk and capping training near 1000–1500 steps with regular checkpoint sampling to catch the moment identity is learned but generalization is still intact. Same architecture, opposite rank, because the *intrinsic rank of the change* differs.

### Stress-testing the rank choice

Walk it to the edges to feel where it breaks. **What happens at rank 2 on a face?** Identity comes out *soft* — the model gets the broad shape and coloring but smears the fine geometry that makes the face recognizable to a human, because two directions cannot carry that much high-frequency information. **What happens at rank 128 on five images?** The opposite failure: the adapter has so much capacity that it fits the *training images themselves*, backgrounds and all, and your "person on a mountain" generations keep dragging in the exact training poses while prompt control evaporates — classic memorization, and the file is 300 MB for the privilege. **What happens if you raise the inference strength on an under-trained rank-8 LoRA?** You can partially compensate for weak training by turning the $\alpha/r$ strength dial up at inference, which is a genuinely useful trick — but past ~1.2 the subject starts to dominate and distort the rest of the image, the telltale "fried" over-saturated look. The dial trades the same axis as rank and steps: more subject, less prompt. Every lever in personalization moves the *same* trade-off, which is why you tune them together against the metrics from Section 9, not in isolation.

## 8. Combining, merging, and the failure modes

Three operations make the LoRA ecosystem powerful, and each has a failure mode worth naming.

### Combining LoRA with Textual Inversion

A common high-quality recipe pairs a LoRA (carries the identity detail in the weights) with a Textual Inversion embedding (carries a clean, portable trigger word). The LoRA does the heavy lifting on fidelity; the embedding gives a stable handle that composes well in prompts and reduces how hard you have to push the LoRA strength. You can also combine a LoRA with **pivotal tuning**, where you first learn a Textual Inversion embedding, then fine-tune a LoRA around it — the embedding finds the right region, the LoRA sharpens it.

### Merging multiple LoRAs

Because the update is additive, you can load several adapters at once:

```python
pipe.load_lora_weights("character-lora", adapter_name="char")
pipe.load_lora_weights("style-lora",     adapter_name="style")
pipe.set_adapters(["char", "style"], adapter_weights=[0.9, 0.6])
```

This sums the updates: $W_0 + 0.9\, B_\text{char} A_\text{char} + 0.6\, B_\text{style} A_\text{style}$. You can also *bake* a merge into a new set of weights with `pipe.fuse_lora()` for faster inference, or merge adapters offline into a single LoRA. The failure mode here is **interference**: two LoRAs that touch overlapping directions in weight space fight each other, and the result is muddy — the character's identity smears, the style half-applies. Lowering the weights helps; so do merge methods like TIES or DARE (available in `peft`) that resolve sign conflicts between adapters before summing. As a rule, two LoRAs compose cleanly when one is mostly *subject* and the other mostly *style*; two subject LoRAs rarely do.

The math of why interference happens is worth a sentence. Two adapters are additive, so $\Delta W = w_1 B_1 A_1 + w_2 B_2 A_2$. If the column spaces of $B_1 A_1$ and $B_2 A_2$ are roughly orthogonal — they re-aim *different* feature directions — the sum applies both effects independently and they compose. If the column spaces overlap — both adapters try to re-aim the *same* directions, as two subject LoRAs trained on faces will — the sum is a contested average that satisfies neither. TIES-merging works precisely by trimming each adapter to its largest-magnitude directions and resolving sign disagreements before summing, which removes the small, conflicting components that cause the muddiness. When you must combine two adapters that fight, lowering both weights and then nudging the *dominant* one back up is the practical recovery.

A subtle point about **fusing**: `fuse_lora()` permanently writes $W_0 + \frac{\alpha}{r}BA$ into the base weights for speed, after which you can no longer change the strength or unload the adapter without reloading the base. Fuse only when you have settled on a final strength and want the fastest inference; keep it unfused while you are still tuning the strength slider.

### The three failure modes to watch

- **Overfitting.** Too many steps or too high a rank on too few images: the model memorizes poses and backgrounds, prompts stop steering, every output looks like a training photo. *Fix:* fewer steps, lower rank, more varied training images, regular sampling to catch the sweet spot, a lower learning rate.
- **Catastrophic forgetting / language drift.** Full fine-tuning (or aggressive LoRA on the text encoder) erodes the model's general competence — "a dog" starts to mean *your* dog. *Fix:* prior-preservation loss (DreamBooth), a rare trigger token, freezing the text encoder, or simply using LoRA instead of full fine-tuning, which leaves $W_0$ untouched and so forgets far less.
- **Style bleed.** The subject's training context (lighting, background, even a watermark) leaks into every generation, or a style LoRA tints subjects it should not. *Fix:* diverse backgrounds in the training set, caption the unwanted context explicitly so the model attributes it to the words rather than the subject, and lower the inference strength.

## 9. Measuring personalization: the fidelity-versus-flexibility trade-off

"It looks like my dog" is not a metric. To tune personalization honestly — and to compare rank 8 against rank 16, or DreamBooth against LoRA — you need to measure two things that *trade off against each other*, and the DreamBooth paper itself introduced the standard pair.

**Subject fidelity** — does the output actually depict your subject? Two metrics: **CLIP-I**, the cosine similarity between CLIP image embeddings of the generated image and the real training images (higher = more like the subject), and **DINO**, the same idea but using self-supervised DINO features, which the DreamBooth authors argue is *more discriminative for identity* because DINO is not trained to be invariant to within-class differences the way CLIP is. (CLIP is trained to put "a dog" near other dogs, so it can be too forgiving of "a generic dog instead of *your* dog"; DINO is stricter.)

**Prompt fidelity** — does the output obey the *rest* of the prompt? **CLIP-T**, the cosine similarity between the CLIP embedding of the generated image and the CLIP embedding of the *prompt text* (higher = better prompt following). This is the metric that collapses when you overfit: a memorized model nails subject fidelity but ignores "as an oil painting" or "on a mountain," so CLIP-T falls.

The crucial insight is that these two are in tension, and personalization quality is a point on their trade-off curve, not a single number:

$$
\text{good personalization} \;=\; \text{high subject fidelity (DINO / CLIP-I)} \;\wedge\; \text{high prompt fidelity (CLIP-T)}.
$$

Crank the strength, rank, or step count and you slide *up* on subject fidelity and *down* on prompt fidelity — past a point you are just memorizing. Back off and you slide the other way. The sweet spot is the knee of that curve, and you find it by plotting both metrics across checkpoints (or across LoRA strengths at inference) and choosing the point where prompt fidelity is still healthy and subject fidelity has plateaued.

### How to measure it honestly

A defensible protocol, fixed-seed and reproducible:

- Hold out the subject's training images as the fidelity reference set (or, better, a few *unseen* photos of the same subject so you measure identity, not memorization).
- Generate a fixed grid: ~25 diverse prompts ("$sks$ subject doing X in style Y"), 4 seeds each, same sampler and step count throughout.
- Compute mean **DINO** and **CLIP-I** of the generations against the reference set (subject fidelity), and mean **CLIP-T** against the prompts (prompt fidelity).
- Repeat across checkpoints / strengths / ranks and read the trade-off.

Two honesty caveats. First, these metrics are *proxies* — high CLIP-I with low CLIP-T is the unmistakable fingerprint of overfitting, but neither number is the same as a human saying "yes, that's my dog and yes, it's a painting," so spot-check with eyes. Second, never measure subject fidelity against the *training* images alone — a model that copied a training photo verbatim scores perfectly and is useless. Use held-out shots of the same subject if you have them. For the broader story of where generation metrics mislead, the series' [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) post is the deeper treatment; here the point is narrow and actionable: **tune to the knee of the subject-fidelity-versus-prompt-fidelity curve, and you will not overfit.**

#### Worked example: choosing a checkpoint by the numbers

You train an SDXL DreamBooth-LoRA, rank 16, and save checkpoints at 500, 1000, 1500, 2000 steps. Measured on a held-out set and a 25-prompt grid, suppose you see DINO climb 0.42 → 0.58 → 0.63 → 0.64 while CLIP-T holds 0.31 → 0.31 → 0.29 → 0.25. The story is plain: subject fidelity plateaus around 1500 steps (0.63, barely improving after), while prompt fidelity is intact through 1000–1500 and then *falls off a cliff* by 2000 as the model starts memorizing. The 1500-step checkpoint is the knee — best subject fidelity that still keeps prompt control. The 2000-step checkpoint is overfit: marginally better identity, badly worse prompt following. You would never have seen the cliff from "it looks more like my dog at 2000" — the numbers caught the regression the eye rationalizes away. (The values here are illustrative of the *shape* of the curve, not measurements from a specific run.)

## 10. Case studies: real numbers from the literature and practice

A few grounded points to calibrate expectations. I cite the sources; where a figure is a rough community-typical value rather than a paper headline, I say so.

**Textual Inversion (Gal et al., 2022).** Learns a single $d$-dimensional embedding ($d=768$ for the SD/CLIP encoder), ~3 KB per concept, with the entire backbone frozen. The paper's headline is that *one word* suffices to capture a concept well enough to recompose it, while being maximally portable. The honest limitation, acknowledged in the paper and abundant in practice, is capacity: it captures style and coarse identity far better than fine, high-frequency identity.

**DreamBooth (Ruiz et al., 2023).** Fine-tunes the full model with the prior-preservation loss; the paper demonstrates strong subject fidelity and recontextualization from 3–5 images, and explicitly diagnoses **language drift** as the failure the prior-preservation loss exists to prevent. The cost is a full-model checkpoint per subject and careful step budgeting against overfitting.

**LoRA (Hu et al., 2021).** Introduced for adapting GPT-3-scale language models, where it reduced trainable parameters by up to ~10,000× and the checkpoint size by ~3× versus full fine-tuning *with no loss in quality on the studied tasks* — the result that made the low-rank-update hypothesis credible. Its migration to diffusion is now near-universal: a typical SDXL subject LoRA is **10–40 MB** at rank 8–16 versus a **~7 GB** full SDXL checkpoint, a roughly **200–700×** size reduction, which is exactly why custom-model sharing standardized on it.

**DoRA (Liu et al., 2024).** By decomposing into magnitude and direction, DoRA reports consistently closing the gap to full fine-tuning at a given rank across several benchmarks, at a modest extra training cost — useful when you want LoRA economics with closer-to-full-finetune fidelity at low rank.

### The full comparison, side by side

Pulling the four methods into one table makes the trade-off space legible. The size figures are SDXL-scale, `float16`, attention-targeted unless noted; the training figures are community-typical for a single subject from 5–10 images and will vary with your data and hardware.

| Method | Trainable params | Artifact size | Trains | Subject fidelity | Forgetting risk | Composable | Best use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Textual Inversion | 1 (or 2–8) × 768 | ~3–24 KB | embedding only | moderate | none (frozen) | yes (it's a word) | styles, broad concepts, max portability |
| DreamBooth (full) | full model (~2.6B) | ~7 GB | every weight | highest | high (needs prior loss) | no | absolute-max fidelity, you own serving |
| LoRA (r=8–16) | ~2–9M | ~10–40 MB | rank-r adapters | high | low | yes (additive) | **default for shareable subjects** |
| DoRA (r=8–16) | ~2–9M + magnitudes | ~10–45 MB | direction + magnitude | high+ | low | yes | squeezing fidelity at low rank |

The column that matters most for production is "composable." Textual Inversion and LoRA are; full DreamBooth is not. That single property — can I keep one base model resident and hot-swap small per-subject artifacts — is why the entire shared-model economy and every multi-subject serving stack is built on LoRA, with Textual Inversion for the lightest cases.

#### Worked example: the training-cost comparison on one GPU

Concretely, on a single RTX 4090 (24 GB) training one SD 1.5 subject from 8 images: **Textual Inversion** runs in roughly 15–30 minutes and ~6–8 GB VRAM, producing a few-KB embedding. **LoRA at rank 16** runs in roughly 10–20 minutes (fewer parameters than the embedding case might suggest, because the loop is the same and the optimizer touches little) and ~10–12 GB VRAM, producing a ~10–30 MB adapter. **Full DreamBooth** needs gradient checkpointing and 8-bit Adam to fit at all, runs longer, and produces a ~2 GB checkpoint. The wall-clock differences are smaller than the *artifact* differences — all three train in tens of minutes — which is exactly why the deciding factor is almost never training time and almost always the size, portability, and forgetting profile of what you ship. (Times are order-of-magnitude, hardware- and config-dependent, not benchmarked figures.)

#### Worked example: the cost ledger for one subject

Say you want to ship a personalized model for one person and serve it cheaply. **Full DreamBooth:** ~2.6B SDXL params updated, a ~7 GB checkpoint to store and load per subject; if you serve 100 subjects you store 700 GB of near-duplicate weights and reload a fresh 7 GB checkpoint to switch subjects. **DreamBooth-LoRA at rank 16:** one shared 7 GB base in memory, plus a ~37 MB adapter per subject; 100 subjects is 700 GB → **3.7 GB** of adapters, and switching subjects is loading 37 MB onto the resident base in well under a second. The serving architecture is only possible because LoRA is small and additive — you keep one base hot and hot-swap megabyte adapters. That single property is why production avatar services are LoRA, not DreamBooth, end to end.

## 11. When to reach for each (and when not to)

Here is the decision, stated plainly. Figure 8 is the same logic as a tree you can follow from "what am I teaching?"

![Decision tree for choosing Textual Inversion for a style, LoRA for most subjects, or DreamBooth for maximum fidelity](/imgs/blogs/personalization-dreambooth-textual-inversion-lora-8.png)

- **Teaching a *style* or a broad concept, and you want maximum portability?** Start with **Textual Inversion**. A 3 KB word that composes with any prompt is often all a style needs, and it cannot corrupt the model because nothing in the model changes. If the style needs more punch, a **rank 4–8 LoRA** is the next step.
- **Teaching a *specific subject* you want to share — a face, a pet, a product?** Use a **DreamBooth-LoRA at rank 8–16**. This is the default for a reason: near-DreamBooth fidelity, a 10–40 MB shareable file, composable with other LoRAs, reversible, and trainable on a consumer GPU. Add a text-encoder LoRA or prior preservation if identity binding is weak. **This is the right answer for the large majority of personalization jobs.**
- **Teaching a subject where fidelity must be absolutely maximal and you control the serving stack?** Full **DreamBooth** still edges out LoRA on the hardest identity cases. But you pay in gigabytes per subject and overfitting fragility, so only choose it when a 37 MB LoRA has been tried and genuinely falls short — which, with good data and rank 16–32, is rare.

And the *don'ts*, because every technique is a cost:

- **Don't full-fine-tune when a LoRA suffices.** It is the single most common waste — gigabytes of storage and a fragile, forgetting-prone model to gain fidelity a rank-16 LoRA would have matched.
- **Don't crank the rank.** Rank 64+ on five images memorizes the set; it does not improve identity past the subject's intrinsic update rank, it just overfits and bloats the file.
- **Don't skip prior preservation if you full-fine-tune.** Without it, language drift will quietly degrade the model's general competence, and you will not notice until "a dog" stops meaning a generic dog.
- **Don't train without sampling.** Generate a fixed test grid every few hundred steps. Personalization's sweet spot — identity learned, generalization intact — is a window, and you find it by watching, not by trusting a step count.

## 12. Connecting forward: personalization meets control

Personalization answers "*what* subject," but it does not answer "*where* and *in what pose*." Those are the conditioning techniques. A LoRA that knows your face plus [ControlNet structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control) gives you your subject in an *exact* pose; a LoRA plus an IP-Adapter (covered in the upcoming [IP-Adapter and reference conditioning](/blog/machine-learning/image-generation/ip-adapter-and-reference-conditioning) post) lets a *reference image* steer identity without any training at all — a different point on the personalization spectrum, trading per-subject training for a frozen adapter that takes an image prompt. Where these inject matters: LoRA modifies the attention *weights*; ControlNet adds a *parallel branch*; IP-Adapter adds *decoupled cross-attention* for image tokens. They compose because they touch different parts of the stack, and the full recipe — base model, LoRA personalization, ControlNet for structure, the right sampler and guidance — is exactly what the [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) capstone assembles end to end.

How the trigger token actually binds to the conditioning, and why some prompts steer a personalized model better than others, is the subject of [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) — worth reading alongside this one, because half of getting a clean LoRA is captioning the training set so the model attributes identity to your trigger and context to ordinary words.

One last forward pointer for the frontier. Everything above trains a *per-subject* artifact: you spend ten minutes of GPU time to learn one dog or one face. The 2024–2026 wave of reference-conditioning methods — IP-Adapter, InstantID, PuLID — asks whether you can skip per-subject training entirely and instead train *once* a small module that reads a reference image at inference and steers identity on the fly, no fine-tune required. That is a fundamentally different point on the cost curve: zero per-subject training, slightly lower fidelity and controllability than a dedicated LoRA, and instant results from a single photo. The two approaches are complementary, not competing — a tuned LoRA still wins on fidelity and on subjects the reference encoder has never generalized to, while reference conditioning wins on speed and "one photo, right now." Knowing both, and when each is the cheaper path to the image you want, is what separates someone who *uses* personalization from someone who *engineers* it. The base model was the expensive part; everything in this post is the cheap, surgical layer you add on top — and choosing the lightest surgery that gets the job done is the whole craft.

## Key takeaways

- All three methods optimize the **same denoising loss**; they differ only in *which parameters carry gradients* — one embedding (Textual Inversion), the full model (DreamBooth), or low-rank adapters (LoRA).
- **Textual Inversion** learns a single ~3 KB embedding with the whole model frozen: maximally portable, great for styles and broad concepts, capacity-limited for precise identity.
- **DreamBooth** fine-tunes the full model and adds a **prior-preservation loss** on generated class images to prevent **language drift** — the highest fidelity, the heaviest artifact, and overfitting-prone.
- **LoRA** writes the update as $BA$ with rank $r$, training $r(d_\text{in}+d_\text{out})$ parameters instead of $d_\text{in} d_\text{out}$ — a ~48× cut per matrix, a 10–40 MB shareable, composable, reversible adapter. It is the default.
- **Rank** trades capacity for size and overfitting: rank 8–16 is the workhorse; rank 4–8 for styles; rank 64+ mostly memorizes a tiny set.
- LoRAs **stack additively** (subject + style), can be **merged** or **fused**, and pair well with Textual Inversion; the risk is **interference** between overlapping adapters.
- Watch the three failure modes — **overfitting**, **catastrophic forgetting / language drift**, **style bleed** — and catch the sweet spot by sampling a fixed test grid during training.
- Production personalization is **DreamBooth-LoRA**: DreamBooth's loss recipe delivered through LoRA's parameter budget, so one hot base serves megabyte adapters per subject.

## Further reading

- Gal et al., *"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"* (2022) — the embedding-only method.
- Ruiz et al., *"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"* (2023) — full fine-tuning with prior preservation; the language-drift diagnosis.
- Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"* (2021) — the low-rank-update hypothesis and parameter/size reductions.
- Liu et al., *"DoRA: Weight-Decomposed Low-Rank Adaptation"* (2024) — magnitude/direction decomposition closing the gap to full fine-tuning.
- 🤗 `diffusers` training docs — the official DreamBooth, Textual Inversion, and LoRA training scripts (`train_dreambooth_lora_sdxl.py`), and the `peft` `LoraConfig` reference.
- Within series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet), [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning), and the [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) capstone.
