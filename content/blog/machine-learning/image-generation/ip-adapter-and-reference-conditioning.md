---
title: "IP-Adapter and Reference Conditioning: Generating From an Image Prompt"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Condition a diffusion model on a reference image instead of fine-tuning it: derive decoupled cross-attention from scratch, see why a separate key-value path beats stuffing image tokens into the text prompt, wire up a 22M-parameter IP-Adapter in diffusers, compose it with ControlNet, and pin down where one-shot identity methods like InstantID and PuLID land against DreamBooth."
tags:
  [
    "image-generation",
    "diffusion-models",
    "ip-adapter",
    "reference-conditioning",
    "cross-attention",
    "identity-preservation",
    "controlnet",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/ip-adapter-and-reference-conditioning-1.png"
---

You have a photograph of a specific thing — a particular dog, a friend's face, a painting whose palette and brushwork you love — and you want the model to generate *new* images that look like that thing. Not "a golden retriever" in the abstract; *this* golden retriever, with its exact rust-and-cream coat and the slightly off-center white blaze. Words cannot get you there. You can write a paragraph of adjectives and the model will produce a plausible golden retriever, but it will be the model's averaged-over-the-internet golden retriever, not yours. The information you need to transfer — the precise hue distribution, the texture of the fur, the shape of a face — lives in the *pixels* of your reference image, and there is no sentence in any language that losslessly encodes it. This is the gap between a text prompt and an **image prompt**: sometimes you need to hand the model a picture and say "make it look like *this*," not describe it.

The obvious answer from the previous post is to fine-tune. [DreamBooth, Textual Inversion, and LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora) all let you teach a model a new subject or style by training on a handful of images. They work, and for the highest-fidelity identity work they are still the gold standard. But fine-tuning costs you ten to twenty minutes of GPU time *per subject*, produces a multi-megabyte weight file *per subject*, and demands three to five curated images. If you want to generate from a thousand different reference images, or from a reference your user just uploaded a half-second ago, training a LoRA for each one is absurd. What you want is a model that takes the reference image as an *input* — at inference time, zero-shot, no training — exactly the way it already takes a text prompt as an input. That is what this post is about: conditioning generation on a reference image without touching the base weights, and the elegant little mechanism, **decoupled cross-attention**, that makes it work.

![A directed graph showing a reference image and a text prompt each flowing into their own cross-attention path that sum inside one denoiser block](/imgs/blogs/ip-adapter-and-reference-conditioning-1.png)

By the end you will be able to: explain *why* an image prompt carries information no text prompt can; derive decoupled cross-attention from the standard cross-attention equation and prove on a whiteboard why a separate key-value path for image tokens preserves text controllability where naive concatenation destroys it; load and run an **IP-Adapter** (Ye et al., 2023) in 🤗 `diffusers` in a dozen lines; tune the image-prompt strength knob; compose an IP-Adapter with [ControlNet](/blog/machine-learning/image-generation/controlnet-and-structural-control) so one image sets the pose and another sets the look; and place the one-shot identity methods — IP-Adapter-FaceID, InstantID, PuLID — honestly against DreamBooth on the only axis that matters, identity fidelity per unit of effort. This is a *Track D* post, the control-and-editing track: it sits right next to ControlNet (structural control) and personalization (the fine-tuning alternative), and it leans hard on the cross-attention machinery introduced in [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) and dissected in [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet).

Let me anchor the running example: **a frozen Stable Diffusion 1.5 model, a single reference image, and the prompt "a photo, best quality."** The base model is untouched throughout; everything we add is a small bolt-on adapter. That constraint — *frozen base, tiny trainable adapter, zero per-image training* — is the whole game, and it is what separates reference conditioning from the fine-tuning track. Tie it back to the spine of the series: in the **diffusion stack** (data → VAE latent → forward noising → denoiser net → sampler → guidance → image), reference conditioning is a new *conditioning* signal entering the denoiser net through cross-attention, sitting alongside the text condition and the timestep. We are not changing the engine; we are adding a second steering wheel.

## What an image prompt is, and why words cannot replace it

Start with the information-theoretic core, because it justifies the entire enterprise. A text prompt is a short discrete code — a few dozen tokens, each from a vocabulary of around 49,000 (CLIP's BPE tokenizer). Even at the generous estimate of $\log_2(49000) \approx 15.6$ bits per token and 77 tokens, the absolute ceiling on the information a CLIP prompt can carry is about 1,200 bits, and in practice far less because tokens are highly correlated and most of the sequence is padding. A $512 \times 512 \times 3$ image, by contrast, is roughly 786,000 pixel values; even a heavily compressed JPEG is tens of thousands of bytes. The reference image carries *orders of magnitude* more information than any prompt you could type. The question is not whether the image has more bits — it obviously does — but which bits are the ones you care about, and whether you can route just those into the generation.

Here is the crucial subtlety, and it is what makes reference conditioning interesting rather than trivial. You almost never want *all* the bits of the reference image. If you wanted a pixel-perfect copy you would just paste the image. What you want is some *property* of it — its style, its subject's identity, its color palette, its composition — abstracted away from the incidental specifics. You hand the model a photo of your dog on a green lawn and you want "this dog," not "this dog, necessarily on grass." So the job of an image-prompt mechanism is twofold: extract a *semantic* representation of the reference (not raw pixels), and inject that representation into generation at the right level of strength so it transfers the property you care about while leaving the text prompt free to specify everything else ("my dog, but wearing a tiny astronaut helmet, on the moon").

This is exactly why we encode the reference with a **CLIP image encoder** rather than feeding raw pixels. CLIP was trained to align images and text in a shared embedding space; its image features are *semantic* — they capture "what is in the image and roughly what it looks like" rather than "the exact value of pixel (113, 240)." A ViT-H/14 CLIP image encoder maps the $224 \times 224$ reference to a sequence of patch embeddings (257 tokens of dimension 1024, or a single pooled global embedding of dimension 1024, depending on which features you tap). Those embeddings are the same *kind* of object the text encoder produces — a sequence of semantic vectors — which is the key insight that makes the whole thing slot cleanly into the existing architecture. We already have a mechanism for injecting a sequence of semantic vectors into the denoiser: **cross-attention**. The text prompt enters through cross-attention. So can an image prompt. The image prompt is, structurally, just *another prompt* — one made of vectors derived from a picture instead of from words.

Let me name the failure mode the rest of the design exists to avoid, because it motivates everything. If you naively take the CLIP image tokens and *concatenate* them onto the CLIP text tokens — making one long sequence of, say, 77 text tokens plus 4 image tokens — and feed that combined sequence into the existing text cross-attention, two bad things happen. First, the image and text tokens now compete inside a *single softmax*: the attention weights over all 81 tokens must sum to one, so any attention mass the model spends on the image tokens is mass it *removes* from the text tokens. The image prompt literally steals the text prompt's influence. Second, you cannot independently control the strength of the image versus the text — they are fused in one attention operation with one set of weights. Crank the image up and the text fades; that is not a knob, it is a tug-of-war. The decoupled design fixes both, and proving *why* is the science block of this post.

#### Worked example: how much does the text fade under concatenation?

Make the tug-of-war concrete. Suppose at a given spatial query, the pre-softmax attention logits to the text tokens sum (in exponentiated form) to $S_t = \sum_{j \in \text{text}} e^{q\cdot k_j}$ and the image tokens contribute $S_i = \sum_{j \in \text{image}} e^{q\cdot k_j}$. Under a single shared softmax over the concatenated sequence, the fraction of attention going to text is $S_t / (S_t + S_i)$. If the image tokens are even moderately salient — say the CLIP image features produce logits comparable to the text, so $S_i \approx S_t$ — then text retains only $\approx 50\%$ of the attention it had before the image was added. The prompt's grip on the output is *halved* purely as a bookkeeping side-effect of normalization, regardless of whether the user wanted the image to dominate that much. Decoupling, as we will see, gives text its own softmax that *always* sums to one over text alone, so the text grip is $100\%$ of its original strength no matter how strong the image signal is.

## Encoding the reference: from pixels to prompt tokens

Before we can inject the image prompt, we have to *make* it — turn a reference photo into a short sequence of vectors the cross-attention can consume. This is the front half of the adapter, and the design choices here decide what the adapter can and cannot transfer. It is worth slowing down, because "encode with CLIP" hides three separate decisions: which encoder, which layer's features, and how many tokens.

**Which encoder.** The reference goes through the same image preprocessing CLIP was trained with — resize so the shorter side is 224, center-crop to $224 \times 224$, normalize with CLIP's mean and standard deviation — and then through a frozen CLIP image transformer (ViT-H/14 in the standard IP-Adapter, or ViT-bigG for SDXL). The encoder is *frozen*: we never update it. This matters for two reasons. First, freezing it means we inherit CLIP's enormous pretraining for free — the encoder already knows how to turn a photo into a semantic vector, and re-learning that would waste the adapter's tiny parameter budget. Second, because the encoder is fixed and deterministic, the reference's embedding can be *cached*: if you generate ten variations from one reference, you run CLIP once and reuse the features ten times, so the marginal cost of the image prompt across a batch is essentially zero.

**Which layer's features.** A ViT produces, at every layer, a sequence of patch tokens (one per $14 \times 14$ image patch, so $16 \times 16 = 256$ patches plus one [CLS] token = 257 tokens) and, after the final projection, a single pooled global embedding. The original IP-Adapter taps the **global pooled embedding** — one vector of dimension 1024 that summarizes the whole image. That is a deliberately *lossy* choice: a single vector cannot encode "the logo is in the top-left," so it transfers global properties (overall subject, dominant style, palette) and discards spatial specifics. IP-Adapter-Plus instead taps the **patch tokens from the penultimate layer** (the layer *before* the final pooling, which empirically carries richer local detail than the very last layer's heavily-pooled representation) — all 257 of them — so it has spatial, local information to transfer. This is the single biggest lever on "gist vs. specifics," and it is purely a question of *which CLIP features you read*.

**How many tokens.** The cross-attention wants a short sequence of prompt tokens, not 257 of them (257 image tokens would be a lot of attention work in every block, and most are redundant). So we compress. For the global-embedding path, a tiny MLP projects the one 1024-d vector into $p = 4$ token vectors of the cross-attention dimension. For the patch-token path, we need something smarter than an MLP because we are compressing a *variable-content* set of 257 tokens down to a fixed 16 — and that is exactly what a **resampler** (a Perceiver-style module) does. The resampler holds a fixed set of $L = 16$ *learned query tokens* $R \in \mathbb{R}^{16 \times d}$ and runs a few layers of cross-attention where these learned queries attend over the 257 CLIP patch tokens:

$$
\tilde{R} = \text{softmax}\!\left(\frac{R W_q^{R}\,(C_{\text{patch}} W_k^{R})^\top}{\sqrt{d}}\right) C_{\text{patch}} W_v^{R},
$$

iterated for a couple of layers with the usual residual + MLP. The 16 learned queries act like 16 "slots" that each pull together a useful summary of the reference — one slot might converge on "overall color," another on "primary texture," another on "subject shape" — and the result is a fixed-length, information-rich set of 16 image-prompt tokens. The resampler is trained jointly with the rest of the adapter; nobody hand-assigns the slots. This is why IP-Adapter-Plus transfers finer detail: 16 spatially-informed tokens carry far more than 4 tokens distilled from a single global vector.

A subtle but important practical point about the *front half*: because the CLIP encoder and the projection/resampler are the same for any reference, the adapter learns a *general* skill — "translate CLIP image features into useful cross-attention K/V" — rather than memorizing any particular subject. That generality is precisely what makes it zero-shot. A DreamBooth model has *baked a specific subject into its weights*; an IP-Adapter has baked *the ability to read any reference* into its weights. The difference between memorizing an answer and learning to read is the difference between fine-tuning and reference conditioning.

#### Worked example: caching the reference embedding across a batch

Concretely, suppose you serve a "style this photo ten ways" endpoint. Naively you would pass the reference into the pipeline ten times and pay ten CLIP image-encoder forward passes. But the reference is identical across the ten calls, so its CLIP features and projected image-prompt tokens are identical too. Encode once — one ViT-H forward, ≈15–30 ms on an A100 — cache the resulting 4 (or 16) image-prompt tokens, and feed those cached tokens to all ten diffusion runs. The image-prompt overhead for the *whole batch* collapses to a single encoder pass plus the cheap extra cross-attention per step. On a ten-image batch that is a ~10× saving on the encoding cost, and it is the kind of thing that turns "technically works" into "ships at scale." 🤗 `diffusers` lets you pass precomputed `ip_adapter_image_embeds` for exactly this reason, bypassing the encoder when you already have the embedding.

## Decoupled cross-attention, derived from scratch

This is the heart of the method, so we build it carefully from the standard cross-attention you already know. Figure 1 above showed the two-path shape at a glance; now we write the equations.

Recall the cross-attention block inside a diffusion U-Net or DiT. At some layer, the spatial features of the image being denoised are flattened into a set of query vectors $Q = X W_q$, where $X \in \mathbb{R}^{n \times d}$ is the $n$ spatial positions of the current feature map and $W_q$ is the query projection. The text prompt, encoded to a sequence $C_t \in \mathbb{R}^{m \times d_c}$ of $m$ token embeddings, is projected to keys and values $K_t = C_t W_k$ and $V_t = C_t W_v$. Standard text cross-attention is:

$$
Z_{\text{text}} = \text{Attention}(Q, K_t, V_t) = \text{softmax}\!\left(\frac{Q K_t^\top}{\sqrt{d}}\right) V_t .
$$

Each spatial query attends over the text tokens and pulls in a weighted combination of their value vectors; that is how "a tiny astronaut helmet" steers the pixels near the dog's head. This is the mechanism we dissected at length in the U-Net post — the *only* place text touches pixels in a standard Stable Diffusion model.

Now the image prompt. We have the reference encoded by CLIP and projected to a short sequence of image-prompt tokens $C_i \in \mathbb{R}^{p \times d_c}$ (in the original IP-Adapter, $p = 4$ tokens from the pooled global CLIP embedding passed through a tiny projection network). The naive move — concatenate $C_i$ onto $C_t$ and run one attention — is the failure mode we just diagnosed. The decoupled move is this: give the image prompt its **own, separate key and value projection matrices**, $W_k'$ and $W_v'$, distinct from the text's $W_k, W_v$. Compute a *second* attention, using the *same queries* but the image's keys and values:

$$
K_i = C_i W_k', \qquad V_i = C_i W_v', \qquad
Z_{\text{image}} = \text{softmax}\!\left(\frac{Q K_i^\top}{\sqrt{d}}\right) V_i .
$$

Then *add* the two attention outputs, with a scalar scale $\lambda$ on the image term:

$$
\boxed{\,Z = Z_{\text{text}} + \lambda \, Z_{\text{image}} = \text{softmax}\!\left(\tfrac{Q K_t^\top}{\sqrt{d}}\right) V_t + \lambda \,\text{softmax}\!\left(\tfrac{Q K_i^\top}{\sqrt{d}}\right) V_i\,}
$$

That single boxed equation *is* the IP-Adapter. Everything else is plumbing. Look at what it buys you and why it is so much better than concatenation.

![A stacked diagram of the reference image flowing through a frozen CLIP encoder, a trained projection, image key-value projections, and decoupled injection into the frozen UNet](/imgs/blogs/ip-adapter-and-reference-conditioning-2.png)

**Why the text path is untouched.** The text term $Z_{\text{text}}$ is *byte-for-byte identical* to what the original model computed — same $Q$, same $W_k, W_v$, same softmax over the same 77 text tokens. The image is added as a *separate summand*; it does not enter the text softmax at all. There is no normalization shared between the two paths, so adding the image cannot subtract from the text. In the language of the worked example above, the text retains $100\%$ of its attention mass because its softmax is computed over text tokens *only*. This is the mathematical guarantee that text controllability survives. You can prove it in one line: $\partial Z_{\text{text}} / \partial C_i = 0$ — the text output has *zero* dependence on the image tokens. Under concatenation, that derivative is emphatically nonzero; that is precisely the leakage.

**Why $\lambda$ is a real, monotone strength knob.** The image contribution is a pure linear scale $\lambda Z_{\text{image}}$ added on top. At $\lambda = 0$ the model is *exactly* the original text-to-image model (the image prompt has no effect whatsoever — a clean, exact fallback). As $\lambda$ rises, the image signal grows linearly while the text signal stays fixed. So $\lambda$ is a genuine continuous dial from "ignore the reference" to "let the reference dominate," and it does not corrupt the text path on the way. Contrast with concatenation, where the only "knob" is how salient you make the image tokens, which is entangled with everything. In `diffusers`, $\lambda$ is exactly the argument to `pipe.set_ip_adapter_scale(λ)`. Typical values run 0.5–0.8; we will see what happens at the extremes.

**Why this composes additively with the structure of attention.** Because the two outputs are summed, you can stack *multiple* image prompts (multiple references) the same way — one extra attention term per reference, each with its own scale — and they compose linearly. You can also gate $\lambda$ per layer or per timestep (some implementations apply the image prompt only in certain U-Net blocks, e.g. only the up-blocks, to bias toward style vs. layout). The additive structure makes all of this clean.

Now count the trainable parameters, because the parameter economy is the practical headline. The base model is *frozen*. The only new trainable weights are: (1) the small projection network that turns the CLIP image embedding into the $p$ image-prompt tokens, and (2) the extra key/value matrices $W_k', W_v'$ in *each* cross-attention layer. For SD 1.5, there are 16 cross-attention layers; each new $W_k'$ and $W_v'$ is a $d_c \times d$ matrix. Summed across all layers plus the projection, the original IP-Adapter is about **22 million parameters** — under 3% of the SD 1.5 U-Net's ~860M, and a ~22 MB file. You train it *once*, on a large image dataset (the model learns the general skill "attend to an image prompt"), and thereafter it works zero-shot on *any* reference image at inference. That is the magic trick: the per-image cost at inference is one CLIP forward pass (milliseconds), not a training run.

### How the adapter is trained

A natural question: if the base is frozen, what is the training objective for those 22M parameters? It is the *same* denoising loss as the base model, just with the image prompt added to the conditioning. You take a training image, encode it with CLIP to get its image-prompt tokens $C_i$, noise the image to a random timestep, and train the adapter (projection + image K/V) to help the frozen U-Net predict the noise:

$$
\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t}\big[\, \| \epsilon - \epsilon_\theta(x_t, t, C_t, C_i) \|^2 \,\big],
$$

where $\epsilon_\theta$ has all original weights frozen and only the image-path weights $\{W_k', W_v', \text{proj}\}$ are updated. Critically, during training the *target image is its own reference* — the model is taught "here is the CLIP embedding of the image you are trying to reconstruct; use it." That is what teaches the adapter to translate CLIP image features into useful K/V signals. The original IP-Adapter was trained on roughly 10 million text-image pairs (a LAION/COYO subset) for about 1M steps on 8 V100s — a one-time cost the community pays once and everyone reuses. To improve text-image-prompt balance, training also randomly *drops* the image condition (and independently the text condition) some fraction of the time, the same classifier-free-guidance dropout trick from the [guidance post](/blog/machine-learning/image-generation/classifier-free-guidance); this teaches the model to function with text alone, image alone, or both, and is what makes the $\lambda$ knob behave well across its range.

To make the "only the image path trains" point concrete, here is the skeleton of a training step. Notice that the only thing in the optimizer is the adapter's parameters — the U-Net, VAE, text encoder, and CLIP image encoder are all frozen and run in `no_grad` where possible.

```python
import torch
import torch.nn.functional as F

# Frozen: vae, unet (base weights), text_encoder, image_encoder (CLIP).
# Trainable: only the IP-Adapter modules -> projection/resampler + image K/V.
optimizer = torch.optim.AdamW(ip_adapter_params, lr=1e-4)
cfg_drop = 0.05  # probability of dropping the image condition (CFG-style)

for images, prompts in dataloader:
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        text_emb = text_encoder(tokenize(prompts))          # C_t
        # The training image is its own reference (self-supervised image prompt).
        clip_feats = image_encoder(clip_preprocess(images)) # frozen CLIP features

    # The trainable front half: CLIP features -> image-prompt tokens C_i.
    image_tokens = ip_adapter.project(clip_feats)           # proj / resampler (grad on)

    # Randomly drop the image condition so the model also works text-only.
    mask = (torch.rand(len(images), 1, 1, device=latents.device) > cfg_drop).float()
    image_tokens = image_tokens * mask

    noise = torch.randn_like(latents)
    t = torch.randint(0, scheduler.num_train_timesteps, (len(images),), device=latents.device)
    noisy = scheduler.add_noise(latents, noise, t)

    # Frozen UNet, but the decoupled image K/V layers (trainable) receive image_tokens.
    pred = unet(noisy, t, encoder_hidden_states=text_emb, image_tokens=image_tokens).sample
    loss = F.mse_loss(pred, noise)                           # the same L_simple

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The shape of that loop is the whole philosophy in code: the expensive, knowledge-laden parts of the system stay frozen and contribute their pretraining for free, while a thin trainable shell learns the *one new skill* — reading an image prompt. It is the same parameter-efficient-fine-tuning instinct behind LoRA, applied to a new modality rather than a new subject. And because the shell is trained on millions of diverse images, it generalizes; it is not memorizing the training set's subjects, it is learning the *mapping* from CLIP-space to attention-space.

## Running an IP-Adapter in diffusers

Enough theory; let us make an image. 🤗 `diffusers` has first-class IP-Adapter support through `load_ip_adapter`, so the whole thing is a few lines on top of a normal pipeline. Here is a complete, runnable SD 1.5 example.

```python
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Load the ~22M-parameter adapter. The CLIP image encoder is pulled in too.
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)

# lambda: the decoupled-attention image scale. 0.0 = ignore image, 1.0 = strong.
pipe.set_ip_adapter_scale(0.6)

reference = load_image("my_dog.png")          # the image prompt
generator = torch.Generator("cuda").manual_seed(42)

image = pipe(
    prompt="a photo of a dog wearing a tiny astronaut helmet, on the moon",
    ip_adapter_image=reference,               # <-- the new input
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
).images[0]
image.save("dog_on_moon.png")
```

Three things to notice. First, the only structural addition to a vanilla call is the `ip_adapter_image=reference` argument — the reference image is passed *exactly like a prompt*, which is the whole conceptual payoff of treating an image as a prompt. Second, `set_ip_adapter_scale(0.6)` is the $\lambda$ from our equation; you can change it between calls without reloading anything. Third, the text prompt still does its job — "tiny astronaut helmet, on the moon" — because, per the derivation, the text cross-attention is untouched. The reference contributes "this specific dog's appearance"; the text contributes the scene. They coexist because they live in separate attention summands.

![A before-and-after panel contrasting a generic text-only dog against a reference-matched dog once the image prompt is added](/imgs/blogs/ip-adapter-and-reference-conditioning-3.png)

The $\lambda$ knob deserves a feel for its behavior, and Figure 4 below maps the internal mechanism the knob controls. Run the same prompt and seed across scales:

- **$\lambda = 0.0$** — pure text-to-image. The reference has *zero* effect (exact fallback, by construction). You get the model's generic dog.
- **$\lambda \approx 0.3$** — a light touch. The reference nudges color and general vibe; text still dominates composition and content. Good for *style* hints.
- **$\lambda \approx 0.6$** — the sweet spot for most subject transfer. The dog clearly resembles the reference while the text scene ("astronaut helmet, moon") still renders.
- **$\lambda \approx 0.9$–$1.0$** — the image dominates. The reference's pose and background start leaking in; the text prompt ("on the moon") may be partly ignored because the image's strong signal crowds the composition. Useful when you want a near-copy with minor variation, bad when you want a genuinely new scene.
- **$\lambda > 1.0$** — over-driven. Saturated, sometimes degenerate outputs; the linear extrapolation pushes the features off-distribution. Rarely worth it.

![A directed graph showing shared queries hitting separate text and image key-value sets whose attention outputs are summed with a scale](/imgs/blogs/ip-adapter-and-reference-conditioning-4.png)

This monotone, well-behaved dial is *the* practical advantage of decoupling. With a single fixed seed you can sweep $\lambda$ and watch the reference's influence grow smoothly while the text content degrades gracefully (and only at high $\lambda$). That smoothness is not an accident — it is the linear $\lambda Z_{\text{image}}$ term doing exactly what the algebra promised.

#### Worked example: choosing $\lambda$ for a product mockup

Say you are generating marketing variations of a sneaker. You have one studio photo of the sneaker (the reference) and you want it placed in different scenes via text ("on a wet city street at night," "on a sunny beach"). You need the *sneaker* to stay recognizable (high image fidelity) but the *scene* to change freely (text must win on composition). Sweep $\lambda \in \{0.4, 0.6, 0.8\}$ with a fixed seed: at 0.4 the sneaker shape drifts (text too dominant); at 0.8 the original studio background bleeds into every scene (image too dominant); at 0.6 you get a recognizable sneaker in genuinely new scenes. You measure this honestly with CLIP-I (cosine similarity between CLIP embeddings of the reference and the output, for subject fidelity) and CLIP-T (output vs. the text prompt, for prompt adherence) — and you pick the $\lambda$ that maximizes CLIP-T subject to CLIP-I staying above your fidelity floor (say 0.78). That two-metric trade-off curve, swept over $\lambda$, is the right way to set the knob; eyeballing one image lies to you.

## IP-Adapter-Plus and finer features

The original IP-Adapter uses the *global* pooled CLIP embedding — one vector summarizing the whole reference — projected into 4 tokens. That is enough for "this general style/subject" but it throws away spatial detail; the global vector cannot say "the logo is *here* and the texture is *that*." **IP-Adapter-Plus** fixes this by tapping the *patch-level* CLIP features instead — the full grid of 257 patch tokens from the second-to-last CLIP layer — and using a small **resampler** (a perceiver-style cross-attention module with a fixed set of learned query tokens) to compress those 257 patch tokens down to 16 image-prompt tokens. More tokens, carrying spatial/local information, mean the adapter can transfer *fine* detail: a specific texture, the exact arrangement of features on a face, the precise rendering style of a logo. The cost is a few more parameters in the resampler and a slightly stronger tendency to copy the reference (because it has more to copy). In `diffusers` it is a one-line swap of the weight file:

```python
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin",   # patch features + resampler
)
pipe.set_ip_adapter_scale(0.6)
```

The decision between base and Plus is a fidelity-vs-flexibility trade. Base IP-Adapter transfers the *gist* and leaves more room for the text to reshape things; Plus transfers more *specifics* and is harder to steer away from the reference. For loose style inspiration, base. For "keep this exact pattern," Plus. There are also *style-only* and *layout-only* IP-Adapter variants that apply the image cross-attention in only certain U-Net blocks: applying it only in the blocks that the community has found to govern style transfers palette and brushwork while leaving composition to the text — a clean way to get "the *look* of this painting, but my composition." This per-block gating is just the additive structure of decoupled attention exploited selectively: you add the $\lambda Z_{\text{image}}$ term in some layers and not others.

## Blending multiple references

Because the image contribution is a *sum* of attention terms, nothing stops you from giving the model *more than one* reference and blending them. This is one of the most useful and least-appreciated capabilities of the decoupled design, and it falls straight out of the algebra. With two references encoded to image tokens $C_i^{(1)}$ and $C_i^{(2)}$, you can either (a) average their image-prompt embeddings before injection, or (b) add two separate image-attention terms with their own scales:

$$
Z = Z_{\text{text}} + \lambda_1 Z_{\text{image}}^{(1)} + \lambda_2 Z_{\text{image}}^{(2)}.
$$

Option (a) — averaging the embeddings — is the cheap, common path and works remarkably well for *interpolating* between references: feed a photo of a husky and a photo of a wolf at a 0.5/0.5 blend and you get something convincingly between them, the same way you would interpolate two text prompts. Option (b) — separate scaled terms — gives you *independent* control over how strongly each reference speaks, which matters when the references play different roles (one for style, one for subject). In `diffusers` you pass a *list* of images and a *list* of scales:

```python
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter_sd15.bin")

style_ref   = load_image("van_gogh_swirls.png")
subject_ref = load_image("my_dog.png")

# Two references, two scales: weak style, strong subject.
pipe.set_ip_adapter_scale([0.4, 0.8])

image = pipe(
    prompt="a portrait, oil painting",
    ip_adapter_image=[style_ref, subject_ref],   # list of references
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]
```

The composition is linear and predictable: the style reference contributes brushwork at weight 0.4 while the subject reference contributes the dog at 0.8, and the text adds "portrait, oil painting." Three independent signals, one frozen model, summed in the cross-attention. There is no retraining, no merging of weights, no per-combination cost — you pick references and weights at inference time. Compare this to the fine-tuning world, where blending two LoRAs means merging weight files and hoping they do not interfere; the additive-attention approach blends *signals at inference*, which is strictly more flexible.

The honest caveat: blending many references dilutes each one. With four references at moderate scales, no single one is strongly expressed, and the result tends toward a mushy average — the same diminishing returns you get from stuffing too many ideas into one text prompt. Two or three references with clearly differentiated roles is the practical sweet spot; beyond that, consider whether you actually want a fine-tune.

## Evaluating reference conditioning honestly

The series rule is that every non-obvious claim gets a measured result, so let us be precise about *how you measure* reference conditioning, because the metrics are subtle and easy to game. There are three things you might care about, and they trade off against each other, so a single number is always a lie.

**Subject/style fidelity — does the output resemble the reference?** The standard metric is **CLIP-I**: encode both the reference and the generated image with a CLIP image encoder and take the cosine similarity of their embeddings. High CLIP-I means "looks like the reference." For *identity* specifically, CLIP-I is too coarse (it does not encode *which* person), so you switch to **face cosine similarity** — the cosine between ArcFace embeddings of the reference and generated faces — which is the metric the identity papers report. A related metric is **DINO similarity** (self-supervised DINO features), which some find more sensitive to subject identity than CLIP-I because DINO is less semantically pooled.

**Prompt adherence — does the output still obey the text?** The standard metric is **CLIP-T**: cosine similarity between the CLIP embedding of the generated *image* and the CLIP embedding of the text *prompt*. This is the metric that *exposes* a bad image-prompt mechanism — if adding the image prompt tanks CLIP-T, your image is stealing the text's control. The whole point of decoupled cross-attention is to keep CLIP-T high while CLIP-I rises.

**The trade-off is the real result.** Neither CLIP-I nor CLIP-T alone tells you anything, because you can trivially max either one: set $\lambda = 0$ and CLIP-T is perfect but CLIP-I is at baseline (no transfer); set $\lambda$ huge and CLIP-I is high but CLIP-T collapses (the reference ate the prompt). The *honest* evaluation sweeps $\lambda$ and plots the **CLIP-I vs. CLIP-T frontier** — and a *better* method is one whose frontier dominates (higher CLIP-I at equal CLIP-T). That frontier is the right way to compare IP-Adapter vs. concatenation, base vs. Plus, or one adapter against another. It is the same Pareto-frontier discipline the speed track applies to quality-vs-steps; here it is fidelity-vs-controllability.

How to measure it without fooling yourself: fix the seed (so you are measuring the method, not sampling noise), fix the sampler and step count, use a *held-out* set of references the adapter never saw, generate at several $\lambda$ values, and report mean CLIP-I and CLIP-T over a few hundred (reference, prompt) pairs with confidence intervals. A single cherry-picked image is worthless; the distribution over a held-out set is the truth. And for identity, *always* report face cosine, not CLIP-I — a face can have high CLIP-I (a face, the right age and hair) and low identity cosine (not *that* person), and conflating them is how identity methods get oversold.

#### Worked example: reading a fidelity-controllability frontier

Imagine you have two candidate adapters, A and B, and you sweep $\lambda \in \{0.2, 0.4, 0.6, 0.8\}$ for each, plotting (CLIP-T on x, CLIP-I on y). Adapter A's curve passes through (0.30, 0.78) at $\lambda{=}0.6$; adapter B's passes through (0.30, 0.83) at its matched point. At *equal* prompt adherence (CLIP-T 0.30), B delivers higher subject fidelity (0.83 vs 0.78) — B's frontier dominates, so B is the better adapter regardless of which single $\lambda$ anyone quotes. If instead B only beat A at one cherry-picked $\lambda$ but A dominated everywhere else, B's apparent win was a measurement artifact. This is exactly the kind of comparison the IP-Adapter ablation runs to show decoupled attention beats a simple/concatenation adapter: the decoupled frontier sits *above and to the right*, meaning more fidelity *and* more controllability at once, which is the only honest way to claim "better."

## Why decoupling beats concatenation — the stress test

We claimed decoupling preserves text controllability and concatenation destroys it. Let us stress-test that claim rather than assert it, because it is the load-bearing design decision. Figure 6 lays the two approaches side by side.

![A before-and-after comparison of naive token concatenation against a separate decoupled key-value attention path](/imgs/blogs/ip-adapter-and-reference-conditioning-6.png)

Set up the adversarial case: a reference image of a *red sports car* and a text prompt "a **blue** bicycle." These conflict — the reference says red-car, the text says blue-bicycle. A good mechanism should let the user dial between them; a bad one collapses to one.

**Under concatenation**, all 77 text tokens and (say) 4 image tokens share one softmax. The 4 image tokens, carrying the dense, high-salience CLIP features of the car, tend to win large attention weights — recall the worked example, where comparable salience already halves the text's share. So the model attends heavily to "red car" and the output drifts toward a red car-ish vehicle even though you asked for a blue bicycle. Worse, you have *no clean way to turn the image down*: the only lever is to somehow reduce the image tokens' salience, which is not exposed as a parameter and interacts with everything. The image and text are *fused*; you cannot separate them after the fact.

**Under decoupling**, the text softmax sees only "a blue bicycle" and computes the *exact same* attention it would with no image present — a blue bicycle. The image attention separately pulls toward red-car, and the two are summed as $Z_{\text{text}} + \lambda Z_{\text{image}}$. Now $\lambda$ is a real slider between them: at $\lambda = 0.3$ you get a blue bicycle with a faint reddish, car-glossy tint; at $\lambda = 1.0$ the red-car features overwhelm and you get something car-like. The point is *you get to choose*, smoothly, because the two signals were never entangled. The text's contribution is mathematically invariant to the image (that $\partial Z_{\text{text}}/\partial C_i = 0$ again), so it is *always* available at full strength as one of the two things being mixed.

There is a second, subtler advantage. Because the image path has its own $W_k', W_v'$ trained specifically to map CLIP *image* features into the attention space, those projections can learn the right transformation for image features — which is *different* from the right transformation for text features (CLIP image and text embeddings, though aligned, are not identical distributions). Under concatenation you would be forced to push image features through the *text's* $W_k, W_v$, which were trained on text and are a poor fit for image embeddings. Decoupling lets each modality have a projection tuned to its own statistics. This is why IP-Adapter, despite its tiny size, transfers reference appearance well: the image K/V matrices are doing modality-appropriate work.

#### Worked example: measuring the controllability gap

How would you *prove* decoupling preserves text control? Fix a set of 100 (reference, conflicting-prompt) pairs like the car/bicycle case. For each, generate with the concatenation baseline and with decoupled IP-Adapter at matched image influence. Score every output with CLIP-T (similarity to the text prompt). The decoupled method should hold CLIP-T markedly higher at equal subject fidelity (CLIP-I), because the text path is intact. In the IP-Adapter paper's ablation this is exactly the result reported — the decoupled design keeps prompt-following high while a "simple adapter" that injects image features without separate K/V degrades text alignment. The headline numbers from their evaluation: IP-Adapter reaches CLIP-T comparable to text-only generation while adding strong image-prompt fidelity (CLIP-I around 0.83 against the reference), and it matches or beats several fine-tuned per-subject baselines on combined image+text alignment — *without any per-subject training*. The decoupling is doing real, measurable work, not just looking tidy on a whiteboard.

## Identity: the hard case, and the methods built for it

General subject and style transfer is the easy regime. The hard regime is **face identity** — generating new images of a *specific person* from one photo, with the face recognizably *theirs*. Faces are the adversarial case for two reasons. First, humans are exquisitely tuned face-perception machines; we notice a 2% deviation in inter-ocular distance that we would never notice in a dog's snout. Identity has almost no error budget. Second, generic CLIP image features are *not* identity-specialized — CLIP knows "a smiling young woman with dark hair," which describes millions of distinct people, but it does not crisply encode *which* person. So a plain IP-Adapter transfers a face's general *type* but smears its specific *identity*. The fix is to bring in a representation that *is* identity-specialized.

![A matrix comparing reference methods by what they transfer, how many reference images they need, and whether they are fine-tune-free](/imgs/blogs/ip-adapter-and-reference-conditioning-5.png)

Pause on *why* a face-recognition embedding works where CLIP does not, because it is a clean illustration of "the right encoder for the property you want to transfer." CLIP was trained on a contrastive image-text objective: pull an image toward its caption, push it from others. That objective rewards capturing the *describable* content of an image — "a woman, dark hair, smiling, outdoors." It has no pressure to distinguish two different people who would receive the *same caption*, because no caption in the training set says "this specific person versus that one." So CLIP's geometry places same-caption faces close together regardless of identity — exactly the wrong geometry for our task. A face-recognition model like ArcFace is trained on the opposite objective: an *additive-angular-margin* loss that explicitly forces the embeddings of *different identities* apart by a margin on the unit hypersphere while pulling *same-identity* faces (across pose, lighting, age) together. Its entire purpose is to make identity the dominant axis of variation. So the ArcFace embedding of a face is, almost by construction, a near-sufficient statistic for *who* it is — which is precisely the information CLIP throws away. Swapping the encoder swaps the property the adapter can transfer; the decoupled-attention plumbing downstream does not change at all. That modularity — "pick the encoder whose embedding isolates the property you care about, keep the same injection mechanism" — is a general design lesson that extends well past faces (you could imagine a pose-embedding adapter, a font-embedding adapter, a material-embedding adapter, each just a different front-end on the same decoupled attention).

**IP-Adapter-FaceID** replaces (or augments) the CLIP image embedding with a **face recognition embedding** — the 512-dimensional vector from a model like ArcFace/InsightFace, trained specifically to map a face to a point where same-identity faces cluster tightly and different identities spread apart. That embedding *is* identity, distilled. The adapter projects it into image-prompt tokens and injects via the same decoupled cross-attention. Because the input is an identity vector rather than a generic CLIP vector, the transferred face is far more *specifically* the reference person. FaceID-Plus and FaceID-Plusv2 combine the ArcFace identity embedding *and* CLIP features (identity from one, finer appearance from the other), with a `set_ip_adapter_scale`-style balance between them.

**InstantID** (Wang et al., 2024) goes further by combining *two* signals: the ArcFace face embedding (injected through an IP-Adapter-style decoupled attention) *and* a lightweight **IdentityNet**, a ControlNet-like branch that conditions on facial *landmarks* (the 5-point or denser keypoints — eyes, nose, mouth corners). The face embedding says *who*; the landmark control says *where the face is and how it is posed*. Together they pin identity tightly and give you pose control, all zero-shot from a single photo. The trade-off is that the landmark control also *constrains* the pose toward the reference, so InstantID is less free to re-pose the face than a pure-embedding method — editability drops a bit in exchange for fidelity.

**PuLID** (Pure and Lightning ID, Guo et al., 2024) tackles a different problem: many identity adapters, when you crank identity fidelity up, *contaminate* the rest of the image — the model's general behavior shifts, prompt-following degrades, style gets pulled toward the reference's photographic look. PuLID introduces a *contrastive alignment* and an ID loss during training that explicitly minimizes the adapter's disturbance to the base model's behavior, so you get high identity similarity *and* keep the base model's text-following and editability intact. It is the "high fidelity, low side-effects" point on the curve.

It is worth naming the fundamental tension these methods are all negotiating, because it explains why there is a *family* of identity methods rather than one winner. There is an inherent tug between **identity fidelity** (how exactly the face matches) and **editability** (how freely the prompt can restyle, re-pose, and re-light the person). Push the identity signal harder — higher scale, more identity tokens, a landmark control that pins the geometry — and you transfer the face more exactly, but you also drag along more of the reference's incidental properties (its expression, its lighting, its pose), which fights the prompt that wants to change those. Relax the identity signal for editability and the face drifts. Every method picks a point on this curve: a plain FaceID adapter sits toward editability (looser identity, freer prompt); InstantID's landmark control pushes toward fidelity at the cost of pose freedom; PuLID's contrastive training is an attempt to *bend the curve outward* — to get more fidelity at the same editability rather than trading one for the other. There is no setting that maximizes both, which is exactly why you choose the method (and the scale) to match what your product needs: a "verify this face" use case wants the fidelity end; a "make me into ten fantasy characters" use case wants the editability end. Treat the choice as a point on a trade-off curve, not a search for a single best method.

Figure 8 places these one-shot methods against DreamBooth on the axes that matter.

![A matrix comparing identity methods by reference image count, identity similarity, editability, and whether training is required](/imgs/blogs/ip-adapter-and-reference-conditioning-8.png)

Be honest about where the one-shot methods land relative to fine-tuning. Measured by **face cosine similarity** (the ArcFace cosine between the reference face and the generated face — the standard identity metric), the rough landscape from the respective papers and community benchmarks is: a plain IP-Adapter on faces lands low (it was never identity-specialized); IP-Adapter-FaceID reaches roughly 0.55–0.65; InstantID roughly 0.65–0.70; PuLID around 0.70 with notably better editability; and a *well-trained DreamBooth or LoRA* on 5–10 images of the person can exceed 0.75. The pattern is clear and worth internalizing: **one-shot methods close most, but not all, of the identity gap to per-subject fine-tuning, at zero training cost.** For a consumer app where a user uploads one selfie and wants stylized portraits in two seconds, that trade is overwhelmingly worth it — you cannot make them wait fifteen minutes for a LoRA. For a film studio that needs an actor's exact likeness across a hundred shots and has budget for training, DreamBooth/LoRA still wins on raw fidelity. Reference conditioning did not *kill* fine-tuning for identity; it made the *common* case (one photo, instant, good-enough) trivially cheap.

#### Worked example: a one-shot portrait endpoint

You are building an avatar feature: user uploads a selfie, gets back ten stylized portraits ("as a Renaissance oil painting," "as a cyberpunk hero"). DreamBooth is a non-starter — fifteen minutes and a GPU per user is unshippable. So you reach for InstantID or PuLID. The flow per request: run face detection + ArcFace on the selfie (≈50 ms on GPU), get the identity embedding, then run the diffusion pipeline with the embedding injected via decoupled attention plus (for InstantID) the landmark ControlNet, ten prompts × ~2 s each on an A100 ≈ 20 s total, no training. Identity cosine lands around 0.68 — recognizably the user in every style. The honest caveat you put in your own QA notes: the face will be *close* but not *forensically exact*; for an avatar that is perfect, for a "verify this is the same person" use case it is not. Pick the method to the fidelity the product actually needs, and never oversell one-shot identity as pixel-exact.

## Composing with ControlNet: structure from one image, look from another

Here is where reference conditioning becomes genuinely powerful in a real pipeline. [ControlNet](/blog/machine-learning/image-generation/controlnet-and-structural-control) conditions generation on *spatial structure* — a depth map, a pose skeleton, an edge map — extracted from one image. An IP-Adapter conditions on *appearance/identity* from another image. These are *orthogonal* channels: ControlNet controls *where things are*, the IP-Adapter controls *what they look like*. Because both enter the frozen U-Net as additive conditioning signals, you can run them together and get "this pose, that face/style." Figure 7 shows the composition.

![A directed graph showing a pose image feeding ControlNet and an identity image feeding the IP-Adapter, both flowing into one frozen UNet](/imgs/blogs/ip-adapter-and-reference-conditioning-7.png)

A complete `diffusers` example wiring both together — an OpenPose ControlNet for the body pose and an IP-Adapter for the subject's appearance:

```python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Appearance/identity comes from the IP-Adapter; structure from ControlNet.
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.7)

pose_map  = load_image("openpose_skeleton.png")   # structure source
reference = load_image("subject_appearance.png")  # look/identity source

image = pipe(
    prompt="a person standing in a forest, cinematic lighting",
    image=pose_map,                 # ControlNet conditioning (the pose)
    ip_adapter_image=reference,     # IP-Adapter conditioning (the look)
    controlnet_conditioning_scale=0.8,
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("posed_subject.png")
```

You now have *three* independent conditioning knobs feeding one frozen model: the text prompt (scene/context), `controlnet_conditioning_scale` (how rigidly to follow the pose), and the IP-Adapter scale (how strongly to copy the reference appearance). They compose because each is an additive signal — the text and image through their respective cross-attention summands, the ControlNet through its residual additions to the U-Net's skip connections (the zero-convolution trick from the ControlNet post). This compositionality is the practical reason the *frozen-base, additive-adapter* design pattern took over the open-source image-generation ecosystem: every control method is a bolt-on you can mix and match, because none of them retrain the base and all of them just *add* to it.

This is also exactly how the higher-end identity methods are built. InstantID, recall, is "IP-Adapter-FaceID (the identity embedding) *plus* a landmark ControlNet (the pose)" — it is a *specific, pre-tuned instance of this composition*, packaged so you do not have to wire it yourself. Understanding the general composition pattern means you understand InstantID's architecture for free: identity through decoupled attention, structure through a ControlNet branch, summed in a frozen U-Net.

## Failure modes and how to debug them

Reference conditioning fails in a small number of characteristic ways, and because the mechanism is well-understood, each failure has a clean diagnosis. Walking the common ones is the fastest way to build the practitioner's instinct for the technique.

**The output ignores the text prompt.** You asked for "on the moon" and got the reference's original earthly background. Diagnosis: the image signal is too strong relative to the text — almost always $\lambda$ is too high, or you are using IP-Adapter-Plus (more reference detail = more dominance) when base would suffice. Because the paths are decoupled, the fix is mechanical and predictable: lower $\lambda$. If lowering $\lambda$ enough to free the text also loses the subject, you have a genuine conflict between reference and prompt and need a different decomposition — e.g. transfer only style (style-only adapter, applied in fewer blocks) and let the text own the subject, or move the structural part to a ControlNet. The wrong instinct is to *add more text* ("on the moon, on the moon, in space, lunar surface") to out-shout the image; that does not work, because the image is in a separate attention term that more text cannot touch. Turn the image *down*; do not turn the text *up*.

**The reference's appearance barely transfers.** You set the image prompt but the output looks generic. Diagnosis: either $\lambda$ is too low, or — for *identity* — you are using a generic CLIP IP-Adapter on a face, where CLIP simply does not encode identity crisply. The fix for the first is to raise $\lambda$; the fix for the second is to switch to a *face-specialized* adapter (FaceID/InstantID/PuLID) that ingests an ArcFace embedding instead of (or alongside) CLIP. If even a face adapter under-transfers, check that the face was *detected and cropped* correctly upstream — a missed or tiny face yields a garbage identity embedding, and the most common identity bug is a face-detection failure, not a diffusion failure.

**The output looks like the reference but the *composition* is wrong** — right subject, wrong layout. Diagnosis: you reached for the wrong channel. The IP-Adapter transfers *appearance*, not *spatial layout*; if you need the reference's exact composition (same pose, same framing), that is a ControlNet job (canny/depth/pose), and the right architecture is the IP-Adapter + ControlNet composition from the previous section. Trying to force layout through the image prompt by cranking $\lambda$ just makes a near-copy of the *whole* reference, including parts you wanted to change.

**Identity is close but not exact.** The face is recognizably the person but a forensic comparison shows it is off — the jaw is slightly wrong, the eyes a touch too wide. Diagnosis: this is the *fundamental* limit of one-shot identity, not a bug. A single embedding cannot capture the full geometry of a face the way 5–10 images and a per-subject fine-tune can. The honest fixes are: use the best identity-specialized method (PuLID for fidelity-with-editability), provide a *higher-quality, frontal, well-lit* reference (the embedding is only as good as the photo), or — if forensic exactness is genuinely required and you have the images — accept that this is the case where DreamBooth/LoRA is the right tool and reference conditioning is not. Knowing when the technique's ceiling has been hit, and saying so, is part of using it well.

**Fine text and counting still fail.** The output has a garbled logo or the wrong number of objects. Diagnosis: these are *general* diffusion weaknesses (the [text-encoder post](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) covers the attribute-binding and counting failure modes), and the image prompt does not fix them — if anything it can worsen counting, because the reference biases the model toward *copying* the reference's arrangement rather than composing a new one to match the text's count. There is no IP-Adapter knob for this; it is a limitation of the base model, and the fix lives elsewhere (a stronger text encoder, layout control, or a model with better compositional grounding).

The through-line in every diagnosis: because the conditioning channels are *separate and additive*, each failure maps to a specific channel, and each channel has a specific lever. That clean attribution — "this is an image-strength problem, that is a structure-channel problem, this is a base-model limit" — is itself a benefit of the decoupled design. A fused, entangled mechanism would give you tangled failures with no clean lever to pull.

## A scheduler/architecture note: SDXL, SD3, and FLUX

Everything above used SD 1.5 for concreteness, but reference conditioning is not tied to the U-Net. The decoupled-attention idea ports to any architecture with cross-attention. For **SDXL** there are dedicated IP-Adapter weights (`ip-adapter_sdxl.bin` and a Plus variant) that account for SDXL's larger and dual-text-encoder cross-attention; the `diffusers` API is identical except you load the SDXL pipeline and SDXL adapter:

```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                     weight_name="ip-adapter_sdxl.bin")
pipe.set_ip_adapter_scale(0.6)
```

For the transformer-based frontier — [MM-DiT models like SD3 and FLUX](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) — the mechanism is the same *idea* but the injection site differs: these models use *joint* attention where text and image tokens attend together in one stream, so the image-prompt tokens are added as an extra modality in the joint attention with their own projections, and `diffusers` exposes FLUX IP-Adapter support along the same `load_ip_adapter`/`set_ip_adapter_scale` API. The decoupling principle — give the new modality its own key/value path and *add* rather than *fuse* — is invariant across U-Net and DiT. That invariance is why the technique aged well: it is an attention pattern, not an architecture.

## Case studies and real numbers

Pin the claims to the literature, accurately, with the standard caveat that exact figures depend on the evaluation protocol (reference set, sample count, seed, the specific checkpoint), so treat these as the reported orders of magnitude.

**IP-Adapter (Ye et al., 2023), the founding result.** The headline is *parameter efficiency at no quality cost*: ~22M trainable parameters (a ~22 MB file) on a frozen SD 1.5, trained once on ~10M image-text pairs. In their evaluation, IP-Adapter matches or exceeds fine-tuned per-subject methods (and the contemporaneous "simple adapter" baselines) on combined image-and-text alignment, reaching CLIP-I around 0.83 to the reference while keeping CLIP-T (prompt following) close to text-only generation. The decisive ablation is the one we derived: replacing decoupled cross-attention with a *concatenation/simple* adapter drops text alignment noticeably — the separate K/V path is *why* it works, not incidental. The same trained adapter generalizes zero-shot to structural-control composition (works with ControlNet) and to inpainting, none of which it was specifically trained for.

**InstantID (Wang et al., 2024), one-shot identity.** Combines an ArcFace identity embedding (decoupled-attention injection) with an IdentityNet landmark-ControlNet, all zero-shot from one face. Reported face similarity sits in the high-0.6 range on standard face-cosine evaluation — a large jump over a plain CLIP IP-Adapter on faces — while remaining training-free. The honest trade is reduced pose freedom: the landmark control anchors the face geometry toward the reference, so InstantID is less able to dramatically re-pose than a pure-embedding adapter. It became a default for "one selfie → stylized portraits" products precisely because the fidelity-per-effort is so high.

**PuLID (Guo et al., 2024), fidelity with fewer side-effects.** Targets the contamination problem — high identity fidelity usually degrades the base model's prompt-following and style range. With a contrastive alignment loss plus an ID loss, PuLID reaches identity similarity around the 0.7 mark while *preserving* editability and the base model's behavior better than prior adapters. It is the method to reach for when you need both "clearly this person" *and* "still fully steerable by the prompt." It also ships a "Lightning"/few-step-friendly variant, fitting the speed track of this series.

**The DreamBooth comparison, stated fairly.** A well-trained DreamBooth or LoRA on 5–10 images of a subject can push face similarity past ~0.75 and gives the most *editable* high-fidelity identity (because the concept is baked into the weights as a learned token, the prompt can recombine it freely). Its costs are exactly what reference conditioning eliminates: minutes of per-subject training, a weight file per subject, and a curated multi-image set. So the comparison is not "which is better" but "which trade you want": DreamBooth buys the top ~5–10 points of identity fidelity and maximal editability with training; reference conditioning buys instant, zero-shot, single-image identity that is 90% as good. Most products want the latter; a few demanding ones want the former. Both belong in your toolbox.

#### Worked example: the cost crossover for a personalization service

Make the trade quantitative with a back-of-envelope cost model, because that is what decides the architecture in production. Suppose you run a personalization service and need to support $N$ distinct subjects, each generating $g$ images. The *fine-tuning* path costs, per subject, one training run — call it $T \approx$ 15 minutes of GPU time at roughly \$2/hr for a rented A100, so about \$0.50 in compute plus a ~50–150 MB LoRA file to store — and then cheap inference. The *reference-conditioning* path costs *zero* per-subject training and one extra CLIP encode (milliseconds, effectively free) amortized over the subject's generations. So the total fine-tuning cost scales as $N \cdot (\$0.50 + \text{storage})$ while reference conditioning scales as a flat one-time adapter load. The crossover math is stark: at $N = 1$ subject generating thousands of images, fine-tuning's \$0.50 is negligible and its higher fidelity wins. At $N = 100{,}000$ users each generating a handful of images, fine-tuning is \$50{,}000 in training compute plus terabytes of per-user weight files *before you generate a single image*, while reference conditioning is one 22 MB adapter serving everyone. The decision is not about quality at all in that regime — it is that per-subject training simply does not scale to a large, churning user base, and reference conditioning is the only thing that does. Quality is the tiebreaker only when $N$ is small enough that the per-subject cost is affordable.

| Method | Transfers | Ref images | Fine-tune-free | Identity cosine (approx) | When to reach for it |
| --- | --- | --- | --- | --- | --- |
| IP-Adapter | Style + subject (coarse) | 1 | Yes | low (not ID-tuned) | Zero-shot style/subject from one image |
| IP-Adapter-Plus | Style + subject (fine) | 1 | Yes | low (not ID-tuned) | Copy a specific texture/pattern |
| IP-Adapter-FaceID | Face identity | 1 | Yes | ~0.55–0.65 | Instant face from one selfie |
| InstantID | Face identity + pose | 1 | Yes (+ ControlNet) | ~0.65–0.70 | One-shot ID with pose control |
| PuLID | Face identity (clean) | 1 | Yes | ~0.70 | High ID *and* keep editability |
| DreamBooth / LoRA | Subject (baked) | 5–10 | No (train) | ~0.75+ | Top fidelity, budget for training |

## When to reach for reference conditioning — and when not to

A decisive recommendation section, because every technique is a cost and the honest engineer says when it is the wrong tool.

**Reach for an IP-Adapter when** you need *zero-shot* conditioning on an image the model has never seen — a user-uploaded reference, a thousand different references, or any case where per-subject training is impractical. Reach for it for *style transfer* ("make it look like this painting") where general appearance, not exact identity, is the goal — base or style-only IP-Adapter excels here. Reach for FaceID/InstantID/PuLID for *one-shot face identity* in interactive products where the user will not tolerate a training wait. Reach for the IP-Adapter + ControlNet composition when you need to decouple structure and appearance — *this pose, that look* — which is a combination fine-tuning alone cannot give you cleanly.

**Do not reach for it when** you need *forensic* identity fidelity and you *have* multiple images and training budget — then a DreamBooth/LoRA is the right ~5–10 points better and more editable. Do not use a high IP-Adapter scale and then complain the text prompt is ignored — that is the knob working as designed; lower $\lambda$ or switch to a style-only variant. Do not expect an image prompt to render *fine text* in the output (legible words in a logo) or to *count* precisely (exactly three of the reference object) — those are general diffusion weaknesses the adapter does not fix and can even worsen, since the reference biases toward copying rather than composing. And do not expect exact *spatial* copying from an IP-Adapter — it transfers appearance, not layout; if you need the reference's exact composition, that is a ControlNet (canny/depth) job, not an image-prompt job. Matching the tool to the channel — appearance vs. structure vs. identity — is the whole skill.

A note on the most common practical failure: **the reference dominating the composition.** If your outputs keep coming out looking like the reference's *scene* (its background, its pose) rather than just its *subject/style*, you have $\lambda$ too high, or you are using Plus when base would do, or you should apply the image prompt only in the style-governing blocks. The fix is always to *reduce the image's reach*, never to fight it with more text. Because the paths are decoupled, reducing $\lambda$ is a clean, predictable lever — which is, one more time, the payoff of the whole decoupled design.

Step back and place reference conditioning in the spine of the series one last time. The **diffusion stack** is data → VAE latent → forward noising → denoiser net → sampler → guidance → image, and every conditioning method in this track enters at the *denoiser net* stage: text through cross-attention, structure through ControlNet's residuals, and now a reference image through a second, decoupled cross-attention summand. None of them touch the engine — the forward/reverse process, the sampler, the VAE — and that is exactly why they compose so freely. The generative trilemma framing also clarifies what reference conditioning does and does not change: it does not move you on the quality-diversity-speed surface (the base model's sampler and step count still govern that, the topic of the [speed track](/blog/machine-learning/image-generation/building-an-image-generation-stack)); it adds an entirely new *control* axis orthogonal to all three. You are not trading quality for speed here; you are buying *controllability* — the ability to say "like this picture" — for the small price of one adapter and one encoder pass. That is a different kind of knob than the samplers and schedulers of the earlier tracks, and recognizing it as a separate axis is what lets you reason about the whole stack cleanly: pick your quality-speed point with the sampler, then pick your control with the conditioning adapters, independently.

## Key takeaways

- An **image prompt** carries information no text prompt can — encode the reference with a CLIP (or face) encoder so you inject *semantics*, not raw pixels, and transfer a *property* (style, subject, identity) rather than a copy.
- **Decoupled cross-attention** is the mechanism: give the image tokens their *own* key/value projections $W_k', W_v'$, attend with the *shared* queries, and *add* the result as $Z_{\text{text}} + \lambda Z_{\text{image}}$. The text path is byte-identical to the original ($\partial Z_{\text{text}}/\partial C_i = 0$), so text controllability is mathematically preserved.
- **Concatenation fails** because one shared softmax forces image and text to compete for a fixed attention budget — adding a salient image *halves* the text's grip purely through normalization. Decoupling gives each modality its own softmax and a tuned projection.
- The adapter is **~22M parameters (~22 MB)** on a frozen base, trained *once*; thereafter it is **zero-shot per reference** — the per-image inference cost is one encoder pass, not a training run. That economy is the entire reason to prefer it over fine-tuning for the common case.
- **$\lambda$ (`set_ip_adapter_scale`) is a real, monotone knob** from "ignore the image" (exact text-to-image fallback at 0) to "image dominates" (≈1). Sweep it against CLIP-I and CLIP-T; do not eyeball a single image.
- **Identity is the hard case.** Generic CLIP features smear identity; FaceID/InstantID/PuLID swap in an ArcFace identity embedding (and, for InstantID, a landmark ControlNet) to reach ~0.6–0.7 face cosine one-shot — most of the way to DreamBooth's ~0.75+, at zero training cost.
- **Compose, don't choose.** IP-Adapter (appearance) + ControlNet (structure) are orthogonal additive channels feeding a frozen model: *this pose, that look*. InstantID is literally this composition, pre-packaged.
- Reference conditioning did not replace fine-tuning — it made the *common* case (one image, instant, 90%-good) trivially cheap, and left the top few fidelity points to DreamBooth/LoRA when you have the images and the budget.

## Further reading

- Ye, Zhang, Liu, Han, Yang — *"IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models"* (2023). The founding paper; derives decoupled cross-attention and the 22M-parameter design.
- Wang, Bai, Wang, et al. — *"InstantID: Zero-shot Identity-Preserving Generation in Seconds"* (2024). ArcFace identity embedding + IdentityNet landmark control for one-shot faces.
- Guo, Wu, Wang, et al. — *"PuLID: Pure and Lightning ID Customization via Contrastive Alignment"* (2024). High identity fidelity with minimal disturbance to the base model's editability.
- Zhang, Rao, Agrawala — *"Adding Conditional Control to Text-to-Image Diffusion Models"* (ControlNet, 2023). The structural-control method IP-Adapters compose with; the zero-convolution trick.
- 🤗 `diffusers` IP-Adapter documentation — the canonical `load_ip_adapter` / `set_ip_adapter_scale` API, SDXL and FLUX weights, and multi-adapter usage.
- Within this series: [personalization: DreamBooth, Textual Inversion, LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora) (the fine-tuning alternative), [ControlNet and structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control) (composes with this), [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) (the cross-attention foundation), and [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet) (where the attention lives). Forward to [instruction and in-context image editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
