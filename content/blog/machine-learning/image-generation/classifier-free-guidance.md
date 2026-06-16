---
title: "Classifier-Free Guidance: The Knob That Made Text-to-Image Work"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How conditioning and guidance turn an unconditional diffusion model into a controllable text-to-image model: the score-decomposition derivation of the CFG extrapolation, the fidelity-versus-diversity knob, the over-saturation fixes, and runnable diffusers code."
tags:
  [
    "image-generation",
    "diffusion-models",
    "classifier-free-guidance",
    "cfg",
    "conditioning",
    "text-to-image",
    "generative-ai",
    "deep-learning",
    "stable-diffusion",
    "score-matching",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/classifier-free-guidance-1.png"
---

Take a diffusion model that was trained to denoise images, and ask it to draw "an astronaut riding a horse on the moon." Sample it the obvious way — feed it the prompt, run the denoiser fifty times, decode the latent — and you will get something disappointing. The image will be vaguely astronaut-ish, vaguely horse-ish, often neither convincingly. The composition will be soft, the prompt only loosely obeyed. It looks like the model heard your request and then mostly did its own thing. Now turn one number, the **guidance scale**, from 1 up to 7, and run the exact same loop. Suddenly the astronaut is crisp, the horse is unmistakable, the moon is in the background, and the whole thing snaps to your prompt. Turn it up to 20 and the image gets *more* on-prompt but the colors blow out, the highlights clip to white, and every sample starts to look like the same over-cooked poster.

That single number is **classifier-free guidance** (CFG), and it is, without exaggeration, the knob that made text-to-image actually work. Stable Diffusion, SDXL, Imagen, DALL·E 2, Midjourney — every one of them leans on it. Without guidance, conditional diffusion models produce washed-out, prompt-ignoring samples; with it, they produce the crisp, controllable images you have seen. Yet CFG is also the source of the most common failure mode in the whole field (the over-saturated, low-diversity "high-CFG look"), and it doubles your inference cost, and the newest distilled models (SDXL-Turbo, FLUX-schnell) have spent enormous effort trying to *bake it away*. So it is worth understanding exactly what it does, why it works, and what it costs.

![A dataflow figure showing a noisy latent run through one denoiser twice to produce a conditional and unconditional noise estimate, whose difference is scaled by w and added back to form the guided estimate fed to the sampler step](/imgs/blogs/classifier-free-guidance-1.png)

This post is the guidance chapter of the series. By the end you will understand: how you *condition* a diffusion model on a label or a text prompt in the first place; the original **classifier guidance** of Dhariwal and Nichol (2021) and why it was a clever hack that nobody wanted to keep; the **classifier-free guidance** of Ho and Salimans (2022) that replaced it with a single network and one elegant extrapolation; the score-decomposition *derivation* that proves CFG samples from a sharpened conditional distribution; the fidelity-versus-diversity trade-off the guidance scale controls; the over-saturation problem at high scale and the three standard fixes (dynamic thresholding, CFG-rescale, limited-interval guidance); negative prompts as a reuse of the unconditional branch; and how the 2024–2026 frontier (SD3, FLUX, Turbo) relates to all of this through guidance distillation. We will ground every claim in the math, in runnable [🤗 `diffusers`](https://huggingface.co/docs/diffusers) code, and in measured numbers. This builds directly on [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and the [score-based / SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view), so if the words "score" or "reverse process" feel shaky, read those first.

Where does guidance sit in our running frame? Recall the **diffusion stack**: data → VAE latent → forward noising → denoiser net → ODE/SDE sampler → **guidance** → image. Guidance is the second-to-last stage, the steering wheel bolted onto the sampler. And recall the **generative trilemma** — sample quality × diversity × speed. CFG is the cleanest lever you have on the *quality-versus-diversity* face of that triangle, and it pushes hard on *speed* too, because the basic version doubles the work per step. Almost everything interesting about guidance is a fight over that trade-off.

## First, what does it mean to condition a diffusion model?

Before we can guide a model toward a prompt, the model has to be able to *take* a prompt at all. An unconditional diffusion model learns a single denoiser $\epsilon_\theta(x_t, t)$: given a noisy image $x_t$ and a timestep $t$, it predicts the noise that was added. Sample from it and you get a random image from the training distribution — a random face if you trained on faces, a random ImageNet picture if you trained on ImageNet. There is no way to ask for a *specific* face or a *specific* class.

To make it controllable, we give the network a third input, the **condition** $c$, and train $\epsilon_\theta(x_t, t, c)$. The condition can be a class label (one of ImageNet's 1000 classes), a text embedding (a caption encoded by CLIP or T5), an image (for image-to-image or ControlNet), or anything else you can encode into a vector. The training objective barely changes from the unconditional case. We still minimize the simple noise-prediction loss, just with $c$ available:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{x_0, c, \, t, \, \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2 \right], \qquad x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon.$$

Here $(x_0, c)$ are sampled together from the dataset — an image paired with its caption — so the network learns to predict the noise *given* that this noisy image is supposed to be of an astronaut on a horse. Mechanically, how does $c$ reach the denoiser? There are two standard injection mechanisms, and they often run side by side.

The first is **additive / FiLM-style conditioning**, used universally for the timestep $t$ and for low-dimensional conditions like a class label. You embed the condition into a vector and *add* it to (or use it to scale-and-shift) the feature maps inside each residual block. For a class label this is the whole story: a learned 1000-row embedding table maps the class index to a vector that is added to the timestep embedding, and that combined signal modulates every ResBlock's activations through a small MLP that produces per-channel scale and bias. This is cheap and global — the same conditioning vector touches every spatial location identically — which is exactly right for "this whole image is class 207 (golden retriever)."

The second, and the one that makes open-ended text work, is **cross-attention**. The prompt is encoded into a *sequence* of token embeddings (not a single vector), and inside every attention block of the U-Net or DiT the spatial image features attend to those tokens. The image features form the queries; the text tokens form the keys and values. Concretely, at a block with $N$ spatial positions and a prompt of $L$ tokens, you compute an $N \times L$ attention matrix and use it to pull a weighted mix of the text values into each spatial position. This is *spatially selective* — the position that will become the horse's head can attend strongly to the "horse" token while the position that will become the moon attends to "moon." That selectivity is what lets a single image obey a multi-part prompt, and it is also where attribute-binding failures live (when the attention sends "red" to the wrong object). So at every layer and every resolution, the denoiser gets to "look at" the prompt and let each region of the image be modulated by the most relevant words.

A subtle but important point for guidance later: the cross-attention is the *only* place the text actually enters in most U-Net designs, but it enters at *every* resolution stage — the 64×64 top blocks, the 32×32, the 16×16 bottleneck, and back up. That means the prompt influences both coarse layout (decided at low resolution, high noise) and fine texture (decided at high resolution, low noise). Guidance, when we add it, amplifies this influence uniformly across all of those — which is part of why guidance at the wrong noise levels (too early, too late) is wasteful, and why limited-interval guidance helps.

![A dataflow figure where a prompt is encoded by a CLIP or T5 text encoder into token embeddings that become cross-attention keys and values, while the noisy latent provides queries, producing a conditional noise prediction in every denoiser block](/imgs/blogs/classifier-free-guidance-2.png)

There is one more detail in this picture that is the entire seed of classifier-free guidance, so mark it now: **during training we sometimes drop the condition.** With some probability — typically 10% to 20% of the time — we replace $c$ with a special null token $\varnothing$ (an empty caption, a learned "unconditional" embedding) and ask the network to denoise with no useful condition. We will return to *why* this matters in a moment. For now, just hold the fact: the network learns both $\epsilon_\theta(x_t, t, c)$ (denoise toward an astronaut-on-a-horse) and $\epsilon_\theta(x_t, t, \varnothing)$ (denoise toward *a generic image*), using the same weights.

#### Worked example: the shape of a conditional denoiser

Concretely, take Stable Diffusion 1.5. Its denoiser is a `UNet2DConditionModel` with about 860M parameters operating on a $4 \times 64 \times 64$ latent (the VAE compresses a $512 \times 512$ image by $8\times$ per side). The text encoder is CLIP ViT-L/14, which turns a prompt into a $77 \times 768$ sequence (77 tokens, 768-dim each). Inside the U-Net there are 16 cross-attention blocks; in each, the latent's spatial positions (up to $64 \times 64 = 4096$ of them at the top resolution) attend to those 77 text tokens. So the prompt influences the prediction at every spatial location and every resolution. That is a *lot* of surface area for the condition to act on — and, as we will see, also a lot of surface area for a *weak* condition signal to get washed out, which is exactly the problem guidance solves.

SDXL makes the conditioning richer, and it is worth seeing how, because it changes what the "unconditional" branch even means. SDXL uses **two** text encoders — CLIP ViT-L (the SD1.5 one) and the larger OpenCLIP ViT-bigG — and concatenates their outputs into a $77 \times 2048$ sequence for cross-attention, *plus* it pools the bigG output into a single vector that gets added FiLM-style (alongside extra "size" and "crop" conditioning embeddings). So SDXL conditions through both mechanisms at once: cross-attention on the concatenated token sequence and additive conditioning from the pooled embedding. When SDXL runs the *unconditional* branch for guidance, it has to drop **all** of these consistently — both encoders' tokens go to their null embeddings and the pooled vector goes to its learned unconditional value. Get this wrong (e.g. only zeroing one encoder) and the unconditional pass is mismatched, the difference vector $\epsilon_\text{cond} - \epsilon_\text{uncond}$ is corrupted, and guidance misbehaves. The diffusers pipeline handles this for you, but if you implement CFG by hand on SDXL you must null *every* conditioning input.

To make the "encode the prompt, then run guidance" flow concrete, here is the prompt-encoding half for SD1.5 — the part that produces the `text_embeds` and `uncond_embeds` the sampler loop above consumes:

```python
import torch
from transformers import CLIPTextModel, CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5",
                                          subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="text_encoder",
    torch_dtype=torch.float16).to("cuda")

def encode(prompt):
    tokens = tokenizer(prompt, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        return text_encoder(tokens).last_hidden_state          # (1, 77, 768)

text_embeds   = encode("an astronaut riding a horse on the moon")
uncond_embeds = encode("")          # the null condition: an empty prompt
neg_embeds    = encode("blurry, low quality, extra limbs")     # optional negative
```

The empty-string encoding is literally the $\varnothing$ in all our formulas — and swapping `uncond_embeds` for `neg_embeds` in the guidance loop is the entire negative-prompt feature. That symmetry between "the null branch" and "the negative branch" is the practical heart of CFG, and we will lean on it twice more below.

## Why naive conditional sampling is too weak

So we have a conditional model. Why isn't that enough? Why do we need guidance at all? The answer is that **the conditional signal, learned honestly from data, is too gentle.**

Here is the intuition, and then the math. When you train $\epsilon_\theta(x_t, t, c)$ on real caption–image pairs, the model learns the *true* conditional distribution $p(x \mid c)$ as it appears in the data. But real data is noisy and the conditioning is loose. A caption like "a dog" is attached to images of thousands of different dogs in thousands of poses and lightings; the caption constrains the image only weakly. So the learned $p(x \mid c)$ is *broad*: many images are roughly equally consistent with the prompt. When you sample from this broad distribution, you draw something that is "a plausible image that is loosely a dog" — and "loosely" is the problem. You wanted a clear, central, unambiguous dog, and the model gave you a sample from the fuzzy edges of dog-space.

What we actually want at inference time is to sample not from $p(x \mid c)$ but from a *sharpened* version of it — a distribution that concentrates on the images that are *most* characteristic of the prompt and suppresses the ambiguous ones. We want a "more dog than the average dog" dog. Formally, we want to sample from something like $p(x \mid c)^{1+w}$ (up to normalization), where $w > 0$ sharpens the distribution: raising a density to a power greater than 1 makes its peaks taller and its tails thinner. Guidance is exactly the mechanism that lets us sample from this sharpened distribution without retraining. The whole rest of this post is about *how* you do that and *what it costs*. To get there, we need the score view of diffusion.

There is a second, complementary reason the honest conditional is too weak, and it is specific to how the conditioning was *learned*. Recall the dropout: 10–20% of training steps used the null condition. That dropout is necessary (it is what builds the unconditional branch), but it also means the model spent a meaningful fraction of its gradient budget learning to ignore the prompt. Combine that with the fact that real captions are short and lossy — "a dog" describes a million images — and the conditional signal the network learned is genuinely soft. At sampling time, following the learned conditional score alone barely tugs you toward the prompt before the natural-image prior ($p(x)$) reasserts itself and pulls you back toward "a generic plausible image." Guidance is, in this light, a way to *overrule the model's natural reticence* and force the prompt's pull to dominate. That framing — guidance as turning up the volume on the prompt relative to the generic prior — is exactly what equation (3) will make quantitative.

A toy picture sharpens the intuition. Suppose, in some 1-D slice of latent space, $p(x)$ (the unconditional prior) is a wide bump and $p(c \mid x)$ (how well each $x$ matches the prompt) is a gentle slope favoring the right side. Their product $p(x\mid c)$ is a bump shifted slightly right — the prompt nudged it, but only a little, because the slope was gentle. Now raise the slope to a power: $p(c\mid x)^{1+w}$ becomes a *steep* ramp, and the product is a much narrower bump pushed firmly to the prompt-favoring region. Same model, same two factors — we just amplified the second one. That is the entire mechanism, and once you see it in 1-D the high-dimensional version is no different in spirit. To make it operational on a real network, though, we need to express this in terms of *scores*, because scores are what the diffusion sampler actually consumes.

### The score connection (one paragraph of refresher)

Recall from the [SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) that a diffusion model's noise prediction is, up to a known scaling, an estimate of the **score** of the noised data distribution — the gradient of the log-density:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{1}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t).$$

The score points "uphill" toward higher-density (more realistic) regions, and the reverse-diffusion sampler is essentially following the score back from noise to data. This identity — *noise prediction is negative score, scaled* — is the bridge that connects everything. Guidance is going to be a manipulation of the score, which we then translate back into a manipulation of the predicted noise. Hold onto it.

## Classifier guidance: the clever hack we replaced

The first working answer to "how do I sharpen toward $c$?" came from Dhariwal and Nichol's 2021 paper, "Diffusion Models Beat GANs on Image Synthesis." Their idea, **classifier guidance**, is worth understanding precisely, both because it is genuinely clever and because seeing its drawbacks makes classifier-free guidance feel inevitable.

Start from Bayes' rule applied to the conditional density:

$$p(x \mid c) = \frac{p(x)\, p(c \mid x)}{p(c)}.$$

Take the log and the gradient with respect to $x$ (the $p(c)$ term is constant in $x$, so it vanishes):

$$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x).$$

Read this aloud, because it is the most important equation in the post. **The conditional score equals the unconditional score plus the gradient of a classifier.** The first term, $\nabla_x \log p(x)$, is what an *unconditional* diffusion model already gives you (it says "move toward realistic images"). The second term, $\nabla_x \log p(c \mid x)$, is the gradient of the log-likelihood that a classifier assigns to class $c$ given the image $x$ (it says "move toward images this classifier reads as $c$"). Add them and you get the conditional score — the direction that makes the image both realistic *and* class-$c$.

Dhariwal and Nichol's move was: train an *unconditional* diffusion model and, separately, train a **classifier on noisy images** $p_\phi(c \mid x_t, t)$. At each sampling step, take the unconditional score from the diffusion model, add the classifier's gradient, and step. To get the *sharpening* effect (not just plain conditioning), they scaled the classifier gradient by a factor $s > 1$:

$$\hat\epsilon = \epsilon_\theta(x_t, t) - s \cdot \sqrt{1 - \bar\alpha_t}\, \nabla_{x_t} \log p_\phi(c \mid x_t, t).$$

The scale $s$ is the guidance strength; $s = 1$ gives ordinary conditioning, $s > 1$ over-weights the classifier and sharpens toward the class. Where does that $\sqrt{1-\bar\alpha_t}$ factor come from? It is the same score-to-noise conversion from the refresher: a gradient of a log-density (the classifier gradient is one) becomes a noise correction by multiplying through the $-\sqrt{1-\bar\alpha_t}$ scaling, so the classifier's "push toward class $c$" lands in the same units as the model's noise prediction and can be subtracted from it. The structure is exactly equation (1) read as noise instead of score: the unconditional noise prediction $\epsilon_\theta(x_t,t)$ plays the role of $-\nabla\log p(x)$, and the (scaled) classifier gradient supplies the $\nabla\log p(c\mid x)$ term. It worked — it was how those 2021 models beat GANs on ImageNet FID. But look at what it requires, and the drawbacks stack up fast.

The training requirement alone is brutal in a way that is easy to gloss over. The classifier $p_\phi(c\mid x_t, t)$ must be accurate across the *entire* noise schedule — at $t$ near zero it sees an almost-clean image (easy), but at $t$ near the maximum it sees something indistinguishable from Gaussian static, and it must *still* emit a useful gradient pointing toward class-$c$-ness. Dhariwal and Nichol trained their classifier on noised ImageNet images at all timesteps, essentially the encoder half of their U-Net with a classification head, which is a second large model with its own training run, its own hyperparameters, and its own failure modes. And the gradient it produces at high noise is, almost by definition, low-signal: there is barely any class information in pure static, so the "guidance" early in sampling is mostly noise itself. This is one more reason the field was primed to drop the separate classifier the moment a cleaner option appeared.

![A two-column figure contrasting classifier guidance, which needs a separate fragile classifier trained on noisy images, with classifier-free guidance, which trains one network with the condition randomly dropped so it produces both scores](/imgs/blogs/classifier-free-guidance-3.png)

The problems with classifier guidance:

- **You need a separate classifier, and it must be trained on noisy images.** An off-the-shelf ImageNet classifier is useless here — it has never seen $x_t$ at high noise levels, where the image is almost pure static. You have to train a custom classifier that can read a class out of a half-destroyed image across every noise level $t$. That is a second training run, a second model to ship, and a finicky one.
- **The classifier gradient is fragile and adversarial-prone.** $\nabla_x \log p_\phi(c \mid x)$ is exactly the kind of gradient that adversarial examples exploit. Pushing an image along a classifier's gradient can produce high classifier-confidence *without* producing a more realistic or more genuinely-class-$c$ image — you can fool the classifier off-manifold. The guidance can drift into garbage that the classifier loves and a human hates.
- **It does not generalize to text.** For 1000 ImageNet classes you can train a classifier. For open-ended text prompts ("an astronaut riding a horse on the moon, oil painting, golden hour") there is no fixed label set to classify. You would need a noisy-image-conditioned text model, which is most of the way to just building a conditional diffusion model anyway.

So classifier guidance was a great proof of concept — it showed that *scaling the classifier gradient* is the right idea — but the classifier itself was a liability. The obvious question: can we get $\nabla_x \log p(c \mid x)$ *without a classifier*?

## Classifier-free guidance: one network, one extrapolation

Ho and Salimans, in their 2022 workshop note "Classifier-Free Diffusion Guidance," answered yes, and the answer is beautiful. Rearrange the score identity:

$$\nabla_x \log p(c \mid x) = \nabla_x \log p(x \mid c) - \nabla_x \log p(x).$$

The classifier gradient is *the difference between the conditional score and the unconditional score*. And we do not need a classifier to get either of those — a conditional diffusion model gives you the conditional score, and the *same* model with the condition dropped gives you the unconditional score. That is exactly why we trained with condition-dropout. So substitute this back into the sharpening recipe. We want to sample using the sharpened score $\nabla_x \log p(x) + (1 + w)\, \nabla_x \log p(c \mid x)$ (the $1+w$ over-weights the implicit classifier, just like Dhariwal–Nichol's $s$). Expand:

$$
\begin{aligned}
\tilde s(x_t, c) &= \nabla_x \log p(x) + (1+w)\, \big[\nabla_x \log p(x \mid c) - \nabla_x \log p(x)\big] \\
&= \nabla_x \log p(x \mid c) + w\, \big[\nabla_x \log p(x \mid c) - \nabla_x \log p(x)\big].
\end{aligned}
$$

Now translate scores back into predicted noise using $\nabla_{x_t} \log p_t \propto -\epsilon_\theta$ (the same scale factor multiplies every term, so it cancels in the structure). You get the **classifier-free guidance formula**:

$$\boxed{\;\hat\epsilon(x_t, t, c) = \epsilon_\theta(x_t, t, c) + w\,\big[\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big]\;}$$

This is the entire algorithm. Let me unpack it because it is so compact. You run the denoiser **twice** at each step: once with the real condition ($\epsilon_\text{cond} = \epsilon_\theta(x_t, t, c)$) and once with the null condition ($\epsilon_\text{uncond} = \epsilon_\theta(x_t, t, \varnothing)$). Then you form the guided estimate by starting at the conditional prediction and stepping *further* in the direction that the conditional prediction differs from the unconditional one:

$$\hat\epsilon = \epsilon_\text{uncond} + (1 + w)\,(\epsilon_\text{cond} - \epsilon_\text{uncond}) = \epsilon_\text{cond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond}).$$

(Those two forms are algebraically identical; the diffusers convention uses a `guidance_scale` $= 1 + w$, so a `guidance_scale` of 7.5 corresponds to $w = 6.5$. People are sloppy about whether "the scale" means $w$ or $1+w$; always check.) The vector $\epsilon_\text{cond} - \epsilon_\text{uncond}$ is the *direction the prompt pulls* — it is, up to scaling, the implicit classifier gradient. CFG amplifies that pull. At $w = 0$ (scale $= 1$) you get plain conditional sampling, no sharpening. As $w$ grows, you extrapolate harder past the conditional prediction, away from the generic unconditional one.

That is why people call it an **extrapolation**, not an interpolation. You are not blending the two predictions and landing between them; you are going *past* the conditional prediction, in the direction pointing away from the unconditional one. Figure 1 at the top of the post shows exactly this geometry: two predictions, a difference vector, and a guided point that sits beyond the conditional one.

### Why this is so much better than classifier guidance

Look at what we just removed. No separate classifier. No noisy-image classification training. No adversarial-gradient fragility — the difference $\epsilon_\text{cond} - \epsilon_\text{uncond}$ comes from the *generative* model itself, which is trained to stay on the image manifold, so guidance pushes along directions the model considers realistic. And it works for *any* condition the model was trained with, text included, because "the implicit classifier" is just the model's own conditional-versus-unconditional disagreement. The cost is real but simple: **two forward passes per step instead of one**, roughly doubling inference compute. (In practice you batch the conditional and unconditional inputs together into one forward pass with batch size 2, so it is one kernel launch but twice the FLOPs and twice the activation memory.) That doubled cost is the single biggest reason the distillation work in Track E exists, and we will come back to it.

## The derivation: what distribution does CFG actually sample from?

We have been hand-waving that CFG "sharpens the conditional." Let us make it precise, because this is the science block and the *why* should be provable, not asserted.

![A dataflow figure showing Bayes rule splitting the conditional score into the unconditional score plus an implicit classifier gradient, with classifier-free guidance scaling that gradient by one plus w to target a sharpened distribution proportional to the data density times the likelihood raised to one plus w](/imgs/blogs/classifier-free-guidance-4.png)

Start again from the exact score identity, valid at every noise level $t$ (I drop the $t$ subscripts for readability; everything is at the noised marginals $p_t$):

$$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x). \tag{1}$$

The CFG sampler uses, as its effective score, the unconditional score plus $(1+w)$ times the implicit classifier gradient:

$$\tilde s(x, c) = \nabla_x \log p(x) + (1+w)\, \nabla_x \log p(c \mid x). \tag{2}$$

Now ask: what density $\tilde p(x \mid c)$ has $\tilde s$ as its score? A score is the gradient of a log-density, so we need a $\tilde p$ with $\nabla_x \log \tilde p(x \mid c) = \tilde s(x, c)$. Integrate (treating $\nabla_x$): the function whose gradient is the right-hand side of (2) is

$$\log \tilde p(x \mid c) = \log p(x) + (1+w)\, \log p(c \mid x) + \text{const}(c),$$

which exponentiates to

$$\boxed{\;\tilde p(x \mid c) \;\propto\; p(x)\, \big[\,p(c \mid x)\,\big]^{1+w}.\;} \tag{3}$$

There it is. **Classifier-free guidance samples (approximately) from a distribution proportional to the data density times the conditional likelihood raised to the power $1+w$.** Compare with the true conditional, which is $p(x \mid c) \propto p(x)\, p(c \mid x)$ (the case $w = 0$). CFG keeps the $p(x)$ factor — so samples stay realistic — but raises the *likelihood* factor $p(c \mid x)$ to a power greater than 1. Raising a likelihood to a power $> 1$ is exactly a **temperature** operation that sharpens it: regions where the prompt is strongly satisfied ($p(c \mid x)$ near 1) are boosted, and regions where it is only weakly satisfied are suppressed. The bigger $w$, the more aggressively you up-weight prompt-satisfaction relative to realism.

A clean way to see the sharpening: equation (3) is equivalent to $\tilde p(x \mid c) \propto p(x \mid c) \cdot [\,p(c \mid x)\,]^{w}$. So CFG takes the honest conditional $p(x \mid c)$ and *reweights* it by the implicit classifier raised to the $w$. Samples that the model's own classifier finds very confidently class-$c$ get up-weighted; ambiguous samples get down-weighted. That is precisely "a more dog than the average dog."

Two honest caveats, because this is the science section and the approximation matters in practice. First, equation (3) is exact only if $\epsilon_\theta$ recovers the *true* scores at every noise level. Real networks have learning error, and the unconditional and conditional branches have *correlated but not identical* error, so the practical guided score is an approximation of (2). Second — and this is the subtle one — the identity (1) holds for the *noised* marginals $p_t$ at each $t$, but the implied "sharpened distribution" you actually sample from is not literally $p_0(x)\, p_0(c\mid x)^{1+w}$ at the data level, because composing per-step sharpening along the whole trajectory does not exactly equal sharpening the endpoint distribution. The power-$1+w$ picture is the right *intuition* and predicts the behavior well, but treat it as a tight heuristic rather than a theorem about the final samples. (Researchers have since studied this gap; it is part of why guidance can over-concentrate and why limited-interval guidance, below, helps.) With those caveats stated, the picture is solid enough to reason with: **bigger $w$ = sharper toward the prompt, at the cost of diversity.**

## The fidelity–diversity trade-off: what the knob actually controls

Equation (3) tells you exactly what to expect as you turn the knob, and it is the central trade-off of the whole post. Let me state it as a law and then show the numbers.

**As $w$ increases: prompt fidelity goes up, sample diversity goes down, and image realism follows an inverted-U** — it improves up to a sweet spot and then degrades. Here is why each piece happens:

- **Fidelity up.** Raising $p(c \mid x)$ to a higher power concentrates mass on images that strongly satisfy the prompt. Measured by **CLIP-score** (cosine similarity between the CLIP embedding of the generated image and the prompt), this rises monotonically with $w$ over the useful range — more guidance, better text alignment.
- **Diversity down.** Sharpening collapses the conditional distribution toward its modes. At high $w$ you keep sampling near the few most-prompt-satisfying images, so different seeds produce near-duplicates. Measured by **recall** (mode coverage) or by simple pairwise diversity of a batch, this falls as $w$ grows.
- **Realism: inverted-U.** At very low $w$ ($w=0$, scale 1) the samples are diverse but only loosely on-prompt and often a bit incoherent, so **FID** (which penalizes both unrealistic *and* off-distribution samples) is mediocre. As $w$ rises, FID *improves* because samples get crisper and more prompt-aligned — up to a point. Past the sweet spot, two things hurt FID: diversity collapse (FID penalizes a distribution that is too narrow relative to the reference set) and the *over-saturation* artifact we will dissect next. So FID traces a U: high at $w=0$, minimum somewhere in the middle, rising again at high $w$.

![A four-by-four comparison matrix showing how CLIP-score rises, FID dips to a minimum near scale five to eight, diversity falls, and color saturation worsens as the guidance scale climbs from one to fifteen](/imgs/blogs/classifier-free-guidance-5.png)

The practical upshot, and the single most useful number in this post: **the FID-optimal guidance scale for most text-to-image models is roughly 5 to 8** (in diffusers' $1+w$ convention). Stable Diffusion 1.5's default is 7.5; SDXL's is 7.0; many people run 5–7 for photorealism and push to 9–12 for "make it really obey a complex prompt, diversity be damned." Below ~3 the prompt is too weakly enforced; above ~12 you are firmly in over-saturation and diversity-collapse territory for standard (non-rescaled) models.

It is worth being precise about *why* FID specifically is U-shaped, because it is a recurring source of confusion. FID measures the Fréchet distance between the distribution of features (from an Inception network) of your generated images and the distribution of features of a reference set of real images. It rewards your generated distribution for matching the real one in *both* mean and covariance. Now watch what guidance does to that match. At $w=0$, your generated distribution is diverse but its samples are individually a bit off (loosely on-prompt, slightly incoherent), so the *mean* of your feature distribution sits in a slightly wrong place — FID is mediocre. As $w$ rises, individual samples get crisper and more clearly the right content, pulling the feature mean toward the reference mean — FID drops. But keep pushing and two things go wrong simultaneously: your *covariance* shrinks (diversity collapse — you are sampling near a few modes, so your feature cloud is too tight compared to the reference cloud), and your samples acquire the saturation artifact (which moves the feature mean *back* off, because over-saturated images are not what the reference set looks like). Both effects inflate the Fréchet distance, so FID climbs again. The U-shape is therefore not an accident of one model — it is structural, a direct consequence of FID rewarding distributional match while guidance trades diversity for per-sample fidelity. Any metric that, like FID, penalizes a too-narrow output distribution will show the same U.

This also explains why **Inception Score (IS) does *not* show a U** and instead rises monotonically with guidance: IS rewards each sample being confidently classifiable (fidelity) and the batch covering many classes (diversity), but it is dominated by the confidence term and does not strongly penalize *within-class* diversity collapse the way FID's covariance term does. So if you tune guidance to maximize IS you will pick a scale that is too high and over-saturated. This is a concrete example of the eval crisis the series keeps returning to: *the metric you optimize decides the model you ship.* Report both, plot the frontier, and pick the knee with your eyes on the actual images.

#### Worked example: reading the trade-off curve like an engineer

Suppose you are tuning guidance for a product that generates one image per prompt (so diversity across seeds matters less) and you care about prompt alignment and realism. You run a sweep: for each of $w$-scale $\in \{1, 3, 5, 7, 11, 15\}$, you generate 5,000 images from 5,000 held-out captions with a *fixed seed schedule*, then compute FID against a reference set of real images for those captions and CLIP-score per image. A representative pattern (numbers in the ballpark reported across SD-family ablations; treat as illustrative, not a single citable table): CLIP-score climbs from roughly 0.26 at scale 1 to about 0.33 at scale 7 and keeps inching up to maybe 0.34 at scale 15. FID falls from the high 20s at scale 1 to a minimum around the low-to-mid teens near scale 5–7, then rises back toward the high teens by scale 15 as saturation and mode-collapse bite. You read the crossover off the two curves: scale ~6 gives you near-peak CLIP-score with near-minimum FID. That is your default. If a downstream eval shows complex prompts (three objects, a count, a spatial relation) still failing, you bump to 8–9 *for those prompts only* and accept the diversity hit. The discipline that makes this trustworthy: fixed seeds, a fixed reference set, the same number of sampling steps across all runs, and FID computed on ≥10k samples (FID is badly biased on small sample sets — never trust an FID computed on 500 images).

A note on *how to measure honestly*, because guidance ablations are where people fool themselves. FID conflates fidelity and diversity into one scalar, so a model can "improve FID" by getting crisper *or* by matching the reference diversity better, and high CFG moves those in opposite directions — that is the whole reason FID is U-shaped here. The clean protocol is to report **both** CLIP-score (fidelity) and a diversity/recall metric, plotted against $w$, so the trade-off is visible rather than collapsed. The CFG paper and the follow-ups all report the fidelity–diversity *frontier* (e.g. FID-vs-IS or precision-vs-recall curves swept over $w$), not a single point, for exactly this reason.

## The over-saturation problem and three fixes

Now the failure mode you have certainly seen even if you did not know its name: at high guidance, images get **over-saturated** — colors clip toward their extremes, highlights blow out to pure white, shadows crush to black, and everything takes on an over-contrasted, "HDR poster" look. Push CFG to 15–20 on vanilla SD and faces get waxy, skies get unnaturally blue, and the whole image looks cooked. This is not a vague aesthetic complaint; it has a precise mechanical cause.

![A two-column figure contrasting low guidance, which yields diverse but weakly-prompt-matched and muted samples, with high guidance, which yields strongly-prompt-matched but over-saturated and low-diversity samples](/imgs/blogs/classifier-free-guidance-6.png)

The cause: the guided noise estimate $\hat\epsilon = \epsilon_\text{uncond} + (1+w)(\epsilon_\text{cond} - \epsilon_\text{uncond})$ is an *extrapolation*, and extrapolating a vector makes its magnitude grow. Let me make the variance growth explicit, because it is the crux. Write $\delta = \epsilon_\text{cond} - \epsilon_\text{uncond}$ for the guidance direction. Then $\hat\epsilon = \epsilon_\text{cond} + w\,\delta$. The variance of the guided prediction is

$$\operatorname{Var}(\hat\epsilon) = \operatorname{Var}(\epsilon_\text{cond}) + 2w\,\operatorname{Cov}(\epsilon_\text{cond}, \delta) + w^2 \operatorname{Var}(\delta).$$

The $w^2 \operatorname{Var}(\delta)$ term dominates as $w$ grows: the magnitude of the guided noise prediction grows roughly *linearly* in $w$ (its variance grows quadratically). Now recall that the predicted clean image is, for $\epsilon$-prediction, $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$. Inflate $\hat\epsilon$ and you inflate $\hat x_0$ proportionally — its values get pushed past the range the model was trained to produce. The denoiser was trained so that $x_0$ lands in a bounded range (latents are roughly standardized; pixels live in $[-1, 1]$). High CFG inflates the predicted $\hat x_0$ past that range, and when you decode, those out-of-range values map to clipped, saturated colors — the brightest regions slam to pure white, the darkest to pure black. In short: **CFG over-drives the signal, the predicted image's statistics drift away from real-image statistics, and saturation is the visible symptom.** Notice that every one of the three fixes below is, at heart, a way to *cancel that $w^2 \operatorname{Var}(\delta)$ inflation* — by clipping it (dynamic thresholding), by rescaling it away (CFG-rescale), or by only paying it where it helps (limited interval). Three standard fixes attack this, each in a different place.

![A dataflow figure showing the high-guidance over-saturation problem branching into three independent fixes — dynamic thresholding from Imagen, CFG-rescale, and limited-interval guidance — all converging on sharp but natural images with FID restored](/imgs/blogs/classifier-free-guidance-7.png)

**Fix 1 — Dynamic thresholding (Imagen, Saharia et al. 2022).** Imagen generates in *pixel* space and uses very high guidance (scales up to 10+ matter for their cascaded pixel model), so saturation was acute. Their fix: after each step, look at the predicted clean image $\hat x_0$, find a high percentile $s$ of its absolute pixel values (say the 99.5th percentile, with $s \geq 1$), then **clip $\hat x_0$ to $[-s, s]$ and rescale by $1/s$** so it lands back in $[-1, 1]$. This actively pushes the over-driven pixels back into range every step instead of letting saturation accumulate. It is called *dynamic* because the threshold $s$ adapts to each image's statistics rather than being a fixed clamp. Dynamic thresholding let Imagen use high guidance for strong text alignment without the saturated look. (It is specific to pixel-space models with a known value range; in latent space the analog is less standard because latents are not bounded the same way.)

**Fix 2 — CFG-rescale (Lin et al. 2024, "Common Diffusion Noise Schedules and Sample Steps are Flawed").** This one is general and widely used. The observation: guidance inflates the *standard deviation* of the guided prediction relative to the conditional prediction. The fix: after computing $\hat\epsilon$, **rescale it so its standard deviation matches that of the conditional prediction** $\epsilon_\text{cond}$, then blend between the rescaled and the raw guided result with a factor $\phi$ (commonly $\phi = 0.7$). In formula form:

$$\hat\epsilon_\text{rescaled} = \hat\epsilon \cdot \frac{\sigma(\epsilon_\text{cond})}{\sigma(\hat\epsilon)}, \qquad \hat\epsilon_\text{final} = \phi\,\hat\epsilon_\text{rescaled} + (1-\phi)\,\hat\epsilon.$$

This directly counters the variance inflation: you keep the *direction* of the guidance (so fidelity stays high) but tame its *magnitude* (so saturation drops). The diffusers pipelines expose this as `guidance_rescale`. It is especially important together with **zero-terminal-SNR** schedule fixes (the same paper) and is the standard way to run high guidance on SDXL without blowing out the image. We will use it in the code below.

**Fix 3 — Limited-interval / guidance-interval guidance (Kynkäänniemi et al. 2024).** The cleanest recent result: you do *not* need guidance at every step. Guidance at very high noise levels (early steps) mostly hurts — it pushes hard when the image is still mush, inflating statistics — and guidance at very low noise levels (final steps) mostly adds saturation without adding content. The fix: **apply CFG only during a middle interval of timesteps**, and run unconditionally (or with scale 1) outside it. Kynkäänniemi et al. showed this *improves* FID substantially on ImageNet generation while letting you use a higher peak scale, because you concentrate guidance where it helps (mid-noise, where structure is being decided) and skip it where it only saturates. This is now a common knob ("guidance start/stop step") in ComfyUI and several pipelines. The mental model: guidance is a medicine with a therapeutic window, not a thing to apply uniformly.

All three fixes share a goal — **keep CFG's fidelity benefit while removing its variance-inflation / saturation side effect** — and they compose: people routinely run SDXL with a moderate scale, `guidance_rescale=0.7`, zero-terminal-SNR, and sometimes a guidance interval, all at once.

#### Worked example: choosing rescale over raw scale on SDXL

You are shipping a stock-photography generator on SDXL and your users complain that outputs look "fake and over-processed" — the classic blown-out look. Your instinct is to *lower* guidance, but lowering it to scale 4 makes prompts get ignored (a "wooden desk in soft morning light" comes out as a generic desk in harsh light). You are stuck on the trade-off's bad diagonal: low scale loses the prompt, high scale over-saturates. The fix is to *decouple fidelity from saturation* with rescale rather than trading one for the other. Run a small ablation: fix the prompt set and seeds, then compare (a) scale 7, no rescale; (b) scale 9, no rescale; (c) scale 9, `guidance_rescale=0.7`; (d) scale 9, rescale + zero-terminal-SNR schedule. You will typically find (b) has the best prompt adherence but the worst exposure (blown highlights, crushed blacks), (a) is a muddy compromise, and (c)/(d) keep (b)'s adherence while restoring natural exposure — (d) additionally fixing the "SD can't make a truly dark or truly bright image" bias that the zero-terminal-SNR paper documents. Your decision: ship scale 9 with rescale 0.7 and the fixed schedule. The point of the example is the *shape* of the reasoning — when two desiderata fight on one knob, look for a second knob that separates them, rather than settling for a compromise on the first. CFG-rescale is exactly that second knob for the fidelity-versus-saturation fight.

## Negative prompts: the unconditional branch, repurposed

Here is a piece of practical magic that falls straight out of the CFG formula and that every Stable Diffusion user has touched: the **negative prompt**. In the basic CFG formula the unconditional branch uses the null condition $\varnothing$ (an empty prompt). But nothing forces the "negative" branch to be empty. Replace $\epsilon_\text{uncond} = \epsilon_\theta(x_t, t, \varnothing)$ with $\epsilon_\theta(x_t, t, c_\text{neg})$ for some *negative* prompt $c_\text{neg}$, and the guidance becomes:

$$\hat\epsilon = \epsilon_\theta(x_t, t, c_\text{neg}) + (1+w)\,\big[\epsilon_\theta(x_t, t, c_\text{pos}) - \epsilon_\theta(x_t, t, c_\text{neg})\big].$$

Now the difference vector points *away from $c_\text{neg}$ and toward $c_\text{pos}$*. So the model is steered toward your positive prompt and **actively away from** whatever you put in the negative prompt. Put "blurry, low quality, watermark, extra fingers" in the negative prompt and the guidance pushes away from blurriness, low quality, watermarks, and extra fingers. It is the same two-forward-pass machinery — you just feed a meaningful prompt to the branch that used to be empty. This is why negative prompts are "free" (they cost no extra compute beyond the unconditional pass you were already running) and why they are so effective: they reuse the most powerful steering mechanism in the model. Note one consequence: **negative prompts only exist when CFG is on.** A model sampled at scale 1 (no guidance) has no unconditional branch to repurpose, so negative prompts do nothing — and distilled models that removed CFG (next section) often lose negative-prompt support as a result.

## The code: implementing CFG and its fixes

Enough theory. Here is CFG in a raw PyTorch sampling loop so you can see every line of the mechanism, then the one-liner version in diffusers, then the rescale fix.

First, the from-scratch version. The two things to notice: we batch the conditional and unconditional embeddings together (`torch.cat`) so both passes happen in one forward call, and we apply the extrapolation `noise_uncond + guidance_scale * (noise_cond - noise_uncond)` before handing the result to the scheduler.

```python
import torch

@torch.no_grad()
def sample_with_cfg(unet, scheduler, text_embeds, uncond_embeds,
                    guidance_scale=7.5, num_steps=30,
                    latent_shape=(1, 4, 64, 64), device="cuda",
                    dtype=torch.float16, generator=None):
    # guidance_scale here is diffusers' convention == 1 + w.
    scheduler.set_timesteps(num_steps, device=device)
    # Start from pure Gaussian noise, scaled by the scheduler's init sigma.
    latents = torch.randn(latent_shape, generator=generator,
                          device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    # Stack [uncond, cond] so one forward pass does both branches.
    embeds = torch.cat([uncond_embeds, text_embeds], dim=0)  # (2, 77, 768)

    for t in scheduler.timesteps:
        # Duplicate the latent for the two branches.
        latent_in = torch.cat([latents] * 2, dim=0)          # (2, 4, 64, 64)
        latent_in = scheduler.scale_model_input(latent_in, t)

        # One forward pass returns both predictions.
        noise_pred = unet(latent_in, t,
                          encoder_hidden_states=embeds).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)

        # The classifier-free guidance extrapolation.
        noise_guided = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Step the sampler with the guided noise estimate.
        latents = scheduler.step(noise_guided, t, latents,
                                 generator=generator).prev_sample
    return latents
```

That single line `noise_uncond + guidance_scale * (noise_cond - noise_uncond)` is the whole idea of this post. Everything else is plumbing. (`scale_model_input` and `init_noise_sigma` are scheduler-specific normalizations; the structure is identical across `DDIMScheduler`, `EulerDiscreteScheduler`, `DPMSolverMultistepScheduler`, etc.)

In real life you do not hand-roll this — diffusers does CFG internally whenever `guidance_scale > 1`. The entire user-facing surface is two arguments:

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

image = pipe(
    prompt="an astronaut riding a horse on the moon, cinematic, golden hour",
    negative_prompt="blurry, low quality, watermark, extra limbs, oversaturated",
    guidance_scale=7.0,        # == 1 + w; below 1.0 disables CFG (single pass)
    num_inference_steps=30,
).images[0]
image.save("astronaut.png")
```

Set `guidance_scale=1.0` and the pipeline skips the unconditional pass entirely (one forward per step, ~2× faster, but no prompt sharpening and no negative-prompt effect). Set it to 7.0 and you are running standard CFG with the negative prompt steering the "negative" branch.

Now the CFG-rescale fix, to run high guidance without saturation. On SDXL it is a single extra argument; internally the pipeline applies the standard-deviation matching we derived above:

```python
image = pipe(
    prompt="portrait of a fox in a forest, sharp focus, natural light",
    negative_prompt="oversaturated, blown highlights, hdr, cartoon",
    guidance_scale=9.0,        # high, for strong prompt adherence
    guidance_rescale=0.7,      # rescale to tame variance / saturation (phi)
    num_inference_steps=30,
).images[0]
```

If you want to implement the rescale yourself (for a custom loop or a scheduler that does not expose it), it is exactly the formula from the fixes section:

```python
def rescale_cfg(noise_guided, noise_cond, phi=0.7):
    # Match the std of the guided prediction to the conditional one,
    # then blend back toward the raw guided result by (1 - phi).
    std_cond = noise_cond.std(dim=list(range(1, noise_cond.ndim)),
                              keepdim=True)
    std_guided = noise_guided.std(dim=list(range(1, noise_guided.ndim)),
                                  keepdim=True)
    noise_rescaled = noise_guided * (std_cond / std_guided)
    return phi * noise_rescaled + (1.0 - phi) * noise_guided
```

Drop `noise_guided = rescale_cfg(noise_guided, noise_cond)` into the from-scratch loop right after the extrapolation, and you have the saturation fix that lets you crank the scale.

Finally, the limited-interval trick in a custom loop is just a conditional: only guide inside a timestep window.

```python
# Apply CFG only on the middle interval of the schedule (e.g. skip the
# first 15% and last 10% of steps), running single-pass elsewhere.
for i, t in enumerate(scheduler.timesteps):
    frac = i / len(scheduler.timesteps)
    guide = (0.15 <= frac <= 0.90)   # therapeutic window for guidance
    if guide:
        latent_in = torch.cat([latents] * 2, dim=0)
        latent_in = scheduler.scale_model_input(latent_in, t)
        noise_pred = unet(latent_in, t, encoder_hidden_states=embeds).sample
        nu, nc = noise_pred.chunk(2)
        noise = nu + guidance_scale * (nc - nu)
    else:
        latent_in = scheduler.scale_model_input(latents, t)
        noise = unet(latent_in, t,
                     encoder_hidden_states=text_embeds).sample   # cond-only
    latents = scheduler.step(noise, t, latents).prev_sample
```

Outside the window you do a single conditional pass (faster *and* less saturating), which is why limited-interval guidance can simultaneously improve quality and reduce compute.

## Case studies: real numbers from shipped models and papers

Let me ground all of this in named results. I will keep numbers to ones reported in the literature or widely reproduced, and flag anything approximate.

**Classifier guidance on ImageNet (Dhariwal & Nichol 2021).** Their ADM model with classifier guidance was the first diffusion result to beat the best GANs (BigGAN-deep) on ImageNet FID. At 256×256, guided ADM reached an FID around 4.6 (and lower with their ADM-G/-U variants), versus BigGAN-deep's ~6.95 — the headline "Diffusion Models Beat GANs" result. The guidance scale traced the now-familiar fidelity–diversity frontier: turning it up improved Inception Score (fidelity) while reducing recall (diversity). This is the result that established *scaling the classifier gradient* as the right idea — and motivated removing the classifier.

**Classifier-free guidance (Ho & Salimans 2022).** On the same class-conditional ImageNet setting, CFG matched or beat classifier guidance *without any classifier*, sweeping out the same FID-versus-IS frontier as the guidance weight varied. The paper's core demonstration is precisely the trade-off curve: small guidance weights give best FID (diverse), larger weights give best Inception Score / fidelity, and you pick your point on the frontier. This is the paper that made guidance the default and killed the separate classifier.

**Stable Diffusion 1.5 / SDXL defaults.** The practical settling point for text-to-image: SD 1.5 ships with `guidance_scale=7.5`, SDXL with 7.0 (the SDXL technical report and the diffusers defaults). Community practice clusters at 5–8 for photorealism. SDXL at base resolution 1024×1024 with ~30 steps of a DPM-Solver++ sampler and scale ~7 is the canonical "good defaults" recipe.

**CFG-rescale + zero-terminal-SNR (Lin et al. 2024).** The "Common Diffusion Noise Schedules… are Flawed" paper showed that fixing the schedule's terminal SNR to zero *and* applying CFG-rescale (with $\phi \approx 0.7$) removes a systematic brightness/saturation bias and improves results, particularly for very-light and very-dark images that vanilla SD struggles to produce. This is now standard practice for fine-tunes that want correct exposure at high guidance.

**Limited-interval guidance (Kynkäänniemi et al. 2024).** On ImageNet-512 with EDM2, restricting guidance to a middle noise interval improved FID markedly over guiding at every step — they reported a new state-of-the-art FID on ImageNet generation at the time, from the *same* model, purely by changing *when* guidance is applied. The lesson generalizes: guidance is most useful at intermediate noise levels.

**SD3 and FLUX (2024) — guidance distillation.** This is the frontier twist. SDXL-Turbo and SD3-Turbo (via Adversarial Diffusion Distillation, Sauer et al. 2023/2024) and the FLUX *schnell/dev* family use **guidance distillation**: instead of running CFG (two passes) at inference, they *distill* the guided behavior into the network so a single forward pass already produces the guided output. FLUX-dev takes a `guidance_scale` as a *conditioning input* the model was trained to respect, not a two-pass extrapolation; FLUX-schnell is distilled to a few steps with guidance baked in and effectively ignores a CFG scale. The payoff is large: you reclaim the ~2× CFG tax, which together with step distillation is how Turbo/schnell models hit 1–4 step generation. The relationship to this post: **CFG is the teacher behavior these models learn to imitate in one pass** — guidance did not go away, it got compiled into the weights. We cover that machinery in the [distribution-matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) post.

It is worth dwelling on *why* guidance distillation is hard enough to be a research topic and not a one-line trick, because it reveals what CFG really is. The naive idea — "just train the network to output the guided prediction directly" — runs into the problem that the guided prediction depends on $w$, which is a free parameter. So a guidance-distilled student must either bake in a *single* fixed $w$ (giving up the runtime knob, the FLUX-schnell choice) or take $w$ as an *input* and learn the whole family of guided behaviors (the FLUX-dev choice, more flexible but harder to train). Either way the student is learning a function the teacher only ever produced via two passes, and getting it to match across all noise levels and all prompts — without the diversity collapsing the way it does at high live-CFG — is exactly the delicate part. This is also why distilled models sometimes feel "stuck" at one aesthetic: the distillation effectively chose a guidance operating point for you and froze it. When you hear someone say "Turbo has less range than base SDXL," the frozen guidance is a big part of what they are feeling.

**A clarifying contrast: live CFG versus distilled guidance in one table.** The two are easy to conflate because both deliver "guided-looking" images, but they behave completely differently at the dials, and knowing which you are holding determines what knobs even exist.

| Aspect | Live CFG (SD1.5, SDXL, SD3) | Distilled guidance (Turbo, FLUX-schnell) |
|---|---|---|
| Forward passes per step | 2 (cond + uncond) | 1 |
| Guidance scale | A real runtime knob you tune | Fixed, or a weak conditioning input |
| Negative prompt | Supported (repurpose uncond branch) | Often ignored / no effect |
| Steps | 20–50 typical | 1–4 |
| Relative cost/image | Baseline (the ~2× tax) | ~5–15× cheaper |
| Best for | Creative tools, research, tuning | Latency-critical APIs, real-time |

The single most common confusion in practice is a user setting `guidance_scale=7` on a distilled model and wondering why nothing changes — the answer is that the model is not doing the two-pass extrapolation that scale controls; it learned a fixed guided behavior, and the scale argument is either ignored or only mildly effective. Always know which regime your model is in.

Here is the comparison consolidated:

| Method | Extra model? | Passes/step | Works for text? | Headline result | When to use |
|---|---|---|---|---|---|
| Classifier guidance (2021) | Yes (noisy classifier) | 1 + classifier grad | No (fixed labels) | Beat GANs on ImageNet, FID ~4.6 | Almost never now; historical |
| Classifier-free guidance (2022) | No | 2 (cond + uncond) | Yes | The default for all T2I | Standard; scale 5–8 |
| CFG + rescale (2024) | No | 2 | Yes | Removes saturation/exposure bias | High guidance, correct exposure |
| Limited-interval CFG (2024) | No | 2 in window, 1 outside | Yes | SOTA FID on ImageNet-512 | Quality + some speed |
| Guidance distillation (2024) | No (distilled once) | 1 | Yes | Reclaims the 2× tax, few-step | Turbo/schnell, latency-critical |

And the guidance lineage as a timeline:

![A left-to-right timeline tracing guidance from classifier guidance in 2021 through classifier-free guidance in 2022, dynamic thresholding and CFG-rescale, to guidance distillation baked into Turbo and FLUX weights by 2024](/imgs/blogs/classifier-free-guidance-8.png)

#### Worked example: the cost of CFG and why distillation pays off

Put the doubled-cost claim in dollars and seconds. Suppose SDXL on an A100 80GB does one denoiser forward pass over a 1024×1024 latent in about 35 ms at fp16. With 30 steps and CFG on (batch the two branches, so effectively 2× the FLOPs per step), a single image takes roughly $30 \times 35\,\text{ms} \times 2 \approx 2.1$ seconds of denoiser compute (plus VAE decode and overhead, call it ~2.5 s end to end). Turn CFG off (scale 1) and the same 30 steps take ~1.05 s of denoiser compute — half — but you lose prompt sharpening and negative prompts, so the images are worse. That ~2× tax, multiplied across a serving fleet, is real money. At a rough \$2/hr for an A100, ~2.5 s/image is about \$0.0014/image; halving the denoiser cost is a meaningful fraction of that, and on top of it you can distill the *step count* from 30 down to 4. That is the whole economic argument for guidance distillation: a Turbo/schnell model that bakes guidance into one pass and runs in 4 steps can be 10–15× cheaper per image than 30-step CFG-on SDXL, which is why latency-critical products ship distilled models even though they give up the live CFG/negative-prompt knob. The trade you are making: a *tunable* knob (live CFG scale, negative prompts) for a *fixed, fast* one (distilled guidance). For interactive/API products at scale, fixed-and-fast usually wins; for a creative tool where users tweak guidance and negative prompts, keep live CFG.

## When to reach for guidance (and how to set it)

A decisive section, because "what scale should I use" is the most-asked question in the whole field.

- **Default to scale 5–8 with CFG on.** For almost any text-to-image use, this is the right starting point: peak-ish CLIP-score, near-minimum FID, acceptable diversity. SD 1.5 → 7.5, SDXL → 7.0, are fine defaults. Do not overthink it until you have a measured reason to.
- **Use a negative prompt — it is free quality.** Since the unconditional pass runs anyway when CFG is on, fill it with "blurry, low quality, watermark, extra limbs, oversaturated, jpeg artifacts" (or domain-specific negatives). It steers away from common failure modes at zero extra cost. Skip it only on distilled models that dropped CFG.
- **Going above ~8? Turn on CFG-rescale (`guidance_rescale≈0.7`) and zero-terminal-SNR.** High guidance without rescale gives you the over-saturated poster look. With rescale you can run scale 9–12 for hard prompts and keep natural exposure.
- **Don't crank the scale to "force" a complex prompt — it mostly fails and saturates.** If the model can't bind three attributes or count to four at scale 7, scale 15 won't fix the *binding*; it will just over-saturate and collapse diversity. Counting and attribute-binding are denoiser/text-encoder limitations (see [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning)), not guidance limitations. Reach for a better model or regional/structural control, not a bigger $w$.
- **Latency-critical and you don't need a live knob? Use a distilled model.** SDXL-Turbo, SD3-Turbo, FLUX-schnell bake guidance into one pass and run in 1–4 steps. You give up the live CFG scale and negative prompts; you gain ~10× throughput.
- **Doing research / ImageNet-style class generation? Use limited-interval guidance.** Restricting CFG to a mid-noise window improves FID and lets you use a higher peak scale. It is close to a free lunch for quality.
- **CFG off (scale 1) is rarely what you want — except as a baseline or on distilled models.** Without guidance, samples are diverse but washed-out and prompt-ignoring. The only times to run scale 1 are: measuring the unguided baseline, or running a model that distilled guidance away.
- **For image-to-image and inpainting, drop to scale 4–6.** The denoiser already has the source image as an anchor, so less prompt-pull is needed; high guidance over-imposes the prompt and produces boundary artifacts.

A short decision recipe you can apply in thirty seconds. *Is the model distilled (Turbo/schnell)?* If yes, accept its baked-in guidance, do not expect the scale knob or negative prompts to do much, and move on. *If it is a live-CFG model, is this text-to-image from noise?* If yes, start at scale 7, add a negative prompt, and only deviate with a measured reason. *Are outputs over-saturated?* Turn on `guidance_rescale=0.7` (and zero-terminal-SNR if your schedule supports it) rather than lowering the scale. *Are complex prompts failing?* Do not crank the scale — the failure is binding/counting, not guidance strength; reach for a stronger text encoder, regional prompting, or structural control instead. *Is it image-to-image?* Drop to scale 4–6. That tree covers the vast majority of real decisions, and notice that in *none* of them is "raise the scale past 12" the right answer — high scale is almost always the wrong tool reached for the right problem.

## Stress-testing the picture

Let me poke at the edges to make sure the picture is robust.

**What happens at extreme guidance ($w \to \infty$)?** The $p(x)$ factor in equation (3) becomes negligible relative to $p(c\mid x)^{1+w}$, so you sample almost purely from "maximize the implicit classifier," ignoring realism. You get the adversarial-example pathology *back* — over-saturated, artifact-laden images that maximally satisfy the model's notion of the prompt while drifting off the natural-image manifold. This is the high-CFG failure mode in its purest form, and it is why guidance has an upper useful bound.

**What happens at scale exactly 1 ($w = 0$)?** You get honest conditional sampling from $p(x \mid c)$: diverse, but loosely on-prompt. And critically, the unconditional pass is unused, so diffusers skips it (single pass, ~2× faster). Negative prompts have no effect here.

**What if the unconditional branch is bad?** CFG's whole premise is that $\epsilon_\text{cond} - \epsilon_\text{uncond}$ is a *meaningful* direction. If condition-dropout was not done during training (the model never learned a good $\epsilon_\theta(x_t,t,\varnothing)$), the unconditional pass is garbage and guidance steers along a meaningless vector. This is why the 10–20% dropout during training is not optional — it is what *creates* the unconditional branch that guidance needs. A model trained with 0% dropout cannot be guided well.

**Does CFG interact with the sampler and step count?** Yes. Guidance and the sampler are *composable but not independent*. Very few steps (4–8) plus high CFG is a bad combination on non-distilled models — the over-driven predictions compound across too-large steps and saturate badly. This is part of why few-step generation needed distillation rather than just "fewer steps + more guidance." The samplers post ([samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive)) covers the step-count side; the takeaway here is that low-step regimes want guidance distilled in, not cranked up live.

**Why not just train on a sharpened objective directly?** Suppose you trained the model to directly output sharpened samples. But then $w$ is fixed at training time — you lose the *runtime knob*. CFG's entire practical value is that one trained model gives you a whole *family* of distributions parameterized by $w$, selectable at inference. That flexibility (and the negative-prompt trick) is why CFG won over baking the sharpening into training — right up until distillation made the *fixed* trade worth it for speed.

**Does guidance behave the same on flow-matching models (SD3, FLUX)?** The *form* changes but the idea survives. SD3 and FLUX are trained with flow matching, so the network predicts a velocity field $v_\theta(x_t, t, c)$ rather than an $\epsilon$. The guidance extrapolation applies to the velocity instead of the noise: $\hat v = v_\text{uncond} + w\,(v_\text{cond} - v_\text{uncond})$, and the same fidelity-versus-diversity and saturation considerations hold (flow-matching samplers can over-shoot just like $\epsilon$-samplers). FLUX-dev goes a step further and takes the guidance scale as a *learned input* — it was trained to internalize the guided behavior, so a single pass already produces a guided velocity and there is no separate unconditional pass at inference. That is guidance distillation in its purest productized form; the knob is still there nominally, but it is steering a network that learned to imitate CFG rather than performing the live two-pass extrapolation. The takeaway: don't assume "this is a flow model, so CFG doesn't apply" — check whether the model does live CFG (two passes, you set the scale and a negative prompt) or distilled guidance (one pass, the scale is a conditioning input and negative prompts may be ignored). The behavior at the dials is completely different.

**What if two seeds give nearly identical images at high CFG — is that a bug?** No, that is diversity collapse working as predicted. At high $w$ the sharpened distribution $p(x)\,p(c\mid x)^{1+w}$ is so peaked that almost all its mass sits on a handful of mode images, so different noise seeds converge to the same few outputs. If you *need* variety at high guidance (e.g. a "generate 8 options" UI), your options are: lower the scale, add a guidance interval (which preserves more early-step diversity), vary the prompt slightly across seeds, or use a model with distilled guidance tuned for variety. Cranking the scale and *also* expecting variety is asking the trade-off to violate itself.

**How does guidance interact with image-to-image and inpainting?** It still applies — you guide the denoiser exactly the same way — but the *useful range shifts*. In image-to-image you start from a partially-noised real image, so there is less to "decide" and high guidance over-imposes the prompt on content that was supposed to be preserved, often producing artifacts at the boundary between preserved and generated regions. Practitioners typically run *lower* guidance (scale 4–6) for image-to-image and inpainting than for text-to-image from scratch, precisely because the denoiser has more anchoring information and needs less prompt-pull to stay on track. This is a good reminder that "scale 7 is the default" is a text-to-image-from-noise default, not a universal constant.

**What if you stack guidance from multiple conditions (text + ControlNet + IP-Adapter)?** Modern pipelines often guide on several conditions at once — a text prompt, a depth map via ControlNet, a reference image via IP-Adapter. Each contributes its own conditional-versus-unconditional difference, and the pipeline composes them, sometimes with per-condition scales. The same trade-off logic applies per channel: too much text guidance over-saturates; too much ControlNet conditioning makes the output rigidly trace the control image and look unnatural; too much IP-Adapter scale over-copies the reference. The art of multi-condition generation is balancing these scales so no single condition dominates — and the failure modes are exactly the per-condition versions of the over-guidance pathology we derived. The mechanism generalizes cleanly: every conditioning signal has a guidance strength, every guidance strength has a fidelity-versus-diversity-versus-naturalness trade-off, and the variance-inflation intuition tells you which way each one fails when pushed too hard. (The composition details live in the [ControlNet](/blog/machine-learning/image-generation/controlnet-and-structural-control) and reference-conditioning posts.)

Step back and the whole topic collapses to one sentence worth memorizing: **guidance is a temperature on the conditional distribution, paid for in a second forward pass, and every practical wrinkle — saturation, diversity collapse, negative prompts, rescale, distillation — is a consequence of that single fact.** The temperature view tells you why fidelity and diversity trade off (sharpening concentrates mass). The second-forward-pass view tells you why it costs 2× and why distillation exists (to reclaim that cost). And the extrapolation view — going *past* the conditional, away from the unconditional — tells you why the magnitude inflates and the colors blow out, and why every fix is some way of taming that magnitude. Hold those three views together and you can predict how any guidance knob will behave before you ever run the model.

## Key takeaways

- **Conditioning** a diffusion model means training $\epsilon_\theta(x_t, t, c)$ — usually via cross-attention on encoded text — with the condition randomly **dropped** 10–20% of the time so the model also learns the unconditional $\epsilon_\theta(x_t, t, \varnothing)$.
- Honest conditional sampling from $p(x \mid c)$ is **too weak** — broad and loosely on-prompt. We want the **sharpened** distribution $p(x)\,p(c\mid x)^{1+w}$.
- **Classifier guidance** (2021) sharpened via a separate noisy-image classifier's gradient — clever, but fragile, off-manifold-prone, and useless for open text.
- **Classifier-free guidance** (2022) gets the same implicit classifier gradient for free as $\epsilon_\text{cond} - \epsilon_\text{uncond}$, giving the extrapolation $\hat\epsilon = \epsilon_\text{uncond} + (1+w)(\epsilon_\text{cond} - \epsilon_\text{uncond})$. One network, two passes, no classifier.
- The **derivation**: CFG's effective score integrates to a density $\propto p(x)\,p(c\mid x)^{1+w}$ — the data density times the likelihood raised to $1+w$, i.e. a temperature-sharpened conditional.
- The **knob**: higher $w$ raises fidelity (CLIP-score) and lowers diversity; FID is U-shaped with a **sweet spot near scale 5–8** for text-to-image.
- **Over-saturation** at high guidance comes from variance inflation in the extrapolated prediction; fix it with **dynamic thresholding** (Imagen), **CFG-rescale** (`guidance_rescale≈0.7`), or **limited-interval** guidance.
- **Negative prompts** are the unconditional branch repurposed — free, powerful, and only available when CFG is on.
- CFG costs **~2× compute per step**; the 2024–2026 frontier (Turbo, FLUX-schnell) **distills guidance into one pass**, trading the live knob for speed.

## Further reading

- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis," NeurIPS 2021 — the classifier-guidance paper.
- Ho & Salimans, "Classifier-Free Diffusion Guidance," NeurIPS 2021 Workshop / 2022 — the CFG paper; read it for the fidelity–diversity frontier plots.
- Saharia et al., "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding" (Imagen), NeurIPS 2022 — dynamic thresholding and high-guidance text-to-image.
- Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed," WACV 2024 — CFG-rescale and zero-terminal-SNR.
- Kynkäänniemi et al., "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models," 2024 — guidance-interval guidance.
- Sauer et al., "Adversarial Diffusion Distillation" (SDXL-Turbo), 2023/2024 — guidance and step distillation for few-step generation.
- 🤗 `diffusers` docs — `guidance_scale`, `negative_prompt`, and `guidance_rescale` on the pipelines.
- Within this series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) (the reverse process), [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) (the score identity guidance manipulates), [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) (where the condition comes from), [distribution-matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) (guidance distillation), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
