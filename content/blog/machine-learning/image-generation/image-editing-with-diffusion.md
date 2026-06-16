---
title: "Image Editing With Diffusion: Inversion, Attention Control, and Instructions"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to edit a real photograph with a diffusion model: SDEdit's noise-strength knob, mask-based inpainting, DDIM and null-text inversion, prompt-to-prompt cross-attention control, and InstructPix2Pix — the science, runnable diffusers code, and where each one breaks."
tags:
  [
    "image-generation",
    "diffusion-models",
    "image-editing",
    "sdedit",
    "ddim-inversion",
    "prompt-to-prompt",
    "instructpix2pix",
    "inpainting",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/image-editing-with-diffusion-1.png"
---

Here is a request that sounds trivial and is not: "Take *this exact photo* of my cat sitting on a windowsill, and make the cat a dog. Don't move the cat. Don't change the windowsill, the light, the curtains, or the pose. Just swap the animal." A text-to-image model can draw you a dog on a windowsill all day long — but it will be a *different* dog, on a *different* windowsill, in a *different* pose. That is generation, not editing. Editing means starting from a real image you already have and changing one thing while holding everything else fixed. And that turns out to be one of the hardest problems in the whole diffusion stack, because a diffusion model is built to start from pure noise, not from your photo.

The reason it is hard is worth stating plainly up front. A diffusion model is a function that maps noise to images. To edit a real image you first have to answer a question the model was never trained to answer: *what noise would have produced this picture?* If you can recover that noise — the latent seed that the model's reverse process turns into your exact photo — then you can perturb it, or perturb the prompt, and resample to get a controlled change. If you cannot recover it faithfully, every edit you make also silently corrupts the parts you wanted to keep. This "recover the latent for a real image" step is called **inversion**, and the drift it suffers is the villain of half this post.

![A dataflow figure showing a real photo encoded to a latent, inverted to a starting noise, perturbed by a prompt or attention edit, and resampled into an edited image that keeps the original structure](/imgs/blogs/image-editing-with-diffusion-1.png)

This post is the editing chapter of the series, and it covers the classic, pre-2024 toolkit — the methods that still quietly underpin most of what ships today. By the end you will understand and be able to run: **SDEdit** (Meng et al., 2021), which adds a controlled amount of noise to your image and denoises with a new prompt, and the noise-strength knob that trades fidelity for change; **inpainting and outpainting** with a mask, which blends known and generated latents at every step; **DDIM inversion**, which recovers a starting latent for a real image, and the exactness condition that classifier-free guidance breaks; **null-text inversion** (Mokady et al., 2022), which optimizes the null embedding to make inversion exact again so you can edit at real guidance scales; **prompt-to-prompt** (Hertz et al., 2022), which edits by manipulating the cross-attention maps that bind words to image regions; and **InstructPix2Pix** (Brooks et al., 2022), which trains a model to follow edit *instructions* by generating a synthetic before/instruction/after dataset using prompt-to-prompt. We will see exactly where each one breaks — large structural edits, identity preservation, multi-object scenes — and why those failures set up the 2025 instruction-editing wave that the [next post](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) cleans up.

Where does editing sit in our running frame? Recall the **diffusion stack**: data → VAE latent → forward noising → denoiser net → ODE/SDE sampler → guidance → image. Editing reaches into the *middle* of that stack. Instead of starting the reverse process from fresh noise, it starts from a latent derived from your image, and it intervenes in the sampler, the guidance, or the cross-attention to steer where the change lands. And recall the **generative trilemma** — quality × diversity × speed. Editing adds a fourth axis the trilemma never had to worry about: **fidelity to a specific input**. Almost every design decision below is a fight over how much of the original you keep versus how much you let the model change.

## The core problem: a diffusion model only knows how to start from noise

Let me make the difficulty concrete with the running example. We have a 512×512 photo of a tabby cat on a windowsill. We want a golden retriever in exactly the same spot. The obvious thing — prompt "a golden retriever on a windowsill," sample from noise — produces a perfectly good but *unrelated* image. None of the original geometry survives. The window is in a different place, the light comes from a different angle, the framing is different. That is because sampling from noise explores the whole distribution of "dog on a windowsill" images; it has no idea which one you started from.

So the entire game of editing is: **get the model to start from something derived from your image, not from random noise, and then change as little as possible.** There are three broad families of how to do that, and they map onto the three places you can intervene in the stack.

The first family perturbs the **latent**: take your image, push it partway back toward noise, and denoise it forward again with a new prompt. That is SDEdit, and it is the simplest thing that works. The second family constrains the **sampler with a mask**: regenerate inside a masked region while continuously re-injecting the known pixels outside it. That is inpainting. The third family intervenes in the **conditioning and attention**: recover the exact latent for your image (inversion), then while resampling, surgically copy or rescale the cross-attention maps so the model changes one word's worth of content and keeps the rest. That is prompt-to-prompt and null-text inversion. InstructPix2Pix then amortizes all of this into a single feed-forward model so you do not pay the per-edit cost at inference time.

Each of these makes a different bet about what "as little as possible" means, and each breaks differently. The figure above is the spine: encode the image, recover or noise a latent, perturb the prompt or attention, resample while reusing structure. Hold that loop in mind; every method below is a specialization of it. Let me build them up one at a time, starting with the one you can run in three lines.

## SDEdit: noise it, then denoise it with a new prompt

SDEdit (Stochastic Differential Editing, Meng et al., 2021, "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations") is the editing method you can explain on a napkin. The forward diffusion process gradually destroys an image by adding Gaussian noise. After enough steps, any image becomes indistinguishable from pure noise. SDEdit's insight: you do not have to go all the way. Stop the noising *partway*, then run the reverse (denoising) process from there with whatever prompt you want.

Here is why that produces an *edit* rather than a fresh image. The forward process at a moderate noise level still leaves the coarse structure of the original visible underneath the noise — the rough layout, the dominant colors, the large shapes. The fine details are gone, but the gist is there. When you denoise from that point, the model fills the destroyed details back in, and it fills them in *consistent with your new prompt*. The coarse structure survives because it was never fully destroyed; the fine content gets rewritten according to the prompt. The result is your composition, repainted.

The single most important parameter is the **noise strength**, usually written $s \in [0, 1]$ (in 🤗 `diffusers` it is the `strength` argument of the img2img pipeline). It says how far along the forward process you go before turning around. Concretely, if your sampler uses $T$ total steps, SDEdit re-enters the reverse process at timestep $t_0 = s \cdot T$ and runs the remaining $t_0$ steps. So $s = 0.8$ means "destroy 80% of the way to noise, then denoise the last 80% of the trajectory with the new prompt." Small $s$ keeps the original almost intact and barely applies the prompt; large $s$ lets the prompt dominate but discards the original's identity.

![A two-column before-and-after figure contrasting low-strength SDEdit which keeps the layout and weakly applies the prompt against high-strength SDEdit which follows the prompt but drifts from the original](/imgs/blogs/image-editing-with-diffusion-2.png)

### The science: which noise level you re-enter at decides what is preserved

Let me make the strength knob precise, because the intuition above hides a real quantitative law. Recall the closed-form forward marginal of a diffusion model (DDPM parameterization): a clean image $x_0$ is noised to

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),$$

where $\bar\alpha_t = \prod_{i=1}^{t}(1 - \beta_i)$ is the cumulative product of the noise schedule and decreases monotonically from near 1 (almost no noise) to near 0 (almost pure noise) as $t$ goes from 0 to $T$. The **signal-to-noise ratio** at level $t$ is

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}.$$

This single number is the dial. At the re-entry level $t_0 = sT$, the image $x_{t_0}$ retains the signal $\sqrt{\bar\alpha_{t_0}}\, x_0$ scaled by the square root of that SNR. The frequencies that survive at a given noise level are the *low* spatial frequencies — the large-scale structure — because the diffusion forward process destroys high-frequency detail first (a well-known property: the per-pixel noise variance is uniform across frequencies, but the *signal* power in natural images is concentrated in low frequencies, so high frequencies hit the noise floor much sooner). That is the precise reason SDEdit preserves layout and overwrites detail: at $s = 0.3$, only the finest texture is below the noise floor and gets rewritten; at $s = 0.8$, even mid-scale structure is gone and the model is free to re-compose it.

So the trade-off is not vague. There is a sweet spot $s^*$ where the surviving SNR is high enough to pin the composition you care about but low enough that the destroyed band is exactly the content you want the new prompt to control. For a "change the texture/style, keep the shape" edit, $s \approx 0.3$–$0.5$ usually works. For "change the subject," you need $s \approx 0.6$–$0.8$, and you accept that the original identity will not survive. There is no value of $s$ that both fully changes the subject *and* perfectly preserves the original layout — that is the fundamental limitation of SDEdit, and it is *why* the attention-control methods later in this post exist.

It is worth being explicit about the SDE that gives SDEdit its name, because it explains *why* re-entering partway is principled rather than a hack. The forward noising process is the variance-preserving SDE $dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw$, whose marginal at any time $t$ is exactly the Gaussian $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ we wrote above. The reverse SDE that undoes it is $dx = \big[-\tfrac{1}{2}\beta(t)x - \beta(t)\nabla_x \log p_t(x)\big]dt + \sqrt{\beta(t)}\,d\bar w$, where $\nabla_x \log p_t(x)$ is the score the denoiser estimates. SDEdit's claim is that you can *initialize the reverse SDE at any intermediate time* $t_0$, seeded with the forward-noised input, and integrate down to 0 — and you will land on a sample that is both realistic (because you followed the true reverse dynamics) and faithful to the input (because you started from its noised version). The strength $s$ is just the choice of $t_0$. The reason a *new prompt* takes effect is that the score $\nabla_x \log p_t(x \mid c)$ you integrate with is now conditioned on the edited prompt $c$, so the reverse trajectory bends toward the new conditional distribution while inheriting its starting point from your image. That is the whole method in one sentence: re-enter the prompt-conditioned reverse SDE partway, from your noised image.

The frequency argument deserves one more turn, because it is the most actionable intuition for choosing $s$. Natural images have a power spectrum that falls off roughly as $1/f^2$ — most of the energy is in low spatial frequencies (the big shapes), very little in high frequencies (the fine texture). The forward process adds *white* noise (flat across frequencies). So as you noise, the high-frequency bands of the image cross below the noise floor first — they are "erased" early — while the low-frequency bands survive much longer. At a given strength $s$, there is a cutoff frequency below which the image's signal still dominates the noise and above which the noise has won. Denoising then *rewrites everything above the cutoff* according to the prompt and *preserves everything below it* from the input. This is why SDEdit is a coarse-to-fine editor: low strength only rewrites the finest detail (good for texture/style), high strength pushes the cutoff down into the mid and low frequencies (rewrites shapes, so you can change the subject — but you have also given up the layout). Choosing $s$ is choosing that cutoff frequency, and choosing the cutoff is choosing what survives.

#### Worked example: the strength sweep on the cat photo

Take the 512×512 cat photo, prompt "a golden retriever sitting on a windowsill," 50 sampling steps, `guidance_scale=7.5`, fixed seed. Sweep `strength` and watch what happens.

- **`strength=0.2`** (re-enter at step 40 of 50, run 10 steps): the image is still clearly the *cat*. The fur picks up a faint golden tint; the prompt barely registers. Useless for a subject swap, but perfect if your goal were "warm up the color grade."
- **`strength=0.5`** (re-enter at step 25, run 25 steps): a chimera — dog-ish head, cat-ish body, the windowsill intact. This is the "uncanny middle" everyone who has played with img2img recognizes. The composition is preserved but the subject is half-converted.
- **`strength=0.75`** (re-enter at step ~12, run 38 steps): a convincing golden retriever, in roughly the original pose, on a windowsill that resembles the original but is not pixel-identical. The light direction usually survives; the exact curtain folds do not.
- **`strength=1.0`** (re-enter at step 0): pure text-to-image. A nice dog, zero relationship to your photo.

The actionable rule: **start at `strength=0.6` for a subject change and tune ±0.15**. If the edit is too weak, raise it; if the original is being destroyed, lower it. There is no free lunch — this is one number trading two things you both want.

### The code: SDEdit in three lines of diffusers

In 🤗 `diffusers`, SDEdit *is* the image-to-image pipeline. The `strength` argument is exactly the $s$ above.

```python
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

init_image = load_image("cat_on_windowsill.png").resize((512, 512))

# SDEdit: noise the init image to t0 = strength * num_inference_steps,
# then denoise with the new prompt.
edited = pipe(
    prompt="a golden retriever sitting on a windowsill, photo",
    image=init_image,
    strength=0.6,            # the s knob: 0 = keep input, 1 = ignore input
    guidance_scale=7.5,
    num_inference_steps=50,  # actual denoise steps run = round(strength * 50)
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

edited.save("cat_to_dog_sdedit.png")
```

Two implementation details that bite people. First, the *actual* number of denoising steps run is `round(strength * num_inference_steps)`, not `num_inference_steps`. At `strength=0.6` and 50 steps you only run 30 reverse steps, so if you want 50 effective steps you must raise `num_inference_steps` accordingly. Second, the init image is encoded by the VAE into a 4×64×64 latent *before* noising — SDEdit operates entirely in latent space (this is latent diffusion, covered in the [latent diffusion post](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion)), so the noise is added to the latent, not the pixels. The VAE round-trip itself loses a little high-frequency detail; for editing that is usually invisible, but it is one reason even `strength=0` is not a perfect identity.

SDEdit's beauty is that it needs *no inversion, no training, no mask* — just a forward noising and a denoise. Its curse is the fidelity-versus-change trade-off has no escape: you cannot make a large semantic change while keeping the layout pixel-faithful. For that we need to either constrain *where* the change happens (a mask) or *what* the change touches (attention).

## Inpainting and outpainting: constrain the change with a mask

If SDEdit's problem is that it changes *everything* a little, inpainting's answer is to change *one region* completely while leaving the rest *exactly* untouched. You provide a binary mask: white where you want new content, black where you want the original kept. "Remove the lamppost from this street photo." "Add a hat to this person's head." "Extend this image to the left" (that last one is outpainting — inpainting with the mask on a region of blank canvas you appended).

The mechanism is elegant and is the reason inpainting preserves the unmasked region *perfectly*, not approximately. At every step of the reverse process, the model denoises the whole latent — but then, before moving to the next step, you **overwrite the unmasked region** with a correctly-noised version of the *original* image at that exact noise level. So the known region is never left to the model's imagination; it is continuously re-injected from the ground truth, noised to match wherever the diffusion is in its trajectory. Only the masked region is free to be generated.

![A vertical stack figure showing the per-step inpainting blend where the known region is renoised from the original and the masked region comes from the model, blended by the mask every step](/imgs/blogs/image-editing-with-diffusion-4.png)

### The science: blending known and generated latents at every step

Let $m$ be the binary mask (1 = generate, 0 = keep), $x_0^{\text{known}}$ the original clean image (or its latent), and $x_t^{\text{gen}}$ the model's running generation at step $t$. The blend that defines inpainting is, at every reverse step:

$$x_{t-1}^{\text{known}} = \sqrt{\bar\alpha_{t-1}}\, x_0^{\text{known}} + \sqrt{1 - \bar\alpha_{t-1}}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),$$

$$x_{t-1} = m \odot x_{t-1}^{\text{gen}} + (1 - m) \odot x_{t-1}^{\text{known}}.$$

The first line *renoises the original* to the next noise level $t-1$ — it is the SDEdit forward marginal again, applied to the known content. The second line composites: inside the mask, take the model's generated latent; outside the mask, take the renoised original. This is the algorithm from Lugmayr et al.'s RePaint (2022), and a simpler variant is what `diffusers`' inpainting pipeline does.

Why renoise the known region at *every* step instead of just pasting the clean original back at the end? Because the masked region needs to be *consistent* with the surroundings at every noise level, not just the final one. The model's self-attention and convolutions look at the whole latent; if the unmasked region were clean while the masked region were noisy, the boundary would be wildly out of distribution and you would get a visible seam. By keeping the known region noised to the *same* level as the generated region, the model sees a coherent, in-distribution latent and blends the new content into the old seamlessly. This is the precise reason RePaint also does *resampling* — jumping back and forth a few steps to give the masked content more chances to harmonize with the re-injected known region.

There is a subtlety that explains a common failure. The model only ever *sees* the renoised known region; it does not get to change it. So if your prompt asks for something that is *globally* inconsistent with the unmasked content — "make it nighttime" while inpainting only a small patch — the patch will fight the surroundings and lose, because the surroundings are pinned. Inpainting is for *local* edits whose context is the unmasked image. It cannot do a global relight; for that you want SDEdit or a dedicated relighting model.

### The code: a diffusers inpainting pipeline with a mask

```python
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",   # a model fine-tuned for inpainting
    torch_dtype=torch.float16,
).to("cuda")

image = load_image("street.png").resize((512, 512))
mask  = load_image("lamppost_mask.png").resize((512, 512))  # white = regenerate

# Inpainting: regenerate only the white region; the black region is re-injected
# from the original at every denoising step, so it is preserved exactly.
result = pipe(
    prompt="empty sidewalk, clean street, photo",
    image=image,
    mask_image=mask,
    guidance_scale=8.0,
    num_inference_steps=40,
    strength=1.0,           # full regeneration inside the mask
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

result.save("lamppost_removed.png")
```

Two practical notes. The dedicated `stable-diffusion-inpainting` checkpoint takes a 9-channel UNet input (4 latent + 4 masked-image latent + 1 mask) and is fine-tuned to use the mask, so it produces noticeably cleaner fills than naively masking a base model. And for **outpainting**, you literally pad the image with blank canvas, set the mask to white over the new canvas, and inpaint — the model extends the scene into the empty region, conditioned on the original via the unmasked border. The same blend equation handles both; outpainting is just inpainting where the mask happens to be on the frame's edge.

Outpainting deserves a few words on its own because it stresses the method in a way object removal does not. When you extend an image to the right, the model has *only* the left border as context, and it has to hallucinate a plausible continuation. The further you extend in one shot, the weaker the constraint from the border and the more the generation drifts — the new region's lighting, perspective, and content slowly diverge from the original. The standard fix is to **outpaint in overlapping tiles**: extend by a modest amount (say 25% of the width), keep a generous overlap with the already-known region so the border constraint stays strong, then repeat. Each tile is a fresh inpaint conditioned on a large known border, so the scene stays coherent across a much larger canvas than a single big outpaint could manage. This is exactly how the "infinite canvas" features in production tools work internally — many small, heavily-overlapped inpaints, not one giant generation. The mask blend equation is unchanged; you are just being disciplined about keeping the known context large relative to the unknown region at every step, because that ratio is what controls how much the generation can drift.

One more subtlety that separates a good inpainting result from a mediocre one: the *soft mask*. A hard binary mask produces a sharp boundary between known and generated content, and even with the per-step blend you can sometimes see a faint seam where the model's generated texture meets the re-injected original. Feathering the mask edge — a few pixels of gradient from 1 to 0 — lets the blend transition smoothly, so the generated and known content interpolate across a band rather than meeting at a hard line. In latent space this matters even more because a single latent pixel covers an 8×8 patch of image pixels (the VAE downsamples by 8×), so a hard latent mask boundary is an abrupt 8-pixel jump in image space. Dilating and feathering the mask in latent space is the difference between an invisible edit and a visible patch.

#### Worked example: removing an object vs adding one

Removing the lamppost is the *easy* direction: the model fills the masked region with plausible background (sidewalk, wall) that is heavily constrained by the surrounding context. It works on the first try at `strength=1.0`. Adding an object is the *hard* direction: "add a red bicycle leaning against the wall" requires the model to invent a coherent object inside the mask, and it often produces a floating or warped bicycle because the mask boundary fights the object's natural extent. The fix in practice is to make the mask slightly *larger* than the intended object so the model has room to render it cleanly, and to phrase the prompt as the *full scene* ("a sidewalk with a red bicycle leaning against the brick wall"), not just the object — the inpainting model conditions on the whole prompt, and a scene-level prompt gives it the context to place the object naturally. Removal is local interpolation; addition is local generation, and generation is always the harder job.

## DDIM inversion: recovering a latent for a real image — and its drift

SDEdit and inpainting are powerful but blunt: neither lets you say "keep this exact composition and swap one noun." To do *that* — surgical, structure-preserving edits — you need to recover the *exact starting latent* that the model's reverse process would turn into your real image, and then perturb the prompt while resampling from that latent. Recovering that latent is **inversion**, and the workhorse is **DDIM inversion**.

Recall from the [DDIM post](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) that DDIM sampling is *deterministic*: given a starting latent $x_T$ and a fixed prompt, the reverse process is an ODE that maps $x_T$ to a unique image $x_0$, with no injected randomness. The DDIM update from $x_t$ to $x_{t-1}$ is

$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \hat{x}_0(x_t, t) + \sqrt{1 - \bar\alpha_{t-1}}\, \epsilon_\theta(x_t, t),$$

where $\hat{x}_0(x_t, t) = \big(x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon_\theta(x_t, t)\big) / \sqrt{\bar\alpha_t}$ is the model's current estimate of the clean image. Because this is a deterministic ODE, you can *run it backwards*: invert the update to step *forward* in noise level, from $x_{t-1}$ to $x_t$, recovering the noise trajectory that leads to your image. That is DDIM inversion.

### The science: the exactness condition, and why CFG breaks it

The inversion step is obtained by rearranging the DDIM update under the assumption that the noise prediction changes slowly between adjacent timesteps — i.e. $\epsilon_\theta(x_t, t) \approx \epsilon_\theta(x_{t-1}, t-1)$. Under that assumption:

$$x_{t} = \sqrt{\bar\alpha_{t}}\, \hat{x}_0(x_{t-1}, t-1) + \sqrt{1 - \bar\alpha_{t}}\, \epsilon_\theta(x_{t-1}, t-1).$$

Start from your real image $x_0$ (its VAE latent) and apply this forward-in-noise step $T$ times; you recover an $x_T$ that, when you run *ordinary* DDIM sampling forward, reconstructs $x_0$. The exactness condition is now explicit: **inversion is accurate only when the local approximation $\epsilon_\theta(x_t, t) \approx \epsilon_\theta(x_{t-1}, t-1)$ holds**, i.e. when the ODE is smooth and the step size is small. With enough steps and an *unguided* model, DDIM inversion reconstructs a real image very faithfully.

Here is the catch, and it is the whole reason null-text inversion exists. To *edit* an image you need the model to actually respond to your prompt, which means you need **classifier-free guidance** at a real scale (CFG ≈ 7.5). But CFG replaces the noise prediction with an *extrapolated* one:

$$\tilde\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big),$$

where $\varnothing$ is the null (empty) embedding and $w$ is the guidance scale. This extrapolated field is *much* less smooth than the plain conditional one — guidance deliberately pushes the prediction *away* from the unconditional direction, amplifying the difference. The local-linearity assumption that made inversion exact now fails badly: the error at each step is amplified by $w$, and it *accumulates* over the trajectory. Run DDIM inversion at $w = 7.5$ and then try to reconstruct, and you get a blurry, color-shifted, structurally-warped version of your image. The composition you were trying to preserve is gone before you have even made an edit.

![A dataflow figure showing DDIM inversion exact at guidance one but drifting at editing-scale guidance, with null-text optimization restoring exact reconstruction so edits can run at full guidance](/imgs/blogs/image-editing-with-diffusion-3.png)

So you are caught in a vice. Invert at $w = 1$ (no guidance): reconstruction is faithful, but the model barely responds to your edited prompt, so edits are weak. Invert at $w = 7.5$: the model responds, but the inversion has already destroyed your image. This is the central tension of inversion-based editing, and the field spent two years fixing it.

### Null-text inversion: optimize the null embedding to make inversion exact

Null-text inversion (Mokady et al., 2022, "Null-text Inversion for Editing Real Images using Guided Diffusion Models") is a clever resolution. The insight: the *un*conditional branch of CFG uses the *null embedding* $\varnothing$ — a fixed, learned "empty prompt" vector. Normally $\varnothing$ is the same constant at every timestep. But there is no law that says it has to be. What if, at each timestep, you *optimize* the null embedding $\varnothing_t$ so that the *guided* DDIM trajectory passes through the latents you recovered with cheap, unguided inversion?

The recipe is two stages:

1. **Pivotal inversion.** Run DDIM inversion *once* at $w = 1$ (unguided) to get a sequence of reference latents $z_T^*, z_{T-1}^*, \dots, z_0^*$. These are a faithful trajectory but for the *unguided* model. Call them the pivot.
2. **Null-text optimization.** Now run *guided* DDIM sampling at $w = 7.5$ from $z_T^*$, but at each timestep $t$, optimize the per-timestep null embedding $\varnothing_t$ (a few gradient steps) to minimize the distance between the guided step's output and the pivot $z_{t-1}^*$:

$$\min_{\varnothing_t} \; \big\| z_{t-1}^* - \text{DDIM-step}\big(z_t, t; \, c, \varnothing_t, w\big) \big\|_2^2.$$

After this optimization, the *guided* model — with these tuned per-timestep null embeddings — reconstructs your real image *exactly*, even at $w = 7.5$. The conditional embedding $c$ (your prompt) is untouched and stays at full strength, so the model still responds to prompt edits. You have decoupled "reconstruct the original faithfully" (absorbed into the optimized null embeddings) from "respond to the edit" (carried by the real conditional embedding). It is a genuinely beautiful trick: you make the *guidance* exact by bending the *unconditional* anchor, not the conditional signal you need for editing.

The cost is real: null-text optimization runs a small inner optimization loop (typically ~10 gradient steps) at *each* of the ~50 diffusion timesteps, so a single image's inversion takes on the order of a minute on an A100 — far more than the seconds a forward pass costs. That is the price of exact, editable inversion at full guidance, and it is exactly why the 2023–2025 line of work (negative-prompt inversion, EDICT, direct inversion, and ultimately feed-forward instruction models) chased ways to avoid the per-image optimization. We will come back to this cost in the case studies.

```python
# Sketch of null-text inversion (the core loop; uses a base SD UNet + DDIM scheduler).
# Stage 1: pivotal DDIM inversion at guidance 1 -> reference latents z_star[t].
# Stage 2: per-timestep optimize the null embedding so guided sampling hits the pivot.

import torch

def null_text_optimization(unet, scheduler, z_star, cond_emb, null_emb_init,
                           guidance=7.5, inner_steps=10, lr=1e-2):
    null_embs = []                       # one optimized null embedding per timestep
    null = null_emb_init.clone().requires_grad_(True)
    z = z_star[-1]                       # start from the inverted x_T (the pivot top)
    timesteps = scheduler.timesteps      # descending, e.g. 50 steps

    for i, t in enumerate(timesteps):
        target = z_star[len(timesteps) - 1 - i]   # the pivot latent at the next level
        opt = torch.optim.Adam([null], lr=lr)
        for _ in range(inner_steps):
            eps_c = unet(z, t, encoder_hidden_states=cond_emb).sample
            eps_u = unet(z, t, encoder_hidden_states=null).sample
            eps = eps_u + guidance * (eps_c - eps_u)     # CFG extrapolation
            z_next = scheduler.step(eps, t, z).prev_sample
            loss = torch.nn.functional.mse_loss(z_next, target)  # match the pivot
            opt.zero_grad(); loss.backward(); opt.step()
        null_embs.append(null.detach().clone())
        with torch.no_grad():                                    # advance with the tuned null
            eps_c = unet(z, t, encoder_hidden_states=cond_emb).sample
            eps_u = unet(z, t, encoder_hidden_states=null).sample
            eps = eps_u + guidance * (eps_c - eps_u)
            z = scheduler.step(eps, t, z).prev_sample
    return null_embs                     # plug these into editing at full guidance
```

This is a sketch, not a turnkey call — the real implementation (in the authors' repo and in `diffusers`' community pipelines) handles the scheduler bookkeeping carefully — but it shows the load-bearing idea: an inner optimization, per timestep, on the null embedding, driving the guided trajectory onto the unguided pivot. Once you have the `null_embs`, you can run editing (e.g. prompt-to-prompt below) at full guidance and the original will be reconstructed exactly except where you intend to change it.

It is worth knowing that null-text optimization is not the only way to defeat inversion drift, because the per-image minute is a real tax and the field attacked it from several angles. **Negative-prompt inversion** (Miyake et al., 2023) observed that the optimized null embedding ends up close to the *conditional* embedding, and showed you can skip the optimization entirely by simply *setting the unconditional embedding equal to the conditional one* during inversion — a closed-form approximation that recovers most of null-text's fidelity for free, turning a minute into seconds. **EDICT** (Wallace et al., 2022) took a structurally different route: it makes the *sampler itself* exactly invertible by maintaining two coupled latent sequences that alternate updates (the same trick affine coupling layers use in normalizing flows), so the round-trip is exact by construction — no approximation, no optimization — at the cost of double the latents and a bit more compute. **Direct inversion** and the various ReNoise/fixed-point schemes push the per-step accuracy further still. The point is not to enumerate them but to see the shape of the problem: inversion drift comes from a local-linearity approximation breaking under guidance, and you can fix it by (a) optimizing an embedding to absorb the error (null-text), (b) approximating the optimum in closed form (negative-prompt), or (c) replacing the sampler with an exactly-invertible one (EDICT). All three buy you the same thing — a faithful, editable latent for a real image at full guidance — and they trade compute against exactness differently. For most editing today you reach for the fast approximations; null-text inversion remains the gold standard when you need the cleanest possible reconstruction and can pay for it.

## Prompt-to-prompt: editing by manipulating cross-attention maps

We now have a faithful, editable latent for our real image. The remaining question is *how* to make a surgical change — swap "cat" for "dog," add "snowy," make the car red — without disturbing everything else. The answer, and the most influential idea in this whole post, is **prompt-to-prompt** (Hertz et al., 2022, "Prompt-to-Prompt Image Editing with Cross-Attention Control"). It edits by reaching into the **cross-attention maps** that bind words to image regions.

Here is the key fact that makes it possible. In a text-conditioned diffusion U-Net, every cross-attention layer computes, for each spatial location (query) and each text token (key), an attention weight — a map saying "how much does this pixel attend to this word." These cross-attention maps are, in effect, a *soft spatial layout* of the prompt: the map for "cat" lights up where the cat is, the map for "windowsill" lights up where the windowsill is. (We cover how text conditioning enters the denoiser in the sibling post on [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning); the cross-attention path inside the U-Net is detailed in the [diffusion U-Net post](/blog/machine-learning/image-generation/the-diffusion-unet).) Crucially, **these maps are established early in the diffusion process and they determine the image's geometry**. If you keep them fixed, the geometry is fixed.

So prompt-to-prompt's recipe: generate (or invert) with the source prompt, *record* its cross-attention maps, and then generate with the edited prompt while *injecting the source maps* for the words that are shared. The change is localized to the word you actually altered. There are three operations:

![A two-column before-and-after figure contrasting word replacement which injects source attention maps to keep geometry against word reweighting which scales one token's attention to strengthen an attribute](/imgs/blogs/image-editing-with-diffusion-6.png)

**1. Word swap (replace).** Change "a photo of a cat" to "a photo of a dog." For every shared word ("a," "photo," "of") you inject the source attention maps, so the layout is pinned. For the changed word ("cat" → "dog"), you let the new word's attention map be used (often the cat's map is injected into the dog token for a fixed number of early steps, then released) so the dog inherits the cat's *location and pose* but is rendered as a dog. The result: same composition, same pose, new subject. This is the structure-preserving subject swap SDEdit could never do.

**2. Adding a word (refine).** Change "a car" to "a red sports car." The new tokens ("red," "sports") get fresh attention maps; the shared tokens keep the source maps. The car stays where it was, but gains the new attributes. This is how you add adjectives without moving the object.

**3. Reweighting (re-weight).** Keep the prompt but scale one token's attention up or down: multiply the attention map for "snowy" by a coefficient $c > 1$ to make the scene snowier, or $c < 1$ to make it less so. This gives you a *continuous* knob on an attribute's strength, like a slider, without changing anything else.

### The science: why swapping the maps localizes the edit

Let me be precise about why this works. A cross-attention layer computes $\text{Attn} = \text{softmax}\!\big(QK^\top / \sqrt{d}\big)V$, where $Q$ comes from the spatial features (one row per latent pixel) and $K, V$ come from the text token embeddings (one row per token). The matrix $M = \text{softmax}(QK^\top/\sqrt{d})$ has shape (pixels × tokens): $M_{ij}$ is how much pixel $i$ attends to token $j$. The output at pixel $i$ is $\sum_j M_{ij} V_j$ — a weighted blend of token *values* according to the attention map.

The geometry of the generated image is determined almost entirely by $M$, not by $V$. The map $M$ says *where* each concept goes; the values $V$ say *what* each concept is. So if you take the edited prompt's values $V'$ (which now encode "dog" instead of "cat") but **inject the source prompt's attention map $M$** (which says where the cat was), the output places "dog-ness" exactly where "cat-ness" used to be. The pixel-to-region assignment is preserved; only the *content* poured into those regions changes. That is the entire mechanism, and it is why prompt-to-prompt edits are so clean: it surgically separates "where" (the map, kept) from "what" (the values, changed).

For reweighting, you literally scale a column of $M$: replace $M_{:,j}$ with $c \cdot M_{:,j}$ for the target token $j$ before the (already-applied) softmax's effect propagates — in practice you scale the pre-softmax logits or the post-softmax weights for that token, increasing how strongly every pixel attends to "snowy." More attention to a token means more of its value bleeds into the image, hence "more snow." The continuity of $c$ gives you the slider.

The catch — and it is important — is that prompt-to-prompt in its original form operates on the model's *own* generated images, where the attention maps come for free during generation. To apply it to a *real* image, you first need inversion to recover the latent and the attention maps, which is exactly why **null-text inversion + prompt-to-prompt** is the canonical real-image editing pipeline: null-text inversion gives you a faithful, full-guidance trajectory for the real image, and prompt-to-prompt then edits it by attention control. The two are designed to compose.

### The code: a prompt-to-prompt attention swap

Prompt-to-prompt is implemented by *swapping the attention layers' behavior* during the forward pass — you register hooks (or replace the attention processor) that, on the edited generation, substitute the stored source maps. Here is a sketch of the core controller.

```python
import torch
import torch.nn.functional as F

class AttnReplace:
    """Inject source cross-attention maps into the edited generation for shared
    tokens, so the edited image keeps the source geometry. Word-swap variant."""
    def __init__(self, source_maps, cross_replace_steps=0.8, num_steps=50):
        self.source_maps = source_maps          # {layer: M_source} recorded earlier
        self.replace_until = int(cross_replace_steps * num_steps)
        self.step = 0

    def __call__(self, attn_probs, layer_name, is_cross):
        # attn_probs: (batch, heads, pixels, tokens) for this attention call.
        if is_cross and self.step < self.replace_until:
            M_src = self.source_maps[layer_name]      # keep WHERE from the source
            return M_src                              # ...let V (WHAT) come from edit
        return attn_probs

    def advance(self):
        self.step += 1

# During the edited sampling loop, the attention processor calls controller(...)
# with the freshly computed probs; for shared tokens it returns the source map.
# For a reweight edit instead of a swap, you scale one token's column:
def reweight(attn_probs, token_index, coef):
    attn_probs[..., token_index] *= coef            # strengthen / weaken an attribute
    return attn_probs / attn_probs.sum(-1, keepdim=True)
```

In `diffusers` you implement this by writing a custom `AttnProcessor` and assigning it with `unet.set_attn_processor(...)`, or by using the community `prompt-to-prompt` pipeline. The `cross_replace_steps` fraction is itself a knob: inject the source maps for the *first* 80% of steps (when geometry is decided) and release them for the last 20% (so the new subject's fine detail can render). Inject for too long and the dog looks like a cat; inject for too short and the geometry drifts. The default of injecting through the early-to-mid steps is where the layout is locked in, consistent with the SDEdit observation that coarse structure is set early.

There is a second family of attention maps the original paper also exploits, and it is worth flagging because it explains *which* edits prompt-to-prompt can and cannot do. Cross-attention maps (pixel × token) carry the *prompt-to-region* binding we have been discussing. But the U-Net also has *self*-attention maps (pixel × pixel) that carry the image's *internal* structure — which pixels belong to the same object, the overall geometry independent of the prompt. Later methods (Plug-and-Play, MasaCtrl) found that injecting the *self*-attention maps and features preserves structure even more robustly than cross-attention alone, and crucially lets you do *non-rigid* edits (change a sitting cat to a standing cat) that cross-attention injection alone fights against — because cross-attention injection pins the *layout* a little too hard. The lesson generalizes: editing is a controlled leak of information from the source generation into the target generation, and *which* internal tensors you copy (cross-attention maps, self-attention maps, residual features) decides exactly which properties survive the edit and which are free to change. Prompt-to-prompt copies cross-attention; the structure-preservation methods that followed copy more, and each choice draws the preserve/change line in a different place.

Let me stress-test the cross-attention approach so its limits are concrete. **What happens with three objects?** Prompt "a cat, a dog, and a bird on a sofa," then try to swap only the cat for a rabbit. The cat's cross-attention map overlaps the dog's and the bird's wherever the animals are close or occluding, so injecting the source maps for "dog" and "bird" while changing "cat" leaks: the rabbit's rendering bleeds into the dog, or the dog faintly turns rabbit-ish. The cleaner the spatial separation, the cleaner the edit; clutter breaks it. **What happens when you reweight too hard?** Scale "snowy" by 5× instead of 2× and the attention saturates — every pixel attends maximally to "snowy," the softmax collapses, and the image turns into an undifferentiated white-out rather than a snowier scene. The reweight knob is only linear over a modest range; past it the softmax nonlinearity dominates. **What happens on a real photo without inversion?** You have no source attention maps to inject, because you never ran a generation that produced them — which is the whole reason prompt-to-prompt on real images *requires* null-text inversion first. Every one of these limits traces back to the same root: cross-attention maps are a clean spatial layout only when concepts are well-separated and the model actually generated the image. When either fails, the surgical edit gets messy.

#### Worked example: the cat-to-dog swap, finally done right

Now we can do the edit we wanted at the very top. Pipeline: null-text-invert the real cat photo at $w = 7.5$ (≈1 minute on an A100), record the source attention maps, then run the edited prompt "a photo of a dog on a windowsill" with `AttnReplace` injecting the source maps for the shared words and the cat's map into the dog token for the first 80% of steps. The result is a *dog in the cat's exact pose, on the cat's exact windowsill, in the cat's exact light* — the windowsill, curtains, and framing are pixel-faithful because their attention maps were pinned and null-text inversion reconstructed them exactly. Compare to SDEdit at `strength=0.6`, which gave a half-cat-half-dog chimera with a drifted windowsill. The attention-control pipeline is strictly better *for this kind of edit* — at the cost of a minute of inversion per image. That cost is the entire economic argument for the feed-forward methods next.

## InstructPix2Pix: train a model to follow edit instructions

Everything so far is a *per-image* procedure: noise-and-denoise (SDEdit), mask-and-blend (inpainting), or invert-and-attention-control (null-text + P2P). The last of these is powerful but slow — a minute of optimization per image — and fiddly. The obvious question: can we *amortize* it? Can we train a single feed-forward model that takes (image, instruction) and outputs the edited image in one shot, the way a text-to-image model takes (prompt) and outputs an image? That is **InstructPix2Pix** (Brooks et al., 2022, "InstructPix2Pix: Learning to Follow Image Editing Instructions").

The problem with training such a model is **data**. To train an instruction-following editor you need a dataset of (input image, instruction, output image) triples — before/after pairs aligned with a natural-language edit. No such dataset exists at scale; nobody has photographed millions of "before and after the edit" pairs. InstructPix2Pix's central contribution is not the model architecture (it is a lightly-modified Stable Diffusion); it is a **synthetic data engine** that *generates* the dataset using the very methods we just covered.

![A dataflow figure showing a language model generating an edit instruction and edited caption, prompt-to-prompt rendering an aligned before-and-after image pair, and a feed-forward editor trained on image plus instruction to output the result](/imgs/blogs/image-editing-with-diffusion-7.png)

### The data engine: GPT-3 + prompt-to-prompt

The pipeline has three stages, and it is a lovely example of bootstrapping a capability out of existing models:

1. **Generate instructions and edited captions with a language model.** Take a real image caption from LAION (e.g. "photograph of a girl riding a horse"). Prompt GPT-3 (fine-tuned on a few hundred human-written examples) to produce an *edit instruction* and the *resulting caption*: instruction "have her ride a dragon," output caption "photograph of a girl riding a dragon." This gives you a pair of captions (before, after) plus the instruction that connects them, at massive scale and near-zero cost.

2. **Turn the caption pair into an image pair with prompt-to-prompt.** Here is the elegant part: you cannot just generate two images from the two captions independently — they would have nothing to do with each other (different girl, different pose). Instead you use **prompt-to-prompt** to generate the before and after images *jointly*, sharing cross-attention so the two images are identical except for the edit. The girl, the pose, the background are the same; only the horse becomes a dragon. This produces a perfectly *aligned* before/after pair, which is exactly what training needs. Prompt-to-prompt is being used here not to edit a user's image but to *manufacture training data*.

3. **Filter for quality.** Many generated pairs are bad — the edit didn't take, or the images drifted too far apart. A CLIP-based filter keeps pairs where the change in image embedding aligns with the change in caption embedding (the "directional CLIP similarity"), discarding misaligned pairs. The result is a curated dataset of ~450,000 (image, instruction, edited image) triples.

With this dataset in hand, training is almost boring: take a pretrained Stable Diffusion, add the *input image* as an extra conditioning (concatenated to the latent input, expanding the first conv layer), keep the text conditioning for the *instruction*, and fine-tune with the standard diffusion denoising loss to predict the edited image. At inference you give it your real photo and an instruction in plain English, and it edits in a single forward pass — no inversion, no per-image optimization, no mask, no attention surgery.

### The science: two guidance scales for two conditions

InstructPix2Pix conditions on *two* things — the input image $c_I$ and the instruction $c_T$ — so it uses **two guidance scales**, one for each, via a generalized classifier-free guidance. With $s_I$ the image guidance and $s_T$ the text guidance:

$$\tilde\epsilon = \epsilon_\theta(z_t, \varnothing, \varnothing) + s_I\big(\epsilon_\theta(z_t, c_I, \varnothing) - \epsilon_\theta(z_t, \varnothing, \varnothing)\big) + s_T\big(\epsilon_\theta(z_t, c_I, c_T) - \epsilon_\theta(z_t, c_I, \varnothing)\big).$$

Read this as two extrapolations stacked. The $s_I$ term pushes toward *fidelity to the input image*; the $s_T$ term pushes toward *following the instruction*. Raising $s_T$ makes the edit stronger but risks ignoring the input; raising $s_I$ keeps more of the original but weakens the edit. This is the *exact same fidelity-versus-change trade-off* as SDEdit's strength knob, now expressed as two independent guidance dials at inference — which is much nicer, because you can keep the image faithful *and* push the edit, somewhat independently, rather than collapsing both into one number. The model was trained with both conditions randomly dropped (so it learned all four combinations of present/absent), which is what makes this dual-guidance possible.

### The code: running InstructPix2Pix

```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

image = load_image("cat_on_windowsill.png").resize((512, 512))

edited = pipe(
    "turn the cat into a dog",          # a natural-language instruction
    image=image,
    num_inference_steps=30,
    image_guidance_scale=1.5,           # s_I: fidelity to the input image
    guidance_scale=7.5,                 # s_T: how hard to follow the instruction
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

edited.save("cat_to_dog_instruct.png")
```

The two scales are the whole tuning surface. The standard advice: **start at `image_guidance_scale=1.5` and `guidance_scale=7.5`**; if the edit is too weak, lower `image_guidance_scale` (toward 1.0) or raise `guidance_scale`; if the original is being destroyed, raise `image_guidance_scale` (toward 1.8–2.2). It is a single forward pass — ~30 steps, a couple of seconds on a consumer GPU — versus a minute of null-text optimization. That is the speed argument that made InstructPix2Pix and its descendants (MagicBrush, which fine-tunes it on human-annotated edits; and the whole 2025 instruction-editing wave) the default.

#### Worked example: where the feed-forward editor wins and loses

Give InstructPix2Pix "turn the cat into a dog" on our photo. It does it in one pass, in seconds, with no inversion — and the result is good for *common* edits like this, because the synthetic dataset is full of subject-swap examples. Now give it "move the cat from the left of the windowsill to the right." It fails: it does not move the cat, or it produces a garbled double-cat. The reason is structural — InstructPix2Pix learned a *distribution* of edits from its synthetic data, and *spatial relocation* edits are rare and hard to manufacture with prompt-to-prompt (which preserves geometry by design, so the training data has very few "move the object" pairs). The model is only as capable as its data engine. This is the central limitation of the whole feed-forward approach in 2022, and closing this gap — instruction edits that involve geometry, counting, and multi-object reasoning — is precisely what the 2025 native-multimodal models in the [next post](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) attack with far larger, better-curated data and stronger backbones.

## Putting the toolkit side by side

We now have five methods that all "edit a real image with a diffusion model," but they make different bets and occupy different cells of the design space. The single most useful thing I can give you is the map of *which method changes what, whether it preserves layout, and whether it needs the expensive inversion step.*

![A matrix figure mapping each editor to its edit type, whether it preserves layout, and whether it needs an inversion step, showing the trade-offs across the five classic methods](/imgs/blogs/image-editing-with-diffusion-5.png)

Here is the same comparison as a table you can act on:

| Method | Edit type it does best | Preserves exact layout? | Needs inversion? | Per-edit cost | Where it breaks |
| --- | --- | --- | --- | --- | --- |
| **SDEdit (img2img)** | Global style / texture; coarse subject change | Partial — fades as strength rises | No (just noise) | ~1× a generation | Large semantic edits destroy identity |
| **Inpainting** | Local fill / removal / outpainting | Yes — outside mask frozen exactly | No (mask + renoise) | ~1× a generation | Global edits; adding coherent objects |
| **DDIM inversion + edit** | Real-image word swap (with care) | Yes if inversion is exact | Yes (cheap but drifts at CFG) | ~1–2× a generation | Reconstruction drifts at editing CFG |
| **Null-text inv. + P2P** | Real-image surgical swap/reweight | Yes — best in class | Yes (slow, ~1 min/image) | ~50× a generation | Structural moves; identity over big swaps |
| **Prompt-to-prompt** | Word swap / add / reweight | Yes — attention pinned | Synthetic images only | ~1–2× a generation | Real images need inversion first |
| **InstructPix2Pix** | Common instruction edits, fast | Mostly — learned prior | No (one forward pass) | ~1× a generation | Spatial moves, counting, rare edits |

A few things jump out of this table that are worth saying out loud. First, **the methods that preserve layout best are the ones that cost the most** (null-text + P2P), and the methods that are cheapest either preserve layout only partially (SDEdit) or only for the masked region (inpainting). That is the fidelity-versus-cost trade-off that runs through the entire field. Second, **only InstructPix2Pix avoids both inversion and a mask** — it pays its cost up front, in dataset generation and training, so inference is one cheap pass. That amortization is why feed-forward editing won. Third, **none of these does large structural edits well** — moving objects, changing counts, re-arranging a scene. Every one of them is built on the assumption that the *layout stays roughly fixed*, which is exactly the assumption that breaks for structural edits, and exactly the gap the modern wave targets.

## How to measure an editing method honestly

Generation has FID; editing has no single clean number, and that is a genuine problem you should understand before trusting any editing benchmark. The difficulty is that a good edit must satisfy *two competing objectives at once*: it must **change** the thing you asked for (edit faithfulness) and **preserve** everything you did not (background/structure consistency). A method can ace either one by sacrificing the other — an edit that changes nothing scores perfectly on consistency, and an edit that regenerates from scratch scores perfectly on faithfulness — so any single metric is gameable. You must report *both* and look at the trade-off.

The standard pair, both built on CLIP, is: **CLIP image similarity** between the input and the output (how much of the original survived — higher means more preserved), and **directional CLIP similarity**, which measures whether the *change* in the image's CLIP embedding aligns with the *change* in the caption's CLIP embedding (did the edit move the image in the direction the instruction asked for). InstructPix2Pix's own evaluation plots one against the other as you sweep the guidance scales, and the right way to read it is as a Pareto frontier: a better method dominates — more edit faithfulness at the *same* consistency, or more consistency at the *same* faithfulness. A single point tells you nothing; the *curve* is the result.

There are sharper, edit-specific benchmarks now. **MagicBrush** provides human-annotated real edit pairs with ground-truth target images, so you can compute pixel-level L1/L2 and perceptual LPIPS to the *intended* result, not just CLIP proxies — a much stronger signal, at the cost of needing humans to make the reference edits. For *masked* edits there is also **masked-region versus unmasked-region** reporting: measure faithfulness only inside the intended edit region and preservation only outside it, which disentangles the two objectives spatially. When you read an editing paper, the questions to ask are exactly these: did they report *both* change and preservation; did they fix the seed and the sampler; did they evaluate on *real* images (the hard case) or only on the model's own generations (the easy case); and did they show the *frontier* or cherry-pick a point. If a method only reports how faithfully it follows the instruction and stays quiet about what it did to the background, be suspicious — it is probably quietly regenerating more than it admits.

#### Worked example: reading a faithfulness–consistency frontier

Concretely, suppose method A reports directional-CLIP 0.22 at image-CLIP 0.78, and method B reports directional-CLIP 0.20 at image-CLIP 0.85, on the same real-image set with the same seed and sampler. Which is better? It depends on your use case, and that is the honest answer. B preserves more of the original (0.85 vs 0.78 image similarity) at slightly weaker edits; A pushes the edit harder at the cost of more collateral change. For a product that retouches user photos where wrecking the background is unacceptable, B wins. For a creative tool where users want bold changes, A wins. The only *wrong* move is to compare A's faithfulness to B's consistency and declare a winner — they are points on two different trade-off curves, and you can only compare methods that report the *whole* curve. This is the same lesson as FID-versus-CLIP for generation: a generative metric in isolation lies, and an editing metric in isolation lies twice, because editing has two objectives instead of one.

## Case studies and real numbers from the literature

Let me ground the trade-offs in numbers from the actual papers, with the usual honesty caveat: editing has no single agreed benchmark the way generation has FID on ImageNet, so these are the metrics each paper reported on its own evaluation set. Where I give a number, it is from the cited paper; where I am approximating, I say so.

**SDEdit (Meng et al., 2021).** Evaluated on stroke-based image synthesis and editing, SDEdit reported a favorable trade-off between *faithfulness* (L2 / LPIPS to the input) and *realism* (a classifier-based realism score) as the strength varied, demonstrating the exact knob this post centers on. The headline qualitative result — that you can turn a crude stroke painting into a photorealistic image by re-entering the reverse SDE partway — is the origin of every "img2img" feature in every diffusion UI today. There is no FID-on-ImageNet number to quote because the task is editing, not class-conditional generation; the contribution is the *method and the trade-off curve*, not a leaderboard score.

**Null-text inversion (Mokady et al., 2022).** The paper's key quantitative claim is reconstruction quality: vanilla DDIM inversion at CFG 7.5 reconstructs a real image with high error, while null-text inversion drives the reconstruction PSNR up dramatically (the paper reports near-lossless reconstruction, with PSNR in the high-20s to low-30s dB range on their test images, versus a visibly degraded vanilla inversion). The cost they report is the per-image optimization: on the order of a minute on a single GPU for a 512×512 image with ~50 DDIM steps and ~10 inner iterations each. That ~1 minute is the number that drove the field toward faster inversions and feed-forward models.

**Prompt-to-prompt (Hertz et al., 2022).** The contribution is qualitative and mechanistic — the demonstration that cross-attention maps *are* the spatial layout and that swapping them localizes edits. The paper's stress is on *editability without spatial drift*: word swaps, attribute addition, and reweighting all preserve the source composition, which earlier methods could not do. It is best understood as the *enabling primitive* — both null-text inversion's editing step and InstructPix2Pix's data engine are built on it.

**InstructPix2Pix (Brooks et al., 2022).** The concrete numbers: a synthetic dataset of ~450,000 (image, instruction, edited image) triples generated with GPT-3 + Stable Diffusion + prompt-to-prompt; the editor itself is a fine-tuned Stable Diffusion (~860M-parameter UNet) that edits in a single forward pass (~30 steps, a couple of seconds on a consumer GPU). The paper measures the trade-off between *image consistency* (CLIP image similarity to the input) and *edit faithfulness* (directional CLIP similarity to the instruction) as the two guidance scales vary — the same fidelity-versus-change curve, now drawn with two independent dials. MagicBrush (Zhang et al., 2023) later fine-tuned InstructPix2Pix on ~10,000 *human-annotated* real edit pairs and reported substantial improvements on held-out human-edit benchmarks, confirming that the synthetic-data ceiling was the limiting factor.

#### Worked example: budgeting an editing pipeline

Suppose you are building an editing feature and must choose. You have a product requirement: "users upload a photo and type an instruction; the edit should land in under 3 seconds and cost under \$0.01 per edit." Walk the options. Null-text + prompt-to-prompt is out immediately — a minute of per-image optimization blows both the latency and the cost budget (at roughly \$2/hour for an A100, a minute is ~\$0.03 just in inversion, before the edit). SDEdit is fast and cheap (~2 seconds, well under \$0.01) but cannot do structure-preserving subject swaps cleanly, so it will frustrate users on exactly the edits they most want. InstructPix2Pix hits the budget — one forward pass, ~2 seconds, well under a cent — and handles the common edits, so it is the right *default*, with the honest caveat that spatial and counting edits will fail. That decision — feed-forward instruction model as the default, with the known failure modes documented — is exactly the choice the 2025 products made, then improved with better backbones and data.

## Where each method breaks (the honest failure catalog)

Every method here is a cost and a compromise. Here is the catalog of where they break, because knowing the failure modes is more useful than knowing the happy path.

**Large structural edits.** None of these methods moves objects, changes their count, or re-arranges a scene well. SDEdit at high strength can re-compose, but then it has discarded the original. Inpainting can add or remove *within a region* but cannot relocate. Prompt-to-prompt *pins* the geometry by design, so it actively *prevents* structural change. InstructPix2Pix's data engine produces almost no structural-edit examples, so the model never learned them. "Move the dog to the left" is the canonical edit that the entire 2022 toolkit fails, and it is the headline capability the 2025 instruction models added.

**Identity preservation over large changes.** When the edit is big — change the breed of a specific dog, change a specific person's expression while keeping their identity — the surviving signal is not enough to pin *identity*. SDEdit at the strength needed to make the change has already noised away the identity-bearing detail. This is exactly the gap that [IP-Adapter and reference conditioning](/blog/machine-learning/image-generation/ip-adapter-and-reference-conditioning) fill: inject an *identity reference* through a separate cross-attention path so identity survives edits that would otherwise wash it out.

**Multi-object scenes.** Cross-attention maps are cleanest when each concept occupies a distinct region. In a cluttered scene with three overlapping objects, the maps overlap, attention "leaks" between objects, and a prompt-to-prompt edit aimed at one object bleeds into its neighbors. This is the same attribute-binding and counting failure we saw with text conditioning generally — editing inherits all of generation's compositionality weaknesses, and then adds the constraint of preserving the rest.

**Inversion drift on hard images.** DDIM inversion's local-linearity assumption fails worst on *high-frequency, high-detail* images — fine textures, text, faces — where the noise prediction changes rapidly between timesteps. Even null-text inversion, which fixes the *guidance* drift, inherits the *step-size* drift; on a face with fine skin texture you will often see a slight identity shift after the round-trip. More DDIM steps reduce this but do not eliminate it; the deeper fixes (EDICT's exactly-invertible coupling, direct inversion) trade compute or memory for exactness.

**The VAE bottleneck.** Everything here happens in the VAE's latent space, and the VAE round-trip itself loses high-frequency detail — small text, fine patterns, sharp edges. No amount of clever editing recovers what the VAE discarded on the way in. For edits where pixel-exact fine detail matters (editing a document, a logo, a face's fine features), the VAE is the floor on how faithful you can be, independent of the editing method.

## When to reach for each (and when not to)

Here is the decision, made decisively. The figure below is the tree; the prose is the reasoning.

![A decision-tree figure that splits first on local versus global edit and then on whether exact layout preservation justifies the cost of inversion, leading to inpainting, prompt-to-prompt, SDEdit, or null-text inversion](/imgs/blogs/image-editing-with-diffusion-8.png)

**Reach for SDEdit (img2img) when** the edit is global and a soft change is acceptable — color grading, style transfer, "make it look like a painting," roughing out a composition from a sketch. It is the cheapest, simplest, training-free option, and for global edits it is often all you need. **Do not** reach for it when you need a structure-preserving subject swap; it will give you a chimera or destroy the layout.

**Reach for inpainting when** the edit is *local and bounded* — remove an object, add an object, fix a region, extend the frame (outpainting). It preserves everything outside the mask *exactly*, which no other method here does. **Do not** reach for it for global edits (relighting, restyling the whole image); the pinned surroundings will fight the prompt and win.

**Reach for InstructPix2Pix (or its descendants) when** you want *fast, instruction-driven* editing of common edits at scale, with no inversion and no mask — the right default for a product. **Do not** rely on it for spatial relocation, counting, or rare/precise edits; it is bounded by its training distribution, and it will silently fail or garble.

**Reach for null-text inversion + prompt-to-prompt when** you need the *highest-fidelity*, structure-preserving, surgical edit of a *real* image and you can afford ~1 minute of inversion per image — a hero shot, a careful retouch, a research comparison. **Do not** use it in a latency- or cost-sensitive product loop; the per-image optimization is too slow. And **do not** expect it to do structural moves; it pins geometry by construction.

**Reach for prompt-to-prompt alone when** you are editing the model's *own* generated images (you have the attention maps for free) or *manufacturing aligned data* à la InstructPix2Pix. For real user images, you must pair it with inversion.

The meta-rule: **match the method to the edit's locality and your fidelity budget.** Local + cheap → inpainting. Global + cheap → SDEdit. Instruction + fast → InstructPix2Pix. Surgical + faithful + slow OK → null-text + P2P. And if you need structural moves or identity preservation across large changes, you have hit the 2022 toolkit's ceiling — that is where the [modern instruction-editing wave](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) and [reference conditioning](/blog/machine-learning/image-generation/ip-adapter-and-reference-conditioning) take over.

## How this connects to the rest of the stack

Editing is not a separate world; it is the same diffusion stack with three new moves. It *starts the reverse process from a latent derived from your image* (via noising or inversion) instead of from fresh noise — that is the SDEdit/inversion move, reaching into the *forward-noising* and *sampler* stages. It *constrains the sampler with a mask* — the inpainting move, reaching into the *sampler* stage with a per-step blend. And it *intervenes in the conditioning and attention* — the prompt-to-prompt move, reaching into the *cross-attention* inside the denoiser. Every classic editor is one or more of those three interventions, layered on the generation machinery you already understand.

This is also why editing rides on every other improvement in the stack. A better VAE (less round-trip loss) gives more faithful inversion and inpainting. A better sampler (DDIM, DPM-Solver) gives smoother, more invertible ODE trajectories. Better guidance (rescaled CFG) reduces inversion drift. A stronger backbone (DiT, MM-DiT) has cleaner attention maps for prompt-to-prompt to manipulate. When SD3 and FLUX moved to MM-DiT and flow matching, the *editing* methods had to be ported — flow-matching inversion is its own subject — but the *ideas* here (re-enter partway, blend by mask, control attention, amortize into a feed-forward model) carried straight over. They are the conceptual primitives of editing, and they are why this pre-2024 toolkit still underpins the field.

Finally, this is the bridge to the modern wave. The 2022 toolkit answered "how do I edit a real image with a *diffusion model that was trained only to generate*?" — and the answer was a stack of clever per-image procedures and one synthetic-data feed-forward model. The 2025 wave answers a different question: "what if the model is *natively trained to edit*, with in-context image inputs and instruction following baked in?" GPT-Image, FLUX-Kontext, and Nano-Banana-style models do not invert or mask or control attention; they take your image and your instruction as *context* and generate the edit directly, with far better structural and multi-object behavior. But they were trained on data and with objectives that are direct descendants of everything here — InstructPix2Pix's instruction-following framing, prompt-to-prompt's aligned-pair generation, and the fidelity-versus-change trade-off that never goes away. Understand this toolkit and the modern wave is a scaling-up, not a reinvention.

## Key takeaways

- **Editing a real image means recovering a latent that reproduces it, then changing as little as possible.** A diffusion model only knows how to start from noise; getting it to start from *your* image is the whole problem.
- **SDEdit's strength knob is a fidelity-versus-change trade-off with no free lunch.** It picks how early you re-enter the reverse process; low strength keeps layout and barely edits, high strength edits hard and discards identity. There is no value that does both.
- **Inpainting preserves the unmasked region *exactly* by re-injecting the renoised original at every step.** It is the only method here that gives pixel-faithful preservation — but only outside the mask, and only for local edits.
- **DDIM inversion is exact only at guidance 1; CFG breaks it.** Guidance's extrapolation violates the local-linearity assumption, and the error accumulates over the trajectory — so naive inversion at editing scale destroys the image before you edit it.
- **Null-text inversion fixes this by optimizing the per-timestep null embedding** to drive the guided trajectory onto an unguided pivot — exact reconstruction at full guidance, at a cost of ~1 minute per image.
- **Prompt-to-prompt edits by swapping cross-attention maps**, which carry the *where* (geometry) separately from the *what* (content). Inject the source maps and the layout is pinned; rescale a token's map and you get an attribute slider.
- **InstructPix2Pix amortizes editing into one feed-forward pass** by manufacturing a synthetic (image, instruction, edited image) dataset with GPT-3 + prompt-to-prompt — fast and cheap at inference, but bounded by its training distribution.
- **Match the method to the edit's locality and your fidelity budget:** local+cheap → inpainting; global+cheap → SDEdit; instruction+fast → InstructPix2Pix; surgical+faithful → null-text + P2P.
- **The whole 2022 toolkit fails on structural moves, identity over large changes, and cluttered multi-object scenes** — exactly the gaps the 2025 native-multimodal instruction-editing models were built to close.

## Further reading

- **Meng et al., 2021** — "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations." The noise-then-denoise editing method and the strength trade-off.
- **Lugmayr et al., 2022** — "RePaint: Inpainting using Denoising Diffusion Probabilistic Models." The per-step known/generated blend and resampling for seamless inpainting.
- **Mokady et al., 2022** — "Null-text Inversion for Editing Real Images using Guided Diffusion Models." Optimizing the null embedding to make DDIM inversion exact at full CFG.
- **Hertz et al., 2022** — "Prompt-to-Prompt Image Editing with Cross-Attention Control." The cross-attention-map manipulation that localizes edits.
- **Brooks et al., 2022** — "InstructPix2Pix: Learning to Follow Image Editing Instructions." The synthetic-data engine and the feed-forward instruction editor.
- **Zhang et al., 2023** — "MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing." Fine-tuning InstructPix2Pix on real human edits and the gains it buys.
- 🤗 [`diffusers` documentation](https://huggingface.co/docs/diffusers) — the `StableDiffusionImg2ImgPipeline`, `StableDiffusionInpaintPipeline`, and `StableDiffusionInstructPix2PixPipeline` used throughout.
- Within this series: [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) (the invertible ODE inversion rests on), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) (why inversion drifts), [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet) (where cross-attention lives), [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) (the foundation), and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
