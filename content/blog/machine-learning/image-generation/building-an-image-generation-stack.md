---
title: "Building an Image Generation Stack: The End-to-End Playbook"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Assemble everything the series taught into real shipping decisions: a base-model decision tree, the inference pipeline and its knobs, when to reach for ControlNet versus a LoRA, a full LoRA fine-tune workflow, the distillation-plus-quantization-plus-caching speed stack, a serving snippet, a cost-per-image model, and the evaluation and safety gate every product needs before it ships."
tags:
  [
    "image-generation",
    "diffusion-models",
    "stable-diffusion",
    "flux",
    "lora",
    "model-serving",
    "inference-optimization",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/building-an-image-generation-stack-1.png"
---

You have a brief and a deadline. The brief says: a product feature that turns a customer's text prompt into a 1024×1024 product photo, on-brand, in under two seconds, on hardware you can actually afford, with a content-safety gate that keeps the company out of the news. The deadline is two weeks. You open a terminal, type `pip install diffusers`, and then freeze — because the next decision is the one that determines everything downstream, and there are at least six base models, four samplers, three ways to inject your brand style, a half-dozen distillation tricks, two quantization formats, and a safety stack to wire up, and every choice trades against every other one. Pick FLUX.1-dev for quality and you blow the latency budget and inherit a non-commercial license. Pick SDXL-Turbo for speed and you give up some fidelity. Pick a LoRA for the brand style and you have to curate data and run a training loop. This post is the map through that maze.

Everything in this series was a single layer of that map. We started by asking [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) — modeling a distribution over millions of correlated pixels on a thin manifold — and built the [diffusion engine from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles): noise a clean image, train a network to undo one step of that noise, and sample by running the network in a loop. Then we made it efficient with [latent diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), swapped the U-Net for a [transformer backbone](/blog/machine-learning/image-generation/diffusion-transformers-dit), and adopted [flow matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) for the [modern SD3/FLUX recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe). We learned to steer it with [guidance](/blog/machine-learning/image-generation/classifier-free-guidance), [structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control), [references](/blog/machine-learning/image-generation/ip-adapter-and-reference-conditioning), and [personalization](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora); to speed it up with [consistency models](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation), [distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation), and [quantization and caching](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference); and to [evaluate it honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) and [ship it safely](/blog/machine-learning/image-generation/safety-watermarking-and-provenance). This post is where all of that becomes one decision tree and one running system.

![A vertical stack diagram of the six layers of an image generation product, from base-model choice at the top through the inference pipeline, control, fine-tuning, speed, and the evaluation and safety gate at the bottom.](/imgs/blogs/building-an-image-generation-stack-1.png)

Figure 1 is the whole stack on one page. Six layers, top to bottom: the **base model** you build on, the **inference pipeline** that turns a prompt into pixels, the **control and customization** layer that makes it do what you want, the **fine-tuning workflow** that teaches it your concept, the **speed and serving** layer that hits your latency and cost targets, and the **evaluation and safety gate** that decides whether it ships. By the end of this post you'll be able to walk down that stack and make a defensible call at every layer: which checkpoint, which sampler and how many steps, which guidance scale, when to fine-tune versus prompt, which distillation and quantization to stack, how to serve it behind a FastAPI endpoint, what it costs per image, and what to measure before you let a single image reach a user. We'll keep tying back to the series spine — the **generative trilemma** (sample quality × diversity × sampling speed) and the **diffusion stack** (data → VAE latent → forward noising → denoiser → sampler → guidance → image) — because every decision in this post is a move along one of those axes.

A word on how to read this. The series taught you the *why* behind each technique; this post is deliberately the *decision layer* on top. Where a topic earned a full derivation in its own post, I'll link out rather than re-derive — the point here is the assembly, not the parts. Treat it as the playbook you keep open in a second tab while you build.

## 1. Choosing a base model: the decision that fixes everything

The first decision is the heaviest, because every later choice is conditioned on it. A LoRA you train, a ControlNet you attach, a quantization recipe you apply, a sampler you pick — all of them target a *specific* base model's architecture and latent space. Choose wrong and you redo the whole stack. So slow down and make this one well.

There are five base models worth your attention in 2026, and they span the full quality/speed/VRAM/license/ecosystem space. **SD1.5** (0.9B U-Net, 512×512 native, fully open) is old and its raw quality is dated, but it has the largest fine-tune and ControlNet ecosystem on earth and runs on a potato — it is still the right answer when you need a thousand community LoRAs or a 4 GB GPU. **SDXL** (2.6B U-Net, 1024×1024 native, OpenRAIL license) is the workhorse: strong quality, a mature ecosystem, a permissive-enough license for most products, and a Turbo/Lightning distillation path for speed. **SD3.5-Large** (8B MM-DiT, flow matching, community license) pushes quality and text rendering further at a heavier VRAM cost. **FLUX.1-dev** (12B DiT, flow matching, *non-commercial* license) is the current quality and prompt-following leader — and that license clause is a real constraint, not a footnote, because shipping it in a paid product is a contract violation unless you license it commercially. **SANA** (0.6B DiT with a deep-compression autoencoder and linear attention) is the speed-and-resolution specialist: it generates 4K images fast on modest hardware because its [deep-compression AE](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) shrinks the latent 32× instead of SD's 8×.

The decision is not "which is best" — there is no best — it's "which clears my binding constraint." And the binding constraint is almost always one of two things first: how much VRAM you have, and whether you can use the output commercially.

![A decision tree for choosing a base model that branches first on available GPU memory, then on commercial-license need versus maximum quality, landing on SD1.5, SANA, SDXL or SD3.5, and FLUX.](/imgs/blogs/building-an-image-generation-stack-2.png)

Figure 2 is the decision tree I actually use. Start at the VRAM gate, because it is a hard physical wall: a 12B-parameter FLUX transformer in bf16 is 24 GB of weights *before* activations and the two text encoders, so on an 8–12 GB GPU it is simply not an option without aggressive offload that kills your latency. If you're VRAM-tight, the branch splits between SD1.5 (when you need the ecosystem and 512×512 is fine) and SANA (when you need speed and high resolution on a small card). If you're VRAM-roomy — a 24 GB RTX 4090 or better — the next split is the license question. If you need clean commercial use, SDXL or SD3.5 is your lane. If maximum quality is the goal and a non-commercial license is acceptable (research, internal tooling, a personal project), FLUX.1-dev is the prompt-following champion. The tree has a real middle level on purpose: the constraint comes *before* the model, because the constraint is what you can't change and the model is what you can.

The science under this decision is a simple resource model. A model's VRAM footprint at inference is roughly

$$
\text{VRAM} \approx \underbrace{P \cdot b_w}_{\text{weights}} \;+\; \underbrace{P_\text{enc} \cdot b_w}_{\text{text encoders}} \;+\; \underbrace{A(\text{res}, \text{batch})}_{\text{activations}} \;+\; \underbrace{V(\text{res})}_{\text{VAE peak}},
$$

where $P$ is the denoiser parameter count, $b_w$ is bytes per weight (2 for fp16/bf16, 1 for fp8, 0.5 for int4), $P_\text{enc}$ is the text-encoder parameters (T5-XXL alone is 4.7B ≈ 9.5 GB in fp16), $A$ is the per-pass activation memory that grows with resolution and batch, and $V$ is the VAE decode peak, which spikes at high resolution. The headline term is $P \cdot b_w$: FLUX's $12\text{B} \times 2 = 24$ GB versus SDXL's $2.6\text{B} \times 2 = 5.2$ GB is a 4.6× difference in the dominant term, and that is why the VRAM gate is the first branch. Quantization changes $b_w$, which is why a 4-bit FLUX (covered in [quantization and caching](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference)) suddenly fits a card that fp16 FLUX never could — the same model, a different point on the resource axis.

#### Worked example: choosing a base for a 24 GB card and a commercial product

You have an RTX 4090 (24 GB) and you're building a paid feature. Walk the tree. VRAM gate: 24 GB is roomy, so you're on the right branch. License gate: it's a paid product, so non-commercial FLUX.1-dev is *out* (you'd be violating the license) unless you buy a commercial license from Black Forest Labs. That leaves SDXL or SD3.5-Large. SDXL fits comfortably (≈5 GB weights, ≈8 GB working set), has the deepest ControlNet and LoRA ecosystem, and has Turbo/Lightning distillations ready for your latency budget. SD3.5-Large gives better text rendering and prompt-following but at ≈18 GB working set and a slower base. The call: **SDXL as the default**, reaching for SD3.5 only if your prompts are text-heavy (posters, packaging with legible words) where SDXL's text rendering visibly fails. That decision took thirty seconds because the constraints made it for you — which is the entire point of leading with constraints.

Here's the matrix I keep pinned, because "which is best" is the wrong question and "which clears my constraint" needs the numbers side by side.

![A five by four matrix comparing SD1.5, SDXL, SD3.5-Large, FLUX.1-dev, and SANA across quality, speed and VRAM, license, and a recommended pick-when column.](/imgs/blogs/building-an-image-generation-stack-6.png)

Figure 6 lays the five contenders against the four axes that actually decide. Read it by column when one axis is binding (license-bound? FLUX's non-commercial cell is a red stop), and by row when you've narrowed to two and need to break the tie. The "pick when" column is the compressed wisdom: SD1.5 for the ecosystem, SDXL as the safe default, SD3.5 when quality and a workable license both matter, FLUX for maximum fidelity where the license allows, SANA for 4K and edge.

| Model | Params | Native res | License | Best for |
| --- | --- | --- | --- | --- |
| SD1.5 | 0.9B U-Net | 512² | Open (CreativeML) | Ecosystem, tiny GPUs, ControlNet zoo |
| SDXL | 2.6B U-Net | 1024² | OpenRAIL-M | The safe commercial default |
| SD3.5-Large | 8B MM-DiT | 1024² | Stability community | Text rendering, prompt-following |
| FLUX.1-dev | 12B DiT | 1024² | **Non-commercial** | Max quality (research/internal) |
| FLUX.1-schnell | 12B DiT | 1024² | Apache-2.0 | Commercial + fast (distilled) |
| SANA | 0.6B DiT | up to 4K | Open | Speed at high resolution, edge |

One row deserves a flag because it dodges the FLUX license trap: **FLUX.1-schnell** is the timestep-distilled sibling of FLUX.1-dev, generates in ~4 steps, *and* ships under Apache-2.0 — so when you want FLUX-family quality, commercial rights, and speed all at once, schnell is frequently the answer the other models can't give. We'll lean on it again in the serving section.

There's a fifth selection axis that doesn't fit in a single matrix cell but quietly decides more projects than quality does: **the ecosystem**. A base model is not just weights — it's the LoRAs, ControlNets, IP-Adapters, fine-tuning scripts, quantized checkpoints, and community knowledge built around it. SD1.5 and SDXL have enormous ecosystems: thousands of community LoRAs, every ControlNet conditioning type, mature training tooling, and a decade of Stack Overflow answers for every error. FLUX and SD3.5, being newer and larger, have thinner-but-growing ecosystems — the ControlNets and IP-Adapters exist but fewer of them, and the community LoRA library is a fraction of SDXL's. This matters concretely: if your product depends on attaching a depth ControlNet *and* a face IP-Adapter *and* three community style LoRAs, SDXL's ecosystem makes that a config change while FLUX might require you to train adapters yourself. The rule that falls out: **for a feature that leans hard on the control-and-customization layer, the ecosystem can outweigh a point of base quality** — a slightly-worse base with the exact adapter you need beats a better base you have to extend yourself. This is why SDXL remains the default for products that compose many control signals, even as FLUX wins the raw-quality benchmarks.

## 2. The inference pipeline and every knob on it

Once you've picked a base, generation is a fixed sequence of stages, and your job in production is to know exactly what each stage costs and which knob it exposes. Get this mental map right and debugging "why is my image mush / slow / saturated" becomes a matter of asking which stage is misconfigured.

![A vertical stack of the inference pipeline showing text encoders running once, the denoiser running N times under a sampler and classifier-free guidance, then a single VAE decode to the final image.](/imgs/blogs/building-an-image-generation-stack-3.png)

Figure 3 is the pipeline. The **text encoder(s)** run *once* at the start: CLIP (and, for SD3/FLUX, a large T5 or LLM) turn your prompt into conditioning embeddings. They cost memory while resident but barely touch latency because they fire a single time — a fact we'll exploit when we offload them. The **denoiser** (U-Net or DiT) runs *once per sampling step*, so it is the entire latency bill: $N$ steps means $N$ forward passes through the big network. The **sampler** (the scheduler) is the cheap arithmetic that turns the network's noise prediction into the next, slightly-cleaner latent — but the *choice* of sampler decides how few steps you can get away with. **Classifier-free guidance** runs the denoiser a *second* time per step (once conditional, once unconditional) and combines them, which silently doubles the bill unless you've distilled it away. Finally the **VAE decode** runs once at the end to turn the final latent into pixels, cheap in FLOPs but a memory spike at high resolution. Six stages, two of which (denoiser, guidance) dominate cost.

Now the knobs, because this is where products are won and lost.

**Sampler and steps.** The sampler is the ODE/SDE solver that integrates the reverse process; the [samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive) derived why higher-order solvers converge in fewer steps. The practical upshot: a second-order multistep solver like `DPMSolverMultistepScheduler` (DPM++ 2M) or `UniPCMultistepScheduler` hits near-converged quality in 20–30 steps where the old `DDPMScheduler` needed hundreds. For a distilled model you go further — 4–8 steps with `LCMScheduler` or the model's native flow-match scheduler. The rule: **start at DPM++ 2M with 25–30 steps for a quality base, or LCM/flow-match with 4–8 steps for a distilled model, and only move off that if you can see a difference.** More steps past convergence is wasted latency, not free quality.

**CFG scale.** Classifier-free guidance extrapolates the conditional prediction away from the unconditional one: $\hat\epsilon = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$, with guidance scale $w$. The [CFG post](/blog/machine-learning/image-generation/classifier-free-guidance) derived why $w$ trades diversity for prompt adherence and why too-high $w$ over-saturates and burns out highlights. Production defaults: **$w \approx 7.0$–$7.5$ for SD1.5/SDXL, $w \approx 3.5$–$5.0$ for SD3/FLUX-dev** (flow-matching models want lower guidance), and **$w = 0$ for distilled Turbo/schnell models** that baked guidance into the weights — passing a CFG scale to a guidance-distilled model is a classic mistake that double-applies it and wrecks the image.

**Resolution.** Generate at the model's *native* resolution (512² for SD1.5, 1024² for SDXL/SD3/FLUX) and upscale afterward if you need more. Generating off-native — 768² on an SD1.5 trained at 512² — produces the infamous duplicated-heads and repeated-objects artifacts, because the model never saw that spatial extent during training. If you need 2K or 4K, generate native and use a tile-based upscaler or SANA, which was *trained* at high resolution.

**Seed.** The seed fixes the initial noise latent, so a fixed seed gives a reproducible image — essential for A/B testing prompts and for debugging ("same seed, different sampler" isolates the sampler's effect). Vary the seed to explore diversity; fix it to compare anything else.

Here is the whole pipeline in real `diffusers`, with the knobs annotated — this is the snippet I'd hand a new engineer on day one.

```python
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,        # half precision: 2x less VRAM, negligible quality loss
    variant="fp16",
    use_safetensors=True,
).to("cuda")

# Sampler swap: DPM++ 2M is the strong 25-30 step default.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
)

# Efficiency flags that cost nothing in quality (covered in the quantization/caching post):
pipe.enable_vae_slicing()             # decode the VAE in chunks -> lower peak memory
pipe.enable_vae_tiling()              # tile the VAE -> enables high-res decode without OOM
# pipe.enable_model_cpu_offload()     # turn on only if VRAM-bound; trades latency for memory

generator = torch.Generator("cuda").manual_seed(42)   # fixed seed = reproducible

image = pipe(
    prompt="a studio product photo of a ceramic coffee mug on marble, soft daylight",
    negative_prompt="blurry, low quality, watermark, text",
    num_inference_steps=28,           # the steps knob
    guidance_scale=7.0,               # the CFG knob (7.0 for SDXL)
    height=1024, width=1024,          # native resolution
    generator=generator,
).images[0]

image.save("mug.png")
```

That is a complete, production-shaped pipeline. The five knobs — scheduler, steps, guidance, resolution, seed — are right there, and 90% of "the output looks wrong" tickets are one of them set badly. Negative prompts deserve a note: they steer *away* from concepts via the unconditional branch of CFG, so a good negative prompt (`"blurry, low quality, extra fingers, watermark"`) is a cheap quality lever on SD-family models, though SD3/FLUX rely on it less because their text encoders are stronger.

There's a sixth knob people forget exists: the **text encoder offload**, which is a memory knob, not a quality one. Because the encoders run once and then sit idle for the entire denoising loop, you can push them to CPU (or unload them) after the encode and reclaim several gigabytes of VRAM for the denoiser and the VAE. T5-XXL alone is ~9.5 GB in fp16; on a memory-bound card, offloading it after the prompt is encoded is the difference between fitting and an out-of-memory crash. The cost is a small latency hit on the *next* request if you have to reload, which is why the right pattern for a server is to keep the encoders resident if you have the VRAM and offload only when you don't. This is the practical face of the resource model from section 1: the encoders contribute to the $P_\text{enc} \cdot b_w$ term, and that term is *removable from the peak* because it doesn't overlap in time with the denoiser's peak.

One more thing the pipeline map clarifies: **why CFG silently doubles your cost and how distillation reclaims it.** Each guided step runs the denoiser twice — once with your prompt's conditioning, once with an empty (unconditional) prompt — so a "30-step" generation with CFG is really 60 denoiser invocations. `diffusers` batches the two passes into one kernel launch (a batch of 2), so they share overhead, but they still do 2× the matmul work. This is the most under-appreciated factor in the latency budget: when you compare a 30-step guided base against a 4-step distilled model, the real speedup isn't 30/4 ≈ 7.5×, it's closer to 60/4 = 15×, because the distilled model also killed the CFG 2×. Keep that in mind when a vendor quotes "8× faster" — the honest comparison accounts for whether the baseline was paying the guidance tax.

#### Worked example: diagnosing washed-out, over-contrasty output

A teammate reports their SDXL images look "fried" — blown highlights, neon saturation, plastic skin. Walk the knobs. Resolution is native (1024², fine). Sampler is DPM++ 2M (fine). Steps are 30 (fine). Seed varies (fine). Guidance scale is **14**. There's your bug: $w = 14$ is roughly double the sane SDXL range, and CFG over-saturation is exactly the "fried" look — high guidance pushes the prediction so far from the unconditional manifold that it pins channels to their extremes. Drop to $w = 7.0$ and the fry vanishes. If they wanted *some* of the punchiness, the right fix is CFG rescaling (the `guidance_rescale` argument, ≈0.7) which preserves the extra adherence while pulling the statistics back toward normal — but the first move is always to put $w$ back in range. This is the kind of five-minute fix the pipeline map makes obvious: the symptom (saturation) points at one stage (guidance), and you don't go spelunking in the VAE.

## 3. Control and customization: when to reach for what

A base model plus a good prompt gets you a generic-but-good image. Products need *specific* images: this pose, this product, this brand style, this person's face. There are six tools for that, and the costly mistake is reaching for the powerful one (a fine-tune) when the cheap one (a prompt) would do, or vice versa. The decision is governed by one question: **what, exactly, are you trying to control, and do you need to teach the model something new or just steer what it already knows?**

![A decision tree for choosing a control method that splits zero-training inference-time tools from methods that require training, leading to prompt, ControlNet, IP-Adapter, LoRA, and DreamBooth.](/imgs/blogs/building-an-image-generation-stack-4.png)

Figure 4 splits the six tools by that question. The top branch is **zero-training, inference-time** control — nothing to train, you just configure it per request. Within it: a **prompt plus CFG** controls *content and style* (the model already knows "watercolor" and "golden hour"); a **ControlNet** controls *spatial layout* — pose, depth, edges, segmentation — by conditioning on a control image; an **IP-Adapter** controls *appearance from a reference image*, transferring the look or identity of a picture you hand it. The bottom branch is **training required**, for teaching a genuinely new concept the model has never seen: a **LoRA** learns a *style or object* from 10–30 images (your brand's aesthetic, a specific product); **DreamBooth** binds an *exact subject* — a particular person or item — from as few as 3–5 images. The order of preference is top-to-bottom and left-to-right: try the cheapest thing that could possibly work first.

Let me make the boundaries concrete, because they blur in practice.

**Prompt first, always.** If the thing you want is a concept the model was trained on — a season, a lighting style, a camera lens, a famous art movement — say it in the prompt and you're done. No training, no adapter, no latency cost. The failure mode is *binding*: prompts struggle with precise spatial layout ("the cat on the *left*"), exact counts ("*three* apples"), and consistent identity across images. When you hit a binding wall, climb to the next tool.

**ControlNet for layout.** When you need the *structure* fixed — replicate a pose from a reference photo, match a depth map, follow a sketch's edges — that's [ControlNet](/blog/machine-learning/image-generation/controlnet-and-structural-control). You preprocess your control image (OpenPose, depth, Canny) and the ControlNet injects that spatial signal into the denoiser via its zero-initialized convolutions. The zero-convolution trick is worth a sentence because it's the elegant part: ControlNet clones the encoder of the base U-Net, connects it to the frozen base through convolution layers initialized to *zero*, and trains only the clone. Because the connections start at zero, at the first training step the ControlNet contributes *nothing* — the base model's behavior is exactly preserved — and the network gradually learns to inject the control signal without ever catastrophically disrupting the pretrained weights. That's why ControlNet trains stably and cheaply on a single conditioning type without wrecking the base. It's inference-time, it composes with prompts and LoRAs, and you can stack multiple ControlNets (pose + depth) for tight control, each with its own `conditioning_scale`. Reach for it when "describe the layout in words" keeps failing.

**IP-Adapter for references.** When the thing you want lives in an *image*, not in words — "make it look like *this* mood board," "keep *this* face" — that's an [IP-Adapter](/blog/machine-learning/image-generation/ip-adapter-and-reference-conditioning). It encodes the reference image and injects it through decoupled cross-attention, so the model conditions on the picture as if it were part of the prompt. Identity-preservation variants (InstantID/PuLID-style) specialize this for faces. Like ControlNet it's training-free and composes with everything.

**LoRA when you must teach a style or object.** When the concept genuinely isn't in the model — your company's specific illustration style, a product that didn't exist at training time — and you'll generate it *repeatedly*, train a [LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora). A LoRA learns low-rank weight updates from 10–30 images, ships as a 50–200 MB file, and loads on top of the base at inference. It's the right tool for *reusable* customization. The full workflow is section 4.

**DreamBooth for an exact subject.** When you need *this specific entity* — a particular person, a particular dog, a particular hero product — rendered faithfully in new scenes, DreamBooth fine-tunes with a rare-token binding from 3–5 images. It's heavier than a LoRA (often you train a LoRA-DreamBooth hybrid to keep it light) and prone to overfitting, but it's the tool when identity must be exact.

**Instruction editing when the input is an existing image plus a command.** A newer branch I'll flag separately: if the task is "take *this* image and *do this to it*" — "remove the background," "make it winter," "add a hat" — the 2025 wave of [instruction and in-context editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) (FLUX-Kontext, native-multimodal models) does it conversationally, no per-edit training. Reach for it when your product is an *editor*, not a generator.

The decision in one sentence: **prompt for what the model knows, ControlNet for layout, IP-Adapter for a reference look, LoRA for a reusable learned concept, DreamBooth for an exact subject, instruction editing for image-in image-out.** Here's a ControlNet + LoRA + IP-Adapter stack composed in `diffusers`, because in production you often need several at once:

```python
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector

controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, torch_dtype=torch.float16, variant="fp16",
).to("cuda")

# Layer a brand-style LoRA on top of the base weights.
pipe.load_lora_weights("./brand-style-lora", weight_name="brand.safetensors")
pipe.set_adapters(["default"], adapter_weights=[0.8])   # LoRA strength knob

# Add an IP-Adapter for a reference look.
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                     weight_name="ip-adapter_sdxl.safetensors")
pipe.set_ip_adapter_scale(0.5)                          # reference strength knob

pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")(
    load_image("reference_pose.png")
)
ref = load_image("mood_board.png")

image = pipe(
    prompt="a model wearing a brand jacket, editorial photography",
    image=pose,                       # ControlNet: the spatial control
    ip_adapter_image=ref,             # IP-Adapter: the reference look
    controlnet_conditioning_scale=0.7,
    num_inference_steps=30, guidance_scale=6.0,
).images[0]
```

Three control signals — a learned style (LoRA at 0.8), a spatial pose (ControlNet at 0.7), a reference look (IP-Adapter at 0.5) — composed on one base, each with its own strength knob. That composition is the real power of the control layer: the tools stack, and the strength scalars let you dial the blend. The art is in those scalars; start each near the values above and adjust by eye.

## 4. A real fine-tuning workflow: LoRA from data to merge

When you've decided a LoRA is the right tool — a reusable learned concept, your brand style, a product family — the workflow is a closed loop with an *evaluation gate*, not a one-shot training run you cross your fingers on. The single most common way fine-tunes fail is skipping the gate: people train, eyeball one cherry-picked sample, declare victory, and ship a LoRA that overfit to the training set and produces the same five compositions forever.

![A dataflow graph of the LoRA fine-tune loop showing data curation feeding training, an evaluation gate that either passes to merge-and-share or fails back to a data-and-rank fix before shipping.](/imgs/blogs/building-an-image-generation-stack-5.png)

Figure 5 is the loop. **Curate** 15–30 images, captioned, diverse in pose and background so the model learns the *concept* and not the *backgrounds*. **Train** a LoRA with `peft`/`diffusers` — rank 16 is a strong default, 1,000–2,000 steps for a style. **Evaluate** on *held-out* prompts you didn't train on, with a CLIP-score check and a human spot-check. If it under- or over-fit, **fix** the data or the rank and loop. Only when it passes the gate do you **merge or keep the adapter** and **ship** the ~50–200 MB file. The gate is the load-bearing node.

The science of *why* a LoRA works in so few parameters: a full fine-tune updates a weight matrix $W \in \mathbb{R}^{d \times k}$ with a dense update $\Delta W$, which is $d \times k$ parameters. LoRA constrains the update to be *low-rank*, $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ for a small rank $r \ll \min(d,k)$, so you train only $r(d+k)$ parameters instead of $dk$. For a $1024 \times 1024$ attention projection at rank 16, that's $16 \times 2048 = 32{,}768$ parameters instead of $1{,}048{,}576$ — a **32× reduction** — and the forward pass becomes $W x + \frac{\alpha}{r} B(Ax)$, where $\alpha$ is a scaling that lets you dial the adapter's strength at inference (the `adapter_weights` knob from section 3). The bet LoRA makes is that the *adaptation* a new concept requires is intrinsically low-rank even though the base weights are full-rank — empirically true for style and subject adaptation, which is why a 100 MB file can re-skin a 5 GB model. (This is the same low-rank-structure insight that powers SVDQuant's outlier branch in the [quantization post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) — low-rank corrections punch above their parameter weight.)

Here's the training launch. First the data: a folder of images and a metadata file mapping each to a caption with your trigger token.

```python
# metadata.jsonl — one line per training image
# {"file_name": "01.png", "text": "a photo in sks-brand style, a red sneaker on white"}
# {"file_name": "02.png", "text": "a photo in sks-brand style, a blue mug on a desk"}
# ... 15-30 lines, diverse subjects, consistent style token "sks-brand"
```

Then the `accelerate launch` of the `diffusers` SDXL LoRA training script — the real flags, not pseudocode:

```bash
accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --dataset_name="./brand_data" \
  --instance_prompt="a photo in sks-brand style" \
  --resolution=1024 \
  --rank=16 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --mixed_precision="fp16" --gradient_checkpointing \
  --use_8bit_adam \
  --checkpointing_steps=500 \
  --seed=42 \
  --output_dir="./brand-style-lora"
```

Three flags carry most of the practical weight. `--rank=16` is the capacity dial: too low (4) underfits and won't capture the style; too high (128) overfits and memorizes the training images. `--max_train_steps=1500` with `--checkpointing_steps=500` lets you evaluate intermediate checkpoints and pick the one *before* overfitting — overfit LoRAs lose prompt-following ("language drift") and reproduce training backgrounds. `--gradient_checkpointing` and `--use_8bit_adam` are the memory levers that let this run on a 24 GB card instead of needing an A100. On a 4090 this trains in roughly 20–40 minutes.

Now the gate — this is the step people skip and the reason their LoRAs are bad. Evaluate on prompts you did **not** train on, varying the *non-style* content, and check both adherence to your style and *retention* of the base model's competence:

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
pipe.load_lora_weights("./brand-style-lora/checkpoint-1000")

# Held-out prompts: subjects NOT in the training set, to test generalization.
held_out = [
    "a photo in sks-brand style, a bicycle in a park",     # new subject
    "a photo in sks-brand style, a cup of coffee at dawn",  # new subject
    "a portrait of a woman, studio lighting",               # NO trigger: tests language drift
]
for i, p in enumerate(held_out):
    for w in [0.6, 0.8, 1.0]:                # sweep adapter strength
        pipe.set_adapters(["default"], adapter_weights=[w])
        img = pipe(p, num_inference_steps=28, guidance_scale=7.0,
                   generator=torch.Generator("cuda").manual_seed(0)).images[0]
        img.save(f"eval_{i}_w{w}.png")
```

What you're looking for in those outputs: (1) the style transfers to *new* subjects, not just the training ones — if the bicycle and coffee come out in your brand look, the LoRA learned the concept; (2) the no-trigger portrait still looks like a normal SDXL portrait — if it's been dragged into the brand style anyway, the LoRA *overfit and leaked*, contaminating the base behavior, and you should lower the rank or train fewer steps; (3) the strength sweep shows a usable range — if only $w = 1.0$ works and 0.6 does nothing, the effect is too weak; if even 0.6 is overpowering, it's too strong. A CLIP-score on the held-out style-vs-prompt alignment gives you a number to track across checkpoints, but the human spot-check catches the overfit-and-leak failure that CLIP-score misses. When it passes, you can keep the adapter as a separate file (flexible, swappable) or `pipe.fuse_lora()` to merge it into the base weights (one fewer file, slightly faster inference, but no longer adjustable). Keep it separate unless you have a strong reason to merge.

#### Worked example: the LoRA that ruined every face

You train a brand-style LoRA on 20 product shots, rank 32, 3,000 steps. It nails the products. Then a tester reports that *portraits* now look subtly wrong — skin tones shifted, a uniform "look" creeping into faces that have nothing to do with products. That's the overfit-and-leak failure: at rank 32 and 3,000 steps the LoRA learned not just "brand product style" but a global recoloring it applies to everything, and the no-trigger portrait in your eval set would have caught it. The fix is two knobs down: rank 16, 1,500 steps. Retrain, re-run the *exact same* held-out eval including the no-trigger portrait, and confirm the portrait is clean before shipping. The cost of skipping the gate was a LoRA that subtly degraded every non-product image — invisible until a user complains. The gate is cheap; the leak is expensive.

## 5. Speed and deployment: hitting a latency and VRAM target

Now the part that decides whether you have a demo or a product. The series spent a whole track on this — the [four-lever cost model](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it), [consistency/LCM distillation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation), [DMD adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation), and [quantization and caching](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) — and the playbook move is to *combine* the levers to hit a stated target, because they attack different terms and their savings multiply.

Start from the cost model, because you optimize what you measure. Latency for one image is

$$
T_\text{image} \;=\; T_\text{encode} \;+\; N \cdot T_\text{denoise} \;+\; T_\text{decode},
$$

where $T_\text{encode}$ is the one-time text encode, $N$ is the step count, $T_\text{denoise}$ is the per-step denoiser forward pass (doubled if CFG isn't distilled away), and $T_\text{decode}$ is the one-time VAE decode. The denoiser term $N \cdot T_\text{denoise}$ dominates — for FLUX at 50 steps it's ~6 of the ~6.1 seconds — so the whole game is shrinking $N$ and $T_\text{denoise}$, and three levers do exactly that:

- **Distillation cuts $N$.** LCM/Turbo/schnell take $N$ from 30–50 down to 4 (and DMD2 to 1), a roughly 8–12× cut, *and* usually kill the CFG 2× because guidance is baked in. This is the biggest single lever.
- **Quantization cuts $T_\text{denoise}$.** fp8 or 4-bit (SVDQuant) shrinks the per-pass cost by moving fewer bytes and running on faster low-precision tensor cores — a 1.5–2× per-pass speedup *and* a 2–4× VRAM cut, which is what makes a 12B model fit a 24 GB card.
- **Caching cuts effective $N \cdot T_\text{denoise}$.** DeepCache (U-Net) and TeaCache (DiT) reuse features across adjacent steps when the latent barely changes, skipping a fraction of the per-step compute for free.

The point is they *stack*. A 4-step Turbo model (distillation), quantized to fp8 (quantization), with TeaCache (caching) is the product of all three savings — which is why you can take FLUX from 6.1 s to under a second.

![A before and after comparison showing FLUX moving from fifty fp16 steps at six seconds to a four-step fp8 distilled model with caching at under one second on one RTX 4090.](/imgs/blogs/building-an-image-generation-stack-8.png)

Figure 8 stacks the levers on a real number. Left: FLUX.1-dev, 50 steps, fp16, ~24 GB, 6.1 s. Right: FLUX.1-schnell (distilled to 4 steps), fp8 (~12 GB), with feature caching — under a second on the same RTX 4090. Same family, three levers pulled, a ~6× latency win and a ~2× memory win, with a quality delta you'd struggle to see in a blind test. Each lever attacked a different term, so none of them stepped on the others.

#### Worked example: hit "under 1 second on a 4090" for FLUX-quality output

The target: FLUX-family quality, under 1 second per 1024² image, on a single 24 GB RTX 4090, commercially licensed. Walk the levers against the cost model. Base fp16 FLUX.1-dev at 50 steps is 6.1 s and 24 GB and the wrong license — fails on all three. **Lever 1, distillation:** switch to FLUX.1-schnell (Apache-2.0, fixes the license) at 4 steps — that's a ~12× cut in $N$, taking the denoiser term from ~6 s toward ~0.5 s. **Lever 2, quantization:** fp8 weights drop VRAM from ~24 GB to ~12 GB (comfortable headroom for a second concurrent request) and give a ~1.3× per-pass speedup. **Lever 3, runtime:** `torch.compile` the transformer and turn on SDPA/FlashAttention for another ~1.2–1.4×, plus channels-last memory format. Net: 4 steps × ~0.15 s/step ≈ 0.6 s plus ~0.2 s of encode+decode ≈ **under 1 second, ~12 GB, Apache-2.0** — target hit on all three constraints, with the heaviest lift (12× of it) coming from distillation. If you needed it *even* faster you'd reach for a 4-bit SVDQuant transformer and a tiny `AutoencoderTiny` decoder, trading a sliver of quality for the last few hundred milliseconds.

Here's the optimized serving pipeline assembled — the efficiency flags from the [quantization/caching post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) in one block:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",          # distilled: 4 steps, Apache-2.0
    torch_dtype=torch.bfloat16,
).to("cuda")

# Runtime levers (free quality, real speed):
pipe.transformer = torch.compile(                # fuse kernels, cut Python overhead
    pipe.transformer, mode="max-autotune", fullgraph=True
)
pipe.transformer.to(memory_format=torch.channels_last)
pipe.enable_vae_tiling()                          # high-res decode without OOM

# Memory lever (turn on only if VRAM-bound):
# pipe.enable_model_cpu_offload()                # text encoders to CPU when idle

def generate(prompt: str, seed: int = 0):
    return pipe(
        prompt,
        num_inference_steps=4,        # schnell is distilled to 4 steps
        guidance_scale=0.0,           # distilled: guidance baked in, CFG OFF
        height=1024, width=1024,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]

# Warm up once so the torch.compile graph is cached before timing/serving.
_ = generate("warmup", seed=0)
```

Two traps in that block worth calling out. First, `guidance_scale=0.0` — schnell is guidance-distilled, so passing a CFG scale double-applies guidance and ruins the image; this is the single most common FLUX-schnell bug. Second, the **warmup** call: `torch.compile` compiles on the first real input, so the first generation is slow (10–60 s of compilation) and every one after is fast — you must warm up before you start the latency clock or serve traffic, or your first user eats the compile.

Now serving. The simplest production server is a FastAPI endpoint that holds the pipeline resident and generates on request. Loading the model per-request is the cardinal sin (model load is 5–30 s); load once at startup, reuse forever:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import base64, io, torch

app = FastAPI()

class Req(BaseModel):
    prompt: str
    seed: int = 0

@app.on_event("startup")
def load():
    global pipe
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune")
    _ = pipe("warmup", num_inference_steps=4, guidance_scale=0.0)   # warm the graph

@app.post("/generate")
@torch.inference_mode()                       # no autograd graph -> less memory, faster
def generate(req: Req):
    img = pipe(req.prompt, num_inference_steps=4, guidance_scale=0.0,
               generator=torch.Generator("cuda").manual_seed(req.seed)).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return {"image": base64.b64encode(buf.getvalue()).decode()}
```

That endpoint serves one request at a time on one GPU. To scale, the central idea is **batching**, and it's worth understanding *why* it works because it's the single biggest throughput lever in serving. The denoiser is a big matmul-bound network, and a GPU running one image is mostly *underutilized* — the matmuls don't fill the tensor cores, and a lot of time goes to kernel-launch overhead and memory round-trips that don't scale with batch size. When you stack 4 prompts into a batch, the denoiser processes all four in one forward pass at *near-constant* latency (the matmuls get wider, not slower, until you saturate compute), so four images cost roughly what one did and your throughput quadruples. The math: if one image is $T$ seconds and a batch of $B$ is $T_B \approx T + (B-1)\delta$ with $\delta \ll T$ (the marginal cost of one more image in the batch), then throughput is $B / T_B$, which climbs steeply until the GPU saturates and $\delta$ starts growing. The practical recipe is **dynamic batching**: a request comes in, you hold it for a few milliseconds to collect more, then run the collected batch together. You trade a little tail latency for a large throughput gain — exactly the right trade for an API serving many users.

Around the batching core, the rest of the serving architecture is standard. Run **one process per GPU** behind a load balancer (a single process can't use two GPUs for one model efficiently, so you replicate). Put a **queue** in front so bursts don't OOM — without it, a traffic spike tries to allocate VRAM for too many concurrent generations at once and the card dies; the queue caps in-flight work at what fits. Cap the **maximum batch size** at what your VRAM allows (each image in the batch costs activation memory that scales with resolution). For complex multi-step graphs (ControlNet + upscale + face-fix), **ComfyUI** as a serving backend gives you a visual node graph and an API, and is the pragmatic choice when the pipeline is more than a single `pipe()` call — it handles the wiring of a dozen nodes that would be brittle hand-written Python. The `@torch.inference_mode()` decorator is a free win — it disables autograd bookkeeping, cutting memory and a little latency, and it should wrap every generation path in production.

A note on the *cold-start* problem that bites every team once. The first request after a deploy is slow for three compounding reasons: the model weights load from disk (5–30 s for a 12B model), the `torch.compile` graph compiles on the first real input shape (another 10–60 s), and the CUDA context and kernels warm up. If you let a real user hit a cold worker, they wait a minute. The fix is a **readiness probe** that doesn't mark a worker healthy until it has loaded *and* warmed up — generate a throwaway image at every shape you serve (1024², any other resolution) during startup, and only then accept traffic. Autoscaling makes this worse, not better: a scale-up event spins a fresh cold worker exactly when you're under load, so pre-warm aggressively and scale up *before* you're saturated, not after. This is the diffusion-serving version of a lesson the [edge-AI and efficient-inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) work hammers — the steady-state numbers lie about the first request.

#### Worked example: sizing a GPU fleet for 100 images per second

The product team says peak load is 100 images per second. Size the fleet. Your optimized stack does $T = 0.8$ s/image single, and batching to $B = 8$ runs at $T_8 \approx 1.4$ s for the batch (so $1.4 / 8 = 0.175$ s/image effective). One GPU's throughput is therefore $\approx 1 / 0.175 \approx 5.7$ images/second at batch 8. To serve 100 images/second you need $\lceil 100 / 5.7 \rceil = 18$ GPUs at peak, plus headroom for the batch-collection latency and request variance — call it 20–24 GPUs. The cost math: at \$0.40/GPU-hour and 20 GPUs, that's \$8/hour at peak, and if average load is a quarter of peak with autoscaling, your steady-state bill is closer to \$2–3/hour. Notice what moved the answer: dropping $T$ from 6.1 s (fp16 base) to 0.175 s (distilled + quantized + batched) cut the fleet from ~600 GPUs to ~20 — a **30× reduction in hardware cost** that came entirely from the speed levers in this section. That is why the speed track isn't a nice-to-have; it's the difference between a feature that's economically viable and one that isn't.

The serving cost model closes the loop on the brief. If your stack does $T$ seconds per image on a GPU that costs \$$C$/hour, the GPU cost per image is

$$
\text{cost/image} \;=\; \frac{T}{3600}\cdot C \;\div\; u,
$$

where $u \in (0,1]$ is your utilization (batching and queueing push $u$ toward 1; idle GPUs waiting for traffic push it down). A concrete instance: a cloud RTX 4090 around \$0.40/hour, $T = 0.8$ s/image, batching to $u \approx 0.8$ → cost ≈ $\frac{0.8}{3600}\times 0.40 \div 0.8 \approx$ **\$0.0001 per image** of raw GPU. That is the number that tells you whether the feature is viable at scale, and it's dominated by $T$ (which the speed levers control) and $u$ (which batching controls) — the two things you just optimized.

## 6. The shipping decision: target-driven recipes

You don't pick a stack in the abstract — you pick it for a *target*. The right way to assemble everything above is to start from the latency and cost the product demands, and let that force the base, the steps, and the efficiency stack. Four targets cover most of what teams actually ship.

![A four by four matrix mapping shipping targets to recommended stacks across base model, steps and sampler, efficiency techniques, and an estimated cost per image.](/imgs/blogs/building-an-image-generation-stack-7.png)

Figure 7 is the recipe table. Read each row as a complete stack the target forced:

- **Real-time demo on a 4090.** Latency is king, quality is "good enough for a live demo." Stack: **SDXL-Turbo**, 4 steps with `LCMScheduler` (`guidance_scale=0`), fp16 with SDPA. Sub-second, ~\$0.0003/image. This is the "make it feel instant in the booth" stack.
- **Cheap API at scale.** You're serving thousands of images and cost-per-image is the metric. Stack: **FLUX.1-schnell** (Apache-2.0, so you can charge for it), 4 steps, fp8 + TeaCache, batched serving. High quality, ~\$0.002/image, scales horizontally. This is the "API product" stack.
- **Max-quality offline render.** No latency budget — a poster, a hero image, a print asset. Stack: **FLUX.1-dev** (or commercial-licensed FLUX), 28–30 steps with DPM++/flow-match, bf16, *no* distillation or caching (you want every drop of quality), maybe an upscale pass. ~\$0.02/image, minutes is fine. This is the "it has to be perfect" stack.
- **Custom brand style.** The defining requirement is *your* aesthetic, applied consistently. Stack: **SDXL + your trained LoRA** (section 4), 25 steps DPM++, fp16 with CPU offload if VRAM-bound. ~\$0.004/image. This is the "on-brand at scale" stack, and it's the one that needs the fine-tune workflow.

Those four cover the space because they sit at the corners of the trilemma: the demo trades quality for speed, the render trades speed for quality, the API balances both at low cost, and the brand stack trades a training investment for consistency. The cost-per-image column is the reality check — a feature that needs to be free-to-the-user had better be in the \$0.0003–\$0.002 rows, not the \$0.02 row, unless each image is worth real money.

| Target | Base | Steps | Efficiency | ~Latency | ~Cost/image |
| --- | --- | --- | --- | --- | --- |
| Real-time demo | SDXL-Turbo | 4 (LCM) | fp16, SDPA | <0.5 s | \$0.0003 |
| Cheap API at scale | FLUX.1-schnell | 4 | fp8 + TeaCache | ~0.8 s | \$0.002 |
| Max-quality render | FLUX.1-dev | 28–30 (DPM++) | bf16, no cache | ~30 s | \$0.02 |
| Brand style at scale | SDXL + LoRA | 25 (DPM++) | fp16 + offload | ~3 s | \$0.004 |

The cost figures are order-of-magnitude GPU-only estimates on commodity cloud pricing (≈\$0.40–\$2/hr depending on the card and provider); your real number adds serving overhead, storage, and the safety stack's inference, and depends heavily on utilization. Treat them as ratios — the demo is ~60× cheaper than the render — more than absolute truths.

## 7. Evaluating before you ship

Before a single generated image reaches a user, you evaluate — and "evaluate" does not mean "look at three samples and feel good." The [honest-evaluation post](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) made the case that any single metric lies; the production move is a *basket* of complementary measures plus a human gate, because each metric covers a blind spot the others miss.

The metric basket has four parts. **FID** (or the more modern **FID-DINOv2**) measures distributional realism — how close your generated images' feature statistics are to a real reference set — and it's the standard for "do these look real," but it's blind to whether the image matches the *prompt* and it's noisy below ~10k samples. **CLIP-score** measures prompt-image alignment — does the picture contain what the text asked — and catches the failure FID can't, but it saturates and can't tell "a red cube" from "a cube that is red-ish." **A human-preference model** (HPSv2, ImageReward, PickScore) approximates aesthetic quality and human taste, correlating better with what users actually prefer than FID does. And **compositional probes** (GenEval, T2I-CompBench) specifically stress the binding failures — counting, spatial relations, attribute binding — that aggregate metrics smear over. You run all four because a model can score great on FID (realistic) and badly on GenEval (can't count to three), and only the probe catches it.

The science of FID is worth stating because it's the one number everyone quotes and few can define. FID is the Fréchet distance between two multivariate Gaussians fit to the Inception (or DINOv2) features of the real and generated sets:

$$
\text{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right),
$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of the real and generated feature distributions. The first term penalizes a shift in the *average* feature (mode mismatch), the second penalizes a difference in the *spread and correlation* of features (diversity mismatch). Lower is better; 0 means the two Gaussians coincide. The honest caveats that the formula makes obvious: it assumes the features are Gaussian (they aren't, exactly), it depends entirely on the *reference set* you compare against (FID against COCO and FID against your product catalog are different numbers), and it needs enough samples — typically 10k–50k — for $\Sigma_g$ to be well-estimated, so an FID computed on 500 images is mostly noise. **Report FID with its reference set and sample count, fixed seeds, and a warm-up, or it's not reproducible.**

Here's a real evaluation harness — `torchmetrics` FID plus a CLIP-score, the two-number minimum:

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

fid = FrechetInceptionDistance(feature=2048, normalize=True).to("cuda")
clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to("cuda")

# real_images, gen_images: uint8 tensors [N,3,H,W]; prompts: list[str] aligned to gen
fid.update(real_images.to("cuda"), real=True)
fid.update(gen_images.to("cuda"),  real=False)
print("FID:", fid.compute().item())                 # lower = more realistic; needs >=10k

for img, prompt in zip(gen_images, prompts):
    clip.update(img.unsqueeze(0).to("cuda"), prompt)
print("CLIP-score:", clip.compute().item())         # higher = better prompt alignment
```

But the metrics are the *floor*, not the ceiling. The human spot-check catches what they miss: a **fixed prompt suite** of 50–100 prompts spanning your product's real distribution (every category, plus the hard compositional cases — "three red apples on the left, two green pears on the right"), generated at a fixed seed, reviewed by a human before each release. The metrics tell you the model didn't regress *on average*; the prompt suite tells you it didn't catastrophically fail on the cases you care about. Both, every release. A model that improved FID by a point but started failing the "three apples" probe is a regression for a product that counts.

#### Worked example: catching a regression that FID missed

You distill your SDXL base to a 4-step Turbo for the latency win. FID barely moves (8.1 → 8.4, within noise), CLIP-score holds, the demo looks snappy — ship it? Run the compositional probe first. GenEval drops from 0.55 to 0.41, and the failure is *counting and spatial relations*: the distilled model, generating in 4 steps, lost the fine multi-step refinement that placed distinct objects correctly, so "three apples on the left" now gives you two-or-four apples scattered. FID couldn't see it because the images are *individually* realistic — each apple looks great, there's just the wrong number in the wrong place. The probe caught a real quality loss the aggregate metric averaged away. The decision: ship the Turbo model for single-subject prompts (where it's indistinguishable and 10× faster) and route multi-object compositional prompts to the 28-step base. That's a *product* decision the eval enabled — not "is the model good" but "good *for which requests*."

## 8. The safety layer: filter, watermark, provenance

A generation product without a safety layer is a liability with a UI. The [safety and provenance post](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) covered the techniques; the playbook is to wire three of them, in order, around every request, because each addresses a different threat and skipping any one leaves a hole.

**First, a safety classifier on the input and the output.** On the *input*, screen the prompt for clearly-disallowed requests before you spend a GPU-second on them. On the *output*, run an NSFW/safety classifier on the generated image and block or blur anything that trips it — because a clean prompt can still produce a problematic image, and the input filter alone is insufficient. `diffusers` ships a `safety_checker` for SD models; in production you'd typically run a stronger dedicated classifier. The cost is one cheap forward pass per image, a rounding error next to the denoiser, and it is non-negotiable for a public product.

**Second, an invisible watermark on every output.** Embed a robust, imperceptible watermark in every generated image so it can later be identified as machine-generated and traced to your system. The [provenance post](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) covered the approaches — Stable Signature (fine-tune the VAE decoder to emit a watermark), Tree-Ring (watermark the *initial noise* so it survives the whole diffusion process), SynthID (a learned encoder/decoder pair). The properties that matter: *imperceptible* (a user can't see it), *robust* (survives JPEG, crop, resize, screenshot), and *detectable* by you with a secret key. This is what lets you answer "did our system make this image?" months later, and it's increasingly a regulatory expectation, not a nice-to-have.

**Third, C2PA content credentials.** Attach cryptographically-signed provenance metadata — Coalition for Content Provenance and Authenticity credentials — recording that the image was AI-generated, by which model, when. Unlike the watermark (which is *in* the pixels and survives stripping), C2PA is *metadata* (which is robust to nothing if stripped, but is a verifiable, standardized, human-readable record when present). You ship *both*: the watermark for robustness, C2PA for standards-compliance and interoperability. Together they're belt-and-suspenders provenance.

Wired into the serving endpoint, the order is: classify input → generate → classify output → watermark → attach C2PA → return. In code shape:

```python
def safe_generate(prompt: str, seed: int = 0):
    if input_classifier(prompt).blocked:              # 1a. input filter
        raise ValueError("prompt rejected by safety policy")

    image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0,
                 generator=torch.Generator("cuda").manual_seed(seed)).images[0]

    if output_classifier(image).nsfw_score > THRESH:  # 1b. output filter
        return blocked_placeholder()

    image = embed_watermark(image, key=WATERMARK_KEY) # 2. invisible watermark
    image = attach_c2pa(image, model="flux-schnell",  # 3. provenance credential
                        generated=True, timestamp=now())
    return image
```

The safety layer adds a few milliseconds and a few lines, and it is the difference between a product and an incident. Don't ship without all three. The classifier stops the obvious harms, the watermark makes outputs traceable, and C2PA makes the provenance verifiable and standards-compliant — three different holes, three different patches.

## 9. The whole stack, assembled: a "ship a product" scenario

Let's tie it together with the brief from the top: a paid feature that turns a customer's prompt into an on-brand 1024² product photo, under two seconds, on affordable hardware, with a safety gate. Walk the stack.

**Base model (section 1).** It's a *paid* product, so FLUX.1-dev's non-commercial license is out. It needs *speed*, so a distilled model. It needs *brand style*, so a fine-tunable base with a good ecosystem. The call: **SDXL** as the base (commercial-safe, deep LoRA ecosystem, Turbo path for speed) — or **FLUX.1-schnell** if we want top-tier quality with commercial rights and accept a thinner LoRA ecosystem. Say SDXL, because the brand-LoRA workflow is most mature there.

**Inference pipeline (section 2).** SDXL native 1024², `DPMSolverMultistepScheduler` for the quality base, but we'll distill for speed (below). Negative prompt for the usual quality floor. Fixed seed in the A/B harness, random in production.

**Control and customization (section 3).** The brand style is a *reusable learned concept* → a **LoRA** (not a prompt, which can't hold a consistent aesthetic; not a fine-tune of full weights, which is overkill). If customers upload a reference product to match, add an **IP-Adapter**; if they need a specific layout, a **ControlNet** — both inference-time, both composable.

**Fine-tune (section 4).** Curate 20–30 on-brand product shots, captioned with a trigger token; train a rank-16 LoRA for ~1,500 steps on the 4090; gate on held-out prompts *including a no-trigger portrait* to catch leak; ship the ~150 MB adapter.

**Speed and serving (section 5).** Target is <2 s, so distill: **SDXL-Lightning/Turbo** at ~8 steps (keeps more quality than 4 for a product), fp16, `torch.compile`, SDPA, the LoRA loaded on top. That's well under 2 s on a 4090. Serve behind a warmed-up FastAPI endpoint, one process per GPU, batched, with a queue. Cost-per-image lands around \$0.001–0.004 — viable for a paid feature.

**Evaluation (section 7).** Before each release: FID against our *product catalog* reference set (not COCO), CLIP-score on the prompt suite, GenEval on the compositional cases, and a human spot-check of the 50-prompt suite at a fixed seed. Confirm the distilled+LoRA stack didn't regress counting or leak the brand style into off-brand prompts.

**Safety (section 8).** Input classifier → generate → output NSFW classifier → invisible watermark → C2PA credential → return. Non-negotiable for a public, paid product.

That's the entire series in one feature: the [foundations](/blog/machine-learning/image-generation/why-generating-images-is-hard) told us *why* it's a distribution-modeling problem; [latent diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and the [DiT/flow-matching recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) gave us the base; [guidance](/blog/machine-learning/image-generation/classifier-free-guidance) and [LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora) gave us control; [distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) and [quantization](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) gave us speed; and [evaluation](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) and [provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) gave us the gate. Every layer was a post; the product is the assembly.

## 10. Case studies and real numbers

A few concrete points from the literature and shipped models, to anchor the recipes in measured reality (cite the source; where I give a round number it's the reported headline, and I flag approximations).

**SDXL vs SD1.5 (Podell et al., 2023).** SDXL is a 2.6B-parameter U-Net (vs SD1.5's ~0.9B) with a dual text-encoder stack (OpenCLIP ViT-bigG + CLIP ViT-L) and size/crop conditioning, native at 1024² vs 512². The jump in prompt-following and 1024² fidelity is the reason SDXL became the commercial default — at roughly 3× the parameters and a correspondingly heavier per-step cost, which is exactly why the distillation track matters for SDXL serving.

**FLUX.1 family (Black Forest Labs, 2024).** A 12B-parameter rectified-flow DiT with a double-stream/single-stream design and a CLIP+T5 encoder stack. **FLUX.1-dev** is the quality leader under a non-commercial license; **FLUX.1-schnell** is timestep-distilled to ~4 steps under Apache-2.0 — the same architecture, traded down the quality/speed axis, with the license difference that decides whether you can charge for it. The 12B size is why fp16 FLUX overshoots a 24 GB card and why the [4-bit SVDQuant result](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) was such a big deal: it brought FLUX to ~7 GB.

**SANA (Xie et al., 2024).** A 0.6B DiT that generates up to 4K images fast on modest hardware via two moves: a deep-compression autoencoder that shrinks the latent **32×** (vs SD's 8×), so the DiT processes far fewer tokens, and **linear attention** that removes the quadratic-in-tokens cost at high resolution. The result is a small model that punches well above its parameter count at high resolution — the reason it's the edge/4K pick in the matrix.

**LCM and SDXL-Turbo (Luo et al., 2023; Sauer et al., 2023).** Consistency distillation (LCM) and adversarial distillation (ADD → SDXL-Turbo) cut SDXL from ~30–50 steps to **1–4 steps**. SDXL-Turbo can produce a usable 512² image in a *single* step; quality lifts a bit at 4. The honest trade-off: distilled few-step models lose some fine detail and compositional precision (the GenEval drop in the worked example), which is why the playbook routes hard compositional prompts to the un-distilled base.

**DMD2 (Yin et al., 2024).** Distribution Matching Distillation pushes to **one-step** generation with reported FID competitive with the multi-step teacher on some benchmarks — the GAN loss returning as a distillation signal, covered in the [distillation post](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation). One-step text-to-image is the extreme end of the speed lever; it's the right tool for true real-time, with a quality ceiling below the 28-step base.

**ControlNet on a real conditioning (Zhang et al., 2023).** The original ControlNet paper showed that a single architecture absorbs wildly different conditioning signals — Canny edges, Hough lines, HED soft edges, depth, normal maps, OpenPose skeletons, segmentation maps, scribbles — each as a separately-trained adapter on a frozen SD1.5 base, trainable on a single consumer GPU in a day or two because only the clone updates. The measured result that matters for the playbook: the control fidelity is high enough that you can pose a character precisely or match a depth layout, while the prompt still governs content and style. That separation of concerns — *structure* from the control image, *content* from the prompt — is exactly why ControlNet sits in the inference-time branch of the decision tree: it adds a controllable axis without you having to train or even touch the base model.

The through-line across all of these: **there is no free lunch on the trilemma.** SDXL bought quality with parameters (and paid in speed). FLUX bought more quality with more parameters (and paid in VRAM and, for -dev, license). SANA bought high-res speed with aggressive latent compression (and a small-model quality ceiling). Distillation bought speed with steps (and paid in compositional precision). Every shipped model is a *chosen point* on quality × diversity × speed, and your job is to choose the point your product needs — which is the whole series in one sentence.

## 11. When to reach for each layer (and when not to)

The most valuable thing a playbook can do is tell you when *not* to do something, because the expensive mistakes are over-engineering, not under-engineering. Plain rules:

- **Don't fine-tune when a prompt works.** If "watercolor style" in the prompt gets you the look, you do not need a LoRA. Training is a cost — data, compute, an eval gate, a file to ship and version. Pay it only for a *reusable* concept the model genuinely doesn't know.
- **Don't full-fine-tune when a LoRA suffices.** Updating all 2.6B SDXL weights to learn a style is wasteful and overfits; a rank-16 LoRA captures style adaptation in 32× fewer parameters and ships as a swappable 150 MB file. Full fine-tunes are for foundation-scale capability changes, not brand styles.
- **Don't run 50 steps when 25 converge.** Past the convergence point, more steps is pure latency with no visible quality. Find the knee (sweep steps at a fixed seed, look for where the image stops changing) and stop there.
- **Don't distill to 1–4 steps for max-quality offline renders.** Distillation trades compositional precision and fine detail for speed; if you have no latency budget (a print asset), run the full base at 28–30 steps. Save distillation for the latency-bound paths.
- **Don't crank CFG to force adherence.** Above ~7.5 on SDXL (or ~5 on FLUX) you get saturation and burned highlights, not better prompts. If adherence is the problem, a better prompt, a ControlNet, or an IP-Adapter is the fix — not more guidance.
- **Don't pass a CFG scale to a distilled model.** Turbo/schnell baked guidance in; passing `guidance_scale > 0` double-applies it. Set it to 0.
- **Don't quantize the sensitive layers.** The first/last layers and the modulation paths do outsized damage when quantized; keep them in higher precision and quantize the bulk blocks, per the [quantization post](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference).
- **Don't ship without the eval basket and the safety gate.** A single metric lies and a missing safety layer is an incident waiting to happen. The basket and the gate are cheap insurance against expensive failures.
- **Don't pick FLUX.1-dev for a commercial product.** The non-commercial license is a real constraint. Use FLUX.1-schnell (Apache-2.0), SDXL, or a commercially-licensed FLUX instead.

The meta-rule behind all of these: **climb the cost ladder only when the cheaper rung fails.** Prompt before adapter, adapter before fine-tune, base before distillation, default knobs before tuning. Most products need far less than their engineers reach for.

## 12. Key takeaways

- **Pick the base model from your binding constraint, not "which is best."** VRAM gates first (a 12B FLUX needs 24 GB in fp16), then license (FLUX.1-dev is non-commercial), then quality vs ecosystem. SDXL is the safe commercial default; FLUX.1-schnell is the fast commercial-licensed option; SANA is the speed-at-resolution pick.
- **The inference pipeline is five knobs:** sampler+steps (DPM++ 25–30, or LCM 4–8 distilled), CFG scale (7 for SDXL, ~4 for FLUX, **0** for distilled), native resolution, seed, and a negative prompt. Ninety percent of "looks wrong" tickets are one of these set badly.
- **Climb the control ladder cheapest-first:** prompt → ControlNet (layout) → IP-Adapter (reference) → LoRA (reusable concept) → DreamBooth (exact subject) → instruction editing (image-in image-out). They compose, each with a strength knob.
- **A fine-tune is a loop with an evaluation gate**, not a one-shot run. Curate diverse data, train a rank-16 LoRA, and gate on *held-out* prompts including a no-trigger case to catch overfit-and-leak — the failure that silently degrades off-target images.
- **Speed levers stack because they attack different terms:** distillation cuts step count $N$, quantization cuts per-pass cost $T_\text{denoise}$ and VRAM, caching cuts effective compute. Combined, they take FLUX from 6.1 s to under 1 s on a 4090.
- **Serve the model resident, warmed up, batched, one process per GPU**, behind a queue. Load-per-request and skipping the `torch.compile` warmup are the two cardinal serving sins. Cost/image = latency × GPU-rate ÷ utilization.
- **Start from the target, not the model.** A real-time demo, a cheap API, a max-quality render, and a brand-style product force four different stacks — and the cost/image column (≈\$0.0003 to \$0.02) is the viability check.
- **Evaluate with a basket, not a metric.** FID for realism, CLIP-score for alignment, a preference model for taste, GenEval/T2I-CompBench for composition — plus a fixed human prompt-suite gate every release. One number always lies.
- **Ship the safety stack: classifier (in and out) → invisible watermark → C2PA.** Three different holes, three patches. Non-negotiable for a public product.
- **There is no free lunch on the trilemma** (quality × diversity × speed). Every model and every technique is a chosen point; your job is to choose the point your product needs.

## Further reading

- **Ho, Jain, Abbeel (2020), "Denoising Diffusion Probabilistic Models"** — the DDPM paper; the engine the whole stack runs on. See [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles).
- **Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models"** — LDM/Stable Diffusion; latent-space diffusion is why this is affordable. See [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion).
- **Podell et al. (2023), "SDXL"** and **Black Forest Labs (2024), FLUX.1** — the workhorse and the frontier; the two bases most of this playbook targets.
- **Esser et al. (2024), "Scaling Rectified Flow Transformers" (SD3)** and **Peebles & Xie (2023), "Scalable Diffusion Models with Transformers" (DiT)** — the modern backbone. See [diffusion transformers](/blog/machine-learning/image-generation/diffusion-transformers-dit) and [the modern recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe).
- **Hu et al. (2022), "LoRA"** and the 🤗 `diffusers` training docs — the fine-tune workflow in section 4. See [personalization](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora).
- **Luo et al. (2023), "Latent Consistency Models"** and **Sauer et al. (2023), "Adversarial Diffusion Distillation" (SDXL-Turbo)** — the distillation that makes real-time possible. See [consistency models](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) and [distribution matching](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation).
- **Heusel et al. (2017), "GANs Trained by a Two Time-Scale Update Rule" (FID)** and the GenEval/T2I-CompBench benchmarks — the eval basket. See [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly).
- **The 🤗 `diffusers` documentation** (pipelines, schedulers, LoRA, optimization) — the canonical reference for every API in this post; and [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) for the shipping gate.
