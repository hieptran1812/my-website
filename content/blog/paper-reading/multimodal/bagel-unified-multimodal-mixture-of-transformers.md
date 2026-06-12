---
title: "BAGEL: one Mixture-of-Transformers that both understands and draws"
date: "2026-06-12"
publishDate: "2026-06-12"
description: "How ByteDance's BAGEL-7B-MoT fuses a vision-language understander and a rectified-flow image generator into one 14B/7B-active model, and why complex editing only emerges after trillions of interleaved tokens."
tags: ["bagel", "unified-multimodal", "mixture-of-transformers", "image-generation", "vision-language-model", "any-to-any", "rectified-flow", "open-weights", "bytedance-seed", "image-editing"]
category: "paper-reading"
subcategory: "Multimodal"
author: "Hiep Tran"
featured: true
readTime: 82
---

For three years the unspoken law of multimodal AI was: a model either *reads* pictures or *paints* them, never both well. The vision-language crowd — LLaVA, Qwen2.5-VL, InternVL — got very good at looking at an image and answering questions, and they shipped on the back of contrastively trained vision encoders that throw away pixel detail in exchange for clean semantics. The generation crowd — Stable Diffusion, FLUX, the diffusion-transformer lineage — got very good at turning text into pixels, and they leaned on variational autoencoders that preserve every brushstroke. The two camps used incompatible image representations, incompatible training objectives, and incompatible mental models. When somebody tried to glue them together — Chameleon quantizing everything to discrete tokens, Janus splitting the encoder, Transfusion interleaving diffusion and autoregression — the result was a model that did each thing *acceptably* and neither thing *well*. The standard wisdom hardened into a tradeoff: unification taxes you.

BAGEL is ByteDance Seed's argument that the tax is an artifact of bad architecture, not a law of nature. It is a single decoder-only foundation model — 14B parameters total, 7B active per token — that natively understands images *and* generates them, trained end-to-end on trillions of tokens of interleaved text, image, video, and web data. On understanding benchmarks it beats Qwen2.5-VL-7B and InternVL2.5-8B, the specialist VLMs. On text-to-image it lands at GenEval 0.88, ahead of SD3-Medium (0.74) and FLUX.1-dev (0.82). And — this is the part that made the paper interesting rather than merely competent — its hardest abilities (free-form editing, future-frame prediction, 3D rotation, world navigation) do not appear gradually. They *emerge*, at distinct data thresholds, after the simpler capabilities have already saturated. The paper's title is "Emerging Properties in Unified Multimodal Pretraining," and it earns the word "emerging."

![BAGEL Mixture-of-Transformers architecture with shared self-attention, understanding expert, generation expert, and dual encoders](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-1.png)

<!-- FIGSPEC 1
kind: layered-stack
claim: BAGEL routes text and two kinds of image tokens through two transformer experts that share self-attention at every layer.
caption: One sequence, two experts, shared attention — the MoT design that lets understanding and generation co-exist without task conflict.
nodes:
  - id: in | label: "Input: text + image + video (interleaved)" | color: gray
  - id: vit | label: "Und encoder: SigLIP2-so400m/14 (semantic tokens)" | color: blue
  - id: vae | label: "Gen encoder: FLUX VAE 8x, 16ch (pixel latents)" | color: green
  - id: attn | label: "Shared self-attention (every layer, all tokens see all)" | color: amber
  - id: undx | label: "Understanding expert: text + ViT tokens" | color: blue
  - id: genx | label: "Generation expert: noised VAE tokens" | color: green
  - id: out | label: "Out: text (next-token) OR image (rectified flow)" | color: gray
notes: vertical stack, top=input, bottom=output; attn band spans both experts; 14B total / 7B active per token
-->

The diagram above is the mental model, and the rest of this post is a tour of it. Read it top to bottom. Input is an interleaved stream — words, pictures, video frames — in whatever order a real document or conversation presents them. Each image is encoded *twice*, by two different encoders that disagree about what an image *is*: a SigLIP2 vision transformer that produces clean semantic tokens for understanding, and a FLUX VAE that produces pixel-faithful latents for generation. Both streams of tokens, plus the text tokens, flow into a transformer stack where **every layer runs a single shared self-attention** over the whole sequence — but routes each token through one of two parameter sets ("experts") depending on what kind of token it is. Text and ViT tokens go through the understanding expert; VAE tokens go through the generation expert. At the output, text is produced autoregressively (next-token prediction) and images are produced by rectified flow (a diffusion-style denoising objective). One model, two output modalities, no glue.

In this post we will pull that apart layer by layer: why two experts instead of one shared backbone, why two encoders instead of one, how the attention mask makes autoregression and diffusion coexist in the same forward pass, what the unified loss actually optimizes, and — the headline — what the emergence curve looks like and why it bends where it does. We will run real inference for both understanding and generation, read the benchmark tables honestly (including where BAGEL loses), and close with named case studies and a blunt "when to reach for this / when not."

## Why "unified" was a trap, and how BAGEL escapes it

The senior-engineer instinct when you hear "one model that does everything" is to reach for your wallet, because *everything* usually means *nothing especially well*. Let us be precise about the trap, because the trap is what BAGEL's architecture is shaped to avoid.

Image understanding and image generation want *opposite* things from their representations. Understanding wants **semantic abstraction**: a good understanding encoder is trained contrastively (CLIP, SigLIP) to map a photo of a golden retriever and the words "a golden retriever" to nearby points, which means it deliberately discards the exact pixels — the lighting, the precise fur texture, the JPEG artifacts — because those are noise relative to the concept. Generation wants the exact opposite: a generation representation must **preserve every pixel** it will later reconstruct, which is why diffusion models use a VAE whose entire job is lossy-but-faithful reconstruction, not semantic clustering. A single shared image tokenizer cannot be both abstract and faithful; it has to compromise, and the compromise shows up as either fuzzy understanding or blurry generation.

This is the rock that earlier unified models broke on. Here is the lineage, and where each one paid the tax.

| Model | Image representation | Generation method | The tax it paid |
|---|---|---|---|
| Chameleon | Single VQ tokenizer (discrete) | Autoregressive over image tokens | Quantization caps fidelity; weak T2I |
| Transfusion | Continuous latents, shared backbone | Diffusion inside one transformer | Shared weights → task interference |
| Show-o | Discrete tokens | Masked + autoregressive hybrid | Discrete tokens cap image quality |
| Janus / Janus-Pro | **Decoupled encoders**, shared transformer | Autoregressive over discrete gen tokens | Shared transformer still mixes objectives |
| **BAGEL** | **Decoupled encoders (SigLIP2 + VAE)** | **Rectified flow in a separate expert** | Decoupled *encoders and weights* |

The progression is a story of progressively *decoupling* the two tasks. Chameleon coupled everything — one tokenizer, one set of weights, one objective. Janus took the crucial first step of decoupling the *encoders* (understanding via SigLIP, generation via a separate path) while keeping a shared transformer. BAGEL takes the second step: it decouples the *transformer weights too*, via the Mixture-of-Transformers design. The understanding expert and the generation expert have separate parameters, separate gradients, separate FFNs and QKV projections — they only meet in the shared self-attention. The bet is that the two tasks need to *communicate* (so they share attention) but should not be forced to *share representational machinery* (so they get separate weights).

> Decoupling is the whole game. Janus decoupled the eyes; BAGEL decoupled the eyes *and* the brain regions, then forced them to talk only through a shared attention bus.

There is a second, quieter reason BAGEL's understanding does not degrade the way you would fear. Because the understanding expert is initialized from a strong instruct LLM (Qwen2.5-7B-Instruct) and a strong vision encoder (SigLIP2-so400m), and because the generation gradients flow through a *separate* parameter set, the understanding path is not constantly being yanked around by the generation objective. The paper's own framing is that MoT "mitigates task conflict." The benchmarks bear this out: BAGEL does not just match the specialist VLMs, it edges ahead of them, which is genuinely surprising for a model that is also spending capacity on learning to paint.

### The MoT vs MoE distinction, because people conflate them

A quick disambiguation, because the acronym collision causes real confusion. A **Mixture-of-Experts (MoE)** routes tokens to experts via a *learned* router that picks the top-k FFNs per token, and the routing is a function of the token's *content*. A **Mixture-of-Transformers (MoT)**, BAGEL's design, routes tokens via a *fixed* rule based on the token's *modality* — text and understanding-image tokens always go to the understanding expert, generation-image tokens always go to the generation expert. There is no learned gate, no load-balancing loss, no router instability. The routing is hard, deterministic, and modality-keyed. This is why the parameter count works out the way it does: 14B parameters live in the model, but any given token only ever activates ~7B of them (its own expert plus the shared attention), so the per-token FLOPs are those of a 7B dense model even though the model has the capacity of a 14B one.

Spell out *exactly* what "two experts" means at the level of weight tensors, because this is where the design earns its keep and where people's mental models go wrong. Take a single transformer layer. In a vanilla decoder it has four weight groups: the query/key/value projections, the output projection, and the feed-forward network (two matrices with a nonlinearity between them), plus the layernorms. In BAGEL's MoT layer, the *attention computation* — the softmax over query-key dot products, the weighted sum of values — is shared: all tokens, text and ViT and VAE, participate in one self-attention over the whole sequence. But the *projections that feed and follow that attention are duplicated per modality*. The understanding expert has its own Q/K/V/O projections and its own FFN; the generation expert has a *separate* Q/K/V/O and a *separate* FFN. When a text token flows through the layer, it uses the understanding expert's projections to produce its query, key, and value; when a VAE token flows through, it uses the generation expert's projections. The two sets of queries, keys, and values then meet in *one* attention operation. So the experts do not attend to *different* sequences — they attend to the *same* sequence, but they project into and out of it with different weights.

Why does that distinction matter so much? Because it is the precise mechanical answer to "how do you let two tasks communicate without forcing them to share representations." The shared attention is the communication channel: a VAE token being denoised can attend to a text token's key and pull in "the user asked for a red sofa," and a text token answering a question can attend to a ViT token's value and pull in "the image shows a sofa." That cross-modal information flow is *exactly* what a pipeline of two separate models cannot do. But the separate projections and FFNs mean the text-reasoning machinery and the pixel-denoising machinery are *different parameter subspaces*, trained by different gradients, so the generation objective never overwrites the weights that make the understanding expert good at reading. You get the communication of a fused model and the specialization of a pipeline, in one layer, repeated for every layer in the stack.

This also reframes the "14B total / 7B active" number in a way the headline obscures. The 14B is not 14B of *capacity you sometimes use* — it is two full 7B-scale parameter sets that happen to interleave through a shared attention. During a pure understanding forward pass (VQA, captioning) only the understanding expert's projections and FFNs ever fire on text and ViT tokens; the generation expert's weights sit idle. During a pure text-to-image forward pass, the generation expert does the heavy lifting on the noised VAE tokens while the understanding expert processes the conditioning text. During *editing* — the interesting case — both experts are live at once, because the sequence contains text tokens (instruction), ViT tokens (what the image means), and VAE tokens (the image being produced), and each routes to its expert while all of them share the attention. That is why editing is the most expensive mode and also the one where the architecture's full machinery is engaged.

A final note on initialization, which is underappreciated. The understanding expert is not trained from scratch — it is initialized from Qwen2.5-7B-Instruct, a fully trained instruction-following LLM, and the generation expert is initialized as a *copy* of those same weights before the two diverge under their respective objectives. Starting both experts from a strong language model is a deliberate choice: it means the generation expert inherits the LLM's world knowledge and its ability to follow instructions, rather than learning image generation in a representational vacuum. This is part of why BAGEL's generation can do things like resolve "the Roman god of war" to "Mars" — the generation expert's weights began life as a language model that already knew that fact, and rectified-flow training adapted those weights to *draw* rather than erasing what they knew.

## The dual-encoder design: two ways of seeing the same image

**The senior rule of thumb: never make one representation serve two masters with opposite needs.** BAGEL encodes every image twice, and the redundancy is the point.

![Two separate model+model stacks versus one unified MoT, showing the encoder split and weight split](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-2.png)

<!-- FIGSPEC 2
kind: before-after
claim: Prior practice ran a separate VLM and a separate diffusion model; BAGEL fuses them but keeps two encoders so each task keeps its ideal representation.
caption: Before, two models with two stacks and no shared context; after, one MoT with two encoders and shared attention.
nodes:
  - id: b1 | label: "BEFORE: VLM (CLIP enc + LLM) for reading" | color: gray
  - id: b2 | label: "BEFORE: Diffusion (VAE + UNet/DiT) for drawing" | color: gray
  - id: b3 | label: "No shared context; pipe image between them" | color: amber
  - id: a1 | label: "AFTER: SigLIP2 enc -> Und expert (read)" | color: blue
  - id: a2 | label: "AFTER: FLUX VAE -> Gen expert (draw)" | color: green
  - id: a3 | label: "Shared self-attention: edit conditions on understanding" | color: amber
notes: left column = before (3 stacked, gray/amber), right column = after (3 stacked, blue/green/amber); contrast no-context vs shared-attention
-->

Look at the contrast. The "before" world — which is still how most production stacks work — runs two entirely separate models. A VLM (a CLIP/SigLIP encoder feeding an LLM) reads the image and emits text. A diffusion model (a VAE feeding a UNet or DiT) reads text and emits an image. To do something like "look at this photo and edit the sky to sunset," you have to pipe the image out of one model and into the other, and the editing model has *no access* to the understanding model's reasoning about what is in the picture. That is the architectural reason classical editing pipelines are so literal and so bad at instructions that require *knowing what they're looking at*.

BAGEL's "after" keeps the two encoders — because the representational conflict is real and unavoidable — but unifies everything downstream of them through shared attention. Concretely:

- **Understanding encoder: SigLIP2-so400m/14.** A ViT operating at a 384×384 base resolution, extended via NaViT-style native-resolution support up to 980×980, feeding a 2-layer MLP connector into the understanding expert. Its tokens are *semantic*: they answer "what is this?" This is the path used for VQA, captioning, OCR, and — critically — for *conditioning* edits on the content of an input image.
- **Generation encoder: a frozen FLUX VAE.** 8× spatial downsampling, 16 latent channels, with a 2×2 patch embedding into the generation expert. Its latents are *pixel-faithful*: they answer "exactly what does this look like?" This is the path the rectified-flow objective denoises into when producing an image, and the path that lets editing *preserve* the parts of the input you didn't ask to change.

The VAE is frozen. The understanding encoder is fine-tuned during pretraining. This asymmetry matters: a VAE's reconstruction quality is a fixed, well-understood quantity that you do not want training to drift; the understanding encoder, by contrast, benefits from adapting to BAGEL's specific token mix and instruction distribution.

### Why an input image needs both encodings at once

Here is the subtle bit that the architecture diagram hides. When BAGEL *edits* an image, the input image is encoded by **both** encoders simultaneously and both sets of tokens enter the sequence. The ViT tokens tell the generation expert *what the image means* ("this is a living room with a sofa and a window"); the VAE tokens tell it *exactly what the pixels are* so it can reproduce the untouched regions faithfully. An editing instruction like "make the sofa red" then routes through shared attention: the generation expert attends to the ViT tokens to *locate and understand* the sofa, and attends to the VAE tokens to *preserve* everything else. A single-encoder model literally cannot do this — it would have to choose between knowing what the sofa is and knowing what its pixels are. This dual-conditioning is the mechanistic reason BAGEL's "intelligent editing" score is so far ahead of classical editors, which we will see in the benchmarks.

```python
# Conceptual: how one input image becomes two token streams in BAGEL.
# (Real call sites live in modeling/bagel and inferencer.py.)
from PIL import Image

img = Image.open("living_room.jpg")

# Path A — understanding tokens (semantic, what-is-this)
vit_tokens = siglip2_encoder(vit_transform(img))        # ~ semantic ViT patches
und_tokens = mlp_connector(vit_tokens)                  # -> understanding expert

# Path B — generation tokens (pixel-faithful, exactly-this)
vae_latents = flux_vae.encode(vae_transform(img))       # 8x down, 16 channels
gen_tokens  = patch_embed_2x2(vae_latents)              # -> generation expert

# Both streams are concatenated into ONE sequence and share self-attention.
sequence = [*text_tokens, *und_tokens, *gen_tokens]      # interleaved per layout
```

The two `*_transform` functions are different on purpose — `vit_transform` does the SigLIP normalization the understanding encoder expects, `vae_transform` does the VAE's normalization. They are not interchangeable, and mixing them up is the first bug people hit when they hack on the inference code.

#### Worked example: a "describe then edit this image" turn

Let us trace one full multimodal turn end to end, because the token bookkeeping is the whole story and it is easy to wave hands at it. Suppose the user uploads a photo of a living room and says: *"What's in this room? Now make the sofa emerald green."* This is a single turn that requires understanding *and* generation, and it exercises every part of the architecture.

**Step 1 — encode the input image, twice.** The living-room photo is pushed through both encoders. SigLIP2-so400m/14 produces, say, a few hundred semantic ViT tokens (the exact count depends on resolution; at 980×980 with a /14 patch you get on the order of (980/14)² ≈ 4,900 patches before any pooling, which the connector and resolution policy reduce to a manageable token budget). The FLUX VAE, with 8× downsampling, turns the same image into a latent of spatial size H/8 × W/8 × 16 channels, which the 2×2 patch embedding folds into a sequence of generation tokens. Both sequences exist simultaneously and carry *different information about the same pixels*: the ViT tokens encode "sofa, window, coffee table, wood floor, afternoon light," the VAE latents encode "exactly these RGB values arranged exactly this way."

**Step 2 — answer the question (understanding path).** The sequence so far is `[text: "What's in this room?", ViT tokens of the photo]`. Note the VAE tokens of the *input* image are *clean* (not noised — they are a real image, not something being generated), so they sit in context too, available to be attended to. The understanding expert decodes a text answer autoregressively: "This is a living room with a gray three-seat sofa, a window on the left, a wooden coffee table, and hardwood floors." Each generated text token attends causally to everything before it — the question, the ViT tokens, the clean VAE tokens — and routes through the understanding expert's projections and FFN. No diffusion has run yet; this is pure LLM decoding over a multimodal context.

**Step 3 — set up the edit (the routing handoff).** The instruction "Now make the sofa emerald green" is appended as more text tokens. The model now needs to *generate* an image, so it appends a block of *noised* VAE tokens — pure Gaussian noise at the target resolution — as scratch space. The sequence is now roughly: `[question, input-ViT, input-VAE(clean), answer-text, edit-instruction, output-VAE(noised)]`. This is the moment the generalized causal attention mask earns its name. The noised output tokens can attend *backward* to everything clean: the input image's ViT tokens (to know what a sofa is and where it is), the input image's clean VAE tokens (to preserve the window, floor, and table pixel-for-pixel), and the instruction text (to know "emerald green, sofa"). But nothing in the future can attend to these noised tokens — they are private scratch space.

**Step 4 — denoise (generation path).** Over 50 rectified-flow steps, the generation expert repeatedly predicts a velocity field on the noised output-VAE tokens, nudging them from noise toward a clean latent. At every step, those tokens attend through the shared self-attention to the input ViT tokens (which localize and identify the sofa) and the input clean VAE tokens (which carry the exact pixels to keep). The result is a latent that, when decoded by the VAE, is the *same room* with a *green sofa* — the rest of the scene preserved because the model literally had the original pixels in context to copy from. Classical inpainting needs a hand-drawn mask to know what to keep; BAGEL gets the mask for free from the dual encoding plus shared attention.

**Step 5 — decode and return.** The denoised VAE latent goes through the FLUX VAE decoder back to a 1024×1024 RGB image, and the turn returns both the text answer (from step 2) and the edited image (from step 4). Two output modalities, one forward context, one model. The reason this is hard to replicate with a two-model pipeline is step 4's dependence on step 2's *understanding*: the edit succeeded because the generation expert could attend to a representation that knew "the sofa is the gray three-seat object on the right," and that representation came from the understanding side of the same model.

## How autoregression and diffusion share one forward pass

This is the cleverest piece of plumbing in the paper, and it lives entirely in the attention mask. BAGEL has to do two things that seem mutually exclusive in a single transformer: **causal, left-to-right autoregression** for text (each token sees only the past), and **bidirectional denoising** for an image being generated (every patch of the image-in-progress should see every other patch). The trick is a *generalized causal attention* that is causal at the granularity of *modality segments* but flexible within an image.

![Inference paths: text token AR path versus image rectified-flow path through the two experts](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-3.png)

<!-- FIGSPEC 3
kind: graph
claim: A single forward pass routes text tokens through autoregressive decoding and image tokens through iterative rectified-flow denoising in the generation expert.
caption: Two inference paths share weights and attention but use different output heads and different sampling loops.
nodes:
  - id: ctx | label: "Shared context (prompt + prior images)" | color: gray
  - id: attn | label: "Shared self-attention (all tokens see all)" | color: amber
  - id: undx | label: "Understanding expert (text+ViT)" | color: blue
  - id: genx | label: "Generation expert (VAE)" | color: green
  - id: kv | label: "KV cache (reused across steps)" | color: gray
  - id: txt | label: "Text head: next-token (AR, causal)" | color: blue
  - id: flow | label: "Flow head: velocity v_theta, 50 RF steps" | color: green
  - id: outt | label: "Text output" | color: gray
  - id: outi | label: "1024x1024 image (CFG 4-8x)" | color: gray
edges:
  - ctx -> attn
  - attn -> undx
  - attn -> genx
  - attn -> kv
  - undx -> txt | label: "text tokens"
  - genx -> flow | label: "VAE tokens"
  - txt -> outt
  - flow -> outi
notes: branching graph; the layer below attn has 3 nodes (undx, genx, kv); amber attn is the shared hub; text path is AR, image path is rectified flow
-->

Trace the two paths. Everything starts from a **shared context**: the prompt, any prior images, the accumulated KV cache. That context passes through the shared self-attention and the two experts. Then it forks at the output:

- **Text path (blue).** Text tokens are produced one at a time, autoregressively, by the understanding expert's text head, under a causal mask — token *t* attends to tokens *< t* only. This is ordinary LLM decoding. When BAGEL is "thinking" (chain-of-thought before generating an image), this is the path running.
- **Image path (green).** When generating an image, BAGEL appends a block of *noised* VAE tokens to the sequence and the generation expert's flow head predicts a velocity field that denoises them over (typically) 50 rectified-flow steps. Within the image block, attention is bidirectional — every patch sees every other patch, exactly as a diffusion transformer needs.

The mask that makes this legal is the *generalized causal attention*: **clean** VAE and ViT tokens of *preceding* images can be attended to by later tokens (so a later edit can condition on an earlier image), but the **noised** VAE tokens of the image currently being generated are *masked from future outputs* — they are scratch space that exists only to be denoised, and letting future tokens attend to transient noise would poison the context. The rule, in one sentence: *the past is clean and visible; the in-progress image is noisy and private.*

### The unified loss

The two objectives are summed with fixed weights. For text, ordinary cross-entropy next-token prediction; for images, the rectified-flow MSE on the predicted velocity. Writing the flow part out: given a clean VAE latent $x_1$ and Gaussian noise $x_0 \sim \mathcal{N}(0, I)$, rectified flow interpolates linearly $x_t = (1-t)\,x_0 + t\,x_1$ for $t \in [0,1]$, and the model $v_\theta$ is trained to predict the constant velocity $x_1 - x_0$:

$$\mathcal{L}_{\text{gen}} = \mathbb{E}_{t,\,x_0,\,x_1}\big[\,\lVert v_\theta(x_t, t, c) - (x_1 - x_0)\rVert_2^2\,\big]$$

where $c$ is the conditioning context (text and any prior-image tokens) reaching the generation expert through shared attention. The total objective is

$$\mathcal{L} = 0.25\cdot\mathcal{L}_{\text{text}} + 1.0\cdot\mathcal{L}_{\text{gen}}$$

The weights (0.25 on text cross-entropy, 1.0 on the flow MSE) are not arbitrary — they balance the *magnitudes* of two losses on very different scales. Cross-entropy over a 150K vocab and an MSE over 16-channel latents do not naturally live in the same range; the 0.25/1.0 split is the lever that keeps the generation gradient from being drowned out, and it is the kind of number you only find by sweeping. The straight-line interpolant is why rectified flow can sample in relatively few steps (50 here, versus the hundreds older DDPM schedules wanted): the model is learning to follow a *straight* path from noise to data, not a curved one.

#### What rectified flow actually is, and why MSE-on-velocity

It is worth slowing down on rectified flow, because if you came from the diffusion world you might expect noise-prediction (ε-prediction) and if you came from the LLM world you might not know what any of this means. Here is the intuition, then the mechanics.

Think of generation as transport: you have a cloud of pure noise and you want to move it, sample by sample, to the cloud of real images. Classical diffusion learns a *curved* path through a stochastic differential equation — at each step you predict the noise to subtract, and the trajectory from noise to data wiggles. Rectified flow asks a sharper question: what if the path from each noise sample to its corresponding data sample were a *straight line*? If the model could learn the velocity field of straight-line transport, you could integrate that field in a handful of large steps instead of hundreds of tiny ones, because a straight line needs few waypoints. That is the whole pitch: straighter paths, fewer function evaluations, faster sampling.

Mechanically, the construction is the one in the equation above. Pair a real latent $x_1$ with a noise sample $x_0$ and define the interpolation $x_t = (1-t)x_0 + t x_1$. As $t$ goes from 0 to 1, $x_t$ slides linearly from noise to data. Differentiate: $\frac{dx_t}{dt} = x_1 - x_0$, a *constant*. So the velocity along this straight path does not depend on where you are on it — it is the same vector $x_1 - x_0$ everywhere. The model $v_\theta(x_t, t, c)$ is trained to predict that constant velocity at a randomly sampled time $t$, conditioned on context $c$, and the loss is just the mean-squared error between predicted and true velocity. That is why it is *MSE on velocity*: the target is a vector (the displacement from noise to data), the prediction is a vector, and MSE is the natural regression loss for matching vectors. At inference you start from noise $x_0$, evaluate $v_\theta$, take a step in that direction, re-evaluate, and repeat 50 times — numerically integrating the learned velocity field from $t=0$ to $t=1$.

Why does BAGEL choose this over ε-prediction diffusion? Three reasons that matter for a *unified* model specifically. First, sampling efficiency: 50 steps is already on the edge of acceptable latency for a model that also has to do autoregressive text decoding, and rectified flow's straight paths make 50 steps produce quality that curved-path diffusion would need many more steps to match. Second, it is the same objective FLUX and SD3 use, so the FLUX VAE that BAGEL borrows is a natural fit and the generation expert lands in a well-understood regime. Third — and this is the subtle one — velocity prediction plays nicely with the *conditioning through attention*. Because $c$ (the text and prior-image tokens) enters via the shared self-attention rather than through a separate cross-attention module bolted on, the velocity field is conditioned on a representation that the *understanding* side also shaped. The flow head is predicting "which way to move these pixels" using a context that includes the model's own understanding of the prompt. That is the mechanical root of the chain-of-thought-helps-generation result we will get to.

#### Generalized causal attention, restated as a mask

One more pass on the attention mask, because it is the single most load-bearing design decision and the equations hide it. Picture the attention matrix for a turn that contains, in order: prompt text, an input image (ViT tokens, then clean VAE tokens), more text, and a to-be-generated image (noised VAE tokens). The mask is a set of allow/deny rules on "can token $i$ attend to token $j$":

- **Text-to-everything-before:** text tokens attend causally — token $t$ sees all tokens $< t$, including clean image tokens. This is standard LLM behavior.
- **Within a generated image, bidirectional:** the noised VAE tokens of the image currently being produced attend to *each other* with no causal restriction. A diffusion transformer needs every patch to see every patch; the mask permits exactly that, but only *within* the current image block.
- **Generated image to clean past:** the noised tokens attend backward to all clean tokens — prior text, prior ViT, prior clean VAE — so generation conditions on the full context. This is what lets an edit see the input image.
- **Future masked from noise:** nothing attends *forward* into the noised tokens. The in-progress image is transient scratch space; if a later text token could attend to half-denoised noise, the context would be poisoned with garbage that changes every denoising step.

The slogan from before — *the past is clean and visible; the in-progress image is noisy and private* — is exactly this mask. And notice it generalizes "causal" from the token level to the *segment* level: text is causal token-by-token, but an image is one bidirectional segment that is causally placed relative to everything else. That generalization is what lets one transformer host both autoregression and diffusion without two separate attention implementations.

> Autoregression and diffusion are not rival religions. They are two output heads on the same body, separated by an attention mask and a loss weight.

## The four-stage training recipe, and where the data comes from

**The senior rule of thumb: capability follows data composition, and data composition is the actual model design.** BAGEL's architecture is elegant, but the paper is at pains to show that *what* it can do is governed by *what it was fed and when*. Four stages, escalating in resolution and quality.

| Stage | Steps | Tokens | What's trainable | Emphasis |
|---|---|---|---|---|
| 1. Alignment | 5K | — | Connector only (enc + LLM frozen) | Wire SigLIP2 → LLM |
| 2. Pre-training | 200K | 2.5T | Everything | Breadth; 60–80% generation samples |
| 3. Continued training | 100K | 2.6T | Everything | Higher resolution, more interleaved |
| 4. Supervised fine-tuning | 15K | 72.7B | Everything | High-quality curated subset |

Stage 1 is the cheap alignment warm-up familiar from any VLM: freeze the strong pretrained pieces (SigLIP2, Qwen2.5-7B) and train *only* the MLP connector so the vision tokens land in a space the LLM can read. At 5K steps and no expensive full-model gradients, this stage costs almost nothing relative to the rest — but skipping it would mean the early pretraining steps waste capacity fighting a misaligned vision-to-language projection. It is the cheapest insurance in the recipe.

Stages 2 and 3 are where the real money is spent — roughly 5.1T tokens combined — and they unfreeze everything. The split between them is *resolution and interleaving*, not a hard task boundary. Pre-training (Stage 2, 2.5T tokens) runs at lower resolution and establishes breadth: it is where basic understanding and basic generation get most of their competence, because those capabilities saturate early on the emergence curve and do not need the highest resolution to learn. Continued training (Stage 3, 2.6T tokens) raises the resolution and *increases the fraction of interleaved data* — the video and web sources that teach temporal, 3D, and document-flow structure. The notable choice across both stages is that generation samples *dominate* (60–80% of the mix), because generation is the harder, lower-data-efficiency task and the understanding side arrives pre-warmed from the LLM and encoder. If you weighted the mix toward understanding, you would be spending tokens on a capability that is already nearly saturated while starving the one that needs the data.

Stage 4 (SFT, 72.7B tokens — note the units: *billions*, two orders of magnitude smaller than the pretraining stages) is a small, high-quality polish on a curated subset. This is where instruction-following sharpens and where the model learns to format its outputs the way downstream users expect. The lesson embedded in the stage sizes: ~99% of the tokens go into pretraining the raw capabilities, and a tiny final fraction shapes how those capabilities are *expressed*. SFT cannot teach a capability the pretraining did not instill; it can only refine and surface what is already there. This is why the emergence thresholds (which we get to next) are all measured against *pretraining* tokens — that is where capabilities are born.

![Unified training data mix grid by source with token counts and modality](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-4.png)

<!-- FIGSPEC 4
kind: grid
claim: BAGEL's 5.5T-token corpus is dominated by text-to-image and interleaved data, not by plain image-text pairs.
caption: The data mix is mostly generation and interleaved sources; pure understanding pairs are a minority.
nodes:
  - id: r1c1 | row: 1 | col: 1 | label: "Text-only 0.4T (language prior)" | color: gray
  - id: r1c2 | row: 1 | col: 2 | label: "Image->Text 0.5T (VQA/caption)" | color: blue
  - id: r2c1 | row: 2 | col: 1 | label: "Text->Image 2.6T (largest block)" | color: green
  - id: r2c2 | row: 2 | col: 2 | label: "Interleaved understand 0.5T" | color: blue
  - id: r3c1 | row: 3 | col: 1 | label: "Interleaved video gen 0.7T" | color: green
  - id: r3c2 | row: 3 | col: 2 | label: "Interleaved web gen 0.4T" | color: green
notes: 3 rows x 2 cols; green = generation-side sources, blue = understanding-side, gray = text; total ~5.5T tokens
-->

The composition is the design. Read it as three families:

- **Understanding-side (blue): ~1.0T tokens.** Plain text (0.4T, to keep the language prior sharp), image→text pairs (0.5T, the VQA/caption data), and interleaved understanding (0.5T). This is *less* data than a dedicated VLM gets, yet BAGEL beats the dedicated VLMs — evidence that the generation data is *helping* understanding, not just coexisting with it.
- **Generation-side (green): the bulk.** Text→image pairs are the single largest block at 2.6T tokens. On top of that sit two interleaved generation sources that are the secret to the emergent abilities: **interleaved video generation (0.7T)**, sourced from Koala36M and MVImgNet2.0, which teaches temporal and 3D consistency (this is where future-frame prediction and 3D rotation come from), and **interleaved web generation (0.4T)**, built on OmniCorpus with a two-stage LLM filtering and a caption-first construction strategy, which teaches the model to generate images *in the flow of a document* — the substrate for sequential, reasoned generation.

That video and web interleaved data is the unsung hero. A model trained only on text→image pairs learns to make pretty pictures from prompts. A model trained additionally on *video frames in sequence* learns that images have a *next frame* — and that is, mechanically, what "future-frame prediction" and "world navigation" are. The capability is not bolted on; it falls out of the data once the model has enough capacity to absorb it.

Three details about the data construction are worth pulling out, because they explain *why* the interleaved sources produce the emergent abilities rather than just more noise. First, the **video sources** — Koala36M and MVImgNet2.0 — are not random clips. Koala36M is a large, temporally coherent video corpus; MVImgNet2.0 is multi-view imagery, meaning multiple camera angles of the *same object or scene*. Training on the former teaches "what comes next in time" (future-frame prediction); training on the latter teaches "what this looks like from another angle" (3D rotation and viewpoint change). The two emergent abilities map almost one-to-one onto the two video data types. This is not a coincidence the paper stumbled into; it is a designed correspondence between data and capability.

Second, the **web data is built on OmniCorpus with a two-stage LLM filtering and a caption-first construction strategy**. The caption-first part matters: rather than taking web images with whatever messy alt-text they came with, the pipeline generates clean captions and structures the interleaved text-image documents so that the textual context *precedes and explains* each image. This is what teaches BAGEL to generate an image *given a paragraph of surrounding context*, which is the substrate for sequential, reasoned, context-aware generation. A model trained on isolated (caption, image) pairs has no notion of "the image that should go here given everything written so far"; a model trained on caption-first interleaved web documents does.

Third, the **dominance of T2I (2.6T tokens) over understanding data (~1.0T combined) is a deliberate inversion** of how a VLM would budget. A pure VLM pours its tokens into image-text understanding pairs. BAGEL spends more than twice as many tokens on generation as on understanding, and *still* beats the VLMs on understanding. The only consistent explanation is that the generation objective is teaching the shared attention something useful about images — forcing the model to *reconstruct* pixels apparently sharpens the representations that the understanding side reads, a kind of free auxiliary task. This is one of the quiet, underexplored results in the paper: generation pretraining is a positive transfer signal for understanding, not a competing drain on it.

## The emergence story: capabilities appear at thresholds, not gradually

This is the paper's intellectual core and the reason it is worth reading rather than skimming. BAGEL's abilities do not improve smoothly and in lockstep as you add data. They **saturate at different token counts**, in a fixed order, simple before complex. The paper tracks each capability's performance against cumulative pretraining tokens and reports where each one reaches ~85% of its eventual peak.

![Timeline of capability emergence as pretraining tokens scale from 0.18T to 3.61T](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-5.png)

<!-- FIGSPEC 5
kind: timeline
claim: Four capability tiers saturate at increasing token thresholds, with reasoning-heavy editing emerging 20x later than understanding.
caption: Understanding saturates first at 0.18T tokens; intelligent editing only emerges near 3.61T — a staged, compositional curve.
nodes:
  - id: t1 | label: "0.18T: understanding saturates (~85% peak)" | color: blue
  - id: t2 | label: "0.68T: basic T2I generation saturates" | color: blue
  - id: t3 | label: "2.64T: classical editing saturates" | color: amber
  - id: t4 | label: "3.61T: intelligent editing / world ops emerge" | color: red
edges:
  - t1 -> t2 | label: "+0.5T"
  - t2 -> t3 | label: "+2.0T"
  - t3 -> t4 | label: "+1.0T"
notes: vertical timeline, render tall; 4 milestones ascending in token count; blue (cheap) -> amber (mid) -> red (emergent) escalation
-->

The four thresholds, in order:

1. **Understanding — ~0.18T tokens.** The cheapest. VQA, captioning, and OCR-style abilities reach their plateau almost immediately, because the understanding expert inherits a fully trained LLM and vision encoder. By the time BAGEL has seen a fifth of a trillion tokens, it already reads images about as well as it ever will.
2. **Basic generation — ~0.68T tokens.** Text→image quality saturates next, at roughly 4× the understanding threshold. Making a coherent, prompt-faithful image is harder than reading one, but it is still a "single-step" capability — map text to pixels.
3. **Classical editing — ~2.64T tokens.** This is the first big jump, almost 4× again. Editing requires the dual-conditioning we discussed: simultaneously preserving the input (VAE tokens) and modifying it per instruction (ViT tokens + text). The model cannot do this until both encoders' tokens are well-integrated through attention, which takes a lot more data.
4. **Intelligent editing and world operations — ~3.61T tokens.** The hardest tier, and the one the paper calls genuinely emergent: free-form manipulation, future-frame prediction, 3D rotation, world navigation. These require *reasoning about* the image content and then generating accordingly — they sit at the intersection of understanding and generation, and they only light up after the model has seen ~20× the data needed for plain understanding.

The shape of this curve is the argument. If unification were merely "two models stapled together," you would expect each capability to track its own data linearly and independently. Instead the capabilities emerge *compositionally*: each tier depends on the tiers below it being solid, and the most interesting abilities are the ones that *combine* understanding and generation — which is exactly what only a genuinely unified model can learn. The emergence is evidence that the unification is real, not cosmetic.

#### Why the order is fixed, not arbitrary

The ordering — understanding, then basic generation, then classical editing, then intelligent editing — is not just "easy things before hard things" in some vague sense. There is a *dependency structure* that forces it, and naming the dependencies makes the curve predictable rather than mysterious.

Understanding comes first because it inherits the most. The understanding expert starts from a fully trained LLM and a fully trained vision encoder; the only thing pretraining has to learn is how to *connect* them and adapt them to the token mix. That is a small delta on top of two strong priors, so it saturates at 0.18T tokens — a rounding error in the full budget. Basic generation comes second because it depends on nothing but text-to-pixel mapping, and while that mapping is genuinely hard to learn from scratch, it is a *single-step* skill: given a prompt, produce a coherent image. No conditioning on an input image, no preservation, no reasoning. It saturates at 0.68T.

Classical editing is the first capability that *requires two things to be true at once*. To edit, the model must already be able to (a) understand an input image well enough to localize what the instruction refers to, and (b) generate a coherent image. Editing is not a third independent skill — it is the *composition* of understanding and generation through the shared attention, plus the additional trick of preserving everything the instruction did not mention. You cannot learn the composition before the components are solid, which is exactly why classical editing saturates at 2.64T, well after both prerequisites. The 4× jump from basic generation to classical editing is the cost of learning to *combine* rather than to *do*.

Intelligent editing — the emergent tier at 3.61T — is the composition of *all* the lower tiers plus a reasoning step. "Make this look like the year 2050" requires understanding the scene, reasoning about what futuristic means for *this specific* scene, and then generating accordingly while preserving structure. It is editing conditioned on a chain of inference about content. This sits at the very top because it depends on everything below it being reliable *and* on the model having enough capacity left over to reason about intent rather than execute a literal instruction. The 20× gap between understanding (0.18T) and intelligent editing (3.61T) is the gap between "read an image" and "reason about an image and then redraw it" — and the fact that the gap is bridged *at all*, within a single model, is the paper's central surprise.

The practical reading of this curve, if you ever train something like BAGEL: do not expect the headline emergent abilities until you are deep into the token budget, and do not panic when they are absent at 1T tokens. They are not missing because the architecture is wrong; they are missing because the compositional prerequisites have not saturated yet. The curve is a roadmap, not a warning.

### Chain-of-thought for generation: thinking before drawing

Because understanding and generation live in the same model, BAGEL can *think in text before it draws* — and this measurably helps. On WISE (a world-knowledge text-to-image benchmark), BAGEL scores 0.52 without reasoning and **0.70 with chain-of-thought**, a +0.18 jump that takes it from behind MetaQuery (0.55) to well ahead. The same trick lifts IntelligentBench from 44.0 to 55.3 — a +11.3 swing on reasoning-heavy editing. These are not marginal nudges; on WISE the CoT mode is the difference between mid-pack and best-in-class.

The mechanism is worth dwelling on because it is the clearest demonstration of *why unification matters at all*. Consider the prompt "the animal that says moo, in the style of Van Gogh." A pure diffusion model receives the text embedding of that whole string and tries to map it directly to pixels. It has no faculty for *resolving the riddle* — "the animal that says moo" is not a visual concept it can render, it is a fact it must look up. The model will often produce something animal-shaped and Van-Gogh-styled but wrong, because the hard part of the prompt was a knowledge hop, not a rendering problem. Now give the same prompt to BAGEL with `think=True`. The understanding expert first decodes text: "The animal that says moo is a cow. I should depict a cow in Van Gogh's post-impressionist style with bold swirling brushstrokes." *That reasoning becomes part of the context.* When the generation expert then denoises, it conditions — through shared self-attention — on a context that now explicitly contains the word "cow" and a description of the target style. The riddle has been resolved into a concrete visual specification before a single denoising step runs.

This is why it is a genuine *unification dividend* and not something you could bolt onto a pipeline. In a two-model pipeline you could, in principle, run an LLM to rewrite the prompt and then feed the rewrite to a separate diffusion model — and indeed "prompt rewriting" is a known trick that lifts GenEval from 0.82 to 0.88. But prompt rewriting is a *lossy text-to-text bottleneck*: whatever the LLM does not write down explicitly is lost before the diffusion model sees anything. BAGEL's CoT is different in kind, because the reasoning tokens stay *in the same attention context* as the generation. The flow head does not just see a rewritten string; it attends to the full hidden states of the reasoning trace, including everything the model "knew" but did not verbalize. The information channel from reasoning to rendering is the model's own residual stream, not a re-serialized prompt. That is a strictly richer conditioning signal than any pipeline can offer.

There is a cost, and it is honest to name it. CoT mode runs the autoregressive text path *before* the 50-step diffusion path, so a thinking generation is slower — you pay for the reasoning tokens on top of the denoising. For prompts that are literal and visual ("a red cube on a blue table"), the reasoning adds latency for no benefit, and you should leave `think=False`. The discipline is to reserve CoT for prompts with an indirect referent (riddles, world-knowledge hops), a compositional constraint that needs planning ("show the scene as it would look if the river had flooded"), or an intelligent edit where intent must be inferred. On exactly those prompts, the WISE 0.52→0.70 and IntelligentBench 44.0→55.3 numbers are what you feel. On everything else, it is wasted compute.

One more subtlety: the reasoning trace is *visible*. Because the text is decoded into the output before the image, you can read what the model decided to draw, which makes failures debuggable in a way that pure diffusion never is. If a CoT generation comes out wrong, you can often see *in the reasoning* where it went off — it resolved the riddle incorrectly, or planned a composition you did not want — and fix the prompt accordingly. A black-box diffusion model gives you only the wrong image and no explanation. This observability is a quiet but real operational advantage of the unified design.

## Versus the prior unified models: why MoT wins

**The senior rule of thumb: when a new architecture beats its predecessors, find the one design axis that changed and confirm the win tracks it.** For BAGEL that axis is *how much it decouples the two tasks*, and the win tracks it cleanly. Let us walk the lineage and say precisely why each predecessor left performance on the table.

**Chameleon (early fusion, single tokenizer).** Chameleon is the purest form of unification: every image is quantized into discrete tokens by a single VQ tokenizer, and the model autoregresses over a joint vocabulary of text and image tokens. It is elegant — one sequence, one objective, one tokenizer — and it pays for that elegance twice. First, *quantization caps fidelity*: a VQ codebook has finite entries, so the image representation is lossy in a way that bounds generation quality below what continuous-latent diffusion achieves; Chameleon's text-to-image is visibly weaker than a dedicated diffusion model. Second, *one tokenizer cannot serve both masters*: the same discrete tokens that are too coarse for high-fidelity generation are also forced to carry semantic content for understanding, and the compromise hurts both ends. Chameleon proves unification is *possible* and simultaneously proves that the naive version taxes you on both axes. BAGEL keeps Chameleon's "one sequence, shared attention" spirit but throws out the single tokenizer entirely.

**Transfusion (continuous latents, shared backbone, diffusion inside the transformer).** Transfusion is a real step forward: it keeps text autoregressive but generates images by running *diffusion on continuous latents inside the same transformer*, avoiding Chameleon's quantization bottleneck. This is much closer to BAGEL — continuous latents, diffusion-style generation, one transformer. The remaining tax is *shared weights*. In Transfusion, the same FFN and attention projections process both the text-reasoning tokens and the image-denoising tokens, which means the two objectives' gradients land on the same parameters and *interfere*. The model has to find weights that are simultaneously good at language modeling and good at velocity prediction, and those are different jobs. BAGEL's MoT is precisely the fix: same continuous-latent, same in-transformer diffusion, but *separate weights per modality* so the gradients never collide. If you wanted a one-line summary of BAGEL's contribution over Transfusion, it is "add the second expert."

**Show-o (discrete tokens, masked + autoregressive hybrid).** Show-o tries to get the best of both worlds with a hybrid objective — masked-token prediction for images, autoregression for text — but it stays in the *discrete* token regime for images, which inherits Chameleon's fidelity ceiling. The hybrid objective is clever engineering, but it is solving the wrong problem: the bottleneck was never the prediction order, it was the discreteness of the image representation. BAGEL sidesteps the entire question by using continuous VAE latents and rectified flow, so there is no masking schedule to tune and no codebook to bound quality.

**Janus / Janus-Pro (decoupled encoders, shared transformer).** This is the most important comparison, because Janus is the closest prior design and the head-to-head numbers are stark. Janus made the *crucial* observation that understanding and generation need different *encoders* — it uses a semantic encoder for understanding and a separate generation path — which is exactly half of BAGEL's insight. But Janus keeps a *single shared transformer* downstream of the encoders, and generates images autoregressively over discrete tokens. So Janus decoupled the eyes but not the brain, and stayed discrete on the generation side. The result, on the benchmarks: Janus-Pro-7B scores 41.0 on MMMU and 50.1 on MM-Vet, where BAGEL scores 55.3 and 67.2 — gaps of +14.3 and +17.1. Those are enormous margins between two models at the same 7B active scale, and the *only* major architectural differences are (a) BAGEL decouples the transformer weights too (MoT), and (b) BAGEL uses continuous-latent rectified flow instead of discrete autoregressive generation. The benchmark gap is the price Janus pays for sharing the transformer.

Here is the lineage as a scorecard of *how far each model pushed decoupling*, which is the axis that predicts the win:

| Model | Decouple encoders? | Decouple transformer weights? | Continuous gen latents? | Understanding parity with specialists? |
|---|---|---|---|---|
| Chameleon | No (one VQ tokenizer) | No | No (discrete) | No |
| Show-o | No (discrete) | No | No (discrete) | No |
| Transfusion | Partial | No (shared) | Yes | Partial |
| Janus-Pro | **Yes** | No (shared) | No (discrete) | No (MMMU 41.0) |
| **BAGEL** | **Yes** | **Yes (MoT)** | **Yes (rectified flow)** | **Yes (MMMU 55.3)** |

Read top to bottom and the pattern is unmistakable: every step toward more decoupling and more continuous generation buys more capability, and BAGEL is the first to take *both* steps at once. The thesis of the whole paper, reduced to this table, is that the unification tax was never fundamental — it was the accumulated cost of designs that had not yet decoupled enough. MoT is the architecture that finally decouples enough to make understanding parity and competitive generation coexist.

It is worth being fair to the predecessors: each was a necessary rung. Chameleon proved unification could be trained at all; Transfusion proved continuous in-transformer diffusion worked; Janus proved decoupled encoders mattered. BAGEL is not a repudiation of that lineage, it is its synthesis — it takes the one correct idea from each and combines them, then scales the result on enough interleaved data to make the emergent abilities appear. That is the normal shape of progress, and it is why reading the predecessors is the fastest way to understand *why* BAGEL is built the way it is.

## Reading the benchmarks honestly

Let us go through the numbers, including where BAGEL loses. House rule: a benchmark table that only shows wins is marketing, not engineering.

### Understanding: it beats the specialists

![Understanding benchmark grid comparing BAGEL to Qwen2.5-VL, InternVL2.5, Janus-Pro across MMBench/MMMU/MM-Vet/MathVista](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-6.png)

<!-- FIGSPEC 6
kind: grid
claim: BAGEL matches or beats specialist VLMs Qwen2.5-VL and InternVL2.5 on understanding, and crushes the prior unified model Janus-Pro by 14-17 points.
caption: A unified model that out-reads the read-only specialists; the one loss is MMMU to Qwen2.5-VL.
nodes:
  - id: r1c1 | row: 1 | col: 1 | label: "MMBench BAGEL 85.0 (win)" | color: green
  - id: r1c2 | row: 1 | col: 2 | label: "MMBench Qwen 83.5 / IntVL 84.6" | color: gray
  - id: r2c1 | row: 2 | col: 1 | label: "MMMU BAGEL 55.3 (loss)" | color: amber
  - id: r2c2 | row: 2 | col: 2 | label: "MMMU Qwen 58.6 / IntVL 56.0" | color: gray
  - id: r3c1 | row: 3 | col: 1 | label: "MM-Vet BAGEL 67.2 (win)" | color: green
  - id: r3c2 | row: 3 | col: 2 | label: "MM-Vet Qwen 67.1 / Janus 50.1" | color: lavender
  - id: r4c1 | row: 4 | col: 1 | label: "MathVista BAGEL 73.1 (win)" | color: green
  - id: r4c2 | row: 4 | col: 2 | label: "MathVista Qwen 68.2 / IntVL 64.4" | color: gray
notes: 4 rows x 2 cols; left col = BAGEL result (green win, amber loss), right col = specialist baselines; lavender = prior unified Janus-Pro
-->

The headline numbers, with the caveats:

| Benchmark | BAGEL | Qwen2.5-VL-7B | InternVL2.5-8B | Janus-Pro-7B |
|---|---|---|---|---|
| MME (perception) | 2388 | — | — | — |
| MMBench | **85.0** | 83.5 | 84.6 | 79.2 |
| MMMU | 55.3 | **58.6** | 56.0 | 41.0 |
| MM-Vet | **67.2** | 67.1 | 62.8 | 50.1 |
| MathVista | **73.1** | 68.2 | 64.4 | — |

Read this carefully. BAGEL wins MMBench, MM-Vet, and MathVista, and is *close* on MMMU but actually **loses MMMU to Qwen2.5-VL** (55.3 vs 58.6) and trails InternVL2.5 there too. MMMU is the hardest, most reasoning-heavy understanding benchmark — college-level multimodal questions — and it is the one place where the dedicated VLMs, which spend *all* their capacity on understanding, still have an edge. That is honest and expected: if you only care about hard visual reasoning and nothing else, a pure VLM is still marginally better. But the gap is small, and BAGEL leads on the broader perception and math benchmarks.

The more dramatic comparison is against **Janus-Pro-7B**, the prior best *unified* model. BAGEL beats it by **+14.3 on MMMU** (55.3 vs 41.0) and **+17.1 on MM-Vet** (67.2 vs 50.1). Those are not incremental margins; they are the difference between "a unified model you tolerate" and "a unified model that out-reads specialists." This is the single most important result in the paper for the unification thesis: decoupling the *weights* (MoT), not just the encoders (Janus), closes essentially the entire understanding gap.

Now go benchmark by benchmark, because each one measures a different thing and the pattern of wins and losses is informative:

- **MME (2388).** MME is a perception-and-cognition benchmark scored on a 2800-point scale across many yes/no subtasks (existence, count, position, OCR, commonsense). A score of 2388 is in the strong-VLM range and confirms BAGEL's *low-level perception* is intact — it sees objects, counts them, reads text in images, and localizes. This is the floor capability, and BAGEL clears it. The number alone does not differentiate it from the field, but a *weak* MME would have been a red flag that the generation training had damaged basic seeing; it did not.

- **MMBench (85.0, the win).** MMBench is a broad, carefully balanced multiple-choice benchmark covering perception and reasoning across twenty-odd ability dimensions. BAGEL's 85.0 edges Qwen2.5-VL-7B (83.5) and InternVL2.5-8B (84.6). The margin is small but the *direction* is the story: a model splitting its capacity between reading and drawing beats two models that spend all their capacity reading. MMBench is the benchmark most representative of "general visual competence," so winning it is the cleanest evidence that unification did not cost general understanding.

- **MMMU (55.3, the loss).** MMMU is college-exam-level multimodal reasoning — physics diagrams, chemistry structures, economics charts — and it is the *one* understanding benchmark where BAGEL loses, to Qwen2.5-VL (58.6) and InternVL2.5 (56.0). Be honest about why: MMMU rewards deep domain reasoning over visual inputs, and that is exactly the capacity a pure VLM can fully dedicate to the task while BAGEL spends some on generation. The 3.3-point gap to Qwen is real but narrow, and it is the *expected* shape of the tradeoff. If your product is a domain-expert tutor over textbook figures and nothing else, this gap is your reason to consider a specialist. For everyone else it is a rounding error against the breadth BAGEL buys.

- **MM-Vet (67.2, the win) and MathVista (73.1, the win).** MM-Vet measures integrated capabilities — recognition, OCR, knowledge, spatial reasoning, language generation — in open-ended answers, and BAGEL's 67.2 beats Qwen (67.1, barely) and InternVL (62.8, clearly). MathVista tests visual mathematical reasoning, and BAGEL's 73.1 beats Qwen (68.2) by ~5 points and InternVL (64.4) by ~9. The MathVista margin is the most interesting in the table: visual math is a *reasoning-heavy* task, and BAGEL winning it by a clear margin while losing the *other* reasoning-heavy task (MMMU) suggests the two benchmarks stress different things — MathVista rewards careful visual parsing of equations and figures, which BAGEL's strong perception handles, while MMMU rewards encyclopedic domain knowledge, which the dedicated VLMs' undiluted capacity handles better.

The composite read: BAGEL is strongest on broad perception (MMBench), integrated open-ended tasks (MM-Vet), and visual math (MathVista), and is competitive-but-behind only on the single hardest domain-knowledge benchmark (MMMU). That is a profile any product team can plan around, and it is *better* than the unification-skeptic's prior, which expected losses across the board.

### Generation and editing: competitive with the specialists, dominant on intelligent editing

![Generation and editing benchmark matrix vs SD3, FLUX, Janus-Pro, Step1X-Edit](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-7.png)

<!-- FIGSPEC 7
kind: matrix
claim: BAGEL leads GenEval against SD3 and FLUX, ties classical editing, and is 3x ahead on reasoning-heavy IntelligentBench.
caption: Competitive T2I, parity on classical edits, and a 30-point lead on edits that require understanding.
rows:
  - GenEval (T2I)
  - WISE (knowledge)
  - GEdit-EN (overall)
  - IntelligentBench
cols:
  - BAGEL
  - SD3-Medium
  - FLUX.1-dev
  - Step1X-Edit
cells:
  - row: GenEval (T2I) | col: BAGEL | val: "0.88" | color: green
  - row: GenEval (T2I) | col: SD3-Medium | val: "0.74" | color: lavender
  - row: GenEval (T2I) | col: FLUX.1-dev | val: "0.82" | color: lavender
  - row: GenEval (T2I) | col: Step1X-Edit | val: "—" | color: gray
  - row: WISE (knowledge) | col: BAGEL | val: "0.70*" | color: green
  - row: WISE (knowledge) | col: SD3-Medium | val: "—" | color: gray
  - row: WISE (knowledge) | col: FLUX.1-dev | val: "—" | color: gray
  - row: WISE (knowledge) | col: Step1X-Edit | val: "—" | color: gray
  - row: GEdit-EN (overall) | col: BAGEL | val: "6.52" | color: amber
  - row: GEdit-EN (overall) | col: SD3-Medium | val: "—" | color: gray
  - row: GEdit-EN (overall) | col: FLUX.1-dev | val: "—" | color: gray
  - row: GEdit-EN (overall) | col: Step1X-Edit | val: "6.70" | color: lavender
  - row: IntelligentBench | col: BAGEL | val: "44.9" | color: green
  - row: IntelligentBench | col: SD3-Medium | val: "—" | color: gray
  - row: IntelligentBench | col: FLUX.1-dev | val: "—" | color: gray
  - row: IntelligentBench | col: Step1X-Edit | val: "14.9" | color: lavender
notes: 4 rows x 4 cols; green=BAGEL; lavender=specialist baselines; amber=the GEdit row where Step1X edges BAGEL; * = with CoT
-->

Three results, three different stories:

- **Text-to-image (GenEval): BAGEL wins.** 0.88 with prompt rewriting / 0.82 without, versus FLUX.1-dev 0.82 and SD3-Medium 0.74. A unified model beating two dedicated diffusion models on the standard T2I compositional benchmark is the result nobody expected three years ago. On WISE (world knowledge in T2I), BAGEL reaches 0.70 with chain-of-thought — ahead of MetaQuery's 0.55, and the CoT lift is the unification dividend discussed above.
- **Classical editing (GEdit-Bench EN): BAGEL ties, slightly trails.** 6.52 overall, versus Step1X-Edit's 6.70 — a 0.18 gap, with sub-scores of 7.36 on semantic consistency (SC) and 6.83 on perceptual quality (PQ). Step1X-Edit is a *dedicated* editing model, so being within a fifth of a point while also doing everything else is strong. This is the amber cell: BAGEL loses, narrowly, on the task a specialist was purpose-built for.
- **Intelligent editing (IntelligentBench): BAGEL dominates.** 44.0 (or 44.9 in one reported run) versus Step1X-Edit's 14.9 — a 3× margin — rising to 55.3 with chain-of-thought, within a couple points of Gemini 2.0's 57.6. IntelligentBench measures edits that require *understanding* the image and the instruction's intent ("make this look like it's the year 2050," "fix the physics error in this diagram"). This is precisely where a unified model's shared attention between understanding and generation pays off, and where a pipeline of separate models structurally cannot compete.

Unpack each benchmark, because they measure very different notions of "good image":

- **GenEval (0.82 raw / 0.88 with rewriter).** GenEval is a *compositional* T2I benchmark: it auto-checks whether the generated image actually contains the requested objects, in the right counts, colors, and spatial relations ("two red apples to the left of a blue cup"). It is unforgiving of the classic diffusion failure modes — wrong object counts, color bleed, ignored spatial constraints. BAGEL's 0.82 raw already beats FLUX.1-dev (0.82, tie) and SD3-Medium (0.74), and the 0.88 with prompt rewriting is best-in-class. That a *unified* model matches or beats two dedicated diffusion models on the benchmark that most directly measures "did you draw what I asked" is the headline generation result. The reason it can: GenEval rewards *prompt understanding*, and BAGEL's generation expert was initialized from a language model and conditions on a context the understanding side shaped — it parses "two red apples to the left of a blue cup" more reliably than a model whose text encoder is a frozen CLIP.

- **WISE (0.52 raw / 0.70 with CoT).** WISE specifically tests *world knowledge* in T2I — prompts that require knowing a fact to render correctly ("the planet closest to the sun," "the flag of the country that hosted the 2016 Olympics"). This is where the unification dividend is largest: raw, BAGEL's 0.52 trails MetaQuery (0.55), but with chain-of-thought it jumps to 0.70 and leads decisively. WISE is the benchmark that *proves* the think-before-draw mechanism is real and not a curiosity — a +0.18 swing on a knowledge benchmark is the model looking up the fact in text before committing it to pixels.

- **GEdit-Bench-EN (6.52 overall; SC 7.36, PQ 6.83).** GEdit measures *classical instruction editing* on three axes: semantic consistency (did the edit do what was asked), perceptual quality (does the result look good), and an overall blend. BAGEL's 6.52 overall trails the dedicated editor Step1X-Edit (6.70) by 0.18 — a real but small loss. Note the sub-scores: a 7.36 on semantic consistency means BAGEL is *very* good at doing what the instruction said, and the 6.83 perceptual quality is where the dedicated editor's specialization shows. For a model that also reads images and generates from scratch, losing classical editing by a fifth of a point to a purpose-built editor is a strong result, not a weak one.

- **IntelligentBench (44.0–44.9 raw / 55.3 with CoT).** This is the benchmark that separates BAGEL from every classical editor, and the margin is not subtle: 44.9 versus Step1X-Edit's 14.9, a 3× lead, rising to 55.3 with CoT — within striking distance of Gemini 2.0's 57.6. IntelligentBench tests edits that require *understanding the image and the intent* ("make this look like the year 2050," "correct the physics error in this diagram"), which a literal instruction-editor simply cannot do because it has no understanding faculty to consult. The 30-point gap over Step1X-Edit is the single clearest quantification of "what unification buys you" anywhere in the paper.

The pattern across all four: BAGEL is *competitive* on the tasks specialists are built for (GenEval tie/win, GEdit narrow loss) and *dominant* on the tasks that require fusing understanding with generation (WISE-with-CoT, IntelligentBench). That is exactly the profile you would predict from the architecture, which is the best kind of benchmark result — the numbers confirm the story the design tells. A skeptic could dismiss any single benchmark, but the *shape* of the results across eight benchmarks — wins on integration, narrow losses on pure specialization — is hard to explain as anything but a real architectural advantage.

## The any-to-any capability surface

Step back from individual benchmarks and look at the full surface of what one model now does. The taxonomy below is the practical reason to care about BAGEL: it collapses what used to be five or six separate models into one checkpoint.

![Tree of BAGEL's any-to-any capabilities branching from the unified model](/imgs/blogs/bagel-unified-multimodal-mixture-of-transformers-8.png)

<!-- FIGSPEC 8
kind: tree
claim: From one checkpoint, BAGEL branches into understanding, generation, editing, and emergent world-modeling abilities.
caption: One model, four capability families, with the emergent branch reachable only after 3.6T tokens.
nodes:
  - id: root | label: "BAGEL-7B-MoT (one checkpoint)" | color: blue
  - id: und | label: "Understand: VQA, OCR, MathVista 73.1" | color: blue
  - id: gen | label: "Generate: T2I, GenEval 0.88" | color: blue
  - id: edit | label: "Edit: free-form, GEdit 6.52" | color: amber
  - id: world | label: "World ops (emergent @3.6T)" | color: red
  - id: u1 | label: "Caption / grounding" | color: gray
  - id: g1 | label: "CoT-to-image, WISE 0.70" | color: blue
  - id: e1 | label: "Intelligent edit, IBench 55.3" | color: amber
  - id: w1 | label: "Future-frame predict" | color: red
  - id: w2 | label: "3D rotate / navigate" | color: red
edges:
  - root -> und
  - root -> gen
  - root -> edit
  - root -> world
  - und -> u1
  - gen -> g1
  - edit -> e1
  - world -> w1
  - world -> w2
notes: 4-way branch from root; world branch (red) splits into 2; depth-2 tree, render tall; blue (established) -> amber (editing) -> red (emergent)
-->

Four families branch from one checkpoint:

- **Understand (blue):** VQA, captioning, OCR, visual grounding, document and chart reading, multimodal math. The MathVista 73.1 / MM-Vet 67.2 numbers live here.
- **Generate (green):** text-to-image, and crucially *reasoned* text-to-image where the model thinks before drawing (WISE 0.70 with CoT).
- **Edit (amber):** both classical instruction editing (GEdit 6.52) and intelligent editing that requires understanding intent (IntelligentBench 55.3 with CoT).
- **World operations (red), the emergent branch:** future-frame prediction, 3D rotation and viewpoint change, and world navigation across real-world, game, and artistic scenes. These only appear after ~3.6T tokens and are the abilities you cannot get from any single specialist model.

The red branch is the one that makes BAGEL more than "a good unified model." Future-frame prediction means you can hand it a sequence of frames and ask what happens next. 3D manipulation means you can ask it to rotate an object or change the camera viewpoint. World navigation means you can move through a generated scene. None of these were trained as explicit tasks with labeled data — they fall out of the interleaved video and web data once the model has enough capacity and tokens.

## Running it: understanding and generation in code

Enough theory. Here is what it actually looks like to call BAGEL. Two examples follow — an understanding (VQA) call and a generation/editing call — using the `InterleaveInferencer` API from the official repo. First, loading the model.

```python
# pip install torch transformers accelerate safetensors pillow
from huggingface_hub import snapshot_download

# 1) Pull the ~29GB checkpoint (Qwen2.5-7B LLM + SigLIP2 + FLUX VAE + MoT weights)
save_dir = "models/BAGEL-7B-MoT"
snapshot_download(
    repo_id="ByteDance-Seed/BAGEL-7B-MoT",
    local_dir=save_dir,
    local_dir_use_symlinks=False,
)

# 2) Build the model + the three transforms + tokenizer, then the inferencer.
#    (See inferencer.py / app.py in the repo for the full assembly; sketched here.)
from inferencer import InterleaveInferencer
# model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
# are constructed from the downloaded checkpoint per the repo's loader.

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,   # VAE normalization (generation path)
    vit_transform=vit_transform,   # SigLIP normalization (understanding path)
    new_token_ids=new_token_ids,
)
```

Hardware reality check before you run it: the full BF16 model wants **32GB+ VRAM**. The repo ships quantized modes — NF4 runs in **as little as 12GB** (`python app.py --mode 2`), INT8 in **22–32GB** (`--mode 3`) — at the usual quality cost. A single 40GB A100 or 48GB workstation card is the comfortable target.

### Understanding (VQA)

The understanding path sets `understanding_output=True`, which routes the input image through the SigLIP2 encoder and decodes a text answer autoregressively. No diffusion runs; this is pure LLM decoding over visual tokens.

```python
from PIL import Image

image = Image.open("invoice.png")
question = "What is the total amount due, and what is the due date?"

# understanding_output=True -> SigLIP2 path -> autoregressive text head.
out = inferencer(
    image=image,
    text=question,
    understanding_output=True,
    do_sample=False,          # greedy for factual extraction
    text_temperature=0.3,
    max_think_token_n=1000,
)
print(out["text"])
# -> "The total amount due is $4,820.00, due on 2026-07-01."
```

`do_sample=False` (greedy) is the right call for factual extraction like OCR or invoice reading; turn sampling on only for open-ended description. Note that the *same* `inferencer` object does this — there is no separate "understanding model" to load.

### Generation and editing

The generation path leaves `understanding_output` off and triggers the rectified-flow loop. The two CFG knobs are the levers that matter most. For pure text-to-image:

```python
# Text-to-image. cfg_text_scale is the prompt-adherence dial (4-8 typical).
result = inferencer(
    text="a cross-section of a sourdough loaf, dramatic side light, 35mm film",
    cfg_text_scale=4.0,       # 1.0 = ignore prompt; 4-8 = follow it hard
    cfg_img_scale=1.0,        # 1.0 = no image conditioning (pure T2I)
    num_timesteps=50,         # rectified-flow denoising steps
    timestep_shift=3.0,       # bias steps toward layout vs. detail
    cfg_interval=[0.4, 1.0],  # apply CFG only on the back 60% of steps
    cfg_renorm_type="global", # default for T2I
    image_shapes=(1024, 1024),
)
result["image"].save("loaf.png")
```

For editing, you pass an input image *and* a text instruction, raise `cfg_img_scale` so the model preserves the input, and switch the renorm type — the repo specifically recommends `text_channel` for edits:

```python
# Image editing. The input image is encoded by BOTH SigLIP2 (what is it)
# and the FLUX VAE (exactly these pixels) so untouched regions survive.
edited = inferencer(
    image=Image.open("living_room.jpg"),
    text="make the sofa emerald green and add warm evening light",
    cfg_text_scale=6.0,        # follow the edit instruction strongly
    cfg_img_scale=1.5,         # preserve the rest of the photo (1.0-2.0)
    cfg_renorm_type="text_channel",  # recommended for editing
    num_timesteps=50,
)
edited["image"].save("living_room_edited.png")
```

And to invoke the chain-of-thought that lifted WISE from 0.52 to 0.70 and IntelligentBench from 44.0 to 55.3, set `think=True` — the model reasons in text first, then conditions the flow head on its own reasoning:

```python
# "Think then draw" — the unification dividend in one flag.
thoughtful = inferencer(
    text="the planet named after the Roman god of war, photorealistic from orbit",
    think=True,               # reason in text ("Mars, the red planet...") first
    max_think_token_n=1000,
    cfg_text_scale=4.0,
    num_timesteps=50,
)
print(thoughtful.get("text"))   # the reasoning trace
thoughtful["image"].save("mars.png")
```

The defaults worth memorizing: `cfg_text_scale` lives in 4–8, `cfg_img_scale` in 1.0–2.0 (1.0 disables image conditioning entirely, which is what makes the same call do both pure T2I and editing), `num_timesteps=50`, `timestep_shift=3.0`. The interleave inference method exposes the full set with sensible defaults (`cfg_text_scale=3.0`, `cfg_img_scale=1.5`, `cfg_interval=[0.4, 1.0]`, `cfg_renorm_min=0.0`).

## Case studies: where the unification earns its keep (and where it doesn't)

Twelve concrete scenarios, drawn from the capability surface and the failure modes the design implies. Each is the kind of thing you would actually try, with the result you should expect.

### 1. The invoice-then-redact pipeline

A finance team needs to read totals off scanned invoices *and* produce a redacted copy with the account number blacked out before forwarding it to an external auditor. The old way is a two-stage pipeline: an OCR/VLM extracts the structured fields, then a separate image editor (or a human in Photoshop) draws the redaction box. That pipeline has two seams where it fails. The OCR model knows *where the text is* but emits only the parsed string, so its spatial knowledge is thrown away before the editor runs; the editor then has to *re-find* the account number from scratch, usually via a brittle regex on a second OCR pass plus bounding-box heuristics. Every handoff between the two models loses the context the next stage needs.

With BAGEL it is one model and two calls on the same checkpoint. First, `understanding_output=True` with the prompt "Extract the total and due date" returns "$4,820.00, due 2026-07-01." Second, an editing call — "black out the account number in the top-right field" — runs on the same image. The win is structural: the editing call's generation expert attends, through shared self-attention, to the *same ViT tokens* that the understanding call used, so the model already has a localized representation of where the account number sits. It does not re-find anything; it inherits the spatial understanding. The redaction lands on the right region because the model that draws the black box is the model that read the field.

The honest caveats. BAGEL's text rendering is weak, so you would not ask it to *rewrite* a field, only to occlude one — occlusion is a region-fill, not a typography task, so it sidesteps the weakness. And for a true compliance workflow you would still verify the redaction with a deterministic check (re-OCR the output, assert the account number is gone) rather than trusting the model blind. But the core lesson holds and generalizes: **unification's payoff is highest where the edit needs to know what it is editing.** Any workflow shaped like "understand the image, then modify the thing you just understood" collapses two models and an information-losing handoff into one model with one shared context. That is the canonical BAGEL use case, and invoice redaction is just its smallest instance.

### 2. The MMMU regression you should expect

A team replaces its Qwen2.5-VL-7B deployment with BAGEL, sold by the headline "beats the specialist VLMs," and expects a free upgrade across the board. Then their internal eval — which happens to be dominated by college-level domain-reasoning questions over textbook figures, an MMMU-shaped distribution — *drops* from roughly 58.6 to 55.3. Panic ensues; someone files a regression; someone proposes rolling back.

This is not a bug, and the rollback is the wrong move. MMMU is the *single* understanding benchmark in the entire suite where dedicated VLMs still beat BAGEL, and the reason is structural and unavoidable: MMMU rewards encyclopedic domain knowledge applied to images, and a pure VLM spends 100% of its parameters on understanding while BAGEL spends some on generation. The 3.3-point gap is the literal cost of that capacity split on the one task that most rewards undiluted understanding capacity. It is the *expected* shape of the tradeoff, documented in the paper's own tables.

The fix is to scope the decision to the *actual* workload rather than to one unrepresentative eval. If this team's product genuinely is "answer hard domain-exam questions over figures and do nothing else," then a specialist VLM is the right tool and they should not have switched — BAGEL's generation and editing abilities are dead weight for that use case. But if the product is broader — if it also captions, does visual math, handles open-ended integrated tasks, or (the real reason to consider BAGEL) needs to *generate or edit images* anywhere in the workflow — then the 3.3-point MMMU dip is dwarfed by BAGEL's wins on MMBench (85.0), MathVista (73.1), and MM-Vet (67.2), plus the entire generation capability the specialist simply does not have.

The deeper lesson, and the one worth tattooing on the wall: **benchmark on your distribution, not on the leaderboard's.** "Unified beats specialists" is a true statement *on average across eight benchmarks*, and it is a false statement *on the single cell* that is MMMU. A team that evaluates on a workload skewed toward that one cell will see a regression and conclude the model is worse, when in fact the model is better everywhere their leaderboard does not look. The mistake is not choosing BAGEL; it is letting an unrepresentative eval stand in for the production distribution.

### 3. The "think before you draw" save on world knowledge

A marketing tool generates hero images from copywriters' prompts, and the copywriters write the way humans talk — indirectly. They ask for "the bird on the classic Twitter logo, but reimagined as a majestic eagle," "the animal that produces wool, rendered in cyberpunk neon," "a still life featuring the fruit Newton supposedly watched fall." Direct text-to-image gets these wrong a frustrating fraction of the time, and the failures are *specific*: the model produces something bird-shaped but generic, or a sheep-like animal that is not quite cyberpunk, or a random fruit. The pattern is always the same — the hard part of the prompt was a *knowledge hop*, not a rendering problem, and the diffusion path has no faculty to make the hop.

Switching on `think=True` changes the failure rate dramatically, and you can watch *why* in the reasoning trace. For "the animal that produces wool," the model first decodes text: "The animal that produces wool is a sheep. I'll render a sheep with neon cyberpunk lighting, chrome accents, and a dark city backdrop." Only then does the generation expert denoise, conditioned on a context that now literally contains "sheep" and a style description. The riddle is resolved into a concrete visual brief before a single denoising step runs. This is the WISE 0.52→0.70 jump made operational — a +0.18 swing on exactly the class of prompt this tool sees all day.

The operational discipline is to *classify prompts* and route them. A prompt that is already literal and visual ("a golden retriever puppy on a red couch") gets `think=False` — the reasoning would add latency for no gain. A prompt with an indirect referent, a world-knowledge dependency, or a compositional constraint that needs planning gets `think=True`. A cheap heuristic that works surprisingly well: if the prompt contains a *definite description* ("the X that does Y") rather than a *direct noun*, turn thinking on. You can even have a small classifier or a regex gate make the call automatically, so copywriters never have to think about the flag.

The lesson: **any prompt with an indirect referent or world-knowledge hop is a candidate for chain-of-thought, and the reasoning is "free" in the sense that the faculty already exists in the same model.** You are not bolting on a second system; you are spending a few hundred extra tokens of the model's own reasoning to turn a riddle into a brief. The visible reasoning trace is a bonus — when a thoughtful generation still fails, you can read where the model's reasoning went wrong and fix the prompt, which is debugging information a black-box diffusion model never gives you.

### 4. Free-form editing that classical pipelines refuse

A real-estate platform wants to show listings across seasons: upload a summer photo of a house, request "show this in winter." This is a deceptively hard edit. "Winter" is not a region or a color you can mask and replace — it is a *globally consistent semantic transformation*: the deciduous trees lose their leaves, the lawn turns to snow, the light goes cold and low-angle, the sky greys, maybe there is frost on the windows and the evergreens hold a dusting of snow while the bare oaks do not. A classical instruction-editor or an inpainting pipeline needs a mask and a literal target, and there is no single mask for "winter" — the change touches the entire image and depends on *what each part of the image is*. The trees need different treatment from the lawn from the sky, and knowing that requires understanding the scene.

BAGEL's intelligent-editing capability — the one that scores 55.3 on IntelligentBench with CoT versus Step1X-Edit's 14.9 — handles this because the edit conditions on a full understanding of the scene through shared attention. The generation expert can attend to ViT tokens that have already parsed "deciduous tree here, evergreen there, lawn in front, sky above" and apply the season-appropriate transformation to each region, while the clean VAE tokens of the input keep the house's architecture and the photo's composition fixed. It is doing, in one model, what a human editor does mentally: *understand the scene, then transform it consistently*.

The honest boundaries. This is the reddest, most emergent branch of the capability tree (3.6T-token threshold), so quality varies and you should A/B against your bar before shipping listing photos that buyers will see — a winter render with melting-in-summer shadows or a snowless lawn under bare trees is worse than no render. And BAGEL is not a substitute for a controlled, deterministic pipeline if you need *exact* fidelity (you cannot promise the house's window count is preserved without checking). But for the broad class of "globally consistent semantic edit," the alternatives are either a hand-built per-region pipeline (expensive and brittle) or commissioning new photography (far more expensive).

The lesson: **when the instruction is semantic rather than spatial — when it describes a *what* rather than a *where* — the 3× IntelligentBench gap is exactly what you feel in production.** Spatial edits ("blur this box," "remove this object") are well-served by classical tools; semantic edits ("make it winter," "age this person twenty years," "make this look abandoned") are where a unified model's understanding-conditioned generation has no real competitor among open models.

### 5. Future-frame prediction as a poor-engineer's world model

A robotics team is prototyping a planner and wants a quick way to *imagine* the next few frames of a tabletop manipulation sequence — gripper approaches block, block tips, block settles — without standing up a full video-generation or world-model stack, which is a multi-month project on its own. They have a handful of frames and a question: "given this trajectory so far, what does the scene look like a few frames later?"

BAGEL's emergent future-frame prediction answers this with zero additional training. The ability is a direct byproduct of the 0.7T tokens of interleaved video data — Koala36M's temporally coherent clips taught the model that an image sequence has a *next frame* that is causally related to the previous ones. Mechanically, you give BAGEL the frames as a sequence of clean image tokens and prompt for continuation; the generation expert denoises the next frame conditioned, through shared attention, on the dynamics implied by the preceding frames. For coarse, short-horizon prediction in visually familiar domains, the results are genuinely useful — useful enough to unblock a planner prototype while the real video model is being built.

The caveats are load-bearing and you ignore them at your peril. This is bleeding-edge emergent behavior: prediction quality degrades fast with horizon (a few frames good, many frames drifts), it has no guarantee of physical consistency (it predicts *plausible-looking* next frames, not physically *correct* ones — momentum and contact dynamics are approximated, not simulated), and it is unvalidated for any safety-relevant decision. A robot that *acts* on BAGEL's frame predictions without a real dynamics model in the loop is a robot that will eventually do something stupid because the model imagined a physically impossible but visually plausible continuation. The right framing is "a fast, free, surprisingly good *prior*" — something to seed a planner or to sanity-check a hypothesis, not something to close a control loop around.

The lesson generalizes to the whole red branch of the capability tree: **the emergent abilities are real and usable, but treat them as strong zero-shot baselines, not as guaranteed-SLA production features.** They are extraordinary for what they cost (nothing — they fall out of pretraining), and that is exactly why it is tempting to over-trust them. Use them to move fast and to establish a floor; replace them with a purpose-built system anywhere correctness, not plausibility, is the requirement.

### 6. The 3D-rotation trick for product photography

An e-commerce team has exactly one product photo per SKU — a single front three-quarter shot from the supplier — and wants the richer multi-angle galleries that lift conversion: a side profile, a back view, a flat-on front. Commissioning new photography means shipping physical samples to a studio, which for a catalog of thousands of SKUs is prohibitively slow and expensive. The question is whether a model can *synthesize* the missing viewpoints from the one photo they have.

BAGEL's emergent 3D-manipulation ability does exactly this. Prompts like "show this product rotated 45 degrees to the right" or "show the back of this object" produce plausible novel viewpoints, and the ability traces directly to the MVImgNet2.0 multi-view data in the 0.7T interleaved-video budget — the model saw many objects photographed from many angles and learned the implicit 3D structure that lets it rotate. The generation expert, conditioned on the input image's clean tokens through shared attention, produces a coherent new-angle render that preserves the object's identity, color, and material while changing the camera.

The honest assessment: this is the *bleeding edge* of the emergence curve (the 3.6T-token tier), and it shows. Quality is inconsistent — simple, rigid objects with familiar geometry (boxes, bottles, shoes) rotate well; complex objects with fine detail or unusual geometry can hallucinate the unseen back as something subtly wrong. For categories where the back genuinely differs from the front in ways that matter to the buyer (a garment with a printed graphic, a device with ports on the rear), the synthesized view can be confidently incorrect, which is worse than no view at all because it misleads. The workable pattern is to use BAGEL's renders for *low-stakes* categories and as *drafts* a human approves before they go live, not as unattended catalog automation.

The lesson, again pointed at the red branch: **the redder the branch on the capability tree, the more you should A/B against your quality bar before shipping, and the more you should keep a human in the loop.** 3D rotation is remarkable for a model that was never explicitly trained on a novel-view-synthesis objective, and for many catalog uses it genuinely beats the alternative of no extra views. But "remarkable for free" is not the same as "production-reliable," and the gap between those two is where careful teams put a review step.

### 7. The VRAM cliff and the quantization ladder

A solo developer reads "7B active parameters," concludes BAGEL is a 7B model, and tries to run the BF16 checkpoint on their 16GB consumer GPU. It OOMs immediately, and the confusion is understandable — the headline number *is* 7B, but the headline number describes *compute*, not *memory*. The two experts are both resident in memory at all times; the 7B-active figure means a given token only flows through 7B of weights, but all 14B of weights must be loaded to route any token at all. The full BF16 checkpoint is ~29GB on disk and wants comparable VRAM plus activation overhead. A 16GB card cannot hold it.

The fix is the quantization ladder the repo ships, and it is worth knowing the exact rungs. Full BF16 needs 32GB+ and is the quality reference. INT8 (`python app.py --mode 3`) fits in roughly 22–32GB with a modest quality cost. NF4 (`--mode 2`) squeezes into as little as 12GB by quantizing weights to 4-bit, which is what finally lets the model run on a 12–16GB consumer card. Our solo developer's path is `--mode 2`, and it works — but with a catch they need to anticipate.

The catch is that quantization hits *generation* harder than understanding. Reading an image and answering a question is robust to 4-bit weights — the autoregressive text path degrades gracefully, the way quantized LLMs do. But the rectified-flow generation path is sensitive: NF4 visibly softens fine detail, worsens the already-weak text rendering, and can introduce subtle color and texture artifacts that a side-by-side reveals immediately. So the practical guidance splits by use case. If you mostly *understand* images, NF4 on a 12GB card is a fine deployment. If you mostly *generate or edit*, the quality hit of NF4 may be unacceptable and you should find a 24GB+ card and run INT8 or BF16. The quantization decision is not one-size-fits-all; it depends on which half of the model you lean on.

The lesson: **BAGEL is a 14B-parameter model wearing a 7B-active costume — plan VRAM for the parameter count, not the active count.** The "7B active" framing is a *compute* and *latency* claim (per-token FLOPs of a 7B dense model), and it is true and useful for throughput planning. It is *not* a memory claim, and reading it as one is the single most common deployment mistake. Size your hardware to hold all 14B, then choose your quantization rung based on whether generation quality matters for your workload.

### 8. The text-rendering weakness

A small-business owner asks BAGEL to "generate a poster that says 'GRAND OPENING — JUNE 12' with a coffee cup illustration." The coffee cup comes out beautifully; the text comes out as garbled pseudo-letters, missing a character here, doubling one there, occasionally spelling something adjacent but wrong. This is a known and documented limitation — the project notes explicitly that "precise text rendering remains limited," especially before the latest training stages — and it is the most common disappointment new users hit, because rendering text *feels* like it should be easy.

It is not easy, and the reason is the VAE. BAGEL generates in the FLUX VAE's latent space, which is optimized for *perceptual* reconstruction — it preserves how an image *looks* — and crisp glyphs are a high-frequency, low-tolerance signal where being off by a few pixels turns an "E" into an "F" or a smear. The latent diffusion process is excellent at textures, lighting, and shapes, where small errors are invisible, and poor at typography, where small errors are spelling mistakes. This is a *general* property of VAE-latent diffusion at this scale, not a BAGEL-specific bug — the dedicated diffusion models share the weakness unless they were specifically trained with text-rendering objectives and high-resolution glyph data. BAGEL's broad data mix did not prioritize that, so its text rendering sits where you would expect.

The practical workarounds are straightforward once you accept the limitation. For posters and any typography-critical output, treat BAGEL as the *illustration* engine and composite the text in a deterministic layer afterward — render "GRAND OPENING — JUNE 12" with an actual font in your design tool or with a few lines of PIL, and overlay it on BAGEL's coffee-cup image. You get perfect text and a great illustration, which is strictly better than asking one model to do both. Alternatively, for short, large words where occasional errors are tolerable, you can generate and regenerate until the text happens to come out right, but that is a slot machine, not a workflow.

The lesson: **for typography-heavy generation, reach for a model with explicit text-rendering training or composite the text in afterward — BAGEL is for imagery, not signage.** This is not a knock on the model; it is a knock on the expectation. Every generation model has a competence frontier, and "render arbitrary crisp text" sits outside BAGEL's. Knowing where the frontier is, and routing around it, is the difference between a frustrated user and a productive one.

### 9. The editing renorm footgun

An engineer integrating BAGEL copies the text-to-image example that worked perfectly, swaps in an input image and an edit instruction, and gets nonsense — edits that either ignore the instruction entirely or apply it while wrecking the rest of the image, smearing the regions that should have been preserved. They conclude the model is bad at editing. They are wrong; they left two parameters at their text-to-image defaults, and those defaults are actively harmful for editing.

The two parameters are `cfg_img_scale` and `cfg_renorm_type`. For pure text-to-image, `cfg_img_scale=1.0` (no image conditioning — there is no input image) and `cfg_renorm_type="global"` are correct. For editing, both are wrong. `cfg_img_scale` must rise to roughly 1.5 so the model actually *preserves the input image* — at 1.0 it ignores the input and effectively regenerates from scratch, which is why the "rest of the image" gets wrecked. And `cfg_renorm_type` should switch to `text_channel`, which the repo specifically recommends for edits, because the global renormalization that works for unconditioned generation interacts badly with strong image conditioning and produces the artifacts. Set `cfg_img_scale=1.5`, `cfg_renorm_type="text_channel"`, keep `cfg_text_scale` strong (6.0 is a good editing value so the instruction is followed firmly), and the same model that produced nonsense now edits cleanly.

What makes this a genuine footgun rather than a simple typo is that *the same function call serves both tasks*. There is no separate `edit()` method — `cfg_img_scale` is the dial that turns a generation call into an editing call, continuously. At 1.0 you are generating; at 1.5 you are editing; the model figures out which from the presence of an input image and the scale of the conditioning. This is elegant and dangerous in equal measure: elegant because one API does everything, dangerous because copying the wrong example silently puts you in the wrong mode with no error, just bad output.

The lesson: **BAGEL's flexibility comes from its CFG knobs, and the same call behaves completely differently depending on `cfg_img_scale` and `cfg_renorm_type` — read the hyperparameter docs before blaming the model.** The repo documents the recommended values per task (T2I vs editing); internalize the small table of "for this task, set these knobs to these values," and most of the "BAGEL is bad at X" complaints evaporate. The model is not bad at editing; the defaults are tuned for generation, and the cost of flexibility is that you must choose the right knobs for the task in front of you.

### 10. The web-flow generation use case

A documentation team is building an illustrated tutorial and wants the figures generated *in the flow* of the prose — a series of diagrams where each image is conditioned not just on its own caption but on everything written and drawn so far, so the visual style stays consistent and each figure builds on the last. With a one-shot text-to-image model this is painful: every prompt is seen in isolation, so figure 3 has no idea what figures 1 and 2 looked like, and you end up with a stylistically incoherent gallery that you have to wrestle into consistency with elaborate prompt engineering and seed-locking.

This is exactly what the 0.4T tokens of interleaved web-generation data trained BAGEL to do. The OmniCorpus-derived, caption-first web documents taught the model to generate "the image that belongs here given the document so far" — to treat image generation as a continuation of an interleaved text-and-image context rather than a stateless prompt-to-pixels mapping. In practice you build up the sequence — prose, then figure 1, then more prose, then a prompt for figure 2 — and BAGEL's generation expert conditions, through shared attention, on the *clean tokens of the previous figures and all the surrounding text*. Figure 2 inherits figure 1's visual language because figure 1 is literally in its context. The style consistency that one-shot models fight for comes more naturally here.

The boundaries are worth stating. This is not a layout engine — BAGEL generates images, not paginated documents, so you still own the composition of text and figures. Context length bounds how much prior material can condition a new figure, so very long documents will eventually lose the earliest figures from context. And the text-rendering weakness from case study 8 still applies, so figures with embedded labels need the labels composited in. But for the core task — *a coherent series of illustrations that build on each other* — BAGEL has a structural advantage that a prompt-at-a-time diffusion model cannot match without bolting on external state.

The lesson: **the interleaved training data unlocks sequential, context-aware generation that a prompt-at-a-time diffusion model structurally lacks.** Any workflow shaped like "a sequence of images that should be mutually consistent and conditioned on shared context" — tutorial figures, storyboards, a children's book where the character must look the same on every page — is a place where BAGEL's interleaved pretraining pays a dividend that is invisible on single-image benchmarks but obvious the moment you need image *N+1* to remember images *1 through N*.

### 11. Replacing a two-model pipeline and halving ops cost

A content platform runs two model deployments side by side: a Qwen2.5-VL-7B for the understanding features (alt-text generation, image moderation, visual search) and a separate FLUX deployment for the generation features (thumbnails, illustrations, edits). That is two model families, two sets of weights resident in GPU memory, two serving stacks with two sets of autoscaling rules, two on-call rotations' worth of operational surface, and two upgrade cadences to keep in sync. The understanding traffic and the generation traffic are bursty and uncorrelated, so the platform over-provisions both tiers to handle their independent peaks, and the GPUs sit half-idle most of the time.

Consolidating to a single BAGEL checkpoint removes an entire serving tier. The arithmetic on capability is favorable: you trade two specialists (roughly 8B and 12B parameters) for one 14B unified model that *matches or beats both on most of the tasks the platform actually runs* — BAGEL wins the broad understanding benchmarks and is competitive-to-winning on generation. You give up a little on the narrowest specialist edges (MMMU-hard reasoning, peak T2I fidelity, crisp text), but the platform's workload does not live there. What you gain operationally is large: one checkpoint to deploy and version, one server to scale, one model's worth of GPUs to provision, and — the subtle structural win — a *shared KV cache across modalities*, so a request that both reads an image and generates from it reuses one context instead of paying for two separate forward passes across two models with a serialization hop between them.

There is a capacity-planning nuance to get right. The consolidated tier must be sized for the *combined* peak, and because both experts are always memory-resident, each replica needs VRAM for the full 14B. But the *utilization* improves: instead of two half-idle pools, you have one pool absorbing both traffic streams, and the statistical multiplexing of uncorrelated bursts means you can run the consolidated tier at higher average utilization than either specialist pool. Fewer replicas, higher utilization, one stack to operate.

The lesson: **the operational simplification — one checkpoint, one server, shared KV cache across modalities, one upgrade path — is often a bigger win than any single benchmark delta.** Teams evaluating BAGEL tend to fixate on whether it beats their current model on metric X, which is the wrong frame if their current *system* is two or three models stapled together. The right question is whether one BAGEL can replace the whole stack at acceptable quality, and for a platform whose workload spans understanding and generation, the answer is increasingly yes — at which point the win is measured in deleted services, not benchmark points.

### 12. The MMMU-hard, typography-hard, real-time-hard exclusion zone

A team is building a real-time tutoring assistant for competition math. The product takes a screenshot of a problem (often with a geometry figure or a chart), reasons through it at a high level, and renders a clean worked solution with *crisp typeset equations* — and it has to feel interactive, sub-second where possible, because students lose patience fast. They evaluate BAGEL because "one model that reads and draws" sounds perfect for "read the problem screenshot, draw the solution." After a week they walk away, and they are right to.

Three strikes, each fatal independently. First, the reasoning: competition math over figures is the MMMU-hard regime, the *one* understanding axis where BAGEL (55.3) trails dedicated VLMs (Qwen 58.6), and a tutoring product cannot afford to be second-best at the core reasoning. Second, the output: a worked math solution is *typography* — equations, fractions, subscripts, exponents — and crisp text rendering is precisely BAGEL's documented weakness (case study 8). The model that garbles "GRAND OPENING" will certainly garble a quadratic formula, which in a math product is not a cosmetic flaw but a correctness failure. Third, the latency: 50-step rectified flow at 1024×1024 is *seconds* of generation, not the milliseconds an interactive tutor needs, and there is no cheap way to make rectified-flow image generation real-time at quality.

The right architecture for this team is the opposite of consolidation: a strong reasoning VLM (or an LLM with a vision adapter) for the math, and a *deterministic* renderer — LaTeX, MathML, a templating engine — for the equations. The solution should be *typeset*, not *generated as an image*, because typesetting is exactly the deterministic, pixel-perfect, instant operation that image generation is not. BAGEL has no role here, and forcing it in would make every one of the three problems worse.

The lesson, and the bridge to the closing section: **BAGEL is a breadth play, not a peak play.** Its value is collapsing many capabilities into one model at *competitive* quality on each, with *dominant* quality on the tasks that fuse understanding and generation. When your product needs the single best score on one narrow axis — hardest reasoning, perfect typography, or real-time latency — a specialist (or a deterministic tool) still wins, and trying to use a generalist there is a category error. The skill in deploying BAGEL is recognizing which of your tasks are breadth tasks (most of them, usually) and which are peak tasks (a few, often the ones you remember), and routing the peak tasks elsewhere.

## When to reach for BAGEL, and when not

BAGEL is the strongest open argument to date that unification need not tax you — but "need not on average" is not "never on your specific task." Here is the decision rule.

**Reach for BAGEL when:**

- You need *both* understanding and generation in one workflow, and especially when the generation step needs to *understand* the input — instruction editing, intelligent editing, "make it look like winter." This is where the shared-attention dividend (IntelligentBench 44.9 vs 14.9) is largest.
- You want one checkpoint to consolidate a multi-model pipeline (a VLM + a diffusion model + an editor) and cut your serving footprint to a single tier.
- Your generation prompts involve world knowledge or indirect references, so the `think=True` chain-of-thought path (WISE 0.52→0.70) earns its keep.
- You want emergent capabilities — future-frame prediction, 3D rotation, sequential context-aware generation — as strong zero-extra-training baselines.
- You value open weights (Apache-2.0) and want to fine-tune or self-host the whole stack.
- Your understanding needs are broad-perception and math (MMBench, MathVista, MM-Vet) rather than peak college-reasoning.

**Skip BAGEL when:**

- You only do *one* of understanding or generation, at the highest quality, on a narrow axis. A dedicated VLM still edges it on MMMU (58.6 vs 55.3); FLUX/SD3 remain strong T2I-only choices when you don't need understanding; Step1X-Edit narrowly leads classical editing (6.70 vs 6.52).
- You need precise **text rendering** in generated images (posters, signage, UI mockups) — text rendering is a known weakness.
- You need **real-time** generation — 50-step rectified flow at 1024×1024 is seconds, not milliseconds.
- You are VRAM-constrained below ~12GB even with NF4, or you cannot tolerate the quality hit of NF4/INT8 quantization. Remember the memory footprint is the full 14B, not the 7B active.
- Your task lives entirely in the MMMU-hard / typography-hard / latency-hard exclusion zone of case study 12.

The deeper takeaway is architectural, and it generalizes beyond BAGEL. The reason unification used to tax you was that people coupled things that wanted to be decoupled — one tokenizer for two opposite representational needs, one weight set for two conflicting objectives. BAGEL's recipe is *decouple what conflicts, share what must communicate*: two encoders (decoupled representations), two experts (decoupled weights), one shared self-attention (the communication bus), one interleaved training stream (the shared experience that makes emergence possible). Get that decomposition right and the tradeoff you thought was fundamental turns out to have been a design smell all along. That is the lesson worth carrying to the next unified-everything model — and to BAGEL's siblings in ByteDance's lineup.

## Further reading

- BAGEL paper: "Emerging Properties in Unified Multimodal Pretraining" (arXiv:2505.14683).
- Model weights and inference code: [ByteDance-Seed/BAGEL-7B-MoT on HuggingFace](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) and the [GitHub repo](https://github.com/ByteDance-Seed/Bagel) (Apache-2.0).
- For the native-resolution vision-language lineage BAGEL's understanding side draws on: [Seed1.5-VL](/blog/paper-reading/multimodal/seed1-5-vl-native-resolution-vision-language) and [Qwen2-VL's any-resolution perception](/blog/paper-reading/multimodal/qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolution).
- For the MoE/decoupled-encoder tradition in open multimodal models: [DeepSeek-VL/VL2's dynamic tiling and MoE](/blog/machine-learning/computer-vision/deepseek-vl-vl2-dynamic-tiling-moe).
- For where BAGEL sits among ByteDance's models: [the ByteDance Seed model universe by use case](/blog/machine-learning/large-language-model/bytedance-seed-model-universe-by-use-case).
