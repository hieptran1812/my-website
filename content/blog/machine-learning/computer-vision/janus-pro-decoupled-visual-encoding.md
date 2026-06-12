---
title: "Janus and Janus-Pro: Decoupling Visual Encoding for Unified Understanding and Generation"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "Why one shared visual encoder is a bad compromise for a model that must both read and draw images, and how DeepSeek's Janus family splits the encoders while keeping a single autoregressive transformer body."
tags: ["janus", "janus-pro", "deepseek", "multimodal", "vision-language-model", "image-generation", "siglip", "vq-tokenizer", "rectified-flow", "unified-model", "computer-vision", "autoregressive"]
category: "machine-learning"
subcategory: "Computer Vision"
author: "Hiep Tran"
featured: true
readTime: 50
---

The first time you try to build a single model that both *reads* an image and *draws* one, you reach for the obvious design: one vision encoder, shared between the two tasks. Pixels go in, features come out, the transformer reads the features when it needs to understand and writes them back when it needs to generate. It is the design that nearly every "unified multimodal model" paper drew on its first whiteboard. It is also the design that quietly sabotages the understanding side of your model, and most people who ship it never figure out why.

The reason is a granularity mismatch that lives below the level most architecture diagrams show. When a model *understands* an image — answers a question about it, reads the text in it, reasons about spatial relationships — it wants high-level semantic features: "there is a dog, it is to the left of the couch, the sign says EXIT." When a model *generates* an image, it wants the opposite: fine-grained, near-pixel detail it can reconstruct, the exact texture of the fur and the precise stroke of the letters. A single encoder cannot be excellent at both at once. Push it toward semantics and your generated images go mushy; push it toward reconstructable detail and your VQA accuracy craters. You get a model that is mediocre at both jobs and you blame the LLM backbone, the data, the learning rate — anything but the encoder, which is the actual culprit.

DeepSeek's Janus family ([arXiv 2410.13848](https://arxiv.org/abs/2410.13848), CVPR 2025) names this the "Chameleon problem," after the earlier unified model that ate the compromise, and fixes it with a move that is obvious in hindsight and almost nobody had committed to: **decouple the visual encoding, keep the transformer single.** Two encoders feed one autoregressive body. Understanding gets its semantic encoder, generation gets its detail-preserving tokenizer, and the LLM in the middle never has to pretend a single feature space serves both masters.

![One shared visual encoder is a forced compromise; Janus splits the encoder in two yet keeps a single transformer body.](/imgs/blogs/janus-pro-decoupled-visual-encoding-1.webp)

The diagram above is the mental model for this entire article. On the left is the shared-encoder design and its cascade of compromises: one encoder, both tasks starved, suboptimal understanding scores. On the right is Janus: two encoders feeding one autoregressive transformer, each task getting the granularity it actually needs, strong on both jobs. Everything that follows — the original Janus, the rectified-flow variant JanusFlow, and the scaled-up Janus-Pro — is a tour of how you make that right-hand picture actually work, and what you have to get right in training to keep it from collapsing back into the left-hand failure mode.

This is not a paper summary. It is a working engineer's account of *why the decoupling is load-bearing*, where the design has sharp edges, and what the three Janus papers teach about training a model with two heads on one body. If you are building or fine-tuning a unified multimodal model — or trying to decide whether "unified" is even worth it over two specialized models — this is the set of tradeoffs you need in your head.

## Why "unified multimodal" is harder than it looks

The selling point of a unified model is seductive. One set of weights that can look at a chart and answer questions, then turn around and generate a diagram. One serving stack, one fine-tuning pipeline, one place where image understanding and image generation can in principle *transfer* to each other. The dream is that understanding makes generation more grounded and generation makes understanding more detailed, and you get a virtuous cycle for free.

The reality is that the two tasks pull the shared parameters in opposite directions, and the visual encoder is where the tension is sharpest. Here is the assumption-versus-reality table I wish someone had handed me before I burned a week on a shared-encoder prototype.

| Assumption | The naive view | The reality |
|---|---|---|
| One encoder is simpler | Fewer parameters, one code path, less to train | The encoder has to satisfy two contradictory objectives; it ends up bad at both |
| Understanding and generation want the same features | Both "look at the image," so both want the same representation | Understanding wants semantics (what is in the scene); generation wants reconstructable detail (how to redraw it) |
| Sharing forces useful transfer | Joint training makes the two tasks help each other | Joint training through a shared encoder makes them *fight* over the encoder's capacity |
| The LLM backbone does the heavy lifting | The encoder is just a thin adapter; the transformer learns the rest | The encoder fixes the granularity ceiling; no amount of transformer capacity recovers detail the encoder discarded |
| Generation must be discrete autoregression | To live inside an LLM, images must become tokens you predict left-to-right | Rectified flow trains inside the same transformer with no codebook ceiling (JanusFlow) |

The deepest of these is the second row. A semantic encoder like CLIP or SigLIP is *trained to throw away* exactly the information generation needs. Contrastive image-text pretraining rewards a representation where "a photo of a golden retriever" and "a golden retriever on a couch" land near the same image embedding — which means the encoder is explicitly optimized to be *invariant* to the couch, the lighting, the exact pose. That invariance is gold for understanding and poison for generation. If you try to reconstruct an image from a SigLIP embedding, you get the gist and lose the specifics, because the specifics were deliberately compressed out. Conversely, a VQ tokenizer trained for reconstruction preserves spatial detail beautifully and carries almost no high-level semantics — its codebook entries are texture-and-edge primitives, not concepts. Ask a VQA model to reason over raw VQ tokens and it flails.

So the shared encoder is not a minor inefficiency. It is a representation that has to be simultaneously invariant to and faithful to the same details. There is no such representation. The best you can do with one encoder is a blurry midpoint, and a blurry midpoint is what shared-encoder unified models deliver: passable generation, disappointing understanding, and a nagging sense that the parts should add up to more than they do.

The third row of the table is the one people most want to be true and most often is not. The whole emotional appeal of a unified model is that understanding and generation will *help each other* — that learning to draw a cat will deepen the model's understanding of cats, and vice versa. That transfer is real and worth chasing, but it does not happen automatically through a shared encoder. Through a shared encoder, the two tasks do not cooperate; they *compete*, each pulling the encoder's limited capacity toward its own granularity. The transfer that actually pays off is the kind JanusFlow engineers explicitly, with a representation-alignment loss that pulls the generation path's internal features toward the understanding encoder's semantics. That is transfer by *design*, across a shared body, not transfer as a hoped-for side effect of a shared encoder. The distinction matters: shared-body transfer you can build and measure; shared-encoder transfer is mostly a story people tell themselves while the encoder quietly degrades both tasks. Janus's architecture is what lets you pursue the former without paying for the latter.

### The second-order trap: you will blame the wrong component

The cruelest part of the shared-encoder failure is that it does not announce itself. Your loss curves look fine. Generation is acceptable. Understanding is just... a few points below where a dedicated VLM of the same size lands, and you spend weeks chasing the gap in the LLM, the projector, the instruction data. The gap is structural. It lives in the encoder, and the only fix is to stop sharing it. This is exactly the diagnosis Janus makes, and it is why the rest of this article spends so much time on a design choice that is, on the surface, a one-line change to the architecture.

## 1. The decoupling: two encoders, one transformer

**Rule of thumb: pick the encoder from the task's information-granularity needs, not from a desire to share parameters. Sharing the body is free; sharing the encoder is not.**

Janus's architecture is precise about *what* it decouples. It is not two separate models stitched together, and it is not a mixture-of-experts router. It is one autoregressive transformer — the DeepSeek-LLM 1.3B backbone, with a 4096-token context window — fed by two different visual front-ends depending on the task. The transformer body is shared completely. Only the visual *encoding* is split.

![Janus architecture: two encoders, one transformer, two heads; each task picks its own encoder while both share the DeepSeek-LLM body.](/imgs/blogs/janus-pro-decoupled-visual-encoding-2.webp)

Walk the diagram left to right. For **understanding**, an input image (384×384) goes through a **SigLIP-Large-Patch16-384** encoder, which produces a 2D grid of high-dimensional semantic features. That grid is flattened to a 1D sequence and passed through a small **understanding adaptor** — a two-layer MLP — that projects the SigLIP feature dimension into the LLM's input embedding space. For **generation**, a target image goes through a **VQ tokenizer** with a codebook of 16,384 entries that downsamples the image by a factor of 16, producing a grid of discrete codebook IDs. Those IDs are looked up in an embedding table and passed through a separate **generation adaptor**, also a two-layer MLP, into the same LLM input space. Text tokens take the ordinary embedding-lookup path. All three streams become sequences of vectors in the same dimensionality, and the transformer processes them as one undifferentiated sequence — no special attention masks, no task-specific routing inside the body.

On the output side there are two prediction heads. The **text head** is the LLM's built-in language-model head: it predicts the next text token. The **image head** is a randomly initialized head that predicts the next VQ codebook ID. Which head fires depends on what kind of token the model is supposed to produce at that position, which is determined by the sequence structure of the task.

That is the whole trick. Two encoders, one body, two heads. The decoupling is surgical: it happens before the transformer and after it, never inside.

### How a sequence is actually formed

The thing that makes this work as one model rather than two is that both tasks are expressed as next-token prediction over a single interleaved sequence. The difference is only in how you lay out the sequence and which head reads each position.

For an **understanding** task, the sequence is `[SigLIP image features] + [question text tokens] + [answer text tokens]`. The image features occupy the front, the question follows, and the model autoregressively predicts the answer text — the text head fires on every answer position. The loss is the standard cross-entropy on the answer tokens. This is exactly how a normal vision-language model works; the only Janus-specific detail is that the image features come from SigLIP through the understanding adaptor.

For a **generation** task, the sequence is `[caption text tokens] + [image VQ tokens]`. The caption conditions the model, and then the model autoregressively predicts the VQ codebook IDs of the target image, one ID at a time, left to right, top to bottom — the image head fires on every image position. The loss is cross-entropy over the 16,384-way codebook at each image position.

Here is a compact sketch of how the two sequence layouts differ, in the kind of pseudocode you would actually find in the data collator:

```python
def build_understanding_example(image, question, answer, tokenizer, siglip, und_adaptor):
    # Understanding: image conditions, model predicts the text answer.
    img_feats = und_adaptor(siglip(image))        # [N_img, d_model], semantic
    q_ids = tokenizer.encode(question)
    a_ids = tokenizer.encode(answer)
    inputs_embeds = cat([img_feats,
                         embed_text(q_ids),
                         embed_text(a_ids)])
    # Supervise only the answer span with the TEXT head.
    labels = [IGNORE] * (len(img_feats) + len(q_ids)) + a_ids
    return inputs_embeds, labels, head="text"

def build_generation_example(caption, image, tokenizer, vq, gen_adaptor):
    # Generation: caption conditions, model predicts discrete image tokens.
    c_ids = tokenizer.encode(caption)
    vq_ids = vq.encode(image)                      # [N_pix, ] in [0, 16384)
    img_embeds = gen_adaptor(vq.embed(vq_ids))     # [N_pix, d_model]
    inputs_embeds = cat([embed_text(c_ids), img_embeds])
    # Supervise only the image span with the IMAGE head over the 16384 codebook.
    labels = [IGNORE] * len(c_ids) + vq_ids
    return inputs_embeds, labels, head="image"
```

The two examples flow through the *same* transformer weights. The body never knows or cares which task it is serving; it just predicts the next token in the embedding space it was handed. The encoders and heads carry all the task-specific machinery. That separation of concerns is what lets one set of transformer parameters serve both jobs without the encoder-level tug-of-war.

### Why the body can be shared but the encoder cannot

It is worth being precise about the asymmetry, because it is the crux of the whole design and it trips people up. Why is sharing the transformer body fine when sharing the encoder is catastrophic?

Because the transformer body operates on *already-projected* features, and the two adaptors have done the work of mapping each task's native representation into a common space. By the time a token reaches the transformer, a SigLIP feature and a VQ embedding both live in the LLM's `d_model`-dimensional space and look, to the attention mechanism, like generic sequence elements. The body learns a single, rich sequence-modeling function that is genuinely useful to both tasks — long-range dependency, compositional reasoning, conditioning on context. That function does *not* have to compromise, because it never sees raw pixels and never has to be simultaneously invariant and faithful. It sees vectors that the encoders have already specialized.

The encoder, by contrast, sits at the raw-pixel boundary where the granularity conflict is unavoidable. The information-destroying decision — keep semantics or keep detail — happens at encoding time and cannot be undone downstream. Once SigLIP has thrown away the couch's exact texture, no transformer can recover it; once VQ has discarded high-level concept structure, no transformer can reason over it cleanly. So the encoder *must* be specialized per task, and the body *can* be shared. Janus puts the split exactly where the conflict lives and nowhere else.

### A worked example of the granularity conflict

Make the conflict concrete with a single image: a photo of a red ceramic mug with the word "COFFEE" printed on it, sitting on a wooden desk next to a laptop.

Run that image through SigLIP. The encoder produces a sequence of patch features whose job is to make the image *retrievable by caption*. Pool them and you get something close to the embedding of "a red mug on a desk near a laptop." The features encode the concepts — mug, red, desk, laptop, the rough left-to-right layout — at a semantic level that lets a downstream VLM answer "what color is the mug?" (red) and "what is next to the laptop?" (a mug) and even, if the resolution allows, "what does the mug say?" (COFFEE). What those features do *not* encode is the exact RGB value of the red, the precise grain of the wood, the specific font of the lettering, or the pixel-level position of the mug's handle. SigLIP was trained to be invariant to all of that, because two photos of red mugs on desks should land near the same point regardless of those specifics. For understanding, this is exactly right. For generation, it is a disaster: if you tried to redraw the image from the SigLIP embedding, you would get *a* red mug on *a* desk, but not *this* mug, *this* desk, *this* font.

Now run the same image through the VQ tokenizer. It splits the 384×384 image into a 24×24 grid of patches (16× downsampling) and assigns each patch the nearest of 16,384 codebook entries. Each entry is a learned visual primitive — a particular gradient, a particular edge orientation, a particular texture cell. The resulting 576 IDs are a near-lossless recipe for *redrawing* the image: feed them back through the VQ decoder and you recover the mug, the wood grain, and the lettering with high fidelity. That is exactly what generation needs. But ask "what color is the mug?" of the raw VQ IDs and there is no clean answer — the concept "red mug" is smeared across dozens of texture-primitive IDs with no single token that *means* "mug." Reasoning over VQ tokens is reasoning over a JPEG's DCT coefficients: the information is all there, but in a form that is hostile to semantics.

This is why one encoder cannot serve both. SigLIP is the right answer for the question "what color is the mug?" and the wrong answer for "redraw this exact mug." VQ is the reverse. A shared encoder would have to land somewhere between "red mug on a desk" and "576 texture primitives," and there is no point on that line that is good at both — the midpoint is a representation that is too lossy to redraw faithfully and too low-level to reason over cleanly. Janus's decoupling lets the understanding path take the SigLIP end and the generation path take the VQ end, and the shared body — which sees only the *projected* features, already specialized — never has to occupy the bad middle.

### Inside the VQ generation path: why discrete IDs work for drawing

The generation encoder deserves a closer look, because the choice of a VQ tokenizer with a 16,384-entry codebook and 16× downsampling is not arbitrary — every number trades fidelity against sequence length, and Janus picked a specific point on that curve.

Start with the downsampling factor. A 16× spatial downsample turns a 384×384 image into a 24×24 grid, which is 576 tokens. That number is the sequence-length budget the generation path imposes on the transformer: every generated image costs 576 autoregressive decode steps. Drop to 8× downsampling and you would get a 48×48 grid — 2,304 tokens, four times the decode cost — in exchange for finer spatial detail. Go to 32× and you would get a 12×12 grid, 144 tokens, cheap but blocky. The 16× choice is the sweet spot where 576 tokens fits comfortably inside the 4,096-token context window alongside a caption, and the spatial resolution is enough for coherent 384×384 images. The downsampling factor is, in effect, a knob that sets how many decode steps an image costs, and Janus tuned it for the context budget.

Now the codebook size. With 16,384 entries, each image token carries $\log_2(16384) = 14$ bits of information — one of 16,384 visual primitives per patch. A larger codebook (say 65,536) would let each patch encode finer distinctions and improve reconstruction fidelity, but it makes the image head's softmax wider and the prediction problem harder, because the model now has to discriminate among more near-identical primitives. A smaller codebook (say 1,024) makes prediction easier but caps reconstruction quality — too few primitives to capture real-world texture diversity. The 16,384 choice, again from prior VQ work, balances "rich enough to redraw faithfully" against "discriminable enough that the image head can actually learn to predict the next ID." Here is the relationship laid out:

```python
def vq_token_budget(image_size=384, downsample=16, codebook=16384):
    grid = image_size // downsample          # 24 -> 24x24 spatial grid
    n_tokens = grid * grid                   # 576 autoregressive image tokens
    bits_per_token = math.log2(codebook)     # 14 bits = one of 16384 primitives
    return {
        "grid": (grid, grid),
        "tokens_per_image": n_tokens,        # 576 -> decode steps per image
        "bits_per_token": bits_per_token,    # 14 -> reconstruction fidelity knob
        "softmax_width": codebook,           # 16384 -> image-head prediction difficulty
    }
    # Smaller downsample -> more tokens, finer detail, costlier decode.
    # Larger codebook  -> finer detail, harder next-ID prediction.
```

The deeper point is that the discrete-token formulation makes generation *look exactly like* language modeling to the transformer body. The image head is a 16,384-way softmax, structurally identical to a language-model head over a 16,384-token vocabulary. The body predicts the next image ID the same way it predicts the next word — same attention, same cross-entropy loss, same sampling machinery. That structural identity is precisely why the decoupling can keep the body shared: once VQ has turned the image into a sequence of discrete IDs, generation *is* next-token prediction, and the body has nothing special to learn. The VQ tokenizer is the component that translates "draw an image" into "predict the next token in this 16,384-symbol language," and that translation is what makes the unified body possible.

### How Janus compares to the other unified designs

The decoupled-encoder move is one of several ways people have tried to make one model both read and draw. It helps to see where Janus sits among the alternatives, because the comparison sharpens *why* the split matters.

| Design | Visual front-end | Generation mechanism | The fundamental tradeoff |
|---|---|---|---|
| **Shared-encoder (Chameleon-style)** | One encoder for both tasks | Discrete tokens, autoregressive | Encoder forced to compromise; understanding suffers |
| **Generator + frozen VLM (stitched)** | Two separate models | Diffusion in a bolted-on module | No shared body, no transfer, two serving stacks |
| **Janus** | SigLIP (understand) + VQ (generate), decoupled | Discrete VQ tokens, autoregressive | Split encoder, shared body; simple to scale and serve |
| **JanusFlow** | SigLIP (understand) + ConvNeXt-in-VAE (generate) | Rectified flow inside the LLM | Continuous generation, no codebook ceiling; heavier inference |

The stitched approach — run a strong VLM for understanding and a separate diffusion model for generation, glued by a router — is the honest baseline, and it is sometimes the right call (see the closing section). But it buys task quality at the cost of *no transfer whatsoever*: the understanding model never makes the generation model more grounded, because they share nothing. The shared-encoder approach buys transfer and a single body at the cost of the granularity compromise. Janus is the design that refuses both bad trades: it keeps the single shared body (so transfer and amortized serving are on the table) while splitting only the one component — the encoder — where sharing is actively harmful. That is the whole conceptual contribution, and the table is the clearest way to see it: every row makes one sacrifice, and Janus's sacrifice (a second small encoder) is by far the cheapest.

### Second-order optimization: the adaptors are not an afterthought

A subtle gotcha: because the two encoders produce features in different native spaces — SigLIP's continuous semantic space and VQ's discrete codebook-embedding space — the two adaptors are doing real work, not just dimension matching. They are learning to make two very different representations *commensurable* inside the shared body. In the Janus training recipe, the first stage trains these adaptors (and the image head) while the SigLIP encoder and the LLM are frozen, precisely so the projectors learn to align the two visual streams into the LLM's space before the body starts adapting to them. Skip or shortchange that stage and the body has to simultaneously learn sequence modeling *and* absorb two un-aligned feature distributions, which is a much harder optimization. The adaptors are cheap parameters doing expensive alignment work; treat them as load-bearing.

## 2. Janus's three-stage training recipe

**Rule of thumb: in a model with frozen-then-unfrozen components, the stage boundaries are where you decide what learns to align to what. Get the order wrong and you waste compute teaching the body to compensate for unaligned projectors.**

The original Janus trains in three stages, and the logic of the staging is the logic of "align the cheap parts first, then let the expensive parts adapt."

**Stage I — train the adaptors and image head, freeze the encoders and LLM.** Only the understanding adaptor, the generation adaptor, and the randomly initialized image head update. The SigLIP encoder, the VQ tokenizer, and the entire LLM stay frozen. The goal is narrow: teach the two MLPs to project visual features into the LLM's embedding space, and teach the fresh image head to predict VQ tokens, all without disturbing the pretrained semantics of SigLIP or the language competence of the backbone. This is the alignment stage, and it is cheap because most parameters are frozen.

**Stage II — unified pretraining, unfreeze the LLM.** Now the LLM body joins the training, and the model sees a mixture of text, understanding, and generation data. This is where the transformer learns to be a unified sequence model that handles all three modalities. The encoders typically stay frozen (you do not want to drift SigLIP's semantics or the VQ codebook), but the body adapts to the projected features it now reliably receives because Stage I aligned them.

**Stage III — supervised fine-tuning, unfreeze almost everything.** Instruction-tuning data across all modalities, fine-tuning all parameters except the generation encoder (you keep the VQ tokenizer fixed so the codebook semantics stay stable). This is where the model learns to follow instructions, hold a conversation, and respond to the prompt formats it will see at inference.

The staging mirrors a principle that shows up all over multimodal training: **freeze the expensive, pretrained competence; train the cheap glue first; then let the body adapt; then polish with instructions.** It is the same instinct behind LLaVA-style projector-first training, applied to a model with two visual front-ends instead of one.

### What the 1.3B Janus actually achieved

Numbers ground the claim that decoupling works. Janus at 1.3B parameters posts a POPE score of 87.0, an MME-Perception score of 1338.0, GQA of 59.1, and MMBench of 69.4 on the understanding side — and on the generation side, GenEval around 61% and an MSCOCO-30K FID of 8.53. The headline comparison the paper draws: Janus-1.3B outperforms LLaVA-v1.5-7B on POPE, MMBench, SEED, and MM-Vet despite having **5.4× fewer parameters**. A 1.3B unified model beating a 7B understanding-specialized model on understanding benchmarks is the proof that the decoupling is not just "doesn't hurt" — it actively helps, because the understanding path finally gets a clean semantic encoder instead of a compromised shared one.

That result is the whole thesis in one line. The unified model is not paying an understanding tax for the privilege of also generating. It is *better* at understanding than a same-or-larger model that only understands, because the architecture stopped forcing the encoder to do two jobs.

## 3. JanusFlow: generation does not have to be discrete tokens

**Rule of thumb: "to live inside an LLM, images must become discrete tokens" is a convention, not a law. The transformer can predict a continuous velocity field just as easily as a discrete codebook ID — and continuous generation has no quantization ceiling.**

The original Janus generates images by autoregressively predicting discrete VQ tokens. This is clean and it fits the LLM's next-token machinery perfectly, but it inherits the limits of VQ: the codebook has 16,384 entries, the image is downsampled 16×, and every token can only ever name one of those 16,384 quantized cells. There is a ceiling on fidelity baked into the discretization, and there is an inherent serialization — you predict the image one token at a time.

JanusFlow ([arXiv 2411.07975](https://arxiv.org/abs/2411.07975), CVPR 2025) asks whether generation inside the unified transformer has to be discrete autoregression at all, and answers no. It replaces the VQ-token generation path with **rectified flow** integrated directly into the LLM, while keeping autoregression for understanding. Same body, two generation paradigms across the two Janus variants.

![Two ways to wire generation into the same transformer: Janus predicts discrete VQ IDs while JanusFlow predicts a velocity field.](/imgs/blogs/janus-pro-decoupled-visual-encoding-3.webp)

The before-after above is the contrast. On the left, the Janus path: a VQ tokenizer maps the image to 16,384-way IDs, the LLM predicts the next ID left to right, the image head is a softmax over the codebook, and quantization caps how much detail any single token can carry. On the right, the JanusFlow path: a ConvNeXt encoder operates in a pretrained SDXL-VAE latent space, the LLM predicts a *velocity* $v(z_t, t)$ at a noised latent $z_t$, an ODE integrates that velocity to walk noise toward the image, and the latents are continuous — no codebook ceiling.

### Rectified flow in one paragraph of math

Rectified flow is a clean way to learn a generative model as a velocity field. You define a straight-line interpolation between a Gaussian noise sample $z_0 \sim \pi_0$ and a data point $x$:

$$z_t = t \cdot x + (1 - t) \cdot z_0, \qquad t \in [0, 1]$$

At $t=0$ you are at pure noise; at $t=1$ you are at the data. The time-derivative of this path is simply $x - z_0$, a constant velocity. So you train a network $v_\theta(z_t, t)$ to predict that velocity at any point along the path:

$$\mathcal{L} = \mathbb{E}_{t, x, z_0}\left[\; \lVert v_\theta(z_t, t) - (x - z_0) \rVert^2 \;\right]$$

At inference you start from noise $z_0$ and integrate the ODE $\frac{dz_t}{dt} = v_\theta(z_t, t)$ forward in time with simple Euler steps, $z_{t+dt} = z_t + v_\theta(z_t, t)\, dt$, until you reach $t=1$. The crucial point for JanusFlow: **the velocity predictor is the LLM itself.** You encode the noised latent $z_t$, concatenate a time embedding, run it through the same transformer that does understanding, and read off a velocity instead of a token logit. No new generative module, no separate diffusion U-Net bolted on — the LLM *is* the flow model.

```python
@torch.no_grad()
def sample_janusflow(prompt, llm, g_enc, g_dec, vae, steps=30, w=2.0):
    # JanusFlow generation: the same transformer predicts a velocity field.
    cond = llm.embed_text(prompt)               # text conditioning tokens
    z = torch.randn(latent_shape)               # z_0 ~ N(0, I) in VAE latent space
    dt = 1.0 / steps
    for i in range(steps):
        t = i * dt
        h = g_enc(z)                            # ConvNeXt encode latent -> tokens
        # Classifier-free guidance: blend conditional and unconditional velocity.
        v_cond = llm.velocity(cat([cond, h, time_embed(t)]))
        v_uncond = llm.velocity(cat([null_cond, h, time_embed(t)]))
        v = w * v_cond + (1.0 - w) * v_uncond   # w >= 1 sharpens prompt adherence
        z = z + v * dt                          # Euler step along the ODE
    latent = g_dec(z)
    return vae.decode(latent)                   # SDXL-VAE -> pixels
```

The classifier-free guidance line is the same trick you know from diffusion: blend the prompt-conditioned velocity with an unconditional one, with weight $w \ge 1$ pushing harder toward prompt adherence. During training, JanusFlow drops the text prompt on 10% of examples so the model learns an unconditional velocity to blend against.

### Why the straight-line path is the whole point

It is worth dwelling on *why* rectified flow uses a straight-line interpolation rather than the curved noise schedule of classic diffusion, because the straightness is the entire efficiency argument. In standard diffusion, the path from noise to data is a curve dictated by a variance schedule, and the model's learned vector field at any point does not point straight at the data — it points along the local tangent of that curve. To follow a curve accurately you need many small steps, which is why classic diffusion samplers run dozens to hundreds of denoising steps.

Rectified flow deliberately trains the model so the *target* trajectory is a straight line: $z_t = t \cdot x + (1-t) \cdot z_0$ is a line segment, and its velocity $x - z_0$ is constant along it. If the learned velocity field were perfectly straight, you could jump from noise to data in a single Euler step — one function evaluation. In practice the field is only approximately straight because each noise sample maps to a different data point and the paths cross, so you take a handful of steps (the code uses 30, but rectified flow is designed to work well at far fewer), but the principle holds: the straighter the learned paths, the fewer steps sampling needs. This is the property that makes rectified flow attractive *inside an LLM*, where each step is a full forward pass through a 1.3B-or-larger transformer — and with CFG, two passes. Halving the step count halves your generation latency, so a generation primitive that is accurate at 10–30 steps instead of 100+ is a serving win, not just an academic nicety.

The contrast with the discrete-VQ path is instructive. The VQ path's cost is fixed at 576 autoregressive decode steps (one per image token), each one a single forward pass — predictable, streamable, but not reducible. The rectified-flow path's cost is `steps × 2` forward passes over the *whole* latent at once (CFG doubles it), which is fewer total passes than 576 when `steps` is small, but each pass processes the full image rather than one token. The two generation paths trade "many cheap sequential steps" (VQ) against "few expensive parallel steps" (flow). Which one wins on latency depends on your batch size and hardware, which is part of why both paradigms coexist in the Janus family rather than one strictly dominating.

### Decoupled encoders survive the paradigm change

The detail that matters for *this* article's thesis: JanusFlow keeps the decoupling. The understanding path still uses SigLIP-Large-Patch/16 (~300M parameters) as the semantic encoder. The generation path uses separate, from-scratch ConvNeXt blocks (an encoder $g_{enc}$ and a decoder $g_{dec}$, ~70M total) operating in a pretrained SDXL-VAE latent space rather than in pixel space. Two encoders, one body — exactly the Janus principle — only the generation encoder swapped from a discrete VQ tokenizer to a continuous ConvNeXt-in-VAE-latent setup, and the generation *head* swapped from a codebook softmax to a velocity regression.

JanusFlow adds one more idea worth flagging: **representation alignment** (REPA-style). During generation training, it aligns the LLM's intermediate features with the *understanding* encoder's features, via a loss like

$$\mathcal{L}_{\text{align}} = -\,\mathbb{E}\left[\,\text{sim}\big(\text{stopgrad}(f_{\text{enc}}(x)),\; h_\phi(q_\theta(z_t))\big)\right]$$

where $f_{\text{enc}}$ is the frozen SigLIP encoder, $q_\theta(z_t)$ is the LLM's internal representation of the noised latent, and $h_\phi$ is a small trainable MLP projecting the LLM features to match SigLIP's dimension. In plain terms: while the model learns to *draw*, it is gently pulled to keep its internal picture of the image *semantically coherent* with what the understanding encoder would see. The generation path borrows the understanding path's sense of meaning. This is the unification dividend finally paying off — generation made more grounded by understanding — and it only becomes expressible because the two paths share a body and you can align across them.

The numbers say the swap is worth it. JanusFlow posts GenEval 0.63 (vs SDXL 0.55, vs the original Janus 0.61), DPG-Bench 80.09%, MJHQ-FID-30k of 9.51 (better than Janus's 10.10), and on understanding MMBench 74.9 and SeedBench 70.5 — all from a 1.3B-parameter DeepSeek-LLM body with 24 transformer blocks. Continuous generation, integrated as a velocity field inside the LLM, beats the discrete-token version on its own generation benchmarks while keeping understanding strong.

### When discrete tokens are still the right call

Do not read this as "rectified flow always wins." Discrete-token generation has real operational advantages: it reuses the LLM's exact next-token sampling machinery (temperature, top-p, the works), it is trivially interruptible and streamable token by token, and it composes naturally with text in a single autoregressive sequence — you can interleave generated text and generated image tokens in one decode loop. Janus-Pro, the production-scale model we turn to next, sticks with the discrete VQ-token path precisely because it is simpler to scale and serve. Rectified flow is the better *generation* primitive on the benchmarks; discrete AR is the better *systems* primitive. Both are legitimate, and the fact that the same decoupled body supports both is the point.

## 4. Janus-Pro: three orthogonal levers, not a new architecture

**Rule of thumb: when a research-scale model is good but not production-grade, the highest-leverage changes are usually in the training schedule and the data, not the architecture. Janus-Pro changed almost nothing about the model and everything about how it was trained.**

Janus-Pro ([arXiv 2501.17811](https://arxiv.org/abs/2501.17811), Jan 2025) is not a redesign. It is the original Janus architecture — same decoupled SigLIP-plus-VQ encoders, same single autoregressive body, same two heads — improved along three independent axes: a reorganized training schedule, scaled-up data, and a larger model. The encoders are unchanged in kind: **SigLIP-Large-Patch16-384** for understanding, and a **VQ tokenizer with a 16,384-entry codebook and 16× downsampling** for generation, at 384×384 image resolution. What changed is everything *around* the architecture.

![Janus-Pro is Janus plus three orthogonal changes: a better training schedule, more data, and a bigger transformer body.](/imgs/blogs/janus-pro-decoupled-visual-encoding-4.webp)

The grid above lays out the three levers and what each one concretely does. The fact that they are *orthogonal* is the engineering lesson: you can reason about each independently, and each contributes a measurable chunk of the final gain. There is no clever architectural insight hiding in Janus-Pro. There is disciplined training engineering, which is frankly rarer and more valuable.

Let me take the three levers in turn, because each one carries a transferable lesson for anyone training a unified model.

## 5. Lever one: reorganizing the training stages

**Rule of thumb: a training stage that mixes a warm-up objective with the real objective is wasting capacity on the warm-up for the entire stage. Separate them — front-load the warm-up, then run the real objective clean.**

The most surgical of Janus-Pro's changes is the stage reorganization, and it is a beautiful example of how moving *when* something trains changes *how efficiently* it trains.

In the original Janus, Stage I trained the adaptors and image head on a mixture that included ImageNet data — used to teach the generation path basic "pixel dependency," the low-level structure of how image tokens relate spatially. But the original schedule kept ImageNet data flowing into Stage II as well, where it consumed a large fraction (around two-thirds) of the generation-data budget. So Stage II — the long, expensive unified-pretraining stage — was spending most of its image-generation capacity re-learning ImageNet class-conditional pixel structure instead of learning the real target task: open-ended, prompt-driven text-to-image generation.

![Reordering the three training stages buys cheaper convergence: ImageNet pixel-dependency learning moves up into Stage I and Stage II goes pure text-to-image.](/imgs/blogs/janus-pro-decoupled-visual-encoding-5.webp)

The timeline above shows the fix. Janus-Pro **extends Stage I** to train longer on ImageNet, so the generation path fully absorbs pixel-dependency modeling while the LLM is still frozen and the only thing learning is the cheap adaptor-and-head front-end. Then it **drops ImageNet entirely from Stage II**, so the expensive unified-pretraining stage spends 100% of its generation budget on actual text-to-image data. Stage III adjusts the data mixture ratios — the multimodal-understanding : pure-text : text-to-image proportions shift from the old 7:3:10 to 5:1:4 — to rebalance the instruction-tuning blend.

The insight is that pixel-dependency learning is a *warm-up* objective and text-to-image is the *real* objective, and the original schedule conflated them across Stage II. Pixel structure is largely class-agnostic spatial regularity; you can learn most of it cheaply, early, with the body frozen. Once it is learned, you do not need to keep paying for it in the expensive stage. By front-loading it into the cheap stage and clearing it out of the expensive one, Janus-Pro gets the same pixel competence for less compute and frees the expensive stage to focus entirely on the task you actually care about. Same total objective, better allocation. That is the whole move, and it is the kind of thing you only see when you look hard at *what each stage is actually spending its gradient steps on.*

### A worked example of the budget shift

Put numbers on the reallocation to feel the leverage. Stage II is the expensive stage: the LLM body is unfrozen, so every gradient step backpropagates through the full transformer. Suppose Stage II runs for a fixed budget of $S$ generation steps. Under the old schedule, ImageNet occupied roughly two-thirds of the generation data, so the gradient steps that actually touched the *target* objective — prompt-driven text-to-image — were about $\frac{1}{3} S$. The other $\frac{2}{3} S$ steps backpropagated through the full unfrozen body to re-teach class-conditional pixel structure that a frozen-body stage could have taught far more cheaply.

The cost asymmetry is the punchline. A gradient step in frozen-body Stage I updates only the adaptors and image head — call it a few hundred million parameters of update. A gradient step in unfrozen-body Stage II updates the full 7B (or 1.5B) transformer. So spending $\frac{2}{3} S$ of the *expensive* steps on pixel dependency is paying full-transformer prices for a job the cheap stage does for pennies. Janus-Pro's fix moves that pixel-dependency learning into the extended Stage I — where the per-step cost is a fraction of a full-body step — and lets all $S$ expensive steps land on the real objective. Roughly speaking, the target-task gradient budget in the expensive stage triples (from $\frac{1}{3} S$ to $S$) at no increase in total cost, because the displaced work moves to a cheaper stage. That is the entire reason convergence on real text-to-image speeds up, and it is a clean instance of a principle that generalizes: *expensive gradient steps should only ever touch the objective you cannot learn more cheaply elsewhere.*

```python
budget = {
    # Old Janus: ImageNet leaks into the expensive (unfrozen-body) Stage II.
    "stage2_old": {"imagenet": 0.66, "real_text_to_image": 0.34},
    # Janus-Pro: Stage I extended; Stage II is pure target objective.
    "stage1_new": {"imagenet": 1.0},               # frozen body -> cheap per-step
    "stage2_new": {"real_text_to_image": 1.0},     # unfrozen body -> all steps on-target
}
print(budget["stage2_new"])  # ~3x more on-target steps, same total cost
```

### Second-order optimization: stage boundaries are budget decisions

The generalizable lesson here is to treat every training stage as a *budget* and ask what fraction of each stage's gradient steps go to the objective you ultimately care about. A stage that is 66% warm-up data is 66% wasted on the warm-up *if the warm-up could have been finished earlier and more cheaply.* The fix is rarely "train longer" — it is "move the warm-up to where it is cheap and let the expensive stage run clean." This applies far beyond Janus: any time you have a frozen-body stage and an unfrozen-body stage, the warm-up objectives belong in the frozen stage, and the real objective belongs alone in the unfrozen one.

## 6. Lever two: scaling data, and the synthetic-aesthetic surprise

**Rule of thumb: more data only helps if it is the right data. Janus-Pro's most interesting finding is that for text-to-image, clean synthetic data beats noisy real data so decisively that a 1:1 synthetic mix is the stable choice.**

The second lever is data scaling, and it has two parts. On the understanding side, Janus-Pro adds approximately **90 million** samples — image captions, document-understanding datasets, the usual ingredients that make a VLM literate. That is a straightforward "more high-quality understanding data makes understanding better" move, and it does.

The generation side is where it gets interesting. Janus-Pro adds approximately **72 million synthetic aesthetic images**, bringing the real-to-synthetic ratio in the generation data to **1:1**. And the reason for that 1:1 ratio is a genuinely non-obvious empirical finding.

![Synthetic aesthetic data converges faster than noisy web pairs; DeepSeek mixes synthetic and real 1:1 to stabilize text-to-image training.](/imgs/blogs/janus-pro-decoupled-visual-encoding-6.webp)

The before-after above captures it. Real web text-to-image pairs are the obvious data source, but they are *noisy*: captions are frequently misaligned with the image (alt-text that describes the page, not the picture), aesthetic quality is wildly uneven, and a large fraction of the images are low-quality. Training on this data, the loss oscillates and convergence is slow, because the model is constantly being pulled toward mislabeled, ugly targets. Synthetic aesthetic data — images generated and curated to have tight caption alignment and uniformly high visual quality — flips all three: captions match, aesthetics are consistent, and the model converges *faster* and produces *more stable* output.

The paper's own words: the model "converges faster when trained on synthetic data, and the resulting text-to-image outputs are not only more stable but also exhibit significantly improved aesthetic quality." This is the kind of result that contradicts the default instinct — "synthetic data is a crutch, real data is ground truth" — and it contradicts it for a concrete, mechanistic reason. For text-to-image, the *bottleneck is caption-image alignment*, and synthetic data can be constructed to have near-perfect alignment, whereas real web data is fundamentally noisy at the caption level. When your training signal is "match this caption to this image," and your real captions are 30% garbage, clean synthetic pairs are not a crutch — they are a *higher-quality training signal* on the exact axis that matters.

The 1:1 ratio is the pragmatic synthesis: keep enough real data to stay grounded in the true diversity of the visual world, but lean equally on synthetic data for the stable, aligned, aesthetically consistent signal that drives fast convergence. It is a hedge, and the fact that DeepSeek landed on exactly 1:1 rather than, say, 9:1 real tells you how much they trusted the synthetic data.

### Second-order optimization: synthetic data is a signal-quality lever, not a quantity lever

The trap is to read "add 72M synthetic images" as a quantity story — more data, bigger model, better results. It is really a *signal-quality* story. The synthetic data helps not because there is more of it but because each synthetic example carries a cleaner caption-image correspondence than the median real example. If you take one lesson from Janus-Pro's data work into your own training, make it this: when your task is alignment between two modalities, and one modality's real-world annotations are noisy, synthetic data that you can construct to be perfectly aligned is often a *better* training signal than real data, not a worse one. Audit your data for alignment noise before you assume you need more of it.

## 7. Lever three: scaling the model, and the results

**Rule of thumb: architecture changes earn the headlines; the boring trio of better schedule, cleaner data, and more parameters earns the benchmark wins. Janus-Pro is a monument to the boring trio.**

The third lever is the most conventional: scale the model. Janus-Pro keeps a 1.5B variant (the spiritual successor to the original 1.3B) and adds a **7B** variant, both built on DeepSeek-LLM (1.5B / 7B) with the 4096-token context window. The 7B model trained for roughly 14 days on 32 nodes of A100-40GB GPUs using DeepSeek's HAI-LLM distributed training framework. Nothing exotic — more parameters, more compute, the standard scaling story.

What the three levers buy together is a unified model that does not just hold its own but *leads*, on both generation and understanding, against models that specialize in only one.

![Janus-Pro-7B beats dedicated generators and the old Janus; one unified model leads on generation benchmarks and on understanding.](/imgs/blogs/janus-pro-decoupled-visual-encoding-7.webp)

The matrix above is the scoreboard. On **GenEval**, the compositional text-to-image benchmark, Janus-Pro-7B scores **0.80** — beating **DALL-E 3 at 0.67** and **SD3-Medium at 0.74**, two dedicated image generators with no understanding capability at all. On **DPG-Bench**, the dense-prompt-following benchmark, it scores **84.19**. And on **MMBench**, the understanding benchmark, it scores **79.2**, a massive jump over the original Janus's **69.4**.

Sit with those numbers for a second, because they are the entire argument of this article made quantitative. A *unified* model — one that also generates images — beats *dedicated* image generators on the premier image-generation benchmark. And the same model improves understanding by nearly ten MMBench points over its predecessor. The decoupled architecture means generation capability is not bought at the cost of understanding, and the training improvements mean both rise together. This is what "unified done right" looks like: no tax, only transfer.

The comparison to DALL-E 3 and SD3-Medium deserves emphasis because it is so counterintuitive. Those models do *one thing* — generate images — and pour all their capacity into it. Janus-Pro-7B splits its attention across understanding and generation and *still wins* on GenEval. That is only possible because (a) the decoupled VQ generation encoder gives the generation path the detail it needs, (b) the reorganized schedule spent the expensive stage entirely on real text-to-image, and (c) the synthetic-aesthetic data gave a cleaner training signal than the noisy web pairs the competition often trains on. Three orthogonal levers, each contributing, compounding into a generation model that beats the specialists while also being a strong VLM.

## 8. Serving one body for both jobs

**Rule of thumb: the operational beauty of a unified model is one set of weights and one serving stack. The cost is a mode flag that decides which encoder runs and which head decodes — get that routing right and inference is genuinely two-for-one.**

The unification pays off at serving time too, and it is worth seeing exactly how one transformer body dispatches both jobs, because it determines your inference architecture.

![One transformer body serves both jobs at inference; the mode flag picks which encoder runs and which head decodes.](/imgs/blogs/janus-pro-decoupled-visual-encoding-8.webp)

The diagram above is the inference-time view of the same decoupled design. A **VQA request** (image plus question) routes through the SigLIP encoder in understand mode, hits the shared DeepSeek-LLM body, and decodes through the text head, streaming an answer. A **text-to-image request** (prompt only) routes through the VQ embedding table in generate mode, hits the *same* body, and decodes through the image head, sampling VQ IDs that the VQ decoder turns back into a 384×384 image. The body is identical across both paths; the encoder and head selection is the entire branch.

In serving code, the dispatch looks roughly like this:

```python
class JanusServer:
    def __init__(self, ckpt):
        self.siglip = load_siglip(ckpt)        # understanding encoder
        self.vq = load_vq_tokenizer(ckpt)      # generation encoder + decoder, 16384 codebook
        self.und_adaptor = load(ckpt, "und_adaptor")
        self.gen_adaptor = load(ckpt, "gen_adaptor")
        self.body = load_deepseek_llm(ckpt)    # SHARED transformer body
        self.text_head = self.body.lm_head     # built-in
        self.image_head = load(ckpt, "image_head")

    def understand(self, image, question, max_new=512):
        feats = self.und_adaptor(self.siglip(image))
        seq = cat([feats, self.body.embed_text(question)])
        # Autoregressive decode through the TEXT head, ordinary LLM sampling.
        return self.body.generate(seq, head=self.text_head, max_new=max_new)

    def generate_image(self, prompt, cfg_scale=5.0, temperature=1.0):
        seq = self.body.embed_text(prompt)
        # Autoregressively sample VQ IDs through the IMAGE head over the codebook.
        ids = self.body.generate(seq, head=self.image_head,
                                 vocab=16384, cfg_scale=cfg_scale,
                                 temperature=temperature)
        return self.vq.decode(ids)             # 16384-codebook IDs -> 384x384 pixels
```

One model object, two entry points, one shared body doing the heavy lifting in both. The encoders and heads are small relative to the body; the expensive parameters are amortized across both tasks. That is the operational dividend of unification: you ship and serve one model and get a VLM and a text-to-image generator out of it, with a memory footprint barely larger than either alone. Run the arithmetic and the saving is concrete: two specialized 7B models would cost two full sets of transformer weights in memory, while Janus-Pro pays for one shared body plus two small encoders and two small heads — roughly the footprint of a single 7B model rather than two. For any deployment where GPU memory is the binding constraint, that halving is the difference between fitting both capabilities on one card and needing two.

### Second-order optimization: the heads have different decode dynamics

A practical gotcha when you build this: the two heads have *very* different decode characteristics, and your serving infrastructure has to account for it. The text head produces variable-length output (an answer is however long it is) and benefits from all the usual LLM-serving machinery — KV-cache reuse, continuous batching, speculative decoding. The image head produces *fixed-length* output: an image is always the same number of VQ tokens, every time, so you know your sequence length up front and can size buffers exactly. The image decode also benefits from classifier-free guidance, which means running the body twice per step (conditional and unconditional) and blending — doubling the per-step compute relative to a text decode. If you batch text and image requests together naively, the image requests' CFG passes and fixed long length will dominate your latency budget in surprising ways. Separate the two request types into different batches with different decode configs. The shared body makes them *look* identical; their decode economics are not.

## Case studies from production

Theory is cheap. Here are the concrete situations where the decoupled-encoder design either saves you or bites you, drawn from the kinds of failures that actually happen when you build on this family of models.

### 1. The shared-encoder understanding gap

A team builds a unified model with a single shared vision encoder — the Chameleon-style design. Generation is fine; understanding lands four points below a same-size dedicated VLM on MMBench. They spend three weeks tuning the LLM: more instruction data, better projector, higher learning rate on the body. Nothing closes the gap. The wrong first hypothesis is always "the body isn't learning to understand." The actual root cause is that the shared encoder is optimized to preserve reconstructable detail for generation, which means it is *not* optimized for the semantic invariance understanding needs — the encoder is the ceiling, and no downstream change lifts it. The fix is the Janus move: split the encoder, give understanding a semantic SigLIP encoder of its own. The lesson is that a structural ceiling does not respond to downstream tuning; when an understanding gap survives every reasonable LLM-side intervention, suspect the encoder.

### 2. The ImageNet-in-Stage-II compute leak

A team reproducing a Janus-style schedule notices their unified-pretraining stage is converging on text-to-image far slower than expected, even with plenty of generation data in the mix. The wrong hypothesis is "we need more text-to-image data." The actual root cause is the original schedule's flaw that Janus-Pro fixed: ImageNet pixel-dependency data is occupying two-thirds of the generation budget *in the expensive stage*, so most generation gradient steps are re-learning class-conditional pixel structure instead of prompt-driven generation. The fix is to front-load ImageNet into the frozen-body Stage I and clear it out of Stage II entirely. After the change, the expensive stage spends 100% of its generation budget on the real task and convergence speeds up dramatically. The lesson: audit *what each stage's gradient steps actually optimize*, not just how much data each stage sees.

### 3. The noisy-caption convergence stall

A team scales their text-to-image data by scraping a large web corpus of image-alt-text pairs. Loss oscillates, aesthetic quality is poor, and adding more of the same data makes it worse, not better. The wrong hypothesis is "we need an even bigger corpus." The actual root cause is caption-image misalignment: a large fraction of web alt-text describes the page or is keyword spam, so the model is being trained to match captions to images that the captions do not describe. The fix is the Janus-Pro synthetic-aesthetic insight — mix in synthetic data with tight caption alignment, up to a 1:1 ratio, and convergence stabilizes while aesthetics jump. The lesson: when your task is cross-modal alignment, the *cleanliness* of the alignment signal matters more than the *quantity* of data, and synthetic data engineered for alignment can beat real data on the exact axis that bottlenecks you.

### 4. The frozen-VQ-codebook drift

A team fine-tunes a Janus-style model on a domain-specific generation task and, in their enthusiasm, unfreezes everything including the VQ tokenizer to "let it adapt to our images." Generation quality on their domain improves slightly, but the model's general generation capability degrades and old prompts start producing artifacts. The wrong hypothesis is "we overfit to the domain." The actual root cause is codebook drift: unfreezing the VQ tokenizer shifts the meaning of the 16,384 codebook entries, which invalidates everything the image head learned about predicting them and everything the body learned about conditioning on them. The Janus recipe deliberately keeps the generation encoder frozen in Stage III for exactly this reason. The fix is to re-freeze the VQ tokenizer and fine-tune only the body and heads. The lesson: the discrete codebook is a *contract* between the encoder, the body, and the head; renegotiating it mid-training breaks both downstream consumers.

### 5. The CFG latency surprise

A team deploys Janus-Pro behind a single inference endpoint that batches all multimodal requests together. Understanding requests are fast; image-generation requests are slow; and when both arrive together, *everything* slows down. The wrong hypothesis is "the image head is just slow." The actual root cause is that classifier-free guidance runs the body twice per decode step (conditional and unconditional) for generation, and image generation produces a fixed, long sequence of VQ tokens — so a single image request costs roughly twice the per-step compute over a long fixed length, and naive batching lets it dominate the shared batch's latency. The fix is to route understanding and generation requests into separate batches with separate decode configurations, as in the serving sketch above. The lesson: a shared body does not mean shared decode economics; the two heads have fundamentally different per-step costs and output-length distributions, and your batching has to respect that.

### 6. The rectified-flow integration temptation

A team loves JanusFlow's quality numbers and decides to bolt rectified-flow generation onto their existing discrete-token Janus deployment "to get the best of both." They underestimate the work. The wrong hypothesis is "it's just a different head." The actual root cause is that the generation *encoder* changes too — JanusFlow replaces the VQ tokenizer with a ConvNeXt-in-SDXL-VAE-latent setup, and the generation path goes from discrete-token prediction to continuous velocity regression with its own ODE solver, CFG schedule, and representation-alignment loss. You are not swapping a head; you are swapping the entire generation half of the model, including the loss, the encoder, the decoder, and the inference loop. The fix is to treat the two generation paradigms as distinct models that happen to share a body design, and to pick one per deployment rather than trying to run both. The lesson: the decoupling makes it *possible* to swap the generation path, but "possible" is not "free" — the generation encoder, head, loss, and inference loop all move together.

### 7. The 384×384 resolution ceiling

A team builds a document-understanding product on Janus-Pro and finds that dense, small text in scanned documents reads poorly. The wrong hypothesis is "the model can't do OCR." The actual root cause is the fixed 384×384 input resolution: the SigLIP-Large-Patch16-384 understanding encoder ingests images at 384×384, which is fine for natural scenes but throws away the spatial resolution needed to resolve small dense text. The fix is either to tile the document into 384×384 crops and aggregate, or to reach for a model purpose-built for high-resolution optical context — the same problem DeepSeek tackles head-on with optical context compression in [DeepSeek-OCR](/blog/machine-learning/computer-vision/deepseek-ocr-optical-context-compression). The lesson: a unified model's encoder resolution is a hard ceiling on what fine spatial detail it can read, and 384×384 is a deliberate efficiency choice, not an oversight — when your task needs more pixels, the encoder, not the body, is your constraint.

### 8. The "do I even need unified" decision

A team needs both a VLM and a text-to-image generator for a product and reflexively reaches for a unified model because "one model is simpler." After building it, they realize their two use cases never actually interact — the VLM answers support tickets, the generator makes marketing thumbnails, and the two never appear in the same request. The wrong hypothesis is "unified is always simpler." The actual root cause is that the operational simplicity of one model is real but the *training and evaluation* complexity of a unified model is higher — you have two objectives to balance, two data pipelines, two sets of benchmarks, and a harder debugging story when one task regresses. The fix, in their case, was to ship two specialized models, each simpler to train and evaluate, since they gained nothing from sharing a body. The lesson: unification's payoff is *transfer* between the tasks (understanding grounding generation, as in JanusFlow's representation alignment) and *amortized serving cost*; if your tasks never interact and serving cost is not your bottleneck, two specialized models may be the simpler engineering choice.

### 9. The adaptor-skip cold start

A team fine-tuning a Janus-style model on a new domain wants to save time and decides to start Stage II directly — unfreeze the body from the first step, skip the cheap Stage I that aligns the two adaptors. They reason that the adaptors will "learn along the way." Training is unstable for thousands of steps, generation output is garbage early on, and understanding accuracy crawls. The wrong hypothesis is "the learning rate is too high." The actual root cause is that the body is being asked to do two hard jobs at once: learn sequence modeling *and* absorb two un-aligned feature distributions whose projectors are still random. With random adaptors, the SigLIP and VQ features arrive in the body's input space as noise, so the body's early gradients are dominated by trying to make sense of garbage rather than learning the task. The fix is to honor the staging: run the cheap frozen-body Stage I first so the adaptors align the two visual streams, *then* unfreeze. The lesson: the alignment stage is not optional polish; it is what makes the expensive stage's gradients meaningful. Skipping a frozen-body warm-up to "save time" usually costs more time than it saves, because the expensive stage spends its early budget cleaning up the mess the warm-up would have prevented.

### 10. The benchmark that rewards the wrong thing

A team optimizes their unified model hard against FID on MSCOCO and ships when FID looks great, only to find users complain that generated images do not follow prompts well. The wrong hypothesis is "FID is a solved problem, the model is fine." The actual root cause is that FID measures distributional similarity to a reference set — it rewards realistic-looking images regardless of whether they match the prompt — so a model can post a strong FID while ignoring half the prompt. This is exactly why Janus-Pro reports GenEval (compositional prompt-following: does the image contain the right objects, counts, colors, and spatial relations?) and DPG-Bench (dense prompt adherence) rather than leaning on FID alone. Janus-Pro-7B's 0.80 GenEval and 84.19 DPG-Bench are *prompt-following* numbers, and they are the ones that correlate with user-perceived quality. The fix is to evaluate against compositional, prompt-following benchmarks, not just distributional ones. The lesson: pick the benchmark that measures the thing your users actually care about; FID measures realism, GenEval measures obedience, and for an instruction-following generator obedience is the one that ships a good product.

## When to reach for a decoupled unified model, and when not to

The Janus design is excellent, but it is not the answer to every multimodal problem. Here is how I decide.

**Reach for a Janus-style decoupled unified model when:**

- You genuinely need *both* image understanding and image generation from one system, and especially when they appear in the same workflow (read this chart, then draw a corrected version).
- You want the transfer dividend — generation grounded by understanding, as in JanusFlow's representation alignment — not just two capabilities bolted together.
- Serving cost matters and you would rather amortize one transformer body across both jobs than pay for two separate models in memory.
- You are building on top of a strong text LLM backbone (the DeepSeek-LLM lineage here) and want to inherit its reasoning for the understanding side. The same backbone family's efficiency work — [FP8 training with multi-token prediction and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) and [multi-head latent attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — is what makes scaling the shared body to 7B economical.
- You have the data discipline to feed two front-ends: clean semantic understanding data *and* well-aligned (ideally partly synthetic) generation data.

**Skip it — and reach for two specialized models — when:**

- Your two tasks never interact and you gain no transfer; two specialized models are simpler to train, evaluate, and debug (case study 8).
- You need state-of-the-art on *only one* of the two tasks and have no use for the other; a dedicated model will usually edge a unified one on a single axis, even though Janus-Pro shows the gap can invert.
- Your understanding task needs high input resolution — dense document OCR, fine medical imaging — that a 384×384 encoder cannot serve; reach for a high-resolution-specialized model instead (case study 7).
- You lack the training infrastructure to run a multi-stage, frozen-then-unfrozen schedule across two visual front-ends; the recipe is the hard part, and getting the stage boundaries and data mixtures wrong wastes more compute than a simpler design would.
- You are early-stage and just need *a* working VLM or *a* working generator; ship the simple thing first and unify when the transfer dividend becomes a real requirement, not a speculative one.

The through-line of the entire Janus story is a single principle that generalizes well beyond this model family: **put the architectural split exactly where the conflict lives, and nowhere else.** The conflict between understanding and generation lives at the visual-encoding boundary, where one representation cannot be simultaneously invariant and faithful. So Janus splits the encoder and shares everything downstream of it. Get that one placement right and the rest — a single transformer body, two heads, three orthogonal training improvements — falls into place, and you get a unified model that beats the specialists at their own benchmarks. Get it wrong, share the encoder, and you get the blurry midpoint that quietly caps every unified model that came before.

## Further reading

- **Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation** — [arXiv 2410.13848](https://arxiv.org/abs/2410.13848) (CVPR 2025). The original paper; the decoupled-encoder thesis and the 1.3B results.
- **JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation** — [arXiv 2411.07975](https://arxiv.org/abs/2411.07975) (CVPR 2025). Rectified flow inside the LLM, plus the representation-alignment trick.
- **Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling** — [arXiv 2501.17811](https://arxiv.org/abs/2501.17811) (Jan 2025). The three-lever scale-up; the synthetic-aesthetic data finding and the GenEval/MMBench numbers.
- **Chameleon: Mixed-Modal Early-Fusion Foundation Models** — the shared-encoder design Janus reacts against; read it to feel the compromise Janus avoids.
- [DeepSeek-OCR: optical context compression](/blog/machine-learning/computer-vision/deepseek-ocr-optical-context-compression) — the sibling line of work on high-resolution optical context, relevant when 384×384 is not enough.
- [DeepSeek-V3: FP8 training, multi-token prediction, and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) and [multi-head latent attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — the backbone-efficiency work that makes scaling the shared Janus body to 7B economical.
