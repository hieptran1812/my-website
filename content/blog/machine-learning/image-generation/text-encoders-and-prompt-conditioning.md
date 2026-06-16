---
title: "Text Encoders and Prompt Conditioning: How Words Steer the Denoiser"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Trace a text prompt from raw characters through tokenization, a CLIP or T5 or LLM encoder, and into the denoiser via cross-attention and pooled conditioning, with the math of attention, runnable diffusers code, and measured GenEval numbers that explain why attribute binding, counting, and text rendering fail."
tags:
  [
    "image-generation",
    "diffusion-models",
    "text-encoders",
    "clip",
    "cross-attention",
    "prompt-conditioning",
    "text-to-image",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/text-encoders-and-prompt-conditioning-1.png"
---

Type "a red cube on top of a blue sphere" into Stable Diffusion 1.5 and run it ten times. Count how many images put the red on the cube and the blue on the sphere. If you get six out of ten you are lucky; a lot of the time you get a *blue* cube on a *red* sphere, or a purple blob that is committed to neither. The model heard "red," "cube," "blue," "sphere" — it just lost track of which adjective belonged to which noun. Now run the same prompt through Stable Diffusion 3 or FLUX.1 and the binding holds far more often. Nothing changed about the cube or the sphere. What changed is the part of the system that reads the sentence: the **text encoder** and the way its output is **injected** into the denoiser.

This post is about that part. Not the U-Net, not the sampler, not the VAE — the path that takes a string of characters and turns it into a control signal precise enough to place a *red* cube above a *blue* sphere. It is the most underappreciated component in the whole text-to-image stack. People obsess over backbones (U-Net vs DiT) and samplers (DPM-Solver vs Euler) while the single biggest lever on *prompt adherence* — whether the image actually shows what you asked for — is which encoder you bolted on and how its embeddings reach the net. That is the lever we are going to pull apart.

![A dataflow figure showing a prompt going through a frozen text encoder into a token-embedding sequence, then into a cross-attention block where image queries read text keys and values, before reaching the denoiser block](/imgs/blogs/text-encoders-and-prompt-conditioning-1.png)

By the end you will be able to: encode a prompt with both CLIP and T5 in 🤗 `transformers` and read off the embedding shapes; explain mathematically how those embeddings steer the denoiser through cross-attention (queries from the image, keys and values from the text); extract and read a cross-attention map to see *which word each region of the image is looking at*; and diagnose the famous failure modes — attribute binding, counting, spatial relations, illegible text — back to the specific weakness of the encoder that caused them. We will keep tying this back to the series spine: conditioning is the "guidance" stage of the **diffusion stack** (data → VAE latent → forward noising → denoiser → sampler → **guidance/conditioning** → image), and the encoder choice is one more place where the **generative trilemma** (quality × diversity × speed) shows up — a bigger encoder buys adherence but costs memory and latency. If you have read [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) you know the denoiser is a function $\epsilon_\theta(x_t, t, c)$ that predicts the noise given the noisy image, the timestep, *and a condition $c$*. This whole post is about what $c$ is and how it gets in.

## 1. The job: from a string to a control signal

Start with the running example. The prompt is `"a red cube on top of a blue sphere"`. The denoiser is a network that, at every step of the reverse process, looks at a partially-denoised latent and predicts how to nudge it toward a clean image. Left alone, that network is *unconditional*: it will denoise toward *some* plausible image, but it has no idea you wanted a cube and a sphere. Conditioning is the act of feeding it the prompt so the denoising trajectory bends toward images that match the words.

The catch is that a neural network does not consume strings. It consumes tensors of floating-point numbers. So the prompt has to become a tensor — and not just any tensor. It has to be a tensor that encodes *meaning* in a way the denoiser can use: "red" near "cube," "blue" near "sphere," "on top of" expressing a vertical relation. The pipeline that does this conversion has three stages, and every one of them is a place where adherence can be won or lost:

1. **Tokenization** splits the string into discrete tokens (subword pieces) and maps each to an integer ID. `"a red cube"` might become `[a, red, cube]` → `[320, 736, 11353]` plus special start/end markers.
2. **The text encoder** — a transformer — turns that integer sequence into a sequence of continuous embedding vectors, one per token, where the vectors carry contextual meaning (the "red" vector is shaped by the words around it).
3. **Injection** routes those embeddings into the denoiser, usually through cross-attention layers, so the image-generation computation can read them at every spatial location and every denoising step.

Figure 1 shows the whole path at a glance. The rest of this post is a slow walk through each box, with the math and the code for each. We will spend the most time on stages 2 and 3, because that is where the interesting choices live — CLIP vs T5 vs an LLM, pooled vs sequence, cross-attention vs joint attention.

One framing to hold onto: the encoder output is **frozen context**, and the denoiser is the only thing being trained to *use* it (in most pipelines the text encoder weights are frozen during diffusion training). So the encoder is a fixed dictionary that translates language into a vector space, and the denoiser learns to look words up in that dictionary at the right places. If the dictionary is bad — if it scrambles word order, or can't fit your prompt, or smears two objects' attributes together — the denoiser has no way to recover the information. Garbage in, garbage steered. That is why the encoder is such a high-leverage choice: it sets a ceiling on adherence that no amount of denoiser training can lift.

It is worth being precise about where conditioning sits in the reverse process, because it changes how you reason about failures. Recall the conditional denoiser $\epsilon_\theta(x_t, t, c)$ predicts the noise in the noisy latent $x_t$ at step $t$, *given* the condition $c$. The condition enters at *every* step of sampling — there is no single moment where the prompt is "applied." At a high-noise step (say $t$ near the start), the latent is almost pure noise, so the conditioning has the most leverage over *coarse structure*: the overall layout, how many objects, where they sit. At a low-noise step (near the end), the structure is locked and conditioning only refines *texture and detail*. This is why prompt edits that change layout must happen early, and why the famous editing methods (prompt-to-prompt, SDEdit) intervene on the early steps. It also means a prompt that is too long for the encoder doesn't fail "a little" — it fails at *every* step, compounding, because the truncated information was never available to shape any part of the trajectory. The encoder is not a one-shot translator; it is a control signal read tens of times per image.

There is one more reason the encoder dominates: **it is shared, frozen, and reused**. A single CLIP or T5 forward pass produces the embeddings, and those same embeddings are consumed by the denoiser at all ~20–50 steps. So the encoder's representation quality is paid once and amortized over the whole sampling loop, whereas the denoiser pays per step. That asymmetry is why it is economically sane to bolt on a 4.7B T5 encoder in front of a 2B denoiser — you run the big encoder *once*, then the smaller denoiser many times. Understanding that cost structure is what makes the "drop T5 for short prompts" knob (§10) make sense: you are deciding whether the one-time encoder cost is worth it for *this* prompt's complexity.

## 2. Tokenization: where the prompt first gets quantized

Before any transformer sees the prompt, a tokenizer chops it into pieces. CLIP, the encoder that powered the entire first generation of text-to-image models, uses a byte-pair-encoding (BPE) tokenizer over a vocabulary of 49,408 tokens. (If you want the full mechanics of BPE — merges, the vocabulary build, why subwords beat whole words — that is its own deep dive in [the BPE tokenizer post](/blog/machine-learning/large-language-model/bpe-tokenizer); here we only need the consequences for prompting.)

Two consequences matter for image generation. First, tokenization is **subword**: rare words and made-up words get split into pieces. `"chiaroscuro"` is not one token; it is several. This is mostly fine for content words but it is *fatal* for spelling. When you ask a model to render the text `"OPEN"` on a sign, CLIP's tokenizer may hand the encoder a sequence of subword fragments with no clean notion of the four letters O-P-E-N in order. The encoder never sees the characters as characters, so the denoiser is trying to draw letters from a representation that doesn't really know what letters are. That is the deep reason early models produced sign text like "0PEN" — the spelling information was destroyed at the tokenizer, before the encoder even ran.

Second, and more famous: CLIP's text transformer has a **77-token context limit**. Two of those slots are the start-of-text and end-of-text markers, leaving 75 for your actual prompt. A long, detailed prompt — the kind people paste from a prompt-engineering guide — gets *truncated* at 75 tokens. Everything past the limit is silently dropped. The model literally never reads the back half of your prompt. Tooling like the Compel library and Automatic1111's prompt chunking works around this by splitting the prompt into 75-token chunks, encoding each, and concatenating — but that is a hack layered on top of a structural limit, and the chunks don't attend to each other, so cross-chunk relations ("the cat from clause 1 wearing the hat from clause 3") still get lost.

Why is the limit 77 and not, say, 512? It is baked into the pretrained CLIP: the text transformer was trained with learned positional embeddings for exactly 77 positions, so positions 78 and beyond have no embedding and the model has never seen them. You cannot simply pass a longer sequence — the positional table runs out. Extending CLIP's context would require retraining or interpolating positional embeddings, and even then the contrastive captions CLIP trained on were short (a sentence or two), so a context-extended CLIP would be extrapolating far outside its training distribution. This is the structural reason the field reached for T5 rather than "just make CLIP longer": T5 was *pretrained* on long sequences with relative position encodings, so feeding it 256 or 512 tokens is in-distribution. The token limit is not a tunable hyperparameter you forgot to set; it is a property of how the encoder was trained, and that is why changing it means changing the encoder.

There is a subtle third consequence of subword tokenization that bites compositional prompts specifically: the *number of tokens an object's description costs* is uneven. "cube" might be one token while "rhombicuboctahedron" is six, and a long descriptive phrase for one object can crowd out the token budget for another. In a prompt like "a meticulously detailed antique brass astrolabe and a plain wooden spoon," the first object consumes most of the slots and the spoon gets a thin representation — so the model under-renders the spoon. This token-budget imbalance is invisible until you tokenize and count, and it is one more reason the longer T5/LLM budgets help: every object gets enough room.

Here is the tokenization step in code, so the limit is concrete:

```python
from transformers import CLIPTokenizer

tok = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
)

prompt = "a red cube on top of a blue sphere"
ids = tok(prompt, padding="max_length", max_length=77,
          truncation=True, return_tensors="pt")

print(ids.input_ids.shape)        # torch.Size([1, 77])
print(ids.input_ids[0, :8])       # [49406, 320, 736, 11353, 525, ...]  49406 = <bos>
# Count the real (non-pad) tokens:
print(int(ids.attention_mask.sum()))   # ~10 for this short prompt
```

The output is always shape `[1, 77]` because we pad to the max length, but the `attention_mask` tells you how many slots are real. For our short prompt only about ten slots carry signal; the rest are padding. For a 200-word prompt, you would see the mask saturate at 77 and a warning that the input was truncated — a silent, common, and costly failure that many people never notice.

#### Worked example: how much prompt fits in CLIP

A typical English word is roughly 1.3 CLIP tokens (common words are one token, longer or rarer words split). So 75 usable token slots hold about **55–60 words**. A prompt like "a cinematic photograph of a red cube resting on top of a glossy blue sphere, studio lighting, shallow depth of field, 85mm lens, the cube casting a soft shadow on a white marble surface, hyperdetailed, 8k" is already ~35 words ≈ ~48 tokens — close to the edge. Add three more descriptive clauses and CLIP starts dropping them. T5-XXL, by contrast, is typically run at 256 or 512 tokens in SD3/FLUX, holding **200–380 words** — which is exactly why long, structured prompts work on those models and fall apart on SD1.5. The token budget is not a footnote; it is a hard ceiling on how much instruction the model can receive.

## 3. The text encoder: turning tokens into contextual embeddings

The tokenizer hands the encoder a sequence of integer IDs. The encoder — a transformer — turns each ID into a continuous vector and then refines those vectors through self-attention so each one is *contextualized* by the others. After the encoder runs, the "cube" vector is no longer a generic "cube"; it has been shaped by "red" and "on top of" and "sphere," so it carries the specific role this word plays in this sentence. That contextualization is the entire point of using a transformer rather than a lookup table.

The encoder produces **two distinct outputs**, and the difference between them is one of the most important and least-discussed details in conditioning:

- The **per-token sequence**: shape `[batch, seq_len, d]`. One vector per token. This is the rich, spatial signal — it preserves *which word is where* and is what cross-attention reads to place content in the image.
- The **pooled embedding**: shape `[batch, d]`. A single vector summarizing the whole prompt. CLIP produces it by taking the hidden state at the end-of-text token position (and projecting it). This is a global gist — useful for overall style but spatially blind.

We will see in §5 and §6 that these two outputs get injected through *completely different mechanisms* and do *completely different jobs*. The sequence drives cross-attention (spatial, word-level); the pooled vector gets added to the timestep embedding (global, style-level). Conflating them is a common source of confusion. Figure 2 lays out both paths.

![A layered stack figure showing the pooled embedding added to the timestep embedding for global AdaLN or FiLM control on one path, and the full token sequence feeding cross-attention keys and values for spatial control on the other path, both merging into conditioned features](/imgs/blogs/text-encoders-and-prompt-conditioning-2.png)

### What "contrastive" pretraining does to the embeddings

CLIP — Contrastive Language-Image Pretraining (Radford et al., 2021) — was not trained to help generate images. It was trained on ~400M image-caption pairs to make the embedding of an image close to the embedding of its caption and far from other captions, using a contrastive (InfoNCE) loss over a batch. The objective only cares about one number per image-text pair: the dot product between the *pooled* image embedding and the *pooled* text embedding. Formally, for a batch of $N$ pairs with image embeddings $u_i$ and text embeddings $v_i$ (both L2-normalized), the loss is

$$
\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{\langle u_i, v_i\rangle/\tau}}{\sum_j e^{\langle u_i, v_j\rangle/\tau}} + \log\frac{e^{\langle u_i, v_i\rangle/\tau}}{\sum_j e^{\langle u_j, v_i\rangle/\tau}}\right]
$$

where $\tau$ is a learned temperature. Read that carefully. The *only* gradient signal flows through the **pooled** embeddings $u_i, v_i$. Nothing in the loss rewards the text encoder for keeping per-token structure, word order, or compositional binding — it just needs the single summary vector to land near the matching image's summary vector. And because images are matched at the whole-image level ("a photo containing a red cube and a blue sphere"), a *bag of the right concepts* is almost as good as the correctly-bound sentence for minimizing the loss. The model learns that "red," "cube," "blue," "sphere" all being present is what makes the caption match the picture. *Which* color binds to *which* shape barely moves the contrastive loss, because both the right-bound and wrong-bound captions contain the same bag of words and the image contains both objects.

This is the mathematical root of CLIP's **bag-of-words tendency**, documented empirically by benchmarks like Winoground and ARO (Yuksekgonul et al., 2023), which show CLIP scoring "the grass eats the horse" almost identically to "the horse eats the grass." The encoder simply isn't pressured to represent the difference. When a diffusion model conditions on such an encoder, it inherits exactly that blindness — and that is why attribute binding fails. We will quantify it in §8.

It helps to see *why* the pooled vector is structurally lossy, beyond hand-waving. The pooled embedding is a fixed-size summary $v = \text{pool}(h_1, \dots, h_L)$ of a variable-length sequence of token hidden states $h_t$. Any pooling — taking the end-of-text token's hidden state (CLIP's choice), or mean-pooling — is a *many-to-one* map from $\mathbb{R}^{L \times d}$ to $\mathbb{R}^{d}$. By a counting argument, an enormous set of distinct token sequences collapse to nearby pooled vectors: "red cube, blue sphere" and "blue cube, red sphere" share the same multiset of tokens, so a permutation-tolerant summary maps them to almost the same point. The information that distinguishes them — the *binding* of adjective to noun — lives in the *relative arrangement* of the $h_t$, precisely the structure that pooling discards. Now recall that CLIP's contrastive loss only ever reads $v$. Gradient descent optimizes the encoder to make $v$ discriminative *for whole-image matching*, and a discriminator that only needs to detect the presence of concepts has no incentive to preserve arrangement in $v$. So the binding information isn't just *unused* by the loss — over training it gets actively squeezed out of the dimensions the loss cares about, because those dimensions are reallocated to whatever *does* improve the contrastive objective (concept presence, global style, photographic vs illustration, and so on). The per-token sequence $h_t$ still carries *some* order information (the self-attention layers that produced it are order-aware), which is why even CLIP-conditioned models bind correctly *sometimes* — the sequence path leaks a little structure into cross-attention. But the dominant, well-optimized signal is the bag, and that is what the denoiser learns to trust.

This is also why simply making CLIP *bigger* (CLIP-L → CLIP-bigG, ~63M → ~354M) helps adherence only modestly: a larger contrastive encoder is a *better bag-of-words detector*, not a fundamentally more compositional one. The objective, not the size, is the binding constraint — which is the whole reason the field switched encoders rather than just scaling CLIP. T5, with a span-corruption objective, optimizes a *completely different* quantity: it must reconstruct masked spans of text, which is impossible without representing order and dependency. That objective difference, not a parameter-count difference, is what buys compositionality.

T5 (Raffel et al., 2020) is a different animal. It is a text-to-text transformer pretrained with a span-corruption (masked-denoising) objective on a giant text corpus — pure language modeling, never paired with images. Because its objective forces it to *reconstruct* text, it must represent word order, syntax, and which adjective modifies which noun, or it can't fill in the blanks. The result is a per-token sequence that is far richer in compositional structure than CLIP's. T5 has no notion of what things *look* like — it was never shown a single image — but it has a much better notion of what a sentence *means*. The frontier insight of 2024 (SD3, FLUX) was: use *both*. Let CLIP provide the visually-grounded pooled gist and let T5 provide the structurally-faithful sequence. We get into the head-to-head next.

## 4. CLIP vs T5 vs LLM encoders: the three families

There are three families of text encoder in use across the 2022–2026 frontier, and the differences explain almost everything about a model's prompt-following behavior. The mental model is: **CLIP knows what things look like; T5 knows what sentences mean; LLMs know both and can reason about long instructions.** Figure 3 shows the adherence jump from CLIP-only to CLIP+T5 on a concrete benchmark; the section below explains why.

![A before-and-after figure contrasting a CLIP-only encoder that bags keywords and swaps attributes against a CLIP plus T5 stack that preserves order and binding, with GenEval scores rising from about 0.55 to above 0.68](/imgs/blogs/text-encoders-and-prompt-conditioning-3.png)

**CLIP text encoders** are small (CLIP-L/14 is ~63M params, OpenCLIP bigG/14 is ~354M) and image-aligned. Their superpower is that their embedding space is *shared with vision*, so a concept like "golden hour lighting" already points at the right visual region. Their weaknesses are the three we have built up: a 77-token ceiling, subword tokenization that mangles spelling, and the contrastive bag-of-words bias that breaks binding. SD1.5 used a single CLIP-L; SDXL used *two* CLIP encoders (CLIP-L plus OpenCLIP-bigG) concatenated, which roughly doubled the conditioning width (768 + 1280 = 2048 dims) and noticeably improved adherence over SD1.5 — but it is still all CLIP, still 77 tokens, still bag-of-words-prone.

**T5 encoders** (T5-XXL is ~4.7B params, the encoder half run at ~256–512 tokens) bring real language understanding. They fix spelling far better (a richer, more language-modeling tokenizer plus an objective that rewards getting words exactly right), they handle long prompts (4–7× the token budget), and they preserve compositional structure. SD3 and FLUX both feed T5-XXL's sequence alongside CLIP's. The cost is real: T5-XXL alone is bigger than the entire SD1.5 U-Net, eats ~10GB in fp16, and adds latency to every generation. SD3 lets you *drop* T5 at inference to save memory, at a measurable adherence cost — a clean illustration of the trilemma, where you trade quality for speed/memory by choosing how much encoder to run.

**Decoder-only LLM encoders** are the newest move. SANA (Xie et al., NVIDIA, 2024) replaced T5 entirely with a small decoder-only LLM (a Gemma-2 variant) as its text encoder, arguing that a modern instruction-tuned LLM understands prompts better than T5 and can handle very long, reasoning-heavy instructions. Other 2025 systems lean the same way, using an LLM's hidden states as conditioning. The appeal is obvious: LLMs have the strongest language understanding available, near-unlimited context, and can interpret instructions ("make the third object slightly smaller than the second") that a contrastive or span-corruption encoder never could. The catch is that an LLM's hidden states were optimized for *next-token prediction*, not for visual grounding, so you typically need a learned adapter/projection and careful training to align them to the denoiser's expectations.

| Encoder | Params | Max tokens | Objective | Compositionality | Spelling | Used by |
|---|---|---|---|---|---|---|
| CLIP-L/14 | ~63M | 77 | Contrastive | Weak (bag-of-words) | Poor | SD1.5 |
| CLIP-L + OpenCLIP-bigG | ~63M + 354M | 77 | Contrastive | Weak–moderate | Poor | SDXL |
| T5-XXL (encoder) | ~4.7B | 256–512 | Span corruption (LM) | Strong | Good | SD3, FLUX, PixArt |
| Decoder-only LLM (e.g. Gemma-2) | ~0.3–7B | thousands | Next-token (LM) | Strong | Good | SANA, newer models |

The table above is the decision grid. Read it as a Pareto frontier: moving down the rows buys adherence and prompt length at the cost of parameters, memory, and latency. Figure 4 renders the same trade-off as a comparison matrix; refer to it when you are choosing an encoder for a build.

![A matrix figure comparing CLIP, T5, and decoder-only LLM text encoders across parameter count, maximum token length, language strength, and which models use them](/imgs/blogs/text-encoders-and-prompt-conditioning-4.png)

### Encoding a prompt with CLIP and with T5 in code

Here is the practical flow: take one prompt, run it through both a CLIP text encoder and a T5 encoder, and inspect the shapes. This is exactly what `diffusers` does internally inside `encode_prompt`, unrolled so you can see it.

```python
import torch
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    T5TokenizerFast, T5EncoderModel,
)

device = "cuda"
prompt = "a red cube on top of a blue sphere, studio lighting"

# --- CLIP text encoder (SD1.5 / SDXL style) ---
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_enc = CLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=torch.float16
).to(device)

clip_ids = clip_tok(prompt, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    clip_out = clip_enc(clip_ids, output_hidden_states=True)

clip_seq = clip_out.last_hidden_state          # [1, 77, 768]  per-token sequence
clip_pooled = clip_out.pooler_output           # [1, 768]      global pooled
print("CLIP sequence:", clip_seq.shape, "pooled:", clip_pooled.shape)

# --- T5-XXL encoder (SD3 / FLUX style) ---
t5_tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
t5_enc = T5EncoderModel.from_pretrained(
    "google/t5-v1_1-xxl", torch_dtype=torch.float16
).to(device)

t5_ids = t5_tok(prompt, padding="max_length", max_length=256,
                truncation=True, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    t5_seq = t5_enc(t5_ids).last_hidden_state  # [1, 256, 4096]  per-token sequence
print("T5 sequence:", t5_seq.shape)            # note: no pooled output by default
```

The shapes tell the whole story. CLIP gives you a **`[1, 77, 768]`** sequence plus a **`[1, 768]`** pooled vector. T5 gives you a **`[1, 256, 4096]`** sequence and *no* pooled vector (T5 is sequence-to-sequence; there is no canonical pooled token, which is why SD3 takes its global pooled signal from CLIP and its rich sequence signal mostly from T5). Two things jump out: T5's sequence is **3.3× longer** (256 vs 77 token slots) and each vector is **wider** (4096 vs 768 dims). That is more capacity to represent the prompt, and it is literally the extra room where "which color goes on which shape" survives. The denoiser in SD3 receives a concatenation: CLIP pooled (for global conditioning) and the T5 sequence projected to the model dimension (for cross-/joint-attention).

#### Worked example: the memory cost of the T5 sequence

The T5 sequence tensor is `256 × 4096 × 2 bytes` (fp16) ≈ **2.1 MB** per prompt — trivial. The cost is the **encoder weights and activations**: T5-XXL's encoder is ~4.7B params ≈ **9.4 GB** in fp16, versus CLIP-L at ~63M ≈ **0.13 GB**. On a 24GB RTX 4090 running FLUX, T5 is the single biggest non-backbone memory consumer; this is why FLUX/SD3 inference recipes quantize T5 to 8-bit or 4-bit (with `bitsandbytes`/`optimum-quanto`) to claw back ~5–7 GB. The adherence loss from 8-bit T5 is small and usually invisible; 4-bit starts to show on spelling-heavy prompts. That is a concrete, actionable trilemma point: 8-bit T5 is almost always the right call on consumer GPUs.

## 5. Injection: how embeddings actually enter the denoiser

We now have the embeddings. How do they get *into* the denoiser so they can steer it? There are two mechanisms, and modern models use both at once.

### Cross-attention: the spatial control channel

The primary mechanism is **cross-attention**, introduced for text-to-image diffusion by Latent Diffusion (Rombach et al., 2022) — see [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) for where it sits in the architecture. The idea is elegant. Inside the [diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet), at several resolution levels, there are attention blocks. Self-attention lets image features attend to *other image features* (so the model keeps the image globally coherent). Cross-attention inserts a second attention where the image features attend to the *text* embeddings.

Concretely: the image feature map at some layer has shape `[h*w, d]` — that is `h*w` spatial positions, each a `d`-dimensional vector. The text embeddings are the sequence `[L, d_text]` — `L` tokens. Cross-attention computes

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
$$

where the **queries come from the image** and the **keys and values come from the text**:

$$
Q = W_Q \, \phi(x_t), \qquad K = W_K \, \tau_\theta(c), \qquad V = W_V \, \tau_\theta(c).
$$

Here $\phi(x_t)$ is the flattened image feature map (shape `[h*w, d]`) and $\tau_\theta(c)$ is the text-encoder output sequence (shape `[L, d_text]`). $W_Q$, $W_K$, $W_V$ are learned projection matrices; note $W_K$ and $W_V$ map from the *text* dimension into the attention dimension, which is how a 768- or 4096-dim text embedding gets aligned with the image features.

Let me make the directionality unmistakable, because it is the crux: **each spatial position in the image (a query) asks "which words are relevant to me?" and reads a weighted blend of those words' value vectors.** The attention matrix $A = \text{softmax}(QK^\top/\sqrt{d_k})$ has shape `[h*w, L]`: row $i$ is a probability distribution over the $L$ text tokens, telling you how much pixel-region $i$ attends to each word. That matrix *is* the steering. If region $i$ is going to become the cube, its row puts high weight on the "cube" and "red" tokens, and it pulls in their value vectors, which carry "make this region cube-shaped and red." Figure 5 unpacks exactly this — image-to-query, text-to-key/value, the softmax map, the context output.

![A dataflow figure showing image features projected to queries, frozen text embeddings projected to keys and values, the softmax of query-key producing an attention map, and the map times values producing context added back to the image features](/imgs/blogs/text-encoders-and-prompt-conditioning-5.png)

This is why cross-attention is the **spatial** control channel. The attention map is computed per spatial position, so different regions of the image can attend to different words. The top of the canvas can look at "cube" while the bottom looks at "sphere." That spatial selectivity is what lets a prompt control *layout*, not just style — and it is also exactly the machinery that ControlNet and prompt-to-prompt editing manipulate later in the series. Forward-link: [ControlNet and structural control](/blog/machine-learning/image-generation/controlnet-and-structural-control) adds a *second* conditioning signal alongside this text cross-attention, and the [image-generation stack capstone](/blog/machine-learning/image-generation/building-an-image-generation-stack) wires the whole conditioning path end to end.

Here is where `encoder_hidden_states` — the variable name `diffusers` uses for the text sequence — actually enters the computation. This is a faithful sketch of the cross-attention forward inside `diffusers`' attention processor:

```python
import torch
import torch.nn.functional as F

def cross_attention(image_features,           # [B, h*w, d]   the queries' source
                    encoder_hidden_states,    # [B, L, d_text] the text sequence (K,V)
                    to_q, to_k, to_v, to_out,  # nn.Linear projections
                    num_heads):
    B, N, d = image_features.shape
    # Queries from the IMAGE; keys and values from the TEXT.
    q = to_q(image_features)                  # [B, N, d]
    k = to_k(encoder_hidden_states)           # [B, L, d]
    v = to_v(encoder_hidden_states)           # [B, L, d]

    # reshape to heads: [B, heads, seq, d_head]
    def heads(t, seq):
        return t.view(B, seq, num_heads, d // num_heads).transpose(1, 2)
    q, k, v = heads(q, N), heads(k, L := encoder_hidden_states.shape[1]), heads(v, L)

    # the attention map A has shape [B, heads, N(image), L(text)]
    out = F.scaled_dot_product_attention(q, v=v, key=k, query=q)  # fused softmax(QK^T/sqrt)·V
    out = out.transpose(1, 2).reshape(B, N, d)
    return to_out(out)                         # [B, h*w, d] context, added to image features
```

The single most important line is `k = to_k(encoder_hidden_states)` — that is, mechanically, the prompt entering the U-Net. When you call a `diffusers` pipeline with `prompt=...`, the pipeline runs the text encoder, gets `encoder_hidden_states`, and threads it into *every* cross-attention block of the U-Net at every denoising step. The U-Net's own weights are the only thing that learned how to use it; the encoder output is fixed context.

A few details of the real implementation matter for understanding behavior. First, cross-attention is a *residual* sub-layer: the context returned above is added back to the image features (`x = x + cross_attn(norm(x), text)`), so the prompt *nudges* the features rather than overwriting them — which is why a weak conditioning signal produces an image that drifts toward the prompt instead of snapping to it. Second, there is a separate `attn2` (cross-attention to text) and `attn1` (self-attention among image tokens) in each transformer block of the U-Net; the self-attention keeps the image internally coherent while the cross-attention injects the prompt, and they alternate. Third, the keys and values are recomputed from the *same* frozen `encoder_hidden_states` at every layer and every step, so caching them across steps is a legitimate and common optimization (the text embeddings never change during a generation). Fourth — and this is the hook every downstream technique uses — because the text projection $W_K, W_V$ is a plain linear layer, you can edit `encoder_hidden_states` *before* it enters and get predictable effects: zero out a token to ablate a word, scale a token to weight it, swap a token's embedding to do prompt-to-prompt editing. The conditioning interface is, deliberately, just a tensor you can reach in and modify.

The number of cross-attention layers matters too. SD1.5's U-Net has 16 cross-attention blocks across its resolution levels; SDXL has 70-plus (it shifted capacity into the attention-heavy middle); a DiT like SD3 puts attention in *every* transformer block. Each one is a fresh chance for the image to consult the prompt, and more consultation points generally means tighter adherence — another axis on which the newer architectures pulled ahead, orthogonal to the encoder itself.

### Pooled embeddings: the global conditioning channel

The second mechanism is for the **pooled** vector. It does not go through cross-attention. Instead it gets combined with the **timestep embedding** — the vector that tells the network which denoising step it is on — and that combined global vector modulates the network through FiLM-style or AdaLN-style scale-and-shift. In SDXL the pooled CLIP embedding (plus size/crop conditioning) is added to the timestep embedding; in [DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit)-based models like SD3 the pooled vector feeds AdaLN-Zero, which predicts per-layer scale, shift, and gate parameters.

The key property: a *single* global vector can only apply a *uniform* modulation across the whole feature map (or whole layer). It has no spatial selectivity — it cannot say "make the top cube-shaped." It can only say "make the overall image warmer / more photographic / match this general vibe." So pooled conditioning controls **global style and gist**; cross-attention controls **spatial, word-level content**. Figure 6 (next section) and Figure 7 both lean on this distinction; the cleanest way to remember it is the contrast in Figure 7: pooled = one knob for the whole image, sequence = one knob per word, placed in space.

Why have a pooled path at all, if the sequence path is so much richer? Two reasons. First, it is *cheap and stable*: a single vector added to the timestep embedding costs nothing and gives the network a reliable global handle that does not depend on the attention dynamics working out. Models train more stably when there is a direct, always-available global conditioning signal; the AdaLN-Zero initialization in DiT specifically starts this path as a no-op and lets the network learn how strongly to lean on it. Second, some properties really *are* global and have no natural spatial address. "Photographic vs illustration," "high contrast," "warm color grade," the SDXL size/crop conditioning that tells the model what resolution and crop the training image had — these modulate the *whole* image uniformly, and forcing them through per-region cross-attention would be both wasteful and less effective. The pooled path is the right tool for global properties precisely because it *cannot* be spatial. The division of labor is not a limitation to work around; it is a sensible factoring of the conditioning problem into "global look" and "local content," each routed to the mechanism suited to it. When you understand which channel owns which property, a lot of mysterious behavior — why a style token affects the whole image but a "small" qualifier on one object barely registers — stops being mysterious.

That division of labor is deliberate and it scales. SD3's MM-DiT (covered in [the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe)) uses the CLIP pooled vector for global AdaLN conditioning *and* the full CLIP+T5 sequence for joint attention — best of both. The pooled path is cheap (one vector) and sets the overall look; the sequence path is expensive (L vectors, full attention) and sets the precise content.

## 6. Cross-attention in U-Nets vs joint attention in MM-DiT

There is a second architectural axis that turns out to matter as much as the encoder choice: *how* the text and image tokens attend. The U-Net/SD1.5/SDXL world uses **cross-attention**; the MM-DiT/SD3/FLUX world uses **joint attention**. The difference is subtle in code and large in behavior.

In **cross-attention** (U-Net), the text is a *frozen, external* source. Image queries read text keys and values, but the text never reads the image, and the text representation never updates inside the denoiser. Information flows one way: image ← text. The text embeddings computed by the encoder at the start are the same at every layer and every step; the U-Net only ever *consumes* them.

In **joint attention** (MM-DiT), the text tokens and image tokens are *concatenated into one sequence* and run through ordinary self-attention together. Now every token attends to every other token: image attends to text *and* text attends to image *and* text attends to text. Information flows both ways: image ↔ text. The text representation gets *updated* by what the image is doing, layer after layer. This bidirectionality is one reason SD3/FLUX bind attributes and render text better — the prompt representation can adapt to the emerging image and disambiguate itself, instead of being a fixed dictionary the image looks up.

The math is the same softmax attention; only the inputs change. For joint attention, concatenate text tokens $c$ and image tokens $x$ into one sequence $z = [c; x]$ of length $L + h w$, then run self-attention on $z$:

$$
Q = W_Q z, \quad K = W_K z, \quad V = W_V z, \quad \text{out} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

(MM-DiT uses *separate* $W_Q, W_K, W_V$ per modality — different projection weights for the text part and the image part of $z$ — then concatenates before the softmax, which is why it is called "joint" rather than plain self-attention. The attention itself is over the combined sequence.) The attention matrix is now `[(L+hw), (L+hw)]` — it contains image→text blocks (like cross-attention), text→image blocks (the new reverse direction), and the within-modality blocks. Cross-attention is, in this light, the *upper-right block* of joint attention with the rest masked off and the text part frozen.

| Property | Cross-attention (U-Net) | Joint attention (MM-DiT) |
|---|---|---|
| Models | SD1.5, SD2, SDXL | SD3, FLUX |
| Information flow | image ← text (one-way) | image ↔ text (two-way) |
| Text representation | frozen, same every layer | updates layer by layer |
| Attention matrix | `[hw, L]` | `[(L+hw), (L+hw)]` |
| Compute | cheaper (L is small) | costlier (text adds to seq len) |
| Attribute binding | weaker | stronger |
| Text rendering | poor | much better |

The cost of joint attention is that text tokens lengthen the attention sequence, and attention is quadratic in sequence length, so adding 256 T5 tokens to 4096 image tokens is not free. But the adherence payoff has been judged worth it across the open frontier. This is the same one-way-vs-two-way contrast that drives the recipe in [the MM-DiT post](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe); here we are seeing it specifically from the *conditioning* angle.

### Assembling the conditioning in a real multi-encoder model

It is worth seeing how a frontier model stitches *three* encoders into the two conditioning channels, because it makes the pooled-vs-sequence split concrete. SD3 runs CLIP-L, CLIP-bigG, and T5-XXL, then assembles them like this (a faithful sketch of `StableDiffusion3Pipeline.encode_prompt`):

```python
# Three encoders, two conditioning channels.
# 1) CLIP-L and CLIP-bigG each give a pooled vector AND a per-token sequence.
clip_l_seq, clip_l_pooled = encode_clip(clip_l, prompt)   # [1,77,768], [1,768]
clip_g_seq, clip_g_pooled = encode_clip(clip_g, prompt)   # [1,77,1280], [1,1280]
# 2) T5 gives only a (long, wide) per-token sequence.
t5_seq = encode_t5(t5, prompt, max_length=256)            # [1,256,4096]

# --- Pooled channel (global, -> timestep / AdaLN) ---
pooled = torch.cat([clip_l_pooled, clip_g_pooled], dim=-1)  # [1, 2048]

# --- Sequence channel (-> joint attention) ---
# Concatenate CLIP sequences on the feature dim, pad to T5 width, then
# concatenate CLIP and T5 along the TOKEN dim into one conditioning sequence.
clip_seq = torch.cat([clip_l_seq, clip_g_seq], dim=-1)      # [1, 77, 2048]
clip_seq = pad_to_width(clip_seq, 4096)                     # [1, 77, 4096]
seq = torch.cat([clip_seq, t5_seq], dim=1)                  # [1, 333, 4096]

# 'pooled' feeds AdaLN-Zero (global style); 'seq' feeds MM-DiT joint attention.
```

Read the shapes: the **pooled** channel is a `[1, 2048]` concatenation of the two CLIP pooled vectors — that is the global-style knob added into the AdaLN conditioning. The **sequence** channel is a `[1, 333, 4096]` tensor — 77 CLIP tokens plus 256 T5 tokens, concatenated along the token axis — that becomes the keys/values (and, in joint attention, also queries) the image tokens attend over. Two CLIPs for the visually-grounded pooled gist, T5 for the long structurally-faithful sequence. This is the literal data structure behind every claim in this post about "pooled does style, sequence does content." When the SD3 paper *drops* T5, the `seq` tensor shrinks to just the 77 CLIP tokens, and adherence on compositional prompts falls — you can see, in the shapes, exactly what information left the building.

## 7. Reading the steering: extracting a cross-attention map

Everything above says the attention map *is* the steering. Let us actually look at one. The attention map $A$ of shape `[hw, L]` can be reshaped, for any token, into an `[h, w]` heatmap showing where in the image that word is being attended to. This is the basis of prompt-to-prompt editing and of diagnosing exactly where binding breaks. It is worth pausing on how unusual it is to have this kind of visibility: in most neural networks the connection between an input and an output region is opaque, but here the cross-attention probabilities give a direct, per-word, per-region readout of *what the model is looking at while it draws each part of the image*. That readout is not a post-hoc explanation bolted on afterward — it is the actual quantity the model computed and used. So when you visualize it, you are not approximating the model's reasoning; you are reading it. Here is how to hook a `diffusers` U-Net and capture the maps:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

attn_maps = {}

def make_hook(name):
    def hook(module, args, kwargs, output):
        # diffusers Attention modules expose the processor; we re-derive the map.
        # Simpler: monkeypatch the attention to stash probs. For a sketch:
        attn_maps[name] = module._last_attn_probs  # set inside a custom processor
    return hook

# Register on cross-attention layers (attn2) only; self-attention is attn1.
for name, module in pipe.unet.named_modules():
    if name.endswith("attn2"):
        module.register_forward_hook(make_hook(name), with_kwargs=True)

image = pipe("a red cube on top of a blue sphere",
             num_inference_steps=30, guidance_scale=7.5).images[0]

# After generation, average the maps over heads/steps, reshape [h*w, L] -> [h, w],
# and pick the column for the token id of "cube" or "sphere" to visualize.
```

In practice you use a custom `AttnProcessor` that stores `softmax(QK^T)` before it multiplies by `V`, because `scaled_dot_product_attention` fuses the softmax away and you can't read it back. Libraries like `daam` (diffusion attentive attribution maps) and the prompt-to-prompt reference code do exactly this. When you visualize the map for the "cube" token, you see a blob of attention sitting where the cube is; for "red," ideally, the same region lights up — *if* binding worked. When binding fails, you can literally see it: the "red" token's attention map smears across *both* objects, or worse, peaks on the sphere. That visualization is the most convincing diagnostic there is, and it is the bridge to the next section: the failure modes, and why they happen.

#### Worked example: spotting an attribute swap in the map

Generate "a red cube and a blue sphere," capture the cross-attention maps at the 16×16 resolution block (the most semantically meaningful in SD1.5), average over heads and over the middle third of denoising steps (steps ~8–20, where layout is decided). Reshape the "red" column to a 16×16 heatmap and overlay it on the image. In a *good* generation the "red" heatmap and the "cube" heatmap overlap tightly. In a *failed* generation you will often see the "red" map peak on the sphere region while "blue" peaks on the cube — the colors and shapes attended to the wrong places. That overlap-or-not is a quantitative signal: the cosine similarity between the "red" map and the "cube" map predicts whether the color landed on the cube. This is the mechanism behind training-free fixes like Attend-and-Excite (Chefer et al., 2023), which nudges the latent at inference to *increase* each subject token's max attention so no object gets ignored.

## 8. The failure modes, and exactly why they happen

Now we can be precise about the four canonical adherence failures. Each one traces to a specific weakness we have already built up, and each has a known mitigation. Figure 6 is the cause-and-fix grid; the prose below explains the mechanism for each row.

![A matrix figure mapping four prompt-adherence failures — attribute binding, counting, spatial relations, and text rendering — to their root cause in the encoder and the technique that mitigates each](/imgs/blogs/text-encoders-and-prompt-conditioning-6.png)

**Attribute binding** ("a red cube and a blue sphere" → swapped colors). Root cause: CLIP's contrastive, bag-of-words representation, derived in §3. The pooled-only contrastive objective never pressured CLIP to keep "red" *attached to* "cube," so the conditioning signal genuinely doesn't carry the binding strongly. The denoiser, conditioned on a smeared representation, paints colors onto whichever object the attention map happens to favor. There is a second, attention-side mechanism worth naming: even when the encoder *does* keep "red" distinct from "cube," the cross-attention is computed *independently per region*, and nothing in the vanilla attention forces the "red" map and the "cube" map to *coincide spatially*. The cube region attends strongly to "cube" (it determines the shape) but the "red" attention can leak onto the sphere region, especially when the two objects are similar in size and the denoiser settles their layout before their color. So binding can fail at two distinct stages — the encoder smears it, or the attention maps fail to align — and the fixes target each: a better encoder repairs the first, while inference-time methods that *force* each attribute token's attention to overlap its noun's attention repair the second. Mitigation: a stronger encoder (T5 preserves the binding in its sequence) *and* joint attention (so the representation can disambiguate against the emerging image). Inference-time fixes (Attend-and-Excite, structured/region prompting, and attention-map alignment losses) help on existing CLIP models without retraining.

**Counting** ("five apples" → four or six). Root cause: neither CLIP nor T5 has explicit numerosity features, and the diffusion training data's captions are noisy about exact counts. The text embedding for "five" is a generic token; nothing in the denoiser robustly maps it to "lay down exactly five blobs." There is a deeper reason counting is hard even with a perfect encoder: generation is a *parallel* spatial process, not a sequential one. The denoiser decides the whole image at once across many regions; there is no internal counter incrementing as it places each apple, the way an autoregressive model could in principle tally tokens. So even if "five" is perfectly represented in the conditioning, the architecture has no mechanism to *enforce* a global count constraint across independently-denoised regions — it can only learn a statistical tendency from captions, which top out around four or five before the tendency washes out. Counting is genuinely hard and even FLUX/SD3 are imperfect past ~4–5 objects. Mitigations are weaker here: layout conditioning (give the model bounding boxes, one per object, which *does* impose a count), RL/preference fine-tuning on counting prompts, or just generating and filtering with a detector. Honest take — counting is the failure that is *least* fixed by the encoder alone, because it is as much an architecture/training-data problem as a conditioning problem.

**Spatial relations** ("A to the left of B," "on top of") — Root cause: position information is weakly encoded and weakly grounded. CLIP's bag-of-words tendency hits relational words hardest ("left of" vs "right of" barely move the contrastive loss because they share every content word). T5 represents the relation better, and joint attention helps the model act on it, but spatial prepositions remain a relative weak point. The mechanism is that a relational word like "above" must be *translated into geometry* — into which region's queries attend to which object's tokens, arranged vertically — and that translation is something the denoiser has to learn from captioned examples, where spatial language is often imprecise or absent. Captions say "a dog and a frisbee," rarely "a frisbee twenty degrees above and to the left of the dog," so the model gets little supervision on exact relations. Mitigation: explicit region/layout conditioning (the strongest fix — you specify the geometry directly instead of hoping the word translates), or a structure-aware approach like ControlNet when you have a layout in mind. This is the cleanest case where the *right answer* is to stop fighting the text channel and add a *spatial* conditioning channel, which is exactly what the next post in the series builds.

**Text rendering** ("a sign that says OPEN") — Root cause: the *combination* of subword tokenization (the letters aren't represented as letters, §2) and the 77-token CLIP limit. To draw the letters O-P-E-N in order, the denoiser needs a conditioning signal that *encodes those four characters in that order*. CLIP's BPE tokenizer may fold "OPEN" into one or two subword tokens whose embeddings say "this is the concept of an open sign," not "draw O, then P, then E, then N." The character identity and order — the only thing that matters for spelling — is exactly what got compressed away. The model then renders *plausible sign-like glyphs*, which is why early text reads like a foreign alphabet: right texture, wrong letters. T5 helps a lot here because its tokenizer and language-modeling objective preserve far more of the exact character content, and longer prompts let the model see the quoted string in full. The frontier models that render legible text — FLUX, Ideogram, SD3.5 — all use T5 or an LLM encoder, and some use character-aware tokenization or byte-level inputs specifically for typography. This is the failure mode where the encoder choice is most *decisive*: it is essentially impossible to render reliable text from CLIP-only conditioning, and quite reliable from a strong T5/LLM stack. If your product needs legible in-image text, the encoder decision is not optional — it is the feature.

The throughline: three of the four failures (binding, spatial, text) are *encoder problems*, and they are exactly the three that adding T5 + joint attention fixes. Counting is the stubborn one that needs layout or RL. That is why the single highest-leverage adherence upgrade in the entire field, 2023→2024, was "add a T5 encoder and switch to joint attention" — it directly attacks the dominant failure modes.

### The pooled-vs-sequence distinction, one more time, because it explains a lot

A surprising number of "the model ignored part of my prompt" complaints come from the **pooled vs sequence** split (Figure 7). If a model leans too heavily on the pooled global vector — say, you are using a pipeline that only wires the pooled embedding for some control — you get an image that matches the *overall vibe* of the prompt but ignores the specific spatial instructions, because pooled conditioning is spatially blind by construction. Conversely, a model with a rich sequence path and good cross-/joint-attention can place each word's content. When you debug adherence, ask: is the failing instruction a *global* property (style, mood — pooled handles it) or a *spatial/compositional* one (layout, binding, count — that needs the sequence path and good attention)? That single question routes most debugging.

![A before-and-after figure contrasting a single pooled vector that sets only global style with no spatial control against the full token sequence in cross-attention that enables word-level spatial placement](/imgs/blogs/text-encoders-and-prompt-conditioning-7.png)

## 9. Prompt weighting and embedding tricks

Because conditioning is "just" a sequence of embedding vectors fed into attention, you can manipulate it directly — which is what all the prompt-weighting syntax does under the surface. Three tricks are worth knowing, because they are mechanism, not magic.

**Prompt weighting** — `(red cube:1.4)` in Automatic1111, or the Compel library's `+`/`-` syntax — scales the embedding vectors of specific tokens. Mechanically, after the encoder runs, you take the token embeddings for "red cube" and push them away from the unconditioned/empty embedding by a factor, then renormalize. Stronger weight → those tokens dominate the cross-attention more → that concept is rendered more forcefully. It is a blunt instrument (it can blow out the rest of the image if you crank it), but it is real and it works because attention is linear in the value vectors.

```python
# diffusers supports precomputed prompt_embeds, so you can weight tokens yourself.
import torch
# embeds: [1, 77, 768] from the CLIP encoder; ids: [1, 77] token ids.
# Suppose tokens 2..3 are "red cube". Emphasize them by 1.4x relative to the
# empty/unconditional embedding (a common, simple weighting scheme):
empty = pipe.text_encoder(pipe.tokenizer("", return_tensors="pt",
            padding="max_length", max_length=77).input_ids.to("cuda"))[0]
w = torch.ones(77, device=embeds.device); w[2:4] = 1.4
embeds = empty + (embeds - empty) * w[None, :, None]
image = pipe(prompt_embeds=embeds, num_inference_steps=30).images[0]
```

The reason this is more than a hack is the linearity of attention in the value vectors. The context a region receives is $\sum_t A_{it} v_t$ — a convex combination of token values weighted by the attention probabilities $A_{it}$. Scaling the *embedding* of a token before projection raises its key's dot-product against queries, which raises $A_{it}$ for that token across regions, which raises that token's share of every region's context. So "weight = 1.4" really does mean "let this concept win more of the attention mass." Crank it too high and the softmax saturates onto that one token, starving the rest of the prompt — the blown-out look you get from `(masterpiece:1.8)`-style over-weighting. The same linearity is why *averaging* two prompts' embeddings produces a sensible blend, and why interpolating between two prompt embeddings gives a smooth semantic morph: the conditioning space is, to first order, a vector space the attention reads linearly.

#### Worked example: weighting a token and watching the attention shift

Take "a red cube on top of a blue sphere" on SD1.5 with a fixed seed. Generate once at default weights; the cube comes out a washed-out pink half the time. Now set the weight on the "red" token to 1.3 (push its embedding 1.3× away from the empty embedding, as in the code above) and regenerate at the *same seed*. Capture the cross-attention map for "red" at the 16×16 block: its peak on the cube region rises measurably (the column's max attention probability goes from, say, ~0.18 to ~0.31 in a typical run), and the rendered cube saturates to a clear red. Push to 1.8 and the map collapses — "red" dominates so hard the whole image tints red and the sphere loses its blue. That is the trade-off curve of prompt weighting in one experiment: a little weight fixes a weak binding, too much weight hijacks the global color. The lesson generalizes — weighting is a scalpel for *under*-attended tokens, not a volume knob to crank on everything.

**Negative prompts** are the same idea on the other side of classifier-free guidance. CFG (covered in [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance)) extrapolates between an unconditional and a conditional prediction: $\hat\epsilon = \epsilon_\varnothing + s\,(\epsilon_c - \epsilon_\varnothing)$. A *negative* prompt simply replaces the unconditional $\epsilon_\varnothing$ with a prediction conditioned on the thing you *don't* want, so the extrapolation pushes *away* from it. "Negative prompt: blurry, extra fingers" means the guidance vector points away from blurry, many-fingered images. It is conditioning used as a repulsor — and it only exists because of how CFG combines two conditioned predictions. Note the interaction with the encoder: a negative prompt is *also* run through the text encoder and *also* subject to the 77-token limit and bag-of-words bias, so an over-long negative prompt is just as silently truncated as a positive one, and a negative like "not red" does not reliably remove red (CLIP barely represents negation). Negatives work best as concrete *style/quality* repulsors ("blurry, low-res, jpeg artifacts"), not as logical negations of content.

**Textual Inversion embeddings** go further: instead of weighting existing tokens, you *learn a new token embedding* for a concept (a specific person, a style) by optimizing a fresh vector to reconstruct example images, freezing everything else. The result is a `<my-token>` you can drop into any prompt, and the encoder treats it as a real word. That is the most direct demonstration that "the prompt" is, to the denoiser, *nothing but a sequence of vectors* — and if you can find the right vector, you can name anything. This is the on-ramp to the personalization post ([DreamBooth, Textual Inversion, LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora) in this series' control track), where we fine-tune the conditioning path properly.

## 10. Case studies: measured prompt-adherence numbers

Now the proof. The standard compositionality benchmarks are **GenEval** (Ghosh et al., 2023) — which checks object presence, counting, color binding, position, and attribute binding using an object detector — and **T2I-CompBench**. The headline pattern across the literature is consistent: adding a strong language encoder and joint attention raises GenEval overall scores substantially, with the biggest gains on the *binding* and *position* sub-scores. Figure 8 places these in the field's timeline of encoder upgrades.

![A timeline figure tracing the text-encoder stack from a single CLIP in SD1.5 to dual CLIP in SDXL to CLIP plus T5 in SD3 and FLUX and finally a decoder-only LLM in SANA, each step improving prompt understanding](/imgs/blogs/text-encoders-and-prompt-conditioning-8.png)

**SDXL vs SD3 on GenEval.** SDXL (dual CLIP, U-Net cross-attention) reports a GenEval overall around the mid-0.5s. SD3 (CLIP + T5-XXL, MM-DiT joint attention) reports an overall in the high-0.6s to ~0.7, per the SD3 paper (Esser et al., 2024), with the *color attribute* and *position* sub-scores improving the most — exactly the binding-and-spatial failures we attributed to CLIP. These are the numbers behind Figure 3's before/after. (Cite the source papers for exact figures; treat the rounded values here as directional, and re-pull the precise sub-scores from the SD3 paper for a report.)

**The T5 ablation in SD3.** The SD3 paper's most instructive result for *this* post is the ablation where they *drop* T5 at inference. Removing T5 leaves the model with CLIP-only conditioning and produces a measurable drop in prompt adherence — most visibly on dense, text-heavy, and highly compositional prompts — while simpler prompts barely change. This is the cleanest single piece of evidence that the encoder, not the backbone, owns adherence: same denoiser, same sampler, same everything, only the encoder signal changes, and adherence moves. It is also a practical knob — drop T5 to save ~10GB when your prompts are short, keep it when they are compositional or contain text.

**PixArt-α: T5 on a budget.** PixArt-α (Chen et al., 2023) used T5-XXL as its *only* text encoder (no CLIP) on a relatively small DiT and still achieved strong prompt following at a fraction of SDXL's training cost, partly *because* the T5 encoder gave it such a clean compositional signal. It is direct evidence that the encoder is doing heavy lifting — a small denoiser with a great encoder can out-adhere a big denoiser with a weak one.

**SANA: the LLM-encoder bet.** SANA (Xie et al., 2024) replaced T5 with a small decoder-only LLM (a Gemma-2 variant) as the text encoder, paired with a deep-compression autoencoder and linear attention. Its claim — supported by its reported numbers — is competitive or better prompt adherence than T5-based models at dramatically lower latency, the LLM contributing strong instruction understanding and long-context handling. It is the leading example of the third family and a signal of where conditioning is heading: the text encoder becoming a genuine language model that *reasons* about the prompt rather than merely embedding it.

| Model | Text encoder(s) | Attention | GenEval overall (approx.) | Notes |
|---|---|---|---|---|
| SD1.5 | CLIP-L (77 tok) | U-Net cross-attn | ~0.43 | bag-of-words, mangles text |
| SDXL | CLIP-L + bigG (77) | U-Net cross-attn | ~0.55 | dual CLIP, better but still 77 tok |
| PixArt-α | T5-XXL only (120 tok) | DiT cross-attn | ~0.48–0.5+ | strong adherence at low train cost |
| SD3 | CLIP-L+G + T5-XXL (256) | MM-DiT joint | ~0.68–0.70 | best binding/position; T5 droppable |
| FLUX.1-dev | CLIP-L + T5-XXL (512) | hybrid DiT joint | ~0.66+ | legible text, long prompts |
| SANA | decoder-only LLM | DiT linear attn | competitive | LLM encoder, very low latency |

Treat the GenEval numbers as approximate and directional — exact values vary by evaluation protocol, detector, and sampler/CFG settings, and you should re-pull the precise figures from each source paper before quoting them in a report. The *pattern*, however, is robust and reproduced across many papers: **more language-capable encoder + two-way attention → higher compositional adherence, biggest gains on binding/position/text.** Measure it yourself honestly with the protocol below.

#### How to measure adherence honestly

GenEval runs your model on a fixed prompt suite (~550 prompts across six skills), generates a fixed number of images per prompt (commonly 4), runs an object detector (Mask2Former/DETR) plus a color/count classifier, and scores how many generations satisfy the prompt's compositional spec. To compare two encoders fairly: fix the *seed set*, the *sampler* (e.g. 28-step DPM-Solver++ or 28-step flow-matching Euler), the *CFG scale* (use each model's recommended value — CFG interacts strongly with adherence, so don't reuse one model's CFG on another), the *resolution*, and the *number of images per prompt*. Only then is a GenEval delta attributable to the encoder. Report sub-scores, not just the overall — the overall hides that most of the encoder gain is concentrated in binding and position.

## 11. Stress test: debugging a prompt that won't bind

Let me put the whole post to work on a real engineering problem, the kind that lands in your lap when you ship a generation feature. The complaint: "the model ignores half my prompt." The prompt is `"a vintage red bicycle leaning against a blue brick wall, a small white dog sitting beside it, a street sign that reads MAPLE AVE, golden hour"`. The output reliably shows a bicycle and a wall, sometimes a dog, the colors wander, and the sign reads "MPALE AVF" or nothing. Here is how to reason from symptom to fix, step by step, using only the machinery above.

**Step 1 — check for truncation first.** Tokenize the prompt and sum the attention mask. This prompt is ~28 words ≈ ~40 tokens — under the 77 limit, so truncation is *not* the cause here. (If it had been a 120-word prompt on SD1.5, we would stop here: the back half was never read, and no amount of weighting fixes information the encoder never saw. The fix would be a longer-context encoder, full stop.) Always run this check first because it is the cheapest and most common culprit; here it comes back clean, so we move on.

**Step 2 — classify each failing instruction as global or spatial.** "Golden hour" is global style — it actually works, because pooled conditioning handles it. The failures are all *spatial/compositional*: which object is which color (binding), the dog's presence and position (placement), the sign text (typography). That diagnosis immediately tells us the problem lives in the *sequence path* and the *attention*, not the pooled path — so cranking some global style knob will do nothing.

**Step 3 — look at the attention maps.** Capture the cross-attention for "red," "blue," "dog," and the sign tokens at the 16×16 block, averaged over the layout-deciding steps. You find "red" leaking across both the bicycle and the wall, "dog" with a weak, diffuse map (it is being under-attended — a classic dropped-object signature), and the sign tokens with scattered, low-mass attention (CLIP barely represents the letters). This is the smoking gun: the encoder is CLIP-only, and we are seeing all three CLIP weaknesses at once — bag-of-words binding, object dropping, and broken typography.

**Step 4 — apply the cheapest fix that targets the diagnosis.** On the *existing* CLIP model, without retraining: (a) weight "red" up to ~1.3 to pull its attention onto the bicycle and away from the wall; (b) use an Attend-and-Excite-style nudge to raise the "dog" token's max attention so it stops getting dropped; (c) accept that the sign text will not render legibly — that is structurally beyond CLIP, and no inference trick fixes it. This recovers binding and the dog, and we ship a version that simply avoids promising legible signage.

**Step 5 — decide whether to change the encoder.** If legible signage is a *requirement*, steps 1–4 cannot deliver it; the decision is to move to a T5/FLUX-class model. We re-run the same prompt on FLUX: binding holds, the dog appears reliably, and "MAPLE AVE" renders cleanly because T5 preserves the characters and joint attention places them. The cost is ~10GB more VRAM and higher latency, which we mitigate with 8-bit T5. That is the full decision tree — *check truncation, classify the instruction, read the map, try the cheap fix, change the encoder only when the requirement is structurally out of CLIP's reach.*

**Stress-testing the decision.** What if we had cranked the weighting harder instead of changing models? Pushing "red" to 1.8 and "dog" to 1.6 to force them through CLIP produces an over-saturated, distorted image — the softmax hijack from §9 — and *still* no legible sign. What if we kept CLIP but added a chunking workaround for length? It would not help, because length was never the problem; binding and typography were. What if the prompt had been short and purely stylistic ("a moody red bicycle, golden hour")? Then CLIP-only is perfectly adequate and moving to FLUX would be wasted VRAM. The discipline is matching the fix to the *diagnosed* cause — the most common failure in practice is reaching for a bigger model when the real bug was a silent truncation, or cranking weights when the real bug was a missing encoder capability. The map and the mask tell you which world you are in.

## 12. When to reach for which encoder (and when not to)

A decisive section, because the choice is a real cost.

**Use CLIP-only (SD1.5/SDXL-class)** when: you need the smallest, fastest stack; your prompts are short (<60 words) and not text-heavy; you are doing style/aesthetic work where global vibe matters more than exact composition; or you are serving at scale on tight VRAM and every GB counts. CLIP-L is ~0.13GB and instant. Do *not* reach for CLIP-only when you need legible text, reliable attribute binding on multi-object scenes, or long prompts — it will fail on exactly those, structurally, and no sampler or CFG trick fully rescues it.

**Add T5 (SD3/FLUX/PixArt-class)** when: prompts are long or compositional; you need legible typography; attribute binding and spatial relations matter (product shots with multiple labeled objects, infographics, anything with text). The cost is ~10GB and real latency. Mitigate it: run T5 in 8-bit (almost free quality-wise), drop T5 for short prompts when memory is tight (SD3 supports this), or cache the T5 embeddings if you reuse prompts. Do *not* pay for T5 if your whole product is single-subject portrait generation with short prompts — you are buying adherence you don't use.

**Reach for an LLM encoder (SANA-class)** when: you want the strongest instruction understanding, very long or reasoning-heavy prompts, or the lowest latency at a given adherence (SANA's pitch). Caveat: the ecosystem is younger, fine-tuning/LoRA support is thinner than the SD/FLUX world, and aligning LLM hidden states to the denoiser is less turnkey. Do *not* adopt it blind for a production pipeline that depends on a deep LoRA/ControlNet ecosystem — that tooling still lives mostly in the SD/SDXL/FLUX world today.

**Cross-cutting rule:** the encoder sets the *ceiling* on adherence; the denoiser and sampler operate under that ceiling. If your images consistently miss compositional instructions, upgrade the encoder *before* you spend weeks tuning the denoiser or sampler — it is the higher-leverage fix nine times out of ten. And always check the boring failure first: is your prompt being truncated at 77 tokens? Print the `attention_mask` sum. A shocking fraction of "the model ignored my prompt" cases are silent CLIP truncation.

One more framing for the build decision, tying back to the series spine. The encoder choice is a **trilemma trade** like every other choice in the diffusion stack: a richer encoder buys *quality* (adherence, typography, long-prompt fidelity) at the cost of *speed and memory* (the encoder forward, the wider conditioning sequence, the longer attention). The reason the trade is usually favorable is the amortization argument from §1 — the encoder runs once per image while the denoiser runs many times — so even a 4.7B T5 adds a bounded, predictable overhead rather than scaling with step count. That is why the frontier converged on "big encoder, sampled few steps": you pay the encoder once and the denoiser cheaply. If you are building a serving stack, profile the encoder forward as its own line item, cache embeddings for repeated prompts, quantize the encoder before you quantize the denoiser (the encoder tolerates 8-bit better than the denoiser tolerates aggressive quantization), and expose the "drop T5" knob to callers who send short prompts. Those four moves recover most of the cost while keeping the adherence ceiling high — the practical resolution of the conditioning trilemma, and the handoff point into the control and serving posts that follow.

## 13. Key takeaways

- **The encoder is the highest-leverage adherence lever.** It translates language into the vector space the denoiser reads; a weak translation caps adherence no matter how good the backbone is. Upgrade the encoder before the backbone.
- **Two outputs, two jobs.** The *pooled* embedding (one vector) is added to the timestep embedding for *global style*; the *token sequence* (one vector per word) drives *cross-/joint-attention* for *spatial, word-level content*. Don't conflate them.
- **Cross-attention is the steering.** Queries come from the image, keys and values from the text; the `[hw, L]` softmax map says which word each region reads. You can extract and visualize it — that map *is* the prompt's grip on the image.
- **Contrastive CLIP is bag-of-words by construction.** Its loss only flows through pooled embeddings and matches whole images, so it never learns to bind "red" to "cube." That is the mathematical root of attribute-binding failures.
- **T5/LLM encoders + joint attention fix three of the four failures.** Binding, spatial relations, and text rendering are encoder problems that a strong language encoder and two-way attention directly attack. Counting is the stubborn one that needs layout or RL.
- **Watch the 77-token wall.** CLIP truncates silently at 75 usable tokens. Long prompts lose their tail. T5 (256–512) and LLM encoders (thousands) remove the wall — check the attention mask before blaming the model.
- **Pay for T5 deliberately.** It costs ~10GB and latency; quantize it to 8-bit (nearly free), drop it for short prompts, or cache embeddings. Use it when prompts are long, compositional, or contain text; skip it for short single-subject work.
- **Prompt weighting, negatives, and Textual Inversion are all embedding manipulation.** They work because conditioning is "just" vectors fed into attention — weight tokens, repel with negatives via CFG, or learn a brand-new token. Mechanism, not magic.

## Further reading

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021 — the contrastive objective and the encoder that started it all.
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), 2020 — the language encoder SD3/FLUX bolted on.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (LDM/Stable Diffusion), 2022 — where text cross-attention conditioning was introduced for diffusion.
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3), 2024 — MM-DiT joint attention, the CLIP+T5 stack, and the T5-drop ablation.
- Ghosh et al., *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment*, 2023 — the compositional benchmark behind the adherence numbers.
- Yuksekgonul et al., *When and Why Vision-Language Models Behave Like Bags-of-Words* (ARO), 2023 — the empirical case for CLIP's compositional blindness.
- Xie et al., *SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers*, 2024 — the decoder-only LLM encoder and deep-compression AE.
- 🤗 `diffusers` docs — `encode_prompt`, attention processors, and prompt weighting (Compel) — the toolchain for everything in this post.
- Within this series: [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet), [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance), and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
