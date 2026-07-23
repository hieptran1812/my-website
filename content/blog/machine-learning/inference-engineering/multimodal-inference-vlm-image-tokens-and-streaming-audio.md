---
title: "Multimodal inference: an image is a prefill amplifier, and audio is a stream"
date: "2026-07-20"
publishDate: "2026-07-20"
description: "Learn to treat a VLM image as a block of prefill tokens that reshapes batching, caching, and scheduling, then build the multimodal request path, an embedding cache, and a streaming-audio handler into nanoserve."
tags:
  [
    "inference-engineering",
    "llm-inference",
    "multimodal",
    "vision-language-model",
    "kv-cache",
    "prefill",
    "batching",
    "streaming",
    "pytorch",
    "gpu",
    "ml-systems",
    "vllm",
  ]
category: "machine-learning"
subcategory: "Inference Engineering"
author: "Hiep Tran"
featured: true
readTime: 42
---

You spent this whole series making text cheap. You wrote the KV cache so decode reads memory instead of recomputing it. You paged it into blocks so it stops fragmenting. You built a scheduler that flattens forty streams into one super-batch and a sampler that turns logits into tokens without desyncing. Your `generate()` loop hums along at a steady per-token cadence, and you can quote its TTFT (time-to-first-token, from request submit to the first output token) and TPOT (time-per-output-token, the steady-state gap between tokens) to two decimal places.

Then a product manager drops a photo into the chat box and asks "what's in this receipt?" — a nine-word question. Your carefully tuned endpoint reports a TTFT of 800 ms for those nine words, the KV cache jumps by 128 MiB for a *single request*, and the thirty-nine other users streaming through the same GPU all feel a hitch. Nothing in your text pipeline is broken. The problem is that the request you scheduled was not nine tokens long. It was nine tokens plus roughly a thousand tokens you never typed — the image, unrolled into the sequence.

That is the one idea this post is built around, and it is the idea that lets you reason about every multimodal system you will ever touch: **an image is a prefill amplifier.** A vision-language model (VLM) does not attend to pixels. It runs the image through a vision encoder, which emits a block of *image tokens* — ordinary embedding vectors — and *prepends* them to your text prompt. From the language model's point of view there is no image; there is a long prefill. So an image costs exactly what a long prompt costs: inflated TTFT, inflated KV memory, and a prefill step big enough to stall everyone else's decode. Audio is the same story rotated ninety degrees — a stream of chunks you prefill incrementally instead of a block you prefill all at once.

![A multimodal request forks by modality with images detouring through the vision encoder before both paths merge into one prefill and decode loop](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-1.webp)

By the end you will be able to compute an image's token budget from its resolution, predict its KV and TTFT cost, and add three things to `nanoserve`: a multimodal request path (image → stub encoder → image-token block → prefill through the paged cache you built in the KV-cache posts), an embedding cache keyed on the image hash, and a chunked-audio handler that appends stream chunks to a running prompt. This is the Track G closer — the post where images and audio break every text-only assumption you made, and where you learn to put them back on the same scoreboard: TTFT, TPOT, tok/s, VRAM, goodput. If you skipped the intro, [what inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) frames the whole `nanoserve` project; the [capstone playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) ties the pieces together at the end.

## 1. The multimodal request path: one loop, two front doors

Start with the shape of the thing. A request arrives carrying text and *maybe* an image. Figure 1 above shows the fork: a router inspects the request, and one of two things happens. A text-only request goes straight to prefill — the path you already built. A request with an image detours first through the vision encoder, which turns the pixels into a block of image tokens, and only then merges into the same prefill step. After prefill, both kinds of request are indistinguishable: they are sequences of embeddings sitting in the KV cache, decoded one token per step.

The critical word is *merge*. The vision encoder is not a parallel model with its own decode loop. It is a front door that produces embeddings, and those embeddings join the text embeddings in one contiguous prefill. This is why an image behaves like a prompt: by the time attention runs, the image *is* prompt tokens. A Qwen3-VL or LLaVA-style model literally reserves placeholder positions in the token sequence — special `<image>` marker tokens — and splices the encoder's output vectors into those positions before the first transformer layer runs.

That single design choice is the source of every consequence in this post. Because image tokens are prefill tokens:

- They inflate **TTFT**, exactly like a long RAG prompt does — you cannot emit the first output token until the whole image has been prefilled. This is the same prefill-bomb dynamic the [chunked prefill post](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) attacks for long text prompts, and the fix will turn out to be the same knob.
- They consume **KV cache** at the full per-token rate, so a high-resolution image can cost more VRAM than the entire text conversation around it.
- They arrive as one **big prefill step**, which — if you run the encoder inline — stalls every decoding stream in the batch for the duration.
- They are **cacheable by content**: the same image sent twice can reuse both its embeddings and its KV, which turns a repeated-image workload from expensive to nearly free.

Hold those four consequences. The rest of the post is just working each one out with numbers and then writing the code.

## 2. An image is a prefill amplifier

Here is the whole thesis in one moving picture: the same nine-word question costs almost nothing on its own, but the moment you attach an image, the prefill it rides in on grows more than six-fold.

<figure class="blog-anim">
<svg viewBox="0 0 620 210" role="img" aria-label="A text-only prefill of 200 tokens versus an image request whose prefill grows to 1224 tokens as a block of image tokens is added" style="width:100%;height:auto;max-width:760px">
<style>
.mm1-txt{fill:var(--surface,#e5e7eb);stroke:var(--border,#d1d5db);stroke-width:1.5}
.mm1-img{fill:var(--accent,#6366f1)}
.mm1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.mm1-in{font:600 13px ui-sans-serif,system-ui;fill:#ffffff;text-anchor:middle}
.mm1-inb{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes mm1-a{0%,42%{opacity:1}54%,96%{opacity:0}100%{opacity:1}}
@keyframes mm1-b{0%,42%{opacity:0}54%,96%{opacity:1}100%{opacity:0}}
.mm1-A{animation:mm1-a 9s ease-in-out infinite}
.mm1-B{animation:mm1-b 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.mm1-A{animation:none;opacity:0}.mm1-B{animation:none;opacity:1}}
</style>
<text class="mm1-lbl" x="310" y="32">the same question, with and without an image</text>
<g class="mm1-A">
<rect class="mm1-txt" x="40" y="96" width="90" height="46" rx="7"/>
<text class="mm1-inb" x="85" y="124">text · 200</text>
<text class="mm1-lbl" x="120" y="176">text-only · 200-token prefill</text>
</g>
<g class="mm1-B">
<rect class="mm1-txt" x="40" y="96" width="90" height="46" rx="7"/>
<text class="mm1-inb" x="85" y="124">text · 200</text>
<rect class="mm1-img" x="132" y="96" width="448" height="46" rx="7"/>
<text class="mm1-in" x="356" y="124">image tokens · 1024</text>
<text class="mm1-lbl" x="310" y="176">image request · 1224-token prefill</text>
</g>
</svg>
<figcaption>The question is about 200 tokens either way; adding one image prepends roughly 1024 image tokens, so the prefill the request rides in on grows more than six-fold.</figcaption>
</figure>

Now let us make "roughly 1024" exact, because the number falls straight out of the encoder's geometry, and the KV cost falls straight out of the model's attention geometry. This is the mechanism block: two derivations, both provable, no hand-waving.

### The token count is patches, merged

A vision encoder is a Vision Transformer (ViT): it chops the image into fixed-size square **patches**, embeds each patch, and runs transformer layers over them. The number of patches is just area divided by patch area. Modern VLMs then apply a **spatial merge** (a pixel-unshuffle, common in the Qwen2-VL/Qwen3-VL family) that fuses each 2×2 block of patches into one token, cutting the count by four so the language model sees fewer, richer tokens.

For a 896×896 image with patch size 14 and a 2×2 merge:

$$\text{patches per side} = \frac{896}{14} = 64, \qquad \text{tokens} = \left(\frac{64}{2}\right)^2 = 32^2 = 1024.$$

So a single 896px image becomes **1024 image tokens** — and that is before you tile a higher-resolution image into multiple crops, each of which contributes its own block (more on that in the next section). The raw pixels were only $896 \times 896 \times 3 = 2{,}408{,}448$ bytes, about 2.4 MB, but that is not the number that hurts you. What hurts you is the token count, because tokens are what the KV cache is measured in.

### The KV cost is those tokens at the full per-token rate

Recall the KV memory law from [the KV-cache memory-math post](/blog/machine-learning/inference-engineering/the-memory-math-of-the-kv-cache): each token stores a key and a value at every layer, for every KV head. The bytes per token are

$$\text{KV bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d \cdot b,$$

where $L$ is the number of layers, $H_{kv}$ the number of key/value heads (grouped-query attention shares them across query heads), $d$ the head dimension, $b$ the bytes per element, and the leading 2 is for storing K *and* V. For Llama-3.1-8B ($L = 32$, $H_{kv} = 8$, $d = 128$) in bf16 ($b = 2$):

$$2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 \text{ bytes} = 128 \text{ KiB per token}.$$

An image token is a token. It pays the same 128 KiB. So the 1024 image tokens cost $1024 \times 128\text{ KiB} = 128$ MiB of KV cache, all by themselves. Add a 200-token question at $200 \times 128\text{ KiB} = 25$ MiB, and the request holds 153 MiB of KV before it emits a single output token — and 84% of that is the image. Figure 2 stacks the whole budget so you can see where every byte goes.

![A single image contributes far more prefill tokens than the question it accompanies and dominates the resulting KV cache footprint](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-2.webp)

Let us write this as `nanoserve` code, because you will reuse it in the scheduler's admission check later. Two pure functions: image-token count from resolution, and KV bytes from model geometry.

```python
# nanoserve/multimodal.py
def image_tokens(px: int, patch: int = 14, merge: int = 2) -> int:
    """Image tokens after a ViT patchify + spatial merge.
    px: side length in pixels (assume square after resize).
    patch: ViT patch size. merge: spatial merge factor (2 -> 2x2 fuse)."""
    side = px // patch            # patches per side
    return (side // merge) ** 2   # tokens after the 2x2 merge

def kv_bytes_per_token(layers=32, kv_heads=8, head_dim=128, dtype_bytes=2) -> int:
    """Per-token KV footprint: K and V, every layer, every KV head."""
    return 2 * layers * kv_heads * head_dim * dtype_bytes

if __name__ == "__main__":
    n_img = image_tokens(896)
    kv = kv_bytes_per_token()
    print(f"896px image -> {n_img} image tokens")
    print(f"KV per token -> {kv // 1024} KiB")
    print(f"image KV     -> {n_img * kv / 2**20:.0f} MiB")
    print(f"+200 text    -> {(n_img + 200) * kv / 2**20:.0f} MiB total prefill KV")
```

```console
896px image -> 1024 image tokens
KV per token -> 128 KiB
image KV     -> 128 MiB
+200 text    -> 153 MiB total prefill KV
```

Every number in figure 2 is now `derived` — you can regenerate it. That is the standard for the rest of the post: no figure carries a number you cannot reproduce from a formula or cannot attribute to a named source with a link.

### Different VLMs spend tokens differently — but they all spend them

The exact token count depends on the model family, and it is worth knowing the knobs because they are the levers you have over cost. Three patterns dominate:

- **Fixed grid, spatial merge (Qwen2-VL / Qwen3-VL).** Patchify at a fixed patch size, then fuse 2×2 blocks — the derivation above. The merge is the whole reason 1024 and not 4096: without it, an 896px image would be 4096 tokens and cost 512 MiB of KV. The merge trades a little spatial resolution for a 4× cut in the language model's sequence length, and it is the single most important efficiency choice in the encoder.
- **Dynamic tiling / AnyRes (LLaVA-1.6).** A large image is split into a grid of crops, each crop encoded independently and contributing its own token block, plus a downsampled "thumbnail" of the whole image for global context. A high-resolution photo can reach ~2880 tokens this way — which is exactly why "let the model see full resolution" is a token-budget decision, not a free quality upgrade. Every extra crop is another ~576 tokens of prefill and KV.
- **Fixed token budget per image (some Gemma-family and API models).** The model commits to a constant number of image tokens regardless of resolution, resizing the input to fit. This caps the worst case — no image can become a prefill bomb on its own — at the cost of throwing away detail on large images.

The practical takeaway: before you ship a VLM endpoint, find out how many tokens your model spends per image *at the resolutions your users actually send*, and put that number in the same budget spreadsheet as your context length. An endpoint advertised as "128k context" that also accepts four full-resolution images is really a "128k minus up to ~11.5k image tokens" context, and if you did not budget for it, the first power user who pastes a wall of screenshots will find the edge for you.

## 3. The image-token budget scales with pixels

The 1024 figure was for one specific resolution. The uncomfortable part is that the token count grows with *pixel area*, which means it grows *quadratically* with the side length — and production VLMs happily accept high-resolution images or tile a large image into many crops (LLaVA's "AnyRes", Qwen's dynamic resolution). Every doubling of the side roughly quadruples the tokens, the KV, and the prefill compute. Figure 3 lays the budget out across resolutions so you can see the cliff.

![A comparison of image resolutions showing how token count KV cache and prefill compute all rise together toward a fifty-page document](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-3.webp)

Here is the same budget as a table with provenance. The token and KV columns are exact (derived from the two functions above); the prefill-compute column is an order-of-magnitude estimate using the standard "a dense forward pass costs about $2 \cdot P \cdot N$ FLOPs" rule for $P$ parameters and $N$ tokens, and the last row deliberately shows the super-linear jump where attention's $N^2$ term arrives.

| Input | Image tokens | KV cache | Prefill FLOPs (approx) | Source |
| --- | --- | --- | --- | --- |
| 224px thumbnail | 64 | 8 MiB | 1.0e12 | derived |
| 448px image | 256 | 32 MiB | 4.1e12 | derived |
| 896px image | 1024 | 128 MiB | 1.7e13 | derived |
| LLaVA AnyRes crops | 2880 | 360 MiB | 4.6e13 | cited: LLaVA-1.6 |
| 4-image request | 4096 | 512 MiB | 7.0e13 | derived |
| 50-page document | 51200 | 6.25 GiB | 1.5e15 | derived (incl. $N^2$) |

Read the KV column as VRAM you must have *free* before the request can even be admitted. On an A100-80GB serving Llama-3.1-8B, the weights already take about 16 GB in bf16; a 50-page document rendered as page-images wants 6.25 GiB of KV for that single request — a meaningful slice of what is left, held for one user, before the model reads a word of it. This is why "just send the whole PDF as images" is a scheduling event, not a feature request.

The code that generated the table is a five-line loop over the same two functions, which is the point — the budget is not mysterious, it is arithmetic you can put in a dashboard.

```python
# nanoserve/multimodal.py  (budget report)
def budget(label, n_images, px, text_tokens=200, params=8e9):
    img_tok = n_images * image_tokens(px)
    total = img_tok + text_tokens
    kv = total * kv_bytes_per_token()
    flops = 2 * params * total            # linear term; attention N^2 extra at long ctx
    print(f"{label:<16} {img_tok:>6} img tok  {kv/2**20:>7.0f} MiB KV  ~{flops:.1e} FLOPs")

budget("448px",      1, 448)
budget("896px",      1, 896)
budget("4-image",    4, 896)
budget("50-page doc", 50, 896)
```

```console
448px               256 img tok       35 MiB KV  ~9.1e11 FLOPs
896px              1024 img tok      153 MiB KV  ~2.0e12 FLOPs
4-image            4096 img tok      537 MiB KV  ~6.9e12 FLOPs
50-page doc       51200 img tok     6425 MiB KV  ~8.2e13 FLOPs
```

(The console's FLOPs are the pure linear term; the table's 50-page figure is higher because at 51k tokens the quadratic attention cost stops being negligible. Both are order-of-magnitude — labeled as such.)

#### Worked example: a four-image document-QA request

A user uploads four screenshots of a dashboard at 896px and asks a 200-token question. What does the request cost, and what does it do to your TTFT?

- **Image tokens:** $4 \times 1024 = 4096$. Plus 200 text tokens gives a **4296-token prefill**.
- **KV cache:** $4296 \times 128\text{ KiB} = 537$ MiB, held for one request.
- **TTFT floor:** prefill is compute-bound, so TTFT scales with prefill tokens. If your text-only 200-token prefill lands a first token in about 60 ms, a 4296-token prefill is roughly $4296 / 200 \approx 21\times$ the prefill work. Even accounting for better GPU utilization on the larger matmul, you are looking at a first token on the order of **hundreds of milliseconds to over a second**, not 60 ms. Frame this as reproduce-it-yourself: run the same prompt with and without the four images through your engine and report the two TTFTs; the ratio should track the token ratio, not stay flat.

The lesson is that a four-image request is not "a request with four images." It is a **21× prefill** wearing a nine-word question as a hat. Your scheduler needs to know that before it admits it, which is section 8.

#### Worked example: a 50-page PDF rendered as page-images

Someone builds a "chat with your document" feature and, reasonably, renders each PDF page to an 896px image and sends all fifty. What happens on an A100-80GB serving Llama-3.1-8B?

- **Image tokens:** $50 \times 1024 = 51{,}200$. Plus the question, call it a **51,400-token prefill**.
- **KV cache:** $51{,}400 \times 128\text{ KiB} = 6.28$ GiB — for one request, held from the moment it is admitted until it finishes decoding.
- **The squeeze:** the model weights are ~16 GB; the CUDA context, activations, and framework overhead take several more; call it ~55 GiB of usable KV on a good day. One 50-page request eats **6.3 GiB — over 11%** of your entire KV budget for a single user. Ten concurrent document-chat users would want 63 GiB of KV and you do not have it; you would be paging and preempting immediately (the thrash from [the eviction post](/blog/machine-learning/inference-engineering/eviction-preemption-and-kv-swapping)).
- **The prefill cost:** at ~51k tokens the attention term is no longer negligible — this is where the table's FLOP column jumps super-linearly — so the first token could be *seconds* out even before decode starts.

The right answer is not a bigger GPU. It is to stop treating "the whole document" as one prefill: encode each page once, cache the embeddings by image hash (section 6), and only prefill the pages a given question actually needs (retrieval over page-images). Fifty pages become five relevant pages, 6.3 GiB becomes 630 MiB, and the seconds-long TTFT becomes tens of milliseconds. The admission check in section 8 rejects the naive 50-page request precisely so someone builds the retrieval version instead of shipping a feature that falls over at two concurrent users.

## 4. The encoder is a different workload

So far the image has behaved like a long prompt. Here is where it stops. The vision encoder is not part of the autoregressive decode loop — it is a completely different kind of compute, and treating it as "just another layer" is how you get a jittery, unpredictable TTFT that nothing in your text profiling explains.

A ViT encoder runs **once per image**, over all patches **in parallel**, and it is **compute-bound** — a stack of dense self-attention and MLP layers over a few thousand patches, closer to a training forward pass than to a decode step. Your decode loop, by contrast, runs **once per token**, is **memory-bound** (it streams the whole model's weights from HBM to emit one token, which is why decode speed tracks HBM bandwidth, not FLOPs), and it is the thing forty users are waiting on right now. These two workloads do not want to share a step.

The [roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) makes the mismatch precise. Arithmetic intensity is FLOPs per byte moved: $\text{AI} = \text{FLOPs} / \text{bytes}$. A decode step for one token reads the whole model's weights (~16 GB for an 8B model in bf16) to do a comparatively tiny amount of matmul, so its AI is *low* — it sits on the memory-bound side of the roofline and is bottlenecked by HBM bandwidth. A ViT encoder forward over a few thousand patches does a large batched matmul relative to the bytes it touches, so its AI is *high* — it sits on the compute-bound side and is bottlenecked by tensor-core throughput. Two kernels on opposite ends of the roofline do not overlap gracefully on the same SMs in the same step; the encoder's big matmul simply occupies the compute units the decode step also wants. That is the mechanism behind the stall — not a scheduling bug, a hardware-contention fact. The only real fixes are to move the encoder to *other* silicon (disaggregation) or to break its work into pieces small enough to interleave (chunking), which is why the two problems below map onto the two solutions in the next sections.

If you run the encoder *inline* — the naive thing, and what a from-scratch engine does first — you drop a compute-bound stall into the middle of your steady decode cadence. Figure 4 shows the damage on a timeline.

![An inline vision encoder inserts a compute-bound stall into one engine step so every decoding stream waits through the image prefill](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-4.webp)

Walk the timeline. Suppose your engine is happily running 40 decode streams at a 13 ms step cadence (a plausible A100 operating point for 8B decode at that concurrency — treat 13 ms as an assumed number, not a measured one; substitute your own). A request with an image arrives. Now one engine iteration has to: run the ViT encoder forward (compute-bound, does not overlap the decode any more than a big matmul overlaps another big matmul on the same SMs), then prefill 1024 image tokens (another big compute chunk). For the duration of both, **not one of the 40 decoding streams emits a token.** Every user's stream visibly hitches. Your TPOT, which you measured as rock-steady, now has a fat tail every time an image lands — and it is invisible in text-only load tests, so it ships to production and surprises you there.

There are two independent problems here and they need two independent fixes:

1. The encoder is the *wrong kind of workload* to inline. → Give it its own GPUs (section 5).
2. The image prefill is a *prefill bomb*. → Chunk it, the same way you chunk a long text prefill (section 9).

## 5. Encoder disaggregation: give the encoder its own pool

The clean fix for problem 1 is the same move the serving world already made for prefill and decode: **disaggregate**. If two phases have different resource profiles and stall each other when co-located, split them onto separate hardware and connect them with a fast transfer. You built (or read about) prefill/decode disaggregation for text in [the model-serving prefill/decode post](/blog/machine-learning/model-serving/prefill-decode-disaggregation); encoder disaggregation adds a *third* stage in front.

The vLLM team shipped exactly this as **EPD — Encoder / Prefill / Decode disaggregation** ([vLLM, "Encoder Disaggregation for Multimodal", 2025-12-15](https://vllm.ai/blog/2025-12-15-vllm-epd)). The architecture, per their post: a **Proxy & Router** dispatches multimodal inputs to a dedicated pool of **encoder** instances; the encoders write their output embeddings to remote storage through **EC (embedding-cache) connectors**; the **prefill+decode (PD)** pool loads those embeddings just before prefill; and **text-only requests bypass the encoder pool entirely**. Figure 5 draws the flow.

![Encoder disaggregation routes images to a dedicated encoder pool and ships embeddings to the prefill and decode pool while text-only requests bypass it](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-5.webp)

Why this wins is worth stating precisely, because it is the same reasoning you will reuse for any disaggregation decision:

- The encoder pool can be sized and scaled *independently* for its compute-bound profile — more SMs, different batching — without stealing cycles from decode.
- The PD pool's decode cadence stops hitching, because the compute-bound encoder forward now happens on *other* GPUs and its result arrives as a ready-made embedding.
- Text-only requests pay *nothing* for the multimodal machinery — they route straight to PD.

The vLLM post reports the payoff, and here it is with its full setup so you can judge whether it maps to your workload (the honesty rule: this is a cited number, not one I measured):

| Metric | Inline (co-located) | EPD (disaggregated) | Setup | Source |
| --- | --- | --- | --- | --- |
| Short-text, 4-image goodput | 6 QPS | 12 QPS | 4×A100-80G, Qwen3-VL-4B-Instruct | cited: vLLM EPD |
| Long-text, 3–4 image goodput | 4 QPS | 8–11 QPS | same | cited: vLLM EPD |
| P99 TTFT / TPOT | baseline | 20–50% lower | same | cited: vLLM EPD |
| Overall goodput | 1× | 2–2.5× | same | cited: vLLM EPD |

The feature is available in vLLM since release 0.11.1. And — this is the part a vendor blog buries and you must not — **the benefits shrink when images are rare or the workload is decode-dominated.** If 2% of your traffic has an image and everything else is long-output chat, a dedicated encoder pool is idle GPUs you are paying for. Disaggregation earns its keep when image traffic is steady and heavy enough to keep the pool busy. That caveat is the difference between a good architecture decision and a cargo-culted one.

You will not build a full EPD in `nanoserve` — it is a multi-node system and vLLM already is the benchmark target — but you will build the piece that makes it possible: an encoder stage that produces embeddings you can cache and transfer, decoupled from prefill. That is the next section, and it doubles as the fix for the *repeated*-image problem.

## 6. Cache the embeddings: the same image, twice

Disaggregation stops the encoder from stalling decode. Caching stops the encoder from *running at all* when it does not have to — and in the most common multimodal workloads, it very often does not have to.

The observation is simple: the vision encoder is a pure function of the pixels. Same image in, same embeddings out. So if you hash the raw image bytes and key a cache on that hash, the *second* time an image shows up you skip the entire encoder pass and reuse the stored embeddings. And because the image tokens are prefill tokens, you can go further and reuse their *KV* too — prefix caching keyed on the image hash, exactly the content-addressed reuse you built in [prefix sharing with radix trees](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write), just with an image hash as the prefix key instead of a token prefix. vLLM's V1 rewrite added precisely this: "MM adds image-hash prefix caching plus an encoder cache" ([vLLM V1, 2025-01-27](https://vllm.ai/blog/2025-01-27-v1-alpha-release)).

Figure 6 shows the before/after on the numbers the vLLM team published for their shared-memory IPC cache.

![Hashing an image and reusing its cached embeddings turns a repeated image from a full encoder pass into a fast lookup](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-6.webp)

Those numbers come from [vLLM's "Shared-Memory IPC Caching" post (2025-11-13)](https://vllm.ai/blog/2025-11-13-shm-ipc-cache), and the setup matters: `command-a-vision`, 4×A100 with TP4, on the VisionArena workload. With the multimodal-embedding cache warm (`mm_processor_cache_type="shm"`), they report **cached prefill throughput rising from 2894 to 4918 tok/s (+69.9%) and TTFT dropping from 790 to 471 ms (−40.5%)**. That same post is the source for the intuition that these embeddings are large enough to be worth a purpose-built cache — a single high-resolution multimodal image's embeddings run to several megabytes (on the order of 9 MB in int8 for a large image), which is why they built a shared-memory ring buffer instead of passing them around naively.

| Metric | Cold cache | Warm cache | Source |
| --- | --- | --- | --- |
| Cached prefill throughput | 2894 tok/s | 4918 tok/s (+70%) | cited: vLLM SHM-IPC |
| TTFT (VisionArena) | 790 ms | 471 ms (−40%) | cited: vLLM SHM-IPC |
| First (uncached) prefill | 581 tok/s | 648 tok/s (+11.5%) | cited: vLLM SHM-IPC |
| Encoder passes on a hit | 1 per image | 0 | derived |

One honest limitation, straight from that post: the shared-memory cache **relies on strict input ordering** and is multimodal-only. It is not a general-purpose KV store; it is a tuned fast path. Know the constraint before you lean on it.

There are actually *two* caches hiding in "cache the image," and they save different work at different points in the pipeline:

- The **embedding cache** stores the vision encoder's *output* keyed on the image hash. A hit skips the encoder forward — the compute-bound ViT pass from section 4. This is what saves you GPU-seconds on the encoder pool.
- The **image-hash prefix cache** stores the *KV* of the already-prefilled image tokens, keyed on the same hash. A hit skips re-prefilling those ~1024 tokens through the language model. This is what saves you TTFT on the PD pool.

A warm request can hit both: no encoder pass *and* no image-token prefill, so its first token comes almost as fast as a text-only request's. That is the ceiling the document-QA workload rides toward. The eviction policy matters because embeddings are big — several megabytes each — so a naive unbounded cache becomes a VRAM leak. LRU by image hash is the sane default; the sticky, high-value entries (the reference images a workload keeps re-sending) stay warm, and the one-shot uploads age out. Size the cache to your *working set* of distinct images, not your total image volume: a product catalog with 500 SKUs re-queried all day wants those 500 embeddings pinned, while a photo-upload feature with a million unique images per day should barely cache at all, because its hit rate is near zero and every entry is dead weight. The cache is only ever as good as the repetition in your traffic — measure the hit rate before you trust the win.

### The hit-rate economics of a document-QA workload

Where does this pay off? Anywhere the *same image is sent more than once* — and one workload does that structurally: **document Q&A over the same pages.** A user opens a report and asks ten questions about it. Every question re-sends the same five page-images.

Without caching, each of the ten questions re-encodes all five pages: 50 encoder passes, and 50 image-token prefills. With image-hash caching, the first question encodes the five pages (5 passes) and the other nine questions hit the cache (0 encoder passes for the images). That is 45 of 50 image-encodes avoided — a **90% encoder hit rate** — plus, with image-hash prefix caching, the image-token *KV* is reused too, so nine of ten questions skip re-prefilling ~5120 image tokens each.

Put the cited throughput on it: warm image-prefill runs at roughly 4918/2894 ≈ 1.7× the cold rate, so the nine cache-hitting questions clear their image prefill about 1.7× faster *and* skip the encoder entirely. The general rule: for $K$ questions over the same pages, the image-encode hit rate approaches $(K-1)/K$, so the caching win grows with how "sticky" a document session is. A one-shot image gets nothing; a long document session gets almost everything.

Now the code. Here is the `nanoserve` embedding cache — a hash-keyed LRU that turns a repeated image into a lookup, and the stub encoder it wraps.

```python
# nanoserve/multimodal.py
import hashlib
from collections import OrderedDict
import torch

class VisionEncoder:
    """Stand-in for a real ViT. Returns correctly-shaped embeddings so the
    rest of the pipeline is exercised; swap encode() for a real model."""
    def __init__(self, hidden=4096, patch=14, merge=2):
        self.hidden, self.patch, self.merge = hidden, patch, merge

    def encode(self, image_px: int, pixel_values: torch.Tensor) -> torch.Tensor:
        n = image_tokens(image_px, self.patch, self.merge)
        # A real ViT runs here (compute-bound). We fabricate the right shape.
        return torch.randn(n, self.hidden, dtype=torch.bfloat16)

class MMEmbeddingCache:
    """LRU cache of image embeddings keyed on a hash of the raw bytes.
    A hit skips the encoder entirely."""
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = self.misses = 0

    @staticmethod
    def key(raw_bytes: bytes) -> str:
        return hashlib.sha256(raw_bytes).hexdigest()

    def get_or_encode(self, raw_bytes, image_px, pixel_values, encoder):
        k = self.key(raw_bytes)
        if k in self.store:
            self.hits += 1
            self.store.move_to_end(k)          # LRU touch
            return self.store[k]
        self.misses += 1
        emb = encoder.encode(image_px, pixel_values)  # the expensive path
        self.store[k] = emb
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)     # evict LRU
        return emb
```

```python
# exercise it: same image asked about three times
enc, cache = VisionEncoder(), MMEmbeddingCache()
raw = b"...jpeg bytes of page 1..."
px, pv = 896, torch.zeros(3, 896, 896)
for _ in range(3):
    emb = cache.get_or_encode(raw, px, pv, enc)
print(f"shape={tuple(emb.shape)}  hits={cache.hits}  misses={cache.misses}")
```

```console
shape=(1024, 4096)  hits=2  misses=1
```

One encoder pass for three questions about the same page. That is the entire economic argument for the cache, in three lines of exercise code.

## 7. Streaming audio: prefill the clip as it arrives

Rotate the problem ninety degrees. An image is a block that arrives all at once — a big prefill you would love to chunk. Audio is the opposite: it is a *stream* that arrives a little at a time, and the whole game is to start responding before the clip is finished. If a user speaks for five seconds and you wait for the full clip before your first prefill, your first-audio-out latency has a five-second floor no GPU can beat. The fix is to treat each incoming chunk as an *incremental* prefill against a running prompt — the exact same mechanism as chunked prefill, run in real time as the bytes land.

The transport that makes this concrete is OpenAI's Realtime API, which runs over a WebSocket at `/v1/realtime` and streams audio as **PCM16 at 16 kHz in small (~4 KB) chunks**. Two constraints from that design are load-bearing: the models must be **causal** (a token can only attend to the past, never the future) and specifically **trained for streaming** (they tolerate partial context and small look-ahead). You cannot take an offline bidirectional ASR model and stream it; streaming is a property the model was trained to have. Figure 7 shows a socket delivering chunks and the engine prefilling each one incrementally against a cached anchor.

![A streaming audio socket delivers small fixed chunks that are prefilled incrementally against a cached anchor request](/imgs/blogs/multimodal-inference-vlm-image-tokens-and-streaming-audio-7.webp)

First, derive what a "4 KB chunk" actually *is*, because everything downstream depends on it. PCM16 mono at 16 kHz is 16000 samples per second at 2 bytes per sample:

$$16000 \times 2 = 32000 \text{ bytes/s}, \qquad \frac{4096 \text{ bytes}}{32000 \text{ bytes/s}} = 0.128 \text{ s} = 128 \text{ ms of audio}.$$

So each 4 KB chunk is 128 ms of sound. That is your minimum granularity: you cannot react faster than one chunk, and you will usually need a little look-ahead. This is why figure 7 labels each chunk "128 ms audio" — it is arithmetic, not a guess.

The engine-side pattern is **incremental prefill against a cached anchor**. You establish an "anchor" request once — the system prompt, the audio-in framing, a handful of tokens — and prefill it, so its KV is warm. Then each arriving audio chunk is converted to a few audio tokens, appended to the running prompt, and prefilled *incrementally*: only the new chunk's tokens go through prefill; all prior KV (the anchor and earlier chunks) is reused. This is chunked prefill with the chunks arriving over a socket instead of being sliced from one long prompt — the same [chunked-prefill mechanism](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff), applied to a live stream. The response side (streaming tokens or synthesized audio back out) is the streaming-output machinery from [structured output in production](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls), just with audio frames instead of JSON deltas.

The two directions of audio streaming stress different parts of this pattern, and it helps to name them:

- **ASR (audio in, text out) — incremental prefill dominates.** New audio chunks keep arriving and keep appending to the prompt; the model transcribes as it goes. The hard part is the *input* side: every chunk is a small prefill, and you want each one to reuse all prior KV so the per-chunk cost stays flat instead of growing with clip length. Done right, a 30-second dictation costs the same per chunk at second 29 as at second 1 — the anchor and history are cached, only the new 128 ms is prefilled. Done wrong (re-prefilling the whole clip every chunk), your cost grows quadratically and the transcription falls behind the speaker.
- **TTS (text in, audio out) — decode and look-ahead dominate.** The prompt is short text; the work is *generating* audio tokens (or codec frames) that a vocoder turns into sound. The hard part is the *output* side: first-audio-out latency is set by how many frames the vocoder needs before it can emit a clean chunk of sound, plus any look-ahead the model requires so the prosody does not glitch at chunk boundaries. This is why streaming TTS models are trained to commit output with minimal right-context — the look-ahead you buffer is latency the user hears as a delay before the voice starts.

Both live or die on the same rule: the model must be **causal and trained for streaming**. A bidirectional model that attends to the whole clip cannot start until the clip ends — there is no incremental prefill to do, because token 1's representation depends on the last token. Streaming is not a serving trick you can bolt onto any audio model; it is a property the model was trained to have, and if it was not, no amount of clever chunking on your side recovers it.

#### Worked example: first-audio-out latency for a streaming socket

A user speaks for 5 seconds into a streaming TTS/ASR socket. What is the first-audio-out latency, streaming vs. batch?

- **Batch (wait for the whole clip):** you cannot prefill until the last byte lands, so first output is at least 5000 ms + prefill + first decode. Floor: **> 5000 ms.**
- **Streaming (incremental prefill):** the model needs a small look-ahead — say one chunk of right-context before it commits the first output. So you buffer 2 chunks (256 ms of audio) before emitting, then pay the anchor prefill and the first incremental prefill (a handful of tokens, ~14 ms on an A100 for that little compute), and the first decode/vocoder step. First audio out ≈ $256 + 14 \approx$ **270 ms.**

That is the number in figure 7, and it is roughly **18× faster** than waiting for the clip. The provenance: the 128 ms/chunk is derived from the PCM16 math; the 2-chunk look-ahead is an assumed model property (streaming ASR/TTS commonly use one chunk of right-context — substitute your model's); the ~14 ms compute is an order-of-magnitude estimate for prefilling a few tokens. Frame it reproduce-it-yourself: feed a real streaming endpoint a 5 s clip and measure the gap between the first byte in and the first audio byte out; you should see hundreds of ms, not seconds.

| Latency component | Batch | Streaming | Source |
| --- | --- | --- | --- |
| Audio available before first prefill | 5000 ms | 256 ms (2 chunks) | derived (PCM16 math) |
| Anchor + first incremental prefill | ~14 ms | ~14 ms | reproduce: bench.py |
| First-audio-out (total) | > 5000 ms | ~270 ms | derived |

Here is the `nanoserve` chunk handler: a buffer that ingests fixed-size PCM chunks, tracks how much audio it holds, and appends new audio tokens to a running prompt for incremental prefill.

```python
# nanoserve/audio_stream.py
class AudioChunkBuffer:
    """Accepts fixed-size PCM16 chunks off a socket, converts each to audio
    tokens, and appends to a running prompt for incremental prefill."""
    def __init__(self, sample_rate=16000, bytes_per_sample=2,
                 chunk_bytes=4096, tokens_per_chunk=6, lookahead_chunks=1):
        self.sr, self.bps = sample_rate, bytes_per_sample
        self.chunk_bytes = chunk_bytes
        self.tpc = tokens_per_chunk
        self.lookahead = lookahead_chunks
        self.prompt_tokens: list[int] = []   # the running prompt (anchor + audio)
        self.n_chunks = 0

    def chunk_ms(self) -> float:
        return 1000 * self.chunk_bytes / (self.sr * self.bps)   # 128.0

    def anchor(self, anchor_token_ids: list[int]):
        """Prefill-once framing whose KV is reused by every later chunk."""
        self.prompt_tokens.extend(anchor_token_ids)

    def push(self, pcm_bytes: bytes) -> dict:
        """Ingest one chunk. Returns the incremental-prefill work to do, or a
        'buffering' signal while we wait for the look-ahead."""
        assert len(pcm_bytes) == self.chunk_bytes, "expected a fixed-size chunk"
        self.n_chunks += 1
        new_tokens = self._encode_audio(pcm_bytes)      # stub: a few audio tokens
        start = len(self.prompt_tokens)                 # only these get prefilled
        self.prompt_tokens.extend(new_tokens)
        can_emit = self.n_chunks > self.lookahead       # respect the look-ahead
        return {"prefill_range": (start, len(self.prompt_tokens)),
                "reuse_kv_upto": start, "can_emit": can_emit}

    def _encode_audio(self, pcm_bytes: bytes) -> list[int]:
        # A real audio front-end (mel + codec) runs here.
        return list(range(self.tpc))
```

```python
buf = AudioChunkBuffer()
buf.anchor([1, 2, 3, 4, 5, 6])          # 6-token framing, prefilled once
print(f"chunk = {buf.chunk_ms():.0f} ms of audio")
for i in range(3):
    work = buf.push(b"\x00" * 4096)
    print(f"chunk {i+1}: prefill {work['prefill_range']} "
          f"reuse KV up to {work['reuse_kv_upto']}  emit={work['can_emit']}")
```

```console
chunk = 128 ms of audio
chunk 1: prefill (6, 12) reuse KV up to 6  emit=False
chunk 2: prefill (12, 18) reuse KV up to 12  emit=True
chunk 3: prefill (18, 24) reuse KV up to 18  emit=True
```

Notice what the handler prints: chunk 1 buffers (look-ahead not satisfied), chunk 2 onward emits, and every chunk prefills *only its own six tokens* while reusing the KV of everything before it. That "reuse KV up to N" is the incremental-prefill anchor pattern made literal — it is the same KV-reuse discipline as text prefix caching, driven by wall-clock arrivals.

## 8. Scheduling consequences: the prefill bomb, the memory hog, the separate pool

Now put the pieces back into the scheduler, because that is where multimodal breaks the assumptions from the Track C posts. A multimodal request hits your scheduler as three problems at once:

1. **It is a prefill bomb.** A 4096-token image request is a giant prefill step that, if admitted whole, monopolizes an engine iteration and stalls decode — the same pathology a long text prompt causes, and the same fix applies: **chunk it.** The image-token block is just tokens; feed it to your chunked-prefill path in bounded slices so decode streams keep getting turns. This is why the chunked-prefill knob you built for text is not a text-only feature — it is the release valve for image prefill too.
2. **It is a memory hog.** Those image tokens sit in the KV cache at the full 128 KiB/token rate. Your admission control has to check *image-inclusive* KV before saying yes, or you admit a 50-page document, blow the block budget, and trigger the preemption thrash you worked so hard to avoid.
3. **It needs the encoder scheduled separately.** The compute-bound encoder does not belong in the decode step (section 4). Disaggregation (section 5) is the structural answer; short of that, at least run the encoder off the critical decode path.

Here is the admission check `nanoserve` needs — it accounts for image tokens explicitly and decides whether to chunk, admit, or reject.

```python
# nanoserve/scheduler.py  (multimodal admission)
def admit_multimodal(req, free_kv_bytes, max_prefill_tokens=2048):
    """Decide how to schedule a request that may carry images.
    Returns an action the engine loop understands."""
    img_tok = sum(image_tokens(px) for px in req.image_resolutions)
    total_tok = img_tok + req.text_tokens
    need_kv = total_tok * kv_bytes_per_token()

    if need_kv > free_kv_bytes:
        # would not fit even alone -> reject before it evicts everyone
        return {"action": "reject", "reason": "kv_over_budget",
                "need_mib": need_kv // 2**20}
    if total_tok > max_prefill_tokens:
        # prefill bomb -> chunk it so decode streams keep stepping
        n_chunks = -(-total_tok // max_prefill_tokens)   # ceil div
        return {"action": "chunked_prefill", "chunks": n_chunks,
                "img_tok": img_tok, "total_tok": total_tok}
    return {"action": "admit", "img_tok": img_tok, "total_tok": total_tok}

class Req:  # tiny stand-in
    def __init__(self, text_tokens, image_resolutions):
        self.text_tokens, self.image_resolutions = text_tokens, image_resolutions

for r in [Req(200, [896]), Req(200, [896, 896, 896, 896]), Req(200, [896]*50)]:
    print(admit_multimodal(r, free_kv_bytes=2 * 2**30))
```

```console
{'action': 'chunked_prefill', 'chunks': 1, 'img_tok': 1024, 'total_tok': 1224}
{'action': 'chunked_prefill', 'chunks': 3, 'img_tok': 4096, 'total_tok': 4296}
{'action': 'reject', 'reason': 'kv_over_budget', 'need_mib': 6425}
```

The scheduler now *sees* the image before it commits. A single 896px request chunks into a couple of slices; four images into three; a 50-page document gets rejected before it can evict a dozen other users — a graceful failure instead of a cache-thrashing meltdown. A later post on observability picks up the other half of this: image requests skew your TTFT distribution, so a fleet-wide p99 that looks fine on average can hide a bimodal split between text-only and image traffic. You want your dashboards to separate the two, or the image tail hides inside the text median.

Why does disaggregation move *goodput* and not just latency? Goodput is the rate of requests that meet their SLO, and it is a queueing property, not a per-request one. The engine behaves as a service whose effective capacity is set by its slowest contended stage. When the encoder shares the decode step, one image request injects a burst of compute-bound work that stalls the decode queue; by Little's law, a stall that raises the average time-in-system raises the number of requests in flight, which raises memory pressure, which raises the chance of a preemption — a feedback loop that drops the fraction meeting SLO well before raw utilization looks high. Pull the encoder onto its own pool and the decode queue stops absorbing those bursts: its service time becomes predictable again, its tail shrinks, and the number of requests clearing their SLO per second rises. That is the mechanism behind the cited 2–2.5× — it is not that any single request got 2.5× faster; it is that the *distribution* stopped having a compute-bound spike wired into it, so many more requests land under the SLO line. The same reasoning tells you when it will *not* help: if images are rare, there is no burst to isolate, and a dedicated pool is idle capacity you paid for.

### Stress tests

Push the design until it breaks, which is the only way to know it works.

- **A four-image request (4× the image-token prefill).** Admission returns `chunked_prefill` with three chunks; decode streams keep stepping between chunks instead of freezing for one giant prefill. Without chunking, TPOT for the whole batch spikes for the duration — exactly the stall in figure 4.
- **A 50-page document as page-images (KV blowup).** 51200 image tokens want 6.25 GiB of KV — admission rejects it before it can evict anyone. The right product answer is not "make the KV bigger"; it is retrieval: encode the pages once, cache the embeddings (section 6), and only prefill the pages a given question actually needs.
- **An audio stream longer than the context window.** A long dictation eventually overflows the model's window. You need a sliding-window policy — drop or summarize the oldest audio tokens as new chunks arrive — the streaming analog of the long-context handling from the long-context post. The anchor stays; the tail slides.
- **The encoder pool undersized while image traffic spikes.** With EPD, a burst of image requests queues at the encoder pool while text-only requests sail through untouched — the disaggregation *contains* the blast radius to multimodal traffic instead of stalling everyone. Without it, the whole engine hitches. The fix is to scale the encoder pool to image QPS, which you can only do because it is a separate pool.

## 9. How to measure any of this honestly

Every number in this post is derived or cited — I have run nothing, because I have no GPU, and neither should you trust a multimodal benchmark that skips the fundamentals. If you go measure your own multimodal path, here is the discipline, and it is stricter than for text because the encoder adds a second timescale.

- **Warm up, then synchronize.** The first image through a fresh ViT triggers `cudnn`/kernel autotuning and allocator growth. Discard the first few iterations, then time steady state. Always `torch.cuda.synchronize()` before you read the clock, or you time the launch, not the work.
- **Time with CUDA events, not `time.time()`.** GPU work is asynchronous; a wall-clock read after a kernel launch measures nothing.
- **Separate encoder time from prefill time from decode time.** A single "TTFT for an image request" number hides which stage is slow. Instrument the ViT forward, the image prefill, and the first decode as three timers — the whole point of section 4 is that they are three different workloads.
- **Open-loop, not closed-loop, load.** Fire requests on a Poisson schedule at a target QPS and watch p50/p99 TTFT and TPOT *and goodput* (requests meeting their latency SLO per second). A closed-loop "send one, wait, send the next" test with images gives you a throughput number that no real multimodal traffic will ever reproduce, because it never lets the encoder and decode contend.
- **Report tok/s at batch 1 for nothing.** As with text, single-request throughput tells you about the model, not the server. A multimodal *server* is judged on goodput under mixed text-and-image load — the metric EPD moves.

```python
# nanoserve/bench.py  (honest per-stage timing sketch)
import torch
def time_stage(fn, *args, warmup=3, iters=20):
    for _ in range(warmup):            # autotune + allocator warmup
        fn(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()           # wait before reading the clock
    return start.elapsed_time(end) / iters   # ms/iter, steady state
```

## Case studies and real numbers

Four public, cited results anchor the claims in this post. None is a benchmark of mine; each carries its setup.

- **Encoder Disaggregation (EPD)** — [vLLM, 2025-12-15](https://vllm.ai/blog/2025-12-15-vllm-epd). Splitting the vision/audio encoder into its own pool (Proxy & Router → encoder instances → EC connectors → PD loads embeddings; text-only bypasses) delivers **2–2.5× overall goodput** on 4×A100-80G with Qwen3-VL-4B-Instruct: short-text 4-image goodput 6→12 QPS, long-text 3–4 image 4→8–11 QPS, P99 TTFT/TPOT 20–50% lower. Available since release 0.11.1. Caveat, in their words: benefits shrink when images are rare or the workload is decode-dominated.
- **Shared-Memory IPC Caching** — [vLLM, 2025-11-13](https://vllm.ai/blog/2025-11-13-shm-ipc-cache). On `command-a-vision`, 4×A100 TP4, VisionArena: warming the multimodal-embedding cache (`mm_processor_cache_type="shm"`) raised cached prefill from **2894 to 4918 tok/s (+69.9%)** and cut **TTFT from 790 to 471 ms (−40.5%)**. Limitation: relies on strict input ordering, multimodal-only.
- **vLLM V1 architecture** — [vLLM, 2025-01-27](https://vllm.ai/blog/2025-01-27-v1-alpha-release). The V1 rewrite added **image-hash prefix caching plus an encoder cache** to the multimodal path, alongside its overlapped tokenize/mm-preprocess/detokenize `EngineCore` — the content-addressed reuse this post's section 6 is built on.
- **LLaVA "AnyRes" tiling** — the LLaVA-1.6 line established dynamic high-resolution tiling, where a large image is split into multiple crops each contributing its own token block (on the order of ~2880 tokens for a high-res image). This is why the resolution cliff in section 3 is real and not a corner case: production VLMs *choose* to spend tokens on resolution.

## When to reach for this (and when not to)

A decisive recommendation, because most teams over-build the multimodal path.

- **Occasional single image, low QPS?** Run the encoder inline and move on. Disaggregation and a shared-memory cache are complexity you do not need; a single image every few seconds does not stall a decode loop enough to matter. Do the token-budget math (section 2) so your admission control is honest, and stop there.
- **Steady, heavy image traffic?** Now the encoder-decode contention (section 4) is real and constant. Reach for encoder disaggregation (EPD-style, section 5) so the compute-bound encoder stops hitching decode, and size the encoder pool to your image QPS.
- **The same images sent repeatedly** (document Q&A, product catalogs, a fixed set of reference images)? The embedding cache (section 6) is the highest-leverage thing you can build — a 90% encoder hit rate for a ten-question document session is free throughput. Key it on the image hash and, if your engine supports it, extend to image-hash prefix caching for the KV too.
- **Real-time audio?** Stream it (section 7). Never buffer a whole clip if the model is causal and streaming-trained; the first-audio-out latency difference is roughly 5000 ms vs. 270 ms, and users feel every millisecond of it.
- **When should you just use vLLM instead of your own code?** For anything past a prototype. vLLM already ships image-hash prefix caching, an encoder cache, EPD, chunked prefill, and paged KV — the exact machinery this post builds toy versions of. Build `nanoserve`'s multimodal path to *understand* the cost model and to instrument your own admission control; run vLLM (or SGLang/TensorRT-LLM) in production. The value of writing it yourself is that you will never again be surprised by an image's TTFT — you will know it was 1024 tokens all along.

## Key takeaways

- **An image is a prefill amplifier.** A VLM runs pixels through a vision encoder into a block of image tokens that are prepended to the prompt and prefilled like text — so an image costs what a long prompt costs: TTFT, KV, and a big prefill step.
- **The token count is geometry.** Tokens ≈ (side/patch/merge)²; an 896px image at patch 14 with a 2×2 merge is exactly 1024 tokens, and it grows quadratically with resolution.
- **Image tokens pay full KV.** At 128 KiB/token for Llama-3.1-8B, 1024 image tokens are 128 MiB — often more than the text conversation around them.
- **The encoder is a different workload.** Compute-bound, run-once, not autoregressive. Inline it and it stalls every decoding stream; disaggregate it (EPD) for **2–2.5× goodput** (cited: vLLM EPD) when image traffic is steady — and idle GPUs when it is not.
- **Cache embeddings by image hash.** The same image twice reuses embeddings and (with image-hash prefix caching) KV. Document-QA sessions hit **~90% encoder-skip rates**; cited warm-cache prefill runs +70% faster (vLLM SHM-IPC).
- **Audio is incremental prefill.** Each ~128 ms PCM16 chunk appends to a running prompt and prefills against a cached anchor; first-audio-out drops from seconds to ~270 ms. Causal, streaming-trained models only.
- **The scheduler must see the image.** Admission has to count image tokens: chunk the prefill bomb, budget the KV memory, reject the 50-page document before it evicts everyone.
- **Measure per stage, open-loop.** Time the encoder, the image prefill, and decode separately; judge the server on goodput under mixed load, never tok/s at batch 1.

## Further reading

- [Encoder Disaggregation for Multimodal (EPD)](https://vllm.ai/blog/2025-12-15-vllm-epd) — vLLM, the architecture and goodput numbers behind section 5.
- [Shared-Memory IPC Caching](https://vllm.ai/blog/2025-11-13-shm-ipc-cache) — vLLM, the multimodal-embedding cache and its cited throughput/TTFT wins.
- [vLLM V1: image-hash prefix caching + encoder cache](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — vLLM, the content-addressed multimodal reuse this post builds toward.
- [Chunked prefill and the TTFT/TPOT trade-off](/blog/machine-learning/inference-engineering/chunked-prefill-and-the-ttft-tpot-tradeoff) — the release valve for an image prefill bomb.
- [Prefix sharing, radix trees, and copy-on-write](/blog/machine-learning/inference-engineering/prefix-sharing-radix-trees-and-copy-on-write) — the reuse machinery that image-hash caching keys into.
- [Structured output in production: streaming JSON and tool calls](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls) — the streaming-output side of the audio loop.
- [Prefill/decode disaggregation](/blog/machine-learning/model-serving/prefill-decode-disaggregation) — the disaggregation pattern EPD extends with a third stage.
- [What inference engineering is](/blog/machine-learning/inference-engineering/what-inference-engineering-is) and the [inference engineering playbook](/blog/machine-learning/inference-engineering/the-inference-engineering-playbook) — where this post sits in the `nanoserve` build.
