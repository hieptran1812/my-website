---
title: "Unlimited OCR: One-Shot Long-Horizon Parsing with Reference Sliding Window Attention"
date: "2026-07-01"
publishDate: "2026-07-01"
description: "A deep read of Baidu's Unlimited OCR (arXiv:2606.23050): how Reference Sliding Window Attention keeps the decoder KV cache constant so a 0.5B-active model can transcribe dozens of pages in a single forward pass — and tops OmniDocBench while doing it."
tags: ["ocr", "document-ai", "attention", "kv-cache", "sliding-window-attention", "long-context", "vision-language-model", "deepseek-ocr", "inference-optimization", "paper-reading"]
category: "paper-reading"
subcategory: "Computer Vision"
author: "Hiep Tran"
featured: true
readTime: 30
---

> [!tldr]
> - **What it claims.** Baidu's *Unlimited OCR* replaces **every** decoder attention layer in a DeepSeek-OCR–style model with **Reference Sliding Window Attention (R-SWA)**: each generated token attends to *all* reference (visual + prompt) tokens but only to the last $n$ output tokens ($n=128$). This pins the decode-side KV cache at a constant $L_m + n$ instead of growing as $L_m + T$.
> - **Why it matters.** No prior end-to-end OCR model can parse even ten pages in one forward pass — they all run a page-by-page `for`-loop that resets memory each step. A constant KV cache plus DeepSeek's 16× visual-token compression lets Unlimited OCR transcribe dozens of pages in a single pass under a 32K context.
> - **The surprising part.** R-SWA is a *free lunch*: throwing away most of the output history does not hurt quality. Unlimited OCR scores **93.23** overall on OmniDocBench v1.5 — **+6.22** over its own DeepSeek-OCR baseline — and tops v1.6 at **93.92**, beating 72B–235B generalist VLMs with only **0.5B active** parameters.
> - **Where it fails.** "Unlimited" is aspirational: the *prefix* still grows with page count, so under a finite 32K context the prefill — not R-SWA — becomes the ceiling. Long-horizon errors at 40+ pages come from small-text resolution in "Base" mode, not from the window losing its place.
> - **Read this if** you care about long-context decoding, KV-cache economics, or how to ship document AI that doesn't fall over on a 50-page PDF.

There is a quiet embarrassment at the heart of modern OCR. We have models that read a single page better than any pipeline ever did — and not one of them can read a *book*. Ask a state-of-the-art end-to-end OCR model to transcribe forty pages and it does not transcribe forty pages; it transcribes one page, throws away everything it just learned, and transcribes the next, forty times, stitched together by a `for`-loop living outside the model. A human copyist does not do this. They keep going.

[Unlimited OCR](https://arxiv.org/pdf/2606.23050) (Baidu, June 2026) is a technical report about closing exactly that gap, and its core idea is small enough to state in one sentence: **make the decoder's attention look at all of the image but only a sliding window of what it has already written.** The diagram above is the mental model — on the left, today's reset-every-page loop; on the right, a single continuous pass whose memory footprint never grows. Everything else in the paper is consequence.

![For-loop page-by-page OCR resets decoder memory every page, while Unlimited OCR prefills the whole document and decodes it in one continuous pass.](/imgs/blogs/unlimited-ocr-reference-sliding-window-attention-1.webp)

This post is a close reading. We will rebuild R-SWA from the human-working-memory analogy that motivates it, write the attention mask and the KV-cache accounting out in full, look at the architecture it bolts onto (DeepSeek-OCR's DeepEncoder plus a tiny MoE decoder), walk the experiments, and then — because every paper deserves a skeptic — pull on the threads that do not quite hold. If you have read our notes on the [KV-cache management literature](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) or on [block-sparse attention in MoBA](/blog/paper-reading/large-language-model/moba), this will feel like the same war fought on a new front: the enemy is always the cache that grows without bound.

## Context: why OCR still runs in a for-loop

Document OCR has lived through three eras. First came **pipelines**: a detector finds text regions, a recognizer reads each crop, and a pile of heuristics — cropping, rectification, reading-order rules — glues the output back together. These still ship widely, because OCR decomposes so naturally into detect-then-read that you can mix a mature detector with a modern recognizer and get something serviceable.

The second era folded the recognizer into a large language model: keep the detector, but replace the zoo of per-region recognizers with one LLM decoder. The third — the one the paper lives in — is **end-to-end**: a vision encoder turns the whole page into tokens, and a single decoder emits the entire page's structured content (text, tables as HTML/LaTeX, formulas, reading order) in one autoregressive stream. DeepSeek-OCR is the exemplar, and its appeal is obvious: the decoder is just an LLM, so it inherits the LLM's prior over language, which measurably improves recognition on degraded or ambiguous glyphs.

The downside is just as obvious, and it is the same downside every autoregressive model has: **the KV cache grows linearly with the number of generated tokens.** Each new output token appends a key and value to every layer's cache. Memory climbs, the attention kernel reads an ever-longer cache per step, and generation slows down the longer it runs. For a chat reply of a few hundred tokens this is a rounding error. For OCR over a document it is fatal, because the output is *enormous*.

Here is the arithmetic the paper leans on. DeepSeek's encoder compresses aggressively, but a page still becomes a few hundred visual tokens, and the visual-to-text ratio is about **1:10** — one visual token decodes into roughly ten text tokens. So 10K visual tokens (call it 20–30 pages at 1024×1024) demand an output of **100K+ tokens** to fully transcribe. A 100K-token decode under standard full attention means a 100K-entry KV cache in every layer, and attention cost that climbs the whole way. This is why nobody parses a book in one pass: not a lack of context window in principle, but the runaway cost of decoding into it.

So the field cheats. It runs the model page by page in a loop, resetting the KV cache at every iteration. It works — your 50-page PDF does get transcribed — but it is an engineering workaround, not a model that *understands* long documents. The model never holds a continuous state across pages; an external scheduler does the holding. As the authors put it, the loop "erases memory entirely at each page, fragmenting a coherent long-horizon process into isolated short tasks."

### The human copyist

The paper's framing device is worth pausing on, because the whole method falls out of it. Watch a person copy a document by hand. Their attention rests on three things: the **source** (the book they are reading), a **small slice of what they just wrote** (a few characters, enough to stay oriented), and the **next character**. They do *not* re-scan everything already transcribed. Distant output fades; recent output is kept just long enough to track progress. Call it soft forgetting.

This is a specific claim about the *shape* of attention, and it is neither of the two shapes we usually reach for:

- It is **not full attention.** The full history is never consulted; you don't re-read page 3 to write page 30.
- It is **not linear attention** (or any recurrent-state scheme). Linear attention folds the past into a running state, and crucially that would fold the *visual* tokens into the state too — progressively blurring the very image features the model needs to keep reading accurately. The source must stay pin-sharp.

What the copyist does is keep the source fully and continuously available, while letting a *bounded window* of recent output slide along. That is R-SWA, and the rest of the method is making it precise.

## Contributions

In the authors' framing, tightened:

1. **Reference Sliding Window Attention (R-SWA).** A decoder attention pattern where every generated token attends to all reference tokens (visual tokens plus the prompt) and to only the preceding $n$ output tokens ($n=128$ by default). It keeps the KV cache constant during inference and cuts attention compute over the growing decode sequence.
2. **Unlimited OCR.** Take DeepSeek-OCR as the baseline, keep its high-compression DeepEncoder, and replace *all* of the decoder's multi-head attention with R-SWA. The result parses dozens of pages in one forward pass — and, surprisingly, *improves* general OCR accuracy: **93%** on OmniDocBench v1.5, **+6%** over the DeepSeek-OCR baseline.
3. **A first look at linear-complexity attention for long-horizon multimodal parsing.** Rather than brute-forcing a longer training context, the paper argues R-SWA is the structurally right primitive, and gestures at extending it beyond OCR to ASR and translation — any task with a fixed reference and a long, reference-grounded output.

## Method: Reference Sliding Window Attention

### The access set

R-SWA constrains every decode step to a two-part set of keys it may attend to. Let $L_m$ be the length of the **prefix** — all visual tokens plus the prompt — and let the decode region be indexed after it. For the token generated at decode step $t$, define the accessible set

$$\mathcal{N}(t) = \mathcal{P} \cup \mathcal{D}_n(t), \qquad \mathcal{P} = \{1, \dots, L_m\},$$

$$\mathcal{D}_n(t) = \big\{\, j \;\big|\; \max(L_m + 1,\; L_m + t - n) \le j \le L_m + t - 1 \,\big\}.$$

Read it slowly, because it is the whole paper. $\mathcal{P}$ is the prefix segment of length $L_m$, **globally visible to every subsequent token** — the source the copyist never stops looking at. $\mathcal{D}_n(t)$ is a **causal sliding window of width $n$** over the decode region: the last $n$ output tokens before $t$, and no further back. The `max(...)` clamps the window so it never reaches into the prefix or before the start of generation.

Attention is then ordinary softmax, but only over $\mathcal{N}(t)$:

$$\alpha_{tj} = \frac{\exp\!\big(\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_k}\big)}{\sum_{i \in \mathcal{N}(t)} \exp\!\big(\mathbf{q}_t^\top \mathbf{k}_i / \sqrt{d_k}\big)}, \qquad j \in \mathcal{N}(t),$$

$$\mathbf{o}_t = \sum_{j \in \mathcal{N}(t)} \alpha_{tj}\, \mathbf{v}_j,$$

where $\mathbf{q}_t$, $\mathbf{k}_j$, $\mathbf{v}_j$ are the usual query, key, and value vectors and $d_k$ is the key dimension. The figure below contrasts the access pattern with vanilla full attention on a tiny example: both rows keep every reference cell lit, but R-SWA marks the older outputs *evicted* and attends only inside its fixed window.

![A generated token under full attention sees all reference plus all prior outputs; under R-SWA it sees all reference plus only the last n outputs, evicting the rest.](/imgs/blogs/unlimited-ocr-reference-sliding-window-attention-3.webp)

Two design choices distinguish R-SWA from the obvious alternatives, and both matter.

**Why not vanilla sliding-window attention (SWA)?** Plain SWA — as in Mistral or [MoBA](/blog/paper-reading/large-language-model/moba)-adjacent designs — slides one window over the *entire* sequence, visual tokens included. Over a long decode, the visual tokens fall out of the window and the model loses the page. Even schemes that keep visual tokens "in the loop" via state updates blur them, because every update is a lossy mix. R-SWA's fix is to make the prefix **never participate in the sliding** — visual tokens are encoded once and remain static, fully visible, for the entire decode. The window only ever slides over *output*.

**Why keep the prefix global at all?** Because the prefix *is* the document. In a chat model the prompt is a few hundred tokens; here the "prompt" is the compressed image, and it is the only thing keeping recognition grounded. Drop it from the window and you are transcribing from memory.

It helps to place R-SWA on the map of attention variants explicitly. The axis that matters is *what each scheme does with two very different things — the static reference and the growing output* — and R-SWA is the only one that treats them differently:

| Scheme | Sees the reference | Sees output history | Decode KV cache | Failure mode for OCR |
|---|---|---|---|---|
| Full attention (MHA) | all, every step | all, every step | $L_m + T$ (grows) | OOM and per-step slowdown on long documents |
| Vanilla SWA | only while in window | only in window | $\approx n$ (but loses reference) | visual tokens fall out of the window — model loses the page |
| Linear attention | folded into a recurrent state | folded into a recurrent state | $O(1)$ state | lossy state updates blur the static visual features |
| **R-SWA** | **all, pinned, every step** | **last $n$ only** | $L_m + n$ (constant) | prefix $L_m$ still grows with *page count* (input side) |

The table makes the design philosophy legible. Full attention is correct but unaffordable. Vanilla SWA is affordable but forgets the source. Linear attention is affordable but corrupts the source. R-SWA is the only row that keeps the source perfect *and* the decode bounded — by refusing to treat them with the same mechanism. That asymmetry is the whole contribution; everything else is plumbing.

### The KV cache as a fixed-capacity queue

The payoff lives in the cache accounting. Under standard multi-head attention (MHA), after generating $T$ tokens the cache holds

$$C_{\text{MHA}}(T) = L_m + T,$$

which grows without bound. Under R-SWA, the prefix is always retained but the decode side keeps only the most recent $n$ tokens:

$$C_{\text{R-SWA}}(T) = L_m + \min(n, T) \;\le\; L_m + n.$$

That is the entire trick: a constant. Implemented concretely, the KV cache is a **queue of capacity $L_m + n$**. Each time a new token is generated, the key/value of the oldest *output* token — position $L_m + t - n$ — is evicted, while the prefix entries $1..L_m$ are pinned and never leave. The animation makes the eviction step explicit:

<figure class="blog-anim">
<svg viewBox="0 0 760 280" role="img" aria-label="A new output token enters the KV queue, the oldest output token is evicted, and the window slides while the prefix stays pinned" style="width:100%;height:auto;max-width:840px">
<style>
.uo2-pin{fill:#3b82f6;opacity:.85}
.uo2-win{fill:#10b981;opacity:.9}
.uo2-tx{font:600 15px ui-sans-serif,system-ui;fill:#ffffff;text-anchor:middle}
.uo2-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.uo2-sub{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.uo2-evict{animation:uo2-evict 8s ease-in-out infinite}
.uo2-slide{animation:uo2-slide 8s ease-in-out infinite}
.uo2-enter{animation:uo2-enter 8s ease-in-out infinite}
@keyframes uo2-evict{0%,12%{opacity:1}32%,100%{opacity:.08}}
@keyframes uo2-slide{0%,35%{transform:translateX(0)}62%,100%{transform:translateX(-110px)}}
@keyframes uo2-enter{0%,40%{opacity:0;transform:translateX(110px)}66%,100%{opacity:1;transform:translateX(0)}}
@media (prefers-reduced-motion:reduce){.uo2-evict{animation:none;opacity:.4}.uo2-slide{animation:none}.uo2-enter{animation:none;opacity:1;transform:translateX(0)}}
</style>
<text class="uo2-cap" x="110" y="44">prefix L_m (pinned)</text>
<text class="uo2-cap" x="445" y="44">window n (last outputs)</text>
<rect class="uo2-pin" x="20" y="64" width="180" height="84" rx="8"/>
<text class="uo2-tx" x="110" y="112">reference</text>
<g class="uo2-evict">
<rect class="uo2-win" x="230" y="64" width="100" height="84" rx="8"/>
<text class="uo2-tx" x="280" y="112">o(k-2)</text>
</g>
<g class="uo2-slide">
<rect class="uo2-win" x="340" y="64" width="100" height="84" rx="8"/>
<text class="uo2-tx" x="390" y="112">o(k-1)</text>
</g>
<g class="uo2-slide">
<rect class="uo2-win" x="450" y="64" width="100" height="84" rx="8"/>
<text class="uo2-tx" x="500" y="112">o(k)</text>
</g>
<g class="uo2-enter">
<rect class="uo2-win" x="560" y="64" width="100" height="84" rx="8"/>
<text class="uo2-tx" x="610" y="112">o(k+1)</text>
</g>
<text class="uo2-sub uo2-evict" x="280" y="184">evicted</text>
<text class="uo2-sub uo2-enter" x="610" y="184">new token</text>
<text class="uo2-sub" x="380" y="232">each step: one token enters, the oldest output leaves, cache size stays L_m + n</text>
</svg>
<figcaption>One decode step of R-SWA: a new output token enters the queue, the oldest output is evicted, and the window slides — the pinned prefix never moves and the cache size never grows.</figcaption>
</figure>

To quantify the saving, define the cache ratio $\rho(T) = C_{\text{R-SWA}}(T) / C_{\text{MHA}}(T)$. Once the generation is long enough that $T \gg n$,

$$\rho(T) = \frac{L_m + \min(n, T)}{L_m + T} \;\longrightarrow\; \frac{L_m + n}{L_m + T},$$

and when the decode length dominates both prefix and window, $\rho(T) \approx (L_m + n)/T \to 0$. The longer the document, the larger the saving — which is precisely the regime OCR cares about. The animation below shows the two curves running the *same* decode: the full-attention cache fills the track without bound while R-SWA stays flat.

<figure class="blog-anim">
<svg viewBox="0 0 720 340" role="img" aria-label="As decoding proceeds the full-attention KV cache grows without bound while R-SWA stays at a fixed size" style="width:100%;height:auto;max-width:820px">
<style>
.uo1-lbl{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.uo1-sub{font:400 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.uo1-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.uo1-mha{fill:#ef4444;opacity:.82}
.uo1-swa{fill:#10b981;opacity:.9}
.uo1-grow{transform-box:fill-box;transform-origin:left center;animation:uo1-grow 9s ease-in-out infinite}
.uo1-flat{transform-box:fill-box;transform-origin:left center;animation:uo1-flat 9s ease-in-out infinite}
@keyframes uo1-grow{0%{transform:scaleX(.05)}82%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes uo1-flat{0%{transform:scaleX(.17)}50%{transform:scaleX(.2)}100%{transform:scaleX(.17)}}
@media (prefers-reduced-motion:reduce){.uo1-grow{animation:none;transform:scaleX(1)}.uo1-flat{animation:none;transform:scaleX(.18)}}
</style>
<text class="uo1-lbl" x="20" y="38">Full attention (vanilla MHA)</text>
<text class="uo1-sub" x="20" y="60">KV cache = L_m + T</text>
<rect class="uo1-track" x="20" y="76" width="680" height="54" rx="8"/>
<rect class="uo1-mha uo1-grow" x="20" y="76" width="680" height="54" rx="8"/>
<text class="uo1-lbl" x="20" y="196">R-SWA (this paper)</text>
<text class="uo1-sub" x="20" y="218">KV cache = L_m + n  (constant)</text>
<rect class="uo1-track" x="20" y="234" width="680" height="54" rx="8"/>
<rect class="uo1-swa uo1-flat" x="20" y="234" width="680" height="54" rx="8"/>
<text class="uo1-sub" x="20" y="324">decode step T increases  --&gt;</text>
</svg>
<figcaption>The same decoding run: the full-attention cache fills without bound as tokens are generated, while R-SWA holds a fixed window.</figcaption>
</figure>

The same bound applies to the attention *compute* per step (you score against $L_m + n$ keys instead of $L_m + T$) and, the paper notes, to GPU memory during inference: linear in $T$ for the baseline, fixed for R-SWA. It is this *joint* stability — compute and memory both flat — that makes one-shot long-horizon parsing feasible at all.

To feel the magnitude, work a concrete 20-page case. At 256 tokens per page, the reference prefix is $L_m \approx 20 \times 256 = 5{,}120$ tokens (plus a short prompt). At the paper's 1:10 visual-to-text ratio, those visual tokens decode into roughly $T \approx 51{,}200$ output tokens. Tally the final per-layer KV cache:

- **Full attention:** $L_m + T = 5{,}120 + 51{,}200 = 56{,}320$ entries.
- **R-SWA:** $L_m + n = 5{,}120 + 128 = 5{,}248$ entries.

So $\rho \approx 5{,}248 / 56{,}320 \approx 0.093$ — R-SWA holds about **9%** of the cache at the end of the document, and the result improves as the document grows. Look only at the *decode side* and the contrast is brutal: 128 retained output tokens against 51,200, or **0.25%**. The reference, not the window, is now the dominant cost — which is exactly why DeepEncoder's 16× compression of that reference is load-bearing, and exactly why the paper's remaining bottleneck is the prefix, not the decode.

### A reference implementation

R-SWA is, mechanically, a mask. The cleanest way to internalize it is to build that mask. Here is the access set of equations (1)–(2) as an additive attention mask you could drop into a PyTorch attention module — every query row exposes the full prefix plus its own causal window of width $n$:

```python
import torch

def rswa_mask(L_m: int, T: int, n: int, device="cuda", dtype=torch.float32):
    """Additive attention mask for Reference Sliding Window Attention.

    Rows index the query positions (prefix then decode); columns index keys.
    A position is visible -> 0.0; masked -> -inf. Mirrors N(t) = P ∪ D_n(t).
    """
    S = L_m + T
    neg = torch.finfo(dtype).min
    mask = torch.full((S, S), neg, device=device, dtype=dtype)

    # 1) Prefix rows: standard causal attention *within* the prefix.
    pre = torch.arange(L_m, device=device)
    mask[:L_m].index_fill_(1, pre, 0.0)
    mask[:L_m] = torch.where(
        torch.arange(S, device=device)[None, :] <= pre[:, None], 0.0, neg
    )[:, :S]

    # 2) Decode rows: every prefix key is always visible (the "reference").
    mask[L_m:, :L_m] = 0.0

    # 3) Decode rows: a causal window of width n over the decode region only.
    q = torch.arange(L_m, S, device=device)          # absolute query positions
    k = torch.arange(L_m, S, device=device)          # absolute decode-key positions
    rel = q[:, None] - k[None, :]                     # query - key distance
    window = (rel >= 0) & (rel < n)                   # last n outputs, causal
    decode_block = torch.where(window, 0.0, neg)
    mask[L_m:, L_m:] = decode_block
    return mask
```

The third block is the only interesting part: `rel >= 0` enforces causality (no peeking ahead) and `rel < n` enforces the window (no peeking back more than $n$). At inference you would not materialize an $S \times S$ mask — you would use a paged/streaming attention kernel — but the mask form makes the semantics unambiguous.

The streaming side is the queue. Conceptually:

```python
class RSWACache:
    """Per-layer KV cache: pinned prefix + fixed-width output ring."""
    def __init__(self, prefix_k, prefix_v, n: int):
        self.pk, self.pv = prefix_k, prefix_v        # [L_m, d], encoded once, frozen
        self.n = n
        self.ok, self.ov = [], []                    # rolling output window

    def step(self, k_t, v_t):
        self.ok.append(k_t); self.ov.append(v_t)
        if len(self.ok) > self.n:                     # evict the oldest output
            self.ok.pop(0); self.ov.pop(0)
        keys = torch.cat([self.pk, torch.stack(self.ok)], dim=0)
        vals = torch.cat([self.pv, torch.stack(self.ov)], dim=0)
        return keys, vals                            # size <= L_m + n, always
```

`self.pk/self.pv` are written exactly once, when the page(s) are encoded, and never touched again — that is "encoded once, never updated during decoding." The output ring is the only thing that moves, and it is capped at $n$. The returned `keys`/`vals` never exceed $L_m + n$ rows, which is the whole point.

### What the kernel actually does

The paper's kernel study (its Figure 3) is the empirical sanity check on all of this. It plots the per-call duration of the Flash Attention v3 kernel as decoding proceeds. For the DeepSeek-OCR baseline the per-call time **climbs** step over step — the kernel reads a longer KV cache each time — with a sharp spike when the cache length crosses an alignment boundary and data-transfer efficiency drops. For Unlimited OCR the per-call duration is **flat**, a direct consequence of R-SWA in every layer. We reconstruct the same story at the throughput level in the efficiency section below; the takeaway is that the constant-cache property is not just an asymptotic argument on paper, it shows up in the kernel timings.

## Method: the model R-SWA bolts onto

R-SWA is an attention pattern, not a model. Unlimited OCR is what you get when you graft it onto DeepSeek-OCR. The architecture has two parts — a heavily-compressing encoder that fills the *reference prefix* once, and a small Mixture-of-Experts decoder whose every attention layer is R-SWA and which writes only the *bounded window* to cache.

![Unlimited OCR: the encoder fills a pinned reference prefix once, while the MoE decoder reads prefix plus window but writes only the bounded window to the KV cache.](/imgs/blogs/unlimited-ocr-reference-sliding-window-attention-4.webp)

**The decoder** is a Mixture-of-Experts LLM with **3B total** parameters but only **~0.5B active** (the paper writes it 3B-A0.5B; activations stay around 500M during inference). Small active parameter count means cheap per-token compute; the MoE gives capacity without paying for it on every token. Departing from the DeepSeek-OCR baseline, *all* of the decoder's vanilla MHA layers are swapped for R-SWA. Long-horizon parsing then comes "for free" structurally: you augment the original reference KV cache of width $L_m$ with a fixed-capacity output buffer of width $n$, and decode.

**The encoder** is the part that makes the reference cheap enough to pin. DeepEncoder, inherited unchanged from DeepSeek-OCR, is a cascade: a window-attention vision transformer (SAM-ViT) processes the raw image patches, a **16× compression** bridge shrinks the token count, and only then does a global-attention transformer (CLIP-ViT) run — on the already-compressed tokens. The ordering is the trick: the expensive global attention never sees the full-resolution token grid, so activations stay low even on high-resolution pages.

![DeepEncoder cascades window-attention SAM-ViT and global-attention CLIP-ViT with a 16x bridge, compressing a 1024x1024 page to 256 visual tokens.](/imgs/blogs/unlimited-ocr-reference-sliding-window-attention-5.webp)

The numbers are what make R-SWA viable. DeepEncoder compresses a **1024×1024** page into just **256 tokens**. The paper keeps two of DeepSeek-OCR's five resolution modes: **"Base"** (1024×1024, used for multi-page) and **"Gundam"** (dynamic resolution, single-page). That compression ratio is load-bearing for long-horizon work in a way that is easy to miss: because visual tokens "do not undergo state transitions alongside the output — they are encoded once and remain static throughout the entire long-horizon parsing process," every page you keep pinned costs only 256 entries. Twenty pages is ~5K reference tokens, not 5K × (some growth factor). The reference is big but *bounded and static*; only the window moves.

Put the two halves together and the division of labor is clean. The encoder decides how much the prefix costs (256/page, fixed). R-SWA decides how much the decode costs ($L_m + n$, fixed). Neither grows with output length. That is the precondition for parsing a book in one pass.

## Experiments

### Setup, honestly

**Data.** About **2M** document OCR samples, **9:1** single-page to multi-page. Single-page PDFs are auto-annotated with Paddle OCR, concatenating each block's coordinates (normalized to 0–1000) and content into end-to-end detection-and-parsing ground truth. Multi-page data is *synthesized* by concatenating single-page samples — ~200K samples of 2–50 pages each, with a `<page>` separator between pages, all packed into 32K-token sequences. Hold that "synthesized" in mind for the critique.

**Training.** Continue-training from the DeepSeek-OCR checkpoint for only **4,000 steps**, global batch size 256, max sequence length 32K, on 8×16 A800 GPUs. The DeepEncoder is **frozen** — only the LLM is trained, on the argument that the encoder is already well-optimized. AdamW, cosine annealing, initial LR 1e-4. Expert parallelism (EP=4) via DeepEP; the pipeline is Megatron-LM. Inference is supported in both Transformers and SGLang, both running under constant tokens/s and GPU memory.

**Benchmarks.** OmniDocBench v1.5 and v1.6 for foundational document parsing, scored on five task-specific metrics: text-recognition **Edit Distance** (↓), formula **CDM** (↑), table **TEDS** and **TEDS-S** (↑, with and without content), and **reading-order Edit Distance** (↓), combined into a weighted overall. For long-horizon, an in-house set of novels/documents/papers bucketed by page count (2, 5, 10, 20, 40+, ≥10 books each), scored on **Distinct-n** (n-gram diversity, higher is better) and Edit Distance.

### The headline result

On OmniDocBench v1.5, Unlimited OCR posts an overall **93.23**, against its own DeepSeek-OCR baseline at **87.01** — a **+6.22** jump from nothing but continue-training on document data with R-SWA swapped in. The per-metric story: text Edit Distance drops from 0.073 to **0.038** (−0.035), table TEDS rises **+5.96**, TEDS-S **+5.27**, reading-order Edit improves by 0.041. On v1.6 it reaches **93.92**, the top of the table.

![On OmniDocBench v1.5, the 0.5B-active Unlimited OCR scores highest overall, ahead of 72B-235B generalist VLMs and its own DeepSeek-OCR baseline.](/imgs/blogs/unlimited-ocr-reference-sliding-window-attention-7.webp)

The comparison that should make you sit up is the parameter count. Unlimited OCR is **0.5B active**. It beats **Qwen3-VL at 235B** (89.15), **Qwen2.5-VL at 72B** (87.02), and **Gemini-2.5 Pro** (88.03) on the overall metric. Specialist document models cluster tightly just below it — DeepSeek-OCR 2 (89.17), dots.ocr (88.41), OCRVerse (88.56) — but Unlimited OCR sits clear at the top. The lesson is not "small models are better"; it is that a *task-shaped* attention prior plus a strong compressing encoder beats raw scale on this particular task.

| Model | Size | v1.5 Overall ↑ | Text Edit ↓ | Table TEDS ↑ | Read-order Edit ↓ |
|---|---|---|---|---|---|
| Qwen2.5-VL | 72B | 87.02 | 0.094 | 88.27 | 0.102 |
| DeepSeek-OCR (baseline) | 3B-A0.5B | 87.01 | 0.073 | 83.37 | 0.086 |
| Gemini-2.5 Pro | — | 88.03 | 0.075 | 85.82 | 0.097 |
| dots.ocr | 3B | 88.41 | 0.048 | 83.22 | 0.053 |
| Qwen3-VL | 235B | 89.15 | 0.069 | 88.14 | 0.068 |
| DeepSeek-OCR 2 | 3B-A0.5B | 89.17 | 0.049 | 86.85 | 0.060 |
| **Unlimited OCR** | **3B-A0.5B** | **93.23** | **0.038** | **92.61** | **0.045** |

OmniDocBench v1.6 is the newer, harder split (296 more test images than v1.5) and pits Unlimited OCR against the current crop of specialist parsers rather than generalist VLMs — and here the margins are thin, which is the honest read. Unlimited OCR's **93.92** edges Qianfan-OCR (93.90), Logics-Parsing-v2 (93.33), and FireRed-OCR (93.26); DeepSeek-OCR 2 (90.25), dots.ocr (90.77), and HunyuanOCR (89.95) trail further back. On v1.6 the win is real but measured in hundredths — among purpose-built document models, Unlimited OCR is best-in-class but not in a different league. The story it tells is consistent across both splits: R-SWA is not a handicap on single-page parsing, and on the long-horizon axis nobody else is even competing.

A subcategory study (the paper's Table 2) breaks v1.5 into nine document types — PPT, academic paper, book, colorful textbook, exam paper, magazine, newspaper, note, research report. Against DeepSeek-OCR, Unlimited OCR improves on **every** metric in **every** category — a genuine "free lunch," gains without a compensating loss somewhere. Against the stronger DeepSeek-OCR 2, it wins seven of nine on both text Edit Distance and reading order, and never embarrasses itself on the layout-heavy types (PPT, newspaper, magazine, note). The read here: replacing *all* standard attention with R-SWA is not a reluctant trade for efficiency — on parsing tasks it is at worst neutral and usually better, because feeding history causally through a bounded window helps the model "locate its OCR progress even though it sees only a few tokens."

### Throughput, and why constant matters

Efficiency is where R-SWA was supposed to pay off, and it does. On OmniDocBench, Unlimited OCR runs at **5580 tokens/s** (512 concurrency) versus DeepSeek-OCR's **4951 TPS** in "Base" DeepEncoder mode — a **+12.7%** speedup even on OmniDocBench's relatively short outputs. The advantage widens with output length, which is the entire thesis.

![As output length grows to 6K tokens, DeepSeek-OCR throughput decays about 35% while Unlimited OCR stays flat.](/imgs/blogs/unlimited-ocr-reference-sliding-window-attention-8.webp)

The paper's theoretical ceiling (its Table 4, prefill fixed at 10) makes the divergence stark. At 256 output tokens the two models are identical (~7229 TPS). By 6144 tokens, DeepSeek-OCR has decayed to **5822.87 TPS** while Unlimited OCR holds at **7847.71** — a **~35%** gap, and growing.

| Output length | 256 | 512 | 1024 | 2048 | 4096 | 6144 |
|---|---|---|---|---|---|---|
| DeepSeek-OCR (TPS) | 7229 | 7468 | 7423 | 7167 | 6430 | 5823 |
| Unlimited OCR (TPS) | 7230 | 7715 | 7841 | 7881 | 7905 | 7848 |

Note the shape: DeepSeek-OCR's throughput actually *rises* slightly to ~512 tokens (kernel warm-up, batching) and then decays monotonically as the cache grows; Unlimited OCR rises and stays risen. Constant TPS is not a vanity metric for long-horizon OCR — it is the difference between a deterministic, schedulable workload and one whose per-page latency creeps up unpredictably over a fifty-page document.

That operational point deserves emphasis, because it is the part that matters to anyone running OCR at scale rather than benchmarking it. A serving system batches requests, and batching works best when every request in the batch advances at the same rate. With a growing cache, the requests that are deep into a long document slow down, stragglers form, the batch stalls on its slowest member, and effective utilization collapses — exactly the long-tail latency problem that makes document pipelines so annoying to capacity-plan. A constant cache makes per-token cost independent of how far along any request is, so a 2-page job and a page-40-of-50 job cost the same per step. You can size the batch and the GPU memory once and trust them. The 12.7% headline speedup on OmniDocBench understates this benefit, because OmniDocBench's outputs are short; the real win is that the *variance* of per-step cost goes to zero, and variance is what kills throughput in production.

### Long-horizon, the actual point

The capability that justifies the title is in the paper's Table 3: parse 2, 5, 10, 15, 20, and 40+ pages in a single pass and measure quality.

| Pages | 2 | 5 | 10 | 15 | 20 | 40+ |
|---|---|---|---|---|---|---|
| Distinct-20 ↑ | 99.76% | 99.78% | 97.49% | 99.92% | 98.73% | 96.08% |
| Distinct-35 ↑ | 99.87% | 99.98% | 99.83% | 99.99% | 99.89% | 96.90% |
| Edit Distance ↓ | 0.0362 | 0.0452 | 0.0526 | 0.0787 | 0.0572 | 0.1069 |

Even at **20 pages in one shot**, edit distance stays around 0.057 and diversity above 98%. At **40+ pages** edit distance is 0.107 with 96.9% Distinct-35 — degraded, but coherent, and it is genuinely doing forty pages in a single forward pass with a fixed KV cache and constant per-token latency throughout. The authors examined the repeated-error cases and found they cluster on **small text that is hard to resolve at 1024×1024 "Base" resolution** — an *encoder* limitation, not R-SWA "losing direction" in the long-horizon process. That is an important attribution: the failure mode is "I can't read the tiny font," not "I forgot where I was."

The Distinct-n metric is worth dwelling on, because it is doing more work than it looks. Distinct-n measures the ratio of unique n-grams to total n-grams in the output, and it is the natural early-warning signal for the classic long-generation pathology: **degenerate repetition**, where an autoregressive decoder falls into a loop and copies the same phrase over and over. A model that "loses its place" on page 30 does not produce garbage — it produces *repetition*, and Distinct-n would crater. That it stays at 96.9% even at 40+ pages is the strongest evidence in the paper that the window is not losing track of where it is in the document. This also suggests a mechanistic hypothesis for the "free lunch" the critique flags below: by *forcing* each token to ground itself in the image plus only a short output window, R-SWA may actually *reduce* error propagation. Under full attention, the model can attend to — and copy from — its own earlier mistakes anywhere in the 50K-token history, and OCR errors compound; under R-SWA, an error scrolls out of the window after $n$ tokens and stops contaminating future predictions. Bounded memory as a regularizer against your own past mistakes is a genuinely appealing story, and it would explain why throwing away history *helps* rather than merely costing nothing.

## Critique: what holds, what wobbles

Now the skeptic's pass. The paper is a technical report, not a peer-reviewed study, and it reads like one — strong engineering, light on adversarial self-examination. Several things deserve a harder look.

**"Unlimited" is the wrong word, and the paper half-admits it.** R-SWA bounds the *decode-side* cache, but the *prefix* is still $L_m$, and $L_m$ grows with page count (256 tokens × pages). Under a finite 32K context, prefill — not decoding — becomes the wall. So this is not unlimited parsing; it is parsing whose *output* length is unbounded while its *input* length is still capped. The honest claim is "constant-cache decoding," which is a great result on its own and didn't need the oversell. The authors' own Limitation section concedes the prefix ceiling and proposes a future "prefill pool" that fetches KV chunks on demand — i.e., the actual unlimited version doesn't exist yet.

**The "free lunch" is real but under-explained.** The most interesting empirical claim is that discarding most of the output history *improves* accuracy. The paper's hand-wave — full attention "could lead to divergence as the output length increases," and the bounded window helps the model "locate its OCR progress" — is plausible but unproven. Is the gain from R-SWA's inductive bias, or simply from continue-training on 2M document samples that the baseline never saw? There is no ablation isolating the two. A clean experiment — same 4,000 steps of continue-training *with* full attention vs *with* R-SWA — is the obvious missing control, and its absence means we cannot cleanly credit R-SWA for the +6.22.

**What would change my mind:** that one ablation. If full-attention continue-training on the same data recovers most of the +6.22, then R-SWA is an *efficiency* win that is accuracy-neutral (still excellent, but a different story). If R-SWA-trained clearly beats full-attention-trained at equal data and steps, the "task-shaped attention prior" claim is earned. As written, the paper conflates the two interventions.

**Synthetic multi-page data may flatter the long-horizon numbers.** Multi-page training (and likely much of evaluation framing) comes from *concatenating single-page samples* with a `<page>` separator. Real books have running headers, cross-page tables, footnote continuations, and figures that span spreads — none of which appear when you staple independent pages together. The strong 20-page numbers may partly reflect a test distribution that is kinder than real documents. The in-house long-horizon set uses real novels/papers, which helps, but the *training* signal for "what page 30 looks like given pages 1–29" is synthetic.

**$n=128$ is asserted, not justified.** The window width is the one hyperparameter that defines R-SWA, and there is no sweep. Why 128 and not 64 or 512? For OCR, where the next token depends mostly on the image and very locally on recent output, a small window is intuitively fine — but "intuitively fine" is what ablations are for. A window-size study would also tell us how much headroom there is for even cheaper decoding.

**Generality is gestured at, not shown.** The claim that R-SWA extends to ASR and translation is reasonable — both have a fixed reference (audio, source text) and a long, reference-grounded output — but the paper shows zero non-OCR results. It is a promissory note. The structural argument is sound; the evidence is OCR-only.

None of this sinks the paper. The constant-cache decoding result is real, the engineering is clean, and the SOTA numbers are SOTA numbers. But "R-SWA causes the accuracy gain" and "this is unlimited parsing" are both stronger than the evidence supports.

## What I'd build with this

A few directions this opens, in rough order of how soon I'd try them.

1. **The missing ablation, as a weekend project.** Continue-train the DeepSeek-OCR baseline on the *same* document data with full attention for the same 4,000 steps, and compare to the R-SWA variant. This single experiment settles the central scientific question of the paper. If you have the baseline checkpoint, it is cheap.

2. **R-SWA for ASR, for real.** Audio is the cleanest transfer target: the mel-spectrogram (or encoder output) is the pinned reference, the transcript is the long output, and recent transcript context is all you need to stay oriented. Swap R-SWA into a Whisper-style or [Qwen3-ASR-style](/blog/paper-reading/speech-processing/qwen3-asr-technical-report) decoder and you get hour-long transcription with constant cache. This is the experiment I'd most want to see.

3. **The prefill pool the paper promises.** The real path to "unlimited" is making the *prefix* streamable too: keep a pool of page-KV chunks and let the model learn to fetch the ones it needs (the page it's currently transcribing, plus neighbors), evicting the rest — literally "flipping through pages." This is retrieval over the KV cache, and it rhymes with offload/recompute ideas in the [KV-cache management literature](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management). Hard, but it's the version that earns the title.

4. **A learned or adaptive window.** Instead of a fixed $n=128$, let the window widen near structure-heavy regions (tables, multi-line formulas) where longer output dependencies matter, and shrink over plain prose. Even a simple schedule keyed to the predicted token type could buy quality at the hard parts for nearly free.

5. **Compose with a sharper encoder for the small-text failure.** Since the long-horizon errors are resolution-bound, pairing R-SWA decoding with "Gundam" dynamic resolution (or a tiling scheme) on the pages that need it — rather than uniform "Base" 1024² — should attack the actual bottleneck without touching the attention. R-SWA and the encoder are orthogonal knobs; the paper turns only one.

The deeper bet here is that **long-horizon generation is an attention-pattern problem, not a context-length problem.** Brute-forcing a 128K or 1M context to fit a book is the expensive way; shaping attention so the cost never grows in the first place is the cheap one. Unlimited OCR is one clean instance of that bet, on a task — OCR — where the structure (fixed reference, long grounded output) makes the right pattern obvious. The interesting question is how many other "long-horizon copying" tasks have the same shape.

## References

- **Paper:** Baidu Inc., *Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing.* arXiv:2606.23050v1, 22 June 2026. [PDF](https://arxiv.org/pdf/2606.23050)
- **Code & weights:** [github.com/baidu/Unlimited-OCR](http://github.com/baidu/Unlimited-OCR)
- **Benchmark:** OmniDocBench (v1.5 / v1.6) — the OCR document-parsing benchmark used throughout.
- Related on this blog:
  - [HunyuanOCR technical report](/blog/paper-reading/computer-vision/hunyuanocr-technical-report) — a sibling end-to-end OCR VLM, useful as a point of comparison.
  - [GLM-OCR training recipe](/blog/paper-reading/computer-vision/glm-ocr-training-recipe) — how another end-to-end OCR model is trained.
  - [A survey on KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management) — the cache-pressure problem R-SWA attacks, in full generality.
  - [MoBA: mixture of block attention](/blog/paper-reading/large-language-model/moba) — a sibling sparse-attention idea for long context.
