---
title: "DeepSeek-OCR: Optical Context Compression, or When an Image Is Cheaper Than Its Text"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "DeepSeek-OCR reframes OCR as a context-compression medium: render text to an image and a vision encoder carries it in about a tenth of the tokens. A deep-dive on the DeepEncoder architecture, the optical-memory thesis, and DeepSeek-OCR-2's learned reading order."
tags: ["deepseek-ocr", "optical-compression", "vision-encoder", "long-context", "ocr", "mixture-of-experts", "document-ai", "multimodal", "kv-cache", "computer-vision"]
category: "machine-learning"
subcategory: "Computer Vision"
author: "Hiep Tran"
featured: true
readTime: 51
---

For about a decade, "OCR" has been a solved-enough corner of computer vision that most engineers stopped thinking about it. You point a model at a scan, it gives you back text, you move on. DeepSeek-OCR (arXiv 2510.18234, October 2025) is interesting precisely because it refuses to stay in that corner. Its actual thesis is not "here is a better document parser." Its thesis is that **text rendered to an image is a context-compression medium** — and that a vision encoder can carry a page of writing in roughly a tenth of the tokens the same writing would cost as text. The paper's own framing word for this is *Contexts Optical Compression*.

That reframing matters because it points at one of the most expensive problems in large language models: context length. Every text token you feed a transformer costs attention compute and key-value cache memory, and those costs do not go away when the token is boring. The fortieth page of a contract, the transcript from three hours ago, the documentation you pasted "just in case" — all of it sits in the window at full token price. DeepSeek-OCR asks a heretical question: what if you took the stale parts of that window, *rendered them back into a picture*, and let a vision encoder re-compress them into a handful of tokens? You would be using vision not as a perception system but as a memory system. That is the idea this post is really about.

![Same passage rendered as text tokens versus optical tokens: the optical path carries a 1000-word page in roughly 100 vision tokens against about 1300 text tokens.](/imgs/blogs/deepseek-ocr-optical-context-compression-1.webp)

The diagram above is the mental model for the entire article. On the left, the ordinary text path: a thousand-word page goes through a byte-pair-encoding tokenizer and lands in the context window as something like 1,300 text tokens. On the right, the optical path: the same page is rendered to a single image, pushed through DeepSeek's vision encoder, and arrives as roughly 100 vision tokens carrying the same information. Same content, an order of magnitude fewer tokens. The rest of this post is a tour of how that compression is actually built, how far you can push it before it breaks, why it doubles as a model of memory and forgetting, and where the sequel — DeepSeek-OCR-2 — takes the idea next.

## Why optical compression is different from "OCR"

The first thing to get straight is the direction of the argument, because it inverts the usual one. Conventional OCR treats the image as the problem and text as the answer: the image is messy, lossy, hard, and your job is to recover the clean symbolic text hiding inside it. Optical compression treats text as the problem and the image as the answer: the text is *expensive* — too many tokens — and your job is to find a denser carrier for it. The image is that carrier.

Here is the table I keep in my head when explaining this to people who think DeepSeek-OCR is "just an OCR model."

| Question | The naive OCR view | The optical-compression reality |
|---|---|---|
| What is the image for? | A noisy source to extract text from | A dense medium that stores text in fewer tokens |
| What is being optimized? | Character accuracy on hard scans | Tokens-per-page at a target accuracy |
| Where does it pay off? | Digitizing paper documents | LLM context windows and training-data pipelines |
| What is the budget? | Compute to run the model once | Tokens that persist in the attention window |
| What is the failure mode? | Misread characters | Compression too aggressive, gist survives but detail blurs |
| What is the unit of comparison? | Word error rate | Compression ratio at iso-accuracy |

Once you see the table, the benchmark numbers stop reading like OCR scores and start reading like compression-codec specs. When DeepSeek-OCR reports that it keeps about **97% OCR accuracy at a ~10x compression ratio** (roughly ten text tokens of content per vision token) and still holds **~60% at 20x**, that is not a leaderboard brag. That is a rate-distortion curve: how much fidelity you keep as you crank the bitrate down. A JPEG quality slider has the same shape. The model is a lossy codec whose channel happens to be "vision tokens" instead of "DCT coefficients," and whose decoder happens to be a language model instead of an inverse cosine transform.

> An OCR model answers "what does this say?" An optical-compression model answers "how few tokens can say this?" DeepSeek-OCR is the second kind wearing the first kind's clothes.

That distinction is the whole post. Hold onto it; every architectural choice below exists to make the second question answerable in production.

## 1. The architecture: SAM, a 16x compressor, CLIP, then a small MoE

**Senior rule of thumb: when a vision encoder needs both high resolution and few output tokens, the resolution and the token count are governed by different stages — keep them separate or one will sabotage the other.** This is the single design insight that makes DeepSeek-OCR work, and it is the reason the encoder is called *DeepEncoder* rather than "a ViT."

A page of text is detail-dense. To read 9-point font you need a lot of input pixels — a 1024x1024 crop, sometimes tiled higher. But "a lot of input pixels" naively means "a lot of patches" means "a lot of tokens" means the exact token explosion you were trying to avoid. The classic vision transformer couples these: more resolution, more patches, more tokens, linearly. DeepSeek-OCR breaks the coupling by putting two different vision backbones in series with a convolutional bottleneck between them.

![DeepEncoder pipeline: a high-resolution windowed SAM stage feeds a 16x convolutional compressor, then a global CLIP stage emits 100 to 800 latent vision tokens into the DeepSeek3B MoE decoder.](/imgs/blogs/deepseek-ocr-optical-context-compression-2.webp)

Walk the pipeline left to right. The page image enters at high resolution. The **SAM stage** (the Segment Anything backbone, used here as a windowed local feature extractor) processes it with *windowed attention* — attention restricted to local neighborhoods rather than the whole image. Windowed attention is what keeps activations affordable at high resolution: you never form the full quadratic attention matrix over every patch, only over patches within a window. So SAM can look closely at fine strokes without the memory bill of global attention over thousands of patches.

Then comes the part that is easy to skim past and is actually the heart of the design: a **16x convolutional compressor**. After SAM has produced a dense, high-resolution feature map, a small stack of strided convolutions downsamples it by 16x *before* the next stage ever sees it. This is the move. Convolutions are cheap, they preserve local structure, and a 16x spatial reduction takes the token count from "thousands" to "hundreds." Crucially it happens between the two attention backbones, so the second backbone — the expensive global one — operates on the already-reduced representation.

That second backbone is the **CLIP stage**, used here for *global* attention: now that there are only a few hundred tokens, you can afford full attention across all of them, which is what lets the encoder reason about page-level layout, reading flow, and long-range structure. SAM saw the trees at high resolution; CLIP sees the forest at low token count. The compressor in the middle is the thing that makes the forest small enough to look at all at once.

The output is **100 to 800 latent vision tokens** depending on the resolution mode you select, and those tokens are what get handed to the decoder: **DeepSeek3B-MoE-A570M**, a 3-billion-parameter mixture-of-experts model with only about **570 million activated parameters** per token. We will come back to why the decoder is an MoE in section 4. For now, register the shape of the whole thing: high-res local encoder → 16x conv squeeze → low-res global encoder → tiny latent → small efficient decoder.

If you have read the DeepSeek-V3 architecture work, this philosophy will feel familiar — the same instinct that produced FP8 training and an auxiliary-loss-free load-balanced MoE shows up here as "spend compute where the information density is, starve it everywhere else." It is the same engineering culture applied to a vision problem. (For the language-model side of that culture, see [the DeepSeek-V3 FP8, MTP, and loss-free balancing write-up](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing).)

Here is the shape of the encoder forward pass in PyTorch-flavored pseudocode. Note the placement of the compressor — between the backbones, not after both.

```python
import torch
import torch.nn as nn

class DeepEncoder(nn.Module):
    def __init__(self, sam_backbone, clip_backbone, compress=16):
        super().__init__()
        self.sam = sam_backbone        # windowed local attention, high-res
        self.compressor = ConvCompressor(stride_total=compress)  # 16x squeeze
        self.clip = clip_backbone      # global attention, low token count

    def forward(self, page_image):
        local_feats = self.sam(page_image)        # dense high-res feature map
        squeezed = self.compressor(local_feats)   # 16x fewer spatial positions
        global_tok = self.clip(squeezed)          # full attention, cheap now
        return global_tok                         # 100..800 vision tokens


class ConvCompressor(nn.Module):
    def __init__(self, stride_total=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 2x
            nn.GELU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 4x
            nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 8x
            nn.GELU(),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),  # 16x
        )

    def forward(self, feat_map):
        return self.net(feat_map)
```

The detail that matters for production: because the compressor sits *before* the global attention stage, the global stage's attention matrix is built over a few hundred tokens, not a few thousand. That is a quadratic saving. The activation-memory peak of the whole encoder is governed by SAM's windowed attention (bounded by window size, not image size) and never by a giant global attention matrix. That is what the paper means when it says the encoder "maintains low activations under high-resolution input."

### Why two backbones instead of one

A reasonable objection: why not a single vision transformer with the compressor bolted on at the end? The answer is that the two backbones do genuinely different jobs that want genuinely different attention patterns, and trying to make one backbone do both is where the activation explosion comes from.

SAM's job is *local acuity*. To resolve fine print you need many input patches at small stride, and you need the model to look closely at each patch's neighborhood — the exact strokes, the serifs, the gaps between characters. Windowed attention is the right tool: each patch attends only to patches within a fixed-size window, so the attention cost is linear in the number of patches rather than quadratic. You can throw a 1024x1024 image at it and the memory stays bounded by window size, not image size. What SAM cannot do is reason about the whole page at once — windowed attention by construction never lets a patch in the top-left see a patch in the bottom-right.

CLIP's job is *global structure*. Reading order, column flow, the relationship between a heading and the paragraph under it, a caption and its figure — all of that is long-range, and long-range relationships need global attention where every token can see every other token. Global attention is quadratic, which is exactly why you cannot afford it at SAM's token count. But after the 16x convolutional squeeze, the token count has dropped from thousands to hundreds, and quadratic-over-hundreds is cheap. So CLIP gets to do the expensive global reasoning on a cheap number of tokens.

The compressor is the impedance-matcher between these two regimes. It takes SAM's high-resolution-but-local output and hands CLIP a low-resolution-but-globally-affordable input. Neither backbone alone could span both regimes without blowing the memory budget; the convolutional squeeze in the middle is what lets each stay in the regime it is good at. That is the deep reason the architecture is three stages and not one — it is a separation of concerns enforced by the cost structure of attention itself.

### Second-order optimization: tile, do not just upscale

The non-obvious gotcha is what happens when one resolution mode is not enough. A dense newspaper page or a multi-column journal article can carry more text than a single 1024x1024 crop resolves cleanly. The wrong instinct is to crank the input resolution globally — that re-inflates the SAM activation cost everywhere, including the margins and whitespace where there is nothing to read. The right move is *tiling*: split the page into regions, run the encoder per tile, and let the token budget scale with the regions that actually contain text. This is why DeepSeek-OCR reports a *range* of token counts (100 to 800) rather than a fixed number — the budget is content-adaptive. A sparse slide deck spends 100 tokens; a dense legal page spends closer to 800. You pay for ink, not for paper.

Tiling has a second benefit beyond cost: it bounds the worst case. A single global crop of an enormous dense page would either lose detail (too few pixels per character) or blow the budget (too many patches). Tiling decomposes the page into pieces each of which the encoder resolves cleanly at a known token cost, so the total token count grows predictably with the page's actual text content rather than exploding with its pixel dimensions. The senior framing: tiling turns "resolution" from a global knob that trades acuity against memory into a local, content-following allocation where every tile gets exactly the patches its text needs and no more.

## 2. The rate-distortion curve: how far you can push compression

**Senior rule of thumb: never quote a compression ratio without the accuracy it holds at — a codec is a curve, not a number.** DeepSeek-OCR's headline result is a two-point sketch of that curve, and reading it correctly tells you exactly which jobs this technique is safe for.

![Compression ratio against OCR accuracy and token budget: about 97 percent accuracy at 10x, about 60 percent at 20x, beating GOT-OCR2.0 at 100 tokens and MinerU2.0 under 800 tokens.](/imgs/blogs/deepseek-ocr-optical-context-compression-3.webp)

The matrix above lays out the two operating points and the two competitive comparisons. Read the rows top to bottom. At a **compression ratio under 10x** — meaning the text would have cost up to ten times as many tokens as the vision tokens carrying it — the model decodes back to text at about **97% accuracy**. That is the sweet spot: near-lossless at an order-of-magnitude saving. Push to **20x** and accuracy falls to about **60%**. That is not a cliff; it is a graceful slope, which is itself a useful property. A codec that degrades gracefully lets you choose your bitrate per use case instead of living in fear of a hard failure threshold.

The two comparison rows are where the practical punch lands:

- Against **GOT-OCR2.0**, which spends **256 vision tokens per page**, DeepSeek-OCR matches or beats it using only **100 vision tokens**. Roughly 2.5x fewer tokens for equal-or-better output.
- Against **MinerU2.0**, a heavyweight document-parsing pipeline that averages **6,000+ tokens per page**, DeepSeek-OCR comes out ahead using **fewer than 800 vision tokens**. That is not 2.5x; that is closer to an order of magnitude, on a much harder layout-parsing task.

Take the MinerU comparison seriously, because it is the one that reframes the whole field. MinerU's 6,000+ tokens are not wasteful by accident — full document parsing genuinely involves a lot of structure: headings, tables, figure captions, multi-column flow, math. The conventional approach represents all that structure explicitly, and explicit structure is verbose. DeepSeek-OCR represents it *implicitly*, inside the geometry of the rendered page, and lets the encoder compress the geometry. The structure is still there; it is just stored in pixels instead of tokens, and pixels-through-a-good-encoder are cheaper than tokens.

Here is a back-of-envelope worked example to make the tokens concrete. Suppose you have a 50-page technical report, about 700 words per page.

```python
def token_budget(pages, words_per_page, tokens_per_word_text=1.33,
                 vision_tokens_per_page=100):
    text_tokens = pages * words_per_page * tokens_per_word_text
    optical_tokens = pages * vision_tokens_per_page
    return text_tokens, optical_tokens, text_tokens / optical_tokens

text, optical, ratio = token_budget(pages=50, words_per_page=700)
print(f"text path:    {text:,.0f} tokens")
print(f"optical path: {optical:,.0f} tokens")
print(f"compression:  {ratio:.1f}x")
```

Running that gives roughly 46,550 text tokens versus 5,000 optical tokens — about a 9.3x reduction, right in the near-lossless band. A 50-page report that would blow past a 32k context window as text fits comfortably as optical tokens with room to spare. That is the practical shape of the win: documents that were "too long to paste" become "paste the whole thing."

### Second-order optimization: pick the operating point per document, not per system

The gotcha here is treating the compression ratio as a global config knob. It is not; it is a per-document decision. A page of dense legal prose where every clause matters wants the 10x near-lossless mode. An archived email thread you only need the gist of can ride at 20x and nobody will notice the dropped 40%. The model supports both because the resolution mode (and therefore the token budget) is selectable at inference time. The senior move is to route documents to operating points by how much their detail matters downstream — keep the bitrate high where errors are expensive, drop it where you only need recall of the topic.

### Why the curve has the shape it does

It is worth understanding *why* accuracy holds near 97% to 10x and then slopes rather than cliffs, because the shape tells you which failures to expect. The information content of a page of text is not uniformly distributed across its pixels. A huge fraction of the pixels are white space between words, margins, line spacing — pure redundancy that a vision encoder discards almost for free. The next tranche is the gross shape of words and lines: where text is, how it flows, the broad silhouette of each token. That survives heavy downsampling because it is low-frequency information. Only the last tranche — the fine distinctions between visually similar characters, the difference between "rn" and "m", between a comma and a period, between an 8 and a 6 — lives in the high-frequency detail that downsampling destroys first.

So as you crank the compression ratio up, you lose information in order of its spatial frequency: white space first (free), then nothing much for a while (the 10x near-lossless band), then the fine character distinctions (the slope from 10x toward 20x), and only at extreme ratios the word-level layout itself. That ordering is exactly why the curve is flat then sloped rather than linear: you spend the first chunk of compression on redundancy that costs nothing, which buys you the near-lossless band, and only start paying in real errors once you reach into the high-frequency character detail. The errors you get at 20x are therefore *character-level confusions in visually similar glyphs*, not scrambled layout or hallucinated paragraphs. That predictability is what makes the lossy mode usable: you know the failure is "occasionally misreads a digit," not "occasionally invents a sentence."

This also explains why the technique loves clean rendered text and struggles more with degraded scans. Rendered text puts all its information in crisp, high-contrast, predictable glyph shapes — the encoder knows exactly what frequency band carries the signal. A coffee-stained fax has its signal smeared across noise, so the encoder cannot cleanly separate redundancy from content, and the near-lossless band shrinks. The cleaner the source, the further right you can push the ratio before the slope begins.

### Resolution modes are quantization levels

DeepSeek-OCR does not expose a continuous compression dial; it exposes a small set of *resolution modes*, each producing a different token budget. Functionally these are quantization levels on the rate-distortion curve — discrete operating points you snap to, the way a video codec exposes a handful of CRF presets rather than a continuous quality float. A low-resolution mode emits around 100 tokens and sits in the cheap part of the curve; a high-resolution mode (or a tiled multi-crop mode) emits up to 800 and sits in the near-lossless part for dense pages. The reason for discrete modes rather than a continuous dial is pragmatic: each mode corresponds to a concrete input geometry the encoder was trained on, and the encoder's behavior is only well-characterized at the geometries it saw during training. You pick the cheapest mode whose accuracy clears your downstream requirement, and you do it per document class, not once for the whole system.

There is a subtlety production teams hit immediately: the token *budget* and the compression *ratio* are not the same axis. A sparse slide at the 100-token mode might be at 30x compression (very little text, heavily compressed), while a dense journal page at the 800-token mode might be at 8x (a lot of text, lightly compressed). The token budget is what your serving system pays; the compression ratio is what your accuracy depends on. When you size a context budget, you reason in tokens; when you predict accuracy, you reason in ratio. Keeping those two straight prevents the most common planning error — assuming "100 tokens" means "high compression" when it might just mean "a nearly empty page."

## 3. Optical tokens as a long-context mechanism

**Senior rule of thumb: the cost of a context window is not the tokens you read once — it is the tokens you carry forever.** This is the section where OCR stops being OCR and becomes a context-engineering technique, and it is the part of the DeepSeek-OCR paper I think people will still be citing in five years.

![Text context window versus optical context window: at about 1.33 tokens per word a text budget holds roughly 6 pages, while optical tokens at about 100 per page hold roughly 80 pages in the same budget.](/imgs/blogs/deepseek-ocr-optical-context-compression-4.webp)

The before/after above quantifies the reframing. On the left, a text context window: at roughly 1.33 tokens per word, key-value cache memory growing linearly per token, an 8k budget fills up after about six pages of real prose. On the right, the same 8k budget spent on optical tokens: at about 100 tokens per page, you fit closer to eighty pages. The information is the same; the *carrier* is denser. And because the KV cache grows per token, a denser carrier directly buys you a longer effective window for the same memory.

This connects to the deepest cost in transformer serving, the one I have written about separately: the [key-value cache](/blog/machine-learning/large-language-model/kv-cache). Every token you keep in context occupies KV-cache memory for as long as it stays in the window, and that memory is the binding constraint on how many requests a server can run concurrently and how long each context can be. Anything that reduces tokens-per-unit-of-information reduces KV pressure proportionally. Optical compression is, from the serving system's point of view, a KV-cache compression scheme that happens to route through a vision encoder. It sits in the same design space as latent-attention tricks like [multi-head latent attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla), which attack the same problem from inside the attention mechanism rather than from outside via the input representation. Two different layers of the stack, one shared enemy: the cache.

Let me be precise about what is and is not being claimed, because this is where hand-waving usually creeps in. The claim is *not* that optical tokens are free — they still occupy KV slots and still cost attention compute. The claim is that they pack more source information per slot. Picture the window as a fixed number of shelves. Text shelves each hold one short phrase; optical shelves each hold a paragraph. Same shelves, more content. The model still has to attend over the shelves, but there are fewer shelves to attend over for the same document, and that is the whole game when your bottleneck is the cache, not the encoder.

There is a sharp consequence worth stating plainly. If optical tokens are a denser context carrier, then the natural place to put *old* context is in optical form. You do not render the user's current question to an image — you need that crisp and token-cheap to manipulate. You render the stuff that has scrolled off, the reference material, the earlier turns whose exact wording no longer matters. Which is exactly the forgetting mechanism the next section describes.

### How optical compression sits among the other long-context tricks

It helps to place optical compression on the map of techniques that all attack context cost, because they operate at different layers and compose rather than compete. There are roughly four families.

The first family is **architectural attention compression** — sparse attention, sliding windows, linear-attention variants. These change *how* the model attends so that attention cost grows sub-quadratically. They keep every token but make attending to them cheaper. Optical compression is orthogonal: it reduces the *number* of tokens, and a sparse-attention model would happily attend over optical tokens.

The second family is **KV-cache compression** — quantizing the cache, evicting low-attention tokens, or projecting keys and values into a smaller latent space the way multi-head latent attention does. These shrink the bytes per token in the cache. Optical compression shrinks the *count* of tokens whose KV you store. Again orthogonal, and again composable: you can store optical tokens in a quantized or latent-projected cache and stack the savings.

The third family is **retrieval** — do not keep the long context in the window at all; store it externally and pull in only the relevant chunks per query. Retrieval is powerful but lossy in a different way: it depends on the retriever guessing what is relevant *before* the model reasons, and it loses the ability to attend jointly over the whole corpus. Optical compression keeps the whole corpus in the window, just cheaply, so the model can attend over all of it. The two combine well — retrieve to decide what to keep hot as text, demote the rest to optical.

The fourth family, the one optical compression belongs to, is **input-representation compression**: change the representation of the input itself so the same information arrives in fewer tokens. This is the least explored family because text seemed like the obvious, irreducible representation for language. Optical compression's contribution is showing that for stale, structured, or bulk context, *the image of the text is a denser input representation than the text* — and that a vision encoder is the compressor that makes it so.

The reason to enumerate the families is to make the composition explicit: optical compression does not replace your sparse attention, your KV quantization, or your retriever. It sits in front of all of them, shrinking the token count before any of them run, and every saving downstream multiplies against the optical saving upstream. That is the most important practical fact about it — it is additive with the entire rest of the long-context toolkit.

### Second-order optimization: render layout, not just characters

A subtle gotcha: if you flatten old context to plain text before rendering it, you throw away the layout that made the optical path efficient in the first place. The encoder is good *because* it exploits 2D structure — a table that is a clean grid in pixels is far cheaper to encode than the same table linearized into pipe-delimited text and re-rendered as a wall of monospace. When you push context into optical form, preserve the original visual layout where you can. The picture of a table is cheaper than the picture of the table's text dump.

## 4. The decoder: a small MoE reading vision tokens back into text

**Senior rule of thumb: a decoder that only ever turns one modality into another does not need to be large — it needs to be specialized and cheap to run a lot.** This is why the decoder is DeepSeek3B-MoE-A570M and not a 70B general-purpose model, and the choice is load-bearing for the throughput numbers in section 6.

![Mixture-of-experts decoder reading vision tokens into text: a router gates each step to two active experts while the rest stay idle, then their outputs combine into decoded text plus layout.](/imgs/blogs/deepseek-ocr-optical-context-compression-6.webp)

The graph above shows the decode path. Vision tokens come in, get projected to the decoder's hidden dimension, and hit a **mixture-of-experts router**. The router's job is to send each token to a small subset of experts — the "active" ones, drawn in green — while the rest of the experts sit idle for that token. The active experts process the token, their outputs are combined, and the model emits decoded text plus layout structure. The headline parameter count is **3 billion total, ~570 million activated**. You store a 3B model; you pay to run a 570M one on any given token.

Why does this shape fit the job so well? Because the decoder's task is narrow and repetitive. It is not reasoning about the world; it is performing a near-deterministic transduction — vision token → the characters and layout that token encodes. A narrow task does not need a dense forward pass over billions of parameters. It needs a *router* that can pick the right small expert for "this looks like a math region" versus "this looks like a paragraph of body text" versus "this is a table cell," and then a cheap specialized forward pass. Mixture-of-experts is exactly the architecture for "many narrow specializations, only a few relevant at a time." If you want the long version of why auxiliary-loss-free MoE routing is the DeepSeek house style, the [DeepSeek-V3 write-up](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) covers the routing machinery in depth.

Here is the decode loop in pseudocode, with the MoE routing made explicit. Note the comments are end-of-line, never at column zero, and the early-stop is what keeps generation bounded.

```python
def decode_page(vision_tokens, decoder, max_new_tokens=4096):
    hidden = decoder.project(vision_tokens)   # vision dim -> decoder dim
    output_ids = []
    state = decoder.init_state(hidden)
    for _ in range(max_new_tokens):
        logits, state = decoder.step(state)   # one MoE forward, ~570M active
        next_id = logits.argmax(dim=-1)       # greedy is fine for transduction
        if next_id == decoder.eos_id:
            break                             # page fully transcribed
        output_ids.append(next_id)
    return decoder.detokenize(output_ids)     # text + layout markup


def moe_forward(token_hidden, router, experts, top_k=2):
    scores = router(token_hidden)             # gate logits over all experts
    chosen = scores.topk(top_k).indices       # pick the k relevant experts
    out = 0.0
    for e in chosen:
        weight = scores[e].softmax_share()    # normalized gate weight
        out = out + weight * experts[e](token_hidden)
    return out                                # only top_k experts ran
```

The reason this matters for the whole optical-compression argument: the decoder has to be cheap enough to run *constantly*. If your plan is to use optical compression as a context mechanism — re-encoding old context, parsing documents on the fly, generating training data at scale — then the decode step is in the hot loop and its cost compounds. A 570M activated decoder is cheap enough to put in that loop. A dense 7B or 70B is not. The MoE is not a flourish; it is what makes the technique deployable.

It is worth being precise about the MoE economics, because "3B total, 570M active" can sound like a contradiction. You store all 3 billion parameters in memory — the full set of experts has to be resident. But for any single token, the router selects only a small top-k subset of experts, so the *compute* per token touches only ~570M parameters. You pay 3B in memory footprint and 570M in per-token FLOPs. For a decoder that runs in a tight loop, FLOPs per token is the cost that compounds across millions of pages, so optimizing the active count is exactly the right thing to optimize. The memory footprint is a one-time cost paid at load; the FLOPs are paid on every one of the billions of tokens you decode. MoE lets you have a large, capable parameter store (3B worth of specialization) while paying small-model inference cost — which is the entire reason the architecture exists, and a near-perfect fit for a transduction task with many narrow sub-skills.

There is a further reason MoE suits transcription specifically. Document content is genuinely heterogeneous in a way that rewards specialization: body prose, tables, math notation, code listings, form fields, headers, footnotes — each is almost a different "language" with its own conventions. A dense model has to encode all those conventions in one shared set of weights, smearing capacity across them. An MoE can dedicate distinct experts to distinct content types and let the router dispatch each region to the expert that knows it. The router learns to recognize "this region looks like math" and send it to the math-fluent experts, "this looks like a table cell" to the table experts. That division of labor is why a 570M-active MoE can match the transcription quality of a much larger dense model on the diverse content a real document throws at it.

### Second-order optimization: greedy decoding is usually correct here

A counterintuitive gotcha for people coming from open-ended generation: you almost never want sampling for this decoder. Transcription is a near-deterministic map from image to text, so temperature and nucleus sampling mostly inject *errors*, not creativity. Greedy or very-low-temperature decoding gives the most faithful transcription. The exception is genuinely ambiguous regions — a smudged digit, an occluded character — where you might want the model to surface its uncertainty rather than commit. But the default should be deterministic, and if you find yourself raising temperature to "fix" outputs, the real problem is upstream resolution, not decoding.

## 5. Optical memory: compression as a model of forgetting

**Senior rule of thumb: human memory keeps recent things sharp and old things as gist — a context system that does the same gets long memory at constant cost.** This is the most speculative and, to me, the most exciting idea in the DeepSeek-OCR paper, and it is the one that fully justifies calling this work "context compression" rather than "OCR."

![Optical memory forgetting across context age: the current and recent turns stay as crisp text, older turns are rendered to images at 10x compression, older still at 20x, and the most ancient as a tiny lossy thumbnail.](/imgs/blogs/deepseek-ocr-optical-context-compression-5.webp)

The timeline above sketches the scheme. Read it left to right as context ages. The **current turn** is full-fidelity text — crisp, token-cheap-to-manipulate, exactly what the model needs to act on right now. The **last few turns** stay as verbatim text too; recent context is usually relevant and you want it lossless. But as a turn ages out of immediate relevance, you **render it to an image at ~10x compression** — still highly recoverable, now far cheaper. Older still, you **downsample harder to ~20x**, accepting the drop to ~60% recoverability because you only need the gist now. And the truly ancient context becomes a **tiny thumbnail** — a lossy impression, enough to know *that* a topic came up and roughly what it was, not enough to quote it verbatim.

This is a strikingly good model of how human memory actually behaves. You remember this morning's conversation almost word for word. Last week's, you remember the substance but not the exact phrasing. Last year's, you remember that it happened and the broad shape, and the details have blurred into gist. The fidelity of a memory decays with its age, and crucially *the storage cost decays with it too* — you are not spending the same neural budget on a decade-old memory as on a fresh one. Optical compression gives an LLM the same property: a tunable knob, per context segment, that trades fidelity for tokens, set as a function of age.

Contrast this with how context windows work today. Today's window is a hard wall with sharp eviction: tokens are either fully present at full price or completely gone. There is no middle setting, no "keep this but blurry." When you hit the limit, something gets dropped entirely, and if you guessed wrong about what was safe to drop, the model simply cannot see it anymore. Optical memory replaces the cliff with a slope. Nothing has to be evicted to zero; it can instead be demoted to a cheaper, lossier representation and kept around. That is a fundamentally different shape of forgetting — graceful instead of catastrophic.

Here is a concrete policy sketch for an optical-memory manager. It is deliberately simple; the point is the *shape*, a per-segment fidelity tier chosen by age and importance.

```python
def memory_tier(age_turns, importance):
    # importance in [0, 1]; higher keeps text longer
    effective = age_turns * (1.0 - 0.7 * importance)
    if effective < 2:
        return "text"            # verbatim, full token cost
    if effective < 8:
        return "optical_10x"     # ~97% recoverable, 10x cheaper
    if effective < 24:
        return "optical_20x"     # ~60% recoverable, 20x cheaper
    return "thumbnail"           # gist only, near-free

def repack_context(segments):
    packed = []
    for seg in segments:
        tier = memory_tier(seg.age, seg.importance)
        if tier == "text":
            packed.append(seg.as_text())
        elif tier == "thumbnail":
            packed.append(seg.render(scale=0.15))   # lossy gist
        else:
            ratio = 10 if tier == "optical_10x" else 20
            packed.append(seg.render_to_optical(ratio))
    return packed
```

I want to be honest about the status of this idea: the *mechanism* (optical compression at a chosen ratio) is demonstrated and benchmarked in the paper; the *memory-management policy* built on top of it is the paper's proposed direction, not a shipped feature. But the direction is sound, and it is the kind of thing that gets built once the primitive exists. The primitive — "re-encode this text segment to N vision tokens at a chosen fidelity" — now exists and is cheap. The forgetting policy is a layer you can write on top.

### Second-order optimization: importance, not just age, should drive the tier

The gotcha in any age-based forgetting scheme is that age is a crude proxy for relevance. A user's stated constraints from turn one ("I am vegan, I have a peanut allergy") must stay sharp forever even as they age, while a tangent from two turns ago can be demoted immediately. The policy sketch above multiplies age by an importance factor for exactly this reason. In practice you want a small classifier or heuristic deciding importance — pinned facts, user constraints, and decisions stay text; chit-chat and resolved sub-threads get demoted fast regardless of age. Age sets the default tier; importance overrides it.

## 6. The data engine: 200k pages a day on one GPU

**Senior rule of thumb: the most valuable thing a cheap, accurate document model produces is not transcriptions — it is training data.** This is the quiet commercial logic underneath DeepSeek-OCR, and it is why the throughput number is in the abstract at all.

![Data engine throughput and benchmark scores: DeepSeek-OCR sustains 200k-plus training pages per day on one A100-40G, and DeepSeek-OCR-2 reaches 91.09 percent on OmniDocBench at 100 to 800 tokens per page.](/imgs/blogs/deepseek-ocr-optical-context-compression-8.webp)

The matrix above puts the throughput next to the competition. A single **A100-40G** GPU generates **200,000+ pages of training data per day**. That number only makes sense once you internalize the architecture from sections 1 and 4: the encoder keeps activations low so you can batch aggressively at high resolution, and the decoder activates only ~570M parameters so each page transcribes cheaply. Low activation memory plus a cheap decoder equals enormous batch throughput. The token efficiency (100–800 per page versus MinerU's 6,000+) compounds it — fewer decode steps per page means more pages per second.

Why does a *data engine* matter more than the transcriptions themselves? Because modern LLM and VLM training is bottlenecked on high-quality, structured document data: clean text with preserved layout, tables that survived parsing, math that rendered correctly, figures with captions still attached. That data is scarce and expensive to produce by hand. A model that can churn out 200k clean structured pages per day, per GPU, is a *flywheel* — it produces the training data that improves the next model that produces the training data faster. The OCR model is a means; the corpus is the end. This is the same flywheel logic behind the most aggressive infrastructure plays in the field, and it is why a "mere OCR model" got a flagship release.

Here is the throughput arithmetic made explicit, because the number sounds implausible until you do it.

```python
def pages_per_day(gpu_batch, decode_ms_per_page, gpus=1):
    pages_per_sec = (gpu_batch / decode_ms_per_page) * 1000.0
    return pages_per_sec * 86400 * gpus

est = pages_per_day(gpu_batch=32,  # large batch from low activation mem
                    decode_ms_per_page=12.0)  # cheap 570M-active decode
print(f"~{est/1000:,.0f}k pages/day")
```

The exact constants depend on resolution mode and page density, but the structure is the point: a large effective batch (enabled by low activation memory) times a cheap per-page decode (enabled by the 570M-active MoE) lands you in the hundreds-of-thousands-per-day range on one card. Multiply by a modest cluster and you are producing tens of millions of structured training pages per day.

The flywheel is worth spelling out because it is the actual economic engine, and it is self-reinforcing in a way that compounds.

| Stage | What happens | What it enables next |
|---|---|---|
| Bootstrap | A first OCR model trained on existing labeled data | Can transcribe pages, imperfectly |
| Harvest | Run it over a huge unlabeled document corpus | Millions of (image, text+layout) pairs, cheaply |
| Filter | Keep high-confidence transcriptions, drop the rest | A clean structured-document training set |
| Retrain | Train a better OCR/VLM on the harvested set | A model that transcribes more accurately and faster |
| Compound | The better model harvests more, cleaner, faster | The training set grows in quality and size each loop |

Each turn of that loop produces a model that makes the next turn cheaper and higher-quality. The throughput number — 200k+ pages/day/GPU — is what sets the *speed* of the flywheel. At hand-labeling speeds the loop takes years; at 200k pages/day/GPU it takes days. That speed difference is the whole moat. It is the same reason the most aggressive labs treat data engines as core infrastructure rather than a preprocessing step: whoever spins the flywheel fastest accumulates the best structured corpus, and the best corpus trains the best document model, which spins the flywheel faster still.

There is also a quality dimension people underweight. The output of this engine is not raw text — it is text with *preserved structure*: reading order, table boundaries, heading hierarchy, figure-caption linkage. That structured supervision is precisely what general document-understanding models are starved for. Plain OCR text teaches a model to read characters; structured output teaches it to understand documents. A flywheel that produces structured output at scale is therefore not just cheaper than buying OCR transcriptions — it produces a fundamentally more valuable kind of data than most commercial OCR APIs return at all.

### Second-order optimization: throughput is a function of resolution mode

The gotcha in quoting "200k pages/day" is that it is mode-dependent. Sparse pages at the 100-token setting decode far faster than dense pages at the 800-token setting, so a corpus of mixed densities will not hit peak throughput uniformly. If you are building a data pipeline, profile your *actual* document mix and route by density: send slide decks and forms through the cheap mode, reserve the expensive mode for dense journal pages, and your aggregate pages-per-day stays high. Quoting the peak number for a corpus that is all dense math papers will disappoint you.

A second, sharper gotcha: confidence filtering is not optional in a data engine. The flywheel only works if the harvested data is *clean*, and at 200k pages/day you will harvest plenty of garbage — failed transcriptions, scrambled tables, hallucinated regions on degraded scans. The discipline that makes the engine produce a moat instead of a swamp is aggressive filtering: keep only transcriptions the model is confident about, cross-check structured regions against simple heuristics (does this "table" actually have consistent columns?), and throw away the rest without sentiment. A smaller, cleaner harvested set trains a better next model than a larger, noisier one. The throughput buys you the luxury of being picky — use it.

## 7. DeepSeek-OCR-2: learned reading order over raster scan

**Senior rule of thumb: a vision model that reads a page in fixed raster order will interleave columns mid-sentence — the reading order is part of the model, not a post-processing afterthought.** DeepSeek-OCR-2 (arXiv 2601.20552, January 2026), subtitled *Visual Causal Flow*, is the sequel that fixes the most stubborn failure mode of the first model, and it does so with a genuinely novel architectural idea.

![DeepSeek-OCR-2 learned reading order versus fixed raster scan: instead of a rigid top-left to bottom-right sweep that interleaves columns, DeepEncoder V2 follows column and block flow and reaches 91.09 percent on OmniDocBench.](/imgs/blogs/deepseek-ocr-optical-context-compression-7.webp)

The before/after above states the problem and the fix. Conventional vision-language models — including DeepSeek-OCR v1 — process visual tokens in **rigid raster-scan order**: top-left to bottom-right, row by row, the way a CRT beam paints a screen. For a single-column page that is fine. For a two-column journal article, a newspaper, or a form with a complex layout, it is a disaster: raster order walks across the top of *both* columns before coming back for the second line, so the decoded text interleaves the left column and the right column mid-sentence. The structure of the page fights the structure of the scan, and the scan wins, garbling the output.

DeepEncoder V2's idea is to make the encoder **reorder visual tokens by the document's own semantics before the decoder ever reads them** — a *learned reading order* instead of a fixed one. The paper frames this as endowing the encoder with causal reasoning: it figures out the logically coherent sequence in which a human would read the page (down column one, then down column two; header before body; caption with its figure) and emits the visual tokens in *that* order. The decoder then receives a sequence that already flows correctly, so it can transduce it linearly without having to untangle the 2D layout itself.

The framing the authors use is worth quoting in spirit: rather than forcing a 2D image into one rigid 1D raster scan, they ask whether 2D understanding can be achieved through *two cascaded 1D causal reasoning structures* — one pass to figure out the reading order, another to read in that order. It is inspired by how human vision handles complex layouts: we do not scan a magazine page like a raster beam, we follow the logical flow, jumping where the layout tells us to. The result lands at **91.09% on OmniDocBench**, the document-understanding benchmark, which is the kind of score that says the column-interleaving problem is genuinely solved, not patched.

Here is the conceptual shape of the two-pass flow. The first pass scores a reading order over the visual tokens; the second emits them in that order.

```python
def visual_causal_flow(vision_tokens, order_head, decoder):
    # pass 1: learn a reading order over the 2D token grid
    order_scores = order_head(vision_tokens)        # causal-flow scoring
    reading_order = order_scores.argsort_sequence() # logical, not raster

    # pass 2: emit tokens in learned order, then decode linearly
    reordered = vision_tokens[reading_order]
    return decoder.decode(reordered)                # flows correctly now
```

The reason this matters beyond document parsing: it generalizes the optical-compression thesis to *structured* content. Optical compression v1 works best on text that reads in a simple order. Once the encoder can learn an arbitrary reading order, the optical channel can faithfully carry arbitrarily complex layouts — tables, forms, multi-column scientific papers — without the structure getting scrambled on the way in. The denser, more structured the source, the more the learned reading order matters, and the more optical compression beats explicit token-level structure representation. v2 widens the set of documents for which "render it to an image" is the *better* representation, not just a cheaper one.

### Why "two cascaded 1D structures" is the clever part

The phrase "two cascaded 1D causal reasoning structures" deserves unpacking, because it is the architectural insight, not just marketing. A 2D image is hard for a causal language-style decoder precisely because causal decoding is inherently 1D — it produces one token after another in a strict order, and there is no canonical order in which to flatten a 2D grid. The raster scan is the default flattening, and it is wrong for anything but single-column text.

DeepSeek-OCR-2's move is to *not* pick a fixed flattening. Instead it uses one 1D causal structure to *decide the order* — a causal pass over the visual tokens that scores a reading sequence — and a second 1D causal structure to *consume that order* and emit text. Two 1D passes, the first feeding the second. The claim is that two well-chosen 1D causal passes can recover the 2D reasoning that a single fixed flattening throws away. The first pass asks "in what order should I read this page?"; the second asks "given that order, what does it say?". Splitting the problem this way lets each pass stay a clean 1D causal model — the kind transformers are very good at — while together they handle genuine 2D layout.

This is a different bet from the obvious alternative, which is to make the decoder itself 2D-aware with positional encodings that capture the grid. That approach exists and is heavier; it asks the decoder to learn layout and content jointly. The cascaded-1D bet is that *separating* layout (pass one) from content (pass two) is both cheaper and more robust, because each pass is a simpler problem than the joint one. The 91.09% OmniDocBench result is the evidence that the bet pays off on real, layout-complex documents.

### What carries over to optical compression

For the optical-compression thesis specifically, v2 matters because it removes the last big caveat on the v1 story. v1's compression is cleanest when the source reads in a simple order; complex layouts risked the column-interleaving garble, which capped how aggressively you could push optical compression on structured documents. With a learned reading order, the optical channel can carry arbitrarily complex layouts faithfully, which means the compression advantage — already largest on dense structured pages — now applies *without* the layout-scrambling risk that used to come with those same pages. v2 does not just improve a benchmark; it makes "render the structured document to an image" a safe default rather than a gamble.

### Second-order optimization: the reading-order pass is also a layout signal

A non-obvious benefit: the learned reading order is itself a useful output, not just an internal step. If the encoder has decided that these tokens form column one and those form column two, you have a *free layout segmentation* — you know which regions are headers, which are body, which are captions, because the reading-order model had to infer exactly that to order them. Downstream consumers that want structured output (reconstruct the table, extract just the abstract, link captions to figures) can read the order model's intermediate decisions instead of re-inferring layout from the flat decoded text. Do not throw that signal away by only keeping the final transcription.

## Cross-cutting concerns: latency, verification, and failure handling

Before the worked examples, three operational concerns that cut across every use case and that I would raise in any design review of a system built on this technique.

**Latency budgeting.** The optical path adds an encode step and a decode step that the pure-text path does not have. For a document you are storing once and querying many times, that round-trip amortizes to nothing — you encode at ingest, then reason over cheap optical tokens forever. For a single-shot request where the document is seen once, the round-trip is pure overhead and you should think hard about whether you need compression at all. The rule: optical compression pays when the encode cost is amortized over many reads. Put the encode at ingest time, off the request hot path, and the latency concern mostly evaporates. Put it in the request path and it becomes the thing you tune.

**Verification before quoting.** At 10x you are at ~97% fidelity, which is excellent for recall and reasoning but not good enough to blind-quote in a context where exactness matters. The discipline is a two-tier read: use the optical tokens for *finding* and *reasoning* — locating the relevant clause, understanding what a section says — but when you need to *quote* it verbatim, re-render that specific region at full resolution and read it losslessly. You get the cheap broad recall of optical compression for the 99% of operations that only need the gist, and pay the full-resolution cost only on the rare operation that needs the exact characters. This two-tier pattern is what makes the lossiness safe in production: the system never commits the model's compressed reading to a place where an error is unrecoverable without first verifying against the source.

**Failure-mode containment.** The failures at high compression are character-level confusions in visually similar glyphs, as the rate-distortion discussion established. That predictability lets you contain them. Numeric fields — account numbers, dollar amounts, dates — are exactly where character confusions hurt most and where they are easiest to catch, because numbers have checksums, ranges, and formats you can validate. The senior pattern is to treat any number read from optical tokens as provisional and validate it structurally (does this date parse? is this amount in a plausible range? does this account number pass its checksum?) before acting on it. Prose can tolerate the occasional misread character because context disambiguates; structured numeric data cannot, so guard it. Knowing the failure *shape* is what lets you build the right guard.

## Worked examples: applying optical compression in practice

Theory is cheap. Here are concrete scenarios with the actual decision logic you would use, written the way I would reason through them in a design review.

### Worked example 1: pasting a 200-page contract into an assistant

A user wants to ask questions about a 200-page contract. As text, at ~700 words per page and ~1.33 tokens per word, that is roughly 186,000 tokens — past most context windows, and ruinously expensive in KV cache even where it fits. The optical path: render all 200 pages, encode each at the 10x near-lossless mode (~100 tokens/page), and you are at ~20,000 vision tokens for the whole contract at ~97% fidelity. The user's *current question* stays as text (crisp, cheap to manipulate); the contract sits in optical form as the reference corpus. When the model needs to quote an exact clause, the 97% fidelity is high enough that you re-render just that page at full resolution to verify. Token budget: 20k optical + a few hundred for the live conversation, versus 186k text. The contract went from "does not fit" to "fits nine times over."

### Worked example 2: a long-running agent's scrollback

An agent has been running for hours: dozens of tool calls, file reads, intermediate results. Most of it is stale — the file you read forty steps ago, the search results you already acted on. Apply the optical-memory tiering from section 5. The last few steps stay text. Steps 5–20 back render at 10x. Everything older becomes a 20x optical thumbnail or gets demoted to gist. The agent's *effective* memory now spans hundreds of steps at a token cost that would otherwise only buy a few dozen. The pinned facts — the task definition, the user's constraints, the success criteria — are tagged high-importance and stay text regardless of age, per the importance override. The agent stops "forgetting" what it was doing on long tasks not because the window grew but because the carrier got denser.

### Worked example 3: building a structured-document training corpus

You need ten million pages of clean, layout-preserved training data for a document VLM. Hand-labeling is out of the question; commercial OCR APIs at scale cost more than the GPUs. Stand up DeepSeek-OCR on a handful of A100s, route by document density (cheap mode for forms and slides, expensive mode for dense journal pages, per section 6's gotcha), and run DeepSeek-OCR-2's learned reading order so multi-column papers come out un-interleaved. At 200k pages/day/GPU, ten cards clear ten million pages in five days. The output is not just text — it is text with preserved reading order and layout signal, which is exactly the structured supervision the next model needs. The data engine is the product.

### Worked example 4: choosing between optical compression and a bigger window

A team is debating whether to license a 1M-token context model or build an optical-compression layer. The right question is *what kind of length you need*. If you need to reason densely over a million tokens where every token might be jointly relevant — say, a giant codebase where any file might call any other — a true long-context model is the right tool, and optical compression's lossiness would hurt. If instead you have a long *tail* of mostly-stale reference material with a small hot working set — the common case for assistants and agents — optical compression gives you the length far cheaper, because the stale tail tolerates lossy storage. Match the tool to the relevance distribution: dense-everywhere wants real long context; hot-set-plus-stale-tail wants optical compression.

### Worked example 5: a multi-column research-paper search index

You are indexing a hundred thousand multi-column research papers so an assistant can answer questions across them. Two failure modes loom. First, multi-column layout: a raster-order encoder would interleave columns and produce garbled, unsearchable text — this is exactly the problem DeepSeek-OCR-2's learned reading order solves, so route papers through v2 and the columns come out in correct reading order. Second, density: research pages are dense, so they want the high-resolution mode (closer to 800 tokens/page) to stay in the near-lossless band. The math, in particular, is unforgiving — a misread exponent changes a formula's meaning — so this is a "keep the bitrate high" corpus, not a place to economize at 20x. The payoff is that each paper, multi-column math and all, lands as a few thousand faithful optical tokens with preserved reading order, indexable and re-renderable on demand. The cost discipline here is the opposite of the email-archive case: density and math mean you pay for fidelity, and you should.

### Worked example 6: deciding what stays text in a customer-support bot

A support bot accumulates context across a long ticket: the customer's original problem, their account details, a dozen back-and-forth messages, and pasted logs. Apply importance-weighted tiering. The customer's stated problem and account constraints are pinned high-importance and stay text forever — getting those wrong is the cardinal sin of support. The pasted logs, which are bulky and only occasionally re-referenced, are prime candidates for optical compression: render them, store them at 10x, and re-render the relevant span at full resolution only if the bot actually needs to quote a specific error line. The resolved earlier exchanges ("did you try restarting?" "yes, didn't help") demote to 20x or thumbnail quickly — they are conversational scaffolding, not facts. The result is a bot that holds the entire ticket in context cheaply while never blurring the two things that must stay sharp: what the customer wants and who they are. The lesson that generalizes: in importance-weighted tiering, identify the two or three facts that must never blur, pin them, and let everything else ride the age curve.

## When to reach for optical compression, and when not to

A technique this novel attracts both overuse and dismissal. Here is the honest boundary.

### Reach for optical compression when

- **You have long, mostly-stale context with a small hot working set.** Reference documents, prior turns, pasted material you need available but rarely quote verbatim — the lossy tail is exactly what tolerates compression. This is the canonical fit.
- **KV-cache memory is your binding constraint**, not encoder compute. Optical compression trades a one-time encode cost for persistent KV savings; that trade only pays when the cache is what is limiting you, which on busy serving fleets it usually is.
- **The source has real 2D layout** — tables, multi-column pages, forms, math. Layout stored in pixels and decoded by a good encoder is cheaper than layout serialized into verbose token-level structure. The denser the layout, the bigger the win, especially with v2's learned reading order.
- **You are producing training data at scale.** The 200k-pages/day/GPU economics make the data-engine use case almost unfair against hand-labeling or commercial OCR APIs.
- **Graceful degradation is acceptable** for the old/cold parts of your context. If "remember the gist, not the exact words" is fine for stale material, the 20x mode is nearly free recall.

### Skip optical compression when

- **Every token might be jointly relevant.** Dense reasoning over a large body where any part can interact with any other (a sprawling codebase, a tightly cross-referenced proof) wants true long context; lossy compression of the cold parts will silently drop something that turns out to matter.
- **You need verbatim fidelity on the whole corpus.** At 10x you are at ~97%, not 100%. For exact legal citation, code that must compile, or anything where a single dropped character is a bug, keep the hot material as text and use optical compression only for the reference tail you will re-verify before quoting.
- **Your context is already short.** If you comfortably fit in the window as text, adding an encode/decode round-trip buys you nothing and adds latency and a failure surface. Compression is for when you are *over* budget.
- **You cannot afford the encode latency in the hot path.** Encoding a page is cheap but not free. For ultra-low-latency single-turn requests where context is small, the round-trip is pure overhead.
- **The source is plain, single-column, short text.** The compression advantage is largest on dense and structured pages; on a short single-column note the token savings are modest and the BPE text path is simpler and lossless.

> Optical compression is a memory-hierarchy trick, not a magic context extender. Treat vision tokens like an L2 cache for old context: bigger, cheaper, slightly slower, and a little lossy. You would not run your CPU entirely out of L2 — and you should not run your model entirely out of optical tokens. Keep the hot set in text; demote the cold set to pixels.

## The bigger picture

Step back from the benchmarks and the architecture and the idea here is almost philosophical. We have spent years treating text as the canonical, lossless, privileged representation for language models — the thing images get *converted into*. DeepSeek-OCR inverts the hierarchy for one specific and important purpose: when text is too expensive to keep, an image of the text is a *better* representation, because a vision encoder can compress it harder than a tokenizer ever will. Vision becomes a compression codec for language, and the OCR decoder becomes the decompressor.

That inversion is the whole contribution. The DeepEncoder, the 16x convolutional squeeze between SAM and CLIP, the 570M-active MoE decoder, the rate-distortion curve, the data engine, v2's learned reading order — every one of those is in service of making "render the text to an image" a *cheaper* operation than "keep the text as tokens," at a controllable fidelity. Once that is true, a cascade of consequences follows: longer effective context at fixed KV cost, a graceful age-based forgetting mechanism that mirrors human memory, a training-data flywheel that runs on one GPU. The OCR is the demo. The codec is the discovery.

The honest open question is how far it generalizes beyond documents. Rendered text is the friendliest possible case for a vision encoder — clean, high-contrast, structured. Whether optical compression at these ratios holds up on noisier sources, on handwriting, on non-Latin scripts at the same fidelity, is exactly the kind of thing the next few papers will pin down. But the primitive is real and cheap today, the rate-distortion curve is measured, and the framing — *an image can be cheaper than its text* — is the kind of idea that, once you have seen it, you cannot unsee in every long-context system you build afterward.

### What to watch for next

If the trajectory from v1 to v2 is any guide, the next moves are predictable in shape. v1 established the codec; v2 fixed the reading-order problem that capped its use on structured pages. The natural v3-shaped questions are about pushing each axis further. On the *fidelity* axis: can the near-lossless band extend past 10x, so the safe operating point gets cheaper? On the *robustness* axis: does the curve hold on degraded scans, handwriting, and dense non-Latin scripts, or does the redundancy-first compression ordering break down when the glyph shapes are noisier or denser? On the *integration* axis: does optical memory move from a proposed policy to a shipped feature inside a serving stack, with a real importance classifier deciding tiers? Each of those is a concrete, measurable next step rather than a vague aspiration, which is a good sign the research direction is healthy.

The integration question is the one I would watch most closely, because it is where the idea stops being a clever OCR result and becomes a piece of LLM infrastructure. The moment a serving system ships an optical-memory tier — old context automatically demoted to vision tokens at age-and-importance-weighted fidelity, transparently, behind the context-window API — the "OCR" framing falls away entirely and what is left is a memory hierarchy. That is the destination this work is pointed at: not better document parsing, but a context window that forgets gracefully the way memory should, with a vision encoder as the compression stage of the hierarchy. The OCR benchmarks are how you prove the codec works. The memory hierarchy is what the codec is *for*.

### The lineage in one sentence

If you want the one-sentence intellectual lineage: DeepSeek spent the V3 era learning to spend compute only where information density justified it — FP8 where precision was not needed, sparse MoE experts where dense capacity was wasted — and DeepSeek-OCR applies that exact instinct to the input representation itself, discovering that for stale and structured context, the densest place to spend your token budget is on an image of the text rather than the text. Same engineering philosophy, one layer further down the stack than anyone had thought to apply it.

## Further reading

- **DeepSeek-OCR: Contexts Optical Compression** — arXiv 2510.18234. The primary source for the DeepEncoder architecture, the 10x/97% and 20x/60% rate-distortion points, and the optical-memory thesis.
- **DeepSeek-OCR-2: Visual Causal Flow** — arXiv 2601.20552. The sequel introducing DeepEncoder V2's learned reading order and the 91.09% OmniDocBench result.
- [DeepSeek-V3: FP8 training, multi-token prediction, and loss-free MoE balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the language-model engineering culture (efficient MoE, spend-compute-where-it-counts) that the OCR work inherits.
- [The KV cache: the real cost of context](/blog/machine-learning/large-language-model/kv-cache) — why tokens-per-unit-of-information is the binding serving constraint that optical compression attacks from the input side.
- [Multi-head latent attention (MLA)](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — a sibling KV-compression technique that attacks the same cache cost from inside the attention mechanism rather than the input representation.
- **GOT-OCR2.0** and **MinerU2.0** — the document baselines DeepSeek-OCR is measured against (256 tokens/page and 6,000+ tokens/page respectively); worth reading to see what "explicit structure representation" looks like by comparison.
