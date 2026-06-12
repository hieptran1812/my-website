---
title: "DeepSeek-VL and VL2: Dynamic Tiling and a MoE/MLA Backbone for Efficient Vision-Language Models"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A technique deep-dive on building efficient vision-language models: DeepSeek-VL's hybrid encoder and modality-competition management, then VL2's dynamic tiling on the vision side and a DeepSeekMoE/MLA decoder on the language side."
tags:
  - "vision-language-models"
  - "deepseek-vl"
  - "deepseek-vl2"
  - "dynamic-tiling"
  - "mixture-of-experts"
  - "multi-head-latent-attention"
  - "siglip"
  - "vision-encoder"
  - "multimodal"
  - "ocr"
  - "document-understanding"
  - "efficient-inference"
category: "machine-learning"
subcategory: "Computer Vision"
author: "Hiep Tran"
featured: true
readTime: 51
---

## The mismatch that DeepSeek-VL set out to fix

A rule of thumb I keep coming back to when reviewing multimodal architectures: **a vision-language model is a language model that happens to have eyes, not a vision model that happens to talk.** Most of the early open VLMs got this backwards. They bolted a vision encoder onto a frozen language model, fine-tuned the glue, and shipped a system that could caption a cat but had quietly lost a chunk of its reasoning, math, and coding ability in the process. The demo looked great. The regression on text-only benchmarks did not.

DeepSeek-VL (arXiv 2403.05525, March 2024) took the opposite stance, and stated it plainly: *a proficient vision-language model should, foremost, possess strong language abilities.* That single sentence is the design axis the whole first paper rotates around. The second paper, DeepSeek-VL2 (arXiv 2412.10302, December 2024), keeps the stance but rebuilds the machinery underneath it — dropping the dual-encoder vision front end for a single tiled ViT, and swapping the dense language decoder for a sparse Mixture-of-Experts decoder with Multi-head Latent Attention.

![Two encoders split the labor: SigLIP reads meaning, SAM-B reads pixels, the LLM reads both.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-1.webp)

The diagram above is the mental model for the first paper, and the rest of this post is a tour of how it evolves. DeepSeek-VL runs two vision encoders in parallel: a SigLIP path that reads semantics at low resolution, and a SAM-B path that reads pixel-level detail at high resolution. Their features are concatenated into a single stream of 576 visual tokens, projected by a small adaptor, and fed into a DeepSeek language decoder. VL2 then asks a sharper question: can we get the same high-resolution coverage with *one* encoder, and can we make the language half cheaper to run at scale without making it dumber? The answers — dynamic tiling on the vision side, DeepSeekMoE plus MLA on the language side — are what make the VL2 series interesting to anyone shipping VLMs in production.

This is a technique deep-dive, not a leaderboard recap. We will spend most of our time on three things: how the hybrid encoder actually fuses two resolutions, how modality competition is managed during training so the language model does not regress, and how VL2's tiling and MoE/MLA backbone trade total parameters for activated parameters. If you serve VLMs, the activated-vs-total distinction alone is worth the read.

### Why this architecture is different

| Assumption | The naive view | The DeepSeek-VL reality |
| --- | --- | --- |
| One vision encoder is enough | Pick CLIP/SigLIP, freeze it, done | VL1 runs *two* encoders (semantic + detail) and fuses them |
| Language ability is a given | The LLM stays smart automatically | Joint training silently erodes text skills unless the data ratio is managed |
| High resolution means a bigger encoder | Scale the ViT to 1024 | VL2 keeps a 384px ViT and *tiles* the image instead |
| MoE means a sparse vision model | Experts everywhere | The MoE lives only in the **language decoder**, not the vision encoder |
| Total params predict cost | A 27B model costs like a 27B model | VL2-Base activates only ~4.5B of its 27B per token |

Hold onto the last two rows. They are the most common things people get wrong about VL2, and we will return to both with numbers.

## 1. The hybrid vision encoder in DeepSeek-VL

**Senior rule of thumb: if a single encoder is forcing you to choose between "sees the whole scene" and "reads the fine print," you do not have a resolution problem, you have an encoder-count problem.**

The tension VL1 confronts is concrete. A SigLIP-style contrastive encoder is trained on image-text pairs at modest resolution — DeepSeek-VL uses SigLIP-L at 384x384. That model is fantastic at *semantics*: it knows a "golden retriever on a skateboard" because millions of captions taught it the association. But 384x384 is brutal for anything that depends on fine spatial detail. Small text in a screenshot, the gridlines of a table, the tick marks on a chart axis — all of it gets averaged into mush by the time you downsample a document to 384 pixels on a side.

The brute-force fix is to scale the encoder to 1024x1024. That works, but a ViT's self-attention cost grows quadratically with token count, and token count grows quadratically with side length. Going from 384 to 1024 is a 2.7x increase per side, roughly 7x the patches, and the attention term blows up faster still. You pay that cost on *every* image, including the ones where the semantics encoder at 384 would have sufficed.

DeepSeek-VL's answer is to run two encoders with complementary jobs, sized for their jobs:

- **SigLIP-L** at 384x384 — the *semantic* path. It answers "what is in this image."
- **SAM-B** (the image encoder from Segment Anything) at 1024x1024 — the *detail* path. SAM was trained for dense segmentation, so its features carry the low-level, high-frequency structure that survives at high resolution. It answers "where are the edges, the strokes, the small text."

### How the two resolutions get fused

The trick is that both paths have to end up as the same shape so they can be combined token-for-token. The concrete reshaping the paper describes for the SAM-B path goes through four steps. The SAM-B encoder emits a 64 x 64 x 256 feature map for a 1024x1024 input. That map is interpolated up to 96 x 96 x 256, then passed through two stride-2 convolutions that reduce it to 24 x 24 x 1024, and finally reshaped to 576 x 1024 detail tokens. The two convolutions are the workhorses: each halves the spatial grid while doubling the channel depth, so the high-frequency structure that SAM captured at 1024px is repackaged into the same 576-token grid the semantic path uses.

The SigLIP-L path produces 576 x 1024 semantic tokens directly. The two are then concatenated along the feature dimension, giving 576 tokens of dimension 2048 — half semantic, half detail — which is exactly the fusion shown in the mental-model figure above. A vision-language adaptor (a small MLP) projects those 576x2048 tokens into the language model's embedding dimension, and the decoder consumes them as if they were 576 extra "word" tokens prepended to the prompt.

```python
import torch

"""DeepSeek-VL fuses two encoder paths: both run, then their 576-token
grids are concatenated along the feature dimension before the adaptor."""

def fuse_vision_features(siglip_tokens, sam_tokens):
    # siglip_tokens: [B, 576, 1024]  (semantic, from 384x384)
    # sam_tokens:    [B, 576, 1024]  (detail,   from 1024x1024 after conv downsample)
    assert siglip_tokens.shape[1] == sam_tokens.shape[1] == 576
    fused = torch.cat([siglip_tokens, sam_tokens], dim=-1)  # [B, 576, 2048]
    return fused  # adaptor MLP projects 2048 -> d_model of the decoder

def vision_token_budget(num_images, tokens_per_image=576):
    # 576 visual tokens per image regardless of which encoder path —
    # the budget the LLM context must absorb.
    return num_images * tokens_per_image
```

The token budget is the part worth internalizing. Whatever clever fusion happens upstream, the decoder sees a fixed 576 visual tokens per image. That is a deliberate efficiency choice: the high-resolution SAM path gives you the *quality* of 1024x1024 without paying the *context cost* of 1024x1024, because the detail is compressed back into the same 576-token grid before it ever touches the LLM. You get high-res perception at a low-res token bill.

### Second-order optimization: the detail path is not free, so spend it deliberately

There is a non-obvious cost here. Running SAM-B at 1024x1024 is the most expensive single operation in the VL1 vision front end, and it runs on every image whether or not the image has any fine detail to read. A photo of a beach gains almost nothing from the detail path; a dense spreadsheet gains everything. VL1 does not adapt — it always pays. That uniform cost is precisely the inefficiency VL2 attacks with dynamic tiling, where the number of high-resolution tiles scales with the image's aspect ratio and content rather than being fixed. Keep that contrast in mind; it is the through-line of this whole post.

A second subtlety: two encoders means two backbones to host, load into memory, and keep version-synchronized. In a serving fleet that is real operational weight — two sets of weights, two warm-up paths, two failure modes. The detail path buys quality; it also buys complexity. The VL2 team clearly decided the trade was not worth it at scale, which brings us to the first big architectural pivot.

### Why SAM specifically, and not a second CLIP

It is worth pausing on *why* the detail encoder is SAM-B rather than, say, a second SigLIP at higher resolution. The two encoders are not redundant copies at different resolutions — they carry genuinely different *kinds* of features, and the difference traces back to how each was trained.

SigLIP is trained with a contrastive image-text objective. Its gradients reward features that *discriminate one caption from another*, which pushes the representation toward global, semantic, object-level abstractions. That is exactly what you want for "what is in this image," and exactly the wrong inductive bias for "where is the boundary of this character stroke." A contrastive encoder has no reason to preserve high-frequency spatial detail that does not help tell captions apart.

SAM, by contrast, is trained for promptable segmentation. Its objective rewards features that *localize boundaries precisely* — every pixel has to be assignable to a mask. That objective forces the encoder to retain dense, high-frequency, spatially-precise structure. When you run SAM-B at 1024x1024, those features carry the edges, strokes, and small-text structure that a contrastive encoder discards. Stacking a second SigLIP at high resolution would give you more semantic tokens, not more *detail* tokens; the inductive biases would overlap. The hybrid works precisely because the two objectives are complementary: one encoder is biased toward meaning, the other toward boundaries, and a document needs both.

This also explains why VL2 could afford to drop SAM. SAM's value is dense boundary structure at high resolution. Dynamic tiling recovers high resolution a different way — by giving each region of the image its own native 384px crop through a strong SigLIP-SO400M encoder — and SigLIP-SO400M is a substantially stronger semantic encoder than the SigLIP-L used in VL1. The bet is that *more native-resolution semantic coverage* beats *one global semantic frame plus a dedicated boundary encoder* for the document-and-chart workloads VL2 targets. For workloads dominated by boundary precision rather than reading, that bet would flip — which is the honest caveat we return to throughout.

## 2. From hybrid to single-encoder: what VL2 changed and why

**Senior rule of thumb: when an architecture has two of something, ask whether the second one is buying capability or buying a workaround for a fixed input size. If it is the latter, a smarter input pipeline can delete it.**

DeepSeek-VL2's vision front end is, on paper, simpler than VL1's: it uses **SigLIP-SO400M-384 only**. The SAM-B detail path is gone. That should set off an alarm — if SAM-B was solving the high-resolution problem, how does VL2 read fine print without it?

![VL2 drops the second encoder and recovers resolution by tiling one ViT instead.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-2.webp)

The before-and-after above is the answer in one picture. VL2 recovers resolution not by adding a second encoder but by changing how the image enters the *single* encoder. Instead of squashing a high-resolution image down to one 384x384 frame and relying on a detail encoder to claw back what was lost, VL2 cuts the image into multiple 384x384 tiles and runs each through the same ViT. Every tile is full-resolution as far as the encoder is concerned, because each tile *is* a native-resolution 384x384 crop. The resolution that VL1 bought with a second encoder, VL2 buys with a smarter tiling of the first.

The operational payoff is exactly the four-row story in the figure. One encoder instead of two: half the vision weights to host. Aspect-ratio-aware tiling instead of a fixed 1024x1024 frame: a tall receipt and a wide spreadsheet each get a tiling that fits their shape. Up to nine local tiles plus a thumbnail instead of a single pooled detail grid: the high-resolution budget scales with the image. And one backbone whose weights are reused across all tiles: the same parameters do all the work, so there is no second model to keep in sync.

### What you give up, honestly

This is not a free lunch. The SAM path carried genuinely different features — segmentation-trained, dense, low-level — that a contrastive SigLIP encoder does not produce in the same form. VL2's bet is that *enough native-resolution tiles* through a strong SigLIP-SO400M encoder recovers more usable detail than one global frame plus a SAM detail grid, and the document-understanding benchmark numbers back that bet up. But if your workload were, say, fine-grained medical segmentation rather than reading documents and charts, the VL1-style dedicated detail encoder might still be the better tool. Architecture choices are workload choices; VL2 is optimized for the text-in-images, charts, tables, and OCR workloads that dominate real VLM usage.

| Dimension | DeepSeek-VL (VL1) | DeepSeek-VL2 |
| --- | --- | --- |
| Vision encoders | SigLIP-L + SAM-B (hybrid) | SigLIP-SO400M-384 only |
| High-res strategy | SAM-B at 1024x1024, pooled | Dynamic tiling, up to 9 tiles |
| Visual tokens / image | 576 (fixed) | Variable: ~196 per tile after pixel shuffle |
| Aspect-ratio handling | Fixed square frame | Per-image tile grid by aspect ratio |
| Backbones to host | 2 | 1 (shared across tiles) |
| Language decoder | Dense DeepSeek LLM | Sparse DeepSeekMoE + MLA |

That last row is a whole second story — the language half — and we will get to it. First, the tiling, because it is the cleverest part of the vision pipeline.

## 3. Dynamic tiling: high resolution from one ViT

**Senior rule of thumb: a 384px encoder is not a low-resolution encoder if you let the image decide how many 384px windows it deserves.**

Dynamic tiling is the heart of VL2's vision side. The idea: rather than resizing every image to a fixed square, choose a tile grid that matches the image's aspect ratio, slice the image into that many 384x384 local tiles, *also* keep one 384x384 global thumbnail of the whole image, and run all of them through the same ViT.

![A wide document picks a 3x3 tile grid; the thumbnail keeps the global view.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-3.webp)

The grid above shows a 3x3 case: nine local tiles, each a native 384x384 crop, plus a global thumbnail on the right that preserves the whole-image context, all forwarded through one shared SigLIP-SO400M-384 ViT. The local tiles give you the fine detail; the thumbnail gives you the global layout so the model does not lose the forest for the trees. Both kinds of token go to the decoder together.

### The tile-grid selection algorithm

The selection is the part worth coding out. VL2 enumerates candidate tilings as $m_i \times n_i$ grids where $m_i, n_i \in \{1, 2, \dots, 9\}$ and the product is capped: $m_i \cdot n_i \le 9$. For each candidate, it computes how much padding (wasted area) results when the image is resized to fit that grid's aspect ratio, and picks the grid that minimizes padding — that is, the tiling whose shape best matches the image's shape.

```python
import math

CANDIDATES = [(m, n) for m in range(1, 10) for n in range(1, 10) if m * n <= 9]

def choose_tiling(img_w, img_h, tile=384):
    """Pick the (cols, rows) tiling whose aspect ratio wastes the least padding.
    Mirrors VL2's dynamic tiling: up to 9 local tiles, each 384x384."""
    ar = img_w / img_h
    best, best_waste = (1, 1), float("inf")
    for cols, rows in CANDIDATES:
        grid_w, grid_h = cols * tile, rows * tile
        # scale the image to fit inside the grid without distortion
        scale = min(grid_w / img_w, grid_h / img_h)
        used = (img_w * scale) * (img_h * scale)
        waste = grid_w * grid_h - used          # padded (wasted) area
        if waste < best_waste:
            best, best_waste = (cols, rows), waste
    return best  # (cols, rows); total local tiles = cols * rows, plus 1 global

def visual_tokens(cols, rows, tokens_per_tile=196):
    local = cols * rows * tokens_per_tile
    glob = tokens_per_tile                      # one global thumbnail tile
    return local + glob

for (w, h) in [(1920, 1080), (768, 1024), (3000, 600), (512, 512)]:
    c, r = choose_tiling(w, h)
    print(f"{w}x{h}: tiling {c}x{r} -> {c*r} local tiles, "
          f"{visual_tokens(c, r)} visual tokens")
```

Run that and a wide 3000x600 banner picks a wide tiling, a tall 768x1024 page picks a tall one, and a small 512x512 image may pick a single tile plus the thumbnail. The token budget is *adaptive* — a simple square logo costs a couple hundred visual tokens, while a dense full-page document can cost close to two thousand. That is the efficiency win in one sentence: **you only pay context tokens proportional to how much resolution the image actually needs.** VL1's fixed 576 tokens could not do that.

### Pixel shuffle: 729 tokens become 196

A raw 384x384 image through a ViT with a 14px patch (SigLIP-SO400M's configuration) yields a 27x27 grid, which is 729 patch tokens per tile. Nine tiles plus a thumbnail at 729 each would be 7,290 visual tokens — far too many to prepend to every prompt. VL2 applies a **2x2 pixel shuffle** (a space-to-depth operation) that folds each 2x2 block of spatial tokens into one token with 4x the channels, cutting 729 down to roughly 196 tokens per tile.

```python
import torch

def pixel_shuffle_2x2(x):
    # x: [B, H, W, C] spatial visual tokens, e.g. [B, 27, 27, C]
    # Pad to even, then fold each 2x2 spatial block into the channel dim.
    B, H, W, C = x.shape
    if H % 2 or W % 2:
        x = torch.nn.functional.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        B, H, W, C = x.shape
    x = x.reshape(B, H // 2, 2, W // 2, 2, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, (H // 2) * (W // 2), 4 * C)
    return x  # ~196 tokens for a 27x27 input -> 14x14 = 196

print(pixel_shuffle_2x2(torch.randn(1, 28, 28, 1152)).shape)  # [1, 196, 4608]
```

The shuffle trades spatial resolution in the token grid for channel depth, which the downstream MLP connector can absorb. It is the single most important reason the tiling stays affordable: without it, nine high-resolution tiles would obliterate the context budget. With it, a nine-tile image lands around 1,960 visual tokens — large, but tractable.

### Special tokens: telling the LLM where the tiles came from

Here is a problem tiling creates that a single frame does not: once you flatten nine tiles plus a thumbnail into a 1D token stream, the decoder has no idea which token came from which spatial position. Tile 7 (bottom-left) and tile 3 (top-right) look identical in sequence order. VL2 fixes this with two special tokens that encode the 2D layout into the 1D stream:

- `<tile_newline>` — marks the end of a row of tiles, so the decoder can reconstruct that tiles wrapped to a new row.
- `<view_separator>` — separates the local-tile view from the global-thumbnail view, so the decoder knows where the detailed tiles stop and the whole-image context begins.

These are cheap — a handful of tokens — but they are load-bearing. They are the difference between the model knowing "this number is in the bottom-right cell of the table" and the model seeing a bag of disconnected patches. Spatial structure is information; the special tokens are how that information survives the flattening.

> The cleverness of dynamic tiling is not the tiling itself — anyone can crop an image into a grid. It is that the tile *count* adapts to the image, the *cost* (visual tokens) adapts with it, the layout is preserved with two cheap markers, and every tile reuses one set of ViT weights. One encoder, variable resolution, layout-aware. That is the whole trick.

### Why keep a global thumbnail at all

A reasonable objection: if the local tiles already cover the whole image at high resolution, why spend tokens on a redundant low-resolution thumbnail of the same content? The answer is *receptive field*. Each local tile is encoded in isolation — the ViT attends within a tile but not across tiles. So a model looking only at local tiles sees nine high-resolution patches with no information about how they relate spatially or what the overall image is. A table spanning multiple tiles, a figure whose caption is three tiles away, a UI whose layout matters globally — none of that survives if every tile is an island.

The global thumbnail is the cross-tile context channel. It is one 384px encoding of the *entire* image, so it carries the global layout, the overall composition, and the relationships between regions that the isolated local tiles cannot. The `<view_separator>` token tells the decoder where the detailed-but-local view ends and the coarse-but-global view begins, and the decoder learns to fuse them: read the fine detail from the local tiles, but anchor it in the global structure from the thumbnail. It is the cheapest possible fix for the receptive-field problem tiling introduces — one extra tile's worth of tokens to restore the global view.

### A worked tiling example

Consider a 2480x3508 scan of an A4 page (a common 300-DPI document scan, aspect ratio ~0.71, portrait). The tiling selector enumerates the candidate grids with at most nine tiles and computes padding waste for each. A portrait page wants a tall grid — something like 2 columns by 4 rows (8 tiles) fits the 0.71 aspect ratio with minimal padding, while a wide grid like 4x2 would waste enormous area padding the sides. The selector picks 2x4: eight local tiles plus one thumbnail, nine tiles total. After pixel shuffle that is 9 x 196 = 1,764 visual tokens. Now the bottom of the page — which a single squashed 384px frame would have rendered illegible — gets two dedicated native-resolution tiles, and the small print at the page foot becomes readable. The cost (1,764 tokens) is the price of that legibility, and it is paid only for documents that actually need it. A simple square logo through the same selector picks 1x1 plus a thumbnail: ~392 tokens. Same model, an order of magnitude apart in cost, each appropriate to its input.

## 4. The VL2 vision path, end to end

**Senior rule of thumb: the connector between vision and language is where most VLMs are silently bottlenecked. Keep it small, keep it learnable, and make the token stream it produces self-describing.**

Let us assemble the full vision path so the pieces have a place to live. An image comes in at any aspect ratio. Dynamic tiling chooses a grid and produces up to nine local 384x384 tiles plus one global thumbnail. Every tile — local and global alike — goes through the single shared SigLIP-SO400M-384 ViT. Pixel shuffle compresses each tile from 729 to ~196 tokens. The special tokens `<tile_newline>` and `<view_separator>` are interleaved to encode the 2D layout. Finally a 2-layer MLP connector projects the visual tokens into the decoder's embedding space, and the whole stream flows into the MoE/MLA language model.

![From raw pixels to MoE decoder, the image becomes a flat token stream with layout markers.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-4.webp)

The pipeline above is the canonical reference for the rest of this section. Note the order: tiling happens *before* the ViT (so the encoder always sees native-resolution crops), pixel shuffle happens *after* the ViT (so we compress encoded features, not raw pixels), and the layout tokens are inserted *after* shuffle (so they index the final, compressed token grid). Getting that order wrong is a classic implementation bug — shuffle before encode and you destroy the patch structure the ViT expects.

### The connector is deliberately boring

The vision-language connector in VL2 is a 2-layer MLP. Not a Q-Former, not a Perceiver resampler, not a cross-attention stack. There is a real design lesson here. The fancy connectors were invented to *reduce* the visual token count, because early VLMs could not afford many visual tokens. VL2 reduces token count a different way — with pixel shuffle and adaptive tiling — which frees the connector to be a simple, learnable projection that does one job: map the visual feature space onto the language embedding space. Fewer moving parts, fewer training instabilities, and the heavy lifting of token reduction happens in a place (pixel shuffle) where it is a fixed, deterministic operation rather than a learned bottleneck that can collapse.

```python
import torch.nn as nn

class VL2Connector(nn.Module):
    """The boring-on-purpose 2-layer MLP that bridges ViT -> decoder."""
    def __init__(self, vision_dim=4608, llm_dim=2048):  # 4608 = 1152 * 4 after shuffle
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, visual_tokens):
        # visual_tokens: [B, num_visual, vision_dim] after pixel shuffle
        return self.proj(visual_tokens)  # [B, num_visual, llm_dim]
```

### Second-order optimization: the visual tokens are still tokens, and they share the KV cache

Once the connector emits its projection, those visual tokens are indistinguishable from text tokens to the decoder. They occupy positions in the sequence, they get attended to, and crucially **they take up space in the KV cache.** A nine-tile document at ~1,960 visual tokens is ~1,960 KV-cache slots before the user has typed a single word. Multiply by a batch of concurrent document-understanding requests and the KV cache becomes the binding constraint on throughput. This is exactly why the language-side choice of Multi-head Latent Attention matters so much for a VLM specifically — VLMs run *long* by default because images are token-expensive. We will quantify that in the language-backbone section, but plant the flag now: the vision pipeline's token economy and the decoder's KV-cache economy are the same problem viewed from two ends.

## 5. Managing vision-language modality competition

**Senior rule of thumb: a multimodal model trained on images and text at the same time is two students fighting over one brain. Whoever gets more study time wins, and by default vision crowds out language.**

This is the section that justifies the whole "language first" framing, and it is where DeepSeek-VL's training methodology earns its keep. The problem the team identified is *modality competition*: when you take a pretrained language model and start training it jointly on a heavy diet of image-text data, the language ability degrades. Not catastrophically in one step, but measurably — text benchmarks slide as the optimizer reshapes the shared weights toward the vision objective.

![Pour in images too fast and text skills bleed away; a warm-up and a held ratio keep both.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-5.webp)

The before-and-after above contrasts the two regimes. On the left, naive joint training: feed the model image data at 100% and language data at 0%, and the LLM weights get pulled hard toward the vision objective. The paper is blunt that a multimodal-to-language ratio of 100%-to-0% causes a severe decline in language metrics. The model learns to see and forgets how to reason. On the right, the DeepSeek-VL recipe: a modality warm-up that ramps the image share *gradually* from a text-heavy starting point, and a held mix of roughly 70% language to 30% multimodal data through joint pretraining. The result is a VLM that gains vision without surrendering its language ability — the held ratio is what keeps the second student in the room.

### The numbers that make the case

The cleanest evidence is the language-only benchmark holding steady. On HellaSwag, DeepSeek-VL-7B scores **68.4** versus the base language model's **68.5** — a rounding-error gap. That is the entire argument in two numbers: a VLM that, after absorbing a full vision training curriculum, has essentially not regressed on a hard commonsense-reasoning benchmark. Compare that to the failure mode the figure warns about, where text benchmarks visibly drop, and you see why the ratio is not a minor hyperparameter — it is the difference between a useful model and a regressed one.

```python
def modality_ratio_schedule(step, total_steps, warmup_frac=0.1,
                            start_img=0.0, hold_img=0.30):
    """Ramp the image-data fraction during warm-up, then hold ~30% image / 70% text.
    Reflects DeepSeek-VL's 'warm up then hold the ratio' strategy."""
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        # linear ramp from a text-heavy start up to the held image fraction
        frac = step / max(1, warmup_steps)
        img = start_img + frac * (hold_img - start_img)
    else:
        img = hold_img
    return {"image": round(img, 3), "language": round(1 - img, 3)}

for s in [0, 500, 1000, 5000, 50000]:
    print(s, modality_ratio_schedule(s, total_steps=50000))
```

The schedule above captures the shape of the strategy: start almost pure-text, ramp the image fraction up during a warm-up window, then hold a stable ~30/70 image/language mix for the rest of joint pretraining. The warm-up matters because a cold start straight into heavy image data is the most destabilizing moment — the model has not yet learned to align visual tokens to its language space, so the gradients are noisiest exactly when the language weights are most exposed.

### The three-stage structure that operationalizes "language first"

The ratio management is one half; *which weights are trainable when* is the other. DeepSeek-VL trains in stages, and the staging is itself a competition-management tool:

1. **Stage 1 — train the VL adaptor only.** The vision encoder and the language model are both frozen; only the adaptor (the projection between them) trains, on roughly 3.75M image-text pairs. The language model cannot regress because its weights do not move. This stage teaches the adaptor to speak the LLM's language without risking the LLM itself.
2. **Stage 2 — joint VL pretraining.** Now the language model unfreezes and trains jointly with the adaptor, under the carefully held ~70/30 language/multimodal ratio. This is where the competition is real and where the ratio earns its keep.
3. **Stage 3 — supervised fine-tuning.** Instruction tuning on a dataset built from a *use-case taxonomy of real user scenarios* — the model learns to follow instructions and hold a conversation, with the full system (except the most expensive frozen detail components) tuned for dialogue.

Notice the logic: freeze first, unfreeze under a controlled ratio, then specialize. Each stage exposes the language weights to a bit more risk, but only after the prior stage has reduced the variance the next stage will introduce. It is risk management applied to gradients.

### Second-order optimization: data diversity is part of competition management

There is a subtle point hiding in the pretraining data mix. DeepSeek-VL's pretraining corpus deliberately includes web screenshots, PDFs, OCR, charts, and knowledge-based content. That diversity is not just for capability coverage — it is also for *modality balance*. Document and OCR data sits at the boundary of vision and language: reading text out of an image exercises both modalities at once, which gives the model gradient signal that reinforces, rather than competes with, language ability. Curating toward text-rich images is a way to make the 30% multimodal slice pull in the same direction as the 70% language slice instead of against it. Competition management is not only a ratio knob; it is also a data-curation strategy.

## 6. The language backbone: DeepSeekMoE plus Multi-head Latent Attention

**Senior rule of thumb: in an efficient VLM the question is never "how big is the model," it is "how much of the model does each token actually wake up, and how much KV cache does each token cost." Total parameters answer neither.**

VL2's biggest departure from VL1 is not the vision encoder — it is the language decoder. VL1 used a dense DeepSeek LLM. VL2 uses **DeepSeekMoE** (a sparse Mixture-of-Experts decoder) combined with **Multi-head Latent Attention (MLA)**. These are two independent efficiency mechanisms attacking two different costs, and a VLM is the workload where both pay off hardest.

![Sparsity is in the language decoder; the vision front end stays a single dense ViT.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-6.webp)

The stack above is the figure to commit to memory, because it answers the single most-misreported fact about VL2: **the Mixture-of-Experts lives in the language decoder, not in the vision encoder.** The vision path — dynamic tiling, the shared SigLIP ViT, the 2-layer MLP connector — is entirely dense. Sparsity enters only when visual tokens and text tokens reach the decoder, where MLA compresses the KV cache and the DeepSeekMoE feed-forward layers route each token to a small subset of experts. If you take one correction away from this post, make it this one. The ViT is dense. The experts are downstream.

### MLA: compressing the KV cache that images inflate

We flagged earlier that VLMs run long because images are token-expensive. A nine-tile document is ~1,960 visual tokens before any text. In a standard multi-head attention decoder, every one of those tokens stores a key and a value vector *per layer per head* in the KV cache. That cache is what makes long-context inference memory-bound, and a VLM is long-context by construction.

Multi-head Latent Attention attacks this directly: instead of caching full-size keys and values, MLA caches a *compressed latent vector* per token and reconstructs the per-head keys and values on the fly. The KV cache shrinks by a large factor, which means more concurrent requests fit in the same GPU memory, which means higher throughput. For a deep treatment of how MLA's low-rank compression actually works, see [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla); for the broader DeepSeek-V3 lineage these techniques come from, see [DeepSeek-V3: FP8, MTP, and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing).

```python
def kv_cache_bytes(seq_len, n_layers, n_heads, head_dim, dtype_bytes=2):
    """Standard MHA: cache full K and V per token, per layer, per head."""
    return 2 * seq_len * n_layers * n_heads * head_dim * dtype_bytes

def kv_cache_bytes_mla(seq_len, n_layers, latent_dim, dtype_bytes=2):
    """MLA: cache one compressed latent per token, per layer."""
    return seq_len * n_layers * latent_dim * dtype_bytes

seq = 1960  # a 9-tile document carries ~1960 visual tokens before any text
mha = kv_cache_bytes(seq, n_layers=30, n_heads=32, head_dim=128)
mla = kv_cache_bytes_mla(seq, n_layers=30, latent_dim=512)
print(f"MHA KV cache: {mha/1e6:.1f} MB")
print(f"MLA KV cache: {mla/1e6:.1f} MB  ({mha/mla:.1f}x smaller)")
```

The illustrative numbers make the point: when the sequence is already ~2,000 tokens before the user speaks, an order-of-magnitude smaller KV cache is the difference between serving four concurrent document requests and serving forty. MLA is not a generic nicety here — it is specifically the lever that makes a *high-resolution, many-visual-token* VLM affordable to serve.

### DeepSeekMoE: sparse capacity without dense cost

The second mechanism is the Mixture-of-Experts feed-forward network. A dense FFN runs every token through the same large weight matrix. A DeepSeekMoE FFN keeps many smaller expert networks and routes each token to only a few of them, plus a couple of always-on shared experts. The model can hold a large *total* parameter count — lots of experts, lots of specialized capacity — while each token only activates a small *fraction* of those parameters. This is the fine-grained, shared-expert design that the DeepSeekMoE lineage is known for, covered in depth in [the DeepSeekMoE lineage](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) and in the broader [MoE LLM architecture guide](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies).

The reason this pairs so naturally with a VLM: vision-language tasks are *heterogeneous*. OCR, chart reading, natural-image description, grounding, and dialogue exercise quite different sub-skills. A mixture of experts can specialize different experts to different sub-skills while still presenting a single model, and only the relevant experts wake up per token. You get the capacity of a large model with the inference cost of a small one — which is the entire pitch of the VL2 variants we will quantify next.

### What the two efficiency mechanisms cost, separately

It helps to keep MLA and MoE conceptually distinct because they attack different resources and can be adopted independently. MLA attacks *memory bandwidth and KV-cache capacity* — the bottleneck during the decode phase of long-context inference. MoE attacks *compute (FLOPs) per token* — the bottleneck during both prefill and decode for a dense model. A VLM with a long visual prefix is bottlenecked on *both*: the prefix is huge (memory pressure on the KV cache, which MLA relieves) and every token of it must be processed (compute pressure, which MoE relieves by activating only a few experts).

You could ship a VLM with MLA and a dense FFN: you would get the KV-cache savings without the sparse-capacity benefit, a perfectly reasonable choice if your team cannot operate MoE training. You could ship a VLM with MoE and standard attention: you would get sparse capacity but the visual prefix would crush your KV cache, undoing much of the throughput win on long inputs. VL2 takes both because the workload pressures both, but the modularity is real — and if you are building your own VLM and can only afford one, MLA is the higher-leverage choice for token-heavy visual inputs, because the KV cache is what cliffs first.

### Benchmark posture: state-of-the-art per activated parameter

The headline result for VL2 is not raw top-of-leaderboard numbers — it is *competitive or state-of-the-art performance with similar or fewer activated parameters* than dense and MoE peers. That framing matters. A model that matches a 7B-dense competitor while activating 4.5B is strictly more efficient at inference, and that is the axis VL2 optimizes.

| Benchmark | What it measures | DeepSeek-VL2 result | Reading |
| --- | --- | --- | --- |
| DocVQA | Document question answering | ~93.3 | Strong; the document workload tiling targets |
| ChartQA | Chart reading and reasoning | ~86.0 | Competitive; dynamic tiles read axes |
| OCRBench | OCR across many image types | ~811 | Strong; native-resolution tiles help |
| TextVQA | Reading text in natural images | ~84.2 | Above comparable models |
| RefCOCO (val) | Visual grounding accuracy | ~95.1 | Best among open-source VLMs at the time |

The pattern across the table is consistent: VL2's strengths cluster in *text-in-image* and *grounding* tasks, which is exactly where dynamic tiling's native-resolution coverage and the grounded fine-tuning stage pay off. These are not cherry-picked wins; they are the predictable consequence of the architecture's bias toward reading and localizing fine detail. A model whose vision pipeline is optimized to deliver legible high-resolution tiles to a language-strong decoder should, and does, win at reading documents.

## 7. The three variants: total versus activated parameters

**Senior rule of thumb: report both numbers or neither. A MoE model has a "size on the hard drive" and a "size in the FLOP budget," and conflating them is how people end up either over-provisioning hardware or under-estimating quality.**

This is the most-corrected fact in every secondary write-up of VL2, so let us be precise. VL2 ships in three variants, and each has a *total* parameter count and an *activated* parameter count:

![The big number is capacity on disk; the small number is what each token actually pays for.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-8.webp)

The matrix above lays it out: **Tiny is 3B total / 1.0B activated, Small is 16B total / 2.8B activated, and Base is 27B total / 4.5B activated.** Read those pairs carefully. The 3B, 16B, and 27B figures are the *total* parameters — the full capacity of all experts, what you load into memory. The 1.0B, 2.8B, and 4.5B figures are the *activated* parameters — what a single forward pass through the MoE decoder actually computes per token. The big number sets your memory footprint; the small number sets your compute cost and roughly tracks the per-token latency.

| Variant | Total params | Activated / token | Activation ratio | Backbone |
| --- | --- | --- | --- | --- |
| VL2-Tiny | 3B | 1.0B | ~33% | DeepSeekMoE + MLA |
| VL2-Small | 16B | 2.8B | ~18% | DeepSeekMoE + MLA |
| VL2-Base | 27B | 4.5B | ~17% | DeepSeekMoE + MLA |

Two things jump out of the activation-ratio column. First, the larger variants are *sparser* — Base activates only about 17% of its weights per token, versus Tiny's ~33%. That is the MoE scaling story: as you add experts, you add total capacity faster than you add activated cost, so the activation ratio falls. Second, a 27B-total model that activates 4.5B is, computationally, closer to a 4.5B dense model than to a 27B dense model — while carrying far more knowledge than a 4.5B dense model could. That is the whole reason MoE exists, and VL2 is a clean demonstration of it in a multimodal setting.

```python
VARIANTS = {
    "VL2-Tiny":  {"total_b": 3.0,  "active_b": 1.0},
    "VL2-Small": {"total_b": 16.0, "active_b": 2.8},
    "VL2-Base":  {"total_b": 27.0, "active_b": 4.5},
}

def provisioning(v, bytes_per_param=2):
    total_mem = v["total_b"] * 1e9 * bytes_per_param / 1e9      # GB to host weights
    compute_proxy = v["active_b"]                               # ~ per-token FLOP scale
    sparsity = 1 - v["active_b"] / v["total_b"]
    return total_mem, compute_proxy, sparsity

for name, v in VARIANTS.items():
    mem, comp, sp = provisioning(v)
    print(f"{name:10s}  host ~{mem:4.0f} GB (fp16)  "
          f"compute ~{comp:.1f}B-equiv/token  sparsity {sp:.0%}")
```

The provisioning snippet is the practical takeaway. When you size hardware for VL2-Base, you provision memory for 27B of weights but you budget *throughput* against ~4.5B of activated compute. Provision memory off the activated number and you will OOM; budget latency off the total number and you will massively over-estimate cost and under-utilize your GPUs. Report both numbers or neither.

### Second-order optimization: the activated number is per-token, and visual tokens are tokens

A trap worth naming: the activated-parameter cost is paid *per token*, and we established that a high-resolution image is ~2,000 visual tokens. So a single document-understanding request pays the ~4.5B activated cost roughly 2,000 times for the image alone, before the prompt and the generated answer. This is where the vision-side and language-side efficiency stories converge into one bill: dynamic tiling controls how many visual tokens you generate, pixel shuffle compresses them, MLA shrinks their KV footprint, and the MoE keeps the per-token compute low. Pull any one of those levers out and the whole thing gets expensive. They were co-designed, and they should be reasoned about together.

### Reading the activation ratio as a capacity dial

There is a way to think about the three variants that makes the design choices legible. The *activated* parameter count sets a floor on quality — a model cannot reason better than its per-token compute allows, roughly speaking — while the *total* parameter count sets a ceiling on how much specialized knowledge the experts can hold. The gap between them is the MoE's leverage.

VL2-Tiny at 3B/1.0B has a 3x leverage: it holds three times the parameters it activates. VL2-Base at 27B/4.5B has a 6x leverage. As you climb the variants, you are not just making the model bigger — you are making it *sparser*, buying more total knowledge per unit of activated compute. This is why a VL2-Base answer can be both fast (4.5B-activated latency) and knowledgeable (27B-total capacity) in a way a dense model of either size cannot match: a dense 4.5B lacks the capacity, and a dense 27B lacks the speed.

The practical corollary for model selection: pick your variant by your *latency budget first*, then take whatever capacity the activation ratio gives you for free. If you can afford ~4.5B-activated latency, take Base and enjoy 27B of total capacity. If you are latency-bound to ~1B-activated, Tiny is your ceiling and you accept its smaller total capacity. Do not pick by total size and hope the latency works out — the activated number is what you actually wait on.

```python
def select_variant(latency_budget_activated_b, variants=VARIANTS):
    # Pick the largest-total variant whose activated cost fits the latency budget.
    affordable = {k: v for k, v in variants.items()
                  if v["active_b"] <= latency_budget_activated_b}
    if not affordable:
        return None  # even Tiny is too slow; use a dense small model
    # among affordable, take the one with the most total capacity
    return max(affordable, key=lambda k: affordable[k]["total_b"])

for budget in [0.9, 1.5, 3.0, 5.0]:
    print(f"latency budget {budget}B-activated -> {select_variant(budget)}")
```

## 8. The VL2 training recipe in three stages

**Senior rule of thumb: freeze what you cannot afford to break, unfreeze under a controlled data diet, and specialize last. The order is the safety mechanism.**

VL2's training mirrors VL1's three-stage philosophy but at a different scale and with the grounding capabilities added.

![Alignment warms the connector, pretraining unfreezes everything, SFT teaches instructions and grounding.](/imgs/blogs/deepseek-vl-vl2-dynamic-tiling-moe-7.webp)

The timeline above lays out the three stages in order. Stage 1 is *vision-language alignment*: the language model and the vision encoder are frozen, and only the MLP connector trains, on roughly 1.2M caption and conversation samples. The job of this stage is narrow — teach the connector to project visual features into the decoder's embedding space — and freezing everything else means the expensive backbones cannot regress while the connector finds its footing. Stage 2 is *joint VL pretraining*: all weights unfreeze and the model trains on approximately **800B image-text tokens**, with the modality ratio managed exactly as the competition-management section described. Stage 3 is *supervised fine-tuning*: instruction following, multi-turn dialogue, and — new in VL2 — visual grounding and grounded conversation, where the model learns to point at specific image regions and refer to them in language.

### Visual grounding is a first-class capability, not an afterthought

VL2 explicitly trains visual grounding and grounded conversation. Grounding means the model can take a phrase like "the red car on the left" and produce the bounding box for it, and conversely take a region and describe what is in it. Grounded conversation extends that into multi-turn dialogue where references resolve to image regions across turns. This is a meaningful step up from caption-and-VQA-only VLMs: it makes the model useful for tasks where *which* object matters, not just *whether* an object is present — UI automation, document field extraction, robotics perception. It is also a capability that benefits directly from dynamic tiling, because grounding fine detail (a single form field, a small UI button) requires the high-resolution tiles that tiling provides.

```python
grounded_example = {
    # A grounded response emits region references inline with text;
    # this is the shape supervised fine-tuning (stage 3) teaches.
    "image": "form.png",
    "prompt": "What is the invoice total and where is it?",
    "response": (
        "The invoice total is $1,240.00 "
        "<box>[0.72, 0.81, 0.94, 0.86]</box>, "
        "in the bottom-right summary cell."
    ),
    # box coordinates are normalized [x0, y0, x1, y1]; grounding training
    # aligns them to the high-resolution tiles holding the referenced region.
}
```

### Second-order optimization: ~800B tokens is a vision-language budget, not a language budget

A number worth contextualizing: 800B image-text tokens is large for a VLM pretraining stage, but remember these are *multimodal* tokens, and a single image can be ~2,000 of them. So 800B tokens is not 800B captions — it is a smaller number of images and documents, each contributing a large block of visual tokens, plus interleaved text. The implication for anyone reproducing this: your data pipeline's bottleneck is image throughput and tiling, not text tokenization. The expensive part of feeding an 800B-token VL pretraining run is decoding, resizing, and tiling images fast enough to keep the GPUs fed. Budget your data infrastructure for pixels, not words.

## 9. Putting it together: the full VL2 forward pass

We have all the pieces; let us trace one request end to end to make the co-design concrete. A user uploads a 1920x1080 screenshot and asks "what does the error dialog say?"

1. **Tiling.** `choose_tiling(1920, 1080)` picks a wide grid — the aspect ratio is 16:9, so a wide tiling like 3x2 minimizes padding. That is six local 384x384 tiles plus one global thumbnail: seven tiles total.
2. **Shared ViT.** All seven tiles go through SigLIP-SO400M-384. Each produces a 27x27 = 729 patch-token grid.
3. **Pixel shuffle.** Each tile's 729 tokens collapse to ~196. Seven tiles give ~1,372 visual tokens.
4. **Layout tokens.** `<tile_newline>` marks the row break after the third tile; `<view_separator>` separates the six local tiles from the global thumbnail. The decoder now knows the spatial arrangement.
5. **Connector.** The 2-layer MLP projects all ~1,372 visual tokens into the decoder's embedding dimension.
6. **Decoder.** The visual tokens, the special tokens, and the text prompt enter the DeepSeekMoE decoder. MLA caches a compressed latent per token, so the ~1,372-token visual prefix costs a fraction of a standard KV cache. Each token routes through a few experts; only the activated parameters compute.
7. **Grounded answer.** The model reads the dialog text out of the high-resolution tiles and can optionally emit a `<box>` for where the dialog sits.

Every efficiency mechanism in this post shows up in that trace, and each one is load-bearing. Pull out tiling and you lose the resolution to read the dialog. Pull out pixel shuffle and the seven tiles blow the context budget. Pull out MLA and the visual prefix crushes the KV cache. Pull out the MoE and the per-token compute scales with the full 27B. They are one system.

### The cost accounting, made explicit

Let us total the bill for that 1920x1080 screenshot request on VL2-Base, because seeing the whole sum is what makes the co-design click. The visual prefix is ~1,372 tokens. Each token's feed-forward cost is governed by the ~4.5B activated parameters, not the 27B total — so the prefill compute is roughly *1,372 x 4.5B-equivalent*, not *1,372 x 27B-equivalent*. That is a 6x compute saving from the MoE alone, on the prefix that dominates the request. The KV cache for those 1,372 tokens stores one compressed MLA latent per token per layer rather than full keys and values — call it a large multiple smaller, which is what lets dozens of such requests share a GPU. And the prefix is ~1,372 tokens rather than the ~9,200 it would be without pixel shuffle, a ~5x token reduction before any compute or memory cost is even computed.

Compose those factors and the difference between a naive implementation and VL2's is not incremental — it is the product of a 5x token reduction, a 6x compute reduction per token, and a large KV-cache reduction. That multiplicative stacking is why VL2 can serve high-resolution document understanding at throughputs that a dense, full-token, full-attention VLM simply cannot reach on the same hardware. The architecture is not one clever idea; it is four efficiency levers that each multiply the others, deliberately co-designed so that the savings compound instead of merely adding.

### How the two papers relate as a lineage

Read together, VL1 and VL2 trace a clear engineering arc. VL1 established the *thesis* — language ability is the foundation, manage modality competition, fuse complementary vision features — and proved it with a hybrid encoder and a dense decoder. VL2 kept the thesis and rebuilt the *implementation* for scale: it replaced the two-encoder fusion with one tiled encoder (simpler to operate, adaptive in cost), and it replaced the dense decoder with a sparse MoE plus MLA backbone (cheaper to run at the long context lengths images force). Nothing in the thesis changed; the machinery underneath it got more efficient at every layer. That is the right way to read a model series — as a stable set of principles expressed through progressively better engineering — and it is why the modality-competition discipline from the first paper still governs how you should fine-tune the second.

## Case studies from production

The following are composite scenarios drawn from the failure modes and design decisions these two papers expose. They are written the way I would debrief them in a design review — symptom, wrong first guess, root cause, fix, lesson.

### 1. The VLM that got dumber after fine-tuning

**Symptom.** A team fine-tuned an open dense VLM on a large in-house image-text corpus and shipped it. Vision tasks improved; then support tickets rolled in that the model's *text* answers — code explanations, arithmetic, multi-step reasoning — had gotten noticeably worse than the base model they started from.

**Wrong first guess.** The team assumed catastrophic forgetting from too few training steps and *added* more fine-tuning, which made it worse.

**Root cause.** Classic modality competition. Their fine-tuning data was nearly 100% image-text, so the optimizer reshaped the shared weights toward the vision objective and away from language — exactly the regime DeepSeek-VL's left-hand figure warns about. More steps deepened the regression.

**Fix.** Re-mix the fine-tuning data to hold roughly 70% language / 30% multimodal, and add a warm-up that ramped the image fraction from a text-heavy start. Language benchmarks recovered to within a point of base; vision gains held.

**Lesson.** A VLM fine-tune is not a vision fine-tune. If you are not holding a language ratio, you are silently trading reasoning for perception. Measure text benchmarks *before and after*, every time.

### 2. The document model that could not find the bottom of the page

**Symptom.** A VL1-style model with a fixed 1024x1024 frame did well on screenshots but failed on tall, multi-column PDFs — it would read the top of the page and hallucinate the bottom.

**Wrong first guess.** The team blamed the OCR training data and added more documents.

**Root cause.** Aspect ratio. A tall A4 page squashed into a square 1024 frame loses vertical resolution; the bottom rows of text smeared into noise. No amount of training data fixes information that was destroyed at the input stage.

**Fix.** Move to VL2-style dynamic tiling. A tall page now picks a tall tile grid (for example 2x4), and the bottom of the page gets its own native-resolution tiles. The hallucination at the page bottom disappeared because the bottom was finally legible.

**Lesson.** Resolution problems that look like model problems are often input-pipeline problems. Check whether your aspect-ratio handling is throwing away the region that fails *before* you touch the model.

### 3. The throughput cliff at batch size eight

**Symptom.** A VL2-class document model served fine at low concurrency, then throughput collapsed past about eight concurrent requests — GPU memory spiked and the scheduler started evicting.

**Wrong first guess.** The team suspected the MoE router was load-imbalanced and spent days profiling expert utilization.

**Root cause.** The KV cache, not the experts. Each document request carried ~1,900 visual tokens, and the deployment had accidentally been built on a dense-attention variant rather than the MLA path. At ~1,900 tokens of visual prefix times eight requests, the KV cache exhausted memory long before compute did.

**Fix.** Ensure the MLA path was actually active so the visual prefix cached a compressed latent per token instead of full keys and values. Concurrency scaled past forty with the same memory.

**Lesson.** In a VLM, the visual prefix is usually the largest single consumer of KV cache. If throughput cliffs at low batch size, suspect the cache before the experts. MLA exists for exactly this workload.

### 4. The tiling that quadrupled the bill

**Symptom.** After enabling dynamic tiling, inference cost roughly quadrupled on a workload of mostly small, simple images (product thumbnails).

**Wrong first guess.** The team assumed the ViT had gotten more expensive.

**Root cause.** A misconfigured tiling policy was always selecting the maximum nine-tile grid regardless of image content, so a 200x200 thumbnail that needed one tile was being padded up and split into nine. Nine tiles times ~196 tokens is ~1,760 visual tokens for an image that warranted ~196.

**Fix.** Restore the padding-minimizing selection so simple images pick a 1x1 (or 1x2) grid. The token budget — and the bill — dropped back to baseline for the thumbnail workload while staying high only for genuinely dense images.

**Lesson.** Dynamic tiling's efficiency *depends* on the selection policy doing its job. A tiling that always picks the max grid is just a fixed high-resolution model with extra steps. Verify the per-image tile count on a sample of your real traffic.

### 5. The grounding boxes that drifted

**Symptom.** A grounded VL2-style model produced bounding boxes that were systematically offset — close, but shifted down and to the right — on tiled images.

**Wrong first guess.** A bug in the box-decoding head.

**Root cause.** The special layout tokens were being inserted at the wrong point in the pipeline (before pixel shuffle instead of after), so the model's internal mapping from token position to image coordinate was off by the shuffle's spatial folding. The grounding head had learned a consistent but *wrong* coordinate frame.

**Fix.** Insert `<tile_newline>` and `<view_separator>` after pixel shuffle so they index the final compressed token grid, matching the coordinate frame the grounding training assumed. The offset vanished.

**Lesson.** When tiling and grounding interact, the order of operations in the token pipeline is a correctness property, not a style choice. Layout tokens must index the same grid the coordinates are trained against.

### 6. The two-encoder model nobody could keep in sync

**Symptom.** A VL1-style deployment with the SigLIP + SAM hybrid encoder kept producing subtly different outputs across replicas after a routine model update.

**Wrong first guess.** Non-determinism in the decoder sampling.

**Root cause.** The two vision encoders were versioned and deployed independently, and a partial rollout left some replicas running a new SigLIP checkpoint against an old SAM checkpoint. The fused features were mismatched on those replicas.

**Fix.** Bundle the two encoders into a single atomically-versioned artifact so they can never drift apart, and add a startup assertion that checks both checksums. This is, incidentally, one fewer failure mode that VL2's single-encoder design eliminates entirely.

**Lesson.** Every extra model in a pipeline is an extra thing to version, deploy, and keep consistent. The operational case for VL2's single encoder is not just FLOPs — it is one fewer artifact to drift.

### 7. The 800B-token run that starved the GPUs

**Symptom.** A team reproducing a VL2-style ~800B-token pretraining stage saw GPU utilization stuck at ~45% despite a healthy batch size.

**Wrong first guess.** The MoE all-to-all communication was assumed to be the bottleneck.

**Root cause.** The data loader. Each training example carried a multi-tile image that had to be decoded, resized, and tiled on the fly, and the CPU-side image pipeline could not produce tiles fast enough to keep the GPUs fed. The model was waiting on pixels, not on gradients.

**Fix.** Pre-tile and cache the image crops offline, and move JPEG decoding to a GPU-accelerated path. Utilization climbed past 85%.

**Lesson.** A VL pretraining run is bottlenecked on image throughput, not text. The "800B tokens" headline hides that most of those tokens are visual, and visual tokens are expensive to *produce*, not just to consume. Budget your data infrastructure for pixels.

### 8. The chart model that read the legend but not the axis

**Symptom.** A VLM read chart legends and titles correctly but consistently misread the y-axis tick values on dense charts.

**Wrong first guess.** The team assumed a numeracy weakness in the language model and added arithmetic fine-tuning.

**Root cause.** Resolution again, but localized. The legend and title sat in regions that survived downsampling; the axis ticks were small and clustered, and at the effective resolution of a single global frame they blurred together. The thumbnail-only context could read the big text and not the small.

**Fix.** Dynamic tiling gave the axis region its own native-resolution tile, so the tick values became legible. Axis-reading accuracy jumped without any change to the language model.

**Lesson.** "The model is bad at numbers" is frequently "the model cannot see the numbers." Before fine-tuning the reasoning, confirm the pixels carrying the answer survive to the encoder at a readable resolution.

### 9. The model that lost track of which tile a number came from

**Symptom.** A tiled VLM reading multi-page financial tables would occasionally attribute a value to the wrong row — reading the correct number but placing it under the wrong header.

**Wrong first guess.** The team assumed a table-structure-understanding weakness and added more table data to fine-tuning.

**Root cause.** The `<tile_newline>` token was being omitted in a fast-path tokenizer optimization that flattened tiles without row markers. Without the row-break markers, the decoder could not reconstruct that tile 4 sat directly below tile 1, so values from adjacent rows blurred together in the model's spatial reconstruction. The structure understanding was fine; the spatial index was corrupted.

**Fix.** Restore the `<tile_newline>` and `<view_separator>` tokens in the fast path. Row attribution accuracy recovered immediately. The team added a unit test asserting the special-token count matches the expected count for a given tile grid.

**Lesson.** The layout special tokens are not decoration you can optimize away. They are the only channel by which 2D spatial structure reaches a 1D decoder. Drop them and the model sees a bag of patches with no geometry.

### 10. The MoE that overloaded one expert on OCR traffic

**Symptom.** A VL2-class deployment serving a near-pure OCR workload showed one expert consistently saturated while others idled, and tail latency suffered under load.

**Wrong first guess.** A bug in the router's top-k selection.

**Root cause.** Workload skew interacting with expert specialization. The shared and routed experts had specialized during training, and a near-homogeneous OCR workload kept routing tokens to the same OCR-relevant experts. The router was working correctly; the *traffic* was pathologically uniform, so the load was uniform too — onto a small set of experts. This is the flip side of expert specialization: it is a feature on diverse traffic and a hotspot on monolithic traffic.

**Fix.** This is a serving-topology problem, not a model bug. The team spread the hot experts across more devices (expert parallelism placement) so the saturated expert was not co-located with the rest of its layer, and capacity factors were tuned to tolerate the skew. The deeper lesson is about workload diversity: MoE load balancing assumes diverse traffic.

**Lesson.** MoE's efficiency assumes a diverse token distribution. A monolithic workload can defeat load balancing by routing everything to the same experts. If your traffic is uniform, profile expert utilization before assuming the activated-parameter math holds — your effective activated set may be smaller and more concentrated than the average suggests.

### 11. The fine-tune that re-broke the language ability VL2 had protected

**Symptom.** A team took a released VL2 checkpoint, fine-tuned it on a domain image-text corpus, and shipped — then found the model's general reasoning had regressed, even though the base VL2 release had carefully preserved it.

**Wrong first guess.** A bad learning rate.

**Root cause.** The same modality competition from case study 1, one level up. The released VL2 checkpoint *arrived* with its language ability intact because the original training managed the ratio. But a downstream fine-tune on near-pure domain image-text data re-introduced the competition the original training had so carefully avoided, and the protection did not survive an unmanaged fine-tune.

**Fix.** Re-introduce a language-data mix into the fine-tune (hold a meaningful language fraction), lower the learning rate, and freeze the most language-critical components for the first phase of fine-tuning — mirroring the staged, ratio-managed approach the base model used. Reasoning recovered.

**Lesson.** Modality competition is not a one-time problem the base model solves for you forever. *Every* subsequent fine-tune re-opens it. If you fine-tune a VLM on image-heavy data without holding a language ratio, you will undo the very protection the base training paid for. Inherit the discipline, do not assume it is baked in.

## When to reach for the DeepSeek-VL / VL2 design, and when not to

### Reach for this design when

- **Your workload is text-in-images.** Documents, forms, screenshots, charts, tables, OCR — the dynamic-tiling-plus-shared-ViT approach is purpose-built for reading fine print at adaptive resolution, and the benchmark strengths (DocVQA, ChartQA, OCRBench, TextVQA) reflect it.
- **You must not regress on language ability.** If your VLM also needs to reason, code, or do math, the modality-competition discipline — warm-up plus a held ~70/30 language ratio plus stage-frozen training — is the recipe that keeps the language brain intact.
- **You serve at scale and care about throughput.** The MLA-compressed KV cache and the MoE's low activated-parameter cost are exactly the levers that make a token-heavy VLM affordable to serve concurrently. A 27B-total / 4.5B-activated model gives you large-model quality at small-model per-token cost.
- **Your images vary wildly in aspect ratio.** Dynamic tiling adapts the tile grid (and therefore the token cost) per image, so a tall receipt and a wide banner each get a fitting tiling instead of a one-size-fits-none square.
- **You want one vision artifact, not two.** VL2's single shared encoder removes the operational drift risk of versioning and deploying two encoders independently.

### Skip or rethink this design when

- **Your images are uniform and simple.** If every input is a small, fixed-size, content-sparse image (icons, thumbnails, fixed-layout UI tiles), dynamic tiling adds machinery you will not use; a single fixed frame through one encoder is simpler and just as good. Tiling pays off on *variable, dense* imagery.
- **You need dense pixel-level vision, not reading.** For fine-grained segmentation or pixel-accurate localization beyond bounding boxes, the contrastive-SigLIP-only front end may under-serve you; a dedicated segmentation-trained detail path (as in VL1's SAM branch) or a purpose-built dense model may be the better tool.
- **You are deploying tiny-scale, single-stream, latency-critical inference.** The MoE's benefit is throughput under concurrency and capacity per activated FLOP. If you serve one request at a time with hard latency limits and small models, a dense small model can have more predictable latency than a sparse one with routing overhead.
- **You cannot host the total parameter count.** A 27B-total model needs memory for 27B of weights even though it activates 4.5B. If your memory budget only fits the activated size, pick a smaller variant (Tiny is 3B total) or a dense model — do not try to run Base on 1.0B-activated-sized hardware.
- **Your team cannot operate MoE training.** Expert routing, load balancing, and all-to-all communication add real systems complexity. If you do not have the infrastructure to train and serve MoE reliably, a dense backbone with MLA still gives you most of the KV-cache win without the routing burden.

## Further reading

- **DeepSeek-VL** (arXiv 2403.05525) — the original hybrid-encoder, language-first VLM and the modality-competition methodology.
- **DeepSeek-VL2** (arXiv 2412.10302) — dynamic tiling, the single shared SigLIP encoder, and the DeepSeekMoE + MLA backbone.
- [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — how VL2's MLA compresses the KV cache that visual tokens inflate.
- [The DeepSeekMoE lineage: fine-grained and shared experts](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) — the sparse decoder VL2 builds on.
- [DeepSeek-V3: FP8, MTP, and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the language-model techniques the VL2 backbone inherits.
- [Janus-Pro: decoupled visual encoding](/blog/machine-learning/computer-vision/janus-pro-decoupled-visual-encoding) — a sibling DeepSeek line that splits understanding and generation encoders.
- [ViT, SigLIP, and DINO](/blog/machine-learning/computer-vision/vit-siglip-dino-explained) — the vision backbones underneath both VL1 and VL2.
- [MoE LLM architecture, training, and finetuning](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) — broader context on the Mixture-of-Experts decoder VL2 uses.
