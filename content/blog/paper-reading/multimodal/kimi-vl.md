---
title: "Kimi-VL: A Mixture-of-Experts Vision-Language Model"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Multimodal"
tags:
  - vision-language-model
  - mixture-of-experts
  - multimodal-reasoning
  - long-context
  - native-resolution
  - gui-agents
  - long-cot
  - moonshot-ai
description: "A deep read of Kimi-VL, an open MoE vision-language model that activates only 2.8B language params yet beats GPT-4o on OCR, document, agent, and spatial tasks, and how its long-CoT Thinking variant pushes multimodal reasoning."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/kimi-vl-1.png"
readTime: 30
---

There is a recurring frustration when you try to run a capable vision-language model in production: the models that actually *see clearly* — that read a dense invoice, follow a multi-step GUI workflow, or reason about a frame of video — tend to be dense, large, and expensive per token. The open ecosystem inherited this constraint almost by default. While language-only models leapt forward by pairing Mixture-of-Experts (MoE) sparsity with long chain-of-thought (long-CoT) reasoning — DeepSeek R1 being the obvious landmark — open vision-language models stayed stubbornly dense and short-context, and almost none of them supported long reasoning at all. You paid full-parameter compute on every token whether the token was a paragraph of OCR text or a single yes/no answer.

Kimi-VL is Moonshot AI's answer to that gap. It is an open MoE vision-language model that activates only **2.8B** language-decoder parameters out of **16B** total in the MoE LLM, plus a **0.4B** vision encoder, for a per-token activated footprint of **3.2B** — hence the "A3B" naming. Despite that small activated path, it matches or beats much larger efficient VLMs across multimodal reasoning, OCR, long-context understanding, video, and GUI-agent tasks. It carries a native **128K** context window, sees images at native resolution up to 3.2 million pixels, and ships a long-CoT "Thinking" variant (Kimi-VL-A3B-Thinking-2506) that pushes reasoning to 64.0 MMMU, 56.9 MathVision, and 80.1 MathVista.

![How Kimi-VL routes pixels and text into one MoE decoder](/imgs/blogs/kimi-vl-1.png)

The diagram above is the mental model: an image enters a native-resolution vision encoder (MoonViT), gets spatially compressed by a pixel-shuffle projector, and its embeddings are interleaved with text embeddings into one long token sequence. That merged sequence flows through a sparse MoE language decoder where, per token, only 2.8B of the 16B parameters fire. There is no separate "fusion module," no cross-attention bridge, no late-fusion adapter — vision tokens and text tokens live in the same sequence and the same decoder. The cleverness is concentrated in three places: how the vision encoder handles arbitrary resolution, how the training schedule protects the language model while folding in pixels, and how a small activated model is taught to reason at length. We will work through each.

> [!tldr] TL;DR
> - **What it claims:** An open MoE VLM with only 2.8B activated language params (3.2B including the vision encoder, 16.4B total) matches or beats much larger efficient VLMs on reasoning, OCR, long-context, video, and GUI-agent tasks, with a 128K window and native-resolution perception.
> - **Why it matters:** It is the first open VLM to combine MoE sparsity, long-context (128K), native resolution, and a long-CoT + RL reasoning recipe in one efficient package — decoupling capability from per-token compute the way DeepSeek-style models did for text.
> - **Most surprising finding:** With 2.8B activated params it beats GPT-4o on OCRBench (867 vs 815), OSWorld (8.22 vs 5.03), EgoSchema (78.5 vs 72.2), MLVU (74.2 vs 64.6), and several others — while still trailing GPT-4o on raw MMMU (57.0 vs 69.1).
> - **Where it fails:** The authors say the model size is "too limited" for highly specialized domains, reasoning "has yet to reach its theoretical upper bound," and long-context ability is "still insufficient" for some advanced uses because the attention layers have relatively few parameters.

## Context: what came before

The lineage here is two-sided. On the language side, the recipe that made 2024-2025 frontier models efficient was MoE sparsity plus long-CoT reasoning. MoE lets you grow total parameters — and therefore knowledge capacity — without growing the per-token FLOPs, because a router sends each token to only a few experts. Long-CoT, popularized by reasoning models, lets a model spend more tokens *thinking* before answering, trading inference compute for accuracy on hard problems. DeepSeek R1 is the obvious reference point: an open MoE model that reasons. Moonshot's own Moonlight checkpoint, trained on 5.2T text tokens with the Muon optimizer, sits directly upstream of Kimi-VL as the LLM it initializes from.

On the vision side, the picture was less rosy. The paper's framing is blunt: open-source VLMs "continue to rely on dense architectures and do not support long-CoT reasoning." The handful of MoE VLMs that did exist — DeepSeek-VL2, Aria — were hamstrung by short context length (roughly 4K), weak fine-grained perception, or weak reasoning. So you could get sparsity *or* you could get good perception *or* you could get long context, but not all three in one open model, and certainly not with reasoning on top.

![The gap Kimi-VL fills in open vision-language models](/imgs/blogs/kimi-vl-4.png)

The before/after above names the gap precisely. Prior open VLMs ran a dense decoder (full params per token), capped context around 4K, and offered no long-CoT path. Kimi-VL replaces all three: a sparse MoE decoder (2.8B of 16B active), a 128K context window, and a long-CoT SFT-plus-RL Thinking variant. The goal the team set was deliberately multi-objective: build an *efficient* (low activated-param) MoE VLM that simultaneously delivers advanced multimodal reasoning, long-context understanding, native-resolution perception ("see clearly"), and strong agent/GUI capabilities. That last objective — agents — is what makes the OCR and screenshot results matter: a GUI agent has to read tiny UI text precisely or it clicks the wrong button.

The gap this paper fills is therefore not "another VLM" but a *combination* that the open ecosystem lacked: the four properties above in a single model whose per-token cost is that of a roughly 3B dense model.

It is worth being precise about why these four properties are hard to co-optimize, because that is what makes the combination a real contribution rather than a checklist. Native resolution fights long context: the higher the resolution, the more vision tokens per image, the faster you exhaust the window — so you cannot naively crank both. MoE sparsity fights reasoning: a small activated path has less raw capacity to hold a long chain of thought together, which is precisely why the broad-knowledge MMMU gap persists even in the Thinking variant. And long-CoT reasoning fights efficiency: the whole point of reasoning is to spend more tokens, which costs more inference compute. Each property pulls against at least one other, and the open MoE VLMs that came before chose one corner of this space and lived there. Kimi-VL's claim is that careful engineering — pixel-shuffle compression to reclaim window budget, a 16B total parameter pool to back the small activated path, and a length-penalized RL objective to keep reasoning from running away — lets a single model occupy a usable point near the center rather than a corner.

## Contributions

The paper's contributions, tightened from the report, are:

1. **MoonViT, a native-resolution vision encoder** initialized from SigLIP-SO-400M and continually pre-trained with NaViT-style patch packing, so images at arbitrary native resolution are processed without tiling or splitting heuristics.
2. **An efficient MoE VLM architecture** — MoonViT plus a two-layer MLP projector with 2×2 pixel shuffle plus a Moonlight-based MoE decoder — that activates only 2.8B of 16B LLM params (3.2B / 16.4B including vision), decoupling capability from per-token compute.
3. **A joint (not sequential) multimodal training recipe** in which every stage that updates the language model mixes pure text data, preserving text ability while integrating vision across 4.4T multimodal tokens.
4. **Long-context activation to 128K** via two staged 4× extensions, a RoPE inverse-frequency reset from 50,000 to 800,000, and a 25%-long / 75%-replay data mix, reaching 87.0% text and 91.7% video needle recall in the 64K–128K range.
5. **A long-CoT Thinking variant** trained with lightweight long-CoT SFT warmup followed by RL using an online policy mirror descent objective with a binary correctness reward and a length penalty against overthinking.
6. **An enhanced Muon optimizer** with added weight decay, per-parameter update-scale adjustment, and a distributed ZeRO-1-style implementation, plus a 4D-parallel training stack that runs roughly 60% faster than a 7B dense VLM baseline.

## Method

The architecture has exactly three components, and the discipline is in how they are wired and trained. We will define each symbol on first use and walk the data path in the order a token travels it.

### MoonViT: the native-resolution eye

The first design choice is the one that pays off most on OCR, documents, and GUI screenshots: process images at their *native* resolution rather than squashing everything to a fixed square. MoonViT is the vision encoder that does this. It has roughly **400M** parameters and is initialized from **SigLIP-SO-400M**, then continually pre-trained.

The mechanism that makes native resolution tractable is **NaViT-style packing**. Instead of resizing an image to a canonical size or splitting it into tiles, "images are divided into patches, flattened, and sequentially concatenated into 1D sequences." A 512×512 image and a 1792×1792 image both become a flat 1D sequence of patch embeddings — they just produce different sequence lengths. This sidesteps the tiling-and-stitching machinery that other high-resolution VLMs use (where you split an image into a grid of crops, encode each, and hope the model reassembles spatial relationships). In Kimi-VL-Thinking-2506, native resolution reaches up to **3.2 million pixels per image** — about 1792×1792, which is 4× the original limit.

Packing loses the 2D grid structure, so MoonViT restores spatial information two ways. First, it interpolates the *fixed-size absolute position embeddings* inherited from SigLIP to the new patch count — keeping SigLIP's pretrained spatial prior intact rather than throwing it away. Second, it adds **2D RoPE** (rotary position embedding) across height and width, so the model has a relative positional signal that generalizes across resolutions. Let $p = (h, w)$ be a patch's row and column index; 2D RoPE rotates the query and key vectors by an angle that depends on $h$ and $w$ separately, giving the attention a sense of "how far apart, in two dimensions" two patches are regardless of the total sequence length.

MoonViT is trained with a CoCa-style dual objective: a SigLIP **contrastive loss** that aligns image and text in a shared embedding space, plus a **cross-entropy caption loss** that teaches the encoder to generate descriptions. The caption term is weighted by $\lambda = 2$, so captioning gets twice the gradient pull of the contrastive term. The intuition is that contrastive learning gives you a good global embedding but a relatively coarse one; adding a generative caption objective forces the encoder to retain fine-grained, localizable detail — exactly what you need to read text in an image.

### The projector: pixel shuffle

The vision encoder emits one feature vector per patch, which is far too many tokens to feed directly into a language model — a single high-resolution image could swamp the context budget. The projector solves two problems at once: it compresses the spatial token count and it maps vision features into the LLM's embedding space.

It is a **two-layer MLP** preceded by a **2×2 pixel-shuffle** operation. Pixel shuffle is a spatial-to-channel trick: it takes a 2×2 block of spatial positions and folds them into the channel dimension, so four spatial tokens become one token with 4× the channels. The spatial token count drops by 4× while no information is discarded — it is relocated, not dropped. The two-layer MLP then projects the resulting wider feature into the LLM embedding dimension. This is why a 1792×1792 image does not produce an unmanageable token sequence: pixel shuffle absorbs the resolution before the LLM ever sees it.

A worked example makes the savings concrete. Suppose MoonViT uses a 14×14 patch size (typical for SigLIP-class encoders). A 1792×1792 image yields $1792 / 14 = 128$ patches per side, so $128 \times 128 = 16{,}384$ patch tokens out of the encoder. Feeding 16,384 vision tokens per image into a 128K-context decoder would be wasteful — a handful of high-resolution images would eat the entire window. After the 2×2 pixel shuffle, the spatial count drops to $64 \times 64 = 4{,}096$ tokens, a 4× reduction, with the lost spatial positions folded into 4× wider channels that the MLP then projects down. So the same image that would have cost 16,384 tokens now costs 4,096 — and you can fit roughly 32 such full-resolution images, or many more lower-resolution ones, inside the 128K window alongside a long text prompt. The pixel-shuffle factor is the single knob that trades vision-token budget against fine spatial granularity, and the team fixed it at 2×2 as the sweet spot for OCR-grade detail at acceptable token cost.

### The MoE language decoder

The decoder is a **Moonlight-based MoE LLM**, structurally DeepSeek-V3-like, with **2.8B activated** parameters and **16B total**. It is initialized not from scratch but from an intermediate Moonlight checkpoint already trained on **5.2T text tokens** — so the language model arrives at multimodal training already fluent in English, Chinese, code, math, and general knowledge. The initial context length is 8K, later extended to 128K (we cover that below).

The MoE mechanism is the source of the efficiency. In a dense decoder, every token passes through the full feed-forward network (FFN). In an MoE decoder, the FFN is replaced by a bank of expert FFNs and a router; each token is routed to a small subset of experts. The total parameter count (and thus knowledge capacity) is large, but the *activated* count per token — the FLOPs you actually pay — is small. Here the ratio is 2.8B activated out of 16B total: roughly 17% of the weights fire per token. The exact expert count, experts-per-token, layer count, and hidden dimensions are not stated in the report beyond "DeepSeek-V3-like," so we will not invent them.

To see why the activated count, not the total, governs your bill, work the rough FLOPs. A standard estimate for a dense transformer forward pass is about $2N$ FLOPs per token, where $N$ is the activated parameter count. For the 3.2B activated path (2.8B LLM plus 0.4B vision), a single token costs on the order of $2 \times 3.2\text{B} \approx 6.4$ GFLOPs in the forward pass — the same as a 3.2B dense model. The 16B of total expert weights sits in memory, contributing knowledge capacity, but never fires on any one token. Contrast that with a hypothetical dense 16B VLM: every token would cost $2 \times 16\text{B} = 32$ GFLOPs, five times more, for the same memory footprint. That 5× compute gap, compounded across 4.4T training tokens and every inference call thereafter, is the entire economic argument for MoE — and it is why the team's training-throughput number (roughly 60% higher than a 7B *dense* VLM baseline) is plausible despite the model holding more than twice as many total parameters as that baseline.

![Where the A3B name comes from: activated vs total params](/imgs/blogs/kimi-vl-3.png)

The stack above is the bookkeeping that explains the "A3B" name. The full model footprint is 16.4B total parameters. The MoE LLM holds 16B of those as expert weights, but only 2.8B of that path is activated (routed) per token. The MoonViT encoder adds 0.4B that is always on (every image token passes through all of it — vision encoders are dense). Sum the activated pieces — 2.8B routed LLM plus 0.4B vision — and you get **3.2B activated per token**, which is the number that governs your inference cost. So you store and serve a 16.4B model but pay compute like a 3.2B dense one. That is the whole pitch in one figure.

A minimal sketch of the forward path, in PyTorch-shaped pseudocode, makes the wiring concrete. Note how vision and text tokens are concatenated into a single sequence before the decoder — there is no separate fusion network:

```python
def kimi_vl_forward(image, input_ids, image_token_mask):
    """One forward pass: pixels through MoonViT + projector,
    text through embedding, merged, then a sparse MoE decoder.
    image_token_mask marks where vision embeds go in the sequence.
    """
    # --- vision branch: native resolution, NaViT packing ---
    patches = patchify_and_pack(image)          # flat 1D patch sequence
    vis_feats = moonvit(patches)                 # 400M encoder, 2D RoPE + abs pos
    vis_feats = pixel_shuffle_2x2(vis_feats)     # 4x spatial -> channel compression
    vis_embeds = mlp_projector(vis_feats)        # two-layer MLP into LLM dim

    # --- text branch ---
    txt_embeds = llm.embed_tokens(input_ids)     # standard token embedding

    # --- merge into one sequence (no cross-attention bridge) ---
    h = txt_embeds.clone()
    h[image_token_mask] = vis_embeds             # splice vision tokens in place

    # --- sparse MoE decoder: only top-k experts fire per token ---
    for layer in llm.layers:
        h = h + layer.attn(rms_norm(h))          # autoregressive attention
        normed = rms_norm(h)
        router_logits = layer.router(normed)     # pick experts per token
        topk_idx, topk_w = top_k(router_logits)  # 2.8B activated of 16B total
        moe_out = layer.moe_ffn(normed, topk_idx, topk_w)
        h = h + moe_out
    return llm.lm_head(rms_norm(h))              # next-token logits
```

The load-bearing line is `h[image_token_mask] = vis_embeds`: vision is injected directly into the token stream, and from there it is *just another token* to the decoder. That uniformity is why the same 128K context machinery, the same MoE router, and the same long-CoT reasoning all apply to image content without special-casing.

### Training schedule: vision first, then joint everywhere

The training schedule is where Kimi-VL avoids a classic failure mode. If you take a strong text LLM and fine-tune it on image-text data, you risk *catastrophic forgetting* — the text ability degrades as vision is learned. The paper's defense is a principle stated plainly: "all stages that update the language model are joint training stages." Every LLM-updating stage mixes pure-text data alongside multimodal data, so the text gradient never disappears.

![The four pretraining stages that spend 4.4T multimodal tokens](/imgs/blogs/kimi-vl-2.png)

The pipeline above is the four-stage pretraining schedule, consuming **4.4T multimodal tokens** on top of the underlying 5.2T text-only LLM pretraining. The four stages:

1. **ViT training (2.1T tokens).** MoonViT is trained standalone on image-text pairs with progressive resolution sampling and the CoCa-style contrastive-plus-caption objectives, at sequence length 8192. This 2.0T phase is followed by a 0.1T *alignment* phase that updates only MoonViT and the projector — the LLM is frozen here. This is the only stage where the language model is not updated, which is exactly why it does not need text replay.
2. **Joint pre-training (1.4T).** The LLM is loaded from the 5.2T checkpoint and the multimodal data ratio is progressively increased while text data is preserved, at sequence length 8192. This is the main "teach the LLM to see" stage.
3. **Joint cooldown (0.6T).** A cooldown on high-fidelity text plus multimodal data: synthetic math and code QA generated via rejection sampling, academic-source filtering and rewriting, and a deliberately *low* QA ratio to avoid overfitting to question-answer formats, again at sequence length 8192.
4. **Joint long-context activation (0.3T).** The context window is grown from 8K to 128K (detailed below).

The split here — 2.1T spent training the eye alone, then 2.3T joint across the LLM — reflects a bet that perception is worth heavy investment before you ever ask the language model to use it.

### Training data: six families

The 4.4T multimodal token budget is drawn from six distinct data families, each engineered for a capability the model needs to ship.

![The six data families behind 4.4T multimodal tokens](/imgs/blogs/kimi-vl-7.png)

The tree above enumerates them: **Caption** (open-source LAION, DataComp plus in-house Chinese/English, with strict limits on synthetic captions to reduce hallucination); **Image-text interleaving** (open-source Obelics plus in-house textbooks, webpages, tutorials, and synthesized interleaving); **OCR** (multilingual, dense layouts, web content, handwritten, with heavy augmentation — rotation, distortion, color, noise — across single- and multi-page documents); **Knowledge** (geometry, infographics, academic materials from textbooks, papers, and the internet); **Agent** (virtual-machine screenshots, desktop/mobile/web action data, GUI icon datasets, and multi-step trajectories with CoT annotations); and **Video** (open-source plus in-house web-scale videos of variable duration with dense captions, for description and grounding). The agent and video families (highlighted) are the differentiators — they are what produce the strong GUI and video numbers later. Underneath all of this sits the text corpus inherited from Moonlight, spanning five domains: English, Chinese, Code, Mathematics & Reasoning, and Knowledge.

The token accounting per stage is worth tabulating, because it shows where the budget actually goes:

| Stage | Tokens | Composition | Seq len |
|---|---|---|---|
| ViT training | 2.0T + 0.1T | image-text pairs + alignment | 8192 |
| Joint pre-training | 1.4T | mixed multimodal + pure text | 8192 |
| Joint cooldown | 0.6T | high-quality text + multimodal QA | 8192 |
| Joint long-context | 0.3T | long text/video/docs (25% long) | up to 131072 |
| **Multimodal total** | **4.4T** | (after 5.2T text-only LLM pretraining) | — |

### Long-context activation: 8K to 128K

The context extension is its own small engineering story. You cannot simply set the context length to 128K and train — the positional encoding was learned for short sequences and will not extrapolate, and naively training only on long sequences degrades short-context performance.

![Walking the context window from 8K to 128K](/imgs/blogs/kimi-vl-6.png)

The timeline above walks the procedure. Starting from an 8K base after pretraining, the window is extended in **two sub-stages, each a 4× extension** (8K → 32K → 128K, where 128K is exactly 131,072 tokens). The key positional fix is a **RoPE inverse-frequency reset from 50,000 to 800,000** — increasing the RoPE base by 16× stretches the rotary frequencies so that positions far apart in a 128K sequence remain distinguishable. The data mix during this stage is **25% long data plus 75% replay**: three-quarters of the tokens are short-context replay specifically to keep short-context ability from regressing. The paper's own framing is that "this composition allows the model to effectively learn long-context understanding while maintaining short-context ability."

The payoff is measured with needle-in-a-haystack (NIAH) recall. At all lengths up to 64K, recall is **100%**. In the hardest bucket — sequence lengths in the (65536, 131072] range — recall is **87.0% on text** and **91.7% on video**. That video recall actually *exceeds* text recall at the longest lengths, which is a nice signal that the video training data was effective. These are honest numbers: 87% is not 100%, and the authors flag long-context as a known weakness later.

### Long-CoT SFT and RL: the Thinking variant

The base Instruct model is a strong perceiver. The Thinking variant is what turns it into a reasoner. The recipe has three post-training stages.

First, **Joint SFT**: two epochs in ChatML format (32K sequence length then 128K), mixing text and VL supervised data, with loss masking on system and user prompts so the model only learns to produce assistant turns. Learning rate runs $2\times10^{-5} \to 2\times10^{-6}$ in the 32K epoch, then rewarms to $1\times10^{-5} \to 1\times10^{-6}$ for the 128K epoch.

Second, a **lightweight long-CoT SFT warmup** on a small verified reasoning set. This warmup explicitly targets four reasoning behaviors — planning, evaluation, reflection, and exploration — to seed the model with the *shape* of long reasoning before RL refines it.

Third, **RL**. The objective is an online policy mirror descent variant:

$$
\max_{\theta}\; \mathbb{E}_{(x,y,y^*)}\big[\, r(x, y, y^*)\,\big] \;-\; \tau \cdot \mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\theta_i}\right)
$$

where $r(x, y, y^*) \in \{0, 1\}$ is a **binary correctness reward** — the answer $y$ either matches the verified target $y^*$ or it does not — $\pi_\theta$ is the current policy, $\pi_{\theta_i}$ is the reference policy from iteration $i$, and $\tau$ controls a KL penalty that keeps the policy from drifting too far from the reference. On top of the binary reward sits a **length-based reward penalty** that curbs *overthinking* — without it, a model rewarded only for correctness learns to ramble, since longer chains sometimes stumble into the right answer. Training uses a curriculum and prioritized sampling so the model spends its RL budget on problems at the right difficulty. Inference remains standard autoregressive decoding — the reasoning is in the generated tokens, not in any special inference machinery.

A compact pseudocode sketch of the RL loop captures the essential moving parts:

```python
def rl_step(policy, ref_policy, batch, tau, beta_len):
    """One online policy-mirror-descent style update.
    Binary correctness reward plus a length penalty against overthinking,
    regularized by KL to the reference policy from this iteration.
    """
    losses = []
    for x, target in batch:                       # curriculum + prioritized sampling
        y = policy.generate(x)                     # long-CoT rollout, autoregressive
        correct = float(verify(y, target))         # r in {0, 1}, verified answer
        len_pen = beta_len * overlength(y)          # discourage rambling chains
        reward = correct - len_pen
        logp = policy.logprob(y, x)
        kl = kl_divergence(policy, ref_policy, x, y)  # stay near reference
        advantage = reward - baseline(x)            # mirror-descent advantage
        losses.append(-(advantage * logp) + tau * kl)
    return torch.stack(losses).mean()
```

The two regularizers are what make this work in practice. The KL term stops reward hacking by anchoring to a sane reference; the length penalty stops the model from buying accuracy with verbosity. Both are cheap to compute and both map directly to failure modes the authors clearly anticipated.

### The optimizer and the training stack

Two infrastructure details deserve a mention because they are doing real work. The optimizer is an **enhanced Muon**: relative to the original Muon, the team adds weight decay, carefully adjusts the per-parameter update scale, and builds a distributed implementation following the ZeRO-1 strategy. Muon is the optimizer that underpins Moonlight, so using it here keeps the vision-language model consistent with the text checkpoint it initializes from.

The training stack uses **4D parallelism** — Data, Expert, Pipeline, and Context parallelism — where the vision tower and decoders are strategically allocated across pipeline stages, combined with ZeRO-1 and selective activation checkpointing. Context parallelism uses FlashAttention. The data pipeline runs on S3-compatible object storage with on-the-fly shuffle, tokenize, loss-mask, and pack operations, augmentation that preserves 2D coordinates, reproducible RNG across workers, and multi-caching for throughput. The headline systems result: training throughput is roughly **60% higher** than a 7B dense VLM baseline. The exact GPU type, count, and total GPU-hours are not stated in the report.

Here is a comparison table that situates Kimi-VL against the architectural families it is reacting to:

| Property | Dense open VLMs | Prior MoE VLMs (DeepSeek-VL2, Aria) | Kimi-VL-A3B |
|---|---|---|---|
| Decoder | dense, full params/token | MoE | MoE, 2.8B of 16B active |
| Activated params/token | full model | varies | 3.2B (incl. 0.4B ViT) |
| Context length | typically short | ~4K | 128K (131,072) |
| Native resolution | often tiled/resized | limited | up to 3.2M px, NaViT packing |
| Long-CoT reasoning | none | none | SFT warmup + RL Thinking variant |
| Vision encoder | various | various | MoonViT, SigLIP-SO-400M init |

## Experiments

The evaluation spans multimodal reasoning, college-level knowledge, OCR, documents, GUI agents, and video. The honest summary is: Kimi-VL-A3B Instruct, with 2.8B activated language params, is competitive with GPT-4o and Qwen2.5-VL-7B across the board, *wins outright* on a meaningful subset (OCR, documents, agents, several video and spatial tasks), and *trails* on raw college-knowledge reasoning (MMMU) where sheer parameter count and reasoning depth dominate.

![Kimi-VL-A3B Instruct against larger and proprietary baselines](/imgs/blogs/kimi-vl-5.png)

The matrix above isolates five tasks where the comparison is most revealing. On **OCRBench**, Kimi-VL scores 867 versus GPT-4o's 815 and Qwen2.5-VL-7B's 864 — a win that directly reflects the native-resolution, caption-weighted vision training. On **OSWorld** (computer-use agent success), it scores 8.22 versus GPT-4o's 5.03 and Qwen2.5-VL-7B's 2.5 — more than 3× the Qwen baseline. On **EgoSchema** (egocentric video QA), 78.5 versus 72.2 and 65.0. On **ScreenSpot-V2** (GUI grounding), 92.8 versus GPT-4o's 18.1 and Qwen2.5-VL-7B's 86.8 — GPT-4o essentially cannot do pixel-precise GUI grounding, which is a known limitation of general-purpose chat models. And on **MMMU val**, the one row colored as a loss, 57.0 versus GPT-4o's 69.1 (it does edge Qwen2.5-VL-7B's 58.6 only narrowly, at 57.0). That last row is the honest caveat: when the task is broad multidisciplinary college knowledge, a 2.8B activated model trails a frontier proprietary model.

Here is the fuller Instruct results table, with named baselines and exact numbers from the report:

| Benchmark | Metric | Kimi-VL-A3B | GPT-4o | Qwen2.5-VL-7B | Other |
|---|---|---|---|---|---|
| MMMU (val) | acc | 57.0 | 69.1 | 58.6 | Gemma3-12B-IT 59.6; DeepSeek-VL2 51.1 |
| MMBench-EN-v1.1 | acc | 83.1 | 83.1 | 82.6 | Gemma3-12B 74.6 |
| MMStar | acc | 61.3 | 64.7 | 63.9 | — |
| MMVet | acc | 66.7 | 69.1 | 67.1 | — |
| RealWorldQA | acc | 68.1 | 75.4 | 68.5 | — |
| AI2D | acc | **84.9** | 84.6 | 83.9 | — |
| BLINK | acc | 57.3 | 68.0 | 56.4 | — |
| MathVista | acc | **68.7** | 63.8 | 68.2 | — |
| MathVision | acc | 21.4 | 30.4 | 25.1 | Gemma3-12B 32.1 |
| InfoVQA | acc | **83.2** | 80.7 | 82.6 | — |
| OCRBench | score | **867** | 815 | 864 | — |
| ScreenSpot-V2 | acc | **92.8** | 18.1 | 86.8 | — |
| ScreenSpot-Pro | acc | **34.5** | 0.8 | 29.0 | — |
| OSWorld | success | **8.22** | 5.03 | 2.5 | — |
| WindowsAgentArena | success | **10.4** | 9.4 | — | GPT-4o-mini 2.7 |
| MMLongBench-Doc | acc | 35.1 | 42.8 | 29.6 | GPT-4o-mini 29.0 |
| Video-MME (w/o sub) | acc | 67.8 | 71.9 | 65.1 | — |
| Video-MME (w/ sub) | acc | 72.6 | 77.2 | 71.6 | — |
| MLVU-MCQ | acc | **74.2** | 64.6 | 70.2 | — |
| LongVideoBench | acc | 64.5 | 66.7 | 56.0 | — |
| EgoSchema (full) | acc | **78.5** | 72.2 | 65.0 | — |
| VSI-Bench | acc | **37.4** | 34.0 | 34.2 | — |
| VideoMMMU | acc | 52.6 | 61.2 | 47.4 | Gemma3-12B 57.2 |
| TOMATO | acc | 31.7 | 37.7 | 27.6 | — |

The bolded rows are the wins over GPT-4o: AI2D (84.9 vs 84.6), MathVista (68.7 vs 63.8), InfoVQA (83.2 vs 80.7), OCRBench (867 vs 815), ScreenSpot-V2/Pro, OSWorld (8.22 vs 5.03), WindowsAgentArena (10.4 vs 9.4), MLVU (74.2 vs 64.6), EgoSchema (78.5 vs 72.2), and VSI-Bench (37.4 vs 34.0). The pattern is consistent: Kimi-VL wins where *perception precision* and *agentic grounding* dominate (OCR, documents, charts, GUI, egocentric video, spatial reasoning), and trails where *broad knowledge and deep reasoning* dominate (MMMU, MathVision, MMVU).

### The Thinking variant changes the reasoning picture

The long-CoT Thinking variant, especially the 2506 revision, moves the reasoning numbers substantially:

| Benchmark | Kimi-VL-Thinking | Thinking-2506 | GPT-4o | Qwen2.5-VL-72B | QVQ-72B | Kimi k1.5 |
|---|---|---|---|---|---|---|
| MathVision | 36.8 | **56.9** | 30.4 | 38.1 | 35.9 | 35.9 |
| MathVista (mini) | 71.3 | **80.1** | 63.8 | 74.8 | 71.0 | 74.9 |
| MMMU (val) | 61.7 | 64.0 | 69.1 | 74.8 | 77.3 | 70.3 |
| MMMU-Pro | 43.0 | 46.3 | 51.7 | 51.1 | — | — |
| VideoMMMU | 55.5 | 65.2 | 61.1 | 60.2 | — | — |

The jump from Instruct's 21.4 MathVision to Thinking-2506's 56.9 is the single most dramatic result in the paper — a 2.8B-activated model with long reasoning beats 72B dense models (Qwen2.5-VL-72B at 38.1, QVQ-72B at 35.9) on competition-style visual math. On MathVista it reaches 80.1, also ahead of the 72B baselines. On VideoMMMU it hits 65.2, beating GPT-4o's 61.1. Note that MMMU still trails the 72B models (64.0 vs 74.8 / 77.3) — reasoning does not fully close the broad-knowledge gap, consistent with the limitations.

Thinking-2506 also improves general (non-reasoning) tasks over Instruct: MMBench-EN-v1.1 84.4 (vs 82.9), MMStar 70.4 (vs 61.7), MMVet 78.1 (vs 66.7), RealWorldQA 70.0 (vs 68.1), ScreenSpot-Pro 52.8 (vs 35.4), OSWorld-G 52.5 (vs 41.6), and MMLongBench-Doc 42.1 (vs 35.1). So the reasoning recipe is not a narrow trick that only helps math — it lifts agent and document tasks too.

The closest the paper comes to an ablation is **test-time scaling** (Figure 13 in the report): on MathVision, accuracy climbs from 18.7% at a 1K thinking-token budget to 36.8% at 16K; MathVista saturates around 70.9% by 4K tokens; MMMU improves consistently with longer thinking. This is the load-bearing evidence that the model genuinely uses its reasoning budget rather than padding — accuracy is monotonic in the thinking budget, which is what you want to see.

### What is load-bearing, and what might not transfer

The wins that I trust most are OCR, documents, and GUI grounding, because they have a clear mechanistic story: native resolution plus a caption-weighted ($\lambda = 2$) vision objective plus dedicated OCR and agent data families directly cause precise text and icon perception. Those should transfer to any deployment that needs to read screens or scans.

The results I would be more cautious about transferring are the *agent success* numbers (OSWorld 8.22, WindowsAgentArena 10.4). These are absolute success rates in the single digits to low teens — Kimi-VL beats the baselines, but everyone is failing most of the time. A 3× relative win on a task where the absolute success rate is 8% is real but fragile; it may not survive a distribution shift to your specific app, OS theme, or screen resolution. The video numbers are strong and have a plausible mechanism (dedicated video data, good long-context recall), but video benchmarks are notoriously sensitive to frame-sampling choices, so I would re-measure on my own sampling before trusting them. And MMMU is a standing reminder that broad knowledge tracks total/activated parameter count — do not expect a 2.8B-activated model to match a frontier model on open-domain college exams.

## Critique

**What is strong.** The central claim — competitive-to-winning multimodal performance at 3.2B activated params — is well supported by a broad, named-baseline benchmark suite, and the wins cluster exactly where the architecture predicts they should (perception-heavy and agentic tasks). The native-resolution-via-NaViT-packing decision is elegant and clearly causal for the OCR results. The joint-training discipline ("every LLM-updating stage mixes text") is a principled defense against catastrophic forgetting, and the long-context recall numbers (100% under 64K, 87%/91.7% at 128K) are reported honestly rather than rounded up. The test-time-scaling curve is the best single piece of evidence that the Thinking variant's reasoning is genuine: accuracy rises monotonically with the thinking-token budget rather than plateauing immediately.

**What is weak or unfalsifiable.** The biggest evidentiary gap is the **absence of formal isolated-component ablations**. The report justifies its design choices qualitatively but provides no head-to-head metric tables for the decisions that matter most. We are told the 75%-replay / 25%-long mix "allows the model to effectively learn long-context understanding while maintaining short-context ability" — but there is no comparison against, say, a 50/50 mix or pure-long training, so the claim that this *specific* ratio is right is unfalsifiable from the paper alone. Likewise, "joint vs sequential training" is justified by appeal to forgetting, with no head-to-head numbers showing how much text ability sequential training would actually cost. The position-embedding choice (interpolated SigLIP absolute + 2D RoPE) has no ablation against alternatives. So while the *end-to-end* result is convincing, the *attribution* of credit to individual design choices rests on intuition rather than controlled experiments.

**What ablation is missing.** The one I most want: a clean MoE-vs-dense comparison at matched *activated* params, multimodally trained on the same data, so we can see how much of the win is the architecture versus the data and recipe. Second: an isolation of the caption-loss weight $\lambda = 2$ against $\lambda = 1$ and $\lambda = 0$, to confirm that the caption objective is what drives the OCR edge. Third: the long-context data-mix sweep mentioned above. None of these exist in the report.

**What would change my mind.** If a controlled ablation showed that a matched-activated *dense* VLM trained on the identical 4.4T-token recipe reached within a point or two of Kimi-VL across the board, I would conclude the MoE architecture is doing far less work than the data and training schedule — and that the real contribution is the recipe, not the sparsity. Conversely, if the agent success rates collapsed under a modest distribution shift (different OS theme, different screen resolution) while OCR held up, I would downgrade the "strong agent capabilities" claim to "strong perception that helps agents in-distribution."

## What I'd build with this

1. **A document-understanding service that respects native resolution.** Most VLM document pipelines downsample or tile, which destroys small text. Kimi-VL's NaViT packing plus pixel-shuffle compression is purpose-built for full-page scans and multi-page PDFs (its OCR data explicitly covered single- and multi-page documents). I would wire it behind a layout-aware retrieval system and lean on the 128K window to feed whole documents rather than chunks.
2. **A GUI agent with a precision-grounding front-end.** The ScreenSpot-V2 score of 92.8 (against GPT-4o's 18.1) says Kimi-VL can locate UI elements pixel-precisely. I would use it as the *grounding* module — "where is the submit button" — and pair it with a stronger reasoning planner for the high-level task decomposition, since OSWorld success is still in single digits.
3. **A long-form video QA tool exploiting the 91.7% video recall at 128K.** Feed an hour of densely sampled frames into the 128K window and ask grounded questions, rather than running a sliding-window summarizer that loses cross-segment relationships.
4. **A budget-tunable reasoning endpoint.** Because the test-time-scaling curve is monotonic (MathVision 18.7% at 1K → 36.8% at 16K thinking tokens), I would expose the thinking-token budget as a first-class latency/accuracy dial: cheap mode for easy queries, deep mode for hard visual math — with the length penalty already trained in to avoid runaway chains.
5. **A self-hosted alternative to a frontier VLM for OCR-and-agent workloads.** At 3.2B activated params the serving cost is modest; for workloads dominated by reading screens and documents (exactly where it beats GPT-4o), an open self-hosted Kimi-VL is an attractive cost and privacy story.

## When to reach for Kimi-VL (and when not to)

Reach for Kimi-VL when your workload is **perception-heavy, agentic, long-context, or video**, and when **per-token cost matters**. If you are building a document reader, a screen-understanding agent, a chart/infographic QA system, or a long-video analyzer, the architecture is aimed directly at you, and the OCR/document/GUI/spatial wins over GPT-4o at a fraction of the activated compute are a strong reason to self-host. The Thinking-2506 variant is the right choice specifically when visual math or multi-step visual reasoning is the bottleneck — a 2.8B-activated model beating 72B dense models on MathVision (56.9 vs 38.1) is genuinely remarkable, and the test-time-scaling dial lets you trade latency for accuracy explicitly.

Do *not* reach for it as a drop-in frontier replacement for **broad open-domain knowledge reasoning**. The MMMU gap (Instruct 57.0, Thinking-2506 64.0, versus GPT-4o 69.1 and 72B models in the mid-70s) is the model telling you where its 2.8B activated path runs out of room. The authors say it themselves: the current size "remains too limited to address highly specialized or domain-specific problems," reasoning "has yet to reach its theoretical upper bound," and long-context ability is "still insufficient for certain advanced applications" because the attention layers carry relatively few parameters. So for a graduate-level multidisciplinary tutor, or a task needing deep domain expertise, a larger model is still the right call. The honest framing is the one the architecture itself implies: Kimi-VL decouples *perception and agentic capability* from per-token compute extremely well, and decouples *broad knowledge* from compute less well — pick it for the former, not the latter.

## References

- **Kimi-VL Technical Report** (Kimi Team, Moonshot AI), arXiv abstract: [https://arxiv.org/abs/2504.07491](https://arxiv.org/abs/2504.07491)
- **GitHub:** [https://github.com/MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL)
- [Kimi k1.5: Scaling Reinforcement Learning with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — the RL recipe Kimi-VL's Thinking variant builds on.
- [Muon is Scalable for LLM Training: Inside Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) — the optimizer and the Moonlight checkpoint Kimi-VL initializes from.
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the broader Moonshot agentic-model lineage.
- [WorldVQA: Measuring Atomic World Knowledge in Multimodal LLMs](/blog/paper-reading/multimodal/worldvqa) — a complementary view on what multimodal benchmarks actually measure.
