---
title: "DeepSeek's MoE Lineage: Fine-Grained Experts, Shared Experts, and Balancing Without a Loss (DeepSeekMoE to V2 to V3)"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer trace of DeepSeek's Mixture-of-Experts architecture across three papers: fine-grained expert segmentation and shared-expert isolation in DeepSeekMoE, device-limited routing with three-tier balance losses in V2, and auxiliary-loss-free balancing in V3 — and why each step targets the resource that just became scarce."
tags: ["llm", "deepseek", "mixture-of-experts", "moe", "deepseekmoe", "deepseek-v2", "deepseek-v3", "expert-routing", "load-balancing", "model-architecture", "sparse-models", "scaling"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most explanations of Mixture-of-Experts stop at the cartoon: a router picks a few experts per token, you get the capacity of a big model at the compute of a small one, everyone goes home. That cartoon is true and almost useless for understanding what DeepSeek actually built. The interesting story is not "MoE works" — it's that across three papers in eleven months, DeepSeek changed the MoE recipe **three times**, and each change was a direct response to a different resource becoming the bottleneck. Routing freedom was scarce, so they made experts smaller and more numerous. Then cross-GPU bandwidth was scarce, so they bounded how far a token's experts could spread and added a balance loss that counted bytes instead of tokens. Then the balance loss itself became the cost — a gradient tax fighting the language objective — so they deleted it and replaced it with a feedback controller that carries no gradient at all.

This post is a technique deep-dive tracing that evolution: **DeepSeekMoE** (arXiv 2401.06066, January 2024), **DeepSeek-V2** (arXiv 2405.04434, May 2024), and **DeepSeek-V3** (arXiv 2412.19437, December 2024). It is deliberately *not* a generic MoE tutorial — if you want the foundational mechanics of gating, top-k routing, the auxiliary load-balancing loss, and capacity factors, read the general primer first: [the MoE architecture, training, and fine-tuning case studies post](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies). Here we assume you know what a router and an expert are, and we focus entirely on the DeepSeek-specific decisions: *fine-grained segmentation → shared-expert isolation → device-limited routing → communication-aware balancing → auxiliary-loss-free control*. Each one is reusable. Most of them are now in everybody's MoE.

![Fine-grained expert segmentation: splitting each expert FFN into m=4 quarter-width slivers and activating four times as many keeps activated FLOPs fixed while multiplying the number of distinct routing combinations.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-1.webp)

The diagram above is the mental model for the whole lineage. On the left is the GShard-style baseline DeepSeek started from: a handful of full-width experts, route to the top two, done. On the right is DeepSeekMoE's first move — cut each expert's feed-forward width to a quarter, quadruple the count, and activate four times as many. The arithmetic is the point: activated FLOPs are *identical* on both sides (you compute the same total expert width per token), but the number of ways the router can combine experts explodes. Everything DeepSeek did afterward is a consequence of taking that idea seriously and scaling it from 64 experts to 256, from one server to thousands of GPUs.

## Why the DeepSeek MoE story is different

The instinct on reading three DeepSeek papers is to file them under "they kept making the model bigger." That misses the structure. The three architectural eras are not bigger versions of one idea — they are *three different ideas*, each one solving a problem the previous step created. Make experts tiny and numerous, and you get glorious routing freedom — but now you have hundreds of experts whose tokens have to be shuffled across GPUs, so communication becomes the wall. Bound the communication with device-limited routing and balance losses, and the balance losses now eat into your token budget by perturbing the gradient — so you rip them out and steer load with a bias instead.

| Common assumption | The naive view | The reality in the DeepSeek lineage |
|---|---|---|
| "MoE is one architecture" | Router + experts, pick top-k | Three distinct recipes across 11 months, each targeting a new bottleneck |
| "More experts just means more parameters" | Bigger lookup table | Fine-grained segmentation buys *combinatorial* routing freedom at fixed FLOPs |
| "All experts are interchangeable" | A bag of FFNs | Shared experts (always-on) vs routed experts (gated) do different jobs |
| "Load balancing is a solved auxiliary loss" | Add `L_aux`, tune the coefficient | V2 stacks three balance losses; V3 deletes the loss entirely |
| "Routing is free; experts cost FLOPs" | Compute dominates | At scale, the all-to-all communication dominates — routing topology is the design |
| "Balancing helps the model" | A harmless regularizer | The balancing gradient *fights* the LM loss; V3 measures and removes that tax |

The rest of this article walks the lineage in order. Sections 1 and 2 are DeepSeekMoE's two ideas (segmentation, shared experts). Section 3 is the combinatorics that justify segmentation. Section 4 is the lineage overview. Section 5 is the full config matrix. Sections 6 and 7 are V2's routing and balance-loss machinery. Section 8 is V3's loss-free balancing. Then case studies, and a when-to / when-not closing.

## 1. Fine-grained expert segmentation: routing freedom at fixed FLOPs

**Senior rule of thumb: if you can't afford to activate more compute, buy more *choices* instead — they're free.**

Start from the GShard-style MoE that DeepSeek used as a baseline. You have $N$ experts, each a standard feed-forward network (FFN) with intermediate dimension $d_{ff}$. A token's router scores all $N$ experts and the top $K$ (say $K=2$) fire. The activated compute per token is $K$ full FFNs. The capacity of the layer is $N$ full FFNs. The classic MoE trade: capacity scales with $N$, cost scales with $K$.

DeepSeekMoE's first change is almost embarrassingly simple. Pick a segmentation factor $m$ (they use $m = 4$). Slice each expert's FFN intermediate dimension by $1/m$ — so each "expert" is now a $0.25\times$ FFN — and split the pool into $mN$ of these slivers. To hold the activated compute constant, activate $mK$ of them. With $m=4$, $N=16$, $K=2$, you go from "2 of 16 full experts" to "8 of 64 quarter experts." The total width you compute per token is unchanged: $8 \times 0.25 = 2 \times 1.0 = 2.0$ FFN-equivalents. Same FLOPs. Same parameter count in the activated path.

What changed is the *granularity of specialization*. With two big experts you are forced to pack broad, heterogeneous skill into each one. With eight small experts the router can assemble a much more precise mixture — one sliver that handles arithmetic, one that handles a particular syntactic construction, one that handles a domain term — and combine them per token. The DeepSeekMoE paper frames this as "a more flexible combination of activated experts," which undersells it. The flexibility is combinatorial, and Section 3 makes the number concrete.

Here is the segmentation as the layer's forward pass, in PyTorch-flavored pseudocode that's close enough to run:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FineGrainedMoE(nn.Module):
    def __init__(self, d_model, d_ff_full, n_experts_full=16, m=4, top_k_full=2):
        super().__init__()
        # Segmentation: each "expert" is now a 1/m-width FFN, and there are
        # m times as many of them. Activated count scales by m to hold FLOPs.
        self.n_experts = n_experts_full * m          # 16 -> 64
        self.d_ff = d_ff_full // m                   # full -> 0.25x width
        self.top_k = top_k_full * m                  # 2 -> 8
        self.router = nn.Linear(d_model, self.n_experts, bias=False)
        self.w_in = nn.Parameter(torch.empty(self.n_experts, d_model, self.d_ff))
        self.w_out = nn.Parameter(torch.empty(self.n_experts, self.d_ff, d_model))
        nn.init.normal_(self.w_in, std=0.02)
        nn.init.normal_(self.w_out, std=0.02)

    def forward(self, x):                            # x: [tokens, d_model]
        logits = self.router(x)                      # [tokens, n_experts]
        gate = F.softmax(logits, dim=-1)
        topv, topi = gate.topk(self.top_k, dim=-1)   # 8 winners per token
        topv = topv / topv.sum(dim=-1, keepdim=True)  # renormalize the kept mass
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = topi[:, slot]                      # which expert each token picked
            w_in = self.w_in[idx]                    # gather per-token weights
            w_out = self.w_out[idx]
            h = F.gelu(torch.einsum("td,tdf->tf", x, w_in))
            y = torch.einsum("tf,tfd->td", h, w_out)
            out = out + topv[:, slot:slot + 1] * y
        return out
```

The gather-per-token loop is naive — real implementations sort tokens by expert and do grouped GEMMs — but it shows the invariant: with `d_ff` cut by `m` and `top_k` multiplied by `m`, the einsum FLOPs are identical to the coarse baseline. You are paying the same compute and getting `m`x finer routing.

### Second-order effect: the router gets harder, and that's mostly fine

There is no free lunch, only a cheap one. Quadrupling the expert count makes the router's job harder in two ways. First, the softmax is now over 64 logits instead of 16, so the router has more opportunities to be miscalibrated, and a poorly initialized router can collapse onto a few popular slivers faster. Second, the per-token gather touches four times as many distinct weight tensors, which in a distributed setting means four times as many experts a token might have to be shipped to. The first problem is what the balance losses (Section 7) and later the bias controller (Section 8) exist to solve. The second problem is what device-limited and node-limited routing (Section 6) exist to solve. In other words: segmentation is the move that created the two problems the rest of the lineage spends its time fixing. It is still worth it, because the quality gain from combinatorial routing is large and the fixes are cheap.

One concrete data point from the paper: DeepSeekMoE-2B, the ablation-scale model, matches GShard-2.9B (which has 1.5x the expert parameters) — and approaches the performance of a *dense* model with the same total parameter count, which is the theoretical ceiling for an MoE. Fine-grained segmentation is most of why.

### The FLOP-equivalence is exact, and worth verifying once

It is worth grinding through the FLOP arithmetic once, because the whole argument for segmentation rests on it being *exactly* free, not approximately free. A single FFN expert with model dimension $d$ and intermediate dimension $d_{ff}$ costs, per token, roughly $2 \times 2 \times d \times d_{ff}$ multiply-accumulates (two matmuls — up-projection $d \to d_{ff}$ and down-projection $d_{ff} \to d$ — each $2 d\, d_{ff}$ FLOPs, the leading 2 for multiply-and-add). Activating $K$ full experts costs $K \cdot 4 d\, d_{ff}$.

Now segment. Each sliver has intermediate dimension $d_{ff}/m$, so it costs $4 d \cdot (d_{ff}/m)$ per token. Activating $mK$ slivers costs $mK \cdot 4 d \cdot (d_{ff}/m) = K \cdot 4 d\, d_{ff}$ — the $m$ cancels exactly. The activated FLOPs are *identical*, to the last multiply-accumulate. The router cost grows slightly (it scores $mN$ experts instead of $N$, a $d \times mN$ matmul vs $d \times N$), but the router is a rounding error next to the experts — for $d = 2048$, $N = 16$, $m = 4$, the router is well under 1% of the layer's FLOPs either way. So segmentation is free in the only budget that matters at training and serving time: the per-token expert compute. You are buying $\binom{mN}{mK} / \binom{N}{K}$ times more routing combinations (Section 3) for a sub-1% router-cost increase. There is no other knob in MoE design with that leverage.

The catch — there is always a catch — is *memory bandwidth*, not FLOPs. Activating $mK$ small experts means gathering $m$ times as many distinct weight tensors per token, and on hardware where the MoE is memory-bound (small batch inference, especially) the gather can dominate. Fine-grained segmentation is a clean win when you are compute- or capacity-bound and a wash-to-loss when you are gather-bound. This is why production serving stacks group tokens by expert and use grouped GEMMs: to amortize the weight load across many tokens and keep the layer compute-bound, where segmentation's win is real.

## 2. Shared-expert isolation: stop re-learning the common case

**Senior rule of thumb: if every specialist on your team keeps re-deriving the same background knowledge, hire one generalist and let the specialists specialize.**

The second DeepSeekMoE change addresses a subtle waste in vanilla MoE. Every routed expert, to be useful on its own, has to internalize a baseline of common knowledge — the grammar, the high-frequency tokens, the basic world model that *every* token needs regardless of which specialist handles it. So that common knowledge gets redundantly baked into many experts. That redundancy is paid for twice: once in parameters (many experts storing the same thing) and once in routing pressure (the router can't cleanly separate "what is this token about" from "what does every token need").

DeepSeekMoE isolates $K_s$ **shared experts** that are *always* activated for *every* token, with no gating. The remaining routed experts are selected as usual, but now they only need to handle what is *left over* after the shared experts have covered the common case. The shared experts soak up the redundant baseline; the routed experts are freed to be genuinely specialized.

![Shared-expert isolation: always-on shared experts absorb the common knowledge every token needs, so the gated routed experts no longer waste capacity re-learning it and are free to specialize.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-2.webp)

The figure traces the two paths a token takes. The token goes *unconditionally* to the shared experts (top path, always on) and *conditionally* through the router to its top-k routed experts (bottom path, gated). Both contributions are summed into the output. The shared experts are not chosen — they are infrastructure. To keep activated FLOPs constant when you add $K_s$ shared experts, you reduce the number of routed experts you activate by $K_s$: in DeepSeekMoE-16B, the router activates the 2 shared experts plus 6 of the 64 routed experts, so 8 expert-slivers fire per token in total.

The forward pass with both kinds of expert:

```python
class SharedExpertMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_routed=64, n_shared=2, top_k_routed=6):
        super().__init__()
        self.shared = nn.ModuleList(
            [FFN(d_model, d_ff) for _ in range(n_shared)]   # always on
        )
        self.routed = nn.ModuleList(
            [FFN(d_model, d_ff) for _ in range(n_routed)]   # gated, top-6
        )
        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.top_k = top_k_routed

    def forward(self, x):
        # Shared experts: unconditional, no gate. Every token pays for these.
        y = sum(expert(x) for expert in self.shared)
        # Routed experts: top-6 by gate weight, contribution scaled by the gate.
        gate = F.softmax(self.router(x), dim=-1)
        topv, topi = gate.topk(self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)
        for slot in range(self.top_k):
            for e in range(len(self.routed)):
                mask = (topi[:, slot] == e)
                if mask.any():
                    y[mask] += topv[mask, slot:slot + 1] * self.routed[e](x[mask])
        return y
```

### Second-order effect: shared experts change what "balanced" means

There is a reason this pairs naturally with fine-grained segmentation. Once experts are small, dedicating two of them to the always-on common case is cheap — you are spending $2 \times 0.25 = 0.5$ FFN-equivalents on shared knowledge, not two full FFNs. With coarse experts, an always-on expert is a large fixed cost; with fine-grained experts it is a rounding error. Segmentation makes shared-expert isolation affordable.

A subtle consequence: because shared experts are always on, they do not participate in load balancing — they are by construction perfectly balanced (every token, every shared expert). The balancing problem applies only to the *routed* pool. This matters in V3, where the single shared expert per layer is a stable floor under the whole layer's output, which makes the routed experts' job easier and the loss-free bias controller's job easier too.

> The shared expert is the part of the MoE that behaves like a dense model. It is the reason a well-built MoE degrades gracefully when routing is imperfect: even if the router sends a token to the wrong specialists, the always-on path still gives it a sensible baseline. Treat the shared expert as the safety floor of the layer.

### How shared experts change the learning dynamics

The deeper reason shared-expert isolation helps is about *what each expert is incentivized to learn*. In a vanilla MoE without shared experts, the gradient that reaches a routed expert is a mixture of two signals: "learn the common baseline this token needs" and "learn the specialized behavior this token needs." Because every token needs the common baseline, that first signal is loud and present in every expert's gradient, so every expert spends capacity tracking it — and because experts are selected sparsely, the common baseline is learned inconsistently across the pool (an expert only updates on the tokens routed to it, so its copy of the baseline is noisier than a dense model's would be).

Isolating shared experts cleanly separates the two signals. The always-on shared expert receives the common-baseline gradient on *every* token — it is dense, so it learns the baseline as well as a dense model does, consistently. The routed experts now receive gradient only for the *residual* — what is left after the shared expert has handled the common case — so their incentive is purely to specialize on that residual. The result is sharper specialization (the routed experts are not diluted by the common case) and a better-learned baseline (the shared expert is dense, not sparse). You can see this in the router's behavior: with a shared expert, routed-expert utilization becomes more *differentiated* (the experts genuinely do different things), which is the opposite of what a naive reading might predict. Removing redundant load from the routed pool does not make them more interchangeable; it makes them more distinct, because they no longer all have to be partial generalists.

The decomposition is clean enough to write down. If $f_{\text{ideal}}(x)$ is the ideal per-token function, shared-expert isolation factors it as $f_{\text{shared}}(x) + \sum_{i \in \text{top-}k} g_i(x)\, f_i(x)$, where $f_{\text{shared}}$ learns the part of $f_{\text{ideal}}$ that is common across tokens and the routed sum learns the token-specific residual. The factorization is not enforced — nothing makes $f_{\text{shared}}$ learn *only* the common part — but the gradient dynamics push it there, because the shared expert is the only component that sees every token and so is the cheapest place to put anything common. Capacity flows to where it is cheapest to use, and for common knowledge that is the always-on path.

## 3. The combinatorics: why segmentation is not just "more experts"

**Senior rule of thumb: the value of fine-grained routing is the number of *subsets* it can form, and subsets grow super-exponentially.**

It is tempting to read fine-grained segmentation as "64 experts is more expressive than 16 experts" and leave it there. That undersells it by about seven orders of magnitude. The right unit of expressiveness is not the number of experts — it is the number of distinct *activation patterns* the router can produce, because each pattern is a different effective sub-network the token can be processed by.

![Routing combinatorics: choosing 8 of 64 quarter-width experts yields roughly 4.4 billion distinct activation patterns versus 120 for 2 of 16, about 37 million times more effective sub-networks at the same activated FLOPs.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-3.webp)

The figure does the arithmetic. The number of ways to choose $K$ experts out of $N$ is the binomial coefficient $\binom{N}{K}$. For the coarse baseline, $\binom{16}{2} = 120$. For the fine-grained version, $\binom{64}{8} = 4{,}426{,}165{,}368 \approx 4.43 \times 10^9$. That is a factor of roughly $3.7 \times 10^7$ — about thirty-seven million times — more distinct subsets of experts, at *identical* activated FLOPs.

$$
\binom{16}{2} = \frac{16!}{2!\,14!} = 120, \qquad
\binom{64}{8} = \frac{64!}{8!\,56!} = 4{,}426{,}165{,}368
$$

You should be a little skeptical of treating every subset as equally meaningful — the router does not actually exercise all 4.4 billion, and many would be near-duplicates. But the directional argument is sound and it is the whole justification for segmentation: the model's effective hypothesis space over "which combination of fine-grained skills processes this token" is vastly larger, and the model is free to use as much of that space as the data warrants. Empirically it uses a lot of it, which is why segmentation buys real quality and not just a bigger parameter count.

A quick worked example to make the scale visceral. Suppose you wanted to *match* the combinatorial routing freedom of "8 of 64" using full-size experts and the same activated FLOPs (i.e. activate 2 full experts). You would need $\binom{N}{2} \ge 4.43 \times 10^9$, which requires $N \ge 94{,}000$ full experts. Ninety-four thousand full-width FFNs is an absurd parameter count. Segmentation gets you the same routing freedom with 64 quarter-width experts. That is the trade: a 4x parameter increase (64 quarters = 16 fulls in parameters) buys you the routing freedom of a 94,000-expert coarse model. This is why "fine-grained" is the single most important word in the DeepSeekMoE title.

### Why "more subsets" actually translates to capacity

The skeptic's objection — that 4.4 billion subsets is a meaningless number because no router uses them all — deserves a real answer, and the answer is about *information*, not raw counts. The right way to measure the router's expressiveness is the entropy of its routing distribution: how many bits the router's choice carries about the token. A router choosing uniformly among $\binom{N}{K}$ subsets carries $\log_2 \binom{N}{K}$ bits. For coarse routing that ceiling is $\log_2 120 \approx 6.9$ bits; for fine-grained it is $\log_2(4.43 \times 10^9) \approx 32$ bits. The fine-grained router *can* condition the token's processing on roughly 32 bits of "which skills apply," versus 7 bits for the coarse one. Whether it uses all 32 depends on the data — but the ceiling is what bounds achievable specialization, and a $25$-bit-higher ceiling is enormous headroom.

There is a second, more mechanical reason segmentation helps that has nothing to do with combinatorics: **granularity of the gate weights.** With two big experts the model can only mix in increments of "half the activated compute." With eight small experts it can mix in increments of "one eighth," and the gate weights let it interpolate continuously between them. The output is a convex combination of the activated experts weighted by gate value, so finer experts give a finer-grained, more expressive convex hull of outputs. Segmentation improves both the *discrete* choice (which experts) and the *continuous* mixing (how much of each). The papers' ablations isolate the discrete win, but the continuous one is real and free.

A worked numerical example for the mixing point. Say a token genuinely needs a processing that is 30% "expert A behavior" and 70% "expert B behavior." A coarse 2-of-2 router can only offer A-or-B at roughly 50/50 mixing granularity — it cannot cleanly express 30/70 without a third option. A fine-grained 8-of-many router can dedicate, say, 3 slivers to A-flavored processing and 5 to B-flavored, landing much closer to the true 30/70 the token wanted. The finer the experts, the closer the achievable mixture to the ideal one. This is the same reason a palette of 256 colors reproduces an image better than a palette of 16: more, smaller primitives approximate any target more faithfully.

## 4. The lineage: each release targets the resource that just got scarce

**Senior rule of thumb: read an architecture's version history as a sequence of bottleneck shifts, not feature additions.**

With the two DeepSeekMoE ideas in hand, step back and look at the whole arc. The three releases are not "v1, v1.5, v2 of the same design." They are three different answers to three different scarcities.

![The DeepSeek MoE lineage: from a 16B model with fine-grained and shared experts, to a 236B model with device-limited routing and three-tier balance losses, to a 671B model that drops the balance loss for a bias control loop.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-4.webp)

The timeline lays out the three eras. **DeepSeekMoE (Jan 2024, 16B total / 2.8B activated)** establishes the architecture: 28 layers, hidden dimension 2048, MoE layers with 2 shared and 64 routed experts (each $0.25\times$ a standard FFN), routing the 2 shared plus 6 of the 64 per token. The scarce resource here is *routing freedom*, and segmentation buys it. Load balancing is handled by expert-level and device-level auxiliary losses.

**DeepSeek-V2 (May 2024, 236B total / 21B activated)** scales the expert pool to 160 routed plus 2 shared, activating 6 routed per token, and confronts the consequence of having that many experts on a large GPU cluster: the all-to-all communication that ships tokens to their experts becomes the wall. The scarce resource is now *communication bandwidth*. V2's answer is device-limited routing (each token's experts confined to at most 3 devices) plus a *third* balance loss that explicitly balances communication volume, not just token counts.

**DeepSeek-V3 (Dec 2024, 671B total / 37B activated)** pushes to 256 routed plus 1 shared expert, activating 8 routed per token, on bandwidth-throttled H800 hardware. By now the balance losses themselves are the problem: they inject a gradient that fights the language-modeling objective, costing quality. The scarce resource is *the loss budget*. V3's answer is auxiliary-loss-free balancing — a per-expert bias steered by a feedback controller, carrying no gradient — plus node-limited routing (at most 4 nodes per token). We cover V3's loss-free balancing at a high level here and link to the dedicated teardown for the gritty details; see [the DeepSeek-V3 FP8, MTP, and loss-free balancing deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing).

The pattern is the whole lesson: **each architectural change is a response to whatever became the binding constraint after the previous change succeeded.** Segmentation made routing free, which made communication the wall. Bounding communication made the balance losses the tax. Removing the balance losses is where the lineage currently rests.

There is a useful way to read this against the broader field. Most MoE papers in this window optimized one axis — better gating, better balance loss, better expert architecture — in isolation. DeepSeek's distinguishing move was to treat the *whole system* as the unit of optimization and to let the hardware dictate which axis to attack next. The 16B model was published before the team had a thousand-GPU bandwidth problem, so it optimized routing freedom. By V2 they were training at a scale where the all-to-all dominated, so the paper is mostly about routing topology and communication. By V3 they were on export-throttled H800s where every byte across InfiniBand was precious and the balance-loss tax was measurable, so the paper deletes the loss and obsesses over overlap. The architecture is a fossil record of the constraints the team faced, in order. If you read the three papers expecting three versions of one idea, you will be confused; if you read them as three different problems, the design choices are crisp. If you only remember one thing about DeepSeek's MoE evolution, remember that it is a chain of bottleneck migrations, and that the design discipline is to always be attacking the *current* binding constraint rather than re-optimizing the one you already solved.

## 5. The config matrix: what actually changed, number by number

**Senior rule of thumb: when you compare model generations, line up the hyperparameters in a single table before you theorize — the deltas tell you the story.**

Prose hides the deltas. Here is the same lineage as a matrix, every cell a verified number from the three papers.

![MoE config across the lineage: routed experts grow from 64 to 256 while shared experts fall from 2 to 1, activated routed grows from 6 to 8, and the routing limit migrates from device-level to node-level as balancing moves from auxiliary losses to a bias control loop.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-5.webp)

The figure is the canonical comparison; here it is again as a table you can copy:

| Dimension | DeepSeekMoE 16B | DeepSeek-V2 236B | DeepSeek-V3 671B |
|---|---|---|---|
| Total parameters | ~16.4B | 236B | 671B |
| Activated per token | ~2.8B | 21B | 37B |
| Routed experts / layer | 64 | 160 | 256 |
| Shared experts / layer | 2 | 2 | 1 |
| Activated routed / token | 6 | 6 | 8 |
| Expert width | 0.25x FFN | fine-grained | 0.25x-class, $d_{ff}$ 2048 |
| Routing constraint | device-level aux loss | device-limited (M=3) | node-limited (M=4) |
| Gating function | softmax | softmax | sigmoid |
| Balancing | expert + device loss | expert + device + comm loss | bias control loop (no balance gradient) + tiny seq loss |
| Balance coefficients | $\alpha_1{=}0.001$, $\alpha_2{=}0.05$ | $\alpha_1{=}0.003$, $\alpha_2{=}0.05$, $\alpha_3{=}0.02$ | $\gamma{=}0.001$ (bias), $\alpha{=}0.0001$ (seq loss) |

A few of these deltas deserve a sentence each, because they encode design decisions, not just scaling.

**Routed experts 64 → 160 → 256, shared 2 → 2 → 1.** The routed pool grows roughly 4x while the shared pool *shrinks* to a single expert by V3. The single shared expert is a deliberate simplification: at 256 routed experts, the always-on baseline only needs one generalist, and one is cheaper to keep perfectly balanced. The shared:routed ratio falls from 2:64 to 1:256 — the model leans harder on specialization as the pool grows.

**Activated routed 6 → 6 → 8.** Activated count barely moves, which is the point: the activated-FLOPs budget per token is held roughly constant (2.8B → 21B → 37B activated tracks total scale, not a blow-up in per-token compute). Capacity scales with the 256; cost scales with the 8 plus the 1 shared.

**Gating softmax → softmax → sigmoid.** A quiet but real change. V3 switches the router from softmax over experts to a per-expert sigmoid affinity, then normalizes the selected experts' gates. Sigmoid decouples the experts' scores from each other, which plays better with the bias controller (you can add a bias to one expert's score without renormalizing a softmax over all of them). This is the kind of detail that only matters once you are doing loss-free balancing — and it is exactly why V3's balancing approach needed a gating change to land cleanly.

**Routing constraint device → device-limited(3) → node-limited(4).** This is the communication story, and it gets its own section next.

## 6. Device-limited then node-limited routing: bounding the all-to-all

**Senior rule of thumb: in a distributed MoE, the router does not just pick experts — it picks a communication pattern, and an unconstrained router picks the worst one.**

Here is the problem fine-grained segmentation handed us. With 160 or 256 experts spread across many GPUs, a token's top-k experts can land on *any* k of those GPUs. The MoE layer is implemented as an all-to-all: every GPU sends each of its tokens to whichever GPUs host that token's chosen experts, the experts compute, and a second all-to-all sends the results back. If a token's k experts are scattered across k different devices, that token contributes to k separate cross-device transfers. Multiply by a batch of thousands of tokens and the all-to-all volume — and worse, its *worst-case fan-out* — balloons. On bandwidth-limited interconnects (and the H800s V3 trained on have NVLink bandwidth cut roughly in half by export controls), this communication is the dominant cost of the MoE layer.

![Device-limited versus node-limited routing: V2 confines each token's experts to at most 3 devices to bound intra-server all-to-all fan-out, while V3 raises the unit to at most 4 whole nodes to bound the cross-node InfiniBand traffic that dominates on bandwidth-throttled H800 hardware.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-6.webp)

The fix is to constrain *where* a token's experts may live. DeepSeek-V2 introduces **device-limited routing**: each token's selected experts are confined to at most $M = 3$ devices. Concretely, the router first picks the $M$ devices with the highest aggregate affinity (summed over the experts they host), then selects the top-k experts only from within those $M$ devices. A token can still pick 6 experts, but those 6 are guaranteed to live on at most 3 devices, so the token participates in at most 3 cross-device sends, not 6. This caps the per-token communication fan-out without meaningfully hurting quality — the paper reports device-limited routing achieves performance roughly on par with unrestricted routing.

DeepSeek-V3 generalizes the same idea one level up the hierarchy. The relevant scarcity on a multi-node H800 cluster is not "how many GPUs" but "how many *nodes*" — because intra-node GPUs talk over fast NVLink while cross-node traffic goes over slower InfiniBand. So V3 uses **node-limited routing**: each token's experts are confined to at most $M = 4$ nodes, chosen as the 4 nodes with the highest sum of the top affinity scores of the experts they host. Within those 4 nodes the router picks its 8 experts. The constraint bounds the *cross-node* InfiniBand traffic specifically, which is the actual bottleneck, and lets DeepSeek overlap the cross-node all-to-all with computation cleanly.

Here is the device-limited selection as code — the two-stage "pick devices, then pick experts within them" pattern:

```python
def device_limited_topk(affinity, expert_to_device, n_devices, M=3, top_k=6):
    # affinity: [tokens, n_experts] router scores (softmax or sigmoid).
    # Stage 1: score each device by its top affinity mass, keep the best M.
    tokens, n_experts = affinity.shape
    dev_score = affinity.new_full((tokens, n_devices), float("-inf"))
    for e in range(n_experts):
        d = expert_to_device[e]
        dev_score[:, d] = torch.maximum(dev_score[:, d], affinity[:, e])
    keep_dev = dev_score.topk(M, dim=-1).indices            # [tokens, M]

    # Stage 2: mask out experts whose device was not kept, then top-k.
    dev_of_expert = expert_to_device.view(1, -1).expand(tokens, -1)
    on_kept = (dev_of_expert.unsqueeze(-1) == keep_dev.unsqueeze(1)).any(-1)
    masked = affinity.masked_fill(~on_kept, float("-inf"))
    topv, topi = masked.topk(top_k, dim=-1)                 # at most M devices touched
    return topv, topi
```

### A worked communication-cost example

Put numbers on why the limit matters. Take V3's setup: 8 experts activated per token, 256 experts spread over (say) 8 nodes, batch of $B$ tokens per step. Without any limit, a token's 8 experts are independently uniform over the 8 nodes, so the expected number of *distinct* nodes a token touches is $8 \cdot (1 - (1 - 1/8)^8) \approx 8 \cdot 0.66 \approx 5.3$ — nearly every token sprays across more than half the nodes. The cross-node all-to-all has to move roughly $B \times 5.3$ token-payloads over InfiniBand per step.

Now apply node-limited routing with $M = 4$. By construction every token touches at most 4 nodes, and because the limit picks the 4 *best* nodes (highest aggregate affinity) the experts cluster, so the realized average is well under 4. The cross-node volume drops from $\sim 5.3 B$ to $\le 4 B$ in the worst case and materially less in practice — and, just as important, the *worst case* is now bounded, which is what lets the scheduler overlap the all-to-all with compute deterministically. An unbounded worst case cannot be overlapped; you have to provision for the spike. The limit converts a probabilistic, spiky communication pattern into a bounded, schedulable one. That predictability is worth more than the average-case savings, because pipeline overlap (V3's DualPipe) only works if the communication volume per micro-step is known ahead of time.

### Second-order effect: routing topology is co-designed with the parallelism plan

The non-obvious consequence is that the routing constraint and the parallelism layout are the *same decision*. $M=3$ devices in V2 and $M=4$ nodes in V3 are not arbitrary — they are chosen against the cluster's communication hierarchy so that the all-to-all the router induces can be efficiently scheduled and overlapped with compute. V3 reports that node-limited routing lets it achieve near-complete computation-communication overlap, which is what makes the FP8 MoE training affordable on throttled hardware in the first place. You cannot design the router and the parallelism separately and bolt them together; the routing limit *is* the interface between them. This is the clearest example in the lineage of architecture being driven by the silicon it has to run on.

> The router is a scheduler in disguise. Every time it picks an expert, it is also picking a network transfer. Constrain the router and you constrain the network; leave it unconstrained and the network constrains you.

## 7. V2's three-tier balance losses: regularizing each scarce resource separately

**Senior rule of thumb: one balance loss balances one thing; if you have three scarce resources, you need three losses — or a smarter idea (see Section 8).**

Load balancing in MoE exists because routers collapse. Left to its own devices a learned router discovers a few "popular" experts and routes almost everything to them, starving the rest. Starved experts are dead capacity; popular experts are stragglers everyone waits on. The generic fix — covered in the [MoE primer](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) and in depth in [the MoE training and inference optimization post](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) — is an auxiliary load-balancing loss that penalizes uneven expert utilization. DeepSeekMoE already used two of these: an **expert-level** loss (spread tokens across the 64 experts) and a **device-level** loss (spread *compute* across device groups so no GPU is a straggler).

DeepSeek-V2 adds a third, because at V2's scale a new resource became scarce: the *bytes on the wire*. You can have perfectly balanced expert utilization and perfectly balanced per-device compute and *still* have an unbalanced all-to-all, because the volume of tokens each device has to *receive* depends on the routing pattern, not just the expert counts. So V2 runs three balance losses, each targeting a different resource.

![V2's three-tier balance losses: the router logits feed three separate balance terms that regularize expert utilization, per-device computation, and inter-device communication volume, summed with the language-model loss under small coefficients.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-7.webp)

The figure shows the three losses fanning out from the router's logits and merging into the total loss. Reading left to right: the **expert-level balance loss** ($\alpha_1 = 0.003$) spreads tokens across all 160 routed experts so none is dead; the **device-level balance loss** ($\alpha_2 = 0.05$) equalizes the computational load across device groups so none is a straggler; the **communication balance loss** ($\alpha_3 = 0.02$) equalizes the number of tokens *sent to* each device so the all-to-all has no hot spot. The total loss is

$$
\mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha_1\,\mathcal{L}_{\text{ExpBal}} + \alpha_2\,\mathcal{L}_{\text{DevBal}} + \alpha_3\,\mathcal{L}_{\text{CommBal}}
$$

Notice the coefficients are tiny and very different from each other: $0.003$, $0.05$, $0.02$. They are tiny because each balance loss is a *regularizer* — too strong and it overrides the language objective, forcing uniform routing that defeats the entire purpose of specialization. They are different because the three resources have different sensitivities. This is the hidden cost of the auxiliary-loss approach: you now have three coefficients to tune, each trading off balance against quality, each fighting the LM loss to some degree. Tuning them is a real engineering burden and a source of run-to-run variance.

Each balance loss follows the same template — a soft penalty proportional to the product of the fraction of tokens routed to a unit and the average gate weight that unit received:

```python
def balance_loss(gate_probs, assign_fraction, n_units, scale):
    # gate_probs:       [tokens, n_units] router softmax probabilities
    # assign_fraction:  [n_units] fraction of tokens whose top-k included this unit
    # The classic Switch-style aux loss: penalize correlation between how often a
    # unit is chosen and how confidently it is chosen. Minimized at uniform load.
    mean_prob = gate_probs.mean(dim=0)               # [n_units]
    loss = (assign_fraction * mean_prob).sum() * n_units
    return scale * loss


total = lm_loss \
    + balance_loss(p, frac_expert, n_experts=160, scale=0.003) \
    + balance_loss(p, frac_device, n_devices=8,   scale=0.05) \
    + balance_loss(p, frac_comm,   n_devices=8,   scale=0.02)
```

### V2's token-dropping safety valve, and why ~10% of sequences are exempt

Balance losses are soft — they make imbalance *cheaper to avoid*, but they cannot *guarantee* a balanced batch. So V2 pairs them with a hard mechanism: **device-level token dropping**. Each device has a capacity; if more tokens route to it than capacity allows, the lowest-affinity overflow tokens are dropped (they skip the MoE layer and pass through via the residual). This bounds the worst-case per-device load and keeps the all-to-all from blowing up on a bad batch.

But dropping tokens during training is dangerous if it is uniform, because it silently throws away signal and can correlate with content (always dropping the same hard tokens). V2's clever detail: it guarantees that the tokens belonging to **approximately 10% of training sequences are never dropped**, no matter what. Those sequences see the full, un-dropped MoE computation. This keeps a clean, drop-free signal flowing for a fraction of the data, which stabilizes training and reduces the train/inference mismatch (at inference there is no dropping). It is a small, pragmatic hedge: enough dropping to bound the cost, enough protected sequences to keep the gradient honest.

| Balance loss | What it equalizes | Coefficient | Failure it prevents |
|---|---|---|---|
| Expert-level | Tokens per expert (all 160) | $\alpha_1 = 0.003$ | Dead experts, collapsed router |
| Device-level | Compute per device group | $\alpha_2 = 0.05$ | Straggler GPU stalls the step |
| Communication | Tokens received per device | $\alpha_3 = 0.02$ | All-to-all hot spot, bandwidth stall |

### Why device balance and communication balance are genuinely different

The most common confusion here is "isn't balancing compute the same as balancing communication?" It is not, and the difference is worth pinning down because it justifies the third loss. Device-level balance is about *how much each GPU computes* — it counts the tokens each GPU's experts have to process. Communication balance is about *how much each GPU receives over the wire* — it counts the tokens that have to be *shipped to* each GPU before that compute can happen. These come apart whenever the routing pattern is asymmetric.

Concretely: imagine two GPUs, each hosting experts that end up processing exactly 1,000 tokens (perfect compute balance). But suppose GPU 0's experts are popular with tokens that *originate* on GPU 0 (no transfer needed), while GPU 1's experts are popular with tokens originating on GPU 0 (every one of GPU 1's tokens must cross the link). Compute is balanced — 1,000 each — but communication is wildly imbalanced: GPU 1 receives 1,000 transferred tokens and GPU 0 receives almost none. The all-to-all bottlenecks on GPU 1's inbound link while GPU 0's link sits idle. Device-level balance does nothing about this because it only looks at the compute totals, which are equal. You need a loss that specifically equalizes *received-token-count per device*, which is V2's communication balance term. This is the scenario behind case study 10: perfect expert and device balance, still bandwidth-bound, because the resource that was scarce — inbound bandwidth — had no loss watching it. The general principle is the lineage's recurring lesson in miniature: you must put a balance term on the *specific* resource you are bottlenecked on, and "balanced experts" or "balanced compute" does not automatically buy you "balanced bytes."

Three losses, three coefficients, three resources. It works — V2 is a strong model — but it is a lot of machinery, and every coefficient is a knob that can be wrong. That cost is exactly what V3 set out to delete.

## 8. V3's auxiliary-loss-free balancing: a control loop, not a loss term

**Senior rule of thumb: if a regularizer is fighting your main objective, take it out of the gradient and make it a feedback controller instead.**

Every auxiliary balance loss is a tax. It adds a term to the gradient whose only job is to make routing uniform — which is, by construction, in tension with the language objective, because the language objective *wants* the router to specialize. You tune the coefficient to a compromise: enough balance to avoid dead experts, little enough that you do not flatten specialization. There is no setting that is free of the trade-off; the balance gradient and the LM gradient are pulling in different directions, and the model pays for the conflict in quality.

DeepSeek-V3's insight is that you do not actually need a *gradient* to balance load. You need a *control signal*. Balancing is a feedback problem: measure each expert's recent load, and if an expert is overloaded, make it slightly less attractive; if underloaded, slightly more attractive. That is a thermostat, not a loss function — and a thermostat does not perturb the gradient at all.

![Auxiliary-loss tax versus V3's loss-free bias control loop: V3 adds a per-expert bias to the affinity score for top-k selection only, keeps the gating value derived from the raw affinity, and nudges the bias by a tiny step on observed load — so no balancing gradient ever touches the language objective.](/imgs/blogs/deepseek-moe-lineage-fine-grained-shared-experts-8.webp)

The figure contrasts the two regimes. On the left, the auxiliary-loss approach: the total loss is $\mathcal{L}_{\text{LM}} + \alpha\,\mathcal{L}_{\text{bal}}$, the balance gradient flows into the router's weights, it fights the LM loss, and you tune $\alpha$ by hand. On the right, V3's loss-free approach: the router computes a raw affinity score $s_i$ for each expert (a sigmoid in V3), and adds a per-expert **bias** $b_i$ to get the *selection* score $s_i + b_i$. The top-k is taken on $s_i + b_i$. Crucially, the *gating weight* applied to the chosen expert's output is still derived from the **raw** $s_i$, not $s_i + b_i$ — the bias only decides *who gets picked*, never *how much they count*. After each step, the controller looks at each expert's load: if an expert is overloaded, it decrements its bias by a fixed step $\gamma$; if underloaded, it increments by $\gamma$. V3 uses $\gamma = 0.001$.

$$
\text{select top-}k \text{ on } (s_{i} + b_{i}), \qquad
\text{gate weight} \propto s_{i}, \qquad
b_{i} \leftarrow b_{i} + \gamma \cdot \operatorname{sign}(\overline{\text{load}} - \text{load}_i)
$$

Because $b_i$ is updated by a hand-rolled rule and never appears in the loss, **no balancing gradient is ever produced.** The LM loss is the only thing the optimizer sees. The bias controller runs alongside, steering selection toward balance without taxing the objective. This is the headline of V3's balancing and the reason it can keep 256 experts per layer balanced without the quality hit a strong auxiliary loss would impose.

Here is the controller, which is genuinely about ten lines:

```python
class LossFreeBiasRouter:
    def __init__(self, n_experts, top_k=8, gamma=1e-3):
        self.bias = torch.zeros(n_experts)      # b_i, NOT a parameter, no grad
        self.top_k = top_k
        self.gamma = gamma

    def route(self, x, router):
        affinity = torch.sigmoid(router(x))     # s_i in [0, 1], per-expert
        select_score = affinity + self.bias     # bias only for SELECTION
        topv, topi = select_score.topk(self.top_k, dim=-1)
        # Gate weights come from the RAW affinity, not the biased score.
        gate = torch.gather(affinity, 1, topi)
        gate = gate / gate.sum(dim=-1, keepdim=True)
        return topi, gate, affinity

    @torch.no_grad()
    def update_bias(self, load_per_expert):     # called once per step
        mean = load_per_expert.float().mean()
        overloaded = load_per_expert > mean
        self.bias[overloaded] -= self.gamma     # make hot experts less attractive
        self.bias[~overloaded] += self.gamma    # make cold experts more attractive
```

V3 does keep one tiny auxiliary loss: a **complementary sequence-wise balance loss** with coefficient $\alpha = 0.0001$. It is two orders of magnitude smaller than V2's expert-level coefficient ($0.003$) and exists only to discourage extreme imbalance *within a single sequence* (a case the per-step bias controller, which operates over a batch, does not directly police). At $\alpha = 0.0001$ it is a whisper, not the V2-era shout — its contribution to the gradient is negligible compared to the LM loss, which is the entire design intent. (A scheduling detail: V3 runs $\gamma = 0.001$ for the first 14.3T tokens and then sets it to $0$ for the final 500B tokens, freezing the biases once the load distribution has stabilized.)

For the full treatment of how this interacts with FP8 GEMMs, Multi-Token Prediction, and the DualPipe schedule — and the measured quality delta versus an auxiliary-loss baseline — see [the dedicated DeepSeek-V3 deep-dive](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing). The short version: at 256 experts per layer the balancing tax of an aux-loss approach would be large, and removing it is a real, measured quality win, not a wash.

### Reading it as a control loop makes the design obvious

It pays to name the control-theory structure explicitly, because once you see it the design choices stop looking like tricks and start looking inevitable. The bias controller is an integral controller. The *plant* is the routing system; its *output* is per-expert load; the *setpoint* is uniform load ($\overline{\text{load}}$); the *error* is $\overline{\text{load}} - \text{load}_i$; and the *control action* is the bias $b_i$. The update $b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\text{error})$ accumulates error over time, which is exactly what an integrator does — and an integrator is the right choice here because it drives steady-state error to zero (an expert that is persistently overloaded keeps accumulating negative bias until it is no longer overloaded). The use of $\operatorname{sign}(\cdot)$ rather than the raw error makes it a *bang-bang-ish* integrator with a fixed step, which is more robust to the scale of the load signal than a proportional term would be — you do not have to tune the controller to the magnitude of the imbalance, only its direction.

This framing immediately explains the failure modes from the case studies. Too large a $\gamma$ (case study 6) is too high an integral gain: the controller overshoots and oscillates, exactly as an over-gained integrator does on any plant. The $\gamma$-to-zero schedule late in training is the standard move of freezing an integrator once the plant has settled, to stop it from chasing noise. And the reason this is *better* than an auxiliary loss is that the loss approach tries to solve the same control problem through the optimizer — it makes "imbalance" a cost the gradient descends — which couples the balance objective to the weight updates and forces the awkward coefficient trade-off. The controller decouples them: the optimizer optimizes language modeling, the controller handles balance, and they communicate only through the (gradient-free) bias. Separation of concerns, applied to a training loop.

### The measured tax: why deleting the loss is not just elegant

The argument so far is that the auxiliary loss *should* hurt because it fights the LM objective. Is the hurt measurable? Yes — and this is the empirical core of V3's contribution. The DeepSeek team ran the controlled comparison: same model, loss-free bias balancing versus a tuned auxiliary balance loss, all else equal. The loss-free variant achieves *better* downstream task performance at equal or better load balance. The mechanism is exactly the tension we described: the auxiliary loss, to achieve good balance, must perturb the router away from the routing it would choose on language-modeling grounds alone, and that perturbation is lost quality. The controller achieves the same balance without the perturbation, because it never touches the gradient. The size of the gap grows with the number of experts, which is why this technique was worth inventing precisely at the 256-expert scale and would have been marginal at DeepSeekMoE's 64. The bottleneck (the loss tax) became binding only after the expert count grew — the lineage pattern again.

### Second-order effect: why this needed the sigmoid gating change

Recall from Section 5 that V3 switched the router from softmax to sigmoid. This is not a coincidence; it is what makes the bias trick clean. Under a softmax, every expert's probability depends on every other expert's logit, so adding a bias $b_i$ to one expert renormalizes the whole distribution — the bias would leak into other experts' gate weights and you could not cleanly separate "selection" from "weighting." Under a per-expert sigmoid, each $s_i$ is independent, so you can add a bias purely for selection and keep the gate weight as the raw $s_i$, exactly as the design requires. The gating change and the balancing change are one co-designed decision. This is the same pattern as Section 6's routing-and-parallelism coupling: in a mature MoE, the pieces are not independent knobs but a single interlocking design.

## Case studies from production

These are composite incidents — the kind of thing you hit when you take the DeepSeek MoE ideas into your own training and serving stack. Each is concrete about the symptom, the wrong first guess, the real cause, and the fix.

### 1. The fine-grained router that collapsed in the first 2,000 steps

A team segmented their 8-expert MoE into 32 fine-grained experts (m=4) to chase the routing-freedom win, kept their old balance-loss coefficient, and watched the router collapse onto 5 experts within 2,000 steps — 27 experts effectively dead. The wrong first hypothesis was "fine-grained experts are unstable." The actual cause: with 4x more experts, the *same* auxiliary-loss coefficient is now spread 4x thinner per expert, so the per-expert balancing pressure dropped by 4x precisely when the larger pool needed *more* pressure to stay spread. The fix was to scale the balance coefficient up with the expert count (and, eventually, to move to a V3-style bias controller, which sidesteps the coefficient entirely). Lesson: when you change the expert count, the balance-loss coefficient is not a constant — it is coupled to N, and segmentation silently dilutes it.

### 2. The all-to-all that ate 60% of the step time

A 64-expert MoE on a 2-node, 16-GPU cluster trained fine on one node and fell off a cliff at two nodes — step time nearly tripled. Profiling blamed the experts; the GEMMs looked fine. The real cause was the all-to-all: with experts scattered across both nodes and no routing constraint, the average token's 8 experts spanned both nodes, so every token crossed the slow inter-node link. The fix was node-limited routing — cap each token's experts to a single node where possible, at most 2 nodes — which cut cross-node traffic by ~70% and brought step time back in line. Lesson: an unconstrained router on a multi-node cluster will route for quality and against your network; the routing limit is what reconciles them. This is exactly the V2→V3 device-to-node migration in miniature.

### 3. The shared expert that quietly carried the whole model

A team building an MoE-from-scratch ablated the shared expert "to save FLOPs" and saw a surprising 0.4 perplexity *regression* that no amount of router tuning fixed. The wrong guess was "we need more routed experts." The real cause: without an always-on shared expert, every routed expert had to independently learn the common-token baseline, so the routed pool wasted capacity on redundancy and the router had a harder separation problem (common vs specialized). Restoring a single shared expert recovered the regression and *improved* router entropy (the routed experts specialized more cleanly). Lesson: the shared expert is not overhead; it is the dense safety floor that lets the routed experts actually specialize. Removing it makes the routed pool worse, not just smaller.

### 4. The balance coefficient that flattened specialization

A team cranked the auxiliary balance-loss coefficient to "definitely fix the dead-expert problem" — set it to 0.1, an order of magnitude above DeepSeek-V2's device-level 0.05. Dead experts vanished; so did most of the model's quality. The router became nearly uniform, which is *perfectly balanced and perfectly useless* — uniform routing is mathematically equivalent to averaging all experts, i.e. a dense model with extra steps. The wrong hypothesis was "balance is always good." The real cause: a balance loss that dominates the LM loss forces uniform routing, destroying the specialization the MoE exists to provide. The fix was to back the coefficient down to the 0.003–0.05 range and add device-level token dropping as the hard safety valve, letting the soft loss stay gentle. Lesson: balance is a constraint, not an objective; over-satisfying it collapses the model into an expensive dense network.

### 5. The token-dropping that silently corrupted eval

A training run used aggressive device-level token dropping with no protected sequences and looked healthy — loss curve smooth, experts balanced. But downstream eval on long, hard documents was inexplicably weak. The cause took a week to find: the dropped tokens were not random. The hardest tokens (lowest router affinity) were dropped most often, and they clustered in exactly the long technical documents the eval cared about, so the model never learned them. The fix was V2's exact mechanism — guarantee that ~10% of sequences are never dropped — which restored a clean drop-free signal and recovered the eval. Lesson: token dropping is a load valve, not a free lunch; if you drop uniformly by affinity, you are systematically discarding your hardest training signal. Protect a slice of sequences.

### 6. The bias controller that oscillated

A team adopting V3-style loss-free balancing set the bias update step too aggressively ($\gamma = 0.01$, 10x V3's value) and watched expert load *oscillate* — an expert would go cold, get a big positive bias, suddenly become the most attractive expert, get slammed, get a big negative bias, go cold again. Classic control-loop instability from too high a gain. The wrong guess was "loss-free balancing does not work." The real cause: the controller's step size is the loop gain, and an over-large gain overshoots. Dropping $\gamma$ to V3's $0.001$ damped the oscillation and load settled. Lesson: the bias controller is a real feedback loop with real stability conditions; $\gamma$ is the gain and it wants to be small. Treat it like a control parameter, not a learning rate.

### 7. The sigmoid-vs-softmax mismatch that leaked the bias

A team ported the loss-free bias controller onto a *softmax* router (they had not changed the gating) and found the balance never improved much, while quality dropped slightly. The cause: under softmax, adding a bias $b_i$ to one expert's logit before the softmax shifts *every* expert's probability, so the bias contaminated the gate weights of unbiased experts and the "selection-only" property was violated. The fix was V3's exact choice — switch the router to per-expert sigmoid affinity so the bias affects only selection and the gate weight stays a clean function of the raw score. Lesson: the loss-free bias trick assumes independent per-expert scores; on a softmax router it leaks. The gating function and the balancing method are one decision, not two.

### 8. The 145B that matched a 67B dense model at a third of the compute

A scaling study reproduced DeepSeekMoE's headline claim and was initially disbelieved by the FLOPs-counting reviewers: a 145B-total fine-grained MoE matching a 67B dense model while using only ~28.5% of the per-token compute. The "wrong hypothesis" was reviewer-side — "the MoE must be undertrained or the eval is easy." The real explanation is the whole point of this article: the MoE's *capacity* scales with the 145B total parameters while its *cost* scales with the small activated subset, and fine-grained segmentation plus shared experts makes that capacity actually usable rather than redundant. The same study found a 16B fine-grained MoE matching LLaMA2-7B at ~40.5% of the compute. Lesson: for these models, "parameters" and "FLOPs" are different axes, and the fine-grained MoE recipe is specifically engineered to push capacity up the parameter axis while holding the FLOPs axis flat.

### 9. The expert-parallel layout that fought the routing limit

A team set up expert parallelism by sharding experts round-robin across GPUs, then bolted on device-limited routing, and got *worse* balance than no limit at all. The cause was a layout/limit mismatch: the round-robin sharding scattered "naturally co-activated" experts across many devices, so the device-limit's "pick the best M devices" stage kept having to drop high-affinity experts to satisfy the constraint, hurting both quality and balance. The fix was to co-design the layout with the limit — place experts so that frequently co-activated ones share a device — which let the device-limit keep high-affinity experts together. Lesson: the routing limit and the expert placement are the same decision (Section 6); pick the placement to make the limit cheap, not to make the limit fight you.

### 10. The MoE that was perfectly balanced and still bandwidth-bound

A serving deployment achieved beautiful expert-level balance (every expert saw ~1/N of tokens) and was still bottlenecked on the all-to-all. The team was baffled: balance was perfect, so why the stall? The cause was the third resource V2's communication loss exists for — *expert utilization* was balanced but *bytes received per device* was not, because a few devices hosted the experts that happened to receive larger token payloads. Expert balance and communication balance are different quantities. The fix was to add a communication-volume balance term (V2's $\alpha_3$) targeting received-bytes-per-device, which evened out the all-to-all. Lesson: "balanced experts" does not imply "balanced communication"; at scale you must balance the resource you are actually bottlenecked on, which is often bytes, not token counts.

### 11. The single shared expert that became a precision bottleneck

A V3-style deployment with one shared expert per layer found that expert disproportionately sensitive to quantization — int8-ing it tanked quality far more than int8-ing any routed expert. The cause: the single shared expert is on the path of *every* token, so its error is not averaged away across a sparse selection the way a routed expert's is — it is a systematic bias on every output. The fix was to keep the shared expert in higher precision (BF16) while quantizing the routed experts more aggressively. Lesson: the shared expert's always-on nature makes it a precision-critical component; it carries more of the model's behavior per parameter than any routed expert, so it earns more bits. The same logic that made it valuable (Section 2) makes it fragile under quantization.

### 12. The router warmup that prevented early collapse

A from-scratch fine-grained MoE kept collapsing in the first few thousand steps regardless of balance-loss tuning, because at initialization the router's scores are near-random and a few experts win the early lottery and snowball. The wrong fix was more balance loss (which flattened specialization, see case 4). The real fix was a *router warmup*: for the first N steps, add noise to the router logits (or use a higher balance coefficient that decays), so no expert can snowball before the experts have differentiated. Once the experts have learned distinct functions, the snowball risk drops and the warmup can decay. Lesson: router collapse is mostly an *early-training* phenomenon driven by the initialization lottery; a time-limited warmup beats a permanently strong balance loss, because it stabilizes the start without taxing the whole run. This rhymes with V3's $\gamma$ schedule (strong early, off late) — both treat balancing pressure as something you need most at the start and can relax once load has stabilized.

## When to reach for the DeepSeek MoE recipe — and when not to

The lineage is a menu, not a monolith. You can adopt the parts that fit your constraints, and most teams should — very few are training at a scale where they need all four ideas at once, but almost every MoE benefits from at least the first two. The decision is not "DeepSeek's whole stack or nothing"; it is "which scarcity am I actually facing, and which piece of the lineage addresses it." Here is how to choose.

**Reach for fine-grained segmentation when:**

- You are FLOPs-constrained but have parameter and memory headroom — segmentation buys routing freedom at fixed activated FLOPs, which is the best trade in MoE.
- Your tasks are heterogeneous enough that fine specialization helps (multi-domain, multilingual, code-plus-prose). The combinatorial routing space (Section 3) only pays off if the data has structure to exploit.
- You can afford to scale the balance pressure with the expert count (case study 1) — segmentation dilutes per-expert balancing, so you must compensate.

**Reach for shared-expert isolation when:**

- A large fraction of your tokens need the same baseline knowledge (almost always true for general LLMs). The shared expert removes that redundancy from the routed pool.
- You want graceful degradation under imperfect routing — the always-on path is a dense safety floor (Section 2, case study 3).
- You are willing to keep the shared expert in higher precision when quantizing (case study 11); it is precision-critical.

**Reach for device/node-limited routing when:**

- You train or serve MoE across multiple GPUs or nodes and the all-to-all is a measurable fraction of step time (case study 2). On a single GPU, there is nothing to limit.
- Your interconnect is heterogeneous (fast intra-node NVLink, slow inter-node InfiniBand) — match the limit to the *slow* tier (node-limited on multi-node, device-limited intra-node).
- You can co-design expert placement with the limit (case study 9); the limit fights a bad layout.

**Reach for auxiliary-loss-free (bias-controller) balancing when:**

- You have many experts per layer (tens to hundreds) where the auxiliary-loss tax is large and you want the quality back.
- You are using or can switch to per-expert sigmoid gating (case study 7) — the bias trick leaks on softmax.
- You can treat $\gamma$ as a control gain and keep it small (case study 6) — it is a feedback loop with stability conditions.

**Skip or simplify when:**

- You have a single GPU or a single node with fast interconnect — routing limits and communication balance losses solve a problem you do not have. Use a plain expert-level balance loss and move on.
- Your model is small (a few experts) — fine-grained segmentation's combinatorial win is marginal at small N, and the router-collapse risk and balance-dilution cost are not worth it. The crossover is roughly "do you have enough experts that $\binom{N}{K}$ is already large?"
- You are fine-tuning a pre-trained dense model and do not control the architecture — none of this applies; you are not designing the MoE, you are using one.
- Your data is homogeneous — if every token needs the same processing, specialization buys little and a dense model (or a small MoE) is simpler and just as good.
- You need maximum reproducibility and minimum moving parts for a research baseline — the auxiliary-loss approach, for all its tax, is simpler to reason about and debug than a feedback controller. Start there, then graduate to loss-free once the baseline is solid.

The throughline across the whole lineage is a single discipline: **always be attacking the resource that is currently scarce, and stop spending effort on the one you already fixed.** DeepSeek made experts fine-grained when routing freedom was scarce, bounded communication when bandwidth was scarce, and deleted the balance loss when the loss budget was scarce. If you adopt the recipe, adopt that discipline first — profile to find your binding constraint, apply the matching piece of the lineage, then re-profile, because the fix will have moved the bottleneck somewhere new. That migration *is* the architecture.

## Further reading

- [DeepSeek-V3: FP8, Multi-Token Prediction, and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the full V3 training recipe, with the measured quality delta of loss-free balancing versus an auxiliary-loss baseline and how it co-designs with FP8 and DualPipe.
- [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) — the systems side: all-to-all kernels, expert parallelism, grouped GEMMs, capacity factors, and how to actually make a sparse model fast.
- [MoE architecture, training, and fine-tuning case studies](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) — the general MoE primer if you want the foundational gating, top-k, and auxiliary-loss mechanics this post assumes.
- [Multi-head Latent Attention (MLA)](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — DeepSeek's other co-designed architectural innovation, the attention-side counterpart to the MoE story told here; the two together are what make V2 and V3 affordable.
- DeepSeekMoE (arXiv 2401.06066), DeepSeek-V2 (arXiv 2405.04434), and DeepSeek-V3 (arXiv 2412.19437) — the three primary sources; read them in order to see the bottleneck migration first-hand.
