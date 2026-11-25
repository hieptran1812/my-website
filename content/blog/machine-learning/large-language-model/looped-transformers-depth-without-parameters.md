---
title: "Looped Transformers: Buying Depth Without Parameters, From Universal Transformers to Inkling 2 Mini and GLM-5.4"
date: "2026-11-25"
description: "Why two flagship releases in one week run their own layers more than once, the theory that says depth and parameters are different budgets, the papers that made looping trainable, and the latency, KV-cache and monitorability bill you pay for it."
tags: ["looped-transformer", "recurrent-depth", "universal-transformer", "inkling-2-mini", "glm-5-4", "huginn", "ouro", "mixture-of-recursions", "latent-reasoning", "llm-architecture"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 12
---

In the week of 20 November 2026, two flagship releases shipped with the same unusual property: they run some of their own layers more than once per token. Thinking Machines' Inkling 2 Mini stores 24 layers and runs the middle 12 three times. Z.ai describes GLM-5.4 as "a group of weight-shared layers" that "runs for multiple iterations", producing "a deeper network without increasing the parameter count." Four months earlier, Nanbeige4.2-3B had already pushed its entire 22-layer stack through twice.

None of this is new as an idea. Universal Transformers proposed it in 2018. What changed is that the people paying for frontier training runs now think it is worth the price. This post walks the theory that justifies looping, the papers that made it trainable, the 2026 models that ship it, and the bill.

![A standard 48-layer stack versus Inkling 2 Mini's 24 stored layers with a looped core](/imgs/blogs/looped-transformers-depth-without-parameters-1.webp)

The picture above is the whole idea. Both rows apply 48 layers to every token, so they spend roughly the same compute. The bottom row stores half the weights.

## Depth and parameters are two different budgets

A dense transformer couples two things that do not have to be coupled. Every layer you add brings new weights (memory, knowledge capacity) and one more sequential step of computation (depth). Most scaling work treats those as one knob. Looping separates them.

The cleanest argument for why you would want to separate them comes from Saunshi et al. (ICLR 2025), whose abstract states it in one line: "many reasoning problems require a large depth but not necessarily many parameters." They show that a $k$-layer block looped $L$ times nearly matches a $kL$-layer model on addition, $p$-hop induction and math word problems, while far outperforming the plain $k$-layer model. They also prove that a looped model can simulate $T$ steps of chain-of-thought with $T$ loops. That is the formal version of "latent thoughts": the reasoning that would otherwise be written as tokens happens inside the hidden state.

Formally, write the model as a prelude $P$, a shared core $C_\theta$, and a coda $D$. With token embeddings $e \in \mathbb{R}^{n \times d}$ for $n$ tokens of width $d$:

$$
h_0 = P(e), \qquad h_t = C_\theta\big(\,\phi(h_{t-1},\, h_0)\,\big) \;\; \text{for } t = 1, \dots, T, \qquad y = D(h_T)
$$

where $h_t \in \mathbb{R}^{n \times d}$ is the hidden state after pass $t$, $T$ is the number of passes, and $\phi$ is the re-injection rule that decides how much of the original input to mix back in on every pass. The parameter count is $|P| + |C_\theta| + |D|$, independent of $T$. The compute is not:

$$
\text{FLOPs per token} \;\approx\; 2\big(|P| + T\,|C_\theta| + |D|\big), \qquad \text{stored weights} = |P| + |C_\theta| + |D|
$$

Plug in Inkling 2 Mini's layout: $P$ and $D$ are 6 layers each, $C_\theta$ is 12, and $T = 3$. That gives $6 + 3 \times 12 + 6 = 48$ layer applications from 24 stored layers. The ratio of compute to memory doubled with no new weights.

The flip side is in the same Saunshi paper. Looping helps *reasoning*, but stored *knowledge* still tracks parameter count. Every later result in this post fits that split.

## The lineage: from proofs to working models

**Universal Transformer (Dehghani et al., 2018).** One self-attention block applied repeatedly over depth, with adaptive computation time letting each position decide when to halt. It gained 0.9 BLEU over a standard Transformer on WMT14 English-German and set a state of the art on LAMBADA. It never became the default, because at the time nobody needed depth more than they needed parameters.

**ALBERT (Lan et al., 2019)** shared weights across all layers to shrink BERT. It is a useful counterexample: sharing saved memory, but every layer still cost full FLOPs, so it was a compression trick, not a reasoning one.

**Looped Transformers as Programmable Computers (Giannou et al., 2023)** showed by construction that a 13-layer transformer, run in a loop over a "punchcard" input holding instructions and memory, can emulate a small instruction-set computer. **Yang et al. (ICLR 2024)** then showed looped transformers *learn* in-context regression as well as standard transformers with under 10% of the parameters. Both are toy settings. Both said the same thing: iteration is the natural shape of an algorithm.

**Huginn (Geiping et al., February 2025)** was the first convincing language model. It uses a (2, 4, 2) prelude, recurrent and coda layout, 3.5B parameters, and 800B training tokens. The embedding is concatenated back into the state on every iteration (concatenation beat addition). The recurrence count is sampled per micro-batch from a heavy-tailed distribution with mean 32, and gradients flow only through the last 8 iterations. At 32 recurrences the effective depth is 132 layers, and the authors report test-time compute equivalent to a 50-billion-parameter model.

![Huginn ARC-Easy accuracy as recurrence steps grow](/imgs/blogs/looped-transformers-depth-without-parameters-2.webp)

The two measured points are the story: ARC-Easy goes from 49.07 at 4 recurrences to 69.91 at 32. On GSM8K, Huginn scored 42.08 against OLMo-7B's 7.28. Easy tasks saturate early, and closed-book QA stays weak because 3.5B parameters can only hold so many facts.

**Ouro (ByteDance Seed, October 2025)** scaled the recipe to 7.7T tokens with 1.4B and 2.6B models running 4 recurrent steps, plus an entropy-regularized objective that teaches the model how deep to go. The authors claim parity with models up to 12B and, importantly, attribute the gain to better *knowledge manipulation*, not more capacity. **Relaxed Recursive Transformers** and **Mixture-of-Recursions** (Bae et al., 2024 and 2025) came at it from the efficiency side. MoR adds a small router that gives each token its own recursion depth and attends only over tokens still active, reaching up to 2.06x throughput and matching a vanilla model with roughly half the non-embedding parameters (167M versus 315M).

## The 2026 models that ship it

### Inkling 2 Mini

Thinking Machines released Inkling 2 Mini on 20 November under Apache-2.0: an MoE with 26.4B total and 3.4B active parameters. The blog's 5.4B "activated" figure counts weight activations per token *including* the loops. The model card config states the loop directly: `core_start 6, core_end 18, num_passes 3`.

<figure class="blog-anim">
<svg viewBox="0 0 720 280" role="img" aria-label="A hidden-state token enters the prelude, runs three passes through one shared core block, and exits through the coda. After each pass the output loops back through a gate that is closed on pass 1, half open on pass 2 and wide open on pass 3." style="width:100%;height:auto;max-width:820px">
<style>
.l1-blk{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.l1-core{fill:var(--surface,#f3f4f6);stroke:var(--text-primary,#1f2937);stroke-width:2}
.l1-glow{fill:var(--accent,#6366f1);opacity:0}
.l1-wire{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2}
.l1-loop{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:6 5}
.l1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.l1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.l1-cnt{font:700 16px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
.l1-door{fill:var(--text-primary,#1f2937)}
.l1-tok{fill:var(--accent,#6366f1);stroke:var(--background,#fff);stroke-width:2}
.l1-msg{fill:var(--background,#fff);stroke:var(--accent,#6366f1);stroke-width:3;opacity:0}
.l1-p2,.l1-p3,.l1-g2,.l1-g3{opacity:0}
@keyframes l1-tokmove{0%{transform:translate(-320px,0);opacity:0}3%{transform:translate(-300px,0);opacity:1}10%{transform:translate(0,0)}78%{transform:translate(0,0)}88%{transform:translate(270px,0);opacity:1}96%{transform:translate(310px,0);opacity:1}100%{transform:translate(310px,0);opacity:0}}
@keyframes l1-msgmove{0%,29%{transform:translate(450px,150px);opacity:0}30%{transform:translate(450px,150px);opacity:1}31.5%{transform:translate(490px,150px)}33%{transform:translate(490px,60px)}35%{transform:translate(312px,60px);opacity:1}36%{transform:translate(290px,60px);opacity:.4}37.5%{transform:translate(230px,60px);opacity:.4}39%{transform:translate(230px,150px);opacity:.4}40%{transform:translate(270px,150px);opacity:0}53%{transform:translate(450px,150px);opacity:0}54%{transform:translate(450px,150px);opacity:1}55.5%{transform:translate(490px,150px)}57%{transform:translate(490px,60px)}59%{transform:translate(312px,60px);opacity:1}60%{transform:translate(290px,60px);opacity:1}61.5%{transform:translate(230px,60px)}63%{transform:translate(230px,150px);opacity:1}64%,100%{transform:translate(270px,150px);opacity:0}}
@keyframes l1-glowp{0%,11%{opacity:0}14%,28%{opacity:.28}31%,37%{opacity:0}40%,52%{opacity:.28}55%,61%{opacity:0}64%,76%{opacity:.28}80%,100%{opacity:0}}
@keyframes l1-doortop{0%,31%{transform:translateY(0)}33%,55%{transform:translateY(-9px)}57%,92%{transform:translateY(-22px)}97%,100%{transform:translateY(0)}}
@keyframes l1-doorbot{0%,31%{transform:translateY(0)}33%,55%{transform:translateY(9px)}57%,92%{transform:translateY(22px)}97%,100%{transform:translateY(0)}}
@keyframes l1-s1{0%,32%{opacity:1}34%,97%{opacity:0}100%{opacity:1}}
@keyframes l1-s2{0%,32%{opacity:0}34%,56%{opacity:1}58%,100%{opacity:0}}
@keyframes l1-s3{0%,56%{opacity:0}58%,95%{opacity:1}98%,100%{opacity:0}}
.l1-tok{animation:l1-tokmove 16s ease-in-out infinite}
.l1-msg{animation:l1-msgmove 16s linear infinite}
.l1-glow{animation:l1-glowp 16s ease-in-out infinite}
.l1-dt{animation:l1-doortop 16s ease-in-out infinite}
.l1-db{animation:l1-doorbot 16s ease-in-out infinite}
.l1-p1,.l1-g1{animation:l1-s1 16s ease-in-out infinite}
.l1-p2,.l1-g2{animation:l1-s2 16s ease-in-out infinite}
.l1-p3,.l1-g3{animation:l1-s3 16s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.l1-tok,.l1-msg,.l1-glow,.l1-dt,.l1-db,.l1-p1,.l1-p2,.l1-p3,.l1-g1,.l1-g2,.l1-g3{animation:none}}
</style>
<path class="l1-wire" d="M10 150 H30"/>
<path class="l1-wire" d="M150 150 H270"/>
<path class="l1-wire" d="M450 150 H570"/>
<path class="l1-wire" d="M690 150 H712"/>
<path class="l1-loop" d="M450 150 H490 V60 H230 V150"/>
<rect class="l1-blk" x="30" y="110" width="120" height="80" rx="8"/>
<rect class="l1-core" x="270" y="100" width="180" height="100" rx="10"/>
<rect class="l1-glow" x="270" y="100" width="180" height="100" rx="10"/>
<rect class="l1-blk" x="570" y="110" width="120" height="80" rx="8"/>
<text class="l1-lbl" x="90" y="146">prelude</text>
<text class="l1-sub" x="90" y="166">L1-6, runs once</text>
<text class="l1-lbl" x="360" y="140">shared core</text>
<text class="l1-sub" x="360" y="160">L7-18, one weight set</text>
<text class="l1-lbl" x="630" y="146">coda</text>
<text class="l1-sub" x="630" y="166">L19-24, runs once</text>
<text class="l1-sub" x="410" y="50">loop back: previous pass</text>
<rect class="l1-door l1-dt" x="295" y="34" width="10" height="26" rx="2"/>
<rect class="l1-door l1-db" x="295" y="60" width="10" height="26" rx="2"/>
<text class="l1-sub" x="262" y="44">gate</text>
<circle class="l1-msg" cx="0" cy="0" r="7"/>
<circle class="l1-tok" cx="360" cy="150" r="9"/>
<text class="l1-cnt l1-p1" x="360" y="236">pass 1/3</text>
<text class="l1-cnt l1-p2" x="360" y="236">pass 2/3</text>
<text class="l1-cnt l1-p3" x="360" y="236">pass 3/3</text>
<text class="l1-sub l1-g1" x="360" y="262">gate closed: the loop message is blocked, the core acts like a plain layer stack</text>
<text class="l1-sub l1-g2" x="360" y="262">gate half open: part of the previous pass is let back in</text>
<text class="l1-sub l1-g3" x="360" y="262">gate wide open: the full previous pass feeds the next one</text>
</svg>
<figcaption>One shared core, three passes: the gate starts closed so a looped model begins life as a plain one, then learns how much of the previous pass to let back in.</figcaption>
</figure>

Two design choices stand out. First, re-injection: every pass starts from the prelude output, and passes 2 and 3 also read the previous pass's output, merged through a learned gate. That gate is **initialized fully closed**, so on day one each loop is a plain non-looping model and the model learns how much of its previous pass to trust. Second, it replaces the single summed residual stream with attention residuals, where each layer reads a learned mix of earlier blocks' outputs. In a looped network that means a layer can effectively read its own output from a previous pass, which is why interpretability researchers with early access started mapping how information moves across layers, blocks and loops.

The ablation that matters most is *where* to loop:

![Looping the middle of the stack versus the end](/imgs/blogs/looped-transformers-depth-without-parameters-3.webp)

Looping the middle layers gave 1.51x to 1.91x compute-efficiency gains versus 1.24x to 1.36x for looping the last layers, and won 27 of 28 evals. That matches the intuition from interpretability work on standard LLMs: early layers parse, late layers format the answer, and the middle is where the composition happens.

The headline numbers against Qwen3.8-27B: SWE-bench Verified 79.2 versus 77.8, BrowseComp 79.0 versus 61.6, and a tie on the Artificial Analysis index at 20.1. The non-looped twin used 7.5% less training compute, but by the blog's scaling fit a non-looped model needs more than 1.5x the compute to reach the same held-out loss. The full-loop model shipped natively in NVFP4 for expert layers, about 20GB, and runs on a 32GB Mac.

### Nanbeige4.2-3B

Nanbeige4.2-3B (July 2026, 28T tokens from scratch) is the simplest version: the whole 22-layer stack, run twice, for 44 effective layers. The paper's reasoning for stopping at two is the most useful practical data point in this post: two loops keep about 75% of a standard transformer's token efficiency, while more passes give "only marginal additional improvement, but substantially slows training and makes optimization less stable." Training from scratch beat upcycling a pretrained model. Each pass keeps its own KV cache; sharing it across passes halved memory but was "consistently worse." The payoff is strong for a 3B model: SWE-bench Verified 63.6 versus 53.1 for Qwen3.5-9B.

### GLM-5.4

GLM-5.4 (23 November) is the first flagship-scale loop. Z.ai's official post renders only with JavaScript, so the details below come from press coverage of it, chiefly VentureBeat. Treat them as reported, not verified. Reportedly, 28 layers of the GLM-5 base are looped twice at roughly 750B scale, added to an already pretrained model through continued training. Three cost controls are described: per-token gated progressive updates (pass 1 writes conservatively, pass 2 opens the gates), a hybrid KV cache where pass 2 reuses pass 1's sparse cache plus a sliding window, and reuse of pass 1's sparse-attention indexer outputs.

The "loop depth versus model size" evidence is from a 30B ablation, not GLM-5.4 itself: about 10% more training compute than a same-size baseline, with the largest gains in math, science and long context, and shorter outputs. GLM-5.4 has published no standard benchmark table, and its "comparable to 2T to 3T parameter models" claim is not explained. Z.ai concedes it "can get caught in overly long reasoning loops."

### Astra: the rumor that started the wave

The Information reported on 1 September, from one anonymous source, that OpenAI's GPT-6 Astra uses a "constrained form of recurrent depth." No system card, paper or code confirms it. OpenAI's Jakub Pachocki said on record that the computation depth of its frontier models, Astra included, is "within a factor of two of GPT-4," and called the coverage confused. Sebastian Raschka's read is that looping is plausible but the gains are mostly data and recipe. A rumor can still move a field: several of the 2026 papers below cite it.

## Making it train

A looped network is a very deep network with tied weights, and both halves of that sentence cause instability. The same weights receive gradients from every pass, and the unrolled depth makes residual streams explode or collapse. The fixes converge on a short list:

| Trick | Where | What it prevents |
| --- | --- | --- |
| Gate initialized closed | Inkling 2 Mini | Early training sees a plain model; loop influence grows gradually |
| Sandwich norm + truncated backprop (last 8 iterations) | Huginn | Hidden-state collapse; model learning to ignore recurrence |
| Random recurrence count per batch | Huginn, Inkling varloop | Overfitting to one depth; enables test-time depth choice |
| Residual scaling derived for tied depth | DeepLoop, Loop-µP | DeepNorm's untied-depth assumption breaking |
| Stop at 2 to 3 loops | Nanbeige, GLM-5.4, Inkling | Optimization instability beyond that point |

DeepLoop (Princeton and UCLA, July 2026) is the most principled entry: it derives a stability condition for $K$ blocks visited over $R$ rounds and sets Post-LN scaling to

$$
\alpha = (2N)^{1/2}, \qquad \beta = (8N)^{-1/2}, \qquad N = K \cdot R
$$

where $N$ is the unrolled depth, $\alpha$ scales the residual branch and $\beta$ scales the initialization. It is neutral at $R = 1$ and lowers validation loss by 0.023 nats at $R = 5$, though only on GPT-2 small and medium with single seeds. A rival, Loop-µP, argues $1/N$ scaling beats $1/\sqrt{N}$. The question is not settled.

A minimal looped core, to make the moving parts concrete (my sketch, not any lab's code):

```python
import torch
import torch.nn as nn

class LoopedCore(nn.Module):
    """Prelude -> shared core run T times -> coda, with a closed-at-init gate."""

    def __init__(self, prelude: nn.Module, core: nn.Module, coda: nn.Module,
                 d_model: int, passes: int = 3, gate_rank: int = 64):
        super().__init__()
        self.prelude, self.core, self.coda = prelude, core, coda
        self.passes = passes
        # low-rank gate on the previous pass's message; zero-init = closed
        self.down = nn.Linear(d_model, gate_rank, bias=False)
        self.up = nn.Linear(gate_rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor, passes: int | None = None) -> torch.Tensor:
        h0 = self.prelude(x)                 # [batch, seq, d_model]
        h = self.core(h0)                    # pass 1: no message yet
        for _ in range((passes or self.passes) - 1):
            message = self.up(self.down(h))  # exactly 0 at init
            h = self.core(h0 + message)      # re-inject prelude output every pass
        return self.coda(h)

if __name__ == "__main__":
    d = 256
    block = lambda: nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True)
    model = LoopedCore(block(), nn.Sequential(block(), block()), block(), d)
    out = model(torch.randn(2, 16, d), passes=5)  # choose depth at inference
    print(out.shape)  # torch.Size([2, 16, 256])
```

At initialization `message` is zero, so every pass computes the same function of `h0` and the model behaves like a non-looped one. Training opens the gate only as far as it helps.

## The bill

![Per-token time lanes: standard stack, three-pass loop, and GLM-5.4's cheaper second pass](/imgs/blogs/looped-transformers-depth-without-parameters-4.webp)

**Latency is not free.** Every pass is a full sequential trip through the core. Inkling's 48 layer applications take about as long as a 48-layer model, and looping only beats a dense model on speed if you are memory-bandwidth bound and the smaller weight set fits where the larger one would not. That is the laptop argument: 20GB of weights doing 48 layers of work.

**The KV cache does not shrink by default.** Each pass writes its own keys and values; Nanbeige measured that sharing them across passes hurts. The 2026 workarounds all reuse something: Huginn shares cache slot $i \bmod k$, MoR reuses the first recursion's cache, and GLM-5.4 reportedly gives pass 2 only a sliding window on top of pass 1's sparse cache. If you serve these models, read our [KV cache primer](/blog/machine-learning/large-language-model/kv-cache) and the [hybrid KV cache manager](/blog/machine-learning/model-serving/hybrid-kv-cache-manager) post with "per pass" in mind. Tooling lags: a study of Nanbeige on Apple Silicon found five deployment bugs, and upstream llama.cpp did not support it at release.

**Knowledge does not loop.** Huginn is weak at closed-book QA, Inkling's factuality tracks its 26.4B total parameters, and Ouro attributes its gains to manipulation, not capacity. Looping makes a model think harder about what it knows. It does not make it know more.

**Out-of-distribution composition is fragile.** DiscoLoop (NeurIPS 2026) found that in multi-hop composition the intermediate answer is decodable after the first loop but not cleanly carried in the continuous state, so looped models collapse out of distribution. Its fix feeds back a prediction-weighted soft token embedding at each handoff, in effect re-discretizing between loops.

**Reasoning moves off the page.**

![Chain-of-thought tokens and no-CoT time horizon as passes increase](/imgs/blogs/looped-transformers-depth-without-parameters-5.webp)

Thinking Machines' variable-loop research model makes this measurable. On GPQA-Diamond, mean chain-of-thought length fell from 3,703 tokens at 1 pass to 1,661 at 5. The length of task it can solve with no chain-of-thought at all rose from 16 seconds at 1 pass to 37 seconds at 3, then flattened. On N-hop lookup, accuracy kept improving up to 10 loops despite training only up to 5. The lab reports "no meaningful reduction in monitorability" at this scale, and Mira Murati added that this "may not stay true." That is the right level of confidence: every token of reasoning that moves into the loop is a token a CoT monitor no longer reads.

## When to reach for looping, and when not to

**Reach for it when** your workload is reasoning-heavy rather than recall-heavy (code, math, multi-step tool use), when you are memory-bound (on-device, a single GPU, NVFP4 weights) and can trade latency for footprint, or when you want a test-time knob: Huginn and Inkling's research model let you choose depth per request.

**Avoid it when** you need factual breadth from a small model, when tail latency is your SLO, when your serving stack cannot yet handle per-pass KV caches, or when your safety case rests on reading the chain of thought. And keep the loop count small. Every production model in this post stops at two or three passes, and Nanbeige's paper explains why: past that, training slows and destabilizes for marginal gains.

The bet these labs are making is the one Saunshi wrote down: depth is cheap to add, parameters are expensive to store, and for reasoning you mostly need the former. The open questions are adaptive depth per token in production (GLM-5.4 lists it as future work) and whether monitorability survives at frontier scale. For a related lesson about what survives when you blend capabilities into one set of weights, see how [Index-Translate handles merging specialists](/blog/paper-reading/large-language-model/index-translate).

## References

- Thinking Machines. *Inkling 2 Mini.* 2026-11-20. https://thinkingmachines.ai/news/inkling-2-mini/ ; config at https://huggingface.co/thinkingmachines/Inkling-2-Mini
- Z.ai. *GLM-5.4.* 2026-11-23. https://z.ai/blog/glm-5.4 ; coverage: https://simonwillison.net/2026/Nov/23/glm-54/ and VentureBeat.
- Nanbeige Lab. *Nanbeige4.2-3B.* arXiv:2607.22083, 2026.
- Saunshi et al. *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* arXiv:2502.17416, ICLR 2025.
- Geiping et al. *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* arXiv:2502.05171, 2025.
- Zhu et al. *Scaling Latent Reasoning via Looped Language Models (Ouro).* arXiv:2510.25741, 2025.
- Dehghani et al. *Universal Transformers.* arXiv:1807.03819, ICLR 2019.
- Giannou et al. *Looped Transformers as Programmable Computers.* arXiv:2301.13196, 2023.
- Yang et al. *Looped Transformers are Better at Learning Learning Algorithms.* arXiv:2311.12424, ICLR 2024.
- Bae et al. *Relaxed Recursive Transformers* (arXiv:2410.20672) and *Mixture-of-Recursions* (arXiv:2507.10524).
- Li et al. *DeepLoop.* arXiv:2607.13491, 2026. Fu et al. *DiscoLoop.* arXiv:2607.00341, NeurIPS 2026.
- Raschka, S. *GPT-6 Astra, looped transformers and ...* https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and
