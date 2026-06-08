---
title: "Kimi K2: Open Agentic Intelligence — How a Trillion-Parameter MoE Learns to Use Tools"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - kimi-k2
  - moonshot-ai
  - mixture-of-experts
  - muonclip
  - agentic-llm
  - reinforcement-learning
  - tool-use
  - mla-attention
  - open-weights
description: "A deep read of Kimi K2 — the 1.04T-parameter, 32.6B-activated open-weight MoE that hits state-of-the-art agentic and coding scores using the MuonClip optimizer, large-scale agentic data synthesis, and a single joint RL stage."
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/kimi-k2-1.png"
readTime: 30
---

Most "frontier open model" releases are a benchmark table and a weight dump. Kimi K2 is more interesting than that, because the paper is honest about the two things that actually gate agentic intelligence — and neither of them is parameter count. The first is whether you can train a trillion-parameter model *at all* without the loss blowing up. The second is where the multi-step, tool-calling, recover-from-your-own-mistakes data comes from, given that almost none of it exists in web text.

Kimi K2 is Moonshot AI's answer to both. It is a **1.04-trillion-parameter Mixture-of-Experts model that activates only 32.6B parameters per token**, pre-trained on 15.5T tokens with the new **MuonClip** optimizer (Muon plus a per-head clipping trick), then post-trained with a **large-scale agentic data synthesis pipeline** and a **single joint reinforcement-learning stage**. The headline is that among *non-thinking* models — no extended chain-of-thought, outputs capped at 8,192 tokens — it beats DeepSeek-V3, Qwen3, and frequently Claude and GPT-4.1 on coding, math, and tool-use benchmarks.

![Kimi K2 training lifecycle](/imgs/blogs/kimi-k2-1.png)

The diagram above is the mental model for the whole paper: a single linear chain from MuonClip pre-training, through long-context extension and supervised fine-tuning, into an agentic-data-synthesis stage that manufactures tool-use trajectories, and finally a joint RL stage that polishes everything at once. There is no zoo of separate specialist models — the entire agentic capability is grown in one pipe. This post walks that pipe end to end, with the numbers, the failure modes, and the parts I think will and won't transfer to your own training runs.

> [!tldr] TL;DR
> - **What it claims:** A 1.04T-parameter (32.6B-activated) open-weight MoE can reach state-of-the-art *agentic* performance — coding, tool use, math — among non-thinking models, beating DeepSeek-V3 and Qwen3 outright and trading blows with Claude Opus 4 and GPT-4.1.
> - **Why it matters:** It ships two genuinely reusable ideas — **MuonClip** (a fix that makes the token-efficient Muon optimizer stable at trillion scale) and a **large-scale agentic data synthesis pipeline** that fabricates multi-step tool-use data the web doesn't contain.
> - **Most surprising finding:** **Zero loss spikes across 15.5T tokens.** QK-clip caps per-head attention logits at τ=100 and the instability that normally kills large Muon runs simply never appears.
> - **Where it fails:** K2 is non-thinking by design, so it trails reasoning-heavy systems on the hardest agentic SWE-bench settings; the synthetic-data strategy is explicitly flagged as unproven for continued scaling; and the report omits total compute, FLOPs, and MFU.

## Context: what came before

To understand why K2 looks the way it does, you have to hold three separate lineages in your head, because the paper is a deliberate fusion of all three.

**The MoE-scaling lineage.** Since GShard and the Switch Transformer, the dominant way to add capacity without proportionally adding compute has been sparse Mixture-of-Experts: route each token to a small subset of expert FFNs, so total parameters grow while *activated* parameters per token stay bounded. DeepSeek-V3 pushed this to 671B total / 37B active and demonstrated that an aggressively sparse MoE with Multi-head Latent Attention (MLA) could be trained economically and serve cheaply. Kimi K2 sits squarely in this tradition — it is, architecturally, a more sparse DeepSeek-V3 — but it pushes the sparsity ratio further: 384 experts with 8 active is a sparsity ratio of 48, versus DeepSeek-V3's 32.

**The optimizer lineage.** Almost everyone trains large language models with AdamW. Muon, a newer optimizer that orthogonalizes the momentum update via a Newton-Schulz iteration, is dramatically more **token-efficient** — it extracts more capability per training token, which matters enormously when high-quality data is the binding constraint. Moonshot's own earlier work (Moonlight) showed Muon was *scalable* in the sense of being economical, but scaling it to a trillion parameters surfaces a nasty failure mode: attention logits explode. K2's MuonClip is the patch that makes Muon survive at this scale. (If you want the optimizer in isolation, see the companion read on [Muon is Scalable for LLM Training](/blog/paper-reading/large-language-model/muon-moonlight).)

**The agentic-data lineage.** This is the part the field has been quietly stuck on. Instruction tuning taught models to follow single-turn requests; RLHF taught them to be helpful and harmless. But *agentic* behavior — call a tool, read its output, decide the next call, recover when it errors, do this for a dozen turns — depends on demonstration data that essentially does not occur in scraped corpora. The lineage here runs through ToolLLM, the various "self-instruct for tools" efforts, and tau-bench-style evaluation environments. K2's contribution is to industrialize this: a synthesis pipeline that generates tool-use trajectories at scale and filters them against rubrics.

The gap K2 claims to fill is the intersection of all three: take the most token-efficient optimizer, make it stable at the sparsest practical MoE scale, and feed the resulting base model a post-training diet of synthetic agentic data plus a unified RL stage. The bet is that *agentic intelligence is a data-and-stability problem, not a test-time-compute problem* — which is why K2 reports everything in non-thinking mode.

That non-thinking framing deserves a moment, because it is a real philosophical stance and not just an evaluation convenience. A "thinking" model spends extra tokens reasoning before it answers, and for hard problems that test-time compute buys accuracy. K2 deliberately competes *without* that crutch, capping outputs at 8,192 tokens and reporting raw single-pass performance. The argument is that if your base model and your post-training are strong enough, you should not *need* to think your way out of a tool-use task — the right action should be close to reflexive. Whether you buy that or not, it makes K2's numbers a cleaner measurement of base capability than a system that mixes model quality with a reasoning budget. It also means every comparison in this post is, if anything, *conservative* toward K2: the closed models it beats are often beating it back the moment you let them reason.

## Contributions

The paper's stated contributions, tightened:

1. **MuonClip optimizer.** A modification of Muon that adds **QK-clip**, a post-update rescaling of query/key projection weights applied per attention head, which eliminates the exploding-attention-logit instability that otherwise derails large-scale Muon training. K2 pre-trains on 15.5T tokens with zero loss spikes.
2. **A 1.04T / 32.6B-activated MoE** with MLA attention, 384 experts (8 active + 1 shared), 61 layers, and a 160K vocabulary — released as open weights in both `Kimi-K2-Base` and `Kimi-K2-Instruct` checkpoints.
3. **Token-utility data engineering.** A rephrasing pipeline that rewrites knowledge and math text into multiple style- and perspective-diverse versions, improving downstream factuality per training token.
4. **A large-scale agentic data synthesis pipeline** spanning 3,000+ real MCP tools and 20,000+ synthetic tools, thousands of synthetic agents, rubric-based task generation, and an LLM-simulated user plus a tool-execution sandbox — keeping only trajectories that meet success criteria.
5. **A single joint RL stage** that unifies verifiable rewards (math, code, instruction following) with a **self-critique rubric reward** for non-verifiable tasks, stabilized by budget control, a PTX auxiliary loss, and temperature decay.
6. **State-of-the-art non-thinking results** across coding, tool-use, and math benchmarks, with full evaluation tables against open and closed baselines.

## Method

### Architecture: a trillion parameters you mostly don't pay for

![Kimi K2 architecture](/imgs/blogs/kimi-k2-2.png)

The architecture is a sparse MoE transformer, and the figure above is the honest accounting of where the parameters live. The model is **1.04T parameters total**, but the routing network keeps only **8 of 384 experts** live for any given token (plus 1 shared expert that is always on), so the **activated** parameter count per token is **32.6B**. That ratio — total over active — is the *sparsity ratio*, and at 48 it is the most aggressive of any major open model at release.

The other dimensions:

| Property | Value |
|---|---|
| Total / activated parameters | 1.04T / 32.6B |
| Layers | 61 (1 dense + 60 MoE) |
| Experts (total / active / shared) | 384 / 8 / 1 |
| Attention | Multi-head Latent Attention (MLA), 64 heads |
| Model hidden dim | 7,168 |
| Expert hidden dim | 2,048 |
| Activation | SwiGLU |
| Vocabulary | 160K |
| Context (pre-train → final) | 4K → 128K via YaRN |

Two choices are worth dwelling on. **MLA** (the Multi-head Latent Attention from the DeepSeek line) compresses the KV cache into a low-rank latent, which is what makes a model this size economical to *serve* — the KV cache, not the parameters, is usually what bounds inference batch size at long context. And the **single dense layer** at the bottom before the 60 MoE layers is a small but common stabilizer: routing in the very first block tends to be noisy, so keeping it dense avoids feeding garbage expert assignments into the rest of the stack.

A subtle point the figure encodes: capacity and compute are decoupled. You are paying inference FLOPs for a 32.6B model and storing the memory of a 1.04T one. That trade is the entire reason MoE exists, and K2 leans into it harder than its predecessors.

#### The sparsity arithmetic, worked out

It is worth doing the napkin math, because "1T parameters" and "32.6B active" sound like marketing until you trace where the numbers come from. With 384 routed experts and 8 active per token, plus 1 always-on shared expert, the fraction of expert capacity touched per token is roughly $9/385 \approx 2.3\%$. The expert bank is where almost all of the trillion parameters live — 384 experts × (expert hidden 2,048, model dim 7,168, three SwiGLU matrices) across 60 MoE layers is the dominant term — so activating 8 of them is what collapses a trillion-parameter *forward pass* down to the FLOP cost of a ~32.6B dense model.

The consequence for a practitioner is a clean separation of two budgets:

| Budget | Scales with | K2 value | What it gates |
|---|---|---|---|
| Inference FLOPs / token | **activated** params | ~32.6B | latency, throughput |
| Weight memory (VRAM/host) | **total** params | 1.04T | how many GPUs to hold it |
| KV cache / sequence | layers × heads × dim, via MLA | low-rank latent | max batch × context |

This is why the sparsity ratio is the headline number, not the trillion. A denser MoE — DeepSeek-V3 activates 37B of its 671B (a ratio near 18) — pays more FLOPs per token for the same slice of total capacity; K2's ratio of 48 is a bet that you can route more selectively and still cover the task distribution. The risk of that bet is expert under-training — with so many experts, each one sees fewer tokens — which is exactly the kind of thing the always-on shared expert and the load-balancing in routing are there to mitigate.

### MuonClip: the fix that makes Muon survive at scale

Here is the single most reusable idea in the paper, and it is worth understanding precisely because it will bite anyone who tries to scale Muon.

![Why Muon needed QK-clip to scale](/imgs/blogs/kimi-k2-3.png)

Muon is more token-efficient than AdamW, which is exactly why you'd want it when data is the bottleneck. But scale it up and you hit a wall: the **maximum attention logits explode**. In Moonshot's mid-scale experiments the max logits exceeded ~1000, and once that happens the softmax saturates, gradients go haywire, and the loss spikes. The before/after above is the whole story: raw Muon on the left diverges; MuonClip on the right caps the logits and trains clean.

**QK-clip** is the mechanism. It is *not* a change to the loss or the attention math during the forward pass — it is a **post-update rescaling of the query and key projection weights**, applied per head, only when that head's max logit crosses a threshold τ = 100. The procedure, per attention head $h$ with observed max logit $S_{\max}^h$:

$$
\gamma = \min\!\left(1, \frac{\tau}{S_{\max}^h}\right)
$$

Then scale the *content* components of the query and key by $\sqrt{\gamma}$, scale the head-specific rotary query component by $\gamma$, and leave the shared rotary key component untouched. In pseudo-PyTorch, applied after each optimizer step:

```python
def apply_qk_clip(layers, logit_tracker, tau=100.0):
    """Run after each MuonClip optimizer step.

    q_c, k_c are content projections; q_r is the head-local rotary query;
    k_r is the shared rotary key, deliberately never rescaled. We rescale the
    *weights*, not the activations, so the cap persists into the next step.
    """
    with torch.no_grad():
        for layer in layers:
            for h in range(layer.num_heads):
                s_max_h = logit_tracker[layer.idx][h]    # max_{i,j} q_i . k_j for head h
                if s_max_h <= tau:
                    continue                             # head is calm; skip it
                gamma = tau / s_max_h                    # gamma < 1 only when over threshold
                layer.Wq_content[h].mul_(gamma ** 0.5)   # split correction across the two
                layer.Wk_content[h].mul_(gamma ** 0.5)   # content factors q_c . k_c
                layer.Wq_rotary[h].mul_(gamma)           # full gamma on head-local rotary q
                # layer.Wk_rotary is shared across heads -> intentionally untouched
```

Why scale the *weights* rather than clamp the *logits*? Because clamping the activation only fixes the current forward pass; the weights that produced the runaway logit are still runaway, and the next step reproduces the spike. By rescaling the projection weights themselves, the cap is *persistent* — and empirically, with MuonClip, K2's max logits rapidly hit the τ=100 cap and then **decay back into a stable range after roughly 30% of training steps**, after which clipping rarely fires. The model trains the remaining 70% essentially unclipped, having been walked through the dangerous early regime.

The asymmetry between $\sqrt{\gamma}$ on content and $\gamma$ on the rotary query is the kind of detail that looks arbitrary until you work out the logit algebra: the content path contributes a $q_c \cdot k_c$ term (two factors, so split the correction as $\sqrt{\gamma}$ each), while the head-local rotary query needs the full $\gamma$ because its partner key is shared and deliberately left alone. Get this wrong and you either over-clip (hurting capacity) or fail to cap the right term.

A worked example makes the cap concrete. Suppose head 12 in layer 3 reaches $S_{\max}^h = 400$ at some step — four times the threshold. Then $\gamma = 100/400 = 0.25$, the content projections are multiplied by $\sqrt{0.25} = 0.5$, and the head-local rotary query by $0.25$. The pre-softmax logit for that head, which is bilinear in the query and key, is pulled back toward 100; the next step starts from rescaled weights rather than the runaway ones, so the spike does not regenerate. Heads below threshold see $\gamma = 1$ and are untouched, so the intervention is surgical — it only ever touches the handful of heads actually misbehaving.

The training-dynamics story is the part I find most reassuring. With MuonClip the max logits **climb to the τ=100 cap early, sit there while clipping fires, and then decay back below it after roughly 30% of training** — at which point the cap essentially stops engaging and the model trains the remaining ~10T tokens unclipped. That shape tells you QK-clip is not a permanent governor fighting the model the whole way; it is a guardrail that walks the run through the unstable early regime and then gets out of the way. The same logic is why the asymmetric $\sqrt{\gamma}$-on-content / $\gamma$-on-rotary split matters: a sloppier "just divide everything by $\gamma$" would over-damp the content path and cost you capacity for the entire run, not just the dangerous opening.

The result is the headline that should make any large-model trainer sit up: **15.5T tokens, zero loss spikes.** No babysitting, no restart-from-checkpoint, no learning-rate surgery mid-run. For anyone who has watched a large run diverge at 2 a.m. and lost a week rewinding to a checkpoint, that is the most valuable sentence in the paper. The pre-training schedule it enabled is also worth recording for reproducers: a constant learning rate of 2e-4 for the first 10T tokens (after a 500-step warmup), cosine decay to 2e-5 over the next 5.5T, a final anneal to 7e-6, weight decay 0.1 throughout, and a global batch of 67M tokens.

### Data: rewriting text to get more capability per token

If your optimizer is token-efficient and your data is the bottleneck, the obvious next move is to make each token carry more signal. K2 does this with **rephrasing**.

Knowledge-domain text is rewritten into multiple style- and perspective-diverse versions via chunk-wise autoregressive generation with a fidelity check, so the model sees the same fact framed several ways rather than memorizing one surface form. Math text is rewritten into a "learning-note" style following the SwallowMath methodology. The ablation is small but clean:

| Data treatment | Epochs | SimpleQA accuracy |
|---|---|---|
| Raw text | 10 | 23.76% |
| 1 rephrasing | 10 | 27.39% |
| 10 rephrasings | 1 | **28.94%** |

The load-bearing comparison is the first row against the last: **same compute budget** (10 passes over the data either way), but ten *diverse rephrasings seen once* beat the *same raw text seen ten times* by more than five points on SimpleQA. Re-reading identical tokens is worth far less than reading paraphrased tokens, because repetition mostly reinforces surface memorization while paraphrase forces the model to abstract the underlying fact. This is a genuinely actionable result for anyone data-constrained — though, as the critique section notes, one benchmark is thin evidence.

### Agentic data synthesis: manufacturing the data the web doesn't have

This is the part that turns a strong base model into an agent, and it is the most operationally involved component of the paper.

![Agentic data synthesis pipeline](/imgs/blogs/kimi-k2-4.png)

The problem is concrete: to teach multi-turn tool use, you need trajectories where an agent receives a task, calls tools, reads their outputs, and iterates to a verified success. That data is essentially absent from pre-training corpora. K2 fabricates it, and the graph above traces the assembly line:

1. **Tool ecosystem.** Start from **3,000+ real MCP tools** and synthesize **20,000+ more** via hierarchical domain evolution, giving the agents a huge and varied action space.
2. **Synthetic agents.** Instantiate thousands of agents with varied system prompts and tool combinations, so the trajectories aren't all in one persona's voice.
3. **Rubric-based task generation.** Generate tasks *with explicit success criteria* attached — the rubric is created alongside the task, not after.
4. **Simulated rollout.** An **LLM simulates the user** while a **tool-execution sandbox** actually runs the calls, producing a real multi-turn trajectory with real tool outputs (including errors to recover from).
5. **Rubric filter.** Only trajectories that meet the success criteria survive into the SFT set. Everything else is discarded.

The branch-and-merge in the figure — task generation fanning out to both the user simulator and the tool environment, then merging into a single trajectory — is the crux. The user simulator supplies intent and follow-ups; the sandbox supplies ground-truth tool behavior. Neither alone produces usable agentic data: a user-only simulation hallucinates tool outputs, and a tool-only environment has no one to serve. The rubric filter at the end is what keeps the SFT distribution from collapsing into plausible-but-failed trajectories, which is the usual failure mode of naive self-instruct-for-tools.

It helps to picture the *shape* of a single surviving trajectory. A rubric-scored task ("book the cheapest flight under \$400 that lands before noon") becomes a multi-turn exchange: the agent calls a search tool, the sandbox returns real (synthetic) flight rows, the agent notices none land before noon, *revises its query*, calls again, picks a candidate, calls a booking tool that errors on a missing passenger field, recovers by asking the simulated user, and finally completes. The two behaviors that make this data valuable — *query revision* after a bad result and *error recovery* after a failed call — are precisely the things web text never demonstrates, because nobody writes down their failed tool calls. The rubric ("did it book a flight meeting all constraints?") is binary enough to filter on but only satisfiable through the full recover-and-retry loop, so the filter implicitly selects for resilience rather than luck. That is the whole trick: you don't teach tool use by showing clean successes, you teach it by manufacturing messy successes and throwing away the messes that didn't succeed.

### Joint RL: one stage, two kinds of reward

Most post-training stacks run a sequence of RL stages — one for math, one for code, one for safety. K2 collapses these into a **single joint RL stage**, and the interesting design problem is that some tasks have a checkable answer and some don't.

![Kimi K2 joint-RL reward design](/imgs/blogs/kimi-k2-6.png)

The tree above is the reward taxonomy. On the **verifiable** side, the reward is rule-based and objective: math and STEM problems with known answers, coding tasks scored by **unit tests on competition problems and real GitHub PRs/issues**, and instruction-following checked by a hybrid of rules and an LLM judge. These are the easy cases — you can compute a clean signal.

The hard case is the **non-verifiable** side: open-ended generation where there is no unit test. K2's answer is a **Self-Critique Rubric Reward**. The model performs pairwise comparisons of its own outputs against three kinds of rubric: a **core rubric** (general quality), **prescriptive rubrics** designed specifically to block reward hacking, and **human-annotated task-specific rubrics**. Critically, the critic that issues these rewards is *itself refined in a closed loop* using the verifiable signals from on-policy rollouts — the objective, checkable tasks are used to keep the subjective critic honest.

Holding all of this stable is a set of three regularizers, shown as the third branch:

- **Budget control** — a per-sample maximum token budget with a penalty for overage, which stops the model from learning to ramble its way to reward.
- **PTX auxiliary loss** — a standard pull back toward curated high-quality samples, so RL doesn't drift away from the SFT distribution.
- **Temperature decay** — anneal exploration into exploitation over the course of training.

The RL objective itself minimizes a squared-advantage term with entropy-style regularization,

$$
\mathcal{L}(\theta) = \mathbb{E}_{x,\,y_i}\Big[\big(r(x, y_i) - \bar{r}(x) - \tau \log\tfrac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\big)^2\Big],
$$

where $r(x,y_i)$ is the reward for sample $y_i$, $\bar{r}(x)$ is the per-prompt mean (the baseline), and $\tau > 0$ controls the KL-style regularization toward the old policy. If you've read the [GRPO post](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo), this is recognizably in the same family — a group-relative advantage with a trust-region pull — and it shares DNA with the RL recipe in [Kimi k1.5](/blog/paper-reading/reinforcement-learning/kimi-k1-5), Moonshot's earlier RL-scaling paper.

The subtraction of the per-prompt mean $\bar{r}(x)$ is doing the same job a value-function baseline does in PPO — it centers the advantage so the gradient pushes up samples that beat the *prompt's own average* and pushes down samples that fall below it, without needing a separately trained critic. That is the GRPO insight: when you sample a group of completions per prompt, the group's mean reward is a free, low-variance baseline. Squaring the residual (rather than the usual policy-gradient log-prob-times-advantage form) turns the update into a regression of the policy's log-ratio onto the centered reward, which is numerically gentler and pairs naturally with the budget-control and PTX terms. The practical upshot is that K2 can run *one* RL loop over a heterogeneous task mix — math next to open-ended writing next to SWE — because the reward source is pluggable (a unit test here, a rubric-judge there) while the optimization machinery stays identical.

#### Long-context: the cheap part, done last

The 128K context is not a pre-training property — K2 pre-trains at a 4K sequence length, activates long context at 32K, and only reaches 128K via **YaRN** rotary interpolation at the end. The schedule is deliberately back-loaded: a 400B-token annealing phase at 4K, then a 60B-token context-extension phase at 32K, then YaRN to stretch the effective window to 128K. The lesson, which generalizes well beyond K2, is that long context is cheapest to buy *after* the model is otherwise trained — you spend a tiny fraction of the token budget (60B of 15.5T, well under half a percent) extending the window rather than paying the quadratic attention cost over the entire run. Models that train long from the start pay for context length on every one of their trillions of tokens; K2 pays for it on the last fraction of a percent.

### Infrastructure: the unglamorous part that makes it possible

The report is refreshingly concrete about systems. Training runs on **NVIDIA H800** nodes (8× H800, 2TB RAM, NVLink/NVSwitch intra-node, 8×400 Gbps RoCE inter-node). The model-parallel group spans **256 GPUs**, with **16-way pipeline parallelism** (virtual stages) and **16-way expert parallelism** layered over ZeRO-1 data parallelism. Memory is clawed back with selective recomputation, FP8 activation storage in 1×128 tiles, and CPU offload, landing at roughly 30GB/GPU for optimizer states.

The RL infrastructure is the part worth stealing: training and inference are **colocated on the same workers**, a distributed checkpoint engine performs a **full parameter update in under 30 seconds**, and the system sustains **10,000+ concurrent sandbox instances** for software-engineering tasks. That sandbox throughput is what makes the agentic RL loop tractable at all — you cannot do RL on SWE tasks if spinning up an environment is slow.

### K2 vs DeepSeek-V3: the same skeleton, three deliberate divergences

Because K2 is so clearly descended from the DeepSeek-V3 design, the most useful way to understand its choices is by diff. The two share the core skeleton — sparse MoE, MLA attention, SwiGLU experts, a shared expert, FP8 training — but K2 diverges in three places that map directly to its three contributions.

| Axis | DeepSeek-V3 | Kimi K2 | Why K2 differs |
|---|---|---|---|
| Total / active params | 671B / 37B | 1.04T / 32.6B | More capacity, *fewer* active FLOPs |
| Experts (total / active) | 256 / 8 | 384 / 8 | Sparsity ratio 18 → 48 |
| Optimizer | AdamW | **MuonClip** | Token efficiency, needs QK-clip to stay stable |
| Pre-training tokens | 14.8T | 15.5T | Comparable scale |
| Post-training emphasis | reasoning + general | **agentic + tool use** | Synthetic trajectories + joint RL |

The pattern is coherent: K2 spends its novelty budget on (1) pushing sparsity harder to get more capacity at lower per-token cost, (2) swapping AdamW for Muon to wring more out of each token — which *forces* the QK-clip invention — and (3) bending post-training toward agentic behavior. Everything else it inherits, sensibly, from a design that already works. If you're trying to decide what to copy, this table is the answer: the sparsity push and MuonClip are the load-bearing bets; the rest is proven scaffolding.

## Experiments

All scores below are **non-thinking mode** with outputs capped at 8,192 tokens — an important caveat, because it means K2 is competing with one hand tied relative to systems that get to think.

![Kimi K2 vs frontier baselines](/imgs/blogs/kimi-k2-5.png)

The matrix above is the compressed verdict: across six headline benchmarks K2 wins five and loses only one — agentic SWE-bench Verified, where Claude Opus 4's 72.5 edges K2's 65.8. The full table tells the rest of the story:

| Benchmark | Metric | Kimi-K2 | DeepSeek-V3 | Qwen3 | Claude Sonnet 4 | Claude Opus 4 | GPT-4.1 |
|---|---|---|---|---|---|---|---|
| LiveCodeBench v6 | Pass@1 | **53.7** | 46.9 | 37.0 | 48.5 | 47.4 | 44.7 |
| OJBench | Pass@1 | **27.1** | 24.0 | 11.3 | 15.3 | 19.6 | 19.5 |
| MultiPL-E | Pass@1 | 85.7 | 83.1 | 78.2 | 88.6 | **89.6** | 86.7 |
| SWE-bench Verified (agentic, single) | Pass@1 | 65.8 | 38.8 | 34.4 | 72.7 | **72.5** | 54.6 |
| SWE-bench Multilingual | Pass@1 | 47.3 | 25.8 | 20.9 | **51.0** | — | 31.5 |
| Tau2-Bench Telecom | Avg@4 | **65.8** | 32.5 | 22.1 | 45.2 | 57.0 | 38.6 |
| AIME 2024 | Avg@64 | **69.6** | 59.4 | 40.1 | 43.4 | 48.2 | 46.5 |
| AIME 2025 | Avg@64 | **49.5** | 46.7 | 24.7 | 33.1 | 33.9 | 37.0 |
| MATH-500 | Accuracy | **97.4** | 94.0 | 91.2 | 94.0 | 94.4 | 92.4 |
| GPQA-Diamond | Avg@8 | **75.1** | 68.4 | 62.9 | 70.0 | 74.9 | 66.3 |
| MMLU | Exact Match | 89.5 | 89.4 | 87.0 | 91.5 | **92.9** | 90.4 |
| SimpleQA | Correct | 31.0 | 27.7 | 13.2 | 15.9 | 22.8 | **42.3** |
| IFEval | Prompt Strict | **89.8** | 81.1 | 83.2 | 87.6 | 87.4 | 88.0 |

What is load-bearing in their setup, and what might not transfer:

- **K2 is the best open model nearly everywhere.** Against DeepSeek-V3 and Qwen3 it is not close — double-digit margins on Tau2 Telecom, AIME, and OJBench. If your baseline is open weights, this is a clear step up.
- **It leads even closed models on competition coding and math.** LiveCodeBench v6, OJBench, AIME 2024/2025, MATH-500, HMMT, GPQA-Diamond, IFEval — these are wins over Claude and GPT-4.1, in non-thinking mode.
- **It trails on three predictable axes.** Broad knowledge (MMLU family, where Claude Opus leads), factual recall (SimpleQA, where GPT-4.1's 42.3 is far ahead), and the hardest *agentic* SWE settings (Claude's harness wins). The first two are pre-training-data effects; the third is exactly where test-time reasoning helps and K2 has opted out.

The thing to internalize is that the *shape* of K2's wins matches its design thesis. It dominates where a strong base model plus verifiable RL pays off (math, competition code, tool-use protocols) and trails where either raw memorized knowledge (SimpleQA) or extended reasoning (agentic SWE) is the deciding factor.

A few cells reward a closer look. **Tau2-Bench Telecom** is the most lopsided win on the board — K2's 65.8 against DeepSeek-V3's 32.5 and Qwen3's 22.1 is not a margin, it is a different league, and it is the clearest evidence that the agentic synthesis pipeline transferred to a held-out tool-use benchmark rather than just memorizing its own training distribution. The Tau2 *Retail* and *Airline* splits are tighter (K2 trails Claude Opus there), which suggests the win is uneven across domains — telecom-style API protocols suit K2's training better than retail flows. On **AIME**, K2's 69.6 (2024) and 49.5 (2025) over every listed baseline including the closed models is a strong signal that verifiable-reward RL on math generalizes; the drop from 2024 to 2025 mirrors every other model's drop and reflects the 2025 set being genuinely harder, not a K2-specific regression. And the **SimpleQA** line is the cleanest illustration of the rephrasing section's limits: even with the data engineering that lifted SimpleQA from 23.76% to 28.94% in the ablation, K2's final 31.0 still trails GPT-4.1's 42.3 by a wide margin — paraphrase-augmentation helps factual recall at the margin, but it does not substitute for whatever pre-training corpus GPT-4.1 is drawing on.

The honest reading is that K2 is a *specialist generalist*: world-class at the agentic, coding, and math tasks it was built and post-trained for, merely competitive on broad knowledge, and behind on pure recall. For most agentic deployments that is exactly the right shape of capability — you care far more about whether it can drive your tools than whether it has memorized trivia.

## Critique

**What's strong.** MuonClip is a real, transferable contribution — a precise, cheap fix to a failure mode that will otherwise stop anyone from scaling Muon, backed by the strongest possible evidence (a clean 15.5T-token run). The agentic data synthesis pipeline is the most complete public description of how to manufacture tool-use data I've seen, and the user-simulator-plus-sandbox design is clearly right. The single-joint-RL-stage with a verifiable-critic-refinement loop is an elegant answer to the non-verifiable-reward problem.

**What's weak or unfalsifiable.** The rephrasing result rests on **one benchmark** (SimpleQA); a +5-point gain on a single factuality test is suggestive, not conclusive, and the paper itself flags synthetic-data scaling as an open question. The **self-critique rubric reward is a reward-hacking risk** the authors acknowledge in their own appendix — a model grading its own outputs against rubrics it can see is the textbook setup for Goodharting, and "prescriptive anti-hack rubrics" is a mitigation, not a proof. And the non-thinking framing, while honest, makes the closed-model comparisons slightly apples-to-oranges: Claude and GPT-4.1 in *their* default modes do reason.

**What ablation is missing.** The big omission is **compute transparency**: no total GPU-hours, no FLOPs, no MFU, no training throughput. Without those you cannot judge whether MuonClip's token-efficiency translates into *wall-clock* or *dollar* efficiency, which is the number that actually matters. There is also no ablation isolating the contribution of the agentic synthesis pipeline versus the joint RL stage — we see the final agentic scores, but not how much each post-training component bought.

There is also a quieter methodological gap worth flagging: the data mixture proportions are never quantified. We're told the corpus spans web text, code, mathematics, and knowledge, and we see the rephrasing ablation, but the actual ratios — how much code, how much math — are absent. For a model whose headline strength is coding and math, the mixture is plausibly *the* most important hyperparameter, and it's the one number a reproducer most needs and least gets. Combined with the missing compute figures, this means K2 is reproducible in *architecture and method* but not in *recipe*: you could rebuild the model described here and still not match it, because the parts that aren't disclosed are exactly the parts that take a lab months to tune.

**What would change my mind.** If an independent group scaled Muon past ~100B parameters *without* QK-clip and saw no logit explosion, I'd downgrade MuonClip from "necessary fix" to "belt-and-suspenders." Conversely, if the self-critique rubric reward were shown to inflate held-out human-preference scores while *degrading* a truly blind evaluation, I'd treat the non-verifiable RL results with much more suspicion. And if someone published an ablation that turned off the agentic synthesis pipeline and the joint RL stage independently, and the agentic scores barely moved, I'd conclude the base model was doing most of the work and the elaborate post-training was theater. The fact that the paper *doesn't* run that ablation is the single biggest gap between "interesting engineering report" and "load-bearing scientific claim" — everything in the post-training stack is presented as a package, and packages are where unfalsifiable contributions hide. None of this makes the results wrong; it makes them harder to attribute, which for a practitioner deciding what to copy is almost as costly.

## What I'd build with this

The point of reading a report like this is not to admire the trillion parameters — almost no one reading this will train a 1T model — but to extract the ideas that transfer down to the scale you actually work at. Most of K2's contributions do transfer, because they are about *method* rather than *scale*. Here is where I'd start.

1. **Port QK-clip to a small Muon run.** The fix is ~20 lines and optimizer-agnostic in spirit. The most valuable experiment is to reproduce the logit-explosion-then-cap-then-decay curve at 1–7B scale and confirm the τ=100 threshold and the $\sqrt{\gamma}/\gamma$ asymmetry are doing what the algebra says.
2. **Steal the user-simulator-plus-sandbox loop for domain agents.** You don't need 20,000 synthetic tools to benefit — point the same loop at *your* tool set (internal APIs, a database, a CI system), generate rubric-scored trajectories, and SFT on the survivors. This is the cheapest path from "model that can call my tools" to "model that can recover when my tools error."
3. **Use rephrasing as a data-efficiency lever, not just augmentation.** For any domain where you're re-epoching scarce data, replace some repeats with diverse paraphrases and measure the factuality delta. The K2 result predicts you'll come out ahead at equal compute.
4. **Build a verifiable-critic-refinement harness.** The idea of using objective, checkable tasks to continuously recalibrate a subjective critic generalizes well beyond K2 — it's a practical recipe for keeping an LLM-judge from drifting.
5. **Serve it behind MLA-aware infrastructure.** If you actually deploy K2, the win is the low-rank KV cache; pair it with paged attention and the 128K context becomes economically reasonable rather than aspirational.

## When to reach for Kimi K2 (and when not to)

**Reach for it** when your workload is agentic or code-heavy and you want open weights you can host and fine-tune. K2's strongest results are exactly the situations where you hand a model a set of tools and a goal and expect it to execute — API orchestration, multi-step coding tasks, math-heavy reasoning with a verifiable answer. The non-thinking framing is a feature here: a reflexive tool-caller that doesn't burn a thousand tokens deliberating is cheaper to run in a loop, and the Tau2 Telecom and LiveCodeBench numbers say it doesn't sacrifice much accuracy to get there. The open weights also mean you can run the agentic-synthesis-plus-RL recipe *yourself* on your own tools, which is the real unlock for a team with proprietary APIs.

**Look elsewhere** when your task is dominated by broad factual recall (K2 trails GPT-4.1 on SimpleQA by eleven points and there is no fine-tuning fix for missing pre-training knowledge), or when you genuinely need extended reasoning and are willing to pay for it — a thinking model, or K2's own thinking-mode sibling, will beat non-thinking K2 on the hardest agentic SWE settings. And if you cannot afford the memory footprint of a trillion-parameter weight set across enough GPUs to hold it, the sparsity that makes K2 cheap to *run* does nothing for the cost to *load* it: you still need the VRAM for 1.04T parameters even though you only compute with 32.6B of them per token. That memory-versus-FLOPs split is the single most important thing to internalize before you put K2 anywhere near production.

The deeper takeaway transcends this one model. K2 is a clean demonstration that the frontier of *agentic* capability is currently gated by training stability and data manufacturing, not by test-time reasoning or raw scale — and both of its load-bearing ideas, MuonClip and the agentic synthesis pipeline, are things a sufficiently determined team can reproduce. That is a more useful thing to take away than any single benchmark number.

## References

- **Paper:** [Kimi K2: Open Agentic Intelligence (arXiv:2507.20534)](https://arxiv.org/abs/2507.20534) — Kimi Team, Moonshot AI.
- **Code & weights:** [github.com/MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2); checkpoints `Kimi-K2-Base` and `Kimi-K2-Instruct` on Hugging Face.
- **Related reads on this blog:**
  - [Muon is Scalable for LLM Training](/blog/paper-reading/large-language-model/muon-moonlight) — the optimizer MuonClip extends.
  - [MoBA: Mixture of Block Attention](/blog/paper-reading/large-language-model/moba) — Moonshot's long-context attention work.
  - [Kimi k1.5: Scaling RL with LLMs](/blog/paper-reading/reinforcement-learning/kimi-k1-5) — the RL lineage K2's joint stage builds on.
  - [Fine-tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the group-relative RL objective family.
