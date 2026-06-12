---
title: "Llama-Nemotron: How NVIDIA Makes Reasoning Models Fast with NAS, FFN Fusion, and a Reasoning Toggle"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into the Llama-Nemotron report — Puzzle neural architecture search, FFN Fusion, a controllable reasoning toggle, and an RLVR-based post-training stack that turns frozen Llama models into fast, controllable reasoners."
tags: ["llm", "llama-nemotron", "nvidia", "reasoning", "neural-architecture-search", "puzzle", "ffn-fusion", "rlvr", "grpo", "distillation", "inference-efficiency", "reasoning-toggle"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 49
---

## The reasoning tax, and the bet that you can refund it

Reasoning models changed what large language models can do — long chain-of-thought, self-correction, verifiable math and code — and in the same stroke made them expensive. A reasoning model is costly twice over: it is usually a *large* model (you need capacity to reason), and it *emits far more tokens* (the chain of thought is the product). DeepSeek-R1 is brilliant and slow. The instinct in the field has been to accept this as the price of admission: if you want reasoning, you pay the reasoning tax in latency, in GPU memory, in tokens-per-answer.

Llama-Nemotron is the report that argues the tax is mostly refundable. NVIDIA took frozen Llama models — Llama-3.1-8B, Llama-3.3-70B, Llama-3.1-405B — and turned them into a family of reasoning models (Nano 8B, Super 49B, Ultra 253B) that are **smaller and dramatically faster than their parents while gaining strong reasoning**. LN-Super delivers **5× the throughput** of Llama-3.3-70B-Instruct at batch 256. LN-Ultra **matches or beats DeepSeek-R1** on reasoning benchmarks like GPQA-Diamond (76.0), AIME-24 (80.8), and MATH-500 (97.0) while running on a single **8×H100** node where R1 wants 8×H200, at **1.71× lower latency**. And every model has a **reasoning toggle** — a system-prompt flag that switches the same weights between a fast direct answer and a full chain-of-thought, so you pay the reasoning tax only when the question is worth it.

This is the third post in a series reading NVIDIA's model reports for their reusable techniques, after [Minitron's pruning-and-distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) and the [Nemotron-4 340B synthetic-data alignment](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment). It draws on the [Llama-Nemotron technical report](https://arxiv.org/abs/2505.00949). Where Minitron compresses by *removing* parameters uniformly and Nemotron-4 aligns with *synthetic data*, Llama-Nemotron compresses by **redesigning the architecture per layer for the target hardware** and then teaches reasoning with a distillation-plus-RL stack. The two ideas that make it work — **Puzzle** (a neural architecture search that builds a heterogeneous model) and **FFN Fusion** (turning sequential depth into parallel width) — are the reusable techniques worth extracting.

The mismatch the whole report resolves:

| Question | The default assumption | What Llama-Nemotron shows |
|---|---|---|
| Must reasoning models be slow? | Yes — capacity and long CoT cost latency | No — reshape the architecture and toggle the CoT |
| Is every transformer layer equally important? | Treat them uniformly | No — some layers can drop attention entirely |
| How do you make a model hardware-efficient? | Pick a smaller uniform model | Search per-layer blocks under a hardware budget |
| Is "depth" free? | More layers, more compute, fine | No — sequential FFN depth is a latency killer; fuse it |
| Do you always want reasoning? | A reasoning model always reasons | No — toggle it per request with a system prompt |
| Where does reasoning come from? | Train it from scratch with RL | Distill it from R1, then sharpen with verifiable-reward RL |

![Pipeline diagram showing a frozen Llama parent (8B, 70B, or 405B) passing through Puzzle NAS for per-layer blocks, then FFN Fusion for parallel FFNs, then knowledge distillation to heal the cuts, continued pretraining, reasoning SFT distilled from R1, and an RL stage combining RLVR and RPO, producing the final Llama-Nemotron reasoning model](/imgs/blogs/llama-nemotron-efficient-reasoning-models-1.webp)

Worth naming up front: the cleverness here is that the efficiency and the capability come from *different* parts of the pipeline, so neither has to be compromised for the other. A team that only wanted a faster Llama could run Puzzle and Fusion and stop. A team that only wanted reasoning could run the post-training stack on a stock Llama. Llama-Nemotron does both, in sequence, on the same model — and because the architecture work happens *before* the reasoning work, the reasoning is installed into an already-efficient model rather than bolted onto a slow one. That ordering is deliberate: you reshape first so that every subsequent training token is spent on the cheaper-to-run architecture, and you teach reasoning last so it lands on a healed, hardware-optimal base. Keep that sequence in mind as we walk the stages, because it is why the final model is fast *and* smart instead of one at the expense of the other.

The diagram above is the mental model. Read it left to right: a frozen Llama parent goes into **Puzzle**, which rebuilds its architecture layer by layer for the deployment hardware; **FFN Fusion** collapses the sequential depth that Puzzle exposes; **knowledge distillation** heals the damage from the architectural surgery; **continued pretraining** restores general capability; and a **reasoning-focused post-training stack** (SFT distilled from DeepSeek-R1, then RL with verifiable rewards) installs the controllable reasoning. The rest of this article walks each stage. The organizing insight to carry throughout:

> Efficiency and capability are usually traded against each other. Llama-Nemotron decouples them: Puzzle and FFN Fusion buy the efficiency from the *architecture*, and the post-training stack buys the reasoning from the *data and RL* — so you get both instead of choosing.

## 1. Puzzle: neural architecture search that builds a heterogeneous model

The central technique, and the one most worth stealing, is **Puzzle**. The premise behind it is a quiet heresy against how transformers are built: that **not every layer needs to be the same**, and in fact not every layer needs attention at all.

![Diagram of Puzzle assembling a heterogeneous architecture: a vertical stack of eight transformer layers where some keep a full block (attention plus FFN, blue), some have attention removed with the FFN shrunk to 87%, 50%, or 10% (amber and red), alongside a block library card listing the options built by block-wise local distillation and a mixed-integer-programming solver card that picks one block per layer to minimize latency under H100 memory and accuracy constraints](/imgs/blogs/llama-nemotron-efficient-reasoning-models-2.webp)

A standard transformer is **homogeneous**: every layer has the same attention block and the same FFN with the same dimensions. Puzzle asks whether that uniformity is necessary, and the answer is no. It works in two phases.

**Phase one: build a library of alternative blocks.** For each layer of the parent model, Puzzle constructs a menu of replacement blocks that trade accuracy for efficiency:

- The **full block** — attention + FFN, unchanged.
- **Attention removed** — drop the attention sublayer entirely, keeping only the FFN. This is the radical option: it eliminates that layer's attention compute *and its KV-cache contribution*, which is a major win for memory and long-context throughput (see the [KV cache deep dive](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) for why KV memory dominates at long context).
- **FFN shrunk** — keep attention but reduce the FFN intermediate dimension, down a ladder from **87% to as low as 10%** of the original width.

Each candidate block is trained to mimic the original block's input-output behavior with **block-wise local distillation** — you freeze everything else and distill just this one block's replacement against the parent block's outputs, which is cheap because it is local (one block at a time, short training) rather than global. The result is a library: for every layer, a set of blocks each with a known (accuracy, latency, memory) profile.

**Phase two: assemble the best model with a solver.** Now you have, for each of $L$ layers, a menu of $k$ blocks, and you want to pick one block per layer to build a model that is as accurate as possible while meeting a hardware constraint — a latency target, a memory budget, a throughput floor on specific GPUs. This is a combinatorial optimization, and Puzzle solves it with **mixed-integer programming (MIP)**: minimize a quality loss subject to the chosen blocks' total latency and memory fitting the budget. The MIP solver searches the exponential space of per-layer choices and returns the optimal heterogeneous architecture for *that hardware*.

The output is a model that looks nothing like a textbook transformer: some layers keep full attention, some have no attention at all, FFN widths vary layer to layer. It is **co-designed with the deployment hardware** — give Puzzle a different latency budget or a different GPU and it assembles a different model from the same library.

Here is the shape of the per-block scoring and the MIP objective, in pseudocode:

```python
def build_block_library(parent, layer_idx, calib):
    """Block-wise local distillation: train each candidate to mimic the parent block."""
    variants = []
    for cfg in [FULL, NO_ATTENTION, FFN_87, FFN_50, FFN_10]:
        block = make_block(parent.layers[layer_idx], cfg)
        block = local_distill(block, parent.layers[layer_idx], calib)  # cheap, local
        variants.append({
            "cfg": cfg,
            "quality_drop": measure_kl(block, parent.layers[layer_idx], calib),
            "latency_ms": profile_latency(block, target_gpu="H100"),
            "kv_bytes": profile_kv_cache(block),
        })
    return variants

def solve_architecture(library, latency_budget, mem_budget):
    """MIP: pick one block per layer to minimize quality drop under hardware limits."""
    model = mip.Model()
    pick = {(l, v): model.binary() for l in layers for v in library[l]}
    for l in layers:                                  # exactly one block per layer
        model.add(sum(pick[l, v] for v in library[l]) == 1)
    model.add(sum(pick[l, v] * v.latency_ms for l, v in pick) <= latency_budget)
    model.add(sum(pick[l, v] * v.kv_bytes  for l, v in pick) <= mem_budget)
    model.minimize(sum(pick[l, v] * v.quality_drop for l, v in pick))
    model.solve()
    return [v for (l, v), chosen in pick.items() if chosen]
```

### Block-wise local distillation: cheap because it is local

The reason Puzzle can afford to build a library of *several* candidate blocks for *every* layer is that each candidate is trained **locally**, not globally. Global training — fine-tuning the whole model after a change — is expensive: every gradient step touches all $L$ layers and requires a forward and backward pass through the entire network. Local distillation sidesteps this. To build a candidate block for layer $\ell$, you freeze the rest of the model, feed calibration data through the frozen parent up to layer $\ell$, and train *only the replacement block* to match the parent block's output given the parent block's input. The loss is a local reconstruction — make this one block behave like the original — so each training run is short and touches one block's worth of parameters. You can build dozens of candidates across all layers for a fraction of the cost of a single global fine-tune, which is what makes the search space explorable at all.

This locality is the same insight that makes Puzzle scalable that made [Minitron's importance estimation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) cheap: you do not need the global, expensive signal to make a good local decision. A block that reconstructs its parent's behavior well *locally* will, after the final global healing pass, integrate well *globally* — the local proxy is good enough to drive the search, and the cheap global heal at the end cleans up the residual mismatch. The general principle is to **decompose an expensive global optimization into cheap local ones plus a final reconciliation**, a pattern that recurs throughout efficient ML systems.

### The MIP formulation, in words

The mixed-integer program is worth understanding at the level of what it is actually optimizing. You have a binary decision variable for every (layer, block-candidate) pair: $x_{\ell,b} = 1$ if layer $\ell$ uses block $b$. Three kinds of constraints shape the solution. First, a **selection constraint**: each layer uses exactly one block, $\sum_b x_{\ell,b} = 1$. Second, **resource constraints**: the total latency of the chosen blocks must fit the budget, $\sum_{\ell,b} x_{\ell,b}\cdot \text{lat}_{\ell,b} \le \text{Latency}_{\max}$, and similarly for KV-cache memory and any other hardware limit. Third, the **objective**: minimize the total quality degradation, $\sum_{\ell,b} x_{\ell,b}\cdot \text{drop}_{\ell,b}$, where each block's quality drop was measured during local distillation. The solver explores this combinatorial space — exponentially many per-layer combinations — and returns the *provably optimal* selection under the constraints, something no amount of human intuition about "which layers can lose attention" can match. The elegance is that all the expensive measurement happens once (during library construction), and the MIP then re-solves cheaply for any new budget: want a model for a tighter latency SLA? Re-run the solver with a smaller $\text{Latency}_{\max}$ and get a different optimal architecture from the same library, no retraining.

### Why dropping attention works

The most surprising Puzzle move is removing attention from some layers entirely. Why does that not destroy the model? Because **attention is redundant across depth**. Once the early and middle layers have moved information between token positions, many later layers are doing position-local refinement that the FFN handles fine on its own — the attention sublayer in those layers is contributing little beyond what the residual stream already carries. This is the same U-shaped-importance intuition from the [Minitron post](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation): not all of the model's machinery is load-bearing everywhere, and the redundant parts can go. Removing attention is the highest-value cut because attention is the most expensive sublayer at long context — it is quadratic in sequence length and it owns the KV cache — so deleting it where it is redundant pays off disproportionately for reasoning models, which run at long context by nature.

### Second-order optimization: search for the architecture, do not guess it

The reusable principle is that **the right architecture is a search problem, not a guess**. The industry default is to pick a model size off a ladder (7B, 13B, 70B) with uniform layers and hope it fits your serving constraints. Puzzle inverts this: you state the constraints (this GPU, this latency, this memory) and *search* for the architecture that maximizes quality within them. The block library plus MIP is how you make that search tractable — local distillation makes building candidates cheap, and the MIP makes choosing among them optimal. For anyone serving models under hard latency budgets, the meta-lesson is that a heterogeneous, hardware-searched architecture can dominate a uniform one of the same parameter count, because it spends capacity where it matters and strips it where it does not.

## 2. FFN Fusion: turning sequential depth into parallel width

Puzzle creates an opportunity that FFN Fusion cashes in. Once Puzzle has removed attention from a run of consecutive layers, what is left in those layers is **a sequence of FFN blocks with nothing between them** — and a sequence of FFNs that only depend on each other in series is a latency problem hiding as an architecture.

![Before-and-after comparison: on the left, after Puzzle removes attention a sequence of FFN blocks runs one after another, creating a long critical path with low GPU utilization; on the right, FFN Fusion identifies the consecutive run and replaces it with one wide FFN executed in parallel, cutting LN-Ultra latency by 1.71x](/imgs/blogs/llama-nemotron-efficient-reasoning-models-3.webp)

The problem with a sequential FFN stack is **critical-path depth**. On a GPU, throughput comes from parallelism, and a chain of operations that must run one after another — FFN₁ then FFN₂ then FFN₃ — leaves the hardware underutilized, because each FFN waits for the previous one to finish before it can start. The depth is sequential latency you pay on every token.

**FFN Fusion** observes that consecutive FFN blocks can be **algebraically restructured into a single wider FFN that runs in parallel**. The intuition: if several FFN layers sit back to back with no attention mixing positions between them, their combined computation can be reorganized so that the work happens in fewer, wider matrix multiplications instead of more, narrower sequential ones. Wider matmuls are exactly what GPUs are good at — they saturate the hardware — whereas a deep chain of narrow ones does not. You trade *depth* (sequential, slow) for *width* (parallel, fast) at roughly equal expressivity.

Applied to LN-Ultra, FFN Fusion delivers a **1.71× latency improvement**. That number is the difference between a 405B-derived reasoning model that is impractical to serve and one that runs on a single 8×H100 node faster than DeepSeek-R1.

```python
def sequential_ffns(x):  # 3 FFNs, no attention between -> long critical path
    x = ffn1(x)          # must finish before ffn2 starts
    x = ffn2(x)          # must finish before ffn3 starts
    x = ffn3(x)          # depth = 3 (sequential)
    return x

def fused_ffn(x):        # fused into one wide FFN, matmuls run in parallel
    return wide_ffn(x)   # depth = 1, wider matmul saturates the GPU
```

### The algebra of why consecutive FFNs fuse

It helps to see why fusion is *valid*, not just desirable. A feed-forward block is, schematically, $\text{FFN}(x) = W_{\text{down}}\,\phi(W_{\text{up}}\,x)$ with a nonlinearity $\phi$ in the middle, wrapped in a residual connection. Stack two of them with nothing in between and you get $x + \text{FFN}_1(x)$ feeding $\text{FFN}_2$. Because there is no attention mixing positions between them, each token's computation through the stack is *independent of every other token's* — the stack is a pure per-position function. That independence is the key: a sequence of per-position transformations can be reorganized into a wider per-position transformation that computes the same function (or a close approximation that the healing distillation pass corrects) with fewer sequential steps. Concretely, you can widen the intermediate dimension and merge the projections so that what was three sequential narrow matmuls becomes one wide matmul. The arithmetic intensity goes up (bigger matmuls), the sequential depth goes down (one step instead of three), and the GPU — which is starved by small sequential ops and fed by large parallel ones — runs far closer to its roofline. The attention layers are precisely what *blocks* this restructuring in a normal transformer, because attention makes each position depend on all others, breaking the per-position independence. Remove attention (via Puzzle) and the independence returns, and with it the ability to fuse.

### Why fusion and Puzzle are a pair

It is worth being explicit that FFN Fusion is *enabled by* Puzzle, not independent of it. In a standard transformer, attention sits between every pair of FFNs, mixing positions, so you cannot fuse across it — the attention is doing real work that breaks the algebraic restructuring. Only after Puzzle has *removed* attention from a run of layers do you get consecutive FFNs with nothing between them, and only then is fusion possible. This is a beautiful example of techniques that compound: Puzzle's attention removal is valuable on its own (less compute, less KV cache), but it *also* creates the structural precondition for a second optimization that Puzzle alone could not deliver. The lesson for systems design is to look for these enabling relationships — one transformation that opens the door to another — because stacked optimizations beat isolated ones.

### Second-order optimization: depth is latency, width is throughput

The general principle underneath FFN Fusion is one every systems engineer should internalize: **sequential depth is the enemy of latency, parallel width is the friend of throughput**. Two models with the same FLOPs can have wildly different latency depending on how those FLOPs are arranged — a deep chain of small ops is slow even if the total work is small, while a shallow set of large ops is fast even if the total work is larger. When you are optimizing inference, count the *critical path*, not just the FLOPs, and look for opportunities to flatten depth into width. FFN Fusion is one instance; the same thinking drives techniques like parallel attention-FFN blocks and speculative decoding.

## 3. The reasoning toggle: pay for thinking only when it helps

The efficiency story so far is about the architecture. The reasoning toggle is an efficiency story about *behavior*, and it is one of the most practically useful features in the whole report.

![Graph showing one set of Llama-Nemotron weights receiving a system prompt that says "detailed thinking on" or "off"; the off branch produces a fast direct answer with few tokens, while the on branch produces a long chain-of-thought followed by the answer with many tokens](/imgs/blogs/llama-nemotron-efficient-reasoning-models-4.webp)

The problem with reasoning models is that they reason about *everything*, including questions that do not need it. Ask a reasoning model "what's the capital of France" and it may emit a paragraph of chain-of-thought before answering "Paris" — burning tokens, latency, and money on a question a non-reasoning model answers instantly. The conventional fix is to run two models: a fast chat model and a slow reasoning model, routing between them. That doubles your serving footprint and your operational complexity.

Llama-Nemotron's fix is **one model, two modes, switched by a system prompt**. Put `detailed thinking on` in the system prompt and the model produces a full chain-of-thought before its answer; put `detailed thinking off` and it answers directly. Same weights, same deployment — the behavior is controlled at inference time, per request:

```python
messages_fast = [                                  # fast path, no chain-of-thought
    {"role": "system", "content": "detailed thinking off"},
    {"role": "user", "content": "What is the capital of France?"},
]  # -> "Paris."

messages_reason = [                                # reasoning path, full CoT
    {"role": "system", "content": "detailed thinking on"},
    {"role": "user", "content": "Prove there are infinitely many primes."},
]  # -> "<think> Suppose finitely many... </think> There are infinitely many primes..."
```

How is one model trained to do both? During **reasoning SFT** (§4), the training data is *labeled with the mode*: reasoning examples (with chain-of-thought) are paired with the `detailed thinking on` system prompt, and non-reasoning examples (direct answers) with `detailed thinking off`. The model learns to condition its behavior on the flag — it associates the `on` prompt with producing a `<think>` block and the `off` prompt with skipping it. At inference the flag is just a system-prompt string, so switching modes costs nothing and requires no model swap.

### Training the toggle without the modes interfering

The non-obvious difficulty is keeping the two modes from contaminating each other. If you simply mix reasoning and non-reasoning examples without distinguishing them, the model learns an *average* behavior — it reasons a little on everything, which is the worst of both worlds (too slow for easy questions, too shallow for hard ones). The mode label is what prevents this: by conditioning every training example on its `detailed thinking on/off` system prompt, the model learns *two distinct conditional distributions* rather than one blended one. The `on` distribution is "produce a `<think>` block then answer," the `off` distribution is "answer directly," and the system prompt is the switch that selects between them. The data balance matters too — enough `on` examples to make the reasoning sharp, enough `off` examples to make the direct mode genuinely terse — and the report tunes this so neither mode degrades. The general technique, *conditional behavior via a labeled control token*, is reusable far beyond reasoning: any time you want one model to exhibit two behaviors on demand, label the training data with the desired mode and condition on a control string, and the model will learn to switch.

### Second-order optimization: controllability is an efficiency feature

The deeper point is that **controllability is a first-class efficiency lever, not a UX nicety**. A reasoning model that cannot turn off its reasoning is a model that overspends on every easy request. By making reasoning *opt-in per request*, the toggle lets a deployment serve the long tail of hard questions with full chain-of-thought while serving the bulk of easy questions cheaply — the same economics as a fast/slow model cascade, but with one model instead of two. For anyone deploying reasoning models, the toggle is the difference between paying the reasoning tax on 100% of traffic and paying it only on the fraction that needs it. Building controllability into the model — rather than bolting routing on top — is the cleaner design.

## 4. Post-training: distill reasoning, then sharpen it

Architecture gives you a fast model; it does not give you a reasoning one. Reasoning comes from the post-training stack, which is staged deliberately: teach the reasoning behavior by imitation first, then improve it with reinforcement learning.

![Timeline of the post-training stages: reasoning SFT using chains distilled from DeepSeek-R1 across math, code, and science; then RLVR with GRPO using verifiable rewards for step-wise reasoning; then RPO for chat preference alignment; then iterative DPO to refine tool use and agentic flows; ending at the toggleable Llama-Nemotron model](/imgs/blogs/llama-nemotron-efficient-reasoning-models-5.webp)

The stages, in order:

1. **Reasoning SFT (distilled from DeepSeek-R1).** The model is fine-tuned on a large corpus of chain-of-thought traces distilled from a strong open reasoning teacher — DeepSeek-R1 — across math, code, and science, plus non-reasoning examples for the `off` mode. This is the imitation phase: the student learns *what reasoning looks like* by copying a teacher that already reasons well. It is the same distillation principle from the [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) and [distillation](/blog/machine-learning/large-language-model/distillation-in-llm) discussions, applied to reasoning traces rather than logits: R1's traces are excellent supervision, and copying them is far cheaper than discovering reasoning from scratch with RL.
2. **RLVR — RL with Verifiable Rewards (GRPO).** Imitation gets you to the teacher's level; to *exceed* it, the model is trained with reinforcement learning where the reward comes from a **verifier**, not a learned reward model (§5). For math and code, you can mechanically check whether an answer is right, so the reward is the verifier's pass/fail signal, and [GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) updates the policy to produce more correct chains. This is where LN-Ultra's reasoning surpasses its distillation teacher.
3. **Preference alignment (RPO).** [Reward-aware Preference Optimization](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — the method from the Nemotron-4 340B report — aligns the model's chat behavior and helpfulness, so the reasoning model is also a good conversational assistant.
4. **Iterative DPO for tool use.** A final stage refines agentic behaviors — tool calling, RAG, multi-turn — with iterative [DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo), so the model works in real agentic workflows, not just on benchmark math.

The staging matters for the same reason it did in the [Nemotron-4 340B recipe](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment): SFT and RL optimize different objectives (imitation vs. reward maximization) and preference methods optimize yet another (alignment), and interleaving them muddies each. Distill first to get a strong starting policy cheaply, then spend the expensive RL compute improving from a good initialization rather than a random one.

### Why distill from DeepSeek-R1 specifically

The choice of DeepSeek-R1 as the reasoning teacher is deliberate and worth unpacking. R1 was, at the time, the strongest *open* reasoning model, with traces that exhibit the behaviors you want a student to learn: long chains of thought, explicit self-correction ("wait, that's wrong, let me reconsider"), and verification of intermediate steps. Distilling from it means the student inherits these reasoning *patterns* — not just correct answers, but the *process* of reaching them, including the backtracking and checking that distinguish good reasoning from confident guessing. An open teacher matters for a second reason: the traces are clean to use and redistribute, the same licensing logic that drove [Nemotron-4's permissive release](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment). The general lesson is that when you distill a *behavior* (reasoning) rather than just outputs, the quality and style of the teacher's process matters as much as its accuracy — you are copying how it thinks, so pick a teacher that thinks the way you want your student to. This is the same principle behind [distilling R1 into smaller models](/blog/machine-learning/large-language-model/distillation-in-llm): the traces are the product.

### Second-order optimization: distill to imitate, RL to exceed

The division of labor between SFT and RL is worth stating as a rule. **Distillation gets you to the teacher; RL gets you past it.** SFT on R1 traces can, at best, reproduce R1's reasoning — the student imitates, and imitation has a ceiling at the teacher's level. RLVR breaks that ceiling because its reward is *ground truth* (a verifier), not the teacher's behavior: the model can discover reasoning chains R1 never produced, as long as they pass the verifier. This is why the pipeline does both rather than either alone — distillation is the cheap way to a strong start, and verifiable-reward RL is the only way to exceed the teacher you distilled from. If you only have a fixed annotation or distillation budget, SFT is the efficient first move; if you want to push past the source model, you need verifiable RL.

## 5. RLVR: reward what you can verify

The reinforcement-learning stage deserves its own treatment because the *source of the reward* is the key design choice, and it is different from classic RLHF.

![Graph showing RLVR: a math or code prompt fans out to three reasoning rollouts A, B, and C, all of which feed a verifier (an answer key or unit tests) that emits a 0 or 1 reward, which drives a group-normalized GRPO update to the policy](/imgs/blogs/llama-nemotron-efficient-reasoning-models-6.webp)

Classic [RLHF](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) trains a *learned reward model* on human preferences and optimizes the policy against it. That works for subjective qualities (helpfulness, tone) but it is **gameable** — the policy can find adversarial inputs the reward model scores highly but humans would not, the reward-hacking failure that haunts RLHF. For reasoning, there is a better signal available: **verifiable correctness**. A math answer is right or wrong against the answer key. Code passes the unit tests or it does not. You do not need a learned model to judge these — you need a *verifier*, and a verifier cannot be gamed, because it checks the actual answer.

**RLVR (RL with Verifiable Rewards)** uses exactly this. The loop:

1. Sample a group of $G$ reasoning rollouts for a prompt (several independent chains of thought).
2. **Verify** each rollout's final answer mechanically — answer key for math, unit tests for code — producing a binary 0/1 reward.
3. Update the policy with **GRPO**, which normalizes the rewards *within the group* (subtract the group mean) to form advantages, then does a clipped policy-gradient step — no separate critic needed, which is GRPO's whole appeal (covered in depth in the [GRPO guide](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) and the [GRPO/DPO/PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide)).

The group normalization is the clever part: by comparing rollouts *to each other* on the same prompt, GRPO gets a baseline for free (the group mean) without training a value function. A rollout that solves a problem most of its siblings failed gets a strong positive advantage; one that fails where others succeeded gets pushed down.

```python
import torch

def rlvr_grpo_step(policy, prompt, verifier, group_size=8):
    # 1. Sample a group of reasoning rollouts.
    rollouts = [policy.generate(prompt) for _ in range(group_size)]
    # 2. Verifiable reward: mechanically check each final answer (0 or 1).
    rewards = torch.tensor([float(verifier(prompt, r)) for r in rollouts])
    # 3. GRPO advantage: normalize within the group (no learned critic).
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    loss = 0.0
    for rollout, a in zip(rollouts, adv):
        logp = policy.log_prob(prompt, rollout)
        loss = loss - a * logp                  # clipped PG in the full implementation
    return loss / group_size
```

The reason RLVR is what lets LN-Ultra *beat* DeepSeek-R1 rather than merely match it: distillation copied R1's chains (ceiling = R1), but RLVR optimizes against ground-truth correctness (ceiling = the verifier, i.e. actual correctness), so the model can climb past the teacher on exactly the tasks where correctness is checkable. Large-scale RLVR is the expensive, decisive final stage.

### Why GRPO and not PPO

A natural question: why GRPO rather than the PPO that classic RLHF uses? PPO needs a **critic** — a learned value function that estimates the expected reward from a given state, used as a baseline to reduce the variance of the policy gradient. Training a critic at the scale of a 253B model is expensive and finicky: it is a second large network, it has to be trained alongside the policy, and a bad critic destabilizes the whole thing. GRPO's insight is that you can get the baseline *for free* by sampling a **group** of rollouts on the same prompt and using the group's mean reward as the baseline. A rollout is judged not against a learned value estimate but against *its siblings on the same problem* — did this chain do better or worse than the other attempts at this exact prompt? That relative signal is a clean, low-variance advantage with no critic to train. For RLVR specifically, where the reward is a cheap binary verifier call, sampling a group is inexpensive, so GRPO's "more rollouts instead of a critic" trade is strongly favorable. The result is a simpler, more stable RL loop — no critic to diverge — which matters enormously when each training run costs a fortune in GPU time.

### Exploration and the entropy of reasoning chains

A subtle property of RLVR on reasoning is that it needs the policy to *explore* different chains of thought, and the group-sampling structure provides exactly that. Each of the $G$ rollouts on a prompt is a different attempt — a different reasoning path — and the verifier tells you which paths reached the right answer. Over training, the policy shifts probability toward the *kinds* of reasoning that tend to verify, not toward a single memorized chain. This is why RLVR can discover reasoning strategies the distillation teacher never demonstrated: it is not imitating a fixed target, it is searching the space of chains under a correctness oracle. Maintaining enough exploration (entropy) in the rollouts is part of the art — too little and the policy collapses onto one chain prematurely, too much and it never converges — which is the same exploration/exploitation balance that the broader [GRPO-variant literature](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) spends much of its effort tuning.

### Second-order optimization: prefer verifiers to learned rewards wherever you can

The principle generalizing RLVR: **a verifier beats a learned reward model whenever the task is checkable**, because a verifier is ground truth and a learned reward is an approximation that can be gamed. The frontier of reasoning RL — from DeepSeek-R1 to this report — runs on verifiable rewards precisely because math and code give you free, unhackable oracles. The strategic implication for your own work is to *engineer verifiability into your tasks* wherever possible: a task with a checkable answer can be improved with RLVR far more reliably than one that needs a learned judge. This is the same lesson as the [Nemotron-4 reward modeling post](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — use the strongest judge available, and a verifier is the strongest judge there is.

## 6. The family: three sizes, three Llama parents

The recipe is applied at three scales, each derived from a different Llama parent, to cover the deployment spectrum from on-device to data-center.

![Matrix showing the three Llama-Nemotron models: Nano derived from Llama-3.1-8B at 8B params for on-device reasoning; Super derived from Llama-3.3-70B at 49B params with 5x throughput over its parent; Ultra derived from Llama-3.1-405B at 253B params, beating DeepSeek-R1 on GPQA while running on 8 H100s](/imgs/blogs/llama-nemotron-efficient-reasoning-models-7.webp)

- **LN-Nano (8B)** from Llama-3.1-8B — small enough for on-device and edge reasoning, the model you reach for when latency and footprint dominate.
- **LN-Super (49B)** from Llama-3.3-70B — note the model is *smaller than its parent* (49B from 70B), because Puzzle compressed it, and it serves **5× the throughput at batch 256**. This is the workhorse serving tier.
- **LN-Ultra (253B)** from Llama-3.1-405B — the frontier model, compressed from 405B to 253B, that matches or beats DeepSeek-R1 on reasoning while fitting a single 8×H100 node.

The pattern is the same at every scale: take a strong Llama parent, reshape it smaller and faster with Puzzle and FFN Fusion, heal it with distillation, then install reasoning. The parent provides the latent capability; the NAS provides the efficiency; the post-training provides the reasoning. Note the through-line to the rest of the series — this is the *same strategic move* as Minitron (derive a family from strong parents rather than training each from scratch), executed with architecture search instead of uniform pruning.

### What the heterogeneity buys, concretely

It is worth grounding the "heterogeneous architecture" abstraction in what it actually changes about serving. A uniform dense model pays, on every layer, for full attention — which means a full KV-cache contribution per layer, full quadratic-in-context attention compute, and full FFN width. When Puzzle removes attention from a layer, three things drop at once for that layer: the attention FLOPs (gone), the KV-cache memory (gone — that layer stores no keys and values), and the sequential attention step on the critical path (gone). Multiply that across the many layers Puzzle strips and the savings are structural, not marginal. For LN-Ultra, the combination of attention removal and FFN fusion is what turns a 253B model — which, as a uniform dense reasoner, would be brutal to serve — into something that fits 8×H100 and beats an 8×H200 model. The heterogeneity is not an aesthetic quirk of the search; it is a direct, measurable reduction in the three things that bound large-model serving: compute, memory, and critical-path latency. When you look at a Puzzle architecture and see layer 3 with no attention, you are looking at a layer that costs a fraction of what its uniform counterpart would, and the model's overall serving profile is the sum of many such savings.

### Second-order optimization: the parent choice is a capability ceiling

A subtle point: each Llama-Nemotron is bounded by its parent's latent capability, the same way a [Minitron-compressed model](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) is. Puzzle and distillation can *reshape and preserve* what the parent knows, and RLVR can *sharpen* its reasoning, but none of them can install knowledge the 405B parent never learned. This is why the family tops out at what Llama-3.1-405B made possible. The lesson for choosing a starting point is that the parent sets your ceiling — pick the strongest parent you can afford to reshape, because everything downstream is bounded by what it already contains.

## 7. The payoff: reasoning that fits the budget

Put the architecture work and the reasoning work together and you get the result that makes the report matter: reasoning models that are *cheaper to serve than their non-reasoning parents*.

![Before-and-after comparison: on the left, the Llama-3.3-70B-Instruct parent is a dense uniform transformer with full attention on every layer, baseline throughput, and no reasoning toggle; on the right, LN-Super 49B is heterogeneous with attention removed from many layers and FFNs fused for parallel execution, delivering 5x throughput at batch 256 with a reasoning toggle](/imgs/blogs/llama-nemotron-efficient-reasoning-models-8.webp)

The headline efficiency numbers:

| Model | Parent | Params | Efficiency win | Reasoning result |
|---|---|---|---|---|
| LN-Nano | Llama-3.1-8B | 8B | on-device footprint | strong small-model reasoning |
| LN-Super | Llama-3.3-70B | 49B | **5× throughput** (batch 256) | competitive with much larger models |
| LN-Ultra | Llama-3.1-405B | 253B | **1.71× latency**, 8×H100 vs R1's 8×H200 | GPQA 76.0, AIME-24 80.8, MATH-500 97.0 |

The LN-Ultra result is the one to dwell on. DeepSeek-R1 is a mixture-of-experts model that needs 8×H200 to serve; LN-Ultra matches or beats it on GPQA-Diamond, AIME, and MATH while running on **8×H100** — older, cheaper, more available hardware — at lower latency. That is not a marginal efficiency gain; it is a model that is *both* state-of-the-art on reasoning *and* meaningfully cheaper to deploy, which is the combination the field had been treating as a trade-off. The architecture work (Puzzle + Fusion) is what closed the gap: a uniform 253B dense reasoning model would not fit the budget, but a heterogeneous, attention-pruned, FFN-fused 253B does.

### Second-order optimization: co-design the model and the serving target

The meta-lesson of the whole report is **co-design**. The 49B and 253B parameter counts, the per-layer attention removals, the FFN fusions — none of them were chosen in the abstract. They were chosen by Puzzle's MIP solver to fit specific GPUs at specific latency budgets. The model is shaped by the serving target rather than the serving target accommodating a pre-chosen model. For teams that serve models at scale, this is the most actionable idea in the report: stop treating the model architecture and the serving hardware as independent, and start searching for the architecture that is optimal *for your hardware and your latency SLA*. The tools to do this — block libraries, local distillation, MIP assembly — are exactly what Puzzle provides.

## 8. Failure modes the recipe guards against

As with the rest of the series, the recipe is best understood as a set of safeguards against specific failure modes. Here is the catalog.

- **Architectural surgery breaking the model.** Symptom: removing attention or shrinking FFNs tanks accuracy. Cause: the cut changed the model's computation and nothing healed it. Safeguard: block-wise local distillation builds each replacement to mimic the original, and a global knowledge-distillation pass heals the assembled model (§1).
- **Searching a space too large to explore.** Symptom: you cannot evaluate every per-layer combination. Cause: the architecture space is exponential. Safeguard: local distillation makes each candidate cheap to build and profile, and MIP makes the selection optimal without brute force (§1).
- **Optimizing FLOPs instead of latency.** Symptom: a model with low FLOPs is still slow. Cause: deep sequential critical paths. Safeguard: FFN Fusion flattens sequential FFN depth into parallel width (§2).
- **Reasoning on every request.** Symptom: easy questions cost as much as hard ones. Cause: a reasoning model that always reasons. Safeguard: the `detailed thinking on/off` toggle makes chain-of-thought opt-in (§3).
- **Reward hacking in RL.** Symptom: the policy games a learned reward model. Cause: an approximate, gameable reward. Safeguard: RLVR uses verifiers (answer keys, unit tests) that cannot be hacked (§5).
- **Hitting the distillation ceiling.** Symptom: the model never exceeds its R1 teacher. Cause: pure imitation caps at the teacher. Safeguard: RLVR optimizes against ground-truth correctness, which lets the model surpass the teacher (§4, §5).
- **Compressing past the parent's capability.** Symptom: the reshaped model lacks knowledge. Cause: trying to get capability the parent never had. Safeguard: choose a strong parent — the architecture work preserves and sharpens, it does not create (§6).

The meta-lesson echoes the earlier posts: each design choice is a "don't" — don't cut without healing, don't optimize FLOPs without latency, don't reason on everything, don't trust a gameable reward — and the recipe is the disciplined accumulation of those don'ts.

## 9. Case studies from the report

### 1. LN-Ultra beats R1 on cheaper hardware

The flagship result: LN-Ultra (253B, from Llama-3.1-405B) matches or exceeds DeepSeek-R1 on GPQA-Diamond (76.0), AIME-24 (80.8), and MATH-500 (97.0), while running on a single 8×H100 node versus R1's 8×H200, at 1.71× lower latency. This is the existence proof that reasoning quality and serving efficiency are not fundamentally opposed — the architecture work bought the efficiency without sacrificing the reasoning. The lesson the field took: a heterogeneous, hardware-searched architecture can deliver frontier reasoning at a fraction of the serving cost of a uniform dense or MoE model, which reframes "how do we afford to serve reasoning models" from a hardware-procurement question into an architecture-search question.

### 2. LN-Super's 5× throughput from a smaller model

LN-Super is 49B distilled from a 70B parent, and it serves 5× the throughput of Llama-3.3-70B-Instruct at batch 256 — *while adding reasoning the parent did not have*. The model got smaller, faster, and more capable simultaneously, which violates the usual intuition that you trade one for another. The mechanism is that Puzzle removed redundant attention and FFN capacity (smaller, faster) and the post-training installed reasoning (more capable), and these are independent levers. The case study is the clearest demonstration that compression and capability are decoupled when you compress the *architecture* and add capability through *training*.

### 3. The attention-removal insight

Puzzle's most radical move — dropping attention from entire layers — is worth treating as its own case study because it overturns a default assumption. Every transformer is built as if attention is needed in every layer; Puzzle showed that for many layers it is not, and removing it is the single highest-value efficiency cut because attention owns the KV cache and the quadratic-in-sequence-length cost. For reasoning models running at long context, the KV-cache savings alone are large. The transferable insight is to question per-layer uniformity: the assumption that every layer needs the same blocks is a convenience of how we *build* transformers, not a property of how they *work*, and relaxing it unlocks efficiency.

### 4. FFN Fusion as a Puzzle-enabled optimization

FFN Fusion's 1.71× latency gain on LN-Ultra is a case study in compounding optimizations. Fusion is only possible *because* Puzzle removed attention, leaving consecutive FFNs to fuse. Neither technique alone delivers the full gain — Puzzle's attention removal saves compute and KV cache, and it *also* creates the precondition for fusion to flatten the resulting sequential depth into parallel width. The lesson is to look for enabling relationships between optimizations: the biggest wins often come not from a single clever trick but from one transformation that unlocks a second.

### 5. The reasoning toggle as a deployment economics decision

The `detailed thinking on/off` toggle is a case study in building controllability into the model rather than the serving layer. Without it, deploying a reasoning model means either paying the chain-of-thought cost on every request or running a separate fast model and routing — both wasteful. With it, one deployment serves easy questions cheaply (`off`) and hard questions thoroughly (`on`), with the choice made per request by a system-prompt string. The lesson is that the most impactful efficiency features are sometimes behavioral, not architectural: a model that can *choose* not to reason saves more than any kernel optimization on the easy-question majority of real traffic.

### 6. Distilling R1, then surpassing it with RLVR

The two-phase reasoning recipe — SFT on R1 traces, then RLVR — is a case study in the division of labor between imitation and reinforcement. The SFT phase is cheap and gets the model to roughly R1's level by copying its chains; the RLVR phase is expensive and pushes past R1 by optimizing against verifiable correctness. The model could not have reached its final reasoning level by either alone: SFT caps at the teacher, and RLVR from a random initialization would be prohibitively expensive. The lesson, reusable across any capability with a verifier, is distill-then-RL: imitate to get a strong cheap start, then reinforce against ground truth to exceed the source.

### 7. MIP assembly and hardware co-design

The use of mixed-integer programming to assemble the architecture is a case study in treating model design as constrained optimization. Rather than hand-tuning which layers keep attention, Puzzle states the hardware constraints (latency, memory on a specific GPU) as MIP constraints and *solves* for the quality-maximizing architecture. Change the target GPU and you re-solve and get a different model from the same block library. The lesson is that for deployment-critical models, architecture is a search problem with a precise objective and hard constraints — exactly the shape a solver handles — and offloading it to a solver beats human guessing.

### 8. Continued pretraining to heal general capability

After the architectural surgery and distillation, the pipeline includes continued pretraining (tens of billions of tokens) before the reasoning post-training. This is the general-capability heal: the aggressive Puzzle cuts and local distillation focus on matching block behavior, but the assembled model needs a broad pretraining pass to fully recover the parent's general competence before reasoning is layered on. It echoes Minitron's distillation-based retraining — compression damages the model, and you must heal it with training before building on top. The lesson is to budget a recovery phase after any aggressive architectural change; skipping it leaves capability on the table that the later stages cannot recover.

### 9. Reasoning at long context and the KV-cache win

A consequence worth its own note: because Puzzle removes attention from many layers, Llama-Nemotron models have a much smaller KV cache than their dense parents, which is exactly what reasoning models need. Reasoning runs at long context — the chain of thought *is* a long sequence — and KV-cache memory grows linearly with context length, so it is the binding constraint on long-context throughput (see [KV cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management)). By cutting attention layers, Puzzle cuts the KV cache, which is why these models can serve long reasoning traces efficiently. The lesson is that efficiency techniques compound with use cases: attention removal happens to be most valuable exactly where reasoning models spend their time, at long context.

### 10. One model instead of a fast/slow cascade

A practical operational case study: the reasoning toggle collapses what would otherwise be a two-model serving architecture (a fast chat model plus a slow reasoning model, with a router deciding between them) into a single model. This halves the serving footprint, eliminates the router as a point of failure and a source of latency, and removes the operational burden of keeping two models in sync. For a production team, the toggle is not just a feature — it is a simplification of the entire serving stack. The lesson is that capability controllability inside one model can replace infrastructure complexity outside it, and the inside-the-model version is usually cheaper to operate.

### 10b. The toggle's economics, in numbers

To make the toggle's value concrete, consider a realistic traffic mix. Suppose 80% of requests are easy (a lookup, a short rewrite, a simple question) and 20% genuinely need reasoning. An always-on reasoning model emits a long chain-of-thought on all 100% — say 800 output tokens average — so you pay for 100 units of reasoning-length generation. With the toggle, the 80% easy requests run in `off` mode at perhaps 50 tokens, and only the 20% hard requests pay the 800-token reasoning cost: your generation volume drops to roughly $0.8 \times 50 + 0.2 \times 800 = 200$ token-units versus $100 \times 800 = 800$ for the always-on model serving the same traffic — a 3–4× reduction in generated tokens, which at decode-bound serving translates almost directly into cost and latency. The exact numbers depend on your traffic, but the structure is universal: reasoning cost is concentrated in the hard-question minority, and a toggle lets you avoid paying it on the easy-question majority. This is why the toggle is not a minor convenience — on realistic traffic it is the difference between a reasoning model you can afford to serve broadly and one you ration. The lesson is to measure your traffic's reasoning-need distribution; the more skewed toward easy, the more the toggle saves.

### 11. Reshaping a 70B into a 49B that beats it

LN-Super is worth a second look as a compression case study specifically. The parent is Llama-3.3-70B; the derivative is 49B — a 30% parameter reduction — and yet it is faster (5×) *and* gains reasoning the parent never had. Compare this to naive size reduction: a uniformly-shrunk 49B trained from scratch would be a strictly weaker model than the 70B, because it has less capacity and less training. Puzzle's 49B is *not* weaker, because it removed only the *redundant* capacity (the attention layers and FFN width that contributed little) while preserving the load-bearing parts, and the reasoning post-training added capability on top. The case study refutes the intuition that fewer parameters means less capable: fewer *redundant* parameters means faster at the same capability, and the post-training can then push capability up. The redundancy was always there; Puzzle just found and removed it.

### 12. The same playbook as Minitron, with a sharper tool

It is instructive to compare Puzzle to [Minitron's pruning](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) directly, because they solve the same problem — derive a smaller model from a strong parent — with different tools. Minitron prunes *uniformly-scored dimensions* (heads, neurons, channels, layers) by activation importance, producing a smaller but still *homogeneous* model. Puzzle searches *heterogeneous per-layer architectures* under hardware constraints, producing a model where layers differ. Puzzle is the sharper, more expensive tool: it can express "this layer keeps full attention, that one has none," which uniform pruning cannot, and it optimizes explicitly for hardware latency rather than just parameter count. The trade is complexity — Puzzle needs the block library, the profiler, and the MIP solver, where Minitron needs only importance scores and a retrain. The case study is a lesson in matching tool to constraint: if you just need a smaller model, prune; if you need a model optimal for a specific serving target, search.

### 13. Continued pretraining as the bridge between surgery and reasoning

The continued-pretraining stage (tens of billions of tokens) between the architectural surgery and the reasoning post-training is easy to overlook but structurally important. Local distillation matches each block's behavior, and a global heal reconciles the assembled model, but the result is still a model whose general capability has been perturbed by aggressive cuts. Continued pretraining on a broad corpus restores that general competence to a level where the reasoning SFT and RL can build on solid ground. Skip it and the reasoning post-training is building on a shakier base, and the final model underperforms. The case study reinforces a theme from across the series: aggressive efficiency surgery always needs a recovery phase before you layer new capability on top — Minitron heals with distillation retraining, Puzzle heals with local distillation plus continued pretraining, and in both the heal is non-optional.

### 14. The toggle changes how you evaluate the model

A subtle consequence of the reasoning toggle is that it complicates — and improves — evaluation. A toggleable model has *two* behaviors to measure: its `off` mode (fast, direct) should be benchmarked against chat models, and its `on` mode (reasoning) against reasoning models. The report measures both, and the interesting finding is that the same weights are competitive in *both* regimes — a strong chat model when reasoning is off and a strong reasoning model when it is on. This is harder to achieve than it sounds, because the two modes pull in different directions (terse vs. elaborate), and training one model to do both well without the modes interfering is a real achievement of the mode-labeled SFT data. The case study is a reminder that controllability has evaluation consequences: a model with modes must be evaluated per mode, and a good toggleable model is competitive in each.

### 15. Re-solving the MIP for a new hardware target

A capability that does not show up in a single benchmark number but matters enormously in practice: because the expensive work (block-library construction via local distillation) is done once and the MIP is cheap to re-solve, Puzzle can produce *different* architectures for *different* deployment targets from the same library. Want a variant tuned for an A100 instead of an H100, or for a batch-1 latency-critical path instead of batch-256 throughput? Re-run the solver with the new constraints and out comes a new architecture, no retraining of the library required. This amortizes the expensive part across many derived models. The case study is a lesson in where to put your fixed costs: build the reusable, expensive artifact (the block library) once, and make the per-target decision (the MIP solve) cheap, so you can serve many hardware targets from one investment.

### 16. Why this is the natural sequel to Minitron and Nemotron-4

Reading the three NVIDIA reports in sequence reveals a deliberate progression. [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) established "derive small models from a strong parent by pruning and distillation." [Nemotron-4 340B](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) established "align with synthetic data judged by a strong reward model, and improve iteratively." Llama-Nemotron combines and extends both: it derives models from strong Llama parents (Minitron's thesis) using a sharper architecture-search tool (Puzzle), and it post-trains them with a distill-then-RL stack that uses RPO (Nemotron-4's method) for alignment and RLVR for reasoning. The case study is the series itself: NVIDIA is assembling a coherent toolkit — compress from strong parents, align with synthetic data and verifiable rewards, search architectures for hardware — and each report adds a tool that composes with the others. Llama-Nemotron is where the compression toolkit (Puzzle) meets the alignment toolkit (RPO + RLVR) to produce a model that is simultaneously efficient and capable.

### 17. The open release and the agentic framing

Like the rest of the series, Llama-Nemotron ships openly, and the framing is explicitly *agentic* — the models are post-trained for tool calling, RAG, and multi-turn workflows, not just benchmark reasoning. This matters because a reasoning model that cannot use tools or follow multi-turn instructions is a benchmark artifact, not a usable agent. The iterative-DPO tool-use stage (§4) is what makes the models work in real agentic flows, and the open release means teams can build on them. The case study is a reminder that frontier reasoning is necessary but not sufficient for real deployment: the model also has to be a competent agent, which is a separate post-training investment that the report deliberately makes. A model that reasons brilliantly but cannot call a function is not an agent; Llama-Nemotron is post-trained to be both.

## The bigger picture: the architecture is a compiler target

Step back and the Llama-Nemotron report makes a claim that goes beyond reasoning models: it treats the model architecture the way a compiler treats source code — as something to be *optimized for a specific target* rather than fixed in advance. In conventional practice, you choose an architecture (a size off the ladder, uniform layers) and then try to make it run well on your hardware with kernel tricks, quantization, and batching. Puzzle inverts the relationship. The hardware is the target; the architecture is the thing being compiled to fit it. Give Puzzle a different GPU or a tighter latency SLA and it emits a different architecture, the way a compiler emits different machine code for different processors. This is a genuinely different mental model for model deployment, and it is the most portable idea in the report.

The implications compound. First, it means **there is no single "best" architecture** — there is a best architecture *for a target*, and the target includes the hardware, the latency budget, the memory limit, and the workload (short context vs long, batch-1 vs batch-256). Second, it means **the heterogeneity that looks strange is actually optimal**: a model where layer 3 has no attention and layer 4 has a 10%-width FFN is not a hack, it is the solver's answer to "maximize quality subject to this latency budget." Third, it means **efficiency work moves upstream**, from runtime kernel optimization into architecture search, where the gains are larger because you are changing what computation happens, not just how fast a fixed computation runs.

For the field, the trajectory this points to is models that are *specialized to deployment targets by search* rather than chosen by convention. Combined with the rest of the series — [Minitron's compression](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), [Nemotron-4's synthetic alignment](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — a coherent NVIDIA philosophy emerges: train strong large models once, then *derive* deployment-ready models from them by compression, architecture search, and distillation, each derivation tuned to its target. The big training run is the capital investment; the derived family is where the value is realized, and architecture search is how you realize it for each specific serving constraint. That is the production reality the next generation of efficient models is being built around.

### 18. Heterogeneous architectures and the kernel-writing burden

A practical caveat the report's success papers over: a heterogeneous architecture is harder to serve *well* than a uniform one, because your inference stack has to handle layers with different shapes — some with attention, some without, FFNs of varying widths. Uniform models benefit from highly-tuned kernels that assume every layer is identical; a Puzzle model breaks that assumption and needs an inference engine that can execute a per-layer-varying graph efficiently. NVIDIA can absorb this because they control the full stack (TensorRT, the kernels, the hardware), which is part of why Puzzle is *their* technique — the architecture-search win is only realized if your serving stack can run the searched architecture at the latency the MIP assumed. The case study is a reminder that an optimization is only as good as your ability to deploy it: a brilliant heterogeneous architecture that your inference engine serves inefficiently gives back the gains. For teams without deep inference-engine control, this is the strongest argument for the simpler uniform-pruning path, and the strongest reason Puzzle pairs naturally with NVIDIA's own serving tooling.

### 19. Reasoning models as the new efficiency frontier

Zoom out and Llama-Nemotron marks a shift in where the efficiency action is. For years, LLM efficiency meant making a *fixed* model run faster — quantization, better kernels, KV-cache tricks, speculative decoding. Llama-Nemotron's contribution is to push efficiency *into the architecture and the behavior*: reshape the model for the hardware (Puzzle), flatten its depth (Fusion), and let it skip reasoning when unneeded (the toggle). These are not runtime optimizations on a fixed model; they change what the model *is*. The case study is the trend itself: as reasoning models make inference the dominant cost (long chains of thought are a lot of tokens), efficiency work is migrating upstream from the serving runtime into model design and training. The teams that win the reasoning-model cost race will be the ones who treat efficiency as an architecture-and-training problem, not just a kernel problem — which is exactly the bet this report makes.

## When to reach for the Llama-Nemotron playbook — and when not to

**Reach for it when:**

- **You serve under hard latency or hardware constraints.** Puzzle's whole premise is searching for the architecture that fits *your* GPU and SLA. If you have a fixed serving budget, a hardware-searched heterogeneous model can beat a uniform one of the same size.
- **You have a strong parent model to reshape.** Like Minitron, this works because the parent has latent capability. Reshape a strong model; do not expect to reshape a weak one into a strong one.
- **You want reasoning without paying for it on every request.** The toggle is the reason to choose a controllable reasoning model over an always-on one.
- **Your reasoning tasks are verifiable.** RLVR shines on math, code, and science where correctness is checkable. If you can verify answers, you can push past your distillation teacher.
- **You run at long context.** Attention removal's KV-cache savings pay off most at the long sequences reasoning requires.

**Skip it (or be careful) when:**

- **You lack the infrastructure for NAS.** Puzzle needs a block-library-building pipeline (local distillation), a latency/memory profiler for your hardware, and a MIP solver. That is real engineering; for a one-off model, [uniform pruning à la Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) is simpler.
- **Your task is not verifiable.** RLVR's advantage evaporates without a verifier; for purely subjective tasks you are back to learned reward models and their hacking risks.
- **You need a capability the parent lacks.** Architecture search preserves and sharpens; it does not create. A missing capability is a pretraining problem.
- **Latency is not your constraint.** If you are throughput-bound at short context, the attention-removal and fusion gains are smaller, and a simpler model may suffice.

### A note on the cost and scale of RLVR

One caveat worth stating plainly: the RLVR stage that pushes LN-Ultra past DeepSeek-R1 is *expensive*. Reinforcement learning at the scale of a 253B model, sampling groups of long reasoning rollouts per prompt and verifying each, is a major compute undertaking — far more than the SFT phase that precedes it. This is the inversion that surprises people coming from the supervised-learning world: the cheap phase (distillation SFT) does most of the *capability transfer*, and the expensive phase (large-scale RLVR) does the *last increment* that separates "matches the teacher" from "beats the teacher." Whether that last increment is worth its cost is a real engineering decision. For a frontier model where being best-in-class matters, it is; for a model where matching a strong teacher is good enough, the SFT-only path is dramatically cheaper and gets you most of the way. The report does the full RLVR because LN-Ultra is meant to be frontier; a team with a tighter budget can stop after distillation and still ship a strong reasoner. Knowing where the cost lives — and that it lives in the final RL increment — lets you make that trade deliberately rather than discovering it on the compute bill.

The one-sentence version:

> Stop choosing between fast and smart: search the architecture for your hardware so the model is fast, distill reasoning from a strong teacher and sharpen it with verifiable-reward RL so the model is smart, and toggle the chain-of-thought so you pay for thinking only when it pays you back.

## Further reading

- [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949) — the full report, with the Puzzle and FFN Fusion details and the complete benchmark tables.
- [Minitron: pruning and distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) — the uniform-compression sibling in this NVIDIA series.
- [Nemotron-4 340B synthetic-data alignment](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment) — where RPO, used here for chat alignment, comes from.
- [Fine-tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the RL algorithm behind RLVR.
- [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — where verifiable-reward RL sits among preference methods.
- [Knowledge distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) — the imitation phase that bootstraps reasoning.
- [KV cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — why attention removal's KV savings matter at long context.

One closing reflection ties the report to where the field is heading. The reasoning-model era made inference the dominant cost of using LLMs — the chain of thought *is* the product, and it is a lot of tokens. That cost pressure is forcing efficiency upstream, out of the serving runtime and into model design itself. Llama-Nemotron is an early, complete example of what that looks like in practice: an architecture searched for the hardware, a depth flattened into width, a behavior that can be switched off when it is not needed, and a reasoning capability installed by distillation and sharpened by verifiable-reward RL. None of these is a runtime trick on a fixed model; each one changes what the model is. As reasoning models proliferate, expect more of the efficiency work to look like this — architecture-and-training problems rather than kernel problems — and expect the techniques in this report, Puzzle and FFN Fusion above all, to recur in new guises. The model is becoming a compiler target, and the compiler — the search that maps a strong parent and a hardware budget to a deployable architecture — is getting good enough that "fast" and "smart" are no longer the trade-off they used to be.

*Next in the series: Nemotron-H, where NVIDIA replaces most of the attention layers with Mamba-2 state-space layers to make a hybrid that serves long context at near-constant memory.*
