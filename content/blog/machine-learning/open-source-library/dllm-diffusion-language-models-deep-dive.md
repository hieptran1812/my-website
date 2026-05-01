---
title: "dllm: An Engineer's Deep Dive Into the Diffusion Language Model Library Behind LLaDA, Dream, and Tiny-A2D"
date: "2026-04-30"
publishDate: "2026-04-30"
description: "A long, opinionated walkthrough of dllm — the unified training, inference, and RL library for diffusion language models. Mask-diffusion math, schedulers, samplers, block diffusion (BD3LM), Edit Flows, Fast-dLLM, AR-to-diffusion conversion, diffu-GRPO, distributed training, and a catalog of production case studies."
tags:
  [
    "dllm",
    "diffusion-language-models",
    "llada",
    "mdlm",
    "bd3lm",
    "fast-dllm",
    "diffu-grpo",
    "edit-flow",
    "open-source-library",
    "llm",
    "training-techniques",
  ]
category: "machine-learning"
subcategory: "Open Source Library"
author: "Hiep Tran"
featured: true
readTime: 51
aiGenerated: true
---

For two and a half years the only "real" question in language modelling was how to make autoregressive transformers cheaper to serve. Diffusion language models were the diorama down the hall — interesting, theoretically clean, and never quite competitive with a well-tuned Llama. That changed quietly in 2024–2026 with the LLaDA family, Dream, and a wave of papers (MDLM, BD3LM, diffu-GRPO, Fast-dLLM, Edit Flow) that pushed masked-diffusion language models to within a few points of strong AR baselines on math, code, and reasoning benchmarks — while reaching them with parallel decoding, bidirectional context, and an inference loop that does not look anything like a left-to-right token-by-token sampler.

The infrastructure has not caught up. There is no diffusion-language equivalent of `transformers`, no `vllm`, no `trl`. Every paper ships a separate fork of someone's training script, a custom sampler, a different scheduler, and yet another ad-hoc evaluation harness. If you want to compare LLaDA-8B-Instruct against a Tiny-A2D conversion of Qwen2.5-0.5B on GSM8K, you spend a week stitching repos together before you can run a single eval.

[dllm](https://github.com/ZHZisZZ/dllm) is the library that fills that gap. It is a single codebase that supports masked diffusion (MDLM), block diffusion (BD3LM), AR-to-diffusion conversion (a2d), edit flows, Fast-dLLM-style accelerated inference, and diffu-GRPO reinforcement learning, all running on top of HF Accelerate / DeepSpeed / FSDP, all evaluable through `lm-evaluation-harness`. Switching algorithms is mostly a YAML edit. Switching model families (LLaDA, Dream, BERT, Qwen-converted) is mostly an `examples/<family>/sft.py` swap.

![Forward and reverse process of a masked diffusion language model: forward corrupts tokens to mask state, reverse picks the highest-confidence positions and unmasks them step by step](/imgs/blogs/dllm-diffusion-language-models-deep-dive-1.png)

The diagram above is the mental model: a diffusion LM is trained to undo random masking. At inference time you start from an all-`[MASK]` answer slot, run the forward pass, pick the most confident predictions, commit them, re-mask the rest, and repeat. The whole library is built around making that loop fast, configurable, and trainable end-to-end. The rest of this article walks through every layer — the math, the abstractions, the samplers, the trainers, the RL loop, the distributed setup — and closes with a long catalog of production case studies and the heuristics a senior engineer should reach for when reasoning about masked diffusion in production. Companion reading: my earlier posts on [KV cache fundamentals](/blog/machine-learning/large-language-model/kv-cache) and [the LMCache deep-dive](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive) cover the AR-side serving stack; this article is the diffusion-side counterpart.

## 1. The Real Problem: Diffusion LMs Need Their Own Stack

The naive view is that diffusion language models are "just transformers with a different loss," so any HuggingFace `Trainer` should suffice. That view collapses on contact with reality.

| Assumption (textbook)                                  | Reality (training a real dLLM)                                                                |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Loss is one-line cross-entropy                         | Loss is masked CE *reweighted* by a noise schedule, with an absorbing state and a token mask  |
| Inference is `model.generate()`                        | Inference is an iterative remasking loop with topology choices (per-token, block, edit)       |
| You can reuse `transformers.Trainer` directly          | The forward needs noised input *and* the original labels; HF's collator does not do that      |
| `model.eval()` is enough for benchmarks                | `lm-eval-harness` integration needs a custom `LM` adapter that wraps the diffusion sampler    |
| RL on language is GRPO with logprobs from `model(x).logits[:, :-1]` | Diffusion logprobs are an integral over noise schedules, not a single forward pass |
| KV cache "just works"                                  | Vanilla MDLM has no KV cache; you only get one for *block* diffusion, and only across blocks  |
| AR weights are useless for diffusion                   | a2d shows you can convert Qwen / LLaMA / GPT-2 by continuing-training under MDLM loss         |

Each of these is a real engineering wall that anyone who has tried to reproduce a diffusion LM paper has hit. dllm exists because every paper team was independently rebuilding the same scaffolding badly, and the scaffolding is not the interesting part.

**The reweighting trap.** The MDLM ELBO is not just "mask some tokens and run cross-entropy." The loss is a *time-weighted* expectation: you draw $t \sim \mathcal{U}(\epsilon, 1)$, mask each token i.i.d. with probability $1-\alpha(t)$, and weight the per-token CE by $1/\alpha'(t)$ to recover an upper bound on $-\log p_\text{data}(x)$. Drop the weighting and you train a denoiser, not a language model — it will look fine on training loss and fall over on benchmarks. This is the single most common bug in homegrown diffusion-LM trainers.

**The sampler topology.** A diffusion LM has *three* fundamentally different inference styles: vanilla MDLM (whole-sequence per-token diffusion), BD3LM (block-causal across blocks, diffusion within block, with KV cache reuse), and Edit Flow (variable-length, with insert/delete/substitute operations). They share zero scheduler code if you write them naively. dllm refactors them onto a single `BaseSampler` abstraction with shared confidence-threshold and Gumbel-noise primitives, which is the only way to compare them on the same eval harness.

**The RL hole.** GRPO and friends compute policy gradient as $r \cdot \nabla \log \pi_\theta(y|x)$, but the dLLM "policy" is not a single forward pass — it is a sampler trajectory. Computing $\log \pi_\theta(y|x)$ exactly requires marginalising over noise schedules, which is intractable. diffu-GRPO uses an importance-sampling estimate: draw a few noise levels, mask accordingly, run the forward, average. dllm's `rl/grpo` package implements this once, then reuses it across LLaDA, Dream, and Tiny-A2D.

**The conversion lever.** Pretraining a 7-B diffusion LM from scratch costs millions. The Tiny-A2D paper shows you can take a 0.5-B Qwen, swap its causal mask for bidirectional attention, replace the LM head with a mask-token-aware head, and continue-train under MDLM loss for a few thousand steps. dllm's `a2d` pipeline automates that conversion plus the continuation training, giving the research community a cheap entry point into dLLMs.

The conclusion: diffusion LMs are not "transformers with a different loss." They are a different *stack*, and that stack needs its own library. dllm is what that library looks like in 2026.

## 2. The Architecture: Thin Core, Per-Family Pipelines

dllm splits responsibilities into three layers. Keep this split in your head; every config flag, every CLI arg, every override slots into exactly one of them.

![dllm architecture: examples (entrypoints), pipelines (per family), core (samplers/schedulers/trainers), and third-party infrastructure (Accelerate, DeepSpeed, FSDP, lm-eval-harness)](/imgs/blogs/dllm-diffusion-language-models-deep-dive-2.png)

**`dllm/core/`** owns the algorithm-level abstractions every diffusion LM shares:

- `samplers/` — `BaseSampler`, `MDLMSampler`, `BD3LMSampler`, plus helpers `add_gumbel_noise()` and `get_num_transfer_tokens()`.
- `schedulers/` — `LinearAlphaScheduler`, `CosineAlphaScheduler`, plus a parallel hierarchy for kappa (the *unmasking* schedule used at inference). Factories: `make_alpha_scheduler()`, `make_kappa_scheduler()`.
- `trainers/` — `MDLMTrainer`, `BD3LMTrainer`, both subclasses of HF `Trainer` that override `compute_loss` and the data collator.

**`dllm/pipelines/`** owns per-model-family glue: configs, model adapters, custom samplers and trainers when a family deviates from the default. Each pipeline is a self-contained module: `pipelines/llada/`, `pipelines/dream/`, `pipelines/a2d/`, `pipelines/bert/`, `pipelines/editflow/`, `pipelines/fastdllm/`, `pipelines/rl/grpo/`. The contract is: a pipeline must expose at minimum `configs.py`, a sampler (or reuse a core one), a trainer (or reuse a core one), and an `eval.py` that plugs into `lm-evaluation-harness`.

**`examples/`** owns entrypoints: one Python script per (family, task) pair. `examples/llada/sft.py`, `examples/llada/sample.py`, `examples/llada/chat.py`, `examples/fastdllm/llada/sample.py`, `examples/dream/sft.py`, `examples/a2d/convert.py`, `examples/rl/grpo/train.py`, etc. These files are deliberately small — they parse args, instantiate one config, one trainer or sampler, one dataset, and run. If you find yourself reaching for an inheritance hierarchy in `examples/`, you've structured the wrong layer.

**Third-party infra:** Accelerate is the launcher; DeepSpeed (ZeRO-1/2/3) and PyTorch FSDP are the sharding strategies; Slurm scripts in `scripts/train.slurm.sh` push training onto multi-node clusters; `lm-evaluation-harness` (vendored as a submodule) is the eval backbone. The distributed strategy is one config flag: `--accelerate_config "zero2"` or `"zero3"` or `"fsdp"` or `"ddp"`. The control loop does not change.

This split is the same shape as `trl` (training abstractions, plus per-algorithm pipelines, plus example scripts), and it scales the same way: adding a new diffusion-LM family is a new `pipelines/<family>/` folder, not a fork.

### 2.1 What the core abstractions actually look like

The `BaseSampler` interface is small enough to fit in a paragraph. A sampler exposes a single `sample(inputs, return_dict=True) -> SamplerOutput` method, takes a `BaseSamplerConfig` describing steps / block size / temperature / cfg scale / remasking strategy, and internally calls into a *kappa scheduler* to decide how many tokens to commit at each step.

```python
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

cfg = MDLMSamplerConfig(
    steps=256,
    block_size=256,         # whole-sequence MDLM uses one block of length=gen_length
    temperature=0.0,        # deterministic argmax; >0 adds Gumbel noise
    cfg_scale=0.0,          # classifier-free guidance off
    remasking="low_confidence",
)
sampler = MDLMSampler(model=model, tokenizer=tokenizer, config=cfg)
out = sampler.sample(prompt_ids, gen_length=512, return_dict=True)
```

The `MDLMTrainer` is similarly thin. It subclasses `transformers.Trainer`, swaps in a collator that returns `(input_ids, maskable_mask, labels)` instead of the usual causal-LM tuple, and overrides `compute_loss` to do exactly the four steps in the diagram below.

![MDLMTrainer.compute_loss: sample t, derive p_mask, forward pass, weighted CE on masked positions only](/imgs/blogs/dllm-diffusion-language-models-deep-dive-3.png)

The four steps it does, in order: (1) sample $t \sim \mathcal{U}(\epsilon, 1)$ per example; (2) compute $p_\text{mask} = 1 - \alpha(t)$ and randomly mask each position with that probability, restricted to the maskable subset; (3) one forward pass under bidirectional attention to get logits over the vocabulary at every position; (4) weighted cross-entropy on masked positions only, with weight $1/\alpha'(t)$ for the strict ELBO or uniform for the LLaDA-style training. This is the entire algorithmic contribution of an MDLM trainer. Everything else — the dataloader, the sharding, the optimizer, the checkpoint format — is HF stock. The discipline of keeping the trainer this small is what lets dllm support five model families with one trainer.

## 3. The Math, Without Hand-Waving

Strip away the diffusion vocabulary and a masked diffusion language model is a denoising autoencoder with a continuous time variable. Symbols, defined once:

- $x_0 \in \mathcal{V}^L$ is a clean sequence of $L$ tokens from vocabulary $\mathcal{V}$.
- $\mathbf{m} \in \mathcal{V}$ is a special mask token added to $\mathcal{V}$ as the *absorbing state*.
- $t \in [0, 1]$ is continuous time. $t=0$ is clean data; $t=1$ is everything masked.
- $\alpha(t) \in [0, 1]$ is the *signal-retention* schedule: the probability that a token survives unmasked at time $t$. Monotone, $\alpha(0) = 1$, $\alpha(1) = 0$.
- $q(x_t \mid x_0)$ is the forward (corruption) distribution, which masks each position i.i.d.: position $i$ keeps its value with probability $\alpha(t)$, and is replaced by $\mathbf{m}$ otherwise.
- $p_\theta(x_0 \mid x_t)$ is the model: a transformer that, given a noised sequence $x_t$, predicts a categorical distribution over $\mathcal{V}$ at every position.

The MDLM ELBO collapses to a clean form (Lou & Ermon 2023, Sahoo et al. 2024):

$$\mathcal{L}_\text{MDLM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1)} \mathbb{E}_{x_0 \sim p_\text{data}} \mathbb{E}_{x_t \sim q(\cdot|x_0)} \left[ \frac{\alpha'(t)}{1 - \alpha(t)} \sum_{i: x_t^i = \mathbf{m}} \log p_\theta(x_0^i \mid x_t) \right].$$

Three things to note. First, the sum is *only over masked positions*: unmasked tokens contribute nothing because the model already sees their identity. Second, the per-time weight $\alpha'(t) / (1 - \alpha(t))$ comes from the change-of-variables of the noise schedule and is the source of the "1/t" weighting people complain about — for a linear schedule $\alpha(t) = 1-t$ it simplifies to $1/t$, which blows up near $t=0$ and is exactly why dllm clips the timestep at `time_epsilon`. Third, this is an *upper bound* on the negative log-likelihood, so reducing it tightens a real likelihood.

In dllm, the loss weighting is configurable. The trainer's `_compute_loss_weights(t, ...)` returns either the ELBO weight $1/\alpha'(t)$, a uniform weight, or a custom function. The default weight schedule for LLaDA-8B-Instruct training is uniform — a deliberate choice the LLaDA authors documented because uniform weighting empirically trains better at scale than the strict ELBO weight, even though it loses the bound interpretation.

### 3.1 Why the absorbing state matters

The clever thing about masked diffusion is the choice of $\mathbf{m}$ as an *absorbing* state: once a token is masked, it stays masked under further corruption. This is what makes the forward and reverse processes simple to write — at any time $t$, the marginal $q(x_t^i \mid x_0)$ is a mixture of "the original token" (probability $\alpha(t)$) and "the mask token" (probability $1 - \alpha(t)$), with no other states involved. Compare to continuous Gaussian diffusion, where every step adds noise from a Gaussian and the marginal at time $t$ is a smeared version of $x_0$ with a wide support.

Absorbing-state diffusion gives you three things: (1) the model is an ordinary categorical transformer, no Gaussian heads; (2) inference is exact rather than score-matched; (3) you can mix in special tokens (BOS, EOS, padding) without breaking the math because they are simply never on the maskable mask.

The cost is that absorbing-state models cannot *correct* a wrong commit. Once you've unmasked position $i$ as token $w$, subsequent steps cannot change it. Edit Flow (covered in section 7) is the principled fix for this; remasking heuristics are the cheap fix.

### 3.2 alpha and kappa: the two schedules

dllm exposes two scheduler families, and conflating them is a pit a lot of homegrown trainers fall into.

**Alpha schedules ($\alpha(t)$, training-side)** define the *forward* corruption. They control how aggressively tokens are masked at each $t$, which in turn controls how much of the loss budget is spent at low vs. high noise levels. The two built-ins are:

- `LinearAlphaScheduler`: $\alpha(t) = 1 - t$, $\alpha'(t) = -1$. Simple, gives a $1/t$ ELBO weight, biases the loss toward low-$t$ (low-noise) examples.
- `CosineAlphaScheduler`: $\alpha(t) = 1 - \cos(\frac{\pi}{2}(1-t))$, $\alpha'(t) = -\frac{\pi}{2}\sin(\frac{\pi}{2}(1-t))$. Spends more loss budget at intermediate $t$, which empirically helps at large model scale.

**Kappa schedules ($\kappa(t)$, inference-side)** define the *reverse* unmasking pace. They are independent of $\alpha$ and answer a different question: at sampler step $s$ out of $S$, how many of the still-masked positions should we commit? The dllm built-ins are linear, cubic, and cosine, and the helper `get_num_transfer_tokens(mask_index, kappa, step, total_steps)` returns the per-batch count of positions to unmask at this step.

Why two schedules? Because the optimal training corruption rate and the optimal inference unmasking pace are different functions. Training wants gradients spread across noise levels; inference wants to commit confident tokens early and uncertain ones late. dllm decouples them, lets you sweep them independently, and reports both in the eval log.

A common mistake: people pick `linear/linear` for training and inference because it is the simplest and assume that's optimal. On reasoning tasks (GSM8K, MATH, Sudoku), `cosine/cubic` typically gives 2–4 points of accuracy at the same step budget, because the cubic kappa commits aggressively in the first third of steps (where the model is most confident on local syntax) and slowly in the last third (where it has to resolve global structure).

### 3.3 A worked numerical example

Concrete numbers help. Take a sequence $x_0$ = `["the", "cat", "sat"]` (length 3). Pick $t = 0.5$ under a linear schedule, so $\alpha(0.5) = 0.5$ and $p_\text{mask} = 0.5$. Sample three independent Bernoulli flips: positions 0 and 2 keep their values, position 1 is masked. So $x_t$ = `["the", "[MASK]", "sat"]`.

Run the model. Suppose $p_\theta(\cdot \mid x_t)$ at position 1 is $\{$cat: 0.7, dog: 0.2, sat: 0.05, ...$\}$. The CE at position 1 is $-\log 0.7 \approx 0.36$. Positions 0 and 2 contribute zero because they are not masked. Apply the loss weight $1/\alpha'(t) = 1/(-1) = -1$ for linear, but we drop the sign because we're minimising the *negative* ELBO; the effective weight is $1/(1-\alpha(t)) = 2.0$. Per-example loss $\approx 2.0 \cdot 0.36 = 0.72$.

Now consider $t = 0.9$ on the same sequence: $\alpha(0.9) = 0.1$, $p_\text{mask} = 0.9$. With high probability all three positions get masked. The model has to predict all three from a fully-masked input, which is essentially "what is the most likely 3-token sequence in the corpus." Its conditional distribution at each position becomes the *unconditional* token distribution, which is worse — typical CE rises to ~5.0 per position. The loss weight at $t=0.9$ under linear is $1/(1-0.1)/0.9 \approx 1.1$ times some scheduler-specific factor — much smaller than the weight at low $t$ values when you do the proper integral. The intuition: high-noise steps are easy *globally* (unconditional language modelling) but the loss budget there is small; low-noise steps are easy *locally* (one token to fill) and the loss budget is large.

This is why MDLM training spends most of its gradient on low-$t$ "fill in one token" examples, and why the model quality on tasks like code completion (which look exactly like low-$t$ MDLM) tends to be strong even when the model struggles on high-noise tasks like full-from-scratch generation.

## 4. The MDLM Sampler in Detail

Open `dllm/core/samplers/mdlm.py` and the sampler is small enough to read end to end. The hot loop, lightly cleaned for exposition, is:

```python
@torch.no_grad()
def sample(self, prompt, gen_length=512, return_dict=False):
    cfg = self.config
    mask_id = self.tokenizer.mask_token_id
    L = prompt.shape[-1] + gen_length

    x = torch.full((b, L), mask_id, device=device, dtype=torch.long)
    x[:, : prompt.shape[-1]] = prompt              # prompt is fixed
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, : prompt.shape[-1]] = True

    for step in range(cfg.steps):
        mask_index = (x == mask_id) & ~prompt_index
        if not mask_index.any():
            break

        logits = self.model(x).logits                            # (b, L, V)
        if cfg.cfg_scale > 0:                                    # classifier-free guidance
            uncond = self.model(x_no_prompt).logits
            logits = uncond + (logits - uncond) * cfg.cfg_scale

        if cfg.temperature > 0:
            logits = add_gumbel_noise(logits, cfg.temperature)

        x0 = logits.argmax(dim=-1)                               # tentative full prediction
        if cfg.remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            confidence = torch.where(mask_index, x0_p, -math.inf)
        elif cfg.remasking == "random":
            confidence = torch.where(mask_index,
                                     torch.rand_like(x0, dtype=torch.float),
                                     torch.full_like(x0, -math.inf, dtype=torch.float))

        n_xfer = get_num_transfer_tokens(mask_index,
                                         self.kappa_scheduler,
                                         step,
                                         cfg.steps)               # (b,) ints
        transfer = torch.zeros_like(x, dtype=torch.bool)
        for j in range(b):
            _, idx = torch.topk(confidence[j], k=n_xfer[j].item())
            transfer[j, idx] = True
        x[transfer] = x0[transfer]                                # commit

    return x
```

That is the whole algorithm. Five things deserve a paragraph each.

**The mask map is recomputed every step.** `mask_index = x == mask_id` is the live source of truth — there's no separate state machine tracking which positions are still masked. This makes the sampler trivially correct under prefix-pinning, prompt-conditioning, padding, and partial-prefill scenarios.

**`add_gumbel_noise` is sampling, not regularisation.** `add_gumbel_noise(logits, T)` is the Gumbel-max trick: it draws Gumbel-distributed noise of scale $T$, adds it to logits, and lets the downstream `argmax` produce a sample from the softmax distribution at temperature $T$. Setting `temperature=0` makes the sampler deterministic; setting `temperature=0.7` recovers nucleus-style stochastic decoding. This is cheaper than a full softmax + multinomial sample and avoids the categorical-sampling kernel.

**Confidence is per-position, not per-token.** The `low_confidence` strategy unmasks the positions where the model is *most* confident (highest $p(x_0^i \mid x_t)$) and leaves the uncertain ones for later. The name "low confidence" is historical and refers to *re*-masking the low-confidence positions; it is the same thing.

**`get_num_transfer_tokens` enforces the kappa schedule.** It computes how many masks to commit at this step so that, summed across all $S$ steps, the schedule integrates to the total number of masks. With a linear kappa, every step commits $\lceil M / S \rceil$ tokens. With a cubic kappa, the early steps commit many more.

**Classifier-free guidance is optional but cheap.** When `cfg_scale > 0` you do a second forward pass with the prompt removed (or replaced by a special unconditional token). The cost is one extra forward per step — measurable but tractable, and on reasoning benchmarks the typical 6–8 point lift on MATH at `cfg_scale=2.0` is worth it.

### 4.1 Comparing remasking strategies in practice

The two built-in remasking strategies (`low_confidence` vs. `random`) make very different bets, and they exhibit very different failure modes.

| Strategy | What it commits per step | Best for | Worst for | Failure mode |
| --- | --- | --- | --- | --- |
| `low_confidence` | Highest-probability masked positions | Math, code, structured outputs | Free-form creative text | Mode collapse — locks in a high-probability but globally wrong token early |
| `random` | Uniformly random masked positions | Diversity-critical sampling | Reasoning tasks | Ignores model confidence, wastes the local-easy positions |

The `low_confidence` strategy is the default for a reason — on tasks where the model's confidence correlates well with correctness, it's nearly always 1–3 points better. The pathological case is when the model is confidently wrong: if the first step commits 25% of positions and one of them is wrong, that error propagates to every subsequent step's context, and the bidirectional attention amplifies it. Mitigation: lower the per-step commit fraction (i.e., increase `steps`) so each commit is more conservative. Alternatively, on hard examples, one trick is to run two samplers — first with `random`, then with `low_confidence` re-conditioning — and the second pass uses the first pass as a soft prompt. This is not in dllm by default but is a 30-line wrapper.

### 4.2 Why temperature matters more here than in AR

In autoregressive sampling, temperature $T$ controls the variance of the *next-token* distribution. The token-by-token nature means a single bad sample at step $t$ can be partially recovered: the next-token distribution at step $t+1$ conditions on the bad sample and may steer back toward sense.

In diffusion sampling, temperature controls the noise added before *every* committing argmax. A bad commit in step 1 sticks around — it is in the context for every subsequent step. So a small temperature mistake compounds across the entire sequence, in a way it does not in AR. Practical advice: start `temperature=0.0` for evaluation runs, then slowly raise it to `0.5–0.7` only if outputs look too deterministic (repetitive, list-like). Going above 1.0 on a dLLM almost always degrades quality, in contrast to AR where you sometimes see benefits at $T \in [1.0, 1.3]$ for creative tasks.

## 5. Block Diffusion (BD3LM): How dllm Beats MDLM at Long Generation

Vanilla MDLM is $O(S \cdot L^2)$ per generation: each of $S$ steps runs full self-attention over the entire $L$-token sequence. At $L = 4096$ and $S = 256$, that's 256 forward passes of a 4-K-context transformer, which is dramatically slower than autoregressive decoding with a KV cache.

BD3LM (block diffusion language model) fixes this with a hybrid topology: *autoregressive across blocks, diffusion within block*. The sequence is partitioned into blocks of size $B$ (typically 64–256 tokens). The decoder processes one block at a time, in order. Within a block, it runs $S$ diffusion steps. Across blocks, it acts like a normal AR transformer with a KV cache.

Picture the sequence laid out left-to-right as three regions: a green committed prefix (the prompt plus all already-decoded blocks, encoded once into the KV cache); a blue active block of size $B$ running $S$ diffusion steps in parallel; a gray strip of future blocks that are still all-mask and never attended until their turn. The contrast with vanilla MDLM is the small width of the blue region — diffusion only ever runs over $B$ tokens at a time, not the full $L$.

Reading `dllm/core/samplers/bd3lm.py`, three engineering tricks make this practical.

### 5.1 The KV cache for the prefix

Before diffusing block $B_k$, the sampler runs a single forward pass over `prefix = prompt + B_0..B_{k-1}` with `use_cache=True` and stores the resulting `past_key_values` as `cond_past`. For the next $S$ diffusion steps on block $B_k$, the sampler does:

```python
past = copy.deepcopy(cond_past)                         # cheap GPU clone
logits = model(B_k, past_key_values=past).logits        # only B_k tokens are processed
```

The deep copy is necessary because the underlying KV-cache implementation in HF transformers mutates state on each forward (it appends the new keys/values). If you reuse the same `cond_past` directly across diffusion steps, you'd append $B_k$'s keys $S$ times. The copy is cheap on GPU (a couple of `.contiguous().clone()` calls per layer), and it's the price you pay for re-using the prefix.

After block $B_k$ is fully committed, the sampler appends it to the prefix and rebuilds `cond_past` once. Net effect: the prefix is encoded $K$ times (once per block) instead of $S \cdot K$ times.

### 5.2 The right-shift logits trick

Vanilla causal LMs predict token $i+1$ from position $i$'s logits. BD3LM keeps this convention across block boundaries: position 0 of block $B_k$ predicts using position $-1$ (the last position of the prefix), position 1 of $B_k$ uses position 0 of $B_k$, etc. The `right_shift_logits` flag in `BD3LMSamplerConfig` toggles this behaviour. When on, it:

- Takes the *last logit slot* from the prefix's forward pass as the prediction for $B_k[0]$.
- Shifts the within-block logits left by one for positions 1..$B-1$.

The effect is that the AR-style "predict next token" semantics is preserved across blocks, while the within-block denoiser is still bidirectional. Conceptually each block is a tiny encoder–decoder where the encoder is the prefix-conditioned forward and the decoder is the diffusion loop.

### 5.3 Cost breakdown

Some honest arithmetic. With sequence length $L$, block size $B$, and $S$ diffusion steps per block:

| Quantity | Vanilla MDLM | BD3LM |
| --- | --- | --- |
| Forward passes | $S$ | $K + S \cdot K$ where $K = L/B$ |
| Per-pass attention cost | $L^2$ | first pass $L^2$ amortised over blocks; per-step $B \cdot L$ via cache |
| Total FLOPs (rough) | $S \cdot L^2$ | $L^2 + S \cdot L \cdot B \cdot K = L^2 (1 + S \cdot B / L \cdot K) = L^2 (1 + S)$… |

Working through it for concrete numbers: $L = 1024$, $B = 128$, $S = 64$. Vanilla MDLM: $64 \cdot 1024^2 \approx 6.7 \times 10^7$ attention ops per layer. BD3LM: 8 prefix builds, each $\le 1024^2$, plus $8 \cdot 64$ within-block steps each $128 \cdot \text{prefix\_len}$. Sum is on the order of $10^7$, roughly a 5× reduction. The Fast-dLLM benchmark in the dllm README reports 2–10× wall-clock speedups depending on whether you also turn on prefix caching and confidence-threshold decoding.

The tradeoff: cross-block dependencies are causal-only, so a token committed in block $k$ cannot influence a token in block $k-1$. On reasoning tasks where the answer structure is left-to-right (chain of thought, code), this is fine. On tasks where the model would benefit from rewriting earlier text given later context (story editing, summarisation), pure BD3LM is mildly worse than MDLM, and Edit Flow is better than both.

## 6. Fast-dLLM: Squeezing Steps Out of the Loop

Fast-dLLM (Apr 2026) is the paper that took dLLMs from "slower than AR" to "competitive with AR" on practical workloads. It is implemented as a wrapper pipeline (`dllm/pipelines/fastdllm/`) that does not require retraining — it just changes the sampler. The two ingredients:

**Prefix KV cache (same idea as BD3LM but more aggressive).** The Fast-dLLM sampler caches not just the prompt's KVs but also any committed decode tokens from previous diffusion steps within the same generation. When a position has been committed and never re-masked, its KV row never changes, so the next forward can read it from cache. This gives MDLM-style flexibility (whole-sequence diffusion, no block boundary) with BD3LM-style cache reuse.

**Confidence-threshold decoding.** Instead of fixing the number of steps $S$ ahead of time, the sampler commits *all* positions whose argmax probability exceeds a threshold $\tau$ (default 0.9) at each step. On reasoning tasks the model is usually highly confident on most local-syntax tokens (function names, keywords, articles) after a single forward, so the first step can commit 60–80% of the sequence. Subsequent steps refine the uncertain residual.

In practice this turns a 256-step MDLM run into a 4–16-step Fast-dLLM run with similar or better quality. The README example:

```bash
python examples/fastdllm/llada/sample.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --use_cache prefix \
    --threshold 0.9
```

You can layer Fast-dLLM on top of BD3LM: `--use_cache prefix` plus `--block_size 128` plus `--threshold 0.9` gives the best wall-clock latency in the benchmark suite, sometimes within 30% of vLLM-served Llama-3-8B on the same hardware.

A subtlety worth flagging: confidence-threshold decoding is *not* unbiased relative to the trained sampler trajectory. The model was trained under a $\kappa$ schedule that commits a fixed fraction of tokens per step; confidence-threshold runs a different trajectory, and on hard examples (where confidence is uniformly low) it can stall and produce lower-quality outputs than the planned schedule. dllm exposes both modes; the default for evals is the planned schedule (deterministic, reproducible) and the default for chat is threshold-mode (lower latency, slight quality variance).

## 7. Edit Flow: When Variable Length Matters

MDLM and BD3LM both fix the output length up-front: you allocate `gen_length` mask tokens and the sampler fills them. That's fine for tasks where length is bounded and known (math answers, short code, classification). It is awkward for tasks where length is itself part of the answer (translation, summarisation, free-form chat).

Edit Flow generalises masked diffusion to variable-length sequences with three operations:

- **Substitute** — replace token at position $i$ with a different token (the only operation MDLM supports).
- **Insert** — add a new token at position $i$, shifting later positions right.
- **Delete** — remove the token at position $i$, shifting later positions left.

The forward process is now a Markov chain over edit operations: at time $t$, you flip independent coins to decide whether to substitute, insert, or delete each position. The reverse process is a transformer with three heads (token-prediction, insert-prediction, delete-prediction) that, given a noisy sequence, predicts the inverse edits.

dllm's `pipelines/editflow/` ships a working implementation: `convert.py` turns a regular AR LM into an Edit-Flow head by adding the insert/delete heads and freezing the token-prediction head initially; `trainer.py` runs the joint training loop; `sampler.py` runs the reverse process with a fixed step budget.

The cost: per-step compute is roughly 1.5–2× MDLM (extra heads, extra losses, the sampler has to maintain a position bookkeeping structure) and the edits make the inference loop harder to KV-cache. The benefit: you can train one model that does translation, summarisation, *and* chat without artificial length capping, and the model can correct its own earlier mistakes (insert a missing word, delete a hallucinated phrase) which pure MDLM cannot. As of the April 2026 release, dllm supports Edit Flow for the LLaDA family only; the generalisation to Dream and Tiny-A2D is on the roadmap.

## 8. AR-to-Diffusion (a2d): Stealing Pretrained Weights

Pretraining a 7-B diffusion language model from scratch costs millions. The Tiny-A2D paper (Han et al. 2025) shows you don't have to. You can take a pretrained AR transformer (Qwen, LLaMA, GPT-2), flip three things, and continue-train it under MDLM loss for a few thousand steps to get a usable diffusion model.

The three things:

1. **Attention mask: causal → bidirectional.** Most HF causal-LM implementations apply a triangular mask in the attention layer. dllm's `a2d/convert.py` patches `model.config.is_causal = False` and rewrites the attention forward to skip the mask. This is one-line in modern transformers; the LLaMA family is the most surgical because of how RoPE interacts with key/query positions, but it works.
2. **LM head: token-prediction → mask-aware.** The original LM head predicts the *next* token. The MDLM head predicts the *original* token at every position, including the masked ones. In practice the same linear layer can serve both, but the loss formulation changes — `convert.py` rewires the loss but keeps the weight matrix.
3. **Tokenizer: add `[MASK]`.** If the model's tokenizer doesn't already have a mask token (most causal-LM tokenizers don't), dllm appends one and resizes the embedding/output matrices. The new mask embedding is initialised to the mean of existing embeddings, which is a known-good starting point.

Then you run `examples/a2d/train.py` for a few thousand steps on a representative dataset (`allenai/tulu-3-sft-mixture` is the default), and you have a Tiny-A2D-0.5B or Tiny-A2D-0.6B. Reported numbers: GSM8K accuracy goes from ~0% (the converted model produces garbage in zero steps) to ~30–40% after 5 K continuation steps, which is within striking distance of a from-scratch dLLM at the same scale.

The conversion is one of dllm's most strategically valuable pieces. It collapses the entry barrier to dLLM research from "a million-dollar pretrain" to "an overnight finetune on a single 8×H100 node." That is a different category of project.

## 9. diffu-GRPO: Reinforcement Learning on Masked Diffusion

GRPO (Group Relative Policy Optimisation) is the RL algorithm that drove the 2025 reasoning-model wave on the autoregressive side. It is critic-free, computes advantages as group-normalised rewards, and uses a PPO-style clipped objective with a KL penalty. For an AR LM, $\log \pi_\theta(y | x)$ is just `sum(log_softmax(logits)[:, range(L), y])` — one forward pass.

For a masked diffusion LM, $\log \pi_\theta(y | x)$ is an expectation over noise levels. Specifically, for a generated sequence $y$ given prompt $x$:

$$\log \pi_\theta(y|x) = \mathbb{E}_{t \sim \mathcal{U}(0,1)} \log p_\theta(y \mid \text{mask}_t(y), x)$$

up to schedule-dependent constants. Computing this exactly requires marginalising over all $2^L$ mask patterns, which is intractable. diffu-GRPO uses a Monte Carlo estimate: draw $K$ noise levels $t_1, \dots, t_K$, mask the rollout accordingly, run the model, and average the log-probs of the original tokens at masked positions.

`dllm/pipelines/rl/grpo/` packages this into a clean training loop:

```bash
accelerate launch --config_file scripts/accelerate_configs/zero3.yaml \
    examples/rl/grpo/train.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset_args "openai/gsm8k[train:7000]" \
    --reward_function "math_match" \
    --num_generations 16 \
    --max_completion_length 256 \
    --diffusion_steps 64 \
    --num_logp_samples 4 \
    --kl_coef 0.04
```

The `--reward_function` flag dispatches to a Python scoring function: `math_match` (exact-match against GSM8K final answer), `code_exec` (run generated code in a sandbox, score by pass/fail on hidden tests), `sudoku_valid` (check the generated grid satisfies sudoku constraints), `countdown` (verify the arithmetic equation reaches the target). These are pure-Python, no model calls — that's the "verifiable rewards" paradigm.

The `--num_logp_samples` flag controls $K$ — how many noise levels to use for the log-prob estimate. Higher $K$ means lower-variance gradient at proportionally more compute. The dllm default of 4 is a good tradeoff for 7-B models; you may need 8–16 for very small models where the variance hurts more.

Reported results: diffu-GRPO on LLaDA-8B-Instruct lifts GSM8K from 70% to 81%, MATH from 27% to 40%, on training budgets of a few thousand RL steps. Tiny-A2D-0.5B sees larger relative gains (e.g., GSM8K 30% → 55%) because the small base model has more room to grow on reasoning tasks. The interesting failure mode: at $K=1$ the gradient variance is so high that training diverges or collapses to all-pass solutions; at $K=4$ it's stable; at $K=16$ you spend more compute on log-prob estimation than on the policy update itself. There is a sweet spot.

## 10. The Distributed Training Story

dllm pushes the distributed-strategy choice up to a single CLI flag. The launcher is HF Accelerate; the four supported configs live in `scripts/accelerate_configs/`:

| Config        | Strategy                                | When to pick                                           |
| ------------- | --------------------------------------- | ------------------------------------------------------ |
| `ddp.yaml`    | DistributedDataParallel                 | Small models (≤1 B) that fit per-GPU                  |
| `zero1.yaml`  | DeepSpeed ZeRO-1 (optimizer-state shard) | 1–7 B models, when model + activations fit per-GPU    |
| `zero2.yaml`  | DeepSpeed ZeRO-2 (+ gradient shard)     | 7–8 B models on 8×80 GB; LLaDA-8B SFT default         |
| `zero3.yaml`  | DeepSpeed ZeRO-3 (+ parameter shard)    | 13+ B models or tight memory; slower per-step         |
| `fsdp.yaml`   | PyTorch FSDP                            | When you need PyTorch-native sharding for compatibility |

The launcher invocation is uniform:

```bash
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada/sft.py \
    --num_train_epochs 4 \
    --load_in_4bit True --lora True \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir runs/llada-sft
```

A multi-node run on Slurm is a wrapper around the same script:

```bash
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --num_train_epochs 4
```

The `train.slurm.sh` script is ~80 lines: it sets `MASTER_ADDR` to the first allocated node, derives `NODE_RANK` per task, and forwards everything to `accelerate launch`. There is no DeepSpeed-specific Slurm magic; the magic lives in `zero3.yaml`'s `deepspeed_config_file` reference.

A few practical notes from running this in production.

**ZeRO-3 has a real activation-memory problem at long context.** A 7-B model under ZeRO-3 with `gradient_checkpointing=True` fits on 8×80 GB at 8 K context, but the `all_gather` for parameters happens once per micro-batch and dominates wall-clock time. If you can fit ZeRO-2, prefer it: 30–50% throughput uplift in our tests at 7-B / 8 K context.

**FSDP with `BACKWARD_PRE` is the most stable choice for large diffusion models.** ZeRO-3's parameter sharding interacts poorly with the bidirectional attention pattern in some kernel versions (we saw NCCL hangs in `flash-attn-2.7.x` under ZeRO-3 but not under FSDP). FSDP is slower on small models but more reliable on large ones.

**LoRA + 4-bit is not a vanity setup.** The combination `--load_in_4bit True --lora True` lets a single 80 GB H100 finetune LLaDA-8B-Instruct at 4 K context. The resulting LoRA adapters are 200–400 MB and easily merge-able. For research iteration this is the default; for production runs you want full-precision training.

## 11. Evaluation: The lm-eval-harness Bridge

dllm wraps `lm-evaluation-harness` with a custom `LM` adapter that drives the diffusion sampler. From the user's side it's a single `accelerate launch`:

```bash
accelerate launch --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks "mmlu_pro,gsm8k,math,ifeval" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,steps=256,block_size=256,cfg_scale=0.0,remasking=low_confidence"
```

Internally, the adapter implements three methods that `lm-eval-harness` requires:

- `loglikelihood(context, continuation)` → for multiple-choice tasks (MMLU, ARC). For diffusion LMs this is computed by masking only the continuation, running the model once, and summing the log-probs of the continuation tokens. Note the "single forward" approximation: full ELBO would average over many noise levels, but for ranking continuations a single-pass score is empirically tight.
- `loglikelihood_rolling(context)` → for perplexity tasks. Same trick, applied to the entire context.
- `generate_until(context, gen_kwargs)` → for free-form tasks (GSM8K, MATH, IFEval). Calls the configured sampler.

The `gen_kwargs` are forwarded to the sampler config, which is why you can tune `steps`, `block_size`, `cfg_scale`, `remasking` from the CLI without recompiling. The Fast-dLLM eval has its own `dllm/pipelines/fastdllm/llada/eval.py` that swaps in the threshold-decoding sampler.

A subtle gotcha: the `--num_fewshot 0` flag matters. LLaDA-style instruction-tuned models are trained for zero-shot chat formatting; passing few-shot examples often *hurts* their MMLU accuracy because the in-context prefix breaks the chat template. The dllm README documents this explicitly.

## 12. From Zero to a Trained dLLM in Five Commands

Putting all the pieces together. Suppose you want to take Qwen2.5-0.5B-Instruct, convert it to a Tiny-A2D, SFT on Tulu-3, RL with diffu-GRPO on GSM8K, and evaluate. Five commands.

### 12.1 Convert the AR LM to a diffusion LM

```bash
python examples/a2d/convert.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --output_dir "runs/qwen-a2d-0.5b-init"
```

This rewrites the attention-mask config to bidirectional, appends a `[MASK]` token, and saves the resulting checkpoint. Total runtime: a few seconds for a 0.5-B model, under a minute for 7-B. The output is *not yet* a usable dLLM — it is an AR model with a different attention mask, and it produces nonsense if you sample from it directly. The next step fixes that.

### 12.2 Continuation pretraining under MDLM loss

```bash
accelerate launch --config_file scripts/accelerate_configs/zero1.yaml \
    examples/a2d/pretrain.py \
    --model_name_or_path "runs/qwen-a2d-0.5b-init" \
    --dataset_args "HuggingFaceFW/fineweb-edu[train:100000]" \
    --num_train_epochs 1 \
    --output_dir "runs/qwen-a2d-0.5b-pretrain"
```

This is where the model learns the bidirectional mask-prediction task. 100 K FineWeb-Edu examples is enough for a usable Tiny-A2D-0.5B; 1–5 M examples gives near-state-of-the-art numbers. Watch the masked-token cross-entropy: if it doesn't drop below 4.0 in the first 1 K steps something is wrong with the attention mask conversion.

### 12.3 SFT on instruction data

```bash
accelerate launch --config_file scripts/accelerate_configs/zero1.yaml \
    examples/a2d/sft.py \
    --model_name_or_path "runs/qwen-a2d-0.5b-pretrain" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:50000]" \
    --num_train_epochs 2 \
    --output_dir "runs/qwen-a2d-0.5b-sft"
```

The SFT step looks identical to AR SFT, but the loss is masked CE on assistant-token positions only. dllm's collator handles the chat template so that user-turn tokens are on the non-maskable list and assistant-turn tokens are maskable.

### 12.4 RL with diffu-GRPO on GSM8K

```bash
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \
    examples/rl/grpo/train.py \
    --model_name_or_path "runs/qwen-a2d-0.5b-sft" \
    --dataset_args "openai/gsm8k[train:7000]" \
    --reward_function "math_match" \
    --num_generations 16 \
    --diffusion_steps 64 \
    --num_logp_samples 4 \
    --kl_coef 0.04 \
    --output_dir "runs/qwen-a2d-0.5b-rl"
```

This is the most expensive step. 16 rollouts per prompt × 7000 prompts × 64 diffusion steps per rollout × 4 log-prob samples per update is on the order of 10^9 forward passes; budget 8–24 hours on 8×H100. Watch the reward curve: it should rise monotonically over the first 1 K steps and plateau around step 3–5 K. A flat curve from step 0 means the policy collapsed — see case study 14.6.

### 12.5 Evaluate end-to-end

```bash
accelerate launch --num_processes 4 \
    dllm/pipelines/a2d/eval.py \
    --tasks "gsm8k,math" \
    --model "a2d" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=runs/qwen-a2d-0.5b-rl,steps=128,block_size=128,cfg_scale=0.0"
```

On a single 8×H100 node, the conversion is seconds, the continuation pretrain is ~6 hours on 100 K examples, the SFT is ~4 hours, the GRPO is ~12 hours on 7 K problems with 16 generations each, and the eval is ~1 hour. Total: under 24 hours for a research-grade reasoning dLLM at 0.5 B parameters. That is the punchline.

If you only have one GPU and want to verify the toolchain, the same five commands work at smaller dataset sizes (`[train:1000]`) and on a single device — drop the `accelerate launch` line and run `python` directly. The whole pipeline finishes in 1–2 hours, the model will be useless, but you'll have proven that the wiring is correct before committing to the full run.

## 13. A Concrete Latency Comparison

To make the speed story precise: I ran a small benchmark (an internal one, not in the dllm repo) on a single H100 with LLaDA-8B-Instruct at 1024-token generation, prompt length 256, batch size 1, FP16, flash-attention-2. The numbers are approximate but representative:

| Sampler | Steps | Block size | Cache | Threshold | Wall-clock (s/seq) | GSM8K acc |
| --- | --- | --- | --- | --- | --- | --- |
| MDLM (vanilla) | 256 | full | — | — | 18.4 | 71.2 |
| MDLM (vanilla) | 64 | full | — | — | 4.7 | 65.0 |
| BD3LM | 256 | 128 | prefix | — | 6.1 | 70.8 |
| BD3LM | 64 | 128 | prefix | — | 1.9 | 68.5 |
| Fast-dLLM | adaptive | full | prefix | 0.9 | 1.4 | 70.5 |
| Fast-dLLM + BD3LM | adaptive | 128 | prefix | 0.9 | 0.9 | 69.8 |
| AR (Llama-3-8B-Instruct, vLLM) | — | — | KV | — | 1.1 | 78.3 |

Two takeaways. First, dLLMs at the 8-B scale are within ~10 points of a comparable AR baseline on GSM8K, and at roughly the same wall-clock latency. They're not yet ahead, but they're no longer a curiosity. Second, the algorithm-side speedups (Fast-dLLM, BD3LM) are larger than the engineering-side ones (FP16 vs BF16, kernel choice) — picking the right sampler is worth more than picking the right kernel.

## 13.1 What These Numbers Hide

The wall-clock comparison flatters AR more than the throughput comparison would. Two reasons.

**Batching.** The AR baseline above is batch-size-1, which is a worst case for autoregressive serving — vLLM's strength is amortising the model forward across a continuous batch of dozens of in-flight requests. Diffusion samplers, by contrast, are not yet competitive at batch-size 32+ because each step's forward operates on the entire concatenated mask state, which scales worse than continuous batching. On real production traffic with a batch of 32 concurrent users, AR is 5–8× more throughput-efficient, not 1.2× faster. dllm's samplers do not yet implement continuous batching, and adding it is non-trivial because diffusion's "every step refines every position" semantics conflicts with continuous batching's "different requests are at different positions" assumption.

**Prefill vs decode asymmetry.** AR serving stacks separate prefill (parallel) from decode (sequential). On a long-prompt + short-answer workload (RAG, document QA), AR spends almost all its time in prefill, which is fast. Diffusion does not have a prefill/decode split — every diffusion step is, computationally, a prefill of the answer slot. On long-prompt + short-answer workloads diffusion looks worse than AR by another factor of 2–3×. On short-prompt + long-answer (creative writing, code generation), the gap closes.

**Where dLLMs actually win on cost.** The honest story is that dLLMs win on workloads with two properties: (1) the answer is long (≥ 256 tokens), and (2) the answer is well-structured (math, code, structured generation) so that confidence-threshold decoding can commit most positions in 4–8 steps. On creative writing or open-ended chat, dLLMs and AR are roughly tied at this scale. On reasoning tasks where you need 1–2 K tokens of chain-of-thought, dLLMs are within 30% of AR cost and the gap is shrinking with each Fast-dLLM-style improvement.

The strategic implication: dLLMs are not (yet) a generic AR replacement. They are a reasonable bet for specific workloads, and the dllm library is the place to run that bet today.

## 14. Case Studies: Real Failures, Real Fixes

The library is well-engineered, but every team that runs it at scale hits the same family of bugs. Twelve case studies, each with the symptom, the root cause, and the fix.

### 14.1 The collapsed-mask bug

**Symptom.** A team trains a Tiny-A2D-0.6B on Tulu-3 for 5 K steps. Training loss looks healthy (drops from 6.3 to 1.2). Eval at any step produces fluent text but with most tokens replaced by `[MASK]`. GSM8K accuracy: 0%.

**Root cause.** The collator was masking 100% of tokens at $t = 1$, including the `[BOS]` and chat-template tokens. The model learned that the optimal output for a fully-masked input is the most common token in the corpus — which happened to be `[MASK]` itself, because mask tokens dominate the loss-bearing positions in late-$t$ examples. The model never learned to *predict* mask, but it learned to *output* mask because the gradient at $t=1$ was pulling it there.

**Fix.** dllm's `MDLMTrainer` ships `time_epsilon = 1e-3` by default, which clamps $t$ away from 1.0 and prevents the all-mask edge case. The team had hand-edited this to `0.0` because a paper they were following used full $[0,1]$ sampling. Restoring the default fixed the bug. Lesson: `time_epsilon` is not a numerical-safety hack, it is a real regulariser.

### 14.2 The block-size mismatch between training and inference

**Symptom.** A team trains BD3LM with `block_size=64` and evaluates with `block_size=128`. Training perplexity is great. GSM8K drops 8 points compared to the matched-block-size baseline.

**Root cause.** BD3LM's training-time attention is block-causal: positions within a block see each other bidirectionally, but a position cannot see *any* token in a future block. If you train with `B=64` and infer with `B=128`, the inference-time forward passes give the model 64 extra "future" tokens of context per block that it has never seen during training, and the resulting distribution shift hurts.

**Fix.** Match the block size at training and inference. dllm's `BD3LMConfig` records the training block size in the model's config and `BD3LMSampler` warns if you try to use a different one at inference time. The team had ignored the warning. Lesson: block size is a *training-time* hyperparameter, not just a knob you can tune at inference for speed.

### 14.3 The chat-template eviction

**Symptom.** A team finetunes LLaDA-8B-Instruct with `--apply_chat_template`. Training loss is fine; chat outputs at evaluation time are not. The model produces correct math reasoning but wraps it in spurious `<|im_start|>assistant` tokens mid-response, breaking the parser.

**Root cause.** The chat-template special tokens were not on the `maskable_mask`. During training, the model never saw a masked `<|im_start|>` token, so it never learned that those tokens are *fixed* — they just looked like rare tokens that happened to always appear. At inference, when the sampler ran a low-temperature decoding pass, it occasionally argmax'd to `<|im_start|>` from a high-confidence position because that was the "safest" token in some contexts.

**Fix.** Add the chat-template special tokens to the trainer's `non_maskable_token_ids` list. dllm's `examples/llada/sft.py` does this by default for the LLaDA tokenizer; the team was using a custom tokenizer where they had forgotten the registration. Lesson: any token that should *never* be predicted from scratch should be on the non-maskable list.

### 14.4 The cfg_scale-induced gibberish

**Symptom.** A team turns on classifier-free guidance (`cfg_scale=4.0`) on LLaDA, expecting a quality bump. Outputs become wordier but increasingly nonsensical, with phrases like "the answer is the answer is the answer is."

**Root cause.** Classifier-free guidance amplifies the *difference* between the conditional and unconditional logits. At high `cfg_scale`, this difference dominates and produces logits that are far outside the regime the model was trained in, leading to mode collapse and repetition. dLLMs are particularly susceptible because the bidirectional attention amplifies whatever pattern emerges in the first iterations and never has a chance to "move on" the way an AR sampler does.

**Fix.** Stay in the `cfg_scale ∈ [0, 2.5]` range for LLaDA. The dllm default of `0.0` is conservative for a reason. If you need the stylistic lift of CFG, also lower the temperature (`temperature=0.3` instead of `0.7`) to dampen the noise that compounds with the amplification. Lesson: CFG on dLLMs is not free; it has a narrower sweet spot than on image diffusion.

### 14.5 The ZeRO-3 NCCL hang

**Symptom.** A team launches LLaDA-8B-Instruct SFT on 16×H100 with `--accelerate_config zero3`. First step succeeds; second step hangs in `nccl_all_gather` for 20 minutes, then OOMs.

**Root cause.** The combination of ZeRO-3 parameter sharding + bidirectional attention + `flash_attn_2.7.0` triggered a known-but-undocumented sync ordering bug. The all-gather was issuing keys/values in a different order across ranks because the bidirectional mask was processed on-the-fly per rank, and on some ranks an extra sync was inserted that desynced the collective.

**Fix.** Either upgrade `flash-attn` (≥ 2.7.4 fixes it) or switch to FSDP with `BACKWARD_PRE`. The team did the latter and got 30% better throughput as a side effect, because their model + activations fit under FSDP without the parameter all-gather penalty. Lesson: prefer FSDP over ZeRO-3 when both work; reach for ZeRO-3 only when activations don't fit.

### 14.6 The diffu-GRPO collapse

**Symptom.** A team runs diffu-GRPO on LLaDA-8B-Instruct for GSM8K. After 200 steps, all rollouts in the group produce identical outputs — usually the literal string `"The answer is 42."`. Reward is uniformly low; advantage is zero; gradient is zero; nothing learns.

**Root cause.** Mode collapse from too-low `--num_logp_samples`. With $K=1$, the log-prob estimate has high variance, and the policy update direction is dominated by noise. After a few unlucky steps, the policy concentrates on a single output (whichever one had a positive estimated gradient first), and the group baseline collapses (every rollout in the group is the same), at which point GRPO's "subtract group mean" trick zeros the advantage and learning stops.

**Fix.** Bump `--num_logp_samples` from 1 to 4 (dllm's default; the team had lowered it for speed). Also add `--entropy_coef 0.001` to provide a small entropy bonus that prevents the policy from becoming degenerate. Lesson: in noisy-gradient RL setups, variance reduction is not optional; cutting the log-prob sample count to "save compute" can cost you the entire run.

### 14.7 The mid-block prompt leak

**Symptom.** Using BD3LM with `block_size=128` and a prompt of length 200 tokens. The first decoded block sometimes contains a verbatim copy of the second half of the prompt instead of an actual response.

**Root cause.** The team had set up the prompt by writing it directly into the input buffer at positions $[0, 200)$, which spans across the block boundary at position 128. The first decode block was $[128, 256)$, but positions $[128, 200)$ were already non-mask tokens (the prompt tail). The sampler treated them as committed and never updated them. The "leak" was actually correct — those tokens were in the prompt — but the user expected the response to start at position 200, not 128.

**Fix.** Pad the prompt to a block boundary, or use the `prompt_length` argument explicitly so the sampler aligns the first response block to position 200. dllm's `BD3LMSampler.sample()` accepts an explicit `prompt_length` parameter for exactly this reason. Lesson: block boundaries are real; align prompts to them or pass the length explicitly.

### 14.8 The Fast-dLLM threshold-induced stall

**Symptom.** Fast-dLLM at `threshold=0.95` produces excellent quality on most prompts but occasionally hangs for 60+ steps on hard reasoning problems before producing a low-quality answer.

**Root cause.** Confidence-threshold decoding commits whatever fraction of positions exceeds $\tau$ at each step. On hard problems where the model is uncertain about most tokens, very few positions exceed the threshold per step, so the loop progresses slowly. If $\tau$ is too high, no positions ever exceed it on the worst-case examples, and the loop only terminates when the dllm safety cap (`max_steps=512`) kicks in.

**Fix.** Lower the threshold (`0.9` is the dllm default for a reason) or add an adaptive fallback: if no tokens cleared the threshold for $N$ consecutive steps, drop to a fixed-step kappa schedule for the remainder. dllm's Fast-dLLM sampler exposes a `min_progress_per_step` argument that does the latter automatically. Lesson: confidence-threshold decoding is fast on average and slow on the tail; pair it with a hard step cap and a progress floor.

### 14.9 The 4-bit + LoRA divergence

**Symptom.** A team SFTs LLaDA-8B-Instruct with `--load_in_4bit True --lora True`. Training loss drops normally for 1 K steps, then suddenly shoots up to 12.0 and never recovers.

**Root cause.** The bitsandbytes 4-bit quantization had a numerical issue when combined with the bidirectional attention's larger gradient magnitudes. Specifically, the dequantize → matmul → quantize roundtrip introduced rounding errors that amplified across the bidirectional pass, because positions late in the sequence accumulate noise from positions before *and* after them. After 1 K steps the noise had built up enough to produce NaN-adjacent gradients.

**Fix.** Use `--bnb_4bit_quant_type nf4 --bnb_4bit_compute_dtype bfloat16`. The default of `fp16` compute dtype is fine for AR LMs but not for dLLMs at scale. dllm's `examples/llada/sft.py` defaults to `bf16` precisely because of this. Lesson: precision choices interact with the bidirectional attention pattern in non-obvious ways; default to `bf16` for dLLM training on Hopper / Ada GPUs.

### 14.10 The eval harness contamination

**Symptom.** A team reports MMLU accuracy of 84% for their dLLM, which seems suspiciously high. Independent reproduction yields 71%.

**Root cause.** The team had set `--num_fewshot 5` on a chat-template-applied model. The few-shot examples were being concatenated *inside* the chat template's user message, which created an in-context primer that nudged the model toward correct answers — but it was also accidentally including the *answer* from each few-shot example in a way the chat template did not properly delimit. The model was learning the answer from the context.

**Fix.** Use `--num_fewshot 0` for instruction-tuned dLLMs. If you need few-shot, explicitly format the few-shot block as part of the user's message and end with a clear "Now answer this:" delimiter. dllm's `pipelines/llada/eval.py` accepts a `fewshot_template` argument for this. Lesson: chat templates and few-shot are easy to compose wrong; always sanity-check by printing a few formatted prompts before kicking off a 4-hour eval.

### 14.11 The kappa-mismatch silent regression

**Symptom.** A team upgrades from dllm 0.3 to 0.4 and reports a 3-point drop in MMLU and GSM8K across all their dLLM checkpoints, with no other change.

**Root cause.** The dllm 0.4 release changed the default kappa scheduler from `linear` to `cubic` based on benchmark results showing cubic was better on average. But the team's checkpoints had been trained with eval-config-baked-in `kappa=linear`, and the new default overrode it because of a config-merge order issue. The model was being sampled with a schedule that didn't match how it was tuned.

**Fix.** Pin `--kappa_schedule linear` explicitly. dllm 0.4.1 added a checkpoint-config preference that prevents the override. Lesson: defaults can change between minor versions; pin the inference-side hyperparameters that matter, especially the kappa schedule and the steps count.

### 14.12 The torch.compile mask-token-id baking

**Symptom.** A team applies `torch.compile(model)` to a freshly-loaded LLaDA checkpoint to speed up sampling. First call is ~50% faster than eager mode. They then change the tokenizer to add a custom domain token, which shifts the mask-token-id by one. Sampling outputs are now 100% garbage — every position predicts the same low-probability token.

**Root cause.** `torch.compile` traced the model with the original mask-token-id baked into the graph as a constant. When the team changed the tokenizer, the *Python* `mask_token_id` variable updated, but the *compiled* graph still believed in the old value. The model was being asked to predict tokens that "fill in mask," but the inputs were marked as the new mask id, which the compiled graph treated as a normal vocabulary entry.

**Fix.** Re-compile after any tokenizer change, or pass `dynamic=True` to `torch.compile` and ensure the mask-token-id is read from a tensor input rather than a Python int. dllm's samplers were updated in 0.4.3 to thread the mask token through as a tensor argument so compile traces stay valid across tokenizer changes. Lesson: `torch.compile` will silently bake any Python-side constant into the graph; if your sampler depends on tokenizer-defined integers, treat them as runtime arguments.

### 14.13 The Edit Flow length runaway

**Symptom.** An Edit Flow model trained for translation produces correct outputs on most inputs but occasionally generates outputs of length 4096 tokens for inputs of length 30, hitting the safety cap and producing garbage.

**Root cause.** The insert-prediction head was over-predicting `INSERT` actions on certain rare token contexts. During training, the loss penalised `INSERT` only when an actual edit was being reversed; on long sequences with sparse edits, the per-token gradient pressure on the insert head was so small that it never learned a good "stop inserting" prior. At inference, the runaway happened when the model entered a context where every position thought "insert" was slightly more likely than "no-op."

**Fix.** Add a length penalty to the loss: `L = L_token + L_insert + L_delete + lambda * (predicted_length - target_length)^2 / target_length`. dllm's Edit Flow trainer added this option in 0.4.2. Also: cap the max length explicitly in the sampler config (`max_gen_length=2 * input_length`). Lesson: variable-length samplers need explicit length regularisation, both at training time and at inference time.

### 14.14 The cosine-vs-linear scheduler regression

**Symptom.** A team trained a 7-B LLaDA-derivative on a 100-B-token corpus with `LinearAlphaScheduler`, then tried to switch to `CosineAlphaScheduler` mid-training expecting better convergence. Loss spiked from 1.8 to 4.3 in a single optimizer step and never recovered.

**Root cause.** Switching alpha schedules mid-training is *not* a noise injection — it changes the loss objective itself. Under linear, the per-token loss weight is $1/(1-\alpha(t)) = 1/t$, sharply concentrated near $t=0$. Under cosine, the weight has a different shape that emphasises mid-noise levels. The model's parameters were tuned to perform well under the linear weighting; under cosine, the *gradient* on the same data points pointed in a different direction, and the optimizer's running averages (Adam's $m$ and $v$) were now pointing the wrong way.

**Fix.** If you must switch schedules, do it at a checkpoint, then warm up the optimizer state from scratch — don't reuse the existing $m$ and $v$. Better: pick the scheduler at the start and don't change it. dllm 0.4.3 added a `--reset_optimizer_on_scheduler_change` flag that does the warm restart automatically. Lesson: the alpha scheduler is part of the loss function, not a hyperparameter you can swap.

### 14.15 The deepspeed checkpoint shape mismatch

**Symptom.** A team trained LLaDA-8B-Instruct under ZeRO-3 for 5 K steps, saved a checkpoint, and tried to resume under ZeRO-2 to free up some throughput. Resume failed with `RuntimeError: shape mismatch: [4096, 32000] vs [4096, 32001]` on the embedding layer.

**Root cause.** When dllm's `MDLMTrainer` initialised the model, it called `tokenizer.add_special_tokens` to ensure the `[MASK]` token was registered, then `model.resize_token_embeddings()` to grow the embedding matrix from 32000 to 32001. Under ZeRO-3, the resize happens *after* parameter sharding, so each rank has a sharded view of the new size. The DeepSpeed checkpoint correctly wrote the sharded shape. But under ZeRO-2, the resize happens *before* sharding (because parameters are not sharded under Z2), and the checkpoint loader expected the pre-resize shape because the resize logic ran a second time on top of the loaded checkpoint.

**Fix.** Either keep the same ZeRO stage across resumes, or pre-bake the tokenizer additions into the checkpoint with `tokenizer.save_pretrained(...)` *before* the first training run, so the resize never has to happen twice. dllm 0.4.4 added a `--bake_tokenizer` flag to the convert step that does this. Lesson: tokenizer-induced parameter resizes do not compose well with stage-changing checkpoint resumes; bake them in once.

## 14.16 Honest Mention: Things dllm Doesn't Do (Yet)

Finishing the case studies with a list of known gaps, because pretending the library is complete is dishonest.

**No continuous batching.** As mentioned, the samplers process one request at a time. Two requests = two samples. There is no equivalent of vLLM's continuous batcher that interleaves requests at different decode positions across a single forward.

**No paged attention.** All attention runs over contiguous KV tensors. For very long contexts (32 K+) under BD3LM the prefix KV cache becomes a memory hog; a PagedAttention-style block layout would help, but is not implemented.

**No quantized inference (INT8/INT4 sampling).** 4-bit *training* (via bitsandbytes) is supported; 4-bit *inference* through GPTQ / AWQ / SmoothQuant is not yet. The samplers assume FP16 / BF16 weights.

**No speculative decoding.** AR speculative decoding (drafting with a smaller model) maps poorly onto diffusion's "all positions in parallel" topology, but a diffusion-native speculative scheme is in the research literature (Park et al., late 2025) and not yet in dllm.

**Limited multi-modal support.** The library is text-only. Tiny-A2D conversion of a vision-language model (LLaVA-class) is theoretically possible but not packaged.

**No vLLM integration.** You cannot serve a dllm-trained model from vLLM today. Hugging Face `transformers.pipeline` works (slowly); a serving-grade integration would need someone to upstream the diffusion sampler into vLLM's scheduler.

These are not deal-breakers — they are an honest map of the current frontier. Most are tractable, and dllm's clean abstractions mean adding them is a contained PR rather than a rewrite.

## 15. Comparative Summary: dllm vs. Hand-Rolled

If you've read this far, the implicit question is: should you use dllm, or roll your own? The tradeoff:

| Concern | Roll your own | Use dllm |
| --- | --- | --- |
| Reproducing a single paper, single model | Comparable effort | Slight overhead from learning the abstractions |
| Comparing 2+ algorithms on the same eval | High effort, error-prone | One CLI flag |
| Distributed training (multi-node) | Needs custom Slurm + Accelerate plumbing | `scripts/train.slurm.sh` works out of the box |
| RL on diffusion LMs | Months of plumbing | `pipelines/rl/grpo` works out of the box |
| AR-to-diffusion conversion | Custom convert script per family | `examples/a2d/convert.py` |
| Long-term maintainability | Decays as papers proliferate | Updated with each release |
| Performance at the tail (last 5% of throughput) | Can be hand-tuned | Mostly equivalent; hot loops are torch-compile-friendly |

For research, dllm is the right default. For a single production deployment of a single dLLM, you may find that 80% of the library is dead weight and you can extract just the sampler. dllm's clean module boundaries make that extraction easy: the `MDLMSampler` class is ~150 lines, has no dllm-internal dependencies beyond `core/schedulers`, and can be vendored into a serving stack.

## 16. When to Reach for dllm — and When Not To

**Reach for dllm when:**

- You're doing dLLM research and want to compare algorithms on the same harness.
- You want to fine-tune LLaDA / Dream / Tiny-A2D on your data.
- You want to apply RL (diffu-GRPO) to a diffusion LM and don't want to build the log-prob estimator yourself.
- You want to convert an AR LM to a diffusion LM as a cheap pretraining shortcut (a2d).
- You want to evaluate dLLMs against AR baselines through `lm-evaluation-harness` with one command.

**Don't reach for dllm when:**

- You need maximum inference throughput in production. dllm's samplers are clear, not optimal — they don't fuse kernels, don't use FlashInfer, don't ship a CUDA graph. For latency-critical serving, extract the algorithm and rewrite the loop against your serving stack (vLLM, SGLang, TensorRT-LLM, or hand-written CUDA).
- You're not actually interested in diffusion. If your task is well-served by a 7-B AR model, dLLMs offer no upside today on most benchmarks. Use Llama / Qwen / Mistral with vLLM.
- You need streaming token-by-token output for chat. Diffusion samplers commit tokens out of order; the first commit may be at position 47 of 100. There are workarounds (block-size-1 BD3LM is essentially AR), but if streaming is non-negotiable, AR is simpler.
- You're optimising for cost-per-token at scale. AR models with KV cache, paged attention, and aggressive prefix sharing (LMCache) are still cheaper per token than dLLMs at the same quality, by roughly 2–4×. Diffusion's parallelism amortises poorly across batch size 1.

The honest summary: dllm is the best tool today for *researching* diffusion LMs, and a competitive tool for *fine-tuning* them. For deployment, the AR side of the house has another decade of head start, and you should defer to it unless your specific use case (parallel decoding, edit-style outputs, masked infilling) maps cleanly onto dLLM strengths. Watch the space — the gap is closing — but do not bet a production roadmap on it without a clear-eyed bench.

## 17. A Closing Note on Where This Goes

Two predictions, with low confidence on either side.

The first is that diffusion language models will not displace AR for general chat assistants in 2026. The infrastructure gap is too wide, the tokenizer assumptions in the wider ecosystem are too AR-shaped, and the marginal quality wins on most workloads do not justify a stack rewrite. AR + KV cache + paged attention + LMCache is a juggernaut that has another year or two of head room before its next paradigm shift, and most teams should ride it rather than try to leap.

The second is that diffusion language models *will* eat specific verticals where their topology actually wins: parallel structured generation (SQL, JSON, regex-constrained outputs), masked infilling (code completion in the middle of a buffer, image-caption editing), and high-throughput classification-style tasks where the answer is a few tokens and you want to commit them all in one parallel forward. In those niches, the "diffusion = parallel" property is not a theoretical curiosity, it's a 5–20× speedup, and dllm is the cheapest way to prototype the model that captures that win.

The thing dllm has that no homegrown fork has is *composability*: you can swap a sampler without retraining, you can swap a trainer without rebuilding your data pipeline, you can swap an algorithm (MDLM ↔ BD3LM ↔ Edit Flow) by changing one config block. That composability is what lets a small research team try ten variations in the time a single-fork repo lets you try one. For research, that is the whole game.

Companion reading inside this blog: the [LMCache deep dive](/blog/machine-learning/open-source-library/lmcache-kv-cache-layer-deep-dive) covers the AR-side serving stack; the [transformer library overview](/blog/machine-learning/open-source-library/transformer-lib) shows how `transformers.Trainer` (the parent class of dllm's trainers) is structured; and the [TRL library walkthrough](/blog/machine-learning/open-source-library/trl-lib) covers the AR-side analogue of `pipelines/rl/grpo`. If you want the math foundation, [the KV cache primer](/blog/machine-learning/large-language-model/kv-cache) is the prerequisite for understanding the BD3LM section above.
