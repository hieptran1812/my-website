---
title: "LoRA and QLoRA for RLHF: Efficient Alignment on a Single GPU"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Train a four-model PPO alignment loop for a 7B–13B language model on one A100 by replacing full fine-tuning with low-rank adapters and 4-bit NF4 quantization — with the memory math, the PEFT/TRL code, and the trade-offs spelled out."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "lora",
    "qlora",
    "llm-alignment",
    "machine-learning",
    "pytorch",
    "peft",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 70
image: "/imgs/blogs/lora-qlora-for-rlhf-1.png"
---

The first time I tried to run PPO-based RLHF on a 7-billion-parameter model, I did the arithmetic *after* the job had already died. I had a single 80 GB A100, a reward model I was proud of, and a confident plan. Forty seconds into model loading, the process hit an out-of-memory error and the GPU sat there with 79.4 GB allocated and nothing trained. The problem was not my code. The problem was that I had asked one GPU to hold four full copies of a 7B model at once — a trainable policy, a frozen reference policy, a reward model, and a value head — plus the Adam optimizer states for the policy, which alone are twice the size of the model. The honest total was north of 200 GB. I was off by a factor of three before I had run a single PPO step.

That failure is the entire reason this post exists. Reinforcement learning from human feedback is the technique that turned raw next-token predictors into assistants that follow instructions, and at its heart it is exactly the RL loop this series keeps returning to: an agent (the language model policy) interacts with an environment (a prompt plus a reward model that scores its response), collects a reward, and updates its policy to get more reward — while a KL penalty keeps it from drifting into gibberish. The trouble is that the "agent" here has billions of parameters, and naive RLHF wants several copies of it resident in memory simultaneously. The figure below shows the gap I walked into: full fine-tuning RLHF on a 7B model needs a small cluster, while the same loop built on **LoRA** (Low-Rank Adaptation) and **QLoRA** (4-bit quantized LoRA) fits on the single A100 I already had.

![Comparison of full fine-tuning RLHF needing eight A100 GPUs against QLoRA RLHF fitting on one A100 with a quantized base and small adapters](/imgs/blogs/lora-qlora-for-rlhf-1.png)

The stakes here are larger than my one dead job. RLHF used to be something only a handful of well-funded labs could run, because the hardware bill alone gated it. The combination of LoRA and QLoRA changed that: it moved alignment from a multi-GPU industrial process to something a single researcher with one rented GPU can iterate on overnight. That democratization is *why* you see so many open aligned models now — the memory barrier that used to wall off RLHF fell, and it fell specifically because of the techniques in this post. Understanding them is not just an optimization detail; it is the difference between being able to align a model at all and not.

By the end of this post you will be able to do three things. First, you will understand *why* RLHF is so memory-hungry and *where* every gigabyte goes, so you can predict whether a run will fit before you launch it. Second, you will understand the mathematics of LoRA and the NF4 quantization scheme behind QLoRA well enough to choose a rank and a quantization config on purpose rather than by superstition. Third, you will have copy-and-run code — using Hugging Face's PEFT, TRL, and BitsAndBytes — that wires up a full QLoRA RLHF loop on a single GPU, including the trick that lets one quantized base model serve as *both* the policy and the reference policy. We will keep coming back to a concrete running example: aligning a 7B base model into a more helpful assistant, where full fine-tuning demands 240 GB and QLoRA does the same job in roughly 40.

## 1. Why RLHF is a memory disaster (and where every byte goes)

Let me make the running example precise so the memory math is not abstract. We have a 7B-parameter base language model. We have already done supervised fine-tuning (SFT) on it, and we have trained a reward model that, given a prompt and a candidate response, outputs a scalar "how good is this" score. Our goal is the standard RLHF objective: fine-tune the policy with PPO to maximize the reward model's score, minus a KL-divergence penalty that keeps the policy close to the SFT reference. That objective is

$$
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \Big[ r_\phi(x, y) \Big] \;-\; \beta \, \mathbb{E}_{x \sim \mathcal{D}} \Big[ \mathrm{KL}\big( \pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x) \big) \Big]
$$

where $\pi_\theta$ is the policy we are training, $r_\phi$ is the frozen reward model, $\pi_{\text{ref}}$ is the frozen reference (the SFT model before RL), and $\beta$ controls how hard we pull the policy back toward the reference. The KL term is not optional decoration: without it the policy will discover that the reward model has blind spots and will produce degenerate text that the reward model loves and humans hate. This is **reward hacking**, and the reference KL is the leash that prevents it. We covered the *why* of that leash in depth in the alignment posts; here our concern is the *cost* of carrying all these models at once.

Count the models the PPO loop actually instantiates. There is the **policy** $\pi_\theta$, which is trainable. There is the **reference policy** $\pi_{\text{ref}}$, a frozen copy used only to compute the KL term. There is the **reward model** $r_\phi$, frozen, used to score completions. And in actor-critic PPO there is a **value head** (a critic) that estimates the expected return from each token position so we can compute advantages. That is effectively four model-sized objects on the GPU.

Now the bytes. Storing one parameter in BF16 (the standard mixed-precision format for training) takes 2 bytes. So 7B parameters is 14 GB just to *store* the weights once. The policy needs more than storage, though, because it is being trained:

- **Weights**: 7B × 2 bytes = 14 GB.
- **Gradients**: one gradient per weight, 7B × 2 bytes = 14 GB.
- **Adam optimizer states**: Adam keeps a first moment (momentum) and a second moment (variance) per parameter, and the reference implementation keeps them in FP32 (4 bytes each). That is 7B × 8 bytes = 56 GB. Even keeping moments in BF16 it is 28 GB.

So the *trainable* policy alone is roughly 14 + 14 + 56 = 84 GB before you account for activations. Add the frozen reference (14 GB), the frozen reward model (14 GB), the value head (small if it is a single linear layer on top of the policy backbone, but often a full second network in implementations, so call it another 14 GB), and activations for a batch of long sequences (easily 10–40 GB depending on batch and context length). The honest total lands somewhere between 160 and 240 GB. A single 80 GB A100 holds less than half of that. This is exactly the cliff I drove off.

It helps to remember *why* the value head and the reference are even there, because it grounds the memory cost in the RL loop rather than treating it as bookkeeping. The policy is the agent. The environment is "sample a prompt, let the policy generate a completion, hand the (prompt, completion) pair to the reward model, get a scalar back." The reward model is the environment's reward function — it stands in for the human raters who would otherwise have to score every completion by hand. PPO, the algorithm doing the updating, is an actor-critic method: the *actor* is the policy that proposes tokens, and the *critic* is the value head that estimates how good the current state is, so that PPO can compute an **advantage** — how much better a particular action was than the critic expected. The advantage is what the policy gradient actually multiplies against, and using a critic to compute it is what keeps the variance of that gradient low enough for training to be stable. We derive the policy gradient and the advantage formulation in depth in the PPO post; the point here is that every one of the four resident models is load-bearing, not redundant, which is exactly why naive RLHF is so expensive.

Concretely, PPO's clipped surrogate objective for the policy is

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( \rho_t(\theta)\, \hat{A}_t,\; \mathrm{clip}(\rho_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\, \hat{A}_t \big) \Big], \qquad \rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)},
$$

where $\rho_t$ is the importance ratio between the new and old policy, $\hat{A}_t$ is the estimated advantage from the value head, and the clip keeps each update small. Computing $\rho_t$ needs the old policy's log-probabilities; computing $\hat{A}_t$ needs the value head; and the full RLHF objective subtracts the reference-KL penalty on top. So the four models map cleanly onto four jobs: policy (propose tokens), reference (anchor via KL), reward model (supply the scalar reward), value head (estimate advantage). Hold that mapping in your head — it is the lens for every memory decision that follows, because each optimization in this post is really "make one of these four jobs cheaper without breaking the loop."

#### Worked example: where 240 GB comes from

Suppose we run PPO on a 7B policy with a batch of 8 prompts, each generating 512 tokens, sequence length 1024 including the prompt. Tally the resident memory in BF16 training with FP32 Adam moments:

| Component | Formula | Memory |
|---|---|---|
| Policy weights | 7B × 2 B | 14 GB |
| Policy gradients | 7B × 2 B | 14 GB |
| Adam moments (FP32) | 7B × 8 B | 56 GB |
| Reference policy weights | 7B × 2 B | 14 GB |
| Reward model weights | 7B × 2 B | 14 GB |
| Value network weights+grad+Adam | 7B × 12 B | 84 GB |
| Activations (batch 8, len 1024) | empirical | ~25 GB |
| **Total** | | **~221 GB** |

That is three A100s' worth, and we have not added the inference KV-cache for the generation phase or any fragmentation overhead. If the value head shares the policy backbone instead of being a separate 7B network, you shave the 84 GB down to roughly 16 GB and land near 153 GB — still nearly two A100s. The conclusion is unavoidable: full-parameter RLHF on a 7B model is a multi-GPU job, and that is before you scale to 13B or 70B. We need to attack the biggest line items, which are the policy's gradients and optimizer states (84 GB) and the redundant frozen copies (42 GB). LoRA attacks the first; the reference-via-disabled-adapters trick attacks the second; QLoRA attacks what remains.

## 2. LoRA: training a tiny correction instead of the whole weight

The insight behind LoRA, introduced by Hu et al. in 2021, starts from an empirical observation about fine-tuning: when you adapt a large pretrained model to a new task, the *change* in the weights — call it $\Delta W$ — has low intrinsic rank. The model does not need to rewrite all of its knowledge; it needs a small, structured correction. If $\Delta W$ is effectively low-rank, then we are wasting enormous memory by representing it as a full dense matrix and training every entry.

Consider a single weight matrix $W_0 \in \mathbb{R}^{d \times k}$ inside the model — say, the query projection in an attention layer. Full fine-tuning replaces it with $W_0 + \Delta W$ where $\Delta W$ is also $d \times k$, so we train $d \times k$ parameters and store gradients and optimizer states for all of them. LoRA instead **freezes $W_0$ entirely** and represents the update as a product of two thin matrices:

$$
W = W_0 + \Delta W = W_0 + B A, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k).
$$

The rank $r$ is the bottleneck. If $d = k = 4096$ and $r = 16$, then full fine-tuning of this one matrix trains $4096 \times 4096 \approx 16.8$ million parameters, while LoRA trains $4096 \times 16 + 16 \times 4096 \approx 131{,}000$ — a 128× reduction for this matrix. The forward pass becomes

$$
h = W_0 x + \frac{\alpha}{r} B A x,
$$

where $\alpha$ is a scaling constant. The figure below traces this dataflow: the input branches into the frozen path through $W_0$ and the trainable path through $A$ then $B$, and the two outputs sum.

![LoRA decomposition showing input branching into a frozen weight path and a trainable low-rank path through matrices A and B before summing](/imgs/blogs/lora-qlora-for-rlhf-2.png)

Why should we *believe* that $\Delta W$ is low-rank in the first place? The argument has both an empirical and a theoretical leg. Empirically, Aghajanyan et al. showed that large pretrained models have a low *intrinsic dimension* — you can fine-tune them to within a few percent of full performance by optimizing in a randomly chosen low-dimensional subspace, sometimes just a few hundred dimensions for a task. That is a strong signal that the *task-specific* part of the weight change lives in a small subspace, even though the model itself is enormous. Theoretically, the rank of $BA$ is at most $r$ by construction (the rank of a product is bounded by the smaller of the two factor ranks), so we are explicitly betting that the best $\Delta W$ of rank $r$ captures most of the benefit of the best $\Delta W$ of full rank. The LoRA paper's ablations show this bet pays off: rank as low as 1 or 2 already recovers a large fraction of full-fine-tuning quality on many tasks, and the gains saturate quickly as you raise $r$. The intuition is that fine-tuning is *adaptation*, not *re-learning* — you are steering a model that already knows the language, and steering is a low-dimensional operation compared to building the knowledge from scratch.

Let me make the gradient flow explicit, because it is where the memory saving lives. In ordinary backpropagation through $h = Wx$, the optimizer needs $\partial L / \partial W$, a $d \times k$ matrix, plus Adam's two moment matrices of the same shape — three full matrices per weight. Under LoRA, the loss flows back only into $A$ and $B$:

$$
\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, \frac{\partial L}{\partial h}\, (Ax)^\top, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top \frac{\partial L}{\partial h}\, x^\top.
$$

These gradients have shapes $d \times r$ and $r \times k$ — the same small shapes as $B$ and $A$ themselves. The frozen $W_0$ contributes to the forward activation but receives no gradient, so the optimizer allocates *nothing* for it: no gradient buffer, no momentum, no variance. For our 4096×4096 example at $r=16$, the optimizer footprint per matrix drops from three 16.8M-element matrices to three 131K-element matrices — a 128× cut on the most expensive line item from Section 1. Multiply that across every adapted matrix in the model and you recover essentially all of the 84 GB the policy's gradients and Adam states cost under full fine-tuning. That is the single biggest lever, and it is pure linear algebra: the rank of the update, not a trick.

Two details make this actually work in practice. First, **initialization**. We initialize $A$ with small random Gaussian values, $A \sim \mathcal{N}(0, \sigma^2)$, and we initialize $B = 0$. This means at the very start of training $\Delta W = B A = 0$, so the adapted model is *exactly* the pretrained model — no random perturbation to the carefully learned weights on step zero. Training then nudges $B$ away from zero and the correction grows smoothly. If you initialized both randomly, the first forward pass would inject noise into every layer and you would start from a worse point than the base model.

Second, the **scaling factor $\alpha / r$**. The point of dividing by $r$ is that it decouples the learning rate from the choice of rank. Without it, doubling the rank would roughly double the magnitude of $\Delta W$ for the same parameter scale, forcing you to retune the learning rate every time you change $r$. With the $\alpha/r$ scaling, you can sweep ranks while keeping $\alpha$ fixed and the effective update magnitude stays comparable. A common convention is $\alpha = 2r$ (so $\alpha/r = 2$) or $\alpha = r$ (so the factor is 1). Treat $\alpha$ as a knob that controls how aggressively the adapter speaks relative to the frozen base.

Here is the core LoRA layer in PyTorch, stripped to its essence so you can see there is no magic:

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base_linear              # frozen pretrained weight W0
        for p in self.base.parameters():
            p.requires_grad = False          # W0 receives no gradient

        d_out, d_in = base_linear.weight.shape
        self.r = r
        self.scaling = alpha / r             # the alpha/r factor

        # A ~ N(0, sigma^2), B = 0  ->  delta W = B A = 0 at init
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        base_out = self.base(x)                          # frozen path: W0 x
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T   # trainable path: B A x
        return base_out + self.scaling * lora_out
```

Notice what gets gradients: only `lora_A` and `lora_B`. The base linear's parameters have `requires_grad = False`, so the optimizer never allocates moments for them. This is the whole memory win in miniature. In a real model you do not hand-wrap every layer like this — the PEFT library does it for you — but it is worth seeing once so the mechanism is concrete rather than mysterious.

## 3. LoRA mathematics deep dive: choosing the rank on purpose

Section 2 asserted that $\Delta W$ is low-rank and that you should therefore pick a small $r$. That is the right conclusion, but if you stop at the assertion you will keep choosing $r$ by superstition. This section derives *why* low rank is enough and turns the choice of $r$ into something you can reason about from a quality budget rather than a gut feeling. There are three threads to pull: the intrinsic-dimensionality argument that tells you the task lives in a tiny subspace, the singular-value structure of $\Delta W$ that tells you a low-rank truncation loses almost nothing, and the rank-sensitivity data that tells you where the quality curve actually flattens.

### 3.1 The intrinsic dimensionality argument

The strongest theoretical justification for LoRA predates LoRA. Aghajanyan, Zettlemoyer, and Gupta asked a deceptively simple question in 2021: how many free parameters do you *actually* need to fine-tune a large pretrained model to a given task? Their method was to constrain fine-tuning to a randomly chosen $d'$-dimensional affine subspace of the full parameter space — you optimize a $d'$-dimensional vector $\theta_{d'}$ and map it up to the full parameter space via a fixed random projection, $\theta = \theta_0 + P\,\theta_{d'}$, where $\theta_0$ is the pretrained checkpoint and $P$ is a fixed random matrix. They then defined the **intrinsic dimension** of a task as the smallest $d'$ at which this constrained fine-tuning reaches 90% of the performance of full fine-tuning.

The numbers are startling. For a model like RoBERTa, many standard NLP tasks have an intrinsic dimension on the order of a few hundred — sometimes as low as ~200 parameters, against a model with hundreds of millions. Equally important, they found that *larger pretrained models have lower intrinsic dimension*: the better the pretraining, the smaller the subspace you need to nudge to specialize the model. The interpretation that matters for LoRA is direct: if 200 free parameters can carry a task, then the weight change $\Delta W$ that full fine-tuning discovers is overwhelmingly redundant. It has the *shape* of a full dense matrix, but the information it carries lives in a tiny subspace. LoRA's $BA$ factorization is precisely a way to *parameterize* that small subspace explicitly — instead of a random projection $P$, LoRA learns the projection (the columns of $B$) jointly with the coordinates (the rows of $A$).

This reframes the rank choice. The intrinsic dimension is a per-task property. A task that only needs the model to change its tone or refuse a category of request has a tiny intrinsic dimension and will be satisfied by $r=4$ or $r=8$ spread across the adapted matrices. A task that asks the model to acquire a genuinely new skill it never saw in pretraining has a larger intrinsic dimension and wants more rank. For RLHF specifically, alignment is closer to the first case — you are steering an already-capable model toward helpfulness and away from harm, not teaching it new facts — which is the deep reason why such low ranks work so well for the RLHF loop. You are operating in the regime Aghajanyan et al. showed is low-dimensional.

### 3.2 Singular-value structure of the weight update

The intrinsic-dimension argument says the subspace is small. The singular value decomposition (SVD) of $\Delta W$ says the same thing in a way you can measure directly and that tells you exactly what a rank-$r$ truncation throws away. Any real matrix $\Delta W \in \mathbb{R}^{d \times k}$ has an SVD

$$
\Delta W = U \Sigma V^\top = \sum_{i=1}^{\min(d,k)} \sigma_i\, u_i v_i^\top, \qquad \sigma_1 \ge \sigma_2 \ge \dots \ge 0,
$$

where the $\sigma_i$ are the singular values in descending order and $u_i, v_i$ are the left and right singular vectors. The Eckart–Young theorem makes the LoRA bet rigorous: among all rank-$r$ matrices, the best approximation of $\Delta W$ in both the Frobenius and spectral norms is the truncated SVD that keeps the top $r$ terms, and the error of that approximation is governed entirely by the *discarded* singular values:

$$
\min_{\mathrm{rank}(M)\le r} \lVert \Delta W - M \rVert_F^2 = \sum_{i=r+1}^{\min(d,k)} \sigma_i^2.
$$

So the question "is rank $r$ enough?" becomes the concrete, measurable question "how fast do the singular values $\sigma_i$ of $\Delta W$ decay?" If they decay rapidly — if $\sigma_1, \dots, \sigma_r$ hold most of the total energy $\sum_i \sigma_i^2$ — then truncating at $r$ loses a tiny fraction of $\Delta W$ and LoRA is nearly lossless. If they decay slowly, $\Delta W$ is genuinely high-rank and LoRA will leave quality on the table.

You can run this experiment yourself, and it is the single most convincing way to see why LoRA works. Take a model, do a *full* fine-tune on your task, subtract the original weights to get $\Delta W$ for each matrix, and plot the singular value spectrum:

```python
import torch

def singular_value_spectrum(W0, W_finetuned):
    """Return the singular values of the weight update, largest first."""
    dW = (W_finetuned - W0).float()
    # full_matrices=False gives the economy SVD; we only need the singular values
    sv = torch.linalg.svdvals(dW)
    return sv  # already sorted descending

def energy_retained(sv, r):
    """Fraction of squared Frobenius energy kept by a rank-r truncation."""
    total = (sv ** 2).sum()
    kept = (sv[:r] ** 2).sum()
    return (kept / total).item()

# Example: a query-projection update from a full fine-tune
sv = singular_value_spectrum(W0_q, W_ft_q)   # shape [4096]
for r in [1, 4, 8, 16, 32, 64, 128]:
    print(f"r={r:>4}  energy retained = {energy_retained(sv, r):.3f}")
```

What you almost always see on a fine-tuning update is a spectrum that looks like a steep cliff followed by a long flat shelf: the first handful of singular values are large, then they drop by an order of magnitude and trail off. A representative printout from a query-projection update looks like this:

| Truncation rank $r$ | Energy retained |
|---|---|
| 1 | 0.34 |
| 4 | 0.71 |
| 8 | 0.86 |
| 16 | 0.94 |
| 32 | 0.98 |
| 64 | 0.995 |
| 128 | 0.999 |

Read that table next to the rank-sweep tables from Section 4 and they tell the same story from two directions. Eighty-six percent of the update's energy lives in the top 8 directions; ninety-four percent in the top 16. Going from $r=16$ to $r=64$ recovers the last six percent of energy — and that last six percent is exactly the part that the rank-sweep experiments show moves your held-out reward by less than the run-to-run noise. The singular spectrum is *why* the quality curve flattens: there is simply not much $\Delta W$ left to capture once you have the top dozen-or-so directions. The cliff-then-shelf shape is the mathematical signature of the intrinsic-dimension result, seen one matrix at a time.

One honest caveat: this argument is about approximating the $\Delta W$ that full fine-tuning *would* find. LoRA does not first compute that $\Delta W$ and then truncate it — it learns $B$ and $A$ directly by gradient descent, and the optimum it reaches is not guaranteed to be the truncated SVD of the unconstrained solution. In practice the learned low-rank update tracks the top singular directions well because those are where the loss gradient is largest, but the SVD picture is a *bound and a guide*, not a literal description of the training dynamics. It tells you that a good rank-$r$ solution exists; gradient descent's job is to find it.

### 3.3 Rank sensitivity in practice

The theory says a knee should exist; the empirical literature says where it usually is. Across instruction-following and preference-optimization tasks, the consistent finding is that quality climbs steeply from $r=1$ to roughly $r=8$, climbs gently from $r=8$ to $r=16$, and is essentially flat from $r=16$ to $r=64$ and beyond. Concretely, on instruction-following evaluations you will commonly see a pattern like this — the absolute scores depend on your model and benchmark, but the *shape* is robust:

| Rank $r$ | Instruction-following score | Δ vs previous |
|---|---|---|
| 8 | 71.2 | — |
| 16 | 72.0 | +0.8 |
| 32 | 72.1 | +0.1 |
| 64 | 72.3 | +0.2 |

The step from $r=8$ to $r=16$ is real and worth taking; the steps beyond are inside the noise floor of the evaluation, where re-running the same config with a different seed moves the score by as much as the rank change does. This is the same lesson the singular-value table delivered analytically: by $r=16$ you have captured the directions that carry the signal, and adding rank is adding capacity the task does not have the intrinsic dimension to use. The practical rule that falls out is the one Section 4 will operationalize — sweep a few ranks on a short run, find the knee, and stop. The math in this section is what lets you trust that the knee you measure is a property of the task and not an artifact of your particular run.

### 3.4 Why $\alpha = 2r$ is a sensible default

Section 2 introduced the $\alpha/r$ scaling and asserted that it decouples the learning rate from the rank. It is worth deriving why, because once you see it you will stop retuning the learning rate every time you change $r$. Consider the magnitude of the LoRA update $\Delta W = \frac{\alpha}{r} BA$. With $A$ initialized at scale $\sigma$ (each entry $\sim \mathcal{N}(0,\sigma^2)$) and $B$ starting at zero, the entries of $BA$ after training reach a magnitude that, for a fixed amount of optimization, tends to grow with the inner dimension $r$ — you are summing $r$ rank-one terms $b_i a_i^\top$, and the variance of that sum scales with the number of terms. Roughly, $\lVert BA \rVert$ scales like $\sqrt{r}$ (or like $r$ depending on how correlated the terms become), so without correction, raising the rank inflates the effective size of the weight update, which is equivalent to silently raising the learning rate.

Dividing by $r$ counteracts this growth so that the *effective* update magnitude — the thing the rest of the network feels — stays comparable as you change $r$. That is the whole point: you sweep ranks while holding $\alpha$ fixed, and the optimization dynamics stay in the same regime instead of requiring a fresh learning-rate search at every rank. The two common conventions follow directly. Setting $\alpha = r$ makes the scaling factor exactly 1, the most conservative choice. Setting $\alpha = 2r$ makes the factor 2, which lets the adapter speak twice as loudly relative to the frozen base — empirically a good default because the adapters start at zero and benefit from a little extra gain to make their correction felt early in training without dominating the base before they have learned anything sensible. The heuristic $\alpha = 2r$ is popular precisely because it keeps the learning-rate-equivalent constant across ranks (you only retune if you change $\alpha$, not $r$) while giving the adapter enough authority to converge in a reasonable number of steps. If you ever see training stall at higher ranks after a rank bump, check whether you left $\alpha$ fixed instead of scaling it with $r$ — that mistake reintroduces exactly the learning-rate coupling the $\alpha/r$ factor exists to remove.

## 4. Which layers get adapters, and how rank trades against quality

You do not have to put LoRA on every weight matrix in the model, and you usually should not. The original LoRA paper studied where the adapters earn their keep and found that applying them to the **attention projection matrices** gives most of the benefit. In transformer attention there are four candidate matrices per layer: the query projection $W_q$, the key projection $W_k$, the value projection $W_v$, and the output projection $W_o$. The paper's headline result was that adapting $W_q$ and $W_v$ alone — leaving $W_k$ and $W_o$ frozen — recovered most of the quality of adapting all four, at half the adapter parameters.

Why query and value specifically? The argument the paper offers is that adapting more *types* of matrices at a low rank beats adapting fewer types at a high rank, for a fixed parameter budget. If you have a budget of, say, 18 million adapter parameters, you are better off spreading rank-4 adapters across $W_q$, $W_k$, $W_v$, $W_o$ than concentrating a rank-16 adapter on $W_q$ alone. The reason is that different projections steer different parts of attention — queries and values shape *what* the model attends to and *what information it pulls*, and touching both gives the adaptation more independent levers than doubling down on one. The practical corollary is that when you scale up your adapter budget, your first move should be to *add target modules*, not to crank a single module's rank into the hundreds.

In modern practice the default has crept toward adapting all four attention matrices and sometimes the feed-forward (MLP) layers too, because the marginal cost is tiny relative to the base model and the extra capacity helps on harder tasks. The MLP layers in a modern transformer hold the bulk of the parameters (the feed-forward dimension is typically 3–4× the hidden dimension), so adapting `gate_proj`, `up_proj`, and `down_proj` gives the model access to far more of its capacity — at the cost of more trainable adapter parameters. For RLHF I default to adapting all four attention projections plus the MLP projections, because the behavioral changes alignment asks for (be more helpful, refuse unsafe requests, follow format instructions) seem to benefit from touching the feed-forward pathways where a lot of the model's "what to say" computation lives, not just the attention routing. If memory is desperately tight you can drop back to query/value only and lose surprisingly little. The trade-off is real but gentle: more target modules means more trainable parameters and a bit more memory, in exchange for more expressive corrections. For RLHF specifically, where the policy must learn fairly subtle behavioral shifts rather than wholesale new knowledge, adapting query and value (and often the MLP up/down projections) is a sensible default.

The other dial is the **rank** $r$ itself, and it is the one people most often set on vibes. The principle is a capacity-versus-cost trade-off:

| Rank $r$ | Trainable params (7B, qv) | Behavior |
|---|---|---|
| 1–2 | ~2M | Often too little capacity; policy barely improves, reward plateaus early |
| 8–16 | ~8–16M | Sweet spot for most 7B RLHF; good reward gains, minimal memory |
| 32–64 | ~30–60M | More capacity for hard tasks; diminishing returns, slightly more memory |
| 128–256 | ~120–240M | Approaches full fine-tuning; you lose the memory advantage |

The honest empirical finding across many tasks is that quality saturates surprisingly early — going from $r=8$ to $r=64$ often moves your metric by less than the run-to-run noise, while going from $r=1$ to $r=8$ moves it a lot. The failure mode at very high rank is not that the model gets worse (it does not, much); it is that you have quietly given up the memory savings that motivated LoRA in the first place. At $r=256$ on a 7B model adapting all linear layers, you can be training hundreds of millions of parameters, and the optimizer-state line item starts to bite again.

#### Worked example: a rank-selection experiment

Here is the kind of sweep I run before committing to a long RLHF job. Take a 7B SFT model, a fixed reward model, and a held-out set of prompts. Run short PPO (say 200 steps) at several ranks and measure two things: the mean reward-model score on held-out completions, and the GPU memory at peak. Representative numbers from a run like this:

| Rank $r$ | Trainable params | Peak GPU mem | Held-out reward (mean) |
|---|---|---|---|
| 4 | ~4M | 38 GB | +0.42 |
| 8 | ~8M | 39 GB | +0.71 |
| 16 | ~16M | 40 GB | +0.78 |
| 32 | ~32M | 42 GB | +0.79 |
| 64 | ~64M | 45 GB | +0.80 |

The reward jumps from $r=4$ to $r=8$ (capacity was binding), then the curve flattens hard: $r=16$ to $r=64$ buys you 0.02 reward for 5 GB more memory. The decision is easy — $r=16$ is the knee of the curve. I would only reach for $r=32$ if I saw the reward still climbing at the end of the short run, which would signal capacity is still binding. The lesson is to *measure the knee* on a short run rather than guessing; the sweep costs an hour and saves you from either an under-capacity run that never converges or an over-budget run that does not fit.

## 5. The reference policy for free: disabled adapters

Now we attack the second-biggest line item from Section 1: the frozen reference policy. Recall that the KL penalty needs $\pi_{\text{ref}}$, the SFT model *before* RL began. In full fine-tuning you keep a whole second copy of the 7B model in memory just for this — 14 GB doing nothing but computing reference log-probabilities.

LoRA makes that copy disappear. The reference policy is, by definition, the base model with *no* adapter correction — that is exactly $W_0$, the frozen weights we already have resident. The policy is $W_0 + \frac{\alpha}{r} BA$. So the policy and the reference share the same base weights and differ only by whether the adapters are active. To get reference log-probabilities, you do a forward pass with the adapters **disabled**; to get policy log-probabilities, you do a forward pass with them **enabled**. One set of base weights, one set of adapters, two behaviors. The figure below shows this four-model setup collapsed onto a single shared base.

![Four RLHF models sharing one frozen base where the policy enables adapters, the reference disables them, and the value head and reward model feed the PPO loss](/imgs/blogs/lora-qlora-for-rlhf-7.png)

Before the code, it is worth being precise about *why* the reference matters at all, because it justifies paying for it even when it is "free." The KL penalty $-\beta\,\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ is a regularizer that keeps the policy's output distribution close to the SFT distribution. Expand the per-token KL:

$$
\mathrm{KL}\big(\pi_\theta(\cdot \mid s) \,\|\, \pi_{\text{ref}}(\cdot \mid s)\big) = \sum_a \pi_\theta(a \mid s) \log \frac{\pi_\theta(a \mid s)}{\pi_{\text{ref}}(a \mid s)} = \mathbb{E}_{a \sim \pi_\theta}\!\left[ \log \pi_\theta(a \mid s) - \log \pi_{\text{ref}}(a \mid s) \right].
$$

That last expectation is exactly `policy_logp - ref_logp` averaged over sampled tokens — which is why the implementation computes the KL as a simple log-prob difference rather than a sum over the whole vocabulary. Now the mechanism: the reward model $r_\phi$ is an *imperfect* proxy for human preference, trained on finite data, so it has regions of input space where it assigns high scores to text humans would dislike — adversarial blind spots. Pure reward maximization will *find* those regions, because optimization is relentless; this is reward hacking, and left unchecked it produces fluent-looking nonsense or repetitive exploits that spike the reward model's score. The KL term fights this by penalizing the policy for moving probability mass away from the reference's plausible-text distribution. The policy can only collect the reward-model's score *while staying close to text the SFT model considered reasonable*. Tune $\beta$ up and the policy stays glued to the reference (safe, but it barely improves); tune $\beta$ down and it chases reward harder (more gains, more hacking risk). The whole art of stable RLHF is finding the $\beta$ where the policy improves on real quality without diverging — and you cannot even compute that penalty without a reference, which is why the disabled-adapter trick is not just a memory optimization but an enabler of correct training on one GPU.

PEFT exposes this directly through a context manager that turns the adapters off and back on. Inside TRL's PPO trainer this is handled for you, but it is worth seeing the mechanism, because the first time you realize the reference is *free* it changes how you budget memory:

```python
import torch

def compute_logprobs(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

# Policy log-probs: adapters ENABLED (default)
policy_logp = compute_logprobs(model, input_ids, attention_mask)

# Reference log-probs: adapters DISABLED -> base model behavior
with model.disable_adapter():
    with torch.no_grad():
        ref_logp = compute_logprobs(model, input_ids, attention_mask)

# KL contribution per token, no separate reference model in memory
kl = policy_logp - ref_logp
```

The `model.disable_adapter()` context manager temporarily zeros out the LoRA contribution so the forward pass uses only $W_0$. We wrap the reference pass in `torch.no_grad()` because the reference never receives gradients. The saving is exactly one full model copy — 14 GB on our 7B example — and it scales: at 70B it is 140 GB you simply do not pay. This single trick is what makes LoRA RLHF qualitatively different from full RLHF rather than just incrementally cheaper.

#### Why the reference *is* the base model with adapters disabled

The trick is so clean that it is worth stating the identity precisely, because it only holds under a specific assumption and that assumption is the source of a real bug if you violate it. The chain of equalities is:

$$
\pi_{\text{ref}} \;=\; \pi_{\text{SFT}} \;=\; \text{base model with adapters disabled} \;=\; W_0.
$$

Read it left to right. The RLHF objective defines the reference as the policy *before* RL began — that is the SFT model, by construction. In a LoRA RLHF run you load the *already-SFT'd* checkpoint as the frozen base $W_0$ and you attach freshly initialized adapters on top of it. At step zero the adapters are zero ($B=0$), so policy and reference coincide; as PPO runs, the adapters move and the policy drifts away from the reference, which is exactly the KL the penalty measures. Disabling the adapters at any point recovers $W_0$, which *is* $\pi_{\text{SFT}}$, which *is* $\pi_{\text{ref}}$. No second model, no second set of weights — the reference is a *mode* of the one model you already loaded, toggled by a context manager. That is why `ref_model=None` is correct rather than a hack: there is genuinely nothing to point a second model at.

The implementation detail is the pair `disable_adapter_layers()` / `enable_adapter_layers()` (or the `disable_adapter()` context manager that wraps them). The pattern in a hand-rolled loop is:

```python
# Reference forward: turn the adapters OFF -> pure W0 == SFT == reference
model.disable_adapter_layers()
with torch.no_grad():
    ref_logp = compute_logprobs(model, input_ids, attention_mask)
model.enable_adapter_layers()   # turn them back ON for the policy

# Policy forward: adapters ON -> W0 + (alpha/r) BA
policy_logp = compute_logprobs(model, input_ids, attention_mask)
```

The memory accounting is the whole point: this eliminates one full model copy. For a 7B model in BF16 that is the 14 GB the second copy would have cost; for 13B it is 26 GB; for 70B it is 140 GB. In the single-A100 budget tables, the "reference policy" line reads **0 GB** entirely because of this identity.

#### The subtle numerical issue: which model did SFT actually produce?

There is one trap, and it bites people who run SFT and RLHF as separate LoRA stages. The identity "reference = base with adapters disabled" is exact *only if the frozen base $W_0$ is itself the SFT model*. That is the case if you **merged** your SFT LoRA adapters into the base before starting RLHF, so $W_0$ literally contains the SFT weights. But if you did SFT with LoRA and then, for RLHF, loaded the *original pretrained* base and stacked *new* RLHF adapters on top — without merging the SFT adapters in — then disabling the RLHF adapters gives you the *pretrained* model, not the SFT model. In that configuration:

$$
\text{base (pretrained)} + \text{disabled RLHF adapters} \;=\; \text{pretrained} \;\ne\; \pi_{\text{SFT}}.
$$

Your KL is now measured against the wrong reference — the pretrained model rather than the SFT model — which silently changes what the penalty is regularizing toward and can destabilize the run or pull the policy back toward un-instruction-tuned behavior. The fix is to be deliberate about the base: **merge the SFT adapters into the base first** (`merge_and_unload()` on the SFT-adapter model), save that merged checkpoint, and load *it* as $W_0$ for RLHF. Then disabling the RLHF adapters truly recovers $\pi_{\text{SFT}}$ and the free-reference identity holds exactly. If for some reason you must keep the SFT adapters separate, you cannot use the disabled-adapter reference — you would need an explicit reference model that has the SFT adapters applied, which gives back the memory you were trying to save. The clean path, and the one TRL assumes, is: SFT, merge, then RLHF on the merged base.

## 6. QLoRA: putting the frozen base in 4 bits

LoRA shrank the trainable footprint and the reference copy. What is left is the frozen base model itself — 14 GB in BF16 for our 7B — plus the frozen reward model. These are pure dead weight in the sense that they are never updated, yet they still cost full-precision storage. QLoRA, introduced by Dettmers et al. in 2023, asks the obvious question: if these weights are frozen and only read during the forward pass, why store them in 16 bits at all?

The QLoRA recipe has three ingredients, and the memory layers stack as shown below: a 4-bit frozen base, BF16 adapters on top, and a paged optimizer that spills to CPU RAM under pressure.

![QLoRA memory layers stacking a 4-bit frozen base under BF16 LoRA adapters with a CPU-offloaded paged Adam optimizer and BF16 activations](/imgs/blogs/lora-qlora-for-rlhf-3.png)

The three ingredients are:

1. **4-bit NF4 quantization of the frozen base.** Each base weight is stored in 4 bits (0.5 bytes) instead of 16 bits (2 bytes), a 4× reduction. Our 7B base drops from 14 GB to about 3.5 GB. We will dig into *why* NF4 specifically in the next section.
2. **LoRA adapters in BF16.** The trainable adapters stay in full BF16 precision because they are the part actually learning, and they are tiny (tens of MB), so there is no point quantizing them. During the forward pass the 4-bit base weights are dequantized to BF16 block-by-block on the fly, the matmul happens in BF16, and the dequantized block is discarded.
3. **Paged optimizer.** The Adam optimizer states for the adapters live in GPU memory normally, but QLoRA uses NVIDIA's unified-memory paging so that if a memory spike (a long sequence, a gradient checkpoint) threatens an OOM, optimizer pages are automatically evicted to CPU RAM and paged back when needed. This prevents the sporadic OOM-on-the-longest-batch failure that otherwise plagues tight runs.

A couple of mechanics are worth understanding so the numbers do not feel like magic. The **dequantize-on-the-fly** step means the 4-bit weights are never used directly in a matmul — GPUs do not have native 4-bit matmul for this, and you would lose accuracy anyway. Instead, for each weight block touched in the forward pass, BitsAndBytes reads the 4-bit codes, multiplies by the stored block scale to recover BF16 values, runs the BF16 matmul, and discards the dequantized block immediately. So the *peak* memory only ever holds one dequantized block at a time, not the whole layer in BF16. The cost is compute: dequantization adds a small overhead per forward pass, which is why plain LoRA (BF16 base, no dequant) is slightly faster per step than QLoRA. You trade a few percent of step time for a 4× memory cut on the base — almost always the right trade when memory is the binding constraint.

The **paged optimizer** deserves a sentence more, because it is the unsung hero of tight runs. RLHF has spiky memory: the generation phase builds a KV-cache, long completions inflate activations, and gradient checkpointing recomputes activations in bursts. Any one of these spikes can push you over the edge for a single batch even if the *average* footprint fits. Paged Adam uses NVIDIA's unified memory so that the optimizer's moment buffers can be transparently evicted to CPU RAM during a spike and paged back when the pressure passes — the same idea as OS virtual memory, applied to optimizer state. The result is that you get OOM resilience without manually orchestrating offloading. Combined with **gradient checkpointing** (recomputing activations on the backward pass instead of storing them, which `prepare_model_for_kbit_training` enables), these two tricks turn a run that would OOM on its longest batch into one that completes.

The combined effect is dramatic. Dettmers et al. demonstrated fine-tuning a **65B** model on a single 48 GB GPU — something that full fine-tuning would spread across a dozen GPUs. The key claim QLoRA defends empirically is that 4-bit NF4 base weights plus BF16 adapters match the task performance of 16-bit full fine-tuning. You pay essentially nothing in quality for a 4× reduction in the largest memory line item.

## 7. NF4: why a custom 4-bit format beats plain INT4

This is the part most people skip, and it is the part worth understanding, because it explains why QLoRA does not degrade quality the way you would fear from "4-bit." The naive way to put weights in 4 bits is INT4: carve the range $[w_{\min}, w_{\max}]$ into 16 evenly spaced buckets and snap each weight to the nearest bucket. That is fine if your values are uniformly distributed. But neural network weights are *not* uniform — they are very close to zero-mean Gaussian. Most weights cluster tightly around zero, and only a few are large. Uniform INT4 spends half its 16 levels covering the sparse tails where almost no weights live, and starves the dense region near zero where precision actually matters.

**NF4 (Normal Float 4)** fixes this by making the quantization levels match the distribution. The idea is *quantile quantization*: instead of equal-width buckets, use buckets that each contain an equal *fraction* of a standard normal distribution. Concretely, you take the standard normal $\mathcal{N}(0,1)$, find the 16 quantile points that split it into equal-probability regions, and use those as your quantization levels. Because the normal is dense near zero, the levels cluster near zero too — exactly where the weights are. The format is *information-theoretically optimal* for normally distributed inputs in the sense that each of the 16 codes is used roughly equally often, maximizing the bits' usefulness.

Let me make the quantile-quantization idea precise, because "match the distribution" is easy to say and worth seeing in formula. Suppose your data $x$ is drawn from a distribution with cumulative distribution function $F$. For a $b$-bit code you have $2^b$ levels; you want each level to be the *representative* of an equal-probability slice of the data. The optimal level for the $i$-th slice is the value whose cumulative probability sits at the midpoint of that slice:

$$
q_i = F^{-1}\!\left( \frac{i - 0.5}{2^b} \right), \qquad i = 1, \dots, 2^b.
$$

For NF4, $F$ is the standard normal CDF $\Phi$ and $b = 4$, so the 16 levels are $\Phi^{-1}$ evaluated at 16 evenly spaced cumulative-probability points. Because $\Phi^{-1}$ is steep near 0 and 1 (the tails) and flat in the middle, the resulting levels are densely packed near zero and sparse in the tails — the mirror image of where the weights actually live, which is what we want. Plain INT4, by contrast, is what you get if you (wrongly) assume $F$ is uniform: equal-width levels everywhere. The whole NF4 win is replacing the uniform $F$ with the normal $F$ that the weights actually follow.

#### Worked example: quantizing a weight block to NF4

Take a single block of weights $[0.02, -0.15, 0.41, -0.03, 0.88, -0.62, 0.07, 0.19]$. Step one, find the block's absolute maximum: $\lvert -0.62 \rvert$ and $0.88$ — the max is $0.88$, so the scale $s = 0.88$. Step two, normalize the block by dividing by $s$, giving $[0.023, -0.170, 0.466, -0.034, 1.0, -0.705, 0.080, 0.216]$, now all in $[-1, 1]$. Step three, snap each normalized value to the nearest of the 16 NF4 levels (the normal quantiles, which include an exact 0.0). The small values near zero — $0.023$, $-0.034$, $0.080$ — land on closely spaced levels, so they keep real precision; this is exactly the region INT4 would have starved. The large value $1.0$ lands on the top level. Step four, store two things: the 8 four-bit codes (4 bytes total for the block) and the FP32 scale $s$ (4 bytes). To dequantize during the forward pass, look up each code's level and multiply by $s$. The error on the small near-zero weights is tiny precisely because NF4 spent its levels there — and on a Gaussian weight matrix the *vast majority* of weights are small, so the average error is far below what uniform INT4 would give for the same 4 bits.

There is a practical subtlety: a single normal's quantiles are not symmetric around zero in a way that includes an exact zero, and you want zero to be representable exactly (because zeros are common and you do not want to add bias). NF4 handles this by constructing the levels so that zero is one of the 16 codes and the rest are the normal quantiles, giving an asymmetric-but-zero-centered codebook. To apply it, each block of weights (typically 64 values) is normalized by its own absolute maximum so it lands in $[-1, 1]$, then each normalized value is matched to the nearest NF4 level. The per-block scale (the absolute max) is stored alongside as the *quantization constant*.

#### The full NF4 quantization procedure, step by step

It is worth writing the whole procedure out as an explicit algorithm, because every step maps to a line of memory in the budget tables and to a concrete bit somewhere on the GPU. Quantizing a tensor of frozen weights to NF4 is four steps.

**Step 1 — compute the 16 NF4 levels once.** These are fixed for the whole model; you compute them a single time and reuse them. They are the theoretical quantile positions of a standard normal for $2^4 = 16$ levels, constructed to include an exact zero. The recipe Dettmers et al. use is to take the standard normal CDF $\Phi$, compute quantile points on the negative side and the positive side separately (so the asymmetry of including zero is handled cleanly), normalize the whole set so it spans $[-1, 1]$, and force the middle level to be exactly $0.0$. Conceptually:

```python
import torch
from scipy.stats import norm

def make_nf4_levels():
    # 8 quantiles for the negative half, 8 for the positive half (one is 0)
    # offsets keep the extreme quantiles away from +/-inf
    neg = norm.ppf(torch.linspace(0.5/8, 0.5, 8)[:-1]).tolist()  # negative side
    pos = norm.ppf(torch.linspace(0.5, 1 - 0.5/9, 9)[1:]).tolist()  # positive side
    levels = sorted(neg + [0.0] + pos)
    levels = torch.tensor(levels)
    return levels / levels.abs().max()          # normalize into [-1, 1]

NF4_LEVELS = make_nf4_levels()   # 16 fixed values, densest near 0
```

The exact arithmetic of the offsets is fiddly (the published table is the canonical reference), but the property that matters is the one we derived in the previous subsection: the 16 levels are densely packed near zero and sparse in the tails, because $\Phi^{-1}$ is flat in the middle and steep at the ends.

**Step 2 — map each weight to its nearest level.** Weights are processed in blocks (typically 64 values). For each block you first normalize by the block's absolute maximum so the block lands in $[-1, 1]$, then snap each normalized value to the nearest of the 16 NF4 levels:

```python
def quantize_block_nf4(block, levels):
    scale = block.abs().max()                    # the per-block constant
    normalized = block / scale                   # now in [-1, 1]
    # nearest-level lookup: index of the closest NF4 level for each weight
    idx = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return idx.to(torch.uint8), scale            # 4-bit codes + FP32 scale
```

**Step 3 — store the 4-bit code.** The output of the lookup is an integer in $0..15$ per weight — exactly 4 bits. Two codes pack into one byte, so the block of 64 weights becomes 32 bytes of codes. This is the 0.5-bytes-per-parameter line in every budget table.

**Step 4 — store a BF16 (or FP32) scaling factor per block.** The per-block absolute maximum from Step 2 is the *quantization constant*; you need it to reconstruct real values at forward time. Stored naively as FP32, it is 4 bytes per 64-weight block. Dequantization at forward time simply inverts the lookup: read the code, index into `NF4_LEVELS`, multiply by the block scale, and you have the BF16 weight back.

```python
def dequantize_block_nf4(codes, scale, levels):
    return levels[codes.long()] * scale          # back to BF16 values
```

That is the entire scheme. The reason it costs almost no quality, restated now that you have seen the mechanism: a Gaussian weight matrix puts the vast majority of its mass in the dense-near-zero region, and Step 1 spent the codebook's resolution exactly there, so the average reconstruction error in `dequantize_block_nf4` is far below what uniform INT4 would produce with the same 4 bits.

#### Double quantization: quantizing the scales themselves

Then comes the second clever trick: **double quantization**. Those per-block quantization constants from Step 4 are themselves numbers you have to store — one FP32 scale per 64-weight block. For 7B parameters that is about 109 million blocks, and at 4 bytes each that is 437 MB of pure overhead. Double quantization quantizes the quantization constants too. It takes the FP32 block scales, groups *them* into blocks of 256, and stores each scale in 8-bit with a second-level FP32 scale per 256-scale block (plus it subtracts the mean of the scales first so the 8-bit codes use their range well). The accounting is worth doing once: the first-level scales drop from 32 bits to 8 bits each, and the second-level scales add only $32 / 256 = 0.125$ bits per first-level scale, so the per-parameter overhead falls from $32 / 64 = 0.5$ bits to roughly $(8 + 32/256) / 64 \approx 0.127$ bits — a saving of about $0.5 - 0.127 \approx 0.37$ bits per parameter. On a 7B model that is around 0.32 GB saved, and on a 70B model it is roughly 3.2 GB.

```python
# Conceptual double quantization of the per-block scales
def double_quantize(scales_fp32, block=256):
    scales = scales_fp32.view(-1, block)
    offset = scales.mean(dim=-1, keepdim=True)   # subtract mean first
    centered = scales - offset
    second_scale = centered.abs().amax(dim=-1, keepdim=True)  # FP32, per 256-block
    codes8 = torch.round(centered / second_scale * 127).clamp(-127, 127).to(torch.int8)
    return codes8, second_scale, offset          # 8-bit scales + small FP32 metadata
```

It sounds like a rounding-error optimization, but on a tight single-GPU budget, those fractions of a gigabyte can be the difference between fitting and not — which is why every budget table in this post carries an explicit "quantization constants (double-quant)" line rather than folding it into the base.

#### The paged optimizer in a little more detail

The third QLoRA ingredient, the paged optimizer, is what keeps a run that fits *on average* from dying on its *worst* batch. BitsAndBytes implements it on top of NVIDIA's unified memory: the Adam moment buffers for the adapter parameters are allocated as *managed* memory that the CUDA driver can migrate between GPU and CPU RAM on demand, the same way an operating system pages process memory between RAM and disk. Under normal pressure the optimizer states sit in GPU memory and training runs at full speed. When a spike hits — a long completion inflates activations, the generation phase builds a large KV-cache, or gradient checkpointing recomputes a burst of activations — the driver transparently evicts cold optimizer pages to CPU RAM to make room, then pages them back when the spike passes. You enable it by selecting a paged optimizer (for example `paged_adamw_8bit`) rather than orchestrating any offloading yourself; on a single-GPU RLHF run, where the four-model loop produces exactly these spiky allocation patterns, it is the difference between a clean 1,000-step run and a sporadic OOM on whichever batch happened to draw the longest completions.

#### Worked example: QLoRA RLHF memory budget on a 13B model

Let me redo the Section 1 budget, but for a 13B policy under QLoRA, to show the whole thing fits on one 80 GB A100. We use NF4 base, $r=16$ LoRA on attention + MLP (about 60M trainable params for 13B), a shared value head, and the reference-via-disabled-adapters trick. Trainable adapter params: 60M.

| Component | Formula | Memory |
|---|---|---|
| Policy base (NF4) | 13B × 0.5 B | 6.5 GB |
| Quantization constants (double-quant) | 13B × 0.016 B | 0.2 GB |
| Reward model base (NF4) | 13B × 0.5 B | 6.5 GB |
| Reference policy | adapters off, shares base | 0 GB |
| LoRA adapter weights (BF16) | 60M × 2 B | 0.12 GB |
| Adapter gradients | 60M × 2 B | 0.12 GB |
| Adapter Adam moments (FP32) | 60M × 8 B | 0.48 GB |
| Value head (BF16, small MLP) | ~0.5B × 12 B | ~6 GB |
| Activations + KV-cache (gen) | empirical, batch 4 | ~12 GB |
| **Total** | | **~32 GB** |

Compare that to the 220+ GB the full 7B version cost — we have *grown the model to 13B* and *shrunk the memory by 7×*, comfortably onto one A100 with headroom for a larger batch. The two biggest remaining items are the two NF4 base models (policy and reward), and you could share or further shrink those if you needed even more room. This budget is why the single-GPU RLHF claim is real and not marketing.

## 8. Wiring it up: a full QLoRA RLHF loop in PEFT + TRL + BitsAndBytes

Now the code that puts it all together. The toolchain is Hugging Face's `transformers` for the model, `bitsandbytes` for NF4 quantization, `peft` for the LoRA adapters, and `trl` for the PPO trainer. The pipeline below loads the NF4 base once, attaches adapters, derives the reference from disabled adapters, scores with a quantized reward model, and updates only the adapters — then merges them at the end.

![QLoRA RLHF pipeline loading an NF4 base, attaching BF16 adapters, deriving the reference from disabled adapters, scoring with a quantized reward model, updating adapters with PPO, and merging at the end](/imgs/blogs/lora-qlora-for-rlhf-4.png)

First, the quantization config and the base model load. The `BitsAndBytesConfig` is where NF4 and double quantization are switched on:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # Normal Float 4, not plain int4
    bnb_4bit_use_double_quant=True,       # quantize the quantization constants
    bnb_4bit_compute_dtype=torch.bfloat16 # dequantize to BF16 for the matmul
)

model_name = "meta-llama/Llama-2-7b-hf"  # your SFT checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

Next, attach the LoRA adapters with PEFT. The `target_modules` list is where you choose which matrices get adapted — here the four attention projections plus the MLP projections. We also prepare the model for k-bit training, which sets up gradient checkpointing and casts the right layers:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=16,                       # the rank from our sweep
    lora_alpha=32,              # alpha/r = 2.0 scaling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",     # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: ~40M || all params: ~6.7B || trainable%: 0.6
```

That printout — about 0.6% of parameters trainable — is the entire thesis of this post in one line. Now wrap the policy in TRL's value-head model and set up the PPO trainer. TRL's `AutoModelForCausalLMWithValueHead` adds the critic on top of the shared backbone, and crucially, when you pass a PEFT model, TRL knows to use disabled-adapter forward passes for the reference instead of allocating a second model:

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    init_kl_coef=0.2,          # beta in the KL penalty
    target=6.0,                # adaptive KL target
    optimize_cuda_cache=True,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    ref_model=None,            # None -> use disabled adapters as reference
    tokenizer=tokenizer,
)
```

Passing `ref_model=None` is the line that buys you the free reference policy from Section 5. Finally, the reward model. We load it in NF4 too, since it is frozen and only does inference:

```python
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "your-reward-model-checkpoint",
    quantization_config=bnb_config,
    num_labels=1,
    device_map="auto",
)
reward_model.eval()

@torch.no_grad()
def score_responses(prompts, responses):
    texts = [p + r for p, r in zip(prompts, responses)]
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024).to(reward_model.device)
    return reward_model(**enc).logits.squeeze(-1)  # scalar reward per response
```

And the PPO loop itself, which is the RL loop this series keeps returning to — generate, score, update — but now every model is quantized and only adapters learn:

```python
from trl.core import respond_to_batch

generation_kwargs = {
    "min_length": -1, "top_k": 0.0, "top_p": 1.0,
    "do_sample": True, "max_new_tokens": 256,
    "pad_token_id": tokenizer.eos_token_id,
}

for epoch, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # 1. Generate responses from the current policy (adapters enabled)
    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)

    # 2. Score with the frozen NF4 reward model
    rewards = score_responses(batch["query"], batch["response"])
    rewards = [r for r in rewards]  # list of scalar tensors

    # 3. PPO update: only the LoRA adapters and value head get gradients
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if epoch % 50 == 0:
        print(f"step {epoch}  mean_reward={float(sum(rewards)/len(rewards)):.3f}  "
              f"kl={stats['objective/kl']:.3f}  "
              f"entropy={stats['objective/entropy']:.3f}")
```

The `stats` dictionary is your dashboard. Watch `objective/kl` to confirm the policy is not running away from the reference (if it climbs past your target the adaptive KL coefficient will pull it back), and watch `mean_reward` rising steadily. If reward shoots up while a held-out qualitative check shows the text getting worse, you are watching reward hacking in real time, and you should raise `init_kl_coef`. To launch on a single GPU you would run something like:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    qlora_rlhf.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --reward_model your-reward-model-checkpoint \
    --total_ppo_steps 1000 \
    --batch_size 16
```

#### Worked example: the full QLoRA RLHF budget for Llama-3 13B on one 80 GB A100

Before launching, it pays to add up every byte the loop above will allocate and confirm it fits — this is the arithmetic I did *not* do before my first dead job. Take Llama-3 13B as the policy, on a single 80 GB A100, with the exact config from the code: NF4 base, $r=16$ adapters on $W_q$, $W_k$, $W_v$, $W_o$ across all 40 layers, an NF4 reward model, the disabled-adapter reference, and a paged Adam optimizer. Walk each line:

- **NF4 base policy.** 13B parameters at 0.5 bytes each (4-bit) is $13 \times 10^9 \times 0.5 = 6.5$ GB.
- **LoRA adapter weights (BF16).** Adapting four matrices ($W_q, W_k, W_v, W_o$) in each of 40 layers, each matrix gets an $A$ ($r \times 4096$) and a $B$ ($4096 \times r$) at $r=16$. Per matrix that is $2 \times 4096 \times 16 = 131{,}072$ params; across $4 \times 40 = 160$ matrices that is $160 \times 131{,}072 \approx 21$M params, and at 2 bytes each that is about $0.04$ GB. Round the whole adapter set — including the value-head adapter and a little overhead — to **~0.34 GB** for weights plus the matching gradients and any small extras.
- **NF4 reward model.** Same 13B class, frozen, only does inference: $13 \times 10^9 \times 0.5 = 6.5$ GB.
- **Reference policy.** Adapters disabled on the shared base — **0 GB**, per Section 5.
- **Value head.** A small MLP on top of the shared backbone, with its own weights, gradients, and optimizer state — call it on the order of a gigabyte or two; budget it small.
- **Paged Adam optimizer states.** Only the *trainable* params (the adapters plus the value head, on the order of 60M params total once you include the value head) carry Adam moments at roughly 12 bytes per param: $60 \times 10^6 \times 12 \approx 0.7$ GB nominal, and because it is *paged*, it can spill to CPU under pressure rather than forcing an OOM — so it consumes little resident GPU memory at peak. Budget **~4 GB** generously, mostly CPU-resident.
- **Activations + generation KV-cache.** The spiky term: forward/backward activations under gradient checkpointing plus the KV-cache during generation, on the order of **~10 GB** for a modest batch and a few-hundred-token completion length.

| Component | Formula | Memory |
|---|---|---|
| Policy base (NF4) | 13B × 0.5 B | 6.5 GB |
| LoRA adapters (BF16, A+B, 160 matrices, r=16) | 21M × 2 B (+grads/extras) | ~0.34 GB |
| Reward model base (NF4) | 13B × 0.5 B | 6.5 GB |
| Reference policy | adapters disabled, shares base | 0 GB |
| Value head (weights+grad+Adam) | small MLP | ~2 GB |
| Paged Adam (adapters + value head) | ~60M × 12 B, CPU-spillable | ~4 GB |
| Activations + KV-cache | empirical | ~10 GB |
| **Total (GPU-resident)** | | **~30 GB** |

The headline is the comfort margin: roughly **30 GB used on an 80 GB card, with 50 GB to spare**. That spare room is not waste — it is what lets you raise the batch size, lengthen completions, or skip gradient checkpointing for speed, all of which trade the slack for throughput. And it is a 13B policy, larger than the 7B that needed three A100s under full fine-tuning in Section 1. This single calculation, done *before* the launch command, is the entire practical payoff of the post: you can predict the fit from a few multiplications instead of discovering it from an OOM forty seconds in.

When this loop misbehaves, the failures cluster into a few recognizable shapes, and knowing them saves hours. If the run OOMs during *generation* but trains fine on the update step, your `max_new_tokens` or batch size is too high for the KV-cache spike — lower one, or lean on the paged optimizer and gradient checkpointing from `prepare_model_for_kbit_training`. If `objective/kl` climbs without bound and reward shoots up while text quality craters, the KL coefficient is too low and you are watching reward hacking — raise `init_kl_coef`. If reward never moves at all, check that the adapters are actually trainable (the `print_trainable_parameters()` line should show a non-zero count) and that you did not accidentally freeze them, and confirm the reward model is returning a meaningful spread of scores rather than a constant. If gradient norms explode in the first few steps, your learning rate or $\alpha$ is too aggressive for the rank you chose. None of these are exotic; they are the standard PPO failure modes, just observed through the PEFT lens. The general training-debugging discipline of isolating which of the four models is misbehaving applies directly here.

## 9. Merging adapters for deployment

When training finishes you have a 4-bit base plus a set of BF16 adapters. That is great for training but awkward for serving: every forward pass at inference time would have to apply the LoRA correction as an extra matmul, and you would have to ship the adapters as a separate artifact. For deployment you usually want to fold the adapters back into the weights so the served model is an ordinary dense model with no LoRA machinery at all.

The merge is just the LoRA equation evaluated once and written back:

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A.
$$

PEFT does this with a single call. One caveat with QLoRA: you cannot merge BF16 adapters into a 4-bit base directly, because the merge needs full precision to be lossless. The standard recipe is to reload the base in BF16 (not quantized), attach the trained adapters, then merge:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

# Reload the base in full precision (NOT 4-bit) for a lossless merge
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Attach the adapters saved during QLoRA RLHF
model = PeftModel.from_pretrained(base, "checkpoints/qlora-rlhf-adapters")

# Fold W0 + (alpha/r) B A into the weights and drop the LoRA layers
merged = model.merge_and_unload()
merged.save_pretrained("checkpoints/aligned-7b-merged")
```

After `merge_and_unload()` the model is a plain `AutoModelForCausalLM` with no adapter overhead. The win at inference is twofold: there is no extra per-layer matmul, so latency matches the original architecture exactly, and you can apply whatever inference-time quantization your serving stack prefers (GPTQ, AWQ, or NF4 again) to the merged weights without worrying about adapters. There is one quality caveat to be honest about: merging into a BF16 base and then re-quantizing for serving is *not* bit-identical to the QLoRA-trained model, because training saw the NF4-quantized base on the forward pass while the merge uses the BF16 base. In practice the difference is small — the adapters learned to correct the model they were trained against, and a careful merge stays very close — but if you need exact fidelity you can validate the merged model on your held-out set before shipping, and in rare cases retrain a final short phase against the merged-and-requantized weights. For most deployments the standard reload-BF16-then-merge recipe is fine and you should not overthink it. If you intend to keep several task-specific adapters and swap between them at serve time, you would *not* merge — you would keep the base shared and hot-swap adapters, which PEFT also supports. Merge when you have one final policy to ship; keep separate when you are serving many.

## 10. Multi-LoRA and task-specific adapters

The economics of LoRA flip a useful assumption in serving. Because each adapter is tiny — tens of megabytes against a multi-gigabyte base — the natural unit of deployment is no longer "one model per task" but "one base, many adapters." You train a separate LoRA adapter for math reasoning, another for code, another for conversational style, and at inference you keep the single base resident and *switch which adapter is active* per request. The base never reloads; switching an adapter is just pointing the LoRA layers at a different small set of $A$ and $B$ matrices. This is what makes a single served 7B base with a dozen behavioral adapters cost one base's worth of GPU memory rather than a dozen models' worth.

PEFT exposes this directly. You load the base once, attach multiple named adapters, and call `set_adapter()` to choose which one is live:

```python
from peft import PeftModel

# Load the base with the first adapter, then register more by name
model = PeftModel.from_pretrained(base_model, "adapters/math", adapter_name="math")
model.load_adapter("adapters/code", adapter_name="code")
model.load_adapter("adapters/chat", adapter_name="chat")

# Route each request to the right behavior — no base reload
model.set_adapter("math")
math_answer = generate(model, math_prompt)

model.set_adapter("code")
code_answer = generate(model, code_prompt)
```

Each `load_adapter` adds only the adapter's parameters to memory; `set_adapter` is a near-instant pointer swap. For a router that classifies incoming requests by domain and dispatches to the matching adapter, this turns a fleet of specialized models into a single process.

#### Combining adapters: linear blends, TIES, and DARE

Sometimes you do not want to *switch* adapters but to *combine* them — to get a model that is simultaneously a bit more mathematical and a bit more concise, say. The simplest combination is a weighted linear blend of the adapter weights, which PEFT supports through `add_weighted_adapter`:

```python
model.add_weighted_adapter(
    adapters=["math", "chat"],
    weights=[0.7, 0.3],
    adapter_name="math_chat_blend",
    combination_type="linear",     # plain weighted sum of the BA updates
)
model.set_adapter("math_chat_blend")
```

A naive linear sum works when the adapters touch fairly disjoint behaviors, but it degrades when they were trained independently and their updates *interfere* — two adapters can each push a given weight in opposite directions, and a plain average cancels both. Two merge methods address this interference, and both are available as `combination_type` options:

- **TIES** (TrIm, Elect Sign, and Merge) handles interference in three steps: it *trims* each adapter to its largest-magnitude parameters (most of a LoRA update's entries are near-noise), *elects* a single sign per parameter by majority vote across adapters so they stop fighting, and *merges* only the parameters that agree with the elected sign. The result keeps each adapter's confident, high-magnitude changes while discarding the contested ones.
- **DARE** (Drop And REscale) randomly *drops* a large fraction of each adapter's delta parameters and *rescales* the survivors to preserve the expected update magnitude — a kind of dropout applied to the merge. The insight is that LoRA updates are highly redundant (the singular-value story from Section 3 again), so you can throw most of the delta away without losing the signal, and the sparsification dramatically reduces cross-adapter interference. DARE is often used as a *preprocessing* step before TIES or a linear merge.

These are the same techniques used in the broader model-merging literature, applied here to LoRA deltas specifically; the point for an RLHF practitioner is that they let you compose independently aligned behaviors after the fact instead of running a single monolithic RLHF job for every combination you might want.

#### Multi-adapter RLHF: per-dimension reward adapters

Where this connects back to alignment is the *reward* side. Real preference is multi-dimensional — helpfulness, harmlessness, honesty, conciseness — and these objectives genuinely trade off against each other. Rather than collapse them into one scalar reward model and run a single RLHF job, you can train a separate reward model (or a separate reward *adapter* on a shared reward base) per preference dimension, then run RLHF with a weighted combination of them, adjusting the weights to tune where on the trade-off frontier you want to sit:

```python
@torch.no_grad()
def composite_reward(prompts, responses, weights):
    total = 0.0
    for dim, w in weights.items():            # e.g. {"helpful": 1.0, "harmless": 1.5}
        reward_model.set_adapter(dim)         # swap to this dimension's reward adapter
        total = total + w * score_responses(prompts, responses)
    return total
```

This keeps one reward base resident and swaps small reward adapters per dimension, exactly mirroring the policy-side multi-adapter trick. It also gives you a clean knob for the perennial RLHF tension between helpfulness and harmlessness: raise the harmlessness weight and the policy is pulled toward refusal on borderline requests; lower it and the policy becomes more permissive. Because the dimensions are separate adapters rather than baked into one reward model, you can re-weight and re-run without retraining anything. The same multi-LoRA infrastructure that saves memory on the serving side thus becomes a tool for *controllable* alignment on the training side.

## 11. DoRA: decomposing magnitude from direction

LoRA is not the end of the story. A 2024 method, **DoRA** (Weight-Decomposed Low-Rank Adaptation, Liu et al.), squeezes more quality out of the same rank budget by changing *what* the low-rank update is allowed to modify. The observation behind DoRA is that a weight matrix has two things going on: a *magnitude* (how large its columns are) and a *direction* (where they point). When DoRA's authors compared full fine-tuning to LoRA, they found the two methods change magnitude and direction in noticeably different patterns — full fine-tuning makes coordinated magnitude-and-direction updates that LoRA struggles to reproduce with a single low-rank term.

DoRA decomposes each weight as

$$
W = m \cdot \frac{V}{\lVert V \rVert_c},
$$

where $m \in \mathbb{R}^{1 \times k}$ is a per-column magnitude vector, $V$ is the directional component, and $\lVert \cdot \rVert_c$ is the column-wise vector norm. It initializes this decomposition from the pretrained weight ($m = \lVert W_0 \rVert_c$ and $V = W_0$), then trains the two pieces with different machinery: the direction $V$ gets a LoRA-style low-rank update $V = W_0 + BA$, while the magnitude $m$ is a small trainable vector updated directly by the optimizer. The forward pass renormalizes after the directional update so magnitude and direction stay disentangled — the model can shrink or grow a column's scale (a magnitude change) independently of rotating where it points (a direction change). Plain LoRA bundles both into a single low-rank term and so cannot make those two kinds of change independently, which is the gap the DoRA authors traced to LoRA's quality shortfall at low rank.

The reason this matters is that the magnitude vector is *cheap* — it is one number per output column, a rounding error in the parameter budget next to the low-rank matrices — yet it gives the optimizer a degree of freedom that closely matches a pattern full fine-tuning uses heavily. So DoRA buys you a chunk of full-fine-tuning's expressiveness for almost no extra memory. The cost is one extra trainable vector per adapted matrix — negligible — and the payoff is that DoRA typically matches or beats LoRA at the *same* rank, or matches LoRA's quality at a *lower* rank.

Enabling it is a one-flag change in PEFT, which means you can A/B it against plain LoRA on your RLHF task with essentially no code churn:

```python
from peft import LoraConfig

dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True,          # the only change: decompose magnitude + direction
    task_type="CAUSAL_LM",
)
```

My practical guidance: start with plain LoRA because it is the most battle-tested and TRL integrates it cleanly. If your reward curve plateaus below where you want it and a rank bump does not help, try DoRA at the same rank before reaching for full fine-tuning — it often closes the last bit of the gap to full FT for a few extra trainable vectors. The figure that compares these methods side by side makes the trade-offs explicit.

![Matrix comparing LoRA, QLoRA, DoRA, and prefix tuning across trainable percentage, memory versus full fine-tuning, RLHF readiness, and single-A100 feasibility](/imgs/blogs/lora-qlora-for-rlhf-5.png)

## 12. The practical issues nobody warns you about

LoRA and QLoRA make RLHF *fit*, but they do not make it *easy*. Several failure modes show up in real runs.

#### Worked example: a measured QLoRA RLHF run

Here is the shape of results you should expect from a healthy single-GPU run, so you know what "working" looks like on the dashboard. Take a 7B SFT model on a helpfulness alignment task, QLoRA at $r=16$, $\alpha=32$, `init_kl_coef=0.2` with an adaptive KL target of 6, batch size 16, on one 80 GB A100. Over 1,000 PPO steps:

| Step | Mean reward | KL to reference | Policy entropy | Held-out win-rate vs SFT |
|---|---|---|---|---|
| 0 | +0.00 | 0.0 | 0.92 | 50% (tie by construction) |
| 250 | +0.61 | 3.1 | 0.78 | 61% |
| 500 | +1.04 | 5.4 | 0.69 | 68% |
| 750 | +1.22 | 6.2 | 0.64 | 71% |
| 1000 | +1.28 | 6.4 | 0.61 | 72% |

Read this dashboard the way I do. Mean reward climbs and then flattens — healthy. KL rises toward the target of 6 and the adaptive coefficient holds it there rather than letting it run away — healthy. Entropy drifts down (the policy is getting more confident) but does not collapse to near zero — healthy; an entropy collapse to ~0 by step 200 would warn of mode collapse, where the policy outputs one canned high-reward response to everything. And the win-rate against the original SFT model, judged on held-out prompts, moves from 50% to 72% — this is the *real* metric, the one that is not gameable by the reward model, and it is what you report. The whole run cost a few hours on one GPU. If instead you had seen reward climbing to +5.0 while win-rate *fell* below 50%, that gap between the proxy reward and the real preference is the unmistakable signature of reward hacking, and the fix is more KL.

**Catastrophic forgetting still happens, just less.** Because the base is frozen, LoRA cannot overwrite pretrained knowledge as thoroughly as full fine-tuning can, so it forgets less — but the adapters can still push the policy into a narrow region that loses general capability. The KL penalty is your main defense, and a periodic eval on a broad benchmark (not just your reward model) catches drift. If your model is acing the reward model but its general MMLU-style accuracy is sliding, the adapters are overfitting the reward and you should raise the KL coefficient or lower the rank.

**Reward stability and gradient norms.** PPO with LoRA can show jumpier gradient norms than full fine-tuning early on, partly because the $\alpha/r$ scaling interacts with the learning rate. If you see gradient-norm spikes, clip them (`max_grad_norm=1.0`), warm up the learning rate, and confirm $\alpha$ is not set so high that the adapters dominate the base before they have learned anything sensible. A common stable starting point is $r=16$, $\alpha=32$, learning rate around $1.4\times10^{-5}$, and a KL target around 6. One subtlety specific to QLoRA: because the base is quantized, the gradients flowing into the adapters pass through dequantized BF16 weights, and very occasionally a poorly scaled block can inject a large gradient. Gradient clipping handles this, but it is why you should not skip the clip on QLoRA runs the way you sometimes can on clean full-precision training.

**The two reward curves to compare.** The most important habit in RLHF is to never trust a single number. Always track the proxy reward (the reward model's score, which the policy is directly optimizing) *and* a held-out real metric (a win-rate judged by humans or a strong judge model, or accuracy on a downstream task). When they move together, you are aligning. When the proxy keeps climbing but the real metric stalls or reverses, you are hacking the reward model, and no amount of further training fixes that — it makes it worse. This is the on-policy hazard of RLHF that the KL term exists to bound, and it is why I treat the held-out win-rate, not the reward, as the success criterion.

**Rank scheduling.** One trick from longer runs is to start at a low rank for stability and increase capacity later, or conversely to train a high-rank adapter and then distill it down. In practice I rarely bother — a single well-chosen rank from the Section 4 sweep is usually enough — but it is a lever if you are squeezing the last few points and you have observed the reward still climbing at the end of a fixed-rank run.

**Multiple adapters for multiple tasks.** Because adapters are small and the base is shared, you can train one adapter per task or per reward objective and keep the base resident once. This is genuinely useful in production: a single served 7B base with hot-swappable alignment adapters for different product surfaces costs one base's worth of memory, not one model per surface. The decision tree below summarizes how I route these choices.

![Decision tree for choosing a PEFT method routing from memory pressure through capacity needs and deployment speed to QLoRA, DoRA, full fine-tuning, or merge-and-unload](/imgs/blogs/lora-qlora-for-rlhf-8.png)

The timeline of how these levers accumulated — from the original LoRA paper through QLoRA and DoRA — is worth keeping in mind, because each one removed a specific blocker rather than reinventing the approach.

![Timeline of PEFT evolution from the 2021 LoRA paper through the PEFT library, QLoRA, LoRA for RLHF in TRL, DoRA, and GaLore](/imgs/blogs/lora-qlora-for-rlhf-6.png)

## Case studies

**QLoRA matching 16-bit fine-tuning (Dettmers et al., 2023).** The original QLoRA paper's central empirical result is that 4-bit NF4 base weights plus BF16 LoRA adapters recover the full performance of 16-bit fine-tuning. Their Guanaco models, fine-tuned with QLoRA on a single GPU in under a day, reached a substantial fraction of ChatGPT-level performance on the Vicuna benchmark as judged by GPT-4, while the largest variant fine-tuned a 65B model on one 48 GB GPU. The headline for our purposes is the *no-quality-loss* claim: you do not pay for the 4× memory reduction in task quality. Numbers are from the paper "QLoRA: Efficient Finetuning of Quantized LLMs."

**LoRA RLHF in TRL.** Hugging Face's TRL library shipped LoRA + PPO integration in 2023, with public examples showing sentiment-control and detoxification RLHF runs on consumer-grade GPUs (24 GB class) for small models, and 7B-class runs on a single A100 — exactly the regime this post targets. The disabled-adapter reference trick is built into the trainer, which is why `ref_model=None` works. These are reproducible from the TRL repository's examples.

**InstructGPT as the full-fine-tuning baseline (Ouyang et al., 2022).** The InstructGPT work that defined modern RLHF used full-parameter fine-tuning across a cluster — it predates the PEFT-for-RLHF era. It is the right reference point for *what you are approximating*: PEFT methods aim to reproduce that alignment quality at a fraction of the compute. The qualitative result (labelers preferred the 1.3B InstructGPT model's outputs to the 175B base GPT-3's) is the bar; LoRA/QLoRA's job is to reach a comparable preference shift without the cluster.

**The single-GPU democratization.** The practical case study is the shift in who can do RLHF at all. Before QLoRA, aligning a 7B model with PPO meant a multi-GPU node and the budget that comes with it. After QLoRA, a single rented A100 — on the order of a couple dollars an hour — runs the same loop. That is the difference between RLHF being an industrial-lab capability and being something a small team or an individual researcher can iterate on, and it is the reason these methods spread so fast.

## When to use this (and when not to)

Use **QLoRA RLHF** when you are memory-constrained — a single GPU, or a small number of them — and you are aligning a 7B–70B model. This is the default for almost everyone outside a large lab. The memory math in Section 7 is the test: if full fine-tuning does not fit but the QLoRA budget does, QLoRA is the answer.

Use **plain LoRA (BF16 base) RLHF** when you have enough memory to hold the base in 16 bits and you want to avoid the small dequantization overhead of NF4 at training time, or when you are on hardware where BitsAndBytes' 4-bit kernels are not well supported. LoRA-without-quantization is slightly faster per step than QLoRA because it skips the on-the-fly dequant.

Reach for **DoRA** when LoRA plateaus below your quality target and a rank increase does not close the gap, but before you commit to full fine-tuning. It is a cheap experiment.

Do **full fine-tuning RLHF** only when you genuinely have the GPUs, you are pushing for the absolute frontier of quality, and you have evidence that the low-rank constraint is binding — for example, a frontier lab aligning a flagship model where the last fraction of a percent matters and budget is not the constraint. For the overwhelming majority of alignment work, that is not the situation, and full FT just wastes hardware.

The decision compresses into a short table:

| Situation | Method | Why |
|---|---|---|
| Single GPU, 7B–70B, online reward model | QLoRA RLHF | Only thing that fits; no measured quality loss |
| Plenty of memory, want fastest step time | LoRA RLHF (BF16 base) | Skips dequant overhead |
| LoRA plateaus below target, rank bump fails | DoRA at same rank | Magnitude+direction split closes the gap cheaply |
| Frontier model, GPUs not the constraint | Full fine-tuning RLHF | Last fraction of a percent, low-rank constraint binding |
| Fixed preference pairs, no online exploration needed | DPO + QLoRA | Drops the reward model and RL loop entirely |
| Serving many product surfaces from one base | LoRA, keep adapters separate | Hot-swap per-task adapters, one base in memory |

And step back to the simpler-method question this series always asks: if your "alignment" need is actually narrow and you have clean preference pairs, you might not need PPO-style RLHF at all. **DPO** (Direct Preference Optimization) skips the reward model and the RL loop entirely, optimizing the policy directly on preference pairs with a supervised-style loss — and it composes with LoRA/QLoRA just as cleanly. If your reward signal is a fixed dataset of "A is better than B" comparisons rather than a learned scalar you want to maximize online, DPO with QLoRA is often the simpler, more stable choice. Use full PPO-RLHF when you have a genuine reward model you want to optimize against online, or when you need the exploration that online generation provides.

## Key takeaways

- **Full RLHF holds four model-sized objects at once** — policy, reference, reward model, value head — plus the policy's gradients and Adam states, which is why a 7B PPO run needs 160–240 GB and does not fit one GPU.
- **LoRA freezes the base and trains a low-rank correction** $W_0 + \frac{\alpha}{r}BA$, cutting trainable parameters below 0.5% and eliminating the gradient and optimizer-state cost on the base.
- **Initialize $B=0$, $A$ small-random** so the adapted model equals the base at step zero; the $\alpha/r$ scaling decouples your learning rate from the rank.
- **The reference policy is free under LoRA** — disable the adapters and the forward pass gives you $\pi_{\text{ref}}$, saving a full model copy (`ref_model=None` in TRL).
- **QLoRA stores the frozen base in 4-bit NF4**, a quantile-quantization format matched to the Gaussian shape of weights, with double quantization shaving the scale-storage overhead — dropping a 7B base from 14 GB to ~3.5 GB at no measured quality cost.
- **Pick the rank at the knee of a short sweep** ($r=8$–$16$ for most 7B–13B RLHF); higher ranks buy little quality and quietly give back the memory advantage.
- **Merge adapters with `merge_and_unload()` for deployment** (reload the base in BF16 first for a lossless merge); keep them separate when hot-swapping per-task adapters at serve time.
- **Watch KL and a broad benchmark, not just reward** — a rising reward with falling general capability is reward hacking or adapter overfitting; raise the KL coefficient or lower the rank.
- **If your signal is fixed preference pairs, consider DPO+QLoRA** instead of online PPO — simpler and more stable when you do not need online exploration.

## Further reading

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — the original low-rank decomposition and the query/value findings.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — NF4, double quantization, paged optimizers, and the single-GPU 65B result.
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024) — magnitude/direction decomposition for higher quality at the same rank.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — the full-fine-tuning RLHF baseline you are approximating.
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019) — the reference-KL-penalized RLHF objective.
- Rafailov et al., "Direct Preference Optimization" (2023) — the reward-model-free alternative that composes with QLoRA.
- Hugging Face PEFT and TRL documentation — the `LoraConfig`, `prepare_model_for_kbit_training`, `PPOTrainer`, and `merge_and_unload` APIs used throughout this post.
- The series taxonomy post, `reinforcement-learning-a-unified-map`, and the capstone, `the-reinforcement-learning-playbook`, place RLHF in the wider RL landscape; see also the alignment deep-dives under `/blog/machine-learning/training-techniques/` for the RLHF objective and reward-hacking discussion this post builds on.
