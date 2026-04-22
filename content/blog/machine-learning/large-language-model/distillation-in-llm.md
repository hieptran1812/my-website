---
title: "Knowledge Distillation in LLMs: A Detailed Guide with Case Studies"
publishDate: "2026-04-22"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "distillation", "knowledge-distillation", "model-compression", "training", "deep-learning", "DistilBERT", "Gemma", "Llama", "DeepSeek"]
date: "2026-04-22"
author: "Hiep Tran"
featured: false
aiGenerated: true
image: "/imgs/blogs/distillation-in-llm-01-teacher-student.png"
excerpt: "A deep dive into knowledge distillation for large language models — the theory, the loss functions, the training recipes, and concrete case studies: DistilBERT, Alpaca/Vicuna, Gemma 2, Llama 3.2, MiniLLM, and DeepSeek-R1-Distill."
---

## Why Distillation Matters

Modern frontier LLMs are huge. GPT-4-class models, Claude, Gemini Ultra, and DeepSeek-V3 range from hundreds of billions to trillions of parameters. They are powerful but expensive to serve, slow to respond, and impossible to run on a laptop or phone.

A 7B or 3B model, on the other hand, is cheap, fast, and fits on a single consumer GPU — but out of the box it is meaningfully weaker at reasoning, instruction following, and nuanced tasks.

**Knowledge distillation bridges the gap.** The core idea is one sentence:

> Train a small "student" model to imitate a large "teacher" model.

Done well, distillation lets a student keep most of the teacher's quality while being **5×–20× smaller and faster**. It is behind DistilBERT, TinyBERT, MobileBERT, Alpaca, Vicuna, TinyLlama, Phi, Gemma 2, Llama 3.2 1B/3B, MiniLLM, and the DeepSeek-R1-Distill series.

This guide walks through the full picture in depth: the intuition, the math (with every loss written out), the major training techniques, and concrete case studies with real hyperparameters and results.

## Section 1 — The Core Intuition

### The "exam" analogy

Imagine two students studying for an exam.

- **Student A** only sees the answer key: "Question 1 = B, Question 2 = D, …". Right or wrong, nothing else.
- **Student B** gets the teacher's full reasoning: "The answer is B, but C is also plausible, D is clearly wrong, and A is a common mistake because…"

Student B learns far more per example because the teacher reveals *the shape of their uncertainty* — not just the final answer.

Supervised next-token training of a small LLM is Student A: it only sees one "correct" token per position. Distillation turns the student into Student B: it sees the **full probability distribution** the teacher would produce. That distribution encodes grammar, semantics, world knowledge, and reasoning all at once.

### What "dark knowledge" actually is

Take a tiny concrete example. Suppose the input prompt is:

```
The cat sat on the ___
```

A small model trained with hard labels sees only "mat" as correct. Every other token gets zero probability in its target, even though "rug", "floor", "couch" are perfectly reasonable continuations and "rocket" is not.

A teacher model might output something like:

```
mat    0.42
rug    0.18
floor  0.11
couch  0.07
chair  0.05
sofa   0.04
bed    0.02
...
rocket 1e-9
```

Training the student to match *this entire distribution* teaches it:
- "mat" is most likely
- "rug/floor/couch/chair" are semantically related synonyms
- "rocket" is wildly implausible

That extra structural signal — the **relative ranking and spacing of non-top tokens** — is "dark knowledge." It is what gives distillation its enormous sample efficiency.

![Teacher to Student imitation and the dark knowledge encoded in soft probabilities vs. a one-hot hard label](/imgs/blogs/distillation-in-llm-01-teacher-student.png)

## Section 2 — The Classical Formulation (Hinton et al., 2015)

The paper that started it all introduced three things: **soft targets**, **temperature**, and the **combined loss**.

### Step 1: Soft targets

For a given input, let the teacher output logits `z^T = (z_1^T, ..., z_V^T)` over a vocabulary of size `V`. The standard softmax gives hard-ish probabilities:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

For a well-trained teacher, these can be extremely peaked — 0.99 on one token, 0.01 split over all others. That throws away most of the dark knowledge.

### Step 2: Temperature

Introduce a temperature `T ≥ 1` *before* the softmax:

```
p_i^T = exp(z_i / T) / Σ_j exp(z_j / T)
```

`T = 1` is the normal softmax. `T > 1` smooths the distribution so the relative magnitudes of smaller logits become visible. Typical values in practice: **`T = 2` to `T = 5`**. DistilBERT used `T = 2`. Some reasoning distillation pipelines use `T = 1` because they want the student to match the teacher's greedy behavior exactly.

### Step 3: The combined loss

The student is trained on a weighted combination of two losses:

```
L = α · L_CE(student_T=1, hard_label)
  + (1 − α) · T² · L_KL(student_T || teacher_T)
```

- `L_CE` — standard cross-entropy against the ground-truth token (keeps the student grounded in real data).
- `L_KL` — KL divergence between the *softened* student and teacher distributions.
- `T²` — compensates for the fact that softening shrinks gradient magnitudes by roughly `1/T²`.
- `α` — blending weight, often `0.1`–`0.5` (more weight on the distillation term than on the hard labels).

Written out more explicitly, the KL term is:

```
L_KL = Σ_i p_i^T(teacher) · [ log p_i^T(teacher) − log p_i^T(student) ]
```

And in actual code:

```python
# logits_t, logits_s: [batch, seq_len, vocab]
T = 2.0
alpha = 0.5

soft_t = F.log_softmax(logits_t / T, dim=-1)   # teacher log-probs at T
soft_s = F.log_softmax(logits_s / T, dim=-1)   # student log-probs at T

# KL(teacher || student), averaged over tokens
kl = F.kl_div(soft_s, soft_t, reduction="batchmean", log_target=True) * (T * T)

# Hard-label cross-entropy at T=1
ce = F.cross_entropy(logits_s.view(-1, V), labels.view(-1))

loss = alpha * ce + (1 - alpha) * kl
```

This is the recipe behind the original wave of small transformers (DistilBERT, TinyBERT, MobileBERT) and is still the backbone of modern white-box LLM distillation.

### Step 4: Understanding the KL Term in Depth

The KL divergence is the mathematical engine of distillation — it quantifies how far the student's distribution is from the teacher's, and its gradient tells the student exactly how to move each token's probability mass. Let's unpack it rigorously.

#### Definition

Given two probability distributions $p$ and $q$ over the same vocabulary of size $V$, the **Kullback–Leibler divergence** from $q$ to $p$ is:

$$
\mathrm{KL}(p \,\|\, q) \;=\; \sum_{i=1}^{V} p_i \, \log \frac{p_i}{q_i}
\;=\; \underbrace{\sum_i p_i \log p_i}_{-H(p)} \;-\; \underbrace{\sum_i p_i \log q_i}_{-H(p, q)}
$$

Two useful identities follow immediately:

- $\mathrm{KL}(p \,\|\, q) \geq 0$, with equality iff $p = q$ (Gibbs' inequality).
- $\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p)$ — it's the *excess* cross-entropy of using $q$ to encode $p$.

KL is **not symmetric**: $\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)$ in general. This asymmetry is the whole point of the "forward vs. reverse KL" debate in Section 4.2.

#### Why KL (and Not Just MSE) on Logits?

You could, in principle, minimize $\lVert z^s - z^t \rVert_2^2$ on logits directly. People have tried; it works less well. KL on softmaxed distributions is preferred because:

1. **Scale invariance.** Softmax is shift-invariant ($\mathrm{softmax}(z + c) = \mathrm{softmax}(z)$), so the student isn't penalized for an overall logit offset the teacher happens to carry.
2. **Probability-space geometry.** KL weights errors by $p_i^t$ — the student is strongly pushed to match tokens the teacher considers likely, and barely pushed on tokens the teacher considers impossible. MSE on logits gives equal weight everywhere.
3. **Information-theoretic interpretation.** KL measures the expected extra bits per token to encode the teacher's distribution under the student's model. Minimizing it has a clean coding-theory meaning.

#### Derivation: Distillation Loss with Temperature

Let $z^t, z^s \in \mathbb{R}^V$ be the teacher and student logits at a single token position. Define the **temperature-softened** probabilities:

$$
p_i^{t,T} \;=\; \frac{\exp(z_i^t / T)}{\sum_{j} \exp(z_j^t / T)}
\qquad
p_i^{s,T} \;=\; \frac{\exp(z_i^s / T)}{\sum_{j} \exp(z_j^s / T)}
$$

The per-token distillation loss — **forward KL** from teacher $\to$ student — is:

$$
\mathcal{L}_{\mathrm{KL}} \;=\; \mathrm{KL}\!\left(p^{t,T} \,\|\, p^{s,T}\right)
\;=\; \sum_{i=1}^{V} p_i^{t,T} \, \log \frac{p_i^{t,T}}{p_i^{s,T}}
$$

Since $\sum_i p_i^{t,T} \log p_i^{t,T}$ is a constant with respect to the student's parameters $\theta$, minimizing the KL above is equivalent to minimizing the **soft cross-entropy**:

$$
\mathcal{L}_{\mathrm{soft\text{-}CE}} \;=\; -\sum_{i=1}^{V} p_i^{t,T} \, \log p_i^{s,T}
$$

This is exactly what the PyTorch code computes — `F.kl_div(log_softmax_student, log_softmax_teacher)` is the soft cross-entropy up to an additive constant.

#### Where the $T^2$ Comes From

Take the gradient of $\mathcal{L}_{\mathrm{KL}}$ with respect to a student logit $z_k^s$. Using the softmax–cross-entropy identity:

$$
\frac{\partial \mathcal{L}_{\mathrm{KL}}}{\partial z_k^s}
\;=\; \frac{1}{T}\bigl(p_k^{s,T} - p_k^{t,T}\bigr)
$$

The $1/T$ factor arises from the chain rule through the softened softmax. For moderate logit differences, a Taylor expansion (Hinton et al., 2015) gives:

$$
p_k^{s,T} - p_k^{t,T} \;\approx\; \frac{z_k^s - z_k^t}{T \cdot V}
\quad\Longrightarrow\quad
\frac{\partial \mathcal{L}_{\mathrm{KL}}}{\partial z_k^s} \;\approx\; \frac{z_k^s - z_k^t}{T^2 \cdot V}
$$

So the gradient magnitude scales like $1/T^2$. Without correction, a larger temperature would silently shrink the distillation loss's contribution to the total gradient. Multiplying the loss by $T^2$ restores the gradient to a $T$-independent scale, so $\alpha$ remains a meaningful blending weight across temperatures:

$$
\mathcal{L}_{\mathrm{total}} \;=\; \alpha \, \mathcal{L}_{\mathrm{CE}}(z^s, y) \;+\; (1 - \alpha) \, T^2 \, \mathcal{L}_{\mathrm{KL}}\!\left(p^{t,T} \,\|\, p^{s,T}\right)
$$

**Practical upshot:** if you change $T$ and forget the $T^2$, your effective distillation weight silently changes by a factor of $T^2$. Easy bug, hard to notice — quality tanks, you blame the data.

#### Forward KL vs. Reverse KL

The distillation literature distinguishes two directions:

$$
\mathrm{KL}_{\mathrm{forward}} \;=\; \mathrm{KL}\!\left(p^t \,\|\, p^s\right) \;=\; \sum_i p_i^t \log \frac{p_i^t}{p_i^s}
$$

$$
\mathrm{KL}_{\mathrm{reverse}} \;=\; \mathrm{KL}\!\left(p^s \,\|\, p^t\right) \;=\; \sum_i p_i^s \log \frac{p_i^s}{p_i^t}
$$

Their behaviors differ sharply:

- **Forward KL is mode-covering.** The penalty $p_i^t \log(p_i^t / p_i^s)$ explodes whenever $p_i^t > 0$ but $p_i^s \approx 0$. The student is forced to put probability mass on *every* token the teacher considers plausible — even low-probability tails. Great for classification, harmful for generation (the student spreads mass too thin and produces mushy text).
- **Reverse KL is mode-seeking.** The penalty $p_i^s \log(p_i^s / p_i^t)$ explodes whenever $p_i^s > 0$ but $p_i^t \approx 0$. The student is punished for putting mass where the teacher doesn't, but *not* punished for ignoring modes the teacher covers. The student concentrates on a subset of the teacher's modes — which is exactly what you want for coherent generation.

This is why **MiniLLM** (Section 4.2) replaces forward KL with reverse KL and sees big gains on instruction-following. The price: reverse KL requires sampling from the student and is trained via policy gradient, making it costlier and less stable.

#### Per-Token Aggregation

Everything above is for a single token position. The full sequence-level distillation loss averages (or sums) over positions. For a sequence of length $L$ and a batch of size $B$:

$$
\mathcal{L}_{\mathrm{KL}}^{\text{batch}} \;=\; \frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{\ell=1}^{L} \mathrm{KL}\!\left(p^{t,T}_{b,\ell} \,\|\, p^{s,T}_{b,\ell}\right)
$$

In PyTorch this corresponds to `reduction="batchmean"`. **Do not use** `reduction="mean"` — it divides by $B \cdot L \cdot V$ instead of $B \cdot L$ and silently attenuates the loss by a factor of $V$ (typically 32K–256K for LLMs), which will make your distillation run look like nothing is happening.

## Section 3 — The Main Flavors of LLM Distillation

In practice, LLM distillation splits into four patterns. Most production systems mix several.

![The four flavors of LLM distillation: logit (white-box), response (black-box), hidden-state, and chain-of-thought](/imgs/blogs/distillation-in-llm-02-four-flavors.png)

### 3.1 Logit / Soft-Label Distillation (white-box)

The classical Hinton setup applied at *every token position*. For each input position, the teacher produces a distribution over the vocabulary; the student matches it via KL.

**Requirements:** teacher and student must be running forward passes you control — i.e. open weights.

**Storage problem:** a single token position has a vocabulary distribution of size `V` (often 32K–256K). Storing full distributions over billions of training tokens is infeasible. Two fixes:
1. **Top-K caching.** Keep only the top `K` logits (K = 50 to 1000) per position, renormalize. Captures > 99% of probability mass.
2. **Online distillation.** Run the teacher forward pass alongside the student during training; never store anything. Costs extra GPU memory and FLOPs but avoids the I/O problem. This is how Gemma 2 2B and most pretraining-scale distillation works today.

**Used by:** DistilBERT, TinyBERT, MiniLM, Gemma 2 2B/9B, Llama 3.2 1B/3B.

**Online logit distillation — minimal training loop:**

```python
import torch, torch.nn.functional as F

def logit_kd_step(student, teacher, batch, optimizer, T=2.0, alpha=0.5):
    input_ids, labels = batch["input_ids"], batch["labels"]

    # Teacher: frozen, no grad
    with torch.no_grad():
        t_logits = teacher(input_ids).logits      # [B, L, V]

    # Student: trainable
    s_logits = student(input_ids).logits           # [B, L, V]

    # Temperature-softened distributions
    log_p_s = F.log_softmax(s_logits / T, dim=-1)
    log_p_t = F.log_softmax(t_logits / T, dim=-1)

    # Forward KL(teacher || student), per token, average over batch+seq
    kl = F.kl_div(log_p_s, log_p_t, reduction="batchmean", log_target=True)
    kl = kl * (T * T)                              # <-- the T^2 correction

    # Hard-label CE at T=1 (ignore pad tokens via label=-100)
    ce = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    loss = alpha * ce + (1 - alpha) * kl
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "ce": ce.item(), "kl": kl.item()}
```

**Top-K cached variant** — when running the teacher online is too expensive, precompute top-K logits once and replay them:

```python
# One-off teacher pass over the whole corpus (days on big clusters)
def cache_topk(teacher, loader, K=256, out_path="teacher_topk.pt"):
    cache = []
    for batch in loader:
        with torch.no_grad():
            logits = teacher(batch["input_ids"]).logits   # [B, L, V]
        vals, idx = logits.topk(K, dim=-1)                # [B, L, K]
        cache.append({"input_ids": batch["input_ids"].cpu(),
                      "topk_vals": vals.cpu(), "topk_idx": idx.cpu()})
    torch.save(cache, out_path)

# Student training on the cached top-K
def topk_kd_loss(s_logits, topk_vals, topk_idx, T=2.0):
    # Gather the K student logits at the teacher's top-K positions
    s_top = s_logits.gather(-1, topk_idx) / T              # [B, L, K]
    t_top = topk_vals / T
    # Renormalize: log-softmax over K only (not full V)
    log_p_s = s_top - torch.logsumexp(s_top, dim=-1, keepdim=True)
    log_p_t = t_top - torch.logsumexp(t_top, dim=-1, keepdim=True)
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(-1).mean() * (T * T)
    return kl
```

### 3.2 Sequence / Response Distillation (black-box)

Generate `(prompt, teacher_response)` pairs and fine-tune the student with standard next-token cross-entropy:

```
L_SFT = − Σ_t log p_student(y_t | prompt, y_<t)
```

where `y` is the teacher's full response. No access to teacher internals required.

**This is the most widely-used form of LLM distillation in the open-source community.** Alpaca, Vicuna, WizardLM, OpenOrca, UltraChat, Nous-Hermes, and the entire SFT-via-GPT-4 ecosystem are built this way.

**A strong upgrade is rejection sampling distillation:**
1. Generate `N = 4` to `64` candidate responses per prompt.
2. Score them — by a reward model, a verifier (for math/code), or by asking the teacher to self-critique.
3. Keep only the best-scoring candidate(s).
4. Train the student on the filtered set.

Llama 2 and Llama 3 both used rejection-sampling against their own larger models to bootstrap instruction-tuning data — essentially self-distillation plus filtering.

**Minimal response-distillation pipeline:**

```python
# Step 1: generate teacher responses (API-based teacher, no weights needed)
from openai import OpenAI
client = OpenAI()

def generate_teacher_responses(prompts, model="gpt-4o", n=8, temperature=0.7):
    dataset = []
    for prompt in prompts:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=n, temperature=temperature, max_tokens=2048,
        )
        for choice in resp.choices:
            dataset.append({"prompt": prompt, "response": choice.message.content})
    return dataset

# Step 2: rejection sampling (math example with automatic verifier)
def verify_math(prompt, response, gold_answer):
    # Extract "\boxed{...}" or final number; compare to gold
    import re
    m = re.search(r"\\boxed\{([^}]+)\}", response)
    return bool(m) and m.group(1).strip() == gold_answer.strip()

def rejection_filter(dataset, gold):
    kept = []
    by_prompt = {}
    for ex in dataset:
        if verify_math(ex["prompt"], ex["response"], gold[ex["prompt"]]):
            by_prompt.setdefault(ex["prompt"], []).append(ex)
    # Keep shortest correct response per prompt
    for prompt, cands in by_prompt.items():
        kept.append(min(cands, key=lambda x: len(x["response"])))
    return kept

# Step 3: standard SFT on the filtered (prompt, response) pairs
def sft_loss(student, tokenizer, prompt, response):
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": response}],
        tokenize=False,
    )
    ids = tokenizer(full, return_tensors="pt").input_ids
    labels = ids.clone()
    # Mask out the prompt tokens (loss only on assistant response)
    prompt_len = len(tokenizer(prompt).input_ids)
    labels[:, :prompt_len] = -100
    return student(input_ids=ids, labels=labels).loss
```

### 3.3 Feature / Hidden-State Distillation

Match the student's internal representations to the teacher's — hidden states, attention maps, embeddings — not just the output logits.

For hidden states at layer `l`:

```
L_hidden = || W · h_student^{l_s} − h_teacher^{l_t} ||²
```

`W` is a learned linear projection to handle dimension mismatch between teacher and student. The student layer `l_s` maps to a teacher layer `l_t` via some schedule (e.g. uniform, or "every k-th").

For attention maps:

```
L_attn = MSE(A_student, A_teacher)
```

where `A` is the attention probability matrix.

**Used heavily in TinyBERT, MobileBERT, MiniLM.** Effective when teacher and student share architectural structure, especially when the student is *initialized* from the teacher via layer pruning — so the representations already align before training starts.

**Hidden-state + attention matching loop:**

```python
import torch, torch.nn.functional as F
import torch.nn as nn

class HiddenStateKD(nn.Module):
    """Projects student hidden states to teacher dim for MSE matching."""
    def __init__(self, d_student, d_teacher, num_pairs):
        super().__init__()
        # One projection per (student_layer, teacher_layer) pair
        self.projs = nn.ModuleList([
            nn.Linear(d_student, d_teacher) for _ in range(num_pairs)
        ])

    def forward(self, s_hiddens, t_hiddens, layer_map):
        # layer_map: list of (student_layer_idx, teacher_layer_idx) pairs
        loss = 0.0
        for i, (ls, lt) in enumerate(layer_map):
            h_s = self.projs[i](s_hiddens[ls])         # [B, L, d_t]
            h_t = t_hiddens[lt]                         # [B, L, d_t]
            loss = loss + F.mse_loss(h_s, h_t)
        return loss / len(layer_map)

def feature_kd_step(student, teacher, proj, batch, optimizer,
                    lam_logit=1.0, lam_hidden=1.0, lam_attn=1.0):
    input_ids = batch["input_ids"]

    with torch.no_grad():
        t_out = teacher(input_ids, output_hidden_states=True,
                        output_attentions=True)
    s_out = student(input_ids, output_hidden_states=True,
                    output_attentions=True)

    # Layer alignment: student 6-layer -> teacher 12-layer = every other
    layer_map = [(i, 2 * i + 1) for i in range(6)]

    # 1. Logit KD (Section 3.1 loss)
    log_p_s = F.log_softmax(s_out.logits / 2.0, dim=-1)
    log_p_t = F.log_softmax(t_out.logits / 2.0, dim=-1)
    l_logit = F.kl_div(log_p_s, log_p_t, reduction="batchmean",
                       log_target=True) * 4.0

    # 2. Hidden-state MSE with learned projection
    l_hidden = proj(s_out.hidden_states, t_out.hidden_states, layer_map)

    # 3. Attention map MSE
    l_attn = 0.0
    for ls, lt in layer_map:
        A_s = s_out.attentions[ls]      # [B, H, L, L]
        A_t = t_out.attentions[lt]
        l_attn = l_attn + F.mse_loss(A_s, A_t)
    l_attn = l_attn / len(layer_map)

    loss = lam_logit * l_logit + lam_hidden * l_hidden + lam_attn * l_attn
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"logit": l_logit.item(), "hidden": l_hidden.item(),
            "attn": l_attn.item()}
```

### 3.4 Reasoning / Chain-of-Thought Distillation

A 2024–2025 development. Instead of distilling only answers, distill the teacher's entire **reasoning trace**: chain of thought, self-reflection, backtracking, verification.

The student's training data is `(question, full_CoT, final_answer)`. Cross-entropy is computed over the *entire* CoT, not just the answer tokens. This forces the student to internalize the teacher's reasoning *procedure*, not just its conclusions.

**Most important example:** DeepSeek-R1-Distill (covered below). Also used for Phi-4, Qwen2.5-Math-Instruct, and several o1-style open replications.

**CoT distillation — data format + loss masking:**

```python
from transformers import AutoTokenizer

CHAT_TEMPLATE = (
    "<|im_start|>user\n{q}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n{cot}\n</think>\n{ans}<|im_end|>"
)

def build_cot_example(tokenizer, question, cot, answer, max_len=32768):
    # Full sequence: question + <think>CoT</think> + final answer
    text = CHAT_TEMPLATE.format(q=question, cot=cot, ans=answer)
    ids = tokenizer(text, truncation=True, max_length=max_len,
                    return_tensors="pt").input_ids[0]

    # Loss mask: learn on assistant turn (CoT + answer), ignore user prompt
    labels = ids.clone()
    user_prefix = tokenizer(
        f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    ).input_ids
    labels[: len(user_prefix)] = -100
    return {"input_ids": ids, "labels": labels}

def cot_sft_step(student, batch, optimizer):
    # Plain next-token CE over the entire assistant turn — including <think>.
    # That's all DeepSeek-R1-Distill does: no RL, no logit matching.
    out = student(input_ids=batch["input_ids"], labels=batch["labels"])
    optimizer.zero_grad(); out.loss.backward(); optimizer.step()
    return out.loss.item()

# Generation side: the student must produce <think>...</think> then answer.
# Stop tokens and decoding config matter for reasoning evaluation:
def generate_reasoning(student, tokenizer, question, max_new_tokens=8192):
    prompt = (f"<|im_start|>user\n{question}<|im_end|>\n"
              f"<|im_start|>assistant\n<think>\n")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = student.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=0.6,           # R1-recommended for reasoning
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    return tokenizer.decode(out[0][ids.size(-1):], skip_special_tokens=False)
```

## Section 4 — Training Techniques in Depth

This section covers the engineering that separates a mediocre distillation from a great one.

### 4.1 Student initialization

Random initialization works but wastes capacity. Better options:
- **Copy alternate layers from the teacher.** DistilBERT kept every other layer of BERT-base; worked remarkably well.
- **Structured pruning + healing.** Remove layers, heads, and hidden dimensions from the teacher, then distill. Used by Llama 3.2.
- **Matryoshka / depth-up-scaling.** Train so that the first `k` layers form a valid sub-model; later distill *from the full model into the `k`-layer sub-model*.

### 4.2 On-policy vs. off-policy distillation

A critical axis.

**Off-policy (standard):** the teacher generates trajectories (or logits on fixed data). The student trains on them. Simple but suffers from **exposure bias**: the student never sees its own mistakes during training, so it drifts at inference.

**On-policy:** the student generates a trajectory; the teacher scores it (produces logits on the student's own continuation); student minimizes KL against the teacher at those states. Fixes exposure bias but much more expensive and unstable.

**Generalized Knowledge Distillation (GKD, 2023)** interpolates: with probability `λ` sample from the student, with probability `1−λ` from fixed data. `λ` can be annealed during training. GKD is now standard practice for instruction-following distillation.

Another notable method: **MiniLLM (ICLR 2024)** uses reverse KL `KL(student || teacher)` instead of the standard forward KL. Forward KL is mode-covering (student tries to cover everything the teacher does, including low-probability tails — bad for generation). Reverse KL is mode-seeking (student focuses on the teacher's high-probability modes — better for producing coherent text). MiniLLM combines reverse KL with on-policy sampling via policy gradient, and substantially outperforms standard KD on instruction following.

### 4.3 Tokenizer alignment

If teacher and student share a tokenizer (e.g. both use Llama's tokenizer), token positions line up and logit distillation is trivial.

If they don't (e.g. teacher = GPT-4, student = Llama), you have three options:
1. **Retokenize** the teacher's output text with the student's tokenizer and fall back to response distillation.
2. **Universal Logit Distillation (ULD, 2024)** — optimal-transport-based matching across vocabularies, works surprisingly well.
3. **Just use response distillation.** The most common choice in practice.

### 4.4 Data strategy

The prompt distribution dominates outcomes. Rules of thumb:
- **Diversity beats volume.** 50K diverse prompts usually beats 500K near-duplicates.
- **Match deployment distribution.** If the student will do code, put code in the prompt mix.
- **Seed with human prompts, expand with Self-Instruct / Evol-Instruct.** Self-Instruct generates new instructions from a few seeds. Evol-Instruct systematically rewrites prompts to be harder, deeper, or more constrained. WizardLM popularized the latter.
- **Curriculum.** Start easy, end hard. Especially effective for reasoning distillation.

### 4.5 Loss composition in practice

Real-world distillation rarely uses a single loss. A typical setup for modern instruction distillation:

```
L_total = λ_1 · L_CE_hard
        + λ_2 · L_KL_logit         # white-box, if available
        + λ_3 · L_hidden            # optional
        + λ_4 · L_attn              # optional
        + λ_5 · L_SFT_response      # on teacher-generated text
```

with λ values tuned per task. For pretraining-scale distillation (Gemma 2), `L_KL_logit` dominates. For instruction distillation (Alpaca-style), `L_SFT_response` dominates.

### 4.6 Post-distillation RLHF / DPO

Distillation gets you close to the teacher. To surpass it on specific axes (helpfulness, safety, style), run RLHF or DPO on top. Zephyr-7B, Tulu-2, and Mistral-Instruct all follow this pattern: SFT-distill from a strong teacher → DPO with preference data.

## Section 5 — Case Studies with Concrete Numbers

### Case Study 1 — DistilBERT (2019)

**Teacher:** `bert-base-uncased`, 110M parameters, 12 transformer layers, hidden size 768.
**Student:** 6 transformer layers, same hidden size 768, 66M parameters.

**Initialization:** every other layer copied from BERT-base — i.e. layers 0, 2, 4, 6, 8, 10 of the teacher become layers 0–5 of the student. This alone gives the student a head start.

**Training data:** the same corpus BERT was pretrained on (BookCorpus + English Wikipedia), no task-specific data.

**Loss (all applied during pretraining):**
- `L_mlm` — masked-language-modeling cross-entropy (hard labels).
- `L_kd` — KL between student and teacher output distributions, `T = 2`.
- `L_cos` — cosine similarity between student and teacher hidden states (last layer).

Final loss: `L = 5 · L_mlm + 2 · L_kd + 1 · L_cos` (approximate weights from the paper).

**Results:**
- 40% fewer parameters.
- 60% faster inference.
- Retained **~97% of BERT-base's GLUE score** (77.0 vs 79.5).

DistilBERT is the canonical demonstration that distillation works during *pretraining*, not just fine-tuning.

### Case Study 2 — Alpaca and Vicuna (2023)

**Alpaca setup:**
- Student: LLaMA 7B.
- Teacher: OpenAI `text-davinci-003`.
- Data: 175 seed instructions → expanded to 52K via Self-Instruct (teacher generates new instructions from few-shot prompts).
- Training: standard SFT, 3 epochs, cosine LR schedule, peak LR 2e-5, batch size 128.
- Cost: **~$600** total (data generation + fine-tuning on 8×A100).

**Vicuna setup:**
- Student: LLaMA 13B.
- Teacher: ChatGPT (conversations scraped from ShareGPT).
- Data: ~70K multi-turn user conversations.
- Training: SFT, 3 epochs, gradient checkpointing, max seq len 2048.

**Results:** Vicuna-13B reached roughly **90% of ChatGPT quality** on MT-Bench style GPT-4-as-judge evaluation, at a tiny fraction of the teacher's training cost.

Caveats: benchmark numbers *overstate* real capability. Distilled models learn the teacher's *style* quickly (fluent, confident, well-formatted) without necessarily learning the teacher's *reasoning* — the "style transfer" problem. GPT-4-as-judge is biased toward style.

### Case Study 3 — MiniLLM (ICLR 2024)

**Key innovation:** replace forward KL with **reverse KL**, and train on-policy using policy gradient.

**Setup:**
- Teacher: GPT-2 1.5B / OPT-13B / Llama-7B depending on experiment.
- Student: GPT-2 120M / OPT-1.3B / Llama-TinyLlama-1.1B.
- Loss: reverse KL `KL(student || teacher)`, estimated via Monte Carlo samples from the student.
- Optimizer: single-step policy gradient with teacher log-prob as reward.

**Results:** on Dolly instruction evaluation, MiniLLM outperformed standard SeqKD (sequence-level distillation) by **10–30% relative**, and crucially produced more coherent long-form generations.

Takeaway: loss choice matters as much as architecture choice. Reverse KL has become the default for modern generation-focused distillation.

### Case Study 4 — Gemma 2 2B / 9B (2024)

**Setup:** Google built Gemma 2 2B and 9B with **on-policy logit distillation during pretraining** — not post-training compression.

- Teacher: a large internal Gemma model (≥ 27B).
- Student: Gemma 2 2B or 9B, pretrained from scratch but with every token supervised by teacher logits (not just the next-token hard label).
- Loss: dominant term is KL against the teacher's full vocabulary distribution; auxiliary standard next-token CE.
- Data: multi-trillion-token web-scale pretraining corpus.

**Results:** Gemma 2 2B hit performance levels that the Gemma 1 generation needed ~7B parameters to reach. This is a big shift: distillation moved from "make a small version of my model" to "use a bigger model as a better teacher than raw text."

### Case Study 5 — Llama 3.2 1B / 3B (2024)

Meta's edge-device Llamas were built in two stages.

**Stage A — structured pruning:**
- Start from Llama 3.1 8B.
- Remove entire transformer layers (depth pruning), attention heads (head pruning), and hidden dimensions (width pruning), guided by importance scores.
- Result: a 1B or 3B "skeleton" that is architecturally a subset of the 8B model, with weights inherited from it.

**Stage B — distillation:**
- Teachers: Llama 3.1 8B and Llama 3.1 70B.
- Student: the pruned 1B / 3B model.
- Loss: logit distillation (top-K cached) plus standard next-token CE.
- Additional: continued pretraining on a large corpus with the distillation loss on top.

**Results:** Llama 3.2 1B runs at > 30 tokens/sec on a modern phone CPU and scores within a few points of Llama 3.1 8B on many instruction benchmarks. Prune-then-distill is now the default recipe for producing edge-scale models.

### Case Study 6 — DeepSeek-R1-Distill (2025)

This is the most important distillation result of 2025 and worth dwelling on.

**Context:** DeepSeek-R1 is a large reasoning model trained with large-scale reinforcement learning (GRPO variant). It produces long CoTs with self-verification, backtracking, and "aha" moments. But it is too expensive to serve at scale.

**Setup for R1-Distill:**
- Teacher: DeepSeek-R1 (large MoE, produces CoTs).
- Students: Qwen2.5-1.5B/7B/14B/32B and Llama-3.1-8B/70B.
- Data: **800K curated reasoning traces** generated by R1 — 600K reasoning (math, code, science) and 200K general-purpose. Filtered by correctness (automatic verifier for math/code) and by format quality.
- Training: plain SFT (!) on the `(question, full_CoT, answer)` sequences. No RL at the student stage.
- Hyperparameters: 2 epochs, LR 1e-5 with cosine decay, batch size ~512, sequence length 32K (to fit the long CoTs).

**Results:**
- DeepSeek-R1-Distill-Qwen-7B: **AIME 2024 pass@1 ≈ 55.5%**, MATH-500 ≈ 92.8%. This is higher than GPT-4o and Claude 3.5 Sonnet on these math benchmarks despite being 7B.
- DeepSeek-R1-Distill-Qwen-32B: approaches or matches o1-mini on several reasoning benchmarks.
- Critically: **no RL was used on the students.** Pure SFT on teacher CoTs was enough to transfer reasoning behavior.

Takeaway: you can distill not just answers but **procedures**. A small model trained on long, careful reasoning traces learns to reason in the same shape. This result implies that expensive RL training of reasoning capabilities only needs to happen *once* at the teacher — everything below can be distilled cheaply via SFT.

## Section 6 — Common Pitfalls

- **Style-transfer illusion.** Distilled models quickly learn to *sound* like the teacher (confident, fluent, nicely formatted) while remaining weaker at actual reasoning. Always evaluate on tasks with ground-truth correctness, not just GPT-4-as-judge.
- **Capability ceiling.** A student cannot exceed its teacher on dimensions supervised by distillation. Teacher hallucinations become student hallucinations. Teacher biases transfer.
- **Tokenizer mismatch.** Cross-tokenizer logit distillation is possible (ULD, MinED) but fiddly — response distillation is usually the pragmatic fallback.
- **Temperature mistake.** Using `T = 1` with a peaky teacher wastes the signal. Using `T = 10` flattens too much. Start with `T = 2`–`4`.
- **Forgetting `T²`.** The gradient compensation term is easy to drop; without it your loss weighting is silently wrong.
- **Data contamination.** If the teacher saw the eval set during its own training and then "generates data" for the student, the student can score artificially high. Use held-out evals and humans.
- **Legal / ToS boundaries.** Several frontier-API providers prohibit using their outputs to train competing models. Check terms.

### Training-Time Issues and How to Debug Them

The pitfalls above are *design* problems — chosen before training. The list below is the opposite: things that look fine on paper and go wrong once the job actually runs. For each: the symptom you'll see on the dashboard, the root cause, and the specific fix.

#### Loss is NaN / Inf after a few hundred steps

**Symptom:** training loss spikes, then `NaN`. Grad norm was climbing for many steps before.

**Common causes & fixes:**

| Cause | Fix |
|---|---|
| bf16 overflow in attention softmax | Use FlashAttention-2/3 (handles scaling internally), or cast softmax to fp32. |
| LR too high for a distilled student | Halve LR and relaunch from the last good checkpoint. For 7B SFT-distill, `1e-5` is a safe ceiling. |
| Bad row in dataset (empty, malformed, all pad tokens) | Wrap the dataloader with a sanity filter that drops sequences where `labels != -100` sums to 0. |
| Mixed-precision gradient underflow | Enable dynamic loss scaling (AMP) or switch fully to bf16 (no scaling needed). |

#### Training loss goes down, student generations are garbage

**Symptom:** loss curve looks textbook. Sampling from a checkpoint produces repetition, broken formatting, or nonsense.

**Root cause (90% of the time):** **loss mask is wrong** — you're training on the user prompt tokens too, so the model memorizes the role marker formatting instead of the assistant content. Verify by printing the label-masked sequence for one batch:

```python
# Should print assistant turn only; all user tokens = "-100"
for t, l in zip(tokenizer.convert_ids_to_tokens(ids[0]), labels[0].tolist()):
    print(f"{t:20s} {l}")
```

Other causes: chat template mismatch between train and eval (use the **exact** same `apply_chat_template` in both), or packing code that incorrectly joins `labels` across sequence boundaries.

#### Train loss plateaus at a high value

**Symptom:** loss drops from 2.5 to 1.8 in the first few hundred steps, then flatlines. Validation matches.

**Diagnosis flow:**
1. Is the **data format** correct? Decode one batch end-to-end and visually confirm the sequences look like what you'd want the student to output.
2. Are you computing loss on the **right tokens**? A plateau near `-log(1/V) ≈ 10.4` (`V = 32K`) means the model is essentially uniform — usually mask bug or optimizer not updating.
3. Is the **LR** too low? Try a 3× LR ladder (`5e-6 → 1e-5 → 3e-5`) and pick the one whose loss falls fastest without diverging.
4. For logit KD: did you forget the **$T^2$ correction**? Without it, effective KD weight collapses when $T > 1$.

#### Validation loss goes up before the student is any good

**Symptom:** train loss still falling, but val loss starts climbing after ~20% of an epoch. Classic overfit pattern — except the student hasn't reached anywhere near teacher quality yet.

**Root cause:** the student is **overfitting to the stylistic surface** of the teacher's traces (formatting, boilerplate, `"Certainly! Let me think step by step..."`), not the semantic content. This is the "style-transfer illusion" in its most concrete form.

**Fixes:**
- More **diverse** prompts (often the single biggest lever).
- **Deduplicate near-duplicates** — teacher responses to paraphrased prompts are frequently 80%+ identical.
- **Length filter** — cap max response length to drop pathological long traces, or log-normalize loss by length.
- Add a small amount (~10%) of **general SFT data** to dilute the style lock-in.

#### Gradient norm spikes, no NaN (yet)

**Symptom:** grad norm is usually ~1.0 but occasionally jumps to 20+. Loss wobbles but recovers.

**Fixes in order of cheapness:**
1. **Gradient clipping at 1.0** (you should always have this on; if you don't, enable it first).
2. Drop outlier-length sequences: a single 30K-token trace can dominate the batch. Filter by `seq_len < quantile_0.99`.
3. Switch from `Adam` to `AdamW` with `beta2=0.95` (more robust to outlier gradients than the usual `0.999`).
4. Warmup longer — 3% of steps isn't always enough; try 5%.

#### Student forgets its base capabilities (catastrophic forgetting)

**Symptom:** post-distill student is great at the target task, but MMLU / HumanEval / general chat quality dropped noticeably vs. the base checkpoint.

**Root cause:** your training mix is too narrow. Logit KD on math traces is effectively SFT on math text — anything outside that distribution gets overwritten.

**Fixes:**
- **Mix in 10–20% general SFT data** (UltraChat, OpenHermes, Tulu-3 mix) unchanged through training.
- Use a **smaller LR** (`5e-6` instead of `1e-5`) — sharper gradients do more damage.
- **Shorter training** (1 epoch instead of 2) — forgetting is monotonic in steps.
- Consider a **LoRA adapter** instead of full-parameter tuning if forgetting persists.

#### OOM with 32K-context reasoning traces

**Symptom:** job crashes in the first few steps with `CUDA out of memory`, or later when a long sequence packs into a batch.

**Fixes, in the order you should try them:**
1. **Reduce micro-batch size to 1**, increase gradient accumulation steps proportionally.
2. **FlashAttention-3** (not FA-2) for 32K+ — the memory win is substantial.
3. **Gradient checkpointing** — recomputes activations during backward; ~30% slower but cuts activation memory by 4–8×.
4. **Sequence packing** — when enabled, group short sequences into a full 32K window so you don't waste memory on padding. Use Axolotl's `sample_packing: true` or TRL's `ConstantLengthDataset`.
5. **Sequence parallelism** (Megatron, Ring-Attention) for sequences that exceed single-GPU activation memory even with gradient checkpointing.
6. **ZeRO-3** parameter sharding if model weights themselves are the problem.

#### Teacher and student logits are numerically very different

**Symptom:** KD loss is huge and doesn't decrease, even though student outputs look reasonable.

**Root cause:** different logit **scales** (some models output logits in `[-20, 20]`, others in `[-5, 5]`). The temperature you pick for one may not transfer.

**Fix:** measure the teacher's **entropy** on a held-out batch. If it's > 4 nats, your teacher is already "soft" — use `T = 1`. If it's < 1 nat, the teacher is peaky — use `T = 4` or higher. A good practical heuristic is to pick $T$ so the softened teacher entropy lands around 2.5–3.5 nats.

```python
# Quick diagnostic: measure teacher entropy at different T
for T in [1, 2, 3, 5]:
    with torch.no_grad():
        p = F.softmax(teacher(batch).logits / T, dim=-1)
        ent = -(p * (p + 1e-10).log()).sum(-1).mean()
    print(f"T={T}  entropy={ent.item():.2f} nats")
```

#### Loss decreases but sample quality plateaus early

**Symptom:** after ~500 steps, eval generations don't improve. Loss keeps falling until overfit.

**Root cause:** the student has matched the **easy** tokens (function words, repeated boilerplate, obvious completions) and is now only reducing loss on the long tail, which contributes little to generation quality.

**Fixes:**
- Switch to **token-level filtering** — skip loss on positions where the teacher's top-1 probability is > 0.95 (the teacher is already certain; the student has nothing to learn).
- Curriculum: start training on short easy prompts, graduate to longer hard ones.
- For reasoning distillation, **weight loss by position** — CoT later tokens (near the answer) often matter more than the early exploratory tokens.

#### Run is fine, but checkpoints differ in generation style

**Symptom:** step 2000 generates confidently and concisely; step 3000 generates verbosely and hedges; step 3500 generates neither well.

**Root cause:** this is normal. The final-step checkpoint is rarely the best one — usually 85–95% through training is better.

**Fix:** always evaluate multiple late checkpoints. A cheap version:

```python
# Eval every 500 steps on a fixed 100-prompt holdout.
# Pick the checkpoint that maximizes downstream metric, not min val loss.
```

Validation loss and downstream capability often **anti-correlate** in late-stage training — more memorization of the trace surface hurts generation.

## Section 7 — End-to-End Pipeline: Data → Distill → Eval

This section walks through every concrete step you would run, in order, for a realistic distillation project. We use a running example: **"Distill DeepSeek-R1 into Qwen2.5-7B to get a small reasoning model for math."** The same pipeline generalizes.

![End-to-end distillation pipeline: Scoping, Data, Training, Evaluation, with iteration feedback](/imgs/blogs/distillation-in-llm-03-pipeline.png)

### Phase A — Scoping and Choosing Models

Before any code, answer five questions:

1. **What exactly should the student be good at?** "Math reasoning for grade-school to olympiad level" is a scope. "Be smart" is not.
2. **What is the latency / memory budget?** This pins down student size. On a single 24GB GPU with FP16 KV cache, 7B is the ceiling.
3. **Do you have access to teacher weights, or only an API?**
   - Weights → white-box distillation (logits, hidden states) is on the table.
   - API only → response/CoT distillation only.
4. **Do teacher and student share a tokenizer?** Strongly prefer yes. If no, plan for ULD or response-only distillation.
5. **What will "success" mean?** Pick 2–4 concrete benchmarks *before* training starts. For math: GSM8K, MATH-500, AIME-2024, plus a held-out internal set.

For the running example: Teacher = DeepSeek-R1 (open weights, DeepSeek tokenizer). Student = Qwen2.5-7B-Base (open, Qwen tokenizer). Tokenizers differ → we fall back to response/CoT distillation (no logit matching). Budget: train on 8×H100 over ~3 days. Success: AIME-2024 pass@1 ≥ 50%.

### Phase B — Data Preparation

This phase usually consumes **50–70% of total project time**. It is the single biggest determinant of outcome.

#### Step B1 — Collect seed prompts

Where prompts come from, in rough order of value:
- **Real user traffic** from your deployed product (gold — captures true distribution).
- **Public high-quality benchmarks' *training* splits** (e.g. MATH train, NuminaMath, OpenMathInstruct).
- **Curated instruction datasets** — OpenHermes, Tulu-3, UltraChat, ShareGPT.
- **Synthetic generation** — Self-Instruct / Evol-Instruct expansions of the above.

For the math example: 200K problems from NuminaMath + MATH-train + GSM8K-train + internal problem bank. Strictly exclude anything contaminating the eval sets (AIME 2024, MATH-500 test).

#### Step B2 — Deduplicate and decontaminate

- **Exact dedup** by SHA of normalized prompt text.
- **Near-dedup** via MinHash / SimHash at e.g. Jaccard 0.85.
- **Decontaminate** against every eval set — compute 13-gram overlaps with AIME, MATH-500, GSM8K test, held-out internal eval. Drop any training item with ≥ 8/13-gram match.

Skipping decontamination is the single most common mistake and silently inflates scores by 5–20 points.

#### Step B3 — Generate teacher responses

```python
# Pseudocode for CoT response generation
for prompt in prompts:
    responses = teacher.generate(
        prompt,
        n=8,                 # multiple samples per prompt
        temperature=0.6,     # R1-recommended for reasoning
        top_p=0.95,
        max_tokens=32768,    # long CoTs
        stop=["</answer>"],
    )
    save(prompt, responses)
```

Notes:
- Use `n > 1` (here 8) so you can do rejection sampling later.
- For reasoning teachers, temperature in 0.5–0.7 is usually best — pure greedy often fails at long CoTs.
- Store the full trace including `<think>…</think>` sections.

Cost example: 200K prompts × 8 samples × avg 8K tokens ≈ **12.8B output tokens**. On a self-hosted R1 via vLLM on 8×H100, this runs in roughly 1.5–2 days.

#### Step B4 — Filter (rejection sampling)

For each `(prompt, response_1, ..., response_8)`:

1. **Correctness filter** — for math/code, run an automatic verifier. Keep responses whose final answer matches ground truth. For non-verifiable tasks, use a reward model or an "LLM-as-judge" score ≥ threshold.
2. **Format filter** — drop truncated generations (hit `max_tokens`), malformed `<think>` blocks, responses with repeated loops (a common R1 failure mode).
3. **Length filter** — drop extremely short (< 50 tokens) or extremely long (> 30K tokens) responses for stability.
4. **Diversity filter** — if multiple responses to the same prompt all pass, keep the shortest correct one (trains the student to be concise) OR keep the top-2 to double data.

Typical survival rate for math at this stage: **40–60%**. Of 1.6M generated candidates you keep ~800K. That matches the DeepSeek-R1-Distill dataset size.

#### Step B5 — Format into training sequences

For CoT distillation, each training example is a single sequence:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>
{chain of thought}
</think>
{final answer}<|im_end|>
```

Compute the **loss mask**: loss is computed only on the assistant tokens (including the `<think>` block), not on the user prompt. This is standard SFT masking.

Shuffle globally. Pack multiple short sequences into the context window to avoid padding waste — use a packing library (e.g. Axolotl's sample packing, TRL's `ConstantLengthDataset`).

#### Step B6 — Hold out a validation split

Carve out 1–2% of the post-filter data as a validation set. You will use it for loss curves and early stopping — it is *not* your benchmark eval. Keep benchmark evals completely untouched.

### Phase C — Distillation Training

#### Step C1 — Environment and framework

For 7B-scale SFT with long sequences:
- **Framework:** Axolotl, LLaMA-Factory, or a custom TRL `SFTTrainer` script.
- **Precision:** bf16 weights + bf16 gradients; optimizer states in fp32.
- **Parallelism:** FSDP or DeepSpeed ZeRO-3 across 8 GPUs. With 32K context, you likely also need sequence parallelism or FlashAttention-3 + gradient checkpointing.
- **Attention:** FlashAttention-3 (critical at 32K context).

#### Step C2 — Hyperparameters (concrete starting point)

For Qwen2.5-7B on 800K reasoning traces:

| Hyperparameter | Value | Notes |
|---|---|---|
| Epochs | 2 | DeepSeek used 2; more causes overfit |
| Global batch size | 512 sequences | via grad accumulation |
| Micro-batch per GPU | 1 | memory-bound at 32K context |
| Max sequence length | 32,768 | to fit long CoTs |
| Learning rate | 1e-5 (peak) | base models; use 5e-6 for instruct |
| LR schedule | Cosine decay to 10% of peak | |
| Warmup | 3% of total steps | |
| Weight decay | 0.0 or 0.01 | 0.0 for pure distillation |
| Optimizer | AdamW, β=(0.9, 0.95) | |
| Gradient clipping | 1.0 | |
| Dropout | 0.0 | distillation is data-hungry, not regularization-hungry |

Total steps ≈ `2 × 800K / 512 ≈ 3,125`. Wall-clock on 8×H100 ≈ 60–80 hours.

#### Step C3 — Run and monitor

Log and watch:
- **Training loss** — should decrease smoothly from ~1.5 to ~0.4 over 2 epochs. A plateau at high loss usually means LR too low or data format wrong.
- **Grad norm** — spikes above 10× median indicate instability; lower LR or clip harder.
- **Validation loss** every ~100 steps on the held-out split.
- **Tokens/sec** to catch performance regressions.
- **Sample generations** every ~500 steps on 10 fixed prompts — watch for mode collapse, repetition, forgotten formatting.

#### Step C4 — Checkpointing strategy

Save checkpoints every ~500 steps. Keep the last 3. After training, run **your eval suite on several late checkpoints** (not just the final one) — the best model is often at 90–95% through training, not the absolute final step.

#### Step C5 — Optional: DPO on top

After SFT-distillation, optionally run DPO with preference pairs (chosen = teacher's correct response, rejected = teacher's incorrect response from the same prompt — you have these from Step B4). DPO for 1 epoch with β=0.1 and LR 5e-7 usually adds a few points on instruction-following benchmarks.

### Phase D — Evaluation

Never ship a distilled model without a thorough eval.

#### Step D1 — Capability benchmarks (task-correctness)

For the math example, at minimum:

| Benchmark | What it measures | Metric |
|---|---|---|
| GSM8K | Grade-school math word problems | pass@1 |
| MATH-500 | Competition-style math, 500 problems | pass@1 |
| AIME 2024 | 30 hard olympiad problems | pass@1, also pass@8 |
| MMLU-STEM | Multiple-choice science/math | accuracy |
| Held-out internal set | Matches your deployment distribution | pass@1 |

Eval settings matter — report all of: temperature, top-p, max tokens, prompt template, decoding format. For reasoning models, **pass@1 at temperature 0.6** is the standard; greedy underestimates reasoning models.

#### Step D2 — Regression benchmarks (no catastrophic forgetting)

A distilled-for-math model can lose general capability. Check:
- **MMLU** (general knowledge) — should not drop more than 2–3 points vs. the base student.
- **HumanEval / MBPP** — check coding is not destroyed.
- **IFEval** — instruction-following constraints.
- **TruthfulQA** — did hallucination get worse?

If any of these collapse, your data was too narrow. Re-mix with 10–20% general SFT data.

#### Step D3 — Style and safety

- **MT-Bench or AlpacaEval 2.0** with GPT-4 as judge — coarse but useful comparator. Remember the style-transfer bias.
- **Safety evals** — ToxicChat, HarmBench, your internal red-team set. Distilled models inherit safety properties of the teacher *and* the general SFT data.

#### Step D4 — Efficiency benchmarks

Measure what you actually shipped:
- **Throughput** (tokens/sec) at batch 1 and batch 32 on target hardware.
- **Latency** — time to first token, time per output token.
- **Memory** — peak GPU memory with the intended context length and KV cache.

#### Step D5 — Compare against the teacher and sensible baselines

A single absolute number is meaningless. Always report:
- Your distilled student.
- The untouched base student (to show distillation helped).
- The teacher (to show the ceiling).
- At least one competitive open model at the same size (e.g. Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct).

A healthy distillation run for our example looks like: AIME 2024 pass@1 of **55%** for the student, vs. **15%** for base Qwen2.5-7B and **79%** for the teacher R1. That's 70% of the way from base to teacher at 1/20th the inference cost.

#### Step D6 — Human spot-check

Sample 50–100 outputs across task types and read them. Benchmarks miss:
- Hallucinated confidence
- Stylistic tics inherited from the teacher ("Certainly! Let me…")
- Subtly wrong reasoning that lands on the right answer
- Refusals on benign prompts

If spot-check fails, no benchmark score saves you.

### Phase E — Iterate

Distillation is rarely one-shot. Typical second-pass fixes:
- **Student hallucinates on X** → add X-flavored prompts to the data mix, re-generate, re-distill.
- **Student too verbose** → filter training responses for length, or add DPO with shorter-is-better preferences.
- **Regression on general tasks** → mix in 10–20% general SFT data.
- **Student "breaks" formatting** → add explicit format-following examples.

Two or three iteration cycles, each 2–3 days, is normal.

---

## Section 8 — When to Use Distillation

Distillation is the right tool when:
- You already have a strong large model and need a cheaper deployable version.
- You want to bring reasoning, tool use, or instruction-following into a small open-weights model.
- You are doing edge / on-device inference with a hard parameter budget.
- You need a specialist — use a generalist teacher plus a domain-focused prompt set.

Distillation is the wrong tool when:
- The task requires capabilities the teacher doesn't have.
- Robustness matters in ways the teacher is weak at — distillation inherits, it doesn't repair.
- Fresh pretraining data and more compute would obviously close the gap more cheaply.

## Section 9 — The Big Picture

![A decade of LLM distillation: from Hinton (2015) to DeepSeek-R1-Distill (2025)](/imgs/blogs/distillation-in-llm-04-timeline.png)

Knowledge distillation began in 2015 as a small-network compression trick. A decade later it has become one of the central pillars of how frontier-quality capabilities flow into cheap, fast, local models:

- **DistilBERT (2019)** — proved pretraining-time distillation works.
- **Alpaca / Vicuna (2023)** — showed black-box response distillation scales with API budget.
- **MiniLLM (2024)** — reverse KL + on-policy fixed the mode-covering problem.
- **Gemma 2 (2024)** — distillation became a first-class pretraining signal.
- **Llama 3.2 (2024)** — prune-then-distill became the edge-model default.
- **DeepSeek-R1-Distill (2025)** — reasoning is transferable; SFT on teacher CoTs replaces RL for the student.

The field has converged on a simple principle: **the best teacher is a much larger, much more expensive model; the best student is the smallest one that can survive being taught by it.** Every knob — temperature, top-K, forward vs. reverse KL, on-policy vs. off-policy, hidden-state matching, CoT distillation — is engineering around that core relationship.

If you ship LLMs in production, distillation is almost certainly going to be in your stack within a year, if it isn't already.
