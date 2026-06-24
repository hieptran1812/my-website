---
title: "Any-Order Flexible Length Masked Diffusion"
publishDate: "2026-03-15"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  [
    "diffusion-model",
    "language-model",
    "masked-diffusion",
    "variable-length-generation",
    "text-generation",
  ]
date: "2026-03-15"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/any-order-flexible-length-masked-diffusion-20260315213706.png"
excerpt: "FlexMDM extends masked diffusion models to support variable-length sequence generation while preserving any-order inference. It matches MDM perplexity, improves length modeling, achieves 60% higher success on maze tasks, and boosts LLaDA-8B from 58% to 67% on GSM8K."
---

## Motivation

Autoregressive language models (GPT, LLaMA, etc.) generate text one token at a time, strictly left-to-right. This is simple and effective, but comes with a fundamental constraint: the model must commit to each token before seeing what comes next. It can't go back and revise, and it can't generate the middle of a sentence before deciding how the sentence ends.

**Masked Diffusion Models (MDMs)** like LLaDA offer an exciting alternative. Instead of generating left-to-right, they start with a sequence of `[MASK]` tokens and gradually reveal (unmask) them in any order — much like how a painter might work on different parts of a canvas simultaneously. This **any-order generation** enables powerful capabilities like infilling (filling in blanks in the middle of text), parallel decoding, and more flexible planning.

But MDMs have a critical limitation: **they require knowing the output length in advance**. Before generation begins, you must specify exactly how many `[MASK]` tokens to start with. This is a serious problem because:

- Natural language has variable length — a question like "What is 2+2?" could have an answer that's 1 token ("4") or 100 tokens (a detailed explanation)
- Planning tasks require variable-length trajectories — a maze solution might need 5 steps or 50 depending on complexity
- Code generation produces outputs of unpredictable length

Existing workarounds are hacky: pad to a maximum length and hope the model learns to predict `[EOS]` tokens for the unused positions. This wastes compute, introduces calibration issues, and often produces poorly calibrated length distributions.

**This paper asks**: Can we build a masked diffusion model that handles variable-length generation naturally, without sacrificing the any-order property?

## Background: How Masked Diffusion Models Work

Before diving into FlexMDM, let's build a thorough understanding of how standard Masked Diffusion Models (MDMs) work. If you're familiar with image diffusion models like Stable Diffusion, the core idea is similar — but instead of adding and removing Gaussian noise on pixels, we're adding and removing `[MASK]` tokens on text.

### The Forward Process (Corrupting Clean Text)

Imagine you have a clean sentence: `"The cat sat on the mat"`. The forward process gradually destroys this sentence by randomly replacing tokens with `[MASK]`, controlled by a timestep $t$ that goes from 0 (clean) to 1 (fully masked).

Think of $t$ as a "corruption dial":

```
t=0.0 (no corruption):     "The  cat  sat  on   the  mat"
                             ✓    ✓    ✓    ✓    ✓    ✓

t=0.3 (30% corrupted):     "The  [M]  sat  [M]  the  mat"
                             ✓    ✗    ✓    ✗    ✓    ✓

t=0.6 (60% corrupted):     "[M]  [M]  sat  [M]  [M]  [M]"
                             ✗    ✗    ✓    ✗    ✗    ✗

t=1.0 (fully corrupted):   "[M]  [M]  [M]  [M]  [M]  [M]"
                             ✗    ✗    ✗    ✗    ✗    ✗
```

**How is each token masked?** At timestep $t$, each token independently flips a biased coin with probability $t$ of landing heads. Heads = mask it, tails = keep it. So at $t=0.3$, each token has a 30% chance of being masked. This is independent per token — sometimes you get lucky and only 1 token is masked, sometimes unlucky and 4 are masked.

**Why is this modeled as a Continuous-Time Markov Chain (CTMC)?** A CTMC is just a fancy way of saying: the system transitions between states continuously over time, and the probability of each transition depends only on the current state (not the history). Here, each token's state is either "clean" or "masked", and the transition rate from clean→masked is determined by $t$. The "Markov" property means: to predict what happens next, you only need to know the current state of each token, not how it got there.

Mathematically, the transition rate matrix $Q_t$ for each token position is:

```
From\To    clean    [MASK]
clean       -r        r        where r = 1/(1-t)
[MASK]       0        0        (once masked, stays masked)
```

The rate $r = 1/(1-t)$ increases as $t$ approaches 1, meaning tokens get masked faster and faster toward the end — like an avalanche. At $t=0.5$, the rate is 2; at $t=0.9$, it's 10; at $t=0.99$, it's 100. This ensures that by $t=1$, essentially all tokens are masked.

### The Reverse Process (Generating Text from Masks)

Generation runs this process **backward in time**: start from all `[MASK]` at $t=1$ and progressively unmask tokens until you reach clean text at $t=0$.

```
t=1.0:  "[M] [M] [M] [M] [M] [M]"
         │
         │  Model sees 6 masks, predicts all tokens
         │  Most confident about position 3: "sat" (92%)
         ▼
t=0.7:  "[M] [M] sat [M] [M] [M]"
         │
         │  Model now has context from "sat"
         │  Predicts: pos1="The" (88%), pos4="on" (85%), pos6="mat" (79%)
         ▼
t=0.4:  "The [M] sat on [M] mat"
         │
         │  With "The ___ sat on ___ mat", model is very confident
         │  pos2="cat" (95%), pos5="the" (91%)
         ▼
t=0.0:  "The cat sat on the mat"  ← Complete!
```

**What the model actually learns**: At each step, the model (a neural network, typically a Transformer) takes in the partially-masked sequence and the current timestep $t$, and outputs a probability distribution over the entire vocabulary for every masked position. For example, at $t=0.7$ with input `"[M] [M] sat [M] [M] [M]"`, it might output:

```
Position 1: P("The")=0.88, P("A")=0.05, P("One")=0.02, ...
Position 2: P("cat")=0.72, P("dog")=0.15, P("bird")=0.03, ...
Position 4: P("on")=0.85, P("in")=0.08, P("by")=0.03, ...
Position 5: P("the")=0.76, P("a")=0.12, P("his")=0.04, ...
Position 6: P("mat")=0.79, P("floor")=0.09, P("bed")=0.05, ...
```

The model doesn't unmask all tokens at once — at each small time step $\Delta t$, each masked token has a small probability of being unmasked. Which tokens get unmasked first depends on both randomness and the model's confidence.

**The key property is any-order**: unlike autoregressive models that must go left-to-right, MDMs can unmask token 3 before token 1, or token 6 before token 2. The model uses **bidirectional context** — it can look at all unmasked tokens (both to the left AND right) when predicting each masked token. This is what enables powerful capabilities like:

- **Infilling**: Given "The **_ sat on _** mat", fill in the blanks
- **Parallel decoding**: Unmask multiple tokens simultaneously
- **Flexible generation order**: Start with the most confident predictions

### The Fixed-Length Problem — Why It Matters More Than You Think

Notice that the sequence length (6 tokens) is **fixed throughout the entire process**. The model needs to know in advance that the output is exactly 6 tokens long. In practice, this means:

1. Pick a maximum length (say, 256 tokens)
2. Allocate 256 `[MASK]` tokens
3. Generate all 256 positions
4. Hope the model fills some positions with `[EOS]` or `[PAD]` tokens

**Why is this worse than just "wasting some compute"?**

**Problem A — The PAD corruption paradox**: During the forward process, `[PAD]` tokens get masked just like real tokens. So the model sees sequences like `"The cat [M] [M] [M] [M]"` and can't tell whether position 3 was originally `"sat"` (a real word that should be predicted) or `[PAD]` (an empty position that should remain empty). This ambiguity makes training much harder.

**Problem B — Multimodal length distributions**: Imagine a question-answering task where the answer is either very short ("Yes") or very long (a paragraph explanation). The padding approach forces the model to allocate the same number of positions for both. The model must learn a complex, bimodal distribution over where to place `[EOS]` — and in practice, it often gets this wrong, producing answers that are too long or too short.

**Problem C — Wasted attention**: In a Transformer, every token attends to every other token. With 256 positions but only 20 real tokens, 92% of the attention computation is wasted on padding. This is $O(n^2)$ waste.

```
Actual content:  [The] [cat] [sat] [on] [the] [mat] [EOS] [PAD] [PAD] ... [PAD]
                  ←────── 7 real tokens ──────→      ←── 249 wasted positions ──→

Attention matrix: 256 × 256 = 65,536 entries
Useful entries:   7 × 7 = 49 (0.07% of total!)
```

## FlexMDM: The Proposed Method

FlexMDM extends MDMs to handle variable-length generation by introducing a fundamentally new operation alongside unmasking: **token insertion**. The model can now both reveal what a masked token should be AND grow the sequence by inserting new masked tokens.

### Core Idea: The Joint Interpolant

The central innovation of FlexMDM is rethinking what "diffusion" means for text. In standard MDMs, diffusion only changes **what tokens are** (masked vs. unmasked). In FlexMDM, diffusion also changes **how many tokens there are**.

Think of it this way:

- **Standard MDM**: You're given a jigsaw puzzle with exactly 100 pieces, all face-down. You flip them over one by one to reveal the picture. The number of pieces is fixed.
- **FlexMDM**: You start with an empty table. Puzzle pieces materialize out of thin air AND flip face-up simultaneously. You don't know in advance how many pieces the final puzzle will have.

Formally, FlexMDM defines a "joint interpolant" — a single mathematical framework that handles both operations. The generation process starts from an **empty sequence** (not a fixed-length mask sequence) and builds up to the target:

```
t=1.0 (start):        ""                            Length: 0
                        │
                        │  Insert 2 mask tokens
                        ▼
t=0.8:                 "[M] [M]"                     Length: 2
                        │
                        │  Unmask position 1 → "The"
                        │  Insert 3 more masks
                        ▼
t=0.6:                 "The [M] [M] [M] [M]"        Length: 5
                        │
                        │  Unmask positions 3,4 → "sat","on"
                        │  Insert 1 more mask
                        ▼
t=0.4:                 "The [M] sat on [M] [M]"     Length: 6
                        │
                        │  Unmask position 2 → "cat"
                        │  Unmask position 5 → "the"
                        ▼
t=0.2:                 "The cat sat on the [M]"      Length: 6
                        │
                        │  Unmask position 6 → "mat"
                        ▼
t=0.0 (done):          "The cat sat on the mat"      Length: 6
```

**Key observation**: The sequence **grows** during generation. At $t=0.8$ it has 2 tokens; by $t=0.4$ it has 6 tokens. The model simultaneously decides where to add new content (insertion) and what that content should be (unmasking).

The process is still a CTMC, but now with **two types of transitions**:

- **Unmasking**: `[MASK]` → actual token (same as standard MDM). A masked placeholder reveals its true identity.
- **Insertion**: A new `[MASK]` appears between two existing tokens. The sequence grows by one position.

These transitions happen concurrently and continuously. At any moment during generation, some tokens might be getting unmasked while new masks are being inserted elsewhere in the sequence.

### The Two Things the Model Must Learn

FlexMDM requires the neural network to learn two distinct skills, each handled by a dedicated output "head" on top of a shared Transformer backbone.

#### Skill 1: Unmasking Posterior $f_\theta$ — "What should each mask become?"

This is identical to what a standard MDM learns. Given a partially-masked sequence and timestep, predict the probability distribution over vocabulary tokens for each masked position:

$$f_\theta(x | \text{position } i, \text{sequence } s_t, t) = P(\text{token at position } i = x)$$

**Concrete example**: Suppose at timestep $t=0.5$, the current sequence is `"The [M] sat on [M] mat"`. The model processes this through its Transformer layers, and the token prediction head outputs:

```
Position 2 (first [M]):
  "cat"  → 72%     ← most likely
  "dog"  → 15%
  "bird" → 3%
  "big"  → 2%
  ...rest of vocabulary shares remaining 8%

Position 5 (second [M]):
  "the"  → 89%     ← very confident
  "a"    → 5%
  "his"  → 2%
  ...rest shares remaining 4%
```

The model is more confident about position 5 ("the") than position 2 ("cat"). In confidence-based inference, position 5 would be unmasked first.

**Why does this work well?** Because the model sees **bidirectional context**. When predicting position 2, it can see "The **_ sat on _** mat" — both the left context ("The") and the right context ("sat on \_\_\_ mat"). An autoregressive model could only see "The" when predicting the second word. This extra context makes predictions much more accurate.

#### Skill 2: Insertion Expectation $g_\theta$ — "Where should the sequence grow?"

This is entirely new to FlexMDM. The model predicts **how many new tokens should be inserted** at each gap between existing tokens:

$$g_\theta(\text{gap } i, \text{sequence } s_t, t) = \mathbb{E}[\text{number of tokens to insert at gap } i]$$

**What is a "gap"?** Between any two adjacent tokens, there's a potential insertion point. For a sequence of length $n$, there are $n+1$ gaps (including before the first token and after the last token):

```
Sequence: "The  [M]  sat"

Gaps:   ↓    ↓    ↓    ↓
      gap0 gap1 gap2 gap3
     (before (between (between (after
      "The")  "The"    "[M]"   "sat")
              & "[M]") & "sat")
```

**Concrete example**: At $t=0.7$, the sequence is `"The [M]"`. The insertion head predicts:

```
Gap 0 (before "The"):              0.1  → almost no insertions expected
Gap 1 (between "The" and "[M]"):   2.3  → expects ~2-3 more tokens here
Gap 2 (after "[M]"):               1.1  → expects ~1 more token after
```

This tells the model: "the sentence needs to grow mainly in the middle and slightly at the end." The actual number of insertions at each gap is sampled from a **Poisson distribution**.

**Why Poisson?** The Poisson distribution models "the number of events in a fixed interval" — perfect for "how many tokens should appear in this gap during the next small time step." If the expected rate is 2.3, you might get 0, 1, 2, 3, or 4 insertions, with 2 being most likely. The randomness is important: it allows the model to explore different length possibilities during generation, rather than committing to a fixed length.

**How does the model know where to insert?** The insertion prediction is informed by the full sequence context. If the model sees `"The [M]"` and knows (from its training data) that sentences starting with "The" typically have 5-8 more tokens, it predicts high insertion rates at the appropriate gaps. The Transformer's attention mechanism allows each gap to "look at" the entire current sequence when making this prediction.

### Architecture: How It All Fits Together

FlexMDM uses a **Diffusion Transformer (DiT)** — the same family of architecture used in modern image diffusion models, adapted for text. Here's how the pieces connect:

```
                    Input Sequence              Timestep t
                          │                         │
                          ▼                         │
               ┌──────────────────┐                 │
               │  Token Embedding │                 │
               │  + Position Enc  │                 │
               └────────┬─────────┘                 │
                        │                           │
                        ▼                           ▼
               ┌──────────────────────────────────────┐
               │                                      │
               │   Shared Transformer Backbone        │
               │                                      │
               │   ┌──────────────────────────────┐   │
               │   │ Layer 1: Self-Attention +     │   │
               │   │          Timestep Conditioning│   │
               │   └──────────────────────────────┘   │
               │   ┌──────────────────────────────┐   │
               │   │ Layer 2: Self-Attention +     │   │
               │   │          Feed-Forward         │   │
               │   └──────────────────────────────┘   │
               │              ...                     │
               │   ┌──────────────────────────────┐   │
               │   │ Layer N: Self-Attention +     │   │
               │   │          Feed-Forward         │   │
               │   └──────────────────────────────┘   │
               │                                      │
               └──────────┬───────────────────────────┘
                          │
                    ┌─────┴──────┐
                    │            │
                    ▼            ▼
           ┌──────────┐  ┌────────────┐
           │  Token    │  │ Insertion  │
           │  Head     │  │   Head     │
           │           │  │            │
           │ (Linear + │  │ (Linear +  │
           │  Softmax) │  │  Softplus) │
           └─────┬─────┘  └─────┬──────┘
                 │              │
                 ▼              ▼
           Per-position    Per-gap
           vocab probs     insert rates
```

**The shared backbone** is the workhorse — it processes the input through multiple self-attention layers, building rich contextual representations. Each token "talks to" every other token through attention, building up an understanding of the sequence as a whole. The timestep $t$ is injected as a conditioning signal (similar to how image diffusion models condition on the noise level), telling the model "how far along are we in the generation process?"

**The token head** takes each position's hidden representation and projects it to a probability distribution over the full vocabulary (using a linear layer + softmax). This is the $f_\theta$ component. It only produces meaningful outputs for masked positions — unmasked tokens don't need prediction.

**The insertion head** takes the hidden representations and computes the expected insertion count for each gap. Since gaps are _between_ tokens, the head typically uses the representations of adjacent tokens (e.g., averaging positions $i$ and $i+1$ to represent the gap between them). It outputs a non-negative number using a **Softplus activation** ($\text{softplus}(x) = \log(1 + e^x)$, which is always positive — you can't insert a negative number of tokens).

**Why two separate heads instead of two separate models?** Because both tasks benefit from the same contextual understanding. To predict what a masked token should be, you need to understand the sequence context. To predict where to insert tokens, you also need to understand the sequence context. The shared backbone computes this context once, and the two heads specialize in their respective tasks. This is more efficient and allows the tasks to help each other through shared representations.

### Training: Teaching the Model Both Skills Simultaneously

The training objective combines two losses:

$$\mathcal{L} = \mathcal{L}\_{\\text{unmask}} + \lambda \mathcal{L}\_{\\text{insert}}$$

**Here's how a single training step works, step by step:**

**Step 1: Start with a clean sequence from the training data**

```
x_0 = "The cat sat on the mat"  (length = 6)
```

**Step 2: Sample a random timestep**

```
t ~ Uniform(0, 1)
Suppose t = 0.6
```

**Step 3: Construct the corrupted sequence by simulating the forward process**

This is the clever part. To create the training example at time $t=0.6$, we need to simulate what the sequence would look like at 60% through the corruption process. This involves two independent corruptions:

**3a. Delete some tokens** (simulating tokens that "haven't been inserted yet"):

Each token is independently deleted with probability $(1-t) = 0.4$. This simulates the fact that at $t=0.6$, only about 60% of the final tokens have been "inserted" so far.

```
"The cat sat on the mat"
        ✓   ✗   ✓   ✗   ✓   ✓     (randomly keep/delete)
→ "The sat the mat"  (2 tokens deleted: "cat" and "on")
```

**3b. Mask some remaining tokens** (simulating tokens that "haven't been unmasked yet"):

Each surviving token is independently masked with probability $t = 0.6$.

```
"The sat the mat"
  ✗   ✓   ✗    ✓    (randomly mask/keep)
→ "[M] sat [M] mat"  (2 tokens masked: "The" and "the")
```

The final corrupted sequence is `"[M] sat [M] mat"` — 4 tokens instead of the original 6.

**Step 4: Compute the two losses**

**Unmasking loss** $\mathcal{L}_{\text{unmask}}$: Standard cross-entropy at masked positions.

```
Model predicts:  Position 1: P("The") = 0.85 → loss = -log(0.85) = 0.16
                 Position 3: P("the") = 0.72 → loss = -log(0.72) = 0.33

Average unmasking loss = (0.16 + 0.33) / 2 = 0.245
```

This is the same loss as in standard MDM training — it teaches the model to predict what masked tokens should be.

**Insertion loss** $\mathcal{L}_{\text{insert}}$: How well does the model predict the number of deleted tokens at each gap?

The ground truth insertion counts are derived from the deletion step. "cat" was deleted between "The" and "sat", and "on" was deleted between "sat" and "the":

```
Current sequence: "[M]  sat  [M]  mat"
                 ↓    ↓    ↓    ↓    ↓
Ground truth:  gap0  gap1  gap2  gap3  gap4
inserts:        0     1     0     1     0
              (nothing ("cat"  (nothing ("on"  (nothing
              before   was     between  was    after
              "[M]")   here)   "sat"    here)  "mat")
                               &"[M]")
```

The model predicts insertion rates, and the loss measures how well these rates match the actual counts using **Poisson negative log-likelihood**:

```
Model predicts: gap0=0.1, gap1=1.2, gap2=0.1, gap3=0.9, gap4=0.1

Poisson NLL for each gap:
  gap0: actual=0, predicted=0.1 → -log(Poisson(0; 0.1)) = 0.10
  gap1: actual=1, predicted=1.2 → -log(Poisson(1; 1.2)) = 0.34
  gap2: actual=0, predicted=0.1 → -log(Poisson(0; 0.1)) = 0.10
  gap3: actual=1, predicted=0.9 → -log(Poisson(1; 0.9)) = 0.47
  gap4: actual=0, predicted=0.1 → -log(Poisson(0; 0.1)) = 0.10

Average insertion loss = 0.222
```

**Step 5: Total loss and backpropagation**

```
Total loss = L_unmask + λ * L_insert = 0.245 + λ * 0.222
```

The weight $\lambda$ balances how much the model focuses on "predicting token identities" vs. "predicting sequence growth." The paper experiments with different values and finds that a moderate $\lambda$ works best.

### Inference: Generating Text from Nothing

Generation proceeds from $t=1$ (empty sequence) to $t=0$ (complete sequence). The process is discretized into $N$ steps (typically 256) using **tau-leaping** — a method borrowed from computational chemistry for simulating CTMCs.

**What is tau-leaping?** Instead of simulating every infinitesimal time step of the continuous process (computationally impossible), we take discrete jumps of size $\Delta t = 1/N$. At each jump, we approximate how many events (insertions and unmaskings) would have occurred during that interval.

Here's the algorithm with detailed commentary:

```python
def generate(model, num_steps=256):
    """
    FlexMDM generation: from empty sequence to complete text.

    Unlike standard MDM which starts with [M][M]...[M] (fixed length),
    we start with an empty sequence and grow it.
    """
    sequence = []          # Start completely empty
    dt = 1.0 / num_steps   # Size of each time step

    for step in range(num_steps, 0, -1):
        t = step / num_steps  # Current time (decreasing from 1→0)

        # ─── Forward pass through the model ───
        # The model sees the current (partially-built) sequence and time t
        # It returns two things:
        #   token_probs: for each position, probability distribution over vocabulary
        #   insert_rates: for each gap, expected number of insertions
        token_probs, insert_rates = model(sequence, t)

        # ─── Phase 1: Insert new [MASK] tokens ───
        # For each gap between adjacent tokens (and before/after),
        # sample how many new masks to insert
        new_sequence = []
        for i in range(len(sequence) + 1):
            # Poisson sampling: expected count = rate * dt
            # If rate=2.5 and dt=0.004, expected insertions ≈ 0.01
            # (most steps insert 0 tokens at each gap)
            n_insert = poisson_sample(insert_rates[i] * dt)
            new_sequence.extend([MASK] * n_insert)
            if i < len(sequence):
                new_sequence.append(sequence[i])
        sequence = new_sequence

        # ─── Phase 2: Unmask some existing [MASK] tokens ───
        # Each [MASK] has a small probability of being revealed this step
        for i, token in enumerate(sequence):
            if token == MASK:
                # Probability of unmasking at this step
                unmask_prob = (1 / t) * dt  # increases as t→0
                if random() < unmask_prob:
                    # Sample a real token from the predicted distribution
                    sequence[i] = sample_from(token_probs[i])

    return sequence
```

**What happens during a typical generation?**

```
Step 256 (t=1.00): ""                                    [empty]
Step 250 (t=0.98): "[M]"                                 [1 token inserted]
Step 230 (t=0.90): "[M] [M] [M] [M]"                    [grew to 4 tokens]
Step 200 (t=0.78): "[M] cat [M] [M] [M] [M]"            [grew to 6, "cat" unmasked]
Step 150 (t=0.59): "The cat [M] on [M] mat"              ["The","on","mat" unmasked]
Step 100 (t=0.39): "The cat sat on [M] mat"              ["sat" unmasked]
Step  50 (t=0.20): "The cat sat on the mat"              [all done!]
Step   1 (t=0.00): "The cat sat on the mat"              [final cleanup]
```

Notice the two phases of generation:

1. **Early steps** ($t$ near 1): Mostly insertion. The sequence is growing rapidly but most tokens are still masked.
2. **Late steps** ($t$ near 0): Mostly unmasking. The sequence has reached its final length and tokens are being revealed.

This is natural — you first decide the "shape" of the output (how long it should be), then fill in the details.

### Adaptive Inference: Smarter Generation Strategies

The basic algorithm above unmasks tokens randomly (each mask has an equal chance per step). The paper proposes two smarter strategies:

**Strategy 1: Confidence-based unmasking**

Instead of randomly choosing which masks to unmask, prioritize the ones where the model is **most confident**:

```
Current: "The [M] sat [M] [M] mat"

Model confidence:
  Position 2: max_prob = 0.95 ("cat")   ← very confident
  Position 4: max_prob = 0.85 ("on")    ← confident
  Position 5: max_prob = 0.62 ("the")   ← less confident

→ Unmask position 2 first, then 4, then 5
```

This is like filling in a crossword puzzle — do the easy clues first, and their letters help you solve the harder ones. Once "cat" is revealed, the model has more context for predicting positions 4 and 5, potentially increasing its confidence.

**Strategy 2: Semi-autoregressive**

Divide the sequence into chunks and process them left-to-right, but within each chunk, use the any-order strategy:

```
Chunk 1: "The [M]"      → unmask in any order → "The cat"
Chunk 2: "sat [M]"      → unmask in any order → "sat on"
Chunk 3: "[M] mat"      → unmask in any order → "the mat"
```

This balances left-to-right coherence (like autoregressive models) with parallel speed (like MDMs).

**The crucial theoretical result (Proposition 3)**: Both strategies are provably valid — they sample from the same distribution as the basic algorithm. The proof relies on the fact that the CTMC's transition rates don't depend on the order of past events, only the current state. So you can unmask in any order without changing the distribution of final outputs.

### Why the Insertion Mechanism is Elegant

To appreciate FlexMDM's design, consider the alternatives for variable-length generation:

**Alternative 1: Length prediction + standard MDM**
First predict the length $L$ with a separate model, then run standard MDM with $L$ masks.

Problem: Length prediction is a hard, discrete decision made before seeing any generated content. If you predict $L=10$ but the ideal output is 15 tokens, you're stuck.

**Alternative 2: Dynamic EOS prediction**
Run MDM with max-length masks, let the model predict `[EOS]` at some position, then truncate.

Problem: The model wastes computation on positions after `[EOS]`, and the masking process corrupts `[EOS]` tokens during training, making them hard to learn.

**FlexMDM's approach**: Length emerges naturally from the generation process. The model doesn't make a single length decision — instead, it makes many tiny "should I insert a token here?" decisions throughout generation. This is inherently more flexible and robust:

- If the model realizes mid-generation that it needs more tokens, it can insert more masks
- If it realizes the sequence is long enough, it stops inserting
- The length is determined by the content, not the other way around

## Key Results

### Text Modeling (OpenWebText)

FlexMDM was evaluated on the OpenWebText dataset for unconditional text generation:

| Model                      | Perplexity      | Length KL Divergence       |
| -------------------------- | --------------- | -------------------------- |
| MDM (fixed-length, padded) | **baseline**    | 0.42 (poorly calibrated)   |
| FlexMDM (256 steps)        | **matches MDM** | **0.03** (well calibrated) |

The critical insight: FlexMDM matches MDM's perplexity (text quality) while dramatically improving **length modeling**. The length KL divergence measures how well the model's generated sequence lengths match the true data distribution — lower is better.

With MDM, even using 1024 denoising steps, the generated lengths are poorly calibrated — the model tends to produce sequences that are too long or too short. FlexMDM nails the length distribution with only 256 steps. This is a **14x improvement** in length calibration.

### Planning Tasks (Maze Subgoal Completion)

This experiment tests whether FlexMDM can handle tasks where the output length varies dramatically:

| Model              | Success Rate |
| ------------------ | ------------ |
| MDM (fixed-length) | ~24%         |
| FlexMDM            | **~90%**     |

That's approximately a **60% absolute improvement**. Why such a dramatic gap?

In maze planning, different start-goal pairs require different numbers of steps. A maze solution might need 5 steps or 50 steps. MDM with fixed-length padding must allocate for the worst case, wasting capacity on short solutions and potentially truncating long ones. FlexMDM naturally adapts to the required length — short paths get short sequences, long paths get long sequences.

### Scaling to 8B Parameters (LLaDA Fine-tuning)

The most impressive result: the authors took **LLaDA-8B** (a pretrained masked diffusion language model with 8 billion parameters) and fine-tuned it into FlexMDM. This took only **3 days on 16 H100 GPUs**.

| Benchmark                  | LLaDA-8B (before) | FlexMDM-8B (after) | Improvement |
| -------------------------- | ----------------- | ------------------ | ----------- |
| GSM8K (math reasoning)     | 58%               | **67%**            | +9%         |
| HumanEval (code infilling) | 52%               | **65%**            | +13%        |

These are substantial gains, especially considering:

- The fine-tuning cost is relatively modest (3 days on 16 GPUs)
- The improvements come from better length handling, not from additional training data
- Code infilling benefits enormously from flexible length — the model can insert exactly as many tokens as needed into a code gap

### Why Code Infilling Improves So Much

Code infilling is a perfect use case for FlexMDM. Consider:

```python
def fibonacci(n):
    # [FILL HERE]
    return result
```

The fill might need 3 lines or 15 lines depending on the implementation approach. With fixed-length MDM, the model must guess the fill length upfront. With FlexMDM, it naturally generates exactly the right amount of code — starting with the key logic and expanding to add edge cases, variable declarations, or loop bodies as needed.

## Analysis: Why FlexMDM Works

### The Length Bottleneck in Standard MDMs

Standard MDMs treat generation as a fixed-size "canvas painting" problem. You decide the canvas size (sequence length), fill it with mask paint, and gradually reveal the image. But text isn't a fixed-size canvas — it's more like writing on an infinitely long scroll where the length is part of what you're creating.

The padding approach (allocating max length and using `[EOS]` / `[PAD]`) creates several compounding problems:

1. **Wasted computation**: If max length is 256 but the target is 20 tokens, 92% of compute is wasted on padding
2. **Length calibration**: The model must learn a secondary task — predicting how many tokens should be `[PAD]`. This is surprisingly hard because the masking process corrupts the `[PAD]` tokens too, making it difficult to learn where "real content" ends
3. **Training-inference mismatch**: During training, the model sees the true length. During inference, it must estimate it. This mismatch degrades generation quality

FlexMDM eliminates all three problems by making length a first-class part of the generation process.

### The Insertion Mechanism as a "Growth" Process

A beautiful way to think about FlexMDM is as a **biological growth process**:

- **Standard MDM**: A sculptor chipping away at a fixed-size block of marble (reveal what's inside a fixed shape)
- **FlexMDM**: A crystal growing from a seed, adding atoms in the right places (the structure grows and refines simultaneously)

The insertion mechanism allows the model to:

1. Start with a rough sketch (a few tokens capturing the gist)
2. Progressively add detail (insert tokens where more content is needed)
3. Refine content (unmask tokens to reveal their identity)

This mirrors how humans often write — starting with key ideas, then expanding and refining.

### Theoretical Foundation: Why Any-Order Still Works

A natural worry: does adding insertion break the any-order property? If tokens are being inserted in the middle of the sequence, does the model need to process them in a specific order?

The paper proves (Proposition 3) that **any-order inference remains valid**. The key insight is that insertion and unmasking operate on different "dimensions" of the problem:

- **Insertion** changes the sequence length (a structural property)
- **Unmasking** changes the token identity (a content property)

These two operations can be interleaved in any order because they are governed by **independent rate matrices** in the CTMC framework. Mathematically, the joint rate matrix decomposes as a sum of the insertion rate matrix and the unmasking rate matrix. Since these matrices don't interact (insertion doesn't affect which token a mask becomes, and unmasking doesn't affect where insertions happen), the operations can be applied in any order. The model doesn't need to "finish inserting" before it starts unmasking, or vice versa.

## Connections to Other Work

### vs. Autoregressive Models (GPT, LLaMA)

| Property              | Autoregressive            | MDM              | FlexMDM                   |
| --------------------- | ------------------------- | ---------------- | ------------------------- |
| Generation order      | Left-to-right only        | Any order        | Any order                 |
| Variable length       | Natural                   | Requires padding | **Natural**               |
| Parallel decoding     | No                        | Yes              | Yes                       |
| Infilling             | Requires special training | Native           | **Native + length-aware** |
| Bidirectional context | No                        | Yes              | Yes                       |

FlexMDM combines the best of both worlds: the flexible-length capability of autoregressive models with the any-order/parallel capabilities of masked diffusion.

### vs. Insertion Transformers

Insertion Transformers (Stern et al., 2019) also generate by inserting tokens. The key difference:

- Insertion Transformers are **deterministic** — they use a learned policy to decide where and what to insert
- FlexMDM is **stochastic** — insertions are sampled from a principled probabilistic framework (CTMC)

The stochastic framework gives FlexMDM theoretical guarantees about sampling correctness that insertion transformers lack.

### vs. LLaDA

LLaDA is the closest predecessor — a large-scale masked diffusion language model. FlexMDM can be seen as a **strict upgrade** to LLaDA:

- Same architecture backbone (can fine-tune existing LLaDA checkpoints)
- Same any-order generation capability
- Added variable-length generation
- Better performance on benchmarks after fine-tuning

## Limitations and Open Questions

1. **Inference cost**: The insertion mechanism adds computational overhead compared to standard MDMs. Each step may change the sequence length, requiring dynamic memory management and preventing fixed-size tensor optimizations. However, the savings from not padding to max length often compensate for this.

2. **Insertion scheduling**: The paper uses a linear schedule for insertions (tokens are inserted at a roughly constant rate over time). It's unclear if more sophisticated schedules (e.g., insert most tokens early, then focus on unmasking) could improve quality.

3. **Scaling behavior**: While the 8B results are promising, it remains to be seen how FlexMDM scales beyond 8B parameters and whether it can close the gap with state-of-the-art autoregressive models on general language modeling benchmarks.

4. **Training data**: The fine-tuning results start from LLaDA-8B, which was trained on a specific data mixture. Pre-training FlexMDM from scratch at scale is an open research direction.

## Key Takeaways

1. **Variable-length generation is a first-class problem**. Simply padding to max length is not just inefficient — it fundamentally degrades generation quality, especially for tasks with variable output lengths.

2. **Insertion + unmasking is a principled framework**. By modeling both token identity and sequence length as part of the same CTMC, FlexMDM achieves theoretical guarantees while being practical.

3. **The gains are real and substantial**. +9% on GSM8K and +13% on HumanEval from a relatively cheap fine-tuning process. The 60% improvement on maze planning shows the framework shines when length variability is high.

4. **Any-order generation survives**. The insertion mechanism doesn't break MDMs' most attractive property. You can still unmask tokens in any order, enabling infilling, parallel decoding, and flexible generation strategies.

5. **The path from MDM to FlexMDM is lightweight**. Fine-tuning an existing LLaDA checkpoint took only 3 days on 16 H100s. This means the community can upgrade existing MDMs without pre-training from scratch.

## References

- Kim, J., Cheuk-Kit, L., Domingo-Enrich, C., Du, Y., Kakade, S., Ngotiaoco, T., Chen, S., & Albergo, M. (2025). [Any-Order Flexible Length Masked Diffusion](https://arxiv.org/abs/2509.01025).
- Nie, S., et al. (2025). [LLaDA: Large Language and Diffusion Assistant](https://arxiv.org/abs/2502.09992).
- Sahoo, S., et al. (2024). [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524). NeurIPS 2024.
- Stern, M., et al. (2019). [Insertion Transformer: Flexible Sequence Generation via Insertion Operations](https://arxiv.org/abs/1902.03249). ICML 2019.
