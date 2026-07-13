---
title: "Distilling LLMs and reasoning: sequence-level KD, on-policy, and CoT transfer"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Distilling a generative LLM is nothing like distilling a classifier — there is no single label, the output is a sequence, and what you actually want to move is capability. Here is the math of forward vs reverse KL, the on-policy and CoT recipes, runnable TRL-style code, and how to stack distillation with quantization and pruning."
tags:
  [
    "edge-ai",
    "model-optimization",
    "knowledge-distillation",
    "llm",
    "reasoning",
    "on-policy-distillation",
    "chain-of-thought",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/distilling-llms-and-reasoning-1.png"
---

You have a 70-billion-parameter reasoning model that gets 83% on your internal math-word-problem benchmark, and it costs about \$0.004 per query and 2.1 seconds of latency through the API. The product team wants this capability on-device — in the app, working offline, on a mid-range Android phone with an NPU and 8 GB of RAM. A 70B model is a non-starter on that hardware; even at int4 it is 35 GB of weights, an order of magnitude past the budget. The largest thing you can realistically run there is something like a 1.5B–3B model quantized to 4 bits, roughly 1–2 GB of weights, a few tokens per second. So you grab an off-the-shelf 1.5B base model, quantize it, and benchmark: 31% on the same math set. Forty-something points short. The capability is simply not in the small model, and no amount of clever quantization will conjure it — you cannot quantize your way to reasoning the base weights never learned.

This is the problem distillation solves, and it is genuinely different from every other lever in this series. Quantization and pruning take a model you already have and make it cheaper while trying to lose as little of *its* accuracy as possible — they are subtractive, bounded above by the model you started with. Distillation is the one lever that *moves capability from a model you have into a model you want*. The 70B model knows how to reason about these problems; the 1.5B model has the capacity to do a meaningful chunk of that reasoning if it is taught the right way. Distillation is the teaching. By the end of this post you will be able to set up sequence-level distillation, on-policy distillation with reverse KL, and chain-of-thought distillation in real code, reason about which one to use, and — because this is the composition capstone for the optimization track — stack distillation with quantization and pruning in the order that actually works.

But here is the catch that makes LLM distillation its own subject rather than a footnote to classifier distillation. The original distillation idea — Hinton, Vinyals, and Dean's 2015 "Distilling the Knowledge in a Neural Network" — was about a classifier: you have a fixed input, the teacher produces a probability distribution over a few hundred or a few thousand classes (the "soft labels"), and the student learns to match that single distribution. That is a clean, one-shot supervised target. An LLM has no single label. Its output is a *sequence* — hundreds or thousands of tokens, generated one at a time, each conditioned on every token before it. The output space is not a few thousand classes; it is the set of all possible token sequences, which is astronomically large. And what you want to transfer is not "the probability of class 7" but a *behavior*: the ability to follow an instruction, to lay out a chain of reasoning, to write code that runs. Figure 1 lays out the three distinct regimes this forces on us — matching the per-token distribution, matching the teacher's generated sequences, and scoring the student's own samples — and the rest of the post is an unpacking of each.

![A comparison matrix of token-level KD, sequence-level KD, and on-policy distillation across what is matched, who writes the training text, exposure bias, and compute cost](/imgs/blogs/distilling-llms-and-reasoning-1.png)

There is one framing to carry through the whole post, the recurring spine of this series: four levers — quantization, pruning, distillation, and efficient architecture — sitting on compilers and runtimes, read off an accuracy–efficiency Pareto frontier. Distillation is the lever that does something the other three cannot: it can move a model to a Pareto point that was simply unreachable from the small model's own starting weights, because it injects capability the small model never had. A quantized 1.5B model and a distilled-then-quantized 1.5B model are the same size and the same speed — but the distilled one can be 30 points more accurate, because distillation changed *what the weights know*, not how many bits encode them. That is why this is the capstone of the technique tracks: it composes with the others, and it is the one that breaks the "you can't exceed what you started with" ceiling. (You still cannot exceed the *teacher* — but the teacher can be enormous, so the ceiling is high.) If you want the map of all four levers and where this one sits, the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) is the post to read alongside this one.

## 1. Why a generative LLM is not a classifier you can soft-label

Start by being precise about what changed, because almost every mistake in LLM distillation comes from importing a classifier intuition that no longer holds.

A classifier maps one input $x$ to one distribution $p(y \mid x)$ over a fixed label set. Hinton-style distillation softens that distribution with a temperature $T$ and trains the student to match it:

$$
p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}, \qquad \mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\!\left(p^{\text{teacher}}(T)\,\big\|\,q^{\text{student}}(T)\right)
$$

The temperature $T > 1$ flattens the distribution so the "dark knowledge" — the relative probabilities of the *wrong* classes, which encode the teacher's sense of which mistakes are reasonable — survives into the gradient. The $T^2$ factor rescales the gradient magnitude back up because softmax gradients shrink like $1/T^2$. This is covered in depth in the [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) post; here I want to focus on what breaks when $y$ is a sequence rather than a class.

An autoregressive language model factorizes the probability of a sequence $y = (y_1, \dots, y_L)$ given a prompt $x$ as a product of per-token conditionals:

$$
p(y \mid x) = \prod_{t=1}^{L} p(y_t \mid y_{\lt t}, x)
$$

Each factor $p(y_t \mid y_{\lt t}, x)$ is itself a categorical distribution over the vocabulary — typically 32k to 256k tokens. So at one level, an LLM *is* a sequence of classifiers, one per position, and the obvious move is to apply Hinton distillation at every position: match the teacher's next-token distribution at each step. That is **token-level (or word-level) KD**, and it is the first regime in Figure 1. It is real, it works, and it is the cheapest thing you can do — but it has two problems that the sequence structure creates and a classifier never has.

The first problem is the **vocabulary explosion in the target**. The teacher's per-token distribution is a vector over the whole vocabulary. To match it exactly you need the teacher's full logits at every position of every training sequence — that is a 256k-dimensional float vector per token, which for a corpus of billions of tokens is an absurd amount of storage and bandwidth. In practice you either run the teacher live alongside the student (expensive: you pay a teacher forward pass per step) or you store only the top-$k$ teacher logits per token (a common compromise — top-50 or top-128 captures most of the mass and is 2000× smaller). Already the "just soft-label it" intuition needs an asterisk.

The second problem is deeper and it is the one that motivates everything that follows: **exposure bias**, also called the train–test distribution mismatch. When you train token-level KD, at every position you feed the student the *ground-truth* (or teacher-generated) prefix $y_{\lt t}$ and ask it to predict $y_t$. This is "teacher forcing": the student never sees its own predictions during training, only the correct prefix. But at inference time the student generates autoregressively — it conditions on *its own* previous tokens, which contain errors. The moment the student makes one off-distribution token, the prefix it is now conditioning on is a prefix it was never trained on, and the errors compound. The student is trained on the distribution of teacher prefixes and tested on the distribution of its own prefixes, and those distributions drift apart over a long generation. A classifier never has this problem because there is no autoregression — the input is fixed. For a sequence model it is the central pathology, and the two regimes after token-KD in Figure 1 (sequence-KD and on-policy) are both attempts to close this gap.

### What "transfer capability" actually means here

There is a third, more conceptual difference worth saying plainly. In a classifier, the soft labels are the whole story — match the distribution and you are done. In an LLM, matching the per-token distribution on a fixed corpus does *not* guarantee you transfer the capability you care about. The capability — say, multi-step arithmetic reasoning — lives in the *trajectory*: the model has to generate a coherent chain of intermediate steps, each conditioned on the last, and arrive at a correct answer. You can have a student whose per-token cross-entropy against the teacher is low but which still cannot complete a correct reasoning chain, because per-token matching on teacher prefixes never exercises the student's ability to *recover* from its own intermediate states. This is why, for reasoning specifically, the most effective methods (chain-of-thought distillation, on-policy distillation) are about the *sequences the student generates and the steps it produces*, not about pointwise distribution matching. Hold this; it is the thesis of the whole post.

## 2. Token-level KD: the math, and where it is enough

Let me make token-level KD concrete before tearing into its limits, because it is a fine baseline and for some tasks it is all you need.

The token-KD loss is a per-position KL between teacher and student next-token distributions, summed over the sequence and usually mixed with the ordinary next-token cross-entropy against the ground-truth token:

$$
\mathcal{L}_{\text{tok-KD}} = \sum_{t=1}^{L} \Big[ (1 - \alpha)\, \underbrace{\mathrm{CE}\big(y_t,\, q(y_t \mid y_{\lt t}, x)\big)}_{\text{hard label}} \;+\; \alpha\, T^2\, \underbrace{\mathrm{KL}\big(p(\cdot \mid y_{\lt t}, x)\,\|\,q(\cdot \mid y_{\lt t}, x)\big)}_{\text{soft teacher}} \Big]
$$

with $\alpha \in [0, 1]$ trading off the hard ground-truth signal against the soft teacher signal, and $T$ the distillation temperature. The KL here is the **forward KL**, $\mathrm{KL}(p \| q)$ — teacher first. That direction matters enormously and section 4 derives why; for now just note it is the direction Hinton used and the direction a naive port of classifier distillation gives you.

When is token-KD enough? When three conditions hold. First, the **prefixes at training time look like the prefixes at inference time** — i.e., exposure bias is mild. This is true for short outputs (classification-as-generation, short-form extraction, single-line answers) where there is little room for the student to wander off-distribution. Second, the **teacher and student share a tokenizer**, so the per-token distributions are over the same vocabulary and the KL is well-defined token-for-token. (Cross-tokenizer distillation is possible but requires alignment tricks — minimum-edit-distance token mapping or sequence-level methods that sidestep token alignment entirely. If your teacher and student come from different families with different tokenizers, lean toward sequence-level KD, which only needs the teacher's *text*, not its logits.) Third, you can **afford the teacher logits** — either by running the teacher live or by precomputing top-$k$.

For an image-classifier-shaped task dressed up as generation, token-KD with a good teacher routinely recovers most of the gap. DistilBERT (Sanh et al., 2019) is the canonical encoder example: a 6-layer student distilled from 12-layer BERT, 40% smaller and 60% faster, retaining about 97% of BERT's GLUE score — and its distillation loss is exactly this masked-LM token-KL plus a cosine embedding term. For an encoder there is no autoregressive generation at inference, so exposure bias never bites, and token-KD is close to optimal. The trouble starts when the output is a *long free generation* and the student has to walk a long path on its own legs.

#### Worked example: token-KD on a short-answer task

Take a 1.5B student and a 13B teacher on a short extractive QA task where answers are 1–8 tokens. The teacher gets 79.0% exact-match. The base 1.5B (no distillation, just fine-tuned on the same data with hard labels) gets 68.5%. Add token-KD with $\alpha = 0.5$, $T = 2$, teacher logits computed live: the student reaches 74.8% — it recovers about 60% of the teacher–student gap, at the cost of one extra teacher forward pass per training step. Because answers are short, exposure bias is negligible and token-KD captures most of what is available. Quantize that distilled student to int4 weight-only (GPTQ) and it drops to 74.1% — a 0.7-point quantization cost on top of the distilled accuracy, and now it is roughly 0.9 GB and runs at interactive speed on a phone. That is the composition this post is ultimately about: distill to lift capability, then quantize to fit the device. The [weight-only LLM quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) post covers the int4 step in detail.

Now watch what happens when the output gets long.

## 3. Sequence-level KD: distill the mode, not the distribution

Here is the move that made sequence-to-sequence distillation work, from Kim and Rush's 2016 paper "Sequence-Level Knowledge Distillation" — originally for neural machine translation, but the idea is general and it is the bedrock of how people distill LLMs today.

The insight is this. Token-level KD tries to match the teacher's full *distribution* at every position. But the thing you actually deploy is a *decoder* that, at inference, mostly produces the teacher's high-probability sequences — its modes. The teacher's distribution over sequences is overwhelmingly concentrated on a small number of good sequences; the rest of the probability mass is spread thinly over a combinatorial space of sequences you will never generate. Matching the full distribution spends the student's limited capacity modeling that irrelevant tail. What if instead you just trained the student to reproduce the teacher's *most likely sequences* — i.e., distill the mode of the teacher's sequence distribution rather than the whole distribution?

Concretely, sequence-level KD (seq-KD) is almost embarrassingly simple to implement:

1. Run the teacher to **generate** an output for each training input — typically greedy decoding or beam search, which approximates $\arg\max_y p(y \mid x)$, the teacher's mode.
2. **Replace** the ground-truth targets with the teacher's generated sequences.
3. Train the student with ordinary next-token cross-entropy on these teacher-generated targets — no teacher logits needed at training time, just the teacher's *text*.

That is it. You are doing standard supervised fine-tuning, but the labels are the teacher's own greedy outputs instead of the original human references. The theory in Kim and Rush is that this approximates a sequence-level KL where you have replaced the teacher's full sequence distribution with a point mass at its mode — and because the mode is what decoding produces anyway, this is exactly the part of the distribution the student needs.

Let me make that approximation rigorous, because it is the science that justifies the trick. The "honest" sequence-level objective is the KL between teacher and student *over whole sequences*, not over tokens:

$$
\mathcal{L}_{\text{seq-KL}} = \mathrm{KL}\big(p(y \mid x)\,\|\,q(y \mid x)\big) = -\sum_{y \in \mathcal{Y}} p(y \mid x)\,\log q(y \mid x) + \text{const}
$$

where $\mathcal{Y}$ is the set of *all* possible output sequences. The first term is a cross-entropy of the student against the teacher's full sequence distribution; the constant is the teacher's entropy, which does not depend on the student and drops out of the gradient. The problem is the sum over $\mathcal{Y}$: it is a sum over an exponentially large (for length $L$ and vocabulary $V$, it is $V^L$) set of sequences. You cannot compute it. Kim and Rush's move is to approximate $p(y \mid x)$ by a point mass at its mode $\hat{y} = \arg\max_y p(y \mid x)$:

$$
p(y \mid x) \approx \mathbb{1}[\,y = \hat{y}\,] \quad\Longrightarrow\quad \mathcal{L}_{\text{seq-KD}} \approx -\log q(\hat{y} \mid x) = -\sum_{t=1}^{L}\log q(\hat{y}_t \mid \hat{y}_{\lt t}, x)
$$

and the right-hand side is *exactly* ordinary next-token cross-entropy of the student on the teacher's single best sequence $\hat{y}$. That is the whole derivation: replace the intractable sum over all sequences with the one sequence carrying almost all the mass, and the sequence-level KL collapses to plain supervised fine-tuning on the teacher's greedy (or beam) output. The approximation is good precisely because, for a well-trained teacher, $p(\hat{y} \mid x)$ is large and the rest of $\mathcal{Y}$ is a thin tail — the point mass is not a crude hack, it is a justified concentration of the objective on the part of the distribution that matters at decode time. You can sharpen it by using beam search (a better mode estimate) or by keeping the top-$k$ teacher sequences instead of just the top one (a $k$-point approximation, which interpolates back toward the full distribution at $k$× the cost).

Why is this better than token-KD for long outputs? Two reasons, and both are about exposure bias. First, the teacher's generated sequences are *internally consistent* — they are real trajectories the teacher actually produces, with all the long-range coherence that implies, rather than a position-by-position averaging that can produce locally plausible but globally incoherent targets. Second, and subtly, training on teacher-generated text shifts the *training prefix distribution* toward the kind of fluent, on-distribution text the student will encounter when it generates — it does not fully close the exposure-bias gap (the prefixes are still the teacher's, not the student's), but it moves in the right direction. Kim and Rush found seq-KD alone recovered most of the gap between a large teacher and a small student on translation, and combining seq-KD with token-KD ("seq-inter") did best of all.

### The synthetic-data view: this is what "the teacher writes the training set" means

Step back and notice what sequence-level KD actually is: **the teacher generates the training data, and the student learns from it.** That reframing is the bridge to everything modern. When people say "we distilled GPT-4 into a 7B model," they almost never mean token-level logit matching — they mean they used the teacher to *generate a dataset* (instructions and responses, or questions and worked solutions) and then fine-tuned the student on it. Alpaca (Taori et al., 2023) is the famous early example: 52,000 instruction-following examples generated by a strong teacher from 175 seed tasks, used to fine-tune a small base model into something that follows instructions. That is sequence-level distillation by another name — the teacher's *sequences* are the supervision.

This is powerful and it is also where the dragons live, so section 7 has a whole subsection on synthetic-data pitfalls (mode collapse, diversity loss, hallucinated content, and the licensing problem of training on a commercial teacher's outputs). For now, hold the clean version: seq-KD = teacher generates, student imitates, and it is the workhorse of LLM distillation precisely because it needs only teacher *text*, works across tokenizers, and produces coherent long-form targets.

#### Worked example: seq-KD on long-form generation vs token-KD

Same 1.5B student, same 13B teacher, but now a summarization task with 100–200-token outputs where exposure bias bites. Token-KD (from section 2's recipe) gets the student to a ROUGE-L of 31.2; the teacher is at 38.9 and the hard-label-only baseline at 27.4. Now seq-KD: run the teacher with beam search to generate a summary for each training document, fine-tune the student on those. ROUGE-L jumps to 34.6 — three and a half points over token-KD, because the targets are coherent full summaries and the training prefixes are fluent teacher text rather than position-wise averages. Mixing the two (seq-KD targets *plus* a token-KL term against the teacher's live logits on those same targets) nudges it to 35.1. The lesson generalizes: the longer and freer the output, the more sequence-level beats token-level. But notice we are still training on the *teacher's* prefixes — the student has not yet had to recover from its *own* mistakes. That is the next, larger step.

## 4. Forward vs reverse KL: mean-seeking vs mode-seeking, derived

This is the most important piece of math in the post, because the choice of KL direction quietly determines what kind of student you get, and most people copy Hinton's forward KL without realizing it is often the wrong choice for an LLM. Figure 2 is the picture to keep in mind as we derive it.

![A before-and-after diagram contrasting forward KL averaging over the teacher and forcing mode coverage against reverse KL averaging over the student and selecting a single sharp mode](/imgs/blogs/distilling-llms-and-reasoning-2.png)

Let $p$ be the teacher distribution and $q$ the student. The two KL divergences are:

$$
\mathrm{KL}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right] \quad \text{(forward — average over teacher)}
$$

$$
\mathrm{KL}(q \,\|\, p) = \sum_x q(x) \log \frac{q(x)}{p(x)} = \mathbb{E}_{x \sim q}\!\left[\log \frac{q(x)}{p(x)}\right] \quad \text{(reverse — average over student)}
$$

The whole difference is *which distribution you take the expectation over*. That single change of measure produces two qualitatively different behaviors. Let me derive each.

**Forward KL is mean-seeking (mass-covering).** Look at the forward expression: it is an average over $p$. Wherever the teacher places probability — $p(x) > 0$ — the term $p(x) \log \frac{p(x)}{q(x)}$ contributes. If $q(x) \to 0$ where $p(x) > 0$, then $\log \frac{p(x)}{q(x)} \to +\infty$ and the divergence blows up. So forward KL *punishes the student for putting near-zero probability anywhere the teacher puts mass*. The student is forced to "cover" every mode of the teacher; if the student lacks the capacity to be sharp everywhere, it spreads its probability to avoid the infinite penalty, smearing mass across modes and into the gaps between them. The result is a student that is too broad — it assigns non-trivial probability to sequences the teacher considers unlikely, and at generation time it can sample those low-quality continuations. This is "mean-seeking": the student tries to put its mean where the teacher's mass is, even at the cost of also covering low-quality regions.

**Reverse KL is mode-seeking (zero-forcing).** Now the reverse expression: it is an average over $q$. The term $q(x) \log \frac{q(x)}{p(x)}$ only contributes where the student itself places probability. Where $q(x) = 0$, the term is zero (with $0 \log 0 = 0$) — the student is *not* penalized for ignoring a region, as long as it puts no mass there. But if the student puts mass where the teacher does not ($q(x) > 0$ while $p(x) \to 0$), then $\log \frac{q(x)}{p(x)} \to +\infty$ and it is punished hard. So reverse KL *punishes the student for putting probability where the teacher does not*, but does not punish it for *ignoring* teacher modes. The optimal strategy under reverse KL is therefore: pick a region where the teacher is confident, concentrate your mass there, and stay zero everywhere else. The student locks onto one (or a few) high-quality modes and abandons the tail. This is "mode-seeking" or "zero-forcing": the student would rather be sharply correct on a subset of the teacher's behavior than blurrily cover all of it.

For an LLM you usually want **reverse KL**. The reason is exactly the failure mode forward KL creates: a small student does *not* have the capacity to faithfully cover all of a huge teacher's distribution, and when forced to (forward KL), it over-generalizes — it puts probability on degenerate or low-quality sequences in the gaps between the teacher's modes, and those leak out at sampling time as hallucinations and incoherence. Reverse KL instead lets the limited student concentrate on producing *some* of the teacher's good outputs really well, and crucially, it does not waste capacity modeling the teacher's tail. This is the core argument of **MiniLLM** (Gu et al., 2023, "Knowledge Distillation of Large Language Models"): replace the standard forward-KL distillation objective with reverse KL precisely to stop the student from over-estimating the void regions of the teacher's distribution, and they show it produces more accurate, more calibrated, lower-exposure-bias students than forward-KL KD.

There is a beautiful catch that ties this back to sequences. Reverse KL is an expectation **over the student** — $\mathbb{E}_{x \sim q}[\cdot]$. To compute it you have to *sample from the student*. Which means: to minimize reverse KL, the student must generate its own sequences and you score them. That is not a coincidence — it is the same thing as on-policy distillation, and it is why MiniLLM's training loop is a policy-gradient-style loop over student samples. Reverse KL and on-policy are two views of the same idea: train on what the student actually produces. The next section makes that loop explicit.

#### Worked example: the same student, two KL directions

Distill a 7B teacher into a 1.3B student on an open-ended instruction-following set, holding everything else fixed, and compare forward-KL token-KD against a reverse-KL on-policy objective. Forward KL gives a student that scores well on per-token perplexity against the teacher (it is *covering* the teacher) but, when you actually sample from it at temperature 1.0, produces a noticeably higher rate of incoherent or repetitive completions — a human-preference win rate against the teacher of around 18%. The reverse-KL student has *worse* teacher-perplexity (it deliberately ignores parts of the teacher) but a human-preference win rate around 24%, with visibly cleaner samples and fewer hallucinations. This is the forward-KL trap in one experiment: the objective that looks better on the matching metric produces the worse model to actually sample from. MiniLLM reports this pattern across model scales. The takeaway: for anything you will *sample from* (which is every generative LLM), optimize the divergence in the direction that matches how you will use it — reverse KL, over student samples.

## 5. On-policy distillation: student generates, teacher scores

Everything so far has trained the student on *someone else's* prefixes — ground truth (token-KD) or the teacher's generations (seq-KD). On-policy distillation is the regime where the student generates and the teacher only *scores*, and it is the cleanest fix for exposure bias because the training distribution becomes, by construction, the student's own inference distribution. Figure 3 is the loop.

![A dataflow graph of the on-policy distillation loop where the student samples a sequence, both student and teacher score those exact tokens, a reverse-KL loss combines the two, and the student updates](/imgs/blogs/distilling-llms-and-reasoning-3.png)

The loop, per step:

1. Take a batch of prompts. The **student** generates completions by sampling from its own policy, $y \sim q(\cdot \mid x)$. These are exactly the kinds of sequences the student produces at inference — including its characteristic mistakes.
2. Score those exact tokens two ways: under the **student** ($q$, with gradient) and under the **teacher** ($p$, no gradient — the teacher is frozen).
3. Form a **per-token reverse-KL** (or a more general $f$-divergence / distillation reward) between $q$ and $p$ on the student's sampled tokens.
4. Backprop into the student only. The teacher is a frozen scorer.

Because the sequences are the student's own samples, the student is now being corrected exactly on the states it actually visits — the off-distribution prefixes that token-KD never showed it. The teacher acts as a dense, per-token reward signal: "given this prefix that you yourself produced, here is how I would have weighted the next token." Over many steps the student's policy moves toward producing sequences the teacher scores highly, *starting from the student's own error-prone trajectories*. This is the train-equals-test fix.

Two important members of this family:

**MiniLLM (Gu et al., 2023)** minimizes reverse KL over student samples with a policy-gradient estimator. The gradient of reverse KL with respect to the student parameters $\theta$ has the form

$$
\nabla_\theta \mathrm{KL}(q_\theta \| p) = -\,\mathbb{E}_{y \sim q_\theta}\!\left[\sum_{t} \nabla_\theta \log q_\theta(y_t \mid y_{\lt t}, x)\cdot \Big(\log \tfrac{p(y_t \mid y_{\lt t},x)}{q_\theta(y_t \mid y_{\lt t},x)} - 1\Big)\right]
$$

which is a REINFORCE-style estimator where the "reward" at each token is the teacher–student log-ratio. MiniLLM adds variance-reduction tricks (a length-normalization, a single-step regularization, and teacher-mixed sampling) because, like all policy-gradient methods, the naive estimator is high-variance. The payoff is a student trained directly to minimize the divergence in the direction that matches sampling.

**Generalized Knowledge Distillation, GKD (Agarwal et al., 2023)** generalizes this with two knobs: (a) a mixture parameter $\lambda$ that interpolates between training on *fixed* data (seq-KD-style, teacher or ground-truth sequences) and *on-policy* student samples — $\lambda = 0$ is pure off-policy, $\lambda = 1$ is pure on-policy, and intermediate values mix them; and (b) a choice of divergence (forward KL, reverse KL, or the symmetric Jensen–Shannon / generalized JSD). GKD's headline finding is that *on-policy data is the dominant factor* — feeding the student its own samples and scoring with the teacher matters more than the exact divergence — and that mixing a little on-policy data into an otherwise off-policy run captures most of the benefit at lower variance. GKD is the practical recipe most people reach for now: it subsumes seq-KD ($\lambda=0$, forward KL) and MiniLLM-like training ($\lambda=1$, reverse KL) as special cases and lets you dial between them.

The cost is real and you should respect it: on-policy distillation requires the student to *generate* during training (expensive — generation is sequential and slow) and the teacher to *score* every step (a teacher forward pass per step). It is the most compute-hungry of the three regimes, often several times the cost of seq-KD. So the engineering question is always: do I need it? You need it when the output is long and free-form and exposure bias is hurting you — open-ended chat, long reasoning, agentic trajectories. You do not need it when seq-KD already hits target, which for many short or structured tasks it does.

### A pragmatic ladder

In practice the cost/benefit ladder is: **token-KD** (cheapest, short outputs, shared tokenizer) → **seq-KD / synthetic data** (the workhorse, coherent long targets, cross-tokenizer) → **on-policy / GKD with reverse KL** (most expensive, fixes exposure bias for long free generation). Start as low on the ladder as your accuracy target allows and only climb when the cheaper rung leaves you short. This mirrors the PTQ-before-QAT discipline from the [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) post: do the cheap thing first, escalate only when measured results force you to.

### Stress-testing the on-policy loop

It is worth reasoning through where on-policy distillation *breaks*, because the failure modes are subtle and they cost real GPU-hours to discover the hard way. Pose the problem: you have switched a long-form chat distillation from seq-KD to GKD with `lmbda=1.0, beta=1.0` (fully on-policy, reverse KL) and the run is slower than expected and the loss is jumpy. Reason through it step by step.

First, the **generation cost dominates**. Each step now requires the student to autoregressively decode a full completion before you can score it — and decoding is sequential, so it does not parallelize across the sequence the way a forward pass does. A 256-token completion is 256 sequential student forward passes per example, then one teacher forward pass to score, then one student forward pass for the gradient. That is why on-policy is "several times the cost of seq-KD": the generation is the tax, not the scoring. The fix is to generate with a fast inference path (key-value cache, batched generation, or a separate vLLM-style generation worker) and to keep `max_new_tokens` as short as the task allows. If your task's real outputs are 80 tokens, do not let the student ramble to 512 in training.

Second, the **gradient is high-variance**, which is the jumpy loss. The REINFORCE-style estimator weights each token's log-prob gradient by the teacher–student log-ratio "reward," and that reward can swing wildly across tokens and samples. Naive policy gradient over long sequences has variance that grows with length. This is exactly why MiniLLM adds length-normalization and a single-step regularizer, and why GKD's `lmbda < 1` (mixing in some off-policy fixed data) is not just a compute saver but a *variance* saver — the off-policy fraction is a low-variance anchor. The practical move when the loss is unstable: drop `lmbda` to 0.5, lower the learning rate, and add a small forward-KL term (`beta` toward 0.5, the symmetric JSD) which is better-behaved than pure reverse KL early in training.

Third — the nastiest one — **reward hacking the teacher signal**. Because the student is optimized to maximize the teacher's score on the student's own samples, it can find degenerate sequences the teacher happens to score highly but that are not what you want: repetitive safe phrasings, hedging boilerplate, or short low-risk completions. The student is gaming the proxy (teacher log-prob) instead of producing good text. Mitigations: regularize toward the reference policy (a KL-to-init term, exactly as in RLHF), cap the length reward, and *hold out a real eval set* — never trust the training reward as your quality metric, because the whole point of reward hacking is that the training reward goes up while quality goes down. The discipline is the same as in RL fine-tuning generally: the teacher is a reward model, and reward models get hacked.

The takeaway from the stress test: on-policy distillation is the most powerful regime *and* the most operationally demanding. If seq-KD plus a touch of on-policy mixing (`lmbda=0.25`) hits your target, that is the sweet spot — most of the exposure-bias benefit, a fraction of the variance and cost. Reserve full on-policy for the cases where long-horizon coherence genuinely fails without it.

## 6. Chain-of-thought distillation: transferring reasoning, not just answers

Now the part that matters most for the opening problem — getting a small model to *reason*. The key realization is that reasoning is not in the answer; it is in the *steps*. A model that outputs only the final answer "42" has learned a lookup; a model that outputs "there are 3 boxes, each with 14 apples, 3 × 14 = 42, so 42" has learned a *procedure* it can apply to new problems. Chain-of-thought (CoT) distillation transfers the procedure by training the student to produce the teacher's reasoning traces, not just its final answers. Figure 4 is the pipeline.

![A timeline of chain-of-thought distillation from collecting tasks with gold answers, to the teacher writing rationales, to filtering by answer correctness, to fine-tuning the student to produce reasoning then answer](/imgs/blogs/distilling-llms-and-reasoning-4.png)

The pipeline, step by step:

1. **Collect tasks** with known gold answers (math word problems, multi-hop QA, code with tests). The gold answer is your verifier — you will use it to filter.
2. **Prompt the teacher to reason**: "Let's think step by step." The teacher produces a rationale (the chain of thought) and a final answer for each task. This is the same chain-of-thought prompting that makes the teacher itself better at reasoning (Wei et al., 2022); here we are *harvesting* those rationales.
3. **Filter by answer correctness**: keep only the (rationale, answer) pairs where the teacher's final answer matches the gold answer. This is "rejection sampling" — you throw away the traces that reached the wrong answer, because a rationale that ends wrong is at best noisy and at worst a confidently-stated wrong procedure. (For harder problems you sample the teacher *multiple* times per question and keep all correct traces, which both filters and augments — this is the STaR / rejection-sampling idea, and it is how reasoning datasets are built at scale.)
4. **Fine-tune the student** on `question → rationale → answer`. The student learns to emit the reasoning *and then* the answer, so at inference it generates its own chain of thought before answering.

This is the recipe behind **"Distilling Step-by-Step"** (Hsieh et al., 2023). Their twist is multi-task: train the student to produce the rationale and the answer as two related outputs, and they show that with CoT distillation a *much smaller* student can match or beat a larger model trained on labels alone, using far less training data — they report a 770M T5 student outperforming a 540B PaLM on specific benchmarks when distilled with rationales, and matching standard fine-tuning with a fraction of the examples. The mechanism is exactly the "steps not answers" point: the rationale gives the student a richer, more sample-efficient training signal than a bare label, because each rationale teaches a *procedure* that generalizes, not a single input-output association.

This is also, not coincidentally, how the open reasoning models you have heard of were built. The pattern of "take a strong reasoning teacher, generate verified reasoning traces, fine-tune a small dense student on them" is precisely what produced the distilled small reasoning models in 2024–2025 — small models that punch far above their parameter count on math and code because they were trained on a large reasoning model's verified traces rather than on raw web text. The capability was *moved*, not grown from scratch.

### Why filtering is non-negotiable

The filtering step is where CoT distillation lives or dies, and skipping it is the most common way people get a bad reasoning student. An unfiltered teacher trace can be wrong in two ways: it can reach the wrong answer (caught by answer-matching), or — more insidiously — it can reach the *right* answer through *wrong* reasoning (a lucky guess, or a step that cancels its own error). The first is easy to filter; the second is hard and is the source of "the student learned to imitate broken reasoning that happens to land on the answer." Mitigations: sample multiple traces and prefer ones whose steps are consistent across samples; for math, check intermediate steps with a calculator or symbolic verifier, not just the final number; for code, run the tests. The rule of thumb: **a reasoning trace is only training data if you can verify it**, and the cheaper your verifier, the more aggressively you can scale the dataset. Math and code are popular distillation targets precisely because they have cheap automatic verifiers (the answer, the tests).

#### Worked example: CoT distillation closes the reasoning gap

Back to the opening problem. The 70B reasoning teacher gets 83% on the internal math set. The base 1.5B model fine-tuned on `question → answer` pairs only (no rationales) gets 34% — a little better than the 31% zero-shot base, but the answer-only signal does not teach the procedure. Now do CoT distillation: prompt the teacher "think step by step" on the training questions, keep only traces whose final answer is correct (this filters out roughly 25% of the teacher's traces and, by sampling 4× per question, yields about 3 verified traces per solvable question), and fine-tune the 1.5B student on `question → rationale → answer`. The student reaches 68% — a 34-point jump over the answer-only student, and 82% of the teacher's accuracy in a model 1/47th the size. The capability that quantization could never recover, distillation moved. This is the single result that justifies the whole post: a small model that *reasons*, because it was taught the steps. It will never beat the 83% teacher (section 7), but 68% in 1.5B parameters is a deployable model, and the next step is to shrink it to fit the phone.

## 7. The composition capstone: distill, then quantize, then prune

This is the section that makes this post the capstone of the optimization track. The three other levers — quantization, pruning, distillation — are not mutually exclusive; you stack them. But order matters, and getting the order wrong wastes accuracy you cannot get back. Figure 6 is the recommended order and Figure 5 (the synthetic-data flow) sits under the distillation step that starts it.

![A dataflow graph of synthetic-data distillation from seed prompts through teacher expansion, dedup and diversity filtering, quality verification, to a clean fine-tuning set and the trained student](/imgs/blogs/distilling-llms-and-reasoning-5.png)

The synthetic-data flow above is the front of the pipeline — it is how you build the dataset that the distillation step (whether seq-KD, on-policy, or CoT) consumes. Seed prompts in, teacher expands them, dedup and quality filters strip the bad rows, clean dataset out. Now the compound order.

![A timeline of the compound compression order from a large teacher, to a distilled small dense student, to quantize-aware distillation for int8, to 2:4 pruning with recovery, to the final edge model](/imgs/blogs/distilling-llms-and-reasoning-6.png)

**The recommended order: distill → QAT (int8/int4) → prune (2:4) → final recovery.** Here is the reasoning, and it is the same principle each time: *every lossy step should start from the most accurate checkpoint the previous step can produce, and any step that can recover accuracy should come after the steps that lose it.*

1. **Distill first.** Distillation is the only step that *adds* capability; it should run on the full-precision, dense student so it has maximum representational headroom to absorb the teacher. Distilling a model that is already quantized or pruned means you are teaching a crippled student — it has less capacity to receive the transfer. Get the capability in first, in fp16/bf16, dense.

2. **Quantize second — and quantize-aware, ideally jointly with distillation (KD+QAT).** Once you have an accurate dense student, make it int8 (or int4) robust. The crucial trick here is **quantize-aware distillation**, sometimes called KD-QAT: run the QAT fine-tune (fake-quant in the forward pass, straight-through estimator on the backward pass) but use the *distillation loss against the teacher* as the training objective instead of, or alongside, the ground-truth loss. The teacher provides a richer signal for the student to recover the accuracy lost to quantization noise than hard labels do. This is the natural marriage of two levers: QAT needs a training signal to recover accuracy, and the teacher is the best signal available. The mechanics of fake-quant and STE are in the [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) post; here the only change is *what loss you fine-tune against* — make it the KD loss.

3. **Prune third — after quantization, with a recovery fine-tune.** Add structured sparsity (2:4 on NVIDIA Sparse Tensor Cores, which gives real speedup; the [N:M sparsity](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores) post covers why 2:4 specifically) after the model is already accurate and quantization-robust, then do a short recovery fine-tune (again, ideally with the KD loss) to claw back the pruning damage. Pruning a model that has not yet been distilled means you prune capacity the model might have needed for the transfer; pruning before quantization can leave the quantizer with a harder, more concentrated weight distribution. For LLM-specific structured pruning (attention heads, FFN dimensions, layers) the methods and order are in the [pruning LLMs and transformers](/blog/machine-learning/edge-ai/pruning-llms-and-transformers) post.

4. **Final recovery distillation.** A last short KD fine-tune on top of the fully compressed model recovers the last fraction of accuracy lost across the lossy steps. Cheap, almost always worth it.

The intuition for the whole order — and "intuition" is the right word, this is a heuristic backed by experiment, not a theorem — is a *capacity budget* argument. Distillation fills the student's capacity with capability. Quantization and pruning both *remove* capacity (fewer bits, fewer weights). You want to add capability into a full tank, then carefully drain the tank while topping it up (recovery fine-tunes). If you drain first (quantize/prune) and add later (distill), you are pouring capability into a tank that is already half-empty and partly sealed — you transfer less. Mental model: distillation is the *fill* operation; quantization and pruning are the *compress* operations; recovery fine-tunes are *re-fill*. Fill, then compress with re-fills, in that order.

NVIDIA's Minitron work (Muralidharan et al., 2024) is the canonical real example of pruning-plus-distillation done right: structurally prune a large model along width and depth, then *distill the original model into the pruned one* as the recovery step, producing small models (e.g. an 8B from a 15B, a 4B from an 8B) at a fraction of the from-scratch training cost and with strong accuracy. The recovery-by-distillation step is doing exactly the "re-fill" job above. (Their detailed write-up is worth reading; in this repo there is a companion piece on the [Minitron pruning-and-distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) approach if you want the LLM-side framing.)

#### Worked example: the full stack on a phone target

Take the 68%-on-math CoT-distilled 1.5B student from section 6 (fp16, 3.0 GB) and run the full stack toward a Pixel-8-class NPU target. KD+QAT to int8 with the teacher's distillation loss as the QAT objective: accuracy holds at 67.6% (a 0.4-point quantization cost, much smaller than the ~2-point cost a naive PTQ would have taken, *because* the QAT used the teacher signal), size drops to 1.5 GB. Push to int4 weight-only (GPTQ/AWQ) for the embedding-and-attention-heavy layers with int8 elsewhere (mixed precision per the [mixed-precision and sensitivity](/blog/machine-learning/edge-ai/mixed-precision-and-sensitivity-analysis) post): 66.3%, 0.9 GB. The 2:4 pruning step is not worth it on this phone NPU — there are no sparse-tensor-core equivalents on the mobile accelerator, so 2:4 buys size but no speed there (it pays off on a Jetson Orin or a server GPU, not this phone), so we skip it for this target and note it as a deployment-dependent decision. Final state: a 0.9 GB int4 model, 66.3% on the math set (80% of the 83% teacher), running at roughly 18–22 tokens/s on the phone's NPU at batch 1 — interactive, offline, and 30+ points better than the 31% base model we started with. That is the opening problem, solved, by composing distillation with quantization. Measure it honestly: warm up the NPU for ~30 tokens before timing, report batch-1 (the real on-device case), and watch for thermal throttling on sustained generation — a phone that does 22 tok/s cold can sag to 14 tok/s after a minute of continuous decode.

## 8. The practical flow: runnable code

Enough theory. Here is real, copy-and-adapt code for the three regimes plus the KD+QAT step, in the actual Hugging Face / TRL toolchain. I will keep each snippet tight and idiomatic.

### Sequence-level KD: teacher generates, student fine-tunes

The simplest and most useful thing. Generate teacher outputs, then run ordinary SFT on them. This is a two-stage flow.

```python
# Stage 1: generate the distillation dataset with the teacher.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

teacher_id = "meta-llama/Llama-3.1-70B-Instruct"   # the big teacher
tok = AutoTokenizer.from_pretrained(teacher_id)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_id, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher.eval()

prompts = load_dataset("your/prompt-set", split="train")

def gen_target(example):
    msgs = [{"role": "user", "content": example["prompt"]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(teacher.device)
    with torch.no_grad():
        out = teacher.generate(
            ids, max_new_tokens=512,
            do_sample=False,            # greedy ~ the teacher's mode (seq-KD)
            temperature=None, top_p=None,
        )
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return {"prompt": example["prompt"], "response": text}

distill_ds = prompts.map(gen_target, remove_columns=prompts.column_names)
distill_ds.to_json("seq_kd_data.jsonl")   # this IS the distilled training set
```

```python
# Stage 2: fine-tune the small student on the teacher's outputs (TRL SFT).
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

student_id = "meta-llama/Llama-3.2-1B"     # the small student
student_tok = AutoTokenizer.from_pretrained(student_id)
student = AutoModelForCausalLM.from_pretrained(student_id, torch_dtype="bfloat16")

ds = load_dataset("json", data_files="seq_kd_data.jsonl", split="train")

def to_chat(ex):
    return {"messages": [
        {"role": "user", "content": ex["prompt"]},
        {"role": "assistant", "content": ex["response"]},
    ]}

ds = ds.map(to_chat)

cfg = SFTConfig(
    output_dir="student-seqkd",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,                 # low LR: we are imitating, not exploring
    bf16=True,
    packing=True,
)
SFTTrainer(model=student, args=cfg, train_dataset=ds, processing_class=student_tok).train()
```

That is sequence-level KD end to end. Note: greedy decoding for the teacher gives you the mode (true seq-KD); if you want a more diverse synthetic set, sample the teacher with `do_sample=True, temperature=0.7` and generate several responses per prompt — but then your diversity-and-quality filtering (next subsection) matters more.

### Token-level KD: KL on next-token distributions

When teacher and student share a tokenizer and you want the per-token soft signal, subclass the trainer and add a KL term against the teacher's logits.

```python
import torch
import torch.nn.functional as F
from trl import SFTTrainer

class TokenKDTrainer(SFTTrainer):
    def __init__(self, *args, teacher=None, alpha=0.5, temperature=2.0, **kw):
        super().__init__(*args, **kw)
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.alpha, self.T = alpha, temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs["labels"]
        out = model(**{k: v for k, v in inputs.items() if k != "labels"})
        student_logits = out.logits

        with torch.no_grad():
            teacher_logits = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits

        # hard-label cross-entropy (shifted for next-token prediction)
        sl = student_logits[:, :-1].reshape(-1, student_logits.size(-1))
        lb = labels[:, 1:].reshape(-1)
        ce = F.cross_entropy(sl, lb, ignore_index=-100)

        # soft forward-KL on next-token distributions, temperature T, masked
        mask = (lb != -100)
        p = F.log_softmax(teacher_logits[:, :-1].reshape(-1, teacher_logits.size(-1))[mask] / self.T, dim=-1)
        q = F.log_softmax(sl[mask] / self.T, dim=-1)
        kd = F.kl_div(q, p, reduction="batchmean", log_target=True) * (self.T ** 2)

        loss = (1 - self.alpha) * ce + self.alpha * kd
        return (loss, out) if return_outputs else loss
```

This is forward KL (`kl_div(q, p)` with the student as the "input" log-probs and teacher as the log-target gives $\mathrm{KL}(p\|q)$ — read the PyTorch convention carefully: `kl_div(input, target)` computes $\sum target\,(\log target - input)$, i.e. forward KL from `target` to `input`). For long free generation you would want reverse KL over student samples instead — which is the on-policy loop below.

### On-policy distillation with TRL's GKDTrainer

TRL ships an on-policy distillation trainer directly — it implements the GKD recipe (on-policy student samples scored by the teacher, with a divergence knob and a mixing parameter). This is the least error-prone way to run on-policy.

```python
from trl import GKDTrainer, GKDConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

student = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype="bfloat16")
teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype="bfloat16")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

ds = load_dataset("your/prompt-set", split="train")   # prompts only; student generates targets

cfg = GKDConfig(
    output_dir="student-gkd",
    lmbda=0.5,                # fraction of on-policy student samples per step (0=off-policy, 1=fully on-policy)
    beta=0.5,                 # JSD interpolation: 0 -> forward KL, 1 -> reverse KL, 0.5 -> symmetric JSD
    temperature=0.9,          # sampling temperature for student generations
    max_new_tokens=256,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    bf16=True,
)

GKDTrainer(
    model=student, teacher_model=teacher, args=cfg,
    train_dataset=ds, processing_class=tok,
).train()
```

Two knobs carry all the meaning. `lmbda` controls how much of each batch is the student's own on-policy samples (set it toward 1.0 when exposure bias is your problem; lower it to save compute). `beta` is the generalized-JSD interpolation between forward KL (`beta=0`) and reverse KL (`beta=1`) — push it toward 1 for the mode-seeking, sample-clean behavior LLMs usually want. Start with `lmbda=0.25, beta=0.5` (cheap, symmetric) and climb to `lmbda=1.0, beta=1.0` only if results demand it.

### CoT distillation: generate verified rationales, then SFT

```python
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct", torch_dtype=torch.bfloat16, device_map="auto").eval()
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")

math = load_dataset("openai/gsm8k", "main", split="train")   # has question + gold answer

def extract_answer(text):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1] if m else None

def harvest(example, n_samples=4):
    msg = [{"role": "user",
            "content": example["question"] + "\nThink step by step, then give the final number."}]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(teacher.device)
    gold = extract_answer(example["answer"])
    kept = []
    with torch.no_grad():
        for _ in range(n_samples):                 # sample several traces per question
            out = teacher.generate(ids, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95)
            trace = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            if extract_answer(trace) == gold:      # FILTER: keep only correct-answer traces
                kept.append(trace)
    return {"question": example["question"], "rationales": kept}

cot = math.map(harvest).filter(lambda e: len(e["rationales"]) > 0)
# flatten to (question -> rationale+answer) SFT rows, then run the Stage-2 SFTTrainer from above.
```

The filter is the load-bearing line. No filter, no reliable reasoning student. Sampling `n_samples` per question gives you both filtering (drop wrong traces) and augmentation (keep several correct ones), which is the rejection-sampling pattern that scales reasoning datasets.

### Synthetic-data quality filtering

The diversity and verification filters from Figure 5, sketched:

```python
from datasets import load_dataset
from rouge_score import rouge_scorer

raw = load_dataset("json", data_files="teacher_synthetic.jsonl", split="train")

# 1) DEDUP / diversity: drop near-duplicate instructions (ROUGE-L overlap > 0.7 with any kept one).
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
kept, seen = [], []
for ex in raw:
    instr = ex["prompt"]
    if all(scorer.score(instr, s)["rougeL"].fmeasure < 0.7 for s in seen[-2000:]):
        kept.append(ex); seen.append(instr)

# 2) QUALITY: drop rows that fail a cheap verifier (length bounds, refusals, broken format,
#    and for verifiable tasks, an answer/test check). Keep it strict; bad rows poison SFT.
def ok(ex):
    r = ex["response"]
    if len(r) < 20 or "I cannot" in r[:40]:           # refusals / empties
        return False
    return True

clean = [ex for ex in kept if ok(ex)]
print(f"kept {len(clean)} / {len(raw)} rows after diversity + quality filtering")
```

That is the whole practical toolkit: seq-KD (two-stage generate-then-SFT), token-KD (KL trainer), on-policy (GKDTrainer with `lmbda`/`beta`), CoT (verified-rationale harvest), and synthetic-data filtering. For the QAT step, take the distilled student and run the `prepare_qat` flow from the QAT post but swap the loss for the `TokenKDTrainer.compute_loss` above — that is KD+QAT in one change.

## 9. Results: what each method transfers, and the compound table

Figure 7 is the method comparison as a matrix; Figure 8 is the decision tree for the distill-or-quantize question. First the comparison.

![A comparison matrix of token-KD, sequence-KD, on-policy, and CoT distillation across what each transfers, its cost, when it wins, and its main risk](/imgs/blogs/distilling-llms-and-reasoning-7.png)

**Method comparison.** This table is the section-9 deliverable — what to reach for and what it costs.

| Method | What it transfers | Training cost | Wins when | Main failure mode |
| --- | --- | --- | --- | --- |
| Token-level KD | Full per-token distribution (dark knowledge) | Low (1 teacher fwd/step) | Short outputs, shared tokenizer, encoder/classifier-as-gen | Exposure bias on long outputs; needs shared vocab |
| Sequence-level KD | The teacher's mode (its greedy/beam outputs) | Medium (offline gen pass) | Long coherent outputs, cross-tokenizer, the synthetic-data workhorse | Mode collapse / low diversity of teacher outputs |
| On-policy / GKD | A robust policy over the student's own states | High (student gen + teacher score each step) | Open-ended chat, long reasoning, agentic trajectories | High variance; reward-hacking the teacher signal |
| CoT distillation | The reasoning *procedure* (verified traces) | Medium (teacher gen + verify/filter) | Math, code, multi-hop QA — anything with steps | Hallucinated or right-for-wrong-reasons rationales |

**The compound-compression result.** Stacking the levers on the section-7 worked example, toward a phone-class int4 target. All numbers are the running math-benchmark accuracy and on-device size; treat the latency as approximate batch-1 NPU figures.

| Stage | Model state | Accuracy (math set) | Size | Notes |
| --- | --- | --- | --- | --- |
| Base small model | 1.5B fp16, no distill | 31% | 3.0 GB | Capability not present |
| + CoT distillation | 1.5B fp16, verified traces | 68% | 3.0 GB | +37 pts — the capability transfer |
| + KD+QAT to int8 | 1.5B int8, teacher loss | 67.6% | 1.5 GB | −0.4 pt; QAT-with-teacher beats naive PTQ |
| + int4 (mixed prec.) | 1.5B int4/int8 mix | 66.3% | 0.9 GB | −1.3 pt; fits phone RAM, ~18–22 tok/s |
| (2:4 prune) | skipped on this NPU | — | — | No sparse cores on mobile; pays off on Jetson/GPU |
| Teacher (reference) | 70B, cloud | 83% | 35 GB int4 | The ceiling; cannot be exceeded |

The story the table tells: distillation did the heavy lifting (+37 points — the only step that adds capability), and quantization did the fitting (−1.7 points total, to go from 3.0 GB to 0.9 GB). Composed, they turn an undeployable-and-incapable starting point into a 0.9 GB model that reasons at 80% of a 70B teacher's accuracy and runs offline on a phone. Neither lever alone gets there: quantizing the 31% base just gives you a small, fast, *still-31%* model; distilling without quantizing gives you a capable 3.0 GB model that does not fit. The composition is the point.

## 10. Case studies: real numbers from the literature

Concrete, cited results so the methods above are grounded in shipped work.

**DistilBERT (Sanh et al., 2019).** Token-level distillation of BERT-base: a 6-layer student (66M params vs BERT's 110M), trained with a triple loss (masked-LM, distillation KL, and cosine embedding alignment). Retains ~97% of BERT-base's GLUE performance while being ~40% smaller and ~60% faster at inference. The canonical proof that token-KD works when there is no autoregressive exposure bias to fight — an encoder.

**Kim and Rush, sequence-level KD (2016).** On English→German and Thai→English translation, seq-KD let a small student recover most of the gap to a large teacher, and "seq-inter" (seq-KD plus token-KD on the teacher-generated targets) did best. The paper that established "train the student on the teacher's greedy outputs" as a method and gave it the mode-distillation theory.

**MiniLLM (Gu et al., 2023).** Reverse-KL distillation of GPT-2 and larger models. Across scales, reverse-KL students beat standard forward-KL KD on instruction-following quality, calibration, and exposure bias, with the gap *growing* as the teacher–student size ratio grows — exactly the regime (big teacher, small student) that matters for the edge. The paper that made "reverse KL, on student samples" the principled default for generative distillation.

**GKD (Agarwal et al., 2023).** Generalized KD with the on-policy mixture and divergence knobs. Headline: on-policy data is the dominant factor, and GKD with on-policy student samples outperforms supervised (off-policy) KD on summarization, translation, and arithmetic reasoning. The paper behind TRL's `GKDTrainer`.

**Distilling Step-by-Step (Hsieh et al., 2023).** CoT distillation with a multi-task rationale objective. A 770M T5 student, distilled with verified rationales, *outperformed* a 540B PaLM on specific benchmarks and matched standard fine-tuning with as little as 12.5% of the training examples. The sharpest demonstration that rationales are a more sample-efficient signal than labels — reasoning transfers as a procedure.

**NVIDIA Minitron (Muralidharan et al., 2024).** Structured pruning *plus* distillation recovery: prune a 15B model to 8B (and 8B to 4B) along width/depth, then distill the original into the pruned model to recover. Strong accuracy at a fraction of from-scratch training compute — the textbook case for "prune then distill-to-recover," and a direct instance of the compound order in section 7. (See the in-repo [Minitron write-up](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) for the LLM-side detail.)

**Alpaca-style synthetic distillation (Taori et al., 2023).** 52k instruction-following examples generated by a strong teacher from 175 seed instructions, used to turn a small base model into an instruction-follower for a few hundred dollars of generation cost. The case that put "the teacher writes the training set" on the map — and, via the licensing controversy that followed, the case that put teacher-output licensing on the map too (section 11).

## 11. Honest limits: what distillation cannot do

Every lever is a cost and a set of failure modes. Here are distillation's, stated plainly, because knowing when *not* to distill is half of using it well.

**You cannot exceed the teacher.** This is the hard ceiling and it is worth internalizing. Distillation moves capability *from* the teacher *into* the student; it does not create new capability. If the teacher gets 83%, no amount of distillation gets the student to 84% on the same distribution — the student's asymptote is the teacher, and in practice it lands below the teacher because the student has less capacity. (There are narrow exceptions: a student distilled from *multiple* teachers, or one whose verified-rationale filtering removes the teacher's errors, can occasionally beat a single teacher on a metric — but you do not get capability the teacher's ensemble never had.) The corollary: invest in the best teacher you can access, because the teacher is your ceiling. A weak teacher distills into a weak student, efficiently.

**Synthetic-data pitfalls: mode collapse and diversity loss.** When the teacher generates the training set, the set inherits the teacher's biases and, worse, *narrows*. Greedy/low-temperature teacher generation produces low-diversity data — the student sees a thin slice of the output space and overfits to the teacher's stylistic tics. Train a student on a model's outputs, then train a third model on *that* student's outputs, and across generations you get distributional collapse — the long tail vanishes, diversity craters, and the model gets blander and more error-prone (the "model collapse" / "recursion" failure documented by Shumailov et al., 2024). Mitigations: sample the teacher at higher temperature with multiple completions per prompt; mix in real human data; run the dedup/diversity filter from section 8; and never recursively distill student-on-student without fresh real data in the loop.

**Hallucinated rationales.** For CoT distillation specifically, the teacher can produce confident, fluent, *wrong* reasoning — and a student trained on it learns to be confidently wrong. The right-answer-wrong-reasoning case is the nasty one because answer-filtering does not catch it. This is why verification has to go as deep as you can afford: step-level checks for math, test execution for code, consistency across sampled traces. A reasoning student is only as trustworthy as the verifier that filtered its training traces.

**Licensing of teacher outputs.** This is a legal and policy constraint, not a technical one, and it has killed projects. Many commercial model providers' terms of service prohibit using their model's outputs to train a competing model. If you distill from a closed API teacher, you may be violating the terms you agreed to, and the resulting student may be encumbered. The Alpaca lineage ran straight into this. Before you build a distillation pipeline on a commercial teacher, read the terms; for anything you intend to ship or open-source, prefer an open-weights teacher with a license that permits distillation (many open models explicitly do). Engineering does not exempt you from the license.

**When to distill an LLM vs just quantize a small one.** This is the decision in Figure 8, and it is the most common real choice. The rule: **if a small open model already nearly has the capability you need, do not distill — just quantize (and maybe lightly fine-tune) it.** Distillation is a *training run* with a teacher, a generated dataset, filtering infrastructure, and (for on-policy) a generation-in-the-loop budget. That is a large, ongoing investment. Quantizing an existing small model is hours, no teacher, no dataset. So you only climb to distillation when *no* small model has the capability — when there is a genuine capability gap that quantization cannot touch (the opening problem: 31% base, need 65%+). And even then, only if you have a strong teacher and the budget. If you have the gap but no budget, the honest answer is often "use the teacher via API for now," not "distill a worse model you cannot afford to train."

![A decision tree for whether to distill or quantize an edge LLM based on whether a small model is already near target and whether there is a capability gap with budget to distill](/imgs/blogs/distilling-llms-and-reasoning-8.png)

The tree above collapses the decision. Already near target → quantize (GPTQ/AWQ int4), optionally a light LoRA SFT first. Real capability gap and you have a strong teacher plus training budget → distill (seq-KD + on-policy + CoT, in the practical ladder of section 5). Gap but no budget → use the teacher via API and revisit. The expensive path (distillation) is the *last* resort, not the first — same discipline as PTQ-before-QAT, same discipline as the whole [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook): cheap levers first, expensive levers only when measured results force the climb.

## 12. When to reach for this (and when not to)

A decisive section, because "it depends" is not engineering.

**Reach for distillation when:** there is a *capability gap* a small model cannot close on its own (reasoning, instruction-following, a domain skill), AND you have access to a strong teacher (open-weights ideally), AND you have the training budget. This is the only lever that injects capability, so it is the right tool exactly when capability — not size or speed — is the missing piece. Within distillation, reach for **seq-KD** as the default workhorse (cheap, robust, cross-tokenizer); add **on-policy/GKD** when long free-form outputs suffer from exposure bias; use **CoT distillation** whenever the target is a *reasoning* task with a verifier.

**Do not reach for distillation when:** a small model is already close (quantize it instead — far cheaper); when you cannot legally use the teacher's outputs (read the license); when you lack a verifier for a reasoning task (you will distill confident nonsense); or when the gap is so large that even the best small student cannot reach target (no 0.5B model will match a 70B teacher on hard reasoning — pick a bigger student or accept the cloud). And do not distill *recursively* on synthetic-only data across generations — diversity collapses.

**The composition rule:** distillation composes with the other levers in the order distill → quantize → prune, with recovery fine-tunes (ideally KD-loss) after each lossy step, because distillation is the only step that *adds* capacity and should run first on the full-precision dense model, while quantization and pruning *remove* capacity and should run after, on the most accurate checkpoint available. Skip any step whose hardware payoff is absent on your target (2:4 sparsity buys nothing on an NPU without sparse cores).

## Key takeaways

1. **Distilling a generative LLM is not classifier distillation.** There is no single label — the output is a sequence, and what you transfer is *capability* (reasoning, instruction-following), not a soft label. The autoregressive structure creates exposure bias, which is the problem every advanced method is solving.
2. **Three regimes, a cost ladder.** Token-KD (cheap, short outputs, shared tokenizer) → sequence-KD / synthetic data (the workhorse: teacher generates, student imitates) → on-policy / GKD (most expensive: student generates, teacher scores, fixes exposure bias). Start low, climb only when results force it.
3. **Reverse KL over student samples is usually right for LLMs.** Forward KL is mean-seeking and forces a small student to over-cover the teacher's tail, leaking hallucinations at sampling time; reverse KL is mode-seeking, lets the student be sharply correct on a subset, and — being an expectation over the student — *is* on-policy distillation.
4. **Reasoning lives in the steps, so distill the steps.** CoT distillation transfers the *procedure*: harvest the teacher's rationales, filter to verified-correct traces, fine-tune `question → rationale → answer`. Rationales are a far more sample-efficient signal than bare labels.
5. **Filtering is load-bearing.** Synthetic and CoT data must be deduped, diversity-filtered, and (for reasoning) verified. Unfiltered teacher output collapses in diversity and teaches confident wrong reasoning. A trace is training data only if you can verify it.
6. **Compose in order: distill → QAT → prune, with recovery fine-tunes.** Distillation adds capability and goes first on the full-precision dense model; quantization and pruning remove capacity and go after, on the most accurate checkpoint. KD+QAT (QAT with the teacher's distillation loss) recovers quantization accuracy better than hard labels.
7. **You cannot exceed the teacher.** The teacher is your ceiling; invest in the best one you can legally use. A weak teacher distills efficiently into a weak student.
8. **Distill only when capability is the gap.** If a small model is already close, quantize it — distillation is a whole training run and the expensive last resort, not the first move. And mind the license on teacher outputs.

## Further reading

- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015) — the original soft-label distillation and temperature.
- Kim, Rush, "Sequence-Level Knowledge Distillation" (2016) — distill the mode; train the student on the teacher's generated sequences.
- Gu et al., "Knowledge Distillation of Large Language Models" / MiniLLM (2023) — reverse-KL distillation over student samples for generative LLMs.
- Agarwal et al., "On-Policy Distillation of Language Models" / GKD (2023) — the on-policy mixture and divergence knobs; behind TRL's `GKDTrainer`.
- Hsieh et al., "Distilling Step-by-Step!" (2023) — CoT/rationale distillation; small students beating large ones with verified rationales.
- Sanh et al., "DistilBERT" (2019) — the canonical token-KD result (40% smaller, 60% faster, ~97% of GLUE).
- Muralidharan et al., "Compact Language Models via Pruning and Knowledge Distillation" / Minitron (2024) — prune then distill-to-recover, the compound order in practice.
- Shumailov et al., "The Curse of Recursion" / model collapse (2024) — what happens when you recursively train on synthetic data.
- TRL docs — `SFTTrainer` and `GKDTrainer` for the runnable flows above.
- Within this series: the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) (the four-lever frame), [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals) (the classifier case), [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) and [weight-only LLM quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) (the quantize step), [pruning LLMs and transformers](/blog/machine-learning/edge-ai/pruning-llms-and-transformers) (the prune step), and the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) (the capstone that composes all four levers).
