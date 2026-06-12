---
title: "Minitron: How NVIDIA Builds a Whole LLM Family by Pruning and Distilling One Model"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into NVIDIA's Minitron recipe — structured pruning across four axes, activation-based importance, distillation-only retraining, and teacher correction — that turns one pretrained model into a 2–4× smaller family for up to 40× fewer tokens."
tags: ["llm", "minitron", "nemotron", "pruning", "knowledge-distillation", "model-compression", "structured-pruning", "nvidia", "neural-architecture-search", "training-techniques", "llama", "efficiency"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

## When training a small model from scratch is the wrong move

Here is the reflex almost every team has when they need a 4B model: pretrain a 4B model. Spin up the cluster, point it at a few trillion tokens, wait a month, and you have your small model. It is the obvious thing to do, and it is usually the wrong thing to do — at least if you already have a good larger model in the same family.

NVIDIA's Minitron line is the clearest demonstration of the alternative. Starting from a single pretrained Nemotron-4 15B, they produce an 8B and a 4B sibling not by training them, but by *cutting the 15B down* and then *re-teaching the cut model with the original's own output distribution*. The 8B that falls out of this process scores **63.8 on MMLU** and matches Mistral-7B, Gemma-7B, and Llama-3-8B — models trained on an order of magnitude more tokens. The whole family — 15B, 8B, 4B — costs about **1.8× the compute of training the 15B alone**, versus the 3× you would pay to train all three from scratch. Each derived model uses up to **40× fewer training tokens** than a from-scratch model of the same size.

This is not magic, and it is not a free lunch — it is a specific engineering recipe with sharp rules about *what* to cut, *how* to decide what to cut, and *how* to put the model back together so it does not fall apart. This post is a tour of that recipe, drawn from the two papers that define it: ["Compact Language Models via Pruning and Knowledge Distillation"](https://arxiv.org/abs/2407.14679) (the Minitron paper) and ["LLM Pruning and Distillation in Practice"](https://arxiv.org/abs/2408.11796) (the Llama-3.1 / Mistral-NeMo follow-up). If you want the broader landscape of either technique on its own, this blog already has a [guide to pruning in LLMs](/blog/machine-learning/large-language-model/pruning-in-llm) and a [guide to knowledge distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm); Minitron is the place where the two stop being separate tricks and become one production pipeline.

The mismatch is worth stating bluntly, because the whole design follows from it:

| Question | The naive assumption | What Minitron shows |
|---|---|---|
| How do you get a 4B model? | Pretrain a 4B model on trillions of tokens | Prune a 15B to 4B, then distill |
| How much data do you need? | The full pretraining corpus, again | < 3% of it (≈ 1.4T → ~94B tokens of retraining) |
| What is the retraining objective? | Next-token cross-entropy on hard labels | The teacher's full soft distribution (KL) |
| Which is better at fixed size? | The from-scratch model | The pruned-and-distilled model wins MMLU by up to 16% |
| What does the family cost? | 3 full pretraining runs | 1 pretraining run + 2 cheap distillations (≈ 1.8×) |
| Do you need the original training data? | Yes | No — correct the teacher on a proxy corpus instead |

![Pipeline diagram: a Nemotron-4 15B pretrained teacher is pruned to about 8B then distilled on ~94B tokens to produce Minitron 8B at MMLU 63.8, which is then pruned again to about 4B and distilled on under 3% of the pretraining data to produce Minitron 4B at MMLU 58.6](/imgs/blogs/nvidia-minitron-pruning-distillation-1.webp)

The diagram above is the mental model for the entire post: you pay for *one* expensive pretraining run, and every smaller model in the family is a prune-then-distill hop off the previous one. Pruning decides which parameters survive; distillation re-teaches the survivors using the larger model as a teacher. The rest of this article is a walk through each box — what gets cut, how importance is measured, why distillation (not ordinary training) is the retraining objective, and the second-order tricks (teacher correction, iterative compression, lightweight architecture search) that separate a working recipe from a model that comes out lobotomized.

A senior framing to keep in your head throughout:

> Pretraining buys you a *distribution of capabilities* spread redundantly across billions of parameters. Pruning is the bet that most of that redundancy is dead weight; distillation is the insurance policy that buys back the capability the cut destroyed — cheaply, because the teacher hands you the answer key.

Why does this work at all, mechanistically? Large models are *over-parameterized on purpose*. The redundancy that makes them easy to optimize — many heads learning overlapping features, many MLP neurons firing for the same concept, middle layers each nudging the representation a little — is exactly the slack that pruning reclaims. A well-trained 15B contains, latent inside it, a very good 8B and a respectable 4B; the parameters of those smaller models are *already present*, just entangled with parameters you do not need. Pruning is the act of selecting the sub-network; distillation is the act of healing the seams where you cut. Everything below is the engineering of doing that selection and healing well enough that the result beats a from-scratch model of the same size.

## 1. The four axes you can cut

The first thing to internalize is that a transformer is not a monolith you shrink uniformly. It has **four structurally independent dimensions**, and each one is a separate pruning decision with a different cost-accuracy profile.

![Anatomy diagram of one transformer layer on the left — residual stream in, multi-head attention with 32 heads, MLP up-projection to d_ff, residual stream out — with four labeled cut axes on the right: embedding d_model 4096 to 3072, attention heads 32 to 24, MLP intermediate d_ff 14336 to 9216, and depth 32 layers to 16](/imgs/blogs/nvidia-minitron-pruning-distillation-2.webp)

Take a concrete model — Llama-3.1-8B — and name its dimensions:

- **Embedding / hidden width** ($d_{\text{model}} = 4096$): the width of the residual stream that flows through every layer. Cutting it is the most invasive change, because every weight matrix in the model has $d_{\text{model}}$ on one of its axes — embeddings, every attention projection, every MLP projection, the final unembedding. Trim $d_{\text{model}}$ and you trim *all* of them at once.
- **Attention heads** (32 heads $\times$ 128 dims, with grouped-query KV heads): you can drop whole heads, which trims the $W_Q, W_K, W_V, W_O$ projections. With GQA you must drop heads in a way that respects the KV-head grouping — you cannot orphan a query head from its key/value group, or you break the attention math.
- **MLP intermediate width** ($d_{\text{ff}} = 14336$): the up-projection dimension inside the feed-forward block. This is the single largest pile of parameters in most transformers — for Llama-3.1-8B the three MLP matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) account for roughly two-thirds of the per-layer weights — and it is usually the most forgiving axis to cut.
- **Depth** (32 layers): the number of transformer blocks stacked end to end. Removing layers is the bluntest cut and, as we will see, the one with the steepest accuracy penalty per parameter.

The first three are **width pruning** — they make every layer thinner while keeping the layer count. The fourth is **depth pruning** — it deletes whole layers and leaves the survivors at full width. Minitron scores all four axes, then keeps the top-k along each.

A critical distinction that trips people up: this is **structured** pruning, not the unstructured kind you may know from the [pruning-in-LLM literature](/blog/machine-learning/large-language-model/pruning-in-llm). Unstructured pruning zeroes individual weights and leaves you a sparse matrix of the *same shape* — which only runs faster if you have specialized sparse kernels, and on most hardware it does not. Structured pruning removes *entire rows, columns, heads, and layers*, so the result is a smaller **dense** model with smaller dense matmuls. That dense-shrink is what turns into real wall-clock speedup: the Llama-3.1-Minitron-4B runs at roughly **1.8× the throughput** of the 8B on an H100. Quantization, covered in the [quantization deep dive](/blog/machine-learning/large-language-model/quantization-in-llm), is orthogonal — it shrinks each weight's bit-width; pruning shrinks the *number* of weights. You can and should stack them.

Here is what the cut looks like as a concrete architecture diff — the kind of config object you would feed a compression tool:

```python
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    hidden_size: int        # d_model — residual stream width
    num_layers: int         # depth
    num_attention_heads: int
    num_kv_heads: int       # GQA groups
    ffn_hidden_size: int    # d_ff — MLP intermediate

source = TransformerConfig(    # Llama-3.1-8B
    hidden_size=4096, num_layers=32,
    num_attention_heads=32, num_kv_heads=8, ffn_hidden_size=14336,
)

target = TransformerConfig(    # Llama-3.1-Minitron-4B (width-pruned)
    hidden_size=3072,             # width axis 1: -25%
    num_layers=32,                # depth UNCHANGED for the width variant
    num_attention_heads=32,       # heads kept; head_dim implied by hidden_size
    num_kv_heads=8,
    ffn_hidden_size=9216,         # width axis 3: -36%, the biggest single saving
)

def param_ratio(a, b):
    def vol(c):                                       # weight volume per model
        attn = 4 * c.hidden_size * c.hidden_size
        mlp = 3 * c.hidden_size * c.ffn_hidden_size
        return c.num_layers * (attn + mlp)
    return vol(b) / vol(a)

print(f"{param_ratio(source, target):.2f}")           # ≈ 0.51 → ~8B shrinks toward ~4B
```

Notice the design choice baked into the target: the 4B width variant **keeps all 32 layers** and shrinks $d_{\text{model}}$ and $d_{\text{ff}}$ instead. That is not an accident — it is the single most important rule of thumb in the whole recipe, and §3 explains why.

### The axes interact, so prune them jointly

A subtlety: the four axes are *structurally* independent but *statistically* coupled. The importance of a given attention head is computed in the context of the current $d_{\text{model}}$; if you slash the residual width, the head-importance ranking you computed on the full-width model is now slightly stale. Minitron sidesteps this by estimating importance for **all axes simultaneously in one forward pass on the full model**, then applying the cuts together to reach the target architecture, rather than pruning one axis, re-measuring, pruning the next. The single-shot joint cut is both cheaper and — per the paper's ablations — no worse than the careful sequential version. We return to this "estimate once" principle in §2.

### Second-order optimization: cut the axes that are cheap to score, not the ones that are easy to cut

The naive instinct is to cut depth first, because "remove 8 layers" is conceptually simpler than "remove 25% of the channels in the residual stream." Resist it. The axes differ not only in their accuracy cost but in how cleanly you can *measure* importance on them. Width axes (heads, neurons, channels) expose clean per-unit activation signals you can rank in a single forward pass — thousands of units, each with a real-valued score, so the top-k cut is statistically stable. Depth importance is coarser and noisier — there are only 32 layers, so you have 32 data points, and dropping the wrong contiguous block is catastrophic in a way that dropping the wrong 5% of MLP neurons never is. The engineering lesson is to spend your pruning budget where the measurement is trustworthy.

## 2. Scoring what matters: activation-based importance

Once you know *what* you can cut, you need to know *which specific* heads, neurons, channels, and layers to cut. Minitron's answer is refreshingly cheap: run a small calibration set through the model **once**, watch the activations, and rank every unit by how much it actually moves.

![Dataflow graph: 1024 calibration samples feed a single forward pass with no gradients, which branches to three per-axis importance estimators — attention heads scored by L2 of head output, MLP neurons by activation times W1, embedding channels by LayerNorm activation — which merge into an aggregation step using L2 over the batch and mean over the sequence, then a final rank-and-keep-top-k step](/imgs/blogs/nvidia-minitron-pruning-distillation-3.webp)

The whole importance estimation runs on a **calibration set of 1024 samples**, in a **single forward pass**, with **no gradients**. That is the part that makes this practical: you are not doing an expensive Hessian computation (as second-order pruners like Optimal Brain Surgeon do) or a sensitivity sweep that retrains the model. You are reading activation magnitudes off one batch. The whole estimation costs less than a single training step.

The per-axis scoring rules, with $X$ the layer input activations on the calibration batch:

- **Attention heads.** Score head $i$ by the L2 norm of its contribution to the attention output, summed over the calibration data:
  $$F_{\text{head}}^{(i)} = \sum_{B, S} \big\lVert \text{Attn}\big(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}\big) \big\rVert_2$$
  Heads that barely move the output are cheap to delete.
- **MLP neurons.** Score neuron $i$ in the intermediate layer by its activation magnitude through the up-projection:
  $$F_{\text{neuron}}^{(i)} = \sum_{B, S} \big| X \cdot (W_1^{(i)})^\top \big|$$
- **Embedding / hidden channels.** Score channel $i$ by the magnitude of the post-LayerNorm activation on that channel:
  $$F_{\text{emb}}^{(i)} = \sum_{B, S} \big| \text{LayerNorm}(X)_i \big|$$

The subtle and important part is the **aggregation**. Each score is a sum over a batch dimension $B$ and a sequence dimension $S$, and *how* you reduce those two dimensions matters more than people expect. Minitron ablated the combinations and found a clear winner — take the **L2 norm across the batch axis** (so a unit that fires hard on a few examples is not washed out by the many examples where it is quiet) and the **mean across the sequence axis** (so importance is per-position-averaged, not dominated by long sequences):

| Batch reduction | Sequence reduction | Behavior | Outcome |
|---|---|---|---|
| mean | mean | washes out peaky units | weaker |
| mean | L2 | over-weights long sequences | weaker |
| **L2** | **mean** | keeps peaky units, length-normalized | **best (chosen)** |
| L2 | L2 | double-counts magnitude | unstable |

This "(batch = L2, seq = mean)" rule is one of the paper's explicit best practices, and it is the kind of detail that looks trivial and is actually the difference between a clean ranking and a noisy one. The intuition: a head that is silent on average but *essential* on the 5% of inputs where it fires (think a head that handles a rare syntactic construction) has a small mean and a large L2 — and you do not want to prune it. L2-over-batch protects exactly those units.

In code, importance estimation is a forward hook that accumulates statistics — no training loop, no backward pass:

```python
import torch

class ImportanceCollector:
    """Activation-based importance for MLP intermediate neurons.
    Aggregation: L2 over batch, mean over sequence (Minitron's best practice)."""
    def __init__(self, mlp_module):
        self.scores = None
        self._n = 0
        self.handle = mlp_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        act = inputs[0].detach().abs()                 # |activation|: [B, S, d_ff]
        seq_mean = act.mean(dim=1)                      # mean over sequence -> [B, d_ff]
        contrib = (seq_mean ** 2).sum(dim=0)           # sum of squares over batch
        self.scores = contrib if self.scores is None else self.scores + contrib
        self._n += seq_mean.shape[0]

    def finalize(self):
        self.handle.remove()
        return torch.sqrt(self.scores)                 # L2 over the accumulated batch

collectors = [ImportanceCollector(l.mlp.down_proj) for l in model.layers]
with torch.no_grad():                                  # one pass, no backward()
    for batch in calibration_loader:                   # ~1024 samples total
        model(batch)
neuron_importance = [c.finalize() for c in collectors]
keep = neuron_importance[0].topk(k=9216).indices       # 14336 -> 9216, per layer
```

### Depth importance is a different, coarser signal

Width units expose clean per-unit signals. Depth does not — you have one number per layer and 32 of them. Minitron offers two ways to score a layer:

| Metric | What it measures | Cost | When to use |
|---|---|---|---|
| **Perplexity (PPL) drop** | remove a layer, measure PPL degradation | one forward pass *per layer/group* | final validation of a depth cut |
| **Block Importance (BI)** | cosine distance between a layer's input and output | one forward pass for *all* layers | fast screening |

Block Importance is defined as

$$\text{BI}_i = 1 - \mathbb{E}\!\left[\frac{X_i^\top X_{i+1}}{\lVert X_i \rVert_2 \, \lVert X_{i+1} \rVert_2}\right]$$

A layer whose output is nearly identical to its input ($\text{BI} \approx 0$) barely transforms the representation and is a prime candidate to drop. BI is computable in a **single forward pass** for all layers at once, which is why it is the default screening tool — but as the case studies show, you always confirm a depth cut with a real downstream eval, because BI and PPL can agree and still be wrong about what the model *uses*.

![Bar chart of block importance across the 32 transformer layers of Llama-3.1-8B, grouped into pairs: importance is high at the first layers (0-9) and last layers (22-31) shown in blue, and dips into a low band across the middle layers 10-21 shown in red, annotated as the low-importance middle band that is safest to drop as a contiguous block](/imgs/blogs/nvidia-minitron-pruning-distillation-4.webp)

The profile above is the single most reusable empirical fact in depth pruning, and it shows up again and again across model families: **block importance is U-shaped.** The first few layers (which lift tokens out of embedding space and establish the basic features) and the last few layers (which shape the representation into something the unembedding can read) move the representation the most. The middle layers each make a small, incremental change — and because the residual stream means each layer's job is "add a small correction," a contiguous band of middle layers is the safest thing to remove. When the Llama-3.1 practitioners depth-pruned 8B toward 4B, they dropped a contiguous block of middle-to-late layers and validated the choice against downstream accuracy (more on that in the case studies).

```python
def block_importance(model, calib_batch):
    """One forward pass; BI_i = 1 - mean cosine(in_i, out_i) per layer."""
    bis = []
    with torch.no_grad():
        hs = model(calib_batch, output_hidden_states=True).hidden_states
    for i in range(len(hs) - 1):
        x_in, x_out = hs[i], hs[i + 1]                 # [B, S, d_model]
        cos = torch.nn.functional.cosine_similarity(x_in, x_out, dim=-1)
        bis.append(1.0 - cos.mean().item())            # low BI -> safe to drop
    return bis                                          # drop the lowest contiguous run
```

### Calibration data is a load-bearing input, not a formality

The 1024 calibration samples are not arbitrary. Importance is *conditional on the distribution you measure it on*. Calibrate a code model on prose and you will under-weight the heads and neurons that handle brackets, indentation, and identifiers — and then prune exactly the units the model needs in production. The calibration set should look like your deployment traffic. This is the same lesson the [quantization world learned with importance matrices](/blog/machine-learning/large-language-model/quantization-in-llm): the corpus you measure on silently decides what survives. Budget a little care here; a confidently-wrong pruning plan from bad calibration data is worse than no plan, because the short retrain will not fully undo a structurally bad cut.

### Why activation magnitude is a good-enough proxy

It is reasonable to ask why something as crude as "how big are the activations" works as well as it does, when the pruning literature has far more sophisticated criteria — gradient-times-weight saliency, second-order Hessian-based importance (Optimal Brain Surgeon and its descendants), and learned masks. The answer is partly empirical (Minitron's ablations show activation importance is competitive at this scale) and partly structural. In a well-trained transformer, a unit's activation magnitude is a reasonable stand-in for its causal contribution because the residual stream is *additive*: each head and each neuron writes its output into a shared stream by addition, so a unit that consistently writes a small-magnitude vector is, almost by definition, making a small contribution to the downstream computation. The expensive criteria try to estimate the *counterfactual* — "what happens to the loss if I remove this unit" — which is more accurate per-unit but costs gradients or a Hessian, and at billion-parameter scale that cost is prohibitive for the marginal gain. Minitron's design philosophy is consistent throughout: use the cheapest criterion that ranks units well enough, then spend the saved compute on the distillation that actually recovers accuracy. The criterion sets the ordering; the distillation does the healing; and a slightly imperfect ordering healed well beats a perfect ordering healed poorly. This is why the method is robust to using a simple importance metric — the retraining is forgiving of small ranking errors, but not of structurally catastrophic cuts, which is why the *coarse* decisions (how much depth, which contiguous block) get the careful downstream validation while the *fine* decisions (which 5% of neurons) ride on activation magnitude alone.

### Second-order optimization: estimate importance once, not iteratively

A tempting idea: prune a little, re-estimate importance on the smaller model, prune a little more, repeat. It feels more principled — surely importance shifts after you cut? Minitron tested exactly this and found **iterative importance estimation provides no measurable benefit** over single-shot estimation. Estimate once on the full model, decide your whole pruning plan, execute it in one shot, then spend all your compute on the retraining instead. This is a best practice precisely because it is counter-intuitive: the obvious "more careful" loop is wasted compute, and the compute is far better spent on distillation.

## 3. Width vs depth: the pruning decision

You have two ways to hit a target parameter count: make the network *thinner* (width) or *shallower* (depth). They land at the same size and behave very differently.

![Before-and-after comparison: on the left, depth pruning drops a contiguous layer block, keeps the same width per layer, and yields faster latency but a bigger accuracy hit; on the right, width pruning keeps all 32 layers, shrinks d_model, heads and d_ff per layer, and yields higher accuracy with memory-bound gains](/imgs/blogs/nvidia-minitron-pruning-distillation-5.webp)

The empirical rule from the Minitron paper is blunt and worth memorizing:

> **Below the ~15B scale, prefer width pruning over depth pruning.**

The reason is mechanistic. Depth is where sequential computation happens — each layer conditions on the fully-formed output of the previous one, so layers are doing genuinely different transformations in sequence. Remove a layer and you remove a *stage* of computation that the rest of the network was counting on; the damage is structural and the model has to re-route reasoning that used to be spread across more steps. Width, by contrast, is parallel redundancy: many heads and many neurons are computing overlapping or low-magnitude features, and trimming the weakest of them removes redundancy rather than a processing stage. Empirically, multi-step reasoning (the kind MMLU and GSM8K probe) degrades faster under depth pruning than under width pruning at the same parameter count.

The flip side is latency, and the latency argument is pure arithmetic. Autoregressive decode runs one token at a time, and each token must pass through every layer in sequence — there is no parallelism *across* layers within a single token. So single-stream decode latency is, to first order,

$$t_{\text{token}} \approx L \times t_{\text{layer}}$$

where $L$ is the layer count. Halve $L$ and you roughly halve the critical-path latency, which is why **depth pruning is the better lever when raw token latency is the hard constraint**. Width pruning shrinks $t_{\text{layer}}$ (smaller matmuls, less memory traffic per token — which matters a great deal in the memory-bound decode regime, see the [KV cache deep dive](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management)) but moves single-stream latency less than dropping whole layers does.

| Axis | What it cuts | Accuracy cost | Latency win | Throughput win | When to prefer |
|---|---|---|---|---|---|
| **Width** | $d_{\text{model}}$, heads, $d_{\text{ff}}$ | Low (trims redundancy) | Modest | Strong | Default below ~15B; accuracy-sensitive |
| **Depth** | Whole layers | High (removes a stage) | Strong | Modest | Hard latency budget; willing to retrain harder |
| **Both** | Mix | Tune the blend | Both | Both | Aggressive 3–4× compression |

In practice the production recipe blends them: prune mostly width, take a little depth if the latency target demands it, and let the architecture search (§6) find the exact blend. Tooling makes this a config choice rather than custom code. With NVIDIA's TensorRT Model Optimizer / NeMo, pruning a checkpoint is a CLI invocation that names the target dimensions and the calibration data. The width-prune invocation:

```bash
python -m modelopt.torch.prune.plugins.mcore_minitron \
    --model        Llama-3.1-8B \
    --calib-data   calib_1024.jsonl \
    --calib-size   1024 \
    --target-hidden-size           3072 \
    --target-ffn-hidden-size       9216 \
    --target-num-attention-heads   32 \
    --target-num-layers            32 \
    --importance-aggregator        "l2_batch_mean_seq" \
    --out          minitron-4b-width-pruned/
```

The depth-prune variant keeps the width and drops 16 layers chosen by block importance:

```bash
python -m modelopt.torch.prune.plugins.mcore_minitron \
    --model        Llama-3.1-8B \
    --calib-data   calib_1024.jsonl \
    --target-hidden-size           4096 \
    --target-ffn-hidden-size       14336 \
    --target-num-layers            16 \
    --depth-importance-metric      "block_importance" \
    --out          minitron-4b-depth-pruned/
```

### Second-order optimization: pruning is the cheap half; budget for the distillation

A trap teams fall into: they obsess over the pruning decision (which heads, which layers) and treat the retraining as an afterthought. The opposite is correct. Pruning is a few forward passes; the retraining is where 99% of the compute goes and where 99% of the recovered accuracy comes from. A *mediocre* pruning plan followed by a *good* distillation beats a *perfect* pruning plan followed by naive fine-tuning. The pruning decision sets the ceiling on what is recoverable; the distillation decides how close you get to that ceiling. The next section is the distillation.

## 4. Retrain by distillation, not by data

A freshly pruned model is broken. You deleted a quarter of its channels and rewired its dimensions; its accuracy has fallen off a cliff — a model that scored 65 on MMLU might be down near random after the cut. The question is how to put it back together. The naive answer is "fine-tune it on more data with the usual next-token loss." Minitron's answer — and the reason the recipe works at all — is **retrain it by distilling from the unpruned model**, using the teacher's full output distribution as the target.

![Graph showing distillation loss wiring: a frozen corrected 8B teacher and a trainable pruned student both feed two loss nodes — logit KL with temperature tau, and a combined hidden-state plus embedding MSE through a linear upscaler — which sum into a total loss of L_CLM plus KL plus alpha times L_is](/imgs/blogs/nvidia-minitron-pruning-distillation-6.webp)

The losses that wire the student to the teacher:

- **Logit KL divergence** — the primary signal. Match the student's output distribution to the teacher's, over the full vocabulary, at a softmax temperature $\tau$:
  $$L_{\text{logits}} = \frac{1}{l} \sum_{k=1}^{l} \text{KL}\big(p_t^{(k)}(x, \tau) \,\big\|\, p_s^{(k)}(x, \tau)\big)$$
  where $p_t$ and $p_s$ are the teacher and student probabilities at token position $k$ and $l$ is the sequence length. This is the part that does the heavy lifting: instead of a one-hot target ("the next token is *cat*"), the student sees the teacher's entire belief ("82% *cat*, 9% *kitten*, 4% *feline*, …"), which is a vastly richer signal per token.
- **Intermediate-state (hidden) loss** — match the student's hidden states to the teacher's, layer-block by layer-block. Because the pruned student is *narrower* than the teacher, you cannot compare hidden states directly; you insert a **learned linear upscaler** $P \in \mathbb{R}^{d_s \times d_t}$ that projects the student's $d_s$ hidden state up to the teacher's $d_t$ before computing the loss. When depth changed too, you also need an **N-to-M layer mapping** that says which student layer is supervised by which teacher layer.
- **Embedding loss** — match the embedding-layer outputs, the same way.

These combine into a single objective:

$$L = L_{\text{CLM}} + L_{\text{logits}} + \alpha \cdot L_{\text{is}}$$

where $L_{\text{CLM}}$ is the ordinary cross-entropy on hard labels (it keeps the student grounded in the real next token, not only the teacher's opinion of it), $L_{\text{is}}$ bundles the intermediate-state and embedding losses, and $\alpha$ is set **dynamically** to $L_{\text{logits}} / L_{\text{is}}$ so the two distillation terms stay balanced as their raw magnitudes drift during training. Hand-tuning a fixed $\alpha$ is fragile; the ratio rule keeps the intermediate-state loss from either dominating or vanishing as the student learns.

Why does this beat ordinary training so decisively? Two reasons. First, **information density per token**. A hard label carries $\log_2(\text{vocab})$ bits of supervision — for a 128k vocabulary, about 17 bits, almost all of it the single answer. The teacher's full distribution carries the entire *shape* of the model's uncertainty: which alternatives were plausible, by how much, and how the probability mass is spread. That is orders of magnitude more signal per token, which is precisely why distillation recovers accuracy in $10^{11}$ tokens where from-scratch training needs $10^{13}$. Second, **the teacher is a perfect, infinitely-available annotator** of exactly the distribution you care about. You are not hoping that more data teaches the student the right behavior; you are copying behavior from a model that demonstrably already has it.

A note on **temperature**. The $\tau$ in the softmax flattens the distribution: $\tau > 1$ sharpens the contrast among the *non-top* tokens, surfacing the "dark knowledge" in the tail (the relative ordering of the 2nd, 3rd, 10th most likely tokens) that a low temperature would crush to near-zero. The gradient is scaled by $\tau^2$ to keep its magnitude comparable across temperatures. Why KL and not, say, MSE on the logits? Because KL on probabilities is the natural divergence between distributions and it weights mismatches by where the probability mass actually is, whereas MSE on raw logits over-penalizes differences in the irrelevant tail. Here is the loss in code — temperature-scaled logit KL plus a projected intermediate MSE:

```python
import torch
import torch.nn.functional as F

def distillation_loss(student_out, teacher_out, hard_labels,
                      upscaler, tau=2.0):
    s_logp = F.log_softmax(student_out.logits / tau, dim=-1)   # 1) logit KL
    t_p    = F.softmax(teacher_out.logits / tau, dim=-1)
    l_logits = F.kl_div(s_logp, t_p, reduction="batchmean") * (tau ** 2)

    s_hidden = student_out.hidden_states[-1]                    # 2) intermediate MSE
    t_hidden = teacher_out.hidden_states[-1]                    #    project narrow student up
    l_is = F.mse_loss(upscaler(s_hidden), t_hidden)            #    upscaler: d_s -> d_t

    l_clm = F.cross_entropy(                                    # 3) hard-label grounding
        student_out.logits.flatten(0, 1), hard_labels.flatten())

    alpha = (l_logits / l_is).detach()                         # dynamic balance
    return l_clm + l_logits + alpha * l_is
```

### Which losses to actually switch on

You do not always use all three losses. Minitron's rule depends on **whether you changed depth**:

![Matrix showing which distillation losses to use: for depth-unchanged width-only pruning, use logit KL and skip both hidden-state MSE and embedding loss; for depth-reduced pruning with layers dropped, use logit KL and add both hidden-state MSE and embedding loss](/imgs/blogs/nvidia-minitron-pruning-distillation-7.webp)

- **Depth unchanged (width-only pruning):** logit KL alone is enough. The student has the same layer structure as the teacher, the representations stay aligned layer-for-layer, and the extra intermediate-state losses buy nothing. This is the common case (because §3 says prefer width), so most Minitron models are trained with **logit-only distillation** — simpler, fewer hyperparameters, and no upscaler to fit.
- **Depth reduced (layers dropped):** now the student's layer $j$ does not correspond to the teacher's layer $j$, so the representations drift apart through the stack. Add the **intermediate-state and embedding losses** (with the learned upscaler and an N-to-M student-to-teacher layer mapping) to pull the surviving layers back into alignment with the teacher's computation. Without this, a depth-pruned student's hidden states wander off-manifold and the logit loss alone struggles to pull them back.

This is a clean example of the recipe's discipline: it does not throw every loss at every model. It uses the minimum supervision the structural change requires, and no more.

### The upscaler and the N-to-M layer mapping, in practice

Two implementation details deserve a closer look because they are where depth-pruned distillation actually gets fiddly. The **upscaler** is a single learned linear map $P \in \mathbb{R}^{d_s \times d_t}$ (no bias, no nonlinearity) trained jointly with the student; its only job is to lift the student's narrower hidden state into the teacher's space so an MSE is well-defined. It is thrown away after training — it exists purely to make the intermediate-state loss computable, not to ship in the final model. Because it is linear and small, it adds negligible compute and converges fast.

The **N-to-M mapping** answers "which teacher layer supervises which student layer" when the student has fewer layers than the teacher. The natural choice is a uniform spacing — if the student kept layers and the teacher had twice as many, you map student layer $j$ to teacher layer $2j$ — so the student's surviving layers are pulled toward evenly-spaced waypoints in the teacher's computation. The subtlety is that the mapping should respect *which* layers you kept: if you dropped a contiguous middle block, the surviving early and late layers should be supervised by the teacher's corresponding early and late layers, so the supervision is consistent with the structural cut rather than fighting it. Get this mapping wrong and the intermediate-state loss actively pulls the student's representations toward states the surviving architecture cannot produce, which is worse than no intermediate loss at all. This is the deeper reason §4's matrix says to skip the intermediate losses entirely when depth is unchanged: with the same layer structure, the mapping is the identity and the whole question evaporates — another instance of the recipe doing the minimum the structural change requires.

### Second-order optimization: retrain *exclusively* with distillation, and use a real LR schedule

A best practice that surprises people: **retrain with the distillation loss instead of conventional training**, not in addition to a long conventional phase. There is no "first do normal fine-tuning, then distill" — the distillation loss *is* the retraining objective from step one. The lightweight retraining uses a real schedule — in the Llama-3.1 recipe, peak LR $1\times10^{-4}$, min LR $1\times10^{-5}$, 40-step linear warm-up, cosine decay, global batch size 1152. The original Minitron paper used a cosine decay from $\approx 2^{-4}$ down to $\approx 4.5\times10^{-7}$. The point is that distillation retraining is *real* training with a *real* schedule, just radically shorter — on the order of $10^{11}$ tokens instead of $10^{13}$.

## 5. Teacher correction: align the teacher first

There is a failure mode that bites every team that tries this without the original training data, and Minitron's fix for it is one of the most quietly important parts of the recipe. You almost never have the exact corpus the base model was trained on — Llama-3.1 and Mistral-NeMo ship weights, not data. So you distill on *your* corpus, a proxy. And the proxy is, by definition, off-distribution relative to what the teacher was trained on.

![Before-and-after comparison: on the left an uncorrected teacher trained on the original corpus sees off-distribution distillation data and emits noisy soft labels that produce a weaker student; on the right a corrected teacher is lightly fine-tuned on the distillation data, matches the new distribution, and emits clean targets that produce a stronger student](/imgs/blogs/nvidia-minitron-pruning-distillation-8.webp)

The problem: a teacher evaluated on data it was not trained on produces **miscalibrated soft labels**. Its probabilities are subtly wrong on your corpus — over-confident in the wrong places, under-confident in others — and since those soft labels *are* the student's training target, you are distilling the teacher's distribution shift straight into the student. The paper's own words: "without correcting for the distribution shift, the teacher provides suboptimal guidance on the dataset when being distilled."

The fix, called **teacher correction**, is almost embarrassingly simple: **lightly fine-tune the unpruned teacher on your distillation corpus first**, before you prune anything. In the practice paper, this meant fine-tuning the full 8B (or 12B) on ~94B tokens of the target dataset, so its output distribution matches the data the student will actually see. Then prune the corrected teacher, and distill the student from it. The teacher now gives clean, on-distribution targets.

The sequence matters — teacher correction must happen *before* importance estimation, because you want to measure importance and prune on the model whose distribution matches the retraining corpus:

1. Fine-tune the unpruned teacher on the (proxy) distillation corpus — *teacher correction*.
2. Estimate importance and prune the corrected teacher to the target architecture.
3. Distill the pruned student from the corrected teacher on the same corpus.

```python
teacher = load_pretrained("Llama-3.1-8B")
teacher = finetune(teacher, distill_corpus, tokens=94e9, lr=1e-5)            # 1) teacher correction
student = prune(teacher, target=target_4b_config, calib=calib_1024)         # 2) prune corrected teacher
student = distill(student, teacher, distill_corpus, loss=distillation_loss) # 3) distill on same corpus
```

### Second-order optimization: teacher correction is what makes "no original data" viable

It is worth being explicit about why this unlocks the whole practical workflow. The original Minitron paper had access to Nemotron-4's training data, so distribution shift was a non-issue — they could distill on the real corpus. The follow-up paper's contribution was showing you can apply the recipe to *someone else's* model — Llama, Mistral — that you only have as weights. Teacher correction is the bridge: it converts "I do not have the original corpus" from a blocker into a one-step fine-tune. Without it, the off-distribution teacher poisons the student and the whole approach degrades to something barely better than ordinary fine-tuning. This single trick is the difference between "a method NVIDIA can use on its own models" and "a method anyone can use on any open-weights model."

## 6. Iterate the family, don't one-shot it

If you want a 4B from a 15B, you have two routes: prune 15B straight to 4B in one shot, or go 15B → 8B → 4B in two hops, distilling at each. The two-hop route wins, and not by a little.

![Branching graph: a 15B teacher distills down to an 8B intermediate, which distills down to a 4B iterative model with higher MMLU; in parallel the 15B teacher prunes straight to a 4B one-shot model with lower MMLU, illustrating that the two-hop iterative path wins by about 12 percent MMLU](/imgs/blogs/nvidia-minitron-pruning-distillation-9.webp)

The Minitron paper measured it: going **15B → 8B → 4B iteratively yields about a 12% improvement in MMLU** over pruning 15B → 4B in a single shot, at the same final 4B size. The reason is the **teacher–student gap**. Distillation works best when the teacher and student are not too far apart in capacity — the student can actually track a teacher that is 2× its size, but a teacher nearly 4× its size is teaching a curriculum the small student cannot follow well. Each hop keeps the gap small: the 8B learns from the 15B (a ~2× gap), and the 4B learns from the 8B (a ~2× gap), rather than the 4B trying to swallow the 15B whole. The intermediate model is a stepping-stone teacher, not wasted work — and you wanted the 8B anyway.

This dovetails with a lightweight **architecture-search** step that the recipe folds in. For a given target size, there is not one architecture — there are many ways to spend a 4B parameter budget (more layers and narrower, or fewer layers and wider, different head counts, different $d_{\text{ff}}$/$d_{\text{model}}$ ratios). Minitron's approach:

1. **Enumerate candidate architectures** within the parameter budget (typically within ±5% of the target), respecting feasibility constraints (head dimensions must divide evenly, GQA groups must stay intact). There are usually on the order of 15–18 viable candidates per target size.
2. **Lightweight retrain** each candidate — only ~1.8 billion tokens (≈ 400 steps) — just enough to **stabilize the ranking**.
3. **Pick the best-performing candidate** after that short retrain, then do the full distillation only on the winner.

The insight that makes this affordable is that **candidate rankings stabilize early** — after a few hundred steps of retraining, the *relative ordering* of architectures is reliable, even though *absolute* accuracy is still climbing for all of them. So you spend ~1.8B tokens to rank, not the full ~94B, and only pay full price for the architecture you keep. This is a poor man's neural architecture search: instead of training every candidate to convergence (impossibly expensive) or using a learned predictor (complex), you exploit the empirical fact that the leaderboard settles long before the models do.

```python
candidates = enumerate_architectures(target_params=4.0e9, tolerance=0.05)   # ~15–18 feasible

ranked = []
for arch in candidates:
    student = prune(corrected_teacher, target=arch, calib=calib_1024)
    student = distill(student, corrected_teacher, corpus, tokens=1.8e9)      # ~400 steps to rank
    ranked.append((eval_proxy(student), arch))

best_arch = max(ranked)[1]
final = prune(corrected_teacher, target=best_arch, calib=calib_1024)
final = distill(final, corrected_teacher, corpus, tokens=94e9)               # full retrain on winner
```

### Second-order optimization: prune from the final pretraining checkpoint

One more best practice that matters if your base model was trained in multiple phases (e.g., a general phase then a high-quality annealing phase, as most modern models are): **prune and retrain starting from the final training-stage checkpoint**, not an earlier one. The last phase shapes the model's most useful behaviors — the high-quality data, the long-context extension, the instruction priors — and pruning from before it throws away exactly the capability you most want to preserve. Compress the finished model, not a midpoint.

## 7. The economics: 40× fewer tokens, 1.8× less compute

Step back and add up the bill, because the economics are the entire reason this technique exists.

![Before-and-after cost comparison: on the left, training each size from scratch is three full pretraining runs for 15B, 8B and 4B; on the right, the Minitron approach is one pretraining run for the 15B plus a roughly 94 billion token distillation for the 8B plus an under-3-percent-data distillation with 40 times fewer tokens for the 4B](/imgs/blogs/nvidia-minitron-pruning-distillation-10.webp)

The headline numbers from the Minitron paper and its follow-up:

| Quantity | From scratch | Minitron | Saving |
|---|---|---|---|
| Tokens per derived model | full pretrain (~10–15T) | ~94B retraining | up to **40×** fewer |
| Family compute (15B + 8B + 4B) | 3 full runs | 1 run + 2 distillations | **1.8×** total (vs 3×) |
| Retraining data fraction | 100% | **< 3%** | ~33× less data |
| MMLU vs from-scratch (same size) | baseline | **+up to 16%** | net accuracy *gain* |

And the resulting models are not compromises. A few concrete results:

- **Minitron-8B** (from Nemotron-4 15B): MMLU **63.8**, comparable to Mistral-7B, Gemma-7B, and Llama-3-8B, while using a fraction of their tokens.
- **Minitron-4B**: MMLU **58.6**, outperforming similarly-sized from-scratch models.
- **Llama-3.1-Minitron-4B** (width-pruned from Llama-3.1-8B): MMLU **60.5** (5-shot), beating Phi-2-2.7B, Gemma2-2.6B, and Qwen2-1.5B on MMLU; it used only ~1.4T tokens of retraining versus the 15T the original 8B saw, and runs at roughly **1.8× the throughput** of the 8B on an H100 at batch 64.
- **MN-Minitron-8B** (width-pruned from Mistral-NeMo-12B): state-of-the-art for an 8B at release, beating Llama-3.1-8B across many benchmarks while training on a small fraction of the tokens.

It is worth doing the arithmetic concretely, because the multipliers compound in a way that is easy to under-appreciate. Suppose training a model from scratch costs roughly $C \propto N \times D$ where $N$ is parameters and $D$ is tokens (the compute-optimal scaling intuition). To get the 15B + 8B + 4B family from scratch, you pay three full $N \times D$ bills. With Minitron you pay the 15B bill once, then each derivation is a distillation on ~94B tokens — a token budget roughly $40\times$ smaller than a from-scratch run of that size. So the 8B derivation costs on the order of $\frac{8}{15} \times \frac{1}{40}$ of the 15B run, and the 4B derivation a bit less; summed, the two derivations add only ~0.8× the cost of the single 15B run, landing the whole family near **1.8×** instead of **3×**. And that arithmetic *understates* the win, because it counts only compute: from-scratch runs also carry data-engineering cost, hyperparameter risk, and the wall-clock latency of N sequential month-long jobs, none of which the cheap derivations incur. The distillations are short enough to run in parallel off the same checkpoint, so the family ships on roughly the timeline of the single big run.

The strategic value compounds when you want a *family*. Inference deployments need a menu of sizes — a 4B for edge and latency-critical paths, an 8B for the main serving tier, the 15B for quality-sensitive work. Training that menu from scratch is N independent megaprojects, each with its own data pipeline, its own month on the cluster, its own risk of a failed run. Minitron makes it one megaproject plus N cheap derivations, and the derivations are *better* than the megaprojects would have been at those sizes. That is the actual product: not a single small model, but the economic ability to ship a whole Pareto frontier of sizes from one pretraining investment. The big run is the capital expense; the family is nearly free marginal cost.

### Second-order optimization: stack compression techniques

Pruning-and-distillation is one axis of compression; it composes with the others. After you have a Minitron-4B, you can [quantize it to 4-bit](/blog/machine-learning/large-language-model/quantization-in-llm) for another ~3–4× memory reduction, and the two are largely independent — pruning removes parameters, quantization shrinks the ones that remain. A pruned-then-quantized model is how you fit a capable LLM on an edge device. The same logic extends to architecture: the latest Nemotron models combine pruning with [Mixture-of-Experts and hybrid attention](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies), distilling a dense or hybrid teacher into a smaller hybrid student. The recipe is a building block, not a destination.

Think of the compression axes as a small toolbox, each acting on a different lever: pruning removes parameters, quantization shrinks each parameter's bits, distillation transfers behavior, and architecture changes (MoE, hybrid SSM-attention) alter how the parameters are organized in the first place. They are not competitors — they are composable, and the strongest production systems reach for several at once. Minitron's specific contribution is making the *pruning + distillation* pair reliable enough to be a default move rather than a research gamble, which is what lets it sit at the bottom of that stack as the step you do first, before quantization and before deployment.

## 8. Failure modes the recipe is built to avoid

It is easy to read a clean recipe and forget that every step exists because something goes wrong without it. Here are the failure modes, each mapped to the safeguard that prevents it — this is the "why" behind the rules above, and the checklist to run when your own compressed model comes out worse than expected.

- **Pruning the wrong units (bad calibration).** Symptom: the model loses a specific capability (code, math, a language) that the calibration set under-represented. Cause: importance was measured on a distribution that did not include that capability's inputs, so the units handling it scored low and got cut. Safeguard: calibrate on data that mirrors deployment traffic; when in doubt, over-include the rare-but-critical slices.
- **Distilling distribution shift (no teacher correction).** Symptom: the student is mysteriously worse than a plain fine-tune, even though distillation "should" help. Cause: the uncorrected teacher emits miscalibrated soft labels on the proxy corpus, and you faithfully copy its errors. Safeguard: teacher correction (§5) — fine-tune the teacher on the corpus first.
- **Over-aggressive single-shot compression.** Symptom: a 4× cut in one hop never recovers, no matter how long you distill. Cause: the teacher–student gap is too large; the student cannot track a teacher 4× its size. Safeguard: iterate through an intermediate size (§6), keeping each gap near 2×.
- **Depth cut that passes PPL but fails downstream.** Symptom: perplexity looks fine after dropping layers, but reasoning benchmarks tank. Cause: block importance and perplexity are screening proxies; a "low-importance" layer can still carry something a downstream task needs. Safeguard: validate every depth cut against a real downstream eval, not just PPL (this is exactly what the Llama practitioners did with Winogrande).
- **Treating retraining as fine-tuning.** Symptom: the student plateaus well below the teacher and never closes the gap. Cause: retraining with ordinary cross-entropy on hard labels instead of the teacher's soft distribution wastes the densest signal you have. Safeguard: retrain *exclusively* with the distillation loss (§4).
- **Breaking GQA when pruning heads.** Symptom: the pruned model produces garbage or throws shape errors at attention. Cause: query heads were dropped without respecting their key/value group membership, orphaning queries from their KV. Safeguard: prune heads group-aware, dropping whole GQA groups or keeping the grouping intact — the same structural-constraint discipline that Minitron-SSM applies to Mamba state.

The meta-lesson: none of these are exotic. Every one is a place where the *obvious* shortcut (skip teacher correction, cut 4× at once, trust PPL, fine-tune normally) quietly degrades the result, and the recipe is largely a list of those shortcuts with "don't" attached.

## Case studies from the papers and the field

### 1. Nemotron-4 15B → 8B → 4B: the original Minitron

The founding result. NVIDIA took their own Nemotron-4 15B and compressed it into an 8B and a 4B, deriving the whole family with up to 40× fewer training tokens per model and 1.8× total family compute. Because they owned the training data, distribution shift was not a concern — this was the clean lab demonstration that the recipe works. The 8B landed at MMLU 63.8, comparable to contemporaries trained on far more data, and the 4B at 58.6. Crucially, the paper showed Minitron models *beat from-scratch models of the same size by up to 16% MMLU* — the compressed model is not a lossy approximation of a bigger one, it is a genuinely better way to spend parameters at that scale. The lesson the field took from this: if you are going to ship multiple sizes in a family, train the big one well and derive the rest. The from-scratch small models are leaving accuracy and compute on the table.

### 2. Llama-3.1-8B → 4B: width versus depth, head to head

The follow-up paper ran the cleanest controlled experiment in the literature on width-vs-depth, producing *two* Llama-3.1-Minitron-4B models from the same 8B: one width-pruned ($d_{\text{model}}$ 4096 → 3072, $d_{\text{ff}}$ 14336 → 9216, all 32 layers kept) and one depth-pruned (width kept, 16 of 32 layers removed). The width-pruned variant won on accuracy — MMLU 60.5 versus the depth model's lower score — confirming the "prefer width below 15B" rule. The depth-pruned variant won on raw latency, because halving the layer count halves the sequential critical path. This case study is the empirical backbone of §3: same base, same target size, same data, and the only variable is the cut axis. If someone tells you depth and width pruning are interchangeable, this is the experiment that says otherwise — they buy different things, and you choose based on whether accuracy or latency is your binding constraint.

### 3. MN-Minitron-8B: a smaller model that beat its larger peers

Width-pruning Mistral-NeMo-12B down to 8B produced MN-Minitron-8B, which at release was **state-of-the-art among 8B models** — beating Llama-3.1-8B across a swath of benchmarks while training on only a few hundred billion tokens of distillation data instead of the multi-trillion-token budgets its competitors used. This is the result that made people take the recipe seriously, because it broke the intuition that a pruned model is a *downgrade*. A 12B carefully pruned and distilled to 8B outperformed natively-trained 8Bs. The takeaway for practitioners: the best way to get a great 8B might be to train (or acquire) a great 12B and compress it, rather than training an 8B directly — the larger model's redundancy, distilled down, beats the smaller model's from-scratch optimization. It reframes "what size should we train?" into "what size should we train, knowing we will compress it afterward?"

### 4. The depth-pruned Llama and the Winogrande sanity check

When the practitioners depth-pruned Llama-3.1-8B, they did not trust block importance blindly. They evaluated candidate layer-drop choices by removing groups of layers, measuring the LM loss on a validation set, and then — critically — validating the survivor against a **downstream task (Winogrande)** rather than just perplexity. The block-importance profile pointed at the middle-to-late layers as least important; dropping a contiguous block there, keeping the first and last layers intact, performed best on the downstream check. The lesson is methodological: importance metrics are screening tools, not oracles. A layer can look low-importance by cosine distance and still carry something a downstream task needs. Always close the loop with a real eval before you commit to a depth cut — perplexity and block importance can agree with each other and still be wrong about what the model uses in practice.

### 5. The teacher-correction ablation

The cleanest demonstration that teacher correction matters is the ablation in the practice paper: distill the student from the *uncorrected* teacher versus the *corrected* teacher, holding everything else fixed. The uncorrected teacher — evaluated on a corpus it was never trained on — emits miscalibrated soft labels, and the student that learns from them is measurably weaker. The corrected teacher, lightly fine-tuned on the distillation corpus first, produces clean targets and a stronger student. This single step is what generalized the recipe from "compress your own model where you have the data" to "compress anyone's open-weights model." If you are pruning a model you downloaded — Llama, Mistral, Qwen, anything — and you skip teacher correction, you are distilling someone else's distribution shift into your student. Do the one-step fine-tune first; it is the cheapest insurance in the whole pipeline.

### 6. Iterative vs one-shot: the 12% MMLU gap

The paper's iterative-compression ablation pruned 15B → 4B two ways: directly, and via an 8B waypoint. The two-hop path landed about **12% higher on MMLU** at the same 4B size. The mechanism is the teacher–student capacity gap: a 4B student tracks an 8B teacher far better than it tracks a 15B teacher. This is a general principle worth carrying beyond Minitron — whenever you distill, mind the gap. A teacher that is too strong relative to the student teaches a curriculum the student cannot absorb, and you get a weaker result than a more modest teacher would have given. Staging the compression keeps every hop inside the distillation sweet spot, and the intermediate models are products you wanted anyway, so the staging is not overhead.

### 7. Minitron-SSM: pruning Mamba and hybrid models

The recipe did not stay confined to dense transformers. ["Minitron-SSM"](https://arxiv.org/abs/2504.11409) extended structured pruning to **state-space (Mamba) and hybrid Mamba-Transformer** models with a technique called **group-aware SSM pruning** — pruning the SSM state and channels in a way that respects the grouped structure of Mamba-2's state, the same way GQA forces you to prune attention heads group-wise. This matters because NVIDIA's newer Nemotron models (Nemotron-H, Nemotron-Nano) are hybrids, and the compression pipeline had to follow the architecture there. The lesson: the four-axes framing generalizes — every architecture has its own structurally-independent dimensions, and the importance-then-distill loop adapts to whatever those dimensions are. Pruning is a *methodology*, not a transformer-specific trick. The constraint that the prunable units have group structure (GQA heads, SSM state groups) recurs, and respecting it is non-negotiable.

### 8. The Nemotron-Nano lineage and productionized tooling

Minitron stopped being a paper and became infrastructure. The compression recipe is implemented in NVIDIA's **TensorRT Model Optimizer** and **NeMo** frameworks, which is how the CLI in §3 exists at all — pruning a checkpoint to target dimensions and distilling it is a supported workflow, not a research script you reimplement. The Nemotron-Nano and later compact Nemotron models carry the lineage forward: train a strong larger model, prune-and-distill to the deployment sizes, ship the family. For a practitioner, this is the most important case study of all, because it means you do not have to reproduce the papers from scratch — the recipe is a few config flags and a calibration set away. The barrier to compressing your own models is now mostly knowing the rules in this post, not building the machinery.

### 9. The code model that forgot how to close a brace

A composite cautionary tale that every team eventually lives some version of: you prune a code-capable model, calibrate importance on a general web corpus because that is what you had lying around, run the short distillation, and the model now writes plausible prose and subtly broken code — it drops a closing brace on long functions, mis-indents, forgets to import. Nothing in the loss curve flagged it; perplexity on the general corpus looked great. The root cause is §2's warning made concrete: the calibration set under-represented code, so the heads and neurons that track bracket-matching and indentation scored low and got pruned, and the short retrain could not regrow structure that was cut out. The fix is upstream, not downstream — re-prune with a calibration set that includes representative code, and validate on a code benchmark (HumanEval, MBPP), not just perplexity. The lesson generalizes to any specialized capability: importance is conditional on what you measure it on, and a capability absent from calibration is a capability on the chopping block.

### 10. Stacking Minitron with 4-bit quantization for the edge

A deployment story. You have a Llama-3.1-Minitron-4B and a target of running it on a memory-constrained accelerator. Pruning already bought you a dense 4B that runs ~1.8× faster than the 8B; now you [quantize it to 4-bit](/blog/machine-learning/large-language-model/quantization-in-llm). The two techniques compose cleanly because they act on orthogonal axes — pruning decided *how many* weights exist, quantization decides *how many bits* each weight gets — so a 4B at 4-bit is roughly a 2GB model that fits where the original 16GB 8B never could. The order matters in one direction: prune and distill *first* (in higher precision, so the distillation signal is clean), then quantize the finished small model. Quantizing before distilling would have the student chasing a teacher through quantization noise. This stacking is how the compact Nemotron and similar models reach edge form factors, and it is the practical answer to "how small can this go" — multiply the savings, do not pick one.

### 11. Minitron versus the classical pruners

It is worth situating Minitron against the structured pruners that preceded it — LLM-Pruner, SliceGPT, SparseGPT, and the Wanda family. Those methods are clever about *what to remove* — gradient-based importance, low-rank slicing, second-order saliency — but most of them are pitched as *one-shot or light-retraining* methods that try to minimize the accuracy drop *at the moment of pruning*. Minitron's bet is different and, at this scale, decisive: it accepts a larger drop at prune time and bets everything on a *strong distillation-based recovery*. The pruning criterion (activation importance) is simpler than some competitors', but the recovery (full soft-distribution distillation, teacher correction, iteration) is far stronger, and the recovery is what dominates the final accuracy. The case study is the comparison itself: across the benchmarks in the Minitron paper, the distillation-retrained models outperform prior structured-compression techniques, and the gap traces almost entirely to the retraining objective, not the pruning criterion. The actionable insight: do not over-invest in a fancy pruning saliency metric; invest in the distillation that follows.

### 12. The GQA head-pruning gotcha

A short, sharp implementation war story. Modern models use grouped-query attention: 32 query heads but only 8 key/value heads, with each KV head shared across a group of 4 query heads. A naive head-importance prune ranks all 32 query heads and keeps the top-k — and promptly orphans query heads from their KV group, producing either a shape mismatch at attention or, worse, a model that runs but computes attention incorrectly because the head-to-group mapping is now scrambled. The fix is to make the pruning *group-aware*: rank and drop whole GQA groups, or constrain the per-group survivor counts so the grouping stays coherent. This is exactly the structural constraint that the modelopt tooling encodes (and that Minitron-SSM generalizes to Mamba state groups). The lesson for anyone hand-rolling structured pruning: the units you prune are rarely as independent as they look — respect the architecture's grouping or the cut will be silently wrong.

### 13. Long context survives pruning better than you would fear

A worry that comes up the first time a team prunes a long-context model: does cutting heads and layers wreck the model's ability to use its full context window? The empirical answer, across the Minitron-derived models, is reassuring — long-context capability is comparatively *robust* to width pruning, provided the calibration and distillation corpora include long sequences. The mechanism is intuitive in hindsight: the machinery that makes long context work — the positional encoding scheme (RoPE and its scaling), the attention pattern, the KV-cache structure — lives in the *architecture*, not in any particular head or neuron you might prune, so width pruning trims redundant capacity without removing the long-context mechanism itself. Where teams get burned is the data side, not the pruning side: if you calibrate and distill only on short sequences, the model's long-range behavior drifts because nothing in the retraining exercised it, and the short retrain budget means there is little opportunity to recover it. The lesson mirrors the calibration-data theme — pruning preserves what the architecture provides and what the data exercises; include long sequences in both, and the context window comes through the compression intact.

### 14. Compressing reasoning models is the harder frontier

The Minitron recipe was developed on base and instruct models, and it transfers cleanly to those. Reasoning models — the ones trained with long chain-of-thought and RL on verifiable rewards, covered in the [pretraining large reasoning models](/blog/machine-learning/large-language-model/pretraining-large-reasoning-models) discussion — are a harder case, and an active one for NVIDIA's later Nemotron work. The difficulty is that reasoning capability is *thin*: it is a behavior that emerges from a specific, hard-won training phase and is more fragile under perturbation than the broad knowledge that pruning trims so forgivingly. Distilling a reasoning teacher into a pruned student means the soft-target signal has to carry not just "what is the next token" but "what is the next *reasoning step*," and short retrains can leave the student fluent but worse at the multi-step traces. The practical adjustment is to distill reasoning models on reasoning-heavy data with longer retraining budgets, and to validate on reasoning benchmarks (GSM8K, MATH, the AIME-style sets) rather than knowledge benchmarks. The case study is a caveat to the whole recipe: the more a capability depends on a delicate late-stage training phase, the more carefully you must preserve it through compression — and the further you should stay from aggressive single-shot cuts.

### 15. When pruning alone is not enough, and you reach for continued pretraining

A final honest case: sometimes the prune-and-distill output is *good but not good enough*, and the right move is to follow distillation with a round of continued pretraining or targeted fine-tuning. This happens when the target size is genuinely too small to hold the teacher's full capability — past 3–4× compression, the student simply lacks the parameters, and no distillation recovers what the architecture cannot represent. The tell is a recovery curve that plateaus well short of the teacher with the gap refusing to close as you add distillation tokens. The fix is to treat distillation as the *recovery* step it is, then add capability-specific training on top: a continued-pretraining phase on domain data, or a fine-tune for the deployment task. This is where prune-and-distill hands off to the broader [fine-tuning toolkit](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) — Minitron gets you a strong small base for a fraction of the cost, and from there it is an ordinary (and much cheaper) fine-tuning problem. The lesson is to know which problem you are solving: pruning compresses existing capability; it does not manufacture new capability, and conflating the two leads to grinding a distillation that was never going to close the gap.

## When to reach for prune-and-distill — and when not to

**Reach for it when:**

- **You already have a strong larger model in the same family** and need smaller deployment sizes. This is the canonical case — the larger model is your teacher and your parameter donor.
- **You want a *family* of sizes** (edge / serving / quality tiers) from one pretraining investment. The economics only really shine when you amortize one big run across several derived models.
- **You are compressing an open-weights model you did not train.** Teacher correction makes this work even without the original corpus — you just need a representative proxy dataset.
- **You need real wall-clock speedup, not just smaller files.** Structured pruning yields dense smaller matmuls that run faster on stock hardware, unlike unstructured sparsity.
- **You are below ~15B and accuracy-sensitive** — width pruning here reliably outperforms training the small model from scratch by a meaningful MMLU margin.

**Skip it (or think twice) when:**

- **You do not have a good teacher.** No strong larger model means nothing to prune and nothing to distill from. Pruning a mediocre model just gives you a smaller mediocre model.
- **Your target is a *different* capability, not a smaller version of the same one.** Pruning preserves and compresses what the teacher already does; it does not add skills. If you need new behavior, that is a fine-tuning or continued-pretraining problem, covered in the [LLM fine-tuning techniques guide](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques).
- **You need extreme compression (8×+) in one shot.** Past 2–4×, accuracy degrades faster than distillation can recover it in a short retrain; stage it through intermediate sizes or accept a bigger retraining budget.
- **A quantized version of the full model already meets your constraints.** If 4-bit quantization of the big model fits your memory and latency budget, you may not need to prune at all — and quantization preserves the full model's behavior more faithfully than aggressive pruning does.
- **You have no calibration data resembling your deployment distribution.** Importance estimation and teacher correction both assume a representative corpus; garbage calibration data gives you a confidently-wrong pruning plan that no amount of retraining fully repairs.

The one-sentence version, the thing to remember when the cluster-time request lands on your desk:

> If you need a smaller version of a model you already have, do not train it — prune it and let the original teach the survivor. Train from scratch only when you need a capability the teacher does not have.

## Further reading

- [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679) — the original Minitron paper, with the full best-practices list and ablations.
- [LLM Pruning and Distillation in Practice: The Minitron Approach](https://arxiv.org/abs/2408.11796) — the Llama-3.1 / Mistral-NeMo follow-up, where teacher correction is introduced.
- [Minitron-SSM: Group-Aware SSM Pruning](https://arxiv.org/abs/2504.11409) — extending the recipe to Mamba and hybrid models.
- [Pruning in LLMs](/blog/machine-learning/large-language-model/pruning-in-llm) — structured vs unstructured pruning, the broader landscape.
- [Knowledge Distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) — distillation theory, losses, and recipes beyond Minitron.
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) — the orthogonal compression axis you should stack on top.
- [MoE architecture, training, and finetuning](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) — where the modern Nemotron compression lineage is heading.

*This is the first post in a series reading NVIDIA's model reports for their reusable techniques. Next up: Nemotron-4 340B and the synthetic-data-plus-reward-modeling pipeline behind it.*
