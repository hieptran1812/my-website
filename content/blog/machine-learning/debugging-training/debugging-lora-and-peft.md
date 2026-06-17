---
title: "Debugging LoRA and PEFT: The Adapter That Silently Never Trains"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Localize and fix the whole family of silent LoRA failures — the adapter that was never inserted, the wrong variable passed to the Trainer, the grad that checkpointing eats — by reading trainable params, grad flow, and merge equivalence."
tags:
  [
    "debugging",
    "model-training",
    "lora",
    "peft",
    "finetuning",
    "deep-learning",
    "llm",
    "pytorch",
    "qlora",
    "quantization",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/debugging-lora-and-peft-1.png"
---

Here is a run that breaks hearts. You load a 7B base model, wrap it with LoRA, hand it to the `Trainer`, and watch the loss curve. It goes down. Not dramatically, but it goes down — `2.31 → 2.28 → 2.26` over a few hundred steps. The bars fill, the ETA ticks, the GPU fans roar. Six hours and a few dollars of compute later, you load the adapter, run your eval, and the model behaves exactly like the base. No new behavior. No learned format. Nothing. You retrain with a higher learning rate, more epochs, a bigger rank — and you get the same nothing, more expensively.

The loss was lying to you. That gentle downward drift was the *base model's* logits settling under whatever tiny signal leaked through, or just the optimizer touching a layer-norm you accidentally left trainable. The LoRA adapter — the thing you actually wanted to train — never entered the computation graph at all. Its trainable parameter count was **zero**, and a `Trainer` will cheerfully optimize a model with nothing to optimize and report a clean run the whole way. This is the defining pathology of parameter-efficient finetuning: LoRA and PEFT multiply the number of ways a run can *silently no-op*. The base is frozen by design, the adapter is the only live part, and if the adapter falls out of the graph for any of a dozen reasons, you get a run that looks like training and is, in fact, an expensive identity function.

The economics are what make this worth a chapter. A silent no-op does not cost you a crash and a stack trace you fix in minutes; it costs you the *whole run* plus the time to even suspect something is wrong plus the retries you do before you stop blaming the learning rate. A six-hour run on a single rented A100 at roughly \$2 per GPU-hour is \$12 down the drain — and that is the cheap case. The expensive case is the multi-GPU run, or the third and fourth retries at different learning rates and ranks (each another \$12, another half-day), or the adapter you shipped to production that quietly does nothing while you debug "why isn't the model better" everywhere *except* the one place the bug actually lives. The fix in every case is a diagnostic that costs *a fraction of a second* — one forward-backward-step before you commit to the long run. The return on learning to read these instruments is enormous: you trade a thirty-line pre-flight check for the entire class of wasted-run failures.

This post is about catching that — and the whole family of failures around it — in seconds, not after a wasted GPU-hour. We will build from the science up: what LoRA actually computes, why $B$ is initialized to zero so the adapter *starts as an exact identity* (which is precisely why a broken adapter is invisible at step one), why the $\alpha/r$ scaling matters, and why only the two low-rank matrices receive gradient. Figure 1 shows that structure. Then we will go through the bugs one at a time, each with its mechanism, a runnable diagnostic in Hugging Face `peft` and `transformers`, and a concrete before→after where fixing one line takes trainable params from `0` to `4.2M` and the loss from flat to falling. The single most important habit you will leave with: **always call `print_trainable_parameters()`, confirm it is nonzero and sane (~0.1–1%), and run overfit-one-batch — a LoRA no-op fails it in under a minute.**

This is the LoRA chapter of a larger series on debugging training and finetuning. The spine of that series is that every training bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and a disciplined debugger *bisects* to the right one before touching code. A LoRA no-op is almost always a **model-code** bug (the adapter is not in the graph) wearing the *costume* of an **optimization** bug (the loss won't go down). Knowing that distinction is half the fight. For the master decision tree this slots into, see [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); for the full symptom→suspect→test→fix playbook, see [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

Why does LoRA deserve its own debugging chapter when full finetuning does not? Because full finetuning has *no silent no-op mode* in the same way. If you full-finetune a model and forget to pass it to the optimizer, you get an obvious crash or a loss that does not move at all from step one — the symptom is loud. LoRA's whole value proposition — freeze 99.8% of the model, train a tiny bolt-on — is also its whole hazard surface. The frozen base is, by construction, a working language model. It produces sensible logits and a sensible loss *with or without* a functioning adapter. So the failure mode is not "the model is broken" but "the model is exactly as good (or bad) as the untouched base, and nothing you did mattered." That is a quieter, more expensive failure, and it requires instruments the base-model loss simply cannot provide. The entire job of this post is to hand you those instruments and the reasoning to read them.

One more framing before we dive in, because it governs every diagnostic that follows. A LoRA finetune is a *circuit*, and learning only happens if current flows all the way around it: a parameter must (1) be a `requires_grad=True` leaf, (2) sit on the forward path that produces the loss, (3) receive a nonzero gradient on backward, and (4) live in an optimizer param-group so `step()` moves it. Break any one link and that parameter silently freezes while the rest of the run looks normal. LoRA gives you many more links to break than full finetuning — the adapter has to be *inserted*, *wired to the right modules*, *handed to the Trainer as the trained object*, *reached by gradient through checkpointing and dtype boundaries*, and *kept trainable through any token-resizing*. We will walk each link, name how it breaks, and give the one-line instrument that proves it intact.

![Stacked layers showing a LoRA layer built from a frozen base weight, a trainable down-projection A, a trainable up-projection B initialized to zero, an alpha over r scale, and an output where only A and B receive gradient](/imgs/blogs/debugging-lora-and-peft-1.png)

## 1. The science: what LoRA actually computes

You cannot debug a thing you cannot write down, so let us write LoRA down precisely. Take any linear layer in the base model with weight $W \in \mathbb{R}^{d \times k}$. In a frozen-base finetune, $W$ never changes. The full-finetuning update would be $W \to W + \Delta W$, where $\Delta W$ is also $d \times k$ — that is a lot of parameters to learn and store. The LoRA hypothesis, from Hu et al. 2021 ("LoRA: Low-Rank Adaptation of Large Language Models"), is that the *useful* part of $\Delta W$ for adapting a pretrained model has **low intrinsic rank**. So instead of learning a full $d \times k$ matrix, you learn two thin matrices whose product approximates the update:

$$
\Delta W \approx \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k).
$$

The forward pass becomes $h = W x + \frac{\alpha}{r} B A x$. The frozen base path $W x$ is untouched; the adapter path $\frac{\alpha}{r} B A x$ is the learnable correction. Here $r$ is the **rank** (typically 8–64), and $\alpha$ is a scaling hyperparameter. The parameter count of the adapter is $r(d + k)$ instead of $dk$. For a $4096 \times 4096$ projection with $r = 16$, that is $16 \times 8192 = 131{,}072$ trainable params instead of $16.7$ million — about **0.8%**. Sum that across the attention and MLP projections of a 7B model and you land near the famous "~0.1–1% trainable" figure.

Three facts about this construction are load-bearing for every bug in this post, so internalize them now.

**Fact one: $B$ is initialized to zero, $A$ to a small random distribution.** At the start of training, $B A = 0$, so $\frac{\alpha}{r} B A x = 0$, and the layer computes *exactly* $W x$. The adapter is a perfect identity at step zero. This is a feature — it means the model behaves like the well-tuned base model at the start and you only ever *add* to it, avoiding a destructive jolt. But it is also the reason a LoRA no-op is invisible. A randomly initialized network produces garbage that *announces* it is broken. A LoRA adapter that is silently disconnected produces the same correct base outputs and the same plausible loss as a *correctly connected but not-yet-trained* adapter. They are indistinguishable at step one from the loss alone. You must read a different instrument.

**Fact two: only $A$ and $B$ receive gradient.** PyTorch builds the autograd graph from the operations you actually run. `peft` sets `requires_grad=False` on $W$ and `requires_grad=True` on $A$ and $B$. Backward therefore fills `A.grad` and `B.grad` and leaves `W.grad` as `None`. If the LoRA layer was never inserted, there are no $A$ and $B$ tensors with `requires_grad=True` anywhere in the model — so the set of trainable parameters is empty, and `optimizer.step()` has nothing to move. That is the no-op, stated mechanically.

**Fact three: the scale is $\alpha/r$, not $\alpha$.** The effective magnitude of the adapter's correction is governed by $\alpha/r$. A common and reasonable default is $\alpha = 2r$, giving a scale of $2.0$. The subtlety: if you increase $r$ to give the adapter more capacity but leave $\alpha$ fixed, you *shrink* the scale, which can make the adapter learn more slowly than you expect. Some implementations (RSLoRA) use $\alpha/\sqrt{r}$ to decouple capacity from scale. The point for debugging is that "I doubled the rank and it learned worse" is a *predictable consequence of the scaling formula*, not a mysterious regression.

It is worth being precise about *why* the low-rank decomposition is a sensible thing to do at all, because that intuition tells you when rank is your bottleneck (section 9). A full update matrix $\Delta W \in \mathbb{R}^{d\times k}$ has at most $\min(d,k)$ nonzero singular values. The LoRA claim is that for adapting an *already-pretrained* model to a downstream task, the directions that matter are few — the singular value spectrum of the *useful* $\Delta W$ decays fast, so a rank-$r$ approximation $B A$ captures most of the signal. Hu et al. measured this and found that even $r=1$ or $r=2$ recovers a surprising fraction of full-finetune quality on many tasks, which is the empirical heart of the method. The debugging consequence: if your task genuinely needs a high-rank update — teaching a new language, a large domain shift, a fundamentally new skill — a tiny $r$ will *cap* what the adapter can express, and no amount of learning-rate tuning fixes a capacity ceiling. That is a real, diagnosable failure (overfit-one-batch passes, the full run plateaus, raising $r$ lowers the plateau) and a different animal from the no-op.

Now the mechanism that produces the no-op, made concrete. PyTorch's autograd is a *tape*: every operation on a tensor that requires grad records a node in a dynamic graph, and `loss.backward()` walks that graph in reverse, applying the chain rule. A parameter's `.grad` gets filled only if there is a recorded path from that parameter to the loss. When `peft` wraps a `Linear`, it (a) sets `weight.requires_grad=False` on the base, (b) creates `lora_A` and `lora_B` `nn.Linear` (or `nn.Parameter`) submodules with `requires_grad=True`, and (c) rewrites the layer's `forward` to compute `base(x) + scaling * lora_B(lora_A(dropout(x)))`. The base term contributes to the output (so gradient flows *through* it to earlier layers) but accumulates no `.grad` because its weight is frozen. The adapter term is the only thing whose parameters are both on the loss path *and* `requires_grad=True`. If step (a)–(c) never ran — because no module name matched `target_modules` — then there are no `requires_grad=True` parameters in the entire model, the trainable set is empty, and backward has nothing to fill. The optimizer's `step()` iterates an empty param-group and returns instantly. The loss you see is pure base-model behavior plus optimizer noise on whatever (if anything) was left trainable. That is the no-op in mechanical detail, and it explains why the only reliable early instrument is the *count of trainable parameters*, not the loss.

#### Worked example: counting trainable params by hand

Take a small concrete case so the numbers are not abstract. A 1.3B-parameter decoder with hidden size $d = 2048$, applying LoRA at rank $r = 8$ to the four attention projections ($q, k, v, o$, each $2048 \times 2048$) across 24 layers.

Per projection, the adapter is $A \in \mathbb{R}^{8 \times 2048}$ and $B \in \mathbb{R}^{2048 \times 8}$, totaling $8 \times 2048 + 2048 \times 8 = 32{,}768$ params. Four projections per layer, 24 layers: $32{,}768 \times 4 \times 24 = 3{,}145{,}728 \approx 3.1\text{M}$ trainable params. Against $1.3\text{B}$ total, that is $3.1\text{M} / 1.3\text{B} \approx 0.24\%$. So when `print_trainable_parameters()` reports something like `trainable params: 3,145,728 || all params: 1,300,000,000 || trainable%: 0.2420`, you can *check the arithmetic* and confirm it matches your config. If it instead prints `trainable params: 0 || trainable%: 0.0000`, you have your no-op, and you have it in the first ten seconds of the run rather than the sixth hour.

The minimal LoRA config that produces those numbers:

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("your/base-1.3b")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,                 # alpha/r = 16/8 = 2.0 scale
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base, lora_config)
model.print_trainable_parameters()
# trainable params: 3,145,728 || all params: 1,303,145,728 || trainable%: 0.2414
```

Notice the assertion you can now make: the reported `trainable%` should land in the **0.1–1%** band for typical attention-only or attention+MLP LoRA. Below `0.05%` means you probably under-targeted (too few modules, too small a rank). Exactly `0.0000%` means no adapter at all. Above a few percent means you accidentally left big things trainable (the embeddings, the LM head, or you forgot to freeze the base) — which is its own bug class we get to in section 8.

## 2. The silent no-op: trainable params = 0

This is the headline bug and it has two distinct root causes that produce the *same* symptom — `trainable params: 0` — so we treat them together and then separate them with a test. Figure 2 shows the before→after.

![Before and after comparison: on the left target modules are wrong so the adapter is never inserted and trainable params are zero with a flat loss, on the right the projections are matched so trainable params reach 4.2 million and the loss falls](/imgs/blogs/debugging-lora-and-peft-2.png)

**Root cause A: `target_modules` matches nothing.** `peft` inserts a LoRA layer wherever a module's *name* matches an entry in `target_modules`. If you pass `["query", "value"]` to a model whose attention projections are named `q_proj` and `v_proj`, nothing matches, no LoRA layer is created, and `get_peft_model` returns a model with zero trainable parameters. This is alarmingly easy: module naming differs across architectures. Llama-family models use `q_proj/k_proj/v_proj/o_proj` and `gate_proj/up_proj/down_proj`; GPT-2 uses `c_attn/c_proj`; some ViTs use `query/key/value`; T5 uses `q/k/v/o`. Copy a config from a Llama tutorial onto a GPT-2 and you get a clean, silent zero.

The naming problem deserves a table, because it is the single most frequent way a copied config no-ops. The "right" `target_modules` is a property of the *architecture*, and there is no universal answer:

| Architecture family | Attention projection names | MLP projection names | Note |
| --- | --- | --- | --- |
| Llama / Mistral / Qwen | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `gate_proj`, `up_proj`, `down_proj` | the de-facto tutorial default |
| GPT-2 / GPT-NeoX (some) | `c_attn` (fused QKV), `c_proj` | `c_fc`, `c_proj` | `Conv1D`, needs `fan_in_fan_out=True` |
| BERT / RoBERTa | `query`, `key`, `value`, `dense` | `intermediate.dense`, `output.dense` | encoder, not causal |
| T5 | `q`, `k`, `v`, `o` | `wi`, `wo` (or `wi_0`/`wi_1`) | short names collide easily |
| ViT (timm/HF) | `qkv` (fused) or `query`/`key`/`value` | `fc1`, `fc2` | fused `qkv` is one Linear |

Paste a Llama config (`["q_proj","v_proj"]`) onto a GPT-2 and *nothing* matches — GPT-2 has no module named `q_proj`. The result is a clean `trainable = 0`. Worse, the failure is *type-correct*: `get_peft_model` does not raise, because "matched zero modules" is a valid (if useless) outcome. The library trusts you. So you must check.

**Root cause B: you wrapped the model but handed the wrong variable to the `Trainer`.** This one is insidious because `print_trainable_parameters()` on the *wrapped* model reads correctly — `4.2M`, all good — but the model that actually trains is the frozen base. The classic shape:

```python
base = AutoModelForCausalLM.from_pretrained("your/base")
peft_model = get_peft_model(base, lora_config)
peft_model.print_trainable_parameters()   # 4.2M — looks great!

trainer = Trainer(
    model=base,                # BUG: passed `base`, not `peft_model`
    args=training_args,
    train_dataset=ds,
)
trainer.train()                # trains the frozen base — nothing trainable
```

Because `get_peft_model` modifies `base` *in place* in most paths, you often get away with this — but not always, and especially not if you reassign or reload. The robust fix is to never keep both names around: assign the wrapped result back, `model = get_peft_model(model, cfg)`, and pass exactly that object to the `Trainer`.

There is a second, sneakier variant of root cause B that bites people using `trl`'s `SFTTrainer` or `DPOTrainer`: those trainers can *also* accept a `peft_config` argument and wrap the model for you internally. If you pass an *already-wrapped* PEFT model *and* a `peft_config`, you can end up double-wrapping (a LoRA on top of a LoRA) or, depending on version, wrapping a model that was already wrapped — confusing the trainable set. Pick exactly one path: either wrap the model yourself with `get_peft_model` and pass the wrapped model with **no** `peft_config`, or pass the *base* model plus a `peft_config` and let the trainer wrap it. Mixing both is a reliable way to get a trainable count you cannot explain. When in doubt, print the count *from inside the trainer* (`trainer.model.print_trainable_parameters()` right before `trainer.train()`), not from the variable you think you wrapped.

### The diagnostic: two lines that catch both

The instrument that catches root cause A is the trainable-parameter count plus a scan of the model's module names. The instrument that catches root cause B is an *identity assertion* that the object the `Trainer` holds is the wrapped object.

```python
from peft import get_peft_model

model = get_peft_model(base, lora_config)
model.print_trainable_parameters()

# (A) Did any LoRA layer get inserted? List them.
lora_layers = [n for n, _ in model.named_modules() if "lora_A" in n]
print(f"LoRA layers inserted: {len(lora_layers)}")
assert len(lora_layers) > 0, "No LoRA layer matched target_modules!"
print("examples:", lora_layers[:3])

# A hard floor on the trainable count — fail fast, not after 6 hours.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 0, f"trainable params = {trainable}; adapter is a no-op"

# (B) The object you train MUST be the wrapped object.
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
assert trainer.model is model, "Trainer is not holding the PEFT-wrapped model!"
```

If you do not know your model's module names, do not guess — print them:

```python
import collections
# Count Linear modules by their *leaf* name to find the real projection names.
names = collections.Counter(
    n.split(".")[-1]
    for n, m in base.named_modules()
    if m.__class__.__name__ in ("Linear", "Conv1D")  # Conv1D for GPT-2
)
print(names.most_common(12))
# e.g. Llama: [('q_proj', 32), ('k_proj', 32), ('v_proj', 32), ('o_proj', 32),
#              ('gate_proj', 32), ('up_proj', 32), ('down_proj', 32)]
```

Modern `peft` accepts `target_modules="all-linear"`, which matches every linear layer (excluding the output head) and sidesteps the naming-mismatch bug entirely. It is the safest default when you are unsure, at the cost of a few more trainable params:

```python
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```

#### Worked example: bisecting a "won't train" report in four minutes

A colleague pings you: "My LoRA finetune of a 7B model won't learn. I've tried LR 1e-4, 3e-4, 1e-3, more epochs, rank 8, rank 32 — loss barely moves. Is the data bad?" Notice they have spent days varying *optimization* knobs (LR, epochs, rank). The six-places frame says: do not touch optimization until you have ruled out model code. Here is the four-minute bisection.

First, the trainable count: `model.print_trainable_parameters()` → `trainable params: 0 || trainable%: 0.0000`. That single line ends the investigation into LR and data — there is nothing to optimize, so every LR was equivalent. Second, *why* zero: scan `named_modules()` for `lora_A` → empty. No adapter was inserted. Third, *why* no adapter: print the real module names with the `Counter` snippet → the model is a Qwen variant using `q_proj`/`k_proj`/`v_proj`/`o_proj`, but the config targeted `["query","value"]` (copied from a BERT example). Nothing matched. The fix is one line: set `target_modules=["q_proj","k_proj","v_proj","o_proj"]` (or `"all-linear"`). Re-run the count → `4.2M, 0.21%`. Run overfit-one-batch → loss collapses `2.31 → 0.04`. Total time: four minutes, zero GPU-hours wasted on more LR sweeps. The lesson the colleague takes away is the same one this whole post is selling: **the trainable count is the first instrument, and it would have saved them the days of LR sweeping.**

### The before→after evidence

Run overfit-one-batch on four examples before and after the fix. (Overfit-one-batch — pushing a tiny fixed batch to near-zero loss — is the single highest-leverage sanity check in all of training; see [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) for the full method.) The contrast is brutal and diagnostic:

| Instrument | Buggy (`target_modules` wrong) | Fixed (matched + verified) |
| --- | --- | --- |
| `trainable params` | `0` | `4,194,304` |
| `trainable%` | `0.0000` | `0.21` |
| LoRA layers inserted | `0` | `224` |
| Loss @ step 0 | `2.31` | `2.31` |
| Loss @ step 200 (4-ex overfit) | `2.30` (flat) | `0.04` (collapsed) |
| `A_proj.grad` after backward | `None` | norm `0.7` |

The flat-vs-collapsing loss on a four-example overfit is the clincher. A *correctly wired* LoRA adapter, even at rank 8, has more than enough capacity to memorize four examples and will drive their loss toward zero within a couple hundred steps. If it does not, the adapter is not learning, full stop. The cause is upstream — in the graph, not the optimizer. This is the bisection move: a failed overfit-one-batch with `trainable = 0` rules out data and optimization entirely and points you at model code.

A subtle reading skill: the loss values at step 0 are *identical* in both columns (`2.31`). They have to be — at step 0 the adapter is an identity (`B=0`), so a correctly wired but untrained adapter and a never-inserted adapter produce the *same* output. The divergence only appears once steps accumulate: the wired adapter pulls the loss down, the broken one sits flat. This is exactly why you cannot judge a LoRA run from its first few steps' loss; you have to either read the trainable count (instant) or let the overfit run long enough for the curves to separate (a minute or two). Engineers who stare at the first 10 steps and conclude "looks fine, loss is going down a little" are reading optimizer noise, not adapter learning.

## 3. Where the gradient reaches (and where it stops)

To debug "is the adapter learning?" you have to know exactly which tensors *should* carry gradient and verify that they do. Figure 3 traces it.

![Dataflow graph where the input feeds a frozen base branch with grad None and a trainable A then B then scale branch, the two branches merging at the output so gradient fills only A and B](/imgs/blogs/debugging-lora-and-peft-3.png)

The forward pass of a LoRA-wrapped linear splits the input into two branches and sums them: $h = W x + \frac{\alpha}{r} B (A x)$. Backward walks this graph in reverse. The base branch $W x$ has `W.requires_grad=False`, so although gradient *passes through* $W$ to reach earlier layers, no `.grad` is *accumulated on* $W$. The adapter branch $B(Ax)$ has `A.requires_grad=True` and `B.requires_grad=True`, so backward accumulates `A.grad` and `B.grad`. After a single `loss.backward()`, the healthy signature is: **every `lora_A` and `lora_B` parameter has a non-`None`, nonzero-norm grad; every base weight has grad `None`.** Anything else is a bug.

There is a beautiful subtlety hiding in $B = 0$. At step zero, $B A = 0$, so the *output* contribution of the adapter is zero. Does that mean $A$ gets no gradient? No — and this is where people get confused. The gradient of the loss with respect to $A$ flows *through* $B$. Since $B = 0$ at init, $\partial \mathcal{L} / \partial A = B^\top (\cdots) = 0$ at the very first step — so $A$ does not move on step zero. But $\partial \mathcal{L} / \partial B$ depends on $A x$, which is nonzero, so **$B$ moves first**. Once $B \neq 0$, gradient begins flowing to $A$ as well. The adapter "wakes up" over the first few steps: $B$ pulls away from zero, then $A$ follows. If you snapshot `B.grad.norm()` at step 0 it is nonzero; if you snapshot `A.grad.norm()` at step 0 it is ~0 and rises by step 5. Seeing that pattern is *positive confirmation* the adapter is wired correctly. Seeing `B.grad is None` is the bug.

Let us make that gradient claim explicit, because it is a genuinely useful sanity check and people get it wrong. For the adapter term $y = \frac{\alpha}{r} B A x$, the chain rule gives, for an upstream gradient $g = \partial \mathcal{L}/\partial y$:

$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r}\, g\,(A x)^\top, \qquad
\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r}\, B^\top g\, x^\top.
$$

Read those two formulas at step 0 with $B=0$: the gradient on $B$ is proportional to $g (Ax)^\top$, which is nonzero (because $A$ is random and $x$ is real data), so $B$ gets a real gradient and moves. The gradient on $A$ is proportional to $B^\top g x^\top$, and with $B=0$ that is exactly zero, so $A$ does *not* move on the first step. This is not a bug — it is the designed warm-up, and seeing it (nonzero `B.grad`, near-zero `A.grad` at step 0, both nonzero by step 5) is the strongest *positive* confirmation that your adapter is correctly in the graph and the autograd path is intact. If both are `None`, the adapter is disconnected. If `B.grad` is a healthy norm but `A.grad` stays at exactly zero past step 10, something is pinning $B$ at zero (a dtype underflow zeroing the update, or a zeroed learning rate on that group) — a rarer but real failure.

A frequent confusion worth heading off: "the adapter starts as identity, so won't the *first* gradient step do nothing?" No. The *output* is identity at step 0, but the *gradient* on $B$ is not zero — the loss still has a slope with respect to $B$ even when $B$'s current value happens to be zero. The identity-at-init property is about the forward pass (so you do not jolt the model), not about the backward pass (which is alive from step 0). Conflating the two leads people to "wait longer for it to warm up" when the real problem is a disconnected graph that will never warm up. The grad audit settles it in one backward pass.

### The diagnostic: assert grad flows into the adapter

```python
import torch

def assert_adapter_grad_flows(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()

    a_params = {n: p for n, p in model.named_parameters() if "lora_A" in n}
    b_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    assert a_params and b_params, "No LoRA params found — adapter not inserted."

    # B should have nonzero grad at step 0 (A x != 0); A's grad starts ~0.
    b_norms = {n: (p.grad.norm().item() if p.grad is not None else None)
               for n, p in b_params.items()}
    dead = [n for n, g in b_norms.items() if g is None or g == 0.0]
    assert not dead, f"These LoRA-B params got no gradient: {dead[:5]}"

    # Base weights must NOT accumulate grad (they are frozen).
    leaked = [n for n, p in model.named_parameters()
              if "lora_" not in n and p.requires_grad and p.grad is not None]
    assert not leaked, f"Frozen base weights are receiving grad: {leaked[:5]}"

    print("OK: adapter receives gradient; base is frozen.")
    print("sample B grad norms:", list(b_norms.values())[:3])
```

Call this once on a single real batch *before* committing to a long run. It is the difference between knowing the adapter is in the graph and *hoping* it is. The `set_to_none=True` matters — it makes "no grad" show up as `None` rather than a stale zero from a previous step, so a disconnected param is unambiguous.

## 4. The diagnostic matrix: bug → signature → check → fix

Before we work through the remaining failure modes individually, here is the field map. Every LoRA bug in this post leaves a *distinct fingerprint* in one of three cheap instruments — the trainable count, the grad audit, or a merge comparison — and each maps to a one-line fix. Figure 4 is this table; keep it next to the code.

![Matrix mapping five LoRA bugs to their instrument signature, confirming check, and fix, covering wrong target modules, wrong wrapped variable, gradient checkpointing without input grad, alpha over r too small, and merging in the wrong dtype](/imgs/blogs/debugging-lora-and-peft-4.png)

| Bug | Instrument signature | Confirming check | Fix |
| --- | --- | --- | --- |
| Wrong `target_modules` | `trainable = 0`, no `lora_A` modules | `print_trainable_parameters()` + scan `named_modules()` | Set real projection names or `"all-linear"` |
| Wrapped wrong variable | wrapped reads `4.2M` but Trainer trains base | `assert trainer.model is peft_model` | Pass the `get_peft_model` output to the Trainer |
| Grad-ckpt eats grad | `trainable = 4.2M` but `A.grad is None`, loss flat | grad audit after backward with checkpointing on | `model.enable_input_require_grads()` |
| `alpha/r` too small | trains but underfits, tiny updates | compute `alpha/r`; sweep in overfit-one-batch | Raise `alpha` (e.g. `2r`) or `r` |
| Merge in wrong dtype | merged ≠ adapter outputs | `torch.allclose` merged vs adapter | Merge in bf16/fp32, never a 4-bit base |

The decision logic behind this table — which instrument to read first given a symptom — is the tree in Figure 5.

![Decision tree starting from a zero trainable count or flat loss, branching to no LoRA layer inserted, wrong variable trained, or gradient blocked by checkpointing, each leading to a specific cause and fix](/imgs/blogs/debugging-lora-and-peft-5.png)

The tree encodes the bisection: read the trainable count *first*. If it is `0`, the adapter is not inserted or the wrong object is being trained — a graph problem, never an optimization one. If it is healthy (`~0.2%`) but the loss is still flat and a grad audit shows `A.grad is None`, the adapter exists but gradient cannot reach it — which in practice means gradient checkpointing or a dtype problem (sections 5 and 6). Only once trainable count is sane *and* grad flows *and* overfit-one-batch still fails do you start suspecting rank, scale, learning rate, or data. Notice how much you rule out before touching the optimizer.

There is an *ordering* to these instruments and it is not arbitrary — it follows the circuit from section 1, link by link, cheapest test first. The trainable count tests link 1 and 4 (is anything a trainable leaf in an optimizer group) and costs a single function call. The module scan confirms the adapter was *physically inserted*. The grad audit tests link 3 (does backward fill the gradient) and costs one forward-backward. The weight-delta check tests that `step()` actually moves the param. Each test is strictly cheaper than the next and rules out a distinct failure, so running them in order means you almost always stop at the first one that fails. This is the opposite of the common anti-pattern — relaunching the full run with a new hyperparameter and waiting hours to see if it "helped." The pre-flight in section 10 bundles this exact ordering into one callable. Internalize the ordering and you will reach for the right instrument reflexively: count, scan, grad, delta, *then* loss.

## 5. Gradient checkpointing eats the adapter's gradient

Here is a bug that has cost the community thousands of GPU-hours because it produces a *correct* trainable count (`4.2M`, looks perfect) and *still* learns nothing. It sits at the intersection of two features you turned on for good reasons: gradient checkpointing (to fit a bigger model in memory) and LoRA (to finetune cheaply). Together, without one extra line, they silently break.

### The science: why checkpointing severs the adapter's input grad

Gradient checkpointing saves memory by *not* storing intermediate activations during the forward pass; instead it re-runs the forward pass during backward to recompute them. PyTorch's `torch.utils.checkpoint` implements this by wrapping a segment of the model in a custom autograd function. The catch: for checkpointing to recompute and backpropagate correctly, the *input* to the checkpointed segment must require gradient — otherwise PyTorch concludes "nothing here needs grad" and the recomputed segment produces no gradient at all.

In a LoRA finetune, the model's *embedding* layer is frozen (it is part of the base). So the output of the embeddings — the input to the first transformer block — has `requires_grad=False`. When that frozen-input tensor enters a checkpointed transformer block, the checkpoint machinery sees a non-grad input and, in the common non-reentrant path, the gradient never propagates back into the block's parameters. The LoRA adapters *inside* those blocks therefore receive **no gradient**, even though they are perfectly inserted and counted as trainable. The result: `trainable = 4.2M`, `A.grad = None`, loss flat. The signature that distinguishes this from the section-2 no-op is exactly that the trainable count is *fine* while the grad is *absent*.

The mechanism is worth one more level of detail because it explains why the *reentrant* vs *non-reentrant* checkpoint variants behave differently, and why the advice has shifted over the years. The original (reentrant) `torch.utils.checkpoint` implementation, used by default for a long time and still the default in older `transformers`, requires at least one input to the checkpointed function to have `requires_grad=True` — otherwise it raises or silently produces no grad, because it relies on the autograd engine re-entering the recomputation with a grad-requiring input to seed the backward. With a frozen embedding feeding the first block, no input requires grad, and the chain breaks at the entrance. The newer non-reentrant implementation (`use_reentrant=False`) is more robust and is becoming the default, but for a frozen-base LoRA setup you *still* want the embeddings' output to require grad so the recomputed activations carry a gradient path to the adapters inside. The single fix — forcing the embedding output to require grad — covers both variants, which is why `enable_input_require_grads()` is the canonical answer regardless of which checkpoint path you are on.

### The fix: `enable_input_require_grads()`

`peft` and `transformers` expose a one-liner that registers a forward hook on the input embeddings forcing their output to require grad, which re-establishes the gradient path into the checkpointed blocks:

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, prepare_model_for_kbit_training

base = AutoModelForCausalLM.from_pretrained("your/base")
base.gradient_checkpointing_enable()          # save activation memory
base.enable_input_require_grads()             # <-- the fix: embeds output requires grad

model = get_peft_model(base, lora_config)
```

For QLoRA, `prepare_model_for_kbit_training(base)` does this *and* a few other things (casts layer norms to fp32, enables input grads, sets up gradient checkpointing) in one call — which is exactly why the QLoRA recipe tells you to call it and why skipping it produces this bug. Newer `transformers` also let you pass `gradient_checkpointing_kwargs={"use_reentrant": False}`, and the non-reentrant path interacts more predictably, but you *still* want `enable_input_require_grads()` for a frozen-embedding LoRA setup. The safest pattern is: call `prepare_model_for_kbit_training` for QLoRA, or `enable_input_require_grads()` explicitly for plain LoRA + checkpointing, and then *verify with the grad audit from section 3*.

#### Worked example: checkpointing turns a 0→4.2M no-op into learning

A team finetunes a 7B model with LoRA (rank 16, `α=32`, attention+MLP) on a single 24 GB GPU. To fit, they enable gradient checkpointing. The run launches; `print_trainable_parameters()` says `trainable params: 4,194,304 || trainable%: 0.21` — they relax. Six hours later the eval is identical to base.

The post-mortem, with one grad audit, takes two minutes:

| Step | Instrument | Reading | Verdict |
| --- | --- | --- | --- |
| 1 | `print_trainable_parameters()` | `4.2M, 0.21%` | adapter inserted ✓ |
| 2 | `named_modules()` for `lora_A` | `224 layers` | adapter wired ✓ |
| 3 | `A.grad.norm()` after backward | `None` | **gradient missing ✗** |
| 4 | overfit-one-batch loss | `2.31 → 2.30` flat | not learning ✗ |

Steps 1–2 *would have passed* the section-2 checks — this is why the section-2 trainable-count gate alone is necessary but not sufficient. Step 3 is the one that catches it. The fix is `base.enable_input_require_grads()` before `get_peft_model`. After: `A.grad.norm()` reads `0.6`, overfit-one-batch collapses `2.31 → 0.03`, and the real run finally moves. The cost of the bug was one wasted six-hour run, roughly **\$12 of GPU time** at \$2/GPU-hour, and a day of confusion — all preventable by a two-minute grad audit before launch. Figure 6 lays out that pre-flight sequence as a checklist.

![Timeline of a four-step LoRA pre-flight check running the trainable count, the named modules scan, the gradient after backward, and the weight delta on an overfit batch, with a branch where any failed step blocks the run](/imgs/blogs/debugging-lora-and-peft-6.png)

The pre-flight in Figure 6 is the whole discipline distilled: four cheap reads, in order, that bracket the entire adapter path — does it exist, is it wired, does grad reach it, does the weight actually move — and any failure stops you *before* the long run, not after.

When you want to *see* the gradient reaching every adapter (not just a sampled few), a per-layer grad-norm print after one backward is the most informative single instrument. It catches the checkpointing bug, partial wiring (some layers adapted, some not), and dtype underflow in one view:

```python
import torch

def lora_grad_report(model, batch, top=8):
    model.train(); model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    rows = []
    for n, p in model.named_parameters():
        if "lora_" in n and p.requires_grad:
            g = None if p.grad is None else p.grad.norm().item()
            rows.append((n, g))
    n_none = sum(1 for _, g in rows if g is None)
    n_zero = sum(1 for _, g in rows if g == 0.0)
    print(f"LoRA params: {len(rows)} | grad None: {n_none} | grad ==0: {n_zero}")
    # Healthy: 0 None, B-params nonzero, a few A-params ~0 only at step 0.
    for n, g in rows[:top]:
        print(f"  {g!s:>12}  {n}")
    assert n_none == 0, "Some adapter params got NO gradient — checkpointing/dtype/wiring bug."
```

A healthy report at step 0 shows `grad None: 0`, every `lora_B` with a real norm (say `0.2`–`1.5`), and the `lora_A` norms near zero (the warm-up from section 3). A checkpointing bug shows `grad None: 224` — *all* adapters dark. A *partial* wiring bug (you targeted only `q_proj` and `v_proj` but expected all four) shows the count of `lora_` params lower than you expect, which the `len(rows)` line surfaces. This one function answers "is the adapter actually learning, everywhere?" in a single backward pass.

## 6. Dtype mismatches: fp16, bf16, and the 4-bit base (QLoRA)

QLoRA — LoRA on a 4-bit-quantized base — is the most memory-efficient finetuning recipe and the most dtype-hazardous. The base lives in 4-bit NF4, the adapter lives in a higher precision, and the compute happens in a third precision. Get the relationships wrong and you get either no learning (grad underflows or the adapter is frozen) or numerical garbage. Figure 7 is the audit grid.

![Grid crossing base precision against adapter and compute dtype, with the healthy four-bit base and bf16 adapter cell marked good and the broken fp16 adapter and wasteful fp32 compute cells flagged](/imgs/blogs/debugging-lora-and-peft-7.png)

### The science: three precisions, three failure modes

In QLoRA (Dettmers et al. 2023), the base weights are stored in 4-bit and *dequantized on the fly* to a compute dtype (bf16) for the matmul; the gradient never touches the 4-bit weights because they are frozen. The adapter ($A$, $B$) is kept in bf16 and is the only thing that trains. Two precision facts make or break this:

**fp16 vs bf16 range.** fp16 has a 5-bit exponent; its smallest normal positive number is about $6.1\times10^{-5}$. Anything smaller underflows to zero. LoRA gradients on a large frozen base — especially early, when $B \approx 0$ and the adapter's contribution is tiny — can be very small. In fp16 they can underflow to exactly zero, and a zero gradient is a non-update. bf16 has an 8-bit exponent (the same range as fp32, down to about $1.2\times10^{-38}$) and a smaller 7-bit mantissa, trading precision for range. For LoRA/QLoRA, **bf16 is the safe default** precisely because the range protects small adapter gradients from underflow. If you must use fp16, you need loss scaling (a `GradScaler`), and you should read the gradient histogram to confirm nothing is pinned at zero. This is the same underflow physics covered in depth in [mixed-precision debugging fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16) — LoRA just makes the stakes sharper because the adapter is *all* you have.

**The frozen-adapter trap.** If you load a model in fp16 and the adapter inherits fp16 with `requires_grad` accidentally off, or if autocast casts the adapter into a context where its grad cannot accumulate, the adapter never moves. The signature is, again, `trainable` correct but loss flat. The audit grid in Figure 7 names the two broken cells: a 4-bit base with an fp16/misconfigured adapter (overflow or no grad, loss flat) and a full base with fp32 compute (works but wastes the memory savings — slow, no benefit).

To put real numbers on the underflow risk: consider a LoRA gradient component of magnitude $3\times10^{-6}$ — entirely plausible early in training when the adapter's contribution is small and the per-token loss gradient is diluted across a long sequence. In fp16, whose smallest *normal* positive value is $\approx 6.1\times10^{-5}$, that gradient is below the normal range and rounds toward zero (it lands in the subnormal range or flushes to zero depending on hardware flags). A gradient of exactly zero is a non-update. Accumulate that across many small components and the adapter crawls or stalls. The standard fix in fp16 training is *loss scaling*: multiply the loss by a large factor $S$ (e.g. $2^{16}$) before backward, which scales every gradient up by $S$ into fp16's representable range, then divide the gradients by $S$ before the optimizer step. `torch.amp.GradScaler` does this with dynamic adjustment. bf16, by contrast, shares fp32's exponent range and does **not** need loss scaling at all — small gradients stay representable — which is exactly why bf16 is the recommended LoRA/QLoRA dtype on hardware that supports it (Ampere and newer). The trade is bf16's 7-bit mantissa (≈3 decimal digits) versus fp16's 10-bit mantissa, but for additive low-rank updates the range matters far more than the extra mantissa bits. The deep version of this argument lives in [mixed-precision debugging fp16 vs bf16](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16); here the upshot is a one-liner: **for LoRA, prefer bf16; if forced into fp16, use a GradScaler and check the gradient histogram for a spike at zero.**

### The diagnostic: print the dtype of every trainable param

```python
import torch
from collections import Counter

def audit_dtypes(model):
    train_dtypes = Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)
    frozen_dtypes = Counter(str(p.dtype) for p in model.parameters() if not p.requires_grad)
    print("trainable param dtypes:", dict(train_dtypes))   # want bf16 (or fp32)
    print("frozen param dtypes:   ", dict(frozen_dtypes))  # 4-bit base shows as uint8

    # LoRA adapter params should be float and trainable.
    for n, p in model.named_parameters():
        if "lora_" in n:
            assert p.requires_grad, f"{n} is frozen — adapter won't train!"
            assert p.dtype in (torch.bfloat16, torch.float32), \
                f"{n} is {p.dtype}; prefer bf16 for stable LoRA grads"
    print("OK: adapter params are trainable and in a safe dtype.")
```

A correct QLoRA setup loads the base in 4-bit and the adapter in bf16:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # compute/dequant dtype
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    "your/base-7b", quantization_config=bnb, torch_dtype=torch.bfloat16,
)
base = prepare_model_for_kbit_training(base)  # input grads + ln fp32 + ckpt
model = get_peft_model(base, lora_config)     # adapters added in bf16
audit_dtypes(model)
```

The `bnb_4bit_compute_dtype=torch.bfloat16` line is the one people omit; leaving it at the fp16 default on hardware that prefers bf16 (Ampere and newer) is a common source of subtle instability. Match your compute dtype to your hardware and keep the adapter in bf16.

#### Worked example: the QLoRA memory-and-dtype budget

It helps to see why QLoRA's dtype choices exist by counting memory, because the whole recipe is a memory optimization and the dtype bugs are the price of that optimization. Take a 7B model finetuned on a single 24 GB GPU. In bf16 full finetuning, the memory bill is roughly: weights $7\text{B} \times 2\text{ bytes} = 14\text{ GB}$, gradients another $14\text{ GB}$, and Adam optimizer state (two moments in fp32) about $7\text{B} \times 8\text{ bytes} = 56\text{ GB}$ — over 80 GB before activations. It does not fit, not even close. LoRA removes the gradient and optimizer cost on the base entirely (the base is frozen, so no grad, no optimizer state for it) — you pay optimizer state only on the ~4 M adapter params, a rounding error. That drops the bill to the 14 GB of frozen bf16 weights plus a few hundred MB of adapter + activations. QLoRA goes further: store the frozen base in 4-bit ($7\text{B} \times 0.5\text{ bytes} \approx 3.5\text{ GB}$), dequantize tile-by-tile to bf16 only for each matmul, and keep the bf16 adapter trainable. Now the base costs ~3.5 GB and the whole run fits in well under 24 GB.

The dtype bugs are exactly the failure modes of that pipeline. The 4-bit storage is *frozen* — gradient never touches it, which is why merging into it (section 7) is invalid. The *compute* dtype is what the dequantized matmul runs in; set it to bf16 to keep the adapter gradients in a safe range. The *adapter* dtype is bf16 and trainable. Mismatch any of the three and you get the section-6 failures: an fp16 compute dtype risks underflowing the small adapter gradients; an accidentally-frozen adapter trains nothing; an fp32 compute dtype works but throws away the speed and memory the whole recipe exists to buy. The memory math is *why* QLoRA is worth the dtype care — and the `audit_dtypes` function above is how you confirm you actually got the cheap, working configuration rather than an expensive broken one.

## 7. Merge-and-unmerge bugs

You trained the adapter; it works; now you want to ship. You have two choices: ship the small adapter separately (load base + adapter at inference) or *merge* the adapter into the base weights to get a single standalone model. Merging is where a working adapter gets silently corrupted. Figure 8 shows a merge that breaks versus one that matches.

![Before and after comparison: merging a four-bit base produces logits that drift and an allclose that fails, while merging in bf16 produces logits that match the adapter model within tolerance on every test input](/imgs/blogs/debugging-lora-and-peft-8.png)

### The science: merge = fold scale·BA into W, exactly

`merge_and_unload()` computes $W' = W + \frac{\alpha}{r} B A$ and replaces the LoRA layer with a plain linear holding $W'$. If done in sufficient precision, the merged model is *mathematically identical* to the adapter model — `model(x)` produces the same logits whether you merge or not. So the merge has a built-in test: **the merged model must match the adapter model on the same inputs, within numerical tolerance.** If it does not, the merge is buggy. Three ways it goes wrong:

**Merging a 4-bit base.** You cannot faithfully add a bf16 $\frac{\alpha}{r}BA$ into a 4-bit NF4 weight — the 4-bit grid cannot represent $W + \Delta W$, so the merge either errors, silently re-quantizes (losing the adapter's correction), or produces a base whose effective weights drifted. The fix is to **dequantize to bf16/fp32 first, then merge**, then optionally re-quantize for serving. Always test allclose after.

**Double-merge.** Calling `merge_adapter()` twice (or `merge_and_unload()` after a manual `merge_adapter()`) adds $\frac{\alpha}{r}BA$ to $W$ *twice*, doubling the correction and producing a model that overshoots in the adapter's direction. The signature is outputs that are *more* adapted than the adapter model — over-formatted, repetitive, off. Track merge state and merge exactly once.

**Saving merged when you meant adapter (or vice versa).** `save_pretrained()` on a PEFT model saves *only the adapter* (a few MB of `adapter_model.safetensors`). `save_pretrained()` after `merge_and_unload()` saves the *full merged model* (many GB). Confusing the two ships either a base-sized blob you thought was an adapter, or an "adapter" directory that is missing the base. The fix is to be explicit about which artifact you intend and to verify the size and the file list.

The two-artifact decision is a real design choice, not just a footgun, and it is worth stating the trade-off so you pick deliberately. Shipping the **adapter only** keeps the artifact tiny (a few MB), lets you swap many adapters onto one shared base in memory (multi-tenant serving), and keeps the base auditable and unchanged — but every consumer must have the *exact* base checkpoint and attach the adapter at load, which adds a step and a version-coupling risk. Shipping the **merged model** gives a single standalone artifact that loads like any other model and serves with no PEFT dependency — but it is base-sized (many GB per adapter), loses the ability to hot-swap, and, critically, is only correct if the merge was numerically faithful. The rule of thumb: keep the adapter for experimentation and multi-adapter serving; merge for a single, frozen production model where you want zero PEFT runtime dependency — and *always* allclose-test the merge before you trust it. A useful guardrail is to assert on the saved file size: an "adapter" directory over, say, 500 MB for a small rank is a sign you accidentally merged or set `modules_to_save` on something huge.

### The diagnostic: merge-equivalence test

```python
import torch

@torch.no_grad()
def assert_merge_equivalent(peft_model, tokenizer, prompts, atol=1e-3):
    peft_model.eval()
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(peft_model.device)

    # 1) logits from the adapter model (base + LoRA, unmerged)
    ref = peft_model(**enc).logits

    # 2) merge a COPY, get logits, compare
    merged = peft_model.merge_and_unload()   # folds scale*B*A into W
    merged.eval()
    got = merged(**enc).logits

    max_diff = (ref - got).abs().max().item()
    print(f"max logit diff merged vs adapter: {max_diff:.2e}")
    assert torch.allclose(ref, got, atol=atol), (
        f"Merge changed outputs (max diff {max_diff:.2e} > {atol}). "
        "Did you merge a 4-bit base, double-merge, or merge in fp16?"
    )
    print("OK: merged model matches adapter model.")
```

Run this on 8 short prompts before you ship the merged weights. A `max diff` of `1e-4` or smaller is expected float noise; a `max diff` of `0.4` (Figure 8's "before") means the merge corrupted something — almost always the 4-bit-base case. Dequantize first, re-run, and it drops back into the noise floor.

#### Worked example: the merge that drifted 0.4 logits

A QLoRA adapter trains well — eval up 9 points over base. The engineer calls `merge_and_unload()` directly on the 4-bit model to produce a deployable checkpoint, saves it, and serves it. Users report the served model is *worse* than base, not better. The merge-equivalence test, run after the fact, reports `max logit diff merged vs adapter: 4.1e-01` — far outside `atol=1e-3`. The merge folded a bf16 $\frac{\alpha}{r}BA$ into NF4-quantized weights and the re-quantization scrambled both the base and the correction.

The fix: reload the base in bf16 (not 4-bit), attach the trained adapter, then merge in bf16, then save. Re-running the test: `max logit diff: 7.3e-05` — clean. The corrected merged model now matches the adapter model and reproduces the +9-point eval. The honest measurement here is the allclose on held-out prompts: it is the *only* check that proves the merge preserved behavior, and it costs one forward pass. Never ship a merge you have not allclose-tested.

Laid out as a before→after, the merge bug is unambiguous:

| Instrument | Buggy (merged 4-bit base) | Fixed (merged in bf16) |
| --- | --- | --- |
| `max logit diff` merged vs adapter | `4.1e-01` | `7.3e-05` |
| `torch.allclose(atol=1e-3)` | `False` | `True` |
| Eval vs base | `−3 pts` (regressed) | `+9 pts` (matches adapter) |
| Artifact | corrupted merged weights | faithful standalone model |

The `7.3e-05` residual in the fixed column is not zero, and it should not be — it is the expected floating-point rounding of folding a bf16 product into bf16 weights, well inside the `1e-3` tolerance. The point of the tolerance is to distinguish *float noise* (fine) from *a real behavioral change* (a bug). If you ever see a "merge" that produces *bit-exact* identical logits, be slightly suspicious too — it may mean the merge silently did nothing (a no-op merge on an adapter that was itself a no-op). The healthy signature is a tiny-but-nonzero diff: the merge did real arithmetic and got the same answer to within rounding.

## 8. Saving, loading, and `modules_to_save`

Two adjacent bugs round out the family: forgetting that the adapter alone is not a model, and forgetting to make newly-needed full layers trainable.

**The adapter is not a model.** `PeftModel.save_pretrained("out/")` writes `adapter_config.json` and `adapter_model.safetensors` — typically a few MB. It does **not** write the base weights. To run it, you load the base *and* attach the adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("your/base-7b")  # same base, same dtype
model = PeftModel.from_pretrained(base, "out/")             # attaches the adapter
tok = AutoTokenizer.from_pretrained("out/")                 # if you saved it there
```

The common failures: (1) loading the adapter onto a *different* base than you trained on (different revision, different quantization) — the adapter's correction was computed against specific $W$ values and is meaningless against others; (2) forgetting to save the tokenizer, especially after adding special/chat tokens, so inference tokenizes differently than training; (3) shipping the adapter directory and assuming someone can run it without the base. Save the tokenizer alongside the adapter and document the exact base checkpoint.

The base-mismatch failure (1) deserves emphasis because it is subtle and silent. The adapter is $\frac{\alpha}{r}BA$, a correction *relative to the specific $W$ it was trained against*. Attach that same adapter to a different base — a different revision of the weights, a different quantization scheme (the adapter trained against bf16 weights but you loaded a 4-bit base, or vice versa), or even the same model from a different upload with permuted layers — and the correction is applied on top of values it was never computed for. The result is not a crash; it is a model that is *slightly wrong everywhere*, often worse than either the clean base or a correctly-loaded adapter. The diagnostic is again allclose-style: if you have the training environment, compare the loaded adapter-model's logits on a few prompts against what you saw at the end of training; a large drift means the base does not match. The defensive practice is to record the exact base model id, revision/commit hash, and dtype in your adapter's `adapter_config.json` (PEFT stores `base_model_name_or_path` for you) and to load with that exact spec. Treating "the base" as fungible is one of the most common ways a perfectly-trained adapter underperforms in production.

**`modules_to_save` for new tokens and heads.** If you added tokens to the tokenizer (chat template tokens, special markers) and resized the embeddings, the embedding and LM-head rows for those new tokens *must* train as full parameters — LoRA's low-rank update on the existing embedding matrix cannot conjure rows for tokens that did not exist. The same applies to a fresh classification head on a base that never had one. `peft` handles this with `modules_to_save`, which keeps the named modules fully trainable (and saves their full weights in the adapter checkpoint):

```python
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    modules_to_save=["embed_tokens", "lm_head"],  # full-train these
    task_type="CAUSAL_LM",
)
```

Now `print_trainable_parameters()` will read *higher* than the bare-adapter ~0.2% (because two large modules are fully trainable) — and that is correct and expected here. The bug signature in the *other* direction is: you added tokens, did *not* set `modules_to_save`, and the model never learns to use the new tokens (their embeddings stay at their random init forever, because no gradient updates an untrained-and-unsaved frozen row). The fix is to add the embeddings and head to `modules_to_save`, and to confirm the saved adapter directory now contains those full tensors (the file is larger).

There is a tie-in worth making explicit between `modules_to_save` and the embedding-resize ordering, because the order of operations matters. You must `model.resize_token_embeddings(len(tokenizer))` on the *base* model **before** `get_peft_model`, so the resized embedding/head exist when PEFT wraps the model and `modules_to_save` can find them. Resize after wrapping and you can end up with a mismatch between the tokenizer's vocab size and the embedding matrix's rows, which surfaces as an index error at the first batch containing a new token — or, worse, silently maps a new token to a stale row. The robust recipe: load base → resize embeddings to the new tokenizer → build `LoraConfig` with `modules_to_save=["embed_tokens","lm_head"]` → `get_peft_model` → verify the trainable count jumped by roughly `2 * d * num_new_tokens` plus the adapter params. If the count did *not* jump, `modules_to_save` did not match the module names (same naming hazard as `target_modules`), and your new tokens will not learn.

### Stress-testing the diagnosis: what if it isn't what it looks like

A good debugger asks "what would I see if my hypothesis is wrong?" Run the LoRA no-op hypothesis through the stress cases:

- **What if the data, not the adapter, is bad?** Then overfit-one-batch on four *clean, hand-checked* examples should still pass for a wired adapter — overfitting four examples does not require good data, only a functioning graph. If overfit-one-batch fails with `trainable = 4.2M` and grad flowing, *then* suspect the data collator (labels all `-100`, so there is no loss signal — a different bug, covered in the loss-masking material). The grad audit separates them: no grad → wiring; grad flows but loss flat → loss has no signal (check the labels).
- **What if it only fails at fp16, not bf16?** Then you have the underflow/scaling bug of section 6, not a wiring bug. The tell: switch the adapter and compute dtype to bf16, re-run overfit-one-batch; if it now learns, it was numerics. A wiring no-op fails identically in every dtype.
- **What if the batch is tiny (batch size 1)?** A wiring no-op is independent of batch size — it fails at batch 1 and batch 256 alike. If a run "works at batch 256 but not batch 1," that is *not* a LoRA wiring bug; look at batch-norm-like statistics, dropout, or a collator that misbehaves on singletons.
- **What if it only fails on multi-GPU?** Under `DistributedDataParallel`, a LoRA model with *unused* adapter branches (e.g. you targeted a module that the forward pass skips for some inputs) can trip the `find_unused_parameters` error, because DDP expects every trainable param to receive grad every step. That is a systems-layer interaction, not the single-GPU no-op. The fix there is `find_unused_parameters=True` (at a throughput cost) or, better, not targeting modules that are conditionally skipped. A single-GPU no-op, by contrast, reproduces on one device — so reproduce on one GPU first.

Each of these is the bisection in miniature: change exactly one variable (data, dtype, batch size, device count) and watch whether the symptom moves. If the symptom is invariant to a variable, that variable is not the cause. A LoRA wiring no-op is invariant to *all* of them — same flat loss everywhere — which is itself a strong fingerprint.

## 9. Rank and alpha: when it trains but underfits

Everything so far has been about a *no-op* — zero learning. The subtler regime is an adapter that *is* learning but learning *poorly*, and the usual suspects are rank and the $\alpha/r$ scale. This is genuinely an optimization/capacity question, not a graph bug, so the instruments are different: overfit-one-batch *passes* (the adapter has enough capacity to memorize a handful of examples), but the real run underfits.

**Rank too small.** Rank $r$ caps the dimensionality of the correction the adapter can express. For light style/format adaptation, $r=8$ is plenty. For teaching genuinely new behavior or a large domain shift, $r=8$ can be a real bottleneck; $r=16$–$64$ gives more headroom. The diagnostic: if overfit-one-batch passes but training loss plateaus above where you expect and *raising rank lowers that plateau*, you were rank-limited. If raising rank does nothing, rank was not the constraint — look at LR or data.

**Scale too small (the $\alpha/r$ trap).** Recall the scale is $\alpha/r$. If you raise $r$ for capacity but hold $\alpha$ fixed, the scale *drops* and the adapter's effective contribution shrinks, which can look like "more capacity made it learn less." The fix is to scale $\alpha$ with $r$ (a common rule is $\alpha = 2r$) so capacity and magnitude move together, or use an implementation with $\alpha/\sqrt{r}$ scaling. The diagnostic is arithmetic: print `alpha/r` and check it is in a sane range (often 1–2); if you changed $r$, recompute it.

| Symptom | Overfit-one-batch | Likely cause | Test | Fix |
| --- | --- | --- | --- | --- |
| Zero learning, flat loss | **fails** | adapter not in graph | trainable count / grad audit | fix `target_modules` / `enable_input_require_grads` |
| Learns but underfits | **passes** | rank too small | raise `r`, watch plateau | `r` 16–64 |
| "More rank, worse" | passes | $\alpha/r$ scale dropped | print `alpha/r` | set `alpha = 2r` |
| Learns then forgets base skills | passes | LR too high / too many epochs | eval on held-out base task | lower LR, 1–3 epochs |

That last row is a reminder that even a perfectly wired adapter can ruin a model by overfitting or inducing catastrophic forgetting — which is the domain of [finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it). LoRA *reduces* forgetting versus full finetuning (the base is frozen) but does not eliminate it, especially at high rank and LR.

The learning rate for LoRA deserves its own note because the intuition transfers poorly from full finetuning. LoRA learning rates are typically *higher* than full-finetune LRs — often `1e-4` to `3e-4`, versus `1e-5`–`2e-5` for full finetuning of the same model. The reason is that the adapter starts from zero (and a random $A$) rather than from carefully pretrained weights, so it can tolerate (and needs) a larger step to move meaningfully, and there are far fewer parameters to destabilize. A common mistake is to copy a full-finetune LR of `2e-5` onto a LoRA run and conclude "LoRA learns slowly" — it does, at that LR, because the LR is 5–10× too small for the adapter. The diagnostic is a quick LR sweep on overfit-one-batch: try `1e-5, 3e-5, 1e-4, 3e-4, 1e-3` and watch which one drives four examples to near-zero fastest without spiking. If the fastest is `3e-4` and you were running `2e-5`, you were starving the adapter, not hitting a capacity wall. This separates "LR too low" (raise the LR) from "rank too small" (raise $r$) from "no-op" (fix the wiring) — three different fixes for what all look like "it won't learn."

#### Worked example: "more rank made it worse"

An engineer reports: "I bumped rank from 16 to 64 to give the model more capacity for a hard domain-adaptation task, and the loss got *worse*, not better." This is the $\alpha/r$ trap in the wild. Their config had `r=16, lora_alpha=32` (scale `32/16 = 2.0`). Bumping to `r=64` while leaving `lora_alpha=32` drops the scale to `32/64 = 0.5` — a 4× reduction in the adapter's effective contribution. More capacity, *less* magnitude, slower learning at the same LR and step budget. The fix is to scale $\alpha$ with $r$: set `lora_alpha=128` for `r=64` to keep the scale at `2.0`, or switch to an RSLoRA-style `α/√r` scaling that holds magnitude roughly constant as rank grows. After matching the scale, rank 64 trains at least as fast as rank 16 and reaches a lower plateau on the genuinely high-rank task — confirming it *was* rank-limited, just masked by the scale drop. The lesson: when you change $r$, recompute $\alpha/r$ and decide whether you meant to change the scale. Print it: `print("scale =", lora_alpha / r)`.

## 10. The one pre-flight function that catches it all

Every diagnostic in this post is cheap and runs before any long training. The discipline is to bundle them into a single `assert`-style pre-flight you call right before `trainer.train()`. If it passes, you have ruled out the entire LoRA no-op family; if it fails, it tells you exactly which link broke. Here is the consolidated version that wires together the trainable count, the module scan, the grad audit, and the weight-delta check:

```python
import torch

@torch.no_grad()
def _snapshot(model):
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if "lora_B" in n}

def lora_preflight(model, batch, optimizer, expected_min_pct=0.05):
    # 1. trainable count is sane
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * train / total
    print(f"trainable: {train:,} ({pct:.3f}%)")
    assert train > 0, "NO-OP: trainable params = 0 (target_modules / wrong variable)"
    assert pct >= expected_min_pct, f"trainable {pct:.3f}% looks under-targeted"

    # 2. LoRA layers actually inserted
    n_lora = sum(1 for n, _ in model.named_modules() if "lora_A" in n)
    assert n_lora > 0, "NO-OP: no lora_A modules found"
    print(f"lora layers: {n_lora}")

    # 3. grad reaches every adapter after one backward
    before = _snapshot(model)
    model.train(); model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    dark = [n for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad and (p.grad is None)]
    assert not dark, f"NO GRAD into {len(dark)} adapter params (checkpointing/dtype)"

    # 4. an optimizer step actually moves B
    optimizer.step()
    moved = sum(1 for n, p in model.named_parameters()
                if "lora_B" in n and not torch.equal(p, before[n]))
    assert moved > 0, "STEP did not move any lora_B param (optimizer/param-group bug)"
    print(f"OK: {moved} lora_B params moved after one step. Adapter is LIVE.")
```

Call `lora_preflight(model, one_batch, optimizer)` and read the printout. A clean run prints a sane trainable percentage, a positive LoRA-layer count, no dark params, and a positive moved count — and then you launch the long run with confidence. A broken run fails on exactly the link that is broken, with a message that names the suspect. This is the entire post compressed into thirty lines, and it costs one forward-backward-step — a fraction of a second — against the six hours a silent no-op would otherwise waste.

## 11. Case studies and known signatures

Five patterns that show up repeatedly in real LoRA/PEFT work, named so you recognize them.

**The Llama-config-on-GPT-2 zero.** The single most common no-op in the wild: a `target_modules=["q_proj","v_proj"]` config copied from a Llama tutorial, applied to a GPT-2-style model whose attention is a fused `c_attn` (a `Conv1D`, not even a `Linear`). Nothing matches, `trainable = 0`, the run is an identity function. The tell is `print_trainable_parameters()` reading `0.0000%`. The fix is to inspect the real module names (the `Counter` snippet in section 2) and target `c_attn` (and note `peft` supports `Conv1D` via `fan_in_fan_out=True` for those layers). This is not exotic — it is the default outcome of trusting a config across architectures.

**The QLoRA-without-`prepare_model_for_kbit_training` flat run.** Reported across many QLoRA threads: a 4-bit base, LoRA attached, `trainable` correct, loss dead flat. The omission is `prepare_model_for_kbit_training` (or at minimum `enable_input_require_grads()` with gradient checkpointing on). The grad audit shows `A.grad is None`; adding the prep call restores gradient and the loss moves. This is section 5 and 6 colliding, and it is why every official QLoRA example starts with that prep call.

**The merge that degraded a shipped model.** A QLoRA adapter that improved eval, then `merge_and_unload()` straight on the 4-bit base, then served — and the served model regressed below base. The merge-equivalence allclose (section 7) reports a large logit drift; dequantizing to bf16 before merging fixes it. The lesson is that a working adapter is not a working *merged* model until you have allclose-tested the merge.

**The added-tokens-without-`modules_to_save` non-learner.** A chat finetune that adds `<|im_start|>`/`<|im_end|>`-style tokens, resizes embeddings, trains LoRA — and the model never reliably emits the new turn tokens. Their embedding rows were never trainable (LoRA does not touch them and they were not in `modules_to_save`), so they stayed at random init. Adding `embed_tokens` and `lm_head` to `modules_to_save` lets those rows learn; the saved adapter directory grows accordingly. This one is easy to miss because *most* of the model trains fine — only the new tokens are dead.

**The DDP `find_unused_parameters` crash that looks like a LoRA bug.** A run that trains fine on one GPU throws `RuntimeError: Expected to have finished reduction in the prior iteration` the moment it goes multi-GPU under `DistributedDataParallel`. The cause is a LoRA adapter on a module whose output does not contribute to the loss on every step (a conditionally-skipped branch, or a head only used for some inputs), so its parameters get no gradient and DDP's reducer hangs waiting for them. It *looks* like a LoRA wiring problem but is a systems-layer interaction. The fix is either `find_unused_parameters=True` (a throughput cost) or, cleaner, not targeting modules that are conditionally skipped. The fingerprint that separates it from a true no-op: it reproduces *only* with more than one rank, never on a single GPU.

These trace to real, documented behavior of `peft`, `transformers`, and `bitsandbytes` (the LoRA paper, Hu et al. 2021; the QLoRA paper, Dettmers et al. 2023; and the `peft` documentation's PEFT-model and quantization guides). Where this post gives specific loss/logit numbers (`2.31 → 0.04`, `max diff 4.1e-01`), treat them as representative figures from this class of run, not measurements of one canonical benchmark — the *pattern* is what is robust, and you should read your own instruments to get your own numbers.

A closing note on the PEFT family beyond vanilla LoRA, because the same debugging discipline transfers with small adjustments. **QLoRA** adds the 4-bit-base dtype hazards of section 6 — the diagnostic is the same trainable-count + grad-audit, plus the `prepare_model_for_kbit_training` check. **DoRA** (weight-decomposed LoRA) splits the update into a magnitude and a direction; it adds a magnitude vector to the trainable set, so the trainable count is slightly higher and the same count/grad audit still applies. **AdaLoRA** prunes rank during training, so the *effective* trainable rank changes over the run — do not be alarmed if a per-layer rank shrinks; that is by design, and the no-op check is still "is anything trainable and getting grad." **Prefix/prompt tuning** and **IA³** are different parameterizations but the same core question governs them all: *is the small set of new parameters actually in the graph, getting gradient, and moving?* The instruments — `print_trainable_parameters()`, the grad audit, overfit-one-batch — are method-agnostic. Whatever PEFT variant you reach for, the first four lines of debugging are identical.

## 12. When this is (and isn't) your bug

A LoRA no-op has a sharp signature, and knowing when a symptom points *elsewhere* saves you from fixing the wrong thing.

**It IS a LoRA/PEFT bug when:** `print_trainable_parameters()` reads `0` or `0.00%`; or the trainable count is fine but a grad audit shows `lora_A.grad is None`; or overfit-one-batch fails to drive a four-example loss toward zero; or the merged model's logits diverge from the adapter model's. These are graph/wiring/dtype problems — model code, in the six-places frame — and they are *deterministic*. The same config no-ops every time.

**It is NOT a LoRA wiring bug when:** the adapter trains (overfit-one-batch passes, grad flows) but the *full run* underfits or the eval is bad. That is capacity (rank/alpha), optimization (LR, epochs), or data — a different track. It is also not a LoRA bug when the loss goes down nicely but the model "forgot" a base skill — that is catastrophic forgetting from LR/epochs, addressed in the finetuning post. And a *smooth-then-NaN* curve is numerics (fp16 overflow, a bad batch), not a wiring no-op — a no-op never NaNs, it just sits flat. The cleanest separator is overfit-one-batch: **if it passes, stop blaming the adapter wiring and look at data, LR, rank, or eval.** If it fails with `trainable = 0`, you have a no-op and the optimizer is innocent.

Here is the symptom-to-suspect map distilled, so you can route a complaint in one glance:

| Symptom | First instrument | Most likely place | Not this |
| --- | --- | --- | --- |
| `trainable = 0` / `0.00%` | `print_trainable_parameters()` | model code (no adapter / wrong var) | not LR, not data |
| trainable fine, loss dead flat | grad audit (`A.grad`) | numerics/systems (ckpt, dtype) | not rank, not LR |
| trains but plateaus high | overfit-one-batch (passes) | capacity (rank, $\alpha/r$) | not wiring |
| trains, then forgets base skills | held-out base eval | optimization (LR, epochs) | not wiring |
| smooth then `NaN` | grad-norm history | numerics (fp16 overflow) | not a no-op |
| merged ≠ adapter outputs | `allclose` merged vs adapter | merge/dtype | not training |
| breaks only on >1 GPU | single-GPU repro | systems (DDP unused params) | not single-GPU wiring |

The discipline this table encodes is the heart of the series: a symptom names a *suspect place*, you confirm with a *cheap test*, and only then do you touch code. A LoRA no-op occupies the top two rows — model code and numerics/systems — and the instrument that distinguishes them is whether the trainable count is zero (no adapter) or fine-but-gradless (adapter present, gradient blocked). Everything below those rows is a different track entirely, and the fastest way to waste a day is to apply a rank or LR fix to a wiring bug, or a wiring fix to a capacity bug. Read the right instrument, route to the right place, fix the right thing.

The general version of this discipline — read the trainable count and grad flow before suspecting the optimizer — is an instance of [your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think), the gradient-flow-audit post that LoRA debugging specializes. For the broader quantization context that QLoRA dtype bugs live in, see the edge-AI series' [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles).

## 13. Key takeaways

- **Always call `print_trainable_parameters()` first.** Nonzero and in the `0.1–1%` band means the adapter is plausibly wired; `0.0000%` is a no-op — a model-code bug, never an optimization one.
- **`target_modules` must match real module names.** When unsure, scan `named_modules()` for the actual `Linear`/`Conv1D` leaf names, or use `target_modules="all-linear"`.
- **Pass the wrapped model to the Trainer.** `assert trainer.model is get_peft_model(...)`. Training the frozen base reads `0` trainable and looks like a slow run.
- **A correct trainable count is necessary but not sufficient.** With gradient checkpointing, also call `enable_input_require_grads()` (or `prepare_model_for_kbit_training` for QLoRA) and *verify `lora_A.grad` is not None after backward*.
- **bf16 for the adapter, bf16 compute for a 4-bit base.** fp16's $6.1\times10^{-5}$ underflow floor can zero out small LoRA gradients; bf16's range protects them.
- **Overfit-one-batch is the universal LoRA sanity check.** A correctly wired adapter drives four examples to near-zero loss in a couple hundred steps. If it does not, the adapter is not learning — bisect to the graph, not the optimizer.
- **Allclose-test every merge.** `merge_and_unload()` must reproduce the adapter model's logits within `~1e-3`. Never merge a 4-bit base; dequantize to bf16 first, merge once, then test.
- **The adapter is not a model.** Save and reload the exact base + the adapter + the tokenizer; add new-token embeddings/heads to `modules_to_save` or they never learn.

## Further reading

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — the original low-rank-update hypothesis and the $B=0$ initialization.
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023) — 4-bit NF4 base, paged optimizers, and the dtype recipe behind `prepare_model_for_kbit_training`.
- Hugging Face `peft` documentation — `LoraConfig`, `get_peft_model`, `target_modules`, `modules_to_save`, and the quantization/PEFT-model guides.
- PyTorch `torch.utils.checkpoint` docs — why a non-grad input severs gradient through a checkpointed segment (the mechanism behind `enable_input_require_grads`).
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the symptom→suspect→test→fix decision tree this post instantiates.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone bisection method and printable checklist.
- [Your model isn't learning what you think](/blog/machine-learning/debugging-training/your-model-isnt-learning-what-you-think) — the general gradient-flow audit that LoRA debugging specializes.
- [Finetuning an LLM without breaking it](/blog/machine-learning/debugging-training/finetuning-an-llm-without-breaking-it) — LR, epochs, and catastrophic forgetting once the adapter actually trains.
