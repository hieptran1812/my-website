---
title: "Finetuning an LLM Without Breaking It: LR, Epochs, and Catastrophic Forgetting"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Diagnose the recipe-level bugs that quietly ruin an LLM finetune — the 100x-too-high learning rate, the extra epochs that memorize your data, and the catastrophic forgetting train loss never shows — and run the probe harness that catches all three early."
tags:
  [
    "debugging",
    "model-training",
    "llm",
    "finetuning",
    "learning-rate",
    "catastrophic-forgetting",
    "deep-learning",
    "pytorch",
    "nlp",
    "hyperparameters",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/finetuning-an-llm-without-breaking-it-1.png"
---

A team I worked with had a clean, simple task: teach a 7B chat model to answer support tickets in their product's voice. They had 4,000 high-quality examples, a single A100, and a Hugging Face `SFTTrainer` config copied from a tutorial. They launched, watched the training loss fall beautifully from 1.8 to 0.05 over ten epochs, shipped the checkpoint, and within an hour the on-call channel lit up. The model answered support tickets fine. It also could no longer do basic arithmetic, refused to follow multi-step instructions it used to handle, and had started repeating the same three sentences verbatim from the training set whenever a ticket looked unfamiliar. The training loss curve was a textbook success. The model was a textbook disaster.

This is the defining trap of LLM finetuning: the instruments that tell you a from-scratch run is healthy actively lie to you on a finetune. A pretrained model arrives already good at thousands of things you are not training on, and the single number you stare at — training loss on your task data — goes *down* precisely when you are destroying those abilities. Three recipe-level choices cause almost all of it: a learning rate borrowed from from-scratch training that is 50–100x too high and detonates the pretrained weights in the first dozen steps; too many epochs, which let the model memorize your small dataset and parrot it back; and catastrophic forgetting, where narrowly optimizing one task quietly erases unrelated general capability. None of the three shows up in train loss. All three show up the moment you run the right held-out probe. Figure 1 is the whole post in one image: the same dataset, the same model, one knob changed, and the difference between a wrecked model and one that learned the task while keeping its mind.

![Side-by-side comparison showing a from-scratch learning rate spiking the loss and collapsing a general probe versus a 50x smaller rate that learns the task and preserves general ability](/imgs/blogs/finetuning-an-llm-without-breaking-it-1.png)

By the end of this post you will be able to take any LLM finetune — full-parameter SFT, LoRA, or a domain adaptation — and answer the only questions that matter: is my learning rate destroying the pretrained basin, am I training long enough to memorize instead of learn, and am I trading general capability for task accuracy without noticing? You will have a runnable `SFTConfig` with defensible defaults, an LR sweep you can run on a subset in twenty minutes, and a forgetting-probe harness that turns an invisible regression into an early-stop signal. This is one post in a series whose spine is simple: a training bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and you *bisect* to the right one before touching code. Finetuning concentrates the danger in two of those places, optimization and evaluation, and we will live there. If you have not seen the master map, start with [the taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the [capstone playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) ties every track together.

## 1. Why a pretrained model is so easy to break

Start with the thing that makes finetuning different from training from scratch, because every bug below follows from it. When you initialize a model randomly and train from zero, the weights begin in a high-loss, gentle region of parameter space. Large steps are *good* there: the loss surface is smooth and shallow, gradients are large and informative, and you want to move fast. A learning rate of `1e-3` with Adam is unremarkable for from-scratch pretraining.

A pretrained model is the opposite. After pretraining on trillions of tokens, the weights sit at the bottom of a deep, comparatively sharp basin. Every capability the model has — grammar, arithmetic, factual recall, instruction-following, code — is encoded in the precise relationships between millions of weight values. The loss surface there is *not* gentle. Near a good minimum, the loss looks locally like a quadratic bowl, and the curvature of that bowl is described by the Hessian $H$. A gradient-descent step of size $\eta$ in a direction with curvature (Hessian eigenvalue) $\lambda$ is only stable if

$$\eta < \frac{2}{\lambda_{\max}}$$

where $\lambda_{\max}$ is the largest eigenvalue of the Hessian — the sharpest direction in the basin. Cross that threshold and the step *overshoots* the minimum, the loss in that direction grows instead of shrinks, and the next gradient is even larger. That is the loss spike, and it is not a metaphor; it is the standard stability condition for gradient descent on a quadratic. Pretrained minima are sharp, meaning $\lambda_{\max}$ is large, meaning the stable $\eta$ is *small*. A learning rate that was fine from scratch because the basin was shallow is now far past $2/\lambda_{\max}$, and the first few steps walk the model straight out of the basin it took a million GPU-hours to find.

This is why the single most common LLM finetuning bug — the wrong learning rate — is also the most destructive. It does not degrade gracefully. It detonates. We dig into the full physics of this in [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem); here we make it concrete for the finetuning case and pair it with the two failures that the LR does *not* cause: memorization and forgetting.

It is worth dwelling on *how sharp* a pretrained basin is, because the sharpness is the whole story. The largest Hessian eigenvalue $\lambda_\max$ of a converged transformer can be large — empirically in the hundreds or more along the sharpest directions — and it is not uniform across directions. The loss surface near a pretrained minimum is highly anisotropic: a few directions are extremely sharp (large $\lambda$), and most directions are nearly flat (tiny $\lambda$). Gradient descent is stable only if $\eta < 2/\lambda$ in *every* direction the gradient touches, so the single sharpest direction sets your maximum stable LR. A from-scratch rate respects the flat directions and violates the sharp ones, which is why the failure is partial and abrupt rather than gradual: the model is fine until a step lands in a sharp direction, and then that direction blows up while the rest of the model looks healthy for a few more steps. That is the mechanism behind the "fine for 20 steps, then a spike" signature — the optimizer wandered into a sharp direction and the stability condition failed there first.

A second consequence of the anisotropy is that the *same* numerical LR is safe on one model and lethal on another, depending on how sharp that model's basin happens to be. A heavily-pretrained, well-converged model has a sharper basin (it found a deeper minimum) than a lightly-pretrained one, so the heavily-pretrained model needs a *smaller* finetuning LR. This is the counterintuitive reason that finetuning a state-of-the-art instruct model is *more* delicate than finetuning a weaker base model — the better the model, the sharper its basin, the smaller the stable step. When someone tells you "I used `5e-5` and it was fine on model X but destroyed model Y," they are usually describing two basins of different sharpness, not a mysterious difference in robustness.

It helps to hold the whole recipe in your head as a stack, because that is how you debug it. Figure 2 lays out the six knobs in the order a bug propagates through them: get the data format wrong and nothing below matters; get the learning rate wrong and you never reach the rest; get epochs wrong and you memorize; get the schedule, batch, and eval wrong in subtler ways. When a finetune misbehaves, you walk this stack top to bottom and ask which layer the symptom belongs to.

![A vertical stack of the six finetuning recipe layers from data and format through learning rate, epochs, schedule, effective batch, and evaluation](/imgs/blogs/finetuning-an-llm-without-breaking-it-2.png)

The reason this stack is worth memorizing is that the three big failures sit at different layers and have different signatures, so once you can read the signature you know which layer to fix. A loss that spikes in the first fifty steps is the learning-rate layer. A loss that glides smoothly to near zero is the epochs layer. A loss that looks perfect while your eval gets worse is the evaluation layer hiding a forgetting problem two layers up. Let us take them one at a time.

Underneath all three sits a single tension that is worth naming, because it explains why there is no setting that is simply "safe": the **stability–plasticity tradeoff**. A model needs *plasticity* to learn your new task — the weights have to move, or nothing is learned. But it needs *stability* to retain what it already knows — the weights must not move so much that old capabilities break. Every knob in the stack is really a plasticity dial. A higher LR, more epochs, full finetuning instead of LoRA, no replay: all of these increase plasticity, which means faster task learning *and* more forgetting, because they are the same thing — moving the weights more. There is no free plasticity. The art of finetuning is buying exactly enough plasticity to learn the task and not one step more, and the only way to know how much is enough is to measure both sides: task learning *and* retention. A debugger who measures only task learning is flying with half the instruments, and the half they are missing is the half that fails silently.

This also explains the deepest "why" of the whole post: **why does train loss hide forgetting?** Train loss measures only the plasticity side — how well the model fits your task data. It is structurally incapable of measuring the stability side, because your task data does not contain the general capabilities you might be losing. If your finetuning set is all support tickets, the train loss never evaluates arithmetic, so arithmetic can rot to nothing and the loss will not move a single decimal. Train loss is not a bad instrument; it is a *partial* one, measuring exactly one of the two things you care about, and it happens to measure the one that is rarely the problem. The forgetting probe exists to measure the other half. You are not replacing train loss; you are completing it.

## 2. The learning rate: a from-scratch rate is a demolition charge

Here are the numbers worth committing to memory, because they are the difference between a working finetune and a destroyed one:

| Scenario | Typical LR (AdamW) | Why |
| --- | --- | --- |
| Pretraining from scratch | `3e-4` to `1e-3` | Shallow basin, large stable steps |
| Full-parameter SFT | `1e-5` to `2e-5` | Sharp pretrained basin, tiny stable steps |
| LoRA / QLoRA SFT | `1e-4` to `3e-4` | Adapter weights start at zero, decoupled scale |
| DPO / preference tuning | `5e-7` to `5e-6` | Even more fragile; tiny updates near a reference |
| Continued pretraining | `1e-5` to `1e-4` | Between scratch and SFT |

The first thing to notice is the 50–100x gap between a from-scratch rate and a full-SFT rate. If you copy `learning_rate=1e-3` from a from-scratch tutorial into a full finetune, you are stepping ~50x past the stable threshold. The second thing to notice is that LoRA's rate is *higher* than full SFT's, which surprises people, and the reason is mechanistic, not arbitrary: LoRA adds low-rank matrices $A$ and $B$ where $B$ is initialized to zero, so the adapter's contribution starts at exactly zero and the effective update to the frozen base weights is scaled by $\alpha/r$. The optimization problem LoRA solves is different — you are not perturbing the sharp base weights directly, you are learning a small additive correction from scratch — so it tolerates and needs a larger rate. Get this backwards (a `2e-5` LoRA run) and your adapter barely moves; the symptom looks like "LoRA does nothing," which we cover in depth in [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

The DPO and preference-tuning row deserves a note because it is the most fragile of all. DPO optimizes the policy relative to a frozen reference model, and the loss is dominated by the *difference* in log-probabilities between chosen and rejected responses. Small weight changes move those log-probs a lot, so the effective sensitivity to the LR is high, and DPO routinely diverges or collapses at rates that would be perfectly safe for SFT. Rates of `5e-7`–`5e-6` are normal there. The general principle holds across all of these: the more the loss depends on *precise* properties of the pretrained weights (precise log-probs, a precise reference policy, a precise sharp minimum), the smaller the safe step. Finetuning is fragile in proportion to how good the starting point is.

### What a too-high finetune actually looks like

The signature is unmistakable once you have seen it. The loss does not slowly drift up. It is fine for a handful of steps — the optimizer's momentum and the warmup mask the instability briefly — and then it spikes, often from 1.5 to 7 or 8 in a single logging interval, and either recovers to a permanently higher plateau (the model is now worse than the base model and never recovers) or climbs to `inf`/`NaN`. Meanwhile, if you generate from the model, the output degrades into repetition or token soup, because the careful weight relationships that produced coherent text have been scrambled.

#### Worked example: watching `1e-3` destroy a 7B model

Take a Qwen2.5-7B-Instruct base, a 4,000-example instruction dataset, AdamW, and a deliberately wrong `learning_rate=1e-3`. Here is the run, logged every 10 steps:

- Step 0: loss 1.82 (the base model's loss on this data — already low because it is a competent instruct model).
- Step 10: loss 1.51. Looks like it is working.
- Step 20: loss 0.94. Looks great, actually.
- Step 30: loss **6.7**. The spike.
- Step 40: loss 8.9.
- Step 50 onward: `nan`.

Now run a general-capability probe — 200 held-out questions spanning arithmetic, factual recall, and instruction-following — at step 20, the moment the task loss looked best: the base scored 62% on it, and at step 20 the model scores **9%**. The model that looked like it was learning fastest had already lost most of its general ability; the low task loss at step 20 was the model overshooting *through* the basin, not settling into it. Drop the LR to `2e-5` and rerun: the loss glides from 1.5 to 0.6 over two epochs with no spike, and the probe holds at 60%. Same data, same everything, one number changed, and the difference is a usable model versus a paperweight. That is Figure 1.

The diagnostic that turns this from a guess into a measurement is an **LR sweep on a subset**. You do not need to run the full job to find a safe rate — train for 50–100 steps on a small slice of the data at several rates and watch which ones spike.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
# A small slice is enough to surface a spike; we are probing stability, not fitting.
subset = load_dataset("your/sft-dataset", split="train[:512]")

def run_lr_probe(lr, max_steps=60):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    cfg = SFTConfig(
        output_dir=f"/tmp/lrprobe_{lr}",
        learning_rate=lr,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch 16
        max_steps=max_steps,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=subset)
    trainer.train()
    # Pull the logged losses; a spike or nan means this LR is unstable.
    losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    spiked = any(
        (l != l) or (l > 3 * losses[0]) for l in losses  # l != l catches NaN
    )
    return losses, spiked

for lr in [5e-4, 2e-4, 5e-5, 2e-5, 5e-6]:
    losses, spiked = run_lr_probe(lr)
    flag = "SPIKE/NaN" if spiked else "stable"
    print(f"lr={lr:>7}: first={losses[0]:.2f} last={losses[-1]:.2f}  [{flag}]")
```

Read the output the way you read a fuse: the highest rate that stays stable *and* still reduces the loss is your ceiling, and you want to run the real job comfortably below it. For full SFT of a 7B model this sweep almost always lands you at `2e-5` or below; for LoRA it lands at `1e-4`–`2e-4`. The sweep costs maybe twenty minutes of GPU time and routinely saves a multi-hour wasted run.

A subtlety in reading the sweep: a rate can be "stable" — no spike, no NaN — and still be *wrong*, in two directions. Too high but below the spike threshold gives you a loss that falls fast but a model that forgets more (we will see why in §4), because every step moves the weights further. Too low gives you a loss that crawls and a finetune that never quite learns the task in the epoch budget you have. The sweep finds the *stability ceiling*; you then pick a rate a notch below it for full SFT, because the goal is not the fastest descent, it is the descent that learns the task while moving the weights as little as possible. "As little as possible" is the through-line of this entire post, and it starts at the LR.

#### Worked example: same rate, two different base models

You have a working recipe for `Mistral-7B-v0.1` (a base, not heavily instruction-tuned model): `learning_rate=5e-5`, 3 epochs, and it finetunes your task cleanly with a smooth loss and a general probe that holds within 3 points. You swap the base for `Qwen2.5-7B-Instruct` — a much more heavily instruction-tuned model — keep `5e-5`, and now the loss has a visible bump in the first 30 steps and the general probe drops 9 points by epoch 1. Nothing about your data or code changed. What happened is the basin-sharpness story from §1: the heavily-instruction-tuned Qwen sits in a sharper minimum, so `5e-5` that was comfortably below the stability ceiling for Mistral is now close to it for Qwen. Drop Qwen to `1e-5`–`2e-5` and the bump vanishes and the probe holds. The lesson is operational: **re-run the LR sweep whenever you change the base model**, even by a "similar" 7B model, because the safe rate is a property of the specific basin, not of the parameter count.

There is one more learning-rate subtlety that bites people, and it lives one layer down in the stack: the LR is coupled to the *effective batch size*, not the per-device batch size. We will get to that in §6, but flag it now — if you change `gradient_accumulation_steps`, you have changed the effective LR-per-example dynamics even though the `learning_rate` field is identical.

## 3. Epochs: how a model memorizes instead of learns

Suppose you fixed the learning rate. The loss is now smooth, no spikes, gliding downward. You are not safe yet, because the second failure produces a *beautiful* loss curve — that is exactly why it is dangerous. Too many epochs cause the model to memorize your finetuning set rather than learn the underlying behavior, and memorization drives training loss to near zero while the model gets *worse* at everything you actually care about.

The mechanism is a counting argument. An instruction dataset of 4,000 examples, each maybe 300 tokens, is on the order of $1.2\times10^6$ tokens. A 7B model has $7\times10^9$ parameters. The model has roughly 6,000 parameters per training token. With that much capacity relative to data, the model does not need to find a general rule to drive the loss down — it can simply store the answer to each specific example. Every additional epoch over the same small set gives the optimizer more chances to overfit those specific sequences. After one or two passes it has learned the *behavior*; after eight passes it has learned the *exact strings*, including their quirks, and it reaches for those strings even when the input does not warrant them. This is why the practical rule for instruction finetuning is **1 to 3 epochs**, and why anything past 3 is a yellow flag you must justify with eval, not with train loss.

Contrast this with pretraining, where the data-to-parameter ratio is inverted: a 7B model is pretrained on trillions of tokens — hundreds or thousands of tokens *per parameter*, the opposite of finetuning's thousands of parameters per token. At pretraining scale, the model *cannot* memorize the corpus even if it wanted to; there is far more data than capacity, so the only way to reduce loss is to learn generalizable structure. This is precisely why pretraining tolerates and benefits from many passes' worth of tokens while finetuning does not: it is not a difference in optimizers or schedules, it is a difference in whether memorization is even *possible*. On a 4,000-example finetuning set, memorization is not just possible, it is the path of least resistance, and gradient descent takes the easy path. Every epoch you add past the point where the behavior is learned is an epoch spent moving from "learned the rule" to "memorized the examples."

There is a measurable fingerprint of memorization worth knowing: as a model memorizes, the gap between training loss and held-out loss widens, and the model's confidence on training examples becomes extreme (near-zero loss, near-one probability on the exact training tokens) while its calibration on new inputs degrades. If you log both train and held-out loss, the divergence point of the two curves is the memorization onset — the same crossover Figure 3 shows, just measured in loss rather than in a probe. Train and held-out loss tracking together means the model is still learning a rule; them peeling apart means it has started storing answers.

The signature is the over-finetune curve, and it is worth being able to read at a glance. Figure 3 traces it: the training loss keeps falling toward zero epoch after epoch, which looks like progress, while a held-out general probe — the only honest instrument here — peels away and drops. The crossover point, where the probe starts falling faster than the task is improving, is where you should have stopped.

![A timeline of training epochs showing training loss collapsing toward zero while a held-out general probe quietly declines, marking the memorization signature](/imgs/blogs/finetuning-an-llm-without-breaking-it-3.png)

#### Worked example: the eight-epoch support-bot

Return to the support-ticket model from the intro. Here is what its instruments read across epochs, with a held-out general probe (200 questions, base = 62%) and a held-out *task* eval (500 tickets the model never trained on, scored for correctness and tone by an LLM judge):

| Epoch | Train loss | Task eval (held-out) | General probe | Verdict |
| --- | --- | --- | --- | --- |
| 1 | 0.71 | 71% | 60% | Healthy: task up, general nearly intact |
| 2 | 0.42 | **78%** | 58% | **Best:** task peaks, small general cost |
| 3 | 0.28 | 77% | 55% | Plateau: task flat, general falling |
| 4 | 0.14 | 73% | 49% | Overfitting: task *down*, general worse |
| 8 | 0.03 | 64% | 41% | Memorized: parrots train set, broadly worse |

Notice three things. First, the train loss kept improving all the way to epoch 8 — it never told you to stop. Second, the held-out *task* eval peaked at epoch 2 and then *declined*; past that point the model was memorizing training tickets, not getting better at unseen ones. Third, the general probe fell monotonically and steeply. The team shipped epoch 10. They should have shipped epoch 2. Everything they lost — arithmetic, instruction-following, the broad competence of the base model — was visible in two cheap held-out evals they were not running.

The diagnostic discipline here is to compute eval on a held-out set every epoch (or every few hundred steps) and to *act on the held-out task eval, not the train loss*. In `transformers`/`trl` this is a few lines:

```python
from trl import SFTConfig, SFTTrainer

cfg = SFTConfig(
    output_dir="out",
    num_train_epochs=3,            # ceiling, not a target; we stop early on eval
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    # --- the part most tutorials omit ---
    eval_strategy="steps",
    eval_steps=100,                # measure held-out loss often
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,   # keep the best checkpoint, not the last
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,          # held-out task examples, NEVER trained on
)
trainer.train()
```

`load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` is the single most underused finetuning safety net. It means that even if you accidentally set `num_train_epochs` too high, the trainer keeps the checkpoint with the lowest held-out loss instead of the final, over-finetuned one. It does not protect against forgetting (held-out *task* loss can keep improving while *general* ability falls), but it kills the simplest form of overfitting for free.

A caution: held-out task loss is necessary but not sufficient. In the table above, the held-out task eval peaked at epoch 2, which is great — but the general probe had *already* dropped 4 points by then. Task eval catches memorization. It does not catch forgetting. For that you need a different instrument, which is the heart of the next section.

## 4. Catastrophic forgetting: the regression train loss can't see

Catastrophic forgetting is the phenomenon where a neural network, trained on a new task, abruptly loses its ability to perform tasks it previously could. The term goes back to McCloskey and Cohen in 1989, and it is not specific to LLMs — but LLMs make it expensive and easy to trigger, because the whole point of a base model is its broad competence, and narrow finetuning data pulls the weights toward a narrow distribution.

Here is the mechanism, made precise. Pretraining places the weights $\theta_0$ at a point that is low-loss for the broad pretraining distribution $D_\text{pre}$. Finetuning minimizes loss on a narrow distribution $D_\text{ft}$ starting from $\theta_0$. Gradient descent on $D_\text{ft}$ has no term that keeps the model good on $D_\text{pre}$ — the gradient $\nabla_\theta \mathcal{L}_{D_\text{ft}}$ points purely toward reducing task loss, and it will happily move weights that were doing important work for $D_\text{pre}$ if doing so reduces $D_\text{ft}$ loss even slightly. The further $\theta$ travels from $\theta_0$, the more the model's behavior on $D_\text{pre}$ can drift, and there is nothing in the objective to stop it.

You can see why the magnitude of forgetting scales with how *far* the weights move, which is governed by exactly the knobs we have been discussing: the learning rate (step size), the number of epochs/steps (number of steps), and whether you constrain the update at all (LoRA, replay, regularization). Large LR and many epochs maximize the distance traveled and therefore maximize forgetting. This is the deep reason the three failures in this post are not independent — they are three faces of "you moved the weights too far from the pretrained point."

We can make the "distance traveled" intuition slightly more rigorous. Approximate the pretraining loss near $\theta_0$ as a quadratic, $\mathcal{L}_\text{pre}(\theta) \approx \mathcal{L}_\text{pre}(\theta_0) + \tfrac{1}{2}(\theta - \theta_0)^\top H_\text{pre}(\theta - \theta_0)$, where $H_\text{pre}$ is the Hessian of the *pretraining* loss at the pretrained minimum (the gradient term vanishes because $\theta_0$ is a minimum). The *increase* in pretraining loss after finetuning — which is exactly the forgetting — is then $\tfrac{1}{2}\,\Delta\theta^\top H_\text{pre}\,\Delta\theta$ where $\Delta\theta = \theta_\text{ft} - \theta_0$ is the total displacement the finetune produced. Two things fall out of this. First, forgetting grows with the *squared* magnitude of the weight displacement, so halving how far you move the weights quarters the forgetting in this approximation — which is why lowering the LR is such a high-leverage fix. Second, forgetting is worst when $\Delta\theta$ points along the sharp directions of $H_\text{pre}$ (the directions the pretrained model cares about most), and this is exactly what Elastic Weight Consolidation exploits: it penalizes movement in proportion to each weight's importance to old tasks, estimated by the Fisher information (a cheap proxy for $H_\text{pre}$). You rarely need EWC for LLM finetuning, but the formula tells you why the simple fixes work: every one of them shrinks $\Delta\theta$ or steers it away from the sharp directions.

Figure 4 shows what this looks like measured, and why it is invisible without the right instrument. On the left, a narrow finetune at `1e-4` for 5 epochs: it *gains* 27 points on the target task, which looks like a win, while GSM8K math drops 18 points and MMLU drops 11. On the right, the same task at `2e-5` for 2 epochs with 20% general-data replay: it gains 22 points on task — barely less — while both general benchmarks hold within a point. The right-hand recipe is strictly better, and you would never know it from train loss, which looks similar in both runs.

![Before and after comparison showing a narrow finetune gaining task accuracy while general benchmarks collapse, versus a careful recipe that preserves both](/imgs/blogs/finetuning-an-llm-without-breaking-it-4.png)

### How to actually see forgetting: the probe harness

The core diagnostic for forgetting is a **frozen general-capability probe** that you score *before* finetuning, *during* it at each checkpoint, and *after*. The probe is a small, fixed set of general questions that have nothing to do with your finetuning task — arithmetic, common-sense reasoning, factual recall, instruction-following — and you watch its score as the run progresses. A 200-question probe is enough to detect a real regression; you are not publishing a benchmark, you are building a smoke detector.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A fixed, frozen probe: general questions UNRELATED to your finetuning task.
# Each item is (prompt, exact_answer). Keep it small (~200) and NEVER change it
# mid-project, or your before/after numbers stop being comparable.
PROBE = [
    ("What is 47 * 13? Answer with just the number.", "611"),
    ("What is the capital of Australia? One word.", "Canberra"),
    ("Sort these ascending: 5, 2, 9, 1. Comma-separated.", "1, 2, 5, 9"),
    # ... ~200 items spanning math, facts, instruction-following ...
]

@torch.no_grad()
def probe_score(model, tok, probe=PROBE, max_new=16):
    model.eval()
    correct = 0
    for prompt, gold in probe:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=False,  # greedy: deterministic
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        # Loose match: gold appears in the model's answer.
        correct += int(gold.lower() in text.lower())
    return correct / len(probe)
```

Wire it into the trainer with a callback so it runs at every save, and you get a forgetting curve for free:

```python
from transformers import TrainerCallback

class ForgettingProbe(TrainerCallback):
    def __init__(self, tok, baseline, threshold_pts=4.0):
        self.tok = tok
        self.baseline = baseline          # base-model probe score, measured ONCE up front
        self.threshold = threshold_pts / 100.0
        self.history = []

    def on_save(self, args, state, control, model=None, **kw):
        score = probe_score(model, self.tok)
        drop = self.baseline - score
        self.history.append((state.global_step, score, drop))
        print(f"[probe] step {state.global_step}: {score:.1%} "
              f"(baseline {self.baseline:.1%}, drop {drop*100:+.1f} pts)")
        if drop > self.threshold:
            print(f"[probe] forgetting exceeds {self.threshold*100:.0f} pts "
                  f"-> consider stopping / lowering LR / adding replay")
            control.should_training_stop = True   # early-stop on forgetting
        return control
```

This callback is the missing instrument. It does for forgetting what `eval_loss` does for overfitting: it converts a silent, invisible regression into a number, logs it every checkpoint, and can halt the run automatically when the general probe falls past a threshold you set (4 points is a reasonable line in the sand). Figure 7 sketches the harness: one frozen probe, scored at the base model and at every checkpoint, with an early-stop trigger.

![A graph of the forgetting probe harness with a frozen probe set scored at the base model and every checkpoint feeding an early-stop trigger](/imgs/blogs/finetuning-an-llm-without-breaking-it-7.png)

Measure the baseline *once*, before any training, and never change the probe set mid-project — if you edit the questions, your before/after numbers stop being comparable and the whole point evaporates.

### Mitigations, in order of how often they fix it

When the probe shows forgetting, you have a ranked menu of fixes, and the order matters because the cheap ones fix most cases:

1. **Lower the learning rate.** Smaller steps move the weights less far from $\theta_0$, so less forgetting. Going from `1e-4` to `2e-5` on full SFT often cuts the general-probe drop by more than half on its own.
2. **Fewer steps/epochs.** Same logic — fewer steps means less distance traveled. Cap at 1–3 epochs and stop early on the probe.
3. **Use LoRA instead of full finetuning.** LoRA constrains the update to a low-rank correction and freezes the base weights, which structurally limits how far the *effective* model can move. Empirically LoRA forgets noticeably less than full SFT at matched task performance for narrow tasks, precisely because most of the pretrained weights are untouched.
4. **Replay / data mixing.** Mix general-purpose data back into the finetuning set — typically 10–30% of the batch drawn from a broad instruction set or a slice of the pretraining distribution. This adds a term to the gradient that explicitly keeps the model good on the broad distribution, directly countering the mechanism above. It is the most reliable heavy-hammer fix when the task is genuinely narrow.
5. **Smaller effective updates / regularization.** Techniques like a KL penalty to the base model, or simply lower weight-decay-free small-norm updates, keep $\theta$ near $\theta_0$. Preference-tuning methods like DPO build a reference-model KL term in for exactly this reason.

The replay mechanism deserves a closer look because it is the fix people skip and then wish they hadn't. The forgetting gradient $\nabla_\theta \mathcal{L}_{D_\text{ft}}$ has no incentive to preserve $D_\text{pre}$ behavior. Replay fixes this directly: by mixing in general examples, the *batch* gradient becomes a weighted sum $\nabla_\theta[(1-\alpha)\mathcal{L}_{D_\text{ft}} + \alpha\mathcal{L}_{D_\text{gen}}]$, and the second term actively pulls the weights back toward configurations that keep general loss low. It is the cheapest way to convert "I hope it doesn't forget" into "the objective explicitly says don't forget." Mechanically it is a few lines — interleave a general instruction set with your task data at the desired ratio:

```python
from datasets import load_dataset, concatenate_datasets

task_ds = load_dataset("your/task-sft", split="train")              # narrow task data
gen_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Keep general data to ~20% of the mixed set: replay against forgetting.
n_gen = int(0.20 / 0.80 * len(task_ds))      # 20% of the final mix
gen_sample = gen_ds.shuffle(seed=0).select(range(min(n_gen, len(gen_ds))))

mixed = concatenate_datasets([task_ds, gen_sample]).shuffle(seed=0)
print(f"task {len(task_ds)} + replay {len(gen_sample)} = {len(mixed)} "
      f"({len(gen_sample)/len(mixed):.0%} general)")
```

The right ratio is task-dependent — more replay protects general ability more but dilutes task learning — and 10–30% is the band that works for most narrow finetunes. The probe harness tells you whether you got it right: if the probe still drops, raise the replay fraction; if the task is undertrained, lower it. You are tuning one knob against two instruments, which is a well-posed problem precisely because you are *measuring* both sides.

#### Worked example: replay turns a -18 into a -1

Take the narrow finetune from Figure 4 — a domain SFT that gained 27 task points but cost 18 points on GSM8K. Apply mitigations one at a time and re-measure the probe:

- Baseline narrow run (`lr 1e-4`, 5 epochs, 0% replay): task +27, GSM8K −18, MMLU −11.
- Drop LR to `2e-5`: task +25, GSM8K −9, MMLU −6. (Halved the forgetting, barely touched task.)
- Cap at 2 epochs: task +23, GSM8K −5, MMLU −4.
- Add 20% general-data replay: task +22, GSM8K **−1**, MMLU **−1**.

The final recipe sacrifices 5 points of task gain to recover 17 points of math and 10 points of general ability. For almost every production use case that is the right trade — a model that is excellent at one task and broken at everything else is rarely what you wanted. And critically, you could only see this trade because you were running the probe; the train loss for all four runs looked nearly identical.

## 5. Packing vs padding, and the bugs each one hides

The next layer of the stack is how you batch your sequences, and it is a source of both wasted compute and subtle correctness bugs. Two strategies dominate.

**Padding** is the simple one: take a batch of sequences of different lengths, pad them all to the length of the longest (or to a fixed `max_length`) with a pad token, and mask the pad positions out of the loss and the attention. It is correct and simple, but it wastes compute — every pad token consumes a forward and backward pass and contributes nothing. If your sequences vary a lot in length, a padded batch can be more than half pad tokens, meaning more than half your GPU's flops are doing literally nothing.

**Packing** is the efficient one: concatenate multiple short sequences into one long sequence of fixed length, so every position holds a real token. This can lift useful-token throughput from ~55% to ~95% and speed up a finetune by 1.5–2x. But packing introduces a correctness trap that bites people constantly: by default, a packed sequence is one long attention context, so token 1 of document B can attend to the end of document A. The model trains on cross-document attention that will never exist at inference time, which is a subtle distribution skew. Figure 6 contrasts the two and names the fix.

![Before and after comparison of padding wasting compute on pad tokens versus packing filling every position but requiring a block-diagonal mask to prevent cross-document attention bleed](/imgs/blogs/finetuning-an-llm-without-breaking-it-6.png)

The fix is a **block-diagonal attention mask** (also called document masking or "FlashAttention with `cu_seqlens`"), which restricts attention so each packed document only attends within itself. Modern `trl` supports this directly:

```python
from trl import SFTConfig

cfg = SFTConfig(
    output_dir="out",
    packing=True,                  # pack short sequences -> ~95% useful tokens
    max_length=2048,               # the packed sequence length
    # Critical: prevent cross-document attention within a packed sequence.
    # In recent trl this is on by default with packing; verify your version.
    padding_free=True,             # uses cu_seqlens / block-diagonal attention
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    bf16=True,
)
```

There is a tell-tale signature when packing leaks: the early training loss is *suspiciously low*. Because the model can peek across document boundaries, it has extra context that makes next-token prediction easier than it will be at inference, so the loss starts lower than a padded run on the same data. If you switch from padding to packing and your loss drops by a noticeable margin with no other change, suspect cross-document bleed before you celebrate. The diagnostic is to run a tiny experiment: train 50 steps padded and 50 steps packed-with-masking on identical data, and confirm the loss curves overlap. If packed-without-masking sits below both, you have found the leak. This family of masking bugs overlaps heavily with [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug), which covers the related trap of training on the prompt tokens.

One more packing subtlety: with packing, "epochs" become fuzzy. If you pack 4,000 examples into 1,000 packed sequences, one epoch over the packed data still sees every token once, but the *boundaries* shift each epoch depending on how examples were concatenated. This is fine, but it means your step count, not your epoch count, is the thing to reason about for the overfitting analysis in §3 — convert to steps and think in steps.

There is a second, sneakier packing bug that interacts with loss masking, and it is worth naming because it silently wastes the run rather than crashing it. When you pack instruction examples, each packed sequence contains several `(prompt, completion)` pairs, and you almost always want to mask the *prompt* tokens out of the loss so the model trains to *produce* completions, not to *predict prompts*. If your packing logic concatenates raw text and then applies a single completion mask, the prompt-masking boundaries can land in the wrong places after concatenation — you end up training on some prompt tokens and masking some completion tokens. The signature is subtle: the loss is plausible, the model trains, but it learns the task slightly worse than it should and sometimes picks up prompt-like phrasings in its outputs. The fix is to compute the loss mask *per example before packing* and carry the per-token mask through the concatenation, never to re-derive it after packing. This is squarely the territory of [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug), which is why packing and masking must be debugged together — they share the same token-boundary bookkeeping, and a bug in one looks like a bug in the other.

#### Worked example: a packing speedup that was a leak

A team switched their 7B finetune from padding to packing to use their GPUs better and reported great news: "packing made training 1.8x faster *and* the loss is lower — it's strictly better." The throughput claim was real; the loss claim was the tell. Padded loss had plateaued around 0.62; packed loss sat at 0.48 with otherwise identical data and hyperparameters. A 0.14 drop in loss from a *batching change* is not free lunch — batching does not make next-token prediction genuinely easier — so it had to be information leaking. The confirming test was exactly the one above: 50 steps padded (loss settles ~0.62), 50 steps packed-with-block-diagonal-masking (loss settles ~0.61, overlapping the padded run), 50 steps packed-without-masking (loss settles ~0.48, well below). The unmasked packed run was letting each document attend to the previous document's tokens, giving the model context it would never have at inference, which made training-time next-token prediction artificially easy. They turned on `padding_free` (the block-diagonal mask), the loss rose to 0.61 where it belonged, kept the 1.8x speedup, and the model's held-out task eval improved by 3 points because it was no longer training on a distribution that does not exist at serving time. The general rule: **a batching or efficiency change that lowers your loss is a bug until proven otherwise.** Efficiency changes the *cost* of computing the loss, never its *value*.

## 6. Warmup, schedule, batch size, and the effective-LR coupling

Three more knobs round out the recipe, and they interact in ways that produce confusing bugs.

**Warmup.** A pretrained model and a freshly-initialized Adam optimizer are a bad first date. At step 0, Adam's second-moment estimate is uninitialized, so its adaptive step sizes are unreliable, and the very first updates can be erratically large — exactly when you least want a large step into a sharp basin. Warmup ramps the learning rate from 0 to its target over the first few percent of steps, giving Adam time to build stable moment estimates before you take full-size steps. For finetuning, 3–5% warmup (`warmup_ratio=0.03`) is standard and cheap insurance against an early spike. Skipping warmup on a finetune is a common cause of a loss bump in the first 20 steps that then recovers — annoying, occasionally fatal, and trivially avoided.

**Schedule.** After warmup, decay the LR — cosine or linear decay to zero over the run. The reason matters: as you approach a minimum, you want smaller steps to settle in rather than bounce around it. Cosine decay is the default for good reason; a constant LR for a finetune leaves you taking full-size steps right up to the end, which both wastes the final epochs and slightly increases forgetting (you keep moving the weights when you should be settling). The schedule is a minor knob compared to the LR magnitude, but a constant-LR finetune is a small unforced error.

**Batch size and gradient accumulation — and the coupling.** This is the one that produces "I copied the config exactly and got different results" bugs. The effective batch size is

$$B_\text{eff} = B_\text{per-device} \times N_\text{grad-accum} \times N_\text{GPUs}$$

and the optimizer sees one update per *effective* batch, not per micro-batch. The learning rate is tuned for a particular effective batch size, because a larger batch produces a lower-variance gradient estimate, which changes the optimal step size. The classic coupling is the **square-root scaling rule** for Adam: if you multiply the effective batch size by $k$, you should scale the LR by roughly $\sqrt{k}$ to keep the per-step update statistics comparable. (Linear scaling, $k\times$, is the rule for SGD with momentum on vision; Adam-family optimizers on language models track closer to square-root in practice.) The full LR–batch interaction is its own deep topic — see [the learning rate post](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) — but the bug to avoid is concrete.

#### Worked example: the silent batch-size change

A colleague hands you a config that worked: `per_device_train_batch_size=4`, `gradient_accumulation_steps=8`, on 1 GPU — effective batch 32, `learning_rate=2e-5`. You have 4 GPUs and want it faster, so you keep the per-device and accum settings and just run on 4 GPUs. Now $B_\text{eff} = 4 \times 8 \times 4 = 128$, a 4x increase, but the LR is unchanged. By the square-root rule the LR is now ~2x too *low* for the new batch, so the model undertrains — the loss falls more slowly, the task is learned less well, and you blame the data or the model. The fix is either to scale the LR up by $\sqrt{4} = 2$ to `4e-5`, or to drop `gradient_accumulation_steps` to 2 to restore the effective batch of 32. The lesson: **a config is only reproducible if the effective batch size is held constant**, and any change to GPU count or accumulation steps silently changes it. We go deep on this in [gradient accumulation and effective-batch bugs](/blog/machine-learning/debugging-training/the-training-debugging-playbook) territory; for now, always log $B_\text{eff}$ and the LR together so the coupling is visible.

A related trap: **weight decay on the wrong parameters.** AdamW's weight decay pulls weights toward zero, which is sensible regularization for the main weight matrices but actively harmful applied to LayerNorm gains, biases, and embeddings — decaying a LayerNorm scale toward zero degrades the normalization the model relies on. The standard fix is to exclude `bias` and normalization parameters from weight decay via parameter groups. Most well-built trainers do this, but a hand-rolled optimizer often does not, and the symptom is a finetune that mysteriously underperforms with no spike — a quiet, distributed degradation rather than a dramatic failure.

```python
# Exclude norms/biases from weight decay (do NOT decay LayerNorm gains or biases).
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim < 2 or "norm" in name.lower() or name.endswith(".bias"):
        no_decay.append(p)      # 1-D params: norms, biases -> no decay
    else:
        decay.append(p)         # 2-D weight matrices -> decay
optimizer = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=2e-5,
    betas=(0.9, 0.999),
)
```

Finally, **gradient clipping.** Clip the global gradient norm (`max_grad_norm=1.0` is the common default) so that a single bad batch — a malformed example, a numerical blip — cannot produce one enormous step that knocks the model out of its basin. Clipping does not fix a wrong LR (if every step is too big, clipping just makes every step uniformly slightly less too-big), but it is excellent insurance against the *occasional* outlier batch that would otherwise cause a one-off spike. It functions as a circuit breaker, not a regulator. The diagnostic value of clipping is also informational: if you log the *pre-clip* gradient norm and it is regularly hitting the clip threshold, that is a signal your LR is too high or your data has outliers — clipping is masking a problem you should look at, not just absorbing noise. A healthy finetune clips rarely; a finetune that clips on most steps is one bad batch away from a spike.

## 7. Eval that catches regression early, not after you ship

Every failure in this post has the same root cause from the *debugging* perspective: you were watching the wrong number. Train loss is the wrong number. So before the diagnostic matrix, it is worth being explicit about the eval discipline that catches all three failures, because it is the cheapest insurance in the entire finetuning workflow and the most commonly skipped.

There are three distinct things you should measure, and they catch different bugs:

1. **Held-out task loss / task eval** — examples drawn from the same distribution as your training data but never trained on. This catches *memorization*: if held-out task performance peaks and then declines while train loss keeps falling, you are overfitting. This is the `eval_loss` in the `SFTConfig`, and `load_best_model_at_end` acts on it automatically.
2. **A frozen general-capability probe** — questions with nothing to do with your task. This catches *forgetting*: the base model scores high, and you watch the score as the run progresses. Nothing else catches forgetting, because train and even held-out *task* loss both look fine while general ability drops.
3. **A generation smoke test** — actually generate from the model on a handful of representative inputs at each checkpoint and read the output. This catches the failures that no scalar metric shows: repetition loops, the model that won't stop (never emits EOS), format collapse, and the degeneration into token soup that a wrong LR produces. A loss number cannot tell you the model has started repeating "I'd be happy to help" forever; reading three generations can.

The reason all three are necessary is that they form a *coverage* of the failure modes. Task loss covers memorization. The probe covers forgetting. The generation test covers degeneration and the qualitative failures that live below the resolution of a loss number. Run only train loss and you are blind to all three. Run all three at every checkpoint and there is essentially no recipe bug from this post that can reach production without tripping at least one of them.

#### Worked example: the eval that would have saved the support bot

Return one last time to the eight-epoch support bot from the intro. Suppose the team had run all three evals every epoch. At epoch 2, task eval peaks at 78% — a clear signal to consider stopping. At epoch 4, task eval has fallen to 73% and the general probe is down 13 points from base — two independent alarms. At epoch 8, the generation smoke test shows the model emitting verbatim training responses on novel tickets — the memorization made visible in text. *Any one* of these three instruments, checked at *any* epoch past 2, would have stopped the team from shipping epoch 10. They shipped it because the only number they watched, train loss, was monotonically improving the whole way down to 0.03. The bug was not in the data or the model. The bug was the eval. The cost of the missing eval was a production incident; the cost of running it would have been a few minutes of GPU time per epoch.

This is the deepest lesson of finetuning debugging, and it generalizes past this post: **the instrument you are not looking at is where the bug is hiding.** Train loss is necessary but radically insufficient for finetuning, because the whole premise of finetuning is that the model already knows things your training data does not measure. You have to measure those things separately, or you will trade them away without noticing. The [taxonomy post](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) frames this as the "evaluation" place in the six-places map — and for finetuning specifically, evaluation is where half the bugs hide.

## 8. A diagnostic matrix and a worked bisection

Let us assemble the recipe knobs, their failure signatures, and their fixes into one diagnostic table you can keep next to the keyboard. Figure 5 renders the core of it; the table below adds the instruments.

![A matrix mapping each finetuning recipe knob to its wrong setting, its failure signature, and its fix](/imgs/blogs/finetuning-an-llm-without-breaking-it-5.png)

| Knob | Wrong setting | Signature | Instrument that catches it | Fix |
| --- | --- | --- | --- | --- |
| Learning rate | `1e-3` (from-scratch) | Loss spike in first 50 steps, then plateau or NaN; garbage generations | Loss curve; LR sweep on a subset | Drop to `2e-5` (full) / `2e-4` (LoRA) |
| Epochs | 8 epochs | Train loss → ~0; held-out *task* eval peaks early then falls; model parrots train data | Held-out task eval per epoch | Cap at 1–3 epochs; `load_best_model_at_end` |
| Forgetting | narrow data, big LR, many steps | Task improves, general ability silently drops; train loss looks fine | Frozen general probe at each checkpoint | Lower LR; fewer steps; LoRA; 10–30% replay |
| Packing | no document mask | Early loss suspiciously low; train/infer skew | Compare packed-vs-padded 50-step loss | Block-diagonal / `padding_free` mask |
| Warmup | none | Loss bump in first ~20 steps | Loss curve early region | `warmup_ratio=0.03` |
| Effective batch | changed GPU/accum, fixed LR | Reproducibility failure; under/over-training | Log $B_\text{eff}$ with LR | Re-scale LR by $\sqrt{k}$ or hold $B_\text{eff}$ |
| Weight decay | decay on norms/biases | Quiet underperformance, no spike | Inspect optimizer param groups | Exclude norms/biases from decay |
| Gradient clip | none | Rare one-off spike from a bad batch | Grad-norm log (occasional outlier) | `max_grad_norm=1.0` |

Now the part the table cannot give you: the *order* in which you check these. When a finetune misbehaves, you do not change eight things at once — you bisect, reading the loss curve shape first because it splits the suspect space cleanly. Figure 8 is the decision tree.

![A decision tree that routes a broken finetune by loss-curve shape to learning rate, epochs, or forgetting, each with a confirming test and fix](/imgs/blogs/finetuning-an-llm-without-breaking-it-8.png)

#### Worked example: bisecting a finetune that "didn't work"

Here is a real-shaped scenario. You finetune Llama-3.1-8B-Instruct on 6,000 examples to make it answer in a specific structured JSON format. After training, the model produces the JSON format well on training-like inputs but has become noticeably worse at general conversation and occasionally emits malformed JSON on unusual inputs. The team's first instinct is "the data is bad." Resist it. Bisect.

**Step 1 — read the loss curve.** It is smooth, no spikes, falling from 1.3 to 0.09 over 6 epochs. Smooth-falling-to-near-zero immediately rules *out* the learning-rate failure (no spike) and rules *in* the epochs/forgetting branch. We have bisected from six suspects to two without touching code.

**Step 2 — check the epochs.** Loss 0.09 after 6 epochs is the memorization range. Pull the held-out *task* eval per epoch from the saved checkpoints: it peaked at epoch 2 (94% valid JSON on unseen inputs) and *fell* to 88% by epoch 6. Confirmed overfitting — the model memorized training JSON and generalizes worse. Suspect #1 confirmed.

**Step 3 — check forgetting.** Run the frozen general probe on the base model (61%) and on the epoch-6 checkpoint (44%). A 17-point drop. Confirmed forgetting. Suspect #2 confirmed. Both failures are present, and both trace to the same root: too many epochs at too high an LR moved the weights too far.

**Step 4 — confirm the root with one experiment.** Re-run at `lr 2e-5` (it was `1e-4`), capped at 2 epochs, with `load_best_model_at_end` and the probe callback. Result: JSON validity 95% on held-out inputs, general probe 60% (−1 from base). The structured task is *better* than the over-finetuned run, and the general ability is preserved. The data was never the problem; the recipe was. Total debugging time once you bisect instead of guess: under an hour, most of it GPU time.

This is the whole method in miniature: the curve shape eliminates four of six suspects, two cheap held-out evals confirm the remaining two, and one controlled re-run proves the fix. You never edited the dataset, never changed the model code, never touched the data pipeline. The bug was in optimization and evaluation, exactly where finetuning concentrates its bugs.

## 9. A complete, defensible finetuning recipe

Putting it all together, here is a full `SFTConfig` for a 7–8B full-parameter finetune that bakes in every safeguard from this post. It is not magic — it is the boring, correct defaults, which is exactly what you want.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
)

cfg = SFTConfig(
    output_dir="ft-out",
    # --- the two knobs that matter most ---
    learning_rate=2e-5,              # full SFT: 50-100x below a from-scratch rate
    num_train_epochs=3,              # ceiling; we stop early on eval/probe
    # --- effective batch (log this with the LR) ---
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # B_eff = 4 * 4 * N_GPUs; re-scale LR if changed
    # --- schedule ---
    warmup_ratio=0.03,               # ramp Adam in; avoids the early bump
    lr_scheduler_type="cosine",      # decay to settle into the minimum
    max_grad_norm=1.0,               # circuit breaker for outlier batches
    weight_decay=0.01,               # trl excludes norms/biases by default
    # --- efficiency ---
    packing=True,                    # ~95% useful tokens
    padding_free=True,               # block-diagonal mask: no cross-doc bleed
    max_length=2048,
    bf16=True,                       # bigger range than fp16; no loss scaling needed
    # --- the safety nets most configs omit ---
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    report_to="wandb",               # watch loss, grad_norm, lr live
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    callbacks=[ForgettingProbe(tok, baseline=0.61, threshold_pts=4.0)],
)
trainer.train()
```

Three lines do most of the protective work: `learning_rate=2e-5` keeps you in the basin, `load_best_model_at_end` discards over-finetuned checkpoints, and the `ForgettingProbe` callback halts the run if general ability falls too far. Note the choice of `bf16` over `fp16`: bf16 has the same exponent range as fp32 (so gradients do not underflow and you need no loss scaling), trading mantissa bits for range, which is the right trade for training. If you are stuck on fp16 hardware, you inherit a whole class of numerical bugs covered in [mixed-precision debugging](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16). For the LoRA variant, the only changes are `learning_rate=2e-4`, a `peft` `LoraConfig` with the right `target_modules`, and the caveat that gradient checkpointing plus LoRA has its own interaction bugs — all of which live in [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

For a broader survey of finetuning *techniques* (as opposed to the *bugs* this post hunts), the [effective LLM finetuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) guide is a good companion — it covers the method space; this post covers the failure space.

#### Worked example: the recipe applied end to end

Concretely, here is what the defensible recipe buys you, measured against the naive tutorial config on the same 6,000-example structured-output task and the same Llama-3.1-8B-Instruct base. The naive config was `learning_rate=1e-4`, `num_train_epochs=6`, no warmup, padding only, no eval callbacks, take the last checkpoint. The defensible config is the one above. The instruments read:

| Metric | Naive config | Defensible config |
| --- | --- | --- |
| Loss spike in first 50 steps | yes, bump to 2.1 then recover | none, smooth from 1.3 |
| Train loss (final) | 0.09 (memorized) | 0.34 (best checkpoint, epoch 2) |
| Held-out task eval | 88% | **95%** |
| General probe (base 61%) | 44% (−17 pts) | 60% (−1 pt) |
| Throughput (tokens/sec) | 1.0x (padded) | 1.7x (packed + masked) |
| Wall-clock to best model | 6 epochs, kept the worst | 2 epochs, kept the best |

Every line improved, and none of it required better data, a bigger model, or more compute — it required the right LR, an epoch cap with `load_best_model_at_end`, a probe callback, and packing with a document mask. The naive config spent more GPU time to produce a worse, broken model. The defensible config is not slower or more expensive; it is faster *and* better, because the safeguards are mostly free and the efficiency knobs are a real speedup. This is the happy truth of finetuning debugging: doing it right is usually cheaper than doing it wrong, because doing it wrong means long runs that memorize and forget.

## 10. Case studies and real signatures

A few well-documented patterns make these failures concrete, so you recognize them in the wild.

**The InstructGPT epoch count.** OpenAI's InstructGPT work and the broader instruction-tuning literature consistently finetune for a *small* number of epochs — on the order of 1–4 — over the supervised data, and explicitly note that the SFT models overfit quickly, with more epochs helping downstream reward-model and RLHF stages more than helping the SFT model's own held-out loss. The practical takeaway that propagated through the field: instruction data overfits fast, and the right move is few epochs plus held-out monitoring, not "train until the loss is low." If a from-scratch instinct tells you "run it for 20 epochs to be safe," the instruction-tuning literature says the opposite.

**Catastrophic forgetting in continual learning.** The phenomenon is old and well-measured. McCloskey and Cohen (1989) named it; the modern continual-learning literature (Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," 2017, which introduced Elastic Weight Consolidation) quantifies it and motivates exactly the mitigations in §4 — constraining how far important weights move (EWC adds a penalty proportional to each weight's importance, estimated via the Fisher information) and replaying old data. For LLM finetuning you rarely need EWC's machinery; lower LR, fewer steps, LoRA, and replay cover the practical cases. But the lesson is the same: nothing in a plain finetuning objective protects old capabilities, so you must add protection deliberately.

**The "lost general ability" reports.** It is a recurring community finding that aggressively finetuning a strong instruct model on a narrow dataset produces a model that is great at the narrow task and visibly worse at general chat — losing arithmetic, multi-step reasoning, and sometimes its safety behavior. The fix that repeatedly works is the §4 menu: a smaller LR, fewer epochs, and mixing 10–30% general instruction data back in. These are not exotic results; they are what you get whenever you forget that the base model's broad competence is a thing you can lose. The reason they keep happening is that the train loss never warns you — which is the entire thesis of this post.

**LoRA forgets less, by construction.** Multiple studies comparing LoRA to full finetuning on narrow tasks find that LoRA retains more of the base model's general ability at comparable task performance, with the mechanistic explanation that LoRA's frozen base weights and low-rank update keep the effective model close to $\theta_0$. The trade is that LoRA can be slightly worse at *deep* task adaptation that genuinely needs to move many weights. For most narrow finetunes — style, format, domain Q&A — that trade favors LoRA, and the forgetting-preservation is a real, repeatable benefit, not a marketing claim.

**The "0.99 train, broken in production" pattern.** A pattern that recurs across modalities, not just LLMs: a finetune reaches near-perfect training metrics and fails in production, and the post-mortem finds the model memorized the training set and generalizes poorly. In the LLM case the production failure is usually one of two things — the model parrots training responses on novel inputs (memorization), or it has lost a capability the production traffic relies on (forgetting). The unifying diagnosis is that the team optimized and monitored a single in-distribution metric and never measured generalization or general ability. The cross-modal version of this story is the subject of [finetuning pitfalls across modalities](/blog/machine-learning/debugging-training/finetuning-pitfalls-across-modalities); the LLM-specific version is everything in this post. The fingerprint is always the same: a beautiful training metric and a held-out or general metric nobody looked at.

**Why bf16 became the finetuning default.** It is worth noting the field's convergence on bf16 over fp16 for training, because it removes a whole class of numerical bugs from finetuning. fp16 has a narrow exponent range and small gradients underflow to zero (the representable minimum normal value is about $6.1\times10^{-5}$), which is why fp16 training needs loss scaling and still occasionally produces NaNs. bf16 keeps fp32's exponent range — gradients do not underflow — at the cost of fewer mantissa bits, which barely matters for training because the optimizer averages over many steps. The "Mixed Precision Training" paper (Micikevicius et al., 2018) is the canonical reference for the fp16 mechanics and loss scaling; the practical takeaway for finetuning is that if your hardware supports bf16, use it and delete loss scaling from your worry list. The deeper treatment is in [mixed-precision debugging](/blog/machine-learning/debugging-training/mixed-precision-debugging-fp16-vs-bf16).

## 11. When this is (and isn't) your bug

Precision about what each symptom rules in and out is what separates a fast debugger from someone changing random hyperparameters.

**A loss that spikes in the first 50 steps is the learning rate, not the data.** Data problems do not produce a clean spike-then-plateau in the first dozen steps; they produce a loss that is too low (leakage), too high and flat (a format/masking bug), or sawtoothed (a dataloader issue). A sharp early spike that scales with the LR is numerics-and-optimization. Lower the LR before you touch the dataset.

**A smooth loss gliding to near zero is the epochs, not a triumph.** If your held-out task eval is not also improving, low train loss is memorization. The confirming test is one held-out eval; if it peaked three epochs ago, you have your answer, and no amount of cleaner data will help because the data was never the problem.

**Worse general ability with fine task ability is forgetting, not a bad base model.** If you swapped in a "smarter" base model to fix degraded general behavior and it did not help, you are treating a symptom. The base model was fine; your recipe moved its weights too far. Run the probe on the base versus your checkpoint — if the base scores high and your checkpoint scores low, the recipe is the culprit, full stop.

**If the model is fine on your task but broken at inference specifically, that is a different bug.** A model that trains and evals well but produces garbage only when you actually generate from it usually points to a train/inference mismatch — chat-template skew, left-vs-right padding for decoder-only generation, or a sampling issue — not the recipe bugs here. Those have their own diagnostics and are out of scope for this post; the tell is that your *held-out training-time* metrics look fine and only *generation* breaks.

**If overfit-one-batch fails, stop blaming the recipe.** Before any of this, the master sanity check still applies: can your finetune drive loss toward zero on a single batch? If it *can't*, your problem is upstream of LR/epochs/forgetting — a broken data pipeline, a frozen model, a masking bug that zeros out the loss. The [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) rules the whole pipeline in or out in two minutes, and you run it *first*. The recipe bugs in this post are what you debug *after* you have confirmed the model can learn at all.

**If the loss won't go down at all and there's no spike, it's not the recipe bugs here either.** A finetune where the loss sits flat from step one — no spike, no descent — points to a dead optimizer (LR so low it's effectively zero, or the trainable parameters aren't actually receiving gradients), a frozen model, or a LoRA adapter that never entered the graph. That is a "the model isn't learning what you think" problem, not an LR-too-high or epochs-too-many problem. The tell is the *direction*: this post's bugs all involve the loss going down (smoothly to memorization, or up-then-down through a spike). A loss that never moves is a different family entirely, and you debug it by confirming gradients are flowing to the parameters you meant to train.

**A model that degraded "over time in production" without a retrain is not a finetuning bug at all.** If you did not change the model and it got worse, the model's weights did not change — the *data* did. That is distribution shift, a data-and-evaluation story, not a recipe story. The recipe bugs in this post are things you cause at *training* time; a model silently degrading at serving time with frozen weights is the world moving under a fixed model, and no LR or epoch change touches it. Knowing the difference saves you from re-running a finetune to fix a problem a finetune cannot fix.

The unifying thread across all of these: name the *direction and timing* of the symptom before you name the cause. Spike early → LR. Smooth to zero → epochs. Task-fine-but-general-worse → forgetting. Flat from step one → no gradients. Worse in production with no retrain → data shift. Five symptoms, five different places in the six-places map, and the loss-curve shape plus two cheap held-out evals tell them apart in minutes.

## 12. Key takeaways

- **Full SFT wants `1e-5`–`2e-5`; LoRA wants `1e-4`–`3e-4`; from-scratch `1e-3` is a demolition charge.** A pretrained model sits in a sharp basin where the stable step size is $\eta < 2/\lambda_\max$, and a from-scratch LR walks straight past it. A loss spike in the first 50 steps is this bug.
- **Instruction data overfits in 1–3 epochs.** Train loss falling to near zero is memorization, not learning. Cap epochs, monitor a held-out *task* eval, and use `load_best_model_at_end` to keep the best checkpoint, not the last.
- **Train loss cannot see catastrophic forgetting.** Narrow finetuning erases general ability while task loss looks great. The only instrument that catches it is a frozen general probe scored before, during, and after the run.
- **The three big failures are one failure.** Wrong LR, too many epochs, and forgetting are all "you moved the weights too far from $\theta_0$." Smaller steps and fewer of them help all three at once.
- **Forgetting mitigations, in order:** lower LR, fewer steps, switch to LoRA, mix in 10–30% general replay data. The cheap ones fix most cases; replay is the heavy hammer for genuinely narrow tasks.
- **Packing buys ~1.7x throughput but needs a block-diagonal mask** or documents bleed across boundaries. A suspiciously low early loss after switching to packing is the tell.
- **The effective batch size is `per_device x grad_accum x GPUs`, and the LR is coupled to it.** Change the GPU count or accumulation steps and you silently changed the effective LR; re-scale by roughly $\sqrt{k}$ or hold $B_\text{eff}$ constant.
- **Bisect by curve shape.** Spike → LR; smooth-to-zero → epochs; task-fine-but-eval-worse → forgetting. Two cheap held-out evals confirm; one controlled re-run proves the fix. You rarely need to touch the data.

## Further reading

- Micikevicius et al., "Mixed Precision Training" (2018) — the loss-scaling and fp16/bf16 numerics behind the dtype choices in the recipe.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022) — the SFT-then-RLHF pipeline and the few-epoch instruction-tuning practice.
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017) — EWC and the formal account of why and how networks forget.
- McCloskey and Cohen, "Catastrophic Interference in Connectionist Networks" (1989) — the original naming of the phenomenon.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — the adapter mechanics behind LoRA's different LR and its reduced forgetting.
- Hugging Face `trl` `SFTTrainer` / `SFTConfig` documentation — packing, `padding_free`, and the config surface used throughout.
- [The learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) — the full loss-landscape physics of LR failures and the LR-range test.
- [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the six-places bisection map this post instantiates.
- [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the capstone decision tree and printable checklist.
- [Debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft) and [the loss-masking bug](/blog/machine-learning/debugging-training/the-loss-masking-bug) — the adapter and masking failures adjacent to this recipe.
