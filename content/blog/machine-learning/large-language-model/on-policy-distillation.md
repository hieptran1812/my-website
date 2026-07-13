---
title: "On-Policy Distillation: Teaching a Student on Its Own Trajectories"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Why distilling a small model on the tokens it actually generates — not the teacher's — fixes compounding error, and why it is really reinforcement learning with a dense, per-token reward you get for free."
tags:
  [
    "llm",
    "distillation",
    "on-policy-distillation",
    "knowledge-distillation",
    "reinforcement-learning",
    "gkd",
    "reverse-kl",
    "reasoning",
    "post-training",
    "training-techniques",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 34
---

Here is a failure mode I have watched bite three different teams. They distill a small model from a strong teacher — generate a few hundred thousand teacher completions, run supervised fine-tuning on them, and the student's offline metrics look fantastic. Token-level agreement with the teacher is 90%-plus. Then they ship it, and on the long multi-step problems that mattered — the agent traces, the chain-of-thought math, the 2,000-token tool-use sequences — the student falls apart in a way the offline numbers never predicted. It starts a derivation correctly, makes one small slip around token 300, and then *spirals*: every subsequent token is conditioned on a mistake the teacher would never have made, in a region of sequence-space the teacher never visited and the student was never taught to handle.

That spiral is not a bug in their data pipeline. It is the defining limitation of **off-policy distillation** — training the student on sequences that *someone else* (the teacher, or a fixed dataset) generated. And the fix is a deceptively small change with deep consequences: let the student generate its own rollouts, and have the teacher grade *those*. That is **on-policy distillation**.

![The on-policy distillation loop](/imgs/blogs/on-policy-distillation-1.webp)

The diagram above is the mental model for the entire article. A prompt goes to the student; the student samples its own rollout; the frozen teacher re-scores the exact tokens the student produced; the divergence between the two distributions at every visited state becomes the loss; one gradient step later, the student is slightly better — and the next rollout comes from the *updated* student. The student is never trained on a single token it would not plausibly have produced. The teacher is a grader, not a script to memorize.

This reframing turns out to be far more than a data-sourcing tweak. On-policy distillation is, almost exactly, **on-policy reinforcement learning where the reward is a dense, per-token signal supplied for free by the teacher.** That single sentence explains why it fixes the compounding-error problem, why it reaches teacher-level reasoning at a fraction of the compute of outcome-reward RL, and why it has quietly become the workhorse behind the small reasoning models you have been using. Let us earn that sentence.

## Off-policy distillation trains for a world the student never sees

Start with the standard recipe, because it is what almost everyone reaches for first and it is genuinely useful — just not for what people assume. Classic [knowledge distillation](/blog/machine-learning/large-language-model/distillation-in-llm) takes a fixed corpus of teacher outputs and trains the student to reproduce them: either the teacher's hard tokens (sequence-level KD) or its full soft distribution (logit distillation). DeepSeek-R1-Distill, Alpaca, Vicuna — the whole first generation of "distilled" open models — are off-policy in this sense. The student learns from a frozen pile of teacher text.

The assumption baked into this approach is that *matching the teacher on the teacher's data implies matching the teacher in deployment*. It does not, and the gap is not subtle.

| What people assume | The naive mental model | What actually happens |
| --- | --- | --- |
| High token-agreement on teacher data ⇒ a good student | The student "learns the teacher's policy" | It learns the teacher's policy **only on states the teacher visits** |
| The student will behave like the teacher at inference | Training and inference see the same distribution | At inference the student samples from **its own** distribution, which drifts |
| One mistake is a local error | Errors are independent across tokens | One mistake shifts every later state off-distribution — errors **compound** |
| More teacher data fixes it | Coverage scales with dataset size | The teacher never demonstrates **recovery from the student's own mistakes** |
| A smaller student just needs more epochs | Capacity is a throughput knob | A small student **cannot represent every mode** of a big teacher, so forced mass-covering blurs it |

The root cause is the *train-inference distribution mismatch*, and it has a name in the imitation-learning literature where it was diagnosed two decades ago: **exposure bias**, or more pointedly, the failure mode of **behavioral cloning**. You can teach a self-driving policy to imitate an expert by recording the expert driving and doing supervised learning on the (state, action) pairs. It works right up until the car drifts slightly toward the shoulder — a state the careful expert never entered, so the policy was never shown how to steer back, so it drifts further, and now you are in a ditch. Off-policy distillation is behavioral cloning for token sequences. The student is a flawless mimic on the demonstration set and a stranger to its own mistakes.

> Off-policy distillation teaches the student to ace an exam it will never sit. On-policy distillation makes it practice the exam it will actually take — and corrects it on the answers it actually writes.

## 1. Why the off-policy wall is a wall, not a speed bump

Let me make the compounding precise, because "errors compound" is the kind of phrase that sounds true and explains nothing.

Model generation as a walk through a tree of states. At step $t$ the student is in state $s_t = (x, y_{\lt t})$ — the prompt plus the tokens it has emitted so far — and samples the next token from its policy $q_\theta(\cdot \mid s_t)$. Off-policy training only ever places the student in states drawn from the *teacher's* trajectory distribution. Call the per-step probability that the student deviates from a teacher-plausible continuation $\epsilon$. On the teacher's own states, supervised training drives $\epsilon$ low — say 1%. But the moment the student takes one off-teacher action, it enters a state with **no training signal at all**, because the teacher's data never covered it. There, the deviation rate is not $\epsilon$; it is whatever the untrained tail of the distribution does, which is much worse.

The classic DAgger analysis makes the scaling explicit: a behavioral-cloning policy with per-step error $\epsilon$ on the demonstration distribution accumulates total error that grows like $\epsilon T^2$ over a horizon of $T$ steps, versus $\epsilon T$ for a policy trained on its own induced state distribution. That extra factor of $T$ is the wall. For a 2,000-token reasoning trace, a quadratic penalty is not a speed bump you tune away with more epochs — it is a structural ceiling. The animation below is that quadratic in motion.

<figure class="blog-anim">
<svg viewBox="0 0 760 400" role="img" aria-label="Two students roll out the same prompt: the on-policy student stays inside the teacher-supervised band while the off-policy student drifts below it as errors compound" style="width:100%;height:auto;max-width:880px">
<style>
.c2-band{fill:var(--surface,#eef2ff);stroke:var(--border,#c7d2fe);stroke-width:1.5}
.c2-on{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
.c2-off{fill:none;stroke:#e5484d;stroke-width:3;stroke-dasharray:8 7}
.c2-start{fill:var(--text-primary,#1f2937)}
.c2-ondot{fill:var(--accent,#6366f1);offset-path:path('M70,170 C 200,150 320,185 440,165 C 560,150 660,178 700,170');offset-distance:100%}
.c2-offdot{fill:#e5484d;offset-path:path('M70,170 C 210,200 310,250 410,290 C 520,330 630,355 700,360');offset-distance:100%}
.c2-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.c2-sub{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.c2-red{font:600 15px ui-sans-serif,system-ui;fill:#e5484d}
@keyframes c2-run{0%{offset-distance:0%;opacity:0}7%{opacity:1}93%{opacity:1}100%{offset-distance:100%;opacity:0}}
.c2-ondot,.c2-offdot{animation:c2-run 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.c2-ondot,.c2-offdot{animation:none;opacity:1}}
</style>
<rect class="c2-band" x="60" y="130" width="650" height="80" rx="12"/>
<text class="c2-sub" x="74" y="120">states the teacher actually supervises</text>
<path class="c2-on" d="M70,170 C 200,150 320,185 440,165 C 560,150 660,178 700,170"/>
<path class="c2-off" d="M70,170 C 210,200 310,250 410,290 C 520,330 630,355 700,360"/>
<circle class="c2-start" cx="70" cy="170" r="8"/>
<text class="c2-lbl" x="40" y="250">shared start</text>
<circle class="c2-ondot" cx="0" cy="0" r="10"/>
<circle class="c2-offdot" cx="0" cy="0" r="10"/>
<text class="c2-lbl" x="470" y="150">on-policy: corrected at every state</text>
<text class="c2-red" x="360" y="388">off-policy: error compounds where the teacher gave no signal</text>
</svg>
<figcaption>The same prompt, rolled out twice. The on-policy student is graded on the states it actually visits, so it stays in the band; the off-policy student, trained only on the teacher's trajectory, drifts into states it was never taught to recover from.</figcaption>
</figure>

There is a second, subtler reason the wall is real, and it has nothing to do with horizon. Even with infinite teacher data, **a small student cannot represent everything a large teacher can.** A 3B model does not have the capacity to match a 32B model's full conditional distribution over a 150,000-token vocabulary at every position. When you force it to (which, as we will see, is exactly what off-policy logit distillation does), it spreads its limited probability mass to *cover* all the teacher's options — including ones it cannot pull off — and the result is a hedged, blurry, lower-confidence model. The choice of *which divergence* you minimize, and *over whose distribution*, decides whether the student hedges or commits. That is the next section.

## 2. The two directions of KL divergence

Distillation minimizes a divergence between the teacher distribution $p_T$ and the student distribution $q_\theta$. The Kullback–Leibler divergence is not symmetric, and the asymmetry is the whole story. Define, at a single state with teacher distribution $p$ and student distribution $q$ over the vocabulary:

$$
\mathrm{KL}(p \,\|\, q) = \sum_{a} p(a)\,\log\frac{p(a)}{q(a)}, \qquad
\mathrm{KL}(q \,\|\, p) = \sum_{a} q(a)\,\log\frac{q(a)}{p(a)}.
$$

The left one is **forward KL** (teacher-first). The right one is **reverse KL** (student-first). They penalize different mistakes:

- **Forward KL** is *mass-covering* (also called mean-seeking). The expectation is taken over the teacher's distribution $p$, so wherever the teacher puts mass, the student is heavily penalized for putting near-zero probability. The student is forced to "cover" every mode the teacher has — even modes it lacks the capacity to model well. With a bimodal teacher and a unimodal student, forward KL smears the student across both humps and, fatally, deposits probability in the *low-probability valley between them* — tokens the teacher would essentially never emit. That valley-filling is a direct mechanism for hallucination and incoherence.
- **Reverse KL** is *mode-seeking*. The expectation is over the *student's* distribution $q$. The student is only penalized where *it* puts mass, so it is free to ignore teacher modes it cannot represent — and it is heavily penalized for putting mass where the teacher does not. The result: the student picks one mode the teacher rates highly, and commits to it sharply. Sharp and coherent beats blurry and hedged for a capacity-limited model.

![Forward vs reverse KL: mass-covering vs mode-seeking](/imgs/blogs/on-policy-distillation-3.webp)

Now connect the directions to *policy*. This is the link people miss. Reverse KL, $\mathrm{KL}(q_\theta \| p_T)$, takes its expectation over the student's own distribution. **To estimate it you must sample from the student** — which means reverse KL is intrinsically on-policy. Forward KL, $\mathrm{KL}(p_T \| q_\theta)$, takes its expectation over the teacher, so you sample from the teacher (or a fixed teacher dataset) — intrinsically off-policy, and equal to ordinary cross-entropy/SFT on teacher data up to a constant. The direction of the KL and the source of the data are *the same choice viewed from two angles*.

| | Forward KL — $\mathrm{KL}(p_T\|q_\theta)$ | Reverse KL — $\mathrm{KL}(q_\theta\|p_T)$ |
| --- | --- | --- |
| Expectation over | Teacher distribution | **Student** distribution |
| Sampling | From teacher / fixed data (off-policy) | From the student (**on-policy**) |
| Behavior | Mass-covering, mean-seeking | **Mode-seeking**, committal |
| Failure mode | Fills low-prob valleys → hedged, hallucinatory | Mode collapse if uncontrolled → repetitive |
| Equivalent to | SFT / cross-entropy on teacher text | A dense-reward policy objective |
| Best when | Student capacity ≈ teacher; want full coverage | Student ≪ teacher; want sharp, coherent output |

Neither direction is universally right. Forward KL is the correct objective when the student has enough capacity to actually cover the teacher and you want calibrated coverage (think same-architecture, mild compression). Reverse KL is the correct objective when there is a real capacity gap and you would rather the student be excellent at one thing than mediocre at everything — which describes essentially every "shrink a frontier model to something I can serve" project. Generalized methods, which we will meet shortly, interpolate between them with a tunable knob precisely because the right answer depends on the capacity gap.

## 3. On-policy distillation, defined

We now have the two ingredients — on-policy data and a mode-seeking divergence — so the definition writes itself. The objective is

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\;
\mathbb{E}_{y \sim q_\theta(\cdot\mid x)}
\left[\; \sum_{t=1}^{|y|} \mathrm{KL}\big(q_\theta(\cdot \mid x, y_{\lt t}) \,\big\|\, p_T(\cdot \mid x, y_{\lt t})\big) \;\right],
$$

read as: draw a prompt $x$; let the student sample a full rollout $y$ (this is the on-policy part); at every position $t$, compute the divergence between the student's and teacher's *full conditional distributions* at the state the student actually reached; sum and minimize.

Two details make this both correct and efficient, and both are worth dwelling on.

**The contexts are on-policy; the per-token loss is supervised.** The prefixes $y_{\lt t}$ that condition each term come from the student's own distribution — that is what cures exposure bias, because the student is graded on the states it really visits. But the loss *at* each position is an ordinary divergence between two known distributions; we have the teacher's logits and the student's logits at that state, so we compute the KL analytically over the vocabulary. We do **not** sample an action and estimate a noisy reward there. In practice we put a stop-gradient on the sampling step (use the student-generated tokens as fixed context) and only backpropagate through the per-position divergence. This is the crucial efficiency win over naive policy gradient: the supervision is *dense* (every token) and *low-variance* (an exact divergence, not a Monte-Carlo reward estimate).

**It is DAgger for language models.** The imitation-learning move that beats behavioral cloning is to roll out the *learner*, then ask the *expert* what it would have done at each visited state, and train on that. On-policy distillation is exactly this: the student rolls out, the teacher answers "here is my distribution at each of your states," and we train toward it. The teacher is an interactive oracle, queryable at any state — not a frozen transcript. That is why "the teacher never demonstrates recovery from the student's mistakes" stops being true: the teacher demonstrates the correct distribution *at the very mistaken state the student just produced*.

A natural worry: if the student samples its own (initially bad) rollouts, are we not just training on garbage? No — and this is the elegant part. The teacher scores even a bad rollout with a *correct* target distribution at every step. A student that wandered into a confused state gets told, token by token, what a competent next-token distribution looks like *from there*. Over training, the rollouts improve because the policy improves, so the state distribution the student is graded on tracks the student's actual competence. The curriculum is automatic and always exactly at the frontier of what the student can currently do.

## 4. It is reinforcement learning with a dense reward

Here is where on-policy distillation stops looking like distillation and starts looking like RL — because it *is* RL, of a particularly well-behaved kind. Rewrite the per-token reverse-KL term as a reward. Define the reward at token $t$ as

$$
r_t \;=\; -\,\mathrm{KL}\big(q_\theta(\cdot\mid s_t)\,\|\,p_T(\cdot\mid s_t)\big)
\quad\text{(or, per-sampled-token, } r_t = \log p_T(y_t\mid s_t) - \log q_\theta(y_t\mid s_t)\text{).}
$$

Maximizing expected reward over student rollouts is on-policy policy optimization. The teacher *is* the reward model. But look at what kind of reward it is: a value at **every token**, computed analytically, with the teacher always knowing the right answer at any state. Compare that to reinforcement learning from verifiable rewards (RLVR) — the [GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo)-style recipe behind the DeepSeek-R1 reasoning breakthrough — where you get a *single scalar* at the end of a 2,000-token trajectory ("the final answer was correct: +1").

<figure class="blog-anim">
<svg viewBox="0 0 760 320" role="img" aria-label="A student rollout of ten tokens: on-policy distillation lights a graded reward bar under every token as a cursor sweeps left to right, while RLVR lights a single reward only after the final token" style="width:100%;height:auto;max-width:880px">
<style>
.d4-tok{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.d4-tlbl{font:600 13px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.d4-rew{fill:var(--accent,#6366f1);opacity:1}
.d4-end{fill:#e5484d;opacity:1}
.d4-cur{fill:var(--accent,#6366f1);opacity:.16}
.d4-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.d4-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes d4-fill{0%{opacity:.12}6%{opacity:1}94%{opacity:1}100%{opacity:.12}}
@keyframes d4-late{0%,78%{opacity:.12}88%,100%{opacity:1}}
@keyframes d4-sweep{0%{transform:translateX(0)}90%,100%{transform:translateX(594px)}}
.d4-r0{animation:d4-fill 9s linear infinite}
.d4-r1{animation:d4-fill 9s linear infinite .7s}
.d4-r2{animation:d4-fill 9s linear infinite 1.4s}
.d4-r3{animation:d4-fill 9s linear infinite 2.1s}
.d4-r4{animation:d4-fill 9s linear infinite 2.8s}
.d4-r5{animation:d4-fill 9s linear infinite 3.5s}
.d4-r6{animation:d4-fill 9s linear infinite 4.2s}
.d4-r7{animation:d4-fill 9s linear infinite 4.9s}
.d4-r8{animation:d4-fill 9s linear infinite 5.6s}
.d4-r9{animation:d4-fill 9s linear infinite 6.3s}
.d4-emv{animation:d4-late 9s linear infinite}
.d4-cmv{animation:d4-sweep 9s linear infinite}
@media (prefers-reduced-motion:reduce){.d4-r0,.d4-r1,.d4-r2,.d4-r3,.d4-r4,.d4-r5,.d4-r6,.d4-r7,.d4-r8,.d4-r9,.d4-emv{animation:none}.d4-cmv{animation:none;opacity:0}}
</style>
<text class="d4-lbl" x="40" y="40">One student rollout — same tokens, two reward schemes</text>
<rect class="d4-tok" x="40"  y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="100" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="160" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="220" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="280" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="340" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="400" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="460" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="520" y="70" width="54" height="46" rx="6"/>
<rect class="d4-tok" x="580" y="70" width="54" height="46" rx="6"/>
<text class="d4-tlbl" x="67"  y="99">t1</text>
<text class="d4-tlbl" x="127" y="99">t2</text>
<text class="d4-tlbl" x="187" y="99">t3</text>
<text class="d4-tlbl" x="247" y="99">t4</text>
<text class="d4-tlbl" x="307" y="99">t5</text>
<text class="d4-tlbl" x="367" y="99">t6</text>
<text class="d4-tlbl" x="427" y="99">t7</text>
<text class="d4-tlbl" x="487" y="99">t8</text>
<text class="d4-tlbl" x="547" y="99">t9</text>
<text class="d4-tlbl" x="607" y="99">t10</text>
<rect class="d4-cur d4-cmv" x="40" y="66" width="54" height="120" rx="6"/>
<text class="d4-sub" x="40" y="150">on-policy distillation: a graded KL reward under every token (dense)</text>
<rect class="d4-rew d4-r0" x="44"  y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r1" x="104" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r2" x="164" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r3" x="224" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r4" x="284" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r5" x="344" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r6" x="404" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r7" x="464" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r8" x="524" y="160" width="46" height="40" rx="4"/>
<rect class="d4-rew d4-r9" x="584" y="160" width="46" height="40" rx="4"/>
<text class="d4-sub" x="40" y="250">RLVR / GRPO: one scalar reward, only after the last token (sparse)</text>
<rect class="d4-end d4-emv" x="584" y="260" width="46" height="40" rx="4"/>
<text class="d4-tlbl" x="607" y="287">+1</text>
</svg>
<figcaption>Both schemes are on-policy — the student samples its own rollout. But on-policy distillation hands back a graded reward at every token (the teacher's per-token KL), while RLVR only learns from a single end-of-sequence scalar. Dense supervision is the whole efficiency story.</figcaption>
</figure>

This density is the entire efficiency argument, and it shows up in three places at once:

1. **Credit assignment is free.** RLVR's single scalar must be smeared back across every token by an advantage estimator — was it token 12 or token 1,400 that earned the +1? The algorithm does not know; it can only guess, slowly, over many samples. On-policy distillation already *has* a target at token 12 and token 1,400 independently. There is no credit-assignment problem because there is no credit to assign — every position is supervised directly.
2. **Variance is low.** An RLVR gradient estimate from a sparse, high-variance reward needs large batches and many rollouts per prompt (GRPO samples a whole group per prompt precisely to beat down variance). The per-token KL is an exact quantity, not a sample mean; the gradient is correspondingly quiet.
3. **The reward model is the teacher, and it is always right.** RLVR either needs a verifier (only available for math/code with checkable answers) or a learned reward model (which can be hacked). The teacher's distribution is a dense reward defined at *every* state for *every* task, with no separate model to train, no reward-hacking surface, and no value network. As we will see in the infra section, that is a large practical saving.

The cost, of course, is that on-policy distillation can only take you as far as the teacher. RLVR can in principle discover behaviors the teacher never had (R1's emergent self-verification came from RL, not distillation). The practitioner's read: use on-policy distillation to *cheaply* transfer everything the teacher already knows into a smaller body, and reserve precious RLVR compute for pushing past the teacher's frontier. They compose — many production pipelines do on-policy distillation first, then a short RLVR phase. For the broader RL design space this sits in, the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) and the survey of [DAPO, Dr. GRPO, and GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) are the natural companions.

## 5. What actually gets computed at each token

Theory is clean; the implementation has three places to get it wrong. Let us write the training step plainly and then dissect the gotchas. The core loop is short:

```python
import torch
import torch.nn.functional as F

def on_policy_distill_step(student, teacher, prompts, optimizer,
                           max_new_tokens=512, temperature=1.0):
    prompt_len = prompts["input_ids"].shape[1]

    # 1. ON-POLICY: the student rolls out its OWN trajectories (no grad here;
    #    sampling is a stop-gradient — we only need the tokens it produced).
    student.eval()
    with torch.no_grad():
        gen = student.generate(**prompts, do_sample=True, temperature=temperature,
                               max_new_tokens=max_new_tokens,
                               return_dict_in_generate=True)
    seqs = gen.sequences                       # [B, prompt_len + gen_len]

    # 2. Re-score the SAME tokens: student WITH grad, teacher WITHOUT.
    student.train()
    s_logits = student(seqs).logits[:, :-1]    # logits predicting token t from <t
    with torch.no_grad():
        t_logits = teacher(seqs).logits[:, :-1]

    # 3. Temperature-match both distributions, then per-token REVERSE KL,
    #    summed over the FULL vocab at each visited state -> dense + analytic.
    s_logp = F.log_softmax(s_logits / temperature, dim=-1)
    t_logp = F.log_softmax(t_logits / temperature, dim=-1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1)        # [B, T] reverse KL

    # 4. MASK the prompt: only response tokens carry the distillation signal.
    resp_mask = torch.arange(kl.shape[1], device=kl.device) >= (prompt_len - 1)
    loss = (kl * resp_mask).sum() / resp_mask.sum().clamp(min=1)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()
```

Three things in that snippet are load-bearing and routinely botched:

**Response masking.** The KL is only meaningful on tokens the student *generated*. Including the prompt tokens in the loss trains the student to predict the (fixed) prompt, which is both pointless and a silent dilution of the real signal — on a 256-token prompt with a 512-token response, a third of your loss is noise if you forget the mask. The off-by-one (`prompt_len - 1`) is because `logits[:, :-1]` predicts `seqs[:, 1:]`, so the position that first predicts a *response* token is `prompt_len - 1`.

**Temperature matching.** If you sample rollouts at temperature $\tau$ but compute the KL on un-tempered logits, you are scoring a distribution the student did not actually sample from, and the gradient is biased. Score at the same temperature you sampled at (or, more carefully, apply an importance correction). It is a one-line bug that produces a slow, mysterious quality regression.

**Full-vocab vs sampled-token.** The snippet computes the exact KL over the full vocabulary at each position — the low-variance, dense form. The cheaper alternative scores only the *sampled* token's log-prob ratio, which is an unbiased but higher-variance estimate. Full-vocab is almost always worth it; the only reason not to is bandwidth, which the top-k trick below handles.

The divergence itself is a swappable component. Keeping the four standard choices in one place clarifies what each knob does:

```python
import math

def forward_kl(s_logp, t_logp):     # KL(teacher || student): mass-covering, off-policy
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1)

def reverse_kl(s_logp, t_logp):     # KL(student || teacher): mode-seeking, on-policy
    return (s_logp.exp() * (s_logp - t_logp)).sum(-1)

def jsd_beta(s_logp, t_logp, beta=0.5):     # generalized JSD (the GKD interpolant)
    # Mixture m = beta * teacher + (1 - beta) * student, in log space.
    m_logp = torch.logsumexp(
        torch.stack([t_logp + math.log(beta), s_logp + math.log(1 - beta)]), dim=0)
    return beta * (t_logp.exp() * (t_logp - m_logp)).sum(-1) \
         + (1 - beta) * (s_logp.exp() * (s_logp - m_logp)).sum(-1)

def topk_teacher_kl(s_logits, t_logits, k=20):
    # Restrict the divergence to the teacher's top-k tokens. Cuts the teacher
    # logits you must store/transfer from ~150k floats/token to ~20 — a ~1000x
    # bandwidth win that barely moves the loss, since the tail mass is tiny.
    idx = t_logits.topk(k, dim=-1).indices
    s_sel = torch.gather(F.log_softmax(s_logits, -1), -1, idx)
    t_sel = torch.gather(F.log_softmax(t_logits, -1), -1, idx)
    return (s_sel.exp() * (s_sel - t_sel)).sum(-1)
```

The `topk_teacher_kl` variant deserves emphasis because it is what makes the whole thing affordable at scale. A full teacher distribution is one float per vocabulary entry per token — for a 150k vocabulary and a 512-token response, that is ~77M floats *per rollout* if you wanted to cache it. Keeping only the teacher's top-20 logits per token throws away a negligible amount of probability mass (the tail past the top-20 is usually well under 1%) and reduces the storage and inter-process transfer by three orders of magnitude. It is the on-policy-distillation analogue of the top-k tricks you already use to keep [inference memory under control](/blog/machine-learning/large-language-model/kv-cache).

## 6. The method map: where on-policy distillation sits

It helps to see the whole family on two axes at once. Distillation methods differ in *how on-policy the training data is* (left: teacher-generated, right: student-generated) and *how dense the supervision signal is* (bottom: a sparse/hard label, top: a full per-token distribution). Plot the major methods and the structure jumps out.

![The distillation method map](/imgs/blogs/on-policy-distillation-5.webp)

The single most clarifying thing on this map is the right edge. **On-policy distillation and RLVR live at the same x-coordinate** — both train on the student's own rollouts — and differ only in *vertical position*: on-policy distillation sits at the top (a dense, per-token teacher distribution) while RLVR sits at the bottom (one sparse outcome reward). They are siblings, not opposites. Everything we said about the efficiency of dense rewards is just "move up the right edge."

The off-policy methods cluster on the left. Sequence-level KD (bottom-left) trains on the teacher's hard tokens; classic logit KD (top-left) trains on the teacher's full distribution but still over the teacher's own states; SFT-distill / R1-Distill is the same off-policy column. The generalized methods — GKD and MiniLLM — sit toward the upper-right because they push the data rightward (student rollouts) while keeping a dense divergence. The table makes the columns precise:

| Method | Data source | Divergence / signal | Reward density | Rollout cost | Reach for it when |
| --- | --- | --- | --- | --- | --- |
| Sequence-level KD | Teacher text (off-policy) | Hard-token cross-entropy | Per token (hard) | None | Cheapest bootstrap; tokenizers may differ |
| Classic logit KD | Teacher text (off-policy) | Forward KL on logits | Per token (soft) | None | Mild compression, student ≈ teacher |
| SFT-distill (R1-Distill) | Teacher text (off-policy) | Cross-entropy | Per token (hard) | None | Scale + simplicity; you have a big teacher corpus |
| **GKD** | Mixed → student (tunable λ) | Generalized JSD(β) | Per token (soft) | Medium–high | The default modern recipe; tune λ, β to capacity gap |
| **MiniLLM** | Student (on-policy) | Reverse KL + policy gradient | Per token (soft) | High | Large capacity gap; want sharp students |
| **On-policy distillation** | Student (on-policy) | Reverse KL / dense teacher | Per token (soft) | High | Reasoning, agents, long horizons; have a strong teacher |
| RLVR / GRPO | Student (on-policy) | Verifier / reward model | **Sparse** (per sequence) | High | Pushing past the teacher; verifiable rewards exist |

## 7. Building the loop in practice

The infrastructure is where on-policy distillation reveals its second advantage over full RL: there is almost nothing to train except the student. The teacher only ever runs *frozen forward passes*, which means you can serve it on a separate, throughput-optimized engine and treat it as a stateless scoring service. There is no reward model to train, no value/critic network to keep in sync — the two big memory and complexity sinks of PPO-style RL simply do not exist here.

![Building the loop: only the student trains](/imgs/blogs/on-policy-distillation-6.webp)

The dataflow is: the trainer holds the student (with optimizer state and gradients, the expensive part), samples a batch of rollouts into a buffer, ships those rollouts to the frozen teacher server for scoring, caches the teacher's top-k logits, computes the per-token reverse-KL loss against the student's own re-scored logits, and steps. The teacher server is just an inference engine — vLLM, SGLang, whatever you already run — pinned to the teacher weights and asked only for logprobs. In the easiest setup, libraries like TRL hold the teacher in-process; at scale you split it out so teacher inference and student training scale independently.

If you want to be running this afternoon rather than building it, TRL ships a `GKDTrainer` that implements exactly this loop with the two knobs that matter exposed directly:

```bash
# Optional: serve the frozen teacher on a throughput engine so rollout scoring
# does not contend with student-training GPUs. TRL can also keep it in-process.
vllm serve Qwen/Qwen3-32B \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

```python
from trl import GKDConfig, GKDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

tok     = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B",  torch_dtype="bfloat16")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B", torch_dtype="bfloat16")

cfg = GKDConfig(
    lmbda=1.0,          # fraction of ON-POLICY data: 1.0 = student rollouts only,
                        #   0.0 = pure supervised on the dataset, in-between = GKD mix
    beta=0.0,           # generalized-JSD coefficient: 0.0 -> forward KL,
                        #   1.0 -> reverse KL, (0,1) -> a JSD interpolation
    temperature=1.0,    # sampling/scoring temperature (keep them matched!)
    max_new_tokens=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    output_dir="qwen3-8b-onpolicy-distill",
)

trainer = GKDTrainer(model=student, teacher_model=teacher,
                     args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
```

Two config lines carry the whole research literature. `lmbda` is the on-policy fraction $\lambda$ — at `1.0` every training sequence is a fresh student rollout (fully on-policy); at `0.0` you are doing plain supervised distillation on a fixed dataset; in between you mix. `beta` is the divergence interpolation between forward and reverse KL. The remaining practical knobs do not all have config flags but every serious run tunes them:

| Knob | What it controls | Typical default | The gotcha |
| --- | --- | --- | --- |
| On-policy fraction `λ` (`lmbda`) | Share of student-generated vs fixed data | 1.0 for reasoning; 0.25–0.5 to bootstrap | Pure on-policy from a *cold* student wastes early compute on garbage rollouts; warm up with λ<1 |
| Divergence `β` (`beta`) | Forward ↔ reverse KL | 0 (reverse) for big capacity gaps | Pure reverse KL can mode-collapse; a little JSD (β≈0.1) regularizes |
| Sampling temperature | Diversity of rollouts | 1.0 | Must match the scoring temperature or the gradient is biased |
| Teacher top-k | Logits stored/transferred per token | 20–50 | Too small clips real teacher mass on high-entropy tokens |
| Rollout refresh cadence | How often rollouts are regenerated | Every step (fully on-policy) | Reusing rollouts for N steps makes them stale → off-policy drift |

That last knob is the one that quietly turns an on-policy method off-policy. Generating fresh rollouts every gradient step is the purest form but the most expensive, because generation is autoregressive and slow. The standard optimization is to reuse a batch of rollouts for several gradient steps — but after a step the student has changed, so those rollouts are now drawn from a slightly *stale* policy, i.e. off-policy. The fix borrowed straight from PPO is **importance weighting with clipping**: weight each token's loss by the ratio of the current to the sampling policy and clip the ratio to a trust region. A little staleness is fine and a big throughput win; a lot of staleness reintroduces exactly the distribution mismatch you adopted on-policy distillation to escape. This is the same on-policy-vs-off-policy tension that the [GRPO family](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) spends most of its design budget on, and the lessons transfer directly.

### Second-order optimization: the tokenizer trap

There is one prerequisite that silently rules out many teacher-student pairs: **on-policy distillation as described requires the teacher and student to share a vocabulary.** The per-token KL compares two distributions over the *same* token set at the *same* positions. If the teacher tokenizes "distillation" as three pieces and the student as four, there is no per-position correspondence to take a KL over. This is why most on-policy distillation happens *within* a model family (Qwen3-32B → Qwen3-8B, Gemma-27B → Gemma-2B): same tokenizer, by construction. Cross-tokenizer distillation exists — Universal Logit Distillation (ULD) and minimum-edit-distance alignment methods relax the requirement — but it is harder, lossier, and an active research area rather than a turn-the-crank recipe. If your teacher and student come from different families, budget for this or pick a same-family teacher.

## 8. Case studies from production

Patterns are easier to trust when attached to names. Here are seven, spanning the method's origins, its modern incarnations, and one instructive counter-example.

### 1. MiniLLM — making reverse KL behave for LLMs

MiniLLM (Gu et al.) was the work that demonstrated reverse-KL, on-policy distillation could beat forward-KL SFT for instruction-following LLMs — and, just as importantly, cataloged *why naive reverse KL is unstable* and how to fix it. The objective needs a policy gradient because the expectation is over the student, and a vanilla REINFORCE estimator on long sequences has brutal variance and a tendency to reward-hack length. Their fixes are the canonical bag of tricks: a **single-step decomposition** of the sequence-level objective that turns one high-variance trajectory reward into a sum of lower-variance per-step terms; **teacher-mixed sampling** (interpolating student and teacher during rollout generation) to keep early rollouts from being useless; and **length normalization** to stop the model from gaming the reward with verbosity. The headline result was that distilled students in the 120M–1.3B range, trained with reverse KL on their own outputs, produced more *coherent and calibrated* generations than forward-KL baselines — lower exposure bias, less hallucination on long responses — exactly the mode-seeking vs mass-covering story made empirical. The lesson that outlived the specific numbers: on-policy distillation is RL, so it inherits RL's variance problems, and you should reach for RL's variance-reduction toolbox.

### 2. GKD — the on-policy fraction is the dominant variable

Generalized Knowledge Distillation (Agarwal et al., DeepMind, 2023) is the paper that gave the field its default recipe and its two knobs. GKD unifies the design space with the generalized Jensen-Shannon divergence (the $\beta$ interpolation between forward and reverse KL) and an explicit **on-policy data fraction** $\lambda$. Its central empirical finding is the one worth tattooing on the wall: across summarization, translation, and arithmetic-reasoning tasks, **the fraction of on-policy (student-generated) data was a larger lever than the choice of divergence.** Moving from off-policy to on-policy data improved students more than any forward-vs-reverse-KL tuning. This is the result that reframed the whole conversation — "distillation" had been a question of *which loss*; GKD showed it was really a question of *whose data*. It also showed the two compose with RLHF: you can run GKD on a model and then RL it, stacking the cheap dense-reward transfer with expensive frontier-pushing RL. TRL's `GKDTrainer`, used above, is a direct implementation.

### 3. Thinking Machines Lab — reasoning at a fraction of RL's compute

In 2025, Thinking Machines Lab published the clearest modern articulation of on-policy distillation as "RL with a dense reward," along with concrete reasoning results. The flagship demonstration distilled a strong reasoning teacher into a Qwen3-8B student on math-competition-style problems, reporting that on-policy distillation reached a target reasoning accuracy at *roughly an order of magnitude* less compute than reaching the same accuracy with reinforcement learning from outcome rewards — because every token of every rollout carried a learning signal instead of one scalar at the end. Their second contribution was a *personalization / continual-learning* recipe: when you fine-tune a model on new domain knowledge it tends to forget its old assistant behavior (catastrophic forgetting). Their fix was to distill on-policy against a *frozen earlier checkpoint of the model itself* as the teacher — the student practices the new domain on its own rollouts while a snapshot of its former self keeps grading it toward the old behavior, preserving general capability while absorbing the new material. The same dense-reward machinery, pointed at a self-teacher, becomes an anti-forgetting regularizer.

### 4. Gemma 2 and Gemma 3 — distillation at pretraining scale

Google's open Gemma models are the most prominent demonstration that distillation belongs at *pretraining* scale, not just post-training. Rather than training the smaller Gemma variants from scratch on raw tokens, the recipe trains them to match the distribution of a much larger teacher (a Gemini-class model), using the teacher's per-token probabilities as the supervision target over enormous token budgets. The smaller members of the family punch well above their parameter count specifically because they were raised on a frontier teacher's distribution rather than on next-token prediction of web text alone. While the public details emphasize the soft-target distillation objective more than the on-policy rollout loop per se, Gemma is the canonical proof point that "learn the teacher's distribution" scales to the regime where it matters most, and that small models distilled from large teachers dominate same-size models trained conventionally.

### 5. DistiLLM — killing the rollout cost and the instability together

DistiLLM (Ko et al., 2024) attacked the two practical pains of on-policy distillation head-on: reverse-KL instability and the cost of generating fresh student rollouts. Its first idea is a **skew KL divergence** — instead of $\mathrm{KL}(q\|p)$ it uses $\mathrm{KL}(q \,\|\, \alpha p + (1-\alpha) q)$ (and a skew forward variant), mixing a little of the *other* distribution into the reference. This bounds the gradient and provably stabilizes training, removing the mode-collapse and exploding-gradient failure modes that MiniLLM patched with heuristics. Its second idea is an **adaptive off-policy approach**: rather than regenerating rollouts every step (pure on-policy, expensive) or never (pure off-policy, drift), DistiLLM maintains a replay buffer of student generations and adaptively decides when fresh rollouts are worth the cost. The combination reported large wall-clock speedups over fully-on-policy baselines at matched quality — the practical message being that "on-policy" is a dial, not a binary, and the sweet spot is usually a controlled amount of staleness.

### 6. DeepSeek-R1-Distill — the off-policy counter-example, on purpose

The instructive counter-example is DeepSeek's own distilled series. After training R1 with large-scale RL, DeepSeek distilled it into smaller dense models (Qwen and Llama bases, 1.5B–70B) using **pure off-policy SFT** on roughly 800k samples generated by R1 — no on-policy rollouts, no reverse KL, just supervised fine-tuning on the teacher's text. And it worked remarkably well: the distilled models inherited a large fraction of R1's reasoning. Why choose off-policy when this whole article argues for on-policy? Two reasons, both pragmatic. First, *simplicity at scale* — generating 800k completions once and running standard SFT is trivially parallelizable and needs no teacher-in-the-loop scoring infrastructure across many student variants. Second, the DeepSeek team explicitly noted that distillation transferred reasoning more cheaply than running RL directly on the small models. What they gave up is the recovery signal: an off-policy-distilled R1 student has never been corrected on *its own* mistaken traces, so it is more brittle on long horizons than an on-policy-distilled equivalent would be. The takeaway is honest engineering: when the teacher corpus is huge, the tokenizers match, and you are spawning many student sizes, off-policy's operational simplicity can outweigh on-policy's quality edge. Know which trade you are making. (See the [knowledge-distillation deep dive](/blog/machine-learning/large-language-model/distillation-in-llm) for the full R1-Distill recipe.)

### 7. Speculative-decoding draft models — on-policy distillation for acceptance rate

A non-obvious application: training the *draft* model in [speculative decoding](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques). Speculative decoding pairs a small fast draft model with the large target model; the draft proposes several tokens, the target verifies them in one pass, and accepted tokens are free speedup. The acceptance rate — the whole value of the scheme — depends on how well the draft's distribution matches the target's *on the sequences the target actually decodes*. That is precisely an on-policy distillation objective: align the draft to the target on the target's own rollout distribution, not on a fixed corpus. Methods in this vein (e.g. on-policy knowledge distillation for draft models) report meaningfully higher acceptance rates than off-policy-trained drafts, because off-policy drafts are well-matched on the training text and poorly matched in the deployment distribution — the same exposure-bias story, now measured directly in tokens-accepted-per-step. It is a clean case where on-policy distillation's benefit is not a soft quality improvement but a hard, measurable throughput number.

## 9. When to reach for on-policy distillation — and when not to

The decision is rarely "on-policy or nothing." It is a small tree of practical questions, and the answer at the leaves is one of four methods.

![Which distillation should you reach for?](/imgs/blogs/on-policy-distillation-7.webp)

Walk it top-down. No stronger teacher available? Then distillation is off the table by definition — you need [RLVR](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) to learn from rewards. Have a strong teacher but a *different tokenizer*? Cross-vocabulary methods (ULD) or plain off-policy SFT on teacher text are your realistic options. Shared tokenizer and a tight budget or a cold student? Bootstrap with off-policy SFT-distill first — it is cheap and gets the student into a competent region fast. Shared tokenizer and you want to *maximize* quality on long-horizon or reasoning tasks? That is the home turf of on-policy distillation. In practice the strongest pipelines chain them: off-policy SFT to warm up, then on-policy distillation to close the exposure-bias gap, then optionally a short RLVR phase to push past the teacher.

Stepping back, this method did not appear from nowhere. It is the convergence of two decade-long threads — imitation learning's discovery that you must train on the learner's own state distribution, and knowledge distillation's discovery that soft teacher targets beat hard labels.

![The road to on-policy distillation](/imgs/blogs/on-policy-distillation-8.webp)

**Reach for on-policy distillation when:**

- You have a **strong teacher that shares the student's tokenizer** (typically same model family) and a real capacity gap to bridge.
- The task is **long-horizon**: multi-step reasoning, agent traces, tool-use sequences, code generation — anywhere a single early mistake compounds. The longer the horizon, the bigger the win over off-policy.
- You tried off-policy SFT-distill and the student looks great offline but **degrades on long deployments** — the textbook exposure-bias signature.
- You need teacher-level quality **far cheaper than RLVR** and you do not need to exceed the teacher.
- You are doing **continual learning / personalization** and want to absorb new knowledge without catastrophic forgetting (distill against a frozen self-checkpoint).
- You are training a **draft model** and acceptance rate is the metric.

**Skip it (or prefer something else) when:**

- **Teacher and student tokenizers differ** and you cannot afford cross-vocab alignment — use off-policy SFT or ULD.
- You are **spawning many student sizes from one huge teacher corpus** and operational simplicity dominates — off-policy SFT (the R1-Distill trade) is fine and far simpler.
- The student is **cold** and you went straight to pure on-policy — early rollouts are garbage and you will burn compute; warm up with λ<1 or an SFT phase first.
- You need to **surpass the teacher**, not match it — only RLVR (or a stronger teacher) can do that; on-policy distillation is capped at the teacher's frontier.
- The **capacity gap is tiny** (mild compression, same architecture) — plain forward-KL logit distillation may be simpler and perfectly adequate.
- You have **no teacher at all** — there is nothing to distill; that is an RL or pretraining problem.

The mental shift that makes all of this click is the one from the introduction: stop thinking of distillation as "copy the teacher's outputs" and start thinking of it as "let the student act, and have the teacher grade every action it takes." Once distillation is reframed as on-policy RL with a dense, free reward, every design decision — the divergence direction, the on-policy fraction, the staleness budget, the top-k cache — falls out of the reinforcement-learning playbook you already know. The teacher was never a script to memorize. It was always a coach standing behind the student, correcting the moves the student actually makes.

## Further reading

- [Knowledge Distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) — the off-policy foundation this article builds on, including the full DeepSeek-R1-Distill recipe.
- [Fine-Tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the sparse-reward RL sibling; read it alongside the "dense reward" section here.
- [GRPO vs DPO vs PPO: a decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — where on-policy distillation fits among post-training methods.
- [Beyond GRPO: DAPO, Dr. GRPO, and GSPO](/blog/machine-learning/large-language-model/beyond-grpo-dapo-dr-grpo-gspo) — the on-policy-vs-staleness and clipping machinery that transfers directly to the rollout-reuse knob.
- **Papers**: GKD (Agarwal et al., 2023); MiniLLM (Gu et al., 2023); DistiLLM (Ko et al., 2024); DAgger (Ross et al., 2011) for the imitation-learning ancestor; and the Thinking Machines Lab "On-Policy Distillation" write-up (2025) for the dense-reward framing.
