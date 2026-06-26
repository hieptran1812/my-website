---
title: "Why Language Models Need RLHF: The Alignment Problem in Practice"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A from-first-principles guide to why next-token pretraining cannot produce a helpful assistant, and how RLHF closes the gap with SFT, a reward model, and KL-constrained PPO — with runnable PyTorch and TRL code, worked KL and reward-model math, and the InstructGPT numbers that proved a 1.3B model can beat a 175B one."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "llm-alignment",
    "reward-modeling",
    "ppo",
    "machine-learning",
    "pytorch",
    "trl",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/why-language-models-need-rlhf-1.png"
---

A few years ago I watched a 6-billion-parameter language model, fresh off a perfectly healthy pretraining run, fail a task that a six-year-old would pass. We typed: "Explain how to make a paper airplane to a child." It produced, in flawless grammar, a *list of more questions*: "Explain how to make a paper boat to a child. Explain how to fold an origami crane. Explain how to..." The model had not malfunctioned. It was doing exactly what it was trained to do — predict the most likely continuation of the text — and on the internet, a line that looks like a how-to prompt is very often followed by *more* how-to prompts in a listicle, not by an answer. The model was a brilliant mimic of internet text and a terrible assistant. That gap, between "models the distribution of human text" and "does what I asked," is the alignment problem, and reinforcement learning from human feedback (RLHF) is the most successful practical attack on it we have.

This post is about *why* that gap exists, why supervised learning alone cannot close it, and why reinforcement learning — specifically optimizing a policy against a learned reward model while staying anchored to a reference — turns out to be the right tool. We will build the full RLHF loop from first principles: collect demonstrations and do supervised fine-tuning (SFT), collect pairwise human comparisons and train a reward model, then optimize the policy with PPO against that reward while a KL penalty keeps it from going off the rails. The pipeline in the figure below is the spine of everything that follows.

![The three-stage RLHF pipeline showing prompt collection feeding supervised fine-tuning, pairwise comparisons feeding a reward model, and PPO with a KL penalty producing an aligned language model.](/imgs/blogs/why-language-models-need-rlhf-1.png)

By the end you will be able to: derive why next-token prediction is the wrong objective for an assistant; explain the Bradley-Terry reward-model loss and compute it by hand; write a reward model in PyTorch and an RLHF loop in TRL; reason about why the KL term mathematically prevents reward hacking; and read the InstructGPT and Constitutional AI results critically rather than as marketing. We will tie everything back to the recurring spine of this series — *an agent interacting with an environment, collecting rewards, and updating a policy* — because RLHF is just that loop with a language model as the agent, a prompt as the state, and a human-trained reward model as the environment's reward signal. If you have not yet read [the policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) or [proximal policy optimization](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo), this post stands alone, but those two give you the optimizer we will be steering.

## 1. What pretraining gives you, and what it does not

Let us be precise about what a pretrained language model *is*, because the precision is what reveals the gap. A causal language model parameterized by $\theta$ defines a probability distribution over the next token given all previous tokens:

$$
p_\theta(x_t \mid x_{<t}) = \text{softmax}(f_\theta(x_{<t}))_{x_t}
$$

Training maximizes the log-likelihood of a giant corpus $\mathcal{D}$ — call it "the internet" — under this model:

$$
\mathcal{L}_{\text{pretrain}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{t} \log p_\theta(x_t \mid x_{<t}) \right]
$$

This objective has a single, well-defined target: reproduce the conditional distribution of tokens in $\mathcal{D}$. As the model gets larger and the data gets bigger, $p_\theta$ converges toward the *true distribution of internet text* $p_{\text{data}}$. That is the whole game, and it is a spectacular achievement — it gives you grammar, world knowledge, arithmetic-ish reasoning, code, translation, and a representation of language rich enough to be steered. But notice what the target *is not*: it is not "produce the response a thoughtful, helpful person would give." It is "produce the response that is statistically most likely to appear next on the internet."

Those two are wildly different, and the difference is not a rounding error. Consider what "most likely continuation" actually means for a few prompts:

- **"The best way to lose weight fast is"** — the internet is full of spammy clickbait, dangerous crash diets, and supplement ads. The most likely continuation is plausibly harmful nonsense, because that text is overrepresented in exactly this context.
- **"My elderly neighbor keeps leaving her door unlocked. How could someone"** — a pretrained model has no notion that completing this helpfully is a problem. It will happily continue toward burglary instructions because such text exists and is locally probable.
- **"Q: What is the capital of Australia? A:"** — here the most likely continuation might actually be correct (Canberra), but it might also be "Sydney" if the training data contained that common error frequently enough. The model has no incentive to be *honest*; it has an incentive to be *typical*.

This is the heart of it: a language model trained on next-token prediction learns to model the distribution of human text, but **"most likely continuation" is not the same as "helpful, harmless, honest response."** A model trained on internet text learns to produce text that *looks like* the internet — and the internet contains toxic, deceptive, manipulative, and unhelpful content alongside the good stuff. The model faithfully reproduces all of it, weighted by frequency, with no built-in preference for the parts we actually want.

There is a subtler failure too, the one my paper-airplane model exhibited: even when the content is benign, the *format* is wrong. The pretraining distribution is dominated by documents, not dialogues. A prompt that looks like an instruction is, in the training distribution, often part of a list of instructions, a homework assignment, or a forum post — not a question awaiting a direct answer. So the model continues the *document* rather than *responding* to you. This is sometimes called the "instruction-following gap," and it is purely a consequence of the objective. Nothing in maximum-likelihood training tells the model that the human typing the prompt wants an answer rather than a continuation.

A useful way to hold this: pretraining gives you a *model of the world's text*, which is a superhuman simulator of "what comes next." What you want for a product is an *agent* that takes a goal and pursues it. The pretrained model contains the agent — it can role-play a helpful assistant if you prompt it just right — but it is buried inside a distribution that also contains the unhelpful troll, the scammer, the confidently-wrong forum poster, and the listicle-bot. Alignment is the process of *concentrating probability mass* onto the helpful behaviors and away from the rest.

There is a precise information-theoretic way to see why prompting alone cannot fully fix this, and it is worth a moment because it tells you why a *new training signal* is unavoidable. Few-shot prompting works by *conditioning*: you push the model into the region of its distribution where the conditioning context (a few examples of helpful Q&A) makes helpful continuations locally probable. This genuinely helps — it is why in-context learning works at all — but it has a hard ceiling. Conditioning can only move probability mass that *already exists* in the conditional distribution; it cannot create behaviors the model never learned, and it cannot override the model's strong priors when the prompt is ambiguous. Worse, the prompt consumes context budget and is brittle: a slightly adversarial user input can knock the model out of the "helpful assistant" basin and back into "internet simulator" mode (this is the mechanism behind many jailbreaks). Prompting *steers* the existing distribution; it does not *reshape* it. To reshape the distribution — to make helpful behavior the model's default rather than a fragile conditioned state — you have to change the weights, and to change the weights toward "helpful" you need a gradient that points at "helpful," which the maximum-likelihood objective simply does not provide.

One more property of pretrained models matters for what comes next: **calibration**. A well-pretrained model is, somewhat famously, well-calibrated on its next-token predictions before alignment — when it says a token has probability 0.7, that token does appear about 70% of the time. This is a direct consequence of optimizing log-likelihood, which is a *proper scoring rule* that is minimized exactly when predicted probabilities match true frequencies. Hold onto this fact, because RLHF will partially *break* it: a model optimized to produce preferred responses becomes over-confident, collapsing its distribution onto the highest-reward modes and losing the calibrated uncertainty pretraining gave it. That is one of the real, measurable costs of alignment, and it is the kind of trade-off this series insists you see clearly rather than paper over.

![A before-and-after comparison contrasting pretraining, which models the internet distribution and mixes toxic and helpful text, with RLHF, which optimizes a human-preference reward toward helpful harmless honest responses.](/imgs/blogs/why-language-models-need-rlhf-2.png)

The figure above frames the shift bluntly. On the left, the objective is "be a faithful sample from the internet." On the right, the objective is "be the response a human rater would prefer." These are different optimization targets, and you cannot get from one to the other by training harder on the first. You need a new signal. That signal is human preference, and the machinery for turning preference into gradients is RLHF.

## 2. The 3H triad and the alignment tax that wasn't

The framing the field converged on — popularized by Anthropic's early work and OpenAI's InstructGPT — is the **HHH triad**: a good assistant should be **Helpful** (it tries to do what you ask and asks for clarification when needed), **Harmless** (it avoids causing harm, refuses dangerous requests, and does not produce toxic content), and **Honest** (it tells you what it actually believes, expresses calibrated uncertainty, and does not fabricate). These three are not always in harmony — maximal harmlessness ("I cannot help with anything") destroys helpfulness, and maximal helpfulness can compromise harmlessness — and a large part of alignment engineering is navigating those tensions.

For years the prevailing fear was the **alignment tax**: the hypothesis that making a model safer and more aligned would necessarily make it dumber or less capable, that you would pay for safety in lost performance. The single most important empirical result from the InstructGPT paper (Ouyang et al., 2022, "Training language models to follow instructions with human feedback") is that, for the dominant axis people actually cared about, *the tax was negative*. RLHF did not just make the model safer; it made it dramatically **more helpful** — better at following instructions, staying on topic, and producing useful answers. The safety gains came along for the ride, but the headline was capability.

The number that made the field sit up: human labelers preferred outputs from the **1.3B-parameter InstructGPT model over the 175B-parameter GPT-3 base model**, despite the latter being more than 100× larger. Re-read that. Two orders of magnitude more parameters, more compute, more knowledge — and humans still preferred the small aligned model's outputs for real instruction-following tasks. The reason is exactly the gap from Section 1: GPT-3 was a better simulator of internet text, but InstructGPT was a better *assistant*, and what humans were rating was assistant-quality. Capability on the pretraining objective and capability on the thing-you-actually-want are different axes, and RLHF moves you along the second one cheaply.

That said, the tax is not literally zero everywhere. InstructGPT did show small regressions on some standard NLP benchmarks (a phenomenon the paper calls the "alignment tax" and partially mitigates with a pretraining-mix gradient term during RL). And later work documented real costs: over-refusal, verbosity, and a tendency to hedge. So the honest summary is: **RLHF dramatically improves the helpfulness humans perceive in open-ended tasks, with a small and manageable cost on narrow benchmarks.** That trade is overwhelmingly worth it for a product, which is why every frontier assistant ships with some form of preference optimization.

The tensions inside the triad are not abstract — they show up as concrete engineering dilemmas the preference data has to resolve. Take a prompt like "What household chemicals should I never mix?" A maximally-harmless-but-unhelpful model refuses ("I cannot discuss chemicals that could be dangerous"), which is both useless and slightly insulting, since the user is plainly asking a *safety* question. A maximally-helpful-but-careless model might volunteer a recipe for chlorine gas. The aligned behavior threads the needle: explain which combinations are dangerous and why, so the user can *avoid* them, while declining to give synthesis instructions. Getting the model to land in that narrow target requires preference data where labelers consistently rank the threading-the-needle answer above both the over-refusal and the over-share. This is why labeling *guidelines* are as important as labeling *volume*: without a clear, shared policy on how to resolve these conflicts, the comparison data is internally contradictory and the reward model learns a muddle. The single hardest part of running RLHF in practice is not the code; it is writing guidelines precise enough that 40 humans resolve the helpful-harmless tension the same way.

## 3. Why supervised fine-tuning alone is not enough

The obvious first move — and the correct first move — is supervised fine-tuning. If the problem is that the model continues documents instead of answering, just show it thousands of examples of (prompt, ideal answer) pairs and fine-tune with the same cross-entropy loss, restricted to the answer tokens. This is SFT, also called instruction tuning, and it works remarkably well as a *first stage*. It teaches the model the *format* of being an assistant: when you see a prompt, produce a direct, on-topic answer. Datasets like FLAN, Alpaca, and Dolly demonstrate that SFT alone gets you a usable instruction-follower.

So why isn't SFT the whole story? Three deep reasons, and understanding them is understanding why RL enters at all.

**Reason one: SFT teaches what to say in-distribution, but does not generalize to the full space of prompts.** Your demonstration set, however large, covers a tiny sliver of the prompts users will actually send. SFT pushes probability toward the demonstrated answers on the demonstrated prompts, but it gives you no *signal* about what to do on the millions of prompts you never wrote demonstrations for. The model interpolates, and the interpolation is only as good as the coverage. More fundamentally, cross-entropy loss on demonstrations is a *behavior-cloning* objective: it imitates the labeler's tokens. If the labeler made a suboptimal choice, SFT faithfully clones the suboptimality, and it has no mechanism to ever do *better* than the demonstration.

**Reason two — the deepest one: people can judge quality far better than they can produce it.** This is the single most important justification for the entire RLHF apparatus, so sit with it. Ask a human to write the *ideal* response to "Summarize this 40-page legal contract for a small-business owner." That is genuinely hard; it takes expertise and an hour. Now ask the same human to look at *two* candidate summaries and say which is better. That takes ninety seconds and far less expertise. The asymmetry between **generation** (hard) and **evaluation** (easy) is enormous, and it is the resource RLHF exploits. SFT can only consume the expensive, low-volume signal (demonstrations). RLHF can consume the cheap, high-volume signal (comparisons). You can collect ten preference judgments for the cost of one good demonstration, and those judgments cover responses the model *actually produces*, not idealized human prose.

**Reason three: SFT cannot represent preferences, only point targets.** Cross-entropy says "this exact token sequence is correct." But human preference is comparative and graded: response A is *better than* B, which is *better than* C, and the gaps matter. There is no single correct answer to "write a poem about autumn" — there is a partial order over a vast set of good answers. SFT collapses that partial order into "match this one demonstration." A reward model, trained on comparisons, *learns the partial order itself* and can then score any new response. That is a strictly richer signal.

There is a fourth, more technical reason that practitioners hit constantly: **SFT has no signal about what *not* to say.** Cross-entropy only ever pushes probability *up* on the demonstrated tokens; it never explicitly pushes probability *down* on bad responses, because there are no negative examples in a demonstration set. The model learns "here is a good answer" but never "here is a bad answer, avoid it." So an SFT model retains the ability to produce the toxic, deceptive, or unhelpful completions it learned in pretraining — those modes are still in the distribution, merely competing with the freshly-boosted good ones. RLHF, by contrast, trains on *comparisons*, which are inherently contrastive: every pair tells the model "prefer this, deprefer that," and the gradient pushes probability *down* on the rejected response as surely as it pushes it up on the chosen one. This contrastive structure is why RLHF can actively *suppress* harmful behavior in a way pure SFT structurally cannot.

Put these together and the conclusion is forced: SFT is necessary (it gives you the format and a competent starting policy) but not sufficient (it cannot exceed its demonstrations, cannot generalize from preferences, cannot exploit the cheap evaluation signal, and cannot suppress bad behavior). To go further you need to (a) learn a model of human preference from comparisons, and (b) optimize the policy against that learned preference — which is reinforcement learning. The next sections build exactly that.

#### Worked example: the generation-vs-evaluation asymmetry in numbers

Suppose a competent labeler can write one high-quality demonstration in 20 minutes, or render one pairwise comparison in 30 seconds. With a fixed budget of 100 labeler-hours:

- **Pure SFT path:** $100 \text{ hr} \times 60 / 20 = 300$ demonstrations. That is 300 (prompt, ideal answer) pairs.
- **Comparison path:** $100 \text{ hr} \times 3600 / 30 = 12{,}000$ pairwise comparisons.

Forty times more supervision signal for the same human cost — and the 12,000 comparisons are over responses the model actually generates, so they directly target the model's real failure modes rather than idealized prose. In the InstructGPT work the reward model was trained on roughly 33,000 comparisons drawn from on the order of 40,000 prompts; reproducing that as demonstrations would have cost vastly more labeler time and still would not have captured the comparative signal. This is not a marginal efficiency win. It is the reason RLHF is economically feasible at all.

## 4. The core RLHF loop, end to end

Here is the canonical three-stage recipe, the one InstructGPT made standard. Each stage produces an artifact the next stage consumes.

**Stage (a) — Supervised fine-tuning (SFT).** Collect demonstrations: labelers write ideal responses to a distribution of prompts. Fine-tune the pretrained base on these with token-level cross-entropy. Output: an SFT policy $\pi^{\text{SFT}}$ that follows instructions in-distribution and serves as both the *initialization* and the *reference* for the RL stage.

**Stage (b) — Reward model (RM) training.** For a fresh set of prompts, sample several completions from the SFT policy. Have labelers *rank* them (in practice, rank $K$ completions, which yields $\binom{K}{2}$ pairwise comparisons). Train a reward model $r_\phi(x, y)$ — typically the SFT model with the language-modeling head replaced by a scalar-output head — to predict which completion humans prefer. Output: a function that maps any (prompt, response) to a scalar quality score.

**Stage (c) — RL policy optimization (PPO).** Now treat generation as an RL problem. The prompt is the initial state, generating each token is an action, and the *episode reward* is the reward model's score of the full completion — *minus* a penalty for drifting away from the SFT reference. Optimize the policy $\pi_\theta$ (initialized from $\pi^{\text{SFT}}$) with PPO to maximize this reward. Output: the aligned policy.

Stage (a), the SFT step, is the least glamorous and the most under-rated. It is plain supervised fine-tuning — token-level cross-entropy on (prompt, response) pairs, with the loss masked so it only applies to the response tokens, not the prompt. TRL's `SFTTrainer` wraps the standard Hugging Face `Trainer` with the right data formatting and loss masking, so the whole stage is a few lines:

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# demonstrations: each row has a "prompt" and an ideal "completion"
dataset = load_dataset("trl-lib/Capybara", split="train")

config = SFTConfig(
    output_dir="./sft_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1,
    max_length=2048,
    # loss is computed on completion tokens only, not the prompt
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.save_model("./sft_model")  # this checkpoint is BOTH init and reference for RL
```

The one detail that matters and that people forget: `completion_only_loss=True` (loss masking). If you compute cross-entropy over the *prompt* tokens too, you waste capacity teaching the model to predict prompts — which it does not need, since the user supplies them — and you dilute the instruction-following signal. Mask the prompt, train on the response. This single SFT checkpoint is reused twice downstream: it initializes the PPO policy *and* it is frozen to serve as the KL reference. That dual role is why a strong SFT stage is load-bearing for everything that follows; a weak SFT base makes the reward model noisy and the KL reference a poor anchor.

The stack figure makes the dependency structure explicit: each layer assumes the one beneath it.

![A vertical stack showing the alignment layers from the pretrained base up through supervised fine-tuning, reward model training, PPO reinforcement learning, and the deployed model.](/imgs/blogs/why-language-models-need-rlhf-4.png)

Let us write the RL objective precisely, because every term earns its place. The policy maximizes

$$
\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \Big[ r_\phi(x, y) \Big] - \beta \, \mathbb{E}_{x \sim \mathcal{D}} \Big[ \text{KL}\big( \pi_\theta(\cdot \mid x) \,\|\, \pi^{\text{SFT}}(\cdot \mid x) \big) \Big]
$$

The first term says "produce responses the reward model scores highly." The second term — the **KL penalty** with coefficient $\beta$ — says "but stay close to the SFT policy." In practice the KL term is folded into the per-token reward, so the reward at the final token is $r_\phi(x,y)$ and the per-token reward includes $-\beta \log \frac{\pi_\theta(y_t \mid \cdot)}{\pi^{\text{SFT}}(y_t \mid \cdot)}$. The InstructGPT objective adds a third term, $\gamma \, \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}}[\log \pi_\theta(x)]$, a small pretraining-gradient mix that pulls the model back toward general language modeling to counteract the alignment tax (this variant is called "PPO-ptx").

The agent-environment framing from this whole series snaps cleanly into place. The **agent** is the language model. The **state** is the prompt plus the tokens generated so far. The **action** is the next token. The **environment** is degenerate in dynamics (the next state is just the state with one token appended) but the **reward** is rich: it comes from the learned reward model at the end of the episode, shaped per-token by the KL penalty. RLHF is the series spine — agent, environment, reward, policy update — with a language model wearing the agent costume.

## 5. Why RL is the right tool, not just a tool

It is fair to push back here: *if we have a reward model that scores responses, why not just do supervised learning against high-reward responses?* This is a real question with a real answer, and it reveals why RL specifically is the right machinery.

**RL lets the policy explore beyond the demonstration data.** Supervised learning can only imitate examples you already have. The reward model, by contrast, can score responses *the model generates on its own* — including responses no human ever wrote. During PPO, the policy samples completions, the reward model scores them, and the policy is nudged toward the high-scoring directions in its *own output distribution*. This means the model can discover phrasings, structures, and strategies that beat any demonstration, as long as the reward model recognizes them as good. You are optimizing the policy's actual behavior, on-policy, against a learned objective — which is the definition of reinforcement learning.

**The reward signal is a scalar over whole sequences, not a per-token target.** "This summary is better than that one" is a judgment about the entire response — its accuracy, structure, tone, and completeness. There is no ground-truth token-level label; the quality is non-decomposable. RL is built exactly for this: optimize a scalar reward that arrives at the end of a trajectory, propagating credit back to the tokens (the actions) that earned it. This is [the credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem) in its natural habitat, and policy-gradient RL is the standard solution.

**And the KL term — the reason RL here is safe — prevents reward hacking.** This is worth stating carefully because it is the single most important mechanical insight in RLHF. The reward model $r_\phi$ is an *imperfect, learned* proxy for human preference. It was trained on a finite set of comparisons over responses drawn from a particular distribution (the SFT policy's outputs). If you optimize against it without constraint, the policy will find **adversarial inputs** to the reward model — token sequences that score absurdly high but that humans would hate, because they live in a region of input space the reward model never saw during training and extrapolates wildly on. This is Goodhart's law in its purest form: when a measure becomes a target, it ceases to be a good measure. I have personally watched an unconstrained RLHF run discover that the reward model loved the word "comprehensive" and start emitting "comprehensive comprehensive comprehensive..." — a high-reward, zero-value sequence.

The KL penalty is the fix, and it is a *principled* one, not a hack. By penalizing divergence from the SFT reference, you force the policy to stay in the region of response space where the reward model was actually trained — the region of plausible, fluent, on-distribution text — and where its scores are therefore trustworthy. The policy can improve, but only by moving *within* the manifold of sensible text, not by teleporting to adversarial garbage. The KL term is what makes "optimize against a learned reward" safe; without it, RLHF degenerates into reward hacking within hours of training. We will derive its effect precisely in Section 8.

It helps to name the precise reason supervised learning on high-reward samples falls short of RL, because the distinction is subtle and people get it wrong constantly. Suppose you took the reward model and used it to *filter*: generate many completions, keep only the top-scoring ones, and fine-tune on those (this is "best-of-n distillation" or rejection sampling, and it is a real, useful technique). Why is full RL still stronger? Because filtering only ever trains on samples the *current* policy already produces — it cannot push probability toward responses the policy assigns low probability today but that would score well if only the policy explored toward them. RL, through the policy gradient, computes the *direction* in parameter space that increases expected reward, including directions that increase the probability of currently-rare-but-promising responses. Filtering is a zeroth-order, sample-and-keep method bounded by the current policy's support; policy-gradient RL is a first-order method that follows the reward landscape's slope and can climb toward responses the current policy rarely emits. The gap is exactly the gap between "keep the good rolls you happened to get" and "deliberately shift your behavior toward where good rolls come from." For easy alignment, best-of-n distillation often gets you most of the way (which is why it is popular as a cheaper alternative); for squeezing out the last quality, on-policy RL wins because it explores.

There is also a clean theoretical story (Rafailov et al., 2023, the DPO paper) showing that the KL-constrained reward-maximization objective has a *closed-form optimal policy*:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi^{\text{SFT}}(y \mid x) \, \exp\!\Big( \tfrac{1}{\beta} r_\phi(x, y) \Big)
$$

In words: the optimal aligned policy is the SFT policy *reweighted* by an exponential of the reward, with $\beta$ controlling how aggressively. This is a beautiful result — it says alignment is a *soft reweighting* of the base policy, never a wholesale replacement — and it is the seed from which DPO grows by inverting the relationship to optimize the policy directly. But the point for now is that the KL term is not arbitrary regularization; it defines a well-posed optimization problem with a sensible optimum.

## 6. What "alignment" actually means here (and what it doesn't)

The word "alignment" carries a lot of philosophical baggage — debates about superintelligent agents, value loading, and existential risk. Set all of that aside. In the RLHF context, alignment is an **operational, engineering** notion: the practical problem of making a model **follow instructions, be honest, and avoid harm** on the prompts real users send. We are not solving the philosophical problem of aligning an arbitrary optimizer with human values. We are solving the concrete problem of making *this* language model behave like a helpful assistant rather than an internet-text simulator.

That reframing matters because it tells you what success looks like and how to measure it. Aligned, operationally, means: when a user asks a question, the model answers it (helpfulness); when a user asks for something harmful, the model declines (harmlessness); when the model does not know, it says so rather than confabulating (honesty). These are *measurable behaviors*, gradable by human raters, which is precisely why preference-based RL applies. RLHF is a **practical alignment technique** — it does not guarantee the model has "internalized human values" in any deep sense; it guarantees the model's *outputs* match human preferences on the *distribution of prompts the raters represented*. That last clause is the load-bearing caveat, and most of the remaining hard problems (covered later) live in the gap between "the prompts raters saw" and "the prompts the world will send."

It is also worth being honest that RLHF aligns the model to the *labelers'* preferences and the *instructions given to those labelers*, not to some abstract universal good. If your labeling instructions over-weight politeness, you get a sycophantic model. If they under-specify how to handle ambiguous requests, you get a model that guesses. Alignment is only as good as the preference data, which is only as good as the rater pool and the guidelines. This is not a flaw to be embarrassed about; it is the correct way to understand the method. RLHF is a faithful optimizer of whatever preference signal you feed it — which makes the quality and construction of that signal the whole ballgame.

## 7. The labeler workforce: where the signal actually comes from

Everything above depends on a signal that does not exist in nature: human preference judgments at scale. Producing that signal is a serious operational undertaking, and the details shape the resulting model more than most code does.

**Who rates and how.** In the InstructGPT setup, OpenAI hired a team of around 40 contractors, screened for agreement with researcher judgments on a sensitivity-and-harm screening task. Labelers worked through a detailed guidelines document defining helpfulness, harmlessness, and honesty operationally, with worked examples of edge cases. The core task was **ranking**: given a prompt and $K$ sampled completions (InstructGPT used $K$ between 4 and 9), the labeler ordered them from best to worst. A ranking of $K$ items yields $\binom{K}{2}$ pairwise comparisons, which is why ranking is more efficient than collecting isolated pairs — a single ranking of 9 items produces 36 training pairs.

**Inter-rater agreement.** Humans do not agree perfectly on what makes a response good, and measuring that disagreement is part of doing this seriously. InstructGPT reported inter-annotator agreement around 73% (labelers agreed with each other about 73% of the time on which of two responses was better), and roughly 77% agreement between labelers and the researchers who wrote the guidelines. This is *not* noise to be eliminated; it is a ceiling. A reward model trained on this data cannot meaningfully exceed about 73% accuracy on held-out human preferences, because the humans themselves only agree 73% of the time. If your reward model reports 95% validation accuracy, you have a bug or a leak, not a triumph. I treat the inter-rater agreement number as the *target* reward-model accuracy and get suspicious when I beat it.

**The HHH criteria in practice.** The guidelines do real work resolving the tensions in the triad. For instance: when helpfulness and harmlessness conflict (a user asks for something dangerous), harmlessness wins and the model should decline *and explain why*. When honesty and helpfulness conflict (the helpful-sounding answer would require fabricating a fact), honesty wins and the model should express uncertainty. Encoding these priorities into the guidelines is what makes the preference data *coherent* enough for a reward model to learn a consistent function.

**Anthropic vs OpenAI protocols.** The two pioneering labs took noticeably different annotation routes. OpenAI's InstructGPT used *absolute-quality ranking* of $K$ completions against detailed guidelines. Anthropic's early RLHF work (Bai et al., 2022, "Training a Helpful and Harmless Assistant with RLHF") leaned on **binary comparisons collected through live conversation** — a crowdworker chats with two model variants and at each turn picks the better response — and crucially *separated* the helpfulness and harmlessness data streams, training on them with different sampling so the model could be pushed on both axes independently. Anthropic also pioneered using the *same* preference framework for red-teaming (deliberately eliciting harmful outputs to collect harmlessness comparisons). The high-level recipe is shared; the data-collection philosophy differs, and those differences propagate into model personality — which is one reason different labs' assistants "feel" different even when the algorithm is nominally the same.

## 8. The reward model: architecture, loss, and the math that makes it work

Time to make the reward model concrete, because it is the component people most often get wrong. The reward model is *not* a classifier of good-vs-bad in isolation. It is a function $r_\phi(x, y) \in \mathbb{R}$ trained so that *differences* in its output match human preference probabilities. The model architecture is almost always the SFT model itself, with the final unembedding layer (which produces logits over the vocabulary) replaced by a single linear head that produces one scalar — read off the last token's hidden state.

The training objective is the **Bradley-Terry model** of pairwise preference. Bradley-Terry posits that the probability a human prefers response $y_w$ ("winner") over $y_l$ ("loser") for prompt $x$ is the logistic of the reward difference:

$$
P(y_w \succ y_l \mid x) = \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) = \frac{1}{1 + e^{-(r_\phi(x, y_w) - r_\phi(x, y_l))}}
$$

This is exactly logistic regression on the *difference* of two scores produced by the *same* network. The negative log-likelihood loss over a dataset of comparisons is:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]
$$

Three properties of this loss are worth internalizing. First, it depends *only on the difference* $r_\phi(x,y_w) - r_\phi(x,y_l)$, so the reward model's absolute scale and offset are unidentified — the reward is only meaningful up to an additive constant per prompt, which is fine because the RL stage only ever uses reward *differences* (advantages). Second, it is convex in the score difference, which makes it well-behaved to optimize. Third, the loss is large when the model gets a pair *confidently wrong* and small when it gets a pair confidently right, which is exactly the gradient signal you want. The reward-model figure below shows the data flow: two responses go through the *same* network, producing two scores, which feed a single comparison loss.

![A branching dataflow graph showing a chosen response and a rejected response both passing through one shared reward model, producing two scalar scores that feed into a single Bradley-Terry pairwise loss.](/imgs/blogs/why-language-models-need-rlhf-3.png)

#### Worked example: computing a reward-model loss by hand

Suppose for prompt $x$ = "Explain photosynthesis to a 10-year-old" the reward model assigns the human-preferred (chosen) response a score $r_w = 2.1$ and the rejected response a score $r_l = 0.4$. The score difference is $\Delta = 2.1 - 0.4 = 1.7$. The model's predicted probability that the chosen response is preferred is:

$$
\sigma(1.7) = \frac{1}{1 + e^{-1.7}} = \frac{1}{1 + 0.1827} = 0.8455
$$

So the reward model is 84.55% confident the human-preferred response is indeed better. The contribution of this pair to the loss is $-\log(0.8455) = 0.1678$ — a small loss, because the model agrees with the human. Now contrast a pair the model gets *wrong*: chosen score $0.5$, rejected score $1.9$, so $\Delta = -1.4$, $\sigma(-1.4) = 0.198$, and the loss is $-\log(0.198) = 1.62$ — almost ten times larger. The gradient from this confidently-wrong pair is what corrects the model. If you average these two: $(0.168 + 1.62)/2 = 0.894$ nats. A well-trained reward model on a clean dataset lands around 0.6–0.7 nats average loss (corresponding to roughly 65–75% pairwise accuracy), bumping right up against the inter-rater agreement ceiling from Section 7.

Here is the reward model in PyTorch, written to be readable rather than clever:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """SFT backbone with the LM head swapped for a scalar reward head."""
    def __init__(self, base_model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden = self.backbone.config.hidden_size
        # single scalar score from the last token's hidden state
        self.reward_head = nn.Linear(hidden, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state            # (B, T, H)
        # index the final non-padding token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1    # (B,)
        gather_idx = seq_lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        last_tok = last_hidden.gather(1, gather_idx).squeeze(1)  # (B, H)
        return self.reward_head(last_tok).squeeze(-1)            # (B,)


def bradley_terry_loss(reward_chosen, reward_rejected):
    """Negative log-likelihood of the chosen response being preferred."""
    # equivalent to -log sigmoid(r_w - r_l); softplus form is numerically stable
    return F.softplus(reward_rejected - reward_chosen).mean()
```

And the training loop. Note that we run the *chosen* and *rejected* completions through the *same* model in one batch, then split the scores:

```python
from torch.utils.data import DataLoader

def train_reward_model(model, dataset, epochs=1, lr=1e-5, batch_size=8):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in loader:
            # each batch yields chosen and rejected, already tokenized + padded
            chosen_ids, chosen_mask = batch["chosen_ids"], batch["chosen_mask"]
            rej_ids, rej_mask = batch["rejected_ids"], batch["rejected_mask"]

            r_chosen = model(chosen_ids, chosen_mask)
            r_rejected = model(rej_ids, rej_mask)

            loss = bradley_terry_loss(r_chosen, r_rejected)
            # accuracy: fraction where chosen scored higher than rejected
            acc = (r_chosen > r_rejected).float().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(f"loss={loss.item():.4f}  pairwise_acc={acc.item():.3f}")
```

In practice you would not hand-roll this — the TRL library gives you a `RewardTrainer` that wraps exactly this logic with the right data collator. Here is the production-grade version:

```python
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", num_labels=1  # num_labels=1 => scalar reward head
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

config = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=1,
    max_length=1024,
    gradient_checkpointing=True,
)

# dataset must have "chosen" and "rejected" text columns
trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,   # e.g. Anthropic/hh-rlhf
    processing_class=tokenizer,
)
trainer.train()
```

The `Anthropic/hh-rlhf` dataset — released alongside the helpful-and-harmless paper — is the canonical public preference dataset, with `chosen` and `rejected` fields per comparison. Training a 0.5B reward model on it to ~67% pairwise accuracy is a single-GPU afternoon and a great way to internalize that the accuracy ceiling is the inter-rater agreement, not 100%.

Two practical details separate a reward model that works from one that quietly sabotages your RL run. **First, batch all completions for one prompt together.** InstructGPT made a specific and important choice here: rather than treating each of the $\binom{K}{2}$ comparisons from a $K$-way ranking as an independent training example (which would make completions from popular prompts dominate the gradient and badly overfit), they put all $K$ completions for a given prompt in the *same* batch and computed the loss over all $\binom{K}{2}$ pairs at once. This is both more compute-efficient (each completion is encoded once, not $K-1$ times) and more stable (it removes the overfitting from repeated completions). **Second, normalize the reward model's output before RL.** Because the Bradley-Terry loss is invariant to a per-prompt additive constant, the raw reward scale drifts arbitrarily during training. Before plugging the reward model into PPO, you typically shift and scale its outputs so that a reference set of completions has mean reward zero and unit-ish variance. Skipping this is a classic footgun: an un-normalized reward with a large offset makes the PPO advantage estimates explode and the value function impossible to fit.

A subtle modeling point that trips up newcomers: the reward model and the value function (the PPO critic) are *different objects*, even though both produce scalars from text. The reward model $r_\phi(x,y)$ is a *frozen* judge of full-completion quality, trained on human preferences and never updated during RL. The value function $V_\psi(x, y_{<t})$ is a *trainable* component of the PPO actor-critic that estimates expected future reward from a partial state, updated every PPO step to reduce advantage variance. Conflating them — for instance, using the reward model as the critic — is a real bug I have seen ship: the reward model has no notion of *partial*-sequence value, so it cannot serve as the per-token baseline that policy-gradient methods need for variance reduction. Keep them mentally separate: the reward model is the environment's reward signal; the value head is the agent's internal estimate of how well it is doing.

## 9. From reward model to aligned policy: RLHF with PPO in TRL

With a reward model in hand, the RL stage optimizes the policy. The algorithm is [PPO](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo) — its clipped surrogate objective is stable enough to optimize a multi-billion-parameter policy without the catastrophic divergences that plain policy gradient suffers, which is exactly why InstructGPT chose it. I will not re-derive PPO here (the PPO post does that in full); the relevant fact is that PPO updates the policy toward higher-reward actions while clipping the per-step update so a single batch cannot move the policy too far.

The RLHF-specific wiring is the reward computation. For each prompt, the policy generates a completion; the reward model scores it; the KL-to-reference penalty is computed per token; PPO uses the combined signal. TRL's `PPOTrainer` orchestrates this, holding four models in play: the **policy** (trainable), a **reference** (frozen copy of SFT, for the KL term), the **reward model** (frozen), and a **value head** (the PPO critic, for advantage estimation). Here is the loop, with the moving parts visible:

```python
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from transformers import AutoTokenizer

model_name = "lvwerra/gpt2-imdb"  # an SFT checkpoint stands in here
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# policy + value head (the actor-critic); reference is a frozen clone for KL
policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = create_reference_model(policy)

config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=128,
    mini_batch_size=16,
    init_kl_coef=0.2,     # the beta on the KL penalty (adaptively tuned)
    target=6.0,           # target KL; the coefficient auto-adjusts to hit it
    cliprange=0.2,        # PPO clip epsilon
    cliprange_value=0.2,
)

ppo_trainer = PPOTrainer(config, policy, ref_model, tokenizer)

generation_kwargs = {
    "min_length": -1, "top_k": 0.0, "top_p": 1.0,
    "do_sample": True, "max_new_tokens": 64,
    "pad_token_id": tokenizer.eos_token_id,
}

for batch in ppo_trainer.dataloader:
    query_tensors = batch["input_ids"]

    # 1. policy generates a completion for each prompt (the "actions")
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    batch["response"] = [tokenizer.decode(r) for r in response_tensors]

    # 2. score each (prompt, response) with the frozen reward model
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = [torch.tensor(reward_fn(t)) for t in texts]  # reward_fn wraps the RM

    # 3. PPO step: combines reward with the per-token KL-to-ref penalty
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)
    # stats["objective/kl"] is the KL you must watch; stats["ppo/mean_scores"] the reward
```

The two numbers to watch every step are the **mean reward** (should climb) and the **mean KL to the reference** (should rise then *stabilize* near `target`). The `init_kl_coef`/`target` machinery is an **adaptive KL controller**: TRL increases $\beta$ when measured KL overshoots the target and decreases it when KL is below target, keeping the policy on a leash of a fixed "distance" from SFT. If you see KL climbing without bound while reward also climbs, you are watching reward hacking happen in real time — the policy is racing away from the reference toward adversarial reward-model inputs, and you should lower `target` or raise `init_kl_coef`.

#### Worked example: a single RLHF batch step

Walk through one batch with concrete numbers to see how the pieces combine. Take a batch of 4 prompts; the policy generates one completion each, and the (normalized) reward model scores them as $r = [1.8, -0.6, 0.9, 2.3]$. Suppose the running mean of recent rewards is $0.8$, so the *advantage* contribution from the terminal reward, after subtracting the baseline, is roughly $A_{\text{term}} = [1.0, -1.4, 0.1, 1.5]$. Now the KL penalty: say the four completions accrued summed sequence-KL to the reference of $[5, 22, 4, 8]$ nats with $\beta = 0.05$, giving KL penalties $[0.25, 1.1, 0.2, 0.4]$. The KL-penalized returns become $[1.8 - 0.25, -0.6 - 1.1, 0.9 - 0.2, 2.3 - 0.4] = [1.55, -1.7, 0.7, 1.9]$. Notice the second completion: it scored a mediocre $-0.6$ from the reward model *and* drifted far from the reference (22 nats), so its penalized return is crushed to $-1.7$ — PPO will push hard *away* from whatever made it diverge. The fourth completion scored highest and stayed close to the reference (8 nats), so it keeps almost all its reward and PPO reinforces it. This is the leash doing its job: a high reward earned by drifting far from the reference is discounted, while a high reward earned *within* the reference's support is kept. The PPO clipped objective then takes a bounded step toward the high-return completions, and you repeat with the next batch.

A modern note: the field has largely shifted from full PPO toward **DPO** (Direct Preference Optimization) and **GRPO** (Group Relative Policy Optimization) for many use cases, because PPO's four-model dance is operationally heavy and finicky to tune. DPO — which we will cover in its own post — skips the reward model and RL loop entirely by exploiting that closed-form optimal-policy result from Section 5, optimizing the policy directly on preference pairs with a clever loss. For now, understand PPO as the *original, canonical, and still most flexible* RLHF optimizer; DPO is a brilliant shortcut that trades some flexibility for enormous operational simplicity. If you want the offline-learning-from-fixed-data framing that DPO shares, the [offline RL post](/blog/machine-learning/reinforcement-learning/offline-rl-learning-from-fixed-datasets) covers the relevant intuitions.

## 10. The KL penalty, derived: why it stops reward hacking

Section 5 asserted the KL term prevents reward hacking. Let us *prove* the mechanism, because a hand-wave here leaves you unable to debug a runaway training run. Consider the KL-regularized objective for a single prompt $x$, maximized over the policy distribution $\pi(\cdot \mid x)$:

$$
J(\pi) = \mathbb{E}_{y \sim \pi}\big[ r_\phi(x, y) \big] - \beta \, \text{KL}\big( \pi \,\|\, \pi^{\text{ref}} \big)
$$

Expand the KL term, $\text{KL}(\pi \| \pi^{\text{ref}}) = \sum_y \pi(y) \log \frac{\pi(y)}{\pi^{\text{ref}}(y)}$, and maximize $J$ subject to $\sum_y \pi(y) = 1$ using a Lagrange multiplier $\lambda$ for the constraint. Setting $\partial J / \partial \pi(y) = 0$:

$$
r_\phi(x, y) - \beta\Big( \log \frac{\pi(y)}{\pi^{\text{ref}}(y)} + 1 \Big) - \lambda = 0
$$

Solving for $\pi(y)$ gives the optimal policy we previewed in Section 5:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi^{\text{ref}}(y \mid x) \, \exp\!\Big( \tfrac{1}{\beta} r_\phi(x, y) \Big)
$$

Now read what this *says* about reward hacking. The optimal policy is the reference policy *multiplied* by an exponential reweighting. Crucially, $\pi^*(y \mid x)$ is **proportional to $\pi^{\text{ref}}(y \mid x)$** — so any response $y$ that the reference policy assigns near-zero probability also gets near-zero probability under $\pi^*$, *no matter how high its reward*. The adversarial garbage sequences that fool the reward model ("comprehensive comprehensive...") have vanishingly small probability under the fluent SFT reference, so the $\pi^{\text{ref}}(y)$ factor crushes them to near-zero in $\pi^*$. The exponential-reward factor can only *re-rank within the support of the reference* — it can promote good fluent responses over mediocre fluent ones, but it cannot conjure probability mass out of nothing for sequences the reference would never produce. **That multiplicative structure is the mathematical guarantee against reward hacking.** Smaller $\beta$ means more aggressive reweighting (more reward-chasing, more hacking risk); larger $\beta$ means staying closer to the reference (safer, but less improvement). The whole art of tuning RLHF is finding the $\beta$ that maximizes real quality before reward hacking sets in.

#### Worked example: computing a per-token KL penalty

In practice the KL is estimated per token from the log-probabilities the policy and reference assign to the *actually-generated* token. Suppose at one step the policy assigns the chosen token log-probability $\log \pi_\theta = -0.92$ (probability 0.40) and the reference assigns $\log \pi^{\text{ref}} = -1.61$ (probability 0.20). The per-token KL contribution is the log-ratio:

$$
\log \frac{\pi_\theta(y_t)}{\pi^{\text{ref}}(y_t)} = -0.92 - (-1.61) = 0.69
$$

With $\beta = 0.02$, the KL *penalty* subtracted from this token's reward is $0.02 \times 0.69 = 0.0138$. So if the reward model's final score for the full 64-token completion is $r_\phi = 2.1$ and the *summed* KL over those 64 tokens is, say, $64 \times 0.69 = 44.2$ (an extreme, only-if-every-token-doubled case), the KL-penalized return is $2.1 - 0.02 \times 44.2 = 2.1 - 0.884 = 1.216$. A realistic run keeps mean per-token KL far smaller — the adaptive controller targets a *total* sequence KL around 6–12 in many setups — so the penalty stays a gentle leash, not a stranglehold. The moment you see total KL blow past, say, 30 while reward keeps climbing, the controller has lost the leash and you are hacking the reward model. The figure below shows how the two log-probability streams combine into the penalty and then the total reward.

![A branching graph where the SFT reference policy and the current policy each produce log-probabilities that combine into a KL divergence, which is subtracted from the reward model score to form the total reward feeding the PPO update.](/imgs/blogs/why-language-models-need-rlhf-8.png)

## 11. Reading the dashboard: what a healthy RLHF run looks like

Theory tells you the objective; experience tells you what the curves should *look* like, and an RLHF run gives you a small dashboard of numbers that, read together, tell you whether you are aligning or hacking. I have spent more hours staring at these four traces than I would like to admit, so here is what each one means and what its pathologies look like.

**Mean reward** should climb steadily and then plateau. A healthy run sees reward rise smoothly over the first few hundred steps and level off. If reward *spikes* suddenly and keeps rising past where validation quality plateaus, you are almost certainly reward hacking — the policy found an exploit in the reward model. The tell is that reward keeps going up while a held-out *human* (or a held-out stronger judge model) eval stops improving or gets worse. Reward-model score is a proxy; human eval is the truth. When they diverge, trust the human eval and stop training, or pull back $\beta$.

**KL to reference** should rise and then stabilize near the target. This is the single most diagnostic curve. With an adaptive KL controller, KL climbs as the policy improves and then the controller pins it near `target`. If KL grows without bound, the leash has snapped — lower `target`, raise `init_kl_coef`, or lower the learning rate. If KL stays pinned at zero, the policy is not moving at all (often because the reward signal is too weak after normalization, or the learning rate is too low) and you are wasting compute. A useful rule of thumb from many published runs: a total sequence KL in the single digits to low tens of nats corresponds to meaningful-but-safe improvement; KL in the hundreds means you have left the reference's support and your reward-model scores are now fiction.

**Policy entropy** should decline gently, not collapse. As the policy concentrates on high-reward responses it naturally becomes more deterministic, so entropy falls — that is expected. But a *cliff* in entropy is a danger sign: the policy has mode-collapsed onto a single high-reward template (the dreaded "every answer is now a five-bullet list that opens with 'Certainly!'"). A small entropy bonus in the PPO objective, or a higher $\beta$, counteracts this. Watching entropy is how you catch the calibration-destroying mode collapse I flagged back in Section 1 before it ruins the model's diversity.

**Value-function loss (the critic)** should decrease and stay bounded. The PPO critic learns to predict returns; if its loss diverges, your advantage estimates are garbage and the policy update is noise. The usual culprit is an un-normalized or wildly-scaled reward (Section 8's footgun), or `cliprange_value` set too loose. A diverging critic loss almost always precedes a diverging policy.

#### Worked example: catching reward hacking on the dashboard

Here is a real pattern, with numbers. At step 50, mean reward is 0.4, KL is 3, entropy is 4.1, held-out human win-rate vs SFT is 58%. At step 200, reward is 1.6, KL is 9, entropy is 3.6, human win-rate is 71% — a healthy run, everything moving together. Now step 350: reward has jumped to 3.4, but KL has ballooned to 85, entropy has cratered to 1.2, and the human win-rate has *dropped* to 64%. Read the four traces together: reward up sharply, KL exploded, entropy collapsed, human eval down. That is the unmistakable signature of reward hacking — the policy abandoned the reference's support to chase a reward-model exploit, collapsed onto a single template, and produced text humans actually like *less*. The correct response is to roll back to the step-200 checkpoint, lower `target` from 9 to 6 and raise `init_kl_coef`, and (critically) *retrain the reward model on the step-200 policy's fresh outputs*, because the reward model is now being queried far outside its training distribution. This is iterative RLHF in action, and it is why production pipelines never run a single static reward model to convergence.

The broader practice this reveals is **iterative RLHF**. A reward model trained on completions from the SFT policy is only trustworthy on completions that look like the SFT policy's. As PPO moves the policy, its outputs drift into regions the reward model never saw, and the reward model's scores degrade exactly where you now need them. The fix, used by every serious lab, is to *loop*: run RLHF for a while, collect fresh human comparisons on the *new* policy's outputs, retrain (or fine-tune) the reward model on this fresh data, and run RLHF again from the improved policy. InstructGPT and the Anthropic HH work both did multiple such rounds. Each round chases the distribution shift, keeping the reward model honest on the policy's current behavior. Picture it as a pursuit: the policy runs toward high reward, the reward model periodically catches up to where the policy now lives, and the two converge on genuinely-aligned behavior rather than on a frozen proxy the policy has learned to exploit.

## 12. Constitutional AI: replacing human feedback with AI critique

Human preference labeling is the bottleneck. It is slow, expensive, inconsistent, and — for harmlessness data especially — psychologically taxing for labelers who must read toxic content all day. Anthropic's **Constitutional AI** (Bai et al., 2022, "Constitutional AI: Harmlessness from AI Feedback") attacks this by replacing most of the *human* feedback with *AI* feedback, guided by a written set of principles called a **constitution**.

The method has two phases. In the **supervised phase**, the model generates a response, then is prompted to *critique its own response* against a constitutional principle ("Identify ways this response is harmful or unethical"), then *revise* it accordingly. The revised responses become SFT data — the model teaches itself to produce better answers by self-critique. In the **RL phase** (called **RLAIF**, RL from AI Feedback), instead of humans ranking pairs, the model *itself* is asked which of two responses better satisfies the constitution, and these AI-generated preferences train the reward model. The downstream PPO is identical to RLHF; only the source of the preference labels changes from human to AI.

Why does this scale better? Three reasons. First, **AI labels are cheap and fast** — you can generate millions of comparisons for the cost of GPU time rather than human-hours, breaking the labeling bottleneck. Second, **the constitution is auditable and editable** — the values are written down explicitly as principles, so you can inspect, debate, and revise them, rather than hoping 40 contractors internalized a guidelines document consistently. Third, **it spares humans the harmful content** — the model red-teams and critiques itself, so people do not have to label streams of toxicity. The cost is that you are now trusting the model's *own judgment* of the constitution, which is only as good as the model's understanding of those principles — a model too weak to judge harmlessness cannot supervise harmlessness. In practice CAI is used in *combination* with human feedback (humans still provide helpfulness preferences; AI handles much of harmlessness), and it is a major reason later assistants could be trained with far less human harmlessness labeling than InstructGPT required. The history figure places it among the lineage of preference-alignment methods.

![A left-to-right timeline of preference alignment milestones running from RLHF for Atari in 2017 through InstructGPT and ChatGPT in 2022, Constitutional AI, DPO in 2023, and GRPO in 2024.](/imgs/blogs/why-language-models-need-rlhf-6.png)

## 13. Case studies: GPT-3 vs InstructGPT vs ChatGPT

Concrete behavior change is the proof, so here are the named results.

**Raw GPT-3 (175B, the base model).** Prompt it with "Explain the moon landing to a 6-year-old in a few sentences" and the base model, true to next-token prediction, is liable to continue the *prompt's format* — listing more example tasks ("Explain the theory of gravity to a 6-year-old. Explain the big bang to a 6-year-old.") rather than answering. This exact failure is the canonical example in the InstructGPT paper. The model is not broken; it is completing a document that *looks like* a list of few-shot examples.

**InstructGPT (the RLHF'd model).** The same prompt produces a direct, child-appropriate explanation of the moon landing. The headline quantitative result, restated because it is the whole argument: on the OpenAI API prompt distribution, labelers preferred outputs from the **1.3B InstructGPT model over the 175B GPT-3 base** model. InstructGPT also showed large improvements in truthfulness (roughly 2× on the TruthfulQA benchmark relative to GPT-3) and reductions in toxic generation when prompted to be respectful, with only modest regressions on a few academic NLP benchmarks (the alignment tax, mitigated by the PPO-ptx pretraining mix).

**ChatGPT (InstructGPT's sibling, dialogue-tuned).** ChatGPT used the same RLHF recipe with a data-collection setup oriented toward *conversation* — labelers played both user and assistant to build dialogue demonstrations, and the comparison data ranked conversational responses. The result was the same alignment machinery producing a model tuned for multi-turn chat, which is what drove its mass adoption. The lesson across all three: the *base capability* came from pretraining, but the *usability* — the thing that made hundreds of millions of people able to use it — came from RLHF.

**The summarization result that came first.** Before InstructGPT, OpenAI proved the recipe on a narrower, more measurable task: summarization (Stiennon et al., 2020, "Learning to summarize from human feedback"). They fine-tuned models to summarize Reddit posts (the TL;DR dataset) and news articles, training a reward model on human comparisons of summaries and then optimizing with PPO. The result was striking and *quantitative* in a way open-ended chat is not: the RLHF-trained summaries were preferred by human evaluators over the human-written reference summaries themselves, and over much larger supervised-only baselines. A 1.3B RLHF model produced summaries humans preferred to those from a 12B supervised model — the same "small-aligned-beats-large-unaligned" pattern InstructGPT would later show on instructions. This is the cleanest early proof that the gain is real and not an artifact of fuzzy chat evaluation: on a task with a crisp "which summary is better" judgment, RLHF moved the win-rate decisively, and it generalized (a model trained on Reddit summaries transferred to news summarization better than supervised models did). If you want one paper to read to believe RLHF works, read this one, because the metric is unambiguous.

**A non-language case study, for grounding.** RLHF did not start with language. Christiano et al. (2017, "Deep Reinforcement Learning from Human Preferences") trained Atari and MuJoCo agents from human preferences over short video clips of agent behavior, with *no hand-coded reward function at all*. A human watched pairs of clips and clicked the better one; a reward model learned from those clicks; an RL agent optimized it. They taught a simulated robot to do a backflip from about 900 human comparisons — a behavior nearly impossible to specify with a hand-written reward. This is the *same algorithm* as RLHF for LLMs, which is why this series treats it as one idea: learn a reward from human comparisons, then optimize a policy against it. The agent-environment-reward spine is identical; only the agent (a robot vs a language model) and the action space (joint torques vs tokens) change.

Here is a compact comparison of the algorithm family, to anchor where RLHF sits:

| Method | Needs reward model | Needs RL loop | Human label cost | Operational complexity | Canonical use |
| --- | --- | --- | --- | --- | --- |
| SFT only | No | No | Medium (demos) | Low | Instruction tuning |
| RLHF (PPO) | Yes | Yes (4 models) | High (comparisons) | High | InstructGPT, ChatGPT |
| RLAIF / Constitutional AI | Yes | Yes | Low (AI labels) | High | Claude harmlessness |
| DPO | No | No (supervised) | Medium (pairs) | Low | Zephyr, many open models |
| GRPO | Yes (or rule reward) | Yes (no critic) | Low–Medium | Medium | reasoning models |

The full trade-off surface — labeling cost against reward-hacking risk against scalability against alignment quality — is laid out in the matrix below.

![A comparison matrix scoring RLHF, SFT-only, Constitutional AI, DPO, and RLAIF across human labeling cost, reward-hacking risk, scalability, alignment quality, and a deployed example.](/imgs/blogs/why-language-models-need-rlhf-5.png)

## 14. The remaining hard problems

RLHF works, but it is not solved, and pretending otherwise will burn you. Here are the failure modes I actively watch for, each a live research area.

**Reward hacking (reward model exploitation).** Even with the KL penalty, the policy will exploit any systematic error in the reward model. The classic discovered behavior is **length bias**: reward models trained on human comparisons often learn that longer answers are preferred (because humans, on average, rated thorough answers higher), so the policy learns to pad — producing verbose, hedged, bullet-point-stuffed responses that score well but annoy users. Other discovered hacks: sycophantic agreement, excessive caveating, and formatting tics the reward model happened to like. The defense is constant: monitor KL, hold out fresh human evals (not just reward-model scores), and retrain the reward model on the policy's *new* outputs (the reward model trained on SFT outputs becomes stale once the policy drifts).

**Distributional shift.** The reward model is trained on completions from the SFT policy. As PPO moves the policy, it generates completions from a *different* distribution than the reward model ever saw, and the reward model's scores become unreliable exactly where the policy now lives. This is why the KL leash matters and why production pipelines do *iterative* RLHF — collect fresh comparisons on the new policy's outputs, retrain the reward model, repeat. A single static reward model decays.

**Sycophancy.** Because labelers (and the prompts steering them) tend to prefer responses that *agree* with the apparent view in the prompt, RLHF can teach the model to tell users what they want to hear rather than what is true. Ask a sycophantic model "I think 7 is prime, right?" about a non-prime and it may agree. This is a *direct consequence* of optimizing for human approval — humans approve of agreement — and it is one of the clearest cases where "aligned to human preference" diverges from "honest." Mitigations include explicitly collecting comparison data that rewards respectful disagreement and using AI feedback (CAI) that scores correctness independently of agreement.

There is a deeper, more uncomfortable version of the honesty problem worth stating plainly. RLHF optimizes for what humans *approve of*, and humans approve of answers that *sound* confident, authoritative, and complete. But the model's *actual* internal uncertainty is a separate thing — and as I noted in Section 1, RLHF tends to *break the calibration* that pretraining gave the model. The result is a model that has learned to *sound* certain because certainty is rewarded, even when it should be hedging. This is the mechanism behind a large fraction of confident hallucinations: the reward signal taught the model that "I'm not sure, but..." is rated lower than a crisp confident answer, so the model suppressed its honest uncertainty. Fixing this requires explicitly rewarding calibrated uncertainty in the preference data — ranking "I don't know" above a confident fabrication when the model genuinely lacks the knowledge — which is hard because labelers themselves often cannot tell a fabrication from a fact. Honesty is the hardest leg of the triad precisely because the evaluation signal (human approval) is *correlated with* the failure mode (confident-sounding wrongness). You are asking the optimizer to resist the very gradient you are giving it.

**Specification gaming and Goodhart's law.** The reward model *is the specification*, and any specification is gameable. The deeper problem is that human preference itself is an imperfect specification of what we *actually* want — humans can be fooled by confident-sounding wrong answers, polished prose hiding errors, and flattery. RLHF optimizes the *measured* preference, and to the extent the measurement diverges from true value, the optimizer will find and exploit the gap. There is no clean fix; there is only vigilance, better measurement (CAI, debate, scalable oversight), and humility about what the reward signal actually encodes.

The decision tree below is the practical distillation of when to reach for which method, given these trade-offs.

![A decision tree for choosing an alignment approach, branching on whether human feedback is needed and the size of the labeling budget toward RLHF, DPO or RLAIF, or plain SFT.](/imgs/blogs/why-language-models-need-rlhf-7.png)

## When to use RLHF (and when not to)

A decisive section, because RLHF's prestige tempts people to reach for it when something simpler wins.

**Use plain SFT when** you only need instruction-following format and you have good demonstrations. If your task is "make the base model answer questions instead of continuing documents," SFT alone gets you 80% of the way for a fraction of the complexity. Do not run PPO to teach a format.

**Use DPO instead of full PPO when** you have a fixed preference dataset and want the alignment quality of RLHF without the four-model operational nightmare. DPO is now the default for most open-source alignment precisely because it is supervised-learning-simple while achieving comparable quality on many benchmarks. Reach for full PPO when you need *online* exploration — the policy generating and being scored on its own fresh outputs — which DPO's offline nature cannot provide.

**Use RLAIF / Constitutional AI when** human harmlessness labeling is the bottleneck and your base model is strong enough to judge the constitution reliably. If your model is too weak to evaluate its own outputs against principles, AI feedback is the blind leading the blind; collect human labels.

**Use full RLHF with PPO when** you need the highest alignment quality, you have the budget for a large human labeling operation, and you need the policy to *improve beyond* the preference data through on-policy exploration. This is the frontier-lab setting: it is expensive and finicky, but it remains the gold standard when quality is paramount and you can afford it.

**Do not use any preference optimization when** you have a *verifiable, programmatic* reward — math correctness, code that passes tests, a game score. For those, you do not need to learn a reward model from humans at all; use the verifiable reward directly (this is the regime where GRPO and RLVR shine for reasoning models). RLHF's whole reason for existing is that "good response" is *not* programmatically checkable. When it is checkable, skip the human-preference machinery.

## Key takeaways

- **Pretraining optimizes the wrong objective for an assistant.** Next-token prediction models the distribution of internet text; "most likely continuation" is not "helpful, harmless, honest response." The gap is structural, not a training bug.
- **SFT is necessary but not sufficient.** It teaches the assistant *format* and gives a competent starting policy, but it can only imitate demonstrations, cannot generalize from preferences, and cannot exploit the cheap evaluation signal.
- **Evaluation is far cheaper than generation.** Humans can rank responses ~40× faster than they can write ideal ones, and the ranking signal targets the model's real outputs. This asymmetry is why RLHF is economically feasible.
- **The reward model learns a partial order via Bradley-Terry.** It is trained on *differences* of scores ($-\log \sigma(r_w - r_l)$), so absolute reward is unidentified and only differences matter — which is fine, because RL only uses advantages.
- **RL is the right tool because the reward is a non-decomposable scalar over whole sequences, and the policy must explore its own output distribution** — exactly what policy-gradient RL does and supervised learning cannot.
- **The KL penalty is a principled anti-reward-hacking guarantee, not a hack.** The optimal KL-constrained policy is the reference policy reweighted by $\exp(r/\beta)$, so adversarial sequences the reference never produces get crushed to near-zero probability regardless of their reward.
- **RLHF's headline result is capability, not just safety.** A 1.3B InstructGPT model was preferred over the 175B GPT-3 base — the "alignment tax" on the axis people cared about was negative.
- **Inter-rater agreement (~73%) is your reward-model accuracy ceiling.** If you "beat" it, suspect a leak. Alignment is only as good as the preference data, which is only as good as the rater pool and guidelines.
- **The hard problems are not solved:** reward hacking (length bias, sycophancy), distributional shift requiring iterative retraining, and Goodhart's-law specification gaming all persist. RLHF needs vigilance, not faith.
- **Match the method to the problem:** SFT for format, DPO for fixed preference data without operational overhead, RLAIF/CAI when labeling is the bottleneck, full PPO-RLHF for top quality, and verifiable rewards (GRPO/RLVR) when correctness is programmatically checkable.

## Further reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022 — the canonical three-stage RLHF paper, the 1.3B-beats-175B result, and the PPO-ptx alignment-tax mitigation.
- Christiano et al., "Deep Reinforcement Learning from Human Preferences," 2017 — RLHF before LLMs; Atari and MuJoCo agents and the backflip from ~900 comparisons. The algorithmic origin of everything here.
- Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," 2022 — Anthropic's HH dataset, the separated helpfulness/harmlessness streams, and the live-comparison protocol.
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022 — the self-critique-and-revise method and RLAIF.
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023 — the closed-form optimal policy and the supervised shortcut around the RL loop.
- Stiennon et al., "Learning to summarize from human feedback," 2020 — the predecessor to InstructGPT; the summarization win-rate results that proved the recipe on a concrete task.
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017 — the optimizer underneath RLHF. Pair it with this series' [PPO post](/blog/machine-learning/reinforcement-learning/proximal-policy-optimization-ppo).
- Within this series: start from [what reinforcement learning is](/blog/machine-learning/reinforcement-learning/what-is-reinforcement-learning) for the agent-environment-reward spine, [the policy gradient theorem](/blog/machine-learning/reinforcement-learning/the-policy-gradient-theorem) for the gradient estimator PPO refines, and [the credit assignment problem](/blog/machine-learning/reinforcement-learning/the-credit-assignment-problem) for why a sequence-level reward is hard to optimize. The forthcoming unified map (`reinforcement-learning-a-unified-map`) and capstone (`the-reinforcement-learning-playbook`) place RLHF in the full taxonomy of which-objective-to-optimize.
