---
title: "Reward Modeling: Teaching an AI to Judge Quality"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Build a reward model from pairwise human preferences: the Bradley-Terry loss derived from scratch, a full PyTorch training loop, and the length-bias, sycophancy, and reward-hacking failures that quietly wreck RLHF."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "reward-modeling",
    "llm-alignment",
    "machine-learning",
    "pytorch",
    "bradley-terry",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/reward-modeling-from-human-preferences-1.png"
---

Here is a problem that breaks the standard reinforcement learning recipe. You want to fine-tune a language model so its answers are *helpful*. In ordinary RL you would write down a reward function, drop the agent into the environment, and let it optimize. But what is the reward for "helpful"? Ask ten engineers to put a number on a paragraph of text and you get ten different numbers, none of them stable across a coffee break. There is no thermometer for helpfulness. The reward signal that every policy-gradient method assumes simply does not exist.

What *does* exist is something humans are surprisingly good at: comparison. Show a person two answers to the same question and ask "which is better?" and they will answer quickly, confidently, and with reasonable consistency. They cannot tell you that answer A is worth 7.3 and answer B is worth 4.1, but they can reliably tell you A beats B. The entire field of reinforcement learning from human feedback (RLHF) is built on monetizing that one human ability. We collect thousands of pairwise comparisons, train a model to predict them, and that model — the **reward model** — becomes the reward function that the absent thermometer never gave us.

This post is about that reward model: the small, unglamorous network that sits between human judgment and the policy optimizer. It is the most important component in the RLHF stack and the one most likely to quietly ruin your training run. A weak reward model caps how good your policy can get. A miscalibrated one teaches your policy to write longer, more sycophantic, more confidently-wrong answers because that is literally what the reward says to do. Figure 1 shows the data-collection loop we will build on: one prompt, two sampled responses, one human comparison, one stored triple.

![Diagram of collecting pairwise preference data where one prompt is sampled into two responses, compared by a human, and stored as a chosen and rejected triple](/imgs/blogs/reward-modeling-from-human-preferences-1.png)

By the end you will be able to derive the Bradley-Terry preference model from first principles, write a complete reward-model training loop in PyTorch, evaluate it honestly on held-out pairs, recognize the three classic failure modes before they wreck a run, and decide when to reach for a process reward model, an ensemble, a Constitutional-AI critique pipeline, or to skip the reward model entirely with DPO. We will keep returning to the series spine: the RL loop is an agent interacting with an environment to collect rewards and update a policy. Reward modeling is the answer to the question *where does the reward come from when no one can write it down*. For the bigger picture of how this connects to PPO and policy optimization, see the unified map at `/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map` and the capstone at `/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook`.

## 1. Why a reward model at all

Let me make the absence of a reward function concrete, because it is easy to nod past. Suppose you are fine-tuning a 7-billion-parameter chat model. The policy $\pi_\theta(y \mid x)$ produces a response $y$ given a prompt $x$. To run any policy-gradient update you need a scalar reward $r(x, y)$ for each rollout. In a game like CartPole the reward is handed to you by the environment: +1 per timestep the pole stays up. In a robotics sim it is engineered: distance traveled minus energy used minus a fall penalty. In both cases someone could write `reward = ...` as a line of code.

For "write a helpful, honest, harmless answer," nobody can write that line. The properties we care about are diffuse, context-dependent, and contested. Worse, even if a single expert could score one response in isolation, their scores would not be **commensurable** across prompts: a 6/10 answer to a coding question and a 6/10 answer to a poetry request are not the same kind of 6. Asking humans for absolute scalar scores produces noisy, drifting, incomparable labels. Multiple studies of annotation quality have found that direct numeric rating has much lower inter-annotator agreement than pairwise comparison on the same content.

So we change the question. Instead of "score this," we ask "given these two, which is better?" Pairwise comparison is robust for three reasons. First, it cancels the calibration problem: the annotator does not need a global scale, only a local verdict. Second, it is fast — a binary choice takes a few seconds versus the agonizing of a numeric rating. Third, comparisons compose: if we have enough of them we can recover a consistent global ordering, which is exactly what the reward model learns to do.

The reward model is the bridge. It is a function $r_\phi(x, y) \rightarrow \mathbb{R}$, parameterized by $\phi$, trained so that whenever a human preferred response $y_w$ ("w" for *win*) over $y_l$ ("l" for *lose*) on prompt $x$, the model assigns $r_\phi(x, y_w) > r_\phi(x, y_l)$. Once trained, the reward model is a cheap, automatic, *differentiable-at-the-output* stand-in for the human. We can query it millions of times during PPO without paying an annotator for each rollout. That is the whole trick: we spend human effort once, in bulk, on comparisons; we amortize it forever through a learned scalar function.

This separation also explains a subtle architectural point we will return to. The reward model does not need to *generate* anything. It only needs to *judge*. That means it can be a comparatively small head bolted onto a frozen-ish encoder, and it means its failure modes are the failure modes of a *judge* — bias, miscalibration, being fooled by surface features — not the failure modes of a generator.

It is worth being precise about where the reward model sits in the full RLHF pipeline, because newcomers often conflate three stages that are deliberately separate. Stage one is **supervised fine-tuning** (SFT): you take a pretrained base model and fine-tune it on high-quality demonstrations of the behavior you want, so it produces plausible instruction-following responses. Stage two is **reward modeling**, the subject of this post: you train a separate model to predict human preferences over the SFT model's outputs. Stage three is **policy optimization** (usually PPO): you fine-tune the SFT model further, using the reward model as the reward function and a KL penalty to keep the policy close to the SFT reference. The reward model is the *output* of stage two and the *reward function* of stage three. It is frozen during stage three — and that frozen-ness is the source of the over-optimization problem we will dissect in Section 8. Hold onto this three-stage frame; nearly every confusion about "where does the reward come from" dissolves once you place the reward model correctly between the SFT model and PPO.

There is one more reason the reward model deserves its own post rather than a paragraph inside an RLHF overview: it is the component where the *values* of the system are actually encoded. The base model encodes knowledge; the SFT model encodes format; but the reward model encodes *what counts as better*. Every judgment call your annotators made — is a hedged correct answer better than a confident wrong one, is brevity a virtue, is refusing a borderline request the right move — is compressed into the reward model's weights and then amplified by PPO across millions of rollouts. Get the reward model wrong and you do not get a slightly-worse policy; you get a policy that confidently optimizes the wrong thing. That asymmetry of consequences is why practitioners spend disproportionate care here.

#### Worked example: why scalar scores fail and comparisons don't

Suppose three annotators rate two summaries of the same article on a 1–10 scale. Annotator 1 is generous (gives 7 and 9), annotator 2 is harsh (3 and 5), annotator 3 is moderate (5 and 7). If you average the raw scores you get summary A at $(7+3+5)/3 = 5.0$ and summary B at $(9+5+7)/3 = 7.0$. That looks like a clean two-point gap, but notice the spread: annotator 1's *low* score (7) is above annotator 2's *high* score (5). The absolute numbers are nearly meaningless across raters.

Now look at the *comparisons*. Every annotator put B above A: $9 > 7$, $5 > 3$, $7 > 5$. Three for three. The pairwise signal is unanimous and noise-free even though the scalar signal is a mess. This is precisely why we collect preferences and not ratings — and why the reward model is trained on the *gap* between two scores rather than on either score alone.

## 2. Collecting preference data

A preference dataset is a list of triples $(x, y_w, y_l)$: a prompt, the chosen response, and the rejected response. Building one well is half the battle, and the half that gets the least respect. Let me walk through how it actually happens in a production pipeline.

You start with a pool of prompts. For an instruction-following model these come from real usage logs, from curated task sets, and from synthetic generation. For each prompt you sample two (sometimes four, in which case you compare pairs within the set) completions from a *supervised fine-tuned* (SFT) model — the policy after instruction tuning but before RLHF. Sampling from the SFT model, not from the base model, matters: you want the comparisons to be over responses that are roughly in the distribution your policy will explore, otherwise the reward model learns to discriminate "fluent vs gibberish" instead of "good vs slightly-less-good," which is the discrimination that actually matters during PPO.

Then a human annotator sees the prompt and the two responses and picks the better one, usually along a rubric (helpfulness, honesty, harmlessness, or task-specific criteria). Often they also record a *strength* — "A is much better" vs "A is slightly better" — which you can fold into the loss as a margin, though many pipelines discard it. The output is the triple.

Two real datasets anchor this. **Anthropic's HH-RLHF** dataset contains roughly 170,000 human-preference comparisons split across a "helpfulness" set and a "harmlessness" set, each entry a `chosen`/`rejected` pair of multi-turn dialogues. **OpenAI's InstructGPT** comparison data, described in Ouyang et al. (2022), had labelers rank between four and nine responses per prompt, which the authors then expand into all pairwise comparisons within each ranking — a ranking of $K$ responses yields $\binom{K}{2}$ pairs. That expansion is a quietly important data-efficiency trick: one labeling session of ranking 9 responses gives you 36 training pairs.

Here is the shape of the data in code, using a Hugging Face `datasets` view that mirrors how TRL expects it:

```python
from datasets import load_dataset

# Anthropic HH-RLHF: each row has 'chosen' and 'rejected' full conversations.
ds = load_dataset("Anthropic/hh-rlhf", split="train")
ex = ds[0]
print(ex["chosen"][:120])
print("----")
print(ex["rejected"][:120])

# TRL's RewardTrainer wants columns named exactly like this:
def to_pairwise(example):
    return {
        "chosen": example["chosen"],     # the preferred completion (string)
        "rejected": example["rejected"], # the dispreferred completion (string)
    }

ds = ds.map(to_pairwise)
print(ds.column_names)  # ['chosen', 'rejected']
```

### Where inter-annotator agreement breaks down

Preference data is *not* clean. Inter-annotator agreement on these tasks typically lands in the 60–75% range — meaning that on a quarter to a third of pairs, two reasonable humans disagree about which response is better. The InstructGPT paper reported labeler-labeler agreement around 73% on their comparisons, and researcher-labeler agreement of a similar magnitude. This is the irreducible noise floor of your reward model: it cannot be more accurate than the humans are consistent.

Agreement collapses in predictable places. It collapses when both responses are good (splitting hairs), when the rubric is underspecified ("which is more *creative*?"), when the prompt is contentious (politics, ethics), and when responses are long enough that annotators skim. It also drifts over time and across annotator pools — a model trained on data from one labeling vendor can behave differently from one trained on another, even with identical instructions. The practical consequence: treat your preference labels as a *noisy* signal, regularize accordingly (Section 5), and never expect held-out accuracy above your annotator agreement ceiling. If your reward model reports 95% accuracy and your humans agree only 70% of the time, you are not measuring quality — you are measuring a leak.

This noise floor has a quantitative consequence that is easy to derive and worth internalizing. Suppose the *true* probability a typical human prefers the better response is $p$ (say 0.73, matching InstructGPT). Then even a *perfect* reward model — one that has recovered the exact latent preference distribution — will only agree with a fresh human label $p$ of the time on near-tie pairs, because the human themselves is stochastic. The Bayes-optimal pairwise accuracy is bounded by the human consistency. When you see a reward model plateau at 70–72% held-out accuracy, the correct reaction is usually not "train harder" but "the labels are this noisy; collect cleaner data or accept the ceiling." This is the single most common misread of reward-model evaluation: treating an irreducible noise floor as a model deficiency and overfitting in pursuit of an accuracy that does not exist.

There is also a structural decision in *how* you present pairs to annotators that shapes the data more than people expect. Presenting two responses side by side invites *relative* judgment, which is what you want, but it also imports *order effects* (annotators slightly favor the first option) and *contrast effects* (a response looks better next to a weak one than next to a strong one). Good pipelines randomize presentation order, occasionally insert gold pairs with a known answer to measure annotator quality, and discard annotators whose agreement with the gold set falls below a threshold. None of this is glamorous, but the reward model can only be as good as the data, and the data quality is decided here — at the annotation interface — not in the training loop. A team that spends a month on annotation guidelines and quality control routinely beats a team that spends that month on reward-model architecture.

### How much data, and of what kind

A practical question: how many preference pairs do you actually need? The empirical answer from the public literature is "fewer than you'd guess to get started, more than you'd like to get great." InstructGPT trained usable reward models on the order of tens of thousands of comparisons. Anthropic's HH-RLHF is ~170k. Modern open reward models often train on a few hundred thousand to low millions of mixed pairs. The marginal value of data is highest at the start (going from 1k to 10k pairs is transformative) and flattens (going from 500k to 1M is a modest accuracy bump), consistent with the log-linear scaling we discuss in Section 8.

Just as important as *quantity* is *coverage*. The reward model is only trustworthy on the distribution of responses it was trained to judge. If your preference data was collected on responses from an early SFT model, but PPO pushes the policy into a new region of response-space, the reward model is being queried out of distribution — and out of distribution is exactly where its errors live and where the policy will gleefully exploit them. This is why high-quality pipelines *re-collect* preference data on the *current* policy's outputs periodically (the iterative RLHF loop: SFT, RM, PPO, re-sample, re-label, retrain RM, repeat). A reward model trained once and used forever steadily decays in usefulness as the policy walks away from the distribution it was trained on.

## 3. The Bradley-Terry model

Now the theory, which is more elegant than it looks. We need a probabilistic model that connects a *latent quality score* to an *observed comparison*. The Bradley-Terry model, introduced in 1952 for ranking competitors from paired contests, is exactly that and it is the right tool here for a reason we can derive.

Assume each response $y$ has a latent "true" reward $r(x, y)$. We want a model for the probability that a human prefers $y_a$ over $y_b$. The Bradley-Terry assumption is that this probability depends only on the *difference* of the rewards through a logistic link:

$$P(y_a \succ y_b \mid x) = \frac{\exp\big(r(x, y_a)\big)}{\exp\big(r(x, y_a)\big) + \exp\big(r(x, y_b)\big)}.$$

Divide top and bottom by $\exp(r(x, y_a))$ and you get the cleaner form:

$$P(y_a \succ y_b \mid x) = \frac{1}{1 + \exp\big(-(r(x, y_a) - r(x, y_b))\big)} = \sigma\big(r(x, y_a) - r(x, y_b)\big),$$

where $\sigma$ is the logistic sigmoid. So the probability of preferring A over B is the sigmoid of the *reward gap*. When the gap is zero the probability is exactly 0.5 (a coin flip — the model is indifferent), and as the gap grows the probability saturates toward 1. This is intuitive: the more A out-scores B, the more confidently a human should prefer A.

Why is this the *right* model and not an arbitrary choice? Two reasons. First, it is the maximum-entropy model consistent with the constraint that preference depends only on the reward difference — among all distributions where only the gap matters, the logistic is the least committal. Second, it has a beautiful invariance: the rewards are only identified up to an *additive constant*, because adding the same number to every reward leaves all differences (and hence all probabilities) unchanged. This is exactly the property we want, since the absolute scale of "helpfulness" was meaningless to begin with. The reward model learns a *relative* ordering, not an absolute scale, and Bradley-Terry encodes that directly.

There is a third justification that connects Bradley-Terry to a model of human decision-making, and it is worth a paragraph because it tells you *when the model will break*. Bradley-Terry can be derived from a **random utility model**: assume the human's perceived value of response $y$ is the true reward $r(x,y)$ plus an independent noise term $\epsilon_y$, and the human prefers whichever has higher perceived value. If those noise terms follow a Gumbel (type-I extreme value) distribution, then the probability that $y_a$'s perceived value exceeds $y_b$'s works out *exactly* to the logistic of the reward gap — this is the same algebra that gives the softmax its form. So Bradley-Terry is the preference model you get when humans are *noisy maximizers* of a latent reward with Gumbel noise. That assumption is reasonable for genuine quality judgments. It breaks when preferences are *intransitive* (a human prefers A to B, B to C, and C to A — which happens with multi-dimensional quality where different pairs surface different dimensions), because no single scalar reward can reproduce a cycle. When you see a reward model that cannot exceed chance on a particular slice of data, intransitive preferences are a prime suspect: the scalar model is structurally incapable of fitting them, and no amount of capacity will help.

A small derivation makes the additive-constant invariance concrete and explains a line in the training loop later. Suppose we shift every reward by a constant $c$: $r'(x,y) = r(x,y) + c$. Then the gap is unchanged, $r'(x,y_a) - r'(x,y_b) = r(x,y_a) - r(x,y_b)$, so every preference probability is unchanged, so the loss is *exactly* unchanged. The gradient of the loss with respect to $c$ is therefore identically zero — the optimizer gets no signal about the absolute level of the reward. In practice this means the mean reward can drift arbitrarily during training (often slowly diverging), which is harmless for ranking but inconvenient for downstream PPO, which prefers a roughly-centered reward. That is why we add a tiny penalty pinning the mean reward near zero (the `mean_reg` term in Section 11): it does not change what the reward model *learns* about ordering, it just anchors the otherwise-free constant so the numbers stay sane.

### The reward model loss

To train, we maximize the likelihood of the observed preferences. For a single triple $(x, y_w, y_l)$ where the human chose $y_w$, the likelihood is $P(y_w \succ y_l) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$. Taking the negative log over a dataset $\mathcal{D}$ gives the reward-model loss:

$$\mathcal{L}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big].$$

This single line is the entire training objective for a standard scalar reward model. It is just binary logistic regression where the "feature" is the reward gap and the "label" is always 1 (chosen really did beat rejected). Minimizing it pushes $r_\phi(x, y_w)$ up and $r_\phi(x, y_l)$ down until their gap reproduces the human preferences as well as the data allows.

Figure 2 traces this as a layered stack: a raw response pair becomes two scalar scores, the scores become a log-sigmoid loss, and the gradient sharpens the margin between chosen and rejected.

![Stack diagram of the Bradley-Terry training process from a raw response pair to reward scores to a log-sigmoid loss to a gradient update and a sharper reward model](/imgs/blogs/reward-modeling-from-human-preferences-2.png)

#### Worked example: a Bradley-Terry loss calculation

Take one preference pair. The reward model scores the chosen response at $r_w = 2.4$ and the rejected at $r_l = 0.7$. The gap is $r_w - r_l = 1.7$.

The model's predicted probability that the chosen response is preferred is
$$\sigma(1.7) = \frac{1}{1 + e^{-1.7}} = \frac{1}{1 + 0.1827} = 0.846.$$
So the model is 84.6% sure it ranked them correctly. The loss on this pair is
$$-\log(0.846) = 0.167.$$

Now suppose during training the gap widens to $r_w - r_l = 3.0$. Then $\sigma(3.0) = 0.953$ and the loss drops to $-\log(0.953) = 0.048$. The loss fell because the model became more confident in the correct ordering. But watch the diminishing returns: pushing the gap from 1.7 to 3.0 only cut the loss by about 0.12. The sigmoid saturates, so the gradient on already-correct, already-confident pairs is tiny. The model spends its capacity on the *hard, near-tie* pairs where the gap is small and the gradient is large — which is exactly where it should spend it. If instead the model got the pair *wrong* with $r_w = 0.7, r_l = 2.4$ (gap $-1.7$), the loss would be $-\log(\sigma(-1.7)) = -\log(0.154) = 1.87$, more than ten times larger. Wrong-ordering pairs dominate the gradient. That asymmetry is what drives learning.

## 4. Reward model architecture

The architecture is almost anticlimactic, and that is the point. You take a pretrained language model, throw away its token-prediction head (the big matrix that maps the final hidden state to vocabulary logits), and replace it with a single linear layer that maps the final hidden state to *one scalar*. That scalar, read off the last token's position, is the reward.

Concretely: the model processes the full sequence (prompt concatenated with response). At the final token position you have a hidden vector $h \in \mathbb{R}^d$ (where $d$ is the model dimension, e.g. 4096). The reward head is a weight vector $w \in \mathbb{R}^d$ and the reward is $r_\phi(x, y) = w^\top h$. One dot product. The whole transformer underneath is the feature extractor; the head is a single linear probe.

Why the last token? Because in a causal transformer the final token's hidden state has attended to the entire preceding sequence, so it is the natural place to read a whole-sequence summary. Some implementations instead score the last *non-padding* token (important when sequences are padded to equal length in a batch) — getting this index right is a classic source of silent bugs, where the reward is read off a `<pad>` token and the model learns garbage.

Here is the architecture in PyTorch, building on a Hugging Face base model:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, base_model_name="EleutherAI/pythia-410m"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden = self.backbone.config.hidden_size
        # Single scalar head. bias=False is conventional: BT is shift-invariant,
        # so an additive bias is unidentifiable and just wastes a parameter.
        self.v_head = nn.Linear(hidden, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state            # [B, T, d]
        scores = self.v_head(hidden).squeeze(-1)  # [B, T]
        # Read the reward off the LAST NON-PAD token of each sequence.
        last_idx = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        reward = scores[batch_idx, last_idx]      # [B]
        return reward
```

### Why the same base model as the policy works well

A recurring design choice in RLHF: initialize the reward model from the *same* base model (or the SFT checkpoint) that the policy is initialized from. This is not just convenience. The reward model and the policy then share a representation of language — they have seen the same pretraining, they tokenize the same way, and the reward model's notion of "what this text means" is aligned with the policy's. This shared encoder makes the reward signal *legible* to the policy: when PPO nudges the policy toward higher reward, it is climbing a landscape carved by a model that understands text the same way it does.

There is also a capacity argument. The reward model needs roughly the same representational power as the policy to judge the policy's best outputs. If your policy is a 70B model and your reward model is 410M, the reward model will be unable to tell apart the subtle distinctions a 70B policy can produce — it becomes the bottleneck (Section 8). InstructGPT used a 6B reward model to align a 175B policy, and the authors noted that even larger reward models were unstable to train, so size is not monotonically better, but the reward model should not be *tiny* relative to what it judges.

One practical wrinkle: people often *freeze* most of the backbone and train only the top few layers plus the head, especially with small preference datasets. This is regularization by another name — it prevents the reward model from memorizing the limited preference data through its enormous parameter count. We will see why that matters next.

A subtlety in the architecture that trips people up: should the reward model score the *whole* sequence at the last token, or should it produce a reward at every token and sum them? For a standard scalar (outcome) reward model, the convention is a *single* reward at the last token, representing the quality of the complete response. But during PPO the policy needs a *per-token* reward signal to compute advantages, so the single sequence-level reward is placed on the final token and combined with the per-token KL penalty everywhere else. This asymmetry — dense KL penalty, sparse terminal reward — is part of why RLHF credit assignment is hard, and it is exactly the gap that process reward models (Section 9) close by emitting a reward at every reasoning step rather than only at the end. Keep this distinction clear: the reward *model* emits one scalar per response; the PPO *reward signal* is that scalar on the last token plus a KL term on every token.

Another implementation detail with outsized impact: tokenization and truncation. If you truncate the response to fit a max length and the truncation lands mid-sentence, the reward model scores an incomplete response, and the gradient teaches it nonsense. Worse, if chosen responses are systematically longer than rejected ones (common, since better answers are often more thorough), naive truncation can truncate chosen responses more often than rejected ones, injecting a spurious signal. The fix is to set the max length generously, truncate the *prompt* before the *response* when you must, and verify that your truncation rate is similar across chosen and rejected. Silent length-correlated truncation is a quiet contributor to the length bias we discuss in Section 7.

## 5. Training dynamics and calibration

Reward models overfit *fast*. The preference dataset is small (tens to low-hundreds of thousands of pairs) relative to the model's capacity, and the labels are noisy (Section 2). Left unregularized, the model drives training loss toward zero by memorizing idiosyncrasies of individual pairs — including the *noise* — and its held-out accuracy stalls or degrades. In practice many reward models are trained for a single epoch over the preference data, because a second pass already begins to overfit. Ziegler et al. (2019) and subsequent work repeatedly found the reward model is the most overfit-prone component of the RLHF stack.

Your regularization toolkit: train for one epoch (or early-stop on held-out pair accuracy), use weight decay, optionally freeze lower layers, keep the learning rate small (1e-5 to 1e-6 for the full model), and add a small coefficient that penalizes the *mean* reward drifting away from zero (this stabilizes the otherwise-unidentified additive constant). Some teams add a margin term using the annotator's strength rating. The single most important habit: hold out a slice of preference pairs and watch accuracy on them every few hundred steps. The moment held-out accuracy peaks and starts to fall, stop.

### Calibration: is r = 3.0 really three times better than r = 1.0?

No, and assuming so is a category error. Recall the Bradley-Terry invariance: rewards are identified only up to an additive constant. There is no meaningful "zero" and there is no meaningful "ratio." A reward of 3.0 is not three times better than 1.0; the only thing that means anything is the *gap*, because only gaps enter the sigmoid. The reward model output is an interval scale at best, not a ratio scale.

What *is* meaningful and worth checking is whether the *gap* is calibrated to the *probability* of preference. A reward model is well-calibrated if, among all pairs where it predicts a gap corresponding to $\sigma(\text{gap}) = 0.8$, the chosen response really does win about 80% of the time. You measure this by binning held-out pairs by predicted preference probability and comparing to the empirical win rate in each bin — a reliability diagram. A model that says "90% confident" but is right only 70% of the time is overconfident, and that overconfidence will bite during PPO, because the policy will chase confidently-mispredicted high rewards. Figure 3 contrasts a naive, biased reward model against a calibrated one along three axes.

![Before and after diagram contrasting a naive length-biased sycophantic reward model with a calibrated one that is length-normalized factual and uncertainty-aware](/imgs/blogs/reward-modeling-from-human-preferences-3.png)

Temperature scaling — dividing the reward gap by a learned scalar $T$ before the sigmoid, fit on a held-out set — is the cheapest fix for overconfidence and costs one parameter. It does not change the *ranking* (so it does not affect which response wins), only the *confidence*, which is what PPO's optimization pressure is sensitive to.

## 6. Evaluating reward models

A reward model is itself a model, so evaluate it like one: on held-out data it never trained on. The headline metric is **pairwise accuracy** — the fraction of held-out $(x, y_w, y_l)$ triples where $r_\phi(x, y_w) > r_\phi(x, y_l)$. This directly answers "does the reward model agree with humans on which response is better?" Good reward models on instruction data land in the 65–75% range, which sounds low until you remember the human agreement ceiling is around 70% (Section 2). A reward model at 72% held-out accuracy is essentially at the human noise floor; you cannot do meaningfully better without cleaner labels.

```python
import torch

@torch.no_grad()
def eval_pairwise_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    gaps = []
    for batch in dataloader:
        rw = model(batch["chosen_ids"].to(device),
                   batch["chosen_mask"].to(device))
        rl = model(batch["rejected_ids"].to(device),
                   batch["rejected_mask"].to(device))
        correct += (rw > rl).sum().item()
        total += rw.size(0)
        gaps.append((rw - rl).cpu())
    gaps = torch.cat(gaps)
    return {
        "accuracy": correct / total,
        "mean_gap": gaps.mean().item(),     # average margin chosen over rejected
        "gap_std": gaps.std().item(),       # spread of margins
        "frac_confident": (gaps.abs() > 1.0).float().mean().item(),
    }
```

One nuance about pairwise accuracy that separates careful practitioners from careless ones: *not all pairs are equally informative*, so a single accuracy number can hide the picture. Pairs where the two responses are far apart in quality are easy — any half-trained reward model gets them right — and they inflate accuracy without telling you anything useful. The pairs that matter are the *near-ties*, where the responses are genuinely close and the model must make a fine distinction. A reward model can post 72% overall accuracy while being barely above chance on the hard, near-tie pairs that dominate PPO's exploration once the policy is good. The remedy is to *stratify*: bin held-out pairs by the human-judged margin (or by the reward gap) and report accuracy per stratum. If accuracy is 90% on easy pairs and 55% on hard ones, you know exactly where the reward model is weak and exactly where PPO will get into trouble. Report the stratified numbers, not just the headline, or you will be blindsided when a model that "looked fine" reward-hacks on the subtle distinctions.

Beyond accuracy, three diagnostics matter. **Correlation with human judgment** on a held-out set with multiple annotators per pair tells you whether the reward model tracks the *consensus* and not one labeler's quirks. The **preference gap** — the mean of $r_w - r_l$ across held-out pairs — tells you how *separated* the model keeps chosen from rejected; a tiny gap means the model is barely discriminating, a huge gap can mean overfitting. And **reward model ensembles**: train several reward models with different seeds (and ideally different data shuffles), and look at their *disagreement* on a response. High ensemble variance flags out-of-distribution or adversarial responses where any single reward model is untrustworthy — and that variance can be used as an uncertainty penalty during PPO, which is one of the cleaner defenses against reward hacking. Multiple papers (e.g. Coste et al. 2023 on reward ensembles) found that ensembling and uncertainty-penalizing meaningfully reduced over-optimization.

#### Worked example: reward model calibration check

You hold out 1,000 preference pairs and bin them by the reward model's predicted preference probability $\sigma(r_w - r_l)$. In the bin where the model predicts 0.85–0.90 confidence, there are 200 pairs and the chosen response actually wins 178 of them — an empirical win rate of 89%. That bin is well-calibrated: predicted ~87.5% matches observed 89%.

But in the highest-confidence bin, predicted 0.95–1.00, there are 150 pairs and the chosen wins only 126 — an empirical 84%. The model claimed near-certainty and was wrong one time in six. That is overconfidence concentrated exactly where PPO will push hardest, because the policy will preferentially generate responses the reward model is *most* sure are great. Fitting a temperature $T = 1.4$ on this held-out set and dividing every gap by 1.4 before the sigmoid pulls the top bin's predicted confidence down to ~0.88, matching the observed 84% far better. Ranking is unchanged; the dangerous overconfidence is gone. This single scalar can be the difference between a PPO run that improves and one that reward-hacks into garbage.

## 7. The three classic reward model failures

If a reward model could be perfect we would not need this section. But reward models are judges trained on noisy data, and they pick up *shortcuts* — features that correlate with human approval in the training set but are not what we actually want. When the policy then optimizes against the reward model, it finds and exploits exactly these shortcuts. This is Goodhart's law in its purest form: the reward model is a *measure* of quality, and once it becomes a *target*, it ceases to be a good measure. Three shortcuts show up in nearly every RLHF pipeline.

**Length bias** is the most notorious. Longer responses tend to get higher reward, almost regardless of content. The mechanism is simple: in the preference data, more thorough answers are often genuinely better, so length correlates with preference. The reward model latches onto length as a cheap proxy. Then PPO discovers that simply making responses *longer* increases reward, and your once-concise model starts padding every answer with restatements, caveats, and bullet-pointed filler. The Llama-2 paper and many others explicitly report fighting length bias. You detect it by plotting reward against response length on held-out data — if reward climbs monotonically with length, you have a bias. You mitigate it by length-normalizing the reward, adding a length penalty during PPO, or balancing the preference data so that length and quality are decorrelated (deliberately including pairs where the *shorter* response is chosen).

**Sycophancy** is the reward model rewarding agreement with the user's stated beliefs. If the preference data was collected from annotators who (consciously or not) preferred responses that validated the prompt's premise, the reward model learns that agreeing is good. The policy then learns to tell users what they want to hear — confidently affirming a false claim in the prompt because that scores higher than politely correcting it. Sharma et al. (2023) at Anthropic documented sycophancy across multiple RLHF'd models and traced it partly to preference data. You detect it by constructing probe prompts containing false claims and checking whether higher reward goes to the response that agrees versus the response that corrects. Mitigation is hard and mostly upstream: instruct annotators explicitly to penalize unwarranted agreement, and include "the user is wrong here" pairs in the data.

**False positives on confident-sounding responses.** Fluency and confidence are surface features that correlate with quality in the training data, so the reward model rewards a smooth, assertive, well-formatted answer over a hedged, correct one. The policy learns to *sound* authoritative regardless of whether it *is* correct — the verbal equivalent of a confident bluffer. This is especially dangerous because it directly trades off against honesty. Detection: hold out pairs where the *correct* response is hedged and the *incorrect* response is confident, and measure how often the reward model is fooled. Mitigation overlaps with calibration (Section 5) and, for factual domains, with grounding the reward in verifiable correctness rather than style — which is exactly what process reward models do.

The deep lesson connecting all three: a reward model trained only on *outcome preferences* over *style* will reward style. To reward substance you must either put substance in the labels (annotator discipline), in the architecture (ensembles flagging uncertainty), or in the signal itself (process supervision). The figure for process reward (Figure 7) shows the most structural fix.

#### Worked example: measuring length bias before it bites

Here is a concrete diagnostic you can run in an afternoon. Take 1,000 held-out preference pairs, and for each compute the reward-model score and the token length of both responses. Now ask a single question: *across all pairs, does the reward model assign higher reward to the longer response more often than the human chose the longer response?* Suppose the human chose the longer response in 540 of 1,000 pairs (a mild real correlation — longer answers genuinely are a bit better on average). But the reward model scores the longer response higher in 780 of 1,000 pairs. That 240-pair gap between 78% and 54% is your length bias, quantified: the reward model is over-attributing quality to length by roughly 24 percentage points. Now fit a simple linear control — regress reward on length and take the residual — and you find the residualized reward agrees with humans on the longer-response question only 55% of the time, matching the human 54%. The bias was almost entirely length. The fix that follows is mechanical: subtract a length term from the reward during PPO (a per-token length penalty), or retrain the reward model on a length-balanced dataset. Either way, you found a bias worth tens of points of PPO over-optimization with a fifteen-line script, before it ever reached the policy. This is the single highest-return diagnostic in the whole reward-modeling toolkit, and it is shocking how often teams skip it and then wonder why their model became verbose.

The connective tissue across all three failures is *Goodhart's law*, and it deserves a sharper statement than the usual slogan. The reward model is a *learned proxy* for human preference. It agrees with humans on the *training distribution* but diverges off it. PPO is a *search procedure* that specifically seeks out high-reward responses — including, eventually, the high-reward responses that are *only* high-reward because of proxy error. So PPO is not a neutral optimizer; it is an *adversary* against your reward model's weaknesses. Anything the reward model rewards for the wrong reason, PPO will find and exploit, given enough KL budget. This adversarial framing is the right mental posture for reward modeling: assume your reward model has exploitable holes, build the diagnostics to find them (length plots, sycophancy probes, confidence-vs-correctness checks, ensemble disagreement), and constrain the optimizer (KL penalty) so it cannot run through the holes faster than you can patch them.

## 8. Scaling laws and the capacity bottleneck

How good does a reward model get as you scale it? The empirical picture, from Gao, Schulman & Hilton (2022) on reward-model over-optimization and from the InstructGPT scaling analysis, is consistent. Reward-model accuracy improves smoothly with both the *number of preference pairs* and the *parameter count* — roughly log-linearly over the ranges studied. More data and more parameters both help, with diminishing returns, exactly as you would expect from a supervised model.

But there is a subtler and more important phenomenon: the **reward-model capacity bottleneck**, sometimes called reward over-optimization. As you run PPO and push the policy to higher *proxy* reward (the reward model's score), the *gold* reward (true human preference, measured separately) first rises with it, then peaks, then *falls* even as proxy reward keeps climbing. The policy has started exploiting the reward model's errors rather than genuinely improving. Gao et al. fit this gold-vs-proxy relationship and found the gap between proxy and gold grows predictably with the KL divergence the policy travels from its initialization, and that *larger* reward models delay the peak — they can be optimized harder before they break.

This is why the reward model's size relative to the policy matters so much. If the reward model is too weak to represent the distinctions a strong policy can make, PPO cannot make real progress: the policy quickly saturates the reward model's discriminative power and then spends the rest of training finding adversarial inputs. A too-weak reward model is not just suboptimal, it is actively misleading — it caps the achievable quality and then degrades it. The practical implication is the one PPO practitioners live by: track gold reward on a held-out human-evaluated set throughout training, and stop when it peaks, regardless of how high proxy reward goes. The KL penalty to the reference policy (the subject of the PPO posts) exists precisely to slow the march into the reward model's blind spots.

#### A small comparison table of what scales

| Lever | Effect on RM accuracy | Effect on over-optimization | Practical range |
|---|---|---|---|
| More preference pairs | Up (log-linear) | Delays the gold-reward peak | 10k → 1M pairs |
| Larger RM parameters | Up (log-linear) | Larger RM optimized harder before breaking | 0.4B → 6B+ |
| More PPO KL budget | No effect (RM fixed) | Pushes policy deeper into RM blind spots | KL 5 → 30 nats |
| RM ensemble (N models) | Slight up + uncertainty signal | Penalizing variance reduces hacking | N = 3 → 5 |

The row that surprises newcomers is the third: spending more optimization budget (KL) does *not* make the reward model better — the reward model is frozen during PPO — it only lets the policy travel further from its start, which past the peak means traveling further into the reward model's errors. Optimization is not the same as improvement once your measure becomes your target.

### Why the KL penalty is really a reward-model trust region

The KL penalty that every PPO-based RLHF setup applies is usually explained as "keep the policy from drifting too far," which is true but undersells what it is doing in reward-model terms. PPO optimizes a *regularized* objective:

$$\max_\theta\; \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big),$$

where $\pi_{\text{ref}}$ is the SFT reference policy and $\beta$ controls the penalty strength. The first term says "get more reward-model reward." The second term says "but stay close to the reference." Read through the lens of this post, the KL term is a **trust region around the distribution where the reward model is reliable**. The reward model was trained on responses sampled from (something near) $\pi_{\text{ref}}$, so its judgments are trustworthy near $\pi_{\text{ref}}$ and increasingly unreliable as the policy moves away. The KL penalty keeps the policy inside the region where the reward model still means what it says. Set $\beta$ too low and the policy escapes the trust region, finds the reward model's blind spots, and reward-hacks — proxy reward soars while gold reward collapses. Set $\beta$ too high and the policy barely moves and learns almost nothing. The reward model's reliability radius *is* the right setting of $\beta$, which is why teams that swap in a stronger, better-calibrated reward model can usually afford a *lower* KL penalty: a better reward model has a larger region of trustworthiness.

This reframing also explains an empirical regularity from Gao et al.: the gold-reward-versus-KL curve has a characteristic shape — it rises, peaks, and declines — and the *KL distance at the peak* is roughly where the reward model's reliability runs out. Bigger reward models push that peak further out (more KL budget before they break) because they have a larger reliable region. The whole RLHF tuning game, then, is matching three things: how good your reward model is, how far the KL penalty lets the policy roam, and where you stop. Misjudge any one and you either underfit or hack.

#### Worked example: spotting over-optimization in a training log

Imagine a PPO run logging both proxy reward (the reward model's score on rollouts) and gold reward (win rate against the SFT model, judged by held-out human eval every 2,000 steps). At step 2,000 proxy reward is 1.2 and gold win rate is 58%. At step 6,000 proxy reward has climbed to 2.8 and gold win rate to 67% — both moving together, healthy. At step 12,000 proxy reward is 4.1 but gold win rate has slipped to 64%. At step 18,000 proxy reward is 5.6 and gold win rate has fallen to 59%, barely above where it started, and a glance at the rollouts shows responses are now 40% longer and stuffed with confident-sounding filler. This is textbook over-optimization: the policy crossed the reward model's reliability radius somewhere around step 8,000–10,000, and everything after that was the policy mining the reward model's length and confidence biases (Section 7) rather than improving. The correct action is to *roll back to the step-6,000 checkpoint*, where gold reward peaked, and either raise the KL penalty, improve the reward model, or re-collect preference data on the current policy's outputs. Proxy reward at step 18,000 is more than 4× the peak and the model is *worse*. If you had only watched proxy reward, you would have shipped the disaster.

## 9. Process reward models versus outcome reward models

So far the reward model has scored a *whole response* with a single scalar — an **outcome reward model** (ORM). It judges the final product. For many tasks that is fine. But for multi-step reasoning, especially mathematics, the ORM has a fatal weakness: a chain of reasoning can reach the *right answer through wrong steps* (lucky cancellation, a sign error that undoes itself) or the *wrong answer through mostly-right steps* (one slip at the end). An ORM that only sees the final answer gives the same reward to a sound derivation and a fluky one, and it gives zero credit to a chain that was 90% correct before a single error. The signal is sparse and it rewards luck.

A **process reward model** (PRM) scores *each reasoning step*. Instead of one scalar for the whole chain, it emits a scalar per step — typically a probability that the step is correct given the steps before it. This is dense supervision: the model learns *where* reasoning went wrong, not just *that* it did. The figure below contrasts the two — a PRM scoring every step in a five-step chain versus an ORM scoring only the final answer, both feeding the policy.

![Graph diagram comparing a process reward model that scores each reasoning step against an outcome reward model that scores only the final answer with both feeding the policy gradient](/imgs/blogs/reward-modeling-from-human-preferences-7.png)

OpenAI's **PRM800K** (Lightman et al. 2023, "Let's Verify Step by Step") is the landmark result. They collected roughly 800,000 step-level correctness labels on solutions to competition math problems and trained a process reward model on them. On the MATH benchmark, using the PRM to rank candidate solutions (best-of-N selection) substantially outperformed an outcome reward model: the PRM picked the genuinely correct solution far more reliably because it could reject chains containing a bad step even when the final answer happened to look plausible. The headline finding was that *process supervision beats outcome supervision* for reasoning, and that the gap widens as you scale the number of candidate solutions you select among.

The cost is annotation. Labeling every step of every solution is dramatically more expensive than a single thumbs-up/down on the final answer — PRM800K's 800k step labels represent an enormous human effort. This is the central tradeoff, and the matrix figure earlier (Figure 5) lays it out: PRMs buy robustness and reasoning accuracy at high annotation cost; scalar ORMs are cheap but hackable; ensembles and Constitutional methods sit in between.

| Property | Outcome RM (ORM) | Process RM (PRM) |
|---|---|---|
| What it scores | Final answer only | Every reasoning step |
| Signal density | Sparse (one scalar) | Dense (one per step) |
| Annotation cost | Low (label the answer) | High (label every step) |
| Rewards lucky correct answers | Yes (can't tell) | No (rejects bad steps) |
| Best for | Open-ended generation | Math, code, multi-step logic |
| Landmark | InstructGPT RM | PRM800K |

A pragmatic middle ground that has gained traction: derive step-level labels *automatically* via rollouts — for each prefix of a reasoning chain, sample several completions and label the step "good" if completions from that prefix reach the correct answer often enough (Math-Shepherd and similar). This gets PRM-style dense signal without per-step human labeling, trading some label noise for vastly lower cost.

## 10. Constitutional AI: rewards without human labels

Every reward model so far depends on human comparisons, and humans are slow, expensive, and inconsistent on the hardest cases (especially safety, where you may not want annotators reading harmful content all day). **Constitutional AI** (Bai et al. 2022, Anthropic) sidesteps the bottleneck by generating the preference signal with an AI instead of a human.

The mechanism has two phases. In the **critique-revision** phase, the model generates a response, then is prompted to *critique* its own response against a written set of principles — the "constitution" (e.g. "choose the response that is least harmful," "the most honest"). It then *revises* the response to address its own critique. Iterating critique-and-revise produces a dataset of improved responses for supervised fine-tuning, with no human labeling the harmful content. In the **RL from AI feedback (RLAIF)** phase, instead of asking a human which of two responses is better, you ask *the model* — given the two responses and a constitutional principle — to pick the better one. Those AI-generated preferences train a reward model exactly as human preferences would, via the same Bradley-Terry loss.

The striking result from the Constitutional AI work was that the AI-generated preferences produced a reward model and a final policy *competitive with* human-feedback RLHF on harmlessness, while requiring far fewer human labels — the humans wrote the constitution (a few dozen principles) instead of labeling hundreds of thousands of pairs. The signal moved from "label every pair" to "write down the rules once." That is a categorical reduction in human effort.

```python
# Sketch of generating an AI preference label for RLAIF.
# The "judge" model picks which response better follows a constitutional principle.
def ai_preference(judge, prompt, response_a, response_b, principle):
    query = (
        f"Principle: {principle}\n\n"
        f"User prompt: {prompt}\n\n"
        f"Response A: {response_a}\n\n"
        f"Response B: {response_b}\n\n"
        "Which response better follows the principle? Answer 'A' or 'B'."
    )
    verdict = judge.generate(query, max_new_tokens=1)
    # 'chosen' is the response the AI judge preferred; feed into BT loss.
    if verdict.strip().upper().startswith("A"):
        return response_a, response_b   # (chosen, rejected)
    return response_b, response_a
```

The caveats are real. AI feedback inherits the judge model's biases, and if the judge is itself sycophantic or length-biased, those biases propagate into the reward model. Constitutional methods reduce human *labeling* but shift the human effort to *writing a good constitution* and to *validating* that the AI preferences track human values — which still requires periodic human audits. It is a powerful tool for scaling, not a free lunch.

There is a deeper conceptual point worth drawing out, because it reframes what a reward model fundamentally *is*. Across all the variants — human pairwise, AI pairwise, process supervision, programmatic verification — the reward model is just a *mechanism for transferring a judgment from somewhere expensive to somewhere cheap*. Human preference is expensive to query, so we distill it into a fast neural network. A verifier (unit test, math solver) is the cheapest and most trustworthy reward when it exists, because it is *exact* — no learned approximation, no exploitable holes. AI feedback sits in between: cheaper than humans, less exact than a verifier, only as trustworthy as the judge. Seen this way, the choice of reward-model variant is a choice about *which judgment to distill and how faithfully*. Constitutional AI distills the judgment encoded in a written constitution as interpreted by a capable model; PRMs distill step-level correctness judgments; programmatic rewards distill an exact specification. The right question is never "which reward model is best" in the abstract, but "what is the most trustworthy and affordable source of the judgment I actually care about, and how do I transfer it with the least distortion?" That single question routes you correctly through the entire design space in Figure 8.

## 11. A full PyTorch reward model and training loop

Let me put it together end to end: the reward model class, the Bradley-Terry loss, and a training loop over preference pairs. This is deliberately framework-light (raw PyTorch + Hugging Face) so the mechanism is visible; the TRL version follows in the next section and collapses all of this into a few lines.

First, the Bradley-Terry loss as a standalone function — note how it is literally `-logsigmoid(gap)` averaged over the batch:

```python
import torch
import torch.nn.functional as F

def bradley_terry_loss(reward_chosen, reward_rejected, margin=0.0):
    """Negative log-likelihood of the chosen response winning, under Bradley-Terry.

    reward_chosen, reward_rejected: [B] tensors of scalar rewards.
    margin: optional fixed margin (use annotator strength here if available).
    """
    gap = reward_chosen - reward_rejected - margin
    # F.logsigmoid is numerically stable: no overflow for large negative gaps.
    return -F.logsigmoid(gap).mean()
```

Now a dataset and collator that tokenizes each chosen and rejected response (with its prompt) into a padded batch:

```python
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PreferenceDataset(Dataset):
    def __init__(self, triples, tokenizer, max_len=512):
        # triples: list of dicts {"prompt", "chosen", "rejected"}
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triples)

    def _encode(self, prompt, response):
        text = prompt + response
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, i):
        t = self.triples[i]
        cid, cmask = self._encode(t["prompt"], t["chosen"])
        rid, rmask = self._encode(t["prompt"], t["rejected"])
        return cid, cmask, rid, rmask

def collate(batch, pad_id):
    cids, cmasks, rids, rmasks = zip(*batch)
    def pad(seqs):
        maxlen = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            out[j, :s.size(0)] = s
            mask[j, :s.size(0)] = 1
        return out, mask
    cid, cmask = pad(cids)
    rid, rmask = pad(rids)
    return {"chosen_ids": cid, "chosen_mask": cmask,
            "rejected_ids": rid, "rejected_mask": rmask}
```

And the training loop, with the regularization habits from Section 5 baked in (small learning rate, single epoch, held-out accuracy logging, a tiny penalty keeping mean reward near zero):

```python
import torch
from torch.utils.data import DataLoader
from functools import partial

def train_reward_model(model, train_ds, val_ds, tokenizer,
                       epochs=1, lr=1e-5, batch_size=8, mean_reg=1e-3,
                       device="cuda"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    coll = partial(collate, pad_id=tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=coll)
    val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=coll)

    step = 0
    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            rw = model(batch["chosen_ids"].to(device),
                       batch["chosen_mask"].to(device))
            rl = model(batch["rejected_ids"].to(device),
                       batch["rejected_mask"].to(device))
            loss = bradley_terry_loss(rw, rl)
            # Keep the unidentified additive constant anchored near zero.
            loss = loss + mean_reg * (rw.mean() ** 2 + rl.mean() ** 2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            if step % 200 == 0:
                acc = eval_pairwise_accuracy(model, val_dl, device)["accuracy"]
                print(f"step {step} loss {loss.item():.4f} val_acc {acc:.3f}")
                model.train()  # eval() inside the metric flips it back
    return model
```

A few things worth noticing in this loop. The same `model` scores both chosen and rejected — they share every weight, which is the whole point of a single reward model (Figure 4 shows this shared forward pass). Gradient clipping at norm 1.0 is not optional; reward-model gradients can spike on hard pairs and blow up the additive constant. And the held-out accuracy print every 200 steps is your early-stopping signal — when it peaks, you stop, no matter what the training loss is doing.

![Pipeline diagram of reward model training where a prompt with chosen and rejected responses runs two forward passes through shared weights producing scores feeding a Bradley-Terry pair loss and backprop](/imgs/blogs/reward-modeling-from-human-preferences-4.png)

### The TRL version

In production you rarely hand-roll this. TRL's `RewardTrainer` implements exactly the Bradley-Terry loss above, handles the last-non-pad-token indexing, and integrates with the Hugging Face `Trainer` ecosystem:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset

model_name = "EleutherAI/pythia-410m"
# num_labels=1 turns the classification head into a scalar reward head.
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

config = RewardConfig(
    output_dir="rm-pythia-410m",
    per_device_train_batch_size=8,
    num_train_epochs=1,            # one epoch — RMs overfit fast
    learning_rate=1e-5,
    logging_steps=50,
    max_length=512,
)

trainer = RewardTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

That is the entire reward-model training stage of an RLHF pipeline. The output is a checkpoint you can load and call as the reward function inside `PPOTrainer`. For why language models need this whole pipeline see `/blog/machine-learning/reinforcement-learning/why-language-models-need-rlhf`, and for how that reward then drives policy optimization with a KL penalty to a reference model see the PPO-for-LLMs post at `/blog/machine-learning/reinforcement-learning/ppo-for-llm-fine-tuning`.

## 12. Case studies

**InstructGPT (Ouyang et al., 2022).** OpenAI trained a 6B reward model on labeler comparison data (rankings of 4–9 responses per prompt, expanded to all pairwise comparisons) and used it to PPO-fine-tune GPT-3 variants. The headline result: outputs from the 1.3B InstructGPT model were *preferred by human evaluators over the 175B GPT-3* base model, despite being more than 100× smaller. The reward model was the linchpin — it converted a few tens of thousands of comparisons into a reward signal that reshaped a giant model's behavior. This is the result that made RLHF the dominant alignment paradigm.

**Anthropic HH-RLHF (Bai et al., 2022).** Anthropic released ~170k human preference comparisons and trained reward models on the helpfulness and harmlessness splits. A consistent finding across their analysis was the tension between the two objectives — a reward model optimized hard for harmlessness could make the policy evasive and unhelpful ("I can't help with that" to benign requests), illustrating that the reward model *encodes the tradeoffs in its training data*. If your data over-weights one axis, your policy will too.

**PRM800K / "Let's Verify Step by Step" (Lightman et al., 2023).** Process supervision on ~800k step-level labels for math solutions. Selecting the best of N candidate solutions with the process reward model substantially beat selecting with an outcome reward model on the MATH benchmark, with the margin growing as N increased. This established process reward models as the state of the art for reasoning and motivated the wave of automatically-labeled PRMs (Math-Shepherd and successors) that followed.

**Llama-2 (Touvron et al., 2023).** Meta trained *two* reward models — one for helpfulness, one for safety — and explicitly engineered against length bias and reward over-optimization. They used a margin term derived from the annotator's preference strength and reported reward-model accuracy improving with both data and model scale, consistent with the scaling-law picture in Section 8. The two-reward-model design is a clean answer to the helpfulness/harmlessness tension Anthropic surfaced: don't average two objectives into one confused scalar, model them separately and combine at PPO time.

The reward model timeline (Figure 6) places these in order, from the 2017 Atari preference-learning work that introduced the pairwise paradigm to the process reward models of 2024.

![Timeline of reward model evolution from Atari pairwise preferences in 2017 through InstructGPT and Anthropic HH-RLHF in 2022 to PRM800K and process reward models by 2024](/imgs/blogs/reward-modeling-from-human-preferences-6.png)

## 13. Reward model types compared

Pulling the design space together. The matrix figure (Figure 5) summarizes four reward-signal designs across annotation cost, robustness to hacking, math-reasoning strength, and where each was deployed.

![Matrix comparing scalar process ensemble and constitutional reward models across annotation cost robustness to hacking math reasoning strength and deployment](/imgs/blogs/reward-modeling-from-human-preferences-5.png)

| Approach | How the signal is made | Strength | Weakness |
|---|---|---|---|
| Scalar RM | Human pairwise preferences, Bradley-Terry | Cheap, simple, well-understood | Hackable (length, sycophancy), sparse |
| Process RM | Per-step correctness labels | Dense signal, strong on reasoning | Very expensive to annotate |
| Ensemble RM | N scalar RMs, use disagreement | Uncertainty signal curbs over-optimization | N× training and inference cost |
| Constitutional / RLAIF | AI judges pairs vs a written constitution | Few human labels, scales to safety | Inherits judge model's biases |
| DPO (implicit RM) | No separate RM; preferences in the policy loss | No reward model to train or hack | Less control; tied to one policy |

That last row points forward. **Direct Preference Optimization** (Rafailov et al., 2023), covered in depth at `/blog/machine-learning/reinforcement-learning/dpo-direct-preference-optimization`, makes the remarkable observation that the optimal RLHF policy has a *closed-form relationship* to the reward model, so you can substitute the policy's own log-probabilities in for the reward and optimize the Bradley-Terry loss *directly on the policy* — no separate reward model, no PPO loop. The reward model becomes *implicit*. DPO is simpler and more stable, and it has become the default for many fine-tunes, but it gives up the explicit reward model's reusability (you can't query a DPO "reward" on arbitrary responses) and its uncertainty estimates. The decision tree in Figure 8 walks the choice.

![Decision tree for choosing a reward model based on whether step-level signal is needed annotation budget and tolerance for AI feedback leading to PRM pairwise constitutional or DPO](/imgs/blogs/reward-modeling-from-human-preferences-8.png)

## When to use this (and when not to)

Reach for an **explicit scalar reward model** when you need to run PPO and want the reward to be reusable across multiple policy iterations, when you want to inspect and debug the reward signal directly, or when you want ensemble uncertainty to defend against over-optimization. It is the workhorse and the default mental model for RLHF.

Reach for a **process reward model** when the task has verifiable intermediate structure — math, code, multi-step logic — and you can afford (or automatically generate) step-level labels. The reasoning gains are large and well-documented, and best-of-N selection with a PRM is often the single highest-leverage change for a reasoning model.

Reach for **Constitutional AI / RLAIF** when human labeling is the bottleneck — especially on safety content you'd rather not have humans read at scale — and you can write a clear set of principles. Validate periodically against human judgment so the AI judge's biases don't silently take over.

**Skip the reward model entirely and use DPO** when you have a fixed preference dataset, want a simpler and more stable pipeline, don't need to query the reward on arbitrary new responses, and are fine-tuning a single policy rather than running an iterative RL loop. For many teams DPO is now the first thing to try, and you only escalate to an explicit reward model plus PPO when DPO's ceiling isn't enough.

Do *not* build a heavyweight reward-modeling pipeline when a verifiable, programmatic reward exists. If you can check correctness with a unit test, a math solver, or a game score, use *that* — it is unhackable in a way no learned reward model can be, and it sidesteps every failure in Section 7. Learned reward models are for the domains where the reward genuinely cannot be written down. When it can, write it down.

## Key takeaways

- Humans can't assign reliable scalar rewards to text, but they *can* compare two responses reliably. The reward model converts cheap, robust pairwise comparisons into a dense, queryable, differentiable reward function.
- The Bradley-Terry model gives the right loss: $\mathcal{L} = -\log\sigma(r_w - r_l)$. It depends only on the reward *gap*, which is exactly why the absolute scale is meaningless and only rankings are identified.
- Architecture is a scalar head on a pretrained LM, read off the last non-pad token. Initialize from the same base as the policy so the reward signal is legible to the optimizer, and don't make the reward model tiny relative to what it judges.
- Reward models overfit fast on noisy preference data: train ~one epoch, early-stop on held-out pair accuracy, and calibrate with temperature scaling so the policy doesn't chase overconfident errors.
- Held-out pairwise accuracy near your human-agreement ceiling (~70%) is good; reported accuracy far above it means a leak, not a great model.
- Watch for the three classic failures — length bias, sycophancy, and rewarding confident-but-wrong fluency — and probe for each explicitly. They are Goodhart's law made manifest once the reward model becomes the policy's target.
- Reward-model quality scales log-linearly with data and size, but more PPO optimization is *not* improvement: gold reward peaks then falls as the policy exploits the frozen reward model. Track gold reward and stop at the peak.
- Process reward models beat outcome reward models on reasoning by giving dense per-step signal; the cost is per-step annotation, which automatic rollout labeling can partly offset.
- Constitutional AI and RLAIF replace human comparisons with AI-judged ones against a written constitution, scaling past the human-labeling bottleneck at the cost of inheriting the judge's biases.
- If a verifiable programmatic reward exists, use it instead of a learned reward model. Learned reward models are for the domains where no one can write the reward down.

## Further reading

- Christiano, Leike, Brown, Martic, Legg & Amodei, "Deep Reinforcement Learning from Human Preferences," 2017 — the Atari work that introduced training a reward model from pairwise human preferences.
- Ziegler, Stiennon, Wu, Brown, Radford, Amodei, Christiano & Irving, "Fine-Tuning Language Models from Human Preferences," 2019 — the first application to language models, with the reward-model overfitting analysis.
- Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT), 2022 — the 6B reward model and the 1.3B-beats-175B result.
- Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022 — critique-revision and RLAIF.
- Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization," 2022 — the gold-vs-proxy reward curves and the capacity bottleneck.
- Lightman et al., "Let's Verify Step by Step" (PRM800K), 2023 — process versus outcome reward models on math.
- Rafailov, Sharma, Mitchell, Manning, Ermon & Finn, "Direct Preference Optimization," 2023 — the implicit reward model that skips PPO.
- Within this series: the unified map at `/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map` and the capstone at `/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook`.
