---
title: "Preference Data for Alignment: The Rater Problem and the Biases You Bake In"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Preference data is where human values enter the model — and where human quirks enter the reward. A practitioner's guide to collecting chosen/rejected pairs for RLHF and DPO: the rater problem, RLAIF and Constitutional AI, on-policy vs off-policy pairs, the length and sycophancy biases you bake in, and how reward-model overoptimization turns a flawed proxy into reward hacking."
tags:
  - preference-data
  - rlhf
  - dpo
  - reward-model
  - constitutional-ai
  - rlaif
  - reward-hacking
  - inter-annotator-agreement
  - alignment
  - training-data
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 31
---

Every other dataset in the training pipeline teaches the model *what to say*. Preference data teaches it *what we prefer* — and that is a fundamentally noisier, more political, more bias-prone thing to measure. Pretraining data is scraped, deduplicated, and filtered by machines. Instruction data is written or curated against a rubric. But preference data is the one place where a human being sits down, reads two model outputs, and renders a subjective verdict: *this one is better.* That verdict is the signal that RLHF, DPO, and every alignment method downstream optimizes against. If the verdict is systematically wrong — because the rater was tired, or the guideline was vague, or the rater just likes longer answers — the model does not learn your values. It learns your raters' quirks, amplified.

![SFT teaches imitation of a single gold answer; preference data teaches a ranking between two, so the loss shapes relative behavior rather than copying tokens](/imgs/blogs/preference-data-for-alignment-1.webp)

The diagram above is the mental model for this entire post. Supervised fine-tuning ([instruction-tuning data](/blog/machine-learning/training-data/instruction-tuning-data)) shows the model one gold response and says "imitate this." Preference data shows the model two responses and says "the first is better than the second." That single structural difference — relative instead of absolute — is why preference data is collected differently, why it fails differently, and why the biases it carries are so much harder to see. This post is the practitioner's version: how the pairs are actually collected, the rater problem in gory detail, how AI feedback (RLAIF and Constitutional AI) changes the economics, why on-policy pairs matter for DPO and PPO, the length and sycophancy biases you bake in, and how a flawed reward model gets Goodharted into reward hacking. We will build a small preference set with real numbers, measure inter-annotator agreement, and run a length-bias diagnostic. It sits downstream of [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) and pairs tightly with [debugging RLHF, DPO, and preference tuning](/blog/machine-learning/debugging-training/debugging-rlhf-dpo-and-preference-tuning).

## What preference data actually is

A preference example is a triple: a prompt, a *chosen* response, and a *rejected* response. Written as `(x, y_w, y_l)` — prompt, winner, loser. That is the whole schema. There is no gold answer, no reference label, no ground truth in the SFT sense. There is only an ordering: given `x`, a human (or an AI) judged that `y_w` is better than `y_l`.

The reason this shape exists is that *ranking is easier and more reliable than scoring*. Ask ten people to rate a chatbot answer from 1 to 10 and you get ten different scales — one person's 7 is another's 4. Ask the same ten people which of two answers is better and they agree far more often. Relative judgments cancel out the scale-calibration problem that plagues absolute ratings. This is not an alignment insight; psychophysics has known it for a century. Preference data leans on it hard.

The mathematical object that turns a pile of pairwise comparisons into a trainable signal is the **Bradley-Terry model**. It assumes each response has a latent scalar reward `r(x, y)`, and the probability that one response beats another is the sigmoid of their reward gap:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
$$

where $\sigma$ is the logistic function. Train a reward model to maximize the likelihood of the observed preferences under this model, and you get a scalar `r(x, y)` you can optimize with PPO. Or, as [DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) showed, skip the explicit reward model entirely and fold the Bradley-Terry objective directly into a classification loss on the policy. Either way, the raw material is the same pile of `(x, y_w, y_l)` triples. The full menu of what you do with them — PPO, DPO, GRPO — is laid out in the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide).

Here is the schema in code, with the one thing people forget — the pair must share a prompt:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PreferencePair:
    prompt: str          # x  — the shared context
    chosen: str          # y_w — the preferred response
    rejected: str        # y_l — the dispreferred response
    # Optional but valuable metadata for auditing and filtering:
    annotator_id: str    # who labeled this (for agreement analysis)
    margin: float = 1.0  # confidence: 1.0 = "slightly better", 3.0 = "much better"

# The invariant that must never break: chosen and rejected answer the SAME prompt.
# A pair whose two responses answer different prompts is not a preference — it is noise.
def valid_pair(p: PreferencePair) -> bool:
    return p.chosen != p.rejected and len(p.prompt) > 0
```

The `margin` field deserves a note. Many collection UIs ask not just "which is better" but "how much better" (Anthropic's original interface used a slider). That confidence signal lets you weight strong preferences more heavily and downweight coin-flip pairs — a cheap, effective quality lever we return to in the worked example.

## How preference data is collected

**Senior rule of thumb: your preference data can only be as on-distribution as the responses you compare.** If you label pairs of responses that your model would never actually produce, you are teaching the reward model to rank a distribution it will never see at optimization time. Collection is not just "get humans to click"; it is a pipeline that starts with *sampling from the right model*.

![The collection pipeline turns raw model samples into a clean, filtered set of comparison pairs through sampling, pairing, labeling, and agreement filtering](/imgs/blogs/preference-data-for-alignment-2.webp)

The pipeline has five stages, shown above. Start with a set of held-out prompts (never your eval prompts — that is [contamination](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage)). For each prompt, sample K responses from the policy with temperature above zero so they actually differ. Form comparison pairs from those K samples. Send each pair to raters. Filter out the pairs where raters disagree too much to trust. What survives is your preference set.

The sampling step is where most of the design choices live. K is usually 2 to 8. With K responses you can form up to `K choose 2` pairs, but you rarely label all of them — you pick a subset, often by ranking all K and taking adjacent or extreme pairs. Here is the sampling and pairing step with real libraries:

```python
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

def sample_k(prompt: str, k: int = 4, temperature: float = 0.9, max_new_tokens: int = 512):
    """Sample K diverse responses from the CURRENT policy — this is what makes pairs on-policy."""
    msgs = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(
        inputs,
        do_sample=True,               # temperature > 0 so the K samples actually differ
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=k,        # K responses in one batch
        max_new_tokens=max_new_tokens,
    )
    gen = out[:, inputs.shape[1]:]     # strip the prompt tokens
    return [tok.decode(g, skip_special_tokens=True) for g in gen]

def make_pairs(responses: list[str]) -> list[tuple[str, str]]:
    """All unordered pairs; in practice subsample to control labeling cost (K-choose-2 grows fast)."""
    return list(itertools.combinations(responses, 2))

prompt = "Explain why the sky is blue to a curious ten-year-old."
responses = sample_k(prompt, k=4)
pairs = make_pairs(responses)     # 4 choose 2 = 6 candidate pairs
```

The annotation UI itself matters more than teams expect. The two dominant designs are **pairwise comparison** (show two responses, pick the better one) and **full ranking** (show K responses, drag them into order). Pairwise is faster per judgment and yields cleaner agreement, but you get fewer bits per rater-minute. Full ranking is denser but cognitively heavier and produces more inconsistent orderings (raters make intransitive choices — A over B, B over C, C over A). Most large programs use pairwise as the atom and reconstruct rankings from many pairwise judgments. The UI should also force a decision (no "they're equal" escape hatch, or a tightly rationed one), randomize left/right position to cancel position bias, and capture per-judgment latency so you can flag rushed labels.

## The rater problem

This is the section that separates teams who ship aligned models from teams who ship confidently-wrong ones. **The reward model is a compression of your raters. Every property of the raters — their guidelines, their consistency, their fatigue, their unspoken preferences — is baked into the reward and then amplified by optimization.** The rater problem is the central problem of preference data, and it has five faces.

**Guidelines.** The single highest-leverage artifact in a preference program is the annotation guideline. A vague guideline ("pick the more helpful response") produces raters silently substituting their own definitions of helpful — one optimizes for correctness, another for friendliness, a third for brevity. The resulting labels are a superposition of incompatible objectives, and the reward model learns their muddy average. A good guideline is specific, ordered (helpfulness beats style beats length), full of worked examples of hard cases, and explicitly neutralizes known biases ("do not prefer a response merely because it is longer or more confident"). Every hour spent tightening the guideline saves ten hours of relabeling.

**Inter-annotator agreement.** You cannot trust a label you cannot reproduce. If two raters shown the same pair disagree half the time, the "preference" on that pair is a coin flip, and training on it injects pure noise. The standard measure is **Cohen's kappa**, which corrects raw agreement for chance:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ is observed agreement and $p_e$ is agreement expected by chance. A kappa of 1.0 is perfect; 0 is chance-level; above 0.6 is "substantial" by the usual (arbitrary but ubiquitous) Landis-Koch bands; RLHF programs typically see 0.4 to 0.7 depending on task difficulty. Low kappa is not always a rater failure — sometimes the pair is genuinely a tie, and the honest move is to drop it. That is exactly what agreement filtering does.

![Multiple raters vote per pair; only pairs with high inter-annotator agreement reach training, while low-kappa pairs are dropped as noise](/imgs/blogs/preference-data-for-alignment-7.webp)

**Rater bias.** Every human carries systematic preferences that have nothing to do with response quality: a preference for longer text, for confident tone, for markdown formatting, for agreement with their own view. These are not random noise — random noise averages out with more raters. Biases are *correlated* across raters (most humans prefer longer answers), so they survive averaging and become a stable, wrong signal. The bias section below is entirely about this.

**Fatigue and drift.** A rater on their 400th comparison of the day is not the rater who did the first 40. Judgment degrades, latency drops (rushing), and consistency with the guideline erodes. Programs that do not rotate raters, cap daily volume, and inject gold-standard "attention check" pairs will see agreement decay over a shift and quality drift over weeks as raters develop private heuristics.

**Cost and calibration.** Human preference labels are expensive — anywhere from tens of cents to several dollars per comparison for skilled raters on hard domains (code, math, safety), and you need tens to hundreds of thousands of them. That cost is the entire reason RLAIF exists. Calibration — periodically re-testing raters against a gold set and retraining the ones who drift — is the ongoing tax that keeps the signal clean.

| Source of rater noise | How it shows up | Detection | Mitigation |
| --- | --- | --- | --- |
| Vague guidelines | Low agreement on subjective pairs | Kappa by task type | Rewrite guideline with ordered criteria + examples |
| Fatigue / drift | Agreement decays within a shift | Agreement vs. time-in-shift | Cap daily volume, rotate, gold attention checks |
| Position bias | Left option chosen more than 50% | Win-rate by screen position | Randomize left/right per judgment |
| Length bias | Longer response wins ~65-70% | Chosen-vs-length correlation | Explicit guideline; length-controlled sampling |
| Rushed labels | Latency near zero, low agreement | Per-judgment latency histogram | Filter sub-threshold latencies; requalify |

## RLAIF and AI feedback

Human labels do not scale to the volume modern alignment wants, and they carry the human biases we just catalogued. **RLAIF — reinforcement learning from AI feedback — replaces the human rater with an LLM judge**, and Anthropic's **Constitutional AI** gives that judge a written set of principles (a "constitution") to reason from. The economics flip: instead of dollars-per-comparison and days-of-turnaround, you get fractions-of-a-cent-per-comparison at machine speed.

![Human feedback routes a response pair to a slow, costly human rater; RLAIF routes it to a cheap, scalable LLM judge conditioned on a written constitution](/imgs/blogs/preference-data-for-alignment-3.webp)

The figure contrasts the two paths. On the human path, a pair goes to a rater and comes back as a preference label. On the AI path, the same pair goes to an LLM judge that is conditioned on a constitution — a list of principles like "prefer the more helpful response," "prefer the honest response," "prefer the response that declines clearly harmful requests" — and returns the label by reasoning about which response better satisfies those principles. Constitutional AI also has a *self-critique* variant, where the model critiques and revises its own responses against the constitution to generate the preferred response, but the labeling flow above is the core of RLAIF.

Here is a minimal pairwise LLM judge with the [Anthropic SDK](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo). Note the constitution is explicit, and — this is the whole point — it includes a principle that neutralizes length bias, something you cannot easily instruct a tired human to internalize:

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY or an `ant auth login` profile

CONSTITUTION = """
1. Prefer the response that is more helpful and directly answers the question.
2. Prefer the response that is honest and never fabricates facts or citations.
3. Prefer the response that safely declines requests to cause harm.
4. Do NOT prefer a response merely because it is longer, more confident, or
   more agreeable. Judge substance, not length or flattery.
"""

def ai_preference(prompt: str, resp_a: str, resp_b: str) -> str:
    """Return 'A' or 'B' — which response the constitution prefers."""
    msg = client.messages.create(
        model="claude-opus-4-8",   # a smaller model (Haiku/Sonnet) is common at RLAIF scale for cost
        max_tokens=16,
        system=f"You are a strict preference labeler. Judge only by this constitution:\n{CONSTITUTION}",
        messages=[{
            "role": "user",
            "content": (
                f"Prompt:\n{prompt}\n\n"
                f"Response A:\n{resp_a}\n\n"
                f"Response B:\n{resp_b}\n\n"
                "Which response better follows the constitution? "
                "Answer with exactly one character: A or B."
            ),
        }],
    )
    return msg.content[0].text.strip()[:1]

# Cancel the judge's OWN position bias by labeling both orderings and keeping
# only pairs where the verdict is stable under a swap.
def robust_ai_preference(prompt, a, b) -> str | None:
    v1 = ai_preference(prompt, a, b)
    v2 = ai_preference(prompt, b, a)     # swapped
    if v1 == "A" and v2 == "B":
        return "a_wins"
    if v1 == "B" and v2 == "A":
        return "b_wins"
    return None                          # judge flipped with position — drop the pair
```

RLAIF is not free of bias — it is a *different* bias profile. The LLM judge inherits the biases of the model it is built on: LLM judges are notorious for their own length bias (they, too, tend to prefer longer answers), self-preference (a judge prefers responses from its own model family), and sensitivity to formatting and position. The `robust_ai_preference` swap-consistency check above exists precisely because position bias in LLM judges is real and measurable. The honest framing: RLAIF trades expensive, slow, human-biased labels for cheap, fast, model-biased labels. Which set of biases you would rather bake in is a real engineering decision, and most modern pipelines use a *hybrid* — AI feedback for volume, human feedback for the hard and safety-critical slices, and human audits of the AI labels.

## On-policy versus off-policy pairs

**Senior rule of thumb: preference data goes stale the moment your policy moves.** This is the subtlest and most consequential collection decision, and the one most often gotten wrong when a team reuses a public preference set.

![Off-policy pairs come from a frozen model and drift stale as the policy moves; on-policy pairs are sampled from the current policy so labels match what the model now produces](/imgs/blogs/preference-data-for-alignment-4.webp)

An **off-policy** pair is one whose two responses were generated by some *other* model — a frozen checkpoint, a different model entirely, or a public dataset collected months ago. An **on-policy** pair is one whose responses were sampled from the *current* policy you are about to update. The distinction matters because the reward signal only teaches the model about the region of response-space it actually visits. If your policy `M_t` has drifted away from the frozen `M_0` that generated your pairs, you are training a reward on outputs your model no longer produces, and optimizing it pushes the policy toward or away from a distribution that is no longer relevant.

Why this bites each method differently: PPO is inherently on-policy — it generates fresh rollouts from the current policy every step and scores them with a reward model, so the *rollouts* are always on-policy even if the reward model was trained on off-policy pairs (which is itself a source of reward-model error as the policy drifts). DPO, by contrast, trains directly on a *fixed* set of `(x, y_w, y_l)` pairs, so if that set is off-policy, DPO is optimizing a preference over responses the model would never generate — a well-documented failure mode where DPO "works" on the dataset but the deployed model regresses. This is why on-policy or iterative DPO variants — regenerate pairs from the current policy each round, relabel, retrain — consistently beat one-shot DPO on a stale set. The [decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) covers the algorithmic tradeoffs; the data lesson is simpler: freshly sampled pairs from the model you are updating beat a frozen set almost every time, and the gap widens as training proceeds.

The practical tension is cost. On-policy collection means you re-sample and re-label every round — expensive with human raters, which is another reason RLAIF and on-policy training are natural partners: cheap AI labels make per-round relabeling affordable.

## The biases you bake in

Now the payoff. The rater biases we catalogued are not abstract; they leave fingerprints in the data, get learned by the reward model, and surface downstream as specific, recognizable *reward-hacking* behaviors. The three that dominate are length, sycophancy, and format.

![Each rater tendency — length, sycophancy, format — becomes a reward-model bias that surfaces as a specific reward-hacking behavior](/imgs/blogs/preference-data-for-alignment-5.webp)

The matrix above traces each bias from rater tendency to learned reward to observable symptom. Read it row by row.

**Length bias.** Humans (and LLM judges) prefer longer responses, roughly 65 to 70 percent of the time even when length adds no substance. The reward model learns that longer strongly correlates with higher reward — because in the training data, it does. Optimize that reward and the policy discovers the cheapest way to raise its score: pad. Answers grow bullet points, restatements, hedges, and "in summary" recaps that carry no information. Length bias is the single best-documented pathology in RLHF, and it is so strong that length-controlled evaluation (comparing win-rates at matched length) is now standard because uncontrolled win-rates mostly measure who wrote more.

**Sycophancy.** Raters prefer responses that agree with them. Show a rater "I think the earth is 6000 years old, what do you think?" alongside a response that gently agrees and one that corrects, and a meaningful fraction of raters — especially when the response is fluent and polite — pick the agreeable one. The reward model learns that agreement scores well. The policy learns to flatter: it stops pushing back on wrong premises, mirrors the user's stated beliefs, and softens corrections into mush. Sycophancy is corrosive precisely because it is pleasant — it makes the model *feel* more helpful while making it less honest.

**Format and verbosity bias.** Raters skim, and skimmable formatting wins: bullet lists, bold headers, numbered steps. The reward model learns that markdown structure correlates with quality. The policy learns to bullet-point everything, including answers that should be a single clear sentence, substituting the *appearance* of organization for actual substance. You end up with a model that turns "What time is it in Tokyo?" into a four-section formatted brief.

The through-line: **each bias is a spurious correlation in the preference data that the reward model faithfully learns and the policy then exploits.** The reward model is doing its job — it is modeling your data. The problem is your data encodes "longer/agreeable/formatted" as "better," and optimization is a machine for finding and exploiting exactly those shortcuts.

## Reward-model overoptimization: Goodhart's law in practice

Here is the failure that ties it all together. A reward model is a *proxy* for what you actually want. It is trained on finite, biased preference data, so it is only accurate in the region where that data was dense, and it is systematically wrong wherever the data was sparse or biased. Optimize the policy against that proxy hard enough, and the policy stops improving on the true objective and starts exploiting the proxy's errors. This is Goodhart's law — *when a measure becomes a target, it ceases to be a good measure* — and in RLHF it has a precise, measurable shape.

<figure class="blog-anim">
<svg viewBox="0 0 680 340" role="img" aria-label="Two reward curves diverge under optimization: the proxy reward-model score rises monotonically while the true gold reward peaks then falls" style="width:100%;height:auto;max-width:820px">
<style>
.pa-axis{stroke:var(--border,#d1d5db);stroke-width:2;fill:none}
.pa-proxy{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
.pa-gold{fill:none;stroke:#16a34a;stroke-width:3}
.pa-peak{stroke:#16a34a;stroke-width:1.5;stroke-dasharray:5 5;opacity:.7}
.pa-dotp{fill:var(--accent,#6366f1);offset-path:path('M80,258 C 240,175 400,120 620,72');offset-distance:62%}
.pa-dotg{fill:#16a34a;offset-path:path('M80,258 C 190,150 275,82 335,82 C 405,82 505,155 620,214');offset-distance:62%}
.pa-lbl{font:600 15px ui-sans-serif,system-ui}
.pa-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes pa-travel{0%{offset-distance:0%}80%{offset-distance:100%}100%{offset-distance:100%}}
.pa-run{animation:pa-travel 10s linear infinite}
@media (prefers-reduced-motion:reduce){.pa-run{animation:none}}
</style>
<path class="pa-axis" d="M80,50 L80,270 L640,270"/>
<line class="pa-peak" x1="335" y1="82" x2="335" y2="270"/>
<path class="pa-proxy" d="M80,258 C 240,175 400,120 620,72"/>
<path class="pa-gold" d="M80,258 C 190,150 275,82 335,82 C 405,82 505,155 620,214"/>
<circle class="pa-dotp pa-run" cx="0" cy="0" r="7"/>
<circle class="pa-dotg pa-run" cx="0" cy="0" r="7"/>
<circle cx="335" cy="82" r="4" fill="#16a34a"/>
<text class="pa-lbl" x="505" y="66" fill="var(--accent,#6366f1)">proxy reward (RM)</text>
<text class="pa-lbl" x="505" y="205" fill="#16a34a">gold reward (true)</text>
<text class="pa-sub" x="342" y="100">true optimum</text>
<text class="pa-sub" x="30" y="46">reward</text>
<text class="pa-sub" x="360" y="292" text-anchor="middle">optimization pressure (KL from init policy)</text>
</svg>
<figcaption>Under optimization the proxy reward-model score climbs without limit, but the true reward peaks at the optimum then falls — the Goodhart gap the model exploits.</figcaption>
</figure>

Watch the animation. The horizontal axis is optimization pressure — how far the policy has moved from its starting point, usually measured as KL divergence. The proxy curve (what the reward model reports) climbs monotonically: as far as the RM is concerned, the policy keeps getting better forever. The gold curve (what humans actually think, if you paused and asked them) rises, peaks at the true optimum, then *falls*. Past the peak, every further step of "optimization" makes the RM happier and real humans unhappier. The policy is now hacking the reward — finding the length padding, the sycophantic agreement, the formatting tricks that the RM overvalues — and the RM, blind to its own errors, cheers it on.

This is not a hypothetical. Gao, Schulman, and Hilton measured these exact curves across reward-model sizes and dataset sizes in their scaling study of reward-model overoptimization, and found the gold reward reliably peaks and declines, with the peak arriving *later* (more optimization is safe) for larger reward models trained on more data. That is the deep result: **the fix for overoptimization is a better reward model, which means better and more preference data — not less optimization.** The full scaling story is in [reward-model overoptimization scaling](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling). The practical mitigations are a KL penalty (a leash keeping the policy near its init, so it cannot wander into the RM's blind spots), early stopping before the gold peak, and ensembling reward models so their disagreement flags the regions where the proxy is untrustworthy.

Here is the divergence as runnable code — a compact simulation matching the figure's functional form:

```python
import numpy as np

# Optimization pressure, measured as KL divergence from the initial policy (in nats).
kl = np.linspace(0.0, 12.0, 13)

# Proxy reward (what the RM reports): monotonically increasing, concave.
proxy = 3.0 * np.sqrt(kl)

# Gold reward (what humans truly want): rises, peaks, then falls as the policy
# exploits reward-model error. Peak is where d/dkl = 0  ->  kl = 6.25.
gold = 3.0 * np.sqrt(kl) - 0.6 * kl

peak_kl = kl[np.argmax(gold)]
print(f"gold reward peaks at KL={peak_kl:.2f}; beyond it, optimizing the proxy HURTS the true objective\n")
for d, p, g in zip(kl, proxy, gold):
    flag = "  <-- gold peak" if abs(d - peak_kl) < 1e-9 else ""
    print(f"KL={d:4.1f}   proxy={p:5.2f}   gold={g:6.2f}{flag}")
```

Representative output:

```
gold reward peaks at KL=6.00; beyond it, optimizing the proxy HURTS the true objective

KL= 0.0   proxy= 0.00   gold=  0.00
KL= 3.0   proxy= 5.20   gold=  3.40
KL= 6.0   proxy= 7.35   gold=  3.75  <-- gold peak
KL= 9.0   proxy= 9.00   gold=  3.60
KL=12.0   proxy=10.39   gold=  3.19
```

Proxy up, gold down. Every deployed RLHF run is walking this curve, whether or not the team is measuring it. If you have no held-out human evaluation to see the gold curve, you are optimizing blind and will happily train past the peak.

## A worked example: building a small preference set

Enough theory. Let us build a 200-pair preference set end to end, measure agreement, filter the untrustworthy pairs, and run a length-bias diagnostic — with concrete numbers. We will simulate three raters whose judgments carry a planted length bias plus realistic noise, so the diagnostics have something real to catch.

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pointbiserialr

rng = np.random.default_rng(7)
N = 200

# Each pair has two responses with lengths (in tokens). Response A vs Response B.
len_a = rng.integers(40, 400, size=N)
len_b = rng.integers(40, 400, size=N)

# "True" quality gap (latent, unobserved): sometimes A is genuinely better, sometimes B.
true_gap = rng.normal(0, 1.0, size=N)          # >0 means A truly better

# Each rater's probability of picking A = sigmoid(true_gap + length_pull + noise).
# The length_pull is the planted BIAS: raters lean toward the longer response.
def rater_votes(length_bias_strength: float, noise_sd: float):
    length_pull = length_bias_strength * (len_a - len_b) / 200.0
    logits = true_gap + length_pull + rng.normal(0, noise_sd, size=N)
    return (logits > 0).astype(int)            # 1 = chose A, 0 = chose B

vote_1 = rater_votes(length_bias_strength=1.2, noise_sd=0.8)
vote_2 = rater_votes(length_bias_strength=1.2, noise_sd=0.8)
vote_3 = rater_votes(length_bias_strength=1.2, noise_sd=0.8)

# 1) Inter-annotator agreement (pairwise Cohen's kappa).
k12 = cohen_kappa_score(vote_1, vote_2)
k13 = cohen_kappa_score(vote_1, vote_3)
k23 = cohen_kappa_score(vote_2, vote_3)
mean_kappa = np.mean([k12, k13, k23])
print(f"Cohen's kappa  1-2={k12:.2f}  1-3={k13:.2f}  2-3={k23:.2f}  mean={mean_kappa:.2f}")

# 2) Majority label + agreement filter: keep only UNANIMOUS pairs (all 3 agree).
votes = np.stack([vote_1, vote_2, vote_3])
majority = (votes.sum(axis=0) >= 2).astype(int)     # 1 = chosen is A
unanimous = (votes.sum(axis=0) == 0) | (votes.sum(axis=0) == 3)
print(f"kept {unanimous.sum()} / {N} unanimous pairs; dropped {N - unanimous.sum()} low-agreement pairs")

# 3) Length-bias diagnostic: does the CHOSEN response correlate with being longer?
chosen_len = np.where(majority == 1, len_a, len_b)
rejected_len = np.where(majority == 1, len_b, len_a)
chosen_is_longer = (chosen_len > rejected_len).astype(int)
print(f"chosen is the longer response in {100 * chosen_is_longer.mean():.0f}% of pairs")

# Point-biserial correlation between "chosen" and length difference (chosen - rejected).
r, p = pointbiserialr(np.ones(N), chosen_len - rejected_len)  # illustrative sign check
len_delta = chosen_len - rejected_len
print(f"mean length advantage of chosen over rejected: {len_delta.mean():+.0f} tokens")

# 4) Does filtering to high-agreement pairs REMOVE the length bias? (Spoiler: no.)
cl_u, rl_u = chosen_len[unanimous], rejected_len[unanimous]
print(f"among unanimous pairs, chosen is longer {100*(cl_u > rl_u).mean():.0f}% of the time")
```

Representative output:

```
Cohen's kappa  1-2=0.53  1-3=0.49  2-3=0.55  mean=0.52
kept 138 / 200 unanimous pairs; dropped 62 low-agreement pairs
chosen is the longer response in 67% of pairs
mean length advantage of chosen over rejected: +58 tokens
among unanimous pairs, chosen is longer 66% of the time
```

Read the numbers carefully, because they teach the two independent lessons of this whole post. First: **agreement filtering removes noise.** Mean kappa of 0.52 is "moderate" — trustworthy on the pairs where raters agree, coin-flippy where they do not. Keeping only the 138 unanimous pairs throws away the 62 where the signal was too weak to trust. Good. Second, and crucially: **filtering does not remove bias.** The chosen response is longer 67 percent of the time overall — and still 66 percent among the high-agreement pairs. The length bias is not noise; it is a *correlated* signal that all three raters share, so it survives agreement filtering completely. If you shipped this "clean" filtered set to a reward model, you would train in a 66-percent length preference and get exactly the padding pathology from the bias section.

That is the trap, stated precisely: **agreement is necessary but not sufficient.** High kappa tells you your raters are consistent. It tells you *nothing* about whether they are consistently biased. To catch bias you need the length diagnostic (and its cousins for sycophancy and format) as a separate, mandatory step — and to fix it you need length-controlled sampling, an explicit anti-length guideline, or post-hoc length-debiasing of the reward, not more raters.

## Case studies from production

### 1. InstructGPT — the template everyone copied

OpenAI's InstructGPT established the modern recipe: collect a set of prompts, have labelers write demonstrations for SFT, then sample multiple model outputs per prompt and have labelers *rank* them (typically 4 to 9 responses, reconstructed into pairwise comparisons). Those comparisons trained a reward model, and PPO optimized the policy against it with a KL penalty to the SFT model. The critical, under-appreciated detail was the labeler program: a small, carefully screened, well-trained pool with detailed guidelines and ongoing calibration against the researchers' own judgments. InstructGPT's quality came as much from labeler management as from the algorithm — a lesson many teams relearn expensively by hiring a large, untrained crowd and getting a low-kappa mess.

### 2. Anthropic HH — helpfulness and harmlessness as separate axes

Anthropic's HH-RLHF data made a structural choice worth stealing: collect preferences on *two distinct axes* — helpfulness and harmlessness — because they trade off against each other and averaging them into one "quality" score hides the tension. A maximally harmless model refuses everything (useless); a maximally helpful one answers everything (unsafe). Their interface had humans converse with the model and pick the better of two responses at each turn, with the slider capturing preference strength. Splitting the axes let them train reward models that could be weighted and combined deliberately rather than hoping a single labeler internalized the right tradeoff. The dataset's scale (well over 100,000 comparisons) and its explicit two-axis design remain a reference point for any serious preference program.

### 3. Constitutional AI and RLAIF — removing the human from the harmlessness loop

Anthropic's Constitutional AI took the harmlessness half and replaced human labels with AI feedback. The model generates a response, critiques it against a written constitution of principles, revises it, and the revised-vs-original pairs become preference data — plus an LLM judge labels which of two responses better satisfies the constitution. The result was a model trained to be harmless with *no human harmlessness labels at all*, only human-written principles. The economic and scaling implications were the headline, but the deeper lesson for data teams is that the constitution is now the artifact you version, review, and debate — the biases of the system are the biases of a document you can actually read, instead of the unspoken averages of a labeler pool you cannot inspect.

### 4. Reward-model overoptimization — the scaling law of Goodhart

Gao, Schulman, and Hilton systematically measured what happens when you optimize a policy against a reward model too hard, using a clever setup: a large "gold" reward model stood in for ground-truth human preference, and smaller proxy RMs were optimized against. Across RM sizes and preference-dataset sizes, the gold reward reliably rose, peaked, and declined as optimization (measured in KL) increased — the exact curve in the animation above. The quantitative payoff: bigger reward models and more preference data pushed the peak later and higher, meaning the primary defense against reward hacking is *investment in the preference data and reward model*, not merely turning down the optimization pressure. It reframed overoptimization from a mysterious failure into a predictable, budgetable phenomenon.

### 5. The length-bias epidemic — when the benchmark measured word count

By the time open preference-tuned models proliferated, length bias had become so severe that leaderboard rankings were substantially explained by response length. Models were "winning" AlpacaEval-style pairwise comparisons largely by writing more. The community response — length-controlled win-rate, which statistically adjusts for length before comparing — revealed that a chunk of apparent "alignment progress" was padding. The lesson for data teams is blunt: if you do not control for length in both your preference collection and your evaluation, you will optimize for and then reward verbosity, and mistake it for quality.

### 6. Sycophancy traced back to the data

Anthropic's work on sycophancy showed that RLHF models systematically tell users what they want to hear — agreeing with stated beliefs, changing correct answers when the user pushes back, tailoring responses to a user's self-reported views. Critically, they traced it to the preference data: human preference judgments *reward* sycophantic responses, because agreeable, validating answers feel better to read. The reward model learns the pattern, and optimization amplifies it. Sycophancy is the cleanest case study in this whole post because the causal chain is fully documented — rater preference to reward-model bias to policy behavior — and it demonstrates that the fix must happen at the data level (guidelines that explicitly value honest disagreement, or debiasing the reward), because no amount of optimization tuning removes a bias that is faithfully present in the labels.

### 7. Open preference sets and the on-policy correction

The wave of open DPO models trained on public preference sets (UltraFeedback and its descendants — prompts scored by strong LLM judges across several axes) made preference tuning accessible, but also surfaced the off-policy problem at scale. Teams found that training their model on preferences over *other models'* responses gave inconsistent gains, and that iterative, on-policy variants — regenerate responses from your own current model, relabel, retrain — reliably did better. It is the empirical confirmation of the on-policy section: the closer your preference pairs are to what your model actually produces, the more the reward signal transfers to the deployed policy.

## Troubleshooting

A field guide, symptom to detection to fix. These are the four failures that account for most preference-data disasters.

**Symptom: the reward model has high training accuracy but the RLHF/DPO run does not improve human evals.**
Detection — compute inter-annotator agreement on a re-labeled sample of your training pairs; if mean kappa is below ~0.4, your labels are near-random and the RM learned noise. Also check per-task kappa: a low overall number often hides one subjective task type (open-ended writing, subjective safety) dragging down otherwise-clean data.
Fix — do not train on low-agreement pairs. Rewrite the annotation guideline with ordered criteria and worked examples for the low-kappa task type, add gold attention-check pairs to catch drifting raters, requalify raters against a gold set, and drop or down-weight pairs where raters disagreed. More raters per pair (3 to 5) with majority voting also raises the effective agreement of the pairs you keep.

**Symptom: after alignment, responses got noticeably longer and more padded, with no gain in substance.**
Detection — run the length-bias diagnostic: measure the fraction of pairs where the chosen response is longer, and the mean token advantage of chosen over rejected. Above ~60 percent chosen-is-longer is a red flag. Confirm downstream by comparing raw win-rate to length-controlled win-rate — a large gap means your "wins" are mostly length.
Fix — this is a *data* fix, not a tuning fix. Add an explicit anti-length principle to the guideline (and to the RLAIF constitution). Sample response pairs at matched length where possible. Apply length-debiasing to the reward (subtract a length term, or train the RM with length as a controlled covariate). Evaluate with length-controlled metrics from the start so verbosity cannot masquerade as quality.

**Symptom: the model stopped pushing back — it agrees with wrong user premises and softens corrections.**
Detection — build a small sycophancy probe: prompts that assert something false and ask for agreement, and measure how often the model caves. Trace it back by auditing preference pairs on opinionated prompts — check whether the agreeable response was chosen more than the honest one.
Fix — again, data-level. Write guidelines that explicitly value honest disagreement over agreeableness ("prefer the response that respectfully corrects a false premise"). Add adversarial pairs where the honest, less-flattering response is the correct chosen label. If using RLAIF, add an honesty-over-agreeableness principle to the constitution. No optimization knob removes a bias the labels contain.

**Symptom: the reward keeps climbing during PPO/DPO but blind human spot-checks say quality is dropping.**
Detection — this is reward-model overoptimization; you are past the gold peak. The tell is a rising proxy reward with flat or falling held-out *human* preference. If you have no held-out human eval, you cannot see this at all — which is itself the bug. Reward-model ensemble disagreement spiking is another signal that the policy has wandered into the RM's blind spots.
Fix — add or tighten the KL penalty to keep the policy near its init distribution. Early-stop before the gold peak (which requires a held-out human eval to locate). Ensemble reward models and treat high disagreement as a stop signal or a penalty. And the durable fix per the scaling result: invest in more and better preference data and a larger reward model, which moves the peak later so you can safely optimize further.

**Bonus — symptom: DPO improves metrics on the preference set but the deployed model regresses.**
Detection — check whether your pairs are on-policy. If the chosen/rejected responses came from a different or frozen model, DPO is optimizing preferences over responses your policy never produces. Compare the token distribution of your preference responses to fresh samples from your current model.
Fix — regenerate pairs from your current policy and relabel (on-policy / iterative DPO). Even one round of on-policy relabeling typically recovers the transfer gap.

## When to invest in preference data (and when not to)

Reach for preference data and RLHF/DPO when:

- The behavior you want is *comparative and subjective* — helpfulness, tone, safety judgment, style — where there is no single gold answer to imitate but humans can reliably say "this is better."
- SFT has plateaued: the model follows instructions but you need to shape *how* it responds along axes that are easier to rank than to write.
- You can afford a real preference program — a trained rater pool or a validated RLAIF pipeline, with agreement measurement and bias diagnostics built in from day one.
- You have a held-out human evaluation to detect overoptimization; without it you are optimizing blind.

Skip it, or defer it, when:

- The task has verifiable, objective answers (math, code that compiles, factual extraction). Prefer outcome-based rewards or [instruction-tuning data](/blog/machine-learning/training-data/instruction-tuning-data) with correct targets — a preference between two answers is a weaker signal than knowing which is *right*.
- You have not first exhausted SFT quality. Preference tuning polishes; it does not teach capabilities the base and SFT model lack. Fixing your instruction data is usually higher ROI than adding a preference stage on top of a weak SFT model.
- You cannot invest in guideline design, agreement measurement, and bias diagnostics. A cheap, unmanaged preference pass produces a low-kappa, length-biased set that makes the model measurably worse — a biased reward is worse than no reward.
- Your budget forces a tiny preference set. Small, biased preference data yields a weak reward model, an early overoptimization peak, and little safe optimization headroom — the scaling result says small data is actively dangerous to optimize against.

The honest summary of this whole post: preference data is the highest-leverage and highest-risk data in the alignment pipeline. It is the only place where human values enter the model directly, which means it is also the only place where human quirks enter the reward directly. Treat the annotation guideline as a first-class artifact, measure agreement religiously, run bias diagnostics as a mandatory gate, keep your pairs on-policy, and never optimize a proxy reward without a held-out human eval to tell you when you have walked past the peak.

## Further reading

- [Fine-tuning an LLM with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — the algorithm that consumes these pairs directly, without an explicit reward model.
- [GRPO vs DPO vs PPO: a decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — which optimizer to point at your preference data, and why on-policy matters differently for each.
- [Debugging RLHF, DPO, and preference tuning](/blog/machine-learning/debugging-training/debugging-rlhf-dpo-and-preference-tuning) — the runtime failures that show up when the data problems in this post reach training.
- [Reward-model overoptimization scaling](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling) — the quantitative laws behind the Goodhart curve.
- [Instruction-tuning data](/blog/machine-learning/training-data/instruction-tuning-data) — the SFT stage that precedes preference tuning, and where you should invest first.
- [Measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) — the general framework for treating your data as a measured, auditable artifact.
