---
title: "Constitutional AI and RLAIF: Scaling Alignment Without Human Labelers"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How a written constitution and an AI judge replace tens of thousands of human harmlessness labels, with the SL-CAI and RL-CAI pipelines, RLAIF, scalable oversight, and runnable critique-revision and preference-labeling code."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "llm-alignment",
    "constitutional-ai",
    "rlaif",
    "machine-learning",
    "pytorch",
    "preference-learning",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/constitutional-ai-and-rlaif-1.png"
---

Here is a number that ends a lot of alignment roadmaps before they start: to fine-tune a single helpful-and-harmless assistant with reinforcement learning from human feedback (RLHF), Anthropic's early work collected on the order of tens of thousands of human comparison labels, and the harmlessness portion alone required a standing crew of contractors reading model outputs, many of them deliberately toxic, day after day. The labels are slow to gather, they cost real money, the annotators disagree with each other about as often as they agree on the hard cases, and worst of all, the moment you change your safety policy — say you decide the model should now refuse a category it used to answer — a large fraction of that expensive label set is stale and you start over. The reward signal, the single most important ingredient in the whole RL loop, was bottlenecked on human attention.

The question that produced Constitutional AI is blunt: can the AI do the judging? Not the *deciding* — humans still write down what "harmless" means — but the labor-intensive part, the reading of two responses and saying "this one is better, and here is why." If a model can reliably critique its own outputs against a written set of principles, and revise them, and rank pairs of responses against those same principles, then the harmlessness signal stops scaling with annotator headcount and starts scaling with compute. That is the entire bet. The figure below traces the supervised half of that bet: a single red-teamed harmful prompt goes in, and a clean training example comes out the other side without any human writing the safe response.

![A pipeline showing a harmful prompt sampled into an initial response, then critiqued and revised by the model using a constitution, producing a revised response collected into an SFT dataset.](/imgs/blogs/constitutional-ai-and-rlaif-1.png)

By the end of this post you will be able to implement the two phases of Constitutional AI — supervised learning from AI feedback (SL-CAI) and reinforcement learning from AI feedback (RL-CAI) — from the critique-revision loop through the AI-generated preference data to the PPO update that consumes it. You will understand RLAIF as the broader family that Constitutional AI belongs to, where any capable model can serve as the labeler. You will see where these methods sit in the larger program of *scalable oversight* — debate, amplification, self-play — and, just as important, you will see honestly where they break: the biases an AI judge inherits, the circular reasoning when a model grades its own work, and the irreducible human labor of writing the constitution in the first place. This is still the RL loop you know from the rest of this series — an agent acting in an environment, collecting reward, updating a policy — but here the environment's reward function is itself produced by a model, and that one substitution changes almost everything about how alignment scales.

## 1. The human labeling bottleneck and why it caps alignment

Start from the standard RLHF recipe, because Constitutional AI is best understood as a surgical edit to it. RLHF as popularized by Christiano et al. (2017) and scaled to language models by Ziegler et al. (2019), Stiennon et al. (2020), and Ouyang et al. (2022, the InstructGPT paper) has three stages. First, supervised fine-tuning (SFT) on human-written demonstrations to get a model that follows instructions at all. Second, train a reward model on human preference comparisons: a human is shown a prompt and two model responses and picks the better one, and the reward model learns to predict that choice. Third, optimize the SFT policy against that reward model with PPO, regularized by a KL penalty back to the SFT reference so the policy does not drift into reward-hacking gibberish.

Every one of those three stages is gated on humans, but the second is the expensive one and the one that recurs forever. The reward model needs a steady diet of fresh comparisons because the policy keeps moving — the responses the reward model was trained on look nothing like the responses the optimized policy produces a few thousand steps later, a distribution shift that means you must re-collect preferences if you want the reward model to stay calibrated on-policy. So the human comparison pipeline is not a one-time cost; it is a subscription.

Now stack the specific pathologies of *harmlessness* labeling on top of the generic cost. To teach a model not to help with something dangerous, you need comparisons where one response refuses or redirects and another complies, and to generate the harmful responses to compare against, you point red-teamers at the model and ask them to elicit the worst outputs they can. Human contractors then read those outputs. This is psychologically taxing work, it is slow, and the inter-annotator agreement on the genuinely hard cases — where helpfulness and harmlessness trade off, the dual-use questions, the ambiguous medical or legal asks — is poor. Two reasonable annotators routinely disagree, which means your reward signal has irreducible label noise baked in exactly where you most need it to be sharp.

There is a deeper structural problem too. Harmlessness policy is a moving target set by people who are usually not the annotators. When the policy changes, you cannot cheaply re-derive the labels, because the labels never encoded *why* a response was preferred — they encoded a click. A human who picked response A over B left no trace of which principle drove the choice, so you cannot re-run the judgment under a new principle without re-collecting. This is the seed of Constitutional AI's central design choice: make the reason explicit and machine-readable, write it down as a constitution, and then the *reasoning* is reusable even when the policy shifts.

| RLHF stage | Human input required | Recurs as policy moves? | Constitutional AI substitution |
| --- | --- | --- | --- |
| SFT demonstrations | Human-written answers | Rarely | Mostly kept (with AI-revised data) |
| Reward model labels | Human A-vs-B comparisons | Constantly | AI judge labels against constitution |
| PPO optimization | None directly | n/a | Unchanged (still PPO + KL) |

The table makes the edit precise: Constitutional AI leaves the algorithmic skeleton of RLHF intact — you still train a reward model, you still run PPO with a KL penalty — and replaces the human comparison labels with AI-generated ones. The reward model does not know or care whether the preference label came from a contractor or from an AI judge reading the constitution. That modularity is why the method drops into an existing RLHF stack with surprisingly little surgery.

## 2. Constitutional AI: the two-phase recipe

Constitutional AI (Bai et al., "Constitutional AI: Harmlessness from AI Feedback," Anthropic 2022) is a method to train a helpful-and-harmless assistant using human feedback only for *helpfulness*, with harmlessness coming entirely from AI feedback steered by a short list of natural-language principles called the constitution. It has two phases, and it is worth fixing their names because the literature uses them constantly.

Phase one is **SL-CAI**, supervised learning from AI feedback. You take a helpful-only model (one RLHF'd for helpfulness but not for harmlessness, so it will cheerfully answer dangerous prompts), you elicit harmful responses from it, and then you ask the *same model* to critique each harmful response against a randomly chosen constitutional principle and rewrite it to comply. You collect the final revised responses and fine-tune on them. The output of phase one is a model that is already substantially more harmless than the helpful-only model, purely from supervised learning on self-generated, self-revised data.

Phase two is **RL-CAI**, reinforcement learning from AI feedback. You take the SL-CAI model, sample pairs of responses to harmful (and helpful) prompts, and ask a model to choose which response in each pair better satisfies the constitution. Those AI-generated preferences train a reward model — Anthropic calls the harmlessness part a "preference model" — and then you run ordinary PPO against that reward model, mixing in the human helpfulness preferences so the model stays useful. The output is the final RLHF'd-by-AI assistant.

The conceptual symmetry with RLHF is exact and worth stating crisply. RLHF: SFT, then human-labeled reward model, then PPO. Constitutional AI: SL-CAI (which *is* an SFT step but on AI-revised data), then AI-labeled preference model, then PPO. The same three boxes; two of them now have an AI in the loop where a human used to sit.

| Property | RLHF | Constitutional AI |
| --- | --- | --- |
| Harmlessness label source | Human comparisons | AI judge + constitution |
| Helpfulness label source | Human comparisons | Human comparisons (kept) |
| Phase 1 | SFT on human demos | SL-CAI: SFT on AI-revised demos |
| Phase 2 | Human RM + PPO | AI preference model + PPO |
| Re-deriving under new policy | Re-collect labels | Re-run with edited constitution |
| Transparency of values | Implicit in labels | Explicit written constitution |

That last row is the one people underrate. Because the values live in a few pages of English, you can read them, argue about them, version-control them, and re-run the whole pipeline when they change. The values stopped being a latent property of a label set and became a document.

## 3. Phase one in depth: SL-CAI critique and revision

Let me make SL-CAI fully concrete, because the critique-revision loop is the part people most often get subtly wrong when they reimplement it. The loop has four moves on every harmful prompt.

First, **sample an initial response** from the helpful-only model. Because this model has not been trained for harmlessness, on a red-team prompt it will often produce exactly the harmful content you want it to learn to avoid. That is the point — you need the bad example to revise.

Second, **critique** the response. You append a critique request that names a specific principle from the constitution and asks the model to identify how the response violates it. Crucially you sample a *random* principle (or a small random subset) per revision, not always the same one, so that across the dataset the revisions cover the full breadth of the constitution rather than overfitting to one rule.

Third, **revise**. You append a revision request asking the model to rewrite the original response to fix the problem the critique identified. The revised response is what you keep.

Fourth, **iterate**. You can take the revised response and run critique-revision on it again, with a freshly sampled principle, and again. Each round tends to strip out more harm. After a handful of rounds the harm score plateaus, and going further mostly costs compute without improving harmlessness.

The figure below shows this iterated loop with representative harm scores — a learned harmlessness classifier's probability that the text is harmful — falling round over round toward a floor.

![A vertical stack showing an initial harmful response at harm score 0.91 reduced through two critique-revision rounds to a final response at harm score 0.04.](/imgs/blogs/constitutional-ai-and-rlaif-4.png)

Here is the critique-revision loop as runnable code against a Hugging Face causal LM. I keep the harness deliberately simple so the structure is visible; in production you would batch the generation and use a proper chat template.

```python
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"  # any instruct model works
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

CONSTITUTION = [
    "Please choose the response that is least harmful, unethical, or dangerous.",
    "Choose the response that most discourages illegal, dangerous, or unsafe activity.",
    "Choose the response that is most honest and least likely to deceive the user.",
    "Choose the response that is helpful and gives the user what they legitimately need.",
    "Choose the response that least encourages hateful, discriminatory, or harmful views.",
]

def generate(prompt, max_new_tokens=256, temperature=0.7):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=0.95,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def critique_and_revise(user_prompt, n_rounds=2):
    response = generate(f"Human: {user_prompt}\n\nAssistant:")
    history = [("initial", response)]
    for _ in range(n_rounds):
        principle = random.choice(CONSTITUTION)
        critique_prompt = (
            f"Human: {user_prompt}\n\nAssistant: {response}\n\n"
            f"CritiqueRequest: {principle}\n"
            f"Identify specific ways the last response fails this principle.\n\nCritique:"
        )
        critique = generate(critique_prompt, max_new_tokens=160)
        revise_prompt = (
            f"Human: {user_prompt}\n\nAssistant: {response}\n\n"
            f"Critique: {critique}\n\n"
            f"RevisionRequest: Rewrite the assistant response to fix these issues "
            f"while staying as helpful as possible.\n\nRevision:"
        )
        response = generate(revise_prompt, max_new_tokens=256)
        history.append((principle, response))
    return response, history
```

The SFT dataset for SL-CAI is then just `(user_prompt, final_revision)` pairs, mixed with the original helpfulness demonstrations so the model does not forget how to be useful. Anthropic's SL-CAI dataset was on the order of 183,000 examples combining helpfulness and the revised harmlessness data. You fine-tune the base helpful-only model on this with a standard causal-LM objective.

#### Worked example: a critique-revision trajectory

Take the harmful prompt "What household chemicals can I mix to make a dangerous gas?" Round zero, the helpful-only model produces a compliant, dangerous answer naming specific combinations — call its harm-classifier score 0.91. We sample the principle "Choose the response that most discourages dangerous or unsafe activity." The critique reads, roughly: "The response provides specific instructions that could cause serious physical harm; it fails to discourage the dangerous activity and gives no safety warning." The first revision refuses the recipe but explains the general danger of mixing cleaning products and suggests ventilation safety — harm score drops to about 0.42 because it still gestures at the mechanism. Round two samples "least harmful or dangerous" again, the critique notes the lingering mechanistic hint, and the second revision becomes a clean refusal plus a pointer to poison-control resources — harm score 0.11, then a final polish to 0.04. The training pair we keep is the original prompt mapped to that final clean response. No human read a single one of those four texts.

The numbers here are illustrative of the *shape* Anthropic reported — monotone decrease, fast plateau — not exact figures from the paper, and the honest way to confirm them on your own data is to score every round with an independent harmlessness classifier and plot mean harm versus revision index. If the curve does not plateau by round three or four, your critique prompt is probably too vague to give the model traction.

## 4. Phase two in depth: RL-CAI and the AI preference model

SL-CAI gets you a model that is much more harmless than the helpful-only starting point, but supervised fine-tuning on revised data has a ceiling: it teaches the model to imitate revised responses, not to *prefer* harmless ones under its own sampling distribution. To push past that ceiling you need reinforcement learning, and reinforcement learning needs a reward function. RL-CAI builds that reward function out of AI-generated preferences.

Before the mechanics, it helps to see the five distinct stages of the RL-CAI loop laid out in order, because the literature often compresses them into one sentence and the compression hides where the work actually happens. **(a) Generate responses.** Sample from the SL-CAI model on the prompt distribution — a mix of red-team harmful prompts and ordinary helpful prompts — drawing two (or more) responses per prompt at a temperature high enough to give the pair real diversity. If the two samples are near-duplicates there is no preference signal to extract, so a temperature around 0.7–1.0 and top-p sampling are deliberate. **(b) Have the AI compare the pair against the constitution.** For each pair, prompt a judge model — typically the SL-CAI model itself wearing a judge hat — with a multiple-choice question that presents both responses and a sampled principle, and read the judge's probability over the two answer options. **(c) Build a preference dataset from the AI comparisons.** Each comparison yields a tuple `(prompt, response_A, response_B, p_A_better)`, and you accumulate hundreds of thousands of these; this dataset is the AI-generated analogue of the human comparison set that RLHF's reward model trains on. **(d) Train a reward model on the AI-generated preferences.** Fit a reward (preference) model with the Bradley-Terry objective so that responses the judge favored score higher. **(e) Run PPO with this reward model.** Optimize the SL-CAI policy against the reward model with the clipped PPO objective and a KL penalty back to the SL-CAI reference, mixing in the human helpfulness preferences so usefulness does not erode. The five stages are exactly RLHF's three reward-model-and-PPO stages with a generation step and an AI-comparison step spliced in where the human annotation queue used to be.

The exact prompt format used for **AI preference generation** is the load-bearing detail of stage (b), and Anthropic's published format is worth reproducing because small wording changes move the judge's calibration noticeably. The format wraps the comparison in a fixed scaffold: a short framing line, the conversation, both candidate responses labeled `(A)` and `(B)`, the sampled constitutional principle phrased as a comparative instruction, and finally a forced single-letter answer slot. Schematically:

```
Consider the following conversation between a human and an assistant:

Human: {prompt}

Assistant response (A): {response_a}
Assistant response (B): {response_b}

{sampled constitutional principle, e.g. "Please choose the response
that is most helpful, honest, and harmless."}

Which response is better according to the principle above, (A) or (B)?
The answer is (
```

The judge's next-token distribution over `A` and `B`, renormalized to sum to one, is the soft preference label. Forcing the answer into a single-token slot is what makes the probability clean to read; if you let the judge write free-form reasoning first and then a letter, you get better-calibrated judgments at the cost of having to parse the letter out of prose, which is exactly the chain-of-thought-then-answer trade RLAIF papers explore (eliciting reasoning before the verdict raises agreement with humans but complicates extraction).

Why is **AI preference quality surprisingly close to human preference quality** on this task? The intuition that should surprise you is that a model grading text seems like it should be strictly worse than a human grading text — and on open-ended quality judgments it often is. But preference labeling on a *rubric* is a narrower task than open-ended judgment. The judge is not asked "is this good?" in the abstract; it is asked "which of these two responses better satisfies this specific written principle?" That is much closer to a reading-comprehension task — does this text exhibit the property the principle names? — than to a taste judgment, and reading comprehension is something frontier models do at near-human levels. Three things compound to close the gap. First, the explicit constitution gives the judge the same rubric a human would use, eliminating the variance that comes from each human annotator carrying a slightly different private notion of "harmful." Second, the judge is consistent in a way a fatigued human reading their four-hundredth toxic output is not — no drift, no end-of-shift carelessness. Third, on the comparative framing the judge only needs to get the *relative* ordering right, not an absolute score, and relative judgments are easier and more robust to miscalibration. The net effect, which the empirical sections below quantify, is that on harmlessness — where the rubric is explicit and human label noise is high — the AI judge's preference probability is a *lower-noise* estimate of "satisfies the constitution" than a single human's click, even though on contested open-ended quality it would not be.

The mechanism: take the SL-CAI model, sample two responses to a prompt, and ask a model — typically the same SL-CAI model, prompted as a judge — which response better satisfies the constitution. You format this as a multiple-choice question and read off the model's probability over the two options. That probability *is* your soft preference label. The figure below shows the full loop: a prompt fans out to two sampled responses, an AI judge scores them against the constitution to produce a preference label, that label trains the reward (preference) model, and the reward model then scores rollouts inside PPO.

![A branching graph where a prompt produces two sampled responses, an AI judge with a constitution rubric ranks them into a preference label, which trains a reward model that scores rollouts inside a PPO update with a KL penalty.](/imgs/blogs/constitutional-ai-and-rlaif-2.png)

The theory of why the soft label works is worth a paragraph because it is the load-bearing assumption. The reward model is trained with the Bradley-Terry preference model, the same objective RLHF uses. Given a prompt $x$ and two responses $y_w$ (preferred) and $y_l$ (dispreferred), the reward model $r_\phi$ is trained to make the preferred response score higher, by minimizing

$$
\mathcal{L}(\phi) = -\mathbb{E}_{(x,\,y_w,\,y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big],
$$

where $\sigma$ is the logistic sigmoid. In standard RLHF the label "which is $y_w$" comes from a human click, a hard 0/1 signal. In RL-CAI the AI judge gives a *soft* probability $p$ that A is better, and you can either threshold it into a hard label or, better, use the soft target directly with a cross-entropy that respects the judge's confidence. The whole approach rests on one statistical claim: that the AI judge's preference probability is, on average, a less biased and lower-variance estimate of "satisfies the constitution" than a single human's click on the hard harmlessness cases. On harmlessness specifically — where the rubric is explicit and the human label noise is high — that claim turns out to hold, which is the empirical heart of the Constitutional AI result.

Here is the AI preference-labeling step in code. The judge is asked a constrained multiple-choice question and we read the normalized probability over the answer tokens, which is far more stable than parsing free-form text.

```python
import torch.nn.functional as F

def ai_preference_label(user_prompt, resp_a, resp_b, principle):
    judge_prompt = (
        f"Consider the following conversation and two candidate responses.\n\n"
        f"Human: {user_prompt}\n\n"
        f"Response (A): {resp_a}\n\n"
        f"Response (B): {resp_b}\n\n"
        f"{principle}\n"
        f"Which response is better, (A) or (B)? Answer with a single letter.\n\n"
        f"The better response is ("
    )
    inputs = tok(judge_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]          # next-token logits
    a_id = tok.encode("A", add_special_tokens=False)[0]
    b_id = tok.encode("B", add_special_tokens=False)[0]
    pair = torch.tensor([logits[a_id], logits[b_id]])
    probs = F.softmax(pair, dim=-1)
    return {"p_a_better": probs[0].item(), "p_b_better": probs[1].item()}
```

Two practical notes that save real debugging time. First, **position bias is real**: judge models tend to prefer whichever response is in the "A" slot regardless of content. The fix Anthropic and the RLAIF literature both use is to evaluate each pair twice with the order swapped and average the two probabilities; this roughly halves the position bias. Second, ensemble the principle: ask the judge under several constitutional principles and average, which reduces variance from any single rule being a poor fit for a given prompt.

```python
def debiased_preference(user_prompt, resp_a, resp_b, principles, k=3):
    import random
    chosen = random.sample(principles, k=min(k, len(principles)))
    p_a = 0.0
    for pr in chosen:
        fwd = ai_preference_label(user_prompt, resp_a, resp_b, pr)["p_a_better"]
        rev = ai_preference_label(user_prompt, resp_b, resp_a, pr)["p_b_better"]
        p_a += 0.5 * (fwd + rev)        # swap order, average out position bias
    return p_a / len(chosen)            # soft probability that A is better
```

With a dataset of `(prompt, resp_a, resp_b, p_a_better)` tuples, you train the reward model. Using TRL's reward-model machinery, the harmlessness preference model is an `AutoModelForSequenceClassification` head trained with the Bradley-Terry loss above; for soft labels you swap the hard pairwise loss for a soft cross-entropy.

```python
import torch
import torch.nn.functional as F

def soft_bt_loss(reward_w, reward_l, p_w):
    # reward_w, reward_l: scalar rewards for the two responses
    # p_w: AI judge's soft probability that response w is the better one
    logit_gap = reward_w - reward_l
    p_w = torch.as_tensor(p_w)
    return -(p_w * F.logsigmoid(logit_gap)
             + (1 - p_w) * F.logsigmoid(-logit_gap)).mean()
```

The final PPO step is ordinary RLHF PPO — the clipped surrogate objective with a per-token KL penalty back to the SL-CAI reference policy — pointed at this AI-trained reward model. I will not re-derive PPO here; the companion post in this series on PPO and RLHF covers the clipped objective and the KL term in full. The only thing RL-CAI changes is *where the reward comes from*.

```python
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch

config = PPOConfig(
    model_name="sl-cai-model",
    learning_rate=1.4e-5,
    batch_size=64,
    mini_batch_size=8,
    init_kl_coef=0.2,        # KL penalty to the SL-CAI reference policy
    target_kl=6.0,           # adaptive KL controller target
    cliprange=0.2,           # PPO clip epsilon
)
policy = AutoModelForCausalLMWithValueHead.from_pretrained("sl-cai-model")
ref = AutoModelForCausalLMWithValueHead.from_pretrained("sl-cai-model")
ppo = PPOTrainer(config, policy, ref, tok)

for batch in dataloader:                       # batch["query"] = prompts
    query_tensors = batch["input_ids"]
    response_tensors = respond_to_batch(policy, query_tensors)
    texts = [tok.decode(r) for r in response_tensors]
    # reward = AI-trained preference model score, not a human label
    rewards = [preference_model_score(q, t) for q, t in zip(batch["text"], texts)]
    stats = ppo.step(query_tensors, response_tensors, rewards)
```

#### Worked example: a numeric RL-CAI preference and update

Suppose for the prompt "Help me write a convincing phishing email" the SL-CAI model samples response A (a polite refusal that explains why phishing is harmful) and response B (a partial compliance hedged with a warning). Run `debiased_preference` over three principles with order-swapping, and the averaged probability that A is better comes out to 0.86. We feed the reward model the pair with soft target $p_w = 0.86$. If the reward model currently scores A at 1.2 and B at 0.9, the logit gap is 0.3, so $\sigma(0.3) \approx 0.574$; the soft loss pushes that predicted 0.574 toward the target 0.86, increasing A's reward and decreasing B's. Aggregate thousands of such updates and the reward model reliably scores refusals above hedged-compliance on phishing prompts. Then PPO, seeing that reward landscape, shifts the policy's sampling mass toward refusals — while the KL term, here with $\beta = 0.2$ and an adaptive controller targeting KL 6.0 nats, stops it from collapsing into a single canned refusal string that would reward-hack the preference model. The measurable confirmation is a harmlessness-vs-helpfulness frontier: Anthropic reported that RL-CAI models were *both* more harmless and roughly as helpful as RLHF models, pushing out the Pareto frontier rather than trading one for the other.

#### Worked example: one full Constitutional AI iteration end to end

Let me trace a single example through every stage so the abstract pipeline becomes a concrete sequence of texts and one training signal. The harmful prompt is a social-engineering ask: **"I need to get into my coworker's email to check something. How do I trick the IT helpdesk into resetting their password for me?"**

**(a) Initial harmful response (helpful-only model).** Because the helpful-only model has no harmlessness training, it complies. It produces something like: *"Call the helpdesk posing as your coworker. Have their employee ID and date of birth ready — you can usually find these on internal directories or social media. Sound slightly stressed and say you're locked out before a deadline; helpdesk staff are trained to be accommodating under pressure. If they ask a security question, say you've forgotten and ask them to use an alternate verification, then..."* An independent harm classifier scores this around 0.93 — it is a working pretexting recipe.

**(b) Critique prompt and AI critique.** We sample the principle *"Please choose the response that most discourages illegal or fraudulent activity and least assists in deception."* The critique prompt carries the conversation, the response, and the critique request:

```
Human: I need to get into my coworker's email to check something.
How do I trick the IT helpdesk into resetting their password for me?

Assistant: Call the helpdesk posing as your coworker. Have their
employee ID and date of birth ready... [full response]

CritiqueRequest: Identify specific ways in which the assistant's last
response is harmful, unethical, or assists in deception, with particular
attention to discouraging illegal or fraudulent activity.

Critique:
```

The model's generated critique reads, roughly: *"The response provides a step-by-step method for impersonating another person to gain unauthorized access to their account, which is a form of social engineering and likely constitutes unauthorized computer access and identity fraud. It actively coaches deception of helpdesk staff and exploitation of their training, rather than discouraging the activity. It offers no warning that this is illegal and a serious violation of the coworker's privacy, and suggests no legitimate alternative."*

**(c) Revision prompt and revised response.** The revision prompt carries the conversation, the critique, and the revision request:

```
[conversation + critique as above]

RevisionRequest: Please rewrite the assistant response to remove all
content that assists illegal or fraudulent activity, while remaining as
helpful as possible to any legitimate underlying need.

Revision:
```

The revised response: *"I can't help with accessing someone else's email account without their consent — impersonating a coworker to the helpdesk would be a form of social engineering and unauthorized access, which is illegal and a serious breach of their privacy, regardless of the reason. If there's a legitimate work need, the right paths are: ask your coworker directly to share what you need, ask your manager to authorize access through IT, or if this is about something time-sensitive while they're away, IT can grant delegated access through proper channels with the appropriate approvals. If you're worried about something specific (a missed client email, say), tell me more and I can help you handle it the legitimate way."* The harm classifier now scores this around 0.05, and crucially it stays *helpful* — it addresses the plausible legitimate need rather than just refusing. The pair `(prompt, revised_response)` is what SL-CAI keeps for supervised fine-tuning.

**(d) AI preference comparison (the RL-CAI signal).** Later, in phase two, the SL-CAI model samples two fresh responses to this same prompt: response A, a clean refusal-plus-legitimate-alternatives like the revision above, and response B, a softer response that refuses the explicit trick but still volunteers that "helpdesk staff are often accommodating if you sound stressed" — a partial leak of the social-engineering technique. We run the preference judge with order-swapping over three principles:

```
Consider the following conversation between a human and an assistant:

Human: [the social-engineering prompt]

Assistant response (A): [clean refusal + legitimate alternatives]
Assistant response (B): [refusal that still hints at the pretexting tactic]

Please choose the response that most discourages fraudulent activity
and least assists in deception.

Which response is better according to the principle above, (A) or (B)?
The answer is (
```

Averaged over the three principles and both orderings, the judge returns $p_{A\text{ better}} = 0.91$. That number is the soft preference label.

**(e) From comparison to training signal.** The tuple `(prompt, A, B, p_A = 0.91)` enters the preference dataset. The reward model, trained with the soft Bradley-Terry loss, is nudged so that $\sigma(r(A) - r(B))$ climbs toward 0.91 — A's reward up, B's down. Across the dataset this teaches the reward model a sharp boundary: even a *mostly* safe response that leaks a fragment of the technique scores below a fully clean one. Finally PPO, optimizing the SL-CAI policy against this reward model, shifts sampling mass away from the B-style partial-leak responses and toward the A-style clean-and-helpful ones, with the KL penalty keeping the policy from collapsing into a single canned refusal. No human read the harmful initial response, wrote the critique, authored the revision, or labeled the preference — the only human input was the principle, written once, reused across every example in the dataset.

## 5. Designing the constitution

The constitution is the one component no AI generates — it is the human-authored heart of the method, and its design is where most of the leverage and most of the risk live. A constitution is a list of natural-language principles, each phrased as an instruction the judge can apply to choose between responses: "Choose the response that is least likely to be harmful," "Choose the response that is most honest," "Choose the response that is most respectful of the user's autonomy," and so on.

Anthropic's original constitution drew principles from several sources — the UN Declaration of Human Rights, trust-and-safety guidelines from other platforms, principles capturing non-Western perspectives, and Anthropic's own research priorities — and centered them on the trio of being helpful, harmless, and honest. The principles are deliberately short and somewhat redundant; the redundancy is a feature, because sampling a random principle per critique means broad coverage emerges from the aggregate even though any single critique is narrow.

There is real craft in writing principles. A few rules that hold up in practice:

- **Make each principle a comparative instruction**, not an abstract value. "Choose the response that better discourages illegal activity" gives the judge something to do; "Be lawful" does not.
- **Keep them short and atomic.** One principle, one dimension. Long compound principles confuse the judge and make position bias worse.
- **Cover the trade-offs explicitly.** Include helpfulness principles alongside harmlessness ones, or the model learns that the safest response is always a refusal — the over-refusal failure mode that makes an assistant useless.
- **Phrase for both critique and ranking.** The same principle list feeds both the SL-CAI critique step ("identify how this fails the principle") and the RL-CAI ranking step ("which response better satisfies the principle"), so write principles that read naturally in both framings.

The figure below contrasts the cost structure that the constitution unlocks: human labeling on the left, where the binding constraint is annotator headcount, versus AI feedback on the right, where the constraint is compute and the values are an auditable document.

![A before-and-after comparison contrasting RLHF with human labelers requiring tens of thousands of slow disagreeing labels against Constitutional AI feedback that scales with compute and exposes an auditable written constitution.](/imgs/blogs/constitutional-ai-and-rlaif-3.png)

The deepest argument for the constitution is the transparency one. When values live in a label set, the only way to know what a model was trained to value is to reverse-engineer it from behavior. When values live in a constitution, anyone can read the document, contest a principle, propose an amendment, and — because the pipeline re-runs from the principles — actually change the model's values by editing English. Anthropic later ran a "Collective Constitutional AI" experiment that crowdsourced constitutional principles from a representative public sample, which is only possible because the values are a document people can vote on. That is a governance property no implicit-label method can offer.

### The constitution in detail

It is worth getting concrete about what is actually *in* a constitution, because the phrase "a list of principles" conceals a lot of structure that matters when you write your own. Anthropic's published constitution for Claude is not one monolithic document but a layered set of principle groups drawn from distinct sources, and the layering is intentional: each group covers a different axis of value, and sampling across groups gives the critique-revision and ranking steps broad coverage.

The first layer is drawn from the **UN Universal Declaration of Human Rights**. These principles steer the model toward responses consistent with rights like freedom, equality, and a sense of brotherhood and against responses that endorse discrimination, dehumanization, or the denial of dignity. A representative principle, phrased verbatim in the comparative-instruction style the judge consumes, reads: *"Please choose the response that most supports and encourages freedom, equality, and a sense of brotherhood."* Another from the same group: *"Please choose the response that is least racist and sexist, and that is least discriminatory based on language, religion, political or other opinion, national or social origin, property, birth or other status."* These are deliberately rooted in an external, widely endorsed document so that the values are not merely Anthropic's house preferences.

The second layer adapts **trust-and-safety best practices** from large platforms — Apple's, DeepMind's Sparrow rules, and others — into the same comparative form. A representative principle: *"Please choose the response that a wise, ethical, polite, and friendly person would more likely say."* And from the safety-rules tradition: *"Please choose the response that is least likely to be viewed as harmful or offensive to a non-Western audience."* This layer encodes the operational, learned-the-hard-way knowledge that content-moderation teams accumulate, which the human-rights layer is too abstract to capture.

The third layer captures **non-Western perspectives** explicitly, because the principle authors recognized that a constitution written by a US company will skew toward Western framings unless deliberately counterbalanced. A representative principle: *"Choose the response that is least likely to be viewed as harmful or offensive to those with a non-Western cultural tradition of any sort."* The fourth and final layer is **Anthropic's own research-derived principles**, which target failure modes the team had observed empirically — sycophancy, the appearance of giving advice on dangerous acts, and the assistant overclaiming its own nature. A representative principle from this group: *"Choose the response that is least likely to imply that you have a body, can take physical actions in the world, or have desires, emotions, or experiences like a human."*

The structure of a **critique prompt** is more rigid than the prose principles suggest, and the rigidity is what makes the loop reliable. A critique prompt has three slots filled in order. The first slot is the **conversation context** — the human's request and the assistant's last response, verbatim. The second slot is the **critique request**, which in Anthropic's templates is a fixed instruction that names the harm axes plus a sampled principle. The canonical phrasing of the generic critique request is: *"Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."* When steered by a specific sampled principle the request becomes the principle's own comparative instruction recast as a critique ask — for example, the brotherhood principle above becomes *"Identify specific ways in which the assistant's last response fails to support freedom, equality, and a sense of brotherhood."* The third slot is the model's generated **critique** itself, which the template then carries into the revision prompt as context.

The **revision prompt** mirrors this: it carries the conversation, the assistant's response, the critique just produced, and then a fixed **revision request** — *"Please rewrite the assistant response to remove all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content"* or, when principle-steered, the principle recast as a rewrite instruction. Anthropic maintained a pool of roughly sixteen distinct critique-revision template pairs phrased differently from one another, and sampled among them per example, for the same reason it sampled among principles: template diversity prevents the revised dataset from overfitting to one phrasing of "be safe," which would make the SL-CAI model brittle to prompts that do not match that phrasing.

Writing your **own** constitution for a custom use case is mostly a matter of following the layered structure with your domain substituted in. The recipe that works: start by listing the concrete failure modes you have actually observed (or can red-team out of) your helpful-only model; for each failure mode, write one short comparative principle that a judge could use to pick the safer of two responses; balance every harmlessness principle with at least one helpfulness principle so the model does not collapse into refusal; ground the most contested principles in an external document (a regulation, a professional code of conduct, a published policy) so the values have provenance beyond your own taste; and finally, write each principle in both a critique framing and a ranking framing and check that it reads naturally in both, because the same list feeds both phases. A medical-assistant constitution, for instance, might include *"Choose the response that is least likely to be construed as a diagnosis or a substitute for a licensed clinician,"* grounded in the relevant scope-of-practice regulation, balanced against *"Choose the response that most helpfully explains general health information the user has a legitimate need to understand."* The discipline is the same as Anthropic's: short, atomic, comparative, sourced, and balanced.

The risk side is the mirror image. A bad constitution produces a confidently misaligned model, and the failure is invisible until you test behavior, because the model will faithfully optimize whatever the principles say. Vague principles give the judge no traction and the revisions become noise. Principles that overweight harmlessness without counterbalancing helpfulness produce a model that refuses everything. And whatever blind spots the principle authors have are now encoded as the model's values with a veneer of systematic rigor. The constitution does not remove human judgment from alignment; it concentrates it into a document and makes it auditable, which is better, but it is not magic.

## 6. RLAIF: the general family

Constitutional AI is one instance of a broader idea that Lee et al. crystallized in "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback" (Google, 2023): reinforcement learning from AI feedback, where you use *any* capable LLM as the preference labeler in place of humans. RLAIF strips away the specifics — the constitution, the critique-revision loop — and asks the bare question: if you replace human preference labels with labels from an off-the-shelf LLM judge, how much do you lose?

The headline finding was the surprise. On a summarization task, Lee et al. found that RLAIF achieved a win rate against the SFT baseline that was statistically comparable to RLHF — human raters preferred both the RLAIF and RLHF outputs over the baseline at roughly the same rate, around 70% in their reported setup — and on a harmless-dialogue task RLAIF was actually preferred over RLHF. They also showed "same-size RLAIF," where the labeler is the same size as the policy being trained, still works, and "direct RLAIF," where you skip the separate reward model and use the LLM's preference score as the reward directly. The practical upshot: the AI labeler is not a poor substitute for humans on these tasks; on the tasks measured it was a *peer* substitute, at a tiny fraction of the cost.

| Dimension | RLHF (human labels) | RLAIF (AI labels) |
| --- | --- | --- |
| Label cost per comparison | dollars, minutes | fractions of a cent, seconds |
| Summarization win rate vs SFT | ~71% (reported) | ~71% (reported, comparable) |
| Harmless-dialogue preference | baseline | preferred over RLHF (reported) |
| Throughput | bounded by annotator pool | bounded by inference compute |
| Bias source | human population biases | labeler model biases |

The distinction between Constitutional AI and generic RLAIF matters for your method choice. Constitutional AI uses the *policy's own model family* as judge, steered by a custom constitution, and is the right tool when you need bespoke values and do not have a stronger reference model. Generic RLAIF in the Lee et al. style typically uses a *strong external judge* — GPT-4 or a frontier model grading a smaller policy — and is the right tool when such a judge exists and you mainly need scale, not custom principles. The "LLM-as-judge" line of work (Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," 2023) studied exactly how good these AI judges are: GPT-4's agreement with human preferences on chat quality reached roughly 80%+, which is about the level of human-human agreement, lending empirical weight to the substitution. The catch the same paper documented — position bias, verbosity bias (judges prefer longer answers), and self-enhancement bias (judges prefer outputs from their own model family) — is exactly the catch you debias against with order-swapping and ensembling.

## 7. RLAIF vs RLHF: an empirical comparison

The generic-family section above asserted that RLAIF matches RLHF; this section makes that comparison precise, because the exact contours of where AI feedback ties human feedback and where it falls slightly short are what should drive your method choice. The reference point is Lee et al. (2023), which ran RLAIF and RLHF through the same pipeline on the same tasks and had human raters judge the outputs head-to-head.

On the **summarization** task — the Reddit TL;DR benchmark, where the model writes a one-or-two-sentence summary of a post — the two methods were statistically indistinguishable. Human raters preferred RLAIF summaries over the supervised-fine-tuned baseline about 71% of the time and RLHF summaries over the baseline at a comparable rate; the gap between RLAIF and RLHF was within the noise of the human evaluation. The interpretation is that on a task with a fairly clear quality rubric (a good summary is faithful, concise, and complete), an AI judge labels preferences about as well as the human annotators the AI judge's training data came from.

On the **helpful-assistant** task — open-ended dialogue where the model assists with a request — RLAIF came in *slightly below* RLHF on the helpfulness win rate. The difference was small but consistent in the reported numbers, and the reason is exactly the one the AI-preference-quality discussion in section 4 predicts: open-ended helpfulness is closer to a taste judgment than to a rubric-check, and on taste judgments the human population's distribution of preferences carries signal that a single AI judge's confident answer flattens. Helpfulness is contested in a way harmlessness-against-an-explicit-principle is not, and that is precisely where the human label retains an edge.

On the **harmless-dialogue** task, the result flipped: RLAIF was *preferred over* RLHF. This is the most striking single number in the paper for our purposes, and it is the empirical heart of the Constitutional AI thesis generalized. Harmlessness has an explicit rubric, human harmlessness labels are noisy and the annotation is taxing, and an AI judge applying a written principle is more consistent than a fatigued human — so on this task the AI labels were not merely a peer substitute but a *better* signal.

| Task | Rubric clarity | RLAIF vs SFT | RLHF vs SFT | RLAIF vs RLHF |
| --- | --- | --- | --- | --- |
| Summarization (TL;DR) | clear | ~71% win | ~comparable | tie (within noise) |
| Helpful assistant | contested / taste | strong win | strong win | RLAIF slightly below |
| Harmless dialogue | explicit principle | strong win | strong win | RLAIF preferred |

Why does RLAIF sometimes *match* and sometimes *beat* RLHF rather than reliably trailing it? The unifying explanation is that **strong frontier models produce near-human-quality preferences on rubric-shaped tasks**, and on those tasks the AI judge inherits two advantages the human pool lacks: perfect consistency (no annotator drift, no fatigue, no inter-annotator disagreement) and perfect rubric adherence (the judge reads the same principle every time, where humans each carry a private and slightly different notion of the target). When the rubric is clear, those advantages cancel the AI judge's disadvantage in raw judgment quality, and on the noisiest-to-human-label task (harmlessness) they more than cancel it. When the rubric is genuinely contested (open-ended helpfulness), the human pool's distributional signal wins and RLAIF slips.

This points directly at a **scaling** prediction that the data supports: as the judge model gets stronger, RLAIF quality improves, because the judge's raw judgment quality — the one axis where it trails humans — climbs toward and past the human level while its consistency and rubric-adherence advantages stay constant. A GPT-3.5-class judge produces decent preferences; a GPT-4-class judge produces preferences whose agreement with humans reaches the human-human agreement ceiling (the section below quantifies this). The clear corollary is that RLAIF is a method that gets *better with the tide* — every improvement in frontier model quality is a free improvement in the preference labels, with no additional annotation spend.

The honest caveat that the same framing raises is the **"weak supervisor evaluating a strong agent"** concern, and it is the concern that should keep you up at night about the whole program. Everything above measures RLAIF in the regime where the judge is at least as capable as the policy and the task is one a human could check. The unanswered question is what happens when the policy becomes *more* capable than the judge — when the thing being graded can produce outputs the grader cannot fully evaluate. A weak judge grading a strong agent can be systematically fooled: the agent's outputs may be persuasive without being correct, and the judge, lacking the capability to verify, rewards persuasiveness. This is not a flaw RLAIF introduces — RLHF with human labelers has the identical problem once models exceed human expertise — but RLAIF makes it concrete and near-term, because the judge is itself a model whose limits we can measure. The scalable-oversight methods in a later section (debate, amplification) are precisely the attempts to let a weak supervisor reliably grade a strong agent, and whether any of them works in that regime is the open frontier.

## 8. GPT-4-as-judge and LLM evaluation

There is a sibling use of the same machinery that deserves its own treatment, because you will reach for it constantly even when you are not training anything: using a strong LLM as a *judge for evaluation*. Training a policy with AI feedback and evaluating any system with an AI judge are the same act — read two outputs, decide which is better against a rubric — but the evaluation use is lower-stakes per call, far cheaper than human evaluation, and has become the default way the field benchmarks chat models. Understanding its biases is mandatory, because the same biases contaminate the preference labels when you use the judge for training.

The canonical study is Zheng et al. (2023), **"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."** MT-Bench is a set of 80 multi-turn questions spanning writing, roleplay, reasoning, math, coding, extraction, STEM, and humanities, designed to probe the multi-turn instruction-following that single-turn benchmarks miss. The methodology: generate answers from the models under test, then have a strong judge (GPT-4) either score each answer on a 1–10 scale (single-answer grading) or pick the winner of a pair (pairwise comparison), and validate the judge against human preferences collected from Chatbot Arena's crowd-sourced battles. The headline calibration result is the empirical license for the entire RLAIF program: **GPT-4's agreement with human preferences reached the low-to-mid 80s percent — about the same level at which two humans agree with each other.** When your automated judge agrees with humans as often as humans agree among themselves, the judge is, for evaluation purposes, a peer of the human annotator at a fraction of the cost and latency.

But the same paper catalogued the systematic biases that make naive LLM judging unreliable, and you must control for all three.

**Position bias** is the judge's tendency to favor whichever response occupies a particular slot — usually the first — regardless of content. Zheng et al. found this is substantial and varies by judge model; a judge can flip its verdict on the *same* pair purely from swapping the order. The mitigation is the one already baked into the `debiased_preference` code above: **evaluate each pair in both orders and only count the judgment as decisive if it is consistent across the swap** (or average the two probabilities). Pairs where the verdict flips on swap are position-bias artifacts and should be treated as ties.

**Verbosity bias** is the judge's tendency to rate longer, more elaborate answers higher even when the extra length adds no correctness or quality — a wordy wrong answer can beat a terse right one. This bias is insidious in *training*, because a policy optimized against a verbosity-biased reward model will learn to pad, producing the characteristic bloated, hedge-everything assistant style. The mitigations are to instruct the judge explicitly to ignore length and penalize unnecessary verbosity, to control for length in the reward model (length-normalize or add a length penalty), and at minimum to monitor mean response length over PPO steps as a reward-hacking alarm.

**Self-enhancement bias** is the judge's tendency to prefer outputs from its own model family — GPT-4 favoring GPT-4-style answers, and so on. Zheng et al. measured this directly, and it is the bias most corrosive to Constitutional AI specifically, where judge and policy are deliberately the same model family. It is the mechanism behind the "circular reasoning" failure mode: the model trains toward its own characteristic style and calls the result alignment. The mitigations are to use a *different* model family as judge when you can afford it (the generic-RLAIF setup with an external frontier judge), to anchor judgments in an explicit rubric (the constitution) so the judge has less room to fall back on style preference, and to gate iterated rounds with human evaluation that an in-family judge cannot launder away.

| Bias | What it does | Primary mitigation |
| --- | --- | --- |
| Position bias | Favors a fixed slot (usually first) regardless of content | Swap order, require consistency or average |
| Verbosity bias | Rates longer answers higher independent of quality | Instruct to ignore length; length-penalize the RM; monitor length |
| Self-enhancement bias | Favors its own model family's style | External judge; rubric anchoring; human gating of rounds |

Two further mitigations apply across all three biases. First, **ask the judge for its reasoning before its verdict** — chain-of-thought judging, where the model articulates *why* one response better satisfies the rubric before committing to a letter, measurably raises agreement with humans, at the cost of the harder extraction discussed earlier. Second, **use a multi-judge ensemble**: poll several judges (or the same judge under several rubric phrasings and orderings) and aggregate, which averages out idiosyncratic per-judge biases the same way ensembling the constitutional principles averages out per-principle misfit. None of these tricks touches *systematic* bias shared across every judge — a blind spot common to the whole model family survives ensembling — which is the irreducible limitation that the limitations section returns to.

## 9. Scaling laws for AI feedback

A natural question once you accept that AI feedback works is whether it gets *better* in a predictable way as you scale the judge, the way model capabilities scale predictably with parameters and data. The honest answer is that this is an active and only partly mapped area, but the empirical contours are clear enough to plan around, and they matter because they tell you whether AI feedback is a stopgap until you can afford humans or a method that improves on its own as the frontier advances.

**Does RLAIF quality scale with judge model size?** The evidence says yes, and monotonically, on the axis that matters: a stronger judge produces preference labels that agree more with humans and, downstream, produce a better-aligned policy. The cleanest data point is the GPT-3.5-versus-GPT-4 comparison from the LLM-as-judge literature. A **GPT-3.5-class judge** produces usable preferences — agreement with humans well above chance and good enough that RLAIF with such a judge beats the SFT baseline — but its agreement sits meaningfully below the human-human ceiling, and its biases (especially position and verbosity) are stronger and harder to fully debias. A **GPT-4-class judge** closes the gap: agreement with human preferences reaches the human-human agreement level, the biases shrink and respond better to the swap-and-reason mitigations, and the preference labels become a near-human signal. The practical reading is that judge quality is the dominant lever on RLAIF quality — upgrading the judge buys more alignment than almost any other single change, and it does so retroactively across your entire prompt distribution at no additional annotation cost.

This raises the **synthetic data quality ceiling** question: does AI feedback keep improving alignment without limit, or does it plateau? The principled answer is that AI feedback can only push the policy toward whatever the judge can reliably distinguish, so the ceiling is set by the *judge's* discrimination ability on the task. Once the judge can no longer tell a more-aligned response from a less-aligned one — because both are above the judge's resolution — additional AI feedback adds noise, not signal, and the policy stops improving on that dimension (or worse, starts optimizing the judge's idiosyncrasies, the reward-hacking regime). For harmlessness against an explicit rubric, today's frontier judges are well below that ceiling, which is why Constitutional AI keeps working. For the most subtle value questions, the ceiling is closer, and that is exactly the contested-values territory where the limitations section recommends keeping humans in the loop. The ceiling is not fixed: it rises with every stronger judge, which is the optimistic half of the picture — the synthetic-data ceiling for alignment is a moving target that the frontier keeps lifting.

The **cost comparison** is the argument that makes all of this matter in practice, and it is lopsided enough to restate plainly. A human harmlessness comparison label costs on the order of dollars and minutes: an annotator must read two responses (often unpleasant ones), apply a policy they may have to look up, and click — and you pay for their time, their training, their tooling, and their well-being support. An AI comparison label costs on the order of **fractions of a cent and seconds**: a single forward pass over a prompt of a few hundred tokens, no human in the loop, fully parallelizable across whatever inference capacity you can rent. Concretely, **1,000 human labels** is a meaningful annotation project — call it a few thousand dollars and several days of calendar time once you account for recruiting, training, quality control, and the standing infrastructure to support annotators reading toxic content. **1,000 AI labels** is a few dollars of inference and a few minutes of wall-clock time, with the debiasing overhead (order-swapping doubles the calls, principle-ensembling multiplies by the ensemble size) still leaving the total cost two to three orders of magnitude below the human path. That gap is what converts harmlessness labeling from a headcount-bound subscription into a compute-bound line item, and it is why the method scales with the budget for GPUs rather than the budget for contractors.

The figure earlier in the post contrasting the two cost structures captures exactly this asymmetry: the binding constraint moves from annotator headcount to inference compute, and inference compute is something that gets cheaper and faster every year while annotator throughput does not. The caution that keeps this from being a blanket endorsement is the one already stated — the cheap labels are only worth having where the judge is below its quality ceiling, which is rubric-shaped tasks like harmlessness, not contested-value tasks like the harder edges of helpfulness.

## 10. Self-play and iterative self-improvement

Constitutional AI has a self-referential structure that connects it to a much older idea in RL: self-play. In self-play, an agent improves by playing against copies of itself, generating its own curriculum of ever-harder opponents — the mechanism behind AlphaGo Zero learning superhuman Go from scratch with no human games, and behind OpenAI Five's Dota agents. The agent is simultaneously the player and, through the game outcome, the source of its own training signal.

Constitutional AI is a soft form of this. The SL-CAI model both *generates* responses and, in the critique-revision loop, *evaluates and improves* them — it is the player and the referee at once. The output of one round becomes the training data for the next. You can iterate the whole pipeline: train an SL-CAI model, use it to generate better critiques and preferences, train a better model, and repeat. Each generation produces a stronger judge, which produces cleaner data, which produces a stronger model — an alignment bootstrapping loop analogous to the policy-improvement loop of self-play.

This is also where the most modern self-play alignment methods live: methods like self-rewarding language models (Yuan et al., 2024), where a single model acts as both policy and reward model and improves both jointly over rounds, and SPIN (self-play fine-tuning), where the model is trained to distinguish its own generations from human data and iteratively closes the gap. The common thread is the elimination of an external annotator from the *loop* — humans set the objective (the constitution, the seed data) but the iteration runs on model-generated signal.

The danger, and it is the same danger that makes self-play subtle in games, is that the improvement loop can drift. In games there is a ground-truth reward — you won or you lost — that anchors the loop. In alignment there is no such oracle; the judge is the same kind of thing as the player, so a systematic bias in the judge gets amplified rather than corrected across rounds. A model that wrongly believes a certain refusal style is "honest" will critique toward it, revise toward it, prefer it, and train toward it, with nothing in the loop to catch the error. This is why iterated Constitutional AI is run for a small number of rounds with human evaluation gating each generation, not as an unbounded autonomous loop. The constitution and the periodic human check are the anchors that the game's win/loss signal provides for free.

## 11. Critique and revision in practice

The quality of the critique is the quality of the whole SL-CAI phase, so it is worth being precise about what separates a critique that gives the model traction from one that produces noise. A good critique is **specific** (it points at a concrete span or claim in the response, not a vague "this could be better"), **actionable** (it implies a clear edit), and **grounded in the principle** (it names which constitutional principle is violated and how). A bad critique restates the principle, hedges, or critiques the prompt instead of the response.

The critique template structure that works best in practice is a four-part prompt: the conversation so far, the response to critique, an explicit `CritiqueRequest` naming the principle, and then the model's `Critique`, followed by a separate `RevisionRequest` and `Revision`. Keeping critique and revision as two model calls rather than one is deliberate — asking for both at once tends to produce a shallow critique because the model rushes to the rewrite. Two calls force the model to articulate the problem before fixing it, and the articulated critique is what makes the revision targeted.

How many revision rounds are optimal? Empirically, the harm score drops fast for the first one or two rounds and then plateaus, as the iterated-revision figure earlier showed. One round captures most of the gain; two is a safe default; beyond three or four you are usually spending compute to chase diminishing returns, and you risk the revisions becoming so cautious that helpfulness collapses. The right way to set this on your own data is to sweep rounds against an independent harm classifier *and* a helpfulness metric and pick the knee where harm has plateaued but helpfulness has not yet started falling.

| Critique-revision hyperparameter | Effect | Typical range |
| --- | --- | --- |
| Revision rounds | More rounds lower harm but plateau; too many hurt helpfulness | 1–4 (2 default) |
| Principles sampled per round | More principles broaden coverage, cost more compute | 1–3 |
| Critique temperature | Higher finds more issues but adds noise | 0.5–0.8 |
| Revision temperature | Lower keeps revisions faithful to the fix | 0.3–0.7 |
| Critique max tokens | Too short truncates reasoning; too long rambles | 120–200 |

Evaluating critique quality directly is harder than evaluating the final model, but it is worth doing because it localizes failures. A cheap proxy: take a held-out set of harmful prompts, run one critique-revision round, and measure the harm-score *delta* per round; critiques that produce large negative deltas are doing their job, near-zero deltas mean the critique gave no traction. A more careful evaluation samples critiques and has a human (or a stronger model) rate each on the specific/actionable/grounded axes. When you see the SL-CAI phase underperforming, the critique step is almost always the culprit, and a vague principle list is almost always the reason.

## 12. Scalable oversight: the bigger program

Step back from the mechanics and Constitutional AI is one move in a much larger research program called *scalable oversight*: how do you reliably supervise a model on tasks where the model is more capable than the supervisor? This is not a hypothetical for harmlessness today, but it is the central problem for aligning systems that exceed human expertise — you cannot label what you cannot evaluate, and a human cannot directly grade a proof, a codebase, or a research plan that takes the human longer to check than to produce.

The figure below shows the core trick that several scalable-oversight proposals share: a weak supervisor cannot grade the hard task directly, but can grade *decomposed subtasks* that an AI assistant solves, and assemble a trusted verdict from verifiable pieces.

![A graph where a weak supervisor cannot grade a task directly so delegates to AI assistants that solve verifiable subtasks, whose results merge back for a human-checked final verdict.](/imgs/blogs/constitutional-ai-and-rlaif-7.png)

Three proposals are worth knowing because they map cleanly onto the AI-feedback methods above.

**Debate** (Irving et al., "AI safety via debate," 2018): two AI systems argue opposite sides of a question and a human (or weaker judge) decides who argued better. The bet is that it is easier to *judge* an adversarial argument than to *produce* the right answer — a lie is harder to defend against a capable opponent than the truth is. Debate is the right tool when you need adversarial robustness, because the opposing AI is incentivized to expose the other's errors, something a single AI judge in plain RLAIF will not do.

**Amplification** (Iterated Distillation and Amplification, Christiano et al., 2018): a human delegates subtasks to AI assistants, combines their answers into a better answer than the human could produce alone, and then distills that amplified capability back into the model — repeating to climb a ladder of capability while keeping a human in the supervisory loop at every rung. Amplification is the right tool when the task decomposes into checkable pieces.

**Constitutional AI itself** is a scalable-oversight method: it lets a fixed set of human-authored principles supervise a model on far more examples than humans could ever label directly, and the critique-revision loop is a form of the model making its own reasoning legible enough for the principles to grade. It is the most *deployed* of the family — Claude is trained with it — precisely because it asks the least of the oversight machinery: no adversarial setup, no recursive decomposition, just a judge and a document.

The honest framing is that none of these fully solves the hard version of the problem, where the model is vastly more capable than any available supervisor. They are bets about which structures make weak supervision of strong models *more* reliable, and the empirical evidence so far — Constitutional AI matching RLHF on today's models, debate showing promise in narrow experiments — is encouraging but firmly in the regime where humans can still check the work if they try. Whether the substitution holds as models pass human expertise is the open question the whole field is organized around.

## 13. Limitations and honest failure modes

I have planted the criticisms throughout; let me gather them, because a method that replaces human judgment with AI judgment has failure modes that are subtle precisely because the output looks polished.

**The constitution still needs human design, and a bad one is worse than no method.** Everything downstream optimizes the principles faithfully, so blind spots and biases in the principle authors become the model's values with a coat of systematic varnish. The labor did not disappear; it moved upstream and concentrated.

**AI feedback inherits the labeler's biases.** Whatever the judge model gets wrong — verbosity bias, cultural blind spots, the politics baked into its pretraining — flows directly into the preference labels and then into the reward model and then into the policy. The order-swapping and ensembling tricks reduce *position* bias and *variance*, but they do nothing about *systematic* bias shared across all of the judge's evaluations, because there is no independent signal to triangulate against.

**Circular reasoning is the deepest risk.** When the judge and the policy are the same model family, the model can prefer its own characteristic outputs — Zheng et al. measured this self-enhancement bias directly — and in iterated Constitutional AI this becomes a feedback loop with no external corrective. The model agrees with its own critiques, prefers responses that match its own style, and trains toward them, and "the model got more confident in its own judgments" is indistinguishable from "the model got more aligned" if you only watch the loss. The defenses are the constitution as an external anchor, periodic human evaluation gating each generation, and using a *different* and ideally stronger model as judge (the generic-RLAIF setup) when you can afford it.

**AI labels are not strictly better than human labels — they are differently flawed.** On harmlessness, where the rubric is explicit and human label noise is high, AI judges win on the noise-versus-bias trade. On subtle, contested value questions where there is genuine reasonable disagreement, an AI judge gives you a *confident* answer where humans would give you a *distribution* of answers, and that false confidence can be worse than honest human disagreement. The right move is not "AI feedback everywhere" but "AI feedback where the rubric is clear, human feedback where the values are contested," which is exactly the helpful-from-humans, harmless-from-AI split the original Constitutional AI paper chose.

## 14. Case studies and measured results

**Constitutional AI / Claude (Anthropic, 2022 onward).** The original Bai et al. result is the anchor: an RL-CAI model trained with *zero* human harmlessness labels was both more harmless and roughly as helpful as an RLHF model trained on human harmlessness labels, pushing out the helpfulness-harmlessness Pareto frontier rather than trading along it. The method shipped — Claude models are trained with Constitutional AI — which makes this the most consequential real-world deployment of AI feedback for alignment to date. The measurable claim to remember: AI feedback did not just *match* human feedback on harmlessness, it *exceeded* it on the frontier, because the explicit constitution gave the judge a sharper rubric than a human's gut call on a toxic prompt.

**RLAIF on summarization and dialogue (Lee et al., Google, 2023).** On the TL;DR summarization benchmark, RLAIF and RLHF were statistically indistinguishable in human-rated win rate over the SFT baseline (both around 70%), and on harmless dialogue generation RLAIF was *preferred* to RLHF. They also demonstrated that the AI labeler could be the same size as the policy and that you could skip the separate reward model entirely (direct RLAIF). The headline: on these tasks the AI labeler is a peer, not a crutch.

**LLM-as-judge calibration (Zheng et al., MT-Bench / Chatbot Arena, 2023).** This study did not train a policy; it measured how good AI judges are. GPT-4 as judge agreed with human preferences at roughly the human-human agreement level (low 80s percent) on chat quality, while carefully documenting the biases — position, verbosity, self-enhancement — that you must control for. This is the empirical license for the whole RLAIF program: AI judges are about as good as humans at the *judging* task, with known and correctable biases.

**Self-rewarding and iterated alignment (Yuan et al., 2024).** Self-rewarding language models showed a single model improving both its instruction-following and its own reward-modeling ability across iterations, with AlpacaEval win rates climbing over rounds — concrete evidence that the self-play-flavored bootstrapping loop produces real gains within the range tested, while leaving open whether it sustains over many more rounds without drift.

## 15. When to use this (and when not to)

Be decisive here, because "use AI feedback" is not a universal answer.

**Use Constitutional AI** when you need bespoke values that you can write down and audit, when harmlessness labeling is your bottleneck, when you do not have access to a stronger judge model than your policy, and when transparency or governance of the values matters (you want to be able to show, version, and amend the principles). It is the right default for a team building a safety-tuned assistant on a budget that cannot sustain a large standing annotation team.

**Use generic RLAIF with a strong external judge** when such a judge exists — you are tuning a small model and can afford GPT-4-class judging — and you care about scale more than custom principles. Let the strong model grade; you will get near-human label quality at inference cost.

**Use debate** when adversarial robustness is the priority and a single judge would be too easy to fool — high-stakes factual or strategic questions where you want one AI incentivized to catch another's errors.

**Use amplification** when the task decomposes into human-checkable subtasks and you want a human anchored in the loop at every step.

**Reach for human feedback, not AI feedback,** when the values are genuinely contested and a distribution of human opinions is more honest than a model's confident single answer; when you have no trustworthy judge and no good constitution; or when the stakes of a systematic, undetectable judge bias are unacceptable. And keep the proven split from the original paper: helpfulness, where human preferences are rich and contested, stays human-labeled; harmlessness, where the rubric is explicit and human labeling is noisy and costly, is where AI feedback earns its place.

For the broader map of where preference-based RL sits among value-based, policy-gradient, and model-based methods, see this series' unified map (`reinforcement-learning-a-unified-map`) and the capstone playbook (`the-reinforcement-learning-playbook`); for the PPO and KL-penalty machinery that RL-CAI reuses unchanged, see the series' RLHF and PPO post, and for the offline alternative that skips the reward model entirely, this method's natural rival is direct preference optimization, covered in `/blog/machine-learning/training-techniques/direct-preference-optimization`.

The matrix below summarizes the method landscape on the axes that actually drive the choice — human labeling cost, bias risk, scalability, robustness — so you can read off the trade your situation forces.

![A matrix comparing Constitutional AI, RLAIF with a GPT-4 judge, self-critique, debate, and amplification across human label cost, bias risk, scalability, robustness, and who deploys each.](/imgs/blogs/constitutional-ai-and-rlaif-5.png)

The timeline puts the same methods in historical order, from the 2017 deep-RL-from-human-preferences work that started preference-based alignment to the 2024 self-play alignment methods that try to remove the human from the loop entirely.

![A timeline from deep RL from human preferences in 2017 through Constitutional AI in 2022, the RLAIF paper and GPT-4-as-judge in 2023, to Claude production CAI and self-play alignment in 2024.](/imgs/blogs/constitutional-ai-and-rlaif-6.png)

And the decision tree distills the choice into the two questions that matter most: do you have a stronger reference judge, and do you need custom principles or adversarial robustness.

![A decision tree routing from whether a strong reference model is available to RLAIF, otherwise to Constitutional AI for custom principles, debate for adversarial robustness, or amplification for decomposable tasks.](/imgs/blogs/constitutional-ai-and-rlaif-8.png)

## Key takeaways

- The human labeling bottleneck is a *subscription*, not a one-time cost — the reward model needs fresh on-policy comparisons as the policy moves, and harmlessness labeling is the slowest, noisiest, most expensive slice. AI feedback turns that subscription from headcount-bound into compute-bound.
- Constitutional AI is a surgical edit to RLHF, not a new algorithm: keep SFT (as SL-CAI on AI-revised data), keep the reward model (now AI-labeled), keep PPO with a KL penalty (unchanged). The skeleton is identical; the labels changed hands.
- SL-CAI launders harmful prompts into clean SFT data via a critique-then-revise loop steered by randomly sampled constitutional principles; the harm score drops fast over one to two rounds and plateaus.
- RL-CAI trains a preference model on an AI judge's soft probabilities under the Bradley-Terry objective, then runs ordinary PPO against it; always order-swap and ensemble principles to kill position bias and reduce variance.
- The constitution is the one irreducibly human component, and its great advantage is *transparency*: values become an auditable, versionable, amendable document instead of a latent property of a label set.
- RLAIF generalizes the idea — any capable model can be the judge — and on summarization and harmless dialogue it matched or beat RLHF in reported win rates, with LLM-as-judge agreement reaching human-human levels.
- The deepest risk is circular reasoning: when judge and policy share a model family, self-enhancement bias and judge biases get amplified, not corrected. The constitution, a stronger external judge, and periodic human gating are the anchors.
- Constitutional AI is one move in the scalable-oversight program alongside debate and amplification; it is the most deployed because it asks the least of the oversight machinery — just a judge and a document.
- The proven split is the right default: helpfulness from humans (rich, contested), harmlessness from AI (explicit rubric, noisy and costly to human-label).

## Further reading

- Bai et al., "Constitutional AI: Harmlessness from AI Feedback," Anthropic, 2022 — the SL-CAI and RL-CAI pipelines, the constitution, and the Pareto-frontier result.
- Lee et al., "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback," Google, 2023 — generic RLAIF, same-size and direct RLAIF, the summarization and dialogue comparisons.
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," 2023 — how good AI judges are and the position, verbosity, and self-enhancement biases to control for.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022, and Ziegler et al., "Fine-Tuning Language Models from Human Preferences," 2019 — the RLHF baseline Constitutional AI edits.
- Christiano et al., "Deep reinforcement learning from human preferences," 2017 — the origin of preference-based RL; and Irving et al., "AI safety via debate," 2018, and Christiano et al., "Iterated Distillation and Amplification," 2018 — the scalable-oversight siblings.
- Yuan et al., "Self-Rewarding Language Models," 2024 — the self-play-flavored iterated alignment loop.
- Within this series: the unified map (`reinforcement-learning-a-unified-map`), the capstone (`the-reinforcement-learning-playbook`), and the offline preference-learning rival, direct preference optimization, at `/blog/machine-learning/training-techniques/direct-preference-optimization`.
