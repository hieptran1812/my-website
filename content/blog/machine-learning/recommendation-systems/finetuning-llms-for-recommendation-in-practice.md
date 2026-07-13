---
title: "Finetuning LLMs for Recommendation in Practice"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The hands-on recipe for turning a base LLM into a recommender: why finetuning beats zero-shot prompting, how to represent items as text, atomic IDs, or RQ-VAE semantic IDs, a runnable LoRA SFT loop in transformers and peft, TIGER-style generative retrieval with constrained decoding, and an honest Recall@10/NDCG@10 cost-and-latency comparison against a tuned two-tower."
tags:
  [
    "recommendation-systems",
    "recsys",
    "llm4rec",
    "finetuning",
    "lora",
    "peft",
    "semantic-ids",
    "generative-retrieval",
    "machine-learning",
    "transformers",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/finetuning-llms-for-recommendation-in-practice-1.png"
---

The first time I tried to "just prompt" a large language model into being a recommender, it recommended a movie that does not exist. The user history was real: a stack of A24 horror films, a couple of slow-burn thrillers, one inexplicable rom-com. The model — a strong, recent chat model — read all of that and confidently suggested *The Lighthouse 2*. There is no *The Lighthouse 2*. The model had absorbed enough of the internet to know that this user likes moody, claustrophobic, prestige horror, and that A24 sequels are a thing people talk about, so it manufactured a plausible-sounding title that happens to map to nothing in our catalog. It was a beautiful answer and a useless recommendation, because a recommendation that points at no buyable, watchable, clickable item is not a recommendation at all. It is a vibe.

That single failure contains the entire argument of this post. A base LLM arrives knowing an enormous amount about the *world* and almost nothing about your *system*. It does not know which of your 200,000 SKUs are in stock. It does not know that in your particular app, people who buy the cheap mascara almost never buy the premium one, and people who buy the premium one buy refills forever. It does not know your objective — are you maximizing watch-time, or first-week retention, or gross merchandise value, or just clicks? All of that lives in your interaction logs, and none of it is in the pretraining corpus. Finetuning is how you pour that knowledge into the model. The job of this post is to show you exactly how, with code you can run, math you can check, and a results table that tells you honestly when the whole exercise is worth it and when a boring two-tower model quietly wins.

![Side by side comparison of a zero-shot prompted LLM that hallucinates titles and knows no behavior versus a finetuned LLM that grounds to valid catalog codes and learns the collaborative signal](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-1.png)

This is the finetuning chapter of the series. It sits downstream of the retrieval-to-ranking funnel we keep coming back to: an LLM recommender is, in the end, just another way to do the *retrieval* stage — generate a short list of candidate items the user is likely to want next — except instead of computing a dot product and running approximate nearest-neighbor search, the model literally *writes down* the next item. By the time you finish you will be able to format interaction logs into supervised finetuning examples, choose between text, atomic-ID, and semantic-ID item representations with eyes open, run a LoRA finetune on a 7-billion-parameter base model on a single GPU, build RQ-VAE semantic IDs, generate recommendations with constrained decoding over a trie of valid item codes, and evaluate the result against a tuned baseline with Recall@10 and NDCG@10 — while keeping a clear-eyed view of training cost and serving latency.

A note on base models before we start, because it matters for the rest of the post. When I say "finetune a base LLM," the practical landscape splits in two. If you have the weights — a Llama, a Qwen, a Mistral, a Gemma — you can do everything in this post directly: add tokens, attach LoRA adapters, run a supervised loop, and serve the result yourself. If instead you are working with an API-only frontier model such as Claude Opus 4.x or Claude Sonnet, you generally do *not* get to attach LoRA adapters or add vocabulary tokens; you finetune through the provider's interface or you lean on long-context prompting with retrieved examples. Most of the published research on finetuning LLMs for recommendation — TALLRec, TIGER, P5, LC-Rec — uses open-weight models precisely because it needs that low-level access. So the runnable code here targets open weights via `transformers` and `peft`, and I will flag clearly where an API model changes the calculus.

## 1. Why finetune at all, instead of just prompting?

The honest first question is whether you need to finetune anything. Prompting is cheap, fast to iterate, and a 2026-era chat model is genuinely good at reasoning over a user's stated preferences. So why not just paste the user's last twenty interactions into a prompt and ask for ten recommendations?

Three things break, and they break in ways that get worse, not better, as your catalog grows.

**The model does not know your catalog.** This is the *The Lighthouse 2* problem. The model's output vocabulary is tokens, and any string of tokens that reads like a plausible title is a valid generation as far as the model is concerned. Nothing constrains it to items you can actually serve. You can mitigate this by stuffing the catalog into the prompt, but a catalog of 200,000 items does not fit in a context window, and even if it did, the model would not reliably copy exact titles back. Grounding — guaranteeing every recommendation maps to a real, serveable item — is the single hardest part of LLM-based recommendation, and prompting alone does not solve it.

**The model does not know your users' collaborative signal.** The whole engine of recommendation is the observation that *people who behaved like you behaved like that next*. That signal — the co-occurrence structure of who-bought-what — is the thing collaborative filtering, matrix factorization, and two-tower models are built to extract from interaction logs. A pretrained LLM has never seen your logs. It can reason about *content* similarity ("you liked one moody horror film, here is another moody horror film") but it cannot reason about *behavioral* similarity ("users who watched this niche documentary overwhelmingly watched that unrelated comedy next") because that pattern exists nowhere except your data. Content similarity is a weak proxy for behavior, and the gap is exactly where a lot of recommendation value lives.

**The model does not know your objective.** Your business optimizes something specific, and it is rarely "the most relevant-sounding item." It might be long-term retention, in which case the right recommendation sometimes pushes a user *out* of their comfort zone. It might be margin, in which case two equally relevant items are not equally valuable. A zero-shot model optimizes a fuzzy notion of relevance baked in during pretraining and alignment; it has no idea what your reward actually is.

Finetuning fixes all three at once by training on examples drawn from your logs, where the input is the user's history (and an instruction) and the target output is the item they *actually* interacted with next. The model is forced, over hundreds of thousands of examples, to learn the mapping from "this behavioral context" to "this real item from your catalog." That is the collaborative signal, the catalog grounding, and (if you build your labels from the outcome you care about) your objective, all delivered through one supervised objective.

The trade-off is data efficiency in reverse. Zero-shot prompting needs *zero* of your interaction data and works on day one — which is genuinely valuable for cold-start, brand-new domains, and the long tail of items with no behavioral history. Finetuning needs a meaningful corpus of interactions (tens of thousands of sequences at minimum to beat a decent baseline, hundreds of thousands to shine) and a training budget. The interesting empirical result from the TALLRec paper (Bao et al., 2023) is that the crossover happens *fast*: a LoRA-finetuned 7B model beat strong traditional baselines using only on the order of a few hundred to a few thousand training examples in their few-shot setting, because the base model's world knowledge does a lot of the heavy lifting and finetuning only has to teach it the local structure. You are not training a recommender from scratch; you are bending a generalist toward your data.

There is a middle ground worth naming explicitly, because it is where a lot of pragmatic systems actually live: **retrieval-augmented prompting**, sometimes called in-context recommendation. Instead of finetuning, you keep a strong base model frozen and, at request time, retrieve a handful of relevant examples — similar users' histories, popular items in the user's recent categories, a few candidate items with their metadata — and paste them into the prompt. The model then reasons over *grounded, retrieved* context rather than from memory alone. This buys you some of the catalog grounding that pure zero-shot lacks (the candidate set is real items you supplied) without a training run. Its ceiling is lower than finetuning — the model never internalizes the collaborative structure, it only reacts to whatever you happened to retrieve into the window, and the context budget caps how much behavior you can show it — but it is the right first move when you have no training corpus, when your catalog or objective changes weekly, or when you are using an API-only model such as Claude Sonnet or Claude Opus 4.x where you cannot attach adapters at all. The decision spectrum runs: zero-shot prompting (no data, lowest ceiling) → retrieval-augmented prompting (no training, grounded context) → LoRA finetuning (a corpus and a GPU, internalizes the signal) → semantic-ID generative finetuning (the full TIGER treatment, the highest ceiling and the highest cost). Most teams climb this ladder one rung at a time, stopping at the first rung that hits target.

One stress test before we move on, because the "finetuning always wins eventually" story has a real failure mode. **What happens when your interaction data is thin and noisy?** If you have only a few thousand users with short, sparse histories, a finetuned LLM can *overfit the collaborative noise* — it memorizes the popular items and the idiosyncrasies of your small log, and it loses the generalization the base model came with. In that regime the finetuned model can underperform a careful retrieval-augmented prompt, because the prompt path keeps the base model's broad prior intact and only steers it with grounded context. The diagnostic is the gap between train and validation NDCG: if the model nails the training sequences and flops on held-out users, you have overfit a thin log and you should either gather more data, regularize harder (lower LoRA rank, fewer epochs, higher dropout), or fall back to the prompting path. Finetuning is not free generalization; it is trading the base model's prior for your data, and that trade is only good when your data is worth more than the prior.

## 2. The central design decision: how do you name an item?

Before any finetuning, you have to answer one question that determines everything downstream: **how does the model refer to an item?** An LLM speaks in tokens. Your catalog is a set of items. The bridge between them — the item representation — is the most consequential design choice in LLM4Rec, and there are three real options. They are not interchangeable, and choosing wrong will cap your ceiling no matter how much you finetune.

![Matrix comparing item as text, atomic ID token, and semantic ID across grounding, cold start, vocabulary cost, and generalization](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-2.png)

**Option 1: item-as-text.** Represent each item by its title, or title plus a few attributes: `"Cerave Hydrating Cleanser, 16oz, fragrance-free"`. This is the natural thing and what every zero-shot prompt does. Its strengths are real: the representation carries semantic content for free (the model already understands "fragrance-free"), and it handles cold-start items gracefully because a brand-new item still has a title the model can read. Its weaknesses are equally real. Titles are verbose — a single item can eat 15–40 tokens, so a 20-item history plus the candidate set blows your context budget fast. Worse, generation is ungrounded: the model emits free text, you have to *match* that text back to your catalog (exact match fails constantly; fuzzy match introduces errors), and there is no guarantee the generated title corresponds to anything real. Item-as-text is where most people start and where the grounding pain shows up immediately.

**Option 2: item-as-atomic-ID-token.** Add one new token to the vocabulary per item: `<item_48172>`. Now an item is exactly one token, histories are compact, and generation is grounded by construction — the model can only emit tokens that exist, and every item token maps to exactly one real item. This is clean and it works, and it is essentially what P5 (Geng et al., 2022) does in some of its formulations. But it has two structural problems. First, the vocabulary explodes: a million items means a million new embedding rows, each of which must be learned from scratch, which inflates the model and demands a lot of data per item to train each embedding. Second, and more fundamentally, atomic IDs are *cold-start-blind and share nothing*. Token `<item_48172>` and token `<item_48173>` are as unrelated as any two random tokens; the model learns each item's embedding independently, with no parameter sharing, so two near-identical items get no transfer between them and a brand-new item gets a freshly-initialized, untrained embedding — a cold token the model has never emitted. Generalization across items is exactly what atomic IDs cannot do.

**Option 3: semantic IDs.** This is the idea that made generative retrieval work, and it is worth understanding deeply because the rest of the post leans on it. Instead of one token per item, you represent each item as a short *sequence* of codes drawn from a small shared codebook — for example item 48172 becomes the tuple `(12, 7, 89)`, three codes each from a codebook of 256. These codes are not random IDs; they are produced by quantizing the item's content embedding with a residual-quantized variational autoencoder (RQ-VAE), so that *semantically similar items share leading codes*. Two similar moisturizers might both start with code `12`, differing only in the third position. Now the magic: the model generates an item by emitting its code sequence autoregressively, one code at a time, exactly like generating text — and because codes are shared across items, the model learns a generalizable mapping. A brand-new item gets quantized into the same shared codebook, reusing codes the model already knows how to emit, so cold-start is handled by construction. The vocabulary cost is tiny: 256 codes times the number of levels (say 4), not one token per item. The generation is grounded if you constrain decoding to valid code sequences (more on that trie in section 6). Semantic IDs are the representation behind TIGER (Rajput et al., 2023), and they are, in my experience, the right default when you commit to a generative LLM recommender.

| Representation | Tokens per item | Grounding | Cold start | Vocab growth | Generalizes across items |
| --- | --- | --- | --- | --- | --- |
| Item-as-text | 15–40 | Weak (must match back) | Good (has a title) | None | Yes (via content) |
| Atomic ID token | 1 | Exact (by construction) | Blind (untrained token) | +1 per item | No (no code sharing) |
| Semantic ID | ~3–4 codes | Exact (if trie-constrained) | Good (reuses codes) | Fixed (256 × L) | Yes (shared codes) |

The decision boils down to this: if you need cold-start and semantic transfer *and* exact grounding, semantic IDs are the only option that gives you all three; item-as-text gives you cold-start and semantics but not grounding; atomic IDs give you grounding but not cold-start or transfer. We will build all three in code, but the semantic-ID path is where the interesting results come from.

## 3. Instruction-tuning for recommendation: the data format

With the item representation chosen, finetuning itself is supervised finetuning (SFT) on instruction-formatted examples. The format is the same idea that powers instruction-tuned chat models: each training example is a (prompt, target) pair, where the prompt frames the task and supplies the user's context, and the target is the answer the model should produce. For recommendation, the answer is the next item.

TALLRec (Bao et al., 2023) and InstructRec (Zhang et al., 2023) established the template. A TALLRec-style example for a "predict whether the user likes this candidate" formulation looks like a natural-language instruction wrapping the user's liked and disliked history and a target candidate, with a yes/no target. A next-item generation formulation — closer to TIGER and what we will train — looks like this:

```markdown
### Instruction:
Given the items the user has interacted with in order, predict the next item.

### Input:
User history: Cerave Hydrating Cleanser; The Ordinary Niacinamide;
Paula's Choice BHA; La Roche-Posay Sunscreen

### Response:
CeraVe Moisturizing Cream
```

For the atomic-ID or semantic-ID variants, the history and the response are sequences of ID tokens or code tuples instead of titles. The instruction text stays roughly constant across examples (you can even drop it to save tokens once the model has learned the task); the input carries the per-user behavioral context; the response is the supervision signal.

The single most important rule when you build these examples: **only the response tokens carry loss.** You mask the loss on the instruction and input tokens (set their labels to the ignore index, conventionally `-100`) so the model is graded only on whether it produces the right *next item*, not on whether it can regurgitate the prompt. Getting this masking wrong is the most common silent bug in rec SFT — the model spends its gradient learning to copy the instruction and never actually learns the recommendation task, and your loss curve looks fine while your Recall@10 sits at the floor.

Here is the data-prep code that turns a temporally-ordered interaction log into masked SFT examples, using the item-as-text representation:

```python
import json
from dataclasses import dataclass

INSTRUCTION = (
    "Given the items the user has interacted with in order, "
    "predict the next item the user will engage with."
)

def build_examples(user_sequences, item_title, max_hist=20):
    """user_sequences: dict[user_id] -> list of item_ids in time order.
    item_title: dict[item_id] -> str. Yields (prompt, target) pairs."""
    for uid, seq in user_sequences.items():
        # leave-one-out: every prefix predicts the next item (temporal, no leakage)
        for t in range(1, len(seq)):
            hist = seq[max(0, t - max_hist):t]
            target_id = seq[t]
            hist_str = "; ".join(item_title[i] for i in hist)
            prompt = (
                f"### Instruction:\n{INSTRUCTION}\n\n"
                f"### Input:\nUser history: {hist_str}\n\n"
                f"### Response:\n"
            )
            yield {"prompt": prompt, "target": item_title[target_id]}

# write a jsonl the tokenizer step will consume
with open("rec_sft.jsonl", "w") as f:
    for ex in build_examples(user_sequences, item_title):
        f.write(json.dumps(ex) + "\n")
```

Two things to note. First, the leave-one-out expansion: every prefix of a user's sequence becomes a training example predicting the next item, which is how you get hundreds of thousands of examples from tens of thousands of users — the standard sequential-recommendation training setup you have seen in the SASRec and BERT4Rec post. Second, the split must be *temporal*, not random: hold out each user's last interaction for test, the second-to-last for validation, and train on everything before. Random splits leak future behavior into training and inflate every metric you report. This is the same leakage discipline the offline-evaluation posts in this series hammer on, and it bites just as hard with LLMs.

A few format decisions matter more than they look. The **history length** is a direct cost knob: at 15–40 tokens per item-as-text entry, a 20-item history is 300–800 tokens before the instruction, so doubling the window roughly doubles your training and serving compute. Most rec finetunes cap history at 10–20 items and rely on the most recent behavior carrying most of the signal — the same recency-weighting intuition behind sequential models, except here it is enforced by the token budget rather than learned. The **ordering** should be chronological (oldest first, most recent last) so the model's causal attention sees the same left-to-right time order it will see at inference. And the **instruction wording** is not just decoration: in the few-shot regime the instruction does real work anchoring the task, but once you have a large corpus you can shorten or even drop it to reclaim tokens, because the model has learned the task from the input→target structure and no longer needs the natural-language framing. Run both and measure; I have seen the short-instruction variant win on throughput with no accuracy loss once the corpus is large enough.

There is also a quietly important choice about **what counts as a positive**. With implicit feedback — clicks, plays, purchases — every interaction in the sequence is a positive, and the next item is the label. But not every click means the user liked the item; some are mis-clicks, some are bounces. If you have dwell time, watch-completion, or an explicit rating, you can filter the sequence to *meaningful* interactions (a play that exceeded 30 seconds, a purchase rather than a cart-add) so the model learns from intent rather than noise. This is the same implicit-vs-explicit-feedback question the rest of the series wrestles with, and it shows up here as a data-filtering step before you ever tokenize: garbage positives in, garbage recommendations out, and the LLM will confidently learn whatever pattern you feed it.

## 4. The science: the SFT objective and why masking matters

Let me make the training objective precise, because the "only the response carries loss" rule falls out of it directly. An autoregressive language model defines a probability over a token sequence by the chain rule:

$$p_\theta(y_1, \dots, y_T) = \prod_{t=1}^{T} p_\theta(y_t \mid y_{\lt t}).$$

Supervised finetuning minimizes the negative log-likelihood of the *target* tokens given the *prompt* tokens. If we split the full sequence into a prompt $x = (x_1, \dots, x_m)$ and a response $y = (y_1, \dots, y_n)$, the loss for one example is

$$\mathcal{L}(\theta) = -\sum_{t=1}^{n} \log p_\theta\!\left(y_t \mid x, y_{\lt t}\right).$$

Notice the sum runs over the response tokens only. The prompt tokens $x$ appear in the *conditioning* of every term but never as a *prediction target*. That is exactly what the label mask implements: by setting the labels of the prompt positions to `-100`, the cross-entropy at those positions contributes zero to the gradient. The model still attends to the prompt (it is in the context), it just is not graded on reproducing it. For a next-item rec model with semantic IDs, $y$ is the short code sequence of the target item, so the loss is simply the cross-entropy of generating that item's codes given the user's history — the collaborative signal, expressed as a sequence-generation likelihood.

This framing also explains why semantic IDs train more efficiently than atomic IDs from a pure optimization standpoint. With atomic IDs, the target $y$ is a single token from a vocabulary of a million items, and the softmax over that giant output space is both expensive and starved of gradient per item (each item token is the target in only a handful of examples). With semantic IDs, the target is ~4 codes each from a vocabulary of 256, so every code is the target in *thousands* of examples, the softmaxes are tiny, and the parameters are shared. The model gets dense, well-distributed gradient signal. That is the optimization reason semantic IDs generalize where atomic IDs starve.

It is worth being precise about *where* the collaborative signal actually enters the model under this objective, because it is not obvious. Nothing in the cross-entropy loss explicitly says "learn co-occurrence." The signal sneaks in through the conditional structure: when thousands of training examples show that histories ending in items A and B are followed by item C, the only way the model can minimize $-\log p_\theta(C \mid A, B)$ across all of them is to encode, somewhere in its weights (or its LoRA adapters), the regularity "A then B implies C." That is collaborative filtering, re-expressed as a sequence-prediction likelihood. The model is not told the co-occurrence matrix; it reconstructs the part of it that reduces loss, exactly as matrix factorization reconstructs the interaction matrix by minimizing reconstruction error. The difference is that the LLM also carries a strong content prior from pretraining, so it can fill in plausible continuations for histories it has never seen — which is precisely the generalization a from-scratch factorization model cannot do.

A subtle consequence: because the loss is a *product* of per-token conditionals, the model is graded on getting the *whole* semantic-ID sequence right, but the early codes dominate the gradient because they condition everything after them. Getting the first code wrong derails the entire item. This is why RQ-VAE's coarse-to-fine structure matters for training and not just for generalization — the first codebook captures the broadest category, so the model learns the easy, high-signal distinction first (skincare vs makeup) and only later refines within a category (which cleanser). The loss landscape is hierarchical because the representation is hierarchical, and that hierarchy is part of why semantic-ID models train stably.

#### Worked example: how many trainable parameters does LoRA save on a 7B model?

Take a concrete 7B base model — say a Llama-7B-class architecture with hidden dimension $d = 4096$ and 32 transformer layers. Full finetuning trains all ~7 billion parameters. Now attach LoRA adapters of rank $r = 16$ to the four attention projection matrices ($q$, $k$, $v$, $o$), each of shape $4096 \times 4096$.

A LoRA adapter on a $d \times d$ matrix replaces the full $d^2$ update with two low-rank factors: a down-projection $A \in \mathbb{R}^{r \times d}$ and an up-projection $B \in \mathbb{R}^{d \times r}$, for $2 \cdot d \cdot r$ trainable parameters. Per matrix:

$$2 \cdot d \cdot r = 2 \times 4096 \times 16 = 131{,}072 \text{ parameters}.$$

Four matrices per layer, 32 layers:

$$131{,}072 \times 4 \times 32 = 16{,}777{,}216 \approx 16.8 \text{ million parameters}.$$

So LoRA trains about **16.8M parameters versus 7B for full finetuning — roughly 0.24%.** If you also adapt the MLP projections (common, and usually worth it for rec), you might land around 40M trainable parameters, still well under 1%. The full $d \times d$ attention update would have been $4096^2 \times 4 \times 32 \approx 2.1$ billion parameters just for attention; LoRA replaces that with 16.8M. That 100-plus-fold reduction is what lets a 7B finetune fit on a single 24GB consumer GPU with QLoRA, and it is why nobody full-finetunes a base LLM for a recommendation task unless they have a very specific reason and a lot of GPUs.

## 5. PEFT and LoRA: the low-rank update that makes this affordable

Full finetuning of a 7B model means storing gradients and optimizer states for all 7B parameters. With the Adam optimizer that is roughly the model weights plus two more copies (first and second moments) plus the gradients — call it four times the parameter memory in fp32, which is tens of gigabytes before you have loaded a single batch. That is overkill for a recommendation task, where the base model already knows how to read and generate; you are only teaching it a relatively small amount of new structure.

Parameter-efficient finetuning (PEFT) sidesteps this. LoRA — Low-Rank Adaptation (Hu et al., 2021) — is the dominant method, and the idea is elegant. For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, instead of learning a full update $\Delta W$, you constrain the update to be low-rank:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r}, \ A \in \mathbb{R}^{r \times k}, \ r \ll \min(d, k).$$

$W_0$ stays **frozen** — no gradients, no optimizer state. Only $A$ and $B$ are trained. $A$ is initialized from a small random distribution and $B$ is initialized to **zero**, so at the start of training $\Delta W = BA = 0$ and the model is exactly the pretrained model — finetuning begins from a no-op and only departs as the adapters learn. The scalar $\alpha/r$ controls how much the adapter perturbs the base; $\alpha$ is a hyperparameter (often set equal to $r$ or $2r$). The hypothesis LoRA rests on, and the reason it works, is that the *update* needed to specialize a pretrained model to a downstream task has low intrinsic rank — you do not need a full-rank change to teach the model your catalog, you need a thin one.

![Stack of LoRA layers showing a frozen weight matrix W with a parallel low rank B times A adapter summed into the output and the resulting parameter savings](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-4.png)

The forward pass is just $h = W_0 x + \frac{\alpha}{r} B(Ax)$ — the original transformation plus a thin detour through the rank-$r$ bottleneck. At inference time you can *merge* the adapter, computing $W = W_0 + \frac{\alpha}{r}BA$ once and folding it back into a single weight matrix, so the finetuned model has *zero* added latency versus the base — there is no extra matmul in the merged model. You can also keep adapters separate and hot-swap them, which is great if you serve multiple recommendation tasks (one adapter for the home feed, one for search, one for the email digest) on top of one shared frozen base, saving enormous memory because you store one 7B base and several tiny adapters.

QLoRA (Dettmers et al., 2023) pushes this further: quantize the frozen base to 4-bit (NF4), keep the LoRA adapters in higher precision, and now a 7B finetune fits comfortably on a single 24GB GPU. The base never updates, so its quantization error does not accumulate; the adapters carry the learning in higher precision. For recommendation finetunes this is the default I reach for — the cost difference between QLoRA on one GPU and full finetuning on a node of eight is the difference between a same-day experiment and a procurement request.

If you want the deeper treatment of LoRA, QLoRA, and the full PEFT family — rank selection, target-module choice, DoRA, and the failure modes — this series leans on the dedicated [effective LLM finetuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) post; here I will keep it to what a rec finetune actually needs.

Here is the LoRA finetuning recipe in `transformers` + `peft`, the item-as-text path end to end:

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "meta-llama/Llama-2-7b-hf"   # any open-weight base works

# 4-bit base (QLoRA): the frozen weights live in NF4, adapters in bf16
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # ~ 0.3% of params trainable
```

The `print_trainable_parameters()` call is your sanity check — it should print a number on the order of tens of millions and a percentage well under 1%, confirming the base is frozen and only adapters learn. If it prints billions, your `LoraConfig` did not attach and you are about to full-finetune by accident.

Now the tokenization with response-only loss masking — the part everyone gets wrong:

```python
def tokenize(example, max_len=512):
    prompt_ids = tok(example["prompt"], add_special_tokens=False)["input_ids"]
    target_ids = tok(example["target"] + tok.eos_token,
                     add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[:max_len]
    # mask the prompt: only the target tokens contribute to the loss
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
    attn = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

ds = load_dataset("json", data_files="rec_sft.jsonl", split="train")
ds = ds.map(tokenize, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir="rec-lora", num_train_epochs=2,
    per_device_train_batch_size=8, gradient_accumulation_steps=4,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, logging_steps=20, save_strategy="epoch",
    optim="paged_adamw_8bit",
)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
trainer.train()
model.save_pretrained("rec-lora/adapter")   # tiny: tens of MB, not GB
```

That is a complete, runnable LoRA SFT loop. The learning rate of `2e-4` is deliberately higher than you would use for full finetuning — LoRA adapters tolerate and want a larger LR because they are a small, well-conditioned subspace. The `paged_adamw_8bit` optimizer keeps optimizer-state memory down, the other half of the QLoRA memory story. The saved adapter is tens of megabytes; you keep the 13GB base once and stack adapters on top.

A few knobs decide whether this finetune is good or merely runs. **Rank** ($r$) is the capacity dial: 8 or 16 is plenty for most rec tasks, and pushing to 64 rarely helps and starts to overfit thin logs. If your validation NDCG plateaus low, the fix is usually more or cleaner data, not more rank. **Target modules** matters more than people expect: adapting only the attention projections ($q$, $k$, $v$, $o$) is the classic LoRA setup, but for recommendation — where the model has to learn a lot of new factual structure about your catalog and codes — including the MLP projections (`gate_proj`, `up_proj`, `down_proj`) consistently helps, at the cost of roughly doubling trainable parameters. **`lora_alpha`** scales the adapter's contribution; the common rule of thumb is $\alpha = 2r$, and the effective learning rate of the update is tied to $\alpha / r$, so if you change rank, revisit alpha. **Epochs** should be small — two to three — because the leave-one-out expansion already gives you many examples per user and the base model learns fast; train too long and you memorize the log and overfit the head of the popularity distribution, which shows up as the model recommending the same ten viral items to everyone, the exact failure the whole series warns about.

#### Worked example: does a 7B QLoRA finetune fit on one 24GB GPU?

Let me check the memory budget that makes the single-GPU claim real, because "it fits" is the whole reason QLoRA matters. A 7B model in 4-bit NF4 stores roughly $7\text{B} \times 0.5\text{ bytes} \approx 3.5$ GB of frozen weights (4 bits = half a byte per parameter, plus a small overhead for the quantization scales). The LoRA adapters — say 40M parameters with MLP modules included — in bf16 are $40\text{M} \times 2\text{ bytes} = 80$ MB. The Adam optimizer keeps two moments per *trainable* parameter; with the 8-bit paged optimizer that is roughly $40\text{M} \times 2 \times 1\text{ byte} = 80$ MB. Gradients for the adapters add another ~80 MB. So the static footprint is about $3.5\text{ GB} + 0.24\text{ GB} \approx 3.7$ GB. The rest of the 24GB goes to **activations**, which scale with batch size and sequence length: at batch size 8, sequence length 512, a 7B model's activation memory under gradient checkpointing runs a few gigabytes. Add it up and you are comfortably under 24GB with headroom — which is exactly why a consumer RTX 4090 can finetune a 7B recommender. Now contrast full finetuning: 7B weights in bf16 (14 GB) + gradients (14 GB) + Adam moments (28 GB) = 56 GB *before activations*, which does not fit on a single 24GB or even 40GB GPU. The 4-bit frozen base plus tiny adapters is the difference between one consumer GPU and a multi-GPU node — a roughly 15x reduction in the memory that dominates the bill.

![Graph of the LoRA SFT pipeline where user history and an instruction flow into a frozen base LLM combined with LoRA adapters to produce next item tokens scored by a masked cross entropy loss](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-3.png)

## 6. Generative retrieval: TIGER and decoding the next item

So far the model emits item *text* and we have to match it back to the catalog — the grounding problem. Generative retrieval, introduced for recommendation by TIGER (Rajput et al., 2023), solves grounding by changing what the model emits and how it decodes. Instead of free text, the model generates the **semantic ID** of the next item, token by token, and decoding is **constrained** so that only valid item codes can be produced. The model literally generates the recommendation, and the recommendation is guaranteed to be a real item. There is no embedding, no nearest-neighbor index, no post-hoc matching — retrieval *is* generation.

Building this has two parts: constructing semantic IDs with an RQ-VAE, and decoding with a constraint.

**RQ-VAE semantic IDs.** Start from a content embedding for each item — say a 768-dimensional vector from a `sentence-transformers` model encoding the item's title and description. A residual-quantized VAE compresses that vector into a short sequence of discrete codes through a stack of codebooks. The first codebook quantizes the embedding to its nearest codeword; you subtract that codeword and quantize the *residual* with the second codebook; subtract again and quantize with the third; and so on for $L$ levels. Each item ends up as a tuple of $L$ code indices, e.g. $(12, 7, 89)$ for $L = 3$. Because the first code captures the coarsest structure, semantically similar items share their leading codes — the entire reason semantic IDs generalize.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class RQVAE(nn.Module):
    """Residual-quantized VAE: item embedding -> tuple of L codes."""
    def __init__(self, dim=768, n_levels=3, codebook_size=256, code_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                     nn.Linear(256, code_dim))
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, code_dim))
             for _ in range(n_levels)])
        self.decoder = nn.Sequential(nn.Linear(code_dim, 256), nn.ReLU(),
                                     nn.Linear(256, dim))

    def quantize(self, z):
        codes, residual, quant = [], z, 0.0
        for cb in self.codebooks:                       # one level at a time
            d = torch.cdist(residual, cb)               # distance to codewords
            idx = d.argmin(dim=-1)                       # nearest code
            sel = cb[idx]
            codes.append(idx)
            quant = quant + sel
            residual = residual - sel                    # quantize the residual
        return torch.stack(codes, dim=-1), quant

    def forward(self, x):
        z = self.encoder(x)
        codes, quant = self.quantize(z)
        x_hat = self.decoder(quant)
        recon = F.mse_loss(x_hat, x)
        commit = F.mse_loss(quant, z.detach())           # commitment loss
        return codes, recon + 0.25 * commit
```

You train this RQ-VAE on all item embeddings to minimize reconstruction plus a commitment loss, then run every item through `quantize` once to get its semantic ID. There is one wrinkle to handle in production: two distinct items can collide on the same code tuple, so the convention (from TIGER) is to append a small disambiguating fourth code that counts collisions — so the final ID is $(12, 7, 89, 0)$, $(12, 7, 89, 1)$, and so on for items that share the first three codes. You add these $256 \times L$ code values as tokens to the LLM's vocabulary, and now item histories and targets are sequences of code tokens.

Two RQ-VAE details determine whether the resulting codes are any good. First, **codebook utilization**: a common failure is codebook collapse, where the model uses only a handful of the 256 codewords and leaves the rest dead, which throws away representational capacity and forces many items onto the same prefix. You diagnose it by counting unique code usage per level after training; if utilization is low, the standard fixes are k-means initialization of the codebooks from the item embeddings (so codewords start spread across the data) and periodically re-initializing dead codes to high-loss examples. TIGER initializes codebooks with k-means precisely for this reason. Second, the **content embedding you quantize sets the ceiling**: if your `sentence-transformers` embedding does not capture the distinctions that matter for your catalog, no amount of quantization will recover them. For a catalog where behavior diverges from content (two products with similar descriptions but very different buyers), this is where LC-Rec's idea — blending collaborative signal into the codes rather than quantizing content alone — earns its keep. The semantic ID is only as good as the embedding it compresses.

**Constrained decoding over a trie.** When the finetuned model generates the next item, you must prevent it from emitting an invalid code sequence — a tuple that corresponds to no item. You do this by building a prefix tree (trie) of all valid item code sequences and, at each decoding step, masking the model's logits down to only the codes that continue a valid prefix. Combined with beam search, the model explores the most probable *valid* item codes and returns the top-$k$ items, every one of them real.

![Graph showing an item embedding quantized by an RQ-VAE into a semantic ID then a finetuned LLM generating that ID under a codeword trie constraint via beam search](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-5.png)

```python
class CodeTrie:
    """Trie of valid semantic-ID sequences for constrained decoding."""
    def __init__(self, item_codes):           # item_codes: list of code tuples
        self.root = {}
        for codes in item_codes:
            node = self.root
            for c in codes:
                node = node.setdefault(c, {})
            node["__item__"] = True            # leaf marks a real item

    def allowed_next(self, prefix):
        node = self.root
        for c in prefix:
            node = node.get(c, {})
        return [k for k in node.keys() if k != "__item__"]

def constrained_logits(logits, prefix, trie, code_token_offset):
    """Mask logits to only codes that continue a valid item prefix."""
    allowed = trie.allowed_next(prefix)
    mask = torch.full_like(logits, float("-inf"))
    for code in allowed:
        mask[code_token_offset + code] = logits[code_token_offset + code]
    return mask
```

In `transformers`, you wire `constrained_logits` into a `LogitsProcessor` (or use the built-in `PrefixConstrainedLogitsProcessor`, passing a function that returns the allowed next token ids given the generated prefix) and call `model.generate(..., num_beams=10, num_return_sequences=10)`. Out comes a list of ten valid semantic-ID sequences, which you map back to items through the codes-to-item dictionary you built when you constructed the IDs. That mapping is exact and grounded by construction — there is no fuzzy matching, no hallucinated *Lighthouse 2*.

The constraint does double duty for performance, not just correctness. Without it, beam search would have to consider the full vocabulary at every step — hundreds of thousands of tokens if you used atomic IDs — and most of those continuations are invalid. The trie prunes the branching factor at each step from "the whole vocabulary" down to "the few codes that continue a real item," which keeps beam search cheap even with a large catalog: the search width is bounded by how many items share each prefix, not by the catalog size. There is a real engineering cost, though — building and holding the trie for a 10M-item catalog, and running the per-step logit masking efficiently on GPU, is non-trivial, and a naive Python-loop `LogitsProcessor` will dominate your latency. Production implementations precompute the trie into a flat tensor representation and do the masking as a vectorized gather, which is the difference between the 130ms in the results table and something five times slower. The lesson recurs throughout LLM4Rec: the model is the easy part; the decoding constraint and its efficient implementation are where the serving latency is actually won or lost.

#### Worked example: tracing one constrained generation

Suppose the model has generated the first two codes of the next item: prefix $(12, 7)$. We have three items in the catalog whose IDs begin with $(12, 7)$: cleanser A is $(12, 7, 89)$, moisturizer B is $(12, 7, 03)$, and toner C is $(12, 7, 89, 1)$ (a collision-disambiguated twin of A). At the third decoding step the trie's `allowed_next((12, 7))` returns exactly the codes $\{89, 03\}$ — every other code among the 256 gets masked to $-\infty$. The model's raw logits might favor code `45`, but `45` continues no valid item, so it is impossible to emit. Among the allowed codes the model assigns, say, probability 0.7 to `89` and 0.3 to `03`. Beam search keeps both partial paths. At the fourth step, prefix $(12, 7, 89)$ allows $\{\text{leaf marking item A}, 1\}$ — so the beam splits into item A (cleanser) and item C (toner, the disambiguated twin). Three beams survive — cleanser A, moisturizer B, toner C — all real items, ranked by their accumulated log-probability. The constraint did two jobs at once: it guaranteed grounding, and it pruned the search to a tiny valid frontier so beam search stays cheap. This is the mechanism that lets a 7B model serve grounded recommendations without an ANN index.

## 7. Generative retrieval versus embedding-and-ANN: the real trade-off

It is tempting to read TIGER as strictly better than the two-tower-plus-faiss retrieval stack from the [approximate nearest neighbor serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) post. It is not. It is a different point on the cost-latency-accuracy surface, and you should choose with eyes open.

![Side by side comparison of embedding plus ANN retrieval that stores a vector index versus generative semantic ID retrieval that emits item codes with no index but pays decoding latency](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-6.png)

The embedding-and-ANN stack computes a user embedding and a per-item embedding, builds an HNSW or IVF-PQ index over the item vectors, and answers a query with a maximum-inner-product search in single-digit milliseconds. To add a new item you embed it and insert it into the index — cheap and incremental. The index costs memory (gigabytes for a large catalog) and you maintain it as a separate serving component.

Generative retrieval throws the index away entirely. The "index" is the model weights; there is no separate vector store to build, host, or keep in sync. Adding a new item means quantizing it into a code tuple — no re-indexing. And because semantic IDs share codes, the model can in principle recommend items it saw rarely or never during finetuning, because their codes are familiar. Those are real wins, especially at the memory and operational-simplicity level.

The cost is **latency and the cold start of new codes**. Generating a 4-code item with 10-way beam search is several sequential forward passes through a 7B model — tens to low-hundreds of milliseconds on a GPU, versus single-digit milliseconds for ANN. For a retrieval stage that has to return candidates in a few milliseconds under a tight serving budget, that is often a deal-breaker; you do not put a 7B autoregressive decode on the hot path of a high-QPS feed without a very good reason or a lot of GPUs. There is also a subtler cold-start issue: while a *new item* reuses existing codes gracefully, a genuinely *novel region* of the catalog might need code combinations the model has rarely emitted, and constrained beam search can under-explore them. The two-tower model, by contrast, just needs the new item's content features to produce a reasonable embedding.

| Property | Embedding + ANN | Generative semantic ID |
| --- | --- | --- |
| Retrieval mechanism | MIPS over a vector index | Autoregressive code generation |
| Serving p99 latency | 5–10 ms | 100–200 ms (beam decode) |
| Memory | Index in RAM (GBs) | Model weights only |
| Add a new item | Embed + insert into index | Quantize to a code tuple |
| Grounding | Index entries are real items | Trie guarantees valid items |
| Cold-start of novel item | Good (content embedding) | Good if codes are familiar |
| Operational complexity | Model + index + sync | Model only |

My honest read: generative retrieval is genuinely compelling for medium-QPS surfaces where catalog churn is high and operational simplicity matters, and where the accuracy and cold-start gains from semantic IDs pay for the latency. For a high-QPS, latency-critical first-stage retriever over a stable catalog, a tuned two-tower with HNSW still wins on the metric that the SRE cares about. Many production systems that adopt LLM ideas do so *one stage down* — using the LLM-generated candidates as one source feeding a fast ranker — rather than putting the LLM on the retrieval hot path.

It is worth stress-testing the generative approach at scale, because the "no index" pitch sounds like it dodges the memory problem entirely, and it does not dodge it so much as relocate it. At 100M items, the embedding-and-ANN stack has a genuine memory headache: 100M vectors at 128 dimensions in fp16 is about 25 GB before any index overhead, and HNSW roughly doubles that, so you are sharding the index across hosts. The generative model "solves" this by storing no vectors — but it pays in two other places. First, the trie of 100M valid code sequences is itself a large data structure that has to live in serving memory and be traversed per decoding step. Second, and more fundamentally, can 100M items even be *distinguished* by a 4-code semantic ID? With a 256-way codebook and 4 levels you have $256^4 \approx 4.3$ billion possible codes, so in principle yes, but the collisions concentrate — popular regions of the catalog crowd into the same prefixes — and the disambiguation tail grows, which lengthens generation and inflates the trie. At true web scale, generative retrieval's advantages erode and the index-based approach's maturity (years of faiss/ScaNN engineering, battle-tested sharding) reasserts itself. The sweet spot for generative retrieval is catalogs in the hundreds of thousands to low millions, not billions — which is exactly the range of the academic benchmarks where it shines, and a real constraint to respect when someone proposes it for a billion-item marketplace.

![Side by side comparison of atomic item IDs that share no parameters and are cold start blind versus semantic IDs whose shared codes generalize across items including new ones](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-7.png)

## 8. Evaluation: a runnable Recall@10 and NDCG@10 harness

A finetuned LLM recommender is evaluated exactly like any other top-K retriever, with the same temporal-split discipline. You hold out each user's last interaction as the test target, generate the model's top-10 predictions from the user's history, and compute Recall@10 (did the held-out item appear in the top 10?) and NDCG@10 (did it appear *high* in the top 10?). The definitions, from the offline-evaluation chapter:

$$\text{Recall@}K = \frac{1}{|U|}\sum_{u} \mathbb{1}\!\left[\text{target}_u \in \text{top-}K_u\right], \qquad \text{NDCG@}K = \frac{1}{|U|}\sum_{u} \frac{1}{\log_2(\text{rank}_u + 1)}$$

(the NDCG simplification holds when there is a single relevant held-out item, the standard sequential-rec setup; $\text{rank}_u$ is the 1-based position of the target in the top-K, and the term is 0 if it is absent).

```python
import numpy as np

def evaluate(model, tok, test_set, trie, codes_to_item, k=10):
    """test_set: list of (history_prompt, target_item_id)."""
    recalls, ndcgs = [], []
    for prompt, target_id in test_set:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids, num_beams=k, num_return_sequences=k,
            max_new_tokens=4, do_sample=False,
            # PrefixConstrainedLogitsProcessor built from `trie` goes here
        )
        # decode each beam to a code tuple, map to an item id
        preds = [codes_to_item[decode_codes(seq, tok)] for seq in out]
        hit = target_id in preds
        recalls.append(1.0 if hit else 0.0)
        if hit:
            rank = preds.index(target_id) + 1            # 1-based
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            ndcgs.append(0.0)
    return {"Recall@%d" % k: np.mean(recalls),
            "NDCG@%d" % k: np.mean(ndcgs)}
```

Two evaluation traps specific to LLM recommenders. First, **do not sample negatives for the metric.** A lot of older sequential-rec papers rank the true item against 100 random negatives and report metrics on that small set; the KDD 2020 result (Krichene and Rendle, "On Sampled Metrics for Item Recommendation") showed sampled metrics can be *inconsistent* — they reorder methods relative to the full-catalog ranking. For a generative model the full-catalog evaluation is natural (beam search ranks over all valid codes), so report the full metric and you sidestep the inconsistency entirely. Second, **measure latency with a warm cache and realistic beam width**, because the beam-decode cost is the headline number that decides whether you can actually ship this; a metric table without a latency column is hiding the most important variable.

There is a third trap that is easy to miss and embarrassing to discover online: **beam width is both an accuracy and a latency knob, and they trade off.** Recall@10 with a beam width of 10 is not the same as Recall@10 with a beam width of 50 — a wider beam explores more of the valid-code frontier and finds the true item more often, lifting recall, but each extra beam is more compute. So you cannot report a single Recall@10 number without saying what beam width produced it, and you cannot compare two systems fairly unless their beams are matched or their latencies are. When I see an LLM recommender with a great offline Recall@10, the first question is always "at what beam width, and what did that do to p99?" — because a method that needs a beam of 50 to hit its headline number may be untenable to serve, and a method that hits target at beam 5 is the one I would actually ship. This is the LLM-specific version of the recall-vs-latency Pareto curve the ANN-serving post draws for HNSW's `efSearch` parameter: same shape, same discipline, different knob.

A note on **online evaluation**, because offline metrics on Amazon Beauty are not the final word for your system. The whole series' refrain — offline NDCG can rise while online engagement falls — applies with full force to LLM recommenders, arguably more so, because the model's outputs are *qualitatively different* (more diverse, more semantically driven, sometimes more surprising) in ways a single offline metric does not capture. A semantic-ID model might recommend a less popular but genuinely well-matched item that the offline temporal split scores as a miss (the user happened to click the popular thing that day) but that drives long-term engagement when shipped. The only way to know is an A/B test with the metrics you actually care about — CTR, watch-time, retention, GMV — over enough time to see retention effects. Treat the offline table as a *filter* (does this clear the bar to be worth an A/B slot?) not as the *decision* (should this replace the incumbent?).

## 9. Results: the honest comparison

Here is the comparison that matters, on the **Amazon Beauty** sequential-recommendation benchmark — one of the datasets TIGER reports on, chosen so the numbers are literature-consistent rather than invented. I am reporting figures in the range the public literature reports for this benchmark family; treat the exact decimals as representative, not as a fresh run on your data, and re-measure on your own catalog before betting on them.

![Matrix of results on Amazon Beauty comparing SASRec two tower, LLM with LoRA on text, and LLM with semantic IDs across Recall at 10, NDCG at 10, training cost, and serving latency](/imgs/blogs/finetuning-llms-for-recommendation-in-practice-8.png)

| Model | Recall@10 | NDCG@10 | Training cost | Serving p99 |
| --- | --- | --- | --- | --- |
| SASRec / two-tower (tuned) | ~0.090 | ~0.043 | ~2 GPU-hr | ~9 ms |
| LLM + LoRA, item-as-text | ~0.094 | ~0.046 | ~40 GPU-hr | ~210 ms |
| LLM + semantic ID (TIGER-style) | ~0.106 | ~0.052 | ~60 GPU-hr | ~130 ms |

Read this table honestly and it tells a clear story. The semantic-ID generative model is the **accuracy winner** — roughly a 15–20% relative Recall@10 lift over a tuned SASRec on this benchmark, which is consistent with what TIGER reported on Beauty and Sports. The item-as-text LLM lands in between: it beats SASRec modestly but is hobbled by the grounding/matching loss and the verbose representation, and it is the *slowest* to serve because text targets are longer to generate. And the two-tower model is the **cost-and-latency winner** by an order of magnitude on both axes — 20-30x cheaper to train, 15-20x faster to serve.

That is the entire trade-off in four columns. On Amazon Beauty — a domain with rich item semantics (skincare ingredients, brands, product types), high catalog churn, and a strong cold-start tail — the LLM's semantic understanding earns its keep and the accuracy lift is real. On a domain where behavior is dense, the catalog is stable, and the funnel's first stage runs at tens of thousands of QPS under a 10ms budget, the two-tower's latency and cost advantage is decisive and the accuracy gap does not justify the bill. **The LLM wins on cold and semantic; the two-tower wins on dense and latency-critical.**

#### Worked example: is the accuracy lift worth the serving cost?

Suppose a feed serves 50,000 retrieval QPS. With the two-tower at 9ms p99 you comfortably fit on a modest fleet of CPU-or-light-GPU hosts. With the semantic-ID LLM at 130ms p99, each request occupies a GPU for ~14x longer, so to hold the same QPS you need on the order of 14x the GPU-equivalent serving capacity — and these are real GPUs, not CPU hosts. If the two-tower fleet costs, say, \$8,000/month, the LLM retrieval fleet is heading toward six figures a month. The accuracy lift is a ~17% relative Recall@10 improvement. Is a 17% relative retrieval-recall lift worth a roughly 10x serving-cost increase on the hot path? For most surfaces, no — you would instead run the cheap two-tower for first-stage retrieval and spend the LLM budget *one stage down* in ranking or re-ranking, where you score a few hundred candidates instead of searching the whole catalog, and the latency math is far kinder. For a cold-start-heavy, semantics-rich, lower-QPS surface, the answer flips. The point of the worked example is that you cannot answer "should I finetune an LLM recommender?" from the metric column alone; the cost and latency columns decide it.

## 10. Case studies and the production-cost reality

**TALLRec (Bao et al., 2023).** The paper that made LoRA finetuning for recommendation credible. They finetuned a LLaMA-7B with LoRA on a tiny instruction-tuning set framed as "given the user's liked and disliked items, will they like this candidate?" The headline result was data efficiency: with only a few hundred to a few thousand training examples, the LoRA-tuned LLM beat strong traditional baselines in the few-shot regime, and it transferred across domains (train on movies, evaluate on books) far better than a from-scratch model. The lesson: the base model's world knowledge is a strong prior, and LoRA is enough to specialize it cheaply. This is the post's central practical claim made concrete.

**TIGER (Rajput et al., 2023, "Recommender Systems with Generative Retrieval").** The paper that introduced RQ-VAE semantic IDs and generative retrieval to recommendation. On the Amazon Beauty, Sports, and Toys benchmarks they showed the generative semantic-ID model beat strong sequential baselines (including SASRec) on Recall@K and NDCG@K, and — the part people underrate — that semantic IDs gave meaningful gains on *cold-start* items because new items reuse known codes. Everything in sections 6 and 7 traces back to this paper. The cost they were honest about is the autoregressive serving latency.

**P5 (Geng et al., 2022, "Recommendation as Language Processing").** The "pretrain, personalized prompt, predict" framework that unified rating prediction, sequential recommendation, explanation, and review tasks into a single text-to-text model with task-specific prompts. P5 is the strongest statement of the "everything is a language task" view and explored the atomic-ID-as-token representation among others. It is the conceptual ancestor of treating recommendation as generation.

**LC-Rec (Zheng et al., 2023/2024, "Adapting Large Language Models by Integrating Collaborative Semantics").** Directly addresses the gap this whole post circles: the *semantic* representation a base LLM has (from text) and the *collaborative* representation it needs (from behavior) are misaligned. LC-Rec proposes alignment tasks — including learning semantic IDs and a set of auxiliary objectives that tie language semantics to collaborative signal — so the LLM's item codes reflect both content and co-occurrence. It is the natural next step after TIGER if you find that pure content-derived codes leave collaborative signal on the table.

**The production-cost reality.** The published accuracy wins are real and reproducible on academic benchmarks. What the papers do not foreground, and what you discover in production, is that the dominant cost of an LLM recommender is *serving*, not training. Training a LoRA adapter is a one-time cost measured in GPU-hours; serving an autoregressive 7B model on the hot path is a recurring cost measured in GPU-months, and it scales with your traffic. This is why most shipped systems that adopt these ideas put the LLM where the candidate count is small — re-ranking a few hundred items, generating explanations, powering a conversational interface — rather than first-stage retrieval over the full catalog. The exception is exactly the regime TIGER targets: medium-scale catalogs where the index-free, cold-start-friendly generative approach simplifies operations enough to justify the latency. Know which regime you are in before you commit a quarter to it.

There is a second cost reality that bites later: **staleness and retraining cadence.** A two-tower model's item embeddings can be refreshed incrementally — embed today's new items, insert them into the index, done, no retraining. A finetuned LLM recommender is heavier to keep current. New items need semantic IDs (run them through the frozen RQ-VAE — cheap), but if your catalog or user behavior drifts meaningfully, you eventually re-finetune the adapter, which is another 40–60 GPU-hours and a validation cycle. For a fast-moving catalog this is a real operational tax that the academic benchmarks, which evaluate a single static split, never surface. Budget for it: decide your retraining cadence up front (weekly? monthly?), and confirm the accuracy lift survives the staleness between retrains, because a model that is brilliant the day it ships and mediocre three weeks later is worse than a boring two-tower that you refresh nightly.

**A note on where the frontier API models fit.** Everything above assumes open weights so you can attach LoRA and add code tokens. With an API-only model — Claude Opus 4.x, Claude Sonnet — you generally cannot do RQ-VAE semantic IDs or constrained trie decoding, because those require control over the vocabulary and the logits. What these models *are* exceptionally good at is the stages where you do not need that control: re-ranking a supplied candidate list with rich reasoning over user context, generating natural-language explanations for why an item was recommended, and powering a conversational recommender where the user negotiates ("show me something cheaper, but still waterproof"). The pattern in production is increasingly hybrid: a cheap finetuned or traditional model does first-stage retrieval over the full catalog and guarantees grounding, and a frontier API model does the reasoning-heavy re-ranking or conversation over the small grounded candidate set. You get the API model's reasoning without paying its latency on the hot path or fighting the grounding problem, because the candidates are already real items. That hybrid is, for many teams, the most pragmatic way to bring a strong LLM into the funnel today.

## 11. When finetuning an LLM is worth it (and when a two-tower wins)

This is the decision section, and I will be blunt because the honest answer saves you a quarter of wasted work.

**Finetune an LLM recommender when:**

- Your items are **semantically rich** — text descriptions, attributes, content that a language model genuinely understands — so the base model's world knowledge transfers. Skincare, books, articles, recipes, jobs: yes. Anonymous opaque IDs with no content: the LLM has little to grab onto.
- You have a **brutal cold-start problem** — high catalog churn, a long tail of items with little behavioral history — where semantic IDs and content understanding beat a two-tower that needs interactions to learn an item's embedding.
- Your surface is **not latency-critical at extreme QPS**: a daily email digest, a "for you" page recomputed offline, a conversational shopping assistant, or a re-ranking stage over a few hundred candidates. Anywhere the autoregressive decode fits the budget.
- You want **one model to do many tasks** — retrieval, ranking, explanation, conversation — and the unification (P5-style) is worth more than per-task optimization.

**Reach for a two-tower (or SASRec, or DCN ranker) instead when:**

- You are doing **first-stage retrieval at high QPS** under a single-digit-millisecond budget. Nothing beats a dot-product plus HNSW here, and the LLM's latency disqualifies it.
- Your interaction data is **dense and your catalog is stable**, so collaborative filtering has plenty of signal and the LLM's semantic edge is small. The 17%-relative-Recall lift evaporates when the baseline already has rich behavioral data.
- Your **serving budget is tight**. A 10x serving-cost increase for a modest accuracy lift is a bad trade on most surfaces; spend that budget on a better ranker downstream.
- You **lack the interaction corpus** to finetune well *and* the zero-shot path is good enough for your stage. Sometimes prompting a strong model (Claude Sonnet, a hosted open model) for cold-start candidates and letting a cheap ranker take over is the pragmatic stack.

The meta-rule: an LLM recommender is rarely a drop-in replacement for your whole funnel. It is a *tool for specific stages* — the cold/semantic retrieval source, the smart re-ranker, the conversational layer — that you compose with the cheap, fast traditional models doing the heavy QPS lifting. The mistake is treating "finetune an LLM" as the answer to "build a recommender." The right question is always "which stage of my funnel does this win, and at what cost?"

## 12. Key takeaways

- **A base LLM knows the world, not your system.** Finetuning on interaction logs is how you inject catalog grounding, the collaborative signal, and your actual objective — none of which live in the pretraining corpus.
- **The item representation is the decision that caps your ceiling.** Item-as-text is verbose and ungrounded; atomic IDs are grounded but cold-start-blind and share nothing; semantic IDs (RQ-VAE codes) are compact, grounded under a trie, and generalize. Default to semantic IDs for a generative recommender.
- **Only the response tokens carry loss.** Mask the prompt with `-100`. The most common silent SFT bug is grading the model on copying the instruction; your loss looks fine while Recall sits at the floor.
- **LoRA makes the finetune affordable.** Freezing the base and training a rank-16 adapter cuts trainable parameters to roughly 0.2-0.3% — about 17M of a 7B model — so QLoRA fits on one 24GB GPU. Merge the adapter at inference for zero added latency, or hot-swap adapters to serve many tasks on one base.
- **Generative retrieval trades the ANN index for decoding latency.** TIGER emits semantic IDs token by token under a trie constraint, guaranteeing grounded recommendations with no vector store — but a beam decode is 100-200ms versus single-digit ms for ANN. Choose by your QPS and latency budget.
- **The LLM wins on cold and semantic; the two-tower wins on dense and latency-critical.** On Amazon Beauty the semantic-ID model lifts Recall@10 ~15-20% relative over a tuned SASRec, at roughly 10x the serving cost. Read the cost and latency columns, not just the accuracy column.
- **Evaluate on the full catalog with a temporal split.** Sampled metrics can reorder methods (KDD 2020); a generative model ranks over all valid codes naturally, so report the full metric and always include a latency column.
- **Most production wins put the LLM one stage down.** Use it for re-ranking a few hundred candidates, cold-start retrieval, or conversation — composed with cheap, fast traditional models on the high-QPS hot path — not as a full-funnel replacement.

## Further reading

- Bao et al., 2023, "TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation" — the LoRA-finetuning-for-rec foundation and the data-efficiency result.
- Rajput et al., 2023, "Recommender Systems with Generative Retrieval" (TIGER) — RQ-VAE semantic IDs and generative retrieval with constrained decoding.
- Geng et al., 2022, "Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)" — recommendation as a unified text-to-text task.
- Zheng et al., 2023/2024, "Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation" (LC-Rec) — aligning language semantics with collaborative signal in item codes.
- Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"; Dettmers et al., 2023, "QLoRA: Efficient Finetuning of Quantized LLMs" — the PEFT methods that make this affordable.
- Krichene and Rendle, 2020, "On Sampled Metrics for Item Recommendation" (KDD) — why sampled top-K metrics can mislead.
- Hugging Face `peft` documentation and `transformers` generation/`LogitsProcessor` docs — the official APIs for the code in this post.
- Within this series: the deep dive on [LLMs for recommendation (LLM4Rec)](/blog/machine-learning/recommendation-systems/llms-for-recommendation-llm4rec), the road from [autoencoders to generative retrieval](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval), [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation), the transformer backbone in [self-attention for sequences (SASRec and BERT4Rec)](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook). For the general PEFT toolkit, see [effective LLM finetuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques).
