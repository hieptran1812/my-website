---
title: "Choosing the right LLM architecture for the task: a decision framework for encoder, decoder, MoE, and beyond"
date: "2026-06-05"
publishDate: "2026-06-05"
description: "A principal-engineer's decision framework for matching an LLM architecture to the task: encoder vs decoder vs encoder-decoder, bi- vs cross-encoder, dense vs MoE, MHA/MQA/GQA/MLA, attention vs SSM hybrids, positional encoding for long context, and compute- vs inference-optimal sizing — with nine production case studies."
tags: ["llm", "architecture", "transformer", "encoder-decoder", "mixture-of-experts", "attention", "gqa", "mla", "long-context", "scaling-laws", "rope", "deep-learning"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

The fastest way to lose three months and a quarter's GPU budget is to start a project by picking a model instead of an architecture. Someone reads a leaderboard, fine-tunes a 70B decoder, wires it behind a queue, and only discovers at load-test time that the task never needed a generative model at all — it needed an encoder that fits on a single GPU and answers in 4 milliseconds. I have watched this exact failure ship to production more than once, and the root cause is always the same: the team treated "which LLM" as a single decision when it is really a stack of five or six independent ones.

"LLM" has quietly become a synonym for "autoregressive decoder," and that conflation is the bug. The transformer is a family, not a model. BERT, GPT, T5, Whisper, Mixtral, DeepSeek-V3, Mamba, and a sentence-embedding model are all "transformers" (or transformer-adjacent), and they make *opposite* architectural commitments because they are solving *opposite* problems. Choosing well means refusing to collapse those commitments into one axis labelled "size."

This article is the decision framework I wish every team used before they touch a checkpoint. It is organized as a sequence of questions — what shape is the information flow, do you need joint scoring, do you need capacity at fixed serving cost, how long is the context, how many tokens will you ever serve — and each question maps to an architectural axis with a concrete, quantifiable tradeoff. We will go deep on the mechanism behind each choice, because the only way to make these calls under pressure is to understand *why* the architecture behaves the way it does, not to memorize a flowchart.

![The architecture decision space: a task's information-flow shape selects the family, then you tune activation, attention, context, and size](/imgs/blogs/choosing-right-llm-architecture-task-1.webp)

The diagram above is the mental model. Read it left to right: your task has a *shape* — does it primarily *understand* an input, *generate* an output, or *transform* one sequence into another — and that shape picks the family. Everything downstream of the family (dense vs sparse, how attention is wired, how long the context window is, how big the model needs to be) is a second set of dials you turn once the family is fixed. The mistake is reaching for the dials before you have chosen the lane. The rest of this post is a tour of that diagram, one axis at a time, and it leans on companion deep-dives like [modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) and [Mixture-of-Experts LLMs](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) for the parts that deserve a book of their own.

## Why architecture choice is different from model choice

Model choice is a leaderboard sport: bigger eval scores, newer checkpoint, win. Architecture choice is an *engineering* decision, and engineering decisions are dominated by constraints the leaderboard never measures — latency budget, cost per million tokens, memory ceiling on your actual GPU, whether you can cache, whether you can batch. The two decisions feel similar because they both end with "we'll use model X," but they are made with different information and they fail in different ways.

| What you assume | The naive default | The reality on the cluster |
|---|---|---|
| "An LLM can do any text task" | Prompt a big decoder for everything | A fine-tuned 300M encoder beats a prompted 70B decoder on classification at 1/1000 the cost and 1/50 the latency |
| "Bigger model = better" | Pick the largest checkpoint that fits | Compute-optimal sizing is rarely deployment-optimal; an over-trained small model often wins on lifetime cost |
| "MoE is a smaller model" | Budget VRAM for the *active* params | MoE saves FLOPs, not memory — the whole expert bank is resident, so Mixtral 8×7B needs ~47B params of VRAM, not 13B |
| "More context window is strictly better" | Buy the 128K-context checkpoint | Past the trained length, recall collapses; "supports 128K" and "is accurate at 128K" are different claims |
| "Retrieval just needs embeddings from the LLM" | Mean-pool a decoder's hidden states | Causal mean-pooling makes poor embeddings; retrieval wants a contrastively trained bi-encoder |
| "Attention is attention" | Ignore the KV head count | MHA vs GQA vs MLA changes KV-cache memory by 8–60×, which sets your max batch size and context |

Every row of that table is a story we will tell in full, several of them with production post-mortems at the end. The thread connecting them: **the leaderboard optimizes quality at infinite budget; you are optimizing quality subject to a latency SLA and a VRAM ceiling.** Architecture is how you buy quality back under those constraints, and the right architecture for the task is frequently *not* the one at the top of the leaderboard.

One more framing before we descend into the axes. Throughout, I will distinguish three cost regimes that pull in different directions:

- **Training cost** — one-time, paid in GPU-hours. Dominated by total FLOPs ≈ $6 N D$ for $N$ parameters and $D$ training tokens.
- **Memory cost** — the ceiling. Set by *total* parameters plus the KV cache, and it decides what fits on your GPU and how large a batch you can run.
- **Inference cost** — recurring, paid per token forever. Dominated by *active* parameters per token, memory bandwidth, and how well you can batch.

Architectures trade these against each other. MoE trades memory for inference FLOPs. GQA trades a little quality for memory bandwidth. Over-training trades training cost for inference cost. You cannot choose well until you know which of the three regimes dominates your bill — and for most production systems that serve at scale, it is inference, by a wide margin. The companion piece on [self-hosting vs API economics](/blog/machine-learning/large-language-model/estimating-llm-capacity-and-self-host-vs-api) works the unit economics if you want the spreadsheet.

## 1. Information flow: which directions can attention see?

The first and most consequential fork is also the one most teams skip. Before you ask how big or how fast, ask: **what does each token need to see?** A token-level classifier needs every token to see the whole sentence in both directions. A text generator must never let a token see the future, or it would cheat at training time and have nothing to predict. A translator needs the target sequence to read the entire source while still generating left-to-right. Those three requirements are not preferences — they are mutually exclusive attention patterns, and they define the three families.

![Attention masks: bidirectional (encoder), causal lower-triangular (decoder), and cross-attention (encoder-decoder)](/imgs/blogs/choosing-right-llm-architecture-task-2.webp)

The figure shows the only thing that actually differs at the core of these three families: the attention mask. An **encoder** uses a full (bidirectional) mask — token $i$ attends to every token $j$, so the representation of "bank" can be informed by "river" three words later. A **decoder** uses a causal mask — token $i$ attends only to $j \le i$ — which is what makes left-to-right generation well-defined and lets you train on every position in parallel with a single forward pass. An **encoder-decoder** runs a bidirectional encoder over the source and a causal decoder over the target, with a *cross-attention* block where each target token reads the full encoded source.

### The mechanism, and why it dictates the task

Bidirectionality is the encoder's superpower and its limitation. Because every token sees the whole sequence, an encoder builds a deeply contextualized representation of *the input you already have* — ideal for understanding tasks where the output is a label, a span, or a vector: sentiment, intent, named-entity recognition, extractive QA, reranking, and embeddings for retrieval. But that same bidirectionality means an encoder cannot generate text autoregressively; there is no "next token" when every token already sees every other. BERT, RoBERTa, and DeBERTa-v3 are the canonical encoders, pretrained with masked-language-modeling (MLM): corrupt 15% of tokens, predict them from both sides.

Causality is the decoder's superpower. The causal mask makes "predict the next token given the past" a coherent objective at every position, which is why a single sequence yields thousands of training signals and why decoders scale so gracefully. It also means a decoder is natively generative — chat, code, agentic tool-use, multi-step reasoning. The cost is that a decoder's representation of token $i$ is built without ever seeing token $i+1$, which is exactly why naively using a decoder's hidden states as embeddings underperforms a purpose-built encoder (more on that in §2). GPT, Llama, Qwen, Mistral, and Gemma are all decoder-only.

The encoder-decoder splits the difference for *transduction* tasks — a fixed source maps to a target sequence in a different space: translation (NLLB), summarization (Pegasus, BART), speech-to-text (Whisper), and OCR (Donut). The encoder gets a clean bidirectional read of the whole source; the decoder generates the target while cross-attending to that read. At a fixed parameter budget, this division of labor often beats a decoder-only model on transduction, because the encoder is not wasting causal capacity re-deriving the source from a prefix.

### The capability matrix

Here is the mapping I actually use, with the pretraining objective that makes each family good at its lane:

| Task | Native family | Why | Representative models |
|---|---|---|---|
| Sentiment / intent / topic classification | Encoder-only | Bidirectional read → label head | DeBERTa-v3, RoBERTa |
| Named-entity recognition / token tagging | Encoder-only | Per-token bidirectional context | DeBERTa-v3, BERT |
| Extractive QA (span selection) | Encoder-only | Start/end pointers over context | DeBERTa-v3 |
| Dense retrieval (embeddings) | Encoder (bi-encoder) | Contrastive, cacheable vectors | E5, BGE, GTE |
| Reranking (pair scoring) | Encoder (cross-encoder) | Joint cross-attention on (q, d) | BGE-reranker, MiniLM cross-encoder |
| Open-ended generation / chat | Decoder-only | Causal LM is natively generative | Llama, Qwen, GPT |
| Code / agentic / reasoning | Decoder-only | Long autoregressive rollouts | Qwen-Coder, DeepSeek-R1 |
| Translation | Encoder-decoder (or decoder) | Source read + target generate | NLLB, T5; large decoders close the gap |
| Speech recognition | Encoder-decoder | Audio encoder + text decoder | Whisper |
| Summarization | Encoder-decoder or decoder | Transduction; both viable | Pegasus, BART; modern decoders |

Two honest caveats keep this table from being dogma. First, **scale blurs the boundaries.** A sufficiently large, well-instructed decoder can do classification, NER, and translation acceptably via prompting — it is just paying a large tax for the privilege. The question is never "can a decoder do this task" (it can) but "is a decoder the *cheapest correct* tool for this task" (often it is not). Second, the field has drifted decoder-ward for generative tasks even where encoder-decoders historically led, because decoder-only training is simpler, scales cleaner, and benefits from the entire instruction-tuning ecosystem. For translation specifically, a 7B+ instruction-tuned decoder is now competitive with dedicated encoder-decoders — but at 600M parameters, NLLB still wins decisively, and that matters when you are shipping to a phone.

### The spectrum between the families

The three masks are the clean corners of a design space that has a populated interior. **Prefix-LM** (used by some PaLM variants and the UL2 recipe) applies bidirectional attention over a prefix and causal attention over the completion, which lets a single model read an instruction with full context and then generate — a middle ground between encoder-decoder cross-attention and pure causal decoding. **UL2's mixture-of-denoisers** trains one model on a blend of MLM-style and causal-LM-style objectives so it can be prompted into either mode. **GLM** uses autoregressive blank-infilling. The practical takeaway is not that you will train one of these from scratch — you almost never will — but that the encoder/decoder boundary is a dial, not a wall, and that is precisely why adapting a decoder into a bidirectional encoder (the LLM2Vec recipe in §2) or extending a decoder with a classification head are both viable. When you read that a model is "decoder-only," that describes its pretraining mask; what you fine-tune it into is a separate decision. Hold onto that flexibility, because it is what lets a single strong base model serve several of the lanes above with the right head bolted on.

### Picking the head, not just the body

A subtlety that trips up otherwise-strong engineers: the *body* (encoder/decoder) and the *head* (the small task-specific layers on top) are separate choices. An encoder body with a classification head is a classifier; the same body with a token-classification head is a tagger; with a span head it does extractive QA. In code, the `transformers` library makes the head the explicit choice:

```python
from transformers import (
    AutoModelForSequenceClassification,  # encoder body + pooled classification head
    AutoModelForTokenClassification,     # encoder body + per-token head (NER)
    AutoModelForCausalLM,                # decoder body + LM head (generation)
    AutoModelForSeq2SeqLM,               # encoder-decoder + LM head (transduction)
)

clf = AutoModelForSequenceClassification.from_pretrained(   # 184M DeBERTa-v3, CPU-servable <10 ms
    "microsoft/deberta-v3-base", num_labels=3)
gen = AutoModelForCausalLM.from_pretrained(                 # a decoder: 100-1000x FLOPs per answer
    "Qwen/Qwen2.5-7B-Instruct")
mt = AutoModelForSeq2SeqLM.from_pretrained(                 # encoder-decoder MT at 600M params
    "facebook/nllb-200-distilled-600M")
```

The senior rule of thumb: **if your output is a fixed label, a span, a vector, or a score, you almost certainly want an encoder body and you should stop reading leaderboards for 70B decoders.** The discriminative head is tiny, the body is small enough to fit anywhere, and you will fine-tune it on your own labels in an afternoon. Reserve the decoder for when the output is genuinely open-ended text.

#### Second-order optimization: the prompting-a-decoder anti-pattern

The reason teams reach for a decoder on discriminative tasks is that prompting feels free — no labels, no training loop, just an API call. The hidden cost shows up at scale. A prompted decoder pays for the entire input *and* the generated output tokens, runs a model 100–1000× larger than needed, cannot be quantized as aggressively without hurting generation quality, and gives you a string you must parse rather than a calibrated probability. A fine-tuned encoder gives you a logit you can threshold, runs on a CPU or a fraction of a GPU, and is trivially batchable. The break-even is brutal: at even modest volume, the encoder's one-day fine-tuning cost is repaid within hours of serving. We will see this exact trade blow up in Case Study 5.

## 2. Retrieval and ranking: bi-encoder vs cross-encoder

Once you are in the encoder lane for understanding tasks, there is a second fork that is responsible for more bad RAG systems than any other single decision: **how does the encoder consume the query and the document — separately or together?** The answer determines whether you can cache, which determines whether your system survives contact with production latency.

![Bi-encoder vs cross-encoder: independent cacheable embeddings vs joint per-pair cross-attention scoring](/imgs/blogs/choosing-right-llm-architecture-task-3.webp)

A **bi-encoder** (dual encoder) runs the query and the document through the *same* encoder *independently*, producing two vectors, and scores their relevance with a cheap similarity — cosine or dot product. Because the document vector does not depend on the query, you can embed your entire corpus *once*, offline, build an approximate-nearest-neighbor (ANN) index, and at query time embed only the query and do a sub-millisecond vector search over millions of documents. This is the architecture behind every production [embedding model](/blog/machine-learning/large-language-model/embedding-models-training-finetuning-case-studies) — E5, BGE, GTE — and it is the only thing fast enough to be the *retrieval* stage.

A **cross-encoder** concatenates the query and document into a single sequence — `[CLS] query [SEP] document [SEP]` — and runs them jointly through the encoder, so every query token cross-attends to every document token. That joint attention is exactly what makes it more accurate: it can model fine-grained term interactions a single dot product cannot. The catch is that the score depends on the *pair*, so there is nothing to precompute — you must run a full encoder forward pass for every (query, document) you want to score. Cross-encoders are the [reranker](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies) — accurate, and far too slow to run over a corpus.

### The math of why you cannot skip the two-stage design

Say you have a 10M-document corpus and a cross-encoder that scores one pair in 5 ms on your GPU. Scoring all 10M pairs for a single query takes 50,000 seconds — about 14 hours. That is not a tuning problem; it is an architectural one. The bi-encoder makes it tractable: embed the query once (5 ms), ANN search over 10M precomputed vectors (~2 ms with a good HNSW index), retrieve the top 100, then run the cross-encoder on just those 100 pairs (500 ms). Two stages, ~510 ms total, and the quality is close to scoring everything because the bi-encoder rarely drops a truly relevant document from the top 100.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

bi = SentenceTransformer("BAAI/bge-base-en-v1.5")  # Stage 1: bi-encoder, embed corpus ONCE offline
corpus = [...]                              # 10M documents
doc_emb = bi.encode(corpus, normalize_embeddings=True)   # cache to an ANN index

def search(query, k_retrieve=100, k_final=10):
    q = bi.encode(query, normalize_embeddings=True)
    # Sub-ms ANN in practice (FAISS/HNSW); brute force shown for clarity.
    scores = doc_emb @ q
    cand = np.argpartition(-scores, k_retrieve)[:k_retrieve]

    # Stage 2: cross-encoder. Re-score ONLY the 100 candidates jointly.
    ce = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [(query, corpus[i]) for i in cand]
    rerank = ce.predict(pairs)               # full attention over (q, d)
    order = cand[np.argsort(-rerank)]
    return order[:k_final]
```

The senior rule of thumb: **bi-encoder to retrieve, cross-encoder to rerank, never the reverse and never just one.** A bi-encoder alone gives you fast-but-fuzzy top-k; a cross-encoder alone is accurate-but-unservable. The two-stage cascade is not an optimization you add later — it is the correct architecture from day one, and Case Studies 1 and 2 are both teams that learned this the expensive way.

#### Second-order optimization: do not embed with a generative decoder

The most common embedding mistake is using the wrong *body* entirely. Teams reach for the decoder LLM they already have — "it's an LLM, it makes representations" — and mean-pool its last-layer hidden states, or grab the final token's vector, and call it an embedding. It works badly. A causal decoder's token representations are built without seeing the future, the last-token vector is dominated by next-token-prediction features rather than whole-sequence semantics, and nothing in pretraining ever pushed two paraphrases to have nearby vectors. Embeddings come from *contrastive* training — pulling positives together and pushing negatives apart — which is a different objective than language modeling. The modern compromise (LLM2Vec, NV-Embed, e5-mistral) is to take a decoder, switch it to bidirectional attention, and *contrastively fine-tune* it — at which point it is no longer "a decoder," it is an encoder you grew from a decoder. The architecture you serve is what matters, not the checkpoint you started from.

## 3. Dense vs Mixture-of-Experts: paying for capacity

You have chosen a decoder for a generative task. The next dial decides whether every parameter fires on every token (**dense**) or whether a router activates a small subset per token (**sparse / Mixture-of-Experts**). This is the axis with the most counterintuitive economics, and the one teams get wrong most expensively, because MoE markets itself as "a bigger model that runs like a smaller one" and that sentence is only half true.

![Dense FFN activates all parameters; MoE routes each token to top-k experts but keeps the whole bank resident in VRAM](/imgs/blogs/choosing-right-llm-architecture-task-4.webp)

In a dense transformer, the feed-forward block multiplies every token by the same large weight matrices — all parameters are active for all tokens. In an **MoE**, the single big FFN is replaced by $E$ smaller "expert" FFNs plus a **router** (a small gating network) that picks the top-$k$ experts per token. Mixtral 8×7B has 8 experts and routes each token to the top 2; DeepSeek-V3 has 256 routed experts plus shared experts and activates 8. The token only pays the FLOPs of the $k$ experts it visited, so a model with 47B or 671B *total* parameters does the per-token compute of a ~13B or ~37B dense model.

### The trade you are actually making

The seductive part is real: at a fixed *training and inference FLOP budget*, MoE buys you more parameters, and more parameters means more knowledge capacity, so MoE models punch above their active-parameter weight on quality. The companion deep-dive on [Mixture-of-Experts](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) and its [optimization sequel](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) cover training these well. But here is the part the marketing buries:

- **MoE saves FLOPs, not memory.** Every expert must be resident in VRAM because the router can send the *next* token to any of them. Mixtral 8×7B needs ~47B parameters of weights loaded (~94 GB in fp16, ~47 GB in int8) even though each token only touches ~13B. You provision for the *total*, you pay compute for the *active*.
- **Routing adds system complexity.** Load balancing across experts (auxiliary losses, token dropping), all-to-all communication when experts are sharded across GPUs (expert parallelism), and capacity-factor tuning are real engineering, not free lunch.
- **Small-batch inference can underutilize experts.** At batch size 1 with top-2 of 256, you are doing many tiny GEMMs; throughput-per-FLOP is worse than a dense model until you batch enough tokens to fill the experts.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoEFFN(nn.Module):
    def __init__(self, d_model=4096, d_ff=14336, n_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(),
                           nn.Linear(d_ff, d_model)) for _ in range(n_experts)]
        )
        self.top_k = top_k

    def forward(self, x):                      # x: [tokens, d_model]
        logits = self.router(x)                # [tokens, n_experts]
        w, idx = torch.topk(logits.softmax(-1), self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):          # gather/scatter per selected expert
            for e in range(len(self.experts)):
                mask = idx[:, slot] == e
                if mask.any():
                    out[mask] += w[mask, slot:slot+1] * self.experts[e](x[mask])
        return out

d_model, d_ff, n_experts, top_k = 4096, 14336, 8, 2   # the economics in two numbers
params_per_expert = 2 * d_model * d_ff
active = top_k * params_per_expert            # FLOPs scale with THIS
resident = n_experts * params_per_expert      # VRAM scales with THIS
print(f"active/token: {active/1e9:.1f}B-equiv   resident in VRAM: {resident/1e9:.1f}B-equiv")
    # -> active/token: 0.2B-equiv  resident in VRAM: 0.9B-equiv  (per layer; x n_layers for model)
```

The senior rule of thumb: **reach for MoE when you are FLOP- or latency-bound and have VRAM to spare; stay dense when you are memory-bound or serving at small batch on constrained hardware.** MoE is a fantastic deal in a datacenter with H100s and high concurrency, and a trap on a single 24 GB card. Case Study 3 is a team that read "8×7B ≈ 13B active" and provisioned a 13B-sized box.

### Not all MoE is Mixtral-shaped

The MoE design itself has axes worth knowing, because they change the memory/quality trade. Early MoE (Switch Transformer, Mixtral) used a handful of large experts with top-1 or top-2 routing. DeepSeek-V2/V3 popularized **fine-grained experts** — many small experts (256 routed in V3) with a higher top-$k$ (8 activated) — which gives the router a richer combinatorial space (choosing 8 of 256 is vastly more expressive than 2 of 8) for the same active-parameter count, and empirically improves the quality-per-active-FLOP. They also add **shared experts** that every token always visits, capturing common patterns so the routed experts can specialize, and a load-balancing scheme (auxiliary-loss-free in V3) that keeps any single expert from being overwhelmed. The **capacity factor** — how many tokens each expert is allowed to process per batch before tokens are dropped — is a serving knob: too low and you drop tokens (silent quality loss), too high and you waste compute on padding. None of this changes the headline trade (total params live in VRAM, active params set FLOPs), but it does mean "MoE" spans a range from "8 chunky experts" to "256 tiny experts plus shared," and the fine-grained end is where the modern frontier models live. The [MoE optimization deep-dive](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) covers the all-to-all dispatch and GroupedGEMM machinery that make the fine-grained variant fast.

#### Second-order optimization: MoE fine-tuning and the router

One non-obvious gotcha: fine-tuning an MoE is not like fine-tuning a dense model. The router learned a token→expert assignment during pretraining; aggressive fine-tuning on a narrow domain can collapse routing (most tokens going to a few experts), which wastes the capacity you paid for. Practical fixes are to keep the auxiliary load-balancing loss on during fine-tuning, use a lower learning rate on the router, or freeze the router entirely and adapt only the experts (or apply LoRA to the experts). If your downstream task is narrow, a *dense* model is often easier to fine-tune and deploy — the capacity advantage of MoE is most valuable for broad, general-purpose models.

## 4. Attention bandwidth: MHA, MQA, GQA, MLA

Now the dial that quietly decides your maximum batch size and context length: **how many key/value heads does attention use, and how big is the KV cache they produce?** During autoregressive decoding, every previously generated token's keys and values are cached and re-read for every new token. That cache is frequently the *largest* consumer of VRAM at long context, and re-reading it is a memory-bandwidth bottleneck that dominates decode latency. The choice of attention variant changes the size of that cache by up to two orders of magnitude.

![MHA, MQA, GQA, and MLA wire query heads to key/value heads differently, shrinking the KV cache from full to a latent](/imgs/blogs/choosing-right-llm-architecture-task-5.webp)

In **multi-head attention (MHA)**, each of the $H$ query heads has its own key and value head — the original design, maximal expressiveness, and a KV cache proportional to $H$. **Multi-query attention (MQA)** keeps all $H$ query heads but shares a *single* key/value head, shrinking the cache by a factor of $H$ at some quality cost. **Grouped-query attention (GQA)** is the compromise that won: $G$ groups of query heads share one KV head each, so the cache is proportional to $G$ (e.g. 8 KV heads for 64 query heads in Llama-2-70B), recovering most of MHA's quality at a fraction of the memory. **Multi-head latent attention (MLA)**, introduced in DeepSeek-V2/V3, goes further — it compresses keys and values into a low-rank latent vector that is cached instead of full KV heads, shrinking the cache ~10× while keeping (or improving) quality through a learned up-projection. The [KV cache deep-dive](/blog/machine-learning/large-language-model/kv-cache) covers the serving-side machinery in detail.

### Quantifying the cache

The KV cache size for a single sequence is:

$$\text{bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times b_{dtype}$$

where $L$ is layers, $H_{kv}$ is key/value heads, $d_{head}$ is head dimension, $S$ is sequence length, the leading 2 covers K and V, and $b_{dtype}$ is bytes per element. The only term the architecture changes is $H_{kv}$ — and it changes it a lot.

```python
def kv_cache_gb(layers, kv_heads, head_dim, seq_len, batch, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes / 1e9

mha = kv_cache_gb(80, 64, 128, 8192, 32)   # 70B-class: 80 layers, d_head 128, 8k ctx, batch 32
gqa = kv_cache_gb(80,  8, 128, 8192, 32)   # GQA-8:     8 KV heads
mqa = kv_cache_gb(80,  1, 128, 8192, 32)   # MQA:       1 KV head
print(f"MHA {mha:.0f} GB | GQA-8 {gqa:.0f} GB | MQA {mqa:.0f} GB")
    # -> MHA 687 GB | GQA-8 86 GB | MQA 11 GB
```

That 687 GB → 86 GB jump from MHA to GQA is the difference between "needs 9 H100s just for the cache" and "fits the cache on one." This is not a micro-optimization; it is the reason every modern decoder (Llama-3 at all sizes, Qwen2.5, Mistral) ships with GQA, and why DeepSeek went all the way to MLA to push context and batch even higher. The senior rule of thumb: **at long context or high batch, KV-cache memory and bandwidth — not parameter count — set your throughput, so the attention variant is a first-class architectural choice, not a footnote.**

Why does this lever matter so much? Because inference has two phases with opposite bottlenecks. **Prefill** — encoding the prompt — is compute-bound: it runs attention over all input tokens in parallel and saturates the GPU's matrix units. **Decode** — generating tokens one at a time — is memory-bandwidth-bound: each step does little compute but must re-read the entire KV cache and all the weights from HBM. Since most of a request's wall-clock at generation time is decode, and decode speed is gated by how many bytes you move per token, shrinking the KV cache with GQA or MLA directly buys decode throughput — you are moving 8× fewer KV bytes per step. This is also why MLA pairs its latent compression with a *decoupled RoPE* trick: it keeps a small rotary component separate from the compressed latent so positional information survives the compression, getting both the bandwidth win and correct relative positions. The takeaway: the attention variant is not just a memory-footprint decision, it is a decode-latency decision, and decode is where your users feel the wait.

#### Second-order optimization: GQA group count and quantization stack on top

Two follow-ons worth knowing. First, the GQA group count is itself tunable — fewer groups means a smaller cache but more quality risk; 8 KV heads has emerged as a sweet spot for large models, but small models sometimes use 2–4. Second, KV-cache *quantization* (storing K and V in int8 or fp8 instead of fp16) composes multiplicatively with GQA: GQA-8 plus int8 KV gives you a 16× reduction over MHA-fp16. vLLM exposes this as `--kv-cache-dtype fp8`. When you are pushing context windows, you stack architectural sharing (GQA/MLA) with numerical compression (fp8 KV), and the [inference optimization guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) shows how to measure the quality cost.

## 5. The sequence mixer: attention, SSM, and hybrids

Everything so far has assumed the token-mixing operation is attention. But attention has a structural cost that no amount of head-sharing removes: it is $O(n^2)$ in sequence length for compute and produces a KV cache that grows with $n$. For very long sequences — genomic data, hour-long audio, book-length documents, high-throughput agents — that quadratic term and the linearly-growing cache become the binding constraint. A different family of **sequence mixers** trades exactness for linear scaling.

![Attention is O(n²) with exact recall; linear/SSM mixers are O(n) but lossy; hybrids interleave a few attention layers into an SSM stack](/imgs/blogs/choosing-right-llm-architecture-task-6.webp)

**State-space models (SSMs)** like Mamba and Mamba-2, along with **linear-attention** variants and **RWKV**, replace the all-pairs attention with a recurrence that carries a *fixed-size state* forward through the sequence. The consequences are dramatic and double-edged: compute is $O(n)$ instead of $O(n^2)$, and at inference the state is $O(1)$ in sequence length — there is no KV cache that grows with context, so a Mamba model can stream a million tokens at constant memory. The cost is **exact recall**. Attention can look back and copy a specific token verbatim (the mechanism behind in-context retrieval and "induction heads"); an SSM compresses the past into a fixed state and therefore *forgets* — it is reliably worse at tasks that require fishing a precise earlier token out of a long context, like copying a UUID or answering "what was the third item in the list 40K tokens ago."

### Why production stacks are hybrids

This is why the models that actually ship at long context are **hybrids**. Jamba interleaves Mamba layers, attention layers, and MoE; a typical ratio is roughly one attention layer per seven Mamba layers. The handful of attention layers restore exact recall and associative lookup, while the Mamba majority keeps compute linear and the cache small. RecurrentGemma (Griffin) and Falcon-Mamba make similar bets. The design question becomes a ratio: how much attention do you need to buy back the recall your task requires?

| Mixer | Compute (seq len $n$) | Inference state | Exact recall | Use when |
|---|---|---|---|---|
| Full attention | $O(n^2)$ | KV cache, grows with $n$ | Strong | Recall-critical, moderate context, default |
| GQA/MLA attention | $O(n^2)$ | Shrunk KV cache | Strong | Same, but long context or high batch |
| SSM / Mamba | $O(n)$ | $O(1)$ fixed state | Weak | Very long, throughput-bound, recall-tolerant |
| Linear attention / RWKV | $O(n)$ | $O(1)$ state | Weak–moderate | Streaming, edge, constant-memory decode |
| Hybrid (Jamba-style) | $\approx O(n)$ | Small cache + state | Moderate–strong | Long context where some exact recall matters |

The senior rule of thumb: **default to attention; reach for SSM/hybrid only when sequence length is the dominant cost and you can tolerate (or test away) the recall loss.** For the vast majority of chat, code, and RAG workloads — where contexts are tens of thousands of tokens and exact retrieval matters — full or GQA attention is still the right call. SSMs are a specialist tool for the long-and-throughput-bound tail, and Case Study 8 is a team that adopted a pure SSM, hit the recall wall, and recovered by going hybrid.

#### Second-order optimization: benchmark recall, not perplexity

The trap with SSMs and hybrids is that they look great on perplexity and average benchmarks — language modeling does not stress exact recall much — and then fail silently on the one task you actually shipped. If you are evaluating a sub-quadratic architecture, do not trust aggregate scores; run targeted recall probes: needle-in-a-haystack, multi-key associative recall, and exact-copy tasks at your real context length. The gap between "good perplexity" and "can retrieve a specific fact from 100K tokens" is exactly where these architectures live or die.

## 6. Context length and positional encoding

Closely related, and frequently conflated with the mixer choice, is the question of *how the model knows where each token is*. Transformers are permutation-invariant without positional information; the **positional encoding** scheme injects order — and, crucially, it decides whether the model degrades gracefully or falls off a cliff when you feed it sequences longer than it was trained on. "Supports 128K context" is a claim about the positional scheme and the long-context training, not a property you get for free by enlarging a buffer.

![Perplexity vs sequence length: vanilla RoPE cliffs past the trained length, YaRN-extended RoPE holds, ALiBi degrades gently](/imgs/blogs/choosing-right-llm-architecture-task-7.webp)

The dominant modern scheme is **RoPE** (rotary positional embeddings): it rotates the query and key vectors by an angle proportional to position, so attention scores depend on *relative* position. RoPE is excellent within the trained length and is what Llama, Qwen, Mistral, and most modern decoders use. Its weakness is extrapolation: feed a RoPE model positions beyond its training length and perplexity spikes — the rotation frequencies it never saw during training produce attention patterns it cannot interpret. The curve in the figure is the real shape: flat and low up to the trained length $L$, then a cliff.

### Extending context is an architecture-adjacent decision

You do not extend context by changing a config value; you change the *positional scheme's frequency basis* and (ideally) do a little continued training. The main techniques:

- **Position Interpolation (PI):** linearly scale positions down so a 4K-trained model "sees" 16K positions as if they were 4K. Cheap, but blunt — it compresses high-frequency detail.
- **NTK-aware scaling:** scale RoPE's base frequency ($\theta$) instead of positions, preserving high-frequency resolution better than PI.
- **YaRN:** a refined frequency interpolation that, with a few hundred steps of fine-tuning, cleanly extends RoPE models to 4–32× their trained length while holding quality — the green-ish "extended" curve. This is the workhorse behind most "128K" community models.
- **ALiBi:** an alternative scheme that adds a distance-proportional penalty to attention scores instead of rotating. It extrapolates *gracefully* — no cliff — but caps peak quality and has largely lost to RoPE+YaRN in practice; BLOOM and MPT used it.

```python
from transformers import AutoModelForCausalLM   # extend a RoPE model with YaRN-style scaling

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    rope_scaling={"type": "yarn", "factor": 4.0,            # 32K trained -> 128K
                  "original_max_position_embeddings": 32768},
    max_position_embeddings=131072,
)
    # Without fine-tuning this often "runs" but degrades; the factor plus a short
    # continued-pretrain on long documents are what actually preserve recall.
```

There is also a structural lever that sidesteps the positional question for part of the stack: **sliding-window attention (SWA)**. Instead of letting every token attend to the entire history, SWA restricts attention to a fixed window of the last $w$ tokens (Mistral used 4,096), which caps the KV cache at $w$ regardless of total sequence length and keeps per-token compute constant. Information still propagates beyond the window across layers — a token at the top of a deep stack can indirectly reach far-back tokens through the chain of windows — but exact long-range recall weakens, the same trade as the sub-quadratic mixers in §5. Gemma-2 and later models interleave *local* (windowed) and *global* (full) attention layers, getting the cache savings of windows on most layers and the exact recall of full attention on a few — structurally the same hybrid bet as Jamba, applied within the attention family rather than between attention and SSM. When you see "sliding window" or "local/global" in a model card, read it as the same lever pulled at a different point: trade some recall for bounded cache and linear cost.

The senior rule of thumb: **a long context window is a training-and-positional-encoding achievement, not a buffer size — verify recall at your target length before you trust it.** The benchmark that matters is not "does it accept 128K tokens" but "does needle-in-a-haystack and RULER hold at 128K." Case Study 6 is a team that shipped a "128K" model whose recall collapsed at 40K because nobody ran that test.

#### Second-order optimization: effective context vs advertised context

Even with perfect positional extension, models exhibit a "lost in the middle" effect — recall is strongest for tokens near the start and end of the context and weakest in the middle. This means the *effective* usable context is often well below the advertised number, and dumping 128K tokens into the prompt is frequently worse (and far more expensive) than retrieving the right 4K with a [good RAG pipeline](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system). The architecture lets you *fit* the context; it does not guarantee the model *uses* all of it. Long context and retrieval are complements, not substitutes — choose the cheapest combination that hits your recall target.

## 7. Sizing the model: compute-optimal vs inference-optimal

The last dial is the one everyone thinks they understand and most get backwards: **how big should the model be?** The reflex is "as big as fits / as big as the budget allows," anchored on the Chinchilla scaling laws. But Chinchilla answers a question most production teams are not asking, and the right answer for a system that serves billions of tokens is usually a *smaller* model than Chinchilla recommends.

![Compute-optimal (Chinchilla) minimizes training FLOPs; inference-optimal over-trains a smaller model to win on lifetime serving cost](/imgs/blogs/choosing-right-llm-architecture-task-8.webp)

The **Chinchilla** result (DeepMind, 2022) says: for a *fixed training compute budget*, the loss-minimizing split is roughly $D \approx 20$ training tokens per parameter $N$ — train a smaller model on more data than the earlier GPT-3-era models did. That is **compute-optimal**: it minimizes *training* FLOPs for a target loss. The subtlety is that "compute-optimal" only counts the one-time training bill. It says nothing about the recurring inference bill, which for a deployed system dwarfs training within weeks.

### The inference-optimal argument

If you are going to serve a model to many users for a long time, your real objective is lifetime cost:

$$\text{cost}_{\text{lifetime}} \approx \text{cost}_{\text{train}} + N_{\text{tokens served}} \times \text{cost}_{\text{inference per token}}$$

Inference cost per token scales with active parameters. So you can *trade training for inference*: deliberately **over-train a smaller model** — push it far past Chinchilla's 20 tokens/param — to reach the same quality as a larger compute-optimal model, then serve the smaller model forever at lower per-token cost. Llama-3-8B was trained on ~15T tokens — roughly 1,900 tokens per parameter, almost 100× past Chinchilla's recommendation — precisely because Meta optimized for the model people would *run*, not the model that was cheapest to *train*. The "wasted" training FLOPs are repaid many times over in cheaper inference across billions of served tokens.

| Regime | Optimizes | Tokens/param | Picks | Right when |
|---|---|---|---|---|
| Compute-optimal (Chinchilla) | Training FLOPs | ~20 | Larger model, less data | You train once, serve little (research, internal one-offs) |
| Inference-optimal (over-trained) | Lifetime cost | 100s–1000s | Smaller model, much more data | You serve at scale for a long time (most products) |

### A worked lifetime-cost example

Make it concrete. Suppose a 70B compute-optimal model and an over-trained 8B model both clear your quality bar (the 8B reaching parity by training on ~10× more data). Training the 8B harder costs, say, an extra \$1–2M in GPU-hours — a real but one-time number. Now price the inference. Per generated token, cost scales roughly with active parameters and memory bandwidth; in round numbers the 70B is ~8–9× more expensive to serve than the 8B at the same throughput target. If your product generates 50B tokens per month, and the 8B costs \$0.10 per million output tokens while the 70B costs \$0.90, that is \$5K/month versus \$45K/month — a \$40K monthly delta, or ~\$480K per year. The extra training spend on the 8B is repaid in roughly three to five months, and everything after that is pure savings, compounding for the life of the product. Flip the volume to 50M tokens per month (an internal tool) and the arithmetic inverts: the inference delta is \$40/month, the training premium never pays back, and the compute-optimal model — or just an API call — is correct. This is the entire argument in one calculation: **the crossover between compute-optimal and inference-optimal sizing is set by your served volume, so estimate that number before you choose a size.**

The senior rule of thumb: **if you will serve more than a trivial number of tokens, pick the smallest model that clears your quality bar, even if it was expensive to train — inference is the bill that never stops.** This is why the 7B–8B class is the workhorse of production: over-trained small models that are cheap to serve, easy to quantize, and fast enough to batch. Reach for the giant model only when quality genuinely requires it or when volume is low enough that training cost dominates.

#### Second-order optimization: distillation and speculative decoding move the frontier

Two techniques let you have a small serving model without giving up the big model's quality. **Distillation** trains a small student to mimic a large teacher's outputs; Gemma-2 and Llama-3.2's small models were distilled, and the [distillation deep-dive](/blog/machine-learning/large-language-model/distillation-in-llm) shows the recipes. **Speculative decoding** pairs a small draft model with the large target model, letting the draft propose tokens the target verifies in parallel — you get the big model's quality at a fraction of its latency, as covered in the [speculative decoding guide](/blog/machine-learning/large-language-model/speculative-decoding). Both are architecture decisions in disguise: they are how you decouple "the quality you ship" from "the FLOPs you pay per token." When the single-model size trade feels too tight, these are the escape hatches.

## Cross-cutting: tokenizer, vocabulary, and the multimodal frontier

Two architectural choices cut across every family and quietly shape cost and capability, and both are easy to inherit by accident rather than decide on purpose.

The **tokenizer and vocabulary** are part of the architecture, not a preprocessing afterthought. Vocabulary size sets the embedding and output-projection parameter count (a 256K-token vocabulary like Gemma's adds hundreds of millions of parameters versus a 32K vocabulary) and, more importantly, sets the *fertility* — how many tokens a given piece of text becomes. A tokenizer tuned for English wastes tokens on code, math, and non-Latin scripts, inflating both latency and cost because you pay per token. If you serve heavy code or multilingual traffic, a model whose tokenizer fragments your text into 1.5× more tokens is 1.5× more expensive to run regardless of its quality, and byte-fallback behavior decides whether rare characters degrade gracefully or explode into byte sequences. The [tokenizer design deep-dive](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm) works through fertility measurement and vocabulary trade-offs; the point for architecture selection is that two models with identical parameter counts can have very different *serving* costs purely because of how their tokenizers chew your specific text.

The **multimodal frontier** extends every lane rather than replacing it. A vision-language model is almost always a vision encoder (a ViT/SigLIP-style bidirectional encoder that turns an image into a sequence of patch embeddings) bolted onto a decoder via a projection layer — the decoder family choice is unchanged, you have just prepended visual tokens. Audio works the same way: Whisper is an audio encoder feeding a text decoder; speech LLMs prepend audio tokens from a [speech tokenizer](/blog/machine-learning/large-language-model/speech-tokenizer) to a decoder. So the decision framework composes: choose the text backbone by the rules above, then attach the modality encoder that produces tokens the backbone can consume. The failure mode here is treating "multimodal model" as a monolithic new thing to evaluate, when it is a vision/audio encoder plus a decoder you already know how to reason about — and the encoder, the projector, and the decoder can each be sized and frozen independently. When a multimodal model is too slow, the question is which of those three components is the bottleneck, and the answer is usually the decoder doing autoregressive generation, which sends you right back to the size and attention dials from §4 and §7.

## Putting it together: the decision path

We have walked seven axes. In practice you do not evaluate them all at once — you evaluate them *in order*, and most projects bottom out after two or three questions because the early forks eliminate whole regions of the design space. Here is the path as a flowchart.

![The decision path: output type, then transduction, then capacity-at-fixed-FLOPs, narrows the family before model selection](/imgs/blogs/choosing-right-llm-architecture-task-9.webp)

Read it top to bottom. Is the output a label, span, vector, or score rather than free text? Then you are in the **encoder** lane — bi-encoder to retrieve, cross-encoder to rerank, classification head to classify — and you are done before you ever consider a 70B decoder. Is it free text, but a fixed source mapping to a target (translate, transcribe, OCR)? **Encoder-decoder.** Otherwise it is open-ended generation, so you are in the **decoder** lane, and the remaining question is whether you need extra capacity at the same serving FLOPs (**decoder + MoE**) or want maximum simplicity and small-batch efficiency (**dense decoder**). Only *after* the family is fixed do you turn the downstream dials — attention variant for your KV budget, positional scheme for your context length, and size for your lifetime volume.

The same logic compresses into a decision function you can actually call:

```python
from dataclasses import dataclass

@dataclass
class Task:
    output: str            # "label" | "span" | "embedding" | "score" | "text"
    transduction: bool     # fixed source -> target (translate/ASR/OCR)?
    candidates: int        # how many docs to score per query (0 if not ranking)
    context_len: int       # max tokens in a request
    monthly_tokens: float  # served volume; sets inference-vs-train weighting
    vram_gb: int           # the GPU you actually have
    latency_ms: int        # p99 budget

def choose_architecture(t: Task) -> dict:
    rec = {}
    # 1. Information flow / output shape -> family
    if t.output in {"label", "span"}:
        rec["family"] = "encoder-only (DeBERTa-v3)"
    elif t.output == "embedding":
        rec["family"] = "bi-encoder (BGE/E5)"
    elif t.output == "score":
        rec["family"] = ("cross-encoder reranker" if 0 < t.candidates <= 200
                         else "bi-encoder retrieve + cross-encoder rerank (two-stage)")
    elif t.transduction:
        rec["family"] = "encoder-decoder (T5/Whisper/NLLB)"
    else:
        rec["family"] = "decoder-only"
        # 3. Capacity at fixed FLOPs, gated by memory
        rec["sparsity"] = ("MoE" if t.vram_gb >= 80 and t.monthly_tokens > 1e9
                           else "dense")
    # 4. Attention variant from KV budget
    if "decoder" in rec["family"] or "encoder-decoder" in rec["family"]:
        rec["attention"] = ("MLA/GQA-8 + fp8 KV" if t.context_len > 32_000
                            else "GQA")
        # 6. Positional scheme for context
        rec["positions"] = ("RoPE + YaRN (verify recall!)" if t.context_len > 32_000
                            else "RoPE")
        # 7. Size from lifetime volume
        rec["size"] = ("smallest model that clears quality (over-trained 7-8B)"
                       if t.monthly_tokens > 1e8 else "largest that fits budget")
    return rec

print(choose_architecture(Task("score", False, 1000, 512, 5e9, 24, 50)))   # a RAG ranking stage
    # -> {'family': 'bi-encoder retrieve + cross-encoder rerank (two-stage)'}
```

And because the architecture choice has to survive contact with a serving stack, here is the same set of decisions expressed as `vllm serve` flags — the attention/KV, context, and parallelism dials all show up on the command line:

```bash
## long-context dense decoder: GQA + fp8 KV cache, 128K context on 2 GPUs
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 131072 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92

## MoE decoder: experts sharded across GPUs (expert parallelism)
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --enable-expert-parallel \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8
```

If you are choosing the GPU itself, the [cost/throughput/latency guide for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) pairs with these flags to size the box. The point of the whole exercise: by the time you type that command, the architecture has already been decided by the task, and the flags are just the architecture made executable.

## Case studies from production

Frameworks are clean; production is not. Each of the following is a real failure pattern — symptom, the wrong first hypothesis, the actual root cause in the architecture, the fix, and the lesson. Names and exact numbers are composited, but every one is a mistake I have watched a competent team make.

### 1. The RAG system that retrieved garbage from a decoder's hidden states

A team built internal-docs RAG and bootstrapped retrieval by mean-pooling the last-layer hidden states of the 7B chat decoder they already served. Recall@10 was a dismal 31%; relevant documents simply were not surfacing. The first hypothesis was that the chunking was wrong, and they spent a week tuning chunk size and overlap — recall barely moved. The actual root cause was architectural: a causal decoder's hidden states are not embeddings. Each token's representation is built without seeing the future, the last-token vector is dominated by next-token-prediction signal, and nothing in language-model pretraining ever pulled paraphrases together in vector space. They swapped to `bge-base-en-v1.5`, a 109M-parameter contrastively-trained bi-encoder, and recall@10 jumped to 89% — with a model 60× smaller and an embedding step 20× faster. The lesson: embeddings come from contrastive training, not from whichever LLM you happen to have loaded. The *body and objective* make the embedding, not the parameter count.

### 2. The reranker that ate the latency budget

A search team got religion about cross-encoders after seeing the quality lift and wired a `bge-reranker-large` cross-encoder to score *every* candidate their bi-encoder returned — and they had set the retrieval depth to 1,000 for safety. p99 latency went to 6.2 seconds. The first hypothesis was that the GPU was undersized, and a procurement request for bigger cards was already drafted. The actual root cause was the architecture being run at the wrong stage: a cross-encoder is $O(N)$ full forward passes, and 1,000 of them per query is a non-starter regardless of GPU. The fix was the standard two-stage cascade — retrieve top-100 with the bi-encoder, rerank only the top-20 with the cross-encoder — which cut p99 to 240 ms with no measurable quality loss, because the bi-encoder almost never dropped a truly relevant doc out of the top 100. The lesson: a cross-encoder is a *reranker*, not a *retriever*; the moment you run one over more than a couple hundred candidates, you have used the right model in the wrong slot.

### 3. The MoE that "was 13B" and OOM'd on a 24 GB card

A startup picked Mixtral 8×7B because the docs said ~13B active parameters and they had a 24 GB GPU that comfortably served 13B dense models. The model would not even load. The first hypothesis was a quantization bug or a broken weights download. The actual root cause was the defining property of MoE: the router can dispatch any token to any expert, so *all* eight experts must be resident — ~47B parameters, ~94 GB in fp16, ~47 GB even in int8 — far beyond 24 GB. "13B active" describes the FLOPs per token, not the memory footprint. They had two honest options: serve a dense 13B model on the card they had, or move to a box with enough VRAM for the full expert bank. They chose a dense Qwen2.5-14B and shipped on the 24 GB card. The lesson: MoE saves FLOPs, not memory — provision VRAM for total parameters and compute for active parameters, and never the other way around.

### 4. The GQA migration that doubled throughput for free

A platform team was serving a 70B-class MHA model and capped at batch size 8 at 8K context — the KV cache was eating all the VRAM left after weights. The first instinct was to shard across more GPUs, which would have doubled the serving bill. Instead they moved to a GQA-8 checkpoint of the same model family. The KV cache shrank roughly 8× (64 KV heads down to 8), which freed enough memory to push batch size to 32 and, because decode is memory-bandwidth-bound and GQA re-reads 8× less KV per step, decode latency per token dropped too. Throughput went up ~2.5× on the *same* hardware with a quality delta inside the noise. They later stacked `--kv-cache-dtype fp8` on top for another ~2×. The lesson — the inverse of a failure, for once: at long context or high batch the KV cache, not the weights, is your binding constraint, and the attention variant is the cheapest lever you have. GQA and MLA are not micro-optimizations; they reshape your throughput curve.

### 5. The classifier that a 70B decoder did worse and 1000× more expensively

A product team needed to route support tickets into 12 categories and shipped it by prompting their 70B chat decoder with the category list and few-shot examples. It worked in the demo. In production it cost \$0.011 per ticket, ran at 900 ms p99, drifted whenever someone edited the prompt, and plateaued at 86% accuracy because the model occasionally invented categories or returned unparseable text. The first hypothesis was prompt engineering — more examples, better instructions — which bought two accuracy points and more latency. The actual fix was architectural: they labeled 4,000 tickets in a day and fine-tuned `deberta-v3-base` (184M params). It hit 94% accuracy, ran at 8 ms on a CPU, cost effectively nothing per ticket, and returned a calibrated probability they could threshold for human-in-the-loop routing. The lesson: a discriminative task wants a discriminative architecture. Prompting a giant decoder for classification is using a crane to hang a picture frame — it works, and it is the wrong tool by three orders of magnitude.

### 6. The "128K context" model that forgot everything past 40K

A team adopted a community fine-tune advertising 128K context for a long-document QA feature. Accuracy was fine on short docs and fell off a cliff on long ones; users reported the model "ignoring" content in big files. The first hypothesis was a chunking or prompt-assembly bug in the application. The actual root cause was the positional encoding: the base model was trained at 32K, the 128K window came from naive RoPE scaling with no long-context continued training, and recall collapsed past ~40K — classic RoPE extrapolation failure dressed up as a config value. A needle-in-a-haystack test, which nobody had run, showed retrieval accuracy dropping from 98% at 8K to 22% at 100K. The fix was twofold: switch to a model with proper YaRN extension *and* long-document continued pretraining (recall held to ~95% at 128K), and, more importantly, add retrieval so the model saw the relevant 4K instead of the whole 128K. The lesson: "supports 128K" is a buffer-size claim; "is accurate at 128K" is a training claim — always verify the second with a recall probe at your real length.

### 7. The translation model where smaller and dedicated beat bigger and general

A localization pipeline used a 13B general-purpose decoder for translation across 40 language pairs, prompted with "translate to X." Quality was acceptable for high-resource languages and poor for low-resource ones, latency was high, and the per-character cost made the business case shaky. The first hypothesis was that they needed an even larger decoder. The actual answer ran the other way: they switched the bulk of traffic to NLLB-200 (a 600M–3.3B encoder-decoder built specifically for 200-language translation), which beat the 13B decoder on low-resource pairs — its bidirectional encoder gets a clean read of the whole source and it was trained on exactly this transduction objective — at a fraction of the size and latency. They kept the big decoder only for the handful of pairs where its broader world knowledge helped with idiom. The lesson: for a fixed transduction task, a smaller architecture purpose-built for it often beats a larger general model. The encoder-decoder's division of labor is a real advantage when the job is "map this sequence to that sequence."

### 8. The pure-SSM model that could not copy a UUID

A team building a high-throughput log-analysis agent adopted a pure Mamba model for its constant-memory streaming over very long inputs — exactly the workload SSMs are built for. Throughput was excellent. Then accuracy cratered on tasks that required quoting a specific earlier value: extracting an exact request ID, copying a config string, answering "what was the error code on line 12,000." The first hypothesis was a tokenization issue with long hex strings. The actual root cause was structural: an SSM compresses the past into a fixed-size state and therefore cannot reliably perform exact, content-addressed recall the way attention's all-pairs lookup can — it is the known associative-recall weakness of state-space models. The fix was to move to a hybrid (Jamba-style) stack that interleaves a few attention layers into the Mamba majority; the attention layers restored exact recall while the Mamba layers kept compute linear, and exact-copy accuracy went from 41% to 96%. The lesson: SSMs trade exact recall for linear scaling — benchmark recall at your real context length before committing, and reach for a hybrid the moment your task needs to quote the past verbatim.

### 9. The over-trained 8B that retired a 70B and cut the inference bill 80%

A team was serving a 70B dense model for a high-volume assistant feature — tens of billions of tokens a month — and the inference bill was the single largest line in the infra budget. The first instinct was to optimize the serving stack harder: better batching, speculative decoding, quantization. Those helped at the margin. The bigger move was to re-examine the size decision: did the task actually need 70B of quality? They evaluated Llama-3-8B (over-trained on ~15T tokens, far past Chinchilla) on their real traffic and found it cleared the quality bar on ~92% of requests, routing only the hard residual to the large model. Serving the 8B for the bulk and the 70B for the tail cut the inference bill by ~80% with a quality delta users did not notice. The lesson: compute-optimal is not deployment-optimal. For a system that serves at scale, the smallest over-trained model that clears your quality bar is almost always the right serving model, and "make the model we already chose faster" is the wrong question when "did we need a model this big" is on the table.

### 10. The speculative-decoding pair that ran slower than the target alone

A latency-sensitive team adopted [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) to speed up their 70B model and paired it with a 7B draft from a *different* model family as the speculator. End-to-end latency got *worse*. The first hypothesis was a bug in the verification kernel. The actual root cause was an architecture mismatch in the pairing: speculative decoding only wins when the draft model's token distribution closely matches the target's, so that the target accepts most drafted tokens — the acceptance rate is everything. A draft from a different family, different tokenizer behavior, and different training data had a low acceptance rate, so the target rejected most drafts and the system paid for drafting *and* full verification with little parallel speedup, plus the drafts' tokenizer needed re-aligning. The fix was to use a small model from the *same* family and tokenizer (an official draft checkpoint, or a distilled small sibling), which pushed acceptance above 70% and delivered the expected ~2× decode speedup. The lesson: speculative decoding is an architecture decision about a *pair* of models, not a free switch — the draft and target must share a tokenizer and have aligned distributions, and "any small model" is not a valid speculator.

### 11. The moderation filter that had to be an encoder at the edge

A platform needed real-time content moderation on user messages — a binary safe/unsafe decision in under 20 ms, running on commodity CPU instances at the edge because round-tripping every message to a GPU service was too slow and too expensive at their volume. The team's first build called a hosted 70B safety model per message; p99 was 700 ms and the bill was untenable at billions of messages. The first hypothesis was caching common messages, which helped the head of the distribution and did nothing for the long tail of unique content. The actual answer was architectural and decided by the latency-and-hardware constraint before anything else: a discriminative task with a sub-20-ms CPU budget *must* be an encoder. They distilled the large model's judgments into a fine-tuned `deberta-v3-small` (44M params), quantized it to int8, and ran it on the CPU instances at 6 ms p99 with accuracy within a point of the teacher on their traffic, escalating only the model's low-confidence band to the big model. The lesson: the cost regime can decide the family outright. When the binding constraint is "sub-20 ms on a CPU," the generative-decoder lane is closed before you evaluate quality — you are in the encoder lane by physics, and the only question is how to distill enough quality into something that small.

## When to reach for each architecture, and when not to

The framework collapses to a handful of conditions. Reach for an architecture when the task's information-flow shape and cost regime line up with it — and refuse it when they do not.

**Reach for an encoder-only model when:**

- The output is a label, span, or token-level tag — classification, NER, extractive QA, content moderation.
- You have (or can cheaply label) supervised data and want a calibrated probability, not a parsed string.
- Latency or cost is tight enough that a sub-second, CPU- or fraction-of-GPU answer matters.
- You need an embedding for dense retrieval — use a contrastively trained bi-encoder, not a decoder's hidden states.

**Reach for an encoder-decoder when:**

- The task is fixed-source-to-target transduction: translation, speech-to-text, OCR, structured summarization.
- A dedicated, smaller model would beat a larger general decoder at this one job — frequently true at the small-model scale you ship to constrained hardware.
- The source benefits from a clean bidirectional read before generation begins.

**Reach for a dense decoder when:**

- The output is open-ended text — chat, code, reasoning, agentic tool-use.
- You serve at small batch or on memory-constrained hardware where MoE's resident expert bank does not fit.
- You want the simplest thing to fine-tune and deploy, and you do not need MoE's extra capacity.

**Reach for an MoE decoder when:**

- You want more knowledge capacity at a fixed per-token FLOP budget and you have the VRAM for the full expert bank.
- You serve at high concurrency in a datacenter where batching keeps experts well-utilized.
- Training a broad, general-purpose model where the capacity advantage actually pays off.

**Reach for SSM/hybrid mixers when:**

- Sequence length is the dominant cost — book-length documents, long audio, streaming agents — and quadratic attention is the binding constraint.
- You can tolerate, or have tested away with interleaved attention layers, the exact-recall weakness.

And the anti-patterns — **skip these architectures when:**

- *Skip a generative decoder* for a discriminative task. Prompting a 70B model to classify is slower, costlier, and less accurate than a fine-tuned 300M encoder. (Case Study 5.)
- *Skip a cross-encoder as a retriever.* It is $O(N)$ forward passes; run it only over a few hundred pre-retrieved candidates. (Case Study 2.)
- *Skip MoE on memory-constrained hardware.* The expert bank is resident regardless of which experts fire. (Case Study 3.)
- *Skip a giant model for high-volume serving* if a smaller over-trained model clears your quality bar — inference is the bill that never stops. (Case Study 9.)
- *Skip a "long-context" checkpoint* until you have verified recall at your real length with a needle test. (Case Study 6.)
- *Skip a pure SSM* for anything requiring exact recall of earlier tokens; go hybrid. (Case Study 8.)

The meta-lesson under all of it: the architecture is a *consequence* of the task and the cost regime, not a fashion choice and not a leaderboard ranking. When you start from the shape of the information flow, narrow by which cost regime dominates your bill, and only then pick a checkpoint, you will spend your GPU budget on the dials that actually move quality — and you will stop fine-tuning 70B decoders for jobs a 184M encoder does better, faster, and for free.

## Further reading

- [Modern LLM architectures: inside Qwen, Llama, Gemma, and DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) — how the production decoders actually wire GQA, MLA, RoPE, and MoE together.
- [Mixture-of-Experts LLMs](/blog/machine-learning/large-language-model/moe-llm-architecture-training-finetuning-case-studies) and [optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) — the sparse-model axis in full depth.
- [KV cache optimization and management](/blog/machine-learning/large-language-model/kv-cache) — the serving-side machinery behind the attention-variant choice.
- [Embedding models](/blog/machine-learning/large-language-model/embedding-models-training-finetuning-case-studies) and [reranker models](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies) — building the bi-encoder/cross-encoder cascade.
- [Estimating LLM capacity: self-host vs API](/blog/machine-learning/large-language-model/estimating-llm-capacity-and-self-host-vs-api) and [choosing a GPU for LLM serving](/blog/machine-learning/large-language-model/choosing-gpu-for-llm-serving-cost-throughput-latency) — the unit economics that decide the size and attention dials.
- Primary sources worth reading end to end: the GQA paper (Ainslie et al., 2023), DeepSeek-V2/V3 for MLA, the Chinchilla scaling-laws paper (Hoffmann et al., 2022), YaRN (Peng et al., 2023), and Mamba/Jamba for the sub-quadratic mixers.
