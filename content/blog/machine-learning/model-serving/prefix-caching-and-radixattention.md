---
title: "Prefix Caching and RadixAttention: Stop Re-Prefilling the Same Tokens"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "Reuse the KV cache across requests that share a prefix. Learn the hit-rate math, vLLM automatic prefix caching, SGLang RadixAttention, and how to measure the TTFT collapse on real hardware."
tags:
  [
    "model-serving",
    "inference",
    "prefix-caching",
    "radixattention",
    "kv-cache",
    "vllm",
    "sglang",
    "ttft",
    "pagedattention",
    "llm-serving",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/prefix-caching-and-radixattention-1.webp"
---

The page came in at 2:14 a.m.: p99 time-to-first-token on our customer-support assistant had crossed 800 ms, well past the 300 ms SLA, and the on-call dashboard showed GPU compute pinned at 100% while throughput sat flat. Nothing had shipped. Traffic had merely doubled during a product launch. When I pulled a request trace, the cause was embarrassingly simple: every single request carried the same 1,800-token system prompt — brand voice, safety rules, tool schemas, a dozen worked examples — and the server was faithfully re-running the full prefill over those 1,800 tokens on every call, only to throw the result away the instant the response finished. We were paying the most expensive part of LLM inference, the prefill, over and over for tokens that never changed.

The fix took one flag. Turning on prefix caching meant the server prefilled that 1,800-token block exactly once, kept its key/value tensors resident, and matched every subsequent request against it. The second request onward prefilled only the handful of tokens that were actually new — the user's question — and reused the cached prefix for free. TTFT for warm requests dropped from roughly 350 ms to under 30 ms on the same A100, GPU compute headroom reappeared, and the launch rode out without a capacity bump. The picture below is the whole idea in one frame: a full re-prefill on the left, a prefix-cache hit on the right.

![Two-panel comparison of a full re-prefill over 2048 tokens versus a prefix-cache hit that reuses 1920 tokens and prefills only 128, dropping TTFT from about 380 ms to about 32 ms](/imgs/blogs/prefix-caching-and-radixattention-1.webp)

This post is about that mechanic and the two production implementations of it: hash-based **automatic prefix caching (APC)** in vLLM, and the **radix tree** approach of SGLang's RadixAttention. Prefix caching sits squarely on the serving SLO triangle — latency, throughput, cost — and it is one of the rare techniques that pushes all three corners in the good direction at once, *when your workload has shared prefixes*. The whole post is organized around three questions this series always asks. The mechanics: how much do you actually save, in FLOPs and in milliseconds, and can we prove it? The practice: what are the real flags and APIs in vLLM and SGLang, and how do you measure the win? The decision: when is prefix caching free money, and when is it a waste of GPU memory that would serve you better as batch capacity? By the end you should be able to look at a workload, estimate its cache hit rate and TTFT reduction on the back of an envelope, turn the feature on correctly, and benchmark the result honestly.

We will assume you already know two ideas from earlier in this series. **KV cache**: during generation a transformer stores the key and value tensors for every past token so it never recomputes attention over them — see [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different). **PagedAttention**: vLLM stores that KV cache in fixed-size blocks (typically 16 tokens each) so memory is allocated on demand rather than as one giant contiguous slab — see [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention). Prefix caching is what you get when you notice that those blocks are content-addressable: two requests that begin with the same tokens should be able to point at the same physical blocks.

## 1. The prefill tax nobody budgets for

LLM inference has two phases with completely different cost profiles, and prefix caching only touches one of them, so it is worth being precise. **Prefill** processes the entire prompt in a single forward pass, computing the KV tensors for all prompt tokens in parallel. It is *compute-bound*: the GPU's tensor cores are the bottleneck, and the cost scales with the number of prompt tokens. **Decode** then generates output tokens one at a time, each step reading the whole KV cache to produce a single new token. It is *memory-bandwidth-bound*: the bottleneck is streaming the KV cache and weights out of HBM, and the cost scales with the number of output tokens and the sequence length.

Two metrics track these phases. **TTFT** (time-to-first-token) is dominated by prefill plus any queueing delay — it is how long the user waits before anything appears. **TPOT** (time-per-output-token) is dominated by decode — it is how fast the response streams once it starts. Prefix caching is a *prefill* optimization, so it moves TTFT, not TPOT. If your latency complaints are about slow streaming, prefix caching will not help; if they are about the pause before the first token, and your prompts share a long common prefix, prefix caching is often the single highest-leverage change you can make.

Why is prefill so expensive for long prompts? A forward pass over $n$ tokens does roughly ${2nP}$ FLOPs of position-wise work against a model with $P$ parameters (the projections and MLP blocks), plus the attention term. For a prompt with a 1,800-token system prefix, the server burns that compute every request even though 1,800 of those tokens produce byte-identical KV tensors every time — the transformer is deterministic in the prefill given the same input tokens. That repeated work is the prefill tax. Prefix caching is the observation that you can pay it once and amortize it across every request that shares the prefix.

The reason prefill is compute-bound while decode is memory-bound comes down to *arithmetic intensity* — the ratio of FLOPs performed to bytes moved from HBM. In prefill, all $n$ prompt tokens flow through the weights together as one big matrix multiply, so each byte of weight loaded from memory is reused across many tokens; the arithmetic intensity is high and the GPU's tensor cores saturate before its memory bandwidth does. In decode, a single token flows through the weights per step, so every weight byte is used exactly once before the next step reloads it; the arithmetic intensity is low, the tensor cores sit mostly idle, and HBM bandwidth is the wall. This is why an H100 with 3.35 TB/s of HBM and roughly 990 TFLOP/s of bf16 compute behaves like two different machines across the two phases. Prefix caching operates entirely on the compute-bound side: it removes prefill FLOPs. It does nothing for the bandwidth-bound decode, which is governed by separate techniques such as quantization and speculative decoding. Being clear about which wall you are hitting is half of serving optimization — measure whether your latency budget is spent before the first token or between tokens, and let that route you to the right tool.

A second consequence of prefill being one large matmul is that it *parallelizes across the prompt for free but not across requests without help*. Two separate requests each carrying the same 1,800-token prefix run two independent prefills unless something notices they are identical and deduplicates the work. Continuous batching alone will not deduplicate them — it batches distinct sequences to fill the GPU, but it still computes each sequence's KV independently. Prefix caching is precisely the missing deduplication layer: it recognizes that the KV for identical leading tokens is identical and shares it, converting $R$ requests' worth of prefix prefill into one.

The reason this matters *now*, more than it did two years ago, is that prompts have gotten enormous. A 2023 chatbot prompt was a one-line instruction. A 2026 production prompt is a small document: a multi-paragraph persona, a safety policy, a set of tool/function schemas, a handful of few-shot exemplars, and a retrieved-context block — routinely 1,000 to 4,000 tokens before the user has typed a word. The fixed, shared portion of the prompt has grown faster than the variable portion, and that ratio is exactly what determines how much prefix caching saves.

## 2. Where shared prefixes actually come from

Before optimizing anything, get honest about whether your workload even has shared prefixes, because that is the entire precondition. Prefix caching pays off in direct proportion to the fraction of the prompt that is shared and stable across requests. Five workload patterns produce shared prefixes, and they cover the large majority of production LLM traffic. The matrix below lays out how each one converts into a hit rate and a TTFT reduction.

![Matrix of five workload types against shared-prefix fraction, hit rate, prefill saved, and TTFT drop, showing few-shot and chat winning big while unique one-shot prompts gain nothing](/imgs/blogs/prefix-caching-and-radixattention-2.webp)

**Long system prompts.** The most common source. Every request to a given assistant carries the same instructions, persona, and rules. If the system prompt is 1,800 tokens and the user turn is 120 tokens, 94% of the prompt is shared. Hit rate approaches 100% after the first request, because the system prompt is byte-identical every time.

**Few-shot exemplars.** Classification, extraction, and formatting tasks prepend a fixed block of labeled examples — often 5 to 20 of them, easily 800 to 2,000 tokens. Only the final query changes. This is the highest-share pattern: the exemplar block can be 95%+ of the prompt, so few-shot workloads see the largest TTFT reductions, frequently 10× or more.

**Multi-turn chat history.** In a conversation, turn $k$ sends the entire history of turns ${1..k-1}$ plus the new user message. Turn $k$'s prompt is a strict superset of turn $k-1$'s prompt, so the shared prefix *grows* with the conversation. By turn 10, the reused prefix might be 90% of a 3,000-token prompt. This is where prefix caching quietly turns a chat app from unusable to snappy: without it, each turn re-prefills the whole growing history; with it, each turn prefills only the newest message.

**RAG contexts.** Retrieval-augmented generation is a partial case. The system prompt is shared, but the retrieved documents vary per query — unless you have a hot set of documents (a popular knowledge-base article, a pinned policy doc) that recur. RAG typically lands at a moderate share, 40–70%, because the retrieved block is a large chunk that is only sometimes reused. A useful trick: put the *stable* context (system prompt, static instructions) before the *variable* retrieved documents, so the stable part is a cacheable prefix even when the documents change. Prefix caching only reuses a *prefix* — the shared tokens must come first.

**Agent scaffolds.** Tool-using agents carry a large fixed preamble: tool/function definitions, the ReAct or plan-execute scaffold, output-format rules, and safety constraints. This preamble is stable across every step of every agent run, often 1,500–3,000 tokens, and it recurs across all users of the same agent. Agent workloads are a strong fit — 70–90% share is typical — and the win compounds because a single agent task issues many LLM calls, each re-using the same scaffold.

The one pattern where prefix caching does *nothing*: genuinely unique one-shot prompts. Ad-hoc summarization of a different document every time, or a search backend where every query is distinct with no shared preamble. Shared prefix under 5%, hit rate near zero, no savings — and, as we will see, a small cost. Knowing which bucket your traffic falls into is the whole decision.

**You can engineer prompts to be cache-friendly, and you should.** Because reuse is prefix-only, the ordering of a prompt determines how much of it is cacheable, and this is a design choice under your control. The rule is simple: put content in order of decreasing stability. The most stable content — the system prompt, tool schemas, safety rules — goes first, because it is shared by every request. Next comes content shared by *some* requests — few-shot exemplars for a given task, a hot RAG document. The variable, per-request content — the user's actual query, retrieved documents unique to this request, any timestamp or session data — goes last. A prompt structured this way maximizes the length of the shared prefix and therefore the hit rate. A prompt that interleaves stable and variable content (a system prompt with a per-request date stamped in the middle, say) fragments the cacheable region and can drop an otherwise 90%-shared prompt to near-zero reuse. This ordering discipline costs nothing and is often the difference between a workload that caches beautifully and one that does not, so treat prompt layout as a serving-performance decision, not just a prompt-engineering one.

## 3. The hit-rate math: how much do you actually save

Here is the mechanics block — the part you can put on a whiteboard and defend. The claim is that the fraction of prefill compute you avoid equals the fraction of the prompt that is a cached shared prefix. Let $s$ be the shared-prefix length in tokens and $n$ the total prompt length, so the unique suffix is $u = n - s$.

The position-wise work of prefill — the projections and MLP blocks, which are the bulk of prefill FLOPs at typical prompt lengths — is linear in token count: about ${2nP}$ FLOPs for $n$ tokens against $P$ parameters. A prefix-cache hit skips the forward pass for the $s$ prefix tokens entirely and prefills only the $u$ suffix tokens. So the saved fraction of that dominant term is exactly:

$$\text{prefill FLOPs saved} \approx \frac{s}{n} = \frac{s}{s + u}$$

The attention term is more subtle and works *in your favor*. Without a cache, attention over $n$ tokens costs on the order of $\tfrac{1}{2}n^2 d$ (each token attends to all prior tokens). With a hit, the prefix's own internal attention — on the order of $\tfrac{1}{2}s^2 d$ — is skipped entirely. The suffix tokens still attend back over the cached prefix keys and values (they read the cached K/V, they do not recompute it), so the suffix attention on the order of $(su + \tfrac{1}{2}u^2)d$ is still paid. The net effect is that the attention component is saved by *at least* $s/n$ and often more when the prefix is long relative to the suffix. So $s/n$ is a clean, conservative headline: **you save at least the shared-prefix fraction of prefill compute.**

Because TTFT for a long prompt on an unloaded server is dominated by prefill compute, TTFT scales with the prefill FLOPs. That gives the number you actually care about:

$$\text{TTFT}_{\text{hit}} \approx \text{TTFT}_{\text{full}} \cdot \left(1 - \frac{s}{n}\right) + t_{\text{lookup}}$$

where $t_{\text{lookup}}$ is the small, roughly constant cost of hashing the prompt blocks and probing the cache — sub-millisecond in practice. The figure below shows this split concretely: a 512-token prompt whose first 480 tokens hash to already-cached blocks is prefilled only over its 32 new tokens.

![Grid of eight blocks showing a 512-token prompt split into 30 cached prefix blocks reused from the system prompt and few-shot examples plus two new user-query blocks, saving 480 of 512 tokens or 93.75 percent of prefill](/imgs/blogs/prefix-caching-and-radixattention-3.webp)

Notice the non-linearity in the payoff. Because you multiply by $(1 - s/n)$, the TTFT reduction accelerates as the shared fraction rises. Going from 50% to 90% shared does not double the win — it takes you from a 2× speedup to a 10× speedup. The last stretch of shared prefix is worth far more than the first.

#### Worked example: a few-shot classifier

A sentiment classifier prompt has a 950-token block of labeled exemplars, followed by a 30-token review to classify. Total prompt $n = 980$, shared $s = 950$, unique $u = 30$.

- Prefill FLOPs saved: $s/n = 950/980 = 0.969$, or **96.9%**.
- On an A100 40GB running Llama-3-8B, a cold full prefill of 980 tokens takes roughly 410 ms of TTFT (queue empty). After the first request warms the exemplar block, a warm request prefills only 30 tokens: $\text{TTFT}_{\text{hit}} \approx 410 \times (1 - 0.969) + 0.5 \approx 13.2$ ms.
- That is a **31× TTFT reduction** for every request after the first, and it is available with a single configuration flag. The exemplar block is prefilled once for the life of the cache entry and reused by every classification call.

#### Worked example: a growing chat conversation

A support chat starts at a 400-token prompt (system prompt + first user turn) and grows by ~150 tokens per turn. At turn 8, the prompt is roughly $400 + 7 \times 150 = 1{,}450$ tokens, of which the newest user message is ~150 tokens and the rest — the system prompt plus all prior turns — is the shared prefix from turn 7's cached state. So $s = 1{,}300$, $n = 1{,}450$, $s/n = 0.897$.

- Prefill FLOPs saved at turn 8: **89.7%**.
- Without caching, turn 8 re-prefills all 1,450 tokens (~250 ms TTFT on A100). With caching, it prefills only the 150 new tokens: $\approx 250 \times 0.103 + 0.5 \approx 26$ ms.
- The savings *grow with the conversation*: turn 2 saves ~62%, turn 8 saves ~90%. Long conversations, which are exactly where re-prefilling hurts most, are exactly where caching helps most.

### Amortized cost across a request stream

The per-request formula above assumes a hit. In production, hit rate is not 1 — the filling request misses, evictions cause occasional re-misses, and some prefixes are cold. The number that matters for capacity planning is the *amortized* prefill cost across a stream of $R$ requests that share a prefix of length $s$, each with a unique suffix of length $u$.

Without caching, every request pays the full prefill, so total prefill work is $R \times (s + u)$ token-equivalents. With caching and a hit rate $h$ (the fraction of requests that hit the prefix), the shared prefix is prefilled only $(1-h)R + 1$ times instead of $R$ times — once for the filling request plus once per miss — while every request still prefills its own $u$ suffix tokens. Total prefill work becomes roughly $[(1-h)R + 1] \times s + R \times u$. The amortized savings fraction is:

$$\text{amortized saved} \approx \frac{h \cdot s}{s + u}$$

which is just the single-request $s/n$ result scaled by the hit rate $h$. Two lessons fall out. First, hit rate multiplies the win linearly, so a workload with a 94% shared prefix but only a 50% hit rate saves ~47%, not 94% — the cache being *full enough to hold the prefix* matters as much as the prefix being long. Second, the filling cost (the `+1`) is negligible over a long stream: one miss amortized over thousands of hits vanishes, which is why steady-state hit rates for stable system prompts sit near 100% and the amortized savings approach the full $s/n$. When you report a benchmark, always state the hit rate alongside the per-hit speedup, because the two multiply.

## 4. Automatic prefix caching in vLLM

vLLM's implementation is hash-based and, true to its name, automatic — you do not annotate anything, you just turn it on. The idea builds directly on PagedAttention's block structure. Recall that vLLM stores KV in fixed-size blocks of (by default) 16 tokens. Automatic prefix caching assigns each *filled* block a hash computed from the token IDs in that block **and the hash of all preceding blocks** — a prefix-chained hash. This chaining is the crucial detail: block 3's hash depends on the tokens in blocks 0, 1, 2, and 3, so a block is only reused when the *entire preceding token sequence* matches. Two prompts that differ in token 5 will produce different hashes for every block from block 0 onward, and correctly will not share anything.

The flow is a hash-and-lookup, with a branch on hit versus miss, as shown below.

![Branching graph where a new request is chunked into 16-token blocks, each block is prefix-chained hashed, a table lookup routes matched blocks to reuse of cached KV and unmatched blocks to prefill plus insertion, and both paths merge into decode](/imgs/blogs/prefix-caching-and-radixattention-4.webp)

When a request arrives, vLLM chunks its prompt into 16-token blocks, computes each block's prefix-chained hash, and probes a global hash table that maps block-hash to physical KV block. For each block that hits, the request's block table simply points at the existing physical block and the scheduler skips prefill for those tokens. For each block that misses, the tokens are prefilled normally and the resulting block is inserted into the hash table so the *next* request with the same prefix hits. Reuse is transparent to correctness because the KV tensors are a pure function of the token IDs — the same tokens always produce the same KV.

Turning it on is one argument in the Python API or one flag on the server. In the vLLM V1 engine (0.7+), prefix caching is on by default; on older builds you enable it explicitly.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,   # hash-based automatic prefix caching (APC)
    gpu_memory_utilization=0.90,  # more KV headroom => more cached blocks retained
    max_model_len=8192,
)

SYSTEM = open("system_prompt.txt").read()   # ~1,800 tokens, identical on every call
sp = SamplingParams(temperature=0.7, max_tokens=256)

# First request: the 1,800-token prefix misses, is prefilled, and its blocks are cached.
llm.generate([SYSTEM + "\n\nUser: What is the refund policy?"], sp)
# Second request: same prefix => hit. Only the new user tokens are prefilled.
llm.generate([SYSTEM + "\n\nUser: How do I reset my password?"], sp)
```

As an online server, the equivalent flag is `--enable-prefix-caching` (again, default-on in V1):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port 8000
```

From the client side, the key to getting hits is that the shared portion must be **byte-identical and come first**. With the OpenAI-compatible chat API, put the stable system message first and vary only the user message. vLLM tokenizes the messages and the leading blocks will match across calls:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
SYSTEM = {"role": "system", "content": open("system_prompt.txt").read()}  # stable, first

def ask(question: str):
    return client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[SYSTEM, {"role": "user", "content": question}],  # only this varies
        max_tokens=256,
    )

# The system message is identical every call, so its KV is prefilled once and reused.
for q in ["What is the refund policy?", "How do I reset my password?", "Do you ship to Canada?"]:
    print(ask(q).choices[0].message.content)
```

One subtlety that trips people up: a single differing token anywhere in the "shared" region destroys the hit from that point on. Injecting a per-request timestamp, a request ID, or a randomized greeting into the system prompt silently drops your hit rate to near zero, because the prefix-chained hash diverges at the first differing block. If you want to A/B two system prompts, keep each variant's shared portion contiguous and stable, and put anything per-request strictly *after* the shared block.

**Block size is a tuning knob with a real trade-off.** vLLM's default block is 16 tokens. A smaller block gives finer-grained sharing — a shared prefix that ends at token 500 is captured to the nearest 16 tokens rather than being rounded down harder — but it means more blocks, more hash-table entries, and slightly more per-block bookkeeping. A larger block reduces overhead but wastes the tail: if two prompts share the first 500 tokens but the block is 128 tokens, only the first $3 \times 128 = 384$ tokens are reused, because the fourth block (tokens 385–512) contains the first divergent token and cannot be shared. For most workloads the 16-token default is the right balance; only reach for a different block size if profiling shows either hash overhead (shrink) or excessive tail waste on near-miss prefixes (this is where SGLang's token-level granularity has a structural edge, discussed next).

**Chunked prefill interacts cleanly with caching.** vLLM can split a long prefill into chunks that interleave with ongoing decode steps, so a giant prompt does not stall the batch. Prefix caching composes with this: the cache lookup happens first, identifies which blocks are already resident, and only the *missing* blocks are fed into the chunked-prefill scheduler. The result is that a request with a 4,000-token cached prefix and a 100-token new suffix does not even enqueue a large prefill — it enqueues 100 tokens' worth, which the scheduler can slip between decode steps without a visible latency bump. This is a big part of why cached requests barely perturb the running batch.

**The hash is on token IDs, not text, and collisions are handled.** vLLM computes each block's hash from the integer token IDs of that block plus the parent block's hash, so it is agnostic to how the string was produced — what matters is the exact token sequence after tokenization. Because a hash collision would be a correctness bug (two different token sequences mapping to the same cached KV), vLLM verifies the actual tokens on a hash match rather than trusting the hash alone, so a collision degrades to a miss, never to wrong output. You do not have to reason about hash collisions in practice; the design fails safe.

**Watch the hit rate, do not assume it.** vLLM exports prefix-cache metrics — query and hit counters per engine — so you can compute the live hit rate and confirm the feature is actually earning its memory. A workload you *thought* had a stable prefix but that carries a hidden per-request token (a session ID, a formatted timestamp) will show a hit rate near zero, and the metric is how you catch it. We wire this into Prometheus in the monitoring section below.

## 5. RadixAttention in SGLang

SGLang attacks the same problem with a richer data structure. Instead of a flat hash table keyed on block hashes, it maintains a **radix tree** (a compressed prefix tree) over token sequences, where each edge is a run of tokens and each node's KV cache is the KV for the tokens along the path from the root. This is RadixAttention, introduced in the SGLang paper (Zheng et al., 2024). The tree structure buys something the flat table cannot express cleanly: reuse across *sibling branches* that share a common ancestor. The figure shows the shape.

![Radix tree three levels deep with a root empty-prefix node, a shared 40-token system-prompt node, a 200-token few-shot block referenced by 32 requests and an 800-token RAG document set referenced by 5, each branching into unique per-request leaf turns](/imgs/blogs/prefix-caching-and-radixattention-5.webp)

Walk the tree from the root. The system prompt is one node near the root, shared by every request. Below it, a few-shot exemplar block is a node referenced by every classification request that uses those exemplars; a different sub-tree holds a RAG document set referenced by the requests that retrieved it. The per-request suffixes — the actual user turns and queries — are the leaves. Each internal node stores its KV once, and every descendant path reuses it. When a new request arrives, SGLang finds the longest matching path from the root (the longest cached prefix), reuses that path's KV, and prefills only the divergent tail, extending the tree with a new branch.

Three properties make RadixAttention distinctive. First, **token-level granularity**: the tree can split an edge at any token boundary to share a partial run, whereas vLLM shares at the 16-token block boundary. This means SGLang can capture a shared prefix that ends mid-block. Second, **LRU eviction on the tree**: when KV memory fills, SGLang evicts the least-recently-used *leaf* nodes first, which are the coldest, least-shared prefixes — internal nodes shared by many live requests are naturally protected because they are referenced (and thus recently used) more often. Third, **cache-aware scheduling**: SGLang can reorder the request queue to batch together requests that share a prefix, maximizing cache hits and reducing redundant prefill. This is the piece a flat hash table cannot do on its own.

RadixAttention is on by default in SGLang. You enable cache-aware scheduling with the longest-prefix-match policy, and you can disable the radix cache entirely for debugging:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --mem-fraction-static 0.85 \      # fraction of GPU memory for the static KV pool
    --schedule-policy lpm \           # longest-prefix-match: cache-aware scheduling
    --port 30000
# --disable-radix-cache would turn RadixAttention off (useful only to A/B the effect)
```

SGLang's frontend language makes the shared structure explicit, which helps the runtime keep the radix tree tidy. The system message and few-shot exemplars form the shared path; only the final generation varies:

```python
import sglang as sgl

@sgl.function
def few_shot_classify(s, review: str):
    s += sgl.system("You are a sentiment classifier. Reply with POSITIVE or NEGATIVE.")
    # These exemplars are identical every call -> one shared node in the radix tree.
    s += sgl.user("Review: 'Best purchase this year.'\nLabel:")
    s += sgl.assistant("POSITIVE")
    s += sgl.user("Review: 'Broke after one day.'\nLabel:")
    s += sgl.assistant("NEGATIVE")
    # Only this suffix diverges per request -> a new leaf branch.
    s += sgl.user(f"Review: '{review}'\nLabel:")
    s += sgl.assistant(sgl.gen("label", max_tokens=1))

# Every call reuses the shared system+exemplar path; only the review tokens are prefilled.
states = few_shot_classify.run_batch(
    [{"review": r} for r in ["Loved it.", "Terrible support.", "Works as described."]]
)
```

RadixAttention also shines on *tree-structured* generation — beam search, tree-of-thought, self-consistency sampling — where many candidate continuations branch off a common prefix. The shared branches map naturally onto shared tree nodes, so the common context is prefilled once and every branch reuses it. This is a case where the radix structure earns its complexity over a flat table.

It is worth being concrete about the three radix-tree operations, because they are what the runtime does on every request. **Match** walks from the root, following edges whose tokens match the incoming prompt, until the longest common prefix is found; the KV along that matched path is reused and the ref-counts on those nodes increment. **Insert** takes the divergent tail — the tokens after the matched prefix — and extends the tree with a new branch, splitting an existing edge if the divergence happens mid-edge (this edge-splitting is what gives token-level, rather than block-level, granularity). **Evict** runs when the KV pool is full: it walks the leaves, finds those with ref-count zero, and frees the least-recently-used ones, collapsing edges as needed. All three are efficient — match and insert are linear in the prompt length, and eviction touches only the cold frontier — so the tree overhead is negligible next to the prefill it saves.

The **longest-prefix-match (LPM) scheduling** policy is the piece that most distinguishes SGLang. A first-come-first-served scheduler processes requests in arrival order, which scatters requests that share a prefix across time; by the time the third request in a shared group arrives, an unrelated request may have evicted the prefix. LPM instead sorts the waiting queue so that requests sharing the longest prefix are batched adjacently, which does two things: it maximizes the chance that a shared prefix is resident when the requests that need it run, and it lets the runtime prefill the shared prefix once for the whole cluster rather than racing to fill and evict it. On a workload with many interleaved distinct prefixes — a multi-tenant endpoint serving several different system prompts, say — LPM scheduling can lift the effective hit rate substantially over plain FCFS with the same cache. The cost is a small scheduling fairness trade-off: aggressively reordering for cache locality can delay a request whose prefix is cold, so SGLang bounds the reordering to avoid starving unlucky requests.

The token-level-versus-block-level granularity difference has a concrete payoff on *near-miss* prefixes. Consider two prompts that share the first 500 tokens and then diverge. vLLM's 16-token blocks capture $\lfloor 500 / 16 \rfloor \times 16 = 496$ shared tokens — it rounds down to the last full shared block. SGLang's radix tree splits the edge exactly at token 500, capturing all 500. The difference is tiny for a long prefix (496 versus 500 is a rounding error) but grows in relative terms when prefixes are short or when the divergence point matters for correctness of reuse. In practice the block-boundary rounding is rarely the deciding factor; the more common reason to reach for SGLang is its cache-aware scheduling and native tree-structured decoding, not the last few tokens of a prefix.

## 6. How prefix caching composes with PagedAttention

Prefix caching is not a replacement for PagedAttention — it rides on top of it. The physical KV blocks are still PagedAttention blocks; prefix caching adds a hashing-and-lookup layer above them and makes the blocks **reference-counted** so that many requests can point at the same physical block. The layered view below traces the path from raw tokens down to the shared physical blocks and the scheduler.

![Layered stack from incoming tokens down through 16-token block chunking, per-block prefix-chained hashing, the block hash table mapping to physical blocks, reference-counted PagedAttention blocks shared across requests, and a cache-aware scheduler routing to maximum overlap](/imgs/blogs/prefix-caching-and-radixattention-6.webp)

The reference count is what makes sharing safe. When request B hits request A's cached prefix blocks, those blocks' ref-count increments; B's block table points at the same physical memory A is using. Neither request can corrupt the other because the shared blocks are read-only KV — a prefix block, once filled, never changes, since its contents are a deterministic function of its (fixed) tokens. When a request finishes, its blocks' ref-counts decrement; a block is only eligible for eviction when its ref-count hits zero, meaning no live request needs it. This is standard copy-on-write discipline applied to KV memory: shared prefix blocks are shared read-only, and the first token that *diverges* starts a fresh, private block.

The copy-on-write discipline deserves one more level of detail, because it is what lets prefix sharing coexist with the fact that generation eventually diverges. Each request has a **block table** — an array mapping its logical block indices to physical block IDs. When request B hits request A's prefix, B's block table entries for the shared logical blocks simply store A's physical block IDs, and those physical blocks' ref-counts go up. As long as both requests only *read* those blocks (which is all a prefix block is ever used for), nothing needs to be copied. The instant a write would occur to a shared block — which for prefix caching happens at the block containing the first divergent token — the runtime allocates a fresh physical block, copies the still-shared portion if the block is only partially shared, and points the writing request's block table at the new private block. Downstream blocks are naturally private because they never existed in the other request. This is the same copy-on-write mechanism operating systems use for `fork()`, applied to KV memory: share aggressively while read-only, split lazily on the first write.

A worked block-table picture makes it concrete. Suppose the block size is 16 and request A prefilled a 48-token system prompt into physical blocks `[7, 12, 3]` (three blocks). Request B arrives with the same 48-token prefix plus a 20-token user turn. B's block table becomes `[7, 12, 3, 21, 22]`: the first three entries are shared with A (ref-count now 2 on blocks 7, 12, 3), and blocks 21 and 22 are freshly allocated and private to B for its 20 new tokens. If A finishes first, blocks 7, 12, 3 drop to ref-count 1 and stay resident because B still needs them; only when B also finishes do they hit zero and become eligible for eviction — but even then they linger in the cache so the *next* request with that system prompt hits them. This is the difference between prefix caching and the plain block sharing that PagedAttention already does for parallel sampling within one request: prefix caching keeps the blocks addressable *across* requests and *after* the originating request is gone.

This composition has a direct memory consequence you must plan for. A cached prefix occupies real GPU KV memory for as long as it is retained. The KV footprint per token is:

$$\text{KV bytes/token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times \text{bytes}$$

where $L$ is the number of layers, $H_{kv}$ the number of key/value heads (with grouped-query attention this is far smaller than the query-head count), $d_{\text{head}}$ the head dimension, and the leading 2 accounts for storing both keys and values. For Llama-3-8B ($L=32$, $H_{kv}=8$, $d_{\text{head}}=128$, fp16): $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, i.e. **128 KB per token**. A retained 1,800-token system prefix therefore holds $1{,}800 \times 128\text{ KB} \approx 225$ MB of GPU KV memory hostage. That memory is not available for active-request KV. The trade-off is explicit: memory spent retaining hot prefixes is memory not spent on batch capacity. On a GPU with tens of gigabytes of KV pool, one shared prefix is a rounding error and the reuse is pure win; but if you retain thousands of *distinct* cold prefixes, you can starve the running batch and *reduce* throughput. This is why eviction policy matters, and it is the subject of the next section.

## 7. The cache lifecycle: fill, hit, and evict

A prefix-cache entry has a life: it is born on a miss (paid for by the request that first prefills it), reused for free on every subsequent hit, and eventually evicted under memory pressure. Understanding this lifecycle is how you reason about steady-state hit rate and memory. The timeline below traces one hot prefix through fill, hits, and an LRU eviction when the KV pool saturates.

![Timeline of six events along a request stream: R1 misses and fills the cache, R2 and R3 hit for free, the KV pool reaches 90 percent, LRU evicts the coldest prefix, and a later R4 gets a partial hit reusing 60 percent](/imgs/blogs/prefix-caching-and-radixattention-7.webp)

**Fill (miss).** The first request carrying a given prefix pays the full prefill for it and inserts the blocks into the cache. This request sees no benefit — someone has to pay. In a workload with a stable system prompt, this happens once, ever (until eviction), which is why steady-state hit rate approaches 100% for system-prompt sharing.

**Hit.** Every subsequent request that matches the prefix reuses its blocks and skips their prefill. Hits are the whole point; the ref-count on the shared blocks rises with the number of concurrent live requests using them, and each hit refreshes the entry's recency for LRU purposes.

**Evict.** When the KV pool approaches full, the cache must free space for new requests. Both vLLM and SGLang use LRU: the least-recently-used blocks (vLLM) or leaf nodes (SGLang) with ref-count zero are freed first. Because hot prefixes are touched constantly, they stay warm and survive; cold, one-off prefixes are the first to go. Critically, a block with a non-zero ref-count — one a live request is currently using — is never evicted, so eviction can never corrupt an in-flight request. The worst case eviction can cause is a future *miss*, not incorrect output.

**Partial hit.** After an eviction, a returning request may find that only part of its prefix survived — say the system prompt is still cached but the specific conversation history was evicted. It reuses what remains (a partial hit, reusing perhaps 60% of the prefix) and re-prefills the rest. Partial hits are common and still valuable; the reuse fraction, not a binary hit/miss, is what feeds the $s/n$ savings formula.

This lifecycle also explains a counterintuitive tuning knob: **more traffic to a shared prefix raises its hit rate**, because the prefix stays warm and never gets evicted. Prefix caching gets *better* under load for shared workloads, which is the opposite of most systems. The failure mode is the reverse — many distinct cold prefixes churning through the cache, each filled once and evicted before it is reused, all cost and no benefit.

### Correctness: when is a hit actually safe

The single most important correctness fact about prefix caching is this: **sampling parameters do not affect prefix reuse, but the token sequence must match exactly.** The KV cache stores the deterministic result of the transformer's forward pass over the prompt tokens — keys and values. That computation does not depend on `temperature`, `top_p`, `top_k`, the random seed, or `max_tokens`, because those only influence how the *next* token is sampled from the logits, which happens *after* the cached KV is used. So two requests with the same prompt tokens but different sampling settings can and should share the same cached prefix KV; the divergence happens only at the sampling step. This is why you never have to worry that caching will make a temperature-0.9 request behave like a temperature-0 one — it won't.

What *must* match is the token IDs, exactly and in order, from the start. Consequences worth internalizing:

- **Tokenization must be identical.** The same string tokenizes to the same IDs under the same tokenizer, so this is usually automatic — but a different chat template, a stray whitespace difference, or a BOS-token toggle changes the token IDs and breaks the match.
- **The shared portion must be a genuine prefix.** Reuse is prefix-only. Tokens shared in the *middle* of two prompts (same document, different preambles) do not share, because the prefix-chained hash diverges at the first differing token. Order your prompt stable-part-first to make sharing possible.
- **Different LoRA adapters do not share prefixes.** A LoRA adapter changes the weights, so the KV for the same tokens differs; vLLM and SGLang key the cache on the adapter identity too. See [multi-LoRA and adapter serving](/blog/machine-learning/model-serving/multi-lora-and-adapter-serving) for how adapters are routed.
- **Quantization and model version are part of the key.** The KV depends on the exact weights; you cannot reuse a prefix cached under an fp16 model for an fp8 one.

Because correctness reduces to "same tokens, same model, same adapter → same KV," prefix caching is one of the few large optimizations that is provably lossless. There is no accuracy trade-off, no approximation, no quality regression — a warm request produces bit-identical logits to a cold one (given the same sampling seed). That is what makes it such an easy call when the workload fits.

One distinction prevents a common confusion: **prefix caching is not semantic caching.** Semantic caching (as in GPTCache and similar layers) embeds the query, finds a semantically *similar* past query, and returns its cached *response* — skipping the model entirely. It is lossy by construction: two questions that mean roughly the same thing get the same answer, which can be wrong, so it needs a similarity threshold and careful validation. Prefix caching, by contrast, reuses the KV of an *exactly matching token prefix* and still runs the model over the suffix, so the output is exactly what a cold run would produce. They operate at different layers and compose cleanly: a semantic cache in front of the server can short-circuit repeat questions entirely, while prefix caching inside the server accelerates everything that reaches the model. If someone proposes "just cache the answers," be clear about which one they mean — prefix caching is lossless and always safe to enable; semantic caching is a product decision with correctness implications. This series treats only the lossless kind.

A related edge case: **multi-modal prompts.** For vision-language models, image tokens are part of the prompt and participate in the prefix hash like any other tokens — a shared image (a fixed diagram in every request, say) can be cached, but an image that differs per request breaks the prefix from its position onward, exactly like a differing text token. The stable-content-first ordering rule applies to modalities too: put shared images and instructions before per-request images and text.

## 8. Measuring it: benchmarks on real hardware

Never trust a serving optimization you have not measured on your own model and workload. Two tools make prefix caching easy to quantify: a back-of-envelope hit-rate calculator to set expectations, and a TTFT benchmark harness to verify them.

The calculator turns the $s/n$ math into a function you can call while sizing a workload:

```python
def prefill_flops_saved(shared_prefix_len: int, unique_suffix_len: int) -> float:
    """Fraction of prefill compute avoided by a prefix-cache hit.
    Position-wise prefill work is ~linear in token count, so the saved
    fraction is (shared prefix) / (total prompt); attention savings are >= this."""
    total = shared_prefix_len + unique_suffix_len
    return shared_prefix_len / total

def effective_ttft(base_ttft_ms: float, saved_fraction: float,
                   lookup_overhead_ms: float = 0.5) -> float:
    """TTFT after a hit: prefill only the unique suffix, plus a small hash-lookup cost.
    Valid when prefill dominates TTFT (long prompt, near-empty queue)."""
    return base_ttft_ms * (1.0 - saved_fraction) + lookup_overhead_ms

# Few-shot classifier: 950-token exemplar block, 30-token review, 410 ms cold TTFT.
saved = prefill_flops_saved(950, 30)                     # 0.969
print(f"prefill saved: {saved:.1%}")                     # 96.9%
print(f"warm TTFT:     {effective_ttft(410, saved):.1f} ms")  # ~13.2 ms
```

The harness measures the real thing against a running vLLM server. Stream the response, timestamp the first token, warm the cache with one call, then measure the rest — which all share the prefix:

```python
import time, statistics
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
SYSTEM = open("system_prompt.txt").read()   # ~1,800 shared tokens

def measure_ttft(question: str) -> float:
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        max_tokens=64, stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:       # first token has arrived
            return (time.perf_counter() - start) * 1000.0
    return float("nan")

questions = [f"Request {i}: summarize the applicable policy in one line." for i in range(50)]
measure_ttft(questions[0])                       # warm the shared prefix (this one is a miss)
ttfts = [measure_ttft(q) for q in questions[1:]]  # all hits on the 1,800-token prefix
ttfts.sort()
print(f"median warm TTFT: {statistics.median(ttfts):.1f} ms")
print(f"p99 warm TTFT:    {ttfts[int(0.99 * len(ttfts))]:.1f} ms")
```

Run this once with `--enable-prefix-caching` and once with `--no-enable-prefix-caching` (or restart with the flag off), and the difference is your real, measured win. The table below reports representative results for Llama-3-8B on a shared-prefix workload — a 1,800-token system prefix with a ~120-token user turn — on two datacenter GPUs. These figures are derived from the FLOP-savings model above and are consistent with published vLLM and SGLang benchmarks; treat them as order-of-magnitude for planning, not a guarantee for your exact model and prompt.

| Hardware | Prefix cache | Median TTFT | p99 TTFT | Prefill throughput | Effective QPS at 300 ms SLA |
|---|---|---|---|---|---|
| A100 40GB | off | ~350 ms | ~780 ms (under load) | 1× (re-prefill every request) | ~10 QPS |
| A100 40GB | on (warm) | ~28 ms | ~55 ms | ~12× on the shared portion | ~45 QPS |
| H100 80GB | off | ~180 ms | ~420 ms (under load) | 1× | ~22 QPS |
| H100 80GB | on (warm) | ~15 ms | ~34 ms | ~12× on the shared portion | ~90 QPS |

The pattern to read out of the table: prefix caching does not just lower median latency, it *tightens the tail*. Under load without caching, TTFT p99 blows out because every request contends for compute to re-prefill the same tokens; with caching, warm requests barely touch the tensor cores, so the queue drains fast and p99 stays close to median. The effective-QPS column is the business case — the same GPU serves 3–4× the SLA-compliant traffic once it stops re-prefilling the shared prefix.

A word on benchmarking methodology, because prefix caching is easy to measure dishonestly. The most common mistake is to report the *warm* TTFT without disclosing that the first request paid the fill cost — a benchmark that fires the identical prompt a thousand times will show a spectacular speedup that no real workload sees, because real traffic has cold prefixes, evictions, and a hit rate below 1. Always: (1) warm the cache explicitly and measure separately from the cold path, so the fill cost is visible; (2) report the hit rate alongside the speedup, since the amortized win is the per-hit speedup times the hit rate; (3) drive the benchmark at your real concurrency, because caching's tail-latency benefit only shows up under load, when re-prefilling would otherwise cause queueing; and (4) compare against the same engine with caching disabled, not against a different engine, so you isolate the caching effect from everything else. A benchmark that violates these will either oversell caching (all-identical prompts, no cold path) or undersell it (single-request, no load, so no queueing to relieve). The harness above is structured to do it right: it warms with the first call and measures the rest as hits.

#### Worked example: an agent workload with many calls per task

An agent carries a 2,400-token scaffold (tool schemas, ReAct instructions, output-format rules) and runs an average of 6 LLM calls per user task, each call appending a short observation and action of ~80 tokens. The scaffold is identical across all 6 calls and across all tasks using this agent.

- Within one task, calls 2–6 hit the scaffold plus the growing trace. Call 3's prompt is ~2,560 tokens, of which ~2,480 is the cached prefix (scaffold + prior steps): $s/n \approx 0.969$.
- Across tasks, the 2,400-token scaffold has a near-100% steady-state hit rate, so it is prefilled essentially once for the whole fleet, not once per task.
- Without caching, a 6-call task prefills roughly $6 \times 2{,}400 = 14{,}400$ scaffold tokens (plus suffixes). With caching, it prefills the scaffold once (~2,400 tokens) and only the per-call suffixes. That is a ~5–6× reduction in prefill work *per task*, which is why agent platforms — with their large fixed scaffolds and many calls — are among the biggest beneficiaries of prefix caching. The win compounds with the call count: a 12-call task saves proportionally more.

#### Worked example: cost per million requests

Suppose the support assistant handles 5 million requests/month, each with the 1,800-token system prefix and a ~120-token user turn, on A100 40GB instances at roughly \$1.10/GPU-hour. Without caching, prefill dominates and each GPU sustains ~10 SLA-compliant QPS, so you need about $5{,}000{,}000 / (10 \times 2{,}592{,}000\ \text{sec/month}) \approx 0.19$ → round up to 1 GPU on average but ~3 GPUs at peak for headroom. With caching at ~45 QPS/GPU, peak headroom drops to ~1 GPU. Even at a conservative 2-GPU-to-1-GPU reduction, that is roughly \$1.10 × 24 × 30 ≈ \$790/GPU-month saved, plus the latency improvement that keeps the product usable. The memory cost — one retained 225 MB prefix — is negligible against a 40 GB card. This is the rare optimization where the latency win and the cost win point the same way.

## 9. Monitoring and tuning in production

A prefix cache you cannot observe is a prefix cache you cannot trust. The single most important production metric is the **cache hit rate** — the fraction of prefix-cache queries (or, more precisely, of prompt blocks) that were served from the cache. A high hit rate confirms the feature is earning its memory; a hit rate near zero means either your workload has no shared prefixes or, more often, a hidden per-request token is poisoning the prefix. vLLM exposes prefix-cache counters on its `/metrics` endpoint, and scraping them into Prometheus turns "I think caching is helping" into a number on a dashboard.

```yaml
# prometheus.yml — scrape the vLLM server's OpenMetrics endpoint
scrape_configs:
  - job_name: "vllm"
    metrics_path: /metrics
    static_configs:
      - targets: ["vllm-server:8000"]
```

With the counters flowing, the hit rate and its trend are one PromQL query each. vLLM publishes cumulative query and hit counters; the ratio of their rates over a window is the live hit rate:

```promql
# Live prefix-cache hit rate over the last 5 minutes (0.0 - 1.0)
sum(rate(vllm:prefix_cache_hits_total[5m]))
  /
sum(rate(vllm:prefix_cache_queries_total[5m]))

# Alert-worthy: hit rate collapsed for a workload we expect to share prefixes
sum(rate(vllm:prefix_cache_hits_total[10m]))
  /
sum(rate(vllm:prefix_cache_queries_total[10m])) < 0.30
```

Wire the second expression into an alert rule for any endpoint whose workload *should* be sharing a prefix — a system-prompt assistant, a few-shot classifier — and it will page you the moment someone injects a timestamp into the system prompt or changes a chat template in a way that breaks tokenization. That alert has caught more silent regressions than any code review, because the breakage is invisible in the output: the model still answers correctly, it just costs 10× more to do it.

The tuning knobs are few and each maps to a lever from the mechanics above. This table is the whole surface for both engines.

| Knob | Engine | Effect | When to change it |
|---|---|---|---|
| `enable_prefix_caching` | vLLM | Turns APC on (default-on in V1) | Leave on unless every prompt is unique |
| `gpu_memory_utilization` | vLLM | Larger KV pool → more retained cached blocks | Raise toward 0.90–0.95 if hit rate is capped by eviction |
| `block_size` | vLLM | Sharing/overhead granularity (default 16) | Rarely; shrink for finer sharing, grow to cut overhead |
| `max_num_seqs` | vLLM | Batch width; competes with cache for KV | Lower if cold prefixes are starving the batch |
| `--mem-fraction-static` | SGLang | Fraction of GPU memory for the KV pool | Raise to retain more of the radix tree |
| `--schedule-policy lpm` | SGLang | Cache-aware longest-prefix-match scheduling | Turn on for interleaved distinct prefixes |

The dominant knob is `gpu_memory_utilization` (vLLM) / `--mem-fraction-static` (SGLang): more KV pool means hotter prefixes survive longer and the steady-state hit rate rises, at the cost of batch capacity. The correct setting is workload-dependent and best found empirically — raise it while watching the hit rate and the p99 of active requests, and stop when either the hit rate plateaus or the running batch starts to feel memory pressure.

### Common pitfalls that silently kill your hit rate

- **A per-request token in the "shared" region.** A timestamp, request ID, user name, or A/B flag placed *inside* the system prompt makes every prefix unique from that token onward. Fix: move all per-request content strictly after the shared block, or template it out of the system prompt entirely.
- **Chat-template or tokenizer drift.** Upgrading the tokenizer, toggling the BOS token, or changing the chat template alters the token IDs, so cached entries from before the change never match after it. Expect a hit-rate cliff after any such change and pre-warm the new prefix.
- **Non-prefix sharing.** Two prompts that share text in the *middle* (same document, different preambles) share nothing, because reuse is prefix-only and the prefix-chained hash diverges at the first differing token. Reorder so the stable content leads.
- **Cache thrashing under many distinct cold prefixes.** A multi-tenant endpoint with hundreds of rarely-repeated system prompts can fill the cache with entries that are evicted before reuse — all memory cost, no hit benefit. Watch for a low hit rate combined with high eviction rate; if you see it, either shrink the cache (spend the memory on batch capacity) or route tenants so each server sees a smaller, hotter set of prefixes.
- **Assuming decode got faster.** Prefix caching does not touch TPOT. If a stakeholder reports "generation is still slow," they mean streaming speed, which is a decode/bandwidth problem, not a prefill/cache one.

## 10. vLLM APC versus SGLang RadixAttention

Both systems implement the same mechanic — reuse shared-prefix KV, evict with LRU, key on exact token match — so for a straightforward shared-system-prompt workload they perform comparably, and the right choice is usually whichever framework you are already running. The differences matter at the margins, and the matrix below lays them out.

![Matrix comparing vLLM automatic prefix caching and SGLang RadixAttention across data structure, granularity, eviction policy, cache-aware scheduling, and enablement, with an edge column noting where each differs](/imgs/blogs/prefix-caching-and-radixattention-8.webp)

| Dimension | vLLM APC | SGLang RadixAttention | Practical edge |
|---|---|---|---|
| Data structure | Flat hash table (block-hash → physical block) | Radix tree over token sequences | Tree reuses across sibling branches; flat table is simpler |
| Granularity | 16-token blocks | Token-level (variable-length edges) | SGLang captures prefixes that end mid-block |
| Eviction | LRU over free blocks | LRU over free tree leaves | Both O(1); protect shared nodes automatically |
| Cache-aware scheduling | No (FCFS + opportunistic hash reuse) | Yes (longest-prefix-match reordering) | SGLang raises hit rate on interleaved prefixes |
| Tree-structured decoding | Limited | Native (beam/ToT branches share nodes) | SGLang for heavy branching workloads |
| Enablement | `enable_prefix_caching=True` (default-on in V1) | On by default; `--schedule-policy lpm` for cache-aware sched | Both a one-flag change |

The honest summary: for the 80% case — a stable system prompt or growing chat history served first-come-first-served — vLLM APC and SGLang RadixAttention give you essentially the same win, and you should not switch frameworks for it. SGLang's radix tree earns its extra machinery in two situations: heavy tree-structured generation (beam search, tree-of-thought, self-consistency), where branches naturally share ancestor nodes; and workloads where many *different* prefixes interleave in the request stream, where cache-aware scheduling reorders requests to cluster hits that FCFS would scatter. If your workload is neither, the flat hash table is doing the same job with less complexity.

## 11. Beyond a single GPU: offloading and multi-tier caches

Everything so far assumes the prefix cache lives in GPU HBM, which is fast but small and shared with the running batch. The moment your working set of hot prefixes exceeds the KV pool — many tenants, many distinct-but-recurring system prompts, long conversation histories that outlive a single request's residency — you start evicting prefixes you would rather keep, and the hit rate stalls below what the workload could support. This is where prefix caching grows a memory hierarchy.

**CPU and disk offload.** The idea is to keep the hottest prefix KV in GPU HBM, spill colder entries to CPU DRAM, and the coldest to NVMe, fetching back on a hit. The economics are favorable because moving KV is far cheaper than recomputing it. A 2,000-token prefix on Llama-3-8B is $2{,}000 \times 128\text{ KB} = 256$ MB of KV. Over a PCIe Gen4 x16 link at roughly 32 GB/s, moving it into HBM takes about 8 ms; recomputing that prefill from scratch on an A100 costs on the order of 200 ms. As long as the transfer is much cheaper than the recompute — which it is by an order of magnitude — a CPU-cache hit beats a cold miss. Open-source layers such as LMCache implement exactly this, offloading KV to CPU and disk and sharing it across vLLM instances; vLLM's KV-connector interface is the integration point. The trade-off shifts from "hit or recompute" to "GPU hit, CPU hit, or recompute," and each tier extends the effective hit rate at the cost of a bounded transfer latency.

**Cross-node prefix-aware routing.** In a fleet of servers, a naive load balancer spreads requests round-robin, so a request whose prefix is cached on server 3 might land on server 5 and miss. Prefix-aware routing hashes the request's prefix and routes it to the server most likely to already hold it — session affinity, but keyed on the shared prefix rather than the user. This turns a fleet-wide cold-miss problem into a per-server warm-hit one, and it is how large deployments keep hit rates high across many GPUs. The cost is a routing constraint: you trade some load-balancing freedom for cache locality, and you need a fallback when the affine server is overloaded.

**Disaggregated KV stores.** At the largest scale, the KV cache becomes a service of its own. Systems like Mooncake (the KV store behind Moonshot's Kimi) separate the KV cache into a distributed, RDMA-connected pool that prefill and decode workers share, so a prefix computed anywhere in the cluster is reusable everywhere. This composes with prefill/decode disaggregation — a separate topic in this series — where prefill and decode run on different hardware and hand off KV over a fast interconnect. The unifying theme: as the shared-prefix working set outgrows one GPU, the cache follows the same tiering logic that CPU caches, page caches, and CDNs have always used — keep the hot set close, spill the cold set to cheaper, larger, slower storage, and fetch back only when the fetch beats the recompute.

The API providers already run this playbook. DeepSeek's context caching is explicitly disk-backed — cache-hit tokens are billed at roughly one-tenth the miss price precisely because a disk fetch of KV is far cheaper than a GPU recompute — and the multi-minute time-to-live on Anthropic's and OpenAI's prompt caches reflects a tiered store with an eviction horizon, not an in-HBM cache that would be far too small to hold every customer's prefix. If you are self-hosting and your single-GPU hit rate is capped by eviction, offloading is the next lever before you buy more GPUs.

## 12. Case studies

**SGLang RadixAttention (Zheng et al., 2024).** The paper that introduced RadixAttention reported throughput improvements up to roughly 5× over prior systems on workloads with heavy prefix sharing — few-shot benchmarks, multi-turn chat, and tree-of-thought reasoning — precisely because the shared context is prefilled once and reused across the batch. The gains were largest on tree-structured programs, where a flat cache leaves reuse on the table. The paper is the canonical reference for the radix-tree approach and for cache-aware scheduling.

**vLLM automatic prefix caching.** vLLM's documentation and release notes describe APC as a default-on feature in the V1 engine, with the largest wins on multi-turn conversations and long-system-prompt serving. The design — prefix-chained block hashing over PagedAttention blocks — is the reference implementation of the flat-hash-table approach, and it composes with continuous batching so that cached and uncached requests batch together seamlessly. See [the vLLM deep-dive](/blog/machine-learning/model-serving/vllm-deep-dive) for how it fits into the broader engine, and the [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization) post for the eviction and memory-pressure mechanics in detail.

**API-level prompt caching (Anthropic, OpenAI, DeepSeek).** The major API providers productized this exact mechanic as a billing feature, which is the strongest possible evidence that it works at scale. Anthropic's prompt caching lets you mark a prefix with a cache breakpoint; cache reads are billed at roughly 10% of the base input price (a 90% discount on cached tokens) with reported latency reductions up to ~85% on long prompts, though cache writes carry a ~25% premium and entries expire after a few minutes of disuse. OpenAI's prompt caching is automatic for prompts at or above ~1,024 tokens, discounting cached input tokens (commonly 50%) with no code change. DeepSeek's context caching bills cache-hit tokens at roughly one-tenth the miss price. The lesson for self-hosting: the same $s/n$ savings you compute for TTFT show up directly on the provider's invoice, and the API vendors are willing to bet their margins on it. The billing structure also encodes the memory hierarchy discussed earlier — the write premium reflects the cost of populating a tiered store, the read discount reflects how cheap a KV fetch is versus a recompute, and the short TTL reflects an eviction horizon on a cache too small to hold every customer's prefix forever.

**Large-scale KV disaggregation (Mooncake / Kimi).** Moonshot AI's Mooncake architecture, which serves the Kimi assistant, treats the KV cache as a disaggregated, cluster-wide store connected over RDMA, so a prefix computed by any prefill worker is reusable by any decode worker. Public descriptions report that this KV-centric design substantially raised effective throughput on their long-context, high-sharing chat workload by turning per-node prefix caches into a fleet-wide one — the extreme end of the offloading and routing ideas from the previous section. It is the strongest demonstration that prefix reuse is not a single-GPU micro-optimization but an architectural principle that scales to the largest deployments.

#### Worked example: reading a case study honestly

Suppose a vendor reports "5× throughput with prefix caching." Before believing it applies to you, decompose it with the amortized formula. A 5× throughput gain on the *shared-prefix portion* implies the workload spent roughly 80% of its prefill on a shared prefix that is now nearly free (${1/(1-0.8) = 5}$). That is consistent with a few-shot or tree-of-thought benchmark where the shared context dwarfs the unique query. If *your* workload is RAG with a 50% shared prefix and a 60% hit rate, your amortized saving is $0.6 \times 0.5 = 0.3$, i.e. a ~1.4× throughput improvement on prefill, not 5×. The vendor number is not wrong; it is measured on a different point of the $s/n$-and-hit-rate surface. Always map a headline speedup back to the shared fraction and hit rate it implies, then re-evaluate at your own operating point.

## 13. When to use this (and when not to)

**Turn it on — it is essentially free — when:**

- You serve a stable system prompt, few-shot exemplars, or an agent scaffold that recurs across requests. This is the overwhelmingly common case, and the win is large (often 5–20× TTFT) with a one-flag change and no accuracy cost.
- You run multi-turn chat. The growing shared history makes caching more valuable with every turn, and without it long conversations degrade badly.
- You do tree-structured generation (beam search, tree-of-thought, self-consistency). Prefer SGLang RadixAttention here; the shared branches map onto shared tree nodes.
- You have RAG with a hot document set or a stable instruction preamble placed *before* the variable context. Order the prompt stable-first to make the shared portion a cacheable prefix.

**Do not bother, or turn it off, when:**

- Every prompt is genuinely unique with no shared prefix — ad-hoc single-document summarization, a search backend where each query stands alone. Hit rate is near zero, and you pay a small hashing cost and, worse, risk cold prefixes churning the cache and stealing KV from the running batch.
- Your shared prefix is tiny (under ~64 tokens). The savings are $s/n$; if $s$ is a handful of tokens, there is nothing to save and the bookkeeping is not worth it.
- Your latency problem is TPOT, not TTFT. Prefix caching does not touch decode. If responses stream slowly, look at [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization), quantization, or speculative decoding instead.
- You are severely KV-memory-constrained and every megabyte of the pool is needed for active-request batching. Retaining many distinct cold prefixes can *reduce* throughput by starving the batch. Measure; if hit rate is low and memory is tight, the cache is a liability.

The decision reduces to one question you can answer from a request sample: what fraction of your prompt tokens are a shared, stable prefix, and how often does that prefix recur? High and often → turn it on and enjoy the free win. Low or never → skip it and spend the memory on batch capacity.

Concretely, here is the procedure I run on a new workload. First, sample a few hundred real prompts and compute the median shared-prefix length as a fraction of prompt length — this is your $s/n$ ceiling. If it is below ~10%, stop; caching will not help. Second, if $s/n$ is high, enable caching and read the live hit rate off the metrics endpoint under real traffic for an hour; if the hit rate is far below what the sharing implies, hunt for a per-request token poisoning the prefix (the alert from the monitoring section is built for exactly this). Third, if the hit rate is capped by eviction rather than by prompt structure, raise `gpu_memory_utilization` / `--mem-fraction-static` and re-measure, then consider CPU/disk offload if a single GPU's KV pool cannot hold the hot working set. Fourth, decide the framework: stay on whatever you run for the common case, and reach for SGLang only if you have heavy tree-structured decoding or many interleaved distinct prefixes that benefit from cache-aware scheduling. Fifth, benchmark honestly — warm the cache, report the hit rate with the speedup, and drive it at production concurrency. That five-step loop turns "should we use prefix caching?" from an opinion into a measurement.

## 14. Key takeaways

- **Prefix caching reuses the KV cache of shared prompt prefixes** so the expensive prefill is paid once and amortized across every request that shares it. It moves TTFT, not TPOT.
- **The savings are the shared-prefix fraction**: prefill FLOPs saved $\approx s/n$, and $\text{TTFT}_{\text{hit}} \approx \text{TTFT}_{\text{full}}(1 - s/n) + t_{\text{lookup}}$. The payoff is non-linear — the last stretch of shared prefix is worth the most.
- **Shared prefixes are everywhere**: long system prompts, few-shot exemplars, growing chat history, agent scaffolds, and hot RAG contexts. Order your prompt stable-part-first so the shared tokens form a genuine prefix.
- **vLLM uses prefix-chained block hashing** over PagedAttention blocks (`enable_prefix_caching=True`, default-on in V1); **SGLang uses a radix tree** with token-level reuse and cache-aware scheduling (`--schedule-policy lpm`). For the common case they perform alike.
- **It rides on PagedAttention** via reference-counted, read-only shared blocks. A cached prefix is safe because KV is a deterministic function of tokens; the first divergent token starts a private block.
- **Correctness is exact and lossless**: sampling parameters (temperature, top_p, seed) do not affect reuse, but the token IDs, tokenizer, model, and adapter must match exactly. A warm request is bit-identical to a cold one.
- **The cache gets better under load** for shared workloads: hot prefixes stay warm and survive LRU eviction; the failure mode is many distinct cold prefixes churning and stealing KV from the batch.
- **Retaining a prefix costs GPU KV memory** (~128 KB/token for Llama-3-8B). One hot prefix is negligible; thousands of cold ones can starve the batch. When to use it comes down to hit rate versus memory pressure.

## 15. Further reading

- Zheng, L. et al. (2024). *SGLang: Efficient Execution of Structured Language Model Programs* — the RadixAttention paper introducing the radix-tree KV cache and cache-aware scheduling.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* — the vLLM paper; the block manager that prefix caching extends.
- vLLM documentation: *Automatic Prefix Caching* — flags, the V1 default, and block-hashing internals.
- SGLang documentation: *RadixAttention and scheduling policies* — `--schedule-policy`, `--mem-fraction-static`, and the frontend language.
- Anthropic and OpenAI prompt-caching guides — the productized, billing-level version of the same mechanic, with real discount and latency figures.
- Within this series: [what is model serving](/blog/machine-learning/model-serving/what-is-model-serving), [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different), [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), [the vLLM deep-dive](/blog/machine-learning/model-serving/vllm-deep-dive), and [KV cache optimization](/blog/machine-learning/model-serving/kv-cache-optimization).
