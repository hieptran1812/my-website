---
title: "Inference at Scale: Batching, the KV Cache, and Tensor-Parallel Serving"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Serving a model is a different HPC problem from training it. Learn why decode is memory-bound on the KV cache, how continuous batching and paged attention fill the GPU, and how to pick a batch size for an SLO."
tags:
  [
    "high-performance-computing",
    "gpu",
    "llm-serving",
    "kv-cache",
    "continuous-batching",
    "paged-attention",
    "vllm",
    "speculative-decoding",
    "tensor-parallelism",
    "deep-learning",
    "ml-systems",
    "inference",
  ]
category: "machine-learning"
subcategory: "High Performance Computing"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-1.png"
---

The first time I owned a model-serving box in production, I made the mistake every training engineer makes: I assumed the hard part was over. We had trained a 7-billion-parameter Transformer, it scored well offline, and shipping it felt like a deployment chore. Two days after launch the on-call pager went off. The box — a single A100 80GB — was returning 502s under a load that, on paper, it should have shrugged off. The GPU utilization graph told a humiliating story: 11%. Eleven percent. We had eighty gigabytes of the fastest accelerator memory money could buy, \$30,000 of silicon, and it was idling while users waited.

The problem was not the model. The problem was that **serving is a different high-performance-computing problem from training**, and I was solving it with training instincts. In training you push one giant batch through a fixed computation graph and you optimize for throughput — samples per second, FLOP utilization, MFU (model FLOPs utilization, the fraction of the GPU's peak arithmetic you actually use). In serving you have hundreds of independent requests arriving at unpredictable times, each generating tokens one at a time, each holding onto a growing chunk of memory called the **KV cache** (the stored keys and values from every previous token, kept so you don't recompute attention over the whole prompt at each step). You are no longer optimizing one number. You are trading two numbers against each other — **throughput** (total tokens per second the box produces across all requests) and **latency** (how long any one request waits) — under a hard service-level objective, an SLO, that says something like "p99 time-to-first-token under 300 ms and p99 inter-token latency under 50 ms."

This post is the serving counterpart to the training-side posts in this series. If you have read [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound) you already have the one tool that explains almost everything here: every kernel is either **compute-bound** (limited by the GPU's arithmetic throughput) or **memory-bound** (limited by how fast it can move bytes from HBM, the high-bandwidth memory on the GPU package). The headline result of this entire post, the thing I wish someone had tattooed on my forehead before that pager went off, is this: **prefill is compute-bound and decode is memory-bound**, and once you internalize that, batching strategy, the KV cache, paged attention, tensor-parallel serving, disaggregation, and speculative decoding all fall out as obvious consequences. By the end you will be able to read a serving roofline, compute the KV-cache size for a 7B model by hand, pick a batch size that meets an SLO, and explain to your team why a continuous-batching server with paged attention does 10–20× the throughput of the naive loop you probably shipped first.

![throughput rises and latency rises together as batch size grows from one to sixty four](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-1.png)

Let us start with the trade-off that governs everything, then take it apart piece by piece.

## 1. Throughput versus latency: the one trade-off that runs serving

Throughput and latency are not the same axis, and serving forces you to choose a point on the curve between them. **Throughput** is the aggregate: tokens per second the whole box emits, summed over every concurrent request. **Latency** is per-request: for a single user, how long until the first token appears (time-to-first-token, TTFT) and how long between subsequent tokens (inter-token latency, ITL, often quoted as time-per-output-token, TPOT). A box can have glorious throughput and terrible latency at the same time — if you batch 256 requests together, you produce a flood of tokens per second, but each individual user is waiting behind 255 others for their slice of each GPU step.

The lever that moves you along the curve is **batch size**: how many requests the GPU processes together in one forward pass. Here is the intuition before any math. A GPU is a throughput machine — it has thousands of arithmetic units (in the [SM-and-warp execution model](/blog/machine-learning/high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model) those are the CUDA cores and Tensor Cores) that all want to be busy. When you decode one token for one request, you load the entire 14 GB of model weights from HBM, do a tiny amount of arithmetic on a single token's worth of activations, and write the result back. The arithmetic is trivial; the weight-loading dominates. So you paid for an enormous memory read and used almost none of the compute. If instead you decode one token for 64 requests in the same pass, you load those same weights *once* and amortize them across 64 tokens of useful work. Throughput goes up roughly 64× with barely any extra latency — until you hit a wall we will get to. That is why batching exists, and why a serving engineer's first question is always "what's my batch size and why isn't it bigger?"

The numbers in figure 1 are representative of a 7B model on a single A100 80GB. At batch 1 you get something like 900 tokens/s aggregate with a p50 latency of ~22 ms per token, and the GPU sits around 12% utilized — almost all of that 22 ms is HBM bandwidth spent reading weights for a single token. At batch 64 the aggregate throughput climbs toward ~14,000 tokens/s and the GPU is ~88% utilized, but the per-token p50 latency has stretched to ~95 ms because each request now shares the step with 63 others. (Treat these as approximate, hardware- and model-config-dependent figures; the *shape* is the robust part, not the exact values.) The throughput went up ~15× and the latency went up ~4×. Whether that is a good trade depends entirely on your SLO. If your product is a batch document-summarization pipeline, crank the batch and bank the throughput. If your product is an interactive chat assistant, you have a latency budget and you cannot just maximize batch size — you have to find the largest batch that still meets p99.

#### Worked example: tokens per dollar at two batch sizes

Suppose the A100 80GB costs you \$2.00 per GPU-hour (a defensible cloud spot-ish figure in 2026; on-demand list is higher, reserved is lower — mark this approximate). At batch 1 you produce 900 tokens/s, which is $900 \times 3600 = 3.24$ million tokens/hour, costing \$2.00, so roughly **\$0.62 per million tokens** of pure compute. At batch 64 you produce 14,000 tokens/s, which is $14000 \times 3600 = 50.4$ million tokens/hour for the same \$2.00, or roughly **\$0.040 per million tokens**. The batched server is about **15× cheaper per token**. This is the entire economic argument for batching in one line: the GPU costs the same whether you keep it 12% busy or 88% busy, so every idle percent is money you set on fire. The job of a serving stack is to keep that GPU full *without* blowing the latency SLO — and that is a scheduling problem, not a model problem.

So the rest of this post is really one question asked five ways: **how do I keep the GPU full while honoring a latency SLO?** The answers — continuous batching, paged attention, tensor parallelism, disaggregation, speculative decoding — are all techniques for pushing the throughput–latency curve up and to the right, so that at any given latency you get more throughput. But to design them you first have to understand why decode behaves so differently from prefill, which means we have to look at the two phases of generation.

One more piece of vocabulary before we go deeper, because it trips people up. The two latency numbers that matter are not interchangeable, and a good serving SLO quotes both. **Time-to-first-token (TTFT)** is dominated by *prefill* — it is how long the user stares at a blank screen before the first word appears, and it scales with prompt length because prefill has to process the whole prompt. **Inter-token latency (ITL)**, also called time-per-output-token, is dominated by *decode* — it is the cadence of words streaming out, and it scales with how full the decode batch is. A chat product cares about both: a 2-second TTFT feels sluggish, and an ITL above ~50 ms makes the text crawl out slower than a person reads. The reason these are governed by *different* phases on *different* rooflines is exactly why you cannot reason about serving latency as a single number — and it is the seed of the disaggregation idea in section 7, where we physically separate the two phases onto hardware tuned for each. Keep TTFT (prefill, compute) and ITL (decode, memory) mentally separate from here on; almost every confusing serving result becomes obvious once you split latency into these two.

## 2. Prefill versus decode: two phases, two rooflines

Every LLM generation request has two phases, and they live on opposite ends of the roofline. Understanding this split is the single highest-leverage idea in serving.

**Prefill** is the first forward pass over the prompt. If the user sends 2,048 tokens of context, prefill processes all 2,048 tokens *in parallel* through every layer, computing the keys and values for each one and producing the first output token. Because it is processing thousands of tokens at once, the matrix multiplications are large and fat — the kind of dense matmul a GPU loves. **Decode** is everything after: the autoregressive loop that produces one token at a time. Each decode step takes the single most-recent token, runs it through every layer, attends over the entire stored KV cache, and emits exactly one new token. Then it repeats. A 500-token response is 500 sequential decode steps.

The reason these phases are so different comes straight from **arithmetic intensity** — the ratio of floating-point operations to bytes moved from memory, $I = \text{FLOPs} / \text{bytes}$, the x-axis of the roofline. The roofline says a kernel is compute-bound when its intensity exceeds the hardware's ridge point (peak FLOP/s ÷ peak bytes/s) and memory-bound below it. For an A100, peak bf16 is about 312 TFLOP/s and HBM bandwidth is about 2.0 TB/s, giving a ridge point around $312\text{e}12 / 2.0\text{e}12 \approx 156$ FLOP/byte. For an H100 SXM it is roughly $989\text{e}12 / 3.35\text{e}12 \approx 295$ FLOP/byte. Any kernel below that intensity is starved on bandwidth, not arithmetic.

![prefill is compute bound with high intensity while decode is memory bound at one token per step](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-2.png)

Now do the intensity for each phase. In **prefill** with a long prompt, each weight matrix is loaded once from HBM and multiplied against a tall activation matrix of shape (sequence-length × hidden). The FLOPs scale with sequence length; the bytes (the weights) do not. So intensity climbs to the hundreds of FLOP/byte — comfortably above the ridge. Prefill is **compute-bound**: it is limited by Tensor Core throughput, and it is the phase that actually delivers high MFU. In **decode**, you process a single token, so the activation matrix is a skinny vector. You still load the *entire* weight matrix from HBM, but you only do one token's worth of arithmetic against it. The FLOPs are tiny and the bytes are the full weight set, so intensity collapses toward ~1 FLOP/byte — far below any ridge point. Decode is **memory-bound**: it is limited by how fast you can stream the weights (and the KV cache) out of HBM, and per-token MFU is dismal.

This is why a single-request decode loop wastes the GPU. You are paying to read 14 GB of weights from a 2 TB/s bus — about 7 ms of pure bandwidth just to move the weights once — and getting a single token for it. The compute units are nearly idle. And it is *also* why batching is so effective for decode: if you decode 64 requests together, you read the 14 GB of weights once and produce 64 tokens, lifting the effective intensity by ~64× and dragging decode up the roofline toward compute-bound territory. Batching does for decode exactly what the roofline predicts: it raises arithmetic intensity by reusing the loaded weights across more useful work.

But — and this is the pivot for the whole post — there is a *second* thing that decode reads from HBM besides the weights, and it does *not* amortize across the batch. That second thing is the KV cache. Each request in the batch has its own KV cache, so batching multiplies the KV-cache reads. At some batch size the KV cache reads and the KV-cache *memory footprint* become the binding constraint, and throughput stops scaling. To see where that wall is, we have to size the KV cache precisely.

#### Worked example: the per-token bandwidth floor of decode

Take the 7B model, weights in bf16 (2 bytes each), so ~14 GB of weights. One decode step at batch 1 must read all 14 GB from HBM at least once. On an A100 at an *achieved* ~1.6 TB/s (you rarely hit the 2.0 TB/s peak), that is $14 / 1600 \approx 8.75$ ms just to stream the weights, before any KV-cache reads or kernel launch overhead. That sets a hard floor: **single-request decode on this box cannot beat ~114 tokens/s**, no matter how fast the math is, because it is bandwidth-bound on the weights. To go faster you must either batch (amortize the weight read across requests), shrink the weights (quantization — see the [numerical formats post](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8)), or reduce the number of decode steps (speculative decoding). Every serving optimization is one of those three moves, and the roofline tells you which one you need.

Run the same calculation on an H100 SXM to see how the hardware changes the floor. The H100 has ~3.35 TB/s of HBM3 bandwidth, so the 14 GB weight read takes $14 / 3350 \approx 4.2$ ms, lifting the single-request decode ceiling to ~240 tokens/s — roughly double the A100, almost entirely because of the bandwidth, not the FLOPs. This is the clearest possible demonstration that decode is memory-bound: the H100 has ~3× the A100's compute (989 vs 312 bf16 TFLOP/s) but only ~1.7× the bandwidth, and decode speed tracks the *bandwidth* ratio, not the compute ratio. If decode were compute-bound you would expect a 3× speedup; you get ~1.7×, because the wall is HBM. When someone tells you a faster GPU "didn't help much" with their decode latency, this is almost always why — they bought compute when they were bandwidth-bound, and the roofline would have told them so before the purchase order went out.

There is a useful corollary for the *batched* case that we will lean on repeatedly. As you raise the batch, the weight read (a fixed 14 GB) amortizes across more tokens, so the *per-token* contribution of the weight read shrinks as $14\,\text{GB} / B$. But the *KV-cache* read does **not** amortize — each request reads its own cache — so the per-step KV traffic grows as $B \times (\text{KV bytes per request})$. There is therefore a crossover batch size at which the KV-cache reads overtake the weight read as the dominant HBM traffic. Below it, batching is pure win (you are amortizing the fixed weight read). Above it, each additional request adds roughly its full KV-read cost with no amortization, so throughput stops scaling and latency climbs steeply. That crossover is the throughput knee you will see in the SLO sweep later, and it moves *left* (smaller batches) as context length grows, because longer contexts mean bigger per-request KV caches. Long-context serving hits the knee sooner — another reason long context is expensive.

## 3. The KV cache: why decode is bandwidth-bound and memory-hungry

The KV cache is the beating heart of LLM serving, and most people who have only trained models have never had to think about it, because in training you compute attention over the whole sequence at once and throw the keys and values away. In serving you keep them, because recomputing attention over the entire history at every decode step would be quadratically wasteful. So at each step you cache the new token's key and value vectors and reuse all the previous ones. The cache is what makes autoregressive generation linear instead of quadratic in the output length — and it is also what makes serving a *memory* problem.

Here is the exact size. For a Transformer, the KV cache stores, per token, per layer, the key and value vectors across all attention heads. The formula is:

$$
\text{KV bytes} = 2 \times L \times H \times d_\text{head} \times S \times B \times \text{bytes}
$$

where the leading 2 counts keys and values, $L$ is the number of layers, $H$ is the number of key/value heads, $d_\text{head}$ is the per-head dimension, $S$ is the sequence length (prompt + generated so far), $B$ is the batch size (number of concurrent requests), and `bytes` is the element size (2 for bf16/fp16). Note that $H \times d_\text{head}$ is just the model's hidden dimension $d_\text{model}$ when there is one KV head per query head, so a common shorthand is $\text{KV bytes} = 2 \times L \times d_\text{model} \times S \times B \times 2$.

![the kv cache scales with batch times sequence and overtakes the fixed weights at long context](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-3.png)

#### Worked example: KV-cache size for a 7B model

Take a 7B model with roughly these dimensions: $L = 32$ layers, $d_\text{model} = 4096$, so with standard multi-head attention $H \times d_\text{head} = 4096$. Per token, per layer, you store 2 (K and V) × 4096 elements × 2 bytes = 16,384 bytes = 16 KB. Across 32 layers that is $32 \times 16\text{ KB} = 512$ KB **per token**. So:

- **One request, 2,048-token context:** $512\text{ KB} \times 2048 \approx 1.0$ GB of KV cache. For a single request.
- **One request, 32,768-token context:** $512\text{ KB} \times 32768 \approx 16$ GB — larger than the 14 GB of model weights. At long context, the KV cache for a *single* request dominates memory.
- **Batch 32, 4,096-token context:** $512\text{ KB} \times 4096 \times 32 \approx 64$ GB. On an 80 GB A100 with 14 GB of weights and a few GB of runtime overhead, that does not fit — you OOM. This is the wall in figure 3.

This calculation explains the most common serving failure mode I have seen: a box that has plenty of room for the weights and falls over the moment real traffic with long contexts arrives, because the KV cache — which scales with **batch × sequence length** — eats the budget. The weights are a fixed cost; the KV cache is the variable cost, and it is the one that decides how many concurrent requests you can actually hold. Here is the calculator I keep in my back pocket:

```python
def kv_cache_bytes(num_layers, hidden_dim, seq_len, batch, dtype_bytes=2, num_kv_heads=None, num_q_heads=None):
    """Bytes of KV cache for a Transformer.
    For multi-head attention (MHA) leave num_kv_heads=None.
    For grouped-query attention (GQA) pass num_kv_heads < num_q_heads to shrink the cache.
    """
    if num_kv_heads is not None and num_q_heads is not None:
        # GQA: KV is stored only for the smaller number of KV heads
        kv_dim = hidden_dim * num_kv_heads / num_q_heads
    else:
        kv_dim = hidden_dim
    # 2 = keys + values
    bytes_per_token = 2 * num_layers * kv_dim * dtype_bytes
    return int(bytes_per_token * seq_len * batch)

gb = 1024 ** 3
print(kv_cache_bytes(32, 4096, 2048, 1) / gb)    # ~1.0 GB  (one request, 2k ctx)
print(kv_cache_bytes(32, 4096, 32768, 1) / gb)   # ~16 GB   (one request, 32k ctx)
print(kv_cache_bytes(32, 4096, 4096, 32) / gb)   # ~64 GB   (batch 32, 4k ctx -> OOM on 80GB)

# Grouped-query attention: 32 query heads but only 8 KV heads -> 4x smaller cache
print(kv_cache_bytes(32, 4096, 4096, 32, num_kv_heads=8, num_q_heads=32) / gb)  # ~16 GB
```

Notice the last line. **Grouped-query attention (GQA)** — used by Llama-2 70B, Llama-3, Mistral, and most modern models — stores keys and values for far fewer heads than the queries use (e.g. 8 KV heads serving 32 query heads). That shrinks the KV cache by 4× in this example, which is *not* a quality trick — it is a serving trick. The whole reason GQA exists is that the KV cache, not the weights, is the binding memory constraint at serving time, and the most direct way to serve more concurrent requests is to make each one's KV cache smaller. (For deeper KV-cache tactics — quantizing the cache to fp8, sliding-window attention, KV eviction — see the dedicated post on [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management).)

Now connect the size back to bandwidth. During decode, every step reads the *entire* KV cache for every request (attention has to look at all prior tokens). So as the context grows, decode reads more bytes per step, and per-token latency climbs. A request at 32k context is doing far more HBM traffic per decode step than the same request at 2k context — not because the math grew much, but because the KV cache it has to stream grew. This is why long-context serving is *expensive*: it is memory-bandwidth-bound on a cache that grows linearly with the conversation. The KV cache is simultaneously a memory-*capacity* problem (it has to fit) and a memory-*bandwidth* problem (it has to be read every step). Both are HBM problems. Neither is a compute problem. Serving lives on the memory wall, and the KV cache is the wall.

It is worth pausing on *why* the KV cache exists at all, because the alternative tells you what you are trading. Without a cache, each decode step would recompute attention over the entire history: at step $t$ you would re-derive the keys and values for all $t$ prior tokens, which is $O(t)$ work per step and $O(S^2)$ over a full generation of length $S$. That is the quadratic cost of attention, paid again and again. The KV cache turns that into $O(1)$ extra work per step (you compute K and V for only the *new* token and append them) at the cost of *storing* all the prior K and V — so it is the canonical time-versus-space trade. You spend memory (and the bandwidth to read it) to avoid recomputation. For a 500-token response that is the difference between ~125,000 attention-context-recomputations and 500 appends; nobody serves without a KV cache. But having made that trade, you now own a memory object that grows linearly with every conversation and must be read in full every step — and managing *that* object well is what paged attention, GQA, fp8 KV quantization, and KV eviction are all about.

One more wrinkle that changes the arithmetic for big models: **grouped-query attention (GQA)** and its extreme, **multi-query attention (MQA)**. Standard multi-head attention gives every query head its own key and value heads, so the KV cache stores $H$ heads' worth of K and V. MQA (Shazeer 2019) shares a *single* K/V head across all query heads, shrinking the cache by a factor of $H$ — dramatic, but it can cost a little quality. GQA (Ainslie 2023) is the pragmatic middle: it shares each K/V head across a small *group* of query heads, e.g. 8 KV heads for 32 query heads, a 4× cache reduction with negligible quality loss. The reason every frontier model since Llama-2 70B ships GQA is not subtle: the model architects looked at the serving roofline, saw that the KV cache — not the weights and not the compute — was going to bound concurrency, and shrank the cache at the source. GQA is a serving optimization baked into the model. When you pick a model to serve, its number of KV heads is one of the most important serving numbers on the spec sheet, and most people never look at it.

## 4. Continuous batching: stop idling on the longest request

Now we have the physics. Time for the first big systems win, and it is enormous: **continuous batching**, also called **in-flight batching** (the term used in the [Orca paper, Yu et al. 2022](https://www.usenix.org/conference/osdi22/presentation/yu), which introduced iteration-level scheduling). To see why it matters, look at what naive **static batching** does.

In static batching you collect a batch of requests, run them all together until *every one* of them finishes, then return the batch and start the next one. The problem is that requests have wildly different output lengths. Suppose you batch eight chat requests. One asks "what's 2+2" and finishes in 3 tokens. Another asks for a 1,000-token essay. With static batching, the seven short requests *sit in the batch doing nothing* for the entire time the long one keeps generating. Their slots are occupied, their KV cache is held, but they finished hundreds of steps ago. The GPU is processing a batch of 8 where 7 of the 8 lanes are dead weight. Worse, no *new* request can join until the whole batch flushes, so requests queue up behind a batch that is mostly idle. This is exactly the 11%-utilization failure I opened with.

![static batching idles finished slots while continuous batching admits new requests every step](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-4.png)

**Continuous batching** schedules at the granularity of a single decode step (an "iteration") instead of a whole request. After every step, the scheduler checks: did any request finish? If so, evict it, free its KV cache, and admit a waiting request into the freed slot — *on the very next step*. The batch is never "full of corpses." A request that finishes in 3 tokens leaves after 3 steps and a fresh request takes its place immediately. The GPU stays full of *live* work. This single change — scheduling per-iteration rather than per-request — is the difference between 35% and 85% GPU utilization in figure 4, and it is the core idea behind every modern serving engine: vLLM, TensorRT-LLM's in-flight batching, TGI, and SGLang all do iteration-level scheduling.

Here is the scheduler sketch. The real implementations are more careful (preemption, priority, chunked prefill), but this captures the essence — admit-on-free, evict-on-finish, every step:

```python
class ContinuousBatchScheduler:
    def __init__(self, model, max_num_seqs, kv_pool):
        self.model = model
        self.max_num_seqs = max_num_seqs   # batch-size cap (memory-bound)
        self.kv_pool = kv_pool             # paged KV-cache allocator (next section)
        self.running = []                  # requests currently decoding
        self.waiting = []                  # admitted, not yet running

    def step(self):
        # 1. Admit waiting requests into any free slots, if KV pages are available.
        while (len(self.running) < self.max_num_seqs
               and self.waiting
               and self.kv_pool.can_allocate(self.waiting[0])):
            req = self.waiting.pop(0)
            req.kv_blocks = self.kv_pool.allocate(req)
            self.running.append(req)

        if not self.running:
            return

        # 2. One fused forward pass over ALL running requests (mixed prefill + decode).
        logits = self.model.forward(self.running)        # batched, one GPU step

        # 3. Sample one new token per running request; grow each KV cache by one page if needed.
        for req in self.running:
            tok = sample(logits[req.idx], req.sampling_params)
            req.append(tok)
            if req.needs_new_page():
                req.kv_blocks += self.kv_pool.allocate_one(req)

        # 4. Evict finished requests THIS step and free their KV pages for the next admit.
        finished = [r for r in self.running if r.is_done()]
        for req in finished:
            self.kv_pool.free(req.kv_blocks)
            req.return_to_client()
        self.running = [r for r in self.running if not r.is_done()]
```

Two subtleties worth calling out, because they bite people. First, `max_num_seqs` is the batch-size cap and it is fundamentally a **memory** limit: it is set by how many requests' KV caches fit in HBM, not by compute. That is the wall from section 3 reappearing as a tuning knob. Second, a forward pass can mix **prefill** (a newly admitted request crunching its whole prompt) and **decode** (running requests producing one token). Naively mixing them lets a long prefill stall every decoder's latency for a step — which is why modern engines use **chunked prefill**: they split a long prompt into chunks and interleave prefill chunks with decode steps so a 4,000-token prompt does not freeze the token stream for everyone else. We will return to that tension in the disaggregation section, because it is one of the deepest in serving.

The measured impact is large. The Orca paper reported up to ~36× throughput improvement over a request-level-batched baseline (FasterTransformer) at the same latency on certain workloads; vLLM's continuous batching plus paged attention reports 2–4× higher throughput than prior serving systems and up to ~24× over naive HuggingFace `generate` in their benchmarks (Kwon et al. 2023, the [PagedAttention/vLLM paper](https://arxiv.org/abs/2309.06180)). The exact multiplier depends heavily on how skewed your output lengths are — the more variance in response length, the more static batching wastes, and the bigger the continuous-batching win. Treat the headline numbers as workload-dependent, but the direction is rock-solid: per-iteration scheduling is the single biggest free lunch in serving.

To see *why* the variance matters so much, do the back-of-envelope. Suppose static batching of 8 requests where 7 finish in 50 tokens and 1 runs to 1,000 tokens. With static batching the whole batch runs for 1,000 steps, and the seven short requests occupy a slot for 950 steps each *after they are done* — that is $7 \times 950 = 6{,}650$ wasted slot-steps out of $8 \times 1000 = 8{,}000$ total, so **83% of the GPU's decode work in that batch is wasted on dead requests.** Continuous batching reclaims essentially all of it: each short request frees its slot at step 50, and a fresh request takes over. The waste is bounded by the one-step granularity of admission, not by the longest request in the batch. This is the entire mechanism, and it is why the win is biggest exactly where it hurts most — chat and agent workloads, where response lengths span two orders of magnitude. For a workload where every response is the same length (e.g. fixed-length classification), static and continuous batching are nearly identical, because there are no dead slots to reclaim. The variance is the value.

A subtlety the sketch glosses over but production engines obsess about: **preemption and fairness.** When the KV pool fills and a running request needs another page, the scheduler may have to *preempt* a request — either recompute its KV later (recomputation) or swap its pages out to host memory (swapping) — to make room. vLLM supports both. Preemption is the pressure-relief valve that lets the engine admit aggressively without OOMing, at the cost of occasionally re-running some work. The practical effect is that under memory pressure your throughput degrades gracefully (some requests get preempted and resumed) instead of the server crashing — but you will see it as latency spikes for the preempted requests, and a tail-latency (p99) that is worse than the median. If your p99 ITL is much worse than p50, preemption under KV pressure is the first thing to check, and the fix is usually a smaller `max_num_seqs`, a cheaper KV cache (GQA/fp8), or more HBM.

## 5. Paged attention: the KV cache in pages, no fragmentation

Continuous batching admits and evicts requests every step. That means the KV cache is being allocated and freed constantly, in chunks of wildly different sizes (a 3-token request and a 4,000-token request), at high frequency. If you allocate each request's KV cache as one contiguous block sized to its *maximum possible* length — which is what naive serving does — you get a memory allocator's nightmare: **fragmentation**.

Here is the concrete waste. Suppose you reserve a contiguous 2,048-token KV region for each request because that is the model's max length. A request that only generates 100 tokens uses 100 slots and wastes 1,948 — about 95% of its reservation sits empty but *reserved*, so no other request can use it. Even averaging over realistic length distributions, the vLLM authors measured that contiguous pre-allocation wastes **60–80% of KV memory** to internal and external fragmentation. That wasted memory directly caps your batch size, which directly caps your throughput. Fragmentation in the KV cache is throughput you are leaving on the floor.

![the kv cache lives in fixed pages mapped by a block table so any free page serves any request](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-5.png)

**PagedAttention** (Kwon et al. 2023, the technique behind vLLM) borrows the oldest trick in operating systems: virtual memory and paging. Instead of one contiguous KV region per request, the KV cache is carved into fixed-size **pages** (vLLM calls them blocks) — typically 16 tokens of KV per page. Each request gets a **block table** that maps its logical token positions to physical pages, which can live *anywhere* in the pool and need not be contiguous. When a request needs more KV space, it grabs one free page from a shared free list. When it finishes, its pages go straight back to the free list for any other request to use. The attention kernel is modified to gather KV from the scattered pages via the block table (this is the "paged" in PagedAttention — the kernel does the page lookups during the attention computation).

The payoff is the same as OS paging: near-zero fragmentation. A request wastes at most *one partial page* — under 16 tokens, a few percent — instead of an entire over-provisioned contiguous region. vLLM measured KV waste dropping from 60–80% to **under 4%**. Because you waste so little, you fit far more concurrent requests in the same HBM, your effective batch size goes up, and throughput rises with it. Paging is *why* vLLM can run 2–4× the batch size of a contiguous-allocation server on the same GPU.

Paging unlocks a second, almost magical, win: **prefix sharing**. If two requests share a prompt prefix — the same system prompt, the same few-shot examples, the same RAG document prepended to every query — they can *share the physical KV pages* for that prefix via copy-on-write, exactly like `fork()` shares pages. You compute the shared prefix's KV once and point every request's block table at it. For a chatbot with a 1,000-token system prompt served to thousands of users, this is a massive saving in both memory and prefill compute. From an end user you would never know; from a serving engineer's seat it is the difference between affording the system prompt and not. Here is what driving vLLM actually looks like — the engine handles paging, continuous batching, and prefix sharing under the API:

```python
from vllm import LLM, SamplingParams

# gpu_memory_utilization: fraction of HBM vLLM may claim for weights + KV pool.
#   Higher -> bigger KV pool -> more concurrent requests, but leave headroom for
#   activations and CUDA context or you OOM mid-flight. 0.90 is a sane default.
# max_num_seqs: the batch-size cap (max requests decoded together). Memory-bound,
#   not compute-bound: too high and the KV pool can't hold every request's cache.
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_num_seqs=256,
    max_model_len=4096,
    enable_prefix_caching=True,    # share KV pages across common prompt prefixes
)

params = SamplingParams(temperature=0.7, max_tokens=256)

# Pass a whole list: vLLM continuously batches these internally, admitting and
# evicting per iteration. You do NOT loop one prompt at a time.
prompts = [f"Summarize document {i} in three sentences." for i in range(1000)]
outputs = llm.generate(prompts, params)

for out in outputs:
    print(out.outputs[0].text)
```

The thing to internalize: `gpu_memory_utilization` and `max_num_seqs` are the two knobs that decide how full your GPU runs. The first sets how much HBM becomes KV-cache pool; the second caps the batch. Both push against the same wall — the KV cache must fit. Set `gpu_memory_utilization` too low and you starve the KV pool, forcing small batches and low throughput. Set it too high and you OOM when a burst of long-context requests arrives and the KV cache balloons past what you left room for. Tuning these two numbers against your traffic's context-length distribution is most of what serving tuning *is*.

## 6. Tensor-parallel serving: when the model (or its cache) won't fit

Everything so far assumed the model fits on one GPU. A 7B model in bf16 (14 GB) fits comfortably on an A100 80GB with room for a healthy KV pool. But a 70B model in bf16 is ~140 GB — it does not fit on any single GPU. And even for models that *do* fit, you sometimes want to spread them across GPUs to get more aggregate HBM bandwidth (remember, decode is bandwidth-bound) and more KV-cache room. The tool is **tensor parallelism** (TP), covered for training in the [parallelism strategies post](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert); here is how it plays out specifically for serving.

Tensor parallelism shards each layer's weight matrices *across* GPUs. For attention, you split the heads — GPU 0 gets heads 0–7, GPU 1 gets heads 8–15, and so on. Each GPU computes its slice in parallel, and then the partial results are combined with an **all-reduce** (every GPU sums its contribution and ends up with the full result — see [collective communication and NCCL](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch) for the mechanics). For the MLP, the first linear is split column-wise and the second row-wise, so the layer needs exactly two all-reduces per forward pass. Crucially for serving, **the KV cache is sharded too**: GPU 0 only stores K and V for its heads. So with TP=4, each GPU holds one-quarter of the weights *and* one-quarter of each request's KV cache. That is the serving superpower of TP — it multiplies both your weight capacity and your KV-cache capacity by the number of GPUs.

![each transformer layer is sharded across four gpus that compute in parallel then all reduce](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-6.png)

Launching a tensor-parallel server is one parameter in vLLM:

```python
from vllm import LLM

# tensor_parallel_size=4 shards the model across 4 GPUs on one node.
# Each GPU holds 1/4 of the weights and 1/4 of every request's KV cache, and the
# four GPUs all-reduce twice per layer (after attention, after the MLP).
# Use NVLink/NVSwitch within a node -- the all-reduce is on the critical path of
# EVERY token, so a slow interconnect (PCIe) directly inflates inter-token latency.
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    dtype="bfloat16",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
)
```

```bash
# The OpenAI-compatible server form -- same knob, run as a service:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --max-model-len 4096
```

Now the honest trade-off, because TP is not free. Those two all-reduces per layer sit on the **critical path of every single token**. With 32 layers and TP=4, decode does 64 all-reduces per token across the GPUs. If those GPUs are connected by NVLink/NVSwitch (~900 GB/s, see [interconnects](/blog/machine-learning/high-performance-computing/interconnects-nvlink-nvswitch-infiniband-and-rdma)), the all-reduce cost is small relative to the compute it overlaps with and TP scales well. If they are connected by PCIe (~64 GB/s), the all-reduce dominates and your inter-token latency *gets worse* as you add GPUs — you bought more compute and made the thing slower. This is the cardinal rule of tensor-parallel serving: **TP only pays inside a fast-interconnect domain.** Keep TP within an NVLink node (typically ≤8 GPUs). Past the node boundary, where you cross InfiniBand, you switch to pipeline parallelism (split layers across nodes, which communicates far less) rather than extending TP, because a cross-node all-reduce on every token is latency suicide.

Let me make that quantitative, because the all-reduce cost is computable and you should compute it before you commit to TP. A ring all-reduce of a tensor of $S$ bytes across $N$ GPUs moves $2(N-1)/N \cdot S$ bytes per GPU (the derivation is in the [NCCL all-reduce post](/blog/machine-learning/high-performance-computing/collective-communication-and-nccl-all-reduce-from-scratch)). At decode, the tensor being reduced is the hidden state for the batch: shape (batch × hidden), so for batch $B = 32$, hidden 4096, bf16, that is $32 \times 4096 \times 2 = 262{,}144$ bytes ≈ 256 KB per all-reduce. With $N = 4$ GPUs, each GPU moves $2 \times 3/4 \times 256\,\text{KB} = 384$ KB per all-reduce. Two all-reduces per layer × 32 layers = 64 all-reduces, so ~24 MB of cross-GPU traffic *per decode step*. On NVLink at 900 GB/s that is $24\text{e}6 / 900\text{e}9 \approx 27$ µs of communication per token — a few percent of a ~5–10 ms decode step, easily hidden behind compute. On PCIe at 64 GB/s the *same* 24 MB takes $24\text{e}6 / 64\text{e}9 \approx 375$ µs, and because the all-reduce is on the critical path (you cannot start the next layer until the reduce completes), it is *not* fully hideable — it inflates every token's latency by hundreds of microseconds, and the inflation grows with $N$. That is the arithmetic behind "TP only pays on NVLink": it is not a vibe, it is a 14× difference in communication time that lands directly on inter-token latency.

#### Worked example: when to add a second GPU for a 7B model

A 7B model fits on one A100 with ~62 GB of KV pool — enough for, say, 60 concurrent requests at 2k context. Suppose your traffic doubles and you want 120 concurrent requests at the same latency. You have two options. **Option A: add a second GPU as a separate replica** (data parallelism — two independent servers behind a load balancer). Each handles 60 requests; no communication; latency unchanged; throughput doubles. **Option B: TP=2 across both GPUs** for one server. Now each request's KV cache is split across two GPUs, so the *single* server can hold more concurrent requests *and* each token's compute is split — but you pay two all-reduces per layer per token. For a model that already fits, **Option A (replicas) almost always wins**: it is simpler, has no communication overhead, and scales linearly. Reach for TP only when the model (or the KV cache for your context length) genuinely does not fit on one GPU, or when you need lower single-request latency than one GPU can deliver. The rule of thumb: *don't add tensor parallelism if the model fits and a replica gets you the throughput.* TP is a capacity tool first and a latency tool second, not a throughput-scaling tool.

## 7. Disaggregated prefill and decode: stop the two phases from fighting

We have a tension we noted back in section 4 and never resolved: prefill and decode want *opposite* things from the GPU, and forcing them to share a server makes both worse. Disaggregation is the architecture that fixes it, and it is one of the most important recent ideas in production serving (used in DeepSeek's inference stack, NVIDIA's Dynamo, and the [DistServe paper, Zhong et al. 2024](https://arxiv.org/abs/2401.09670), among others).

Recall the phases. **Prefill is compute-bound** — it wants to run the biggest matmuls possible and saturate the Tensor Cores. **Decode is memory-bound** — it wants HBM bandwidth and lots of small concurrent requests to amortize the weight read. When you run both on the same GPU with continuous batching, you constantly interleave them, and they interfere. A long prefill (a 4,000-token prompt) monopolizes a step and freezes every decoder's token stream — that is a **TTFT-vs-ITL** conflict. Chunked prefill softens it, but the fundamental mismatch remains: you are tuning one GPU's batch size and scheduling for two workloads with opposite roofline positions, and you cannot optimize both at once.

![a router sends prompts to prefill gpus then streams the kv cache to decode gpus so neither stalls](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-7.png)

**Disaggregation** runs prefill and decode on *separate GPU pools*. A router classifies each request, sends the prompt to a **prefill pool** (sized and tuned to be compute-bound — large prefill batches, maybe TP for the big matmuls), which computes the full KV cache and the first token. Then it **transfers the KV cache** over NVLink or RDMA to a **decode pool** (sized and tuned to be memory-bound — many small concurrent requests, maximizing HBM-bandwidth amortization). The decode pool streams out the rest of the tokens. Now each pool does one thing and does it well: prefill GPUs run hot on compute, decode GPUs run hot on bandwidth, and a long prompt arriving at the prefill pool *cannot* stall the token stream of users already in the decode pool. You can scale the two pools independently — if your traffic is prompt-heavy (RAG with huge contexts), add prefill GPUs; if it is generation-heavy (long responses), add decode GPUs.

The cost is the **KV-cache transfer**: after prefill you have to move the full KV cache (gigabytes for a long prompt — recall ~1 GB for a 2k-context 7B request) from the prefill GPU to the decode GPU. Over NVLink (~900 GB/s) that 1 GB takes ~1.1 ms, negligible against the prefill itself; over a slower interconnect it can become the bottleneck, which is why disaggregation pays off most when prefill and decode pools share a fast fabric. The DistServe paper reported being able to serve **2–4× more requests** (or meet tighter SLOs) under the same hardware budget by disaggregating, precisely because it stops the two phases from interfering and lets each be tuned to its own roofline. The headline lesson, and it is the through-line of this whole post: **prefill and decode are different HPC problems, so on a big enough deployment you stop pretending they are the same workload and give each its own optimized pool.**

There is a deeper reason disaggregation wins that is worth stating plainly: it lets you choose *different hardware* for the two phases. Prefill, being compute-bound, benefits from the highest-FLOP GPUs you can buy and from large prefill batches. Decode, being bandwidth-bound, benefits from the highest-HBM-bandwidth GPUs and cares far less about peak FLOPs. In a unified server you must buy one GPU type and compromise. In a disaggregated fleet you could, in principle, put prefill on compute-dense parts and decode on bandwidth-dense parts, and size the *number* of each independently to match your traffic's prompt-to-output ratio. The DistServe and related work frames this as optimizing **goodput** — requests served per second that *meet their SLO* — rather than raw throughput, because raw throughput is easy to inflate by sacrificing latency, while goodput forces you to respect the TTFT and ITL budgets simultaneously. Disaggregation raises goodput precisely because it stops the two phases from forcing latency compromises on each other.

When is disaggregation *not* worth it? On a single GPU or a small deployment, the transfer overhead and the operational complexity outweigh the benefit — chunked prefill on a unified server is simpler and good enough. Disaggregation is a large-fleet optimization: it shines when you have enough traffic to keep dedicated pools busy and a fast interconnect to move the KV cache. Below that scale, keep it unified. The progression is the same one this whole series preaches: start on one GPU, get the single-box fundamentals right (continuous batching, paging, a sane batch size), and only reach for the cross-GPU and cross-pool machinery when the single-box roofline says you have run out of room. Disaggregation is the last lever, not the first.

## 8. Speculative decoding: spend compute to buy back latency

Decode is sequential and memory-bound: each token needs the previous one, and each step reads the whole model from HBM to produce a single token. The weight-read floor we computed (~8.75 ms/token for a 7B on an A100) is a *latency* floor for a single request that no amount of batching fixes — batching helps throughput, not the latency of one user's stream. **Speculative decoding** attacks that latency floor directly, and it does so with a beautiful trick: turn the memory-bound sequential decode into a compute-bound parallel *verification* (Leviathan et al. 2023 and Chen et al. 2023, the two foundational [speculative decoding papers](https://arxiv.org/abs/2211.17192)).

The idea: use a small, cheap **draft model** to *guess* several tokens ahead, then use the big **target model** to *verify* all of them in a single forward pass. Because the target verifies $k$ proposed tokens in *one* pass — the same cost as generating one token, since the forward pass is over a short sequence and the work is dominated by the fixed weight read either way — you get up to $k$ tokens for the price of one target step, *as long as the draft's guesses are accepted*. And here is the elegant part: the verification uses the target model's own probabilities to accept or reject each proposed token in a way that is **provably distribution-preserving** — the output is mathematically identical to what plain target decoding would have produced. Speculative decoding is not an approximation. It changes nothing about the output quality; it only changes the *speed*.

![a draft model proposes four tokens that the target verifies in one pass accepting the agreed prefix](/imgs/blogs/inference-at-scale-batching-kv-cache-and-tensor-parallel-serving-8.png)

The mechanism per step: (1) the draft model autoregressively proposes $k$ tokens (cheap — the draft is maybe 10–50× smaller). (2) The target model runs *one* forward pass over the prompt plus all $k$ proposed tokens, getting its own probability for each position in parallel. (3) Walk the proposals left to right, accepting each as long as it agrees with the target's distribution (via a rejection-sampling criterion); at the first disagreement, reject the rest and resample that one position from the target. So you always accept *at least* one token (the resampled one) and *at most* $k+1$.

The expected speedup is exactly the **average number of tokens accepted per target step**. If the draft is good and your acceptance rate is high, you advance several tokens per expensive target pass. Quantitatively, with acceptance probability $\alpha$ per token and $k$ proposals, the expected accepted length per step is roughly $\frac{1 - \alpha^{k+1}}{1 - \alpha}$. At $\alpha = 0.75$ and $k = 4$, that is about $\frac{1 - 0.75^5}{0.25} \approx \frac{1 - 0.237}{0.25} \approx 3.05$ tokens per target step — so you do ~one target forward pass to advance ~3 tokens instead of ~3 passes, a roughly **2.4–3× wall-clock speedup** on the decode phase (net of the draft's own cost). The reported numbers in the literature land in the 2–3× range for well-matched draft/target pairs.

Why does verifying $k$ tokens cost roughly the same as generating one? Because — and this is the whole reason speculative decoding works on a memory-bound phase — a single decode step is dominated by the *weight read* (those ~14 GB streamed from HBM), not by the arithmetic. Running the target forward over 1 token versus over $k+1 = 5$ tokens reads the same weights once and does only $5\times$ the (tiny) per-token arithmetic. Since decode was compute-idle to begin with (intensity ~1, far below the ridge), that extra arithmetic is nearly free — it slots into compute units that were sitting empty while HBM did all the work. Speculative decoding is, in roofline terms, a way of *converting idle decode compute into reduced decode latency.* It moves decode rightward on the roofline (raises its arithmetic intensity by doing more useful math per weight-read) and shortens the sequential chain of steps. That is also precisely why it stops paying at high batch sizes: once you have batched enough requests to saturate the compute units, there is no idle compute left to convert, and the verification work now competes for the same Tensor Cores the batch needs. The technique lives in the gap between "memory-bound, compute idle" (low batch) and "compute-bound, compute full" (high batch), and it is strongest at the low-batch end where single-request latency is the pain.

Here is a compact, runnable sketch of the verification loop — the real implementations (vLLM's spec-decode, Medusa, EAGLE) are more optimized, but the accept/reject logic is exactly this:

```python
import torch

def speculative_step(draft, target, prefix, k=4):
    """Advance generation by 1..k+1 tokens per call, output identical to target-only decoding."""
    # 1. Draft proposes k tokens autoregressively (cheap: draft is ~10-50x smaller).
    proposed, draft_probs = draft.propose(prefix, k)        # k tokens, k distributions

    # 2. Target verifies ALL k proposals in ONE forward pass over prefix + proposed.
    #    target_probs[i] is the target's distribution at position i, computed in parallel.
    target_probs = target.forward(prefix + proposed)        # one weight-read, k+1 positions

    # 3. Walk left to right; accept token i with prob min(1, p_target/p_draft) (rejection sampling).
    accepted = []
    for i, tok in enumerate(proposed):
        r = torch.rand(1).item()
        if r < min(1.0, target_probs[i][tok] / draft_probs[i][tok]):
            accepted.append(tok)                            # agreed: keep it, free
        else:
            # First disagreement: resample THIS position from the adjusted target dist, stop.
            resampled = sample_adjusted(target_probs[i], draft_probs[i])
            accepted.append(resampled)
            return accepted                                  # 1..k tokens this step
    # All k accepted: also take the target's free next-token prediction -> up to k+1.
    accepted.append(sample(target_probs[k]))
    return accepted                                          # k+1 tokens this step
```

The accept criterion `min(1, p_target / p_draft)` plus the adjusted resampling on rejection is what makes the output *distribution-identical* to plain target decoding — it is rejection sampling that corrects the draft's bias exactly. This is the property that lets you turn it on without a quality regression test: you are not approximating the target, you are sampling from it, just faster.

#### Worked example: when speculative decoding helps and when it backfires

Acceptance rate is everything. If the draft model agrees with the target 75% of the time, you get ~2.4–3× — great. If the draft is poorly matched and acceptance drops to 30%, the expected accepted length per step collapses toward ~1.4 tokens, and once you subtract the draft model's own forward-pass cost, you can end up *slower* than plain decoding — you paid for draft passes and threw most of them away. So speculative decoding is a bet on a *cheap, well-aligned* draft. The practical patterns: use a small model from the same family (Llama-7B drafting for Llama-70B), or **self-speculation** like Medusa (extra prediction heads on the target itself, no separate draft), or **n-gram / prompt-lookup** speculation (propose tokens copied from the prompt — superb for summarization and code where the output quotes the input). There is also a subtlety at high batch size: speculative decoding trades *extra compute* (the verification pass over $k+1$ tokens) for *fewer sequential steps*. That is a great trade when the GPU has spare compute — which is exactly the case in **low-batch, latency-sensitive** decode, where the box is memory-bound and the compute units are idle anyway. At very high batch sizes the GPU is already compute-saturated, the spare compute is gone, and the extra verification work can *hurt* throughput. So the rule is: **speculative decoding is a latency tool for low-to-moderate batch sizes, not a throughput tool for a saturated box.** It buys back the single-request latency floor by spending the compute that decode was wasting.

## Case studies / real numbers

Let me ground all of this in named results from the literature and from production. I have marked figures approximate where they depend on workload or config; never trust a serving number without knowing the model, the GPU, and the traffic shape.

**Continuous batching and paged attention (vLLM, Kwon et al. 2023).** On an NVIDIA A100, vLLM reported **2–4× higher throughput** than the previous-best serving systems (FasterTransformer, Orca) at the same latency, and up to **~24×** over naive HuggingFace `generate`, for LLaMA-family models. The gain comes from two compounding effects: continuous batching (no idle slots) and paged attention (KV waste cut from 60–80% to <4%, so the batch size roughly 2–4×). The lesson: the algorithm and hardware were the same; the *systems* changed, and that was a 10–20× swing. If you are still serving with a `for` loop over `model.generate`, you are leaving an order of magnitude on the table.

**The KV cache as the serving bottleneck.** Across essentially every production LLM deployment I have seen or read about, the binding constraint on concurrency is KV-cache memory, not weights and not compute. This is *why* GQA exists (Llama-2 70B, Llama-3, Mistral all use it to shrink the cache 4–8×), why fp8 KV-cache quantization is now standard in vLLM and TensorRT-LLM (halving the cache bytes), and why paged attention was worth a paper. A useful mental anchor: for a 7B model, ~512 KB of KV per token; for a 70B with GQA, far less per token but still the dominant variable cost at long context. When someone asks "how many concurrent users can this box serve," the answer is almost always "(usable HBM − weights) ÷ (KV bytes per request at your context length)," and that arithmetic is the [KV-cache management problem](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) in one line.

**Speculative decoding in production.** Google reported ~2–3× decode speedups with speculative decoding (Leviathan et al. 2023); the technique now ships in vLLM, TensorRT-LLM, and the major inference stacks. Medusa (self-speculation with extra heads) reports ~2–3× on conversational workloads without a separate draft model. Prompt-lookup / n-gram speculation reports large speedups (sometimes >3×) on input-grounded tasks like summarization and retrieval-augmented generation, where the model frequently quotes its context. The common thread: the speedup tracks the acceptance rate, and the acceptance rate tracks how predictable the output is and how well the draft matches the target.

**Tensor-parallel serving for large models.** Serving a 70B model essentially *requires* TP (it does not fit on one 80 GB GPU). With TP=4 or TP=8 inside an NVLink node, decode latency stays reasonable because the all-reduces overlap with compute on the fast fabric. The reported MFU and latency are good *inside* the node and degrade sharply if you try to extend TP across an InfiniBand boundary — which is the empirical basis for "keep TP within the NVLink domain." For the production serving stack as a whole — load balancing, autoscaling, multi-replica routing — see [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) and the broader survey of [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques).

Here is a consolidated before→after table for a single 7B model on one A100 80GB, going from a naive loop to a tuned vLLM server. These are representative magnitudes, not a benchmark you should quote verbatim — your numbers depend on prompt/response length distribution:

| Serving setup | Batch (effective) | Aggregate throughput | p50 inter-token | GPU util | KV waste |
|---|---|---|---|---|---|
| Naive `model.generate` loop, batch 1 | 1 | ~900 tok/s | ~22 ms | ~12% | n/a |
| Static batching, contiguous KV | ~8 (padded) | ~3,500 tok/s | ~40 ms | ~35% | 60–80% |
| Continuous batching, contiguous KV | ~24 | ~9,000 tok/s | ~70 ms | ~70% | 60–80% |
| Continuous batching + paged attention (vLLM) | ~64 | ~14,000 tok/s | ~95 ms | ~85% | <4% |
| + speculative decoding (low batch, α≈0.75) | low batch | n/a (latency win) | ~9 ms effective | varies | <4% |

And a comparison of the techniques by what problem each solves and what it costs:

| Technique | Solves | Mechanism | Cost / when it backfires |
|---|---|---|---|
| Continuous batching | Idle slots, head-of-line blocking | Per-iteration scheduling, admit-on-free | Almost always worth it; tiny scheduler overhead |
| Paged attention | KV fragmentation, small batches | Fixed KV pages + block table | Slightly more complex attention kernel |
| Grouped-query attention | KV cache too big | Fewer KV heads than query heads | Built into the model; tiny quality cost |
| Tensor parallelism | Model or KV won't fit | Shard layers, all-reduce per layer | Backfires across slow interconnect (PCIe) |
| Disaggregation | Prefill stalls decode | Separate prefill/decode pools | Overkill on small deployments; KV transfer cost |
| Speculative decoding | Single-request latency floor | Draft proposes, target verifies in parallel | Backfires at low acceptance or high batch |

## How to read a serving roofline and pick a batch size for an SLO

This is the practical skill the whole post builds toward: given an SLO, pick the batch size. Here is the procedure I actually use.

First, locate your workload on the roofline. Prefill sits on the compute-bound (sloped-then-flat) side; decode sits deep in the memory-bound region. You confirm this with a profiler — `nsys profile` or `torch.profiler` to see where decode spends its time (it will be in the GEMM and attention kernels, dominated by HBM traffic). See [profiling GPU workloads](/blog/machine-learning/high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck) for the method; the key serving-specific habit is to measure TTFT and ITL *separately*, because they are governed by the two different rooflines and a single average latency number hides the prefill/decode split.

Second, sweep batch size against your SLO. Throughput rises with batch size until the KV cache caps it; latency rises monotonically. You want the largest batch whose p99 latency still clears the SLO. Measure it — do not guess — with warm-up passes and `torch.cuda.synchronize()` before timing, at steady state under realistic concurrency:

```python
import time, torch
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", dtype="bfloat16",
          gpu_memory_utilization=0.90, max_model_len=2048)
params = SamplingParams(temperature=0.0, max_tokens=128)

def measure(concurrency, prompt, reps=3):
    prompts = [prompt] * concurrency
    # Warm-up: let CUDA graphs / kernels compile and caches fill.
    llm.generate(prompts, params)
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        # Aggregate throughput and a crude per-request latency proxy.
        thr = toks / dt
        lat_ms = dt / (params.max_tokens) * 1000   # ms per output token at this concurrency
        best = (thr, lat_ms) if best is None else best
    return best

SLO_ITL_MS = 50.0   # p99 inter-token latency budget
prompt = "Explain the roofline model in two sentences."
for c in [1, 4, 8, 16, 32, 64, 128]:
    thr, itl = measure(c, prompt)
    ok = "OK" if itl <= SLO_ITL_MS else "SLO BREACH"
    print(f"concurrency={c:4d}  throughput={thr:8.0f} tok/s  itl~={itl:6.1f} ms  {ok}")
```

```bash
# Run it and read the table -- pick the largest concurrency that still prints OK:
python slo_sweep.py
# concurrency=   1  throughput=     900 tok/s  itl~=  22.0 ms  OK
# concurrency=   8  throughput=    3500 tok/s  itl~=  31.0 ms  OK
# concurrency=  32  throughput=   11000 tok/s  itl~=  44.0 ms  OK   <- pick here
# concurrency=  64  throughput=   14000 tok/s  itl~=  95.0 ms  SLO BREACH
```

Third, read the result. In this sweep the SLO of 50 ms inter-token latency is met up to concurrency 32 (throughput ~11,000 tok/s) and breached at 64. So you set `max_num_seqs` to cap the batch near 32, accept ~11,000 tok/s as your per-GPU throughput, and *then* compute how many GPU replicas you need to handle your total request rate. The discipline here is that **batch size is an SLO-derived number, not a number you maximize.** You find the knee of the curve where latency is about to blow the budget and you sit just under it. If you need more total throughput, add replicas (data parallelism) rather than bigger batches — replicas raise throughput without touching per-request latency, whereas bigger batches always raise latency. And if you need *lower* latency than even batch 1 gives you, that is the cue for speculative decoding or a smaller/quantized model — there is no batching answer to a single-request latency floor.

Fourth, watch for the KV-cache wall during the sweep. As you raise concurrency, at some point vLLM will start *queuing* requests (it cannot admit them because the KV pool is full) rather than batching them, and your throughput curve will flatten or the engine will report preemptions. That flat spot is the KV-cache wall from section 3, and it tells you the limit is memory, not compute — at which point your levers are GQA, fp8 KV quantization, a longer-context model with cheaper KV, or tensor parallelism to spread the cache across GPUs. The serving roofline and the KV-cache budget together tell you *which* lever, and that is the entire point of reading them.

## When to reach for each technique (and when not to)

Serving optimizations are not a checklist you apply blindly; each is a cost, and several backfire when misapplied. Here is the decisive guidance.

**Always use continuous batching.** There is essentially no workload where per-iteration scheduling loses to static batching. If your serving stack does not do it, switch stacks (vLLM, TGI, TensorRT-LLM, SGLang all do). This is the one unconditional recommendation in the post.

**Use paged attention whenever you batch.** It is the enabler for large batches; without it fragmentation caps your concurrency. In practice you get it for free by using a modern engine — you would only hand-roll contiguous KV if you were building a serving stack from scratch, and you should not.

**Use a GQA model and consider fp8 KV cache** when KV memory is your binding constraint (it usually is at long context). GQA is a model-architecture choice you make at training/selection time; fp8 KV quantization is a serving-time flag. Both directly raise the concurrent-request count.

**Reach for tensor parallelism only when the model or its KV cache won't fit on one GPU**, or when you need lower single-request latency than one GPU delivers, *and* you have a fast interconnect (NVLink/NVSwitch). Do not add TP to a model that fits just to "scale" — replicas scale throughput better and simpler. And never extend TP across a slow PCIe or cross-node InfiniBand boundary for decode; the per-token all-reduce will dominate and you will make latency worse.

**Reach for disaggregation only at fleet scale** with enough traffic to keep dedicated prefill and decode pools busy and a fast fabric to ship the KV cache between them. On a single box or a small deployment, chunked prefill on a unified server is simpler and nearly as good. Disaggregation is a large-deployment architecture, not a default.

**Reach for speculative decoding when single-request latency is the problem and the batch is low-to-moderate.** It buys back the latency floor by spending idle compute. Do not enable it on a compute-saturated, high-batch throughput box — the verification work competes with the batch for compute and can reduce throughput. And it only pays with a cheap, well-aligned draft (high acceptance rate); a mismatched draft makes you slower. Prompt-lookup speculation is the safest first try for input-grounded tasks.

The meta-rule, true across this whole series: **measure first, then pick the lever the roofline points at.** A serving box is memory-bound on the KV cache far more often than it is compute-bound, so the highest-leverage moves are almost always the ones that shrink or better-manage the KV cache (paging, GQA, fp8, disaggregation) and the ones that keep the GPU full (continuous batching), not the ones that add raw compute. If you find yourself adding GPUs and getting sublinear returns, you are probably fighting the memory wall, and the answer is a memory optimization, not more silicon.

## Key takeaways

- **Serving is a different HPC problem from training.** Training optimizes throughput on a fixed graph; serving trades throughput against per-request latency under an SLO, with dynamic, variable-length requests each holding a growing KV cache.
- **Prefill is compute-bound; decode is memory-bound.** Prefill processes the whole prompt in parallel (high arithmetic intensity, saturates Tensor Cores). Decode produces one token per step (intensity ~1, starved on HBM bandwidth). This split explains every technique in the post.
- **The KV cache is the binding constraint.** Its size is $2 \cdot L \cdot d_\text{model} \cdot S \cdot B \cdot \text{bytes}$ — for a 7B model, ~512 KB per token. It scales with batch × sequence and overtakes the fixed weights at long context, capping how many requests you can serve.
- **Continuous batching is the biggest free lunch.** Per-iteration scheduling (admit-on-free, evict-on-finish) keeps the GPU full of live work instead of idling on the longest request — a 10–20× swing in real throughput with no model change.
- **Paged attention kills fragmentation.** Fixed KV pages + a block table cut KV waste from 60–80% to <4%, multiplying batch size, and enable prefix sharing across common prompts.
- **Tensor parallelism is a capacity tool, not a throughput tool.** Use it when the model/KV won't fit or for lower latency, only inside a fast interconnect. For a model that fits, add replicas instead.
- **Disaggregation gives prefill and decode their own roofline-tuned pools** — a fleet-scale architecture that stops long prompts from stalling token streams; ~2–4× more served requests in the reported results.
- **Speculative decoding buys back the single-request latency floor** by spending idle decode compute; expected speedup = accepted tokens per target step (~2.4–3× at 75% acceptance). It is a latency tool for low batch, not a throughput tool for a saturated box.
- **Batch size is an SLO-derived number.** Sweep it, find the largest batch whose p99 latency clears the SLO, sit just under the knee, and scale total throughput with replicas — not with bigger batches that always inflate latency.

## Further reading

- Kwon et al., **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (2023) — the vLLM/PagedAttention paper. The KV-fragmentation analysis and the paging design.
- Yu et al., **"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (OSDI 2022) — iteration-level (continuous/in-flight) scheduling.
- Leviathan et al., **"Fast Inference from Transformers via Speculative Decoding"** (2023) and Chen et al., **"Accelerating Large Language Model Decoding with Speculative Sampling"** (2023) — the two foundational speculative-decoding papers, including the distribution-preserving proof.
- Zhong et al., **"DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving"** (2024) — the disaggregation architecture and its measured gains.
- Shazeer, **"Fast Transformer Decoding: One Write-Head is All You Need"** (2019) and Ainslie et al., **"GQA"** (2023) — multi-query and grouped-query attention, the KV-cache-shrinking architecture choices.
- NVIDIA A100 and H100 architecture whitepapers — the peak bf16 TFLOP/s and HBM bandwidth numbers behind every roofline in this post.
- Within this series: [the roofline model](/blog/machine-learning/high-performance-computing/the-roofline-model-compute-bound-vs-memory-bound), [kernel fusion and FlashAttention](/blog/machine-learning/high-performance-computing/kernel-fusion-and-flashattention-beating-the-memory-wall), [parallelism strategies](/blog/machine-learning/high-performance-computing/parallelism-strategies-data-tensor-pipeline-and-expert), [numerical formats and mixed precision](/blog/machine-learning/high-performance-computing/numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8), and the capstone [HPC playbook for AI engineers](/blog/machine-learning/high-performance-computing/the-hpc-playbook-for-ai-engineers).
- Out of series: [KV-cache optimization and management](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management), [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems), and [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques).
