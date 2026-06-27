---
title: "Fast Rollouts With vLLM: The Generation Bottleneck in RLHF"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why generation eats most of your RLHF wall-clock time, how PagedAttention and continuous batching fix it, and how to wire vLLM and SGLang into a rollout loop with per-token log-probs and live weight sync."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "rlhf",
    "vllm",
    "llm-alignment",
    "machine-learning",
    "pytorch",
    "inference-optimization",
    "ppo",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 67
image: "/imgs/blogs/vllm-async-rollout-collection-1.png"
---

The first time I profiled an RLHF run, I expected the gradient step to be the villain. I had eight A100s, a 7B policy, a 7B reward model, and a PPO loop straight out of TRL. The training was healthy — KL stayed bounded, the reward curve climbed. But each iteration took ninety seconds, and the GPUs were mostly *idle*. When I dropped a timer into the loop, the answer was embarrassing: about seventy of those ninety seconds were spent in `model.generate()`. The backward pass — the part everyone thinks of as "the expensive part of deep learning" — was a rounding error next to the cost of having the model *talk*.

That is the dirty secret of reinforcement learning from human feedback, and of every modern RL-on-LLMs method that came after it (PPO, GRPO, RLOO, RLAIF). The learning is cheap. The *rollout* — generating the responses the policy will be graded on — is where your cluster goes to wait. If you do nothing about it, you will spend 60–80% of your wall-clock time autoregressively decoding tokens one at a time, on hardware built for massively parallel matrix multiplies, doing the one thing that hardware is worst at.

This post is about fixing that. We will start from *why* generation is slow in a way training is not — it is a memory-bandwidth problem, not a compute problem — and then build up the two ideas that made vLLM 2–24× faster than naive generation: **PagedAttention** for the KV cache, and **continuous batching** for the scheduler. We will wire vLLM into an actual RLHF rollout loop, extract the per-token log-probabilities PPO needs, solve the weight-sync problem (the policy keeps changing; the inference engine needs the new weights), and look at SGLang, tensor parallelism, and speculative decoding as further multipliers. By the end you will be able to read a rollout profile, pick the right engine, and cut a number like ninety seconds down to fifteen.

![A side by side comparison of slow HuggingFace generate against a fast vLLM rollout engine showing throughput climbing from 200 to 2000 tokens per second](/imgs/blogs/vllm-async-rollout-collection-1.png)

This sits inside the recurring frame of the series: an agent interacts with an environment, collects rewards, and updates a policy. In RLHF the "environment" is *generate a response, then score it with a reward model*. The rollout is the agent acting. Everything here is about making the acting fast so the learning is not starved. If you want the wider map of where this fits, the unified taxonomy post `reinforcement-learning-a-unified-map` places RLHF among the policy-gradient family, and the capstone `the-reinforcement-learning-playbook` ties the rollout/learning split together with the rest of the toolbox.

## 1. Why generation is the bottleneck

Let us be precise about *why* training parallelizes and generation does not, because the asymmetry is the whole story.

A training step on a batch of token sequences is one giant feed-forward and one giant backward pass. Every token in every sequence is known in advance — they are the ground-truth or already-sampled tokens — so the model processes the entire batch in parallel. A transformer layer is a stack of matrix multiplies, and matrix multiplies on a GPU are *compute-bound*: the chip has thousands of multiply-accumulate units, you feed them a big matrix, and they chew through it at close to peak FLOPs. A modern GPU can sustain hundreds of teraFLOPs on a well-shaped matmul. Training a 7B model on a batch of 1,024 sequences of length 512 is, from the hardware's point of view, a happy workload.

Autoregressive generation is the opposite workload. To produce token $t$, the model must have already produced token $t-1$, because token $t$'s logits depend on the full prefix. So generation is a *sequential* loop: forward pass, sample one token, append it, forward pass again. You cannot parallelize across the time dimension of a single sequence — that is the definition of autoregression.

Worse, each of those forward passes is *tiny* in the dimension the GPU cares about. When you generate the next token for a batch of $B$ sequences, you are doing a forward pass on exactly $B$ tokens (one new token per sequence). The matrices are skinny: $B \times d_{\text{model}}$ times $d_{\text{model}} \times d_{\text{model}}$. The GPU finishes the arithmetic almost instantly and then sits waiting for the *weights* to arrive from memory. This is the crux: decoding is **memory-bandwidth-bound**, not compute-bound. The bottleneck is the speed at which you can stream the model's weights and KV cache out of HBM, not the speed of the arithmetic.

You can see this in a back-of-envelope arithmetic-intensity argument. For a single decode step the model reads roughly all of its parameters once. A 7B model in bfloat16 is about 14 GB of weights. An A100 has roughly 2 TB/s of memory bandwidth. So one decode step is bounded below by $14 / 2000 \approx 7$ ms just to *read the weights*, regardless of how few tokens you are decoding. The arithmetic for one token against 7B params is about $2 \times 7\text{B} = 14$ GFLOPs — which at 300 TFLOPs takes about 0.05 ms. The compute is 140× faster than the memory access. The hardware spends 99% of the decode step waiting on memory.

The fix is not "make the arithmetic faster." The fix is *amortize the weight read across many tokens*. If a single decode step costs 7 ms whether you decode 1 token or 256 tokens (because reading the 14 GB of weights dominates), then decoding 256 tokens in one batched step gives you 256 tokens for the price of one weight read. That is the entire economic argument for batching during generation, and it is why a serving engine that keeps the decode batch large can hit ten times the throughput of one that processes sequences one at a time.

There is a useful number that crystallizes this: **arithmetic intensity**, the ratio of FLOPs performed to bytes moved. A GPU has a *roofline* — a ridge point where the workload transitions from memory-bound to compute-bound. For an A100, that ridge is around $300\,\text{TFLOPs} / 2\,\text{TB/s} \approx 150$ FLOPs per byte. Below that intensity you are memory-bound; above it you are compute-bound. A single-token decode step has an arithmetic intensity of roughly $2$ FLOPs per parameter-byte (two FLOPs per multiply-accumulate, one byte-ish per bf16 weight read) — about $1$, two orders of magnitude *below* the ridge. So decoding sits deep in the memory-bound regime. Every token you add to the batch raises the intensity (more FLOPs over the same weight read) and walks you up the roofline toward the compute-bound regime where the chip is actually busy. The practical implication for rollouts: you want the decode batch as large as memory allows, because you are climbing the roofline with every sequence you add, right up until you saturate compute (or run out of KV memory, which is usually the binding constraint first — hence Section 3).

Grouped-query attention (GQA), used by Llama-3, Mistral, and most recent models, changes the constants but not the story. GQA shares KV heads across groups of query heads, shrinking the KV cache (and therefore the per-token memory traffic for the cache) by the grouping factor — often 4× or 8×. That makes the *weights* an even larger share of the memory read, which means batching matters even more, and it means you fit more sequences in KV memory, which is exactly what you want for high-throughput rollouts. None of this changes the fundamental asymmetry: training stays compute-bound and parallel; decoding stays memory-bound and sequential. The engine's whole job is to claw back as much of that lost utilization as it can.

#### Worked example: where the ninety seconds go

Take my original run. 7B policy, batch of 1,024 prompts per PPO iteration, average response length 256 tokens. Naive HuggingFace `generate()` on 8 GPUs, no batching cleverness, gave me about 200 tokens/second/GPU on the decode loop. Total tokens to generate per iteration: $1024 \times 256 = 262{,}144$. Across 8 GPUs at 200 tok/s each that is $1600$ tok/s aggregate, so $262{,}144 / 1600 \approx 164$ seconds — except the prompts were short and I overlapped a little, so the real number landed around 70 s of pure generation. The PPO backward pass on the collected experience? About 12 s. Reward scoring, another 6 s. So 70 out of 88 seconds — 80% — was generation. Swapping to a vLLM engine pushed decode to roughly 2,000 tok/s/GPU and the generation phase dropped to about 13 s. Same algorithm, same hyperparameters, same reward — a 5–6× wall-clock speedup on the iteration, purely from how the tokens get produced.

That is the prize. Now let us earn it.

## 2. vLLM architecture: what actually makes it fast

vLLM is a serving engine that came out of Berkeley in 2023, built around one insight that turned out to be worth an order of magnitude: **the KV cache is managed badly by every naive implementation, and managing it like an operating system manages memory fixes everything.** The two pillars are PagedAttention (memory) and continuous batching (scheduling). Stacked, they take generation from "GPU mostly idle" to "GPU mostly busy."

It is worth seeing where vLLM sits in a lineage of inference optimizations, because a fast rollout today inherits the whole stack. Each milestone removed a different bottleneck, and RLHF rollouts get all of the gains for free by using a modern engine.

![A timeline of inference optimization milestones from FlashAttention in 2022 through PagedAttention, continuous batching, RadixAttention, speculative decoding, and the vLLM weight update API in 2024](/imgs/blogs/vllm-async-rollout-collection-6.png)

The high-level shape: a vLLM engine owns a pool of GPU memory carved into fixed-size *blocks*. A scheduler decides, every iteration, which requests get to run a decode step this iteration based on how many free blocks exist. A block manager hands out and reclaims blocks as sequences grow and finish. The attention kernel is written to read KV from non-contiguous blocks via an indirection table. Around all of that sits an API — a synchronous `LLM` class for offline batch jobs, and an `AsyncLLMEngine` for streaming online serving.

The throughput numbers vLLM reported in its paper and that the community has reproduced are in the 2–4× range over the best prior serving systems and up to 24× over a naive HuggingFace pipeline that does static batching with full padding. The exact multiplier depends entirely on your workload: long shared prefixes, high concurrency, and variable response lengths are where vLLM crushes the naive baseline, because those are exactly the cases where naive padding and static batching waste the most.

![A request lifecycle stack showing a request arriving, the scheduler queue, a parallel prefill pass, a continuous batched decode loop, detokenization, and the returned response with log probabilities](/imgs/blogs/vllm-async-rollout-collection-3.png)

Notice the lifecycle has two distinct compute phases. **Prefill** processes the whole prompt in one parallel forward pass — this is compute-bound and fast, like training. **Decode** is the autoregressive loop, memory-bound and slow, one token per sequence per iteration. The two phases have completely different performance characteristics, which is why later optimizations (chunked prefill, prefill/decode disaggregation) treat them separately. For rollouts the decode phase dominates, because RLHF responses are long relative to prompts.

## 3. PagedAttention: KV cache as virtual memory

Here is the problem PagedAttention solves. During generation, the model caches the key and value tensors of every past token so it does not recompute them — that is the KV cache, and it is the memory hog of inference. For one sequence the cache size is:

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L \times \text{dtype\_bytes}
$$

The factor of 2 is for K and V; $L$ is the sequence length so far. The cache *grows by one token's worth every decode step*, and you do not know in advance how long the sequence will be.

The naive solution — the one HuggingFace `generate()` uses with static batching — is to pre-allocate the cache for the *maximum* possible length, for every sequence in the batch, up front. If `max_seq_len` is 2,048 but your average response is 256 tokens, you have allocated 8× more KV memory than you use. That over-allocation is *internal fragmentation*. And because the buffer is one contiguous tensor per sequence, two sequences with an identical prompt prefix each store their own copy of that prefix's KV — *no sharing*. The vLLM paper measured that naive systems waste 60–80% of KV memory to fragmentation and duplication. Wasted KV memory means fewer sequences fit on the GPU at once, which means a smaller decode batch, which means worse amortization of the weight read — straight back to the bottleneck from Section 1.

PagedAttention borrows the operating-system trick of *virtual memory with paging*. Instead of one contiguous KV buffer per sequence, the cache is split into fixed-size **blocks** (vLLM's default is 16 tokens per block). A sequence's logical token positions map, through a per-sequence **block table**, to physical blocks that can live *anywhere* in the pool and need not be contiguous. When a sequence needs to store its 17th token, the block manager hands it a fresh physical block and adds an entry to its block table. When a sequence finishes, its blocks return to the free pool for the next request.

![A graph showing two logical sequences mapping through a shared page table into non-contiguous physical pages with a shared prefix page and a free pool reclaiming finished blocks](/imgs/blogs/vllm-async-rollout-collection-2.png)

Three consequences fall out, and all three matter for rollouts:

1. **Near-zero internal waste.** A sequence allocates one block at a time as it grows, so the only waste is the unused slots in its *last* partially-filled block — at most 15 tokens, a fraction of a percent for any real response. KV utilization jumps from ~30% to ~96%.
2. **Prefix sharing.** If many sequences share a prompt prefix (extremely common in RLHF — you often sample several responses per prompt for GRPO or best-of-n), their block tables can *point at the same physical blocks* for the shared prefix. The KV for that prefix is stored once and read by all of them. This is the seed of SGLang's RadixAttention, which we will get to.
3. **More concurrent sequences.** Because each sequence uses only the memory it needs, far more sequences fit on the GPU simultaneously, which means a bigger decode batch, which means the weight read amortizes over more tokens.

The attention kernel has to be rewritten to handle this: instead of reading a contiguous KV tensor, it walks the block table and gathers KV from scattered physical blocks. That is the "PagedAttention" CUDA kernel. The indirection costs a little, but the memory savings buy back an order of magnitude in batch size, so it is a wild net win.

There is a fourth consequence that is pure gift for RLHF: **copy-on-write for the shared prefix.** When you sample $k$ responses from the same prompt (GRPO, best-of-n), all $k$ sequences start by pointing their block tables at the *same* physical blocks holding the prompt's KV. They only allocate fresh blocks once they start diverging — that is, once they generate their first different token. This is the exact analogue of `fork()` in an operating system: the child shares the parent's pages until it writes, at which point that page is copied. For an RLHF run that samples 8 responses per prompt with a 512-token shared prompt, copy-on-write means the prompt's KV is stored *once* instead of eight times, and the prefill of that prompt happens *once* instead of eight times. That is a 7/8 reduction in both KV memory and prefill compute for the shared portion — and it requires zero work from you beyond enabling prefix caching. It is also the conceptual bridge to SGLang's RadixAttention, which generalizes this from "explicitly batched identical prompts" to "any prefix that happens to already be in the cache."

The block size is a real tuning knob, even if the default of 16 is good. Smaller blocks (say 8) cut the last-block internal waste in half but double the number of block-table entries and the per-step indirection overhead. Larger blocks (32) reduce indirection but waste more of the last partial block. For the long, variable-length responses typical of RLHF, 16 is a sound default; I have rarely seen moving it pay off, but it is there if a profile points at block-table overhead.

#### Worked example: PagedAttention memory savings on a 7B rollout

Take Llama-2-7B: 32 layers, 32 heads, head dim 128, bf16 (2 bytes). Per token the KV cache is $2 \times 32 \times 32 \times 128 \times 2 = 1{,}048{,}576$ bytes — exactly 1 MB per token. Suppose you run rollouts with `max_seq_len = 2048` but the actual prompt+response averages 320 tokens.

Naive static allocation reserves $2048 \times 1\text{ MB} = 2$ GB of KV *per sequence*. An A100 with 80 GB, after the 14 GB of weights, has roughly 60 GB for KV, so you fit $60 / 2 = 30$ sequences in the batch. PagedAttention allocates only the ~320 tokens actually used, rounding up to 20 blocks of 16 tokens = 320 tokens = 320 MB per sequence. Now you fit $60 / 0.32 \approx 187$ sequences. That is a **6× larger decode batch** from the same 60 GB. Since decode throughput scales roughly with batch size until you hit the compute ceiling, that 6× batch translates to a large multiple on tokens/second — and it is *free*, no quality change, just better bookkeeping. (If your model uses grouped-query attention, e.g. Llama-3 with 8 KV heads instead of 32, the per-token KV shrinks 4× and you fit even more.)

## 4. Continuous batching: iteration-level scheduling

PagedAttention fixes *memory*. Continuous batching fixes *scheduling*, and the two compound.

The naive batching strategy — *static batching* or *request-level batching* — collects a batch of requests, runs them all to completion, then starts the next batch. The problem is that requests in a batch finish at wildly different times. One prompt produces a 12-token answer; another rambles for 400 tokens. With static batching, the 12-token sequence finishes and then *sits there as dead weight*, occupying a slot in the batch and contributing nothing, while the kernel keeps running decode steps for the 400-token sequence. The whole batch is held hostage by its longest member. Effective GPU utilization with static batching on variable-length workloads is often around 50% or worse.

Continuous batching (also called *iteration-level scheduling* or *dynamic batching*, popularized by the Orca paper and built into vLLM) operates at the granularity of a single decode iteration rather than a whole request. After *every* decode step, the scheduler checks: did any sequence emit an end-of-sequence token or hit its length limit? If so, evict it immediately, free its blocks, and admit a waiting request into the freed slot. The batch is continuously refilled. No sequence ever waits for an unrelated sequence to finish.

This is exactly the bin-packing problem the OS scheduler solves, and it lifts utilization from ~50% to ~85% or higher on realistic mixed-length traffic. Combined with PagedAttention's larger batches, you get the headline 2–24× over naive serving.

For RLHF rollouts this matters enormously because response lengths are *extremely* variable — that variability is partly what you are training. Early in training the policy produces short, low-reward responses; later it produces longer, structured ones. A static batch would be dominated by length variance the whole time. Continuous batching just absorbs it.

There is a subtlety worth flagging for the rollout use case. Continuous batching is built for *online serving*, where requests arrive over time. In RLHF you submit a whole batch of rollout prompts at once (an *offline* batch job). vLLM's `LLM.generate()` handles this beautifully: it internally schedules your big list of prompts with continuous batching, admitting and evicting as the engine churns through them. You hand it 1,024 prompts; it keeps the GPU packed until all 1,024 are done. You do not have to manage the scheduling — that is the point.

#### Worked example: static vs continuous batching on skewed lengths

Suppose you generate a batch of 64 responses whose lengths are heavily skewed: 60 of them finish in 50 tokens, 4 of them ramble to 500 tokens (a realistic shape early in RLHF, when most outputs are short but a few degenerate into repetition). With **static batching**, the batch runs for 500 decode steps — the length of the longest member — and for the last 450 of those steps, 60 of the 64 slots are dead weight, generating padding or simply idling. Total useful tokens: $60 \times 50 + 4 \times 500 = 5{,}000$. Total slot-steps spent: $64 \times 500 = 32{,}000$. Useful fraction: $5{,}000 / 32{,}000 \approx 16\%$ — the GPU spent 84% of its decode budget on dead slots. With **continuous batching**, the moment one of the 60 short sequences finishes at step 50, its slot is freed and either a waiting request fills it or the batch shrinks; no slot idles waiting for the 500-token stragglers. The useful fraction climbs toward 80–90%, and on a fixed token budget you finish the batch roughly 4–5× faster. This is not a contrived example — RLHF response-length distributions are genuinely this skewed, which is exactly why continuous batching is non-negotiable for rollouts.

The scheduler also has to decide *prefill versus decode* every iteration, and this is where chunked prefill (Section 11) earns its keep. When a fresh request arrives, its prompt must be prefilled — a compute-heavy parallel pass — and naively running that prefill blocks the decode steps of all the in-flight sequences for that iteration, causing a latency spike. Chunked prefill slices the prompt into pieces and interleaves them with ongoing decode steps so neither phase starves the other. For rollouts with long prompts (system prompt + few-shot examples + the actual query) this smoothing is the difference between a jagged throughput curve and a flat, high one.

## 5. Using vLLM as a rollout engine

Now the practical part. Here is the minimal offline rollout with vLLM, including the thing RLHF actually needs that a chatbot does not: **per-token log-probabilities**. PPO's importance-sampling ratio is $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$, and $\pi_{\theta_{\text{old}}}$ is exactly the log-prob the rollout policy assigned to each token it sampled. You must capture it at generation time.

```python
from vllm import LLM, SamplingParams

# Load the policy as a rollout engine. gpu_memory_utilization leaves
# headroom; for co-located training+inference you tune this down.
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    gpu_memory_utilization=0.45,   # leave room for the trainer copy
    max_model_len=2048,
)

# logprobs=0 means "give me the logprob of the *sampled* token" only.
# (logprobs=k would also return the top-k alternatives per position.)
sampling = SamplingParams(
    temperature=1.0,        # RLHF samples, never greedy
    top_p=1.0,
    max_tokens=256,
    logprobs=0,
    seed=None,              # do NOT fix the seed: you want diverse rollouts
)

prompts = [build_prompt(x) for x in prompt_batch]   # list[str]
outputs = llm.generate(prompts, sampling)

rollouts = []
for out in outputs:
    comp = out.outputs[0]
    token_ids = comp.token_ids                       # list[int], the response
    # comp.logprobs is a list (one dict per generated position).
    # Each dict maps token_id -> Logprob; we want the sampled token's logprob.
    old_logprobs = [
        lp[tid].logprob for tid, lp in zip(token_ids, comp.logprobs)
    ]
    rollouts.append({
        "prompt": out.prompt,
        "prompt_token_ids": out.prompt_token_ids,
        "response_token_ids": token_ids,
        "old_logprobs": old_logprobs,                # for the PPO ratio
        "text": comp.text,
    })
```

A few things that bite people the first time. First, `temperature` must be positive and you must *not* fix the seed across iterations — RLHF needs stochastic, diverse samples, otherwise the policy gradient sees no variation to learn from. Second, `comp.logprobs[i]` is a dict keyed by token id, so you index it with the *sampled* token id `token_ids[i]` to get back that token's log-prob; this is the on-policy log-prob $\log \pi_{\theta_{\text{old}}}$. Third, these log-probs come from the *bf16 inference forward pass*, which can differ slightly from the trainer's fp32/bf16 forward pass — that numerical mismatch is a real source of RLHF instability, and we will return to it in the weight-sync section, because it is exactly why "the rollout engine and the trainer disagree about $\log \pi$" can blow up your KL.

For streaming and concurrency — say you want to overlap generation with reward scoring — use the async engine:

```python
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(model="meta-llama/Llama-2-7b-hf",
                    dtype="bfloat16",
                    gpu_memory_utilization=0.45)
)
sampling = SamplingParams(temperature=1.0, max_tokens=256, logprobs=0)

async def rollout_one(prompt: str, req_id: str):
    final = None
    async for out in engine.generate(prompt, sampling, req_id):
        final = out                      # last yielded is the complete output
    comp = final.outputs[0]
    lps = [lp[t].logprob for t, lp in zip(comp.token_ids, comp.logprobs)]
    return comp.token_ids, lps

async def rollout_batch(prompts):
    tasks = [rollout_one(p, f"req-{i}") for i, p in enumerate(prompts)]
    return await asyncio.gather(*tasks)   # engine interleaves them internally
```

The async engine lets you fire many `generate` calls and the engine's continuous-batching scheduler interleaves them on the GPU. This is the foundation for the *async RLHF* architecture in Section 8, where rollout workers stream experience into a buffer while the trainer drains it.

One design decision deserves more than a passing mention: **do you trust the rollout engine's log-probs, or do you recompute them?** vLLM hands you $\log\pi_{\theta_{\text{old}}}$ for free at generation time, and it is tempting to use that directly as the PPO baseline. Many production stacks instead do a *separate forward pass* — either in the trainer or in a dedicated "log-prob" pass — to recompute $\log\pi_{\theta_{\text{old}}}$ over the generated tokens with the *same* kernel and precision the trainer will use for $\log\pi_\theta$. Why pay for an extra forward pass? Because the PPO ratio $\exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$ is only meaningful if numerator and denominator come from the *same* computation; mixing vLLM's bf16 generation log-probs with the trainer's log-probs introduces a systematic bias into the ratio that has nothing to do with the policy actually changing. The trade-off is concrete: trust-vLLM's-logprobs saves a forward pass per rollout but risks the mismatch from Section 6; recompute-in-trainer costs a forward pass (cheap — it is one parallel pass over known tokens, prefill-shaped, not a decode loop) but gives you a clean ratio. My default is to recompute, and to *also* log vLLM's reported log-probs so I can diff them and catch the mismatch when it appears. The recompute looks like this:

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def recompute_logprobs(model, prompt_ids, response_ids):
    # One parallel forward pass over prompt+response (prefill-shaped, fast).
    full = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
    logits = model(full).logits[0]                 # [T, vocab]
    logp = F.log_softmax(logits.float(), dim=-1)   # fp32 for stability
    # logits[t] predicts token t+1, so align to the response region.
    start = len(prompt_ids) - 1
    idx = response_ids.unsqueeze(-1)
    tok_logp = logp[start:start + len(response_ids)].gather(-1, idx)
    return tok_logp.squeeze(-1)                     # log pi over response
```

Note the off-by-one that bites everyone: `logits[t]` predicts token `t+1`, so the log-prob of the first *response* token comes from the logits at the *last prompt* position. Get this alignment wrong and your ratios are silently shifted by one token, which manifests as a slow, mysterious training degradation rather than an outright crash. Always sanity-check by confirming the recomputed log-probs are close to vLLM's reported ones for the same tokens; if they are wildly off, you have an alignment bug, not a precision issue.

## 6. The weight sync problem

Here is the problem that makes RLHF rollouts harder than ordinary serving, and the one that trips up everyone wiring this together for the first time. In serving, the model weights are *fixed*. In RLHF, the policy weights *change every iteration* — that is the whole point. After each PPO update, the trainer has new weights, and the vLLM engine is still holding the *old* ones. If you do nothing, your rollouts are generated by a stale policy: the importance ratios are computed against the wrong $\pi_{\theta_{\text{old}}}$, the KL estimate drifts, and training degrades or diverges.

So every iteration you must **sync the updated weights from the trainer into the vLLM engine.** There are three architectures for this, and the right one depends on your memory budget and how much asynchrony you tolerate.

![An RLHF rollout pipeline showing prompts sampled, vLLM generating responses and log probabilities, a reward model scoring, advantages computed, a PPO update, and weights synced back to vLLM](/imgs/blogs/vllm-async-rollout-collection-4.png)

**Architecture A — co-located, sleep/wake.** The trainer and the vLLM engine live on the *same* GPUs. vLLM v0.4+ exposes a "sleep" capability (`enable_sleep_mode=True`, then `llm.sleep()` / `llm.wake_up()`) that lets the engine *release its KV cache and optionally offload weights* so the trainer can use that memory for the backward pass, then reload for the next rollout phase. The flow per iteration: wake vLLM → generate rollouts → sleep vLLM (free the KV/inference memory) → run the PPO update on the freed memory → write the new weights into vLLM's parameters → wake again. The big win is that you do not pay for two model copies *simultaneously*; the cost is the time to swap memory in and out, and the careful dance of `gpu_memory_utilization` so both phases fit. This is what TRL's newer vLLM integration and verl's co-located mode do. Concretely:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf", dtype="bfloat16",
          enable_sleep_mode=True, gpu_memory_utilization=0.85)
sampling = SamplingParams(temperature=1.0, max_tokens=256, logprobs=0)

for iteration in range(num_iters):
    llm.wake_up()                              # reclaim KV/weights memory
    outputs = llm.generate(prompt_batch, sampling)
    experience = build_experience(outputs)     # logprobs, rewards, advantages
    llm.sleep(level=1)                         # free inference memory

    # Now the freed GPU memory belongs to the trainer:
    ppo_trainer.step(experience)               # backward + optimizer.step
    new_state = ppo_trainer.policy.state_dict()

    llm.wake_up()                              # bring the engine back
    llm.llm_engine.model_executor.collective_rpc(
        "load_weights", args=(list(new_state.items()),))
    llm.llm_engine.reset_prefix_cache()
```

`sleep(level=1)` discards the KV cache and offloads weights to CPU (fast to reload); a deeper sleep can free even more at the cost of a slower wake. The numbers that make this work: a 7B policy with Adam in fp32-master needs roughly 70–80 GB on its own, and vLLM at `gpu_memory_utilization=0.85` wants most of the card during rollout — they cannot both be resident on one 80 GB GPU, so the sleep/wake swap is what makes single-node co-location feasible at all. The cost you pay is the offload/reload latency each iteration, typically a few seconds, which is cheap next to the rollout you just saved.

**Architecture B — co-located, both resident.** Keep both the training copy (with optimizer states) and the vLLM inference copy resident at once. After each update, copy the new parameter tensors directly into vLLM's model — vLLM exposes a way to do this without a full reload, e.g. through a collective `update_weights` call or by directly writing into the model's parameters via `llm.llm_engine.model_executor`. This is fast (no reload) but expensive in memory: you are paying for weights twice plus optimizer state. For a 7B model with Adam (fp32 params + two moment buffers + bf16 inference copy) you are looking at roughly $7\text{B} \times (4 + 4 + 4) + 7\text{B} \times 2 \approx 98$ GB just for the model and optimizer — too much for one 80 GB GPU, so you shard with FSDP/DeepSpeed and the vLLM copy is tensor-parallel alongside.

**Architecture C — disaggregated, NCCL broadcast.** The trainer runs on one set of GPUs, the vLLM rollout workers on *another* set. After each update, the trainer broadcasts the updated weights over a NCCL communicator into the rollout workers. This is the scalable architecture — you can have many rollout replicas — and it is what the async RLHF systems use. The cost is the network broadcast of 14 GB (7B bf16) per iteration; over NVLink/InfiniBand this is fast, but it is non-zero, and if your update is cheap you can become *broadcast-bound*.

A sketch of the weight-update side using vLLM's collective RPC interface (the exact API surface has shifted across versions, but the shape is stable):

```python
# Trainer side: after the PPO optimizer.step(), push new weights to vLLM.
def sync_weights_to_vllm(policy_model, llm):
    # Gather full (unsharded) params if using FSDP/DeepSpeed-ZeRO.
    state_dict = get_full_state_dict(policy_model)   # FSDP summon / ZeRO gather
    # vLLM exposes a way to load a state dict into the running engine's model.
    llm.llm_engine.model_executor.collective_rpc(
        "load_weights",
        args=(list(state_dict.items()),),
    )
    # Reset the prefix cache: old KV was computed with old weights.
    llm.llm_engine.reset_prefix_cache()
```

That last line is easy to forget and important. If you have prefix caching on (KV reuse across requests with shared prefixes), the cached KV was produced by the *old* weights. After a weight update it is stale and would corrupt the next generation. Reset it.

The deepest subtlety in weight sync is not mechanical — it is numerical. The trainer computes $\log \pi_\theta$ for the PPO loss using its own forward pass (often fp32 or a different attention kernel). The rollout engine computed $\log \pi_{\theta_{\text{old}}}$ using vLLM's bf16 kernels. Even with *identical weights*, these two numbers differ by small amounts because of kernel and precision differences. PPO's ratio $\exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$ is exponentially sensitive to that difference, so a 0.01 nat mismatch becomes a ~1% ratio error before you have learned anything. The practical mitigations are: (1) keep precision consistent where you can, (2) clip the ratio (PPO already does), and (3) if you see KL exploding right after switching to vLLM rollouts, suspect the log-prob mismatch before you suspect the algorithm. I lost a week to this once; the algorithm was fine, the engines just disagreed about $\log \pi$.

#### Worked example: how a tiny log-prob gap becomes a fake KL signal

Imagine the policy has *not changed at all* between rollout and the first PPO microbatch — you froze the update to debug. In a correct setup the ratio should be exactly 1.0 for every token and the surrogate gradient should be near zero. But vLLM's bf16 generation reports an average per-token log-prob of, say, $-1.840$, while the trainer's recompute reports $-1.832$ for the same tokens — a gap of just $0.008$ nats from kernel and precision differences. The ratio is $\exp(-1.832 - (-1.840)) = \exp(0.008) \approx 1.008$, so the importance weight is off by 0.8% on average, and the *estimated* KL between "old" and "new" policy reads as roughly $0.008$ nats per token even though the policy is byte-for-byte identical. Multiply across a 256-token response and your KL controller sees a phantom KL of ~2 nats and reacts by yanking the KL coefficient up, which then over-penalizes the *real* update on the next step, and the loop oscillates. The fix that made my run stable: recompute $\log\pi_{\theta_{\text{old}}}$ with the trainer's own forward pass (Section 5) so numerator and denominator share a kernel, making the frozen-policy ratio exactly 1.0 and the phantom KL zero. The lesson generalizes — whenever two different code paths compute the "same" log-prob, the difference between them masquerades as a policy change, and PPO's exponential ratio amplifies it.

### Weight synchronization, in mechanical detail

The three architectures above tell you *where* the weights live; this section is about *how* the bytes actually move, because the mechanics are where the time goes and where the bugs hide. The root of the difficulty is structural: vLLM keeps its **own copy of the model weights in GPU memory**, laid out for inference (often fused QKV projections, possibly a different sharding than the trainer's), and the training loop updates a **different copy** with a different layout. Synchronization is the act of reconciling those two copies after every optimizer step. It is not a `model.load_state_dict()` you can call once; it is a per-iteration data movement on the critical path, and how you do it sets a hard floor on your iteration time.

There are three mechanisms in practice, in increasing order of sophistication.

**(a) The `sleep_mode` API (vLLM 0.6+).** This is the lever that makes single-GPU co-location possible. `llm.sleep(level=1)` frees the KV cache outright and offloads the inference weights to CPU pinned memory; `llm.sleep(level=2)` discards the weights entirely (you reload them from the trainer on wake). `llm.wake_up()` brings the engine back. The point of `sleep` is not the weight update itself — it is *reclaiming the GPU memory* that the KV cache and inference weights occupied so the trainer's backward pass and optimizer states have room. The weight update happens *while the engine is awake but idle*, just before generation: you write the trainer's freshly-updated parameters into vLLM's parameter tensors. The sequence is `wake_up → load_weights → generate → sleep`, and the `sleep` is what gives the memory back to the trainer for its `step`.

**(b) `load_format` and weight reloading.** When the inference copy must be rebuilt from scratch — after a `sleep(level=2)`, or on a cold rollout worker — vLLM reloads weights through its model loader. The `load_format` argument controls the source: `"auto"` (the default) probes for safetensors then falls back to PyTorch `.bin`; `"dummy"` skips real weights entirely (for profiling the memory layout); and for live RLHF the relevant path is loading from an in-memory state dict rather than disk. A from-disk reload of a 7B model is *slow* — several seconds of file I/O plus the layout transform — which is why you avoid it on the hot path and prefer in-memory tensor passing.

**(c) Shared-memory tensor passing.** The fastest co-located sync writes the trainer's updated tensors **directly into vLLM's parameter storage** with no serialization and no disk. Because both copies are on the same node (often the same GPUs), you can hand vLLM a list of `(name, tensor)` pairs and have its model executor copy them in place via `collective_rpc("load_weights", ...)`. No NCCL broadcast, no file round-trip — just a device-to-device or host-to-device `copy_`. This is what the `sync_weights` method in Section 12 does, and it is the lowest-latency option when the trainer and engine share hardware.

**How OpenRLHF does it: Ray shared memory.** OpenRLHF's disaggregated design (Architecture C) places the trainer and the vLLM rollout workers in **separate Ray actors**, potentially on different nodes. After the optimizer step, the trainer gathers the full (unsharded) parameter tensors from its FSDP/DeepSpeed-ZeRO shards and broadcasts them to the vLLM workers over a dedicated **NCCL communicator** that OpenRLHF sets up between the trainer process group and the inference process group at startup. Within a node, Ray's object store uses shared memory (Plasma) so large tensors move without a copy through Python; across nodes the NCCL broadcast carries the bytes over InfiniBand/NVLink. The key engineering trick OpenRLHF contributed upstream is a `update_weight` hook on the vLLM worker that receives each tensor *by name* from the broadcast and writes it into the live model — so the trainer streams parameters one tensor at a time rather than materializing a full 14 GB state dict in one place, which would blow up memory on the gather.

**The overhead, measured.** What does a sync actually cost? For a 7B model in bf16 (14 GB of weights), the three mechanisms land in different places:

- *Shared-memory in-place copy (co-located):* dominated by the device-to-device `copy_` of 14 GB. At ~600 GB/s effective intra-GPU bandwidth for the copy plus Python/RPC overhead, this is well under a second — call it **0.5–1 s**.
- *NCCL broadcast (disaggregated):* 14 GB over NVLink (~300–600 GB/s achievable for a broadcast) or InfiniBand (~25 GB/s per link, more with multi-rail) lands at **0.5 s on NVLink, 2–4 s across nodes on IB**. The gather from FSDP shards adds to this.
- *Disk reload (`load_format="auto"` from safetensors):* file I/O plus layout transform pushes this to **5–10 s** for 7B — which is why it is a cold-start path, never a per-iteration one.

So the honest budget for weight sync on a 7B is roughly **2–5 seconds** in the disaggregated case that most production setups use — small next to a rollout phase, but not free, and it grows linearly with model size (a 70B sync is ~10× the bytes).

**The frequency tradeoff: sync every step vs every N steps.** You do not strictly *have* to sync after every single optimizer step. If you let the rollout engine run $N$ optimizer steps behind the trainer, you amortize the sync cost over $N$ updates — but you pay in *off-policy staleness*: the rollouts were generated by a policy $N$ steps stale, so the importance ratios are computed against a $\pi_{\theta_{\text{old}}}$ that is further from the current $\pi_\theta$, widening the ratio and stressing PPO's clipping. The sweet spot depends on your sync cost relative to your step cost. If a sync is 3 s and a step is 12 s, syncing every step adds 25% overhead — usually worth paying for on-policy freshness. If a sync is 4 s and a step is 1 s (a cheap GRPO update on a small model), syncing every step *quadruples* your iteration time, and batching the sync every 4–8 steps with importance correction is the better trade. The general rule: **sync as often as your sync-to-step cost ratio allows while keeping staleness within PPO's clip range** (empirically, a few steps of lag is tolerable; tens of steps starts to bite). This is exactly the staleness knob that the async architecture in Section 8 exposes — async RLHF is, in a sense, "sync every N steps" taken to its concurrent limit, with the trainer and rollout workers running on independent clocks and a bounded lag between them.

#### Worked example: the sync budget on a 7B disaggregated run

Take the async setup: 8 trainer GPUs (FSDP), 8 rollout GPUs (vLLM replicas), 7B policy. After each optimizer step the trainer gathers the 14 GB of bf16 params from its 8 FSDP shards — the all-gather over NVLink is ~0.3 s — then broadcasts to the rollout workers. Within the node over NVLink at ~400 GB/s effective, $14\,\text{GB} / 400\,\text{GB/s} \approx 0.035$ s of pure transfer, but with the per-tensor RPC dispatch and the `copy_` into each worker's model, call it ~1.5 s wall-clock. Add the FSDP gather and you are at ~2 s per sync. If the PPO step is 12 s and rollout is 13 s (overlapped in the async design), the 2 s sync is hidden inside the rollout phase — the trainer broadcasts while the next batch of rollouts is already generating on the *previous* weights, and the workers swap to the new weights at the next batch boundary. The sync becomes effectively free *because* it overlaps. Contrast the synchronous version: there the 2 s sync sits on the critical path between training and the next rollout, adding 2 s to every 25 s iteration — an 8% tax. The async design does not make the sync cheaper; it makes it *concurrent*, which is the whole point.

## 7. SGLang: RadixAttention and structured generation

vLLM is not the only rollout engine, and for some RLHF workloads SGLang is the better pick. SGLang (also out of Berkeley, 2024) keeps the paged-KV idea but adds two things vLLM did not originally have: **RadixAttention** and first-class **structured generation**.

**RadixAttention** generalizes vLLM's prefix sharing. Instead of sharing prefixes only when you explicitly batch identical prompts together, SGLang maintains a *radix tree* (a compressed prefix trie) over *all* the KV caches it currently holds. When a new request comes in, SGLang walks the radix tree to find the longest cached prefix that matches the request and *reuses that KV directly*, only computing the suffix. The cache is managed with LRU eviction over the tree.

Why does this matter for rollouts? Two patterns. First, **multi-sample-per-prompt** methods like GRPO, RLOO, and best-of-n generate $k$ responses for the *same* prompt. With RadixAttention the prompt's KV is computed once and shared across all $k$ samples automatically — no manual batching gymnastics. Second, **multi-turn or tool-use rollouts** where many trajectories share a long system prompt or a common conversation prefix: the shared prefix is cached once. On workloads with heavy prefix sharing, SGLang's published numbers show several-fold throughput gains over vLLM, precisely because it eliminates the redundant prefill of shared prefixes.

The mechanism is worth one more level of detail because it explains *when* the win materializes. A radix tree stores strings (here, token sequences) such that common prefixes share a path from the root, and a node is split only where sequences diverge. SGLang attaches the KV blocks for each token range to the tree edges. When request $R$ arrives, SGLang matches $R$'s tokens against the tree from the root, following the longest path that agrees; the KV for that matched prefix already exists and is reused verbatim, so SGLang only prefills the unmatched suffix. The cache is bounded, so least-recently-used subtrees are evicted under memory pressure — which means the win is largest when your *working set* of shared prefixes fits in cache. For GRPO with a few hundred distinct prompts in flight, each sampled $k$ times, the shared prompts stay hot and you get the full benefit. For a rollout stream with no prefix overlap at all (every prompt unique, no system prompt), RadixAttention degrades gracefully to ordinary paged KV with no penalty — you simply do not get the sharing bonus. So the question to ask before choosing SGLang for its RadixAttention is concrete: *does my rollout distribution actually share prefixes?* Multi-sample methods and shared system prompts say yes; a stream of unrelated single-sample prompts says no.

#### The RadixAttention algorithm, step by step

It is worth walking the algorithm precisely, because the data structure is the whole idea. A **radix tree** (a compressed prefix trie, also called a Patricia trie) is a trie in which every chain of single-child nodes is collapsed into one edge labeled by the whole substring. SGLang uses one keyed by *token sequences*: the root is the empty prefix, and each edge carries a run of token ids plus a pointer to the KV blocks that hold those tokens' keys and values. Two requests that begin with the same tokens follow the same path from the root until the first token where they differ; at that point the tree *splits* a node, and from there each request gets its own branch. The invariant the tree maintains is that **the KV for any given token-prefix is stored exactly once**, on the edge that represents it, no matter how many in-flight requests share it.

The lifecycle of one rollout request against the tree:

1. **Match.** When request $R$ arrives with token sequence $r_1, r_2, \dots, r_m$, SGLang walks the tree from the root, following edges whose token labels agree with $R$'s tokens, consuming as many of $R$'s leading tokens as match. This yields the **longest cached prefix** — say the first $j$ tokens — and a pointer to that prefix's already-computed KV blocks.
2. **Reuse.** Those $j$ tokens are *not* prefilled again. SGLang attaches their existing KV blocks to $R$'s attention computation directly (the same block-table indirection PagedAttention uses, now pointed at shared blocks). The prefill cost for the matched prefix is *zero*.
3. **Extend.** SGLang prefills only the unmatched suffix $r_{j+1}, \dots, r_m$, allocating fresh KV blocks for it, and inserts a new branch into the tree recording those blocks. Then it decodes the response, extending the same branch token by token.
4. **Evict.** The tree is memory-bounded. When the KV pool is full, SGLang evicts the **least-recently-used leaf** subtrees — never an internal node that an active request still depends on, because reference counts on the shared blocks prevent freeing KV that a live sequence is reading. This LRU-over-the-tree policy is what keeps the *hot* prefixes (your system prompt, your few-shot block) resident while cold one-off prefixes age out.

The reference-counting detail is the subtle correctness guarantee: a shared prefix node carries a count of how many active sequences point at it, and its KV blocks are only returned to the free pool when that count hits zero. This is exactly PagedAttention's copy-on-write generalized across *all* requests over *time*, not just within one explicitly-batched group.

**The cache hit rate for RLHF with system prompts.** The number that decides whether RadixAttention pays is the **prefix cache hit rate** — the fraction of a request's tokens that match a cached prefix and skip prefill. In RLHF this is usually dominated by the system prompt. A typical setup has a fixed system prompt (instructions, format spec, safety preamble) of, say, 300–600 tokens prepended to every rollout, followed by a variable user query of 50–200 tokens. Every single rollout shares that system-prompt prefix verbatim, so the hit rate on the system-prompt portion is essentially **100%** once it is hot — and across the whole request, with a 500-token system prompt and a 150-token query, that is a **70–90% prefix hit** on the prompt tokens. The practical effect: the system prompt is prefilled *once* for the entire run instead of once per rollout. For a run doing 1,024 rollouts per iteration with a 500-token system prompt, naive prefill would recompute $1{,}024 \times 500 = 512{,}000$ prompt-prefix tokens of attention every iteration; RadixAttention prefills those 500 tokens *once* and reuses the KV 1,024 times — a ~1,000× reduction in shared-prefix prefill compute for that component.

**The speedup from prefix sharing.** Where does that translate into wall-clock? Prefill is compute-bound and proportional to prompt length (quadratically in the all-to-all attention within the prompt, linearly in the number of prompts). Eliminating the redundant prefill of a long shared prefix removes a chunk of the prefill phase entirely. On workloads with heavy sharing — multi-sample GRPO with a long system prompt is the canonical case — SGLang's published benchmarks report several-fold end-to-end throughput gains over a non-sharing baseline, with the gain scaling in the ratio of shared-prefix length to unique-suffix length. The intuition to carry: **the longer and more-shared your prefix, the bigger the RadixAttention win**, because you are amortizing a fixed prefill over more reuses.

**Structured generation in RLHF** ties directly into verifiable rewards. A growing slice of RLHF — especially RLAIF and reasoning-RL — uses *programmatic* rewards: does the output parse as JSON, does it match a schema, does the extracted answer equal the ground truth? If the model can emit malformed output, you either waste rollouts on unparseable responses (zero reward through no fault of the policy's reasoning) or you build brittle retry logic. SGLang's constrained decoding masks the logits at each step against a compiled grammar/regex/JSON-schema automaton so that **only grammar-valid tokens have nonzero probability**. The rollout is *guaranteed* parseable, the reward function always gets well-formed input, and you spend your whole token budget on responses that can actually score. For verifiable-reward setups this is not a convenience — it removes an entire class of reward noise.

The `sglang.Engine` API is the offline-batch entry point that mirrors vLLM's `LLM` class, and it is the one you want for rollouts (as opposed to the `@sgl.function` frontend, which is for composing multi-step programs):

```python
import sglang as sgl

# Offline rollout engine — the LLM-class analogue for batch generation.
engine = sgl.Engine(
    model_path="meta-llama/Llama-2-7b-hf",
    mem_fraction_static=0.45,          # KV pool fraction (cf. gpu_memory_utilization)
    enable_radix_cache=True,           # RadixAttention prefix reuse (default on)
)

sampling = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 256}

# A shared system prompt across the whole batch -> high radix hit rate.
SYS = "You are a careful assistant. Answer in valid JSON with keys 'reason' and 'answer'.\n"
prompts = [SYS + build_user_turn(x) for x in prompt_batch]

outputs = engine.generate(
    prompts,
    sampling,
    return_logprob=True,               # per-token logprobs for the PPO ratio
)
rollouts = [{
    "text": o["text"],
    "old_logprobs": [lp for lp, _tid, _txt in o["meta_info"]["output_token_logprobs"]],
} for o in outputs]
```

Because every prompt here is prefixed with the identical `SYS` string, the radix tree caches that system prompt's KV on the first request and every subsequent request in the batch matches it for free — the 70–90% prefix hit in action. The `enable_radix_cache` flag is on by default; the only time you turn it off is the rare no-sharing stream where the bookkeeping is pure overhead.

**Structured generation** is the other reason to reach for SGLang. If your reward depends on the response being valid JSON, matching a regex, or conforming to a grammar (common in agentic RLHF, tool-calling, or verifiable-reward setups), SGLang's constrained-decoding engine masks the logits at each step so only grammar-valid tokens can be sampled. You get guaranteed-parseable outputs at full speed instead of generating-then-validating-then-retrying.

```python
import sglang as sgl

# A rollout that must emit valid JSON matching a schema, with logprobs.
@sgl.function
def scored_response(s, prompt, schema):
    s += prompt
    s += sgl.gen(
        "answer",
        max_tokens=256,
        temperature=1.0,
        json_schema=schema,         # constrained decoding: always valid JSON
        return_logprob=True,        # per-token logprobs for the PPO ratio
        top_logprobs_num=0,
    )

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf",
                      mem_fraction_static=0.45)
sgl.set_default_backend(runtime)

states = scored_response.run_batch(
    [{"prompt": p, "schema": SCHEMA} for p in prompt_batch],
    progress_bar=True,
)
rollouts = [{
    "text": st["answer"],
    "logprobs": st.get_meta_info("answer")["output_token_logprobs"],
} for st in states]
```

When do you use which? Reach for **SGLang** when you have heavy prefix sharing (many samples per prompt, long shared system prompts) or when you need structured/constrained outputs. Reach for **vLLM** when you want the most mature, widely-integrated, maximum-throughput general engine and your outputs are free-form. In practice both are excellent and the RLHF frameworks (TRL, verl, OpenRLHF) support both behind a flag; benchmark on *your* prompt distribution before committing, because the prefix-sharing advantage is entirely workload-dependent.

| Engine | Best for | KV scheme | Structured outputs | RLHF integration |
| --- | --- | --- | --- | --- |
| vLLM | General max throughput, free-form rollouts | PagedAttention | Via grammar backend (added later) | Mature (TRL, verl, OpenRLHF) |
| SGLang | Heavy prefix sharing, JSON/grammar rewards | RadixAttention | First-class | Growing (verl, OpenRLHF) |
| HF `generate()` | Quick baseline, tiny models, debugging | Static/contiguous | Via `outlines` add-on | Baseline, slow |
| TensorRT-LLM | Squeezing max throughput on fixed model | Paged | Manual | Hard to wire into a changing policy |
| LMDeploy | Strong throughput, TurboMind kernels | Paged | Limited | Limited RLHF tooling |

![A matrix comparing vLLM, SGLang, HuggingFace generate, TensorRT-LLM, and LMDeploy across throughput, latency, KV efficiency, log-prob extraction, and RLHF wiring](/imgs/blogs/vllm-async-rollout-collection-5.png)

## 8. Throughput vs latency, and async RLHF

There is a tension baked into batching that you have to make peace with. **Large batches maximize throughput** — more tokens per weight read, higher GPU utilization, more tokens/second aggregate. But large batches **increase per-request latency**, because any individual request waits in a big queue and shares the kernel with many others. **Small batches minimize latency** but waste the weight read and crater throughput.

For most RLHF you want throughput: you are doing a big offline batch of rollouts, nobody is waiting interactively, so crank the batch size and feed the GPU. But two cases pull the other way. First, **interactive or human-in-the-loop RLHF**, where a human is rating responses as they stream — there you care about time-to-first-token and want lower latency. Second, and more importantly for systems design, the **synchronous PPO loop wastes the trainer's GPUs during rollout and wastes the rollout GPUs during training.** While vLLM generates, the trainer sits idle; while the trainer updates, the rollout engine sits idle. On a synchronous loop your *effective* utilization across the whole cluster is the sum of two phases that never overlap.

**Asynchronous RLHF** decouples the two. Rollout workers continuously generate experience and push it into a shared buffer. The trainer continuously pulls batches from the buffer and updates. The two run *concurrently* on separate GPUs, so neither idles waiting for the other. The cost is that the experience in the buffer is slightly *off-policy* — it was generated by a policy a few updates behind the current trainer — which you bound with importance correction and by limiting how stale the buffer is allowed to get (a "staleness" or "max off-policy lag" knob).

![A graph showing three parallel vLLM rollout workers feeding a shared experience buffer, a trainer pulling batches and pushing weight broadcasts to the next iteration over NCCL](/imgs/blogs/vllm-async-rollout-collection-7.png)

This is the architecture that production RLHF at scale converges to. The rollout workers are tensor-parallel vLLM engines (Section 9); the buffer is a queue, sometimes a Ray actor; the trainer is FSDP/DeepSpeed-sharded PPO; the weight broadcast is NCCL (Architecture C from Section 6). Frameworks like OpenRLHF and verl implement exactly this. The async design is *why* you reach for the `AsyncLLMEngine` in Section 5 — it lets a rollout worker keep generating new requests while the trainer is broadcasting weights and the buffer is being drained, instead of stalling.

#### Worked example: synchronous vs async cluster utilization

Suppose rollout takes 13 s and the PPO update takes 12 s, on a 16-GPU cluster split 8 rollout / 8 train. *Synchronous*: total iteration is $13 + 12 = 25$ s, but during the 13 s of rollout the 8 training GPUs idle, and during the 12 s of training the 8 rollout GPUs idle. Average utilization across all 16 GPUs is roughly $(8 \times 13 + 8 \times 12) / (16 \times 25) = 200/400 = 50\%$. *Async*: rollout workers and trainer run concurrently; the iteration cadence is set by the slower phase, ~13 s, and both halves of the cluster stay busy. Utilization climbs toward ~90% (you lose a little to the weight broadcast and buffer sync). Same hardware, nearly 2× the throughput — the only price is managing bounded off-policy staleness, which PPO's importance weighting was built to handle anyway.

## 9. Multi-GPU rollout: tensor parallelism vs replicas

A 7B model fits on one 80 GB GPU; a 70B model does not. When the policy is too big for one GPU, or when you want more rollout throughput than one GPU provides, you have two ways to use multiple GPUs, and they are *not* interchangeable.

**Tensor parallelism (TP)** splits *one* model across GPUs. vLLM shards each layer — attention heads and MLP columns — across `tensor_parallel_size` GPUs, and the GPUs exchange activations with an all-reduce at each layer boundary. You use TP when the model does not fit on one GPU, or when you want lower latency for a single large request. The cost is the per-layer all-reduce communication, which scales with model depth and is only cheap over fast interconnect (NVLink within a node). Across nodes TP gets expensive fast.

```python
# Tensor-parallel rollout engine: one 70B model across 4 GPUs.
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,        # shard one model over 4 GPUs (NVLink)
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
outputs = llm.generate(prompts, SamplingParams(temperature=1.0,
                                               max_tokens=512,
                                               logprobs=0))
```

**Multi-replica** runs *several complete copies* of the model, each on its own GPU(s), each handling a slice of the prompts. No inter-GPU communication during generation at all — each replica is independent and you load-balance prompts across them. You use replicas when the model fits on one GPU (or one TP group) and you simply want more aggregate throughput. This is *embarrassingly parallel* and scales almost linearly, which is why the async architecture in Section 8 draws multiple independent rollout workers.

The decision rule: **if the model fits on one GPU, prefer replicas — linear scaling, no communication tax. Only use tensor parallelism when the model (plus its KV cache for your batch) does not fit on one GPU.** A common production setup for a 70B policy is TP=2 *within* each rollout worker (so the model fits) and *several* such workers as replicas (for throughput) — combining both axes. You can also mix in pipeline parallelism for very large models, but for rollout-sized models TP-within-worker plus replicas covers the vast majority of cases.

#### Worked example: TP overhead vs replica scaling

Take a 13B model that *just* fits on one 80 GB GPU with a modest batch. You have 4 GPUs and want maximum rollout throughput. Option one: TP=4, one model sharded across all 4 GPUs. Each decode step now incurs an all-reduce per layer (40 layers → 40 all-reduces per step), and the per-step latency includes that communication. Even over NVLink, the all-reduce overhead on the skinny decode activations means you might see, say, 3.1× the single-GPU throughput from 4 GPUs — sub-linear, because communication eats into the gain, and the KV cache is now also split so each GPU holds a quarter, not helping batch size as much as you would hope. Option two: 4 replicas, each a full model on one GPU, prompts load-balanced 1/4 to each. No inter-GPU communication during generation at all, so you get very close to 4.0× the single-GPU throughput — near-linear. The replicas win decisively here *because the model fits on one GPU*. Flip the model to 70B (does not fit on one 80 GB GPU) and the calculus inverts: you *must* shard, so TP=2 or TP=4 within a worker is mandatory, and then you add replicas on top for whatever GPUs remain. The rule is mechanical: shard only as much as you must to fit, replicate the rest.

## 10. Speculative decoding for faster rollouts

Speculative decoding attacks the bottleneck from Section 1 directly. Recall that one decode step costs ~7 ms whether you decode 1 token or many, because the weight read dominates. So if you could *verify several candidate tokens in a single forward pass*, you would get several tokens for one weight read.

That is the trick. A small, fast **draft model** generates the next $K$ tokens cheaply (it is small, so its forward passes are quick). Then the large **target model** (your policy) does a *single* forward pass over all $K$ drafted tokens at once — which it can, because they are all known — and checks, position by position, which drafted tokens it would have produced anyway. Thanks to a rejection-sampling acceptance rule, the accepted tokens are *distributed exactly as if the target had sampled them itself* — so speculative decoding is **lossless**: the output distribution is identical to plain target sampling. You accept the longest correct prefix of the draft (say $m \le K$ tokens), keep the target's own correction token at position $m{+}1$, and repeat. For coherent, predictable text the draft is right most of the time and you get a 2–3× decode speedup.

```python
# vLLM speculative decoding: a small draft model proposes, the policy verifies.
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",          # target = the policy
    tensor_parallel_size=4,
    speculative_config={
        "model": "meta-llama/Llama-2-7b-hf",     # small draft model
        "num_speculative_tokens": 5,             # K: draft 5 ahead
    },
    dtype="bfloat16",
)
outputs = llm.generate(prompts, SamplingParams(temperature=1.0,
                                               max_tokens=512,
                                               logprobs=0))
```

The caveat for RLHF: the speedup depends on the *acceptance rate*, which depends on how well the draft predicts the target. At high temperature (which RLHF uses for diverse sampling) acceptance drops, because the target's sampling is more random and harder for the draft to anticipate — so you get less than the 2–3× you would see at temperature 0. It still helps, especially for the more-predictable parts of a response, but do not expect the greedy-decoding numbers. Variants like Medusa (extra decoding heads instead of a separate draft model) and EAGLE (a lightweight draft trained on the target's features) push acceptance higher and are increasingly integrated into vLLM. For rollouts, speculative decoding is a *bonus* multiplier on top of PagedAttention and continuous batching, not a replacement for them.

The expected speedup has a clean form worth carrying in your head. If $\alpha$ is the per-token acceptance probability and you draft $K$ tokens, the expected number of tokens accepted per target forward pass is the expected length of the accepted run plus one (the target's correction token always lands), which works out to $\frac{1 - \alpha^{K+1}}{1 - \alpha}$ for the run plus the bonus token. The wall-clock speedup is that expected token count divided by the cost ratio of running the draft alongside the target. The key sensitivity is $\alpha$: it shows up geometrically, so acceptance is everything.

### Speculative decoding, the full algorithm

The acceptance rule above is the heart of why speculative decoding is *lossless*, and it is worth spelling out because it explains exactly when the technique helps and when it hurts a rollout. Here is the complete loop for drafting $K$ tokens against a target distribution.

1. **Draft.** Starting from the current context, the small draft model $q$ autoregressively samples $K$ tokens $x_1, \dots, x_K$, recording the draft probability $q(x_i)$ it assigned to each. These $K$ forward passes are cheap because the draft is small (a 7B draft for a 70B target, or a distilled head).
2. **Verify in parallel.** The target model $p$ does a **single forward pass** over the context plus all $K$ drafted tokens at once. Because all $K$ tokens are already known, this is one parallel prefill-shaped pass — *not* $K$ sequential decode steps — so it costs roughly one target forward pass, the same ~7 ms weight read we are trying to amortize. From this pass you get the target probability $p(x_i)$ for every drafted position, plus the target's full next-token distribution at position $K{+}1$.
3. **Accept / reject, token by token.** Walk the $K$ drafted tokens left to right. For each, draw $u \sim \text{Uniform}(0,1)$ and **accept** $x_i$ if $u \le \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$. If the draft was over-confident relative to the target ($q(x_i) > p(x_i)$) you accept with probability $p(x_i)/q(x_i) < 1$; if the target likes the token at least as much as the draft did ($p(x_i) \ge q(x_i)$) you always accept. Stop at the **first rejection**.
4. **Correct and resample.** At the first rejected position $m{+}1$, you do not just drop the token — you resample it from the *adjusted residual distribution* $p_{\text{adj}}(x) \propto \max(0,\, p(x) - q(x))$, which corrects exactly for the draft's bias. If all $K$ are accepted, you instead get a free *bonus* token by sampling from the target's distribution at position $K{+}1$ (which the verify pass already computed). Either way you always advance by at least one token per target pass.

The acceptance rule is what makes the output **distributionally identical to plain target sampling** — Leviathan et al. prove that accepting on $\min(1, p/q)$ and resampling rejections from the residual yields samples from $p$ exactly. So speculative decoding changes *how fast* you sample, never *what* you sample. For RLHF this is a critical property: your rollouts have the same distribution as if you had decoded normally, so the log-probs and the on-policy guarantee are untouched.

**Practical considerations for RLHF.** The technique interacts with RLHF's sampling regime in ways that do not show up in chatbot serving:

- **The draft model must produce diverse outputs — no temperature 0.** RLHF samples at temperature ~1.0 to get the variation the policy gradient learns from. Both the draft *and* the verify must respect that temperature; the acceptance rule uses the *temperature-scaled* $p$ and $q$. If you accidentally run the draft greedily while the target samples hot, the draft's distribution is wildly narrower than the target's, $q(x_i) \approx 1$ on its one greedy pick while $p(x_i)$ is spread thin, the ratio $p/q$ collapses, and acceptance craters. The draft must sample at the *same* temperature as the target.
- **When speculative decoding hurts: very diverse generation tasks.** When the target distribution is genuinely high-entropy — creative writing, brainstorming-style rollouts, or any task where many continuations are equally likely — no small draft can predict it well, $\alpha$ falls toward the inverse vocabulary odds, and you pay the draft's cost for almost no accepted tokens. In that regime speculative decoding is a *net slowdown*: you ran the draft for nothing and still paid for the target pass. The break-even is roughly when the expected accepted run length exceeds the draft's relative cost; below that, turn it off.
- **It composes with continuous batching, but contends for the same memory.** The draft model needs its own weights and KV resident, eating into the budget that would otherwise grow your decode batch. On a memory-tight co-located RLHF setup, the batch you give up to host the draft can cost more throughput than speculation buys back — another reason to measure on *your* workload rather than trust the headline number.

**vLLM's `--speculative-model` flag.** vLLM wires all of this behind configuration. On the CLI it is `--speculative-model <draft>` with `--num-speculative-tokens K`; in the Python API it is the `speculative_config` dict shown above. vLLM runs the draft, the parallel verify, and the rejection-sampling accept/reject inside its scheduler so it still integrates with continuous batching — the engine speculates per-sequence while keeping the decode batch packed. The flag also accepts the draft-free variants (`"method": "eagle"` or `"medusa"`) that replace the separate draft model with extra heads on the target, trading a little quality of prediction for not having to host a second model's weights.

#### Worked example: speculative speedup at RLHF temperatures

Take $K = 5$ drafted tokens. At temperature 0 on coherent text, a well-matched 7B draft for a 70B target might hit $\alpha = 0.8$. Expected accepted run $\approx \frac{1 - 0.8^{6}}{1 - 0.8} = \frac{1 - 0.262}{0.2} \approx 3.69$ tokens, plus the correction token gives roughly 3.7–4 tokens per target pass — close to the headline 3× after subtracting the draft's own (small) cost. Now crank to RLHF's temperature 1.0, where sampling is far more random and the draft predicts the target less reliably; acceptance might fall to $\alpha = 0.5$. Expected run $\approx \frac{1 - 0.5^{6}}{0.5} = \frac{0.984}{0.5} \approx 1.97$, so about 2 tokens per target pass — a ~2× speedup, still real but well below the temperature-0 number. This is exactly why I treat speculative decoding as a *bonus* on rollouts rather than a headline feature: it pays, but the high temperature RLHF demands clips its wings. If you ever run low-temperature rollouts (some RLAIF or verifier setups do), the gain is larger.

## 10b. Chunked prefill and prefill-decode disaggregation

Section 2 split generation into two phases with opposite performance profiles: **prefill** (process the whole prompt in one parallel pass — compute-bound, like training) and **decode** (the autoregressive loop — memory-bound, slow). Two newer optimizations exploit that split directly, and both matter for RLHF because rollout prompts are often *long* — a system prompt plus few-shot examples plus the query can run to thousands of tokens.

**Why standard prefill is slow for long prompts.** Prefill computes attention over the entire prompt at once, and attention is all-to-all: every prompt token attends to every earlier prompt token. The cost is quadratic in prompt length. A 2,000-token prompt does not just cost 4× a 500-token prompt — the attention term scales with the square, so a long prompt's prefill is a genuinely heavy compute burst. In a continuous-batching scheduler, that burst is a problem for everyone else: when a fresh long-prompt request arrives, naively running its full prefill in one iteration **monopolizes the GPU for that step**, stalling the decode steps of every in-flight sequence. The result is a latency spike — a jagged throughput curve where decode briefly freezes whenever a long prompt enters.

**Chunked prefill** fixes this by *slicing the prompt into fixed-size chunks* and feeding one chunk per scheduler iteration, **interleaved with the ongoing decode steps** of other sequences. Instead of "prefill all 2,000 tokens now, then resume decoding," the scheduler does "prefill 512 prompt tokens *and* take a decode step for everyone else, then the next 512, and so on." The long prefill is spread across several iterations so it never starves the decode loop. The mechanism is a budget: `max_num_batched_tokens` caps the total tokens processed per iteration, and the scheduler fills that budget first with decode tokens (one per active sequence) and then with as much prefill as fits, chunking a long prompt across as many iterations as needed.

```python
# vLLM: interleave long-prompt prefill with decode to avoid stalls.
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="bfloat16",
    enable_chunked_prefill=True,        # slice long prefills into chunks
    max_num_batched_tokens=2048,        # per-iteration token budget (chunk size)
    gpu_memory_utilization=0.85,
)
```

For rollouts the payoff is concrete: with long shared system prompts, chunked prefill turns a throughput curve full of prefill-induced stalls into a flat, high one. The measured effect on long-prompt workloads is typically a **20–40% improvement in decode-phase latency consistency** and a meaningful bump in sustained tokens/second, because the GPU never sits in a single-request prefill while a hundred decode-ready sequences wait. It is on by default in recent vLLM; the knob you tune is `max_num_batched_tokens` (the chunk size), trading prefill granularity against per-iteration overhead.

**Prefill-decode (PD) disaggregation** takes the separation one step further: run prefill and decode on **different machines**. Because prefill is compute-bound and decode is memory-bandwidth-bound, they want different hardware utilization profiles, and mixing them on the same GPU means neither phase runs at its ideal operating point — a long prefill burst leaves the memory bus idle, while a steady decode loop leaves the compute units idle. PD disaggregation assigns **prefill machines** (kept compute-saturated, churning through prompts) and **decode machines** (kept at a large batch to amortize the weight read), and ships the computed KV cache from the prefill node to the decode node over a fast interconnect when a request transitions from prefill to decode.

Why does this matter specifically for RLHF with long system prompts? Two reasons. First, the prefill of a long shared system prompt is exactly the kind of heavy, bursty compute that a dedicated prefill pool absorbs without disturbing the decode pool's steady throughput. Second, it composes with the prefix sharing from Sections 3 and 7: the prefill machines can hold the radix/prefix cache, prefill the shared system prompt once, and stream its KV to whichever decode machine needs it — concentrating the cache-reuse benefit where the prefill compute lives. The cost is the KV-cache transfer between nodes (the same KV bytes from the Section 3 formula, moved over the interconnect once per request), which is why PD disaggregation pays off when prompts are *long* (transfer amortizes over a long decode) and the interconnect is fast. For short prompts the transfer overhead dominates and co-located prefill+decode wins. PD disaggregation is the architecture large-scale serving stacks converge to, and the RLHF frameworks are beginning to adopt it for the rollout pool when prompts are long enough to justify the KV ship.

## 11. Measuring and optimizing rollout throughput

You cannot optimize what you do not measure, and the right metric for rollouts is **tokens/second/GPU on the decode phase**, not requests/second (which hides response-length variance). Instrument the rollout phase directly:

```python
import time, torch

def timed_rollout(llm, prompts, sampling):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    n_gpus = sampling.n if hasattr(sampling, "n") else 1  # adjust per setup
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"generated {gen_tokens} tokens in {dt:.2f}s "
          f"-> {gen_tokens/dt:,.0f} tok/s aggregate")
    return outputs
```

The knobs that move the number, in roughly the order I reach for them:

1. **`gpu_memory_utilization`** — the single biggest lever. The fraction of GPU memory vLLM may use for KV. Higher means a bigger decode batch means more throughput. Push it as high as your *other* memory consumers (the trainer copy, optimizer) allow. In a co-located setup this is the constant tug-of-war.
2. **`--enable-chunked-prefill`** — splits long prefills into chunks and *interleaves them with decode steps*, so a big prefill does not stall the decode loop of other sequences. With long prompts this smooths throughput and improves utilization. On by default in recent vLLM.
3. **`max_num_seqs` / `max_num_batched_tokens`** — caps on the decode batch. Raise them until you hit the memory or compute ceiling, then back off.
4. **`max_model_len`** — set it to your *actual* prompt+response budget, not some huge default. A smaller max length lets the block manager pack more sequences.
5. **Sequence-length distribution** — a few pathologically long rollouts can dominate the tail of the batch. A `max_tokens` cap bounds the worst case; for very skewed distributions, sorting prompts by expected length before submission (length bucketing) reduces straggler effects.

#### Worked example: reading a rollout profile and fixing it

A run reports 1,100 tok/s aggregate on 8 GPUs — 138 tok/s/GPU, suspiciously close to the naive 200 baseline, so something is wrong. Checking the config: `gpu_memory_utilization=0.30` (set low to be safe for the trainer) and `max_model_len=4096` while responses average 256 tokens. The low memory fraction is starving the decode batch and the huge max length is wasting block-table headroom. Raising `gpu_memory_utilization` to `0.55` (the trainer was over-provisioned) and `max_model_len` to `1024` (well above the real 256+prompt) widened the decode batch ~4×; throughput jumped to ~7,600 tok/s aggregate (~950 tok/s/GPU). Then turning on chunked prefill smoothed a prefill stall and it settled near 9,200 tok/s. No model change, no algorithm change — just feeding the scheduler enough memory and not lying to it about sequence lengths.

#### Worked example: rolling out 1024 responses three ways

Put the whole post to work on one concrete task: generate **1,024 responses from a 7B policy, 512 tokens each**, and compare three engines. Total tokens to produce: $1{,}024 \times 512 = 524{,}288$. Assume Llama-2-7B (14 GB bf16 weights, 1 MB of KV per token from the Section 3 worked example) on A100-80GB cards (~2 TB/s HBM, ~300 TFLOPs bf16).

**(a) Naive HuggingFace `generate()`, `batch_size=8`, 1 A100.** Static batching with full padding. Each decode step processes 8 sequences, and from Section 1 a single decode step is floored by the 14 GB weight read at $14/2000 \approx 7$ ms — so 8 sequences cost ~7 ms per token, about $8 / 0.007 \approx 1{,}140$ tok/s *if* the batch stayed full, but it does not: lengths vary, finished sequences sit as dead padding, and real-world HF decode on a 7B lands around **150–250 tok/s** sustained. Take 200 tok/s. To produce 524,288 tokens you need $524{,}288 / 200 \approx 2{,}620$ s — **~44 minutes**. The GPU is memory-bound the entire time, batch too small to amortize the weight read, padding wasting most slots.

**(b) vLLM, continuous batching, 1 A100.** PagedAttention frees ~60 GB for KV after the 14 GB of weights; at 1 MB/token that is ~60,000 KV tokens resident, so even at 512 tokens each you hold $60{,}000 / 512 \approx 117$ concurrent sequences — far past the point where the 7 ms weight read amortizes. From Section 1, one 7 ms decode step now yields ~117 tokens, so the ceiling is $117 / 0.007 \approx 16{,}700$ tok/s, but you bump the compute roofline and lose some to scheduling and prefill, so realistic sustained decode is **~2,000–2,500 tok/s** on one A100. Take 2,200. Then $524{,}288 / 2{,}200 \approx 238$ s — **~4 minutes**. Same GPU, same model: an **~11× speedup** over (a), purely from PagedAttention's bigger batch plus continuous batching keeping it full.

**(c) vLLM, tensor parallelism over 4 A100s.** TP=4 shards the 7B across 4 cards. Now each card holds ~3.5 GB of weights and a quarter of the KV, and the aggregate KV pool grows ~4×, so the decode batch can grow further — but every decode step pays an all-reduce per layer over NVLink on the skinny activations. The model already fit on one GPU, so TP buys throughput sub-linearly: expect roughly **3–3.5× the single-GPU rate**, say ~7,000 tok/s aggregate, not the naive 4× (this is exactly the TP-overhead lesson from Section 9 — for a model that *fits*, four replicas would scale closer to linear at ~8,800 tok/s and beat TP, but TP is the right tool the moment the model stops fitting). At 7,000 tok/s, $524{,}288 / 7{,}000 \approx 75$ s — **~1.25 minutes**, a further ~3× over single-GPU vLLM and **~35× over naive HF**.

The arc is the entire post in one number: 44 minutes of memory-bound, padding-wasting HF decode becomes 75 seconds once you amortize the weight read with a big paged batch, keep that batch full with continuous batching, and add GPUs the right way. And the footnote from Section 9 still holds — for this *fitting* 7B model, four replicas would edge out TP=4; reach for tensor parallelism when the policy stops fitting on one card, not before.

## 12. End-to-end: a vLLM rollout collector wired into PPO

Putting the pieces together — generation, log-prob extraction, reward scoring, advantage computation, and weight sync — here is the skeleton of a rollout collector that produces exactly the experience a PPO trainer consumes. This is the inner loop of the pipeline in figure 4.

```python
import torch
from vllm import LLM, SamplingParams

class VLLMRolloutCollector:
    def __init__(self, model_name, reward_fn, mem_util=0.45, max_len=1024):
        self.llm = LLM(model=model_name, dtype="bfloat16",
                       gpu_memory_utilization=mem_util,
                       max_model_len=max_len,
                       enable_prefix_caching=True)
        self.reward_fn = reward_fn          # callable(text) -> float
        self.sampling = SamplingParams(temperature=1.0, top_p=1.0,
                                       max_tokens=256, logprobs=0)

    def collect(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling)
        batch = []
        for out in outputs:
            comp = out.outputs[0]
            tok_ids = comp.token_ids
            old_lp = [lp[t].logprob
                      for t, lp in zip(tok_ids, comp.logprobs)]
            reward = self.reward_fn(comp.text)        # reward model score
            batch.append({
                "prompt_ids": list(out.prompt_token_ids),
                "response_ids": list(tok_ids),
                "old_logprobs": torch.tensor(old_lp),
                "reward": float(reward),
            })
        return batch

    def sync_weights(self, policy_model):
        # After the trainer's optimizer.step(): push new params in.
        sd = {k: v for k, v in policy_model.state_dict().items()}
        self.llm.llm_engine.model_executor.collective_rpc(
            "load_weights", args=(list(sd.items()),))
        self.llm.llm_engine.reset_prefix_cache()      # old KV is now stale


def compute_advantages(batch, gamma=1.0, kl_coef=0.05, ref_logprobs=None):
    # Token-level reward: terminal RM score minus per-token KL to reference.
    advs = []
    for i, ex in enumerate(batch):
        n = len(ex["response_ids"])
        kl = ex["old_logprobs"] - ref_logprobs[i]     # log pi - log pi_ref
        token_reward = -kl_coef * kl
        token_reward[-1] += ex["reward"]              # terminal RM reward
        # GAE/return reduces to discounted cumulative for terminal-only RM:
        returns = torch.flip(torch.cumsum(
            torch.flip(token_reward, [0]), 0), [0])
        advs.append(returns - returns.mean())         # simple baseline
    return advs
```

The shape to internalize: the *collector* owns generation and the on-policy log-probs; the *advantage* computation folds in the reward-model score and the per-token KL penalty to a reference policy (the term that keeps the policy from reward-hacking by drifting out of distribution); the *trainer* (not shown — it is your PPO clipped-surrogate update) consumes `old_logprobs`, recomputes $\log\pi_\theta$ under the current weights, forms the clipped ratio, and steps. Then `sync_weights` closes the loop so the next `collect` runs on the fresh policy. For the PPO objective itself and the KL-penalty derivation, the training-techniques RLHF post at `/blog/machine-learning/training-techniques/sequence-packing-llm-fine-tuning` covers the loss side that consumes this experience; this post is the *rollout* side that produces it.

## Case studies

**InstructGPT / the original RLHF recipe (Ouyang et al., 2022).** OpenAI's InstructGPT pipeline — SFT, then a reward model, then PPO — is the template every RLHF system descends from. The published recipe trained the policy with PPO against a learned reward model with a per-token KL penalty to the SFT reference. The systems lesson, only fully appreciated later, is that the PPO step is dominated by generation cost; the entire subsequent ecosystem of fast rollout engines exists to make that generation phase cheap. The 175B policy in that work made the generation bottleneck especially brutal, which is part of why later open reproductions leaned so hard on vLLM-style serving.

**The vLLM paper (Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023).** This is the source for the headline numbers: PagedAttention reduces KV-cache waste from the 60–80% of naive systems to under 4%, and the resulting larger batches deliver 2–4× throughput over the best prior serving systems (and far more over a naive HuggingFace pipeline). The paper frames KV management explicitly as an OS virtual-memory problem, which is the mental model this whole post is built on.

**OpenRLHF and verl (2023–2024).** These open RLHF frameworks were among the first to make vLLM (and later SGLang) the rollout backend behind a flag, and to implement the async, disaggregated architecture from Section 8 — separate rollout and training GPU pools with NCCL weight broadcast. Their published throughput comparisons show the async vLLM design substantially outpacing a synchronous HuggingFace-generate baseline on large policies, with the gap widening as the model and batch grow. They are the reference implementations if you want to read production-grade rollout code rather than the skeletons here.

**SGLang RadixAttention (Zheng et al., 2024).** SGLang's paper reports multi-fold throughput improvements on workloads with shared prefixes — exactly the multi-sample-per-prompt pattern of GRPO-style RLHF — by caching shared prefix KV in a radix tree. This is the empirical basis for the Section 7 recommendation to prefer SGLang when you sample many responses per prompt.

## When to use this (and when not to)

Use a paged-KV, continuous-batching rollout engine — vLLM or SGLang — for **essentially any RLHF or RL-on-LLM run on a model of 1B parameters or more.** The speedup is large, free of quality cost, and the frameworks have done the integration for you. If you are doing PPO, GRPO, RLOO, or RLAIF on a real model, this is not optional; it is the difference between a run that finishes overnight and one that finishes next week.

Reach for **SGLang** specifically when you sample many responses per prompt (GRPO, best-of-n) or need structured/constrained outputs for a verifiable reward. Reach for **plain HuggingFace `generate()`** only when debugging, when the model is tiny enough that generation is not the bottleneck, or when you need bit-exact agreement between the rollout and trainer forward passes and are willing to pay for it — sometimes useful when chasing a log-prob mismatch. Reach for **tensor parallelism** only when the model does not fit on one GPU; otherwise **replicas** scale better.

The decision collapses to: match the engine to your hardest constraint rather than defaulting to whatever a tutorial used. The tree below is the version of this I keep on a sticky note.

![A decision tree for choosing a rollout engine based on whether you need structured outputs, maximum throughput, custom CUDA kernels, or just a quick baseline](/imgs/blogs/vllm-async-rollout-collection-8.png)

Do *not* bother with a fancy rollout engine if generation genuinely is not your bottleneck — profile first. For a tiny reward-model-free setup, or a contextual-bandit-style single-token "action," there is no autoregressive loop to optimize and the whole apparatus is overkill. And do not turn on speculative decoding expecting greedy-decoding speedups at the high temperatures RLHF uses; the acceptance rate falls and the gain is smaller. The honest rule: measure tokens/second/GPU on the decode phase, and only optimize the thing the profile says is slow.

## Key takeaways

- **Generation, not training, is the RLHF bottleneck** — typically 60–80% of wall-clock time — because decoding is sequential and memory-bandwidth-bound, while training is parallel and compute-bound.
- **One decode step costs roughly the same whether you decode 1 token or many**, because reading the model weights from HBM dominates; this is why batching the decode loop is the core optimization.
- **PagedAttention** treats the KV cache like OS virtual memory — fixed blocks, a page table, non-contiguous physical pages — cutting waste from 60–80% to under 4% and enabling far larger batches.
- **Continuous batching** schedules at the iteration level, evicting finished sequences and admitting waiting ones every step, lifting GPU utilization from ~50% to ~85%+.
- **Capture per-token log-probs at generation time** (`logprobs=0` for the sampled token); they are $\log\pi_{\theta_{\text{old}}}$ for the PPO importance ratio, and there is no recovering them later.
- **Weight sync is the RLHF-specific hard part** — the policy changes every iteration, so push fresh weights into the engine (sleep/wake, resident-copy, or NCCL broadcast) and reset the prefix cache afterward.
- **A log-prob mismatch between the bf16 rollout engine and the trainer's forward pass** can blow up your KL even with identical weights; suspect it before suspecting the algorithm.
- **Async RLHF** decouples rollout workers from the trainer so neither idles, roughly doubling cluster utilization at the cost of bounded off-policy staleness.
- **Replicas beat tensor parallelism** when the model fits on one GPU; use TP only when it does not.
- **Profile in tokens/second/GPU on the decode phase**, then turn the knobs that matter: `gpu_memory_utilization`, chunked prefill, `max_model_len`, batch caps.

## Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023 — the vLLM paper; the source for PagedAttention and the throughput numbers.
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," OSDI 2022 — the iteration-level (continuous) batching idea.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," 2024 — RadixAttention and structured generation.
- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), 2022 — the RLHF/PPO recipe whose rollout cost motivates all of this.
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," 2023 — the lossless speculative-decoding algorithm and its acceptance analysis.
- Schulman et al., "Proximal Policy Optimization Algorithms," 2017 — the PPO objective that consumes the experience this post produces.
- The vLLM documentation (docs.vllm.ai) and OpenRLHF / verl repositories — production-grade rollout-engine and async-RLHF reference implementations.
- Within this series: the unified map `reinforcement-learning-a-unified-map`, the capstone `the-reinforcement-learning-playbook`, and the rollout-consuming training post at `/blog/machine-learning/training-techniques/sequence-packing-llm-fine-tuning`.
