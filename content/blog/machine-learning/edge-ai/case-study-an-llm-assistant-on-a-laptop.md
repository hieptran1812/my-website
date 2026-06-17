---
title: "Case study: shipping a private LLM assistant on a laptop"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "An end-to-end case study that takes a coding-and-writing assistant from an un-shippable fp16 model to a fast, private, offline helper on a 16 GB laptop — picking the model, quantizing to Q4_K_M, serving with llama.cpp, and stacking KV-quant, prefix caching, and speculative decoding, with the RAM budget math, real commands, and a cumulative before-after table."
tags:
  [
    "edge-ai",
    "model-optimization",
    "llm",
    "llama-cpp",
    "gguf",
    "quantization",
    "kv-cache",
    "speculative-decoding",
    "inference",
    "efficient-ml",
    "on-device",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-1.png"
---

Here is the goal, stated as plainly as I can: a ChatGPT-like helper that runs **entirely on your laptop**. No API key, no network call, no token of your code or your draft email ever leaving the machine. You close the lid on a plane and it still works. You're under an NDA that forbids pasting client code into a cloud service and it still works. The model file lives on your SSD, the inference runs on your own silicon, and the only thing the assistant "phones home" about is nothing at all.

That is an appealing goal and, in 2026, a completely achievable one — but only if you treat it as an engineering problem with a budget, not a download-and-pray exercise. The brief I'll hold myself to in this post is concrete: on a **16 GB consumer laptop** (the running example is a 16 GB M-series MacBook, with notes for a 16 GB Windows laptop with a small discrete GPU), ship an assistant that has **time-to-first-token (TTFT) under about one second** on a moderate prompt, decodes at **at least 15 to 20 tokens per second** (faster than most people read, so it never feels like it's stalling), produces **genuinely useful quality for code and writing**, and — the part everyone underestimates — **holds a long, multi-turn conversation without running the machine out of memory**, all while leaving enough RAM free that the user's browser, editor, and chat app keep working. That last clause is the whole game. A model that fits when it's the only thing running is not a product; a model that coexists with a person's actual desktop is.

We will walk this end to end, as a journey with six deliberate steps, and at each step I'll name the lever, show the command, and give a measured before-and-after number. Figure 1 lays out the whole route so you can see where we're going before we start: pick the model, quantize it, serve it, make it fast, fit long context, and productionize it. The numbers throughout are representative of what I see on this class of hardware — I'll flag where a figure is approximate and tell you how to measure your own — but the *shape* of every result is real and reproducible.

![Timeline of the six-step journey to ship a private laptop LLM assistant from picking a model through quantizing, serving, making it fast, fitting long context, and productionizing](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-1.png)

By the end you'll be able to: do the laptop RAM budget arithmetic in your head and know before you download a single byte whether a model will fit at your target context; convert and quantize a model to a 4-bit GGUF and serve it with an OpenAI-compatible local endpoint; stack the decode-time tricks that buy you throughput and TTFT for free; reason about the RAM-versus-context trade so a 32k chat doesn't OOM-kill the process mid-sentence; and read the final configuration off the accuracy-latency Pareto frontier instead of guessing. This is the capstone-style application of the whole series: it touches every lever — model choice, quantization, the runtime, and honest measurement — so I'll cross-link the deeper single-topic posts as we use each one, and tie the whole thing back to the four-lever frame in [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression). If you want the decision tree distilled into a checklist after reading this, that's [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

## Step 0: the budget that governs everything

Before any download, do the arithmetic. On a laptop, the binding constraint is almost never raw compute — it is **memory**, both how much fits and how fast you can read it. Get the memory model right and every later decision becomes obvious.

A running LLM occupies RAM in four buckets, and you should be able to size each one cold:

$$
\text{RAM}_{\text{total}} = \underbrace{P \cdot b_w / 8}_{\text{weights}} \;+\; \underbrace{\text{KV}(L)}_{\text{context}} \;+\; \underbrace{A}_{\text{activations}} \;+\; \underbrace{\text{OS} + \text{apps}}_{\text{not yours to control}}
$$

where $P$ is the parameter count, $b_w$ the average bits per weight, $L$ the number of tokens in the conversation, $\text{KV}(L)$ the key-value cache (which grows *linearly* with $L$), and $A$ a small activation scratch buffer that, at batch size 1, is almost a rounding error. The terms you control are the weights (via quantization and model choice) and the KV-cache (via context length and KV quantization). The OS-and-apps term you do *not* control, and ignoring it is the single most common reason a "16 GB model" gets the laptop swapping.

Figure 2 shows this as a stack, in the order the memory actually gets claimed. The OS and the user's apps take their cut first — call it 6 GB on a real working machine with a browser, an editor, and a chat client open. You should reserve roughly another gigabyte of headroom so the OS never starts paging to disk (on macOS, once you cross into swap, latency-sensitive inference falls off a cliff). What's left — about 9 GB on a 16 GB machine — is your entire budget for weights plus KV-cache plus activations.

![Stack diagram of the 16 GB laptop RAM budget showing OS and apps, safety headroom, model weights, KV-cache, and activations claiming memory in order](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-2.png)

Nine gigabytes. Hold that number. Everything that follows is a fight to fit a useful model plus a long conversation into nine gigabytes while keeping the read bandwidth high enough to decode fast.

There is a second budget, just as important and more often forgotten: the **bandwidth** budget, which sets your tokens-per-second ceiling. During decode — the token-by-token generation phase — the chip must stream every weight byte from memory for each new token. So the throughput ceiling is, to first order,

$$
\text{tok/s}_{\max} \approx \frac{\text{BW}_{\text{mem}}}{P \cdot b_w / 8}
$$

memory bandwidth divided by the bytes you must read per token. This is the roofline model applied to autoregressive decode, and it is the reason quantization speeds up generation even on a CPU: halving the bytes per weight roughly doubles the token rate, because decode is memory-bound, not compute-bound. I derive this properly in [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), but you need the one-line version now because it tells you *which* optimizations will help. Anything that reduces bytes-read-per-token (quantization, smaller model, KV-quant) speeds up decode. Anything that only reduces FLOPs (the classic "fewer multiply-adds" win) barely moves the needle here, because the chip is waiting on memory, not on the ALU.

A word on why the bandwidth bound is so tight and so unavoidable, because it surprises people coming from cloud GPU work. A datacenter A100 has ~2,000 GB/s of memory bandwidth; a laptop has 50 to 200 GB/s depending on the chip. That's a 10-to-40x gap, and since decode throughput is bandwidth divided by bytes-per-token, it transfers *directly* to a 10-to-40x throughput gap for the same model at the same precision. You cannot close that gap with cleverness; it's silicon. What you *can* do is shrink the numerator's enemy — the bytes-per-token — by quantizing, and that's why a laptop assistant lives or dies on quantization in a way a cloud deployment never has to care about. The laptop forces the discipline. On a unified-memory Mac the bandwidth is shared between CPU and GPU and is unusually high for a laptop (the M-series Pro/Max chips reach 200 to 400 GB/s), which is precisely why Apple Silicon punches so far above other laptops for local LLMs — the architecture that helps graphics also helps memory-bound decode.

#### Worked example: will an 8B fit at all?

Take a 16 GB M-series MacBook with roughly 100 GB/s of usable memory bandwidth, running an 8-billion-parameter model. At fp16, weights alone are $8\times10^9 \times 2 = 16$ GB — they don't even fit *before* you add a KV-cache or leave room for the OS. Dead on arrival. At Q4_K_M (about 4.5 effective bits per weight), weights are $8\times10^9 \times 4.5/8 \approx 4.5$ GB. Add an 8k-context KV-cache (we'll size this precisely later — about 1 GB with grouped-query attention) and the model footprint is about 5.5 GB. Against our 9 GB budget, that fits comfortably with room for the conversation to grow. And the decode ceiling: $100\,\text{GB/s} \div 4.5\,\text{GB} \approx 22$ tokens per second from bandwidth alone, before any speedups — right at the edge of our target. The arithmetic told us, before downloading anything, that fp16 is impossible and Q4 is exactly right. That is the whole point of doing the budget first.

One more thing the budget tells you that's easy to miss: TTFT is a *compute* cost, not a memory-bandwidth cost, so it scales differently. Prefill (processing the prompt to produce the first token) runs every prompt token through every weight matrix in one big, parallel matrix-matrix multiply — that's compute-bound, and its latency scales with prompt length times model FLOPs. Decode (every token after the first) is the memory-bound matrix-vector regime. So the two halves of "fast" have two different bottlenecks: TTFT is gated by how fast the chip can *compute* a prefill, and tok/s is gated by how fast it can *read* memory during decode. That split is why we attack them with two different levers in Step 4 — prefix caching for TTFT (avoid redundant prefill compute), speculative decoding for tok/s (use idle compute to dodge memory waits). Keep the split in mind; it organizes the entire optimization story.

## Step 1: pick the model — the quality-versus-size trade

The first real decision is which model to run, and it is a genuine fork with a non-obvious answer. You have a fixed weight budget — say 4 to 5 GB. You can spend it two ways: a **small model at high precision** (a native 3B at fp16, an "SLM-by-design" trained specifically to be small and good), or a **bigger model quantized harder** (an 8B at 4-bit). Same RAM. Different quality. Which wins?

The deep treatment of the first option is [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design), and of the second, [weight-only LLM quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq). Here I'll give you the decision and the reasoning, because for a *coding and writing* assistant the answer is clear and a little counterintuitive: **spend the budget on the bigger model, quantized.**

The reason is that 4-bit quantization is remarkably cheap in quality terms for a well-trained model, while raw parameter count buys capability that no amount of precision can fake. An 8B has roughly 2.7x the parameters of a 3B; that extra capacity shows up directly as better code generation, more reliable instruction-following, and longer coherent reasoning. Quantizing that 8B from 16 bits to 4.5 bits costs you a fraction of a perplexity point — on the order of 1% relative degradation for a modern k-quant — which is imperceptible in interactive use. So you trade an *imperceptible* quality loss (4-bit on the 8B) for a *large* quality gain (8B over 3B), at the same memory cost. Figure 7 makes the head-to-head explicit at a matched budget.

![Before-after comparison of a native 3B at full precision versus a quantized 8B at Q4_K_M at a matched RAM budget showing the bigger quantized model winning on code and writing quality](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-7.png)

It helps to be precise about *why* this asymmetry holds, because it is the load-bearing decision of the whole project. There are two competing forces. The first is that quality scales with parameter count along a smooth, well-documented curve — the scaling laws — so an 8B is reliably and measurably smarter than a 3B at the same training budget, and the gap is largest exactly on the hard, multi-step tasks (writing correct code, following a long instruction, holding a chain of reasoning) that a *coding and writing* assistant is judged on. The second force is that the quality cost of quantization is a *small, bounded* perturbation for a well-trained model — the network's redundancy absorbs zero-mean rounding noise, and k-quants spend their precision budget on the few sensitive tensors. So you are pitting a large, structural quality gain (more parameters) against a small, bounded quality loss (fewer bits). The large gain wins, and it isn't close. The only way the small model wins is when the *budget itself* is the binding constraint — when 4.5 GB is genuinely all you have and the 8B-at-Q4 doesn't leave room for the conversation — at which point you've moved to a different point on the frontier and the 3B is correct. That's the exception, not the rule, on a 16 GB laptop.

There are exceptions, and you should know them. If your target is even tighter — a phone, or you need to coexist with a memory-hungry app and can only spare 2 GB — a native 3B at Q4 (about 1.8 GB) may be the only thing that fits, and a well-trained 3B like Phi-3-mini or Qwen2.5-3B is genuinely useful. If you need extreme speed over quality (autocomplete-style, where 60+ tok/s matters more than the smartest answer), the smaller model's higher token rate can win on UX. And if your task is narrow — say, only SQL generation — a small model finetuned for that task can beat a generalist 8B. But for a general coding-and-writing assistant on a 16 GB laptop, the 8B-at-Q4 is the default I reach for, and I'll use it as the spine of this case study.

A concrete shortlist of what I'd actually consider in this class, all strong instruction-tuned models that quantize well: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, and for the smaller tier Phi-3-mini-4k (3.8B) and Qwen2.5-3B-Instruct. For the running numbers I'll use an 8B-class instruct model; the recipe is identical for any of them. Here's the decision laid out as a table, all at Q4_K_M so the comparison is apples to apples:

| Option | Weights (Q4) | Decode tok/s (M-series) | Quality for code/writing | Reach for it when |
|---|---|---|---|---|
| 8B at Q4_K_M | ~4.5 GB | ~18 base, ~34 spec | Strong; the default | 16 GB laptop, quality-first |
| 7B at Q4_K_M | ~4.0 GB | ~20 base, ~36 spec | Strong; near-8B | Slightly tighter budget |
| 3.8B at Q4 (Phi-3) | ~2.0 GB | ~32 | Good; punches up | Budget under ~3 GB, or a phone |
| 3B at Q4 | ~1.8 GB | ~40 | Decent; weaker on hard code | Speed-first, narrow tasks |
| 14B at Q4_K_M | ~8.0 GB | ~10 | Best, but slow + tight | 32 GB machine, quality over speed |

The 8B-at-Q4 row is the one I default to for a general 16 GB assistant: the best quality that still leaves room for a real conversation and runs fast enough with the Step 4 levers. The table also shows the escape hatches — drop to Phi-3 if the budget tightens, jump to 14B only if you have 32 GB and patience.

#### Worked example: SLM-by-design versus quantized-big at matched RAM

Hold the weight budget fixed at ~4.5 GB and compare. A native 3B at fp16 is ~6 GB (slightly *over* budget, actually — illustrating that fp16 small models aren't even the free lunch they look like); drop it to fp16-with-a-trim or accept 6 GB. It scores roughly 45% on HumanEval (a code-completion benchmark) and runs fast at ~40 tok/s because there are fewer bytes to stream. The 8B at Q4_K_M is 4.5 GB — *under* the 3B's fp16 footprint — scores roughly 62% on HumanEval, and runs at ~18 tok/s base (we'll triple-ish that later). For writing and coding, 62% versus 45% is the difference between an assistant you trust and one you double-check constantly. The 8B wins decisively on the metric that matters, at less RAM. (Benchmark figures here are representative of published instruct-model results; measure your own task, not HumanEval, before you commit — see the eval discussion in Step 6.)

## Step 2: quantize to Q4_K_M — the landing zone

We've chosen an 8B and decided to run it at 4-bit. Now the mechanics. The format we want is **GGUF** (the file format used by `llama.cpp`), quantized with the **Q4_K_M** scheme — a "k-quant" that mixes 4-bit and a bit of higher precision on the most sensitive tensors, and is the consensus sweet spot for quality-per-byte. The full treatment of the file format and the runtime is [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf); here's the part that matters for our budget.

**Why weight-only, and why 4-bit specifically?** Decode is memory-bound (Step 0), so the lever that helps is *fewer bytes per weight* — weight-only quantization. We are not quantizing activations to int8 here (that's a throughput trick for compute-bound prefill on hardware with int8 matmul units; see [LLM quantization for activations: SmoothQuant and the KV-cache](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache)). On a laptop doing single-user decode, weight-only 4-bit is the right tool: it directly attacks the bandwidth bottleneck. Why 4 bits and not 3 or 2? Because quality falls off a cliff below ~4 bits for an 8B. Q4_K_M holds quality to within a hair of fp16; Q3_K_M is noticeably worse on code (more subtle bugs, more hallucinated APIs); Q2 is usually a mistake for a model you actually rely on. Figure 4 shows the fp16-to-Q4 transition as the high-leverage move it is.

![Before-after comparison of fp16 weights versus Q4_K_M weights showing RAM dropping from 16 GB to 4.5 GB at a tiny quality cost](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-4.png)

The science of *why* 4-bit costs so little is worth a few paragraphs because it justifies the whole strategy. Quantization replaces each weight with the nearest of $2^b$ representable levels, spaced $\Delta$ apart over the value range. The rounding introduces an error uniformly distributed on $[-\Delta/2, \Delta/2]$, with variance $\sigma_q^2 = \Delta^2/12$. Each extra bit halves $\Delta$, which quarters the error variance — that's the classic $\text{SQNR} \approx 6.02\,b + 1.76$ dB rule, about 6 dB of signal-to-quantization-noise per bit. The reason an LLM tolerates this is that the error is *zero-mean noise injected into a high-dimensional matmul*, and the network's redundancy averages much of it out; the layers that *don't* tolerate it (a few sensitive projections) are exactly the ones k-quants keep at higher precision. That's the whole trick of Q4_K_M: spend bits where they matter, starve them where they don't, and land at ~4.5 effective bits with near-fp16 quality.

Trace the noise through one matmul to see why it stays small. A layer computes $y = Wx$. Quantizing $W$ to $\hat{W} = W + E$ where $E$ is the rounding error gives $\hat{y} = Wx + Ex$ — the output is corrupted by $Ex$, a sum of $d$ independent error-times-activation terms. By the central limit theorem that sum has variance proportional to $d \cdot \sigma_q^2 \cdot \sigma_x^2$, so the *relative* error in each output grows like $\sqrt{d}\,\sigma_q$ over the signal's $\sqrt{d}\,\sigma_w$ — it scales with the *ratio* $\sigma_q / \sigma_w$, not with $d$. The dimensionality that makes the model big does *not* amplify the relative quantization error; it averages it. That's the deep reason large models quantize gracefully: more dimensions means more averaging, not more damage. The damage that *does* happen comes from a few **outlier** weights and activations whose magnitude is far above the rest — they dominate the value range, blow up $\Delta$, and crush the precision of the ordinary weights. Modern schemes (GPTQ's error-feedback rounding, AWQ's per-channel scaling, k-quants' mixed precision) are all, at heart, outlier-handling tricks. That's the subject of [weight-only LLM quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq); the takeaway here is that the math *predicts* 4-bit is the landing zone, and measurement confirms it.

And the math also predicts where it stops working, which is the stress test you should internalize before getting greedy. Going from 4 bits to 3 doubles $\Delta$ and quadruples the error variance — a 6 dB SQNR loss — and at that point the outliers a few k-quant bits were protecting are no longer protected, so the degradation is *super*-linear in the bits you remove, not gradual. That's why Q3 is noticeably worse and Q2 is usually broken for a model you rely on: you've crossed from the regime where redundancy absorbs the noise into the regime where the noise corrupts the signal. The asymmetry — 4-bit is nearly free, 3-bit hurts, 2-bit breaks — isn't a fluke of one model; it's the SQNR curve meeting the outlier structure of trained transformers. Respect the cliff.

The practical flow has two paths. The easy path: download a pre-quantized GGUF from a reputable source (most popular models have community Q4_K_M GGUFs). The path that teaches you something and that you'll need for your own finetuned model: convert from Hugging Face weights yourself.

```bash
# 1. Clone and build llama.cpp (one-time)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
# Build with Metal on Apple Silicon (auto-detected); on Linux/CUDA use -DGGML_CUDA=ON
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j

# 2. Get the model weights (HF format) and convert to a GGUF in fp16 first
pip install -r requirements.txt
python convert_hf_to_gguf.py /path/to/Meta-Llama-3.1-8B-Instruct \
    --outfile models/llama-3.1-8b-f16.gguf \
    --outtype f16

# 3. Quantize the fp16 GGUF down to Q4_K_M
./build/bin/llama-quantize \
    models/llama-3.1-8b-f16.gguf \
    models/llama-3.1-8b-Q4_K_M.gguf \
    Q4_K_M
```

That `llama-quantize` step is where the 16 GB fp16 file becomes a ~4.7 GB Q4_K_M file. (The on-disk size is slightly above the 4.5 GB I quote for RAM because the format carries some metadata and the embedding/output tensors are often kept at higher precision; in-RAM resident size after load is what your budget cares about.) Verify the result before trusting it:

```bash
# Sanity-check size and run a quick perplexity measurement on a held-out text
ls -lh models/llama-3.1-8b-Q4_K_M.gguf      # ~4.7 GB
./build/bin/llama-perplexity \
    -m models/llama-3.1-8b-Q4_K_M.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    -c 512
```

Compare that perplexity to the fp16 file's. You're looking for a delta on the order of +0.05 to +0.10 — a fraction of a percent. If you see Q4 perplexity a full point or more above fp16, something is wrong (bad conversion, wrong chat template baked in, or the model genuinely doesn't quantize well — rare for mainstream instruct models). This is the first honest measurement of the journey and it costs five minutes.

#### Worked example: the size and quality landing

Before: Llama-3.1-8B at fp16, 16.0 GB on disk and in RAM, WikiText perplexity ~5.80 (representative). After: Q4_K_M, ~4.7 GB on disk, ~4.5 GB resident, perplexity ~5.86. That's a 3.4x size reduction for +0.06 perplexity — about 1% relative. In a blind chat, you will not feel that 1%. What you *will* feel is that the model now fits in your 9 GB budget with 4.5 GB to spare for the conversation. The single quantize command did the heaviest lifting of the entire project.

## Step 3: serve it — a local OpenAI-compatible endpoint

A model file is not an assistant. To get a ChatGPT-like experience you need a *server*: something that loads the model once, holds it in memory, exposes an HTTP endpoint, manages the conversation's KV-cache across turns, and speaks a protocol your UI can talk to. `llama.cpp` ships exactly this as `llama-server`, and crucially it exposes an **OpenAI-compatible** API — the same `/v1/chat/completions` shape the OpenAI SDK uses — so any chat UI or editor plugin that talks to OpenAI can point at `localhost` instead and never know the difference. That compatibility is what makes "private assistant" a drop-in, not a rewrite.

```bash
# Launch the server: model, full GPU offload, context window, host/port
./build/bin/llama-server \
    -m models/llama-3.1-8b-Q4_K_M.gguf \
    -ngl 99 \
    -c 8192 \
    --host 127.0.0.1 --port 8080 \
    --metrics
```

Two flags carry most of the weight here. `-ngl 99` (`--n-gpu-layers`) offloads up to 99 transformer layers to the GPU — on Apple Silicon, to the Metal/GPU cores that share the unified memory; on a Windows laptop with a discrete GPU, to CUDA. "99" just means "all of them" (an 8B has 32). On a unified-memory Mac, offloading is nearly free because CPU and GPU share the same RAM — there's no copy across a PCIe bus. On a discrete-GPU Windows laptop the story is different and important: if the model fits in **VRAM**, offload all layers and it flies; if it doesn't (say a 6 GB laptop GPU and a 4.5 GB model plus KV that spills over), you offload a *partial* number of layers (`-ngl 20`) and the rest run on CPU, with a performance cliff at the boundary. Knowing your VRAM is as important as knowing your RAM on that platform.

The `-c 8192` sets the context window — and therefore pre-allocates the KV-cache. This is a budget decision, not a free knob; we'll size it carefully in Step 5. `--metrics` exposes a Prometheus endpoint so you can watch tokens/s and TTFT honestly rather than eyeballing them.

The Windows-with-a-discrete-GPU path deserves its own walkthrough, because it's the most common non-Mac laptop and its memory model is genuinely different. A unified-memory Mac has one pool of RAM that both CPU and GPU share, so `-ngl 99` is nearly free and "does the model fit" is a single question against one budget. A Windows laptop has *two* separate pools: system RAM (say 16 GB) and the GPU's VRAM (say 6 or 8 GB on a laptop RTX card), connected by a relatively slow PCIe bus. The fast path is to fit the *entire* model plus KV in VRAM and offload all layers — then decode runs at the GPU's high VRAM bandwidth (often 300+ GB/s) and flies. But a 4.5 GB model plus a multi-gigabyte KV-cache often *won't* fit in 6 GB of VRAM, and then you face the partial-offload cliff: `-ngl 20` keeps 20 of 32 layers on the GPU and runs the other 12 on the CPU, with every token paying a CPU-GPU hop. Throughput on partial offload is gated by the *slowest* part of the path, so it can be dramatically worse than either all-GPU or all-CPU. The practical rule on a small-VRAM Windows laptop: pick a model and context that fit *entirely* in VRAM (a 7B-Q4 at 8k can fit in 6 GB with int8 KV), or accept CPU-only with its lower-but-predictable rate, and avoid the partial-offload valley unless you've measured that your specific split lands well. Build `llama.cpp` with `-DGGML_CUDA=ON` to get the CUDA backend; everything else in this post is identical.

Now talk to it. Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a concise coding assistant."},
        {"role": "user", "content": "Write a Python function to debounce calls."},
    ],
    stream=True,
)
for chunk in resp:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

That's it — a private assistant, running locally, streaming tokens, speaking the OpenAI protocol so it slots into Open WebUI, Continue.dev, or your own UI. On the M-series test machine, this baseline configuration gives roughly **18 tokens per second decode** and **about 0.7 seconds TTFT** on a 2k-token prompt. Both are inside the brief. We could almost stop here — but "almost" hides the failure modes that show up when a real user has a real conversation, and the next two steps are where the assistant goes from *works in a demo* to *survives a workday*.

A note on backends, because it's a frequent confusion. The Metal/CUDA `-ngl` path is one option; there are others. `llama.cpp` runs fine **CPU-only** (drop `-ngl`), which is slower but works on a laptop with no usable GPU — and on a high-bandwidth CPU it's surprisingly close, because decode is bandwidth-bound regardless of which compute units you use. For the broader landscape of local serving stacks — MLC, Ollama (which wraps `llama.cpp`), LM Studio, and the mobile-oriented runtimes — see [running LLMs locally with MLC and mobile stacks](/blog/machine-learning/edge-ai/running-llms-locally-mlc-and-mobile-stacks). For this case study `llama-server` is the workhorse because it's the most transparent: every flag maps to something you can reason about.

It's worth understanding what the server is actually *doing* across turns, because it changes how you think about the assistant. When the first request arrives, the server allocates a **slot** — a region of the pre-allocated KV-cache assigned to that conversation — and prefills the system prompt and first user message into it. The slot's KV-cache is the conversation's memory. On the next turn, the server recognizes that the new request shares a prefix with the slot's current state (same system prompt, same prior turns) and reuses that cached KV, prefilling only the new user tokens. This is the slot/cache-reuse machinery that makes prefix caching (Step 4) work, and it's why a *stateful* server beats statelessly re-sending the whole history to a stateless endpoint: the server can keep the expensive KV around. The flip side: each concurrent conversation needs its own slot and its own KV-cache, so on a memory-tight laptop you serve **one user, one slot** — the `--parallel 1` default — and you do not try to multiplex. Multi-tenant serving is a different budget and a different post; the laptop is gloriously single-user, and you should lean into that simplicity.

There's a security point hiding in "private," too, and it's the whole reason you're doing this. Bind the server to `127.0.0.1`, never `0.0.0.0`, so nothing off the machine can reach it. Don't expose the OpenAI-compatible port to your network unless you mean to. The privacy guarantee of a local assistant is only as good as your network binding — a model that runs locally but listens on every interface has quietly become a tiny unauthenticated inference service for your coffee shop's Wi-Fi. Loopback only, by default, always.

## Step 4: make it fast — three decode-time levers

Eighteen tokens per second is readable, but a good assistant should feel *snappy*, and TTFT matters more than throughput for perceived quality — nobody minds a fast stream, everyone minds a long pause before the first word. There are three levers that buy speed, and the beautiful thing about two of them is they cost **zero quality**: they're pure systems wins. I cover the mechanics in depth in [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast); here I'll apply them and measure. Figure 6 decomposes the speed stack so you can see that TTFT and throughput are two separate problems fixed by two separate levers.

![Matrix decomposing the speed stack into KV-cache quantization, prefix caching, and speculative decoding with their effect on TTFT and tokens per second](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-6.png)

### Lever A: KV-cache quantization (frees RAM, enables context)

The KV-cache is stored in fp16 by default. You can store it in int8 (or even int4) instead, halving (or quartering) its memory at a tiny, usually-negligible quality cost. This doesn't speed up decode directly — it frees RAM, which *buys you context length*, which is why I'm introducing it under "fast" but it really belongs to Step 5's long-context fight. Turn it on:

```bash
./build/bin/llama-server \
    -m models/llama-3.1-8b-Q4_K_M.gguf \
    -ngl 99 -c 8192 \
    -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port 8080 --metrics
```

`-ctk` and `-ctv` set the KV-cache type for keys and values; `q8_0` is int8. This halves KV RAM for about +0.02 perplexity — imperceptible. (You can push to `q4_0` for a 4x cut, but I'd measure carefully; int4 KV occasionally degrades long-context recall, the exact thing you turned on long context to get.)

### Lever B: prefix caching (slashes TTFT)

Here's a subtle, enormous TTFT win. Every chat turn sends the **same system prompt** and the **same conversation history** back to the model. Naively, the server re-runs prefill over all of it every turn — so a long history means a long pause before *every* reply, growing turn by turn. Prefix caching means the server *keeps the KV-cache of the shared prefix* between requests and only prefills the genuinely new tokens (the user's latest message). The system prompt is prefilled exactly once, ever.

`llama-server` does this automatically when consecutive requests share a prefix (it's the cache-reuse / slot mechanism). The effect is dramatic: a request whose 1,800 tokens of system-prompt-plus-history are already cached, with only 200 new user tokens, prefills 9x less. On an 8k-token prompt where the system prompt and prior turns are cached, TTFT drops from ~1.8 s to ~0.2 s. That is the difference between "it's thinking" and "it answered." For a stable, reusable system prompt, this is the highest-value TTFT lever you have, and it costs nothing but a little RAM to hold the cached prefix.

The arithmetic of *why* this works is just the prefill cost model. Prefill latency is roughly proportional to the number of tokens you prefill, because each one runs through the full model: $\text{TTFT} \approx L_{\text{new}} \times t_{\text{token}}$ where $t_{\text{token}}$ is the per-token prefill time. Without caching, $L_{\text{new}}$ is the *entire* conversation every turn — 2,000 tokens of history reprocessed to answer a 50-token follow-up. With prefix caching, $L_{\text{new}}$ is only the genuinely new tokens — the 50, not the 2,050. As the conversation grows, the gap widens without bound, which is exactly when it matters most: the naive setup gets *slower to respond* the longer you talk to it, which feels broken; the cached setup keeps TTFT flat because the new-token count per turn stays small. There's a subtlety to respect: prefix caching only helps if the prefix is *stable*. If your application rebuilds the system prompt every turn (injecting the current time, a changing tool list, or shuffled few-shot examples), the prefix changes, the cache misses, and you're back to full reprefill. Keep the stable part of your prompt at the *front* and the volatile part at the *end*, so the maximum prefix stays cacheable. That ordering is a free, real win that most people leave on the table.

### Lever C: speculative decoding (boosts throughput)

The throughput lever. Decode is sequential — one token per forward pass of the big model — and memory-bound, so the big model's forward pass spends most of its time waiting on memory with the compute units idle. Speculative decoding exploits that idle compute: a tiny, fast **draft model** proposes several tokens ahead; the big "target" model then verifies all of them *in a single forward pass* (verification is a parallel prefill-shaped op, which the idle compute units handle nearly for free). Every draft token the target accepts is a token you got without a separate slow forward pass.

```bash
# Run the 8B target with a small draft model for speculative decoding
./build/bin/llama-server \
    -m models/llama-3.1-8b-Q4_K_M.gguf \
    -md models/llama-3.2-1b-Q4_K_M.gguf \
    --draft-max 6 --draft-min 2 \
    -ngl 99 -ngld 99 -c 8192 \
    -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port 8080 --metrics
```

`-md` is the draft model (a 1B from the same family, so its vocabulary and style match — crucial for acceptance rate), `-ngld 99` offloads the draft too, and `--draft-max` is how many tokens it proposes per round. The speedup depends entirely on the **acceptance rate** $\alpha$: the fraction of drafted tokens the target accepts. The math of why this works, and how much, is worth deriving.

The crucial property — and the thing that makes speculative decoding *safe* to turn on — is that it is **exact**: the output distribution is provably identical to plain decoding from the target. The draft only *proposes*; the target's verification step uses a rejection-sampling rule that guarantees any accepted token is distributed exactly as the target would have sampled it, and any rejected token is resampled from the corrected distribution. So unlike quantization, speculative decoding costs you *zero* quality — it cannot change what the model would have said, only how fast it says it. That's why it's a pure systems win and why you should reach for it without worrying about a quality regression. The only thing it can hurt is throughput, if acceptance is so low that the draft's overhead outweighs the tokens it saves.

The science of speculative decoding. Let the draft propose $k$ tokens per round, the target accept each independently with probability $\alpha$, and let $c$ be the ratio of draft-forward-pass cost to target-forward-pass cost (small, since the draft is tiny). In one round you run the target *once* (to verify) and the draft $k$ times, and you produce, on expectation, $\frac{1-\alpha^{k+1}}{1-\alpha}$ accepted tokens (a geometric series — you keep accepting until the first rejection). The expected speedup over plain decode is approximately

$$
\text{speedup} \approx \frac{1 - \alpha^{k+1}}{(1 - \alpha)\,(1 + c k)}
$$

The numerator is tokens-produced-per-round; the denominator is cost-per-round in units of target forward passes. At $\alpha = 0.68$ (a realistic acceptance for a well-matched 1B draft on an 8B target for code and prose), $k = 5$, and $c \approx 0.08$, that's roughly $\frac{1 - 0.68^{6}}{(0.32)(1.4)} \approx \frac{0.90}{0.45} \approx 2.0\times$ — and after the inevitable real-world overheads, you land around 1.9x. Important caveat: speculative decoding only helps when decode is **memory-bound with spare compute**, which is exactly the single-user laptop case. On a saturated batched server it can *hurt*, because the spare compute it relies on isn't spare anymore. This is a quintessential edge-specific optimization.

#### Worked example: the speculative-decoding speedup at a measured acceptance rate

Base decode: 18 tok/s. Add the 1B Q4 draft, measure acceptance over a representative coding session — say it logs $\alpha = 0.68$, $k = 5$. The formula predicts ~2.0x; measured, accounting for sampling and verification overhead, ~1.9x, landing at **34 tok/s**. The cost: the draft model adds ~0.5 GB of weights to RAM and a little extra compute per token. The benefit: generation goes from "I can read along" to "it's done before I finish reading the question." If acceptance had come in low — say $\alpha = 0.4$ because the draft was a different model family — the speedup would collapse to ~1.3x and you'd reconsider whether the 0.5 GB is worth it. *Measure your acceptance rate; don't assume it.* (See the stress-test section for what to do when it's low.)

After all three levers, the assistant decodes at ~34 tok/s with ~0.2 s TTFT on a cached 8k conversation. We've blown past the brief on speed. Now the harder fight: keeping it alive over a *long* conversation.

## Step 5: fit long context — the RAM-versus-context trade

This is where naive local-LLM setups die. Everything's fast and fits at 8k; the user pastes a 30k-token codebase or has a long multi-turn debugging session, and the process gets OOM-killed mid-sentence. The culprit is the KV-cache, the one memory term that grows with the conversation. You have to budget it deliberately.

### Deriving the KV-cache size

For each layer, for each token, the cache stores one key vector and one value vector, each of dimension equal to the model's hidden size split across heads. With **grouped-query attention (GQA)** — which every modern efficient model uses — the keys and values are shared across groups of query heads, so the cached dimension is $n_{\text{kv}} \cdot d_{\text{head}}$ where $n_{\text{kv}}$ is the *much smaller* number of key-value heads, not the full head count. The total KV bytes are:

$$
\text{KV}(L) = 2 \times n_{\text{layers}} \times n_{\text{kv}} \times d_{\text{head}} \times L \times b_{\text{kv}}/8
$$

The leading 2 is for keys *and* values; $L$ is the context length; $b_{\text{kv}}$ is bits per KV element. The thing to internalize: it's **linear in $L$**. Double the context, double the cache. GQA is what makes long context affordable at all — without it, the multiplier would be the full head count, often 4 to 8 times larger.

Plug in an 8B-class model: $n_{\text{layers}} = 32$, $n_{\text{kv}} = 8$ KV heads, $d_{\text{head}} = 128$. At fp16 ($b_{\text{kv}} = 16$):

$$
\text{KV}(L) = 2 \times 32 \times 8 \times 128 \times L \times 2 \;=\; 131072 \times L \;\text{bytes} \approx 0.125\,\text{MB per token}
$$

So 8k tokens is ~1.0 GB; 32k tokens is ~4.0 GB. Figure 5 puts the 8k and 32k cases side by side — the 4x jump is the whole problem.

![Before-after comparison of KV-cache memory at 8k versus 32k context showing the cache quadrupling and how int8 KV brings 32k back inside the budget](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-5.png)

Now run the budget at 32k. Weights 4.5 GB + KV 4.0 GB (fp16) + OS 6 GB = 14.5 GB. On a 16 GB machine that's *right on the OOM edge* — one Chrome tab away from swapping. This is exactly the situation that kills naive setups: it works until the conversation gets long, then dies. Two fixes, both already in our toolbox:

1. **int8 KV-cache** (`-ctk q8_0 -ctv q8_0`): halves KV to 2.0 GB. Now 4.5 + 2.0 + 6 = 12.5 GB — comfortable, for +0.02 perplexity. This alone rescues 32k on a 16 GB machine.
2. **Cap the context to what you need.** Not every assistant needs 32k. If your real use is "answer questions about the file I'm editing," 8k or 16k is plenty and leaves enormous headroom. Set `-c` to your *actual* need, not the model's maximum. A common mistake is setting `-c 131072` because the model "supports 128k" — and pre-allocating a 16 GB KV-cache that instantly OOMs.

```bash
# Long-context config that fits 32k on a 16 GB laptop
./build/bin/llama-server \
    -m models/llama-3.1-8b-Q4_K_M.gguf \
    -md models/llama-3.2-1b-Q4_K_M.gguf --draft-max 6 \
    -ngl 99 -ngld 99 \
    -c 32768 \
    -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port 8080 --metrics
```

#### Worked example: the RAM budget at 32k context

Target: hold a 32k-token conversation on a 16 GB MacBook with a browser and editor open. Budget: weights 4.5 GB (8B Q4_K_M) + KV 2.0 GB (32k, int8) + activations 0.3 GB + OS/apps 6 GB = **12.8 GB**, leaving ~3 GB of headroom. It fits, with room for the OS to breathe. Compare the naive version: same model, *fp16* KV, *128k* context pre-allocated = 4.5 + 16 + 6 = 26.5 GB — instant OOM, the model never even loads. Same model, two flags different, the difference between shipping and a crash report. This is why the budget arithmetic is the most important skill in the whole post.

There's a UX subtlety worth a sentence: when a real conversation eventually *exceeds* your context window, you need a policy — truncate the oldest turns, or summarize them into a compact note the assistant keeps. That's an application-layer decision, not a serving one, but plan for it; "infinite chat" doesn't exist, and a graceful sliding window beats an abrupt OOM.

## The application layer: making it actually useful

A served model that fits and runs fast is the *engine*; an assistant is the *car*. Between the OpenAI-compatible endpoint and a useful product sit a few application-layer decisions that determine whether people keep using it, and they interact with the memory budget in ways worth naming.

**Conversation management against the context budget.** Your context window is a hard wall — at `-c 8192` the server has room for exactly 8,192 tokens of system prompt plus history plus the in-flight answer, and one token past that is an error or a silent truncation. So the application must *manage* the conversation to live inside the wall. The two standard policies are a **sliding window** (drop the oldest turns once you approach the limit, keeping the system prompt pinned) and **summarization** (when history gets long, ask the model to compress the old turns into a short note, then replace them with it). Summarization preserves more long-range context per token but costs an extra model call and can lose detail; the sliding window is dumb, cheap, and surprisingly good for most coding chats where recent context dominates. Pick one *before* you ship, because the failure mode of having no policy is the worst possible one: the conversation silently corrupts or the request errors out mid-sentence when it crosses the boundary. A reasonable default: pin the system prompt, keep the last $N$ turns verbatim, and summarize anything older into a running scratchpad — a hybrid that respects the budget while keeping the assistant coherent.

**Grounding the assistant in your files (local RAG).** The single biggest quality upgrade for a *coding* assistant isn't a bigger model — it's giving the model the *right context*. An 8B that can see the relevant function signatures and the project's conventions writes far better code than a 70B guessing blind. The local, private way to do this is retrieval-augmented generation entirely on-device: embed your codebase or notes with a small local embedding model, store the vectors in a local index, and at query time retrieve the few most relevant chunks and prepend them to the prompt. Everything stays on the laptop — no embeddings sent to a cloud vector DB, consistent with the whole privacy thesis. The cost is more context tokens (and therefore more KV-cache and more prefill), so RAG and the context budget are coupled: retrieve *few, relevant* chunks, not many marginal ones, both for quality and for memory. A practical setup runs a ~100 MB embedding model (it fits trivially in the budget alongside the 8B), indexes lazily, and retrieves the top 3 to 5 chunks per query — a few hundred extra prompt tokens for a large quality win.

**A note on tools.** If you want the assistant to *do* things — run code, search files, call a calculator — the OpenAI-compatible endpoint supports function/tool calling, and modern instruct models are trained for it. On-device, the constraint is that each tool round-trip is another prefill (the tool result gets appended and re-processed), so tools amplify the TTFT story: prefix caching and a stable prompt prefix matter even more in a tool-using agent than in a plain chat. Keep tool schemas compact and stable so they stay in the cacheable prefix.

The point of this section is that the engineering in Steps 0 through 5 *enables* a useful assistant but doesn't *constitute* one — and the application-layer choices (conversation policy, grounding, tools) all spend the same scarce resource, context tokens, that the memory budget governs. Everything routes back to the budget.

## Step 6: productionize — from "runs on my machine" to "ships"

You have a fast, fitting assistant. Shipping it — to yourself reliably, or to teammates, or as a feature in an app — is its own discipline, and skipping it is how a working demo becomes an unreproducible mess three months later. Four things matter, and they map onto the broader edge-MLOps practice covered in [edge MLOps: registries, OTA updates, and on-device monitoring](/blog/machine-learning/edge-ai/edge-mlops).

**Model as an asset: bundle versus download.** The ~4.7 GB GGUF is too big to embed in an app binary that ships through an app store (size limits, update churn). The standard pattern is **download-on-first-run**: ship a small app, fetch the model from a CDN on first launch, verify its checksum, cache it. This also lets you update the model independently of the app. The trade-off is the first-run experience (a multi-gigabyte download) versus a bloated, hard-to-update bundle. For a personal setup, just keep the GGUF in a known directory and pin its path; for a product, treat the model file as a versioned, checksummed asset with its own release lifecycle.

**Updating the model (OTA).** When a better quant or a finetune ships, you want users to get it without reinstalling the app. A model registry with version tags, a manifest the app checks on launch, and an atomic swap (download new, verify, switch the pointer, delete old) is the pattern. The risk to manage: a new model version that's better on your benchmark but worse on a user's actual workflow. Which is why the next point is non-negotiable.

**Measuring honestly.** Every number in this post is one you can and must verify on *your* hardware, because they're sensitive to memory bandwidth, thermal state, and the OS load. The honest-measurement discipline is its own topic — [profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device) — but the laptop-specific traps are: **warm up** before timing (the first run pays page-faults and kernel compilation; throw it away), measure **batch=1 / single-user** because that's the real workload, time **TTFT and decode tok/s separately** (they're different bottlenecks), and watch for **thermal throttling** on sustained generation (more on this in the stress test). `llama-bench` gives you reproducible prefill and decode numbers:

```bash
# Reproducible prefill (pp) and decode (tg) throughput, with warm-up
./build/bin/llama-bench \
    -m models/llama-3.1-8b-Q4_K_M.gguf \
    -p 512 -n 128 \
    -ngl 99 -r 5
```

Beyond `llama-bench`, you'll want to measure the *user-facing* numbers — real TTFT and decode rate as the client experiences them, plus actual resident RAM — because the bench tool measures the engine, not the round-trip. A few lines against the streaming endpoint give you the honest UX numbers:

```python
import time, requests

def measure(prompt, system="You are a concise coding assistant."):
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    with requests.post("http://127.0.0.1:8080/v1/chat/completions",
                       json=body, stream=True) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            if ttft is None:                       # first content chunk
                ttft = time.perf_counter() - t0
            n_tokens += 1
    total = time.perf_counter() - t0
    decode_tps = (n_tokens - 1) / (total - ttft) if total > ttft else 0
    return ttft, decode_tps

# Warm up first (throw the first run away), then measure
measure("ping")
ttft, tps = measure("Write a Python LRU cache decorator.")
print(f"TTFT {ttft*1000:.0f} ms   decode {tps:.1f} tok/s")
```

Run that after a warm-up call, report the median of several runs, and watch resident memory in Activity Monitor (macOS) or `nvidia-smi` / Task Manager (Windows) at the same time. Those three numbers — TTFT, decode tok/s, peak RAM — are your honest scorecard against the brief, and they're the rows of the cumulative table you'll build at the end.

The hardest measurement is **quality**, because it has no single number. Perplexity (Step 2) catches gross regressions cheaply but doesn't tell you if the assistant is *useful*. For that you need a **small task eval**: a held-out set of, say, 50 prompts representative of your real use — code snippets to complete, emails to draft, questions to answer — scored by a rubric or by a stronger model as judge. It doesn't need to be big; it needs to be *yours*. Then gate every model change on it: a new quant or finetune ships only if it doesn't regress your eval. This is the "good enough" gate in Figure 8, and it's what separates "I tried a quantization and it felt fine" from "I measured that Q4 costs 1.2 points on my task eval and decided that's acceptable."

Figure 3 is the centerpiece of the whole journey: the cumulative before-after, every step's contribution in one place. Read it top to bottom and you watch an un-shippable fp16 baseline become a fast, fitting assistant.

![Matrix showing cumulative RAM, time-to-first-token, and decode tokens per second across the fp16 baseline, Q4 quantization, KV int8, prefix caching, and speculative decoding steps](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-3.png)

## The cumulative before-after: the whole journey in one table

Here is every step's contribution, measured (representative numbers on a 16 GB M-series MacBook, 8B-class instruct model, 8k context for the RAM and TTFT columns):

| Step | RAM (8k ctx) | TTFT (2k prompt) | Decode tok/s | Quality (perplexity) | Max context that fits |
|---|---|---|---|---|---|
| fp16 baseline | 16.5 GB (over budget) | ~4 s (thrashing) | ~3 tok/s | 5.80 (reference) | 0k (won't load) |
| + Q4_K_M | 5.5 GB | 0.7 s | 18 tok/s | 5.86 (+0.06) | ~24k |
| + KV int8 | 4.5 GB | 0.7 s | 18 tok/s | 5.88 (+0.02) | ~48k |
| + prefix cache | 4.5 GB | 0.1 s | 18 tok/s | 5.88 | ~48k |
| + spec decode | 5.0 GB | 0.1 s | 34 tok/s | 5.88 | ~44k |

The fp16 baseline row deserves a word, because its numbers look almost cartoonishly bad and they're *meant to*. At fp16 the 16.5 GB working set doesn't fit in 16 GB, so the OS swaps continuously — the "~4 s TTFT, ~3 tok/s" isn't the model being slow, it's the *SSD* being the bottleneck as pages fault in and out on every forward pass. That row is what "didn't do the budget" looks like in production: not a clean failure, but a machine grinding through swap while the user wonders why their laptop is on fire. Every subsequent row is the budget being respected.

Read the rest of the story in the columns. **Quantization** (row 2) is the move that makes everything possible — it's the only row that changes RAM by gigabytes and is the difference between "won't load" and "fits comfortably." **KV int8** (row 3) doesn't touch speed but doubles the context that fits. **Prefix caching** (row 4) is a TTFT lever only — 7x faster first token, nothing else changes. **Speculative decoding** (row 5) is a throughput lever only — 1.9x tokens/s, at the cost of 0.5 GB for the draft (note the max-context dip as the draft eats a little RAM). No single trick does everything; the shippable assistant is the *stack*. And the quality column barely moves the whole way down — from 5.80 to 5.88, well under 2% — which is the entire reason this is a good trade: you bought a 10x improvement in fit-and-speed for a quality cost you cannot feel.

The deepest lesson of the table is about *ordering*. You apply quantization first because it's the precondition for everything — there's no point tuning the speed of a model that doesn't fit. KV-quant comes next because it's the precondition for long context. Then the speed levers, because they only matter once the thing fits and holds a conversation. This is the same lever-ordering logic the capstone formalizes: fit before fast, and within "fit," weights before cache. Get the order wrong — try to speed-tune an un-fitting model — and you waste effort optimizing something the budget will reject anyway.

## What broke, and how I fixed it (the stress test)

A clean table hides the debugging. Here's what actually goes wrong shipping this, and the fix for each — because the value of a case study is in the failures, not the happy path.

**The model plus KV exceeded 16 GB at long context.** First real outage. Set `-c 32768` with fp16 KV on the 16 GB machine, opened a few browser tabs, pasted a long document — and the process OOM-died. The 32k fp16 KV is 4 GB, weights 4.5 GB, and the OS had crept to 7 GB; the math doesn't close. *Fix:* int8 KV (`-ctk q8_0 -ctv q8_0`) cut the cache to 2 GB and bought back the headroom, as derived in Step 5. *Lesson:* always do the budget at your *maximum* context, not your typical one, and always assume the OS will take more than you think. The OOM doesn't happen in your test; it happens to the user with 40 tabs open.

The subtler version of this bug is even nastier and worth dwelling on: on macOS, *before* it OOM-kills, the system starts **swapping** — paging memory to the SSD to make room. The process doesn't crash, it just gets catastrophically slow, because now some of your weights or KV-cache live on disk and decode has to fault them back in. Your "memory-bound at 100 GB/s" assistant is suddenly bound by SSD random-read at a tiny fraction of that, and tok/s collapses to single digits or worse with the fan screaming. This is *worse* than a clean OOM because it's silent and intermittent — it works, then it's molasses, then it works again — which makes it hell to diagnose from a bug report. The fix is the same (cut the KV with int8, cap the context, leave headroom) but the *detection* is different: watch the swap/compressed-memory counter, not just total RAM, and treat *any* swap during inference as a budget failure even if it didn't crash. The whole reason I reserve a gigabyte of headroom in the Step 0 budget is to stay clear of the swap threshold, not the OOM threshold — by the time you're swapping, you've already lost.

**Quality dip at Q3.** I got greedy and tried Q3_K_M to claw back another gigabyte. Perplexity looked only slightly worse, but the task eval told the real story: the model started generating subtly wrong code — off-by-one errors, hallucinated method names, plausible-looking bugs. *Fix:* back to Q4_K_M; the gigabyte wasn't worth a model I couldn't trust. *Lesson:* perplexity under-reports code-quality damage from aggressive quantization. This is precisely why you need a task eval, not just perplexity — the metric that's cheap to measure is not the metric that matters, and below ~4 bits an 8B's code quality degrades faster than its perplexity suggests.

**Draft-model low acceptance.** First attempt at speculative decoding I grabbed a random 1B as the draft — different model family from the 8B target. Acceptance came in at ~0.40, the speedup was a measly ~1.3x, and the draft's 0.5 GB felt wasted. *Fix:* switched to a 1B from the *same* family as the target (matched tokenizer, matched training distribution); acceptance jumped to ~0.68 and the speedup to ~1.9x. *Lesson:* speculative decoding lives or dies on draft-target agreement. The draft must *think like* the target — same family, same vocabulary, ideally same instruct-tuning lineage. A mismatched draft is worse than no draft.

**Thermal throttling on sustained generation.** The 34 tok/s held for a 30-second answer but sagged toward ~26 tok/s during a multi-minute generation as the laptop heated up and the chip down-clocked. This is real and easy to miss: a 5-second benchmark looks great, a 3-minute one tells the truth. *Fix:* there's no magic flag — it's physics. What you *do* is (a) measure sustained, not burst, throughput so your published number is honest; (b) set UX expectations to the sustained rate; and (c) if it matters, a lower-power config (smaller model or fewer offloaded layers) can paradoxically sustain *higher* steady-state throughput than a hotter one that throttles. The honest-benchmarking discipline from [profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device) is exactly about catching this.

Reading the final config off the Pareto frontier. Each lever I added is a point in a multi-dimensional trade space (RAM, TTFT, throughput, quality), and the "right" config is the one on the **Pareto frontier** for *my* constraints — the set of configs where you can't improve one axis without sacrificing another. The 8B-at-Q4-with-int8-KV-prefix-cache-and-spec-decode sits on that frontier for a 16 GB laptop targeting quality-first interactive use. Move the constraints and the frontier point moves: a 32 GB laptop could afford Q5 or fp16 KV for a hair more quality; a 8 GB machine forces a 3B and a smaller context; a latency-obsessed autocomplete use would drop the 8B for a fast 3B. The framework for building and reading these frontiers is [the accuracy-latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier); the discipline is to *name your constraints first*, then pick the frontier point — not to chase one metric and accidentally fall off the frontier on another.

Figure 8 is that decision distilled into a gate: three checks — fits RAM at max context, hits the latency target, clears the task eval — and you ship only when all three pass, otherwise you change a lever and re-measure.

![Decision tree showing the three gates an on-device assistant must pass before shipping, RAM fit, latency target, and task eval, with the lever to pull when each one fails](/imgs/blogs/case-study-an-llm-assistant-on-a-laptop-8.png)

## Case studies: real shipped local assistants

This isn't theoretical — local LLM assistants on consumer hardware are shipping in volume, and the public numbers confirm the recipe.

**llama.cpp / Ollama / LM Studio on Apple Silicon.** The whole `llama.cpp` ecosystem (Georgi Gerganov and contributors) is the reference proof that 7B-8B-class models run interactively on M-series Macs at 15 to 40+ tok/s depending on chip and quant. Ollama and LM Studio wrap it with a friendlier UX; underneath, it's the GGUF + Metal + k-quant stack this post built. The community's standard recommendation — Q4_K_M as the default quant for quality-per-byte — is exactly the landing zone the SQNR math predicts.

**Apple Intelligence and on-device foundation models.** Apple ships a ~3B-class on-device model on recent iPhones and Macs, quantized (Apple uses palettization to roughly 3.5 bits per weight on average, a more aggressive scheme than Q4) and served through the Neural Engine. It's the SLM-by-design branch of Figure 7 taken to production: a smaller model, quantized hard, tuned for the device, with a server-side larger model for harder requests. The architecture validates the "fit quality to the budget" decision — on a phone, the budget forces the smaller branch.

**Microsoft Phi-3 / Phi-4 on laptops and phones.** The Phi family is the SLM-by-design thesis stated outright: a 3.8B model (Phi-3-mini) trained on heavily curated data to punch far above its weight class, explicitly designed to run on-device. Microsoft published it running on an iPhone in 4-bit. It's the model you reach for when 4.5 GB is too much and you need a genuinely capable assistant in under 2 GB. The Phi line is the clearest public evidence that data quality can substitute for parameter count up to a point — Phi-3-mini benchmarks competitively with much larger models on reasoning and code — which is exactly what makes it the right answer for the tighter budget where the 8B-at-Q4 won't fit. It's the SLM branch of the Step 1 decision, validated at production scale, and a reminder that "pick the model" is a real fork with two good answers depending on your budget, not a one-size verdict.

**Whisper.cpp and the broader "X.cpp" pattern.** The same person's `whisper.cpp` showed near-real-time speech-to-text on a laptop CPU using the identical recipe — quantize to GGUF, memory-map, run with hand-tuned kernels. It's a useful reminder that the techniques in this post generalize beyond text LLMs to any transformer you want to run privately and locally; the multimodal angle is covered in [multimodal and speech at the edge](/blog/machine-learning/edge-ai/multimodal-and-speech-at-the-edge).

These are not lab curiosities; they're shipped products and tools millions of people use. The recipe — pick the model to fit the budget, quantize to ~4 bits, serve with a KV-cache-aware runtime, stack the free speed levers — is the consensus practice, not a clever hack.

## When to reach for this (and when not to)

A local laptop assistant is the right call when **privacy is non-negotiable** (NDA'd code, medical or legal text, anything you can't send to a third party), when **offline operation matters** (travel, air-gapped environments, unreliable connectivity), when **per-request cost** at scale would dominate (a heavily-used assistant whose API bill would be brutal), or when **latency floor** matters and you don't want a network round-trip on every keystroke.

It is the *wrong* call — and be honest about this — when you genuinely need frontier-model quality (a local 8B is good, not GPT-class; for hard reasoning the gap is real), when the user's hardware is too constrained (a 4 GB Chromebook is not going to run a useful assistant, full stop), when you'd be reinventing serving infrastructure that a hosted API gives you for free and your volume is low (the API is cheaper than your engineering time below some threshold), or when the model needs to change frequently and pushing 5 GB updates to every device is operationally worse than a central deployment. The decisive rule: **a local assistant trades quality and operational simplicity for privacy, offline capability, and zero marginal cost.** If you don't value the things on the right side of that trade, use the API.

It helps to put the cost trade in numbers, because "zero marginal cost" is the local assistant's quiet superpower and people underweight it. A hosted frontier API might bill on the order of \$2 to \$10 per million output tokens. A heavy individual user generating, say, two million tokens a month sits around \$4 to \$20 a month — modest, and for that money you get frontier quality and zero ops. But a *team* of fifty such users is \$200 to \$1,000 a month, recurring, forever, and it scales linearly with usage; an automated workflow hammering the API can blow past that in a week. The local assistant's marginal cost per token is *electricity* — a laptop drawing tens of watts during generation costs a fraction of a cent per million tokens. So the cost crossover depends entirely on volume and quality-tolerance: below some usage threshold the API is cheaper than your engineering time and you should just use it; above it, or under a privacy or offline constraint that makes the API a non-starter at any price, local wins decisively. Name your volume and your constraints, then the cost comparison answers itself.

And within the "yes, go local" decision, don't over-engineer. If a 3B at Q4 hits your task eval, you don't need the 8B and its speculative-decoding stack — ship the simpler thing. If your conversations are never long, you don't need 32k context or KV-quant — set `-c 8192` and move on. Every lever in this post is a cost (RAM, complexity, a draft model to maintain); add them only when a measurement says you need them. The point of the budget-first method is to know which levers you actually need *before* you build them.

## Key takeaways

- **Do the budget arithmetic before you download.** $\text{RAM} = \text{weights} + \text{KV}(L) + \text{activations} + \text{OS}$. On a 16 GB laptop you have ~9 GB to work with after the OS and apps; size everything against that, at your *maximum* context, not your typical one.
- **Quantization is the move that makes it possible.** Q4_K_M cuts an 8B from 16 GB to ~4.5 GB for under 1% quality loss — the single highest-leverage step. Everything else is refinement.
- **Spend the budget on a bigger model quantized harder.** An 8B at Q4 beats a 3B at fp16 at the same RAM for coding and writing, because 4-bit costs almost nothing and parameters buy capability.
- **Decode is memory-bound; tokens/s ≈ bandwidth ÷ bytes-per-token.** That's why quantization speeds up generation and why FLOP-only optimizations don't. Know your memory bandwidth.
- **TTFT and throughput are different problems.** Prefix caching slashes TTFT (cache the system prompt and history); speculative decoding boosts throughput (a matched-family draft model). They stack because they don't overlap.
- **The KV-cache is what OOMs you, and it's linear in context.** int8 KV halves it; GQA makes long context affordable; cap `-c` to your real need, never the model's max.
- **Speculative decoding lives on acceptance rate.** Match the draft to the target's family or the speedup collapses. Measure $\alpha$; don't assume it.
- **Perplexity is cheap but under-reports code damage.** Gate model changes on a small, *yours* task eval, not just perplexity — that's how you catch the Q3 code regression.
- **Measure honestly: warm up, batch=1, separate TTFT from tok/s, and benchmark *sustained* throughput** so thermal throttling doesn't make your published number a lie.
- **Order the levers: fit before fast, weights before cache.** Quantize first (the precondition for everything), then KV-quant for context, then the speed levers. Optimizing the speed of a model that doesn't fit is wasted effort.
- **Pin and version the shipping config.** The exact model file, quant, KV type, context length, and draft model are a config artifact; record it, checksum the model, and gate model changes on your task eval so a "better" update can't silently regress.
- **Read the final config off the Pareto frontier for your constraints.** Name the constraints first, then pick the point. Don't chase one metric off the frontier on another.

## Further reading

- **Within this series:** [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) (the four-lever frame this case study applies) and [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) (the capstone checklist and decision tree).
- **The LLM-on-laptop track:** [small language models by design](/blog/machine-learning/edge-ai/small-language-models-by-design), [weight-only LLM quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq), [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf), and [making on-device LLMs fast](/blog/machine-learning/edge-ai/making-on-device-llms-fast).
- **The science and the discipline:** [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), [profiling and benchmarking on device](/blog/machine-learning/edge-ai/profiling-and-benchmarking-on-device), [the accuracy-latency Pareto frontier](/blog/machine-learning/edge-ai/the-accuracy-latency-pareto-frontier), and [edge MLOps](/blog/machine-learning/edge-ai/edge-mlops).
- **Papers and docs:** Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023); Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023); Lin et al., "AWQ: Activation-aware Weight Quantization" (2023); Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023); the `llama.cpp` repository documentation for GGUF, k-quants, `llama-server`, and `llama-bench`; and Abdin et al., "Phi-3 Technical Report" (2024).
