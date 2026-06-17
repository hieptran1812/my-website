---
title: "Running LLMs on phones: MLC-LLM, mobile runtimes, and the prefill/decode split"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Take a 3B-class LLM from a laptop onto a phone GPU with a compiled stack — MLC-LLM and friends — and learn why prefill and decode are different beasts, why time-to-first-token and tokens-per-second are separate metrics, and how to size a model to phone RAM without melting the chassis."
tags:
  [
    "edge-ai",
    "model-optimization",
    "mlc-llm",
    "on-device-llm",
    "mobile",
    "tvm",
    "inference",
    "efficient-ml",
    "prefill-decode",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-1.png"
---

The first time I watched a 3-billion-parameter language model answer a question on an airplane, with the phone in airplane mode and the cellular icon crossed out, it felt like a small magic trick. No round-trip to a data center, no API key, no token meter ticking up somewhere. I typed a prompt into a little chat app, the phone thought for about a third of a second, and then words started streaming out at a readable pace while the back of the device grew gently warm in my hand. The whole model — every weight, the tokenizer, the runtime, the GPU kernels — was sitting in a few gigabytes of the phone's memory, computing on the same silicon that renders the home screen.

If you have read the companion piece on running models on a laptop with [llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf), you already know the laptop story: a portable C++ runtime with hand-tuned kernels, a quantized GGUF file, and enough RAM and cooling headroom that you rarely have to think about thermal limits. A laptop is a forgiving place to run an LLM. A phone is not. A phone has a fraction of the memory, that memory is *shared* with the operating system and every other app, the chip throttles itself aggressively when it gets hot, and the user notices every milliwatt because they can watch the battery percentage drop. Getting an LLM to run *well* on a phone — using the mobile GPU or NPU, staying inside a thermal and battery budget, fitting in an app you can actually ship — is a genuinely harder problem, and it needs a different kind of stack.

This post is about that stack. The centerpiece is **MLC-LLM**, a project that takes the radical position that the right way to run an LLM on a heterogeneous zoo of mobile GPUs is to *compile* the model — ahead of time, through a real ML compiler ([Apache TVM Unity](/blog/machine-learning/edge-ai/ml-compilers-and-autotuning-tvm-mlir-xla)), into native autotuned kernels for each target backend (Metal on Apple, Vulkan and OpenCL on Android, WebGPU in the browser, CUDA on a desktop). That is a very different philosophy from llama.cpp's hand-written kernels, and understanding *why* compilation helps on phones specifically is half the lesson. The other half is the single most important mental model for on-device LLMs: the **prefill/decode split**. Prefill — reading your prompt — is compute-bound and loves the GPU. Decode — generating tokens one at a time — is memory-bound and is limited by how fast the chip can stream weights out of memory. These two phases have completely different performance characteristics, and once you internalize that, every confusing on-device LLM benchmark suddenly makes sense.

By the end you will be able to take a 3B-class model, compile it with MLC-LLM for a specific phone, run it on-device, and measure the three numbers that actually matter — time-to-first-token, decode tokens-per-second, and peak memory — honestly, including what happens when the chip throttles after a minute of sustained generation. You will know when a compiled stack like MLC wins, when plain llama.cpp is the better call, when to reach for a vendor NPU SDK, and how to size a model so it fits a phone's RAM without getting killed by the OS. This is the on-device-LLM corner of the broader [four-lever, accuracy–efficiency Pareto frame](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) this whole series is built around: quantization and efficient architecture get the model small enough; the compiler and runtime are what turn "small enough to fit" into "fast enough to use."

![Before and after comparison of running the same language model on a laptop with the llama.cpp runtime versus on a phone with a compiled MLC stack](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-1.png)

## Why a phone needs a compiled stack, not just a smaller model

It is tempting to think the only thing standing between a laptop LLM and a phone LLM is size — quantize harder, pick a smaller model, and ship the same runtime. That intuition is half right and half dangerously wrong. Size is necessary but not sufficient. The deeper problem is *kernels*: the small pieces of GPU code that actually do the matrix multiplications, the attention, the dequantization. On a laptop CPU, llama.cpp ships hand-written SIMD kernels (AVX2, AVX-512, NEON) and hand-written Metal shaders for Apple GPUs, and those cover the vast majority of laptops because there are only a handful of CPU instruction sets and GPU families to support. On phones, the GPU landscape is a fractured mess: Apple's own GPU with the Metal API, Qualcomm Adreno and ARM Mali and Imagination GPUs on Android (programmed through Vulkan or OpenCL), plus the browser's WebGPU as a fourth target. Each of these has different memory layouts, different optimal tile sizes, different warp/wavefront widths, and different quirks in how they handle the low-precision arithmetic an LLM needs.

Hand-writing a fast kernel for every (operation × GPU family × quantization format) combination is a combinatorial nightmare. This is precisely the problem ML compilers were built to solve, and it is worth reading the [compilers and autotuning deep-dive](/blog/machine-learning/edge-ai/ml-compilers-and-autotuning-tvm-mlir-xla) for the full machinery. The short version: instead of writing a kernel by hand, you describe the *computation* abstractly (a matrix multiply with these shapes, this quantization), and a compiler searches over many possible *schedules* (loop orderings, tilings, memory-staging strategies) to find a fast one for the specific GPU you are targeting. The compiler measures candidate kernels on the actual hardware — this is **autotuning** — and keeps the winners. MLC-LLM rides on Apache TVM's "Unity" stack (the `relax` graph IR plus TensorIR for kernels) to do exactly this. You compile once, ahead of time, per target, and you get kernels that are tuned to *that* GPU rather than a generic fallback.

Why does this matter *more* on phones than on a workstation? Three reasons. First, the hardware diversity is genuinely larger and you cannot pick one chip and forget the rest — an Android app might run on a hundred different SoCs. Second, mobile GPUs are weird: their programming models (OpenCL, Vulkan compute) are less mature for ML than CUDA, so a generic kernel leaves enormous performance on the table that autotuning recovers. Third, the margins are tighter: on a phone you are fighting for every token-per-second against a memory-bandwidth ceiling, so a kernel that is 30% slower than optimal is the difference between "fluid" and "frustrating." A compiled, autotuned kernel that saturates the GPU's memory bandwidth is not a nicety on a phone; it is the whole game.

The trade-off, and there is always a trade-off, is that compilation is *ahead-of-time and per-target*. llama.cpp ships one binary that detects your hardware at runtime and runs everywhere, at the cost of leaving some performance on the table. MLC ships a model compiled *for* a specific backend, which is faster but means you do a compile step per target and carry per-target artifacts. For a desktop hobbyist, that overhead is annoying. For a product team shipping to a known set of phones — or to "all WebGPU browsers" — it is exactly the right shape. We will make this concrete with the actual `mlc_llm` commands shortly.

There is a second, subtler reason compilation pays off on a phone that is easy to miss: **kernel fusion across a fragmented op graph.** An LLM forward pass is not one giant matmul; it is hundreds of small ones interleaved with elementwise work — dequantize the 4-bit weights, scale them, matmul, add the bias, RMSNorm, apply rotary embeddings, softmax the attention scores, and so on. On a phone GPU, every kernel launch carries fixed overhead (driver dispatch, command-buffer submission, a memory round-trip to write the intermediate result and read it back). If each of those operations is its own kernel, you pay that overhead hundreds of times per token and you bounce every intermediate tensor through main memory — which, as we are about to see, is the scarcest resource on the device. A compiler can *fuse* a chain like dequantize → scale → matmul → bias-add into a single kernel that keeps the intermediates in registers or shared memory and never writes them to DRAM. On a bandwidth-starved mobile GPU, fusing the dequant into the matmul is not a 5% trick; it can be the difference between bandwidth-bound and bandwidth-*saturated*, because the 4-bit weights get expanded to fp16 *inside* the compute unit instead of being expanded in DRAM and streamed at twice the byte count. Hand-writing every fused variant for every GPU is the combinatorial wall again; letting TVM generate and autotune them is how MLC gets a tuned, fused kernel for `q4f16_1` on Adreno that nobody had to write by hand.

The third reason is **diversity of warp/wavefront geometry.** A kernel that is optimal on Apple's GPU (32-wide SIMD groups, a particular shared-memory size, a specific number of registers per thread) is often mediocre on Adreno (different wavefront width, different register file, different texture-cache behavior) and outright wrong on Mali (which historically preferred different tiling and had a smaller, slower local memory). A hand-written kernel encodes one set of these choices. Autotuning *searches* the tile sizes, unroll factors, and memory-staging strategies separately for each GPU and keeps whichever wins on that silicon — so the same logical matmul becomes a 64×64-tiled kernel on one chip and a 32×128-tiled kernel on another, automatically. This is exactly why a compiled stack's advantage *widens* as the hardware zoo grows: every new GPU is just another autotuning target, not another kernel to write and maintain by hand.

### The mobile constraints that change everything

Before the mechanics, internalize the four physical constraints that make a phone different from a server, because every design decision downstream is a response to one of them. These are the same constraints catalogued in [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device), seen specifically through the LLM lens.

**Unified, shared memory.** A phone's CPU and GPU share one pool of RAM — there is no separate VRAM. A flagship phone in 2025 ships with 8–12 GB; a mid-range phone has 6–8 GB. But you do not get all of it. The operating system, the foreground app's own UI, the keyboard, background services, and the rendering surface all want memory too. On iOS, an app that grows past a memory limit (often a couple of GB for a normal app, more with an explicit entitlement) gets killed by the OS with no warning — the dreaded `Jetsam` event. On Android, `onTrimMemory` and the low-memory killer do the same. So your *practical* budget for an LLM is far smaller than the spec sheet: assume you have maybe 3–5 GB to play with on an 8 GB device after the system takes its cut. This is the single hardest ceiling, and it is why 7B is the rough upper bound for phones and 3B is the sweet spot.

Unified memory has a second, performance-side consequence that matters as much as the capacity ceiling: **bandwidth contention.** On a discrete-GPU desktop, the GPU has its own dedicated VRAM with its own bus, and the CPU's memory traffic does not compete with it. On a phone, the CPU, the GPU, the display controller (which is continuously reading the framebuffer to refresh a 120 Hz screen), the camera ISP, and the NPU all draw from the *same* LPDDR bus. The headline "68 GB/s" on a spec sheet is the *aggregate* the memory controller can deliver, and your decode kernel never gets all of it — the display alone can be quietly consuming several GB/s redrawing your scrolling chat UI while the model is trying to stream weights. This is why the achievable bandwidth $B_{\text{eff}}$ in the decode bound below is meaningfully lower than the rated peak, and why decode throughput is *worse* when the app is animating a busy UI than when the screen is static. The practical lesson is real: a chat app that streams tokens into a constantly re-laid-out, animated bubble can throttle its own model by stealing bandwidth from it. Quieting the UI during generation — fewer reflows, no heavy background blur, a static layout that only appends text — gives the decode kernel back the bandwidth it is starving for.

**Thermal throttling.** A phone has no fan. It dissipates heat through its chassis, and when the silicon gets too hot the SoC governor reduces clock frequencies to protect the chip and your fingers. LLM decode is one of the most thermally punishing workloads you can put on a phone GPU because it runs the GPU near full tilt continuously. The consequence is brutal and counterintuitive: your benchmark on a cold phone will show one number, and after sixty seconds of sustained generation the real number is 30–40% lower. Any tokens-per-second figure you report without saying "cold burst" or "sustained" is close to meaningless.

**Battery.** Every joule the GPU burns comes out of a battery the user is watching. Running a 3B model at full GPU tilt can pull several watts, which on a typical phone battery is a measurable percentage-per-minute drain and produces audible-to-the-user warmth. This caps how long a "local AI" feature can realistically run and pushes you toward smaller models, shorter generations, and aggressive idling.

**App packaging.** The model has to get *into* the app. Bundling a 2 GB model into the app binary inflates the install enormously (and may exceed app-store size limits), so most teams download the model on first launch and cache it. That introduces a first-run wait, version-management headaches, and storage pressure. None of this exists on a laptop where you just `curl` a GGUF file into a folder.

Hold those four constraints in mind — shared memory, thermal, battery, packaging — because the rest of this post is, in a sense, a guided tour of how the mobile LLM stack copes with each of them.

## The MLC compile pipeline, end to end

Let us walk the actual path a model takes through MLC-LLM, because the pipeline is the thing people find most mysterious and it is genuinely not that complicated once you see the stages. The figure below lays out the sequence; the prose fills in what happens at each step.

![Timeline of the MLC-LLM compilation pipeline from a Hugging Face checkpoint through weight conversion, TVM relax intermediate representation, kernel autotuning, and library compilation to a shipped phone app](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-2.png)

You start with a standard checkpoint — say a Llama-3.2-3B or Phi-3-mini or Qwen2.5-3B from Hugging Face, in the usual `safetensors` format with fp16 or bf16 weights. MLC's pipeline has three logically distinct stages, exposed as three subcommands of the `mlc_llm` CLI: **convert the weights**, **generate the config**, and **compile the model**.

**Stage 1 — `convert_weight`.** This reads the fp16 checkpoint and re-quantizes the weights into MLC's on-device format. The most common choice is `q4f16_1`: 4-bit grouped quantization for the weights, with fp16 accumulation and scales. (There are others — `q4f32_1`, `q0f16` for unquantized fp16, `q3f16_1` for more aggressive 3-bit — but `q4f16_1` is the default sweet spot for phones, mirroring the same 4-bit weight-only logic covered in [weight-only LLM quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq).) The output is a directory of quantized weight shards that the device will memory-map at runtime. Crucially, *weight conversion is target-independent* — the quantized weights are the same whether you eventually run on Metal or Vulkan, because quantization is a numerical transformation, not a hardware one.

**Stage 2 — `gen_config`.** This produces an `mlc-chat-config.json` describing the model: its architecture, the quantization scheme, context window, the conversation template (system prompt format, role tags), sampling defaults, and tokenizer wiring. This is the file you tweak to change the chat template or default temperature. It is small and human-readable.

**Stage 3 — `compile`.** This is where the compiler earns its keep. MLC lowers the model graph into TVM's `relax` IR — a functional graph representation where each operator (attention, the MLP, RMSNorm, the dequantize-then-matmul fusion) is a node. It then lowers each operator into TensorIR and applies a library of schedules and autotuned kernels for the *specific target backend you name*: `--device metal` for Apple, `--device vulkan` or `--device opencl` for Android GPUs, `--device webgpu` for the browser, `--device cuda` for an NVIDIA desktop. The output is a compiled model library — a `.so` or `.tar` on Android, a `.dylib` or static archive on iOS, a `.wasm` for the web. This library contains the actual GPU kernels. *This stage is target-specific*: you run it once per backend you intend to ship.

So the artifact you ship to a device is two things: the (target-independent) quantized weights from stage 1, and the (target-specific) compiled kernel library from stage 3, wired together by the config from stage 2. The chat app loads the library, memory-maps the weights, and runs.

Here is the actual command sequence. Note the column-zero comments are inside a fenced code block, which is fine.

```bash
# 0) Install MLC-LLM (nightly wheels include the TVM Unity runtime)
python -m pip install --pre -U -f https://mlc.ai/wheels \
    mlc-llm-nightly mlc-ai-nightly

# 1) Convert + quantize the weights (target-independent)
mlc_llm convert_weight ./Llama-3.2-3B-Instruct/ \
    --quantization q4f16_1 \
    -o dist/Llama-3.2-3B-Instruct-q4f16_1-MLC

# 2) Generate the chat/runtime config (template, context window, sampling)
mlc_llm gen_config ./Llama-3.2-3B-Instruct/ \
    --quantization q4f16_1 \
    --conv-template llama-3 \
    --context-window-size 4096 \
    -o dist/Llama-3.2-3B-Instruct-q4f16_1-MLC

# 3a) Compile the model library FOR ANDROID (Vulkan)
mlc_llm compile \
    dist/Llama-3.2-3B-Instruct-q4f16_1-MLC/mlc-chat-config.json \
    --device android \
    -o dist/libs/Llama-3.2-3B-q4f16_1-android.tar

# 3b) Compile FOR iOS (Metal) — separate target, same weights
mlc_llm compile \
    dist/Llama-3.2-3B-Instruct-q4f16_1-MLC/mlc-chat-config.json \
    --device iphone \
    -o dist/libs/Llama-3.2-3B-q4f16_1-iphone.tar
```

The pattern that matters: one `convert_weight`, one `gen_config`, and then *N* `compile` calls — one per target you ship. The weights are shared; the kernels are not.

Once compiled, you can sanity-check on your development machine before touching a phone. MLC ships a Python API and a CLI chat loop that uses the same engine as the mobile apps:

```python
from mlc_llm import MLCEngine

# Point at the compiled model directory (weights + config + library)
engine = MLCEngine(
    model="dist/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    model_lib="dist/libs/Llama-3.2-3B-q4f16_1-metal.so",  # desktop Metal lib
)

# OpenAI-compatible chat completion, streamed token by token
for chunk in engine.chat.completions.create(
    messages=[{"role": "user", "content": "Explain prefill vs decode in one paragraph."}],
    stream=True,
):
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)

engine.terminate()
```

For the phones themselves, MLC publishes reference apps — **MLCChat** for iOS (a TestFlight/Xcode project) and for Android (an APK / Android Studio project) — plus a `mlc4j` (Android) and a Swift package (iOS) you embed in your own app. The app bundles or downloads the weights, links the compiled library, and exposes the same streaming chat engine. The on-device API mirrors the Python one: you create an engine, feed it a prompt, and pull tokens out of a callback. Nothing exotic; the magic is all in the compiled library.

Before you trust any phone number, instrument the *desktop* run, because the same engine code path lets you separate prefill from decode quantitatively and catch a bad compile early. The snippet below times the two phases independently — it measures TTFT as the wall-clock to the first streamed token, then computes steady-state decode tok/s over the remaining tokens — which is exactly the prefill/decode decomposition we are about to derive:

```python
import time
from mlc_llm import MLCEngine

engine = MLCEngine(
    model="dist/Llama-3.2-3B-Instruct-q4f16_1-MLC",
    model_lib="dist/libs/Llama-3.2-3B-q4f16_1-metal.so",
)

def time_generation(engine, prompt, max_tokens=200):
    t0 = time.perf_counter()
    t_first = None
    n = 0
    for chunk in engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    ):
        delta = chunk.choices[0].delta.content or ""
        if delta:
            if t_first is None:
                t_first = time.perf_counter()   # boundary: prefill done
            n += len(delta)  # crude; use a tokenizer for exact token counts
    t_end = time.perf_counter()
    ttft = t_first - t0
    decode_s = t_end - t_first
    return ttft, n, decode_s

# Short prompt: prefill is cheap, TTFT is tiny
ttft_s, _, dec_s = time_generation(engine, "Hi", max_tokens=200)
print(f"short prompt  TTFT={ttft_s*1000:6.0f} ms")

# Long prompt: prefill dominates, TTFT balloons even though decode is unchanged
long_prompt = "Summarize this:\n" + ("the model reads every token. " * 200)
ttft_l, _, _ = time_generation(engine, long_prompt, max_tokens=200)
print(f"long  prompt  TTFT={ttft_l*1000:6.0f} ms  ({ttft_l/ttft_s:.1f}x longer)")
engine.terminate()
```

Run that and you will see the long prompt's TTFT is many times the short one's while the decode rate barely budges — the first concrete evidence of the split. Keep this harness; it is the same shape we will port to the phone for the honest on-device measurement later.

## The prefill/decode split: the most important idea on this page

If you take one concept away from this post, make it this one. Autoregressive LLM inference happens in two phases with completely different performance physics, and almost every on-device LLM surprise traces back to confusing them. The figure contrasts the two.

![Before and after comparison of the prefill phase as a parallel compute-bound matrix multiply versus the decode phase as a serial memory-bound weight sweep on a mobile device](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-3.png)

**Prefill** is the phase where the model reads your prompt. Suppose your prompt is $L$ tokens long. The model has to compute the key/value vectors for all $L$ positions and the attention over them, and — critically — it can do this for *all $L$ tokens at once*, in parallel, as big batched matrix multiplications. A matmul of shape (sequence × hidden) by (hidden × hidden) is exactly the kind of dense, parallel arithmetic a GPU is built to chew through. Prefill keeps the GPU's arithmetic units busy; it is **compute-bound**. The output of prefill is the first generated token plus a populated **KV cache** (the stored keys and values for every prompt position, which decode will reuse).

**Decode** is the phase where the model generates the answer, one token at a time. To produce token $t+1$, the model needs token $t$, so the steps are inherently *serial* — you cannot generate token 50 before token 49. Each decode step is, computationally, tiny: a single token flowing through the network, which is a batch of size one. The expensive matrix multiplications become matrix-*vector* multiplications (one row times a big weight matrix). And here is the crux: to compute that one token, the GPU must read *every weight in the model* out of memory. For a 3B model in 4-bit, that is roughly 1.6 GB of weights that must be streamed from RAM to the GPU's compute units *for every single token*. The arithmetic is trivial; the memory traffic is enormous. Decode is **memory-bound** — limited not by how fast the GPU can multiply but by how fast it can read.

This is not hand-waving; it falls straight out of the [roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives). Let me make the decode bound quantitative, because the math directly predicts the tokens-per-second you will see.

### Deriving the decode speed limit

Let the model have weight footprint $W$ bytes (after quantization) and let the device's achievable memory bandwidth be $B$ bytes per second. In decode, each generated token requires reading essentially all the weights once. So the time per token is bounded below by the time to move the weights:

$$t_{\text{token}} \;\ge\; \frac{W}{B}$$

and therefore the decode throughput is bounded above by:

$$\text{tok/s} \;\le\; \frac{B}{W}.$$

That is the whole story for decode in one inequality. Plug in numbers. A flagship phone in 2025 has a memory bandwidth on the order of $B \approx 50\text{–}68$ GB/s (LPDDR5/5X), of which a GPU compute workload might realistically *achieve* maybe 60–80% — call it $B_{\text{eff}} \approx 40$ GB/s to be honest about real-world efficiency. A 3B model quantized to 4-bit has $W \approx 1.6$ GB (3B parameters × 0.5 bytes/param + overhead). Then:

$$\text{tok/s} \;\lesssim\; \frac{40 \times 10^9}{1.6 \times 10^9} \;\approx\; 25 \text{ tokens/s}.$$

In practice you see something like 12–20 tok/s for a 3B model on a flagship phone — *below* the ceiling, because of attention overhead, the KV-cache reads, kernel launch overhead, and bandwidth you do not actually achieve. But the ceiling is real and it explains everything: a 7B model (≈3.8 GB in 4-bit) on the same phone is bounded near $40/3.8 \approx 10$ tok/s and you will see 5–8; a 1.1B model (≈0.7 GB) is bounded near 57 and you will see 25–30. **Decode speed is roughly inversely proportional to model size, because it is a memory-bandwidth race, not a compute race.** That single fact is why everyone obsesses over making models smaller for on-device: quantizing from 8-bit to 4-bit does not just save space, it nearly *doubles* decode speed by halving $W$. This is the deepest reason the [compression levers](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) matter so much for LLMs specifically.

### Why prefill and decode become two separate metrics

Because the two phases have different bottlenecks, they get two different metrics, and conflating them is the most common benchmarking error I see.

**Time-to-first-token (TTFT)** is dominated by prefill. It is the latency from "user hit send" to "first token appears." Since prefill is compute-bound and roughly linear in prompt length $L$, TTFT grows with the prompt:

$$\text{TTFT} \;\approx\; T_{\text{fixed}} \;+\; \frac{L}{R_{\text{prefill}}}$$

where $R_{\text{prefill}}$ is the prefill rate in tokens/s (often *much* higher than decode tok/s, because prefill batches many tokens into compute-efficient matmuls) and $T_{\text{fixed}}$ is the fixed cost of loading and warming up. A long prompt means a long wait before the first word.

**Decode throughput (tok/s)** is the steady-state generation rate after the first token, governed by the memory-bound bound above. It is essentially *independent of prompt length* (until the KV cache grows large enough to add meaningful read traffic of its own).

### Why TTFT scales with prompt length — the compute side of the same coin

It is worth deriving *why* TTFT grows with $L$ rather than just asserting it, because the reason is the mirror image of the decode bound. In prefill, the dominant cost is the matmuls in the attention and MLP blocks, and the FLOP count for processing $L$ prompt tokens through a model with $P$ parameters is approximately

$$\text{FLOP}_{\text{prefill}} \;\approx\; 2 \, P \, L \;+\; (\text{attention term} \propto L^2)$$

— the leading $2PL$ term is the two FLOPs (a multiply and an add) per parameter per token, summed over all $L$ tokens, and the $L^2$ term is self-attention's quadratic cost that only starts to bite at long context. For the prompt lengths a phone actually sees (tens to a couple thousand tokens), the linear $2PL$ term dominates, so prefill compute — and therefore TTFT — is roughly *linear* in $L$. Divide by the GPU's achievable compute throughput $C$ (in FLOP/s) and you recover the TTFT formula's slope:

$$\text{TTFT} \;\approx\; T_{\text{fixed}} \;+\; \frac{2 P L}{C}, \qquad R_{\text{prefill}} \;=\; \frac{C}{2P}.$$

Two things fall out of this immediately. First, **prefill is compute-bound while decode is memory-bound, and they live on opposite sides of the roofline** — prefill processes $L$ tokens in one batched pass, so the same weight bytes get reused across all $L$ tokens (high arithmetic intensity, GPU-friendly), whereas decode reads all the weights to produce *one* token (arithmetic intensity near 1, bandwidth-bound). That is *why* $R_{\text{prefill}}$ (often 300–800 tok/s on a phone GPU) is an order of magnitude higher than decode tok/s on the same chip and same model: same hardware, opposite bottleneck. Second, the quadratic attention term is a sleeper. At 512 tokens it is negligible; at 8K tokens the $L^2$ attention work becomes a real fraction of prefill, so TTFT on very long prompts grows *faster* than linearly. A phone "summarize my whole document" feature that pushes 16K tokens of context will feel that superlinear knee hard, which is one more reason long-context features are expensive *to use* on device even when the model technically supports the window.

These are orthogonal. You can have a model with great tok/s but a painful TTFT on long prompts (because prefill compute dominates), or a model that starts instantly on short prompts but generates slowly. A "RAG on phone" feature that stuffs 4,000 tokens of retrieved context into the prompt will feel sluggish *to start* even if generation is snappy, purely because of prefill. Knowing which phase your latency lives in tells you what to optimize: shrink the prompt (or use prompt caching) to fix TTFT; shrink the model (or quantize harder) to fix tok/s.

#### Worked example: prefill vs decode timing for a short vs long prompt

Take a 3B model on a flagship phone GPU with MLC, where you have measured a prefill rate $R_{\text{prefill}} \approx 350$ tok/s and a decode rate of $14$ tok/s, with fixed overhead $T_{\text{fixed}} \approx 0.05$ s.

- **Short prompt (32 tokens), generate 100 tokens.** TTFT $\approx 0.05 + 32/350 \approx 0.14$ s. Generation of 100 tokens at 14 tok/s $\approx 7.1$ s. Total $\approx 7.2$ s, and it *feels* responsive because the first word shows in 140 ms.
- **Long prompt (1024 tokens), generate 100 tokens.** TTFT $\approx 0.05 + 1024/350 \approx 2.97$ s. Generation is still 100 tokens at 14 tok/s $\approx 7.1$ s. Total $\approx 10.1$ s — but the *experience* is dramatically worse, because the user stares at a blank screen for almost three seconds before anything happens.

Same model, same decode speed, wildly different feel. The long prompt did not slow down generation at all; it slowed down the *start*. If a user complains "the AI feels slow," your first question should be "slow to start, or slow to type?" — those are different problems with different fixes.

#### Worked example: predicting the prefill rate from FLOPs and chip throughput

Let me show that $R_{\text{prefill}} \approx C / 2P$ is not just an abstract identity but actually predicts the number. Take the same 3B model ($P = 3 \times 10^9$) on a flagship phone GPU whose sustained, achievable compute for this mixed 4-bit-weight / fp16-activation workload is, say, $C \approx 2.1 \times 10^{12}$ FLOP/s (about 2.1 effective TFLOP/s — well below the GPU's fp16 peak, because dequantization and attention overhead eat into it). Then the predicted prefill rate is

$$R_{\text{prefill}} \;\approx\; \frac{C}{2P} \;=\; \frac{2.1 \times 10^{12}}{2 \times 3 \times 10^{9}} \;=\; 350 \text{ tok/s},$$

which is exactly the figure I used above — because that is where it came from. Now sanity-check the asymmetry: the *decode* bound for the same model was $\approx 25$ tok/s. So prefill chews through tokens about $350/25 \approx 14\times$ faster than decode on the very same chip. That single ratio is the whole prefill/decode story in one number: reading is cheap and parallel, writing is expensive and serial. And it tells you where your engineering effort goes — if your TTFT is bad, the lever is prompt length or a faster prefill kernel; if your tok/s is bad, the lever is a smaller or more-quantized model. Pulling on the wrong lever (shrinking the model to fix a long-prompt TTFT, or trimming the prompt to fix slow generation) wastes effort on the bottleneck you do not have.

The figure below shows how TTFT scales while tok/s stays flat.

![Matrix showing time-to-first-token rising with prompt length from 32 to 256 to 1024 tokens while steady decode tokens-per-second stays nearly flat](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-4.png)

The matrix makes the asymmetry visible: prefill work and TTFT climb roughly linearly with prompt length, while decode tok/s barely moves (it dips slightly at very long context as the KV cache adds read traffic). This is why "context length is cheap to *support* but expensive to *use*" on a phone — allocating a bigger KV cache costs memory, and actually filling it costs TTFT.

## Thermal throttling: the benchmark that lies to you

Here is the trap that has burned every team I know that shipped an on-device LLM. You compile your model, run it on a phone fresh out of your pocket, and measure 18 tok/s. You write "18 tok/s on Pixel" in the design doc. Then a user runs a real conversation — several back-and-forth turns over a couple of minutes — and reports it slows to a crawl. You cannot reproduce it because you keep testing in short bursts on a cool phone. The model did not change. The *temperature* did.

![Before and after comparison of phone decode throughput during a cold twenty-second burst versus sustained generation past sixty seconds as the SoC throttles to stay within its thermal budget](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-5.png)

Decode is a sustained, near-100% GPU workload — exactly the kind of thing a phone's thermal governor is designed to clamp down on. A phone has no active cooling, so it relies on spreading heat into the chassis and then *throttling clocks* when the silicon crosses temperature thresholds. The pattern is predictable: for the first 10–30 seconds the GPU runs at peak clocks and you get your headline tok/s. As the die heats up, the governor steps the GPU frequency down in stages. By 60–120 seconds of continuous decode, the steady-state throughput can be 30–50% below the cold-burst number, and it stays there as long as you keep generating. The chassis is warm, the battery is draining faster, and the model that benchmarked at 18 tok/s is now delivering 11.

The *why* is a simple thermal-mass argument worth making explicit, because it explains the characteristic shape of the throttle curve — high plateau, knee, low plateau — rather than a smooth fade. The SoC has a small thermal mass and a fixed rate at which the chassis can shed heat to the air. When you start a sustained decode, the GPU dissipates power $\dot Q_{\text{in}}$ (several watts) while the chassis sheds $\dot Q_{\text{out}}$ roughly in proportion to how much hotter the die is than ambient. At the start, the die is cold, $\dot Q_{\text{out}}$ is tiny, so almost all the input power goes into *heating the silicon* — the die temperature climbs along an exponential-approach curve toward its steady state. The "high plateau" is the window before the die crosses the governor's first throttle threshold; you are running at peak clocks on stored thermal headroom. The "knee" is the moment the die hits that threshold and the governor cuts frequency to drop $\dot Q_{\text{in}}$. The "low plateau" is the thermal *equilibrium*: the governor settles the GPU at whatever sustained clock makes $\dot Q_{\text{in}} = \dot Q_{\text{out}}$, the temperature where heat in equals heat out. That equilibrium frequency — not the boost frequency — is the one your sustained workload actually runs at, and because decode tok/s is roughly proportional to GPU clock (it is bandwidth-and-clock-bound, and the memory clock often steps down with the GPU clock), the sustained tok/s is set by the chassis's ability to shed heat, a physical constant of the phone's industrial design. A thicker phone, a metal frame, or a cooler ambient buys a higher equilibrium clock and thus a higher sustained tok/s; a slim phone in a hot pocket buys a lower one. This is why two phones with the same SoC can post different *sustained* numbers: the silicon is identical, the heat sink (the chassis) is not.

This is why the [on-device metrics post](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) insists on reporting *sustained* numbers. Honest measurement practice for an on-device LLM looks like this:

```python
import time

def measure_decode_sustained(engine, prompt, total_tokens=600, window=20):
    """Report tok/s in rolling windows so throttling shows up instead of hiding."""
    t_start = time.perf_counter()
    t_first = None
    produced = 0
    window_start = t_start
    window_count = 0

    for chunk in engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=total_tokens,
        stream=True,
    ):
        now = time.perf_counter()
        if t_first is None:
            t_first = now            # first token => TTFT boundary
        produced += 1
        window_count += 1
        if now - window_start >= window:               # close a window
            rate = window_count / (now - window_start)
            elapsed = now - t_start
            print(f"t={elapsed:6.1f}s  window tok/s={rate:5.1f}")
            window_start, window_count = now, 0

    ttft = (t_first - t_start) if t_first else float("nan")
    print(f"TTFT = {ttft*1000:.0f} ms over {produced} tokens")
```

Run that for ten minutes and plot the windowed tok/s, and the throttling curve appears: a high plateau, a knee, a lower plateau. The number you put in the spec should be the *lower* plateau (sustained), with the cold-burst number reported separately and clearly labeled. If your product does short interactions (a one-shot summarize, a quick reply suggestion), the cold-burst number is what users feel. If it does long generations or sustained chat, the sustained number is the truth. Reporting only the cold-burst number is the on-device equivalent of quoting a car's top speed measured downhill with a tailwind.

#### Worked example: a 3B model on a flagship phone GPU, with the throttle curve

Concrete, measured-style numbers for Llama-3.2-3B at `q4f16_1` via MLC-LLM on a recent flagship Android phone (Snapdragon-class GPU, 12 GB RAM), batch=1:

| Metric | Cold burst (first 20 s) | Sustained (after 90 s) |
| --- | --- | --- |
| Decode throughput | ~18 tok/s | ~11 tok/s |
| TTFT (64-token prompt) | ~0.30 s | ~0.34 s |
| Peak process memory | ~2.6 GB | ~2.6 GB |
| GPU clock state | peak | thermally capped |
| Battery drain | — | ~0.4–0.7 %/min |

A few things to read off this. Memory does *not* change with throttling — the working set is the weights plus the KV cache, which is fixed by the model and context length, not by clock speed. TTFT barely moves because prefill is short and over before the chip heats up; throttling hits *decode* hardest because that is the sustained part. And the decode number drops ~40%, which is exactly the gap that turns "fast" into "frustrating." If you only test in 20-second bursts, you will ship the 18 and your users will get the 11.

The honest move is to *design for* the sustained number. If your feature needs to feel fluid (≥10 tok/s is roughly the lower bound for comfortable reading), and the *sustained* rate of a 3B model is 11 tok/s, you are right at the edge — you might drop to a 1.5B model for margin, or accept that long generations will feel slow once the phone warms up, or cap generation length so the chip never reaches the lower plateau. These are real product decisions that fall out of taking thermal throttling seriously instead of pretending the cold-burst number is the truth.

## The mobile stack landscape: who runs LLMs on phones, and how

MLC-LLM is not the only way to run an LLM on a phone, and which stack you pick depends heavily on your platform constraints and how much you value peak GPU performance versus simplicity. Here is the honest landscape. The matrix summarizes; the prose explains the trade-offs.

![Matrix comparing mobile LLM runtimes MLC-LLM llama.cpp ExecuTorch Core ML MLX and Qualcomm Genie across backend kernels and platform reach](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-6.png)

**MLC-LLM** is the compiled-everything option. Its superpower is *breadth with peak GPU performance*: through TVM it targets Metal, Vulkan, OpenCL, WebGPU, and CUDA from one source, with autotuned kernels on each. If you need to ship the same model to iOS, a wide range of Android GPUs, *and* the browser, MLC is uniquely positioned — WebLLM (the browser sibling) runs the very same compiled-to-WebGPU models entirely client-side. The cost is the AOT compile step per target and a heavier build process.

**llama.cpp** also runs on phones, and it is the simplicity champion. It ships hand-written kernels (CPU NEON, Metal, and increasingly Vulkan) and a single GGUF file you can swap freely. On Apple it uses Metal well; on Android it historically leaned on the CPU (NEON), which is slower than a tuned GPU path but dead simple and robust. If your model is small, your latency budget is loose, and you value a single portable artifact over peak speed — or you are already using GGUF on the laptop and want one format everywhere — llama.cpp on the phone is a perfectly reasonable, low-friction choice. It is the [llama.cpp/GGUF stack](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf) you already know, just running on ARM.

**ExecuTorch** is PyTorch's official on-device runtime. You `torch.export` a model, lower it to an ExecuTorch program, and run it through delegates — XNNPACK for CPU, Apple's MPS/Core ML for Apple GPU/ANE, Vulkan on Android, and vendor backends (Qualcomm, MediaTek) where available. For LLMs it has a dedicated path (the `llama` example and a KV-cache-aware export). Its pull is *PyTorch nativeness*: if your model lives in PyTorch and you want to stay in that ecosystem end-to-end, ExecuTorch keeps you there, and it is the same runtime discussed in the [mobile deployment end-to-end](/blog/machine-learning/edge-ai/mobile-deployment-end-to-end) post for non-LLM models too.

**Core ML and MLX (Apple).** On Apple silicon you have two first-party options. **Core ML** (via `coremltools`) compiles models to run on the Apple Neural Engine (ANE), GPU, or CPU, and is the most power-efficient path when the ANE can take the model — but the ANE is finicky about which LLM operations it accepts, and large LLMs often fall back to GPU. **MLX** is Apple's array framework built for unified memory, with an `mlx-lm` package that runs LLMs very efficiently on the GPU of Apple silicon (great on Macs and iPhones/iPads). MLX is increasingly the default for "run an LLM well on Apple hardware" because it is designed around the unified-memory architecture. The catch, of course, is that both are Apple-only.

**Vendor NPU SDKs.** Qualcomm ships **Genie** (a high-level LLM runtime) on top of **QNN** (the Qualcomm Neural Network SDK) targeting the Hexagon NPU; MediaTek and others have analogous stacks. When the NPU can run the model, it is the most power-efficient option — NPUs are purpose-built for low-energy matrix math. The cost is portability: a QNN-compiled LLM runs only on Qualcomm silicon, the toolchain is heavier and less open, and operator coverage for the latest model architectures lags. Vendor NPU paths win when you control the hardware (e.g., a single phone model, or a kiosk) and need the best energy efficiency.

**WebGPU / WebLLM.** The browser is now a viable LLM runtime via **WebGPU**, and **WebLLM** (MLC's browser project) runs compiled models entirely client-side — no install, no app store, the model downloads and caches in the browser and runs on the user's GPU. It is the most frictionless distribution channel imaginable (a URL) at the cost of WebGPU's still-maturing performance and the model-download-over-the-web problem.

Laid out side by side, the trade space is easier to hold in your head. The table below scores each stack on the axes that actually drive the decision — what it runs on, how it gets its kernels, how wide its platform reach is, where its energy efficiency lands, and the one-line "pick it when":

| Stack | Kernel strategy | Backends | Platform reach | Energy | Pick it when |
| --- | --- | --- | --- | --- | --- |
| **MLC-LLM** | Compiled + autotuned (TVM) | Metal, Vulkan, OpenCL, WebGPU, CUDA | Widest (iOS + Android GPUs + web) | Good (tuned GPU) | Broad GPU reach with peak GPU perf, or you need the browser |
| **llama.cpp** | Hand-written kernels | CPU NEON, Metal, (Vulkan) | Wide, single GGUF artifact | Moderate (CPU-heavy on Android) | Simplicity, one portable file, loose latency budget |
| **ExecuTorch** | Exported + delegated | XNNPACK, MPS/Core ML, Vulkan, vendor | Wide via delegates | Good–great (vendor delegate) | You live in PyTorch and want to stay there |
| **Core ML / MLX** | First-party Apple compile | ANE, Apple GPU, CPU | Apple-only | Best on Apple (ANE/unified mem) | Apple-only, most native + efficient path |
| **Qualcomm Genie / QNN** | Vendor NPU compile | Hexagon NPU | Snapdragon-only | Best (purpose-built NPU) | You control the hardware, energy is paramount |

Read the "Energy" and "Platform reach" columns as a tension: the more an option leans into one specific piece of silicon (QNN on Hexagon, Core ML on the ANE), the better its joules-per-token and the narrower its reach. The compiled cross-GPU options (MLC, ExecuTorch) sit in the middle — near-vendor performance on each GPU they target, at the cost of an AOT build per backend. llama.cpp trades the bottom-row energy and the top-row peak performance for the thing neither of them has: a single file you can drop anywhere and forget.

The through-line: **compiled stacks (MLC, ExecuTorch, vendor) trade build complexity for performance and reach; handwritten stacks (llama.cpp) trade some performance for simplicity; vendor NPU stacks trade portability for energy efficiency.** There is no universally best choice — only the best choice for your platform spread, your latency budget, and how much you value a single portable artifact.

## Measuring it honestly on a phone

Reporting on-device LLM performance requires more discipline than reporting server numbers, because the device fights you. The three numbers that matter — TTFT, decode tok/s, and peak memory — each have a "naive way" and an "honest way."

**TTFT.** The naive way times from the API call. The honest way separates *cold* TTFT (first inference after launch, which includes loading and compiling/warming kernels — can be hundreds of ms to seconds) from *warm* TTFT (steady state). Report both, because the first inference a user triggers is *cold*, and a 1.5-second cold TTFT followed by fast warm responses is a very different experience from a uniformly slow model. Always do a throwaway warm-up generation at app start to pay the kernel-compile and allocation costs before the user is watching.

**Decode tok/s.** The naive way averages over a short burst. The honest way uses rolling windows over a *sustained* run (the function above) and reports the throttled steady-state, clearly labeled, alongside the cold-burst peak.

**Peak memory.** The naive way reads the model file size. The honest way measures the *resident process memory* during decode — which is the quantized weights plus the KV cache plus runtime overhead plus the framework's own footprint. On iOS use Instruments (Allocations / VM Tracker) or `os_proc_available_memory()`; on Android use `Debug.getMemoryInfo` / `android.os.Debug` or `adb shell dumpsys meminfo <pkg>`. This number is the one that gets your app killed, so measure it on your *lowest-RAM target device*, not your dev phone.

Here is a minimal harness sketch for the device side (Android, via the MLC `mlc4j` engine; the iOS Swift API is analogous):

```kotlin
// Pseudocode-level Kotlin sketch of an honest on-device measurement
val engine = MLCEngine()
engine.reload(modelPath, modelLib)          // load weights + compiled lib

// 1) WARM UP off-screen so the user never sees cold TTFT
engine.generate(prompt = "warm up", maxTokens = 4)

// 2) Cold-vs-warm TTFT: time to the first streamed token
val tStart = System.nanoTime()
var tFirst = 0L
var produced = 0
engine.chatCompletion(prompt, maxTokens = 600) { token ->
    if (produced == 0) tFirst = System.nanoTime()
    produced += 1
}
val ttftMs = (tFirst - tStart) / 1_000_000.0
// 3) tok/s windows handled inside the callback (see Python version),
//    and peak memory read separately from Debug.getMemoryInfo().
```

For peak memory specifically — the number that gets your app *killed* rather than merely slowed — the honest measurement is the high-water mark of resident memory across the whole worst-case run, sampled while you also know how much headroom the OS is willing to give you before it pulls the trigger. On Android you can poll both your own footprint and the system's remaining budget from a background thread during a max-context, max-generation run:

```kotlin
// Sample peak resident memory + remaining OS budget during a worst-case run
val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
var peakKb = 0L
val sampler = Thread {
    while (generating) {
        val pid = intArrayOf(Process.myPid())
        val info = am.getProcessMemoryInfo(pid)[0]
        peakKb = maxOf(peakKb, info.totalPss.toLong())   // PSS in KB
        Thread.sleep(250)
    }
}
sampler.start()
// ... run the max-context prompt with max_tokens here ...
generating = false; sampler.join()

val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
Log.i("LLM", "peak PSS = ${peakKb / 1024} MB, " +
        "system lowMem=${mi.lowMemory}, threshold=${mi.threshold / (1024*1024)} MB")
```

The two numbers together are the truth: `peak PSS` is what your model actually costs, and `mi.lowMemory` / the kill threshold tell you how close the OS is to reaping you. If `peak PSS` plus the system's own pressure pushes `lowMemory` true on your *lowest-RAM* device, you are one backgrounded camera app away from a crash even though the run "succeeded" on your dev phone. Measure this with the largest prompt and the longest generation your feature allows, with a couple of memory-hungry apps already open in the background, because that is the state a real user's phone is in.

The discipline is the point. A latency number without a "cold or warm? burst or sustained? which device?" qualifier is not a measurement, it is a vibe.

## Results: phone LLM numbers you can size against

Let us put real, defensible numbers on the page so you can size a feature. These are order-of-magnitude figures for MLC-LLM `q4f16_1` models on a recent flagship phone GPU (Snapdragon 8-class / A17-class), batch=1, *sustained* decode (post-throttle), measured the honest way. Treat them as planning estimates, not guarantees — the exact figure depends on the specific SoC, thermal state, and prompt.

![Matrix of phone language model results showing time-to-first-token decode tokens-per-second and peak memory for 1.1B 3B and 7B four-bit models](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-7.png)

| Model (4-bit) | Weights $W$ | TTFT (64-tok prompt) | Decode tok/s (sustained) | Peak process RAM |
| --- | --- | --- | --- | --- |
| TinyLlama 1.1B | ~0.7 GB | ~0.15 s | ~25–30 | ~1.2 GB |
| Llama-3.2-3B | ~1.6 GB | ~0.30 s | ~11–16 | ~2.6 GB |
| Phi-3-mini 3.8B | ~2.0 GB | ~0.40 s | ~9–13 | ~3.0 GB |
| Llama-2/3 7B | ~3.8 GB | ~0.80 s | ~5–8 | ~4.8 GB |

Read these against the decode bound $\text{tok/s} \le B_{\text{eff}}/W$. With $B_{\text{eff}} \approx 40$ GB/s, the 1.1B model's ceiling is ~57 (you see ~28, about half — attention and overhead eat the rest), the 3B's ceiling is ~25 (you see ~14), the 7B's is ~10 (you see ~6). The *ratios* track the inverse-size law beautifully: halving the model roughly doubles tok/s. The peak RAM grows with weights plus the KV cache (which itself grows with context length), and the 7B model at 4.8 GB is genuinely tight on an 8 GB phone — one big background app and the OS may kill you. That is the sizing reality: 7B is the *ceiling* for a phone and risky; 3B is the comfortable sweet spot for an 8 GB device; 1–1.5B is what you reach for when you need real margin or you target mid-range hardware.

### The phone-RAM sizing rule

Here is a back-of-envelope rule you can apply before you ever touch a device. The peak memory of an on-device LLM is approximately:

$$M_{\text{peak}} \;\approx\; W \;+\; M_{\text{KV}} \;+\; M_{\text{rt}}$$

where $W$ is the quantized weight footprint, $M_{\text{rt}}$ is fixed runtime/framework overhead (call it 200–500 MB), and the KV cache is:

$$M_{\text{KV}} \;=\; 2 \cdot n_{\text{layers}} \cdot L_{\text{ctx}} \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot b_{\text{kv}}$$

— the factor 2 for keys and values, over all layers and context positions $L_{\text{ctx}}$, with $n_{\text{kv}}$ key/value heads of dimension $d_{\text{head}}$, at $b_{\text{kv}}$ bytes per element (2 for fp16). For a 3B model with, say, 28 layers, 8 KV heads, head dim 128, fp16 KV, at a 4096-token context: $M_{\text{KV}} = 2 \cdot 28 \cdot 4096 \cdot 8 \cdot 128 \cdot 2 \approx 0.47$ GB. So $M_{\text{peak}} \approx 1.6 + 0.47 + 0.35 \approx 2.4$ GB — consistent with the table. Notice the KV cache scales linearly with context length: doubling $L_{\text{ctx}}$ to 8192 adds another ~0.47 GB, which can be the difference between fitting and getting killed. **The sizing rule: budget for weights + KV-at-your-max-context + ~400 MB overhead, and keep $M_{\text{peak}}$ comfortably under your *lowest-RAM* device's practical limit (≈3–5 GB usable on an 8 GB phone).** If you blow the budget, your levers are: a smaller model (cuts $W$), harder quantization (cuts $W$), a shorter max context (cuts $M_{\text{KV}}$), or quantizing the KV cache itself to int8 (halves $M_{\text{KV}}$ — see [activation and KV-cache quantization](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache)).

#### Stress-testing the sizing rule

What breaks the rule? A few things, and it is worth naming them. **Long context** is the big one: a 32K-context feature on a 3B model would need ~3.7 GB of KV cache alone, blowing past the weight footprint — at that point KV-cache quantization or a sliding-window/streaming attention is mandatory, not optional. **A model with many KV heads** (no grouped-query attention) inflates $M_{\text{KV}}$ several-fold; modern models use GQA precisely to shrink it. **An NPU op fallback** can force tensors to be copied between NPU and CPU memory, transiently spiking peak memory beyond the steady-state estimate. And **the OS taking more than expected** — a phone under memory pressure from other apps gives you less than the nominal budget — means you should leave a safety margin, not size to the theoretical limit. The honest engineering move is to measure $M_{\text{peak}}$ on your worst-case device with your max-context, max-generation prompt, then keep a buffer.

#### Worked example: the energy cost of a generation, and what it does to battery

Memory gets your app killed; energy gets your app uninstalled. Let me put numbers on the battery drain, because "it runs warm" is not a spec. Suppose decode pulls $\approx 4$ W at the GPU during a sustained run (a realistic figure for a 3B model holding a phone GPU near its thermal equilibrium). A typical flagship battery holds about $\approx 15$ Wh ($\approx 4{,}000$ mAh at $\approx 3.85$ V). So a *continuous* generation burns

$$\frac{4 \text{ W}}{15 \text{ Wh}} \;\approx\; 0.27 \text{ /hour} \;=\; 27\% \text{ of the battery per hour of solid decoding,}$$

or roughly $0.4\text{–}0.5\%$ per minute — which matches the throttle table's battery row. Now convert that to *energy per token*. At a sustained 12 tok/s, the GPU produces $12 \times 3600 = 43{,}200$ tokens per hour while drawing 4 W for that hour, so each token costs about

$$\frac{4 \text{ Wh}}{43{,}200 \text{ tok}} \;\approx\; 0.33 \text{ mWh/token.}$$

A 150-token summary therefore costs about $0.33 \times 150 \approx 50$ mWh — a rounding error, perfectly affordable, the user will never notice. But a feature that streams a 2{,}000-token essay costs $\approx 0.66$ Wh, about 4–5% of the battery for *one* generation, and an "always-on assistant" that decodes for ten minutes of every hour would cost $\approx 4.5\%$ per hour just in LLM energy. This is the math that decides whether on-device generation is a tap-to-summarize feature (cheap, ship it) or an always-listening companion (a battery-and-thermal problem that probably wants a smaller model, an NPU, or a server). The energy-per-token number — derivable before you ship — tells you which product you can actually afford to build.

## Case studies: real on-device LLM deployments

A few named, real-world data points to ground the numbers, drawn from the literature and shipped products. As always, exact figures vary by device and build; treat these as the documented order of magnitude.

**MLC-LLM across backends.** The MLC team's own reports show the same compiled model family running across Apple Metal, Android (Adreno/Mali via OpenCL/Vulkan), CUDA, and WebGPU, with the headline result that a single TVM-compiled pipeline reaches competitive tok/s on GPUs where no hand-written kernel previously existed — for example, running a 7B-class model interactively on a phone GPU and running quantized models entirely in the browser via WebLLM with no server. The takeaway is not a single number but the *portability with performance* claim: compile once per backend, run on a GPU zoo. (See the MLC-LLM docs and the WebLLM project.)

**Apple's MLX and on-device foundation models.** Apple's MLX framework and its on-device foundation models (shipped as part of Apple Intelligence) demonstrate the unified-memory, Apple-silicon-native path: a ~3B-class on-device model running efficiently on the iPhone/iPad GPU and ANE, with aggressive quantization (down to ~3.5 bits-per-weight via palettization in some configs) to fit memory and bandwidth budgets. The lesson is that the most efficient Apple path is a first-party, hardware-co-designed stack — at the cost of being Apple-only.

**Google AI Edge / MediaPipe LLM and Gemma.** Google's on-device LLM path (the MediaPipe LLM Inference API, now under Google AI Edge / LiteRT) ships Gemma-class models to phones with a GPU/CPU runtime and documented interactive tok/s on flagship Android devices. It is the "first-party Android" analog to Apple's stack and integrates with the broader LiteRT (TFLite) ecosystem covered in [from model to deployable artifact](/blog/machine-learning/edge-ai/from-model-to-deployable-artifact).

**Qualcomm on-device generative AI.** Qualcomm has publicly demonstrated multi-billion-parameter LLMs (and even Stable Diffusion) running entirely on Snapdragon via the Hexagon NPU through the QNN/Genie stack, emphasizing energy efficiency — the NPU does the matrix math at a fraction of the power a GPU would draw. The trade-off is the Snapdragon-only portability and a heavier, less open toolchain. This is the vendor-NPU end of the spectrum: best energy, narrowest reach.

The pattern across all four: the *compiled or hardware-co-designed* stacks win on performance and efficiency, and the price is always either build complexity (MLC, ExecuTorch) or platform lock-in (MLX/Core ML on Apple, QNN on Qualcomm). The portable, simple option (llama.cpp) is the fallback when those costs are not worth paying.

## Stress test: the three ways an on-device LLM falls over

Sizing rules and benchmark tables describe the happy path. The engineering happens at the edges, where the device pushes back. There are three classic failure modes for an on-device LLM, each tracing to one of the four mobile constraints, and a team that ships without having deliberately *driven the model into* each of them will meet all three in the field instead — in a one-star review. Let us push the 3B-on-a-flagship baseline past each cliff and watch what breaks.

**Failure one: sustained decode throttles past the usable floor.** Take the model that benchmarked at 18 tok/s cold and ask it for a 1{,}500-token essay — a real, long generation, not a 20-second burst. The throttle curve we derived plays out exactly: the high plateau holds for the first ~20 seconds (~360 tokens), the knee hits, and the model settles onto its thermal-equilibrium plateau at ~11 tok/s. But the essay takes over two minutes, and by the second minute the chassis is hot enough that a *deeper* throttle threshold trips — some governors have a second, more aggressive step for prolonged thermal load — and tok/s can sag further toward 8–9. If your "fluid reading" floor is 10 tok/s, the generation that *started* comfortably fast crosses below the usable line partway through, and the user watches the text visibly slow down as they read it, which is worse than uniform slowness because it feels like the app is failing in real time. The fix is not a faster kernel; the kernel is already saturated. The fix is a product decision: cap generation length so the chip never reaches the deep-throttle plateau, drop to a 1.5B model for headroom, or chunk a long generation with brief pauses that let the die shed heat. You cannot out-engineer thermodynamics; you can only design around it.

**Failure two: the long-prompt TTFT blowup.** Now stress the *other* phase. Feed the same model a 4{,}000-token document and ask "what's the key takeaway?" Decode is fine — the answer is short. But prefill must process 4{,}000 tokens, and at $R_{\text{prefill}} \approx 350$ tok/s that is $4000/350 \approx 11.4$ seconds of TTFT before a single word appears. Eleven seconds of a blank, spinning screen reads as a *crash* to most users; they will background the app before the first token lands. Worse, recall the quadratic attention term: at 4K tokens the $L^2$ work is no longer negligible, so the real TTFT is *above* the linear estimate, not below it. And if the feature later pushes 16K tokens, prefill compute roughly quadruples from the linear term while the attention term grows fourfold, and TTFT can blow past 40 seconds — at which point the feature is simply broken on-device regardless of how fast decode is. This is the failure that catches RAG-on-phone teams: they prove the concept on a 200-token prompt, ship a retriever that stuffs 4K tokens of context, and discover prefill was the whole cost all along. The fixes are all prefill-side: cap the retrieved context, summarize-then-answer in two cheap passes instead of one expensive one, cache the prefill of a stable system prompt across turns, or stream a "reading your document…" affordance so the wait is legible rather than mysterious.

**Failure three: RAM pressure kills the app.** The third cliff is the quietest and the most dangerous, because it produces no slow-down warning — just a sudden death. Push context length up: a 3B model with an 8{,}192-token context needs $W \approx 1.6$ GB plus $M_{\text{KV}} \approx 0.94$ GB plus ~0.4 GB runtime $\approx 2.9$ GB resident. On a dev phone with 12 GB and nothing else open, that is fine. On a *user's* 8 GB phone with a browser, a maps app, and a music player in the background, the OS's practical budget for your foreground app might be 3 GB — and the moment your peak PSS crosses it during the longest generation, iOS issues a Jetsam kill or Android's low-memory killer reaps you, with no exception, no log the user sees, just the app vanishing. The cruelty is that it is *intermittent*: it only happens when the prompt is long *and* the phone is already under pressure, so it is nearly impossible to reproduce on a clean test device and shows up as irreproducible crash reports. The defenses stack: quantize the KV cache to int8 (halving the 0.94 GB to ~0.47 GB), cap max context to what you actually need, drop to a smaller model on low-RAM devices detected at launch, and — non-negotiably — measure peak PSS on your lowest-RAM target *with background apps open*, using the sampler from the measurement section, then size to that worst case with a buffer rather than to the dev phone's comfortable headroom.

The meta-lesson across all three: each failure lives in a *different* phase or constraint (decode/thermal, prefill/compute, memory/OS), so each needs a *different* fix, and the only way to know which one you are about to hit is to drive the model deliberately into each cliff during testing — long generation for throttle, long prompt for TTFT, long context on a loaded low-RAM phone for the kill. The model that "worked on my phone" failed on all three counts on someone else's.

## When to reach for a compiled mobile stack (and when not to)

Time for the decisive part. The figure is a decision tree; the prose is the argument.

![Decision tree for choosing a mobile LLM stack based on cross-platform reach then on whether a vendor NPU GPU peak or CPU simplicity best fits the constraint](/imgs/blogs/running-llms-locally-mlc-and-mobile-stacks-8.png)

**Reach for MLC-LLM (or another compiled cross-GPU stack) when** you need *broad GPU support with peak performance* — shipping to a wide range of Android GPUs, or to iOS *and* Android *and* the browser from one model source. MLC's compile-per-backend model is exactly right when the GPU diversity is your problem and you want autotuned kernels rather than a generic fallback. It is also the answer when you specifically want the *browser* (WebLLM / WebGPU) as a zero-install distribution channel. The cost you accept: an AOT compile step per target and a heavier build/integration than dropping in a single file.

**Reach for llama.cpp on the phone when** simplicity and portability matter more than peak speed: a small model, a loose latency budget, a single GGUF artifact you already use on the laptop, or a CPU-only target. It is the lowest-friction way to get *an* answer on-device, especially on Apple where its Metal path is solid. It will usually be slower than a tuned MLC Vulkan/Metal path on Android GPUs, but "slower and shipping today" often beats "faster and still integrating the compiler toolchain."

**Reach for a vendor NPU SDK (Qualcomm Genie/QNN, etc.) when** energy efficiency is paramount and you control the hardware — you ship to a known SoC (one phone model, a kiosk, an embedded device) and you need the lowest power draw and longest battery life. The NPU will beat the GPU on joules-per-token when it can run your model, but you pay in portability (one vendor) and toolchain pain (heavier, op-coverage gaps for new architectures, CPU fallback when an op is unsupported).

**Reach for Apple's MLX / Core ML when** you are Apple-only and want the most efficient, most native path — MLX for GPU-bound LLMs on Apple silicon, Core ML when the ANE can take your model for the best power efficiency. You give up cross-platform reach entirely.

**And do NOT over-engineer.** If a 1.5B model in plain llama.cpp on the CPU already hits your latency and quality bar, do not stand up a TVM compile pipeline to claw back tok/s you do not need. The compiled stacks earn their complexity when you are *fighting* the device — wide GPU diversity, a tight latency budget, a peak-performance requirement. If you are not fighting it, the simple path is the correct path. The most expensive mistake in on-device ML is solving a performance problem you do not have. This is the same discipline the whole series preaches and that the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) capstone ties together: measure first, reach for the heavier lever only when the lighter one misses the target.

## Putting it together: a full on-device LLM decision

Let me walk a complete, realistic decision end-to-end, the way you would actually reason about it on a real product, to show how every piece above composes.

**The brief.** You are adding an offline "summarize this article" feature to a reading app on both iOS and Android. Articles are up to ~1,500 tokens; summaries are ~150 tokens. It must work in airplane mode. Target devices include 8 GB Android phones and recent iPhones. Latency goal: the summary should *start* within ~2 seconds and *finish* within ~15 seconds, and it must not get the app killed for memory.

**Step 1 — size the model.** Quality-wise, a 3B instruction-tuned model summarizes well. Memory check via the sizing rule: a 3B at 4-bit is $W \approx 1.6$ GB; the KV cache for a 1,500-token prompt + 150-token output (≈1,650 ctx) at 28 layers, 8 KV heads, head dim 128, fp16 is $2 \cdot 28 \cdot 1650 \cdot 8 \cdot 128 \cdot 2 \approx 0.19$ GB; plus ~0.4 GB runtime. $M_{\text{peak}} \approx 2.2$ GB — comfortably under a 3–5 GB budget on an 8 GB phone. Good, 3B fits. (A 7B at 4.8 GB+ would be too risky on 8 GB; a 1.1B might summarize too weakly. 3B is the Pareto pick.)

**Step 2 — check the latency budget against prefill/decode.** The 1,500-token prompt means prefill dominates TTFT. At $R_{\text{prefill}} \approx 350$ tok/s, TTFT $\approx 0.05 + 1500/350 \approx 4.4$ s — that *blows the 2-second start budget*. This is the prefill insight earning its keep: my problem is not generation speed, it is prefill on a long prompt. Fixes: (a) feed the article in chunks and start summarizing incrementally; (b) use a model/runtime with a faster prefill kernel; (c) relax the start budget; or (d) truncate/pre-filter the article to fewer tokens. Generation itself: 150 tokens at a sustained ~12 tok/s $\approx 12.5$ s — within the 15-second finish budget, but only just, *after* throttling, so I should cap summary length and not let it run for minutes.

**Step 3 — pick the stack.** I need iOS *and* a wide range of Android GPUs, peak performance to fight the prefill problem, and I do not control the hardware. That points squarely at **MLC-LLM**: compile `q4f16_1` for `iphone` (Metal) and `android` (Vulkan/OpenCL) from the same converted weights, autotuned per backend. If I also wanted a web version of the reader, WebLLM would let me reuse the same compiled-to-WebGPU model — a strong argument for MLC over llama.cpp here. (If this were Apple-only, I would seriously consider MLX for the most native path; if I shipped to a single Snapdragon device, QNN for the energy win.)

**Step 4 — measure honestly and design for the throttle.** Warm up at app launch to hide cold TTFT. Measure *sustained* decode on the lowest-RAM Android target, not my dev phone. Report TTFT (cold and warm), sustained tok/s, and peak RAM on the worst-case device. Because summaries are short (~150 tokens), the chip never reaches the deep-throttle plateau, so the cold-burst-ish rate is roughly what users feel — a happy accident of a short-generation feature. If the feature were long-form chat, I would design around the sustained number instead.

That is the whole reasoning chain: size to RAM with the formula, diagnose latency by phase (prefill vs decode), pick the stack by platform reach and performance need, and measure the truth (sustained, worst-case device, peak memory). Every step used a tool from this post.

## Key takeaways

- **A phone needs a *compiled* stack, not just a smaller model.** Mobile GPUs are a fractured zoo (Metal, Vulkan, OpenCL, WebGPU); MLC-LLM compiles the model through TVM Unity into autotuned per-backend kernels, recovering performance that generic kernels leave on the table — the opposite philosophy to llama.cpp's hand-written kernels.
- **The MLC flow is three stages:** `convert_weight` (target-independent 4-bit quantization), `gen_config` (template/context/sampling), and `compile` (per-target GPU kernel library). One weight conversion, *N* compiles — one per backend you ship.
- **Prefill and decode are different physics.** Prefill (reading the prompt) is parallel and *compute-bound* — it sets time-to-first-token, which grows with prompt length. Decode (generating tokens) is serial and *memory-bound* — it sets steady tok/s, which is roughly independent of prompt length.
- **Decode speed is a bandwidth race:** $\text{tok/s} \lesssim B_{\text{eff}}/W$. Halving the model (or quantizing 8→4 bit) roughly doubles tok/s, because you stream half the weight bytes per token. This is *the* reason compression matters so much for on-device LLMs.
- **TTFT and tok/s are separate metrics.** "Slow to start" (long prompt → prefill) and "slow to type" (memory-bound decode) are different problems with different fixes. Shrink the prompt for TTFT; shrink the model for tok/s.
- **Thermal throttling makes cold benchmarks lie.** Sustained decode on a fanless phone drops 30–50% below the cold-burst rate after ~60–120 s. Report *sustained* numbers, measured with rolling windows, or you will ship a number your users never see.
- **Size to RAM with the formula** $M_{\text{peak}} \approx W + M_{\text{KV}} + M_{\text{rt}}$, where the KV cache grows linearly with context length. Budget for max-context KV, keep peak under your lowest-RAM device's practical limit (≈3–5 GB on 8 GB), and quantize the KV cache or shorten context if you blow it. 3B is the phone sweet spot; 7B is the risky ceiling.
- **Pick the stack by constraint:** MLC for broad GPU reach + peak performance (and the browser via WebLLM); llama.cpp for simplicity and a single portable artifact; MLX/Core ML for the most native Apple path; vendor NPU SDKs (Qualcomm Genie/QNN) for best energy on hardware you control. Do not stand up a compiler pipeline for a performance problem you do not have.

## Further reading

- **MLC-LLM documentation and the TVM Unity stack** — the official guide to `convert_weight` / `gen_config` / `compile`, the supported backends (Metal, Vulkan, OpenCL, WebGPU, CUDA), and the iOS/Android reference apps; the foundation of everything in this post.
- **WebLLM** — MLC's browser project that runs the same compiled models entirely client-side over WebGPU; the zero-install distribution channel.
- **ExecuTorch (PyTorch) on-device LLM guide** — the `torch.export` → ExecuTorch program → delegate flow and the dedicated LLM/KV-cache export path.
- **Apple MLX and `mlx-lm`** — Apple's unified-memory array framework and LLM package, the most native path on Apple silicon; plus Apple's on-device foundation-model technical reports on quantization and ANE.
- **Google AI Edge / MediaPipe LLM Inference API and Gemma** — the first-party Android on-device LLM runtime within the LiteRT ecosystem.
- **Qualcomm AI Engine Direct (QNN) and Genie** — the Hexagon NPU LLM stack and Qualcomm's on-device generative-AI demonstrations; the vendor-NPU energy-efficiency path.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever Pareto frame, [running LLMs locally with llama.cpp and GGUF](/blog/machine-learning/edge-ai/running-llms-locally-llama-cpp-and-gguf) for the laptop counterpart, [ML compilers and autotuning with TVM, MLIR, and XLA](/blog/machine-learning/edge-ai/ml-compilers-and-autotuning-tvm-mlir-xla) for the compilation machinery MLC rides on, [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for honest measurement, and the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) capstone that ties the levers together.
