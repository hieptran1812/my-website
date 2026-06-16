---
title: "Why optimize AI models for the edge: latency, privacy, cost, and the four levers"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A first-principles tour of why on-device AI beats the cloud on latency, privacy, cost, reliability, and energy — and the four levers (quantization, pruning, distillation, efficient architecture) you pull to make a model fit, with runnable code to baseline your own model today."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "on-device-inference",
    "inference",
    "efficient-ml",
    "tinyml",
    "latency",
    "energy-efficiency",
    "pareto-frontier",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/why-optimize-ai-models-for-the-edge-1.png"
---

A few years ago I watched a team demo a beautiful keyword-spotting model. In the cloud, on an A100, it was flawless: 99.1% accuracy, a crisp confusion matrix, the works. The product was a battery-powered doorbell that had to wake on the phrase "open the gate" and run for a year on a coin cell. The model was 11 MB of fp32 weights and wanted 240 MFLOPs per inference. The target was a Cortex-M4 with 256 KB of SRAM and a power budget measured in microwatts. The model did not fit. It did not fit by two orders of magnitude. The demo that wowed the room was, on the actual device, completely impossible to ship.

That gap — between the model that works in the notebook and the model that fits on the thing in your hand — is the entire subject of this series. It shows up everywhere: the dashcam that has to flag a pedestrian in under 30 ms with no signal in a tunnel; the hearing aid that must denoise speech in real time on a 1 mWh budget; the phone keyboard that predicts your next word without shipping every keystroke to a server; the factory vibration sensor that has to detect a failing bearing offline, forever, on a battery. In every one of these the cloud model is the easy part. Making it run on the device — fast enough, small enough, private enough, cheap enough — is the engineering.

This post is the front door to the whole "Optimizing AI Models for the Edge" series. By the end of it you will be able to: explain, with numbers, *why* on-device inference beats a cloud round-trip on five distinct axes; place any target device on the cloud-to-microcontroller spectrum and know its rough compute and memory budget; name the four levers you pull to shrink a model and what each one costs you; reason about the accuracy–efficiency Pareto frontier as the master mental model the rest of the series returns to again and again; and — most concretely — run a tiny script *today* that measures your own model's size and single-inference latency so you have a baseline to optimize against. The edge spectrum that frames the rest of this discussion is laid out in the figure just below.

![Stacked diagram of the edge compute spectrum from cloud GPU down to a Cortex-M microcontroller, with compute and memory budgets shrinking by roughly an order of magnitude at each tier](/imgs/blogs/why-optimize-ai-models-for-the-edge-1.png)

I want to be honest from the first page: the edge is not always the right answer, and a big chunk of this post is about *when not to bother*. But when it is the right answer, the difference between "we shipped it" and "the demo never made it to a product" is exactly the set of techniques this series teaches. Let us start with why anyone would put themselves through this in the first place.

## Five reasons on-device wins (each one quantified)

When people argue for on-device AI they usually wave at "privacy" and "latency" and move on. That is not good enough to make engineering decisions with. There are five distinct reasons, each with a different shape, each measurable, and you should be able to say which ones your product actually cares about. The figure below lays them out side by side; the rest of this section puts numbers on each.

![Matrix figure listing the five reasons on-device wins, mapping each reason to the gain it buys and a concrete example application](/imgs/blogs/why-optimize-ai-models-for-the-edge-2.png)

### Latency: the speed of light is not your friend

A cloud round-trip is not free even on a perfect network. You pay for the request to leave the device, cross the radio link, traverse the internet to the data center, queue, run, and come all the way back. On a good 4G connection the radio leg alone is 30–50 ms each way; on a congested cell or a far region it is routinely 100–300 ms; satellite or a weak signal pushes past 500 ms. None of that is compute. It is the network tax you pay *before* the model has done a single multiply.

On-device, that tax is zero. A well-optimized vision model on a phone NPU (neural processing unit — the dedicated matrix-multiply accelerator that ships in modern phone SoCs, or systems-on-chip) runs an inference in single-digit milliseconds. A keyword spotter on a microcontroller runs in well under a millisecond. The relevant comparison is not "cloud is a bit slower," it is "cloud is impossible for this workload":

- Augmented reality has to render at 60–90 frames per second, which is an 11–16 ms frame budget for *everything*, perception included. A 200 ms cloud round-trip is 12–18 frames of lag. The hand you reach for is no longer where the headset thinks it is.
- Automatic speech recognition (ASR) for live captioning needs to keep up with speech, roughly 150–200 ms of acceptable lag end to end. Burn 300 ms on the network and you are already over budget before transcription starts.
- A control loop — a drone stabilizing, a robot arm avoiding a collision — needs deterministic single-digit-millisecond response. A network round-trip is not just slow, it is *jittery*, and jitter is poison for control.

The latency argument is the one that most often forces the edge. If your loop budget is below the network round-trip time, there is no cloud architecture that saves you. You compute locally or you do not compute in time.

#### Worked example: an AR frame budget

Make the AR case concrete. A headset rendering at 72 frames per second has a frame budget of $1000 / 72 \approx 13.9$ ms for *everything* in that frame: read the cameras, run hand and surface detection, render the scene, and push pixels to the display. Perception cannot eat the whole budget — call it a 6 ms slice for the neural network. Now compare the two architectures:

- **Cloud perception.** Capture the frame, compress it, send it over the radio (say a good 25 ms each way, 50 ms round-trip on a healthy network), run the model server-side, and return the result. Total: ~55 ms before the renderer can even start. That is *four frames* of latency. The virtual object lags four frames behind your hand, which the human vestibular system reads as wrongness and, often, nausea. There is no server fast enough to fix this; the speed of light and the radio stack set the floor.
- **On-device perception.** Run an optimized detector on the headset's NPU in, say, 4 ms. It fits the 6 ms slice with room to spare, the renderer gets its result inside the same frame, and the object tracks your hand. The model that fits 4 ms is almost certainly a quantized, efficient-architecture model — which is to say, the levers are what made the experience possible at all.

The arithmetic is brutal and clarifying: when the loop budget is single-digit milliseconds, the network round-trip is not a tax you optimize, it is a wall you cannot climb. The edge is not the better choice here; it is the only choice, and the engineering question collapses to "how small and fast can we make the model and still hit the accuracy bar."

### Privacy: data that never leaves cannot leak

The strongest privacy guarantee is the one you do not have to trust anyone to keep: the raw data never leaves the device. On-device inference means the microphone audio, the camera frames, the keystrokes, the health-sensor stream are processed where they are captured and only a result — a label, a transcript, an embedding, an alert — ever travels, if anything travels at all.

This is not a soft "users feel better" benefit. It is a hard regulatory and architectural one:

- Under GDPR, personal data you never collect on a server is data you never have to store, secure, audit, or honor a deletion request for. On-device processing can move an entire feature out of the scope of a data-protection regime.
- Under HIPAA, a hearing aid or a glucose monitor that classifies on-device is handling protected health information without that information ever entering a system you must certify.
- On-device speech (the wake-word and dictation models on modern phones) is the canonical example: the always-listening audio buffer is processed locally, and only after a wake word fires does anything potentially leave. The privacy posture is categorically different from streaming every second of ambient audio to a server.

Privacy is binary in a way latency is not: either the sensitive bytes left the device or they did not. On-device makes the answer "they did not," which is a guarantee no amount of server-side encryption can quite match.

There is also a trust and attack-surface angle that engineers feel more keenly than marketers. Data you never collect cannot be breached, subpoenaed, sold by a future owner of the company, or leaked by a misconfigured bucket. The most defensible security posture for sensitive inference is to make the sensitive computation physically incapable of leaving the user's hardware, and on-device inference is how you get there. It also sidesteps an entire category of compliance work: data-residency rules that require certain data to stay in a country are trivially satisfied when the data never leaves the phone in the user's pocket. The cost, of course, is that the model — and the lever-driven optimization that made it fit — now lives on a device you do not control, which raises model-theft and tampering concerns of its own. That trade is real and we will treat it honestly in the MLOps track, but it does not undo the central point: the strongest privacy guarantee is the byte that was never transmitted.

### Cost: the cloud bills you per inference, forever

Cloud inference has a beautiful property for the vendor and a painful one for you: it is a recurring per-call cost. Every inference is a fraction of a cent that you pay every time, forever, for as long as the product lives. On-device inference is amortized: the silicon is paid for once (you, or your user, already bought the phone), and every inference after that is effectively free at the margin. Let us put real numbers on it.

#### Worked example: a 10-million-user app

Suppose you ship an app to 10 million users and each one triggers, on average, 20 inferences a day — a photo classifier, a smart-reply suggester, whatever. That is 200 million inferences a day, or 6 billion a month.

If you serve those from the cloud, even a cheap small-model endpoint runs around \$0.20 per million inferences once you include the GPU instance amortization, autoscaling overhead, and the load balancer in front of it (some managed endpoints are pricier; this is a deliberately optimistic figure). Then:

$$\text{monthly cost} = 6{,}000 \text{ million} \times \$0.20/\text{million} = \$1{,}200 \text{ per month}.$$

That sounds small until you realize it is *recurring and grows with usage*: \$14,400 a year, climbing every time the app gets more popular, plus the egress bandwidth for shipping inputs up and results down. If each input is a 200 KB image, 6 billion uploads a month is 1.2 PB of ingress; even at a few cents per GB that is another five-figure monthly line item. Success makes the bill worse.

Run the same model on-device and the marginal cloud cost is \$0. You traded a recurring operational expense for a one-time engineering cost: the work of shrinking the model to fit the phone. That engineering is what this series is about, and the trade is overwhelmingly favorable at scale. The break-even is a real calculation — a low-volume internal tool that runs a thousand inferences a month will never justify the optimization effort, and we will be honest about that in the "when the cloud wins" section. But for anything that ships to a large install base, the per-call cloud meter is a powerful reason to move compute to the device.

The economics get sharper still when you remember that cloud inference cost is *coupled to success*. A feature that takes off does not just earn more revenue; it incurs proportionally more inference bill, every single month, in perpetuity. On-device inference *decouples* cost from usage: a feature that goes viral costs you nothing extra at the margin because the compute runs on hardware your users already own and power. There is a genuine economy-of-scale inversion here — the cloud has economies of scale on the *supply* side (a hyperscaler runs GPUs cheaper than you can), but the edge has a structural advantage on the *demand* side, because the per-call cost is zero no matter how many calls there are. For a consumer product with millions of users and high per-user inference frequency, that structural zero is worth more than any volume discount a cloud vendor will give you, and it is why the most cost-sensitive high-volume features — keyboards, photo libraries, on-device search — went to the edge first and stayed there.

### Reliability and offline: no network, no problem

A cloud-dependent feature has a single point of failure that you do not control: the network. On-device inference simply does not care. The dashcam in the tunnel, the translation app on the international flight, the agricultural sensor in a field with no cell coverage, the medical device in a hospital basement — all of them work because the model is local.

It is worth being precise about what "offline" buys, because it is not only the rare no-signal case. Connectivity is not a binary; it is a spectrum of degraded states — a flaky cell handoff, a saturated stadium Wi-Fi, a captive portal that swallows your requests, a backhaul outage at the carrier. A cloud feature has to handle *all* of those gracefully or it feels broken, and "handle gracefully" usually means building a local fallback anyway. Once you have built the local fallback, you have built the on-device model, and the cloud path becomes the redundant one. Designing for the edge from the start often turns out to be simpler than designing a robust cloud path that degrades well, precisely because the edge path has no degraded states to handle — it either runs or the device is off. That inversion, where the "harder" on-device engineering actually yields the *simpler* and more robust system, is one of the quiet reasons experienced teams reach for the edge more often than newcomers expect.

There is a subtler reliability win even when the network *is* available: tail latency. A cloud service's median response might be a fine 80 ms, but the p99 (the 99th-percentile latency — the slowest 1% of requests) can be 800 ms or worse when the network is congested, the endpoint is autoscaling, or a noisy neighbor is hammering the same instance. For a user-facing feature, the p99 is what people remember, because the slow ones are the ones they notice. An on-device model has a tight, predictable latency distribution: there is no network, no queue, no autoscaling cold-start. Your p50 and p99 sit close together, and that consistency is a feature in itself.

The reliability argument compounds with scale in a way that is easy to underestimate. A cloud-served feature has a dependency chain — the device, the radio, the carrier, the public internet, your load balancer, your autoscaler, your inference fleet — and the feature is only as available as the *product* of every link's availability. Five links at 99.9% each multiply to roughly 99.5% end-to-end, which is over three hours of feature downtime a month that you did not directly cause and cannot directly fix. An on-device model collapses that chain to one link: the device, which the user is already holding and which does not have a bad-network day. For a safety-relevant feature — a collision warning, a medical alert — that collapse from a long fragile chain to a single local computation is not a nicety, it is the difference between a feature you can certify and one you cannot.

### Energy: moving a bit can cost more than computing on it

This is the reason people most often get wrong, and it is the one with the most beautiful physics behind it, so we will return to it in full rigor in the science section. The short version: transmitting data over a radio is expensive. Sending a kilobyte over a cellular or Wi-Fi link can cost on the order of tens of millijoules once you account for the radio waking up, associating, transmitting, and the tail-energy of keeping the radio powered afterward. Running a small model locally can cost a fraction of that.

For a battery-powered always-on sensor, this flips the entire architecture. A naive design wakes the radio and streams sensor data to the cloud for classification. An on-device design runs a tiny model locally and only transmits when it has something to say — a detected event, an anomaly. The local-compute design can use *less* total energy than the transmit-everything design, sometimes dramatically less, because the radio is the energy hog and you have stopped feeding it. That is the counterintuitive crux: for many edge workloads, *computing is cheaper than communicating*. We will quantify exactly why below.

The energy argument also reframes what "optimization" even means for a battery device. On a plugged-in server, you optimize for throughput per dollar; on a battery sensor, you optimize for inferences per joule, and the two objectives can point in different directions. A model that is slightly larger but lets you keep the radio asleep for an extra hour is the *more* energy-efficient choice even though it does more compute, because the radio's wake-transmit-tail cost dwarfs the extra multiplies. This is why the energy lever is not simply "make the model smaller" — it is "make the whole system transmit less," and the smaller model is the means, not the end. The series' TinyML track lives entirely inside this constraint, where every kilobyte of model and every milliwatt of duty cycle is fought over, and the four levers are wielded in service of a battery-life number, not an accuracy leaderboard.

## What "edge" actually spans

"The edge" is not one thing. It is a spectrum that runs from a beefy edge server sitting in a telco closet all the way down to a microcontroller with less RAM than a 1990s PC. Where your target lands on this spectrum changes everything about which optimization levers matter and how hard you have to pull them. The figure at the top of this post (the edge spectrum) is the map; here is the territory in order-of-magnitude terms.

- **Cloud GPU.** Hundreds to ~1000 TFLOPs (trillions of floating-point operations per second), 40–80 GB of high-bandwidth memory, effectively unlimited power and cooling. This is where the model is trained and where it runs unoptimized. It is the reference point we are running *away* from.
- **Edge server.** A rack-mounted box or an on-prem appliance: one or a few datacenter or workstation GPUs, tens of TFLOPs, 16–64 GB RAM, wall power. Used for low-latency regional serving, video analytics on many camera streams, on-prem privacy. Still generous, but you are starting to care about throughput per watt.
- **Laptop / desktop SoC.** A few TFLOPs from an integrated GPU or a unified-memory chip like Apple's M-series, 8–32 GB RAM, mains or a large battery. This is where "local LLM" lives today: a quantized 7B–13B model runs comfortably here. Memory bandwidth, not raw FLOPs, is usually the limiter.
- **Phone / mobile SoC with NPU.** This is the highest-volume edge target on earth. A modern phone has a CPU, a GPU, and an NPU (the neural accelerator) delivering on the order of 5–50 TOPS (trillions of *integer* operations per second — note TOPS is usually quoted for int8, not fp32), with 4–12 GB of RAM shared across the whole system and a battery you must not drain. The NPU loves int8 and is often useless for anything it cannot accelerate, which is why quantization is the headline lever for phones.
- **Single-board computer.** A Raspberry Pi 5 (CPU-only, a few GFLOPs of usable NN throughput) up to an NVIDIA Jetson Orin Nano (around 40 TOPS int8 from its GPU), 2–8 GB RAM, a \$35–\$500 board, often on a real power budget. The workhorse of robotics, smart cameras, and hobby-to-industrial prototypes.
- **Microcontroller (MCU).** A Cortex-M class chip running at 80–480 MHz, with KILObytes to a couple of megabytes of SRAM and a few hundred KB to a couple of MB of flash, drawing milliwatts or microwatts. No operating system in the usual sense, no malloc you want to rely on, no GPU. This is TinyML territory, and it is a completely different sport: a model here is measured in *kilobytes*, and you fight for every byte of SRAM in the tensor arena (the single pre-allocated buffer that holds all the model's activations). The doorbell from the opening story lives here.

The single most useful habit you can build is to *name your target tier first* and let it dictate the optimization budget. A model that is "small enough" for a Jetson is wildly too big for a Cortex-M. This series has a dedicated deep-dive on the hardware itself — see the sibling post [the edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for the SoC, NPU, and MCU details — but for now the takeaway is the staircase in the figure: roughly an order of magnitude of compute and memory disappears at every step down, and your model has to descend the same staircase.

## The core tension: models grew faster than edge hardware

Here is the uncomfortable arithmetic that makes all of this necessary. Model sizes and FLOP counts have grown explosively, while edge device memory and compute have grown gently. The two curves diverge, and the gap between them is the problem we are paid to close. The figure below makes the mismatch concrete.

![Before-and-after style figure contrasting model weight sizes such as a 14 GB 7B language model against edge memory budgets like 6 to 8 GB of phone RAM and 256 KB of microcontroller SRAM](/imgs/blogs/why-optimize-ai-models-for-the-edge-6.png)

Take a 7-billion-parameter language model, the kind that is now "small" by frontier standards. In fp16 (16-bit floating point, 2 bytes per parameter) its weights alone are:

$$7 \times 10^{9} \text{ params} \times 2 \text{ bytes} = 14 \times 10^{9} \text{ bytes} = 14 \text{ GB}.$$

A flagship phone has 8–12 GB of RAM *total*, shared with the operating system, the app, and everything else. The model's weights do not fit, full stop, before you have allocated a single byte for the KV cache (the per-token attention memory that grows with context length) or activations. Now quantize those weights to 4 bits and the picture changes:

$$7 \times 10^{9} \text{ params} \times 0.5 \text{ bytes} = 3.5 \times 10^{9} \text{ bytes} = 3.5 \text{ GB}.$$

Suddenly it fits, with room for the runtime. That single 4× reduction is the difference between "impossible on a phone" and "ships in a consumer app." This is not a hypothetical — it is exactly why a 4-bit GGUF (the quantized weight format used by `llama.cpp`) of a 7B model is the standard way people run local LLMs on laptops and high-end phones today.

The same story plays out for vision. A ResNet-50 is about 25.6 million parameters, ~98 MB in fp32, ~4 GFLOPs per 224×224 image. That is trivial on a GPU and fine on a phone, but on a Cortex-M7 at 480 MHz with a single-issue pipeline you have *megaflops*, not gigaflops, of headroom per inference if you want real-time response. The model wants 4 billion operations; the chip can do a few hundred million per second within a reasonable latency budget. The ratio is the wall you hit. You do not optimize because optimizing is fun — you optimize because the model and the device are separated by one to four orders of magnitude and *something* has to give.

The good news, and the reason this series exists, is that models are spectacularly overparameterized and over-precise. They are trained in fp32 for the convenience of gradient descent, not because inference needs 32 bits. They have far more parameters than the task requires, because overparameterization helps optimization, not deployment. The slack between "what training needed" and "what inference needs" is exactly the budget the four levers spend.

## The four levers (the spine of this series)

Almost everything in model optimization is one of four moves, plus a substrate that makes the moves real and a discipline that tells you whether they worked. Each lever attacks a different source of cost, and — crucially — they *compose*: a production model on a phone is usually distilled *and* quantized *and* running on a compiled, hardware-specific runtime. The figure below is the tree we will return to in every post of the series.

![Tree figure rooting at shrink the model and branching into the four levers quantization, pruning, distillation, and efficient architecture plus the compiler runtime substrate and the profiling discipline](/imgs/blogs/why-optimize-ai-models-for-the-edge-3.png)

### Lever 1: Quantization (spend bits)

Quantization lowers the numerical precision of the weights and often the activations: fp32 (32-bit float) down to fp16, int8, int4, or lower. The headline win is memory and bandwidth — int8 is 4× smaller than fp32 — and because most edge accelerators have dedicated int8 hardware, you usually get a speed-up too. The headline cost is accuracy: representing a weight with fewer bits introduces rounding error, and if you are careless that error compounds through the network into visible accuracy loss.

**What it cuts:** model size (4× for int8, 8× for int4), memory bandwidth, and often latency and energy. **What it costs:** a small accuracy drop (well under 1% for a well-calibrated int8 CNN, more for aggressive int4 on sensitive models) and engineering effort to calibrate or fine-tune. Quantization is the first lever almost everyone reaches for because the ratio of win to pain is the best of the four. We devote multiple posts to it — post-training quantization, quantization-aware training, the integer arithmetic itself, and LLM-specific schemes like GPTQ and AWQ.

### Lever 2: Pruning and sparsity (introduce zeros)

Pruning removes weights — sets them to exactly zero — on the theory that a trained network has many connections that contribute almost nothing. Magnitude pruning zeroes the smallest weights; structured pruning removes whole channels, heads, or filters. The famous *Lottery Ticket Hypothesis* (Frankle and Carbin, 2018) argues that a dense network contains a small sparse subnetwork that, trained in isolation from the right initialization, matches the full network's accuracy — which is a deep statement about *why* pruning works at all.

**What it cuts:** parameter count and, for *structured* pruning, real FLOPs and latency. **What it costs:** accuracy (usually recoverable with fine-tuning), and — the big asterisk — *unstructured* sparsity rarely speeds anything up on commodity hardware, because a CPU or GPU still does the multiply, it just multiplies by zero. You need hardware that exploits the zeros (NVIDIA's 2:4 sparse tensor cores, for example) or you need structured pruning that actually shrinks the dense tensor. We have a full post on this, because the gap between "fewer parameters on paper" and "faster on the actual chip" is where people get burned.

### Lever 3: Distillation (train a small model with a big one)

Knowledge distillation (Hinton et al., 2015) trains a small "student" model to mimic a large "teacher," not just on the hard labels but on the teacher's soft probability distribution, which carries far richer information ("this is a 7, but it looks a bit like a 1"). The student learns a smoother, more generalizable function than it could from labels alone, and ends up punching well above its parameter count. DistilBERT is the textbook result: ~40% smaller, ~60% faster, retaining ~97% of BERT's language-understanding performance.

**What it cuts:** parameter count and architecture size — you design the student to be whatever shape fits your device. **What it costs:** a full training run with the teacher in the loop (so it is the most compute-hungry lever up front) and access to a good teacher and representative data. Distillation composes beautifully with the others: distill to a small student, then quantize the student. We dedicate a post to the loss functions, the temperature trick, and feature-matching variants.

### Lever 4: Efficient architecture and NAS (design for the device)

Sometimes the right move is not to shrink a model built for the cloud but to *design a different model* that is cheap by construction. MobileNets (Howard et al., 2017) replaced standard convolutions with depthwise-separable convolutions, cutting FLOPs roughly 8–9× for a small accuracy cost. EfficientNet found a principled way to scale depth, width, and resolution together. Neural Architecture Search (NAS) automates the design, and *hardware-aware* NAS searches directly for architectures that are fast on a specific chip, optimizing measured latency rather than proxy FLOPs (because, as we will see, FLOPs and latency are not the same thing).

**What it cuts:** FLOPs, parameters, and memory at the source — the model is small because it was born small. **What it costs:** you cannot reuse an off-the-shelf pretrained giant; you design and train a new architecture, and NAS itself can be expensive. But the resulting models sit on a better frontier than anything you can compress your way to. We cover depthwise-separable convs, inverted residuals, attention-efficient transformers, and hardware-aware NAS in their own posts.

### The substrate: compilers and runtimes

A lever only matters if the hardware actually realizes the saving. An int8 model that falls back to fp32 on the CPU because the runtime did not have an int8 kernel for one op is not faster — it might be slower. Compilers and runtimes (TensorFlow Lite / LiteRT, ONNX Runtime, TensorRT, `llama.cpp`, Core ML, ExecuTorch, TFLite-Micro) fuse operations, pick kernels, lay out memory, and dispatch to the NPU or the sparse tensor cores. They are the difference between a theoretical FLOP reduction and a measured millisecond. This is not a footnote; it is a whole track in the series.

### The four levers at a glance

Before we move on, here is the comparison table the rest of the series fills in with hard numbers. Treat it as the cheat sheet: when you have a target and a budget, this is the first thing to consult. The figures are representative ranges, not promises.

| Lever | What it cuts | Typical win | Accuracy cost | Effort | Biggest gotcha |
| --- | --- | --- | --- | --- | --- |
| Quantization | bits per value | 4× size (int8), 8× (int4); ~2–4× speed | <1% (int8), more at int4 | low (PTQ) to medium (QAT) | outliers wreck the range; op coverage on the runtime |
| Pruning | weights (zeros) | 2–10× fewer params | recoverable with fine-tune | medium | unstructured sparsity rarely speeds commodity HW |
| Distillation | model shape | 40% smaller, 60% faster (DistilBERT) | small if student is well-chosen | high (a training run) | needs a good teacher and real data |
| Efficient arch / NAS | FLOPs at the source | 8–9× fewer convs (MobileNet) | small; often a *better* frontier | high (design + train) | cannot reuse an off-the-shelf giant |

The columns that matter most when you choose are *effort* and *biggest gotcha*. Quantization is first because its effort is low and its win is large; you escalate to the costlier levers only when quantization alone does not get you onto the frontier point you need. And the gotcha column is where careers are made: the engineer who knows that unstructured sparsity will not speed up a phone CPU, or that an unsupported op will silently fall back from the NPU, saves the team a wasted quarter. Each of these gets a full post; this table is the map.

### The discipline: profiling

And the lever you must never skip is measurement. Every claim in this series is a *measured* claim, on a *named* target, at *batch size 1* (the reality on a device), *after warm-up* (the first inference is always slow), and *aware of thermal throttling* (a phone that has been running the model for 30 seconds is slower than one that just woke up). "It should be faster because it has fewer FLOPs" is a hypothesis, not a result. Profiling is what turns it into a result, and a model can be memory-bound rather than compute-bound, in which case cutting FLOPs does nothing and you needed to cut bytes. We will keep coming back to this.

## The science: why fewer bits save energy super-linearly

Now the rigorous part, because this series promises the *why* and not just the *what*. The deepest reason smaller models save energy is not that they do fewer multiplies — multiplies are cheap. It is that they move fewer bytes, and *moving bytes is where the energy goes*. This is the single most important quantitative fact in edge ML, and it comes from Mark Horowitz's 2014 ISSCC keynote, "Computing's Energy Problem (and what we can do about it)." The figure below is the table everyone in this field eventually memorizes.

![Matrix figure comparing the energy cost of arithmetic operations and memory accesses, showing a 32-bit DRAM access at roughly 640 picojoules dwarfing a 32-bit integer add at roughly 0.1 picojoules](/imgs/blogs/why-optimize-ai-models-for-the-edge-5.png)

Horowitz's numbers, for a 45 nm process (the exact figures shift with process node, but the *ratios* are remarkably stable and are what matter), are approximately:

- A 32-bit integer ADD: **~0.1 pJ** (picojoules).
- A 32-bit floating-point MULT: **~3.7 pJ**.
- A 32-bit SRAM (on-chip cache) read: **~5 pJ**.
- A 32-bit DRAM (off-chip main memory) access: **~640 pJ**.

Read that last ratio again. A DRAM access costs about **6,400×** the energy of an integer add and roughly **170×** the energy of a floating-point multiply. The arithmetic is essentially free; the memory traffic is the entire bill. This is why a memory-bound model — one that spends its time waiting for weights to arrive from DRAM rather than computing on them — is the common case on edge hardware, and why the metric that predicts energy is *bytes moved*, not FLOPs computed.

### Arithmetic intensity and the roofline, in one breath

Let us make "memory-bound" precise, because it is the concept that explains half of all surprising benchmark results. Define a kernel's **arithmetic intensity** as

$$I = \frac{\text{FLOPs performed}}{\text{bytes moved from memory}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right].$$

A processor has a peak compute rate $\pi$ (FLOP/s) and a peak memory bandwidth $\beta$ (byte/s). The *roofline model* says the achievable performance $P$ for a kernel of intensity $I$ is

$$P = \min\big(\pi,\ \beta \cdot I\big).$$

If $I$ is small — you do only a few FLOPs per byte you fetched — you are pinned to the slanted $\beta \cdot I$ roof and you are **memory-bound**: the chip's ALUs sit idle waiting for data. If $I$ is large you hit the flat $\pi$ roof and you are **compute-bound**. The crossover, the *ridge point*, is at $I^{*} = \pi / \beta$.

Here is the punchline for edge LLM inference. Generating one token at batch size 1 reads *every weight in the model once* and does roughly *two FLOPs per weight* (a multiply and an add in the matrix-vector product). So the arithmetic intensity of single-token decoding is approximately

$$I \approx \frac{2 N}{N \cdot b} = \frac{2}{b} \ \left[\frac{\text{FLOP}}{\text{byte}}\right],$$

where $N$ is the parameter count and $b$ is bytes per weight. For fp16, $b = 2$, so $I \approx 1$ FLOP/byte — far below the ridge point of any modern chip (often tens of FLOP/byte). Token generation is *deeply memory-bound*. The implication is exact and important: at batch 1, **decoding speed is set by memory bandwidth, not by FLOPs**. Time per token is approximately

$$t_{\text{token}} \approx \frac{N \cdot b}{\beta}.$$

Now quantize the weights from fp16 to int4. You have changed $b$ from 2 bytes to 0.5 bytes — a 4× reduction in bytes moved per token — and the FLOPs barely changed. Since you are memory-bound, time per token drops by ~4× and so does the energy spent moving weights. *That* is why quantization speeds up LLM decoding: not because int4 multiplies are faster (they are, but you were not compute-bound), but because you are hauling one quarter of the bytes across the memory bus per token. This is the super-linear-feeling win: a 4× cut in precision yields a ~4× cut in the dominant cost because the dominant cost was bandwidth all along.

#### Worked example: energy of compute versus transmit on a sensor

Take a battery-powered acoustic sensor that wants to classify one second of audio. Suppose the on-device model is a small CNN doing 5 MFLOPs (5 million multiply-adds, so ~10 million ops) per inference. Even at a pessimistic ~5 pJ per operation including the memory accesses to feed it, the compute energy is

$$E_{\text{compute}} \approx 10 \times 10^{6} \text{ ops} \times 5 \text{ pJ} = 50 \times 10^{6} \text{ pJ} = 50\ \mu\text{J} = 0.014\ \text{mWh-ish per } 10^{3}\ \text{inferences}.$$

Now compare transmitting the raw one-second audio clip to the cloud. Say it is 16 kHz, 16-bit mono — about 32 KB — and the radio costs on the order of ~10–50 µJ *per byte* once you fold in the radio wake-up, association, transmit power, and the tail energy of the radio staying awake afterward (numbers vary widely by radio and link quality; this is an order-of-magnitude figure). Even at the low end:

$$E_{\text{transmit}} \approx 32{,}000 \text{ bytes} \times 10\ \mu\text{J/byte} = 320{,}000\ \mu\text{J} = 320\ \text{mJ}.$$

That is over **6,000×** the energy of running the model locally, and we have not counted the energy of the *return* trip or the duty-cycle cost of keeping the radio reachable. For an always-on sensor that fires rarely, the local-compute design transmits only when it detects something, so it can run for months where the stream-everything design dies in days. The physics — radio energy dominates compute energy — is what makes edge inference an *energy* win and not just a latency or privacy one. This is the rigorous backbone behind the fifth reason from the opening section.

The takeaway you carry into every later post: **count bytes, not just FLOPs.** Whether you are memory-bound or compute-bound decides which lever helps, and on edge hardware running batch-1 workloads, memory-bound is the common case.

### How much accuracy does quantization actually cost? A derivation

The energy and bandwidth argument tells you *why* you want fewer bits. The natural follow-up is the question that makes or breaks a quantization project: how much accuracy does dropping bits cost, and can you predict it? You can, to first order, and the derivation is worth doing once because it explains every quantization result you will ever read.

When you quantize a real-valued weight $w$ to a fixed number of bits, you map it onto a grid of evenly spaced levels. With $b$ bits you have $2^b$ levels spanning the range $[w_{\min}, w_{\max}]$, so the spacing — the *quantization step* — is

$$\Delta = \frac{w_{\max} - w_{\min}}{2^{b} - 1}.$$

Each weight is rounded to the nearest grid point, so the rounding error $e = \hat{w} - w$ lands somewhere in $[-\Delta/2, +\Delta/2]$. If we model that error as uniformly distributed over that interval — a standard and remarkably accurate assumption when the signal is busy relative to the step — its variance is the classic uniform-distribution result:

$$\sigma_{e}^{2} = \frac{1}{\Delta}\int_{-\Delta/2}^{+\Delta/2} e^{2}\, de = \frac{\Delta^{2}}{12}.$$

Now define the **signal-to-quantization-noise ratio** (SQNR), the ratio of the signal's power to the quantization noise's power, in decibels. For a full-scale signal you can show, after carrying the $\Delta^2/12$ through, the famous result:

$$\text{SQNR} \approx 6.02\, b + 1.76 \ \text{dB}.$$

That linear-in-$b$ law is the whole story in one line: **every bit you add buys you about 6 dB of signal-to-noise**, and every bit you drop costs you the same. Going from int8 to int4 throws away 4 bits, which is roughly 24 dB of SQNR — a 250-fold increase in noise power. That is exactly why int8 is usually nearly free on a robust CNN but int4 starts to bite, and why the sensitive layers (the first conv that sees raw pixels, the final classifier, attention logits) need to be protected at higher precision. The math is not a curiosity; it is the budget you are spending, and it tells you in advance which layers will hurt.

There is one more piece the derivation makes obvious. The step $\Delta$ depends on the *range* $w_{\max} - w_{\min}$. If a single outlier weight stretches the range, every other weight gets a coarser grid and more error — the outlier "wastes" levels on empty space. This is precisely why outliers are the central villain of LLM quantization, and why schemes like AWQ (activation-aware weight quantization) and per-channel scales exist: they stop one fat-tailed channel from coarsening the grid for everyone. When the quantization track gets into GPTQ and AWQ, this $\Delta^2/12$ picture is the foundation under all of it. The reason "just quantize it" sometimes works and sometimes destroys a model is entirely about how the range, the outliers, and the per-layer sensitivity interact with that 6 dB per bit.

## Practical: baseline your own model in five minutes

You cannot optimize what you have not measured, so before any of the levers, you establish a baseline: how big is the model, and how long does one inference take, on the hardware you actually have. Here is a small, real, copy-and-adapt PyTorch script that does exactly that for a CPU baseline. Run it today on whatever model you are working with.

```python
import time
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 1. Load a real model as our running example (swap in your own).
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

# 2. Measure model size on disk (the number users feel as download / storage).
def model_size_mb(m):
    total_bytes = 0
    for p in m.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in m.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024 ** 2)

print(f"Model size: {model_size_mb(model):.1f} MB")

# 3. Single-inference latency on CPU, done honestly:
#    warm up first, then time many runs, then report p50 and p99.
torch.set_num_threads(4)              # pin threads so the number is reproducible
x = torch.randn(1, 3, 224, 224)       # batch size 1: the on-device reality

with torch.inference_mode():
    for _ in range(10):               # WARM-UP: the first runs are always slow
        model(x)

    times_ms = []
    with torch.inference_mode():
        for _ in range(100):
            t0 = time.perf_counter()
            model(x)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

times_ms.sort()
p50 = times_ms[len(times_ms) // 2]
p99 = times_ms[int(len(times_ms) * 0.99)]
print(f"Latency batch=1  p50: {p50:.1f} ms   p99: {p99:.1f} ms")
```

Three things in this script are not optional, and they are the difference between a number you can trust and a number that lies to you. First, **warm-up**: the first few inferences pay for lazy initialization, cache population, and (on GPU) kernel compilation; if you time them you will overstate latency badly. Second, **batch size 1**: on a server you batch to amortize cost, but on a device requests arrive one at a time, so batch-1 latency is the real user experience. Third, **report a distribution, not a single number**: p50 tells you the typical case, p99 tells you the worst case the user remembers; thermal throttling and OS scheduling make the tail fat, and the tail is what gets you paged.

For a language model the equivalent baseline is tokens per second, which — given the memory-bound analysis above — you should expect to track memory bandwidth. Here is the same discipline applied to an LLM with Hugging Face `transformers`:

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "your-small-llm"  # e.g. a 1-3B chat model you can load on CPU/GPU
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).eval()

prompt = "Explain why on-device inference is memory-bound in one paragraph."
inputs = tok(prompt, return_tensors="pt").to(model.device)

# Warm up the generation path (compiles kernels, fills caches).
with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=8)

# Time a real generation and compute tokens/s.
new_tokens = 128
with torch.inference_mode():
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    dt = time.perf_counter() - t0

generated = out.shape[-1] - inputs["input_ids"].shape[-1]
print(f"Generated {generated} tokens in {dt:.2f} s -> {generated / dt:.1f} tok/s")
```

Run that, write the number on a sticky note, and you now have a baseline. Every optimization in the rest of the series is judged against it: did the model get smaller, did it get faster, and how much accuracy did it cost?

To make the next step concrete — and to show the first lever is not a black box — here is the smallest honest int8 post-training quantization (PTQ) flow in PyTorch. PTQ means you quantize an already-trained model without retraining, using a small *calibration* pass to learn the activation ranges (the $w_{\max} - w_{\min}$ from the SQNR derivation, but for activations). It is the cheapest possible win and the right thing to try before anything fancier.

```python
import torch
import torch.ao.quantization as tq
from torch.ao.quantization import get_default_qconfig, prepare, convert

# Start from your trained fp32 model in eval mode.
model_fp32 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

# 1. Pick a quantization config. fbgemm is the x86 server/edge-CPU backend;
#    use "qnnpack" for ARM mobile CPUs. This sets int8 weights + int8 activations.
model_fp32.qconfig = get_default_qconfig("fbgemm")

# 2. Insert observers that watch activation ranges during calibration.
model_prepared = prepare(model_fp32)

# 3. CALIBRATION: run a few hundred representative inputs so the observers
#    estimate min/max per tensor. This sample MUST look like production data.
with torch.inference_mode():
    for images in calibration_loader:   # ~100-500 real, representative batches
        model_prepared(images)

# 4. Convert: fold the observed ranges into real int8 weights and quantized ops.
model_int8 = convert(model_prepared)

# 5. Re-measure size and latency with the SAME script as the baseline.
print(f"int8 size: {model_size_mb(model_int8):.1f} MB")   # ~4x smaller than fp32
```

Notice what is and is not happening here. There is no training, no gradients — just a forward pass over a few hundred representative inputs to learn the ranges, then a conversion. That is why PTQ is the first lever: it is minutes of work, not hours. The one place it bites is step 3: if `calibration_loader` is unrepresentative (wrong class balance, wrong lighting, too few samples), the observers learn the wrong ranges, the scales are off, and accuracy drops more than the SQNR math says it should. When PTQ's accuracy loss is unacceptable, the next lever up is quantization-aware training (QAT), which simulates the int8 rounding *during* fine-tuning so the network learns weights that are robust to it — but you only reach for QAT when PTQ has demonstrably failed to hit your target, because it costs a training run. That escalation ladder — try the cheap lever, measure, escalate only if needed — is the whole methodology of the series in miniature. Which brings us to the first taste of what a lever actually buys.

## A measured before to after teaser

Here is the result that motivates the entire quantization track, framed the way every results section in this series will be: a named target, a before, an after, and an honest accuracy delta. The numbers below are *representative* of well-executed int8 post-training quantization of a convolutional classifier — they are the ballpark you should expect, not a guarantee for your specific model, and the exact figures depend on the architecture, the calibration set, and the runtime.

| Variant | Precision | Size (MB) | Latency p50 (ms) | Top-1 accuracy | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline | fp32 | ~98 | ~120 | 76.1% | reference, unoptimized |
| Float16 | fp16 | ~49 | ~95 | 76.1% | ~2× smaller, lossless on most CNNs |
| Int8 PTQ | int8 | ~25 | ~38 | 75.4% | ~4× smaller, ~3× faster, <1% drop |
| Int8 + structured prune | int8 | ~16 | ~28 | 74.9% | levers compose |

*Target: a representative ARM CPU edge device (e.g. a single-board computer or phone CPU), batch=1, warmed, latency p50 over 100 runs.*

Read the int8 row closely, because it is the whole pitch in one line: roughly **4× smaller, ~3× faster, less than 1% accuracy lost**. That is an extraordinary trade — you would take a 0.7-point accuracy drop for a 3× speed-up and a model that now fits the device almost every time. And the last row shows the levers stacking: prune the int8 model and you shave more size and latency for another fraction of a point. The accuracy column is the catch, and the rest of the series is largely about *minimizing that catch* — calibration sets, quantization-aware training to recover the drop, choosing what to prune. But the headline is real, and it is why people bother: you can usually get most of the model into a quarter of the footprint for a cost you can barely measure on the validation set.

This is also where the master mental model earns its keep.

## The Pareto frontier: the master mental model

There is one frame that ties together every lever and every trade-off in this series, and you should install it now because we will reference it constantly: the **accuracy–efficiency Pareto frontier**. The figure below contrasts the one-model world with the frontier world.

![Before-and-after figure contrasting a single unoptimized model as one point against a frontier of optimized variants at different size, latency, and accuracy trade-offs](/imgs/blogs/why-optimize-ai-models-for-the-edge-4.png)

Plot every possible model with accuracy on one axis and efficiency on the other — say, accuracy versus latency, or accuracy versus model size. Each model is a point. Some points are strictly worse than others: if model A is both more accurate *and* faster than model B, then B is *dominated* — there is no reason to ever pick it. The set of *non-dominated* points — the ones where you cannot improve one axis without giving up the other — is the **Pareto frontier**. Everything below and to the right is wasteful; the frontier is the set of rational choices.

This reframes the entire job in two moves:

1. **Pick the best point on the frontier for your constraints.** You have a latency budget (the 16 ms AR frame, the sub-millisecond wake word) or a size budget (the 256 KB SRAM arena, the 3.5 GB phone RAM). The frontier tells you the most accurate model that fits. You do not get to be both maximally accurate and maximally small; you choose where on the curve to sit. A surveillance camera that must never miss a person picks a high-accuracy point; a battery doorbell picks a tiny-and-fast point and accepts a few more false wakes.

2. **Push the frontier out.** This is what the four levers and better architectures actually do — they discover models that are *both* more accurate and more efficient than anything previously known, moving the whole curve up-and-left. MobileNet pushed the frontier out relative to compressing a VGG; quantization-aware training pushes it out relative to naive post-training quantization; a better distillation recipe pushes it out relative to training the student from scratch.

You never get something for nothing. Every lever moves you *along* a frontier (trading accuracy for efficiency) or, on a good day, helps *push the frontier itself* outward. The mistake beginners make is treating optimization as free lunch ("just quantize it") or as pure loss ("quantization ruins accuracy"); the truth is it is a *trade*, and the frontier is the map of which trades are available. The two recurring reference posts in this series formalize this: the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) organizes the levers against the frontier, and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) walks an end-to-end decision down it. When a later post says "this moves you down the frontier," you will know exactly what that means.

## What this series covers

This is a long series, organized by the spine you have now seen. Each track goes deep — the science, runnable code, and measured before-to-after results — for one piece of the puzzle:

- **Foundations** (you are here): why the edge, what the edge is, the four levers, the frontier, and the hardware landscape that the sibling post details.
- **Quantization**: post-training quantization, the integer arithmetic and zero-point/scale math, calibration, quantization-aware training, and LLM-specific schemes (GPTQ, AWQ, k-quants, NF4) — the highest-leverage track for most people.
- **Pruning and sparsity**: magnitude vs structured pruning, the lottery ticket hypothesis, 2:4 sparse tensor cores, and SparseGPT/Wanda for LLMs — with the hard truth about when sparsity does and does not speed up real hardware.
- **Distillation**: the soft-label loss, temperature, feature and attention matching, and the DistilBERT-style results that make a small student punch above its weight.
- **Efficient architectures and NAS**: depthwise-separable convolutions, inverted residuals, attention-efficient transformers, and hardware-aware neural architecture search that optimizes measured latency, not proxy FLOPs.
- **Compilers and runtimes**: TensorFlow Lite / LiteRT, ONNX Runtime, TensorRT, Core ML, ExecuTorch, and `llama.cpp` — the substrate that turns a theoretical saving into a measured millisecond, with the operator-fallback traps that silently kill performance.
- **TinyML and LLMs on the edge**: the two extremes — kilobyte models on Cortex-M microcontrollers with TFLite-Micro and CMSIS-NN, and multi-billion-parameter models squeezed onto phones and laptops.
- **Profiling, MLOps, and case studies**: how to measure honestly, how to keep an optimized model healthy in production, and named real-world results to learn from.

Read them in order if you are new, or jump to your lever if you have a target in mind. Every post cross-links back to this frame.

## A model of the lifecycle

Putting a model on a device is not a one-shot conversion; it is a measured loop. You baseline, you pick a lever, you apply and recover accuracy, you compile to the target runtime, you profile *on the actual device*, and only when the budget is met do you ship — and if it is not met, you go around again with the next lever. The figure below lays out that lifecycle.

![Timeline figure showing the six-step optimize-to-ship lifecycle from fp32 baseline through choosing a lever, applying and calibrating, compiling, profiling on target, and shipping](/imgs/blogs/why-optimize-ai-models-for-the-edge-7.png)

The two steps people skip, to their cost, are the first and the fifth. Skipping the **baseline** means you have no honest reference and end up celebrating a "2× speed-up" against a number you never actually measured. Skipping **on-target profiling** — measuring on a laptop and assuming the phone behaves the same — is how you discover in the field that the NPU does not support one of your ops and the whole model fell back to the CPU, or that thermal throttling halves your throughput after 30 seconds of sustained use. The loop is cheap to run and expensive to skip; treat the device as the source of truth, not your development machine.

## Case studies: real numbers from shipped work

Abstract levers are easy to nod along to; named results are what build intuition for what is achievable. A few load-bearing examples from the literature and shipped products, all approximate but grounded in published figures:

- **Deep Compression (Han, Mao, Dally, 2016).** The paper that lit the field: combining pruning, trained quantization, and Huffman coding, they compressed AlexNet ~35× and VGG-16 ~49× *with no loss of accuracy*, fitting models that were hundreds of MB into a handful. It is the existence proof that the slack in a trained network is enormous — and the template for "the levers compose."
- **MobileNets (Howard et al., 2017).** Depthwise-separable convolutions cut the computation of a standard conv by roughly 8–9× for a small accuracy cost, delivering a family of models explicitly designed for the FLOP and latency budget of phones. This is the efficient-architecture lever in its purest form: don't compress a cloud model, design a small one.
- **DistilBERT (Sanh et al., 2019).** A distilled BERT that is ~40% smaller and ~60% faster while retaining ~97% of BERT's performance on the GLUE language-understanding benchmark — the canonical demonstration that a well-distilled student keeps almost all of the teacher's ability at a fraction of the cost.
- **GPTQ and AWQ on LLaMA-class models (2022–2023).** Post-training 4-bit quantization schemes that bring multi-billion-parameter LLMs down to a quarter of their fp16 footprint with single-digit-percent perplexity changes, which — combined with the memory-bound analysis above — is precisely why a 7B model in 4-bit GGUF runs interactively on a laptop or high-end phone today.
- **MCUNet (Lin et al., 2020).** A co-designed network-plus-runtime that fit an ImageNet-class model into a microcontroller with ~256 KB of SRAM and ~1 MB of flash, reaching ~70% top-1 — a result that seemed impossible until you co-optimize the architecture, the quantization, and the memory scheduler together. This is the TinyML end of the spectrum and the spiritual successor to the doorbell from the opening.

The thread through all of them: nobody pulled one lever in isolation. Deep Compression is three levers; MCUNet is architecture plus quantization plus a custom runtime; the LLM results are quantization riding on a memory-bound workload that a good runtime exploits. The frontier moves outward when the levers compose.

There is a second thread worth naming, because it is the difference between a paper result and a shipped product: every one of these results is *measured on a target*, not asserted from FLOPs. DistilBERT's "60% faster" is wall-clock on real hardware, not a FLOP ratio. MobileNet's whole point — and the reason hardware-aware NAS exists — is that FLOP count is a poor predictor of latency: a depthwise convolution has few FLOPs but is *memory-bound* and can run slower per FLOP than a dense conv that keeps the ALUs fed. MCUNet's headline is not "small model" but "small model that fits a specific 256 KB SRAM budget with a scheduler that never exceeds it." The lesson the case studies teach in unison is the lesson of the profiling discipline: the frontier is plotted in *measured* units on a *named* device, and a result quoted in proxy units (FLOPs, parameter count) is a hypothesis waiting to be embarrassed by the hardware. Carry that skepticism into every benchmark you read in the rest of the series, including the ones I write.

## When the edge is the wrong choice

I promised honesty, so here it is: a great deal of on-device-AI enthusiasm is misplaced, and shipping a model to the edge when the cloud would do is a self-inflicted wound. The decision tree below captures the call; the prose after it explains the leaves.

![Decision tree figure asking whether a workload needs real-time, private, or offline operation, branching to ship on-device when yes and keep huge low-volume models in the cloud when there is no such constraint](/imgs/blogs/why-optimize-ai-models-for-the-edge-8.png)

The cloud is the right answer when:

- **The model is genuinely huge and the device cannot hold even a quantized version.** A 400B-parameter frontier model is not going on a phone, and pretending otherwise wastes months. If your product *needs* that capability and there is no smaller model that suffices, serve it from the cloud and spend your effort on latency-hiding (streaming, speculative UI) instead of impossible compression.
- **Request volume is low.** The cost argument flips entirely below a threshold. If a feature runs a few thousand times a month, the recurring cloud bill is trivial and the engineering cost of optimizing for the edge will never pay back. Optimization is an investment; it needs volume to amortize.
- **You retrain or update the model frequently.** A model that changes weekly is painful to re-quantize, re-validate, and re-ship to a fleet of devices through app-store review cycles. The cloud lets you deploy a new model instantly to everyone. Frequent iteration is a strong vote for server-side.
- **There is no real latency, privacy, or offline constraint.** If a 300 ms response is fine, the data is not sensitive, and connectivity is assumed, then the edge buys you very little and costs you real engineering. The cloud is simpler, more flexible, and easier to operate. Use it.

The honest decision rule is the inverse of the five reasons: edge wins when you have a *hard* requirement on latency, privacy, offline operation, energy, or per-call cost at scale. Absent at least one of those, the cloud is usually the better engineering choice, and a senior engineer's job includes saying so before the team spends a quarter shrinking a model that never needed to leave the data center. Optimization is a cost; spend it where it pays.

A useful intermediate to remember: it is rarely all-or-nothing. Hybrid designs are common and often best — run a small, fast model on-device for the common case and the latency-sensitive path, and fall back to a big cloud model only for the hard or rare inputs. The wake word and the cheap classifier live on the device; the heavy reasoning lives in the cloud and is invoked sparingly. That cascade gets you most of the latency, privacy, and cost wins while keeping the cloud's flexibility for the long tail.

## Stress-testing the decision

Because every post in this series ends by poking at its own conclusions, let us stress-test the "optimize for the edge" decision against the cases that break naive intuition:

- **What if the model is memory-bound, not compute-bound?** Then cutting FLOPs (via pruning the compute, say) does nothing, and you must cut *bytes* (via quantization or a smaller architecture). This is the common case for batch-1 LLM decoding, as the roofline analysis showed. Always profile to learn which regime you are in before picking a lever.
- **What if the NPU does not support one of your ops?** The runtime falls back to CPU for that op, which can mean a round-trip of data off the accelerator and back, and the "accelerated" model is suddenly slower than the plain CPU version. This is why on-target profiling and op-coverage checks (covered in the compilers track) are non-negotiable, and why hardware-aware design beats FLOP-counting.
- **What if the calibration set is tiny?** Post-training quantization estimates the dynamic range of activations from a small representative sample; if that sample is unrepresentative or too small, your scales are wrong and accuracy craters. The fix — a better calibration set or quantization-aware training — is the quantization track's bread and butter.
- **What about int4 instead of int8?** The size and bandwidth win doubles, but the accuracy cost grows nonlinearly and some layers (the first and last, attention scores) are far more sensitive. Mixed-precision — keeping sensitive layers at higher precision — is how production int4 actually works, and it is a frontier-pushing move rather than a free one.

In every case the move is the same: form the hypothesis, *measure on the target*, and read the result off the frontier. The discipline is the product.

## Key takeaways

- **The edge problem is a gap problem.** Models grew one-to-four orders of magnitude faster than edge compute and memory; a 7B model in fp16 is 14 GB against a phone's 8 GB. Optimization exists to close that gap, and the gap is real because models are overparameterized and over-precise by training convenience, not deployment need.
- **On-device wins for five distinct, measurable reasons:** latency (sub-10 ms local vs 50–500 ms network), privacy (sensitive bytes never leave), cost (no recurring per-call cloud bill), reliability (works offline, tight p99), and energy (radio transmit dwarfs local compute). Know *which* of these your product actually needs.
- **Count bytes, not just FLOPs.** Horowitz's numbers say a DRAM access costs ~6,400× an int add; on-chip arithmetic is nearly free and memory movement dominates energy. Batch-1 edge workloads are usually memory-bound, so quantizing weights to fewer bits speeds them up almost in proportion to the byte reduction.
- **There are four levers, and they compose:** quantization (fewer bits, ~4× smaller), pruning (zeros, but watch for no real speed-up on commodity hardware), distillation (a small student trained by a big teacher), and efficient architecture/NAS (small by design). They ride on compilers/runtimes and are validated by profiling.
- **The accuracy–efficiency Pareto frontier is the master frame.** Every lever moves you along a frontier or pushes the frontier out. Pick the best point for your constraint; never expect a free lunch.
- **Measure honestly or not at all:** named target, batch size 1, after warm-up, aware of thermal throttling, report p50 and p99. A latency claim without a measurement is a hypothesis.
- **The edge is often the wrong choice.** When the model is huge and volume is low, when you retrain constantly, or when there is no hard latency/privacy/offline/energy constraint, the cloud is the better engineering call. Hybrid on-device-plus-cloud cascades capture most of the wins with less pain.
- **Baseline first.** Run the size-and-latency script in this post on your own model before you optimize anything, so every later gain is measured against a real number.

## Further reading

- Han, Mao, Dally, **"Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"** (ICLR 2016) — the foundational result that the levers compose for 35–49× compression with no accuracy loss.
- Howard et al., **"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"** (2017) — depthwise-separable convolutions and the efficient-architecture lever.
- Hinton, Vinyals, Dean, **"Distilling the Knowledge in a Neural Network"** (2015) — the soft-label distillation idea behind every small-student result.
- Horowitz, **"Computing's Energy Problem (and what we can do about it)"** (ISSCC 2014) — the energy-per-operation numbers that explain why moving bytes, not computing on them, dominates the energy bill.
- Frankle, Carbin, **"The Lottery Ticket Hypothesis"** (ICLR 2019) — why sparse subnetworks exist and pruning works at all.
- Lin et al., **"MCUNet: Tiny Deep Learning on IoT Devices"** (NeurIPS 2020) — the TinyML end of the spectrum, ImageNet-class accuracy in 256 KB of SRAM.
- Official runtime docs worth bookmarking: TensorFlow Lite / LiteRT, ONNX Runtime, NVIDIA TensorRT, and `llama.cpp` (GGUF and k-quants) — the substrate that realizes every saving.
- Within this series: the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the levers-against-the-frontier map, the [edge hardware landscape](/blog/machine-learning/edge-ai/the-edge-hardware-landscape) for the device tiers in detail, and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for the end-to-end decision walked down the frontier.
