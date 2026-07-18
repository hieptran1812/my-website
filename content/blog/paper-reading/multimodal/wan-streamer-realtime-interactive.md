---
title: "Wan-Streamer: Collapsing the Real-Time Avatar Cascade into One Causal Transformer"
date: "2026-07-18"
publishDate: "2026-07-18"
description: "A detailed, intuition-first read of Wan-Streamer: one causal Transformer that takes text, audio, and video in and out, uses block-causal attention and flow matching, and hits ~200 ms model-side latency for full-duplex audio-visual conversation."
tags: ["paper-reading", "multimodal", "wan-streamer", "full-duplex", "streaming-inference", "flow-matching", "audio-visual-generation", "real-time", "thinker-performer", "block-causal-attention", "foundation-model", "video-generation"]
category: "paper-reading"
subcategory: "Multimodal"
author: "Hiep Tran"
featured: true
readTime: 39
paper:
  title: "Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models"
  authors: "Wan Team, Alibaba Group (Lianghua Huang et al.)"
  venue: "arXiv 2026 (cs.CV)"
  url: "https://arxiv.org/abs/2606.25041"
---

Build a video-call agent the way the industry has built them for the last three years and you end up with a relay race. A voice-activity detector waits until it is sure you have stopped talking. An ASR model turns your last utterance into text. A language model reads the text and writes a reply. A text-to-speech model turns the reply into a waveform. An audio-driven animation model turns the waveform into lip and face motion. A renderer turns the motion into video. Six handoffs, six queues, six chances for an error early in the chain to poison everything downstream — and a human on the other end who can feel every one of those queues as *lag*.

The Wan Team's [Wan-Streamer](https://arxiv.org/abs/2606.25041) report is an argument that this architecture is not just slow but *structurally* slow, and that the fix is not faster modules but *no modules*. It puts language, audio, and video — on both the input and the output side — into **one causal Transformer**, and reports roughly **200 ms** of model-side latency and roughly **550 ms** end-to-end for a remote user, while emitting a synchronized 25-FPS talking-head video. This is a v0.1 technical report: short, light on ablations, validated at a preliminary 192p resolution. But the design choices are worth reading closely, because they are a clean case study in *why streamability has to be a modeling constraint, not a serving trick*.

> [!tldr] TL;DR
> - **What it is.** A native-streaming, end-to-end foundation model for real-time full-duplex audio-visual conversation. One Transformer consumes user text/audio/video and emits agent text/audio/video, coordinated by *block-causal attention*.
> - **The core mechanism.** Text is generated as discrete tokens with next-token prediction; audio and video are generated as continuous latents with *conditional flow matching*, both denoised from the same causal context so lip motion and prosody are synchronized *natively* rather than repaired afterward.
> - **Why it matters.** It replaces the VAD → ASR → LLM → TTS → animation → render cascade — which waits at every module boundary and accumulates recognition/sync errors — with a single model, cutting pipeline latency and error accumulation.
> - **The surprising bit.** A two-GPU *thinker–performer* split preserves the semantics of one unified model by exchanging KV-cache slices, while overlapping perception, decoding, and flow-matching generation across adjacent 160 ms units — reaching ~200 ms model-side latency.
> - **Where it's thin.** No quantitative quality benchmarks, essentially no ablations, 192p resolution, and the latency numbers are a serving-path measurement rather than a decomposed, reproducible budget. It is a proof of concept, and says so.

![Figure 1 from Wan Team (2026): Wan-Streamer models text, audio, and video as both input and output within a single Transformer, using block-causal attention for incremental streaming across 160 ms frames.](/imgs/blogs/wan-streamer-realtime-interactive-fig1.webp)

The diagram above is the mental model, and it is worth staring at before we open any equations. On the bottom of each 160 ms *streaming frame*, user text, user audio, and user video enter through a tokenizer and two encoders and become interleaved input tokens. Those tokens flow into the single "Wan-Streamer" block. On top, the model emits interleaved output tokens that a tokenizer and two decoders turn back into agent text, agent speech, and an agent video frame. Then the whole frame — user side and agent side — is committed to history, and the next frame begins. There is no ASR box, no TTS box, no separate animator. The rest of this post unpacks how a single sequence model can play all of those roles at once, and why doing so is what buys the latency.

## The problem: a cascade cannot be made real-time by engineering

Start with why the relay race is slow, because the paper's whole design is a response to it. A cascade has two costs that no amount of kernel tuning removes.

The first is **module-boundary waiting**. Each stage in the cascade is trained and run independently, so each one imposes its own commit point. The VAD has to *decide* you are done speaking before ASR can produce a final transcript; ASR has to finish a chunk before the LLM can read it; the LLM has to emit text before TTS can speak it; TTS has to produce audio before the animator can drive a face. Even if every module is individually fast, the *serial* dependency means the perceived latency is a sum of commit points, plus the network round-trips between services. Worse, several of these commit points are *semantic*: endpointing (deciding a turn is over) is a guess, and a conservative guess adds hundreds of milliseconds of dead air.

The second is **error accumulation**. Text is used as a hidden intermediate representation between separately trained components. If ASR mis-hears a word, the LLM reasons over the wrong word; if TTS mispronounces, the animator lip-syncs to the wrong phonemes. Because each module was trained to be good *in isolation* — not to be good *as part of one behavior* — nothing in the pipeline is optimizing the thing you actually care about: a coherent, well-timed, identity-preserving audio-visual response. Response timing, turn management, identity preservation, and long-horizon consistency are exactly the properties that fall in the cracks *between* modules, and a cascade has no place to learn them.

![Redrawn: a cascade waits and compounds errors at every module boundary, while one causal Transformer learns perception, speaking, and listening jointly and commits each unit back to history.](/imgs/blogs/wan-streamer-realtime-interactive-1.webp)

The deeper point the authors make is that real-time audio-visual interaction is **not** the union of multimodal understanding and multimodal generation. It is *intrinsically full-duplex*. When you are speaking, a good listener is still producing visible behavior — nodding, holding eye contact, raising an eyebrow. When the agent is speaking, it should still be *perceiving* you, so it can stop when you interrupt. The modalities have different token rates and different latency budgets, yet they must stay causally aligned inside one ongoing process: your incoming speech should immediately affect the agent's outgoing motion; the agent's generated audio and video should be coupled *before* decoding, not stitched together afterward; and every emitted unit should become part of the history so identity, scene, and speaking rhythm survive over a long session.

Those requirements are why the authors call streamability a **modeling constraint**. A system built from offline encoders, bidirectional video decoders, round-based dialogue, or post-hoc audio-visual synchronization has already thrown away the information it would need to be low-latency and full-duplex — you cannot recover truly causal behavior from a fundamentally non-causal design by engineering alone. If you want a model that can be interrupted mid-sentence, you have to *train* it on interleaved streams where interruptions happen, with an architecture that never assumes it has seen the whole utterance.

This is the gap Wan-Streamer claims to fill. Prior work advanced pieces of it — multimodal LLMs that reason over audio and vision, large video generators that synthesize realistic motion, causal/streaming generation methods that make incremental synthesis feasible — but these were usually assembled into asymmetric or cascaded systems. Some perceive audio and video but respond only in text or speech; some generate audio-visual behavior but rely on external language, ASR, TTS, or animation modules. Wan-Streamer's bet is that a *single* Transformer, made causal end to end, can do all of it.

## What Wan-Streamer contributes

Stripped to its load-bearing claims, the report makes three:

1. **A single-Transformer interactive foundation model** that supports language, audio, and video as both inputs and outputs, without external language, speech, animation, or video-generation modules. Perception, reasoning, generation, response timing, turn management, and cross-modal synchronization are learned *jointly*.
2. **A fully causal multimodal architecture** for real-time interaction: strictly causal audio and video VAEs, causal audio-visual encoders and decoders, block-causal multimodal attention, and full-history autoregressive streaming with units as short as 160 ms at 25 FPS.
3. **A low-latency thinker–performer inference system** that preserves the unified model state through KV-cache exchange while overlapping understanding and generation, reaching ~200 ms model-side and ~550 ms total interaction latency.

The rest of this post is a technique-by-technique read of how those three fit together. We will climb, for each mechanism, from the problem it solves to the intuition to the mechanism to the math to a small worked example — because the equations here are compressions of ideas that only make sense once you hold the streaming picture in your head.

## The method

The method rests on seven mechanisms: the streaming contract that defines the sequence, the full-duplex overlap it enables, the two generation objectives on one backbone, the flow-matching scheme that couples audio and video, the end-to-end causal stack, the training recipe, and the thinker–performer serving split. Take them in order.

### 1. The streaming contract: one interleaved causal sequence

**The problem.** You have three modalities with wildly different natural rates — text is a few tokens per turn, audio is tens of tokens per second, video is a stream of frames — and they arrive on *both* sides of a conversation. How do you lay all of that out as one sequence a Transformer can process incrementally, so that "generate the next 160 ms of response" is a well-defined next-step prediction?

**The intuition.** Think of a single shared timeline, like a multitrack recording where the tracks are *interleaved into one tape* rather than played in parallel. At each tick of the clock (one 160 ms unit), you first splice in the user's newly-arrived text, audio, and video, then splice in the agent's freshly-generated text, audio, and video. Reading the tape left to right *is* the causal order: everything to the left has already happened and can be attended to; everything to the right has not happened yet and must stay hidden. Generation is just "given the tape so far, write the next segment."

**The mechanism.** At the $k$-th streaming unit, let $u_k = (u_k^t, u_k^a, u_k^v)$ be the user's language, audio, and video observations and let $y_k = (y_k^t, y_k^a, y_k^v)$ be the agent's response. The model predicts the next response from the complete causal history across *both* sides of the interaction, then appends the generated unit together with the user's observations to the history so it becomes context for unit $k+1$. Concretely, the joint distribution factorizes autoregressively over units:

$$
p_\theta(y_{1:K} \mid u_{1:K}) = \prod_{k=1}^{K} p_\theta\!\left(y_k^t, y_k^a, y_k^v \;\middle|\; u_{\le k}^t, u_{\le k}^a, u_{\le k}^v,\; y_{\lt k}^t, y_{\lt k}^a, y_{\lt k}^v\right)
$$

where $K$ is the number of streaming units, $u_{\le k}$ denotes every user observation up to and including unit $k$ (the model may condition on the *current* user input while producing the current response — that is what makes it full-duplex), and $y_{\lt k}$ denotes agent responses already committed before unit $k$. Every symbol on the right of the bar is "the tape so far"; every symbol on the left is "the next segment to write."

The thing that makes this cheap to run incrementally is the attention mask. A vanilla causal (lower-triangular) mask works at the token level, but here the natural unit is a *block* — the group of tokens belonging to one modality in one streaming unit. The paper coordinates the sequence with **block-causal attention**: a query block attends to itself and to all blocks committed before it, and to nothing in the future.

![Redrawn: the block-causal attention mask. Each new streaming block (user input, then agent output) attends to all committed history but nothing in the future; the input block precedes the output block within each 160 ms unit.](/imgs/blogs/wan-streamer-realtime-interactive-2.webp)

**The math, made explicit.** Group the sequence into blocks $B_1, B_2, \dots$ where within each unit the user input block precedes the agent output block. Define the block-causal mask $M$ over blocks:

$$
M_{ij} = \begin{cases} 0 & \text{block } i \text{ may attend to block } j \ \ (j \le i) \\ -\infty & \text{otherwise} \ \ (j \gt i) \end{cases}
$$

and the attention within a layer becomes $\text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d}} + M\right)V$, where $Q, K, V \in \mathbb{R}^{n \times d}$ are the query/key/value projections of the $n$ tokens in the window and $d$ is the head dimension. The $-\infty$ entries drive those attention weights to zero after the softmax, so a token in the current unit can look back across the entire committed history but never forward. Because the mask is block-lower-triangular, the KV representations of past blocks are *fixed* once committed — which is exactly the property a KV cache needs. Nothing about a future block can change a past block's keys and values, so you compute each block's KV once and reuse it forever.

**A worked micro-example.** Take four blocks in order: $u_k$ (user input at unit $k$), $y_k$ (agent output at unit $k$), $u_{k+1}$, $y_{k+1}$. The mask says: $u_k$ attends only to $u_k$; $y_k$ attends to $\{u_k, y_k\}$ — crucially it *can* see the current user input, which is how the agent's speech reacts to what you are saying *right now*; $u_{k+1}$ attends to $\{u_k, y_k, u_{k+1}\}$; and $y_{k+1}$ attends to all four. That is precisely the green lower-triangle in the figure above. Notice the ordering choice: input-before-output *within* a unit is what lets the response condition on the freshest user observation, shrinking the reaction latency to a single unit.

**Why it works / when it fails.** Block-causal attention is what makes "one model, streaming" tractable: it gives you a well-defined autoregressive target, a reusable KV cache, and the ability to react within one 160 ms unit. The cost is that the context grows without bound — full-history streaming means the KV cache keeps growing over a long session, and attention over a very long tape gets expensive. The paper leans on "full-history autoregressive streaming" as a feature (long-horizon consistency) but does not discuss the eviction or compression strategy you will eventually need; over a genuinely long call, this is where a real deployment will feel pressure. (For the general shape of that problem, see [KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management).)

### 2. Full-duplex means perception and expression overlap

**The problem.** Turn-based dialogue models assume a clean alternation: you talk, then I talk. Real conversation does not work that way. A person keeps watching and listening while they speak, and shows visible listening behavior while the other person speaks. If the agent stops perceiving the moment it starts generating, it cannot be interrupted, and it goes dead-eyed and frozen whenever it is not the one talking.

**The intuition.** Look down any vertical slice of the conversation timeline and you should see *more than one thing happening at once*. While the user speaks, the agent is simultaneously listening *and* producing small non-verbal feedback. While the agent speaks, it is simultaneously perceiving the user. Full-duplex is not a turn-taking rule bolted on top; it is the property that the perception track never switches off.

<figure class="blog-anim">
<svg viewBox="0 0 760 360" role="img" aria-label="A shared timeline: while the user speaks the agent shows listening behavior, and while the agent speaks it keeps perceiving the user, so perception and expression overlap continuously" style="width:100%;height:auto;max-width:820px">
<style>
.d5-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.d5-speak{fill:var(--accent,#6366f1)}
.d5-listen{fill:var(--border,#d1d5db)}
.d5-perc{fill:#34d399}
.d5-lblw{font:600 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.d5-lbld{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.d5-lane{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:start}
.d5-zone{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.d5-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
.d5-head{fill:var(--accent,#6366f1)}
@keyframes d5-sweep{0%{transform:translateX(0);opacity:0}6%{opacity:1}94%{opacity:1}100%{transform:translateX(567px);opacity:0}}
@keyframes d5-pulse{0%,100%{opacity:.30}50%{opacity:.55}}
.d5-sweep{animation:d5-sweep 9s linear infinite}
.d5-pulse{animation:d5-pulse 3s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.d5-sweep{animation:none}.d5-pulse{animation:none;opacity:.42}}
</style>
<text class="d5-zone" x="265" y="34">(1) user speaks</text>
<text class="d5-zone" x="570" y="34">(2) agent responds</text>
<text class="d5-lane" x="12" y="104">User</text>
<text class="d5-lane" x="12" y="164">Agent</text>
<text class="d5-lane" x="12" y="180">· express</text>
<text class="d5-lane" x="12" y="234">Agent</text>
<text class="d5-lane" x="12" y="250">· perceive</text>
<rect class="d5-bg" x="150" y="76" width="570" height="48" rx="6"/>
<rect class="d5-speak" x="150" y="76" width="230" height="48" rx="6"/>
<rect class="d5-listen" x="384" y="76" width="336" height="48" rx="6"/>
<text class="d5-lblw" x="265" y="105">speaking</text>
<text class="d5-lbld" x="552" y="105">pause / listens back</text>
<rect class="d5-bg" x="150" y="146" width="570" height="48" rx="6"/>
<rect class="d5-listen" x="150" y="146" width="266" height="48" rx="6"/>
<rect class="d5-speak" x="420" y="146" width="300" height="48" rx="6"/>
<text class="d5-lbld" x="283" y="175">listening (nod, gaze)</text>
<text class="d5-lblw" x="570" y="175">speaking (lip-synced)</text>
<rect class="d5-bg" x="150" y="216" width="570" height="48" rx="6"/>
<rect class="d5-perc d5-pulse" x="150" y="216" width="570" height="48" rx="6"/>
<text class="d5-lbld" x="435" y="245">perceiving user audio + video — always on</text>
<line class="d5-axis" x1="150" y1="288" x2="720" y2="288"/>
<text class="d5-zone" x="694" y="308">time</text>
<rect class="d5-head d5-sweep" x="148" y="64" width="4" height="230" rx="2"/>
</svg>
<figcaption>Full-duplex: the agent's perception bar never turns off — it shows listening behavior while the user speaks and keeps watching while it speaks, so any column of the timeline has perception and expression active at once, which is what lets the user interrupt.</figcaption>
</figure>

**The mechanism.** This falls out of the streaming contract almost for free. Because the input block $u_k$ precedes the output block $y_k$ within each unit and $y_k$ attends to $u_{\le k}$, the model is *architecturally* always conditioning its current output on the freshest user observation. Full-duplex behavior — when to continue, pause, overlap, interrupt, yield, or resume — is then *learned from data* in which those events actually occur, not scripted by a VAD and a turn-taking state machine. At inference the model keeps consuming user audio-video even while generating, so it can stop, shorten, or redirect its own speech the moment you cut in. The same always-on context enables *proactive* speaking: when something salient appears in the user's video — a gesture, an object, an expression — the model can initiate a comment based on what it sees rather than waiting for a spoken request.

**Why it works / when it fails.** The strength is honesty: there is no separate "interruption module" whose thresholds you have to tune, because interruption is just what the model does when the training data taught it that overlapping speech means "yield." The weakness is that this behavior is only as good as the interleaved interaction data, and the report does not quantify it — there is no interruption-latency number, no false-interrupt rate, no measure of how often the model talks over you. We are told the capability exists; we are not shown its operating curve. This is the single biggest "trust us" in the paper, and it is exactly the kind of behavior that is easy to demo and hard to make robust.

### 3. Two objectives on one backbone

**The problem.** Text and video are different *kinds* of signal. Text is discrete and low-rate; you can enumerate a vocabulary and predict the next symbol. Video and audio are continuous and high-rate; enumerating a "vocabulary" of frames is hopeless, and next-token prediction over a discretized codec tends to blur or drift. A single model has to generate both, well, from the same hidden state.

**The intuition.** Use the right tool for each output head, but share the body. The Transformer is a general sequence model that produces a context vector for each position; what you *attach* to that context can differ by modality. For text, attach a classifier over the vocabulary and train it with cross-entropy — the same machinery every LLM uses. For audio and video, attach a *generative* head that denoises a continuous latent conditioned on that context.

**The mechanism and math.** The language response is a sequence of discrete tokens optimized with next-token prediction. For the text tokens $y^t$ in a unit,

$$
\mathcal{L}_{\text{text}} = -\sum_{i} \log p_\theta\!\left(y_i^t \mid y_{\lt i}^t,\, c\right)
$$

where $y_i^t$ is the $i$-th text token, $y_{\lt i}^t$ are the preceding text tokens in the unit, and $c$ is the causal context (everything committed so far). This is ordinary cross-entropy: maximize the log-probability the model assigns to the ground-truth next token.

The audio and video responses live in *continuous latent spaces* — a strictly causal VAE maps waveforms and frames into latents, which we will meet in §5 — and are generated with **conditional flow matching**, the subject of the next section. The important structural point here is that both heads are driven by the *same* Transformer over the *same* interleaved sequence. Perception, language reasoning, and latent generation are aligned in one sequence model, not optimized as isolated modules. During pretraining, understanding tasks and generation tasks are *mixed* so that the shared body learns representations that serve both.

**Why it works / when it fails.** Splitting the loss by output type is standard and robust — it is essentially what unified multimodal models like [BAGEL](/blog/paper-reading/multimodal/bagel-unified-multimodal-mixture-of-transformers) and the omni family do. The risk in any such design is *interference*: the gradients from a hard generative loss and a hard discrete loss can pull the shared parameters in different directions, and one modality quietly degrades. The report claims the mixed pretraining keeps understanding metrics "approaching" dedicated multimodal models and dialogue ability "comparable" to turn-based models of similar scale — but "approaching" and "comparable" are unquantified hedges, and this is exactly the no-free-lunch question that [Qwen3-Omni](/blog/paper-reading/multimodal/qwen3-omni-technical-report) tried to answer with controlled numbers. Wan-Streamer v0.1 does not.

### 4. Flow matching: denoising audio and video together

This is the heart of the generative side, and the one place where reading the equation first would genuinely mislead you. Build the picture, then read the math.

**The problem.** You need to *generate* the next 160 ms of speech and video as continuous latents, in a way that is (a) fast enough to run inside a streaming budget and (b) *coupled*, so the lip motion matches the phonemes and the facial dynamics match the prosody. Standard diffusion generates by reversing a many-step stochastic noising process — accurate but slow, and if you generate audio and video with two independent samplers you have to *repair* their synchronization afterward.

**The intuition.** Flow matching replaces the wiggly, many-step diffusion trajectory with a **straight line**. Picture the space of all possible latents. At one end sits pure Gaussian noise; at the other sits the clean target latent you want. Flow matching says: connect them with a straight segment, and train the model to predict the *direction and speed* along that segment — the velocity. Because the path is a straight line, the velocity is *constant everywhere on it*: it is just "clean minus noise" (up to sign). The model does not have to learn a complicated curved flow; it has to learn one arrow. To generate, you start at noise and step along the arrow toward the clean end.

![Redrawn: flow matching walks a straight line from Gaussian noise (τ = 1) to the clean latent (τ = 0); the velocity is constant along the line, and the same context conditions both the audio and video velocity fields.](/imgs/blogs/wan-streamer-realtime-interactive-3.webp)

**The mechanism.** For modality $m \in \{a, v\}$ (audio, video), let $z_0^m$ be the clean target latent and $\epsilon^m \sim \mathcal{N}(0, I)$ be Gaussian noise. Define a *flow time* $\tau \in [0, 1]$ where $\tau = 1$ is pure noise and $\tau = 0$ is clean. The noisy latent at flow time $\tau$ is the straight-line interpolation between clean and noise, and its time-derivative is the constant velocity:

$$
z_\tau^m = (1-\tau)\, z_0^m + \tau\, \epsilon^m, \qquad \frac{\partial z_\tau^m}{\partial \tau} = \epsilon^m - z_0^m
$$

Read the two pieces. The first equation is the interpolation: at $\tau = 0$ it returns $z_0^m$ (clean), at $\tau = 1$ it returns $\epsilon^m$ (noise), and in between it is a linear blend. The second is what falls out by differentiating: the velocity $\partial z_\tau^m / \partial \tau = \epsilon^m - z_0^m$ has *no $\tau$ in it* — that is the "constant along the straight line" property, and it is why flow matching trains stably and samples in few steps. (For the broader family this sits in, the latents themselves come from a VAE the way [latent diffusion](/blog/paper-reading/diffusion-model/high-resolution-image-synthesis-with-latent-diffusion-models) works.)

Now the coupling. Let

$$
c_k = \{\, u_{\le k}^t,\, u_{\le k}^a,\, u_{\le k}^v,\, y_{\lt k}^t,\, y_{\lt k}^a,\, y_{\lt k}^v \,\}
$$

be the **clean streaming context** — the user observations that have arrived and the agent responses already committed to history. The current audio and video latents are the noisy variables being denoised; $c_k$ stays available as causal history. The unified diffusion transformer $f_\theta$ is trained to predict the velocity for *both* noisy latents, conditioned on the same context and the noise level $\tau$:

$$
\mathcal{L}_{\text{FM}}^{m} = \mathbb{E}_{\epsilon^m}\left\| f_\theta\!\left(z_\tau^a,\, z_\tau^v,\, c_k,\, \tau\right) - \frac{\partial z_\tau^m}{\partial \tau} \right\|^2
$$

The crucial detail is the argument list of $f_\theta$: it takes *both* $z_\tau^a$ and $z_\tau^v$ at once, plus the shared $c_k$. The *same clean context conditions both velocity predictions*, so speech, motion, appearance, and scene evolution are optimized as a single coupled response instead of two streams that must be re-aligned. After denoising, the estimated clean latents are appended *directly* to the history as clean context for later units, while the causal decoders render them into external audio and video.

**A worked micro-example.** Take a scalar to strip away the tensors. Say the clean latent is $z_0 = 4$ and the sampled noise is $\epsilon = 10$. Then the velocity target is $\epsilon - z_0 = 6$, everywhere on the path. At $\tau = 0.5$ the noisy point is $z_{0.5} = 0.5 \cdot 4 + 0.5 \cdot 10 = 7$. To *sample*, you invert the process: start at $z_1 = \epsilon = 10$ and integrate the predicted velocity backward toward $\tau = 0$. With a single Euler step of size $\Delta\tau = 1$: $z_0 \approx z_1 - \Delta\tau \cdot (\epsilon - z_0) = 10 - 1 \cdot 6 = 4$ — exactly the clean value. In one step, because the path was straight and the velocity was right. Real latents are high-dimensional and the learned velocity is imperfect, so you take a handful of steps rather than one, but the geometry is this simple. Here is the shape of it in code:

```python
import torch

# z0: clean target latent for one modality, shape [B, C, T, H, W] for video
# ctx: causal context features (KV of committed history), consumed inside f_theta
def flow_matching_loss(f_theta, z0_a, z0_v, ctx):
    # sample one flow time per example and independent Gaussian noise per modality
    tau = torch.rand(z0_a.shape[0], device=z0_a.device)            # [B], in (0,1)
    eps_a, eps_v = torch.randn_like(z0_a), torch.randn_like(z0_v)
    t = tau.view(-1, *([1] * (z0_a.ndim - 1)))                     # broadcast to latent rank

    # straight-line interpolation: tau=1 -> noise, tau=0 -> clean
    z_a = (1 - t) * z0_a + t * eps_a
    z_v = (1 - t) * z0_v + t * eps_v

    # constant velocity target = noise - clean (no tau dependence)
    v_a_target, v_v_target = eps_a - z0_a, eps_v - z0_v

    # ONE network sees BOTH noisy latents + shared context -> coupled velocities
    v_a_pred, v_v_pred = f_theta(z_a, z_v, ctx, tau)
    return ((v_a_pred - v_a_target) ** 2).mean() + ((v_v_pred - v_v_target) ** 2).mean()

@torch.no_grad()
def sample(f_theta, ctx, shape_a, shape_v, steps=4):
    # start at pure noise (tau = 1) and integrate toward tau = 0
    z_a, z_v = torch.randn(shape_a), torch.randn(shape_v)
    taus = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        dtau = taus[i] - taus[i + 1]                               # positive step toward 0
        v_a, v_v = f_theta(z_a, z_v, ctx, taus[i].expand(shape_a[0]))
        z_a, z_v = z_a - dtau * v_a, z_v - dtau * v_v              # Euler step
    return z_a, z_v                                                # clean audio/video latents
```

**Why it works / when it fails.** The straight-line path is what makes generation *few-step*, which is what makes it fit inside a streaming budget; coupling both modalities through one network with a shared context is what makes lip-sync and prosody emerge *natively* rather than being repaired by a post-hoc aligner. The failure modes are the usual ones for few-step flow/diffusion: with too few solver steps the samples get soft and lose high-frequency detail (hence 192p and the distillation stage below), and because the model conditions on its *own* previously-generated clean latents, small per-unit errors can compound over a long rollout — the classic train-test mismatch that §6's rolling distillation exists to fight.

### 5. A stack that is causal from encoder to decoder

**The problem.** Flow matching in latent space assumes you *have* a latent space. But the standard VAEs and encoders used for video are *bidirectional*: they look at a whole clip at once to encode or decode a frame. Drop one of those into a streaming loop and you have reintroduced the exact non-causal waiting the whole design is trying to kill — the decoder would need future frames to render the current one.

**The intuition.** Every component that touches the stream must obey the same rule the Transformer does: *only look backward*. If the video VAE needs the next frame to encode this one, it cannot run in real time; if the audio decoder needs the rest of the utterance to synthesize this chunk, it cannot emit incrementally. So the authors rebuild the whole stack — VAEs, encoders, decoders — to be strictly causal.

**The mechanism.** The report specifies four causal pieces around the temporally-causal Transformer:

- **Strictly causal audio and video VAEs** for streaming latent coding. These map incoming waveforms and frames to the continuous latents that flow matching denoises, using only past and present input — so a latent for the current 160 ms is available as soon as that 160 ms has arrived.
- **Causal audio-visual encoders** on the input side, turning user audio and video into the interleaved input tokens the Transformer consumes.
- **Causal audio and video decoders** on the output side, turning the generated clean latents into an emittable waveform and video frame — again without peeking at future latents.
- **The block-causal Transformer** from §1, tying it all together.

The number that anchors this is the **streaming unit: 160 ms at 25 FPS**. At 25 FPS a frame is 40 ms, so a 160 ms unit is four video frames — small enough that "wait for one unit" is a barely-perceptible delay, and the whole stack is designed so that one unit's worth of input can be turned into one unit's worth of output without any component reaching forward in time.

**Why it works / when it fails.** Causality end-to-end is the non-negotiable enabler: it is what makes "encode the current unit, generate the next, decode the previous" a legal pipeline (§7). The cost is quality headroom. A bidirectional VAE that can see the whole clip will always reconstruct better than a strictly causal one that cannot; a causal decoder gives up the temporal smoothing that two-sided context buys. The 192p resolution and the "proof of concept" framing are, I suspect, partly downstream of this: causal components are harder to make high-fidelity, and the team chose to prove the streaming design before pushing resolution. The report asserts scaling resolution is "straightforward," which is the kind of claim that is cheap to make and expensive to keep.

### 6. The three-stage training recipe

**The problem.** You cannot train a model this heterogeneous from scratch in one shot. It needs language competence, per-modality understanding and generation, *and* the full-duplex interaction behavior — and it needs to end up fast enough to deploy. Doing all of that at once would be unstable and wasteful.

**The intuition.** Build competence in layers, each stage assuming the last. First teach the model to understand and generate each modality in isolation, borrowing a strong language model as the starting point. Then teach it the *interaction* — how the streams interleave in a real duplex conversation. Finally, compress the slow, high-quality version into a fast student that can actually run in real time.

![Redrawn: the three training stages — task pretraining aligns the modalities (initialized from a Qwen LM), duplex interaction training teaches turn-taking, and distillation buys the low latency via rolling distillation and self-forcing.](/imgs/blogs/wan-streamer-realtime-interactive-4.webp)

**The mechanism, stage by stage.**

**Stage 1 — independent-task pretraining.** The unified Transformer is initialized from a language model (the Qwen family) and the multimodal interface is trained around it on a broad mixture: image/audio/video understanding, text dialogue, ASR, TTS, audio dialogue on the understanding side; image, audio, video, and joint audio-visual generation on the generation side. Critically, the causal audio and video encoders are trained *together with* the Transformer, and understanding and generation tasks are *mixed*, so perception, language reasoning, and latent generation land in one aligned sequence model rather than three bolted-together ones.

**Stage 2 — end-to-end interaction training.** Now train on *duplex interaction data* where user text/audio/video inputs and agent text/audio/video outputs are interleaved in the same causal stream. This adapts the model from independent tasks to the target real-time setting: it must update its state from the current user observation, generate synchronized language/audio/video, and commit the generated clean latents back into history for the next unit. Response timing, active-listening behavior, interruption handling, and long-context consistency are all learned here, under the *same causal format used at inference time* — which is the whole point, because a model trained in one regime and served in another is where subtle failures breed.

**Stage 3 — distillation for low-latency streaming.** A stronger *teacher* — with classifier-free guidance (CFG) and more flow-matching solver steps — is distilled into an efficient *student* used at deployment. Distillation absorbs the effect of CFG into the student and reduces the number of solver steps while preserving audio-visual quality. (This is the same "make a great slow model, then compress it into a fast one" move as [SDXL-Lightning](/blog/paper-reading/diffusion-model/sdxl-lightning-progressive-adversarial-diffusion-distillation) and the one-/few-step diffusion line.) On top of that, **rolling distillation** fights long-horizon degradation: the student is rolled out over consecutive streaming units and trained on *its own generated history*, using a **self-forcing** strategy with **distribution matching** to align the student's trajectory with the teacher under realistic rollout conditions.

That last piece is worth dwelling on, because it names a real problem. In training, if you always feed the model ground-truth history (teacher forcing), it never learns to cope with its own mistakes; at inference it feeds on its own outputs, drifts off the training distribution, and quality collapses over a long video. Self-forcing closes that gap by rolling the student out on its *own* samples during training, and distribution matching supplies the objective that keeps the student's output distribution close to the teacher's. This is the mechanism that lets the model hold identity, gaze, and scene state over a long call instead of melting after ten seconds.

**Why it works / when it fails.** Staging is the pragmatic backbone: it lets each capability be trained in the cheapest regime that will teach it, and it puts the expensive interaction data last. The exposed risk is that everything about *quality* rides on Stage 3 — CFG distillation and self-forcing are individually delicate, and the report gives no ablation isolating how much each contributes or how far quality drops without rolling distillation. We are told the recipe; we are not shown the sensitivity.

### 7. Thinker–performer: how they reach ~200 ms

**The problem.** Even with a causal, few-step model, the per-unit work is real: encode the current user audio-video, update the Transformer state, run the flow-matching solver to generate the next latents, and decode the previous latents for emission. Do those *serially* on one GPU and the per-unit wall time blows past the 160 ms budget.

**The intuition.** Split the work across two GPUs by *cost profile*, and overlap it in a pipeline so the cheap work hides under the expensive work. One GPU — the **thinker** — does everything except the expensive generation: it perceives, updates the shared state, and decodes. The other — the **performer** — does *only* the flow-matching solver, the single most expensive step. Then you stagger them: while the performer is grinding out the next unit's latents, the thinker is simultaneously perceiving the current unit and decoding the *previous* unit's latents for immediate emission. Nobody waits.

![Figure 2 from Wan Team (2026): the thinker–performer overlap. At unit k the thinker encodes the current user observations, updates the KV cache, and decodes the previous response for emission, while the performer runs only the flow-matching solver for the next latents; KV slices and latents cross the boundary each unit.](/imgs/blogs/wan-streamer-realtime-interactive-fig2.webp)

**The mechanism.** After a system prefill, the thinker broadcasts the initial KV cache to the performer so both sides share the same full-history state. Then, at streaming step $k$:

1. The thinker consumes the current user audio-visual observations, applies the causal encoders, and runs token-causal decoding over the language and state slots to produce the **current KV-cache slice** $\text{KV}_k$.
2. Around the same communication boundary, the thinker *receives* from the performer the clean audio/video latents produced in the *preceding* step, *sends* the new KV slice to the performer, and *decodes* the returned latents into the audio-visual output emitted immediately.
3. The performer appends the received KV slice into its own full-history cache and runs the flow-matching solver *only* for the next audio-visual latent unit. The resulting clean latents stay on the performer and are sent back to the thinker at the next step.

The elegance is that the unified causal state is preserved through **KV exchange** — the performer always has the full history it needs to generate correctly, because the thinker keeps shipping it the newest KV slice — while the expensive latent generation lives on the performer and the latency-critical perception/decoding lives on the thinker. Because the performer never runs decoders and the thinker never runs the flow-matching solver, the deployment preserves the *semantics of one unified model* while overlapping almost all of the latency-critical work.

**Two different clocks.** The report is careful, and correct, to separate two numbers that are easy to conflate:

- **Throughput** is governed by the performer's wall time: the system runs in real time as long as the performer time *plus* the small KV/latent communication fits inside one 160 ms unit. This is what the "short thinker work is hidden under the longer performer window" caption in the figure means.
- **Model-side response latency** is the full signal-to-signal path: from a 160 ms user unit becoming available to the thinker, through encoding, thinker state update, performer latent generation, and decoding, to the response unit being emitted at 25 FPS. That sum is **~200 ms**. Add a **350 ms** bidirectional network budget and total interaction latency is **~550 ms** for a remote user.

On top of the pipeline, the report stacks the usual systems wins — CUDA graph capture, compilation, optimized kernels, and the KV-cache exchange — to push throughput.

**A worked micro-example of the overlap.** Read the figure as four adjacent columns. In column $k$: thinker does `Encode u_k → Update KV_k → Decode y_{k-1}` while the performer does `Generate y_k latents`. The thinker's three short ops finish well inside the performer's one long op, so when the performer hands back $y_k$ at the boundary, the thinker immediately decodes and emits it in column $k+1$, and the cycle repeats. The "latency" you perceive is one unit of pipeline fill (~200 ms model-side); the "throughput" is set purely by whether `performer + comms < 160 ms`.

**Why it works / when it fails.** This is a genuinely clean piece of systems design: partition by cost, overlap by dependency, and preserve correctness with a KV broadcast. But note what the 200 ms number *is*: a measurement of one specific two-GPU serving path, not a decomposed budget. The report does not break the 200 ms into encode/update/generate/decode, does not report the KV/latent transfer size or bandwidth assumption behind "negligible," and the 350 ms network figure is a *budget*, not a measurement of a real network. The architecture is convincing; the latency accounting is a headline, not a reproducible breakdown.

## Experiments and results

The experimental section is a *latency and scope* comparison, not a quality benchmark — there are no FID/FVD/MOS/WER tables, no human-preference studies, no ablations. Read it for what it is: a careful argument about *measurement boundaries*, making the case that Wan-Streamer's ~550 ms covers a path that most reported numbers do not.

**The measurement-boundary argument.** The report's most useful contribution here is methodological: it insists you read latency tables *by boundary*, not by smallest raw number. Public systems variously report model-internal latency, first-packet latency, first-token latency, endpointing time, or API time-to-first-byte — all useful for engineering, none identical to "the delay a remote user perceives." And many "omni" systems accept audio/video input but never close the loop with a *synchronized visual agent output*. So the report keeps model-only, first-packet, and API/product numbers separate rather than stacking them into one misleading figure.

**Table 1 — response latency across speech and omni-modal systems.** ("N/R" = no aligned absolute response latency publicly reported.)

| System | Interaction | User-visible response | Other reported metric | Boundary caveat |
|---|---|---|---|---|
| Doubao Realtime Voice | speech↔speech | ~1 s overall | ~700 ms bare-model | Speech-only product; no visual output |
| Seeduplex | speech↔speech | N/R absolute | −250 ms endpoint, −300 ms interruption vs. prior Doubao | Relative improvement; speech-only |
| GPT-4o / Realtime API | speech↔speech, A/V in | protocol-dependent | 232/320 ms audio; ~500 ms API TTFB; ~800 ms target voice-to-voice | Mixes model, API TTFB, endpointing, network |
| Hume EVI 3 | speech↔speech | 0.9–1.4 s web-app | under 300 ms model response | Vendor benchmark; no visual stream |
| Gemini Live API | speech↔speech | 1.2–3.6 s API benchmark | N/R model-side | Vendor benchmark |
| Sesame web app | speech↔speech | 0.8–1.2 s web-app | N/R model-side | Vendor benchmark; speech-only |
| Moshi | speech↔speech | N/R product | 160 ms theoretical; 200 ms practical model | Native full-duplex *speech*; no visual agent |
| Qwen3/3.5-Omni | A/V/text in, speech/text out | N/R interaction loop | first-packet 234/547 ms; Flash 235/426 ms; Plus 435/651 ms | First-packet; no synchronized avatar |
| MiniCPM-o 4.5 | A/V in, speech/text out | N/R interaction loop | 0.58 s first-token; RTF 0.20–0.27 | First-token/RTF; no avatar |
| **Wan-Streamer (ours)** | **text/audio/video in & out** | **~550 ms total (incl. 350 ms network)** | **~200 ms model-side; 25 FPS video** | **One end-to-end model; text I/O, speech, and synchronized video share one causal stream** |

The honest reading: several speech-only systems report *lower* raw model numbers (Moshi's 200 ms practical model latency, GPT-4o's 232 ms audio response). Wan-Streamer's claim is not "smallest number" — it is "smallest number *that also generates a synchronized visual response*." A speech-only 200 ms and an audio-visual 200 ms are not the same achievement, and the table is structured to make that legible rather than to win a leaderboard.

**Table 2 — runtime and covered scope for visual agents.** These are mostly component-level runtime metrics, not aligned response latencies, split into full-loop systems vs. rendering/generation components.

| System | Visual scope | Reported runtime | Difference from Wan-Streamer |
|---|---|---|---|
| Body of Her | end-to-end humanoid agent | next frame within 42 ms @ 24 FPS | Preliminary; no deployed signal-to-signal latency |
| MIDAS | digital-human video synthesis | real-time frame-by-frame | No absolute response latency |
| U-Mind | text/speech/motion/video loop | real-time rendering claimed | Text-first pipeline; no public breakdown |
| X-Streamer | open-ended video chat from a portrait | 25 FPS on two A100 | Absolute response latency not disclosed |
| VASA-1 | audio-driven talking face | 40 FPS, 170 ms preceding latency | Renderer only; no dialogue or user perception |
| StreamAvatar | talking/listening avatar | FFD 0.33–0.39 s; video latency ~1.20 s | Driven by external speech; no dialogue model |
| AvatarForcing (Cui et al.) | one-step streaming talking avatar | 34 ms/frame; 0.51 s audio-to-visual | Strong visual metric, not perceptual dialogue |
| LiveTalk | interactive avatar video | 24.82 FPS; 0.33 s first-frame | Uses Qwen3-Omni for speech reasoning |
| Hallo-Live | text-driven joint audio-video | 20.38 FPS; 0.94 s latency | Text-driven; no continuous user perception |
| OmniForcing | text-to-audio-video streaming | TTFC ~0.7 s; ~25 FPS | First-chunk latency, not user response |
| **Wan-Streamer (ours)** | **perceptual A/V dialogue + synchronized speech/video** | **25 FPS; ~550 ms total; ~200 ms model-side** | **One causal Transformer learns text I/O, perception, speaking, listening, interruption, and visual response together** |

The load-bearing distinction across both tables is *scope*: a renderer can be fast once clean audio or text is handed to it, but its *perceived* latency still depends on the upstream dialogue and speech stack it does not include. Wan-Streamer's argument is that it reports the *whole* remote path in one number because the whole path is one model.

**What's load-bearing in their setup that might not transfer.** Three things. First, the 200 ms rests on a *two-GPU* serving path with a specific thinker/performer partition — a single-GPU or edge deployment would not overlap the same way, and the number would move. Second, the 350 ms network budget is an assumption; a worse network turns 550 ms into something the user actually notices. Third, the whole comparison is about *latency and scope*, so it is silent on the axis a product manager would ask about next — is the video *good*? At 192p, with no perceptual metric reported, that question is genuinely open.

**Naturalness, interruption, and proactive speaking.** Beyond speed, the report claims qualitative behaviors that the full-duplex design is supposed to produce: in the *idle* state the agent maintains identity, gaze, posture, breathing, and subtle facial motion rather than freezing into a portrait; in the *listening* state it produces gaze shifts, nods, and micro-expressions temporally coupled to the user; and because speech and video latents are predicted from the same context *before* decoding, lip motion and prosody are synchronized natively. It also claims interruption handling (stop/shorten/redirect mid-response) and proactive speaking (comment on salient visual events unprompted). These are the payoff of §2's always-on perception — and they are entirely qualitative in the report. Compelling as design goals, unmeasured as results.

## Critique

**What is genuinely strong.** The thesis is right and cleanly argued: real-time full-duplex interaction is a modeling problem, not a serving problem, and a cascade has thrown away the information needed to solve it before serving ever begins. The commitment to *end-to-end causality* — causal VAEs, causal encoders, causal decoders, block-causal attention — is the honest consequence of that thesis, and the report follows it all the way down instead of stopping at a causal Transformer with bidirectional components smuggled underneath. The thinker–performer split is an elegant, correctness-preserving way to overlap the work, and the KV-exchange trick genuinely keeps "one model" semantics across two GPUs. And the latency-by-boundary framing in the experiments is a public service — it is the most intellectually honest latency comparison I have seen in this space, precisely because it refuses to cherry-pick the smallest number.

**What is weak or unfalsifiable.** The report is a design document wearing the clothes of an evaluation. There is essentially no quantitative result: no video-quality metric, no speech-quality metric, no lip-sync accuracy, no interruption-latency distribution, no false-interrupt rate, no long-horizon-drift curve, and — most tellingly — *no ablations*. Every architectural choice (block-causal vs. token-causal, coupled vs. independent audio/video generation, self-forcing vs. teacher forcing, two-GPU vs. one) is asserted to matter; none is isolated. The "no degradation" style claims about understanding and dialogue are hedged with "approaching" and "comparable" and backed by no table. The 200 ms is a single serving-path number, not a decomposed budget, and the 350 ms network is a stipulation. The 192p resolution and "scaling is straightforward" line are exactly where I would want data and get a promise.

**What ablation or baseline is missing.** The one I want most is the **coupling ablation**: generate audio and video with *independent* flow-matching heads on the same backbone and measure lip-sync/prosody synchronization against the coupled version. The paper's central quality claim — that synchronization emerges *natively* rather than being repaired — is untested without it. Second, an **interruption-latency benchmark** against a cascaded baseline with a tuned VAD: the full-duplex claim lives or dies on whether the model actually yields faster and more appropriately than a well-engineered turn-taking state machine. Third, a **latency decomposition** so the 200 ms can be reproduced and attributed.

**What would change my mind.** I currently read Wan-Streamer as a *promising architecture with an unproven quality story*. What would move me from "promising" to "state-of-the-art" is a single quantitative result that a cascade cannot match: a controlled study showing that at equal or better video/speech quality, the end-to-end model's interruption and response latency beats a strong cascaded baseline — or, conversely, a coupling ablation showing the native-synchronization claim is real. Absent that, the honest verdict is that the report proves the *design is feasible and low-latency*, not that it is *better*.

## What I'd build with this

These are my extrapolations, not the paper's claims — directions the design invites.

- **A coupling-ablation harness.** Fork the generation head into (a) coupled audio+video velocity prediction and (b) two independent heads, hold everything else fixed, and measure a lip-sync metric (e.g., SyncNet-style offset/confidence) plus a prosody-alignment score across a fixed test set. This is the missing experiment, and it is the one that would most directly validate or puncture the central quality claim.
- **A KV-eviction policy for long calls.** Full-history streaming means the KV cache grows unbounded; a two-hour call will hit memory and attention-cost walls. I would prototype a block-level eviction/compression policy (keep recent units at full resolution, summarize distant history) and measure identity/scene drift as a function of retained context — turning §1's "feature" into a tunable knob.
- **A single-GPU degradation curve.** The 200 ms depends on the two-GPU overlap. I would measure how model-side latency and achievable resolution/FPS degrade on one GPU, and where the performer's flow-matching solver becomes the binding constraint — the honest answer to "can this run on a phone?" is a curve, and the report does not draw it.
- **An interruption-latency benchmark suite.** Build a scripted set of interruption scenarios (user cuts in early/late, overlaps, backchannels) and measure the model's yield latency and appropriateness against a tuned cascade baseline. This operationalizes the full-duplex claim into a number product teams can compare.
- **A resolution-scaling study.** Take the "scaling to higher resolutions is straightforward" claim at its word and test it: retrain the causal VAE/decoders at 360p/720p and report the quality-vs-latency Pareto front. If it is straightforward, the front should move cleanly; if it is not, that is the most important thing to know about v0.2.

## References

- **Paper:** Wan Team, Alibaba Group. *Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models.* arXiv:2606.25041 (cs.CV), 2026. [arxiv.org/abs/2606.25041](https://arxiv.org/abs/2606.25041) · Project site: [wan-streamer.com](https://wan-streamer.com/)
- **Sibling omni model:** [Qwen3-Omni: One Model for Text, Image, Audio, and Video](/blog/paper-reading/multimodal/qwen3-omni-technical-report) — the thinker–talker split and first-packet latency Wan-Streamer benchmarks against.
- **Streaming avatar prior art:** [Live Avatar: Streaming Real-time Audio-Driven Avatar Generation](/blog/paper-reading/multimodal/live-avatar-streaming-real-time-audio-driven-avatar-generation-with-infinite-length) — the renderer-side lineage of the avatar half.
- **Latent generation background:** [High-Resolution Image Synthesis with Latent Diffusion Models](/blog/paper-reading/diffusion-model/high-resolution-image-synthesis-with-latent-diffusion-models) — the VAE latent space flow matching operates in.
- **Few-step distillation:** [SDXL-Lightning: Progressive Adversarial Diffusion Distillation](/blog/paper-reading/diffusion-model/sdxl-lightning-progressive-adversarial-diffusion-distillation) — the "great slow teacher → fast student" move behind Stage 3.
