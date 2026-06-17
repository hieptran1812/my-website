---
title: "The 2026 Video Model Landscape: Who Leads What, and How to Choose"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A buyer's-and-builder's map of the 2025–2026 video-generation field — the axes that matter, the leaderboards that lie, the converged recipe, and a decision guide that tells you which model to reach for and why."
tags:
  [
    "video-generation",
    "diffusion-models",
    "text-to-video",
    "video-diffusion",
    "model-comparison",
    "veo",
    "sora",
    "kling",
    "wan",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-2026-video-model-landscape-1.png"
---

A client sends you one line: "We need a 12-second product clip, with a voiceover, character has to stay on-model across three shots, and it can't leave our cloud — we're a healthcare company." You open your tabs. Veo 3.1 nails the audio and the polish, but it's an API and the bytes leave your VPC. Wan 2.2 runs on the A100 you already rent, but it has no native audio and tops out around five seconds before you start tiling. Kling 3.0 does multi-shot character consistency better than anything, but it's also an API. Sora 2 is stunning and also an API. LTX-2 renders faster than real time, which you don't need, and has no audio, which you do.

There is no single answer because there is no single best model. That is the whole point of this post. By 2026 the video-generation field has split into two tracks — a set of **closed, hosted, API-only leaders** competing on raw quality, native audio, physics, and consistency, and a set of **open, self-hostable models** competing on access, cost, and control — and the gap between them has narrowed to the point where the right choice is dictated less by "which is best" and more by "which constraint binds your job." Figure 1 is the scorecard I keep open during these calls: five representative models across five axes, where you can see at a glance that nobody owns every column.

![A scorecard matrix comparing Veo 3.1, Sora 2, Kling 3.0, Wan 2.2, and LTX-2 across access, native audio, max length, consistency, and self-hostability, showing each model leads on a different axis](/imgs/blogs/the-2026-video-model-landscape-1.png)

This is the synthesis capstone of the series. The earlier posts dissected the machine — the causal 3D-VAE that makes length affordable, the spacetime-patch DiT that does the generation, flow matching that trains it, the conditioning paths that steer it, and the [metrics that judge it](/blog/machine-learning/video-generation/the-metrics-of-video-generation). This post zooms back out. It is the map you hand someone who asks "okay, but which one do I actually use, and why?" By the end you will have a principled framework for comparing video models on the right axes (not vibes), an honest read of what the leaderboards do and do not measure, an understanding of why every serious model has converged on the same recipe — and where the real differentiation hides — and a decision guide that points each kind of job at the model that fits it. We will also build a small reproducible comparison harness so you can stop trusting vendor charts and make your own leaderboard. Throughout, the recurring frame from the [foundations post](/blog/machine-learning/video-generation/why-video-generation-is-hard) still governs everything: video is spatial generation times temporal coherence under a brutal compute budget, and every model on this map is a different point in the trade-off space of **coherence × motion × length × cost**.

One disclaimer I will repeat: the standings and numbers here are **approximate and dated to mid-2026**. Video models ship monthly, version numbers churn, and an arena ranking can flip in a week. Treat specific figures as order-of-magnitude and directional, not as a spec sheet. The *framework* — the axes, the way the leaderboards behave, the converged recipe, the decision logic — is what's durable. Memorize that, and you can re-rank the models yourself when the next wave lands.

## 1. The two-track market, and why it formed

Start with the shape of the field, because the shape explains the strategy of every team in it. There are two tracks, and they are competing on different things.

The **closed track** is the hosted, API-only frontier: Google's Veo (3.1 by mid-2026), OpenAI's Sora (Sora 2), Kuaishou's Kling (3.0), ByteDance's Seedance (2.0), Runway (Gen-4.x), MiniMax's Hailuo, Pika, and Luma's Dream Machine. You send a prompt and get back pixels. You never see the weights. These teams compete on the things that are expensive to do and easy to perceive: maximum visual fidelity, **native synchronized audio**, physical plausibility, and **character and multi-shot consistency**. Their moat is data, compute, and tuning — none of which you can replicate at home — and their business model is selling generations by the second.

The **open track** is the set of downloadable, self-hostable models: Alibaba's Wan (2.x), Tencent's HunyuanVideo (1.5), Zhipu's CogVideoX, Genmo's Mochi, and Lightricks' LTX (LTX-2). You download the weights, run them on your own GPU, and never send a frame to anyone. These models compete on the axes the closed track structurally cannot match: **access** (no API key, no rate limit, no content filter you didn't write), **cost** (your electricity bill, not a per-second meter), and **control** (LoRA fine-tuning, ControlNet-style conditioning, frame-level intervention, and inspection of every intermediate latent). Their quality lags the absolute frontier, but — and this is the story of 2024–2026 — the lag has shrunk from "embarrassing" to "good enough for most jobs."

Figure 2 draws the split as a tree. The root forks into the two tracks, and under each sits a representative slate. The point of the picture is the *fork itself*: a team is on one side or the other, and that placement predicts almost everything about how you'll interact with the model.

![A tree showing the 2026 video model market splitting into a closed API track with Veo, Sora, and Kling and an open self-hostable track with Wan, HunyuanVideo, and LTX](/imgs/blogs/the-2026-video-model-landscape-2.png)

Why did the market form this way rather than, say, everyone open-sourcing or everyone going closed? Three forces. First, **training video models is brutally capital-intensive** — the largest models see tens of millions of GPU-hours and curated video corpora that cost real money to license and clean — so the teams that can afford the frontier want to monetize it, which means an API, not a download. Second, **the open recipe leaked and then converged** (Section 4): once causal 3D-VAE + DiT + flow matching became the obvious stack, a well-resourced lab could train a competitive open model and capture mindshare, developer goodwill, and a fine-tuning ecosystem. Third, **audio and safety are easier to sell than to give away** — synchronized native audio is a genuine moat that closed labs guard, and the liability of an unfiltered generator pushes the highest-fidelity models behind a gate.

The practical upshot for you: your *first* decision is not "which model" but "which track." If the job has a hard privacy, cost, or control constraint, you are on the open track and the closed leaders are irrelevant no matter how good their arena Elo is. If the job needs the absolute top of quality or native audio and the bytes can leave your network, you are on the closed track and the open models are a fallback. Most of the rest of this post is about choosing *within* the track once you've picked the track.

It helps to name the specific players on each side so the map isn't abstract. On the **closed** side, by mid-2026 the recognizable tier is roughly: Google **Veo 3.1** (the audio-and-4K leader, integrated into Google's creative tooling), OpenAI **Sora 2** (the world-simulator brand, strong on emergent 3D consistency and now with synchronized audio), Kuaishou **Kling 3.0** (a quiet arena leader out of China with the best multi-shot character consistency and lip-sync I've seen), ByteDance **Seedance 2.0** (frequently at or near the top of the public arenas, built on ByteDance's enormous short-video data advantage), **Runway Gen-4.x** (the creative-tools incumbent, control-and-editing-forward), MiniMax's **Hailuo** (strong motion, popular for expressive character clips), and the fast-iteration tier of **Pika** and Luma's **Dream Machine** (lower latency, lighter weight, built for quick drafts). On the **open** side: Alibaba **Wan 2.x** (the consumer-GPU milestone, Apache-licensed for most releases), Tencent **HunyuanVideo 1.5** (the largest-quality open DiT, strong aesthetics), Zhipu **CogVideoX** (the documented open reference recipe everyone learned from), Genmo **Mochi 1** (the asymmetric-DiT open T2V model), and Lightricks **LTX-2** (the efficiency-and-speed leader, faster-than-real-time on a single GPU). You do not need to memorize this list — it will be stale in a quarter — but you do need the *shape*: a half-dozen well-funded closed leaders trading blows on quality and audio, and a half-dozen open models trading access for a polish tier.

One subtlety worth flagging before we go further: the two tracks are **not hermetically sealed**. Several closed labs publish detailed technical reports (Movie Gen is the standout) that hand the open track a blueprint, and several "open" releases come with licenses that are open-for-research but restricted-for-commercial-use, which puts them in a gray zone between the tracks. So treat "open vs closed" as the dominant axis but read the license text before you bet a product on it — the difference between Apache-2.0 and a non-commercial community license can be the difference between shipping and a legal review. Section 6's table flags this per model.

## 2. The axes that actually differentiate models

"Which is better" is the wrong question because video quality is not a scalar. A model can be the best in the world at motion realism and mediocre at text rendering; another can hold a character's face perfectly across a minute and fumble a glass of water pouring. To compare models honestly you need to fix the axes first, and — this is the scientific core of the post — you need to **define each axis operationally**, so the comparison is principled rather than a beauty contest. Here are the axes that matter, each with a concrete way to measure it.

**Maximum resolution.** The native output grid. Veo 3.1 advertises native 4K; most open models train at 480p–720p and upsample. Measure it honestly by generating at the claimed resolution and checking for genuine high-frequency detail (FFT energy in the high bands) versus a bilinear upscale wearing a 4K filename.

**Native audio.** Does the model emit a *synchronized* audio track jointly, or do you bolt on a separate text-to-speech and lip-sync pass? This is binary in capability but graded in quality. Measure sync with an audio-visual offset detector (the kind used for lip-sync evaluation) and measure whether the audio is *semantically* tied to the scene (does the dog's bark land when its mouth opens?).

**Maximum clip length.** The longest *coherent* clip, not the longest clip that renders. Many models will happily produce 30 seconds that drift off-character at second 12. Measure the usable length as the point where [subject consistency](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) drops below a threshold, not the API's max-duration parameter.

**Motion realism and physics.** Does the motion obey learned physics — gravity, momentum, contact, fluid — or does it slide and morph? This is the hardest axis to measure and the easiest to fake. Operationally, score it on a physics-probing benchmark: prompts with a known correct outcome (a ball dropped, a cloth draped, water poured) and a rubric or learned scorer for plausibility. VBench's *dynamic degree* and *motion smoothness* are proxies, but they are gameable (Section 5).

**Character and multi-shot consistency.** Does a person, product, or character keep the same identity across a long clip and across *cuts* between shots? This is the axis enterprise buyers care about most and the one open models lag hardest. Measure it with [VBench's subject-consistency](/blog/machine-learning/video-generation/the-metrics-of-video-generation) dimension within a shot, and with a face/identity-embedding distance across shots for the multi-shot case.

**Control surface.** How many knobs do you get? First/last-frame conditioning, motion strength, camera trajectory, reference image, LoRA fine-tuning, ControlNet/pose, region masks. Open models win this axis by construction — you have the weights — while closed models expose whatever the API designers chose to surface.

**Price and latency.** Cost per second of output and wall-clock seconds per clip. For closed models this is the published per-second price; for open models it is your GPU's amortized cost plus the seconds-per-clip we measure in Section 8.

**License and self-hostability.** Can you run it on-prem, and can you use the output commercially? Apache-2.0 (some Wan/LTX releases) versus a custom non-commercial or "community" license versus a closed API's terms of service. This is a yes/no that can veto a model regardless of quality.

The reason to enumerate these is that *the right model for a job is the one whose strong axis matches the job's binding constraint*, and you can only see that match if you've named the axes. A vendor chart that collapses all of this into one "quality" number is, at best, lossy and, at worst, marketing. The whole rest of the post is built on these nine axes.

A word on *why operational definitions matter so much here*, because it's the difference between a principled comparison and an argument. Take "physics," the axis people argue about most. If "good physics" means "looks physically plausible to me," then every comparison is a vibe and two reasonable engineers will disagree forever. But if "good physics" means "on a fixed set of 50 prompts with known correct outcomes — a ball dropped from a height lands and bounces with decaying amplitude, a cloth draped over a sphere conforms to it, water poured into a glass fills from the bottom — a rubric or learned scorer rates the outcome 0–1," then "good physics" is a *number you can re-run*, two engineers can disagree about the rubric but not about the score, and a model's physics claim becomes falsifiable. The same move works for every axis: "consistency" becomes "mean pairwise identity-embedding similarity of the main subject across frames," "audio" becomes "lip-sync offset in milliseconds plus a semantic-grounding score," "length" becomes "the clip duration at which subject-consistency crosses 0.90." Operationalizing the axis is what turns the landscape from a flame war into engineering. It's also what lets you re-rank the models yourself when the next version ships — the *definitions* are stable even when the *numbers* churn.

There's a tenth axis lurking that I'll mention but not table, because it's softer: **prompt adherence and steerability** — how faithfully the model follows a complex prompt, and how predictably small prompt edits change the output. The closed leaders generally win here too (it's a tuning-and-data axis), and it interacts with the control surface: a model with few control knobs but excellent prompt adherence can be easier to steer than a model with many knobs but erratic adherence. Measure it with a text-alignment score (CLIP-style or, better, a VLM-as-judge that reads the prompt and the clip and scores whether each requested element appears). I leave it off the main table only to keep the table scannable, not because it doesn't matter.

#### Worked example: turning "which is better" into a vector

Suppose you're choosing between Model A and Model B for a talking-head explainer. Collapsed to a scalar, A scores 8.7 and B scores 8.5 on some arena, so you pick A. Now decompose. The job's binding constraints are: native audio with lip-sync (weight high), 60-second coherent length (weight high), 1080p (weight medium), motion realism (weight low — it's a talking head). On the *audio* axis, B emits synchronized lip-synced audio natively and A bolts on TTS with a 120 ms average sync offset. On the *length* axis, B holds subject consistency to 60s while A's identity drifts (subject-consistency VBench score falls from 0.96 to 0.88 past 40s). A's 0.2-point arena edge came almost entirely from *aesthetic quality on short clips* — an axis your job weights low. Decomposed, **B is the correct pick** and the scalar was actively misleading. This is why we fix axes before we compare.

## 3. The leaderboards, and exactly how they lie

Now to the leaderboards, because you will be tempted to outsource the decision to them, and you shouldn't — not because they're useless, but because each measures something narrow and each can be gamed in a specific way. There are two families, and they fail differently. Figure 5 contrasts them.

![A before-after diagram contrasting an arena leaderboard that produces one non-reproducible Elo from human votes against VBench which produces reproducible but gameable per-axis scores](/imgs/blogs/the-2026-video-model-landscape-5.png)

**Arena-style leaderboards** (the Artificial Analysis video arena and similar) work like the LLM chatbot arenas: show a human two clips from the same prompt, ask which is better, aggregate the pairwise votes into an Elo rating. What this measures is *human preference under blind comparison*, which is genuinely valuable — it's the closest thing to "which clip would a person rather watch." But it has three structural problems. First, it is **not reproducible**: the prompt set shifts, the voter pool shifts, and you cannot re-run last month's ranking to check it. Second, it collapses everything into **one blended number** — a model that wins on aesthetics and loses on motion gets one Elo, and you can't see the trade. Third, it is **preference-biased toward the salient**: voters reward sharpness, color, and "wow" over physical correctness, so a pretty-but-physically-wrong clip can out-Elo an accurate-but-plainer one. Arena Elo answers "which looks better to people," not "which is more correct," and definitely not "which fits my job."

**Reproducible benchmarks** ([VBench and VBench-2.0](/blog/machine-learning/video-generation/the-metrics-of-video-generation)) go the other way: a fixed prompt suite, fixed seeds, and a battery of automatic scorers for sixteen-plus dimensions — subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality, object class, multiple objects, spatial relationship, temporal flicker, and so on. The win is **reproducibility**: you can re-run VBench on your own model with the same prompts and seeds and get a comparable number, and you get **per-axis** scores so you can see *where* a model is strong. The catch is that automatic scorers are **gameable per axis**, and the gaming is not hypothetical. The cleanest example is the **dynamic-degree-versus-stability trap**: subject-consistency and motion-smoothness both go *up* when the video barely moves, so a model that outputs a near-still image scores beautifully on "consistency" and "smoothness" while failing the actual job of generating motion. VBench mitigates this with the dynamic-degree axis, but you have to *read all the axes together* — a high consistency score is only meaningful next to a healthy dynamic degree. Optimize one VBench dimension in isolation and you will ship a worse model with a better scorecard.

The honest synthesis: **arena ≠ reproducible benchmark, and neither is your job.** Use the arena to sanity-check "is this model in the top tier people like," use VBench to see the per-axis profile and to track your own model across training, and use *your own task-specific eval* (Section 8) to make the actual decision. Any time a vendor leads with a single leaderboard rank and no per-axis breakdown, mentally discount it. The number is real; it just isn't the number you need.

It's worth being precise about a third failure mode that affects *both* families: **prompt-set mismatch.** Every leaderboard is implicitly a leaderboard *on its prompt distribution*. The Artificial Analysis arena tends toward cinematic, aesthetically-loaded prompts because that's what voters enjoy comparing; VBench's suite is deliberately diverse but still finite and known. If your job is "generate top-down clips of industrial machinery for a training video," neither leaderboard's prompt distribution looks anything like yours, and the rankings transfer poorly. This is not a flaw you can patch by picking a "better" leaderboard — it's intrinsic to the idea of a fixed benchmark, and it's the deepest reason the eventual decision has to run *your* prompts. A leaderboard tells you which models are *generally* strong; it cannot tell you which is strong *on your distribution*, and the distribution gap can flip rankings entirely. When someone shows you a chart, the first question is "on what prompts?" — and if the answer is far from your use case, the chart is a weak prior at best.

There's also a subtler statistical point about arena Elo: the rankings near the top are often *within noise* of each other. When the top five models are separated by 30 Elo points and the confidence intervals are 25 points wide, the "rank 1" model is not reliably better than "rank 3" — they're a statistical tie, and the ordering can flip with the next batch of votes. Reading an arena, look at the *gaps and the error bars*, not just the order: a model that leads by 80 points with tight intervals is genuinely ahead; a model that leads by 15 is sharing the top tier. Treating a noisy ordinal rank as a precise cardinal quality is one of the most common ways teams overpay for a model that isn't actually better at their job.

#### Worked example: reading a VBench row without fooling yourself

A model report claims a VBench *subject consistency* of 0.97 and a *motion smoothness* of 0.99 — both near the top of the table. Before you're impressed, look two columns over: its *dynamic degree* is 0.21, near the bottom. Decoded: this model produces clips that hold identity and flow smoothly *because they barely move*. The 0.97/0.99 are not evidence of a great model; they're evidence that the dynamic-degree axis is doing its job exposing a near-static generator. A genuinely strong model posts, say, subject consistency 0.94, motion smoothness 0.97, *and* dynamic degree 0.62 — slightly lower on the first two, dramatically higher on the third. The "worse" scorecard is the better model. Always read the consistency and smoothness numbers *as a ratio against dynamic degree*; in isolation they reward stillness.

## 4. The converged recipe — and where the differentiation actually is

Here is the fact that reframes the whole landscape: **the architectures have converged.** If you cracked open Wan, HunyuanVideo, CogVideoX, Mochi, LTX, and — as far as their reports reveal — Sora, Veo, Kling, Seedance, and Movie Gen, you would find astonishingly similar machinery. Figure 4 lays out the shared stack.

![A stack showing the converged 2026 recipe from curated video data through a causal 3D-VAE, an MM-DiT denoiser, flow matching, optional joint audio, and preference tuning to final frames](/imgs/blogs/the-2026-video-model-landscape-4.png)

The shared recipe, top to bottom, is:

1. **A causal 3D-VAE** that compresses video jointly in space and time — typically something like 4× temporal and 8×8 spatial, for a 256× volumetric reduction in element count, sometimes far more aggressive. This is, as the [autoencoder post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) argues, the real enabler of length and cost, because it sets how many tokens the denoiser must process.
2. **An (MM-)DiT denoiser** operating on spacetime tokens — a [diffusion transformer](/blog/machine-learning/image-generation/diffusion-transformers-dit) adapted to video, where the "MM" (multimodal) variants jointly attend over text and video tokens in a single stream rather than cross-attending into a frozen backbone.
3. **Flow matching** as the training objective — [rectified-flow / conditional flow matching](/blog/machine-learning/video-generation/flow-matching-for-video) with a resolution-dependent timestep shift — because its straight probability paths scale cleanly to the long-token video regime.
4. **Scale** — more parameters, more data, more compute — because the [scaling thesis](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) holds in video: coherence and physical plausibility improve with compute in a way that no architectural trick substitutes for.
5. **Preference tuning** — RLHF, DPO, or reward-model fine-tuning on top of the base diffusion model, to align outputs with what humans actually want to watch.

The components differ in detail — Mochi's asymmetric DiT puts more parameters on the visual stream, LTX pushes an extreme-compression VAE for speed, the [open-frontier models](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox) vary their VAE ratios — but the *template* is shared. This is the single most important thing to internalize about the 2026 landscape: **architecture is no longer the differentiator.** You cannot explain why Veo's clips look better than an open model's by pointing at a cleverer attention pattern, because they're running the same attention pattern.

So where *does* the differentiation come from? Four levers, none of them architectural:

- **Data.** The biggest and least-visible moat. The closed labs train on enormous, expensively curated, licensed video corpora with rich captions; the quality, diversity, and caption fidelity of that data shows up directly in the output. You cannot download Google's training set.
- **Compute.** The frontier closed models see, plausibly, 10–100× the training compute of a typical open model. Compute buys coherence, physical plausibility, and the long tail of rare-scene quality. This is the most expensive lever and the clearest source of the open-closed gap.
- **Audio.** Native synchronized audio (Veo 3, Movie Gen Audio) is a genuine capability the open models mostly lack, and it's a hard problem — joint audio-video generation with [proper sync](/blog/machine-learning/video-generation/audio-and-joint-av-generation) — so it's a real differentiator, not just a feature checkbox.
- **Tuning.** Preference optimization and the curation that goes into it. Two models with identical architectures and similar data can diverge a lot in *perceived* quality based on how aggressively and well they're tuned to human preference.

Figure 6 makes this explicit as a causal graph: a shared architecture flows into the levers that actually move quality — data and compute feed tuning, and tuning closes (or doesn't) the gap. Read it as the answer to "why is the closed model better": not because of a secret layer, but because of more data, more compute, and better tuning, all of which are buyable with capital, not cleverness.

![A graph showing a shared architecture feeding into licensed data and training compute, both feeding preference tuning, which drives the shrinking open-closed quality gap](/imgs/blogs/the-2026-video-model-landscape-6.png)

This is *good news* for the open track. If the gap were architectural, it would be permanent — you'd need the secret design. Because the gap is data, compute, and tuning, it is a *funding* gap, and funding gaps close: open labs get bigger budgets, open datasets improve, and open preference-tuning recipes mature. That's exactly the trajectory of 2024–2026, which we trace next.

### Why this convergence happened — a short derivation

It's worth making the convergence feel inevitable rather than coincidental. The video generation problem is: model a distribution over high-dimensional spatiotemporal data $x \in \mathbb{R}^{T \times H \times W \times 3}$ under a fixed compute budget. Three pressures force the recipe:

First, **the token count is the cost**, and the cost is super-linear. A transformer denoiser over $N$ spacetime tokens costs $O(N^2)$ for full attention. Raw video has $N \propto T \cdot H \cdot W$ enormous, so *something* must shrink $N$ before the denoiser sees it — hence a heavy autoencoder. And because the redundancy in video is *both* spatial and temporal (adjacent frames are nearly identical), the autoencoder must compress *both* axes — hence a 3D (not 2D) VAE. The causal-in-time variant wins because it lets you encode/decode streaming and extend length without retraining the whole VAE. The 3D-VAE is not a design choice; it's forced by the arithmetic.

Second, **the denoiser must mix information globally across space and time**, because coherence is a long-range property — the object at frame 1 must match the object at frame 48. Convolutions have bounded receptive fields; attention does not. So the denoiser converges on a transformer over the compressed spacetime tokens — a DiT. The multimodal variant (joint text-video attention) wins over cross-attention into a frozen image backbone because it lets text steer *every* layer of the video stream, which matters more as prompts get complex.

Third, **the training objective must be stable and scalable over long token sequences.** Flow matching's straight-line transport gives a constant-variance velocity target that trains stably at scale and samples in few steps, which is why it displaced the noise-prediction DDPM objective in the video regime (see [flow matching for video](/blog/machine-learning/video-generation/flow-matching-for-video) for the full argument). Each pressure independently points at one component, and the intersection is exactly the converged recipe. That's why everyone landed in the same place.

Let me make the first pressure quantitative, because it's the one that most directly explains the open-closed economics. Suppose you want to generate a 5-second clip at 720p and 24 fps: that's $T = 120$ frames, $H \times W = 720 \times 1280$ pixels. Raw, that's about $120 \times 720 \times 1280 \times 3 \approx 3.3 \times 10^{8}$ values. A DiT can't attend over a third of a billion tokens — full attention is $O(N^2)$, so even at one token per pixel you'd be doing $\sim 10^{17}$ attention operations per layer, which is absurd. The causal 3D-VAE rescues this. With a typical $4 \times 8 \times 8$ compression (4× in time, 8× in each spatial dimension) the latent grid is $\frac{120}{4} \times \frac{720}{8} \times \frac{1280}{8} = 30 \times 90 \times 160$ latent positions. Patchify that with, say, a $1 \times 2 \times 2$ patch and you get $30 \times 45 \times 80 = 108{,}000$ tokens — a $\sim 3000\times$ reduction from the raw element count, and now $N^2 \approx 1.2 \times 10^{10}$ per layer, which is large but trainable. The VAE compression ratio is therefore the single knob that sets whether a clip length is affordable, and a more aggressive VAE (LTX's high-compression design) is what lets the efficiency-first open models render fast. This is why the [autoencoder post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) calls the 3D-VAE the real enabler: it's not a preprocessing detail, it's the term that decides the whole cost equation, and every model on this map lives or dies by how good its VAE is.

The second pressure deserves one more sentence of rigor because it explains the *MM* in MM-DiT. Coherence is a constraint that couples *all* tokens: the dog at frame 1 and the dog at frame 30 must be the same dog, which is a relationship between tokens 80 layers and 27 frames apart. Only global self-attention expresses arbitrary long-range token coupling in one hop; a convolution would need $O(\log N)$ stacked layers to connect distant tokens and would still leak coherence through its bounded receptive field. And once you're doing global attention over video tokens anyway, the cheapest way to let *text* steer the video is to throw the text tokens into the *same* attention stream rather than cross-attending into a frozen image backbone — that's the multimodal DiT, and it wins because text can then influence every spatial-temporal token at every layer instead of only at the cross-attention bottlenecks. The convergence on MM-DiT is the field discovering that joint attention beats cross-attention for complex prompts. None of this is a secret; it's all in the open reports, which is exactly why the architecture stopped being a moat.

## 5. 2024 → 2026: how the gap narrowed

The timeline is the proof that the open-closed gap is a funding gap, not a permanent one, because you can watch it close. Figure 3 marks the milestones.

![A timeline from the 2024 Sora preview through CogVideoX, Movie Gen, HunyuanVideo, Wan, and the 2026 frontier of Kling 3, LTX-2, and Seedance 2, showing the closed lead narrowing](/imgs/blogs/the-2026-video-model-landscape-3.png)

**Early 2024 — Sora's preview** reset everyone's expectations. The [spacetime-patch-at-scale framing](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) and the minute-long coherent clips were a discontinuity, and crucially Sora was *closed* — a demo, not a download. At that moment the gap looked enormous: nothing open came close.

**Mid-to-late 2024 — the open recipe arrived.** CogVideoX (5B, an open 3D-VAE + DiT) showed a credible open model with a real VAE and a published recipe. Then HunyuanVideo (13B) raised the open ceiling dramatically — a genuinely large open video DiT. In parallel, Meta's Movie Gen report (30B, with native audio and personalization) laid out the closed frontier recipe in the open literature, which — even though the weights stayed closed — handed the whole field a blueprint.

**Early-to-mid 2025 — the gap became a quarter, not a year.** Wan 2.1 landed as an open model you could run on a single 24GB consumer GPU, which mattered enormously: it moved "self-hostable video generation" from data-center-only to prosumer hardware. The frontier closed models answered with native audio (Veo 3) and the second generation of arena leaders (Kling 2.x, Sora 2), pushing quality and audio further — but the *delta* a typical user could perceive between "best open" and "best closed" on a generic prompt had shrunk to something you'd describe as a polish gap, not a capability gap.

**2026 — convergence and specialization.** Kling 3.0, LTX-2, Seedance 2.0, Wan 2.2, HunyuanVideo 1.5. The closed leaders pulled ahead specifically on audio, multi-shot consistency, and physical plausibility — the compute-and-data-heavy axes — while the open models matched them on short-clip aesthetic quality and *won* on access, cost, control, and speed (LTX-2's faster-than-real-time generation has no closed equivalent you can run yourself). The field specialized: instead of one quality frontier, you get a *Pareto frontier* where different models dominate different axes, which is exactly why the decision guide later in this post is a *match*, not a ranking.

The trend line is unambiguous. In two years the open track went from "not in the conversation" to "best-open is one polish-tier behind best-closed, and ahead on every access axis." Extrapolating is risky, but the mechanism — open budgets and datasets growing while the architecture stays shared — points at the gap continuing to narrow on the buyable axes and persisting longest on the most compute-and-data-intensive ones (audio, long-form physical coherence, multi-shot identity).

## 6. The big comparison table

Synthesis demands a single table you can scan. Below is the landscape as of mid-2026, across the axes from Section 2. **Everything here is approximate and dated** — version numbers and prices move monthly, lengths are "usable coherent" estimates not API maxes, and consistency/physics are my read of the reports and arena behavior, not a single reproducible number. Use it as a map, not a spec sheet.

| Model | Track | Access | Native audio | Max coherent length | Consistency / physics | Price & latency (approx) | License |
|---|---|---|---|---|---|---|---|
| **Veo 3.1** (Google) | Closed | API | Yes, synced | ~60s+ | Top-tier | Per-second API; seconds-to-minutes server-side | Closed / ToS |
| **Sora 2** (OpenAI) | Closed | API | Yes, synced | ~20–25s | Top-tier, strong world-sim | Per-second API; server-side | Closed / ToS |
| **Kling 3.0** (Kuaishou) | Closed | API | Yes, lip-sync | ~100–120s | Top multi-shot consistency | Per-second / credits | Closed / ToS |
| **Seedance 2.0** (ByteDance) | Closed | API | Partial / evolving | ~30s+, multi-shot | Arena-top, strong multi-shot | Per-second / credits | Closed / ToS |
| **Runway Gen-4.x** | Closed | API | Limited | ~10–20s | Strong, control-focused | Per-second / credits | Closed / ToS |
| **Hailuo / MiniMax** | Closed | API | Limited | ~6–10s | Strong motion | Credits | Closed / ToS |
| **Pika / Luma** | Closed | API | Limited | ~5–10s | Good, fast iteration | Credits, low latency | Closed / ToS |
| **Wan 2.2** (Alibaba) | Open | Weights | No | ~5s native, tile longer | Good consistency | Self-host; ~24GB-class GPU | Apache-2.0 (most) |
| **HunyuanVideo 1.5** (Tencent) | Open | Weights | No | ~5–10s | Good, strong aesthetics | Self-host; 8.3B DiT, 24–48GB | Custom community |
| **CogVideoX-5B** (Zhipu) | Open | Weights | No | ~6s (49 frames @ 8fps) | Decent, well-documented | Self-host; ~24GB | Apache-2.0 (some variants) |
| **Mochi 1** (Genmo) | Open | Weights | No | ~5s | Good motion, AsymDiT | Self-host; multi-GPU-friendly | Apache-2.0 |
| **LTX-2** (Lightricks) | Open | Weights | No | ~10s | Good, efficiency-first | Self-host; faster-than-real-time | Open (check release) |

Read this table the way you'd read the scorecard in Figure 1, not as a ranking. The closed rows dominate the audio, length, and consistency columns; the open rows dominate the access, license, and self-host columns. There is no row that wins everything, and that's the structure of the market, not an accident of which models I happened to list. The model reports for several of these go deeper — see [Kling](/blog/machine-learning/video-generation/kling-deep-dive), [Seedance and the ByteDance stack](/blog/machine-learning/video-generation/seedance-and-the-bytedance-video-stack), [Movie Gen](/blog/machine-learning/video-generation/movie-gen-deep-dive), [Mochi](/blog/machine-learning/video-generation/mochi-and-asymmetric-dit-video), and [LTX](/blog/machine-learning/video-generation/ltx-video-deep-dive) — and the synthesis of the open trio lives in [the open frontier post](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox).

## 7. Build your own leaderboard: a reproducible comparison harness

Here is the antidote to trusting any of the tables above, including mine: **run the comparison yourself.** Because the strong open models are in 🤗 `diffusers`, you can put the *same prompt and the same seed* through several of them and score them on the same VBench-style dimensions, producing a small reproducible leaderboard for *your* prompts rather than a vendor's. Figure 8 is the shape of the harness: one fixed spec fans out to several open models, each output gets scored on identical axes, and you collect a reproducible board.

![A graph showing a fixed prompt and seed fanning out to Wan, CogVideoX, and LTX in diffusers, each scored on the same VBench subject and motion axes to produce a reproducible leaderboard](/imgs/blogs/the-2026-video-model-landscape-8.png)

The discipline that makes this *fair* is the same discipline that makes any benchmark fair: fix the prompt, fix the seed, fix the resolution and frame count, warm up the GPU before timing, and score every model on the identical axes. Here's a harness sketch. It generates from three open models and exports each clip; the scoring functions are kept small and explicit so you can see what they measure.

```python
import torch
from diffusers import (
    WanPipeline,
    CogVideoXPipeline,
    LTXPipeline,
)
from diffusers.utils import export_to_video

# --- the fixed spec: identical for every model under test ---
PROMPT = "a golden retriever running across a sunny beach, waves behind it, cinematic"
SEED = 0
NUM_FRAMES = 49          # ~6s at 8 fps; pick one and hold it
HEIGHT, WIDTH = 480, 720 # fix resolution so the comparison is apples-to-apples
FPS = 8
DTYPE = torch.bfloat16

def load(model_id, pipe_cls):
    pipe = pipe_cls.from_pretrained(model_id, torch_dtype=DTYPE)
    pipe.enable_model_cpu_offload()   # fit on a single 24GB card
    pipe.vae.enable_tiling()          # VAE decode is often the VRAM wall
    return pipe

# Map a short key to (repo, pipeline class). Swap in your own checkpoints.
MODELS = {
    "wan":  ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", WanPipeline),
    "cog":  ("THUDM/CogVideoX-5b",               CogVideoXPipeline),
    "ltx":  ("Lightricks/LTX-Video",             LTXPipeline),
}

def generate(key):
    repo, cls = MODELS[key]
    pipe = load(repo, cls)
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    # warm up so the first-call compile/cuDNN autotune doesn't pollute timing
    _ = pipe(prompt=PROMPT, num_frames=9, height=HEIGHT, width=WIDTH,
             num_inference_steps=4, generator=gen)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = pipe(prompt=PROMPT, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH,
               num_inference_steps=50, generator=gen).frames[0]
    end.record(); torch.cuda.synchronize()
    secs = start.elapsed_time(end) / 1000.0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    export_to_video(out, f"out_{key}.mp4", fps=FPS)
    del pipe; torch.cuda.empty_cache()
    return out, secs, peak_gb
```

Now the scoring. You do not need the full VBench harness to start — two cheap, defensible dimensions get you most of the signal: **subject consistency** (how stable is the main subject's appearance across frames) approximated by the mean cosine similarity of per-frame image embeddings, and **motion magnitude / dynamic degree** (is anything actually moving) approximated by the mean optical-flow magnitude. The first rewards staying on-model; the second guards against the gaming trap from Section 5. Scored together, they catch the "smooth because static" failure that a single number hides.

```python
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

# --- subject consistency: mean cosine sim of per-frame feature embeddings ---
_weights = ResNet50_Weights.DEFAULT
_backbone = resnet50(weights=_weights).eval().cuda()
_backbone.fc = torch.nn.Identity()   # use the 2048-d penultimate features
_preprocess = _weights.transforms()

@torch.no_grad()
def subject_consistency(frames):
    # frames: list of HxWx3 uint8 numpy arrays
    feats = []
    for f in frames:
        x = _preprocess(torch.from_numpy(f).permute(2, 0, 1)).unsqueeze(0).cuda()
        feats.append(F.normalize(_backbone(x), dim=-1))
    feats = torch.cat(feats, dim=0)                     # [T, D]
    sims = feats[:-1] @ feats[1:].T                     # adjacent-frame sims
    return sims.diagonal().mean().item()

# --- dynamic degree: mean optical-flow magnitude (needs opencv) ---
import cv2
def dynamic_degree(frames):
    mags = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for f in frames[1:]:
        cur = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(np.linalg.norm(flow, axis=-1).mean())
        prev = cur
    return float(np.mean(mags))

def to_uint8(frames):
    # diffusers returns PIL or float; normalize to a list of HxWx3 uint8 arrays
    return [np.array(f).astype("uint8") for f in frames]

# --- run the whole board ---
board = []
for key in MODELS:
    frames, secs, peak_gb = generate(key)
    fr = to_uint8(frames)
    board.append({
        "model": key,
        "subject_consistency": round(subject_consistency(fr), 3),
        "dynamic_degree": round(dynamic_degree(fr), 2),
        "seconds_per_clip": round(secs, 1),
        "peak_vram_gb": round(peak_gb, 1),
    })

# sort by a composite that PENALIZES stillness: consistency only counts if it moves
for row in board:
    row["composite"] = round(
        row["subject_consistency"] * min(row["dynamic_degree"] / 3.0, 1.0), 3
    )
board.sort(key=lambda r: r["composite"], reverse=True)
for r in board:
    print(r)
```

Two things make this harness honest rather than another vibe check. First, the **composite score multiplies consistency by a clamped dynamic-degree term**, so a model that scores 0.99 consistency *by not moving* gets its score crushed by a near-zero motion factor — the gaming trap is closed by construction. Second, it **records seconds-per-clip and peak VRAM on your own GPU**, so the "price/latency" axis is measured, not quoted. Run this on an A100 80GB or an RTX 4090 24GB and you'll get numbers like the worked example below. The point is not that these two dimensions are sufficient — for a real eval you'd add background consistency, aesthetic quality, and text-alignment via a VLM judge, exactly as the [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) describes — but that *you, not a vendor, set the prompts and the seed*, which is the only way to get a leaderboard that predicts your job.

#### Worked example: a three-model open board on one GPU

Run the harness above on an RTX 4090 24GB with the fixed spec (49 frames, 480×720, seed 0, 50 steps) and you might see something like this. Wan 2.2: subject consistency 0.95, dynamic degree 2.8, ~180s per clip, ~22 GB peak. CogVideoX-5B: consistency 0.93, dynamic degree 2.1, ~150s, ~18 GB. LTX-2: consistency 0.90, dynamic degree 3.4, ~12s (its whole pitch is speed), ~16 GB. Naively, Wan "wins" on consistency. But the composite — consistency × clamped(dynamic/3) — gives Wan 0.95 × 0.93 ≈ 0.88, CogVideoX 0.93 × 0.70 ≈ 0.65, LTX 0.90 × 1.0 = 0.90. **LTX edges ahead on the composite** because it actually moves *and* renders fifteen times faster, which for a draft-iteration job is decisive. Change the spec to a static-subject portrait and the ranking flips toward Wan. That flip is the whole lesson: the "best" open model is a function of *your* prompt and *your* latency budget, which is exactly what a fixed-spec harness reveals and a vendor chart hides. (Treat the exact numbers as illustrative — re-run on your hardware to get yours.)

### Why the harness is the open track's structural advantage

Notice what the harness *required*: weights you can load, latents you can read, a seed you can fix, a scheduler you can swap, and VRAM you can profile. You cannot run any of this against a closed API — you get pixels back and nothing else. That asymmetry is the deepest reason the open track wins the *control* and *transparency* axes, and it compounds. Because you have the weights, you can do things the API will never let you: fine-tune a [video-LoRA](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) on a specific character or product so it stays on-model across shots; inject a ControlNet-style pose or depth conditioning to lock camera and motion; intervene at a specific denoising step to steer the trajectory; quantize the model to FP8 or 4-bit to fit a smaller GPU; cache features across steps to cut latency; and inspect every intermediate latent when something looks wrong. For a *research* or *heavily-customized production* workload, this isn't a nice-to-have — it's the entire reason to be on the open track even when the closed model is prettier. The closed leaders sell a finished output; the open models sell a machine you can open. When the job needs the machine, the polish gap is a price worth paying, and the harness is how you verify the open model clears your bar before you commit to building on it.

The flip side, in fairness: running that machine is *work*. You own the GPU provisioning, the CUDA and driver versions, the OOM debugging when the VAE decode blows past your VRAM at the last frame, the model-update treadmill, and the eval harness itself. The closed API is a button; the open model is a system. For a small team with no ML-ops capacity and a low-volume need, the closed API's "just call it" simplicity can be worth more than the control and the cost saving — which is exactly why the decision guide weighs *ops capacity* alongside the technical axes. Control is only an advantage if you have the hands to use it.

## 8. How to measure each axis honestly

The harness scored two axes; a production decision needs you to think about all nine from Section 2, and each has an honest and a dishonest way to measure it. Pulling the measurement methodology together in one place:

- **Resolution** — generate at the claimed native resolution and inspect high-frequency FFT energy; a true 4K model has real detail in the high bands, an upscaler has a filename. Don't trust the resolution parameter alone.
- **Native audio** — measure audio-visual sync offset with a lip-sync detector and check semantic grounding (does the sound match the event). A model that needs a separate TTS pass is *not* native-audio, regardless of marketing.
- **Coherent length** — sweep clip length and find where subject-consistency crosses a threshold; report *that*, not the API max. The usable length is almost always well below the renderable length.
- **Motion / physics** — score on a physics-probing prompt set with known correct outcomes (drop, drape, pour) using a rubric or learned scorer; cross-check VBench dynamic degree and motion smoothness *together* so stillness can't masquerade as quality.
- **Consistency** — VBench subject consistency within a shot; identity-embedding distance across shots for multi-shot. Always pair with dynamic degree.
- **Control** — count and test the actual knobs (first/last frame, motion strength, camera, reference, LoRA, ControlNet); for open models, verify the fine-tuning path actually works end to end, not just that it's "supported."
- **Price / latency** — measure seconds-per-clip with a GPU warm-up and report peak VRAM on a named GPU; for closed models, compute cost-per-second from the published price and a representative clip.
- **License** — read the actual license text for commercial use and self-hosting rights; "open" can mean Apache-2.0 or a non-commercial community license, and the difference can veto the model.

The meta-rule, which is the scientific spine of honest video evaluation: **fix the seed, fix the prompt set, warm up before timing, and never report a single number where a profile exists.** A model is a *vector* of axis scores. Anyone who hands you a scalar has thrown away the information you need to choose, and frequently the information that would have changed your choice.

## 9. The decision guide — which model for which job

Now the payoff: a decision guide that maps the *job* to the *model*, because — as everything above has argued — the right model is the one whose strong axis matches the job's binding constraint. Figure 7 is the compact version; the table after it adds the reasoning.

![A matrix mapping job scenarios like marketing clip, native audio, privacy, real-time draft, and character series to a recommended model, a reason, and whether it is API or open](/imgs/blogs/the-2026-video-model-landscape-7.png)

| Job scenario | Binding constraint | Reach for | Why | Track |
|---|---|---|---|---|
| Hero marketing clip, max quality | Polish, fidelity | **Veo 3.1 or Kling 3.0** | Top arena quality; Kling for multi-shot, Veo for audio+4K | Closed / API |
| Clip *with* synchronized voiceover/SFX | Native audio | **Veo 3.1 or Sora 2** | Joint AV generation with real sync; open models have none | Closed / API |
| Anything that can't leave your network | Privacy / on-prem | **Wan 2.2 or HunyuanVideo 1.5** | Self-hostable on 24–48GB; weights never leave your VPC | Open |
| Fast draft / iteration / previz | Latency, cost | **LTX-2** | Faster-than-real-time on your own GPU; iterate cheaply | Open |
| Character-consistent multi-shot series | Cross-shot identity | **Kling 3.0 or Veo 3.1** | Best multi-shot character consistency; reference-image control | Closed / API |
| High volume at low cost-per-second | Unit economics | **Open (Wan/LTX) self-hosted** | Amortized GPU cost beats per-second API at scale | Open |
| Heavy custom style / fine-tuned subject | Control surface | **Wan or CogVideoX + LoRA** | You own the weights; train a [video-LoRA](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) on your subject | Open |
| Image-to-video from a fixed first frame | First-frame control | **Wan/CogVideoX I2V or Runway** | I2V beats T2V when you can supply the opening frame | Either |
| Research / inspect intermediate latents | Transparency | **Any open model** | You can read every latent, swap schedulers, profile VRAM | Open |

The logic threading all of these: **identify the one constraint that, if violated, kills the job, and let that constraint pick the track first and the model second.** Privacy violated? Open track, full stop — the best closed model is unusable, so you compare *within* open. Need synchronized audio? Closed track, because no open model emits it natively yet. Cost-per-second is the binding constraint at high volume? Open and self-hosted, because amortized GPU cost beats a per-second meter past a crossover volume. Polish on a one-off hero shot with no privacy concern and a budget? Closed, because the frontier's data-and-compute advantage is real and shows up exactly on the high-fidelity hero shot.

#### Worked example: the healthcare clip from the intro

Back to the opening call: 12-second clip, voiceover, character consistent across three shots, must not leave the VPC. Walk the constraints in order of severity. The *killer* constraint is "must not leave the VPC" — that's a privacy/on-prem veto, which forces the **open track** regardless of how much better Veo's audio is. So Veo, Sora, and Kling are out, period. Within open, the next constraints are 12-second coherent length and three-shot character consistency. No open model gives you native audio, so the voiceover becomes a *separate* on-prem TTS + lip-sync pass — annoying but doable, and it keeps the bytes inside. For the visual: **Wan 2.2 or HunyuanVideo 1.5**, self-hosted, with a **video-LoRA fine-tuned on the character** to hold identity across the three shots (the control surface you only get with open weights), and generation tiled to reach 12 seconds with sliding-window conditioning to keep each shot coherent. The decision wasn't "which is the best model" — Veo is the best model — it was "which constraint binds," and the privacy veto made a mid-tier open model with a LoRA the correct answer over the world's best closed model. That inversion is the entire decision framework in one example.

## 10. Case studies: real numbers from shipped models

A few concrete, cited data points to ground the landscape — stated as approximate and dated, because that's the honest register for a fast-moving field.

**Sora's spacetime-patch scaling.** OpenAI's Sora technical report (Brooks et al., 2024) framed video as sequences of *spacetime patches* and showed sample quality improving substantially with training compute — the clearest public statement of the scaling thesis for video, and the reason the whole field treats compute as a first-class quality lever rather than a detail.

**CogVideoX, the open reference.** Zhipu's CogVideoX (2024) shipped an open 3D-VAE + expert-transformer recipe at 2B and 5B scales, generating roughly 6-second clips (49 frames at 8 fps) at 480×720, runnable on a single ~24GB GPU with offloading. Its value was less the absolute quality and more that it published a *complete, reproducible open recipe* the rest of the open track built on — its VAE and pipeline are the ones in the harness above.

**HunyuanVideo and Wan, the open ceiling.** Tencent's HunyuanVideo (13B, late 2024) and Alibaba's Wan (2.x, 2025) pushed open quality to within a polish-tier of the closed frontier on generic prompts, with Wan notably designed to run on consumer 24GB-class hardware — the release that made "self-hostable frontier-ish video" real for individuals, not just labs. Their reports confirm the converged recipe (causal 3D-VAE + DiT + flow matching).

**Movie Gen, the open blueprint for the closed recipe.** Meta's Movie Gen (2024) is the unusually transparent one: a 30B video model with a joint *Movie Gen Audio* model, personalization, and instruction-based editing, with the flow-matching training recipe documented in the paper even though the weights stayed closed. It's the best public window into how a frontier closed model is actually built — see the [Movie Gen deep dive](/blog/machine-learning/video-generation/movie-gen-deep-dive).

**Veo 3.1 and the audio frontier.** Google's Veo line reached native 4K with *synchronized* audio (Veo 3 onward), which is the capability the open track conspicuously lacks. It's the canonical example of a real moat that is *not* architectural — joint AV generation at that quality is a data-, compute-, and tuning-heavy problem, exactly the kind the closed track's resources buy.

**LTX-2 and the real-time frontier.** Lightricks' LTX line pushed an efficiency-first DiT with an aggressive high-compression VAE to *faster-than-real-time* generation on a single GPU — a capability with no self-hostable closed equivalent. It's the clearest case of the open track *leading* an axis (latency/cost) rather than chasing — detailed in the [LTX deep dive](/blog/machine-learning/video-generation/ltx-video-deep-dive).

The pattern across these: the closed models lead on the *compute-and-data-and-audio* axes, the open models lead on the *access-and-cost-and-latency* axes, and the converged architecture means neither lead is permanent on the buyable axes.

#### Worked example: pricing a closed-vs-open choice for a real workload

Make it concrete with numbers. A startup needs roughly 4,000 short clips per month for an automated social-content product — 6-second clips, 720p, no native audio required (they add captions and music in post). Closed option: at ~\$0.05/second, each 6-second clip is ~\$0.30, so 4,000 clips is ~\$1,200/month, scaling linearly forever and leaving them dependent on an API's rate limits and content filter. Open option: a single A100 80GB renting at ~\$2/hour, fully utilized, costs ~\$1,440/month — *more* than the API at this exact volume — but it renders a 6-second 720p Wan clip in ~60–90 seconds, so one GPU running 24/7 produces ~1,000–1,400 clips/day, far more than the 130/day they need. So they don't need the GPU full-time: ~130 clips/day at ~75 seconds each is ~2.7 GPU-hours/day, or ~\$160/month of compute. The open option is **roughly 7× cheaper** at this volume, *and* it removes the API content filter (which matters because their prompts occasionally trip false positives) *and* it lets them fine-tune a brand-style LoRA. The closed option only wins if they value the polish tier and audio more than the 7× cost and the control — which, for an automated high-volume product, they don't. The decision inverts entirely on volume and on whether audio is native-required; at 40 clips/month instead of 4,000, the API's trivial bill and zero ops burden would win easily. Same two models, opposite answers, set entirely by the workload's binding constraints. (Numbers illustrative; re-price at current rates.)

## 11. The trajectory and the open questions

Where is this going? A few directional calls, held loosely.

**Real-time generation is arriving on the open track first.** LTX-2's faster-than-real-time generation, step-distilled few-step video, and [feature caching](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) are pushing toward interactive, sub-second drafts. The closed leaders optimize for quality-per-clip server-side; the open track optimizes for latency-on-your-GPU, and that's where real-time, interactive, "video as a UI" experiences will appear first.

**Long-form coherence is the hardest remaining axis.** Holding identity, scene, and physical state across minutes — via [chunked diffusion, Diffusion Forcing, and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout) — is where error accumulation still bites and where the closed leaders' compute advantage shows most. Minute-plus coherent clips with stable characters remain a frontier capability, not a commodity.

**World models blur the line.** The [video-as-world-model line](/blog/machine-learning/video-generation/video-models-as-world-models) (Genie, action-conditioned next-frame prediction) is pulling video generation toward interactive, playable, agent-relevant simulation — which reframes "video model" as "learned simulator" and raises the [physics-versus-pattern-matching question](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation) the whole field is circling.

**Physics is the deepest open question.** Whether scale alone yields genuine physical understanding or just better-and-better pattern matching is unresolved, and it's the question that determines whether these models become reliable simulators or stay impressive-but-fragile illustrators. The honest answer in 2026 is "we don't know, and the benchmarks that would tell us are themselves immature."

**The cost curve keeps bending down.** Better VAEs, few-step samplers, caching, and quantization are dropping the cost-per-second of generation steadily, which is what's democratizing the open track. The economic question for any closed-API business is how long a per-second-meter survives against an open model that's one polish-tier behind and free to self-host.

Let me put a rough number on that crossover, because it's the calculation every team eventually runs. Say a closed API charges on the order of a few cents per second of generated video — call it \$0.05 per second for round numbers, so a 10-second clip is \$0.50. On the open side, an A100 80GB rents at roughly \$1.50–\$2.00 per hour, and a self-hosted open model might render that same 10-second clip in, say, 90 seconds of wall-clock at 720p. That's 90 seconds of a \$2/hour GPU, or about \$0.05 of compute — *ten times cheaper per clip* than the API, before you amortize the engineering cost of running the pipeline. The crossover where self-hosting wins is therefore *volume*: if you generate a handful of clips a week, the API's \$0.50 each is trivial and not worth the ops burden of a GPU; if you generate thousands a day, the 10× per-clip saving dominates and self-hosting is obviously correct. The break-even is roughly "when your monthly API bill exceeds the fully-loaded cost of a GPU plus the engineer-time to run it" — which, depending on your team, lands somewhere in the low-thousands-of-clips-per-month range. This is the quantitative version of the "cost-at-volume" decision rule, and it's why the open track's economic gravity grows as your usage grows. (As always, plug in current prices — these move, and the API providers cut rates specifically to push the crossover higher.)

**Personalization and editing become table stakes.** Movie Gen shipped instruction-based editing and personalization; the open track is catching up via LoRA and reference-image conditioning. The next frontier is *consistent editing* — changing one element of a generated clip without re-rolling the whole thing — which matters enormously for production workflows where you generate, get notes, and revise. The model that makes "change the jacket to red, keep everything else" a one-shot edit rather than a re-generation will own the professional creative market, and it's an open race.

## 12. When to reach for closed, and when to reach for open

A decisive recommendation, because that's what a buyer's guide owes you.

**Reach for the closed track when** the job's binding constraint is on an axis the closed leaders own: native synchronized audio, absolute top-tier polish on a hero shot, multi-shot character consistency across a series, 4K native output, or long coherent clips — *and* the bytes are allowed to leave your network, *and* the volume is low enough that per-second pricing doesn't dominate. A one-off marketing hero clip with a voiceover is the canonical closed-track job.

**Reach for the open track when** the binding constraint is privacy/on-prem (the bytes can't leave), cost-at-volume (you're generating enough that amortized GPU beats a per-second meter), control (you need LoRA fine-tuning, ControlNet, or to inspect intermediate latents), latency-on-your-hardware (real-time drafts), or license (you need commercial rights and self-hosting). A high-volume, privacy-constrained, or heavily-customized pipeline is the canonical open-track job.

**Don't** pick a model off a single arena rank — the rank is a blended preference number that may weight axes your job doesn't care about. **Don't** assume "open is worse" as a blanket — on access, cost, control, and latency, open *leads*, and on generic short-clip quality the gap is a polish tier, not a chasm. **Don't** pay for the closed frontier when an open model clears your quality bar and your constraint is cost or privacy — that's overpaying for axes you're not using. And **don't** reach for an open model when you need native audio today — it isn't there yet, and a TTS-plus-lip-sync bolt-on is a real quality and engineering tax.

A few more directional rules that fall out of the framework. **Don't conflate "best demo" with "best for me"** — the demos you see are cherry-picked from many rolls on prompts the lab knows the model handles well; your prompts are not those prompts, which is the whole reason for the fixed-spec harness. **Don't lock into a closed API without an exit** — version churn cuts both ways, and a model you build a product on can degrade, get deprecated, or change its content policy under you; keep an open fallback wired up so a vendor change doesn't strand you. **Don't ignore the license on an open model you ship commercially** — a "community" or non-commercial license can be a legal landmine that no amount of quality redeems. And **don't over-index on length** — most jobs need a few coherent seconds, not a coherent minute, and the models that chase maximum length often trade away the short-clip polish that your actual deliverable needs. The recurring discipline is the same every time: the job's binding constraint, not the leaderboard's headline, decides.

One last framing that I find clarifies the whole landscape: think of the closed leaders as *renting you the output of a machine you can't afford to build*, and the open models as *selling you the machine*. If you need a few beautiful outputs and the output is the product, rent — the closed frontier's data-and-compute advantage is exactly what you're paying for, and it's real. If the *machine* is the product — because you generate at volume, customize heavily, keep data in-house, or need to inspect and modify the pipeline — buy, and accept the polish gap as the cost of ownership. Almost every confused model-selection conversation I've sat in resolves the moment someone asks "are we buying outputs or buying a machine?" — and that question, not any leaderboard, is the one to lead with.

The decision is always: *name the binding constraint, let it pick the track, then compare within the track on the job's weighted axes using your own fixed-prompt harness.* That procedure is robust to the version churn that will date every specific number in this post.

## 13. Stress-testing the framework

Where does this framework strain? Four cases.

**When the constraint set is empty (a generic short clip, no privacy, modest budget).** Then the framework under-determines the choice, and that's fine — pick the top-tier model on whichever track is convenient, because the axes that differentiate them don't bind your job. The framework is most useful precisely when a constraint *does* bind; for a constraint-free job, the leaderboards are a fine tiebreaker.

**When two constraints conflict — privacy *and* native audio.** This is the healthcare case, and it has no clean answer: privacy forces open, audio forces closed. You resolve it by *relaxing the softer constraint* — here, audio becomes a separate on-prem pass rather than a native capability, paying an engineering tax to honor the hard privacy veto. The framework's job is to surface the conflict explicitly so you make the trade consciously instead of discovering it after you've shipped to a closed API.

**When the arena and your harness disagree.** Trust your harness. The arena measures generic human preference on generic prompts; your harness measures your prompts on your axes. If the arena's top model loses on your fixed-spec board, the arena is telling you about a job that isn't yours.

**When the model you chose ships a new version mid-project.** Re-run the harness, don't re-read the marketing. A version bump changes the *numbers* on the axes, not the *axes* — so the framework holds and only the measured vector updates. This is the whole reason the framework is built on operationally-defined axes rather than memorized rankings: it survives the churn.

## 14. Key takeaways

- **The field is two tracks**: closed API leaders selling quality, audio, and consistency; open self-hostable models selling access, cost, and control. Pick the track before the model.
- **Quality is a vector, not a scalar.** Compare on operationally-defined axes (resolution, native audio, coherent length, motion/physics, consistency, control, price/latency, license) — the right model matches its strong axis to your binding constraint.
- **Leaderboards measure narrow things.** Arena Elo is human preference but not reproducible and blended; VBench is reproducible and per-axis but gameable — read consistency/smoothness *as a ratio against dynamic degree*, never alone.
- **The architecture has converged** — causal 3D-VAE + (MM-)DiT on spacetime tokens + flow matching + scale + preference tuning — so differentiation comes from **data, compute, audio, and tuning**, not a secret design.
- **The open-closed gap is a funding gap, not an architectural one**, which is why it narrowed from "a year behind" to "a polish tier behind" between 2024 and 2026, and keeps narrowing on the buyable axes.
- **Build your own leaderboard.** Run the same prompt and seed through open models in `diffusers`, score on VBench-style axes with a stillness-penalizing composite, and measure seconds-per-clip and peak VRAM on your own GPU.
- **Name the binding constraint, let it pick the track, then compare within the track** on your job's weighted axes — that procedure survives the monthly version churn that dates every specific number.
- **Treat all standings and figures as approximate and dated.** The durable artifact is the framework, not the ranking.

## 15. Further reading

- Brooks, Peebles, et al., *Video generation models as world simulators* (Sora technical report), OpenAI, 2024 — the spacetime-patch scaling thesis.
- Polyak, Zohar, et al., *Movie Gen: A Cast of Media Foundation Models*, Meta, 2024 — the most transparent public frontier recipe, with joint audio and flow matching.
- Yang, Teng, et al., *CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer*, Zhipu AI, 2024 — the open reference recipe.
- *HunyuanVideo* (Tencent, 2024) and *Wan* (Alibaba, 2025) technical reports — the open ceiling and the consumer-hardware milestone.
- Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models*, 2023, and *VBench-2.0* — the per-axis reproducible benchmark and its caveats.
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT), 2023, and Lipman et al., *Flow Matching for Generative Modeling*, 2023 — the architecture and objective the field converged on. For the underlying image-diffusion mechanics, the image series covers [diffusion transformers](/blog/machine-learning/image-generation/diffusion-transformers-dit), [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), and [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly).
- Within this series: the foundations post [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the [metrics deep dive](/blog/machine-learning/video-generation/the-metrics-of-video-generation), the [open frontier synthesis](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
- The model reports: [Kling](/blog/machine-learning/video-generation/kling-deep-dive), [Seedance and the ByteDance stack](/blog/machine-learning/video-generation/seedance-and-the-bytedance-video-stack), [Movie Gen](/blog/machine-learning/video-generation/movie-gen-deep-dive), [Mochi and asymmetric DiT](/blog/machine-learning/video-generation/mochi-and-asymmetric-dit-video), and [LTX-Video](/blog/machine-learning/video-generation/ltx-video-deep-dive).
