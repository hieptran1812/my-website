---
title: "Why Video Generation Is Hard: Coherence, Motion, and the Compute Wall"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A first-principles map of video generation: temporal coherence, plausible motion and implicit physics, the quadratic-in-frames compute wall, the lineage from GAN video to spacetime diffusion transformers, and a runnable hello-world that makes a clip in fifteen lines."
tags:
  [
    "video-generation",
    "diffusion-models",
    "text-to-video",
    "video-diffusion",
    "temporal-coherence",
    "spacetime-transformer",
    "generative-ai",
    "deep-learning",
    "sora",
    "stable-video-diffusion",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/why-video-generation-is-hard-1.png"
---

Render a five-second clip of a dog running across a lawn and one of three things tends to happen. The first: the dog is gorgeous frame by frame, but its face subtly re-rolls every few frames, the lawn shimmers, and a tree in the background pops in and out of existence. That is a coherence failure. The second: every frame is rock solid and identical, the dog stands perfectly still, the camera barely drifts, and you have generated a high-quality slideshow rather than a video. That is a motion failure. The third, and the one that ends most weekend experiments: the render reaches second four, the VAE starts decoding the final frames, the GPU's memory meter slams into the ceiling, and the process dies with `CUDA out of memory` at second six of an eight-second target. That is the compute wall.

These three failure modes are not bugs you fix one at a time. They are three faces of the same underlying difficulty, and they trade against each other so directly that improving one usually degrades another. This post is the map of that difficulty. It is the introduction to a series that takes you from "I have seen Sora and Veo clips" to "I understand the spatiotemporal modeling, can read the papers, and can build, fine-tune, and serve a video model myself." By the end of *this* post you will be able to: name the four model families and what each sacrifices; explain why video is fundamentally harder than image generation rather than just bigger; do the back-of-envelope compute math that tells you whether a clip will fit on your GPU; and run a fifteen-line script that makes an actual video clip on your own machine.

![A comparison matrix scoring per-frame GAN, frame interpolation, image-inflated diffusion, latent video diffusion, and spacetime DiT across coherence, motion, length, and cost](/imgs/blogs/why-video-generation-is-hard-1.png)

Here is the thesis of the whole field in one sentence: **a video is a sequence of correlated images, so generating one means solving image generation and then, on top of that, enforcing temporal coherence and plausible motion across the sequence, all under a compute budget that scales with the number of frames times the resolution.** That extra axis — time — is the entire story. It is what makes video harder, more expensive, and more interesting than the image problem this series' companion already covered. If you have read the image-generation series, you already understand diffusion, latent compression, the diffusion transformer, and flow matching; this series spends its depth on the part those tools do not cover, which is time. If you have not, the foundation you need is in [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) and [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), and you can read this post first as orientation and follow those links when a pure image-diffusion mechanism comes up.

## A video is a correlated sequence, and that changes everything

Start with what a video literally is. A clip is an ordered list of frames, each frame an image. A five-second clip at 24 frames per second is 120 images. A modest 720p frame is $3 \times 720 \times 1280 = 2{,}764{,}800$ numbers, so the raw clip is about 332 million numbers — a point in a space of roughly $3.3 \times 10^8$ dimensions. The image problem already lives in a space of millions of dimensions and is, by the manifold argument from the image series, one of the hardest sampling problems in machine learning. Video multiplies that by the frame count.

But the multiplication is not the interesting part. If frames were *independent*, video generation would just be image generation run 120 times, which is expensive but otherwise straightforward. The reason video is its own problem is that consecutive frames are *extremely* correlated. Look at any two adjacent frames of real footage: 95% or more of the pixels barely change. The dog's body shifts a few pixels to the left; the lawn's blades sway; the lighting holds. This correlation is simultaneously the curse and the cure.

It is the cure because it is a powerful prior. The set of valid videos is a vanishingly thin manifold inside that 330-million-dimensional space — far thinner, *proportionally*, than the image manifold inside pixel space, because on top of every per-frame constraint (faces have two eyes, the sky is up) there is now a stack of *cross-frame* constraints: a dog that exists in frame 40 must still exist, recognizably the same dog, in frame 41; a wall that is brick in frame 1 cannot be concrete in frame 50; an object that leaves the frame and comes back must come back the same. Temporal redundancy is what makes video learnable at all. A model that has learned the prior can predict most of the next frame from the current one, and only needs to generate the *changes*.

It is the curse because enforcing those cross-frame constraints is exactly the hard part, and small per-frame errors compound along the sequence. If your model is 99% accurate at keeping a face stable from one frame to the next, then over 120 frames the probability that the face survives unchanged is $0.99^{120} \approx 0.30$ — a 70% chance the identity drifts noticeably by the end of a single short clip. Coherence is not a property you get for free by making each frame good; it is a constraint you must actively model, and it gets harder the longer the clip.

### The bitrate argument: why redundancy is the whole game

There is a precise, information-theoretic way to see why temporal redundancy is both what makes video tractable and what shapes every design choice, and it is worth a paragraph of rigor because it underlies the next post in the series on representation. Consider how much *new information* each frame actually carries. The first frame of a clip carries the full content of an image — call it $H_0$ bits, the entropy of a single frame. But each subsequent frame, given the one before it, carries only the *conditional* entropy $H(x_{t} \mid x_{t-1})$, which is far smaller because most of the next frame is predictable from the current one. This is exactly what every video codec — H.264, H.265, AV1 — exploits: it sends one full "keyframe" and then, for every following frame, sends only the *residual* after motion-compensated prediction. A raw 5-second 720p clip is hundreds of megabytes; the same clip as an MP4 is a few megabytes. That 50-to-100× compression ratio is a direct measurement of how redundant video is.

The generative consequence is sharp. The total information a model must produce for a $T$-frame clip is approximately

$$
H_\text{clip} \approx H_0 + (T-1) \cdot H(x_t \mid x_{t-1}),
$$

and because the conditional term is small, the *marginal* cost of an extra frame in information is low even though its *compute* cost is not. This is the central tension of the whole field stated in one line: **video is information-cheap per frame but compute-expensive per frame.** The redundancy is what lets a model learn video from a finite dataset at all — there is far less genuinely new structure than the raw dimensionality suggests — and it is also why the right move is to compress aggressively before generating. If most of a frame is predictable, do not spend transformer FLOPs re-deriving it; compress it away in the VAE and let the denoiser model only what is left. That insight is the seed of the entire stack, and we develop the representation side of it in [representing video, redundancy, and tokens](/blog/machine-learning/video-generation/representing-video-redundancy-and-tokens).

There is a corollary that bites in practice. Because the model only needs to produce the *changes* between frames, the quality of a clip is dominated by how well the model handles *motion boundaries* — the regions where the redundancy assumption breaks, where an object's edge moves, where something is revealed or occluded. The flat, static parts of a frame are nearly free; the hard, expensive, error-prone parts are exactly the moving edges. This is why generated video so often looks great in still backgrounds and subtly wrong at the boundaries of moving objects — the model is spending its limited capacity exactly where the redundancy prior gives it the least help.

![A two-column comparison contrasting image generation as one frame with spatial coherence against video generation as many frames with space-and-time coherence and out-of-memory risk](/imgs/blogs/why-video-generation-is-hard-2.png)

So the right way to hold video generation in your head is as a product of two requirements operating under a constraint. You need **spatial generation** (each frame must be a plausible image, the thing the image series is about) times **temporal coherence** (the frames must be consistent with each other), all under a **brutal compute budget**. That framing — image generation times time, under cost — is the spine of this entire series, and every later post is a deep dive into one part of it.

### The four-way tension: coherence, motion, length, cost

Within that frame, four properties pull against each other, and figure 1 above scores the model families on exactly these axes. Internalize them now, because they are the recurring vocabulary of the whole series.

- **Coherence** is whether the video holds together: no flicker, stable object identity, a background that does not boil, a character whose face stays the same. Coherence is *consistency over time*.
- **Motion** is whether things actually move in a plausible way: the dog's legs cycle, momentum carries through a jump, water falls, cloth settles. Crucially, motion drags in *implicit physics* — gravity, momentum, object permanence, fluid behavior. A model that "moves" by teleporting objects or melting them between frames has motion without plausibility.
- **Length** is how many seconds you can generate before the clip falls apart or the memory runs out. Two seconds is easy; ten is hard; a minute is the current frontier; coherent multi-minute video is open.
- **Cost** is the compute to produce a clip: the FLOPs, the seconds-per-clip, and above all the *peak VRAM*, which is the wall most people actually hit.

The tension is that you can almost always buy one by spending another. Want more motion? You raise the dynamic range of what the model is allowed to change between frames, and coherence gets harder because there is more to keep consistent. Want longer clips? Cost rises with frame count and coherence degrades because errors have more steps to compound. Want lower cost? You compress harder or sample fewer steps, and you pay in either coherence or motion. **No current model wins all four.** Sora-class spacetime transformers get coherence, motion, and length but cost is enormous; per-frame GANs are cheap but flicker; frame interpolation is smooth but invents no new motion. The art of the field — and of every model report you will read — is choosing which corner to sacrifice for a given use case.

## Why video is genuinely harder than image generation

It is tempting to say "video is just image generation with more frames, so it is the same problem but bigger." That is wrong in three specific, important ways, and getting them straight is the point of this section.

**First, the extra axis is expensive *and* correlated, which is the worst combination.** If the time axis were cheap, the cost would not matter. If it were uncorrelated, you could model frames independently. It is neither. It is expensive because every frame adds tokens the model must attend over, and it is correlated because the model must attend *across* frames to enforce coherence — so you cannot decompose the problem to make it cheap. You are forced to model long-range dependencies over a large tensor, which is exactly where compute explodes.

**Second, small per-frame errors compound.** In a single image, a small error is a slightly imperfect image and that is the end of it. In a video, an error in frame $t$ becomes the *context* for frame $t+1$, which builds on it, and the error propagates and amplifies. This is the same pathology that makes autoregressive language generation drift, and it is why long-video generation is genuinely an open research problem rather than a solved one with a bigger GPU. We will quantify this drift later in the post and devote a whole later post to it.

**Third, the autoencoder has to compress time too.** Modern image generation does not denoise in pixel space; it works in the latent space of a VAE that compresses a $1024 \times 1024$ image to, say, a $128 \times 128 \times 4$ latent — an order of magnitude fewer numbers, which is why latent diffusion is tractable (see [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion)). Video does the same trick but the autoencoder must now compress the *time* axis as well, jointly with space. That is a strictly harder learning problem — the VAE must encode motion, not just appearance, into a compact code and decode it back without introducing flicker — and as we will see, this **video VAE is the real bottleneck and the real lever** of the whole stack. It deserves and gets its own post: [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression).

There is a fourth difficulty that is easy to miss and that bites in production: **the data is harder, and the captions are worse.** Image generation rode on billions of image-text pairs scraped from the web with usable alt-text captions. Video has no comparable corpus: high-quality, motion-rich, well-captioned clips are far scarcer, web video is dominated by talking heads and slideshows with little useful motion, and writing a caption that describes *what happens over time* (not just what is in the frame) is a much harder labeling problem. This is why the SVD report spends so much of its length on data curation and why every serious model invests in re-captioning pipelines that describe motion explicitly. The scarcity also skews what models are good at: there is abundant footage of common scenes and a long tail of rare motions the model has barely seen, which is part of why rare or precise motion is exactly where generated video looks wrong. Data, not just compute, is a first-class constraint here.

Put those together and the slogan "video is image plus time" is exact, but "time" is doing enormous work in that sentence. It is the most expensive axis you can add, it is correlated so you cannot factor it away, it forces every component of the stack — the autoencoder, the denoiser, the sampler — to grow a time dimension, and the data needed to learn it well is scarcer and harder to label than image data ever was.

### The compute wall, made quantitative

Let me make the cost argument rigorous, because it is the one thing about video that you can actually compute on a napkin, and it explains nearly every engineering decision in the field.

Start with token count. A diffusion transformer does not see pixels; it sees a grid of latent tokens. For an image, the number of tokens is roughly

$$
N_\text{img} \approx \frac{H \cdot W}{(f_s \cdot p)^2}
$$

where $H, W$ are pixel height and width, $f_s$ is the VAE's spatial downsampling factor, and $p$ is the patch size the transformer applies on top. For video you add a time dimension with its own temporal downsampling factor $f_t$:

$$
N_\text{vid} \approx \frac{T}{f_t} \cdot \frac{H \cdot W}{(f_s \cdot p)^2}
$$

So tokens grow *linearly* in the number of frames $T$. That alone is not catastrophic. The catastrophe is the attention. A transformer's self-attention cost is quadratic in the number of tokens it attends over:

$$
\text{FLOPs}_\text{attn} \propto N^2 \cdot d
$$

where $d$ is the model dimension. If full attention runs over all spacetime tokens at once — the most expressive option, the one that models coherence best — then because $N_\text{vid} \propto T$, the attention cost goes as

$$
\text{FLOPs}_\text{attn} \propto N_\text{vid}^2 \propto T^2.
$$

**The compute is quadratic in the number of frames.** Double the clip length and full-attention compute quadruples. This single fact is why naive "just run the image transformer on a stack of frames" does not scale, why the field invented factorized (spatial-then-temporal) attention to claw the cost back down, and why length is the hardest of the four properties to buy. We will spend a whole later post, [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), on the architectures that fight this curve.

![A flow graph showing pixel video compressed by a 3D-VAE into spacetime tokens that feed either quadratic full attention or cheaper factorized attention, both leading to the FLOPs-per-clip wall](/imgs/blogs/why-video-generation-is-hard-6.png)

There is a direct lever against the quadratic curve, and it is the single most important architectural choice in the denoiser: **factorized attention**. Instead of one full self-attention over all $N$ spacetime tokens, you alternate a *spatial* attention that attends only within each frame (over the $S = N/T'$ tokens of that frame, where $T' = T/f_t$ is the number of latent frames) and a *temporal* attention that attends only across frames at each spatial position (over the $T'$ tokens at that location). The cost of full attention is $\propto N^2 = (T' S)^2 = T'^2 S^2$. The cost of factorized attention is the spatial part, $T' \cdot S^2$ (one $S \times S$ attention per frame), plus the temporal part, $S \cdot T'^2$ (one $T' \times T'$ attention per spatial position), for a total $\propto T' S^2 + S T'^2 = T' S (S + T')$. The ratio of full to factorized cost is

$$
\frac{\text{full}}{\text{factorized}} = \frac{T'^2 S^2}{T' S (S + T')} = \frac{T' S}{S + T'}.
$$

For a clip where the spatial token count $S$ is large (thousands) and the latent frame count $T'$ is moderate (tens), this ratio is close to $\min(T', S)$, which is to say factorized attention is *cheaper by roughly a factor of the smaller of the two axes* — often a 10–30× FLOP reduction at typical video shapes. The price is expressiveness: factorized attention cannot directly relate a token in frame 3, position A to a token in frame 7, position B in a single layer; it has to route that interaction through intermediate layers. That is the compute-versus-coherence trade made concrete, and it is why almost every shipped model factorizes — the coherence loss is usually small and the FLOP savings are enormous. We dissect exactly where the FLOPs go in [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns).

The other half of the wall is memory, and memory bites differently from FLOPs. The peak VRAM during generation is dominated by activations — the intermediate tensors held in flight during the forward pass — and during *decoding*, by the VAE reconstructing full-resolution frames. Here is the cruel irony that surprises people the first time: it is often the **VAE decode, not the denoiser, that OOMs**. The denoiser runs in the small latent space; the VAE has to expand that latent back to full-resolution pixels for every frame at once, and that final tensor — `num_frames × 3 × H × W` in float16 — can be larger than everything else combined. This is why `diffusers` video pipelines expose `decode_chunk_size`: it decodes frames in small batches instead of all at once, trading speed for a lower memory peak. Knowing that the decode is the wall is the difference between a clip that renders and one that dies at second six.

### The compression ratio is the master lever

Step back and notice that the VAE's compression ratio appears in *every* cost above. The token count $N \propto T H W / (f_t f_s^2 p^2)$ has the VAE's temporal factor $f_t$ and spatial factor $f_s$ in the denominator; the attention cost goes as $N^2$, so it scales as $1/(f_t f_s^2)^2$; the decode memory and the latent footprint both shrink with compression. A causal 3D-VAE with $f_t = 4, f_s = 8$ reduces the *element count* of the input by roughly $4 \times 8 \times 8 = 256\times$ (before channel changes), and because attention is quadratic, a 2× improvement in compression is close to a 4× reduction in denoiser attention cost. This is why I keep insisting the VAE is the real lever and not a preprocessing detail: it sits upstream of the most expensive operation in the stack, with a quadratic multiplier on its effect. Push the compression too far, though, and the decoder cannot reconstruct fine detail or fast motion, and you get blur and flicker — so the VAE is a tense trade between cost (more compression) and quality (less). The whole next-but-one post is about how to win that trade: [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression).

#### Worked example: will a 5-second 720p clip fit?

Let's do the napkin math for a concrete target: a 5-second clip at 24 fps and 720p, which is $T = 120$ frames at $720 \times 1280$.

Take a typical modern causal 3D-VAE with spatial factor $f_s = 8$ and temporal factor $f_t = 4$, and a transformer patch size $p = 2$. The token count is

$$
N \approx \frac{120}{4} \cdot \frac{720 \cdot 1280}{(8 \cdot 2)^2} = 30 \cdot \frac{921{,}600}{256} = 30 \cdot 3{,}600 = 108{,}000 \text{ tokens}.
$$

That is a *lot*: a large language model's context window for comparison is often 8K to 128K, and here a single 5-second clip is already 108K tokens — and full self-attention over 108K tokens means a $108{,}000^2 \approx 1.17 \times 10^{10}$-entry attention matrix *per head per layer*. In float16 that one matrix is over 23 GB, which already does not fit on a 24 GB card, and that is before you count the model weights or any other activations. This is the concrete reason full 3D attention is not used at this scale and why every shipped model factorizes the attention or restricts it to windows. The napkin tells you the architecture choice before you write a line of code.

Now flip the same math into the question people actually ask: *what can I run on a 24 GB RTX 4090?* You drop to a shorter clip (say 49 frames, the CogVideoX default) at a lower resolution, you enable VAE tiling and model CPU offload, you use `decode_chunk_size` to chunk the decode, and you accept ~5 minutes per clip. That is a real, runnable configuration, and the rest of this series is largely about making that configuration faster and longer without falling off the coherence cliff.

## Coherence is a constraint you have to model, not a side effect

The compute wall is the cost half of the story. The coherence half is subtler and, in a sense, more fundamental, because it is *why* the expensive cross-frame attention exists in the first place. Let me make the coherence argument as rigorous as the compute one.

Define coherence operationally. A clip is temporally coherent if, for properties that should be stable (object identity, background, lighting, texture), the per-frame variation is small, and for properties that should change (position of moving objects), the change is smooth and physically plausible. Flicker is high-frequency variation in properties that should be stable. Identity drift is low-frequency variation in those same properties. Both are coherence failures; they just live at different time scales.

Why does a naive per-frame model fail at this? Suppose you take a strong text-to-image model and generate 120 frames independently from the same prompt with 120 different noise seeds. Each frame is a great image. But the model has no mechanism to make frame 41 *the same scene* as frame 40 — it samples 120 unrelated points from the "dog on a lawn" manifold, and you get 120 different dogs on 120 different lawns. The video is pure flicker. Coherence requires the generation of frame $t+1$ to be *conditioned on* frame $t$ (or on a shared latent that ties all frames together). There is no way around it: coherence is a dependency you must build into the model.

The cleanest way to see the compounding problem is to model identity as a per-frame survival process. Let $q$ be the probability that the model preserves a given identity feature (the dog's face) unchanged from frame $t$ to frame $t+1$, given it was correct at $t$. If errors are independent across steps, the probability the feature survives all $T-1$ transitions is $q^{T-1}$. This decays geometrically in $T$.

#### Worked example: how fast does identity drift?

Suppose a model has a strong but imperfect $q = 0.995$ per-frame identity-preservation probability — a 0.5% chance of a noticeable drift each frame, which sounds excellent. Over a 2-second clip ($T = 48$ frames) the survival probability is $0.995^{47} \approx 0.79$, so about a 21% chance of visible drift — annoying but often acceptable. Over a 10-second clip ($T = 240$) it is $0.995^{239} \approx 0.30$: a 70% chance the character's face has visibly drifted by the end. Over a 60-second clip ($T = 1440$) it is $0.995^{1439} \approx 0.0007$: essentially certain to drift. This is why "the model is great for 4-second clips but the character is a different person by second 12" is the universal experience, and why coherent long video is a separate research problem and not a matter of raising `num_frames`. The geometry of compounding is brutal: even a near-perfect per-step model falls apart over enough steps. The fix is not a better per-step $q$ alone — it is architectural mechanisms (overlapping windows, keyframe anchoring, explicit memory) that break the independence assumption, which we cover in the long-video post.

The survival-probability model captures *single-shot* drift, but there is a worse version that appears when you generate long video by *rolling out* — generating a chunk, then conditioning the next chunk on the last frames of the previous one, and so on. Now errors do not just accumulate independently; they feed back. Model the deviation of the generated state from the true manifold at step $t$ as $\epsilon_t$. Each rollout step does two things: it adds a fresh per-step error $\delta$, and it *amplifies* whatever error was already present by some factor $\gamma$ (because the model conditions on its own imperfect output, and a transformer's response to off-distribution input is itself off-distribution). To first order,

$$
\epsilon_{t+1} \approx \gamma \, \epsilon_t + \delta.
$$

If $\gamma < 1$ the error settles to a steady state $\epsilon_\infty = \delta / (1 - \gamma)$ and the video stays usable indefinitely — this is the regime every long-video method is trying to reach. If $\gamma > 1$ the error grows *geometrically*, $\epsilon_t \sim \gamma^t$, and the clip diverges into color smears or a frozen frame within seconds, which is exactly what naive autoregressive rollout does. The entire long-video research program — Diffusion Forcing, CausVid, sliding-window and overlapped-chunk diffusion — is, in this language, a set of tricks to push $\gamma$ below 1: anchoring to keyframes, re-injecting the conditioning image, noise-augmenting the context so the model is trained to tolerate its own errors. We give that program a full post, [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout). For now, internalize the dichotomy: long video is stable if and only if per-step error amplification stays below one, and getting there is hard precisely because conditioning on your own output is inherently destabilizing.

There is a second, sneakier coherence trap that the field calls the **dynamic-degree versus stability tension**, and it is the single most important honesty point in video evaluation. The cheapest way to make a clip "coherent" is to make nothing move — a static clip has perfect subject and background consistency by construction. So a model can score beautifully on consistency metrics by being boring. Conversely, a model with lots of motion has more opportunities to break consistency. This means any consistency metric is gameable by reducing motion, and any motion metric is gameable by adding chaotic jitter. Good evaluation has to measure both and look at their *joint* behavior, which is precisely what VBench's dynamic-degree dimension is designed to expose. We devote a whole post to this — [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation) — and you should be suspicious of any "our model is more consistent" claim that does not report dynamic degree alongside it.

### Motion drags in physics, whether you ask for it or not

The flip side of coherence is motion, and motion is where video quietly becomes a physics problem. To move a dog's legs convincingly the model must implicitly know how legs cycle. To drop a ball it must implicitly produce acceleration under gravity. To pour water it must produce fluid that conserves roughly the right volume and falls the right way. None of this is given to the model as equations; it is learned, statistically, from footage. The result is models that get *common* motion impressively right and *rare or precise* motion subtly wrong — a glass that refills itself when occluded, a person who gains a finger mid-gesture, a flag that ripples in a way no wind produces.

This is the heart of the "world simulator" debate that Sora kicked off and that we examine in the frontier track. The optimistic reading is that a model trained on enough video learns an implicit, approximate physics — object permanence, momentum, collision — as a byproduct of predicting frames, and that scaling this up moves toward a learned simulator of the visual world. The skeptical reading is that it learns the *statistics* of how things usually look while moving, which is not the same as a causal model of why, and that the failures (objects appearing and disappearing, broken conservation laws) reveal the difference. Both readings have evidence. For this introduction the point is narrower and certain: **plausible motion implies implicit physics, the physics is learned and approximate, and its failure modes are a recurring source of the "uncanny" quality in generated video.** We will not resolve the simulator debate here; we will simply note that motion is never just motion.

## The lineage: how the field got here

The four-way tension explains the *shape* of progress in video generation: each era of models picked a different corner to sacrifice, and the frontier moved as researchers found ways to relax the trade-offs. Walking the lineage is the fastest way to understand why today's models look the way they do.

![A tree of video generator families branching into per-frame GAN, frame interpolation, and video diffusion, with the diffusion branch leading to inflated, latent, and spacetime DiT recipes](/imgs/blogs/why-video-generation-is-hard-3.png)

**Per-frame GANs and the early days (≈2016–2020).** The first generation of video models extended GANs to video — MoCoGAN, TGAN, and relatives — typically by decomposing a clip into a content code and a per-frame motion code, or by adding a temporal discriminator. They could produce short, low-resolution clips of constrained domains (faces, simple scenes). They flickered, they were hard to train (GANs already are; see [generative adversarial networks and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost)), and they did not scale to open-domain text-to-video. They picked *cheap and short* and sacrificed coherence and generality. Their lasting contribution was the content/motion decomposition idea, which echoes in how modern models separate a shared scene from per-frame dynamics.

**Frame interpolation (a parallel, still-useful branch).** A different and very practical line never tried to *generate* video from scratch — it took real keyframes and synthesized the in-between frames. RIFE, FILM, and the broader video-frame-interpolation literature are excellent at producing smooth slow-motion from sparse frames, and they are coherent by construction because they are anchored to real images. But they invent no *new* content or motion beyond interpolating what is given. They picked *smooth and cheap* and sacrificed the ability to generate. This branch matters today because interpolation is still used as a *post-process*: many pipelines generate at low fps and interpolate up, which is cheaper than generating every frame. Keep it in your toolbox; it is the cheapest fps you will ever buy.

**Image diffusion, inflated to video (2022).** The breakthrough idea was to reuse the enormous progress in image diffusion. Take a pretrained text-to-image diffusion U-Net and "inflate" it to video by inserting temporal layers — temporal convolutions and temporal attention — between the existing spatial layers, so the model can look across frames. Video Diffusion Models (Ho et al., 2022) introduced a 3D U-Net factorization; Make-A-Video (Singer et al., 2022) inflated a text-to-image model and learned motion from unlabeled video without paired text-video data. This era established the template the field still uses: *do not learn video from scratch; bootstrap from a strong image model and teach it time.* These models picked up real coherence and motion but were short (often 16 frames) and low-resolution, because they ran in or near pixel space and the compute wall hit fast.

**Latent video diffusion (2023).** The next leap was to move the whole process into a compressed latent space, exactly as latent diffusion did for images, but now compressing time too. Stable Video Diffusion (Blattmann et al., 2023) is the canonical open example: an image-to-video latent diffusion model with careful data curation, producing a few seconds of coherent motion from a single input image, with a VAE that includes a temporal decoder to reduce flicker. AnimateDiff (Guo et al., 2023) took a complementary route — a plug-in motion module trained once and dropped on top of *any* frozen Stable Diffusion checkpoint, instantly animating community image models. Latent video diffusion bought length and resolution by paying the compression cost up front, and it is the recipe most self-hosted video still uses. We dedicate a post to it: latent video diffusion, SVD, and AnimateDiff.

A note on the sampler is worth inserting here, because it is the quietest of the big shifts. The early diffusion-video models inherited the DDPM/DDIM noise schedule from image diffusion, which works but needs many sampling steps. The frontier has largely moved to **flow matching** (rectified flow), which learns a velocity field that transports noise to data along nearly straight paths and therefore integrates in fewer steps — Mochi, Wan, and others are flow-matching models. The reason flow matching matters *specifically* for video is the scale: a video denoiser is enormous and each step is expensive, so cutting the step count from 50 to 20–30 is a direct, large reduction in seconds-per-clip. The mechanics are the same as in images — see [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) — with a noise-schedule *shift* toward higher noise levels for high-resolution and long clips, which we cover in the flow-matching-for-video post. The takeaway for this map: the sampler is no longer the bottleneck it once was, which is part of why length and resolution could grow.

**Spacetime-patch diffusion transformers (2024).** The current frontier replaced the U-Net with a transformer and reframed the input as a sequence of **spacetime patches**. Sora (Brooks et al., OpenAI, 2024) is the landmark: compress video into a spacetime latent, cut that latent into patches that tile across *both* space and time, and run a diffusion transformer over the patch sequence. Because patches are a flexible tokenization, one model handles variable resolution, duration, and aspect ratio, and — this is Sora's central claim — quality and coherence *scale with compute*, the same scaling thesis that drove large language models, now applied to video. This is the [diffusion transformer](/blog/machine-learning/image-generation/diffusion-transformers-dit) recipe with a time axis, and it is why the frontier is now a question of scale and data rather than a missing architectural idea.

![A spacetime patch grid laid out across three time rows and three spatial columns showing how a transformer tokenizes a latent video into patches over both axes](/imgs/blogs/why-video-generation-is-hard-8.png)

**The open and proprietary frontier (2024–2026).** The recipe converged. The open models — CogVideoX (Zhipu, 2024), HunyuanVideo and HunyuanVideo 1.5 (Tencent, 2024–2025), Wan 2.x (Alibaba, 2025), Mochi, LTX-Video — all share the same skeleton: a causal 3D-VAE for spatiotemporal compression, a spacetime diffusion transformer, and increasingly flow matching for the sampler. They differ in scale, data, and conditioning. The proprietary leaders — Sora 2, Google Veo 3.1, Kling 3.0 — push length, resolution (Veo to native 4K), and now synchronized audio, but as far as anyone outside can tell they sit on the same architectural foundation. The frontier is a scale-and-data race on a shared design. That convergence is good news for a learner: master the one recipe and you understand all of them.

![A timeline from VDM in 2022 through SVD, Sora, CogVideoX, Veo 3, and Wan showing the field moving from short clips to long audio-synced video](/imgs/blogs/why-video-generation-is-hard-4.png)

### The families on the four axes

Pulling the lineage together against the four-way tension gives a comparison you can reason from. This is the same data as figure 1, in table form, with the *when to reach for it* added.

| Family | Coherence | Motion | Max length | Cost | When to reach for it |
| --- | --- | --- | --- | --- | --- |
| Per-frame GAN | Flickers | Limited, domain-bound | ~1–2 s | Cheap | Legacy; constrained domains only |
| Frame interpolation | Smooth (anchored) | None new (in-betweens) | Bounded by inputs | Cheap | Post-process fps boost; slow-mo |
| Image-inflated diffusion | Good, can drift | Modest | ~2–4 s | Medium | When you have an image model to reuse |
| Latent video diffusion | Stable | Good | ~2–5 s | High | Self-hosted I2V/T2V today (SVD, AnimateDiff) |
| Spacetime DiT | Strong | Large | 10–60 s+ | Huge | The frontier; quality and length at scale |

Read the table as a history *and* a decision guide. You move down the rows as you buy coherence, motion, and length, and you pay for it in cost. Where you stop depends on your budget and your clip length. For a fixed 4-second product shot you do not need Sora-class compute; for a coherent 30-second narrative you have no other choice.

## A fifteen-line hello world

Enough theory. The fastest way to make the trade-offs concrete is to make a clip. We will use 🤗 `diffusers`, which wraps the modern video models behind a clean pipeline API. There are two natural entry points: **image-to-video** (I2V), where you supply a first frame and the model animates it, and **text-to-video** (T2V), where you supply only a prompt. I2V is the easier, more controllable starting point because you hand the model the hardest part — a coherent first frame — and it only has to produce motion, so we start there with Stable Video Diffusion.

```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

# Load the image-to-video pipeline in fp16.
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()  # fits a 24 GB card by streaming weights

# A single conditioning frame (your "first frame").
image = load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(
    image,
    num_frames=25,            # SVD-XT generates 25 frames (~1 s at 25 fps)
    decode_chunk_size=8,      # decode 8 frames at a time -> lower peak VRAM
    motion_bucket_id=127,     # higher = more motion; the coherence/motion dial
    fps=7,
    generator=generator,
).frames[0]

export_to_video(frames, "rocket.mp4", fps=7)
```

That is the whole thing. Three notes that connect directly to the theory above. `decode_chunk_size=8` is the memory lever from the compute-wall section: it chunks the VAE decode so you do not materialize all 25 full-resolution frames at once, which is what would OOM you. `motion_bucket_id` is the motion dial from the four-way tension made literal — turn it up and you get more movement at the cost of coherence, turn it down and the clip is steadier but stiller; it is the coherence-versus-motion trade you can feel with one integer. And `enable_model_cpu_offload()` is what lets a frontier-ish model run on consumer hardware at all, by keeping only the active submodule on the GPU.

If you would rather start from text — no input image — CogVideoX is the cleanest open T2V to call, and it shows off the spacetime-DiT recipe and the causal 3D-VAE:

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()  # tile the 3D-VAE decode to cut peak VRAM

prompt = "A golden retriever running across a sunlit lawn, slow motion, cinematic."
video = pipe(
    prompt=prompt,
    num_frames=49,          # 49 latent-decoded frames (~6 s at 8 fps)
    guidance_scale=6.0,     # classifier-free guidance strength
    num_inference_steps=50,
).frames[0]

export_to_video(video, "dog.mp4", fps=8)
```

`vae.enable_tiling()` is the same memory lever in a different shape: the causal 3D-VAE decode is the VRAM wall for CogVideoX, and tiling decodes the latent in overlapping spatial tiles so the peak activation never holds the whole frame. `num_frames=49` is the length dial; push it up and you will watch both the render time and the VRAM climb in line with the linear-in-frames token-count formula from earlier. `guidance_scale` is plain classifier-free guidance, exactly as in image diffusion — see [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) — so we do not re-derive it here. Run either script and you have done the thing this whole series unpacks: turned noise (and a prompt or a frame) into a coherent moving clip.

## The stack, top to bottom

Now that you have run a pipeline, let us open it up. Every modern video generator, open or closed, is the same stack of components, and naming them gives you the table of contents for the rest of the series. Read the stack bottom-up as data flowing through.

![A vertical stack of the video generation pipeline from raw video data through a causal 3D-VAE, spacetime latent, spacetime DiT denoiser, flow-matching sampler, conditioning, and decoded frames](/imgs/blogs/why-video-generation-is-hard-5.png)

**Data → causal 3D-VAE → spacetime latent.** Raw video enters and is compressed by a causal 3D-VAE, jointly across space and time, into a compact spacetime latent — typically downsampling space by 8× and time by 4×, for an overall compression of roughly $4 \times 8 \times 8 = 256\times$ in element count before channel changes. "Causal" means the temporal convolutions only look backward in time, so the VAE can encode arbitrarily long video by streaming and can keep the first frame crisp. This is the component I keep calling the real enabler: it determines how many tokens the denoiser must process (and thus the cost), how long a clip you can fit (the length lever), and how much flicker survives compression (the coherence floor). It is the post I would read first after this one: [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression).

**Spacetime latent → spacetime-DiT denoiser.** The latent is cut into spacetime patches and fed to a diffusion transformer that denoises the whole spacetime tensor. This is where temporal attention lives — where coherence is enforced — and where the quadratic-in-frames compute lands. The architectural choices here (full versus factorized attention, patch size, model scale) are the subject of the architecture track.

To make the factorized-attention idea from earlier concrete rather than abstract, here is the heart of a factorized spatiotemporal attention block in PyTorch. The trick is entirely in the reshaping: you fold the time axis into the batch dimension to do spatial attention frame by frame, then fold the spatial axis into the batch to do temporal attention position by position. Self-attention itself is the ordinary scaled-dot-product attention from any transformer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedSpaceTimeAttention(nn.Module):
    """One spatial attention (within each frame) then one temporal
    attention (across frames at each spatial position). Cost scales as
    T*S^2 + S*T^2 instead of full attention's (T*S)^2."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.spatial_qkv = nn.Linear(dim, dim * 3)
        self.temporal_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _attn(self, qkv, B):
        q, k, v = qkv.chunk(3, dim=-1)
        # split heads -> (B, heads, seq, head_dim)
        q, k, v = (x.unflatten(-1, (self.heads, -1)).transpose(1, 2)
                   for x in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)  # SDPA / FlashAttention
        return out.transpose(1, 2).flatten(-2)

    def forward(self, x, T, S):
        # x: (B, T*S, dim). B is the real batch.
        B = x.shape[0]
        x = x.unflatten(1, (T, S))                     # (B, T, S, dim)

        # --- spatial attention: attend within each frame ---
        xs = x.flatten(0, 1)                           # (B*T, S, dim)
        xs = self._attn(self.spatial_qkv(xs), B * T)
        x = xs.unflatten(0, (B, T))                    # (B, T, S, dim)

        # --- temporal attention: attend across frames per position ---
        xt = x.transpose(1, 2).flatten(0, 1)           # (B*S, T, dim)
        xt = self._attn(self.temporal_qkv(xt), B * S)
        x = xt.unflatten(0, (B, S)).transpose(1, 2)    # (B, T, S, dim)

        return self.proj(x.flatten(1, 2))              # (B, T*S, dim)
```

Read the shapes and the cost falls out: the spatial pass runs $B \cdot T$ independent $S \times S$ attentions, the temporal pass runs $B \cdot S$ independent $T \times T$ attentions, and neither ever forms the full $(TS) \times (TS)$ matrix that would blow up memory. This is the literal mechanism behind the FLOP ratio derived above, and it is roughly what CogVideoX, Wan, and HunyuanVideo do inside their DiT blocks (with rotary position embeddings and modulation layers stripped out here for clarity). `scaled_dot_product_attention` dispatches to FlashAttention when available, which is what keeps even the factored attentions memory-efficient.

**Denoiser → flow-matching / SDE sampler.** The denoiser is iterated by a sampler that integrates from noise to a clean latent. Modern video models increasingly use flow matching rather than the older DDPM/DDIM schedule, because flow matching's straight probability paths sample well in few steps at the large scales video demands — the same idea as in image generation, covered in [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), with a noise-schedule shift for high resolution and long clips.

**Conditioning (text / image / motion / audio).** Orthogonal to the main path, conditioning signals are injected: a text prompt via a T5 or CLIP encoder for T2V, a first frame for I2V, motion strength or camera trajectory for control, even audio for synchronized AV. Conditioning is *how you steer* the model, and it is a whole post of its own.

**Decoded frames.** Finally the clean latent is decoded back to pixels by the 3D-VAE's decoder — the step that, as noted, is frequently the VRAM wall — and exported as a video file.

That stack — data, 3D-VAE, spacetime-DiT, sampler, conditioning, frames — is the spine. Hold it in your head and every model report you read becomes a variation on a theme: *which VAE compression ratio, which attention factorization, which sampler, which conditioning, at which scale.*

## A stress test: pushing one decision past its limits

Maps are useful, but the way you really learn a system is to push one decision until it breaks and watch *how* it breaks. Take the most common real engineering question in this field — *"I have a model that makes a clean 5-second clip; how do I get 30 seconds?"* — and stress-test the obvious answers against the machinery we just built.

**Naive answer: raise `num_frames` to cover 30 seconds.** This fails at the VAE before it fails anywhere else. A causal 3D-VAE is *trained* on clips of a certain length (often 49 to 129 frames at the latent level), and its temporal convolutions and normalization statistics are tuned for that range. Feed it six times the frames and the decode either runs out of memory — recall the decode is the VRAM wall, and it scales linearly with frame count — or, if you chunk the decode, you get visible seams where chunks meet because the causal context resets at chunk boundaries. Even if you survive the VAE, the denoiser's positional encodings were trained over a bounded number of latent frames, and extrapolating them produces degraded attention. The clip does not gracefully get longer; it falls off a cliff at the trained length.

**Second answer: generate in chunks and stitch.** Generate five clean 6-second clips and concatenate them. Now the failure is coherence: each clip is internally consistent but they are *unrelated* — different dog, different lawn, a hard cut every six seconds. You have traded the VAE failure for the cross-frame-consistency failure, which is the fundamental coherence problem from earlier in this post. The cut is the visible symptom of the model having no shared latent across chunks.

**Third answer: autoregressive rollout with conditioning.** Generate the first chunk, then condition the next chunk on the *last few frames* of the previous one, so each chunk continues the last. Now you are back in the $\epsilon_{t+1} \approx \gamma \epsilon_t + \delta$ regime: if the per-step amplification $\gamma$ exceeds 1, the identity drifts and the scene smears within a few chunks, exactly the geometric divergence we derived. This is *the* hard problem, and it is why the third answer is a research area rather than a setting.

The honest resolution is that there is no free 30 seconds. You either accept the chunked-and-cut look (fine for some content), or you reach for a method explicitly designed to keep $\gamma < 1$ — overlapped windows with noise-augmented context, keyframe anchoring, Diffusion Forcing — and pay in complexity and compute. The stress test makes the lesson stick: **length is not a slider you turn up; it is a coherence-and-cost problem that gets fundamentally harder past the model's trained horizon.** Every "minute-long video" claim you read should be interrogated for *which* of these three paths it took and what it gave up. That habit — read a capability claim, then ask which trade it paid — is the single most useful one this series can give you.

## Case studies: the frontier in real numbers

Theory is cheap; let us ground it in shipped models and reported numbers. I will flag every figure that is approximate, and where a number is undisclosed I will say so rather than invent one. The landmark models, by year and key idea, look like this — figure 7 is the same data as a matrix.

![A matrix of landmark video models from VDM to Veo listing approximate parameter counts, clip length, and the key architectural idea of each](/imgs/blogs/why-video-generation-is-hard-7.png)

| Model | Year | Params (approx.) | Max length | Key idea |
| --- | --- | --- | --- | --- |
| VDM | 2022 | ~1B (approx.) | 16 frames | 3D U-Net, factorized space-time |
| Make-A-Video | 2022 | — | short | Inflate a T2I model; learn motion unlabeled |
| SVD | 2023 | ~1.5B | ~4 s I2V | Latent image-to-video + temporal decoder |
| Sora | 2024 | Undisclosed | ~60 s | Spacetime patches + DiT at scale |
| CogVideoX | 2024 | 2B / 5B | ~6 s | Open causal-3D-VAE + DiT |
| HunyuanVideo | 2024 | 13B | ~5 s | Large open DiT, strong motion |
| Wan 2.x | 2025 | 1.3B / 14B | ~5 s | Open frontier, two scales |
| Veo 3 | 2025 | Undisclosed | ~8 s + audio | Native 4K + synchronized audio |

A few of these deserve a sentence of context because they each illustrate one of the four axes.

**Sora and the scaling thesis (length and coherence).** Sora's technical report (Brooks et al., 2024) made one claim that reframed the field: treat video as spacetime patches, train a diffusion transformer on them, and *quality scales with compute* — more training compute yields more coherent, longer, more physically plausible video, and emergent properties like rough 3D consistency and object permanence appear as scale grows. Whether those are genuine "world model" capabilities or impressive statistics is the debate we take up in the frontier track; the reframing itself is what mattered, and the whole open frontier followed it.

**SVD and curation (the data lesson).** Stable Video Diffusion's report (Blattmann et al., 2023) is, as much as anything, a paper about *data*: it documents a careful multi-stage curation and captioning pipeline and shows that data quality moved the needle as much as architecture. The lesson — video data curation is a first-class lever, not a preprocessing afterthought — is one the open models internalized.

**CogVideoX, HunyuanVideo, Wan (the open recipe).** These reports are the most useful to a practitioner because they actually disclose the recipe and the numbers. They share the causal-3D-VAE + spacetime-DiT + flow-matching skeleton, report VBench dimensions and parameter counts, and ship weights you can run. The convergence is striking and worth dwelling on: CogVideoX (5B and 2B), HunyuanVideo (13B), and Wan (1.3B and 14B) were developed by independent teams and arrived at nearly the same architecture, which tells you the recipe is not a lucky guess but the current local optimum of the design space. They differ in the details that matter for *your* hardware: HunyuanVideo at 13B parameters wants a large GPU or aggressive offloading, while Wan's 1.3B tier is deliberately sized to run on a consumer card, and CogVideoX-2B sits in between. They also differ in their VAE compression and their conditioning — some lead with T2V, some ship strong I2V variants. When you want exact VBench numbers (subject consistency, dynamic degree, and the rest) and measured VRAM figures on named GPUs, these reports are the primary sources, and we go through them side by side in the open-frontier post.

The practical lesson from the open recipe is that the *two-tier release* is the gift: a 1.3B-class model you can actually run on a 24 GB card to learn, prototype, and fine-tune, and a 14B-class sibling for when quality matters more than iteration speed. The small model is not a toy — it is a coherent, usable generator for short clips — and it lets you do the entire workflow (I2V, T2V, video-LoRA, sampler tuning) on hardware you own before deciding whether the large model is worth the cloud bill. That progression, small-model-to-large-model, is the backbone of the capstone playbook.

**Veo 3 and native audio-video (the multimodal frontier).** Google's Veo 3 / 3.1 pushed two axes the open models had not: native high resolution (up to 4K) and *synchronized audio* generated jointly with the video. Joint audio-video is genuinely harder — the model must keep lips, footfalls, and impacts in sync with the visuals — and it marks the next frontier of *what* video generation outputs, not just how long or how coherent.

#### Worked example: pricing a render farm decision

Suppose you are choosing between self-hosting an open model and calling a proprietary API for a workload of one thousand 5-second 720p clips. On a self-hosted CogVideoX-5B with a single A100 80GB, a 49-frame clip at 50 steps lands in the neighborhood of one to three minutes depending on resolution and offload settings (treat this as an order-of-magnitude figure, not a benchmark). Call it 2 minutes per clip; a thousand clips is ~33 GPU-hours, and at a rough cloud rate of \$2 per A100-hour that is roughly \$66 of compute plus your engineering time, with full control and no per-clip license cost. A proprietary API might charge on the order of cents to low dollars per generated second; at, say, \$0.10 per second that is \$0.50 per 5-second clip and \$500 for the batch — far less setup, more cost at volume, less control over the model. The decision is the same shape as every build-versus-buy call: the API wins for low volume and zero ops, self-hosting wins as volume grows and as you need fine-tuning, custom conditioning, or data privacy. The numbers here are illustrative — get current quotes before committing — but the *structure* of the trade-off is stable, and it is exactly the kind of decision the capstone post drills into.

## How to measure any of this honestly

You cannot reason about coherence, motion, and quality if you cannot measure them, and video metrics are noisier and easier to game than image metrics, so a word of method before the takeaways.

The standard automatic quality metric is **FVD** (Fréchet Video Distance): run real and generated clips through a pretrained video feature extractor (an I3D network trained on action recognition), fit a Gaussian to each set of features, and report the Fréchet distance between the two Gaussians — the direct video analogue of FID for images. Concretely, if the real clips have feature mean $\mu_r$ and covariance $\Sigma_r$, and the generated clips have $\mu_g, \Sigma_g$, then

$$
\text{FVD} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right).
$$

Lower is better. The first term penalizes a difference in *average* features (the generated videos look like a different distribution on average); the second penalizes a difference in feature *spread and correlation* (the generated videos are less diverse, or diverse in the wrong directions). Because the features come from an *action-recognition* network, FVD is sensitive to motion and temporal structure, not just per-frame appearance — which is exactly what you want for video, and also why it is fragile. It is *noisy*: it depends on the number of samples (a few hundred clips gives a high-variance estimate of a covariance matrix in feature space), the clip length, the I3D preprocessing (resolution, frame rate, the number of frames fed in), and the exact real-data reference set. An FVD of 130 reported in one paper and 140 in another are often not comparable at all because the protocols differ. Always check the sample count, clip length, and reference set before trusting an FVD delta, and never compare FVD numbers across papers without confirming they used the same evaluation harness.

The more informative modern benchmark is **VBench**, which decomposes quality into interpretable dimensions — subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, imaging quality, and several more — and scores each separately. The decomposition is the whole point, because it exposes the dynamic-degree-versus-stability gaming problem directly: a model can win subject consistency by barely moving, and VBench shows you that by reporting a low dynamic degree right next to the high consistency. Read VBench as a profile, not a single number.

Three rules for honest video measurement, which we expand in the metrics post. First, **fix the protocol**: same seeds, same resolution, same frame count, same number of sampling steps, with warm-up runs excluded, or your seconds-per-clip and FVD are not comparable. Second, **report cost next to quality**: an FVD or VBench number without the seconds-per-clip and peak VRAM on a *named* GPU is half a result, because the whole game is the quality-cost-length trade. Third, **always report motion alongside consistency**: a consistency score without a dynamic-degree score is gameable and should be distrusted. And for anything user-facing, automatic metrics are a screen, not a verdict — human preference still decides, because the failure modes that matter most (uncanny motion, broken physics, identity drift) are exactly the ones I3D features capture poorly.

## When to reach for each approach

Let me close the technical content with the decisive recommendations the four-way tension implies, the section you can actually act on.

**Reach for image-to-video over text-to-video whenever you can supply a first frame.** I2V hands the model a coherent starting point and asks only for motion, which is strictly easier than conjuring both appearance and motion from text. If your product has a key image — a product shot, a storyboard frame, a generated still — animate it with I2V rather than describing it to a T2V model. Use T2V when you genuinely have only words.

**Do not reach for autoregressive long-video rollout for a fixed short clip.** If you need a clean 5-second clip, generate it in one shot with a model whose trained clip length covers it. Autoregressive chunking exists to break the length ceiling, and it pays for that with error accumulation and identity drift; invoking it for a clip that fits in one window is buying a problem you do not have.

**Do not reach for full 3D attention when factorized attention clears your coherence bar.** Full spacetime attention is the most expressive and the most expensive, scaling quadratically in frames; factorized (spatial-then-temporal) attention usually hits the coherence target at a fraction of the FLOPs. Start factorized, measure, and only spend on full attention if the coherence is genuinely insufficient — which, at typical clip lengths, it usually is not.

**Reach for frame interpolation as a post-process, not a generator.** If you need 24 fps and your model is coherent at 8 fps, generate at 8 and interpolate up with RIFE/FILM. It is far cheaper than generating every frame and, because interpolation is anchored to real generated frames, it adds smoothness without adding drift.

**Match the model scale to the clip length, not to the demo.** A 4-second product shot does not need a 13B frontier model; a small open model with VAE tiling on a single consumer GPU will do it. Reserve the frontier-scale models and their compute for what actually needs length, large motion, or 4K-plus-audio. The honest default for self-hosting in 2026 is a small-to-mid open model (CogVideoX-class, small Wan) with offload and tiling, and you scale up only when the clip demands it.

**Reach for quantization and caching before you reach for a bigger GPU.** When a model nearly fits but OOMs, the first moves are not a more expensive card — they are FP8 or 4-bit weight quantization, VAE tiling and slicing, model CPU offload, and step-reducing samplers or feature caching that skip redundant denoiser computation across steps. These routinely turn an impossible render into a slow-but-working one on the hardware you have, and the same efficiency toolkit from the image world transfers directly — see [quantization, caching, and efficient inference](/blog/machine-learning/image-generation/quantization-caching-and-efficient-inference) for the image-side mechanics and the serving post in this series for the video specifics. Buy hardware last, after you have spent the free memory and latency you already have.

## Key takeaways

- **Video is image generation times temporal coherence, under a brutal compute budget.** The extra axis is time, and it is the whole difficulty. Every component of the stack grows a time dimension.
- **The four properties — coherence, motion, length, cost — trade against each other.** No current model wins all four; the engineering is choosing which corner to sacrifice for your use case.
- **Compute is linear in frames for tokens but quadratic in frames for full attention.** That single fact explains factorized attention, the length ceiling, and why a 5-second clip can be 100K+ tokens.
- **The VAE decode, not the denoiser, is often the VRAM wall.** Chunked decode and VAE tiling are the levers that turn an OOM into a finished render.
- **Coherence is a constraint you must model, and small per-frame errors compound geometrically.** Even a 99.5%-per-frame model drifts over a few hundred frames, which is why long video is a separate research problem.
- **Motion drags in implicit, learned, approximate physics.** Common motion looks right; rare or precise motion reveals the model learned statistics, not causal physics.
- **The lineage converged on one recipe:** causal 3D-VAE + spacetime diffusion transformer + flow matching, the same skeleton across CogVideoX, HunyuanVideo, Wan, and the closed leaders.
- **Measure quality next to cost and motion, or do not trust the number.** Consistency without dynamic degree is gameable; FVD without its sample protocol is not comparable.
- **Video is information-cheap but compute-expensive per frame.** Temporal redundancy means each frame adds little new information, which is what makes video learnable, but each frame still adds tokens and quadratic attention, which is what makes it expensive. Compress before you generate.
- **Data and captions are a first-class constraint, not an afterthought.** Well-captioned, motion-rich video is scarce; curation and motion-aware re-captioning move quality as much as architecture, and the rare-motion gaps in the data are exactly where generated video looks wrong.
- **Read every length claim by asking which path it took.** Single-shot within the trained horizon, chunked-and-cut, or autoregressive rollout with $\gamma < 1$ — the trade each one pays is the real story behind any "minute-long video" headline.
- **Pick I2V when you have a frame, factorized attention by default, interpolation for fps, quantization and caching before a bigger GPU, and model scale matched to clip length.** Those defaults handle most real workloads.

## The map of the rest of the series

This post framed the problem. The rest of the series fills in the stack and the frontier across four tracks.

**Track A — Foundations** establishes the pieces: how video is represented and why temporal redundancy is the prior that makes it learnable; [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), the causal 3D-VAE that is the real enabler; [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion), the architectures that add the time axis; and [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation), FVD, VBench, and honest evaluation.

**Track B — The diffusion-video architecture** goes deep on the denoiser: spatiotemporal attention patterns and the compute-coherence trade; the spacetime diffusion transformer and Sora's patch framing; latent video diffusion with SVD and AnimateDiff; flow matching for video; conditioning by text, image, motion, and camera; and joint audio-video generation.

**Track C — The frontier models** reads the actual systems: Sora and the world-simulator thesis; Veo and cinematic generation; the open frontier of Wan, HunyuanVideo, and CogVideoX; efficient and real-time generation; long-video and autoregressive rollout; and camera control and 4D generation.

**Track D — World models, frontier, and practice** closes the loop: video models as world models (Genie); the physics limits of learned simulation; evaluating and red-teaming video; efficient inference and serving; and the capstone, [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook), which assembles a real production pipeline end to end.

Whenever a post needs a pure image-diffusion mechanism — the DDPM loss, the score identity, the reverse SDE, classifier-free guidance, the DiT block, flow matching — it links out to the companion [image generation series](/blog/machine-learning/image-generation/diffusion-from-first-principles) rather than re-deriving it. Your job in this series is the part image generation does not cover: time. Start with the 3D-VAE post; it is where the difficulty in this map becomes a thing you can build.

## Further reading

- Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet. "Video Diffusion Models." 2022. The 3D U-Net factorization that opened the diffusion-video era.
- Singer et al. "Make-A-Video: Text-to-Video Generation without Text-Video Data." 2022. The inflate-an-image-model trick.
- Blattmann et al. "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets." 2023. Latent video diffusion and the data-curation lesson.
- Guo et al. "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning." 2023. The plug-in motion module.
- Brooks et al. "Video Generation Models as World Simulators" (Sora technical report). OpenAI, 2024. Spacetime patches and the scaling thesis.
- Peebles, Xie. "Scalable Diffusion Models with Transformers" (DiT). 2023. The transformer backbone the frontier adopted.
- Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow Matching for Generative Modeling." 2023. The sampler that scales to the video regime.
- CogVideoX, HunyuanVideo, and Wan technical reports (2024–2025). The disclosed open recipe with VBench numbers — the practitioner's primary sources.
- 🤗 `diffusers` documentation, video pipelines (`StableVideoDiffusionPipeline`, `CogVideoXPipeline`). The runnable reference for the code in this post.
