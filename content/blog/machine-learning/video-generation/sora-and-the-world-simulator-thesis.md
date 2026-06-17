---
title: "Sora and the World-Simulator Thesis: What Holds Up and What Is Hype"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A rigorous, honest read of OpenAI's Sora and Sora 2 and the claim that scaling video models yields world simulators — separating the emergent capabilities that genuinely hold up from the physics failures that prove it is not a simulator."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "text-to-video",
    "sora",
    "world-models",
    "spacetime-patches",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/sora-and-the-world-simulator-thesis-1.png"
---

In February 2024, OpenAI published a short technical report titled *Video generation models as world simulators*, attached a reel of one-minute clips that were visibly a generation ahead of anything public, and let a single sentence do the heavy lifting: that scaling video generation "is a promising path towards building general purpose simulators of the physical world." The clips were extraordinary. A woman walks through a neon Tokyo street and the reflections in the puddles track her correctly across a dozen seconds. A papercraft coral reef holds its geometry as the camera pushes in. And then, in the same report, a glass of liquid tips over and the liquid appears *before* the glass breaks, a cookie gets a bite taken out of it with no bite mark left behind, and a treadmill runs a man backward. The same model that tracked puddle reflections for twelve seconds could not reliably conserve the volume of a liquid for one.

That tension is the entire subject of this post. Sora is, by a wide margin, the most consequential video generation system of the 2024–2026 frontier, and the "world simulator" framing OpenAI chose for it is the most consequential *claim* in the field. Both deserve a careful, technical, unsentimental read. By the end of this post you will be able to do four concrete things. First, you will know *what Sora actually is* at the architecture level — a large diffusion transformer denoising spacetime patches of a video latent, trained on variable durations and resolutions via patch packing, with no secret sauce beyond scale, data, and that recipe. Second, you will be able to *separate the emergent capabilities that genuinely hold up* — long-range coherence, approximate 3D consistency, frequent object permanence — *from the ones that do not* — gravity, conservation laws, causal state. Third, you will understand *why* a model can be visually coherent and physically wrong at the same time, in terms of statistical versus causal modeling. Fourth, you will know what changed in Sora 2 and what the thesis implies if it is even partly true — the bridge to agents and robotics learning from generated experience.

![Stacked diagram of the Sora recipe running from web-scale video through a three-dimensional VAE and spacetime patches into a scaled diffusion transformer and back out to coherent frames](/imgs/blogs/sora-and-the-world-simulator-thesis-1.png)

A note on sourcing before we start. Sora is a closed model. OpenAI's technical report is deliberately light on numbers — it gives no parameter count, no training-set size, no FVD, no VBench scores, and the most quantitatively informative figure in the whole document is a qualitative "base compute, 4x compute, 32x compute" comparison with no axis labels. So everywhere in this post that I give a specific figure for Sora, treat it as an *order-of-magnitude estimate* unless I say it comes directly from the report, and I will flag the difference. The architecture, by contrast, is well understood, because it is the same recipe the [open frontier](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox) converged on independently — the [video diffusion transformer on spacetime patches](/blog/machine-learning/video-generation/video-diffusion-transformers) that this series has already built from the ground up. We will recap it precisely enough to reason about, then spend the bulk of our depth on the claim. The recurring frame of the [whole series](/blog/machine-learning/video-generation/why-video-generation-is-hard) holds throughout: video is spatial generation times temporal coherence under a brutal compute budget, and the world-simulator thesis is the bet that pushing the third factor — compute — far enough makes the first two emergent rather than engineered.

## 1. What Sora actually is

Let us be deflationary first, because the marketing is loud and the architecture is calm. Stripped of the framing, Sora is the standard video-generation stack from this series, scaled up. There is no exotic component. The report itself says as much, in plain language: Sora is a "diffusion transformer" that operates on "spacetime patches" of a video latent, and the whole document is organized around the claim that this *ordinary* recipe, scaled, produces extraordinary behavior. That is the interesting part — not that the recipe is novel, but that OpenAI bet the recipe needed no fundamental new idea, only more of it.

Walk the stack from the bottom. First, a **visual encoder** — a network OpenAI calls a "video compression network" and that the rest of us call a [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression). It takes raw video and compresses it in both space and time into a lower-dimensional latent. The report is explicit that this compression happens "both temporally and spatially," which is the defining property of a 3D-VAE versus a frame-by-frame image VAE. This is the single most important component for cost and length, and the series has a whole post on why. The denoiser never touches pixels; it works entirely in this compressed latent space, which is the only reason a one-minute clip is computationally tractable at all.

Second, **spacetime patches**. Once you have the latent video — a 4D block of shape roughly $T' \times H' \times W' \times C$ (latent frames, height, width, channels) — you slice it into small cuboids, each spanning a little time and a little space, and flatten those cuboids into a sequence of tokens. The report's own words: Sora represents video as "a collection of spacetime patches," analogous to tokens in a language model. This is the tokenization that lets a transformer eat video, and the [spacetime-patch post](/blog/machine-learning/video-generation/video-diffusion-transformers) derived the exact token-count arithmetic that makes attention quadratic in clip length. There is no new idea here either; it is the [DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit) patch embedder with one more axis.

Third, the **diffusion transformer** itself — Brooks, Peebles, and colleagues built directly on Peebles and Xie's DiT, the transformer-based diffusion backbone from the image world. The model is conditioned on text (Sora uses a re-captioning step, more on that shortly) and trained with the standard diffusion objective: predict the noise added to a patchified latent at a sampled timestep. Sampling runs the reverse process — start from Gaussian noise in latent space, denoise iteratively to a clean latent video, decode with the VAE. Everything here you have seen in this series and the [image series](/blog/machine-learning/image-generation/diffusion-from-first-principles). The novelty budget is spent entirely on scale and data.

It helps to see the whole path at once, because the same components serve both training and sampling. During training, raw video is encoded by the 3D-VAE into a latent and re-captioned by a captioner into a dense text description; the diffusion transformer learns to denoise the noised latent conditioned on that text. At sampling time the *same* conditioned denoiser is driven by a user prompt (itself expanded by a language model into the dense style the model was trained on), runs the iterative denoise loop, and the VAE decodes the result. There is one merged path through these blocks, which is why understanding the training graph tells you exactly what happens at inference.

![Graph of the Sora training and sampling path where a recaptioner and a three-dimensional VAE feed a text-conditioned denoiser whose loop decodes to frames](/imgs/blogs/sora-and-the-world-simulator-thesis-5.png)

The figure makes the dependency structure explicit: the captioner and the VAE are *upstream* of the conditioned transformer, and the denoising loop is the only iterative part. Two consequences follow that matter for the rest of this post. First, much of what looks like "Sora understanding a prompt" is actually the captioner-and-LLM front end doing careful text expansion — the transformer's job is narrower than the demos suggest. Second, every frame Sora produces passes through the *same* learned denoiser conditioned on the *same* text; there is no per-frame physics solver, no separate dynamics module, nothing in the path that could enforce a physical law. The path is text-conditioned pixel-statistics denoising, end to end, and that single fact constrains everything the model can and cannot do.

So when someone asks "what is the secret of Sora," the honest answer is: there is no secret component. The recipe is public and has been independently reproduced by [CogVideoX, HunyuanVideo, and Wan](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox). What OpenAI had that the field did not, in early 2024, was (a) an enormous, well-captioned, high-quality video dataset, (b) the compute to train a very large DiT on very long, high-resolution sequences, and (c) the conviction to do it at a scale nobody else had committed to. The thesis — "scale this and a world simulator emerges" — is a claim about what that scale *buys*, and that is what we have to evaluate.

#### Worked example: counting Sora's tokens

Let us make the scale concrete with the series' standard arithmetic, applied to a Sora-sized clip. The headline demos are roughly twenty seconds at up to $1920 \times 1080$. Take a one-minute clip — Sora's stated maximum — at $1280 \times 720$, 30 fps, so $T = 1800$ frames. A causal 3D-VAE compressing $4\times$ in time and $8\times$ in each spatial dimension gives a latent of $T' = 450$, $H' = 90$, $W' = 160$. With a $1 \times 2 \times 2$ patch the sequence length is

$$
L = \frac{T'}{1} \cdot \frac{H'}{2} \cdot \frac{W'}{2} = 450 \cdot 45 \cdot 80 = 1{,}620{,}000 \text{ tokens}.
$$

That is 1.6 *million* tokens for one minute of 720p. Self-attention is quadratic, so a naive full-3D attention pass over that sequence is on the order of $L^2 \approx 2.6 \times 10^{12}$ score entries — comfortably outside what any single device holds. This is exactly why the [attention pattern](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns) and the [3D-VAE compression ratio](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) are the load-bearing engineering choices, not the "world simulator" framing. The numbers here are illustrative — OpenAI never published its VAE's compression factor or patch size — but they are the right order of magnitude, and they tell you immediately that Sora's real achievement is *systems and data*, executed against a recipe that was already known.

## 2. Spacetime patches and variable-size training

The one genuinely elegant idea in the Sora report — the one worth dwelling on — is not a new layer. It is what the spacetime-patch representation *enables*: training on video at its native size. This is worth getting exactly right, because it is the mechanism behind several of the capabilities OpenAI later attributes to scale.

Here is the problem it solves. Prior video models, almost without exception, trained on a fixed shape: resize and crop every clip to, say, $256 \times 256$, sample a fixed number of frames, say 16, and train. That is convenient — every batch element is the same tensor shape — but it throws away enormous amounts of information and bakes in a fixed aspect ratio. A model trained only on square crops generates square video and has never seen a wide cinematic frame or a vertical phone clip. Sora's report calls this out directly: prior approaches "resize, crop or trim videos to a standard size," and Sora instead trains "on video and images at their native sizes."

The reason it *can* is the patch representation. A spacetime patch is a fixed-size cuboid of latent regardless of how big the clip is. A short vertical clip becomes a short sequence of patches; a long wide clip becomes a long sequence of patches. Both are just *sequences of the same kind of token*, differing only in length and in their position encodings. So you do not need a fixed tensor shape — you need a transformer that handles variable-length sequences, which is exactly what transformers are built for. You pack clips of different durations, resolutions, and aspect ratios into one training run, and the model learns the full range.

The one piece that makes variable size *work* rather than merely *fit* is the position encoding. A token at latent position $(t, h, w)$ needs a positional signal that means the same thing whether the clip is two seconds or sixty, square or widescreen — otherwise a model trained at one size would generalize poorly to another. The modern answer, used across the open recipe and almost certainly in Sora, is a 3D extension of rotary position embedding (RoPE): split the head dimension into three groups, rotate one by an angle proportional to $t$, one by $h$, one by $w$, and the transformer reads absolute and relative position along each axis directly from the dot product. Crucially, RoPE *extrapolates* — a model trained on clips up to length $T'$ can be sampled at a somewhat longer $T'$ because the rotation generalizes past the trained range (with degradation, but gracefully). That extrapolation property is part of why a single Sora model serves a range of durations: the position encoding does not hard-code a maximum length, it encodes a *continuous* notion of where each patch sits in space and time. This is a real mechanism, not emergence — and it is another case where keeping the mechanism distinct from the mystery clarifies what scale is and is not responsible for.

![Grid showing clips of different durations and aspect ratios each sliced into spacetime patches that pack into one variable-length token sequence](/imgs/blogs/sora-and-the-world-simulator-thesis-2.png)

The payoff is concrete and the report is unusually specific about it for once. Training on native aspect ratios, OpenAI reports, gives better framing and composition than training on cropped square video — a model trained on square crops sometimes generates subjects with their heads cut off, while the native-trained model frames them properly. And training on variable durations means one model serves a vertical phone clip, a square social post, and a widescreen cinematic shot, sampling each at the requested size by simply choosing how many patches to denoise. This is the "flexible duration, resolution, and aspect ratio" property the report advertises, and it falls directly out of treating video as a packed token sequence rather than a fixed tensor.

#### Worked example: why packing changes the data distribution

Suppose your corpus is 40% landscape 16:9, 35% vertical 9:16, and 25% square 1:1 (a realistic web-video mix). Crop everything to square and you have, in effect, thrown away the outer regions of 75% of your data and taught the model that the world is square. Train on native shapes via patch packing and the model sees the full landscape composition, the full vertical composition, and learns that a "cinematic wide shot" and a "phone video" are different framings of the same physical scenes. The capability gain here is real and *mechanistic* — it comes from the data, not from emergence. I flag this because it matters for the thesis: when the report shows that scale improves "framing," part of that is genuinely the variable-size training enabled by patches, not a mysterious emergent property. Keeping the mechanism and the mystery separate is the whole discipline of reading this report honestly.

Here is the conceptual patch-packing sketch in PyTorch — closed model, so this mirrors the *recipe* rather than Sora's code, and we will run a real open stand-in in section 6.

```python
import torch
import torch.nn.functional as F

def patchify_clip(latent, patch=(1, 2, 2)):
    """latent: (C, T', H', W') from a 3D-VAE.
    Returns a token sequence (L, C*pt*ph*pw) plus 3D positions (L, 3)."""
    C, Tt, Hh, Ww = latent.shape
    pt, ph, pw = patch
    # fold each non-overlapping cuboid into a token vector
    x = latent.reshape(C, Tt // pt, pt, Hh // ph, ph, Ww // pw, pw)
    x = x.permute(1, 3, 5, 0, 2, 4, 6)            # (T'',H'',W'',C,pt,ph,pw)
    Td, Hd, Wd = x.shape[:3]
    tokens = x.reshape(Td * Hd * Ww // pw * 0 + Td * Hd * Wd, -1)
    # 3D position for every token: (t_index, h_index, w_index)
    pos = torch.stack(torch.meshgrid(
        torch.arange(Td), torch.arange(Hd), torch.arange(Wd), indexing="ij"
    ), dim=-1).reshape(-1, 3)
    return tokens, pos

def pack_variable_clips(clips, patch=(1, 2, 2)):
    """Pack clips of DIFFERENT shapes into one padded batch.
    Each clip contributes a different number of tokens; we pad to the max
    and build an attention mask so padding tokens are ignored."""
    seqs, poss = zip(*(patchify_clip(c, patch) for c in clips))
    lengths = [s.shape[0] for s in seqs]
    Lmax = max(lengths)
    d = seqs[0].shape[1]
    batch = torch.zeros(len(seqs), Lmax, d)
    mask = torch.zeros(len(seqs), Lmax, dtype=torch.bool)
    for i, (s, L) in enumerate(zip(seqs, lengths)):
        batch[i, :L] = s
        mask[i, :L] = True       # True = real token, False = padding
    return batch, mask, lengths
```

The point of this sketch is the second function. A landscape clip and a vertical clip produce token sequences of *different lengths*, and the only machinery you need to train on both is padding plus an attention mask — no resizing, no cropping, no fixed shape. That is the entire trick that lets Sora train on native sizes, and it is why the patch representation is the quietly important idea in the report.

## 3. The emergent capabilities OpenAI highlighted

Now the interesting half. OpenAI's report leans on a specific list of behaviors that, it argues, "emerge" purely from training at scale, with no explicit engineering for them. The report's own phrasing is that these are "emerging simulation capabilities" that "video models exhibit ... when trained at scale." Let us take the list seriously, one item at a time, and ask the only question that matters: is this a genuine model of the world, or sophisticated interpolation over the training distribution that *looks* like one?

The headline capabilities, in OpenAI's framing, are: **long-range temporal coherence** (subjects and scenes stay consistent over many seconds), **3D consistency** (the camera can move and the scene's geometry stays plausible, as if there were a real 3D space being filmed), **object permanence** (people and objects persist even when temporarily occluded or out of frame), and **simulated interactions** (a painter leaves strokes on a canvas, a person eating a burger leaves bite marks). OpenAI also notes that Sora can simulate "digital worlds" — it can render Minecraft-like footage with a plausibly controlled player and a coherently rendered environment.

![Matrix mapping each world-model capability claim to its supporting evidence and its counter-evidence with a verdict column](/imgs/blogs/sora-and-the-world-simulator-thesis-3.png)

The honest first-order read is that these capabilities are *real but graded*. Long-range coherence is the most defensible: a twelve-second Sora clip genuinely holds a consistent subject identity, lighting, and scene in a way that 2023 models could not, and this is verifiable from the public clips, not just the report. The series' [scaling discussion](/blog/machine-learning/video-generation/video-diffusion-transformers) explains *why* — more attention capacity over a longer spacetime sequence means more of the clip can be mutually conditioned, so the model can enforce "this is the same person ten seconds later." That is a real, mechanistic account of coherence, and it holds up.

3D consistency and object permanence are weaker but still present. The camera in a good Sora clip does move "like a camera" — parallax is roughly right, near objects move faster than far ones — and occluded objects often reappear correctly. But "often" is the operative word. These are *statistical tendencies the model learned because they are overwhelmingly true in real video*, not guarantees it enforces. We will see in section 4 that they fail in exactly the cases where the training distribution is thin: unusual occlusions, rare object interactions, long horizons. The capability is real; the framing of it as a *property* rather than a *tendency* is where the hype creeps in.

### 3D consistency as learned parallax, not geometry

It is worth pinning down what "3D consistency" actually means in a model that has no 3D representation, because this is where the strong reading is most tempting. When the report shows a camera orbiting a subject and the occluded sides revealing themselves plausibly, the natural conclusion is "the model has a 3D scene in its head and is rendering it from different viewpoints." That conclusion is wrong, and the way it is wrong is instructive. The model has no voxel grid, no mesh, no neural radiance field, no explicit notion of depth. What it has is a transformer that learned, from millions of real camera moves, the *2D-pixel statistics of how scenes transform under camera motion* — near things slide faster, far things slower, newly-revealed surfaces tend to be continuations of nearby ones. That learned regularity reproduces parallax well enough to *look* 3D-consistent over a few seconds of moderate camera motion.

The tell is where it breaks. Push the camera through a large angle, or have it return to a viewpoint it occupied seconds earlier, and the geometry quietly fails to match — a wall that was a certain length grows or shrinks, a room's layout reorganizes, a previously-seen back of an object comes back different. A model with an explicit 3D scene would be exactly self-consistent across viewpoints by construction; a model with learned 2D parallax statistics is consistent only over the short, common camera moves that dominate its training data, and drifts otherwise. So "3D consistency" is real as a short-horizon visual effect and false as a claim about an internal 3D model. The honest phrasing is *approximate, learned, view-local* consistency, which is impressive and useful and is not the same thing as geometry.

### Object permanence as an attention property, not a guarantee

It is worth being precise about object permanence, because it is the capability most often cited as evidence of a "world model." Here is what is actually happening. The transformer's self-attention lets every spacetime patch attend to every other patch in the sequence (or, under factorization, to the same spatial region across time). When a ball rolls behind a couch and out again, the patches depicting the re-emerged ball can attend to the patches that depicted it before the occlusion, and the model — having seen millions of real occlusions — has learned that the post-occlusion appearance should match the pre-occlusion one. That is permanence, and it is genuinely impressive.

But notice what it is *not*. There is no variable in the model that says "a ball exists at position $x$ with velocity $v$, currently hidden." There is no persistent object representation, no scene graph, no state. There is only attention over patches, and a learned statistical regularity that occluded-then-revealed things tend to match. When the occlusion is long enough, or the scene busy enough, that the relevant pre-occlusion patches get drowned out in the attention, permanence simply fails — the object comes back wrong, or doesn't come back, or a new one appears. A real world model with explicit state would not fail this way; a statistical pattern-matcher fails exactly this way. This distinction — *learned correlation that mimics state* versus *actual state* — is the crux of the entire thesis, and we will formalize it in section 5.

#### Worked example: the occlusion stress test

Here is the test I run on any "world model" claim, conceptually. Generate a clip where a counter has five identical apples. A hand sweeps across, briefly occluding three of them, then withdraws. A model with object permanence as a *property* returns exactly five apples in their original positions every time. A model with permanence as a *tendency* returns five most of the time, four or six some of the time, and occasionally swaps an apple's position or color — because it is sampling a plausible continuation, not tracking five tracked entities. Empirically, frontier video models including Sora behave like the second case: permanence holds for short, simple occlusions and degrades with occlusion duration, scene complexity, and object count. That degradation curve *is* the signature of statistical modeling. If you only ever test the easy case, you will mistake the tendency for a property — which is precisely how the world-simulator claim gets oversold.

## 4. The failure cases that reveal it is not a physics engine

OpenAI, to its credit, includes a "Discussion" section in the report that is unusually candid about limitations, and it is the most important paragraph in the whole document. Sora, it admits, "does not accurately model the physics of many basic interactions, like glass shattering," and "other interactions, like eating food, do not always yield correct changes in object state." Let us catalog the failure modes precisely, because each one is diagnostic of *what kind of model this is*.

**Broken conservation laws.** The canonical example, straight from the report's failure reel: a glass tips, and liquid spills, but the volume is not conserved — liquid appears, vanishes, or changes amount. Eating food does not reliably leave the food diminished. These are violations of conservation of mass and volume, the most basic invariants of physical reality. A physics engine enforces conservation by construction; Sora has no notion of "amount of stuff" to conserve.

**Gravity and trajectory errors.** Objects float, fall at wrong rates, or follow trajectories no real object would. A thrown object's arc is sometimes a plausible parabola and sometimes not, depending on whether the training distribution had enough similar examples. There is no gravitational constant in the model — only a learned tendency for things to generally move downward when unsupported, which holds in common cases and breaks in uncommon ones.

**Object teleportation and spontaneous generation.** Objects pop into and out of existence, especially in busy scenes or under occlusion (the permanence failure from section 3). Multiple instances of an object appear or merge. Causality runs backward — the report's example of the glass where the liquid appears before the breakage is a *causal ordering* failure, which is even more damning than a magnitude error.

![Before and after comparison contrasting a coherent but physics-broken clip against a physically plausible one across conservation gravity and causal ordering](/imgs/blogs/sora-and-the-world-simulator-thesis-4.png)

**State-change errors.** Cut a cookie and it should stay cut; the report notes Sora often fails to apply the persistent state change. This is the deepest failure, because it shows the model does not maintain a *consistent world state* that interactions modify — each frame is a plausible image given the prompt and recent frames, but there is no ledger of "what has happened to this object."

To make this concrete and checkable, here is a named catalog of the physics failure modes that recur across the Sora report and the broader video-generation literature, each with the invariant it violates:

- **Non-conservation of mass/volume** — liquid, food, or material that appears, vanishes, or changes amount (the spilling-glass and bitten-cookie cases). Violates conservation of mass.
- **Incorrect gravity and free-fall** — objects that float, fall too slowly or too fast, or hang unsupported. Violates the constant-acceleration law of free fall.
- **Non-ballistic trajectories** — thrown or falling objects following arcs no real projectile would. Violates Newtonian mechanics under gravity.
- **Object permanence failure** — occluded objects that fail to return, return wrong, duplicate, or merge. Violates the persistence of objects through occlusion.
- **Spontaneous generation/deletion** — objects popping into or out of existence in busy scenes. Violates the continuity of matter.
- **State-change non-persistence** — a cut that doesn't stay cut, a footprint that doesn't remain. Violates the persistence of irreversible state changes.
- **Reversed causal ordering** — effects preceding causes (liquid before the glass breaks). Violates temporal causality, the deepest of the set.
- **Implausible articulated motion** — limbs that bend the wrong way, gait that loses leg count, the treadmill running backward. Violates the kinematic constraints of bodies.

The pattern across this list is uniform: every entry is a violation of a *conservation law, a continuity constraint, or a causal ordering* — exactly the structural facts that a passive statistical model has no machinery to enforce, and exactly the facts a real simulator enforces by construction. This is not a grab-bag of unrelated bugs; it is the coherent signature of a model that learned what video *looks like* without learning what *generates* it.

Here is the crucial point about these failures: they are *not* the same as the flickering, identity-drift, and motion-jitter failures the rest of this series catalogs. Those are coherence failures — the model fails to keep the *appearance* consistent. The failures in this section are different and more fundamental: the appearance can be perfectly coherent, every frame sharp and consistent with its neighbors, and the *physics* is still wrong. The glass-then-liquid clip is not blurry or flickery; it is a crisp, coherent rendering of something that cannot happen. That is the signature that distinguishes a *renderer of plausible pixels* from a *simulator of the world*. Coherence and physical correctness are orthogonal, and Sora can have a lot of the first with very little of the second.

#### Worked example: coherent and wrong at the same time

Take the series' running thread and bend it. Generate "a glass of water falling off a table and shattering on the floor." Score the result on two independent axes. On the [VBench-style coherence axis](/blog/machine-learning/video-generation/the-metrics-of-video-generation) — subject consistency, background consistency, motion smoothness — a good Sora clip scores high: the glass is the same glass throughout, the kitchen is the same kitchen, the motion is smooth. On a *physical-correctness* axis — is volume conserved, does the glass break before or after impact, do the shards follow ballistic arcs, does the water pool correctly — it frequently fails. The two scores are nearly uncorrelated. This is the single most important empirical fact for the thesis, and it is why pointing at a beautiful coherent clip proves coherence, not world modeling. The series' [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) makes the deeper point that *most published metrics measure the first axis and almost none measure the second*, which is part of why the thesis is hard to falsify with the standard benchmarks — they were not built to test physics.

## 5. The scaling thesis and the counterarguments

Now we can state the thesis precisely and weigh it. OpenAI's claim, distilled: a diffusion transformer trained at sufficient scale on spacetime patches develops, as an *emergent* consequence of scale alone, an internal model of the physical world good enough to call a "general purpose simulator," and continued scaling is a "promising path" toward that simulator. The report's strongest empirical support is the compute-scaling comparison: the same prompt at "base compute," "4x compute," and "32x compute," where the 32x sample is dramatically more coherent and detailed. The argument is: coherence demonstrably improves with scale, therefore physical understanding will too.

Let us grant the premise that is actually demonstrated — *coherence* improves with scale — and then examine the inference to *physical understanding*. This is where the thesis is most contestable, and the honest position is that the inference does not go through, for a reason that is precise and provable rather than merely skeptical.

### Statistical modeling versus causal modeling

A diffusion model trained with the standard objective learns to approximate the data distribution $p_\text{data}(x)$ over clips $x$. The training loss — predicting the noise added to a clean latent — is, up to constants, a bound on the negative log-likelihood of the data under the model. So what the model provably learns is to assign high probability to clips that look like the training videos and low probability to clips that don't. In math, it learns the score $\nabla_x \log p(x)$ of the data distribution (the [score-based view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) makes this exact). That is a model of *pixel statistics over spacetime* — what configurations of patches tend to co-occur.

A world model, in the sense that would justify "simulator," is something else: a model of the *causal dynamics* of the world. Formally, it is a transition function or a structural causal model — given a state $s_t$ and an action or intervention $a_t$, it predicts the *next state* $s_{t+1}$ in a way that respects the causal structure of physics, so that it answers counterfactual and interventional questions correctly ("what if I had pushed harder"), not just observational ones ("what usually happens next"). The distinction is Pearl's ladder of causation, and it is not a matter of degree — observational and interventional distributions can agree on everything you have observed and disagree on every intervention you have not.

Here is the crux. *Matching $p_\text{data}(x)$ arbitrarily well does not entail learning the causal dynamics.* Two systems with identical observational distributions can have completely different causal structure — this is the standard identifiability problem in causal inference. A model that has perfectly learned "videos of falling glasses look like *this*" has learned the observational distribution of falling-glass videos. It has *not* learned the law "force equals mass times acceleration" that generates that distribution, and crucially it has no way to, because the law is not identifiable from passive observation alone — you would need interventions, or a built-in inductive bias toward the law. Sora has neither. It watches; it does not act, and it has no physics prior. So the most it can learn is the observational regularities, which is exactly what we see: it reproduces common physical scenarios well (because they are dense in the data) and fails on rare or counterfactual ones (because the regularity it learned is statistical, not causal).

![Tree splitting the world-model claim into a weak statistical reading the model satisfies and a strong causal reading with explicit state it does not](/imgs/blogs/sora-and-the-world-simulator-thesis-7.png)

#### Worked example: two worlds, one video distribution

To see that matching the data does not pin down the dynamics, here is the smallest concrete case. Imagine a corpus of clips of a ball that is dropped and bounces. World A is real Newtonian physics: the ball falls under gravity $g$, loses a fixed fraction of energy per bounce, and its trajectory is fully determined by its initial height. World B is a lookup table: it has memorized, for each starting height seen in training, the exact pixel sequence that followed, and for unseen heights it copies the nearest memorized one. On the *training distribution* — the heights that appear in the data — worlds A and B produce identical clips, so a model that perfectly fits the data cannot tell them apart from observation alone. They have the same observational distribution $p_\text{data}(x)$.

Now intervene: drop the ball from a height *between* two memorized values, or on the Moon where $g$ is one-sixth. World A answers correctly — it computes the new trajectory from the law. World B fails — it has no law, only interpolation between memorized cases, so it produces a plausible-looking but physically wrong bounce, and it has no notion of $g$ to rescale. A diffusion model trained on passive video is structurally far closer to World B than World A: it is an extremely sophisticated interpolator over the observed distribution, with no $g$ inside it. This is not a slur on the model — interpolating a high-dimensional video distribution well is a staggering achievement — but it is the precise reason it breaks on the rare and the counterfactual, and the precise reason scale (more memorized cases, finer interpolation) reduces the *frequency* of failure without changing its *nature*. That frequency-not-nature distinction is the single most important thing to hold onto about the whole thesis.

This reframes the whole debate cleanly. The word "world model" is doing two jobs. Under a *weak reading* — "a model that has internalized many statistical regularities of how the visual world behaves" — Sora absolutely qualifies, and the evidence (coherence, approximate 3D, frequent permanence) is real. Under a *strong reading* — "a causal simulator with explicit state that you could substitute for a physics engine or a game engine to generate ground-truth experience" — Sora clearly does not qualify, and the failure cases prove it. The report's rhetorical move is to demonstrate the weak reading and let the reader infer the strong one. That is the precise location of the hype.

### Why scaling does not obviously close the gap

The natural rebuttal is: granted, current Sora is statistical, but scaling has surprised us before (large language models acquired capabilities nobody predicted), so maybe enough scale *does* induce causal structure. This is not crazy, and I want to give it its due. There is a real argument that to predict the next frame *well enough* across a sufficiently diverse distribution, a model may be forced to internalize something functionally close to the generating dynamics — a "the best way to predict the data is to model the process that made it" argument. That is genuinely the most interesting open question in the field, and it is the [world-models](/blog/machine-learning/video-generation/video-models-as-world-models) frontier.

But there are two strong reasons for doubt. First, the identifiability result above is not a "we haven't scaled enough" problem; it is a *structural* one. No amount of purely observational data identifies the causal model, so scale over passive video cannot, in the limit, close the gap by itself — you need interventional data (action-conditioned video, where the model sees the effect of *doing* things) or an explicit physics inductive bias. This is exactly why the interactive line — [Genie, GameNGen, and action-conditioned models](/blog/machine-learning/video-generation/video-models-as-world-models) — is the more promising route to genuine world models than scaling passive text-to-video: those models are trained on the consequences of actions, which is the data type causality actually requires. Second, the empirical scaling evidence we have shows coherence and aesthetic quality improving smoothly with compute, but there is no published evidence of a *phase transition into physical correctness* — no scale at which conservation laws suddenly start holding. Sora 2, trained with more everything, has *better* physics (section 7) but still breaks it; the improvement is gradual, which is what you expect from a statistical model getting more data coverage, not from a causal model switching on.

So my read, stated plainly: the thesis is *half right in a way that matters*. Video models are real, useful, increasingly powerful learners of visual-world statistics, and that is genuinely valuable. They are not, and scaling passive video generation alone will likely not make them, causal physical simulators. The honest version of the thesis is "video models are powerful world *statistical* models, and action-conditioned video models are a promising path toward world *causal* models." Drop the conflation and the claim becomes both true and useful.

## 6. The recipe in runnable code

Sora is closed, so to make any of this concrete and runnable we use an open stand-in that follows the *same* recipe — a causal 3D-VAE plus a spacetime-patch DiT plus iterative sampling. [CogVideoX](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox) is the cleanest public model that matches Sora's architecture closely, and it ships in 🤗 `diffusers`. Running it is the best way to build intuition for what Sora's transformer is doing under each flag.

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
# the two flags that decide whether this fits in 24 GB:
pipe.enable_model_cpu_offload()     # stream weights on/off the GPU
pipe.vae.enable_tiling()            # tile the 3D-VAE decode (the real VRAM wall)

prompt = (
    "A glass of water falls off a wooden table and shatters on a tile floor, "
    "water splashing outward, captured in slow motion, cinematic lighting."
)
video = pipe(
    prompt=prompt,
    num_frames=49,            # CogVideoX-5b native length (~6 s at 8 fps latent)
    num_inference_steps=50,   # denoising steps; quality vs latency lever
    guidance_scale=6.0,       # classifier-free guidance strength
    generator=torch.Generator().manual_seed(0),
).frames[0]

export_to_video(video, "shatter.mp4", fps=8)
```

Two things in this snippet matter for the thesis. First, `vae.enable_tiling()`: the [3D-VAE decode is the memory wall](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), not the transformer — the same lesson Sora's "video compression network" embodies, and the reason length is bounded by the VAE far more than by the DiT. Second, run the falling-glass prompt and watch what happens: you will get a coherent, often beautiful clip whose physics is wrong in exactly the ways section 4 predicts — water volume not conserved, shards behaving oddly. That failure is not CogVideoX being weak; it is the *same statistical-not-causal limitation* that section 5 proved is structural to the recipe. Sora is a much bigger model trained on much more data, so it fails *less often* — but the failure mode is identical because the recipe is identical.

Now the conceptual variable-length DiT sampling sketch that mirrors Sora's "denoise a packed sequence of any length" behavior. This is illustrative PyTorch, not a real model, to show the *shape* of the computation.

```python
import torch

@torch.no_grad()
def sample_video_dit(dit, vae, text_emb, latent_shape, steps=50, cfg=6.0):
    """Mirror the Sora-style loop: denoise a packed spacetime latent of ANY
    shape, then decode. latent_shape = (C, T', H', W') — change it freely and
    the same model handles a different duration/resolution (variable-size)."""
    C, Tt, Hh, Ww = latent_shape
    x = torch.randn(1, C, Tt, Hh, Ww)             # noise in latent space
    null_emb = torch.zeros_like(text_emb)         # for classifier-free guidance
    sigmas = torch.linspace(1.0, 0.0, steps + 1)  # a simple noise schedule

    for i in range(steps):
        t = sigmas[i].expand(1)
        # patchify happens INSIDE the dit; the dit accepts a 5D latent of any
        # T',H',W' and internally flattens to a token sequence of length L.
        eps_cond = dit(x, t, text_emb)            # conditional prediction
        eps_uncond = dit(x, t, null_emb)          # unconditional prediction
        eps = eps_uncond + cfg * (eps_cond - eps_uncond)   # CFG combine
        # one Euler step of the reverse process toward less noise
        x = x + (sigmas[i + 1] - sigmas[i]) * eps
    return vae.decode(x)                           # latent video -> frames
```

The load-bearing comment is the one inside the loop: `latent_shape` is a free parameter, and the same `dit` denoises a $T'=12$ vertical clip or a $T'=450$ widescreen clip without any architectural change, because both are token sequences. That single property — variable-length denoising of packed spacetime patches — is the technical core of what made Sora's "any duration, any aspect ratio" demos possible. Everything else in the loop is the standard [classifier-free-guided](/blog/machine-learning/image-generation/classifier-free-guidance) diffusion sampling you already know from the image series; the video-specific part is entirely in the shape of `x`.

#### Worked example: VRAM and the VAE wall on a 4090

Run CogVideoX-5b on an RTX 4090 (24 GB) for a 49-frame, $720 \times 480$ clip. With `enable_model_cpu_offload()` and `vae.enable_tiling()`, peak VRAM lands around 12–18 GB and a 50-step generation takes on the order of a few minutes — both approximate and hardware-dependent. Turn *off* VAE tiling and the decode step alone tries to materialize the full $49 \times 480 \times 720 \times 3$ output tensor through the decoder's activations and OOMs on the 4090 — the [decode, not the denoiser, hits the wall first](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression). This is the practical face of why Sora's max length is one minute and not ten: the VAE's trained clip length and decode cost, not the transformer's cleverness, set the ceiling. The "world simulator" can only simulate as long a window as its autoencoder can decode.

## 7. What Sora 2 changed

Sora 2, which OpenAI positioned as a major step up, is the natural test of the scaling thesis: if "more scale yields world understanding," Sora 2 should show it. What actually changed is informative precisely because it is *not* a new architecture — it is the same spacetime-patch diffusion-transformer core, scaled and refined, with the gains spent on three axes: physical consistency, audio, and controllability.

![Timeline from the 2024 Sora technical report through the public release to Sora 2 with synchronized audio and improved physics](/imgs/blogs/sora-and-the-world-simulator-thesis-6.png)

**Better physical consistency, not physics.** OpenAI's framing for Sora 2 emphasizes more physically plausible motion and fewer of the egregious conservation and trajectory errors that defined Sora 1's failure reel. This is real and visible — Sora 2 clips break physics *less often*. But note carefully what kind of improvement this is. It is the gradual improvement section 5 predicts from a statistical model with more data coverage and better training, not the phase transition into causal correctness the strong thesis would predict. Sora 2 still fails on rare interactions, still has no explicit state, and still produces coherent-but-impossible clips under stress. The needle moved on *frequency*, not on *kind*. That is exactly the signature of "more of the same statistical model," and it is the single most important data point for evaluating the thesis: the biggest available scaling step improved physics quantitatively but did not change its nature.

**Synchronized audio.** Sora 2 generates audio jointly with video — speech, sound effects, ambient sound, roughly synchronized to the visuals. This is a genuine capability addition and connects to the [joint audio-video frontier](/blog/machine-learning/video-generation/audio-and-joint-av-generation) the series covers. Architecturally it means the model now generates in a joint audio-visual latent space rather than video alone. It is impressive product engineering and it makes the output far more useful, but it is orthogonal to the world-model question — synchronized audio is a *multimodal output* capability, not evidence of causal world understanding.

**Controllability.** Sora 2 added meaningful control: the ability to insert a consistent character ("cameos"), to remix and edit existing clips, and tighter prompt adherence. This is what turns Sora from a demo into a tool, and it matters commercially. From the thesis's standpoint, controllability is a *conditioning* improvement — better steering of the same generative process — not a change in what the process understands.

![Matrix comparing Sora 1 and Sora 2 across core recipe audio physics and controllability](/imgs/blogs/sora-and-the-world-simulator-thesis-8.png)

The verdict on Sora 2 for the thesis is therefore clarifying. It is a better video model in every practical sense — more coherent, audio-enabled, controllable, with fewer physics errors. And it is the same kind of thing as Sora 1: a powerful statistical model of audiovisual data. If you believed the strong world-simulator thesis, Sora 2 is mild evidence *against* it, because the largest scaling step the public has seen bought gradual quality, not a qualitative jump to physical correctness. If you held the weak thesis — "these are excellent learners of visual-world statistics" — Sora 2 confirms it beautifully. Read either way, Sora 2 says: scale the recipe, get a better statistical model; do not expect the statistical model to become a physics engine by getting bigger.

## 8. Case studies and real numbers

Let me ground the discussion in named, checkable facts and clearly-flagged estimates, because the thesis lives or dies on evidence, not vibes. I will mark each as **reported** (stated in OpenAI's materials or a paper) or **estimate** (my defensible order-of-magnitude).

**The compute-scaling figure (reported, qualitative).** The Sora report's most-cited piece of evidence is the base / 4x / 32x compute comparison on a fixed prompt, showing dramatically better coherence and detail at higher compute. This is reported and real, but it is *qualitative* — no FVD, no axis, no metric. It demonstrates that coherence scales with compute. It does not demonstrate that physics scales with compute, which is the claim that matters, and the report does not provide a physics-versus-compute curve.

**Maximum length (reported).** OpenAI states Sora can generate up to one minute of video. This is a real, checkable number, and as section 6's worked example argued, it is set by the VAE's decode budget and trained clip length far more than by the transformer.

**Parameter count and training data (estimate, undisclosed).** OpenAI published neither. Reasonable community estimates put Sora in the multi-billion-parameter DiT range (single-digit to low-double-digit billions), trained on a very large, undisclosed video corpus with heavy re-captioning. Treat any specific number you see for these as speculation — I am not going to invent one.

**Re-captioning (reported, borrowed from DALL-E 3).** Sora uses a re-captioning technique — train a captioner, use it to generate detailed text descriptions for the training videos, train on those — directly analogous to the method in the DALL-E 3 report. At inference, user prompts are expanded by a language model into the detailed style the model was trained on. This is reported and it is a real part of why prompt adherence is strong. It is also a reminder that a lot of Sora's "understanding" lives in the *text encoder and captioner*, not in an emergent physics model.

**Open-model comparators (reported, from papers).** Because Sora's numbers are hidden, the open models are where we get hard measurements that follow the *same recipe*. The table below collects published-or-defensible figures for the open frontier so you can see the recipe's actual cost-quality envelope; the Sora row is qualitative by necessity.

| Model | Params (est.) | Recipe | Max length / res | Measured quality | VRAM to run | Source |
|---|---|---|---|---|---|---|
| Sora (2024) | multi-B (undisclosed) | 3D-VAE + spacetime DiT | ~60 s / up to 1080p | qualitative only | closed | OpenAI report (reported length; params estimate) |
| Sora 2 (2025) | larger (undisclosed) | same + joint audio | longer / HD + audio | qualitative only | closed | OpenAI (reported capabilities) |
| CogVideoX-5B | ~5B | 3D-VAE + DiT + FM | ~6 s / 720×480 | strong VBench | ~12–18 GB (offload+tiling) | CogVideoX report |
| HunyuanVideo | ~13B | 3D-VAE + DiT + FM | ~5 s / 720p | top-tier VBench | 45–60 GB+ (tiling helps) | HunyuanVideo report |
| Wan 2.x (14B) | ~14B | 3D-VAE + DiT + FM | ~5 s / 720p+ | top-tier open | high (offload needed) | Wan report |

Every numeric cell that is not "undisclosed" or "qualitative only" comes from the respective model's report or is a clearly-flagged estimate; the Sora rows are deliberately sparse because OpenAI did not publish the numbers, and I will not fabricate them. The point of the table is the *pattern*: the open models reproduce Sora's recipe at known parameter counts and known quality, which tells you Sora's recipe is not magic — it is this recipe with more scale and better data. The [open frontier post](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox) goes deep on these numbers.

**The "Minecraft" digital-world demo (reported).** The report's most provocative single result is Sora rendering a Minecraft-like game, controlling a player and rendering the world coherently, with no explicit game engine. This is real and genuinely striking, and it is the strongest single piece of evidence for *some* form of learned world dynamics. But read it carefully: it shows Sora has learned the *visual statistics* of Minecraft footage well enough to continue them plausibly, including plausible player motion. It does not show a consistent game state — blocks placed do not reliably persist, the world is not editable in a state-consistent way, and it drifts. It is the weak-reading world model rendering a digital world, not a playable game-engine substitute. The line from here to a *genuine* playable model runs through the [action-conditioned world-models post](/blog/machine-learning/video-generation/video-models-as-world-models) (Genie, GameNGen), where action-conditioning supplies the interventional data that passive Sora lacks.

**Why the open comparators matter so much for reading Sora.** It is worth dwelling on the table above, because the existence of the open models is the single best epistemic tool we have for evaluating a closed one. When OpenAI says "scale this recipe and capabilities emerge," we cannot inspect Sora to check — but we *can* inspect CogVideoX, HunyuanVideo, and Wan, which implement the same causal-3D-VAE-plus-spacetime-DiT-plus-flow-matching recipe at *known* parameter counts and *published* benchmarks, and we can watch their capabilities scale across the family. What we find is exactly the deflationary story: as these open models grow from a few billion to fourteen billion parameters and train on more data, their coherence, aesthetic quality, and VBench scores climb smoothly, their physics gets gradually better, and — critically — none of them crosses into reliable conservation or causal correctness. They fail the falling-glass test the same way Sora does, scaled down. The open frontier is, in effect, a controlled experiment on the Sora thesis that OpenAI's closed report cannot provide, and it returns a clear verdict: the recipe scales coherence, not causation. Any time you are tempted to read magic into a Sora demo, run the equivalent prompt on a 14B open model and watch the same recipe produce the same kind of beautiful, physically-confused output — it is the fastest cure for the world-simulator spell.

**A note on what is genuinely unprecedented.** None of this deflation should obscure what Sora actually achieved, because it was real and large. Before Sora, the public state of the art in early 2024 was clips of a few seconds with visible flicker, weak motion, and identity drift; Sora demonstrated *one minute* of coherent, high-resolution, strongly-moving video with stable identity and plausible camera work, and it did so by committing to the spacetime-patch DiT recipe at a scale and data quality nobody else had. That is a genuine leap, and it reset the field's expectations overnight — every model in the table above is, in part, a response to Sora's demonstration that the recipe scales. The criticism in this post is aimed squarely at the *world-simulator framing*, not at the system: Sora is an extraordinary statistical video model and a milestone in the field. It is just not a physics engine, and the two should not be confused.

## 9. The implications if the thesis is even partly true

Suppose the *weak* thesis is right — video models are excellent learners of visual-world statistics, getting better with scale — and suppose, more speculatively, that action-conditioned versions partly cross into causal territory. What follows? This is the part of the thesis that is genuinely exciting and worth taking seriously even while we deflate the strong version.

The headline implication is **generated experience for agents and robotics**. If a video model can roll out plausible futures of a scene conditioned on actions, then an agent can *anticipate* the consequences of its actions and plan against them, or be trained on synthetic rollouts without ever touching the real world. This is the dream behind learned simulators in reinforcement learning, and it is why DeepMind's [Genie line](/blog/machine-learning/video-generation/video-models-as-world-models) is so significant: Genie 2 generates playable, action-conditioned environments, which is the data type that *does* support causal learning, unlike passive Sora. A robot that learns a manipulation policy in a learned video world model, then transfers it to hardware, would be transformative — and partial versions of this already work in narrow domains.

But the honesty discipline applies here most of all, and it is the subject of the dedicated post on [the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation). For generated experience to be *useful* for training agents, the generated dynamics have to be *correct enough* in the dimensions the agent cares about. A policy trained in a world model that violates conservation of mass will learn to exploit those violations — it will find the model's physics bugs and optimize against them, producing a policy that works beautifully in the dream and fails in reality. This is the "hallucinated reward" failure mode of model-based RL, and it is worse with a video model than a hand-built simulator because the video model's errors are *correlated with rarity*: it is most wrong exactly in the unusual situations where careful planning matters most. So the implication is real but conditional: video world models are a promising substrate for agent training *to the exact extent that their causal dynamics are correct*, which today is "narrow domains and short horizons," and the research frontier is widening that envelope with action-conditioning and physics priors rather than with raw scale.

#### Worked example: the dream-exploit failure in model-based RL

Make the conditional concrete. Suppose you train a robot arm to pour water from a jug into a cup, using a learned video world model as the simulator: the policy proposes an action, the video model rolls out the predicted result, and a reward function scores "water in cup." Section 4 told us the video model does not conserve liquid volume. So during training, the policy will discover — because optimizers always discover the easiest path to reward — that certain pouring motions cause the video model to *hallucinate extra water appearing in the cup*, scoring high reward for free. The policy converges to a motion that exploits the model's conservation bug, looks great in the dream, and pours water all over the table on the real robot. This is not hypothetical; it is the well-documented "model exploitation" failure of model-based reinforcement learning, and it is *worse* with a learned video model than with a hand-built physics simulator, because the video model's errors are densest exactly in the unusual, precise situations that careful policies seek out. The lesson is sharp: a learned world model is only as trustworthy for planning as its dynamics are correct *in the region the policy explores*, and that region is adversarially chosen to find the model's weakest spots. Validating the dynamics, not admiring the renders, is the whole job.

The second implication is for **the video-generation field itself**. If coherence and approximate 3D consistency reliably emerge from scale (the well-supported part of the thesis), then a lot of the explicit machinery the field built — separate camera-control modules, explicit 3D-aware architectures, hand-designed temporal-consistency losses — may be partially subsumed by scale, the way many NLP-specific architectures were subsumed by scaled transformers. That is a real strategic implication for anyone building video models: bet on the scalable recipe and spend engineering on data and the VAE, not on bespoke temporal gadgets. The [series capstone](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook) takes exactly this position for practitioners.

The third implication is **epistemic, and it is about how we evaluate claims in this field at all**. The Sora episode is a case study in how a precise weak claim ("video models learn rich visual-world statistics that scale with compute") gets communicated as a vague strong claim ("video models are world simulators") and is then defended with evidence that only supports the weak version. The defense against this is not skepticism for its own sake; it is *axis discipline*. Whenever someone claims a model "understands" something, ask which axis the evidence is on — appearance or dynamics, observation or intervention, frequency or kind — and whether the demo could distinguish the strong claim from the weak one. A coherent clip cannot distinguish them; only a stress test on rare, counterfactual, or conservation-critical cases can. This discipline is what the series' [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) operationalizes, and it is the most transferable thing to take from the whole Sora debate: the gap between what a system *does* and what we *say it understands* is exactly the gap between its observational competence and its causal competence, and the two are routinely conflated in exactly the way the world-simulator framing conflates them.

## 10. A protocol for stress-testing the thesis

The cleanest way to hold yourself honest about any "world model" claim is to have a repeatable protocol that targets the *dynamics* axis, not the appearance axis. Here is the one I use, framed as an engineering procedure you can run on Sora, any open model, or whatever ships next. The whole design principle is: a beautiful clip is not evidence, so build prompts that are easy to render coherently and hard to render *correctly*, then score correctness explicitly.

Start by choosing prompts whose physics is unambiguous and checkable. Good families: **conservation tests** (pour a measured amount of liquid between containers; count the apples before and after an occlusion), **trajectory tests** (throw or drop an object and check the arc against a parabola), **state-persistence tests** (cut, break, or mark something and check the change persists), and **causal-ordering tests** (events that have a strict before/after, like a switch and a light). Avoid prompts that are visually busy or aesthetically loaded — they reward the appearance axis and hide the dynamics failures.

Then score each generation on two independent rubrics. The coherence rubric is the standard one — subject consistency, motion smoothness, no flicker — and frontier models will pass it easily. The *physical-correctness* rubric is the one that matters: did volume conserve, did the count survive occlusion, did the trajectory follow gravity, did the state change persist, did cause precede effect. Score these as hard pass/fail, not impressions, and report the *fraction* of generations that pass each, across a fixed seed set, because a single cherry-picked clip tells you nothing about the distribution. The series' [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) explains why fixed seeds, a fixed sample set, and reporting the full distribution rather than the best clip are non-negotiable for any honest video evaluation.

```python
# A correctness-probe harness skeleton: generate N seeds per physics prompt,
# then hand (or auto) score on a hard physical-correctness rubric.
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

physics_prompts = {
    "conservation": "Exactly five red apples on a white counter; a hand sweeps "
                    "across briefly hiding three apples, then withdraws.",
    "trajectory":   "A basketball thrown across an empty gym, arcing and bouncing once.",
    "persistence":  "A knife cuts a tomato cleanly in half on a cutting board.",
}

for name, prompt in physics_prompts.items():
    for seed in range(8):                     # fixed seed set, report the fraction
        frames = pipe(
            prompt=prompt, num_frames=49, num_inference_steps=50,
            guidance_scale=6.0,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
        export_to_video(frames, f"probe_{name}_{seed}.mp4", fps=8)
        # then score probe_{name}_{seed}.mp4 on the HARD rubric:
        #   conservation -> count apples after withdrawal == 5 ?
        #   trajectory   -> ball follows a single parabola, one bounce ?
        #   persistence  -> tomato stays cut for the rest of the clip ?
```

Run this and you will reproduce, on your own hardware, the central finding of this post: coherence passes near-universally, physical correctness passes *some* of the time and degrades exactly where the kit predicts. The discipline is to report that second number out loud. When a vendor shows you a flawless demo, the question that cuts through everything is "what fraction of your seeds pass a hard physical-correctness rubric on conservation and trajectory prompts," and the answer — which is never 100% — tells you precisely where on the spectrum from renderer to simulator the model actually sits.

## 11. When to invoke the world-simulator framing (and when not to)

A decisive section, because the framing has real consequences for how you build and how you communicate.

**Reach for the "world model" framing when** you mean the *weak, statistical* reading and you are precise about it: "Sora has learned strong statistical regularities of visual-world dynamics" is true, useful, and defensible. Use it to justify *coherence and quality* claims — those scale, and the evidence is solid. Use it to motivate the [action-conditioned research direction](/blog/machine-learning/video-generation/video-models-as-world-models) — passive video models are the foundation that interactive world models build on.

**Do not reach for it when** you mean the *strong, causal* reading without the interventional data and physics priors that the reading requires. Do not claim Sora "understands physics" — the failure reel disproves it, and section 5 explains why scaling passive video cannot fix it structurally. Do not promise it as a drop-in physics engine or game engine — the lack of explicit, consistent state means it cannot maintain a ground-truth world. Do not train an agent on its rollouts in a high-stakes domain and trust the result without verifying the dynamics the agent actually exploited. And do not let a beautiful coherent demo stand in as evidence of world modeling — coherence and physical correctness are orthogonal axes (section 4), and the demos overwhelmingly showcase the first.

**The practitioner's version:** if you need *plausible, controllable, beautiful* video, Sora and its open cousins are spectacular and the recipe is exactly right — use it, and spend your effort on data, captioning, and the VAE. If you need *physically correct* dynamics — a simulator you can trust for planning, robotics, or science — a learned passive video model is the wrong tool today; reach for a real physics engine, or an action-conditioned world model in a narrow validated domain, and treat anything broader as research, not production. The dividing line is whether you need pixels that *look* right or dynamics that *are* right, and Sora is firmly, impressively, on the "look right" side.

## Key takeaways

- **Sora is the standard stack scaled.** A causal 3D-VAE, spacetime-patch tokenization, a large diffusion transformer, iterative sampling — the same recipe the open frontier reproduced. The novelty budget is spent on scale, data, and re-captioning, not on a secret component.
- **Variable-size training via patch packing is the one quietly elegant idea.** Treating video as a packed token sequence lets one model train on native durations, resolutions, and aspect ratios, which mechanistically (not mysteriously) improves framing and flexibility.
- **The emergent capabilities are real but graded.** Long-range coherence holds up strongly; 3D consistency and object permanence are genuine *tendencies*, not enforced *properties*, and they degrade exactly where the training distribution is thin.
- **Object permanence is an attention regularity, not state.** There is no persistent object variable — only learned correlations that mimic state and fail under long occlusion, scene complexity, and rare interactions.
- **Coherence and physical correctness are orthogonal.** Sora can render a crisp, frame-consistent clip of something physically impossible. That orthogonality is the clearest evidence that it is a renderer of plausible pixels, not a simulator.
- **The statistical-vs-causal gap is structural, not a scale gap.** A diffusion model learns $p_\text{data}(x)$; causal dynamics are not identifiable from passive observation alone. Scaling passive video cannot, by itself, close the gap — interventional (action-conditioned) data or physics priors are required.
- **Sora 2 confirms the deflationary read.** The biggest public scaling step improved physics *quantitatively* (fewer errors), added audio and controllability, but produced no phase transition into causal correctness — exactly what a bigger statistical model predicts.
- **The exciting implication is conditional.** Video world models are a promising substrate for agent and robotics training *to the exact extent their dynamics are correct* — narrow and short-horizon today — and the path widens via action-conditioning, not raw scale.

## Further reading

- **Brooks, Peebles, et al., "Video generation models as world simulators" (OpenAI, 2024)** — the Sora technical report. Read the "Discussion" section on limitations as carefully as the demos; it is the most honest paragraph in the document.
- **Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023)** — the transformer-diffusion backbone Sora builds on; pair with the image series' [diffusion transformers post](/blog/machine-learning/image-generation/diffusion-transformers-dit).
- **Bruce et al., "Genie: Generative Interactive Environments" (DeepMind, 2024)** and the Genie 2 release — action-conditioned world models, the data type that passive Sora lacks and that genuine causal world modeling requires.
- **Pearl, "Causality" (2009)** — the ladder of causation and identifiability, the formal backbone of section 5's statistical-versus-causal argument.
- **CogVideoX, HunyuanVideo, and Wan technical reports** — the open reproductions of Sora's recipe with published parameter counts and VBench numbers; the hard data the closed model withholds.
- **Within this series:** the foundation [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard); the architecture in [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers) and [spatiotemporal attention patterns](/blog/machine-learning/video-generation/spatiotemporal-attention-patterns); honest measurement in [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation); the forward arc in [video models as world models](/blog/machine-learning/video-generation/video-models-as-world-models) and [the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation); and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
