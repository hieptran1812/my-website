---
title: "Evaluating and Red-Teaming Video Generation: Benchmarks, Safety, and Provenance"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build an honest eval harness for a video model you ship — VBench-2.0 in depth, text-video alignment, human arenas — then the safety half: the misuse surface, data filtering, watermarking, C2PA provenance, and why generated-video detection keeps losing."
tags:
  [
    "video-generation",
    "diffusion-models",
    "evaluation",
    "vbench",
    "safety",
    "watermarking",
    "c2pa",
    "provenance",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/evaluating-and-red-teaming-video-generation-1.png"
---

You have a checkpoint that scores beautifully. FVD is down forty points from last week, every VBench dimension on your dashboard is green, the CLIP-score says it follows prompts, and the leaderboard would rank it near the top of the open models. So you write the release note, you draft the model card, and then someone on the trust-and-safety team asks the one question that turns the whole launch into a different project: *can it make a convincing fake video of a real person saying something they never said, and if it does, can anyone downstream ever tell it was generated?* You do not have an answer. Nobody on the eval team measured that, because it is not a quality metric. It is a safety property, and it lives in a part of the release process that almost no benchmark touches.

This is the gap this post closes. Shipping a video model is not one pipeline, it is two pipelines in series. The first is the eval pipeline you already half-know — and we will go deeper into it than [the metrics foundations post](/blog/machine-learning/video-generation/the-metrics-of-video-generation), all the way into VBench-2.0's dimension tree, VLM-as-judge alignment, and the human arena that is the only real arbiter. The second is the safety and provenance pipeline that decides whether the weights ever leave the building: the misuse surface (deepfakes, fraud, non-consensual imagery, misinformation), the mitigations (data filtering, prompt and output classifiers, identity protections), and the provenance stack (SynthID-style watermarks across frames, C2PA Content Credentials, and the brutal robustness reality that re-encoding and frame sampling impose). By the end you will be able to build an honest eval harness *and* reason about the safety stack like an engineer rather than a press release.

![A vertical stack figure showing the release pipeline running from a quality basket through targeted probes and a human spot-check into safety classifiers, frame watermarking, and a signed C2PA manifest](/imgs/blogs/evaluating-and-red-teaming-video-generation-1.png)

A note on framing before we start, because the back half of this post is about misuse. We describe the misuse surface *only* to motivate and pressure-test mitigations — the whole point is detection and defense. There is nothing here that helps anyone make a harmful video; there is a great deal here about why the defenses are weaker than you hope and what to do about it. The recurring tension of this whole series is **coherence × motion × length × cost**; this post adds the two axes that production forces on top of those: **honesty** (does your number mean what you think it means) and **safety** (what happens when the output meets the world). Hold the running example the whole way through: a model that renders a 5-second, 24 fps, 720p clip of *a dog running across a field* — the same thread from the metrics post — but now we also ask it to render *a specific named person*, and that single change is where evaluation ends and red-teaming begins.

## 1. The two pipelines, and why they are different jobs

The mistake that produces unshippable models is treating safety as a quality dimension — one more bar on the VBench chart. It is not. Quality evaluation and safety evaluation answer structurally different questions, and they fail in opposite directions.

Quality evaluation is a **ranking-and-grading** problem over the *typical* output. You sample a few hundred prompts, score the results, and ask whether checkpoint A is, on average, better than checkpoint B. The relevant statistic is central tendency. A metric that is right on average and noisy per-clip is usable, because you average the noise away over the prompt set. This is the world of FVD and VBench, and it is the world [the metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) lives in.

Safety evaluation is a **worst-case** problem over the *tail* of the output distribution. You do not care whether the model is safe on average — you care whether there exists *any* prompt, possibly adversarially constructed, that produces a harmful output. The relevant statistic is the maximum over an adversary's search, not the mean over a benign prompt set. A model that refuses 99.9% of harmful prompts and complies with 0.1% is, from a safety standpoint, *not 99.9% safe* — it is exploitable, because an attacker runs the 0.1% a thousand times. Averaging is exactly the wrong operation. This asymmetry is the single most important idea in the back half of this post, and it explains why a green dashboard tells you nothing about whether you should ship.

There is a second structural difference: who you are protecting against. Quality evaluation protects against *your own optimization pressure* — the danger is that you Goodhart your metric and ship the still-dog. Safety evaluation protects against an *external adversary* who actively searches for the model's worst behavior and who adapts as you patch. The first adversary is your own gradient descent; the second is a motivated human (or a script they wrote). You red-team for the second the way a security team does: assume the attacker is smart, assume they will find the seam you did not test, and build defense in depth so that no single failure is catastrophic.

Both pipelines share one discipline, though, and it is worth stating up front because everything below is an application of it: **a measurement is only as good as the protocol it is measured under, and a defense is only as good as the attack it has survived.** An FVD without a fixed sample set and clip length is a number with no meaning. A watermark that has never been re-encoded is a watermark that has never been tested. The honesty axis and the safety axis are the same demand — *say what you actually checked, and check the thing that actually matters* — pointed at two different failure modes.

## 2. Reading VBench as a tree, not a number

The metrics post introduced VBench's refusal to give you one number. Here we go deeper, because *how you read the dimensions together* is the entire skill, and the structure is a tree, not a flat list.

VBench (Huang et al., 2024) and its successor VBench-2.0 decompose video quality into sixteen-plus dimensions, but those dimensions are not independent peers — they fall into families that you must read at the family level first and the leaf level second. There are three families. The first is **video quality**, which is frame-level and temporal-local: aesthetic quality (does each frame look good to a trained aesthetic predictor), imaging quality (is it free of low-level distortion, blur, noise), and temporal flickering (do adjacent frames agree in low-level detail). The second is **semantic fidelity**, which asks whether the video contains what the prompt asked for: object class (is the dog actually a dog), spatial relationship (is the ball *on* the table not *under* it), color, scene, and overall consistency between the prompt and the rendered content. The third — and this is the family that breaks naive readers — is the **motion-versus-stability pair**: subject consistency and background consistency on one side, dynamic degree and motion smoothness on the other.

![A taxonomy tree figure grouping the VBench dimensions into a video-quality family, a semantic-fidelity family, and a motion-versus-stability pair that must be read together](/imgs/blogs/evaluating-and-red-teaming-video-generation-2.png)

Read that tree top-down and the trap from the metrics post becomes structural rather than anecdotal. **Subject consistency** measures whether the main subject's appearance is stable across frames — VBench computes it as the mean DINO feature cosine similarity between consecutive frames and between each frame and the first. **Dynamic degree** measures whether anything moves at all — VBench estimates it with an optical-flow magnitude (RAFT) and thresholds it into a binary "this clip has motion / does not." Here is the formal statement of why optimizing one alone is gameable. Let $f_t$ be the DINO feature of frame $t$. Subject consistency is, roughly,

$$
\text{SC} = \frac{1}{T-1}\sum_{t=2}^{T} \cos\big(f_t, f_{t-1}\big),
$$

and this is *maximized* — it hits exactly $1.0$ — by the degenerate video where every frame is identical, $f_t = f_{t-1}$ for all $t$. Dynamic degree, meanwhile, is monotonically *increasing* in the optical-flow magnitude $\lVert v_t \rVert$ between frames, and the identical-frame video has $v_t = 0$ everywhere, so its dynamic degree is the floor. The two metrics are, on the degenerate axis, in direct opposition: the still-dog clip wins subject consistency and loses dynamic degree, and the violently-jittering clip does the reverse. **Neither metric alone is a quality signal. Only the pair is.** A model is good when it holds subject consistency high *while* clearing a dynamic-degree floor — you must read the two columns side by side, and any leaderboard that lets a model trade one for the other is rewarding a gamed submission.

VBench-2.0 (released 2024–2025, mark this as the current frontier and approximate in exact dimension count) extends this with dimensions aimed squarely at the failures the first VBench could not see. The first VBench measured *fidelity and consistency* — does it look good and stay stable — but a model can ace those while violating physics, and as the open and closed models saturated the original dimensions the benchmark had to reach for harder properties. So VBench-2.0 adds **commonsense and physics** dimensions (does a dropped object fall, does a poured liquid behave like a liquid, do shadows track the light source), **identity and instance** consistency over longer clips (does *this specific* dog stay the same dog across 20 seconds, not just "a dog-shaped thing"), and finer-grained **interaction and motion-order** dimensions (do the events happen in the prompted sequence, do two objects interact plausibly). These are exactly the properties that are easy for a human to judge and hard for a feature-similarity metric to score, which is why VBench-2.0 leans more on VLM-based and model-based scoring for the new axes rather than pure DINO/optical-flow heuristics — and that shift inherits the judge-calibration caveats we hit in the alignment section. The philosophy is unchanged — refuse the single number, decompose into named axes, force the reader to confront the trade-offs — but the axes now reach directly into the physics failures covered in [physics and the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation), which is where a model that scores well on appearance but cannot model gravity gets caught. For a release, the practical move is to pick a *basket* of axes that matters for your use case and read each pair honestly, not to chase a weighted average that hides the trade-off — and if your use case is anything that implies physical realism (product shots, simulation, anything with falling, pouring, or collision), the VBench-2.0 physics axes belong in your gate, not your nice-to-have list.

#### Worked example: the still-dog submission, in VBench numbers

Take two checkpoints of our dog model on a fixed 200-prompt set. Checkpoint A (the one you want to ship) scores subject consistency 0.93, background consistency 0.95, dynamic degree 0.71 (71% of clips clear the motion threshold), motion smoothness 0.97, imaging quality 0.68. Checkpoint B (the still-dog) scores subject consistency 0.99, background consistency 0.99, dynamic degree 0.08, motion smoothness 0.99, imaging quality 0.74. If your leaderboard reports an *unweighted mean* of the consistency-and-quality dimensions, B wins — it is higher on four of five. But B's dynamic degree of 0.08 says **92% of its clips barely move**: it is a photo generator with a tremor. A is the correct ship. The lesson is not "weight dynamic degree higher" — any fixed weighting is gameable — it is "*set a dynamic-degree floor of, say, 0.5 as a gate, and only then rank on the rest.*" The floor turns a gameable trade into a hard constraint the still-dog cannot satisfy.

## 3. FVD, CMMD, and the per-frame metrics, briefly

The metrics post derived FVD in full; here is the compressed restatement you need to read the rest of this post, plus the two refinements production teams actually use.

**FVD** lifts FID to video by swapping the per-frame Inception backbone for a 3D-convolutional network — classically I3D (Inflated 3D ConvNet, Carreira and Zisserman, 2017), pretrained on Kinetics. You pass a few thousand real clips and a few thousand generated clips through I3D, grab a spatiotemporal feature vector per clip, fit a single multivariate Gaussian to each cloud, and report the Fréchet (2-Wasserstein) distance between the two Gaussians:

$$
\text{FVD}^2 = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\!\Big(\Sigma_r + \Sigma_g - 2\big(\Sigma_r \Sigma_g\big)^{1/2}\Big).
$$

Because I3D has a *temporal* receptive field, its features change when you shuffle the frames — which is exactly the property a per-frame metric lacks, and exactly why FVD can see flicker and jitter that frame-averaged FID cannot. The whole pipeline forks from the clip into a realism path (I3D → Gaussian → FVD) and, in parallel, an alignment path we will get to next.

![A directed dataflow figure where a clip forks into an I3D feature path producing an FVD score and a parallel CLIP-encoder path producing a per-frame text-alignment score](/imgs/blogs/evaluating-and-red-teaming-video-generation-3.png)

Two things every honest FVD report must state, because the metric moves with knobs that have nothing to do with quality. First, **sample size**: the Fréchet distance has a positive finite-sample bias — too few clips *always* inflates FVD — so a number computed on 256 clips is not comparable to one computed on 2048, and you must fix $N$ (use at least 2048) and report it. The bias is not a vague tendency; it has a known source. The estimated covariance $\hat{\Sigma}$ from $N$ samples is noisy, and that noise enters the trace term $\operatorname{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ in a way that does *not* cancel — the matrix square root is a concave-ish function whose expectation under sampling noise is biased upward, so $\mathbb{E}[\widehat{\text{FVD}}] > \text{FVD}_\infty$ and the gap shrinks roughly like $O(1/N)$. The practical consequence is concrete: if you compute FVD on 512 clips for model A and 2048 clips for model B, A is *penalized purely for having fewer samples*, and you will rank them backwards. The fix is to fix $N$ across everything you compare and, if you want an absolute number, extrapolate FVD to infinite $N$ by computing it at several sample sizes and fitting the $1/N$ trend to its intercept — the same $\widehat{\text{FID}}_\infty$ trick from the image literature. Second, **clip length and backbone**: a 16-frame FVD and a 128-frame FVD are different metrics (the I3D temporal receptive field sees a different fraction of the motion), and an I3D FVD and a VideoMAE-FVD live on entirely different scales. Report all three (sample set, clip length, backbone) or the number is meaningless. FVD is a passable *ranker* of your own checkpoints under a fixed protocol and a useless *grader* across papers — and the single most common eval bug in published video work is treating a ranker as a grader, reading an absolute FVD of 180 as "good" when it only ever meant "lower than our last run on our protocol."

The image world has largely moved past FID toward **CMMD** (CLIP Maximum-Mean-Discrepancy, Jayasumana et al., 2024), which replaces the unjustified single-Gaussian assumption with a kernel two-sample test (MMD) on CLIP features and is far less biased at small sample sizes. The video analog — MMD on a video-feature backbone — is the natural successor to FVD for exactly the reasons CMMD beat FID: no Gaussian assumption, better small-sample behavior, and a backbone (CLIP-style) that tracks human judgment better than Kinetics-trained I3D. As of 2026 FVD is still the reported default in most papers, but if you are building a fresh harness, an MMD-based distribution metric is the better engineering choice; mark this as a still-settling area.

Here is the FVD-plus-CLIP-score sketch you would actually run, using the structure most teams build on top of `torchmetrics` and a custom I3D wrapper:

```python
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np

@torch.no_grad()
def i3d_features(clips, i3d):
    # clips: (N, T, C, H, W) in [-1, 1], i3d returns (N, D) spatiotemporal feats
    clips = clips.permute(0, 2, 1, 3, 4)  # I3D wants (N, C, T, H, W)
    return i3d(clips).flatten(1)  # (N, D)

def frechet_distance(mu1, sig1, mu2, sig2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # numerical guard: sqrtm of a near-singular product can go complex
    return float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))

def fvd(real_clips, fake_clips, i3d):
    fr = i3d_features(real_clips, i3d).cpu().numpy()  # demand N >= 2048
    fa = i3d_features(fake_clips, i3d).cpu().numpy()
    mu_r, sig_r = fr.mean(0), np.cov(fr, rowvar=False)
    mu_a, sig_a = fa.mean(0), np.cov(fa, rowvar=False)
    return frechet_distance(mu_r, sig_r, mu_a, sig_a)

@torch.no_grad()
def per_frame_clip_score(frames, prompt, clip_model, clip_proc):
    # frames: list of PIL images for ONE clip; returns mean cosine alignment
    inputs = clip_proc(text=[prompt], images=frames,
                       return_tensors="pt", padding=True).to(clip_model.device)
    out = clip_model(**inputs)
    img_emb = F.normalize(out.image_embeds, dim=-1)      # (T, d)
    txt_emb = F.normalize(out.text_embeds, dim=-1)       # (1, d)
    return (img_emb @ txt_emb.T).mean().item()           # mean over frames
```

The `per_frame_clip_score` is the metric we are about to show is structurally blind to time — keep it in view, because its blindness motivates the next section.

## 4. Text-video alignment: CLIP-score is a bag of frames

Ask the model for *"a dog runs left, stops, then jumps over a log."* A per-frame CLIP-score will happily report high alignment for a clip that contains a dog, a log, and some leftward motion *in no particular order* — including a clip where the dog jumps first and runs second, or never stops at all. The reason is the same permutation-invariance that sinks per-frame quality metrics: averaging a frame-wise cosine over time discards order. CLIP-score sees the *nouns* and roughly the *appearance*, but it is a **bag of frames** with respect to *events*. It cannot score "stops, then jumps" because "then" is a temporal relation and CLIP has no temporal axis.

![A two-column before-and-after figure contrasting a per-frame CLIP-score that averages over frames against a VLM-as-judge that samples frames and reasons about order and relations](/imgs/blogs/evaluating-and-red-teaming-video-generation-5.png)

The modern fix is **VLM-as-judge**: sample a handful of frames (say 8, evenly spaced, optionally laid out as a grid), hand them to a capable vision-language model — Claude, GPT-4o-class, or an open VLM like Qwen2-VL — with a rubric, and ask it to score temporal and relational fidelity. Because the VLM reasons over the *sequence* of sampled frames, it can catch "the dog never stops" and "the ball is under the table, not on it" in a way CLIP-score cannot. This is the same shift the LLM world made with [LLM-as-judge for text](/blog/machine-learning/large-language-model), now pointed at video, and it inherits the same caveats: the judge has biases (position bias, verbosity bias, self-preference if you judge with the same family that generated), and it is *not* a ground-truth oracle — it is a cheaper, noisier proxy for a human that you must calibrate against actual human labels before you trust it.

Here is a VLM-as-judge alignment scorer, written against the Claude API as the judge — it samples frames, builds a grid, and asks for a structured rubric score:

```python
import base64, io
import anthropic
from PIL import Image

def sample_grid(frames, n=8, cols=4):
    # frames: list of PIL images; return a single grid image of n sampled frames
    idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    picks = [frames[i].resize((256, 256)) for i in idx]
    rows = (n + cols - 1) // cols
    grid = Image.new("RGB", (cols * 256, rows * 256), "black")
    for k, im in enumerate(picks):
        grid.paste(im, ((k % cols) * 256, (k // cols) * 256))
    buf = io.BytesIO(); grid.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()

def vlm_align_score(frames, prompt, client):
    img_b64 = sample_grid(frames, n=8, cols=4)
    rubric = (
        "These 8 frames are sampled in time order (left-to-right, top-to-bottom) "
        f"from one generated video. The prompt was: '{prompt}'. "
        "Score 1-5 on: (a) object presence, (b) spatial relations, "
        "(c) temporal order of events, (d) overall faithfulness. "
        "Return strict JSON: {\"presence\":_, \"relations\":_, \"order\":_, "
        "\"faithful\":_, \"why\":\"...\"}."
    )
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/png", "data": img_b64}},
            {"type": "text", "text": rubric},
        ]}],
    )
    return msg.content[0].text  # parse the JSON; average over a prompt set
```

The honest way to use this: score your whole prompt set with the VLM, **but validate the VLM judge against ~100 human-labeled clips first** and report the agreement (e.g., Spearman correlation between VLM and human rank). A VLM judge that disagrees with humans is just an expensive random number. When it agrees, it is the cheapest scalable signal you have for event-level alignment — but it never replaces the human arena, which is next.

## 5. The human arena is the real arbiter

Every automatic metric is a proxy. The thing they are all proxies *for* is human preference: would a person, shown two clips, pick yours. So the ground truth of video quality is a **human pairwise arena**, and the discipline of running one well is what separates a credible eval from a vibe.

The arena format that has become standard — borrowed from the LLM Chatbot Arena and adapted for video by efforts like the Artificial Analysis video arena and similar — is **pairwise comparison with Elo (or Bradley-Terry) aggregation**. You take a fixed, frozen prompt set; for each prompt you generate a clip from model A and model B; you show a rater the two clips side by side with the prompt and ask "which is better?" (with a tie option). You collect thousands of such votes across many raters and many model pairs, and you fit a Bradley-Terry model: each model $i$ gets a latent skill $\theta_i$, and the probability that $i$ beats $j$ is the logistic

$$
P(i \succ j) = \frac{1}{1 + e^{-(\theta_i - \theta_j)}}.
$$

You fit the $\theta$'s by maximum likelihood over the vote record, convert to an Elo-style scale, and the ranking is the sorted $\theta$. This is robust in a way single-model scores are not: it never asks a human for an absolute number (which drifts between raters and days), only for a *relative* judgment (which is far more stable), and the pairwise structure means a model cannot win by being loud — it must be *preferred* head-to-head.

Three disciplines make an arena trustworthy. **Frozen prompt set**: if the prompts change, the Elo is not comparable across time, and worse, you can game the arena by overfitting to a known prompt distribution — so freeze it and keep a held-out private slice. **Blind and randomized**: the rater must not know which model produced which clip, and left/right position must be randomized, or position bias and brand bias corrupt the votes. **Enough votes for the confidence interval**: Elo has a standard error; a 15-point gap with a ±40-point CI is *noise*, and reporting it as a ranking is dishonest — you need enough votes that the CIs separate. The arena is expensive and slow, which is exactly why you do not run it on every checkpoint — you run automatic metrics in the inner loop and the arena on release candidates only.

Two subtleties separate a credible arena from a sloppy one. The first is **tie handling**. Many video comparisons are genuine ties — two clips a rater cannot distinguish — and how you treat ties changes the ranking. If you drop ties you inflate the apparent decisiveness of the model and overstate the gap; the honest move is to model ties explicitly (a Bradley-Terry-with-ties or Rao-Kupper formulation) so that "indistinguishable" is a recorded outcome, not a discarded one. A model that ties the leader 80% of the time and loses 20% is *not* the leader, and an arena that throws away ties will sometimes say it is. The second is **inter-rater agreement**. Human raters disagree, and if they disagree *randomly* your votes are noise that no amount of aggregation fixes; if they disagree *systematically* (one rater always prefers more motion, another always prefers stability) you have hidden the very motion-vs-stability trade-off inside your ground truth. Measure agreement (Krippendorff's alpha or simple pairwise concordance on a shared subset of comparisons), and if it is low, your prompts or your instructions are ambiguous — fix the protocol before you trust the Elo. An arena with unmeasured rater agreement is an arena reporting its raters' confusion as if it were the models' quality. These disciplines are tedious, and they are exactly what makes the difference between an arena you can stake a release on and a leaderboard screenshot.

#### Worked example: how many votes to separate two models

Suppose models A and B are genuinely close — A wins 55% of non-tie comparisons. How many votes do you need before the arena reliably says A > B? The standard error of an estimated win-rate $p$ over $n$ comparisons is $\sqrt{p(1-p)/n}$. For $p = 0.55$ that is $\sqrt{0.2475/n}$. To get a 95% CI (≈ ±1.96 SE) that excludes 0.50 — i.e., a half-width below 0.05 — you need $1.96\sqrt{0.2475/n} < 0.05$, which gives $n > 0.2475 \cdot (1.96/0.05)^2 \approx 380$ comparisons. So *just to distinguish a 55/45 model pair* you need on the order of 400 head-to-head votes, and a closer pair needs far more (a 52/48 pair needs roughly $0.25 \cdot (1.96/0.02)^2 \approx 2400$). This is why arenas need thousands of votes and why a "we beat them in a quick A/B" with twenty raters is statistically empty. Numbers approximate; the scaling ($n \propto 1/\text{gap}^2$) is exact.

## 6. Building an honest eval harness for a model you ship

Now assemble the pieces into the harness you actually run before a release. The shape is the same as the figure that opened this post: a **basket** of automatic metrics, a set of **targeted probes** for the specific failures you fear, and a **human spot-check** that the gaming cannot pass. The naive alternative — one leaderboard number — rewards the still-dog and ships broken models.

![A two-column before-and-after figure contrasting a single gameable leaderboard number against an honest harness that pairs a metric basket with targeted probes and a frozen human spot-check](/imgs/blogs/evaluating-and-red-teaming-video-generation-8.png)

The **basket** is FVD (or an MMD-based distribution metric) under a fixed protocol, plus the VBench dimensions read as pairs with a dynamic-degree floor as a hard gate, plus a VLM-as-judge alignment score validated against humans. No single number; a scorecard. The **targeted probes** are the part teams skip and regret — small, hand-built prompt sets that stress the specific failure modes of *your* model. For the dog model: an *identity-drift* probe (a 20-second clip, does the dog stay the same dog — covered for long video in [long video and autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout)); a *large-motion* probe (fast pan, does it tear); a *physics* probe (drop a ball, does it bounce or float); a *prompt-adherence* probe with compositional prompts CLIP-score cannot grade. The **human spot-check** is a frozen 30-to-50-prompt set you watch *yourself*, every release, because the one thing no metric catches is the failure you have not named yet.

Here is the harness sketch — a multi-dimension scorer that runs the basket and gates on the floor before ranking:

```python
from dataclasses import dataclass

@dataclass
class Scorecard:
    fvd: float
    subject_consistency: float
    dynamic_degree: float        # fraction of clips clearing the motion floor
    motion_smoothness: float
    vlm_align: float             # validated against human labels
    imaging_quality: float

def gate_and_rank(cards: dict[str, Scorecard],
                  dyn_floor=0.5, fvd_max=None):
    # 1. HARD GATE: a model that does not move is disqualified, period.
    eligible = {name: c for name, c in cards.items()
                if c.dynamic_degree >= dyn_floor
                and (fvd_max is None or c.fvd <= fvd_max)}
    disqualified = set(cards) - set(eligible)
    # 2. Rank survivors on a transparent, REPORTED weighting -- never hide it.
    def quality(c: Scorecard) -> float:
        return (0.30 * c.subject_consistency
                + 0.20 * c.motion_smoothness
                + 0.25 * c.vlm_align
                + 0.15 * c.imaging_quality
                - 0.10 * (c.fvd / 100.0))      # lower FVD is better
    ranked = sorted(eligible, key=lambda n: quality(eligible[n]), reverse=True)
    return ranked, disqualified
```

The dynamic-degree gate is doing the heavy lifting: it makes the still-dog *ineligible* rather than letting it trade motion for consistency. The weighting is reported, not hidden — anyone can see and contest it, which is the honesty discipline. And critically, this harness output is *advisory*: the release candidate it surfaces still goes through the human arena and the spot-check before it ships. The full eval-dimension-by-exploit picture is the matrix below — every axis captures something real and is gamed by something specific, and the basket exists to close each one's blind spot.

![A matrix figure mapping each evaluation axis to what it captures, the specific way it can be gamed, and the basket element that closes that blind spot](/imgs/blogs/evaluating-and-red-teaming-video-generation-4.png)

| Eval axis | What it captures | Gameable by | Closed in the basket by |
|---|---|---|---|
| FVD / MMD | Distribution realism vs reference | Matching the reference *domain* without quality; tiny $N$ | Fixed $N\geq 2048$, fixed clip length + backbone |
| Subject consistency | Frames agree over time | Freezing the subject (still-dog) | Hard dynamic-degree floor |
| Dynamic degree | Things actually move | Random jitter that is not real motion | Pair with motion smoothness |
| Motion smoothness | Motion is temporally coherent | Slow, tiny motion that is trivially smooth | Pair with dynamic degree |
| CLIP-score | Frame-level prompt match | One on-prompt frame; ignores event order | VLM-as-judge over sampled frames |
| VLM-as-judge | Event order + relations | Judge bias; same-family self-preference | Human-label calibration |
| Human arena | What people actually prefer | Cherry-picked prompts; too few votes | Frozen prompt set; CI on Elo |

That table is the quality half of the release. Now the safety half — which is not on any leaderboard, and which is the part that decides whether you ship at all.

## 7. The misuse surface, named so we can defend it

To build defenses you have to name the threats, the way a security team enumerates an attack surface. We do this in the abstract, motivated entirely by mitigation — none of this is a how-to, all of it is a what-to-defend. Video generation has a misuse surface that is *worse* than image generation along two specific axes, and naming why is the whole point.

The misuse categories, at the level a model card and a safety review must address them: **non-consensual intimate imagery** (NCII) and sexual deepfakes, which is the single most common real-world abuse of generative video and disproportionately targets women; **impersonation and fraud**, where a synthetic clip of a real person (an executive, a family member, a public figure) is used for financial fraud, "grandparent" scams, or fake authorization; **disinformation**, where a fabricated clip of an event or a statement that never happened is injected into the information ecosystem at a sensitive moment; **CSAM**, the generation of child sexual abuse material, which is categorically illegal and the highest-severity item on any safety review; and **harassment and defamation**, putting a real person in a fabricated, damaging situation. These are not hypothetical; they are the documented abuse patterns of every capable open and closed video model.

Two things make video *more* dangerous than a still image, and a serious safety review must internalize both. The first is **perceived realism**: motion is a powerful authenticity cue. A still image can be dismissed as a doctored photo, but a moving, temporally-coherent clip of a person triggers a much stronger "this really happened" response in viewers — the brain treats smooth motion as evidence of a real physical event. The second is **audio**. The 2024–2026 frontier models — Veo 3 and beyond, Movie Gen Audio — generate *synchronized speech and sound*, covered in [audio and joint AV generation](/blog/machine-learning/video-generation/audio-and-joint-av-generation). A silent deepfake is suspicious; a deepfake where the person's lips move in sync with a cloned voice saying the fraudulent words is *vastly* more convincing, and lip-sync plus voice cloning is precisely the combination that powers the highest-impact fraud. Realism times audio is the multiplier that turns a curiosity into a weapon, and it is why video provenance is a more urgent problem than image provenance ever was.

This is the threat model. Everything from here is defense — but before the controls, one more thing the threat model demands: a *method* for finding the holes, because you cannot defend a surface you have not probed.

### Red-teaming as a measurement, not a vibe

Red-teaming a video model is the safety analog of the eval harness from the first half: a *protocol* that produces a *number* you can track release over release. The number you want is the **attack success rate** (ASR) — the fraction of adversarial attempts that produce a policy-violating output — measured against a *fixed, versioned* red-team set, the same way the quality basket runs against a frozen prompt set. An ASR you cannot reproduce is a vibe; an ASR against a frozen adversarial set is a release gate.

The red-team set is built from *categories times techniques*. The categories are the misuse surface above (NCII, impersonation, CSAM-adjacent, disinformation, harassment). The techniques are the ways an adversary dodges your controls: **direct requests** (the lazy baseline — does the model just comply); **paraphrase and euphemism** (reword to slip the input classifier); **compositional prompts** (each clause benign, the composition harmful); **multi-turn or context priming** (build up to the violation across an interaction, where the model's later behavior is conditioned on earlier benign exchanges); **encoding tricks** (describe the target obliquely — "the CEO of the company whose logo is a bitten fruit" instead of a name, to dodge a name block-list); and **image-conditioning attacks** specific to I2V models (supply a real person's photo as the first frame and let the model animate it, which routes *around* a text prompt classifier entirely because the identity entered through the image, not the prompt). That last one is the video-specific seam: an [image-to-video](/blog/machine-learning/video-generation/conditioning-video-text-image-motion-camera) pipeline that filters text prompts but not conditioning images has a wide-open door, and any honest red-team must include first-frame-injection attacks.

The methodology that makes this rigorous: build the set once, version it, run it every release, report ASR per category and per technique, and — critically — keep a **held-out private slice** so you can tell whether you fixed the vulnerability or merely overfit to the public probes. When ASR drops on the public set but holds on the private set, you patched the symptom, not the cause. This is the same frozen-vs-held-out discipline as the human arena, pointed at safety. And like detection, red-teaming is asymmetric: a *low* ASR is weak evidence of safety (you only tested the attacks you thought of), while a *high* ASR is strong evidence of danger (one attack you found is one an adversary will find). You can never prove a model safe by red-teaming; you can only fail to break it, and then ship with humility and a fast incident-response path.

#### Worked example: reading an attack-success-rate table

Suppose your red-team set has 500 prompts (5 categories × 10 techniques × 10 prompts each). On the pre-mitigation base model, ASR is 38% overall — but it is not uniform: direct requests succeed 60% of the time, first-frame-injection 71%, and compositional 44%, while encoding tricks only 15%. After adding the input+output classifier stack, overall ASR drops to 9%, but read the breakdown: direct requests now 2%, compositional 6%, *but first-frame-injection is still 64%* because your classifier only looked at text. The aggregate "9%" hides a wide-open door. The lesson mirrors the still-dog: **never trust the aggregate; read the breakdown.** The fix is an output-frame classifier (which sees the animated person regardless of how the identity entered) and a conditioning-image classifier on the first frame — and you re-run the *same* 500-prompt set to confirm first-frame ASR actually falls. Numbers illustrative; the structure — aggregate hides the worst category — is the real and recurring pattern.

## 8. Mitigations: defense in depth, none of it sufficient alone

There is no single control that stops misuse, so the honest stack is **layered**, and the engineering skill is knowing exactly what each layer stops and exactly how it breaks. The five layers are training-data filtering, prompt and output safety classifiers, identity and likeness protections, watermarking, and provenance. The first three are the *generation-time* defenses; the last two are the *post-generation* defenses that follow the clip into the world. Read the layers as a matrix of what-it-stops against how-it-breaks, because every layer has a known failure and the point of layering is that an attacker who beats one still faces the others.

![A matrix figure listing the five safety layers against what each one stops and the specific condition under which each one breaks](/imgs/blogs/evaluating-and-red-teaming-video-generation-7.png)

**Training-data filtering** is the strongest control because it works at the source: you cannot generate well what the model never saw. The mechanism is worth stating precisely, because it is the only control with a *capability* effect rather than a *gate* effect — the others stop a request, this one removes the ability. A diffusion model learns to sample from the distribution of its training data; if a class of content is absent or heavily down-weighted in that distribution, the model's mass on that class is correspondingly low, and it generates it poorly or not at all. For CSAM the discipline is absolute and non-negotiable — scan the training corpus against known-CSAM hash databases (PhotoDNA-style perceptual hashing, operated through NCMEC and partners), aggressively filter, and never train on uncurated web video, because the web-scraped video corpora that power these models are exactly where this contamination enters. For NCII and real-person likenesses the move is filter-and-down-weight: remove sexual content paired with identifiable real people, and cap the representation of any individual so the model cannot build a high-fidelity generator of a specific person from sheer data volume. The practical pipeline is a cascade of cheap-to-expensive filters — perceptual-hash match (cheap, exact), then a content classifier over sampled frames (medium), then human review of the flagged tail (expensive) — applied to the corpus *before* training, because re-filtering after a model is trained does nothing; the capability is already in the weights.

The failure mode is honest to state and it is the crux of the open-vs-closed decision: **filtering the base model does not survive fine-tuning.** An attacker with the open weights can fine-tune the filtered capability back in on a few thousand examples of their own data, in hours on a single GPU, because the base model's representations (faces, bodies, motion) are intact and the fine-tune only has to re-route them. This is not a hypothetical — it is the documented reality of every open image and video model, and it means an open release *cannot* rely on data filtering as a durable safety control; the filtering protects the *default* behavior and the *reputation* of the released checkpoint, but it does not bound what a determined downstream actor can recover. The corollary for the release decision is stark: a closed API model can put a classifier in front of the weights and keep it there; an open-weight model hands the weights over and keeps nothing. Both are legitimate choices, but the model card must state plainly which safety surface the release actually has, and "we filtered the training data" must never be presented as if it bounds an open model's worst-case behavior, because it does not.

**Prompt and output safety classifiers** are the runtime gate: a classifier on the *input* prompt blocks obvious requests (named real people in sexual contexts, CSAM-seeking prompts), and a classifier on the *output frames* catches what slips through the prompt filter (NSFW content, a recognized face). This is medium-strength. The failure mode is **paraphrase and obfuscation**: an attacker rewords the prompt to dodge the input classifier, or constructs a prompt whose individual words are benign but whose composition is not. Output classification is more robust than input classification (it sees what was actually produced) but is not free at video scale — you are running a classifier over frames of every generation, which is real compute. The honest posture: input + output classifiers stop the lazy attacker and raise the cost for the determined one; they do not stop the determined one.

**Identity and likeness protections** are the policy-and-tooling layer specific to the deepfake threat: refusing to generate identifiable real people without consent, maintaining a block-list of public figures for sensitive contexts, and — emerging in 2025–2026 — *likeness opt-out* registries and the legal backdrop (the U.S. has moved on NCII with the 2025 TAKE IT DOWN Act mandating rapid takedown of non-consensual intimate imagery including AI-generated; the EU AI Act requires disclosure of AI-generated "deepfake" content; several U.S. states have election-deepfake and likeness statutes). Treat the exact statutes as approximate and fast-moving — the engineering takeaway is that "we refuse named real people in sensitive contexts and honor takedowns fast" is now a baseline expectation, not a nicety.

Here is a minimal safety-classifier hookup — a gate that runs *before* the expensive generation on the prompt and *after* it on sampled frames, the way a production pipeline wires it:

```python
def safe_generate(pipe, prompt, prompt_clf, frame_clf, **kw):
    # 1. INPUT GATE: cheap, runs before any GPU time is spent.
    verdict = prompt_clf(prompt)              # {"block": bool, "category": str}
    if verdict["block"]:
        raise SafetyRefusal(f"prompt blocked: {verdict['category']}")

    # 2. GENERATE.
    out = pipe(prompt, **kw)                  # diffusers video pipeline
    frames = out.frames[0]                    # list of PIL frames

    # 3. OUTPUT GATE: sample frames, classify; do not ship if any frame trips.
    sampled = frames[::max(1, len(frames) // 8)]   # ~8 frames
    flags = [frame_clf(f) for f in sampled]        # NSFW / known-face / etc.
    if any(f["unsafe"] for f in flags):
        raise SafetyRefusal(f"output blocked: {[f['label'] for f in flags]}")

    return frames                              # only safe output is returned
```

This is intentionally simple, and that simplicity is the point: the gates are cheap relative to generation, they fail closed (a refusal, not a leak), and the input gate spends no GPU time on a prompt that should never run. The economics are worth a moment, because they are why the *order* of the gates matters. A video generation is expensive — seconds of H100 time and gigabytes of VRAM for a single clip, per the deployment numbers in [efficient video inference and serving](/blog/machine-learning/video-generation/efficient-video-inference-and-serving) — while a text-classifier call on the prompt is milliseconds and a frame classifier over eight sampled frames is a fraction of the generation cost. So the input gate goes first not only for safety but for cost: it lets you refuse the obviously-harmful prompt before spending the expensive compute. The output gate cannot be skipped even when the input gate passes, though, precisely because of the first-frame-injection and compositional attacks from the red-team section — the identity or the harm can enter through a channel the text classifier never saw, and only a classifier on the *produced frames* catches it. The cost of running an output classifier on every generation is real (it is the one safety control that scales with your generation volume), but it is the control that closes the seams the cheaper gates leave open, and it is not optional for anything user-facing.

It is *not* sufficient even so — the matrix above tells you exactly how each gate breaks — but it is the table-stakes layer, and shipping without it is indefensible. The two layers that follow the clip into the world are watermarking and provenance, and they are the only defenses that still operate after the output has left your servers and your classifiers can no longer see it.

## 9. Watermarking video: invisible, per-frame, and under attack

A watermark is a signal you embed in the output that says *this was AI-generated* and survives normal handling well enough that a detector can recover it later. For images the state of the art is SynthID (DeepMind), which embeds an imperceptible pattern in the pixels that a matched detector reads back. For video the problem is harder and more interesting, because video has a time axis the watermark must use, and a distribution channel that attacks the watermark before any detector sees it.

![A directed dataflow figure tracing a watermark from per-frame embedding through a re-encoding-and-crop attack and a frame-sampling attack into a detector that recovers the key only if redundancy survives](/imgs/blogs/evaluating-and-red-teaming-video-generation-6.png)

The science of why a video watermark survives — or fails — is a redundancy-and-channel argument, and it is worth making precise. Model the watermark as a payload of $k$ bits embedded redundantly across $T$ frames, with some per-frame embedding strength. The distribution channel applies a sequence of degradations: H.264/H.265 re-encoding (lossy compression that perturbs every pixel), spatial cropping and resizing (which can shift or destroy a spatially-localized pattern), and — the attack unique to video — **frame sampling**: re-uploading at a different frame rate, dropping frames, or extracting a short sub-clip. Detection works if and only if enough redundant copies of the payload survive the channel for a matched filter to cross its decision threshold.

The crucial design consequence: **a watermark that lives in the temporal pattern across frames is killed by frame sampling, while a watermark embedded redundantly within each individual frame survives it.** Concretely, if your scheme encodes bits in the *relationship between consecutive frames* (a temporal modulation), then dropping every other frame destroys the relationship and the payload is gone. If instead each frame independently carries the full payload (a SynthID-style spatial watermark replicated per frame), then sampling down to even a handful of frames still leaves several intact copies, and the detector recovers the key. So the robust design is **per-frame spatial redundancy** — accept that you cannot rely on the temporal axis because the temporal axis is exactly what an attacker resamples.

The cleanest way to think about this is as an **error-correcting code over a noisy channel.** The payload is a message; the channel is the distribution pipeline (re-encode, crop, resample); and the question is whether the code rate is below the channel capacity that survives. Per-frame redundancy is repetition coding: the same payload written $T$ times, so the detector can majority-vote or average over every surviving copy. If an attacker drops a fraction $1-\rho$ of frames, you still have $\rho T$ copies, and the matched-filter response — whose signal-to-noise ratio grows like $\sqrt{\rho T}$ — degrades *gracefully* rather than falling off a cliff. The detector's decision is a hypothesis test (watermark present vs absent), and its reliability is set by that SNR: enough surviving redundant copies push the response above threshold; too few and it sinks into the false-negative region. This is exactly why the per-frame design has a *robustness budget* you can reason about — you choose the per-frame strength and the redundancy to keep the post-channel SNR above threshold for the *incidental* degradations you expect (a platform transcode drops maybe 10–20% of quality, not 90%), while accepting that a *deliberate* attacker who pushes the channel past capacity (crush the bitrate, crop to a corner, resample to a few frames) drives the SNR below threshold and wins. There is no free lunch: raising the strength to survive harder attacks makes the watermark more visible, and an invisible watermark is, by definition, a low-SNR signal that a high-noise channel can erase. That trade — imperceptibility against robustness, mediated by channel capacity — is the whole game, and it is why the honest claim is "survives incidental handling, not deliberate removal." The cost is that per-frame spatial watermarks are more vulnerable to *spatial* attacks (heavy crop, strong re-compression), which is why no watermark is robust to a *determined* adversary who re-encodes hard, crops aggressively, and resamples frames. The realistic claim — and the one to put in the model card — is: **watermarks survive incidental handling (a normal re-upload, a platform transcode) but not a deliberate removal attempt.** They raise the cost of laundering provenance; they do not make it impossible.

Here is an embed-and-detect sketch at the level that conveys the design — a per-frame redundant payload with a matched-filter detector. It is illustrative, not production SynthID (the real schemes are learned and far more sophisticated), but it makes the redundancy argument concrete:

```python
import numpy as np

def make_pattern(key: int, shape, strength=2.0):
    rng = np.random.default_rng(key)
    # a fixed, key-derived, zero-mean spatial pattern (one per channel)
    return strength * rng.standard_normal(shape).astype(np.float32)

def embed_watermark(frames, key, strength=2.0):
    # frames: (T, H, W, C) float array in [0, 255]; SAME pattern every frame
    pat = make_pattern(key, frames.shape[1:], strength)   # (H, W, C)
    return np.clip(frames + pat[None], 0, 255)            # redundant across T

def detect_watermark(frames, key, strength=2.0, thresh=0.5):
    pat = make_pattern(key, frames.shape[1:], strength)
    pat = (pat - pat.mean()) / (pat.std() + 1e-8)
    scores = []
    for f in frames:                          # detect per surviving frame
        x = f - f.mean()
        scores.append(float((x * pat).mean() / (x.std() + 1e-8)))
    # average the matched-filter response over whatever frames SURVIVED
    response = float(np.mean(scores))
    return response > thresh, response        # robust BECAUSE it is redundant
```

Notice the detector averages over *whatever frames survived* — drop half the frames and the response degrades gracefully rather than collapsing, because every surviving frame still carries the full pattern. That graceful degradation is the entire reason per-frame redundancy is the right design, and it is also why a watermark that smears one payload across the temporal dimension would fail this exact test. SynthID for video (DeepMind has extended SynthID to video and audio) follows this philosophy with a learned, far more imperceptible and robust pattern, but the redundancy principle is the same. Watermarking is the *invisible* half of provenance; the *visible, cryptographic* half is C2PA.

## 10. Provenance: C2PA Content Credentials for video

Watermarking hides a signal *in* the content. Provenance attaches a verifiable record *to* the content. The two are complementary, and the second is, in 2026, the more credible long-term answer because it is cryptographic rather than a signal-processing arms race.

**C2PA** (the Coalition for Content Provenance and Authenticity — Adobe, Microsoft, the BBC, and others, now standardized and adopted across major tools) defines **Content Credentials**: a tamper-evident, cryptographically-signed manifest embedded in the media file that records its provenance — who or what created it, what tools and AI models touched it, and a chain of edits. For an AI-generated video the manifest asserts, in a signed claim, "this was generated by model X on date Y," and anyone with the public key can *verify* the signature. Crucially, C2PA does not rely on the signal surviving re-encoding — it relies on cryptography: if the manifest is intact and the signature verifies, the provenance is trustworthy with cryptographic certainty.

The honest limitation is the inverse of the watermark's. A watermark survives stripping but loses to signal degradation; a C2PA manifest is cryptographically unforgeable but is *trivially stripped* — re-encode the video through a tool that does not preserve the manifest, or screen-record it, and the credential is simply *gone*. C2PA tells you, with certainty, the provenance of a file that *still has its manifest*; it tells you *nothing* about a file whose manifest was removed, and absence of a credential is not evidence of authenticity (it might be an honest clip from a non-C2PA camera, or a deepfake with the manifest stripped). This is why C2PA and watermarking are deployed *together*: the watermark is the fallback that survives when the manifest is stripped, and the manifest is the strong cryptographic claim when it survives. The 2024–2026 trajectory — major model providers attaching C2PA to AI outputs, camera manufacturers signing at capture, platforms beginning to surface "AI info" labels by reading credentials — is the most promising provenance direction, but it is an *ecosystem* play: it only works when generation, distribution, and display all participate, and that buildout is incomplete.

Here is a sketch of reading and writing a C2PA-style manifest (the real toolchain is `c2patool` / the `c2pa` libraries; this conveys the shape — a signed claim with an assertion that the content is AI-generated):

```python
import json, hashlib, time
# Illustrative shape of a C2PA-style claim. In production use the c2pa SDK,
# which handles real X.509 signing, the JUMBF embedding, and verification.

def build_manifest(video_path, model_id, signer_key):
    with open(video_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    claim = {
        "claim_generator": "my-video-model/1.0",
        "assertions": [
            {"label": "c2pa.actions",
             "data": {"actions": [{"action": "c2pa.created",
                                   "digitalSourceType": "trainedAlgorithmicMedia"}]}},
            {"label": "com.example.ai_model",
             "data": {"model": model_id, "generated": True}},
            {"label": "c2pa.hash.data", "data": {"alg": "sha256", "hash": digest}},
        ],
        "timestamp": int(time.time()),
    }
    payload = json.dumps(claim, sort_keys=True).encode()
    claim["signature"] = signer_key.sign(payload)   # real C2PA uses X.509 + COSE
    return claim

def verify_manifest(video_path, manifest, public_key):
    sig = manifest.pop("signature")
    payload = json.dumps(manifest, sort_keys=True).encode()
    if not public_key.verify(sig, payload):
        return False, "signature invalid -- manifest tampered"
    with open(video_path, "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != \
           next(a["data"]["hash"] for a in manifest["assertions"]
                if a["label"] == "c2pa.hash.data"):
            return False, "content hash mismatch -- file altered after signing"
    return True, "verified AI-generated provenance"
```

The `digitalSourceType: trainedAlgorithmicMedia` assertion is the actual C2PA vocabulary for "this is AI-generated," and the content-hash assertion is what makes the manifest detect post-signing tampering. Provenance answers "where did this come from" for cooperative content. The last layer — *detection* — is the adversarial, post-hoc attempt to answer it for content that carries no watermark and no manifest, and it is the weakest link.

## 11. Detecting generated video, and why it keeps losing

Detection is the defense of last resort: given a clip with *no* watermark and *no* C2PA manifest, can a classifier decide whether it was AI-generated? This is the deepfake-detector problem, and it is fundamentally a losing game against improving generators. Understanding *why* it loses is the most important thing a defender can internalize, because it tells you not to rely on it.

Frame detection as a binary classifier $D(x) \in [0,1]$ trained to separate real video from generated video. It works by finding the *artifacts* that distinguish current generators from reality — temporal inconsistencies, unnatural blink rates, physically-impossible lighting, telltale frequency-domain signatures of the generator's upsampling. Here is the structural problem, stated as the adversarial dynamic it is. The generator's training objective is, by construction, to *minimize* exactly the distributional gap the detector *exploits* — FVD, the very metric the first half of this post is about, *is* a measure of the detector's job, and every point of FVD a generator improves is an artifact a detector loses. As generators get better, the artifacts the detector relies on disappear. Formally, if $p_g \to p_r$ (the generated distribution converges to the real one, which is the entire point of training), then the Bayes-optimal detector's accuracy converges to $0.5$ — chance. **A perfect generator is, by definition, undetectable by any classifier.** Detection accuracy is therefore not a fixed property; it is a *decreasing function of generator quality*, and it degrades every time the generators improve.

This has three brutal consequences for anyone tempted to lean on detection. First, **detectors do not generalize across generators**: a detector trained on Sora artifacts fails on Veo artifacts and on next year's model, because the artifacts are generator-specific and the new generator does not have the old one's tells. Second, **detectors are trivially attacked**: a small amount of added noise, a re-compression, or an adversarial perturbation flips the classifier, because the detector keys on subtle features that are easy to perturb. Third, **the false-positive cost is asymmetric and high**: a detector that flags real video as fake is its own harm (it is used to dismiss authentic evidence — the "liar's dividend"), so you cannot just crank the threshold. The honest engineering conclusion: **detection is a useful triage signal at scale, never a verdict.** It is worth running to flag suspicious content for human review; it is malpractice to treat its output as proof either way. This is precisely why the provenance-and-watermarking approach — proactively marking content at generation — is the strategically correct bet, and detection is the weak fallback for the content that escaped marking. The same conclusion holds for the image world, derived in [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance); video only makes the artifacts disappear faster, because there is more signal for the generator to get right.

#### Worked example: a detector's half-life

Suppose your deepfake detector reaches 95% accuracy on clips from this year's generator. Next year's generator closes roughly half the remaining FVD gap to real video. Empirically (and this is the documented pattern across detector-generator generations, mark it approximate), detector accuracy on the *new* generator, *without retraining*, tends to fall toward the 60–70% range — barely better than chance for a high-stakes decision — and after another generation toward the 50s. The detector has a *half-life* measured in model generations, and it is short. The cost of keeping it useful is a perpetual retraining treadmill where you are always a generation behind, because you can only train on artifacts you have already seen. Contrast a watermark or a C2PA manifest, whose reliability does *not* decay as generators improve — it depends on the embedding and the cryptography, not on the generator being detectably flawed. That asymmetry is the entire argument for proactive provenance over reactive detection.

## 12. Case studies and real numbers

A few grounded reference points from shipped models and the literature, to anchor the claims above in real numbers. Treat exact figures as approximate and version-dependent — the point is the *shape* of the numbers, not the third significant digit.

**VBench as the open-model scoreboard.** The open frontier models report VBench prominently: HunyuanVideo, CogVideoX, and the Wan 2.x line all publish per-dimension VBench breakdowns, and the credible ones report dynamic degree alongside consistency precisely so reviewers can check for the still-dog gaming. The pattern in the leaderboards is exactly the trade-off this post formalizes — the models near the top of *total* score are not the ones that maxed any single dimension but the ones that held a *balanced* profile, high consistency *with* a real dynamic degree. A model card that reports only the dimensions it wins is a red flag.

**Veo 3 and the audio multiplier.** Google's Veo 3 (2025) shipped synchronized native audio — speech, ambient sound, lip-sync — which is exactly the realism-times-audio multiplier from the misuse-surface section, and it is why Google paired it with SynthID watermarking on every output and C2PA participation. This is the industry acknowledging, in product decisions, that the audio frontier raises the provenance stakes. It is the right architecture: proactive marking at generation, not reactive detection.

**SynthID's reach.** DeepMind's SynthID, which began as an image watermark, has been extended to audio and video and is applied to outputs across Google's generative products; by 2024–2025 Google reported billions of pieces of SynthID-marked content, and opened a SynthID detector. This is the largest real-world deployment of generative watermarking, and its very existence is the proof-of-concept that per-output proactive marking is operationally feasible at scale — the redundancy-and-imperceptibility engineering is solved well enough to deploy, even though the robustness-against-deliberate-removal limit remains.

**C2PA adoption.** Content Credentials moved from a coalition spec to broad adoption across creative tools, camera makers signing at capture, and platform-side "AI info" labels reading the credential — the 2024–2026 trajectory is real, if incomplete. The honest read is that C2PA is the most credible *long-term* provenance answer and simultaneously useless on any clip whose manifest was stripped, which is why it must ship *with* a watermark, not instead of one.

**The regulatory backdrop.** The EU AI Act's transparency obligations require labeling of AI-generated/deepfake content; the U.S. TAKE IT DOWN Act (2025) mandates rapid takedown of non-consensual intimate imagery including AI-generated; multiple U.S. states regulate election deepfakes and likeness. Treat specifics as fast-moving and approximate — the engineering consequence is stable: *disclosure, provenance, and fast takedown are now baseline obligations, not optional polish.*

Here is a compact comparison of the five mitigation layers as a reference table — the same content as the safety matrix figure, in a form you can paste into a model card's risk section:

| Mitigation | What it stops | Robustness | Primary failure mode |
|---|---|---|---|
| Training-data filtering | CSAM, NCII, real-person likeness at the source | Strong (at source) | Fine-tuning the open weights restores the capability |
| Prompt + output classifier | Obvious harmful prompts and unsafe frames | Medium | Paraphrase / compositional jailbreak; output-classifier compute cost |
| Identity / likeness protection | Non-consensual real-person deepfakes | Policy-dependent | Coverage gaps; off-platform fine-tunes; statute lag |
| Watermark (SynthID-style) | Laundering provenance via casual re-upload | Survives incidental handling, not deliberate removal | Heavy crop + re-encode + frame resample |
| C2PA Content Credentials | Forging provenance of credentialed content | Cryptographically unforgeable | Trivially stripped by re-encode; absence ≠ authenticity |
| Detection (post-hoc) | Flagging unmarked content for review | Weak and decaying | Degrades as generators improve; flips under perturbation |

## 13. When to reach for what (and when not to)

A decisive section, because the failure mode of safety work is doing everything badly instead of the right things well.

**For the eval harness:** *do* build a basket with a hard dynamic-degree gate and a frozen human spot-check; *do not* ship on a single leaderboard number, ever — it is the still-dog trap mechanized. *Do* run FVD/VBench in the inner loop and the human arena on release candidates only; *do not* run the arena every checkpoint (too slow, too expensive) or trust an A/B with twenty raters (statistically empty — you need hundreds of votes per pair). *Do* validate your VLM-as-judge against human labels before trusting it; *do not* judge with the same model family that generated, or self-preference will inflate your own scores.

**For watermarking:** *do* embed a per-frame, spatially-redundant watermark so it survives frame sampling; *do not* design a watermark that lives in the temporal pattern across frames — resampling kills it. *Do* describe the watermark's real robustness honestly in the model card ("survives incidental handling, not deliberate removal"); *do not* claim it is tamper-proof, because a determined attacker re-encodes and crops it away.

**For provenance:** *do* attach C2PA Content Credentials to every generated output *and* watermark — they cover each other's failure (manifest stripped → watermark survives; watermark degraded → manifest verifies); *do not* rely on C2PA alone (trivially stripped) or treat absence of a credential as proof of authenticity.

**For detection:** *do* use it as a triage signal to route suspicious content to human review; *do not* treat its output as a verdict in any high-stakes decision, and *do not* invest your safety budget here expecting it to hold — it has a half-life of model generations and you will always be behind. Put the budget into proactive provenance instead.

**For the open-vs-closed release decision:** *do* recognize that training-data filtering and runtime classifiers are *defeated by fine-tuning* on open weights, so an open release leans entirely on the post-generation layers and on community norms; *do not* pretend an open-weight model has the same safety surface as an API behind a classifier — it does not, and the model card should say so plainly.

## 14. Key takeaways

- **Shipping is two pipelines.** Quality eval is an averaging problem over typical output; safety eval is a worst-case problem over the adversarial tail. They fail in opposite directions, and a green quality dashboard tells you nothing about safety.
- **Read VBench as a tree of pairs, not a number.** Subject consistency and dynamic degree are mathematically opposed — the identical-frame clip maxes one and floors the other — so gate on a dynamic-degree floor and only then rank.
- **CLIP-score is a bag of frames.** It is permutation-invariant over time and blind to event order; use a VLM-as-judge over sampled frames, validated against human labels, for temporal and relational alignment.
- **The human arena is the only arbiter.** Pairwise, blind, randomized, on a frozen prompt set, with enough votes that the Elo confidence intervals separate ($n \propto 1/\text{gap}^2$). Run it on release candidates, not every checkpoint.
- **Video misuse is worse than image misuse** because perceived realism plus synchronized audio multiplies the deception — which is exactly why proactive provenance is more urgent for video.
- **Defense is layered and nothing is sufficient alone.** Data filtering (defeated by fine-tuning), classifiers (defeated by paraphrase), watermarks (defeated by re-encode+resample), C2PA (defeated by stripping), detection (defeated by better generators). The point of the stack is that no single failure is catastrophic.
- **Watermark per-frame and redundantly** so it survives frame sampling; the temporal axis is exactly what an attacker resamples. Describe its real robustness honestly.
- **Ship C2PA *and* a watermark together** — cryptographic-but-strippable plus signal-based-but-degradable cover each other's failure mode.
- **Detection is a losing game by construction.** As $p_g \to p_r$ the Bayes-optimal detector tends to chance; it has a half-life of model generations. Use it for triage, never as a verdict, and put the safety budget into proactive provenance.

## 15. Further reading

- Huang et al., *VBench: Comprehensive Benchmark Suite for Video Generative Models*, 2024 — the dimension decomposition and the per-dimension protocols; and the VBench-2.0 extension toward physics, identity, and commonsense.
- Unterthiner et al., *Towards Accurate Generative Models of Video: A New Metric and Challenges* (FVD), 2018; Carreira and Zisserman, *Quo Vadis, Action Recognition?* (I3D), 2017 — the FVD backbone.
- Jayasumana et al., *Rethinking FID: Towards a Better Evaluation Metric for Image Generation* (CMMD), 2024 — why an MMD-based distribution metric beats Fréchet-on-a-Gaussian, the same argument that should move FVD.
- DeepMind, *SynthID* technical materials (image, audio, video watermarking) — the largest real-world deployment of generative watermarking.
- C2PA, *Content Credentials* specification (Coalition for Content Provenance and Authenticity) — the cryptographic provenance standard and the `trainedAlgorithmicMedia` vocabulary; pair it with the `c2patool` and `c2pa` libraries for the real signing, JUMBF embedding, and verification path the sketch above only gestures at.
- On the regulatory backdrop, treat the EU AI Act transparency provisions, the U.S. TAKE IT DOWN Act (2025), and the patchwork of state election-deepfake and likeness statutes as the fast-moving floor of obligations a release must meet — disclosure, provenance, and rapid takedown are now baseline, not optional, and the exact text changes faster than any blog post can track.
- The [metrics foundations post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) for the full FVD/VBench derivation; the image-series companions [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) and [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) for the image-side derivations this post builds on.
- Within-series context: [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) for the coherence × motion × length × cost frame; [sora and the world simulator thesis](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) and [physics and the limits of learned simulation](/blog/machine-learning/video-generation/physics-and-the-limits-of-learned-simulation) for the realism claims this post pressure-tests; the deployment economics in [efficient video inference and serving](/blog/machine-learning/video-generation/efficient-video-inference-and-serving); and the end-to-end pipeline that puts eval and safety in their place, the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
