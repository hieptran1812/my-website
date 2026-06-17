---
title: "The Metrics of Video Generation: FVD, VBench, and the Motion-vs-Stability Trap"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to score generated video honestly: how FVD works and why it is noisy, what VBench actually measures, and why a model can win on consistency by barely moving."
tags:
  [
    "video-generation",
    "diffusion-models",
    "evaluation",
    "fvd",
    "vbench",
    "generative-ai",
    "deep-learning",
    "text-to-video",
    "metrics",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-metrics-of-video-generation-1.png"
---

Two checkpoints from the same training run land on your desk. Checkpoint A renders a dog running across a field: the fur catches the light, the legs cycle, the camera drifts, and at second three the dog's body shears apart for two frames before snapping back. Checkpoint B renders the same prompt as a dog standing almost perfectly still, breathing faintly, the background rock-solid. You run your metric. Checkpoint B wins. It has a lower FVD, a higher subject-consistency score, a higher motion-smoothness score, and a higher imaging-quality score. By every number on your dashboard, the still dog is the better video model.

Anyone who has actually watched the two clips knows this is backwards. Checkpoint A is the one you want to ship — it has a temporal bug to fix, but it is *generating video*. Checkpoint B is generating a photograph with a slight tremor and calling it a video. The metric is not lying, exactly; it is answering a different question than the one you care about. And this gap — between what a number says and what a human sees — is the entire subject of this post. Video evaluation is not image evaluation with an extra dimension bolted on. It is harder, noisier, and far easier to game, and if you do not understand exactly how each metric can be fooled, you will optimize your model straight into the still-dog trap.

![A directed dataflow figure showing real and generated clips flowing into a shared I3D backbone, then into two feature Gaussians, then into a single Frechet distance score](/imgs/blogs/the-metrics-of-video-generation-1.png)

This post is the measurement chapter of the series. We will build up the standard toolkit from first principles: **FVD** (Fréchet Video Distance), the distribution metric that lifts FID to video; the **per-frame image metrics** people reach for and why they are blind to exactly the artifacts video introduces; **VBench**, the multi-dimensional benchmark that refuses to give you one number; the **text-video alignment** metrics; and **human evaluation**. Along the way we will formalize the one trade-off that matters most — the **consistency-versus-dynamic-degree** trap — and show why optimizing either axis alone is gameable. By the end you will have an opinionated, runnable recipe for evaluating a video model you actually intend to ship. If you have not read it yet, [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) sets up the coherence × motion × length × cost frame this whole post lives inside, and the image-series companion [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly) is the prerequisite for understanding why FID — and therefore FVD — is shakier than its ubiquity suggests.

## 1. Why measuring video is a different problem

Start with the obvious fact and then look at why it is not obvious at all. A video is a sequence of frames. The naive instinct is: I already know how to score an image, so I will score every frame and average. This fails, and it fails in a way that is worth dwelling on, because the failure mode is the seed of every video-specific metric.

The frame-average throws away the one thing that makes a video a video: **order**. If you take the frames of a clip, shuffle them into a random permutation, and re-run a per-frame metric, you get the identical score. A per-frame metric is *permutation-invariant* over the time axis by construction. But a human cannot un-see a shuffle — a shuffled clip is a strobing mess. So any metric that is permutation-invariant over time is provably blind to temporal coherence, which is the central quality axis of video. This is not a subtle bias; it is a hard mathematical property. No amount of per-frame averaging recovers it.

That single observation forks the field into two metric families. The first family says: *fine, then we need features that look at multiple frames at once* — features whose value changes when you shuffle the frames. That is the road to FVD, where we replace the per-frame Inception network with a 3D-convolutional network that has a temporal receptive field. The second family says: *one number cannot possibly capture all of this* — so we decompose video quality into separate, named axes and score each one. That is the road to VBench. Modern practice uses both, plus humans, because each one has a blind spot the others cover.

There is a second reason video evaluation is harder, and it is statistical rather than conceptual. Video features are *expensive and few*. To estimate a distribution of natural images for FID you might pass 50,000 images through Inception in a couple of minutes. To estimate a distribution of natural *clips* you pass a few thousand clips — each one sixteen or more frames — through a heavier 3D network, and you typically have far fewer real reference clips of any given content type than you have reference images. Smaller sample sizes mean higher variance in the estimated statistics, and as we will see, the Fréchet distance has a known finite-sample bias that *always* inflates the score when you have too few samples. So the video version of the metric is born noisier than its image parent, before you have even chosen a backbone.

Hold the running example in your head the whole way through: **a 5-second clip of a dog running, at 24 fps, 720p**. That is 120 frames, which after the causal 3D-VAE's temporal compression (covered in [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression)) becomes a much smaller latent, but for evaluation we score the decoded pixels. Every metric below, we will ask the same three questions of: what does it capture, what is it blind to, and how would a lazy optimizer game it to beat you without making the video any better.

There is a third, deeper reason video evaluation is hard, and it is worth naming early because it shapes everything: **video quality is not one quantity, it is several quantities in tension.** When you score an image you are mostly asking one question — is this a good image — and fidelity, sharpness, and prompt-match mostly point the same way. When you score a video you are asking at least four questions that *actively conflict*: is each frame good (fidelity), do frames agree with each other (coherence/consistency), does the thing actually move (dynamics), and does the motion make sense (smoothness, physical plausibility). The conflict is not incidental. Pushing on consistency mechanically lowers dynamics, as we will prove. Pushing on dynamics mechanically risks coherence. So a single scalar that tries to summarize "video quality" must implicitly *weight* these conflicting axes, and whatever weighting it picks is gameable by a model that over-invests in the cheaply-maximized axes. This is the root reason the field abandoned the single-number dream that FID enjoyed in the image world: in video, the single number is not just noisy, it is *structurally* exploitable. Every section below is, in one way or another, a consequence of this one fact.

One more framing before we dig in, because it determines how you should *read* every metric in this post. There are two distinct things you might want a metric to do, and confusing them is the source of a lot of bad evaluation. The first is **ranking** — telling you reliably whether checkpoint A is better than checkpoint B. The second is **grading** — putting an absolute, interpretable quality number on a single model that means the same thing across papers and time. Almost every video metric is a passable *ranker* under a fixed protocol and a terrible *grader* across protocols. FVD ranks your own checkpoints fine and grades nothing absolutely. VBench dimensions rank within a version and grade only loosely. The mistake that wrecks comparisons is using a ranker as a grader — reading an absolute FVD of 200 as "good" when it only ever meant "lower than our last run." Hold the ranker-vs-grader distinction the whole way through and most of the protocol discipline in this post will feel obvious rather than fussy.

## 2. FID in one paragraph, so FVD makes sense

You need FID before FVD, so here is the compressed version; the full treatment is in the [image evaluation post](/blog/machine-learning/image-generation/evaluating-image-generation-honestly). The Fréchet Inception Distance does not compare images to images. It compares two *distributions*. You take a large set of real images and a large set of generated images, push both through a frozen Inception-v3 network, and grab a high-level feature vector (the 2048-dim pooled activation) for each image. Now you have two clouds of points in feature space. FID models each cloud as a single multivariate Gaussian — it estimates a mean vector and a covariance matrix for each — and reports the **Fréchet distance** (also called the 2-Wasserstein distance) between those two Gaussians.

For two Gaussians $\mathcal{N}(\mu_r, \Sigma_r)$ and $\mathcal{N}(\mu_g, \Sigma_g)$ that distance has a closed form:

$$
d^2 = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right).
$$

The first term punishes a difference in the *average* feature — roughly, are your generated images in the right region of feature space at all. The second term, the trace term, punishes a difference in the *spread and correlation* of features — roughly, do your generated images have the right diversity and the right covariance structure. A model that produces gorgeous but near-identical images (mode collapse) keeps $\mu$ close but gets the covariance wrong, and the trace term lights up. That is why FID rewards both fidelity and diversity, and it is the whole reason the field standardized on it.

Two properties carry straight over to video and are worth stating plainly because they are the source of half of FVD's problems. First, FID is **only as meaningful as its backbone**: it measures distance in Inception's feature space, so it sees what Inception was trained to see (ImageNet object categories) and is partly blind to what Inception ignores. Second, the Gaussian assumption is a *modeling choice*, not a fact — real feature clouds are not Gaussian, and the closed-form distance is an approximation that gets worse as the clouds get less Gaussian and as your sample size shrinks. Keep both of these in mind. FVD inherits them and adds a third axis of fragility on top.

## 3. FVD: the same idea, with a backbone that can see motion

Fréchet Video Distance is the obvious and correct generalization: do exactly what FID does, but swap the per-frame Inception network for a network whose features depend on multiple frames in temporal order. The standard choice is **I3D** — the Inflated 3D ConvNet from Carreira and Zisserman's 2017 "Quo Vadis" paper — trained on the Kinetics human-action dataset. I3D takes a short clip (commonly 16 frames) and runs 3D convolutions over it, so each feature is computed from a *spatiotemporal* receptive field. Shuffle the frames and the I3D features change, because the convolution kernels span time. That is the entire trick, and it is why FVD can see what per-frame FID cannot.

![A two-column before-after figure contrasting a 2D Inception backbone that is blind to frame order against a 3D-convolution I3D backbone whose features depend on motion](/imgs/blogs/the-metrics-of-video-generation-2.png)

Mechanically, FVD is FID with clips:

1. Sample $N$ real clips and $N$ generated clips, each a fixed number of frames (typically 16) at a fixed resolution (I3D's native input, usually $224\times224$, with center-crop and the network's normalization).
2. Push every clip through frozen I3D and grab the logits-layer or pre-logits feature (commonly a 400-dim or pooled feature, depending on the implementation).
3. Fit a Gaussian to each set of features: estimate $\mu_r, \Sigma_r$ and $\mu_g, \Sigma_g$.
4. Report the same Fréchet distance formula as above.

So FVD is literally $d^2 = \lVert \mu_r - \mu_g \rVert^2 + \operatorname{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$ computed on I3D features instead of Inception features. The figure above and the pipeline figure in the intro are the same four steps drawn two ways.

### Why I3D sees motion that Inception cannot

It is worth being precise about *why* the 3D convolution matters, because "it captures motion" is the kind of phrase that sounds like an explanation and is not. Consider a single 3D convolution kernel of temporal extent 3, spatial extent $3\times3$. Its output at a spatiotemporal location is a weighted sum over a $3\times3\times3$ window — three consecutive frames. If the dog's leg is in one position in frame $t$ and a slightly different position in frame $t+1$, the kernel sees both at once and can fire on the *displacement*, not just the appearance. Stack many such layers and the network builds features that respond to coherent motion patterns: a leg cycling, a camera panning, water flowing. A 2D Inception kernel, by contrast, only ever sees one frame; the best it can do after per-frame averaging is report that each frame, individually, looks like a plausible image — which a strobing shuffle also satisfies.

So the formal claim is: I3D features are *not* invariant to temporal permutation, and per-frame Inception features (averaged over the clip) *are*. FVD therefore has a non-zero gradient with respect to temporal artifacts — flicker, jitter, frame-order errors, motion that is physically implausible — and per-frame FID has exactly zero gradient with respect to all of them. That is the precise sense in which FVD "captures motion." It does not understand physics; it has a feature space in which motion-realistic and motion-broken clips land in different places, and the Fréchet distance measures how far your generated cloud sits from the real one in that space.

#### Worked example: the strobe test

Take 256 real clips of the running dog and compute FVD against (a) a copy of those same clips and (b) the same clips with every clip's frames randomly shuffled. Case (a) gives an FVD near the small positive floor you always get from finite samples (more on that floor below) — call it roughly 8 on a typical I3D implementation. Case (b), the shuffle, gives a dramatically higher FVD — easily 300 to 600 on the same setup — even though every individual frame is a pixel-perfect real frame. Per-frame FID between (a) and (b) is *exactly zero*: same frames, same distribution. That single contrast is the cleanest demonstration that FVD measures something FID structurally cannot. It is also a good sanity check to run on any FVD implementation you adopt: if shuffling frames does not move the score, your features are not temporal and you have a bug.

## 4. The three ways FVD lies to you

FVD is the best single number we have, and it is still treacherous. Its problems are not academic footnotes; they are the reason two papers can report "FVD 290" and "FVD 410" and the comparison can be meaningless. There are three families of problem.

![A four-row matrix figure listing FVD noise sources — sample size, clip length, backbone, content-versus-motion — with the direction of each bias and its fix](/imgs/blogs/the-metrics-of-video-generation-3.png)

**High variance and sample-size bias.** The Fréchet distance between two Gaussians estimated from finite samples is a *biased* estimator, and the bias is always positive — it inflates the distance — and shrinks as $N$ grows. With too few clips your covariance estimate $\hat\Sigma$ is noisy and systematically off, and the trace term picks up that noise as if it were real distributional difference. The practical consequence: FVD computed on 256 clips can be 30–50% higher than the same comparison on 2048 clips, for no reason but the sample size. Two papers using different $N$ are simply not comparable. The fix is to fix $N$ — the de-facto standard is 2048 generated clips against a large real reference set, but the *real* fix is to report $N$ and never compare across different ones. Because of the positive bias you also cannot trust tiny absolute differences: an FVD of 188 versus 195 on a 2048-clip run is inside the noise band. Run multiple seeds and report the spread.

**Clip-length and frame-rate sensitivity.** I3D ingests a fixed number of frames. If you feed it 16 frames sampled from a 2-second clip versus 16 frames sampled from an 8-second clip, you have changed the temporal content per frame-window, and the FVD shifts. Different sampling strides, different clip lengths, different fps — all move the number. A model evaluated on 16-frame windows is not comparable to one evaluated on 128-frame windows. The fix is, again, to fix the protocol: same clip length, same fps, same number of frames per I3D window, same sampling stride.

**Backbone bias and content-vs-motion entanglement.** This is the deep one. I3D was trained on Kinetics — human actions. Its feature space is tuned to *human-action* motion and *natural-video* texture. Evaluate a model that generates anime, or abstract animation, or a domain far from Kinetics, and I3D's features are partly out of distribution; the FVD is measuring "distance in a space that does not represent your content well." Worse, I3D features entangle **content** and **motion**. A clip that gets the *appearance* slightly wrong (a color cast, a style shift, a sharpening artifact) moves in feature space the same direction a clip that gets the *motion* wrong does — the metric cannot tell you which. So a model that fixed its flicker but introduced a faint color shift can come out with a *worse* FVD, and you would conclude, falsely, that the flicker fix hurt. There is a growing move to swap I3D for **VideoMAE** features, which are self-supervised and arguably more general, but that changes the absolute scale of the metric entirely — VideoMAE-FVD and I3D-FVD are different numbers and you must never compare them. The fix here is partly damage control: report the backbone, keep the reference set in the same domain as your generations, and treat FVD as a *relative* signal within one fixed protocol, never an absolute quality grade.

A blunt way to hold all of this: **FVD is a thermometer with no marked scale.** Within one carefully fixed setup it tells you reliably whether checkpoint B is hotter or colder than checkpoint A. Across setups, the readings are in different units. Most published FVD comparisons quietly violate this, which is why you should be skeptical of any FVD leaderboard that spans papers.

### Where the sample-size bias actually comes from

It is worth deriving the positive bias rather than asserting it, because once you see *why* it is always positive you will stop trusting small-$N$ FVD forever. Suppose, hypothetically, that your generated distribution is *identical* to the real one — the true Fréchet distance is exactly zero. Now estimate two Gaussians, $(\hat\mu_r, \hat\Sigma_r)$ from $N$ real samples and $(\hat\mu_g, \hat\Sigma_g)$ from $N$ generated samples, both drawn from that same true distribution. Will the *estimated* Fréchet distance be zero? No. The estimated means differ by sampling noise of order $\sigma/\sqrt{N}$, so the mean term $\lVert \hat\mu_r - \hat\mu_g\rVert^2$ is a sum of squared noise — a strictly positive quantity whose expectation scales like $D\sigma^2 / N$, where $D$ is the feature dimension. The covariance term behaves similarly: two independent noisy estimates of the same covariance do not cancel in the trace term, they add a positive discrepancy. So the *expected* estimated distance is positive even when the true distance is zero, and it decays roughly as $1/N$. That is the bias, and it is *always upward* because it is built from squared and trace-of-positive quantities that cannot be negative.

The dimension factor $D$ is why video is hit harder than images here. I3D features are high-dimensional, and the covariance matrix you must estimate has $O(D^2)$ entries; estimating a $D \times D$ covariance well needs a number of samples that grows with $D$, and with a few thousand clips you are chronically under-sampled for the covariance. The trace term, which depends on that covariance estimate, is therefore the noisiest part of the FVD, and it is also the part that carries the diversity signal you care about. This is the precise mechanism behind the "256 clips reads 30–50% higher than 2048 clips" rule of thumb: you are watching the $1/N$ bias term shrink as you add samples. It also explains why the strobe-test floor in Section 3 was "roughly 8" and not zero — that floor *is* the finite-sample bias of comparing a real set against itself.

The operational consequence is a discipline, not a formula. **Pick one $N$, keep it fixed forever, run at least three seeds, and report the mean and spread.** Treat any FVD difference smaller than the seed-to-seed spread as zero. If you change $N$ — even once, even to "be more rigorous" — every historical comparison in your project silently breaks, because you have shifted the bias floor underneath all of them. The most common way teams fool themselves with FVD is not a wrong backbone or a bad reference set; it is quietly changing $N$ between experiments and reading the bias change as a quality change.

## 5. Computing FVD in PyTorch

Enough theory. Here is a real, runnable FVD computation. The structure is: load a pretrained I3D (we use the `torchmetrics` Fréchet Video Distance, which wraps a Kinetics-pretrained backbone and the Gaussian math), feed it real and fake clips as $(B, T, C, H, W)$ tensors, and read the score. I will then show the from-scratch version of the Gaussian-and-Fréchet step so the math is not a black box.

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance  # for reference / images
from torchmetrics.video import FrechetVideoDistance           # the video version

# FVD expects clips as (B, T, C, H, W), uint8 in [0, 255] or float in [0, 1]
# depending on the metric's `normalize` flag. We use float in [0, 1] here.
device = "cuda"
fvd = FrechetVideoDistance().to(device)   # wraps a pretrained I3D backbone

def add_clips(clips_real, clips_fake):
    # clips_*: (B, T, 3, H, W) float in [0, 1]
    fvd.update(clips_real.to(device), real=True)
    fvd.update(clips_fake.to(device), real=False)

# --- feed your evaluation set in batches ---
# Fix the protocol: N >= 2048 clips, fixed T (e.g. 16), fixed resolution.
N, T, H, W = 2048, 16, 224, 224
batch = 16
for _ in range(N // batch):
    real = sample_real_clips(batch, T, H, W)     # your data loader -> (B,T,3,H,W) in [0,1]
    fake = sample_generated_clips(batch, T, H, W) # your model's outputs, same shape
    add_clips(real, fake)

score = fvd.compute()   # a single scalar: lower is closer to the real distribution
print(f"FVD over N={N} clips, T={T}: {score.item():.1f}")
```

The one-line `fvd.compute()` hides the four steps from Section 3. Here is the Gaussian-fit-and-Fréchet-distance step written out, so you can see there is no magic — it is the closed-form formula, with a matrix square root computed via eigendecomposition for numerical stability:

```python
import torch

def fit_gaussian(feats: torch.Tensor):
    # feats: (N, D) I3D features for one set of clips
    mu = feats.mean(dim=0)                                  # (D,)
    centered = feats - mu
    sigma = centered.T @ centered / (feats.shape[0] - 1)    # (D, D) covariance
    return mu, sigma

def frechet_distance(mu_r, sigma_r, mu_g, sigma_g, eps=1e-6):
    # d^2 = ||mu_r - mu_g||^2 + Tr(Sr + Sg - 2 (Sr Sg)^{1/2})
    diff = mu_r - mu_g
    mean_term = diff @ diff

    # symmetric PSD matrix square root of the product Sr @ Sg, via eigendecomp
    prod = sigma_r @ sigma_g
    # symmetrize for numerical safety, then take the real PSD sqrt
    prod = 0.5 * (prod + prod.T)
    vals, vecs = torch.linalg.eigh(prod)
    vals = torch.clamp(vals, min=0.0)                       # kill tiny negatives
    sqrt_prod = (vecs * vals.sqrt()) @ vecs.T
    cov_term = torch.trace(sigma_r + sigma_g - 2.0 * sqrt_prod)

    return (mean_term + cov_term).clamp_min(0.0)

# usage, once you have I3D features for both sets:
mu_r, s_r = fit_gaussian(real_feats)   # real_feats: (N, D)
mu_g, s_g = fit_gaussian(fake_feats)   # fake_feats: (N, D)
fvd_value = frechet_distance(mu_r, s_r, mu_g, s_g).sqrt()  # report the distance
print(float(fvd_value))
```

Two engineering notes that bite people. First, **the matrix square root is where numerical errors live**. The product of two estimated covariance matrices is not guaranteed to be symmetric PSD after floating-point rounding, so you symmetrize and clamp eigenvalues at zero — skip that and you get NaNs or complex numbers. Reference implementations like the widely used `common_metrics_on_video_quality` repo do the same thing with a SciPy `sqrtm`; the eigendecomposition version above is the stable GPU-native equivalent. Second, **fix the seed and warm up**. Generate your evaluation clips with a fixed seed set so the comparison across checkpoints is on the *same prompts and same noise*; otherwise prompt and seed variance leaks into the FVD and you cannot tell a real improvement from a lucky draw.

#### Worked example: FVD that moved for the wrong reason

A team tightens their temporal-attention module and the flicker visibly drops. They re-run FVD and it goes *up*, from 212 to 231. Panic. The fix is to check what else changed: the new checkpoint also runs the VAE decoder at a slightly different `decode_chunk_size`, which introduced a faint brightness seam at chunk boundaries. I3D's content-vs-motion entanglement read that appearance artifact as a distributional shift and inflated the score even though the *motion* — the thing they fixed — got better. The lesson is not "FVD is useless." It is "FVD moved 19 points and we cannot attribute it without a controlled comparison." They fixed the seam, the FVD dropped to 196, and the flicker stayed fixed. This is the daily reality of using FVD: it is a sensitive instrument that does not tell you *which* knob you turned.

## 6. Per-frame metrics, and the artifacts they cannot see

People still reach for per-frame image metrics on video, and sometimes for good reason — but you must know exactly what they are blind to. The two common ones are **per-frame FID** (run FID on the pooled set of all individual frames) and **per-frame CLIP score** (average the CLIP image-text similarity across frames). Both inherit the permutation-invariance problem from Section 1.

![A vertical stack figure showing a clip being split into frames, scored per frame, then averaged into a permutation-invariant number that is blind to flicker](/imgs/blogs/the-metrics-of-video-generation-4.png)

Per-frame FID answers "are my individual frames, considered as a bag of images, distributed like real frames." That is a *real* and useful question — it catches per-frame quality collapse, blur, and color problems. But it is silent on everything temporal. A clip where frame 5 is a beautiful dog and frame 6 is a beautiful but *completely different* dog scores fine per-frame; the identity jump is invisible because the metric never compares frame 5 to frame 6. Flicker — high-frequency brightness or texture oscillation between frames — is similarly invisible, because each frame on its own is fine. The pathological case is the one from Section 3: shuffle the frames and per-frame FID does not move at all.

Per-frame CLIP score has the same blindness plus a more subtle failure. CLIP scores each frame's similarity to the text prompt and you average. This catches whether the *content* matches the prompt ("a dog running across a field") on a frame-by-frame basis. It cannot catch whether the *motion* matches — "running" versus "standing" is largely an inter-frame property, and CLIP sees one frame at a time. A perfectly still dog in a field scores nearly as high on per-frame CLIP as a running one, because each individual frame contains a dog in a field. So per-frame CLIP is a content-presence check, not a motion or temporal-coherence check. We will return to text-video alignment done *properly* in Section 9.

The honest place for per-frame metrics is as **cheap, fast guards** alongside the temporal metrics, never as a primary score. Per-frame FID will catch a frame-quality regression in minutes when FVD would take longer and might mask it. But if per-frame FID is your *only* number, you are flying blind to the entire category of bugs that distinguishes video from a slideshow. Here is the per-frame CLIP score, kept deliberately simple so you see it really is just averaging:

```python
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().cuda()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

@torch.no_grad()
def per_frame_clip_score(frames, prompt):
    # frames: list of PIL Images (the decoded video), prompt: str
    text = proc(text=[prompt], return_tensors="pt", padding=True).to("cuda")
    t_emb = F.normalize(clip.get_text_features(**text), dim=-1)   # (1, D)

    scores = []
    for f in frames:
        img = proc(images=f, return_tensors="pt").to("cuda")
        i_emb = F.normalize(clip.get_image_features(**img), dim=-1)  # (1, D)
        scores.append((i_emb @ t_emb.T).item())                     # cosine sim
    # the average is permutation-invariant: shuffle `frames` and this is identical
    return sum(scores) / len(scores)

# This rewards "a dog is present in each frame", NOT "the dog is running".
# Use it as a cheap content-presence guard, never as your motion metric.
```

The comment on the return line is the whole point. Shuffle `frames` and the number is unchanged. Anything that is unchanged under a shuffle cannot be your temporal-quality metric.

## 7. VBench: refusing to give you one number

The deep insight behind VBench (Huang et al., 2024, with VBench-2.0 extending it in 2025) is that *no single number can be both meaningful and un-gameable for video*, so it does not try. Instead it decomposes video quality into a set of named, separately-scored **dimensions**, each computed by a purpose-built detector, and validated to correlate with human judgment on that specific axis. You get a vector of scores, not a scalar, and the vector is the point: it makes the trade-offs *visible*.

![A vertical stack figure grouping VBench dimensions into consistency, temporal quality, dynamic degree, frame fidelity, and text-video alignment](/imgs/blogs/the-metrics-of-video-generation-5.png)

The dimensions cluster into a few groups. I will describe them in the language of "what detector computes this," because that is what determines what each one can and cannot see.

**Consistency dimensions.** *Subject consistency* asks whether the main subject stays the same entity across frames — typically measured with DINO or CLIP image features, checking that frame-to-frame feature similarity of the subject region stays high. *Background consistency* does the same for the scene behind the subject. High scores mean the dog stays the same dog and the field stays the same field. Critically — and this is the heart of the trap — these scores are *maximized by not changing anything*. A frozen frame has perfect subject and background consistency.

**Temporal quality dimensions.** *Temporal flickering* measures high-frequency fluctuation between consecutive frames (low flicker is good). *Motion smoothness* checks whether motion follows a physically smooth trajectory rather than jerking — often estimated by how well a frame-interpolation model can predict intermediate frames, which is high when motion is smooth and *also* high when there is barely any motion to predict. Again: a near-static clip aces both.

**Dynamic degree.** This is the dimension that exists *specifically to catch the cheat*. *Dynamic degree* measures how much actual motion is in the clip — typically via optical-flow magnitude (e.g. RAFT) thresholded into "this clip genuinely moves" versus "this clip is basically static." It is the antidote to consistency and smoothness: those reward stillness, dynamic degree punishes it. You cannot read consistency without reading dynamic degree, and we devote the next section to why.

**Frame fidelity dimensions.** *Aesthetic quality* (a learned aesthetic predictor) and *imaging quality* (a technical-quality / distortion predictor like MUSIQ) score how good each frame looks on its own. These are per-frame in spirit and carry per-frame blindnesses, but they catch blur, artifacts, and ugliness that the temporal metrics ignore.

**Text-video alignment.** Whether the video matches the prompt — both the content and, in VBench-2.0's expanded suite, the *action* and compositional structure. Measured with CLIP/ViCLIP-style video-text similarity and increasingly with VLM-as-judge prompting.

You do not compute these by hand; you run the VBench harness, which ships the detectors and a standardized prompt suite. The reading you *do* by hand — and this is the engineering skill — is interpreting the vector. Here is the shape of pulling dimension scores out of a VBench results file and reading them as a vector, with the one pairing rule baked in:

```python
import json

# VBench writes a results JSON per evaluated model; load and read the vector.
with open("results/my_model_t2v_vbench.json") as fh:
    res = json.load(fh)

# Each dimension maps to a normalized score in [0, 1]; higher is better
# EXCEPT you must read consistency/smoothness TOGETHER WITH dynamic_degree.
dims = {
    "subject_consistency":   res["subject_consistency"][0],
    "background_consistency":res["background_consistency"][0],
    "motion_smoothness":     res["motion_smoothness"][0],
    "dynamic_degree":        res["dynamic_degree"][0],   # the anti-cheat axis
    "aesthetic_quality":     res["aesthetic_quality"][0],
    "imaging_quality":       res["imaging_quality"][0],
    "temporal_flickering":   res["temporal_flickering"][0],
    "overall_consistency":   res["overall_consistency"][0],  # text-video align proxy
}

def trap_flag(d, dyn_floor=0.30):
    # A model that aces consistency but moves less than the floor is gaming it.
    high_consistency = d["subject_consistency"] > 0.95 and d["motion_smoothness"] > 0.97
    return high_consistency and d["dynamic_degree"] < dyn_floor

for k, v in dims.items():
    print(f"{k:24s} {v:.3f}")
print("MOTION-VS-STABILITY TRAP TRIGGERED:", trap_flag(dims))
```

The `trap_flag` function is the whole moral of the post compressed into five lines: **a model that tops consistency and smoothness while falling below a dynamic-degree floor is not a better video model — it is a still-image model wearing a video's clothes.**

## 8. The motion-vs-stability trap, made rigorous

Let us formalize why this trap is not a quirk but a *structural* property of how these metrics are built, so you understand it as a theorem rather than a heuristic.

![A two-column before-after figure showing a near-static model winning consistency and smoothness but failing dynamic degree, against a dynamic model with slightly lower consistency but real motion](/imgs/blogs/the-metrics-of-video-generation-6.png)

Define, for a clip $x$ with frames $x_1, \dots, x_T$, a **consistency** functional and a **dynamics** functional. Consistency is an average frame-to-frame *similarity* in some feature space $\phi$:

$$
C(x) = \frac{1}{T-1}\sum_{t=1}^{T-1} \operatorname{sim}\big(\phi(x_t), \phi(x_{t+1})\big).
$$

Dynamics is, roughly, an average frame-to-frame *difference* — for instance the mean optical-flow magnitude:

$$
D(x) = \frac{1}{T-1}\sum_{t=1}^{T-1} \big\lVert \text{flow}(x_t \to x_{t+1}) \big\rVert.
$$

Now look at the extreme: the **static clip** $x_t = x_1$ for all $t$. Every consecutive pair is identical, so $\operatorname{sim}(\phi(x_t),\phi(x_{t+1})) = 1$ for all $t$ and $C(x)$ is *maximal*. And the flow between identical frames is zero, so $D(x) = 0$ is *minimal*. The static clip is the unique global maximizer of consistency and the global minimizer of dynamics simultaneously. Motion smoothness behaves like consistency here — a frame-interpolation model predicts a static frame perfectly — so it is also maximized at the static clip.

This is the rigorous statement of the trap: **consistency, smoothness, and "no-flicker" are all maximized by the degenerate static clip, and dynamic degree is the only one of the common dimensions that the static clip *fails*.** Therefore any optimizer — a training objective, a checkpoint-selection rule, a benchmark-chaser tuning sampling — that maximizes consistency *without a dynamics constraint* has a global optimum at "stop moving." This is not a risk; it is where the gradient points. If you select checkpoints by subject-consistency alone, you will, run after run, drift toward stiller and stiller models, and your dashboard will show monotonic improvement while your videos become slideshows.

The trade-off is genuine, not just a measurement artifact — and that is the subtle part. Real motion *does* make a clip slightly less self-similar frame to frame, because things are actually moving and changing. So there is a true Pareto frontier: at a fixed model quality, pushing dynamics up costs you a little consistency, and pushing consistency up costs you dynamics. A good model sits high on the frontier (lots of motion at high consistency); a gamed model sits at the degenerate corner (perfect consistency, no motion). The metrics individually cannot tell a *frontier* model from a *corner* model — only reading $C$ and $D$ **together** distinguishes them. The single most important habit in video evaluation is: **never report or optimize a consistency/smoothness number without the dynamic-degree number sitting right next to it.**

#### Worked example: the still dog beats the running dog

Concretely, take our two checkpoints. Checkpoint B (near-static dog): subject consistency 0.992, background consistency 0.995, motion smoothness 0.991, temporal flickering 0.98 (low flicker), **dynamic degree 0.08**. Checkpoint A (running dog with a brief shear at second three): subject consistency 0.943, background consistency 0.961, motion smoothness 0.962, temporal flickering 0.94, **dynamic degree 0.71**. If you rank by an unweighted average of the *first four* dimensions, B wins 0.990 to 0.953. If you also computed FVD against a reference set that happens to contain a lot of slow footage, B can even win on FVD, because its frames are clean and its distribution sits in a low-motion pocket the I3D Gaussian covers. By every "stability" number, B is better.

Now the human eval: show 50 raters both clips and ask "which is a better video of a dog running across a field?" They pick A something like 90% of the time, because B is not running — it is a photo with a tremor. The dynamic degree of 0.08 versus 0.71 is the *only* automatic number that agreed with the humans. That is what dynamic degree is *for*, and it is why a consistency-only scorecard would have shipped you the wrong model. The fix in practice is a **dynamics floor**: reject any checkpoint with dynamic degree below a task-appropriate threshold *before* you compare anything else, then rank the survivors by the full basket.

### The frontier is a curve, not a point

It helps to picture the consistency-dynamics relationship as an actual curve and reason about where models sit on it. On the horizontal axis put dynamic degree $D$ (how much the clip moves) and on the vertical axis put consistency $C$ (how self-similar consecutive frames are). The degenerate static clip sits at the top-left corner: $D = 0$, $C = 1$. A chaotic, incoherent clip with random motion sits at the bottom-right: high $D$, low $C$. The interesting region is the *frontier* — the upper envelope of what is achievable at a given model quality. A better model pushes that whole envelope up and to the right: it can produce *more* motion at the *same* consistency, or the *same* motion at *higher* consistency. A worse model has a lower envelope.

This reframing dissolves a confusion that trips up a lot of people. They notice that adding motion lowers consistency and conclude "the metrics are just measuring opposite things, so the trade-off is fake." It is not fake — moving things really are less self-similar frame to frame, that part is real physics of the pixels. But the trade-off being real does *not* mean every point on it is equally good. A model at $(D=0.7, C=0.94)$ and a model at $(D=0.08, C=0.99)$ are not "two valid points on the same frontier" — the first is on a *higher* frontier (it achieves substantial motion at high consistency) and the second is at the degenerate corner of a *lower* one. The single-number consistency score cannot tell these apart because it only reads the vertical axis. Reading $C$ and $D$ together places the model on the plane, and *that* is what reveals whether you have a good model with real motion or a stiff one hiding behind a consistency score.

The practical move that follows: when you compare two checkpoints, plot them on the $(D, C)$ plane rather than ranking them on either axis alone. If checkpoint A dominates B — more motion *and* more consistency — ship A, no debate. If they are on the trade-off (A has more motion, B has more consistency), the decision is a *product* decision about how much motion your use case needs, and you make it deliberately with the dynamics floor encoding the answer, rather than letting an unweighted average make it for you by accident. The unweighted average always quietly favors the stiller model, because three or four stability dimensions outvote the single dynamics dimension.

## 9. Text-video alignment: from CLIP to VLM-as-judge

Alignment — does the video do what the prompt asked — is its own axis with its own evolution, and per-frame CLIP (Section 6) is only its weakest form. There are three tiers.

**Per-frame CLIP / image-text similarity.** Cheap, fast, and only checks content *presence* per frame. It confirms a dog and a field appear; it is largely blind to whether the dog is running, and it is blind to anything compositional ("a dog *behind* a tree," "a *red* ball then a *blue* ball"). Use it as a coarse guard.

**Video-text similarity (ViCLIP-style).** Models like ViCLIP and other video-language encoders embed the *whole clip* (with temporal modeling) and the text into a joint space, so similarity reflects motion and temporal content, not just per-frame appearance. This is a real improvement: "running" and "standing" land in different places because the video embedding sees the motion. VBench's text-alignment dimensions lean on this family. It is still a similarity score, so it is coarse on fine compositional structure and counting.

**VLM-as-judge.** The current frontier for alignment is to hand the generated clip (as sampled frames, or natively if the model is video-capable) plus the prompt to a strong vision-language model and *ask* it, with a rubric: "Does this video show a dog running across a field? Score the action match, the object match, and the scene match from 1 to 5, and explain." A capable judge model can reason about action, count, spatial relations, and temporal order in a way no similarity score can. This is the video analog of LLM-as-judge, and it inherits all of LLM-as-judge's caveats — position bias, verbosity bias, self-preference, and sensitivity to the rubric — so you pin the judge model and version, fix the rubric, and validate the judge against a human-labeled subset before you trust it. Here is the shape of a VLM-as-judge alignment call against the Claude API, which is well suited to this kind of rubric-scored multimodal judging:

```python
import base64, json, anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY in env

def vlm_judge_alignment(frame_paths, prompt, model="claude-opus-4-8"):
    # frame_paths: a few evenly-spaced frames sampled from the generated clip
    images = []
    for p in frame_paths:
        with open(p, "rb") as fh:
            b64 = base64.standard_b64encode(fh.read()).decode()
        images.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })

    rubric = (
        f'These frames are sampled in order from a generated video.\n'
        f'The prompt was: "{prompt}".\n'
        "Score 1-5 each: (a) object match, (b) action/motion match, "
        "(c) scene match, (d) temporal coherence across the frames. "
        'Return strict JSON: {"object":n,"action":n,"scene":n,"temporal":n,"why":"..."}'
    )

    msg = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": images + [{"type": "text", "text": rubric}]}],
    )
    return json.loads(msg.content[0].text)

# Pin model + version, fix the rubric, and validate against human labels
# on a held-out subset before trusting the judge's absolute scores.
```

The reason this is worth the cost and latency is that it is the only automatic method that genuinely evaluates *action* and *compositional* alignment — exactly the parts of the prompt that per-frame CLIP cannot see. The reason you do not trust it blindly is that it is an LLM with biases; treat its scores as a strong, validated *signal*, not ground truth.

## 10. Human evaluation and arena-style preference

Every automatic metric above is a proxy for the thing you actually want, which is: do people think this is a good video. So at some point you ask people. There are two shapes of human eval and you want both.

**Mean Opinion Score (MOS) / Likert rating.** Show raters a single clip and ask them to rate it on named axes — overall quality, motion realism, prompt adherence, artifacts — typically 1–5. This gives you absolute, per-axis numbers you can track over time. It is sensitive to rater calibration drift (raters anchor differently on different days), so you randomize, you include hidden reference clips of known quality as anchors, and you collect enough raters per clip (commonly 3–5) to average out individual idiosyncrasy.

**Pairwise / arena-style preference.** Show two clips for the same prompt and ask "which is better." Pairwise comparison is far more reliable than absolute rating because humans are much better at *comparing* than at *scoring on an absolute scale* — the cognitive task is easier and the answers are more consistent. Aggregate the win/loss records across many pairs into a ranking, often with an Elo or Bradley–Terry model exactly like a chess ladder or the LLM chatbot arenas. This is how the public video-generation arenas rank Sora, Veo, Kling, and the open models, and it is the closest thing the field has to ground truth. The catch is throughput and cost: human eval is slow and expensive, you cannot run it on every checkpoint, and it does not scale to a training loop. So you use it where it counts — on release candidates and to *validate your automatic metrics*.

That last phrase is the key discipline. Human eval's most valuable job is not to be your daily metric; it is to **calibrate your cheap metrics**. You run human pairwise preference on a batch of model pairs, then check: does FVD agree with the humans? does the VBench basket? does the VLM judge? Where they agree, you trust the cheap metric to stand in for humans between human evals. Where they disagree — as they did spectacularly for the still-dog checkpoint, where every stability metric disagreed with the humans — you have found a blind spot and you fix your scorecard (in that case, by adding the dynamics floor). Metrics that have been validated against human preference on *your* content are trustworthy; metrics you have never checked against humans are decoration.

A few hard-won notes on running human eval so the numbers mean something. **Show the clips, not stills** — raters judging from a contact sheet of frames will miss flicker and motion problems exactly the way per-frame metrics do, so play the actual video, loop it, and let them scrub. **Control the viewing conditions** — frame rate, resolution, and whether the clip autoplays all change ratings; a 24-fps clip judged at 12 fps in a janky web player will look worse than it is. **Randomize and blind** — never let raters see which model produced which clip, randomize left/right position in pairwise to kill position bias, and shuffle prompt order. **Use anchor clips** — seed the batch with a few clips of known quality (a great one, a deliberately broken one) so you can detect and discard raters who are clicking randomly or anchoring strangely. **Ask the right question** — "which is a better video of *the prompt*" beats "which do you like more," because the former forces raters to weight prompt-adherence and motion, while the latter drifts toward whichever clip is merely prettier per-frame, reintroducing the very bias you ran human eval to escape. And **measure agreement** — report inter-rater agreement; if raters disagree wildly on an axis, that axis is either ambiguous or your clips are genuinely borderline, and either way a single mean hides it.

The reason all this care is worth it: human pairwise preference is the *anchor* of the entire measurement stack. Every cheap metric is ultimately justified by "it correlates with human preference on our content." If your human eval is sloppy — stills instead of clips, biased questions, no anchors — then your anchor is wrong, and you will validate your cheap metrics against a distorted target and trust the wrong things. The most expensive mistake in video evaluation is not a noisy FVD; it is a sloppy human eval that quietly miscalibrates everything downstream of it. Spend the care here, because this is the one measurement that is supposed to be ground truth.

## 11. The metric scorecard: captures, blind spots, and how to game each

Here is the consolidated view — every metric, what it actually captures, what it is structurally blind to, and the specific cheat that beats it. Tape this above your desk.

![A five-row matrix figure pairing each video metric with what it captures, what it is blind to, and the specific move that games it](/imgs/blogs/the-metrics-of-video-generation-7.png)

| Metric | Captures | Blind to | Gamed by |
|---|---|---|---|
| **FVD (I3D)** | Motion realism + distributional fidelity | Per-prompt correctness; content-vs-motion confound | Matching the reference *domain* without improving motion; low-motion reference pockets |
| **FVD (VideoMAE)** | Motion realism, more general backbone | Same confound; different absolute scale | Same; not comparable to I3D-FVD |
| **Per-frame FID** | Per-frame appearance, blur, color, mode collapse | Flicker, identity jumps, all frame-order info | Sharpening frames; per-frame polish on a slideshow |
| **Per-frame CLIP** | Content *presence* per frame | Motion, action verbs, composition, temporal order | Putting the prompt's objects in every still frame |
| **Video-text (ViCLIP)** | Content + coarse action alignment | Fine composition, counting, exact order | Caption-bait scenes that hit keywords |
| **VLM-as-judge** | Action, composition, temporal reasoning | Judge biases (position, verbosity, self-preference) | Prompt-engineering the scene to flatter the judge |
| **Subject/bg consistency** | Identity & scene stability | *Whether the clip moves at all* | Generating near-static video |
| **Motion smoothness** | Jerk-free trajectories | Stillness (no motion to be smooth about) | Generating near-static video |
| **Temporal flickering** | High-freq frame oscillation | Identity drift; large-scale incoherence | Generating near-static video |
| **Dynamic degree** | Amount of motion | *Quality* of the motion | Adding random jitter to fake "motion" |
| **Aesthetic / imaging** | Per-frame prettiness / technical quality | Everything temporal | Polishing frames; over-smoothing |
| **Human MOS** | Absolute per-axis judgment | Rater drift; slow; expensive | Cherry-picking prompts shown to raters |
| **Human pairwise (Elo)** | Relative preference, ground-truth-ish | Cost; throughput; prompt-set bias | Cherry-picking the prompt set / opponents |

Read the "Gamed by" column top to bottom and a pattern jumps out: **almost every individual metric is beaten by a different cheat, and several are beaten by the *same* cheat — generate near-static video.** A model that wanted to top a naive leaderboard would generate sharp, still, prompt-keyword-stuffed frames: it would ace per-frame FID, per-frame CLIP, all three consistency/smoothness dimensions, and aesthetic quality, and only dynamic degree and human pairwise would catch it. That is precisely why your scorecard must include the metrics that catch the others' cheats. No single number is safe. A *basket chosen so the members' blind spots do not overlap* is.

## 12. Case studies: real numbers from shipped models

Concrete numbers ground all of this, with the standing caveat that **FVD across papers is not comparable** (different $N$, clip length, backbone, reference set) and VBench numbers are comparable only within the same VBench version and prompt suite. Treat absolute figures as approximate and the *relationships* as the lesson.

**VBench as the field's de-facto leaderboard.** Since its 2024 release, VBench (and VBench-2.0 in 2025) has become the standard multi-dimensional report card, and essentially every open frontier release — CogVideoX, HunyuanVideo, Wan 2.x — reports a VBench breakdown rather than a single FVD. CogVideoX (Yang et al., 2024) reported strong VBench totals in the high-70s to low-80s percent range on the standard suite, with the *interesting* story being the per-dimension shape: high consistency and smoothness, with the dynamic-degree dimension as the axis that separated genuinely-animated outputs from stiller ones. The very fact that the leading reports moved from "our FVD is X" to "here is our VBench radar" is the field internalizing this post's thesis — one number was never enough.

**The dynamic-degree caveat baked into the benchmark.** VBench's own authors flagged the consistency-vs-dynamics tension explicitly: a model can climb consistency and smoothness by reducing motion, so the benchmark *reports dynamic degree alongside them by design* and cautions readers not to celebrate consistency in isolation. When you see a model topping subject consistency, the first thing to check is its dynamic degree on the same row. Several early text-to-video models posted suspiciously high consistency precisely because they generated low-motion clips; the dynamic-degree column is what exposed it.

**Stable Video Diffusion and the I2V protocol.** SVD (Blattmann et al., 2023) was evaluated heavily with FVD and human preference for image-to-video, and its release made a methodological point that matters here: they leaned on *human preference studies* for the headline quality claims because FVD alone could not adjudicate the motion-vs-fidelity trade-offs raters cared about. The pattern — automatic metrics to filter, human preference to decide — is the one every serious release follows.

**Arena rankings for the closed frontier.** For models you cannot run yourself (Sora 2, Veo 3.1, Kling 3.0), FVD and VBench are often not even available — you have no access to generate the 2048-clip evaluation set under a controlled protocol. The community fell back on **arena-style pairwise human preference**, ranking these models by Elo from head-to-head votes. This is not a downgrade; for end-user quality it is arguably the *most* trustworthy signal, because it measures exactly what you want (do humans prefer it) without any backbone bias or gaming surface beyond prompt-set selection. The lesson: as models get better and more closed, the field leans *harder* on human pairwise preference and *less* on FVD.

| Model | Reported via | Headline signal | What the numbers hide |
|---|---|---|---|
| CogVideoX (2024) | VBench breakdown | High consistency + competitive total | Read dynamic degree to judge real motion |
| HunyuanVideo / Wan 2.x | VBench breakdown | Strong totals on standard suite | Cross-version VBench not comparable |
| SVD (2023) | FVD + human pref | Human-preferred I2V | FVD alone couldn't rank motion trade-offs |
| Sora 2 / Veo 3.1 / Kling 3.0 | Arena Elo | Pairwise human preference | No controlled FVD; prompt-set bias |

**Why the open reports moved to radar charts.** A telling detail in the CogVideoX, HunyuanVideo, and Wan reports is the *presentation*: they show VBench as a radar chart with one spoke per dimension, not as a leaderboard row with a single total. That is a deliberate communication choice and it encodes exactly the lesson of Section 7 — a radar chart makes a lopsided model *look* lopsided. A model that is all consistency and no dynamics shows up as a spike on the consistency spokes and a dent on the dynamics spoke; you see the shape and immediately distrust it. A single total would have averaged that shape into a flattering number. When a report gives you only a total and hides the per-dimension shape, treat that as a yellow flag and ask for the dynamic-degree spoke specifically — it is the one most likely to be quietly low.

**The reproducibility footnote nobody reads.** Every careful VBench or FVD report includes a methods footnote pinning the version, the prompt suite, the number of clips, and the backbone. Those footnotes are not boilerplate — they are the entire reason the numbers mean anything. Two of the most common ways published video numbers turn out to be incomparable are (1) a VBench *version* bump that re-tuned a detector, silently shifting a dimension's scale between two papers that both say "VBench," and (2) an unstated FVD clip length. When you cite or reproduce a number, copy the footnote, not just the figure. A video metric without its protocol is a temperature without its unit.

The through-line of every case: the field has *already* converged on this post's recommendation — no single automatic number, a multi-dimensional benchmark read with dynamic degree in view, and human preference as the deciding vote.

## 13. How to actually evaluate a model you're shipping

Here is the opinionated, runnable recipe. The goal is a scorecard whose members' blind spots do not overlap, gated so the still-dog model cannot sneak through, and validated against humans so you can trust it between expensive human evals.

![A tree figure showing the shipping evaluation basket branching into a distribution metric, paired VBench axes, and human checks](/imgs/blogs/the-metrics-of-video-generation-8.png)

The basket, in priority order:

1. **A dynamics floor, applied first as a gate.** Before you compare anything, reject any checkpoint whose dynamic degree falls below a task-appropriate threshold. For a "running dog" task that floor is high; for a "slow product turntable" task it is low. This single gate kills the entire family of cheats that generate near-static video. It is the most important line in your eval and it is one threshold.

2. **FVD under a frozen protocol, as a relative signal.** Fix $N \ge 2048$, fix clip length, fix fps, fix resolution, fix the backbone (I3D *or* VideoMAE, and label which), fix the seed set, and keep the reference set in your generations' domain. Use FVD to rank checkpoints *within this frozen setup* and never across setups. Run multiple seeds and treat sub-5% differences as noise.

3. **The relevant VBench dimensions, read in pairs.** You do not need all of them; pick the axes your product cares about and *always* pair the stability ones with dynamic degree. For most T2V work that is: subject + background consistency (paired with dynamic degree), motion smoothness (paired with dynamic degree), temporal flickering, imaging quality, and a text-alignment dimension.

4. **Text-video alignment, escalating by need.** Per-frame CLIP as a cheap content guard; ViCLIP-style video-text similarity for action; VLM-as-judge (validated against humans) for action and composition on your hardest prompts.

5. **A human spot-check, every release candidate.** A fixed set of ~30 prompts spanning your use cases, run as pairwise comparison against the current production model and the previous candidate. This is your ground truth and your metric-validation set in one. If a candidate wins your automatic basket but loses the human pairwise, you have found a blind spot — fix the scorecard, do not ship the candidate.

```python
def ship_eval(checkpoint, ref_set, prompts, prod_model):
    # 1) DYNAMICS GATE — first, non-negotiable
    vb = run_vbench(checkpoint, prompts)          # VBench harness -> dim scores
    if vb["dynamic_degree"] < task_dynamic_floor(prompts):
        return REJECT("near-static: failed dynamics gate")

    # 2) FVD under a FROZEN protocol (N>=2048, fixed T/fps/res/backbone/seeds)
    fvd = compute_fvd(checkpoint, ref_set, n=2048, frames=16, backbone="i3d",
                      seeds=range(3))             # report mean +/- spread
    # 3) VBench axes read in PAIRS (stability always next to dynamic_degree)
    scorecard = {
        "fvd_mean": fvd.mean, "fvd_std": fvd.std,
        "consistency": (vb["subject_consistency"], vb["background_consistency"]),
        "dynamic_degree": vb["dynamic_degree"],   # ALWAYS shown with consistency
        "smoothness": vb["motion_smoothness"],
        "flicker": vb["temporal_flickering"],
        "imaging": vb["imaging_quality"],
        # 4) alignment, escalating
        "clip_frame": per_frame_clip(checkpoint, prompts),
        "vlm_align": vlm_judge_batch(checkpoint, prompts),  # validated judge
    }
    # 5) HUMAN pairwise vs production on a fixed 30-prompt set — the deciding vote
    human = human_pairwise(checkpoint, prod_model, fixed_30_prompts())
    scorecard["human_winrate_vs_prod"] = human.winrate

    # SHIP rule: pass the gate, win or tie the basket, AND win human pairwise.
    return decide(scorecard)
```

Notice the structure mirrors the case studies in Section 12: a gate that kills the dominant cheat, a relative distribution metric, a paired multi-dimensional read, escalating alignment, and human preference as the final word. This is not theory — it is the workflow the shipped models' release reports describe, reverse-engineered into something you can run on Monday. The full production pipeline that this slots into — model selection, fine-tuning, sampler and caching choices, the works — is the subject of the capstone, [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook). And the red-teaming and provenance side of evaluation — deepfakes, consent, C2PA watermarking, the safety stack on top of quality — gets its own deep dive in [evaluating and red-teaming video generation](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation).

### Making the basket cheap enough to run often

The recipe in the previous section is correct but, run naively, it is too slow to use during training — and a metric you only run at the end is a metric that lets you discover regressions far too late. The engineering problem is to make a *fast approximation* of the basket that you can run every few thousand steps, while reserving the *full* basket for release candidates. The structure is a tiered cadence.

The **every-N-steps tier** is the cheapest signals only: per-frame FID and per-frame CLIP on a small fixed prompt set, plus dynamic degree (optical-flow magnitude is fast). This catches three of the most common regressions early — frame-quality collapse, content drift, and the model going static — for a tiny fraction of the cost of full FVD. Crucially, dynamic degree belongs in this tier *because* the static-drift failure can creep in slowly during training, and you want to catch the first step where dynamic degree starts sliding, not discover at the end that the last 20k steps quietly killed your motion. The cost of this tier is dominated by generating the clips, not scoring them, so keep the prompt set small (32–64 prompts) and the clips short.

The **every-checkpoint tier** adds FVD under the frozen protocol (the expensive part: generating 2048 clips) and the full VBench dimension run. You do this on saved checkpoints rather than every training step, because generating a 2048-clip evaluation set is itself a meaningful compute cost — on a single A100 a 5-second 720p clip can take tens of seconds, so 2048 of them is hours. Budget for it and run it on a cadence you can afford (say, every checkpoint you would actually consider keeping).

The **release-candidate tier** is the full basket including VLM-as-judge on the hard prompts and human pairwise preference. This is the slow, expensive, deciding tier, and you run it only on the handful of candidates that survived the cheaper tiers.

```python
def eval_cadence(step, checkpoint):
    report = {}
    # TIER 1 — every N steps: cheap guards + the static-drift early warning
    if step % 2000 == 0:
        report["frame_fid"]  = per_frame_fid(checkpoint, small_prompts)   # fast
        report["frame_clip"] = per_frame_clip(checkpoint, small_prompts)  # fast
        report["dynamic"]    = dynamic_degree(checkpoint, small_prompts)  # flow mag
        if report["dynamic"] < trend_floor(step):     # is motion sliding over time?
            alert("dynamic degree trending down — model going static")

    # TIER 2 — every kept checkpoint: the expensive frozen-protocol metrics
    if is_keepable(step):
        report["fvd"]    = compute_fvd(checkpoint, ref_set, n=2048, seeds=range(3))
        report["vbench"] = run_vbench(checkpoint, vbench_prompts)  # full dimensions

    # TIER 3 — release candidates only: judge + humans decide
    if is_release_candidate(checkpoint):
        report["vlm"]   = vlm_judge_batch(checkpoint, hard_prompts)
        report["human"] = human_pairwise(checkpoint, prod_model, fixed_30_prompts())
    return report
```

The single most valuable line in that function is the `dynamic degree trending down` alert, because the static-drift failure is *insidious*: it improves your consistency numbers while it happens, so any dashboard that does not watch dynamic degree as a time series will show you a model getting "better" right up until you watch a clip and realize it stopped moving 30k steps ago. Watching dynamic degree as a *trend*, not just a threshold, is how you catch the trap forming instead of discovering it after the fact.

#### Worked example: catching the static drift at step 84k

A run looked healthy on the dashboard: per-frame FID flat, subject consistency climbing from 0.96 to 0.98 over 100k steps, motion smoothness climbing too. The Tier-1 dynamic-degree trend told the real story — it slid from 0.64 at step 40k to 0.31 at step 110k, crossing the trend floor around step 84k. The "improving consistency" was the model learning to move less. Rolling back to the step-80k checkpoint and adding a small motion-preserving term to the objective recovered dynamic degree to 0.61 with consistency at 0.97 — a real point on the frontier instead of a slide toward the degenerate corner. Without the dynamic-degree time series, this run would have trained 30k extra steps in the wrong direction and shipped a stiller model that the consistency dashboard called the best checkpoint yet.

## 14. Stress-testing the recipe

A good evaluation recipe should survive contact with the failure modes that break naive ones. Let us stress it.

**What happens when motion is large between frames?** FVD's I3D backbone was trained on Kinetics, where motion is mostly human-scale and continuous. A clip with very large inter-frame motion — fast camera whip, rapid scene cuts — can land out of I3D's comfortable distribution and inflate FVD even when the motion is *correct and intended*. The recipe survives because FVD is only one member and only a relative signal; dynamic degree confirms the motion is real, the VLM judge confirms it matches the prompt, and human pairwise adjudicates whether the fast motion looks good. No single metric is asked to carry the judgment.

**What happens at long clip lengths?** FVD over 16-frame windows says nothing about whether identity drifts over 30 seconds — the window is too short to see the drift. This is the autoregressive-rollout problem from [the long-video post](/blog/machine-learning/video-generation/why-video-generation-is-hard) showing up in the *metric*: a window-based metric is blind to slow drift across windows. The recipe's answer is subject-consistency measured over the *full* clip (not just a window) plus a human spot-check specifically on long clips, because identity drift is exactly the artifact humans notice instantly and short-window FVD misses entirely.

**What happens when the reference set is wrong?** If you evaluate an anime model against a photoreal reference set, FVD is measuring "distance from photoreal," which is not what you want — your good anime clip looks far from the reference and scores terribly. The recipe survives only if you respect step 2's discipline: the reference set must be in the generations' domain. Get this wrong and FVD becomes actively misleading. This is the single most common FVD mistake in practice, and the recipe's defense is the protocol rule, not a metric.

**What happens when someone games it on purpose?** Suppose an adversary wants to top your leaderboard. They generate sharp, still, keyword-stuffed clips. Walk it through the gate: the dynamics floor rejects them at step 1 before any other metric is even computed. That is the whole reason the dynamics floor is *first*. If they instead add random jitter to fake motion and clear the dynamics floor, the motion-smoothness dimension and the VLM judge catch the jitter as incoherent, and human pairwise buries it. The basket is designed so that beating one member exposes you to another.

#### Worked example: a metric improvement that wasn't

A team reports "+3.1 VBench points" between two checkpoints and wants to ship. Decompose the gain: subject consistency +0.04, motion smoothness +0.03, temporal flickering +0.02 — and dynamic degree **−0.21**. The "+3.1 points" was an unweighted average that *summed stability gains while hiding a large motion loss*. Read in pairs, the story flips: the new checkpoint is stiller, and the consistency gains are the stillness, not a real improvement. Human pairwise confirms it — raters prefer the *old* checkpoint 2-to-1 because it moves. The "+3.1 points" was the still-dog trap wearing a percentage. This is why you never ship on an aggregate VBench number and always read the dynamic-degree delta next to the consistency delta.

## 15. When to reach for each metric (and when not to)

A decisive guide, because the failure mode is using the wrong metric for the wrong question.

**Reach for FVD** when you are comparing checkpoints of *your own* model under a controlled protocol and want a single sensitive signal for "is this distribution closer to real." **Do not** use FVD to compare against another paper's number, to grade absolute quality, or as your *only* metric — its content-vs-motion confound and sample-size noise make it unreliable alone.

**Reach for per-frame FID/CLIP** as cheap, fast guards that run on every checkpoint to catch per-frame quality and content-presence regressions in minutes. **Do not** ever treat them as temporal-quality metrics — they are provably blind to flicker, identity jumps, and frame order.

**Reach for VBench** when you want the multi-dimensional report card that makes trade-offs visible and is validated per-axis against humans. **Do not** collapse it to a single average — that average is exactly where the motion-vs-stability trap hides — and do not compare across VBench versions.

**Reach for dynamic degree** always, as a *gate* and as the mandatory companion to every consistency and smoothness number. **Do not** ever report consistency or smoothness without it; alone, it is meaningless because the static cheat tops it.

**Reach for VLM-as-judge** for action and compositional alignment on your hardest prompts, after validating the judge against human labels. **Do not** trust its absolute scores unvalidated — it is an LLM with position, verbosity, and self-preference biases.

**Reach for human pairwise preference** for every release candidate, as ground truth and as the validator for all your cheap metrics. **Do not** try to run it on every checkpoint (too slow and expensive) or treat absolute MOS as more reliable than pairwise (it is not — humans compare better than they score).

There is also a decision about *how much* evaluation to build, and the honest answer is "less than you think, but the right less." If you are fine-tuning an open model for a narrow use case — say, product turntables — you do not need the full frontier eval stack; you need a dynamics floor appropriate to turntables (low, because turntables move slowly), per-frame quality guards, and a 20-clip human spot-check. Building a 2048-clip FVD harness for that job is over-engineering. Conversely, if you are training a general T2V model from scratch and competing on quality, skimping on the basket will burn you — you will chase a leaderboard number into the static trap. Match the *weight* of your evaluation to the stakes: the gate and a human spot-check are non-negotiable at any scale, and everything else scales with how much you are betting on the model.

The meta-rule: **match the metric to the question, gate on dynamics, and let no single number make the ship decision.** The cost of getting this wrong is not abstract — it is shipping the still dog.

## 16. Key takeaways

- **Per-frame metrics are permutation-invariant over time, so they are provably blind to flicker, identity jumps, and every temporal artifact.** Averaging a per-frame score over a clip cannot see motion; that one fact forks the whole field.
- **FVD is FID with a 3D-conv backbone.** It fits Gaussians to I3D (or VideoMAE) spatiotemporal features and reports their Fréchet distance; the 3D convolution is what gives it a temporal receptive field and lets it see motion that Inception cannot.
- **FVD is noisy by construction.** Sample-size bias (always inflating, shrinking with $N$), clip-length sensitivity, backbone bias, and content-vs-motion entanglement mean it is a *relative* signal under a frozen protocol, never a comparable absolute grade.
- **VBench refuses to give one number** and instead scores named axes — consistency, smoothness, flicker, dynamic degree, aesthetic/imaging quality, text alignment — each validated against humans. The vector is the point: it makes trade-offs visible.
- **The motion-vs-stability trap is a theorem, not a quirk.** Consistency, smoothness, and no-flicker are all *maximized by the degenerate static clip*; only dynamic degree fails it. Any optimizer that maximizes consistency without a dynamics constraint converges on "stop moving."
- **Never report consistency or smoothness without dynamic degree beside it.** A near-static model can win every stability number and even FVD while losing dynamic degree and human preference — the still dog beats the running dog on the dashboard and loses with the humans.
- **Text alignment escalates:** per-frame CLIP checks content presence, ViCLIP-style video-text similarity adds action, VLM-as-judge adds composition and temporal reasoning — but the judge is an LLM and must be validated against humans.
- **Human pairwise preference is the closest thing to ground truth** and its highest-value job is calibrating your cheap metrics, not being your daily number.
- **Ship on a basket, gated on dynamics.** Dynamics floor first, FVD under a frozen protocol as a relative signal, paired VBench axes, escalating alignment, and human pairwise as the deciding vote. Choose members whose blind spots do not overlap.

## 17. Further reading

- **Unterthiner et al., "Towards Accurate Generative Models of Video: A New Metric & Challenges" (2018)** — the paper that introduced FVD: the Fréchet distance on I3D features and the case for a video-native metric.
- **Carreira & Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (2017)** — the I3D inflated-3D-ConvNet backbone that FVD's features come from, and why Kinetics training shapes (and biases) the metric.
- **Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (2017)** — the original FID, the Gaussian-and-Fréchet idea FVD generalizes; read alongside the image-series [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly).
- **Huang et al., "VBench: Comprehensive Benchmark Suite for Video Generative Models" (2024)** and **VBench-2.0 (2025)** — the multi-dimensional benchmark, its per-axis detectors, and the explicit dynamic-degree-vs-consistency caveat.
- **Tong et al., "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training" (2022)** — the self-supervised backbone increasingly used as an FVD feature extractor in place of I3D.
- **Blattmann et al., "Stable Video Diffusion" (2023)** and **Yang et al., "CogVideoX" (2024)** — shipped models whose evaluations show the FVD-plus-VBench-plus-human pattern in practice.
- **🤗 `diffusers` video pipeline docs** and the `common_metrics_on_video_quality` / `torchmetrics` FVD implementations — the real toolchain for computing these metrics.
- **Within series:** [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), [evaluating and red-teaming video generation](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation), and the capstone [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
