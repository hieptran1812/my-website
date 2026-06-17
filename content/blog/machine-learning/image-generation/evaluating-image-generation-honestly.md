---
title: "Evaluating Image Generation Honestly: What FID, CLIP-Score, and Human Preference Actually Measure"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A practitioner's guide to the metrics that judge image generators — how FID is computed and why it lies, where CLIP-score is blind, how precision/recall expose mode collapse, what GenEval and human-preference models add, and how to actually evaluate a model you're about to ship."
tags:
  [
    "image-generation",
    "diffusion-models",
    "evaluation-metrics",
    "fid",
    "clip-score",
    "human-preference",
    "geneval",
    "generative-ai",
    "deep-learning",
    "benchmarking",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/evaluating-image-generation-honestly-1.png"
---

Two teams hand you the same text-to-image model checkpoint with one difference: a fine-tune. Team A reports **FID 7.9** and a press release. Team B reports **FID 8.4** and a sheepish shrug. The numbers say ship A. So you put both behind a blind A/B test with real users, 1,000 prompts each, and the result lands the other way: users pick **B 62% of the time**. You dig in, and the reason is almost insulting in its simplicity — when the prompt says "three red apples," model A reliably draws four or five apples in beautiful light, and model B draws exactly three. FID, the number that crowned A, **never read a single caption**. It compared two clouds of feature statistics and declared the prettier cloud the winner. The metric was not wrong; it was answering a question nobody asked.

This is the central, uncomfortable truth of generative-image evaluation: **every metric we have lies in its own specific way**, and the only defense is to know exactly *how* each one lies so you can assemble a basket that covers each other's blind spots. A single number cannot simultaneously tell you whether your samples look real, whether they cover the diversity of the training distribution, and whether they actually depict what the prompt asked for. Those are three different axes — fidelity, diversity, and alignment — and the headline leaderboard number you are tempted to optimize collapses them into one scalar that throws away the very information you need to make a shipping decision. Figure 1 is the map for the whole post: rows are the metrics, columns are what each one captures and what it is structurally blind to.

![A matrix showing four metric families with the axis each one captures and the failure each one is blind to](/imgs/blogs/evaluating-image-generation-honestly-1.png)

This post is the evaluation deep-dive for the series. We built the probabilistic foundation earlier — [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) introduced FID, Inception Score, CLIP-score, and precision/recall at the level of "here is the definition." This post is the practitioner's sequel: how FID is *actually computed* in code, the four distinct ways it is biased (sample size, the ImageNet-Inception backbone, resize/JPEG sensitivity, and the Gaussian assumption), the modern fixes (FID-DINOv2 and CMMD), why Inception Score is obsolete, how precision/recall and density/coverage split fidelity from diversity, exactly why CLIP-score cannot count, what the compositional benchmarks (GenEval, T2I-CompBench, DPG-Bench) add, how learned human-preference models (HPSv2, ImageReward, PickScore) are trained and where they overfit, and finally — the part that matters most — **how to actually evaluate a model you are about to ship**. You will leave able to compute every one of these metrics in PyTorch, read a leaderboard with the skepticism it deserves, and design an evaluation that does not lie to you.

By the end you will be able to: derive the FID formula and explain its sample-size bias from the covariance estimator; compute FID with `torchmetrics` and `clean-fid`, CLIP-score with `open_clip`, and an ImageReward/HPSv2 score, all in a few lines; demonstrate the sample-size sensitivity yourself; design a compositional probe that catches a counting failure FID would miss; and assemble a shipping-grade evaluation basket. We tie this back to the series' spine — the generative trilemma of **quality × diversity × speed** — because evaluation is literally how you *see* each corner of that trilemma. Get the metrics wrong and you will spend a month fine-tuning toward a number that is quietly making your model worse.

## 1. The one rule: a metric measures a proxy, not the thing you care about

Before any formula, internalize the meta-principle, because it explains every failure that follows. **You care about whether a human, shown your image and a prompt, would say "yes, that's a good picture of that."** That is the target. It is expensive (humans, time, money) and noisy (taste varies). So every automatic metric is a *proxy*: a cheap, deterministic computation that correlates — to varying and often surprisingly weak degrees — with that human judgment. The proxy is useful exactly to the extent that the correlation holds, and dangerous exactly where it breaks.

FID is a proxy for "do my samples look like real images, as a distribution." CLIP-score is a proxy for "does the image match the prompt." HPSv2 is a learned proxy for "would a human prefer this." Each proxy was fit (or designed) against some slice of human judgment, and each generalizes poorly outside that slice. The error is never "the metric is broken." The error is **using a proxy outside the regime where it tracks the target**, then optimizing against it until you have driven the proxy up and the target down. This is Goodhart's law in its purest form: *when a measure becomes a target, it ceases to be a good measure.* The entire eval crisis we discuss in section 10 is Goodhart's law applied to FID.

The practical consequence: never trust one proxy. The reason a basket of metrics works is that the regimes where each lies are *different*. FID is blind to whether the image matches the caption; CLIP-score covers that. CLIP-score is blind to counting; GenEval covers that. GenEval is blind to aesthetics; HPSv2 covers that. HPSv2 is blind to factual correctness and can be gamed by over-saturated, high-contrast "metric-pretty" images; human spot-checks cover that. Stack the proxies so their blind spots do not overlap, and the basket as a whole tracks the target far better than any single number. That stacking is the whole thesis of this post, and we will build it piece by piece.

A useful frame for the rest of the post: every metric we discuss answers a *yes/no* form of one of four questions. *Do my images look real?* (FID, CMMD, IS, precision). *Does my model cover the full variety of real images?* (recall, coverage). *Does the image depict what the prompt asked for?* (CLIP-score, GenEval, T2I-CompBench, DPG-Bench). *Would a person prefer this image?* (HPSv2, ImageReward, PickScore, human arena). Notice that no metric answers more than one of these well — the ones that try (a single FID standing in for "is this model good") are exactly the ones that mislead. Keep the four questions in view and you will always know which sensor you are missing.

## 2. Fréchet Inception Distance, derived

FID is the workhorse of the field, so we derive it properly. The idea is two moves. **Move one:** real images and generated images are both pushed through a fixed, pre-trained image classifier — InceptionV3 trained on ImageNet — and we keep the 2048-dimensional activation from the final pooling layer (`pool3`). This turns each image into a feature vector that captures high-level content rather than raw pixels. **Move two:** we model each set of feature vectors as a multivariate Gaussian, then measure the distance between the two Gaussians. Figure 2 traces that pipeline end to end.

![A graph of the FID computation pipeline with real and generated images merging into the Inception feature extractor then a Frechet distance](/imgs/blogs/evaluating-image-generation-honestly-2.png)

Why model the features as Gaussians? Because there is a closed-form distance between two Gaussians — the **2-Wasserstein distance**, also called the Fréchet distance — and it only needs the mean and covariance of each. Let the real features have mean `$\mu_r$` and covariance `$\Sigma_r$`, and the generated features have mean `$\mu_g$` and covariance `$\Sigma_g$`. The Fréchet distance between the two multivariate Gaussians is:

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \operatorname{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Read the two terms physically. The first term, `$\|\mu_r - \mu_g\|_2^2$`, is how far apart the *average* feature vectors are — a coarse measure of "are these the same kind of image on average." The second term is the covariance mismatch: `$\operatorname{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$` is zero exactly when `$\Sigma_r = \Sigma_g$` and grows as the two feature *spreads* (the shape and correlation structure of the feature cloud) diverge. The matrix square root `$(\Sigma_r \Sigma_g)^{1/2}$` is the cross term that couples the two covariances; computing it is the numerically delicate part (`scipy.linalg.sqrtm` on a 2048×2048 matrix). Lower FID means the two Gaussians are closer, which we *interpret* as "the generated distribution matches the real distribution." That interpretation is exactly where the lies start.

Here is a derivation worth carrying: the second term is the trace of the squared **Bures metric** between the two covariances, and for the special case of commuting (simultaneously diagonalizable) covariances with eigenvalues `$\lambda_i^r$` and `$\lambda_i^g$`, it reduces to `$\sum_i (\sqrt{\lambda_i^r} - \sqrt{\lambda_i^g})^2$` — a sum over feature directions of the squared difference of standard deviations. This makes the bias source vivid: FID is comparing the *standard deviations* of the two feature clouds, direction by direction, and those standard-deviation estimates are exactly what a small sample gets wrong.

It is worth being explicit about *why* the Fréchet distance is the natural choice once you have committed to the Gaussian model. The 2-Wasserstein distance `$W_2(P, Q)$` between two distributions is the minimum cost of transporting the mass of `$P$` onto `$Q$`, with cost equal to squared Euclidean distance moved. For two arbitrary distributions this is an intractable optimal-transport problem. But for two Gaussians `$P = \mathcal{N}(\mu_r, \Sigma_r)$` and `$Q = \mathcal{N}(\mu_g, \Sigma_g)$`, the optimal transport map is *affine* and `$W_2^2$` has the closed form written above — that is the entire reason FID fits Gaussians. So FID is precisely "the 2-Wasserstein distance between the Gaussian approximations of the two feature distributions." Every word in that sentence is a modeling choice you are silently accepting: *Gaussian* (Bias 4), *of the feature distributions* under InceptionV3 (Bias 2), estimated from finite samples (Bias 1). Naming the assumptions is half of evaluating honestly.

One more property worth internalizing: FID is **not symmetric in its sensitivity** to the two error types. Because the mean term is squared Euclidean and the covariance term is a Bures distance, FID is quite sensitive to a *shift* in the average feature (everything tinted blue, a systematic blur) but comparatively *insensitive to dropped modes* as long as the surviving modes keep the overall mean and covariance roughly intact. That asymmetry is exactly why a mode-collapsed model can post a deceptively decent FID — the collapse changes higher moments more than it changes the first two, and FID only reads the first two. Hold that thought; it is the entire motivation for precision/recall in section 6.

#### Worked example: computing FID in code

Here is the real `torchmetrics` path. Note the dtype: `FrechetInceptionDistance` expects `uint8` images in `[0, 255]`, and silently produces nonsense if you hand it normalized floats.

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

# feature=2048 selects the pool3 layer; normalize=False expects uint8 [0,255]
fid = FrechetInceptionDistance(feature=2048, normalize=False).cuda()

# real_loader and fake_loader yield uint8 tensors [B, 3, 299, 299]
for real_batch in real_loader:
    fid.update(real_batch.cuda(), real=True)
for fake_batch in fake_loader:
    fid.update(fake_batch.cuda(), real=False)

print(f"FID = {fid.compute().item():.3f}")
```

The `update`/`compute` split matters: `update` accumulates the running sum and sum-of-outer-products needed for `$\mu$` and `$\Sigma$`, and `compute` does the matrix square root once at the end. Internally this is the formula above. Two images sets in, one scalar out. Clean, fast, and — as we will now see — biased in four independent ways.

## 3. The four ways FID lies

FID earned its place because it correlates with human judgment of *realism* better than pixel metrics, and it is sensitive to many failure modes (blur, artifacts, mode dropping). But it carries four structural biases, and each one will burn you if you do not control it.

**Bias 1: sample-size dependence.** This is the one that silently invalidates most cross-paper comparisons. The covariance estimate `$\hat{\Sigma}$` from `$N$` samples is biased: with finite `$N$`, the sample covariance systematically *underestimates* the small eigenvalues and the trace term in FID picks up an additive bias that scales like `$O(1/N)$`. The practical effect is that **the very same model reports a lower (better) FID simply by generating more samples**. A model at FID 11 on 1k samples might be FID 3 on 50k samples — same model, same weights. Figure 3 shows the curve as a matrix of sample counts versus reported FID.

![A matrix of sample counts versus reported FID showing the score shrinking as more samples are drawn](/imgs/blogs/evaluating-image-generation-honestly-3.png)

The numbers in Figure 3 are illustrative of the well-documented shape (Chong & Forsyth's 2020 "Effectively Unbiased FID" paper measured exactly this monotone decay and proposed extrapolating to `$N \to \infty$` to get `$\overline{\text{FID}}_\infty$`). The rule that falls out: **FID is only comparable at a fixed sample count against a fixed reference set.** The community convention is 50,000 generated samples against the full reference set (e.g., all of FFHQ, or ImageNet's 50k validation images, or COCO's 30k captions). If a paper reports FID at 10k and you reproduce at 50k, you are not comparing the same quantity. Always state your `$N$` and your reference set; a FID without those two facts is meaningless.

Where does the `$O(1/N)$` bias come from, mechanically? The sample covariance `$\hat{\Sigma} = \frac{1}{N-1}\sum_i (x_i - \hat{\mu})(x_i - \hat{\mu})^\top$` is an unbiased estimator of `$\Sigma$`, but FID is a *nonlinear* function of `$\hat{\Sigma}$` — it takes a matrix square root. By Jensen's inequality, a nonlinear function of an unbiased estimate is generally *biased*, and a second-order Taylor expansion of the `$\operatorname{Tr}(\cdots)^{1/2}$` term around the true `$\Sigma$` shows the leading bias is proportional to the variance of `$\hat{\Sigma}$`, which scales as `$1/N$`. Concretely, the estimation noise in `$\hat{\Sigma}$` adds spurious "spread" that the matrix-square-root term reads as extra distance, so finite-`$N$` FID is *inflated*, and the inflation decays like `$1/N$`. That is why Chong & Forsyth's fix is to compute FID at several values of `$N$`, fit a line in `$1/N$`, and read off the intercept at `$1/N = 0$` — the extrapolated `$\overline{\text{FID}}_\infty$` is the bias-free estimate. The practical takeaway is blunter: never compare FIDs computed at different sample counts, and if you cannot match a paper's `$N$`, do not cite its FID as a baseline.

#### Worked example: how a 50k-vs-10k mismatch fakes a win

You are reproducing a paper that reports FID 3.6 at 10k samples for its model. Your model is genuinely *better*, with a true `$\overline{\text{FID}}_\infty$` of 3.1. But you evaluate at the proper 50k and get FID 2.6, while their 10k number sits at 3.6. You report "we beat them, 2.6 vs 3.6." A reviewer reruns *their* model at 50k and it drops to 2.9 — now your margin is 2.6 vs 2.9, a third of what you claimed, and well inside estimator noise. Half of your "improvement" was the sample-count bias, not the model. The reverse failure is worse: if *you* had evaluated at 10k and they at 50k, your better model would have *lost* on paper. This is not a hypothetical; it is the single most common way FID comparisons in the wild are silently wrong. Fix the protocol or do not report the number.

**Bias 2: the ImageNet-Inception backbone.** The feature extractor is InceptionV3 trained to classify the 1,000 ImageNet categories. Its features are tuned to discriminate dog breeds and vehicle types. This bakes in two problems. First, **domain mismatch**: if you generate faces, or medical images, or anime, the Inception features are a poor basis — the network was never asked to tell two faces apart at the granularity that matters, so FID on faces is noisier and less aligned with human perception than FID on ImageNet-like photos. Second, **adversarial blind spots**: because the features are a fixed, finite-capacity classifier, you can produce images that move the FID without improving perceived quality, by pushing on feature directions the classifier happens to encode. FID rewards "looks like ImageNet textures" as much as "looks like a good image."

**Bias 3: resize and compression sensitivity.** This one is shockingly large and routinely corrupts reported numbers. InceptionV3 wants 299×299 input. How you resize from your native resolution to 299 matters: bilinear vs bicubic vs Lanczos, antialias on or off, PIL vs OpenCV vs PyTorch — these can shift FID by **several points**, which is larger than the gap between competing models. Parmar, Zhang & Zhu's 2022 "On Aliased Resizing and Surprising Subtleties in GAN Evaluation" paper (the `clean-fid` paper) showed that the standard resizing in many codebases introduces aliasing artifacts that systematically bias FID, and that saving as JPEG (even at quality 95) versus PNG shifts the number too. The fix is `clean-fid`, which standardizes the resize and image I/O:

```python
from cleanfid import fid

# Both folders read with antialiased Lanczos resize, no JPEG surprises
score = fid.compute_fid("path/to/generated", "path/to/real")
print(f"clean-FID = {score:.3f}")

# Or against a precomputed reference (FFHQ, COCO, etc.)
score = fid.compute_fid("path/to/generated",
                        dataset_name="FFHQ", dataset_res=1024,
                        dataset_split="trainval70k")
```

If two papers use different resizing pipelines, their FIDs are not comparable even at the same `$N$` and reference set. `clean-fid` exists precisely to remove this variance; use it.

**Bias 4: the Gaussian assumption.** FID models each feature set as a single multivariate Gaussian. Real Inception features are emphatically *not* Gaussian — they are multimodal, heavy-tailed, and live on a curved manifold. The Fréchet distance between fitted Gaussians can be small even when the true distributions differ in ways the first two moments cannot see (e.g., a bimodal real distribution vs a unimodal fake one with matched mean and covariance). This is the deepest bias and the one CMMD (next section) was built to remove. It also means FID can *saturate*: once two models both match the real mean and covariance well, FID stops discriminating between them even if one is visibly better, because the differences live in higher moments FID ignores.

There is a fifth trap that is not a bias in the formula but kills more real comparisons than any of the above: **the reference set.** FID is a distance *to a specific set of real images*, and that set is a modeling choice with enormous leverage. Evaluate against COCO-val and you measure "how close are my samples to photos with COCO's caption distribution." Evaluate against your *training* set and you reward memorization — a model that overfits and reproduces training images will post a spuriously low FID while being worthless as a generator (it generalizes to nothing new). Evaluate against a tiny or non-representative reference and the Gaussian fit is garbage. The rule: the reference set must be a *held-out* sample of the target distribution, large enough for a stable covariance estimate (tens of thousands of images), and *identical* across every model you compare. Two FIDs against different reference sets are not comparable, full stop — and a FID against the training set is not just incomparable, it is actively misleading. When a paper's FID looks too good to be true, the first thing to check is what it was measured against.

## 4. The fixes: FID-DINOv2 and CMMD

Two modern proposals fix the worst of FID's biases, and both are worth adopting.

**FID-DINOv2** keeps the FID formula but swaps the backbone. Instead of ImageNet-InceptionV3 features, it uses features from **DINOv2**, a self-supervised vision transformer trained on a far larger, more diverse corpus without classification labels. Stein et al. (2023, "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models") showed that FID computed in DINOv2 feature space correlates substantially better with human judgments of quality and is fairer across domains, because DINOv2 features are richer and less ImageNet-specific. It directly attacks Bias 2 (backbone) and partly Bias 4 (richer features are less degenerate under the Gaussian fit). The code change is just the feature extractor:

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

# Custom DINOv2 feature extractor returning a [B, D] embedding
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").cuda().eval()

class DinoFeatures(torch.nn.Module):
    def forward(self, x):           # x: float images normalized for DINOv2
        return dino(x)              # CLS embedding, [B, 1024]

fid_dino = FrechetInceptionDistance(feature=DinoFeatures(),
                                    normalize=True).cuda()
# ...update with real/fake, then compute() as before
```

**CMMD (CLIP-MMD)** is the more radical fix. Sadat & Otani et al. / Jayasumana et al. (Google, 2023, "Rethinking FID: Towards a Better Evaluation Metric for Image Generation") argued that the right move is to (a) replace Inception features with **CLIP image embeddings** — semantically rich and aligned with how images relate to language — and (b) replace the Gaussian-Fréchet distance with the **Maximum Mean Discrepancy (MMD)**, a distribution distance that makes *no* parametric assumption. MMD with a Gaussian kernel `$k$` between samples `$\{x_i\}$` (real) and `$\{y_j\}$` (fake) is:

$$
\text{MMD}^2 = \frac{1}{n^2}\sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{m^2}\sum_{j,j'} k(y_j, y_{j'}) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)
$$

This is an *unbiased* (or nearly so) estimator that compares the full distributions via all pairwise kernel similarities, not just the first two moments. That kills Bias 4 (no Gaussian assumption) and, critically, Bias 1: MMD is far less sample-size dependent, so CMMD gives stable estimates with as few as a couple thousand images instead of needing 50k. The paper's punchline is sharp — they show cases where FID *disagrees with obvious visual quality improvements* (FID goes up when images clearly get better, because of the Gaussian/backbone artifacts) while CMMD tracks the improvement. CMMD is becoming the recommended distribution metric for exactly these reasons.

| Metric | Backbone | Distance | Sample-size bias | Gaussian assumption |
| --- | --- | --- | --- | --- |
| FID | InceptionV3 (ImageNet) | Fréchet (2-Wasserstein) | strong, `$O(1/N)$` | yes |
| clean-FID | InceptionV3, fixed resize | Fréchet | strong | yes |
| FID-DINOv2 | DINOv2 (self-supervised) | Fréchet | strong | yes (but richer features) |
| CMMD | CLIP image encoder | MMD (kernel) | weak | no |

My recommendation, stated plainly: report **clean-FID** for backward comparability with the literature, and **CMMD** as your primary distribution metric for actual decisions. If you only get one, take CMMD.

There is a subtlety worth flagging so you do not over-rotate on CMMD: it inherits *CLIP's* biases instead of *Inception's*. CLIP image embeddings are excellent at semantic content but, like CLIP-score, are weaker at fine spatial detail and exact attribute binding, so CMMD is a fantastic *distribution* metric but does not magically solve prompt-following — that is still GenEval's job. The point of swapping backbones is not "CLIP is bias-free" (nothing is); it is that CLIP's biases are *better aligned with human semantic judgment* than ImageNet-Inception's texture-classifier biases, and MMD's lack of a Gaussian assumption removes a whole class of artifacts. You are trading a worse set of biases for a better set, not eliminating them. That framing — *which biases am I willing to accept for this question?* — is the right way to choose any metric.

## 5. Inception Score and why it is obsolete

Before FID there was the **Inception Score (IS)**, and you will still see it on old leaderboards, so know why it died. IS pushes generated images through InceptionV3 and looks at the predicted class distribution `$p(y|x)$`. The intuition was twofold: a good image should be *confidently* classified (low-entropy `$p(y|x)$` — it clearly depicts *something*), and across many generated images the *marginal* class distribution `$p(y) = \int p(y|x)p(x)\,dx$` should be *diverse* (high-entropy — the model covers many classes). IS rolls both into one number via the KL divergence:

$$
\text{IS} = \exp\!\left(\mathbb{E}_{x \sim p_g}\,\big[\,\text{KL}\big(p(y|x)\,\|\,p(y)\big)\,\big]\right)
$$

Higher is "better." The fatal flaws: (1) **It never looks at the real data at all** — IS only inspects the generated images' classifier outputs, so a model that produces one perfect, confidently-classified image per ImageNet class and nothing else scores near-maximal while being a catastrophic generator. (2) **It is entirely ImageNet-bound** — the score is defined over ImageNet's 1,000 classes, so it is meaningless for faces, text-to-image of arbitrary prompts, or any non-ImageNet domain. (3) **It is trivially gameable** and saturates: modern models max it out, so it cannot rank them. FID fixed flaw (1) by comparing to real data. The field has moved entirely to FID and its successors. Report IS only if a reviewer demands it for historical comparison; never make a decision on it.

The lesson IS teaches is more general than IS itself: **a metric that only inspects the generated samples, never the real distribution, cannot detect the most important failure — drawing beautiful images that do not match the data.** This is why "the image looks good" (which IS and, to a degree, learned-preference models measure) is fundamentally different from "the distribution is right" (which FID/CMMD measure). You need both kinds of sensor, because a model can be perfect at one and terrible at the other. IS is the cautionary tale of a metric that measured only the easy half and got crowned anyway for a few years. Barratt & Sharma's 2018 "A Note on the Inception Score" catalogued these failures formally and effectively ended IS's reign; it is worth reading once as a case study in how a plausible-looking metric can be deeply broken.

## 6. Precision and recall: splitting fidelity from diversity

FID, even fixed, collapses two things into one scalar: are my samples *realistic* (fidelity), and do they *cover the diversity* of the real distribution (recall)? A model can score a mediocre FID by being highly realistic but mode-collapsed (every sample gorgeous, but it only ever draws golden retrievers), or by being diverse but low-fidelity. **You cannot tell these apart from FID alone**, and they call for opposite fixes — the first needs more diversity (less guidance, more coverage), the second needs more fidelity (better model, more guidance). This is the diversity corner of the generative trilemma, and it is exactly where [GANs lost to diffusion](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost): GANs achieve stunning per-sample fidelity but chronically drop modes, which a single FID partially hides.

The fix, from Kynkäänniemi et al. (2019, "Improved Precision and Recall Metric for Assessing Generative Models"), is to estimate the *support manifolds* of the real and generated distributions in feature space and measure two quantities:

- **Precision** = the fraction of *generated* samples that fall inside the *real* data manifold. High precision means "my samples look real" — fidelity.
- **Recall** = the fraction of *real* samples that fall inside the *generated* manifold. High recall means "my model can produce the full variety of real data" — diversity/coverage.

The manifolds are estimated non-parametrically: around each real feature point, draw a hypersphere out to its `$k$`-th nearest neighbor; the union of these spheres approximates the real manifold's support. A generated point is "inside the real manifold" if it lands within any real point's hypersphere. Swap the roles for recall. Formally, with real features `$\{r_i\}$` and a per-point radius `$\rho_k(r_i)$` equal to the distance to its `$k$`-th nearest real neighbor, precision is

$$
\text{precision} = \frac{1}{M}\sum_{j=1}^{M}\;\mathbb{1}\!\left[\,\exists\, i :\; \|g_j - r_i\|_2 \le \rho_k(r_i)\,\right]
$$

where `$\{g_j\}$` are the `$M$` generated features. In words: a generated point counts toward precision if it lands inside *any* real point's `$k$`-NN ball. Recall is the mirror image — swap reals and fakes, build the balls around the *generated* points, and count how many *real* points fall inside. The choice of `$k$` (typically 3–5) sets how aggressively the union of balls fills in the gaps between samples; too small and the manifold is a scatter of disconnected dots, too large and it bloats to cover everything. Figure 4 makes the split concrete: a mode-collapsed model and a well-covered model with the same precision but very different recall.

![A before and after comparison of a mode-collapsed model with low recall versus a well-covered model with high recall](/imgs/blogs/evaluating-image-generation-honestly-4.png)

Now the asymmetry is legible. The mode-collapsed model on the left scores high *precision* (every sample it draws lands inside the real manifold — it looks real) but low *recall* (it only covers one of the five real clusters — it dropped four modes). The well-covered model on the right trades a hair of precision for a large gain in recall. A single FID might rank these two similarly; precision/recall tells you *which lever to pull*. This is why I never report FID without precision/recall (or its refinement) on any model where mode coverage matters.

**Density and Coverage** (Naeem et al., 2020, "Reliable Fidelity and Diversity Metrics for Generative Models") refine the precision/recall estimators to be robust to outliers. The `$k$`-NN hyperspheres in precision/recall are fragile: one outlier real point with a huge nearest-neighbor distance inflates the manifold and corrupts the estimate. **Density** replaces the binary "inside any sphere" with a *count* of how many real spheres a fake point falls into (so it is not saturated by a single sphere), and **Coverage** measures the fraction of real points whose neighborhood contains at least one fake point (more robust than recall to fake outliers). In practice I prefer density/coverage over raw precision/recall for the same reason I prefer CMMD over FID: they are more stable estimators of the same intuition. Here is the practical computation pattern:

```python
# Using prdc (pip install prdc), Naeem et al.'s reference implementation
from prdc import compute_prdc
import numpy as np

# real_feats, fake_feats: [N, D] numpy arrays of Inception/DINO features
metrics = compute_prdc(real_features=real_feats,
                       fake_features=fake_feats,
                       nearest_k=5)
print(metrics)
# {'precision': 0.86, 'recall': 0.78, 'density': 0.91, 'coverage': 0.74}
```

#### Worked example: diagnosing a fine-tune with precision/recall

You fine-tune SDXL on 5,000 product photos and FID *improves* from 14.2 to 11.8. Looks like a win — until you compute precision/recall. Precision jumps from 0.71 to 0.93 (samples look very on-brand), but recall *craters* from 0.62 to 0.28. The model has overfit to your narrow product aesthetic and lost the ability to generate anything else — classic fine-tuning mode collapse. FID *improved* because your eval prompts were also on-brand, so the narrowing matched the eval set. Recall caught what FID hid. The fix: lower the LoRA rank or learning rate, add regularization images, or stop the fine-tune earlier. Without precision/recall you would have shipped a model that can only draw your five product categories.

## 7. CLIP-score and why it cannot count

Everything so far measured *realism* and *diversity* — none of it read the prompt. For text-to-image, the prompt is half the job. The standard automatic alignment metric is **CLIP-score**: embed the image with CLIP's image encoder, embed the prompt with CLIP's text encoder, and take the cosine similarity (scaled). Hessel et al. (2021, CLIPScore) defines it as:

$$
\text{CLIP-score}(I, T) = w \cdot \max\!\big(\cos(\,E_I(I),\, E_T(T)\,),\, 0\big), \quad w = 2.5
$$

where `$E_I$` and `$E_T$` are CLIP's image and text encoders and `$w$` rescales to a friendlier range. Higher means "the image and prompt are more aligned in CLIP's joint embedding space." It is cheap, reference-free (no ground-truth image needed), and correlates decently with coarse relevance — "is this image about a cat" versus "about a car." Computing it is a few lines with `open_clip`:

```python
import torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.cuda().eval()

@torch.no_grad()
def clip_score(image_path, prompt):
    img = preprocess(Image.open(image_path)).unsqueeze(0).cuda()
    txt = tokenizer([prompt]).cuda()
    img_f = model.encode_image(img)
    txt_f = model.encode_text(txt)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return 2.5 * (img_f @ txt_f.T).clamp(min=0).item()

print(clip_score("out.png", "three red apples on a wooden table"))
```

Now the blind spots, and they are severe. **CLIP is a bag-of-words model in disguise.** CLIP's contrastive training pulls matching image-text pairs together and pushes mismatches apart, but the text encoder learns to represent prompts in a way that is largely insensitive to *word order and binding*. The classic demonstration (from the ARO and Winoground analyses): CLIP gives nearly identical scores to "a red cube on a blue sphere" and "a blue cube on a red sphere," because the embedding captures the *set* of concepts {red, blue, cube, sphere} far better than which adjective binds to which noun. This is **attribute-binding failure**, and it is the same weakness we covered in [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) — the conditioning signal and the metric that scores it share a blind spot.

**CLIP cannot count.** Ask for "three apples" and CLIP-score barely distinguishes an image with two apples, three apples, or seven apples, because counting requires precise spatial enumeration that CLIP's global pooled embedding does not represent. An image with *apples in it* scores high regardless of how many. **CLIP is weak on spatial relations** ("to the left of," "above," "behind") for the same reason. And because CLIP-score is bounded and saturates, once an image is roughly on-topic, piling on more relevant detail barely moves the number — so it cannot rank two on-topic images by *how well* they follow a complex prompt.

Why is CLIP structurally a bag-of-words model? It comes straight from the training objective. CLIP is trained with a *contrastive* loss over a batch: each image embedding is pulled toward its matching caption embedding and pushed away from all the other captions in the batch. That objective rewards getting the *gist* right — enough to tell a correct caption from a random other caption in the batch — but it provides almost no pressure to distinguish "red cube on blue sphere" from "blue cube on red sphere," because both contain the same content words and a random negative caption in the batch is usually about something completely different (a dog, a beach) rather than the same objects with swapped attributes. The training signal never *needed* compositional precision, so the encoder never learned it. The Winoground benchmark made this brutally concrete: it pairs two images and two captions that differ only in *how the same words are arranged*, and showed that CLIP and similar models score barely above random at matching them correctly. A metric built on those embeddings inherits the exact same blindness.

There is a second-order consequence that bites in evaluation specifically. Because CLIP-score is a *cosine similarity*, it is sensitive to the *length and specificity* of the prompt in ways unrelated to image quality. A short prompt ("a cat") and a long prompt ("a fluffy orange tabby cat sitting on a windowsill at golden hour, photorealistic, 50mm lens") produce text embeddings with different geometry, so CLIP-scores are not comparable *across prompts of different lengths* — you can only compare two models on the *same* prompt set. Average CLIP-score over a fixed prompt set is fine for ranking two models; a raw CLIP-score on one image tells you almost nothing in isolation.

The deeper trap: **CLIP-score can be gamed by adding CLIP-favored tokens.** Because the same CLIP family often sits *inside* the generator's guidance and *outside* as the metric, optimizing CLIP-score can degenerate into producing images that CLIP "likes" (high-frequency textures, prompt words rendered as literal text in the image) rather than images a human would call correct. If you fine-tune a model to maximize CLIP-score, you will get a model that games CLIP. This is Goodhart again. Use CLIP-score as a *coarse* alignment filter — it reliably catches "the image is completely off-topic" — but never as your alignment *verdict*. For real compositional alignment, you need the next section.

## 8. Compositional benchmarks: GenEval, T2I-CompBench, DPG-Bench

If CLIP-score cannot count or bind attributes, how do we measure those skills? The answer is the 2023–2024 wave of **compositional benchmarks** that replace a fuzzy similarity score with *explicit, verifiable questions* about the image, checked by an external model that actually can count and localize. Figure 5 is the canonical case these benchmarks exist to catch: two models tie on FID, but one follows the prompt and one does not.

![A before and after comparison where the model with better FID fails counting while the human-preferred model follows the prompt](/imgs/blogs/evaluating-image-generation-honestly-5.png)

The shared idea: generate from a *structured* prompt with a known correct answer, then run an **object detector or vision-language model** on the output and check the answer programmatically. Figure 6 lays out the three benchmarks and the verifier each uses.

![A matrix of three compositional benchmarks with the skill each probes and the verifier model it uses](/imgs/blogs/evaluating-image-generation-honestly-6.png)

**GenEval** (Ghosh et al., 2023) is the most widely cited. It uses an object-detection model (Mask2Former / a strong detector) to check six skills from templated prompts: *single object* (is the object present?), *two objects* (are both present?), *counting* (exactly N objects?), *colors* (is the object the named color?), *position* (is A left/right/above/below B?), and *color attribution* (does the right object have the right color — the binding test). The score is the fraction of generated images that pass the detector check. Because the verifier is a detector, GenEval *can* count and localize where CLIP-score cannot. A typical headline: SDXL scores around **0.55** overall on GenEval, while a strong T5-conditioned model like SD3 or a well-tuned FLUX pushes into the **0.65–0.70** range, with the biggest gaps on *counting* and *position* — exactly the skills CLIP-score is blind to.

**T2I-CompBench** (Huang et al., 2023) is broader, with categories for *attribute binding* (color, shape, texture), *object relationships* (spatial and non-spatial), and *complex* compositions. It uses a mix of verifiers: BLIP-VQA (ask "is the apple red?" and read the answer), a detector for spatial relations, and CLIP for some categories. It reports per-category scores so you can see *which* compositional skill a model lacks rather than one blended number.

**DPG-Bench** (Dense Prompt Graph Benchmark, Hu et al., 2024) targets *long, dense* prompts — paragraphs describing many objects, attributes, and relations at once — and uses a vision-language model to answer a graph of questions derived from the prompt. It stresses the regime where short-prompt benchmarks saturate: can the model hold *all* the constraints of a 60-word prompt simultaneously? This is where the T5/LLM-conditioned models (SD3, FLUX) pull decisively ahead of CLIP-only models, because a richer text encoder is the prerequisite for following dense prompts.

The key property all three share: **a verifiable per-prompt correct answer.** That is what lets them catch the counting and binding failures FID and CLIP-score average away. The cost is that they only measure what their templates probe and what their verifier model can reliably check — a detector that miscounts will mis-score, so the benchmark inherits the verifier's errors. Still, for prompt-following, a compositional benchmark is non-negotiable. I run GenEval on every text-to-image model I evaluate, and DPG-Bench when dense prompts are in scope.

A word of caution that the leaderboards rarely print: the verifier is itself a model, and its errors set a *ceiling* on the benchmark's reliability. If GenEval's object detector misses small objects, then a model that correctly draws a tiny third apple gets marked wrong — the benchmark *under*-credits a correct generation because the verifier failed. Conversely, if the detector hallucinates an object, a model gets *over*-credited. These verifier errors are usually small and roughly unbiased across models, so the *ranking* stays trustworthy even when the *absolute* scores carry a few points of verifier noise. The practical implication: trust GenEval to *rank* two models, and trust its *per-skill breakdown* to tell you *where* a model is weak, but do not over-interpret a 1–2 point absolute difference as meaningful — it can be verifier noise. As verifiers improve (a stronger detector, a better VLM), the benchmarks get sharper, which is one reason newer compositional benchmarks lean on capable vision-language models rather than narrow detectors. The general principle stands: a benchmark is only as honest as the model that grades it, so know your grader's failure modes before you trust its verdict.

#### Worked example: designing a counting probe in 20 minutes

Suppose you suspect your model miscounts. You do not need the full benchmark to get a signal. Take 50 prompts of the form "exactly N [object]" for N in {1,2,3,4,5} and 10 common objects, generate 4 images each, run a detector, and score the fraction where the detected count equals N. Here is the scoring core:

```python
from ultralytics import YOLO          # any solid detector works
detector = YOLO("yolov8x.pt")

def counting_accuracy(image_paths, target_class, target_count):
    hits = 0
    for p in image_paths:
        result = detector(p, verbose=False)[0]
        names = [result.names[int(c)] for c in result.boxes.cls]
        detected = sum(1 for n in names if n == target_class)
        hits += (detected == target_count)
    return hits / len(image_paths)

# e.g. accuracy for "exactly 3 apples" across 4 samples
acc = counting_accuracy(["a.png","b.png","c.png","d.png"], "apple", 3)
print(f"counting accuracy = {acc:.2f}")
```

When I ran a probe like this, a CLIP-conditioned SD1.5-class model scored near **chance on N≥3** while reporting a perfectly healthy CLIP-score on the same prompts — the metric said "great alignment," the probe said "cannot count past two." That gap is the entire argument for compositional benchmarks in one experiment.

## 9. Learned human-preference models: HPSv2, ImageReward, PickScore

The metrics so far are *designed* (FID, CLIP-score) or *rule-checked* (GenEval). The newest and most influential class is **learned**: train a model to directly predict which image a human would prefer, from a large dataset of human comparisons. These are the metrics that actually correlate best with held-out human preference, and they are what modern models are increasingly tuned toward (often as the reward in RLHF-style fine-tuning).

The recipe is consistent across all three. Collect a dataset of (prompt, image A, image B, human choice) triples — humans pick the better image. Initialize from CLIP (image+text towers). Fine-tune with a **preference loss**: for a preferred image `$x_w$` and a dispreferred `$x_l$` given prompt `$c$`, maximize the margin between their predicted scores, typically a Bradley–Terry / logistic objective:

$$
\mathcal{L} = -\,\mathbb{E}_{(c,\,x_w,\,x_l)}\;\log \sigma\!\big(\, r_\theta(c, x_w) - r_\theta(c, x_l)\,\big)
$$

where `$r_\theta$` is the learned reward (often `$\cos$` similarity in a fine-tuned CLIP space plus a head) and `$\sigma$` is the sigmoid. The model learns a scalar reward whose *differences* predict human choices. The three flavors:

- **ImageReward** (Xu et al., 2023): trained on 137k expert comparisons over DiffusionDB prompts, scoring fidelity, prompt alignment, and aesthetics jointly. Returns an unbounded reward (typically roughly `$-2$` to `$+2$`); higher is better. Strong general-purpose preference proxy.
- **HPSv2** (Human Preference Score v2, Wu et al., 2023): trained on the large HPD v2 dataset (798k human choices over 433k images spanning many models and prompt sources), specifically designed to be a *fair, stable* preference benchmark across models. Reports a score you average over a fixed prompt set. The standard "is my model aesthetically/preference-competitive" number.
- **PickScore** (Kirstain et al., 2023): trained on Pick-a-Pic, a million-plus web-collected real-user preferences. Designed to mirror *actual user* choices in a deployed setting rather than expert annotators.

Computing an ImageReward score is two lines:

```python
import ImageReward as RM
model = RM.load("ImageReward-v1.0")          # downloads the checkpoint

# Higher reward = more human-preferred for this prompt
reward = model.score("a corgi astronaut on the moon, cinematic", "out.png")
print(f"ImageReward = {reward:.3f}")

# Rank a candidate set for best-of-n selection
ranks, rewards = model.inference_rank(
    "a corgi astronaut on the moon, cinematic",
    ["a.png", "b.png", "c.png", "d.png"])
```

The blind spots, because these lie too. **(1) They inherit annotator bias.** What "preferred" means is whatever the annotators picked, which skews toward high-contrast, saturated, sharp, "Instagram-pretty" images. A model tuned hard toward HPSv2 drifts toward over-saturated, over-sharpened output — visually punchy, often subtly unnatural. **(2) They are blind to factual correctness.** A preference model will happily score a beautiful image of "the Eiffel Tower" that has six legs or melted text, because it judges aesthetics and rough relevance, not facts. **(3) They saturate and can be gamed**: once a model's outputs match the preference model's learned aesthetic, the score plateaus and stops discriminating, and direct optimization against the reward (reward hacking) produces images that score high and look worse to fresh humans. **(4) Distribution shift**: a preference model trained on 2023-era diffusion outputs may mis-rank a 2026 autoregressive model whose failure modes it never saw. Use preference models as your best *automatic* proxy for human taste, weight them in your basket, but never let them be the *only* human-facing signal — which is why the basket ends with real humans.

| Preference model | Training data | Output range | Best for |
| --- | --- | --- | --- |
| ImageReward | 137k expert comparisons | unbounded (~−2..+2) | general fidelity + alignment + aesthetics |
| HPSv2 | 798k choices, 433k images | averaged score | fair cross-model preference benchmark |
| PickScore | 1M+ real-user picks (Pick-a-Pic) | logit/softmax | mirroring deployed user choices |

## 10. The eval crisis: when the metric becomes the target

Now the honest, opinionated part. The field has a quiet crisis, and it is Goodhart's law at scale. Here is the chain of events that plays out repeatedly:

1. A metric (say FID) becomes the field's leaderboard number.
2. Researchers optimize architectures, schedules, and hyperparameters to push FID down, because that is what gets papers accepted and models cited.
3. FID **saturates** — top models cluster at FID 2–3 on ImageNet, differences within the noise of the estimator — so it stops discriminating.
4. Worse, the optimization starts to *diverge from human preference*: models tuned to FID can get visibly worse in ways FID's Gaussian/backbone biases cannot see (the CMMD paper's central finding — FID going *up* as images get better).
5. Meanwhile **human preference and FID disagree**: the model with the better FID is not the model humans prefer (the corgi at the top of this post). Studies repeatedly find weak or even negative rank correlation between FID and human preference on modern, already-good models.

The result is an evaluation landscape where **no single number is trustworthy**, and the leaderboards that rank models on one number are, at the frontier, mostly measuring noise plus whatever bias that metric carries. This is not a reason to despair — it is a reason to evaluate *honestly*, with a basket. The crisis exists precisely because people reached for one number. The cross-link here is direct: [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) is the perfect illustration — raising the CFG scale *improves* FID and CLIP-score up to a point and then *degrades* diversity (recall) while continuing to look "better" to quick human glances, so the very knob you tune trades off the metrics against each other. There is no setting that maximizes all of them. You must decide what you are optimizing and measure all the corners.

A concrete symptom you will see on real leaderboards: SDXL, SD3, FLUX, and a strong consistency-distilled model can be within a point of each other on FID while differing by **10–20 points of GenEval** and flipping order entirely on HPSv2 and human arena win-rate. The FID ranking and the human ranking are nearly uncorrelated at the top. If you picked your model on FID, you picked on noise.

There is a quantitative way to say "you picked on noise," and you should compute it. A FID difference is only meaningful if it exceeds the *estimator's own variance*. Bootstrap it: resample your generated set with replacement, recompute FID 50 times, and look at the spread. On 50k samples, FID's standard deviation is typically a few *tenths* of a point — so a 0.5 FID gap between two models is roughly one standard deviation, i.e. *not significant*. The same discipline applies to human A/B: a 55% win rate over 200 prompts has a 95% confidence interval of roughly ±7 points (binomial standard error `$\sqrt{p(1-p)/n} \approx 0.035$`), so 55% is *not* reliably above 50% — you need either more prompts or a larger effect to call it. Reporting a metric without its uncertainty is how the field convinces itself that noise is progress. Always bootstrap the distribution metrics and always put a confidence interval on the human win rate; a difference inside the interval is a tie, full stop.

## 11. How to actually evaluate a model you are shipping

Here is the playbook I use, and the heart of this post. Stop looking for the one metric. **Assemble a basket whose blind spots do not overlap, hold the protocol fixed, and end with human eyes.** Figure 7 is the basket as a stack — five layers, each catching what the others miss.

![A stack of the shipping evaluation basket from distribution metric through alignment compositional preference and human spot check](/imgs/blogs/evaluating-image-generation-honestly-7.png)

**Layer 1 — Distribution realism.** Compute **CMMD** as your primary (stable, no Gaussian assumption, sample-efficient) and **clean-FID** for comparability with the literature, both at a *fixed sample count* (state your `$N$`) against a *fixed reference set*, with *fixed resizing* (clean-fid handles this). This answers "do my samples look like real images, as a distribution." It does *not* read prompts.

**Layer 2 — Prompt alignment.** Compute **CLIP-score** as a coarse on-topic filter — it reliably catches catastrophic mismatch. Do not trust it for fine-grained correctness.

**Layer 3 — Compositional correctness.** Run **GenEval** (and **DPG-Bench** if dense prompts matter). This catches counting, binding, color, and spatial failures that Layers 1–2 are blind to. Report the *per-skill* breakdown, not just the average — "0.65 overall" hides "0.9 on single-object, 0.3 on counting."

**Layer 4 — Learned preference.** Compute **HPSv2** and **ImageReward** over a fixed, public prompt set (Parti-Prompts' 1,600 prompts are the standard; the DrawBench set is another). This is your best automatic proxy for human taste. Weight it, but know it skews toward saturated images.

**Layer 5 — Human spot-check.** The final word. Take **100–200 prompts** (a mix of easy, compositional, and adversarial), generate with a *fixed seed and fixed sampler/steps/CFG* for every model, and run a **blind A/B** (arena-style pairwise comparison) with real annotators. Report **win rate** with confidence intervals. This is the only layer that measures the actual target. Everything above it is a cheap proxy you use *between* human evals to iterate fast.

The protocol rules that make this honest, learned the hard way:

- **Fix everything that is not the model.** Same prompts, same seeds, same number of inference steps, same sampler, same CFG scale, same resolution, same reference set, same `$N$`. A FID difference from changing the sampler is not a model difference. If you change two things, you have measured nothing.
- **State your sample size and reference set on every distribution metric.** A FID without `$N$` and reference is uninterpretable.
- **Warm up and use enough samples.** CMMD is stable at ~2k; FID needs ~50k for a stable estimate; precision/recall need a few thousand. Under-sampling adds variance that masquerades as a model difference.
- **Report a basket, never a single number.** A table of (metric × value) for every model, with the human win-rate as the tiebreaker.
- **Probe your specific failure mode.** If you ship product images, build a targeted compositional probe for *your* objects and attributes (the 20-minute probe in section 8). Generic benchmarks will not catch your domain's specific failure.

In practice I pin the entire protocol in a config so a teammate can reproduce the exact numbers six months later. The point is that *everything* that is not the model under test is fixed and recorded:

```yaml
# eval_protocol.yaml — pin every knob so the comparison is honest
prompts: parti_prompts_1600.txt   # fixed public set, never ad hoc
seed: 0                           # same seed for every model
resolution: 1024
sampler: dpmpp_2m
steps: 30
guidance_scale: 5.0
distribution_metric:
  primary: cmmd                   # stable, sample-efficient
  secondary: clean_fid            # for literature comparability
  num_samples: 50000              # fixed N, stated explicitly
  reference_set: coco_val2014_30k # fixed reference, stated explicitly
  resize: cleanfid_lanczos        # removes resize/JPEG variance
diversity_metric: density_coverage
alignment: [clip_score, geneval, dpg_bench]
preference: [hpsv2, imagereward]
human_eval:
  prompts: 200                    # easy + compositional + adversarial
  protocol: blind_pairwise        # arena-style A/B
  report: win_rate_with_ci        # confidence interval, not point estimate
```

A second engineer who runs this file against two checkpoints gets numbers that *mean* something, because the only thing that varied is the weights. The moment any line of this drifts between models — a different sampler, a different `$N$`, a different reference set — the comparison is contaminated and the right move is to rerun, not to hand-wave the difference away.

Figure 8 turns this into a decision tree: start from the *question you are asking*, and it routes you to the right metric, because realism, prompt-following, and human taste need different tools.

![A decision tree routing from the evaluation question to the appropriate metric for realism alignment or human taste](/imgs/blogs/evaluating-image-generation-honestly-8.png)

#### Worked example: the FID tie that the basket breaks

Return to the opening scenario with real structure. You have two SDXL fine-tunes, A and B, and you run the full basket on a fixed protocol — 10k samples for CMMD, 50k for clean-FID, GenEval on its standard prompts, HPSv2/ImageReward on Parti-Prompts, and a 200-prompt blind human A/B, all at seed 0, 30 DPM-Solver++ steps, CFG 5.0, 1024×1024.

| Metric | Model A | Model B | Who wins |
| --- | --- | --- | --- |
| clean-FID (50k) | 7.9 | 8.4 | A (barely) |
| CMMD | 0.62 | 0.58 | B |
| CLIP-score | 0.31 | 0.31 | tie |
| GenEval (overall) | 0.48 | 0.67 | B (decisive) |
| — counting subskill | 0.22 | 0.61 | B (huge) |
| HPSv2 | 0.271 | 0.284 | B |
| ImageReward | 0.41 | 0.78 | B |
| Human win rate (200 prompts) | 38% | 62% | B (decisive) |

Read the table the way a shipping engineer must. A wins *only* on clean-FID, and only by 0.5 — well inside the metric's noise at this sample size, and exactly the kind of difference the CMMD paper warns is an artifact. On *every* metric that reads the prompt (GenEval, especially counting) and *every* metric that tracks human preference (HPSv2, ImageReward, and the human A/B itself), B wins, and on the ones that matter most — GenEval counting and the human win rate — B wins decisively. The FID "win" for A was the model drawing the wrong number of beautiful objects. **You ship B.** If you had a single-metric process, you would have shipped A and shipped a worse product. The basket did its job: the blind spots did not overlap, so the truth had nowhere to hide.

## 12. Evaluation is how you see the generative trilemma

Step back and connect this to the spine of the whole series. The **generative trilemma** says you cannot simultaneously maximize sample *quality*, mode *coverage/diversity*, and sampling *speed* — improving one tends to cost another. Evaluation is not a separate concern bolted on at the end; it is *literally the instrument* you use to read where your model sits on each axis of that trilemma. Each metric family is a sensor for one corner.

**Quality** is what FID/CMMD and the precision side of precision/recall measure — do individual samples look real, and does the distribution match. **Diversity/coverage** is what recall, coverage, and the *spread* term of FID measure — does the model cover the full variety of the data, or has it collapsed. **Speed** is not measured by an image metric at all, but it is the axis you are *trading against*, so every quality/diversity number must be reported *at a stated step count and latency* or it is meaningless. A FID of 3.0 at 50 steps and a FID of 3.0 at 4 steps are wildly different achievements; the second is a distillation triumph, the first is table stakes. This is why the kit insists every result names the GPU, the step count, and the latency: those numbers locate you on the speed axis so the quality numbers can be read in context.

The reason a *basket* is mandatory is now structural, not just empirical: a single scalar cannot represent a point in a three-dimensional trade-off space. You need at least one sensor per axis — a quality metric, a diversity metric, and an explicit speed/step budget — plus the alignment and preference sensors that the trilemma's pure-distribution framing leaves out (the trilemma is about matching `$p_\text{data}$`, but text-to-image also has to match the *prompt*, a fourth axis the classic trilemma does not name). The honest evaluation is a small dashboard, not a single needle: distribution quality, coverage, prompt alignment, compositional correctness, learned preference, and a speed budget, each read separately and traded against the others on purpose. When someone hands you a single FID and asks "is this model good," the correct answer is "good at *what*, at *how many steps*, for *which prompts* — show me the dashboard."

#### Worked example: the same model at two step counts

Take one FLUX-class model and evaluate it at 50 steps and at 4 steps (consistency-distilled), fixed everything else. At 50 steps: CMMD 0.55, recall 0.74, GenEval 0.66, latency ~6 s/image on an A100. At 4 steps: CMMD 0.61 (slightly worse), recall **0.58** (clearly worse — diversity dropped), GenEval 0.63 (close), latency ~0.6 s/image. If you reported only CMMD you would say "4 steps costs you almost nothing, ship it." The *recall* number tells the real story: the 12× speedup cost you a sixth of your diversity — the distilled model is subtly mode-narrowing, producing more "samey" images per prompt. Whether that trade is acceptable is a product call, but you can only *make* the call because you measured the diversity axis, not just quality. One number would have hidden the entire cost of the speedup. That is the trilemma made visible by the basket.

## 13. Case studies: real numbers from the literature

Concrete results to anchor the principles, with sources. Treat exact figures as approximate where noted — they depend on sample count, reference set, and resizing, which is the whole point of this post.

**SDXL vs SD1.5 (Podell et al., 2023, SDXL).** SDXL (a ~2.6B U-Net with two text encoders) reports markedly better human preference and prompt alignment than SD1.5 (~0.86B), yet the *FID* comparison is famously murky — SDXL does **not** dominate SD1.5 on raw COCO FID, and the SDXL paper deliberately leads with **human preference win rates** rather than FID, precisely because FID failed to capture the quality jump that humans clearly saw. This is one of the most-cited real examples of FID-vs-preference divergence, and it is why the field shifted toward preference and compositional evaluation.

**SD3 and the GenEval story (Esser et al., 2024, SD3).** The SD3 paper reports GenEval scores and shows that scaling the MM-DiT architecture and improving the text-conditioning (CLIP + T5) lifts compositional scores — particularly attribute binding and spatial relations — more than it moves FID. SD3-class and FLUX-class models land in the rough **0.6–0.7 GenEval** band, versus SDXL's ~0.55, with the gains concentrated in the compositional subskills CLIP-score cannot see. The lesson: progress at the 2024–2026 frontier is measured in GenEval and preference, not FID, because FID saturated.

**The CMMD demonstration (Jayasumana et al., 2023).** The "Rethinking FID" paper's headline experiment: as you *progressively improve* a model (e.g., more sampling steps, better guidance), there are regimes where **FID gets worse while the images get better**, and CMMD correctly improves throughout. They also show FID's strong sample-size dependence and CMMD's stability at small `$N$`. This is the single most important empirical argument for adopting CMMD as your primary distribution metric.

**FID-DINOv2 fairness (Stein et al., 2023).** Their large human study across many generative models found that **FID in DINOv2 space ranks models in much closer agreement with humans** than Inception-FID, and that Inception-FID systematically *disadvantages diffusion models* relative to GANs in ways humans do not — a backbone bias with real consequences for which family looks "better" on a leaderboard. If you must use a Fréchet-style metric, use the DINOv2 backbone.

**Consistency/distilled models (LCM, SDXL-Turbo, DMD2).** Few-step models are where evaluation gets *most* treacherous, because they trade quality for speed and the trade is exactly what you must measure. SDXL-Turbo and DMD2-class one-to-four-step models report FID and preference numbers competitive with their many-step teachers on *easy* prompts, but the honest evaluation shows the gap opens on *hard compositional* prompts and *diversity* (recall drops as steps drop). A single FID at one step versus the teacher at 30 steps hides this; you need precision/recall and a compositional probe to see what the speedup cost. This connects directly to the speed corner of the trilemma — never report a distillation result on FID alone.

**Human-preference leaderboards (the arena).** The most honest "metric" the field has converged on for frontier models is not a metric at all — it is a *crowdsourced arena*: show two images for the same prompt to thousands of users, collect pairwise votes, and fit an Elo or Bradley–Terry rating. Image-generation arenas (the same idea as the LLM chatbot arenas) rank models by aggregated human preference over real user prompts, which is as close to the *target* as automatic evaluation gets. The catch is that arenas are slow, expensive, and not reproducible (the prompt and voter distribution shift over time), so you cannot use them to iterate during training — which is precisely why you need the automatic basket as a fast proxy *between* arena snapshots. The relationship is the whole picture of honest evaluation: cheap automatic metrics to iterate fast, the human arena as ground truth to check that your proxies have not drifted away from what people actually want. When the two disagree, the arena wins and your basket needs reweighting.

## 14. When to reach for each metric (and when not to)

Decisive recommendations. Every metric is a cost in compute and a risk of being misled; here is when each is worth it.

- **Reach for CMMD** as your default distribution metric — it is stable, sample-efficient, makes no Gaussian assumption, and tracks human quality judgments better than FID. Make it primary.
- **Reach for clean-FID** when you need to compare against published numbers (most of the literature reports FID). Use `clean-fid` specifically to remove the resize/JPEG variance. **Do not** compare a FID you computed at 10k against a paper's 50k, or with different resizing — it is not the same quantity.
- **Reach for FID-DINOv2** when you suspect backbone bias — non-ImageNet domains (faces, art, medical) or cross-family comparisons (diffusion vs GAN vs autoregressive). **Do not** keep using Inception-FID for faces; it is known to be noisy there.
- **Reach for precision/recall or density/coverage** whenever mode coverage matters — generative diversity, after a fine-tune, after raising CFG, after distillation. **Do not** ship a model on FID alone if it could be mode-collapsed; FID hides it.
- **Reach for CLIP-score** as a *coarse* on-topic filter and a cheap regression guard. **Do not** use it as your alignment verdict, and **never** fine-tune directly toward it — you will get a CLIP-gaming model that cannot count.
- **Reach for GenEval / T2I-CompBench / DPG-Bench** for any text-to-image model where prompt-following matters — which is all of them. **Do not** report only the average; report per-skill, because the average hides the counting cliff.
- **Reach for HPSv2 / ImageReward / PickScore** as your best automatic preference proxy and as a *reward* for best-of-n selection or light RLHF. **Do not** optimize hard against them (reward hacking → over-saturated images), and **do not** treat them as factual-correctness checks.
- **Reach for a human blind A/B** before any real ship decision. **Do not** skip it because the automatic numbers look good — the automatic numbers and human preference diverge at the frontier, every time.
- **Reach for Inception Score** essentially never — only if a reviewer demands it for historical comparison. It ignores the real data and is obsolete.

The meta-rule: **the metric you optimize is the behavior you get.** If you tune to FID you get FID-pretty mush; tune to CLIP-score you get bag-of-words images; tune to HPSv2 you get saturated punch. Tune to a *basket* with a human anchor, and you get a model that is actually good.

## 15. Key takeaways

- **No single metric is trustworthy.** Fidelity, diversity, and prompt-alignment are three different axes; any one scalar collapses them and hides the others. Always report a basket.
- **FID lies four ways**: it shrinks as you add samples (`$O(1/N)$` covariance bias), it inherits ImageNet-Inception's backbone bias, it shifts by points with resize/JPEG choices, and its Gaussian assumption ignores higher moments and lets it saturate.
- **Make distribution metrics comparable or not at all**: fixed sample count, fixed reference set, fixed resizing (`clean-fid`). A FID without `$N$` and reference is meaningless. Prefer **CMMD** (no Gaussian, sample-efficient) as primary.
- **Split fidelity from diversity** with precision/recall or density/coverage — it is the only way to catch mode collapse a single FID hides, especially after fine-tuning or distillation.
- **CLIP-score cannot count, bind attributes, or read word order** — it is a bag-of-words proxy. Use it as a coarse on-topic filter, never as the alignment verdict, and never as a fine-tuning target.
- **Compositional benchmarks (GenEval, DPG-Bench) are non-negotiable** for prompt-following — they use a detector/VLM that actually counts and localizes. Report per-skill, not the average.
- **Learned preference models (HPSv2, ImageReward, PickScore) correlate best with human taste** but skew toward saturated images, ignore facts, and can be reward-hacked. Weight them; do not worship them.
- **End every shipping decision with a human blind A/B.** At the frontier, FID and human preference are nearly uncorrelated — the human win rate is the only metric that measures the actual target.
- **Goodhart's law is the whole story**: the moment a metric becomes the target, it stops being a good measure. Optimize a basket with a human anchor, not one number.

## 16. Further reading

- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (2017) — the original FID paper.
- Chong & Forsyth, *Effectively Unbiased FID and Inception Score and where to find them* (2020) — the sample-size bias and the `$\overline{\text{FID}}_\infty$` extrapolation.
- Parmar, Zhang & Zhu, *On Aliased Resizing and Surprising Subtleties in GAN Evaluation* (2022) — the clean-fid resize/compression bias.
- Jayasumana et al., *Rethinking FID: Towards a Better Evaluation Metric for Image Generation* (2023) — CMMD (CLIP-MMD).
- Stein et al., *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models* (2023) — FID-DINOv2 and the human-correlation study.
- Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models* (2019); Naeem et al., *Reliable Fidelity and Diversity Metrics* (2020) — precision/recall and density/coverage.
- Hessel et al., *CLIPScore: A Reference-free Evaluation Metric for Image Captioning* (2021); Ghosh et al., *GenEval* (2023); Huang et al., *T2I-CompBench* (2023) — alignment and compositional benchmarks.
- Xu et al., *ImageReward* (2023); Wu et al., *Human Preference Score v2* (2023); Kirstain et al., *Pick-a-Pic / PickScore* (2023) — learned human-preference models.
- Within this series: [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) (the metrics' foundations), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) (the diversity↔fidelity knob), [generative adversarial networks and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost) (mode coverage), and [text encoders and prompt conditioning](/blog/machine-learning/image-generation/text-encoders-and-prompt-conditioning) (compositional failure modes). For where evaluation goes next, see the forthcoming [autoregressive vs diffusion showdown](/blog/machine-learning/image-generation/autoregressive-vs-diffusion-the-2026-showdown) and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
