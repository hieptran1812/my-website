---
title: "Flow Matching for Video: Why Straight Paths Scale to the Hardest Regime"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why rectified flow became the training objective of choice for Wan, Mochi, and the open video frontier — the same straight-path velocity target as the image case, the schedule shift that high-resolution long-duration latents force, and the few-step sampling that pays off when every step is a giant spacetime forward pass."
tags:
  [
    "video-generation",
    "diffusion-models",
    "flow-matching",
    "rectified-flow",
    "video-diffusion",
    "text-to-video",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/flow-matching-for-video-1.png"
---

Here is a number that decides architectures. A single sampling step on a modern video model is not the cheap matrix multiply it is for an image model. When you denoise a 5-second 720p clip, every step runs a forward pass over a spacetime latent of more than a million tokens, with full attention reaching across both space and time. On an H100 that one step can take a meaningful fraction of a second; the whole render is dozens of those steps stacked end to end. So the question "how many sampling steps do we need?" is not an academic one for video the way it sometimes is for images. Cut the step count from 50 to 15 and you cut the wall-clock render — and the cloud bill — by more than three times. The objective you train under is the single biggest lever on how few steps you can get away with, and that is the whole reason this post exists.

The answer the field converged on, almost unanimously, between 2024 and 2026, is flow matching — specifically the rectified-flow flavor of conditional flow matching. Wan trains on it. Mochi trains on it. HunyuanVideo trains on it. Stable Diffusion 3 popularized it for images and the video models inherited it wholesale. If you open the scheduler config of nearly any open video model released in this window you will find a `FlowMatchEulerDiscreteScheduler` and a shifted timestep schedule, not the `DDPMScheduler` you would have found two years earlier. This is not fashion. It is a direct response to the brutal compute budget that defines video: flow matching trains stably at the absurd token counts video produces, its objective does not care how big the latent is, and its near-straight sampling paths let a solver cross the distribution in far fewer of those expensive steps.

![Before and after comparison of a curved DDPM sampling path that needs many steps versus a straight flow-matching path an ODE solver crosses in a handful of steps](/imgs/blogs/flow-matching-for-video-1.png)

This post is about why straight paths win in the hardest regime. We will recap conditional flow matching just enough to fix notation — the straight-path velocity target $v = x_1 - x_0$, regressed without ever simulating a trajectory — and link out to the image series for the full derivation rather than repeat it. Then we spend our real depth on the three things that are genuinely different for video. First, why the objective's indifference to latent size is exactly the property video needs. Second, the noise-schedule and timestep-sampling shift: high-resolution, long-duration latents have to be trained under a schedule pushed toward higher noise, and we will derive why the shift scales with token count rather than treat it as a magic constant. Third, what few-step sampling buys you when each step is a giant spacetime pass, and how that composes with the distillation we cover separately. By the end you will be able to write a flow-matching training step over a video latent in PyTorch, read a video model's scheduler config and know what its shift means, and decide when fewer steps is a free win versus a quality cliff.

This sits squarely on the series spine: video is spatial generation times temporal coherence under a brutal compute budget. Flow matching is how we buy the spatial-and-temporal denoising cheaply enough to afford the rest. It trains the [spacetime diffusion transformer](/blog/machine-learning/video-generation/video-diffusion-transformers) that is the real workhorse, operates on the latent the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) hands it, and inherits everything mechanical from the image case we covered in [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion). If you have not read [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), the one-line version is that the time axis multiplies your token count and your sampling cost, and flow matching is the part of the stack that fights back on both.

## 1. A two-minute recap of conditional flow matching

Let me fix the picture before we touch video at all, because the whole argument rests on getting the objective right, and the objective is genuinely simple. We are not going to re-derive it from the continuity equation here — the full derivation lives in [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) — but we need the shape of it in front of us.

You have a data distribution (clean latents) and a noise distribution (a standard Gaussian). Diffusion thinks of this as a stochastic corruption process you run forward and then learn to reverse. Flow matching thinks of it differently: it posits a smooth, deterministic flow that transports the noise distribution into the data distribution over a unit time interval, governed by an ordinary differential equation

$$
\frac{dx_t}{dt} = v_\theta(x_t, t), \qquad t \in [0, 1].
$$

Here $x_0$ is a noise sample, $x_1$ is a data sample, and $v_\theta$ is the velocity field we want to learn — a network that, given a point $x_t$ along the way and the time $t$, tells you which direction and how fast to move. Sampling is then just integrating this ODE forward from a fresh noise draw at $t=0$ to a generated sample at $t=1$. Notice already a difference in temperament from diffusion: there is no stochasticity in the generative process here, no noise injected at each step. Flow matching's sampling is a deterministic ODE solve, where diffusion's ancestral sampling is a stochastic SDE solve (the two are connected — the probability-flow ODE is the deterministic backbone of the diffusion SDE, a correspondence we trace in [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view)). For video this determinism is a quiet asset: a deterministic path is easier to distill into a few-step student and easier to make reproducible across runs, both of which matter when each generation is expensive enough that you cannot afford to roll the dice many times.

The genius of conditional flow matching is in how you build the training target without ever solving the ODE. You pick the simplest possible path between a specific noise sample and a specific data sample: a straight line. Define the interpolation

$$
x_t = (1-t)\, x_0 + t\, x_1,
$$

a constant-velocity walk from $x_0$ to $x_1$. Differentiate it and the velocity along this path is, trivially,

$$
\frac{dx_t}{dt} = x_1 - x_0.
$$

That constant, $x_1 - x_0$, is the regression target. The conditional-flow-matching loss is just least squares against it:

$$
\mathcal{L}_\text{CFM} = \mathbb{E}_{x_0,\, x_1,\, t}\left[\, \left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2 \,\right],
$$

with $x_0 \sim \mathcal{N}(0, I)$, $x_1$ a data sample, and $t$ drawn from some distribution on $[0,1]$ we will obsess over later. The remarkable theorem behind flow matching is that even though each training pair sees its own straight line, regressing against these per-pair straight-line velocities recovers the *marginal* velocity field whose ODE actually transports the full noise distribution to the full data distribution. You train on straight segments; you get a globally correct (and, in the rectified-flow construction, nearly straight) flow.

Three properties of this objective matter for everything that follows, so let me state them plainly.

It is **simulation-free**. You never roll out a trajectory during training. You draw a random $t$, form one interpolation point, run the network once, and compare to a target you computed in closed form. That is a single forward and backward pass per sample, identical in cost structure to a diffusion training step. No backprop-through-time, no solver in the loop.

The target is **clean and bounded**. $x_1 - x_0$ is just data minus noise. It does not blow up, it does not depend on a fragile noise schedule the way $\epsilon$-prediction's effective target can at the extremes, and its scale is stable across $t$. This is the property that buys training stability, and stability is the thing that breaks first at video scale.

The learned path is **straight, by construction of the target and by the rectified-flow refinement**. Diffusion's probability-flow ODE has a curved trajectory; tracking a curve accurately needs many small solver steps. A straight (or rectified, nearly straight) path can be crossed by a coarse solver — even, in the limit, a single Euler step — with little error. That is the property that buys few-step sampling, and few-step sampling is where video's expensive forward pass makes the payoff enormous.

It is worth being precise about *why* the path can be straight, because it is not obvious that you can interpolate noise and data linearly and get a valid generative flow out of it. The subtlety is the difference between the conditional paths and the marginal one. For a single fixed pair $(x_0, x_1)$, the path $x_t = (1-t)x_0 + t x_1$ is exactly straight and its velocity is exactly the constant $x_1 - x_0$. But the network does not see one pair; it sees many pairs sharing the same intermediate point $x_t$ — the same noisy latent can sit on the straight line to many different clean videos. When you regress the constant per-pair velocity at a shared $x_t$, least squares drives the network's output to the *conditional mean* of those per-pair velocities, and the flow-matching theorem says this conditional-mean field is precisely the marginal velocity that transports the whole noise distribution to the whole data distribution. So the field the network learns is generally *not* perfectly straight — it bends where many data points compete for the same intermediate region — but it is straight*er* than diffusion's, and the reflow procedure of rectified flow iteratively re-straightens it by retraining on the model's own (already nearly collinear) noise-data pairs. The straightness is real and it is what lets the solver be coarse; it is just earned by the construction rather than free.

Everything in this post is one of those three properties cashed out against the specific pain of the video regime.

## 2. Make $x_0$ and $x_1$ video latents and the loss does not change

The first genuinely useful fact about flow matching for video is a non-fact: almost nothing changes. This is worth dwelling on because it is the source of flow matching's scalability.

In the image case, $x_1$ is a latent of shape $C \times H \times W$ — channels by height by width, produced by a VAE encoder from a single image. In the video case, $x_1$ is a latent *clip* of shape $C \times T \times H \times W$: $T$ latent frames, each $H \times W$, with $C$ channels, produced by the [causal 3D-VAE](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) from a sequence of pixel frames. The noise sample $x_0$ is a Gaussian tensor of exactly that shape. The interpolation $x_t = (1-t)x_0 + t x_1$ is computed elementwise over the whole spacetime tensor. The velocity target $x_1 - x_0$ is computed elementwise. The network $v_\theta$ is a [spacetime diffusion transformer](/blog/machine-learning/video-generation/video-diffusion-transformers) that ingests the $T \times H \times W$ latent and outputs a velocity of the same shape. The loss sums the squared error over every voxel and averages over the batch:

$$
\mathcal{L}_\text{CFM} = \mathbb{E}_{x_0,\, x_1,\, t}\left[\, \left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2 \,\right], \qquad x_1 \in \mathbb{R}^{C \times T \times H \times W}.
$$

Character for character, this is the same expression as the image case. The expectation now runs over clips, the tensors now carry a $T$ dimension, but the *form* is invariant. That invariance is not a coincidence; it is the deep reason flow matching scales. The objective is defined per voxel and summed. It has no term that depends on the spatial extent, no term that depends on the temporal extent, no normalization that has to be retuned when the latent gets bigger. You can take a model trained on 16-frame clips and continue training it on 121-frame clips and the loss function does not change one symbol. Compare this to objectives or regularizers whose constants are tuned for a particular resolution — those force you to re-derive your hyperparameters every time the latent grows, and video latents grow a lot.

![Graph showing the flow-matching cycle where training regresses a velocity target on one random interpolation point and sampling integrates the learned field with an ordinary differential equation solver](/imgs/blogs/flow-matching-for-video-2.png)

There is a second invariance that matters just as much: the *cost structure* of a training step is also unchanged in form. One training step is one forward pass and one backward pass over the latent — no rollout, no solver. The absolute cost of that pass goes up because the latent is bigger and the spacetime attention is more expensive, but the *number* of passes per training step stays at one. A diffusion training step is also one forward-backward pass, so on this axis flow matching and diffusion are even. Where flow matching pulls ahead is not training cost per step; it is sampling cost (fewer steps) and training *stability* (a cleaner target survives the scale-up), which we get to next. The point of this section is the foundation under both: because the objective is blind to latent size, you can scale to video's token counts without touching it.

Let me make the token count concrete, because the rest of the post leans on it. Take our running example, a 5-second clip at 720p and 24 fps — 120 pixel frames. A typical causal 3D-VAE compresses by roughly $4 \times 8 \times 8$ (temporal $\times$ spatial $\times$ spatial), so 120 frames at $720 \times 1280$ become on the order of 31 latent frames at $90 \times 160$. A spacetime-DiT then patchifies that, say with a $1 \times 2 \times 2$ patch, into roughly $31 \times 45 \times 80 \approx 112{,}000$ tokens — and that is the *latent* token count; the attention operates over all of them jointly. Higher resolutions and longer clips push this past a million tokens. The flow-matching loss does not flinch at any of these numbers. That is the whole game.

It is worth dwelling on what "the loss does not flinch" rules *out*, because the comparison sharpens the point. Picture instead an objective with a normalization constant tuned per spatial resolution, or a loss weight schedule fit to a particular latent's SNR profile. Every time you grew the clip — more frames, higher resolution — you would have to re-derive those constants, and worse, you would have to re-derive them in the regime where they matter most (the high-noise steps), where they are most sensitive. You would end up maintaining a table of "the right loss weights for $16 \times 256 \times 256$ versus $49 \times 480 \times 720$ versus $121 \times 720 \times 1280$," and curriculum training that grows the clip during training would force you to interpolate between rows of that table mid-run. Flow matching deletes the table. The objective is one expression for every row, and the only thing that changes across rows is the single shift scalar, which you compute from a formula rather than fit. For a regime where you genuinely do grow the latent — most video training uses a resolution-and-duration curriculum, starting on short low-res clips and scaling up — this is not a minor convenience; it is what makes the curriculum tractable.

#### Worked example: a resolution-and-duration curriculum

Concretely, a common training curriculum runs in three phases. Phase one: pretrain on $16$-frame $256\times256$ clips to learn basic motion and appearance cheaply, with a modest shift (say $\alpha \approx 3$) because the latent is small. Phase two: continue on $49$-frame $480\times720$ clips to learn longer, higher-resolution motion, raising the shift to roughly $\alpha \approx 7$ as the token count climbs about an order of magnitude. Phase three: fine-tune on $121$-frame $720\times1280$ clips for the final quality, with the shift up around $\alpha \approx 12$. Across all three phases the loss function is *one line of code* — the `flow_matching_step` from the next section — and the only thing the curriculum changes between phases is the clip shape passed in and the `shift` scalar, both of which the data loader and scheduler already carry. Try to run this curriculum with an $\epsilon$-prediction objective and a per-resolution min-SNR weighting and you are now maintaining and re-tuning the loss weights at every phase boundary, in the high-noise regime where they are most fragile, while also fighting the schedule-coupling instability at the larger latents. The curriculum is the place where flow matching's size-blindness stops being an elegant property and starts being the thing that lets the training run finish.

## 3. Why the objective's size-blindness is exactly what video needs

It is tempting to treat "the loss is the same" as a throwaway observation. It is not. Let me argue why size-blindness is specifically the property that makes flow matching the right objective for video, by contrast with what goes wrong otherwise.

The image-diffusion objective, $\epsilon$-prediction under a $\beta$-schedule, has a subtle coupling to the data dimension that shows up at the extremes of the schedule. The effective signal-to-noise ratio (SNR) at a timestep, and therefore how hard the denoising problem is at that timestep, depends on how the noise schedule was set — and the "right" schedule for a small latent is not the right one for a large one, because of an averaging effect we will derive in the schedule section. With $\epsilon$-prediction you fight this by reweighting the loss across timesteps (the SNR-weighting and min-SNR tricks from the image literature) and by retuning the $\beta$-schedule. Those are knobs that have to be re-tuned as the latent grows. They are tunable, but they are fragile, and at video's token counts the fragility bites: the regime where you want the model to spend most of its capacity (the high-noise, structure-forming steps) is exactly where the $\epsilon$-target's behavior is most sensitive to schedule choices.

Flow matching sidesteps the worst of this. The velocity target $x_1 - x_0$ has the same scale at every $t$ — it is literally a fixed vector for a given pair, independent of where along the path you sample. There is no point on the path where the target explodes or collapses. The loss does not need an SNR reweighting to be well-behaved across $t$; the target is already balanced. So instead of tuning a $\beta$-schedule and a loss reweighting and praying they compose at the new latent size, you tune exactly one thing: the distribution you draw $t$ from. That single knob — the timestep-sampling distribution — is the entire schedule story for flow matching, and it is the thing video shifts. We have collapsed a fragile multi-knob problem into a single, interpretable one.

![Matrix comparing flow-matching video training against DDPM video training across objective form, steps to good FVD, stability at huge token counts, and schedule control](/imgs/blogs/flow-matching-for-video-3.png)

This collapse is why the same objective works from a single image up to a minute-long clip with nothing but the $t$-distribution changing. It is also why the open recipe converged: once flow matching gave everyone a stable, size-blind objective with a single schedule knob, there was no reason for Wan, Mochi, and HunyuanVideo to each invent a bespoke diffusion schedule. They all train velocity, they all sample $t$ from a shifted distribution, and they all read out with an ODE solver. The convergence is a signal that the objective is genuinely the right fit for the regime.

#### Worked example: the stability difference at scale

Suppose you take a working image-diffusion recipe — $\epsilon$-prediction, a cosine $\beta$-schedule, min-SNR loss weighting — and you naively inflate it to video by stacking frames into a $C \times T \times H \times W$ latent and training. Two things tend to go wrong, and both trace to the schedule coupling. First, because the larger latent's effective SNR at a fixed timestep is higher (the averaging effect), the model spends too much of its capacity on easy, low-noise steps and underfits the high-noise structure-forming steps; your clips look locally sharp but globally incoherent — the dog is crisp but teleports across the field. Second, the min-SNR weights you tuned for $64 \times 64$ image latents are wrong for the video latent, and the loss landscape gets noisier; training is jumpier and you babysit the learning rate. Swap to flow matching with a token-count-aware shifted $t$-distribution and both problems soften at once: the velocity target is uniformly scaled so no reweighting is needed, and the shift directly puts more training mass on the high-noise steps the big latent was starving. In practice teams report this as "flow matching just trained more stably at the resolutions and durations we cared about," and the mechanism is exactly this collapse from many fragile knobs to one principled one.

## 4. The schedule shift: why video trains under more noise

Now the genuinely video-specific science. The objective is size-blind, but the *timestep-sampling distribution* is not — and it should not be. This is the single most important practical difference between image and video flow matching, and it is the thing people get wrong when they port an image model up to video. Let me derive why bigger latents need a schedule shifted toward higher noise, because the "why" makes the "how much" obvious.

Start with the image case. Stable Diffusion 3 popularized drawing the flow-matching timestep $t$ not uniformly but from a **logit-normal** distribution: you sample $u \sim \mathcal{N}(m, s^2)$ and pass it through the logistic sigmoid, $t = \sigma(u) = 1/(1 + e^{-u})$. With $m = 0$ this concentrates $t$ near the middle of $[0,1]$ and thins out the endpoints, putting the most training mass on the mid-noise steps where the hardest, most informative denoising happens. The mean $m$ slides the mass: negative $m$ toward low noise, positive $m$ toward high noise. That is the image schedule.

The video question is: should the mass sit in the same place? The answer is no, and here is the averaging argument that says why. Consider a fixed noise level — a fixed $t$ in the interpolation $x_t = (1-t)x_0 + t x_1$. The "difficulty" of the denoising problem at this $t$ is governed by how much of the clean signal $x_1$ survives relative to the noise $x_0$, i.e. by an effective signal-to-noise ratio. Now here is the key: a video latent has vastly more correlated voxels than an image latent. Neighboring frames are highly redundant — that is the whole prior that makes video learnable, which we argued in [representing video, redundancy and tokens](/blog/machine-learning/video-generation/representing-video-redundancy-and-tokens). When the model attends across this large, correlated tensor, the per-voxel noise partially **averages out**: independent Gaussian perturbations on correlated signal cancel in aggregate, so the *effective* SNR the network experiences at a fixed $t$ is higher on a big correlated latent than on a small one. Concretely, if you have $N$ tokens carrying correlated signal corrupted by independent noise, the aggregate signal-to-noise scales roughly as $\sqrt{N}$ relative to a single token, because the signal adds coherently while the noise adds in quadrature.

Let me make the $\sqrt{N}$ claim concrete with the simplest model that captures it, because the scaling is the whole derivation. Suppose the network is trying to recover one low-dimensional global quantity — say the position of the subject, or the camera's pan — that is encoded redundantly across many latent voxels. Model that quantity as a scalar signal $s$ present in each of $N$ correlated voxels, each corrupted by independent Gaussian noise of variance $\sigma^2(t)$ set by the timestep. An estimator that pools across the $N$ voxels (which is exactly what attention does) averages the noise: the signal stays at amplitude $s$ because it is coherent across voxels, while the pooled noise has standard deviation $\sigma(t)/\sqrt{N}$ because independent noises add in quadrature. The effective signal-to-noise ratio of the pooled estimate is therefore

$$
\text{SNR}_\text{eff} \;=\; \frac{s}{\sigma(t)/\sqrt{N}} \;=\; \sqrt{N}\,\cdot\,\frac{s}{\sigma(t)},
$$

a factor $\sqrt{N}$ larger than the per-voxel SNR $s/\sigma(t)$. A video latent has far more correlated voxels than an image latent, so its $N$ — and thus its effective SNR at a fixed $t$ — is much larger. To put the *effective* difficulty of the video problem at timestep $t$ back where the image problem was, you must raise the nominal noise $\sigma(t)$ until $\sigma(t)/\sqrt{N}$ matches the image case's $\sigma_\text{img}(t)$; that means scaling the noise by $\sqrt{N/N_\text{img}}$, i.e. shifting the schedule toward higher noise by exactly the square root of the token-count ratio. The derivation and the empirical SD3 shift formula agree on the $\sqrt{\cdot}$ scaling, which is the reassuring sign that the toy model has the right exponent even though real latents are messier than one redundant scalar.

The consequence is that a fixed timestep $t$ "looks easier" on a video latent than on an image latent. The model gets a cleaner effective view of the structure at the same nominal noise level. If you leave the schedule where it was, the model spends its capacity on steps that are, for this latent, already easy, and it under-trains the genuinely hard high-noise steps where global structure and motion are decided. The fix is to **shift the schedule toward higher noise** — push the logit-normal mean $m$ up, or equivalently apply a multiplicative shift to $t$ — so the training mass lands where the problem is actually hard for this latent size.

![Stack diagram explaining why a larger video latent averages out per-voxel noise so a fixed timestep looks easier and the schedule must shift toward higher noise](/imgs/blogs/flow-matching-for-video-4.png)

How much shift? The $\sqrt{N}$ argument gives the scaling directly. SD3's resolution-shift formula makes this precise for the image case and the video case extends it along the token axis. The shift is applied to $t$ as

$$
t' = \frac{\alpha\, t}{1 + (\alpha - 1)\, t},
$$

a monotone reparameterization of $[0,1]$ onto itself that, for shift factor $\alpha > 1$, pushes mass toward $t = 1$ (the high-noise end of the flow-matching convention where $x_0$ is noise). The shift factor itself scales with the square root of the token-count ratio relative to a base resolution:

$$
\alpha \;\propto\; \sqrt{\frac{N_\text{tokens}}{N_\text{base}}}.
$$

This is the same shape as SD3's spatial resolution shift — there it was $\sqrt{H W / (H_0 W_0)}$ because doubling the side length quadruples the token count — but for video $N_\text{tokens}$ now includes the temporal factor $T$. A clip is not just higher spatial resolution; it is many frames. So the video shift compounds the spatial shift with the temporal one: a long high-resolution clip has both more pixels per frame and more frames, and both push $\alpha$ up. This is why video models run *larger* shifts than even high-resolution image models. The schedule is not pushed up because video is "different" in some hand-wavy way; it is pushed up by a specific, derivable amount set by how many correlated tokens the latent holds.

![Matrix showing how latent token count rises with resolution and clip length and how the schedule shift factor and high-noise budget climb with it](/imgs/blogs/flow-matching-for-video-6.png)

Two practical notes that save you grief. First, the shift is a *training-time* choice and a *sampling-time* choice, and they should match: you train with $t$ drawn from the shifted distribution and you sample with the same shift baked into the scheduler. Mismatched train and sample shifts produce a model that is biased toward noise levels it was not trained on, and the symptom is washed-out or oversmoothed motion. Second, the shift interacts with clip length at inference: if your model was trained with a shift for 5-second clips and you ask it for a 10-second clip in one shot, the token count is now larger than the shift assumes and the high-noise steps are again under-served — this is one of the mechanisms behind quality degradation past the trained clip length, alongside the VAE's trained-length limits we discussed in the autoencoder post.

#### Worked example: reading a real scheduler config

Open the scheduler config of a video model in diffusers and you will see something like `shift: 7.0` or `shift: 17.0` on the `FlowMatchEulerDiscreteScheduler`. That number is exactly the $\alpha$ above (the libraries call it `shift`). A shift of 1.0 is the unshifted image baseline. A shift of 3.0 is a mild push you might see on short low-res clips. A shift in the 7-to-17 range is what you see on high-resolution, longer video models, and the magnitude tracks the token-count argument: HunyuanVideo and Wan-class models, which generate hundreds of thousands to over a million latent tokens, sit at the high end. If you fine-tune one of these on shorter clips and forget to lower the shift, you are training under a schedule tuned for a bigger latent than you are feeding it, and you will starve the low-noise detail steps — the fine texture and small motion will look mushy. The fix is to recompute the shift from your actual token count, not inherit the base model's blindly.

## 5. The flow-matching training step on a spacetime-DiT, in code

Enough theory; let us write the step. This is the inner loop of training a video flow-matching model, and it is shorter than you might expect because the objective does so little. The network is a [spacetime diffusion transformer](/blog/machine-learning/video-generation/video-diffusion-transformers) — we treat it as a black box `model(x_t, t, cond)` here and link out to that post for its internals. Everything around it is the flow-matching scaffolding.

![Graph of a single flow-matching training step that samples a shifted timestep, forms an interpolation point, runs the spacetime-DiT once, and regresses its velocity against the straight-path target](/imgs/blogs/flow-matching-for-video-7.png)

```python
import torch
import torch.nn.functional as F

def sample_shifted_logit_normal(batch_size, shift, m=0.0, s=1.0, device="cuda"):
    # Draw a base timestep from a logit-normal, then apply the multiplicative
    # resolution/token-count shift. shift == alpha in the derivation above.
    u = torch.randn(batch_size, device=device) * s + m       # N(m, s^2)
    t = torch.sigmoid(u)                                      # logit-normal in (0, 1)
    t = (shift * t) / (1.0 + (shift - 1.0) * t)               # push mass toward high noise
    return t                                                  # shape: (B,)

def flow_matching_step(model, x1, cond, shift):
    # x1: clean VIDEO latent from the causal 3D-VAE, shape (B, C, T, H, W)
    # cond: text/image conditioning passed straight through to the DiT
    B = x1.shape[0]

    # 1) Draw noise of the SAME spacetime shape as the latent.
    x0 = torch.randn_like(x1)                                 # (B, C, T, H, W)

    # 2) Sample a shifted timestep per clip in the batch.
    t = sample_shifted_logit_normal(B, shift, device=x1.device)

    # 3) Form the straight-path interpolation point. Broadcast t over C,T,H,W.
    t_b = t.view(B, 1, 1, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * x1                         # x_t = (1-t) x0 + t x1

    # 4) The straight-path velocity target is constant along the path.
    target = x1 - x0                                          # v = x1 - x0

    # 5) One forward pass of the spacetime-DiT predicts the velocity.
    v_pred = model(x_t, t, cond)                              # (B, C, T, H, W)

    # 6) Plain MSE. No SNR reweighting needed: the target is uniformly scaled.
    loss = F.mse_loss(v_pred, target)
    return loss
```

A few things to notice, because they are the whole point of the post made executable. There is no `for` loop over timesteps and no solver call — step 5 is a single forward pass, exactly as we argued. The target in step 4 is computed in closed form and does not touch the network. The only video-specific lines are the tensor shapes carrying a `T` dimension and, crucially, the `shift` in step 2 — strip the shift and `view`/`randn_like` differences and this is verbatim the image training step. And there is no loss reweighting in step 6: a uniformly scaled velocity target does not need it, which is the stability win cashed out as deleted code.

The training launch is standard PyTorch with `accelerate` for mixed precision and multi-GPU; the only video-flavored concern is that the spacetime latent and its activations are large, so you lean on gradient checkpointing and, for long clips, sequence parallelism.

```bash
# Multi-GPU flow-matching training launch. The video latent is large, so we run
# bf16, gradient checkpointing, and shard activations across GPUs.
accelerate launch \
  --mixed_precision bf16 \
  --num_processes 8 \
  train_video_fm.py \
  --vae "causal-3d-vae" \
  --model "spacetime-dit-xl" \
  --num_frames 49 --height 480 --width 720 \
  --schedule_shift 7.0 \
  --gradient_checkpointing \
  --sequence_parallel_size 2 \
  --batch_size 1 --gradient_accumulation_steps 8
```

Note `--schedule_shift 7.0`: that is the $\alpha$ from the derivation, set for this clip's token count, passed straight through to `sample_shifted_logit_normal`. Change `--num_frames`, `--height`, or `--width` and you should recompute it; the launcher should ideally derive it from the latent token count rather than make you type a constant, but most reference repos make you set it and most bugs come from forgetting to.

## 6. Sampling: integrate the ODE in a handful of expensive steps

Training regresses a velocity. Sampling integrates it. Because we trained a (near-)straight field, a coarse ODE solver suffices — and "coarse" is where the money is when each step is a spacetime forward pass.

The simplest integrator is forward Euler. Start from a noise draw $x_0 \sim \mathcal{N}(0, I)$ at $t = 0$, take $K$ steps of size $\Delta t = 1/K$, and at each step move along the predicted velocity:

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t).
$$

After $K$ steps you arrive at $x_1$, the generated clip latent, which the 3D-VAE decoder turns into pixel frames. That is the entire sampler. With a straight path, the local truncation error of Euler is small even for $K$ in the tens, because the true trajectory is close to the straight line Euler assumes between grid points. With diffusion's curved probability-flow ODE you would need a higher-order solver or many more steps to track the curvature; flow matching's straight path is what lets a first-order solver with few steps land in the right place.

In diffusers this is wrapped in `FlowMatchEulerDiscreteScheduler`, and a video sampling call looks like this:

```python
import torch
from diffusers import WanPipeline   # or MochiPipeline, HunyuanVideoPipeline, ...
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()          # the spacetime-DiT + 3D-VAE are large
pipe.vae.enable_tiling()                 # decode the big latent in spatial tiles

# The scheduler is a flow-match Euler solver. Its `shift` is the alpha we derived;
# `num_inference_steps` is the K in the Euler loop above. Straight paths let K be small.
print(type(pipe.scheduler).__name__)     # FlowMatchEulerDiscreteScheduler
print(pipe.scheduler.config.shift)       # e.g. 5.0 / 7.0 / 17.0 depending on the model

frames = pipe(
    prompt="a golden retriever running across a sunny field, cinematic",
    height=480, width=832,
    num_frames=81,                       # ~3.4 s at 24 fps after VAE temporal upsample
    num_inference_steps=30,              # K: 30 Euler steps over the straight field
    guidance_scale=5.0,
).frames[0]

export_to_video(frames, "dog_run.mp4", fps=24)
```

The line that matters for this post is `num_inference_steps=30`. On an image model you might run 28–50 and not think hard about it. On a video model, 30 versus 50 is the difference between, say, 4 minutes and nearly 7 on a single high-end GPU for a clip of this size, because each of those 20 extra steps is a full spacetime forward pass over ~100k+ tokens. The straight path is what makes 30 — and with distillation far fewer — actually land at acceptable quality.

A subtlety worth flagging: the scheduler's `shift` at sampling time should match the shift the model was trained under. The pipeline ships with the right value baked into its config, which is why you generally do not touch it; but if you swap schedulers or fine-tune, mismatching the sampling shift to the training shift is a classic source of "the base model looks great and my fine-tune looks washed out." Read the config; respect the shift.

Why does the straight path let Euler get away with so few steps, quantitatively? Euler's local truncation error per step is proportional to the second derivative of the trajectory — the curvature — times the squared step size, $\tfrac{1}{2}\lVert \ddot{x}_t \rVert (\Delta t)^2$. For a perfectly straight path $\ddot{x}_t = 0$ and Euler is *exact* at any step size; a single step lands on the target. Real flow-matching fields are not perfectly straight (the conditional-mean argument from section 1), so $\ddot{x}_t$ is small but nonzero, and the error grows gently as you coarsen the grid — gently enough that 20–30 steps sit comfortably in the flat part of the FVD-versus-steps curve. Diffusion's probability-flow ODE has a genuinely curved trajectory, larger $\ddot{x}_t$, so the same Euler grid accumulates more error and you either add steps or switch to a higher-order solver (a Heun or DPM-Solver variant) that approximates the curvature. This is the numerical statement of the whole argument: straightness is low curvature, low curvature is small Euler error, small Euler error is few steps, and few steps is cheap video. If you ever do need to push the step count very low without distillation, a second-order solver buys back some of the curvature error for one extra function evaluation per step — but on video, where the function evaluation is the entire cost, doubling evaluations to halve steps is usually a wash, and distillation is the better lever.

#### Worked example: Euler error on a near-straight path

Make the curvature concrete. Suppose along a particular generation the trajectory bows away from the straight line by a small amount, so that integrating with many steps gives a reference clip and integrating with few steps gives a slightly different one. With a near-straight path the per-step error scales as $(\Delta t)^2$, so going from 50 steps ($\Delta t = 0.02$) to 25 steps ($\Delta t = 0.04$) inflates the per-step error by $4\times$ but halves the number of steps, for a net roughly $2\times$ rise in accumulated error — small enough that the FVD barely moves on gentle-motion content because you started deep in the flat region. Go from 25 to 8 steps and the per-step error inflates by roughly $(0.125/0.04)^2 \approx 10\times$ while you take a third as many steps, so accumulated error rises a few-fold and now you can see it on fast motion as slightly smeared trajectories. The takeaway is the shape, not the exact figures: the error is forgiving down to the knee because of straightness and then climbs, and where the knee sits is set by how straight your particular model's path is — reflow and distillation push the knee lower by making the path straighter, which is the whole reason they exist.

## 7. Few-step sampling and how it composes with distillation

Straight paths get you to the tens of steps for free, from the objective alone. Getting to the single digits — or to one step — is where flow matching composes with distillation, which we cover in depth in [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation). The relationship is worth making precise here because it explains why the field invests so heavily in the step count.

The marginal value of cutting a step is higher for video than for any other generative regime, and it is higher precisely because the step is so expensive. Put numbers on it. Suppose one spacetime forward pass over our 720p clip takes 5 seconds on an H100 (large clips genuinely cost this). At 50 steps that is 250 seconds per clip; at 30 steps, 150 seconds; at 8 steps, 40 seconds; at 4 steps, 20 seconds; at 1 step, 5 seconds. The render time is linear in the step count and the per-step constant is huge, so every halving of the step count halves the wall-clock and the cost. For an image model where a step is milliseconds, going from 50 to 4 steps saves you a fraction of a second nobody notices; for video it is the difference between a tool you can put in an interactive product and a batch job you wait minutes for.

Flow matching's straight path is the *foundation* that makes aggressive step reduction viable, and distillation is the *amplifier*. Two families build on top. **Rectification** (the "reflow" procedure from the rectified-flow line) literally re-straightens the learned path: you generate noise-data pairs with the trained model, then retrain on those pairs, and because the new pairs are already nearly collinear the resulting field is straighter still, so even coarser solvers — down to a few steps — track it accurately. **Step distillation** (consistency-style and adversarial-style methods adapted to video) trains a student to jump along the path in one or a few steps, matching what the teacher produces in many; because the teacher's path is already straight, the student has an easy target to hit. Both stack on flow matching rather than replace it. You train the base model with flow matching for stability and a straight path, then distill for the few-step student you actually serve.

It is worth being concrete about why distillation composes so naturally with flow matching specifically, rather than treating "and then you distill" as a black box. A step-distillation student is trained to take a big jump along the trajectory — from $x_t$ at one timestep to a point much later (or all the way to $x_1$) in a single network call — and to land where the multi-step teacher would have landed. The student's job is easy exactly to the degree that the teacher's trajectory is predictable from $x_t$, and a straight trajectory is maximally predictable: if the path from $x_t$ is a straight line, the endpoint is simply $x_t$ plus the remaining time times the (constant) velocity, which is the easiest possible function for the student to approximate. On a curved diffusion path the endpoint depends on the whole curve between here and there, which the student has to internalize implicitly; on a straight flow-matching path it depends only on the local velocity, which the student already has. This is the mechanical reason consistency-style and adversarial-style distillation hit lower step counts on flow-matching teachers than on diffusion teachers, and it is why the open video models that want real-time inference (the LTX line, distilled Wan and Hunyuan variants) build their few-step students on flow-matching bases rather than retrofitting diffusion ones.

The reflow procedure deserves its own mention because it attacks the problem from the training side rather than the inference side. After training a flow-matching model, you use it to generate a large set of noise-to-data pairs by actually integrating the ODE, then you *retrain* a fresh flow-matching model on those pairs — but now pairing each generated $x_1$ with the specific $x_0$ that produced it, rather than a random independent noise draw. Because those pairs were produced by an already-straight-ish flow, the straight line between them is a better approximation to the true transport than a random pairing was, so the retrained field is straighter still. Iterate and the path approaches a genuine straight line, at which point a single Euler step suffices. Reflow costs a generation pass over a large dataset and a retraining run, which on video is expensive, so in practice teams do one or two reflow rounds rather than many, and often combine a single reflow with a step-distillation pass for the final student. The point for this post is that both reflow and distillation are *amplifiers of the straightness that flow matching already provides* — neither would work nearly as well on a curved diffusion base.

There is a quality floor to respect, and honesty demands stating it. Few-step sampling is not free quality — it is a trade. At very low step counts you lose some fine motion detail and some of the high-frequency texture the extra steps would have refined, and the loss is most visible in scenes with large, fast motion where the straight-line approximation between coarse grid points is least accurate. The right framing is: straight paths move the quality-versus-steps curve far to the left so that the *knee* — the point past which fewer steps starts to hurt — sits at a much lower step count than diffusion's. For many video models that knee is around 20–30 steps for the base model and 4–8 for a distilled student, and below the knee you pay in motion fidelity. Measure where your knee is on your own content; do not assume the paper's number transfers to your motion distribution.

#### Worked example: choosing a step count under a latency budget

You are serving I2V for a product with a 30-second latency budget per 4-second clip on a single H100, and your spacetime forward pass is about 1.5 seconds at your resolution. Base flow-matching sampling at 30 steps costs $30 \times 1.5 = 45$ seconds — over budget. You have two levers. Lever one: drop to 18 steps ($27$ seconds, in budget) and accept slightly softer fast-motion detail; for an I2V product where the first frame is supplied and motion is modest this is often imperceptible. Lever two: keep quality and distill to an 8-step student ($12$ seconds, comfortably in budget) at the cost of a one-time distillation run and a small, content-dependent quality gap you validate against held-out clips. The decision rule: if your motion is gentle, just lower the steps; if you need both speed and large-motion fidelity, distill. Either way, the reason you *have* these options is that flow matching put the quality knee at a low step count to begin with — on a curved diffusion path neither 18 base steps nor an 8-step student would hold up.

## 8. FM-video versus DDPM-video: the comparison that decided the recipe

Let me put the head-to-head in one place, because the migration from diffusion to flow matching for video was a deliberate engineering decision and the table is the argument.

The two objectives produce comparable peak quality when both are well-tuned — flow matching is not magic that makes FVD plummet on its own. What flow matching changes is the cost and stability profile around that quality. The properties below are the ones teams actually weighed.

![Before and after comparison of image flow matching and video flow matching showing the loss and target stay identical while the latent shape grows and the schedule shifts toward higher noise](/imgs/blogs/flow-matching-for-video-5.png)

| Property | DDPM-video ($\epsilon$-pred) | FM-video (rectified flow) |
| --- | --- | --- |
| Objective form | $\epsilon$-prediction, schedule-coupled | velocity regression, size-blind |
| Target stability across $t$ | needs SNR/min-SNR reweighting | uniformly scaled, no reweighting |
| Steps to good FVD (base) | ~50–250 | ~20–50 |
| Steps after distillation | few-step possible but harder | few-step / 1-step, straighter to start |
| Schedule control | $\beta$-schedule + loss weights (multi-knob) | one knob: shifted $t$-distribution |
| Stability at $>10^6$ tokens | fragile, retune per resolution | stable, schedule shift derived from token count |
| Sampler | DDPM/DDIM/DPM-Solver | FlowMatchEuler (ODE), coarse-solver friendly |

The decisive rows are "steps to good FVD," "schedule control," and "stability at huge token counts." On steps, flow matching's straight path roughly quartered the base step count, which on video's expensive forward pass is a direct multiple on render cost. On schedule control, collapsing the $\beta$-schedule-plus-loss-weights tuning problem into a single shifted-$t$ knob made the recipe portable across resolutions and durations — you compute one shift from your token count instead of re-tuning a schedule. On stability, the uniformly scaled velocity target survived the scale-up to million-token latents where the $\epsilon$-target's schedule sensitivity bit hardest. None of these is a quality claim; all of them are cost-and-engineering claims, and at video's compute budget cost-and-engineering is what ships.

How would you measure the step-count claim honestly? Fix everything but the objective: same data, same spacetime-DiT, same 3D-VAE, same compute budget, same evaluation clips and seeds. Train one model with $\epsilon$-prediction plus a tuned schedule and one with flow matching plus a token-count shift. Then sweep `num_inference_steps` for each and plot FVD against steps. The honest comparison is not "FM beats DDPM at 30 steps" — it is the *shape* of the two curves: flow matching's FVD-versus-steps curve has its knee at a much lower step count, so it reaches its quality plateau with far fewer of those expensive passes. Report the knee, the steps to within 5% of the plateau, and the wall-clock per clip on a named GPU; a single-step-count comparison hides the whole point. And mind the FVD caveats from [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation): FVD is noisy, so use a large enough sample set and fixed seeds, and pair it with VBench dimensions because a model can sometimes trade motion (dynamic degree) for stability and game a single number.

## 9. Applying it on a spacetime-DiT: what the velocity field has to learn

The training step in section 5 treats the spacetime-DiT as a black box. It is worth one section on what that box has to do differently for video, because the velocity field's job is harder than in the image case and that hardness is why the schedule shift earns its keep.

In the image case, $v_\theta(x_t, t)$ predicts a velocity that, integrated, moves a noisy image toward a clean one — a purely spatial task. In the video case, $v_\theta$ predicts a velocity over the whole $T \times H \times W$ latent, and to get it right the network must model how voxels relate *across time*, not just across space. A correct velocity at a voxel in frame 10 depends on what frames 9 and 11 are doing, because temporal coherence is exactly the constraint that frame 10 be frame 9 plus a little motion. The spacetime-DiT supplies this through spacetime attention — full or factorized, as we dissected in [from image diffusion to video diffusion](/blog/machine-learning/video-generation/from-image-diffusion-to-video-diffusion) — and the velocity it regresses is therefore a *temporally coherent* field, not a stack of independent per-frame velocities.

This is where the high-noise emphasis of the shifted schedule pays off concretely. The decisions that matter most for coherence and motion — where the camera is, how the subject moves, the global layout of the scene — are made at high noise, when the latent is mostly Gaussian and the network is sketching structure from almost nothing. The fine texture is filled in at low noise, near $t = 1$. Because the video latent's averaging makes high-noise steps effectively easier (the section-4 argument), an unshifted schedule would let the network coast through exactly the steps where temporal structure is decided, and the symptom is the worst kind of video failure: locally crisp frames that do not cohere into believable motion. Shifting the schedule up forces more training and sampling budget onto those structure-forming steps, which is to say onto the steps that buy temporal coherence. The schedule shift is not a generic "more noise is better" heuristic; it is specifically buying budget for the steps that decide whether the dog runs or teleports.

There is a stress test worth naming. Push the clip length past what the schedule shift assumes — ask a model trained and shifted for 5 seconds to produce 15 seconds in one shot — and the token count exceeds what the shift was set for, so the high-noise steps are under-served again and global coherence degrades: identity drifts, the scene reorganizes, motion loses its through-line. This is one of several mechanisms (alongside the VAE's trained clip length and attention's reach) behind the long-video wall, and it is why long video is handled by autoregressive or sliding-window rollout rather than one-shot generation, a topic we take up separately. The flow-matching schedule is tuned for a clip-length regime, and stepping outside that regime is stepping outside the schedule's assumptions.

There is a second, more subtle interaction worth understanding: the velocity field has to be coherent *across the timesteps of the sampling trajectory*, not just across the spatial-temporal voxels at one timestep. When the solver takes its first coarse Euler step at high noise, it commits to a coarse layout — where the dog is, which way it faces, where the camera sits. Every subsequent step refines within that commitment; the later, lower-noise steps cannot undo a bad high-noise decision, only decorate it. This is why a model that under-trains its high-noise steps fails in a way no amount of low-noise refinement repairs: the structure was decided wrong early and the rest of the trajectory faithfully renders the wrong structure in high fidelity. The schedule shift is, in this light, a way of buying the model more competence at the exact steps whose decisions are irreversible. It is the generative analog of "measure twice, cut once" — spend budget where the commitment is made, because the later steps are committed to whatever the early steps chose. For video this is more acute than for images because the early commitment now includes a motion trajectory, not just a static layout, and a wrong motion commitment at high noise is the teleporting-dog failure rendered crisply.

A useful way to sanity-check this on a trained model is to vary only the high-noise portion of the schedule at inference and watch what changes. If you increase the number of steps spent in the high-noise band (the early part of the trajectory) and the global coherence improves while fine texture stays put, your model was high-noise-starved and the shift was too low for your latent. If extra high-noise steps do nothing and only low-noise steps sharpen things, your shift is roughly right and you are detail-limited, not structure-limited. This diagnostic is cheap — it is just a scheduler tweak, no retraining — and it tells you which end of the schedule your quality bottleneck lives at, which is exactly the information you need to set the shift or decide whether the problem is the objective at all versus the VAE or the attention.

## 10. Case studies: who trains with flow matching and how

The strongest evidence that flow matching is the right objective for video is that the open frontier converged on it independently. Here are the concrete data points, stated at the precision the public reports support — and flagged as approximate where the reports are vague, because fabricating a number would betray the whole "measured results" mandate of this series.

![Matrix listing open video models trained with flow matching alongside their objective, schedule shift, and sampler choices](/imgs/blogs/flow-matching-for-video-8.png)

**Stable Diffusion 3 set the template (image, but the source of the recipe).** Esser et al. (2024) trained SD3 with conditional flow matching, the logit-normal timestep sampling, and the resolution-dependent shift $t' = \alpha t / (1 + (\alpha - 1)t)$. This is the recipe the video models inherited; the SD3 paper is where the shift formula and the logit-normal schedule were popularized for large-scale generation, and reading it is the fastest way to understand the image baseline the video shift extends.

**Mochi 1 (Genmo, 2024)** is an openly released text-to-video model trained with a flow-matching objective on a spacetime-DiT (their "AsymmDiT") over a causal 3D-VAE latent. It samples with a flow-match Euler solver. It is a clean example of the converged recipe — causal 3D-VAE plus spacetime-DiT plus flow matching — at a large parameter scale (on the order of 10B parameters), and its open weights make it a good study object for the objective in practice.

**Wan 2.x (Alibaba, 2024–2025)** trains with flow matching / rectified flow and ships with a `FlowMatchEulerDiscreteScheduler` carrying a resolution-dependent shift in its diffusers config; the 14B text-to-video and image-to-video variants are the workhorses of the open self-hosting community. The diffusers `WanPipeline` exposes the scheduler `shift` directly, which is the cleanest way to see the token-count argument cashed out in a real config.

**HunyuanVideo (Tencent, 2024) and HunyuanVideo-1.5 (2025)** likewise train with a flow-matching objective on a causal-3D-VAE-plus-DiT stack and sample with a flow-match solver under a shifted schedule. HunyuanVideo is one of the largest openly described video models (on the order of 13B parameters), and the 1.5 line pushes toward longer durations on consumer hardware; both sit on the flow-matching recipe.

**CogVideoX (Zhipu/THU, 2024)** is the partial exception that proves the rule: its earlier releases use a v-prediction-style diffusion objective with the `CogVideoXDDIMScheduler`, and later work in the line moves toward the flow-matching/rectified-flow recipe. It is worth including precisely because it shows the migration in progress — the field did not start on flow matching, it converged onto it, and CogVideoX is a checkpoint of that convergence.

The pattern across these is unmistakable: causal 3D-VAE for the latent, spacetime-DiT for the denoiser, and flow matching with a token-count-aware shifted schedule for the objective. When four independent teams land on the same objective for the same regime, the objective is telling you something about the regime — here, that size-blindness, target stability, and straight paths are exactly what a million-token, expensive-per-step latent rewards.

The convergence is also visible in the tooling, which is its own kind of evidence. Two years ago, porting an image model to video meant reimplementing the schedule and reweighting from scratch; today the diffusers `WanPipeline`, `MochiPipeline`, and `HunyuanVideoPipeline` all instantiate the same `FlowMatchEulerDiscreteScheduler` class, differing only in the `shift` value baked into each config, and the same `export_to_video` writes the frames out. That a single scheduler class serves models from three different labs is the practical face of the convergence: the objective is standard enough that the library does not need a bespoke schedule per model, only a per-model scalar. When you are choosing what to build on, this matters operationally — the recipes, the distillation methods, the LoRA tooling, and the caching tricks all assume a flow-matching base, so building on it puts you in the center of the ecosystem rather than off to the side maintaining a divergent stack. Picking the diffusion objective today would mean reimplementing, for yourself, machinery the flow-matching ecosystem already ships.

The historical arc is worth stating plainly because it guards against a false impression. Video generation did not begin with flow matching — the early video diffusion models (VDM, Make-A-Video, the first Stable Video Diffusion) were $\epsilon$-prediction diffusion models, and they worked. Flow matching won not because diffusion *failed* at video but because, as the latents grew to a million tokens and as serving latency became the product constraint, flow matching's particular bundle of properties — size-blind objective, single schedule knob, straight path, clean distillation — paid off more and more relative to diffusion's. The migration is a story of a regime growing into an objective's strengths, not of one objective being wrong and another right. That framing also tells you when the choice could shift again: if a future regime rewards stochastic sampling or a different path geometry, the field could move once more. For the 2024–2026 video regime, though, the evidence is one-directional, and flow matching is the answer.

A note on numbers and honesty. Parameter counts above (Mochi ~10B, HunyuanVideo ~13B) are from the public reports and are approximate to the leading digit; exact figures vary by variant and release. I am deliberately *not* quoting precise FVD or VBench numbers for each model here, because those numbers depend heavily on the evaluation protocol (sample set, resolution, seed, which VBench version) and quoting a single headline figure without its protocol would be the kind of fabrication this series forbids. For protocol-controlled comparisons, see the model reports and the VBench leaderboard, and treat any single number you see quoted without its evaluation setup with suspicion — the dynamic-degree-versus-stability gaming problem from [the metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) means a single number is easy to cherry-pick.

## 11. When to reach for flow matching (and when the choice does not matter)

A decisive recommendation, because "use flow matching" is right but "use flow matching for these reasons in these cases" is more useful.

**Reach for flow matching when you are training a video model from scratch or substantially.** This is the default and the right one. The size-blind objective, the single schedule knob, the straight path, and the few-step sampling are all aligned with video's pain, and the open ecosystem has standardized on it, so you inherit working scheduler configs, distillation recipes, and tooling. There is no good reason in 2026 to train a new video diffusion model on $\epsilon$-prediction with a hand-tuned $\beta$-schedule; you would be re-solving problems flow matching already dissolves.

**Reach especially hard for flow matching when your step count is your latency or cost bottleneck.** If you are building anything interactive or cost-sensitive, the straight path's low quality knee and its clean composition with distillation are decisive. The path from a slow batch tool to a near-real-time product runs through few-step sampling, and few-step sampling runs through flow matching plus distillation. If render latency is fine and you batch overnight, this matters less.

**The choice matters less when you are only fine-tuning an existing checkpoint on a narrow domain.** If you take a pretrained Wan or HunyuanVideo and LoRA-tune it on your style, you inherit its objective and schedule; you are not choosing flow matching versus diffusion, you are living inside the base model's choice. Here the actionable rule is not "pick flow matching" but "respect the base model's shift" — recompute it if you change clip length or resolution, and otherwise leave it alone.

**Do not over-shift the schedule on short, low-resolution clips.** The shift is a function of token count; a short 256p clip does not need a shift of 17, and applying a high shift to a small latent starves the low-noise detail steps and gives you mushy texture. Compute the shift from your actual latent token count. The most common self-inflicted wound is inheriting a high shift from a big base model and applying it to a small fine-tuning latent.

**Do not push few-step sampling below your measured quality knee for large-motion content.** Few steps is a free win up to the knee and a quality cliff past it, and the cliff is steepest where motion is fast. If your content is gentle motion or I2V from a supplied first frame, push the steps low; if it is fast, complex motion, find your knee empirically and respect it, or distill rather than just dropping steps.

**Do not expect flow matching to fix coherence problems that are really VAE or attention problems.** Flow matching trains the denoiser well, but if your clips flicker because the 3D-VAE's temporal compression is too aggressive or your temporal attention is too weak, no objective will save you. Diagnose where the failure lives — the [VAE post](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression) and the architecture post cover those — before blaming the objective.

## 12. Key takeaways

- **The flow-matching loss is identical for image and video.** Regress the velocity $v_\theta(x_t, t)$ against the straight-path target $x_1 - x_0$ at a random interpolation point $x_t = (1-t)x_0 + t x_1$. Making the latent a $C \times T \times H \times W$ clip changes the tensor shapes and nothing in the objective — that size-blindness is exactly why flow matching scales to video.
- **Training is simulation-free and the target is uniformly scaled.** One forward-backward pass per step, no rollout, no solver in the loop, and no SNR loss reweighting needed because the velocity target has the same scale at every $t$. That cleanliness is the training-stability win at video's million-token latents.
- **Video forces a schedule shifted toward higher noise, and the shift is derivable.** A large correlated latent averages out per-voxel noise, so a fixed timestep looks easier; the fix is to shift the logit-normal $t$-distribution up by a factor $\alpha \propto \sqrt{N_\text{tokens}/N_\text{base}}$ via $t' = \alpha t/(1 + (\alpha-1)t)$. Longer, higher-resolution clips run larger shifts.
- **The shift buys budget for the steps that decide coherence.** Structure and motion are set at high noise; shifting up forces training and sampling budget onto exactly those steps, which is what keeps the dog running instead of teleporting.
- **Straight paths sample in far fewer steps, and on video that is the whole game.** Each step is a full spacetime forward pass costing seconds, so the base step count dropping from ~50–250 to ~20–50 is a direct multiple on render cost and the path to interactive latency.
- **Few-step is a knee, not a free lunch.** Straight paths move the quality-versus-steps knee far left; below the knee you pay in fast-motion fidelity. Distillation (reflow, consistency, adversarial) amplifies the win down to a few or one step, and it composes with flow matching because the teacher's path is already straight.
- **Train and sample under the same shift.** Mismatched train/sample shifts bias the model toward noise levels it never saw and produce washed-out motion. When fine-tuning, recompute the shift from your actual token count rather than inheriting the base model's.
- **The open frontier converged on this recipe.** Wan, Mochi, HunyuanVideo, and (increasingly) CogVideoX all train velocity with a shifted schedule and sample with a flow-match Euler solver over a causal-3D-VAE-plus-spacetime-DiT stack — four independent teams landing on the same objective is the regime telling you it is the right one.

## Further reading

- **Lipman, Chen, Ben-Hamu, Nickel, Le — "Flow Matching for Generative Modeling" (2023).** The foundational paper: the conditional-flow-matching objective, the simulation-free training, and the marginal-velocity result that makes per-pair straight lines recover the global flow.
- **Liu, Gong, Liu — "Rectified Flow" (2023).** The straight-path / reflow construction that makes the learned trajectory nearly linear and enables few- and one-step sampling.
- **Esser et al. — "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (Stable Diffusion 3, 2024).** Where the logit-normal timestep sampling and the resolution-dependent shift formula were popularized at scale; the image baseline the video shift extends.
- **Genmo — Mochi 1 technical report (2024)** and the **HunyuanVideo report (Tencent, 2024)** and the **Wan technical report (Alibaba, 2024–2025).** The open video models built on the causal-3D-VAE + spacetime-DiT + flow-matching recipe; read these for real scheduler configs and shift values.
- **Peebles & Xie — "Scalable Diffusion Models with Transformers" (DiT, 2023)** and **Brooks et al. — Sora technical report (2024).** The spacetime-DiT backbone that flow matching trains, and the scaling thesis behind it.
- **🤗 `diffusers` documentation — `FlowMatchEulerDiscreteScheduler` and the video pipelines (`WanPipeline`, `MochiPipeline`, `HunyuanVideoPipeline`).** The runnable APIs, the `shift` and `num_inference_steps` flags, and `export_to_video`.
- **Within this series:** start from [why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard), see the backbone in [video diffusion transformers](/blog/machine-learning/video-generation/video-diffusion-transformers) and the latent in [video autoencoders and spatiotemporal compression](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression), then follow the step-count thread into [efficient and real-time video generation](/blog/machine-learning/video-generation/efficient-and-real-time-video-generation) and the converged recipe in [the open video frontier: Wan, HunyuanVideo, CogVideoX](/blog/machine-learning/video-generation/the-open-video-frontier-wan-hunyuanvideo-cogvideox), and tie it together in [building with video generation: the playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
- **Link out to the image series** for the foundations: [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) for the full derivation and [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo) for the schedule and parameterization background the shift builds on.
