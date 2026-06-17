---
title: "Physics and the Limits of Learned Simulation: Where Video Models Break"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A rigorous, skeptical, and fair look at what video models do and do not understand about the physical world — the catalog of physics failures, why they happen, the pattern-matching versus simulation debate, the benchmarks that measure it, and a runnable harness that quantifies a broken bounce instead of eyeballing it."
tags:
  [
    "video-generation",
    "diffusion-models",
    "video-diffusion",
    "world-models",
    "physics",
    "simulation",
    "evaluation",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Video Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/physics-and-the-limits-of-learned-simulation-1.png"
---

Drop a ball in a generated video and watch it carefully. For the first half-second it falls, and it falls beautifully — the motion blur is right, the shadow tracks underneath it, the lighting is consistent. Then it hits the floor and one of three things happens. Sometimes it bounces back up to almost exactly the height it fell from, gaining energy it never had. Sometimes it sinks half into the floor and stops, as if the floor were soft. And sometimes, in the worst and most revealing case, a second ball quietly fades into existence in the corner of the frame, because the model decided that *this kind of clip* tends to have two balls in it. Every one of those frames is sharp. Every transition is smooth. The clip would pass a casual glance. And it is physically wrong in a way no real camera pointed at a real ball could ever be.

That gap — between a video that *looks* right frame to frame and a video that *is* right as physics — is the subject of this post, and it sits at the dead center of the most consequential debate in the field: are these models world simulators, or are they extremely good pattern matchers wearing a simulator's clothes? The [Sora technical report](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) titled itself *video generation models as world simulators* and bet that scaling the recipe would make physical understanding emerge. The clips it shipped were a generation ahead of anything public. They were also, in the same reel, full of liquids that un-poured themselves and cookies that got bitten with no bite mark left behind. This post is the careful, technical, unsentimental read of that tension.

![Side-by-side contrast of a video model sampling likely next pixels from a learned distribution versus a simulator advancing an explicit physical state through a conservation-preserving transition function](/imgs/blogs/physics-and-the-limits-of-learned-simulation-1.png)

By the end you will be able to do four concrete things. First, you will have a precise, named **catalog of the physics failures** — object permanence violations, non-conservation, gravity and momentum errors, broken contact and collision, implausible deformation, and the perennial hands and text — and you will know which are fundamental and which are merely hard. Second, you will understand *why* they happen, framed formally: a video model learns a conditional distribution over pixel sequences, $p(\text{frames})$, and not an explicit physical state with causal dynamics, $s_{t+1} = f(s_t, a_t)$ — and **nothing in the next-frame objective enforces a conservation law**. Third, you will know how the field *measures* this — the Physics-IQ and VideoPhy benchmarks, the probing and intervention studies — and what those measurements actually reveal versus what they are claimed to reveal. Fourth, you will have a small **runnable evaluation harness** in PyTorch that quantifies a broken bounce — tracking a centroid against $\tfrac{1}{2} g t^2$ and counting objects per frame — so you can stop eyeballing physics and start measuring it. We will keep coming back to a single running example, a dropped and bouncing ball, because it is the simplest scenario that breaks every model in print, and because a ball is something you can *measure*.

A word on stance before we begin. This is not a takedown. Video models are genuinely, surprisingly good at physics-adjacent coherence — far better than a naive frame-interpolator, and in some narrow regimes good enough that the failures are the interesting part precisely because they are exceptions. The honest position is neither "it's a world simulator" nor "it's a stochastic parrot for pixels." It is somewhere specific in between, and the entire value of this post is locating that *somewhere* precisely. The recurring frame of [this series](/blog/machine-learning/video-generation/why-video-generation-is-hard) holds throughout: video is spatial generation times temporal coherence under a brutal compute budget, and physical understanding — if it exists at all — is an *emergent byproduct* of optimizing temporal coherence, never a thing the objective asks for directly. That distinction is the whole game.

## 1. The distinction that everything hinges on: $p(\text{frames})$ versus $s_{t+1} = f(s_t, a_t)$

Before cataloging failures, we have to be ruthlessly precise about what a video model is computing, because almost every confused argument about "does it understand physics" comes from blurring two very different objects. Let me state them formally and then never let go of the distinction.

A **generative video model** learns a probability distribution over sequences of frames. Given some conditioning $c$ (a text prompt, a first frame) and the frames produced so far, it represents

$$
p_\theta(x_{1:T} \mid c) = \prod_{t} p_\theta(x_t \mid x_{<t}, c),
$$

or, in the diffusion formulation this series actually uses, a denoiser that turns noise into a sample from $p_\theta(x_{1:T} \mid c)$ by iteratively estimating a score $\nabla_x \log p_\theta$. (We will not re-derive the diffusion mechanics — that is the job of the [DDPM post](/blog/machine-learning/image-generation/the-math-of-ddpm) and the [score-based view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) in the image series. Take the sampler as given.) The crucial point is the *type signature*: the model is a machine that, given context, produces **a plausible-looking next frame** — one that is high-probability under the distribution of pixel sequences it was trained on. Its native currency is pixels and their statistics.

A **simulator** is a different kind of object entirely. It carries an explicit **state** $s_t$ — positions, velocities, masses, temperatures, the actual degrees of freedom of the system — and advances it through a **transition function**

$$
s_{t+1} = f(s_t, a_t),
$$

where $a_t$ is any external action or force, and $f$ is typically derived from physical law: Newton's equations, the Navier–Stokes equations, a rigid-body solver. A renderer $g$ then turns state into pixels, $x_t = g(s_t)$, but that is a *separate, downstream* step. The physics lives in $f$, and $f$ has a property the video model's $p_\theta$ does not: it conserves things. Mass in equals mass out. Momentum is preserved across a collision up to the coefficient of restitution. Energy does not spontaneously appear. These are not soft tendencies the simulator learned from data; they are hard invariants baked into the form of $f$.

Here is the load-bearing observation, and it is almost embarrassingly simple once stated: **the video model has no $s_t$.** There is no variable inside it that *is* the ball's position and velocity, advanced by a rule that conserves momentum. There is a latent activation that *correlates* with the ball's appearance, denoised toward whatever pixel configuration is most likely given the training distribution. When the ball bounces correctly, it is because clips of bouncing balls were common enough in training that "ball goes down, ball comes up a bit lower" is the high-probability continuation. When it bounces *wrong* — recovering full height, gaining energy — it is because the model has no energy variable to conserve; it only has a sense of what bouncing footage tends to look like, and that sense is approximate. The introduction's figure draws exactly this contrast: on one side a model picking likely pixels with no state to read out, on the other a simulator carrying explicit positions and momenta forward under a law — and the whole rest of this post elaborates why that one structural difference produces every failure we are about to catalog.

It is worth dwelling on *why* the appearance-correlation route gets bouncing approximately right at all, because that is what makes the failures surprising rather than expected. Real footage is saturated with falling and bouncing objects, and the visual statistics of that footage encode a great deal of real physics implicitly: objects that go up come back down, fast things blur more, shadows track feet, water darkens what it wets. A model that fits those statistics absorbs a *compressed, lossy summary* of the dynamics — enough to reproduce the look of a fall, not enough to compute its exact arc. This is the crucial nuance that keeps us from the lazy "it's just a parrot" conclusion: the statistics of pixels *do* contain physics, because the physical world generated the pixels in the first place, so a good enough fit to the pixels recovers *some* physics for free. The question is only how much, how reliably, and where it breaks — and the answer, which the next sections earn, is "a surprising amount, unreliably, and exactly at the conserved quantities the fit had no reason to nail."

This reframes the entire "understanding" debate into something answerable. The question is not the unfalsifiable "does it understand physics?" It is the sharp, empirical: **does the learned $p_\theta$ implicitly encode something functionally equivalent to a state $s_t$ and a transition $f$, accurate enough to behave like a simulator on novel inputs?** That is a question you can probe, test, and measure — and the bulk of this post is about how the field tries to, and what the answers look like. The short version, which we will earn: it encodes a *partial, leaky, interpolative* approximation of $f$ over the region of state-space that the training distribution covered densely, and it falls apart outside that region in exactly the way an interpolator does and a simulator does not.

One more piece of formalism worth pinning down, because it explains *why the failures are systematic rather than random*. The training objective — whether you write it as maximum likelihood or as the diffusion denoising loss $\mathcal{L} = \mathbb{E}\,\lVert \epsilon - \epsilon_\theta(x_t, t, c)\rVert^2$ — is a statement about **pixel reconstruction**. It rewards the model for assigning high probability to pixel sequences that resemble training data. There is *no term* in that loss that measures whether mass was conserved, whether momentum was preserved across a collision, whether the number of objects stayed constant. The gradient never once tells the model "you lost a liter of water between frame 9 and frame 10." So the model has no pressure to get conservation right except insofar as conservation-respecting clips happened to dominate the training set — which they do *on average*, but with enough exceptions and enough out-of-distribution gaps that the model's grip on any specific invariant is weak. Failures are systematic because the *absence* of the constraint is systematic. We will make this concrete in Section 4.

## 2. The catalog of physics failures

Let me lay out the failure families precisely, because "video models are bad at physics" is too vague to be useful and too easy to dismiss with a cherry-picked good clip. Each family below is distinct, has a distinct cause, and — crucially for the rest of this post — admits a distinct, mechanical test. The figure organizes them as a grid of failure × cause × test; the prose gives each one its due.

![Matrix of physics failure families crossed with what breaks, why it happens, and how to test each one, spanning permanence, conservation, gravity, contact, and deformation](/imgs/blogs/physics-and-the-limits-of-learned-simulation-2.png)

**Object permanence violations.** Things appear and vanish. A person walks behind a pillar and a *different* person walks out the other side; a dropped object disappears mid-fall; a cup on a table is simply gone three frames later; a bird splits into two birds and then merges back into one. The deep cause is that the model has no persistent object registry — no notion that "the cup" is a single entity with continuity of identity across time. It has a per-frame field of pixel likelihoods, lightly coupled across frames by temporal attention. When an object is occluded, the information about its existence has to survive purely in the latent's memory of recent frames, and that memory is finite, lossy, and biased toward whatever is statistically common. Permanence is *frequently* respected — because most training clips do conserve objects — but it is never *guaranteed*, and the failure rate rises sharply under occlusion, fast motion, and clip length. Test: count the instances of a tracked object per frame; the count should be constant.

**Non-conservation.** This is the deepest family and the one that most cleanly separates statistics from simulation. Liquids un-pour — a stream of coffee flows *back up* into the pot. A pile of objects grows or shrinks in count for no reason. A candle burns and gets *longer*. Smoke condenses back into a coherent puff. Each of these violates a conservation law (mass, count, the second law of thermodynamics expressed as irreversibility) that no human ever sees violated, which is exactly why they are so jarring. The cause is the one from Section 1: there is no conserved quantity *in the model*, only a learned tendency. And conservation is a *global* property of a sequence — to enforce "the same liquid volume across all frames" you would need the model to track a running integral, which a feedforward denoiser conditioned on a short window simply does not do. Test: measure total area (or pixel count, or estimated volume) of the substance across frames and check that it is monotone or constant as physics requires.

**Gravity and momentum errors.** Objects fall too slowly, float, or accelerate at the wrong rate. Thrown objects follow arcs that are not parabolas. A bouncing ball gains energy. A car's momentum vanishes when it should plow through. These are *quantitative* errors — the trajectory is the right *shape* qualitatively but the wrong *function* numerically — which makes them the easiest family to measure rigorously. The cause is subtle and worth stating carefully: the model has seen vast amounts of footage of objects moving, and it has learned the *visual texture* of plausible motion, but the training distribution is dominated by **slow, gentle, camera-stabilized motion** (most video is people talking, slow pans, ambient scenes). Fast ballistic motion under gravity is comparatively rare and comparatively hard, so the model's grip on the exact $9.8\,\text{m/s}^2$ acceleration is loose. Test: track the centroid and fit $y(t) = y_0 + v_0 t + \tfrac{1}{2} a t^2$; check whether the fitted $a$ is constant and physically plausible, and how large the residual is. This is the test we will actually implement in Section 5.

The bouncing ball deserves a moment on its own, because it is the cleanest single demonstration of the whole thesis and our running example. A real ball obeys two laws at the bounce. Before contact, energy is conserved as it converts from potential to kinetic, so the speed at the floor is fixed by the drop height. At contact, the coefficient of restitution $e \in [0, 1]$ caps the rebound: the return speed is $e$ times the impact speed, and because $e < 1$ for any real ball, *each bounce is strictly lower than the last* — the bounce heights form a decaying geometric sequence $h_n = e^{2n} h_0$. A model with no energy variable has no reason to respect either law. It frequently produces a *first* bounce that looks plausible, then a second bounce *higher* than the first, or a sequence that fails to decay, because "ball bounces" as a visual pattern does not encode "and loses a fixed fraction of energy each time." This is gravity and momentum failure made precise: the model reproduces the *event* (a bounce) without the *quantity* (the energy budget that governs it), and the giveaway — bounce heights that do not monotonically decay — is something you can measure in three lines of code, which is exactly what Section 5 does.

**Broken contact and collision.** Hands pass through solid surfaces; feet sink into or hover above the ground; a glass on a table either floats a few pixels above it or merges into it; two objects interpenetrate or fail to interact when they touch. The cause is that contact is a *constraint* — a hard inequality (no two solids overlap) that the model has no mechanism to enforce. A simulator handles contact with an explicit constraint solver; the video model handles it with "what does footage of a hand near a table usually look like," which is approximately but not exactly non-penetrating. Contact errors are especially visible because the human visual system is exquisitely tuned to detect penetration and floating. Test: estimate object masks and measure their overlap over time; persistent overlap of two rigid objects is a penetration failure.

**Implausible deformation, hands, and text.** Soft bodies deform in ways no material does — flesh that stretches like taffy, cloth that folds against gravity. And then the two perennial scourges: **hands** (fingers that multiply, merge, or bend backward) and **text** (letters on a sign that morph into gibberish and back, a clock whose digits scramble). Hands and text are special cases of the same problem: they are **high-detail, low-tolerance, rigid-ish structures that are rare relative to their visual complexity**. A hand has a precise anatomical structure (five fingers, specific joint constraints) that the model must reproduce exactly to look right, but hands appear in a thousand configurations and the model has limited capacity to memorize the manifold of valid hand poses. Text is worse: it is *discrete* and *combinatorial* — every word is a precise arrangement of glyphs — and a continuous pixel model is fundamentally mismatched to discrete symbol sequences. The model knows "this region should look texty" without knowing *which letters*. Test: OCR consistency across frames for text; finger-count and pose-validity checks for hands.

A fair caveat runs through this whole catalog: **the frontier is moving, and moving fast.** Many of these failures are markedly less frequent in 2026 frontier models than in 2024 ones. Sora 2 and Veo 3.1 conserve objects more reliably, get gravity closer, and even produce legible short text far more often than their predecessors. The *families* persist — the catalog is structural, rooted in the objective — but the *rates* are dropping with scale and data and architecture work. The honest claim is not "video models cannot do physics." It is "video models do physics by statistical interpolation, which works impressively in-distribution and fails systematically out of it, and the failure families are predictable from the objective." Keep that distinction sharp and you will not be surprised by either the wins or the losses.

## 3. Pattern-matching versus simulation: the evidence on both sides

Now to the central debate, stated fairly, with the evidence laid out for each side rather than caricatured. The two positions are:

- **The simulation hypothesis (strong form):** scaling video generation causes an explicit-enough internal model of physical state and dynamics to *emerge*, such that the model is, functionally, a learned simulator of the world. This is the [Sora world-simulator thesis](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) in its boldest reading.
- **The pattern-matching hypothesis (strong form):** the model interpolates and recombines motion patterns from its training distribution without any forward physical computation, so it is coherent on in-distribution motion and breaks on anything requiring genuine extrapolation of dynamics.

The truth, as usual, is that both strong forms are wrong and the interesting action is in the middle. But let us be disciplined and look at what actually supports each.

![Branching graph showing a learned video model and a simulator both producing frames from the same context, with probing and intervention as the test that distinguishes likely-pixel sampling from explicit-state dynamics](/imgs/blogs/physics-and-the-limits-of-learned-simulation-4.png)

**Evidence for emergent simulation-like behavior.** This evidence is real and should not be dismissed. First, **long-range visual coherence**: frontier models maintain a consistent scene, lighting, and object identity over many seconds, far beyond what a naive frame-by-frame generator could. That coherence requires *something* like a persistent representation of the scene. Second, **approximate 3D consistency**: as the camera orbits an object, the model often produces views consistent with a single rigid 3D shape — occluded surfaces are correctly hidden and revealed, parallax is roughly right. The model was trained only on 2D pixels and never told what 3D is, so a roughly-consistent 3D behavior is a genuinely nontrivial emergent property, and it is the strongest single piece of evidence that *some* structured world representation is being learned. Third, **frequent object permanence**: objects often *do* survive occlusion correctly, which means the latent carries forward more than just the visible pixels. Fourth, and most provocatively, the **world-model line** — [Genie and its successors, action-conditioned interactive video](/blog/machine-learning/video-generation/video-models-as-world-models) — shows that you can condition next-frame prediction on actions and get controllable, navigable environments, which is a strictly stronger and more simulator-like capability than open-loop generation. If a model can respond correctly to "press right, the character moves right, and the parallax updates," it has learned *something* that functions like a transition under an action.

**Evidence against — that it is, at root, interpolation.** This evidence is equally real. First, the **out-of-distribution cliff**: present a model with a physical scenario rare in training — a glass shattering in slow motion, a precise collision, a Rube-Goldberg chain of cause and effect — and coherence collapses. The model is great on the common and bad on the rare, which is the signature of interpolation, not of a general dynamics engine (a real simulator does not care whether a scenario is common; $f$ runs the same either way). Second, the **systematic, structured nature of the failures** from Section 2: a true simulator would never violate conservation, because conservation is in $f$; the fact that violations occur *and cluster around exactly the quantities the objective does not constrain* is strong evidence that no $f$-like object is present. Third, the **probing studies** (more below): attempts to read an explicit, consistent physical state out of the internal activations find a representation that is *partial and inconsistent* — present enough to support coherence, far short of a clean state variable you could integrate forward. Fourth, the **prompt sensitivity and shortcut behavior**: models reproduce the *visual style* of a physical event (the look of a splash) more reliably than its *causal structure* (the right amount of water moving the right way), which is what you expect if the model matched appearance rather than computing dynamics.

So where does that leave us? At a position I will state as precisely as I can and defend for the rest of the post. **The model learns an interpolative, appearance-grounded approximation of dynamics over the manifold of motions its training distribution covered densely.** Inside that manifold — common motions, common scenes, short horizons — the approximation is good enough to *look* like simulation and even to support some genuinely structured behavior like rough 3D consistency. Outside it — rare physics, long horizons, precise quantitative conservation — it degrades exactly as an interpolator degrades, because there is no $f$ to fall back on. It is not a parrot and it is not a physics engine. It is a very high-dimensional, learned interpolator of plausible pixel motion that has absorbed real statistical regularities of the visual world, some of which *are* physics, encoded implicitly and leakily. That is a more interesting and more useful thing to be than either caricature.

#### Worked example: the in-distribution versus out-of-distribution coherence gap

Make it concrete. Take a fixed frontier I2V model and feed it two first frames. Frame A: a person sitting at a desk in an office, prompt "the person picks up a coffee cup and drinks." This is maximally in-distribution — millions of clips like it exist. The model produces a clip that is coherent for the full duration, the cup persists, the hand grips plausibly, identity holds; on a physics rubric it might score, say, 8 of 10 (small contact imperfections, the rest fine). Frame B: the *same* person, prompt "the person juggles three glass balls that collide and shatter." This is out-of-distribution — juggling-with-shattering clips are rare and physically demanding. The model produces a clip where balls merge and split (permanence fails), shards appear before contact (causality fails), and the count of balls drifts from three to two to four (conservation fails); on the same rubric, maybe 2 of 10. *Same model, same person, same number of sampling steps.* The only thing that changed is how densely the training distribution covered the requested physics. That delta — call it roughly a 6-point swing on a 10-point physics rubric driven purely by in- versus out-of-distribution — is the empirical fingerprint of interpolation, and you can reproduce it on any model you have access to. (Numbers illustrative; the *direction and size* of the effect is robust and reproducible, the exact points depend on your rubric.)

## 4. The science: why next-frame likelihood does not enforce physical law

This is the rigorous core. We have asserted that the objective does not constrain conservation; now let us prove it, carefully enough that the claim is unarguable, and then look at what the probing studies actually measure.

![Stacked layers showing the training pipeline from plausible-only video through a 3D-VAE and a denoiser whose loss is pure noise reconstruction with no mass or energy term, leaving physics implicit and often wrong](/imgs/blogs/physics-and-the-limits-of-learned-simulation-3.png)

Start from the objective. A diffusion video model is trained to minimize, over the data distribution $p_{\text{data}}$ of real clips,

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_{1:T} \sim p_{\text{data}},\, t,\, \epsilon}\Big[\big\lVert \epsilon - \epsilon_\theta(x^{(t)}_{1:T}, t, c)\big\rVert^2\Big],
$$

where $x^{(t)}$ is the clip noised to diffusion time $t$ and $\epsilon$ is the added Gaussian noise. Minimizing this is, up to the standard reweighting, equivalent to learning the score $\nabla_x \log p_{\text{data}}(x_{1:T}\mid c)$ and hence to fitting $p_\theta \approx p_{\text{data}}$. (Again, the equivalence is image-diffusion machinery — see the [score-SDE post](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) — and we import it rather than re-derive it.) The object being optimized is a **density over pixel sequences**. Write it out: the loss is a sum of per-pixel, per-frame reconstruction errors. There is no functional anywhere in $\mathcal{L}$ that reads "compute the total mass in frame $t$ and the total mass in frame $t+1$ and penalize their difference."

Now the formal argument. Consider two clips, $x^{\text{phys}}$ and $x^{\text{viol}}$, that are *pixel-wise identical for the first $T-1$ frames* and differ only in frame $T$: in $x^{\text{phys}}$ the liquid volume is conserved, in $x^{\text{viol}}$ a small amount of liquid has vanished. Suppose — as is empirically true of real video corpora — that the training distribution contains clips visually close to *both*: most are conservation-respecting, but a nonzero fraction (compression artifacts, edits, cuts, genuinely ambiguous footage, occlusion making volume hard to read) are close to the violating one. The model's optimal $p_\theta(x_T \mid x_{<T}, c)$ is then a *smooth distribution* that places most mass near $x^{\text{phys}}$ but **nonzero mass near $x^{\text{viol}}$**, because the loss is minimized by matching the data density, and the data density is not a delta on the conservation-respecting frame. The model is *correct, in the likelihood sense*, to sometimes sample the violating frame. The violation is not a bug in the fit; it is a faithful reflection of a training distribution that does not perfectly enforce the law, smoothed by a model that has no hard constraint to override the smoothing.

Contrast a simulator. The transition $s_{t+1} = f(s_t, a_t)$ with $f$ derived from physics assigns probability **zero** to the violating transition — not low probability, *zero*, because mass conservation is an algebraic identity in $f$, not a learned tendency. This is the formal heart of the matter: **a likelihood objective can at best concentrate probability near physically valid sequences; it can never make the probability of a violation exactly zero, because (a) the data itself is not a perfect oracle of physics, and (b) a smooth model spreads mass.** Conservation is an equality constraint, and you do not enforce an equality constraint by maximizing a smooth likelihood. You enforce it by building it into the function class. The video model's function class — a denoiser over pixel sequences — has no slot for it.

There is a deeper, information-theoretic way to see the same thing. To conserve a global quantity (total volume, object count) across a long clip, the model must maintain a *running accumulator* — a piece of state that integrates the quantity and constrains every future frame to match it. A feedforward denoiser conditioned on a finite window of recent frames has bounded memory; it cannot maintain an exact running integral over an arbitrarily long sequence. So even setting aside the smooth-likelihood argument, the *architecture* cannot represent long-horizon exact conservation. It can approximate it over short windows where "the volume looks about the same as a moment ago" suffices, which is precisely why short clips conserve better than long ones and why the failure rate climbs with length. This connects directly to the [error-accumulation problem in autoregressive rollout](/blog/machine-learning/video-generation/long-video-and-autoregressive-rollout): the same lack of a corrective state variable that lets identity drift over 30 seconds lets conserved quantities drift too.

**What probing and intervention studies measure.** Given all this, the empirical question becomes: *how much* of a physical state has the model nonetheless learned implicitly? Two methodologies attack this.

A **linear probe** trains a small linear (or shallow) classifier to predict a physical variable — an object's position, its velocity, whether a collision is about to occur, the depth of a pixel — from the model's internal activations at some layer. If a clean linear readout exists, the information is present and linearly accessible, which is suggestive (though not proof) of a structured representation. The published results on video and world models are *mixed and partial*: position and depth probe out reasonably well (consistent with the rough-3D-consistency evidence), but precise dynamical quantities like velocity and especially *conserved* quantities probe out poorly and inconsistently across layers and scenes. The representation is there for appearance-grounded quantities and thin for the quantities that the objective never constrained — exactly what the theory predicts.

An **intervention (or counterfactual) test** is stronger and more in the spirit of causality. You perturb the input in a physically meaningful way — change an object's initial velocity, remove a support so something should fall, apply a force — and check whether the model's output changes in the physically correct direction. A genuine $f$ responds correctly to *any* intervention by construction. An interpolator responds correctly to interventions *that resemble variations seen in training* and incorrectly to novel ones. The findings here line up with the OOD cliff: in-distribution interventions ("the ball is thrown harder") often produce roughly-right responses; out-of-distribution interventions ("remove the table the cup rests on") frequently produce physically wrong responses (the cup floats, or the model ignores the change). Intervention tests are the closest the field has to directly asking "is there an $f$ in there?", and the answer they return is "a partial, distribution-bounded one."

The figure above frames this as the distinction it is: both the learned model and a simulator emit frames from the same context, but only the simulator carries an explicit state forward under conserved law — and probing plus intervention is precisely the apparatus that tells you which path produced a given clip.

#### Worked example: why a per-frame loss cannot see a lost liter

Put numbers on the smooth-likelihood argument. Suppose a clip is 48 frames and at frame 24 a poured liquid loses 5% of its volume — a clear conservation violation a human spots instantly. What does the training loss "see"? The denoising loss at frame 24 is a pixel MSE against the *training frame*, which (if that training clip itself happened to show the artifact, or if a nearby training clip did) is perfectly happy with the violating pixels. Even if every training clip conserved volume, the model's *generalization* — its smooth interpolation between training points — will assign nonzero density to the 5%-lost frame, because the manifold of "plausible pour frames" is continuous and the conserving frame is not isolated on it. The gradient contribution from "you violated conservation" is exactly **zero**, because no term computes volume. Compare: a physics-informed loss with an explicit penalty $\lambda\,(V_{t+1} - V_t)^2$ on estimated volume *would* push back — but no production video model trains with such a term, because estimating $V$ from pixels is itself hard and the whole appeal of the generative approach is that you do not specify physics. The lost liter is invisible to the objective by construction, and that is why you see it in the output.

## 5. A harness that measures physics instead of eyeballing it

Talk is cheap and "looks wrong" is not a metric. The practical contribution of this post is a small, runnable evaluation harness that turns physics failures into numbers. The philosophy: pick a scenario with a *known* physical property (a ball under gravity has a parabolic centroid trajectory; a single ball has count one in every frame), generate clips, extract the property frame-to-frame, and quantify the deviation. This is exactly how you should test any claim that a model "understands physics" — not by curating a good clip, but by measuring a conserved or law-governed quantity over a sample of clips.

![Before-and-after panels contrasting an eyeballed verdict that a clip looks fine against a measured verdict that tracks the ball centroid versus a gravity parabola and counts objects per frame](/imgs/blogs/physics-and-the-limits-of-learned-simulation-6.png)

The harness has three parts: (1) generate clips of the scenario with a real `diffusers` video pipeline, (2) extract a per-frame measurement (centroid, object count), (3) fit the physical law and report residuals and violation rates. Let us build each.

First, generation. We use an image-to-video pipeline so we can fix the *initial* condition (a ball at the top of the frame) and let the model roll the dynamics forward — this isolates the model's physics from its scene-composition ability, and gives us a known $y_0$ to fit against. Any I2V pipeline works; here is CogVideoX I2V, which is open and runs on a single 24 GB card with offload.

```python
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()        # fit in ~12-16 GB peak
pipe.vae.enable_tiling()               # VAE decode is the memory wall

first_frame = load_image("ball_top.png")   # a red ball near the top, plain floor
prompt = ("a red rubber ball falls straight down under gravity, "
          "hits the floor and bounces, fixed camera, plain background")

def generate_clip(seed: int):
    out = pipe(
        image=first_frame,
        prompt=prompt,
        num_frames=49,                 # CogVideoX native clip length
        guidance_scale=6.0,
        num_inference_steps=50,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    return out.frames[0]               # list of PIL frames

clips = [generate_clip(s) for s in range(8)]   # 8 seeds for a sample, not n=1
for i, frames in enumerate(clips):
    export_to_video(frames, f"ball_{i}.mp4", fps=8)
```

The `num_frames=49`, `guidance_scale`, `enable_model_cpu_offload`, and `vae.enable_tiling` are real, load-bearing flags — tiling matters because, as this series keeps stressing, the [VAE decode, not the denoiser, is often the VRAM wall](/blog/machine-learning/video-generation/video-autoencoders-and-spatiotemporal-compression). Note the eight seeds: **never evaluate physics on $n=1$.** A single lucky clip tells you nothing; you want the distribution of behavior, which means a sample and a violation *rate*, not an anecdote.

Second, the measurement. We track the ball's centroid per frame. For a controlled red-ball-on-plain-background scenario, a color threshold plus a connected-components pass is robust and dependency-light; in a messier scene you would swap in an off-the-shelf tracker or a segmentation model, but the *interface* — frame in, centroid and count out — is the same.

```python
import numpy as np

def measure_frame(frame_pil, lo=(120, 0, 0), hi=(255, 90, 90)):
    """Return (centroid_xy or None, object_count) for a red ball on plain bg."""
    img = np.asarray(frame_pil).astype(np.int32)        # H x W x 3
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = ((r >= lo[0]) & (g <= hi[1]) & (b <= hi[2]))  # red, low green/blue
    if mask.sum() < 30:                                  # ball not visible
        return None, 0
    # connected components via a simple flood-free label (scipy if available)
    from scipy import ndimage
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    big = [i + 1 for i, s in enumerate(sizes) if s >= 30]  # ignore specks
    count = len(big)
    # centroid of the largest component (the ball)
    main = big[int(np.argmax([sizes[i - 1] for i in big]))]
    ys, xs = np.where(labels == main)
    return (float(xs.mean()), float(ys.mean())), count

def measure_clip(frames):
    centroids, counts = [], []
    for f in frames:
        c, n = measure_frame(f)
        centroids.append(c)
        counts.append(n)
    return centroids, counts
```

Third, the scoring. Two numbers fall out. (a) **A gravity residual**: during the *fall* phase (before the first bounce), the vertical centroid should obey $y(t) = y_0 + \tfrac{1}{2} a t^2$ with a single constant $a > 0$ (image $y$ increases downward, so falling is increasing $y$). Fit a parabola to the fall segment, report the residual and check $a$ is roughly constant and positive. (b) **A count-stability / permanence rate**: the object count should be exactly one in every frame the ball is present; any frame with count $\neq 1$ (or a `None` mid-flight, meaning the ball vanished) is a permanence violation. We aggregate over the eight clips.

```python
def gravity_residual(centroids):
    """Fit y = y0 + 0.5*a*t^2 to the fall segment; return (a, rms_residual)."""
    ys = [(i, c[1]) for i, c in enumerate(centroids) if c is not None]
    if len(ys) < 5:
        return None, None
    # fall segment = up to the frame of maximum y (lowest point) before bounce
    t_bottom = max(ys, key=lambda p: p[1])[0]
    seg = [(i, y) for i, y in ys if i <= t_bottom]
    if len(seg) < 4:
        return None, None
    t = np.array([i for i, _ in seg], dtype=float)
    y = np.array([v for _, v in seg], dtype=float)
    # design matrix for y = c0 + c2 * t^2  (no linear term if dropped from rest)
    A = np.stack([np.ones_like(t), t, t ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = 2.0 * coef[2]                       # acceleration in px / frame^2
    resid = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return a, resid

def permanence_violations(counts):
    """Fraction of frames where the ball count is not exactly one."""
    present = [c for c in counts if c is not None]
    if not present:
        return 1.0
    bad = sum(1 for c in present if c != 1)
    return bad / len(present)

# aggregate over the sample
accs, resids, perm_rates = [], [], []
for frames in clips:
    centroids, counts = measure_clip(frames)
    a, r = gravity_residual(centroids)
    if a is not None:
        accs.append(a); resids.append(r)
    perm_rates.append(permanence_violations(counts))

print(f"mean fitted accel (px/frame^2): {np.mean(accs):.2f}")
print(f"accel coeff. of variation:      {np.std(accs)/max(np.mean(accs),1e-6):.2f}")
print(f"mean parabola RMS residual (px): {np.mean(resids):.2f}")
print(f"permanence violation rate:       {np.mean(perm_rates):.3f}")
```

What do you learn from running this? Three things, in roughly increasing severity. The **acceleration's coefficient of variation across seeds** tells you whether the model applies a *consistent* gravity — a real simulator gives the same $a$ every time; a high CV means the model's "gravity" is seed-dependent, i.e. not a law but a sample. The **parabola RMS residual** tells you whether the fall is even the right *shape* — a large residual means the trajectory is not a clean parabola at all (it stutters, floats, or changes acceleration mid-fall). The **permanence violation rate** tells you how often the ball vanishes or duplicates. On 2024-era models these numbers are sobering; on 2026 frontier models the gravity residual and CV tighten considerably while permanence on a single clean object gets quite good — which is the measured, quantitative version of "the frontier is improving but the families persist." The point is not the specific numbers your run produces; it is that **you now have numbers at all**, and a violation *rate* over a sample, instead of an argument about a cherry-picked clip.

A measurement caveat, stated honestly because the harness is only as good as its tracker: color-threshold tracking fails when the model changes the ball's color, adds motion blur that desaturates it, or occludes it — and those failures will show up as `None` frames that you must not naively count as permanence violations (a ball *behind* an object is correctly absent). For anything beyond a controlled red-ball scene, use a real segmentation or point-tracking model and validate the tracker on real footage first, so a tracker error is not misread as a physics error. The discipline is the same discipline as [measuring video quality honestly](/blog/machine-learning/video-generation/the-metrics-of-video-generation): the metric has failure modes, and you have to characterize them before you trust the number.

## 6. The benchmarks that test physics, and what they actually reveal

The harness above is a do-it-yourself probe for one scenario. The field has built standardized benchmarks that test physical plausibility at scale, and they are worth understanding both for what they measure and for the gap between their scores and the marketing.

![Matrix of physics benchmark scores comparing the real-world ceiling against the best 2026 model and the resulting gap across Physics-IQ, VideoPhy physical-commonsense, semantic adherence, and visual quality](/imgs/blogs/physics-and-the-limits-of-learned-simulation-5.png)

**Physics-IQ.** This benchmark (introduced in 2025) takes the cleanest possible approach: it uses *real* physical scenarios filmed with a real camera, gives the model the first few seconds as conditioning, asks it to predict the continuation, and then *compares the prediction to the actual filmed future* on physically meaningful metrics — where things moved, whether the right regions changed, spatiotemporal agreement with ground truth. Because there is a real future to compare against, Physics-IQ sidesteps the "looks plausible" trap entirely: a clip can look great and still score low if it did not predict what *actually happened*. The headline finding is stark and worth internalizing: **frontier video models score far below the real-world ceiling, often in the low tens out of a normalized 100, where real footage defines 100.** The benchmark's authors are explicit that high visual realism does *not* track high physical understanding — some of the most visually impressive models do not top the physics ranking. That dissociation — realism and physics scored almost independently — is the single most important empirical fact in this whole area, and it is exactly what Section 4's theory predicts: the objective optimizes the thing that drives realism (pixel likelihood) and is silent on the thing that drives physics (conservation and dynamics).

**VideoPhy.** This benchmark (2024, with a VideoPhy-2 follow-up) attacks physical *commonsense* in generated video via prompts describing material interactions (a solid hitting a liquid, an object on an incline, pouring, stacking). It scores two axes with trained auto-evaluators (validated against human labels): **semantic adherence** (did the video depict what the prompt asked?) and **physical commonsense** (did it obey intuitive physics — gravity, solidity, conservation?). The finding mirrors Physics-IQ: across many models, a large fraction of generated clips fail physical commonsense even when semantic adherence is decent, and the *joint* success rate — both right at once — is low. Models can show you the requested scene and still get the physics of it wrong, and the proportion that nails both is the number that matters and the number that stays stubbornly modest. VideoPhy-2 hardened the prompts and the gap widened, which is the usual story when a benchmark stops being easy.

**Other and related probes.** A growing family rounds out the picture: benchmarks that test specific laws (projectile motion, collisions, fluid behavior), counterfactual and intervention suites that check whether changing an initial condition changes the outcome correctly, and the [VBench / VBench-2.0 line](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation) which folds some physics-adjacent dimensions (motion smoothness, dynamic degree) into a broader quality score. A subtlety worth flagging: VBench's *motion smoothness* and *dynamic degree* are **not** physics scores and can be actively misleading as proxies — a model can max smoothness by barely moving (a near-static clip is trivially smooth and trivially conserves everything), so smoothness rewards the wrong thing if you read it as physical correctness. This is the same **dynamic-vs-stable gaming problem** the [metrics post](/blog/machine-learning/video-generation/the-metrics-of-video-generation) dissects: optimizing for one quality axis can silently sabotage another, and "smooth" and "physically correct" are not the same axis. The physics-specific benchmarks exist precisely because the general quality metrics do not capture physics, and you should not let a high VBench score stand in for a physics claim.

One methodological distinction is worth pulling out because it separates a *good* physics benchmark from a misleading one. There are two ways to score physical correctness, and they are not equally trustworthy. The first is **plausibility judging**: show a clip to a human or an auto-evaluator and ask "does this look physically plausible?" This is what VideoPhy's physical-commonsense axis does, and it is useful but soft — a clip can be subtly, systematically wrong in a way a quick judgment misses, and the judge's own biases leak in. The second, stronger method is **comparison to a real future**: film the actual physical event, give the model the lead-in, and score how close its prediction is to *what really happened*. This is Physics-IQ's design, and it is far harder to game, because there is a ground-truth continuation to be wrong about — no amount of plausible-looking smoke saves you if the real smoke went the other way. The gap between these two methodologies matters when you read a leaderboard: a high score on a plausibility benchmark is weaker evidence than a high score on a comparison-to-reality benchmark, and the most honest physics claims come from the latter. When someone tells you a model "understands physics," ask which kind of test produced the number; if it was plausibility judging alone, discount it accordingly.

A second subtlety: physics scores are *scenario-dependent*, and aggregate numbers hide enormous variance. A model can score well on rigid-body motion (common, geometrically simple) and catastrophically on fluids (rare, high-dimensional, chaotic), and a single averaged "physics score" blends these into a number that misrepresents both. When you evaluate for a specific use, you care about the physics of *your* scenario, not the average — a model that is great at falling rigid objects may be useless for pouring liquids, and the aggregate score will not tell you which. This is the same lesson as the harness in Section 5: physics is not one capability but a family of them, tested per scenario, and the responsible move is to measure the specific law your application depends on rather than trust a headline average.

What the benchmarks collectively reveal, then, is a consistent and theory-aligned picture: **the best 2024–2026 models are visually excellent and physically mediocre, the two are scored nearly independently, and the physics gap is large and only slowly closing.** That is not a knock on the models — they were not trained to be physics engines — but it is a hard empirical boundary on the world-simulator claim. A system that scores in the low tens on Physics-IQ against a real-footage ceiling of 100 is not, in any rigorous sense, a simulator of the physical world. It is a generator of physically *plausible-looking* video that is right often enough to be useful and wrong often enough to be untrustworthy as an oracle.

#### Worked example: reading a benchmark table honestly

Suppose you see a table reporting Model X at "Physics-IQ 24" and "VBench 84" and a press release leading with the 84. How should you read it? The VBench 84 is a *general quality* score — it says the clips look good, are smooth, and are reasonably consistent. The Physics-IQ 24 is the physics score against a real-footage ceiling of 100 — it says the model predicts the actual physical future correctly about a quarter as well as reality does. *Both numbers are about the same model.* The honest reading is "beautiful clips, weak physics," and the 60-point gap between the two scores *is the entire content of this post* expressed as a number: realism and physics are different axes, the model is strong on one and weak on the other, and a headline that quotes only the high one is hiding the relevant fact. When you evaluate a model for a physics-dependent use, the Physics-IQ-style number is the one that binds; the VBench number tells you it will look nice while being wrong. (Specific values illustrative and approximate — check the current leaderboard for live numbers — but the *shape* of the gap is the robust, reproducible finding.)

## 7. What real physical understanding would require

If a likelihood-only pixel model cannot, in principle, enforce conservation, what *would*? This is the constructive half of the post, and the answers fall into four ingredients, each of which adds something the pure objective lacks. The figure stacks them as a progression from a bare pixel model toward genuine physical grounding.

![Stacked progression from a pure pixel video model adding persistent 3D structure, explicit object state, action-conditioning, and a solver in the loop to reach conserved laws and causal dynamics](/imgs/blogs/physics-and-the-limits-of-learned-simulation-7.png)

**Ingredient 1: explicit state.** The model needs a variable that *is* the physical state — object positions, velocities, masses — carried forward and constrained, rather than implicit in a pixel latent. Approaches here range from object-centric representations (slot attention, factored latents that bind to entities) to outright structured-latent world models that maintain a per-object state. The point is to give the model an $s_t$ to integrate, so that conservation can be expressed as a constraint *on $s_t$* rather than hoped for in pixels. The cost is that object-centric models are harder to train and scale than monolithic pixel denoisers, which is precisely why the frontier did not go this way — monolithic scaled better.

**Ingredient 2: 3D (and 4D) structure.** Physics happens in 3D over time (4D), not in the 2D pixel plane, and a model reasoning purely in 2D is fighting the geometry. Giving the model a persistent 3D scene representation — explicit geometry, depth, or a neural radiance / Gaussian-splat-style structure that it renders to pixels — means occlusion, parallax, and rigid motion come out *consistent by construction* rather than by interpolation. This is the bridge from video to 3D and 4D generation, the subject of the [camera-control and 4D post](/blog/machine-learning/video-generation/camera-control-and-4d-generation), and it is the most promising near-term lever because rough 3D consistency is *already* the model's strongest emergent capability — making it explicit hardens it.

**Ingredient 3: action-conditioning and intervention.** A simulator is defined by its response to actions: $s_{t+1} = f(s_t, a_t)$ has an $a_t$ in it. A model that only does open-loop $p(\text{frames}\mid \text{prompt})$ cannot be *intervened on* the way a simulator can. Action-conditioned next-frame prediction — the [world-model line: Genie, GameNGen, playable video](/blog/machine-learning/video-generation/video-models-as-world-models) — adds exactly this $a_t$, and it is a genuine step toward simulation because it forces the model to learn a *transition under control*, not just a distribution over rollouts. Training on action-outcome pairs (including counterfactual ones — *this* action leads to *that* state) is how you would pressure a model to learn $f$ rather than $p$, and it connects video generation directly to agents, robotics, and reinforcement learning, where a learned $f$ that you can plan against is the whole prize.

**Ingredient 4: a solver in the loop (neuro-symbolic / hybrid).** The most direct route to *guaranteed* physical correctness is to not learn the physics at all where you do not have to: run an actual physics engine (a rigid-body or fluid solver) for the parts that need to be exact, and use the neural model for the parts it is good at — appearance, texture, style, the long tail of visual detail the solver cannot author. The solver provides $f$ with hard conservation; the neural net provides $g$, the rich renderer. This **neuro-symbolic / simulator-in-the-loop** design trades the elegance of an end-to-end learned system for correctness where it counts, and it is the pragmatic answer whenever a downstream decision actually depends on the physics being right (robotics, engineering simulation, safety-critical synthetic data). The cost is engineering complexity and the loss of the "just scale it" simplicity — but if you *need* conservation, you cannot get it from likelihood, and a solver is the honest way to get it.

The unifying theme across all four: each ingredient adds back something the pixel-likelihood objective structurally omits — a state to conserve, a geometry to be consistent in, an action to respond to, or a law to enforce exactly. None of them is free; all of them complicate the clean, scalable, monolithic recipe that made the frontier models what they are. Which is the crux of the engineering trade-off, and the reason the frontier has *mostly* bet on scale plus data rather than on explicit physical structure: scale has been buying steady improvement on the benchmarks, and structure is hard, so the field is wagering that a good-enough interpolation of physics, reached by scale, beats an explicit physics module that is hard to build and hard to scale — for *most* uses. The wager is reasonable for creative video and clearly wrong for anything that needs an oracle, which is the takeaway the builder section makes operational.

## 8. Case studies and real numbers

Let me anchor the argument in named, real results rather than generalities. Four short case studies, each with the caveat that exact figures move fast and several frontier models are closed, so where I give a number I flag whether it is reported or estimated.

**Sora and the conservation failures in its own demo reel.** The most instructive case is OpenAI's own. The [Sora technical report](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) advertised emergent 3D consistency and object permanence as evidence for the world-simulator thesis — and those claims hold up partially, as we discussed. But the *same report and surrounding demos* contained the canonical failures: a glass tipping over with liquid appearing before the break, a person taking a bite of a cookie that leaves no bite mark, a treadmill running a man the wrong way. These were not adversarial finds; they were in the official material. The lesson is not that Sora is bad — it was a generational leap — but that the *strongest* video model of its moment, presented explicitly as a step toward world simulation, visibly violated conservation and causality in its own best clips. That is the world-simulator thesis meeting its own evidence, and it lands exactly where Section 3 said it would: real emergent coherence, real systematic physics failure, side by side.

**Physics-IQ's realism-physics dissociation.** The Physics-IQ paper's central measured result deserves restating as a case study because it is the cleanest evidence in the field. Across a set of leading 2024–2025 video models, the benchmark found physical-understanding scores far below the real-footage ceiling — and, critically, found that the ranking by physical understanding was *different* from the ranking by visual quality. A model people perceived as among the most realistic was not the best at physics. This is the dissociation made quantitative: realism and physics are separable, measured axes, and excelling at one does not imply the other. If you take one number from this whole post, take this one, because it converts "video models are bad at physics" from an opinion into a measured, replicated finding with a methodology that compares to real footage rather than to human taste.

**VideoPhy's low joint success rate.** VideoPhy measured, across many open and closed models, the fraction of generated clips that satisfy *both* semantic adherence and physical commonsense, and found it low — a large share of clips that correctly depict the requested scene still violate intuitive physics. VideoPhy-2 hardened the test and the gap widened. The case-study value is methodological: by scoring physics *separately* from "did it follow the prompt," VideoPhy isolates physics as its own failure mode and shows it is not just a prompt-following problem. The model often understands *what* to show and gets *how it should physically behave* wrong, which is precisely the appearance-versus-dynamics split from Section 3.

**The world-model line as the partial counter-evidence.** Fairness demands the other side. Genie and its successors — [action-conditioned, playable, interactive generated worlds](/blog/machine-learning/video-generation/video-models-as-world-models) — demonstrate that conditioning next-frame prediction on actions yields environments you can *navigate*, with parallax and object interactions that respond to control. This is meaningfully more simulator-like than open-loop generation: it learns a transition *under actions*, the $a_t$ that open-loop $p(\text{frames})$ lacks. It is not a physics engine — these worlds still drift, hallucinate, and violate conservation over long horizons, and they are trained in narrow domains (game-like environments) rather than the open physical world — but they are the strongest existing evidence that the *form* $s_{t+1} = f(s_t, a_t)$ can be partially learned, and they point at where genuine learned simulation, if it comes, will come from: action-conditioning and interaction, not bigger open-loop models.

Here is a compact comparison of the failure families against their causes and tests, the kind of table you can hand to a teammate who asks "so what actually breaks and how would we check?"

| Failure family | What you observe | Root cause | How to measure | Frontier trend (approx) |
| --- | --- | --- | --- | --- |
| Object permanence | Objects vanish, duplicate, swap identity through occlusion | No persistent object state; finite latent memory | Count tracked object per frame; flag count ≠ 1 | Improving; good on single clean objects, weaker under occlusion |
| Non-conservation | Liquids un-pour, counts drift, irreversible processes reverse | No conserved quantity in objective; no running integral | Total area / count / est. volume over time | Slowly improving; still systematic on long clips |
| Gravity / momentum | Wrong acceleration, non-parabolic arcs, energy gain on bounce | Trained mostly on slow gentle motion; no force law | Fit centroid to ½at²; check a constant, residual small | Improving; CV of fitted a tightening with scale |
| Contact / collision | Penetration, floating, no interaction on touch | Contact is a constraint with no enforcement mechanism | Mask overlap of rigid bodies over time | Persistent; hard family |
| Deformation / hands / text | Taffy flesh, morphing fingers, scrambling letters | Rare high-detail rigid structures; pixel↔discrete mismatch (text) | OCR consistency; finger-count / pose validity | Hands and text improving fast; deformation slower |

And a second table contrasting the two paradigms head-on, the formal distinction from Section 1 turned into an operational checklist.

| Property | Learned video model — $p(\text{frames})$ | Simulator — $s_{t+1}=f(s_t,a_t)$ |
| --- | --- | --- |
| Internal object | Pixel latent, no explicit state | Explicit state (position, velocity, mass) |
| Conservation | Approximate, learned tendency | Exact, algebraic invariant in $f$ |
| Out-of-distribution input | Degrades like an interpolator | Runs unchanged; $f$ is input-agnostic |
| Response to intervention | Right if intervention resembles training | Always correct by construction |
| Probability of a violation | Nonzero (smooth likelihood) | Exactly zero |
| Strength | Rich, realistic appearance; cheap to scale | Guaranteed physical correctness |
| Where it shines | Creative video, in-distribution motion | Engineering, robotics, anything needing an oracle |

## 9. When to trust generated video as physics — and when never to

This is the decisive section for builders, because the entire skeptical analysis above resolves into one practical rule: **the acceptable level of physical wrongness is set by your downstream use, not by the model.** The figure is a decision tree from "does anything actually depend on the physics?" to a concrete recommendation.

![Decision tree on whether to trust generated video for physics, branching on whether the clip only needs to look right or a decision depends on the physics, down to ship-it versus verify-against-a-simulator outcomes](/imgs/blogs/physics-and-the-limits-of-learned-simulation-8.png)

**When generated physics is good enough (use it freely).** If the video only has to *look* right to a human for a few seconds, and no decision depends on the physics being correct, ship it. This covers an enormous and valuable space: B-roll, advertising, concept visualization, mood pieces, social content, game cinematics, design exploration, storyboarding. A coffee cup that floats one pixel above the table for two frames does not matter in a 4-second ad; a slightly-too-bouncy ball does not matter in a music video. The frontier models are *genuinely excellent* here, and the failures are mostly invisible at the duration and scrutiny these uses involve. For short, in-distribution, look-right content, generated video is not just good enough — it is the best tool there is, and the physics caveats are academic.

**When you must verify (decision depends on physics).** The moment a downstream decision depends on a physical quantity being *correct*, generated video stops being an oracle and you must verify against ground truth or a real solver. Examples: synthetic training data for a robot manipulator (if the generated grasp physics is wrong, the robot learns the wrong policy); engineering or scientific visualization presented as accurate; safety analysis; anything where a viewer will *reason from* the physics rather than just enjoy it. Here the rule is hard: **do not trust the generated physics; validate it.** Run the harness from Section 5 on your scenario and measure the violation rate; better, run a real physics engine for the quantities that must be exact and use the generator only for appearance (the neuro-symbolic route from Section 7). If you cannot verify, do not ship the physics claim.

**The bright line: never use generated video as a physics oracle alone.** This is the one non-negotiable. A physics oracle is a system you *query for the right answer to a physical question* — "where will this ball land?", "will this structure hold?", "how much water spills?". Generated video cannot fill that role, by the argument of Section 4: it has no $f$, it has a smooth $p$ that assigns nonzero probability to violations, and it degrades out of distribution exactly where novel physics questions live. Using it as an oracle means trusting a measured-low-tens-out-of-100 Physics-IQ system to give you ground-truth physics, which is a category error. If you need an oracle, use a simulator. Use the video model for the picture.

#### Worked example: choosing the tool for a robotics data pipeline

A team wants synthetic video of a robot arm pouring water into a cup, to augment training data for a pouring policy. Tempting to generate it with a frontier I2V model — it would look great. Walk the decision tree. Does a decision depend on the physics? *Yes* — the policy will learn the relationship between arm motion and water behavior, so if the generated water conserves volume wrong or spills unphysically, the policy learns a false dynamics. So: do **not** use generated video as the oracle. The right design is neuro-symbolic: run a real fluid/rigid solver to get the *correct* pour physics and the arm-water dynamics, and optionally use a video model only to add photorealistic appearance on top of the solver's correct motion (a learned renderer over a simulated state). Cost: more engineering than "just generate it," but the alternative is a policy trained on physically wrong data, which is worse than no data. Contrast the *advertising* version of the same scene — a beauty shot of water pouring into a cup for a commercial — where you generate it directly, in seconds, and the slightly-wrong splash is invisible and irrelevant. Same scene, opposite decision, because the downstream dependency on physics is opposite. That is the whole rule in one example.

## 10. Key takeaways

- **Two different objects.** A video model computes $p(\text{frames})$ — a learned distribution over plausible pixel sequences — not $s_{t+1} = f(s_t, a_t)$, an explicit state advanced by conserved law. Almost every confusion about "understanding physics" comes from blurring these. Keep the type signatures separate.
- **The objective is the cause.** Next-frame likelihood (the denoising loss) has *no term* that measures conservation, so conserved quantities are unconstrained by construction. A smooth likelihood can concentrate probability near valid physics but can never make a violation's probability exactly zero — only a hard constraint in the function class can, and the pixel denoiser has no such slot.
- **The failures are systematic, not random.** Permanence, conservation, gravity, contact, and deformation each have a distinct cause and a distinct test, and they cluster precisely around the quantities the objective ignores. The catalog is structural; the *rates* fall with scale but the *families* persist.
- **It is interpolation, not simulation — and not a parrot either.** The model learns an appearance-grounded, interpolative approximation of dynamics over the densely-covered region of its training distribution. It looks like simulation in-distribution (including genuinely emergent rough 3D consistency) and fails like an interpolator out of distribution. That middle position is the accurate one.
- **Measure, do not eyeball.** Track a centroid against $\tfrac{1}{2} g t^2$, count objects per frame, and report a violation *rate* over a sample of seeds — never $n=1$. The harness in Section 5 turns "looks wrong" into a number, with honest caveats about the tracker's own failure modes.
- **The benchmarks agree, quantitatively.** Physics-IQ scores frontier models far below a real-footage ceiling and shows realism and physics are *separable* axes; VideoPhy shows a low joint rate of semantic-plus-physical success. A high VBench score is not a physics score, and smoothness can be gamed by barely moving.
- **Real understanding needs added structure.** Explicit state, 3D/4D geometry, action-conditioning, or a solver-in-the-loop — each adds back something likelihood omits. The frontier mostly bet on scale instead, which is right for creative use and wrong for an oracle.
- **The use sets the tolerance.** If it only has to look right for a few seconds, ship it. If a decision depends on the physics, verify against a real solver. Never use generated video as a physics oracle alone — that is a category error.

The honest one-line verdict: today's video models are extraordinary generators of physically *plausible-looking* footage and unreliable computers of physical *fact*, and the gap between those two is measured, structural, and exactly where you should expect it from how they are trained. Treat them as the brilliant cinematographers they are, not the physics engines they are marketed as, and you will use them well. The deeper trajectory — whether action-conditioning and world-model training narrow this gap into genuine learned simulation — is the open question the [world-models post](/blog/machine-learning/video-generation/video-models-as-world-models) takes up, and the practical synthesis of all of it lands in the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).

## Further reading

- Brooks, Peebles, et al., *Video Generation Models as World Simulators* (OpenAI Sora technical report, 2024) — the source of the thesis this post scrutinizes; read it alongside its own demo failures.
- Motamed, Culp, et al., *Do generative video models understand physical principles?* / **Physics-IQ** (2025) — the real-footage benchmark that quantifies the realism-versus-physics dissociation; the single cleanest measurement in this area.
- Bansal, et al., **VideoPhy** and **VideoPhy-2** (2024–2025) — physical-commonsense evaluation of generated video; the low joint success rate of semantic-plus-physical correctness.
- Kang, et al., *How Far is Video Generation from World Models: A Physical Law Perspective* (2024) — a controlled study of whether scaling learns physical laws, finding combinatorial generalization failures consistent with interpolation.
- Ho, Salimans, et al., *Video Diffusion Models* (2022) and Peebles & Xie, *Scalable Diffusion Models with Transformers (DiT)* (2023) — the architectural foundations whose objective this post analyzes.
- [Sora and the world-simulator thesis](/blog/machine-learning/video-generation/sora-and-the-world-simulator-thesis) — the companion post on what holds up and what is hype in the world-simulator claim.
- [Video models as world models](/blog/machine-learning/video-generation/video-models-as-world-models) — the action-conditioned, interactive line (Genie, GameNGen) that is the strongest partial counter-evidence and the likeliest path to learned simulation.
- [Why video generation is hard](/blog/machine-learning/video-generation/why-video-generation-is-hard) (series foundation), [the metrics of video generation](/blog/machine-learning/video-generation/the-metrics-of-video-generation), [evaluating and red-teaming video generation](/blog/machine-learning/video-generation/evaluating-and-red-teaming-video-generation), and the [capstone playbook](/blog/machine-learning/video-generation/building-with-video-generation-the-playbook).
