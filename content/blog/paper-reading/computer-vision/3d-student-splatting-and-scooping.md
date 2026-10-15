---
title: "3D Student Splatting and Scooping: Splats Need Not Be Gaussian, or Positive"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How SSS replaces 3D Gaussians with learnable Student's t primitives, lets some of them carry negative density, and trains the result with a friction-and-noise SGHMC sampler, matching 3DGS quality with a fraction of the components."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2025", "cvpr-2025-award-candidate", "gaussian-splatting", "student-t", "novel-view-synthesis", "sghmc", "mixture-models"]
author: "Hiep Tran"
featured: true
readTime: 21
image: "/imgs/blogs/3d-student-splatting-and-scooping-fig2.webp"
excerpt: "3DGS is an unnormalized mixture model, so nothing forces its components to be Gaussian or positive. SSS swaps in fat-tailed Student's t splats, adds negative scoops, and needs a principled sampler to train them."
paper:
  title: "3D Student Splatting and Scooping"
  authors: "Jialin Zhu, Jiangbei Yue, Feixiang He, He Wang"
  venue: "CVPR 2025 (oral, award candidate)"
  url: "https://arxiv.org/abs/2503.10148"
---

> [!tldr]
> **Claim.** 3D Gaussian Splatting (3DGS) is just an unnormalized mixture model fitted to a radiance field, so its components need be neither Gaussian nor positive. SSS replaces each Gaussian with a Student's t distribution whose tail fatness $\nu$ is learned, and lets opacity range over $[-1, 1]$ so a component can **scoop** color out as well as **splat** it in.
>
> **Why it matters.** Fat tails cover flat regions with fewer primitives, and negative primitives carve holes and edges cheaply. On Mip-NeRF 360, Tanks and Temples and Deep Blending, SSS gets the best score on 6 of 9 headline metrics, and at small component budgets it beats every baseline in 15 of 15 settings.
>
> **Most surprising result.** The richer primitive is not enough on its own. With plain SGD the learned $\nu$ values collapse toward 1; the gains need a Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) sampler that decouples $\nu$ from position and shape.
>
> **Where it fails.** Training is up to about 2.75 times slower than 3DGS, the primitives are still symmetric and smooth (sharp shapes stay hard), and the supplement shows SGHMC with plain Gaussians already captures most of the gain on Tanks and Temples.

Most 3DGS follow-ups keep the Gaussian and tune everything around it. This paper (Zhu, Yue, He and Wang, UCL and Leeds) steps back: if the scene is a mixture model, the component is a design choice, and so is the sign of its weight. Three changes follow, each forcing the next. A fat-tailed primitive, then negative weights, then a sampler, because the first two make optimization badly coupled.

## 3DGS, seen as a mixture model

**The problem.** We want a 3D representation that renders photorealistic novel views in real time. 3DGS solves this with millions of small, colored, semi-transparent blobs that are projected ("splatted") onto the image and alpha-blended.

The paper's framing is that this is a mixture model. Written out, the scene density is

$$
P(x) = \sum_{i} w_i\, G_i(x), \qquad G(x) = e^{-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)}, \qquad w_i > 0
$$

where $x \in \mathbb{R}^3$ is a point in space, $\mu$ is a component's center, and $\Sigma = R S S^\top R^\top$ is its covariance built from a rotation $R$ and a diagonal scale $S$. Each component also carries an opacity $o \in [0,1]$ and a view-dependent color $c$ stored as spherical harmonics.

Rendering a pixel $u$ projects every component to a 2D blob $G^{2D}$ and composites front to back:

$$
C(u) = \sum_{i=1}^{N} c_i\, o_i\, G_i^{2D}(u) \prod_{j=1}^{i-1} \big(1 - o_j\, G_j^{2D}(u)\big)
$$

Here $N$ is the number of components hit by the ray through $u$, sorted by depth. The product is the transmittance: how much light survives everything in front of component $i$.

Gaussians were the obvious choice for two reasons. They stay Gaussian under affine maps and marginalization, so projection is closed form. And the mixture is **monotonic**: weights are positive, so components only add color. The paper argues the first property is not unique to Gaussians and the second is a restriction, not a feature.

## Technique 1: the Student's t as the primitive

**The problem.** A Gaussian falls off fast. Three standard deviations out, it is already about 1% of its peak. To paint a large smooth region such as a sky or a white wall, 3DGS has to either stretch a few Gaussians (and blur detail) or tile many small ones (and spend parameters). The paper's qualitative section blames exactly this tension for the blur 3DGS shows at small budgets.

**The analogy.** A Gaussian is a spotlight with a hard-ish edge. A fat-tailed distribution is a bare bulb: bright at the center, but it still lights the far corners of the room. To light a whole room evenly you need fewer bulbs than spotlights.

![Thin Gaussian tails need many tiles and still leave seams, while fat Student's t tails cover the same flat patch with two components (illustrative)](/imgs/blogs/3d-student-splatting-and-scooping-1.webp)

This is my illustration of the paper's argument, not one of its experiments: seven sharp-falloff Gaussians still leave seams, while two slow-falloff components cover the patch.

### The mechanism

SSS replaces each Gaussian with an unnormalized multivariate Student's t distribution:

$$
T(x \mid \nu) = \Big[ 1 + \tfrac{1}{\nu} (x-\mu)^\top \Sigma^{-1} (x-\mu) \Big]^{-\frac{\nu+3}{2}}, \qquad \nu \in [1, +\infty)
$$

The new symbol is $\nu$, the **degrees of freedom**, which controls tail fatness. The exponent's "+3" is the dimension of the space. The usual normalizing constant (a ratio of Gamma functions times $|\Sigma|^{1/2}$) is dropped because opacity absorbs the scale anyway.

The two limits are what make this a generalization rather than a replacement:

- As $\nu \to \infty$, $T$ becomes a Gaussian. Anything 3DGS can represent, SSS can too.
- As $\nu \to 1$, $T$ becomes a Cauchy distribution, with polynomial rather than exponential tails.

Because $\nu$ is learned per component, the scene becomes a mixture drawn from a continuum of distribution families instead of one.

![Figure 1 from Zhu et al. (2025): Student's t densities for several degrees of freedom, approaching the Gaussian as nu grows](/imgs/blogs/3d-student-splatting-and-scooping-fig1.webp)

### It still projects in closed form

The second reason for picking the t distribution is that, like the Gaussian, it is closed under affine transforms and marginalization. After the view transform $W$, translation $t$ and the local affine approximation $J$ of the perspective projection, the 2D footprint is

$$
T^{2D}(u) = \Big[ 1 + \tfrac{1}{\nu} (u - \mu^{2D})^\top (\Sigma^{2D})^{-1} (u - \mu^{2D}) \Big]^{-\frac{\nu+2}{2}}
$$

with $\mu^{2D} = (W\mu + t)_{1:2} / (W\mu + t)_3$ and $\Sigma^{2D} = (J W \Sigma W^\top J^\top)_{1:2,1:2}$. The exponent drops from $\nu+3$ to $\nu+2$ because one dimension was integrated out along the ray. The authors derive all gradients exactly in the supplement, which they contrast with GES, whose generalized exponential kernel needs approximations.

### A worked micro-example

Take a projected component and a pixel at squared Mahalanobis distance $d^2$ from its center. For a Gaussian, the footprint is $e^{-d^2/2}$. For $\nu = 1$ the 2D t footprint is $(1 + d^2)^{-3/2}$.

| Distance | Gaussian | t, $\nu = 1$ | Ratio |
| --- | --- | --- | --- |
| 1 sigma ($d^2 = 1$) | 0.607 | 0.354 | 0.6x |
| 3 sigma ($d^2 = 9$) | 0.0111 | 0.0316 | 2.8x |
| 5 sigma ($d^2 = 25$) | 0.0000037 | 0.0075 | about 2,000x |

Near the center the Cauchy-like component is actually lower. Five sigmas out, it still contributes visible color where the Gaussian has vanished. That is the whole bet: one component with a long reach replaces several short ones.

### Engineering consequences

Two pieces of 3DGS machinery break and the supplement describes the fixes:

- **No three-sigma cutoff.** 3DGS culls each 2D splat at three sigmas. A fat-tailed t still has real mass there, so SSS takes critical values of the t distribution for several $\nu$ (from the NIST handbook) and interpolates a per-component truncation radius.
- **No low-pass filter.** 3DGS adds 0.3 to the diagonal of $\Sigma^{2D}$ to keep splats at least a pixel wide. That is a convolution with a Gaussian, and a t convolved with anything is no longer a t, so the full model drops it and relies on the sampler to keep sizes sensible.

**When it fails.** The t is still symmetric and smooth. The authors list this as a limitation: sharp edges and asymmetric shapes remain hard, which is why the next technique matters.

## Technique 2: scooping with negative components

**The problem.** With positive weights only, a hole has to be built by surrounding it. The paper's toy experiment makes this concrete: fitting a torus seen from the front, two positive components underfit, and it takes at least five to get the topology right.

**The analogy.** A sculptor working only with clay can build a donut by laying a ring of lumps. A sculptor who can also carve takes one ball and scoops out the middle.

![Figure 2 from Zhu et al. (2025): a torus target, underfit with 2 positive components, fit with 5 positive, and fit with 1 positive plus 1 negative component](/imgs/blogs/3d-student-splatting-and-scooping-fig2.webp)

In panel (d), one positive component covers the torus and the hole, and one negative component at the same center subtracts the middle. Two components do what five could not.

### Why not a proper subtractive mixture?

There is a principled way to allow negative weights while keeping a valid density: square the mixture, as in the subtractive mixture models of Loconte et al.

$$
P(x) = \Big( \sum_{i=1}^{K} w_i T_i(x) \Big)^2 = \sum_{i=1}^{K} \sum_{j=1}^{K} w_i w_j T_i(x) T_j(x), \qquad w \in \mathbb{R}
$$

The authors tried it and rejected it. Every pair of components now interacts, so evaluation is $O(n^2)$ in the number of components, which is fatal at a million splats.

Instead, SSS keeps the ordinary compositing equation and simply lets opacity be signed. The weight becomes $w_i = c_i o_i \prod_{k \lt i} (1 - o_k T_k^{2D}(u))$ with $o \in [-1, 1]$, produced by a $\tanh$ so a component can change sign smoothly during training. A negative component removes its color from the pixel.

### A worked micro-example

Take a ray through two co-located components with the same gray color $c = 0.8$ and footprint 1 at this pixel. The splat has $o_1 = +0.9$ and the scoop has $o_2 = -0.5$. Compositing gives

$C = 0.8 \cdot 0.9 + 0.8 \cdot (-0.5) \cdot (1 - 0.9) = 0.72 - 0.04 = 0.68.$

Swap the order and you get ${0.8 \cdot (-0.5) + 0.8 \cdot 0.9 \cdot 1.5 = 0.68}$ again. For two same-colored components, the effective opacity is ${1 - (1 - o_1)(1 - o_2)}$, here 0.85 instead of 0.9. The scoop dimmed the pixel, and note that a negative component in front makes the transmittance factor exceed 1, which is part of why the paper calls the result not a well-defined distribution.

### Splat plus scoop in one dimension

The torus result is easiest to see as a cross-section through the center.

![A wide positive splat plus a narrow negative scoop at one centre leaves a ring, 0.05 in the middle and 0.47 on the rim](/imgs/blogs/3d-student-splatting-and-scooping-2.webp)

A wide splat ($o = +1.0$) plus a narrow scoop ($o = -0.95$) at the same center sum to a ring with a hole at the center: 0.05 at the dip, 0.47 on the rim at $x = \pm 1.56$. The curve shapes are mine; the construction is the paper's Figure 2(d).

**When it fails.** A signed opacity makes the mixture non-monotonic, and the usual densification heuristics of 3DGS, which clone and split based on size and gradient magnitude, assume components only add. The supplement says adding negatives to vanilla adaptive density control gives worse results, so negatives only work together with the sampler below.

## Technique 3: learning by sampling (SGHMC)

**The problem.** With the t primitive, three parameters are tightly coupled: $\nu$, $\mu$ and $\Sigma$. The authors' explanation is that changing $\nu$ changes the distribution family inside which $\mu$ and $\Sigma$ are being optimized, so every step in $\nu$ moves the target for the others. With plain SGD, the supplement reports that learned $\nu$ values pile up near 1, which the authors read as mode collapse.

![Supplementary Figure 4 from Zhu et al. (2025): histograms of learned nu, broad under SGHMC and concentrated near 1 under SGD-style optimization](/imgs/blogs/3d-student-splatting-and-scooping-fig5.webp)

**The analogy.** Plain gradient descent is a ball with no inertia on a curved valley floor: it bounces between the walls and settles in the first dip it finds. Give the ball mass and a little friction and it rolls along the valley; add some random jostling and it can hop out of shallow dips.

### The mechanism

SGHMC (Chen et al., 2014) treats the parameters $\theta$ as a particle with momentum $r$ and samples from an energy-based posterior instead of minimizing directly:

$$
P(\theta, r) \propto \exp\!\big( -\mathcal{L}_\theta(x) - \tfrac{1}{2} r^\top I r \big)
$$

Here $\mathcal{L}_\theta(x)$ is the training loss and $I$ is the identity. The energy form sidesteps the fact that a signed mixture is not a proper distribution: we never need to normalize it, only to prefer low-loss parameters.

The continuous dynamics are

$$
d\theta = M^{-1} r\, dt, \qquad dr = -\nabla U(\theta)\, dt - C M^{-1} r\, dt + \mathcal{N}(0, 2C\, dt)
$$

with mass matrix $M$, friction coefficient $C$, and $U$ the energy. Friction drains momentum; the noise term injects exactly enough randomness to keep the sampler exploring, and with stochastic minibatch gradients it compensates for the gradient noise.

The loss itself is the 3DGS-MCMC loss: a mix of L1 and D-SSIM image terms plus two regularizers, an L1 penalty on $|o_i|$ (opacity is large only where needed) and an L1 penalty on the square roots of the eigenvalues of $\Sigma$ (components stay as spiky as possible). Together they push the model toward few, small components.

### Friction and noise scheduling

SSS runs SGHMC only on the positions $\mu$, and Adam on everything else, including $\nu$, which further decouples the parameters. The discretized position update is

$$
\mu_{t+1} = \mu_t - \varepsilon^2 \frac{\partial \mathcal{L}}{\partial \mu} + F + N, \qquad F = \sigma(o)\, \varepsilon (1 - \varepsilon C)\, r_{t-1}, \qquad N = \sigma(o)\, \mathcal{N}(0, 2\varepsilon^{3/2} C)
$$

where $\varepsilon$ is the decaying learning rate. The momentum is updated as $r_{t+1} = r_t - \varepsilon\, \partial \mathcal{L}/\partial \mu - \varepsilon C r_{t-1} + \mathcal{N}(0, 2\varepsilon C)$.

The switch $\sigma(o)$ is a steep sigmoid of opacity (the paper uses $k = 100$). Its job is to turn friction and noise **on** only for nearly transparent components and **off** for confident ones, which the authors treat as already near a local optimum. With the switch off, a component moves by the gradient scaled by $\varepsilon^2$, a small local search. With friction removed entirely, the update reduces to Stochastic Gradient Langevin Dynamics, the sampler 3DGS-MCMC uses.

![SGD vs SGHMC on a coupled landscape: SGD stalls in the shallow basin near nu = 1 while SGHMC carries through to the better fit (illustrative)](/imgs/blogs/3d-student-splatting-and-scooping-3.webp)

This is my sketch on a made-up loss surface: $\nu$ and the scale of $\Sigma$ are coupled along a curved valley, SGD stalls in the shallow basin near $\nu = 1$, and SGHMC carries momentum through to the better fit.

Training has two phases:

1. **Burn-in for exploration.** Run without friction ($F$ off), so it behaves like Langevin dynamics, and multiply the noise by the component's $\Sigma$ to preserve its anisotropy.
2. **Full sampling for exploitation.** Turn friction back on and stop scaling noise by $\Sigma$; momentum now preserves the anisotropy.

Per the supplement, SGLD explores well but exploits poorly and SGHMC is the reverse, hence both phases.

## Technique 4: adding and recycling components

**The problem.** During sampling, some components fade to near-zero opacity. 3DGS deletes them and grows new ones by cloning and splitting. 3DGS-MCMC instead recycles them by moving them onto a high-opacity component, but the move must not change the rendered image.

**The mechanism.** SSS picks parameters for the relocated components that minimize the squared difference between the color profile before and after the move, integrated over the whole domain. For $N$ components placed at the location of one high-opacity component, the solution keeps the center, splits the opacity, and shrinks the covariance:

$$
\mu_{\text{new}} = \mu_{\text{old}}, \qquad (1 - o_{\text{new}})^N = 1 - o_{\text{old}}
$$

The covariance update is a longer expression involving Beta functions of $\nu_{\text{old}}$ and $\nu_{\text{new}}$ (the paper's Eq. 12); I will not reproduce it, but its role is to keep the combined footprint the same width.

### A worked micro-example

One bright component has $o_{\text{old}} = 0.80$, and two dead components (opacity 0.003) are recycled onto it, so $N = 3$. Then ${1 - o_{\text{new}} = 0.2^{1/3} \approx 0.585}$, so each copy gets $o_{\text{new}} \approx 0.415$.

![Recycling a dead component without changing the picture: two dead components move onto a live one and the three split its opacity, so a ray still sees 0.80](/imgs/blogs/3d-student-splatting-and-scooping-4.webp)

The check is the bar stack under the right panel. One copy absorbs 0.415 of the light, two copies 0.658, three copies 0.800, exactly what the single component absorbed before.

Three details matter in practice:

- **Sign follows the target.** Relocation does not distinguish positive from negative components, so a recycled component inherits the sign of the one it lands on. The authors say this sign perturbation helps the sampler mix.
- **Rate limits.** At most 5% of components are relocated at a time, for stability.
- **Growth without clone/split.** New capacity is added as 5% fresh components at zero opacity, which then get recycled like any other.

## Results

### Setup

The evaluation uses 11 scenes: 7 from Mip-NeRF 360, 2 outdoor scenes from Tanks and Temples, and 2 indoor scenes from Deep Blending. Metrics are PSNR, SSIM and LPIPS. SSS numbers are the mean of three runs on one RTX 4090; baseline numbers in the main table are copied from each baseline's paper.

| Method | Mip-NeRF 360 PSNR / SSIM / LPIPS | Tanks and Temples | Deep Blending |
| --- | --- | --- | --- |
| Mip-NeRF | 29.23 / 0.844 / 0.207 | 22.22 / 0.759 / 0.257 | 29.40 / 0.901 / 0.245 |
| 3DGS | 28.69 / 0.870 / 0.182 | 23.14 / 0.841 / 0.183 | 29.41 / 0.903 / 0.243 |
| GES | 26.91 / 0.794 / 0.250 | 23.35 / 0.836 / 0.198 | 29.68 / 0.901 / 0.252 |
| 3DHGS | 29.56 / 0.873 / 0.178 | 24.49 / 0.857 / 0.169 | 29.76 / 0.905 / 0.242 |
| Fre-GS | 27.85 / 0.826 / 0.209 | 23.96 / 0.841 / 0.183 | 29.93 / 0.904 / 0.240 |
| Scaffold-GS | 28.84 / 0.848 / 0.220 | 23.96 / 0.853 / 0.177 | 30.21 / 0.906 / 0.254 |
| 3DGS-MCMC | 29.89 / 0.900 / 0.190 | 24.29 / 0.860 / 0.190 | 29.67 / 0.890 / 0.320 |
| **SSS** | **29.90** / 0.893 / **0.145** | **24.87** / **0.873** / **0.138** | 30.07 / **0.907** / 0.247 |

SSS is best on 6 of 9 columns. The misses are honest ones: 3DGS-MCMC has the higher SSIM on Mip-NeRF 360, Scaffold-GS the higher PSNR on Deep Blending, and Fre-GS the lowest Deep Blending LPIPS (0.240, a gap of 0.007). The largest single-scene margin is on Train: 23.23 / 0.844 / 0.170 against 3DHGS's 22.95 / 0.827 / 0.197.

### Parameter efficiency is the real headline

The more interesting experiment caps the component count. With $\delta$ the number of SfM points used for initialization, every method is limited to $\delta$, ${1.4\delta}$, ${1.8\delta}$, ${2.2\delta}$ or ${2.6\delta}$ components. Even ${2.6\delta}$ is small: from 1.7% of 3DGS's count on Stump (83k against 4.75m) to 42.5% on Train (468k against 1.1m).

![Figure 4 from Zhu et al. (2025): average PSNR against component budget on Mip-NeRF 360, Tanks and Temples and Deep Blending, with SSS on top in every setting](/imgs/blogs/3d-student-splatting-and-scooping-fig3.webp)

SSS is best in all 15 dataset-by-budget settings, and its curve degrades the slowest as the budget shrinks. On Tanks and Temples it reaches 23.6 PSNR with about 180k components, already above Mip-NeRF, 3DGS and GES at their full budgets of a million or more.

![Figure 5 from Zhu et al. (2025): the Train scene rendered by each method at budgets from 180k to 468k components](/imgs/blogs/3d-student-splatting-and-scooping-fig4.webp)

At 180k components, 3DGS and 3DHGS blur, GES smears sky into hill, and 3DGS-MCMC separates them but adds white patches to the sky. SSS keeps a clean sky and resolves trees on the hill, which the authors attribute to fat tails for flat areas and thin ones for detail.

### Ablation: where the gain actually comes from

The supplement's Table 20 is the most important table in the paper, and it is more nuanced than the main text.

| Setting (Tanks and Temples) | PSNR | SSIM | LPIPS |
| --- | --- | --- | --- |
| 3DGS | 23.14 | 0.841 | 0.183 |
| SGD + positive t | 23.80 | 0.838 | 0.191 |
| SGHMC + Gaussians | 24.52 | 0.869 | 0.150 |
| SGHMC + positive t | 24.53 | 0.864 | 0.155 |
| Full SSS (SGHMC + signed t) | 24.87 | 0.873 | 0.138 |

Read it row by row. The t primitive alone (with SGD) adds 0.66 PSNR. The sampler alone, with ordinary Gaussians, adds 1.38. Putting the t primitive on top of the sampler adds 0.01 PSNR and slightly worsens SSIM and LPIPS. The negatives then add 0.34 PSNR. On Deep Blending the pattern differs: SGHMC with Gaussians gets 29.56, positive t gets 29.75, and the full model 30.07.

### Cost

| Dataset | 3DGS FPS / train | 3DGS-MCMC | SSS |
| --- | --- | --- | --- |
| Mip-NeRF 360 | 99 / 21 min | 79 / 32 min | 71 / 45 min |
| Tanks and Temples | 120 / 12 min | 105 / 17 min | 87 / 33 min |
| Deep Blending | 119 / 21 min | 138 / 29 min | 100 / 21 min |

SSS stays real-time (above 70 FPS everywhere) but trains up to about 2.75 times slower than 3DGS on Tanks and Temples.

**What might not transfer.** The baseline numbers in the main table come from different papers, codebases and possibly different image resolutions. The reduced-budget comparison is cleaner because the authors re-ran every baseline, but they did so by capping densification in code not designed for it, which may handicap clone-and-split methods more than samplers.

## Critique

**What is strong.**

- The reframing is clean and general. Once 3DGS is read as a mixture model, both the primitive and the sign become free design choices, and the paper shows both can pay off.
- The closed-form projection is a real contribution. A heavy-tailed primitive that rasterizes exactly, with exact gradients, is much easier to build on than one needing approximations.
- The budget sweep, over 11 scenes, 5 budgets and 3 metrics (165 comparisons), is a far more informative test of expressivity than a single full-budget table.

**What is weak.**

- Attribution is muddier than the abstract suggests. Table 20 shows SGHMC with Gaussians matches SGHMC with positive t on Tanks and Temples. On that dataset, the t primitive's contribution is close to zero once the sampler is in place; the gain is the sampler plus negatives.
- There is no count of how many components end up negative, or where they sit in the scene. Without that, "scooping" is shown on a torus and inferred on real scenes.
- The ablation is on 2 of the 3 datasets, and there is no "SGD + signed t" row. The authors say negatives are designed for the sampler, which is plausible but untested in the paper.
- The opacity switch is described with $k = 100$ and $t = 0.995$ and said to activate below opacity 0.005; as printed, the formula and the threshold do not obviously agree, so readers should check the released code.

**What would change my mind.** If a budget sweep with SGHMC plus plain Gaussians (no t, no negatives) matched SSS at ${1.4\delta}$ to ${2.6\delta}$, I would read SSS as mostly a sampler paper. Conversely, a per-scene count showing that a meaningful fraction of the final components are negative and sit on edges and holes would make the "scooping" claim much stronger than the torus alone does.

## Where this goes next

The paper's own future work is to mix in other families (Laplace) and make the negative fraction self-tuning. The ideas below are mine; each is chosen because SSS makes it cheap to try now.

1. **Signed, fat-tailed heads for feed-forward 3D.** *Question:* can a single-pass predictor (pixelSplat or MVSplat style) learn to emit useful scoops and per-splat $\nu$ without any per-scene sampler? *Why now:* SSS gives a closed-form 2D footprint and exact gradients, so swapping the Gaussian head is a drop-in change, and feed-forward models are exactly the budget-limited regime where SSS shines. *First experiment:* replace the Gaussian parameter head of an open feed-forward model with $(\mu, \Sigma, \nu, \tanh o)$, train on RealEstate10K at a fixed number of splats per pixel, and report PSNR plus the histogram of predicted $\nu$ and sign. If $\nu$ collapses to 1 as it did under SGD here, that is a result too.

2. **Heavy tails in time for 4D scenes.** *Question:* does a t distribution along the time axis let one primitive persist across long static stretches while thin-tailed ones handle fast motion? *Why now:* the closed-form marginalization that gives SSS its 2D projection also gives clean temporal slicing. *First experiment:* in a 4D Gaussian splatting codebase, give each primitive a temporal $\nu_t$, cap the primitive count, and compare PSNR per primitive on a dynamic benchmark, logging how $\nu_t$ correlates with measured motion.

3. **Signed Student's t for 2D image splatting.** *Question:* in image representation and compression with 2D splats (GaussianImage style), do fat tails win on smooth gradients and negative splats on thin dark lines? *Why now:* the 2D case removes projection and sorting, so the primitive's effect is isolated and fast to measure. *First experiment:* fit Kodak images at fixed splat budgets with Gaussian, positive t and signed t, and plot rate-distortion curves (PSNR against bits per pixel after quantization).

4. **Asymmetric and sharp primitives with scooping.** *Question:* the authors' main limitation is symmetric, smooth primitives. Does a positive smooth splat plus a negative half-space or half-t component reproduce a sharp edge more cheaply than either alone? *Why now:* 3DHGS already shows half-Gaussians rasterize; SSS shows signs and sampling work. *First experiment:* on the paper's torus toy and on a synthetic scene with crisp edges, count the components each primitive family needs to reach a target PSNR.

5. **What do negative components actually learn?** *Question:* where do negatives end up in a trained real scene, and what fraction survive? *Why now:* the code is released and the opacity sign is directly readable. *First experiment:* train SSS on all 11 scenes, record the negative fraction over training, and render negatives alone to see whether they concentrate on edges, holes, or floaters.

For sampling background, the [MCMC primer](/blog/trading/math-for-quants/mcmc-metropolis-gibbs-math-for-quants) covers Metropolis and Gibbs from scratch. For a different CVPR 2025 fix for reconstructions, [Difix3D+](/blog/paper-reading/computer-vision/difix3d-improving-3d-reconstructions-with-single-step-diffusion-models) repairs 3DGS artifacts with a single-step diffusion model instead of changing the primitive.

## References

- Jialin Zhu, Jiangbei Yue, Feixiang He, He Wang. *3D Student Splatting and Scooping.* CVPR 2025. [arXiv:2503.10148](https://arxiv.org/abs/2503.10148), [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_3D_Student_Splatting_and_Scooping_CVPR_2025_paper.html).
- Code: [github.com/realcrane/3D-student-splatting-and-scooping](https://github.com/realcrane/3D-student-splatting-and-scooping).
- Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG 2023.
- Kheradmand et al. *3D Gaussian Splatting as Markov Chain Monte Carlo.* NeurIPS 2024.
- Chen, Fox, Guestrin. *Stochastic Gradient Hamiltonian Monte Carlo.* ICML 2014.
- Loconte et al. *Subtractive Mixture Models via Squaring: Representation and Learning.* ICLR 2024.
