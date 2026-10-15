---
title: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How 3DGUT swaps the Jacobian linearization inside 3D Gaussian Splatting for the Unscented Transform, so a rasterizer can train on fisheye and rolling-shutter cameras and share one scene with a ray tracer for reflections and refractions."
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["3d-reconstruction", "cvpr-2025", "cvpr-2025-oral", "gaussian-splatting", "unscented-transform", "fisheye", "rolling-shutter", "novel-view-synthesis", "ray-tracing"]
author: "Hiep Tran"
featured: true
readTime: 20
image: "/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-fig1.webp"
excerpt: "3DGS can only see the world through a perfect pinhole. 3DGUT replaces one linearization step with seven sigma points and suddenly fisheye lenses, rolling shutters and traced reflections all work inside a rasterizer."
paper:
  title: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
  authors: "Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic"
  venue: "CVPR 2025 (oral)"
  url: "https://arxiv.org/abs/2412.12507"
---

> [!tldr]
> **Claim.** 3D Gaussian Splatting (3DGS) projects each 3D Gaussian to the screen by linearizing the camera at the particle's center (EWA splatting). 3DGUT keeps the camera exact and approximates the *particle* instead: seven sigma points go through any projection function, and a 2D Gaussian is re-fitted from where they land.
> **Why it matters.** That one swap lets a rasterizer train directly on fisheye and rolling-shutter footage with no per-camera Jacobian, and, combined with evaluating particles in 3D, makes the same scene renderable by a ray tracer for reflections and refractions.
> **Most surprising result.** On fisheye ScanNet++ it beats FisheyeGS, a method with a hand-derived fisheye Jacobian, by about 1 dB PSNR (29.11 vs 28.15) while using 0.38M Gaussians instead of 1.07M.
> **Where it fails.** It is still about 30% slower than vanilla 3DGS on pinhole data (3.77 ms vs 2.88 ms per frame), the re-fitted 2D Gaussian stops describing the true footprint under extreme distortion, and one sample per particle per ray cannot render overlapping Gaussians correctly.

Gaussian splatting is fast because it is a rasterizer, and rasterizers assume the camera is a simple map from 3D to 2D. Cameras on cars and robots are not: fisheye lenses bend lines into arcs, and rolling shutters read the image row by row while the vehicle moves. 3DGUT (NVIDIA and University of Toronto) asks for the smallest change to 3DGS that removes the pinhole assumption without giving up rasterization. The answer borrows a 30-year-old trick from Kalman filtering.

![Figure 1 from Wu et al. (2025): left, a scene trained on undistorted crops vs the original fisheye views; right, a reflective sphere and a refractive statue inserted into a 3DGUT reconstruction](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-fig1.webp)

The left half of Figure 1 is the practical argument: undistorting a fisheye image for a pinhole renderer throws away the periphery, while training on the original views keeps every pixel. The right half is the second contribution, secondary rays, which no plain rasterizer can produce.

## The scene representation, in three equations

3DGS represents a scene as an unordered cloud of fuzzy 3D blobs ("particles"), each with a center, a shape, an opacity and a view-dependent color.

The particle's **response** at a 3D point $\mathbf{x} \in \mathbb{R}^3$ is an unnormalized Gaussian:

$$
\rho(\mathbf{x}) = \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

Here $\boldsymbol{\mu} \in \mathbb{R}^3$ is the particle center and $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times 3}$ its covariance, which sets the blob's size and orientation. The response is 1 at the center and falls off smoothly in every direction.

To keep $\boldsymbol{\Sigma}$ valid during gradient descent, it is factored as

$$
\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top
$$

where $\mathbf{R} \in SO(3)$ is a rotation (stored as a quaternion $\mathbf{q} \in \mathbb{R}^4$) and $\mathbf{S}$ is a diagonal scale matrix (stored as $\mathbf{s} \in \mathbb{R}^3$). Read it as: take a unit sphere, stretch it along three axes by $s_1, s_2, s_3$, then rotate. The columns $\mathbf{r}_k$ of $\mathbf{R}$ are the ellipsoid's principal axes. That factorization will matter a lot in a minute.

Each particle also has an opacity $\sigma \in \mathbb{R}$ and a spherical-harmonic color $\phi_\beta(\mathbf{d})$. A pixel's color is front-to-back alpha compositing over the $N$ particles its ray touches:

$$
\mathbf{c}(\mathbf{o},\mathbf{d}) = \sum_{i=1}^{N} \mathbf{c}_i(\mathbf{d})\,\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j), \qquad \alpha_i = \sigma_i\,\rho_i(\mathbf{o}+\tau\mathbf{d})
$$

with ray origin $\mathbf{o}$, direction $\mathbf{d}$, and $\tau$ a distance along the ray. The product term is transmittance: how much light survives the particles in front.

This equation leaves three things open: which particles touch which pixels, in what order, and at which $\tau$ to evaluate $\rho_i$. The paper frames its whole method as answers to those three questions.

## How 3DGS projects a particle, and why that is the bottleneck

A rasterizer needs, for every particle, a 2D footprint so it can bin particles into 16x16 pixel tiles, sort them, and blend. 3DGS gets that footprint from EWA splatting (Zwicker et al., 2002).

**The problem.** A camera projection $g$ is nonlinear: even a pinhole divides by depth, and a fisheye maps the angle off the optical axis to image radius. A Gaussian pushed through a nonlinear function is no longer Gaussian. Its footprint on the image is some curved, skewed blob, and there is no closed form for it.

**The analogy.** EWA is a flat paper map of a patch of the globe: excellent where the paper touches, stretched far from it. If the patch is large relative to the curvature, the map lies.

**The mechanism.** EWA takes the first-order Taylor expansion of the projection at the particle center. Locally, $g$ becomes a matrix $\mathbf{J}$, the Jacobian. A linear map sends a Gaussian to a Gaussian, so the 2D covariance is just the 3D one sandwiched by that matrix:

$$
\boldsymbol{\Sigma}' = \mathbf{J}_{[:2,:3]}\,\mathbf{W}\,\boldsymbol{\Sigma}\,\mathbf{W}^\top\,\mathbf{J}_{[:2,:3]}^\top
$$

Here $\mathbf{W} \in SE(3)$ moves the particle from world to camera coordinates, $\mathbf{J}$ is the Jacobian of the projection evaluated at the center, and $\boldsymbol{\Sigma}' \in \mathbb{R}^{2\times 2}$ is the screen-space covariance (the "2D conic"). The 2D center is simply $g(\boldsymbol{\mu})$.

![Figure 2 from Wu et al. (2025): Monte Carlo sampling gives the true projected distribution, EWA linearizes the projection, the Unscented Transform pushes sigma points through it exactly](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-fig2.webp)

**When it fails.** The paper lists three failure modes, and each one motivates part of the method:

- **Truncation error.** Dropping the higher-order Taylor terms causes projection errors *even for a perfect pinhole* (citing Huang et al., 2024), and they grow with distortion.
- **One Jacobian per camera model.** Every new lens model needs a new, hand-derived $\mathbf{J}$. FisheyeGS did exactly that for the equidistant fisheye. It is tedious and easy to get subtly wrong.
- **One function for the whole particle.** Linearization needs a single projection function. A rolling-shutter camera does not have one: each image row is captured from a slightly different pose. How do you take the Jacobian of "a different camera for every row"?

![Before and after: EWA linearizes the fisheye camera at the particle center and produces a straight ellipse that misses the curved true footprint, while the Unscented Transform pushes each sigma point through the exact projection and re-fits an ellipse to the crescent](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-1.webp)

This is the one-picture summary of the paper. On the left, the tangent at $g(\boldsymbol{\mu})$ is good near the center and bad at the tips, so the EWA ellipse covers pixels the particle never reaches and misses pixels it does. On the right, only the particle is approximated, never the camera.

## The Unscented Transform: approximate the particle, not the camera

**The problem it solves.** We want the mean and covariance of $g(\mathbf{x})$ where $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, for an arbitrary $g$ we can evaluate but do not want to differentiate.

**The analogy.** To learn how a crowd spreads out after a twisting corridor, you could model the corridor's curvature (EWA), or send seven well-chosen scouts through it and see where they come out. The scouts experience the corridor exactly; only the crowd is approximated. That is the Unscented Transform (UT), from Julier and Uhlmann's nonlinear Kalman filtering work.

**The mechanism.** For a 3D Gaussian, pick ${2N+1 = 7}$ deterministic "sigma points": the center, plus one point on each side of the center along each principal axis. Push each through the exact projection. Then compute a weighted mean and weighted covariance of the seven 2D landing spots. The weights are chosen so that, for a linear $g$, you recover the exact answer; for a nonlinear $g$ the error is pushed to higher order than linearization leaves it.

**The math.** The sigma points $\mathcal{X} = \{\mathbf{x}_i\}_{i=0}^{6}$ are

$$
\mathbf{x}_i =
\begin{cases}
\boldsymbol{\mu} & i = 0\\[2pt]
\boldsymbol{\mu} + \sqrt{(3+\lambda)\boldsymbol{\Sigma}}_{[i]} & i = 1,2,3\\[2pt]
\boldsymbol{\mu} - \sqrt{(3+\lambda)\boldsymbol{\Sigma}}_{[i-3]} & i = 4,5,6
\end{cases}
$$

where $\sqrt{\cdot}_{[i]}$ means the $i$-th column of a matrix square root. This is where the factorization pays off: $\mathbf{R}\mathbf{S}$ *is* a square root of $\boldsymbol{\Sigma}$, so column $k$ is just $s_k \mathbf{r}_k$, the $k$-th principal axis scaled by its length. No Cholesky, no eigendecomposition. The 3 in ${3+\lambda}$ is the dimension of the space.

The spread parameter is $\lambda = \alpha^2(3+\kappa) - 3$, where $\alpha$ controls how far the points sit from the mean and $\kappa$ is a secondary scale usually set to 0. The weights for the mean ($w^\mu$) and covariance ($w^\Sigma$) are

$$
w_0^\mu = \frac{\lambda}{3+\lambda}, \quad
w_0^\Sigma = \frac{\lambda}{3+\lambda} + (1-\alpha^2+\beta), \quad
w_i^\mu = w_i^\Sigma = \frac{1}{2(3+\lambda)} \;\; (i=1,\dots,6)
$$

where $\beta$ encodes prior knowledge about the distribution's shape ($\beta = 2$ is the standard choice for Gaussians in the UKF literature). Each point is projected exactly, $\mathbf{v}_{x_i} = g(\mathbf{x}_i)$, and the 2D Gaussian is re-estimated as

$$
\mathbf{v}_{\mu} = \sum_{i=0}^{6} w_i^\mu\,\mathbf{v}_{x_i}, \qquad
\boldsymbol{\Sigma}' = \sum_{i=0}^{6} w_i^\Sigma\,(\mathbf{v}_{x_i}-\mathbf{v}_{\mu})(\mathbf{v}_{x_i}-\mathbf{v}_{\mu})^\top
$$

with $\mathbf{v}_\mu \in \mathbb{R}^2$ the projected center and $\boldsymbol{\Sigma}' \in \mathbb{R}^{2\times 2}$ the projected covariance. Compare with EWA: there is no $\mathbf{J}$ anywhere. The camera appears only as a black-box function $g$ called seven times.

![The seven sigma points of a 3D Gaussian sit at the center and at plus and minus sqrt(3) scaled lengths along each principal axis; with alpha 1, beta 2, kappa 0 the center has mean weight 0 and covariance weight 2 while the six outer points each weigh one sixth](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-2.webp)

**The paper's settings make this unusually clean.** The authors use $\alpha = 1$, $\beta = 2$, $\kappa = 0$ in all experiments. Plug in:

| Quantity | Value | Meaning |
| --- | --- | --- |
| $\lambda = 1\cdot(3+0) - 3$ | 0 | no extra spread |
| distance of points from center | $\sqrt{3}\,s_k$ | about 1.73 standard deviations per axis |
| $w_0^\mu$ | 0 | the center does not move the mean |
| $w_i^\mu$, $i \ge 1$ | 1/6 | the mean is the average of the six outer points |
| $w_0^\Sigma$ | 0 + (1 − 1 + 2) = 2 | the center pulls the covariance |
| $w_i^\Sigma$, $i \ge 1$ | 1/6 | outer points share the rest |

The center's covariance weight of 2 multiplies $(\mathbf{v}_{x_0} - \mathbf{v}_\mu)$. For a linear camera the outer points are symmetric around $g(\boldsymbol{\mu})$, so that term vanishes. It only switches on when the projection *bends* the particle and its mean drifts off the projected center: the UT quietly correcting for curvature.

**A worked micro-example (my numbers, not the paper's).** Take an equidistant fisheye, image radius $r = f\theta$ where $\theta$ is the angle off the optical axis, with focal length $f = 1$. Put an isotropic Gaussian with $s_k = 0.15$ at $\boldsymbol{\mu} = (1, 0, 0)$ in camera coordinates, exactly 90 degrees off axis. A pinhole cannot image this point at all; a fisheye can.

With $\lambda = 0$ the outer points sit $\sqrt{3}\times 0.15 \approx 0.26$ from the center. Projecting the seven points, the two along the x axis both land at $u = 1.571$ (they share the same 90 degree angle), the one pushed toward the camera's front lands at 1.317, the one pushed backward at 1.825, and the two along y at 1.520. The weighted mean is

$$
v_{\mu,u} = \tfrac{1}{6}(1.571 + 1.571 + 1.520 + 1.520 + 1.317 + 1.825) \approx 1.554
$$

EWA reports the center at $g(\boldsymbol{\mu}) = \pi/2 \approx 1.571$. A 400,000-sample Monte Carlo reference gives 1.553. So the UT mean is right to the third decimal and EWA is off by 0.017 focal lengths, which on a camera with a 500-pixel focal length is a shift of about 9 pixels. On the spread, the honest picture is mixed: UT matches Monte Carlo on the x variance (0.0227 vs 0.0227, EWA 0.0225) but EWA happens to be closer on the y variance (EWA 0.0555, UT 0.0520, Monte Carlo 0.0559). Seven points are a cheap approximation, not an exact one.

**Why it works, and when it fails.** UT sees curvature because its points sample the curved region; linearization sees only the slope at one point. But under extreme distortion the true footprint is not elliptical, so *any* 2D Gaussian describes it poorly, and since the ellipse decides which tiles a particle touches, an under-sized one wrongly culls the particle from pixels it should cover.

## Rolling shutter falls out for free

Rolling-shutter cameras expose one row at a time, so on a moving car the top and bottom rows see the world from different poses and a vertical pole comes out slanted. For linearization this is a nightmare: the projection depends on which row the point lands in, which depends on the projection. For the UT it is almost trivial. The paper's phrasing is that time-dependent effects "can directly be represented by transforming each sigma point with a different extrinsic matrix". Each sigma point is projected with the camera pose at the time its row was read out.

![Rolling shutter geometry: the sensor reads rows top to bottom over time, the camera pose changes during readout, and each sigma point of a Gaussian is projected with the pose at the time of the row it lands in, so a box comes out sheared like the real image](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-3.webp)

The paper does not spell out how a sigma point's row time is chosen when the row itself depends on the pose. A fixed-point iteration (project with a guess, read the row, re-project with that row's pose) is the natural way to do it; treat that detail as my reconstruction, not the authors' description.

![Figure 7 from Wu et al. (2025): frame tiles of a solid box rendered by a panning rolling-shutter camera; 3DGRT (a) and 3DGUT (b) recover the true geometry, 3DGS (c) cannot model the time-dependent sensor](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-fig7.webp)

On the synthetic rolling-shutter box from the 3DGRT paper, the PSNR values printed under the panels all sit between 46.7 and 48.7 dB, so the score is not the story; the recovered geometry is. The more useful evidence is in the supplementary KL study, discussed below: EWA's projection error explodes under rolling shutter, and UT's barely moves.

## Evaluating the particle in 3D, not on the screen

The UT gives us a 2D ellipse per particle. The surprising design decision is that **3DGUT does not use that ellipse to compute how much a particle contributes to a pixel**. The ellipse is only an acceleration structure: it decides which tiles and pixels a particle may touch. The actual contribution is computed back in 3D.

**The problem.** 3DGS evaluates the response on the projected 2D conic, so gradients flow through the approximated projection, and the same particles render differently under a ray tracer, which evaluates in 3D.

**The mechanism.** Following 3DGRT (the same group's Gaussian ray tracer), for each ray and particle we take a single sample at the point along the ray where the particle's response is largest, $\tau_{\max} = \arg\max_\tau \rho(\mathbf{o}+\tau\mathbf{d})$.

**The math.** Transform the ray into the particle's canonical space, where the ellipsoid becomes a unit sphere at the origin:

$$
\mathbf{o}_g = \mathbf{S}^{-1}\mathbf{R}^\top(\mathbf{o}-\boldsymbol{\mu}), \qquad
\mathbf{d}_g = \mathbf{S}^{-1}\mathbf{R}^\top\mathbf{d}
$$

In that space the response depends only on distance to the origin, so its maximum along the ray is the foot of the perpendicular from the origin:

$$
\tau_{\max} = \frac{(\boldsymbol{\mu}-\mathbf{o})^\top\boldsymbol{\Sigma}^{-1}\mathbf{d}}{\mathbf{d}^\top\boldsymbol{\Sigma}^{-1}\mathbf{d}} = \frac{-\mathbf{o}_g^\top\mathbf{d}_g}{\mathbf{d}_g^\top\mathbf{d}_g}
$$

If $\omega_g$ is the distance from the origin to that closest point, the particle's alpha for this ray is $\alpha = \sigma e^{-0.5\,\omega_g^2}$. A ray straight through the center gets the full opacity $\sigma$; a ray grazing the edge gets almost nothing.

![Evaluating a particle in 3D: the ray through a tilted ellipsoid becomes a straight line past a unit sphere in canonical space, tau_max is the closest approach to the origin, and the response along the ray is a bell curve peaking there with alpha equal to sigma times exp of minus half omega_g squared](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-4.webp)

**Tiny example.** Take a unit-sphere particle ($\mathbf{R} = \mathbf{I}$, $\mathbf{S} = \mathbf{I}$) at the origin, a ray from $\mathbf{o} = (-3, 0.5, 0)$ along $\mathbf{d} = (1, 0, 0)$. Then $\mathbf{o}_g = \mathbf{o}$, $\mathbf{d}_g = \mathbf{d}$, $\tau_{\max} = 3$, the closest point is $(0, 0.5, 0)$, $\omega_g = 0.5$, and $\alpha = \sigma e^{-0.125} \approx 0.88\,\sigma$.

**Why it works.** The ellipse's quality now affects only *which* pixels are considered, not *what* they receive, and the backward pass never differentiates through the projection (the supplement derives a stable chain, e.g. $\partial \mathbf{o}_g / \partial \boldsymbol{\mu} = -\mathbf{S}^{-1}\mathbf{R}^\top$).

**The cost.** 3DGS densification uses 2D screen-space gradients, which no longer exist, so (like 3DGRT) the authors substitute 3D positional gradients divided by half the camera distance. That heuristic matters for particle counts, as the critique notes.

## Sorting: getting particles in ray order

3DGRT collects hits in exact $\tau_{\max}$ order using a BVH. 3DGS sorts once per tile by center depth, which is wrong for overlapping or elongated particles and causes "popping" as the camera moves. 3DGUT's **sorted** variant uses multi-layer alpha blending (MLAB), as StopThePop does: each ray keeps a buffer of its $k = 16$ farthest hits so far in sorted order, and hits that do not fit are blended incrementally until transmittance vanishes. Hybrid transparency would recover the exact $k$ closest hits but must visit every particle.

With UT projection, 3D evaluation and per-ray sorting, the rasterizer and 3DGRT agree on the rendering equation, the response and (approximately) the order. That alignment unlocks secondary rays.

## Hybrid rendering: rasterize primary rays, trace secondary ones

Because a 3DGUT scene renders almost the same under 3DGRT, you can mix them in one frame: rasterize primary rays with 3DGUT, discard Gaussian hits behind each ray's closest intersection with an inserted object, then trace reflected or refracted rays with 3DGRT through the same particles.

![Figure 9 from Wu et al. (2025): top, a rolling-shutter render and a strong lens distortion applied at render time; bottom, hybrid rendering with splatted primary rays and traced secondary rays producing refraction and reflection](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-fig9.webp)

The paper's Figure 8 quantifies the alignment by tracing scenes trained with each method: 3DGS and StopThePop score PSNR 19.80 and 19.26, 3DGUT (sorted) scores 25.10. That gap is why hybrid rendering works here and would look broken on vanilla 3DGS.

## Results

All timings are on a single NVIDIA RTX 6000 Ada. Training follows 3DGS: 30k iterations, loss $\mathcal{L} = \mathcal{L}_2 + 0.2\,\mathcal{L}_{\text{SSIM}}$, densify and prune every 300 iterations.

**Pinhole benchmarks (Table 1).** On standard data, the goal is "no regression", and that is roughly what happens.

| Method | Complex cameras | No popping | MipNeRF360 PSNR | SSIM | LPIPS | FPS | T&T PSNR | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ZipNeRF | yes | n/a | 28.54 | 0.828 | 0.219 | 0.2 | n/a | n/a |
| 3DGS | no | no | 27.26 | 0.803 | 0.240 | 347 | 23.64 | 476 |
| 3DGUT | yes | no | 27.26 | 0.810 | 0.218 | 265 | 23.21 | 277 |
| StopThePop | no | yes | 27.14 | 0.804 | 0.235 | 340 | 23.15 | 482 |
| 3DGRT | yes | yes | 27.20 | 0.818 | 0.248 | 52 | 23.20 | 190 |
| EVER | yes | yes | 27.51 | 0.825 | 0.233 | 36 | n/a | n/a |
| 3DGUT (sorted) | yes | yes | 27.26 | 0.812 | 0.215 | 200 | 22.90 | 272 |

Among complex-camera methods, 3DGUT is 5x faster than 3DGRT on MipNeRF360 (265 vs 52 FPS) at equal PSNR. Against 3DGS it ties PSNR and wins LPIPS on MipNeRF360, but on Tanks and Temples loses 0.4 to 0.7 dB and runs at 58% of the frame rate. Table 2 shows where the time goes: per frame, 3DGS takes 2.88 ms, 3DGUT 3.77 ms, sorted 4.98 ms, 3DGRT 19.24 ms; the UT preprocess step more than doubles (0.59 to 1.34 ms).

**Fisheye (Table 3, ScanNet++).** This is where the method should shine, and it does.

| Method | PSNR | SSIM | LPIPS | Gaussians |
| --- | --- | --- | --- | --- |
| 3DGS (undistorted training, from FisheyeGS paper) | 22.76 | 0.798 | n/a | 1.31M |
| FisheyeGS (reported) | 27.86 | 0.897 | n/a | 1.25M |
| FisheyeGS (rerun) | 28.15 | 0.901 | 0.261 | 1.07M |
| 3DGUT (sorted) | 29.11 | 0.910 | 0.252 | 0.38M |

A generic method with no fisheye-specific code beats a method built around a custom fisheye Jacobian by about 1 dB, with roughly a third of the particles.

**Distorted plus rolling shutter (Table 5, Waymo, 9 static scenes).** 3DGUT (sorted) scores PSNR 30.16 / SSIM 0.900 vs 29.99 / 0.897 for 3DGRT. 3DGS (29.83 / 0.917) is trained on rectified images with rolling shutter ignored, which the paper flags as not directly comparable.

![Figure 14 from Wu et al. (2025): per-Gaussian KL divergence of EWA and UT projections against a Monte Carlo reference, for pinhole and fisheye cameras with static and rolling-shutter poses](/imgs/blogs/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting-fig14.webp)

**Projection quality, measured directly.** The cleanest evidence is in the supplement: for each visible Gaussian of a MipNeRF360 bicycle reconstruction, EWA and UT 2D Gaussians are compared to a 500-sample Monte Carlo reference by KL divergence. Static pinhole: alike. Static fisheye: UT better. Rolling shutter: UT still tracks the reference while EWA "breaks down" into tearing artifacts. Sweeping radial distortion $k_2$ from 0 to 0.5 leaves UT's median KL essentially flat (${4.4\times10^{-3}}$ to ${4.3\times10^{-3}}$), and rolling-shutter lateral translation from 0 to 0.35 moves it from ${4.4\times10^{-3}}$ to ${4.6\times10^{-3}}$.

## Critique

**What is strong.** The idea is small and drop-in: the supplement shows it replaces exactly one function (`Estimate2DGaussian`) of the 3DGS rasterizer, removing a whole category of per-camera engineering. Using the ellipse only for culling and evaluating in 3D is the quiet second idea that makes rasterizer and tracer agree. The KL study measures the quantity the method changes, not just end-to-end PSNR.

**What is weak or missing.**

- **No clean ablation of UT alone on fisheye.** ScanNet++ reports only the sorted variant (one baseline, six scenes), so part of the 1 dB may come from 3D evaluation and sorting rather than the projection. Waymo also excludes dynamic objects.
- **The particle count gap is unexplained.** 0.38M vs 1.07M Gaussians is a large difference. It could come from better footprints, or from the 3D-gradient densification heuristic simply densifying less. The paper does not separate these.
- **The approximation caps out.** The authors acknowledge non-Gaussian footprints under large distortion, but never measure how often particles get wrongly culled at, say, a 200 degree field of view.
- **Overlapping particles.** One sample at $\tau_{\max}$ per particle ignores the integral through overlapping Gaussians. EVER handles this exactly; 3DGUT does not.
- **Speed parity is not quite parity.** About 30% slower than 3DGS on pinhole data, and Tanks and Temples PSNR is lower.

**What would change my mind.** If an ablation showed that most of the fisheye gain disappears when you hold sorting and 3D evaluation fixed and swap UT back for a correct fisheye Jacobian, I would read 3DGUT as mainly a rendering-alignment paper with a convenient projection, rather than a projection paper. Conversely, a culling-error measurement showing UT footprints stay conservative up to very wide fields of view would make me much more confident in deploying it on 220 degree automotive lenses.

## Where this goes next

Here the platform is "the camera is a black box called seven times per particle". These follow-ups are mine, not the authors'.

1. **Calibrate the camera inside the splat optimizer.** *Question:* can lens distortion coefficients and rolling-shutter readout time be learned jointly with the scene? *Why now:* the UT needs only forward evaluations of $g$, so gradients reach unknown camera parameters through the sigma point projections with no hand-derived Jacobian. *First experiment:* start Waymo or ScanNet++ training from deliberately wrong $k_1, k_2$ and readout time, and plot recovered parameters vs ground truth calibration.

2. **Refraction-aware primary projection.** *Question:* can you rasterize through a flat refracting interface (underwater housings, windshields, aquarium glass) instead of tracing it? *Why now:* Snell refraction at a known plane is a closed-form nonlinear map, exactly what sigma points can go through. *First experiment:* render a synthetic scene behind a planar glass interface, compare UT-through-refraction rasterization with full 3DGRT tracing on PSNR and FPS.

3. **Adaptive sigma point sets for extreme distortion.** *Question:* when the footprint stops being elliptical, can more points (a higher-order or spherical-simplex sigma set) or a mixture of two ellipses per particle fix culling errors? *Why now:* the paper names this limit and already has a Monte Carlo KL harness to measure it. *First experiment:* sweep equidistant fisheye FoV from 60 to 260 degrees, measure the fraction of Monte Carlo footprint mass outside the UT-derived tile rectangle, then repeat with 13 or 25 points.

4. **Hybrid rendering as a training signal.** *Question:* can a scene be trained with splatted primary rays and traced secondary rays, so reflections in the training images become learnable? *Why now:* the paper shows inference-time hybrid rendering works because the two renderers agree; training through both is the next step. *First experiment:* a synthetic scene with one known mirror, compare reconstruction of the reflected region with and without the traced branch in the loss.

## References

- Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic. *3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting.* CVPR 2025. [arXiv:2412.12507](https://arxiv.org/abs/2412.12507), [CVF open access](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_3DGUT_Enabling_Distorted_Cameras_and_Secondary_Rays_in_Gaussian_Splatting_CVPR_2025_paper.html).
- Code: [github.com/nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut). Project page: [research.nvidia.com/labs/toronto-ai/3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT).
- Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG 2023.
- Moenne-Loccoz et al. *3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes.* ACM TOG / SIGGRAPH Asia 2024.
- On this site: [3D Student Splatting and Scooping](/blog/paper-reading/computer-vision/3d-student-splatting-and-scooping), which changes the particle kernel rather than the projection, and [Difix3D+](/blog/paper-reading/computer-vision/difix3d-improving-3d-reconstructions-with-single-step-diffusion-models), which repairs splat renders with a diffusion model.
