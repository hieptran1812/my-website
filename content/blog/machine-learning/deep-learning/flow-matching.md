---
title: "Flow Matching: A Simpler Path to Generative Modeling"
publishDate: "2026-04-16"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "flow-matching",
    "generative-models",
    "deep-learning",
    "continuous-normalizing-flows",
    "optimal-transport",
    "rectified-flow",
    "diffusion-models",
    "stable-diffusion-3",
  ]
date: "2026-04-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A clear, in-depth guide to Flow Matching — the simpler alternative to diffusion models that powers Stable Diffusion 3 and Flux. Covers the core idea, math, conditional flow matching, optimal transport, rectified flows, implementation, and common interview questions."
---

## What Is Flow Matching?

![Flow matching: straight-path vector field from Gaussian noise to data distribution, with comparison to diffusion SDE](/imgs/blogs/flow-matching-diagram.png)

Flow matching is a framework for training generative models that is **simpler, faster, and more principled** than traditional diffusion models, while achieving equal or better quality.

The core idea: learn a **velocity field** that transports samples from a simple distribution (Gaussian noise) to the data distribution (real images) along smooth paths.

Think of it like this. Imagine you have a cloud of random dust particles (noise) and you want them to rearrange into a photograph. Flow matching learns the "wind pattern" — a velocity field — that tells each particle which direction to move and how fast, at every moment in time, so that the cloud of particles smoothly transforms into the target image.

```
t = 0 (noise)          t = 0.5 (midway)        t = 1 (data)
  ·  ·  ·                 ·  ·                   ████
·  ·  · ·              ·  ███                   ██  ██
  · ·   ·               ████  ·                 ██  ██
·   ·  ·                 ███                    ██  ██
  ·  · ·                  ██ ·                   ████

    noise          particles moving along        clean image
                    learned velocity field
```

Mathematically, we define a time-dependent map that transforms a noise sample $x_0 \sim \mathcal{N}(0, I)$ into a data sample $x_1 \sim p_\text{data}$ by following an ordinary differential equation (ODE):

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$

Where $v_\theta$ is a neural network that predicts the velocity at any point $x_t$ and time $t$.

## How Is This Different from Diffusion Models?

This is the first question people ask, and the distinction is fundamental:

| Aspect | Diffusion Models | Flow Matching |
|--------|-----------------|---------------|
| **Process** | Add noise gradually (forward SDE), learn to reverse it | Learn a direct transport from noise to data |
| **Training target** | Predict the noise $\epsilon$ that was added | Predict the velocity $v$ that moves samples toward data |
| **Noise schedule** | Must design $\beta_t$ schedule carefully (linear, cosine, etc.) | No noise schedule — paths are defined directly |
| **Paths** | Curved, schedule-dependent paths through noisy states | Straight lines (with optimal transport) |
| **Sampling** | Solve reverse SDE/ODE (needs many steps for curved paths) | Solve forward ODE (fewer steps for straighter paths) |
| **Theory** | ELBO / score matching / SDE reversal | Continuous normalizing flows / optimal transport |
| **Loss** | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | $\|v_\text{target} - v_\theta(x_t, t)\|^2$ |

**The key insight**: Diffusion models define an **indirect** path — you first corrupt data with noise, then learn to reverse the corruption. This reversal path is inherently curved. Flow matching defines a **direct** path — you explicitly construct straight-line transport from noise to data and train the model to follow it. Straighter paths need fewer ODE solver steps to follow accurately.

## The Math: Building Up from First Principles

### Step 1: Continuous Normalizing Flows (CNFs)

A continuous normalizing flow defines a time-dependent transformation of probability distributions via an ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$

Starting from $x_0 \sim p_0$ (a simple prior like a Gaussian), and integrating this ODE from $t=0$ to $t=1$, we get $x_1 \sim p_1$. We want $p_1$ to match the data distribution $p_\text{data}$.

The density at any time $t$ is governed by the **continuity equation** (conservation of probability mass):

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \cdot v_t) = 0$$

This says: as particles move according to velocity $v_t$, probability density is neither created nor destroyed — it just flows.

**The problem with vanilla CNFs**: To train a CNF, the standard approach uses the **instantaneous change of variables formula**:

$$\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \nabla \cdot v_\theta(x_t, t)\, dt$$

Computing the divergence $\nabla \cdot v_\theta$ requires backpropagation through the ODE solver, which is extremely expensive (memory and compute). This made CNFs impractical for high-dimensional data like images.

**Flow matching solves this** by providing a simulation-free training objective — no ODE solver needed during training.

### Step 2: The Flow Matching Objective

Instead of computing log-likelihoods through the ODE, we directly regress the velocity field against a known target:

$$L_\text{FM} = \mathbb{E}_{t \sim \mathcal{U}[0,1]} \mathbb{E}_{x_t \sim p_t} \left[\| v_\theta(x_t, t) - u_t(x_t) \|^2\right]$$

Where $u_t(x_t)$ is the **true velocity field** that generates the probability path $p_t$.

**The problem**: We don't know $p_t$ or $u_t$ in closed form — computing them requires knowing the full coupling between $p_0$ and $p_1$, which is intractable for complex data distributions.

### Step 3: Conditional Flow Matching (The Breakthrough)

This is the key contribution of Lipman et al. (2023). Instead of working with the marginal velocity field $u_t$, we condition on individual data points and work with **conditional** probability paths.

For each data point $x_1$ from the training set, we define a **conditional probability path** $p_t(x | x_1)$ that starts at the prior $p_0$ and ends at a point mass at $x_1$:

$$p_t(x | x_1) = \mathcal{N}(x;\; t \cdot x_1,\; (1 - t)^2 \mathbf{I})$$

This says: at time $t$, the conditional distribution is a Gaussian centered at $t \cdot x_1$ (moving linearly from the origin toward $x_1$) with variance $(1-t)^2$ (shrinking from 1 to 0).

The conditional velocity field for this path is simply:

$$u_t(x_t | x_1) = \frac{x_1 - x_t}{1 - t}$$

**Intuition**: At any time $t$, the velocity points from the current position $x_t$ toward the target $x_1$, scaled by the remaining time $1-t$. This is the velocity needed to arrive at $x_1$ when $t=1$ if you travel in a straight line.

The **Conditional Flow Matching (CFM)** loss is:

$$L_\text{CFM} = \mathbb{E}_{t \sim \mathcal{U}[0,1]} \mathbb{E}_{x_1 \sim p_\text{data}} \mathbb{E}_{x_t \sim p_t(x|x_1)} \left[\| v_\theta(x_t, t) - u_t(x_t | x_1) \|^2\right]$$

**The critical theorem**: Lipman et al. proved that the gradient of $L_\text{CFM}$ equals the gradient of $L_\text{FM}$ — minimizing the conditional objective also minimizes the marginal one. This means we can train using simple per-sample conditional paths and still learn the correct marginal velocity field.

### Step 4: The Simplest Form

With the linear interpolation path, training becomes remarkably simple:

1. Sample a data point $x_1 \sim p_\text{data}$
2. Sample noise $x_0 \sim \mathcal{N}(0, I)$
3. Sample time $t \sim \mathcal{U}[0, 1]$
4. Compute the interpolated point: $x_t = (1-t) \cdot x_0 + t \cdot x_1$
5. The target velocity is: $v_\text{target} = x_1 - x_0$
6. Train: $L = \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$

That's it. No noise schedule. No $\bar{\alpha}_t$ bookkeeping. No forward/reverse process distinction. Just linear interpolation and velocity regression.

```python
def flow_matching_loss(model, x_1, x_0=None):
    """
    Conditional Flow Matching training loss.
    
    Args:
        model: Neural network v_theta(x_t, t) -> velocity prediction
        x_1: Clean data samples from the training set, shape (B, C, H, W)
        x_0: Noise samples (optional, sampled if None)
    
    Returns:
        loss: MSE between predicted and target velocity
    """
    batch_size = x_1.shape[0]
    
    # Sample noise
    if x_0 is None:
        x_0 = torch.randn_like(x_1)
    
    # Sample random time
    t = torch.rand(batch_size, device=x_1.device)
    t_expand = t[:, None, None, None]  # (B, 1, 1, 1) for broadcasting
    
    # Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    
    # Target velocity: simply x_1 - x_0
    v_target = x_1 - x_0
    
    # Predict velocity
    v_pred = model(x_t, t)
    
    # MSE loss
    loss = F.mse_loss(v_pred, v_target)
    
    return loss
```

Compare this to the DDPM training code: no precomputed schedules ($\alpha$, $\bar{\alpha}$, $\beta$), no parameterization choices ($\epsilon$ vs $x_0$ vs $v$), no weighting decisions. The simplicity is not cosmetic — it eliminates entire categories of hyperparameters and design decisions.

## Optimal Transport and Straighter Paths

### Why Straight Paths Matter

The linear interpolation $x_t = (1-t)x_0 + t \cdot x_1$ with a random pairing of $x_0$ and $x_1$ creates paths that are straight **conditionally** (each individual sample travels in a straight line) but the **marginal** velocity field can still be complex because paths from different samples may cross.

When paths cross, the velocity field becomes multi-valued at that point, requiring the network to learn an averaged velocity — which can be suboptimal and require more ODE steps to follow.

```
Random pairing (paths cross):          OT pairing (paths don't cross):

  x_0^a ·─────── × ──────· x_1^b        x_0^a ·──────────────· x_1^a
                  ╱                                               
                 ╱                                                
  x_0^b ·──────× ────────· x_1^a        x_0^b ·──────────────· x_1^b

     Paths cross → complex              Paths parallel → simple
     velocity field                     velocity field
```

### Optimal Transport (OT) Coupling

**Optimal transport** finds the pairing between noise samples and data samples that minimizes total transport cost (total distance traveled). With OT coupling:

- Nearby noise points map to nearby data points
- Paths are less likely to cross
- The marginal velocity field is smoother
- Fewer ODE steps are needed during sampling

For mini-batch OT (practical approximation):

```python
from scipy.optimize import linear_sum_assignment

def mini_batch_ot_pairing(x_0, x_1):
    """
    Find optimal transport pairing within a mini-batch.
    
    Instead of random pairing (x_0[i] <-> x_1[i]), find the permutation
    of x_1 that minimizes total squared distance to x_0.
    
    Args:
        x_0: Noise samples, shape (B, D)
        x_1: Data samples, shape (B, D)
    
    Returns:
        x_1_paired: Optimally paired data samples
    """
    # Compute pairwise cost matrix
    x_0_flat = x_0.view(x_0.shape[0], -1)
    x_1_flat = x_1.view(x_1.shape[0], -1)
    
    # Squared Euclidean distances
    cost = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2)
    
    # Solve assignment problem (Hungarian algorithm)
    row_idx, col_idx = linear_sum_assignment(cost.cpu().numpy())
    
    return x_1[col_idx]


def ot_flow_matching_loss(model, x_1):
    """Flow matching with mini-batch OT pairing."""
    x_0 = torch.randn_like(x_1)
    
    # Find optimal pairing
    x_1_paired = mini_batch_ot_pairing(x_0, x_1)
    
    # Standard flow matching with OT-paired samples
    t = torch.rand(x_1.shape[0], device=x_1.device)
    t_expand = t[:, None, None, None]
    
    x_t = (1 - t_expand) * x_0 + t_expand * x_1_paired
    v_target = x_1_paired - x_0
    v_pred = model(x_t, t)
    
    return F.mse_loss(v_pred, v_target)
```

**Cost of OT**: The Hungarian algorithm has $O(B^3)$ complexity for batch size $B$. For $B = 256$, this is fast enough. For larger batches, approximate OT (Sinkhorn iterations) can be used.

## Rectified Flows

**Rectified flows** (Liu et al., 2023) take the idea of straight paths further with an iterative procedure called **reflow** that progressively straightens the learned flow.

### The Reflow Procedure

1. **Train** a flow matching model $v_\theta$ on random $(x_0, x_1)$ pairs
2. **Generate** new pairs by running the learned ODE: sample $x_0 \sim \mathcal{N}(0,I)$, integrate to get $\hat{x}_1 = \text{ODE}(x_0, v_\theta)$
3. **Retrain** a new model on the pairs $(x_0, \hat{x}_1)$
4. **Repeat** steps 2-3

Each iteration produces straighter paths because the noise-data pairs are already connected by the learned flow. After 2-3 iterations, paths are nearly straight, enabling accurate 1-step generation.

```python
def reflow_iteration(model, dataloader, num_samples=50000):
    """
    One reflow iteration: generate new (x_0, x_1) pairs using the current model,
    then retrain on these pairs.
    """
    # Step 1: Generate paired data using current model
    paired_data = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(num_samples // batch_size):
            x_0 = torch.randn(batch_size, C, H, W, device=device)
            
            # Integrate ODE from t=0 to t=1 using Euler method
            x_t = x_0.clone()
            num_steps = 100
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.full((batch_size,), step * dt, device=device)
                v = model(x_t, t)
                x_t = x_t + v * dt
            
            x_1 = x_t
            paired_data.append((x_0.cpu(), x_1.cpu()))
    
    # Step 2: Train new model on these pairs
    new_model = create_model()  # fresh model
    # ... train with standard flow matching loss on (x_0, x_1) pairs
    
    return new_model
```

### Distillation to 1-Step Generation

After reflowing, the paths are straight enough that a single Euler step gives a good approximation:

$$x_1 \approx x_0 + v_\theta(x_0, 0) \cdot 1 = x_0 + v_\theta(x_0, 0)$$

This means the model can generate images in a **single forward pass** — as fast as a GAN but with the training stability of flow matching.

## Sampling (Inference)

At inference time, we solve the ODE from $t=0$ to $t=1$ using a numerical ODE solver:

### Euler Method (Simplest)

```python
@torch.no_grad()
def euler_sample(model, shape, num_steps=50):
    """
    Generate samples using Euler method ODE integration.
    
    Args:
        model: Trained velocity field v_theta(x_t, t)
        shape: Output shape (B, C, H, W)
        num_steps: Number of Euler steps (more = better quality)
    """
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)  # x_0 ~ N(0, I)
    
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = torch.full((shape[0],), step * dt, device=device)
        v = model(x, t)
        x = x + v * dt  # Euler step
    
    return x  # x_1 ≈ generated data
```

### Midpoint Method (Better Accuracy)

```python
@torch.no_grad()
def midpoint_sample(model, shape, num_steps=25):
    """Midpoint method — 2nd order, much better accuracy per step."""
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = step * dt
        t_batch = torch.full((shape[0],), t, device=device)
        t_mid_batch = torch.full((shape[0],), t + 0.5 * dt, device=device)
        
        # Evaluate velocity at current point
        v1 = model(x, t_batch)
        
        # Take half step
        x_mid = x + 0.5 * dt * v1
        
        # Evaluate velocity at midpoint
        v_mid = model(x_mid, t_mid_batch)
        
        # Take full step using midpoint velocity
        x = x + dt * v_mid
    
    return x
```

### Adaptive Solvers

For production use, adaptive ODE solvers (e.g., `torchdiffeq.odeint` with the Dormand-Prince method) automatically adjust step sizes based on local error estimates:

```python
from torchdiffeq import odeint

class VelocityFieldWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, t, x):
        t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
        return self.model(x, t_batch)

@torch.no_grad()
def adaptive_sample(model, shape, atol=1e-5, rtol=1e-5):
    """Adaptive ODE solver for highest quality."""
    x_0 = torch.randn(shape, device=next(model.parameters()).device)
    wrapper = VelocityFieldWrapper(model)
    
    t_span = torch.tensor([0.0, 1.0], device=x_0.device)
    solution = odeint(wrapper, x_0, t_span, atol=atol, rtol=rtol, method="dopri5")
    
    return solution[-1]  # x at t=1
```

### How Many Steps Do You Need?

| Method | Steps | Quality | Notes |
|--------|-------|---------|-------|
| Euler | 100+ | Good | Simple but needs many steps |
| Euler (with OT) | 20-50 | Good | Straighter paths → fewer steps |
| Midpoint | 15-25 | Very good | 2× better accuracy per step |
| Dopri5 (adaptive) | ~20-40 (auto) | Excellent | Adapts step size to local curvature |
| Euler (after reflow) | 1-5 | Good-Excellent | Near-straight paths enable few-step generation |

## Conditioning: Text-to-Image Flow Matching

Flow matching supports conditioning (text, class labels, etc.) in the same way as diffusion models — the velocity field becomes $v_\theta(x_t, t, c)$ where $c$ is the condition.

### Classifier-Free Guidance with Flow Matching

The same CFG trick applies:

$$\hat{v} = v_\theta(x_t, t, \emptyset) + w \cdot \left(v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset)\right)$$

During training, randomly drop the condition with probability $p_\text{uncond}$ (typically 10%):

```python
def conditional_flow_matching_loss(model, x_1, condition, p_uncond=0.1):
    """Flow matching with classifier-free guidance training."""
    x_0 = torch.randn_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device)
    t_expand = t[:, None, None, None]
    
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    v_target = x_1 - x_0
    
    # Randomly drop condition for CFG training
    mask = torch.rand(x_1.shape[0], device=x_1.device) < p_uncond
    condition_masked = condition.clone()
    condition_masked[mask] = 0  # null condition (or learned null embedding)
    
    v_pred = model(x_t, t, condition_masked)
    
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def cfg_euler_sample(model, shape, condition, num_steps=50, guidance_scale=7.5):
    """Sample with classifier-free guidance."""
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    
    null_condition = torch.zeros_like(condition)
    
    for step in range(num_steps):
        t = torch.full((shape[0],), step * dt, device=device)
        
        # Unconditional and conditional predictions
        v_uncond = model(x, t, null_condition)
        v_cond = model(x, t, condition)
        
        # Guided velocity
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        x = x + v * dt
    
    return x
```

## Real-World Models Using Flow Matching

### Stable Diffusion 3 / 3.5 (Stability AI)

SD3 uses **Rectified Flow** with a modified noise schedule called **logit-normal timestep sampling** — instead of sampling $t$ uniformly, it samples $t \sim \sigma(\mathcal{N}(0, 1))$ (logit-normal distribution), which concentrates more training signal at intermediate timesteps where the learning signal is strongest.

The architecture is a **MM-DiT** (Multimodal Diffusion Transformer) that jointly processes text and image tokens with bidirectional attention.

### Flux (Black Forest Labs)

Flux uses flow matching with a pure **Transformer architecture** (no U-Net). It replaces the U-Net's hierarchical structure with parallel attention between text and image tokens, achieving state-of-the-art quality with faster training.

### Stable Audio / Stable Video

Audio and video generation models from Stability AI also use flow matching as their core framework, demonstrating its generality beyond images.

## Complete Training Pipeline

Here's a minimal but complete flow matching training pipeline:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FlowMatchingTrainer:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
    
    def train(self, dataloader, epochs, lr=1e-4):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (x_1, _) in enumerate(dataloader):
                x_1 = x_1.to(self.device)  # Real data
                x_0 = torch.randn_like(x_1)  # Noise
                
                # Random time
                t = torch.rand(x_1.shape[0], device=self.device)
                t_expand = t[:, None, None, None]
                
                # Interpolate
                x_t = (1 - t_expand) * x_0 + t_expand * x_1
                
                # Target velocity
                v_target = x_1 - x_0
                
                # Predict and compute loss
                v_pred = self.model(x_t, t)
                loss = F.mse_loss(v_pred, v_target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")
    
    @torch.no_grad()
    def sample(self, shape, num_steps=50):
        """Generate samples using Euler method."""
        self.model.eval()
        x = torch.randn(shape, device=self.device)
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.full((shape[0],), step * dt, device=self.device)
            v = self.model(x, t)
            x = x + v * dt
        
        self.model.train()
        return x
```

## Equivalence with Diffusion Models

An important theoretical result: flow matching and diffusion models can be shown to be **different parameterizations of the same underlying process**.

### The Connection

Consider the diffusion model's DDIM ODE:

$$\frac{dx_t}{dt} = f(t) x_t + \frac{g(t)^2}{2\sigma_t} \epsilon_\theta(x_t, t)$$

And the flow matching ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$

These are the same ODE with a reparameterization. Specifically, for the variance-preserving (VP) diffusion process:

$$v_\theta(x_t, t) = f(t) x_t + \frac{g(t)^2}{2\sigma_t} \epsilon_\theta(x_t, t)$$

This means you can convert between the two: a trained diffusion model's $\epsilon$-predictions can be converted to flow matching's $v$-predictions and vice versa.

### So Why Use Flow Matching?

If they're equivalent, why bother with flow matching? Because:

1. **The training objective is simpler** — no schedule-dependent weighting, no choice between $\epsilon$/$x_0$/$v$ parameterizations
2. **Straight paths from OT coupling** — while diffusion's paths are determined by the noise schedule, flow matching's paths can be explicitly straightened via OT, leading to fewer sampling steps
3. **Cleaner theoretical framework** — optimal transport gives principled answers to questions that are ad-hoc in diffusion (how to pair noise and data, how to weight different timesteps)
4. **Flexibility** — easy to define custom interpolation paths, non-Gaussian priors, or structured transport maps

## Common Interview Questions and Answers

### Q: Explain flow matching in simple terms.

Flow matching trains a neural network to predict the velocity that moves a sample from noise to data. Given a noisy intermediate point $x_t$ (which is a linear interpolation between noise and a real image) at time $t$, the model predicts the direction and speed to continue toward the target image. The training signal is simple: the target velocity is just $x_1 - x_0$ (data minus noise), and the model learns to approximate this.

### Q: What is the difference between flow matching and diffusion models?

Both are generative models that iteratively transform noise into data. The key differences are:

1. **Path definition**: Diffusion defines paths implicitly through a noise schedule; flow matching defines paths explicitly via interpolation
2. **Training target**: Diffusion predicts noise $\epsilon$; flow matching predicts velocity $v = x_1 - x_0$
3. **Noise schedule**: Diffusion requires careful schedule design ($\beta_t$); flow matching has no noise schedule
4. **Path geometry**: Flow matching with OT produces straighter paths, enabling fewer sampling steps
5. **Equivalence**: Mathematically, they are reparameterizations of the same process, but flow matching is simpler to implement and reason about

### Q: What is Conditional Flow Matching and why is it needed?

The "marginal" flow matching objective requires knowing the full probability path $p_t$ and its velocity field $u_t$, which is intractable. Conditional Flow Matching (CFM) conditions on individual data points and works with simple conditional paths (e.g., Gaussian distributions that interpolate between noise and a single data point). The key theorem proves that optimizing the conditional objective has the same gradient as optimizing the intractable marginal one, making training practical.

### Q: Why are straight paths better?

Straight paths mean the velocity field is simpler (more uniform across space), so:
1. The neural network needs less capacity to represent it
2. Numerical ODE solvers can take larger steps with less error
3. Fewer total steps are needed for accurate generation
4. 1-step generation becomes possible (since one Euler step perfectly follows a straight line)

Curved paths, like those in diffusion models, require more steps because each Euler step introduces truncation error proportional to the curvature.

### Q: What is Optimal Transport coupling and how does it help?

Standard flow matching pairs noise and data samples randomly within each mini-batch. Optimal transport finds the pairing that minimizes total transport cost (sum of squared distances). This produces less crossing paths and a smoother marginal velocity field. The practical benefit is faster convergence during training and fewer ODE steps needed during sampling (typically 20-30 steps with OT vs. 50-100 without).

### Q: What are Rectified Flows?

Rectified flows iteratively straighten the learned flow through a procedure called "reflow." After training a flow matching model, you generate $(x_0, x_1)$ pairs by running the ODE, then retrain on these pairs. The new pairs are already connected by the learned flow, so the paths become straighter. After 2-3 iterations, paths are nearly straight enough for 1-step generation via distillation.

### Q: How does flow matching handle conditioning (e.g., text-to-image)?

The same way as diffusion models: the velocity field becomes conditioned on the prompt, $v_\theta(x_t, t, c)$. Classifier-free guidance works identically — train with random condition dropping, then extrapolate between conditional and unconditional predictions at inference: $\hat{v} = v_\text{uncond} + w(v_\text{cond} - v_\text{uncond})$.

### Q: Why does Stable Diffusion 3 use flow matching instead of standard diffusion?

1. **Simpler training**: No noise schedule design or parameterization choice
2. **Fewer sampling steps**: OT-coupled rectified flows need fewer steps than DDPM/DDIM
3. **Better scalability**: The simpler loss landscape scales better with model size (SD3 uses a large Transformer backbone)
4. **Logit-normal timestep sampling**: Concentrates training signal at mid-range timesteps where learning is most efficient — an optimization that's more natural in the flow matching framework

### Q: Can you convert a diffusion model to a flow matching model?

Yes. Given a trained diffusion model $\epsilon_\theta(x_t, t)$ with noise schedule $\alpha_t, \sigma_t$, the equivalent velocity field is:

$$v_\theta(x_t, t) = \frac{d\alpha_t}{dt} \frac{x_t}{\alpha_t} + \alpha_t \frac{d(\sigma_t/\alpha_t)}{dt} \epsilon_\theta(x_t, t)$$

This conversion doesn't require retraining — it's a closed-form transformation applied at inference time.

### Q: What are the limitations of flow matching?

1. **Same inference speed as diffusion** (for comparable quality) when not using OT or reflow — the straight-path advantage only materializes with explicit path straightening
2. **OT coupling adds cost**: Mini-batch OT has $O(B^3)$ complexity; approximations (Sinkhorn) are needed for large batches
3. **Reflow is expensive**: Each reflow iteration requires generating a full dataset of ODE trajectories, which is computationally heavy
4. **Less mature ecosystem**: Fewer open-source implementations, tutorials, and pre-trained models compared to diffusion
5. **Theoretical equivalence means practical gains can be modest**: When both methods are fully optimized, the quality gap is small — flow matching's advantage is primarily in simplicity and training convenience

### Q: How does flow matching relate to continuous normalizing flows?

Flow matching is a practical method for training continuous normalizing flows (CNFs). Traditional CNFs required backpropagation through an ODE solver (expensive) or Hutchinson trace estimation (noisy). Flow matching bypasses this entirely by providing a simulation-free training objective that directly regresses a target velocity field. This made CNFs practical for high-dimensional data like images for the first time.

## References

1. Lipman, Y., Chen, R. T. Q., et al. "Flow Matching for Generative Modeling." ICLR 2023.
2. Liu, X., Gong, C., & Liu, Q. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023.
3. Albergo, M. S. & Vanden-Eijnden, E. "Building Normalizing Flows with Stochastic Interpolants." ICLR 2023.
4. Tong, A., et al. "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport." TMLR 2024.
5. Esser, P., et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." (Stable Diffusion 3) ICML 2024.
6. Pooladian, A., et al. "Multisample Flow Matching: Straightening Flows with Minibatch Couplings." ICML 2023.
7. Chen, R. T. Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
8. Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
