---
title: "Diffusion Models: A Complete Guide from Theory to Implementation"
publishDate: "2026-04-16"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "diffusion-models",
    "DDPM",
    "DDIM",
    "score-matching",
    "generative-models",
    "deep-learning",
    "stable-diffusion",
    "denoising",
  ]
date: "2026-04-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A comprehensive, interview-ready guide to diffusion models — covering the forward and reverse process, the math behind DDPM/DDIM, score matching, classifier-free guidance, latent diffusion, and practical implementation. Written to build clear intuition from first principles."
---

## What Are Diffusion Models?

![Diffusion forward (noise) and reverse (denoise) chain with U-Net noise predictor](/imgs/blogs/diffusion-models-diagram.png)

Diffusion models are a class of generative models that learn to create data (images, audio, video) by learning to **reverse a gradual noising process**. The core idea is beautifully simple:

1. **Forward process**: Take a clean image and slowly add Gaussian noise to it, step by step, until it becomes pure random noise
2. **Reverse process**: Train a neural network to undo this — to take noisy data and gradually remove the noise, step by step, until a clean image emerges

Think of it like this: if you slowly dissolve an ice sculpture into a puddle of water (forward process), and you record every intermediate frame, you can train a model to learn the reverse — to "un-melt" a puddle back into a sculpture (reverse process). The model doesn't need to create the sculpture from scratch in one shot; it just needs to learn how to make tiny improvements at each step.

```
Forward (destroying data):
Clean Image → Slightly Noisy → More Noisy → ... → Pure Noise
    x_0    →      x_1       →    x_2     → ... →    x_T

Reverse (creating data):
Pure Noise → Less Noisy → Even Less Noisy → ... → Clean Image
    x_T    →   x_{T-1}  →    x_{T-2}     → ... →    x_0
```

This is fundamentally different from other generative models:
- **GANs** generate in a single forward pass (but training is unstable)
- **VAEs** compress to a latent space and decode (but produce blurry outputs)
- **Diffusion models** generate iteratively over many steps (slower, but produce high-quality, diverse outputs with stable training)

## The Forward Process (Adding Noise)

The forward process is fixed (not learned) — it simply adds Gaussian noise according to a predefined schedule.

### One Step at a Time

At each timestep $t$, we add a small amount of Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t \mathbf{I})$$

Where:
- $x_{t-1}$ is the image at the previous step
- $x_t$ is the image after adding noise
- $\beta_t$ is the **noise schedule** — a small number (e.g., 0.0001 to 0.02) that controls how much noise is added at step $t$
- $\mathcal{N}$ denotes a Gaussian distribution

In plain English: to get $x_t$, we **shrink** $x_{t-1}$ slightly (multiply by $\sqrt{1-\beta_t}$, which is close to 1) and **add** a bit of noise (with variance $\beta_t$).

### The Key Trick: Jumping Directly to Any Timestep

A critical property of the forward process is that we don't need to iterate through all $t$ steps. We can jump directly from the clean image $x_0$ to any noisy version $x_t$ in a single computation.

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (the cumulative product). Then:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, \mathbf{I})$$

Which means we can sample $x_t$ directly:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

**Why this matters**: During training, we don't simulate the entire forward chain. We just sample a random timestep $t$, compute $x_t$ from $x_0$ using the formula above, and train the model on that pair. This is what makes training efficient.

**Intuition for the formula**: $\sqrt{\bar{\alpha}_t}$ controls how much of the original signal remains, and $\sqrt{1 - \bar{\alpha}_t}$ controls how much noise is present. At $t=0$, $\bar{\alpha}_0 \approx 1$, so we have mostly signal. At $t=T$, $\bar{\alpha}_T \approx 0$, so we have mostly noise.

```python
import torch

def forward_diffusion(x_0, t, alpha_bar, noise=None):
    """
    Add noise to clean data x_0 to get x_t.
    
    Args:
        x_0: Clean data, shape (B, C, H, W)
        t: Timestep indices, shape (B,)
        alpha_bar: Cumulative product of (1 - beta), shape (T,)
        noise: Optional pre-sampled noise
    
    Returns:
        x_t: Noisy data at timestep t
        noise: The noise that was added (needed for training)
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # Gather alpha_bar values for each sample's timestep
    alpha_bar_t = alpha_bar[t][:, None, None, None]  # (B, 1, 1, 1)
    
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    return x_t, noise
```

### The Noise Schedule

The noise schedule $\{\beta_t\}_{t=1}^T$ controls the rate of noise addition. The choice of schedule significantly affects generation quality.

**Linear schedule** (original DDPM):

$$\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}(\beta_{\text{end}} - \beta_{\text{start}})$$

Typical values: $\beta_{\text{start}} = 10^{-4}$, $\beta_{\text{end}} = 0.02$, $T = 1000$.

**Cosine schedule** (improved DDPM):

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

Where $s = 0.008$ is a small offset to prevent $\beta_t$ from being too small near $t = 0$.

**Why cosine is better**: The linear schedule destroys too much information too quickly in the early steps, making it harder for the model to learn. The cosine schedule provides a more gradual degradation, preserving more signal in the middle timesteps.

```python
def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32) / T
    alpha_bar = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.999)
```

## The Reverse Process (Removing Noise)

The reverse process is where the magic happens — and where we need to train a neural network.

### What We Want to Learn

We want to learn the reverse distribution $p_\theta(x_{t-1} | x_t)$ — given a noisy image at step $t$, what does the slightly-less-noisy image at step $t-1$ look like?

It turns out that if the forward steps are small enough (small $\beta_t$), the reverse step is also approximately Gaussian:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2 \mathbf{I})$$

So the neural network needs to predict $\mu_\theta(x_t, t)$ — the mean of the reverse step distribution.

### Three Ways to Parameterize the Network

There are three equivalent ways to think about what the network predicts. This is an important concept that comes up frequently in interviews:

#### 1. Predict the noise ($\epsilon$-prediction)

The network $\epsilon_\theta(x_t, t)$ predicts the noise that was added. This is the **DDPM parameterization** and the most common choice:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

**Intuition**: "Here is a noisy image. Tell me what the noise component looks like, and I'll subtract it."

#### 2. Predict the clean image ($x_0$-prediction)

The network $\hat{x}_\theta(x_t, t)$ directly predicts the clean image $x_0$:

$$\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \hat{x}_\theta(x_t, t) + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

**Intuition**: "Here is a noisy image. Tell me what the clean version looks like."

#### 3. Predict the score ($\nabla_x \log p(x)$ — score prediction)

The network predicts the **score function** — the gradient of the log probability density. This connects diffusion models to score-based generative modeling:

$$\nabla_{x_t} \log q(x_t) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

So predicting the noise $\epsilon$ is equivalent to predicting the score (up to a scaling factor).

**Intuition**: "Here is a noisy image. Point me in the direction where the data becomes more likely."

All three parameterizations are mathematically equivalent — they can be converted into each other. In practice, $\epsilon$-prediction works best for most timesteps, while $x_0$-prediction is better for small $t$ (near-clean images) and $v$-prediction (a mixture of the two, used in later works) offers a good compromise.

### The Training Objective

The simplified DDPM training loss is surprisingly elegant:

$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]$$

In words: sample a clean image $x_0$, a random timestep $t$, and random noise $\epsilon$. Create the noisy image $x_t$. Ask the network to predict $\epsilon$ from $x_t$ and $t$. Penalize the squared error.

That's it. This simple MSE loss is all you need.

```python
def training_step(model, x_0, alpha_bar, T):
    """One training step for a DDPM model."""
    batch_size = x_0.shape[0]
    
    # 1. Sample random timesteps
    t = torch.randint(0, T, (batch_size,), device=x_0.device)
    
    # 2. Sample noise
    noise = torch.randn_like(x_0)
    
    # 3. Create noisy images
    x_t, _ = forward_diffusion(x_0, t, alpha_bar, noise=noise)
    
    # 4. Predict the noise
    noise_pred = model(x_t, t)
    
    # 5. Simple MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    
    return loss
```

### Derivation: Where Does This Loss Come From?

The true objective comes from maximizing the **evidence lower bound (ELBO)**, similar to VAEs. The full derivation gives:

$$L = \mathbb{E}\left[\sum_{t=1}^{T} D_{KL}\left(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)\right)\right]$$

The key insight is that $q(x_{t-1}|x_t, x_0)$ — the true reverse step when we know both $x_t$ and $x_0$ — is a tractable Gaussian:

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1};\; \tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t \mathbf{I})$$

Where:

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

The KL divergence between two Gaussians with the same variance reduces to a squared difference in means, which further simplifies to the $\| \epsilon - \epsilon_\theta \|^2$ loss when you substitute $x_0 = (x_t - \sqrt{1-\bar{\alpha}_t}\epsilon) / \sqrt{\bar{\alpha}_t}$.

Ho et al. (2020) showed that dropping the time-dependent weighting factor and using the simplified loss $\| \epsilon - \epsilon_\theta \|^2$ actually produces better samples in practice. This is the "simple" loss that everyone uses.

## The Neural Network Architecture: U-Net

The standard architecture for $\epsilon_\theta(x_t, t)$ is a **U-Net** — an encoder-decoder network with skip connections, augmented with **attention layers** and **timestep conditioning**.

### Why U-Net?

The model needs to:
1. Process the full-resolution noisy image
2. Understand global context (what kind of image is this?)
3. Preserve local details (edges, textures)
4. Know which timestep it's operating at (denoising strategy differs for very noisy vs. slightly noisy images)

The U-Net's encoder-decoder structure with skip connections handles (1-3), and timestep embeddings handle (4).

### Architecture Overview

```
Input: x_t (noisy image) + t (timestep)
                │
        ┌───────▼───────┐
        │  Timestep MLP  │ ──── sinusoidal embedding → MLP → t_emb
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ Conv_in (3→C)  │
        └───────┬───────┘
                │
     ┌──────── │ ────────┐
     │  ENCODER (downsampling)  │
     │  ┌─────────────┐  │
     │  │ ResBlock + t │──┤ skip
     │  │ Attention    │  │ connections
     │  │ Downsample   │  │
     │  └─────────────┘  │
     │  (repeat × N)     │
     └────────┬──────────┘
              │
     ┌────────▼──────────┐
     │   MIDDLE BLOCK     │
     │  ResBlock + Attn   │
     └────────┬──────────┘
              │
     ┌────────▼──────────┐
     │  DECODER (upsampling)   │
     │  ┌──────────────┐ │
     │  │ ResBlock + t  │ │ ← skip connections
     │  │ Attention     │ │
     │  │ Upsample      │ │
     │  └──────────────┘ │
     │  (repeat × N)     │
     └────────┬──────────┘
              │
        ┌─────▼─────┐
        │ Conv_out   │
        │ (C→3)      │
        └─────┬─────┘
              │
        Output: predicted noise ε_θ
```

### Key Components

**Timestep embedding**: The integer timestep $t$ is converted to a continuous vector using sinusoidal positional encoding (same idea as in Transformers), then projected through an MLP. This embedding is added to or modulated into each ResBlock.

```python
import math

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (B, dim)

class TimestepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embed = SinusoidalTimestepEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )
    
    def forward(self, t):
        return self.mlp(self.embed(t))
```

**ResBlock with timestep conditioning**: Each residual block receives the timestep embedding, which modulates the feature maps:

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]  # add timestep info
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
```

**Self-attention layers**: Added at lower resolutions (e.g., 16×16, 8×8) to capture global dependencies. At full resolution, attention is too expensive, so it's only used in the bottleneck layers.

## Sampling: Generating New Images

### DDPM Sampling (Stochastic)

The standard DDPM sampling algorithm iterates through all $T$ timesteps in reverse:

```python
@torch.no_grad()
def ddpm_sample(model, shape, T, alpha, alpha_bar, beta):
    """Generate samples using DDPM (stochastic) sampling."""
    device = next(model.parameters()).device
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        eps_pred = model(x, t_batch)
        
        # Compute mean of p(x_{t-1} | x_t)
        mu = (1 / torch.sqrt(alpha[t])) * (
            x - (beta[t] / torch.sqrt(1 - alpha_bar[t])) * eps_pred
        )
        
        if t > 0:
            # Add stochastic noise (not at the final step)
            sigma = torch.sqrt(beta[t])
            z = torch.randn_like(x)
            x = mu + sigma * z
        else:
            x = mu
    
    return x
```

**Problem**: This requires $T = 1000$ sequential forward passes through the network, which is very slow (30-60 seconds per image on a GPU).

### DDIM Sampling (Deterministic and Faster)

Song et al. (2020) showed that you can use a **non-Markovian** reverse process that skips timesteps, using only a subset of $S \ll T$ steps:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } x_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\; \epsilon_\theta(x_t, t) + \sigma_t\, z$$

When $\sigma_t = 0$ (the DDIM case), sampling becomes **deterministic** — the same initial noise always produces the same image. This also means:

1. **You can skip steps**: Use $S = 50$ or even $S = 20$ steps instead of 1000
2. **Interpolation in latent space**: Since the mapping is deterministic, you can smoothly interpolate between two noise vectors to get smooth transitions between generated images
3. **Inversion**: You can run DDIM forward to find the noise vector that corresponds to a real image, enabling editing

```python
@torch.no_grad()
def ddim_sample(model, shape, timesteps, alpha_bar, eta=0.0):
    """
    DDIM sampling with configurable stochasticity.
    
    Args:
        timesteps: Subset of timesteps to use, e.g., [999, 949, 899, ..., 49, 0]
        eta: 0.0 = fully deterministic (DDIM), 1.0 = fully stochastic (DDPM)
    """
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        eps_pred = model(x, t_batch)
        
        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar[t]) * eps_pred) / torch.sqrt(alpha_bar[t])
        x0_pred = x0_pred.clamp(-1, 1)  # optional clipping for stability
        
        # Compute sigma (controls stochasticity)
        sigma = eta * torch.sqrt(
            (1 - alpha_bar[t_prev]) / (1 - alpha_bar[t]) * (1 - alpha_bar[t] / alpha_bar[t_prev])
        )
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar[t_prev] - sigma**2) * eps_pred
        
        # Compute x_{t-1}
        x = torch.sqrt(alpha_bar[t_prev]) * x0_pred + dir_xt
        
        if t_prev > 0 and sigma > 0:
            x = x + sigma * torch.randn_like(x)
    
    return x
```

### Other Fast Samplers

| Sampler | Steps Needed | Quality | Key Idea |
|---------|-------------|---------|----------|
| DDPM | 1000 | Excellent | Original stochastic sampler |
| DDIM | 20-50 | Very good | Deterministic, skip steps |
| DPM-Solver | 10-20 | Excellent | High-order ODE solver |
| DPM-Solver++ | 10-20 | Excellent | Improved DPM-Solver for guided diffusion |
| UniPC | 5-10 | Very good | Unified predictor-corrector framework |
| Consistency Models | 1-2 | Good | Distilled single-step generation |

## Classifier-Free Guidance (CFG)

This is one of the most important techniques in modern diffusion models and a very common interview topic.

### The Problem

A conditional diffusion model $\epsilon_\theta(x_t, t, c)$ (where $c$ is a text prompt or class label) generates images that match the condition, but the results are often **generic** — they satisfy the condition but lack the vivid, detailed quality we want.

### The Solution

Train the model to work both **with** and **without** the condition by randomly dropping the condition during training (replacing $c$ with a null token $\emptyset$ with some probability, typically 10-20%). At inference, compute both the conditional and unconditional predictions and extrapolate away from the unconditional one:

$$\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right)$$

Where $w$ is the **guidance scale** (typically 3-15).

**Intuition**: The difference $\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)$ represents "the direction that makes the image more consistent with the condition $c$." By multiplying this direction by $w > 1$, we amplify the condition's influence, producing more vivid and prompt-faithful images.

```python
@torch.no_grad()
def guided_sample_step(model, x_t, t, condition, guidance_scale=7.5):
    """One DDIM step with classifier-free guidance."""
    # Unconditional prediction (null condition)
    eps_uncond = model(x_t, t, condition=None)
    
    # Conditional prediction
    eps_cond = model(x_t, t, condition=condition)
    
    # Guided prediction: extrapolate away from unconditional
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    
    return eps_guided
```

**Trade-offs of guidance scale**:
- $w = 1.0$: No guidance. Diverse but may not follow the prompt well
- $w = 3.0-7.5$: Sweet spot for most applications. Good balance of quality and diversity
- $w > 15$: Over-saturated, artifact-prone images. Colors become extreme, details become exaggerated

**Why two forward passes?** In practice, you can batch the conditional and unconditional inputs together, so it's only ~1.5x the cost of a single forward pass (not 2x), since most of the compute is in shared attention layers.

## Latent Diffusion Models (Stable Diffusion)

Running diffusion directly on pixel space (e.g., 512×512×3 images) is extremely expensive. **Latent Diffusion Models (LDM)** solve this by running the diffusion process in a compressed latent space.

### Architecture

```
Text Prompt ─── Text Encoder (CLIP) ──── cross-attention ────┐
                                                              │
Image (512×512×3) ─── VAE Encoder ─── z (64×64×4) ─── U-Net (diffusion in latent space) ─── z_0 ─── VAE Decoder ─── Image (512×512×3)
```

1. **VAE**: A pre-trained variational autoencoder compresses images from pixel space (512×512×3) to latent space (64×64×4). This is an 8× spatial downsampling and reduces the data dimension by 48×
2. **U-Net**: The diffusion model operates entirely in this compressed latent space, making training and inference much faster
3. **Text conditioning**: Text prompts are encoded by a frozen text encoder (CLIP or T5) and injected into the U-Net via cross-attention at each resolution level

```python
class LatentDiffusion(nn.Module):
    def __init__(self, vae, unet, text_encoder):
        super().__init__()
        self.vae = vae            # Pre-trained, frozen
        self.unet = unet          # Trainable
        self.text_encoder = text_encoder  # Pre-trained, frozen
    
    def encode(self, x):
        """Compress pixel-space image to latent."""
        return self.vae.encode(x).latent_dist.sample() * 0.18215
    
    def decode(self, z):
        """Decompress latent to pixel space."""
        return self.vae.decode(z / 0.18215).sample
    
    def training_step(self, images, text_prompts):
        # 1. Encode images to latent space
        with torch.no_grad():
            z_0 = self.encode(images)
        
        # 2. Encode text prompts
        with torch.no_grad():
            text_emb = self.text_encoder(text_prompts)
        
        # 3. Standard diffusion training in latent space
        t = torch.randint(0, self.T, (z_0.shape[0],), device=z_0.device)
        noise = torch.randn_like(z_0)
        z_t = self.forward_diffusion(z_0, t, noise)
        
        noise_pred = self.unet(z_t, t, text_emb)
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
```

### Why Latent Space Works

| Aspect | Pixel Space | Latent Space |
|--------|------------|--------------|
| Spatial size | 512 × 512 | 64 × 64 |
| Channels | 3 (RGB) | 4 (learned) |
| Total dimensions | 786,432 | 16,384 |
| Compression ratio | 1× | 48× |
| Training cost | Very high | Manageable |
| Perceptual quality | Direct | Near-lossless via VAE |

The VAE preserves perceptually important information while discarding high-frequency imperceptible details, making the diffusion model's job much easier.

## Score Matching Connection

Diffusion models are deeply connected to **score-based generative models**. Understanding this connection clarifies the theoretical foundations.

### What Is the Score?

The **score function** of a probability distribution $p(x)$ is the gradient of its log density:

$$s(x) = \nabla_x \log p(x)$$

The score points in the direction where the data becomes more probable. If you follow the score, you move toward high-density regions of the data distribution.

### Score Matching

We can't compute $\nabla_x \log p(x)$ directly (we don't know $p(x)$), but we can train a neural network $s_\theta(x)$ to approximate it using **denoising score matching**:

$$L = \mathbb{E}_{x_0 \sim p_\text{data}} \mathbb{E}_{\sigma} \mathbb{E}_{x \sim \mathcal{N}(x_0, \sigma^2 I)} \left[\left\| s_\theta(x, \sigma) + \frac{x - x_0}{\sigma^2} \right\|^2\right]$$

The target score $-\frac{x - x_0}{\sigma^2}$ points from the noisy sample back toward the clean data. This is exactly what DDPM does — predicting $\epsilon$ is the same as predicting the score (up to a scaling factor):

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

### Langevin Dynamics Sampling

Once you have the score function, you can generate samples using **Langevin dynamics** — an iterative procedure that follows the score with added noise:

$$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta}\, z, \quad z \sim \mathcal{N}(0, I)$$

This is the theoretical justification for why iterative denoising works: each reverse diffusion step is essentially one step of annealed Langevin dynamics.

## Complete Training Pipeline

Putting it all together, here's a complete minimal DDPM training pipeline:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DDPM:
    def __init__(self, model, T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.model = model.to(device)
        self.T = T
        self.device = device
        
        # Precompute schedule
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def train(self, dataloader, epochs, lr=2e-4):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (x_0, _) in enumerate(dataloader):
                x_0 = x_0.to(self.device)
                
                # Sample timesteps uniformly
                t = torch.randint(0, self.T, (x_0.shape[0],), device=self.device)
                
                # Sample noise and create noisy input
                noise = torch.randn_like(x_0)
                alpha_bar_t = self.alpha_bar[t][:, None, None, None]
                x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
                
                # Predict noise and compute loss
                noise_pred = self.model(x_t, t)
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
    
    @torch.no_grad()
    def sample(self, shape, num_steps=None):
        """Generate samples using DDIM-style sampling."""
        if num_steps is None:
            num_steps = self.T
        
        # Create timestep subsequence
        step_size = self.T // num_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        
        x = torch.randn(shape, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = self.model(x, t_batch)
            
            # Predict x_0
            alpha_bar_t = self.alpha_bar[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bar[t_prev]
                # DDIM update (deterministic)
                x = (torch.sqrt(alpha_bar_prev) * x0_pred +
                     torch.sqrt(1 - alpha_bar_prev) * eps_pred)
            else:
                x = x0_pred
        
        return x
```

## Common Interview Questions and Answers

### Q: Why do diffusion models produce higher quality images than GANs?

Diffusion models optimize a stable MSE loss (no adversarial training), have full support of the data distribution (no mode collapse), and can trade compute time for sample quality by using more denoising steps. GANs can produce sharp images but often miss modes of the distribution and are notoriously difficult to train.

### Q: Why does the model predict noise instead of the clean image directly?

Both parameterizations are mathematically equivalent, but noise prediction works better empirically. The intuition is that noise is "simpler" to predict — it's always approximately standard Gaussian — whereas predicting the clean image requires the network to output a sharp, globally coherent image from a very noisy input. The noise prediction also provides a more uniform gradient signal across timesteps.

### Q: How is the training loss derived? What's the connection to the ELBO?

The training loss is a simplified version of the variational lower bound (ELBO). The full ELBO decomposes into a sum of KL divergences at each timestep. Because both the forward posterior $q(x_{t-1}|x_t,x_0)$ and the reverse model $p_\theta(x_{t-1}|x_t)$ are Gaussian, each KL term reduces to a weighted MSE between their means. The simplified loss drops the time-dependent weighting, which empirically improves sample quality.

### Q: What is the difference between DDPM and DDIM?

DDPM uses a stochastic reverse process (adds fresh noise at each step) and requires all $T$ steps. DDIM uses a deterministic reverse process (no added noise when $\eta=0$), can skip timesteps (e.g., use 50 steps instead of 1000), and enables exact inversion (mapping images back to their noise vectors). DDIM is derived from the same trained model — you don't need to retrain.

### Q: What is classifier-free guidance and why does it work?

Classifier-free guidance trains a single model to handle both conditional and unconditional generation (by randomly dropping the condition during training). At inference, it extrapolates away from the unconditional prediction: $\hat{\epsilon} = \epsilon_\text{uncond} + w(\epsilon_\text{cond} - \epsilon_\text{uncond})$. Geometrically, this amplifies the "direction" that makes the output more aligned with the condition. It works because it increases the effective signal-to-noise ratio of the conditioning signal without requiring a separate classifier.

### Q: Why use latent space instead of pixel space?

Pixel-space diffusion is computationally prohibitive for high-resolution images (512²×3 = 786K dimensions). A pre-trained VAE compresses images to a much smaller latent space (64²×4 = 16K dimensions, a 48× reduction) while preserving perceptual quality. The diffusion model then operates in this compressed space, making training and inference dramatically cheaper.

### Q: How do diffusion models relate to score-based models?

They are the same thing viewed from different angles. Predicting the noise $\epsilon$ in DDPM is equivalent to predicting the score $\nabla_x \log p(x_t)$ up to a scaling factor: $s(x_t) = -\epsilon / \sqrt{1-\bar{\alpha}_t}$. The DDPM reverse process is equivalent to discretized Langevin dynamics with annealed noise levels. Song et al. (2021) unified both perspectives under the framework of stochastic differential equations (SDEs).

### Q: What are the main limitations of diffusion models?

1. **Slow sampling**: Requires many sequential denoising steps (mitigated by DDIM, DPM-Solver, consistency distillation)
2. **High training cost**: Large U-Net models, many training iterations, large batch sizes
3. **Memory intensive**: The U-Net must store activations for all resolution levels during training
4. **Evaluation metrics**: FID and IS don't fully capture perceptual quality; human evaluation is expensive
5. **Controllability**: Fine-grained spatial control requires additional techniques (ControlNet, IP-Adapter)

### Q: Explain the noise schedule. Why does it matter?

The noise schedule $\{\beta_t\}$ controls how quickly the forward process destroys information. If noise is added too quickly (large $\beta_t$ early on), the model must learn large denoising jumps — a harder problem. If too slowly, training is inefficient because many timesteps contain nearly identical difficulty levels. The cosine schedule produces a more uniform information destruction rate across timesteps compared to the linear schedule, leading to better results because the model faces a balanced learning problem at each timestep.

### Q: What is the relationship between the number of sampling steps and quality?

More steps = higher quality but slower generation. With DDPM (stochastic), you need all 1000 steps for best quality. With DDIM, quality degrades gracefully: 50 steps ≈ 95% of 1000-step quality, 20 steps ≈ 85%. Advanced solvers (DPM-Solver++) can achieve 1000-step quality in 10-20 steps by using higher-order ODE integration. Consistency models aim for 1-2 step generation via distillation.

## References

1. Ho, J., Jain, A., & Abbeel, P. "Denoising Diffusion Probabilistic Models (DDPM)." NeurIPS 2020.
2. Song, J., Meng, C., & Ermon, S. "Denoising Diffusion Implicit Models (DDIM)." ICLR 2021.
3. Nichol, A. & Dhariwal, P. "Improved Denoising Diffusion Probabilistic Models." ICML 2021.
4. Dhariwal, P. & Nichol, A. "Diffusion Models Beat GANs on Image Synthesis." NeurIPS 2021.
5. Ho, J. & Salimans, T. "Classifier-Free Diffusion Guidance." NeurIPS Workshop 2022.
6. Rombach, R. et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
7. Song, Y. et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
8. Lu, C. et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling." NeurIPS 2022.
9. Song, Y. et al. "Consistency Models." ICML 2023.
