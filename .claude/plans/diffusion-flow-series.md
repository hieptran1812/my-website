# Diffusion Models & Flow Matching: From Theory to Practice
## Series Roadmap

**Goal:** 32-post deep-dive series covering diffusion models and flow matching completely —
theory → math → code → practice. Each post ≥ 9,000 words, ≥ 8 Excalidraw figures.

**Folder:** `content/blog/machine-learning/diffusion-flow-matching/`
**Skill:** blog-writer (deep-dive)
**Kit:** `.cache/blog-writer/_diffusion-flow-series-kit.md`
**Render:** `.cache/blog-writer/_render-diffusion.sh`

---

## Track A — Generative Modeling Foundations (4 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| A1 | `generative-modeling-landscape-from-pixels-to-probability` | Generative Modeling Landscape: How Machines Learn to Create | TODO |
| A2 | `probability-essentials-for-diffusion-models` | Probability Essentials for Diffusion: KL, ELBO, and the Score Function | TODO |
| A3 | `variational-autoencoders-compression-and-generation` | Variational Autoencoders: Learning the Shape of Data | TODO |
| A4 | `from-gans-to-diffusion-why-diffusion-models-won` | From GANs to Diffusion: Why Mode Coverage Beats Mode Seeking | TODO |

## Track B — DDPM: The Complete Story (6 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| B1 | `forward-process-turning-data-into-noise` | The Forward Process: A Mathematical Recipe for Destroying Images | TODO |
| B2 | `reverse-process-learning-to-denoise` | The Reverse Process: Teaching a Network to Reconstruct the World | TODO |
| B3 | `ddpm-training-algorithm-and-loss-function` | DDPM Training: From the ELBO to the Simple MSE Loss | TODO |
| B4 | `noise-schedules-linear-cosine-sigmoid` | Noise Schedules: How You Destroy Data Changes Everything | TODO |
| B5 | `parameterization-tricks-epsilon-x0-v-prediction` | Parameterization Tricks: Predicting Noise, Clean Image, or Velocity | TODO |
| B6 | `ddpm-pytorch-implementation-from-scratch` | DDPM From Scratch in PyTorch: Complete Implementation Guide | TODO |

## Track C — Faster Sampling (4 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| C1 | `ddim-deterministic-fast-sampling-explained` | DDIM: How to Sample 50× Faster Without Retraining | TODO |
| C2 | `dpm-solver-high-order-ode-diffusion-sampling` | DPM-Solver: High-Order ODE Methods for Blazing-Fast Diffusion | TODO |
| C3 | `consistency-models-one-step-generation` | Consistency Models: From 1000 Steps to One | TODO |
| C4 | `sampling-strategies-comparison-and-tradeoffs` | Sampling Strategies: DDPM vs DDIM vs DPM-Solver vs Consistency | TODO |

## Track D — Score Functions & SDE Framework (4 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| D1 | `score-matching-learning-the-gradient-of-data` | Score Matching: Learning the Gradient of the Data Distribution | TODO |
| D2 | `langevin-dynamics-sampling-with-score-functions` | Langevin Dynamics: Walking Toward High-Probability Regions | TODO |
| D3 | `sde-framework-unifying-all-diffusion-models` | The SDE Framework: One Equation to Rule All Diffusion Models | TODO |
| D4 | `probability-flow-ode-deterministic-sampling` | Probability Flow ODE: Exact Likelihoods from Diffusion Models | TODO |

## Track E — Flow Matching: The Complete Story (6 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| E1 | `normalizing-flows-from-bijections-to-continuous` | Normalizing Flows: Change of Variables from Scratch | TODO |
| E2 | `continuous-normalizing-flows-ode-based-generation` | Continuous Normalizing Flows: Neural ODEs for Generation | TODO |
| E3 | `flow-matching-the-key-insight` | Flow Matching: Regressing Vector Fields Instead of Noise | TODO |
| E4 | `conditional-flow-matching-optimal-transport-paths` | Conditional Flow Matching: Straight Paths via Optimal Transport | TODO |
| E5 | `rectified-flow-straight-trajectories` | Rectified Flow: Why Straight Lines Beat Curved Paths | TODO |
| E6 | `flow-matching-pytorch-implementation` | Flow Matching From Scratch in PyTorch: Complete Guide | TODO |

## Track F — Architecture Deep Dives (3 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| F1 | `unet-architecture-for-diffusion-complete-guide` | The Diffusion UNet: Every Block, Skip Connection, and Attention Layer | TODO |
| F2 | `diffusion-transformers-dit-scalable-backbone` | DiT: Replacing UNet with a Transformer (and Why It Scales Better) | TODO |
| F3 | `flux-architecture-flow-matching-at-scale` | FLUX Architecture: Flow Matching Meets Mixture of Experts | TODO |

## Track G — Conditioning & Control (4 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| G1 | `latent-diffusion-working-in-compressed-space` | Latent Diffusion Models: Moving the Battlefield to a Smaller Space | TODO |
| G2 | `classifier-free-guidance-the-practical-standard` | Classifier-Free Guidance: The One Trick That Powers Every Image Generator | TODO |
| G3 | `controlnet-spatial-conditioning-explained` | ControlNet: Adding Depth Maps, Edges, and Poses to Diffusion | TODO |
| G4 | `text-conditioning-clip-cross-attention-t5` | Text Conditioning: From CLIP Embeddings to Cross-Attention | TODO |

## Track H — Finetuning & Personalization (2 posts)

| # | Slug | Title | Status |
|---|------|-------|--------|
| H1 | `lora-for-diffusion-models-complete-guide` | LoRA for Diffusion: Fine-Tuning with 1% of the Parameters | TODO |
| H2 | `textual-inversion-dreambooth-personalization` | Textual Inversion & DreamBooth: Teaching a Model New Concepts | TODO |

## Capstone (1 post)

| # | Slug | Title | Status |
|---|------|-------|--------|
| C | `capstone-building-image-generator-diffusion-to-flow` | Capstone: Building an Image Generator, From DDPM to Flow Matching | TODO |

---

## Wave Schedule

| Wave | Track | Posts | Status |
|------|-------|-------|--------|
| 1 | A | A1–A4 (4 posts) | TODO |
| 2 | B | B1–B3 (3 posts) | TODO |
| 3 | B | B4–B6 (3 posts) | TODO |
| 4 | C | C1–C4 (4 posts) | TODO |
| 5 | D | D1–D4 (4 posts) | TODO |
| 6 | E | E1–E3 (3 posts) | TODO |
| 7 | E | E4–E6 (3 posts) | TODO |
| 8 | F+G | F1–F3, G1–G2 (5 posts) | TODO |
| 9 | G+H | G3–G4, H1–H2 (4 posts) | TODO |
| 10 | Capstone | 1 post | TODO |

---

## Cross-Link Map (existing posts to reference, not duplicate)

- `/blog/machine-learning/image-generation/diffusion-from-first-principles` — broad overview (link from A1)
- `/blog/machine-learning/image-generation/the-math-of-ddpm` — math baseline (link from B1–B3)
- `/blog/machine-learning/image-generation/score-based-models-and-the-sde-view` — SDE overview (link from D3)
- `/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling` — DDIM (link from C1)
- `/blog/machine-learning/image-generation/flow-matching-and-rectified-flow` — flow matching overview (link from E3–E5)
- `/blog/machine-learning/image-generation/diffusion-transformers-dit` — DiT (link from F2)
- `/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion` — LDM (link from G1)
- `/blog/machine-learning/image-generation/classifier-free-guidance` — CFG (link from G2)
- `/blog/machine-learning/image-generation/controlnet-and-structural-control` — ControlNet (link from G3)
- `/blog/machine-learning/deep-learning/fine-tuning-diffusion-models-with-lora` — LoRA (link from H1)
