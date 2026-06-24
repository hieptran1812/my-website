---
title: "Diffusion Transformers with Representation Autoencoders"
publishDate: "2025-10-16"
category: "paper-reading"
subcategory: "Computer Vision"
tags: ["diffusion-model", "transformer"]
date: "2025-10-16"
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-copy-20251016142823.png"
excerpt: ""
---

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-copy-20251016142823.png)

# Motivation

Modern Diffusion Transformers (DiT) often use a method called latent generative modeling. In this approach, an autoencoder converts images into a smaller latent space before the diffusion process happens. However, most current models still depend on an old type of Variational Autoencoder (VAE). This creates several problems: 

- The architecture is outdated
- The latent space is too small to hold enough information
- The image quality is limited because the VAE is trained only to reconstruct pixels

This research introduces a new idea called Representation Autoencoders (RAEs). Instead of using a traditional VAE, the authors use powerful pretrained encoders such as DINO, SigLIP, or MAE, together with a trained decoder that rebuilds the images. These pretrained encoders produce latent spaces that are both high quality and semantically meaningful.

RAEs make Diffusion Transformers train faster, generate higher quality images, and scale more easily. Although the latent spaces are high dimensional, the researchers explain why this is difficult and present both theoretical and experimental solutions. Their results on ImageNet show excellent performance with an FID score of 1.51 at 256×256 pixels without guidance, and 1.13 at 256×256 and 512×512 pixels with guidance.

# HIGH FIDELITY RECONSTRUCTION FROM FROZEN ENCODERS

The section challenges the common belief that pretrained models such as DINOv2 or SigLIP2 are not suitable for reconstructing images. These models are often thought to focus too much on high-level meaning (semantics) and ignore fine visual details.

However, the authors show that when these pretrained encoders are paired with a properly trained decoder, they can actually work very well as encoders for diffusion models. They call this new design a Representation Autoencoder (RAE).

![](/imgs/blogs/diffusion-transformers-with-representation-autoencoders-20251016145838.png)

RAE works like this:

* The **encoder** $E$ is a pretrained representation model, such as DINOv2, SigLIP2, or MAE. It converts an image into a compact latent representation.
* The **decoder** $D$ is a Vision Transformer (ViT) that learns to reconstruct the original image from this latent space.
* Together, they form the RAE, which gives reconstructions as good as or even better than the traditional **SD-VAE** used in diffusion models.

Unlike VAEs, RAEs do not lose much detail, because VAEs usually compress images too heavily. For example, an SD-VAE maps a $256^2$ image to a $32^2 \times 4$ latent tensor, which limits both reconstruction fidelity and the richness of the learned representation. RAEs address this by using higher-dimensional latent spaces.

Given an input image $x \in \mathbb{R}^{3 \times H \times W}$ and a frozen encoder $E$ with patch size $p_e$ and hidden size $d$, the number of tokens is
$$
N = \frac{H W}{p_e^2}.
$$

A ViT decoder $D$ with patch size $p_d$ maps these tokens back to pixels, producing an output of shape $3 \times \frac{H p_d}{p_e} \times \frac{W p_d}{p_e}$.
By default, $p_d = p_e$, so the reconstructed image matches the input resolution.

For example, with $256 \times 256$ images, the encoder produces 256 tokens, matching the token count used by most prior DiT-based models trained with SD-VAE latents.

The decoder is trained with a combination of reconstruction, perceptual, and adversarial losses:

$$
z = E(x), \quad \hat{x} = D(z)
$$

and the total loss is

$$
\mathcal{L}_{rec}(x) = \omega_L , \text{LPIPS}(\hat{x}, x) + \text{L1}(\hat{x}, x) + \omega_G \lambda , \text{GAN}(\hat{x}, x)
$$

where LPIPS measures perceptual similarity, L1 measures pixel-level error, and GAN encourages realism.

---

### **Encoders Used in Experiments**

The authors evaluate three pretrained encoders from different learning paradigms:

1. **DINOv2-B** ($p_e = 14, d = 768$): a self-supervised, self-distillation model.
2. **SigLIP2-B** ($p_e = 16, d = 768$): a language-supervised model.
3. **MAE-B** ($p_e = 16, d = 768$): a masked autoencoder.

For DINOv2, they also test different model sizes: **S, B, L** with $d = 384, 768, 1024$.
Unless otherwise stated, all RAEs use a **ViT-XL decoder**.

# Experiments on LLMs

## Toxicity Mitigation in LLMs

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004212.png)

The experiments evaluate how well AcT (Activation Transport) mitigates toxic language generation in large language models (LLMs) like Gemma2-2B and Llama3-8B.

- Linear-AcT achieves the best toxicity reduction, lowering toxic outputs by up to 7.5× on Gemma2-2B and 4.3× on Llama3-8B, with minimal impact on perplexity (PPL) and reasoning accuracy (MMLU).

- Compared to other methods (AURA, AcTADD, and ITI-C), Linear-AcT and Mean-AcT provide the most stable and robust performance across layer choices and the intervention strength parameter (λ).

- ITI-C also performs strongly (up to 5.6× reduction) but is highly sensitive to λ and layer type.

- AURA shows moderate improvement (up to 3.1×), while AcTADD provides the weakest mitigation.

In short, Linear-AcT is both effective and robust, showing strong toxicity reduction without hurting model fluency or general performance.

## Concept Induction in LLMs

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004250.png)

This section tests AcT’s ability to induce specific semantic concepts into model generations (e.g., “football”, “cloud”, “baby”).

- Linear-AcT can reliably inject arbitrary concepts at a consistent λ = 1, maintaining low perplexity and strong concept presence in generated text.

- Compared to ITI-C and Mean-AcT, Linear-AcT exhibits smoother and more interpretable control of concept strength, aligning with the theoretical optimal transport (OT) formulation.

- ITI-C peaks at λ ≈ 2.5, but is less stable across tasks and layers.

Thus, Linear-AcT generalizes well for concept control, achieving high concept presence (p(yes) ≈ 0.87) while keeping text quality consistent.

## Inducing Truthfulness in LLMs

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004342.png)

The final experiment applies AcT to improve truthfulness in language generation using the TruthfulQA benchmark.

- Both Linear-AcT and Mean-AcT significantly increase factual accuracy (MC1 and MC2) compared to baselines.

- On Gemma2-2B, Linear-AcT raises MC1 by about +5%, and on Llama3-8B, by nearly +8%.

- The improvement comes with minimal trade-off in general reasoning ability (MMLU decreases by less than 0.5%).

- Again, λ = 1 is found to be a stable and effective default setting, corresponding to full transport in AcT’s formulation.

Overall, AcT—especially Linear-AcT—can induce truthfulness and desirable behaviors in LLMs while preserving general performance and text fluency.

# Controlling image diffusion models

## Fine-Grained Style Control

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004823.png)

A key challenge in T2I models is precisely controlling visual style attributes (e.g., sketchiness, impressionism, watercolor) without distorting image semantics.

- Linear-AcT significantly improves fine-grained control, increasing the presence of the desired style from ~12% to ~95% on SDXL while maintaining ~80% similarity to the original image (measured by CLIP similarity).

- The optimal strength λ = 1 yields the best results for both SDXL and FLUX, consistent with AcT’s theoretical framework and prior LLM experiments.

- ITI-C can also achieve style control but performs inconsistently across models, requiring different λ values (λ = 2 for SDXL, λ = 1 for FLUX) and often exaggerating visual traits or distorting semantics.

## Concept Negation

![](/imgs/blogs/controlling-language-and-diffusion-models-by-transporting-activations-20251009004847.png)

Concept negation refers to preventing the model from generating undesired objects (e.g., instructing it not to draw a “pink elephant”). This is a persistent challenge for diffusion models like SDXL and DALL·E 3.

- The study found that Linear-AcT effectively suppresses unwanted concepts while preserving image semantics, performing better than ITI-C and native “negative prompt” mechanisms in SDXL and FLUX.

- Quantitatively, Linear-AcT achieved higher CLIP similarity scores (indicating better semantic preservation) and lower unintended concept presence than ITI-C.

- In contrast, ITI-C required stronger interventions (higher λ) to suppress undesired concepts, which often caused semantic degradation or over-suppression.

# Some stuff that I interest when reading this paper :))

The paper takes a brilliant and refreshing approach to model control. Instead of using ad-hoc vector shifts or neuron suppression, it frames the whole problem under Optimal Transport (OT). This is a principled mathematical view that connects and generalizes all previous activation-steering methods. I find this idea elegant because it not only unifies earlier techniques but also gives a clear, interpretable way to control model behavior through a single parameter λ that smoothly adjusts intervention strength.

What really stands out is how Linear-AcT performs across both language and image diffusion models. It manages to reduce toxicity, induce truthfulness, and even control fine-grained visual styles or remove unwanted concepts, all with consistent results and minimal side effects. This kind of cross-domain robustness is rare, and it shows that AcT isn’t just a theoretical idea but a genuinely practical tool for generative model control.

That said, I do think the paper relies on some strong assumptions. The linear mapping between activations is a bit too simplistic for such high-dimensional, nonlinear spaces. Also, treating each activation dimension independently ignores important correlations inside neural layers. So while AcT is a powerful and interpretable first step, it still feels like an approximation, maybe a good one, but not yet the full story of how activations could be transported in complex generative systems.

# References

1. [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/pdf/2510.11690)
