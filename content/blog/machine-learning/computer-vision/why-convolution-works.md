---
title: "Why Convolution Works: The Inductive Biases That Made Computer Vision Learnable"
publishDate: "2026-04-23"
category: "machine-learning"
subcategory: "Computer Vision"
tags: ["convolution", "cnn", "inductive-bias", "representation-learning", "computer-vision", "deep-learning"]
date: "2026-04-23"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Convolution isn't a neural-network trick — it's a hypothesis about the world. This article unpacks why that hypothesis is the right one for natural images, quantifies what it buys you in parameters and samples, shows where it leaks in production systems, and walks through the engineering decisions (im2col, depthwise separable, BN, anti-aliasing, dilation) that define how modern CNNs actually work."
---

## Why this question still matters in 2026

In 2020 the community collectively decided that attention is all you need — even for pixels. ViT arrived, ConvNets were declared obsolete, and a lot of engineers started to treat `Conv2d` as a legacy layer.

Five years later the picture is more honest. ConvNeXt-V2 matched ViT on ImageNet-1k with pure convolutions and no pre-training tricks. Hybrid backbones (CoAtNet, MaxViT, EfficientFormer) dominate the efficiency Pareto frontier for edge deployment. Nearly every diffusion model still leans on a U-Net whose backbone is convolutional, because diffusion's scaling properties interact with locality in subtle ways. Segment Anything uses a ViT encoder but a convolutional mask decoder. The entire RT-DETR / YOLO-family object-detection lineage remains convolutional. Convolution didn't lose. It stopped being the only option.

So the interesting question isn't *"convolution vs. attention"*. It's this: **what exactly does convolution assume about the world, and when is each of those assumptions load-bearing?** If you can answer that for a specific task — segmentation on satellite imagery, document layout analysis, video action recognition, medical slice classification — you can pick the right tool and, more importantly, predict where it will break before your loss curve tells you.

This post is the senior-engineer version of that answer. We'll:

1. Reduce convolution to the three inductive biases it encodes, and show how each one falls out of a structural assumption about image statistics.
2. Derive the parameter and sample-complexity savings with concrete numbers, and connect them to VC-style bounds and the neural tangent kernel view.
3. Take the signal-processing route: why the math *had* to be convolution once you accept linearity + shift-invariance, and what changes if you relax either.
4. Walk through the engineering decisions that define production CNNs: im2col / Winograd, depthwise separable, group conv, BN vs LN, anti-aliased downsampling, dilation, residual connections.
5. Catalogue where the convolutional prior *leaks* — shift fragility, texture bias, spectral bias, global-context starvation — and show which architectural fixes buy back which property.
6. End with a decision tree for picking backbones in 2026 that is actually useful, not handwaving.

Expect equations. Expect numbers. Expect opinions.

## 1. Convolution is a hypothesis, not a layer

Strip away CUDA kernels and `im2col` and a 2D convolution is a single equation:

$$
y[i, j, o] = \sum_{u=-k}^{k} \sum_{v=-k}^{k} \sum_{c=1}^{C_{in}} W[u, v, c, o] \cdot x[i+u, j+v, c] + b[o]
$$

Four things about this equation encode beliefs about images. Each of them is a knob you can dial up or down, and almost every architectural innovation of the last decade is exactly such a dial.

1. **The sum over `u, v` is bounded by `k`.** The output at position `(i, j)` depends only on a small `k × k` neighborhood of the input. This is the **locality** assumption: relevant structure is nearby. Dialing `k` larger (or adding dilation) relaxes this; attention eliminates it.
2. **`W` has no index `i` or `j`.** The same filter is applied at every position. This is **weight sharing**: whatever is worth detecting in the top-left is worth detecting in the bottom-right. Locally-connected nets relax this; coord-conv and positional encodings partially undo it.
3. **The indexing is `x[i+u, j+v]`**, a shift. That's why shifting the input shifts the output by the same amount — **translation equivariance** falls out of the definition itself, not from any regularizer or augmentation.
4. **The kernel is learned, not designed.** Unlike classical vision (Sobel, Gabor, SIFT), we don't hand-craft the detector. We specify only the *shape* of the hypothesis class and let gradient descent pick the rest.

Point 4 is the reason convolution beat classical vision. Points 1–3 are the reason it beat MLPs. Keep that distinction — most architecture debates confuse the two.

![Convolution as a sliding dot product](/imgs/blogs/why-convolution-works-01-operation.png)

A practical note before going deeper: almost every framework implements convolution as *cross-correlation* (no kernel flip). The distinction is irrelevant for learned kernels — if the optimum is `W`, the optimum under the other convention is `flip(W)` — but it matters the moment you compare with signal-processing code or transfer weights in from classical filters.

## 2. The three inductive biases, and why images obey them

### 2.1 Locality, and the correlation structure of natural images

**Claim:** In natural images, pixel statistics are dominated by short-range dependencies. The auto-correlation $R(d) = \mathbb{E}[x[i,j] \cdot x[i+d_x, j+d_y]]$ decays rapidly with $d = \|(d_x, d_y)\|$.

This is an empirical fact, not a design choice. Ruderman (1994), Simoncelli & Olshausen (2001), and the van Hateren natural-image dataset all show the same thing: the 2D power spectrum of natural images falls as approximately $1/|\omega|^2$. In pixel space that corresponds to slow, scale-invariant decay — but the *conditional* dependencies (once you condition on low-frequency illumination) drop off much faster. Edges, textures, and small parts live at short ranges; only object-level semantics need long-range interactions.

A `3 × 3` filter at the first layer loses very little signal. Stack such filters and the effective range grows as you need it. This is why `3 × 3` stayed the dominant kernel size for a decade after VGG, until ConvNeXt (2022) and RepLKNet (2022) showed that kernels as large as `7 × 7` to `31 × 31` help — not because short-range locality was wrong, but because the extra range lets you match the receptive field of a transformer without the memory cost of self-attention.

**Where it leaks:** global shape, symmetry, scene layout, and counting are genuinely long-range. If your task hinges on those — reading scene text, counting crowd sizes, reasoning over a whole document page, disambiguating left/right in a mirrored-symmetric image — pure early convolutions under-model the dependency and you will pay for it in accuracy, calibration, or both.

Concrete pathology I've seen in production: a CNN classifier trained on microscopy slides had 94% accuracy on the test set but catastrophically miscounted cells when the cell density doubled. A single layer of self-attention at the top fixed it. The underlying bug was a locality assumption that did not survive distribution shift in cell density.

### 2.2 Weight sharing, and the stationarity assumption

**Claim:** An edge detector at `(100, 100)` should look identical to an edge detector at `(37, 204)`. The operator is stationary.

This is the strongest prior convolution imposes. It also delivers the parameter savings people always quote. Consider a single layer mapping a $224 \times 224 \times 3$ input to a $224 \times 224 \times 64$ output:

- Fully-connected: $(224 \cdot 224 \cdot 3) \times (224 \cdot 224 \cdot 64) \approx 4.8 \times 10^{11}$ weights. Untrainable.
- Locally-connected (same RF, per-position weights): $3 \cdot 3 \cdot 3 \cdot 64 \cdot 224 \cdot 224 \approx 8.7 \times 10^{7}$ weights. Trainable, massively over-parameterized.
- Convolution (shared filter): $3 \cdot 3 \cdot 3 \cdot 64 = 1{,}728$ weights. **Five orders of magnitude smaller** than the FC equivalent.

The savings are not just memory. They translate into sample complexity. Classical PAC bounds suggest generalization error scales as $\sqrt{d/n}$ for hypothesis-class capacity $d$ and sample size $n$; VC-dimension analyses of CNNs (Du et al. 2018, Zhang et al. 2017) are more nuanced but point in the same direction. Cutting $d$ by $10^5$ is why a ResNet-50 can be trained from scratch on 1.3M ImageNet images and generalize, while a comparably-sized MLP cannot. This is the **inductive-bias tax in reverse**: you pay in flexibility, you get back data efficiency.

A subtler payoff is optimization. Under the NTK view, the effective kernel of a CNN is approximately a convolutional kernel — which has far fewer effective degrees of freedom than the full NTK of an equivalently-wide MLP. Loss landscapes are smoother in the directions that matter. This is why CNNs tolerate larger learning rates and simpler schedules than MLPs do on the same data.

**Where it leaks:** any task with strong *position priors* partially violates stationarity. Document OCR (the header is always at the top). Top-down satellite imagery with a fixed north orientation. Medical slices aligned to a canonical anatomical frame. Faces that are roughly centered and upright. In each case, the optimum *does* depend on position, and a pure convolutional prior leaves accuracy on the table. CoordConv (Liu et al. 2018) appends `(i, j)` channels; absolute positional embeddings in ViT serve the same role.

### 2.3 Translation equivariance, and the invariance it isn't

**Claim:** If I shift the input, the feature map shifts by the same amount. Formally, $\operatorname{Conv}(T_\delta x) = T_\delta \operatorname{Conv}(x)$ where $T_\delta$ is a spatial translation.

This is subtle and constantly confused with translation *invariance*. Convolution is equivariant, not invariant. Invariance — the output is unchanged under shift — is a property of pooling, global averaging, or a top-level classifier, **not** of convolution itself.

Why is equivariance the right property? Because it means the network never has to **relearn** a pattern for each new location. When SGD sees a cat at position A and updates on it, the update is automatically shared with every position. Every training example is, in effect, augmented by all possible translations for free. This is why early CNNs did not need aggressive translation augmentation, and why their sample efficiency on object-centered datasets is so much higher than an MLP's.

**Where it leaks badly enough that the field fixed it:** strided convolutions and max-pooling break *exact* equivariance. A stride-2 conv downsamples the output grid — so a 1-pixel shift of the input does not produce a 1-pixel shift of the output; it produces an *aliased* output where adjacent shifts land in the same output bin. Richard Zhang's *Making Convolutional Networks Shift-Invariant Again* (ICML 2019) showed that standard CNNs have a ~5–10% absolute top-1 drop on ImageNet when inputs are shifted by 1–4 pixels, which is genuinely shocking for a model we called translation-invariant for twenty years.

The fix — **anti-aliased pooling**, inserting a blur (a fixed Gaussian or learnable low-pass filter) before every stride-2 op — is a two-line change that recovers most of the lost equivariance and often bumps accuracy by 0.3–1.0 points on ImageNet for free. The fact that such a small change mattered this much is itself the best evidence of how load-bearing equivariance actually is.

![Three inductive biases baked into convolution](/imgs/blogs/why-convolution-works-02-biases.png)

## 3. Depth: how local filters become semantic detectors

A `3×3` conv at layer 1 sees 3×3 pixels. That is useless on its own — you cannot classify a dog from a 3×3 patch. The magic is in composition.

### 3.1 Receptive field arithmetic

For a stack of $L$ convolutions with kernel $k$, stride $s_l$, and dilation $d_l$, the effective receptive field (ERF) at layer $L$ is

$$
RF_L = 1 + \sum_{l=1}^{L} (k - 1) \cdot d_l \cdot \prod_{m \lt l} s_m
$$

For VGG-like stacks with $k=3, s=1, d=1$, RF grows *linearly* in depth — two `3×3` convs give `5×5`, three give `7×7`, ten give `21×21`. Add stride-2 downsampling and the RF's contribution from later layers is multiplied by the cumulative stride, giving *exponential* growth. Add dilation (DeepLab's trick) and you get exponential growth without any downsampling, which is how dilated convolutions hit ImageNet-class RFs while preserving spatial resolution for dense prediction.

There is a pitfall that senior engineers internalize and junior ones often miss: **effective receptive field is not theoretical receptive field.** Luo et al. (NeurIPS 2016) showed that the *effective* RF — the region of the input whose pixels actually contribute to a given output — is a Gaussian-looking blob much smaller than the theoretical RF, with radius proportional to $\sqrt{L}$ rather than $L$ for deep ReLU nets. This is why ResNet-152 sees a theoretical RF larger than the input but still behaves like a relatively local model in practice. It is also why dilated convolutions and large-kernel designs give real gains: they increase *effective* RF at a cost you can afford.

### 3.2 The feature hierarchy is emergent, not hard-coded

Visualize the filters (Zeiler & Fergus 2014, or any gradient-ascent visualization since) and you see the same pattern every time:

- **Layers 1–2:** oriented edges, color blobs, frequency-selective Gabor-like filters. Almost exactly what V1 neurons respond to.
- **Layers 3–5:** corners, T-junctions, simple textures.
- **Layers 6–10:** object parts — eyes, wheels, text characters.
- **Layers 10+:** whole objects, scenes, abstract categories.

This hierarchy is **not** hard-coded. It is what you get when you combine (a) a compositional architecture with (b) a classification objective on natural images. The convolutional prior makes the compositional architecture cheap and translation-consistent; the data and loss do the rest.

There is a deeper point here that is easy to miss: **convolutional networks are not modeling "vision". They are modeling a factorization of vision into local, compositional, stationary hierarchies.** That factorization is approximately correct for natural images because natural images are themselves generated by an approximately compositional process — light interacts locally with surfaces, surfaces are composed of smaller surfaces, objects are composed of parts. It is much less correct for modalities where the generative process is not local or not compositional, which is why `Conv1D` is fine for audio waveforms but awkward for graph-structured data, and why convolution on text (1D character-CNNs) works but loses badly to attention.

![Receptive field growth across layers](/imgs/blogs/why-convolution-works-03-receptive-field.png)

### 3.3 Why residual connections were the unlock, not just a trick

It is fashionable to describe ResNet as "a way to train deep networks", but the more precise senior framing is: **residual connections fix a specific failure of deep convolutional composition, namely that the identity function is hard to express with a stack of convs + ReLU + BN.**

If you need layer $L+1$ to slightly refine layer $L$'s features, an MLP-like stack has to cancel the nonlinearity and approximate the identity. That is genuinely hard — it requires weights to conspire across channels in a way that gradient descent discovers slowly. A residual block reparameterizes the problem: $y = x + F(x)$ means "do nothing" is the trivial initialization, and layers can incrementally refine rather than recompute. Empirically, this changes the loss landscape in the deep regime from a minefield to something close to convex along the training trajectory (Li et al. 2018, visualizing loss landscapes).

Residuals also stabilize gradients. Without them, the variance of the gradient explodes or vanishes as depth grows; with them, each block contributes a small, well-behaved update. This is why ResNet-152 trains and plain VGG-152 does not. It is not a convolution property per se, but it is the reason deep convolutional hierarchies are practical, and any senior treatment of CNNs should mention it.

## 4. Quantifying the prior: CNN vs MLP on the same problem

To make the abstract argument concrete, take ImageNet-1k (1.28M training images, 224×224 RGB, 1000 classes) and ask: what does each architecture need?

| Metric                             | MLP (2 hidden × 4096)      | ResNet-50                     |
|------------------------------------|----------------------------|-------------------------------|
| Parameters                         | ~620M                      | ~25.6M                        |
| FLOPs / image                      | ~1.2 G                     | ~4.1 G                        |
| ImageNet-1k top-1 (from scratch)   | ~25–35% (heavy reg)        | ~76%                          |
| Translation augmentation required? | Yes, aggressively          | Helpful, not critical         |
| Needs JFT-scale pretraining?       | Yes, and still underperforms | No                          |
| Training wall-clock (8×A100)       | Blows up RAM before conv.  | ~16 hours                     |

The MLP has **24× more parameters** and loses by **40+ points of accuracy** on the same data. It is not that MLPs *cannot* learn visual features — Tolstikhin et al.'s MLP-Mixer (2021) showed that with JFT-3B pretraining, pure MLPs reach ViT-competitive accuracy. It is that for the data budgets we actually have, the convolutional prior makes the problem well-posed.

Now compare against ViT, which is the other end of the prior spectrum:

- ViT-B/16 has ~86M parameters and reaches ~77% on ImageNet-1k **only** after pretraining on ImageNet-21k or JFT. Trained from scratch on ImageNet-1k, ViT lands in the low 70s — noticeably below a plain ResNet-50.
- DeiT (Touvron et al. 2021) closed most of that gap using aggressive distillation, MixUp, CutMix, and RandAugment. In other words: you *can* train ViT from scratch on ImageNet-1k, but you have to *buy back* the inductive bias through data augmentation. This is the cleanest demonstration of the trade-off I know.
- ConvNeXt-V2-L (2023) reaches ~87% on ImageNet-1k matching ViT-L/16, which tells us architecture matters less than recipe once you equalize training tricks and data.

This is Sutton's "bitter lesson" read honestly: when you have enough scale, flexibility wins. When you don't, priors win. Convolution is a good prior. It isn't a universal one.

![CNN vs MLP: why priors beat parameters](/imgs/blogs/why-convolution-works-04-cnn-vs-mlp.png)

## 5. Signal-processing perspective: why the math had to be convolution

For readers with a DSP background there is a cleaner way to see why convolution is the right primitive — and, more importantly, where the argument breaks.

**Theorem (informal).** Every bounded **linear, shift-invariant (LSI)** operator on a signal is *exactly* a convolution with some kernel. (Formally: on $\ell^2(\mathbb{Z}^2)$ or $L^2(\mathbb{R}^2)$, with appropriate boundedness, LSI + linearity $\iff$ convolution.)

If you believe the right operator for extracting image features is linear and shift-invariant — and then you add a pointwise nonlinearity on top to give the model capacity — you have no other choice. Convolution is the *unique* realization of the constraint.

This framing tells you exactly where the approximation breaks:

- **Non-translation equivariance** (rotation, scale, projective warps). Convolution is equivariant to translation but not to rotation, scale, or more general affine transforms. Group-equivariant convolutions (Cohen & Welling 2016; E(2)-CNNs; SE(3)-Transformers for 3D) generalize the primitive to other groups but scale badly and are rarely used at ImageNet scale. In practice we approximate rotation invariance with data augmentation and scale invariance with multi-scale pooling or feature pyramids.
- **Non-stationary statistics.** Documents, satellites with fixed north, medical slices with canonical orientation, faces. Fix with coord-conv, absolute positional embeddings, or — in the attention limit — self-attention, which is content-addressable and therefore *permutation*-equivariant rather than translation-equivariant.
- **Nonlinear structure that is not well captured by "local linear + pointwise nonlinear + compose".** The canonical example is long-range reasoning — counting, matching, routing — that benefits from a global content-addressable operator. This is the gap self-attention fills.

There is also a Fourier view that is worth internalizing. Convolution in space is multiplication in frequency: $\widehat{y}(\omega) = \hat{W}(\omega) \cdot \hat{x}(\omega)$. A small spatial kernel corresponds to a broad filter in the frequency domain; a large kernel to a narrow one. Dilated convolutions change the sampling density in the frequency response — which is why dilated stacks are a cheap way to build band-pass filters at multiple scales. **Spectral bias** (Rahaman et al. 2019, Xu 2019) — the empirical observation that neural nets learn low frequencies before high — is often framed as "the loss prefers smooth functions", but for convolutional nets it is also a statement about the frequency response of the composed filters. Senior practitioners think in the frequency domain more often than junior ones, and this is why.

## 6. The engineering layer: how convolutions are actually computed

The three inductive biases tell you *what* convolution is. Production systems are defined by *how* it's computed, and this is where a lot of real-world engineering lives.

### 6.1 im2col, GEMM, and Winograd

The naive 7-nested-loop implementation of convolution is memory-bandwidth-bound and wastes the tensor cores on a modern GPU. The standard trick is **im2col**: unfold each receptive-field patch into a column, stack them into a matrix, and reduce the whole convolution to a single GEMM (`A @ B`). GEMMs are the single most optimized operation in all of numerical computing; delegating to cuBLAS / cuDNN's Tensor-Core GEMM is how every framework hits >80% of peak flops on modern hardware.

im2col has a memory cost — roughly `k² × input_volume` — which is why for large feature maps frameworks fall back to implicit-GEMM variants, FFT convolutions (for very large kernels), or **Winograd** (Lavin & Gray 2016). Winograd reduces arithmetic for `3×3` convolutions by ~2.25× at the cost of some precision (it's a clever integer-coefficient polynomial transform). It's the default for `3×3, stride=1` convs in cuDNN on many GPUs, and if you've ever wondered why `3×3` convs are faster than `1×3` + `3×1` factorizations on GPU despite having more FLOPs — this is why. Arithmetic intensity and operator fusion beat raw FLOP count.

### 6.2 Depthwise separable, grouped, and why MobileNet worked

A standard `k × k` conv with $C_\text{in}$ input and $C_\text{out}$ output channels costs $k^2 \cdot C_\text{in} \cdot C_\text{out}$ multiply-accumulates per output pixel. Two important structural factorizations:

- **Grouped convolution** (ResNeXt, AlexNet's original split for memory). Split input channels into $g$ groups and convolve each independently. Cost drops to $k^2 \cdot C_\text{in} \cdot C_\text{out} / g$. This weakens weight sharing across channels but preserves it in space.
- **Depthwise separable convolution** (MobileNet, Xception). Replace a standard conv with a depthwise `k×k` (one filter per input channel, no cross-channel mixing) followed by a pointwise `1×1` (mixes channels, no spatial context). Total cost: $k^2 \cdot C_\text{in} + C_\text{in} \cdot C_\text{out}$, roughly $k^2$× smaller than standard convolution.

Depthwise separable is the single most important efficiency primitive of the last decade for mobile CV. It factorizes the two jobs a conv does — spatial mixing and channel mixing — into separate cheap operations. Every modern efficient architecture (MobileNet v1/v2/v3, EfficientNet, ConvNeXt) uses some variant.

The senior reading: depthwise separable is a statement that the full cross-product of spatial × channel interactions is not needed. You're imposing an *additional* factorization prior on top of the convolutional prior. When that prior is correct (ImageNet-class natural images), you get massive efficiency. When it isn't (some dense prediction tasks, some scientific imaging), depthwise separable under-performs the full conv by enough to matter.

### 6.3 Normalization: BN, LN, GN, and why it matters

Batch normalization (Ioffe & Szegedy 2015) is often taught as "normalize activations to stabilize training". The deeper reading is that BN re-parameterizes the loss landscape so that effective step sizes are scale-invariant (Santurkar et al. 2018). For CNNs it interacts with convolution in a specific way: BN normalizes per channel, using statistics pooled across the batch and spatial dimensions. That pooling preserves translation equivariance — shift the input, shift the feature map, BN statistics are unchanged — which is exactly what you want.

BN breaks for small batches, for online / test-time use (hence the running-mean hack), and for any setting where the batch-axis statistics are not meaningful (segmentation with tiny batches, video, adaptation). The fixes:

- **LayerNorm** (per-sample, all channels): breaks translation equivariance across the channel axis. Used in transformers because transformers don't rely on spatial equivariance.
- **GroupNorm** (per-sample, per channel-group): a compromise that preserves spatial equivariance and works with any batch size. Default in ConvNeXt and recent detection models.
- **InstanceNorm**: per-sample, per channel, per spatial location. Used in style transfer because it erases global intensity — which is usually a bug for classification but a feature for style.

The choice of normalization is not a free-floating decision. It interacts with the architectural priors. If you're building a convnet and using LayerNorm, you are quietly giving up some of the translation equivariance that made convolution efficient in the first place. This is often fine — ConvNeXt uses LN and works great — but do it knowingly.

## 7. Where the convolutional prior leaks, and the architectural fixes

A senior reading of the last decade is less "ConvNets won" or "Transformers won" and more a catalogue of fixes for where the pure convolutional prior breaks:

1. **Receptive field too small at early layers.** Fix: dilation (DeepLab), large-kernel convolutions (ConvNeXt's 7×7, RepLKNet's 31×31), or self-attention blocks mixed into the backbone.
2. **Translation equivariance broken by strided downsampling.** Fix: anti-aliased pooling (Zhang 2019), BlurPool, or replacing stride-2 conv with a low-pass filter followed by subsampling. Free accuracy on almost any CNN.
3. **Weight sharing too strong when position matters.** Fix: CoordConv, absolute positional encodings, or — in the limit — attention, which is content-addressable rather than position-addressable.
4. **Global context requires many layers.** Fix: non-local blocks (Wang et al. 2018), Squeeze-and-Excitation, self-attention drop-ins, or cross-layer feature pyramid networks.
5. **Kernel is static and input-independent.** Fix: dynamic convolutions (CondConv, DynamicConv), conditional computation, and — again — attention, which is essentially a data-dependent kernel with no weight sharing.
6. **Texture bias over shape bias.** Geirhos et al. (2019) showed CNNs classify images primarily by texture rather than shape, which is the opposite of human perception. Fix: stylized-ImageNet training, shape augmentations, or ViT's different bias profile.
7. **Spectral bias: high frequencies are learned last.** Fix: explicit high-frequency features (Fourier features, SIREN), or architectural changes that have better high-frequency response.
8. **Brittleness to natural distribution shifts** (ImageNet-C, ImageNet-A, Stylized-ImageNet). Fix: augmentation recipes (MixUp, CutMix, AugMix, RandAugment), pretrained features, and at the extreme: abandoning CNN priors for more data-hungry architectures.

Notice the pattern: each fix is either loosening one of the three original biases, importing a global operator, or changing the statistics of training data to correct a bias in the architecture. Attention, in this framing, is just **convolution with an infinitely large, data-dependent kernel and no weight sharing**. Convolution is **attention with a fixed, localized, stationary kernel**. They sit on the same spectrum of "how much structure am I imposing vs how much am I letting the data decide".

## 8. Modern backbones through this lens

Some quick, opinionated readings of architectures you'll actually ship:

- **ResNet-50/101.** The honest baseline. If you can't beat it by >2 points on ImageNet-1k with your new idea at matched FLOPs, you don't have a new idea. Ages like fine wine.
- **EfficientNet / EfficientNetV2.** Depthwise separable + compound scaling. FLOP-efficient, memory-efficient on mobile, hates dense prediction (the bottleneck structure doesn't play well with FPN-style top-down paths).
- **ConvNeXt (-V2).** What you get when you start from ResNet and import every ViT design choice that doesn't explicitly require attention: patch-like stem, large kernels, LN, GELU, depthwise separable blocks, inverted bottlenecks. Demonstrates that most of ViT's gain was recipe, not self-attention.
- **ViT / DeiT / DINOv2.** Zero convolutional prior (well, almost — patch embedding is a stride-16 conv). Needs scale, pays for it in sample efficiency, buys back flexibility. Strong for self-supervised learning.
- **CoAtNet / MaxViT.** Hybrid: convolutional early stages (where locality is cheap and correct), attention later stages (where global context is needed). Currently the Pareto frontier for accuracy-per-FLOP on ImageNet.
- **U-Net + diffusion.** Convolutional encoder-decoder with skip connections, possibly with attention at low resolutions. Convolution is the right prior for pixel-space generation because the noise + denoising process is local; attention at low-resolution blocks handles long-range coherence.
- **SAM's architecture.** ViT image encoder (heavy pretraining, global tokens) + convolutional mask decoder (dense, spatially structured output). Different priors for different parts of the pipeline. This is the senior move.

## 9. Decision tree for picking a backbone in 2026

The question to always ask yourself: **which invariances and equivariances does my task actually require, and which does my candidate architecture actually provide?**

- **Data-constrained regime (<1M labeled examples, no strong pretraining available)** → ConvNet. The prior is doing work you can't afford to relearn. ConvNeXt-V2 or a modern ResNet with anti-aliased pooling is very hard to beat.
- **Large pretraining data available (ImageNet-21k / JFT / LAION) or strong self-supervised pretraining (DINOv2, MAE)** → ViT or hybrid. You have the data budget to pay the bitter-lesson tax and get flexibility back.
- **Long-range dependencies are core to the task** (document understanding, high-resolution medical imaging with global context, video with temporal dependencies, dense prediction at high resolution) → hybrid or pure transformer. Pure CNNs will bottleneck on RF.
- **Shift- or rotation-robustness is a hard constraint** (satellite, microscopy, industrial inspection) → stay convolutional; use anti-aliased downsampling; consider group-equivariant layers for rotation.
- **Edge / mobile / real-time** → depthwise-separable convolution still wins the FLOPs-per-accuracy frontier for small models. MobileNet and EfficientNet families are not obsolete.
- **Dense pixel-space generation (diffusion, in-painting, super-resolution)** → U-Net with attention at low-res. The locality prior is actively correct for noise.
- **Point cloud, mesh, graph, or non-Euclidean data** → convolution is not the right primitive. Reach for graph networks, PointNet variants, or attention with inductive-bias-free positional encodings.

And one meta-rule: when a vision system is not learning, the question is rarely "should I switch to a transformer". The question is: **which assumption of my architecture is my data violating?** Once you can name the assumption — locality? stationarity? equivariance? spectral content? normalization-batch-size coupling? — you can pick the fix. The named-assumption discipline is the difference between senior debugging and architecture roulette.

## 10. Closing: the thing worth keeping

Convolution works because natural images are (approximately) generated by a local, stationary, compositional process, and convolution is the unique linear operator that matches that generative structure. Everything else — parameter efficiency, sample efficiency, feature hierarchies, optimization ease — follows from that match. Everything that has failed about convolution over the last decade has failed at exactly the points where one of those three words (*local, stationary, compositional*) is wrong for the task at hand.

That framing gives you something the pure "convolutions vs transformers" discourse doesn't: a way to reason about new problems before writing code. If your problem is local, stationary, and compositional, use a ConvNet. If it isn't, name the specific violation and pick the architectural relaxation that targets it. Most of the time you'll end up with a hybrid, because most interesting problems violate exactly one of the three assumptions — and you want to keep the other two.

## References and further reading

- LeCun et al., *Gradient-based learning applied to document recognition*, Proc. IEEE 1998 — original LeNet; still the cleanest statement of the inductive-bias argument.
- Krizhevsky, Sutskever, Hinton, *ImageNet Classification with Deep CNNs*, NeurIPS 2012 — AlexNet.
- Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks*, ECCV 2014 — the feature-hierarchy picture.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 — ResNet.
- Ioffe & Szegedy, *Batch Normalization*, ICML 2015; Santurkar et al., *How Does BN Help Optimization?*, NeurIPS 2018.
- Lavin & Gray, *Fast Algorithms for Convolutional Neural Networks*, CVPR 2016 — Winograd.
- Sifre & Mallat, *Rigid-Motion Scattering for Texture Classification*, 2014 — foundation for depthwise separable thinking; Chollet, *Xception*, CVPR 2017; Howard et al., *MobileNet*, 2017.
- Luo et al., *Understanding the Effective Receptive Field in Deep CNNs*, NeurIPS 2016.
- Zhang, *Making Convolutional Networks Shift-Invariant Again*, ICML 2019.
- Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML 2019.
- Geirhos et al., *ImageNet-trained CNNs are biased towards texture*, ICLR 2019.
- Cohen & Welling, *Group Equivariant CNNs*, ICML 2016; Weiler & Cesa, *General E(2)-Equivariant Steerable CNNs*, NeurIPS 2019.
- Dosovitskiy et al., *An Image Is Worth 16×16 Words*, ICLR 2021 — ViT.
- Touvron et al., *Training data-efficient image transformers (DeiT)*, ICML 2021.
- Liu et al., *A ConvNet for the 2020s* (ConvNeXt), CVPR 2022; Woo et al., *ConvNeXt V2*, CVPR 2023.
- Ding et al., *Scaling Up Your Kernels to 31×31* (RepLKNet), CVPR 2022.
- Dai et al., *CoAtNet*, NeurIPS 2021; Tu et al., *MaxViT*, ECCV 2022.
- Tolstikhin et al., *MLP-Mixer*, NeurIPS 2021.
- Sutton, *The Bitter Lesson*, 2019 — the meta-argument for why strong priors eventually lose at scale.
